#!/usr/bin/env python
"""Largest TEI --max-batch-tokens for Qwen3-Embedding-4B that stays under 21 GiB of GPU memory.

Input: the longest real documents of the canonical Phase 1 table. Each tested value runs in a
fresh TEI container under the vram_cap allocation cap and an NVML watchdog.
Outputs: results tables, environment record and REPORT.md.

Stages (run in the WSL2 venv; see ../README.md):
  prepare   verify inputs, pick candidate documents, build the cap shim, run a calibration container
  search    below-minimum check, bracket, bisection, confirmation rerun
  report    tables, environment record and REPORT.md
  all       prepare, search, report
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------------

SCRIPT_PATH = Path(__file__).resolve()
FEAS_DIR = SCRIPT_PATH.parents[1]  # feasibility_tei_qwen3_4b_batch_tokens/
PHASE2_DIR = SCRIPT_PATH.parents[2]  # phase2_semantic_embeddings/
DEFAULT_PROJECT_DIR = SCRIPT_PATH.parents[4]  # the repository root 
PRIOR_FEAS_DIR = PHASE2_DIR / "feasibility_qwen3_embedding_worst_case"
SHIM_SRC_DIR = FEAS_DIR / "scripts" / "vram_cap"

MODEL_REPO = "Qwen/Qwen3-Embedding-4B"
MODEL_REVISION = "5cf2132abc99cad020ac570b19d031efec650f2b"
MODEL_CACHE_DIR = "models--Qwen--Qwen3-Embedding-4B"
IMAGE = "ghcr.io/huggingface/text-embeddings-inference:86-1.9.3"
IMAGE_EXPECTED_DIGEST = (
    "sha256:a7d82dfef16c3bf1a95e93f5b226f358312512dbb0d585b48c3cf886f9d470a9"
)
CONTAINER_PREFIX = "tei_batch_token_limit"
CANONICAL_TEXT_FIELD = "text"
CANONICAL_TEXT_RULE = "comment: body; submission: title + ' ' + selftext (Phase 1 DATA_DICTIONARY.md field 25)"
DECIMAL_CEILING_BYTES = 21_000_000_000
PACKAGE_NAMES = ["httpx", "pyarrow", "numpy", "psutil", "nvidia-ml-py", "pyyaml"]
TEI_ARGS_FIXED = [
    "--auto-truncate",
    "false",
    "--max-client-batch-size",
    "1024",
    "--payload-limit",
    "100000000",
    "--json-output",
]

# ----------------------------------------------------------------------------------------------
# Pure functions (unit-tested)
# ----------------------------------------------------------------------------------------------


def mib(n_bytes):
    """Bytes to mebibytes, one decimal."""
    return round(int(n_bytes) / 1048576, 1)


def gib(n_bytes):
    """Bytes to gibibytes, three decimals."""
    return round(int(n_bytes) / 1073741824, 3)


def rank_candidates(n_tokens, n_chars, doc_keys, n):
    """Indices of the n longest documents: tokens descending, then characters descending,
    then doc_key ascending (a total order, so the ranking is reproducible)."""
    n_tokens = np.asarray(n_tokens, dtype=np.int64)
    n_chars = np.asarray(n_chars, dtype=np.int64)
    doc_keys = np.asarray(doc_keys).astype(str)
    order = np.lexsort((doc_keys, -n_chars, -n_tokens))
    return [int(i) for i in order[:n]]


def fill_batch(candidates, budget):
    """First-fit-decreasing: walk the candidates (sorted by n_tokens descending) and take every
    document whose token count still fits under the budget; stop when the budget is met exactly."""
    selected, total = [], 0
    for c in candidates:
        n = int(c["n_tokens"])
        if total + n <= budget:
            selected.append(c)
            total += n
            if total == budget:
                break
    return selected, total


_PROM_LINE = re.compile(r"^([A-Za-z_:][A-Za-z0-9_:]*)(\{[^}]*\})?\s+(\S+)")
_PROM_LABEL = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)="((?:[^"\\]|\\.)*)"')


def parse_prometheus(text):
    """Prometheus text exposition to {metric name: [(labels dict, value)]}."""
    metrics = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _PROM_LINE.match(line)
        if not m:
            continue
        name, labels_text, value_text = m.groups()
        labels = {}
        if labels_text:
            for k, v in _PROM_LABEL.findall(labels_text):
                labels[k] = v.replace('\\"', '"').replace("\\\\", "\\")
        try:
            value = float(value_text)
        except ValueError:
            continue
        metrics.setdefault(name, []).append((labels, value))
    return metrics


def histogram(metrics, name):
    """The buckets (cumulative counts by `le`), sum and count of a histogram metric."""
    buckets = {}
    for labels, value in metrics.get(name + "_bucket", []):
        if "le" in labels:
            buckets[labels["le"]] = value
    total = (
        sum(v for _, v in metrics.get(name + "_sum", []))
        if metrics.get(name + "_sum")
        else 0.0
    )
    count = (
        sum(v for _, v in metrics.get(name + "_count", []))
        if metrics.get(name + "_count")
        else 0.0
    )
    return {"buckets": buckets, "sum": float(total), "count": float(count)}


def histogram_delta(before, after):
    keys = set(before["buckets"]) | set(after["buckets"])
    return {
        "buckets": {
            k: after["buckets"].get(k, 0.0) - before["buckets"].get(k, 0.0)
            for k in keys
        },
        "sum": after["sum"] - before["sum"],
        "count": after["count"] - before["count"],
    }


def counter_value(metrics, name, labels):
    for lab, value in metrics.get(name, []):
        if lab == labels:
            return value
    return 0.0


def verify_single_batch(size_delta, tokens_delta, k, blocker_tokens, batch_tokens):
    """Did TEI serve the two blockers and the test documents as exactly three batches of sizes
    1, 1 and k? Empty polls of the queue record a 0 in both histograms; because every input has
    at least two tokens (text plus the appended end-of-text token), an observation at or below
    1 token is an empty poll."""
    n_empty = tokens_delta["buckets"].get("1", 0.0)
    real = tokens_delta["count"] - n_empty
    singletons = size_delta["buckets"].get("1", 0.0) - n_empty
    checks = {
        "three_real_batches": real == 3,
        "two_singleton_batches": singletons == (3 if k == 1 else 2),
        "size_sum": size_delta["sum"] == 2 + k,
        "token_sum": tokens_delta["sum"] == 2 * blocker_tokens + batch_tokens,
    }
    info = {
        "n_empty_polls": n_empty,
        "real_batches": real,
        "singleton_batches": singletons,
    }
    return all(checks.values()), checks, info


def compute_cap_bytes(ceiling_bytes, device_used_before, untracked_overhead):
    """The allocation cap of one test: the ceiling minus what the device already holds before
    the container starts minus the process's untracked device memory."""
    return max(
        0, int(ceiling_bytes) - int(device_used_before) - int(untracked_overhead)
    )


def next_test_value(tested, n_min, resolution, growth=2):
    """The next `max_batch_tokens` to test, from the conclusive results so far.

    tested: list of {"max_batch_tokens": int, "success": bool}. Returns (value, purpose) or
    (None, reason). Bracket by multiplying the largest safe value by `growth`; once a failure
    above it exists, bisect on the `resolution` grid until the two are `resolution` apart."""
    if not tested:
        return int(n_min), "start"
    successes = [int(t["max_batch_tokens"]) for t in tested if t["success"]]
    failures = [int(t["max_batch_tokens"]) for t in tested if not t["success"]]
    if not successes:
        return None, "no permitted value succeeded"
    safe = max(successes)
    above = [f for f in failures if f > safe]
    if not above:
        return safe * int(growth), "bracket"
    fail = min(above)
    if fail - safe <= resolution:
        return None, "bracketed within resolution"
    mid = safe + (fail - safe) // 2
    mid -= mid % resolution
    if mid <= safe:
        mid = safe + resolution
    return int(mid), "bisect"


def classify_outcome(
    started, below_minimum, refused, batch_ok, verified, watchdog, peak_over
):
    """(outcome, success, failure_at_ceiling) from the flags of one test. A refusal at the cap is
    the primary event even when the watchdog also fired (the device excursion of a rejected pool
    attempt) or the device peak went over the ceiling during that excursion."""
    if refused and not below_minimum:
        return (
            ("batch_refused_at_cap" if started else "startup_refused_at_cap"),
            False,
            True,
        )
    if watchdog:
        return "watchdog_kill", False, True
    if not started:
        if below_minimum:
            return "refused_below_minimum", False, False
        if peak_over:
            return "device_peak_over_ceiling", False, True
        return "startup_error", False, False
    if peak_over:
        return "device_peak_over_ceiling", False, True
    if not batch_ok:
        return "batch_error", False, False
    if not verified:
        return "inconclusive_unverified_batch", False, False
    return "success", True, False


def _parse_ts(value):
    try:
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


def parse_tei_log_events(lines):
    """Timestamps of the start-up milestones in TEI's JSON log lines and the derived durations."""
    events = {
        "backend_start": None,
        "model_start": None,
        "model_variant": None,
        "warmup_start": None,
        "ready": None,
        "load_s": None,
        "warmup_s": None,
    }
    for line in lines:
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        msg = str(rec.get("message", ""))
        ts = rec.get("timestamp")
        if msg == "Starting model backend":
            events["backend_start"] = ts
        m = re.match(r"Starting (\w+) model on", msg)
        if m:
            events["model_start"] = ts
            events["model_variant"] = m.group(1)
        if msg == "Warming up model":
            events["warmup_start"] = ts
        if msg == "Ready":
            events["ready"] = ts
    t_model = _parse_ts(events["model_start"]) if events["model_start"] else None
    t_warm = _parse_ts(events["warmup_start"]) if events["warmup_start"] else None
    t_ready = _parse_ts(events["ready"]) if events["ready"] else None
    if t_model and t_warm:
        events["load_s"] = (t_warm - t_model).total_seconds()
    if t_warm and t_ready:
        events["warmup_s"] = (t_ready - t_warm).total_seconds()
    return events


def tei_log_errors(lines):
    """Error messages in the container log: TEI's ERROR records and the plain-text error lines
    printed when the router aborts."""
    errors = []
    for line in lines:
        s = line.rstrip()
        if s.startswith("{"):
            try:
                rec = json.loads(s)
            except Exception:
                continue
            if str(rec.get("level", "")).upper() == "ERROR":
                errors.append(str(rec.get("message", "")))
        elif s.startswith("Error") or s.startswith("Caused by") or s.startswith("    "):
            errors.append(s.strip())
    return errors


def summarise_samples(samples, t0, t1):
    """Peak, first and last value of the (t, value) samples inside [t0, t1] (None = unbounded)."""
    window = [
        (t, v)
        for t, v in samples
        if (t0 is None or t >= t0) and (t1 is None or t <= t1)
    ]
    if not window:
        return {
            "peak": None,
            "t_peak": None,
            "first": None,
            "last": None,
            "n_samples": 0,
        }
    t_peak, peak = max(window, key=lambda x: x[1])
    return {
        "peak": peak,
        "t_peak": t_peak,
        "first": window[0][1],
        "last": window[-1][1],
        "n_samples": len(window),
    }


def downsample_max(samples, bin_s):
    """The maximum value per time bin of width bin_s, as (bin start, max)."""
    bins = {}
    for t, v in samples:
        b = (t // bin_s) * bin_s
        bins[b] = v if b not in bins else max(bins[b], v)
    return sorted(bins.items())


class PeakSampler:
    """Samples a value in a background thread, keeps every (time, value) pair and the peak, and
    calls `on_sample(value)` for each reading (used for the watchdog)."""

    def __init__(self, sample_fn, interval_s=0.02, on_sample=None):
        self._fn = sample_fn
        self._interval = interval_s
        self._on_sample = on_sample
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.samples = []
        self.peak = None
        self.first = None
        self.last = None
        self.n_samples = 0

    def _observe(self):
        try:
            value = self._fn()
        except Exception:
            return
        if value is None:
            return
        now = time.time()
        if self.first is None:
            self.first = value
        self.last = value
        self.peak = value if self.peak is None else max(self.peak, value)
        self.n_samples += 1
        self.samples.append((now, value))
        if self._on_sample is not None:
            try:
                self._on_sample(value)
            except Exception:
                pass

    def _run(self):
        while not self._stop.is_set():
            self._observe()
            self._stop.wait(self._interval)

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join()
        self._observe()
        return {
            "peak": self.peak,
            "first": self.first,
            "last": self.last,
            "n_samples": self.n_samples,
        }


def build_manifest(
    output_rel,
    output_hash,
    output_format,
    rows,
    cols,
    inputs,
    script_rel,
    script_hash,
    parameters,
    packages,
    git_commit,
    timestamp,
    notes,
    os_name,
    python_version,
):
    """A provenance manifest as an ordered mapping."""
    m = {
        "manifest_version": 1,
        "output_file": output_rel,
        "output_hash": output_hash,
        "output_format": output_format,
    }
    if rows is not None:
        m["output_rows"] = int(rows)
    if cols is not None:
        m["output_cols"] = int(cols)
    m["input_files"] = list(inputs)
    transformation = {
        "script": script_rel,
        "script_hash": script_hash,
        "parameters": dict(parameters),
    }
    if git_commit:
        transformation["git_commit"] = git_commit
    m["transformation"] = transformation
    m["software"] = {
        "language": "Python",
        "language_version": python_version,
        "packages": list(packages),
        "os": os_name,
    }
    m["schema_diff"] = {"added": [], "removed": [], "renamed": [], "retyped": []}
    m["timestamp"] = timestamp
    m["notes"] = notes
    return m


def manifest_yaml(manifest):
    import yaml

    return yaml.safe_dump(
        manifest,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
        width=100,
    )


def _cell(value):
    if value is None:
        return ""
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_table(columns, rows):
    """A Markdown table from a list of dicts."""
    lines = [
        "| " + " | ".join(_cell(c) for c in columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_cell(row.get(c)) for c in columns) + " |")
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------------
# Runtime helpers
# ----------------------------------------------------------------------------------------------


def utc_now():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(message):
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


def git_short_sha(project_dir):
    try:
        out = subprocess.run(
            ["git", "-C", str(project_dir), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:
        pass
    try:
        git_dir = Path(project_dir) / ".git"
        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
        if head.startswith("ref: "):
            ref = head[5:]
            ref_file = git_dir / ref
            if ref_file.exists():
                return ref_file.read_text(encoding="utf-8").strip()[:7]
            packed = git_dir / "packed-refs"
            if packed.exists():
                for line in packed.read_text(encoding="utf-8").splitlines():
                    if line.endswith(" " + ref):
                        return line.split()[0][:7]
        else:
            return head[:7]
    except Exception:
        pass
    return ""


def package_versions(names=PACKAGE_NAMES):
    out = []
    for name in names:
        try:
            out.append({"name": name, "version": importlib.metadata.version(name)})
        except importlib.metadata.PackageNotFoundError:
            out.append({"name": name, "version": "not-installed"})
    return out


def rel_path(path, project_dir):
    try:
        return str(
            Path(path).resolve().relative_to(Path(project_dir).resolve())
        ).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def read_sidecar_hash(sidecar_path):
    import yaml

    data = yaml.safe_load(Path(sidecar_path).read_text(encoding="utf-8"))
    return data.get("output_hash", "")


FORMAT_BY_SUFFIX = {
    ".csv": "csv",
    ".parquet": "parquet",
    ".json": "other",
    ".txt": "other",
    ".md": "other",
    ".log": "other",
}


def table_shape(path):
    path = Path(path)
    if path.suffix == ".csv":
        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, [])
            rows = sum(1 for _ in reader)
        return rows, len(header)
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        md = pq.read_metadata(path)
        return md.num_rows, md.num_columns
    return None, None


class Sidecars:
    """Writes the provenance sidecar of every output of a stage."""

    def __init__(self, project_dir, parameters, notes):
        self.project_dir = Path(project_dir)
        self.parameters = parameters
        self.notes = notes
        self.script_rel = rel_path(SCRIPT_PATH, project_dir)
        self.script_hash = sha256_file(SCRIPT_PATH)
        self.git_commit = git_short_sha(project_dir)
        self.packages = package_versions()
        self.os_name = platform.platform()
        self.python_version = platform.python_version()
        self.written = []

    def input_entry(self, path, role, hash_value=None, with_shape=False):
        path = Path(path)
        entry = {
            "path": rel_path(path, self.project_dir),
            "hash": hash_value or sha256_file(path),
            "format": FORMAT_BY_SUFFIX.get(path.suffix, "other"),
            "role": role,
        }
        if with_shape:
            rows, cols = table_shape(path)
            if rows is not None:
                entry["rows"], entry["cols"] = rows, cols
        return entry

    def write(self, output_path, inputs, notes=None, parameters=None):
        output_path = Path(output_path)
        fmt = FORMAT_BY_SUFFIX.get(output_path.suffix, "other")
        rows, cols = (
            table_shape(output_path) if fmt in ("csv", "parquet") else (None, None)
        )
        manifest = build_manifest(
            output_rel=rel_path(output_path, self.project_dir),
            output_hash=sha256_file(output_path),
            output_format=fmt,
            rows=rows,
            cols=cols,
            inputs=inputs,
            script_rel=self.script_rel,
            script_hash=self.script_hash,
            parameters=parameters if parameters is not None else self.parameters,
            packages=self.packages,
            git_commit=self.git_commit,
            timestamp=utc_now(),
            notes=notes if notes is not None else self.notes,
            os_name=self.os_name,
            python_version=self.python_version,
        )
        sidecar = output_path.with_name(output_path.name + ".manifest.yml")
        sidecar.write_text(manifest_yaml(manifest), encoding="utf-8")
        self.written.append(sidecar)
        return sidecar


def write_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def append_csv_row(path, row, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerow(row)


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, ensure_ascii=False, default=_json_default)
    return path


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run(cmd, timeout=None, check=False, merge_stderr=False, cwd=None):
    out = subprocess.run(
        cmd,
        capture_output=not merge_stderr,
        stdout=subprocess.PIPE if merge_stderr else None,
        stderr=subprocess.STDOUT if merge_stderr else None,
        text=True,
        timeout=timeout,
        cwd=cwd,
    )
    if check and out.returncode != 0:
        raise RuntimeError(
            f"command failed ({out.returncode}): {' '.join(map(str, cmd))}\n{out.stdout}\n{out.stderr or ''}"
        )
    return out


class Nvml:
    def __init__(self, index=0):
        import pynvml

        self.p = pynvml
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(index)

    def used(self):
        return int(self.p.nvmlDeviceGetMemoryInfo(self.handle).used)

    def total(self):
        return int(self.p.nvmlDeviceGetMemoryInfo(self.handle).total)

    def name(self):
        value = self.p.nvmlDeviceGetName(self.handle)
        return value.decode() if isinstance(value, bytes) else str(value)

    def driver(self):
        value = self.p.nvmlSystemGetDriverVersion()
        return value.decode() if isinstance(value, bytes) else str(value)

    def shutdown(self):
        try:
            self.p.nvmlShutdown()
        except Exception:
            pass


def nvidia_smi_text():
    try:
        out = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=60)
        return out.stdout + out.stderr
    except Exception as e:
        return f"nvidia-smi not available: {e!r}"


# --- Docker -------------------------------------------------------------------------------------


def docker_image_record():
    """Identity of the TEI image actually present: id, repo digests, creation time, size, and
    the router's own version string."""
    out = run(
        ["docker", "image", "inspect", IMAGE, "--format", "{{json .}}"], timeout=60
    )
    if out.returncode != 0:
        raise RuntimeError(
            f"image {IMAGE} is not present: {out.stderr.strip()} (docker pull {IMAGE})"
        )
    info = json.loads(out.stdout)
    version = run(
        [
            "docker",
            "run",
            "--rm",
            "--gpus",
            "all",
            "--entrypoint",
            "/usr/local/bin/text-embeddings-router",
            IMAGE,
            "--version",
        ],
        timeout=120,
        merge_stderr=True,
    )
    return {
        "image": IMAGE,
        "image_id": info.get("Id"),
        "repo_digests": info.get("RepoDigests"),
        "expected_index_digest": IMAGE_EXPECTED_DIGEST,
        "digest_matches_expected": any(
            IMAGE_EXPECTED_DIGEST in d for d in (info.get("RepoDigests") or [])
        ),
        "created": info.get("Created"),
        "size_bytes": info.get("Size"),
        "architecture": info.get("Architecture"),
        "os": info.get("Os"),
        "labels": (info.get("Config") or {}).get("Labels"),
        "env": (info.get("Config") or {}).get("Env"),
        "router_version_output": version.stdout.strip(),
    }


def docker_versions_text():
    parts = []
    for cmd in (
        ["docker", "--version"],
        [
            "docker",
            "info",
            "--format",
            "server={{.ServerVersion}} runtimes={{.Runtimes}} default_runtime={{.DefaultRuntime}} storage={{.Driver}} root={{.DockerRootDir}}",
        ],
        ["containerd", "--version"],
        ["nvidia-ctk", "--version"],
        ["nvidia-container-cli", "--version"],
        [
            "dpkg-query",
            "-W",
            "-f=${Package} ${Version}\\n",
            "docker-ce",
            "docker-ce-cli",
            "containerd.io",
            "nvidia-container-toolkit",
            "nvidia-container-toolkit-base",
            "libnvidia-container-tools",
            "libnvidia-container1",
        ],
        ["cat", "/etc/docker/daemon.json"],
    ):
        try:
            out = run(cmd, timeout=60, merge_stderr=True)
            parts.append(f"$ {' '.join(cmd)}\n{out.stdout.strip()}\n")
        except Exception as e:
            parts.append(f"$ {' '.join(cmd)}\n(failed: {e!r})\n")
    return "\n".join(parts)


def docker_state(name):
    out = run(
        [
            "docker",
            "inspect",
            "-f",
            "{{.State.Running}} {{.State.ExitCode}} {{.State.OOMKilled}} {{.State.Error}}",
            name,
        ],
        timeout=60,
    )
    if out.returncode != 0:
        return {"exists": False, "running": False, "exit_code": None}
    parts = out.stdout.strip().split(" ", 3)
    return {
        "exists": True,
        "running": parts[0] == "true",
        "exit_code": int(parts[1])
        if len(parts) > 1 and parts[1].lstrip("-").isdigit()
        else None,
        "oom_killed": parts[2] == "true" if len(parts) > 2 else None,
        "error": parts[3] if len(parts) > 3 else "",
    }


def docker_logs(name):
    out = run(["docker", "logs", name], timeout=120, merge_stderr=True)
    return out.stdout


def docker_stop(name, timeout_s=60):
    run(["docker", "stop", "-t", str(int(timeout_s)), name], timeout=timeout_s + 60)


def docker_kill(name):
    subprocess.Popen(
        ["docker", "kill", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )


def docker_remove(name):
    run(["docker", "rm", "-f", name], timeout=120)


def docker_remove_stale():
    out = run(
        [
            "docker",
            "ps",
            "-a",
            "--filter",
            f"name={CONTAINER_PREFIX}",
            "--format",
            "{{.Names}}",
        ],
        timeout=60,
    )
    for name in out.stdout.split():
        log(f"removing stale container {name}")
        docker_remove(name)


def docker_run_tei(ctx, name, max_batch_tokens, cap_bytes, report_path):
    """Starts one TEI container with the cap shim; returns the docker run command used."""
    snapshot = f"/data/{MODEL_CACHE_DIR}/snapshots/{MODEL_REVISION}"
    cmd = [
        "docker",
        "run",
        "-d",
        "--name",
        name,
        "--gpus",
        "all",
        "--network",
        "host",
        "-v",
        f"{ctx['hub_dir']}:/data:ro",
        "-v",
        f"{ctx['shim_build_dir']}:/vram_cap:ro",
        "-v",
        f"{ctx['shim_report_dir']}:/vram_cap_out",
        "-e",
        "HF_HUB_OFFLINE=1",
        "-e",
        "LD_LIBRARY_PATH=/vram_cap:/usr/local/cuda/lib64",
        "-e",
        f"VRAM_CAP_BYTES={int(cap_bytes)}",
        "-e",
        f"VRAM_CAP_REPORT=/vram_cap_out/{Path(report_path).name}",
        "-e",
        "VRAM_CAP_VERBOSE=0",
        IMAGE,
        "--model-id",
        snapshot,
        "--port",
        str(ctx["port"]),
        "--max-batch-tokens",
        str(int(max_batch_tokens)),
        *TEI_ARGS_FIXED,
    ]
    out = run(cmd, timeout=120)
    if out.returncode != 0:
        raise RuntimeError(f"docker run failed: {out.stderr.strip()}")
    return cmd


# --- HTTP ---------------------------------------------------------------------------------------


def http_client(ctx, timeout_s):
    import httpx

    return httpx.Client(
        base_url=f"http://127.0.0.1:{ctx['port']}",
        timeout=httpx.Timeout(timeout_s, connect=10.0),
    )


def wait_ready(ctx, name, timeout_s):
    """Polls /health until it answers 200 (ready), the container exits, or the timeout."""
    t0 = time.time()
    with http_client(ctx, 5.0) as client:
        while time.time() - t0 < timeout_s:
            state = docker_state(name)
            if state["exists"] and not state["running"]:
                return "exited", time.time() - t0, state
            try:
                r = client.get("/health")
                if r.status_code == 200:
                    return "ready", time.time() - t0, state
            except Exception:
                pass
            time.sleep(0.25)
    return "timeout", time.time() - t0, docker_state(name)


def get_info(ctx):
    with http_client(ctx, 30.0) as client:
        return client.get("/info").json()


def get_metrics(ctx):
    with http_client(ctx, 30.0) as client:
        return parse_prometheus(client.get("/metrics").text)


HEADER_KEYS = [
    "x-compute-tokens",
    "x-compute-characters",
    "x-total-time",
    "x-tokenization-time",
    "x-queue-time",
    "x-inference-time",
    "x-compute-time",
]


def post_embed(ctx, texts, timeout_s):
    """One /embed request; returns status, the timing headers, the embedding shape, the error."""
    t0 = time.perf_counter()
    result = {
        "n_inputs": len(texts),
        "status": None,
        "headers": {},
        "wall_s": None,
        "n_embeddings": None,
        "dim": None,
        "error": None,
        "exception": None,
    }
    try:
        with http_client(ctx, timeout_s) as client:
            r = client.post(
                "/embed", json={"inputs": texts, "truncate": False, "normalize": True}
            )
        result["wall_s"] = time.perf_counter() - t0
        result["status"] = r.status_code
        result["headers"] = {
            k: r.headers.get(k) for k in HEADER_KEYS if r.headers.get(k) is not None
        }
        if r.status_code == 200:
            data = r.json()
            result["n_embeddings"] = len(data)
            result["dim"] = len(data[0]) if data else 0
        else:
            try:
                result["error"] = r.json()
            except Exception:
                result["error"] = r.text[:2000]
    except Exception as e:
        result["wall_s"] = time.perf_counter() - t0
        result["exception"] = repr(e)
    return result


def post_tokenize(ctx, texts):
    with http_client(ctx, 600.0) as client:
        r = client.post("/tokenize", json={"inputs": texts, "add_special_tokens": True})
    r.raise_for_status()
    return r.json()


def read_shim_report(path):
    for _ in range(3):
        try:
            return json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:
            time.sleep(0.05)
    return None


# --- data ---------------------------------------------------------------------------------------


def canonical_paths(project_dir):
    parquet = (
        Path(project_dir)
        / "pipeline"
        / "phase1_message_event_foundation"
        / "derived"
        / "message_events.parquet"
    )
    return parquet, parquet.with_name(parquet.name + ".manifest.yml")


def token_table_paths():
    parquet = PRIOR_FEAS_DIR / "results" / "token_counts_all_documents.parquet"
    return parquet, parquet.with_name(parquet.name + ".manifest.yml")


def verify_inputs(project_dir):
    """Hash the canonical table and the token table against their sidecars."""
    checks = {}
    for label, (path, sidecar) in (
        ("canonical", canonical_paths(project_dir)),
        ("token_table", token_table_paths()),
    ):
        if not path.exists():
            raise SystemExit(f"{label} table missing: {path}")
        measured = sha256_file(path)
        expected = read_sidecar_hash(sidecar) if sidecar.exists() else ""
        checks[label] = {
            "path": rel_path(path, project_dir),
            "size_bytes": path.stat().st_size,
            "sha256_measured": measured,
            "sha256_in_sidecar": expected,
            "hash_equal_to_sidecar": measured == expected,
        }
        if measured != expected:
            raise SystemExit(
                f"{label} table hash {measured} differs from its sidecar {expected}: refusing to run on unverified data"
            )
    return checks


def load_candidates(project_dir, n_candidates):
    """The n longest documents (by the Phase 2 exact token counts) with their texts."""
    import pyarrow.parquet as pq

    token_parquet, _ = token_table_paths()
    table = pq.read_table(
        token_parquet, columns=["doc_key", "record_type", "n_chars", "n_tokens_4B"]
    )
    doc_keys = np.array(table.column("doc_key").to_pylist(), dtype=str)
    record_types = np.array(table.column("record_type").to_pylist(), dtype=str)
    n_chars = np.array(table.column("n_chars").to_pylist(), dtype=np.int64)
    n_tokens = np.array(table.column("n_tokens_4B").to_pylist(), dtype=np.int64)
    n_rows = table.num_rows
    del table
    idx = rank_candidates(n_tokens, n_chars, doc_keys, n_candidates)
    # A tail of shorter documents, one per token length below the shortest top candidate (the
    # longest document of that length), so that first-fit-decreasing can meet any budget exactly.
    idx, pools = list(idx), ["top"] * len(idx)
    min_top = int(n_tokens[idx[-1]]) if idx else 0
    seen = set()
    for i in np.lexsort((doc_keys, -n_chars, -n_tokens)):
        n = int(n_tokens[i])
        if n >= min_top:
            continue
        if n < 2:
            break
        if n not in seen:
            seen.add(n)
            idx.append(int(i))
            pools.append("tail")
    wanted = {str(doc_keys[i]) for i in idx}
    canonical, _ = canonical_paths(project_dir)
    ct = pq.read_table(canonical, columns=["doc_key", CANONICAL_TEXT_FIELD])
    keys = ct.column("doc_key").to_pylist()
    texts_col = ct.column(CANONICAL_TEXT_FIELD)
    positions = {k: i for i, k in enumerate(keys) if k in wanted}
    del keys
    candidates = []
    for rank, (i, pool) in enumerate(zip(idx, pools), start=1):
        key = str(doc_keys[i])
        text = texts_col[positions[key]].as_py()
        if not text:
            continue
        candidates.append(
            {
                "rank": rank,
                "doc_key": key,
                "record_type": str(record_types[i]),
                "n_chars": int(n_chars[i]),
                "n_tokens": int(n_tokens[i]),
                "pool": pool,
                "text": text,
            }
        )
    del ct
    return candidates, {
        "n_documents_in_token_table": int(n_rows),
        "sum_tokens_all_documents": int(n_tokens.sum()),
        "max_tokens": int(n_tokens.max()),
        "n_top_candidates": sum(1 for c in candidates if c["pool"] == "top"),
        "n_tail_candidates": sum(1 for c in candidates if c["pool"] == "tail"),
        "shortest_top_candidate_tokens": min_top,
    }


CANDIDATE_FIELDS = (
    "rank",
    "pool",
    "doc_key",
    "record_type",
    "n_chars",
    "n_tokens",
    "n_tokens_tei",
)


def snapshot_record(hub_dir):
    """The model files actually used: the snapshot path and the blob names (Hub LFS blobs are named
    by their SHA256) with sizes; small files hashed directly."""
    snap = Path(hub_dir) / MODEL_CACHE_DIR / "snapshots" / MODEL_REVISION
    files = []
    for p in sorted(snap.rglob("*")):
        if p.is_dir():
            continue
        entry = {"file": str(p.relative_to(snap)), "size_bytes": p.stat().st_size}
        if p.is_symlink():
            entry["blob"] = os.readlink(p).split("/")[-1]
        if p.stat().st_size < 50_000_000:
            entry["sha256"] = sha256_file(p)
        files.append(entry)
    return {
        "repo": MODEL_REPO,
        "revision": MODEL_REVISION,
        "snapshot_path": str(snap),
        "files": files,
    }


def environment_record(nvml, image_record):
    record = {
        "platform": platform.platform(),
        "kernel_release": platform.release(),
        "wsl_distro": os.environ.get("WSL_DISTRO_NAME", ""),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "cpu_count": os.cpu_count(),
        "gpu_name": nvml.name(),
        "driver_version": nvml.driver(),
        "gpu_total_memory_bytes": nvml.total(),
        "tei_image": image_record.get("image"),
        "tei_image_id": image_record.get("image_id"),
        "tei_repo_digests": ";".join(image_record.get("repo_digests") or []),
        "tei_router_version": image_record.get("router_version_output"),
    }
    try:
        import psutil

        record["ram_total_bytes"] = int(psutil.virtual_memory().total)
    except Exception:
        pass
    for pkg in package_versions():
        record[f"package_{pkg['name']}"] = pkg["version"]
    return record


def results_dir(args):
    return Path(args.feas_dir) / "results"


def environment_dir(args):
    return Path(args.feas_dir) / "environment"


def common_parameters(args):
    return {
        "model": f"{MODEL_REPO} @ {MODEL_REVISION} (local snapshot, HF_HUB_OFFLINE=1)",
        "tei_image": IMAGE,
        "tei_expected_index_digest": IMAGE_EXPECTED_DIGEST,
        "tei_fixed_args": " ".join(TEI_ARGS_FIXED)
        + " --port <port> --max-batch-tokens <tested value>",
        "tei_dtype": "float16 (TEI default; the only precision of the flash Qwen3 path)",
        "tei_pooling": "from 1_Pooling/config.json (last_token)",
        "embed_request": '{"inputs": [...], "truncate": false, "normalize": true}',
        "vram_ceiling_mib": args.ceiling_mib,
        "vram_ceiling_bytes": args.ceiling_mib * 1048576,
        "decimal_ceiling_bytes": DECIMAL_CEILING_BYTES,
        "enforcement": "in-process allocation cap (scripts/vram_cap libcuda.so.1 shim) = ceiling - device used before start - untracked overhead; NVML watchdog kills the container if the device total exceeds the ceiling; success requires every NVML sample at or below the ceiling",
        "nvml_sample_interval_ms": args.nvml_interval_ms,
        "search": f"start at the TEI minimum with truncation disabled; multiply by {args.growth} until failure; bisect on a {args.resolution}-token grid; confirmation rerun of the safe value",
        "batch_construction": "first-fit-decreasing over the longest documents (tokens desc, chars desc, doc_key asc) until the budget is met",
        "single_batch_protocol": f"two single-document blocker requests {args.blocker_gap_ms} ms apart, then the batch request; verified from the Prometheus deltas and x-compute-tokens; up to {args.attempts} attempts",
        "n_candidates": args.n_candidates,
        "calibration_cap_bytes": int(args.calibration_cap_gib * 1073741824),
        "below_minimum_demo_value": args.below_minimum_value,
        "canonical_text_field": CANONICAL_TEXT_FIELD,
        "canonical_text_rule": CANONICAL_TEXT_RULE,
        "token_counts_source": rel_path(token_table_paths()[0], args.project_dir)
        + " (n_tokens_4B; counts include the appended <|endoftext|>)",
    }


def make_ctx(args, nvml):
    return {
        "args": args,
        "nvml": nvml,
        "port": args.port,
        "hub_dir": str(Path(args.hub_dir).expanduser()),
        "shim_build_dir": str(Path(args.shim_build_dir).expanduser()),
        "shim_report_dir": str(Path(args.shim_report_dir).expanduser()),
        "ceiling_bytes": args.ceiling_mib * 1048576,
    }


def build_shim(args):
    build_dir = Path(args.shim_build_dir).expanduser()
    out = run(
        [
            "bash",
            str(SHIM_SRC_DIR / "build_vram_cap.sh"),
            str(SHIM_SRC_DIR),
            str(build_dir),
        ],
        timeout=600,
        merge_stderr=True,
    )
    if out.returncode != 0:
        raise SystemExit(f"shim build failed:\n{out.stdout}")
    so = build_dir / "libcuda.so.1"
    record = {
        "build_output": out.stdout,
        "build_dir": str(build_dir),
        "libcuda_so_1_sha256": sha256_file(so),
        "vram_cap_c_sha256": sha256_file(SHIM_SRC_DIR / "vram_cap.c"),
        "gen_trampolines_py_sha256": sha256_file(SHIM_SRC_DIR / "gen_trampolines.py"),
        "build_script_sha256": sha256_file(SHIM_SRC_DIR / "build_vram_cap.sh"),
    }
    return record


# ----------------------------------------------------------------------------------------------
# One test = one container
# ----------------------------------------------------------------------------------------------


def settle_device(ctx, reference_used, margin_bytes, timeout_s):
    """Waits until the device total has come back down near the reference reading."""
    t0 = time.time()
    used = ctx["nvml"].used()
    while used > reference_used + margin_bytes and time.time() - t0 < timeout_s:
        time.sleep(0.5)
        used = ctx["nvml"].used()
    return used


def submit_batch(
    ctx, blocker_text, blocker_tokens, batch_texts, batch_tokens, k, timeout_s, gap_s
):
    """Two blockers then the batch; returns everything measured, including the verification."""
    before = get_metrics(ctx)
    results = {}

    def send(label, texts):
        results[label] = post_embed(ctx, texts, timeout_s)

    threads = [
        threading.Thread(target=send, args=("blocker_1", [blocker_text])),
        threading.Thread(target=send, args=("blocker_2", [blocker_text])),
        threading.Thread(target=send, args=("batch", batch_texts)),
    ]
    t_start = time.time()
    threads[0].start()
    time.sleep(gap_s)
    threads[1].start()
    time.sleep(gap_s)
    t_batch_sent = time.time()
    threads[2].start()
    for th in threads:
        th.join()
    t_end = time.time()
    after = get_metrics(ctx)
    size_delta = histogram_delta(
        histogram(before, "te_batch_next_size"), histogram(after, "te_batch_next_size")
    )
    tokens_delta = histogram_delta(
        histogram(before, "te_batch_next_tokens"),
        histogram(after, "te_batch_next_tokens"),
    )
    inference_delta = histogram_delta(
        histogram(before, "te_embed_inference_duration"),
        histogram(after, "te_embed_inference_duration"),
    )
    failures = {}
    for labels, value in after.get("te_request_failure", []):
        key = labels.get("err", "?")
        failures[key] = value - counter_value(before, "te_request_failure", labels)
    batch = results["batch"]
    ok_http = (
        batch["status"] == 200
        and results["blocker_1"]["status"] == 200
        and results["blocker_2"]["status"] == 200
    )
    verified, checks, info = verify_single_batch(
        size_delta, tokens_delta, k, blocker_tokens, batch_tokens
    )
    compute_tokens = batch["headers"].get("x-compute-tokens")
    compute_tokens_equal = (
        compute_tokens is not None and int(compute_tokens) == batch_tokens
    )
    return {
        "t_start": t_start,
        "t_batch_sent": t_batch_sent,
        "t_end": t_end,
        "requests": results,
        "all_requests_ok": ok_http,
        "metrics_delta": {
            "te_batch_next_size": size_delta,
            "te_batch_next_tokens": tokens_delta,
            "te_embed_inference_duration": inference_delta,
            "te_request_failure": failures,
        },
        "verification": {
            "single_batch_verified": bool(
                verified and ok_http and compute_tokens_equal
            ),
            "histogram_checks": checks,
            "histogram_info": info,
            "x_compute_tokens": compute_tokens,
            "x_compute_tokens_equals_batch_total": compute_tokens_equal,
        },
    }


def run_test(ctx, seq, max_batch_tokens, purpose, candidates, calibration):
    """One tested value in one fresh container; returns the full record and writes its files."""
    args = ctx["args"]
    nvml = ctx["nvml"]
    name = f"{CONTAINER_PREFIX}_{seq:02d}"
    rdir = results_dir(args)
    report_path = (
        Path(ctx["shim_report_dir"]) / f"test_{seq:02d}_{max_batch_tokens}.json"
    )
    if report_path.exists():
        try:
            report_path.unlink()
        except Exception:
            pass
    selected, batch_tokens = fill_batch(candidates, max_batch_tokens)
    blocker = candidates[0]
    rec = {
        "seq": seq,
        "purpose": purpose,
        "max_batch_tokens": int(max_batch_tokens),
        "container": name,
        "timestamp_start": utc_now(),
        "ceiling_bytes": ctx["ceiling_bytes"],
        "untracked_overhead_bytes": calibration["untracked_overhead_bytes"],
        "batch_plan": {
            "n_documents": len(selected),
            "batch_tokens": batch_tokens,
            "budget_gap_tokens": int(max_batch_tokens) - batch_tokens,
            "doc_keys": [c["doc_key"] for c in selected],
            "n_tokens": [c["n_tokens"] for c in selected],
            "n_chars": [c["n_chars"] for c in selected],
            "record_types": [c["record_type"] for c in selected],
            "blocker_doc_key": blocker["doc_key"],
            "blocker_tokens": blocker["n_tokens"],
        },
    }
    settle_used = settle_device(
        ctx,
        ctx["session_baseline_used"],
        args.settle_margin_mib * 1048576,
        args.settle_s,
    )
    device_used_before = nvml.used()
    cap_bytes = compute_cap_bytes(
        ctx["ceiling_bytes"],
        device_used_before,
        calibration["untracked_overhead_bytes"],
    )
    rec.update(
        {
            "device_used_before_start_bytes": device_used_before,
            "device_used_after_settle_bytes": settle_used,
            "cap_bytes": cap_bytes,
        }
    )
    log(
        f"test {seq:02d} [{purpose}] max_batch_tokens={max_batch_tokens}: K={len(selected)} docs, {batch_tokens} tokens; device before {mib(device_used_before)} MiB; cap {mib(cap_bytes)} MiB"
    )
    watchdog = {
        "killed": False,
        "used_at_kill": None,
        "first_over_t": None,
        "over_samples": 0,
        "over_peak": 0,
        "refusal_seen": False,
        "last_report_check": 0.0,
    }

    def on_sample(used):
        if used <= ctx["ceiling_bytes"]:
            watchdog["first_over_t"] = None
            return
        now = time.time()
        watchdog["over_samples"] += 1
        watchdog["over_peak"] = max(watchdog["over_peak"], used)
        if watchdog["first_over_t"] is None:
            watchdog["first_over_t"] = now
        if watchdog["killed"]:
            return
        # A refusal in progress: the shim has rejected a pool attempt and is trimming the pool;
        # the device is allowed a short grace period to come back under the ceiling.
        if now - watchdog["last_report_check"] > 0.1:
            watchdog["last_report_check"] = now
            rep = read_shim_report(report_path)
            if rep and int(rep.get("n_refusals", 0)) > 0:
                watchdog["refusal_seen"] = True
        if (
            watchdog["refusal_seen"]
            and now - watchdog["first_over_t"] < args.watchdog_grace_s
        ):
            return
        watchdog["killed"] = True
        watchdog["used_at_kill"] = used
        log(
            f"  WATCHDOG: device total {mib(used)} MiB exceeds the ceiling"
            f"{' after a shim refusal' if watchdog['refusal_seen'] else ''}; killing {name}"
        )
        docker_kill(name)

    sampler = PeakSampler(
        nvml.used, interval_s=args.nvml_interval_ms / 1000.0, on_sample=on_sample
    ).start()
    t_run = time.time()
    rec["docker_run_command"] = docker_run_tei(
        ctx, name, max_batch_tokens, cap_bytes, report_path
    )
    state, startup_s, dstate = wait_ready(ctx, name, args.startup_timeout_s)
    t_ready = time.time()
    rec.update(
        {
            "startup_state": state,
            "startup_wait_s": startup_s,
            "container_state_after_startup": dstate,
        }
    )
    started = state == "ready"
    below_minimum = False
    batch = None
    info = None
    shim_at_ready = None
    if started:
        try:
            info = get_info(ctx)
        except Exception as e:
            info = {"error": repr(e)}
        shim_at_ready = read_shim_report(report_path)
        time.sleep(1.0)
        rec["device_used_at_ready_bytes"] = nvml.used()
        rec["shim_at_ready"] = shim_at_ready
        attempts = []
        for attempt in range(1, args.attempts + 1):
            log(
                f"  batch attempt {attempt}: sending 2 blockers + {len(selected)} documents ({batch_tokens} tokens)"
            )
            result = submit_batch(
                ctx,
                blocker["text"],
                blocker["n_tokens"],
                [c["text"] for c in selected],
                batch_tokens,
                len(selected),
                args.request_timeout_s,
                args.blocker_gap_ms / 1000.0,
            )
            result["attempt"] = attempt
            result["shim_after"] = read_shim_report(report_path)
            attempts.append(result)
            if not result["all_requests_ok"]:
                log(
                    f"  request failure: batch status {result['requests']['batch']['status']} {str(result['requests']['batch'].get('error') or result['requests']['batch'].get('exception'))[:300]}"
                )
                break
            if result["verification"]["single_batch_verified"]:
                log(
                    f"  verified: one batch of {len(selected)} documents, {result['verification']['x_compute_tokens']} tokens; inference {result['requests']['batch']['headers'].get('x-inference-time')} ms"
                )
                break
            log(
                f"  batch not verified as a single batch ({result['verification']['histogram_checks']}); retrying"
            )
            time.sleep(1.0)
        rec["batch_attempts"] = attempts
        batch = attempts[-1]
    # Stop (gracefully, so that the shim writes its exit record) and collect the evidence.
    if docker_state(name)["running"]:
        try:
            docker_stop(name, 60)
        except Exception as e:
            log(f"  docker stop failed: {e!r}; killing")
            docker_kill(name)
            time.sleep(2)
    t_stopped = time.time()
    logs_text = docker_logs(name)
    final_state = docker_state(name)
    docker_remove(name)
    sampler.stop()
    shim_final = read_shim_report(report_path)
    log_lines = logs_text.splitlines()
    events = parse_tei_log_events(log_lines)
    errors = tei_log_errors(log_lines)
    if not started:
        below_minimum = any(
            "exceeds `--max-batch-tokens" in ln or "exceeds --max-batch-tokens" in ln
            for ln in log_lines
        )
    refusals_final = int(shim_final.get("n_refusals", 0)) if shim_final else 0
    refused = refusals_final > 0 or any("[vram_cap] REFUSED" in ln for ln in log_lines)
    batch_ok = bool(batch and batch["all_requests_ok"])
    verified = bool(batch and batch["verification"]["single_batch_verified"])
    startup_peak = summarise_samples(sampler.samples, t_run, t_ready)
    batch_peak = (
        summarise_samples(sampler.samples, batch["t_start"], batch["t_end"])
        if batch
        else summarise_samples([], None, None)
    )
    overall_peak = summarise_samples(sampler.samples, t_run, t_stopped)
    peak_over = (
        overall_peak["peak"] is not None and overall_peak["peak"] > ctx["ceiling_bytes"]
    )
    outcome, success, failure_at_ceiling = classify_outcome(
        started,
        below_minimum,
        refused,
        batch_ok,
        verified,
        watchdog["killed"],
        peak_over,
    )
    detail = ""
    if outcome == "refused_below_minimum":
        detail = next(
            (ln for ln in errors if "max-batch-tokens" in ln), "; ".join(errors)[:600]
        )
    elif outcome in ("startup_refused_at_cap", "batch_refused_at_cap"):
        detail = (shim_final or {}).get("last_refusal", "") or "; ".join(errors)[:600]
        if batch and batch["requests"]["batch"].get("error"):
            detail += f" | TEI: {json.dumps(batch['requests']['batch']['error'])[:400]}"
    elif outcome == "watchdog_kill":
        detail = f"device total reached {mib(watchdog['used_at_kill'])} MiB"
    elif outcome == "device_peak_over_ceiling":
        detail = f"device total peaked at {mib(overall_peak['peak'])} MiB"
    elif outcome in ("startup_error", "batch_error"):
        detail = "; ".join(errors)[:600] or (
            json.dumps(
                batch["requests"]["batch"].get("error")
                or batch["requests"]["batch"].get("exception")
            )[:600]
            if batch
            else ""
        )
    elif outcome == "inconclusive_unverified_batch":
        detail = json.dumps(batch["verification"]["histogram_checks"]) if batch else ""
    rec.update(
        {
            "timestamp_end": utc_now(),
            "started": started,
            "below_minimum": below_minimum,
            "info": info,
            "log_events": events,
            "log_errors": errors,
            "shim_final": shim_final,
            "shim_refusals": refusals_final,
            "refused_at_cap": refused,
            "watchdog": watchdog,
            "container_final_state": final_state,
            "nvml": {
                "interval_ms": args.nvml_interval_ms,
                "startup": startup_peak,
                "batch": batch_peak,
                "overall": overall_peak,
                "series_1s_max": [
                    (round(t - t_run, 3), v)
                    for t, v in downsample_max(sampler.samples, 1.0)
                ],
            },
            "batch_ok": batch_ok,
            "verified_single_batch": verified,
            "peak_over_ceiling": peak_over,
            "outcome": outcome,
            "success": success,
            "failure_at_ceiling": failure_at_ceiling,
            "failure_detail": detail,
        }
    )
    stem = f"test_{seq:02d}_{max_batch_tokens}"
    (rdir / "container_logs").mkdir(parents=True, exist_ok=True)
    (rdir / "container_logs" / f"{stem}.log").write_text(logs_text, encoding="utf-8")
    (rdir / "shim_reports").mkdir(parents=True, exist_ok=True)
    if shim_final is not None:
        write_json(rdir / "shim_reports" / f"{stem}.json", shim_final)
    write_json(rdir / "tests" / f"{stem}.json", rec)
    append_csv_row(rdir / "tested_values.csv", tested_row(rec), TESTED_COLUMNS)
    for pos, (key, n, ch, rt) in enumerate(
        zip(
            rec["batch_plan"]["doc_keys"],
            rec["batch_plan"]["n_tokens"],
            rec["batch_plan"]["n_chars"],
            rec["batch_plan"]["record_types"],
        ),
        start=1,
    ):
        append_csv_row(
            rdir / "batch_compositions.csv",
            {
                "seq": seq,
                "max_batch_tokens": max_batch_tokens,
                "position": pos,
                "doc_key": key,
                "n_tokens": n,
                "n_chars": ch,
                "record_type": rt,
            },
            COMPOSITION_COLUMNS,
        )
    log(
        f"  outcome: {outcome}; peak device {mib(overall_peak['peak']) if overall_peak['peak'] else None} MiB; {detail[:200]}"
    )
    return rec


TESTED_COLUMNS = [
    "seq",
    "purpose",
    "max_batch_tokens",
    "outcome",
    "success",
    "failure_at_ceiling",
    "cap_bytes",
    "cap_mib",
    "device_used_before_start_mib",
    "untracked_overhead_mib",
    "started",
    "startup_wait_s",
    "load_s",
    "warmup_s",
    "startup_peak_device_mib",
    "n_documents",
    "batch_tokens_planned",
    "x_compute_tokens",
    "verified_single_batch",
    "attempts",
    "batch_wall_s",
    "tei_inference_ms",
    "tei_queue_ms",
    "tei_tokenization_ms",
    "batch_peak_device_mib",
    "overall_peak_device_mib",
    "overall_peak_under_ceiling",
    "overall_peak_under_21e9_bytes",
    "shim_peak_mib",
    "shim_physical_peak_mib",
    "shim_pool_reserved_high_mib",
    "shim_overshoot_mib",
    "shim_used_at_ready_mib",
    "shim_refusals",
    "watchdog_killed",
    "enforcement",
    "device_over_ceiling_samples",
    "device_over_ceiling_ms",
    "device_over_ceiling_peak_mib",
    "embedding_dim",
    "failure_detail",
]
COMPOSITION_COLUMNS = [
    "seq",
    "max_batch_tokens",
    "position",
    "doc_key",
    "n_tokens",
    "n_chars",
    "record_type",
]


def tested_row(rec):
    batch = (rec.get("batch_attempts") or [None])[-1]
    breq = batch["requests"]["batch"] if batch else {}
    headers = breq.get("headers", {}) if batch else {}

    def hdr(key):
        v = headers.get(key)
        return int(v) if v is not None and str(v).isdigit() else v

    shim = rec.get("shim_final") or {}
    peak = rec["nvml"]["overall"]["peak"]
    return {
        "seq": rec["seq"],
        "purpose": rec["purpose"],
        "max_batch_tokens": rec["max_batch_tokens"],
        "outcome": rec["outcome"],
        "success": rec["success"],
        "failure_at_ceiling": rec["failure_at_ceiling"],
        "cap_bytes": rec["cap_bytes"],
        "cap_mib": mib(rec["cap_bytes"]),
        "device_used_before_start_mib": mib(rec["device_used_before_start_bytes"]),
        "untracked_overhead_mib": mib(rec["untracked_overhead_bytes"]),
        "started": rec["started"],
        "startup_wait_s": round(rec["startup_wait_s"], 2),
        "load_s": rec["log_events"].get("load_s"),
        "warmup_s": rec["log_events"].get("warmup_s"),
        "startup_peak_device_mib": mib(rec["nvml"]["startup"]["peak"])
        if rec["nvml"]["startup"]["peak"]
        else None,
        "n_documents": rec["batch_plan"]["n_documents"],
        "batch_tokens_planned": rec["batch_plan"]["batch_tokens"],
        "x_compute_tokens": hdr("x-compute-tokens"),
        "verified_single_batch": rec["verified_single_batch"],
        "attempts": len(rec.get("batch_attempts") or []),
        "batch_wall_s": round(breq["wall_s"], 3)
        if batch and breq.get("wall_s") is not None
        else None,
        "tei_inference_ms": hdr("x-inference-time"),
        "tei_queue_ms": hdr("x-queue-time"),
        "tei_tokenization_ms": hdr("x-tokenization-time"),
        "batch_peak_device_mib": mib(rec["nvml"]["batch"]["peak"])
        if rec["nvml"]["batch"]["peak"]
        else None,
        "overall_peak_device_mib": mib(peak) if peak else None,
        "overall_peak_under_ceiling": (peak <= rec["ceiling_bytes"]) if peak else None,
        "overall_peak_under_21e9_bytes": (peak <= DECIMAL_CEILING_BYTES)
        if peak
        else None,
        "shim_peak_mib": mib(shim.get("peak_bytes", 0)) if shim else None,
        "shim_physical_peak_mib": mib(shim.get("peak_physical_bytes", 0))
        if shim
        else None,
        "shim_pool_reserved_high_mib": mib(shim.get("pool_reserved_high_bytes", 0))
        if shim
        else None,
        "shim_overshoot_mib": mib(shim.get("max_overshoot_bytes", 0)) if shim else None,
        "shim_used_at_ready_mib": mib(rec["shim_at_ready"]["used_bytes"])
        if rec.get("shim_at_ready")
        else None,
        "shim_refusals": rec["shim_refusals"],
        "watchdog_killed": rec["watchdog"]["killed"],
        "enforcement": enforcement_label(rec),
        "device_over_ceiling_samples": rec["watchdog"].get("over_samples", 0),
        "device_over_ceiling_ms": int(
            round(rec["watchdog"].get("over_samples", 0) * rec["nvml"]["interval_ms"])
        ),
        "device_over_ceiling_peak_mib": mib(rec["watchdog"]["over_peak"])
        if rec["watchdog"].get("over_peak")
        else None,
        "embedding_dim": breq.get("dim") if batch else None,
        "failure_detail": rec["failure_detail"],
    }


def enforcement_label(rec):
    """Which mechanism decided a failed test (or 'none' for a success)."""
    if rec["outcome"] == "refused_below_minimum":
        return "TEI argument check (no container start)"
    if rec["shim_refusals"] > 0:
        return "cap: allocation refused inside TEI"
    if rec["watchdog"]["killed"]:
        return "watchdog: device total over the ceiling, container killed"
    if rec["outcome"] == "device_peak_over_ceiling":
        return "NVML: device total over the ceiling"
    if rec["success"]:
        return "none (success)"
    return "none"


# ----------------------------------------------------------------------------------------------
# Stage: prepare
# ----------------------------------------------------------------------------------------------


def stage_prepare(args):
    t_stage = time.perf_counter()
    rdir, edir = results_dir(args), environment_dir(args)
    rdir.mkdir(parents=True, exist_ok=True)
    edir.mkdir(parents=True, exist_ok=True)
    for name in (
        "prepare.json",
        "candidate_documents.csv",
        "tokenize_check.csv",
        "tei_info.json",
    ):
        for p in (rdir / name, rdir / (name + ".manifest.yml")):
            if p.exists():
                p.unlink()
    log("verifying the canonical table and the token table against their sidecars")
    checks = verify_inputs(args.project_dir)
    log(f"selecting the {args.n_candidates} longest documents and reading their texts")
    t0 = time.perf_counter()
    candidates, corpus = load_candidates(args.project_dir, args.n_candidates)
    t_load = time.perf_counter() - t0
    log(
        f"  {len(candidates)} candidates; longest {candidates[0]['doc_key']} with {candidates[0]['n_tokens']} tokens; sum of the candidates' tokens {sum(c['n_tokens'] for c in candidates)}"
    )
    log("building the cap shim")
    shim = build_shim(args)
    (edir / "vram_cap_build.txt").write_text(shim["build_output"], encoding="utf-8")
    log("recording the image, Docker, driver and model facts")
    image = docker_image_record()
    write_json(edir / "tei_image.json", image)
    (edir / "docker_versions.txt").write_text(docker_versions_text(), encoding="utf-8")
    (edir / "nvidia_smi.txt").write_text(nvidia_smi_text(), encoding="utf-8")
    nvml = Nvml()
    ctx = make_ctx(args, nvml)
    Path(ctx["shim_report_dir"]).mkdir(parents=True, exist_ok=True)
    snapshot = snapshot_record(ctx["hub_dir"])
    docker_remove_stale()
    # Calibration container: generous cap, the minimum permitted budget.
    name = f"{CONTAINER_PREFIX}_calibration"
    report_path = Path(ctx["shim_report_dir"]) / "calibration.json"
    if report_path.exists():
        report_path.unlink()
    cap = int(args.calibration_cap_gib * 1073741824)
    device_before = nvml.used()
    sampler = PeakSampler(nvml.used, interval_s=args.nvml_interval_ms / 1000.0).start()
    t_run = time.time()
    log(
        f"calibration run: max_batch_tokens={args.n_min}, cap {mib(cap)} MiB, device before {mib(device_before)} MiB"
    )
    cmd = docker_run_tei(ctx, name, args.n_min, cap, report_path)
    state, startup_s, dstate = wait_ready(ctx, name, args.startup_timeout_s)
    t_ready = time.time()
    if state != "ready":
        logs_text = docker_logs(name)
        docker_remove(name)
        raise SystemExit(
            f"calibration container did not become ready ({state}):\n{logs_text[-4000:]}"
        )
    info = get_info(ctx)
    write_json(rdir / "tei_info.json", info)
    time.sleep(3.0)
    device_steady = nvml.used()
    shim_steady = read_shim_report(report_path)
    shim_physical_steady = int(
        shim_steady.get("physical_bytes", shim_steady["used_bytes"])
    )
    untracked = (device_steady - device_before) - shim_physical_steady
    info_checks = {
        "auto_truncate_false": info.get("auto_truncate") is False,
        "max_input_length": info.get("max_input_length"),
        "max_batch_tokens_equals_requested": info.get("max_batch_tokens") == args.n_min,
        "dtype_float16": info.get("model_dtype") == "float16",
        "pooling_last_token": (info.get("model_type") or {})
        .get("embedding", {})
        .get("pooling")
        == "last_token",
        "version": info.get("version"),
        "sha": info.get("sha"),
        "docker_label": info.get("docker_label"),
    }
    log(
        f"  ready in {startup_s:.1f} s; device steady {mib(device_steady)} MiB; shim physical {mib(shim_physical_steady)} MiB (live {mib(shim_steady['used_bytes'])} MiB, pool known {shim_steady.get('pool_known')}); untracked overhead {mib(untracked)} MiB; /info version {info.get('version')} sha {info.get('sha')}"
    )
    # TEI's own token counts of every candidate.
    log(
        f"tokenizing the {len(candidates)} candidates through /tokenize in chunks of {args.tokenize_chunk}"
    )
    t0 = time.perf_counter()
    tei_counts = []
    for start in range(0, len(candidates), args.tokenize_chunk):
        chunk = candidates[start : start + args.tokenize_chunk]
        tokens = post_tokenize(ctx, [c["text"] for c in chunk])
        tei_counts.extend(len(t) for t in tokens)
    t_tokenize = time.perf_counter() - t0
    n_equal = 0
    check_rows = []
    for c, n_tei in zip(candidates, tei_counts):
        c["n_tokens_tei"] = int(n_tei)
        equal = int(n_tei) == c["n_tokens"]
        n_equal += equal
        check_rows.append(
            {
                "rank": c["rank"],
                "doc_key": c["doc_key"],
                "record_type": c["record_type"],
                "n_chars": c["n_chars"],
                "n_tokens_stored": c["n_tokens"],
                "n_tokens_tei": int(n_tei),
                "equal": equal,
            }
        )
    log(f"  {n_equal}/{len(candidates)} counts equal to the stored counts")
    first_tokens = post_tokenize(ctx, [candidates[0]["text"]])[0]
    special = [t for t in first_tokens if t.get("special")]
    embed = post_embed(ctx, [candidates[0]["text"]], args.request_timeout_s)
    log(
        f"  single-document embed: status {embed['status']}, dim {embed['dim']}, inference {embed['headers'].get('x-inference-time')} ms"
    )
    docker_stop(name, 60)
    logs_text = docker_logs(name)
    docker_remove(name)
    sampler.stop()
    shim_final = read_shim_report(report_path)
    events = parse_tei_log_events(logs_text.splitlines())
    (rdir / "container_logs").mkdir(parents=True, exist_ok=True)
    (rdir / "container_logs" / "calibration.log").write_text(
        logs_text, encoding="utf-8"
    )
    write_csv(
        rdir / "candidate_documents.csv",
        [{k: c[k] for k in CANDIDATE_FIELDS} for c in candidates],
    )
    write_csv(rdir / "tokenize_check.csv", check_rows)
    prepare = {
        "stage": "prepare",
        "timestamp": utc_now(),
        "input_checks": checks,
        "corpus": corpus,
        "candidates": [{k: c[k] for k in CANDIDATE_FIELDS} for c in candidates],
        "tokenize_check": {
            "n_candidates": len(candidates),
            "n_equal": n_equal,
            "all_equal": n_equal == len(candidates),
            "special_tokens_of_longest": special,
            "wall_s": t_tokenize,
        },
        "shim_build": shim,
        "image": image,
        "snapshot": snapshot,
        "calibration": {
            "container": name,
            "docker_run_command": cmd,
            "max_batch_tokens": args.n_min,
            "cap_bytes": cap,
            "device_used_before_start_bytes": device_before,
            "startup_wait_s": startup_s,
            "log_events": events,
            "device_used_steady_bytes": device_steady,
            "shim_steady": shim_steady,
            "shim_final": shim_final,
            "untracked_overhead_bytes": untracked,
            "startup_peak_device_bytes": summarise_samples(
                sampler.samples, t_run, t_ready
            )["peak"],
            "overall_peak_device_bytes": sampler.peak,
            "info": info,
            "info_checks": info_checks,
            "single_document_embed": embed,
        },
        "timings": {
            "load_candidates_s": t_load,
            "tokenize_s": t_tokenize,
            "stage_total_s": time.perf_counter() - t_stage,
        },
    }
    write_json(rdir / "prepare.json", prepare)
    nvml.shutdown()
    sidecars = Sidecars(
        args.project_dir,
        common_parameters(args),
        "TEI batch-token limit test, prepare stage.",
    )
    inputs = [
        sidecars.input_entry(
            canonical_paths(args.project_dir)[0],
            "canonical Phase 1 table, read-only",
            checks["canonical"]["sha256_measured"],
            with_shape=True,
        ),
        sidecars.input_entry(
            token_table_paths()[0],
            "Phase 2 exact token counts (n_tokens_4B), read-only",
            checks["token_table"]["sha256_measured"],
            with_shape=True,
        ),
    ]
    for path, note in (
        (
            rdir / "prepare.json",
            "inputs, candidates, calibration run, tokenize check, image and snapshot facts",
        ),
        (
            rdir / "candidate_documents.csv",
            "the candidate documents (longest first) with the stored and TEI token counts",
        ),
        (
            rdir / "tokenize_check.csv",
            "stored token count vs TEI /tokenize count for every candidate",
        ),
        (rdir / "tei_info.json", "the /info document of the calibration container"),
        (
            rdir / "container_logs" / "calibration.log",
            "container log of the calibration run",
        ),
        (
            edir / "vram_cap_build.txt",
            "build log and hashes of the allocation-cap shim",
        ),
        (edir / "tei_image.json", "identity of the TEI image used"),
        (
            edir / "docker_versions.txt",
            "Docker Engine and NVIDIA Container Toolkit versions",
        ),
        (edir / "nvidia_smi.txt", "nvidia-smi inside WSL2 before the calibration run"),
    ):
        sidecars.write(path, inputs, notes=note)
    log(f"prepare done in {time.perf_counter() - t_stage:.1f} s")
    return prepare


# ----------------------------------------------------------------------------------------------
# Stage: search
# ----------------------------------------------------------------------------------------------


def stage_search(args):
    t_stage = time.perf_counter()
    rdir = results_dir(args)
    prepare = read_json(rdir / "prepare.json")
    if not prepare["tokenize_check"]["all_equal"]:
        raise SystemExit(
            "the stored token counts differ from TEI's for some candidates; see results/tokenize_check.csv"
        )
    for name in ("tested_values.csv", "batch_compositions.csv", "search_summary.json"):
        for p in (rdir / name, rdir / (name + ".manifest.yml")):
            if p.exists():
                p.unlink()
    for sub in ("tests", "container_logs", "shim_reports"):
        d = rdir / sub
        if d.exists():
            for p in d.glob("test_*"):
                p.unlink()
    log("reading the candidate texts")
    candidates, _ = load_candidates(args.project_dir, args.n_candidates)
    prepared = {c["doc_key"]: c for c in prepare["candidates"]}
    if [c["doc_key"] for c in candidates] != [
        c["doc_key"] for c in prepare["candidates"]
    ]:
        raise SystemExit(
            "the candidate list differs from the prepare stage; rerun prepare"
        )
    for c in candidates:
        c["n_tokens_tei"] = prepared[c["doc_key"]]["n_tokens_tei"]
    nvml = Nvml()
    ctx = make_ctx(args, nvml)
    Path(ctx["shim_report_dir"]).mkdir(parents=True, exist_ok=True)
    docker_remove_stale()
    ctx["session_baseline_used"] = nvml.used()
    n_min = int(prepare["calibration"]["info"]["max_input_length"])
    calibration = prepare["calibration"]
    log(
        f"search: n_min={n_min} (TEI's max_input_length with truncation disabled), resolution {args.resolution}, ceiling {args.ceiling_mib} MiB, untracked overhead {mib(calibration['untracked_overhead_bytes'])} MiB, session baseline {mib(ctx['session_baseline_used'])} MiB"
    )
    tests = []
    seq = 0
    aborted = None
    # 1. The value below TEI's minimum: documents the refusal (not a VRAM result).
    if args.below_minimum_value and args.below_minimum_value < n_min:
        tests.append(
            run_test(
                ctx,
                seq,
                args.below_minimum_value,
                "below_minimum_demo",
                candidates,
                calibration,
            )
        )
        seq += 1
    # 2. Bracket, bisect, confirm.
    inconclusive_repeats = {}
    confirmed = set()
    while seq <= args.max_tests:
        conclusive = [
            t
            for t in tests
            if t["purpose"] != "below_minimum_demo"
            and (t["success"] or t["failure_at_ceiling"])
        ]
        value, purpose = next_test_value(
            conclusive, n_min, args.resolution, args.growth
        )
        if value is None:
            if purpose == "bracketed within resolution":
                safe = max(t["max_batch_tokens"] for t in conclusive if t["success"])
                if safe in confirmed:
                    break
                rec = run_test(ctx, seq, safe, "confirmation", candidates, calibration)
                seq += 1
                tests.append(rec)
                if rec["success"]:
                    confirmed.add(safe)
                    break
                if rec["failure_at_ceiling"]:
                    for t in tests:
                        if t["max_batch_tokens"] == safe and t["success"]:
                            t["success"] = False
                            t["invalidated_by_confirmation"] = True
                            t["outcome"] = t["outcome"] + "_invalidated_by_confirmation"
                    continue
                # inconclusive confirmation: repeat once
                key = ("confirmation", safe)
                inconclusive_repeats[key] = inconclusive_repeats.get(key, 0) + 1
                if inconclusive_repeats[key] > 1:
                    aborted = f"confirmation of {safe} was inconclusive twice: {rec['outcome']} {rec['failure_detail']}"
                    break
                continue
            aborted = None if purpose == "bracketed within resolution" else purpose
            break
        rec = run_test(ctx, seq, value, purpose, candidates, calibration)
        seq += 1
        tests.append(rec)
        if not rec["success"] and not rec["failure_at_ceiling"]:
            key = (purpose, value)
            inconclusive_repeats[key] = inconclusive_repeats.get(key, 0) + 1
            if inconclusive_repeats[key] > 1:
                aborted = f"test of {value} was inconclusive twice: {rec['outcome']} {rec['failure_detail']}"
                break
            log(f"  inconclusive ({rec['outcome']}); repeating the value once")
    if seq > args.max_tests and aborted is None:
        aborted = f"stopped after {args.max_tests} tests"
    summary = search_summary(tests, n_min, args, aborted, ctx)
    write_json(rdir / "search_summary.json", summary)
    nvml.shutdown()
    sidecars = Sidecars(
        args.project_dir,
        common_parameters(args),
        "TEI batch-token limit test, search stage.",
    )
    inputs = [
        sidecars.input_entry(
            rdir / "prepare.json", "prepare stage result (candidates, calibration)"
        ),
        sidecars.input_entry(
            canonical_paths(args.project_dir)[0],
            "canonical Phase 1 table, read-only",
            prepare["input_checks"]["canonical"]["sha256_measured"],
        ),
    ]
    for path in (
        sorted((rdir / "tests").glob("test_*.json"))
        + sorted((rdir / "container_logs").glob("test_*.log"))
        + sorted((rdir / "shim_reports").glob("test_*.json"))
    ):
        sidecars.write(
            path, inputs, notes=f"one tested max_batch_tokens value: {path.name}"
        )
    for path, note in (
        (rdir / "tested_values.csv", "one row per test"),
        (rdir / "batch_compositions.csv", "the documents of every test batch"),
        (
            rdir / "search_summary.json",
            "the largest measured safe value, the nearest failing value above it, the confirmation",
        ),
    ):
        if path.exists():
            sidecars.write(path, inputs, notes=note)
    log(
        f"search done in {(time.perf_counter() - t_stage) / 60:.1f} min: {summary['headline']}"
    )
    if aborted:
        log(f"SEARCH ABORTED: {aborted}")
    return summary


def search_summary(tests, n_min, args, aborted, ctx):
    search_tests = [t for t in tests if t["purpose"] != "below_minimum_demo"]
    successes = [t for t in search_tests if t["success"]]
    failures = [t for t in search_tests if t["failure_at_ceiling"]]
    safe = max(successes, key=lambda t: t["max_batch_tokens"]) if successes else None
    confirmation = next(
        (
            t
            for t in search_tests
            if t["purpose"] == "confirmation"
            and safe
            and t["max_batch_tokens"] == safe["max_batch_tokens"]
        ),
        None,
    )
    failing_above = None
    if safe:
        above = [
            t for t in failures if t["max_batch_tokens"] > safe["max_batch_tokens"]
        ]
        failing_above = (
            min(above, key=lambda t: t["max_batch_tokens"]) if above else None
        )
    non_monotonic = [
        t["max_batch_tokens"]
        for t in failures
        if safe and t["max_batch_tokens"] < safe["max_batch_tokens"]
    ]

    def brief(t):
        if not t:
            return None
        return {
            "seq": t["seq"],
            "purpose": t["purpose"],
            "max_batch_tokens": t["max_batch_tokens"],
            "outcome": t["outcome"],
            "n_documents": t["batch_plan"]["n_documents"],
            "batch_tokens": t["batch_plan"]["batch_tokens"],
            "overall_peak_device_bytes": t["nvml"]["overall"]["peak"],
            "startup_peak_device_bytes": t["nvml"]["startup"]["peak"],
            "batch_peak_device_bytes": t["nvml"]["batch"]["peak"],
            "shim_peak_bytes": (t.get("shim_final") or {}).get("peak_bytes"),
            "failure_detail": t["failure_detail"],
        }

    headline = (
        "no permitted value succeeded"
        if not safe
        else f"largest measured safe max_batch_tokens = {safe['max_batch_tokens']} (peak {mib(safe['nvml']['overall']['peak'])} MiB); nearest failing above = {failing_above['max_batch_tokens'] if failing_above else 'none measured'}; confirmation {'succeeded' if confirmation and confirmation['success'] else 'not confirmed'}"
    )
    return {
        "stage": "search",
        "timestamp": utc_now(),
        "n_min": n_min,
        "resolution": args.resolution,
        "ceiling_bytes": ctx["ceiling_bytes"],
        "n_tests": len(tests),
        "tested_values_in_order": [
            (t["seq"], t["purpose"], t["max_batch_tokens"], t["outcome"]) for t in tests
        ],
        "largest_safe": brief(safe),
        "confirmation": brief(confirmation),
        "confirmation_succeeded": bool(confirmation and confirmation["success"]),
        "nearest_failing_above": brief(failing_above),
        "gap_tokens": (failing_above["max_batch_tokens"] - safe["max_batch_tokens"])
        if safe and failing_above
        else None,
        "non_monotonic_failures_below_safe": non_monotonic,
        "aborted": aborted,
        "headline": headline,
    }


# ----------------------------------------------------------------------------------------------
# Stage: report
# ----------------------------------------------------------------------------------------------


def stage_report(args):
    rdir, edir = results_dir(args), environment_dir(args)
    prepare = read_json(rdir / "prepare.json")
    summary = read_json(rdir / "search_summary.json")
    # Re-record the image identity (the router's own version string needs the GPU libraries).
    try:
        image = docker_image_record()
        image["recorded_at"] = "report stage"
        write_json(edir / "tei_image.json", image)
        prepare["image"] = image
    except Exception as e:
        log(f"image record not refreshed: {e!r}")
    tests = [read_json(p) for p in sorted((rdir / "tests").glob("test_*.json"))]
    tests.sort(key=lambda t: t["seq"])
    rows = [tested_row(t) for t in tests]
    write_csv(rdir / "tested_values.csv", rows, TESTED_COLUMNS)
    comp_rows = []
    for t in tests:
        for pos, (key, n, ch, rt) in enumerate(
            zip(
                t["batch_plan"]["doc_keys"],
                t["batch_plan"]["n_tokens"],
                t["batch_plan"]["n_chars"],
                t["batch_plan"]["record_types"],
            ),
            start=1,
        ):
            comp_rows.append(
                {
                    "seq": t["seq"],
                    "max_batch_tokens": t["max_batch_tokens"],
                    "position": pos,
                    "doc_key": key,
                    "n_tokens": n,
                    "n_chars": ch,
                    "record_type": rt,
                }
            )
    write_csv(rdir / "batch_compositions.csv", comp_rows, COMPOSITION_COLUMNS)
    nvml = Nvml()
    env = environment_record(nvml, prepare["image"])
    nvml.shutdown()
    write_csv(
        edir / "runtime_record.csv",
        [{"key": k, "value": v} for k, v in env.items()],
        ["key", "value"],
    )
    dists = sorted(
        importlib.metadata.distributions(), key=lambda d: d.metadata["Name"].lower()
    )
    write_csv(
        edir / "python_packages.csv",
        [{"name": d.metadata["Name"], "version": d.version} for d in dists],
        ["name", "version"],
    )
    report = render_report(args, prepare, summary, tests, rows, env)
    (Path(args.feas_dir) / "REPORT.md").write_text(report, encoding="utf-8")
    sidecars = Sidecars(
        args.project_dir,
        common_parameters(args),
        "TEI batch-token limit test, report stage.",
    )
    inputs = [
        sidecars.input_entry(rdir / "prepare.json", "prepare stage result"),
        sidecars.input_entry(rdir / "search_summary.json", "search stage summary"),
    ]
    inputs += [
        sidecars.input_entry(p, "one tested value")
        for p in sorted((rdir / "tests").glob("test_*.json"))
    ]
    for path, note in (
        (rdir / "tested_values.csv", "one row per tested max_batch_tokens value"),
        (rdir / "batch_compositions.csv", "the documents of every test batch"),
        (
            edir / "runtime_record.csv",
            "host, kernel, driver, GPU, image and library versions",
        ),
        (edir / "python_packages.csv", "every distribution in the venv"),
        (
            edir / "tei_image.json",
            "identity of the TEI image used (re-recorded at the report stage)",
        ),
        (Path(args.feas_dir) / "REPORT.md", "the generated report"),
    ):
        if path.exists():
            sidecars.write(path, inputs, notes=note)
    log("report written")


def render_report(args, prepare, summary, tests, rows, env):
    cal = prepare["calibration"]
    info = cal["info"]
    image = prepare["image"]
    ceiling = args.ceiling_mib * 1048576
    lines = []
    lines.append(
        "# Phase 2 feasibility report: the TEI `max_batch_tokens` limit of Qwen3-Embedding-4B under 21 GiB"
    )
    lines.append("")
    lines.append(
        f"Generated {utc_now()} by `scripts/tei_batch_token_limit.py` (stage `report`). Every number is measured; nothing is estimated or extrapolated."
    )
    lines.append("")
    lines.append("## Answer")
    lines.append("")
    safe, fail, conf = (
        summary.get("largest_safe"),
        summary.get("nearest_failing_above"),
        summary.get("confirmation"),
    )
    if safe:
        lines.append(
            f"- **Largest directly measured safe `max_batch_tokens`: {safe['max_batch_tokens']}** ({safe['n_documents']} documents, {safe['batch_tokens']} real tokens in the batch; peak device total {mib(safe['overall_peak_device_bytes'])} MiB = {gib(safe['overall_peak_device_bytes'])} GiB, ceiling {args.ceiling_mib} MiB)."
        )
        lines.append(
            f"- Peak VRAM at that value: start-up (load + warm-up of {safe['max_batch_tokens']} synthetic tokens) {mib(safe['startup_peak_device_bytes'])} MiB; the real batch {mib(safe['batch_peak_device_bytes'])} MiB; tracked allocations peak {mib(safe['shim_peak_bytes'] or 0)} MiB."
        )
        if fail:
            lines.append(
                f"- **Nearest directly measured failing value above it: {fail['max_batch_tokens']}** (outcome `{fail['outcome']}`; peak device total {mib(fail['overall_peak_device_bytes']) if fail['overall_peak_device_bytes'] else 'n/a'} MiB; {fail['failure_detail']})."
            )
        else:
            lines.append(
                "- No failing value above it was measured (the search was aborted or hit its test limit; see below)."
            )
        if conf:
            lines.append(
                f"- Confirmation rerun of {conf['max_batch_tokens']} (test {conf['seq']}): **{'succeeded' if summary['confirmation_succeeded'] else 'FAILED'}**, peak device total {mib(conf['overall_peak_device_bytes'])} MiB."
            )
        else:
            lines.append("- Confirmation rerun: not performed.")
        if summary.get("gap_tokens") is not None:
            lines.append(
                f"- Resolution reached: the safe and failing values are {summary['gap_tokens']} tokens apart (grid {summary['resolution']})."
            )
    else:
        lines.append("- No permitted value succeeded under the ceiling.")
    invalidated = [
        t for t in tests if t["outcome"].endswith("_invalidated_by_confirmation")
    ]
    for t in invalidated:
        conf = next(
            (
                c
                for c in tests
                if c["purpose"] == "confirmation"
                and c["max_batch_tokens"] == t["max_batch_tokens"]
            ),
            None,
        )
        lines.append(
            f"- {t['max_batch_tokens']} succeeded once (test {t['seq']}, peak {mib(t['nvml']['overall']['peak'])} MiB) but its confirmation rerun"
            + (
                f" (test {conf['seq']}) reached {mib(conf['nvml']['overall']['peak'])} MiB and failed (`{conf['outcome']}`)"
                if conf
                else " failed"
            )
            + ", so it is not the answer; the search continued below it."
        )
    by_value = {}
    for t in tests:
        if t["purpose"] != "below_minimum_demo" and t["nvml"]["overall"]["peak"]:
            by_value.setdefault(t["max_batch_tokens"], []).append(
                t["nvml"]["overall"]["peak"]
            )
    spreads = {v: max(p) - min(p) for v, p in by_value.items() if len(p) > 1}
    if spreads:
        v, s = max(spreads.items(), key=lambda kv: kv[1])
        lines.append(
            f"- Run-to-run variation at the same value: up to {mib(s)} MiB between the peaks of repeated runs (at {v}: {', '.join(str(mib(p)) for p in by_value[v])} MiB); the desktop's share and the pool's layout differ between containers, which is why the safe value had to succeed twice."
        )
    search_tests = [t for t in tests if t["purpose"] != "below_minimum_demo"]
    n_cap = sum(1 for t in search_tests if t["shim_refusals"] > 0)
    n_wd = sum(
        1 for t in search_tests if t["watchdog"]["killed"] and t["shim_refusals"] == 0
    )
    if n_cap or n_wd:
        wd_over = [
            t["watchdog"]["over_peak"] - ctx_ceiling
            for t in search_tests
            if t["watchdog"]["killed"] and t["shim_refusals"] == 0
            for ctx_ceiling in [args.ceiling_mib * 1048576]
        ]
        wd_ms = [
            t["watchdog"]["over_samples"] * t["nvml"]["interval_ms"]
            for t in search_tests
            if t["watchdog"]["killed"] and t["shim_refusals"] == 0
        ]
        cap_ms = [
            t["watchdog"]["over_samples"] * t["nvml"]["interval_ms"]
            for t in search_tests
            if t["shim_refusals"] > 0
        ]
        lines.append(
            f"- How the failures were decided: {n_cap} by the cap inside TEI (a pool growth request refused and undone; the device stayed over the ceiling for "
            + (f"{min(cap_ms)} to {max(cap_ms)} ms" if cap_ms else "0 ms")
            + f" during the rejected attempt), {n_wd} by the watchdog (the device total went "
            + (
                f"{mib(min(wd_over))} to {mib(max(wd_over))} MiB over the ceiling for {min(wd_ms)} to {max(wd_ms)} ms before the kill took effect"
                if wd_over
                else "over the ceiling"
            )
            + "; in those runs the memory outside the shim's count, context plus module memory, was larger than the calibrated untracked overhead, and the NVML rule covered the difference). Every success kept every 20 ms sample at or below the ceiling."
        )
    if summary.get("non_monotonic_failures_below_safe"):
        lines.append(
            f"- Failures were also recorded below the safe value at {summary['non_monotonic_failures_below_safe']} (non-monotonic; see the table)."
        )
    if summary.get("aborted"):
        lines.append(f"- **Search aborted:** {summary['aborted']}")
    lines.append("")
    lines.append("## What ran")
    lines.append("")
    lines.append(
        f"- TEI: `{image['image']}`, version {info.get('version')}, git sha `{info.get('sha')}`, docker label `{info.get('docker_label')}`; image id `{image.get('image_id')}`; repo digest `{'; '.join(image.get('repo_digests') or [])}` (expected index digest matches: {image.get('digest_matches_expected')}); created {image.get('created')}; `{image.get('router_version_output')}`."
    )
    lines.append(
        f"- Runtime: Docker Engine and NVIDIA Container Toolkit inside WSL2 (`environment/docker_versions.txt`), driver {env.get('driver_version')}, {env.get('gpu_name')} with {mib(env.get('gpu_total_memory_bytes', 0))} MiB, kernel {env.get('kernel_release')}."
    )
    snap = prepare["snapshot"]
    shards = [f for f in snap["files"] if f["file"].endswith(".safetensors")]
    lines.append(
        f"- Model: `{snap['repo']}` revision `{snap['revision']}` loaded from the local snapshot `{snap['snapshot_path']}` with `HF_HUB_OFFLINE=1`; weight shards "
        + ", ".join(
            f"`{f['file']}` (blob sha256 `{f.get('blob')}`, {f['size_bytes']} bytes)"
            for f in shards
        )
        + "."
    )
    lines.append(
        f"- TEI settings from `/info`: max_input_length {info.get('max_input_length')}, auto_truncate {info.get('auto_truncate')}, dtype {info.get('model_dtype')}, pooling {(info.get('model_type') or {}).get('embedding', {}).get('pooling')}, max_client_batch_size {info.get('max_client_batch_size')}, tokenization_workers {info.get('tokenization_workers')}; model variant from the log: {cal['log_events'].get('model_variant')}. Fixed arguments: `{' '.join(TEI_ARGS_FIXED)}`; requests send `truncate: false`."
    )
    lines.append(
        f"- Enforcement of the 21 GiB ceiling ({ceiling} bytes = {args.ceiling_mib} MiB of total device memory as NVML reports it, other processes included). (1) The allocation-cap shim `scripts/vram_cap/`, loaded as `libcuda.so.1` in the container, caps TEI's physical device memory (live non-pool allocations plus the bytes the stream-ordered memory pool has reserved, which is what candle's `cuMemAllocAsync` consumes) at `cap = ceiling - device total before the container starts - untracked overhead`; a request that cannot fit is refused with CUDA_ERROR_OUT_OF_MEMORY inside TEI, and a pool request that could only be served by growing the pool past the cap is undone (freed on an idle stream, pool trimmed) and refused, with the short excursion of that rejected attempt recorded (`shim_overshoot_mib`, `device_over_ceiling_ms`). (2) The untracked overhead ({mib(cal['untracked_overhead_bytes'])} MiB) is the calibration run's device total at rest minus its baseline minus the shim's physical bytes; it is the CUDA context and module memory that no allocation call reports, and it moves by roughly +-150 MiB with the desktop's own share between runs. (3) An NVML sampler ({args.nvml_interval_ms} ms) records the device total for the whole test and kills the container if a sample exceeds the ceiling (after a 3 s grace when the shim has just logged a refusal). A test is a success only if TEI started, the batch was embedded as one verified batch and every sample stayed at or below the ceiling; the `enforcement` column says which mechanism decided each failure. Shim build: `libcuda.so.1` sha256 `{prepare['shim_build']['libcuda_so_1_sha256']}`, source `{prepare['shim_build']['vram_cap_c_sha256']}`."
    )
    lines.append(
        f"- Documents: the {prepare['tokenize_check']['n_candidates']} longest documents of the canonical table (SHA256 `{prepare['input_checks']['canonical']['sha256_measured']}`, equal to its sidecar: {prepare['input_checks']['canonical']['hash_equal_to_sidecar']}) by the Phase 2 exact counts (token table SHA256 `{prepare['input_checks']['token_table']['sha256_measured']}`, equal to its sidecar: {prepare['input_checks']['token_table']['hash_equal_to_sidecar']}); TEI's `/tokenize` count equals the stored count for {prepare['tokenize_check']['n_equal']} of {prepare['tokenize_check']['n_candidates']} candidates; special tokens appended by TEI: {[t.get('text') for t in prepare['tokenize_check']['special_tokens_of_longest']]}. Batches are filled first-fit-decreasing from the longest document down; texts unchanged, nothing synthetic, nothing duplicated."
    )
    lines.append(
        "- Single-batch proof: two single-document blocker requests then the batch request; the Prometheus deltas must show exactly three non-empty batches of sizes 1, 1 and K with the exact token sum, and `x-compute-tokens` must equal the batch total."
    )
    lines.append(
        f"- Calibration run ({cal['max_batch_tokens']} tokens, cap {mib(cal['cap_bytes'])} MiB): ready in {cal['startup_wait_s']:.1f} s (load {cal['log_events'].get('load_s')} s, warm-up {cal['log_events'].get('warmup_s')} s), device total {mib(cal['device_used_steady_bytes'])} MiB at rest (baseline before start {mib(cal['device_used_before_start_bytes'])} MiB) with {mib(cal['shim_steady'].get('physical_bytes', cal['shim_steady']['used_bytes']))} MiB physical in TEI ({mib(cal['shim_steady']['used_bytes'])} MiB live tensors, pool reserved {mib(cal['shim_steady'].get('pool_reserved_bytes', 0))} MiB); embedding dimension {cal['single_document_embed'].get('dim')}."
    )
    lines.append("")
    lines.append("## Every tested value, in order")
    lines.append("")
    lines.append(
        "Peaks are the device total (all processes) sampled every 20 ms: 'start-up' covers model load and the warm-up batch of `max_batch_tokens` synthetic tokens, 'batch' covers the blocker and test requests. 'x-compute-tokens' is TEI's own count of the tokens in the batch request. Wall-clock is the client's for the batch request; 'TEI inference' is the backend duration of the batch (`x-inference-time`)."
    )
    lines.append("")
    cols = [
        "seq",
        "purpose",
        "max_batch_tokens",
        "outcome",
        "n_documents",
        "batch_tokens_planned",
        "x_compute_tokens",
        "verified_single_batch",
        "load_s",
        "warmup_s",
        "startup_peak_device_mib",
        "batch_wall_s",
        "tei_inference_ms",
        "batch_peak_device_mib",
        "overall_peak_device_mib",
        "overall_peak_under_ceiling",
        "cap_mib",
        "shim_physical_peak_mib",
        "shim_overshoot_mib",
        "enforcement",
        "device_over_ceiling_ms",
        "device_over_ceiling_peak_mib",
        "failure_detail",
    ]
    lines.append(render_table(cols, rows))
    lines.append("")
    if safe:
        st = next(t for t in tests if t["seq"] == safe["seq"])
        lines.append(
            f"## The batch of the safe value ({safe['max_batch_tokens']}, test {safe['seq']})"
        )
        lines.append("")
        lines.append(
            f"{st['batch_plan']['n_documents']} documents, {st['batch_plan']['batch_tokens']} tokens (budget gap {st['batch_plan']['budget_gap_tokens']}); TEI reported x-compute-tokens {st['batch_attempts'][-1]['verification']['x_compute_tokens']}; metrics deltas: {json.dumps(st['batch_attempts'][-1]['metrics_delta']['te_batch_next_size'])}, tokens {json.dumps(st['batch_attempts'][-1]['metrics_delta']['te_batch_next_tokens'])}."
        )
        lines.append("")
        lines.append(
            render_table(
                ["position", "doc_key", "record_type", "n_tokens", "n_chars"],
                [
                    {
                        "position": i + 1,
                        "doc_key": k,
                        "record_type": r,
                        "n_tokens": n,
                        "n_chars": c,
                    }
                    for i, (k, r, n, c) in enumerate(
                        zip(
                            st["batch_plan"]["doc_keys"],
                            st["batch_plan"]["record_types"],
                            st["batch_plan"]["n_tokens"],
                            st["batch_plan"]["n_chars"],
                        )
                    )
                ],
            )
        )
        lines.append("")
    lines.append("## Files")
    lines.append("")
    base = rel_path(args.feas_dir, args.project_dir)
    for rel in (
        "scripts/tei_batch_token_limit.py",
        "scripts/setup_docker_wsl2.sh",
        "scripts/vram_cap/vram_cap.c",
        "scripts/vram_cap/gen_trampolines.py",
        "scripts/vram_cap/build_vram_cap.sh",
        "tests/test_tei_batch_token_limit_functions.py",
        "results/prepare.json",
        "results/candidate_documents.csv",
        "results/tokenize_check.csv",
        "results/tei_info.json",
        "results/tested_values.csv",
        "results/batch_compositions.csv",
        "results/search_summary.json",
        "results/tests/",
        "results/container_logs/",
        "results/shim_reports/",
        "environment/tei_image.json",
        "environment/docker_versions.txt",
        "environment/nvidia_smi.txt",
        "environment/vram_cap_build.txt",
        "environment/runtime_record.csv",
        "environment/python_packages.csv",
        "REPORT.md",
    ):
        lines.append(f"- `{base}/{rel}`")
    lines.append("")
    lines.append("## Not done, by design")
    lines.append("")
    lines.append(
        "No full-corpus embedding, similarity analysis, clustering, quality comparison or model choice; nothing under `legacy/scripts/` or the raw data was touched."
    )
    lines.append("")
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--stage", choices=["prepare", "search", "report", "all"], default="all"
    )
    p.add_argument("--project-dir", default=str(DEFAULT_PROJECT_DIR))
    p.add_argument("--feas-dir", default=str(FEAS_DIR))
    p.add_argument("--hub-dir", default="~/.cache/huggingface/hub")
    p.add_argument("--shim-build-dir", default="~/vram_cap/build")
    p.add_argument("--shim-report-dir", default="~/vram_cap/reports")
    p.add_argument("--ceiling-mib", type=int, default=21504)
    p.add_argument(
        "--n-min",
        type=int,
        default=40960,
        help="max_batch_tokens of the calibration run (TEI's minimum with truncation disabled)",
    )
    p.add_argument(
        "--below-minimum-value",
        type=int,
        default=32768,
        help="a value below TEI's minimum, run once to document the refusal (0 = skip)",
    )
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--growth", type=int, default=2)
    p.add_argument("--calibration-cap-gib", type=float, default=17.0)
    p.add_argument("--n-candidates", type=int, default=2000)
    p.add_argument("--tokenize-chunk", type=int, default=128)
    p.add_argument("--nvml-interval-ms", type=int, default=20)
    p.add_argument("--blocker-gap-ms", type=int, default=30)
    p.add_argument("--attempts", type=int, default=3)
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--startup-timeout-s", type=int, default=600)
    p.add_argument("--request-timeout-s", type=int, default=1800)
    p.add_argument("--settle-s", type=int, default=30)
    p.add_argument("--settle-margin-mib", type=int, default=300)
    p.add_argument("--max-tests", type=int, default=40)
    p.add_argument(
        "--watchdog-grace-s",
        type=float,
        default=3.0,
        help="how long the device may stay over the ceiling after a shim refusal before the container is killed",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    args.project_dir = str(Path(args.project_dir).resolve())
    args.feas_dir = str(Path(args.feas_dir).resolve())
    try:
        if args.stage in ("prepare", "all"):
            stage_prepare(args)
        if args.stage in ("search", "all"):
            stage_search(args)
        if args.stage in ("report", "all"):
            stage_report(args)
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
