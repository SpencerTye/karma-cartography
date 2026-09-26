#!/usr/bin/env python
"""Feasibility test of Qwen3-Embedding-4B and 8B on the longest documents of the corpus.

Counts the tokens of every canonical Phase 1 document, then measures VRAM, RAM and wall-clock time
for embedding the longest one, three and eight documents in single batches (BF16, no truncation)
under a total-VRAM ceiling. Outputs: results tables, environment record and REPORT.md.

Stages (run in the WSL2 venv; see ../README.md):
  tokens          verify the table, check tokenizer identity, count tokens of every document
  bench --model   load one model and run Cases A, B, C (plus the default-cache Case A)
  report          results tables, environment record and REPORT.md
  all             tokens, bench for 4B and 8B in separate processes, report
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
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
FEAS_DIR = SCRIPT_PATH.parents[1]  # feasibility_qwen3_embedding_worst_case/
DEFAULT_PROJECT_DIR = SCRIPT_PATH.parents[4]  # the repository root 

MODELS = {
    "4B": {
        "repo": "Qwen/Qwen3-Embedding-4B",
        "revision": "5cf2132abc99cad020ac570b19d031efec650f2b",
    },
    "8B": {
        "repo": "Qwen/Qwen3-Embedding-8B",
        "revision": "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af",
    },
}
TOKENIZER_FILES = [
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "tokenizer_config.json",
]
TOKENIZER_ONLY_PATTERNS = TOKENIZER_FILES + [
    "config.json",
    "special_tokens_map.json",
    "generation_config.json",
    "modules.json",
    "config_sentence_transformers.json",
    "1_Pooling/config.json",
]
CASES = [("A", 1), ("B", 3), ("C", 8)]
CANONICAL_TEXT_FIELD = "text"
CANONICAL_TEXT_RULE = "comment: body; submission: title + ' ' + selftext (Phase 1 DATA_DICTIONARY.md field 25)"
DECIMAL_CEILING_BYTES = 21_000_000_000
PACKAGE_NAMES = [
    "torch",
    "flash_attn",
    "transformers",
    "tokenizers",
    "accelerate",
    "safetensors",
    "huggingface_hub",
    "pyarrow",
    "numpy",
    "psutil",
    "nvidia-ml-py",
    "pyyaml",
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


def fits_supported_length(n_tokens, limit):
    return int(n_tokens) <= int(limit)


def within_ceiling(peak_bytes, ceiling_bytes):
    return int(peak_bytes) <= int(ceiling_bytes)


def padding_per_document(lengths):
    """Padded length of a batch padded to its longest member, and the padding of each document."""
    lengths = [int(x) for x in lengths]
    padded = max(lengths)
    return padded, [padded - x for x in lengths]


def rank_longest(n_tokens, n_chars, doc_keys, n):
    """Indices of the n longest documents: tokens descending, then characters descending,
    then doc_key ascending (a total order, so the ranking is reproducible)."""
    n_tokens = np.asarray(n_tokens, dtype=np.int64)
    n_chars = np.asarray(n_chars, dtype=np.int64)
    doc_keys = np.asarray(doc_keys).astype(str)
    order = np.lexsort((doc_keys, -n_chars, -n_tokens))
    return [int(i) for i in order[:n]]


def summarise_token_lengths(n_tokens, limit):
    a = np.asarray(n_tokens, dtype=np.int64)
    p = np.percentile(a, [50, 90, 99, 99.9], method="nearest")
    return {
        "n_documents": int(a.size),
        "min": int(a.min()),
        "p50": int(p[0]),
        "p90": int(p[1]),
        "p99": int(p[2]),
        "p99_9": int(p[3]),
        "max": int(a.max()),
        "sum": int(a.sum()),
        "limit": int(limit),
        "n_over_limit": int((a > int(limit)).sum()),
        "n_over_8192": int((a > 8192).sum()),
        "n_over_40960": int((a > 40960).sum()),
    }


class PeakSampler:
    """Samples a value in a background thread and keeps its peak, first and last readings."""

    def __init__(self, sample_fn, interval_s=0.02):
        self._fn = sample_fn
        self._interval = interval_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
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
        if self.first is None:
            self.first = value
        self.last = value
        self.peak = value if self.peak is None else max(self.peak, value)
        self.n_samples += 1

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


def last_token_pool(last_hidden_states, attention_mask):
    """Verbatim from the Qwen3-Embedding model card ("Transformers Usage")."""
    import torch

    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
        ]


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
    return str(value).replace("|", "\\|").replace("\n", " ")


def side_by_side_table(models):
    """Markdown table, one column per model, with the reported rows."""
    keys = list(models.keys())
    lines = [
        "| Measure | "
        + " | ".join(_cell(models[k]["model_name"]) for k in keys)
        + " |",
        "|---|" + "|".join("---" for _ in keys) + "|",
    ]

    def add(label, fn):
        cells = []
        for k in keys:
            try:
                cells.append(_cell(fn(models[k])))
            except Exception:
                cells.append("n/a")
        lines.append("| " + _cell(label) + " | " + " | ".join(cells) + " |")

    base = "baseline_after_load"
    add(
        "Loaded-model VRAM baseline: device total used (MiB)",
        lambda r: mib(r[base]["device_used_bytes"]),
    )
    add(
        "Loaded-model VRAM baseline: allocator allocated (MiB)",
        lambda r: mib(r[base]["allocated_bytes"]),
    )
    add(
        "Loaded-model VRAM baseline: allocator reserved (MiB)",
        lambda r: mib(r[base]["reserved_bytes"]),
    )
    add(
        "Loaded-model VRAM baseline: attributable to the test process (MiB)",
        lambda r: mib(r[base]["process_attributable_bytes"]),
    )
    add(
        "Eight largest documents: token counts, rank 1 to 8",
        lambda r: ", ".join(str(d["n_tokens"]) for d in r["top_documents"]),
    )

    def case_rows(case):
        def get(r):
            return r["cases"][case]

        add(
            f"Case {case}: result",
            lambda r: (
                "success"
                if get(r)["success"]
                else "FAILURE: " + str(get(r)["failure_reason"])
            ),
        )
        add(
            f"Case {case}: exact input token counts, rank order",
            lambda r: ", ".join(str(x) for x in get(r)["token_counts"]),
        )
        add(
            f"Case {case}: input tensor shape (batch, padded length)",
            lambda r: list(get(r)["input_shape"]),
        )
        add(
            f"Case {case}: padding applied per document (tokens)",
            lambda r: ", ".join(str(x) for x in get(r)["padding_per_document"]),
        )
        add(
            f"Case {case}: peak device VRAM, total (MiB)",
            lambda r: mib(get(r)["peak_device_used_bytes"]),
        )
        add(
            f"Case {case}: incremental peak VRAM above the loaded baseline (MiB)",
            lambda r: mib(get(r)["incremental_device_bytes"]),
        )
        add(
            f"Case {case}: peak allocator allocated / reserved (MiB)",
            lambda r: (
                f"{mib(get(r)['peak_allocated_bytes'])} / "
                f"{mib(get(r)['peak_reserved_bytes'])}"
            ),
        )
        add(
            f"Case {case}: peak RAM, process resident set / system used (MiB)",
            lambda r: (
                f"{mib(get(r)['peak_rss_bytes'])} / "
                f"{mib(get(r)['peak_system_used_bytes'])}"
            ),
        )
        add(
            f"Case {case}: wall-clock of the forward pass (s)",
            lambda r: (
                round(float(get(r)["forward_wall_s"]), 3)
                if get(r)["success"]
                else f"failed after {float(get(r)['forward_wall_s']):.2f} s"
            ),
        )
        add(
            f"Case {case}: ran within the VRAM ceiling",
            lambda r: (
                "yes"
                if get(r)["within_ceiling"]
                else "no: " + str(get(r).get("budget_status", ""))
            ),
        )

    for case in ("A", "B", "C"):
        case_rows(case)
    add("Embedding dimension", lambda r: r["embedding_dimension"])
    add("Cache: config.use_cache", lambda r: r["cache"]["config_use_cache"])
    add(
        "Cache: use_cache passed in the measured cases",
        lambda r: r["cache"]["measured_use_cache"],
    )
    add(
        "Cache: past_key_values returned in the measured cases",
        lambda r: r["cache"]["measured_past_key_values"],
    )
    add(
        "Cache: created by a call with the default setting (runtime probe)",
        lambda r: (
            ("yes: " if r["cache"]["default_probe_creates_cache"] else "no: ")
            + str(r["cache"]["default_probe_summary"])
        ),
    )
    add(
        "Attention: implementation (config) / function called / kernels observed",
        lambda r: (
            f"{r['attention']['config_attn_implementation']} / "
            f"{r['attention']['registry_function']} / {r['attention']['kernels_observed']}"
        ),
    )
    add(
        "Ran within the VRAM ceiling in every measured case",
        lambda r: "yes" if r["ceiling"]["all_measured_cases_within_ceiling"] else "no",
    )
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------------
# Runtime helpers
# ----------------------------------------------------------------------------------------------


def utc_now():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, ensure_ascii=False, default=_json_default)
    return path


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


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def remove_outputs(folder, names):
    folder = Path(folder)
    for name in names:
        for candidate in (folder / name, folder / (name + ".manifest.yml")):
            if candidate.exists():
                candidate.unlink()


def log(message):
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


# --- process and device memory -----------------------------------------------------------------


def rss_bytes():
    import psutil

    return int(psutil.Process().memory_info().rss)


def system_used_bytes():
    import psutil

    return int(psutil.virtual_memory().used)


def reset_rss_high_water_mark():
    """Linux: writing 5 to /proc/self/clear_refs resets the peak resident set size (VmHWM)."""
    try:
        with open("/proc/self/clear_refs", "w") as f:
            f.write("5")
        return True
    except Exception:
        return False


def rss_high_water_mark_bytes():
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
    except Exception:
        pass
    return None


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


def cuda_stats():
    import torch

    return {
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def nvidia_smi_text():
    try:
        out = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=60)
        return out.stdout + out.stderr
    except Exception as e:
        return f"nvidia-smi not available: {e!r}"


def environment_record(nvml):
    import torch

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
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device_capability": ".".join(
            str(x) for x in torch.cuda.get_device_capability(0)
        ),
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        "allocator_backend": torch.cuda.get_allocator_backend(),
        "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
    }
    try:
        import psutil

        record["ram_total_bytes"] = int(psutil.virtual_memory().total)
    except Exception:
        pass
    for pkg in package_versions():
        record[f"package_{pkg['name']}"] = pkg["version"]
    return record


# --- transformers internals ---------------------------------------------------------------------


def dtype_kwarg(torch_dtype):
    """transformers renamed torch_dtype to dtype in 4.56; pass the name the installed version uses."""
    import transformers

    parts = transformers.__version__.split(".")
    major, minor = (
        int(parts[0]),
        int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0,
    )
    if (major, minor) >= (4, 56):
        return {"dtype": torch_dtype}
    return {"torch_dtype": torch_dtype}


def wrap_attention_registry(impl):
    """Wrap the attention function registered for `impl` so that calls are counted; returns the
    counter and a function that restores the registry."""
    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    except Exception as e:
        return {"calls": 0, "function": f"registry not available: {e!r}"}, lambda: None
    try:
        original = ALL_ATTENTION_FUNCTIONS[impl]
    except Exception as e:
        return {
            "calls": 0,
            "function": f"'{impl}' not in the registry: {e!r}",
        }, lambda: None
    counter = {"calls": 0, "function": f"{original.__module__}.{original.__name__}"}

    def wrapped(*args, **kwargs):
        counter["calls"] += 1
        return original(*args, **kwargs)

    ALL_ATTENTION_FUNCTIONS[impl] = wrapped

    def restore():
        ALL_ATTENTION_FUNCTIONS[impl] = original

    return counter, restore


def describe_cache(past_key_values):
    """What the runtime returned as past_key_values: type, layers, key tensor shape/dtype, bytes."""
    import torch

    if past_key_values is None:
        return {
            "created": False,
            "type": "None",
            "summary": "None (no cache object returned)",
        }
    info = {"created": True, "type": type(past_key_values).__name__}
    tensors = []
    n_layers = None
    try:
        if hasattr(past_key_values, "layers"):
            layers = list(past_key_values.layers)
            n_layers = len(layers)
            for layer in layers:
                for attr in ("keys", "values"):
                    t = getattr(layer, attr, None)
                    if isinstance(t, torch.Tensor):
                        tensors.append(t)
        elif hasattr(past_key_values, "key_cache"):
            n_layers = len(past_key_values.key_cache)
            for t in list(past_key_values.key_cache) + list(
                past_key_values.value_cache
            ):
                if isinstance(t, torch.Tensor):
                    tensors.append(t)
        elif isinstance(past_key_values, (tuple, list)):
            n_layers = len(past_key_values)
            for layer in past_key_values:
                for t in layer:
                    if isinstance(t, torch.Tensor):
                        tensors.append(t)
    except Exception as e:
        info["error"] = repr(e)
    total_bytes = sum(t.numel() * t.element_size() for t in tensors)
    first = tensors[0] if tensors else None
    info.update(
        {
            "n_layers": n_layers,
            "n_tensors": len(tensors),
            "key_shape": list(first.shape) if first is not None else None,
            "key_dtype": str(first.dtype) if first is not None else None,
            "total_bytes": int(total_bytes),
        }
    )
    info["summary"] = (
        f"{info['type']} with {n_layers} layers, {len(tensors)} tensors, first key tensor shape "
        f"{info['key_shape']} dtype {info['key_dtype']}, {mib(total_bytes)} MiB"
    )
    return info


def profile_kernels(run_fn):
    """Run `run_fn` once under the torch profiler and list the CUDA kernels it launched."""
    import torch

    try:
        from torch.profiler import ProfilerActivity, profile

        torch.cuda.synchronize()
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=False,
            profile_memory=False,
        ) as prof:
            t0 = time.perf_counter()
            run_fn()
            torch.cuda.synchronize()
            t1 = time.perf_counter()
        cuda_rows = []
        for ev in prof.key_averages():
            device_type = getattr(ev, "device_type", None)
            if device_type is None or str(device_type).split(".")[-1] != "CUDA":
                continue
            device_time = getattr(ev, "device_time_total", None)
            if device_time is None:
                device_time = getattr(ev, "cuda_time_total", 0)
            if device_time <= 0:
                continue
            cuda_rows.append(
                {
                    "name": ev.key,
                    "count": int(ev.count),
                    "device_time_ms": round(device_time / 1000, 3),
                }
            )
        cuda_rows.sort(key=lambda r: -r["device_time_ms"])
        markers = ("flash", "fmha", "attn", "attention", "sdpa")
        attention = [
            r for r in cuda_rows if any(m in r["name"].lower() for m in markers)
        ]
        return {
            "available": len(cuda_rows) > 0,
            "wall_s": t1 - t0,
            "n_cuda_kernels": len(cuda_rows),
            "top_kernels": cuda_rows[:15],
            "attention_kernels": attention[:15],
        }
    except Exception as e:
        return {"available": False, "error": repr(e)}


def kernel_names(profile_result):
    if not profile_result or not profile_result.get("available"):
        return "profiler not available"
    names = []
    for row in profile_result.get("attention_kernels", []):
        short = row["name"].split("<")[0].split("(")[0]
        if short not in names:
            names.append(short)
    return "; ".join(names) if names else "no attention-named kernel observed"


# ----------------------------------------------------------------------------------------------
# Data access
# ----------------------------------------------------------------------------------------------


def canonical_paths(project_dir):
    base = (
        Path(project_dir)
        / "pipeline"
        / "phase1_message_event_foundation"
        / "derived"
    )
    parquet = base / "message_events.parquet"
    return parquet, parquet.with_name(parquet.name + ".manifest.yml")


def read_canonical(parquet_path, columns):
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path, columns=columns)
    return table


def results_dir(args):
    return Path(args.feas_dir) / "results"


def environment_dir(args):
    return Path(args.feas_dir) / "environment"


def common_parameters(args):
    return {
        "canonical_text_field": CANONICAL_TEXT_FIELD,
        "canonical_text_rule": CANONICAL_TEXT_RULE,
        "expected_rows": args.expected_rows,
        "supported_input_length_tokens": args.supported_length,
        "vram_ceiling_mib": args.vram_ceiling_mib,
        "vram_ceiling_bytes": args.vram_ceiling_mib * 1048576,
        "decimal_ceiling_bytes": DECIMAL_CEILING_BYTES,
        "top_n": args.top_n,
        "cases": "A: 1 document; B: 3 documents; C: 8 documents; one padded forward pass each",
        "models": {k: f"{v['repo']} @ {v['revision']}" for k, v in MODELS.items()},
        "dtype": "bfloat16",
        "attn_implementation": "flash_attention_2",
        "padding_side": "left",
        "truncation": "none",
        "pooling": "last_token_pool (model card) + L2 normalisation",
        "use_cache_in_measured_cases": False,
        "tokenizer_call": "tokenizer(texts, add_special_tokens=True, padding=<per stage>, truncation=False)",
        "tokenize_batch_size": args.tokenize_batch_size,
        "identity_sample_size": args.sample_size,
        "seed": args.seed,
        "nvml_sample_interval_ms": args.nvml_interval_ms,
    }


# ----------------------------------------------------------------------------------------------
# Stage: tokens
# ----------------------------------------------------------------------------------------------


def stage_tokens(args):
    t_stage = time.perf_counter()
    timings = {}
    out_dir = results_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    remove_outputs(
        out_dir,
        [
            "canonical_input_check.csv",
            "tokenizer_identity.csv",
            "token_counts_all_documents.parquet",
            "token_length_summary.csv",
            "longest_documents_by_model.csv",
            "tokens_stage.json",
        ],
    )
    parquet_path, sidecar_path = canonical_paths(args.project_dir)

    # 1. The canonical table: hash against its sidecar, row count against the expectation.
    log(f"hashing {parquet_path}")
    t0 = time.perf_counter()
    parquet_hash = sha256_file(parquet_path)
    sidecar_hash = read_sidecar_hash(sidecar_path) if sidecar_path.exists() else ""
    timings["hash_canonical_s"] = time.perf_counter() - t0
    log(f"reading {CANONICAL_TEXT_FIELD} of the canonical table")
    t0 = time.perf_counter()
    table = read_canonical(
        parquet_path, ["doc_key", "record_type", CANONICAL_TEXT_FIELD, "text_n_chars"]
    )
    n_rows = table.num_rows
    doc_keys = np.array(table.column("doc_key").to_pylist(), dtype=str)
    record_types = np.array(table.column("record_type").to_pylist(), dtype=str)
    texts = table.column(CANONICAL_TEXT_FIELD).to_pylist()
    text_n_chars = np.array(table.column("text_n_chars").to_pylist(), dtype=np.int64)
    del table
    timings["read_canonical_s"] = time.perf_counter() - t0
    n_null_text = sum(1 for t in texts if t is None)
    input_check = [
        {
            "item": "canonical_parquet",
            "value": rel_path(parquet_path, args.project_dir),
            "detail": f"{parquet_path.stat().st_size} bytes",
        },
        {"item": "sha256_measured", "value": parquet_hash, "detail": ""},
        {
            "item": "sha256_in_phase1_sidecar",
            "value": sidecar_hash,
            "detail": rel_path(sidecar_path, args.project_dir),
        },
        {
            "item": "hash_equal_to_sidecar",
            "value": str(parquet_hash == sidecar_hash),
            "detail": "",
        },
        {
            "item": "rows",
            "value": str(n_rows),
            "detail": f"expected {args.expected_rows}",
        },
        {
            "item": "rows_equal_to_expected",
            "value": str(n_rows == args.expected_rows),
            "detail": "",
        },
        {
            "item": "text_field",
            "value": CANONICAL_TEXT_FIELD,
            "detail": CANONICAL_TEXT_RULE,
        },
        {
            "item": "text_null_count",
            "value": str(n_null_text),
            "detail": "documents with a null text",
        },
        {
            "item": "doc_key_unique",
            "value": str(len(np.unique(doc_keys)) == n_rows),
            "detail": "",
        },
    ]
    write_csv(
        out_dir / "canonical_input_check.csv", input_check, ["item", "value", "detail"]
    )
    if n_rows != args.expected_rows:
        raise SystemExit(
            f"canonical table has {n_rows} rows, expected {args.expected_rows}"
        )
    if n_null_text:
        raise SystemExit(f"{n_null_text} documents have a null {CANONICAL_TEXT_FIELD}")
    log(
        f"canonical table: {n_rows} rows; hash equal to sidecar: {parquet_hash == sidecar_hash}"
    )

    # 2. Tokenizers of both models at the pinned revisions; identity of their files and behaviour.
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    t0 = time.perf_counter()
    snapshots = {}
    tokenizers = {}
    for key, spec in MODELS.items():
        snapshots[key] = Path(
            snapshot_download(
                repo_id=spec["repo"],
                revision=spec["revision"],
                allow_patterns=TOKENIZER_ONLY_PATTERNS,
            )
        )
        tokenizers[key] = AutoTokenizer.from_pretrained(
            spec["repo"], revision=spec["revision"], padding_side="left"
        )
    timings["load_tokenizers_s"] = time.perf_counter() - t0
    identity_rows = []
    for fname in TOKENIZER_FILES:
        values = {}
        sizes = {}
        for key in MODELS:
            p = snapshots[key] / fname
            values[key] = sha256_file(p) if p.exists() else "missing"
            sizes[key] = p.stat().st_size if p.exists() else -1
        identity_rows.append(
            {
                "item": f"file:{fname}",
                "value_4B": values["4B"],
                "value_8B": values["8B"],
                "size_4B": sizes["4B"],
                "size_8B": sizes["8B"],
                "identical": values["4B"] == values["8B"] and values["4B"] != "missing",
            }
        )

    def add_identity(item, v4, v8, identical=None):
        identity_rows.append(
            {
                "item": item,
                "value_4B": v4,
                "value_8B": v8,
                "size_4B": "",
                "size_8B": "",
                "identical": (v4 == v8) if identical is None else identical,
            }
        )

    tok4, tok8 = tokenizers["4B"], tokenizers["8B"]
    add_identity("class", type(tok4).__name__, type(tok8).__name__)
    vocab4, vocab8 = tok4.get_vocab(), tok8.get_vocab()
    add_identity("vocabulary_size", len(vocab4), len(vocab8))
    add_identity("vocabulary_equal", "see identical", "see identical", vocab4 == vocab8)
    added4 = {int(k): str(v) for k, v in tok4.added_tokens_decoder.items()}
    added8 = {int(k): str(v) for k, v in tok8.added_tokens_decoder.items()}
    add_identity("added_tokens_count", len(added4), len(added8))
    add_identity(
        "added_tokens_equal", "see identical", "see identical", added4 == added8
    )
    add_identity(
        "special_tokens_map",
        json.dumps(tok4.special_tokens_map, sort_keys=True),
        json.dumps(tok8.special_tokens_map, sort_keys=True),
    )
    add_identity(
        "pad_token/id",
        f"{tok4.pad_token}/{tok4.pad_token_id}",
        f"{tok8.pad_token}/{tok8.pad_token_id}",
    )
    add_identity(
        "eos_token/id",
        f"{tok4.eos_token}/{tok4.eos_token_id}",
        f"{tok8.eos_token}/{tok8.eos_token_id}",
    )
    add_identity(
        "bos_token/id",
        f"{tok4.bos_token}/{tok4.bos_token_id}",
        f"{tok8.bos_token}/{tok8.bos_token_id}",
    )
    add_identity("model_max_length", tok4.model_max_length, tok8.model_max_length)
    add_identity("padding_side", tok4.padding_side, tok8.padding_side)
    post4 = json.loads(tok4.backend_tokenizer.to_str()).get("post_processor")
    post8 = json.loads(tok8.backend_tokenizer.to_str()).get("post_processor")
    add_identity(
        "post_processor",
        json.dumps(post4, sort_keys=True),
        json.dumps(post8, sort_keys=True),
    )
    probe = "hello world"
    with_special = tok4(probe, add_special_tokens=True)["input_ids"]
    without_special = tok4(probe, add_special_tokens=False)["input_ids"]
    if with_special[: len(without_special)] == without_special:
        appended = with_special[len(without_special) :]
        special_note = (
            f"appended {tok4.convert_ids_to_tokens(appended)} (ids {appended})"
            if appended
            else "none"
        )
    else:
        special_note = f"prefix differs: with {with_special}, without {without_special}"
    add_identity(
        "special_tokens_added_by_default",
        special_note,
        (
            lambda w, wo: (
                f"appended {tok8.convert_ids_to_tokens(w[len(wo) :])} (ids {w[len(wo) :]})"
                if w[: len(wo)] == wo
                else "prefix differs"
            )
        )(
            tok8(probe, add_special_tokens=True)["input_ids"],
            tok8(probe, add_special_tokens=False)["input_ids"],
        ),
    )
    rng = np.random.default_rng(args.seed)
    sample_idx = rng.choice(n_rows, size=min(args.sample_size, n_rows), replace=False)
    sample_texts = [texts[int(i)] for i in sample_idx]
    t0 = time.perf_counter()
    ids4 = tok4(sample_texts, add_special_tokens=True, padding=False, truncation=False)[
        "input_ids"
    ]
    ids8 = tok8(sample_texts, add_special_tokens=True, padding=False, truncation=False)[
        "input_ids"
    ]
    timings["identity_sample_encode_s"] = time.perf_counter() - t0
    add_identity(
        f"sample_encoding_equal_{len(sample_texts)}_documents_seed_{args.seed}",
        "see identical",
        "see identical",
        ids4 == ids8,
    )
    files_identical = all(
        r["identical"] for r in identity_rows if r["item"].startswith("file:")
    )
    tokenizers_identical = all(bool(r["identical"]) for r in identity_rows)
    log(
        f"tokenizer files identical: {files_identical}; all identity checks: {tokenizers_identical}"
    )

    # 3. Token counts of every document (one tokenizer if identical, both otherwise).
    def count_all(tok, label):
        counts = np.empty(n_rows, dtype=np.int64)
        t_start = time.perf_counter()
        batch = args.tokenize_batch_size
        for start in range(0, n_rows, batch):
            enc = tok(
                texts[start : start + batch],
                add_special_tokens=True,
                padding=False,
                truncation=False,
                return_attention_mask=False,
            )
            counts[start : start + batch] = [len(ids) for ids in enc["input_ids"]]
            if (start // batch) % 25 == 0:
                log(
                    f"  {label}: {min(start + batch, n_rows)}/{n_rows} documents tokenised"
                )
        return counts, time.perf_counter() - t_start

    log("tokenising every document with the Qwen3-Embedding-4B tokenizer")
    n_tokens_4b, timings["tokenize_all_4B_s"] = count_all(tok4, "4B")
    if tokenizers_identical:
        n_tokens_8b = n_tokens_4b
        timings["tokenize_all_8B_s"] = 0.0
        counts_note = "tokenizer files and behaviour identical: the 4B counts apply to the 8B model"
    else:
        log(
            "tokenizers differ: tokenising every document with the 8B tokenizer as well"
        )
        n_tokens_8b, timings["tokenize_all_8B_s"] = count_all(tok8, "8B")
        counts_note = "tokenizers differ: counted separately for each model"
    n_chars = np.fromiter((len(t) for t in texts), dtype=np.int64, count=n_rows)
    n_chars_equal = int((n_chars == text_n_chars).sum())

    # 4. Ranking, and the eight longest documents under both tokenizers.
    per_model = {"4B": n_tokens_4b, "8B": n_tokens_8b}
    top = {}
    longest_rows = []
    for key in MODELS:
        counts = per_model[key]
        idx = rank_longest(counts, n_chars, doc_keys, args.top_n)
        top_texts = [texts[i] for i in idx]
        ids_top4 = tok4(
            top_texts, add_special_tokens=True, padding=False, truncation=False
        )["input_ids"]
        ids_top8 = tok8(
            top_texts, add_special_tokens=True, padding=False, truncation=False
        )["input_ids"]
        docs = []
        for rank, i in enumerate(idx, start=1):
            docs.append(
                {
                    "model": MODELS[key]["repo"].split("/")[-1],
                    "rank": rank,
                    "doc_key": str(doc_keys[i]),
                    "record_type": str(record_types[i]),
                    "n_tokens": int(counts[i]),
                    "n_chars": int(n_chars[i]),
                    "supported_input_length": args.supported_length,
                    "fits_supported_input_length": fits_supported_length(
                        counts[i], args.supported_length
                    ),
                    "max_position_embeddings": 40960,
                    "fits_max_position_embeddings": fits_supported_length(
                        counts[i], 40960
                    ),
                    "row_index": int(i),
                }
            )
        top[key] = {
            "documents": docs,
            "top_ids_equal_under_both_tokenizers": ids_top4 == ids_top8,
            "top_lengths_4B_tokenizer": [len(x) for x in ids_top4],
            "top_lengths_8B_tokenizer": [len(x) for x in ids_top8],
        }
        longest_rows.extend(docs)
    write_csv(out_dir / "longest_documents_by_model.csv", longest_rows)

    # 5. Tables.
    summary_rows = []
    for key in MODELS:
        counts = per_model[key]
        for population, mask in (
            ("all", np.ones(n_rows, dtype=bool)),
            ("comment", record_types == "comment"),
            ("submission", record_types == "submission"),
        ):
            s = summarise_token_lengths(counts[mask], args.supported_length)
            summary_rows.append(
                {
                    "model": MODELS[key]["repo"].split("/")[-1],
                    "population": population,
                    **s,
                }
            )
    write_csv(out_dir / "token_length_summary.csv", summary_rows)
    write_csv(
        out_dir / "tokenizer_identity.csv",
        identity_rows,
        ["item", "value_4B", "value_8B", "size_4B", "size_8B", "identical"],
    )

    import pyarrow as pa
    import pyarrow.parquet as pq

    columns = {
        "doc_key": pa.array(doc_keys.tolist(), type=pa.string()),
        "record_type": pa.array(record_types.tolist(), type=pa.string()),
        "n_chars": pa.array(n_chars, type=pa.int64()),
        "n_tokens_4B": pa.array(n_tokens_4b, type=pa.int64()),
        "n_tokens_8B": pa.array(n_tokens_8b, type=pa.int64()),
    }
    pq.write_table(
        pa.table(columns),
        out_dir / "token_counts_all_documents.parquet",
        compression="zstd",
        compression_level=3,
    )

    timings["stage_total_s"] = time.perf_counter() - t_stage
    stage = {
        "stage": "tokens",
        "timestamp": utc_now(),
        "canonical": {
            "path": rel_path(parquet_path, args.project_dir),
            "sha256": parquet_hash,
            "sidecar_sha256": sidecar_hash,
            "hash_equal_to_sidecar": parquet_hash == sidecar_hash,
            "rows": int(n_rows),
            "expected_rows": args.expected_rows,
            "text_field": CANONICAL_TEXT_FIELD,
            "text_rule": CANONICAL_TEXT_RULE,
            "n_chars_equal_to_phase1_text_n_chars": n_chars_equal,
        },
        "tokenizer": {
            "snapshots": {k: str(v) for k, v in snapshots.items()},
            "files": TOKENIZER_FILES,
            "files_identical": files_identical,
            "all_identity_checks_passed": tokenizers_identical,
            "identity_rows": identity_rows,
            "counts_note": counts_note,
            "call": "tokenizer(texts, add_special_tokens=True, padding=False, truncation=False)",
            "special_tokens_added_by_default": special_note,
            "versions": {
                p["name"]: p["version"]
                for p in package_versions(["transformers", "tokenizers"])
            },
        },
        "summary": summary_rows,
        "top_documents": top,
        "timings": timings,
    }
    write_json(out_dir / "tokens_stage.json", stage)

    # 6. Sidecars.
    sidecars = Sidecars(
        args.project_dir, common_parameters(args), "Phase 2 feasibility, tokens stage."
    )
    canonical_input = sidecars.input_entry(
        parquet_path,
        "canonical Phase 1 table, read-only",
        parquet_hash,
        with_shape=True,
    )
    tokenizer_inputs = [
        {
            "path": str(snapshots[k] / f),
            "hash": sha256_file(snapshots[k] / f),
            "format": "other",
            "role": f"tokenizer file of {MODELS[k]['repo']} @ {MODELS[k]['revision']}",
        }
        for k in MODELS
        for f in TOKENIZER_FILES
        if (snapshots[k] / f).exists()
    ]
    for name, note in (
        (
            "canonical_input_check.csv",
            "hash and row-count check of the canonical table",
        ),
        (
            "tokenizer_identity.csv",
            "identity of the two tokenizers: files, vocabulary, special tokens, post-processor, sample encodings",
        ),
        (
            "token_counts_all_documents.parquet",
            "exact token count of every document under each model's tokenizer (identical tokenizers give identical columns)",
        ),
        (
            "token_length_summary.csv",
            "token-length distribution by model and record type",
        ),
        (
            "longest_documents_by_model.csv",
            "the longest documents by token count for each model",
        ),
        ("tokens_stage.json", "every result of the tokens stage"),
    ):
        sidecars.write(out_dir / name, [canonical_input] + tokenizer_inputs, notes=note)
    log(
        f"tokens stage done in {timings['stage_total_s']:.1f} s; longest document {top['4B']['documents'][0]['doc_key']} "
        f"with {top['4B']['documents'][0]['n_tokens']} tokens; n_chars equal to Phase 1 for {n_chars_equal}/{n_rows}"
    )
    return stage


# ----------------------------------------------------------------------------------------------
# Stage: bench (one model per process)
# ----------------------------------------------------------------------------------------------


def load_top_texts(args, tokens_stage, model_key):
    docs = tokens_stage["top_documents"][model_key]["documents"]
    parquet_path, _ = canonical_paths(args.project_dir)
    table = read_canonical(parquet_path, ["doc_key", CANONICAL_TEXT_FIELD])
    keys = np.array(table.column("doc_key").to_pylist(), dtype=str)
    wanted = [d["doc_key"] for d in docs]
    positions = {k: int(np.flatnonzero(keys == k)[0]) for k in wanted}
    text_col = table.column(CANONICAL_TEXT_FIELD)
    texts = [text_col[positions[k]].as_py() for k in wanted]
    del table
    return docs, texts


def run_case(ctx, case_name, docs, texts, use_cache_passed, label=None):
    """One case: tokenize, one forward pass, pool, normalise; every measurement; failures kept."""
    import torch
    import torch.nn.functional as F

    model, tok, nvml, args = ctx["model"], ctx["tokenizer"], ctx["nvml"], ctx["args"]
    baseline = ctx["baseline_after_load"]
    ceiling_bytes = args.vram_ceiling_mib * 1048576
    OOM = getattr(torch, "OutOfMemoryError", torch.cuda.OutOfMemoryError)
    rec = {
        "case": case_name,
        "label": label or case_name,
        "n_documents": len(texts),
        "doc_keys": [d["doc_key"] for d in docs],
        "record_types": [d["record_type"] for d in docs],
        "n_chars": [d["n_chars"] for d in docs],
        "expected_token_counts": [d["n_tokens"] for d in docs],
        "use_cache_passed": "not passed (configuration default)"
        if use_cache_passed is None
        else use_cache_passed,
    }
    t0 = time.perf_counter()
    enc = tok(texts, padding=True, truncation=False, return_tensors="pt")
    t_tok = time.perf_counter() - t0
    lengths = [int(x) for x in enc["attention_mask"].sum(dim=1).tolist()]
    padded_len, pads = padding_per_document(lengths)
    rec.update(
        {
            "token_counts": lengths,
            "token_counts_match_tokens_stage": lengths == rec["expected_token_counts"],
            "input_shape": list(enc["input_ids"].shape),
            "input_dtype": str(enc["input_ids"].dtype),
            "padded_length": padded_len,
            "padding_per_document": pads,
            "total_pad_tokens": int(sum(pads)),
            "padding_side": tok.padding_side,
            "tokenize_wall_s": t_tok,
        }
    )
    over = [
        (d, n)
        for d, n in zip(rec["doc_keys"], lengths)
        if not fits_supported_length(n, args.supported_length)
    ]
    if over:
        rec.update(
            {
                "success": False,
                "skipped_before_inference": True,
                "failure_reason": "document(s) exceed the supported input length "
                + f"{args.supported_length}: "
                + "; ".join(f"{d} has {n} tokens" for d, n in over),
                "forward_wall_s": 0.0,
                "h2d_wall_s": 0.0,
                "total_wall_s": t_tok,
                "peak_device_used_bytes": nvml.used(),
                "incremental_device_bytes": 0,
                "peak_allocated_bytes": baseline["allocated_bytes"],
                "incremental_allocated_bytes": 0,
                "peak_reserved_bytes": baseline["reserved_bytes"],
                "peak_rss_bytes": rss_bytes(),
                "peak_system_used_bytes": system_used_bytes(),
                "within_ceiling": False,
                "within_decimal_ceiling": False,
                "device_peak_under_ceiling": True,
                "device_peak_under_decimal_ceiling": True,
                "budget_status": "not run: a document exceeds the supported input length",
                "embedding_dimension": None,
            }
        )
        log(f"  case {rec['label']}: {rec['failure_reason']}")
        return rec, None

    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    rec["rss_high_water_mark_reset"] = reset_rss_high_water_mark()
    rec["before"] = {
        **cuda_stats(),
        "device_used_bytes": nvml.used(),
        "rss_bytes": rss_bytes(),
    }
    interval = args.nvml_interval_ms / 1000.0
    nvml_sampler = PeakSampler(nvml.used, interval).start()
    rss_sampler = PeakSampler(rss_bytes, interval).start()
    sys_sampler = PeakSampler(system_used_bytes, interval).start()
    ctx["attention_counter"]["calls"] = 0
    embeddings = None
    t_h2d = t_fwd = 0.0
    try:
        t1 = time.perf_counter()
        input_ids = enc["input_ids"].to("cuda:0")
        attention_mask = enc["attention_mask"].to("cuda:0")
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        t_h2d = t2 - t1
        kwargs = {} if use_cache_passed is None else {"use_cache": use_cache_passed}
        with torch.inference_mode():
            out = model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
            pooled = last_token_pool(out.last_hidden_state, attention_mask)
            normalised = F.normalize(pooled, p=2, dim=1)
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        t_fwd = t3 - t2
        rec.update(
            {
                "success": True,
                "failure_reason": "",
                "last_hidden_state_shape": list(out.last_hidden_state.shape),
                "last_hidden_state_dtype": str(out.last_hidden_state.dtype),
                "embedding_shape": list(normalised.shape),
                "embedding_dtype": str(normalised.dtype),
                "embedding_dimension": int(normalised.shape[1]),
                "embedding_norms": [
                    round(float(x), 6) for x in normalised.float().norm(dim=1).tolist()
                ],
                "embedding_first_values": [
                    [round(float(v), 6) for v in row[:4]]
                    for row in normalised.float().tolist()
                ],
                "past_key_values": describe_cache(
                    getattr(out, "past_key_values", None)
                ),
            }
        )
        embeddings = normalised.float().cpu().numpy()
        del out, pooled, normalised
    except OOM as e:
        t3 = time.perf_counter()
        t_fwd = t3 - (t2 if t_h2d else t1)
        rec.update(
            {
                "success": False,
                "failure_reason": "torch.OutOfMemoryError at the allocator cap: "
                + str(e).replace("\n", " ")[:800],
                "embedding_dimension": None,
            }
        )
    except RuntimeError as e:
        t3 = time.perf_counter()
        t_fwd = t3 - (t2 if t_h2d else t1)
        rec.update(
            {
                "success": False,
                "failure_reason": "RuntimeError: " + str(e).replace("\n", " ")[:800],
                "embedding_dimension": None,
            }
        )
    finally:
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        stats = cuda_stats()
        nvml_res = nvml_sampler.stop()
        rss_res = rss_sampler.stop()
        sys_res = sys_sampler.stop()
        hwm = rss_high_water_mark_bytes()
    input_ids = attention_mask = (
        None  # release the device copies before the next measurement
    )
    gc.collect()
    torch.cuda.empty_cache()
    after_device_used = nvml.used()
    peak_device = max(int(nvml_res["peak"] or 0), after_device_used)
    peak_rss = max(int(rss_res["peak"] or 0), int(hwm or 0))
    rec.update(
        {
            "h2d_wall_s": t_h2d,
            "forward_wall_s": t_fwd,
            "total_wall_s": t_tok + t_h2d + t_fwd,
            "attention_function_calls": ctx["attention_counter"]["calls"],
            "peak_allocated_bytes": stats["max_allocated_bytes"],
            "peak_reserved_bytes": stats["max_reserved_bytes"],
            "peak_device_used_bytes": peak_device,
            "device_used_after_case_bytes": after_device_used,
            "nvml_samples": nvml_res["n_samples"],
            "incremental_device_bytes": peak_device - baseline["device_used_bytes"],
            "incremental_allocated_bytes": stats["max_allocated_bytes"]
            - baseline["allocated_bytes"],
            "incremental_reserved_bytes": stats["max_reserved_bytes"]
            - baseline["reserved_bytes"],
            "peak_rss_bytes": peak_rss,
            "peak_rss_high_water_mark_bytes": hwm,
            "peak_rss_sampled_bytes": rss_res["peak"],
            "peak_system_used_bytes": sys_res["peak"],
            # Enforcement evidence: the device total never went above the ceiling.
            "device_peak_under_ceiling": within_ceiling(peak_device, ceiling_bytes),
            "device_peak_under_decimal_ceiling": within_ceiling(
                peak_device, DECIMAL_CEILING_BYTES
            ),
            # The question: did the case run to completion within the budget.
            "within_ceiling": bool(rec["success"])
            and within_ceiling(peak_device, ceiling_bytes),
            "within_decimal_ceiling": bool(rec["success"])
            and within_ceiling(peak_device, DECIMAL_CEILING_BYTES),
        }
    )
    if rec["success"]:
        rec["budget_status"] = (
            "ran within the ceiling"
            if rec["device_peak_under_ceiling"]
            else "ran, but the device total exceeded the ceiling (memory held by other "
            "processes grew during the case)"
        )
    elif "OutOfMemoryError" in rec["failure_reason"]:
        rec["budget_status"] = (
            "exceeded the permitted VRAM budget: the allocation that would have gone past "
            f"the ceiling was refused at the allocator cap (device peak {mib(peak_device)} MiB)"
        )
    else:
        rec["budget_status"] = "failed: " + rec["failure_reason"][:120]
    status = "success" if rec["success"] else "FAILURE"
    log(
        f"  case {rec['label']}: {status}; shape {rec['input_shape']}; forward {t_fwd:.2f} s; "
        f"peak device {mib(peak_device)} MiB (+{mib(rec['incremental_device_bytes'])}); "
        f"peak allocated {mib(stats['max_allocated_bytes'])} MiB; within ceiling {rec['within_ceiling']}"
    )
    if not rec["success"]:
        log(f"    {rec['failure_reason']}")
    # Profiler replay (kernel evidence); its own timing, never the case's wall-clock.
    if rec["success"] and not args.no_profiler_replay:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        def replay():
            ids = enc["input_ids"].to("cuda:0")
            mask = enc["attention_mask"].to("cuda:0")
            with torch.inference_mode():
                o = model(input_ids=ids, attention_mask=mask, **kwargs)
                p = F.normalize(last_token_pool(o.last_hidden_state, mask), p=2, dim=1)
            return p

        prof = profile_kernels(replay)
        prof["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
        rec["profiler_replay"] = prof
        gc.collect()
        torch.cuda.empty_cache()
        log(
            f"    replay under profiler: {kernel_names(prof)}; wall {prof.get('wall_s', 0):.2f} s"
        )
    return rec, embeddings


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def stage_bench(args, model_key):
    import torch

    spec = MODELS[model_key]
    model_name = spec["repo"].split("/")[-1]
    t_stage = time.perf_counter()
    timings = {}
    out_dir = results_dir(args)
    env_dir = environment_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    env_dir.mkdir(parents=True, exist_ok=True)
    remove_outputs(out_dir, [f"bench_{model_key}.json", f"cases_{model_key}.csv"])
    remove_outputs(env_dir, [f"nvidia_smi_{model_key}.txt"])
    tokens_stage = read_json(out_dir / "tokens_stage.json")
    docs, texts = load_top_texts(args, tokens_stage, model_key)
    ceiling_bytes = args.vram_ceiling_mib * 1048576

    result = {
        "stage": "bench",
        "model_key": model_key,
        "model_name": model_name,
        "repo": spec["repo"],
        "revision_requested": spec["revision"],
        "timestamp": utc_now(),
        "top_documents": docs,
        "cases": {},
        "timings": timings,
    }
    nvml = Nvml(0)
    result["environment"] = environment_record(nvml)
    smi_path = env_dir / f"nvidia_smi_{model_key}.txt"
    smi_path.write_text(nvidia_smi_text(), encoding="utf-8")
    device_total = nvml.total()
    used_before_context = nvml.used()
    torch.cuda.init()
    probe_tensor = torch.zeros(1, device="cuda:0")
    torch.cuda.synchronize()
    del probe_tensor
    used_after_context = nvml.used()
    cap_bytes = ceiling_bytes - used_after_context
    ceiling = {
        "ceiling_mib": args.vram_ceiling_mib,
        "ceiling_bytes": ceiling_bytes,
        "decimal_ceiling_bytes": DECIMAL_CEILING_BYTES,
        "device_total_bytes": device_total,
        "device_used_before_context_bytes": used_before_context,
        "device_used_after_context_bytes": used_after_context,
        "cuda_context_bytes": used_after_context - used_before_context,
        "allocator_cap_bytes": cap_bytes,
        "allocator_fraction": cap_bytes / device_total if cap_bytes > 0 else 0.0,
        "enforcement": "torch.cuda.set_per_process_memory_fraction(cap / total) after the CUDA context exists; "
        "NVML device total sampled during every case; a case whose allocation is refused at the cap is "
        "recorded as exceeding the budget",
    }
    result["ceiling"] = ceiling
    if cap_bytes <= 0:
        result["status"] = (
            f"no room under the ceiling before loading: device used {mib(used_after_context)} MiB"
        )
        result["ceiling"]["all_measured_cases_within_ceiling"] = False
        write_json(out_dir / f"bench_{model_key}.json", result)
        nvml.shutdown()
        raise SystemExit(result["status"])
    torch.cuda.set_per_process_memory_fraction(ceiling["allocator_fraction"], 0)
    log(
        f"{model_name}: device used before/after context {mib(used_before_context)}/{mib(used_after_context)} MiB; "
        f"ceiling {args.vram_ceiling_mib} MiB; allocator cap {mib(cap_bytes)} MiB (fraction {ceiling['allocator_fraction']:.4f})"
    )

    # Load
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        spec["repo"], revision=spec["revision"], padding_side="left"
    )
    torch.cuda.reset_peak_memory_stats()
    reset_rss_high_water_mark()
    rss_sampler = PeakSampler(rss_bytes, args.nvml_interval_ms / 1000.0).start()
    nvml_sampler = PeakSampler(nvml.used, args.nvml_interval_ms / 1000.0).start()
    t0 = time.perf_counter()
    load_error = None
    try:
        model, loading_info = AutoModel.from_pretrained(
            spec["repo"],
            revision=spec["revision"],
            attn_implementation="flash_attention_2",
            device_map={"": 0},
            output_loading_info=True,
            **dtype_kwarg(torch.bfloat16),
        )
        model.eval()
        torch.cuda.synchronize()
    except Exception as e:
        load_error = f"{type(e).__name__}: {str(e)[:600]}"
        traceback.print_exc()
    timings["load_s"] = time.perf_counter() - t0
    rss_load = rss_sampler.stop()
    nvml_load = nvml_sampler.stop()
    if load_error:
        result["status"] = "model load failed: " + load_error
        result["ceiling"]["all_measured_cases_within_ceiling"] = False
        result["load"] = {
            "peak_rss_bytes": rss_load["peak"],
            "peak_device_used_bytes": nvml_load["peak"],
            **cuda_stats(),
        }
        write_json(out_dir / f"bench_{model_key}.json", result)
        nvml.shutdown()
        raise SystemExit(result["status"])
    stats = cuda_stats()
    baseline = {
        "allocated_bytes": stats["allocated_bytes"],
        "reserved_bytes": stats["reserved_bytes"],
        "device_used_bytes": nvml.used(),
        "process_attributable_bytes": nvml.used() - used_before_context,
        "load_wall_s": timings["load_s"],
        "load_peak_rss_bytes": max(
            int(rss_load["peak"] or 0), int(rss_high_water_mark_bytes() or 0)
        ),
        "load_peak_device_used_bytes": nvml_load["peak"],
        "load_peak_allocated_bytes": stats["max_allocated_bytes"],
    }
    result["baseline_after_load"] = baseline
    from collections import Counter

    dtype_counts = Counter(str(p.dtype) for p in model.parameters())
    n_params = sum(p.numel() for p in model.parameters())
    config = model.config
    result["model_facts"] = {
        "class": type(model).__name__,
        "dtype": str(model.dtype),
        "parameter_dtypes": dict(dtype_counts),
        "n_parameters": int(n_params),
        "device": str(next(model.parameters()).device),
        "revision_loaded": getattr(config, "_commit_hash", None) or spec["revision"],
        "tokenizer_revision_loaded": getattr(tok, "_commit_hash", None)
        or spec["revision"],
        "tokenizer_class": type(tok).__name__,
        "tokenizer_padding_side": tok.padding_side,
        "attn_implementation": getattr(config, "_attn_implementation", None),
        "config_use_cache": bool(getattr(config, "use_cache", None)),
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "head_dim": getattr(config, "head_dim", None),
        "hidden_size": config.hidden_size,
        "intermediate_size": config.intermediate_size,
        "max_position_embeddings": config.max_position_embeddings,
        "vocab_size": config.vocab_size,
        "rope_theta": getattr(config, "rope_theta", None),
        "loading_missing_keys": len(loading_info.get("missing_keys", [])),
        "loading_unexpected_keys": len(loading_info.get("unexpected_keys", [])),
        "loading_unexpected_keys_list": list(loading_info.get("unexpected_keys", []))[
            :10
        ],
        "forward_use_cache_default": _forward_default(model, "use_cache"),
    }
    log(
        f"{model_name} loaded in {timings['load_s']:.1f} s: {model.dtype}, {n_params} parameters, attention "
        f"{result['model_facts']['attn_implementation']}; device used {mib(baseline['device_used_bytes'])} MiB "
        f"(allocated {mib(baseline['allocated_bytes'])}, reserved {mib(baseline['reserved_bytes'])})"
    )

    counter, restore = wrap_attention_registry(
        result["model_facts"]["attn_implementation"]
    )
    ctx = {
        "model": model,
        "tokenizer": tok,
        "nvml": nvml,
        "args": args,
        "baseline_after_load": baseline,
        "attention_counter": counter,
    }

    # Cache probe on a short input: the configuration default, then use_cache=False.
    probe_text = " ".join(["The quick brown fox jumps over the lazy dog."] * 6)
    enc = tok(probe_text, return_tensors="pt")
    ids = enc["input_ids"].to("cuda:0")
    mask = enc["attention_mask"].to("cuda:0")
    torch.cuda.reset_peak_memory_stats()
    counter["calls"] = 0
    with torch.inference_mode():
        out_default = model(input_ids=ids, attention_mask=mask)
    default_probe = describe_cache(getattr(out_default, "past_key_values", None))
    default_probe["attention_function_calls"] = counter["calls"]
    default_probe["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
    del out_default
    torch.cuda.reset_peak_memory_stats()
    counter["calls"] = 0
    with torch.inference_mode():
        out_off = model(input_ids=ids, attention_mask=mask, use_cache=False)
    off_probe = describe_cache(getattr(out_off, "past_key_values", None))
    off_probe["attention_function_calls"] = counter["calls"]
    off_probe["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
    del out_off, ids, mask
    torch.cuda.empty_cache()
    result["cache_probe"] = {
        "probe_tokens": int(enc["input_ids"].shape[1]),
        "default_call": default_probe,
        "use_cache_false_call": off_probe,
    }
    log(
        f"  cache probe ({enc['input_ids'].shape[1]} tokens): default call -> {default_probe['summary']}; use_cache=False -> {off_probe['summary']}"
    )

    # Warm-up (16 tokens) so that Case A does not carry one-time initialisation.
    warm = tok("warm-up " * 8, return_tensors="pt")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    with torch.inference_mode():
        _ = model(
            input_ids=warm["input_ids"].to("cuda:0"),
            attention_mask=warm["attention_mask"].to("cuda:0"),
            use_cache=False,
        )
    torch.cuda.synchronize()
    timings["warmup_s"] = time.perf_counter() - t0
    del _
    torch.cuda.empty_cache()
    result["warmup"] = {
        "tokens": int(warm["input_ids"].shape[1]),
        "wall_s": timings["warmup_s"],
        **cuda_stats(),
        "device_used_bytes": nvml.used(),
    }

    # The three cases.
    embeddings = {}
    for case_name, k in CASES:
        t0 = time.perf_counter()
        rec, emb = run_case(ctx, case_name, docs[:k], texts[:k], use_cache_passed=False)
        timings[f"case_{case_name}_s"] = time.perf_counter() - t0
        result["cases"][case_name] = rec
        if emb is not None:
            embeddings[case_name] = emb

    # Supplementary: Case A with the configuration default (whatever the runtime does with the cache).
    t0 = time.perf_counter()
    rec, emb = run_case(
        ctx, "A", docs[:1], texts[:1], use_cache_passed=None, label="A_default_cache"
    )
    timings["case_A_default_cache_s"] = time.perf_counter() - t0
    result["supplementary"] = {"A_default_cache": rec}
    if emb is not None:
        embeddings["A_default_cache"] = emb

    # Agreement of the same document across cases (left padding + varlen attention should agree).
    consistency = []
    pairs = [
        ("A", "B", 0),
        ("A", "C", 0),
        ("B", "C", 0),
        ("B", "C", 1),
        ("B", "C", 2),
        ("A", "A_default_cache", 0),
    ]
    for x, y, i in pairs:
        if (
            x in embeddings
            and y in embeddings
            and i < len(embeddings[x])
            and i < len(embeddings[y])
        ):
            consistency.append(
                {
                    "model": model_name,
                    "case_x": x,
                    "case_y": y,
                    "document_rank": i + 1,
                    "doc_key": docs[i]["doc_key"],
                    "cosine": round(cosine(embeddings[x][i], embeddings[y][i]), 6),
                }
            )
    result["cross_case_consistency"] = consistency

    restore()
    measured = [result["cases"][c] for c, _ in CASES]
    result["ceiling"]["all_measured_cases_within_ceiling"] = all(
        c["within_ceiling"] for c in measured
    )
    result["ceiling"]["all_measured_cases_within_decimal_ceiling"] = all(
        c["within_decimal_ceiling"] for c in measured
    )
    result["ceiling"]["device_peak_under_ceiling_all_cases"] = all(
        c["device_peak_under_ceiling"] for c in measured
    )
    result["ceiling"]["device_peak_under_decimal_ceiling_all_cases"] = all(
        c["device_peak_under_decimal_ceiling"] for c in measured
    )
    result["embedding_dimension"] = next(
        (c["embedding_dimension"] for c in measured if c.get("embedding_dimension")),
        config.hidden_size,
    )
    result["cache"] = {
        "config_use_cache": result["model_facts"]["config_use_cache"],
        "forward_use_cache_default": result["model_facts"]["forward_use_cache_default"],
        "measured_use_cache": False,
        "measured_past_key_values": "; ".join(
            f"{c}: {result['cases'][c].get('past_key_values', {}).get('summary', 'n/a')}"
            for c, _ in CASES
        ),
        "default_probe_creates_cache": bool(default_probe["created"]),
        "default_probe_summary": default_probe["summary"]
        + f" (probe of {result['cache_probe']['probe_tokens']} tokens); supplementary Case A with the default: "
        + str(
            rec.get("past_key_values", {}).get(
                "summary", rec.get("failure_reason", "n/a")
            )
        ),
    }
    replays = [
        result["cases"][c].get("profiler_replay")
        for c, _ in CASES
        if result["cases"][c].get("profiler_replay")
    ]
    observed = []
    for r in replays:
        for name in kernel_names(r).split("; "):
            if name not in observed:
                observed.append(name)
    result["attention"] = {
        "config_attn_implementation": result["model_facts"]["attn_implementation"],
        "registry_function": counter["function"],
        "calls_per_forward_case_A": result["cases"]["A"].get(
            "attention_function_calls"
        ),
        "kernels_observed": "; ".join(observed) if observed else "no profiler replay",
    }

    # Unload.
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    result["unload"] = {"device_used_bytes": nvml.used(), **cuda_stats()}
    nvml.shutdown()
    timings["stage_total_s"] = time.perf_counter() - t_stage
    result["status"] = "completed"

    # Tables and sidecars.
    case_rows = [
        case_row(result, rec)
        for rec in measured + [result["supplementary"]["A_default_cache"]]
    ]
    write_csv(out_dir / f"cases_{model_key}.csv", case_rows)
    write_json(out_dir / f"bench_{model_key}.json", result)
    sidecars = Sidecars(
        args.project_dir,
        common_parameters(args),
        f"Phase 2 feasibility, bench stage, {model_name}.",
    )
    parquet_path, _ = canonical_paths(args.project_dir)
    inputs = [
        sidecars.input_entry(
            parquet_path,
            "canonical Phase 1 table, read-only",
            tokens_stage["canonical"]["sha256"],
        ),
        sidecars.input_entry(
            out_dir / "tokens_stage.json",
            "tokens stage result (which documents are the longest)",
        ),
    ]
    sidecars.write(
        out_dir / f"bench_{model_key}.json",
        inputs,
        notes=f"every measurement of the bench stage for {model_name}",
    )
    sidecars.write(
        out_dir / f"cases_{model_key}.csv",
        inputs,
        notes=f"one row per case for {model_name}",
    )
    sidecars.write(
        smi_path, [], notes=f"nvidia-smi output captured before loading {model_name}"
    )
    log(f"{model_name}: bench stage done in {timings['stage_total_s']:.1f} s")
    return result


def _forward_default(model, name):
    import inspect

    try:
        param = inspect.signature(model.forward).parameters.get(name)
        return "absent" if param is None else repr(param.default)
    except Exception as e:
        return f"unknown ({e!r})"


def case_row(result, rec):
    base = result["baseline_after_load"]
    return {
        "model": result["model_name"],
        "case": rec["label"],
        "success": rec["success"],
        "failure_reason": rec.get("failure_reason", ""),
        "n_documents": rec["n_documents"],
        "doc_keys": ";".join(rec["doc_keys"]),
        "token_counts": ";".join(str(x) for x in rec["token_counts"]),
        "token_counts_match_tokens_stage": rec["token_counts_match_tokens_stage"],
        "input_shape": "x".join(str(x) for x in rec["input_shape"]),
        "padded_length": rec["padded_length"],
        "padding_per_document": ";".join(str(x) for x in rec["padding_per_document"]),
        "padding_side": rec["padding_side"],
        "use_cache_passed": rec["use_cache_passed"],
        "tokenize_wall_s": round(rec["tokenize_wall_s"], 4),
        "h2d_wall_s": round(rec["h2d_wall_s"], 4),
        "forward_wall_s": round(rec["forward_wall_s"], 4),
        "total_wall_s": round(rec["total_wall_s"], 4),
        "baseline_device_used_mib": mib(base["device_used_bytes"]),
        "baseline_allocated_mib": mib(base["allocated_bytes"]),
        "baseline_reserved_mib": mib(base["reserved_bytes"]),
        "peak_device_used_mib": mib(rec["peak_device_used_bytes"]),
        "peak_allocated_mib": mib(rec["peak_allocated_bytes"]),
        "peak_reserved_mib": mib(rec["peak_reserved_bytes"]),
        "incremental_device_mib": mib(rec["incremental_device_bytes"]),
        "incremental_allocated_mib": mib(rec["incremental_allocated_bytes"]),
        "peak_rss_mib": mib(rec["peak_rss_bytes"]),
        "peak_system_used_mib": mib(rec["peak_system_used_bytes"]),
        "embedding_dimension": rec.get("embedding_dimension"),
        "ran_within_ceiling_21504_mib": rec["within_ceiling"],
        "budget_status": rec.get("budget_status", ""),
        "device_peak_under_21504_mib": rec.get("device_peak_under_ceiling"),
        "device_peak_under_21e9_bytes": rec.get("device_peak_under_decimal_ceiling"),
        "attention_function_calls": rec.get("attention_function_calls"),
        "past_key_values": rec.get("past_key_values", {}).get("summary", ""),
        "replay_wall_s": round(rec.get("profiler_replay", {}).get("wall_s", 0.0), 4)
        if rec.get("profiler_replay")
        else "",
        "replay_attention_kernels": kernel_names(rec.get("profiler_replay"))
        if rec.get("profiler_replay")
        else "",
    }


# ----------------------------------------------------------------------------------------------
# Stage: report
# ----------------------------------------------------------------------------------------------


def md_table(rows, columns):
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_cell(row.get(c, "")) for c in columns) + " |")
    return "\n".join(lines)


def stage_report(args):
    t_stage = time.perf_counter()
    out_dir = results_dir(args)
    env_dir = environment_dir(args)
    env_dir.mkdir(parents=True, exist_ok=True)
    remove_outputs(
        out_dir, ["case_results.csv", "model_facts.csv", "cross_case_consistency.csv"]
    )
    remove_outputs(
        env_dir, ["runtime_record.csv", "python_packages.csv", "stage_timings.csv"]
    )
    report_path = Path(args.feas_dir) / "REPORT.md"
    tokens_stage = read_json(out_dir / "tokens_stage.json")
    benches = {}
    for key in MODELS:
        p = out_dir / f"bench_{key}.json"
        if p.exists():
            benches[key] = read_json(p)
    if not benches:
        raise SystemExit("no bench_*.json found; run the bench stage first")

    # Long-format tables.
    case_rows, fact_rows, consistency_rows = [], [], []
    for key, b in benches.items():
        if b.get("status") != "completed":
            fact_rows.append(
                {"model": b["model_name"], "fact": "status", "value": b.get("status")}
            )
            continue
        for c, _ in CASES:
            case_rows.append(case_row(b, b["cases"][c]))
        case_rows.append(case_row(b, b["supplementary"]["A_default_cache"]))
        facts = {
            "status": b["status"],
            "repo": b["repo"],
            "revision_requested": b["revision_requested"],
            **{
                f"model_{k}": v
                for k, v in b["model_facts"].items()
                if k != "loading_unexpected_keys_list"
            },
            "embedding_dimension": b["embedding_dimension"],
            **{f"baseline_{k}": v for k, v in b["baseline_after_load"].items()},
            **{f"ceiling_{k}": v for k, v in b["ceiling"].items()},
            **{f"cache_{k}": v for k, v in b["cache"].items()},
            **{f"attention_{k}": v for k, v in b["attention"].items()},
            "warmup_tokens": b["warmup"]["tokens"],
            "warmup_wall_s": b["warmup"]["wall_s"],
            "warmup_device_used_bytes": b["warmup"]["device_used_bytes"],
            "unload_device_used_bytes": b["unload"]["device_used_bytes"],
            **{f"timing_{k}": v for k, v in b["timings"].items()},
        }
        for k, v in facts.items():
            fact_rows.append({"model": b["model_name"], "fact": k, "value": v})
        consistency_rows.extend(b.get("cross_case_consistency", []))
    write_csv(out_dir / "case_results.csv", case_rows)
    write_csv(out_dir / "model_facts.csv", fact_rows, ["model", "fact", "value"])
    write_csv(
        out_dir / "cross_case_consistency.csv",
        consistency_rows,
        ["model", "case_x", "case_y", "document_rank", "doc_key", "cosine"],
    )

    # Environment records.
    env_source = next(iter(benches.values()))["environment"]
    write_csv(
        env_dir / "runtime_record.csv",
        [{"key": k, "value": v} for k, v in env_source.items()],
        ["key", "value"],
    )
    dists = sorted(
        {
            d.metadata["Name"]: d.version
            for d in importlib.metadata.distributions()
            if d.metadata["Name"]
        }.items(),
        key=lambda kv: kv[0].lower(),
    )
    write_csv(
        env_dir / "python_packages.csv",
        [{"package": n, "version": v} for n, v in dists],
        ["package", "version"],
    )
    timing_rows = [
        {"stage": "tokens", "item": k, "seconds": round(v, 3)}
        for k, v in tokens_stage["timings"].items()
    ]
    for key, b in benches.items():
        timing_rows.extend(
            {"stage": f"bench_{key}", "item": k, "seconds": round(v, 3)}
            for k, v in b["timings"].items()
        )

    # REPORT.md
    completed = {k: b for k, b in benches.items() if b.get("status") == "completed"}
    lines = []
    lines.append(
        "# Phase 2 feasibility report: Qwen3-Embedding-4B vs 8B on the worst-case documents"
    )
    lines.append("")
    lines.append(
        f"Generated {utc_now()} by `scripts/phase2_feasibility_qwen3_worst_case.py` (stage `report`). Every number is measured; nothing is estimated or extrapolated."
    )
    lines.append("")
    can = tokens_stage["canonical"]
    lines.append("## Data")
    lines.append("")
    lines.append(
        f"- Canonical table `{can['path']}`: {can['rows']:,} rows (expected {can['expected_rows']:,}: {'verified' if can['rows'] == can['expected_rows'] else 'MISMATCH'})."
    )
    lines.append(
        f"- SHA256 `{can['sha256']}`; equal to the Phase 1 sidecar: {can['hash_equal_to_sidecar']}."
    )
    lines.append(
        f"- Text field: `{can['text_field']}` ({can['text_rule']}); `len(text)` equals Phase 1 `text_n_chars` for {can['n_chars_equal_to_phase1_text_n_chars']:,} of {can['rows']:,} documents."
    )
    lines.append("")
    tk = tokens_stage["tokenizer"]
    lines.append("## Tokenizer identity")
    lines.append("")
    lines.append(
        f"- Tokenizer files byte-identical between the two repositories: {tk['files_identical']}; all identity checks passed: {tk['all_identity_checks_passed']} ({tk['counts_note']})."
    )
    lines.append(
        f"- Call: `{tk['call']}`; special tokens added by default: {tk['special_tokens_added_by_default']}."
    )
    lines.append(
        f"- transformers {tk['versions'].get('transformers')}, tokenizers {tk['versions'].get('tokenizers')}; snapshots at the pinned revisions."
    )
    lines.append("")
    lines.append(
        md_table(tk["identity_rows"], ["item", "value_4B", "value_8B", "identical"])
    )
    lines.append("")
    lines.append("## Token-length distribution (all 902,144 documents)")
    lines.append("")
    lines.append(
        md_table(
            tokens_stage["summary"],
            [
                "model",
                "population",
                "n_documents",
                "min",
                "p50",
                "p90",
                "p99",
                "p99_9",
                "max",
                "sum",
                "limit",
                "n_over_limit",
                "n_over_8192",
                "n_over_40960",
            ],
        )
    )
    lines.append("")
    for key in MODELS:
        top = tokens_stage["top_documents"][key]
        lines.append(
            f"## Eight longest documents by token count: {MODELS[key]['repo'].split('/')[-1]}"
        )
        lines.append("")
        lines.append(
            f"Ids identical under both tokenizers: {top['top_ids_equal_under_both_tokenizers']}. Supported input length {args.supported_length:,} tokens (model card: 32k); config `max_position_embeddings` 40,960."
        )
        lines.append("")
        lines.append(
            md_table(
                top["documents"],
                [
                    "rank",
                    "doc_key",
                    "record_type",
                    "n_tokens",
                    "n_chars",
                    "fits_supported_input_length",
                    "fits_max_position_embeddings",
                ],
            )
        )
        lines.append("")
    lines.append("## Side-by-side results")
    lines.append("")
    lines.append(
        f"VRAM ceiling: {args.vram_ceiling_mib:,} MiB ({gib(args.vram_ceiling_mib * 1048576)} GiB = {args.vram_ceiling_mib * 1048576:,} bytes) of total device memory as NVML reports it, including the share held by other processes. The measured cases pass `use_cache=False`; the supplementary row uses the configuration default. Wall-clock is the forward pass (host-to-device copy and tokenisation are in `results/case_results.csv`). 'Ran within the VRAM ceiling' means the case completed with the device total at or below the ceiling; a case whose allocation was refused at the allocator cap exceeded the permitted budget and is reported as a failure, with the device peak reached before the refusal."
    )
    lines.append("")
    if completed:
        lines.append(side_by_side_table(completed))
    for key, b in benches.items():
        if b.get("status") != "completed":
            lines.append(f"- {b['model_name']}: {b.get('status')}")
    lines.append("")
    lines.append(
        "## Supplementary: Case A with the configuration default cache setting"
    )
    lines.append("")
    supp_rows = []
    for key, b in completed.items():
        r = b["supplementary"]["A_default_cache"]
        supp_rows.append(
            {
                "model": b["model_name"],
                "result": "success"
                if r["success"]
                else "FAILURE: " + r["failure_reason"],
                "input_shape": r["input_shape"],
                "peak_device_used_mib": mib(r["peak_device_used_bytes"]),
                "incremental_device_mib": mib(r["incremental_device_bytes"]),
                "peak_allocated_mib": mib(r["peak_allocated_bytes"]),
                "forward_wall_s": round(r["forward_wall_s"], 3),
                "past_key_values": r.get("past_key_values", {}).get("summary", ""),
                "ran_within_ceiling": r["within_ceiling"],
                "budget_status": r.get("budget_status", ""),
            }
        )
    lines.append(
        md_table(
            supp_rows,
            [
                "model",
                "result",
                "input_shape",
                "peak_device_used_mib",
                "incremental_device_mib",
                "peak_allocated_mib",
                "forward_wall_s",
                "past_key_values",
                "ran_within_ceiling",
                "budget_status",
            ],
        )
    )
    lines.append("")
    lines.append("## Per-model facts")
    lines.append("")
    fact_cols = [
        "model",
        "dtype",
        "n_parameters",
        "revision_loaded",
        "tokenizer_revision_loaded",
        "attn_implementation",
        "config_use_cache",
        "forward_use_cache_default",
        "num_hidden_layers",
        "num_key_value_heads",
        "head_dim",
        "hidden_size",
        "max_position_embeddings",
        "loading_unexpected_keys",
    ]
    lines.append(
        md_table(
            [
                {"model": b["model_name"], **b["model_facts"]}
                for b in completed.values()
            ],
            fact_cols,
        )
    )
    lines.append("")
    lines.append("### Cache behaviour (runtime probe)")
    lines.append("")
    for b in completed.values():
        cp = b["cache_probe"]
        lines.append(
            f"- {b['model_name']}: a default call on {cp['probe_tokens']} tokens returned {cp['default_call']['summary']} (peak allocated {mib(cp['default_call']['peak_allocated_bytes'])} MiB); with `use_cache=False` it returned {cp['use_cache_false_call']['summary']} (peak allocated {mib(cp['use_cache_false_call']['peak_allocated_bytes'])} MiB). Measured cases: {b['cache']['measured_past_key_values']}."
        )
    lines.append("")
    lines.append("### Attention (what actually ran)")
    lines.append("")
    for b in completed.values():
        a = b["attention"]
        lines.append(
            f"- {b['model_name']}: config `{a['config_attn_implementation']}`; function resolved from the transformers attention registry: `{a['registry_function']}`, called {a['calls_per_forward_case_A']} times in the Case A forward pass ({b['model_facts']['num_hidden_layers']} layers); CUDA kernels observed in the profiler replays: {a['kernels_observed']}."
        )
    lines.append("")
    lines.append(
        "### Cross-case agreement of the same document (cosine of the normalised embeddings)"
    )
    lines.append("")
    lines.append(
        md_table(
            consistency_rows,
            ["model", "case_x", "case_y", "document_rank", "doc_key", "cosine"],
        )
        if consistency_rows
        else "(none)"
    )
    lines.append("")
    lines.append("## Load and ceiling record")
    lines.append("")
    load_rows = []
    for b in completed.values():
        bl, ce = b["baseline_after_load"], b["ceiling"]
        load_rows.append(
            {
                "model": b["model_name"],
                "load_wall_s": round(bl["load_wall_s"], 2),
                "load_peak_rss_mib": mib(bl["load_peak_rss_bytes"]),
                "device_used_before_context_mib": mib(
                    ce["device_used_before_context_bytes"]
                ),
                "cuda_context_mib": mib(ce["cuda_context_bytes"]),
                "allocator_cap_mib": mib(ce["allocator_cap_bytes"]),
                "device_used_after_load_mib": mib(bl["device_used_bytes"]),
                "allocated_after_load_mib": mib(bl["allocated_bytes"]),
                "reserved_after_load_mib": mib(bl["reserved_bytes"]),
                "process_attributable_after_load_mib": mib(
                    bl["process_attributable_bytes"]
                ),
                "device_used_after_unload_mib": mib(b["unload"]["device_used_bytes"]),
                "all_measured_cases_ran_within_21504_mib": ce[
                    "all_measured_cases_within_ceiling"
                ],
                "device_peak_under_21504_mib_all_cases": ce.get(
                    "device_peak_under_ceiling_all_cases"
                ),
                "device_peak_under_21e9_bytes_all_cases": ce.get(
                    "device_peak_under_decimal_ceiling_all_cases"
                ),
            }
        )
    lines.append(
        md_table(load_rows, list(load_rows[0].keys()) if load_rows else ["model"])
    )
    lines.append("")
    lines.append("## Environment")
    lines.append("")
    lines.append(
        md_table(
            [{"key": k, "value": v} for k, v in env_source.items()], ["key", "value"]
        )
    )
    lines.append("")
    timing_rows.append(
        {
            "stage": "report",
            "item": "stage_total_s_up_to_report_assembly",
            "seconds": round(time.perf_counter() - t_stage, 3),
        }
    )
    write_csv(env_dir / "stage_timings.csv", timing_rows, ["stage", "item", "seconds"])
    lines.append("## Files")
    lines.append("")
    feas_rel = rel_path(args.feas_dir, args.project_dir)
    lines.append(
        f"- Diagnostic script: `{feas_rel}/scripts/phase2_feasibility_qwen3_worst_case.py`"
    )
    lines.append(f"- Environment build: `{feas_rel}/scripts/setup_wsl2_env.sh`")
    lines.append(
        f"- Unit test: `{feas_rel}/tests/test_phase2_feasibility_functions.py`"
    )
    for name in sorted(
        p.name
        for p in out_dir.iterdir()
        if p.is_file() and not p.name.endswith(".manifest.yml")
    ):
        lines.append(f"- `{feas_rel}/results/{name}`")
    for name in sorted(
        p.name
        for p in env_dir.iterdir()
        if p.is_file() and not p.name.endswith(".manifest.yml")
    ):
        lines.append(f"- `{feas_rel}/environment/{name}`")
    lines.append(
        f"- This report: `{feas_rel}/REPORT.md` (every output has a `.manifest.yml` sidecar)"
    )
    lines.append("")
    lines.append("## Not done, by design")
    lines.append("")
    lines.append(
        "No full-corpus embedding, similarity, clustering, indexing, production batch size, extrapolated runtime, or choice between the two models."
    )
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")

    sidecars = Sidecars(
        args.project_dir, common_parameters(args), "Phase 2 feasibility, report stage."
    )
    inputs = [
        sidecars.input_entry(out_dir / "tokens_stage.json", "tokens stage result")
    ]
    inputs += [
        sidecars.input_entry(
            out_dir / f"bench_{k}.json", f"bench stage result, {MODELS[k]['repo']}"
        )
        for k in benches
    ]
    for path, note in (
        (out_dir / "case_results.csv", "one row per model and case, every measurement"),
        (out_dir / "model_facts.csv", "per-model facts in long format"),
        (
            out_dir / "cross_case_consistency.csv",
            "cosine agreement of the same document across cases",
        ),
        (
            env_dir / "runtime_record.csv",
            "host, GPU, driver and library versions of the bench runs",
        ),
        (env_dir / "python_packages.csv", "every distribution installed in the venv"),
        (env_dir / "stage_timings.csv", "wall-clock of every stage and step"),
        (report_path, "the side-by-side report"),
    ):
        sidecars.write(path, inputs, notes=note)
    log(f"report written to {report_path}")


# ----------------------------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------------------------


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--stage", choices=["tokens", "bench", "report", "all"], default="all"
    )
    ap.add_argument(
        "--model",
        choices=list(MODELS.keys()),
        default=None,
        help="bench stage: which model",
    )
    ap.add_argument("--project-dir", default=str(DEFAULT_PROJECT_DIR))
    ap.add_argument("--feas-dir", default=str(FEAS_DIR))
    ap.add_argument("--expected-rows", type=int, default=902144)
    ap.add_argument(
        "--supported-length",
        type=int,
        default=32768,
        help="supported input length in tokens (model card: 32k)",
    )
    ap.add_argument(
        "--vram-ceiling-mib",
        type=int,
        default=21504,
        help="total device memory ceiling (21 GiB)",
    )
    ap.add_argument("--top-n", type=int, default=8)
    ap.add_argument("--tokenize-batch-size", type=int, default=4096)
    ap.add_argument("--sample-size", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260909)
    ap.add_argument("--nvml-interval-ms", type=int, default=20)
    ap.add_argument("--no-profiler-replay", action="store_true")
    args = ap.parse_args(argv)
    args.project_dir = Path(args.project_dir).resolve()
    args.feas_dir = Path(args.feas_dir).resolve()
    return args


def passthrough(args):
    return [
        "--project-dir",
        str(args.project_dir),
        "--feas-dir",
        str(args.feas_dir),
        "--expected-rows",
        str(args.expected_rows),
        "--supported-length",
        str(args.supported_length),
        "--vram-ceiling-mib",
        str(args.vram_ceiling_mib),
        "--top-n",
        str(args.top_n),
        "--tokenize-batch-size",
        str(args.tokenize_batch_size),
        "--sample-size",
        str(args.sample_size),
        "--seed",
        str(args.seed),
        "--nvml-interval-ms",
        str(args.nvml_interval_ms),
    ] + (["--no-profiler-replay"] if args.no_profiler_replay else [])


def main(argv=None):
    args = parse_args(argv)
    if args.stage == "tokens":
        stage_tokens(args)
    elif args.stage == "bench":
        if args.model is None:
            raise SystemExit("--model is required for the bench stage")
        stage_bench(args, args.model)
    elif args.stage == "report":
        stage_report(args)
    else:
        stage_tokens(args)
        for key in MODELS:
            log(
                f"starting the bench stage for {MODELS[key]['repo']} in a separate process"
            )
            proc = subprocess.run(
                [sys.executable, str(SCRIPT_PATH), "--stage", "bench", "--model", key]
                + passthrough(args)
            )
            log(f"bench {key} exited with code {proc.returncode}")
        stage_report(args)


if __name__ == "__main__":
    main()
