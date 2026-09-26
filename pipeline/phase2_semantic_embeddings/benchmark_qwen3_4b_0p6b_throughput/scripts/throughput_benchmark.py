#!/usr/bin/env python
"""Phase 2 benchmark: steady-state throughput of Qwen3-Embedding-4B and 0.6B on GPU (TEI) and CPU.

Measures, on real documents of the canonical Phase 1 table, the warmed-up embedding throughput
and resource use of four configurations: Qwen3-Embedding-4B and Qwen3-Embedding-0.6B served by
Hugging Face Text Embeddings Inference (TEI, the release image) on the RTX 3090 under a 21 GiB
device ceiling, and the same two models on the CPU with transformers + PyTorch inside a 21 GiB
memory scope on the 8 logical CPUs of the WSL2 VM. It also measures, directly, the largest TEI
`max_batch_tokens` with which the 0.6B model stays under the 21 GiB ceiling. Nothing is
extrapolated; no full-corpus embedding, similarity analysis, clustering, indexing or model choice.

Stages (run inside the WSL2 venv; see ../README.md):
  prepare                   verify the inputs, compare the 0.6B tokenizer with the 4B tokenizer on
                            every document, build the benchmark stream, build the cap shim, record
                            the image, Docker, driver, host and model facts
  gpu_calibrate --model M   one TEI container at the model's minimum budget: untracked device
                            overhead, /info, TEI's token counts of the candidates and a stream sample
  gpu_search                the 0.6B max_batch_tokens search (bracket, bisection, confirmation)
  gpu_throughput --model M  3 warm-up and 10 measured pipelined batches at the model's budget
  cpu_sweep --model M       the bounded CPU batching sweep (one worker, 21 GiB scope, 8 threads)
  cpu_throughput --model M  3 warm-up and 10 measured CPU batches at the chosen budget
  cpu_worker                internal: the CPU process started by cpu_sweep and cpu_throughput
  compare                   float16 (GPU, TEI) versus float32 (CPU) embeddings of the same documents
  report                    the tables, the environment record and REPORT.md
  all                       every stage in order

Layout: pure functions first (unit-tested by ../tests/test_throughput_benchmark_functions.py),
then the runtime helpers, then the stages.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
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
BENCH_DIR = SCRIPT_PATH.parents[1]  # benchmark_qwen3_4b_0p6b_throughput/
PHASE2_DIR = SCRIPT_PATH.parents[2]  # phase2_semantic_embeddings/
DEFAULT_PROJECT_DIR = SCRIPT_PATH.parents[4]  # the repository root 
PRIOR_FEAS_DIR = PHASE2_DIR / "feasibility_qwen3_embedding_worst_case"
TEI_TEST_DIR = PHASE2_DIR / "feasibility_tei_qwen3_4b_batch_tokens"
SHIM_SRC_DIR = TEI_TEST_DIR / "scripts" / "vram_cap"


def _sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


# The hash of the script as it was when this process started: every sidecar records it, so an
# edit made while a long run is in progress cannot be attributed to that run's outputs.
SCRIPT_SHA256 = _sha256_of(SCRIPT_PATH)

MODELS = {
    "4B": {
        "repo": "Qwen/Qwen3-Embedding-4B",
        "revision": "5cf2132abc99cad020ac570b19d031efec650f2b",
        "cache_dir": "models--Qwen--Qwen3-Embedding-4B",
        "slug": "4b",
        "n_min": 40960,
    },
    "0.6B": {
        "repo": "Qwen/Qwen3-Embedding-0.6B",
        "revision": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        "cache_dir": "models--Qwen--Qwen3-Embedding-0.6B",
        "slug": "0p6b",
        "n_min": 32768,
    },
}
IMAGE = "ghcr.io/huggingface/text-embeddings-inference:86-1.9.3"
IMAGE_EXPECTED_DIGEST = (
    "sha256:a7d82dfef16c3bf1a95e93f5b226f358312512dbb0d585b48c3cf886f9d470a9"
)
CONTAINER_PREFIX = "tei_throughput_bench"
# No --max-batch-requests: TEI then has no document-count limit per batch. The client batch size
# and concurrency limits are raised far above any request of this benchmark (TEI takes one
# concurrency permit per input and rejects a request with more inputs than the client limit).
TEI_ARGS_FIXED = [
    "--auto-truncate",
    "false",
    "--max-client-batch-size",
    "100000",
    "--max-concurrent-requests",
    "200000",
    "--payload-limit",
    "100000000",
    "--json-output",
]
STREAM_SEED = 20260910
CANONICAL_TEXT_FIELD = "text"
CANONICAL_TEXT_RULE = "comment: body; submission: title + ' ' + selftext (Phase 1 DATA_DICTIONARY.md field 25)"
DECIMAL_CEILING_BYTES = 21_000_000_000
PACKAGE_NAMES = [
    "torch",
    "transformers",
    "tokenizers",
    "safetensors",
    "huggingface_hub",
    "httpx",
    "pyarrow",
    "numpy",
    "psutil",
    "nvidia-ml-py",
    "pyyaml",
]
TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
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


def benchmark_order(doc_keys, excluded_rows, seed):
    """Row indices of the benchmark stream: every row not excluded, ordered by a keyed hash of
    (seed, doc_key) (BLAKE2b, 8 bytes), ties by doc_key. A document's position depends only on
    the seed and its key, not on the row order of the table or on the NumPy version."""
    excluded = {int(i) for i in excluded_rows}
    keyed = []
    for i, key in enumerate(doc_keys):
        if i in excluded:
            continue
        digest = hashlib.blake2b(
            f"{seed}:{key}".encode("utf-8"), digest_size=8
        ).digest()
        keyed.append((int.from_bytes(digest, "big"), str(key), i))
    keyed.sort()
    return np.array([i for _, _, i in keyed], dtype=np.int64)


def pack_greedy(n_tokens, start, budget, n_batches):
    """Consecutive batches from position `start` of a stream of token counts, by TEI's own rule
    for a non-padded model: take documents while the sum stays within the budget; the first
    document that does not fit starts the next batch. A document longer than the budget (possible
    only for CPU budgets below the longest document) forms a batch on its own (`oversize`).
    Stops early at the end of the stream."""
    batches = []
    pos = int(start)
    n = len(n_tokens)
    for _ in range(int(n_batches)):
        if pos >= n:
            break
        begin, total = pos, 0
        while pos < n and total + int(n_tokens[pos]) <= budget:
            total += int(n_tokens[pos])
            pos += 1
        oversize = False
        if pos == begin:
            total = int(n_tokens[pos])
            pos += 1
            oversize = True
        batches.append(
            {
                "begin": begin,
                "end": pos,
                "n_documents": pos - begin,
                "n_tokens": total,
                "oversize": oversize,
            }
        )
    return batches


def plan_forward_passes(lengths, budget):
    """Forward passes of one CPU batch: the documents sorted by length (longest first, ties by
    position) and grouped while (documents in the group) x (longest length in the group) stays
    within the budget of padded tokens; a document longer than the budget runs alone. Returns
    lists of positions into `lengths`."""
    order = sorted(range(len(lengths)), key=lambda i: (-int(lengths[i]), i))
    groups, current, current_max = [], [], 0
    for i in order:
        length = int(lengths[i])
        if current and (len(current) + 1) * current_max <= budget:
            current.append(i)
            continue
        if current:
            groups.append(current)
        current, current_max = [i], length
    if current:
        groups.append(current)
    return groups


def padded_tokens(lengths, groups):
    """Total padded tokens of the forward passes: each group is padded to its longest member."""
    return sum(len(g) * max(int(lengths[i]) for i in g) for g in groups if g)


def timing_stats(times_s):
    """Total, mean, median, minimum, maximum and sample standard deviation (n - 1) of batch times."""
    a = [float(x) for x in times_s]
    n = len(a)
    if n == 0:
        return {
            "n": 0,
            "total_s": 0.0,
            "mean_s": None,
            "median_s": None,
            "min_s": None,
            "max_s": None,
            "std_s": None,
        }
    total = sum(a)
    mean = total / n
    s = sorted(a)
    median = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0
    std = math.sqrt(sum((x - mean) ** 2 for x in a) / (n - 1)) if n > 1 else None
    return {
        "n": n,
        "total_s": total,
        "mean_s": mean,
        "median_s": median,
        "min_s": min(a),
        "max_s": max(a),
        "std_s": std,
    }


def steady_state(batches):
    """The steady-state statistics of measured batches (dicts with n_documents, n_tokens, time_s):
    the timing statistics plus totals, documents and tokens per second (totals over the total
    measured time) and the largest and smallest batch."""
    out = dict(timing_stats([b["time_s"] for b in batches]))
    docs = [int(b["n_documents"]) for b in batches]
    toks = [int(b["n_tokens"]) for b in batches]
    total = out["total_s"]
    out.update(
        {
            "total_documents": sum(docs),
            "total_tokens": sum(toks),
            "documents_per_s": (sum(docs) / total) if total > 0 else None,
            "tokens_per_s": (sum(toks) / total) if total > 0 else None,
            "max_documents_in_batch": max(docs) if docs else None,
            "min_documents_in_batch": min(docs) if docs else None,
            "max_tokens_in_batch": max(toks) if toks else None,
            "min_tokens_in_batch": min(toks) if toks else None,
        }
    )
    return out


def sweep_next(points, gpu_budget, start=1024, min_gain=0.05):
    """The next CPU batch budget of the sweep, or (None, reason) to stop.

    points: the completed points in order, dicts with budget, tokens_per_s (None if the point did
    not complete), failed_at_ceiling, error. The budget starts at `start` and doubles; it never
    exceeds the GPU budget of the same model (the last point is that budget). The sweep stops at a
    failure, at the GPU budget, or at the first point whose throughput is not at least `min_gain`
    above the best earlier point (improvement stopped or throughput worsened)."""
    gpu_budget = int(gpu_budget)
    if not points:
        return min(int(start), gpu_budget), "start"
    last = points[-1]
    if last.get("failed_at_ceiling"):
        return None, f"failure at the RAM ceiling at {last['budget']}"
    if last.get("error") or last.get("tokens_per_s") is None:
        return None, f"error at {last['budget']}: {last.get('error')}"
    earlier = [
        p["tokens_per_s"] for p in points[:-1] if p.get("tokens_per_s") is not None
    ]
    if earlier:
        best = max(earlier)
        gain = last["tokens_per_s"] / best - 1.0
        if gain < min_gain - 1e-12:
            return None, (
                f"no material improvement: {last['tokens_per_s']:.1f} tokens/s at {last['budget']} "
                f"vs best earlier {best:.1f} ({gain:+.1%}; the sweep continues only at +{min_gain:.0%} or more)"
            )
    if int(last["budget"]) >= gpu_budget:
        return None, f"reached the GPU budget ({gpu_budget})"
    nxt = int(last["budget"]) * 2
    if nxt >= gpu_budget:
        return gpu_budget, f"double, capped at the GPU budget ({gpu_budget})"
    return nxt, "double"


def sweep_choice(points):
    """The completed sweep point with the highest tokens per second (a tie goes to the smaller
    budget), or None."""
    ok = [
        p
        for p in points
        if p.get("tokens_per_s") is not None
        and not p.get("failed_at_ceiling")
        and not p.get("error")
    ]
    if not ok:
        return None
    return max(ok, key=lambda p: (p["tokens_per_s"], -int(p["budget"])))


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


def _bucket_bound(key):
    return math.inf if key.lstrip("+").lower() == "inf" else float(key)


def verify_interval(size_delta, tokens_delta, expected):
    """Did TEI form exactly the expected batches between two /metrics readings?

    size_delta, tokens_delta: the deltas of the te_batch_next_size and te_batch_next_tokens
    histograms; expected: [(n_documents, n_tokens)] of the batches that should have formed. TEI
    records every poll of its queue in both histograms, empty polls as 0; every real batch has at
    least two tokens (text plus the appended end-of-text token), so the observations at or below
    1 token are the empty polls. The number of real batches, the sums and every cumulative bucket
    count must equal those of the expected batches (plus the empty polls)."""
    n_empty = int(round(tokens_delta["buckets"].get("1", 0.0)))
    real = int(round(tokens_delta["count"])) - n_empty
    exp_sizes = [int(e[0]) for e in expected]
    exp_tokens = [int(e[1]) for e in expected]

    def buckets_match(delta, values):
        for key, observed in delta["buckets"].items():
            bound = _bucket_bound(key)
            want = n_empty + sum(1 for v in values if v <= bound)
            if int(round(observed)) != want:
                return False
        return True

    checks = {
        "n_real_batches": real == len(expected),
        "size_sum": int(round(size_delta["sum"])) == sum(exp_sizes),
        "token_sum": int(round(tokens_delta["sum"])) == sum(exp_tokens),
        "size_buckets": buckets_match(size_delta, exp_sizes),
        "token_buckets": buckets_match(tokens_delta, exp_tokens),
        "size_count_equals_token_count": int(round(size_delta["count"]))
        == int(round(tokens_delta["count"])),
    }
    info = {"n_empty_polls": n_empty, "real_batches": real}
    return all(checks.values()), checks, info


def compute_cap_bytes(ceiling_bytes, device_used_before, untracked_overhead):
    """The allocation cap of one container: the ceiling minus what the device already holds
    before the container starts minus the process's untracked device memory."""
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
    """(outcome, success, failure_at_ceiling) from the flags of one container. A refusal at the
    cap is the primary event even when the watchdog also fired (the device excursion of a
    rejected pool attempt) or the device peak went over the ceiling during that excursion."""
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


def cosine_rows(a, b):
    """Row-wise cosine similarity of two matrices of the same shape (float64 arithmetic)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    num = np.sum(a * b, axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den


def agreement_summary(cos):
    """Distribution of per-document cosines (non-finite values counted and left out)."""
    c = np.asarray(cos, dtype=np.float64)
    finite = c[np.isfinite(c)]
    out = {"n": int(c.size), "n_nonfinite": int(c.size - finite.size)}
    keys = ("min", "p0_1", "p1", "p5", "median", "mean", "max")
    if finite.size:
        out.update(
            {
                "min": float(finite.min()),
                "p0_1": float(np.percentile(finite, 0.1)),
                "p1": float(np.percentile(finite, 1)),
                "p5": float(np.percentile(finite, 5)),
                "median": float(np.median(finite)),
                "mean": float(finite.mean()),
                "max": float(finite.max()),
                "n_below_0_9999": int((finite < 0.9999).sum()),
                "n_below_0_999": int((finite < 0.999).sum()),
                "n_below_0_99": int((finite < 0.99).sum()),
            }
        )
    else:
        out.update({k: None for k in keys})
        out.update({"n_below_0_9999": 0, "n_below_0_999": 0, "n_below_0_99": 0})
    return out


def parse_cpu_stat(text):
    """A cgroup v2 cpu.stat file to {key: int}."""
    out = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].lstrip("-").isdigit():
            out[parts[0]] = int(parts[1])
    return out


def cpu_utilisation(usage_usec_delta, wall_s, n_cpus):
    """(average busy CPUs, fraction of the n_cpus) from a cpu.stat usage delta over a wall time."""
    if wall_s is None or wall_s <= 0:
        return None, None
    busy = usage_usec_delta / 1e6 / wall_s
    return busy, busy / n_cpus


_STATUS_KEYS = {
    "VmPeak",
    "VmSize",
    "VmHWM",
    "VmRSS",
    "RssAnon",
    "RssFile",
    "RssShmem",
    "VmSwap",
    "Threads",
}


def parse_proc_status(text):
    """The memory lines (in bytes) and the thread count of a /proc/<pid>/status file."""
    out = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if key not in _STATUS_KEYS:
            continue
        parts = value.split()
        if not parts or not parts[0].isdigit():
            continue
        n = int(parts[0])
        out[key] = n * 1024 if len(parts) > 1 and parts[1] == "kB" else n
    return out


def json_top_level_diff(a, b):
    """The top-level keys whose values differ between two JSON objects (or exist in one only)."""
    return sorted(
        k for k in set(a) | set(b) if k not in a or k not in b or a[k] != b[k]
    )


def classify_cpu_exit(returncode, oom_kills, peak_mem_bytes, limit_bytes):
    """(outcome, failure_at_ceiling) of a CPU worker. A SIGKILL is a failure at the RAM ceiling
    when the kernel counted an OOM kill in the scope or the scope's memory had reached 95% of
    its limit."""
    if returncode == 0:
        return "success", False
    killed = returncode in (-9, 137)
    if killed and (
        (oom_kills or 0) > 0 or (peak_mem_bytes or 0) >= 0.95 * int(limit_bytes)
    ):
        return "killed_at_ram_ceiling", True
    if killed:
        return "killed", False
    return "error", False


def speed_ratio(a, b):
    """a / b, or None when either is missing or b is zero."""
    if a is None or b is None or b == 0:
        return None
    return a / b


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
# Runtime helpers: files, provenance, processes
# ----------------------------------------------------------------------------------------------


def utc_now():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(message):
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def sha256_file(path):
    return _sha256_of(path)


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
    ".npy": "other",
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
    """Writes the provenance sidecar of every output of a stage (script hash taken at process start)."""

    def __init__(self, project_dir, parameters, notes, script_path=None):
        self.project_dir = Path(project_dir)
        self.parameters = parameters
        self.notes = notes
        self.script_rel = rel_path(script_path or SCRIPT_PATH, project_dir)
        self.script_hash = sha256_file(script_path) if script_path else SCRIPT_SHA256
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
    if isinstance(value, (bytes, bytearray)):
        return f"<{len(value)} bytes>"
    if isinstance(value, Path):
        return str(value)
    return str(value)


def write_json(path, data):
    """Writes JSON through a temporary file and a rename, so a kill mid-write never leaves a
    truncated file behind."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, ensure_ascii=False, default=_json_default)
    os.replace(tmp, path)
    return path


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def remove_outputs(paths):
    """Removes earlier outputs of a stage and their sidecars before the stage writes new ones."""
    for p in paths:
        p = Path(p)
        for q in (p, p.with_name(p.name + ".manifest.yml")):
            if q.exists() and q.is_file():
                q.unlink()


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


# --- cgroups, /proc and system memory ------------------------------------------------------------


def read_text(path):
    try:
        return Path(path).read_text(encoding="utf-8")
    except Exception:
        return None


def cgroup_dir_of_pid(pid):
    """The cgroup v2 directory of a process (host view), or None."""
    text = read_text(f"/proc/{pid}/cgroup")
    if not text:
        return None
    for line in text.splitlines():
        if line.startswith("0::"):
            return "/sys/fs/cgroup" + line[3:].strip()
    return None


def cgroup_int(cgdir, name):
    text = read_text(Path(cgdir) / name)
    if text is None:
        return None
    text = text.strip()
    return int(text) if text.isdigit() else None


def cgroup_keyed(cgdir, name):
    """A cgroup 'key value' file (memory.events, memory.stat, cpu.stat) to {key: int}."""
    text = read_text(Path(cgdir) / name)
    return parse_cpu_stat(text) if text else {}


def proc_status(pid):
    text = read_text(f"/proc/{pid}/status")
    return parse_proc_status(text) if text else {}


def vm_used_bytes():
    """Memory in use in the WSL2 VM (total minus available), as the kernel reports it."""
    import psutil

    vm = psutil.virtual_memory()
    return int(vm.total - vm.available)


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
            "server={{.ServerVersion}} runtimes={{.Runtimes}} default_runtime={{.DefaultRuntime}} storage={{.Driver}} root={{.DockerRootDir}} cgroup_driver={{.CgroupDriver}} cgroup_version={{.CgroupVersion}}",
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
            "{{.State.Running}} {{.State.ExitCode}} {{.State.OOMKilled}} {{.State.Pid}} {{.State.Error}}",
            name,
        ],
        timeout=60,
    )
    if out.returncode != 0:
        return {"exists": False, "running": False, "exit_code": None}
    parts = out.stdout.strip().split(" ", 4)
    return {
        "exists": True,
        "running": parts[0] == "true",
        "exit_code": int(parts[1])
        if len(parts) > 1 and parts[1].lstrip("-").isdigit()
        else None,
        "oom_killed": parts[2] == "true" if len(parts) > 2 else None,
        "pid": int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else None,
        "error": parts[4] if len(parts) > 4 else "",
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


def snapshot_dir_in_container(model_key):
    spec = MODELS[model_key]
    return f"/data/{spec['cache_dir']}/snapshots/{spec['revision']}"


def docker_run_tei(ctx, name, model_key, max_batch_tokens, cap_bytes, report_path):
    """Starts one TEI container with the cap shim; returns the docker run command used."""
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
        snapshot_dir_in_container(model_key),
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


def post_embed(ctx, texts, timeout_s, keep_body=False):
    """One /embed request with truncation disabled and L2 normalisation; returns the status, the
    timing headers, the send and arrival times, the body size and (on request) the raw body."""
    t0 = time.perf_counter()
    result = {
        "n_inputs": len(texts),
        "status": None,
        "headers": {},
        "t_send": time.time(),
        "t_done": None,
        "wall_s": None,
        "body_bytes": None,
        "body": None,
        "error": None,
        "exception": None,
    }
    try:
        payload = json.dumps(
            {"inputs": texts, "truncate": False, "normalize": True}
        ).encode("utf-8")
        with http_client(ctx, timeout_s) as client:
            r = client.post(
                "/embed",
                content=payload,
                headers={"content-type": "application/json"},
            )
        body = r.content
        result["t_done"] = time.time()
        result["wall_s"] = time.perf_counter() - t0
        result["status"] = r.status_code
        result["headers"] = {
            k: r.headers.get(k) for k in HEADER_KEYS if r.headers.get(k) is not None
        }
        result["body_bytes"] = len(body)
        if r.status_code == 200:
            if keep_body:
                result["body"] = body
        else:
            result["error"] = body[:2000].decode("utf-8", "replace")
    except Exception as e:
        result["t_done"] = time.time()
        result["wall_s"] = time.perf_counter() - t0
        result["exception"] = repr(e)
    return result


def embeddings_from_body(body):
    """The embeddings of an /embed response body as a float32 matrix."""
    return np.asarray(json.loads(body), dtype=np.float32)


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


def header_int(headers, key):
    v = (headers or {}).get(key)
    return int(v) if v is not None and str(v).isdigit() else None


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


def tei_search_summary_paths():
    path = TEI_TEST_DIR / "results" / "search_summary.json"
    return path, path.with_name(path.name + ".manifest.yml")


def verify_inputs(project_dir):
    """Hash the canonical table, the token table and the TEI test's search summary against their
    sidecars; refuse to run on unverified inputs."""
    checks = {}
    for label, (path, sidecar) in (
        ("canonical", canonical_paths(project_dir)),
        ("token_table", token_table_paths()),
        ("tei_4b_search_summary", tei_search_summary_paths()),
    ):
        if not path.exists():
            raise SystemExit(f"{label} missing: {path}")
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
                f"{label} hash {measured} differs from its sidecar {expected}: refusing to run on unverified data"
            )
    return checks


class Corpus:
    """The canonical documents: keys, record types, character counts and the stored exact
    Qwen3-Embedding token counts (n_tokens_4B, which include the appended end-of-text token);
    texts are read from the canonical table on first use."""

    def __init__(self, project_dir):
        import pyarrow.parquet as pq

        self.project_dir = project_dir
        table = pq.read_table(
            token_table_paths()[0],
            columns=["doc_key", "record_type", "n_chars", "n_tokens_4B"],
        )
        self.doc_keys = np.array(table.column("doc_key").to_pylist(), dtype=object)
        self.record_types = np.array(
            table.column("record_type").to_pylist(), dtype=object
        )
        self.n_chars = np.array(table.column("n_chars").to_pylist(), dtype=np.int64)
        self.n_tokens_4b = np.array(
            table.column("n_tokens_4B").to_pylist(), dtype=np.int64
        )
        self.n_rows = table.num_rows
        self._texts = None

    def texts_column(self):
        if self._texts is None:
            import pyarrow.parquet as pq

            ct = pq.read_table(
                canonical_paths(self.project_dir)[0],
                columns=["doc_key", CANONICAL_TEXT_FIELD],
            )
            keys = ct.column("doc_key").to_pylist()
            if keys != list(self.doc_keys):
                raise SystemExit(
                    "the canonical table's row order differs from the token table's"
                )
            self._texts = ct.column(CANONICAL_TEXT_FIELD)
        return self._texts

    def texts(self, rows):
        import pyarrow as pa

        col = self.texts_column()
        return col.take(pa.array(np.asarray(rows, dtype=np.int64))).to_pylist()


def model_token_counts(args, corpus, model_key):
    """The exact token count of every document under the model's tokenizer: the stored counts
    for the 4B; for the 0.6B the same counts when the prepare stage found the two tokenizers
    identical on every document, otherwise the 0.6B counts it wrote."""
    if model_key == "4B":
        return corpus.n_tokens_4b
    prep = read_json(results_dir(args) / "prepare.json")
    if prep["tokenizer"]["token_ids_identical_on_every_document"]:
        return corpus.n_tokens_4b
    import pyarrow.parquet as pq

    table = pq.read_table(results_dir(args) / "token_counts_0p6b.parquet")
    if table.column("doc_key").to_pylist() != list(corpus.doc_keys):
        raise SystemExit("token_counts_0p6b.parquet is not in canonical order")
    return np.array(table.column("n_tokens_0p6B").to_pylist(), dtype=np.int64)


def benchmark_stream(args, corpus):
    """The benchmark stream (row indices), recomputed and checked against the prepare record."""
    prep = read_json(results_dir(args) / "prepare.json")
    st = prep["stream"]
    order = benchmark_order(corpus.doc_keys, st["excluded_rows"], st["seed"])
    digest = stream_digest(corpus, order)
    if digest != st["order_sha256"]:
        raise SystemExit(
            "the benchmark stream differs from the one recorded by prepare"
        )
    return order, st


def stream_digest(corpus, order):
    h = hashlib.sha256()
    for i in order:
        h.update(str(corpus.doc_keys[i]).encode("utf-8") + b"\n")
    return "sha256:" + h.hexdigest()


def plan_batches(corpus, order, counts, budget, n_warmup, n_measured, midpoint):
    """Warm-up batches packed from the start of the stream and measured batches packed from its
    midpoint, both by TEI's greedy rule under the same budget (so the measured documents of every
    configuration start at the same document)."""
    stream_counts = counts[order]
    out = []
    for phase, start, n in (
        ("warmup", 0, n_warmup),
        ("measured", midpoint, n_measured),
    ):
        for k, b in enumerate(pack_greedy(stream_counts, start, budget, n)):
            rows = order[b["begin"] : b["end"]]
            out.append(
                {
                    "phase": phase,
                    "phase_index": k + 1,
                    "stream_begin": b["begin"],
                    "stream_end": b["end"],
                    "rows": rows,
                    "doc_keys": [str(x) for x in corpus.doc_keys[rows]],
                    "n_tokens_list": [int(x) for x in counts[rows]],
                    "n_documents": b["n_documents"],
                    "n_tokens": b["n_tokens"],
                    "oversize": b["oversize"],
                }
            )
    warm_end = max((b["stream_end"] for b in out if b["phase"] == "warmup"), default=0)
    if warm_end > midpoint:
        raise SystemExit("warm-up batches would overlap the measured region")
    for i, b in enumerate(out, start=1):
        b["seq"] = i
    return out


def candidate_rows(corpus, counts, n_candidates):
    """The n longest documents (tokens desc, chars desc, doc_key asc) plus the longest document of
    every shorter token length down to two tokens, so that first-fit-decreasing can meet any
    budget exactly; the empty document (1 token) is never a candidate."""
    idx = rank_candidates(counts, corpus.n_chars, corpus.doc_keys, n_candidates)
    pools = ["top"] * len(idx)
    min_top = int(counts[idx[-1]]) if idx else 0
    seen = set()
    for i in np.lexsort(
        (np.asarray(corpus.doc_keys).astype(str), -corpus.n_chars, -counts)
    ):
        n = int(counts[i])
        if n >= min_top:
            continue
        if n < 2:
            break
        if n not in seen:
            seen.add(n)
            idx.append(int(i))
            pools.append("tail")
    texts = corpus.texts(idx)
    rows = []
    for rank, (i, pool, text) in enumerate(zip(idx, pools, texts), start=1):
        if not text:
            continue
        rows.append(
            {
                "rank": rank,
                "pool": pool,
                "row": int(i),
                "doc_key": str(corpus.doc_keys[i]),
                "record_type": str(corpus.record_types[i]),
                "n_chars": int(corpus.n_chars[i]),
                "n_tokens": int(counts[i]),
                "text": text,
            }
        )
    return rows


# --- records ------------------------------------------------------------------------------------


def snapshot_path(hub_dir, model_key):
    spec = MODELS[model_key]
    return (
        Path(hub_dir).expanduser() / spec["cache_dir"] / "snapshots" / spec["revision"]
    )


def snapshot_record(hub_dir, model_key):
    """The model files actually used: the snapshot path and the blob names (Hub LFS blobs are named
    by their SHA256) with sizes; small files hashed directly."""
    snap = snapshot_path(hub_dir, model_key)
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
        "repo": MODELS[model_key]["repo"],
        "revision": MODELS[model_key]["revision"],
        "snapshot_path": str(snap),
        "files": files,
    }


def host_record():
    """The WSL2 VM as the benchmark sees it (CPUs, memory, swap, kernel) and the .wslconfig that
    sets its limits."""
    rec = {
        "uname": " ".join(platform.uname()),
        "os_cpu_count": os.cpu_count(),
        "sched_getaffinity": len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
    }
    for label, cmd in (
        ("lscpu", ["lscpu"]),
        ("free_b", ["free", "-b"]),
        ("swapon", ["swapon", "--show"]),
        ("nproc", ["nproc"]),
    ):
        try:
            rec[label] = run(cmd, timeout=30, merge_stderr=True).stdout
        except Exception as e:
            rec[label] = f"(failed: {e!r})"
    wslconfigs = {}
    for p in sorted(Path("/mnt/c/Users").glob("*/.wslconfig")):
        wslconfigs[str(p)] = read_text(p)
    rec["wslconfig_files"] = wslconfigs
    return rec


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
    return {
        "build_output": out.stdout,
        "build_dir": str(build_dir),
        "libcuda_so_1_sha256": sha256_file(so),
        "vram_cap_c_sha256": sha256_file(SHIM_SRC_DIR / "vram_cap.c"),
        "gen_trampolines_py_sha256": sha256_file(SHIM_SRC_DIR / "gen_trampolines.py"),
        "build_script_sha256": sha256_file(SHIM_SRC_DIR / "build_vram_cap.sh"),
    }


def results_dir(args):
    return Path(args.results_dir)


def environment_dir(args):
    return Path(args.environment_dir)


def stage_dir(args, name):
    d = results_dir(args) / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def gpu_parameters(args):
    return {
        "models": {
            k: f"{v['repo']} @ {v['revision']} (local snapshot, HF_HUB_OFFLINE=1)"
            for k, v in MODELS.items()
        },
        "tei_image": IMAGE,
        "tei_expected_index_digest": IMAGE_EXPECTED_DIGEST,
        "tei_fixed_args": " ".join(TEI_ARGS_FIXED)
        + " --port <port> --max-batch-tokens <budget>",
        "tei_dtype": "float16 (TEI default on CUDA; the only precision of its flash Qwen3 path)",
        "tei_pooling": "from 1_Pooling/config.json (last_token)",
        "embed_request": '{"inputs": [...], "truncate": false, "normalize": true}',
        "vram_ceiling_mib": args.ceiling_mib,
        "vram_ceiling_bytes": args.ceiling_mib * 1048576,
        "decimal_ceiling_bytes": DECIMAL_CEILING_BYTES,
        "enforcement": "in-process allocation cap (feasibility_tei_qwen3_4b_batch_tokens/scripts/vram_cap libcuda.so.1 shim) = ceiling - device used before start - calibrated untracked overhead; NVML watchdog kills the container if the device total exceeds the ceiling; success requires every sample at or below the ceiling",
        "nvml_sample_interval_ms": args.nvml_interval_ms,
        "stream": f"seeded order of the canonical documents without the empty one (BLAKE2b of '{STREAM_SEED}:<doc_key>'); warm-up batches from position 0, measured batches from the midpoint; TEI's greedy packing rule",
        "n_warmup": args.n_warmup,
        "n_measured": args.n_measured,
        "canonical_text_field": CANONICAL_TEXT_FIELD,
        "canonical_text_rule": CANONICAL_TEXT_RULE,
        "quick_run": bool(args.quick),
    }


def cpu_parameters(args):
    return {
        "models": {
            k: f"{v['repo']} @ {v['revision']} (local snapshot, HF_HUB_OFFLINE=1)"
            for k, v in MODELS.items()
        },
        "implementation": "transformers AutoModel, float32, attn_implementation=sdpa with use_gqa_in_sdpa forced to False (keys and values repeated to the query heads; PyTorch's memory-efficient CPU kernel), use_cache=False, left padding, last-token pooling, L2 normalisation, no truncation, no prompt",
        "threads": args.cpu_threads,
        "memory_max_bytes": args.ram_ceiling_mib * 1048576,
        "memory_swap_max_bytes": 0,
        "scope": "systemd-run --user --scope -p MemoryMax=<bytes> -p MemorySwapMax=0 (the kernel kills the worker at the ceiling)",
        "concurrency": "one batch at a time, one process, the model loaded once per worker",
        "batch": "a planned batch of real tokens within the budget (TEI's greedy rule on the same stream); forward passes of at most the budget in padded tokens, documents sorted by length (a document longer than the budget runs alone); batch time = forward passes + pooling + normalisation (tokenisation timed separately)",
        "sweep": f"budget doubles from {args.cpu_sweep_start}, never above the GPU budget of the same model; {args.cpu_sweep_warmup} warm-up and {args.cpu_sweep_timed} timed batches per point; stop at the first point without a {args.cpu_min_gain:.0%} gain over the best earlier point, at a failure, or at the GPU budget",
        "n_warmup": args.n_warmup,
        "n_measured": args.n_measured,
        "quick_run": bool(args.quick),
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


def settle_device(ctx, reference_used, margin_bytes, timeout_s):
    """Waits until the device total has come back down near the reference reading."""
    t0 = time.time()
    used = ctx["nvml"].used()
    while used > reference_used + margin_bytes and time.time() - t0 < timeout_s:
        time.sleep(0.5)
        used = ctx["nvml"].used()
    return used


# ----------------------------------------------------------------------------------------------
# Stage: prepare
# ----------------------------------------------------------------------------------------------


def _brief(value, limit=200):
    """A short JSON rendering of a tokenizer setting (vocabularies and merge lists as counts)."""
    if isinstance(value, dict):
        value = {
            k: (f"<{len(v)} entries>" if k in ("vocab", "merges") else v)
            for k, v in value.items()
        }
    text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return text if len(text) <= limit else text[:limit] + f"...({len(text)} chars)"


def _by_id_diff(a_list, b_list, id_key):
    """Entries (dicts keyed by `id_key`) present in one list only or different between lists."""
    a = {str(x.get(id_key)): x for x in (a_list or [])}
    b = {str(x.get(id_key)): x for x in (b_list or [])}
    ids = sorted((i for i in set(a) | set(b) if a.get(i) != b.get(i)), key=str)
    return ids, a, b


def tokenizer_identity(args, corpus):
    """Compares the 0.6B tokenizer with the tested 4B tokenizer: the files, every section of
    tokenizer.json and tokenizer_config.json, the loaded tokenizers' vocabulary, special tokens and
    post-processor, and the token ids of every canonical document. Returns (rows, summary, counts
    of the 0.6B tokenizer)."""
    from transformers import AutoTokenizer

    snaps = {k: snapshot_path(args.hub_dir, k) for k in MODELS}
    rows = []

    def add(item, v4, v06, identical=None, note=""):
        rows.append(
            {
                "item": item,
                "value_4B": v4,
                "value_0p6B": v06,
                "identical": (v4 == v06) if identical is None else bool(identical),
                "note": note,
            }
        )

    for fname in TOKENIZER_FILES:
        vals, sizes = {}, {}
        for k in MODELS:
            p = snaps[k] / fname
            vals[k] = sha256_file(p) if p.exists() else "absent"
            sizes[k] = p.stat().st_size if p.exists() else None
        add(
            f"file:{fname}",
            vals["4B"],
            vals["0.6B"],
            note=f"sizes {sizes['4B']} / {sizes['0.6B']} bytes"
            if vals["4B"] != "absent" or vals["0.6B"] != "absent"
            else "absent in both repositories",
        )

    tj = {
        k: json.loads((snaps[k] / "tokenizer.json").read_text(encoding="utf-8"))
        for k in MODELS
    }
    for key in sorted(set(tj["4B"]) | set(tj["0.6B"])):
        a, b = tj["4B"].get(key), tj["0.6B"].get(key)
        if key == "model":
            add(
                "tokenizer.json:model.vocab",
                f"{len((a or {}).get('vocab', {}))} entries",
                f"{len((b or {}).get('vocab', {}))} entries",
                (a or {}).get("vocab") == (b or {}).get("vocab"),
            )
            add(
                "tokenizer.json:model.merges",
                f"{len((a or {}).get('merges', []))} merges",
                f"{len((b or {}).get('merges', []))} merges",
                (a or {}).get("merges") == (b or {}).get("merges"),
            )
            rest_a = {
                k: v for k, v in (a or {}).items() if k not in ("vocab", "merges")
            }
            rest_b = {
                k: v for k, v in (b or {}).items() if k not in ("vocab", "merges")
            }
            add(
                "tokenizer.json:model (other settings)",
                _brief(rest_a),
                _brief(rest_b),
                rest_a == rest_b,
            )
            continue
        if key == "added_tokens":
            ids, am, bm = _by_id_diff(a, b, "id")
            add(
                "tokenizer.json:added_tokens",
                f"{len(a or [])} tokens",
                f"{len(b or [])} tokens",
                not ids,
                "; ".join(
                    f"id {i}: 4B {am.get(i)} | 0.6B {bm.get(i)}" for i in ids[:8]
                ),
            )
            continue
        add(f"tokenizer.json:{key}", _brief(a), _brief(b), a == b)

    tc = {
        k: json.loads((snaps[k] / "tokenizer_config.json").read_text(encoding="utf-8"))
        for k in MODELS
    }
    differing = json_top_level_diff(tc["4B"], tc["0.6B"])
    for key in differing:
        a, b = tc["4B"].get(key), tc["0.6B"].get(key)
        note = ""
        if (
            key == "added_tokens_decoder"
            and isinstance(a, dict)
            and isinstance(b, dict)
        ):
            ids = sorted(
                (i for i in set(a) | set(b) if a.get(i) != b.get(i)),
                key=lambda x: int(x) if str(x).isdigit() else 0,
            )
            note = "; ".join(
                f"id {i}: 4B {a.get(i)} | 0.6B {b.get(i)}" for i in ids[:8]
            )
        add(f"tokenizer_config.json:{key}", _brief(a), _brief(b), False, note)
    same_keys = sorted(set(tc["4B"]) & set(tc["0.6B"]) - set(differing))
    add(
        "tokenizer_config.json:identical keys",
        str(len(same_keys)),
        str(len(same_keys)),
        True,
        ", ".join(same_keys),
    )

    toks = {
        k: AutoTokenizer.from_pretrained(str(snaps[k]), padding_side="left")
        for k in MODELS
    }
    t4, t6 = toks["4B"], toks["0.6B"]
    add("loaded class", type(t4).__name__, type(t6).__name__)
    v4, v6 = t4.get_vocab(), t6.get_vocab()
    add("loaded vocabulary size", len(v4), len(v6))
    add("loaded vocabulary equal", "", "", v4 == v6)
    added4 = {int(k): str(v) for k, v in t4.added_tokens_decoder.items()}
    added6 = {int(k): str(v) for k, v in t6.added_tokens_decoder.items()}
    add("loaded added tokens", len(added4), len(added6), added4 == added6)
    add(
        "loaded special_tokens_map",
        json.dumps(t4.special_tokens_map, sort_keys=True),
        json.dumps(t6.special_tokens_map, sort_keys=True),
    )
    add(
        "pad token/id",
        f"{t4.pad_token}/{t4.pad_token_id}",
        f"{t6.pad_token}/{t6.pad_token_id}",
    )
    add(
        "eos token/id",
        f"{t4.eos_token}/{t4.eos_token_id}",
        f"{t6.eos_token}/{t6.eos_token_id}",
    )
    add("model_max_length", t4.model_max_length, t6.model_max_length)
    post4 = json.loads(t4.backend_tokenizer.to_str()).get("post_processor")
    post6 = json.loads(t6.backend_tokenizer.to_str()).get("post_processor")
    add("loaded post_processor", _brief(post4), _brief(post6), post4 == post6)

    # The token ids of every document under both tokenizers.
    log(
        "tokenising every canonical document with both tokenizers and comparing the token ids"
    )
    t0 = time.perf_counter()
    col = corpus.texts_column()
    n = corpus.n_rows
    counts06 = np.empty(n, dtype=np.int64)
    n_ids_equal = n_counts_equal = n_stored4 = n_stored6 = 0
    first_differences = []
    step = args.tokenize_batch_size
    for start in range(0, n, step):
        texts = col.slice(start, step).to_pylist()
        kw = dict(
            add_special_tokens=True,
            padding=False,
            truncation=False,
            return_attention_mask=False,
        )
        ids4 = t4(texts, **kw)["input_ids"]
        ids6 = t6(texts, **kw)["input_ids"]
        for j, (a, b) in enumerate(zip(ids4, ids6)):
            i = start + j
            counts06[i] = len(b)
            same = a == b
            n_ids_equal += same
            n_counts_equal += len(a) == len(b)
            n_stored4 += len(a) == int(corpus.n_tokens_4b[i])
            n_stored6 += len(b) == int(corpus.n_tokens_4b[i])
            if not same and len(first_differences) < 20:
                first_differences.append(
                    {
                        "doc_key": str(corpus.doc_keys[i]),
                        "n_4B": len(a),
                        "n_0p6B": len(b),
                    }
                )
        if (start // step) % 50 == 0:
            log(f"  {min(start + step, n)}/{n} documents compared")
    wall = time.perf_counter() - t0
    identical_ids = n_ids_equal == n
    add("corpus: documents compared", n, n, True)
    add(
        "corpus: token ids identical",
        n_ids_equal,
        n_ids_equal,
        identical_ids,
        f"{n - n_ids_equal} documents differ",
    )
    add(
        "corpus: token counts identical",
        n_counts_equal,
        n_counts_equal,
        n_counts_equal == n,
    )
    add(
        "corpus: equal to the stored n_tokens_4B",
        n_stored4,
        n_stored6,
        n_stored4 == n and n_stored6 == n,
        "the stored counts of the Phase 2 feasibility test (token_counts_all_documents.parquet)",
    )
    files_identical = all(r["identical"] for r in rows if r["item"].startswith("file:"))
    summary = {
        "snapshots": {k: str(v) for k, v in snaps.items()},
        "tokenizer_files_byte_identical": files_identical,
        "tokenizer_json_sections_differing": [
            r["item"]
            for r in rows
            if r["item"].startswith("tokenizer.json:") and not r["identical"]
        ],
        "tokenizer_config_keys_differing": differing,
        "vocabulary_equal": v4 == v6,
        "added_tokens_equal": added4 == added6,
        "post_processor_equal": post4 == post6,
        "n_documents": n,
        "n_token_ids_equal": n_ids_equal,
        "n_token_counts_equal": n_counts_equal,
        "n_equal_to_stored_counts_4B": n_stored4,
        "n_equal_to_stored_counts_0p6B": n_stored6,
        "token_ids_identical_on_every_document": identical_ids,
        "first_differences": first_differences,
        "call": "tokenizer(texts, add_special_tokens=True, padding=False, truncation=False)",
        "compare_wall_s": wall,
        "counts_used_for_0p6B": "the stored n_tokens_4B counts (identical token ids on every document)"
        if identical_ids
        else "results/token_counts_0p6b.parquet (the 0.6B tokenizer's own counts)",
        "versions": {
            p["name"]: p["version"]
            for p in package_versions(["transformers", "tokenizers"])
        },
    }
    return rows, summary, counts06


def stage_prepare(args):
    t_stage = time.perf_counter()
    os.environ["HF_HUB_OFFLINE"] = "1"
    rdir, edir = results_dir(args), environment_dir(args)
    rdir.mkdir(parents=True, exist_ok=True)
    edir.mkdir(parents=True, exist_ok=True)
    remove_outputs(
        [
            rdir / "prepare.json",
            rdir / "tokenizer_identity.csv",
            rdir / "token_counts_0p6b.parquet",
            edir / "vram_cap_build.txt",
            edir / "tei_image.json",
            edir / "docker_versions.txt",
            edir / "nvidia_smi.txt",
            edir / "model_snapshots.json",
            edir / "host.json",
        ]
    )
    log(
        "verifying the canonical table, the token table and the TEI test's summary against their sidecars"
    )
    checks = verify_inputs(args.project_dir)
    tei_summary = read_json(tei_search_summary_paths()[0])
    corpus = Corpus(args.project_dir)
    log(
        f"{corpus.n_rows} documents; tokenizer comparison of Qwen3-Embedding-0.6B with the 4B"
    )
    tok_rows, tok_summary, counts06 = tokenizer_identity(args, corpus)
    write_csv(
        rdir / "tokenizer_identity.csv",
        tok_rows,
        ["item", "value_4B", "value_0p6B", "identical", "note"],
    )
    if not tok_summary["token_ids_identical_on_every_document"]:
        import pyarrow as pa
        import pyarrow.parquet as pq

        pq.write_table(
            pa.table(
                {
                    "doc_key": pa.array(
                        [str(x) for x in corpus.doc_keys], type=pa.string()
                    ),
                    "n_tokens_0p6B": pa.array(counts06, type=pa.int64()),
                    "n_tokens_4B": pa.array(corpus.n_tokens_4b, type=pa.int64()),
                }
            ),
            rdir / "token_counts_0p6b.parquet",
            compression="zstd",
            compression_level=3,
        )
    log(
        f"  token ids identical on {tok_summary['n_token_ids_equal']}/{corpus.n_rows} documents; "
        f"tokenizer files byte-identical: {tok_summary['tokenizer_files_byte_identical']}; "
        f"differing tokenizer.json sections: {tok_summary['tokenizer_json_sections_differing']}"
    )

    log("building the benchmark stream")
    import pyarrow.compute as pc

    lengths = np.asarray(
        pc.fill_null(pc.utf8_length(corpus.texts_column()), 0).to_numpy(
            zero_copy_only=False
        ),
        dtype=np.int64,
    )
    excluded = [int(i) for i in np.flatnonzero(lengths == 0)]
    order = benchmark_order(corpus.doc_keys, excluded, STREAM_SEED)
    stream = {
        "seed": STREAM_SEED,
        "rule": "rows ordered by BLAKE2b-64 of '<seed>:<doc_key>' (ties by doc_key), the empty-text documents left out",
        "n_documents": int(len(order)),
        "excluded_rows": excluded,
        "excluded_doc_keys": [str(corpus.doc_keys[i]) for i in excluded],
        "midpoint": int(len(order) // 2),
        "order_sha256": stream_digest(corpus, order),
        "first_doc_keys": [str(corpus.doc_keys[i]) for i in order[:10]],
        "midpoint_doc_keys": [
            str(corpus.doc_keys[i])
            for i in order[len(order) // 2 : len(order) // 2 + 10]
        ],
        "sum_tokens_4B_counts": int(corpus.n_tokens_4b[order].sum()),
    }
    log(
        f"  {stream['n_documents']} documents; excluded {stream['excluded_doc_keys']}; midpoint position {stream['midpoint']}"
    )

    log(
        "building the cap shim and recording the image, Docker, driver, host and model facts"
    )
    shim = build_shim(args)
    (edir / "vram_cap_build.txt").write_text(shim["build_output"], encoding="utf-8")
    image = docker_image_record()
    write_json(edir / "tei_image.json", image)
    (edir / "docker_versions.txt").write_text(docker_versions_text(), encoding="utf-8")
    (edir / "nvidia_smi.txt").write_text(nvidia_smi_text(), encoding="utf-8")
    snapshots = {k: snapshot_record(args.hub_dir, k) for k in MODELS}
    write_json(edir / "model_snapshots.json", snapshots)
    host = host_record()
    write_json(edir / "host.json", host)

    prepare = {
        "stage": "prepare",
        "timestamp": utc_now(),
        "quick_run": bool(args.quick),
        "input_checks": checks,
        "tei_4b_search": {
            "largest_safe_max_batch_tokens": (
                tei_summary.get("largest_safe") or {}
            ).get("max_batch_tokens"),
            "confirmation_succeeded": tei_summary.get("confirmation_succeeded"),
            "nearest_failing_above": (
                tei_summary.get("nearest_failing_above") or {}
            ).get("max_batch_tokens"),
            "headline": tei_summary.get("headline"),
        },
        "tokenizer": tok_summary,
        "stream": stream,
        "shim_build": {k: v for k, v in shim.items() if k != "build_output"},
        "image": image,
        "timings": {"stage_total_s": time.perf_counter() - t_stage},
    }
    write_json(rdir / "prepare.json", prepare)

    sidecars = Sidecars(
        args.project_dir, gpu_parameters(args), "Throughput benchmark, prepare stage."
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
        sidecars.input_entry(
            tei_search_summary_paths()[0],
            "the TEI 4B max_batch_tokens search summary (92,160 confirmed)",
            checks["tei_4b_search_summary"]["sha256_measured"],
        ),
    ]
    tokenizer_inputs = [
        {
            "path": str(snapshot_path(args.hub_dir, k) / f),
            "hash": sha256_file(snapshot_path(args.hub_dir, k) / f),
            "format": "other",
            "role": f"tokenizer file of {MODELS[k]['repo']} @ {MODELS[k]['revision']}",
        }
        for k in MODELS
        for f in TOKENIZER_FILES
        if (snapshot_path(args.hub_dir, k) / f).exists()
    ]
    outputs = [
        (
            rdir / "prepare.json",
            "input checks, tokenizer comparison summary, benchmark stream, shim build, image",
        ),
        (
            rdir / "tokenizer_identity.csv",
            "0.6B tokenizer vs the tested 4B tokenizer: files, sections, vocabulary, special tokens, token ids of every document",
        ),
        (edir / "vram_cap_build.txt", "build log of the allocation-cap shim"),
        (edir / "tei_image.json", "identity of the TEI image used"),
        (
            edir / "docker_versions.txt",
            "Docker Engine and NVIDIA Container Toolkit versions",
        ),
        (edir / "nvidia_smi.txt", "nvidia-smi inside WSL2 at the prepare stage"),
        (
            edir / "model_snapshots.json",
            "the model files of both snapshots (blob names, sizes, hashes)",
        ),
        (edir / "host.json", "the WSL2 VM: CPUs, memory, swap, kernel, .wslconfig"),
    ]
    if (rdir / "token_counts_0p6b.parquet").exists():
        outputs.append(
            (
                rdir / "token_counts_0p6b.parquet",
                "token counts of every document under the 0.6B tokenizer",
            )
        )
    for path, note in outputs:
        sidecars.write(
            path,
            inputs
            + (
                tokenizer_inputs
                if "tokenizer" in path.name or path.suffix == ".parquet"
                else []
            ),
            notes=note,
        )
    snap06 = edir / "model_snapshot_0p6b.json"
    download_script = BENCH_DIR / "scripts" / "download_qwen3_embedding_0p6b.py"
    if snap06.exists() and download_script.exists() and not args.quick:
        Sidecars(
            args.project_dir,
            {"repo": MODELS["0.6B"]["repo"], "revision": MODELS["0.6B"]["revision"]},
            "",
            script_path=download_script,
        ).write(
            snap06,
            [],
            notes="download and hash verification of Qwen3-Embedding-0.6B at the pinned revision (every file equal to the Hub's hashes)",
        )
    log(f"prepare done in {time.perf_counter() - t_stage:.1f} s")
    return prepare


# ----------------------------------------------------------------------------------------------
# GPU: one TEI container under the cap
# ----------------------------------------------------------------------------------------------


def make_watchdog(ctx, name, report_path):
    """The NVML watchdog: kills the container when a sample exceeds the ceiling; after a shim
    refusal the device gets `watchdog_grace_s` to come back under the ceiling first, so that the
    refusal is the recorded outcome."""
    args = ctx["args"]
    state = {
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
            state["first_over_t"] = None
            return
        now = time.time()
        state["over_samples"] += 1
        state["over_peak"] = max(state["over_peak"], used)
        if state["first_over_t"] is None:
            state["first_over_t"] = now
        if state["killed"]:
            return
        if now - state["last_report_check"] > 0.1:
            state["last_report_check"] = now
            rep = read_shim_report(report_path)
            if rep and int(rep.get("n_refusals", 0)) > 0:
                state["refusal_seen"] = True
        if (
            state["refusal_seen"]
            and now - state["first_over_t"] < args.watchdog_grace_s
        ):
            return
        state["killed"] = True
        state["used_at_kill"] = used
        log(
            f"  WATCHDOG: device total {mib(used)} MiB exceeds the ceiling"
            f"{' after a shim refusal' if state['refusal_seen'] else ''}; killing {name}"
        )
        docker_kill(name)

    return state, on_sample


class TeiContainer:
    """One TEI container under the allocation-cap shim, with the NVML sampler and watchdog, a
    sampler of the container's cgroup memory and one of the VM's memory in use."""

    def __init__(self, ctx, name, model_key, max_batch_tokens, cap_bytes, report_path):
        self.ctx = ctx
        self.name = name
        self.model_key = model_key
        self.max_batch_tokens = int(max_batch_tokens)
        self.cap_bytes = int(cap_bytes)
        self.report_path = Path(report_path)
        self.cgroup = None
        self.mem_sampler = None
        self.container_memory = {}

    def start(self):
        args = self.ctx["args"]
        self.watchdog, on_sample = make_watchdog(self.ctx, self.name, self.report_path)
        self.nvml_sampler = PeakSampler(
            self.ctx["nvml"].used,
            interval_s=args.nvml_interval_ms / 1000.0,
            on_sample=on_sample,
        ).start()
        self.vm_sampler = PeakSampler(vm_used_bytes, interval_s=0.1).start()
        self.t_run = time.time()
        self.command = docker_run_tei(
            self.ctx,
            self.name,
            self.model_key,
            self.max_batch_tokens,
            self.cap_bytes,
            self.report_path,
        )
        st = docker_state(self.name)
        if st.get("pid"):
            self.cgroup = cgroup_dir_of_pid(st["pid"])
        if self.cgroup:
            cg = self.cgroup
            self.mem_sampler = PeakSampler(
                lambda: cgroup_int(cg, "memory.current"), interval_s=0.1
            ).start()
        state, startup_s, dstate = wait_ready(
            self.ctx, self.name, args.startup_timeout_s
        )
        self.t_ready = time.time()
        self.startup_state = state
        self.startup_s = startup_s
        self.state_after_startup = dstate
        return state == "ready"

    def finish(self):
        """Reads the container's memory record, stops the container gracefully (the shim writes
        its exit record), collects the log and removes the container."""
        if self.cgroup:
            stat = cgroup_keyed(self.cgroup, "memory.stat")
            self.container_memory = {
                "cgroup": self.cgroup,
                "memory_peak_bytes": cgroup_int(self.cgroup, "memory.peak"),
                "memory_current_bytes": cgroup_int(self.cgroup, "memory.current"),
                "memory_stat": {
                    k: stat.get(k)
                    for k in ("anon", "file", "kernel", "shmem", "file_mapped")
                },
                "memory_events": cgroup_keyed(self.cgroup, "memory.events"),
            }
        if docker_state(self.name)["running"]:
            try:
                docker_stop(self.name, 60)
            except Exception as e:
                log(f"  docker stop failed: {e!r}; killing")
                docker_kill(self.name)
                time.sleep(2)
        self.t_stopped = time.time()
        self.logs_text = docker_logs(self.name)
        self.final_state = docker_state(self.name)
        docker_remove(self.name)
        self.nvml_sampler.stop()
        self.vm_sampler.stop()
        if self.mem_sampler:
            self.mem_sampler.stop()
        self.shim_final = read_shim_report(self.report_path)
        lines = self.logs_text.splitlines()
        self.log_events = parse_tei_log_events(lines)
        self.log_errors = tei_log_errors(lines)
        self.shim_refusals = (
            int(self.shim_final.get("n_refusals", 0)) if self.shim_final else 0
        )
        self.refused = self.shim_refusals > 0 or any(
            "[vram_cap] REFUSED" in ln for ln in lines
        )
        self.below_minimum = any(
            "exceeds `--max-batch-tokens" in ln or "exceeds --max-batch-tokens" in ln
            for ln in lines
        )
        return self

    def memory_summary(self):
        mem = dict(self.container_memory)
        if self.mem_sampler:
            mem["memory_current_peak_sampled_bytes"] = self.mem_sampler.peak
            mem["n_samples"] = self.mem_sampler.n_samples
        mem["vm_used_peak_bytes"] = self.vm_sampler.peak
        mem["vm_used_first_bytes"] = self.vm_sampler.first
        return mem


def failure_detail(outcome, errors, shim_final, watchdog, overall_peak, extra=""):
    if outcome == "refused_below_minimum":
        return next(
            (ln for ln in errors if "max-batch-tokens" in ln), "; ".join(errors)[:600]
        )
    if outcome in ("startup_refused_at_cap", "batch_refused_at_cap"):
        return (
            (shim_final or {}).get("last_refusal", "") or "; ".join(errors)[:600]
        ) + extra
    if outcome == "watchdog_kill":
        return f"device total reached {mib(watchdog['used_at_kill'])} MiB"
    if outcome == "device_peak_over_ceiling":
        return f"device total peaked at {mib(overall_peak['peak'])} MiB"
    if outcome in ("startup_error", "batch_error"):
        return ("; ".join(errors)[:600] or extra)[:800]
    return extra


def gpu_calibration_path(args, model_key):
    return results_dir(args) / f"gpu_calibration_{MODELS[model_key]['slug']}.json"


def longest_rows(corpus, counts, n):
    """The n longest documents (tokens desc, chars desc, doc_key asc) with their texts."""
    idx = rank_candidates(counts, corpus.n_chars, corpus.doc_keys, n)
    texts = corpus.texts(idx)
    return [
        {
            "row": int(i),
            "doc_key": str(corpus.doc_keys[i]),
            "n_tokens": int(counts[i]),
            "n_chars": int(corpus.n_chars[i]),
            "text": t,
        }
        for i, t in zip(idx, texts)
    ]


def stage_gpu_calibrate(args, model_key):
    """One container at the model's minimum budget under a generous cap: the untracked device
    overhead (CUDA context and module memory no allocation call reports), /info, TEI's own token
    counts of the search candidates and of a stream sample from the measured region, the
    embedding dimension."""
    spec = MODELS[model_key]
    slug = spec["slug"]
    rdir = results_dir(args)
    out_json = gpu_calibration_path(args, model_key)
    out_csv = rdir / f"gpu_calibration_{slug}_tokenize_check.csv"
    out_log = rdir / "container_logs" / f"calibration_{slug}.log"
    remove_outputs([out_json, out_csv, out_log])
    corpus = Corpus(args.project_dir)
    counts = model_token_counts(args, corpus, model_key)
    order, st = benchmark_stream(args, corpus)
    log(
        f"calibration {model_key}: selecting the {args.n_candidates} longest documents and a stream sample"
    )
    candidates = candidate_rows(corpus, counts, args.n_candidates)
    sample_rows = order[st["midpoint"] : st["midpoint"] + args.tokenize_sample]
    sample_texts = corpus.texts(sample_rows)
    nvml = Nvml()
    ctx = make_ctx(args, nvml)
    Path(ctx["shim_report_dir"]).mkdir(parents=True, exist_ok=True)
    docker_remove_stale()
    name = f"{CONTAINER_PREFIX}_calibration_{slug}"
    report_path = Path(ctx["shim_report_dir"]) / f"calibration_{slug}.json"
    if report_path.exists():
        report_path.unlink()
    cap = int(args.calibration_cap_gib * 1073741824)
    device_before = nvml.used()
    log(
        f"  container at max_batch_tokens={spec['n_min']}, cap {mib(cap)} MiB, device before {mib(device_before)} MiB"
    )
    tc = TeiContainer(ctx, name, model_key, spec["n_min"], cap, report_path)
    if not tc.start():
        tc.finish()
        raise SystemExit(
            f"calibration container did not become ready ({tc.startup_state}):\n{tc.logs_text[-4000:]}"
        )
    info = get_info(ctx)
    time.sleep(3.0)
    device_steady = nvml.used()
    shim_steady = read_shim_report(report_path)
    shim_physical = int(shim_steady.get("physical_bytes", shim_steady["used_bytes"]))
    untracked = (device_steady - device_before) - shim_physical
    info_checks = {
        "auto_truncate_false": info.get("auto_truncate") is False,
        "max_input_length": info.get("max_input_length"),
        "max_input_length_equals_n_min": info.get("max_input_length") == spec["n_min"],
        "max_batch_tokens_equals_requested": info.get("max_batch_tokens")
        == spec["n_min"],
        "max_batch_requests_unset": info.get("max_batch_requests") is None,
        "max_client_batch_size": info.get("max_client_batch_size"),
        "max_concurrent_requests": info.get("max_concurrent_requests"),
        "dtype_float16": info.get("model_dtype") == "float16",
        "pooling_last_token": (info.get("model_type") or {})
        .get("embedding", {})
        .get("pooling")
        == "last_token",
        "tokenization_workers": info.get("tokenization_workers"),
        "version": info.get("version"),
        "sha": info.get("sha"),
        "docker_label": info.get("docker_label"),
    }
    log(
        f"  ready in {tc.startup_s:.1f} s; device steady {mib(device_steady)} MiB; shim physical {mib(shim_physical)} MiB; "
        f"untracked overhead {mib(untracked)} MiB; /info dtype {info.get('model_dtype')}, max_input_length {info.get('max_input_length')}"
    )
    t0 = time.perf_counter()
    check_rows = []
    n_equal = 0
    for label, docs in (
        ("candidate", candidates),
        (
            "stream_sample",
            [
                {
                    "rank": k + 1,
                    "pool": "stream",
                    "doc_key": str(corpus.doc_keys[i]),
                    "record_type": str(corpus.record_types[i]),
                    "n_chars": int(corpus.n_chars[i]),
                    "n_tokens": int(counts[i]),
                    "text": t,
                }
                for k, (i, t) in enumerate(zip(sample_rows, sample_texts))
            ],
        ),
    ):
        for start in range(0, len(docs), args.tokenize_chunk):
            chunk = docs[start : start + args.tokenize_chunk]
            tokens = post_tokenize(ctx, [c["text"] for c in chunk])
            for c, tk in zip(chunk, tokens):
                equal = len(tk) == c["n_tokens"]
                n_equal += equal
                check_rows.append(
                    {
                        "set": label,
                        "rank": c["rank"],
                        "pool": c["pool"],
                        "doc_key": c["doc_key"],
                        "record_type": c["record_type"],
                        "n_chars": c["n_chars"],
                        "n_tokens_stored": c["n_tokens"],
                        "n_tokens_tei": len(tk),
                        "equal": equal,
                    }
                )
    t_tokenize = time.perf_counter() - t0
    log(
        f"  TEI token counts equal to the stored counts for {n_equal}/{len(check_rows)} documents"
    )
    special = [
        t for t in post_tokenize(ctx, [candidates[0]["text"]])[0] if t.get("special")
    ]
    embed = post_embed(
        ctx, [candidates[0]["text"]], args.request_timeout_s, keep_body=True
    )
    dim = None
    if embed["status"] == 200:
        dim = int(embeddings_from_body(embed["body"]).shape[1])
    embed["body"] = None
    tc.finish()
    (rdir / "container_logs").mkdir(parents=True, exist_ok=True)
    out_log.write_text(tc.logs_text, encoding="utf-8")
    write_csv(out_csv, check_rows)
    cal = {
        "stage": "gpu_calibrate",
        "model": model_key,
        "timestamp": utc_now(),
        "quick_run": bool(args.quick),
        "container": name,
        "docker_run_command": tc.command,
        "max_batch_tokens": spec["n_min"],
        "cap_bytes": cap,
        "device_used_before_start_bytes": device_before,
        "startup_wait_s": tc.startup_s,
        "log_events": tc.log_events,
        "device_used_steady_bytes": device_steady,
        "shim_steady": shim_steady,
        "shim_final": tc.shim_final,
        "untracked_overhead_bytes": untracked,
        "startup_peak_device_bytes": summarise_samples(
            tc.nvml_sampler.samples, tc.t_run, tc.t_ready
        )["peak"],
        "container_memory": tc.memory_summary(),
        "info": info,
        "info_checks": info_checks,
        "candidate_doc_keys": [c["doc_key"] for c in candidates],
        "tokenize_check": {
            "n_documents": len(check_rows),
            "n_candidates": len(candidates),
            "n_stream_sample": len(sample_texts),
            "n_equal": n_equal,
            "all_equal": n_equal == len(check_rows),
            "special_tokens_of_longest": special,
            "wall_s": t_tokenize,
        },
        "embedding_dim": dim,
        "single_document_embed": embed,
    }
    write_json(out_json, cal)
    nvml.shutdown()
    sidecars = Sidecars(
        args.project_dir,
        gpu_parameters(args),
        f"Throughput benchmark, GPU calibration of the {model_key} model.",
    )
    inputs = [
        sidecars.input_entry(
            rdir / "prepare.json", "prepare stage result (stream, tokenizer comparison)"
        ),
        sidecars.input_entry(
            canonical_paths(args.project_dir)[0], "canonical Phase 1 table, read-only"
        ),
    ]
    for path, note in (
        (
            out_json,
            "calibration run: untracked overhead, /info, token-count check, embedding dimension",
        ),
        (
            out_csv,
            "stored token count vs TEI /tokenize count for the candidates and a stream sample",
        ),
        (out_log, "container log of the calibration run"),
    ):
        sidecars.write(path, inputs, notes=note)
    return cal


# --- one-batch tests (the 0.6B search) ----------------------------------------------------------


def submit_one_batch(ctx, blockers, batch_texts, batch_tokens, k, timeout_s, gap_s):
    """Two single-document blockers then the batch; TEI forms the batch documents as one batch
    because they are all queued while the first blocker runs and the second waits in TEI's
    one-batch channel. Verified from the /metrics deltas and x-compute-tokens."""
    before = get_metrics(ctx)
    results = {}

    def send(label, texts, keep):
        results[label] = post_embed(ctx, texts, timeout_s, keep_body=keep)

    threads = [
        threading.Thread(target=send, args=("blocker_1", [blockers[0]["text"]], False)),
        threading.Thread(target=send, args=("blocker_2", [blockers[1]["text"]], False)),
        threading.Thread(target=send, args=("batch", batch_texts, True)),
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
    failures = {}
    for labels, value in after.get("te_request_failure", []):
        failures[labels.get("err", "?")] = value - counter_value(
            before, "te_request_failure", labels
        )
    batch = results["batch"]
    ok_http = all(
        results[k2]["status"] == 200 for k2 in ("blocker_1", "blocker_2", "batch")
    )
    expected = [
        (1, blockers[0]["n_tokens"]),
        (1, blockers[1]["n_tokens"]),
        (k, batch_tokens),
    ]
    verified, checks, info = verify_interval(size_delta, tokens_delta, expected)
    compute_tokens = header_int(batch["headers"], "x-compute-tokens")
    n_emb = dim = finite = None
    if batch["status"] == 200 and batch["body"] is not None:
        emb = embeddings_from_body(batch["body"])
        n_emb, dim, finite = (
            int(emb.shape[0]),
            int(emb.shape[1]),
            bool(np.isfinite(emb).all()),
        )
        del emb
    batch["body"] = None
    return {
        "t_start": t_start,
        "t_batch_sent": t_batch_sent,
        "t_end": t_end,
        "requests": results,
        "all_requests_ok": ok_http,
        "metrics_delta": {
            "te_batch_next_size": size_delta,
            "te_batch_next_tokens": tokens_delta,
            "te_request_failure": failures,
        },
        "embeddings": {"n": n_emb, "dim": dim, "all_finite": finite},
        "verification": {
            "single_batch_verified": bool(
                verified and ok_http and compute_tokens == batch_tokens and n_emb == k
            ),
            "histogram_checks": checks,
            "histogram_info": info,
            "x_compute_tokens": compute_tokens,
            "x_compute_tokens_equals_batch_total": compute_tokens == batch_tokens,
            "n_embeddings_equals_documents": n_emb == k,
        },
    }


TESTED_COLUMNS = [
    "seq",
    "purpose",
    "max_batch_tokens",
    "outcome",
    "success",
    "failure_at_ceiling",
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
    "batch_peak_device_mib",
    "overall_peak_device_mib",
    "overall_peak_under_ceiling",
    "overall_peak_under_21e9_bytes",
    "shim_peak_mib",
    "shim_physical_peak_mib",
    "shim_overshoot_mib",
    "shim_refusals",
    "watchdog_killed",
    "enforcement",
    "device_over_ceiling_ms",
    "device_over_ceiling_peak_mib",
    "container_memory_peak_mib",
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


def tested_row(rec):
    batch = (rec.get("batch_attempts") or [None])[-1]
    breq = batch["requests"]["batch"] if batch else {}
    headers = breq.get("headers", {}) if batch else {}
    shim = rec.get("shim_final") or {}
    peak = rec["nvml"]["overall"]["peak"]
    return {
        "seq": rec["seq"],
        "purpose": rec["purpose"],
        "max_batch_tokens": rec["max_batch_tokens"],
        "outcome": rec["outcome"],
        "success": rec["success"],
        "failure_at_ceiling": rec["failure_at_ceiling"],
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
        "x_compute_tokens": header_int(headers, "x-compute-tokens"),
        "verified_single_batch": rec["verified_single_batch"],
        "attempts": len(rec.get("batch_attempts") or []),
        "batch_wall_s": round(breq["wall_s"], 3)
        if batch and breq.get("wall_s") is not None
        else None,
        "tei_inference_ms": header_int(headers, "x-inference-time"),
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
        "shim_overshoot_mib": mib(shim.get("max_overshoot_bytes", 0)) if shim else None,
        "shim_refusals": rec["shim_refusals"],
        "watchdog_killed": rec["watchdog"]["killed"],
        "enforcement": enforcement_label(rec),
        "device_over_ceiling_ms": int(
            round(rec["watchdog"].get("over_samples", 0) * rec["nvml"]["interval_ms"])
        ),
        "device_over_ceiling_peak_mib": mib(rec["watchdog"]["over_peak"])
        if rec["watchdog"].get("over_peak")
        else None,
        "container_memory_peak_mib": mib(rec["container_memory"]["memory_peak_bytes"])
        if (rec.get("container_memory") or {}).get("memory_peak_bytes")
        else None,
        "embedding_dim": (batch or {}).get("embeddings", {}).get("dim")
        if batch
        else None,
        "failure_detail": rec["failure_detail"],
    }


def run_search_test(
    ctx, seq, model_key, max_batch_tokens, purpose, candidates, calibration, sdir
):
    """One tested value in one fresh container; returns the full record and writes its files."""
    args = ctx["args"]
    nvml = ctx["nvml"]
    name = f"{CONTAINER_PREFIX}_search_{seq:02d}"
    stem = f"test_{seq:02d}_{max_batch_tokens}"
    report_path = Path(ctx["shim_report_dir"]) / f"search_{stem}.json"
    if report_path.exists():
        report_path.unlink()
    selected, batch_tokens = fill_batch(candidates, max_batch_tokens)
    blockers = candidates[:2]
    rec = {
        "seq": seq,
        "purpose": purpose,
        "model": model_key,
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
            "blocker_doc_keys": [b["doc_key"] for b in blockers],
            "blocker_tokens": [b["n_tokens"] for b in blockers],
        },
    }
    settle_used = settle_device(
        ctx,
        ctx["session_baseline_used"],
        args.settle_margin_mib * 1048576,
        args.settle_s,
    )
    device_before = nvml.used()
    cap = compute_cap_bytes(
        ctx["ceiling_bytes"], device_before, calibration["untracked_overhead_bytes"]
    )
    rec.update(
        {
            "device_used_after_settle_bytes": settle_used,
            "device_used_before_start_bytes": device_before,
            "cap_bytes": cap,
        }
    )
    log(
        f"test {seq:02d} [{purpose}] max_batch_tokens={max_batch_tokens}: K={len(selected)} docs, {batch_tokens} tokens; device before {mib(device_before)} MiB; cap {mib(cap)} MiB"
    )
    tc = TeiContainer(ctx, name, model_key, max_batch_tokens, cap, report_path)
    started = tc.start()
    rec["docker_run_command"] = tc.command
    batch = None
    info = None
    if started:
        try:
            info = get_info(ctx)
        except Exception as e:
            info = {"error": repr(e)}
        rec["shim_at_ready"] = read_shim_report(report_path)
        time.sleep(1.0)
        rec["device_used_at_ready_bytes"] = nvml.used()
        attempts = []
        for attempt in range(1, args.attempts + 1):
            log(
                f"  batch attempt {attempt}: 2 blockers + {len(selected)} documents ({batch_tokens} tokens)"
            )
            result = submit_one_batch(
                ctx,
                blockers,
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
                b = result["requests"]["batch"]
                log(
                    f"  request failure: batch status {b['status']} {str(b.get('error') or b.get('exception'))[:300]}"
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
    tc.finish()
    samples = tc.nvml_sampler.samples
    batch_ok = bool(batch and batch["all_requests_ok"])
    verified = bool(batch and batch["verification"]["single_batch_verified"])
    startup_peak = summarise_samples(samples, tc.t_run, tc.t_ready)
    batch_peak = (
        summarise_samples(samples, batch["t_start"], batch["t_end"])
        if batch
        else summarise_samples([], None, None)
    )
    overall_peak = summarise_samples(samples, tc.t_run, tc.t_stopped)
    peak_over = (
        overall_peak["peak"] is not None and overall_peak["peak"] > ctx["ceiling_bytes"]
    )
    outcome, success, failure_at_ceiling = classify_outcome(
        started,
        (not started) and tc.below_minimum,
        tc.refused,
        batch_ok,
        verified,
        tc.watchdog["killed"],
        peak_over,
    )
    extra = ""
    if batch and batch["requests"]["batch"].get("error"):
        extra = f" | TEI: {str(batch['requests']['batch']['error'])[:400]}"
    elif outcome == "inconclusive_unverified_batch" and batch:
        extra = json.dumps(batch["verification"]["histogram_checks"])
    rec.update(
        {
            "timestamp_end": utc_now(),
            "started": started,
            "startup_state": tc.startup_state,
            "startup_wait_s": tc.startup_s,
            "info": info,
            "log_events": tc.log_events,
            "log_errors": tc.log_errors,
            "shim_final": tc.shim_final,
            "shim_refusals": tc.shim_refusals,
            "refused_at_cap": tc.refused,
            "watchdog": tc.watchdog,
            "container_final_state": tc.final_state,
            "container_memory": tc.memory_summary(),
            "nvml": {
                "interval_ms": args.nvml_interval_ms,
                "startup": startup_peak,
                "batch": batch_peak,
                "overall": overall_peak,
                "series_1s_max": [
                    (round(t - tc.t_run, 3), v) for t, v in downsample_max(samples, 1.0)
                ],
            },
            "batch_ok": batch_ok,
            "verified_single_batch": verified,
            "peak_over_ceiling": peak_over,
            "outcome": outcome,
            "success": success,
            "failure_at_ceiling": failure_at_ceiling,
            "failure_detail": failure_detail(
                outcome, tc.log_errors, tc.shim_final, tc.watchdog, overall_peak, extra
            ),
        }
    )
    for sub in ("container_logs", "shim_reports", "tests"):
        (sdir / sub).mkdir(parents=True, exist_ok=True)
    (sdir / "container_logs" / f"{stem}.log").write_text(tc.logs_text, encoding="utf-8")
    if tc.shim_final is not None:
        write_json(sdir / "shim_reports" / f"{stem}.json", tc.shim_final)
    write_json(sdir / "tests" / f"{stem}.json", rec)
    append_csv_row(sdir / "tested_values.csv", tested_row(rec), TESTED_COLUMNS)
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
            sdir / "batch_compositions.csv",
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
        f"  outcome: {outcome}; peak device {mib(overall_peak['peak']) if overall_peak['peak'] else None} MiB; {rec['failure_detail'][:200]}"
    )
    return rec


def search_summary(tests, n_min, args, aborted, ctx):
    successes = [t for t in tests if t["success"]]
    failures = [t for t in tests if t["failure_at_ceiling"]]
    safe = max(successes, key=lambda t: t["max_batch_tokens"]) if successes else None
    confirmation = next(
        (
            t
            for t in tests
            if t["purpose"] == "confirmation"
            and safe
            and t["max_batch_tokens"] == safe["max_batch_tokens"]
            and t["success"]
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
        else f"largest measured safe max_batch_tokens = {safe['max_batch_tokens']} (peak {mib(safe['nvml']['overall']['peak'])} MiB); nearest failing above = {failing_above['max_batch_tokens'] if failing_above else 'none measured'}; confirmation {'succeeded' if confirmation else 'not confirmed'}"
    )
    return {
        "stage": "gpu_search",
        "timestamp": utc_now(),
        "n_min": n_min,
        "resolution": args.resolution,
        "growth": args.growth,
        "ceiling_bytes": ctx["ceiling_bytes"],
        "n_tests": len(tests),
        "tested_values_in_order": [
            (t["seq"], t["purpose"], t["max_batch_tokens"], t["outcome"]) for t in tests
        ],
        "largest_safe": brief(safe),
        "confirmation": brief(confirmation),
        "confirmation_succeeded": bool(confirmation),
        "nearest_failing_above": brief(failing_above),
        "gap_tokens": (failing_above["max_batch_tokens"] - safe["max_batch_tokens"])
        if safe and failing_above
        else None,
        "non_monotonic_failures_below_safe": non_monotonic,
        "aborted": aborted,
        "headline": headline,
    }


def stage_gpu_search(args):
    model_key = "0.6B"
    slug = MODELS[model_key]["slug"]
    t_stage = time.perf_counter()
    sdir = stage_dir(args, f"gpu_search_{slug}")
    remove_outputs(
        [
            sdir / "tested_values.csv",
            sdir / "batch_compositions.csv",
            sdir / "search_summary.json",
        ]
    )
    for sub in ("tests", "container_logs", "shim_reports"):
        if (sdir / sub).exists():
            for p in (sdir / sub).glob("test_*"):
                p.unlink()
    calibration = read_json(gpu_calibration_path(args, model_key))
    if not calibration["tokenize_check"]["all_equal"]:
        raise SystemExit(
            "TEI's token counts differ from the stored counts for some documents; see the calibration check"
        )
    corpus = Corpus(args.project_dir)
    counts = model_token_counts(args, corpus, model_key)
    log("reading the candidate texts")
    candidates = candidate_rows(corpus, counts, args.n_candidates)
    if [c["doc_key"] for c in candidates] != calibration["candidate_doc_keys"]:
        raise SystemExit(
            "the candidate list differs from the calibration's; rerun gpu_calibrate"
        )
    nvml = Nvml()
    ctx = make_ctx(args, nvml)
    Path(ctx["shim_report_dir"]).mkdir(parents=True, exist_ok=True)
    docker_remove_stale()
    ctx["session_baseline_used"] = nvml.used()
    n_min = int(calibration["info"]["max_input_length"])
    log(
        f"search {model_key}: n_min={n_min}, resolution {args.resolution}, ceiling {args.ceiling_mib} MiB, untracked overhead {mib(calibration['untracked_overhead_bytes'])} MiB, session baseline {mib(ctx['session_baseline_used'])} MiB"
    )
    tests = []
    seq = 1
    aborted = None
    inconclusive_repeats = {}
    confirmed = set()
    while seq <= args.max_tests:
        conclusive = [t for t in tests if t["success"] or t["failure_at_ceiling"]]
        value, purpose = next_test_value(
            conclusive, n_min, args.resolution, args.growth
        )
        if value is None:
            if purpose == "bracketed within resolution":
                safe = max(t["max_batch_tokens"] for t in conclusive if t["success"])
                if safe in confirmed:
                    break
                rec = run_search_test(
                    ctx,
                    seq,
                    model_key,
                    safe,
                    "confirmation",
                    candidates,
                    calibration,
                    sdir,
                )
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
                key = ("confirmation", safe)
                inconclusive_repeats[key] = inconclusive_repeats.get(key, 0) + 1
                if inconclusive_repeats[key] > 1:
                    aborted = f"confirmation of {safe} was inconclusive twice: {rec['outcome']} {rec['failure_detail']}"
                    break
                continue
            aborted = purpose
            break
        rec = run_search_test(
            ctx, seq, model_key, value, purpose, candidates, calibration, sdir
        )
        seq += 1
        tests.append(rec)
        if not rec["success"] and not rec["failure_at_ceiling"]:
            key = (purpose, value)
            inconclusive_repeats[key] = inconclusive_repeats.get(key, 0) + 1
            if inconclusive_repeats[key] > 1:
                aborted = f"test of {value} was inconclusive twice: {rec['outcome']} {rec['failure_detail']}"
                break
            log(f"  inconclusive ({rec['outcome']}); repeating the value once")
    if seq > args.max_tests and aborted is None and not confirmed:
        aborted = f"stopped after {args.max_tests} tests"
    for t in tests:
        if t.get("invalidated_by_confirmation"):
            write_json(
                sdir / "tests" / f"test_{t['seq']:02d}_{t['max_batch_tokens']}.json", t
            )
    write_csv(
        sdir / "tested_values.csv", [tested_row(t) for t in tests], TESTED_COLUMNS
    )
    summary = search_summary(tests, n_min, args, aborted, ctx)
    summary["wall_s"] = time.perf_counter() - t_stage
    write_json(sdir / "search_summary.json", summary)
    nvml.shutdown()
    sidecars = Sidecars(
        args.project_dir,
        dict(
            gpu_parameters(args),
            search=f"start at the TEI minimum ({n_min}); multiply by {args.growth} until failure; bisect on a {args.resolution}-token grid; confirmation rerun of the safe value in a fresh container; batches first-fit-decreasing over the {args.n_candidates} longest documents plus one per shorter length",
        ),
        f"Throughput benchmark, {model_key} max_batch_tokens search.",
    )
    inputs = [
        sidecars.input_entry(
            gpu_calibration_path(args, model_key),
            "calibration of the 0.6B (untracked overhead, candidates)",
        ),
        sidecars.input_entry(
            canonical_paths(args.project_dir)[0], "canonical Phase 1 table, read-only"
        ),
    ]
    for path in (
        sorted((sdir / "tests").glob("test_*.json"))
        + sorted((sdir / "container_logs").glob("test_*.log"))
        + sorted((sdir / "shim_reports").glob("test_*.json"))
    ):
        sidecars.write(
            path, inputs, notes=f"one tested max_batch_tokens value: {path.name}"
        )
    for path, note in (
        (sdir / "tested_values.csv", "one row per tested value"),
        (sdir / "batch_compositions.csv", "the documents of every test batch"),
        (
            sdir / "search_summary.json",
            "the largest measured safe value, the nearest failing value above it, the confirmation",
        ),
    ):
        if path.exists():
            sidecars.write(path, inputs, notes=note)
    log(f"search done in {summary['wall_s'] / 60:.1f} min: {summary['headline']}")
    if aborted:
        log(f"SEARCH ABORTED: {aborted}")
    return summary


# --- pipelined throughput runs ------------------------------------------------------------------


def resolve_gpu_budget(args, model_key):
    """The GPU token budget of a model: the command line in a quick run; for the 4B the confirmed
    safe value of the TEI test (92,160); for the 0.6B the confirmed safe value of gpu_search."""
    if args.budget:
        return int(args.budget), "command line (--budget)"
    if model_key == "4B":
        s = read_json(tei_search_summary_paths()[0])
        if not s.get("confirmation_succeeded") or not s.get("largest_safe"):
            raise SystemExit("the TEI 4B search summary has no confirmed safe value")
        return int(s["largest_safe"]["max_batch_tokens"]), rel_path(
            tei_search_summary_paths()[0], args.project_dir
        ) + " (largest safe value, confirmed in a fresh container)"
    path = (
        results_dir(args)
        / f"gpu_search_{MODELS[model_key]['slug']}"
        / "search_summary.json"
    )
    s = read_json(path)
    if not s.get("confirmation_succeeded") or not s.get("largest_safe"):
        raise SystemExit("the 0.6B search has no confirmed safe value")
    return int(s["largest_safe"]["max_batch_tokens"]), rel_path(
        path, args.project_dir
    ) + " (largest safe value, confirmed in a fresh container)"


def run_pipeline(ctx, name, blockers, batches):
    """Sends the planned batches so that TEI forms each one exactly and the GPU runs them back to
    back: two single-document blockers, then batch 1; each later batch j is sent as soon as every
    earlier document has been batched (seen in /metrics), which is while the backend is still
    running an earlier batch; TEI then forms batch j as one batch the moment the backend takes the
    batch before it. Returns every response (the bodies of measured batches kept), the /metrics
    snapshot at each release and the verification of every batch."""
    args = ctx["args"]
    poll_client = http_client(ctx, 30.0)

    def metrics_now():
        return parse_prometheus(poll_client.get("/metrics").text)

    results = {}
    threads = []

    def send(label, texts, keep):
        results[label] = post_embed(ctx, texts, args.request_timeout_s, keep_body=keep)

    def launch(label, texts, keep):
        th = threading.Thread(target=send, args=(label, texts, keep), daemon=True)
        th.start()
        threads.append(th)

    snaps = []

    def snapshot(m, t):
        snaps.append(
            {
                "t": t,
                "size": histogram(m, "te_batch_next_size"),
                "tokens": histogram(m, "te_batch_next_tokens"),
            }
        )

    def wait_for_docs(target):
        t0 = time.time()
        last_state_check = t0
        while time.time() - t0 < args.pipeline_timeout_s:
            try:
                m = metrics_now()
                if histogram(m, "te_batch_next_size")["sum"] >= target - 0.5:
                    return m, time.time(), None
            except Exception as e:
                last_error = repr(e)
                time.sleep(0.05)
            else:
                last_error = None
            if time.time() - last_state_check > 1.0:
                last_state_check = time.time()
                if not docker_state(name)["running"]:
                    return None, None, f"container not running ({last_error})"
            time.sleep(args.poll_ms / 1000.0)
        return (
            None,
            None,
            f"timeout after {args.pipeline_timeout_s} s waiting for {target} batched documents",
        )

    m0 = metrics_now()
    t_start = time.time()
    snapshot(m0, t_start)
    base = histogram(m0, "te_batch_next_size")["sum"]
    gap = args.blocker_gap_ms / 1000.0
    launch("blocker_1", [blockers[0]["text"]], False)
    time.sleep(gap)
    launch("blocker_2", [blockers[1]["text"]], False)
    time.sleep(gap)
    releases = {}
    aborted = None
    cumulative = 2
    for j, b in enumerate(batches):
        if j > 0:
            m, t_seen, err = wait_for_docs(base + cumulative)
            if err:
                aborted = f"before batch {b['seq']}: {err}"
                break
            snapshot(m, t_seen)
        releases[b["seq"]] = time.time()
        launch(f"batch_{b['seq']}", b["texts"], b["phase"] == "measured")
        cumulative += b["n_documents"]
    if aborted is None:
        m, t_seen, err = wait_for_docs(base + cumulative)
        if err:
            aborted = f"after the last batch: {err}"
        else:
            snapshot(m, t_seen)
    for th in threads:
        th.join(timeout=args.request_timeout_s)
    try:
        metrics_end = metrics_now()
    except Exception:
        metrics_end = None
    poll_client.close()
    t_end = max(
        (r["t_done"] for r in results.values() if r.get("t_done")), default=time.time()
    )
    intervals = {}
    for j, b in enumerate(batches[: max(0, len(snaps) - 1)]):
        before, after = snaps[j], snaps[j + 1]
        size_d = histogram_delta(before["size"], after["size"])
        tok_d = histogram_delta(before["tokens"], after["tokens"])
        expected = [(b["n_documents"], b["n_tokens"])]
        if j == 0:
            expected = [
                (1, blockers[0]["n_tokens"]),
                (1, blockers[1]["n_tokens"]),
            ] + expected
        ok, checks, info = verify_interval(size_d, tok_d, expected)
        intervals[b["seq"]] = {
            "verified": ok,
            "checks": checks,
            "info": info,
            "size_delta_sum": int(round(size_d["sum"])),
            "tokens_delta_sum": int(round(tok_d["sum"])),
            "t_formed_seen": after["t"],
        }
    failures = {}
    if metrics_end:
        for labels, value in metrics_end.get("te_request_failure", []):
            failures[labels.get("err", "?")] = value - counter_value(
                m0, "te_request_failure", labels
            )
    return {
        "t_start": t_start,
        "t_end": t_end,
        "releases": releases,
        "snapshot_times": [s["t"] for s in snaps],
        "intervals": intervals,
        "results": results,
        "aborted": aborted,
        "request_failures": failures,
    }


BATCH_COLUMNS = [
    "seq",
    "phase",
    "phase_index",
    "stream_begin",
    "stream_end",
    "n_documents",
    "n_tokens",
    "oversize",
    "tei_documents",
    "tei_tokens",
    "x_compute_tokens",
    "verified",
    "status",
    "inference_ms",
    "time_s",
    "queue_ms",
    "tokenization_ms",
    "client_wall_s",
    "t_release_s",
    "t_formed_s",
    "embeddings_n",
    "embeddings_dim",
    "embeddings_finite",
]


def run_throughput_attempt(
    ctx, model_key, budget, batches, blockers, calibration, attempt, tdir
):
    """One fresh container at the budget: start-up (load and TEI's own warm-up), the blockers, the
    warm-up batches and the measured batches, with every resource sampled."""
    args = ctx["args"]
    nvml = ctx["nvml"]
    slug = MODELS[model_key]["slug"]
    name = f"{CONTAINER_PREFIX}_{slug}_attempt{attempt}"
    report_path = (
        Path(ctx["shim_report_dir"]) / f"throughput_{slug}_attempt_{attempt}.json"
    )
    if report_path.exists():
        report_path.unlink()
    settle_used = settle_device(
        ctx,
        ctx["session_baseline_used"],
        args.settle_margin_mib * 1048576,
        args.settle_s,
    )
    device_before = nvml.used()
    cap = compute_cap_bytes(
        ctx["ceiling_bytes"], device_before, calibration["untracked_overhead_bytes"]
    )
    rec = {
        "attempt": attempt,
        "container": name,
        "model": model_key,
        "max_batch_tokens": int(budget),
        "timestamp_start": utc_now(),
        "ceiling_bytes": ctx["ceiling_bytes"],
        "untracked_overhead_bytes": calibration["untracked_overhead_bytes"],
        "device_used_after_settle_bytes": settle_used,
        "device_used_before_start_bytes": device_before,
        "cap_bytes": cap,
        "blockers": [
            {k: b[k] for k in ("doc_key", "n_tokens", "n_chars")} for b in blockers
        ],
    }
    log(
        f"{model_key} GPU attempt {attempt}: max_batch_tokens={budget}; {len(batches)} batches; device before {mib(device_before)} MiB; cap {mib(cap)} MiB"
    )
    tc = TeiContainer(ctx, name, model_key, budget, cap, report_path)
    started = tc.start()
    rec["docker_run_command"] = tc.command
    pipe = None
    if started:
        try:
            rec["info"] = get_info(ctx)
        except Exception as e:
            rec["info"] = {"error": repr(e)}
        time.sleep(1.0)
        rec["device_used_at_ready_bytes"] = nvml.used()
        rec["shim_at_ready"] = read_shim_report(report_path)
        log(
            f"  ready in {tc.startup_s:.1f} s; device at rest {mib(rec['device_used_at_ready_bytes'])} MiB; sending the batches"
        )
        pipe = run_pipeline(ctx, name, blockers, batches)
    tc.finish()
    results = pipe["results"] if pipe else {}
    intervals = pipe["intervals"] if pipe else {}
    samples = tc.nvml_sampler.samples
    rows = []
    for b in batches:
        r = results.get(f"batch_{b['seq']}", {})
        iv = intervals.get(b["seq"], {})
        tei_docs, tei_tokens = iv.get("size_delta_sum"), iv.get("tokens_delta_sum")
        if b["seq"] == 1 and tei_docs is not None:
            tei_docs -= 2
            tei_tokens -= blockers[0]["n_tokens"] + blockers[1]["n_tokens"]
        inference_ms = header_int(r.get("headers"), "x-inference-time")
        rows.append(
            {
                "seq": b["seq"],
                "phase": b["phase"],
                "phase_index": b["phase_index"],
                "stream_begin": b["stream_begin"],
                "stream_end": b["stream_end"],
                "n_documents": b["n_documents"],
                "n_tokens": b["n_tokens"],
                "oversize": b["oversize"],
                "tei_documents": tei_docs,
                "tei_tokens": tei_tokens,
                "x_compute_tokens": header_int(r.get("headers"), "x-compute-tokens"),
                "verified": bool(iv.get("verified"))
                and header_int(r.get("headers"), "x-compute-tokens") == b["n_tokens"],
                "status": r.get("status"),
                "inference_ms": inference_ms,
                "time_s": inference_ms / 1000.0 if inference_ms is not None else None,
                "queue_ms": header_int(r.get("headers"), "x-queue-time"),
                "tokenization_ms": header_int(r.get("headers"), "x-tokenization-time"),
                "client_wall_s": round(r["wall_s"], 4)
                if r.get("wall_s") is not None
                else None,
                "t_release_s": round(pipe["releases"][b["seq"]] - pipe["t_start"], 4)
                if pipe and b["seq"] in pipe["releases"]
                else None,
                "t_formed_s": round(iv["t_formed_seen"] - pipe["t_start"], 4)
                if iv.get("t_formed_seen")
                else None,
                "embeddings_n": None,
                "embeddings_dim": None,
                "embeddings_finite": None,
            }
        )
    measured = [r for r in rows if r["phase"] == "measured"]
    blockers_ok = all(
        results.get(k, {}).get("status") == 200 for k in ("blocker_1", "blocker_2")
    )
    batch_ok = (
        bool(pipe)
        and pipe["aborted"] is None
        and blockers_ok
        and all(r["status"] == 200 for r in rows)
    )
    verified = bool(measured) and all(r["verified"] for r in measured)
    first_measured = measured[0]["seq"] if measured else None
    formed = {r["seq"]: r["t_formed_s"] for r in rows}
    if (
        pipe
        and first_measured is not None
        and formed.get(first_measured + 1) is not None
    ):
        t_measured_start = pipe["t_start"] + formed[first_measured + 1]
        measured_start_rule = "formation of the batch after the first measured batch (= the moment the backend took the first measured batch)"
    elif pipe and first_measured is not None and first_measured in pipe["releases"]:
        t_measured_start = pipe["releases"][first_measured]
        measured_start_rule = "release of the first measured batch"
    else:
        t_measured_start = None
        measured_start_rule = None
    startup_peak = summarise_samples(samples, tc.t_run, tc.t_ready)
    warm_peak = (
        summarise_samples(samples, pipe["t_start"], t_measured_start)
        if pipe and t_measured_start
        else summarise_samples([], None, None)
    )
    meas_peak = (
        summarise_samples(samples, t_measured_start, pipe["t_end"])
        if pipe and t_measured_start
        else summarise_samples([], None, None)
    )
    overall_peak = summarise_samples(samples, tc.t_run, tc.t_stopped)
    peak_over = (
        overall_peak["peak"] is not None and overall_peak["peak"] > ctx["ceiling_bytes"]
    )
    outcome, success, failure_at_ceiling = classify_outcome(
        started, False, tc.refused, batch_ok, verified, tc.watchdog["killed"], peak_over
    )
    extra = ""
    if pipe and pipe["aborted"]:
        extra = f"pipeline aborted: {pipe['aborted']}"
    bad = [r for r in rows if r["status"] not in (200, None)]
    if bad:
        r0 = results.get(f"batch_{bad[0]['seq']}", {})
        extra += f" | batch {bad[0]['seq']} status {bad[0]['status']}: {str(r0.get('error'))[:300]}"
    unverified = [r["seq"] for r in measured if not r["verified"]]
    if unverified and outcome == "inconclusive_unverified_batch":
        extra += f" | measured batches not verified: {unverified}"
    rec.update(
        {
            "timestamp_end": utc_now(),
            "started": started,
            "startup_state": tc.startup_state,
            "startup_wait_s": tc.startup_s,
            "log_events": tc.log_events,
            "log_errors": tc.log_errors,
            "shim_final": tc.shim_final,
            "shim_refusals": tc.shim_refusals,
            "refused_at_cap": tc.refused,
            "watchdog": tc.watchdog,
            "container_final_state": tc.final_state,
            "container_memory": tc.memory_summary(),
            "nvml": {
                "interval_ms": args.nvml_interval_ms,
                "startup": startup_peak,
                "warmup_batches": warm_peak,
                "measured_batches": meas_peak,
                "overall": overall_peak,
                "measured_window_start_rule": measured_start_rule,
                "series_1s_max": [
                    (round(t - tc.t_run, 3), v) for t, v in downsample_max(samples, 1.0)
                ],
            },
            "pipeline": {
                "aborted": pipe["aborted"] if pipe else "container did not start",
                "t_start_offset_s": round(pipe["t_start"] - tc.t_run, 3)
                if pipe
                else None,
                "wall_s": round(pipe["t_end"] - pipe["t_start"], 3) if pipe else None,
                "request_failures": pipe["request_failures"] if pipe else None,
                "intervals": {
                    str(k): {kk: vv for kk, vv in v.items() if kk != "t_formed_seen"}
                    for k, v in intervals.items()
                },
                "blocker_status": [
                    results.get(k, {}).get("status") for k in ("blocker_1", "blocker_2")
                ],
            },
            "batches": rows,
            "batch_ok": batch_ok,
            "verified_measured_batches": verified,
            "peak_over_ceiling": peak_over,
            "outcome": outcome,
            "success": success,
            "failure_at_ceiling": failure_at_ceiling,
            "failure_detail": failure_detail(
                outcome, tc.log_errors, tc.shim_final, tc.watchdog, overall_peak, extra
            ),
        }
    )
    (tdir / f"container_attempt_{attempt}.log").write_text(
        tc.logs_text, encoding="utf-8"
    )
    if tc.shim_final is not None:
        write_json(tdir / f"shim_attempt_{attempt}.json", tc.shim_final)
    write_json(tdir / f"attempt_{attempt}.json", rec)
    log(
        f"  attempt {attempt}: {outcome}; peak device {mib(overall_peak['peak']) if overall_peak['peak'] else None} MiB; {rec['failure_detail'][:300]}"
    )
    return rec, pipe


def stage_gpu_throughput(args, model_key):
    spec = MODELS[model_key]
    slug = spec["slug"]
    t_stage = time.perf_counter()
    tdir = stage_dir(args, f"gpu_throughput_{slug}")
    for p in list(tdir.glob("*")):
        if p.is_file():
            p.unlink()
    budget, budget_source = resolve_gpu_budget(args, model_key)
    calibration = read_json(gpu_calibration_path(args, model_key))
    corpus = Corpus(args.project_dir)
    counts = model_token_counts(args, corpus, model_key)
    order, st = benchmark_stream(args, corpus)
    batches = plan_batches(
        corpus, order, counts, budget, args.n_warmup, args.n_measured, st["midpoint"]
    )
    log(
        f"{model_key} GPU throughput at max_batch_tokens={budget} ({budget_source}); reading the texts of {sum(b['n_documents'] for b in batches)} documents in {len(batches)} batches"
    )
    for b in batches:
        b["texts"] = corpus.texts(b["rows"])
    blockers = longest_rows(corpus, counts, 2)
    nvml = Nvml()
    ctx = make_ctx(args, nvml)
    Path(ctx["shim_report_dir"]).mkdir(parents=True, exist_ok=True)
    docker_remove_stale()
    ctx["session_baseline_used"] = nvml.used()
    attempts = []
    accepted = None
    pipe_ok = None
    for attempt in range(1, args.attempts + 1):
        rec, pipe = run_throughput_attempt(
            ctx, model_key, budget, batches, blockers, calibration, attempt, tdir
        )
        attempts.append(rec)
        if rec["success"]:
            accepted, pipe_ok = rec, pipe
            break
        if pipe:
            for r in pipe["results"].values():
                r["body"] = None
        if attempt < args.attempts:
            log("  repeating in a fresh container")
    kept_keys = []
    if accepted:
        mats = []
        by_seq = {r["seq"]: r for r in accepted["batches"]}
        for b in batches:
            if b["phase"] != "measured":
                continue
            res = pipe_ok["results"][f"batch_{b['seq']}"]
            emb = embeddings_from_body(res["body"])
            res["body"] = None
            row = by_seq[b["seq"]]
            row["embeddings_n"], row["embeddings_dim"] = (
                int(emb.shape[0]),
                int(emb.shape[1]),
            )
            row["embeddings_finite"] = bool(np.isfinite(emb).all())
            take = min(emb.shape[0], args.keep_embeddings - len(kept_keys))
            if take > 0:
                mats.append(emb[:take].copy())
                kept_keys.extend(b["doc_keys"][:take])
            del emb
        if mats:
            np.save(tdir / "measured_embeddings.npy", np.concatenate(mats))
            write_csv(
                tdir / "measured_embedding_doc_keys.csv",
                [{"position": i, "doc_key": k} for i, k in enumerate(kept_keys)],
            )
        write_json(tdir / f"attempt_{accepted['attempt']}.json", accepted)
    final = accepted or attempts[-1]
    write_csv(tdir / "batches.csv", final["batches"], BATCH_COLUMNS)
    doc_rows = []
    for b in batches:
        for pos, (key, n) in enumerate(zip(b["doc_keys"], b["n_tokens_list"]), start=1):
            doc_rows.append(
                {
                    "seq": b["seq"],
                    "phase": b["phase"],
                    "phase_index": b["phase_index"],
                    "position": pos,
                    "doc_key": key,
                    "n_tokens": n,
                }
            )
    write_csv(
        tdir / "batch_documents.csv",
        doc_rows,
        ["seq", "phase", "phase_index", "position", "doc_key", "n_tokens"],
    )
    measured_rows = [r for r in final["batches"] if r["phase"] == "measured"]
    steady = None
    if accepted:
        steady = steady_state(
            [
                {
                    "n_documents": r["tei_documents"],
                    "n_tokens": r["tei_tokens"],
                    "time_s": r["time_s"],
                }
                for r in measured_rows
            ]
        )
    run_rec = {
        "stage": "gpu_throughput",
        "model": model_key,
        "timestamp": utc_now(),
        "quick_run": bool(args.quick),
        "max_batch_tokens": budget,
        "budget_source": budget_source,
        "n_warmup": args.n_warmup,
        "n_measured": args.n_measured,
        "model_loaded": bool(final["started"]),
        "accepted_attempt": accepted["attempt"] if accepted else None,
        "attempts": [
            {
                "attempt": a["attempt"],
                "outcome": a["outcome"],
                "failure_detail": a["failure_detail"],
                "overall_peak_device_bytes": a["nvml"]["overall"]["peak"],
            }
            for a in attempts
        ],
        "steady_state": steady,
        "timing_rule": "per batch: TEI's backend duration of the batch (x-inference-time, ms; every input of the request is in that one batch, verified), i.e. the forward pass, pooling and normalisation on the GPU plus the copy of the embeddings to the host; tokenisation, queueing, HTTP and JSON are excluded; the model stays loaded across the warm-up and measured batches",
        "resources": {
            "device_used_before_start_bytes": final["device_used_before_start_bytes"],
            "device_used_at_ready_bytes": final.get("device_used_at_ready_bytes"),
            "tei_physical_at_ready_bytes": (final.get("shim_at_ready") or {}).get(
                "physical_bytes"
            ),
            "tei_live_at_ready_bytes": (final.get("shim_at_ready") or {}).get(
                "used_bytes"
            ),
            "startup_peak_device_bytes": final["nvml"]["startup"]["peak"],
            "warmup_peak_device_bytes": final["nvml"]["warmup_batches"]["peak"],
            "measured_peak_device_bytes": final["nvml"]["measured_batches"]["peak"],
            "overall_peak_device_bytes": final["nvml"]["overall"]["peak"],
            "shim_physical_peak_bytes": (final.get("shim_final") or {}).get(
                "peak_physical_bytes"
            ),
            "container_memory": final["container_memory"],
        },
        "embeddings_kept": {
            "n_documents": len(kept_keys),
            "file": "measured_embeddings.npy" if kept_keys else None,
            "rule": f"the first {args.keep_embeddings} documents of the measured batches (float32 as returned by TEI)",
        },
        "info": final.get("info"),
        "stage_wall_s": time.perf_counter() - t_stage,
    }
    write_json(tdir / "run.json", run_rec)
    nvml.shutdown()
    sidecars = Sidecars(
        args.project_dir,
        dict(
            gpu_parameters(args), max_batch_tokens=budget, budget_source=budget_source
        ),
        f"Throughput benchmark, {model_key} on the GPU (TEI).",
    )
    inputs = [
        sidecars.input_entry(
            results_dir(args) / "prepare.json",
            "prepare stage result (stream, tokenizer comparison)",
        ),
        sidecars.input_entry(
            gpu_calibration_path(args, model_key),
            f"calibration of the {model_key} (untracked overhead)",
        ),
        sidecars.input_entry(
            canonical_paths(args.project_dir)[0], "canonical Phase 1 table, read-only"
        ),
    ]
    for path in sorted(tdir.glob("*")):
        if path.is_file() and not path.name.endswith(".manifest.yml"):
            sidecars.write(
                path, inputs, notes=f"{model_key} GPU throughput run: {path.name}"
            )
    if steady:
        log(
            f"{model_key} GPU steady state: {steady['tokens_per_s']:.0f} tokens/s, {steady['documents_per_s']:.1f} documents/s over {steady['n']} measured batches ({steady['total_s']:.2f} s)"
        )
    else:
        log(
            f"{model_key} GPU throughput: NO ACCEPTED RUN ({final['outcome']}: {final['failure_detail'][:300]})"
        )
    return run_rec


# ----------------------------------------------------------------------------------------------
# CPU: the worker (inside the memory-capped scope) and its orchestration
# ----------------------------------------------------------------------------------------------


def last_token_pool(last_hidden_states, attention_mask):
    """Verbatim from the Qwen3-Embedding model card ("Transformers Usage")."""
    import torch

    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
    ]


def reset_high_water_mark():
    """Resets the process's peak resident set size (VmHWM) so each batch reports its own peak."""
    try:
        with open("/proc/self/clear_refs", "w") as f:
            f.write("5")
        return True
    except Exception:
        return False


def stage_cpu_worker(args):
    """Runs inside the memory-capped scope: loads the model once on the CPU in float32, runs the
    planned batches one at a time and rewrites its record after every batch (so a kill at the
    ceiling leaves everything measured before it)."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    t_process = time.time()
    plan = read_json(args.plan)
    out_path = Path(args.worker_out)
    model_key = plan["model"]
    import torch
    import torch.nn.functional as F

    torch.set_num_threads(args.cpu_threads)
    import transformers.integrations.sdpa_attention as sdpa_mod
    from transformers import AutoModel, AutoTokenizer

    # Keys and values repeated to the query heads instead of enable_gqa: PyTorch's CPU kernel for
    # scaled_dot_product_attention supports GQA only in its math path, which materialises the whole
    # attention matrix (79 s and +5.4 GB for the longest document on the 0.6B); with the flag off the
    # memory-efficient CPU kernel runs and the embeddings are unchanged (design record, CPU probes).
    sdpa_mod.use_gqa_in_sdpa = lambda *a, **k: False
    pid = os.getpid()
    cg = cgroup_dir_of_pid(pid)

    def cpu_usage():
        return cgroup_keyed(cg, "cpu.stat").get("usage_usec") if cg else None

    rec = {
        "stage": "cpu_worker",
        "model": model_key,
        "mode": plan["mode"],
        "pid": pid,
        "cgroup": cg,
        "memory_max": (read_text(Path(cg) / "memory.max") or "").strip()
        if cg
        else None,
        "memory_swap_max": (read_text(Path(cg) / "memory.swap.max") or "").strip()
        if cg
        else None,
        "torch_version": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "parallel_info": torch.__config__.parallel_info(),
        "use_gqa_in_sdpa_forced_false": True,
        "t_process_start": t_process,
        "status": "loading",
        "batches": [],
        "points": [],
    }
    write_json(out_path, rec)
    # This VM's CPU-time accounting runs fast against its wall clock (a Hyper-V guest: one thread
    # busy for 5.000 s was charged 5.200 s of CPU time). The ratio is measured before any other
    # work so that the busy-CPU figures can be corrected for it.
    t0c, c0c = time.perf_counter(), time.process_time()
    while time.perf_counter() - t0c < 2.0:
        pass
    rec["cpu_clock_ratio"] = (time.process_time() - c0c) / (time.perf_counter() - t0c)
    write_json(out_path, rec)
    snap = str(snapshot_path(args.hub_dir, model_key))
    reset_high_water_mark()
    u0, t0 = cpu_usage(), time.time()
    tok = AutoTokenizer.from_pretrained(snap, padding_side="left")
    model = AutoModel.from_pretrained(
        snap, dtype=torch.float32, attn_implementation="sdpa"
    ).eval()
    t1, u1 = time.time(), cpu_usage()
    st = proc_status(pid)
    rec["load"] = {
        "t_start": t0,
        "t_end": t1,
        "wall_s": t1 - t0,
        "cpu_usage_usec": (u1 - u0) if u0 is not None and u1 is not None else None,
        "vm_hwm_bytes": st.get("VmHWM"),
        "rss_bytes": st.get("VmRSS"),
        "rss_anon_bytes": st.get("RssAnon"),
        "threads": st.get("Threads"),
        "param_dtype": str(next(model.parameters()).dtype),
        "n_parameters": int(sum(p.numel() for p in model.parameters())),
        "attn_implementation": getattr(model.config, "_attn_implementation", None),
        "scope_memory_current_bytes": cgroup_int(cg, "memory.current") if cg else None,
    }
    rec["status"] = "running"
    write_json(out_path, rec)
    docs = plan["docs"]

    def run_batch(b, keep):
        positions = range(b["stream_begin"], b["stream_end"])
        texts = [docs[str(p)]["text"] for p in positions]
        lengths = [int(docs[str(p)]["n_tokens"]) for p in positions]
        groups = plan_forward_passes(lengths, b["budget"])
        t_tok0 = time.perf_counter()
        encs = [
            tok(
                [texts[i] for i in g],
                padding=True,
                truncation=False,
                return_tensors="pt",
            )
            for g in groups
        ]
        t_tok = time.perf_counter() - t_tok0
        counts_ok = all(
            int(e["attention_mask"][k].sum()) == lengths[i]
            for g, e in zip(groups, encs)
            for k, i in enumerate(g)
        )
        padded = sum(int(e["input_ids"].numel()) for e in encs)
        reset_high_water_mark()
        p_a = os.times()
        u_a, t_a, wall_a = cpu_usage(), time.perf_counter(), time.time()
        vecs = [None] * len(texts)
        with torch.inference_mode():
            for g, e in zip(groups, encs):
                out = model(**e, use_cache=False)
                v = F.normalize(
                    last_token_pool(out.last_hidden_state, e["attention_mask"]),
                    p=2,
                    dim=1,
                )
                for k, i in enumerate(g):
                    vecs[i] = v[k]
                del out
        t_b, wall_b, u_b = time.perf_counter(), time.time(), cpu_usage()
        p_b = os.times()
        stacked = torch.stack(vecs)
        s = proc_status(pid)
        busy, frac = (
            cpu_utilisation(u_b - u_a, t_b - t_a, args.cpu_threads)
            if u_a is not None and u_b is not None
            else (None, None)
        )
        row = {
            "seq": b["seq"],
            "phase": b["phase"],
            "phase_index": b["phase_index"],
            "budget": b["budget"],
            "stream_begin": b["stream_begin"],
            "stream_end": b["stream_end"],
            "n_documents": len(texts),
            "n_tokens": sum(lengths),
            "padded_tokens": padded,
            "n_forward_passes": len(groups),
            "largest_pass_padded_tokens": max(
                len(g) * max(lengths[i] for i in g) for g in groups
            ),
            "max_document_tokens": max(lengths),
            "oversize": b["oversize"],
            "time_s": t_b - t_a,
            "tokenize_s": t_tok,
            "counts_equal_attention_mask": counts_ok,
            "embeddings_finite": bool(torch.isfinite(stacked).all()),
            "embedding_dim": int(stacked.shape[1]),
            "cpu_usage_usec": (u_b - u_a)
            if u_a is not None and u_b is not None
            else None,
            "busy_cpus": busy,
            "busy_cpus_corrected": (busy / rec["cpu_clock_ratio"])
            if busy is not None
            else None,
            "busy_cpus_process": (
                ((p_b.user + p_b.system) - (p_a.user + p_a.system)) / (t_b - t_a)
                if t_b > t_a
                else None
            ),
            "cpu_utilisation": frac,
            "vm_hwm_bytes": s.get("VmHWM"),
            "rss_bytes": s.get("VmRSS"),
            "rss_anon_bytes": s.get("RssAnon"),
            "threads": s.get("Threads"),
            "scope_memory_peak_bytes": cgroup_int(cg, "memory.peak") if cg else None,
            "t_start": wall_a,
            "t_end": wall_b,
        }
        emb = stacked.numpy().copy() if keep else None
        del stacked, vecs
        return row, emb

    if plan["mode"] == "sweep":
        points = rec["points"]
        while True:
            nxt, reason = sweep_next(
                points,
                plan["gpu_budget"],
                start=plan["sweep_start"],
                min_gain=plan["min_gain"],
            )
            if points:
                points[-1]["decision_after"] = reason
            if nxt is None:
                rec["sweep_stop_reason"] = reason
                break
            point = {
                "budget": nxt,
                "tokens_per_s": None,
                "failed_at_ceiling": False,
                "error": None,
                "status": "running",
            }
            points.append(point)
            write_json(out_path, rec)
            try:
                for b in plan["batch_sets"][str(nxt)]:
                    row, _ = run_batch(b, False)
                    rec["batches"].append(row)
                    write_json(out_path, rec)
            except Exception as e:
                point["error"] = repr(e)[:500]
                point["status"] = "error"
                write_json(out_path, rec)
                continue
            timed = [
                r
                for r in rec["batches"]
                if r["budget"] == nxt and r["phase"] == "timed"
            ]
            s = steady_state(timed)
            point.update(
                {
                    "status": "complete",
                    "tokens_per_s": s["tokens_per_s"],
                    "documents_per_s": s["documents_per_s"],
                    "total_s": s["total_s"],
                    "n_timed_batches": len(timed),
                    "total_tokens": s["total_tokens"],
                    "total_documents": s["total_documents"],
                    "padded_tokens": sum(r["padded_tokens"] for r in timed),
                }
            )
            write_json(out_path, rec)
        rec["choice"] = sweep_choice(points)
    else:
        mats, keys = [], []
        for b in plan["batch_sets"]["run"]:
            keep = b["phase"] == "measured"
            row, emb = run_batch(b, keep)
            rec["batches"].append(row)
            write_json(out_path, rec)
            if keep:
                mats.append(emb)
                keys.extend(
                    docs[str(p)]["doc_key"]
                    for p in range(b["stream_begin"], b["stream_end"])
                )
        if mats and args.worker_embeddings:
            np.save(args.worker_embeddings, np.concatenate(mats))
            write_csv(
                Path(args.worker_embeddings).with_name(
                    "measured_embedding_doc_keys.csv"
                ),
                [{"position": i, "doc_key": k} for i, k in enumerate(keys)],
            )
            rec["embeddings_file"] = str(args.worker_embeddings)
    rec["status"] = "complete"
    rec["t_end"] = time.time()
    rec["final_status"] = proc_status(pid)
    write_json(out_path, rec)


def cpu_budget_bound(args, model_key):
    """The upper bound of the CPU sweep: the GPU budget of the same model."""
    if args.cpu_max_budget:
        return int(args.cpu_max_budget), "command line (--cpu-max-budget)"
    return resolve_gpu_budget(args, model_key)


def write_worker_plan(path, corpus, order, counts, meta, batch_sets):
    """The worker's input: the planned batches (stream ranges) and the texts of the documents they
    use, so the capped process holds only what it embeds (not the whole corpus)."""
    positions = sorted(
        {
            p
            for bs in batch_sets.values()
            for b in bs
            for p in range(b["stream_begin"], b["stream_end"])
        }
    )
    rows = order[positions] if positions else np.array([], dtype=np.int64)
    texts = corpus.texts(rows)
    docs = {
        str(p): {
            "doc_key": str(corpus.doc_keys[r]),
            "n_tokens": int(counts[r]),
            "text": t,
        }
        for p, r, t in zip(positions, rows, texts)
    }
    plan = dict(meta, batch_sets=batch_sets, docs=docs)
    write_json(path, plan)
    return len(docs)


def stream_batch_set(stream_counts, start, budget, n, phases, seq0=1):
    """Planned batches (stream ranges) packed by the greedy rule, labelled with their phases."""
    out = []
    for k, b in enumerate(pack_greedy(stream_counts, start, budget, n)):
        out.append(
            {
                "seq": seq0 + k,
                "phase": phases[k],
                "phase_index": sum(1 for x in phases[:k] if x == phases[k]) + 1,
                "budget": int(budget),
                "stream_begin": b["begin"],
                "stream_end": b["end"],
                "n_documents": b["n_documents"],
                "n_tokens": b["n_tokens"],
                "oversize": b["oversize"],
            }
        )
    return out


def launch_cpu_worker(args, plan_path, out_path, emb_path, log_path):
    """Starts the worker in a transient systemd scope with MemoryMax = the RAM ceiling and no swap
    and samples the scope's memory, the worker's resident memory, the VM's memory in use and the
    GPU's memory in use until it exits."""
    limit = args.ram_ceiling_mib * 1048576
    cmd = [
        "systemd-run",
        "--user",
        "--scope",
        "--quiet",
        "-p",
        f"MemoryMax={limit}",
        "-p",
        "MemorySwapMax=0",
        "--",
        "env",
        "CUDA_VISIBLE_DEVICES=",
        f"OMP_NUM_THREADS={args.cpu_threads}",
        f"MKL_NUM_THREADS={args.cpu_threads}",
        "TOKENIZERS_PARALLELISM=true",
        "HF_HUB_OFFLINE=1",
        "PYTHONUNBUFFERED=1",
        sys.executable,
        str(SCRIPT_PATH),
        "--stage",
        "cpu_worker",
        "--plan",
        str(plan_path),
        "--worker-out",
        str(out_path),
        "--worker-embeddings",
        str(emb_path or ""),
        "--cpu-threads",
        str(args.cpu_threads),
        "--hub-dir",
        args.hub_dir,
        "--results-dir",
        args.results_dir,
        "--environment-dir",
        args.environment_dir,
        "--project-dir",
        args.project_dir,
    ]
    nvml = Nvml()
    gpu_before = nvml.used()
    interval = args.cpu_sample_ms / 1000.0
    logf = open(log_path, "w", encoding="utf-8")
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)
    cg = None
    for _ in range(400):
        c = cgroup_dir_of_pid(proc.pid)
        if c and c.endswith(".scope") and cgroup_int(c, "memory.max") == limit:
            cg = c
            break
        if proc.poll() is not None:
            break
        time.sleep(0.025)
    last = {
        "memory_peak": None,
        "oom_kill": 0,
        "memory_max": cgroup_int(cg, "memory.max") if cg else None,
        "swap_max": (read_text(Path(cg) / "memory.swap.max") or "").strip()
        if cg
        else None,
    }

    def scope_current():
        v = cgroup_int(cg, "memory.current") if cg else None
        p = cgroup_int(cg, "memory.peak") if cg else None
        if p is not None:
            last["memory_peak"] = p
        ev = cgroup_keyed(cg, "memory.events") if cg else {}
        if ev.get("oom_kill"):
            last["oom_kill"] = ev["oom_kill"]
        return v

    def worker_rss():
        s = proc_status(proc.pid)
        return s.get("VmRSS")

    def worker_anon():
        return proc_status(proc.pid).get("RssAnon")

    samplers = {
        "scope_memory": PeakSampler(scope_current, interval_s=interval).start(),
        "worker_rss": PeakSampler(worker_rss, interval_s=interval).start(),
        "worker_anon": PeakSampler(worker_anon, interval_s=interval).start(),
        "vm_used": PeakSampler(vm_used_bytes, interval_s=max(interval, 0.1)).start(),
        "gpu_used": PeakSampler(nvml.used, interval_s=max(interval, 0.1)).start(),
    }
    try:
        rc = proc.wait(timeout=args.worker_timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        rc = proc.wait()
        last["timeout"] = True
    t1 = time.time()
    for s in samplers.values():
        s.stop()
    logf.close()
    nvml.shutdown()
    peak = max(
        x
        for x in (last["memory_peak"], samplers["scope_memory"].peak, 0)
        if x is not None
    )
    outcome, failure_at_ceiling = classify_cpu_exit(rc, last["oom_kill"], peak, limit)
    return {
        "command": cmd,
        "returncode": rc,
        "outcome": outcome,
        "failure_at_ceiling": failure_at_ceiling,
        "t_start": t0,
        "t_end": t1,
        "wall_s": t1 - t0,
        "scope_cgroup": cg,
        "scope_memory_max_bytes": last["memory_max"],
        "scope_memory_swap_max": last["swap_max"],
        "scope_memory_peak_bytes": peak,
        "scope_oom_kill_events": last["oom_kill"],
        "memory_limit_bytes": limit,
        "gpu_used_before_bytes": gpu_before,
        "samplers": samplers,
        "timed_out": bool(last.get("timeout")),
    }


def window_peaks(run, t0, t1):
    """Peaks of the orchestrator's samples between two times."""
    s = run["samplers"]
    return {
        "scope_memory_peak_bytes": summarise_samples(s["scope_memory"].samples, t0, t1)[
            "peak"
        ],
        "worker_rss_peak_bytes": summarise_samples(s["worker_rss"].samples, t0, t1)[
            "peak"
        ],
        "worker_anon_peak_bytes": summarise_samples(s["worker_anon"].samples, t0, t1)[
            "peak"
        ],
        "vm_used_peak_bytes": summarise_samples(s["vm_used"].samples, t0, t1)["peak"],
        "gpu_used_peak_bytes": summarise_samples(s["gpu_used"].samples, t0, t1)["peak"],
    }


def run_summary(run):
    """The orchestrator's view of a worker run without the raw sample lists."""
    out = {k: v for k, v in run.items() if k != "samplers"}
    s = run["samplers"]
    out["peaks"] = window_peaks(run, None, None)
    out["gpu_used_min_bytes"] = min((v for _, v in s["gpu_used"].samples), default=None)
    out["n_samples"] = {k: v.n_samples for k, v in s.items()}
    out["series_10s_max"] = {
        k: [
            (round(t - run["t_start"], 1), v)
            for t, v in downsample_max(s[k].samples, 10.0)
        ]
        for k in ("scope_memory", "worker_rss", "gpu_used")
    }
    return out


CPU_BATCH_COLUMNS = [
    "seq",
    "phase",
    "phase_index",
    "budget",
    "stream_begin",
    "stream_end",
    "n_documents",
    "n_tokens",
    "padded_tokens",
    "n_forward_passes",
    "largest_pass_padded_tokens",
    "max_document_tokens",
    "oversize",
    "time_s",
    "tokenize_s",
    "counts_equal_attention_mask",
    "embeddings_finite",
    "busy_cpus",
    "busy_cpus_corrected",
    "busy_cpus_process",
    "cpu_utilisation",
    "vm_hwm_bytes",
    "rss_anon_bytes",
    "threads",
    "scope_memory_peak_bytes",
]


def stage_cpu_sweep(args, model_key):
    """The bounded CPU batching sweep of one model in one worker (model loaded once)."""
    spec = MODELS[model_key]
    slug = spec["slug"]
    cdir = stage_dir(args, f"cpu_sweep_{slug}")
    for p in list(cdir.glob("*")):
        if p.is_file():
            p.unlink()
    bound, bound_source = cpu_budget_bound(args, model_key)
    corpus = Corpus(args.project_dir)
    counts = model_token_counts(args, corpus, model_key)
    order, st = benchmark_stream(args, corpus)
    stream_counts = counts[order]
    budgets = []
    b = min(args.cpu_sweep_start, bound)
    while True:
        budgets.append(b)
        if b >= bound:
            break
        b = min(2 * b, bound)
    phases = ["warmup"] * args.cpu_sweep_warmup + ["timed"] * args.cpu_sweep_timed
    batch_sets = {}
    seq = 1
    for B in budgets:
        batch_sets[str(B)] = stream_batch_set(
            stream_counts, 0, B, len(phases), phases, seq0=seq
        )
        seq += len(phases)
    work = Path(args.work_dir).expanduser()
    work.mkdir(parents=True, exist_ok=True)
    plan_path = work / f"plan_cpu_sweep_{slug}.json"
    n_docs = write_worker_plan(
        plan_path,
        corpus,
        order,
        counts,
        {
            "model": model_key,
            "mode": "sweep",
            "gpu_budget": bound,
            "sweep_start": args.cpu_sweep_start,
            "min_gain": args.cpu_min_gain,
        },
        batch_sets,
    )
    del corpus
    log(
        f"{model_key} CPU sweep: candidate budgets {budgets} (bound {bound}, {bound_source}); {n_docs} documents in the plan; worker in a {args.ram_ceiling_mib} MiB scope with {args.cpu_threads} threads"
    )
    out_path = cdir / "worker_record.json"
    run = launch_cpu_worker(args, plan_path, out_path, None, cdir / "worker.log")
    worker = (
        read_json(out_path)
        if out_path.exists()
        else {"points": [], "batches": [], "status": "no record"}
    )
    points = worker.get("points", [])
    if points and points[-1].get("status") == "running":
        points[-1]["failed_at_ceiling"] = run["failure_at_ceiling"]
        points[-1]["error"] = (
            None
            if run["failure_at_ceiling"]
            else f"worker ended: {run['outcome']} (return code {run['returncode']})"
        )
        points[-1]["status"] = (
            "killed_at_ram_ceiling" if run["failure_at_ceiling"] else "worker_ended"
        )
    choice = sweep_choice(points)
    point_rows = []
    for p in points:
        rows_p = [r for r in worker.get("batches", []) if r["budget"] == p["budget"]]
        t0 = min((r["t_start"] for r in rows_p), default=None)
        t1 = max((r["t_end"] for r in rows_p), default=None)
        peaks = window_peaks(run, t0, t1) if t0 else {}
        timed = [r for r in rows_p if r["phase"] == "timed"]
        point_rows.append(
            {
                "budget": p["budget"],
                "status": p.get("status"),
                "tokens_per_s": round(p["tokens_per_s"], 2)
                if p.get("tokens_per_s")
                else None,
                "documents_per_s": round(p["documents_per_s"], 3)
                if p.get("documents_per_s")
                else None,
                "n_timed_batches": len(timed),
                "timed_documents": sum(r["n_documents"] for r in timed),
                "timed_tokens": sum(r["n_tokens"] for r in timed),
                "timed_padded_tokens": sum(r["padded_tokens"] for r in timed),
                "timed_total_s": round(sum(r["time_s"] for r in timed), 3),
                "busy_cpus_mean": round(
                    sum(r.get("busy_cpus_corrected") or r["busy_cpus"] or 0 for r in timed) / len(timed), 2
                )
                if timed
                else None,
                "worker_vm_hwm_peak_mib": mib(
                    max(r["vm_hwm_bytes"] or 0 for r in rows_p)
                )
                if rows_p
                else None,
                "scope_memory_peak_mib": mib(peaks["scope_memory_peak_bytes"])
                if peaks.get("scope_memory_peak_bytes")
                else None,
                "failed_at_ceiling": p.get("failed_at_ceiling"),
                "error": p.get("error"),
                "decision_after": p.get("decision_after"),
                "chosen": bool(choice and p["budget"] == choice["budget"]),
            }
        )
    sweep = {
        "stage": "cpu_sweep",
        "model": model_key,
        "timestamp": utc_now(),
        "quick_run": bool(args.quick),
        "bound": bound,
        "bound_source": bound_source,
        "candidate_budgets": budgets,
        "rule": f"start at {args.cpu_sweep_start}, double, never above the bound; {args.cpu_sweep_warmup} warm-up and {args.cpu_sweep_timed} timed batches per point from the start of the stream; stop at the first point without a {args.cpu_min_gain:.0%} gain over the best earlier point, at a failure, or at the bound; the choice is the point with the highest tokens per second",
        "points": point_rows,
        "stop_reason": worker.get("sweep_stop_reason")
        or (points[-1].get("status") if points else None),
        "choice": {"budget": choice["budget"], "tokens_per_s": choice["tokens_per_s"]}
        if choice
        else None,
        "worker_status": worker.get("status"),
        "worker_load": worker.get("load"),
        "worker_threads": {
            k: worker.get(k)
            for k in (
                "torch_threads",
                "torch_interop_threads",
                "cpu_clock_ratio",
                "omp_num_threads",
                "mkl_num_threads",
                "cuda_available",
                "cuda_visible_devices",
                "cpu_capability",
            )
        },
        "run": run_summary(run),
    }
    write_json(cdir / "sweep.json", sweep)
    write_csv(cdir / "points.csv", point_rows)
    write_csv(cdir / "batches.csv", worker.get("batches", []), CPU_BATCH_COLUMNS)
    sidecars = Sidecars(
        args.project_dir,
        dict(cpu_parameters(args), bound=bound, bound_source=bound_source),
        f"Throughput benchmark, {model_key} CPU batching sweep.",
    )
    inputs = [
        sidecars.input_entry(
            results_dir(args) / "prepare.json", "prepare stage result (stream)"
        ),
        sidecars.input_entry(
            canonical_paths(args.project_dir)[0], "canonical Phase 1 table, read-only"
        ),
    ]
    for path in sorted(cdir.glob("*")):
        if path.is_file() and not path.name.endswith(".manifest.yml"):
            sidecars.write(path, inputs, notes=f"{model_key} CPU sweep: {path.name}")
    log(
        f"{model_key} CPU sweep: {run['outcome']}; points {[(r['budget'], r['tokens_per_s']) for r in point_rows]}; stop: {sweep['stop_reason']}; choice {sweep['choice']}"
    )
    return sweep


def stage_cpu_throughput(args, model_key):
    """3 warm-up and 10 measured CPU batches at the sweep's chosen budget, fresh worker."""
    spec = MODELS[model_key]
    slug = spec["slug"]
    t_stage = time.perf_counter()
    cdir = stage_dir(args, f"cpu_throughput_{slug}")
    for p in list(cdir.glob("*")):
        if p.is_file():
            p.unlink()
    if args.cpu_budget:
        budget, source = int(args.cpu_budget), "command line (--cpu-budget)"
    else:
        sweep = read_json(results_dir(args) / f"cpu_sweep_{slug}" / "sweep.json")
        if not sweep.get("choice"):
            raise SystemExit(f"the {model_key} CPU sweep has no completed point")
        budget, source = (
            int(sweep["choice"]["budget"]),
            f"results/cpu_sweep_{slug}/sweep.json (the point with the highest tokens per second)",
        )
    corpus = Corpus(args.project_dir)
    counts = model_token_counts(args, corpus, model_key)
    order, st = benchmark_stream(args, corpus)
    stream_counts = counts[order]
    run_set = stream_batch_set(
        stream_counts, 0, budget, args.n_warmup, ["warmup"] * args.n_warmup, seq0=1
    )
    run_set += stream_batch_set(
        stream_counts,
        st["midpoint"],
        budget,
        args.n_measured,
        ["measured"] * args.n_measured,
        seq0=len(run_set) + 1,
    )
    work = Path(args.work_dir).expanduser()
    work.mkdir(parents=True, exist_ok=True)
    plan_path = work / f"plan_cpu_throughput_{slug}.json"
    write_worker_plan(
        plan_path,
        corpus,
        order,
        counts,
        {"model": model_key, "mode": "throughput", "budget": budget},
        {"run": run_set},
    )
    doc_rows = []
    for b in run_set:
        rows = order[b["stream_begin"] : b["stream_end"]]
        for pos, r in enumerate(rows, start=1):
            doc_rows.append(
                {
                    "seq": b["seq"],
                    "phase": b["phase"],
                    "phase_index": b["phase_index"],
                    "position": pos,
                    "doc_key": str(corpus.doc_keys[r]),
                    "n_tokens": int(counts[r]),
                }
            )
    del corpus
    log(
        f"{model_key} CPU throughput at budget {budget} ({source}): {len(run_set)} batches; worker in a {args.ram_ceiling_mib} MiB scope with {args.cpu_threads} threads"
    )
    out_path = cdir / "worker_record.json"
    emb_path = cdir / "measured_embeddings.npy"
    run = launch_cpu_worker(args, plan_path, out_path, emb_path, cdir / "worker.log")
    worker = (
        read_json(out_path)
        if out_path.exists()
        else {"batches": [], "status": "no record"}
    )
    rows = worker.get("batches", [])
    measured = [r for r in rows if r["phase"] == "measured"]
    warm = [r for r in rows if r["phase"] == "warmup"]
    complete = (
        worker.get("status") == "complete"
        and run["outcome"] == "success"
        and len(measured) == args.n_measured
    )
    steady = steady_state(measured) if complete else None
    t_meas0 = min((r["t_start"] for r in measured), default=None)
    t_meas1 = max((r["t_end"] for r in measured), default=None)
    t_warm0 = min((r["t_start"] for r in warm), default=None)
    t_warm1 = max((r["t_end"] for r in warm), default=None)
    usage = sum(r["cpu_usage_usec"] or 0 for r in measured)
    busy, frac = (
        cpu_utilisation(usage, sum(r["time_s"] for r in measured), args.cpu_threads)
        if measured
        else (None, None)
    )
    run_rec = {
        "stage": "cpu_throughput",
        "model": model_key,
        "timestamp": utc_now(),
        "quick_run": bool(args.quick),
        "budget": budget,
        "budget_source": source,
        "n_warmup": args.n_warmup,
        "n_measured": args.n_measured,
        "model_loaded": bool(worker.get("load")),
        "complete": complete,
        "outcome": run["outcome"],
        "steady_state": steady,
        "timing_rule": "per batch: the forward passes (at most the budget in padded tokens each; documents sorted by length), last-token pooling and L2 normalisation, float32, one batch at a time; tokenisation timed separately; the model stays loaded across the warm-up and measured batches",
        "worker_load": worker.get("load"),
        "worker_threads": {
            k: worker.get(k)
            for k in (
                "torch_threads",
                "torch_interop_threads",
                "cpu_clock_ratio",
                "omp_num_threads",
                "mkl_num_threads",
                "cuda_available",
                "cuda_visible_devices",
                "cpu_capability",
                "memory_max",
                "memory_swap_max",
            )
        },
        "resources": {
            "scope_memory_peak_bytes": run["scope_memory_peak_bytes"],
            "scope_memory_limit_bytes": run["memory_limit_bytes"],
            "load": window_peaks(
                run,
                worker.get("load", {}).get("t_start"),
                worker.get("load", {}).get("t_end"),
            )
            if worker.get("load")
            else None,
            "warmup_batches": window_peaks(run, t_warm0, t_warm1) if t_warm0 else None,
            "measured_batches": window_peaks(run, t_meas0, t_meas1)
            if t_meas0
            else None,
            "worker_vm_hwm_peak_bytes_measured": max(
                (r["vm_hwm_bytes"] or 0 for r in measured), default=None
            ),
            "worker_rss_anon_peak_bytes_measured": max(
                (r["rss_anon_bytes"] or 0 for r in measured), default=None
            ),
            "measured_busy_cpus": busy,
            "measured_busy_cpus_corrected": (busy / worker["cpu_clock_ratio"])
            if busy is not None and worker.get("cpu_clock_ratio")
            else None,
            "cpu_clock_ratio": worker.get("cpu_clock_ratio"),
            "measured_busy_cpus_process": (
                sum((r.get("busy_cpus_process") or 0) * r["time_s"] for r in measured)
                / sum(r["time_s"] for r in measured)
                if measured
                else None
            ),
            "measured_cpu_utilisation": frac,
            "threads_in_process_max": max(
                (r["threads"] or 0 for r in rows), default=None
            ),
            "gpu_used_before_bytes": run["gpu_used_before_bytes"],
        },
        "run": run_summary(run),
        "stage_wall_s": time.perf_counter() - t_stage,
    }
    write_json(cdir / "run.json", run_rec)
    write_csv(cdir / "batches.csv", rows, CPU_BATCH_COLUMNS)
    write_csv(
        cdir / "batch_documents.csv",
        doc_rows,
        ["seq", "phase", "phase_index", "position", "doc_key", "n_tokens"],
    )
    sidecars = Sidecars(
        args.project_dir,
        dict(cpu_parameters(args), budget=budget, budget_source=source),
        f"Throughput benchmark, {model_key} on the CPU.",
    )
    inputs = [
        sidecars.input_entry(
            results_dir(args) / "prepare.json", "prepare stage result (stream)"
        ),
        sidecars.input_entry(
            canonical_paths(args.project_dir)[0], "canonical Phase 1 table, read-only"
        ),
    ]
    if not args.cpu_budget:
        inputs.append(
            sidecars.input_entry(
                results_dir(args) / f"cpu_sweep_{slug}" / "sweep.json",
                "the CPU sweep (chosen budget)",
            )
        )
    for path in sorted(cdir.glob("*")):
        if path.is_file() and not path.name.endswith(".manifest.yml"):
            sidecars.write(
                path, inputs, notes=f"{model_key} CPU throughput run: {path.name}"
            )
    if steady:
        log(
            f"{model_key} CPU steady state: {steady['tokens_per_s']:.1f} tokens/s, {steady['documents_per_s']:.2f} documents/s over {steady['n']} measured batches ({steady['total_s']:.1f} s)"
        )
    else:
        log(
            f"{model_key} CPU throughput: NOT COMPLETE ({run['outcome']}, return code {run['returncode']}, worker status {worker.get('status')})"
        )
    return run_rec


# ----------------------------------------------------------------------------------------------
# Stage: compare (float16 on the GPU versus float32 on the CPU, the same documents)
# ----------------------------------------------------------------------------------------------


def read_csv_rows(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_kept_embeddings(d):
    emb, keys = d / "measured_embeddings.npy", d / "measured_embedding_doc_keys.csv"
    if not emb.exists() or not keys.exists():
        return None, None
    return np.load(emb), [r["doc_key"] for r in read_csv_rows(keys)]


def stage_compare(args):
    """For every document embedded in the measured batches of both the GPU run (TEI, float16) and
    the CPU run (float32) of the same model: the cosine between its two embeddings and their largest
    absolute difference. A document is compared with itself only; no document-to-document
    similarity is computed."""
    rdir = results_dir(args)
    out = rdir / "precision_agreement.csv"
    remove_outputs([out])
    rows = []
    inputs = []
    sidecars = Sidecars(
        args.project_dir,
        {
            "rule": "cosine and largest absolute difference between the GPU (TEI, float16 compute) and CPU (float32) embeddings of each document embedded by both measured runs"
        },
        "Throughput benchmark, precision agreement.",
    )
    for model_key, spec in MODELS.items():
        slug = spec["slug"]
        gdir, cdir = rdir / f"gpu_throughput_{slug}", rdir / f"cpu_throughput_{slug}"
        out_docs = rdir / f"precision_agreement_documents_{slug}.csv"
        remove_outputs([out_docs])
        G, gk = load_kept_embeddings(gdir)
        C, ck = load_kept_embeddings(cdir)
        if G is None or C is None:
            rows.append(
                {
                    "model": model_key,
                    "status": "embeddings missing (a run did not complete)",
                }
            )
            continue
        gi = {k: i for i, k in enumerate(gk)}
        pairs = [(j, gi[k]) for j, k in enumerate(ck) if k in gi]
        cj = np.array([a for a, _ in pairs], dtype=np.int64)
        gj = np.array([b for _, b in pairs], dtype=np.int64)
        Cc, Gc = C[cj].astype(np.float64), G[gj].astype(np.float64)
        cos = cosine_rows(Gc, Cc)
        maxabs = np.abs(Gc - Cc).max(axis=1)
        tokens = {}
        if (cdir / "batch_documents.csv").exists():
            tokens = {
                r["doc_key"]: int(r["n_tokens"])
                for r in read_csv_rows(cdir / "batch_documents.csv")
            }
        write_csv(
            out_docs,
            [
                {
                    "doc_key": ck[j],
                    "n_tokens": tokens.get(ck[j]),
                    "cosine": f"{c:.9f}",
                    "max_abs_diff": f"{m:.3e}",
                    "gpu_norm": f"{np.linalg.norm(Gc[i]):.6f}",
                    "cpu_norm": f"{np.linalg.norm(Cc[i]):.6f}",
                }
                for i, (j, c, m) in enumerate(zip(cj, cos, maxabs))
            ],
            ["doc_key", "n_tokens", "cosine", "max_abs_diff", "gpu_norm", "cpu_norm"],
        )
        s = agreement_summary(cos)
        rows.append(
            {
                "model": model_key,
                "status": "compared",
                "n_gpu_embeddings_kept": len(gk),
                "n_cpu_embeddings": len(ck),
                "n_compared": len(pairs),
                "dim": int(G.shape[1]),
                **{
                    k: (f"{v:.9f}" if isinstance(v, float) else v) for k, v in s.items()
                },
                "max_abs_diff_median": f"{float(np.median(maxabs)):.3e}"
                if len(maxabs)
                else None,
                "max_abs_diff_max": f"{float(maxabs.max()):.3e}"
                if len(maxabs)
                else None,
                "gpu_all_finite": bool(np.isfinite(G).all()),
                "cpu_all_finite": bool(np.isfinite(C).all()),
            }
        )
        doc_inputs = [
            sidecars.input_entry(
                gdir / "measured_embeddings.npy",
                f"{model_key} GPU measured embeddings (TEI, float16 compute)",
            ),
            sidecars.input_entry(
                cdir / "measured_embeddings.npy",
                f"{model_key} CPU measured embeddings (float32)",
            ),
        ]
        inputs += doc_inputs
        sidecars.write(
            out_docs, doc_inputs, notes=f"per-document agreement, {model_key}"
        )
    write_csv(
        out,
        rows,
        sorted(
            {k for r in rows for k in r}, key=lambda k: (k != "model", k != "status", k)
        ),
    )
    sidecars.write(out, inputs, notes="per-model summary of the precision agreement")
    for r in rows:
        log(
            f"precision {r['model']}: {r.get('status')}; compared {r.get('n_compared')}; min cosine {r.get('min')}; median {r.get('median')}"
        )
    return rows


# ----------------------------------------------------------------------------------------------
# Stage: report
# ----------------------------------------------------------------------------------------------


def _f(x, nd=3):
    return "" if x is None else f"{x:,.{nd}f}"


def _i(x):
    return "" if x is None else f"{int(x):,}"


def _mib(b):
    return "" if b is None else f"{mib(b):,.1f} MiB"


def _gib(b):
    return "" if b is None else f"{gib(b):,.2f} GiB"


def load_optional(path):
    return read_json(path) if Path(path).exists() else None


def gpu_config(args, model_key):
    """Everything the report needs about one GPU configuration, or None if it has not run."""
    tdir = results_dir(args) / f"gpu_throughput_{MODELS[model_key]['slug']}"
    run = load_optional(tdir / "run.json")
    if not run:
        return None
    att_no = run.get("accepted_attempt") or (
        run["attempts"][-1]["attempt"] if run.get("attempts") else None
    )
    att = load_optional(tdir / f"attempt_{att_no}.json") if att_no else None
    rows = (att or {}).get("batches", [])
    return {
        "run": run,
        "attempt": att,
        "rows": rows,
        "measured": [r for r in rows if r["phase"] == "measured"],
        "warmup": [r for r in rows if r["phase"] == "warmup"],
    }


def cpu_config(args, model_key):
    slug = MODELS[model_key]["slug"]
    cdir = results_dir(args) / f"cpu_throughput_{slug}"
    run = load_optional(cdir / "run.json")
    if not run:
        return None
    worker = load_optional(cdir / "worker_record.json") or {}
    rows = worker.get("batches", [])
    return {
        "run": run,
        "worker": worker,
        "rows": rows,
        "measured": [r for r in rows if r["phase"] == "measured"],
        "warmup": [r for r in rows if r["phase"] == "warmup"],
        "sweep": load_optional(results_dir(args) / f"cpu_sweep_{slug}" / "sweep.json"),
    }


def comparison_rows(g, c):
    """The rows of the comparison table: {measurement: {config: text}}."""
    cols = ["4B GPU", "0.6B GPU", "4B CPU", "0.6B CPU"]
    confs = {
        "4B GPU": g.get("4B"),
        "0.6B GPU": g.get("0.6B"),
        "4B CPU": c.get("4B"),
        "0.6B CPU": c.get("0.6B"),
    }

    def st(conf):
        return (conf or {}).get("run", {}).get("steady_state") if conf else None

    table = []

    def add(label, fn):
        row = {"Measurement": label}
        for col in cols:
            conf = confs[col]
            try:
                row[col] = fn(col, conf) if conf else "not run"
            except Exception as e:
                row[col] = f"(unavailable: {type(e).__name__})"
        table.append(row)

    add(
        "Model loaded successfully",
        lambda col, x: "yes" if x["run"].get("model_loaded") else "NO",
    )

    def budget_text(col, x):
        if "GPU" in col:
            return f"TEI max_batch_tokens = {x['run']['max_batch_tokens']:,} (no document-count limit; float16)"
        return f"{x['run']['budget']:,} real tokens per batch, forward passes of at most {x['run']['budget']:,} padded tokens; float32; {x['run']['worker_threads'].get('torch_threads')} threads"

    add("Production benchmark token budget/batch configuration", budget_text)
    add(
        "Mean batch time",
        lambda col, x: f"{st(x)['mean_s']:.3f} s" if st(x) else "no complete run",
    )
    add(
        "Median batch time",
        lambda col, x: f"{st(x)['median_s']:.3f} s" if st(x) else "no complete run",
    )
    add(
        "Documents/sec",
        lambda col, x: (
            f"{st(x)['documents_per_s']:,.1f}" if st(x) else "no complete run"
        ),
    )
    add(
        "Tokens/sec",
        lambda col, x: f"{st(x)['tokens_per_s']:,.0f}" if st(x) else "no complete run",
    )

    def ram_text(col, x):
        res = x["run"]["resources"]
        if "GPU" in col:
            cm = res.get("container_memory") or {}
            peak = cm.get("memory_peak_bytes") or cm.get(
                "memory_current_peak_sampled_bytes"
            )
            return f"{_gib(peak)} (TEI container, cgroup peak); VM in use {_gib(cm.get('vm_used_peak_bytes'))}"
        return f"{_gib(res.get('scope_memory_peak_bytes'))} (worker scope peak: anonymous memory plus page cache charged to the worker; limit {_gib(res.get('scope_memory_limit_bytes'))}); worker anonymous memory peak {_gib(res.get('worker_rss_anon_peak_bytes_measured'))}"

    add("Peak RAM", ram_text)

    def vram_text(col, x):
        if "CPU" in col:
            return "N/A"
        res = x["run"]["resources"]
        return f"{_mib(res.get('overall_peak_device_bytes'))} device total (other processes held {_mib(res.get('device_used_before_start_bytes'))} before start); TEI's own physical peak {_mib(res.get('shim_physical_peak_bytes'))}"

    add("Peak VRAM", vram_text)
    add(
        "Highest document count observed in one measured batch",
        lambda col, x: _i(st(x)["max_documents_in_batch"]) if st(x) else "",
    )
    add(
        "Highest token count observed in one measured batch",
        lambda col, x: _i(st(x)["max_tokens_in_batch"]) if st(x) else "",
    )
    add(
        "Total measured inference time (10 batches)",
        lambda col, x: f"{st(x)['total_s']:.3f} s" if st(x) else "",
    )
    add(
        "Minimum / maximum batch time",
        lambda col, x: (
            f"{st(x)['min_s']:.3f} / {st(x)['max_s']:.3f} s" if st(x) else ""
        ),
    )
    add(
        "Standard deviation of batch time (n - 1)",
        lambda col, x: (
            f"{st(x)['std_s']:.3f} s" if st(x) and st(x)["std_s"] is not None else ""
        ),
    )
    add(
        "Documents / tokens in the measured batches",
        lambda col, x: (
            f"{_i(st(x)['total_documents'])} / {_i(st(x)['total_tokens'])}"
            if st(x)
            else ""
        ),
    )
    add(
        "Lowest document / token count in one measured batch",
        lambda col, x: (
            f"{_i(st(x)['min_documents_in_batch'])} / {_i(st(x)['min_tokens_in_batch'])}"
            if st(x)
            else ""
        ),
    )
    return table, cols


def ratio_text(a, b, key):
    sa = (a or {}).get("run", {}).get("steady_state") if a else None
    sb = (b or {}).get("run", {}).get("steady_state") if b else None
    r = speed_ratio(sa[key] if sa else None, sb[key] if sb else None)
    return "n/a (a configuration did not complete)" if r is None else f"{r:,.2f}x"


PRECISION_TEXT = """### What precision means here

A neural network stores every weight and every intermediate result of its calculation as a
floating-point number: a number held in a fixed budget of binary digits. *Precision* is that
budget. It decides how finely a value can be represented (how many significant digits survive
each multiplication) and how large a value can become before it overflows.

- **Full precision, float32:** 32 bits per number, about 7 significant decimal digits, largest
  value about 3.4 x 10^38. The CPU runs of this benchmark compute in float32.
- **Half precision, float16:** 16 bits per number, about 3 significant decimal digits, largest
  value 65,504. TEI computes in float16 on the GPU: its fast (flash-attention) implementation of
  Qwen3 accepts no other precision, and its CUDA build offers only float16 or float32.
- **bfloat16:** the 16-bit format in which Qwen publishes the weights of both models (the
  earlier feasibility test ran in bfloat16): the range of float32 but only about 2 to 3
  significant digits.

### Why it matters beyond memory

1. **Speed.** A 16-bit number takes half the memory and half the memory traffic of a 32-bit one,
   and the GPU's tensor cores, on which flash attention relies, work on 16-bit inputs. Much of the
   GPU's speed in this benchmark comes from running in half precision.
2. **The numbers come out slightly different.** Every multiplication rounds to fewer digits and
   the rounding errors accumulate through the model's 28 (0.6B) or 36 (4B) layers. A document's
   float16 embedding therefore points in almost, but not exactly, the same direction as its
   float32 embedding.
3. **Overflow.** float16 cannot hold a value above 65,504. An intermediate value past that
   becomes infinity, and the embedding would be corrupted (not-a-number). Every embedding of every
   measured batch was checked for non-finite values; the results are below.
4. **Downstream steps.** Semantic similarity (STS), retrieval, clustering and indexing all
   compare embeddings with the cosine. A small change in each vector changes those cosines by a
   small amount. It matters only if it changes which documents come out nearest, or moves
   documents across a cluster boundary. Whether it does depends on how large the precision change
   is compared with the real differences between documents. That has to be measured, not
   assumed. This benchmark measured the first half directly: the same documents were embedded by
   both devices, and each document's two embeddings are compared below. The second half, whether
   rankings or clusters change on this corpus, is a similarity analysis, which is out of scope here.
   It is the check to run before any production embedding if the precision is in doubt.

### What was decided, and by whom

float16 on the GPU was not a choice made inside this benchmark. It comes with serving the models
through TEI, as required here. It was recorded in the earlier TEI test (`tei_dtype:
float16`) but never put to you as a decision, and it should have been. It is not locked in for
production: TEI can run float32 (without flash attention: about twice the model memory and
slower), and a transformers-based build can run bfloat16 or float32. Which precision the
production embedding uses is your decision. The measurements below are the evidence for it.
"""


def assumptions_list(args, prep, g, c):
    items = [
        "**The 21 GiB ceiling** is 21 GiB = 21,504 MiB = 22,548,578,304 bytes of total device memory as NVML reports it, other processes (the Windows desktop) included, as established in the earlier Phase 2 tests. It is enforced inside TEI by the allocation-cap shim, with a 20 ms NVML watchdog and a check of every sample.",
        "**The CPU's 21 GiB** is the memory of the worker's process scope (`MemoryMax` = 22,548,578,304 bytes, `MemorySwapMax` = 0). The kernel kills the worker at the ceiling, which counts as a failure at the ceiling, as a refusal does on the GPU. Page cache charged to the scope counts toward it.",
        "**Concurrency.** Both devices process one batch at a time, as TEI's backend does. The CPU worker uses 8 compute threads, the 8 logical CPUs of the WSL2 VM (set by `.wslconfig`), which are the CPUs the TEI container had during the GPU runs. The WSL configuration was not changed. The CPU tokenizer runs on the same 8 CPUs, and its time is recorded separately.",
        "**CPU implementation.** transformers + PyTorch, not TEI's CPU image: TEI's CPU backend limits every batch to 4 documents (`max_batch_size() = Some(4)` in `backends/candle/src/lib.rs` v1.9.3), which would make the requested batching sweep impossible and impose a document-count limit. The CPU runs use the same model files, tokenizer, last-token pooling, L2 normalisation, no truncation and no prompt. They compute in float32, which is TEI's own CPU default and the fastest precision on this CPU (AVX2, no hardware bfloat16).",
        "**GQA flag on the CPU.** transformers' `use_gqa_in_sdpa` is forced to False on the CPU (keys and values repeated to the query heads, the same arithmetic). Otherwise PyTorch's CPU attention falls back to a kernel that materialises the whole attention matrix. Probe: 79 s and +5.4 GB for the longest document on the 0.6B, against 26 s and +0.27 GB with identical embeddings.",
        "**CPU batch.** A CPU batch is the same kind of planned batch as on the GPU (the same stream and greedy rule, a budget of real tokens). The CPU model pads, so the batch runs in forward passes of at most the budget in padded tokens, documents sorted by length; a document longer than the budget runs alone. Padded tokens are reported per batch. Throughput is computed from real tokens, as on the GPU.",
        "**CPU sweep.** The budget starts at 1,024 and doubles, never above the GPU budget of the same model, with 1 warm-up and 3 timed batches per point and the model loaded once. The sweep stops at the first point without a 5% gain over the best earlier point, at a failure, or at the GPU budget. The final run uses the point with the highest tokens per second.",
        "**Documents.** The benchmark stream is the canonical documents in a seeded order (BLAKE2b of `20260910:<doc_key>`), without the one empty document (`t1_ovhv2aj`, no text to embed). Warm-up batches are packed from the start of the stream and measured batches from its midpoint, so the measured documents of all four configurations begin at the same document. Batches are packed by TEI's own rule: take documents while the sum stays within the budget; the first that does not fit starts the next batch.",
        "**Timing.** GPU: TEI's backend duration of each batch (`x-inference-time`, 1 ms resolution; every input of a request is proven to be in one batch). CPU: the forward passes plus pooling and normalisation. Tokenisation, queueing, HTTP and JSON are excluded on both, and model loading is excluded. Statistics cover the 10 measured batches only. The standard deviation is the sample standard deviation (n - 1).",
        "**GPU batch release.** Two single-document blockers (the two longest documents) start each GPU run; they and the 3 warm-up batches are excluded from timing. Each later batch is sent as soon as TEI has batched every earlier document, so it waits in TEI's queue while the GPU runs the batch before it. TEI's own batch counters then verify every measured batch's document and token counts.",
        "**4B GPU budget.** 92,160 is taken from the TEI test (confirmed twice there). It was not searched again, by design. It was established with the desktop holding about 1.1 to 1.4 GB of VRAM; the desktop's share at run time is recorded with every run.",
        "**0.6B GPU budget.** The search uses the longest documents of the corpus (first-fit-decreasing), as the 4B search did, so the budget is safe for any batch of the corpus. A value counts as safe only after it succeeds twice, the second time in a fresh container.",
        "**Tokenizer.** The 0.6B tokenizer files are not byte-identical to the 4B's; the token ids of every document were compared (result below). The stored 4B counts are used for the 0.6B only if every document's ids are identical.",
        "**Precision check.** It covers the documents embedded in the measured batches of both runs of a model (the CPU's measured documents are the first documents of the GPU's measured batches). It compares each document with itself, not with other documents.",
        "**CPU utilisation.** This VM's CPU-time accounting runs about 4% fast against its wall clock (a Hyper-V guest; one thread busy for 5.000 s was charged 5.200 s of CPU time). Every CPU worker measures the ratio for itself before loading the model (a 2-second single-thread busy loop), and the reported busy CPUs are divided by it; the uncorrected values are kept in the batch tables.",
    ]
    return items


def stage_report(args):
    rdir, edir = results_dir(args), environment_dir(args)
    prep = read_json(rdir / "prepare.json")
    cal = {m: load_optional(gpu_calibration_path(args, m)) for m in MODELS}
    search = load_optional(rdir / "gpu_search_0p6b" / "search_summary.json")
    search_rows = (
        read_csv_rows(rdir / "gpu_search_0p6b" / "tested_values.csv")
        if (rdir / "gpu_search_0p6b" / "tested_values.csv").exists()
        else []
    )
    g = {m: gpu_config(args, m) for m in MODELS}
    c = {m: cpu_config(args, m) for m in MODELS}
    precision = (
        read_csv_rows(rdir / "precision_agreement.csv")
        if (rdir / "precision_agreement.csv").exists()
        else []
    )
    try:
        image = docker_image_record()
        image["recorded_at"] = "report stage"
        write_json(edir / "tei_image.json", image)
    except Exception as e:
        log(f"image record not refreshed: {e!r}")
        image = prep.get("image", {})
    nvml = Nvml()
    env = environment_record(nvml, image)
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
    table, cols = comparison_rows(g, c)
    write_csv(rdir / "comparison_table.csv", table, ["Measurement"] + cols)

    def st(conf):
        return conf["run"].get("steady_state") if conf else None

    L = []
    L.append(
        "# Phase 2 benchmark report: steady-state throughput of Qwen3-Embedding-4B and 0.6B on GPU and CPU"
    )
    L.append("")
    L.append(
        f"Generated {utc_now()} by `scripts/throughput_benchmark.py` (stage `report`). Every number is measured on this machine; nothing is estimated or extrapolated."
    )
    if args.quick:
        L.append("")
        L.append(
            "**QUICK RUN (smoke test with reduced settings): these numbers are not the benchmark.**"
        )
    L.append("")
    L.append("## Answers")
    L.append("")
    for label, key in (
        ("4B GPU", ("4B", "g")),
        ("0.6B GPU", ("0.6B", "g")),
        ("4B CPU", ("4B", "c")),
        ("0.6B CPU", ("0.6B", "c")),
    ):
        conf = (g if key[1] == "g" else c).get(key[0])
        s = st(conf)
        if s:
            L.append(
                f"- **{label} steady state: {s['tokens_per_s']:,.0f} tokens/s, {s['documents_per_s']:,.1f} documents/s** ({s['n']} measured batches, {s['total_documents']:,} documents, {s['total_tokens']:,} tokens in {s['total_s']:.2f} s)."
            )
        else:
            L.append(
                f"- **{label}: no complete measured run** ({'not run' if not conf else (conf['run'].get('outcome') or (conf['run'].get('attempts') or [{}])[-1].get('outcome'))})."
            )
    if search and search.get("largest_safe"):
        ls, nf = search["largest_safe"], search.get("nearest_failing_above")
        L.append(
            f"- **0.6B maximum confirmed safe GPU token budget under 21 GiB: {ls['max_batch_tokens']:,}** (peak {_mib(ls['overall_peak_device_bytes'])}; confirmation {'succeeded' if search.get('confirmation_succeeded') else 'NOT confirmed'}; nearest measured failing value above it: {_i(nf['max_batch_tokens']) if nf else 'none measured'})."
        )
    batching = []
    for label, conf in (
        ("4B GPU", g.get("4B")),
        ("0.6B GPU", g.get("0.6B")),
        ("4B CPU", c.get("4B")),
        ("0.6B CPU", c.get("0.6B")),
    ):
        s = st(conf)
        if not conf:
            continue
        budget = conf["run"].get("max_batch_tokens") or conf["run"].get("budget")
        if s:
            batching.append(
                f"{label}: budget {budget:,} tokens, {s['min_documents_in_batch']:,} to {s['max_documents_in_batch']:,} documents and {s['min_tokens_in_batch']:,} to {s['max_tokens_in_batch']:,} tokens per measured batch"
            )
    if batching:
        L.append("- **Measured batching:** " + "; ".join(batching) + ".")
    ram = []
    for label, conf in (("4B GPU", g.get("4B")), ("0.6B GPU", g.get("0.6B"))):
        if conf:
            res = conf["run"]["resources"]
            cm = res.get("container_memory") or {}
            ram.append(
                f"{label}: VRAM peak {_mib(res.get('overall_peak_device_bytes'))} (device total), RAM {_gib(cm.get('memory_peak_bytes') or cm.get('memory_current_peak_sampled_bytes'))} (TEI container)"
            )
    for label, conf in (("4B CPU", c.get("4B")), ("0.6B CPU", c.get("0.6B"))):
        if conf:
            res = conf["run"]["resources"]
            ram.append(
                f"{label}: RAM {_gib(res.get('scope_memory_peak_bytes'))} (worker scope peak), no GPU memory"
            )
    if ram:
        L.append("- **Peak memory:** " + "; ".join(ram) + ".")
    L.append(
        f"- **Speed ratios (tokens/s):** 0.6B GPU vs 4B GPU {ratio_text(g.get('0.6B'), g.get('4B'), 'tokens_per_s')}; 0.6B CPU vs 4B CPU {ratio_text(c.get('0.6B'), c.get('4B'), 'tokens_per_s')}; "
        f"4B GPU vs 4B CPU {ratio_text(g.get('4B'), c.get('4B'), 'tokens_per_s')}; 0.6B GPU vs 0.6B CPU {ratio_text(g.get('0.6B'), c.get('0.6B'), 'tokens_per_s')}."
    )
    L.append(
        f"- **Speed ratios (documents/s):** 0.6B GPU vs 4B GPU {ratio_text(g.get('0.6B'), g.get('4B'), 'documents_per_s')}; 0.6B CPU vs 4B CPU {ratio_text(c.get('0.6B'), c.get('4B'), 'documents_per_s')}; "
        f"4B GPU vs 4B CPU {ratio_text(g.get('4B'), c.get('4B'), 'documents_per_s')}; 0.6B GPU vs 0.6B CPU {ratio_text(g.get('0.6B'), c.get('0.6B'), 'documents_per_s')}."
    )
    L.append(
        "- Scripts and results: see [Files](#files). No full-corpus runtime is extrapolated and no model is chosen."
    )
    L.append("")
    L.append("## Comparison table")
    L.append("")
    L.append(render_table(["Measurement"] + cols, table))
    L.append("")
    L.append("## The 0.6B GPU token budget (directly measured)")
    L.append("")
    if search:
        ls, nf, cf = (
            search.get("largest_safe"),
            search.get("nearest_failing_above"),
            search.get("confirmation"),
        )
        if ls:
            L.append(
                f"- Largest directly confirmed safe `max_batch_tokens`: **{ls['max_batch_tokens']:,}** ({ls['n_documents']:,} of the longest documents, {ls['batch_tokens']:,} tokens in one verified TEI batch)."
            )
            L.append(
                f"- Peak VRAM at that value: {_mib(ls['overall_peak_device_bytes'])} device total (start-up with TEI's warm-up {_mib(ls['startup_peak_device_bytes'])}; the batch {_mib(ls['batch_peak_device_bytes'])})."
            )
            L.append(
                f"- Confirmation run: {'succeeded' if search.get('confirmation_succeeded') else 'NOT confirmed'}"
                + (
                    f", test {cf['seq']}, peak {_mib(cf['overall_peak_device_bytes'])}."
                    if cf
                    else "."
                )
            )
            L.append(
                f"- Nearest directly measured failing value above it: {('**' + format(nf['max_batch_tokens'], ',') + '**' + ' (' + nf['outcome'] + '; ' + nf['failure_detail'][:300] + ')') if nf else 'none measured'}."
            )
        else:
            L.append("- No permitted value succeeded.")
        if search.get("aborted"):
            L.append(f"- Search aborted: {search['aborted']}")
        L.append("")
        cols_s = [
            "seq",
            "purpose",
            "max_batch_tokens",
            "outcome",
            "n_documents",
            "batch_tokens_planned",
            "x_compute_tokens",
            "verified_single_batch",
            "startup_peak_device_mib",
            "batch_peak_device_mib",
            "overall_peak_device_mib",
            "overall_peak_under_ceiling",
            "cap_mib",
            "enforcement",
            "failure_detail",
        ]
        L.append(render_table(cols_s, search_rows))
    else:
        L.append("The search has not run.")
    L.append("")
    L.append("## The batches actually processed")
    L.append("")
    for label, conf in (("4B GPU", g.get("4B")), ("0.6B GPU", g.get("0.6B"))):
        if not conf:
            continue
        L.append(f"### {label}")
        L.append("")
        run = conf["run"]
        L.append(
            f"TEI `max_batch_tokens` {run['max_batch_tokens']:,} ({run['budget_source']}); accepted attempt {run.get('accepted_attempt')}; attempts: "
            + "; ".join(
                f"{a['attempt']}: {a['outcome']}" for a in run.get("attempts", [])
            )
            + ". `tei_documents`/`tei_tokens` are TEI's own counts of the batch (Prometheus deltas); `x_compute_tokens` is TEI's count of the request's tokens; `time_s` is TEI's backend duration of the batch."
        )
        L.append("")
        L.append(
            render_table(
                [
                    "seq",
                    "phase",
                    "phase_index",
                    "n_documents",
                    "n_tokens",
                    "tei_documents",
                    "tei_tokens",
                    "x_compute_tokens",
                    "verified",
                    "time_s",
                    "t_release_s",
                    "t_formed_s",
                    "embeddings_finite",
                ],
                conf["rows"],
            )
        )
        L.append("")
    for label, conf in (("4B CPU", c.get("4B")), ("0.6B CPU", c.get("0.6B"))):
        if not conf:
            continue
        L.append(f"### {label}")
        L.append("")
        run = conf["run"]
        L.append(
            f"Budget {run['budget']:,} real tokens per batch ({run['budget_source']}); outcome {run['outcome']}. `padded_tokens` counts the padding of the forward passes; `busy_cpus` is the scope's CPU time divided by the batch time, corrected for this VM's CPU-time clock (see Resources)."
        )
        L.append("")
        rows = [
            dict(
                r,
                time_s=round(r["time_s"], 3),
                tokenize_s=round(r["tokenize_s"], 3),
                busy_cpus=round(r.get("busy_cpus_corrected") or r["busy_cpus"], 2)
                if r.get("busy_cpus") is not None
                else None,
                vm_hwm=_gib(r.get("vm_hwm_bytes")),
            )
            for r in conf["rows"]
        ]
        L.append(
            render_table(
                [
                    "seq",
                    "phase",
                    "phase_index",
                    "n_documents",
                    "n_tokens",
                    "padded_tokens",
                    "n_forward_passes",
                    "time_s",
                    "tokenize_s",
                    "busy_cpus",
                    "vm_hwm",
                    "embeddings_finite",
                ],
                rows,
            )
        )
        L.append("")
    L.append("## Resources")
    L.append("")
    for label, conf in (("4B GPU", g.get("4B")), ("0.6B GPU", g.get("0.6B"))):
        if not conf:
            continue
        res = conf["run"]["resources"]
        cm = res.get("container_memory") or {}
        L.append(
            f"- **{label}:** other processes held {_mib(res.get('device_used_before_start_bytes'))} before the container started; "
            f"model-loaded baseline (after TEI's start-up warm-up) {_mib(res.get('device_used_at_ready_bytes'))} device total, of which TEI's own physical memory {_mib(res.get('tei_physical_at_ready_bytes'))} (live tensors {_mib(res.get('tei_live_at_ready_bytes'))}); "
            f"peak during TEI's start-up {_mib(res.get('startup_peak_device_bytes'))}; peak during the warm-up batches {_mib(res.get('warmup_peak_device_bytes'))}; peak during the measured batches {_mib(res.get('measured_peak_device_bytes'))}; highest observed {_mib(res.get('overall_peak_device_bytes'))} (ceiling 21,504 MiB); "
            f"TEI's physical peak {_mib(res.get('shim_physical_peak_bytes'))}. System RAM: TEI container peak {_gib(cm.get('memory_peak_bytes'))} (anonymous at the end {_gib((cm.get('memory_stat') or {}).get('anon'))}); VM in use peak {_gib(cm.get('vm_used_peak_bytes'))} (includes the benchmark client, which holds the measured responses)."
        )
    for label, conf in (("4B CPU", c.get("4B")), ("0.6B CPU", c.get("0.6B"))):
        if not conf:
            continue
        run = conf["run"]
        res = run["resources"]
        th = run["worker_threads"]
        mb = res.get("measured_batches") or {}
        L.append(
            f"- **{label}:** worker scope limit {_gib(res.get('scope_memory_limit_bytes'))}, swap limit {th.get('memory_swap_max')}; scope peak {_gib(res.get('scope_memory_peak_bytes'))}; "
            f"worker anonymous memory peak during the measured batches {_gib(res.get('worker_rss_anon_peak_bytes_measured'))} (resident peak {_gib(mb.get('worker_rss_peak_bytes'))}, which also counts shared file pages charged to other processes, such as model-file pages first read by an earlier process); resident after loading {_gib((run.get('worker_load') or {}).get('rss_bytes'))}; "
            f"CPU: {th.get('torch_threads')} PyTorch compute threads (OMP {th.get('omp_num_threads')}, MKL {th.get('mkl_num_threads')}, inter-op pool {th.get('torch_interop_threads')}), up to {res.get('threads_in_process_max')} OS threads in the process; "
            f"average busy CPUs during the measured batches {_f(res.get('measured_busy_cpus_corrected'), 2)} of 8 ({_f((res.get('measured_busy_cpus_corrected') or 0) / 8 * 100, 1)}% utilisation: the scope's CPU time divided by the batch time, corrected for this VM's CPU-time clock, which the worker measured running {_f(((res.get('cpu_clock_ratio') or 1) - 1) * 100, 1)}% fast against the wall clock; uncorrected {_f(res.get('measured_busy_cpus'), 2)} from the scope and {_f(res.get('measured_busy_cpus_process'), 2)} from the process's own CPU time); "
            f"CUDA visible to the worker: {th.get('cuda_available')}; GPU memory in use during the run: {_mib((run['run'].get('peaks') or {}).get('gpu_used_peak_bytes'))} peak, {_mib(run['run'].get('gpu_used_min_bytes'))} minimum (the desktop's share; the worker used none); model load {_f((run.get('worker_load') or {}).get('wall_s'), 1)} s."
        )
    L.append("")
    L.append("## CPU batching sweeps")
    L.append("")
    for model_key in MODELS:
        conf = c.get(model_key)
        sw = (
            conf["sweep"]
            if conf
            else load_optional(
                results_dir(args)
                / f"cpu_sweep_{MODELS[model_key]['slug']}"
                / "sweep.json"
            )
        )
        if not sw:
            continue
        L.append(f"### {model_key}")
        L.append("")
        L.append(
            f"Bound {sw['bound']:,} ({sw['bound_source']}); candidate budgets {sw['candidate_budgets']}; stop: {sw['stop_reason']}; choice: {sw['choice']}. `busy_cpus_mean` is corrected for this VM's CPU-time clock ({_f((sw.get('worker_threads') or {}).get('cpu_clock_ratio'), 3)} CPU seconds per wall second for one busy thread)."
        )
        L.append("")
        L.append(
            render_table(
                [
                    "budget",
                    "status",
                    "tokens_per_s",
                    "documents_per_s",
                    "n_timed_batches",
                    "timed_documents",
                    "timed_tokens",
                    "timed_padded_tokens",
                    "timed_total_s",
                    "busy_cpus_mean",
                    "worker_vm_hwm_peak_mib",
                    "scope_memory_peak_mib",
                    "decision_after",
                    "chosen",
                ],
                sw["points"],
            )
        )
        L.append("")
    L.append("## Precision: float16 on the GPU, float32 on the CPU")
    L.append("")
    L.append(PRECISION_TEXT)
    L.append("### Measured: the same documents embedded on both devices")
    L.append("")
    if precision:
        L.append(
            render_table(
                [
                    "model",
                    "status",
                    "n_compared",
                    "dim",
                    "min",
                    "p0_1",
                    "p1",
                    "median",
                    "mean",
                    "n_below_0_9999",
                    "n_below_0_999",
                    "n_below_0_99",
                    "n_nonfinite",
                    "max_abs_diff_median",
                    "max_abs_diff_max",
                    "gpu_all_finite",
                    "cpu_all_finite",
                ],
                precision,
            )
        )
        L.append("")
        L.append(
            "A cosine of 1 means the two embeddings of a document point in exactly the same direction. The per-document values are in `results/precision_agreement_documents_<model>.csv`."
        )
    else:
        L.append("Not available (a run did not complete).")
    L.append("")
    L.append("## Tokenizer: 0.6B versus the tested 4B")
    L.append("")
    tk = prep["tokenizer"]
    L.append(
        f"- Tokenizer files byte-identical: {tk['tokenizer_files_byte_identical']}; differing `tokenizer.json` sections: {tk['tokenizer_json_sections_differing'] or 'none'}; differing `tokenizer_config.json` keys: {tk['tokenizer_config_keys_differing'] or 'none'}."
    )
    L.append(
        f"- Vocabulary equal: {tk['vocabulary_equal']}; added tokens equal: {tk['added_tokens_equal']}; post-processor equal: {tk['post_processor_equal']}."
    )
    L.append(
        f"- Token ids identical on {tk['n_token_ids_equal']:,} of {tk['n_documents']:,} documents; counts equal to the stored 4B counts: 4B tokenizer {tk['n_equal_to_stored_counts_4B']:,}, 0.6B tokenizer {tk['n_equal_to_stored_counts_0p6B']:,}. Counts used for the 0.6B: {tk['counts_used_for_0p6B']}. Details: `results/tokenizer_identity.csv`."
    )
    for m in MODELS:
        if cal.get(m):
            ck = cal[m]["tokenize_check"]
            L.append(
                f"- TEI's own token count ({m} container, `/tokenize`) equals the stored count for {ck['n_equal']:,} of {ck['n_documents']:,} documents ({ck['n_candidates']:,} longest documents plus {ck['n_stream_sample']:,} stream documents)."
            )
    L.append("")
    L.append("## Assumptions and decisions")
    L.append("")
    for item in assumptions_list(args, prep, g, c):
        L.append(f"- {item}")
    L.append("")
    L.append("## Environment and provenance")
    L.append("")
    L.append(
        f"- TEI `{image.get('image')}`, image id `{image.get('image_id')}`, digest matches `{IMAGE_EXPECTED_DIGEST[:19]}...`: {image.get('digest_matches_expected')}; `{image.get('router_version_output')}`; fixed arguments `{' '.join(TEI_ARGS_FIXED)}`."
    )
    for m in MODELS:
        if cal.get(m):
            info = cal[m]["info"]
            L.append(
                f"- {m} `/info`: dtype {info.get('model_dtype')}, pooling {(info.get('model_type') or {}).get('embedding', {}).get('pooling')}, max_input_length {info.get('max_input_length')}, auto_truncate {info.get('auto_truncate')}, max_batch_requests {info.get('max_batch_requests')}, max_client_batch_size {info.get('max_client_batch_size')}, tokenization_workers {info.get('tokenization_workers')}; untracked device overhead {_mib(cal[m]['untracked_overhead_bytes'])}; embedding dimension {cal[m].get('embedding_dim')}."
            )
    L.append(
        "- Models: "
        + "; ".join(f"`{v['repo']}` @ `{v['revision']}`" for v in MODELS.values())
        + " (local snapshots, `HF_HUB_OFFLINE=1`)."
    )
    L.append(
        f"- GPU {env.get('gpu_name')}, driver {env.get('driver_version')}, kernel {env.get('kernel_release')}; VM CPUs {env.get('cpu_count')}, VM RAM {_gib(env.get('ram_total_bytes'))}; Python {env.get('python_version')}; torch {env.get('package_torch')}, transformers {env.get('package_transformers')}, tokenizers {env.get('package_tokenizers')}."
    )
    L.append(
        f"- Canonical table SHA256 `{prep['input_checks']['canonical']['sha256_measured']}` (equal to its sidecar: {prep['input_checks']['canonical']['hash_equal_to_sidecar']}); token table `{prep['input_checks']['token_table']['sha256_measured']}` ({prep['input_checks']['token_table']['hash_equal_to_sidecar']}); benchmark stream `{prep['stream']['order_sha256']}` ({prep['stream']['n_documents']:,} documents, midpoint {prep['stream']['midpoint']:,})."
    )
    L.append(
        f"- Script SHA256 at this report: `{SCRIPT_SHA256}`; every output's sidecar records the script hash of the process that wrote it."
    )
    L.append("")
    L.append("## Files")
    L.append("")
    base = rel_path(BENCH_DIR, args.project_dir)
    for rel in (
        "scripts/throughput_benchmark.py",
        "scripts/download_qwen3_embedding_0p6b.py",
        "tests/test_throughput_benchmark_functions.py",
        "README.md",
        "REPORT.md",
        "results/prepare.json",
        "results/tokenizer_identity.csv",
        "results/gpu_calibration_4b.json",
        "results/gpu_calibration_0p6b.json",
        "results/gpu_search_0p6b/ (search_summary.json, tested_values.csv, batch_compositions.csv, tests/, container_logs/, shim_reports/)",
        "results/gpu_throughput_4b/ and results/gpu_throughput_0p6b/ (run.json, attempt_<n>.json, batches.csv, batch_documents.csv, container_attempt_<n>.log, shim_attempt_<n>.json)",
        "results/cpu_sweep_4b/ and results/cpu_sweep_0p6b/ (sweep.json, points.csv, batches.csv, worker_record.json, worker.log)",
        "results/cpu_throughput_4b/ and results/cpu_throughput_0p6b/ (run.json, batches.csv, batch_documents.csv, worker_record.json, worker.log)",
        "results/precision_agreement.csv and results/precision_agreement_documents_<model>.csv",
        "results/comparison_table.csv",
        "environment/ (tei_image.json, docker_versions.txt, nvidia_smi.txt, vram_cap_build.txt, model_snapshots.json, model_snapshot_0p6b.json, host.json, runtime_record.csv, python_packages.csv)",
    ):
        L.append(f"- `{base}/{rel}`")
    L.append(
        f"- The allocation-cap shim used by the GPU runs: `{rel_path(SHIM_SRC_DIR, args.project_dir)}/`."
    )
    L.append("")
    L.append("## Not done, by design")
    L.append("")
    L.append(
        "No full-corpus embedding, no runtime extrapolation, no similarity analysis between documents, no clustering or indexing, no choice between the two models. The raw data, Phase 1 and `legacy/scripts/` were not modified. The 8B model was not tested."
    )
    L.append("")
    report_path = Path(args.report_path)
    report_path.write_text("\n".join(L), encoding="utf-8")
    sidecars = Sidecars(
        args.project_dir,
        dict(gpu_parameters(args), cpu=cpu_parameters(args)),
        "Throughput benchmark, report stage.",
    )
    inputs = [sidecars.input_entry(rdir / "prepare.json", "prepare stage result")]
    for p in (
        sorted(rdir.glob("*/run.json"))
        + sorted(rdir.glob("*/sweep.json"))
        + sorted(rdir.glob("gpu_calibration_*.json"))
    ):
        inputs.append(sidecars.input_entry(p, f"stage result: {rel_path(p, rdir)}"))
    if (rdir / "gpu_search_0p6b" / "search_summary.json").exists():
        inputs.append(
            sidecars.input_entry(
                rdir / "gpu_search_0p6b" / "search_summary.json",
                "the 0.6B search summary",
            )
        )
    if (rdir / "precision_agreement.csv").exists():
        inputs.append(
            sidecars.input_entry(
                rdir / "precision_agreement.csv", "precision agreement"
            )
        )
    for path, note in (
        (
            rdir / "comparison_table.csv",
            "the comparison table of the four configurations",
        ),
        (
            edir / "runtime_record.csv",
            "host, kernel, driver, GPU, image and library versions",
        ),
        (edir / "python_packages.csv", "every distribution in the venv"),
        (
            edir / "tei_image.json",
            "identity of the TEI image used (re-recorded at the report stage)",
        ),
        (report_path, "the generated report"),
    ):
        if path.exists():
            sidecars.write(path, inputs, notes=note)
    log(f"report written: {report_path}")


# ----------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------


STAGES = [
    "prepare",
    "gpu_calibrate",
    "gpu_search",
    "gpu_throughput",
    "cpu_sweep",
    "cpu_throughput",
    "cpu_worker",
    "compare",
    "report",
    "all",
]


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--stage", choices=STAGES, default="all")
    p.add_argument("--model", choices=["4B", "0.6B", "both"], default="both")
    p.add_argument(
        "--quick",
        action="store_true",
        help="marks the outputs as a smoke test (the reduced settings are passed explicitly)",
    )
    p.add_argument("--project-dir", default=str(DEFAULT_PROJECT_DIR))
    p.add_argument("--results-dir", default=str(BENCH_DIR / "results"))
    p.add_argument("--environment-dir", default=str(BENCH_DIR / "environment"))
    p.add_argument("--report-path", default=str(BENCH_DIR / "REPORT.md"))
    p.add_argument(
        "--work-dir",
        default="~/throughput_bench_work",
        help="where the CPU worker's plan (with document texts) is written; outside the repository",
    )
    p.add_argument("--hub-dir", default="~/.cache/huggingface/hub")
    p.add_argument("--shim-build-dir", default="~/vram_cap/build")
    p.add_argument("--shim-report-dir", default="~/vram_cap/reports")
    p.add_argument(
        "--ceiling-mib",
        type=int,
        default=21504,
        help="GPU device ceiling (total device memory, other processes included)",
    )
    p.add_argument(
        "--ram-ceiling-mib",
        type=int,
        default=21504,
        help="MemoryMax of the CPU worker's scope",
    )
    p.add_argument("--n-warmup", type=int, default=3)
    p.add_argument("--n-measured", type=int, default=10)
    p.add_argument(
        "--budget",
        type=int,
        default=None,
        help="GPU max_batch_tokens override (smoke tests only)",
    )
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--growth", type=int, default=2)
    p.add_argument("--max-tests", type=int, default=40)
    p.add_argument("--calibration-cap-gib", type=float, default=17.0)
    p.add_argument("--n-candidates", type=int, default=5000)
    p.add_argument("--tokenize-chunk", type=int, default=128)
    p.add_argument("--tokenize-sample", type=int, default=2000)
    p.add_argument("--tokenize-batch-size", type=int, default=4096)
    p.add_argument("--nvml-interval-ms", type=int, default=20)
    p.add_argument("--poll-ms", type=int, default=5)
    p.add_argument("--blocker-gap-ms", type=int, default=30)
    p.add_argument("--attempts", type=int, default=3)
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--startup-timeout-s", type=int, default=900)
    p.add_argument("--request-timeout-s", type=int, default=1800)
    p.add_argument("--pipeline-timeout-s", type=int, default=900)
    p.add_argument("--settle-s", type=int, default=30)
    p.add_argument("--settle-margin-mib", type=int, default=300)
    p.add_argument("--watchdog-grace-s", type=float, default=3.0)
    p.add_argument(
        "--keep-embeddings",
        type=int,
        default=30000,
        help="documents of the measured GPU batches whose embeddings are kept for the precision check",
    )
    p.add_argument("--cpu-threads", type=int, default=8)
    p.add_argument("--cpu-sweep-start", type=int, default=1024)
    p.add_argument("--cpu-sweep-warmup", type=int, default=1)
    p.add_argument("--cpu-sweep-timed", type=int, default=3)
    p.add_argument("--cpu-min-gain", type=float, default=0.05)
    p.add_argument(
        "--cpu-max-budget",
        type=int,
        default=None,
        help="upper bound of the CPU sweep override (smoke tests only; default: the GPU budget of the same model)",
    )
    p.add_argument(
        "--cpu-budget",
        type=int,
        default=None,
        help="CPU batch budget override for cpu_throughput (default: the sweep's choice)",
    )
    p.add_argument("--cpu-sample-ms", type=int, default=100)
    p.add_argument("--worker-timeout-s", type=int, default=8 * 3600)
    p.add_argument("--plan", help=argparse.SUPPRESS)
    p.add_argument("--worker-out", help=argparse.SUPPRESS)
    p.add_argument("--worker-embeddings", help=argparse.SUPPRESS)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    args.project_dir = str(Path(args.project_dir).resolve())
    args.results_dir = str(Path(args.results_dir).expanduser().resolve())
    args.environment_dir = str(Path(args.environment_dir).expanduser().resolve())
    args.report_path = str(Path(args.report_path).expanduser().resolve())
    args.hub_dir = str(Path(args.hub_dir).expanduser())
    models = ["4B", "0.6B"] if args.model == "both" else [args.model]
    try:
        if args.stage == "cpu_worker":
            stage_cpu_worker(args)
            return
        if args.stage in ("prepare", "all"):
            stage_prepare(args)
        if args.stage in ("gpu_calibrate", "all"):
            for m in models:
                stage_gpu_calibrate(args, m)
        if args.stage in ("gpu_search", "all") and "0.6B" in models:
            stage_gpu_search(args)
        if args.stage in ("gpu_throughput", "all"):
            for m in models:
                stage_gpu_throughput(args, m)
        if args.stage in ("cpu_sweep", "cpu_throughput", "all"):
            for m in models:
                if args.stage in ("cpu_sweep", "all"):
                    stage_cpu_sweep(args, m)
                if args.stage in ("cpu_throughput", "all"):
                    stage_cpu_throughput(args, m)
        if args.stage in ("compare", "all"):
            stage_compare(args)
        if args.stage in ("report", "all"):
            stage_report(args)
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
