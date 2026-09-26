#!/usr/bin/env python3
"""
Three-day GPU UMAP + HDBSCAN feasibility test (Phase 3, blind semantic
structure discovery).

Runs, on the existing three-day 0.6B embedding matrix (101,722 x 1024, float32,
L2-normalized):

  1. GPU UMAP (RAPIDS cuML)  : n_neighbors=15, n_components=5, min_dist=0.0, metric=cosine
  2. GPU HDBSCAN (RAPIDS cuML): min_cluster_size=10, min_samples=10, metric=euclidean,
                                cluster_selection_method=eom

on the 5-D UMAP output. The analytical settings above are FIXED by the task and
are never tuned or swept here.

The script measures, for each operation, wall-clock runtime, peak WHOLE-CARD GPU
memory (NVML, all processes on the device), and peak host RAM; preserves every
HDBSCAN diagnostic cuML exposes (labels, membership probabilities, cluster
persistence, condensed / single-linkage / minimum-spanning trees); records
directly observed effects of repeated / tied embeddings; and writes keyed
outputs + provenance manifests + diagnostic figures.

GPU memory ceiling: whole-card usage must stay <= --ceiling-mib (default 21504
MiB = 21 GiB). We do NOT probe or maximize usage; we measure. If a fit cannot run
within the limit (OOM, or measured peak over the ceiling), we stop and report the
measured failure rather than changing any analytical setting.

Reproducibility note: cuML UMAP is run with the fixed config's default
random_state=None. Per the cuML docs, setting random_state "enables reproducible
embeddings, but at the cost of slower training and increased memory usage", and
for N>50K build_algo='auto' selects NN-Descent whose KNN graph is itself
non-deterministic. Because this is a runtime/peak-memory feasibility measurement,
seeding would distort the very quantities being measured, so the run is left
unseeded and outputs are therefore run-specific. HDBSCAN is deterministic given
its input (subject to tie-breaking among equal distances).
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import subprocess
import threading
import time
import warnings
from datetime import datetime, timezone

import numpy as np

# ----------------------------------------------------------------------------
# Fixed task configuration (do NOT change these values)
# ----------------------------------------------------------------------------
UMAP_DEFAULTS = dict(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine")
HDBSCAN_DEFAULTS = dict(
    min_cluster_size=10,
    min_samples=10,
    metric="euclidean",
    cluster_selection_method="eom",
)
EXPECTED_ROWS = 101722
EXPECTED_DIM = 1024
FULL_CORPUS_ROWS = 902144

REPO_ROOT = os.environ.get("KC_REPO_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..")))


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    print(f"[{utcnow()}] {msg}", flush=True)


def relpath(p: str) -> str:
    """Repo-root-relative path with forward slashes (for manifests)."""
    ap = os.path.abspath(p).replace("\\", "/")
    root = REPO_ROOT.replace("\\", "/").rstrip("/") + "/"
    return ap[len(root) :] if ap.startswith(root) else ap


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


# ----------------------------------------------------------------------------
# NVML (whole-card GPU memory) + host RAM sampler
# ----------------------------------------------------------------------------
class ResourceSampler:
    """Background thread: samples whole-card GPU memory (NVML) and this
    process's host RSS at a fixed interval. Records peaks, a time series, a
    ceiling-breach flag, and whether this PID is reported as a compute process
    (often unavailable under WSL2/WDDM -- absence is not proof of no GPU use)."""

    def __init__(self, handle, pynvml, proc, interval_ms: int, ceiling_mib: float):
        self.handle = handle
        self.pynvml = pynvml
        self.proc = proc
        self.interval = max(interval_ms, 5) / 1000.0
        self.ceiling_mib = ceiling_mib
        self._stop = threading.Event()
        self._thread = None
        self.samples = []  # list of (t_rel_s, gpu_used_mib, host_rss_mib)
        self.gpu_peak_mib = 0.0
        self.host_rss_peak_mib = 0.0
        self.ceiling_breached = False
        self.breach_first_mib = None
        self.pid_seen_on_gpu = False
        self._t0 = None

    def _read_gpu_mib(self) -> float:
        mi = self.pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        return mi.used / (1024.0 * 1024.0)

    def _check_pid(self) -> None:
        try:
            procs = self.pynvml.nvmlDeviceGetComputeRunningProcesses(self.handle)
            mypid = os.getpid()
            if any(getattr(p, "pid", None) == mypid for p in procs):
                self.pid_seen_on_gpu = True
        except Exception:
            pass  # WSL2/WDDM commonly does not report per-process info

    def _loop(self):
        self._t0 = time.perf_counter()
        while not self._stop.is_set():
            try:
                gpu = self._read_gpu_mib()
            except Exception:
                gpu = float("nan")
            rss = self.proc.memory_info().rss / (1024.0 * 1024.0)
            t = time.perf_counter() - self._t0
            self.samples.append((t, gpu, rss))
            if gpu == gpu:  # not NaN
                if gpu > self.gpu_peak_mib:
                    self.gpu_peak_mib = gpu
                if gpu > self.ceiling_mib and not self.ceiling_breached:
                    self.ceiling_breached = True
                    self.breach_first_mib = gpu
            if rss > self.host_rss_peak_mib:
                self.host_rss_peak_mib = rss
            if not self.pid_seen_on_gpu:
                self._check_pid()
            self._stop.wait(self.interval)

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)


def measure(op_name, fn, handle, pynvml, proc, interval_ms, ceiling_mib):
    """Run fn() with resource sampling; return (result, metrics_dict)."""
    baseline_mib = pynvml.nvmlDeviceGetMemoryInfo(handle).used / (1024.0 * 1024.0)
    sampler = ResourceSampler(handle, pynvml, proc, interval_ms, ceiling_mib)
    log(
        f"{op_name}: baseline whole-card GPU used = {baseline_mib:.1f} MiB; starting ..."
    )
    captured_warnings = []
    sampler.start()
    t0_wall, t0_perf = time.time(), time.perf_counter()
    err = None
    result = None
    try:
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            result = fn()
            captured_warnings = [f"{w.category.__name__}: {w.message}" for w in wlist]
    except Exception as e:  # includes CUDA OOM
        err = f"{type(e).__name__}: {e}"
    finally:
        perf_s = time.perf_counter() - t0_perf
        wall_s = time.time() - t0_wall
        sampler.stop()
    metrics = dict(
        op=op_name,
        ok=(err is None),
        error=err,
        wall_clock_s=round(perf_s, 4),
        wall_clock_time_s=round(wall_s, 4),
        gpu_baseline_mib=round(baseline_mib, 1),
        gpu_peak_mib=round(sampler.gpu_peak_mib, 1),
        gpu_peak_gib=round(sampler.gpu_peak_mib / 1024.0, 4),
        gpu_delta_mib=round(sampler.gpu_peak_mib - baseline_mib, 1),
        host_rss_peak_mib=round(sampler.host_rss_peak_mib, 1),
        host_rss_peak_gib=round(sampler.host_rss_peak_mib / 1024.0, 4),
        ceiling_mib=ceiling_mib,
        ceiling_breached=sampler.ceiling_breached,
        breach_first_mib=(
            round(sampler.breach_first_mib, 1)
            if sampler.breach_first_mib is not None
            else None
        ),
        pid_seen_on_gpu=sampler.pid_seen_on_gpu,
        n_samples=len(sampler.samples),
        warnings=captured_warnings,
    )
    log(
        f"{op_name}: done ok={metrics['ok']} wall={perf_s:.2f}s "
        f"peak_gpu={metrics['gpu_peak_gib']:.3f} GiB (delta {metrics['gpu_delta_mib']:.0f} MiB) "
        f"peak_rss={metrics['host_rss_peak_gib']:.3f} GiB breach={metrics['ceiling_breached']}"
    )
    return result, metrics, sampler.samples


def free_gpu(cupy_mod):
    gc.collect()
    if cupy_mod is not None:
        try:
            cupy_mod.get_default_memory_pool().free_all_blocks()
            cupy_mod.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass


# ----------------------------------------------------------------------------
# Provenance manifest (matches the project's sidecar format)
# ----------------------------------------------------------------------------
def software_block(extra_note=None):
    import importlib.metadata as im

    names = [
        "cuml",
        "cupy-cuda12x",
        "cupy",
        "numpy",
        "pandas",
        "pyarrow",
        "matplotlib",
        "hdbscan",
        "scipy",
        "scikit-learn",
        "nvidia-ml-py",
        "psutil",
        "pyyaml",
    ]
    pkgs = []
    seen = set()
    for n in names:
        try:
            v = im.version(n)
        except Exception:
            continue
        key = n.replace("cupy-cuda12x", "cupy")
        if key in seen:
            continue
        seen.add(key)
        pkgs.append({"name": key, "version": v})
    return {
        "language": "Python",
        "language_version": platform.python_version(),
        "packages": pkgs,
        "os": platform.platform(),
    }


def write_manifest(
    output_path,
    inputs,
    script_path,
    parameters,
    git_commit,
    *,
    tabular=None,
    out_format=None,
    notes="",
    seed=None,
    schema_diff=None,
):
    """Write <output_path>.manifest.yml. `tabular` = (rows, cols) or None for
    a lite manifest (npy / tree object / image)."""
    import yaml

    man = {
        "manifest_version": 1,
        "output_file": relpath(output_path),
        "output_hash": sha256_file(output_path),
    }
    if out_format is not None:
        man["output_format"] = out_format
    if tabular is not None:
        man["output_format"] = out_format or "parquet"
        man["output_rows"] = int(tabular[0])
        man["output_cols"] = int(tabular[1])
    man["input_files"] = inputs
    man["transformation"] = {
        "script": relpath(script_path),
        "script_hash": sha256_file(script_path),
        "parameters": parameters,
        "git_commit": git_commit,
    }
    man["software"] = software_block()
    if tabular is not None and schema_diff is not None:
        man["schema_diff"] = schema_diff
    man["seed"] = seed
    man["timestamp"] = utcnow()
    man["notes"] = notes
    with open(output_path + ".manifest.yml", "w") as f:
        yaml.safe_dump(man, f, sort_keys=False, default_flow_style=False)


def git_short_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", REPO_ROOT, "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


# ----------------------------------------------------------------------------
# Parquet / npy writers (pyarrow; zstd to match project convention)
# ----------------------------------------------------------------------------
def write_parquet(path, columns: dict):
    import pyarrow as pa
    import pyarrow.parquet as pq

    tbl = pa.table(columns)
    pq.write_table(tbl, path, compression="zstd")
    return tbl.num_rows, tbl.num_columns


# ----------------------------------------------------------------------------
# Duplicate / tie observation (directly measured; no interpretation)
# ----------------------------------------------------------------------------
def duplicate_stats(arr: np.ndarray) -> dict:
    uniq, counts = np.unique(arr, axis=0, return_counts=True)
    n = arr.shape[0]
    n_unique = int(uniq.shape[0])
    tied = counts[counts > 1]
    return dict(
        n_rows=int(n),
        n_unique_rows=n_unique,
        n_rows_in_tied_groups=int(counts[counts > 1].sum()) if tied.size else 0,
        n_tied_groups=int(tied.size),
        max_tie_group_size=int(counts.max()) if counts.size else 0,
    )


# ----------------------------------------------------------------------------
# Figures (diagnostics only; title + axis/tick labels, no captions/annotations)
# ----------------------------------------------------------------------------
def make_figures(
    fig_dir, summary, cluster_sizes, probabilities, persistence, measurements
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(fig_dir, exist_ok=True)

    # 1. cluster-size distribution (histogram of per-cluster sizes, log-y)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if cluster_sizes.size:
        ax.hist(cluster_sizes, bins=min(60, max(10, cluster_sizes.size)))
    ax.set_yscale("log")
    ax.set_xlabel("Cluster size (members)")
    ax.set_ylabel("Number of clusters")
    ax.set_title("HDBSCAN cluster-size distribution")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig_cluster_size_distribution.png"), dpi=150)
    plt.close(fig)

    # 2. assigned vs noise (label -1) counts
    fig, ax = plt.subplots(figsize=(6, 4.5))
    assigned = summary["hdbscan"]["assigned_count"]
    noise = summary["hdbscan"]["noise_count"]
    ax.bar(["Assigned", "Noise (-1)"], [assigned, noise])
    ax.set_ylabel("Number of records")
    ax.set_title("Assigned vs noise records")
    for i, v in enumerate([assigned, noise]):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig_assigned_vs_noise.png"), dpi=150)
    plt.close(fig)

    # 3. membership-probability distribution
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if probabilities is not None and probabilities.size:
        ax.hist(probabilities, bins=50, range=(0.0, 1.0))
    ax.set_xlabel("HDBSCAN membership probability")
    ax.set_ylabel("Number of records")
    ax.set_title("Membership-probability distribution")
    fig.tight_layout()
    fig.savefig(
        os.path.join(fig_dir, "fig_membership_probability_distribution.png"), dpi=150
    )
    plt.close(fig)

    # 4. cluster-persistence distribution
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if persistence is not None and persistence.size:
        ax.hist(persistence, bins=50, range=(0.0, 1.0))
    ax.set_xlabel("Cluster persistence")
    ax.set_ylabel("Number of clusters")
    ax.set_title("Cluster-persistence distribution")
    fig.tight_layout()
    fig.savefig(
        os.path.join(fig_dir, "fig_cluster_persistence_distribution.png"), dpi=150
    )
    plt.close(fig)

    # 5. runtime by operation
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ops = [m["op"] for m in measurements]
    walls = [m["wall_clock_s"] for m in measurements]
    ax.bar(ops, walls)
    ax.set_ylabel("Wall-clock runtime (s)")
    ax.set_title("Runtime by operation")
    for i, v in enumerate(walls):
        ax.text(i, v, f"{v:.1f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig_runtime_by_operation.png"), dpi=150)
    plt.close(fig)

    # 6. peak whole-card GPU memory by operation
    fig, ax = plt.subplots(figsize=(6, 4.5))
    peaks = [m["gpu_peak_gib"] for m in measurements]
    ax.bar(ops, peaks)
    ax.axhline(summary["config"]["ceiling_mib"] / 1024.0, linestyle="--")
    ax.set_ylabel("Peak whole-card GPU memory (GiB)")
    ax.set_title("Peak GPU memory by operation")
    for i, v in enumerate(peaks):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig_peak_gpu_memory_by_operation.png"), dpi=150)
    plt.close(fig)

    log(f"figures: wrote 6 diagnostic PNGs to {fig_dir}")


def full_dataset_estimate(measurements, umap_meta, hdbscan_meta):
    """Measured -> extrapolated resource estimate for FULL_CORPUS_ROWS. Keeps
    measured, extrapolated, assumptions, and not-estimable strictly separate."""
    n_small = EXPECTED_ROWS
    n_full = FULL_CORPUS_ROWS
    ratio = n_full / n_small
    import math

    nlogn = (n_full * math.log(n_full)) / (n_small * math.log(n_small))
    m_umap = next((m for m in measurements if m["op"] == "umap"), None)
    m_hdb = next((m for m in measurements if m["op"] == "hdbscan"), None)
    out = {
        "measured": {
            "n_rows": n_small,
            "umap_wall_clock_s": m_umap["wall_clock_s"] if m_umap else None,
            "umap_peak_gpu_gib": m_umap["gpu_peak_gib"] if m_umap else None,
            "hdbscan_wall_clock_s": m_hdb["wall_clock_s"] if m_hdb else None,
            "hdbscan_peak_gpu_gib": m_hdb["gpu_peak_gib"] if m_hdb else None,
        },
        "target_n_rows": n_full,
        "scale_ratio_linear": round(ratio, 4),
        "scale_ratio_n_log_n": round(nlogn, 4),
        "extrapolated_runtime_s": {},
        "assumptions": [
            "UMAP full runtime is bracketed between linear (N) and N*log(N) scaling; "
            "the 101,722-row run uses build_algo='auto' -> NN-Descent (approximate KNN), "
            "whose construction is ~N*log(N)-ish, while the fixed-epoch optimization phase "
            "is ~linear in N. A single measured point cannot separate these, so a range is given.",
            "HDBSCAN here uses the cuML default build_algo='brute_force' KNN. Brute-force "
            "KNN cost grows ~O(N^2) in the number of points, so HDBSCAN runtime is expected "
            "to grow SUPERLINEARLY and the linear figure is a lower bound only.",
            "GPU peak memory does not follow the runtime scaling. UMAP peak scales roughly "
            "with N * n_neighbors (graph) plus N * n_components (embedding); HDBSCAN "
            "brute-force peak can scale with N^2 intermediate distances unless partitioned.",
        ],
        "not_reliably_estimable_from_one_run": [
            "Absolute UMAP runtime at 902,144 rows (scaling exponent unmeasured; only one N).",
            "Whether HDBSCAN brute-force KNN stays within the 21 GiB whole-card ceiling at "
            "902,144 rows (the O(N^2) distance work is the risk; not measured here).",
            "Effect of the corpus's heavy exact-duplicate structure at full scale on cluster "
            "sizes, tie-breaking, and memory.",
        ],
    }
    if m_umap and m_umap["wall_clock_s"]:
        out["extrapolated_runtime_s"]["umap_linear"] = round(
            m_umap["wall_clock_s"] * ratio, 1
        )
        out["extrapolated_runtime_s"]["umap_n_log_n"] = round(
            m_umap["wall_clock_s"] * nlogn, 1
        )
    if m_hdb and m_hdb["wall_clock_s"]:
        out["extrapolated_runtime_s"]["hdbscan_linear_lower_bound"] = round(
            m_hdb["wall_clock_s"] * ratio, 1
        )
        out["extrapolated_runtime_s"]["hdbscan_quadratic"] = round(
            m_hdb["wall_clock_s"] * ratio * ratio, 1
        )
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument(
        "--embeddings",
        default=f"{REPO_ROOT}/pipeline/phase2_semantic_embeddings/three_day_embedding_0p6b/results/embeddings.npy",
    )
    ap.add_argument(
        "--index",
        default=f"{REPO_ROOT}/pipeline/phase2_semantic_embeddings/three_day_embedding_0p6b/results/embedding_index.parquet",
    )
    ap.add_argument("--results-dir", default=os.path.join(base, "results"))
    ap.add_argument("--figures-dir", default=os.path.join(base, "figures"))
    ap.add_argument("--environment-dir", default=os.path.join(base, "environment"))
    ap.add_argument("--ceiling-mib", type=float, default=21504.0)
    ap.add_argument("--nvml-interval-ms", type=int, default=50)
    ap.add_argument(
        "--n-rows",
        type=int,
        default=0,
        help="0 = all rows; >0 = subset (smoke test only)",
    )
    ap.add_argument(
        "--quick",
        action="store_true",
        help="smoke mode: small subset unless --n-rows given",
    )
    ap.add_argument(
        "--figures-only", action="store_true", help="rebuild figures from saved results"
    )
    # Fixed analytical settings (defaults ARE the task spec; not tuned).
    ap.add_argument(
        "--umap-n-neighbors", type=int, default=UMAP_DEFAULTS["n_neighbors"]
    )
    ap.add_argument(
        "--umap-n-components", type=int, default=UMAP_DEFAULTS["n_components"]
    )
    ap.add_argument("--umap-min-dist", type=float, default=UMAP_DEFAULTS["min_dist"])
    ap.add_argument("--umap-metric", default=UMAP_DEFAULTS["metric"])
    ap.add_argument(
        "--hdbscan-min-cluster-size",
        type=int,
        default=HDBSCAN_DEFAULTS["min_cluster_size"],
    )
    ap.add_argument(
        "--hdbscan-min-samples", type=int, default=HDBSCAN_DEFAULTS["min_samples"]
    )
    ap.add_argument("--hdbscan-metric", default=HDBSCAN_DEFAULTS["metric"])
    ap.add_argument(
        "--hdbscan-cluster-selection-method",
        default=HDBSCAN_DEFAULTS["cluster_selection_method"],
    )
    args = ap.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.figures_dir, exist_ok=True)
    os.makedirs(args.environment_dir, exist_ok=True)
    script_path = os.path.abspath(__file__)
    summary_path = os.path.join(args.results_dir, "summary.json")

    if args.figures_only:
        with open(summary_path) as f:
            summary = json.load(f)
        cs = (
            np.loadtxt(os.path.join(args.results_dir, "_cluster_sizes.txt"))
            if os.path.exists(os.path.join(args.results_dir, "_cluster_sizes.txt"))
            else np.array([])
        )
        import pyarrow.parquet as pq

        labels_tbl = pq.read_table(
            os.path.join(args.results_dir, "hdbscan_labels.parquet")
        )
        probabilities = labels_tbl.column("probability").to_numpy()
        pers = np.array([])
        pcsv = os.path.join(args.results_dir, "hdbscan_cluster_persistence.csv")
        if os.path.exists(pcsv):
            import csv

            with open(pcsv) as f:
                pers = np.array([float(r["persistence"]) for r in csv.DictReader(f)])
        make_figures(
            args.figures_dir,
            summary,
            np.atleast_1d(cs),
            probabilities,
            pers,
            summary["measurements"],
        )
        log("figures-only: done")
        return

    n_rows = args.n_rows if args.n_rows > 0 else (3000 if args.quick else 0)

    # ---- imports that need the GPU ----
    import psutil
    import pynvml

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    gpu_name = pynvml.nvmlDeviceGetName(handle)
    if isinstance(gpu_name, bytes):
        gpu_name = gpu_name.decode()
    driver = pynvml.nvmlSystemGetDriverVersion()
    if isinstance(driver, bytes):
        driver = driver.decode()
    total_mib = pynvml.nvmlDeviceGetMemoryInfo(handle).total / (1024.0 * 1024.0)
    log(f"GPU: {gpu_name} | driver {driver} | total {total_mib:.0f} MiB")
    if "3090" not in gpu_name:
        log(f"WARNING: expected RTX 3090, NVML reports '{gpu_name}'")
    proc = psutil.Process(os.getpid())

    import cuml
    import cupy

    cuml.set_global_output_type("numpy")
    from cuml.manifold import UMAP
    from cuml.cluster import HDBSCAN

    log(
        f"cuml {cuml.__version__} | cupy {cupy.__version__} | "
        f"cuda runtime {cupy.cuda.runtime.runtimeGetVersion()} driver {cupy.cuda.runtime.driverGetVersion()}"
    )

    # ---- load inputs (read-only) ----
    log(f"loading embeddings: {args.embeddings}")
    t = time.perf_counter()
    X = np.load(args.embeddings, mmap_mode=None)
    log(f"embeddings loaded {X.shape} {X.dtype} in {time.perf_counter() - t:.1f}s")
    assert X.dtype == np.float32, f"expected float32, got {X.dtype}"
    if n_rows == 0:
        assert X.shape == (EXPECTED_ROWS, EXPECTED_DIM), f"unexpected shape {X.shape}"
    X = np.ascontiguousarray(X)

    import pyarrow.parquet as pq

    idx_tbl = pq.read_table(args.index)
    idx_rows = idx_tbl.column("row").to_numpy()
    doc_keys = np.array(idx_tbl.column("doc_key").to_pylist(), dtype=object)
    record_type = np.array(idx_tbl.column("record_type").to_pylist(), dtype=object)
    assert len(idx_rows) == X.shape[0] or n_rows > 0, "index/embeddings row mismatch"
    order_ok = bool(
        np.array_equal(
            idx_rows[: X.shape[0]] if n_rows == 0 else idx_rows[:n_rows],
            np.arange(min(len(idx_rows), X.shape[0] if n_rows == 0 else n_rows)),
        )
    )

    if n_rows > 0:
        X = X[:n_rows]
        doc_keys = doc_keys[:n_rows]
        record_type = record_type[:n_rows]
        idx_rows = idx_rows[:n_rows]
    N = X.shape[0]
    rows_col = np.arange(N, dtype=np.int32)
    all_finite = bool(np.isfinite(X).all())
    log(
        f"working set: N={N} dim={X.shape[1]} finite={all_finite} row_order_ok={order_ok}"
    )

    # ---- input duplicate/tie observation ----
    log("computing input duplicate/tie statistics ...")
    in_dups = duplicate_stats(X)
    log(f"input duplicates: {in_dups}")

    inputs_block = [
        {
            "path": relpath(args.embeddings),
            "hash": sha256_file(args.embeddings),
            "format": "numpy",
            "role": "three-day 0.6B embeddings (float32, L2-normalized), read-only",
            "rows": int(X.shape[0]) if n_rows else EXPECTED_ROWS,
            "cols": int(X.shape[1]),
        },
        {
            "path": relpath(args.index),
            "hash": sha256_file(args.index),
            "format": "parquet",
            "role": "row -> doc_key index for the embeddings, read-only",
            "rows": len(idx_tbl),
            "cols": idx_tbl.num_columns,
        },
    ]
    git_commit = git_short_head()

    measurements = []
    all_samples = {}

    # ------------------------------------------------------------------ UMAP
    umap_params = dict(
        n_neighbors=args.umap_n_neighbors,
        n_components=args.umap_n_components,
        min_dist=args.umap_min_dist,
        metric=args.umap_metric,
    )
    log(f"UMAP params (fixed): {umap_params}")
    reducer = UMAP(**umap_params)

    def _run_umap():
        return reducer.fit_transform(X)

    emb, umap_metrics, umap_samples = measure(
        "umap", _run_umap, handle, pynvml, proc, args.nvml_interval_ms, args.ceiling_mib
    )
    measurements.append(umap_metrics)
    all_samples["umap"] = umap_samples
    if not umap_metrics["ok"]:
        _finalize_failure(
            args,
            summary_path,
            "umap",
            umap_metrics,
            measurements,
            gpu_name,
            driver,
            total_mib,
            umap_params,
            None,
            in_dups,
            N,
            git_commit,
        )
        return
    emb = np.ascontiguousarray(np.asarray(emb, dtype=np.float32))
    assert emb.shape == (N, args.umap_n_components), f"UMAP output shape {emb.shape}"
    umap_build_algo = (
        "brute_force_knn (auto, N<=50000)"
        if N <= 50000
        else "nn_descent (auto, N>50000)"
    )
    log(f"UMAP output {emb.shape}; auto build_algo -> {umap_build_algo}")

    # save UMAP coords (keyed) + raw npy
    coords_pq = os.path.join(args.results_dir, "umap_5d_coordinates.parquet")
    cols = {
        "row": rows_col,
        "doc_key": doc_keys.astype(str),
        "record_type": record_type.astype(str),
    }
    for j in range(args.umap_n_components):
        cols[f"umap_{j}"] = emb[:, j]
    r, c = write_parquet(coords_pq, cols)
    coords_npy = os.path.join(args.results_dir, "umap_5d_coordinates.npy")
    np.save(coords_npy, emb)
    out_dups = duplicate_stats(emb)
    log(f"UMAP-output duplicates/ties: {out_dups}")

    reducer = None  # drop ref to free UMAP GPU memory (avoid `del` on a closure var)
    free_gpu(cupy)

    # --------------------------------------------------------------- HDBSCAN
    hdbscan_params = dict(
        min_cluster_size=args.hdbscan_min_cluster_size,
        min_samples=args.hdbscan_min_samples,
        metric=args.hdbscan_metric,
        cluster_selection_method=args.hdbscan_cluster_selection_method,
    )
    log(
        f"HDBSCAN params (fixed): {hdbscan_params} (+ gen_min_span_tree=True for MST diagnostic only)"
    )
    clusterer = HDBSCAN(gen_min_span_tree=True, **hdbscan_params)

    def _run_hdbscan():
        clusterer.fit(emb)
        return clusterer

    _, hdb_metrics, hdb_samples = measure(
        "hdbscan",
        _run_hdbscan,
        handle,
        pynvml,
        proc,
        args.nvml_interval_ms,
        args.ceiling_mib,
    )
    measurements.append(hdb_metrics)
    all_samples["hdbscan"] = hdb_samples
    if not hdb_metrics["ok"]:
        _finalize_failure(
            args,
            summary_path,
            "hdbscan",
            hdb_metrics,
            measurements,
            gpu_name,
            driver,
            total_mib,
            umap_params,
            hdbscan_params,
            in_dups,
            N,
            git_commit,
        )
        return

    labels = np.asarray(clusterer.labels_)
    probabilities = np.asarray(clusterer.probabilities_, dtype=np.float64)
    persistence = np.asarray(
        getattr(clusterer, "cluster_persistence_", np.array([])), dtype=np.float64
    )
    assert labels.shape == (N,), f"labels shape {labels.shape}"
    assert probabilities.shape == (N,), f"probabilities shape {probabilities.shape}"
    all_assigned_output = bool(labels.shape[0] == N)

    # ---- cluster statistics (raw labels preserved, including -1) ----
    noise_mask = labels == -1
    assigned_mask = ~noise_mask
    uniq_labels = np.unique(labels[assigned_mask])
    n_clusters = int(uniq_labels.size)
    sizes = np.array(
        [int((labels == lab).sum()) for lab in uniq_labels], dtype=np.int64
    )
    assigned_count = int(assigned_mask.sum())
    noise_count = int(noise_mask.sum())

    def q(a, p):
        return float(np.percentile(a, p)) if a.size else None

    size_summary = dict(
        n_clusters=n_clusters,
        min=int(sizes.min()) if sizes.size else None,
        q1=q(sizes, 25),
        median=q(sizes, 50),
        mean=float(sizes.mean()) if sizes.size else None,
        q3=q(sizes, 75),
        max=int(sizes.max()) if sizes.size else None,
    )

    # ---- save keyed labels + probabilities ----
    labels_pq = os.path.join(args.results_dir, "hdbscan_labels.parquet")
    lr, lc = write_parquet(
        labels_pq,
        {
            "row": rows_col,
            "doc_key": doc_keys.astype(str),
            "record_type": record_type.astype(str),
            "label": labels.astype(np.int64),
            "probability": probabilities,
        },
    )

    # ---- persistence + sizes table ----
    pers_csv = os.path.join(args.results_dir, "hdbscan_cluster_persistence.csv")
    import csv as _csv

    with open(pers_csv, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["cluster_label", "size", "persistence"])
        for i, lab in enumerate(uniq_labels.tolist()):
            pers_val = float(persistence[i]) if i < persistence.size else ""
            w.writerow([lab, int(sizes[i]), pers_val])

    np.savetxt(os.path.join(args.results_dir, "_cluster_sizes.txt"), sizes, fmt="%d")

    # ---- tree diagnostics (if available) ----
    trees = {}

    def _save_tree(attr, fname, colmap_desc):
        try:
            obj = getattr(clusterer, attr)
            df = obj.to_pandas()
            path = os.path.join(args.results_dir, fname)
            write_parquet(path, {c: df[c].to_numpy() for c in df.columns})
            trees[attr] = {
                "saved": True,
                "path": relpath(path),
                "rows": int(len(df)),
                "columns": list(map(str, df.columns)),
            }
            return path, (len(df), len(df.columns))
        except Exception as e:
            trees[attr] = {"saved": False, "error": f"{type(e).__name__}: {e}"}
            log(f"{attr}: not saved ({trees[attr]['error']})")
            return None, None

    ct_path, ct_dim = _save_tree(
        "condensed_tree_", "hdbscan_condensed_tree.parquet", "condensed tree"
    )
    slt_path, slt_dim = _save_tree(
        "single_linkage_tree_", "hdbscan_single_linkage_tree.parquet", "single linkage"
    )
    mst_path, mst_dim = _save_tree(
        "minimum_spanning_tree_", "hdbscan_minimum_spanning_tree.parquet", "MST"
    )

    # ---- assemble summary ----
    summary = {
        "task": "three-day GPU UMAP + HDBSCAN feasibility",
        "timestamp": utcnow(),
        "git_commit": git_commit,
        "n_rows_processed": N,
        "quick_mode": bool(n_rows > 0),
        "row_order_verified": order_ok,
        "input_all_finite": all_finite,
        "gpu": {"name": gpu_name, "driver": driver, "total_mib": round(total_mib, 1)},
        "software": software_block(),
        "cuda": {
            "cupy": cupy.__version__,
            "runtime_version": cupy.cuda.runtime.runtimeGetVersion(),
            "driver_version": cupy.cuda.runtime.driverGetVersion(),
        },
        "config": {
            "umap": umap_params,
            "umap_build_algo_auto_selected": umap_build_algo,
            "umap_random_state": None,
            "hdbscan": hdbscan_params,
            "hdbscan_gen_min_span_tree": True,
            "ceiling_mib": args.ceiling_mib,
            "seed": None,
        },
        "measurements": measurements,
        "umap": {
            "output_shape": list(emb.shape),
            "input_dim": int(X.shape[1]),
            "output_dim": int(emb.shape[1]),
            "all_rows_processed": bool(emb.shape[0] == N),
            "coordinates_parquet": relpath(coords_pq),
            "coordinates_npy": relpath(coords_npy),
        },
        "hdbscan": {
            "n_clusters": n_clusters,
            "assigned_count": assigned_count,
            "assigned_pct": round(100.0 * assigned_count / N, 4),
            "noise_count": noise_count,
            "noise_pct": round(100.0 * noise_count / N, 4),
            "all_rows_have_label": all_assigned_output,
            "cluster_size_summary": size_summary,
            "probabilities_available": True,
            "probability_summary": {
                "min": float(probabilities.min()),
                "mean": float(probabilities.mean()),
                "max": float(probabilities.max()),
                "n_zero": int((probabilities == 0).sum()),
                "n_one": int((probabilities == 1).sum()),
            },
            "cluster_persistence_available": bool(persistence.size > 0),
            "cluster_persistence_summary": {
                "n": int(persistence.size),
                "min": float(persistence.min()) if persistence.size else None,
                "mean": float(persistence.mean()) if persistence.size else None,
                "max": float(persistence.max()) if persistence.size else None,
            },
            "trees": trees,
            "labels_parquet": relpath(labels_pq),
        },
        "repeated_tied_embeddings": {
            "input": in_dups,
            "umap_output": out_dups,
            "largest_cluster_size": size_summary["max"],
            "note": "Directly observed structural/computational quantities only; no interpretation.",
            "umap_warnings": umap_metrics.get("warnings", []),
            "hdbscan_warnings": hdb_metrics.get("warnings", []),
        },
    }
    summary["full_dataset_estimate"] = full_dataset_estimate(
        measurements, umap_metrics, hdb_metrics
    )

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # ---- measurements.csv + gpu sample series ----
    with open(os.path.join(args.results_dir, "measurements.csv"), "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(
            [
                "op",
                "wall_clock_s",
                "gpu_baseline_mib",
                "gpu_peak_mib",
                "gpu_peak_gib",
                "gpu_delta_mib",
                "host_rss_peak_mib",
                "host_rss_peak_gib",
                "ceiling_mib",
                "ceiling_breached",
                "n_samples",
            ]
        )
        for m in measurements:
            w.writerow(
                [
                    m["op"],
                    m["wall_clock_s"],
                    m["gpu_baseline_mib"],
                    m["gpu_peak_mib"],
                    m["gpu_peak_gib"],
                    m["gpu_delta_mib"],
                    m["host_rss_peak_mib"],
                    m["host_rss_peak_gib"],
                    m["ceiling_mib"],
                    m["ceiling_breached"],
                    m["n_samples"],
                ]
            )
    for op, samples in all_samples.items():
        with open(
            os.path.join(args.results_dir, f"gpu_memory_samples_{op}.csv"),
            "w",
            newline="",
        ) as f:
            w = _csv.writer(f)
            w.writerow(["t_rel_s", "gpu_used_mib", "host_rss_mib"])
            for row in samples:
                w.writerow([round(row[0], 4), round(row[1], 1), round(row[2], 1)])
    with open(os.path.join(args.results_dir, "full_dataset_estimate.json"), "w") as f:
        json.dump(summary["full_dataset_estimate"], f, indent=2)

    # ---- manifests ----
    umap_par = dict(umap_params, build_algo_auto=umap_build_algo, random_state=None)
    seed_note = (
        "UMAP run unseeded (random_state=None, the fixed config default). cuML docs: a "
        "random_state slows training and raises memory; and build_algo='auto' selects "
        "non-deterministic NN-Descent for N>50000. Outputs are therefore run-specific."
    )
    write_manifest(
        coords_pq,
        inputs_block,
        script_path,
        umap_par,
        git_commit,
        tabular=(r, c),
        out_format="parquet",
        notes="GPU cuML UMAP 5-D coordinates keyed to doc_key. " + seed_note,
        seed=None,
        schema_diff={
            "added": [
                {"name": f"umap_{j}", "type": "float32"}
                for j in range(args.umap_n_components)
            ]
            + [
                {"name": "row", "type": "int32"},
                {"name": "doc_key", "type": "string"},
                {"name": "record_type", "type": "string"},
            ],
            "removed": [],
            "renamed": [],
            "retyped": [],
        },
    )
    write_manifest(
        coords_npy,
        inputs_block,
        script_path,
        umap_par,
        git_commit,
        out_format="other",
        notes="Raw 5-D UMAP coordinate matrix (.npy, float32). " + seed_note,
        seed=None,
    )
    write_manifest(
        labels_pq,
        inputs_block,
        script_path,
        dict(hdbscan_params, gen_min_span_tree=True, umap=umap_params),
        git_commit,
        tabular=(lr, lc),
        out_format="parquet",
        notes="Raw HDBSCAN labels (incl. -1) + membership probability, keyed to doc_key.",
        seed=None,
        schema_diff={
            "added": [
                {"name": "row", "type": "int32"},
                {"name": "doc_key", "type": "string"},
                {"name": "record_type", "type": "string"},
                {"name": "label", "type": "int64"},
                {"name": "probability", "type": "double"},
            ],
            "removed": [],
            "renamed": [],
            "retyped": [],
        },
    )
    for path, dim in ((ct_path, ct_dim), (slt_path, slt_dim), (mst_path, mst_dim)):
        if path:
            write_manifest(
                path,
                inputs_block,
                script_path,
                dict(hdbscan_params, gen_min_span_tree=True),
                git_commit,
                tabular=dim,
                out_format="parquet",
                notes="HDBSCAN tree diagnostic exported via cuML .to_pandas().",
                seed=None,
                schema_diff={"added": [], "removed": [], "renamed": [], "retyped": []},
            )

    # ---- figures ----
    make_figures(
        args.figures_dir, summary, sizes, probabilities, persistence, measurements
    )

    log("=== RESULT SUMMARY ===")
    log(
        f"UMAP ok, wall {umap_metrics['wall_clock_s']}s, peak {umap_metrics['gpu_peak_gib']:.3f} GiB"
    )
    log(
        f"HDBSCAN ok, wall {hdb_metrics['wall_clock_s']}s, peak {hdb_metrics['gpu_peak_gib']:.3f} GiB"
    )
    log(
        f"clusters={n_clusters} assigned={assigned_count} ({summary['hdbscan']['assigned_pct']}%) "
        f"noise={noise_count} ({summary['hdbscan']['noise_pct']}%)"
    )
    log(f"cluster sizes: {size_summary}")
    log(f"summary -> {summary_path}")
    pynvml.nvmlShutdown()


def _finalize_failure(
    args,
    summary_path,
    failed_op,
    metrics,
    measurements,
    gpu_name,
    driver,
    total_mib,
    umap_params,
    hdbscan_params,
    in_dups,
    N,
    git_commit,
):
    log(
        f"!!! {failed_op} FAILED or breached ceiling: {metrics.get('error')} "
        f"(peak {metrics['gpu_peak_gib']:.3f} GiB, breach={metrics['ceiling_breached']})"
    )
    summary = {
        "task": "three-day GPU UMAP + HDBSCAN feasibility",
        "timestamp": utcnow(),
        "git_commit": git_commit,
        "status": "FAILED",
        "failed_operation": failed_op,
        "n_rows_processed": N,
        "gpu": {"name": gpu_name, "driver": driver, "total_mib": round(total_mib, 1)},
        "config": {
            "umap": umap_params,
            "hdbscan": hdbscan_params,
            "ceiling_mib": args.ceiling_mib,
            "seed": None,
        },
        "measurements": measurements,
        "repeated_tied_embeddings": {"input": in_dups},
        "note": (
            "Stopped safely without changing analytical settings, per task. "
            "Reported measured failure/ceiling breach."
        ),
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"failure summary -> {summary_path}")


if __name__ == "__main__":
    main()
