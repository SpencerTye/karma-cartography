#!/usr/bin/env python3
"""GPU UMAP + HDBSCAN baseline on the full 902,144-document 0.6B embedding matrix (Phase 3).

Same settings as the three-day run: UMAP to 5-D (n_neighbors=15, min_dist=0.0, cosine), HDBSCAN on
that (min_cluster_size=10, min_samples=10, eom), plus a separate 2-D UMAP used only for figures.
Outputs: coordinates, labels, HDBSCAN diagnostics and trees keyed to doc_key, runtime and peak
GPU/host memory per step, manifests and verification. Stops cleanly if a step fails or exceeds
--ceiling-mib (default 21504). UMAP is unseeded, so outputs are run-specific.
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
# Fixed analysis settings (do not change)
# ----------------------------------------------------------------------------
UMAP_DEFAULTS = dict(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine")
VIZ_UMAP_DEFAULTS = dict(n_neighbors=15, n_components=2, min_dist=0.0, metric="cosine")
HDBSCAN_DEFAULTS = dict(
    min_cluster_size=10,
    min_samples=10,
    metric="euclidean",
    cluster_selection_method="eom",
)
EXPECTED_ROWS = 902144
EXPECTED_DIM = 1024

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
    """Background thread: samples whole-card GPU memory (NVML) and this process's
    host RSS at a fixed interval. Records peaks, a time series, a ceiling-breach
    flag, and whether this PID is reported as a compute process (often unavailable
    under WSL2/WDDM -- absence is not proof of no GPU use)."""

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
    """Run fn() with resource sampling; return (result, metrics_dict, samples)."""
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


def stage_failed(metrics) -> bool:
    """A stage must stop the pipeline if it raised OR its measured whole-card peak
    exceeded the ceiling (task: stop safely, preserve evidence, do not proceed)."""
    return (not metrics["ok"]) or bool(metrics["ceiling_breached"])


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
def software_block():
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
    """Write <output_path>.manifest.yml. `tabular` = (rows, cols) or None for a
    lite manifest (npy / tree object / image)."""
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
    """Exact identical-row census via np.unique(axis=0). Directly observed
    structure only; no meaning is inferred."""
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
# Verification checklist
# ----------------------------------------------------------------------------
def build_verification(
    *,
    quick,
    n,
    input_rows,
    emb5d,
    coords2d,
    labels,
    doc_keys,
    input_all_finite,
    row_order_ok,
    emb_hash_before,
    emb_hash_after,
    idx_hash_before,
    idx_hash_after,
):
    """Return the completion-checklist verification dict. In quick (smoke) mode the
    '== 902,144' checks are marked not-applicable but every branch still executes."""
    n_expected = None if quick else EXPECTED_ROWS
    uniq_keys = np.unique(doc_keys)

    def rows_ok(x):
        return x is not None and int(x) == int(n)

    checks = {
        "quick_mode": bool(quick),
        "expected_full_corpus_rows": EXPECTED_ROWS,
        "n_rows_processed": int(n),
        # rows == 902,144 (n/a in quick mode)
        "input_rows_equals_902144": (
            "n/a (quick mode)" if quick else bool(input_rows == n_expected)
        ),
        "umap5d_rows_equals_902144": (
            "n/a (quick mode)"
            if quick
            else bool(emb5d is not None and emb5d.shape[0] == n_expected)
        ),
        "umap2d_rows_equals_902144": (
            "n/a (quick mode)"
            if quick
            else bool(coords2d is not None and coords2d.shape[0] == n_expected)
        ),
        "hdbscan_rows_equals_902144": (
            "n/a (quick mode)"
            if quick
            else bool(labels is not None and labels.shape[0] == n_expected)
        ),
        # internal consistency (always checked, against the working set N)
        "input_rows_equal_working_set": bool(input_rows == n),
        "umap5d_rows_equal_working_set": bool(
            emb5d is not None and emb5d.shape[0] == n
        ),
        "umap2d_rows_equal_working_set": bool(
            coords2d is not None and coords2d.shape[0] == n
        ),
        "hdbscan_rows_equal_working_set": bool(
            labels is not None and labels.shape[0] == n
        ),
        "umap5d_output_dim_is_5": bool(emb5d is not None and emb5d.shape[1] == 5),
        "umap2d_output_dim_is_2": bool(coords2d is not None and coords2d.shape[1] == 2),
        # alignment / keys
        "input_row_order_is_0_to_n": bool(row_order_ok),
        "n_doc_keys_equals_rows": bool(len(doc_keys) == n),
        "doc_keys_unique_no_dupes_introduced": bool(uniq_keys.shape[0] == n),
        "n_unique_doc_keys": int(uniq_keys.shape[0]),
        # finiteness
        "input_all_finite": bool(input_all_finite),
        "umap5d_all_finite": bool(emb5d is not None and np.isfinite(emb5d).all()),
        "umap2d_all_finite": bool(coords2d is not None and np.isfinite(coords2d).all()),
        # exactly one label per document
        "every_doc_has_exactly_one_label": bool(
            labels is not None and labels.shape == (n,)
        ),
        # inputs unchanged (read-only): hash before == hash after this run
        "embeddings_unchanged": bool(emb_hash_before == emb_hash_after),
        "index_unchanged": bool(idx_hash_before == idx_hash_after),
        "embeddings_sha256": emb_hash_after,
        "index_sha256": idx_hash_after,
    }
    boolean_checks = [
        v
        for k, v in checks.items()
        if isinstance(v, bool) and not k.startswith("quick")
    ]
    checks["all_boolean_checks_pass"] = bool(all(boolean_checks))
    return checks


# ----------------------------------------------------------------------------
# Figures (diagnostics only; title + axis/tick labels, no captions/annotations)
# ----------------------------------------------------------------------------
def make_figures(
    fig_dir,
    summary,
    cluster_sizes,
    probabilities,
    persistence,
    measurements,
    coords2d,
    labels,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(fig_dir, exist_ok=True)
    n_written = 0

    # 1. cluster-size distribution (histogram of per-cluster sizes, log-y)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if cluster_sizes.size:
        ax.hist(cluster_sizes, bins=min(80, max(10, cluster_sizes.size)))
    ax.set_yscale("log")
    ax.set_xlabel("Cluster size (members)")
    ax.set_ylabel("Number of clusters")
    ax.set_title("HDBSCAN cluster-size distribution")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig_cluster_size_distribution.png"), dpi=150)
    plt.close(fig)
    n_written += 1

    # 2. ranked cluster sizes (sorted descending; log-log)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if cluster_sizes.size:
        ranked = np.sort(cluster_sizes)[::-1]
        ax.plot(np.arange(1, ranked.size + 1), ranked, linewidth=1.0)
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlabel("Cluster rank (largest to smallest)")
    ax.set_ylabel("Cluster size (members)")
    ax.set_title("Ranked cluster sizes")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig_ranked_cluster_sizes.png"), dpi=150)
    plt.close(fig)
    n_written += 1

    # 3. assigned vs noise (label -1) counts
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
    n_written += 1

    # 4. membership-probability distribution
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if probabilities is not None and probabilities.size:
        ax.hist(probabilities, bins=50, range=(0.0, 1.0))
    ax.set_yscale("log")
    ax.set_xlabel("HDBSCAN membership probability")
    ax.set_ylabel("Number of records")
    ax.set_title("Membership-probability distribution")
    fig.tight_layout()
    fig.savefig(
        os.path.join(fig_dir, "fig_membership_probability_distribution.png"), dpi=150
    )
    plt.close(fig)
    n_written += 1

    # 5. cluster-persistence distribution
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
    n_written += 1

    # 6. runtime by operation
    fig, ax = plt.subplots(figsize=(7, 4.5))
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
    n_written += 1

    # 7. peak whole-card GPU memory by operation
    fig, ax = plt.subplots(figsize=(7, 4.5))
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
    n_written += 1

    # 8. global 2-D UMAP density map (all documents; hexbin, log count) --------
    if coords2d is not None and coords2d.shape[0] > 0:
        fig, ax = plt.subplots(figsize=(7.5, 6.5))
        hb = ax.hexbin(
            coords2d[:, 0],
            coords2d[:, 1],
            gridsize=400,
            bins="log",
            mincnt=1,
            cmap="viridis",
            linewidths=0,
        )
        cb = fig.colorbar(hb, ax=ax)
        cb.set_label("Number of documents (log scale)")
        ax.set_xlabel("UMAP-1 (2-D visualisation)")
        ax.set_ylabel("UMAP-2 (2-D visualisation)")
        ax.set_title("Global 2-D UMAP density (all documents)")
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "fig_umap2d_global_density.png"), dpi=150)
        plt.close(fig)
        n_written += 1

    # 9. global 2-D UMAP: assigned vs noise (-1) (all points; rasterized) ------
    if coords2d is not None and labels is not None and coords2d.shape[0] > 0:
        noise_mask = labels == -1
        assigned_mask = ~noise_mask
        fig, ax = plt.subplots(figsize=(7.5, 6.5))
        # draw the larger group first so the smaller stays visible on top
        groups = [
            (noise_mask, "Noise (-1)", "#bdbdbd"),
            (assigned_mask, "Assigned (label >= 0)", "#1f77b4"),
        ]
        if int(assigned_mask.sum()) > int(noise_mask.sum()):
            groups = groups[::-1]
        for mask, lab, color in groups:
            if int(mask.sum()) == 0:
                continue
            ax.scatter(
                coords2d[mask, 0],
                coords2d[mask, 1],
                s=0.2,
                c=color,
                alpha=0.15,
                linewidths=0,
                marker=".",
                rasterized=True,
                label=lab,
            )
        leg = ax.legend(markerscale=20, loc="best")
        for lh in leg.legend_handles:
            lh.set_alpha(1.0)
        ax.set_xlabel("UMAP-1 (2-D visualisation)")
        ax.set_ylabel("UMAP-2 (2-D visualisation)")
        ax.set_title("Global 2-D UMAP: assigned vs noise")
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "fig_umap2d_assigned_vs_noise.png"), dpi=150)
        plt.close(fig)
        n_written += 1

    log(f"figures: wrote {n_written} diagnostic PNGs to {fig_dir}")


# ----------------------------------------------------------------------------
# Keyed output writers (parquet + npy) with manifests
# ----------------------------------------------------------------------------
def save_coords(
    results_dir, name, coords, doc_keys, record_type, rows_col, n_components
):
    pq_path = os.path.join(results_dir, f"{name}.parquet")
    cols = {
        "row": rows_col,
        "doc_key": doc_keys.astype(str),
        "record_type": record_type.astype(str),
    }
    for j in range(n_components):
        cols[f"umap_{j}"] = coords[:, j]
    r, c = write_parquet(pq_path, cols)
    npy_path = os.path.join(results_dir, f"{name}.npy")
    np.save(npy_path, coords)
    return pq_path, npy_path, (r, c)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fc = f"{REPO_ROOT}/pipeline/phase2_semantic_embeddings/full_corpus_embedding_0p6b/results"
    ap.add_argument("--embeddings", default=f"{fc}/embeddings.npy")
    ap.add_argument("--index", default=f"{fc}/embedding_index.parquet")
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
        "--reuse",
        action="store_true",
        help="if a stage's output .npy already exists in results/, load it and skip "
        "recomputation (recovery/resume; the committed canonical run does NOT use this)",
    )
    ap.add_argument(
        "--figures-only", action="store_true", help="rebuild figures from saved results"
    )
    # Fixed analysis settings (not tuned).
    ap.add_argument(
        "--umap-n-neighbors", type=int, default=UMAP_DEFAULTS["n_neighbors"]
    )
    ap.add_argument(
        "--umap-n-components", type=int, default=UMAP_DEFAULTS["n_components"]
    )
    ap.add_argument("--umap-min-dist", type=float, default=UMAP_DEFAULTS["min_dist"])
    ap.add_argument("--umap-metric", default=UMAP_DEFAULTS["metric"])
    ap.add_argument(
        "--viz-n-components", type=int, default=VIZ_UMAP_DEFAULTS["n_components"]
    )
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
        _figures_only(args, summary_path)
        return

    n_rows = args.n_rows if args.n_rows > 0 else (3000 if args.quick else 0)
    quick = n_rows > 0

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
    from cuml.cluster import HDBSCAN
    from cuml.manifold import UMAP

    log(
        f"cuml {cuml.__version__} | cupy {cupy.__version__} | "
        f"cuda runtime {cupy.cuda.runtime.runtimeGetVersion()} "
        f"driver {cupy.cuda.runtime.driverGetVersion()}"
    )

    # ---- load inputs (read-only) + hash before ----
    log("hashing inputs (read-only) ...")
    emb_hash_before = sha256_file(args.embeddings)
    idx_hash_before = sha256_file(args.index)
    log(f"loading embeddings: {args.embeddings}")
    t = time.perf_counter()
    X = np.load(args.embeddings, mmap_mode=None)
    log(f"embeddings loaded {X.shape} {X.dtype} in {time.perf_counter() - t:.1f}s")
    assert X.dtype == np.float32, f"expected float32, got {X.dtype}"
    if not quick:
        assert X.shape == (EXPECTED_ROWS, EXPECTED_DIM), f"unexpected shape {X.shape}"
    X = np.ascontiguousarray(X)

    import pyarrow.parquet as pq

    idx_tbl = pq.read_table(args.index)
    idx_rows = idx_tbl.column("row").to_numpy()
    doc_keys = np.array(idx_tbl.column("doc_key").to_pylist(), dtype=object)
    record_type = np.array(idx_tbl.column("record_type").to_pylist(), dtype=object)
    assert len(idx_rows) == X.shape[0] or quick, "index/embeddings row mismatch"
    input_rows = int(len(idx_tbl))
    order_ok = bool(
        np.array_equal(
            idx_rows[: X.shape[0]] if not quick else idx_rows[:n_rows],
            np.arange(min(len(idx_rows), X.shape[0] if not quick else n_rows)),
        )
    )

    if quick:
        X = X[:n_rows]
        doc_keys = doc_keys[:n_rows]
        record_type = record_type[:n_rows]
        idx_rows = idx_rows[:n_rows]
        input_rows = n_rows
    N = X.shape[0]
    rows_col = np.arange(N, dtype=np.int32)
    all_finite = bool(np.isfinite(X).all())
    log(
        f"working set: N={N} dim={X.shape[1]} finite={all_finite} row_order_ok={order_ok} quick={quick}"
    )

    # ---- input duplicate/tie observation (directly measured) ----
    log("computing input duplicate/tie statistics (exact, np.unique axis=0) ...")
    t = time.perf_counter()
    in_dups = duplicate_stats(X)
    log(f"input duplicates: {in_dups} ({time.perf_counter() - t:.1f}s)")

    inputs_block = [
        {
            "path": relpath(args.embeddings),
            "hash": emb_hash_before,
            "format": "numpy",
            "role": "full-corpus 0.6B embeddings (float32, L2-normalized), read-only",
            "rows": int(X.shape[0]) if quick else EXPECTED_ROWS,
            "cols": int(X.shape[1]),
        },
        {
            "path": relpath(args.index),
            "hash": idx_hash_before,
            "format": "parquet",
            "role": "row -> doc_key index for the embeddings, read-only",
            "rows": len(idx_tbl),
            "cols": idx_tbl.num_columns,
        },
    ]
    git_commit = git_short_head()

    measurements = []
    all_samples = {}
    gpu_block = {"name": gpu_name, "driver": driver, "total_mib": round(total_mib, 1)}
    cuda_block = {
        "cupy": cupy.__version__,
        "runtime_version": cupy.cuda.runtime.runtimeGetVersion(),
        "driver_version": cupy.cuda.runtime.driverGetVersion(),
    }

    def reuse_npy(name):
        p = os.path.join(args.results_dir, f"{name}.npy")
        if args.reuse and os.path.exists(p):
            arr = np.load(p)
            log(f"--reuse: loaded {name} {arr.shape} from {p} (skipping compute)")
            return np.ascontiguousarray(np.asarray(arr, dtype=np.float32))
        return None

    def finalize_stop(reason, failed_op, partial):
        _finalize_failure(
            args,
            summary_path,
            failed_op,
            reason,
            measurements,
            all_samples,
            gpu_block,
            cuda_block,
            umap_params,
            hdbscan_params,
            viz_params,
            in_dups,
            N,
            input_rows,
            all_finite,
            order_ok,
            emb_hash_before,
            idx_hash_before,
            args.embeddings,
            args.index,
            quick,
            partial,
        )

    # ------------------------------------------------------------------ 5-D UMAP
    umap_params = dict(
        n_neighbors=args.umap_n_neighbors,
        n_components=args.umap_n_components,
        min_dist=args.umap_min_dist,
        metric=args.umap_metric,
    )
    hdbscan_params = dict(
        min_cluster_size=args.hdbscan_min_cluster_size,
        min_samples=args.hdbscan_min_samples,
        metric=args.hdbscan_metric,
        cluster_selection_method=args.hdbscan_cluster_selection_method,
    )
    viz_params = dict(VIZ_UMAP_DEFAULTS, n_components=args.viz_n_components)
    partial = {}

    log(f"UMAP 5-D params (fixed): {umap_params}")
    emb = reuse_npy("umap_5d_coordinates")
    if emb is None:
        reducer = UMAP(**umap_params)

        def _run_umap():
            return reducer.fit_transform(X)

        emb, umap_metrics, umap_samples = measure(
            "umap_5d",
            _run_umap,
            handle,
            pynvml,
            proc,
            args.nvml_interval_ms,
            args.ceiling_mib,
        )
        measurements.append(umap_metrics)
        all_samples["umap_5d"] = umap_samples
        if not umap_metrics["ok"]:
            finalize_stop(
                f"umap_5d raised: {umap_metrics['error']}", "umap_5d", partial
            )
            return
        emb = np.ascontiguousarray(np.asarray(emb, dtype=np.float32))
        assert emb.shape == (N, args.umap_n_components), f"UMAP 5-D shape {emb.shape}"
        # save immediately (evidence preserved even if a later stage stops)
        c5_pq, c5_npy, (r5, c5) = save_coords(
            args.results_dir,
            "umap_5d_coordinates",
            emb,
            doc_keys,
            record_type,
            rows_col,
            args.umap_n_components,
        )
        partial["umap_5d_coordinates"] = (c5_pq, c5_npy, (r5, c5))
        umap_build_algo = (
            "brute_force_knn (auto, N<=50000)"
            if N <= 50000
            else "nn_descent (auto, N>50000)"
        )
        log(
            f"UMAP 5-D output {emb.shape}; auto build_algo -> {umap_build_algo}; saved."
        )
        reducer = None
        free_gpu(cupy)
        if umap_metrics["ceiling_breached"]:
            finalize_stop(
                f"umap_5d completed but whole-card peak {umap_metrics['gpu_peak_gib']:.3f} GiB "
                f"exceeded the {args.ceiling_mib / 1024:.3f} GiB ceiling",
                "umap_5d",
                partial,
            )
            return
    else:
        umap_metrics = {"op": "umap_5d", "reused_from_cache": True}
        umap_build_algo = (
            "brute_force_knn (auto, N<=50000)"
            if N <= 50000
            else "nn_descent (auto, N>50000)"
        )
        c5_pq, c5_npy, (r5, c5) = save_coords(
            args.results_dir,
            "umap_5d_coordinates",
            emb,
            doc_keys,
            record_type,
            rows_col,
            args.umap_n_components,
        )
        partial["umap_5d_coordinates"] = (c5_pq, c5_npy, (r5, c5))

    out_dups_5d = duplicate_stats(emb)
    log(f"UMAP 5-D output duplicates/ties: {out_dups_5d}")

    # --------------------------------------------------------------- HDBSCAN
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
        finalize_stop(f"hdbscan raised: {hdb_metrics['error']}", "hdbscan", partial)
        return

    labels = np.asarray(clusterer.labels_)
    probabilities = np.asarray(clusterer.probabilities_, dtype=np.float64)
    persistence = np.asarray(
        getattr(clusterer, "cluster_persistence_", np.array([])), dtype=np.float64
    )
    assert labels.shape == (N,), f"labels shape {labels.shape}"
    assert probabilities.shape == (N,), f"probabilities shape {probabilities.shape}"

    # ---- cluster statistics (raw labels preserved, incl. -1); scale-safe via unique
    noise_mask = labels == -1
    assigned_count = int((~noise_mask).sum())
    noise_count = int(noise_mask.sum())
    uniq_labels, uniq_counts = np.unique(labels[~noise_mask], return_counts=True)
    n_clusters = int(uniq_labels.size)
    sizes = uniq_counts.astype(np.int64)

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

    # ---- save keyed labels + probabilities (immediately) ----
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
    partial["hdbscan_labels"] = (labels_pq, (lr, lc))

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

    def _save_tree(attr, fname):
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

    ct_path, ct_dim = _save_tree("condensed_tree_", "hdbscan_condensed_tree.parquet")
    slt_path, slt_dim = _save_tree(
        "single_linkage_tree_", "hdbscan_single_linkage_tree.parquet"
    )
    mst_path, mst_dim = _save_tree(
        "minimum_spanning_tree_", "hdbscan_minimum_spanning_tree.parquet"
    )
    clusterer = None
    free_gpu(cupy)

    if hdb_metrics["ceiling_breached"]:
        finalize_stop(
            f"hdbscan completed but whole-card peak {hdb_metrics['gpu_peak_gib']:.3f} GiB "
            f"exceeded the {args.ceiling_mib / 1024:.3f} GiB ceiling",
            "hdbscan",
            partial,
        )
        return

    # --------------------------------------------------------- 2-D viz UMAP
    log(f"UMAP 2-D (visualisation-only) params (fixed): {viz_params}")
    coords2d = reuse_npy("umap_2d_visualization_coordinates")
    if coords2d is None:
        viz_reducer = UMAP(**viz_params)

        def _run_viz():
            return viz_reducer.fit_transform(X)

        coords2d, viz_metrics, viz_samples = measure(
            "umap_2d_viz",
            _run_viz,
            handle,
            pynvml,
            proc,
            args.nvml_interval_ms,
            args.ceiling_mib,
        )
        measurements.append(viz_metrics)
        all_samples["umap_2d_viz"] = viz_samples
        if not viz_metrics["ok"]:
            # analytical outputs (5-D + HDBSCAN) already saved; record the viz failure
            log(
                f"!!! 2-D viz UMAP raised: {viz_metrics['error']} (analytical outputs are saved)"
            )
            coords2d = None
        else:
            coords2d = np.ascontiguousarray(np.asarray(coords2d, dtype=np.float32))
            assert coords2d.shape == (N, args.viz_n_components), (
                f"2-D shape {coords2d.shape}"
            )
            c2_pq, c2_npy, (r2, c2) = save_coords(
                args.results_dir,
                "umap_2d_visualization_coordinates",
                coords2d,
                doc_keys,
                record_type,
                rows_col,
                args.viz_n_components,
            )
            partial["umap_2d_visualization_coordinates"] = (c2_pq, c2_npy, (r2, c2))
            viz_reducer = None
            free_gpu(cupy)
            log(f"UMAP 2-D output {coords2d.shape}; saved.")
    else:
        viz_metrics = {"op": "umap_2d_viz", "reused_from_cache": True}
        c2_pq, c2_npy, (r2, c2) = save_coords(
            args.results_dir,
            "umap_2d_visualization_coordinates",
            coords2d,
            doc_keys,
            record_type,
            rows_col,
            args.viz_n_components,
        )
        partial["umap_2d_visualization_coordinates"] = (c2_pq, c2_npy, (r2, c2))

    out_dups_2d = duplicate_stats(coords2d) if coords2d is not None else None
    if out_dups_2d is not None:
        log(f"UMAP 2-D output duplicates/ties: {out_dups_2d}")

    # ---- run status (5-D UMAP + HDBSCAN already passed the ceiling/OK guards;
    # only the viz-only 2-D UMAP can still be incomplete or breaching here) ----
    viz_ok = coords2d is not None and viz_metrics.get("ok", True)
    viz_breach = bool(viz_metrics.get("ceiling_breached", False))
    if viz_ok and not viz_breach:
        run_status = "OK"
    else:
        run_status = "OK_ANALYTICAL__VIZ_INCOMPLETE_OR_BREACHED"
        log(
            f"NOTE: analytical outputs (5-D UMAP + HDBSCAN) are complete and saved, but the "
            f"visualisation-only 2-D UMAP did not finish within budget "
            f"(computed={coords2d is not None}, viz_breach={viz_breach}). Reported, not hidden."
        )

    # ---- re-hash inputs (confirm unchanged by this run) ----
    emb_hash_after = sha256_file(args.embeddings)
    idx_hash_after = sha256_file(args.index)

    # ---- verification ----
    verification = build_verification(
        quick=quick,
        n=N,
        input_rows=input_rows,
        emb5d=emb,
        coords2d=coords2d,
        labels=labels,
        doc_keys=doc_keys,
        input_all_finite=all_finite,
        row_order_ok=order_ok,
        emb_hash_before=emb_hash_before,
        emb_hash_after=emb_hash_after,
        idx_hash_before=idx_hash_before,
        idx_hash_after=idx_hash_after,
    )

    # ---- assemble summary ----
    summary = {
        "task": "full-corpus GPU UMAP + HDBSCAN baseline (902,144 documents)",
        "timestamp": utcnow(),
        "git_commit": git_commit,
        "status": run_status,
        "n_rows_processed": N,
        "quick_mode": quick,
        "reuse_mode": bool(args.reuse),
        "row_order_verified": order_ok,
        "input_all_finite": all_finite,
        "gpu": gpu_block,
        "software": software_block(),
        "cuda": cuda_block,
        "config": {
            "umap_5d": umap_params,
            "umap_5d_build_algo_auto_selected": umap_build_algo,
            "umap_2d_viz": viz_params,
            "umap_random_state": None,
            "hdbscan": hdbscan_params,
            "hdbscan_gen_min_span_tree": True,
            "ceiling_mib": args.ceiling_mib,
            "seed": None,
        },
        "measurements": measurements,
        "umap_5d": {
            "output_shape": list(emb.shape),
            "input_dim": int(X.shape[1]),
            "output_dim": int(emb.shape[1]),
            "all_rows_processed": bool(emb.shape[0] == N),
            "coordinates_parquet": relpath(partial["umap_5d_coordinates"][0]),
            "coordinates_npy": relpath(partial["umap_5d_coordinates"][1]),
        },
        "umap_2d_viz": {
            "computed": coords2d is not None,
            "output_shape": list(coords2d.shape) if coords2d is not None else None,
            "coordinates_parquet": (
                relpath(partial["umap_2d_visualization_coordinates"][0])
                if coords2d is not None
                else None
            ),
            "coordinates_npy": (
                relpath(partial["umap_2d_visualization_coordinates"][1])
                if coords2d is not None
                else None
            ),
            "note": "Visualisation only; NOT used as HDBSCAN input and does not change cluster assignments.",
            "duplicates": out_dups_2d,
        },
        "hdbscan": {
            "n_clusters": n_clusters,
            "assigned_count": assigned_count,
            "assigned_pct": round(100.0 * assigned_count / N, 4),
            "noise_count": noise_count,
            "noise_pct": round(100.0 * noise_count / N, 4),
            "all_rows_have_label": bool(labels.shape[0] == N),
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
                "all_exactly_one": bool(
                    persistence.size > 0 and np.all(persistence == 1.0)
                ),
            },
            "trees": trees,
            "labels_parquet": relpath(labels_pq),
        },
        "repeated_tied_embeddings": {
            "input": in_dups,
            "umap_5d_output": out_dups_5d,
            "umap_2d_output": out_dups_2d,
            "largest_cluster_size": size_summary["max"],
            "note": "Directly observed structural/computational quantities only; no interpretation.",
            "umap_5d_warnings": umap_metrics.get("warnings", []),
            "hdbscan_warnings": hdb_metrics.get("warnings", []),
            "umap_2d_warnings": viz_metrics.get("warnings", []),
        },
        "verification": verification,
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.results_dir, "verification.json"), "w") as f:
        json.dump(verification, f, indent=2)

    _write_measurements_and_samples(args, measurements, all_samples)

    # ---- manifests ----
    _write_primary_manifests(
        args,
        inputs_block,
        script_path,
        git_commit,
        umap_params,
        viz_params,
        hdbscan_params,
        partial,
        ct_path,
        ct_dim,
        slt_path,
        slt_dim,
        mst_path,
        mst_dim,
        umap_build_algo,
    )

    # ---- figures ----
    make_figures(
        args.figures_dir,
        summary,
        sizes,
        probabilities,
        persistence,
        measurements,
        coords2d,
        labels,
    )

    log("=== RESULT SUMMARY ===")
    for m in measurements:
        if m.get("reused_from_cache"):
            log(f"{m['op']}: reused from cache")
        else:
            log(
                f"{m['op']} ok, wall {m['wall_clock_s']}s, peak {m['gpu_peak_gib']:.3f} GiB"
            )
    log(
        f"clusters={n_clusters} assigned={assigned_count} ({summary['hdbscan']['assigned_pct']}%) "
        f"noise={noise_count} ({summary['hdbscan']['noise_pct']}%)"
    )
    log(f"cluster sizes: {size_summary}")
    log(
        f"persistence all 1.0: {summary['hdbscan']['cluster_persistence_summary']['all_exactly_one']}"
    )
    log(f"verification all_pass: {verification['all_boolean_checks_pass']}")
    log(f"summary -> {summary_path}")
    pynvml.nvmlShutdown()


def _write_measurements_and_samples(args, measurements, all_samples):
    import csv as _csv

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
            if m.get("reused_from_cache"):
                continue
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


def _write_primary_manifests(
    args,
    inputs_block,
    script_path,
    git_commit,
    umap_params,
    viz_params,
    hdbscan_params,
    partial,
    ct_path,
    ct_dim,
    slt_path,
    slt_dim,
    mst_path,
    mst_dim,
    umap_build_algo,
):
    seed_note = (
        "UMAP run unseeded (random_state=None, the fixed config default). cuML docs: a "
        "random_state slows training and raises memory; and build_algo='auto' selects "
        "non-deterministic NN-Descent for N>50000. Outputs are therefore run-specific."
    )
    umap_par = dict(umap_params, build_algo_auto=umap_build_algo, random_state=None)
    coord_diff = lambda ncomp: {  # noqa: E731
        "added": [{"name": f"umap_{j}", "type": "float32"} for j in range(ncomp)]
        + [
            {"name": "row", "type": "int32"},
            {"name": "doc_key", "type": "string"},
            {"name": "record_type", "type": "string"},
        ],
        "removed": [],
        "renamed": [],
        "retyped": [],
    }
    if "umap_5d_coordinates" in partial:
        c5_pq, c5_npy, (r5, c5) = partial["umap_5d_coordinates"]
        write_manifest(
            c5_pq,
            inputs_block,
            script_path,
            umap_par,
            git_commit,
            tabular=(r5, c5),
            out_format="parquet",
            notes="GPU cuML UMAP 5-D analytical coordinates keyed to doc_key. "
            + seed_note,
            seed=None,
            schema_diff=coord_diff(umap_params["n_components"]),
        )
        write_manifest(
            c5_npy,
            inputs_block,
            script_path,
            umap_par,
            git_commit,
            out_format="other",
            notes="Raw 5-D UMAP coordinate matrix (.npy, float32). " + seed_note,
            seed=None,
        )
    if "umap_2d_visualization_coordinates" in partial:
        c2_pq, c2_npy, (r2, c2) = partial["umap_2d_visualization_coordinates"]
        viz_par = dict(viz_params, build_algo_auto=umap_build_algo, random_state=None)
        vnote = (
            "GPU cuML UMAP 2-D VISUALISATION coordinates keyed to doc_key; NOT used as "
            "HDBSCAN input and does not affect cluster assignments. " + seed_note
        )
        write_manifest(
            c2_pq,
            inputs_block,
            script_path,
            viz_par,
            git_commit,
            tabular=(r2, c2),
            out_format="parquet",
            notes=vnote,
            seed=None,
            schema_diff=coord_diff(viz_params["n_components"]),
        )
        write_manifest(
            c2_npy,
            inputs_block,
            script_path,
            viz_par,
            git_commit,
            out_format="other",
            notes="Raw 2-D UMAP visualisation coordinate matrix (.npy, float32). "
            + seed_note,
            seed=None,
        )
    if "hdbscan_labels" in partial:
        labels_pq, (lr, lc) = partial["hdbscan_labels"]
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


def _finalize_failure(
    args,
    summary_path,
    failed_op,
    reason,
    measurements,
    all_samples,
    gpu_block,
    cuda_block,
    umap_params,
    hdbscan_params,
    viz_params,
    in_dups,
    N,
    input_rows,
    all_finite,
    order_ok,
    emb_hash_before,
    idx_hash_before,
    emb_path,
    idx_path,
    quick,
    partial,
):
    log(f"!!! STOP after {failed_op}: {reason}")
    emb_hash_after = sha256_file(emb_path)
    idx_hash_after = sha256_file(idx_path)
    summary = {
        "task": "full-corpus GPU UMAP + HDBSCAN baseline (902,144 documents)",
        "timestamp": utcnow(),
        "status": "STOPPED_FAILED_OR_BREACHED",
        "failed_operation": failed_op,
        "stop_reason": reason,
        "n_rows_processed": N,
        "quick_mode": quick,
        "gpu": gpu_block,
        "cuda": cuda_block,
        "config": {
            "umap_5d": umap_params,
            "umap_2d_viz": viz_params,
            "hdbscan": hdbscan_params,
            "ceiling_mib": args.ceiling_mib,
            "seed": None,
        },
        "measurements": measurements,
        "repeated_tied_embeddings": {"input": in_dups},
        "artifacts_saved_before_stop": sorted(partial.keys()),
        "inputs_unchanged": {
            "embeddings": bool(emb_hash_before == emb_hash_after),
            "index": bool(idx_hash_before == idx_hash_after),
        },
        "note": (
            "Stopped safely without changing any analytical setting or substituting "
            "another method/dataset, per task. Evidence for the completed part is "
            "preserved in results/. Reported the measured failure / ceiling breach."
        ),
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.results_dir, "verification.json"), "w") as f:
        json.dump(
            {"status": "STOPPED", "failed_operation": failed_op, "reason": reason},
            f,
            indent=2,
        )
    _write_measurements_and_samples(args, measurements, all_samples)
    log(f"failure summary -> {summary_path}")


def _figures_only(args, summary_path):
    with open(summary_path) as f:
        summary = json.load(f)
    cs = (
        np.loadtxt(os.path.join(args.results_dir, "_cluster_sizes.txt"))
        if os.path.exists(os.path.join(args.results_dir, "_cluster_sizes.txt"))
        else np.array([])
    )
    import pyarrow.parquet as pq

    labels_tbl = pq.read_table(os.path.join(args.results_dir, "hdbscan_labels.parquet"))
    probabilities = labels_tbl.column("probability").to_numpy()
    labels = labels_tbl.column("label").to_numpy()
    pers = np.array([])
    pcsv = os.path.join(args.results_dir, "hdbscan_cluster_persistence.csv")
    if os.path.exists(pcsv):
        import csv

        with open(pcsv) as f:
            pers = np.array(
                [
                    float(r["persistence"])
                    for r in csv.DictReader(f)
                    if r["persistence"] != ""
                ]
            )
    coords2d = None
    c2 = os.path.join(args.results_dir, "umap_2d_visualization_coordinates.npy")
    if os.path.exists(c2):
        coords2d = np.load(c2)
    make_figures(
        args.figures_dir,
        summary,
        np.atleast_1d(cs),
        probabilities,
        pers,
        summary["measurements"],
        coords2d,
        labels,
    )
    log("figures-only: done")


if __name__ == "__main__":
    main()
