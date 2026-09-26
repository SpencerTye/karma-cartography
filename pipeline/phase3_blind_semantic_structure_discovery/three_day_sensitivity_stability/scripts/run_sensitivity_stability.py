#!/usr/bin/env python3
"""Sensitivity and stability of UMAP + HDBSCAN on the three-day embeddings (101,722 x 1024).

One factor at a time, unseeded: B0..B4 repeat the baseline, U_* change one UMAP setting,
H_* change one HDBSCAN setting on B0's 5-D coordinates. B0 is the stability reference; label -1
stays unassigned. Baseline: UMAP n_neighbors=15 n_components=5 min_dist=0.0 cosine;
HDBSCAN min_cluster_size=10 min_samples=10 euclidean eom.
Outputs: results/, results/umap/ and figures/ (no manifests or report).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import threading
import time
import warnings
from datetime import datetime, timezone
from itertools import combinations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import scipy.sparse as sp
from sklearn.metrics import adjusted_rand_score

import cuml
import cupy
from cuml.cluster import HDBSCAN
from cuml.manifold import UMAP
import pynvml
import psutil

cuml.set_global_output_type("numpy")

REPO_ROOT = os.environ.get("KC_REPO_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..")))
BASE = f"{REPO_ROOT}/pipeline/phase3_blind_semantic_structure_discovery/three_day_sensitivity_stability"
EMB_DEFAULT = f"{REPO_ROOT}/pipeline/phase2_semantic_embeddings/three_day_embedding_0p6b/results/embeddings.npy"
IDX_DEFAULT = f"{REPO_ROOT}/pipeline/phase2_semantic_embeddings/three_day_embedding_0p6b/results/embedding_index.parquet"

EXPECTED_ROWS = 101722
EXPECTED_DIM = 1024

BASE_UMAP = dict(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine")
BASE_HDB = dict(
    min_cluster_size=10,
    min_samples=10,
    metric="euclidean",
    cluster_selection_method="eom",
)


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    print(f"[{utcnow()}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Run specifications (B0 first: its 5-D UMAP coords feed every H_* run)
# ---------------------------------------------------------------------------
def _spec(run_id, phase, changed, umap=None, hdb=None, umap_source="own"):
    return dict(
        run_id=run_id,
        phase=phase,
        changed=changed,
        umap=dict(BASE_UMAP, **(umap or {})),
        hdb=dict(BASE_HDB, **(hdb or {})),
        umap_source=umap_source,
    )


def build_specs():
    specs = [_spec("B0", "baseline", "(reference)")]
    for i in range(1, 5):
        specs.append(_spec(f"B{i}", "baseline", "(baseline repeat)"))
    specs += [
        _spec("U_nn5", "umap", "UMAP n_neighbors=5", umap={"n_neighbors": 5}),
        _spec("U_nn30", "umap", "UMAP n_neighbors=30", umap={"n_neighbors": 30}),
        _spec("U_nn50", "umap", "UMAP n_neighbors=50", umap={"n_neighbors": 50}),
        _spec("U_dim3", "umap", "UMAP n_components=3", umap={"n_components": 3}),
        _spec("U_dim10", "umap", "UMAP n_components=10", umap={"n_components": 10}),
        _spec("U_md01", "umap", "UMAP min_dist=0.1", umap={"min_dist": 0.1}),
        _spec("U_md05", "umap", "UMAP min_dist=0.5", umap={"min_dist": 0.5}),
        _spec(
            "H_mcs20",
            "hdbscan",
            "HDBSCAN min_cluster_size=20",
            hdb={"min_cluster_size": 20},
            umap_source="B0",
        ),
        _spec(
            "H_mcs50",
            "hdbscan",
            "HDBSCAN min_cluster_size=50",
            hdb={"min_cluster_size": 50},
            umap_source="B0",
        ),
        _spec(
            "H_mcs100",
            "hdbscan",
            "HDBSCAN min_cluster_size=100",
            hdb={"min_cluster_size": 100},
            umap_source="B0",
        ),
        _spec(
            "H_ms5",
            "hdbscan",
            "HDBSCAN min_samples=5",
            hdb={"min_samples": 5},
            umap_source="B0",
        ),
        _spec(
            "H_ms20",
            "hdbscan",
            "HDBSCAN min_samples=20",
            hdb={"min_samples": 20},
            umap_source="B0",
        ),
        _spec(
            "H_ms50",
            "hdbscan",
            "HDBSCAN min_samples=50",
            hdb={"min_samples": 50},
            umap_source="B0",
        ),
        _spec(
            "H_leaf",
            "hdbscan",
            "HDBSCAN cluster_selection_method=leaf",
            hdb={"cluster_selection_method": "leaf"},
            umap_source="B0",
        ),
    ]
    return specs


# ---------------------------------------------------------------------------
# Whole-card GPU memory sampler (passive; matches the existing Phase 3 impl)
# ---------------------------------------------------------------------------
class Sampler:
    def __init__(self, handle, proc, interval_ms, ceiling_mib):
        self.handle = handle
        self.proc = proc
        self.interval = max(interval_ms, 5) / 1000.0
        self.ceiling_mib = ceiling_mib
        self._stop = threading.Event()
        self._thread = None
        self.gpu_peak_mib = 0.0
        self.host_rss_peak_mib = 0.0
        self.ceiling_breached = False

    def _loop(self):
        while not self._stop.is_set():
            try:
                used = pynvml.nvmlDeviceGetMemoryInfo(self.handle).used / 1048576.0
            except Exception:
                used = float("nan")
            rss = self.proc.memory_info().rss / 1048576.0
            if used == used:
                if used > self.gpu_peak_mib:
                    self.gpu_peak_mib = used
                if used > self.ceiling_mib:
                    self.ceiling_breached = True
            if rss > self.host_rss_peak_mib:
                self.host_rss_peak_mib = rss
            self._stop.wait(self.interval)

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)


def measure(fn, handle, proc, interval_ms, ceiling_mib):
    """Run fn() under resource sampling. Returns (result, metrics)."""
    s = Sampler(handle, proc, interval_ms, ceiling_mib)
    s.start()
    err = None
    result = None
    t0 = time.perf_counter()
    try:
        with warnings.catch_warnings(record=True) as wl:
            warnings.simplefilter("always")
            result = fn()
            warns = [f"{w.category.__name__}: {w.message}" for w in wl]
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        warns = []
    finally:
        dt = time.perf_counter() - t0
        s.stop()
    metrics = dict(
        ok=(err is None),
        error=err,
        runtime_s=round(dt, 4),
        gpu_peak_gib=round(s.gpu_peak_mib / 1024.0, 4),
        host_rss_peak_gib=round(s.host_rss_peak_mib / 1024.0, 4),
        ceiling_breached=s.ceiling_breached,
        warnings=warns,
    )
    return result, metrics


def free_gpu():
    gc.collect()
    try:
        cupy.get_default_memory_pool().free_all_blocks()
        cupy.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Cluster / stability analytics (raw labels; -1 kept as unassigned)
# ---------------------------------------------------------------------------
def _pct(a, p):
    return float(np.percentile(a, p)) if a.size else None


def cluster_summary(labels):
    labels = np.asarray(labels)
    assigned_mask = labels >= 0
    uniq, counts = np.unique(labels[assigned_mask], return_counts=True)
    sizes = counts.astype(np.int64)
    n = labels.size
    assigned = int(assigned_mask.sum())
    noise = int(n - assigned)
    return dict(
        n_clusters=int(uniq.size),
        assigned_count=assigned,
        assigned_pct=round(100.0 * assigned / n, 4),
        unassigned_count=noise,
        unassigned_pct=round(100.0 * noise / n, 4),
        size_min=int(sizes.min()) if sizes.size else None,
        size_q1=_pct(sizes, 25),
        size_median=_pct(sizes, 50),
        size_mean=float(sizes.mean()) if sizes.size else None,
        size_q3=_pct(sizes, 75),
        size_max=int(sizes.max()) if sizes.size else None,
    ), sizes


def best_match_jaccard(ref, cmp):
    """For each ref cluster (>=0), best Jaccard to any cmp cluster (>=0).
    Jaccard = |ref_c & cmp_d| / |ref_c | cmp_d|. -1 excluded on both sides."""
    ref = np.asarray(ref)
    cmp = np.asarray(cmp)
    ref_lab, ref_cnt = np.unique(ref[ref >= 0], return_counts=True)
    cmp_lab, cmp_cnt = np.unique(cmp[cmp >= 0], return_counts=True)
    R, C = ref_lab.size, cmp_lab.size
    best = np.zeros(R, dtype=np.float64)
    best_cmp = np.full(R, -1, dtype=np.int64)
    if R == 0 or C == 0:
        return ref_lab, ref_cnt, best, best_cmp
    mask = (ref >= 0) & (cmp >= 0)
    ri = np.searchsorted(ref_lab, ref[mask])
    ci = np.searchsorted(cmp_lab, cmp[mask])
    cont = sp.coo_matrix(
        (np.ones(ri.size, dtype=np.int64), (ri, ci)), shape=(R, C)
    ).tocsr()
    for r in range(R):
        row = cont.getrow(r)
        if row.nnz == 0:
            continue
        cols = row.indices
        inter = row.data.astype(np.float64)
        union = ref_cnt[r] + cmp_cnt[cols] - inter
        j = inter / union
        k = int(np.argmax(j))
        best[r] = float(j[k])
        best_cmp[r] = int(cmp_lab[cols[k]])
    return ref_lab, ref_cnt, best, best_cmp


def jaccard_summary(best):
    return dict(
        median=_pct(best, 50),
        q1=_pct(best, 25),
        q3=_pct(best, 75),
        pct_ge_050=(
            round(100.0 * float((best >= 0.50).mean()), 4) if best.size else None
        ),
        pct_ge_080=(
            round(100.0 * float((best >= 0.80).mean()), 4) if best.size else None
        ),
        n_ref_clusters=int(best.size),
    )


def status_agreement(a, b):
    return float(((np.asarray(a) == -1) == (np.asarray(b) == -1)).mean())


def both_assigned_ari(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    m = (a >= 0) & (b >= 0)
    if int(m.sum()) < 2:
        return float("nan")
    return float(adjusted_rand_score(a[m], b[m]))


def transitions(ref, cmp):
    ref = np.asarray(ref)
    cmp = np.asarray(cmp)
    ra = ref >= 0
    ca = cmp >= 0
    return dict(
        assigned_to_assigned=int((ra & ca).sum()),
        assigned_to_unassigned=int((ra & ~ca).sum()),
        unassigned_to_assigned=int((~ra & ca).sum()),
        unassigned_to_unassigned=int((~ra & ~ca).sum()),
    )


# ---------------------------------------------------------------------------
# CSV writer (stdlib csv via list-of-dicts)
# ---------------------------------------------------------------------------
def write_csv(path, fieldnames, rows):
    import csv

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


# ---------------------------------------------------------------------------
# Figures (title + axis/tick labels + legend/colorbar/gridlines only)
# ---------------------------------------------------------------------------
def fig1_baseline_heatmap(path, order, ari_mat):
    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    offdiag = ari_mat[~np.eye(len(order), dtype=bool)]
    vmin = float(np.floor(offdiag.min() * 100) / 100) if offdiag.size else 0.0
    im = ax.imshow(ari_mat, cmap="viridis", vmin=vmin, vmax=1.0)
    ax.set_xticks(range(len(order)))
    ax.set_yticks(range(len(order)))
    ax.set_xticklabels(order)
    ax.set_yticklabels(order)
    ax.set_xlabel("Run")
    ax.set_ylabel("Run")
    ax.set_title("Baseline-repeat all-document ARI (B0-B4)")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("Adjusted Rand Index")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig2_sensitivity_stability(path, runs, all_ari, both_ari, agree):
    x = np.arange(len(runs))
    w = 0.27
    fig, ax = plt.subplots(figsize=(max(9.0, 0.55 * len(runs) + 3), 5.0))
    ax.bar(x - w, all_ari, w, label="all-document ARI")
    ax.bar(x, both_ari, w, label="both-assigned ARI")
    ax.bar(x + w, agree, w, label="assigned/unassigned status agreement")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(runs, rotation=60, ha="right")
    ax.set_xlabel("Run (vs B0)")
    ax.set_ylabel("Value (0-1)")
    ax.set_title("Stability vs B0 across non-B0 runs")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig3_count_assigned(path, runs, n_clusters, assigned_pct, ref_idx):
    x = np.arange(len(runs))
    colors = ["#d62728" if i == ref_idx else "#1f77b4" for i in range(len(runs))]
    fig, axes = plt.subplots(
        2, 1, figsize=(max(9.0, 0.55 * len(runs) + 3), 7.4), sharex=True
    )
    axes[0].bar(x, n_clusters, color=colors)
    axes[0].set_ylabel("Number of clusters")
    axes[0].set_title("Cluster count by run")
    axes[0].grid(axis="y", linestyle=":", alpha=0.5)
    axes[1].bar(x, assigned_pct, color=colors)
    axes[1].set_ylabel("Assigned documents (%)")
    axes[1].set_ylim(0.0, 100.0)
    axes[1].set_title("Assigned percentage by run")
    axes[1].grid(axis="y", linestyle=":", alpha=0.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(runs, rotation=60, ha="right")
    axes[1].set_xlabel("Run")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig4_reference_overlap(path, runs, median_j, pct50, pct80):
    x = np.arange(len(runs))
    fig, axes = plt.subplots(
        2, 1, figsize=(max(9.0, 0.55 * len(runs) + 3), 7.4), sharex=True
    )
    axes[0].bar(x, median_j, color="#2ca02c")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("Median best-match Jaccard")
    axes[0].set_title("B0-cluster median best-match Jaccard by run")
    axes[0].grid(axis="y", linestyle=":", alpha=0.5)
    w = 0.4
    axes[1].bar(x - w / 2, pct50, w, label="B0 clusters with best Jaccard >= 0.50")
    axes[1].bar(x + w / 2, pct80, w, label="B0 clusters with best Jaccard >= 0.80")
    axes[1].set_ylim(0.0, 100.0)
    axes[1].set_ylabel("Percentage of B0 clusters (%)")
    axes[1].set_title("B0-cluster best-match Jaccard thresholds by run")
    axes[1].grid(axis="y", linestyle=":", alpha=0.5)
    axes[1].legend(loc="upper right")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(runs, rotation=60, ha="right")
    axes[1].set_xlabel("Run (vs B0)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig5_cluster_size_dist(path, runs, sizes_by_run):
    data = [s if s.size else np.array([np.nan]) for s in sizes_by_run]
    fig, ax = plt.subplots(figsize=(max(9.0, 0.55 * len(runs) + 3), 5.4))
    ax.boxplot(data, showfliers=True, whis=(0, 100))
    ax.set_yscale("log")
    ax.set_xticks(range(1, len(runs) + 1))
    ax.set_xticklabels(runs, rotation=60, ha="right")
    ax.set_xlabel("Run")
    ax.set_ylabel("Cluster size (members, log scale)")
    ax.set_title("Cluster-size distribution by run")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--embeddings", default=EMB_DEFAULT)
    ap.add_argument("--index", default=IDX_DEFAULT)
    ap.add_argument("--results-dir", default=f"{BASE}/results")
    ap.add_argument("--figures-dir", default=f"{BASE}/figures")
    ap.add_argument("--umap-dir", default=f"{BASE}/results/umap")
    ap.add_argument("--ceiling-mib", type=float, default=21504.0)
    ap.add_argument("--nvml-interval-ms", type=int, default=50)
    ap.add_argument("--n-rows", type=int, default=0, help="0=all; >0 subset (smoke)")
    args = ap.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.figures_dir, exist_ok=True)
    os.makedirs(args.umap_dir, exist_ok=True)

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    gpu_name = pynvml.nvmlDeviceGetName(handle)
    gpu_name = gpu_name.decode() if isinstance(gpu_name, bytes) else gpu_name
    driver = pynvml.nvmlSystemGetDriverVersion()
    driver = driver.decode() if isinstance(driver, bytes) else driver
    total_mib = pynvml.nvmlDeviceGetMemoryInfo(handle).total / 1048576.0
    proc = psutil.Process(os.getpid())
    log(
        f"GPU {gpu_name} | driver {driver} | total {total_mib:.0f} MiB | ceiling {args.ceiling_mib:.0f} MiB"
    )
    log(
        f"cuml {cuml.__version__} | cupy {cupy.__version__} | "
        f"cuda rt {cupy.cuda.runtime.runtimeGetVersion()} drv {cupy.cuda.runtime.driverGetVersion()}"
    )

    # ---- inputs (read-only) ----
    X = np.load(args.embeddings, mmap_mode=None)
    assert X.dtype == np.float32, f"expected float32, got {X.dtype}"
    tbl = pq.read_table(args.index)
    idx_rows = tbl.column("row").to_numpy()
    doc_keys = np.array(tbl.column("doc_key").to_pylist(), dtype=object)
    if args.n_rows > 0:
        X = X[: args.n_rows]
        doc_keys = doc_keys[: args.n_rows]
        idx_rows = idx_rows[: args.n_rows]
    else:
        assert X.shape == (EXPECTED_ROWS, EXPECTED_DIM), f"unexpected shape {X.shape}"
    X = np.ascontiguousarray(X)
    N = X.shape[0]
    row_order_ok = bool(np.array_equal(idx_rows, np.arange(N)))
    all_finite = bool(np.isfinite(X).all())
    assert len(doc_keys) == N, "index/embeddings row mismatch"
    log(
        f"working set N={N} dim={X.shape[1]} finite={all_finite} row_order_ok={row_order_ok}"
    )

    specs = build_specs()
    order = [s["run_id"] for s in specs]

    # ---- execute all runs ----
    results = {}  # run_id -> dict(labels, probs, summary, sizes, umap_rt, hdb_rt, ...)
    b0_coords = None
    t_compute0 = time.perf_counter()
    for s in specs:
        rid = s["run_id"]
        if s["umap_source"] == "own":
            up = s["umap"]
            reducer = UMAP(**up)
            emb, um = measure(
                lambda: reducer.fit_transform(X),
                handle,
                proc,
                args.nvml_interval_ms,
                args.ceiling_mib,
            )
            if not um["ok"]:
                log(f"!!! {rid} UMAP FAILED: {um['error']}")
                results[rid] = dict(spec=s, umap=um, hdb=None, failed="umap")
                reducer = None
                free_gpu()
                continue
            emb = np.ascontiguousarray(np.asarray(emb, dtype=np.float32))
            build_algo = (
                "nn_descent (auto, N>50000)"
                if N > 50000
                else "brute_force (auto, N<=50000)"
            )
            np.save(os.path.join(args.umap_dir, f"umap_{rid}.npy"), emb)
            if rid == "B0":
                b0_coords = emb
            coords = emb
            reducer = None
            free_gpu()
            log(
                f"{rid} UMAP ok {emb.shape} {um['runtime_s']}s peak {um['gpu_peak_gib']:.3f} GiB algo={build_algo}"
            )
        else:  # reuse B0 coordinates
            assert b0_coords is not None, "B0 coords missing before H_* run"
            coords = b0_coords
            um = None
            build_algo = "n/a (reuses B0 5-D UMAP coordinates)"

        hp = s["hdb"]
        clusterer = HDBSCAN(gen_min_span_tree=False, **hp)

        def _fit():
            clusterer.fit(coords)
            return clusterer

        _, hm = measure(_fit, handle, proc, args.nvml_interval_ms, args.ceiling_mib)
        if not hm["ok"]:
            log(f"!!! {rid} HDBSCAN FAILED: {hm['error']}")
            results[rid] = dict(
                spec=s, umap=um, hdb=hm, failed="hdbscan", build_algo=build_algo
            )
            clusterer = None
            free_gpu()
            continue
        labels = np.asarray(clusterer.labels_).astype(np.int64)
        probs = np.asarray(clusterer.probabilities_, dtype=np.float64)
        assert labels.shape == (N,) and probs.shape == (N,)
        summ, sizes = cluster_summary(labels)
        results[rid] = dict(
            spec=s,
            umap=um,
            hdb=hm,
            build_algo=build_algo,
            labels=labels,
            probs=probs,
            summary=summ,
            sizes=sizes,
            failed=None,
        )
        clusterer = None
        free_gpu()
        log(
            f"{rid} HDBSCAN ok {hm['runtime_s']}s peak {hm['gpu_peak_gib']:.3f} GiB "
            f"clusters={summ['n_clusters']} assigned%={summ['assigned_pct']}"
        )
    compute_wall_s = round(time.perf_counter() - t_compute0, 2)

    ok_runs = [r for r in order if results[r].get("failed") is None]
    assert "B0" in ok_runs, "B0 must succeed to serve as reference"
    b0 = results["B0"]["labels"]
    non_b0 = [r for r in ok_runs if r != "B0"]
    baseline_ok = [r for r in ["B0", "B1", "B2", "B3", "B4"] if r in ok_runs]

    # ---- stability vs B0 for every non-B0 run ----
    stab = {}
    best_jacc_arrays = {}
    for r in non_b0:
        lab = results[r]["labels"]
        all_ari = float(adjusted_rand_score(b0, lab))
        b_ari = both_assigned_ari(b0, lab)
        agree = status_agreement(b0, lab)
        trans = transitions(b0, lab)
        ref_lab, ref_cnt, best, best_cmp = best_match_jaccard(b0, lab)
        best_jacc_arrays[r] = (ref_lab, ref_cnt, best, best_cmp)
        jsum = jaccard_summary(best)
        stab[r] = dict(
            all_ari=all_ari, both_ari=b_ari, agree=agree, trans=trans, jsum=jsum
        )

    # ---- pairwise baseline B0-B4 ----
    pair_rows = []
    for a, bb in combinations(baseline_ok, 2):
        la, lb = results[a]["labels"], results[bb]["labels"]
        all_ari = float(adjusted_rand_score(la, lb))
        b_ari = both_assigned_ari(la, lb)
        agree = status_agreement(la, lb)
        _, _, best, _ = best_match_jaccard(la, lb)  # ref = lower-index run
        pair_rows.append(
            dict(
                pair=f"{a}-{bb}",
                ref_run=a,
                cmp_run=bb,
                all_doc_ari=round(all_ari, 6),
                both_assigned_ari=round(b_ari, 6),
                status_agreement=round(agree, 6),
                median_best_jaccard=round(_pct(best, 50), 6) if best.size else "",
            )
        )

    def _range(vals):
        v = np.array([x for x in vals if x == x], dtype=float)
        if not v.size:
            return dict(min=None, median=None, max=None)
        return dict(min=float(v.min()), median=float(np.median(v)), max=float(v.max()))

    baseline_ranges = {
        "all_doc_ari": _range([p["all_doc_ari"] for p in pair_rows]),
        "both_assigned_ari": _range([p["both_assigned_ari"] for p in pair_rows]),
        "status_agreement": _range([p["status_agreement"] for p in pair_rows]),
        "median_best_jaccard": _range(
            [
                p["median_best_jaccard"]
                for p in pair_rows
                if p["median_best_jaccard"] != ""
            ]
        ),
    }

    # ---- 5x5 all-document ARI matrix for the heatmap ----
    ari_mat = np.eye(len(baseline_ok))
    for i, a in enumerate(baseline_ok):
        for j, bb in enumerate(baseline_ok):
            if j > i:
                v = float(
                    adjusted_rand_score(results[a]["labels"], results[bb]["labels"])
                )
                ari_mat[i, j] = ari_mat[j, i] = v

    # =====================================================================
    # Save derived data
    # =====================================================================
    # 1. long-form assignment table
    n_runs = len(ok_runs)
    doc_col = np.tile(doc_keys.astype(str), n_runs)
    run_col = np.repeat(np.array(ok_runs, dtype=object).astype(str), N)
    lab_col = np.concatenate([results[r]["labels"] for r in ok_runs]).astype(np.int64)
    prob_col = np.concatenate([results[r]["probs"] for r in ok_runs]).astype(np.float64)
    long_tbl = pa.table(
        {
            "doc_key": pa.array(doc_col, type=pa.string()),
            "run_id": pa.array(run_col, type=pa.string()),
            "cluster_label": pa.array(lab_col, type=pa.int64()),
            "membership_probability": pa.array(prob_col, type=pa.float64()),
        }
    )
    long_path = os.path.join(args.results_dir, "assignments_long.parquet")
    pq.write_table(long_tbl, long_path, compression="zstd")
    log(f"assignments_long.parquet {long_tbl.num_rows} rows -> {long_path}")

    # 2. run_configurations_and_summary.csv
    cfg_fields = [
        "run_id",
        "phase",
        "changed_param",
        "umap_source",
        "umap_n_neighbors",
        "umap_n_components",
        "umap_min_dist",
        "umap_metric",
        "umap_build_algo",
        "hdbscan_min_cluster_size",
        "hdbscan_min_samples",
        "hdbscan_metric",
        "hdbscan_cluster_selection_method",
        "umap_runtime_s",
        "hdbscan_runtime_s",
        "gpu_peak_gib_umap",
        "gpu_peak_gib_hdbscan",
        "ceiling_breached",
        "n_clusters",
        "assigned_count",
        "assigned_pct",
        "unassigned_count",
        "unassigned_pct",
        "size_min",
        "size_q1",
        "size_median",
        "size_mean",
        "size_q3",
        "size_max",
    ]
    cfg_rows = []
    for r in order:
        R = results[r]
        s = R["spec"]
        summ = R.get("summary", {})
        um = R.get("umap")
        hm = R.get("hdb")
        cfg_rows.append(
            dict(
                run_id=r,
                phase=s["phase"],
                changed_param=s["changed"],
                umap_source=s["umap_source"],
                umap_n_neighbors=s["umap"]["n_neighbors"],
                umap_n_components=s["umap"]["n_components"],
                umap_min_dist=s["umap"]["min_dist"],
                umap_metric=s["umap"]["metric"],
                umap_build_algo=R.get("build_algo", ""),
                hdbscan_min_cluster_size=s["hdb"]["min_cluster_size"],
                hdbscan_min_samples=s["hdb"]["min_samples"],
                hdbscan_metric=s["hdb"]["metric"],
                hdbscan_cluster_selection_method=s["hdb"]["cluster_selection_method"],
                umap_runtime_s=(um["runtime_s"] if um else ""),
                hdbscan_runtime_s=(hm["runtime_s"] if hm else ""),
                gpu_peak_gib_umap=(um["gpu_peak_gib"] if um else ""),
                gpu_peak_gib_hdbscan=(hm["gpu_peak_gib"] if hm else ""),
                ceiling_breached=bool(
                    (um and um["ceiling_breached"]) or (hm and hm["ceiling_breached"])
                ),
                **{
                    k: summ.get(k, "")
                    for k in [
                        "n_clusters",
                        "assigned_count",
                        "assigned_pct",
                        "unassigned_count",
                        "unassigned_pct",
                        "size_min",
                        "size_q1",
                        "size_median",
                        "size_mean",
                        "size_q3",
                        "size_max",
                    ]
                },
            )
        )
    write_csv(
        os.path.join(args.results_dir, "run_configurations_and_summary.csv"),
        cfg_fields,
        cfg_rows,
    )

    # 3. baseline pairwise + ranges
    write_csv(
        os.path.join(args.results_dir, "baseline_pairwise_repeatability.csv"),
        [
            "pair",
            "ref_run",
            "cmp_run",
            "all_doc_ari",
            "both_assigned_ari",
            "status_agreement",
            "median_best_jaccard",
        ],
        pair_rows,
    )
    range_rows = [
        dict(metric=k, minimum=v["min"], median=v["median"], maximum=v["max"])
        for k, v in baseline_ranges.items()
    ]
    write_csv(
        os.path.join(args.results_dir, "baseline_repeatability_ranges.csv"),
        ["metric", "minimum", "median", "maximum"],
        range_rows,
    )

    # 4. b0_vs_run_stability.csv
    stab_fields = [
        "run_id",
        "phase",
        "changed_param",
        "all_doc_ari",
        "both_assigned_ari",
        "status_agreement",
        "median_best_jaccard",
        "q1_best_jaccard",
        "q3_best_jaccard",
        "pct_ge_050",
        "pct_ge_080",
        "n_b0_clusters",
    ]
    stab_rows = []
    for r in non_b0:
        s = results[r]["spec"]
        st = stab[r]
        stab_rows.append(
            dict(
                run_id=r,
                phase=s["phase"],
                changed_param=s["changed"],
                all_doc_ari=round(st["all_ari"], 6),
                both_assigned_ari=round(st["both_ari"], 6),
                status_agreement=round(st["agree"], 6),
                median_best_jaccard=st["jsum"]["median"],
                q1_best_jaccard=st["jsum"]["q1"],
                q3_best_jaccard=st["jsum"]["q3"],
                pct_ge_050=st["jsum"]["pct_ge_050"],
                pct_ge_080=st["jsum"]["pct_ge_080"],
                n_b0_clusters=st["jsum"]["n_ref_clusters"],
            )
        )
    write_csv(
        os.path.join(args.results_dir, "b0_vs_run_stability.csv"),
        stab_fields,
        stab_rows,
    )

    # 5. assigned_unassigned_transitions.csv
    tr_fields = [
        "run_id",
        "assigned_to_assigned",
        "assigned_to_unassigned",
        "unassigned_to_assigned",
        "unassigned_to_unassigned",
        "total",
    ]
    tr_rows = []
    for r in non_b0:
        t = stab[r]["trans"]
        tr_rows.append(dict(run_id=r, **t, total=sum(t.values())))
    write_csv(
        os.path.join(args.results_dir, "assigned_unassigned_transitions.csv"),
        tr_fields,
        tr_rows,
    )

    # 6. b0_cluster_best_match_jaccard.csv (per non-B0 run, per B0 cluster)
    bj_fields = [
        "run_id",
        "b0_cluster_label",
        "b0_cluster_size",
        "best_match_cmp_label",
        "best_jaccard",
    ]
    bj_rows = []
    for r in non_b0:
        ref_lab, ref_cnt, best, best_cmp = best_jacc_arrays[r]
        for i in range(ref_lab.size):
            bj_rows.append(
                dict(
                    run_id=r,
                    b0_cluster_label=int(ref_lab[i]),
                    b0_cluster_size=int(ref_cnt[i]),
                    best_match_cmp_label=int(best_cmp[i]),
                    best_jaccard=round(float(best[i]), 6),
                )
            )
    write_csv(
        os.path.join(args.results_dir, "b0_cluster_best_match_jaccard.csv"),
        bj_fields,
        bj_rows,
    )

    # 7. cluster_size_summaries.csv
    cs_fields = [
        "run_id",
        "n_clusters",
        "size_min",
        "size_q1",
        "size_median",
        "size_mean",
        "size_q3",
        "size_max",
    ]
    cs_rows = [
        dict(run_id=r, **{k: results[r]["summary"].get(k, "") for k in cs_fields[1:]})
        for r in ok_runs
    ]
    write_csv(
        os.path.join(args.results_dir, "cluster_size_summaries.csv"), cs_fields, cs_rows
    )

    # =====================================================================
    # Figures
    # =====================================================================
    fig1_baseline_heatmap(
        os.path.join(args.figures_dir, "fig1_baseline_ari_heatmap.png"),
        baseline_ok,
        ari_mat,
    )
    fig2_sensitivity_stability(
        os.path.join(args.figures_dir, "fig2_sensitivity_stability.png"),
        non_b0,
        [stab[r]["all_ari"] for r in non_b0],
        [stab[r]["both_ari"] for r in non_b0],
        [stab[r]["agree"] for r in non_b0],
    )
    fig3_count_assigned(
        os.path.join(args.figures_dir, "fig3_clustercount_assignedpct.png"),
        ok_runs,
        [results[r]["summary"]["n_clusters"] for r in ok_runs],
        [results[r]["summary"]["assigned_pct"] for r in ok_runs],
        ok_runs.index("B0"),
    )
    fig4_reference_overlap(
        os.path.join(args.figures_dir, "fig4_reference_cluster_overlap.png"),
        non_b0,
        [(stab[r]["jsum"]["median"] or 0.0) for r in non_b0],
        [(stab[r]["jsum"]["pct_ge_050"] or 0.0) for r in non_b0],
        [(stab[r]["jsum"]["pct_ge_080"] or 0.0) for r in non_b0],
    )
    fig5_cluster_size_dist(
        os.path.join(args.figures_dir, "fig5_cluster_size_distribution.png"),
        ok_runs,
        [results[r]["sizes"].astype(float) for r in ok_runs],
    )

    # =====================================================================
    # Headline summary (analytical results, machine-readable)
    # =====================================================================
    # largest / smallest deviation among tested-change runs (U_* and H_*)
    change_runs = [r for r in non_b0 if r.startswith(("U_", "H_"))]

    def extremes(metric_fn, lower_is_more_deviation=True):
        vals = [(r, metric_fn(r)) for r in change_runs if metric_fn(r) == metric_fn(r)]
        if not vals:
            return None
        largest = (min if lower_is_more_deviation else max)(vals, key=lambda kv: kv[1])
        smallest = (max if lower_is_more_deviation else min)(vals, key=lambda kv: kv[1])
        return dict(
            largest_deviation=dict(run=largest[0], value=round(largest[1], 6)),
            smallest_deviation=dict(run=smallest[0], value=round(smallest[1], 6)),
        )

    deviations = {
        "all_doc_ari": extremes(lambda r: stab[r]["all_ari"]),
        "both_assigned_ari": extremes(lambda r: stab[r]["both_ari"]),
        "status_agreement": extremes(lambda r: stab[r]["agree"]),
        "median_best_jaccard": extremes(lambda r: stab[r]["jsum"]["median"] or 0.0),
    }

    summary = dict(
        task="Phase 3 sensitivity/stability on three-day 101,722-doc embeddings",
        timestamp=utcnow(),
        n_rows=N,
        row_order_verified=row_order_ok,
        input_all_finite=all_finite,
        gpu=dict(name=gpu_name, driver=driver, total_mib=round(total_mib, 1)),
        environment=dict(
            python=platform.python_version(),
            cuml=cuml.__version__,
            cupy=cupy.__version__,
            os=platform.platform(),
        ),
        baseline_reference="B0",
        unseeded=True,
        ceiling_mib=args.ceiling_mib,
        compute_wall_s=compute_wall_s,
        total_umap_runtime_s=round(
            sum(
                results[r]["umap"]["runtime_s"]
                for r in ok_runs
                if results[r].get("umap")
            ),
            2,
        ),
        total_hdbscan_runtime_s=round(
            sum(
                results[r]["hdb"]["runtime_s"] for r in ok_runs if results[r].get("hdb")
            ),
            2,
        ),
        failed_runs=[r for r in order if results[r].get("failed")],
        warnings_by_run={
            r: (
                (results[r].get("umap") or {}).get("warnings", [])
                + (results[r].get("hdb") or {}).get("warnings", [])
            )
            for r in order
            if (
                (results[r].get("umap") or {}).get("warnings")
                or (results[r].get("hdb") or {}).get("warnings")
            )
        },
        ceiling_breaches=[
            r
            for r in order
            if (
                (results[r].get("umap") or {}).get("ceiling_breached")
                or (results[r].get("hdb") or {}).get("ceiling_breached")
            )
        ],
        baseline_pairwise_ranges=baseline_ranges,
        compact_table=[
            dict(
                run_id=r,
                changed_param=results[r]["spec"]["changed"],
                n_clusters=results[r]["summary"]["n_clusters"],
                assigned_pct=results[r]["summary"]["assigned_pct"],
                unassigned_pct=results[r]["summary"]["unassigned_pct"],
                all_doc_ari=(1.0 if r == "B0" else round(stab[r]["all_ari"], 6)),
                both_assigned_ari=(1.0 if r == "B0" else round(stab[r]["both_ari"], 6)),
                status_agreement=(1.0 if r == "B0" else round(stab[r]["agree"], 6)),
                median_best_jaccard=(1.0 if r == "B0" else stab[r]["jsum"]["median"]),
            )
            for r in ok_runs
        ],
        deviations_among_tested_changes=deviations,
        outputs=dict(
            results_dir=args.results_dir,
            figures_dir=args.figures_dir,
            umap_dir=args.umap_dir,
        ),
    )
    with open(os.path.join(args.results_dir, "sensitivity_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    log("=== DONE ===")
    log(
        f"compute wall {compute_wall_s}s | UMAP total {summary['total_umap_runtime_s']}s "
        f"| HDBSCAN total {summary['total_hdbscan_runtime_s']}s"
    )
    log(
        f"failed_runs={summary['failed_runs']} ceiling_breaches={summary['ceiling_breaches']}"
    )
    log(f"baseline all-doc ARI range: {baseline_ranges['all_doc_ari']}")
    pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
