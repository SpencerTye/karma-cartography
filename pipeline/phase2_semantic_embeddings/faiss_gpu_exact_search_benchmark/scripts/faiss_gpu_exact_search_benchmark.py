#!/usr/bin/env python3
"""
FAISS GPU exact nearest-neighbor search benchmark
==================================================

Exact, exhaustive nearest-neighbor search over the full-corpus
Qwen3-Embedding-0.6B embeddings (902,144 x 1024, float32, L2-normalized).

Index: FAISS ``GpuIndexFlatIP`` -- a flat (brute-force) inner-product index.
Because the stored vectors are L2-normalized, inner product == cosine
similarity, and the ranking corresponds to cosine similarity. This is the
mature FAISS exact-search primitive; there is NO IVF / HNSW / PQ / quantization
/ vector compression / clustering / approximation of any kind. Vectors are kept
at float32 exactly as stored on disk (the embeddings are neither rebuilt nor
altered).

What it measures (single process, index loaded once):
  1. index build/load time; whole-card GPU memory before/after; host RAM;
     confirmation that all 902,144 x 1024 float32 vectors are present + finite
     + L2-normalized.
  2. untimed warm-up, then a short query-batch-size sweep (a few sizes) to pick
     a practically efficient batch size (NOT a memory-limit / bisection search).
  3. a main timed test: 10,000 real corpus documents queried against the whole
     902,144-doc index, retrieving the 100 closest *other* documents (request
     k+1, drop the query's own row). Per-batch and aggregate timing + peak
     whole-card GPU memory.
  4. a 20 / 100 / 500 returned-neighbor-count comparison (throughput effect of k).
  5. an independent exactness validation: for a few real queries, recompute
     cosine similarity against the entire float32 matrix with a plain numpy
     fp32 matmul and compare identities, order, scores, self-match to FAISS.
  6. a full-corpus (902,144-query) runtime extrapolation from the measured rate.

Writes only small evidence files (JSON/CSV). It never writes an all-corpus
neighbor table.

Whole-card GPU memory is sampled via NVML (nvidia-ml-py), which on this WSL2
host reports total device usage including the Windows desktop apps; a one-shot
nvidia-smi reading is captured alongside as a cross-check. The hard whole-card
ceiling for this project is 21 GiB (21,504 MiB); this test stays far below it
and does not probe the boundary.
"""

import argparse
import csv
import json
import platform
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

MIB = 1024.0 * 1024.0
GIB = 1024.0 * 1024.0 * 1024.0
CEILING_MIB = 21504.0  # project hard whole-card ceiling (21 GiB), for reference only


# --------------------------------------------------------------------------- #
# GPU memory sampling (whole-card, via NVML with an nvidia-smi fallback)
# --------------------------------------------------------------------------- #
def make_gpu_reader():
    """Return (reader_callable, backend_name). reader() -> used bytes (int) or None."""
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)

        def reader():
            try:
                return int(pynvml.nvmlDeviceGetMemoryInfo(handle).used)
            except Exception:
                return None

        return reader, "nvml"
    except Exception:

        def reader():
            try:
                out = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                )
                return int(out.strip().splitlines()[0]) * 1024 * 1024
            except Exception:
                return None

        return reader, "nvidia-smi"


def nvidia_smi_used_mib():
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        used, total, drv = [x.strip() for x in out.strip().splitlines()[0].split(",")]
        return {"used_mib": float(used), "total_mib": float(total), "driver": drv}
    except Exception as exc:  # pragma: no cover
        return {"error": str(exc)}


class PeakSampler:
    """Background thread tracking peak whole-card GPU memory over a region."""

    def __init__(self, reader, interval_ms=20.0):
        self.reader = reader
        self.interval = interval_ms / 1000.0
        self._stop = threading.Event()
        self._thread = None
        self.peak_bytes = 0
        self.baseline_bytes = 0
        self.samples = 0

    def _run(self):
        while not self._stop.is_set():
            b = self.reader()
            if b is not None:
                self.samples += 1
                if b > self.peak_bytes:
                    self.peak_bytes = b
            time.sleep(self.interval)

    def start(self):
        b = self.reader() or 0
        self.baseline_bytes = b
        self.peak_bytes = b
        self.samples = 0
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        b = self.reader()
        if b is not None and b > self.peak_bytes:
            self.peak_bytes = b
        return self.peak_bytes

    @property
    def peak_mib(self):
        return self.peak_bytes / MIB


def host_rss_bytes():
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except Exception:
        try:
            with open("/proc/self/status") as fh:
                for line in fh:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) * 1024
        except Exception:
            return None
    return None


# --------------------------------------------------------------------------- #
# Data loading + verification
# --------------------------------------------------------------------------- #
def load_embeddings(path):
    arr = np.load(path, mmap_mode=None)
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    return arr


def verify_matrix(xb, expected_rows, expected_dim):
    n, d = xb.shape
    finite = bool(np.isfinite(xb).all())
    norms = np.linalg.norm(xb, axis=1)
    return {
        "shape": [int(n), int(d)],
        "dtype": str(xb.dtype),
        "n_rows": int(n),
        "dim": int(d),
        "rows_equals_expected": bool(n == expected_rows),
        "dim_equals_expected": bool(d == expected_dim),
        "expected_rows": int(expected_rows),
        "expected_dim": int(expected_dim),
        "c_contiguous": bool(xb.flags["C_CONTIGUOUS"]),
        "all_finite": finite,
        "nbytes": int(xb.nbytes),
        "l2_norm_min": float(norms.min()),
        "l2_norm_max": float(norms.max()),
        "l2_norm_mean": float(norms.mean()),
        "l2_normalized_within_1e-3": bool(np.allclose(norms, 1.0, atol=1e-3)),
    }


def read_doc_keys(parquet_path, n_rows):
    """Return doc_key array ordered by row, plus an alignment check dict."""
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path, columns=["row", "doc_key"])
    rows = table.column("row").to_numpy()
    keys = np.asarray(table.column("doc_key").to_pylist(), dtype=object)
    order = np.argsort(rows)
    rows_sorted = rows[order]
    keys_sorted = keys[order]
    aligned = bool(
        len(rows) == n_rows and np.array_equal(rows_sorted, np.arange(n_rows))
    )
    return keys_sorted, {
        "index_rows": int(len(rows)),
        "matches_embedding_rows": bool(len(rows) == n_rows),
        "row_column_is_0_to_n_minus_1": aligned,
    }


# --------------------------------------------------------------------------- #
# Index build + search
# --------------------------------------------------------------------------- #
def build_gpu_flat_ip(xb, reader, interval_ms):
    import faiss

    d = int(xb.shape[1])
    sampler = PeakSampler(reader, interval_ms).start()
    before_bytes = reader() or 0
    t0 = time.perf_counter()
    res = faiss.StandardGpuResources()
    index = faiss.GpuIndexFlatIP(res, d)
    index.add(xb)
    build_s = time.perf_counter() - t0
    after_bytes = reader() or 0
    peak_bytes = sampler.stop()
    info = {
        "build_seconds": build_s,
        "gpu_used_before_bytes": int(before_bytes),
        "gpu_used_before_mib": before_bytes / MIB,
        "gpu_used_after_bytes": int(after_bytes),
        "gpu_used_after_mib": after_bytes / MIB,
        "gpu_index_delta_mib": (after_bytes - before_bytes) / MIB,
        "gpu_peak_during_build_mib": peak_bytes / MIB,
        "ntotal": int(index.ntotal),
        "index_dim": int(index.d),
        "ntotal_equals_rows": bool(index.ntotal == xb.shape[0]),
    }
    return index, res, info


def search_all(index, xq, k, batch_size, reader, interval_ms, time_it=True):
    """Search xq in consecutive batches of batch_size. Returns (D, I, batches, region_peak_mib)."""
    n = xq.shape[0]
    D = np.empty((n, k), dtype=np.float32)
    nbr_ids = np.empty((n, k), dtype=np.int64)
    batches = []
    sampler = PeakSampler(reader, interval_ms).start() if time_it else None
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        chunk = np.ascontiguousarray(xq[start:end])
        t0 = time.perf_counter()
        d_b, i_b = index.search(chunk, k)
        dt = time.perf_counter() - t0
        D[start:end] = d_b
        nbr_ids[start:end] = i_b
        if time_it:
            nq = end - start
            batches.append(
                {
                    "batch_index": len(batches),
                    "n_queries": int(nq),
                    "elapsed_s": dt,
                    "queries_per_s": nq / dt if dt > 0 else float("nan"),
                    "peak_gpu_mib_so_far": sampler.peak_mib,
                }
            )
    region_peak = sampler.stop() / MIB if time_it else None
    return D, nbr_ids, batches, region_peak


def warmup(index, xq_pool, reader, interval_ms, n_batches, batch_size, ks):
    """Untimed warm-up: exercise the search path (including the largest k) before timing."""
    n = xq_pool.shape[0]
    done = 0
    for b in range(n_batches):
        start = (b * batch_size) % max(1, n - batch_size)
        chunk = np.ascontiguousarray(xq_pool[start : start + batch_size])
        for k in ks:
            index.search(chunk, min(k, n))
        done += chunk.shape[0]
    return {
        "n_warmup_batches": n_batches,
        "batch_size": batch_size,
        "ks": list(ks),
        "queries_touched": done,
    }


# --------------------------------------------------------------------------- #
# Self-match removal + neighbor keeping
# --------------------------------------------------------------------------- #
def drop_self_keep_k(nbr_ids, D, query_rows, k_keep):
    """From (k_keep+margin) retrieved ids, drop the query's own row; keep k_keep others.

    Returns kept ids/scores arrays and self-match stats.
    """
    nq = nbr_ids.shape[0]
    self_at_rank0 = 0
    self_found = 0
    self_scores = []
    kept_ids = np.full((nq, k_keep), -1, dtype=np.int64)
    kept_scores = np.full((nq, k_keep), np.nan, dtype=np.float32)
    for r in range(nq):
        own = int(query_rows[r])
        ids = nbr_ids[r]
        scores = D[r]
        if ids[0] == own:
            self_at_rank0 += 1
        mask = ids != own
        if (~mask).any():
            self_found += 1
            self_scores.append(float(scores[ids == own][0]))
        others_ids = ids[mask][:k_keep]
        others_scores = scores[mask][:k_keep]
        kept_ids[r, : others_ids.shape[0]] = others_ids
        kept_scores[r, : others_scores.shape[0]] = others_scores
    stats = {
        "n_queries": int(nq),
        "k_keep": int(k_keep),
        "self_found_in_results": int(self_found),
        "self_at_rank0": int(self_at_rank0),
        "self_at_rank0_fraction": self_at_rank0 / nq if nq else float("nan"),
        "self_score_min": float(np.min(self_scores)) if self_scores else None,
        "self_score_max": float(np.max(self_scores)) if self_scores else None,
        "self_score_mean": float(np.mean(self_scores)) if self_scores else None,
    }
    return kept_ids, kept_scores, stats


# --------------------------------------------------------------------------- #
# Exactness validation (independent numpy fp32 matmul)
# --------------------------------------------------------------------------- #
def exactness_check(index, xb, query_rows, exact_k, doc_keys, score_tol):
    """Compare FAISS GpuIndexFlatIP top-exact_k against an independent numpy fp32
    matmul, for each query.

    The rigorous, tie-robust exactness criterion is agreement of the *sorted
    similarity-score vectors*: an exact search must return the exact_k largest
    inner products, in descending order, whatever order it happens to break ties
    in. Identity/rank differences only signal a real error when the swapped
    documents have *different* scores; differences among equal (tied) scores are
    legitimate and expected here -- the corpus holds many exact-duplicate texts
    whose embeddings are identical and tie at cosine 1.0. So we (a) compare the
    sorted score vectors and (b) confirm every positional identity mismatch sits
    on an equal score.
    """
    xq = np.ascontiguousarray(xb[query_rows])
    faiss_D, faiss_I = index.search(xq, exact_k)
    # Independent exact similarities: full fp32 matmul against the whole matrix.
    sims = (xb @ xq.T).T  # [nq, N] fp32 BLAS -- a different implementation from FAISS
    per_query = []
    all_self_top1 = True
    all_self_is_max = True
    all_sorted_within_tol = True
    all_mismatches_are_ties = True
    max_sorted_diff_overall = 0.0
    max_matched_diff_overall = 0.0
    for r in range(len(query_rows)):
        own = int(query_rows[r])
        s = sims[r]
        np_order = np.argsort(-s, kind="stable")[:exact_k]  # ties -> ascending id
        f_ids = faiss_I[r]
        f_scores = faiss_D[r].astype(np.float64)
        np_ids = np_order
        np_scores = s[np_order].astype(np.float64)

        # (a) tie-robust exactness: the sorted score vectors must agree
        sorted_diff = float(np.max(np.abs(f_scores - np_scores)))
        max_sorted_diff_overall = max(max_sorted_diff_overall, sorted_diff)
        sorted_within = sorted_diff <= score_tol
        all_sorted_within_tol = all_sorted_within_tol and sorted_within

        set_match = set(int(x) for x in f_ids) == set(int(x) for x in np_ids)
        same = f_ids == np_ids
        pos_match = int(np.sum(same))
        matched_diff = (
            float(np.max(np.abs(f_scores[same] - np_scores[same])))
            if same.any()
            else float("nan")
        )
        if same.any():
            max_matched_diff_overall = max(max_matched_diff_overall, matched_diff)

        # (b) are all positional identity mismatches at (near-)equal score?
        mism = np.where(~same)[0]
        mismatch_gaps = [float(abs(f_scores[j] - np_scores[j])) for j in mism]
        mismatches_are_ties = bool(all(g <= score_tol for g in mismatch_gaps))
        all_mismatches_are_ties = all_mismatches_are_ties and mismatches_are_ties

        # self-match behaviour
        self_top1 = bool(f_ids[0] == own)
        all_self_top1 = all_self_top1 and self_top1
        self_score_faiss = (
            float(f_scores[np.where(f_ids == own)[0][0]])
            if (f_ids == own).any()
            else None
        )
        self_score_numpy = float(s[own])
        global_max = float(s.max())
        self_is_max = bool(self_score_numpy >= global_max - score_tol)
        all_self_is_max = all_self_is_max and self_is_max
        n_tied_at_max = int(np.sum(s >= global_max - score_tol))

        per_query.append(
            {
                "query_row": own,
                "query_doc_key": str(doc_keys[own]) if doc_keys is not None else None,
                "sorted_score_vector_max_abs_diff": sorted_diff,
                "sorted_scores_within_tol": bool(sorted_within),
                "topk_identity_set_match": bool(set_match),
                "positional_id_matches": pos_match,
                "positional_k": int(exact_k),
                "n_positional_mismatches": int(mism.size),
                "all_positional_mismatches_are_ties": mismatches_are_ties,
                "max_score_gap_at_mismatch": (
                    max(mismatch_gaps) if mismatch_gaps else 0.0
                ),
                "max_abs_score_diff_positional_matches": matched_diff,
                "faiss_self_top1": self_top1,
                "self_score_faiss": self_score_faiss,
                "self_score_numpy": self_score_numpy,
                "self_is_global_max_similarity": self_is_max,
                "n_docs_tied_at_max_score": n_tied_at_max,
                "score_tol": score_tol,
            }
        )
    verdict = (
        "EXACT: the sorted top-k similarity-score vectors agree with an "
        "independent numpy fp32 matmul within tolerance, and every identity/order "
        "difference occurs only among documents with tied (equal) scores "
        "(exact-duplicate embeddings)."
        if (all_sorted_within_tol and all_mismatches_are_ties)
        else "DISAGREEMENT beyond tie-breaking -- inspect per-query detail."
    )
    summary = {
        "n_queries": len(query_rows),
        "exact_k": int(exact_k),
        "method": "FAISS GpuIndexFlatIP vs numpy fp32 (xb @ q) full matmul, independent",
        "tie_robust_criterion": "agreement of the sorted top-k similarity-score vectors",
        "max_sorted_score_vector_abs_diff_overall": max_sorted_diff_overall,
        "all_sorted_scores_within_tol": bool(all_sorted_within_tol),
        "max_abs_score_diff_positional_matches_overall": max_matched_diff_overall,
        "all_identity_mismatches_are_ties": bool(all_mismatches_are_ties),
        "all_self_top1": bool(all_self_top1),
        "all_self_is_global_max_similarity": bool(all_self_is_max),
        "score_tol": score_tol,
        "verdict": verdict,
    }
    return summary, per_query


# --------------------------------------------------------------------------- #
# Timing helpers
# --------------------------------------------------------------------------- #
def summarize_batches(batches, total_queries):
    times = [b["elapsed_s"] for b in batches]
    total_time = float(sum(times))
    return {
        "n_batches": len(batches),
        "total_queries": int(total_queries),
        "total_search_time_s": total_time,
        "sustained_queries_per_s": total_queries / total_time
        if total_time > 0
        else float("nan"),
        "avg_batch_time_s": statistics.mean(times) if times else float("nan"),
        "median_batch_time_s": statistics.median(times) if times else float("nan"),
        "min_batch_time_s": min(times) if times else float("nan"),
        "max_batch_time_s": max(times) if times else float("nan"),
    }


def fmt_hms(seconds):
    seconds = float(seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}:{m:02d}:{s:05.2f}"


# --------------------------------------------------------------------------- #
# Environment capture
# --------------------------------------------------------------------------- #
def package_versions():
    import importlib.metadata as im

    pkgs = {}
    for name in [
        "faiss-gpu-cu12",
        "faiss-cpu",
        "numpy",
        "pyarrow",
        "psutil",
        "nvidia-ml-py",
    ]:
        try:
            pkgs[name] = im.version(name)
        except Exception:
            pkgs[name] = None
    return pkgs


def capture_environment(reader_backend):
    import faiss

    env = {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "uname": list(platform.uname()),
        "faiss_version": getattr(faiss, "__version__", None),
        "faiss_num_gpus": int(faiss.get_num_gpus()),
        "faiss_module_file": getattr(faiss, "__file__", None),
        "gpu_memory_backend": reader_backend,
        "nvidia_smi": nvidia_smi_used_mib(),
        "packages": package_versions(),
        "numpy_version": np.__version__,
    }
    # FAISS GPU/CUDA build hints, if exposed
    for attr in ["get_compile_options"]:
        fn = getattr(faiss, attr, None)
        if callable(fn):
            try:
                env["faiss_" + attr] = fn()
            except Exception:
                pass
    return env


# --------------------------------------------------------------------------- #
# Writers
# --------------------------------------------------------------------------- #
def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, default=str)
        fh.write("\n")


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    script_dir = Path(__file__).resolve().parent
    test_dir = script_dir.parent

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--embeddings",
        default=str(
            test_dir.parent / "full_corpus_embedding_0p6b/results/embeddings.npy"
        ),
    )
    ap.add_argument(
        "--index-parquet",
        default=str(
            test_dir.parent
            / "full_corpus_embedding_0p6b/results/embedding_index.parquet"
        ),
    )
    ap.add_argument("--results-dir", default=str(test_dir / "results"))
    ap.add_argument("--environment-dir", default=str(test_dir / "environment"))
    ap.add_argument("--expected-rows", type=int, default=902144)
    ap.add_argument("--expected-dim", type=int, default=1024)
    ap.add_argument("--n-queries", type=int, default=10000)
    ap.add_argument(
        "--k",
        type=int,
        default=100,
        help="closest OTHER docs to retain (benchmark only)",
    )
    ap.add_argument("--seed", type=int, default=20260910)
    ap.add_argument("--batch-sizes", default="256,1024,4096,8192")
    ap.add_argument("--neighbor-ks", default="20,100,500")
    ap.add_argument("--n-exact", type=int, default=6)
    ap.add_argument("--exact-k", type=int, default=110)
    ap.add_argument("--score-tol", type=float, default=1e-4)
    ap.add_argument("--n-warmup", type=int, default=5)
    ap.add_argument("--warmup-batch", type=int, default=4096)
    ap.add_argument("--main-batch", type=int, default=0, help="0 = choose from sweep")
    ap.add_argument(
        "--plateau-frac",
        type=float,
        default=0.90,
        help="pick smallest batch whose QPS >= this fraction of the max",
    )
    ap.add_argument("--nvml-interval-ms", type=float, default=20.0)
    ap.add_argument(
        "--quick", action="store_true", help="tiny smoke run; exercises every path"
    )
    args = ap.parse_args()

    if args.quick:
        args.n_queries = 300
        args.batch_sizes = "128,256"
        args.n_exact = 3
        args.exact_k = 40
        args.n_warmup = 2
        args.warmup_batch = 128

    batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x.strip()]
    neighbor_ks = [int(x) for x in args.neighbor_ks.split(",") if x.strip()]

    results_dir = Path(args.results_dir)
    env_dir = Path(args.environment_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    env_dir.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    run_t0 = time.perf_counter()
    reader, backend = make_gpu_reader()
    log(f"GPU memory backend: {backend}")

    import faiss

    log(f"faiss {faiss.__version__}  num_gpus={faiss.get_num_gpus()}")
    if faiss.get_num_gpus() < 1:
        log("ERROR: FAISS reports 0 GPUs; aborting.")
        sys.exit(2)

    env = capture_environment(backend)
    write_json(env_dir / "environment.json", env)

    # ----- host RAM before/after load -----
    rss_before = host_rss_bytes()
    log(f"Loading embeddings from {args.embeddings} ...")
    load_t0 = time.perf_counter()
    xb = load_embeddings(args.embeddings)
    load_s = time.perf_counter() - load_t0
    rss_after_load = host_rss_bytes()
    log(f"Loaded {xb.shape} {xb.dtype} in {load_s:.2f}s")

    matrix_info = verify_matrix(xb, args.expected_rows, args.expected_dim)
    doc_keys = None
    index_align = {}
    try:
        doc_keys, index_align = read_doc_keys(args.index_parquet, xb.shape[0])
    except Exception as exc:
        index_align = {"error": str(exc)}
        log(f"WARN: could not read doc_key parquet: {exc}")

    # ----- build GPU flat IP index -----
    log("Building GpuIndexFlatIP and adding all vectors ...")
    index, res, build_info = build_gpu_flat_ip(xb, reader, args.nvml_interval_ms)
    rss_after_index = host_rss_bytes()
    log(
        f"Index built in {build_info['build_seconds']:.2f}s  ntotal={build_info['ntotal']} "
        f"gpu_after={build_info['gpu_used_after_mib']:.1f} MiB"
    )

    index_load = {
        "embeddings_path": args.embeddings,
        "npy_load_seconds": load_s,
        "matrix": matrix_info,
        "index_parquet_alignment": index_align,
        "build": build_info,
        "host_rss_before_bytes": rss_before,
        "host_rss_after_load_bytes": rss_after_load,
        "host_rss_after_index_bytes": rss_after_index,
        "host_rss_load_delta_bytes": (rss_after_load - rss_before)
        if (rss_before and rss_after_load)
        else None,
        "host_rss_after_load_gib": (rss_after_load / GIB) if rss_after_load else None,
        "xb_nbytes": int(xb.nbytes),
        "xb_gib": xb.nbytes / GIB,
        "ceiling_mib": CEILING_MIB,
        "under_ceiling_after_build": bool(
            build_info["gpu_peak_during_build_mib"] < CEILING_MIB
        ),
        "nvidia_smi_at_index_loaded": nvidia_smi_used_mib(),
    }
    write_json(results_dir / "index_load.json", index_load)

    # ----- select real query rows -----
    rng = np.random.default_rng(args.seed)
    n_q = min(args.n_queries, xb.shape[0])
    query_rows = np.sort(rng.choice(xb.shape[0], size=n_q, replace=False)).astype(
        np.int64
    )
    xq_main = np.ascontiguousarray(xb[query_rows])
    log(f"Selected {n_q} real query rows (seed={args.seed})")

    # ----- warm-up (untimed) -----
    log("Warm-up (untimed) ...")
    warm_info = warmup(
        index,
        xq_main,
        reader,
        args.nvml_interval_ms,
        args.n_warmup,
        min(args.warmup_batch, n_q),
        neighbor_ks,
    )

    # ----- query-batch-size sweep -----
    log(f"Query-batch-size sweep over {batch_sizes} (k={args.k}) ...")
    sweep_rows = []
    for bs in batch_sizes:
        bs_eff = min(bs, n_q)
        # small per-size warm
        index.search(np.ascontiguousarray(xq_main[:bs_eff]), args.k)
        _, _, sb, peak = search_all(
            index, xq_main, args.k, bs_eff, reader, args.nvml_interval_ms, time_it=True
        )
        summ = summarize_batches(sb, n_q)
        row = {
            "batch_size": bs_eff,
            "n_queries": n_q,
            "k": args.k,
            "total_search_time_s": summ["total_search_time_s"],
            "queries_per_s": summ["sustained_queries_per_s"],
            "avg_batch_time_s": summ["avg_batch_time_s"],
            "peak_gpu_mib": peak,
            "under_ceiling": bool(peak < CEILING_MIB),
        }
        sweep_rows.append(row)
        log(
            f"  batch={bs_eff:>6}  {row['queries_per_s']:>10.1f} q/s  peak={peak:.1f} MiB"
        )

    max_qps = max(r["queries_per_s"] for r in sweep_rows)
    n_on_plateau = sum(
        1 for r in sweep_rows if r["queries_per_s"] >= args.plateau_frac * max_qps
    )
    chosen = max(
        sweep_rows, key=lambda r: r["queries_per_s"]
    )  # fastest = clearly efficient
    chosen_batch = args.main_batch if args.main_batch > 0 else chosen["batch_size"]
    log(
        f"Chosen practical batch size = {chosen_batch} "
        f"(fastest of {len(sweep_rows)} tested; {n_on_plateau} within "
        f"{args.plateau_frac:.0%} of max {max_qps:.1f} q/s)"
    )

    write_csv(
        results_dir / "batch_sweep.csv",
        sweep_rows,
        [
            "batch_size",
            "n_queries",
            "k",
            "total_search_time_s",
            "queries_per_s",
            "avg_batch_time_s",
            "peak_gpu_mib",
            "under_ceiling",
        ],
    )
    write_json(
        results_dir / "batch_sweep.json",
        {
            "batch_sizes_tested": batch_sizes,
            "k": args.k,
            "n_queries": n_q,
            "max_queries_per_s": max_qps,
            "plateau_fraction": args.plateau_frac,
            "n_batch_sizes_within_plateau": int(n_on_plateau),
            "chosen_batch_size": int(chosen_batch),
            "chosen_reason": "fastest measured batch size (argmax q/s)"
            if args.main_batch <= 0
            else "overridden via --main-batch",
            "rows": sweep_rows,
        },
    )

    # ----- MAIN timed test: 10,000 queries, k+1 then drop self, keep k -----
    request_k = args.k + 1  # +1 so the query's own row can be removed
    log(
        f"MAIN timed test: {n_q} queries, request_k={request_k}, keep {args.k} others, "
        f"batch={chosen_batch} ..."
    )
    # extra warm at the chosen batch/k
    index.search(np.ascontiguousarray(xq_main[: min(chosen_batch, n_q)]), request_k)
    main_wall_t0 = time.perf_counter()
    D, nbr_ids, main_batches, main_peak = search_all(
        index,
        xq_main,
        request_k,
        chosen_batch,
        reader,
        args.nvml_interval_ms,
        time_it=True,
    )
    kept_ids, kept_scores, self_stats = drop_self_keep_k(nbr_ids, D, query_rows, args.k)
    main_wall_s = time.perf_counter() - main_wall_t0

    main_summary = summarize_batches(main_batches, n_q)
    main_summary.update(
        {
            "requested_k": request_k,
            "kept_k_others": args.k,
            "peak_gpu_mib": main_peak,
            "under_ceiling": bool(main_peak < CEILING_MIB),
            "wall_s_including_self_removal": main_wall_s,
            "self_match": self_stats,
            "batch_size": int(chosen_batch),
        }
    )
    write_csv(
        results_dir / "main_batches.csv",
        main_batches,
        [
            "batch_index",
            "n_queries",
            "elapsed_s",
            "queries_per_s",
            "peak_gpu_mib_so_far",
        ],
    )
    write_json(results_dir / "main_summary.json", main_summary)
    log(
        f"MAIN: {main_summary['sustained_queries_per_s']:.1f} q/s sustained; "
        f"peak={main_peak:.1f} MiB; self@rank0={self_stats['self_at_rank0']}/{n_q}"
    )

    # ----- 20 / 100 / 500 neighbor-count comparison (same queries, warmed) -----
    log(f"Neighbor-count comparison over ks={neighbor_ks} (batch={chosen_batch}) ...")
    ncmp_rows = []
    for k in neighbor_ks:
        index.search(
            np.ascontiguousarray(xq_main[: min(chosen_batch, n_q)]), k
        )  # warm this k
        _, _, kb, peak = search_all(
            index, xq_main, k, chosen_batch, reader, args.nvml_interval_ms, time_it=True
        )
        summ = summarize_batches(kb, n_q)
        row = {
            "neighbors_k": k,
            "n_queries": n_q,
            "batch_size": int(chosen_batch),
            "total_search_time_s": summ["total_search_time_s"],
            "queries_per_s": summ["sustained_queries_per_s"],
            "avg_batch_time_s": summ["avg_batch_time_s"],
            "peak_gpu_mib": peak,
        }
        ncmp_rows.append(row)
        log(f"  k={k:>4}  {row['queries_per_s']:>10.1f} q/s  peak={peak:.1f} MiB")
    base = next((r for r in ncmp_rows if r["neighbors_k"] == 100), ncmp_rows[0])
    for r in ncmp_rows:
        r["qps_relative_to_k100"] = r["queries_per_s"] / base["queries_per_s"]
    write_csv(
        results_dir / "neighbor_count_comparison.csv",
        ncmp_rows,
        [
            "neighbors_k",
            "n_queries",
            "batch_size",
            "total_search_time_s",
            "queries_per_s",
            "avg_batch_time_s",
            "peak_gpu_mib",
            "qps_relative_to_k100",
        ],
    )
    write_json(results_dir / "neighbor_count_comparison.json", {"rows": ncmp_rows})

    # ----- exactness validation -----
    log(f"Exactness validation on {args.n_exact} queries (exact_k={args.exact_k}) ...")
    exact_rows_sel = query_rows[np.linspace(0, n_q - 1, args.n_exact).astype(int)]
    exact_summary, exact_per_query = exactness_check(
        index,
        xb,
        exact_rows_sel,
        min(args.exact_k, xb.shape[0]),
        doc_keys,
        args.score_tol,
    )
    write_json(
        results_dir / "exactness_validation.json",
        {"summary": exact_summary, "per_query": exact_per_query},
    )
    log(
        f"Exactness: sorted_score_diff={exact_summary['max_sorted_score_vector_abs_diff_overall']:.2e} "
        f"sorted_within_tol={exact_summary['all_sorted_scores_within_tol']} "
        f"mismatches_are_ties={exact_summary['all_identity_mismatches_are_ties']} "
        f"self_is_max={exact_summary['all_self_is_global_max_similarity']}"
    )

    # ----- full-corpus runtime extrapolation -----
    qps = main_summary["sustained_queries_per_s"]
    full_n = args.expected_rows
    proj_s = full_n / qps if qps > 0 else float("nan")
    full_est = {
        "basis": f"EXTRAPOLATION from the measured {n_q}-query timed test "
        f"(k+1={request_k} retrieved, batch={chosen_batch})",
        "measured_sustained_queries_per_s": qps,
        "full_corpus_queries": full_n,
        "projected_search_time_s": proj_s,
        "projected_search_time_min": proj_s / 60.0,
        "projected_search_time_hours": proj_s / 3600.0,
        "projected_search_time_hms": fmt_hms(proj_s),
        "note": "Search-time only (index already built/loaded). NOT run in this task.",
    }
    write_json(results_dir / "full_corpus_estimate.json", full_est)
    log(
        f"Full-corpus estimate: {full_n} queries at {qps:.1f} q/s -> "
        f"{proj_s:.1f}s ({fmt_hms(proj_s)})"
    )

    # ----- top-level run.json -----
    run = {
        "task": "FAISS GPU exact (flat inner-product) search benchmark on the "
        "full-corpus Qwen3-Embedding-0.6B embeddings",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "quick_mode": bool(args.quick),
        "seed": args.seed,
        "config": {
            "n_queries": n_q,
            "k_keep_others": args.k,
            "requested_k_main": request_k,
            "batch_sizes_swept": batch_sizes,
            "chosen_batch_size": int(chosen_batch),
            "neighbor_ks_compared": neighbor_ks,
            "n_exact": args.n_exact,
            "exact_k": int(min(args.exact_k, xb.shape[0])),
            "nvml_interval_ms": args.nvml_interval_ms,
        },
        "index_type": "faiss.GpuIndexFlatIP (flat, exhaustive, inner-product == cosine on L2-normalized vectors)",
        "no_approximation": {
            "ivf": False,
            "hnsw": False,
            "pq": False,
            "quantization": False,
            "compression": False,
            "clustering": False,
            "approximate": False,
            "float32_vectors": True,
        },
        "environment": env,
        "index_load": index_load,
        "warmup": warm_info,
        "batch_sweep_chosen": int(chosen_batch),
        "main_summary": main_summary,
        "neighbor_count_comparison": ncmp_rows,
        "exactness_summary": exact_summary,
        "full_corpus_estimate": full_est,
        "total_script_wall_s": time.perf_counter() - run_t0,
    }
    write_json(results_dir / "run.json", run)
    log(f"DONE in {time.perf_counter() - run_t0:.1f}s. Results in {results_dir}")


if __name__ == "__main__":
    main()
