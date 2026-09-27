#!/usr/bin/env python3
"""Re-embed the largest topic clusters on their own with a new 2-D UMAP (GPU, cuML).

Takes the posts in the largest HDBSCAN clusters from Phase 3, leaving out the
removed-post placeholder cluster, and fits a fresh 2-D UMAP to their Phase 2
embeddings at one or more min_dist values. The Phase 3 labels and coordinates are
only read. Each output parquet gets a .manifest.yml next to it.

Env: KC_REPO_ROOT (default: the repository root). Paths can also be given as flags.
"""

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone

import numpy as np

REPO_ROOT = os.environ.get(
    "KC_REPO_ROOT",
    os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..")
    ),
)
PIPE = os.path.join(REPO_ROOT, "pipeline")
STEP = os.path.join(
    PIPE, "phase4_semantic_structure_description_qc", "4d_topic_figures"
)

# Cluster 160 is the "[removed]" placeholder text left when a post is removed. It is the
# largest cluster but not a topic, so it is left out of the ranking.
PLACEHOLDER_CLUSTERS = (160,)
N_TOPICS = 15
UMAP_SETTINGS = dict(n_neighbors=15, n_components=2, metric="cosine", random_state=42)
MIN_DISTS = (0.0, 0.3, 0.5)


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    print(f"[{utcnow()}] {msg}", flush=True)


def relpath(p: str) -> str:
    ap = os.path.abspath(p).replace("\\", "/")
    root = REPO_ROOT.replace("\\", "/").rstrip("/") + "/"
    return ap[len(root) :] if ap.startswith(root) else ap


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


def top_cluster_ids(labels: np.ndarray, k: int, exclude=PLACEHOLDER_CLUSTERS) -> list:
    """The k largest cluster ids by size, largest first. Noise (-1) and `exclude` are
    left out. Ties are broken by the smaller id, so the result is deterministic."""
    labels = np.asarray(labels)
    ids, counts = np.unique(labels[labels >= 0], return_counts=True)
    keep = ~np.isin(ids, np.asarray(exclude, dtype=ids.dtype))
    ids, counts = ids[keep], counts[keep]
    order = np.lexsort((ids, -counts))
    return [int(i) for i in ids[order][:k]]


def select_rows(labels: np.ndarray, cluster_ids) -> np.ndarray:
    """Row indices (ascending) of every document whose label is in `cluster_ids`."""
    return np.flatnonzero(np.isin(np.asarray(labels), np.asarray(list(cluster_ids))))


def check_alignment(index_keys, label_keys) -> None:
    """The embedding index and the label table must list the same doc_keys in the
    same order, because rows are matched by position."""
    a, b = np.asarray(index_keys), np.asarray(label_keys)
    if a.shape != b.shape or not np.array_equal(a, b):
        raise ValueError("embedding index and labels are not row-aligned by doc_key")


def md_tag(md: float) -> str:
    return f"md{md:.1f}"


def git_short_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", REPO_ROOT, "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def software_block():
    import importlib.metadata as im

    out = []
    for n in ("cuml", "cupy", "cupy-cuda12x", "numpy", "pandas", "pyarrow", "pyyaml"):
        try:
            out.append({"name": n, "version": im.version(n)})
        except Exception:
            continue
    return out


def write_manifest(output_path, inputs, parameters, rows, cols, notes=""):
    import yaml

    man = {
        "manifest_version": 1,
        "output_file": relpath(output_path),
        "output_hash": sha256_file(output_path),
        "output_format": "parquet",
        "output_rows": int(rows),
        "output_cols": int(cols),
        "input_files": inputs,
        "transformation": {
            "script": relpath(os.path.abspath(__file__)),
            "script_hash": sha256_file(os.path.abspath(__file__)),
            "parameters": parameters,
            "git_commit": git_short_head(),
        },
        "software": software_block(),
        "seed": UMAP_SETTINGS["random_state"],
        "timestamp": utcnow(),
        "notes": notes,
    }
    with open(output_path + ".manifest.yml", "w") as f:
        yaml.safe_dump(man, f, sort_keys=False, default_flow_style=False)


def main():
    p2 = os.path.join(
        PIPE, "phase2_semantic_embeddings", "full_corpus_embedding_0p6b", "results"
    )
    p3 = os.path.join(
        PIPE,
        "phase3_blind_semantic_structure_discovery",
        "full_corpus_gpu_umap_hdbscan_baseline",
        "results",
    )
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--embeddings", default=os.path.join(p2, "embeddings.npy"))
    ap.add_argument(
        "--embedding-index", default=os.path.join(p2, "embedding_index.parquet")
    )
    ap.add_argument("--labels", default=os.path.join(p3, "hdbscan_labels.parquet"))
    ap.add_argument("--results-dir", default=os.path.join(STEP, "results"))
    ap.add_argument("--n-topics", type=int, default=N_TOPICS)
    args = ap.parse_args()

    import pyarrow as pa
    import pyarrow.parquet as pq
    from cuml.manifold import UMAP

    os.makedirs(args.results_dir, exist_ok=True)
    idx = (
        pq.read_table(args.embedding_index, columns=["doc_key"])
        .column("doc_key")
        .to_numpy(zero_copy_only=False)
    )
    lab = pq.read_table(args.labels, columns=["doc_key", "label"])
    keys = lab.column("doc_key").to_numpy(zero_copy_only=False)
    labels = lab.column("label").to_numpy()
    check_alignment(idx, keys)

    tids = top_cluster_ids(labels, args.n_topics)
    rows = select_rows(labels, tids)
    log(f"topics {tids}; {len(rows):,} documents")

    emb = np.load(args.embeddings, mmap_mode="r")
    if emb.shape[0] != len(labels):
        raise ValueError(
            f"embeddings have {emb.shape[0]} rows, labels have {len(labels)}"
        )
    X = np.ascontiguousarray(emb[rows], dtype=np.float32)
    if not np.isfinite(X).all():
        raise ValueError("non-finite values in the selected embeddings")

    inputs = [
        {"path": relpath(pth), "hash": sha256_file(pth)}
        for pth in (args.embeddings, args.embedding_index, args.labels)
    ]
    summary = {"topics": tids, "n_documents": int(len(rows)), "runs": []}
    for md in MIN_DISTS:
        t0 = time.perf_counter()
        Y = UMAP(min_dist=md, **UMAP_SETTINGS).fit_transform(X)
        Y = np.asarray(Y.get() if hasattr(Y, "get") else Y)
        secs = time.perf_counter() - t0
        if not np.isfinite(Y).all():
            raise ValueError(f"non-finite coordinates at min_dist={md}")
        out = os.path.join(
            args.results_dir,
            f"top_topics_umap_nn{UMAP_SETTINGS['n_neighbors']}_{md_tag(md)}.parquet",
        )
        tbl = pa.table(
            {
                "row": rows.astype(np.int64),
                "doc_key": keys[rows],
                "label": labels[rows],
                "umap_0": Y[:, 0].astype(np.float32),
                "umap_1": Y[:, 1].astype(np.float32),
            }
        )
        pq.write_table(tbl, out, compression="zstd")
        params = dict(
            UMAP_SETTINGS,
            min_dist=md,
            topics=tids,
            placeholder_clusters_excluded=list(PLACEHOLDER_CLUSTERS),
        )
        write_manifest(
            out,
            inputs,
            params,
            tbl.num_rows,
            tbl.num_columns,
            notes="2-D coordinates for figures only; not used for clustering.",
        )
        summary["runs"].append(
            {"min_dist": md, "seconds": round(secs, 1), "output": relpath(out)}
        )
        log(f"min_dist={md}: {secs:.1f} s -> {relpath(out)}")

    with open(os.path.join(args.results_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    log("done")


if __name__ == "__main__":
    main()
