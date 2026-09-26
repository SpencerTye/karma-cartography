#!/usr/bin/env python3
"""Write .manifest.yml sidecars for the secondary outputs of the three-day feasibility run.

Covers summary tables, measurement records, telemetry and figures; the main data files already
get manifests from run_umap_hdbscan.py. No GPU needed.
"""

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_umap_hdbscan as R  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(BASE, "results")
FIG = os.path.join(BASE, "figures")
SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_umap_hdbscan.py")

EMB = f"{R.REPO_ROOT}/pipeline/phase2_semantic_embeddings/three_day_embedding_0p6b/results/embeddings.npy"
IDX = f"{R.REPO_ROOT}/pipeline/phase2_semantic_embeddings/three_day_embedding_0p6b/results/embedding_index.parquet"

git = R.git_short_head()
inputs = [
    {
        "path": R.relpath(EMB),
        "hash": R.sha256_file(EMB),
        "format": "numpy",
        "role": "three-day 0.6B embeddings (float32, L2-normalized), read-only",
        "rows": 101722,
        "cols": 1024,
    },
    {
        "path": R.relpath(IDX),
        "hash": R.sha256_file(IDX),
        "format": "parquet",
        "role": "row -> doc_key index for the embeddings, read-only",
        "rows": 101722,
        "cols": 6,
    },
]
params = {
    "record_of": "three-day GPU UMAP + HDBSCAN feasibility run",
    "umap": R.UMAP_DEFAULTS,
    "hdbscan": dict(R.HDBSCAN_DEFAULTS, gen_min_span_tree=True),
    "note": "See results/summary.json for the full config, measurements, and environment.",
}
EMPTY_DIFF = {"added": [], "removed": [], "renamed": [], "retyped": []}


def csv_dims(path):
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    return (max(len(rows) - 1, 0), len(rows[0]) if rows else 0)


def already(path):
    return os.path.exists(path + ".manifest.yml")


written = []
for fn in sorted(os.listdir(RES)):
    p = os.path.join(RES, fn)
    if (
        not os.path.isfile(p)
        or fn.endswith(".manifest.yml")
        or fn == "_cluster_sizes.txt"
    ):
        continue
    if already(p):
        continue  # primary artifacts already manifested by the run
    ext = fn.rsplit(".", 1)[-1].lower()
    if ext == "csv":
        rows, cols = csv_dims(p)
        R.write_manifest(
            p,
            inputs,
            SCRIPT,
            params,
            git,
            tabular=(rows, cols),
            out_format="csv",
            notes="Summary/telemetry table from the feasibility run.",
            seed=None,
            schema_diff=EMPTY_DIFF,
        )
    else:
        R.write_manifest(
            p,
            inputs,
            SCRIPT,
            params,
            git,
            out_format="other",
            notes=f"Measurement/record artifact ({ext}) from the feasibility run.",
            seed=None,
        )
    written.append(R.relpath(p))

for fn in sorted(os.listdir(FIG)):
    if not fn.endswith(".png"):
        continue
    p = os.path.join(FIG, fn)
    if already(p):
        continue
    R.write_manifest(
        p,
        inputs,
        SCRIPT,
        params,
        git,
        out_format="image/png",
        notes="Diagnostic figure (title + axis/tick labels only; clusters not named or interpreted).",
        seed=None,
    )
    written.append(R.relpath(p))

print(f"wrote {len(written)} extra manifests:")
for w in written:
    print("  " + w)
