#!/usr/bin/env python3
"""Write provenance sidecars for the remaining (non-primary) outputs of the
full-corpus baseline run -- summary/verification tables, measurement records,
telemetry, and the diagnostic figures -- so every non-code output carries a
.manifest.yml, matching the project norm. The primary data artifacts (UMAP 5-D
and 2-D coordinates, HDBSCAN labels, and the three trees) are already manifested
by run_full_corpus_umap_hdbscan.py, which is also recorded as the producing script
here. No GPU required.
"""

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_full_corpus_umap_hdbscan as R  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(BASE, "results")
FIG = os.path.join(BASE, "figures")
SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "run_full_corpus_umap_hdbscan.py"
)

FC = f"{R.REPO_ROOT}/pipeline/phase2_semantic_embeddings/full_corpus_embedding_0p6b/results"
EMB = f"{FC}/embeddings.npy"
IDX = f"{FC}/embedding_index.parquet"

git = R.git_short_head()
inputs = [
    {
        "path": R.relpath(EMB),
        "hash": R.sha256_file(EMB),
        "format": "numpy",
        "role": "full-corpus 0.6B embeddings (float32, L2-normalized), read-only",
        "rows": R.EXPECTED_ROWS,
        "cols": R.EXPECTED_DIM,
    },
    {
        "path": R.relpath(IDX),
        "hash": R.sha256_file(IDX),
        "format": "parquet",
        "role": "row -> doc_key index for the embeddings, read-only",
        "rows": R.EXPECTED_ROWS,
        "cols": 6,
    },
]
params = {
    "record_of": "full-corpus GPU UMAP + HDBSCAN baseline run (902,144 documents)",
    "umap_5d": R.UMAP_DEFAULTS,
    "umap_2d_viz": R.VIZ_UMAP_DEFAULTS,
    "hdbscan": dict(R.HDBSCAN_DEFAULTS, gen_min_span_tree=True),
    "note": "See results/summary.json for the full config, measurements, verification, and environment.",
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
            notes="Summary/telemetry table from the full-corpus baseline run.",
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
            notes=f"Measurement/record artifact ({ext}) from the full-corpus baseline run.",
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
