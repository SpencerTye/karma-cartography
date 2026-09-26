# Three-day GPU UMAP + HDBSCAN feasibility test

**Phase 3 — blind semantic structure discovery (feasibility only).**

A single feasibility run of RAPIDS **cuML** UMAP followed by cuML HDBSCAN on the
existing three-day 0.6B embedding matrix (101,722 documents × 1024 dims), on the
**RTX 3090**, measuring runtime and **whole-card** GPU memory against a 21 GiB
ceiling. This directory contains the reproducible code, the fixed configuration,
the exact environment, all measurements, the keyed 5-D coordinates and HDBSCAN
labels/diagnostics, diagnostic figures, provenance manifests, and the report.

This is a **feasibility and measurement** exercise. Clusters are **not named or
interpreted**; the analytical settings are **fixed** (not tuned or swept).

---

## Inputs (read-only; never modified)

| Input | Path | Shape |
|---|---|---|
| Embeddings | `pipeline/phase2_semantic_embeddings/three_day_embedding_0p6b/results/embeddings.npy` | float32 `[101722, 1024]`, L2-normalized |
| Row → doc_key index | `.../three_day_embedding_0p6b/results/embedding_index.parquet` | 101,722 × 6 |

All 101,722 records are used exactly as provided — no filtering, removal,
collapsing, deduplication, sampling, or recoding. Embedding row *i* is aligned to
`embedding_index` row *i* (verified: `row` column equals `0..101721`).

## What is run

1. **GPU UMAP** (`cuml.manifold.UMAP`) on all 101,722 × 1024 embeddings →
   5-D representation.
2. **GPU HDBSCAN** (`cuml.cluster.hdbscan.HDBSCAN`) on the 5-D UMAP output.

### Fixed configuration (task spec — not tuned)

```
UMAP     : n_neighbors=15, n_components=5, min_dist=0.0, metric=cosine
HDBSCAN  : min_cluster_size=10, min_samples=10, metric=euclidean,
           cluster_selection_method=eom
```

Everything else is the cuML default. `metric='cosine'` is explicitly supported by
cuML UMAP. For N > 50,000 cuML UMAP's `build_algo='auto'` selects **NN-Descent**
(approximate KNN); this is the default and was not overridden. HDBSCAN is run with
`gen_min_span_tree=True` **only** so the minimum-spanning-tree diagnostic can be
exported — this does not change the labels, probabilities, or persistence. All
computation is **float32** (the stored embedding dtype); no precision change.

Reproducibility: UMAP is run with the fixed config's default `random_state=None`.
Per the cuML docs a `random_state` "enables reproducible embeddings, but at the
cost of slower training and increased memory usage", and NN-Descent (auto-selected
for N > 50k) is itself non-deterministic. Because this run measures runtime and
peak memory, seeding would distort the measured quantities, so the run is left
unseeded. **The 5-D coordinates and the labels derived from them are therefore
run-specific**; re-running produces statistically similar but not identical output
(e.g., two runs here gave 883 vs 873 clusters). HDBSCAN is deterministic given its
input, subject to tie-breaking among equal distances.

## Environment

Isolated `uv` venv `~/venvs/rapids-umap-hdbscan` (Python 3.12), built by
`scripts/setup_env.sh`. **The existing Phase 2 (`socialmediadgg-phase2`) and FAISS
(`faiss-bench`) venvs were not touched.** Runs in WSL2 Ubuntu-24.04.

Key versions (full list in `environment/packages.txt`; capture in
`environment/cuml_env.json`):

- `cuml-cu12 26.08.00`, `cudf-cu12 26.8.1`, `libcuvs-cu12 26.8.1`, `rmm-cu12 26.8.0`
- `cupy-cuda12x 14.2.0` (CUDA runtime 12.9), NVIDIA driver 596.49 (CUDA 13.2)
- `hdbscan 0.8.44` (CPU package, for tree `.to_pandas()` export only)
- `numpy 2.4.6`, `pyarrow 23.0.1`, `pandas 3.0.3`, `matplotlib 3.11.1`,
  `scikit-learn 1.9.1`, `scipy 1.18.1`, `nvidia-ml-py 13.610.43`, `pyyaml 6.0.3`
- GPU: NVIDIA GeForce RTX 3090, 24,576 MiB, compute capability 8.6

## Reproduce

From WSL2 Ubuntu-24.04:

```bash
# 1. build the isolated env (idempotent; writes environment/*)
bash "pipeline/phase3_blind_semantic_structure_discovery/three_day_gpu_umap_hdbscan_feasibility/scripts/setup_env.sh"

# 2. run the full feasibility test (writes results/ and figures/)
~/venvs/rapids-umap-hdbscan/bin/python \
  "pipeline/phase3_blind_semantic_structure_discovery/three_day_gpu_umap_hdbscan_feasibility/scripts/run_umap_hdbscan.py"

# 3. write provenance sidecars for the summary tables/figures
~/venvs/rapids-umap-hdbscan/bin/python \
  "pipeline/phase3_blind_semantic_structure_discovery/three_day_gpu_umap_hdbscan_feasibility/scripts/write_extra_manifests.py"
```

Useful flags on `run_umap_hdbscan.py`: `--quick` (small-subset smoke test),
`--n-rows N` (subset), `--ceiling-mib` (default 21504), `--figures-only`
(rebuild figures from saved results). Unit tests:
`~/venvs/rapids-umap-hdbscan/bin/python tests/test_run_umap_hdbscan_functions.py`.

## Outputs

`results/`

| File | Contents |
|---|---|
| `umap_5d_coordinates.parquet` | `row, doc_key, record_type, umap_0..umap_4` — keyed 5-D coordinates |
| `umap_5d_coordinates.npy` | raw float32 `[101722, 5]` matrix (row-aligned to the index) |
| `hdbscan_labels.parquet` | `row, doc_key, record_type, label, probability` — raw labels (incl. -1) + membership |
| `hdbscan_cluster_persistence.csv` | per-cluster `cluster_label, size, persistence` |
| `hdbscan_condensed_tree.parquet` | condensed tree (`parent, child, lambda_val, child_size`) |
| `hdbscan_single_linkage_tree.parquet` | single-linkage tree |
| `hdbscan_minimum_spanning_tree.parquet` | MST (`from, to, distance`) |
| `summary.json` | full config, measurements, counts, cluster stats, tie stats, environment, estimate |
| `measurements.csv` | runtime + memory per operation |
| `gpu_memory_samples_{umap,hdbscan}.csv` | NVML whole-card + host-RSS time series |
| `full_dataset_estimate.json` | measured vs extrapolated 902,144-doc estimate |
| `_cluster_sizes.txt` | internal helper (per-cluster sizes; supports `--figures-only`) |

`figures/` — six diagnostic PNGs (see `REPORT.md`). `environment/` — versions,
`nvidia-smi`, setup log. Every non-code output carries a `<file>.manifest.yml` provenance
provenance sidecar (input hashes, output hash, producing-script hash, parameters,
package versions, git commit); all 18 validate against the project schema.

## Scope

Feasibility and measurement only. No cluster is named or interpreted; no analytical
setting is tuned; the 902,144-document run is **not** performed here (only estimated).
See `REPORT.md`.
