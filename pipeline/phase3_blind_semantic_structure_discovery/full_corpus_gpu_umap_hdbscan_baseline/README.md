# Full-corpus GPU UMAP + HDBSCAN baseline

**Phase 3 — blind semantic structure discovery (full corpus).**

Runs RAPIDS **cuML** UMAP → cuML HDBSCAN on the **complete 902,144-document** 0.6B
embedding matrix, plus a separate 2-D UMAP for visualisation, on the **RTX 3090**,
measuring runtime and **whole-card** GPU memory against a 21 GiB ceiling. This
directory contains the reproducible code, the fixed configuration, the exact
environment, all measurements, the keyed 5-D / 2-D coordinates and HDBSCAN
labels/diagnostics, diagnostic figures, provenance manifests, the completion
verification, and the report.

This scales the validated three-day feasibility run
(`../three_day_gpu_umap_hdbscan_feasibility`) to the full corpus **without
redesigning the workflow**: the analytical cuML calls, parameters, and diagnostic
options are identical. The only additions are the full-corpus inputs, the 2-D
visualisation UMAP, the full-corpus diagnostic figures, and the completion
verification. Clusters are **not named or interpreted**; the analytical settings are
**fixed** (not tuned or swept); no record is filtered, removed, collapsed,
deduplicated, or recoded.

---

## Inputs (read-only; never modified)

| Input | Path | Shape |
|---|---|---|
| Embeddings | `pipeline/phase2_semantic_embeddings/full_corpus_embedding_0p6b/results/embeddings.npy` | float32 `[902144, 1024]`, L2-normalized |
| Row → doc_key index | `.../full_corpus_embedding_0p6b/results/embedding_index.parquet` | 902,144 × 6 |

Verified source hashes (confirmed unchanged before and after this run):

- `embeddings.npy` — `sha256:b5755f233187ddb99da7bc445851dce173b13864331a7409c6bcc28132952913`
- `embedding_index.parquet` — `sha256:2c89f296bd5d7a2ebd462d00ece02a2778db6a88a7e9932836abf722ba53c7f5`

All 902,144 records are used exactly as provided. Embedding row *i* is aligned to
`embedding_index` row *i* (verified: `row` column equals `0..902143`). Index columns:
`row, doc_key, record_type, created_utc, created_date_utc, n_tokens_4B`.

## What is run (in order)

1. **GPU UMAP** (`cuml.manifold.UMAP`) on all 902,144 × 1024 embeddings → **5-D**
   analytical coordinates.
2. **GPU HDBSCAN** (`cuml.cluster.HDBSCAN`) on the 5-D UMAP output → clustering.
3. **GPU UMAP** (`cuml.manifold.UMAP`) → **2-D visualisation** coordinates. This 2-D
   representation is **visualisation only**: it is never fed to HDBSCAN and never
   changes the cluster assignments.

The analytical outputs (5-D coordinates, HDBSCAN labels) are saved to disk the
moment each stage completes, so evidence is preserved even if a later stage stops.

### Fixed configuration (task spec — not tuned)

```
UMAP (5-D analytical)   : n_neighbors=15, n_components=5, min_dist=0.0, metric=cosine
HDBSCAN                 : min_cluster_size=10, min_samples=10, metric=euclidean,
                          cluster_selection_method=eom
UMAP (2-D visualisation): n_neighbors=15, n_components=2, min_dist=0.0, metric=cosine
```

Everything else is the cuML default. `metric='cosine'` is explicitly supported by
cuML UMAP. For N > 50,000 cuML UMAP's `build_algo='auto'` selects **NN-Descent**
(approximate KNN); this is the default and was not overridden. HDBSCAN is run with
`gen_min_span_tree=True` **only** so the minimum-spanning-tree diagnostic can be
exported — this does not change the labels, probabilities, or persistence. All
computation is **float32** (the stored embedding dtype); no precision change.

Reproducibility: UMAP is run with the fixed config's default `random_state=None`.
Per the cuML docs a `random_state` "enables reproducible embeddings, but at the cost
of slower training and increased memory usage", and NN-Descent (auto-selected for
N > 50k) is itself non-deterministic. Because this run measures runtime and peak
memory, seeding would distort the measured quantities, so the run is left unseeded.
**The 5-D / 2-D coordinates and the labels derived from them are therefore
run-specific**; re-running produces statistically similar but not identical output.
HDBSCAN is deterministic given its input, subject to tie-breaking among equal
distances.

## GPU-memory ceiling behaviour

The hard whole-card GPU-memory limit is **21 GiB (21,504 MiB), including all other
GPU use**. Whole-card usage is sampled every 50 ms via **NVML** (all processes on the
device), alongside this process's host RSS. We **measure**; we do not probe or
maximise usage, and we do **not** add any memory cap or change any analytical setting
to reduce memory. The RTX 3090 has 24 GiB physical, giving headroom to *observe* a
breach without necessarily triggering an out-of-memory error.

If a fit **raises** (e.g. CUDA OOM) **or** its measured whole-card peak **exceeds the
ceiling**, the run **stops safely**: it saves whatever evidence exists for the
completed part, writes a `STOPPED_FAILED_OR_BREACHED` summary + verification, and does
**not** start the next GPU stage — rather than substituting another method or dataset.

## Environment

The **same** isolated `uv` venv built for the three-day feasibility run
(`~/venvs/rapids-umap-hdbscan`, Python 3.12), reused as-is so the software stack is
identical to the validated run. **The Phase 2 (`socialmediadgg-phase2`) and FAISS
(`faiss-bench`) venvs are not touched.** Runs in WSL2 Ubuntu-24.04. Exact versions in
`environment/packages.txt` / `environment/cuml_env.json`. Key versions:

- `cuml-cu12 26.08.00`, `cupy-cuda12x 14.2.0` (CUDA runtime 12.9), NVIDIA driver 596.49 (CUDA 13.2)
- `hdbscan 0.8.44` (CPU package, for tree `.to_pandas()` export only)
- `numpy 2.4.6`, `pyarrow 23.0.1`, `pandas 3.0.3`, `matplotlib 3.11.1`, `nvidia-ml-py 13.610.43`
- GPU: NVIDIA GeForce RTX 3090, 24,576 MiB, compute capability 8.6

## Reproduce

From WSL2 Ubuntu-24.04:

```bash
BASE="pipeline/phase3_blind_semantic_structure_discovery/full_corpus_gpu_umap_hdbscan_baseline"

# 1. (re-)capture the environment into environment/ (reuses the existing venv)
bash "$BASE/scripts/setup_env.sh"

# 2. run the full-corpus baseline (writes results/ and figures/)
~/venvs/rapids-umap-hdbscan/bin/python "$BASE/scripts/run_full_corpus_umap_hdbscan.py"

# 3. write provenance sidecars for the summary tables/figures
~/venvs/rapids-umap-hdbscan/bin/python "$BASE/scripts/write_extra_manifests.py"
```

Useful flags on `run_full_corpus_umap_hdbscan.py`: `--quick` (small-subset smoke
test), `--n-rows N` (subset), `--ceiling-mib` (default 21504), `--reuse` (load a
stage's saved `.npy` and skip recompute — recovery/resume; the canonical run does not
use it), `--figures-only` (rebuild figures from saved results). Unit tests (no GPU):
`~/venvs/rapids-umap-hdbscan/bin/python tests/test_run_full_corpus_functions.py`.

## Outputs

`results/`

| File | Contents |
|---|---|
| `umap_5d_coordinates.parquet` / `.npy` | keyed 5-D analytical coordinates (`row, doc_key, record_type, umap_0..umap_4`) + raw matrix |
| `umap_2d_visualization_coordinates.parquet` / `.npy` | keyed 2-D visualisation coordinates (`umap_0, umap_1`) + raw matrix (viz only) |
| `hdbscan_labels.parquet` | `row, doc_key, record_type, label, probability` — raw labels (incl. -1) + membership |
| `hdbscan_cluster_persistence.csv` | per-cluster `cluster_label, size, persistence` |
| `hdbscan_condensed_tree.parquet` | condensed tree (`parent, child, lambda_val, child_size`) |
| `hdbscan_single_linkage_tree.parquet` | single-linkage tree |
| `hdbscan_minimum_spanning_tree.parquet` | MST (`from, to, distance`) |
| `summary.json` | full config, measurements, counts, cluster stats, tie stats, environment, verification |
| `verification.json` | the completion-checklist verification results |
| `measurements.csv` | runtime + memory per operation |
| `gpu_memory_samples_{umap_5d,hdbscan,umap_2d_viz}.csv` | NVML whole-card + host-RSS time series |
| `_cluster_sizes.txt` | internal helper (per-cluster sizes; supports `--figures-only`) |

`figures/` — nine diagnostic PNGs (titles + axis/tick labels only; clusters not
named or interpreted):

- `fig_umap2d_global_density.png` — global 2-D UMAP density of all documents (hexbin, log count)
- `fig_umap2d_assigned_vs_noise.png` — global 2-D UMAP, assigned vs label -1 (all points, rasterized)
- `fig_cluster_size_distribution.png` — cluster-size histogram
- `fig_ranked_cluster_sizes.png` — ranked (sorted) cluster sizes
- `fig_assigned_vs_noise.png` — assigned vs noise counts
- `fig_membership_probability_distribution.png` — membership-probability histogram
- `fig_cluster_persistence_distribution.png` — cluster-persistence histogram
- `fig_runtime_by_operation.png` — wall-clock runtime per operation
- `fig_peak_gpu_memory_by_operation.png` — peak whole-card GPU memory per operation

`environment/` — versions, `nvidia-smi`, setup log. Every non-code output carries a
`<file>.manifest.yml` provenance sidecar (input hashes, output hash,
producing-script hash, parameters, package versions, git commit).

## Verification (completion checklist)

`results/verification.json` records: input rows = 902,144; 5-D UMAP rows = 902,144;
2-D visualisation UMAP rows = 902,144; HDBSCAN output rows = 902,144; row/doc_key
alignment preserved; no duplicate or missing doc_keys introduced; all saved UMAP
coordinates finite; every document has exactly one raw HDBSCAN label; source
embeddings and index unchanged (hash before == hash after).

## Scope

Baseline structure-discovery run and measurement only. Raw HDBSCAN labels (including
-1) are preserved exactly. No cluster is named, described, ranked, characterised, or
interpreted; no analytical setting is tuned; no record is removed or collapsed. See
`REPORT.md` for the measured results.
