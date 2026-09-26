# Full-corpus clustering: GPU UMAP and HDBSCAN

Runs UMAP and then HDBSCAN on all 902,144 embeddings with RAPIDS cuML on the RTX 3090, plus
a separate 2-D UMAP for plots. It uses the same code and settings as the
[three-day trial](../three_day_gpu_umap_hdbscan_feasibility/), scaled to the full corpus.
The clusters are not named or interpreted here, and no documents are removed or merged.

## Results (run of 2026-09-11)

| Step | Time | Peak GPU memory (whole card) |
|---|---:|---:|
| UMAP, 1,024 to 5 dimensions | 44.0 s | 4.9 GiB |
| HDBSCAN on the 5-D output | 31.4 s | 3.6 GiB |
| UMAP, 2-D, for plots only | 45.5 s | 4.9 GiB |
| Total | 121.0 s | |

- 5,167 clusters. 457,880 documents (50.8%) are assigned to a cluster and 444,264 (49.2%)
  are labeled noise.
- Cluster sizes: minimum 10, median 26, mean 88.6, maximum 24,862.
- 65,743 documents share an identical embedding with at least one other document (7,050
  groups, the largest 26,681). UMAP still maps them to distinct 5-D points, and no fit
  reported warnings.
- Compared with the three-day trial (101,722 documents), the corpus is 8.9 times larger.
  UMAP took 8.3 times as long and HDBSCAN 23.8 times as long, while HDBSCAN's memory rose
  only 1.1 GiB.
- cuML reports a persistence of exactly 1.0 for every cluster, as it did in the trial. The
  values are saved as returned and are not used.

## Settings

```
UMAP (5-D)   n_neighbors=15, n_components=5, min_dist=0.0, metric=cosine
HDBSCAN      min_cluster_size=10, min_samples=10, metric=euclidean, cluster_selection_method=eom
UMAP (2-D)   n_neighbors=15, n_components=2, min_dist=0.0, metric=cosine
```

Everything else is the cuML default. For more than 50,000 points cuML builds the UMAP
neighbor graph with NN-Descent. UMAP is left unseeded: in cuML a seed makes training slower
and uses more memory, which would distort the time and memory measurements. Rerunning
therefore gives similar but not identical coordinates and labels. HDBSCAN runs with
`gen_min_span_tree=True` only so the minimum spanning tree can be saved; this doesn't change
the labels. Everything is float32.

The GPU memory limit is 21 GiB for the whole card. Memory is sampled every 50 ms with NVML.
If a step fails or goes over the limit, the run saves what it has, marks itself as stopped,
and does not start the next step.

## Checks

`results/verification.json` confirms that all 902,144 rows are present in every output, that
rows still line up with their `doc_key`, that no keys were duplicated or lost, that every
coordinate is finite, that every document has exactly one label, and that the input
embeddings were unchanged (hashed before and after).

## Running it

Under WSL2 Ubuntu 24.04, with the environment from `scripts/setup_env.sh`
([`environment/requirements-clustering.txt`](../../../environment/requirements-clustering.txt)):

```bash
bash scripts/setup_env.sh
python scripts/run_full_corpus_umap_hdbscan.py
python scripts/write_extra_manifests.py
python tests/test_run_full_corpus_functions.py   # unit test, no GPU needed
```

Options: `--quick` (small test run), `--n-rows N`, `--ceiling-mib 21504`, `--reuse` (resume
from a step's saved output), `--figures-only`. `scripts/make_improved_umap_views.py` makes
the zoomed 2-D density plots, including the one in the main README.

## Outputs

In `results/`: the 5-D and 2-D coordinates keyed by `doc_key`, HDBSCAN labels and
membership probabilities, cluster persistence, the condensed tree, the single-linkage tree
and the minimum spanning tree, timing and memory measurements, and the checks. `figures/`
holds nine diagnostic plots. Each output has a `.manifest.yml`.
