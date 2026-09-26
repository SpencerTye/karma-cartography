# Three-day clustering trial: GPU UMAP and HDBSCAN

A trial run of UMAP and HDBSCAN with RAPIDS cuML on the three-day embeddings (101,722
documents), to measure time and GPU memory before the full corpus. The settings are the same
as in the [full run](../full_corpus_gpu_umap_hdbscan_baseline/). The clusters are not named
or interpreted.

## Results

| Step | Time | Peak GPU memory (whole card) |
|---|---:|---:|
| UMAP, 1,024 to 5 dimensions | 5.3 s | 2.6 GiB |
| HDBSCAN on the 5-D output | 1.3 s | 2.5 GiB |

UMAP is unseeded, so the result changes a little between runs. Two runs gave 883 and 873
clusters.

## Settings

```
UMAP     n_neighbors=15, n_components=5, min_dist=0.0, metric=cosine
HDBSCAN  min_cluster_size=10, min_samples=10, metric=euclidean, cluster_selection_method=eom
```

Everything else is the cuML default. See the full run's README for why UMAP is left unseeded.

## Running it

Under WSL2 Ubuntu 24.04. `scripts/setup_env.sh` builds the RAPIDS environment
([`environment/requirements-clustering.txt`](../../../environment/requirements-clustering.txt)).

```bash
bash scripts/setup_env.sh
python scripts/run_umap_hdbscan.py
python scripts/write_extra_manifests.py
python tests/test_run_umap_hdbscan_functions.py   # unit test, no GPU needed
```

Options: `--quick`, `--n-rows N`, `--ceiling-mib 21504`, `--figures-only`.

The outputs in `results/` are the keyed 5-D coordinates, HDBSCAN labels and membership, the
cluster trees, measurements, and an estimate for the full corpus. `figures/` holds six
diagnostic plots.
