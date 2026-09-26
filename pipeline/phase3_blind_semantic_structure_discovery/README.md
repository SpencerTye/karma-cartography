# Phase 3: clustering

UMAP (1,024 to 5 dimensions, cosine distance) and HDBSCAN on the GPU with RAPIDS cuML. A
separate 2-D UMAP is used only for plots. The clusters are not named or interpreted here.

| Folder | What it does |
|---|---|
| `three_day_gpu_umap_hdbscan_feasibility/` | Trial run on three days of data (101,722 messages) to measure time, memory and outputs |
| `three_day_sensitivity_stability/` | Reruns the three-day data with one UMAP or HDBSCAN setting changed at a time, and repeats the baseline five times, to see how much the clusters change |
| `full_corpus_gpu_umap_hdbscan_baseline/` | The full run: 902,144 messages, 5,167 clusters, about two minutes on the GPU |

Results are in [`docs/METHODS.md`](../../docs/METHODS.md).
