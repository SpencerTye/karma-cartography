# Phase 3: structure discovery

GPU UMAP (1,024 → 5 dimensions, cosine) and HDBSCAN with RAPIDS cuML, plus a 2-D UMAP for
visualisation only. Clusters are measured here, not named or interpreted.

| Folder | Purpose |
|---|---|
| `three_day_gpu_umap_hdbscan_feasibility/` | Trial on three days (101,722 messages): runtime, memory, outputs |
| `three_day_sensitivity_stability/` | How stable the clusters are across parameter settings and repeated runs |
| `full_corpus_gpu_umap_hdbscan_baseline/` | The full run: 902,144 messages, 5,167 clusters, about two minutes on the GPU |

Results are summarised in [`docs/METHODS.md`](../../docs/METHODS.md).
