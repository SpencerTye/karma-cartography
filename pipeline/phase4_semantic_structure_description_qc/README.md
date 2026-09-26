# Phase 4: describing the clusters and checking them

Reconnects the 5,167 clusters from Phase 3 to the original messages and describes them,
without naming or interpreting them yet. All scripts are R and fully deterministic.

| Folder | Script | What it does |
|---|---|---|
| `4a_cluster_description_foundation/` | `run_phase4a_cluster_description.R` | Cluster inventory; c-TF-IDF keywords per cluster; representative messages (nearest the centroid, and highest membership probability) with their parent comments |
| `4b_metadata_and_automated_qc/` | `run_phase4b_metadata_and_qc.R` | Per-cluster authors, timing, threads, links and reply context; automated quality checks |
| `4c_manual_qc/` | `select_phase4c_qc_clusters.R`, `build_phase4c_review_package.R` | Picks a review sample (largest clusters plus a random set, fixed seed) and builds a review package for each |

Next: the manual review of that sample, then naming and describing clusters.
Inputs are the Phase 1 table, the Phase 2 embeddings and the Phase 3 labels; all are read
only. Run from the repository root, for example
`Rscript pipeline/phase4_semantic_structure_description_qc/4a_cluster_description_foundation/scripts/run_phase4a_cluster_description.R`.
