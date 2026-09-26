# Phase 4: cluster description and QC

Links the 5,167 clusters from Phase 3 back to the original messages and describes them.
The clusters are not named yet. All scripts are in R and have no random steps except the
seeded sample in 4c.

| Folder | Script | What it does |
|---|---|---|
| `4a_cluster_description_foundation/` | `run_phase4a_cluster_description.R` | Lists every cluster, computes c-TF-IDF keywords for each, and picks representative messages (nearest the centroid, and highest membership probability) with their parent comments |
| `4b_metadata_and_automated_qc/` | `run_phase4b_metadata_and_qc.R` | Summarizes each cluster's authors, timing, threads, links and reply context, and runs automated quality checks |
| `4c_manual_qc/` | `select_phase4c_qc_clusters.R`, `build_phase4c_review_package.R` | Picks a sample of clusters for manual review (the largest plus a random set, with a fixed seed) and builds a review file for each |

The next step is the manual review of that sample. The inputs are the Phase 1 table, the
Phase 2 embeddings and the Phase 3 labels, and none of them are modified. Run the scripts
from the repository root, for example:

```bash
Rscript pipeline/phase4_semantic_structure_description_qc/4a_cluster_description_foundation/scripts/run_phase4a_cluster_description.R
```
