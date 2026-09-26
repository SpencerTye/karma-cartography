# =====================================================================
# Phase 4C manual-QC sample selection (WORKLOAD PLANNING ONLY)
#
# Purpose: select the 10 largest + 15 random assigned Phase 3 clusters
# from the completed Phase 4A cluster inventory and report the exact
# document workload of that 25-cluster manual-QC sample. No manual QC,
# no re-run of 4A/4B, no content inspection, no new metrics.
#
# Inputs  (read-only):
#   4a_cluster_description_foundation/derived/cluster_inventory.parquet
# Outputs:
#   4c_manual_qc/planning/selected_clusters.csv
#   4c_manual_qc/planning/WORKLOAD.md
# =====================================================================

suppressWarnings(suppressMessages({
  library(nanoparquet)
  library(digest)
  library(readr)
}))

script_path <- file.path(Sys.getenv("KC_REPO_ROOT", unset = getwd()), "pipeline/phase4_semantic_structure_description_qc/4c_manual_qc/scripts/select_phase4c_qc_clusters.R")
inv_path <- file.path(Sys.getenv("KC_REPO_ROOT", unset = getwd()), "pipeline/phase4_semantic_structure_description_qc/4a_cluster_description_foundation/derived/cluster_inventory.parquet")
out_dir  <- file.path(Sys.getenv("KC_REPO_ROOT", unset = getwd()), "pipeline/phase4_semantic_structure_description_qc/4c_manual_qc")
plan_dir <- file.path(out_dir, "planning")
dir.create(plan_dir, recursive = TRUE, showWarnings = FALSE)

SEED <- 20260912L

# ---- Load inventory (read-only) -------------------------------------
inv <- as.data.frame(nanoparquet::read_parquet(inv_path), stringsAsFactors = FALSE)

# ---- Integrity checks (fail loudly, do not silently proceed) --------
stopifnot(nrow(inv) == 5167L)
stopifnot(!any(duplicated(inv$cluster_id)))
stopifnot(all(c("cluster_id", "n_docs", "n_comments", "n_submissions",
                "prob_mean", "prob_median") %in% names(inv)))
stopifnot(all(inv$n_docs == inv$n_comments + inv$n_submissions))
stopifnot(all(inv$n_docs > 0L))

# ---- 10 largest: n_docs DESC, ties broken by cluster_id ASC ---------
ord_largest <- order(-inv$n_docs, inv$cluster_id)
largest_idx <- ord_largest[1:10]
largest_ids <- inv$cluster_id[largest_idx]

# boundary transparency: 10th vs 11th document count
docs_sorted <- inv$n_docs[ord_largest]
boundary_10 <- docs_sorted[10]
boundary_11 <- docs_sorted[11]
tie_at_boundary <- (boundary_10 == boundary_11)

# ---- Eligible random pool = all clusters MINUS the 10 largest -------
# Canonical, fully specified pool order: cluster_id ascending.
# (sample() output depends on input-vector order, so we fix it.)
eligible_ids <- sort(inv$cluster_id[!(inv$cluster_id %in% largest_ids)])
stopifnot(length(eligible_ids) == 5167L - 10L)

# ---- 15 random: simple random sampling WITHOUT replacement ----------
set.seed(SEED)
rng_kind <- RNGkind()
random_ids <- sample(eligible_ids, size = 15L, replace = FALSE)

stopifnot(length(intersect(largest_ids, random_ids)) == 0L)
stopifnot(!any(duplicated(random_ids)))

# ---- Assemble 25-cluster selection table ----------------------------
# largest rows kept in rank order (n_docs desc); random rows cluster_id asc.
cols <- c("cluster_id", "selection_group", "n_docs", "n_comments",
          "n_submissions", "prob_mean", "prob_median")

sel_largest <- inv[match(largest_ids, inv$cluster_id), ]
sel_largest$selection_group <- "largest"

random_ids_sorted <- sort(random_ids)
sel_random <- inv[match(random_ids_sorted, inv$cluster_id), ]
sel_random$selection_group <- "random"

selected <- rbind(sel_largest[cols], sel_random[cols])
rownames(selected) <- NULL

# ---- Group / combined summaries -------------------------------------
grp <- function(d) list(
  n          = nrow(d),
  total_docs = sum(d$n_docs),
  total_com  = sum(d$n_comments),
  total_sub  = sum(d$n_submissions),
  min        = min(d$n_docs),
  median     = median(d$n_docs),
  max        = max(d$n_docs)
)
s_l <- grp(sel_largest)
s_r <- grp(sel_random)
s_a <- grp(selected)

# ---- Write CSV deliverable ------------------------------------------
csv_path <- file.path(plan_dir, "selected_clusters.csv")
readr::write_csv(selected, csv_path)

# ---- Provenance -----------------------------------------------------
input_sha  <- digest::digest(file = inv_path,    algo = "sha256")
csv_sha    <- digest::digest(file = csv_path,     algo = "sha256")
script_sha <- digest::digest(file = script_path,  algo = "sha256")
stamp <- format(as.POSIXct(Sys.time(), tz = "UTC"), "%Y-%m-%dT%H:%M:%SZ")

# ---- Build WORKLOAD.md (generated in R for exact numeric fidelity) --
fmt <- function(x) if (isTRUE(x == round(x))) format(as.integer(round(x)), trim = TRUE) else as.character(x)
id_size_table <- function(ids, sizes) {
  c("| cluster_id | n_docs |",
    "| ---: | ---: |",
    paste0("| ", ids, " | ", sizes, " |"))
}

L <- c(
  "# Phase 4C Manual-QC Sample — Workload",
  "",
  "Planning artifact only. This selects the manual-QC cluster sample and reports the",
  "exact document workload it represents. No manual review, scoring, labeling, or",
  "content inspection was performed. Phase 4A and Phase 4B were neither rerun nor modified.",
  "",
  "## Provenance",
  "",
  paste0("- Input: `4a_cluster_description_foundation/derived/cluster_inventory.parquet`"),
  paste0("  - SHA256: `", input_sha, "`"),
  paste0("  - Rows (assigned clusters): ", nrow(inv)),
  paste0("- Generating script: `4c_manual_qc/scripts/select_phase4c_qc_clusters.R`"),
  paste0("  - SHA256: `", script_sha, "`"),
  paste0("- Output CSV: `4c_manual_qc/planning/selected_clusters.csv`"),
  paste0("  - SHA256: `", csv_sha, "`"),
  paste0("- Environment: ", R.version.string, "; nanoparquet ", packageVersion("nanoparquet")),
  paste0("- RNG: seed = ", SEED, "; RNGkind = ", paste(rng_kind, collapse = " / "),
         " (R >= 3.6.0 default sample() algorithm)"),
  paste0("- Generated (UTC): ", stamp),
  "",
  "## Selection procedure (exact)",
  "",
  "1. Read all 5,167 assigned clusters from `cluster_inventory.parquet` (read-only).",
  "2. **10 largest**: order clusters by document count (`n_docs`) descending; break any",
  "   document-count ties by `cluster_id` ascending; take the first 10.",
  "3. **Eligible pool**: all remaining 5,157 clusters (the 10 largest excluded). No",
  "   stratification, weighting, or filtering. Pool ordered by `cluster_id` ascending so",
  "   the draw is fully reproducible (`sample()` depends on input-vector order).",
  paste0("4. **15 random**: `set.seed(", SEED, ")` then simple random sampling WITHOUT",
         " replacement of 15 cluster_ids from the eligible pool."),
  "5. Report per-cluster and aggregate document workload. `cluster size` = `n_docs`.",
  "",
  paste0("Top-10 boundary: 10th-ranked cluster has n_docs = ", boundary_10,
         "; 11th-ranked has n_docs = ", boundary_11, ". ",
         if (tie_at_boundary) "A document-count tie occurred at the boundary and was resolved by ascending cluster_id."
         else "No document-count tie at the top-10 boundary."),
  "",
  "## 10 largest clusters (by document count)",
  "",
  id_size_table(sel_largest$cluster_id, sel_largest$n_docs),
  "",
  paste0("- Total documents: **", fmt(s_l$total_docs), "**"),
  paste0("- Min / median / max cluster size: ", fmt(s_l$min), " / ", fmt(s_l$median), " / ", fmt(s_l$max)),
  paste0("- Total comments / submissions: ", fmt(s_l$total_com), " / ", fmt(s_l$total_sub)),
  "",
  paste0("## 15 random clusters (seed ", SEED, ", cluster_id ascending)"),
  "",
  id_size_table(sel_random$cluster_id, sel_random$n_docs),
  "",
  paste0("- Total documents: **", fmt(s_r$total_docs), "**"),
  paste0("- Min / median / max cluster size: ", fmt(s_r$min), " / ", fmt(s_r$median), " / ", fmt(s_r$max)),
  paste0("- Total comments / submissions: ", fmt(s_r$total_com), " / ", fmt(s_r$total_sub)),
  "",
  "## Combined 25-cluster manual-QC workload",
  "",
  paste0("- **Total documents (messages) to inspect if every doc in all 25 clusters is reviewed: ",
         fmt(s_a$total_docs), "**"),
  paste0("- Total comments: ", fmt(s_a$total_com)),
  paste0("- Total submissions: ", fmt(s_a$total_sub)),
  paste0("- Min / median / max cluster size: ", fmt(s_a$min), " / ", fmt(s_a$median), " / ", fmt(s_a$max)),
  paste0("- Largest group contributes ", fmt(s_l$total_docs), " docs; random group contributes ",
         fmt(s_r$total_docs), " docs."),
  "",
  "Per-cluster measurements (document/comment/submission counts and HDBSCAN",
  "membership-probability mean & median) are in `planning/selected_clusters.csv`.",
  ""
)

md_path <- file.path(plan_dir, "WORKLOAD.md")
writeLines(L, md_path, sep = "\n")

# ---- Console echo (summary printed to the console) -----------------------
cat("\n================ SELECTED CLUSTERS (25) ================\n")
print(selected, row.names = FALSE)

cat("\n================ GROUP SUMMARIES ================\n")
cat(sprintf("10 LARGEST : total_docs=%d  min=%s  median=%s  max=%s  (com=%d sub=%d)\n",
            s_l$total_docs, fmt(s_l$min), fmt(s_l$median), fmt(s_l$max), s_l$total_com, s_l$total_sub))
cat(sprintf("15 RANDOM  : total_docs=%d  min=%s  median=%s  max=%s  (com=%d sub=%d)\n",
            s_r$total_docs, fmt(s_r$min), fmt(s_r$median), fmt(s_r$max), s_r$total_com, s_r$total_sub))
cat(sprintf("ALL 25     : total_docs=%d  total_com=%d  total_sub=%d  min=%s  median=%s  max=%s\n",
            s_a$total_docs, s_a$total_com, s_a$total_sub, fmt(s_a$min), fmt(s_a$median), fmt(s_a$max)))

cat("\nLargest cluster_ids (rank order): ", paste(sel_largest$cluster_id, collapse = ", "), "\n")
cat("Random  cluster_ids (asc order) : ", paste(sel_random$cluster_id,  collapse = ", "), "\n")
cat(sprintf("\nTop-10 boundary: 10th n_docs=%d, 11th n_docs=%d, tie=%s\n",
            boundary_10, boundary_11, tie_at_boundary))

cat("\n================ PROVENANCE ================\n")
cat("input  sha256:", input_sha, "\n")
cat("csv    sha256:", csv_sha, "\n")
cat("script sha256:", script_sha, "\n")
cat("seed:", SEED, " RNGkind:", paste(rng_kind, collapse = " / "), "\n")
cat("wrote:", csv_path, "\n")
cat("wrote:", md_path, "\n")
cat("DONE\n")
