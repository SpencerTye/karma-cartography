#!/usr/bin/env Rscript
# Phase 4C: build the manual-QC review package for the 25 selected clusters (package construction only).
# Links the clusters to their documents as in Phase 4A (hdbscan_labels -> message_events by doc_key),
# samples 100 documents from each of the 10 largest clusters (seed 20260912), takes the 15 random
# clusters whole, and adds the Phase 4A representatives and c-TF-IDF descriptors.
# Inputs: planning/selected_clusters.csv, Phase 1-3 outputs, Phase 4A derived/ files.
# Outputs: review_package/ (documents, descriptors, overview, blank review form, per-cluster .md,
#   REPORT.md with provenance) and derived/manual_review_sampled_documents.csv. Env: KC_REPO_ROOT.

suppressWarnings(suppressMessages({
  library(nanoparquet)
  library(data.table)
  library(readr)
  library(digest)
}))

t_start <- Sys.time()
options(width = 200, warn = 1)
data.table::setDTthreads(0L)

# ---- paths --------------------------------------------------------------------------
BASE <- Sys.getenv("KC_REPO_ROOT", unset = getwd())
PIPE <- file.path(BASE, "pipeline")
P4   <- file.path(PIPE, "phase4_semantic_structure_description_qc")

P1_ME    <- file.path(PIPE, "phase1_message_event_foundation/derived/message_events.parquet")
P2_IDX   <- file.path(PIPE, "phase2_semantic_embeddings/full_corpus_embedding_0p6b/results/embedding_index.parquet")
P3_LAB   <- file.path(PIPE, "phase3_blind_semantic_structure_discovery/full_corpus_gpu_umap_hdbscan_baseline/results/hdbscan_labels.parquet")
P4A_DER  <- file.path(P4, "4a_cluster_description_foundation/derived")
P4A_REPS <- file.path(P4A_DER, "representative_documents.parquet")
P4A_CT   <- file.path(P4A_DER, "ctfidf_top_terms.parquet")
P4A_INV  <- file.path(P4A_DER, "cluster_inventory.parquet")

OUT_4C   <- file.path(P4, "4c_manual_qc")
SEL_CSV  <- file.path(OUT_4C, "planning/selected_clusters.csv")
SCRIPT   <- file.path(OUT_4C, "scripts/build_phase4c_review_package.R")

RP   <- file.path(OUT_4C, "review_package")
RPC  <- file.path(RP, "clusters")
DER  <- file.path(OUT_4C, "derived")
dir.create(RPC, recursive = TRUE, showWarnings = FALSE)
dir.create(DER, recursive = TRUE, showWarnings = FALSE)

# ---- fixed parameters ---------------------------------------------------------------
SEED     <- 20260912L   # matches the established Phase 4C planning seed
N_SAMPLE <- 100L        # 100-document manual-review depth for the 10 largest clusters

# expected Phase 3 measurements (consistency checks, not re-validation)
EXP_TOTAL    <- 902144L
EXP_ASSIGNED <- 457880L
EXP_NOISE    <- 444264L
EXP_CLUSTERS <- 5167L

# ---- small helpers ------------------------------------------------------------------
STAGE <- data.table(stage = character(0), seconds = numeric(0), peak_gb = numeric(0))
timeit <- function(label, expr) {
  invisible(gc(reset = TRUE))
  t0 <- Sys.time()
  val <- force(expr)
  el <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  g <- gc()
  peak_gb <- (g["Vcells", "max used"] * 8 + g["Ncells", "max used"] * 56) / 2^30
  cat(sprintf("[stage] %-30s %8.1f s   peak ~%.2f GB\n", label, el, peak_gb))
  STAGE <<- rbind(STAGE, data.table(stage = label, seconds = el, peak_gb = peak_gb))
  val
}
# epoch-seconds -> "YYYY-MM-DD HH:MM:SS UTC" (NA-safe), matching the Phase 4A rep format
fmt_utc <- function(x) {
  out <- rep(NA_character_, length(x))
  ok <- !is.na(x)
  out[ok] <- format(as.POSIXct(x[ok], origin = "1970-01-01", tz = "UTC"),
                    "%Y-%m-%d %H:%M:%S UTC")
  out
}
sha256f <- function(p) digest::digest(p, algo = "sha256", file = TRUE)
# shortest backtick fence that cannot collide with backticks inside `s`
fence_for <- function(s) {
  m <- regmatches(s, gregexpr("`+", s))[[1]]
  maxrun <- if (length(m) == 0) 0L else max(nchar(m))
  strrep("`", max(3L, maxrun + 1L))
}
md_text_block <- function(s) {
  if (length(s) != 1L || is.na(s) || s == "") return("`(empty canonical text)`")
  s <- gsub("\r", "", s, fixed = TRUE)
  f <- fence_for(s)
  paste0(f, "\n", s, "\n", f)
}

# =====================================================================================
# SECTION 1. Phase 3 label/doc_key linkage -- reproduce the Phase 4A checks exactly.
# =====================================================================================
lab <- timeit("load labels + index (align check)", {
  l <- as.data.table(nanoparquet::read_parquet(P3_LAB))   # row, doc_key, record_type, label, probability
  i <- as.data.table(nanoparquet::read_parquet(P2_IDX))   # row, doc_key, record_type, created_utc, ...
  stopifnot(nrow(l) == nrow(i),
            all(l$row == i$row),
            all(l$row == 0:(nrow(l) - 1L)),
            all(l$doc_key == i$doc_key),
            all(l$record_type == i$record_type))
  l[, label := as.integer(label)]
  l
})
n_total     <- nrow(lab)
is_assigned <- lab$label != -1L
n_assigned  <- sum(is_assigned)
n_noise     <- sum(!is_assigned)
clusters_present <- sort(unique(lab$label[is_assigned]))
stopifnot(n_total == EXP_TOTAL, n_assigned == EXP_ASSIGNED, n_noise == EXP_NOISE,
          length(clusters_present) == EXP_CLUSTERS,
          identical(clusters_present, 0:(EXP_CLUSTERS - 1L)))
cat(sprintf("[ok] linkage: total=%d assigned=%d noise=%d clusters=%d (Phase 4A checks pass)\n",
            n_total, n_assigned, n_noise, length(clusters_present)))

# =====================================================================================
# SECTION 2. Authoritative 25-cluster selection (read planning CSV; preserve its order).
# =====================================================================================
sel <- as.data.table(readr::read_csv(SEL_CSV, show_col_types = FALSE))
sel[, cluster_id := as.integer(cluster_id)]
stopifnot(nrow(sel) == 25L,
          sum(sel$selection_group == "largest") == 10L,
          sum(sel$selection_group == "random")  == 15L,
          all(sel$cluster_id %in% clusters_present),
          !any(duplicated(sel$cluster_id)))
ids_all     <- sel$cluster_id                                   # file order: largest(size desc) then random(id asc)
largest_ids <- sel[selection_group == "largest", cluster_id]    # size desc, ties cluster_id asc
random_ids  <- sel[selection_group == "random",  cluster_id]    # cluster_id asc
grp_of      <- setNames(sel$selection_group, as.character(sel$cluster_id))
cat(sprintf("[ok] selected 25 clusters: 10 largest + 15 random\n"))

# =====================================================================================
# SECTION 3. Reconnect the 25 clusters to their assigned documents + canonical records.
# =====================================================================================
cdocs <- lab[label %in% ids_all, .(doc_key, cluster_id = label, probability)]
stopifnot(nrow(cdocs) == 68109L)

me <- timeit("load message_events (col_select)", {
  keep <- c("doc_key", "record_type", "author", "created_utc", "subreddit", "thread_id",
            "parent_id", "parent_kind", "parent_in_dataset", "text")
  m <- as.data.table(nanoparquet::read_parquet(P1_ME, col_select = keep))
  setkey(m, doc_key)
  m
})
stopifnot(nrow(me) == EXP_TOTAL, !any(duplicated(me$doc_key)))

cd <- me[cdocs, on = "doc_key"]                                 # one row per cluster doc, metadata attached
stopifnot(nrow(cd) == 68109L, !anyNA(cd$cluster_id), !anyNA(cd$record_type))
cd[, selection_group := unname(grp_of[as.character(cluster_id)])]
# per-cluster membership must equal the Phase 4A inventory sizes
inv_all <- as.data.table(nanoparquet::read_parquet(P4A_INV))
cd_sizes <- cd[, .N, by = cluster_id][order(cluster_id)]
inv_sel  <- inv_all[cluster_id %in% ids_all][order(cluster_id)]
stopifnot(all(cd_sizes$N == inv_sel$n_docs))
cat(sprintf("[ok] reconnected %d docs across 25 clusters; per-cluster sizes match inventory\n", nrow(cd)))

# =====================================================================================
# SECTION 4. Manual-review document selection.
#   largest : simple random sampling without replacement of 100 docs per cluster.
#   random  : the entire cluster (no additional sampling).
# Determinism: pin RNG kind + seed once; process the 10 largest clusters in the planning
# CSV order (n_docs desc, ties cluster_id asc); within each cluster sample positions over
# the doc_key-ascending document order via sample.int(n, 100).
# =====================================================================================
RNGkind("Mersenne-Twister", "Inversion", "Rejection")
rng_kind <- RNGkind()
set.seed(SEED)
sampled <- list()
for (cid in largest_ids) {
  keys <- sort(cd[cluster_id == cid, doc_key])                  # stable ascending order
  stopifnot(length(keys) >= N_SAMPLE, !any(duplicated(keys)))
  pos  <- sample.int(length(keys), N_SAMPLE)                    # SRSWOR of 100 positions
  sampled[[as.character(cid)]] <- keys[pos]
}
sampled_keys <- unlist(sampled, use.names = FALSE)
stopifnot(length(sampled_keys) == 1000L, !any(duplicated(sampled_keys)))
sample_hash <- digest::digest(paste(sort(sampled_keys), collapse = ","), algo = "sha256")

cd[, in_manual_random_sample := (selection_group == "largest") & (doc_key %chin% sampled_keys)]
cd[, in_full_cluster_review  := (selection_group == "random")]
stopifnot(sum(cd$in_manual_random_sample) == 1000L, sum(cd$in_full_cluster_review) == 647L)
cat(sprintf("[ok] manual-review set: 1000 sampled (10x100) + 647 full-cluster = %d; sample_hash=%s\n",
            sum(cd$in_manual_random_sample) + sum(cd$in_full_cluster_review), substr(sample_hash, 1, 16)))

# =====================================================================================
# SECTION 5. Existing Phase 4A representatives (5 centroid_nearest + 5 highest_membership).
#   Overlap (a doc chosen by both methods, or also in the manual set) is kept as flags
#   on a single row.
# =====================================================================================
reps <- as.data.table(nanoparquet::read_parquet(P4A_REPS))
reps <- reps[cluster_id %in% ids_all]
stopifnot(uniqueN(reps$cluster_id) == 25L)
cn <- reps[method == "centroid_nearest",
           .(cluster_id, doc_key, centroid_nearest_rank = rank, centroid_nearest_cosine = cosine_to_centroid)]
hm <- reps[method == "highest_membership",
           .(cluster_id, doc_key, highest_membership_rank = rank)]
stopifnot(!any(duplicated(cn[, .(cluster_id, doc_key)])),
          !any(duplicated(hm[, .(cluster_id, doc_key)])),
          nrow(cn) == 125L, nrow(hm) == 125L)          # 25 clusters x 5

cd[, c("centroid_nearest_rank", "centroid_nearest_cosine", "highest_membership_rank") :=
     .(NA_integer_, NA_real_, NA_integer_)]
cd[cn, on = .(cluster_id, doc_key),
   c("centroid_nearest_rank", "centroid_nearest_cosine") := .(i.centroid_nearest_rank, i.centroid_nearest_cosine)]
cd[hm, on = .(cluster_id, doc_key), highest_membership_rank := i.highest_membership_rank]
cd[, is_centroid_nearest   := !is.na(centroid_nearest_rank)]
cd[, is_highest_membership := !is.na(highest_membership_rank)]
stopifnot(sum(cd$is_centroid_nearest) == 125L, sum(cd$is_highest_membership) == 125L)

# sanity: membership probability stored in 4A reps must equal the labels probability
repchk <- unique(reps[, .(cluster_id, doc_key, membership_probability)])
m2 <- merge(repchk, cd[, .(cluster_id, doc_key, probability)], by = c("cluster_id", "doc_key"))
stopifnot(nrow(m2) == nrow(repchk), max(abs(m2$membership_probability - m2$probability)) < 1e-9)
cat("[ok] representatives folded in; membership probabilities reconcile with labels\n")

# =====================================================================================
# SECTION 6. Final package = every document carrying any evidence flag (one row each).
# =====================================================================================
cd[, in_package := in_manual_random_sample | in_full_cluster_review |
     is_centroid_nearest | is_highest_membership]
pkg <- cd[in_package == TRUE]
pkg[, manual_review_inclusion_type :=
      fifelse(in_manual_random_sample, "random_sample",
      fifelse(in_full_cluster_review,  "full_cluster", "representative_only"))]

# ---- parent context for package comments (Phase 4A method: match parent_id -> me$doc_key)
is_com <- pkg$record_type == "comment"
plk <- fifelse(is_com, pkg$parent_id, NA_character_)
mi  <- match(plk, me$doc_key)                                   # NA when parent not in-dataset
pkg[, parent_doc_key     := me$doc_key[mi]]
pkg[, parent_record_type := me$record_type[mi]]
pkg[, parent_author      := me$author[mi]]
pkg[, parent_created_utc := me$created_utc[mi]]
pkg[, parent_text        := me$text[mi]]
pkg[, created_datetime_utc        := fmt_utc(created_utc)]
pkg[, parent_created_datetime_utc := fmt_utc(parent_created_utc)]

# =====================================================================================
# SECTION 7. review_documents (machine-readable): one row per unique (cluster, doc_key).
# =====================================================================================
rd <- pkg[, .(
  cluster_id, selection_group, doc_key, record_type, text, author,
  created_utc, created_datetime_utc, subreddit, thread_id,
  membership_probability = probability,
  manual_review_inclusion_type,
  in_manual_random_sample, in_full_cluster_review,
  is_centroid_nearest, centroid_nearest_rank, centroid_nearest_cosine,
  is_highest_membership, highest_membership_rank,
  parent_id, parent_kind, parent_in_dataset,
  parent_doc_key, parent_record_type, parent_author,
  parent_created_utc, parent_created_datetime_utc, parent_text
)]
setorder(rd, cluster_id, doc_key)
stopifnot(!any(duplicated(rd[, .(cluster_id, doc_key)])))
n_pkg_unique <- nrow(rd)

nanoparquet::write_parquet(rd, file.path(RP, "review_documents.parquet"),
                           compression = "zstd", options = parquet_options(compression_level = 3L))
readr::write_csv(rd, file.path(RP, "review_documents.csv"), na = "")
cat(sprintf("[ok] review_documents: %d unique documents written (parquet + csv)\n", n_pkg_unique))

# =====================================================================================
# SECTION 8. Descriptor table (existing Phase 4A c-TF-IDF top terms for the 25 clusters).
# =====================================================================================
ct  <- as.data.table(nanoparquet::read_parquet(P4A_CT))
ctS <- merge(ct[cluster_id %in% ids_all], sel[, .(cluster_id, selection_group)], by = "cluster_id")
ctS[, ngram_type := fifelse(ngram == 2L, "bigram", "unigram")]
setorder(ctS, cluster_id, rank)
desc_out <- ctS[, .(cluster_id, selection_group, rank, term, ngram, ngram_type, ctfidf, term_count)]
readr::write_csv(desc_out, file.path(RP, "review_cluster_descriptors.csv"))

ndesc <- merge(data.table(cluster_id = ids_all),
               ctS[, .(n_descriptors_available = .N), by = cluster_id],
               by = "cluster_id", all.x = TRUE)
ndesc[is.na(n_descriptors_available), n_descriptors_available := 0L]
n_desc_eq10 <- sum(ndesc$n_descriptors_available == 10L)
n_desc_1to9 <- sum(ndesc$n_descriptors_available >= 1L & ndesc$n_descriptors_available <= 9L)
n_desc_0    <- sum(ndesc$n_descriptors_available == 0L)
cat(sprintf("[ok] descriptors: %d rows; clusters with 10=%d, 1-9=%d, 0=%d\n",
            nrow(desc_out), n_desc_eq10, n_desc_1to9, n_desc_0))

# =====================================================================================
# SECTION 9. Cluster overview (one row per selected cluster; factual counts only).
# =====================================================================================
agg <- cd[, .(
  manual_review_n_documents       = sum(in_manual_random_sample | in_full_cluster_review),
  n_centroid_nearest              = sum(is_centroid_nearest),
  n_highest_membership            = sum(is_highest_membership),
  n_distinct_representative_docs   = sum(is_centroid_nearest | is_highest_membership),
  n_reps_both_methods             = sum(is_centroid_nearest & is_highest_membership),
  n_reps_in_manual_set            = sum((is_centroid_nearest | is_highest_membership) &
                                        (in_manual_random_sample | in_full_cluster_review)),
  n_unique_review_package_documents = sum(in_package)
), by = cluster_id]

ov <- Reduce(function(a, b) merge(a, b, by = "cluster_id", all.x = TRUE),
             list(sel[, .(cluster_id, selection_group)],
                  inv_sel[, .(cluster_id, cluster_n_documents = n_docs)],
                  agg, ndesc))
setcolorder(ov, c("cluster_id", "selection_group", "cluster_n_documents",
                  "manual_review_n_documents", "n_centroid_nearest", "n_highest_membership",
                  "n_unique_review_package_documents", "n_descriptors_available",
                  "n_distinct_representative_docs", "n_reps_both_methods", "n_reps_in_manual_set"))
ov <- ov[match(ids_all, cluster_id)]                            # planning display order
readr::write_csv(ov, file.path(RP, "review_cluster_overview.csv"))
cat("[ok] review_cluster_overview written\n")

# =====================================================================================
# SECTION 10. Blank human-review form (one row per cluster; judgment fields left blank).
# =====================================================================================
form <- ov[, .(cluster_id, selection_group, cluster_n_documents, manual_review_n_documents)]
form[, descriptor_correspondence := ""]
form[, internal_variation_notes  := ""]
form[, exceptions_notes          := ""]
form[, reviewer_notes            := ""]
readr::write_csv(form, file.path(RP, "manual_qc_review_form.csv"))
cat("[ok] manual_qc_review_form written (human-judgment fields blank)\n")

# =====================================================================================
# SECTION 11. Minimal derived artifact: the 1,647 manual-review document keys.
# =====================================================================================
msamp <- cd[in_manual_random_sample | in_full_cluster_review,
            .(cluster_id, selection_group, doc_key,
              manual_review_inclusion_type = fifelse(in_manual_random_sample, "random_sample", "full_cluster"))]
setorder(msamp, cluster_id, doc_key)
stopifnot(nrow(msamp) == 1647L)
readr::write_csv(msamp, file.path(DER, "manual_review_sampled_documents.csv"))

# =====================================================================================
# SECTION 12. Human-readable per-cluster review files.
#   Order: cluster summary -> c-TF-IDF descriptors -> centroid_nearest reps ->
#          highest_membership reps -> manual-review documents. Full text preserved.
# =====================================================================================
render_doc <- function(r, n) {
  flags <- character(0)
  if (isTRUE(r$is_centroid_nearest))   flags <- c(flags, sprintf("centroid_nearest #%d", r$centroid_nearest_rank))
  if (isTRUE(r$is_highest_membership)) flags <- c(flags, sprintf("highest_membership #%d", r$highest_membership_rank))
  flagstr <- if (length(flags)) paste(flags, collapse = "; ") else "none"
  head <- sprintf("#### %d. `%s`  (%s)", n, r$doc_key, r$record_type)
  meta <- sprintf(paste0("- author: `%s` | time: %s | subreddit: r/%s | thread: `%s`\n",
                         "- membership_probability: %.6f | inclusion: %s | representative: %s"),
                  r$author, r$created_datetime_utc, r$subreddit, r$thread_id,
                  r$membership_probability, r$manual_review_inclusion_type, flagstr)
  if (isTRUE(r$is_centroid_nearest) && !is.na(r$centroid_nearest_cosine))
    meta <- paste0(meta, sprintf("\n- centroid_nearest cosine: %.6f", r$centroid_nearest_cosine))
  body <- c(head, meta, "", "Text:", md_text_block(r$text))
  if (identical(r$record_type, "comment")) {
    if (!is.na(r$parent_doc_key)) {
      body <- c(body, "",
                sprintf("Parent context (parent_kind=`%s`, parent_in_dataset=%s):",
                        ifelse(is.na(r$parent_kind), "NA", r$parent_kind), r$parent_in_dataset),
                sprintf("- parent doc_key: `%s` | type: %s | author: `%s` | time: %s",
                        r$parent_doc_key, r$parent_record_type, r$parent_author, r$parent_created_datetime_utc),
                "Parent text:", md_text_block(r$parent_text))
    } else {
      body <- c(body, "",
                sprintf("Parent context: parent_id=`%s`, parent_kind=`%s`, parent_in_dataset=%s -- parent not resolvable in the canonical dataset (not inferred).",
                        ifelse(is.na(r$parent_id), "NA", r$parent_id),
                        ifelse(is.na(r$parent_kind), "NA", r$parent_kind),
                        ifelse(is.na(r$parent_in_dataset), "NA", r$parent_in_dataset)))
    }
  }
  c(body, "", "---", "")
}

write_cluster_md <- function(cid) {
  grp   <- grp_of[[as.character(cid)]]
  orow  <- ov[cluster_id == cid]
  drows <- desc_out[cluster_id == cid][order(rank)]
  rdc   <- rd[cluster_id == cid]
  fn <- file.path(RPC, sprintf("cluster_%04d_%s.md", cid, grp))

  L <- c(
    sprintf("# Cluster %d (%s)", cid, grp),
    "",
    "## Cluster summary",
    "",
    sprintf("- cluster_id: %d", cid),
    sprintf("- selection_group: %s", grp),
    sprintf("- full cluster size (documents): %d", orow$cluster_n_documents),
    sprintf("- manual-review documents: %d%s", orow$manual_review_n_documents,
            if (grp == "largest") " (100 simple-random-sampled, without replacement)"
            else " (ENTIRE cluster — every assigned document)"),
    sprintf("- centroid_nearest representatives: %d", orow$n_centroid_nearest),
    sprintf("- highest_membership representatives: %d", orow$n_highest_membership),
    sprintf("- distinct representative documents (after method overlap): %d", orow$n_distinct_representative_docs),
    sprintf("- unique documents in this cluster's review package: %d", orow$n_unique_review_package_documents),
    sprintf("- c-TF-IDF descriptors available: %d (of up to 10)", orow$n_descriptors_available),
    "",
    "Review questions (for the human reviewer; not answered here):",
    "1. Do the reviewed messages substantively correspond to the c-TF-IDF descriptors below?",
    "2. What internal variation or exceptions are visible within the cluster?",
    "",
    "## c-TF-IDF descriptors (existing Phase 4A; top 10 where available)",
    ""
  )
  if (nrow(drows) == 0L) {
    L <- c(L, "_No c-TF-IDF descriptors exist for this cluster (preserved as measured)._", "")
  } else {
    L <- c(L,
           "| rank | term | type | c-TF-IDF | term_count |",
           "| ---: | --- | --- | ---: | ---: |",
           drows[, sprintf("| %d | %s | %s | %.6f | %d |",
                           rank, gsub("|", "\\|", term, fixed = TRUE), ngram_type, ctfidf, term_count)],
           "")
  }

  cn_rows <- rdc[is_centroid_nearest == TRUE][order(centroid_nearest_rank)]
  hm_rows <- rdc[is_highest_membership == TRUE][order(highest_membership_rank)]
  if (grp == "largest") {
    man_rows <- rdc[in_manual_random_sample == TRUE][order(doc_key)]
    man_hdr  <- sprintf("## Manual-review documents — 100 random sample (of %d)", orow$cluster_n_documents)
  } else {
    man_rows <- rdc[in_full_cluster_review == TRUE][order(doc_key)]
    man_hdr  <- sprintf("## Manual-review documents — ENTIRE cluster (%d documents)", orow$cluster_n_documents)
  }

  L <- c(L, "## Centroid-nearest representatives (existing Phase 4A)", "")
  for (k in seq_len(nrow(cn_rows))) L <- c(L, render_doc(as.list(cn_rows[k]), k))
  L <- c(L, "## Highest-membership representatives (existing Phase 4A)", "")
  for (k in seq_len(nrow(hm_rows))) L <- c(L, render_doc(as.list(hm_rows[k]), k))
  L <- c(L, man_hdr,
         "_A document flagged as a representative above may also appear here; overlap is preserved, not deduplicated._",
         "")
  for (k in seq_len(nrow(man_rows))) L <- c(L, render_doc(as.list(man_rows[k]), k))

  writeLines(L, fn, sep = "\n")
  fn
}
md_files <- vapply(ids_all, write_cluster_md, character(1))
cat(sprintf("[ok] wrote %d per-cluster review files under review_package/clusters/\n", length(md_files)))

# =====================================================================================
# SECTION 13. Aggregate counts for REPORT.md.
# =====================================================================================
n_manual_largest <- sum(cd$in_manual_random_sample)                  # 1000
n_manual_random  <- sum(cd$in_full_cluster_review)                   # 647
n_manual_total   <- n_manual_largest + n_manual_random               # 1647
n_rep_rows       <- nrow(reps)                                       # 250 (25 x 10)
n_rep_distinct   <- sum(cd$is_centroid_nearest | cd$is_highest_membership)
n_rep_both       <- sum(cd$is_centroid_nearest & cd$is_highest_membership)
n_rep_in_manual  <- sum((cd$is_centroid_nearest | cd$is_highest_membership) &
                        (cd$in_manual_random_sample | cd$in_full_cluster_review))
n_rep_in_manual_largest <- sum((cd$is_centroid_nearest | cd$is_highest_membership) &
                               cd$in_manual_random_sample)
n_rep_in_manual_random  <- sum((cd$is_centroid_nearest | cd$is_highest_membership) &
                               cd$in_full_cluster_review)
n_rep_context_only <- sum(pkg$manual_review_inclusion_type == "representative_only")

com_rows     <- rd[record_type == "comment"]
n_com        <- nrow(com_rows)
n_com_parent <- sum(!is.na(com_rows$parent_doc_key))
n_sub        <- sum(rd$record_type == "submission")

elapsed <- as.numeric(difftime(Sys.time(), t_start, units = "secs"))
g <- gc()
peak_gb <- (g["Vcells", "max used"] * 8 + g["Ncells", "max used"] * 56) / 2^30

out_files <- c(
  "review_package/review_documents.parquet"        = file.path(RP, "review_documents.parquet"),
  "review_package/review_documents.csv"            = file.path(RP, "review_documents.csv"),
  "review_package/review_cluster_descriptors.csv"  = file.path(RP, "review_cluster_descriptors.csv"),
  "review_package/review_cluster_overview.csv"     = file.path(RP, "review_cluster_overview.csv"),
  "review_package/manual_qc_review_form.csv"       = file.path(RP, "manual_qc_review_form.csv"),
  "derived/manual_review_sampled_documents.csv"    = file.path(DER, "manual_review_sampled_documents.csv")
)

# =====================================================================================
# SECTION 14. REPORT.md (provenance folded in; no manifest sidecar by design).
# =====================================================================================
stamp <- format(as.POSIXct(Sys.time(), tz = "UTC"), "%Y-%m-%dT%H:%M:%SZ")
ovstr <- function() {
  hdr <- c("| cluster_id | group | full_size | manual_review | cn | hm | pkg_unique | descriptors | distinct_reps | reps_both | reps_in_manual |",
           "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
  rows <- ov[, sprintf("| %d | %s | %d | %d | %d | %d | %d | %d | %d | %d | %d |",
                       cluster_id, selection_group, cluster_n_documents, manual_review_n_documents,
                       n_centroid_nearest, n_highest_membership, n_unique_review_package_documents,
                       n_descriptors_available, n_distinct_representative_docs,
                       n_reps_both_methods, n_reps_in_manual_set)]
  c(hdr, rows)
}
R <- c(
  "# Phase 4C Manual-QC Review Package — REPORT",
  "",
  "Construction report for the human-review package covering the 25 pre-selected semantic",
  "clusters. This script only assembled evidence: it performed no manual QC, assigned no",
  "labels, computed no new metrics, and interpreted no cluster. Phases 4A/4B and the 4C",
  "cluster selection were neither rerun nor modified.",
  "",
  "## Inputs (read-only; SHA256)",
  "",
  sprintf("- `planning/selected_clusters.csv`  \n  `%s`", sha256f(SEL_CSV)),
  sprintf("- `phase3 .../hdbscan_labels.parquet`  \n  `%s`", sha256f(P3_LAB)),
  sprintf("- `phase2 .../embedding_index.parquet`  \n  `%s`", sha256f(P2_IDX)),
  sprintf("- `phase1 .../message_events.parquet`  \n  `%s`", sha256f(P1_ME)),
  sprintf("- `4a .../representative_documents.parquet`  \n  `%s`", sha256f(P4A_REPS)),
  sprintf("- `4a .../ctfidf_top_terms.parquet`  \n  `%s`", sha256f(P4A_CT)),
  sprintf("- `4a .../cluster_inventory.parquet`  \n  `%s`", sha256f(P4A_INV)),
  sprintf("- generating script `scripts/build_phase4c_review_package.R`  \n  `%s`", sha256f(SCRIPT)),
  "",
  "## Linkage (identical to Phase 4A)",
  "",
  "Cluster membership comes from `hdbscan_labels.parquet` (`cluster_id == label`, assigned =",
  "`label != -1`), row-aligned against `embedding_index.parquet` and joined to",
  "`message_events.parquet` by `doc_key`. Verified: total = 902,144; assigned = 457,880;",
  "noise = 444,264; clusters = 5,167 (labels 0..5166). The 25 selected clusters reconnect to",
  "exactly 68,109 assigned documents, each cluster matching its Phase 4A inventory size.",
  "",
  "## Selection of the 10 x 100 sampled documents",
  "",
  sprintf("- Seed: `set.seed(%d)` (matches the Phase 4C planning seed).", SEED),
  sprintf("- RNG kind (pinned): %s.", paste(rng_kind, collapse = " / ")),
  "- Procedure: the RNG is seeded once; the 10 largest clusters are processed in the",
  "  `selected_clusters.csv` order (n_docs descending, ties by cluster_id ascending):",
  sprintf("  %s.", paste(largest_ids, collapse = ", ")),
  "  Within each cluster the assigned `doc_key` values are sorted ascending and 100 positions",
  "  are drawn with `sample.int(n, 100)` — simple random sampling WITHOUT replacement. No",
  "  weighting by probability, author, time, thread, text, or record type; representatives are",
  "  not preferentially sampled.",
  sprintf("- Re-run check: SHA256 of the sorted 1,000 sampled doc_keys = `%s`.", sample_hash),
  "- The 15 random clusters take the ENTIRE cluster (no additional sampling step).",
  "",
  "## Manual-review workload",
  "",
  sprintf("- 10 largest clusters: %d documents (10 x 100 simple random sample).", n_manual_largest),
  sprintf("- 15 random clusters: %d documents (complete clusters).", n_manual_random),
  sprintf("- **Total manual-review set: %d documents.**", n_manual_total),
  "",
  "## Representatives and overlap (measured, not forced)",
  "",
  sprintf("- Representative rows pulled from Phase 4A: %d (25 clusters x 5 centroid_nearest + 5 highest_membership).", n_rep_rows),
  sprintf("- Distinct representative documents: %d.", n_rep_distinct),
  sprintf("- Documents selected by BOTH representative methods (centroid_nearest AND highest_membership): %d.", n_rep_both),
  sprintf("- Representative documents that are also in the manual-review set: %d (largest-cluster sampled: %d; random-cluster full: %d).",
          n_rep_in_manual, n_rep_in_manual_largest, n_rep_in_manual_random),
  sprintf("- Representative documents present ONLY as representative context (largest clusters, not sampled): %d.", n_rep_context_only),
  sprintf("- **Unique documents in the final review package: %d** (= %d manual-review + %d representative-only).",
          n_pkg_unique, n_manual_total, n_rep_context_only),
  "",
  "## Descriptor availability among the 25 clusters",
  "",
  sprintf("- Clusters with the full 10 descriptors: %d.", n_desc_eq10),
  sprintf("- Clusters with 1–9 descriptors: %d.", n_desc_1to9),
  sprintf("- Clusters with 0 descriptors: %d.", n_desc_0),
  "- Descriptor counts are preserved exactly as computed in Phase 4A; none were created or replaced.",
  "",
  "## Parent-context availability",
  "",
  sprintf("- Package documents: %d (comments: %d; submissions: %d).", n_pkg_unique, n_com, n_sub),
  sprintf("- Comments whose direct parent resolves in the canonical dataset: %d of %d (%.2f%%).",
          n_com_parent, n_com, 100 * n_com_parent / max(n_com, 1L)),
  "- Parent fields use the Phase 4A method (match `parent_id` to a canonical `doc_key`); missing",
  "  parent content is left blank and never inferred.",
  "",
  "## Per-cluster overview",
  "",
  ovstr(),
  "",
  "cn = centroid_nearest representatives; hm = highest_membership representatives;",
  "pkg_unique = unique documents in the cluster's review package.",
  "",
  "## Outputs",
  "",
  "Machine-readable:",
  sprintf("- `review_package/review_documents.parquet` (%d rows)  \n  `%s`", n_pkg_unique, sha256f(out_files[["review_package/review_documents.parquet"]])),
  sprintf("- `review_package/review_documents.csv`  \n  `%s`", sha256f(out_files[["review_package/review_documents.csv"]])),
  sprintf("- `review_package/review_cluster_descriptors.csv` (%d rows)  \n  `%s`", nrow(desc_out), sha256f(out_files[["review_package/review_cluster_descriptors.csv"]])),
  sprintf("- `review_package/review_cluster_overview.csv` (25 rows)  \n  `%s`", sha256f(out_files[["review_package/review_cluster_overview.csv"]])),
  sprintf("- `review_package/manual_qc_review_form.csv` (25 rows; judgment fields blank)  \n  `%s`", sha256f(out_files[["review_package/manual_qc_review_form.csv"]])),
  sprintf("- `derived/manual_review_sampled_documents.csv` (%d rows)  \n  `%s`", n_manual_total, sha256f(out_files[["derived/manual_review_sampled_documents.csv"]])),
  "",
  "Human-readable:",
  sprintf("- `review_package/clusters/` — %d per-cluster review files (`cluster_<id>_<group>.md`),", length(md_files)),
  "  each ordered: summary, c-TF-IDF descriptors, centroid_nearest reps, highest_membership reps,",
  "  manual-review documents. Full original Reddit text is preserved (no truncation).",
  "",
  "## Environment and cost",
  "",
  sprintf("- %s; nanoparquet %s; data.table %s; readr %s.",
          R.version.string, packageVersion("nanoparquet"), packageVersion("data.table"), packageVersion("readr")),
  sprintf("- Runtime: %.1f s. Approximate peak RAM (gc max used): %.2f GB.", elapsed, peak_gb),
  sprintf("- Generated (UTC): %s.", stamp),
  "",
  "## Departures from specification",
  "",
  "- None affecting the data. As specified, no manifest sidecars or git commits are produced;",
  "  provenance is folded into this REPORT.md (consistent with Phase 4A/4B/4C-planning). The",
  "  heavy `raw_json` column of `message_events.parquet` is not loaded (column-selective read);",
  "  this changes nothing in the outputs.",
  ""
)
writeLines(R, file.path(RP, "REPORT.md"), sep = "\n")

# =====================================================================================
# SECTION 15. Console summary.
# =====================================================================================
cat("\n===== STAGE TIMING =====\n"); print(STAGE)
cat(sprintf("\nTOTAL runtime: %.1f s ; peak RAM ~ %.2f GB\n", elapsed, peak_gb))
cat(sprintf("manual-review total=%d (largest=%d, random=%d)\n", n_manual_total, n_manual_largest, n_manual_random))
cat(sprintf("reps: rows=%d distinct=%d both_methods=%d in_manual=%d context_only=%d\n",
            n_rep_rows, n_rep_distinct, n_rep_both, n_rep_in_manual, n_rep_context_only))
cat(sprintf("UNIQUE package documents=%d\n", n_pkg_unique))
cat(sprintf("descriptors: clusters 10=%d, 1-9=%d, 0=%d\n", n_desc_eq10, n_desc_1to9, n_desc_0))
cat(sprintf("comments=%d parent_resolved=%d (%.2f%%) submissions=%d\n",
            n_com, n_com_parent, 100 * n_com_parent / max(n_com, 1L), n_sub))
cat(sprintf("sample_hash=%s\n", sample_hash))
cat("===== DONE =====\n")
