#!/usr/bin/env Rscript
# =====================================================================================
# Phase 4B: metadata description and automated QC
# =====================================================================================
# Descriptive / QC measurements ONLY for the 5,167 assigned full-corpus HDBSCAN semantic
# clusters (Phase 3), reconnected to the canonical Reddit records (Phase 1) and to the
# Phase 2 embeddings and the Phase 4A c-TF-IDF descriptors. This script measures:
#   1. author participation and concentration          6. direct-parent / reply context
#   2. temporal distribution                           7. combined one-row-per-cluster summary
#   3. thread distribution                             8. topic diversity (from 4A c-TF-IDF)
#   4. subreddit distribution                          9. within-cluster semantic similarity
#   5. observed canonical URL / domain distribution   10. TC-W2V coherence -> resource check only
#
# It does NOT interpret, name, score, or relabel any cluster; it does NOT infer coordination,
# suspiciousness, mobilization, disinformation, or importance; it does NOT re-run Phase 4A,
# embeddings, UMAP, or HDBSCAN, and it does NOT modify any source artifact. Only the 457,880
# assigned documents are summarised; the 444,264 unassigned ("label -1") documents are left
# untouched in the source and not summarised here.
#
# All filtering, joining, aggregation, feature construction, and QC math is explicit below.
# Fully deterministic: no random operations; every rank tie is broken by an explicit key.
#
# Usage:
#   Rscript run_phase4b_metadata_and_qc.R           # FULL run
#   Rscript run_phase4b_metadata_and_qc.R --quick   # smoke run over a few clusters
# =====================================================================================

suppressWarnings(suppressMessages({
  library(nanoparquet)
  library(data.table)
  library(stringi)
  library(ggplot2)
  library(digest)
  library(scales)
}))

t_start <- Sys.time()
options(width = 200, warn = 1)
data.table::setDTthreads(0L)

QUICK <- any(commandArgs(trailingOnly = TRUE) == "--quick")
cat("MODE:", if (QUICK) "QUICK (smoke)" else "FULL", "\n")

# ---- paths (inputs are READ-ONLY) -----------------------------------------------------
BASE <- Sys.getenv("KC_REPO_ROOT", unset = getwd())
PIPE <- file.path(BASE, "pipeline")
P1_ME  <- file.path(PIPE, "phase1_message_event_foundation/derived/message_events.parquet")
P2_NPY <- file.path(PIPE, "phase2_semantic_embeddings/full_corpus_embedding_0p6b/results/embeddings.npy")
P2_IDX <- file.path(PIPE, "phase2_semantic_embeddings/full_corpus_embedding_0p6b/results/embedding_index.parquet")
P3_LAB <- file.path(PIPE, "phase3_blind_semantic_structure_discovery/full_corpus_gpu_umap_hdbscan_baseline/results/hdbscan_labels.parquet")
P4A_CT <- file.path(PIPE, "phase4_semantic_structure_description_qc/4a_cluster_description_foundation/derived/ctfidf_top_terms.parquet")
P4A_CE <- file.path(PIPE, "phase4_semantic_structure_description_qc/4a_cluster_description_foundation/derived/cluster_centroids.parquet")

OUT   <- file.path(PIPE, "phase4_semantic_structure_description_qc/4b_metadata_and_automated_qc")
D_DER <- file.path(OUT, "derived")
D_FIG <- file.path(OUT, "figures")
dir.create(D_DER, recursive = TRUE, showWarnings = FALSE)
dir.create(D_FIG, recursive = TRUE, showWarnings = FALSE)

# ---- expected Phase 3 / 4A measurements (consistency checks, NOT re-validation) -------
EXP_TOTAL    <- 902144L
EXP_CLUSTERS <- 5167L
EXP_ASSIGNED <- 457880L
EXP_NOISE    <- 444264L
EMB_DIM      <- 1024L
NPY_HDR      <- 128L        # verified in Phase 4A: 10-byte preamble + 118-byte header dict
EXP_CT_ROWS  <- 49472L      # Phase 4A c-TF-IDF descriptor rows

# ---- helpers --------------------------------------------------------------------------
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
write_pq <- function(x, name) {
  p <- file.path(D_DER, name)
  nanoparquet::write_parquet(x, p, compression = "zstd",
                             options = parquet_options(compression_level = 3L))
  cat(sprintf("[write] %-38s %10s rows x %d cols\n", name, format(nrow(x), big.mark=","), ncol(x)))
  invisible(p)
}
fmt_utc <- function(x) format(as.POSIXct(x, origin = "1970-01-01", tz = "UTC"),
                              "%Y-%m-%d %H:%M:%S UTC")
# Reproducible host/domain extraction from a URL string (mechanical; no public-suffix
# reduction, no www-stripping -- those would require an external list / a judgment call):
#   lowercase+trim -> strip scheme:// -> keep authority (up to first / ? #) ->
#   drop userinfo@ -> drop :port. Empty / unparsable -> NA.
url_host <- function(u) {
  s <- stringi::stri_trans_tolower(trimws(u))
  s <- sub("^[a-z][a-z0-9+.-]*://", "", s)   # scheme://
  s <- sub("[/?#].*$", "", s)                # authority = up to first / ? #
  s <- sub("^[^@]*@", "", s)                 # drop userinfo@
  s <- sub(":[0-9]+$", "", s)                # drop :port
  s[is.na(u) | s == ""] <- NA_character_
  s
}
# rank a long (cluster_id, key, doc_count) table within cluster: count desc, key asc
add_rank <- function(dt, keycol) {
  setorderv(dt, c("cluster_id", "doc_count", keycol), c(1L, -1L, 1L))
  dt[, rank := rowid(cluster_id)]
  dt[]
}

# =====================================================================================
# SECTION 1. Load HDBSCAN labels + embedding index; verify alignment and Phase 3 counts.
# =====================================================================================
lab <- timeit("load labels+index", {
  l <- as.data.table(nanoparquet::read_parquet(P3_LAB))   # row, doc_key, record_type, label, probability
  i <- as.data.table(nanoparquet::read_parquet(P2_IDX))   # row, doc_key, record_type, created_utc, ...
  l[, row := as.integer(row)]; i[, row := as.integer(row)]
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
n_clusters  <- length(clusters_present)

cat(sprintf("consistency: total=%d (exp %d) assigned=%d (exp %d) noise=%d (exp %d) clusters=%d (exp %d)\n",
            n_total, EXP_TOTAL, n_assigned, EXP_ASSIGNED, n_noise, EXP_NOISE, n_clusters, EXP_CLUSTERS))
stopifnot(n_total == EXP_TOTAL, n_assigned == EXP_ASSIGNED,
          n_noise == EXP_NOISE, n_clusters == EXP_CLUSTERS,
          identical(clusters_present, 0:(EXP_CLUSTERS - 1L)))
cat("[ok] input consistency checks pass\n")

# =====================================================================================
# SECTION 2. Load canonical message_events (only the columns Phase 4B needs).
# =====================================================================================
me <- timeit("load message_events", {
  m <- nanoparquet::read_parquet(P1_ME)
  keep <- c("doc_key","record_type","author","created_utc","created_date_utc",
            "subreddit","thread_id","parent_id","parent_kind","parent_in_dataset","url")
  m <- as.data.table(m[, keep])
  setkey(m, doc_key)
  m
})
stopifnot(nrow(me) == n_total, !any(duplicated(me$doc_key)))

# =====================================================================================
# SECTION 3. Assemble the assigned-document working table `ad` (one row per assigned doc).
# =====================================================================================
ad0 <- lab[is_assigned, .(doc_key, row0 = row, cluster = label, probability)]
ad  <- me[ad0, on = "doc_key"]         # attach metadata from canonical me
stopifnot(nrow(ad) == n_assigned, !anyNA(ad$cluster), !anyNA(ad$record_type))

# explicit, visible UTC-calendar-day derivation (epoch seconds -> whole UTC days since epoch,
# locale-independent); assert value-equality with the canonical date column (the nanoparquet
# Date is integer-backed, so compare with == not identical()).
ad[, day_utc := as.Date(created_utc %/% 86400, origin = "1970-01-01")]
stopifnot(all(ad$day_utc == ad$created_date_utc))

if (QUICK) {
  # choose clusters that exercise every branch: a few small ones, the largest cluster,
  # and the cluster holding the most submissions (guarantees the URL/domain path runs)
  sub_by_cl  <- ad[record_type == "submission", .N, by = cluster][order(-N)]
  keep_cl <- unique(c(head(clusters_present, 4L),
                      ad[, .N, by = cluster][order(-N)][1L, cluster],
                      if (nrow(sub_by_cl)) sub_by_cl[1L, cluster]))
  keep_cl <- sort(keep_cl)
  ad <- ad[cluster %in% keep_cl]
  cat(sprintf("[quick] restricted to %d clusters (%s), %d docs, %d submissions\n",
              length(keep_cl), paste(keep_cl, collapse = ","), nrow(ad),
              ad[record_type == "submission", .N]))
}
ad_clusters   <- sort(unique(ad$cluster))
n_ad_clusters <- length(ad_clusters)
all_clusters  <- data.table(cluster_id = ad_clusters)

# cluster document counts (the shared denominator for "share of cluster documents")
cl_n <- ad[, .(n_docs = .N,
               n_comments    = sum(record_type == "comment"),
               n_submissions = sum(record_type == "submission")), by = .(cluster_id = cluster)]
setorder(cl_n, cluster_id)
stopifnot(cl_n$n_docs == cl_n$n_comments + cl_n$n_submissions, sum(cl_n$n_docs) == nrow(ad))

# =====================================================================================
# SECTION 4. AUTHOR description (participation + concentration).
# =====================================================================================
cluster_authors <- ad[, .(doc_count = .N), by = .(cluster_id = cluster, author)]
cluster_authors <- add_rank(cluster_authors, "author")
cluster_authors <- merge(cluster_authors, cl_n[, .(cluster_id, n_docs)], by = "cluster_id")
cluster_authors[, share := doc_count / n_docs]
cluster_authors[, n_docs := NULL]
setcolorder(cluster_authors, c("cluster_id","author","doc_count","share","rank"))
setorder(cluster_authors, cluster_id, rank)

cluster_author_summary <- cluster_authors[, {
  nd <- sum(doc_count)
  .(n_docs                   = nd,
    n_unique_authors         = .N,
    docs_per_author_mean     = nd / .N,
    largest_author_doc_count = doc_count[rank == 1L],
    largest_author_share     = doc_count[rank == 1L] / nd,
    top5_author_doc_count    = sum(doc_count[rank <= 5L]),
    top5_author_share        = sum(doc_count[rank <= 5L]) / nd)
}, by = cluster_id]
setorder(cluster_author_summary, cluster_id)
write_pq(cluster_author_summary, "cluster_author_summary.parquet")
write_pq(cluster_authors,        "cluster_authors.parquet")

# =====================================================================================
# SECTION 5. TEMPORAL description.
# =====================================================================================
cluster_days <- ad[, .(doc_count = .N), by = .(cluster_id = cluster, date_utc = day_utc)]
setorder(cluster_days, cluster_id, date_utc)

cluster_temporal_summary <- ad[, .(
  first_utc      = min(created_utc),
  last_utc       = max(created_utc),
  span_seconds   = max(created_utc) - min(created_utc),
  span_days      = (max(created_utc) - min(created_utc)) / 86400,
  n_active_days  = uniqueN(day_utc)
), by = .(cluster_id = cluster)]
cluster_temporal_summary[, `:=`(first_datetime_utc = fmt_utc(first_utc),
                                last_datetime_utc  = fmt_utc(last_utc))]
setcolorder(cluster_temporal_summary,
            c("cluster_id","first_utc","last_utc","first_datetime_utc","last_datetime_utc",
              "span_seconds","span_days","n_active_days"))
setorder(cluster_temporal_summary, cluster_id)
write_pq(cluster_temporal_summary, "cluster_temporal_summary.parquet")
write_pq(cluster_days,             "cluster_days.parquet")

# =====================================================================================
# SECTION 6. THREAD description.
# =====================================================================================
cluster_threads <- ad[, .(doc_count = .N), by = .(cluster_id = cluster, thread_id)]
cluster_threads <- add_rank(cluster_threads, "thread_id")
cluster_threads <- merge(cluster_threads, cl_n[, .(cluster_id, n_docs)], by = "cluster_id")
cluster_threads[, share := doc_count / n_docs][, n_docs := NULL]
setcolorder(cluster_threads, c("cluster_id","thread_id","doc_count","share","rank"))
setorder(cluster_threads, cluster_id, rank)

cluster_thread_summary <- cluster_threads[, {
  nd <- sum(doc_count)
  .(n_docs                   = nd,
    n_unique_threads         = .N,
    docs_per_thread_mean     = nd / .N,
    largest_thread_doc_count = doc_count[rank == 1L],
    largest_thread_share     = doc_count[rank == 1L] / nd)
}, by = cluster_id]
setorder(cluster_thread_summary, cluster_id)
write_pq(cluster_thread_summary, "cluster_thread_summary.parquet")
write_pq(cluster_threads,        "cluster_threads.parquet")

# =====================================================================================
# SECTION 7. SUBREDDIT description.
# =====================================================================================
cluster_subreddits <- ad[, .(doc_count = .N), by = .(cluster_id = cluster, subreddit)]
cluster_subreddits <- add_rank(cluster_subreddits, "subreddit")
cluster_subreddits <- merge(cluster_subreddits, cl_n[, .(cluster_id, n_docs)], by = "cluster_id")
cluster_subreddits[, share := doc_count / n_docs][, n_docs := NULL]
setcolorder(cluster_subreddits, c("cluster_id","subreddit","doc_count","share","rank"))
setorder(cluster_subreddits, cluster_id, rank)

cluster_subreddit_summary <- cluster_subreddits[, {
  nd <- sum(doc_count)
  .(n_docs                      = nd,
    n_unique_subreddits         = .N,
    docs_per_subreddit_mean     = nd / .N,
    largest_subreddit_doc_count = doc_count[rank == 1L],
    largest_subreddit_share     = doc_count[rank == 1L] / nd)
}, by = cluster_id]
setorder(cluster_subreddit_summary, cluster_id)
write_pq(cluster_subreddit_summary, "cluster_subreddit_summary.parquet")
write_pq(cluster_subreddits,        "cluster_subreddits.parquet")

# =====================================================================================
# SECTION 8. URL and DOMAIN description (observed canonical Phase 1 `url`).
#   `url` is populated for submissions only; comments have NA. "Observed URL" = non-missing
#   and non-empty (after trim). The exact observed URL is preserved; the domain is the
#   reproducibly-derived host (url_host()).
# =====================================================================================
ad[, has_url := !is.na(url) & trimws(url) != ""]
ad[, url_host := ifelse(has_url, url_host(url), NA_character_)]
# any observed URL that failed host extraction (QC transparency)
n_url_no_host <- ad[has_url & is.na(url_host), .N]

# cluster URL summary over ALL clusters (zero-URL clusters included, with zeros)
cluster_url_summary <- ad[, .(
  n_docs               = .N,
  n_url_docs           = sum(has_url),
  url_doc_share        = sum(has_url) / .N,
  n_unique_exact_urls  = uniqueN(url[has_url]),
  n_unique_domains     = uniqueN(url_host[has_url])
), by = .(cluster_id = cluster)]
setorder(cluster_url_summary, cluster_id)
write_pq(cluster_url_summary, "cluster_url_summary.parquet")

# cluster x exact-URL counts (only URL-bearing docs; shares are of the cluster's URL docs)
cluster_urls <- ad[has_url == TRUE, .(doc_count = .N), by = .(cluster_id = cluster, url)]
if (nrow(cluster_urls)) {
  setorderv(cluster_urls, c("cluster_id","doc_count","url"), c(1L, -1L, 1L))
  cluster_urls[, rank := rowid(cluster_id)]
  cluster_urls <- merge(cluster_urls, cluster_url_summary[, .(cluster_id, n_url_docs)], by = "cluster_id")
  cluster_urls[, share_of_url_docs := doc_count / n_url_docs][, n_url_docs := NULL]
  setcolorder(cluster_urls, c("cluster_id","url","doc_count","share_of_url_docs","rank"))
  setorder(cluster_urls, cluster_id, rank)
} else {
  cluster_urls <- data.table(cluster_id = integer(0), url = character(0),
                             doc_count = integer(0), share_of_url_docs = numeric(0), rank = integer(0))
}
write_pq(cluster_urls, "cluster_urls.parquet")

# cluster x domain counts
cluster_domains <- ad[has_url == TRUE & !is.na(url_host), .(doc_count = .N),
                      by = .(cluster_id = cluster, domain = url_host)]
if (nrow(cluster_domains)) {
  setorderv(cluster_domains, c("cluster_id","doc_count","domain"), c(1L, -1L, 1L))
  cluster_domains[, rank := rowid(cluster_id)]
  cluster_domains <- merge(cluster_domains, cluster_url_summary[, .(cluster_id, n_url_docs)], by = "cluster_id")
  cluster_domains[, share_of_url_docs := doc_count / n_url_docs][, n_url_docs := NULL]
  setcolorder(cluster_domains, c("cluster_id","domain","doc_count","share_of_url_docs","rank"))
  setorder(cluster_domains, cluster_id, rank)
} else {
  cluster_domains <- data.table(cluster_id = integer(0), domain = character(0),
                                doc_count = integer(0), share_of_url_docs = numeric(0), rank = integer(0))
}
write_pq(cluster_domains, "cluster_domains.parquet")

# =====================================================================================
# SECTION 9. DIRECT-PARENT / reply-context description (COMMENTS only).
#   Every comment has a non-missing parent_id and parent_kind. parent_in_dataset marks
#   whether that parent_id exists as a doc_key in the canonical corpus. We resolve a
#   canonical parent_doc_key by matching parent_id against me$doc_key (NA when absent) and
#   cross-check it against the stored parent_in_dataset flag.
# =====================================================================================
cmt <- ad[record_type == "comment",
          .(cluster_id = cluster, doc_key, parent_id, parent_kind, parent_in_dataset)]
cmt[, parent_doc_key := me$doc_key[match(parent_id, me$doc_key)]]
stopifnot(all((!is.na(cmt$parent_doc_key)) == (cmt$parent_in_dataset %in% TRUE)))  # flag agrees with linkage

# long: one row per (cluster, parent_id)
cluster_parents <- cmt[, .(
  parent_kind       = parent_kind[1L],
  parent_in_dataset = parent_in_dataset[1L],
  parent_doc_key    = parent_doc_key[1L],
  n_replies         = .N
), by = .(cluster_id, parent_id)]
# share is of applicable cluster comments (= all cluster comments, since every comment has a parent_id)
ncmt_by_cl <- cmt[, .(n_cluster_comments = .N), by = cluster_id]
cluster_parents <- merge(cluster_parents, ncmt_by_cl, by = "cluster_id")
cluster_parents[, share := n_replies / n_cluster_comments][, n_cluster_comments := NULL]
setorderv(cluster_parents, c("cluster_id","n_replies","parent_id"), c(1L, -1L, 1L))
cluster_parents[, rank := rowid(cluster_id)]
setcolorder(cluster_parents, c("cluster_id","parent_id","parent_kind","parent_in_dataset",
                               "parent_doc_key","n_replies","share","rank"))
setorder(cluster_parents, cluster_id, rank)
write_pq(cluster_parents, "cluster_parents.parquet")

# per-cluster parent summary (comment-bearing clusters), then extended to all clusters
parent_core <- cmt[, .(
  n_comments                = .N,
  n_parent_in_dataset       = sum(parent_in_dataset %in% TRUE),
  parent_in_dataset_share   = mean(parent_in_dataset %in% TRUE),
  n_parent_kind_comment     = sum(parent_kind == "comment"),
  n_parent_kind_submission  = sum(parent_kind == "submission"),
  share_parent_kind_comment    = mean(parent_kind == "comment"),
  share_parent_kind_submission = mean(parent_kind == "submission"),
  n_unique_parents          = uniqueN(parent_id),
  docs_per_parent_mean      = .N / uniqueN(parent_id)
), by = cluster_id]
parent_big <- cluster_parents[, .(largest_parent_reply_count = n_replies[rank == 1L],
                                  largest_parent_share       = n_replies[rank == 1L] / sum(n_replies)),
                              by = cluster_id]
cluster_parent_summary <- merge(all_clusters, parent_core, by = "cluster_id", all.x = TRUE)
cluster_parent_summary <- merge(cluster_parent_summary, parent_big, by = "cluster_id", all.x = TRUE)
# clusters with zero comments: fill counts with 0, shares stay NA (no applicable denominator)
zcols <- c("n_comments","n_parent_in_dataset","n_parent_kind_comment","n_parent_kind_submission",
           "n_unique_parents","largest_parent_reply_count")
for (cc in zcols) set(cluster_parent_summary, which(is.na(cluster_parent_summary[[cc]])), cc, 0L)
setorder(cluster_parent_summary, cluster_id)
write_pq(cluster_parent_summary, "cluster_parent_summary.parquet")

# =====================================================================================
# SECTION 10. TOPIC DIVERSITY QC (from the existing Phase 4A c-TF-IDF descriptors). [sec 8]
#   topic_diversity = (# unique descriptor strings present) / (# descriptor rows present)
#   over ctfidf_top_terms.parquet, WITHOUT removing repeats before the denominator. Absent
#   descriptors (clusters with <10 or 0 rows) are simply not present -- not treated as dups.
# =====================================================================================
ct <- as.data.table(nanoparquet::read_parquet(P4A_CT))
if (QUICK) ct <- ct[cluster_id %in% ad_clusters]
if (!QUICK) stopifnot(nrow(ct) == EXP_CT_ROWS)

total_descriptor_rows     <- nrow(ct)
n_unique_descriptor_terms <- uniqueN(ct$term)
topic_diversity           <- n_unique_descriptor_terms / total_descriptor_rows

desc_per_cluster <- ct[, .(k = .N), by = cluster_id]                 # clusters with >=1 descriptor
n_desc_clusters_any <- nrow(desc_per_cluster)
n_desc_eq10  <- desc_per_cluster[k == 10L, .N]
n_desc_1to9  <- desc_per_cluster[k >= 1L & k <= 9L, .N]
n_desc_zero  <- n_ad_clusters - n_desc_clusters_any                 # clusters with NO rows at all

topic_diversity_qc <- data.table(
  total_descriptor_rows     = total_descriptor_rows,
  n_unique_descriptor_terms = n_unique_descriptor_terms,
  topic_diversity           = topic_diversity,
  n_clusters_with_10        = n_desc_eq10,
  n_clusters_with_1to9      = n_desc_1to9,
  n_clusters_with_0         = n_desc_zero,
  n_clusters_with_any       = n_desc_clusters_any,
  top_n_descriptors         = 10L
)
write_pq(topic_diversity_qc, "topic_diversity_qc.parquet")

# how often each descriptor string recurs ACROSS different clusters (term appears once per
# cluster, so n_clusters == number of distinct clusters containing it)
descriptor_recurrence <- ct[, .(n_clusters = uniqueN(cluster_id)), by = term]
setorder(descriptor_recurrence, -n_clusters, term)
write_pq(descriptor_recurrence, "descriptor_recurrence.parquet")
recurrence_dist <- descriptor_recurrence[, .(n_terms = .N), by = n_clusters][order(n_clusters)]

# =====================================================================================
# SECTION 11. WITHIN-CLUSTER semantic-similarity QC (original 1024-d L2-normalized emb). [sec 9]
#   For cluster size n and vector sum s:  mean_pairwise_cosine = (||s||^2 - n) / (n*(n-1)),
#   exact for n>1 (all clusters have n>=10). No UMAP coordinates are used.
# =====================================================================================
read_npy_rows <- function(path, rows0, dim = EMB_DIM, hdr = NPY_HDR) {
  con <- file(path, "rb"); on.exit(close(con))
  out <- matrix(NA_real_, nrow = dim, ncol = length(rows0))
  for (j in seq_along(rows0)) {
    seek(con, where = hdr + as.numeric(rows0[j]) * dim * 4, origin = "start")
    out[, j] <- readBin(con, what = "double", n = dim, size = 4, endian = "little")
  }
  out
}
read_npy_all <- function(path, ntot, dim = EMB_DIM, hdr = NPY_HDR) {
  con <- file(path, "rb"); on.exit(close(con))
  seek(con, where = hdr, origin = "start")
  v <- readBin(con, what = "double", n = as.numeric(ntot) * dim, size = 4, endian = "little")
  stopifnot(length(v) == as.numeric(ntot) * dim)
  dim(v) <- c(dim, ntot)
  v
}

emb <- timeit("load embeddings", {
  if (QUICK) {
    read_npy_rows(P2_NPY, ad$row0)
  } else {
    Mall <- read_npy_all(P2_NPY, n_total)
    E <- Mall[, ad$row0 + 1L, drop = FALSE]
    rm(Mall); gc()
    E
  }
})
stopifnot(ncol(emb) == nrow(ad), nrow(emb) == EMB_DIM)
# alignment cross-check vs the trusted per-row seek reader
chk_idx <- unique(round(seq(1, nrow(ad), length.out = min(6L, nrow(ad)))))
stopifnot(max(abs(emb[, chk_idx, drop = FALSE] - read_npy_rows(P2_NPY, ad$row0[chk_idx]))) < 1e-6)
# verify L2-unit-norm (the formula requires it)
cn_check <- sqrt(colSums(emb[, seq_len(min(500L, ncol(emb))), drop = FALSE]^2))
cat(sprintf("embedding column-norm sample: min=%.6f max=%.6f\n", min(cn_check), max(cn_check)))
stopifnot(max(abs(cn_check - 1)) < 1e-3)

within_cluster_similarity <- timeit("within-cluster cosine", {
  col_by_cluster <- split(seq_len(nrow(ad)), ad$cluster)
  cl_ids <- as.integer(names(col_by_cluster))
  mpc <- numeric(length(cl_ids)); nn <- integer(length(cl_ids)); snorm2 <- numeric(length(cl_ids))
  for (k in seq_along(cl_ids)) {
    cols <- col_by_cluster[[k]]
    n <- length(cols)
    s <- rowSums(emb[, cols, drop = FALSE])   # sum of the (unit) document vectors
    ss <- sum(s * s)                           # ||s||^2
    nn[k] <- n; snorm2[k] <- ss
    mpc[k] <- (ss - n) / (n * (n - 1))         # n >= 10 guaranteed
  }
  data.table(cluster_id = cl_ids, n_docs = nn, sum_norm_sq = snorm2,
             mean_pairwise_cosine = mpc)
})
setorder(within_cluster_similarity, cluster_id)
stopifnot(all(within_cluster_similarity$n_docs >= 2L))
write_pq(within_cluster_similarity[, .(cluster_id, n_docs, mean_pairwise_cosine)],
         "within_cluster_similarity.parquet")

# independent cross-check against the Phase 4A stored centroids:
#   centroid c = s/n  =>  ||s||^2 = n^2 * ||c||^2  =>  mpc = (n^2||c||^2 - n)/(n(n-1))
if (file.exists(P4A_CE)) {
  ce <- as.data.table(nanoparquet::read_parquet(P4A_CE))
  ce <- ce[cluster_id %in% ad_clusters]
  vcols <- paste0("v", 0:(EMB_DIM - 1L))
  chk_cl <- ce$cluster_id[unique(round(seq(1, nrow(ce), length.out = min(20L, nrow(ce)))))]
  cnorm2 <- rowSums(as.matrix(ce[cluster_id %in% chk_cl, ..vcols])^2)
  cc_ids <- ce[cluster_id %in% chk_cl, cluster_id]
  ref <- merge(data.table(cluster_id = cc_ids, cnorm2 = cnorm2),
               within_cluster_similarity[, .(cluster_id, n_docs, mean_pairwise_cosine)], by = "cluster_id")
  ref[, mpc_from_centroid := (n_docs^2 * cnorm2 - n_docs) / (n_docs * (n_docs - 1))]
  max_abs <- max(abs(ref$mean_pairwise_cosine - ref$mpc_from_centroid))
  cat(sprintf("[ok] within-cluster cosine cross-check vs 4A centroids on %d clusters: max abs diff %.2e\n",
              nrow(ref), max_abs))
  stopifnot(max_abs < 1e-6)
}

wcs <- within_cluster_similarity$mean_pairwise_cosine
sim_summary <- data.table(
  metric = "mean_pairwise_cosine_within_cluster",
  n_clusters = length(wcs),
  unweighted_mean_across_clusters = mean(wcs),   # PRIMARY headline (each cluster counts once)
  median = median(wcs),
  p25 = quantile(wcs, 0.25, names = FALSE, type = 7),
  p75 = quantile(wcs, 0.75, names = FALSE, type = 7),
  min = min(wcs), max = max(wcs)
)
write_pq(sim_summary, "within_cluster_similarity_summary.parquet")

# =====================================================================================
# SECTION 12. COMBINED one-row-per-cluster metadata summary.
#   Join of the principal descriptive measurements from sections 4-9 plus the topic-QC
#   descriptor count and the within-cluster cosine. No composite / coordination score.
# =====================================================================================
cluster_metadata_summary <- Reduce(function(a, b) merge(a, b, by = "cluster_id", all.x = TRUE), list(
  cl_n[, .(cluster_id, n_docs, n_comments, n_submissions)],
  cluster_author_summary[, .(cluster_id, n_unique_authors, largest_author_share, top5_author_share)],
  cluster_temporal_summary[, .(cluster_id, first_utc, last_utc, first_datetime_utc, last_datetime_utc,
                               span_seconds, span_days, n_active_days)],
  cluster_thread_summary[, .(cluster_id, n_unique_threads, largest_thread_share)],
  cluster_subreddit_summary[, .(cluster_id, n_unique_subreddits, largest_subreddit_share)],
  cluster_url_summary[, .(cluster_id, n_url_docs, url_doc_share, n_unique_exact_urls, n_unique_domains)],
  cluster_parent_summary[, .(cluster_id, n_comments_ctx = n_comments, parent_in_dataset_share,
                             n_unique_parents, largest_parent_share)],
  within_cluster_similarity[, .(cluster_id, mean_pairwise_cosine)]
))
# attach descriptor count per cluster (0 when the cluster has no c-TF-IDF rows)
cluster_metadata_summary <- merge(cluster_metadata_summary,
                                  desc_per_cluster[, .(cluster_id, n_descriptors = k)],
                                  by = "cluster_id", all.x = TRUE)
set(cluster_metadata_summary, which(is.na(cluster_metadata_summary$n_descriptors)), "n_descriptors", 0L)
stopifnot(all(cluster_metadata_summary$n_comments == cluster_metadata_summary$n_comments_ctx))
cluster_metadata_summary[, n_comments_ctx := NULL]
setorder(cluster_metadata_summary, cluster_id)
stopifnot(nrow(cluster_metadata_summary) == n_ad_clusters)
write_pq(cluster_metadata_summary, "cluster_metadata_summary.parquet")

# =====================================================================================
# SECTION 13. Figures (title + axis/tick labels only; no captions/subtitles/annotations).
# =====================================================================================
hist_fig <- function(dt, col, title, xlab, fname, log10x = FALSE, binwidth = NULL, bins = 40) {
  g <- ggplot(dt, aes(x = .data[[col]])) +
    { if (!is.null(binwidth)) geom_histogram(binwidth = binwidth, fill = "grey30")
      else geom_histogram(bins = bins, fill = "grey30") } +
    scale_y_continuous(labels = scales::label_comma()) +
    labs(title = title, x = xlab, y = "Number of clusters") +
    theme_minimal(base_size = 12)
  if (log10x) g <- g + scale_x_log10(labels = scales::label_comma())
  ggsave(file.path(D_FIG, fname), g, width = 7, height = 5, dpi = 150)
}
hist_fig(cluster_author_summary, "n_unique_authors",
         "Unique authors per cluster", "Unique authors per cluster (log10 scale)",
         "fig_unique_authors_per_cluster.png", log10x = TRUE)
hist_fig(cluster_author_summary, "largest_author_share",
         "Largest-author share of cluster documents", "Largest-author share",
         "fig_largest_author_share.png", binwidth = 0.02)
hist_fig(cluster_temporal_summary, "span_days",
         "Observed cluster time span", "Observed span (days)",
         "fig_time_span_days.png", bins = 40)
hist_fig(cluster_temporal_summary, "n_active_days",
         "Active UTC days per cluster", "Active UTC days",
         "fig_active_days.png", binwidth = 1)
hist_fig(cluster_thread_summary, "n_unique_threads",
         "Unique threads per cluster", "Unique threads per cluster (log10 scale)",
         "fig_unique_threads_per_cluster.png", log10x = TRUE)
hist_fig(cluster_subreddit_summary, "n_unique_subreddits",
         "Unique subreddits per cluster", "Unique subreddits per cluster (log10 scale)",
         "fig_unique_subreddits_per_cluster.png", log10x = TRUE)
hist_fig(cluster_url_summary, "url_doc_share",
         "URL-bearing document share per cluster", "URL-bearing document share",
         "fig_url_bearing_share.png", binwidth = 0.02)
# direct-parent coverage is defined for comment-bearing clusters only
hist_fig(cluster_parent_summary[n_comments > 0L], "parent_in_dataset_share",
         "Direct-parent coverage per cluster", "Share of cluster comments with in-dataset parent",
         "fig_parent_coverage.png", binwidth = 0.02)
hist_fig(within_cluster_similarity, "mean_pairwise_cosine",
         "Within-cluster mean pairwise cosine similarity", "Mean pairwise cosine similarity",
         "fig_within_cluster_cosine.png", bins = 40)
cat("figures: 9 written\n")

# =====================================================================================
# SECTION 14. Provenance + run statistics (printed; folded into REPORT.md by the analyst).
# =====================================================================================
sha <- function(p, do = TRUE) if (do && file.exists(p)) digest::digest(p, algo = "sha256", file = TRUE) else "(skipped)"
cat("\n===== PROVENANCE (sha256 of inputs) =====\n")
cat("message_events.parquet  :", sha(P1_ME), "\n")
cat("embedding_index.parquet :", sha(P2_IDX), "\n")
cat("hdbscan_labels.parquet  :", sha(P3_LAB), "\n")
cat("ctfidf_top_terms.parquet:", sha(P4A_CT), "\n")
cat("cluster_centroids.parquet:", sha(P4A_CE), "\n")
cat("embeddings.npy          :", sha(P2_NPY, do = !QUICK), "\n")

cat("\n===== STAGE TIMING / PEAK =====\n"); print(STAGE)
cat(sprintf("TOTAL runtime: %.1f s\n", as.numeric(difftime(Sys.time(), t_start, units = "secs"))))
g <- gc()
cat(sprintf("overall peak (gc max used) ~ %.2f GB\n",
            (g["Vcells","max used"] * 8 + g["Ncells","max used"] * 56) / 2^30))

cat("\n===== KEY NUMBERS (for REPORT.md) =====\n")
cat(sprintf("MODE=%s\n", if (QUICK) "QUICK" else "FULL"))
cat(sprintf("total_docs=%d assigned=%d noise=%d clusters=%d\n", n_total, n_assigned, n_noise, n_clusters))
cat(sprintf("described_clusters=%d described_docs=%d\n", n_ad_clusters, nrow(ad)))
cat("\n-- table row counts --\n")
rc <- function(x) format(nrow(x), big.mark=",")
for (nm in c("cluster_metadata_summary","cluster_author_summary","cluster_authors",
             "cluster_temporal_summary","cluster_days","cluster_thread_summary","cluster_threads",
             "cluster_subreddit_summary","cluster_subreddits","cluster_url_summary","cluster_urls",
             "cluster_domains","cluster_parent_summary","cluster_parents","topic_diversity_qc",
             "descriptor_recurrence","within_cluster_similarity")) {
  cat(sprintf("  %-30s %s\n", nm, rc(get(nm))))
}
cat("\n-- authors --\n")
cat(sprintf("unique_authors: min=%d median=%.0f max=%d | largest_author_share: median=%.3f mean=%.3f\n",
            min(cluster_author_summary$n_unique_authors), median(cluster_author_summary$n_unique_authors),
            max(cluster_author_summary$n_unique_authors),
            median(cluster_author_summary$largest_author_share), mean(cluster_author_summary$largest_author_share)))
cat(sprintf("top5_author_share: median=%.3f mean=%.3f\n",
            median(cluster_author_summary$top5_author_share), mean(cluster_author_summary$top5_author_share)))
cat("\n-- temporal --\n")
cat(sprintf("span_days: min=%.4f median=%.3f max=%.3f | active_days: min=%d median=%.0f max=%d\n",
            min(cluster_temporal_summary$span_days), median(cluster_temporal_summary$span_days),
            max(cluster_temporal_summary$span_days),
            min(cluster_temporal_summary$n_active_days), median(cluster_temporal_summary$n_active_days),
            max(cluster_temporal_summary$n_active_days)))
cat(sprintf("corpus day range: %s .. %s\n", min(cluster_days$date_utc), max(cluster_days$date_utc)))
cat("\n-- threads / subreddits --\n")
cat(sprintf("unique_threads: min=%d median=%.0f max=%d | largest_thread_share median=%.3f\n",
            min(cluster_thread_summary$n_unique_threads), median(cluster_thread_summary$n_unique_threads),
            max(cluster_thread_summary$n_unique_threads), median(cluster_thread_summary$largest_thread_share)))
cat(sprintf("unique_subreddits: min=%d median=%.0f max=%d | largest_subreddit_share median=%.3f\n",
            min(cluster_subreddit_summary$n_unique_subreddits), median(cluster_subreddit_summary$n_unique_subreddits),
            max(cluster_subreddit_summary$n_unique_subreddits), median(cluster_subreddit_summary$largest_subreddit_share)))
cat("\n-- urls --\n")
cat(sprintf("clusters_with_any_url=%d (%.1f%%) | total_url_docs=%d | unique_exact_urls=%d unique_domains=%d | url_no_host=%d\n",
            cluster_url_summary[n_url_docs > 0, .N], 100*cluster_url_summary[n_url_docs>0,.N]/n_ad_clusters,
            sum(cluster_url_summary$n_url_docs), uniqueN(ad$url[ad$has_url]), uniqueN(ad$url_host[ad$has_url]),
            n_url_no_host))
cat("\n-- parents (comments) --\n")
cat(sprintf("total_comments=%d | parent_in_dataset_share over comment-clusters: median=%.4f mean=%.4f\n",
            sum(cluster_parent_summary$n_comments),
            median(cluster_parent_summary[n_comments>0, parent_in_dataset_share]),
            mean(cluster_parent_summary[n_comments>0, parent_in_dataset_share])))
cat(sprintf("largest_parent_share over comment-clusters: median=%.4f max=%.4f\n",
            median(cluster_parent_summary[n_comments>0, largest_parent_share]),
            max(cluster_parent_summary[n_comments>0, largest_parent_share])))
cat("\n-- topic diversity --\n")
print(topic_diversity_qc)
cat("recurrence distribution (n_clusters : n_terms), head:\n"); print(utils::head(recurrence_dist, 12))
cat(sprintf("most-recurring descriptors:\n")); print(utils::head(descriptor_recurrence, 10))
cat("\n-- within-cluster cosine --\n")
print(sim_summary)
cat("\n-- TC-W2V coherence --\n")
cat("No explicitly documented Word2Vec coherence model/resource found in this project (repo-wide\n",
    "search for word2vec/w2v/gensim/glove/O'Callaghan/coherence returned nothing). TC-W2V coherence\n",
    "is therefore PENDING an analyst choice of reference Word2Vec resource and was NOT computed.\n", sep="")
cat("\n===== DONE =====\n")
