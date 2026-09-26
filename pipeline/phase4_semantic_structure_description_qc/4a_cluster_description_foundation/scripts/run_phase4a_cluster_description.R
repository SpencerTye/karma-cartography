#!/usr/bin/env Rscript
# Phase 4A: cluster-description foundation.
# Joins the 5,167 assigned HDBSCAN clusters (Phase 3) to the Phase 1 records and writes a cluster
# inventory, top-10 c-TF-IDF unigram/bigram descriptors, five centroid-nearest and five
# highest-membership representative documents per cluster (with in-dataset parent context), and
# centroids. Noise (label -1) is excluded. Outputs go to derived/ and figures/; the run is deterministic.
# Usage: Rscript run_phase4a_cluster_description.R [--quick]   (--quick = smoke run on a few clusters)
# Env: KC_REPO_ROOT (default: working directory).

suppressWarnings(suppressMessages({
  library(nanoparquet)
  library(data.table)
  library(stringi)
  library(Matrix)
  library(ggplot2)
  library(digest)
  library(scales)
}))

t_start <- Sys.time()
options(width = 200, warn = 1)
data.table::setDTthreads(0L)  # use all cores for data.table

QUICK <- any(commandArgs(trailingOnly = TRUE) == "--quick")
cat("MODE:", if (QUICK) "QUICK (smoke)" else "FULL", "\n")

# ---- paths ----------------------------------------------------------------------------
BASE <- Sys.getenv("KC_REPO_ROOT", unset = getwd())
PIPE <- file.path(BASE, "pipeline")
P1_ME    <- file.path(PIPE, "phase1_message_event_foundation/derived/message_events.parquet")
P2_NPY   <- file.path(PIPE, "phase2_semantic_embeddings/full_corpus_embedding_0p6b/results/embeddings.npy")
P2_IDX   <- file.path(PIPE, "phase2_semantic_embeddings/full_corpus_embedding_0p6b/results/embedding_index.parquet")
P3_LAB   <- file.path(PIPE, "phase3_blind_semantic_structure_discovery/full_corpus_gpu_umap_hdbscan_baseline/results/hdbscan_labels.parquet")
SW_XLSX  <- file.path(BASE, "r_analysis_outputs/stopword_lists/curation/stopword_curation_tier2_cumulative_310_words.xlsx")

OUT      <- file.path(PIPE, "phase4_semantic_structure_description_qc/4a_cluster_description_foundation")
D_DER    <- file.path(OUT, "derived")
D_FIG    <- file.path(OUT, "figures")
dir.create(D_DER, recursive = TRUE, showWarnings = FALSE)
dir.create(D_FIG, recursive = TRUE, showWarnings = FALSE)

# ---- expected Phase 3 measurements (consistency checks, not re-validation) -------------
EXP_TOTAL    <- 902144L
EXP_CLUSTERS <- 5167L
EXP_ASSIGNED <- 457880L
EXP_NOISE    <- 444264L
EMB_DIM      <- 1024L
NPY_HDR      <- 128L   # verified: 10-byte preamble + 118-byte header dict = 128

# ---- profanity additions (appended to the Tier 2 list) ---------------
PROFANITY <- c(
  "fuck","fucks","fucked","fucking","fuckin","fucker","fuckers",
  "shit","shits","shitting","shitty","bullshit",
  "bitch","bitches","bitching","bastard","bastards","asshole","assholes",
  "damn","damned","dammit","goddamn","goddamned",
  "dick","dicks","dickhead","dickheads","cunt","cunts",
  "motherfucker","motherfuckers","motherfucking","piss","pissed","pissing",
  "prick","pricks","crap","crappy"
)

WPAT <- "\\w\\w+"  # sklearn CountVectorizer default token pattern semantics: runs of 2+ word chars

# small timing/memory helper --------------------------------------------------------------
STAGE <- data.table(stage = character(0), seconds = numeric(0), peak_gb = numeric(0))
timeit <- function(label, expr) {
  invisible(gc(reset = TRUE))
  t0 <- Sys.time()
  val <- force(expr)
  el <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  g <- gc()
  peak_gb <- (g["Vcells", "max used"] * 8 + g["Ncells", "max used"] * 56) / 2^30
  cat(sprintf("[stage] %-28s %8.1f s   peak ~%.2f GB\n", label, el, peak_gb))
  STAGE <<- rbind(STAGE, data.table(stage = label, seconds = el, peak_gb = peak_gb))
  val
}

# =====================================================================================
# Helper: sklearn-equivalent tokenizer + n-gram / stopword semantics
# =====================================================================================
# sklearn CountVectorizer(lowercase=TRUE, token_pattern=r"(?u)\b\w\w+\b",
#   stop_words=<list>, ngram_range=(1,2)) processes each document as:
#   1. lowercase;
#   2. tokenize into maximal runs of >=2 word chars;
#   3. remove any token equal to a stop word (raw-string comparison to the token);
#   4. build unigrams (= remaining tokens) and bigrams (= adjacent pairs in the
#      stop-word-filtered token stream, so bigrams bridge a removed stop word and no
#      bigram ever contains one).
# STOPSET is assigned after the stopword list is built (below).

# Reference implementation (one document) -- used only for unit tests / equivalence check.
sklearn_terms_ref <- function(x, stopset) {
  toks <- stringi::stri_extract_all_regex(stringi::stri_trans_tolower(x), WPAT,
                                          omit_no_match = TRUE)[[1]]
  toks <- toks[!(toks %in% stopset)]
  uni <- toks
  bi  <- if (length(toks) >= 2L) paste(toks[-length(toks)], toks[-1L]) else character(0)
  c(uni, bi)
}

# Vectorized extraction over many documents -> long (doc, term) one row per occurrence.
extract_terms_long <- function(texts, stopset) {
  lc  <- stringi::stri_trans_tolower(texts)
  tl  <- stringi::stri_extract_all_regex(lc, WPAT, omit_no_match = TRUE)
  lens <- lengths(tl)
  DT <- data.table(doc = rep.int(seq_along(texts), lens),
                   tok = unlist(tl, use.names = FALSE))
  # order is doc-ascending then within-document token position (rep + unlist preserve it)
  if (nrow(DT)) DT <- DT[!(tok %chin% stopset)]      # step 3: remove stop-word tokens
  if (!nrow(DT)) return(data.table(doc = integer(0), term = character(0)))
  DT[, nxt := shift(tok, type = "lead"), by = doc]   # next kept token within the doc
  uni <- DT[, .(doc, term = tok)]
  bi  <- DT[!is.na(nxt), .(doc, term = paste(tok, nxt))]
  rbindlist(list(uni, bi))
}

# =====================================================================================
# SECTION 0. Tokenizer unit tests (hand-computed against the sklearn algorithm)
# =====================================================================================
run_tokenizer_tests <- function() {
  ss <- c("the", "a", "is", "of", "to", "and", "have", "don't", "it's")
  chk <- function(x, exp) {
    got <- sklearn_terms_ref(x, ss)
    if (!identical(sort(got), sort(exp)))
      stop(sprintf("tokenizer test FAILED for '%s'\n  got: %s\n  exp: %s",
                   x, paste(got, collapse="|"), paste(exp, collapse="|")))
  }
  # 1 basic stopword removal + bigrams from remaining tokens
  chk("The quick brown fox",
      c("quick","brown","fox","quick brown","brown fox"))
  # 2 bigram bridges across two removed stopwords ("and","the")
  chk("cats and the dogs", c("cats","dogs","cats dogs"))
  # 3 apostrophe: "don't" -> token "don" (t dropped); stopword "don't" != "don" so inert
  chk("don't stop", c("don","stop","don stop"))
  # 4 single chars and single digits dropped; multi-digit kept
  chk("I have 3 cats 42 a b", c("cats","42","cats 42"))
  # 5 punctuation stripped
  chk("hello, world!", c("hello","world","hello world"))
  # 6 no word>=2 chars -> empty
  chk("!!! . , 1 x", character(0))
  # 7 underscore is a word char (kept inside a token)
  chk("foo_bar baz", c("foo_bar","baz","foo_bar baz"))
  # 8 "it's" -> "it" (s dropped); "it" not in stopset here so kept
  chk("it's fine", c("it","fine","it fine"))
  cat("[ok] tokenizer unit tests passed (8 cases)\n")
}

# =====================================================================================
# SECTION 1. Load HDBSCAN labels + embedding index; verify alignment and counts.
# =====================================================================================
lab <- timeit("load labels+index", {
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

n_total    <- nrow(lab)
is_assigned <- lab$label != -1L
n_assigned <- sum(is_assigned)
n_noise    <- sum(!is_assigned)
clusters_present <- sort(unique(lab$label[is_assigned]))
n_clusters <- length(clusters_present)

cat(sprintf("consistency: total=%d (exp %d) assigned=%d (exp %d) noise=%d (exp %d) clusters=%d (exp %d)\n",
            n_total, EXP_TOTAL, n_assigned, EXP_ASSIGNED, n_noise, EXP_NOISE, n_clusters, EXP_CLUSTERS))
stopifnot(n_total == EXP_TOTAL, n_assigned == EXP_ASSIGNED,
          n_noise == EXP_NOISE, n_clusters == EXP_CLUSTERS,
          identical(clusters_present, 0:(EXP_CLUSTERS - 1L)))
cat("[ok] input consistency checks pass\n")

# =====================================================================================
# SECTION 2. Build the effective Phase 4 stopword list (Tier 2 + profanity) and save it.
# =====================================================================================
tier2 <- readxl::read_excel(SW_XLSX, sheet = 1)$word
stopifnot(length(tier2) == 310L, !anyNA(tier2), all(tier2 == tolower(tier2)))
tier2 <- tolower(tier2)
prof_to_add <- setdiff(PROFANITY, tier2)                    # profanity not already present
STOPSET <- unique(c(tier2, prof_to_add))
eff_sw <- data.table(
  word   = STOPSET,
  source = c(rep("tier2", length(tier2)), rep("profanity_added", length(prof_to_add)))
)
# "reachable" = able to match a \w\w+ token at all (no apostrophe/space, length>=2, lowercase [a-z0-9_])
eff_sw[, reachable := stringi::stri_detect_regex(word, "^\\w\\w+$") &
                      !stringi::stri_detect_regex(word, "[^\\w]")]
nanoparquet::write_parquet(eff_sw, file.path(D_DER, "effective_stopwords_phase4.parquet"),
                           compression = "zstd", options = parquet_options(compression_level = 3L))
fwrite(eff_sw, file.path(D_DER, "effective_stopwords_phase4.csv"))
cat(sprintf("stopwords: tier2=%d + profanity_added=%d = effective=%d (reachable under \\w\\w+ = %d)\n",
            length(tier2), length(prof_to_add), nrow(eff_sw), sum(eff_sw$reachable)))

run_tokenizer_tests()

# =====================================================================================
# SECTION 3. Load canonical message_events (only the columns Phase 4A needs).
# =====================================================================================
me <- timeit("load message_events", {
  m <- nanoparquet::read_parquet(P1_ME)
  keep <- c("doc_key","record_type","author","created_utc","subreddit","thread_id",
            "parent_id","parent_kind","parent_in_dataset","text")
  m <- as.data.table(m[, keep])
  setkey(m, doc_key)
  m
})
stopifnot(nrow(me) == n_total, !any(duplicated(me$doc_key)))

# =====================================================================================
# SECTION 4. Assemble the assigned-document working table (ad).
#   ad has one row per assigned document, carrying the cluster label, membership
#   probability, embedding row index, and canonical text. In QUICK mode ad is a small
#   subset spanning a few clusters so every downstream code path still executes.
# =====================================================================================
ad0 <- lab[is_assigned, .(doc_key, row0 = row, cluster = label, probability)]
ad  <- me[ad0, on = "doc_key"]         # attach record_type + text + metadata from canonical me
stopifnot(nrow(ad) == n_assigned, !anyNA(ad$cluster), !anyNA(ad$record_type))

if (QUICK) {
  keep_cl <- c(head(clusters_present, 5L),
               ad[, .N, by = cluster][order(-N)][1L, cluster])   # 5 small + the largest
  keep_cl <- sort(unique(keep_cl))
  ad <- ad[cluster %in% keep_cl]
  cat(sprintf("[quick] restricted to %d clusters (%s), %d docs\n",
              length(keep_cl), paste(keep_cl, collapse=","), nrow(ad)))
}
ad_clusters <- sort(unique(ad$cluster))
n_ad_clusters <- length(ad_clusters)

# =====================================================================================
# SECTION 5. c-TF-IDF descriptors (BERTopic ClassTfidfTransformer, default settings).
#   tf'_{x,c} = count(x in class c) / (total term tokens in class c)      [L1 per class]
#   idf_x     = log(1 + A / f_x),  A = (grand total term tokens)/(#classes),
#               f_x = total count of term x across all classes.
#   score     = tf'_{x,c} * idf_x.  No BM25, no seed weighting, no reduce-frequent-words,
#   no min_df/max_df, no extra vocabulary exclusions. Unigrams and bigrams share one
#   vocabulary and both count toward the class token total (ngram_range = (1,2)).
# =====================================================================================
cls <- timeit("c-tf-idf: tokenize+count", {
  CLV <- ad$cluster                                     # cluster per doc index (1..nrow(ad))
  # --- equivalence check: vectorized pipeline vs the reference tokenizer on a sample ---
  ns <- min(1000L, nrow(ad))
  samp <- if (ns == nrow(ad)) seq_len(ns) else round(seq(1, nrow(ad), length.out = ns))
  ref <- rbindlist(lapply(samp, function(d)
           data.table(doc = d, term = sklearn_terms_ref(ad$text[d], STOPSET))))
  refc <- ref[, .N, by = .(doc, term)][order(doc, term)]
  pipe_s <- extract_terms_long(ad$text[samp], STOPSET)
  pipe_s[, doc := samp[doc]]
  pipec <- pipe_s[, .N, by = .(doc, term)][order(doc, term)]
  if (!isTRUE(all.equal(refc, pipec)))
    stop("pipeline vs reference tokenizer MISMATCH on the equivalence sample")
  cat(sprintf("[ok] tokenizer equivalence check passed on %d sampled docs\n", ns))

  # --- full extraction and aggregation to (cluster, term) counts ---
  lng <- extract_terms_long(ad$text, STOPSET)
  lng[, cl := CLV[doc]]
  x <- lng[, .(n = .N), by = .(cl, term)]               # class-term count matrix X (long)
  rm(lng); gc()
  x
})

top_terms <- timeit("c-tf-idf: score+rank", {
  x <- copy(cls)
  x[, class_total := sum(n), by = cl]                   # tokens per class (uni+bi)
  x[, f_x := sum(n), by = term]                         # global term frequency across classes
  grand <- sum(x$n)
  n_classes_terms <- uniqueN(x$cl)   # classes with >= 1 term after stopword removal
  n_classes <- n_ad_clusters         # all assigned clusters are classes (incl. any 0-term class)
  A <- grand / n_classes             # average number of term tokens per class (BERTopic X.shape[0])
  x[, tf := n / class_total]
  x[, idf := log(1 + A / f_x)]
  x[, ctfidf := tf * idf]
  x[, ngram := fifelse(stringi::stri_detect_fixed(term, " "), 2L, 1L)]
  # deterministic ranking: score desc, then raw count desc, then term asc (unique within cl)
  setorder(x, cl, -ctfidf, -n, term)
  x[, rank := rowid(cl)]
  tt <- x[rank <= 10L]
  attr(tt, "A") <- A; attr(tt, "grand") <- grand
  attr(tt, "n_classes") <- n_classes; attr(tt, "n_classes_terms") <- n_classes_terms
  tt
})

ctfidf_out <- top_terms[, .(cluster_id = cl, rank, term, ngram, ctfidf, term_count = n)]
nanoparquet::write_parquet(ctfidf_out, file.path(D_DER, "ctfidf_top_terms.parquet"),
                           compression = "zstd", options = parquet_options(compression_level = 3L))

desc_per_cluster <- ctfidf_out[, .(k = .N), by = cluster_id]
clusters_lt10 <- desc_per_cluster[k < 10L]
cat(sprintf("c-tf-idf: %d descriptor rows for %d clusters; clusters with <10 descriptors = %d\n",
            nrow(ctfidf_out), uniqueN(ctfidf_out$cluster_id), nrow(clusters_lt10)))

# =====================================================================================
# SECTION 6. Cluster centroids (original 1024-d Phase 2 embeddings) + centroid_nearest.
#   Centroid = mean embedding of the cluster's assigned docs (persisted, unnormalized).
#   Cosine to centroid uses the L2-normalized centroid; document embeddings are already
#   unit-norm (verified), so cosine = normalized-centroid . document.
# =====================================================================================
read_npy_rows <- function(path, rows0, dim = EMB_DIM, hdr = NPY_HDR) {
  con <- file(path, "rb"); on.exit(close(con))
  out <- matrix(NA_real_, nrow = dim, ncol = length(rows0))     # column j = doc j
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
  dim(v) <- c(dim, ntot)                                         # in place; column j = global row j-1
  v
}

emb <- timeit("load embeddings", {
  if (QUICK) {
    read_npy_rows(P2_NPY, ad$row0)                               # 1024 x nrow(ad)
  } else {
    Mall <- read_npy_all(P2_NPY, n_total)                        # 1024 x 902144
    E <- Mall[, ad$row0 + 1L, drop = FALSE]                      # align columns to ad rows
    rm(Mall); gc()
    E
  }
})
stopifnot(ncol(emb) == nrow(ad), nrow(emb) == EMB_DIM)
# validate embedding alignment against the per-row seek reader (cheap; in full mode
# this catches any whole-file read / column-subset misalignment)
chk_idx <- unique(round(seq(1, nrow(ad), length.out = min(6L, nrow(ad)))))
stopifnot(max(abs(emb[, chk_idx, drop = FALSE] - read_npy_rows(P2_NPY, ad$row0[chk_idx]))) < 1e-6)
# sanity: document embeddings are unit-norm
cn_check <- sqrt(colSums(emb[, seq_len(min(200L, ncol(emb))), drop = FALSE]^2))
cat(sprintf("embedding column-norm sample: min=%.6f max=%.6f\n", min(cn_check), max(cn_check)))

centroids_cosine <- timeit("centroids + cosine", {
  col_by_cluster <- split(seq_len(nrow(ad)), ad$cluster)        # doc-column indices per cluster
  cl_ids <- as.integer(names(col_by_cluster))
  cent <- matrix(NA_real_, nrow = EMB_DIM, ncol = length(cl_ids))  # mean centroids (unnormalized)
  cosv <- numeric(nrow(ad))                                     # cosine of each doc to its centroid
  for (k in seq_along(cl_ids)) {
    cols <- col_by_cluster[[k]]
    sub <- emb[, cols, drop = FALSE]
    ctr <- rowMeans(sub)
    cent[, k] <- ctr
    nrm <- sqrt(sum(ctr^2))
    cn <- if (nrm > 0) ctr / nrm else ctr
    cosv[cols] <- as.numeric(crossprod(cn, sub))               # docs unit-norm => cosine
  }
  list(cl_ids = cl_ids, cent = cent, cosv = cosv)
})
ad[, cosine_to_centroid := centroids_cosine$cosv]

# persist centroids (cluster_id + v0..v1023) with an explicit cluster-ID mapping column
cent_dt <- as.data.table(t(centroids_cosine$cent))
setnames(cent_dt, paste0("v", 0:(EMB_DIM - 1L)))
cent_dt <- cbind(cluster_id = centroids_cosine$cl_ids, cent_dt)
setorder(cent_dt, cluster_id)
nanoparquet::write_parquet(cent_dt, file.path(D_DER, "cluster_centroids.parquet"),
                           compression = "zstd", options = parquet_options(compression_level = 3L))
cat(sprintf("centroids: %d x %d persisted\n", nrow(cent_dt), EMB_DIM))

# =====================================================================================
# SECTION 7. Representative-document selection (deterministic; overlap preserved).
#   centroid_nearest : top-5 by cosine to centroid, ties -> doc_key ascending.
#   highest_membership: top-5 by HDBSCAN probability, ties -> doc_key ascending.
# =====================================================================================
pick_top5 <- function(dt, score_col, method) {
  o <- if (score_col == "cosine_to_centroid")
         order(-dt$cosine_to_centroid, dt$doc_key)
       else
         order(-dt$probability, dt$doc_key)
  d2 <- dt[o]
  d2[, rank := rowid(cluster)]
  sel <- d2[rank <= 5L]
  sel[, method := method]
  sel
}
rep_cn <- pick_top5(ad, "cosine_to_centroid", "centroid_nearest")
rep_hm <- pick_top5(ad, "probability", "highest_membership")
reps <- rbindlist(list(rep_cn, rep_hm), use.names = TRUE)
# cosine is the selection metric only for centroid_nearest ("where applicable")
reps[method == "highest_membership", cosine_to_centroid := NA_real_]

# ---- parent reconnection for representative comments -------------------------------------
fmt_utc <- function(x) format(as.POSIXct(x, origin = "1970-01-01", tz = "UTC"),
                              "%Y-%m-%d %H:%M:%S UTC")
reps[, is_comment := record_type == "comment"]
# resolve the direct parent only for representative comments, and only when the parent_id
# is present as a doc_key in the canonical dataset (match() -> NA otherwise). Parent columns
# stay NA for submissions and for comments whose parent is not in-dataset.
plk <- fifelse(reps$is_comment, reps$parent_id, NA_character_)
mi  <- match(plk, me$doc_key)                       # index into me; NA when parent not in-dataset
reps[, parent_doc_key     := me$doc_key[mi]]
reps[, parent_record_type := me$record_type[mi]]
reps[, parent_author      := me$author[mi]]
reps[, parent_created_utc := me$created_utc[mi]]
reps[, parent_text        := me$text[mi]]

rep_out <- reps[, .(
  cluster_id = cluster, method, rank, doc_key, record_type, author,
  created_utc, created_datetime_utc = fmt_utc(created_utc),
  subreddit, thread_id, membership_probability = probability, cosine_to_centroid,
  text,
  parent_id, parent_kind, parent_in_dataset,
  parent_doc_key, parent_record_type, parent_author,
  parent_created_utc, parent_created_datetime_utc = fmt_utc(parent_created_utc),
  parent_text
)]
setorder(rep_out, cluster_id, method, rank)
nanoparquet::write_parquet(rep_out, file.path(D_DER, "representative_documents.parquet"),
                           compression = "zstd", options = parquet_options(compression_level = 3L))

# ---- representative-selection diagnostics -----------------------------------------------
reps_per <- rep_out[, .N, by = .(cluster_id, method)]
lt5_cn <- reps_per[method == "centroid_nearest"   & N < 5L]
lt5_hm <- reps_per[method == "highest_membership" & N < 5L]
rc <- rep_out[record_type == "comment"]
rc_rows <- nrow(rc)
rc_resolved <- sum(!is.na(rc$parent_doc_key))
rc_flag_in <- sum(rc$parent_in_dataset %in% TRUE)
cat(sprintf("representatives: centroid_nearest=%d rows, highest_membership=%d rows (total %d)\n",
            sum(rep_out$method == "centroid_nearest"), sum(rep_out$method == "highest_membership"),
            nrow(rep_out)))
cat(sprintf("rep comments: %d rows; parent resolved in-dataset=%d (%.2f%%); parent_in_dataset flag=TRUE=%d\n",
            rc_rows, rc_resolved, 100 * rc_resolved / max(rc_rows, 1L), rc_flag_in))
cat(sprintf("clusters with <5 centroid_nearest=%d; <5 highest_membership=%d\n",
            nrow(lt5_cn), nrow(lt5_hm)))

# =====================================================================================
# SECTION 8. Basic cluster inventory (factual counts + probability distribution).
# =====================================================================================
inv <- ad[, .(
  n_docs        = .N,
  n_comments    = sum(record_type == "comment"),
  n_submissions = sum(record_type == "submission"),
  prob_mean     = mean(probability),
  prob_median   = median(probability),
  prob_min      = min(probability),
  prob_max      = max(probability),
  prob_p25      = quantile(probability, 0.25, names = FALSE, type = 7),
  prob_p75      = quantile(probability, 0.75, names = FALSE, type = 7)
), by = .(cluster_id = cluster)]
setorder(inv, cluster_id)
stopifnot(inv$n_docs == inv$n_comments + inv$n_submissions)
nanoparquet::write_parquet(inv, file.path(D_DER, "cluster_inventory.parquet"),
                           compression = "zstd", options = parquet_options(compression_level = 3L))
cat(sprintf("inventory: %d clusters; size min=%d median=%.0f max=%d (sum=%d)\n",
            nrow(inv), min(inv$n_docs), median(inv$n_docs), max(inv$n_docs), sum(inv$n_docs)))

# =====================================================================================
# SECTION 9. Figures (title + axis labels only; no captions/subtitles/annotations).
# =====================================================================================
p1 <- ggplot(inv, aes(x = n_docs)) +
  geom_histogram(bins = 40, fill = "grey30") +
  scale_x_log10(labels = scales::label_comma()) +
  labs(title = "Cluster size distribution",
       x = "Documents per cluster (log10 scale)", y = "Number of clusters") +
  theme_minimal(base_size = 12)
ggsave(file.path(D_FIG, "fig_cluster_size_distribution.png"), p1,
       width = 7, height = 5, dpi = 150)

p2 <- ggplot(ad, aes(x = probability)) +
  geom_histogram(bins = 50, fill = "grey30") +
  scale_y_continuous(labels = scales::label_comma()) +
  labs(title = "HDBSCAN membership probability distribution",
       x = "Membership probability", y = "Number of assigned documents") +
  theme_minimal(base_size = 12)
ggsave(file.path(D_FIG, "fig_membership_probability_distribution.png"), p2,
       width = 7, height = 5, dpi = 150)
cat("figures: 2 written\n")

# =====================================================================================
# SECTION 10. Provenance + run statistics (printed; folded into REPORT.md by the analyst).
# =====================================================================================
sha <- function(p, do = TRUE) if (do) digest::digest(p, algo = "sha256", file = TRUE) else "(skipped)"
cat("\n===== PROVENANCE (sha256 of inputs) =====\n")
cat("message_events.parquet :", sha(P1_ME), "\n")
cat("embedding_index.parquet:", sha(P2_IDX), "\n")
cat("hdbscan_labels.parquet :", sha(P3_LAB), "\n")
cat("stopword_tier2.xlsx    :", sha(SW_XLSX), "\n")
cat("embeddings.npy         :", sha(P2_NPY, do = !QUICK), "\n")

cat("\n===== STAGE TIMING / PEAK =====\n")
print(STAGE)
cat(sprintf("TOTAL runtime: %.1f s\n", as.numeric(difftime(Sys.time(), t_start, units = "secs"))))
g <- gc()
cat(sprintf("overall peak (gc max used) ~ %.2f GB\n",
            (g["Vcells","max used"] * 8 + g["Ncells","max used"] * 56) / 2^30))

cat("\n===== KEY NUMBERS (for REPORT.md) =====\n")
cat(sprintf("MODE=%s\n", if (QUICK) "QUICK" else "FULL"))
cat(sprintf("total_docs=%d assigned=%d noise=%d clusters=%d\n",
            n_total, n_assigned, n_noise, n_clusters))
cat(sprintf("described_clusters=%d described_docs=%d\n", n_ad_clusters, nrow(ad)))
cat(sprintf("ctfidf_rows=%d ctfidf_clusters=%d clusters_lt10_desc=%d\n",
            nrow(ctfidf_out), uniqueN(ctfidf_out$cluster_id), nrow(clusters_lt10)))
{
  sz_all    <- ad[, .(size = .N), by = .(cluster_id = cluster)]
  zero_term <- setdiff(ad_clusters, unique(ctfidf_out$cluster_id))  # clusters with 0 descriptors
  low_all   <- union(zero_term, clusters_lt10$cluster_id)           # all clusters with <10 descriptors
  adx <- ad[cluster %in% low_all, .(n_distinct_text = uniqueN(text)), by = .(cluster_id = cluster)]
  adx <- merge(adx, sz_all, by = "cluster_id")
  cat(sprintf("descriptors: clusters_eq10=%d clusters_1to9=%d clusters_0=%d total_lt10=%d\n",
      sum(desc_per_cluster$k == 10L), nrow(clusters_lt10), length(zero_term), length(low_all)))
  cat(sprintf("lt10 clusters: size min=%d median=%.0f max=%d ; distinct_text median=%.0f max=%d ; pct_le3_distinct=%.1f ; exactly_1_desc=%d\n",
      min(adx$size), median(adx$size), max(adx$size),
      median(adx$n_distinct_text), max(adx$n_distinct_text),
      100 * mean(adx$n_distinct_text <= 3), nrow(desc_per_cluster[k == 1L])))
}
cat(sprintf("ctfidf_A=%.4f grand_tokens=%.0f n_classes=%d n_classes_with_terms=%d\n",
            attr(top_terms,"A"), attr(top_terms,"grand"),
            attr(top_terms,"n_classes"), attr(top_terms,"n_classes_terms")))
cat(sprintf("rep_centroid_nearest=%d rep_highest_membership=%d rep_total=%d\n",
            sum(rep_out$method=="centroid_nearest"), sum(rep_out$method=="highest_membership"), nrow(rep_out)))
cat(sprintf("clusters_lt5_cn=%d clusters_lt5_hm=%d\n", nrow(lt5_cn), nrow(lt5_hm)))
cat(sprintf("rep_comment_rows=%d rep_comment_parent_resolved=%d pct=%.2f\n",
            rc_rows, rc_resolved, 100 * rc_resolved / max(rc_rows, 1L)))
cat(sprintf("cluster_size_min=%d median=%.1f max=%d\n",
            min(inv$n_docs), median(inv$n_docs), max(inv$n_docs)))
cat(sprintf("stopwords_effective=%d tier2=%d profanity_added=%d reachable=%d\n",
            nrow(eff_sw), length(tier2), length(prof_to_add), sum(eff_sw$reachable)))
cat("\n===== DONE =====\n")
