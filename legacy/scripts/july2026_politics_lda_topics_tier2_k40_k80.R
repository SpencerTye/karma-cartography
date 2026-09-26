# July 2026 r/politics sample — Phase 1A LDA, Tier 2 stopword condition:
# K = 40 promoted to retained-output status, and the K = 40 versus K = 80 comparison package
#
# The delivered Tier 2 run (july2026_politics_lda_topics_tier2.R, outputs under
# r_analysis_outputs/lda_topics_tier2/) retained K = 10 and K = 80 by its rule and
# wrote the full document-level tables for them only. The user has since selected
# K = 40 and K = 80 as the two candidate representations to carry forward. This
# companion script
#   (1) produces for K = 40 exactly the retained-model outputs the delivered run
#       produced for K = 80 (the same thirteen tables and nine figures, the same
#       arithmetic, the same file names with the K040 tag), from the cached
#       K = 40 fits in lda_topics_tier2/models/ — nothing is re-fitted;
#   (2) proves the arithmetic by regenerating every delivered K = 80 retained
#       table in a scratch folder and matching the SHA256 recorded in each
#       delivered manifest (nothing under lda_topics_tier2/ is written);
#   (3) compares the two resolutions on the delivered measured evidence and on
#       measured topic-to-topic correspondence (cosine of smoothed topic-word
#       distributions, the quantity the delivered stability matching uses), with
#       the AutoModerator civility-notice copies quantified in both;
#   (4) writes inspection tables (top words with weights, representative
#       documents with identifiers, topic probabilities) for every correspondence.
# Every output goes under r_analysis_outputs/lda_topics_tier2_k40_k80/. The
# delivered outputs, the model cache and the source Parquet are read-only and
# untouched; the main script's own folders are never written to (it clears its
# tables/ and figures/ on every run, so K = 40 files placed there would not
# survive a re-run). Topics are identified only by number; no topic is named or
# interpreted, and no choice between K = 40 and K = 80 is made here.
#
# The text-processing, split, retained-model and figure code below is copied
# from the delivered script so that the K = 40 outputs are built by the same
# arithmetic; the reference-only counts of the delivered tokeniser (case-preserved
# tokens, SMART and Snowball occurrence counts) are omitted because they do not
# enter any output here. Section headers name what each block reads and writes.

# ---- Setup: packages, paths, parameters, output directories ----
# INPUT : none.
# DOES  : load packages; define the source paths, the delivered run's analytical
#         parameters (unchanged) and this package's correspondence thresholds;
#         create the output folder and clear this script's own outputs.
# OUTPUT: parameter objects; figures/ and tables/ under the package folder.
suppressPackageStartupMessages({
  library(nanoparquet); library(dplyr); library(tidyr); library(stringi); library(Matrix)
  library(clue); library(readxl); library(ggplot2); library(scales); library(readr); library(digest); library(yaml)
})
startup_elapsed_seconds <- proc.time()[["elapsed"]]
run_start     <- Sys.time()
stage_log     <- list()
stage_cursor  <- run_start
mark_stage    <- function(name) {
  now <- Sys.time()
  stage_log[[length(stage_log) + 1L]] <<- tibble(stage = name, start_utc = stage_cursor, end_utc = now,
                                                 seconds = as.numeric(difftime(now, stage_cursor, units = "secs")))
  stage_cursor <<- now
}
log_line <- function(...) { cat(format(Sys.time(), "%H:%M:%S"), ..., "\n"); flush.console() }

project_dir       <- Sys.getenv("KC_REPO_ROOT", unset = getwd())
comments_dir      <- file.path(project_dir, "data_sample/comments/2026-07")
submissions_dir   <- file.path(project_dir, "data_sample/submissions/2026-07")
script_path       <- file.path(project_dir, "july2026_politics_lda_topics_tier2_k40_k80.R")
main_script_path  <- file.path(project_dir, "july2026_politics_lda_topics_tier2.R")
tier2_dir         <- file.path(project_dir, "r_analysis_outputs/lda_topics_tier2")     # delivered run (read-only)
tier2_tab_dir     <- file.path(tier2_dir, "tables")
model_dir         <- file.path(tier2_dir, "models")                                     # cached fits (read-only)
stopword_workbook <- file.path(project_dir, "r_analysis_outputs/stopword_lists/curation/stopword_curation_tier2_cumulative_310_words.xlsx")
stopword_sheet    <- "tier2_cumulative_310"
out_dir   <- file.path(project_dir, "r_analysis_outputs/lda_topics_tier2_k40_k80")
fig_dir   <- file.path(out_dir, "figures")
tab_dir   <- file.path(out_dir, "tables")
scratch_dir <- file.path(Sys.getenv("LDA_K40_SCRATCH", tempdir()), "lda_tier2_k80_reproduction")
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(tab_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(scratch_dir, recursive = TRUE, showWarnings = FALSE)
# This script owns the package folder only; its previous outputs are removed so no stale file survives.
invisible(file.remove(list.files(c(fig_dir, tab_dir), pattern = "\\.(csv|png|yml)$", full.names = TRUE)))
invisible(file.remove(list.files(scratch_dir, pattern = "\\.csv$", full.names = TRUE)))

# Analytical parameters of the delivered run (unchanged; they reproduce the documents, split and tables).
candidate_k                <- c(10L, 20L, 30L, 40L, 50L, 60L, 80L, 100L)
fit_seeds                  <- c(1L, 2L, 3L)
alpha_sum                  <- 5
beta                       <- 0.01
n_iter_max                 <- 1000L
convergence_tol            <- 1e-4
n_check_convergence        <- 10L
n_iter_inference           <- 10L
heldout_share              <- 0.05
split_seed                 <- 20260731L
min_token_chars            <- 2L
min_document_frequency     <- 5L
max_document_share         <- 0.5
stopword_source            <- "tier2_cumulative_310"
n_top_words                <- 20L
coherence_top_n            <- 10L
relevance_lambda           <- 0.6
n_representative           <- 5L
representative_min_tokens  <- 25L
active_topic_threshold     <- 0.1
stability_cosine_threshold <- 0.8
# This package's settings.
selected_k              <- c(40L, 80L)   # the two user-selected candidate representations
k_new                   <- 40L           # outputs written here
k_reference             <- 80L           # delivered outputs reproduced in scratch and hash-checked, not written
correspondence_threshold <- 0.5          # a K = 80 topic corresponds to a K = 40 topic at or above this cosine
clear_threshold         <- stability_cosine_threshold   # a correspondence is "clear" at or above this cosine (the delivered stability threshold)
notice_anchor           <- list(K = 10L, seed = 1L, topic = 10L)   # the delivered README's AutoModerator civility-notice topic
notice_threshold        <- 0.9           # a topic is a notice copy at or above this cosine to the anchor (every cosine is reported)
repetitive_ttr          <- 0.3           # type-token ratio below which a representative document is flagged as repetitive

comment_files    <- list.files(comments_dir,    pattern = "\\.parquet$", full.names = TRUE)
submission_files <- list.files(submissions_dir, pattern = "\\.parquet$", full.names = TRUE)
sha256 <- function(path) paste0("sha256:", digest(path, algo = "sha256", file = TRUE))
log_line("start; selected K =", paste(selected_k, collapse = ", "), "; writing K =", k_new)

# ---- Stopword list: the Tier 2 workbook (read-only) ----
# INPUT : the curation workbook.
# DOES  : read the 'word' column of the Tier 2 sheet exactly as stored (the
#         delivered rule); record its SHA256.
# OUTPUT: stopword_list (character, 310); workbook_hash.
workbook_hash <- sha256(stopword_workbook)
tier2_raw     <- read_excel(stopword_workbook, sheet = stopword_sheet, col_types = "text")
stopword_list <- tier2_raw$word
token_regex   <- "\\p{L}+(?:'\\p{L}+)*"
log_line("workbook read:", length(stopword_list), "entries; hash", workbook_hash)

# ---- Read source data (read-only) ----
# INPUT : 7 comment Parquet files + 1 submission Parquet file.
# DOES  : read and stack; keep the columns used (column selection only).
# OUTPUT: comments; submissions.
comments <- lapply(comment_files, read_parquet) |>
  bind_rows() |>
  select(id, link_id, parent_id, author, created_utc, subreddit, body) |>
  mutate(created_utc = as.numeric(created_utc))
submissions <- lapply(submission_files, read_parquet) |>
  bind_rows() |>
  select(id, author, created_utc, subreddit, title, selftext) |>
  mutate(created_utc = as.numeric(created_utc))
n_comments_all    <- nrow(comments)
n_submissions_all <- nrow(submissions)
mark_stage("read")
log_line("read:", n_comments_all, "comments,", n_submissions_all, "submissions")

# ---- Document table (the delivered rule) ----
# INPUT : comments; submissions.
# DOES  : one document per source record; doc_key = Reddit fullname.
# OUTPUT: docs (source order: comments then submissions).
docs <- bind_rows(
  comments |>
    transmute(doc_type = "comment", doc_id = id, doc_key = paste0("t1_", id),
              author, created_utc, subreddit, link_id, parent_id,
              submission_id = substr(link_id, 4L, nchar(link_id)), text = body),
  submissions |>
    transmute(doc_type = "submission", doc_id = id, doc_key = paste0("t3_", id),
              author, created_utc, subreddit, link_id = paste0("t3_", id),
              parent_id = NA_character_, submission_id = id,
              text = paste(title, selftext, sep = " "))) |>
  mutate(source_row           = row_number(),
         submission_in_sample = submission_id %in% submissions$id,
         created_date_utc     = as.Date(as.POSIXct(created_utc, origin = "1970-01-01", tz = "UTC")),
         n_chars              = stri_length(text))
n_docs <- nrow(docs)
mark_stage("document_table")

# ---- Text processing (the delivered steps 1-8; reference-only counts omitted) ----
# INPUT : docs$text.
# DOES  : case folding; U+2019 -> '; URLs removed; letter-run tokens with
#         internal apostrophes; trailing 's stripped; tokens shorter than
#         min_token_chars removed; the 310 Tier 2 entries removed; tokens of
#         removal-marker-only documents excluded.
# OUTPUT: tok / tdoc; per-document token counts on docs.
removal_markers  <- c("[deleted]", "[removed]", "[ Removed by Reddit ]")
docs$marker_only <- stri_trim_both(docs$text) %in% removal_markers
url_regex   <- "(https?://|www\\.)\\S+"
url_opts    <- stri_opts_regex(case_insensitive = TRUE)
text_step <- stri_trans_tolower(docs$text)                                    # step 1
text_step <- stri_replace_all_fixed(text_step, "\u2019", "'")                 # step 2
text_step <- stri_replace_all_regex(text_step, url_regex, " ", opts_regex = url_opts)  # step 3
tokens_list <- stri_extract_all_regex(text_step, token_regex, omit_no_match = TRUE)  # step 4
rm(text_step)
docs$n_tokens_letters <- lengths(tokens_list)
tok  <- unlist(tokens_list, use.names = FALSE)
tdoc <- rep(seq_len(n_docs), docs$n_tokens_letters)
rm(tokens_list)
n_letter_tokens <- length(tok)
is_possessive <- stri_endswith_fixed(tok, "'s")                              # step 5
tok[is_possessive] <- stri_sub(tok[is_possessive], 1L, -3L)
rm(is_possessive)
in_marker_tok <- docs$marker_only[tdoc]
is_short <- stri_length(tok) < min_token_chars                                # step 6
is_stop  <- !is_short & tok %in% stopword_list                                # step 7 (Tier 2)
rm(in_marker_tok)
docs$n_tokens_short <- tabulate(tdoc[is_short], nbins = n_docs)
docs$n_tokens_stop  <- tabulate(tdoc[is_stop],  nbins = n_docs)
keep_tok <- !is_short & !is_stop
tok  <- tok[keep_tok]
tdoc <- tdoc[keep_tok]
rm(is_short, is_stop, keep_tok)
docs$n_tokens_candidate <- tabulate(tdoc, nbins = n_docs)
n_candidate_tokens <- length(tok)
in_marker_doc <- docs$marker_only[tdoc]                                       # step 8
n_tokens_in_marker_docs <- sum(in_marker_doc)
tok  <- tok[!in_marker_doc]
tdoc <- tdoc[!in_marker_doc]
rm(in_marker_doc)
mark_stage("tokenize")
log_line("tokenize:", n_letter_tokens, "letter tokens;", sum(docs$n_tokens_stop), "Tier 2 stopword tokens removed")

# ---- Vocabulary and document-term matrix (the delivered rule) ----
# INPUT : tok / tdoc.
# DOES  : terms kept when min_document_frequency <= df <= max_document_share x
#         non-marker documents; one dtm row per source document.
# OUTPUT: vocabulary; dtm_all; docs$model_status.
vocab_all  <- sort(unique(tok), method = "radix")
term_index <- match(tok, vocab_all)
tf_all <- sparseMatrix(i = tdoc, j = term_index, x = 1, dims = c(n_docs, length(vocab_all)), repr = "C")
rm(tok, tdoc, term_index)
term_df_all      <- diff(tf_all@p)
n_nonmarker_docs <- sum(!docs$marker_only)
term_below_floor   <- term_df_all < min_document_frequency
term_above_ceiling <- term_df_all > max_document_share * n_nonmarker_docs
term_kept        <- !term_below_floor & !term_above_ceiling
vocabulary       <- vocab_all[term_kept]
n_vocabulary     <- length(vocabulary)
dtm_all <- tf_all[, term_kept, drop = FALSE]
dimnames(dtm_all) <- list(docs$doc_key, vocabulary)
rm(tf_all)
docs$n_tokens_modeled <- as.integer(rowSums(dtm_all))
docs <- docs |>
  mutate(model_status = case_when(
    marker_only               ~ "not modeled: removal marker only",
    n_tokens_letters == 0L    ~ "not modeled: no letter tokens",
    n_tokens_candidate == 0L  ~ "not modeled: only single-character tokens or stopwords",
    n_tokens_modeled == 0L    ~ "not modeled: only terms outside the vocabulary (document-frequency floor or ceiling)",
    TRUE                      ~ "modeled"))
mark_stage("vocabulary_dtm")
log_line("dtm:", n_vocabulary, "terms;", sum(docs$model_status == "modeled"), "modeled documents")

# ---- Training / held-out split (the delivered rule) ----
# INPUT : dtm_all; docs.
# DOES  : the seeded 5% held-out sample; modeled_order = training rows then
#         held-out rows (the row order of every cached per-document object).
# OUTPUT: train_rows / heldout_rows; dtm_train; docs$split; modeled_order.
modeled_rows     <- which(docs$model_status == "modeled")
n_modeled        <- length(modeled_rows)
heldout_eligible <- modeled_rows[docs$n_tokens_modeled[modeled_rows] >= 2L]
n_heldout        <- floor(heldout_share * n_modeled)
set.seed(split_seed)
heldout_rows <- sort(sample(heldout_eligible, n_heldout))
train_rows   <- setdiff(modeled_rows, heldout_rows)
n_train      <- length(train_rows)
docs$split   <- NA_character_
docs$split[train_rows]   <- "train"
docs$split[heldout_rows] <- "held-out"
modeled_order <- c(train_rows, heldout_rows)
dtm_train        <- dtm_all[train_rows, , drop = FALSE]
train_doc_tokens <- as.numeric(rowSums(dtm_train))
n_train_tokens   <- sum(train_doc_tokens)
term_frequency_share <- colSums(dtm_train) / n_train_tokens
modeled_doc_key <- docs$doc_key[modeled_order]
modeled_split   <- docs$split[modeled_order]
rm(dtm_all)
mark_stage("split")
log_line("split:", n_train, "training,", n_heldout, "held-out documents")

# ---- Cached fits and delivered tables (read-only) ----
# INPUT : models/fit_K###_seed#.rds for K = 40 and 80 (all seeds), the notice
#         anchor fit, models/theta_K###_seed1.rds for K = 40 and 80; the
#         delivered tables used for comparison and for consistency checks; the
#         manifests of everything read.
# DOES  : load; rebuild smoothed phi from the counts (the delivered arithmetic);
#         record the SHA256 of every file read and compare it with the hash in
#         its delivered manifest (integrity of the inputs).
# OUTPUT: fits (named "K_seed"); anchor_fit; theta_files; delivered (list of
#         tables); input_integrity (one row per file read).
fit_key   <- function(K, seed) paste(K, seed, sep = "_")
fit_path  <- function(K, seed) file.path(model_dir, sprintf("fit_K%03d_seed%d.rds", K, seed))
theta_path <- function(K, seed) file.path(model_dir, sprintf("theta_K%03d_seed%d.rds", K, seed))
finalize_fit <- function(fit) {
  fit$phi <- (fit$components + beta) / (rowSums(fit$components) + ncol(fit$components) * beta)
  fit
}
fits <- list()
for (K in selected_k) for (s in fit_seeds) fits[[fit_key(K, s)]] <- finalize_fit(readRDS(fit_path(K, s)))
anchor_fit <- finalize_fit(readRDS(fit_path(notice_anchor$K, notice_anchor$seed)))
anchor_phi <- anchor_fit$phi[notice_anchor$topic, , drop = FALSE]
theta_files <- setNames(lapply(selected_k, function(K) theta_path(K, fit_seeds[1])), as.character(selected_k))
stopifnot(all(vapply(fits, function(f) identical(colnames(f$components), vocabulary), logical(1))))

delivered_names <- c("stability_pairs", "topic_stability_seed1_all_k", "stability_summary", "model_comparison_by_fit",
                     "model_comparison_summary", "topic_prevalence_all_models", "representative_documents_all_models",
                     "top_words_by_probability_all_models", "top_words_by_relevance_all_models",
                     "topic_pairs_word_distributions_all_models", "topic_pairs_document_scores_all_models",
                     "fit_runtime", "fit_traces", "parameters", "run_info")
delivered_paths <- setNames(file.path(tier2_tab_dir, paste0(delivered_names, ".csv")), delivered_names)
delivered <- lapply(delivered_paths, function(p) read_csv(p, show_col_types = FALSE, progress = FALSE, guess_max = 100000, trim_ws = FALSE))
manifest_hash <- function(path) tryCatch(read_yaml(paste0(path, ".manifest.yml"))$output_hash, error = function(e) NA_character_)
files_read <- c(delivered_paths,
                setNames(vapply(names(fits), function(k) fit_path(fits[[k]]$K, fits[[k]]$seed), ""), paste0("fit_", names(fits))),
                anchor_fit = fit_path(notice_anchor$K, notice_anchor$seed),
                setNames(unlist(theta_files), paste0("theta_", selected_k, "_1")))
input_integrity <- tibble(role = names(files_read), path = unname(files_read)) |>
  mutate(sha256_now = vapply(path, sha256, ""), sha256_in_delivered_manifest = vapply(path, manifest_hash, ""),
         hash_equals_manifest = sha256_now == sha256_in_delivered_manifest)
mark_stage("load_inputs")
log_line("inputs loaded:", length(fits), "fits;", sum(input_integrity$hash_equals_manifest), "of", nrow(input_integrity), "files match their delivered manifests")

# ---- Per-fit word tables (the delivered arithmetic) ----
# INPUT : fits; vocabulary; term_frequency_share.
# DOES  : the top-20 word tables and the ten-word strings used by the tables and
#         figures, for the six fits.
# OUTPUT: top_words_probability; top_words_relevance; paste_top().
top_words_table <- function(fit, idx, lambda) {
  n <- ncol(idx)
  k_rep <- rep(seq_len(fit$K), each = n)
  w_idx <- as.vector(t(idx))
  tibble(K = fit$K, seed = fit$seed, topic = k_rep, rank = rep(seq_len(n), times = fit$K),
         term = vocabulary[w_idx], phi = fit$phi[cbind(k_rep, w_idx)],
         term_share_of_training_tokens = term_frequency_share[w_idx], relevance_lambda = lambda)
}
top_words_probability <- bind_rows(lapply(fits, function(f) top_words_table(f, f$top_probability, 1)))
top_words_relevance   <- bind_rows(lapply(fits, function(f) top_words_table(f, f$top_relevance, relevance_lambda)))
paste_top <- function(fit, idx, n = 10) apply(idx[, seq_len(n), drop = FALSE], 1, function(i) paste(vocabulary[i], collapse = ", "))
words_weighted <- function(fit, n = 10) vapply(seq_len(fit$K), function(k) {
  i <- fit$top_probability[k, seq_len(n)]
  paste(sprintf("%s (%.4f)", vocabulary[i], fit$phi[k, i]), collapse = "; ")
}, "")

# ---- Stability across seeds for K = 40 and K = 80 (the delivered rule, recomputed) ----
# INPUT : fits.
# DOES  : for each K and seed pair, the cosine matrix of topic-word distributions,
#         the Hungarian one-to-one assignment (clue::solve_LSAP, maximum total
#         cosine), the matched cosines, top-20 Jaccard, best any-topic cosine, ARI
#         of dominant topics; per seed-1 topic the weaker and stronger matched
#         cosine. Checked against the delivered stability_pairs.csv below.
# OUTPUT: stability_pairs; seed1_topic_stability; stability_cos (seed 1 vs 2, 3).
adjusted_rand_index <- function(a, b) {
  tab <- table(a, b)
  n <- sum(tab)
  sum_ij <- sum(choose(tab, 2)); sum_i <- sum(choose(rowSums(tab), 2)); sum_j <- sum(choose(colSums(tab), 2))
  expected <- sum_i * sum_j / choose(n, 2)
  (sum_ij - expected) / ((sum_i + sum_j) / 2 - expected)
}
jaccard <- function(x, y) length(intersect(x, y)) / length(union(x, y))
cosine_between <- function(phi1, phi2) tcrossprod(phi1 / sqrt(rowSums(phi1^2)), phi2 / sqrt(rowSums(phi2^2)))
seed_pairs <- combn(fit_seeds, 2, simplify = FALSE)
stability_pairs <- list(); stability_cos <- list(); stability_assignment_ok <- logical(0)
for (K in selected_k) for (pair in seed_pairs) {
  f1 <- fits[[fit_key(K, pair[1])]]; f2 <- fits[[fit_key(K, pair[2])]]
  cos_mat    <- cosine_between(f1$phi, f2$phi)
  assignment <- as.integer(solve_LSAP(cos_mat, maximum = TRUE))
  stability_assignment_ok <- c(stability_assignment_ok, setequal(assignment, seq_len(K)))
  if (pair[1] == fit_seeds[1]) stability_cos[[paste(K, pair[2], sep = "_")]] <- list(cos = cos_mat, assignment = assignment, seed_b = pair[2])
  topics      <- seq_len(K)
  matched_cos <- cos_mat[cbind(topics, assignment)]
  matched_jac <- vapply(topics, function(k) jaccard(f1$top_probability[k, ], f2$top_probability[assignment[k], ]), numeric(1))
  stability_pairs[[length(stability_pairs) + 1]] <- tibble(
    K = K, seed_a = pair[1], seed_b = pair[2], topic_a = topics, topic_b = assignment,
    cosine = matched_cos, jaccard_top_words = matched_jac,
    best_cosine_any_topic_b = apply(cos_mat, 1, max), best_topic_b = apply(cos_mat, 1, which.max),
    matched_is_row_maximum = assignment == apply(cos_mat, 1, which.max),
    token_share_a = f1$token_share, token_share_b = f2$token_share[assignment],
    ari_dominant_topics = adjusted_rand_index(f1$dominant, f2$dominant))
}
stability_pairs <- bind_rows(stability_pairs)
seed1_topic_stability <- stability_pairs |>
  filter(seed_a == fit_seeds[1]) |>
  group_by(K, topic_a) |>
  summarise(weaker_matched_cosine = min(cosine), stronger_matched_cosine = max(cosine),
            reproduced_in_all_other_seeds = all(cosine >= stability_cosine_threshold),
            reproduced_in_no_other_seed = all(cosine < 0.5), .groups = "drop")
# Agreement with the delivered table (same rule, same inputs): assignments identical, values equal to 1e-9.
sp_delivered <- delivered$stability_pairs |> filter(K %in% selected_k) |> arrange(K, seed_a, seed_b, topic_a)
sp_here      <- stability_pairs |> arrange(K, seed_a, seed_b, topic_a)
stability_agrees <- nrow(sp_delivered) == nrow(sp_here) && all(sp_delivered$topic_b == sp_here$topic_b) && all(sp_delivered$best_topic_b == sp_here$best_topic_b) &&
  all(sp_delivered$matched_is_row_maximum == sp_here$matched_is_row_maximum) &&
  max(abs(sp_delivered$cosine - sp_here$cosine)) < 1e-9 && max(abs(sp_delivered$jaccard_top_words - sp_here$jaccard_top_words)) < 1e-9 &&
  max(abs(sp_delivered$best_cosine_any_topic_b - sp_here$best_cosine_any_topic_b)) < 1e-9 &&
  max(abs(sp_delivered$ari_dominant_topics - sp_here$ari_dominant_topics)) < 1e-9
mark_stage("stability")
log_line("stability recomputed for K =", paste(selected_k, collapse = ", "), "; agrees with delivered table:", stability_agrees)

# ---- Representative documents for the six fits (the delivered rule, from the workers' indices) ----
# INPUT : fits (rep_docs); docs.
# DOES  : join the workers' representative-document indices to identifiers,
#         metadata and the stored text, exactly as the delivered script does.
# OUTPUT: representative_documents_six_fits; agreement flag against the delivered all-model table.
representative_documents_six_fits <- bind_rows(lapply(fits, function(f) {
  r <- modeled_order[f$rep_docs$modeled_index]
  tibble(K = f$K, seed = f$seed, topic = f$rep_docs$topic, topic_label = paste("Topic", f$rep_docs$topic), rank = f$rep_docs$rank,
         topic_share_in_document = f$rep_docs$share,
         doc_key = docs$doc_key[r], doc_id = docs$doc_id[r], doc_type = docs$doc_type[r], author = docs$author[r],
         created_utc = docs$created_utc[r], submission_id = docs$submission_id[r], link_id = docs$link_id[r],
         parent_id = docs$parent_id[r], split = docs$split[r], n_tokens_modeled = docs$n_tokens_modeled[r],
         dominant_topic = f$dominant_m[f$rep_docs$modeled_index], dominant_topic_share = f$max_share_m[f$rep_docs$modeled_index],
         text = docs$text[r])
}))
rd_delivered <- delivered$representative_documents_all_models |> filter(K %in% selected_k) |> arrange(K, seed, topic, rank)
rd_here      <- representative_documents_six_fits |> arrange(K, seed, topic, rank)
representative_agrees <- nrow(rd_delivered) == nrow(rd_here) && identical(rd_delivered$doc_key, rd_here$doc_key) &&
  identical(rd_delivered$text, rd_here$text) && max(abs(rd_delivered$topic_share_in_document - rd_here$topic_share_in_document)) < 1e-9 &&
  all(rd_delivered$dominant_topic == rd_here$dominant_topic) && all(rd_delivered$topic == rd_here$topic) && all(rd_delivered$rank == rd_here$rank)
mark_stage("representative_documents")

# ---- Retained-model tables (the delivered arithmetic, as a function of K) ----
# INPUT : a K, its seed-1 fit and theta file; docs; the recomputed stability tables.
# DOES  : exactly what the delivered script's retained-model loop does: load the
#         worker's theta, check it against the fit, and derive over all modeled
#         documents the token-weighted shares, shares by type and by UTC day,
#         the topic x topic correlation, the concentration tables, the topic
#         summary, the full document-topic table (one row per source document)
#         and the full topic-word table, plus the cross-seed matrices.
# OUTPUT: assemble_retained() returning the list of tables and per-document vectors.
band_breaks <- c(0, 1, 4, 9, 24, 49, 99, Inf)
band_labels <- c("1", "2-4", "5-9", "10-24", "25-49", "50-99", "100+")
assemble_retained <- function(K) {
  tag   <- sprintf("K%03d", K)
  fit   <- fits[[fit_key(K, fit_seeds[1])]]
  t_ret <- Sys.time()
  th    <- readRDS(theta_files[[as.character(K)]])
  theta_file_consistent <- identical(th$doc_key, docs$doc_key[modeled_order]) && th$K == K && th$seed == fit_seeds[1] &&
    identical(th$sampler_index, fit$sampler_index) && identical(dim(th$theta), c(n_modeled, K))
  theta_full <- matrix(NA_real_, n_docs, K)
  theta_full[modeled_order, ] <- th$theta
  rm(th)
  theta_m   <- theta_full[modeled_rows, , drop = FALSE]
  n_tok_m   <- as.numeric(docs$n_tokens_modeled[modeled_rows])
  type_m    <- docs$doc_type[modeled_rows]
  day_m     <- as.character(docs$created_date_utc[modeled_rows])
  dominant  <- max.col(theta_m, ties.method = "first")
  max_share <- theta_m[cbind(seq_len(n_modeled), dominant)]
  n_active  <- as.integer(rowSums(theta_m >= active_topic_threshold))
  active_share <- colMeans(theta_m >= active_topic_threshold)
  worker_summaries_identical <- identical(dominant, fit$dominant_m[match(modeled_rows, modeled_order)]) &&
    identical(max_share, fit$max_share_m[match(modeled_rows, modeled_order)]) &&
    identical(n_active, fit$n_active_m[match(modeled_rows, modeled_order)])
  weighted  <- theta_m * n_tok_m
  token_share <- colSums(weighted) / sum(n_tok_m)
  by_type   <- rowsum(weighted, type_m) / as.numeric(rowsum(n_tok_m, type_m))
  by_day    <- rowsum(weighted, day_m)  / as.numeric(rowsum(n_tok_m, day_m))
  rm(weighted)
  topic_cor <- cor(theta_m)
  dominant_full <- rep(NA_integer_, n_docs); dominant_full[modeled_rows] <- dominant
  max_share_full <- rep(NA_real_, n_docs);   max_share_full[modeled_rows] <- max_share
  n_active_full <- rep(NA_integer_, n_docs); n_active_full[modeled_rows] <- n_active

  topic_labels <- paste("Topic", seq_len(K))
  topic_cols   <- sprintf("topic_%0*d", nchar(K), seq_len(K))
  stab <- stability_pairs |> filter(K == !!K, seed_a == fit_seeds[1])
  stab_wide <- stab |> select(topic_a, seed_b, cosine) |>
    pivot_wider(names_from = seed_b, values_from = cosine, names_prefix = "matched_cosine_vs_seed_")
  best_wide <- stab |> select(topic_a, seed_b, best_cosine_any_topic_b) |>
    pivot_wider(names_from = seed_b, values_from = best_cosine_any_topic_b, names_prefix = "best_cosine_any_topic_vs_seed_")
  top_author <- tibble(topic = dominant, author = docs$author[modeled_rows]) |>
    count(topic, author, name = "docs") |>
    arrange(topic, desc(docs), author) |>
    distinct(topic, .keep_all = TRUE) |>
    transmute(topic, top_author = author, top_author_dominant_docs = docs)
  topic_summary <- tibble(topic = seq_len(K), topic_label = topic_labels, sampler_index = fit$sampler_index,
                          token_share_train = fit$token_share, token_share = token_share, doc_mean_share = colMeans(theta_m),
                          dominant_docs = tabulate(dominant, nbins = K), dominant_share = tabulate(dominant, nbins = K) / n_modeled,
                          token_share_comments = by_type["comment", ], token_share_submissions = by_type["submission", ],
                          coherence_umass = fit$topic_umass, coherence_npmi = fit$topic_npmi, exclusivity = fit$topic_exclusivity,
                          word_cosine_max_other_topic = vapply(seq_len(K), function(k) max(fit$word_cos[k, -k]), numeric(1)),
                          word_cosine_closest_topic = vapply(seq_len(K), function(k) { o <- setdiff(seq_len(K), k); o[which.max(fit$word_cos[k, o])] }, integer(1)),
                          doc_correlation_max_other_topic = vapply(seq_len(K), function(k) max(topic_cor[k, -k]), numeric(1)),
                          doc_correlation_closest_topic = vapply(seq_len(K), function(k) { o <- setdiff(seq_len(K), k); o[which.max(topic_cor[k, o])] }, integer(1)),
                          co_activation_max_other_topic = vapply(seq_len(K), function(k) max(fit$co_active[k, -k]), numeric(1))) |>
    left_join(top_author, by = "topic") |>
    mutate(top_author_share_of_dominant_docs = top_author_dominant_docs / dominant_docs) |>
    left_join(stab_wide, by = c("topic" = "topic_a")) |>
    left_join(best_wide, by = c("topic" = "topic_a")) |>
    left_join(seed1_topic_stability |> filter(K == !!K) |> select(topic_a, weaker_matched_cosine, reproduced_in_all_other_seeds),
              by = c("topic" = "topic_a")) |>
    mutate(top_10_words_by_probability = paste_top(fit, fit$top_probability),
           top_10_words_by_relevance   = paste_top(fit, fit$top_relevance))

  topic_prevalence_by_type <- tibble(doc_type = rep(rownames(by_type), each = K), topic = rep(seq_len(K), times = nrow(by_type)),
                                     token_share = as.vector(t(by_type))) |>
    left_join(tibble(doc_type = names(table(type_m)), modeled_docs = as.integer(table(type_m)),
                     modeled_tokens = as.numeric(rowsum(n_tok_m, type_m))), by = "doc_type")
  topic_prevalence_by_day <- tibble(created_date_utc = as.Date(rep(rownames(by_day), each = K)),
                                    topic = rep(seq_len(K), times = nrow(by_day)), token_share = as.vector(t(by_day))) |>
    left_join(tibble(created_date_utc = as.Date(rownames(by_day)), modeled_docs = as.integer(table(day_m)),
                     modeled_tokens = as.numeric(rowsum(n_tok_m, day_m))), by = "created_date_utc")
  wide_matrix <- function(m, quantity) as_tibble(unname(m), .name_repair = ~ topic_cols) |>
    mutate(topic = seq_len(K), quantity = quantity, .before = 1)
  topic_correlation <- wide_matrix(topic_cor, "Pearson correlation of document shares (theta columns) over all modeled documents; seed 1")
  co_activation     <- wide_matrix(fit$co_active, sprintf("share of modeled documents in which both topics have a share of at least %g; seed 1", active_topic_threshold))
  topic_word_cosine <- wide_matrix(fit$word_cos, "cosine similarity of smoothed topic-word distributions (phi rows); seed 1")

  concentration <- tibble(n_tokens_modeled = n_tok_m, dominant_share = max_share, n_active_topics = n_active) |>
    mutate(token_band = cut(n_tokens_modeled, breaks = band_breaks, labels = band_labels, right = TRUE),
           share_bin  = pmin(floor(dominant_share / 0.05), 19)) |>
    group_by(token_band, share_bin) |>
    summarise(docs = n(), mean_active_topics = mean(n_active_topics), .groups = "drop") |>
    mutate(share_lower = share_bin * 0.05, share_upper = share_lower + 0.05) |>
    group_by(token_band) |> mutate(docs_in_band = sum(docs), prop_in_band = docs / docs_in_band) |> ungroup()
  concentration_summary <- tibble(n_tokens_modeled = n_tok_m, dominant_share = max_share, n_active_topics = n_active) |>
    mutate(token_band = cut(n_tokens_modeled, breaks = band_breaks, labels = band_labels, right = TRUE)) |>
    group_by(token_band) |>
    summarise(docs = n(), dominant_share_median = median(dominant_share), dominant_share_mean = mean(dominant_share),
              share_docs_dominant_at_least_0.5 = mean(dominant_share >= 0.5),
              active_topics_mean = mean(n_active_topics), active_topics_median = median(n_active_topics), .groups = "drop")

  representative_documents <- representative_documents_six_fits |> filter(K == !!K, seed == fit_seeds[1]) |> select(-K, -seed)

  theta_out <- round(theta_full, 6)
  colnames(theta_out) <- topic_cols
  document_topic_table <- docs |>
    transmute(doc_key, doc_id, doc_type, author, created_utc, created_date_utc, submission_id, link_id, parent_id,
              submission_in_sample, n_chars, n_tokens_letters, n_tokens_modeled, model_status, split,
              dominant_topic = dominant_full, dominant_topic_share = max_share_full, n_active_topics = n_active_full) |>
    bind_cols(as_tibble(theta_out)) |>
    arrange(created_utc, doc_key)
  rm(theta_out, theta_full)

  phi_out <- t(fit$phi); colnames(phi_out) <- topic_cols
  topic_word_table <- tibble(term = vocabulary, document_frequency_train = diff(dtm_train@p),
                             term_frequency_train = as.numeric(colSums(dtm_train))) |>
    bind_cols(as_tibble(signif(phi_out, 6)))
  rm(phi_out)

  stability_matrices <- lapply(fit_seeds[-1], function(s) {
    sc <- stability_cos[[paste(K, s, sep = "_")]]
    tibble(seed1_topic = rep(seq_len(K), times = K), seed_b = s, seed_b_topic = rep(seq_len(K), each = K),
           cosine = as.vector(sc$cos), matched = rep(sc$assignment, times = K) == rep(seq_len(K), each = K),
           quantity = "cosine similarity of topic-word distributions between the seed-1 topic and the seed-b topic (different fits, same K)")
  })
  names(stability_matrices) <- as.character(fit_seeds[-1])
  invisible(gc())
  list(K = K, tag = tag, fit = fit, theta_m = theta_m, dominant = dominant, max_share = max_share, n_active = n_active,
       theta_file_consistent = theta_file_consistent, worker_summaries_identical = worker_summaries_identical, active_share = active_share,
       topic_summary = topic_summary, topic_prevalence_by_type = topic_prevalence_by_type,
       topic_prevalence_by_day = topic_prevalence_by_day, topic_correlation = topic_correlation,
       co_activation = co_activation, topic_word_cosine = topic_word_cosine,
       concentration = concentration, concentration_summary = concentration_summary,
       representative_documents = representative_documents,
       document_topic_table = document_topic_table, topic_word_table = topic_word_table,
       stability_matrices = stability_matrices, topic_cols = topic_cols, topic_labels = topic_labels,
       seconds = as.numeric(difftime(Sys.time(), t_ret, units = "secs")))
}
retained_table_names <- function(r) {
  tag <- r$tag
  c(setNames(list(r$topic_summary, r$topic_prevalence_by_type, r$topic_prevalence_by_day, r$topic_correlation, r$co_activation,
                  r$topic_word_cosine, r$concentration, r$concentration_summary, r$representative_documents),
             sprintf(c("topic_summary_%s.csv", "topic_prevalence_by_type_%s.csv", "topic_prevalence_by_day_%s.csv", "topic_correlation_%s.csv",
                       "topic_coactivation_%s.csv", "topic_word_cosine_%s.csv", "document_concentration_bins_%s.csv",
                       "document_concentration_summary_%s.csv", "representative_documents_%s.csv"), tag)),
    setNames(r$stability_matrices, sprintf("stability_matrix_seed1_vs_seed%s_%s.csv", names(r$stability_matrices), tag)),
    setNames(list(r$topic_word_table, r$document_topic_table), sprintf(c("topic_word_distribution_%s.csv", "document_topic_distribution_%s.csv"), tag)))
}

# ---- K = 80: reproduce the delivered retained tables in scratch and hash-check them (nothing written to the delivered folder) ----
# INPUT : the K = 80 seed-1 fit and theta; the delivered K080 manifests.
# DOES  : assemble every K = 80 retained table with the function above, write it
#         to the scratch folder with the same writer, and compare its SHA256 with
#         the output_hash recorded in the delivered manifest and with the
#         delivered file's current hash. The scratch files are deleted afterwards.
# OUTPUT: k80_reproduction (one row per table); retained_ref (theta and vectors kept for the cross-K tables).
retained_ref <- assemble_retained(k_reference)
ref_tables <- retained_table_names(retained_ref)
k80_reproduction <- bind_rows(lapply(names(ref_tables), function(name) {
  scratch_path   <- file.path(scratch_dir, name)
  delivered_path <- file.path(tier2_tab_dir, name)
  write_csv(ref_tables[[name]], scratch_path)
  h_scratch    <- sha256(scratch_path)
  h_manifest   <- manifest_hash(delivered_path)
  h_delivered  <- if (file.exists(delivered_path)) sha256(delivered_path) else NA_character_
  file.remove(scratch_path)
  tibble(table = name, rows = nrow(ref_tables[[name]]), cols = ncol(ref_tables[[name]]),
         sha256_regenerated_here = h_scratch, sha256_in_delivered_manifest = h_manifest,
         sha256_of_delivered_file_now = h_delivered, delivered_file_present = file.exists(delivered_path),
         regenerated_equals_manifest = h_scratch == h_manifest,
         delivered_file_equals_manifest = !is.na(h_delivered) & h_delivered == h_manifest)
}))
retained_ref$document_topic_table <- NULL; retained_ref$topic_word_table <- NULL; rm(ref_tables); invisible(gc())
mark_stage("k80_reproduction")
log_line("K = 80 reproduction:", sum(k80_reproduction$regenerated_equals_manifest), "of", nrow(k80_reproduction), "tables equal their delivered manifests")

# ---- K = 40: the retained-model outputs (written to the package) ----
retained_new <- assemble_retained(k_new)
mark_stage("retained_K040")
log_line("K = 40 retained tables assembled in", round(retained_new$seconds, 1), "s")

# ---- Cross-K correspondence: K = 40 topics against K = 80 topics ----
# INPUT : fits (phi, top-20 word indices) for all seeds; the two seed-1 thetas
#         and dominant-topic vectors from the retained assembly.
# DOES  : the correspondence measure is the cosine similarity between the
#         smoothed topic-word distributions (phi rows) of a K = 40 topic and a
#         K = 80 topic — the quantity the delivered cross-seed matching uses,
#         applied across K. It is computed for every K = 40 topic x K = 80 topic
#         pair and every seed pair (9), so the structure can be checked across
#         seeds. For the seed-1 pair three supporting measures are added: the
#         Jaccard overlap of the two topics' top-20 words, the Pearson
#         correlation of their document shares over all modeled documents, and
#         the cross-tabulation of dominant topics (how the documents dominated
#         by a K = 40 topic are dominated under K = 80). A K = 80 topic
#         "corresponds" to a K = 40 topic at cosine >= correspondence_threshold
#         (0.5) and the correspondence is "clear" at >= clear_threshold (0.8, the
#         delivered stability threshold). These thresholds are this package's
#         choices; every cosine is reported so other cuts can be read off.
#         No lineage or meaning is inferred: a correspondence is a measured
#         similarity of word distributions, nothing more.
# OUTPUT: cross tables (long and wide), per-topic correspondence tables in both
#         directions, the per-seed-pair structure summary.
f40 <- fits[[fit_key(k_new, 1L)]]; f80 <- fits[[fit_key(k_reference, 1L)]]
cross_quantity <- "cosine similarity of smoothed topic-word distributions (phi rows) between the K = 40 topic and the K = 80 topic (different fits, different K)"
cross_cos_all <- list()
for (a in fit_seeds) for (b in fit_seeds)
  cross_cos_all[[paste(a, b, sep = "_")]] <- cosine_between(fits[[fit_key(k_new, a)]]$phi, fits[[fit_key(k_reference, b)]]$phi)
cross_cos <- cross_cos_all[["1_1"]]
cross_jac <- matrix(0, k_new, k_reference)
for (i in seq_len(k_new)) for (j in seq_len(k_reference)) cross_jac[i, j] <- jaccard(f40$top_probability[i, ], f80$top_probability[j, ])
cross_doc_cor <- cor(retained_new$theta_m, retained_ref$theta_m)
crosstab   <- unclass(table(factor(retained_new$dominant, levels = seq_len(k_new)), factor(retained_ref$dominant, levels = seq_len(k_reference))))
row_share  <- crosstab / rowSums(crosstab)
col_share  <- sweep(crosstab, 2, colSums(crosstab), "/")

cross_k_all_seed_pairs <- bind_rows(lapply(names(cross_cos_all), function(nm) {
  ab <- as.integer(strsplit(nm, "_")[[1]]); m <- cross_cos_all[[nm]]
  tibble(seed_K040 = ab[1], seed_K080 = ab[2], topic_K040 = rep(seq_len(k_new), times = k_reference),
         topic_K080 = rep(seq_len(k_reference), each = k_new), cosine = as.vector(m), quantity = cross_quantity)
}))
k80_cols <- sprintf("topic_%02d", seq_len(k_reference))
cross_k_wide <- as_tibble(unname(cross_cos), .name_repair = ~ k80_cols) |>
  mutate(topic_K040 = seq_len(k_new), quantity = paste(cross_quantity, "; seed 1 fits; columns = K = 80 topics"), .before = 1)
cross_k_pairs <- tibble(topic_K040 = rep(seq_len(k_new), times = k_reference), topic_K080 = rep(seq_len(k_reference), each = k_new),
                        word_cosine = as.vector(cross_cos), jaccard_top20_words = as.vector(cross_jac),
                        doc_share_correlation = as.vector(cross_doc_cor),
                        docs_dominated_by_both = as.vector(crosstab),
                        share_of_K040_dominant_docs = as.vector(row_share), share_of_K080_dominant_docs = as.vector(col_share)) |>
  mutate(at_or_above_correspondence_threshold = word_cosine >= correspondence_threshold,
         is_best_K080_for_K040 = topic_K080 == apply(cross_cos, 1, which.max)[topic_K040],
         is_best_K040_for_K080 = topic_K040 == apply(cross_cos, 2, which.max)[topic_K080])

stab40 <- seed1_topic_stability |> filter(K == k_new) |> arrange(topic_a)
stab80 <- seed1_topic_stability |> filter(K == k_reference) |> arrange(topic_a)
notice_cos <- lapply(fits, function(f) as.vector(cosine_between(f$phi, anchor_phi)))
is_notice <- lapply(notice_cos, function(cs) cs >= notice_threshold)
list_topics <- function(m, i, thr, by_row = TRUE) {
  v <- if (by_row) m[i, ] else m[, i]
  o <- order(-v); o <- o[v[o] >= thr]
  if (!length(o)) "" else paste(sprintf("T%d (%.3f)", o, v[o]), collapse = "; ")
}
best80_of <- apply(cross_cos, 1, which.max); best40_of <- apply(cross_cos, 2, which.max)
correspondence_class <- function(n05, best) case_when(n05 == 0 ~ "no counterpart at cosine >= 0.5",
                                                     n05 == 1 & best >= clear_threshold ~ "one clear counterpart (cosine >= 0.8)",
                                                     n05 == 1 ~ "one moderate counterpart (cosine 0.5 to 0.8)",
                                                     best >= clear_threshold ~ "split: several counterparts, one clear (>= 0.8)",
                                                     TRUE ~ "split: several counterparts, all moderate (0.5 to 0.8)")
topic_correspondence_K040 <- tibble(
  topic_K040 = seq_len(k_new), top_10_words_K040 = paste_top(f40, f40$top_probability),
  token_share_K040 = retained_new$topic_summary$token_share, dominant_docs_K040 = retained_new$topic_summary$dominant_docs,
  is_notice_topic_K040 = is_notice[[fit_key(k_new, 1L)]],
  best_topic_K080 = best80_of, best_cosine = apply(cross_cos, 1, max), best_is_mutual = best40_of[best80_of] == seq_len(k_new),
  top_10_words_best_K080 = paste_top(f80, f80$top_probability)[best80_of],
  n_K080_topics_at_or_above_0.5 = rowSums(cross_cos >= correspondence_threshold),
  n_K080_topics_at_or_above_0.7 = rowSums(cross_cos >= 0.7), n_K080_topics_at_or_above_0.8 = rowSums(cross_cos >= clear_threshold),
  K080_topics_at_or_above_0.5 = vapply(seq_len(k_new), function(i) list_topics(cross_cos, i, correspondence_threshold), ""),
  token_share_of_K080_correspondents = vapply(seq_len(k_new), function(i) sum(retained_ref$topic_summary$token_share[cross_cos[i, ] >= correspondence_threshold]), numeric(1)),
  share_of_dominant_docs_to_best_K080 = row_share[cbind(seq_len(k_new), best80_of)],
  share_of_dominant_docs_to_K080_correspondents = vapply(seq_len(k_new), function(i) sum(row_share[i, cross_cos[i, ] >= correspondence_threshold]), numeric(1)),
  jaccard_top20_with_best_K080 = cross_jac[cbind(seq_len(k_new), best80_of)],
  doc_share_correlation_with_best_K080 = cross_doc_cor[cbind(seq_len(k_new), best80_of)],
  weaker_matched_cosine_across_seeds_K040 = stab40$weaker_matched_cosine, stronger_matched_cosine_across_seeds_K040 = stab40$stronger_matched_cosine,
  reproduced_in_both_other_seeds_K040 = stab40$reproduced_in_all_other_seeds, no_counterpart_in_other_seeds_K040 = stab40$reproduced_in_no_other_seed) |>
  mutate(correspondence_class = correspondence_class(n_K080_topics_at_or_above_0.5, best_cosine))

topic_predecessors_K080 <- tibble(
  topic_K080 = seq_len(k_reference), top_10_words_K080 = paste_top(f80, f80$top_probability),
  token_share_K080 = retained_ref$topic_summary$token_share, dominant_docs_K080 = retained_ref$topic_summary$dominant_docs,
  is_notice_topic_K080 = is_notice[[fit_key(k_reference, 1L)]],
  best_topic_K040 = best40_of, best_cosine = apply(cross_cos, 2, max), best_is_mutual = best80_of[best40_of] == seq_len(k_reference),
  top_10_words_best_K040 = paste_top(f40, f40$top_probability)[best40_of],
  n_K040_topics_at_or_above_0.5 = colSums(cross_cos >= correspondence_threshold),
  K040_topics_at_or_above_0.5 = vapply(seq_len(k_reference), function(j) list_topics(cross_cos, j, correspondence_threshold, by_row = FALSE), ""),
  best_K040_topic_is_split = topic_correspondence_K040$n_K080_topics_at_or_above_0.5[best40_of] >= 2,
  share_of_dominant_docs_from_best_K040 = col_share[cbind(best40_of, seq_len(k_reference))],
  jaccard_top20_with_best_K040 = cross_jac[cbind(best40_of, seq_len(k_reference))],
  doc_share_correlation_with_best_K040 = cross_doc_cor[cbind(best40_of, seq_len(k_reference))],
  weaker_matched_cosine_across_seeds_K080 = stab80$weaker_matched_cosine, stronger_matched_cosine_across_seeds_K080 = stab80$stronger_matched_cosine,
  reproduced_in_both_other_seeds_K080 = stab80$reproduced_in_all_other_seeds, no_counterpart_in_other_seeds_K080 = stab80$reproduced_in_no_other_seed,
  coherence_npmi_K080 = f80$topic_npmi, exclusivity_K080 = f80$topic_exclusivity) |>
  mutate(predecessor_class = case_when(best_cosine >= clear_threshold ~ "clear K = 40 predecessor (cosine >= 0.8)",
                                       best_cosine >= correspondence_threshold ~ "moderate K = 40 predecessor (cosine 0.5 to 0.8)",
                                       TRUE ~ "no K = 40 predecessor at cosine >= 0.5"))

correspondence_structure_by_seed_pair <- bind_rows(lapply(names(cross_cos_all), function(nm) {
  ab <- as.integer(strsplit(nm, "_")[[1]]); m <- cross_cos_all[[nm]]
  n05 <- rowSums(m >= correspondence_threshold); n07 <- rowSums(m >= 0.7)
  b40 <- apply(m, 1, max); b80 <- apply(m, 2, max)
  tibble(seed_K040 = ab[1], seed_K080 = ab[2],
         K040_topics_with_one_counterpart_0.5 = sum(n05 == 1), K040_topics_split_0.5 = sum(n05 >= 2), K040_topics_without_counterpart_0.5 = sum(n05 == 0),
         K040_topics_with_one_counterpart_0.7 = sum(n07 == 1), K040_topics_split_0.7 = sum(n07 >= 2), K040_topics_without_counterpart_0.7 = sum(n07 == 0),
         K040_topics_best_at_or_above_0.8 = sum(b40 >= clear_threshold), K040_best_cosine_mean = mean(b40), K040_best_cosine_min = min(b40),
         K080_topics_without_predecessor_0.5 = sum(b80 < correspondence_threshold), K080_topics_without_predecessor_0.3 = sum(b80 < 0.3),
         K080_topics_best_at_or_above_0.8 = sum(b80 >= clear_threshold), K080_best_cosine_mean = mean(b80), K080_best_cosine_median = median(b80),
         mutual_best_pairs = sum(apply(m, 2, which.max)[apply(m, 1, which.max)] == seq_len(k_new)),
         pairs_at_or_above_0.5 = sum(m >= correspondence_threshold), pairs_at_or_above_0.9 = sum(m >= 0.9))
}))
mark_stage("cross_k")
log_line("cross-K: seed 1: ", sum(topic_correspondence_K040$n_K080_topics_at_or_above_0.5 >= 2), "K = 40 topics split;",
         sum(topic_predecessors_K080$best_cosine < correspondence_threshold), "K = 80 topics without predecessor")

# ---- AutoModerator civility-notice copies in the two K ----
# INPUT : fits (all seeds of both K); the anchor topic; docs; the retained tables.
# DOES  : a topic is a notice copy when the cosine between its topic-word
#         distribution and the anchor (the delivered README's civility-notice
#         topic, K = 10 seed 1 Topic 10) is at least notice_threshold; every
#         topic's cosine to the anchor is recorded. Per copy: token shares,
#         dominant documents, top author, coherence, exclusivity, its matched
#         cosine against the other seeds. Per fit: number of copies, share of
#         the K slots, combined token mass, documents dominated by any copy,
#         between-copy similarities on all four topic x topic quantities, and
#         how many copies have a near-identical copy in each other seed.
#         Nothing is removed, filtered or altered.
# OUTPUT: notice_topics; notice_summary.
author_m <- docs$author[modeled_order]
notice_topics <- bind_rows(lapply(names(fits), function(k) {
  f <- fits[[k]]; cs <- notice_cos[[k]]; hit <- which(is_notice[[k]])
  if (!length(hit)) return(NULL)
  others <- setdiff(fit_seeds, f$seed)
  matched <- vapply(others, function(s) {
    r <- stability_pairs |> filter(K == f$K, (seed_a == f$seed & seed_b == s) | (seed_b == f$seed & seed_a == s))
    vapply(hit, function(t) { v <- r$cosine[(r$seed_a == f$seed & r$topic_a == t) | (r$seed_b == f$seed & r$topic_b == t)]; if (length(v)) v[1] else NA_real_ }, numeric(1))
  }, numeric(length(hit)))
  matched <- matrix(matched, nrow = length(hit))
  dom_tab <- tabulate(f$dominant_m, nbins = f$K)
  top_auth <- vapply(hit, function(t) { a <- author_m[f$dominant_m == t]; if (!length(a)) return(c(NA_character_, NA_character_)); tb <- sort(table(a), decreasing = TRUE); c(names(tb)[1], as.character(tb[[1]])) }, character(2))
  tibble(K = f$K, seed = f$seed, topic = hit, copy_rank_in_fit = seq_along(hit), cosine_to_anchor = cs[hit],
         token_share_train = f$token_share[hit], doc_mean_share_train = f$doc_mean_share[hit],
         dominant_docs_all_modeled = dom_tab[hit], dominant_share_all_modeled = dom_tab[hit] / n_modeled,
         top_author = top_auth[1, ], top_author_dominant_docs = as.integer(top_auth[2, ]),
         top_author_share_of_dominant_docs = as.integer(top_auth[2, ]) / dom_tab[hit],
         coherence_npmi = f$topic_npmi[hit], exclusivity = f$topic_exclusivity[hit],
         matched_cosine_vs_other_seed_a = matched[, 1], other_seed_a = others[1], matched_cosine_vs_other_seed_b = matched[, 2], other_seed_b = others[2],
         top_10_words_by_probability = paste_top(f, f$top_probability)[hit])
}))
notice_summary <- bind_rows(lapply(names(fits), function(k) {
  f <- fits[[k]]; cs <- notice_cos[[k]]; hit <- which(is_notice[[k]]); n <- length(hit)
  pair_stat <- function(m, fun) if (n < 2) NA_real_ else fun(m[hit, hit][upper.tri(m[hit, hit])])
  others <- setdiff(fit_seeds, f$seed)
  copies_matched_elsewhere <- vapply(others, function(s) {
    g <- fits[[fit_key(f$K, s)]]; oh <- which(is_notice[[fit_key(f$K, s)]])
    if (!length(oh) || !n) return(0L)
    cm <- cosine_between(f$phi[hit, , drop = FALSE], g$phi[oh, , drop = FALSE])
    sum(apply(cm, 1, max) >= 0.99)
  }, integer(1))
  tibble(K = f$K, seed = f$seed, notice_copies = n, notice_topics = paste(hit, collapse = "; "),
         share_of_topic_slots = n / f$K, effective_distinct_topics = f$K - max(n - 1L, 0L),
         combined_token_share_train = sum(f$token_share[hit]),
         combined_token_share_all_modeled = if (f$seed == 1L) sum((if (f$K == k_new) retained_new else retained_ref)$topic_summary$token_share[hit]) else NA_real_,
         docs_dominated_by_any_copy = sum(f$dominant_m %in% hit), share_docs_dominated_by_any_copy = mean(f$dominant_m %in% hit),
         largest_copy_token_share_train = if (n) max(f$token_share[hit]) else NA_real_, smallest_copy_token_share_train = if (n) min(f$token_share[hit]) else NA_real_,
         between_copy_word_cosine_min = pair_stat(f$word_cos, min), between_copy_word_cosine_max = pair_stat(f$word_cos, max),
         between_copy_top20_jaccard_min = pair_stat(f$top_jaccard, min), between_copy_top20_jaccard_max = pair_stat(f$top_jaccard, max),
         between_copy_doc_share_correlation_max = pair_stat(f$doc_cor, max), between_copy_co_activation_max = pair_stat(f$co_active, max),
         cosine_to_anchor_min_among_copies = if (n) min(cs[hit]) else NA_real_, cosine_to_anchor_max_among_other_topics = max(cs[-hit]),
         copies_with_near_identical_copy_in_seed_a = copies_matched_elsewhere[1], other_seed_a = others[1],
         copies_with_near_identical_copy_in_seed_b = copies_matched_elsewhere[2], other_seed_b = others[2],
         npmi_mean_all_topics = mean(f$topic_npmi), npmi_mean_excluding_notice = mean(f$topic_npmi[-hit]),
         exclusivity_mean_all_topics = mean(f$topic_exclusivity), exclusivity_mean_excluding_notice = mean(f$topic_exclusivity[-hit]))
})) |> arrange(K, seed)
mark_stage("notice_topics")

# ---- Representative-document inspectability proxies (six fits) ----
# INPUT : representative_documents_six_fits; fits; the tokeniser rules.
# DOES  : per representative document: how many of its topic's ten top words
#         occur in it (tokens recomputed from the stored text by the delivered
#         rules), its type-token ratio, whether it is dominated by its topic and
#         whether AutoModerator wrote it; per topic and fit the summaries used
#         in the comparison.
# OUTPUT: representative_document_rows (with proxies); representative_document_proxies (per topic and fit).
recompute_tokens <- function(text) {
  x <- stri_replace_all_regex(stri_replace_all_fixed(stri_trans_tolower(text), intToUtf8(0x2019), "'"), url_regex, " ", opts_regex = url_opts)
  t <- stri_extract_all_regex(x, token_regex, omit_no_match = TRUE)[[1]]
  t <- stri_replace_last_regex(t, "'s$", "")
  t <- t[stri_length(t) >= min_token_chars & !(t %in% stopword_list)]
  t[t %in% vocabulary]
}
representative_document_rows <- representative_documents_six_fits |>
  mutate(tokens = lapply(text, recompute_tokens)) |>
  mutate(top_10_words_of_topic = mapply(function(K, s, t) paste_top(fits[[fit_key(K, s)]], fits[[fit_key(K, s)]]$top_probability)[t], K, seed, topic),
         top_10_words_present = mapply(function(K, s, t, tk) sum(vocabulary[fits[[fit_key(K, s)]]$top_probability[t, seq_len(coherence_top_n)]] %in% tk), K, seed, topic, tokens),
         n_tokens_recomputed = lengths(tokens), n_distinct_tokens = vapply(tokens, function(tk) length(unique(tk)), integer(1)),
         type_token_ratio = n_distinct_tokens / pmax(n_tokens_recomputed, 1L),
         dominated_by_topic = dominant_topic == topic, author_is_automoderator = author == "AutoModerator") |>
  select(-tokens)
representative_document_proxies <- representative_document_rows |>
  group_by(K, seed, topic) |>
  summarise(top5_mean_share = mean(topic_share_in_document), top5_min_share = min(topic_share_in_document),
            docs_dominated_by_topic = sum(dominated_by_topic), top_10_words_present_mean = mean(top_10_words_present),
            min_type_token_ratio = min(type_token_ratio), has_repetitive_document = any(type_token_ratio < repetitive_ttr),
            automoderator_rows = sum(author_is_automoderator), .groups = "drop") |>
  mutate(flag_mean_share_below_0.5 = top5_mean_share < 0.5, flag_fewer_than_2_top_words_on_average = top_10_words_present_mean < 2,
         flag_not_all_dominated = docs_dominated_by_topic < n_representative)
mark_stage("representative_proxies")

# ---- Inspection tables for every correspondence and split ----
# INPUT : the correspondence tables; representative_document_rows; fits.
# DOES  : one row per K = 40 x K = 80 pair at or above the correspondence
#         threshold (seed 1), with both topics' ten top words and weights,
#         shares, stability and notice flags, plus one row per K = 80 topic
#         without a predecessor; and a document table in which every K = 40
#         topic's five representative documents are followed by those of each
#         K = 80 topic whose best predecessor it is (K = 80 topics without a
#         predecessor form the last group). Every K = 80 topic appears once.
# OUTPUT: correspondence_inspection; correspondence_representative_documents.
ww40 <- words_weighted(f40); ww80 <- words_weighted(f80)
pair_rows <- cross_k_pairs |> filter(at_or_above_correspondence_threshold) |> arrange(topic_K040, desc(word_cosine))
no_pred <- topic_predecessors_K080 |> filter(best_cosine < correspondence_threshold)
correspondence_inspection <- bind_rows(
  pair_rows |> transmute(group = sprintf("K = 40 Topic %d", topic_K040), row_type = "correspondence at or above 0.5", topic_K040, topic_K080,
                         word_cosine, jaccard_top20_words, doc_share_correlation, share_of_K040_dominant_docs, share_of_K080_dominant_docs,
                         is_best_K080_for_K040, is_best_K040_for_K080),
  no_pred |> transmute(group = "K = 80 topics without a K = 40 predecessor at cosine >= 0.5", row_type = "K = 80 topic without predecessor (its best K = 40 topic shown)",
                       topic_K040 = best_topic_K040, topic_K080, word_cosine = best_cosine, jaccard_top20_words = jaccard_top20_with_best_K040,
                       doc_share_correlation = doc_share_correlation_with_best_K040, share_of_K040_dominant_docs = row_share[cbind(best_topic_K040, topic_K080)],
                       share_of_K080_dominant_docs = share_of_dominant_docs_from_best_K040, is_best_K080_for_K040 = best_is_mutual, is_best_K040_for_K080 = TRUE)) |>
  mutate(correspondence_class_K040 = topic_correspondence_K040$correspondence_class[topic_K040],
         predecessor_class_K080 = topic_predecessors_K080$predecessor_class[topic_K080],
         top_10_words_with_phi_K040 = ww40[topic_K040], top_10_words_with_phi_K080 = ww80[topic_K080],
         token_share_K040 = retained_new$topic_summary$token_share[topic_K040], token_share_K080 = retained_ref$topic_summary$token_share[topic_K080],
         dominant_docs_K040 = retained_new$topic_summary$dominant_docs[topic_K040], dominant_docs_K080 = retained_ref$topic_summary$dominant_docs[topic_K080],
         coherence_npmi_K040 = f40$topic_npmi[topic_K040], coherence_npmi_K080 = f80$topic_npmi[topic_K080],
         exclusivity_K040 = f40$topic_exclusivity[topic_K040], exclusivity_K080 = f80$topic_exclusivity[topic_K080],
         weaker_matched_cosine_across_seeds_K040 = stab40$weaker_matched_cosine[topic_K040], weaker_matched_cosine_across_seeds_K080 = stab80$weaker_matched_cosine[topic_K080],
         reproduced_in_both_other_seeds_K040 = stab40$reproduced_in_all_other_seeds[topic_K040], reproduced_in_both_other_seeds_K080 = stab80$reproduced_in_all_other_seeds[topic_K080],
         no_counterpart_in_other_seeds_K040 = stab40$reproduced_in_no_other_seed[topic_K040], no_counterpart_in_other_seeds_K080 = stab80$reproduced_in_no_other_seed[topic_K080],
         is_notice_topic_K040 = is_notice[[fit_key(k_new, 1L)]][topic_K040], is_notice_topic_K080 = is_notice[[fit_key(k_reference, 1L)]][topic_K080])

rep_cols <- c("rank", "topic_share_in_document", "dominant_topic", "dominant_topic_share", "doc_key", "doc_id", "doc_type", "author", "created_utc",
              "submission_id", "link_id", "parent_id", "split", "n_tokens_modeled", "top_10_words_of_topic", "top_10_words_present", "type_token_ratio",
              "dominated_by_topic", "author_is_automoderator", "text")
rd40 <- representative_document_rows |> filter(K == k_new, seed == 1L); rd80 <- representative_document_rows |> filter(K == k_reference, seed == 1L)
group_order <- c(seq_len(k_new), NA)
correspondence_representative_documents <- bind_rows(lapply(group_order, function(g) {
  if (is.na(g)) {
    members <- no_pred |> arrange(desc(best_cosine))
    bind_rows(lapply(seq_len(nrow(members)), function(i) rd80 |> filter(topic == members$topic_K080[i]) |>
      transmute(group = "K = 80 topics without a K = 40 predecessor at cosine >= 0.5", K, topic, role = "K = 80 topic without predecessor",
                cosine_to_group_K040_topic = members$best_cosine[i], best_K040_topic = members$best_topic_K040[i], across(all_of(rep_cols)))))
  } else {
    members <- topic_predecessors_K080 |> filter(best_topic_K040 == g, best_cosine >= correspondence_threshold) |> arrange(desc(best_cosine))
    bind_rows(rd40 |> filter(topic == g) |> transmute(group = sprintf("K = 40 Topic %d", g), K, topic, role = "K = 40 topic", cosine_to_group_K040_topic = 1,
                                                       best_K040_topic = g, across(all_of(rep_cols))),
              lapply(seq_len(nrow(members)), function(i) rd80 |> filter(topic == members$topic_K080[i]) |>
                transmute(group = sprintf("K = 40 Topic %d", g), K, topic, role = "K = 80 topic whose best predecessor is the group topic",
                          cosine_to_group_K040_topic = members$best_cosine[i], best_K040_topic = g, across(all_of(rep_cols)))))
  }
}))
mark_stage("inspection_tables")

# ---- The K = 40 versus K = 80 comparison on the delivered measured evidence ----
# INPUT : the delivered per-fit, per-K, stability and runtime tables; the fits
#         (traces, per-document summaries, K x K matrices); the notice and
#         representative-document tables above.
# DOES  : one row per item with the K = 40 and K = 80 values (per-seed triples
#         where the quantity is per fit), the source table and a note. Items
#         marked "derived here" are arithmetic on the cached fits or on the
#         delivered tables, not new fits.
# OUTPUT: comparison_summary.
mcf <- delivered$model_comparison_by_fit; mcs <- delivered$model_comparison_summary; ss <- delivered$stability_summary; frt <- delivered$fit_runtime
cmp <- list()
add_item <- function(section, item, k40, k80, source, note = "") cmp[[length(cmp) + 1L]] <<- tibble(section = section, item = item, K040 = as.character(k40), K080 = as.character(k80), source = source, note = note)
triple <- function(tbl, K, col, digits = 3) paste(formatC(tbl[[col]][tbl$K == K][order(tbl$seed[tbl$K == K])], format = "f", digits = digits, big.mark = ","), collapse = " / ")
single <- function(tbl, K, col, digits = 3) formatC(tbl[[col]][tbl$K == K], format = "f", digits = digits, big.mark = ",")
per_seed <- function(fun, K, digits = 3) paste(vapply(fit_seeds, function(s) formatC(fun(fits[[fit_key(K, s)]]), format = "f", digits = digits, big.mark = ","), ""), collapse = " / ")
gini <- function(p) { p <- sort(p); n <- length(p); 2 * sum(seq_len(n) * p) / (n * sum(p)) - (n + 1) / n }
trace_change <- function(f, back) { ll <- f$trace$loglikelihood; n <- length(ll); (ll[n] - ll[n - back]) / abs(ll[n - back]) }
notice_n <- function(K) paste(vapply(fit_seeds, function(s) notice_summary$notice_copies[notice_summary$K == K & notice_summary$seed == s], integer(1)), collapse = " / ")
non_notice_pairs <- function(f, m, thr) { nn <- !is_notice[[fit_key(f$K, f$seed)]]; mm <- m[nn, nn]; sum(mm[upper.tri(mm)] >= thr) }
most_similar_non_notice <- function(f) { nn <- which(!is_notice[[fit_key(f$K, f$seed)]]); mm <- f$word_cos[nn, nn]; diag(mm) <- 0; i <- which(mm == max(mm), arr.ind = TRUE)[1, ]; sprintf("T%d-T%d (%.3f)", nn[i[1]], nn[i[2]], max(mm)) }
S <- "held-out perplexity"
add_item(S, "held-out perplexity per seed (document completion; lower is better)", triple(mcf, 40, "heldout_perplexity", 1), triple(mcf, 80, "heldout_perplexity", 1), "delivered: model_comparison_by_fit.csv")
add_item(S, "held-out perplexity, mean (SD over seeds)", sprintf("%s (%s)", single(mcs, 40, "heldout_perplexity_mean", 2), single(mcs, 40, "heldout_perplexity_sd", 2)), sprintf("%s (%s)", single(mcs, 80, "heldout_perplexity_mean", 2), single(mcs, 80, "heldout_perplexity_sd", 2)), "delivered: model_comparison_summary.csv")
add_item(S, "rank of the mean over the eight candidate K (1 = lowest)", single(mcs, 40, "rank_heldout_perplexity", 0), single(mcs, 80, "rank_heldout_perplexity", 0), "delivered: model_comparison_summary.csv")
add_item(S, "within one seed SD of the minimum (one-standard-error band)", mcs$heldout_perplexity_within_one_sd_of_best[mcs$K == 40], mcs$heldout_perplexity_within_one_sd_of_best[mcs$K == 80], "delivered: model_comparison_summary.csv", sprintf("band: minimum %.2f + SD %.2f = %.2f", min(mcs$heldout_perplexity_mean), mcs$heldout_perplexity_sd[which.min(mcs$heldout_perplexity_mean)], min(mcs$heldout_perplexity_mean) + mcs$heldout_perplexity_sd[which.min(mcs$heldout_perplexity_mean)]))
perp_diff <- mcs$heldout_perplexity_mean[mcs$K == 40] - mcs$heldout_perplexity_mean[mcs$K == 80]
add_item(S, "difference of the means, K = 40 minus K = 80", sprintf("%.2f", perp_diff), "", "derived here from model_comparison_summary.csv",
         sprintf("%.1f seed SDs of K = 40 and %.1f seed SDs of K = 80", perp_diff / mcs$heldout_perplexity_sd[mcs$K == 40], perp_diff / mcs$heldout_perplexity_sd[mcs$K == 80]))
add_item(S, "training perplexity per seed", triple(mcf, 40, "train_perplexity", 1), triple(mcf, 80, "train_perplexity", 1), "delivered: model_comparison_by_fit.csv")
S <- "coherence"
add_item(S, "NPMI over the top 10 words, mean over topics, per seed (higher is better)", triple(mcf, 40, "coherence_npmi", 4), triple(mcf, 80, "coherence_npmi", 4), "delivered: model_comparison_by_fit.csv")
add_item(S, "NPMI mean over seeds (rank over the eight K)", sprintf("%s (rank %s)", single(mcs, 40, "coherence_npmi_mean", 4), single(mcs, 40, "rank_coherence_npmi", 0)), sprintf("%s (rank %s)", single(mcs, 80, "coherence_npmi_mean", 4), single(mcs, 80, "rank_coherence_npmi", 0)), "delivered: model_comparison_summary.csv")
add_item(S, "NPMI mean excluding the notice copies, per seed", per_seed(function(f) mean(f$topic_npmi[!is_notice[[fit_key(f$K, f$seed)]]]), 40, 4), per_seed(function(f) mean(f$topic_npmi[!is_notice[[fit_key(f$K, f$seed)]]]), 80, 4), "derived here from the cached fits", "notice copies identified by cosine to the anchor (notice_topics_summary)")
add_item(S, "NPMI median over topics, per seed", triple(mcf, 40, "coherence_npmi_median", 4), triple(mcf, 80, "coherence_npmi_median", 4), "delivered: model_comparison_by_fit.csv")
add_item(S, "NPMI minimum over topics, per seed", triple(mcf, 40, "coherence_npmi_min", 4), triple(mcf, 80, "coherence_npmi_min", 4), "delivered: model_comparison_by_fit.csv")
add_item(S, "topics with NPMI below 0.1, per seed", per_seed(function(f) sum(f$topic_npmi < 0.1), 40, 0), per_seed(function(f) sum(f$topic_npmi < 0.1), 80, 0), "derived here from the cached fits")
add_item(S, "UMass coherence, mean over seeds", single(mcs, 40, "coherence_umass_mean", 3), single(mcs, 80, "coherence_umass_mean", 3), "delivered: model_comparison_summary.csv")
add_item(S, "exclusivity of the top 10 words, mean over seeds (rank)", sprintf("%s (rank %s)", single(mcs, 40, "exclusivity_mean", 3), single(mcs, 40, "rank_exclusivity", 0)), sprintf("%s (rank %s)", single(mcs, 80, "exclusivity_mean", 3), single(mcs, 80, "rank_exclusivity", 0)), "delivered: model_comparison_summary.csv")
add_item(S, "exclusivity mean excluding the notice copies, per seed", per_seed(function(f) mean(f$topic_exclusivity[!is_notice[[fit_key(f$K, f$seed)]]]), 40, 3), per_seed(function(f) mean(f$topic_exclusivity[!is_notice[[fit_key(f$K, f$seed)]]]), 80, 3), "derived here from the cached fits")
add_item(S, "Cao et al. 2009 (mean pairwise word cosine; lower is better) / Deveaud et al. 2014 (higher) / Arun et al. 2010 (lower), means over seeds", sprintf("%s / %s / %s", single(mcs, 40, "cao_juan_2009_mean", 4), single(mcs, 40, "deveaud_2014_mean", 3), single(mcs, 40, "arun_2010_mean", 4)), sprintf("%s / %s / %s", single(mcs, 80, "cao_juan_2009_mean", 4), single(mcs, 80, "deveaud_2014_mean", 3), single(mcs, 80, "arun_2010_mean", 4)), "delivered: model_comparison_summary.csv")
S <- "cross-seed stability"
add_item(S, "matched-topic cosine (Hungarian one-to-one, topic-word distributions): mean / median / minimum over the three seed pairs", sprintf("%s / %s / %s", single(ss, 40, "matched_cosine_mean", 3), single(ss, 40, "matched_cosine_median", 3), single(ss, 40, "matched_cosine_min", 3)), sprintf("%s / %s / %s", single(ss, 80, "matched_cosine_mean", 3), single(ss, 80, "matched_cosine_median", 3), single(ss, 80, "matched_cosine_min", 3)), "delivered: stability_summary.csv", "the minimum is a forced match of a surplus notice copy in both K (K = 40 seed 2 to seed 3; K = 80 seed 1 to seed 3)")
add_item(S, "rank of the matched cosine mean over the eight K", single(mcs, 40, "rank_stability", 0), single(mcs, 80, "rank_stability", 0), "delivered: model_comparison_summary.csv")
add_item(S, "share of matched pairs at cosine >= 0.8", single(ss, 40, "share_matched_cosine_at_least_threshold", 3), single(ss, 80, "share_matched_cosine_at_least_threshold", 3), "delivered: stability_summary.csv")
add_item(S, "share of matches that are not the row maximum", single(ss, 40, "share_matched_not_row_maximum", 3), single(ss, 80, "share_matched_not_row_maximum", 3), "delivered: stability_summary.csv")
add_item(S, "mean top-20 Jaccard of matched pairs", single(ss, 40, "jaccard_top_words_mean", 3), single(ss, 80, "jaccard_top_words_mean", 3), "delivered: stability_summary.csv")
add_item(S, "adjusted Rand index of dominant topics (document labels), mean over seed pairs", single(ss, 40, "ari_dominant_topics_mean", 3), single(ss, 80, "ari_dominant_topics_mean", 3), "delivered: stability_summary.csv")
add_item(S, "seed-1 topics matched at >= 0.8 in both other seeds (count; share)", sprintf("%d of 40; %.3f", sum(stab40$reproduced_in_all_other_seeds), mean(stab40$reproduced_in_all_other_seeds)), sprintf("%d of 80; %.3f", sum(stab80$reproduced_in_all_other_seeds), mean(stab80$reproduced_in_all_other_seeds)), "delivered: topic_stability_seed1_all_k.csv (recomputed here, equal)")
add_item(S, "seed-1 topics below 0.5 in both other seeds (no counterpart)", sprintf("%d: %s", sum(stab40$reproduced_in_no_other_seed), paste0("T", stab40$topic_a[stab40$reproduced_in_no_other_seed], collapse = ", ")), sprintf("%d: %s", sum(stab80$reproduced_in_no_other_seed), paste0("T", stab80$topic_a[stab80$reproduced_in_no_other_seed], collapse = ", ")), "delivered: topic_stability_seed1_all_k.csv")
add_item(S, "seed-1 topics below 0.5 in at least one other seed", sum(stab40$weaker_matched_cosine < 0.5), sum(stab80$weaker_matched_cosine < 0.5), "derived here from the matched pairs")
add_item(S, "mean of each seed-1 topic's weaker matched cosine", single(ss, 40, "seed1_weaker_matched_cosine_mean", 3), single(ss, 80, "seed1_weaker_matched_cosine_mean", 3), "delivered: stability_summary.csv")
add_item(S, "matched pairs below 0.5 per seed pair (1-2 / 1-3 / 2-3)", paste(vapply(seed_pairs, function(p) sum(stability_pairs$cosine[stability_pairs$K == 40 & stability_pairs$seed_a == p[1] & stability_pairs$seed_b == p[2]] < 0.5), integer(1)), collapse = " / "), paste(vapply(seed_pairs, function(p) sum(stability_pairs$cosine[stability_pairs$K == 80 & stability_pairs$seed_a == p[1] & stability_pairs$seed_b == p[2]] < 0.5), integer(1)), collapse = " / "), "derived here from the matched pairs")
S <- "duplicate and near-duplicate topics"
add_item(S, "notice copies per seed (1 / 2 / 3)", notice_n(40), notice_n(80), "derived here: notice_topics_summary.csv", sprintf("copy = cosine >= %.2f to the anchor topic", notice_threshold))
add_item(S, "topic pairs with word cosine >= 0.99, per seed (all are pairs of notice copies)", per_seed(function(f) sum(f$word_cos[upper.tri(f$word_cos)] >= 0.99), 40, 0), per_seed(function(f) sum(f$word_cos[upper.tri(f$word_cos)] >= 0.99), 80, 0), "derived here from the cached fits")
add_item(S, "topic pairs with word cosine >= 0.5 excluding notice pairs, per seed", per_seed(function(f) non_notice_pairs(f, f$word_cos, 0.5), 40, 0), per_seed(function(f) non_notice_pairs(f, f$word_cos, 0.5), 80, 0), "derived here from the cached fits")
add_item(S, "topic pairs with word cosine >= 0.7 excluding notice pairs, per seed", per_seed(function(f) non_notice_pairs(f, f$word_cos, 0.7), 40, 0), per_seed(function(f) non_notice_pairs(f, f$word_cos, 0.7), 80, 0), "derived here from the cached fits")
add_item(S, "most similar non-notice pair (word cosine), per seed", paste(vapply(fit_seeds, function(s) most_similar_non_notice(fits[[fit_key(40, s)]]), ""), collapse = " / "), paste(vapply(fit_seeds, function(s) most_similar_non_notice(fits[[fit_key(80, s)]]), ""), collapse = " / "), "derived here from the cached fits")
add_item(S, "topic pairs with top-20 Jaccard >= 0.3 excluding notice pairs, per seed", per_seed(function(f) non_notice_pairs(f, f$top_jaccard, 0.3), 40, 0), per_seed(function(f) non_notice_pairs(f, f$top_jaccard, 0.3), 80, 0), "derived here from the cached fits")
add_item(S, "largest document-share correlation between two topics excluding notice pairs, per seed", per_seed(function(f) { nn <- !is_notice[[fit_key(f$K, f$seed)]]; m <- f$doc_cor[nn, nn]; max(m[upper.tri(m)]) }, 40, 3), per_seed(function(f) { nn <- !is_notice[[fit_key(f$K, f$seed)]]; m <- f$doc_cor[nn, nn]; max(m[upper.tri(m)]) }, 80, 3), "derived here from the cached fits")
add_item(S, "most co-active pair: share of documents with both topics >= 0.1, per seed", triple(mcf, 40, "co_activation_pair_max", 4), triple(mcf, 80, "co_activation_pair_max", 4), "delivered: model_comparison_by_fit.csv")
add_item(S, "effective distinct topics = K minus surplus notice copies, per seed", paste(notice_summary$effective_distinct_topics[notice_summary$K == 40], collapse = " / "), paste(notice_summary$effective_distinct_topics[notice_summary$K == 80], collapse = " / "), "derived here: notice_topics_summary.csv")
S <- "convergence"
add_item(S, "fit-phase sampling iterations per seed (ceiling 1,000; stop when 10 iterations improve the pseudo log-likelihood by < 0.01%)", triple(mcf, 40, "iterations_fit", 0), triple(mcf, 80, "iterations_fit", 0), "delivered: model_comparison_by_fit.csv")
add_item(S, "reached the iteration ceiling", all(mcf$iterations_fit[mcf$K == 40] >= n_iter_max), all(mcf$iterations_fit[mcf$K == 80] >= n_iter_max), "derived here from model_comparison_by_fit.csv")
add_item(S, "held-out inference iterations (half A) / full held-out transform iterations, per seed", paste(triple(mcf, 40, "iterations_heldout_inference", 0), triple(mcf, 40, "iterations_heldout_full_transform", 0), sep = " ; "), paste(triple(mcf, 80, "iterations_heldout_inference", 0), triple(mcf, 80, "iterations_heldout_full_transform", 0), sep = " ; "), "delivered: model_comparison_by_fit.csv")
add_item(S, "final pseudo log-likelihood per training token, per seed", triple(mcf, 40, "final_pseudo_loglikelihood_per_token", 4), triple(mcf, 80, "final_pseudo_loglikelihood_per_token", 4), "delivered: model_comparison_by_fit.csv")
add_item(S, "relative improvement over the last 10 iterations, per seed", per_seed(function(f) trace_change(f, 1), 40, 6), per_seed(function(f) trace_change(f, 1), 80, 6), "derived here from the cached traces", "the stopping rule fires below 1e-4")
add_item(S, "relative improvement over the last 100 iterations, per seed", per_seed(function(f) trace_change(f, 10), 40, 5), per_seed(function(f) trace_change(f, 10), 80, 5), "derived here from the cached traces")
add_item(S, "trace checkpoints that decreased (non-monotone steps), per seed", per_seed(function(f) sum(diff(f$trace$loglikelihood) < 0), 40, 0), per_seed(function(f) sum(diff(f$trace$loglikelihood) < 0), 80, 0), "derived here from the cached traces")
S <- "topic prevalence and concentration"
add_item(S, "largest topic token share (training), per seed", triple(mcf, 40, "largest_topic_share", 4), triple(mcf, 80, "largest_topic_share", 4), "delivered: model_comparison_by_fit.csv", "the largest topic is a notice copy in every fit of both K")
add_item(S, "smallest topic token share (training), per seed", triple(mcf, 40, "smallest_topic_share", 4), triple(mcf, 80, "smallest_topic_share", 4), "delivered: model_comparison_by_fit.csv")
add_item(S, "Gini coefficient of topic token shares, per seed", per_seed(function(f) gini(f$token_share), 40, 3), per_seed(function(f) gini(f$token_share), 80, 3), "derived here from the cached fits")
add_item(S, "topics dominant in fewer than 1% of modeled documents, per seed", per_seed(function(f) sum(tabulate(f$dominant_m, f$K) < 0.01 * n_modeled), 40, 0), per_seed(function(f) sum(tabulate(f$dominant_m, f$K) < 0.01 * n_modeled), 80, 0), "derived here from the cached per-document summaries")
add_item(S, "median share of a document's dominant topic (all modeled documents), per seed", per_seed(function(f) median(f$max_share_m), 40, 3), per_seed(function(f) median(f$max_share_m), 80, 3), "derived here from the cached per-document summaries")
add_item(S, "share of modeled documents with a dominant share >= 0.5, per seed", per_seed(function(f) mean(f$max_share_m >= 0.5), 40, 3), per_seed(function(f) mean(f$max_share_m >= 0.5), 80, 3), "derived here from the cached per-document summaries")
add_item(S, "mean number of active topics (share >= 0.1) per document, per seed", per_seed(function(f) mean(f$n_active_m), 40, 3), per_seed(function(f) mean(f$n_active_m), 80, 3), "derived here from the cached per-document summaries")
cs40 <- retained_new$concentration_summary; cs80 <- retained_ref$concentration_summary
for (b in c("5-9", "10-24", "25-49")) add_item(S, sprintf("median dominant share, documents of %s modeled tokens (seed 1)", b), sprintf("%.3f", cs40$dominant_share_median[cs40$token_band == b]), sprintf("%.3f", cs80$dominant_share_median[cs80$token_band == b]), "this package: document_concentration_summary_K040.csv; delivered: document_concentration_summary_K080.csv")
S <- "representative documents"
rp <- representative_document_proxies
rp_item <- function(K, fun, digits = 3) paste(vapply(fit_seeds, function(s) formatC(fun(rp[rp$K == K & rp$seed == s, ]), format = "f", digits = digits), ""), collapse = " / ")
add_item(S, "mean topic share of the five representative documents, mean over topics, per seed", rp_item(40, function(d) mean(d$top5_mean_share)), rp_item(80, function(d) mean(d$top5_mean_share)), "derived here from representative_documents_all_models.csv")
add_item(S, "topics whose five documents average a share below 0.5, per seed", rp_item(40, function(d) sum(d$flag_mean_share_below_0.5), 0), rp_item(80, function(d) sum(d$flag_mean_share_below_0.5), 0), "derived here")
add_item(S, "topics with fewer than five documents dominated by the topic, per seed", rp_item(40, function(d) sum(d$flag_not_all_dominated), 0), rp_item(80, function(d) sum(d$flag_not_all_dominated), 0), "derived here")
add_item(S, "mean count of the topic's ten top words present per document, per seed", rp_item(40, function(d) mean(d$top_10_words_present_mean)), rp_item(80, function(d) mean(d$top_10_words_present_mean)), "derived here (tokens recomputed by the delivered rules)")
add_item(S, "topics whose documents contain on average fewer than two of the ten top words, per seed", rp_item(40, function(d) sum(d$flag_fewer_than_2_top_words_on_average), 0), rp_item(80, function(d) sum(d$flag_fewer_than_2_top_words_on_average), 0), "derived here")
add_item(S, sprintf("topics with at least one repetitive document (type-token ratio < %.1f), per seed", repetitive_ttr), rp_item(40, function(d) sum(d$has_repetitive_document), 0), rp_item(80, function(d) sum(d$has_repetitive_document), 0), "derived here")
add_item(S, "representative rows written by AutoModerator, per seed", rp_item(40, function(d) sum(d$automoderator_rows), 0), rp_item(80, function(d) sum(d$automoderator_rows), 0), "derived here")
S <- "runtime and memory (fitting run, 12 workers)"
rt <- function(K, col, digits = 0) paste(formatC(frt[[col]][frt$K == K & frt$role == "primary"][order(frt$seed[frt$K == K & frt$role == "primary"])], format = "f", digits = digits, big.mark = ","), collapse = " / ")
add_item(S, "worker job wall seconds, per seed", rt(40, "elapsed_seconds"), rt(80, "elapsed_seconds"), "delivered: fit_runtime.csv")
add_item(S, "fit-to-held-out-completion seconds, per seed", rt(40, "elapsed_seconds_fit_to_completion"), rt(80, "elapsed_seconds_fit_to_completion"), "delivered: fit_runtime.csv")
add_item(S, "CPU user seconds of the job, per seed", rt(40, "cpu_user_seconds_total"), rt(80, "cpu_user_seconds_total"), "delivered: fit_runtime.csv")
add_item(S, "seconds per fit iteration, per seed", rt(40, "seconds_per_fit_iteration", 2), rt(80, "seconds_per_fit_iteration", 2), "delivered: fit_runtime.csv")
add_item(S, "worker peak working set MB (lifetime peak; clean per-K reading only for a worker's first job), per seed", rt(40, "process_peak_working_set_mb"), rt(80, "process_peak_working_set_mb"), "delivered: fit_runtime.csv", paste("job index in worker:", rt(40, "job_index_in_worker"), "|", rt(80, "job_index_in_worker")))
add_item(S, "cached theta file size (MB, uncompressed; all modeled documents x K)", sprintf("%.0f", file.size(theta_files[["40"]]) / 1024^2), sprintf("%.0f", file.size(theta_files[["80"]]) / 1024^2), "file sizes on disk", "document-topic table sizes are in preservation_status.csv and the delivered manifests")
S <- "retention rule (delivered) and selection"
add_item(S, "composite rank (mean of the four ranks; lowest wins)", single(mcs, 40, "composite_rank", 2), single(mcs, 80, "composite_rank", 2), "delivered: model_comparison_summary.csv")
add_item(S, "retained by the delivered rule; basis", sprintf("%s; %s", mcs$retained[mcs$K == 40], coalesce(mcs$retention_basis[mcs$K == 40], "")), sprintf("%s; %s", mcs$retained[mcs$K == 80], coalesce(mcs$retention_basis[mcs$K == 80], "")), "delivered: model_comparison_summary.csv")
add_item(S, "status in this package", "user-selected candidate representation; retained outputs written here", "user-selected candidate representation; delivered retained outputs unchanged and hash-verified", "this package")
S <- "cross-K correspondence (seed 1; cosine of topic-word distributions)"
cc <- topic_correspondence_K040; pp <- topic_predecessors_K080
add_item(S, "K = 40 topics with one K = 80 counterpart at >= 0.5 / with two or more (split) / with none", sprintf("%d / %d / %d", sum(cc$n_K080_topics_at_or_above_0.5 == 1), sum(cc$n_K080_topics_at_or_above_0.5 >= 2), sum(cc$n_K080_topics_at_or_above_0.5 == 0)), "", "this package: topic_correspondence_K040_to_K080.csv")
add_item(S, "K = 40 topics whose best K = 80 counterpart reaches 0.8 / 0.9", sprintf("%d / %d", sum(cc$best_cosine >= 0.8), sum(cc$best_cosine >= 0.9)), "", "this package")
add_item(S, "K = 80 topics with a clear predecessor (>= 0.8) / a moderate one (0.5 to 0.8) / none (< 0.5)", "", sprintf("%d / %d / %d", sum(pp$best_cosine >= 0.8), sum(pp$best_cosine >= 0.5 & pp$best_cosine < 0.8), sum(pp$best_cosine < 0.5)), "this package: topic_predecessors_K080_from_K040.csv")
add_item(S, "K = 80 topics without predecessor: reproduced at >= 0.8 in both other seeds / no counterpart in either other seed", "", sprintf("%d / %d", sum(pp$reproduced_in_both_other_seeds_K080[pp$best_cosine < 0.5]), sum(pp$no_counterpart_in_other_seeds_K080[pp$best_cosine < 0.5])), "this package")
add_item(S, "mutual best pairs (each is the other's best match)", sum(cc$best_is_mutual), sum(pp$best_is_mutual), "this package")
add_item(S, "share of documents dominated by a K = 40 topic that are dominated by one of its counterparts at >= 0.5 (mean over topics, document-weighted)", sprintf("%.3f", sum(cc$share_of_dominant_docs_to_K080_correspondents * cc$dominant_docs_K040) / sum(cc$dominant_docs_K040)), "", "this package: dominant-topic cross-tabulation")
comparison_summary <- bind_rows(cmp)
mark_stage("comparison_summary")

# ---- Figures ----
# INPUT : the K = 40 retained tables; the cross-K matrices.
# DOES  : the nine retained-model figures of the delivered run for K = 40 (the
#         delivered figure code, unchanged) and two cross-K matrices. Every
#         figure carries a title, axis titles, tick labels and a legend where a
#         colour scale is shown; every topic x topic figure names the quantity
#         it shows.
# OUTPUT: figures/*.png.
ink_series   <- "#2a78d6"
ink_series_2 <- "#eb6834"
ink_red      <- "#e34948"
ink_text     <- "#0b0b0b"
ink_sub      <- "#52514e"
ink_muted    <- "#898781"
grid_col     <- "#e1e0d9"
axis_col     <- "#c3c2b7"
mid_grey     <- "#f0efec"
blue_ramp    <- c("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")
theme_fig <- theme_minimal(base_size = 12) +
  theme(panel.grid.minor = element_blank(),
        panel.grid.major = element_line(colour = grid_col, linewidth = 0.3),
        axis.line.x      = element_line(colour = axis_col, linewidth = 0.4),
        axis.text        = element_text(colour = ink_muted),
        axis.title       = element_text(colour = ink_sub),
        plot.title       = element_text(colour = ink_text, face = "bold"),
        strip.text       = element_text(colour = ink_sub, face = "bold", hjust = 0),
        legend.title     = element_text(colour = ink_sub),
        legend.text      = element_text(colour = ink_muted),
        plot.background  = element_rect(fill = "white", colour = NA))
condition_label <- "Tier 2 stopwords"
matrix_figure <- function(long, x_col, y_col, fill_col, fill_name, title, x_lab, y_lab, limits, colours, K, x_levels, y_levels, diverging = FALSE) {
  d <- long |> mutate(label_x = factor(paste("Topic", .data[[x_col]]), levels = x_levels),
                      label_y = factor(paste("Topic", .data[[y_col]]), levels = y_levels))
  p <- ggplot(d, aes(x = label_x, y = label_y, fill = .data[[fill_col]])) + geom_tile()
  p <- if (diverging) p + scale_fill_gradient2(low = ink_series, mid = mid_grey, high = ink_red, midpoint = 0, limits = limits, name = fill_name,
                                               breaks = breaks_pretty(n = 5)) else
    p + scale_fill_gradientn(colours = colours, limits = limits, name = fill_name, breaks = breaks_pretty(n = 5))
  p + labs(title = title, x = x_lab, y = y_lab) +
    theme_fig + theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1, size = 7), axis.text.y = element_text(size = 7),
                      legend.position = "bottom", legend.title.position = "top", legend.key.width = grid::unit(2.2, "cm"),
                      panel.grid.major = element_blank())
}
matrix_size <- function(K) list(width = max(11, 3 + 0.2 * K), height = max(10, 3.5 + 0.2 * K))

r <- retained_new; K <- r$K; tag <- r$tag
topic_levels_desc <- rev(r$topic_labels)
ms <- matrix_size(K)
prev_long <- r$topic_summary |>
  select(topic_label, `Share of modeled tokens` = token_share, `Mean share within documents` = doc_mean_share) |>
  pivot_longer(-topic_label, names_to = "measure", values_to = "share") |>
  mutate(topic_label = factor(topic_label, levels = topic_levels_desc),
         measure = factor(measure, levels = c("Share of modeled tokens", "Mean share within documents")))
fig4 <- ggplot(prev_long, aes(x = share, y = topic_label, fill = measure)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7) +
  scale_fill_manual(values = c(ink_series, ink_series_2), name = "Measure") +
  scale_x_continuous(labels = label_percent(accuracy = 0.1), expand = expansion(mult = c(0, 0.05))) +
  labs(title = sprintf("Topic prevalence, K = %d, %s — r/politics, July 2026", K, condition_label), x = "Share", y = NULL) +
  theme_fig + theme(legend.position = "bottom", panel.grid.major.y = element_blank())
ggsave(file.path(fig_dir, sprintf("fig4_topic_prevalence_%s.png", tag)), fig4, width = 9, height = 3 + 0.22 * K, dpi = 150, bg = "white", limitsize = FALSE)

tw <- top_words_probability |>
  filter(K == !!K, seed == fit_seeds[1], rank <= 10) |>
  arrange(topic, rank) |>
  mutate(topic_label = factor(paste("Topic", topic), levels = r$topic_labels), key = paste0(term, "  #", topic)) |>
  mutate(key = factor(key, levels = rev(unique(key))))
fig_cols <- if (K <= 12) 3 else if (K <= 30) 5 else if (K <= 60) 6 else 8
fig5 <- ggplot(tw, aes(x = phi, y = key)) +
  geom_col(fill = ink_series, width = 0.7) +
  facet_wrap(~ topic_label, scales = "free", ncol = fig_cols) +
  scale_y_discrete(labels = function(x) sub("  #.*$", "", x)) +
  scale_x_continuous(labels = label_number(accuracy = 0.001), n.breaks = 3, expand = expansion(mult = c(0, 0.05))) +
  labs(title = sprintf("Highest-probability words per topic, K = %d, %s — r/politics, July 2026", K, condition_label),
       x = "Probability of the word in the topic", y = NULL) +
  theme_fig + theme(panel.grid.major.y = element_blank(), axis.text.y = element_text(size = 8), axis.text.x = element_text(size = 7))
ggsave(file.path(fig_dir, sprintf("fig5_top_words_%s.png", tag)), fig5, width = 2.6 * fig_cols + 1, height = 2.2 * ceiling(K / fig_cols) + 1, dpi = 130, bg = "white", limitsize = FALSE)

fig6 <- r$topic_prevalence_by_day |>
  mutate(topic_label = factor(paste("Topic", topic), levels = topic_levels_desc)) |>
  ggplot(aes(x = created_date_utc, y = topic_label, fill = token_share)) +
  geom_tile() +
  scale_fill_gradientn(colours = blue_ramp, labels = label_percent(accuracy = 1), name = "Share of the day's modeled tokens") +
  scale_x_date(date_breaks = "5 days", date_labels = "%b %d", expand = expansion(mult = 0.01)) +
  labs(title = sprintf("Topic prevalence by day, K = %d, %s — r/politics, July 2026 (UTC)", K, condition_label), x = "Day (UTC)", y = NULL) +
  theme_fig + theme(legend.position = "bottom", panel.grid.major = element_blank())
ggsave(file.path(fig_dir, sprintf("fig6_topic_prevalence_by_day_%s.png", tag)), fig6, width = 10, height = 3 + 0.2 * K, dpi = 150, bg = "white", limitsize = FALSE)

fig7 <- r$concentration |>
  mutate(token_band = factor(paste(token_band, "modeled tokens"), levels = paste(band_labels, "modeled tokens"))) |>
  ggplot() +
  geom_rect(aes(xmin = share_lower, xmax = share_upper, ymin = 0, ymax = prop_in_band), fill = ink_series) +
  facet_wrap(~ token_band, ncol = 4) +
  scale_x_continuous(breaks = seq(0, 1, by = 0.25), limits = c(0, 1)) +
  scale_y_continuous(labels = label_percent(accuracy = 1), expand = expansion(mult = c(0, 0.05))) +
  labs(title = sprintf("Share of the dominant topic within documents, K = %d, %s — r/politics, July 2026", K, condition_label),
       x = "Share of the document's tokens in its dominant topic (bins of 0.05)", y = "Share of documents in the length band") +
  theme_fig
ggsave(file.path(fig_dir, sprintf("fig7_document_concentration_%s.png", tag)), fig7, width = 11, height = 6, dpi = 150, bg = "white")

cor_long <- r$topic_correlation |> select(-quantity) |>
  pivot_longer(-topic, names_to = "topic_b", values_to = "correlation") |>
  mutate(topic_b = as.integer(sub("topic_", "", topic_b))) |> filter(topic != topic_b)
cor_limit <- max(abs(cor_long$correlation))
fig8 <- matrix_figure(cor_long, "topic_b", "topic", "correlation", "Pearson correlation of the two topics' document shares (theta)",
                      sprintf("Topic co-occurrence within documents, K = %d, %s — r/politics, July 2026", K, condition_label),
                      "Topic (document shares)", "Topic (document shares)", c(-cor_limit, cor_limit), NULL, K, r$topic_labels, topic_levels_desc, diverging = TRUE)
ggsave(file.path(fig_dir, sprintf("fig8_topic_correlation_%s.png", tag)), fig8, width = ms$width, height = ms$height, dpi = 150, bg = "white", limitsize = FALSE)

sc <- stability_cos[[paste(K, fit_seeds[2], sep = "_")]]
fig9 <- matrix_figure(r$stability_matrices[[as.character(fit_seeds[2])]], "seed_b_topic", "seed1_topic", "cosine",
                      "Cosine similarity of topic-word distributions (phi)",
                      sprintf("Topic matching between seeds 1 and 2, K = %d, %s — r/politics, July 2026", K, condition_label),
                      "Seed 2 topics, in matched order (topic-word distributions)", "Seed 1 topics (topic-word distributions)", c(0, 1), blue_ramp, K,
                      paste("Topic", sc$assignment), topic_levels_desc)
ggsave(file.path(fig_dir, sprintf("fig9_stability_matrix_%s.png", tag)), fig9, width = ms$width, height = ms$height, dpi = 150, bg = "white", limitsize = FALSE)

fig10 <- r$topic_prevalence_by_type |>
  mutate(topic_label = factor(paste("Topic", topic), levels = topic_levels_desc),
         doc_type = factor(doc_type, levels = c("comment", "submission"), labels = c("Comments", "Submissions"))) |>
  ggplot(aes(x = token_share, y = topic_label, fill = doc_type)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7) +
  scale_fill_manual(values = c(ink_series, ink_series_2), name = "Document type") +
  scale_x_continuous(labels = label_percent(accuracy = 0.1), expand = expansion(mult = c(0, 0.05))) +
  labs(title = sprintf("Topic prevalence by document type, K = %d, %s — r/politics, July 2026", K, condition_label),
       x = "Share of the document type's modeled tokens", y = NULL) +
  theme_fig + theme(legend.position = "bottom", panel.grid.major.y = element_blank())
ggsave(file.path(fig_dir, sprintf("fig10_topic_prevalence_by_type_%s.png", tag)), fig10, width = 9, height = 3 + 0.22 * K, dpi = 150, bg = "white", limitsize = FALSE)

wc_long <- r$topic_word_cosine |> select(-quantity) |>
  pivot_longer(-topic, names_to = "topic_b", values_to = "cosine") |>
  mutate(topic_b = as.integer(sub("topic_", "", topic_b))) |> filter(topic != topic_b)
fig11 <- matrix_figure(wc_long, "topic_b", "topic", "cosine", "Cosine similarity of the two topics' topic-word distributions (phi)",
                       sprintf("Topic-word similarity between topics, K = %d, %s — r/politics, July 2026", K, condition_label),
                       "Topic (topic-word distributions)", "Topic (topic-word distributions)", c(0, 1), blue_ramp, K, r$topic_labels, topic_levels_desc)
ggsave(file.path(fig_dir, sprintf("fig11_topic_word_cosine_%s.png", tag)), fig11, width = ms$width, height = ms$height, dpi = 150, bg = "white", limitsize = FALSE)

ca_long <- r$co_activation |> select(-quantity) |>
  pivot_longer(-topic, names_to = "topic_b", values_to = "share") |>
  mutate(topic_b = as.integer(sub("topic_", "", topic_b))) |> filter(topic != topic_b)
fig12 <- matrix_figure(ca_long, "topic_b", "topic", "share", sprintf("Share of modeled documents with both topics at or above %g (document shares)", active_topic_threshold),
                       sprintf("Topic co-activation within documents, K = %d, %s — r/politics, July 2026", K, condition_label),
                       "Topic (document shares)", "Topic (document shares)", c(0, max(ca_long$share)), blue_ramp, K, r$topic_labels, topic_levels_desc)
ggsave(file.path(fig_dir, sprintf("fig12_topic_coactivation_%s.png", tag)), fig12, width = ms$width, height = ms$height, dpi = 150, bg = "white", limitsize = FALSE)

# Cross-K figures: K = 40 topics (rows) against K = 80 topics (columns ordered by their best K = 40 predecessor, then by cosine).
k80_order <- topic_predecessors_K080 |> arrange(best_topic_K040, desc(best_cosine)) |> pull(topic_K080)
cross_levels_x <- paste("Topic", k80_order)
cross_levels_y <- rev(paste("Topic", seq_len(k_new)))
cross_ms <- list(width = max(11, 3 + 0.2 * k_reference), height = max(10, 3.5 + 0.2 * k_new))
figc1 <- matrix_figure(cross_k_pairs, "topic_K080", "topic_K040", "word_cosine", "Cosine similarity of the two topics' topic-word distributions (phi)",
                       sprintf("Topic correspondence between K = 40 and K = 80, %s — r/politics, July 2026", condition_label),
                       "K = 80 topics, ordered by best K = 40 match (topic-word distributions)", "K = 40 topics (topic-word distributions)",
                       c(0, 1), blue_ramp, k_reference, cross_levels_x, cross_levels_y)
ggsave(file.path(fig_dir, "figc1_cross_k_topic_word_cosine_K040_K080.png"), figc1, width = cross_ms$width, height = cross_ms$height, dpi = 150, bg = "white", limitsize = FALSE)
figc2 <- matrix_figure(cross_k_pairs, "topic_K080", "topic_K040", "share_of_K040_dominant_docs", "Share of the documents dominated by the K = 40 topic that are dominated by the K = 80 topic (document shares)",
                       sprintf("Dominant-topic flow from K = 40 to K = 80, %s — r/politics, July 2026", condition_label),
                       "K = 80 topics, ordered by best K = 40 match (document shares)", "K = 40 topics (document shares)",
                       c(0, max(cross_k_pairs$share_of_K040_dominant_docs)), blue_ramp, k_reference, cross_levels_x, cross_levels_y)
ggsave(file.path(fig_dir, "figc2_dominant_topic_flow_K040_K080.png"), figc2, width = cross_ms$width, height = cross_ms$height, dpi = 150, bg = "white", limitsize = FALSE)
mark_stage("figures")
log_line("figures written")

# ---- Validation checks ----
# INPUT : everything above.
# DOES  : integrity of the inputs; agreement of the recomputed stability and
#         representative documents with the delivered tables; the K = 80
#         reproduction hashes; the delivered script's own retained-model checks
#         applied to K = 40; the cross-K, notice and inspection tables'
#         structure. The script stops before writing any other table if a
#         check fails.
# OUTPUT: validation_checks.
checks <- list()
add_check <- function(check, detail, pass) checks[[length(checks) + 1L]] <<- tibble(check = check, detail = detail, pass = isTRUE(pass))
add_check("inputs: every delivered table and every cached model file read here has the SHA256 recorded in its delivered manifest (the delivered outputs and the cache are unchanged)",
          sprintf("%d of %d files match", sum(input_integrity$hash_equals_manifest), nrow(input_integrity)), all(input_integrity$hash_equals_manifest))
add_check("stability: the cross-seed matching recomputed here for K = 40 and K = 80 equals the delivered stability_pairs.csv (assignments identical; cosines, Jaccard, best any-topic cosine and ARI equal to 1e-9); every assignment is a permutation",
          sprintf("agrees %s; %d of %d permutations", stability_agrees, sum(stability_assignment_ok), length(stability_assignment_ok)), stability_agrees && all(stability_assignment_ok))
sts_delivered <- delivered$topic_stability_seed1_all_k |> filter(K %in% selected_k) |> arrange(K, topic_a)
sts_here <- seed1_topic_stability |> arrange(K, topic_a)
add_check("stability: the seed-1 topic reproducibility table recomputed here equals the delivered topic_stability_seed1_all_k.csv for K = 40 and 80",
          sprintf("%d rows; max cosine difference %g", nrow(sts_here), max(abs(sts_delivered$weaker_matched_cosine - sts_here$weaker_matched_cosine), abs(sts_delivered$stronger_matched_cosine - sts_here$stronger_matched_cosine))),
          nrow(sts_delivered) == nrow(sts_here) && max(abs(sts_delivered$weaker_matched_cosine - sts_here$weaker_matched_cosine)) < 1e-9 &&
            identical(sts_delivered$reproduced_in_all_other_seeds, sts_here$reproduced_in_all_other_seeds) && identical(sts_delivered$reproduced_in_no_other_seed, sts_here$reproduced_in_no_other_seed))
add_check("representative documents: the rows rebuilt here from the workers' indices for the six fits equal the delivered representative_documents_all_models.csv rows (doc_key, text, share, dominant topic)",
          sprintf("%d rows; agrees %s", nrow(rd_here), representative_agrees), representative_agrees)
add_check("K = 80 reproduction: every delivered K = 80 retained table regenerated here has the SHA256 recorded in its delivered manifest, and every delivered file on disk equals its manifest",
          sprintf("%d of %d regenerated tables equal the manifests; %d of %d delivered files equal the manifests", sum(k80_reproduction$regenerated_equals_manifest), nrow(k80_reproduction),
                  sum(k80_reproduction$delivered_file_equals_manifest), nrow(k80_reproduction)),
          all(k80_reproduction$regenerated_equals_manifest) && all(k80_reproduction$delivered_file_equals_manifest))
for (rr in list(retained_ref, retained_new)) {
  add_check(sprintf("%s: theta file from the worker is consistent (doc keys in modeled order, K, seed, sampler index, dimensions) and its per-document summaries equal the worker's", rr$tag),
            sprintf("file consistent %s; summaries identical %s", rr$theta_file_consistent, rr$worker_summaries_identical), rr$theta_file_consistent && rr$worker_summaries_identical)
}
r <- retained_new; K <- r$K; tag <- r$tag; dt <- r$document_topic_table; tc <- r$topic_cols
theta_mat <- as.matrix(dt[, tc])
modeled_dt <- dt$model_status == "modeled"
row_sums <- rowSums(theta_mat[modeled_dt, , drop = FALSE])
add_check(sprintf("%s: document-topic table has one row per source document; modeled rows sum to 1 (rounded to 6 decimals); non-modeled rows are NA", tag),
          sprintf("%d rows, %d distinct keys; %d modeled rows, sums %.6f..%.6f; %d NA cells in non-modeled rows of %d",
                  nrow(dt), n_distinct(dt$doc_key), sum(modeled_dt), min(row_sums), max(row_sums), sum(is.na(theta_mat[!modeled_dt, ])), sum(!modeled_dt) * K),
          nrow(dt) == n_docs && n_distinct(dt$doc_key) == n_docs && sum(modeled_dt) == n_modeled &&
            all(abs(row_sums - 1) < 1e-4) && all(is.na(theta_mat[!modeled_dt, ])) && !anyNA(theta_mat[modeled_dt, ]))
src <- docs |> select(doc_key, s_author = author, s_created = created_utc, s_link = link_id, s_parent = parent_id, s_sub = submission_id, s_type = doc_type)
joined <- dt |> inner_join(src, by = "doc_key")
n_agree <- with(joined, sum(author == s_author & created_utc == s_created & link_id == s_link & submission_id == s_sub & doc_type == s_type &
                              (is.na(parent_id) & is.na(s_parent) | !is.na(parent_id) & !is.na(s_parent) & parent_id == s_parent)))
add_check(sprintf("%s: identifiers and metadata in the document-topic table agree with the source when re-joined by doc_key", tag),
          sprintf("%d of %d rows agree", n_agree, n_docs), n_agree == n_docs && nrow(joined) == n_docs)
dom_check <- max.col(theta_mat[modeled_dt, , drop = FALSE], ties.method = "first")
add_check(sprintf("%s: dominant topic column equals the arg max of the stored probabilities (ties: first) on modeled rows", tag),
          sprintf("%d of %d rows agree", sum(dom_check == dt$dominant_topic[modeled_dt]), sum(modeled_dt)),
          all(dom_check == dt$dominant_topic[modeled_dt]) && all(is.na(dt$dominant_topic[!modeled_dt])))
rm(theta_mat, joined, src)
ts <- r$topic_summary
add_check(sprintf("%s: prevalence: token shares, document-mean shares and dominant counts reconcile; shares by type and by day sum to 1; numbering follows training shares", tag),
          sprintf("token %.9f; doc-mean %.9f; dominant %d = %d; type sums %.6f..%.6f; day sums %.6f..%.6f over %d days; training shares descending %s",
                  sum(ts$token_share), sum(ts$doc_mean_share), sum(ts$dominant_docs), n_modeled,
                  min(tapply(r$topic_prevalence_by_type$token_share, r$topic_prevalence_by_type$doc_type, sum)), max(tapply(r$topic_prevalence_by_type$token_share, r$topic_prevalence_by_type$doc_type, sum)),
                  min(tapply(r$topic_prevalence_by_day$token_share, r$topic_prevalence_by_day$created_date_utc, sum)), max(tapply(r$topic_prevalence_by_day$token_share, r$topic_prevalence_by_day$created_date_utc, sum)),
                  n_distinct(r$topic_prevalence_by_day$created_date_utc), all(diff(ts$token_share_train) <= 1e-12)),
          abs(sum(ts$token_share) - 1) < 1e-9 && abs(sum(ts$doc_mean_share) - 1) < 1e-9 && sum(ts$dominant_docs) == n_modeled &&
            all(abs(tapply(r$topic_prevalence_by_type$token_share, r$topic_prevalence_by_type$doc_type, sum) - 1) < 1e-9) &&
            all(abs(tapply(r$topic_prevalence_by_day$token_share, r$topic_prevalence_by_day$created_date_utc, sum) - 1) < 1e-9) &&
            n_distinct(r$topic_prevalence_by_day$created_date_utc) == n_distinct(docs$created_date_utc[modeled_rows]) &&
            all(diff(ts$token_share_train) <= 1e-12) && abs(sum(ts$token_share_train) - 1) < 1e-9)
rd <- r$representative_documents
rd_check <- rd |> inner_join(dt |> select(doc_key, all_of(tc)), by = "doc_key") |> inner_join(docs |> select(doc_key, s_text = text, s_tokens = n_tokens_modeled), by = "doc_key")
rd_theta <- as.matrix(rd_check[, tc])[cbind(seq_len(nrow(rd_check)), rd_check$topic)]
add_check(sprintf("%s: representative documents: share agrees with the table (6 decimals), text equals the source, >= %d modeled tokens, distinct texts and authors, %d rows per topic", tag, representative_min_tokens, n_representative),
          sprintf("%d rows; max share difference %g; %d texts agree; min tokens %d", nrow(rd), max(abs(rd_theta - rd_check$topic_share_in_document)), sum(rd_check$text == rd_check$s_text), min(rd_check$s_tokens)),
          nrow(rd_check) == nrow(rd) && max(abs(rd_theta - rd_check$topic_share_in_document)) < 1e-6 && all(rd_check$text == rd_check$s_text) &&
            min(rd_check$s_tokens) >= representative_min_tokens && all(tapply(rd$text, rd$topic, n_distinct) == n_representative) &&
            all(tapply(rd$author, rd$topic, n_distinct) == n_representative) && nrow(rd) == K * n_representative)
tw_mat <- as.matrix(r$topic_word_table[, tc])
add_check(sprintf("%s: topic-word table has one row per vocabulary term and every topic column sums to 1 (6 significant digits)", tag),
          sprintf("%d rows; column sums %.6f..%.6f", nrow(tw_mat), min(colSums(tw_mat)), max(colSums(tw_mat))),
          nrow(tw_mat) == n_vocabulary && all(abs(colSums(tw_mat) - 1) < 1e-4) && identical(r$topic_word_table$term, vocabulary))
rm(tw_mat)
cor_m <- as.matrix(r$topic_correlation[, tc]); wc_m <- as.matrix(r$topic_word_cosine[, tc]); ca_m <- as.matrix(r$co_activation[, tc])
add_check(sprintf("%s: concentration bins contain every modeled document; the three K x K matrices are symmetric with unit diagonal (correlation, word cosine) or a diagonal equal to the topic's active share recomputed from the unrounded theta file (co-activation)", tag),
          sprintf("%d docs in bins; %d x %d; max |co-activation diagonal - active share| %g", sum(r$concentration$docs), nrow(cor_m), length(tc), max(abs(diag(ca_m) - r$active_share))),
          sum(r$concentration$docs) == n_modeled && nrow(cor_m) == K && all(abs(diag(cor_m) - 1) < 1e-9) && isSymmetric(unname(cor_m)) &&
            all(abs(diag(wc_m) - 1) < 1e-9) && isSymmetric(unname(wc_m)) && isSymmetric(unname(ca_m)) && all(abs(diag(ca_m) - r$active_share) < 1e-9))
sm40 <- lapply(r$stability_matrices, function(m) sum(m$matched))
add_check(sprintf("%s: the two cross-seed matrices have K x K rows, cosines in [0, 1] and exactly K matched cells forming a permutation", tag),
          sprintf("rows %s; matched cells %s", paste(vapply(r$stability_matrices, nrow, 1L), collapse = ", "), paste(unlist(sm40), collapse = ", ")),
          all(vapply(r$stability_matrices, nrow, 1L) == K * K) && all(unlist(sm40) == K) &&
            all(vapply(r$stability_matrices, function(m) all(m$cosine >= 0 & m$cosine <= 1 + 1e-9) && n_distinct(m$seed_b_topic[m$matched]) == K, logical(1))))
tp_delivered <- delivered$topic_prevalence_all_models |> filter(K %in% selected_k) |> arrange(K, seed, topic)
tp_here <- bind_rows(lapply(fits, function(f) tibble(K = f$K, seed = f$seed, topic = seq_len(f$K), token_share = f$token_share, words = paste_top(f, f$top_probability)))) |> arrange(K, seed, topic)
add_check("the cached fits agree with the delivered topic_prevalence_all_models.csv for the six fits (training token shares to 1e-9; the ten-word strings identical)",
          sprintf("%d rows; max share difference %g; %d of %d word strings identical", nrow(tp_here), max(abs(tp_delivered$token_share - tp_here$token_share)), sum(tp_delivered$top_10_words_by_probability == tp_here$words), nrow(tp_here)),
          nrow(tp_delivered) == nrow(tp_here) && max(abs(tp_delivered$token_share - tp_here$token_share)) < 1e-9 && all(tp_delivered$top_10_words_by_probability == tp_here$words))
add_check("cross-K: the seed-1 cosine matrix is 40 x 80 with values in [0, 1]; the long table has 40 x 80 x 9 rows; the wide table has 40 rows and 80 topic columns; the dominant-topic cross-tabulation sums to the modeled documents with row and column sums equal to the two topic summaries' dominant counts",
          sprintf("%d x %d; range %.4f..%.4f; long %d rows; wide %d x %d; crosstab sum %d", nrow(cross_cos), ncol(cross_cos), min(cross_cos), max(cross_cos), nrow(cross_k_all_seed_pairs), nrow(cross_k_wide), ncol(cross_k_wide) - 2, sum(crosstab)),
          identical(dim(cross_cos), c(k_new, k_reference)) && all(cross_cos >= 0 & cross_cos <= 1 + 1e-9) && nrow(cross_k_all_seed_pairs) == k_new * k_reference * length(fit_seeds)^2 &&
            nrow(cross_k_wide) == k_new && ncol(cross_k_wide) == k_reference + 2 && sum(crosstab) == n_modeled &&
            all(rowSums(crosstab) == retained_new$topic_summary$dominant_docs) && all(colSums(crosstab) == retained_ref$topic_summary$dominant_docs) &&
            all(abs(rowSums(row_share) - 1) < 1e-9) && all(abs(colSums(col_share) - 1) < 1e-9))
add_check("cross-K: every K = 40 topic and every K = 80 topic appears exactly once in the correspondence document table (five documents each); the inspection table has one row per pair at or above the threshold plus one per K = 80 topic without predecessor; the per-topic tables have one row per topic",
          sprintf("document table %d rows (%d K = 40 topics, %d K = 80 topics); inspection %d rows = %d + %d; per-topic %d and %d rows",
                  nrow(correspondence_representative_documents), n_distinct(correspondence_representative_documents$topic[correspondence_representative_documents$K == k_new]),
                  n_distinct(correspondence_representative_documents$topic[correspondence_representative_documents$K == k_reference]),
                  nrow(correspondence_inspection), sum(cross_k_pairs$at_or_above_correspondence_threshold), nrow(no_pred), nrow(topic_correspondence_K040), nrow(topic_predecessors_K080)),
          nrow(correspondence_representative_documents) == (k_new + k_reference) * n_representative &&
            n_distinct(correspondence_representative_documents$topic[correspondence_representative_documents$K == k_new]) == k_new &&
            n_distinct(correspondence_representative_documents$topic[correspondence_representative_documents$K == k_reference]) == k_reference &&
            nrow(correspondence_inspection) == sum(cross_k_pairs$at_or_above_correspondence_threshold) + nrow(no_pred) &&
            nrow(topic_correspondence_K040) == k_new && nrow(topic_predecessors_K080) == k_reference &&
            all(topic_correspondence_K040$best_cosine == apply(cross_cos, 1, max)) && sum(topic_correspondence_K040$n_K080_topics_at_or_above_0.5) == sum(cross_cos >= correspondence_threshold))
notice_gap_ok <- all(vapply(names(fits), function(k) { cs <- notice_cos[[k]]; h <- is_notice[[k]]; all(cs[h] >= 0.99) && all(cs[!h] < 0.5) && sum(h) >= 1 }, logical(1)))
add_check(sprintf("notice copies: in each of the six fits the topics at cosine >= %.2f to the anchor are all at >= 0.99 and every other topic is below 0.5; the K = 80 seed-1 copies are Topics 1, 2 and 80 as the delivered README states; K = 40 seed 1 has one copy", notice_threshold),
          sprintf("gap holds in all fits %s; K = 80 seed 1 copies: %s; K = 40 seed 1 copies: %s", notice_gap_ok, paste(which(is_notice[[fit_key(k_reference, 1L)]]), collapse = ", "), paste(which(is_notice[[fit_key(k_new, 1L)]]), collapse = ", ")),
          notice_gap_ok && identical(which(is_notice[[fit_key(k_reference, 1L)]]), c(1L, 2L, 80L)) && identical(which(is_notice[[fit_key(k_new, 1L)]]), 1L) &&
            nrow(notice_summary) == length(fits) && sum(notice_summary$notice_copies) == nrow(notice_topics))
add_check("representative-document proxies: one row per topic of each of the six fits; every representative row tokenised; the comparison summary has no empty item",
          sprintf("%d proxy rows; %d document rows; %d summary rows", nrow(representative_document_proxies), nrow(representative_document_rows), nrow(comparison_summary)),
          nrow(representative_document_proxies) == sum(selected_k) * length(fit_seeds) && nrow(representative_document_rows) == sum(selected_k) * length(fit_seeds) * n_representative &&
            !anyNA(representative_document_rows$top_10_words_present) && nrow(comparison_summary) > 0 && all(nzchar(comparison_summary$item)))
validation_checks <- bind_rows(checks)
print(as.data.frame(validation_checks[, c("check", "pass")]), right = FALSE)
mark_stage("validation")

# ---- Write tables ----
# INPUT : the objects above.
# DOES  : one CSV per table; stop before writing anything else if a check failed.
# OUTPUT: CSV files under tables/.
table_dims <- list()
write_table <- function(tbl, name) {
  path <- file.path(tab_dir, name)
  write_csv(tbl, path)
  table_dims[[path]] <<- c(rows = nrow(tbl), cols = ncol(tbl))
  invisible(path)
}
write_table(validation_checks, "validation_checks.csv")
stopifnot(all(validation_checks$pass))
new_tables <- retained_table_names(retained_new)
for (name in names(new_tables)) write_table(new_tables[[name]], name)
rm(new_tables); retained_new$document_topic_table <- NULL; retained_new$topic_word_table <- NULL; invisible(gc())
write_table(comparison_summary,                    "comparison_summary_K040_vs_K080.csv")
write_table(topic_correspondence_K040,             "topic_correspondence_K040_to_K080.csv")
write_table(topic_predecessors_K080,               "topic_predecessors_K080_from_K040.csv")
write_table(correspondence_structure_by_seed_pair, "correspondence_structure_by_seed_pair.csv")
write_table(cross_k_pairs,                         "cross_k_topic_pairs_K040_K080.csv")
write_table(cross_k_wide,                          "cross_k_topic_word_cosine_K040_K080.csv")
write_table(cross_k_all_seed_pairs,                "cross_k_topic_word_cosine_all_seed_pairs.csv")
write_table(correspondence_inspection,             "correspondence_inspection_K040_K080.csv")
write_table(correspondence_representative_documents, "correspondence_representative_documents_K040_K080.csv")
write_table(notice_topics,                         "notice_topics_K040_K080.csv")
write_table(notice_summary,                        "notice_topics_summary_K040_K080.csv")
write_table(representative_document_proxies,       "representative_document_proxies_K040_K080.csv")
write_table(top_words_probability,                 "top_words_by_probability_K040_K080.csv")
write_table(top_words_relevance,                   "top_words_by_relevance_K040_K080.csv")
write_table(stability_pairs,                       "stability_pairs_K040_K080.csv")
write_table(seed1_topic_stability,                 "topic_stability_seed1_K040_K080.csv")
fit_diagnostics <- delivered$model_comparison_by_fit |> filter(K %in% selected_k) |>
  left_join(delivered$fit_runtime |> filter(role == "primary") |> select(K, seed, pid, worker, job_index_in_worker, start_utc, end_utc, cpu_user_seconds_total,
                                                                          process_peak_working_set_mb, seconds_per_fit_iteration, seconds_fit_transform, from_cache), by = c("K", "seed")) |>
  arrange(K, seed)
write_table(fit_diagnostics,                       "fit_diagnostics_K040_K080.csv")
write_table(k80_reproduction,                      "k80_reproduction_check.csv")
write_table(input_integrity,                       "input_integrity.csv")
parameters_table <- tribble(
  ~parameter, ~value,
  "selected_k (user-selected candidate representations)", paste(selected_k, collapse = ", "),
  "k_new (retained outputs written here)", as.character(k_new),
  "k_reference (delivered retained outputs reproduced in scratch and hash-checked; not written)", as.character(k_reference),
  "primary seed", as.character(fit_seeds[1]),
  "correspondence measure", "cosine similarity of smoothed topic-word distributions (phi rows) between a K = 40 topic and a K = 80 topic",
  "correspondence_threshold", as.character(correspondence_threshold),
  "clear_threshold (= the delivered stability_cosine_threshold)", as.character(clear_threshold),
  "notice anchor", sprintf("K = %d seed %d Topic %d of the delivered run", notice_anchor$K, notice_anchor$seed, notice_anchor$topic),
  "notice_threshold (cosine to the anchor)", as.character(notice_threshold),
  "repetitive document rule (type-token ratio below)", as.character(repetitive_ttr),
  "delivered run script", main_script_path,
  "delivered run script SHA256 (as on disk now)", sha256(main_script_path),
  "delivered fitting run id", delivered$parameters$value[delivered$parameters$parameter == "fitting_run_id"],
  "model cache run key", readLines(file.path(model_dir, "run_key.txt"), warn = FALSE)[1],
  "alpha_sum / beta / n_iter_max / convergence_tol / heldout_share / split_seed (delivered, unchanged)", sprintf("%s / %s / %s / %s / %s / %s", alpha_sum, beta, n_iter_max, convergence_tol, heldout_share, split_seed),
  "min_token_chars / min_document_frequency / max_document_share / stopword_source (delivered, unchanged)", sprintf("%s / %s / %s / %s", min_token_chars, min_document_frequency, max_document_share, stopword_source),
  "n_top_words / coherence_top_n / relevance_lambda / n_representative / representative_min_tokens / active_topic_threshold (delivered, unchanged)", sprintf("%s / %s / %s / %s / %s / %s", n_top_words, coherence_top_n, relevance_lambda, n_representative, representative_min_tokens, active_topic_threshold),
  "models re-fitted here", "0")
write_table(parameters_table, "parameters.csv")
mark_stage("tables")

# ---- Preservation status, run record, provenance sidecars ----
# INPUT : the files written; the source files and model files read.
# DOES  : record for every output whether it is tracked by git (the K = 40
#         document-topic table is git-ignored like the delivered ones) and which
#         cached objects it depends on; the stage timings; one manifest per
#         output in the delivered format (input hashes including the cached
#         model files, this script's hash, git commit, parameters, packages).
# OUTPUT: tables/preservation_status.csv, stage_timings.csv, run_info.csv; *.manifest.yml.
git_ignored <- function(path) { out <- tryCatch(system2("git", c("-C", project_dir, "check-ignore", "-q", shQuote(path)), stdout = FALSE, stderr = FALSE), error = function(e) 1L); identical(out, 0L) }
depends_on <- function(name) {
  if (grepl("^document_topic_distribution_K040|^document_concentration|^topic_correlation_K040|^topic_prevalence_by|^topic_summary_K040|^topic_coactivation_K040", name)) "theta_K040_seed1.rds and fit_K040_seed1..3.rds (cache); source Parquet for identifiers"
  else if (grepl("^cross_k_topic_pairs|^correspondence_|^figc", name)) "theta_K040_seed1.rds, theta_K080_seed1.rds and the six fit_*.rds (cache); source Parquet for the documents"
  else if (grepl("^k80_reproduction|^input_integrity", name)) "the cache and the delivered tables (hash comparison)"
  else if (grepl("^comparison_summary|^representative_document_proxies|^notice_", name)) "the six fit_*.rds and fit_K010_seed1.rds (cache); the delivered tables"
  else if (grepl("^topic_word_distribution_K040|^topic_word_cosine_K040|^stability_|^cross_k_topic_word_cosine|^topic_correspondence|^topic_predecessors|^top_words|^topic_stability|^fig(4|5|9|11)_", name)) "fit_*.rds topic-word counts (cache); the committed topic_word_distribution tables carry the same phi at 6 significant digits"
  else if (grepl("^fit_diagnostics|^parameters|^validation|^preservation|^stage_timings|^run_info", name)) "delivered tables and this run's record"
  else "theta_K040_seed1.rds (cache)"
}
outputs_written <- c(list.files(tab_dir, pattern = "\\.csv$", full.names = TRUE), list.files(fig_dir, pattern = "\\.png$", full.names = TRUE))
preservation_status <- tibble(file = sub(paste0("^", out_dir, "/"), "", outputs_written), bytes = file.size(outputs_written),
                              git_ignored = vapply(outputs_written, git_ignored, logical(1))) |>
  mutate(git_status = if_else(git_ignored, "git-ignored: local only; its .manifest.yml (SHA256, rows, columns) is tracked", "tracked: committed-capable"),
         depends_on = vapply(basename(file), depends_on, ""),
         regenerated_by = "july2026_politics_lda_topics_tier2_k40_k80.R (cached fits; no re-fit)")
write_table(preservation_status, "preservation_status.csv")
run_end <- Sys.time()
stage_timings <- bind_rows(stage_log) |>
  mutate(start_utc = format(start_utc, "%Y-%m-%dT%H:%M:%OS3Z", tz = "UTC"), end_utc = format(end_utc, "%Y-%m-%dT%H:%M:%OS3Z", tz = "UTC"), seconds = round(seconds, 3))
write_table(stage_timings, "stage_timings.csv")
run_id <- format(run_start, "%Y%m%dT%H%M%SZ", tz = "UTC")
run_info <- bind_rows(
  tibble(item = "run_start_utc", value = format(run_start, "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")),
  tibble(item = "run_end_utc",   value = format(run_end,   "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")),
  tibble(item = "total_seconds", value = as.character(round(as.numeric(difftime(run_end, run_start, units = "secs")), 1))),
  tibble(item = "seconds_before_run_start (R start-up and package loading)", value = as.character(round(startup_elapsed_seconds, 1))),
  stage_timings |> transmute(item = paste0("stage_seconds_", stage), value = as.character(seconds)),
  tibble(item = "this_run_id", value = run_id),
  tibble(item = "models_fitted_in_this_run", value = "0"),
  tibble(item = "delivered_fitting_run_id", value = delivered$parameters$value[delivered$parameters$parameter == "fitting_run_id"]),
  tibble(item = "models_dir_bytes", value = as.character(sum(file.size(list.files(model_dir, pattern = "\\.rds$", full.names = TRUE))))),
  tibble(item = "tables_written", value = as.character(length(list.files(tab_dir, pattern = "\\.csv$")) + 1L)),
  tibble(item = "figures_written", value = as.character(length(list.files(fig_dir, pattern = "\\.png$")))),
  tibble(item = "R_version", value = R.version.string))
write_table(run_info, "run_info.csv")

git_commit <- tryCatch(system2("git", c("-C", project_dir, "rev-parse", "--short", "HEAD"), stdout = TRUE, stderr = FALSE), error = function(e) "")
manifest_packages <- c("nanoparquet", "dplyr", "tidyr", "stringi", "Matrix", "clue", "readxl", "ggplot2", "scales", "readr", "digest", "yaml")
package_versions  <- lapply(manifest_packages, function(p) list(name = p, version = as.character(packageVersion(p))))
input_files <- c(lapply(c(comment_files, submission_files), function(p) list(path = p, hash = sha256(p), format = "parquet")),
                 list(list(path = stopword_workbook, hash = workbook_hash, format = "xlsx", rows = nrow(tier2_raw), cols = ncol(tier2_raw), sheet = stopword_sheet)),
                 lapply(seq_len(nrow(input_integrity)), function(i) list(path = input_integrity$path[i], hash = input_integrity$sha256_now[i],
                                                                        format = tolower(tools::file_ext(input_integrity$path[i])),
                                                                        role = if (grepl("\\.rds$", input_integrity$path[i])) "cached fit of the delivered Tier 2 run, read-only" else "delivered Tier 2 table, read-only")))
manifest_parameters <- c(
  list(purpose = "K = 40 promoted to retained-output status (the delivered K = 80 arithmetic applied to the cached K = 40 fit) and the K = 40 versus K = 80 comparison package; no model re-fitted; delivered outputs unchanged",
       correspondence_measure = "cosine similarity of smoothed topic-word distributions (phi rows) between a K = 40 topic and a K = 80 topic; supporting: Jaccard of top-20 words, Pearson correlation of document shares, dominant-topic cross-tabulation",
       delivered_run = "july2026_politics_lda_topics_tier2.R; text processing, split, retained-model tables and figures copied unchanged"),
  setNames(as.list(parameters_table$value), parameters_table$parameter))
write_sidecar <- function(output) {
  ext <- tolower(tools::file_ext(output))
  manifest <- list(manifest_version = 1L, output_file = output, output_hash = sha256(output),
                   output_format = switch(ext, csv = "csv", png = "image/png", ext))
  if (ext == "csv") {
    dims <- table_dims[[output]]
    if (is.null(dims)) { tbl <- read_csv(output, show_col_types = FALSE, guess_max = 1000); dims <- c(rows = nrow(tbl), cols = ncol(tbl)) }
    manifest$output_rows <- unname(dims[["rows"]])
    manifest$output_cols <- unname(dims[["cols"]])
  }
  manifest$input_files    <- input_files
  manifest$transformation <- list(script = script_path, script_hash = sha256(script_path), parameters = manifest_parameters, git_commit = git_commit)
  manifest$software <- list(language = "R", language_version = paste(R.version$major, R.version$minor, sep = "."),
                            packages = package_versions, os = paste(Sys.info()[["sysname"]], Sys.info()[["release"]]))
  manifest$seed      <- fit_seeds[1]
  manifest$timestamp <- format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")
  manifest$notes     <- "Phase 1A LDA, Tier 2 (310-word) stopword condition: K = 40 retained-model outputs and the K = 40 versus K = 80 comparison, built from the cached fits of the delivered run (no re-fit); source Parquet, the workbook, the delivered tables and the cache read-only."
  write_yaml(manifest, paste0(output, ".manifest.yml"))
}
invisible(lapply(list.files(tab_dir, pattern = "\\.csv$", full.names = TRUE), write_sidecar))
invisible(lapply(list.files(fig_dir, pattern = "\\.png$", full.names = TRUE), write_sidecar))
mark_stage("manifests")

# ---- Console summary ----
print(as.data.frame(comparison_summary[, c("section", "item", "K040", "K080")]), right = FALSE)
print(as.data.frame(topic_correspondence_K040[, c("topic_K040", "best_topic_K080", "best_cosine", "n_K080_topics_at_or_above_0.5", "correspondence_class")]))
print(as.data.frame(topic_predecessors_K080 |> filter(best_cosine < correspondence_threshold) |> select(topic_K080, best_topic_K040, best_cosine, weaker_matched_cosine_across_seeds_K080, top_10_words_K080)))
print(as.data.frame(notice_summary[, c("K", "seed", "notice_copies", "notice_topics", "share_of_topic_slots", "combined_token_share_train", "docs_dominated_by_any_copy")]))
print(as.data.frame(k80_reproduction[, c("table", "regenerated_equals_manifest")]))
print(as.data.frame(run_info))
cat("\nDone. Figures ->", fig_dir, "\nTables ->", tab_dir, "\n")
