# July 2026 r/politics, Phase 1A LDA topic models with the Tier 2 cumulative stopword list (310 words).
# Same analysis as july2026_politics_lda_topics.R (SMART stopwords) except step 7, which removes the entries of
# stopword_curation_tier2_cumulative_310_words.xlsx. Workers also keep theta, run the diagnostics and record runtime.
# Inputs (read-only): source Parquet, the stopword workbook, r_analysis_outputs/lda_topics/ for comparison.
# Outputs: r_analysis_outputs/lda_topics_tier2/ (tables, figures, models/ cache). Topics are identified by number only.
# Env: KC_REPO_ROOT; LDA_QUICK=<folder> (smoke test) with LDA_QUICK_K/_ITER/_WORKERS; LDA_WORKERS; LDA_DETERMINISM_CHECK=0.

# ---- Setup: packages, paths, parameters, output directories ----
# Load packages, define paths and analytical parameters, and create output folders.
# LDA_QUICK=<folder> writes a reduced smoke test there instead of the project outputs; LDA_WORKERS sets the
# worker count (results do not depend on it); LDA_DETERMINISM_CHECK=0 skips the determinism job.
suppressPackageStartupMessages({
  library(nanoparquet); library(dplyr); library(tidyr); library(stringi); library(Matrix)
  library(text2vec); library(stopwords); library(clue); library(parallel); library(readxl)
  library(ggplot2); library(scales); library(readr); library(digest); library(yaml)
})
lgr::get_logger("text2vec")$set_threshold("warn")
startup_elapsed_seconds <- proc.time()[["elapsed"]]   # seconds since the R process started, i.e. package loading
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

project_dir      <- Sys.getenv("KC_REPO_ROOT", unset = getwd())
comments_dir     <- file.path(project_dir, "data_sample/comments/2026-07")
submissions_dir  <- file.path(project_dir, "data_sample/submissions/2026-07")
script_path      <- file.path(project_dir, "july2026_politics_lda_topics_tier2.R")
smart_run_dir    <- file.path(project_dir, "r_analysis_outputs/lda_topics")            # earlier SMART run (read-only)
stopword_workbook <- file.path(project_dir, "r_analysis_outputs/stopword_lists/curation/stopword_curation_tier2_cumulative_310_words.xlsx")
stopword_sheet   <- "tier2_cumulative_310"
quick_run        <- nzchar(Sys.getenv("LDA_QUICK"))
out_dir <- if (quick_run) file.path(Sys.getenv("LDA_QUICK"), "lda_topics_tier2_quick") else
  file.path(project_dir, "r_analysis_outputs/lda_topics_tier2")
fig_dir   <- file.path(out_dir, "figures")
tab_dir   <- file.path(out_dir, "tables")
model_dir <- file.path(out_dir, "models")
dir.create(fig_dir,   recursive = TRUE, showWarnings = FALSE)
dir.create(tab_dir,   recursive = TRUE, showWarnings = FALSE)
dir.create(model_dir, recursive = TRUE, showWarnings = FALSE)
# Figures and tables are fully regenerated on every run; remove those of a previous run so no stale file survives.
# models/ is a keyed cache of the fits (see the fitting section) and is cleared there when its key no longer matches.
invisible(file.remove(list.files(c(fig_dir, tab_dir), pattern = "\\.(csv|png|yml)$", full.names = TRUE)))

# Analytical parameters: identical to the SMART run except the stopword source.
candidate_k                <- c(10L, 20L, 30L, 40L, 50L, 60L, 80L, 100L)  # topic counts explored
fit_seeds                  <- c(1L, 2L, 3L)   # one fit per seed and K; seed 1 is the primary fit
alpha_sum                  <- 5               # document-topic prior alpha = alpha_sum / K (symmetric)
beta                       <- 0.01            # topic-word prior (symmetric)
n_iter_max                 <- 1000L           # ceiling on sampling iterations per fit
convergence_tol            <- 1e-4            # stop when 10 iterations improve the pseudo log-likelihood by less than this share
n_check_convergence        <- 10L
n_iter_inference           <- 10L             # post-burn-in samples averaged for document-topic proportions (text2vec default)
heldout_share              <- 0.05            # share of modeled documents held out for document-completion perplexity
split_seed                 <- 20260731L
min_token_chars            <- 2L
min_document_frequency     <- 5L              # a term must occur in at least this many non-marker documents
max_document_share         <- 0.5             # ... and in at most this share of non-marker documents
stopword_source            <- "tier2_cumulative_310"   # the 310 'word' entries of the Tier 2 workbook, used exactly as stored
n_top_words                <- 20L
coherence_top_n            <- 10L
relevance_lambda           <- 0.6             # Sievert & Shirley (2014) relevance weight for the second word ranking
n_representative           <- 5L
representative_min_tokens  <- 25L
active_topic_threshold     <- 0.1             # a topic is "active" in a document when its share is at least this
stability_cosine_threshold <- 0.8
# Implementation settings (results do not depend on them).
n_workers          <- as.integer(Sys.getenv("LDA_WORKERS", "12"))
determinism_check  <- !identical(Sys.getenv("LDA_DETERMINISM_CHECK"), "0")
if (quick_run) {
  candidate_k <- if (nzchar(Sys.getenv("LDA_QUICK_K"))) as.integer(strsplit(Sys.getenv("LDA_QUICK_K"), ",")[[1]]) else c(5L, 8L)
  fit_seeds   <- c(1L, 2L)
  n_iter_max  <- as.integer(Sys.getenv("LDA_QUICK_ITER", "30"))
  n_workers   <- as.integer(Sys.getenv("LDA_QUICK_WORKERS", "2"))
}
candidate_k <- sort(candidate_k)

comment_files    <- list.files(comments_dir,    pattern = "\\.parquet$", full.names = TRUE)
submission_files <- list.files(submissions_dir, pattern = "\\.parquet$", full.names = TRUE)
sha256 <- function(path) paste0("sha256:", digest(path, algo = "sha256", file = TRUE))
log_line("start; quick_run =", quick_run, "; workers =", n_workers, "; K =", paste(candidate_k, collapse = ","))

# ---- Stopword list: the Tier 2 workbook ----
# Read the 'word' column of the Tier 2 sheet as stored (no case folding or trimming), check its SHA256 against
# the workbook manifest, and flag entries the tokenizer cannot produce. SMART and Snowball are loaded for
# reference counts only.
workbook_hash <- sha256(stopword_workbook)
workbook_manifest_hash <- tryCatch(read_yaml(paste0(stopword_workbook, ".manifest.yml"))$output_hash, error = function(e) NA_character_)
tier2_raw <- read_excel(stopword_workbook, sheet = stopword_sheet, col_types = "text")
tier2_table <- tibble(rank = as.integer(tier2_raw$rank), word = tier2_raw$word,
                      number_of_lists = as.integer(tier2_raw$number_of_lists),
                      decision = tier2_raw$decision, notes = tier2_raw$notes)
stopword_list <- tier2_table$word
smart_list    <- stopwords("en", source = "smart")
snowball_list <- stopwords("en", source = "snowball")
token_regex   <- "\\p{L}+(?:'\\p{L}+)*"
tier2_table <- tier2_table |>
  mutate(n_chars = stri_length(word),
         in_smart = word %in% smart_list, in_snowball = word %in% snowball_list,
         tokenizer_status = case_when(
           !stri_detect_regex(word, paste0("^", token_regex, "$")) ~ "not producible: contains characters outside letters and internal apostrophes",
           stri_endswith_fixed(word, "'s") ~ "not producible: ends in 's, which step 5 strips from every token",
           n_chars < min_token_chars ~ sprintf("producible but removed at step 6 (shorter than %d characters), never at step 7", min_token_chars),
           TRUE ~ "producible; removed at step 7 when it occurs"))
log_line("workbook read:", length(stopword_list), "entries; hash", workbook_hash)

# ---- Read source data ----
# Read and stack the 7 comment files and 1 submission file; keep the columns used (no rows dropped).
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

# ---- Document table ----
# One document per source record. Comment text = body; submission text = title + " " + selftext.
# doc_key is the Reddit fullname (t1_/t3_ + id); submission_id is the thread. Nothing is dropped.
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

# ---- Text-field availability ----
# Classify each text field value as missing, empty, a deletion/removal marker, or other text, and count.
text_state <- function(x) case_when(
  is.na(x)                     ~ "missing (NA)",
  x == ""                      ~ "empty string",
  x == "[deleted]"             ~ "[deleted]",
  x == "[removed]"             ~ "[removed]",
  x == "[ Removed by Reddit ]" ~ "[ Removed by Reddit ]",
  TRUE                         ~ "other text")

text_availability <- bind_rows(
  tibble(field = "comment body",        state = text_state(comments$body)),
  tibble(field = "comment author",      state = text_state(comments$author)),
  tibble(field = "submission title",    state = text_state(submissions$title)),
  tibble(field = "submission selftext", state = text_state(submissions$selftext)),
  tibble(field = "submission author",   state = text_state(submissions$author))) |>
  count(field, state, name = "n") |>
  group_by(field) |>
  mutate(rows_in_field = sum(n), prop = n / rows_in_field) |>
  ungroup()
n_text_missing <- sum(is.na(comments$body)) + sum(is.na(submissions$title)) + sum(is.na(submissions$selftext))
mark_stage("document_table")

# ---- Text processing, step by step ----
# Same as the SMART run except step 7, counted at every step:
#   (1) lower case; (2) U+2019 -> '; (3) URLs removed; (4) tokens = runs of letters with internal apostrophes;
#   (5) trailing 's stripped; (6) tokens shorter than min_token_chars removed; (7) the 310 Tier 2 entries removed;
#   (8) documents that are only a removal marker ([deleted], [removed], [ Removed by Reddit ]) excluded.
# No stemming. SMART and Snowball counts are for reference only. Per-entry counts are taken after step 5.
removal_markers  <- c("[deleted]", "[removed]", "[ Removed by Reddit ]")
docs$marker_only <- stri_trim_both(docs$text) %in% removal_markers
url_regex   <- "(https?://|www\\.)\\S+"
url_opts    <- stri_opts_regex(case_insensitive = TRUE)

text_step <- stri_trans_tolower(docs$text)                                    # step 1
n_curly_apostrophes <- sum(stri_count_fixed(text_step, "\u2019"))
text_step <- stri_replace_all_fixed(text_step, "\u2019", "'")                 # step 2
url_per_doc <- stri_count_regex(text_step, url_regex, opts_regex = url_opts)
text_step   <- stri_replace_all_regex(text_step, url_regex, " ", opts_regex = url_opts)  # step 3
ws_tokens_per_doc    <- stri_count_regex(text_step, "\\S+")                   # reference only
digit_tokens_per_doc <- stri_count_regex(text_step, "\\S*\\d\\S*")            # reference only
tokens_list <- stri_extract_all_regex(text_step, token_regex, omit_no_match = TRUE)  # step 4
rm(text_step)
docs$n_tokens_letters <- lengths(tokens_list)
tok  <- unlist(tokens_list, use.names = FALSE)
tdoc <- rep(seq_len(n_docs), docs$n_tokens_letters)
rm(tokens_list)
n_letter_tokens <- length(tok)

tokens_cased <- stri_extract_all_regex(                                       # reference: case preserved (not applied)
  stri_replace_all_regex(stri_replace_all_fixed(docs$text, "\u2019", "'"), url_regex, " ", opts_regex = url_opts),
  token_regex, omit_no_match = TRUE)
n_distinct_tokens_cased  <- n_distinct(unlist(tokens_cased, use.names = FALSE))
rm(tokens_cased)
n_distinct_tokens_folded <- n_distinct(tok)

is_possessive <- stri_endswith_fixed(tok, "'s")                              # step 5
n_possessive  <- sum(is_possessive)
tok[is_possessive] <- stri_sub(tok[is_possessive], 1L, -3L)
n_distinct_after_possessive <- n_distinct(tok)
rm(is_possessive)

# Entry-level occurrences on the post-step-5 stream (all documents / non-marker documents).
in_marker_tok <- docs$marker_only[tdoc]
count_entries <- function(entries) {
  m <- match(tok, entries)
  hit <- !is.na(m)
  list(all = tabulate(m[hit], nbins = length(entries)), nonmarker = tabulate(m[hit & !in_marker_tok], nbins = length(entries)))
}
occ_tier2    <- count_entries(stopword_list)
occ_smart    <- count_entries(smart_list)
occ_snowball <- count_entries(snowball_list)

is_short <- stri_length(tok) < min_token_chars                                # step 6
n_tokens_after_step6_all       <- sum(!is_short)
n_tokens_after_step6_nonmarker <- sum(!is_short & !in_marker_tok)
is_stop  <- !is_short & tok %in% stopword_list                                # step 7 (Tier 2)
n_stop_tier2_nonmarker    <- sum(is_stop & !in_marker_tok)
is_stop_smart_reference   <- !is_short & tok %in% smart_list                  # reference only (the SMART run's condition)
n_stop_smart_reference    <- sum(is_stop_smart_reference)
n_stop_smart_reference_nonmarker <- sum(is_stop_smart_reference & !in_marker_tok)
n_stop_snowball_reference <- sum(!is_short & tok %in% snowball_list)
n_smart_only_tokens       <- sum(is_stop_smart_reference & !is_stop)          # tokens SMART removed that Tier 2 keeps
n_tier2_only_tokens       <- sum(is_stop & !is_stop_smart_reference)          # tokens Tier 2 removes that SMART kept (0 when Tier 2 is a subset of SMART)
rm(is_stop_smart_reference, in_marker_tok)
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

# ---- Vocabulary and document-term matrix ----
# Keep terms with min_document_frequency <= df <= max_document_share x non-marker documents. One dtm row per
# source document; documents with no kept term are labelled in model_status.
vocab_all  <- sort(unique(tok), method = "radix")
term_index <- match(tok, vocab_all)
tf_all <- sparseMatrix(i = tdoc, j = term_index, x = 1,
                       dims = c(n_docs, length(vocab_all)), repr = "C")
rm(tok, tdoc, term_index)
term_df_all      <- diff(tf_all@p)          # stored entries per column = documents containing the term
term_tf_all      <- colSums(tf_all)
n_nonmarker_docs <- sum(!docs$marker_only)
term_below_floor <- term_df_all < min_document_frequency
term_above_ceiling <- term_df_all > max_document_share * n_nonmarker_docs
term_kept        <- !term_below_floor & !term_above_ceiling
vocabulary       <- vocab_all[term_kept]
n_vocabulary     <- length(vocabulary)
dtm_all <- tf_all[, term_kept, drop = FALSE]
dimnames(dtm_all) <- list(docs$doc_key, vocabulary)
rm(tf_all)

vocabulary_table <- tibble(term = vocab_all, document_frequency = term_df_all, term_frequency = term_tf_all,
                           kept = term_kept,
                           reason = case_when(term_below_floor ~ "below document-frequency floor",
                                              term_above_ceiling ~ "above document-share ceiling",
                                              TRUE ~ "kept"),
                           in_smart_list = vocab_all %in% smart_list) |>
  arrange(desc(document_frequency), term)

docs$n_tokens_modeled <- as.integer(rowSums(dtm_all))
docs$n_tokens_rare    <- if_else(docs$marker_only, 0L, docs$n_tokens_candidate - docs$n_tokens_modeled)
docs$n_tokens_excluded_with_document <- if_else(docs$marker_only, docs$n_tokens_candidate, 0L)
docs <- docs |>
  mutate(model_status = case_when(
    marker_only               ~ "not modeled: removal marker only",
    n_tokens_letters == 0L    ~ "not modeled: no letter tokens",
    n_tokens_candidate == 0L  ~ "not modeled: only single-character tokens or stopwords",
    n_tokens_modeled == 0L    ~ "not modeled: only terms outside the vocabulary (document-frequency floor or ceiling)",
    TRUE                      ~ "modeled"))
status_levels <- c("modeled", "not modeled: removal marker only", "not modeled: no letter tokens",
                   "not modeled: only single-character tokens or stopwords",
                   "not modeled: only terms outside the vocabulary (document-frequency floor or ceiling)")
mark_stage("vocabulary_dtm")
log_line("dtm:", n_vocabulary, "terms;", sum(docs$model_status == "modeled"), "modeled documents")

# ---- Training / held-out split and document-completion halves ----
# Sample the held-out set (documents with >= 2 modeled tokens) with split_seed; fit on training documents only.
# For document-completion perplexity each held-out document's tokens are shuffled and alternated into half A
# (inference) and half B (scored). modeled_order = training rows, then held-out rows.
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

dtm_train        <- dtm_all[train_rows,   , drop = FALSE]
dtm_heldout      <- dtm_all[heldout_rows, , drop = FALSE]
train_doc_tokens <- as.numeric(rowSums(dtm_train))
n_train_tokens   <- sum(train_doc_tokens)
n_heldout_tokens <- sum(dtm_heldout)

heldout_t <- as(dtm_heldout, "TsparseMatrix")
occ_doc   <- rep(heldout_t@i + 1L, times = as.integer(heldout_t@x))
occ_term  <- rep(heldout_t@j + 1L, times = as.integer(heldout_t@x))
set.seed(split_seed + 1L)
occ_order <- order(occ_doc, runif(length(occ_doc)))
occ_doc   <- occ_doc[occ_order]
occ_term  <- occ_term[occ_order]
occ_rank  <- sequence(rle(occ_doc)$lengths)
half_a    <- occ_rank %% 2L == 1L
dtm_heldout_a <- sparseMatrix(i = occ_doc[half_a], j = occ_term[half_a], x = 1,
                              dims = dim(dtm_heldout), dimnames = dimnames(dtm_heldout), repr = "C")
dtm_heldout_b <- sparseMatrix(i = occ_doc[!half_a], j = occ_term[!half_a], x = 1,
                              dims = dim(dtm_heldout), dimnames = dimnames(dtm_heldout), repr = "C")
rm(heldout_t, occ_doc, occ_term, occ_order, occ_rank, half_a)

# Small per-document vectors the workers need for the representative-document rule (as in the SMART run:
# highest topic share among modeled documents with >= representative_min_tokens modeled tokens, one document
# per distinct stored text and per distinct author, ties by doc_key), in modeled_order.
modeled_text_id   <- match(docs$text[modeled_order],   unique(docs$text[modeled_order]))
modeled_author_id <- match(docs$author[modeled_order], unique(docs$author[modeled_order]))
modeled_eligible  <- docs$n_tokens_modeled[modeled_order] >= representative_min_tokens
modeled_doc_key   <- docs$doc_key[modeled_order]
term_frequency_share <- colSums(dtm_train) / n_train_tokens
n_terms_absent_from_train <- sum(term_frequency_share == 0)
mark_stage("split")
log_line("split:", n_train, "training,", n_heldout, "held-out documents")

# ---- Model fitting (one worker process per fit) ----
# Each K x seed is fitted in a worker: WarpLDA Gibbs sampling, doc_topic_prior = alpha_sum / K, topic_word_prior
# = beta, at most n_iter_max iterations with early stop at convergence_tol. Held-out perplexity is document
# completion (infer on half A, score half B). The worker also transforms all held-out documents, numbers topics
# by prevalence, saves theta and the counts to models/, and computes the per-fit diagnostics and runtime.
# Fits are cached in models/ under a key of all inputs and settings. A determinism job re-fits the smallest K,
# seed 1, and must match the primary fit.
smooth_theta <- function(theta, n_tokens, alpha) {
  (theta * n_tokens + alpha) / (n_tokens + ncol(theta) * alpha)
}

log_likelihood <- function(X, phi, theta, chunk = 200000L) {
  # sum over (document, term) cells of count x log( sum_k theta[d, k] phi[k, w] ), per-token perplexity
  X <- as(X, "TsparseMatrix")
  d <- X@i + 1L; w <- X@j + 1L; cnt <- X@x
  phi_t <- t(phi)
  ll <- 0
  for (s in seq(1L, length(d), by = chunk)) {
    e <- min(s + chunk - 1L, length(d))
    p <- rowSums(theta[d[s:e], , drop = FALSE] * phi_t[w[s:e], , drop = FALSE])
    ll <- ll + sum(cnt[s:e] * log(p))
  }
  c(loglik = ll, tokens = sum(cnt), perplexity = exp(-ll / sum(cnt)))
}

# Word rankings and topic diagnostics (same arithmetic as the SMART run, evaluated in the worker).
top_words_for <- function(phi, n, lambda) {
  score <- if (lambda == 1) log(phi) else {
    p_w <- ifelse(term_frequency_share > 0, term_frequency_share, NA_real_)
    lambda * log(phi) + (1 - lambda) * log(sweep(phi, 2, p_w, "/"))
  }
  score[is.na(score)] <- -Inf
  t(apply(score, 1, function(s) order(-s)[seq_len(n)]))
}

coherence_for <- function(phi, K, n = coherence_top_n) {
  # UMass (Mimno et al. 2011) and NPMI (Lau et al. 2014) per topic over the top n words by probability,
  # document co-occurrence within the training split; NPMI = -1 for pairs that never co-occur.
  idx   <- top_words_for(phi, n, 1)
  words <- sort(unique(as.vector(idx)))
  B <- dtm_train[, words, drop = FALSE]
  B@x[] <- 1
  co  <- unname(as.matrix(crossprod(B)))
  dfw <- unname(diag(co))
  pos <- matrix(match(as.vector(idx), words), nrow = nrow(idx))
  n_d <- nrow(dtm_train)
  t(vapply(seq_len(K), function(k) {
    p <- pos[k, ]
    umass <- 0; npmi <- 0; n_pairs <- 0
    for (m in 2:n) for (l in 1:(m - 1)) {
      d_ml  <- co[p[m], p[l]]
      umass <- umass + log((d_ml + 1) / dfw[p[l]])
      p_ml  <- d_ml / n_d
      npmi  <- npmi + if (d_ml == 0) -1 else if (p_ml >= 1) 1 else
        log(p_ml / ((dfw[p[m]] / n_d) * (dfw[p[l]] / n_d))) / (-log(p_ml))
      n_pairs <- n_pairs + 1
    }
    c(umass = unname(umass) / n_pairs, npmi = unname(npmi) / n_pairs)
  }, numeric(2)))
}

exclusivity_for <- function(phi, K, n = coherence_top_n) {
  idx     <- top_words_for(phi, n, 1)
  col_tot <- colSums(phi)
  vapply(seq_len(K), function(k) mean(phi[k, idx[k, ]] / col_tot[idx[k, ]]), numeric(1))
}

pairwise_metrics_for <- function(phi, K, token_share) {
  # cos_mat: cosine similarity between the smoothed topic-word distributions of two topics of the same fit.
  cos_mat   <- tcrossprod(phi / sqrt(rowSums(phi^2)))
  log_phi   <- log(phi)
  sym_kl    <- matrix(0, K, K)
  for (i in seq_len(K)) for (j in seq_len(K)) if (i < j)
    sym_kl[i, j] <- 0.5 * sum(phi[i, ] * (log_phi[i, ] - log_phi[j, ])) +
                    0.5 * sum(phi[j, ] * (log_phi[j, ] - log_phi[i, ]))
  cm1 <- svd(phi, nu = 0, nv = 0)$d; cm1 <- cm1 / sum(cm1)
  cm2 <- token_share / sum(token_share)
  list(cos_mat = cos_mat,
       metrics = c(cao_juan_2009 = mean(cos_mat[upper.tri(cos_mat)]),
                   deveaud_2014  = mean(sym_kl[upper.tri(sym_kl)]),
                   arun_2010     = sum(cm1 * log(cm1 / cm2)) + sum(cm2 * log(cm2 / cm1))))
}

jaccard_matrix <- function(idx) {
  # Jaccard overlap of the top-word sets (rows of idx) of every pair of topics of the same fit.
  K <- nrow(idx); n <- ncol(idx)
  out <- matrix(1, K, K)
  for (i in seq_len(K)) for (j in seq_len(K)) if (i < j) {
    inter <- length(intersect(idx[i, ], idx[j, ]))
    out[i, j] <- out[j, i] <- inter / (2 * n - inter)
  }
  out
}

worker_memory <- function() {
  g <- gc()
  info <- tryCatch(ps::ps_memory_info(ps::ps_handle()), error = function(e) NULL)
  c(r_max_used_mb = sum(g[, ncol(g)]),
    process_peak_working_set_mb = if (is.null(info)) NA_real_ else as.numeric(info[["peak_wset"]]) / 1024^2,
    process_working_set_mb      = if (is.null(info)) NA_real_ else as.numeric(info[["wset"]]) / 1024^2)
}

# One progress file per worker process (concurrent appends from several processes to one file interleave on Windows).
progress_log <- function() file.path(model_dir, sprintf("fit_progress_pid%d.log", Sys.getpid()))
log_progress <- function(...) cat(format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"), ..., "\n", file = progress_log(), append = TRUE)
job_prefix   <- function(role) if (role == "determinism") "determinism_" else ""
theta_path   <- function(role, K, seed) file.path(model_dir, sprintf("theta_%sK%03d_seed%d.rds", job_prefix(role), K, seed))
fit_path     <- function(role, K, seed) file.path(model_dir, sprintf("fit_%sK%03d_seed%d.rds",   job_prefix(role), K, seed))

run_job <- function(job) {
  K <- job$K; seed <- job$seed; role <- job$role
  t_mark <- list(); tick <- function(name) t_mark[[name]] <<- Sys.time()
  cpu_at <- list(); cpu_tick <- function(name) cpu_at[[name]] <<- proc.time()
  invisible(gc(reset = TRUE))
  tick("start"); cpu_tick("start")
  log_progress(sprintf("role=%s K=%d seed=%d pid=%d event=start", role, K, seed, Sys.getpid()))
  result <- tryCatch({
    alpha <- alpha_sum / K
    set.seed(seed)
    model <- LDA$new(n_topics = K, doc_topic_prior = alpha, topic_word_prior = beta,
                     n_iter_inference = n_iter_inference)
    theta <- model$fit_transform(dtm_train, n_iter = n_iter_max, convergence_tol = convergence_tol,
                                 n_check_convergence = n_check_convergence, progressbar = FALSE)
    tick("fit_transform"); cpu_tick("fit_transform")
    trace      <- attr(theta, "likelihood")
    attr(theta, "likelihood") <- NULL
    components <- model$components
    phi        <- (components + beta) / (rowSums(components) + ncol(components) * beta)
    train_fit  <- log_likelihood(dtm_train, phi, smooth_theta(theta, train_doc_tokens, alpha))
    tick("train_loglik")
    theta_a    <- model$transform(dtm_heldout_a, n_iter = n_iter_max, convergence_tol = convergence_tol,
                                  n_check_convergence = n_check_convergence, progressbar = FALSE)
    heldout_iterations <- max(attr(theta_a, "likelihood")$iter)
    heldout_fit <- log_likelihood(dtm_heldout_b, phi,
                                  smooth_theta(theta_a, as.numeric(rowSums(dtm_heldout_a)), alpha))
    rm(theta_a)
    tick("heldout_completion")
    theta_h    <- model$transform(dtm_heldout, n_iter = n_iter_max, convergence_tol = convergence_tol,
                                  n_check_convergence = n_check_convergence, progressbar = FALSE)
    heldout_full_iterations <- max(attr(theta_h, "likelihood")$iter)
    attr(theta_h, "likelihood") <- NULL
    rm(model)
    tick("heldout_transform"); cpu_tick("heldout_transform")
    # Topic numbering: descending token-weighted prevalence over the training documents (ties by sampler
    # index). Applied before any per-document label is derived, so ties in a document's shares are broken in
    # the same column order everywhere.
    token_share <- as.numeric(crossprod(theta, train_doc_tokens)) / sum(train_doc_tokens)
    ord         <- order(-token_share, seq_along(token_share))
    theta       <- theta[, ord, drop = FALSE]
    theta_h     <- theta_h[, ord, drop = FALSE]
    components  <- components[ord, , drop = FALSE]
    phi         <- phi[ord, , drop = FALSE]
    token_share <- token_share[ord]
    dominant    <- max.col(theta, ties.method = "first")
    max_theta   <- theta[cbind(seq_len(nrow(theta)), dominant)]
    doc_mean_share <- colMeans(theta)
    theta_m <- rbind(theta, theta_h)          # rows in modeled_order: training documents, then held-out documents
    rm(theta, theta_h)
    topic_cols <- sprintf("topic_%0*d", nchar(K), seq_len(K))
    dimnames(theta_m) <- list(modeled_doc_key, topic_cols)
    dominant_m  <- max.col(theta_m, ties.method = "first")
    max_share_m <- theta_m[cbind(seq_len(nrow(theta_m)), dominant_m)]
    n_active_m  <- as.integer(rowSums(theta_m >= active_topic_threshold))
    # text2vec normalises the summed sample counts by a reciprocal, so a share that is exactly the threshold in
    # exact arithmetic can be stored one unit in the last place below it; these cells are counted, not altered.
    theta_boundary <- c(cells_just_below_threshold = sum(theta_m >= active_topic_threshold - 1e-9 & theta_m < active_topic_threshold),
                        cells_at_threshold = sum(theta_m == active_topic_threshold), cells_total = length(theta_m))
    doc_cor     <- cor(theta_m)               # Pearson correlation of two topics' document shares over modeled documents
    active      <- theta_m >= active_topic_threshold; storage.mode(active) <- "double"
    co_active   <- crossprod(active) / nrow(theta_m)   # share of modeled documents in which both topics are active
    rm(active)
    cand <- which(modeled_eligible)
    rep_docs <- do.call(rbind, lapply(seq_len(K), function(k) {
      o <- cand[order(-theta_m[cand, k], modeled_doc_key[cand])]
      o <- o[!duplicated(modeled_text_id[o]) & !duplicated(modeled_author_id[o])]
      r <- head(o, n_representative)
      data.frame(topic = k, rank = seq_along(r), modeled_index = r, share = theta_m[r, k])
    }))
    tick("document_summaries")
    saveRDS(list(K = K, seed = seed, role = role, sampler_index = ord, doc_key = modeled_doc_key, split = modeled_split,
                 theta = theta_m), theta_path(role, K, seed), compress = FALSE)
    rm(theta_m)
    tick("save_theta")
    coh  <- coherence_for(phi, K)
    excl <- exclusivity_for(phi, K)
    pw   <- pairwise_metrics_for(phi, K, token_share)
    top_probability <- top_words_for(phi, n_top_words, 1)
    top_relevance   <- top_words_for(phi, n_top_words, relevance_lambda)
    top_jaccard     <- jaccard_matrix(top_probability)
    tick("diagnostics"); cpu_tick("end")
    mem <- worker_memory()
    secs <- function(a, b) as.numeric(difftime(t_mark[[b]], t_mark[[a]], units = "secs"))
    cpu_secs <- function(a, b) unname((cpu_at[[b]] - cpu_at[[a]])[c("user.self", "sys.self")])
    list(role = role, K = K, seed = seed, from_cache = FALSE,
         iterations          = max(trace$iter),
         trace               = trace,
         final_pseudo_loglik = trace$loglikelihood[nrow(trace)],
         sampler_index       = ord,
         components          = components,
         token_share         = token_share,
         doc_mean_share      = doc_mean_share,
         dominant            = dominant,
         max_theta           = max_theta,
         train_loglik        = train_fit[["loglik"]],
         train_tokens        = train_fit[["tokens"]],
         train_perplexity    = train_fit[["perplexity"]],
         heldout_loglik      = heldout_fit[["loglik"]],
         heldout_tokens      = heldout_fit[["tokens"]],
         heldout_perplexity  = heldout_fit[["perplexity"]],
         heldout_iterations  = heldout_iterations,
         heldout_full_iterations = heldout_full_iterations,
         dominant_m = dominant_m, max_share_m = max_share_m, n_active_m = n_active_m, theta_boundary = theta_boundary,
         doc_cor = doc_cor, co_active = co_active, rep_docs = rep_docs,
         topic_umass = coh[, "umass"], topic_npmi = coh[, "npmi"], topic_exclusivity = excl,
         pairwise = pw$metrics, word_cos = pw$cos_mat, top_probability = top_probability,
         top_relevance = top_relevance, top_jaccard = top_jaccard,
         theta_file = theta_path(role, K, seed),
         timing = c(elapsed_seconds = secs("start", "diagnostics"),
                    elapsed_seconds_fit_to_completion = secs("start", "heldout_completion"),
                    seconds_fit_transform = secs("start", "fit_transform"),
                    seconds_train_loglik = secs("fit_transform", "train_loglik"),
                    seconds_heldout_completion = secs("train_loglik", "heldout_completion"),
                    seconds_heldout_transform = secs("heldout_completion", "heldout_transform"),
                    seconds_document_summaries = secs("heldout_transform", "document_summaries"),
                    seconds_save_theta = secs("document_summaries", "save_theta"),
                    seconds_diagnostics = secs("save_theta", "diagnostics")),
         cpu = c(cpu_user_seconds_fit_transform = cpu_secs("start", "fit_transform")[1],
                 cpu_system_seconds_fit_transform = cpu_secs("start", "fit_transform")[2],
                 cpu_user_seconds_total = cpu_secs("start", "end")[1],
                 cpu_system_seconds_total = cpu_secs("start", "end")[2]),
         memory = mem,
         start_utc = format(t_mark$start, "%Y-%m-%dT%H:%M:%OS3Z", tz = "UTC"),
         end_utc   = format(t_mark$diagnostics, "%Y-%m-%dT%H:%M:%OS3Z", tz = "UTC"),
         pid = Sys.getpid(), node = unname(Sys.info()[["nodename"]]))
  }, error = function(e) list(role = role, K = K, seed = seed, error = conditionMessage(e)))
  if (is.null(result$error)) {
    saveRDS(result, fit_path(role, K, seed))
    log_progress(sprintf("role=%s K=%d seed=%d pid=%d event=end iterations=%d elapsed=%.1f peak_mb=%.0f",
                         role, K, seed, Sys.getpid(), result$iterations, result$timing[["elapsed_seconds"]],
                         result$memory[["process_peak_working_set_mb"]]))
  } else {
    log_progress(sprintf("role=%s K=%d seed=%d pid=%d event=error message=%s", role, K, seed, Sys.getpid(), result$error))
  }
  result
}

jobs <- expand.grid(seed = fit_seeds, K = candidate_k) |> arrange(desc(K), seed)
job_list <- lapply(seq_len(nrow(jobs)), function(i) list(K = jobs$K[i], seed = jobs$seed[i], role = "primary"))
if (determinism_check) job_list[[length(job_list) + 1L]] <- list(K = min(candidate_k), seed = fit_seeds[1], role = "determinism")

# Cache key: every matrix, every fitting and post-processing setting, the exported per-document vectors,
# the software versions and the worker code. A changed key clears models/.
modeled_split <- docs$split[modeled_order]
run_key <- digest(list(
  dtm_train, dtm_heldout, dtm_heldout_a, dtm_heldout_b, train_doc_tokens, candidate_k, fit_seeds, alpha_sum, beta,
  n_iter_max, convergence_tol, n_check_convergence, n_iter_inference, active_topic_threshold, n_representative,
  representative_min_tokens, coherence_top_n, n_top_words, relevance_lambda, modeled_text_id, modeled_author_id,
  modeled_eligible, modeled_doc_key, modeled_split, term_frequency_share,
  as.character(packageVersion("text2vec")), R.version.string,
  deparse(run_job), deparse(log_likelihood), deparse(smooth_theta), deparse(top_words_for), deparse(coherence_for),
  deparse(exclusivity_for), deparse(pairwise_metrics_for), deparse(jaccard_matrix)))
run_key_file <- file.path(model_dir, "run_key.txt")
stored_key   <- if (file.exists(run_key_file)) readLines(run_key_file, warn = FALSE)[1] else ""
if (!identical(stored_key, run_key)) {
  invisible(file.remove(list.files(model_dir, pattern = "\\.(rds|log|yml)$", full.names = TRUE)))
  writeLines(run_key, run_key_file)
}
job_cached <- vapply(job_list, function(j) file.exists(fit_path(j$role, j$K, j$seed)) && file.exists(theta_path(j$role, j$K, j$seed)), logical(1))
log_line("fits: run key", run_key, ";", sum(job_cached), "of", length(job_list), "jobs cached in models/")
mark_stage("fit_cache_key")

fit_results <- vector("list", length(job_list))
for (i in which(job_cached)) { fit_results[[i]] <- readRDS(fit_path(job_list[[i]]$role, job_list[[i]]$K, job_list[[i]]$seed)); fit_results[[i]]$from_cache <- TRUE }
jobs_to_run <- which(!job_cached)
n_workers_used <- min(n_workers, length(jobs_to_run))
fit_stage_start <- Sys.time()
if (length(jobs_to_run)) {
  cl <- makeCluster(n_workers_used)
  invisible(clusterEvalQ(cl, {
    suppressPackageStartupMessages({library(text2vec); library(Matrix)})
    lgr::get_logger("text2vec")$set_threshold("warn")
    NULL
  }))
  mark_stage("cluster_start")
  clusterExport(cl, c("dtm_train", "dtm_heldout", "dtm_heldout_a", "dtm_heldout_b", "train_doc_tokens",
                      "run_job", "smooth_theta", "log_likelihood", "top_words_for", "coherence_for", "exclusivity_for",
                      "pairwise_metrics_for", "jaccard_matrix", "worker_memory", "log_progress", "progress_log",
                      "theta_path", "fit_path", "job_prefix", "model_dir",
                      "alpha_sum", "beta", "n_iter_max", "convergence_tol", "n_check_convergence", "n_iter_inference",
                      "active_topic_threshold", "n_representative", "coherence_top_n", "n_top_words", "relevance_lambda",
                      "term_frequency_share", "modeled_text_id", "modeled_author_id", "modeled_eligible",
                      "modeled_doc_key", "modeled_split"))
  mark_stage("cluster_export")
  log_line("fits: dispatching", length(jobs_to_run), "jobs to", n_workers_used, "workers")
  fit_results[jobs_to_run] <- parLapplyLB(cl, job_list[jobs_to_run], run_job, chunk.size = 1L)
  stopCluster(cl)
  mark_stage("fits")
} else {
  mark_stage("fits")
}
fit_errors <- Filter(function(r) !is.null(r$error), fit_results)
if (length(fit_errors)) stop("worker errors: ", paste(vapply(fit_errors, function(r) sprintf("K=%d seed=%d (%s): %s", r$K, r$seed, r$role, r$error), ""), collapse = " | "))
log_line("fits: done in", round(as.numeric(difftime(Sys.time(), fit_stage_start, units = "secs")), 1), "s")
# Record of the run that fitted the cached models, so a cached re-run reports the fitting run's worker count.
run_id <- format(run_start, "%Y%m%dT%H%M%SZ", tz = "UTC")
fit_meta_file <- file.path(model_dir, "fit_run_meta.rds")
if (length(jobs_to_run)) {
  fit_meta <- list(fitting_run_id = run_id, n_workers_used = n_workers_used, jobs_fitted = length(jobs_to_run),
                   fit_stage_seconds = as.numeric(difftime(Sys.time(), fit_stage_start, units = "secs")))
  saveRDS(fit_meta, fit_meta_file)
} else {
  fit_meta <- if (file.exists(fit_meta_file)) readRDS(fit_meta_file) else list(fitting_run_id = NA_character_, n_workers_used = NA_integer_, jobs_fitted = 0L, fit_stage_seconds = NA_real_)
}

determinism_result <- NULL
if (determinism_check) {
  is_det <- vapply(fit_results, function(r) r$role == "determinism", logical(1))
  det     <- fit_results[[which(is_det)]]
  primary <- fit_results[[which(!is_det & vapply(fit_results, function(r) r$K == det$K && r$seed == det$seed, logical(1)))]]
  determinism_result <- list(
    K = det$K, seed = det$seed, pid_primary = primary$pid, pid_determinism = det$pid,
    components_identical = identical(unname(det$components), unname(primary$components)),
    dominant_identical   = identical(det$dominant_m, primary$dominant_m),
    theta_digest_primary = digest(readRDS(primary$theta_file)$theta), theta_digest_determinism = digest(readRDS(det$theta_file)$theta),
    theta_identical = identical(readRDS(det$theta_file)$theta, readRDS(primary$theta_file)$theta),
    same_process = det$pid == primary$pid, from_cache = det$from_cache || primary$from_cache)
  determinism_fit <- det
  fit_results <- fit_results[!is_det]
  job_list    <- job_list[!is_det]
}
main_memory_after_fits <- worker_memory()
mark_stage("fits_collect")

# ---- Per-fit derived quantities (assembled from the worker results) ----
# Topics arrive numbered by descending training token share (sampler index kept). Rebuild smoothed phi from the
# counts and assemble the word-ranking, prevalence and trace tables.
finalize_fit <- function(fit) {
  fit$phi <- (fit$components + beta) / (rowSums(fit$components) + ncol(fit$components) * beta)
  fit
}
fit_key <- function(K, seed) paste(K, seed, sep = "_")
fits <- lapply(fit_results, finalize_fit)
names(fits) <- vapply(fits, function(f) fit_key(f$K, f$seed), "")
rm(fit_results)

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
offdiag_stats <- function(m) { v <- m[upper.tri(m)]; c(max = max(v), min = min(v), mean = mean(v), mean_abs = mean(abs(v))) }
topic_prevalence_all_models <- bind_rows(lapply(fits, function(f) tibble(
  K = f$K, seed = f$seed, topic = seq_len(f$K), sampler_index = f$sampler_index,
  token_share = f$token_share, doc_mean_share = f$doc_mean_share,
  dominant_docs = tabulate(f$dominant, nbins = f$K), dominant_share = tabulate(f$dominant, nbins = f$K) / n_train,
  dominant_docs_all_modeled = tabulate(f$dominant_m, nbins = f$K),
  coherence_umass = f$topic_umass, coherence_npmi = f$topic_npmi, exclusivity = f$topic_exclusivity,
  word_cosine_max_other_topic = vapply(seq_len(f$K), function(k) max(f$word_cos[k, -k]), numeric(1)),
  doc_correlation_max_other_topic = vapply(seq_len(f$K), function(k) max(f$doc_cor[k, -k]), numeric(1)),
  co_activation_max_other_topic = vapply(seq_len(f$K), function(k) max(f$co_active[k, -k]), numeric(1)),
  top_10_words_by_probability = paste_top(f, f$top_probability),
  top_10_words_by_relevance = paste_top(f, f$top_relevance))))

fit_traces <- bind_rows(lapply(fits, function(f) tibble(K = f$K, seed = f$seed, iteration = f$trace$iter,
                                                        pseudo_loglikelihood = f$trace$loglikelihood,
                                                        pseudo_loglikelihood_per_token = f$trace$loglikelihood / n_train_tokens)))

# Topic x topic tables for every fit (upper triangle, topic_a < topic_b), each row naming its quantity.
pair_rows <- function(f, m) { ut <- which(upper.tri(m), arr.ind = TRUE); tibble(K = f$K, seed = f$seed, topic_a = ut[, 1], topic_b = ut[, 2], value = m[ut]) }
topic_pairs_word <- bind_rows(lapply(fits, function(f) {
  bind_rows(pair_rows(f, f$word_cos) |> mutate(quantity = "cosine similarity of the two topics' smoothed topic-word distributions (phi rows), same fit"),
            pair_rows(f, f$top_jaccard) |> mutate(quantity = sprintf("Jaccard overlap of the two topics' top-%d words by probability, same fit", n_top_words)))
})) |> select(K, seed, topic_a, topic_b, quantity, value)
topic_pairs_document <- bind_rows(lapply(fits, function(f) {
  bind_rows(pair_rows(f, f$doc_cor) |> mutate(quantity = "Pearson correlation of the two topics' document shares (theta columns) over all modeled documents, same fit"),
            pair_rows(f, f$co_active) |> mutate(quantity = sprintf("share of modeled documents in which both topics have a share of at least %g (co-activation), same fit", active_topic_threshold)))
})) |> select(K, seed, topic_a, topic_b, quantity, value)
mark_stage("diagnostics_assembly")
log_line("diagnostics assembled for", length(fits), "fits")

# ---- Stability across seeds ----
# For each K and seed pair, match topics one-to-one by the Hungarian assignment (clue::solve_LSAP) on topic-word
# cosine. Per pair: cosine, top-word Jaccard and best any-topic cosine; per seed pair: ARI of dominant topics.
# Per K: share of seed-1 topics matched at stability_cosine_threshold in both other seeds.
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
for (K in candidate_k) for (pair in seed_pairs) {
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
stability_summary <- stability_pairs |>
  group_by(K) |>
  summarise(seed_pairs                 = n_distinct(paste(seed_a, seed_b)),
            matched_cosine_mean        = mean(cosine),
            matched_cosine_median      = median(cosine),
            matched_cosine_min         = min(cosine),
            share_matched_cosine_at_least_threshold = mean(cosine >= stability_cosine_threshold),
            share_matched_not_row_maximum = mean(!matched_is_row_maximum),
            jaccard_top_words_mean     = mean(jaccard_top_words),
            ari_dominant_topics_mean   = mean(ari_dominant_topics[!duplicated(paste(seed_a, seed_b))]),
            .groups = "drop") |>
  left_join(seed1_topic_stability |> group_by(K) |>
              summarise(seed1_share_topics_reproduced_in_both_other_seeds = mean(reproduced_in_all_other_seeds),
                        seed1_share_topics_below_0.5_in_both_other_seeds = mean(reproduced_in_no_other_seed),
                        seed1_weaker_matched_cosine_mean = mean(weaker_matched_cosine), .groups = "drop"), by = "K")
mark_stage("stability")

# ---- Model comparison and retention rule ----
# Per-fit diagnostics and per-K means and SDs over seeds. Ranks across K for perplexity, NPMI, exclusivity and
# matched cosine; composite = mean rank. Retained for full tables (for inspection only): lowest composite rank,
# highest NPMI, and the one-standard-error perplexity choice.
model_comparison_by_fit <- bind_rows(lapply(fits, function(f) {
  wc <- offdiag_stats(f$word_cos); jc <- offdiag_stats(f$top_jaccard); dc <- offdiag_stats(f$doc_cor); ca <- offdiag_stats(f$co_active)
  tibble(
    K = f$K, seed = f$seed, iterations_fit = f$iterations, iterations_heldout_inference = f$heldout_iterations,
    iterations_heldout_full_transform = f$heldout_full_iterations,
    final_pseudo_loglikelihood = f$final_pseudo_loglik,
    final_pseudo_loglikelihood_per_token = f$final_pseudo_loglik / n_train_tokens,
    train_perplexity = f$train_perplexity, train_tokens = f$train_tokens,
    heldout_perplexity = f$heldout_perplexity, heldout_tokens_scored = f$heldout_tokens,
    coherence_npmi = mean(f$topic_npmi), coherence_npmi_median = median(f$topic_npmi), coherence_npmi_min = min(f$topic_npmi),
    coherence_umass = mean(f$topic_umass), exclusivity = mean(f$topic_exclusivity),
    cao_juan_2009 = f$pairwise[["cao_juan_2009"]], deveaud_2014 = f$pairwise[["deveaud_2014"]], arun_2010 = f$pairwise[["arun_2010"]],
    word_cosine_pair_max = wc[["max"]], top_words_jaccard_pair_mean = jc[["mean"]], top_words_jaccard_pair_max = jc[["max"]],
    doc_correlation_pair_max = dc[["max"]], doc_correlation_pair_min = dc[["min"]], doc_correlation_pair_mean_abs = dc[["mean_abs"]],
    co_activation_pair_max = ca[["max"]], co_activation_pair_mean = ca[["mean"]],
    theta_cells_just_below_active_threshold = f$theta_boundary[["cells_just_below_threshold"]],
    theta_cells_at_active_threshold = f$theta_boundary[["cells_at_threshold"]],
    largest_topic_share = max(f$token_share), smallest_topic_share = min(f$token_share),
    elapsed_seconds = f$timing[["elapsed_seconds_fit_to_completion"]], elapsed_seconds_whole_job = f$timing[["elapsed_seconds"]])
})) |> arrange(K, seed)

model_comparison_summary <- model_comparison_by_fit |>
  group_by(K) |>
  summarise(fits = n(),
            across(c(iterations_fit, train_perplexity, heldout_perplexity, coherence_npmi, coherence_umass,
                     exclusivity, cao_juan_2009, deveaud_2014, arun_2010, word_cosine_pair_max, top_words_jaccard_pair_mean,
                     doc_correlation_pair_max, doc_correlation_pair_mean_abs, co_activation_pair_max,
                     largest_topic_share, elapsed_seconds),
                   list(mean = mean, sd = sd)),
            .groups = "drop") |>
  left_join(stability_summary, by = "K") |>
  mutate(rank_heldout_perplexity = rank(heldout_perplexity_mean),
         rank_coherence_npmi     = rank(-coherence_npmi_mean),
         rank_exclusivity        = rank(-exclusivity_mean),
         rank_stability          = rank(-matched_cosine_mean),
         composite_rank          = (rank_heldout_perplexity + rank_coherence_npmi + rank_exclusivity + rank_stability) / 4) |>
  arrange(K)

k_composite  <- model_comparison_summary$K[which.min(model_comparison_summary$composite_rank)]
k_coherence  <- model_comparison_summary$K[which.max(model_comparison_summary$coherence_npmi_mean)]
perplexity_best_row  <- which.min(model_comparison_summary$heldout_perplexity_mean)
k_perplexity_minimum <- model_comparison_summary$K[perplexity_best_row]
perplexity_threshold <- model_comparison_summary$heldout_perplexity_mean[perplexity_best_row] +
  coalesce(model_comparison_summary$heldout_perplexity_sd[perplexity_best_row], 0)
k_perplexity <- min(model_comparison_summary$K[model_comparison_summary$heldout_perplexity_mean <= perplexity_threshold])
retained_k   <- unique(c(k_composite, k_coherence, k_perplexity))
retention_basis_for <- function(k) paste(c(if (k == k_composite) "lowest composite rank",
                                           if (k == k_coherence) "highest NPMI coherence",
                                           if (k == k_perplexity) "lowest held-out perplexity (one-standard-error rule)"), collapse = "; ")
model_comparison_summary <- model_comparison_summary |>
  mutate(heldout_perplexity_within_one_sd_of_best = heldout_perplexity_mean <= perplexity_threshold,
         retained = K %in% retained_k,
         retention_basis = vapply(K, retention_basis_for, ""))
mark_stage("comparison")
log_line("comparison: retained K =", paste(retained_k, collapse = ", "))

# ---- Representative documents for every fit ----
# Per topic, the n_representative documents with the highest share (>= representative_min_tokens modeled tokens,
# distinct texts and authors), selected in the worker and joined here to identifiers and text.
representative_documents_all_models <- bind_rows(lapply(fits, function(f) {
  r <- modeled_order[f$rep_docs$modeled_index]
  tibble(K = f$K, seed = f$seed, topic = f$rep_docs$topic, topic_label = paste("Topic", f$rep_docs$topic), rank = f$rep_docs$rank,
         topic_share_in_document = f$rep_docs$share,
         doc_key = docs$doc_key[r], doc_id = docs$doc_id[r], doc_type = docs$doc_type[r], author = docs$author[r],
         created_utc = docs$created_utc[r], submission_id = docs$submission_id[r], link_id = docs$link_id[r],
         parent_id = docs$parent_id[r], split = docs$split[r], n_tokens_modeled = docs$n_tokens_modeled[r],
         dominant_topic = f$dominant_m[f$rep_docs$modeled_index], dominant_topic_share = f$max_share_m[f$rep_docs$modeled_index],
         text = docs$text[r])
}))
mark_stage("representative_documents")

# ---- Retained models: full document-topic distributions and inspection tables ----
# For each retained K, load the seed-1 theta from models/ (no re-fit) and build over all modeled documents: shares,
# shares by type and day, correlations, co-activation, word cosine, concentration, representative documents,
# and the full document-topic table (NA for documents not modeled).
band_breaks <- c(0, 1, 4, 9, 24, 49, 99, Inf)
band_labels <- c("1", "2-4", "5-9", "10-24", "25-49", "50-99", "100+")
retained <- list()
for (K in retained_k) {
  tag   <- sprintf("K%03d", K)
  fit   <- fits[[fit_key(K, fit_seeds[1])]]
  t_ret <- Sys.time()
  th    <- readRDS(fit$theta_file)
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

  representative_documents <- representative_documents_all_models |> filter(K == !!K, seed == fit_seeds[1]) |> select(-K, -seed)

  theta_out <- round(theta_full, 6)
  colnames(theta_out) <- topic_cols
  document_topic_table <- docs |>
    transmute(doc_key, doc_id, doc_type, author, created_utc, created_date_utc, submission_id, link_id, parent_id,
              submission_in_sample, n_chars, n_tokens_letters, n_tokens_modeled, model_status, split,
              dominant_topic = dominant_full, dominant_topic_share = max_share_full, n_active_topics = n_active_full) |>
    bind_cols(as_tibble(theta_out)) |>
    arrange(created_utc, doc_key)
  rm(theta_out)

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

  retained[[tag]] <- list(K = K, tag = tag, fit = fit, theta_full = theta_full, dominant_full = dominant_full,
                          max_share_full = max_share_full, theta_file_consistent = theta_file_consistent,
                          worker_summaries_identical = worker_summaries_identical, active_share = active_share,
                          topic_summary = topic_summary, topic_prevalence_by_type = topic_prevalence_by_type,
                          topic_prevalence_by_day = topic_prevalence_by_day, topic_correlation = topic_correlation,
                          co_activation = co_activation, topic_word_cosine = topic_word_cosine,
                          concentration = concentration, concentration_summary = concentration_summary,
                          representative_documents = representative_documents,
                          document_topic_table = document_topic_table, topic_word_table = topic_word_table,
                          stability_matrices = stability_matrices, topic_cols = topic_cols, topic_labels = topic_labels,
                          seconds = as.numeric(difftime(Sys.time(), t_ret, units = "secs")))
  rm(theta_m, theta_full, document_topic_table, topic_word_table)
  invisible(gc())
  mark_stage(paste0("retained_", tag))
  log_line("retained", tag, "assembled in", round(retained[[tag]]$seconds, 1), "s")
}

# ---- Accounting and processing-report tables ----
# Document accounting from source to modeled corpus, the text-processing report, stopword accounting and list
# comparison with SMART, document length bands and the parameter table.
count_by_type <- function(mask) c(sum(mask & docs$doc_type == "comment"), sum(mask & docs$doc_type == "submission"), sum(mask))
accounting_row <- function(stage, mask) tibble(stage = stage, comments = count_by_type(mask)[1],
                                               submissions = count_by_type(mask)[2], total = count_by_type(mask)[3])
document_accounting <- bind_rows(
  accounting_row("documents in source", rep(TRUE, n_docs)),
  accounting_row("not modeled: removal marker only", docs$model_status == status_levels[2]),
  accounting_row("not modeled: no letter tokens", docs$model_status == status_levels[3]),
  accounting_row("not modeled: only single-character tokens or stopwords", docs$model_status == status_levels[4]),
  accounting_row("not modeled: only terms outside the vocabulary", docs$model_status == status_levels[5]),
  accounting_row("modeled documents", docs$model_status == "modeled"),
  accounting_row("modeled: training set", !is.na(docs$split) & docs$split == "train"),
  accounting_row("modeled: held-out set", !is.na(docs$split) & docs$split == "held-out"),
  tibble(stage = "letter-run tokens (all documents)", comments = sum(docs$n_tokens_letters[docs$doc_type == "comment"]),
         submissions = sum(docs$n_tokens_letters[docs$doc_type == "submission"]), total = n_letter_tokens),
  tibble(stage = "modeled tokens", comments = sum(docs$n_tokens_modeled[docs$doc_type == "comment"]),
         submissions = sum(docs$n_tokens_modeled[docs$doc_type == "submission"]), total = sum(docs$n_tokens_modeled)))

n_tokens_short_total <- sum(docs$n_tokens_short)
n_tokens_stop_total  <- sum(docs$n_tokens_stop)
n_tokens_rare_total  <- sum(docs$n_tokens_rare)
n_tokens_modeled_total <- sum(docs$n_tokens_modeled)
text_processing_report <- tribble(
  ~item, ~value,
  "documents in source (comments + submissions)",                                     n_docs,
  "documents whose whole text is a removal marker (excluded, step 8)",                sum(docs$marker_only),
  "step 1: case folding — distinct letter-run tokens if case were preserved (not applied)", n_distinct_tokens_cased,
  "step 1: case folding — distinct letter-run tokens after case folding",             n_distinct_tokens_folded,
  "step 2: curly apostrophes (U+2019) normalised to ' (occurrences)",                 n_curly_apostrophes,
  "step 3: URL matches removed",                                                      sum(url_per_doc),
  "step 3: documents containing at least one URL",                                    sum(url_per_doc > 0),
  "reference: whitespace-delimited tokens after URL removal (not a model unit)",      sum(ws_tokens_per_doc),
  "reference: whitespace-delimited tokens containing a digit (letter runs inside them are still extracted)", sum(digit_tokens_per_doc),
  "step 4: letter-run tokens extracted",                                              n_letter_tokens,
  "step 4: documents with zero letter-run tokens",                                    sum(docs$n_tokens_letters == 0),
  "step 5: tokens with a trailing possessive 's stripped",                            n_possessive,
  "step 5: distinct tokens after possessive stripping",                               n_distinct_after_possessive,
  sprintf("step 6: tokens shorter than %d characters removed", min_token_chars),      n_tokens_short_total,
  "tokens entering step 7 (after step 6, all documents)",                             n_tokens_after_step6_all,
  sprintf("step 7: stopword tokens removed (Tier 2 cumulative list, %d entries)", length(stopword_list)), n_tokens_stop_total,
  sprintf("step 7: stopword tokens the SMART list (%d terms, the original condition) would remove instead (not applied)", length(smart_list)), n_stop_smart_reference,
  sprintf("step 7: stopword tokens the Snowball list (%d terms) would remove instead (not applied)", length(snowball_list)), n_stop_snowball_reference,
  "candidate tokens after steps 1-7 (all documents)",                                 n_candidate_tokens,
  "step 8: candidate tokens inside removal-marker-only documents (excluded with the document)", n_tokens_in_marker_docs,
  "candidate tokens in non-marker documents",                                         n_candidate_tokens - n_tokens_in_marker_docs,
  "distinct candidate terms in non-marker documents (vocabulary before pruning)",     length(vocab_all),
  sprintf("step 9: terms in fewer than %d non-marker documents removed", min_document_frequency), sum(term_below_floor),
  "step 9: tokens removed with those terms",                                          n_tokens_rare_total,
  sprintf("step 9: terms in more than %.0f%% of non-marker documents removed", 100 * max_document_share), sum(term_above_ceiling),
  "step 9: highest document share of any term after stopword removal",               max(term_df_all) / n_nonmarker_docs,
  "vocabulary size (terms modeled)",                                                  n_vocabulary,
  "tokens modeled",                                                                   n_tokens_modeled_total,
  "documents modeled (at least one modeled token)",                                   n_modeled,
  "documents not modeled: no letter tokens",                                          sum(docs$model_status == status_levels[3]),
  "documents not modeled: only single-character tokens or stopwords",                sum(docs$model_status == status_levels[4]),
  "documents not modeled: only terms outside the vocabulary",                         sum(docs$model_status == status_levels[5]),
  "vocabulary terms absent from the training split",                                  n_terms_absent_from_train)

stopwords_used <- tier2_table |>
  mutate(occurrences_all_documents = occ_tier2$all, occurrences_nonmarker_documents = occ_tier2$nonmarker,
         removed_at = case_when(stri_startswith_fixed(tokenizer_status, "not producible") ~ "never (no token can equal the entry)",
                                n_chars < min_token_chars ~ "step 6 (length rule), not step 7",
                                occurrences_all_documents == 0 ~ "step 7 would apply, but the entry never occurs",
                                TRUE ~ "step 7 (stopword rule)"),
         source = stopword_source) |>
  select(rank, word, number_of_lists, source, n_chars, in_smart, in_snowball, tokenizer_status, removed_at,
         occurrences_all_documents, occurrences_nonmarker_documents, decision, notes)

union_words <- sort(unique(c(stopword_list, smart_list)), method = "radix")
occ_union <- {
  m_t <- match(union_words, stopword_list); m_s <- match(union_words, smart_list)
  list(all = ifelse(!is.na(m_t), occ_tier2$all[m_t], occ_smart$all[m_s]),
       nonmarker = ifelse(!is.na(m_t), occ_tier2$nonmarker[m_t], occ_smart$nonmarker[m_s]))
}
stopword_list_comparison <- tibble(word = union_words,
                                   in_tier2 = union_words %in% stopword_list, in_smart = union_words %in% smart_list,
                                   in_snowball = union_words %in% snowball_list,
                                   tier2_rank = tier2_table$rank[match(union_words, tier2_table$word)],
                                   tier2_number_of_lists = tier2_table$number_of_lists[match(union_words, tier2_table$word)],
                                   status = case_when(in_tier2 & in_smart ~ "removed by both lists",
                                                      in_tier2 ~ "removed by Tier 2 only (kept by SMART)",
                                                      TRUE ~ "restored: removed by SMART, kept by Tier 2"),
                                   occurrences_all_documents = occ_union$all, occurrences_nonmarker_documents = occ_union$nonmarker,
                                   n_chars = stri_length(union_words),
                                   in_modeled_vocabulary = union_words %in% vocabulary,
                                   document_frequency_nonmarker = vocabulary_table$document_frequency[match(union_words, vocabulary_table$term)]) |>
  arrange(status, desc(occurrences_all_documents), word)

restored <- stopword_list_comparison |> filter(status == "restored: removed by SMART, kept by Tier 2")
tier2_only <- stopword_list_comparison |> filter(status == "removed by Tier 2 only (kept by SMART)")
stopword_accounting <- tribble(
  ~item, ~value,
  "stopword workbook (read-only)",                                                    stopword_workbook,
  "workbook SHA256 (measured at run time)",                                           workbook_hash,
  "workbook SHA256 recorded in its provenance manifest",                              coalesce(workbook_manifest_hash, "manifest not found"),
  "workbook hash equals the manifest hash",                                           as.character(identical(workbook_hash, workbook_manifest_hash)),
  "sheet read",                                                                       stopword_sheet,
  "entries in the 'word' column (used exactly as stored)",                            as.character(length(stopword_list)),
  "distinct entries",                                                                 as.character(n_distinct(stopword_list)),
  "entries with a non-blank 'decision' cell",                                         as.character(sum(!is.na(tier2_table$decision))),
  "entries with a non-blank 'notes' cell",                                            as.character(sum(!is.na(tier2_table$notes))),
  "number_of_lists range over the entries",                                           paste(range(tier2_table$number_of_lists), collapse = "-"),
  "entries that are also in the SMART list",                                          as.character(sum(tier2_table$in_smart)),
  "entries producible by the tokenizer (letters with internal apostrophes, not ending in 's)", as.character(sum(!stri_startswith_fixed(tier2_table$tokenizer_status, "not producible"))),
  "entries not producible: end in 's (step 5 strips it)",                            as.character(sum(stri_endswith_fixed(stopword_list, "'s"))),
  "entries not producible: other characters",                                         as.character(sum(!stri_detect_regex(stopword_list, paste0("^", token_regex, "$")))),
  sprintf("entries shorter than %d characters (removed at step 6, not step 7)", min_token_chars), as.character(sum(tier2_table$n_chars < min_token_chars)),
  "entries that match at least one token of the post-step-5 stream (all documents)", as.character(sum(occ_tier2$all > 0)),
  "entries that match at least one token and are removed at step 7",                 as.character(sum(occ_tier2$all > 0 & tier2_table$n_chars >= min_token_chars)),
  "entries that match at least one token but are removed at step 6 instead",         as.character(sum(occ_tier2$all > 0 & tier2_table$n_chars < min_token_chars)),
  "entries that match no token",                                                      as.character(sum(occ_tier2$all == 0)),
  "token occurrences before stopword removal (after step 6, all documents)",         as.character(n_tokens_after_step6_all),
  "token occurrences before stopword removal (after step 6, non-marker documents)",  as.character(n_tokens_after_step6_nonmarker),
  "occurrences removed by Tier 2 (step 7, all documents)",                           as.character(n_tokens_stop_total),
  "share of occurrences removed by Tier 2 (all documents)",                          sprintf("%.4f%%", 100 * n_tokens_stop_total / n_tokens_after_step6_all),
  "occurrences removed by Tier 2 (non-marker documents)",                            as.character(n_stop_tier2_nonmarker),
  "share of occurrences removed by Tier 2 (non-marker documents)",                   sprintf("%.4f%%", 100 * n_stop_tier2_nonmarker / n_tokens_after_step6_nonmarker),
  "occurrences remaining after step 7 (all documents)",                               as.character(n_candidate_tokens),
  "occurrences remaining after step 7 (non-marker documents)",                       as.character(n_candidate_tokens - n_tokens_in_marker_docs),
  "reference: occurrences SMART would remove (all documents; not applied)",          as.character(n_stop_smart_reference),
  "reference: share SMART would remove (all documents; not applied)",                sprintf("%.4f%%", 100 * n_stop_smart_reference / n_tokens_after_step6_all),
  "words removed by Tier 2 but not by SMART (count)",                                as.character(nrow(tier2_only)),
  "words removed by Tier 2 but not by SMART (list)",                                 if (nrow(tier2_only)) paste(tier2_only$word, collapse = " ") else "(none)",
  "occurrences removed by Tier 2 that SMART would have kept (all documents)",        as.character(n_tier2_only_tokens),
  "words restored: removed by SMART, kept by Tier 2 (count)",                        as.character(nrow(restored)),
  "words restored: removed by SMART, kept by Tier 2 (list, most frequent first)",    paste(restored$word, collapse = " "),
  "occurrences restored: removed by SMART, kept by Tier 2 (all documents)",          as.character(n_smart_only_tokens),
  "restored words that enter the modeled vocabulary",                                as.character(sum(restored$in_modeled_vocabulary)),
  "restored words with zero occurrences",                                            as.character(sum(restored$occurrences_all_documents == 0)),
  "modelable documents (at least one modeled token)",                                as.character(n_modeled),
  "vocabulary size (terms modeled)",                                                 as.character(n_vocabulary),
  "tokens modeled",                                                                  as.character(n_tokens_modeled_total))

document_length_bands <- docs |>
  filter(model_status == "modeled") |>
  mutate(token_band = cut(n_tokens_modeled, breaks = band_breaks, labels = band_labels, right = TRUE)) |>
  count(doc_type, token_band, name = "docs") |>
  group_by(doc_type) |> mutate(prop = docs / sum(docs)) |> ungroup()
document_length_summary <- docs |>
  filter(model_status == "modeled") |>
  group_by(doc_type) |>
  summarise(docs = n(), tokens = sum(n_tokens_modeled), min = min(n_tokens_modeled),
            p05 = quantile(n_tokens_modeled, 0.05, type = 7, names = FALSE), p25 = quantile(n_tokens_modeled, 0.25, type = 7, names = FALSE),
            median = median(n_tokens_modeled), mean = mean(n_tokens_modeled),
            p75 = quantile(n_tokens_modeled, 0.75, type = 7, names = FALSE), p95 = quantile(n_tokens_modeled, 0.95, type = 7, names = FALSE),
            max = max(n_tokens_modeled), .groups = "drop")

parameters_table <- tribble(
  ~parameter, ~value,
  "candidate_k", paste(candidate_k, collapse = ", "),
  "fit_seeds", paste(fit_seeds, collapse = ", "),
  "alpha_sum (doc_topic_prior = alpha_sum / K)", as.character(alpha_sum),
  "beta (topic_word_prior)", as.character(beta),
  "n_iter_max", as.character(n_iter_max),
  "convergence_tol", as.character(convergence_tol),
  "n_check_convergence", as.character(n_check_convergence),
  "n_iter_inference", as.character(n_iter_inference),
  "heldout_share", as.character(heldout_share),
  "split_seed", as.character(split_seed),
  "min_token_chars", as.character(min_token_chars),
  "min_document_frequency", as.character(min_document_frequency),
  "max_document_share", as.character(max_document_share),
  "stopword_source", stopword_source,
  "stopword_workbook", stopword_workbook,
  "stopword_workbook_sha256", workbook_hash,
  "stopword_entries", as.character(length(stopword_list)),
  "n_top_words", as.character(n_top_words),
  "coherence_top_n", as.character(coherence_top_n),
  "relevance_lambda", as.character(relevance_lambda),
  "n_representative", as.character(n_representative),
  "representative_min_tokens", as.character(representative_min_tokens),
  "active_topic_threshold", as.character(active_topic_threshold),
  "stability_cosine_threshold", as.character(stability_cosine_threshold),
  "n_workers (requested in this run)", as.character(n_workers),
  "n_workers (used in the run that fitted the models)", as.character(fit_meta$n_workers_used),
  "fitting_run_id", as.character(fit_meta$fitting_run_id),
  "determinism_check", as.character(determinism_check),
  "k_composite (lowest composite rank)", as.character(k_composite),
  "k_coherence (highest NPMI coherence)", as.character(k_coherence),
  "k_perplexity_minimum (lowest mean held-out perplexity)", as.character(k_perplexity_minimum),
  "k_perplexity (one-standard-error rule)", as.character(k_perplexity),
  "retained_k (models with full document-level tables; not a final choice of K)", paste(retained_k, collapse = ", "))

# ---- Runtime record ----
# One row per fit with phase timings, CPU, memory, worker pid, iterations and concurrent fits; plus stage timings.
runtime_row <- function(f) tibble(K = f$K, seed = f$seed, role = f$role, from_cache = f$from_cache, pid = f$pid, node = f$node,
                                  start_utc = f$start_utc, end_utc = f$end_utc,
                                  iterations_fit = f$iterations, iterations_heldout_inference = f$heldout_iterations,
                                  iterations_heldout_full_transform = f$heldout_full_iterations,
                                  tibble::as_tibble_row(f$timing), tibble::as_tibble_row(f$cpu), tibble::as_tibble_row(f$memory))
fit_runtime <- bind_rows(c(lapply(fits, runtime_row), if (exists("determinism_fit")) list(runtime_row(determinism_fit)))) |>
  mutate(start_time = as.POSIXct(start_utc, format = "%Y-%m-%dT%H:%M:%OSZ", tz = "UTC"),
         end_time   = as.POSIXct(end_utc,   format = "%Y-%m-%dT%H:%M:%OSZ", tz = "UTC"))
fit_runtime$concurrent_fits_at_start <- vapply(seq_len(nrow(fit_runtime)), function(i) sum(fit_runtime$start_time <= fit_runtime$start_time[i] & fit_runtime$end_time > fit_runtime$start_time[i]) - 1L, integer(1))
fit_runtime$concurrent_fits_at_end   <- vapply(seq_len(nrow(fit_runtime)), function(i) sum(fit_runtime$start_time < fit_runtime$end_time[i] & fit_runtime$end_time >= fit_runtime$end_time[i]) - 1L, integer(1))
# process_peak_working_set_mb is the worker process's lifetime peak, so for a worker's second and later jobs it
# reflects the largest earlier job; job_index_in_worker makes that visible.
fit_runtime <- fit_runtime |>
  mutate(seconds_per_fit_iteration = seconds_fit_transform / iterations_fit,
         worker = match(pid, sort(unique(pid)))) |>
  arrange(start_time) |>
  group_by(pid) |> mutate(job_index_in_worker = row_number()) |> ungroup() |>
  select(-start_time, -end_time)
mark_stage("accounting_tables")

# ---- Comparison with the earlier SMART run (read-only inputs) ----
# Side-by-side values under both stopword conditions. Perplexity and coherence use different vocabularies, so
# those rows are descriptive and cannot be compared directly as model quality.
smart_tables <- c("text_processing_report", "document_accounting", "run_info", "model_comparison_summary", "parameters", "stability_summary")
smart_paths  <- setNames(file.path(smart_run_dir, "tables", paste0(smart_tables, ".csv")), smart_tables)
smart_available <- all(file.exists(smart_paths))
comparison_with_smart_run <- NULL
if (smart_available) {
  sm <- lapply(smart_paths, function(p) read_csv(p, show_col_types = FALSE, progress = FALSE, col_types = cols(.default = col_character())))
  lookup <- function(tbl, key_col, key, val_col) { v <- tbl[[val_col]][tbl[[key_col]] == key]; if (length(v)) v[1] else NA_character_ }
  cmp <- list()
  add_cmp <- function(item, smart, tier2, note = "") cmp[[length(cmp) + 1L]] <<- tibble(item = item, smart_run = as.character(smart), tier2_run = as.character(tier2), note = note)
  tpr <- sm$text_processing_report
  for (it in c("step 4: letter-run tokens extracted", "step 6: tokens shorter than 2 characters removed", "candidate tokens after steps 1-7 (all documents)",
               "distinct candidate terms in non-marker documents (vocabulary before pruning)", "vocabulary size (terms modeled)", "tokens modeled",
               "documents modeled (at least one modeled token)", "documents not modeled: only single-character tokens or stopwords",
               "documents not modeled: only terms outside the vocabulary", "step 9: highest document share of any term after stopword removal"))
    add_cmp(it, lookup(tpr, "item", it, "value"), lookup(text_processing_report, "item", it, "value"))
  add_cmp("step 7: stopword tokens removed", lookup(tpr, "item", "step 7: stopword tokens removed (SMART list, 571 terms)", "value"), n_tokens_stop_total, "SMART 571 terms vs Tier 2 310 entries")
  add_cmp("stopword entries in the list", 571, length(stopword_list))
  for (it in c("modeled: training set", "modeled: held-out set"))
    add_cmp(paste0("documents, ", it), lookup(sm$document_accounting, "stage", it, "total"), lookup(document_accounting, "stage", it, "total"))
  mcs <- sm$model_comparison_summary
  for (K in candidate_k) {
    kk <- as.character(K)
    for (col in c("iterations_fit_mean", "heldout_perplexity_mean", "heldout_perplexity_sd", "train_perplexity_mean", "coherence_npmi_mean", "exclusivity_mean",
                  "cao_juan_2009_mean", "matched_cosine_mean", "ari_dominant_topics_mean", "composite_rank", "elapsed_seconds_mean")) {
      note <- if (grepl("perplexity|coherence|exclusivity|cao", col)) "vocabulary-dependent: not a like-for-like model-quality comparison across stopword conditions" else
        if (col == "elapsed_seconds_mean") "SMART: 8 concurrent workers; Tier 2: see n_workers (used); same scope (fit to held-out completion)" else ""
      add_cmp(sprintf("K = %d: %s", K, col), lookup(mcs, "K", kk, col), model_comparison_summary[[col]][model_comparison_summary$K == K], note)
    }
  }
  par <- sm$parameters
  for (it in c("k_composite (lowest composite rank)", "k_coherence (highest NPMI coherence)", "k_perplexity_minimum (lowest mean held-out perplexity)", "k_perplexity (one-standard-error rule)"))
    add_cmp(it, lookup(par, "parameter", it, "value"), lookup(parameters_table, "parameter", it, "value"))
  add_cmp("retained_k", lookup(par, "parameter", "retained_k", "value"), paste(retained_k, collapse = ", "))
  add_cmp("n_workers", lookup(par, "parameter", "n_workers", "value"), fit_meta$n_workers_used)
  ri <- sm$run_info
  smart_stage_items <- c("total_seconds", "stage_seconds_read", "stage_seconds_tokenize", "stage_seconds_split", "stage_seconds_fits",
                         "stage_seconds_diagnostics", "stage_seconds_comparison", "stage_seconds_retained", "stage_seconds_figures",
                         "stage_seconds_validation", "stage_seconds_tables", "fits_wall_seconds_sum_over_workers")
  comparison_with_smart_run <- bind_rows(cmp)
  comparison_smart_stage_rows <- tibble(item = smart_stage_items, smart_run = vapply(smart_stage_items, function(it) lookup(ri, "item", it, "value"), ""))
}
mark_stage("smart_comparison")

# ---- Figures ----
# PNG figures. Every topic x topic figure names its quantity (topic-word distributions or document shares).
ink_series   <- "#2a78d6"
ink_series_2 <- "#eb6834"
ink_series_3 <- "#1baf7a"
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
seed_colours <- setNames(c(ink_series, ink_series_2, ink_series_3)[seq_along(fit_seeds)], as.character(fit_seeds))
condition_label <- "Tier 2 stopwords"

# Figure 1: candidate comparison: diagnostics against K, one point per seed, line through the seed mean.
metric_labels <- c(heldout_perplexity = "Held-out perplexity, document completion (lower is better)",
                   train_perplexity   = "Training perplexity (lower is better)",
                   coherence_npmi     = "Coherence, NPMI over top 10 words (higher is better)",
                   coherence_umass    = "Coherence, UMass over top 10 words (higher is better)",
                   exclusivity        = "Exclusivity of top 10 words (higher is better)",
                   cao_juan_2009      = "Mean pairwise topic-word cosine, Cao et al. 2009 (lower is better)",
                   deveaud_2014       = "Mean pairwise symmetric KL of topic-word distributions, Deveaud et al. 2014 (higher is better)",
                   arun_2010          = "Singular-value divergence, Arun et al. 2010 (lower is better)")
stability_labels <- c(matched_cosine_mean     = "Stability: mean matched-topic cosine across seeds, topic-word distributions (higher is better)",
                      ari_dominant_topics_mean = "Stability: adjusted Rand index of dominant topics, document labels (higher is better)")
metric_levels <- c(unname(metric_labels), unname(stability_labels))
comparison_long <- model_comparison_by_fit |>
  select(K, seed, all_of(names(metric_labels))) |>
  pivot_longer(-c(K, seed), names_to = "metric", values_to = "value") |>
  mutate(metric_label = factor(metric_labels[metric], levels = metric_levels))
stability_long <- stability_summary |>
  select(K, all_of(names(stability_labels))) |>
  pivot_longer(-K, names_to = "metric", values_to = "value") |>
  mutate(metric_label = factor(stability_labels[metric], levels = metric_levels))
comparison_means <- bind_rows(
  comparison_long |> group_by(K, metric_label) |> summarise(value = mean(value), .groups = "drop"),
  stability_long |> select(K, metric_label, value)) |>
  mutate(metric_label = factor(metric_label, levels = metric_levels))
fig1 <- ggplot() +
  geom_line(data = comparison_means, aes(x = K, y = value), colour = ink_sub, linewidth = 0.6) +
  geom_point(data = comparison_long, aes(x = K, y = value, colour = factor(seed)), size = 2) +
  geom_point(data = stability_long, aes(x = K, y = value), colour = ink_series, size = 2) +
  facet_wrap(~ metric_label, scales = "free_y", ncol = 2, labeller = label_wrap_gen(48)) +
  scale_colour_manual(values = seed_colours, name = "Random seed") +
  scale_x_continuous(breaks = candidate_k) +
  labs(title = sprintf("LDA candidate topic counts, %s — r/politics, July 2026", condition_label),
       x = "Number of topics (K)", y = "Diagnostic value") +
  theme_fig + theme(legend.position = "bottom")
ggsave(file.path(fig_dir, "fig1_model_comparison.png"), fig1, width = 11, height = 13, dpi = 150, bg = "white")

# Figure 2: pseudo log-likelihood per training token along the sampling iterations, one panel per K.
fig2 <- fit_traces |>
  mutate(K_label = factor(paste("K =", K), levels = paste("K =", candidate_k))) |>
  ggplot(aes(x = iteration, y = pseudo_loglikelihood_per_token, colour = factor(seed))) +
  geom_line(linewidth = 0.6) +
  facet_wrap(~ K_label, scales = "free_y", ncol = 4) +
  scale_colour_manual(values = seed_colours, name = "Random seed") +
  labs(title = sprintf("Sampling traces of the LDA fits, %s — r/politics, July 2026", condition_label),
       x = "Sampling iteration", y = "Pseudo log-likelihood per training token") +
  theme_fig + theme(legend.position = "bottom")
ggsave(file.path(fig_dir, "fig2_fit_traces.png"), fig2, width = 12, height = 6.5, dpi = 150, bg = "white")

# Figure 3: stability: matched-topic cosine similarity between seeds, per K.
fig3 <- stability_pairs |>
  mutate(K_factor = factor(K, levels = candidate_k)) |>
  ggplot(aes(x = K_factor, y = cosine)) +
  geom_boxplot(outlier.shape = NA, colour = ink_sub, fill = NA, width = 0.5, linewidth = 0.4) +
  geom_jitter(position = position_jitter(width = 0.18, height = 0, seed = 1), colour = ink_series, alpha = 0.45, size = 1.2) +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, by = 0.2)) +
  labs(title = sprintf("Topic stability across random seeds, %s — r/politics, July 2026", condition_label),
       x = "Number of topics (K)", y = "Cosine similarity of matched topic-word distributions") +
  theme_fig
ggsave(file.path(fig_dir, "fig3_stability_matched_cosine.png"), fig3, width = 9, height = 6, dpi = 150, bg = "white")

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

for (tag in names(retained)) {
  r <- retained[[tag]]; K <- r$K
  topic_levels_desc <- rev(r$topic_labels)
  ms <- matrix_size(K)
  # Figure 4: topic prevalence, token-weighted and document-mean shares.
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
  ggsave(file.path(fig_dir, sprintf("fig4_topic_prevalence_%s.png", tag)), fig4,
         width = 9, height = 3 + 0.22 * K, dpi = 150, bg = "white", limitsize = FALSE)

  # Figure 5: the ten highest-probability words of every topic.
  tw <- top_words_probability |>
    filter(K == !!K, seed == fit_seeds[1], rank <= 10) |>
    arrange(topic, rank) |>
    mutate(topic_label = factor(paste("Topic", topic), levels = r$topic_labels),
           key = paste0(term, "  #", topic)) |>
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
  ggsave(file.path(fig_dir, sprintf("fig5_top_words_%s.png", tag)), fig5,
         width = 2.6 * fig_cols + 1, height = 2.2 * ceiling(K / fig_cols) + 1, dpi = 130, bg = "white", limitsize = FALSE)

  # Figure 6: topic prevalence by UTC day.
  fig6 <- r$topic_prevalence_by_day |>
    mutate(topic_label = factor(paste("Topic", topic), levels = topic_levels_desc)) |>
    ggplot(aes(x = created_date_utc, y = topic_label, fill = token_share)) +
    geom_tile() +
    scale_fill_gradientn(colours = blue_ramp, labels = label_percent(accuracy = 1), name = "Share of the day's modeled tokens") +
    scale_x_date(date_breaks = "5 days", date_labels = "%b %d", expand = expansion(mult = 0.01)) +
    labs(title = sprintf("Topic prevalence by day, K = %d, %s — r/politics, July 2026 (UTC)", K, condition_label), x = "Day (UTC)", y = NULL) +
    theme_fig + theme(legend.position = "bottom", panel.grid.major = element_blank())
  ggsave(file.path(fig_dir, sprintf("fig6_topic_prevalence_by_day_%s.png", tag)), fig6,
         width = 10, height = 3 + 0.2 * K, dpi = 150, bg = "white", limitsize = FALSE)

  # Figure 7: document concentration: share of the dominant topic, by document length band.
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

  # Figure 8: Pearson correlation of topic shares across documents (document scores, not word distributions).
  cor_long <- r$topic_correlation |> select(-quantity) |>
    pivot_longer(-topic, names_to = "topic_b", values_to = "correlation") |>
    mutate(topic_b = as.integer(sub("topic_", "", topic_b))) |> filter(topic != topic_b)
  cor_limit <- max(abs(cor_long$correlation))
  fig8 <- matrix_figure(cor_long, "topic_b", "topic", "correlation", "Pearson correlation of the two topics' document shares (theta)",
                        sprintf("Topic co-occurrence within documents, K = %d, %s — r/politics, July 2026", K, condition_label),
                        "Topic (document shares)", "Topic (document shares)", c(-cor_limit, cor_limit), NULL, K, r$topic_labels, topic_levels_desc, diverging = TRUE)
  ggsave(file.path(fig_dir, sprintf("fig8_topic_correlation_%s.png", tag)), fig8, width = ms$width, height = ms$height, dpi = 150, bg = "white", limitsize = FALSE)

  # Figure 9: stability matrix: topic-word cosine similarity, seed 1 topics against seed 2 topics.
  sc <- stability_cos[[paste(K, fit_seeds[2], sep = "_")]]
  fig9 <- matrix_figure(r$stability_matrices[[as.character(fit_seeds[2])]], "seed_b_topic", "seed1_topic", "cosine",
                        "Cosine similarity of topic-word distributions (phi)",
                        sprintf("Topic matching between seeds 1 and 2, K = %d, %s — r/politics, July 2026", K, condition_label),
                        "Seed 2 topics, in matched order (topic-word distributions)", "Seed 1 topics (topic-word distributions)", c(0, 1), blue_ramp, K,
                        paste("Topic", sc$assignment), topic_levels_desc)
  ggsave(file.path(fig_dir, sprintf("fig9_stability_matrix_%s.png", tag)), fig9, width = ms$width, height = ms$height, dpi = 150, bg = "white", limitsize = FALSE)

  # Figure 10: topic prevalence by document type.
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
  ggsave(file.path(fig_dir, sprintf("fig10_topic_prevalence_by_type_%s.png", tag)), fig10,
         width = 9, height = 3 + 0.22 * K, dpi = 150, bg = "white", limitsize = FALSE)

  # Figure 11: within-fit cosine similarity of topic-word distributions (word distributions, not document scores).
  wc_long <- r$topic_word_cosine |> select(-quantity) |>
    pivot_longer(-topic, names_to = "topic_b", values_to = "cosine") |>
    mutate(topic_b = as.integer(sub("topic_", "", topic_b))) |> filter(topic != topic_b)
  fig11 <- matrix_figure(wc_long, "topic_b", "topic", "cosine", "Cosine similarity of the two topics' topic-word distributions (phi)",
                         sprintf("Topic-word similarity between topics, K = %d, %s — r/politics, July 2026", K, condition_label),
                         "Topic (topic-word distributions)", "Topic (topic-word distributions)", c(0, 1), blue_ramp, K, r$topic_labels, topic_levels_desc)
  ggsave(file.path(fig_dir, sprintf("fig11_topic_word_cosine_%s.png", tag)), fig11, width = ms$width, height = ms$height, dpi = 150, bg = "white", limitsize = FALSE)

  # Figure 12: co-activation: share of modeled documents in which both topics are active (document scores).
  ca_long <- r$co_activation |> select(-quantity) |>
    pivot_longer(-topic, names_to = "topic_b", values_to = "share") |>
    mutate(topic_b = as.integer(sub("topic_", "", topic_b))) |> filter(topic != topic_b)
  fig12 <- matrix_figure(ca_long, "topic_b", "topic", "share", sprintf("Share of modeled documents with both topics at or above %g (document shares)", active_topic_threshold),
                         sprintf("Topic co-activation within documents, K = %d, %s — r/politics, July 2026", K, condition_label),
                         "Topic (document shares)", "Topic (document shares)", c(0, max(ca_long$share)), blue_ramp, K, r$topic_labels, topic_levels_desc)
  ggsave(file.path(fig_dir, sprintf("fig12_topic_coactivation_%s.png", tag)), fig12, width = ms$width, height = ms$height, dpi = 150, bg = "white", limitsize = FALSE)
}

# Figure 13: K selection: stability of the seed-1 topics against the other seeds (topic-word distributions), by K.
k_stab_labels <- c(seed1_share_topics_reproduced_in_both_other_seeds = sprintf("Share of seed-1 topics matched at cosine >= %g in both other seeds (topic-word)", stability_cosine_threshold),
                   seed1_share_topics_below_0.5_in_both_other_seeds = "Share of seed-1 topics matched below cosine 0.5 in both other seeds (topic-word)",
                   seed1_weaker_matched_cosine_mean = "Mean of each seed-1 topic's weaker matched cosine (topic-word)",
                   share_matched_not_row_maximum = "Share of matched pairs where the one-to-one match is not the row maximum (topic-word)")
fig13 <- stability_summary |>
  select(K, all_of(names(k_stab_labels))) |>
  pivot_longer(-K, names_to = "metric", values_to = "value") |>
  mutate(metric_label = factor(k_stab_labels[metric], levels = unname(k_stab_labels))) |>
  ggplot(aes(x = K, y = value)) +
  geom_line(colour = ink_sub, linewidth = 0.6) + geom_point(colour = ink_series, size = 2) +
  facet_wrap(~ metric_label, scales = "free_y", ncol = 2, labeller = label_wrap_gen(52)) +
  scale_x_continuous(breaks = candidate_k) +
  labs(title = sprintf("Topic reproducibility across seeds by K, %s — r/politics, July 2026", condition_label),
       x = "Number of topics (K)", y = "Value") +
  theme_fig
ggsave(file.path(fig_dir, "fig13_k_selection_stability.png"), fig13, width = 11, height = 7, dpi = 150, bg = "white")

# Figure 14: K selection: within-fit topic distinctness, word-based and document-based, one point per seed.
distinct_labels <- c(word_cosine_pair_max = "Most similar topic pair: cosine of topic-word distributions (lower = more distinct)",
                     top_words_jaccard_pair_mean = "Mean Jaccard overlap of top-20 words over topic pairs (topic-word; lower = more distinct)",
                     doc_correlation_pair_max = "Most correlated topic pair: Pearson correlation of document shares (document scores)",
                     doc_correlation_pair_mean_abs = "Mean absolute Pearson correlation of document shares over topic pairs (document scores)",
                     co_activation_pair_max = sprintf("Most co-active topic pair: share of documents with both topics >= %g (document scores)", active_topic_threshold),
                     co_activation_pair_mean = sprintf("Mean co-activation share over topic pairs (document scores, threshold %g)", active_topic_threshold))
distinct_long <- model_comparison_by_fit |>
  select(K, seed, all_of(names(distinct_labels))) |>
  pivot_longer(-c(K, seed), names_to = "metric", values_to = "value") |>
  mutate(metric_label = factor(distinct_labels[metric], levels = unname(distinct_labels)))
distinct_means <- distinct_long |> group_by(K, metric_label) |> summarise(value = mean(value), .groups = "drop")
fig14 <- ggplot() +
  geom_line(data = distinct_means, aes(x = K, y = value), colour = ink_sub, linewidth = 0.6) +
  geom_point(data = distinct_long, aes(x = K, y = value, colour = factor(seed)), size = 2) +
  facet_wrap(~ metric_label, scales = "free_y", ncol = 2, labeller = label_wrap_gen(52)) +
  scale_colour_manual(values = seed_colours, name = "Random seed") +
  scale_x_continuous(breaks = candidate_k) +
  labs(title = sprintf("Topic distinctness within fits by K, %s — r/politics, July 2026", condition_label),
       x = "Number of topics (K)", y = "Value") +
  theme_fig + theme(legend.position = "bottom")
ggsave(file.path(fig_dir, "fig14_k_selection_distinctness.png"), fig14, width = 11, height = 9, dpi = 150, bg = "white")

# Figure 15: the fit schedule: one bar per fit, by worker process, from the first fit start.
sched <- fit_runtime |>
  mutate(start_time = as.POSIXct(start_utc, format = "%Y-%m-%dT%H:%M:%OSZ", tz = "UTC"),
         end_time   = as.POSIXct(end_utc,   format = "%Y-%m-%dT%H:%M:%OSZ", tz = "UTC"))
t0 <- min(sched$start_time)
sched <- sched |> mutate(start_min = as.numeric(difftime(start_time, t0, units = "mins")),
                         end_min = as.numeric(difftime(end_time, t0, units = "mins")),
                         worker_label = factor(sprintf("Worker %d (pid %d)", worker, pid), levels = rev(unique(sprintf("Worker %d (pid %d)", worker, pid)[order(worker)]))),
                         K_label = factor(paste("K =", K), levels = paste("K =", candidate_k)))
fig15 <- ggplot(sched) +
  geom_segment(aes(x = start_min, xend = end_min, y = worker_label, yend = worker_label, colour = K_label), linewidth = 6) +
  scale_colour_manual(values = colorRampPalette(blue_ramp)(length(candidate_k)), name = "Topic count") +
  labs(title = sprintf("Fit schedule across worker processes, %s — r/politics, July 2026", condition_label),
       x = "Minutes since the first fit started", y = NULL) +
  theme_fig + theme(legend.position = "bottom", panel.grid.major.y = element_blank())
ggsave(file.path(fig_dir, "fig15_fit_schedule.png"), fig15, width = 11, height = 2 + 0.35 * n_distinct(sched$worker), dpi = 150, bg = "white", limitsize = FALSE)

# Figure 16: per-fit wall time and peak memory against K.
rt_long <- fit_runtime |>
  filter(role == "primary") |>
  select(K, seed, `Wall time of the whole worker job (s)` = elapsed_seconds, `Wall time, fit to held-out completion (s)` = elapsed_seconds_fit_to_completion,
         `CPU time of the whole worker job (s)` = cpu_user_seconds_total, `Peak working set of the worker process (MB)` = process_peak_working_set_mb) |>
  pivot_longer(-c(K, seed), names_to = "measure", values_to = "value") |>
  mutate(measure = factor(measure, levels = c("Wall time of the whole worker job (s)", "Wall time, fit to held-out completion (s)",
                                              "CPU time of the whole worker job (s)", "Peak working set of the worker process (MB)")))
fig16 <- ggplot(rt_long, aes(x = K, y = value, colour = factor(seed))) +
  geom_point(size = 2) +
  facet_wrap(~ measure, scales = "free_y", ncol = 2) +
  scale_colour_manual(values = seed_colours, name = "Random seed") +
  scale_x_continuous(breaks = candidate_k) +
  labs(title = sprintf("Runtime and memory per fit, %s — r/politics, July 2026", condition_label), x = "Number of topics (K)", y = "Value") +
  theme_fig + theme(legend.position = "bottom")
ggsave(file.path(fig_dir, "fig16_fit_runtime_memory.png"), fig16, width = 11, height = 7, dpi = 150, bg = "white")
mark_stage("figures")
log_line("figures written")

# ---- Validation checks ----
# Reconcile counts, recompute sampled dtm rows by a second code path, and check the split, stopword manifest,
# counts against the SMART run's report, distributions, determinism job, perplexity (vs text2vec::perplexity),
# topic matching, retention rule, theta files and retained tables. Stops if any check fails.
checks <- list()
add_check <- function(check, detail, pass) checks[[length(checks) + 1L]] <<- tibble(check = check, detail = detail, pass = isTRUE(pass))

add_check("comment and submission ids unique and non-missing; doc_key unique across both types",
          sprintf("%d comment ids, %d submission ids, %d doc_keys distinct of %d", n_distinct(comments$id), n_distinct(submissions$id), n_distinct(docs$doc_key), n_docs),
          n_distinct(comments$id) == n_comments_all && !anyNA(comments$id) && n_distinct(submissions$id) == n_submissions_all &&
            !anyNA(submissions$id) && n_distinct(docs$doc_key) == n_docs)
add_check("every comment link_id matches ^t3_[0-9a-z]+$ and every submission id is bare base36",
          sprintf("%d of %d link_ids; %d of %d submission ids", sum(grepl("^t3_[0-9a-z]+$", comments$link_id)), n_comments_all,
                  sum(grepl("^[0-9a-z]+$", submissions$id)), n_submissions_all),
          sum(grepl("^t3_[0-9a-z]+$", comments$link_id)) == n_comments_all && sum(grepl("^[0-9a-z]+$", submissions$id)) == n_submissions_all)
add_check("no missing (NA) text in body, title, selftext",
          sprintf("%d missing values", n_text_missing), n_text_missing == 0)
add_check("stopword workbook: 310 distinct entries read from the named sheet; file hash equals the hash in its provenance manifest; decision and notes blank",
          sprintf("%d entries, %d distinct; hash %s; manifest %s; %d decisions, %d notes", length(stopword_list), n_distinct(stopword_list),
                  workbook_hash, coalesce(workbook_manifest_hash, "NA"), sum(!is.na(tier2_table$decision)), sum(!is.na(tier2_table$notes))),
          length(stopword_list) == 310L && n_distinct(stopword_list) == 310L && identical(workbook_hash, workbook_manifest_hash))
add_check("stopword accounting: Tier 2 removals + tokens kept = tokens entering step 7; Tier 2-only + SMART-only tokens reconcile with the two reference counts",
          sprintf("%d + %d = %d; SMART reference %d = both %d + SMART-only %d; Tier 2 %d = both + Tier 2-only %d",
                  n_tokens_stop_total, n_candidate_tokens, n_tokens_after_step6_all, n_stop_smart_reference,
                  n_stop_smart_reference - n_smart_only_tokens, n_smart_only_tokens, n_tokens_stop_total, n_tier2_only_tokens),
          n_tokens_stop_total + n_candidate_tokens == n_tokens_after_step6_all &&
            n_tokens_stop_total - n_tier2_only_tokens == n_stop_smart_reference - n_smart_only_tokens &&
            sum(occ_tier2$all[tier2_table$n_chars >= min_token_chars]) == n_tokens_stop_total)
smart_report_path <- file.path(smart_run_dir, "tables/text_processing_report.csv")
if (file.exists(smart_report_path)) {
  smart_report <- read_csv(smart_report_path, show_col_types = FALSE, progress = FALSE, col_types = cols(.default = col_character()))
  invariant_items <- c("documents in source (comments + submissions)", "documents whose whole text is a removal marker (excluded, step 8)",
                       "step 1: case folding — distinct letter-run tokens if case were preserved (not applied)",
                       "step 1: case folding — distinct letter-run tokens after case folding",
                       "step 2: curly apostrophes (U+2019) normalised to ' (occurrences)", "step 3: URL matches removed",
                       "step 3: documents containing at least one URL", "step 4: letter-run tokens extracted",
                       "step 4: documents with zero letter-run tokens", "step 5: tokens with a trailing possessive 's stripped",
                       "step 5: distinct tokens after possessive stripping", sprintf("step 6: tokens shorter than %d characters removed", min_token_chars))
  smart_vals <- as.numeric(smart_report$value[match(invariant_items, smart_report$item)])
  tier2_vals <- as.numeric(text_processing_report$value[match(invariant_items, text_processing_report$item)])
  smart_stop <- as.numeric(smart_report$value[smart_report$item == "step 7: stopword tokens removed (SMART list, 571 terms)"])
  add_check("tokenizer reproduced: the 12 stopword-invariant counts (steps 1-6) equal the SMART run's report, and the SMART reference count here equals its recorded step-7 removals",
            sprintf("%d of %d counts equal; SMART reference %d vs original %d", sum(smart_vals == tier2_vals), length(invariant_items), n_stop_smart_reference, smart_stop),
            all(smart_vals == tier2_vals) && length(smart_stop) == 1 && smart_stop == n_stop_smart_reference)
}
status_counts <- table(factor(docs$model_status, levels = status_levels))
add_check("document accounting: statuses sum to source documents; modeled = train + held-out",
          sprintf("%s = %d; %d = %d + %d", paste(status_counts, collapse = " + "), n_docs, n_modeled, n_train, n_heldout),
          sum(status_counts) == n_docs && n_modeled == n_train + n_heldout && n_docs == n_comments_all + n_submissions_all)
add_check("token accounting: letter tokens = short + stopword + marker-document + rare + modeled",
          sprintf("%d = %d + %d + %d + %d + %d", n_letter_tokens, n_tokens_short_total, n_tokens_stop_total, n_tokens_in_marker_docs,
                  n_tokens_rare_total, n_tokens_modeled_total),
          n_letter_tokens == n_tokens_short_total + n_tokens_stop_total + n_tokens_in_marker_docs + n_tokens_rare_total + n_tokens_modeled_total &&
            n_candidate_tokens == n_letter_tokens - n_tokens_short_total - n_tokens_stop_total)
add_check("vocabulary: kept terms satisfy the df floor and ceiling; dtm columns are the vocabulary; column sums equal term frequencies; no Tier 2 entry is in the vocabulary",
          sprintf("%d terms; df %d..%d; ceiling %.0f; %d Tier 2 entries in vocabulary", n_vocabulary, min(term_df_all[term_kept]), max(term_df_all[term_kept]),
                  max_document_share * n_nonmarker_docs, sum(vocabulary %in% stopword_list)),
          all(term_df_all[term_kept] >= min_document_frequency) && all(term_df_all[term_kept] <= max_document_share * n_nonmarker_docs) &&
            identical(colnames(dtm_all), vocabulary) && all(colSums(dtm_all) == term_tf_all[term_kept]) && nrow(dtm_all) == n_docs &&
            identical(rownames(dtm_all), docs$doc_key) && !any(vocabulary %in% stopword_list))
add_check("marker-only documents and non-modeled documents have empty dtm rows; modeled rows are non-empty",
          sprintf("%d marker rows with tokens; %d non-modeled rows with tokens; %d modeled rows empty",
                  sum(docs$n_tokens_modeled[docs$marker_only] > 0), sum(docs$n_tokens_modeled[docs$model_status != "modeled"] > 0),
                  sum(docs$n_tokens_modeled[modeled_rows] == 0)),
          all(docs$n_tokens_modeled[docs$model_status != "modeled"] == 0) && all(docs$n_tokens_modeled[modeled_rows] >= 1))

recompute_tokens <- function(text) {
  x <- stri_replace_all_regex(stri_replace_all_fixed(stri_trans_tolower(text), "\u2019", "'"), url_regex, " ", opts_regex = url_opts)
  t <- stri_extract_all_regex(x, token_regex, omit_no_match = TRUE)[[1]]
  t <- stri_replace_last_regex(t, "'s$", "")
  t <- t[stri_length(t) >= min_token_chars & !(t %in% stopword_list)]
  t[t %in% vocabulary]
}
sample_docs <- seq(1L, n_docs, by = 5000L)
row_matches <- vapply(sample_docs, function(i) {
  expected <- if (docs$marker_only[i]) character(0) else recompute_tokens(docs$text[i])
  counts   <- table(expected)
  row      <- dtm_all[i, ]
  nz       <- which(row > 0)
  length(nz) == length(counts) && (length(nz) == 0 || all(row[match(names(counts), vocabulary)] == as.numeric(counts)))
}, logical(1))
add_check("dtm rows equal table() of tokens recomputed from the stored text by a second code path (every 5,000th document)",
          sprintf("%d of %d sampled documents match", sum(row_matches), length(sample_docs)), all(row_matches))

add_check("held-out split: sampled from modeled documents with >= 2 tokens; size = floor(share x modeled); disjoint from training",
          sprintf("%d held-out (min tokens %d); %d training; overlap %d", n_heldout, min(docs$n_tokens_modeled[heldout_rows]), n_train,
                  length(intersect(train_rows, heldout_rows))),
          n_heldout == floor(heldout_share * n_modeled) && min(docs$n_tokens_modeled[heldout_rows]) >= 2 &&
            length(intersect(train_rows, heldout_rows)) == 0 && all(docs$split[heldout_rows] == "held-out") && all(docs$split[train_rows] == "train"))
add_check("held-out halves: A + B equals the held-out matrix; every held-out document has >= 1 token in each half",
          sprintf("max |A + B - X| = %g; min half A %d; min half B %d; tokens %d + %d = %d",
                  max(abs(dtm_heldout_a + dtm_heldout_b - dtm_heldout)), min(rowSums(dtm_heldout_a)), min(rowSums(dtm_heldout_b)),
                  sum(dtm_heldout_a), sum(dtm_heldout_b), n_heldout_tokens),
          max(abs(dtm_heldout_a + dtm_heldout_b - dtm_heldout)) == 0 && min(rowSums(dtm_heldout_a)) >= 1 && min(rowSums(dtm_heldout_b)) >= 1)

fit_ok <- vapply(fits, function(f) {
  nrow(f$components) == f$K && ncol(f$components) == n_vocabulary && abs(sum(f$components) - n_train_tokens) < 1e-6 &&
    all(abs(rowSums(f$phi) - 1) < 1e-9) && abs(sum(f$token_share) - 1) < 1e-9 && abs(sum(f$doc_mean_share) - 1) < 1e-9 &&
    all(f$dominant >= 1 & f$dominant <= f$K) && length(f$dominant) == n_train && all(f$max_theta > 0 & f$max_theta <= 1 + 1e-12) &&
    f$iterations >= n_check_convergence && f$iterations <= n_iter_max && all(diff(f$token_share) <= 1e-12) &&
    length(f$dominant_m) == n_modeled && identical(f$dominant_m[seq_len(n_train)], f$dominant) &&
    all(abs(diag(f$doc_cor) - 1) < 1e-9) && all(abs(diag(f$word_cos) - 1) < 1e-9) && all(f$co_active >= 0 & f$co_active <= 1 + 1e-12) &&
    nrow(f$rep_docs) == f$K * n_representative && file.exists(f$theta_file)
}, logical(1))
add_check("every fit: K x V topic-word counts summing to the training tokens; phi rows, token shares and document-mean shares sum to 1; dominant topics valid and equal on the training rows of the all-document summary; topics ordered by token share; K x K matrices well formed; representative rows complete; theta file present",
          sprintf("%d of %d fits pass (%d requested)", sum(fit_ok), length(fits), length(job_list)), all(fit_ok) && length(fits) == length(job_list))

if (determinism_check) {
  add_check(sprintf("determinism: a second worker process re-fitting K = %d, seed %d reproduces the topic-word counts, the full theta (matrix digest) and the dominant topics exactly", determinism_result$K, determinism_result$seed),
            sprintf("components identical %s; theta identical %s (digests %s / %s); dominant identical %s; pids %d / %d; from cache %s",
                    determinism_result$components_identical, determinism_result$theta_identical, determinism_result$theta_digest_primary,
                    determinism_result$theta_digest_determinism, determinism_result$dominant_identical, determinism_result$pid_primary,
                    determinism_result$pid_determinism, determinism_result$from_cache),
            determinism_result$components_identical && determinism_result$theta_identical && determinism_result$dominant_identical &&
              identical(determinism_result$theta_digest_primary, determinism_result$theta_digest_determinism))
}

check_fit <- fits[[fit_key(retained_k[1], fit_seeds[1])]]
perp_rows <- seq(1L, n_train, by = 400L)
theta_check <- matrix(0, length(perp_rows), check_fit$K)
theta_check[cbind(seq_along(perp_rows), check_fit$dominant[perp_rows])] <- 1
theta_check <- smooth_theta(theta_check, train_doc_tokens[perp_rows], alpha_sum / check_fit$K)
perp_own <- log_likelihood(dtm_train[perp_rows, , drop = FALSE], check_fit$phi, theta_check)[["perplexity"]]
perp_ref <- perplexity(dtm_train[perp_rows, , drop = FALSE], check_fit$phi, theta_check)
add_check("perplexity implementation agrees with text2vec::perplexity on the same inputs (every 400th training document)",
          sprintf("%.6f vs %.6f on %d documents", perp_own, perp_ref, length(perp_rows)), abs(perp_own - perp_ref) / perp_ref < 1e-6)

npmi_all <- unlist(lapply(fits, function(f) f$topic_npmi)); umass_all <- unlist(lapply(fits, function(f) f$topic_umass))
excl_all <- unlist(lapply(fits, function(f) f$topic_exclusivity))
add_check("coherence and exclusivity within their ranges: NPMI in [-1, 1], UMass <= 0, exclusivity in (0, 1]",
          sprintf("NPMI %.3f..%.3f; UMass %.3f..%.3f; exclusivity %.3f..%.3f", min(npmi_all), max(npmi_all), min(umass_all), max(umass_all), min(excl_all), max(excl_all)),
          all(npmi_all >= -1 & npmi_all <= 1) && all(umass_all <= 1e-12) && all(excl_all > 0 & excl_all <= 1 + 1e-12))
add_check("stability: every Hungarian assignment is a permutation; matched cosines in [0, 1] and never above the row maximum; ARI of a labelling with itself = 1",
          sprintf("%d of %d assignments are permutations; cosine %.3f..%.3f; self-ARI %.6f", sum(stability_assignment_ok), length(stability_assignment_ok),
                  min(stability_pairs$cosine), max(stability_pairs$cosine), adjusted_rand_index(fits[[1]]$dominant, fits[[1]]$dominant)),
          all(stability_assignment_ok) && all(stability_pairs$cosine >= 0 & stability_pairs$cosine <= 1 + 1e-9) &&
            all(stability_pairs$cosine <= stability_pairs$best_cosine_any_topic_b + 1e-12) &&
            abs(adjusted_rand_index(fits[[1]]$dominant, fits[[1]]$dominant) - 1) < 1e-12 &&
            nrow(stability_pairs) == sum(candidate_k) * length(seed_pairs))
recomputed_composite <- with(model_comparison_summary, (rank(heldout_perplexity_mean) + rank(-coherence_npmi_mean) + rank(-exclusivity_mean) + rank(-matched_cosine_mean)) / 4)
recomputed_threshold <- with(model_comparison_summary, min(heldout_perplexity_mean) + coalesce(heldout_perplexity_sd[which.min(heldout_perplexity_mean)], 0))
add_check("retention rule recomputed from the summary table gives the retained K",
          sprintf("composite best K = %d; coherence best K = %d; held-out perplexity minimum K = %d (threshold %.2f), one-standard-error K = %d; retained %s",
                  k_composite, k_coherence, k_perplexity_minimum, perplexity_threshold, k_perplexity, paste(retained_k, collapse = ", ")),
          all(recomputed_composite == model_comparison_summary$composite_rank) &&
            model_comparison_summary$K[which.min(recomputed_composite)] == k_composite &&
            model_comparison_summary$K[which.max(model_comparison_summary$coherence_npmi_mean)] == k_coherence &&
            model_comparison_summary$K[which.min(model_comparison_summary$heldout_perplexity_mean)] == k_perplexity_minimum &&
            min(model_comparison_summary$K[model_comparison_summary$heldout_perplexity_mean <= recomputed_threshold]) == k_perplexity &&
            k_perplexity <= k_perplexity_minimum && setequal(retained_k, c(k_composite, k_coherence, k_perplexity)))
rd_all <- representative_documents_all_models |> inner_join(docs |> select(doc_key, s_text = text, s_tokens = n_tokens_modeled), by = "doc_key")
add_check("representative documents of every fit: text equals the source, >= 25 modeled tokens, distinct texts and authors within a topic, n_representative rows per topic",
          sprintf("%d rows over %d fits; %d texts agree; min tokens %d; %d of %d topics with distinct texts and authors", nrow(rd_all), length(fits),
                  sum(rd_all$text == rd_all$s_text), min(rd_all$s_tokens),
                  sum((rd_all |> group_by(K, seed, topic) |> summarise(ok = n_distinct(text) == n() & n_distinct(author) == n() & n() == n_representative, .groups = "drop"))$ok), sum(candidate_k) * length(fit_seeds)),
          nrow(rd_all) == sum(candidate_k) * length(fit_seeds) * n_representative && all(rd_all$text == rd_all$s_text) && min(rd_all$s_tokens) >= representative_min_tokens &&
            all((rd_all |> group_by(K, seed, topic) |> summarise(ok = n_distinct(text) == n() & n_distinct(author) == n() & n() == n_representative, .groups = "drop"))$ok))
add_check("runtime record: one row per job with start before end, positive phase times summing to the job time, and a process id",
          sprintf("%d rows; %d jobs; max |sum of phases - elapsed| = %.3f s", nrow(fit_runtime), length(job_list) + as.integer(determinism_check),
                  max(abs(with(fit_runtime, seconds_fit_transform + seconds_train_loglik + seconds_heldout_completion + seconds_heldout_transform +
                                 seconds_document_summaries + seconds_save_theta + seconds_diagnostics - elapsed_seconds)))),
          nrow(fit_runtime) == length(job_list) + as.integer(determinism_check) && all(fit_runtime$elapsed_seconds > 0) && !anyNA(fit_runtime$pid) &&
            max(abs(with(fit_runtime, seconds_fit_transform + seconds_train_loglik + seconds_heldout_completion + seconds_heldout_transform +
                           seconds_document_summaries + seconds_save_theta + seconds_diagnostics - elapsed_seconds))) < 0.01)

for (tag in names(retained)) {
  r <- retained[[tag]]; K <- r$K; dt <- r$document_topic_table; tc <- r$topic_cols
  add_check(sprintf("%s: theta file from the worker is consistent (doc keys in modeled order, K, seed, sampler index, dimensions) and its per-document summaries equal the worker's", tag),
            sprintf("file consistent %s; summaries identical %s", r$theta_file_consistent, r$worker_summaries_identical),
            r$theta_file_consistent && r$worker_summaries_identical)
  theta_mat <- as.matrix(dt[, tc])
  modeled_dt <- dt$model_status == "modeled"
  row_sums <- rowSums(theta_mat[modeled_dt, , drop = FALSE])
  add_check(sprintf("%s: document-topic table has one row per source document; modeled rows sum to 1 (rounded to 6 decimals); non-modeled rows are NA", tag),
            sprintf("%d rows, %d distinct keys; %d modeled rows, sums %.6f..%.6f; %d NA cells in non-modeled rows of %d",
                    nrow(dt), n_distinct(dt$doc_key), sum(modeled_dt), min(row_sums), max(row_sums),
                    sum(is.na(theta_mat[!modeled_dt, ])), sum(!modeled_dt) * K),
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
  ts <- r$topic_summary
  add_check(sprintf("%s: prevalence: token shares, document-mean shares and dominant counts reconcile; shares by type and by day sum to 1; numbering follows training shares", tag),
            sprintf("token %.9f; doc-mean %.9f; dominant %d = %d; type sums %.6f..%.6f; day sums %.6f..%.6f over %d days; training shares descending %s",
                    sum(ts$token_share), sum(ts$doc_mean_share), sum(ts$dominant_docs), n_modeled,
                    min(tapply(r$topic_prevalence_by_type$token_share, r$topic_prevalence_by_type$doc_type, sum)),
                    max(tapply(r$topic_prevalence_by_type$token_share, r$topic_prevalence_by_type$doc_type, sum)),
                    min(tapply(r$topic_prevalence_by_day$token_share, r$topic_prevalence_by_day$created_date_utc, sum)),
                    max(tapply(r$topic_prevalence_by_day$token_share, r$topic_prevalence_by_day$created_date_utc, sum)),
                    n_distinct(r$topic_prevalence_by_day$created_date_utc), all(diff(ts$token_share_train) <= 1e-12)),
            abs(sum(ts$token_share) - 1) < 1e-9 && abs(sum(ts$doc_mean_share) - 1) < 1e-9 && sum(ts$dominant_docs) == n_modeled &&
              all(abs(tapply(r$topic_prevalence_by_type$token_share, r$topic_prevalence_by_type$doc_type, sum) - 1) < 1e-9) &&
              all(abs(tapply(r$topic_prevalence_by_day$token_share, r$topic_prevalence_by_day$created_date_utc, sum) - 1) < 1e-9) &&
              n_distinct(r$topic_prevalence_by_day$created_date_utc) == n_distinct(docs$created_date_utc[modeled_rows]) &&
              all(diff(ts$token_share_train) <= 1e-12) && abs(sum(ts$token_share_train) - 1) < 1e-9)
  rd <- r$representative_documents
  rd_check <- rd |> inner_join(dt |> select(doc_key, all_of(tc)), by = "doc_key") |>
    inner_join(docs |> select(doc_key, s_text = text, s_tokens = n_tokens_modeled), by = "doc_key")
  rd_theta <- as.matrix(rd_check[, tc])[cbind(seq_len(nrow(rd_check)), rd_check$topic)]
  add_check(sprintf("%s: representative documents: share agrees with the table (6 decimals), text equals the source, >= %d modeled tokens, distinct texts and authors, %d rows per topic",
                    tag, representative_min_tokens, n_representative),
            sprintf("%d rows; max share difference %g; %d texts agree; min tokens %d; %d topics with %d distinct texts; %d topics with %d distinct authors",
                    nrow(rd), max(abs(rd_theta - rd_check$topic_share_in_document)), sum(rd_check$text == rd_check$s_text), min(rd_check$s_tokens),
                    sum(tapply(rd$text, rd$topic, n_distinct) == n_representative), n_representative,
                    sum(tapply(rd$author, rd$topic, n_distinct) == n_representative), n_representative),
            nrow(rd_check) == nrow(rd) && max(abs(rd_theta - rd_check$topic_share_in_document)) < 1e-6 && all(rd_check$text == rd_check$s_text) &&
              min(rd_check$s_tokens) >= representative_min_tokens && all(tapply(rd$text, rd$topic, n_distinct) == n_representative) &&
              all(tapply(rd$author, rd$topic, n_distinct) == n_representative) && nrow(rd) == K * n_representative)
  tw_mat <- as.matrix(r$topic_word_table[, tc])
  add_check(sprintf("%s: topic-word table has one row per vocabulary term and every topic column sums to 1 (6 significant digits)", tag),
            sprintf("%d rows; column sums %.6f..%.6f", nrow(tw_mat), min(colSums(tw_mat)), max(colSums(tw_mat))),
            nrow(tw_mat) == n_vocabulary && all(abs(colSums(tw_mat) - 1) < 1e-4) && identical(r$topic_word_table$term, vocabulary))
  cor_m <- as.matrix(r$topic_correlation[, tc]); wc_m <- as.matrix(r$topic_word_cosine[, tc]); ca_m <- as.matrix(r$co_activation[, tc])
  add_check(sprintf("%s: concentration bins contain every modeled document; the three K x K matrices are symmetric with unit diagonal (correlation, word cosine) or a diagonal equal to the topic's active share recomputed from the unrounded theta file (co-activation)", tag),
            sprintf("%d docs in bins; %d x %d; correlation diagonal %.6f..%.6f; word-cosine diagonal %.6f..%.6f; co-activation diagonal %.4f..%.4f; max |diagonal - active share| %g",
                    sum(r$concentration$docs), nrow(cor_m), length(tc), min(diag(cor_m)), max(diag(cor_m)), min(diag(wc_m)), max(diag(wc_m)), min(diag(ca_m)), max(diag(ca_m)),
                    max(abs(diag(ca_m) - r$active_share))),
            sum(r$concentration$docs) == n_modeled && nrow(cor_m) == K && all(abs(diag(cor_m) - 1) < 1e-9) && isSymmetric(unname(cor_m)) &&
              all(abs(diag(wc_m) - 1) < 1e-9) && isSymmetric(unname(wc_m)) && isSymmetric(unname(ca_m)) &&
              all(abs(diag(ca_m) - r$active_share) < 1e-9))
}
add_check("figure data: comparison rows = fits x metrics; trace rows = sum of checks per fit; stability rows = sum of K x seed pairs; topic-pair rows = 2 x sum of K(K-1)/2 x seeds per table",
          sprintf("%d = %d x %d; %d; %d; %d and %d", nrow(comparison_long), length(fits), length(metric_labels), nrow(fit_traces), nrow(stability_pairs),
                  nrow(topic_pairs_word), nrow(topic_pairs_document)),
          nrow(comparison_long) == length(fits) * length(metric_labels) && nrow(fit_traces) == sum(vapply(fits, function(f) nrow(f$trace), 1L)) &&
            nrow(stability_pairs) == sum(candidate_k) * length(seed_pairs) &&
            nrow(topic_pairs_word) == 2 * sum(candidate_k * (candidate_k - 1) / 2) * length(fit_seeds) && nrow(topic_pairs_document) == nrow(topic_pairs_word))

validation_checks <- bind_rows(checks)
print(as.data.frame(validation_checks[, c("check", "pass")]), right = FALSE)
mark_stage("validation")

# ---- Write tables ----
# One CSV per table; stops before writing anything else if a validation check failed.
table_dims <- list()
write_table <- function(tbl, name) {
  path <- file.path(tab_dir, name)
  write_csv(tbl, path)
  table_dims[[path]] <<- c(rows = nrow(tbl), cols = ncol(tbl))
  invisible(path)
}
write_table(validation_checks, "validation_checks.csv")
stopifnot(all(validation_checks$pass))

write_table(document_accounting,         "document_accounting.csv")
write_table(text_processing_report,      "text_processing_report.csv")
write_table(stopword_accounting,         "stopword_accounting.csv")
write_table(text_availability,           "text_availability.csv")
write_table(stopwords_used,              "stopwords_used.csv")
write_table(stopword_list_comparison,    "stopword_list_comparison.csv")
write_table(vocabulary_table,            "vocabulary.csv")
write_table(document_length_bands,       "document_length_bands.csv")
write_table(document_length_summary,     "document_length_summary.csv")
write_table(parameters_table,            "parameters.csv")
write_table(model_comparison_by_fit,     "model_comparison_by_fit.csv")
write_table(model_comparison_summary,    "model_comparison_summary.csv")
write_table(fit_traces,                  "fit_traces.csv")
write_table(stability_pairs,             "stability_pairs.csv")
write_table(stability_summary,           "stability_summary.csv")
write_table(seed1_topic_stability,       "topic_stability_seed1_all_k.csv")
write_table(top_words_probability,       "top_words_by_probability_all_models.csv")
write_table(top_words_relevance,         "top_words_by_relevance_all_models.csv")
write_table(topic_prevalence_all_models, "topic_prevalence_all_models.csv")
write_table(topic_pairs_word,            "topic_pairs_word_distributions_all_models.csv")
write_table(topic_pairs_document,        "topic_pairs_document_scores_all_models.csv")
write_table(representative_documents_all_models, "representative_documents_all_models.csv")
write_table(fit_runtime,                 "fit_runtime.csv")
if (!is.null(comparison_with_smart_run)) write_table(comparison_with_smart_run, "comparison_with_smart_run.csv")
for (tag in names(retained)) {
  r <- retained[[tag]]
  write_table(r$topic_summary,             sprintf("topic_summary_%s.csv", tag))
  write_table(r$topic_prevalence_by_type,  sprintf("topic_prevalence_by_type_%s.csv", tag))
  write_table(r$topic_prevalence_by_day,   sprintf("topic_prevalence_by_day_%s.csv", tag))
  write_table(r$topic_correlation,         sprintf("topic_correlation_%s.csv", tag))
  write_table(r$co_activation,             sprintf("topic_coactivation_%s.csv", tag))
  write_table(r$topic_word_cosine,         sprintf("topic_word_cosine_%s.csv", tag))
  write_table(r$concentration,             sprintf("document_concentration_bins_%s.csv", tag))
  write_table(r$concentration_summary,     sprintf("document_concentration_summary_%s.csv", tag))
  write_table(r$representative_documents,  sprintf("representative_documents_%s.csv", tag))
  for (s in names(r$stability_matrices)) write_table(r$stability_matrices[[s]], sprintf("stability_matrix_seed1_vs_seed%s_%s.csv", s, tag))
  write_table(r$topic_word_table,          sprintf("topic_word_distribution_%s.csv", tag))
  write_table(r$document_topic_table,      sprintf("document_topic_distribution_%s.csv", tag))
}
mark_stage("tables")

run_end <- Sys.time()
main_memory_end <- worker_memory()
stage_timings <- bind_rows(stage_log) |>
  mutate(start_utc = format(start_utc, "%Y-%m-%dT%H:%M:%OS3Z", tz = "UTC"), end_utc = format(end_utc, "%Y-%m-%dT%H:%M:%OS3Z", tz = "UTC"),
         seconds = round(seconds, 3))
# Stage groups comparable to the SMART run's ten stages.
stage_group <- c(read = "read", document_table = "tokenize", tokenize = "tokenize", vocabulary_dtm = "split", split = "split",
                 fit_cache_key = "fits", cluster_start = "fits", cluster_export = "fits", fits = "fits", fits_collect = "fits",
                 diagnostics_assembly = "diagnostics", stability = "comparison", comparison = "comparison",
                 representative_documents = "retained", accounting_tables = "retained", smart_comparison = "retained",
                 figures = "figures", validation = "validation", tables = "tables")
stage_timings <- stage_timings |> mutate(saved_run_stage = if_else(grepl("^retained_", stage), "retained", unname(stage_group[stage])))
grouped_seconds <- stage_timings |> group_by(saved_run_stage) |> summarise(seconds = sum(seconds), .groups = "drop")
run_info <- bind_rows(
  tibble(item = "run_start_utc", value = format(run_start, "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")),
  tibble(item = "run_end_utc",   value = format(run_end,   "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")),
  tibble(item = "total_seconds", value = as.character(round(as.numeric(difftime(run_end, run_start, units = "secs")), 1))),
  tibble(item = "seconds_before_run_start (R start-up and package loading)", value = as.character(round(startup_elapsed_seconds, 1))),
  grouped_seconds |> transmute(item = paste0("stage_seconds_", saved_run_stage), value = as.character(round(seconds, 1))),
  tibble(item = "fits_wall_seconds_sum_over_workers", value = as.character(round(sum(fit_runtime$elapsed_seconds[fit_runtime$role == "primary"]), 1))),
  tibble(item = "fits_wall_seconds_sum_over_workers_fit_to_completion (SMART-run scope)", value = as.character(round(sum(fit_runtime$elapsed_seconds_fit_to_completion[fit_runtime$role == "primary"]), 1))),
  tibble(item = "fits_cpu_user_seconds_sum_over_workers", value = as.character(round(sum(fit_runtime$cpu_user_seconds_total[fit_runtime$role == "primary"]), 1))),
  tibble(item = "fits_peak_working_set_mb_max_over_workers", value = as.character(round(max(fit_runtime$process_peak_working_set_mb, na.rm = TRUE)))),
  tibble(item = "fits_peak_working_set_mb_sum_over_first_jobs_of_each_worker (first wave)", value = as.character(round(sum(fit_runtime$process_peak_working_set_mb[fit_runtime$job_index_in_worker == 1L], na.rm = TRUE)))),
  tibble(item = "workers_requested (this run)", value = as.character(n_workers)),
  tibble(item = "workers_used (run that fitted the models)", value = as.character(fit_meta$n_workers_used)),
  tibble(item = "fitting_run_id", value = as.character(fit_meta$fitting_run_id)),
  tibble(item = "this_run_id", value = run_id),
  tibble(item = "jobs_total (incl. determinism job)", value = as.character(nrow(fit_runtime))),
  tibble(item = "jobs_fitted_in_this_run", value = as.character(length(jobs_to_run))),
  tibble(item = "jobs_loaded_from_models_cache", value = as.character(sum(fit_runtime$from_cache))),
  tibble(item = "determinism_check_run", value = as.character(determinism_check)),
  tibble(item = "main_process_peak_working_set_mb", value = as.character(round(main_memory_end[["process_peak_working_set_mb"]]))),
  tibble(item = "main_process_peak_working_set_mb_after_fits", value = as.character(round(main_memory_after_fits[["process_peak_working_set_mb"]]))),
  tibble(item = "main_process_r_max_used_mb", value = as.character(round(main_memory_end[["r_max_used_mb"]]))),
  tibble(item = "models_dir_bytes", value = as.character(sum(file.size(list.files(model_dir, pattern = "\\.rds$", full.names = TRUE))))),
  tibble(item = "cpu_logical_processors", value = as.character(detectCores(logical = TRUE))),
  tibble(item = "cpu_physical_cores", value = as.character(detectCores(logical = FALSE))),
  tibble(item = "R_version", value = R.version.string),
  tibble(item = "text2vec_version", value = as.character(packageVersion("text2vec"))),
  tibble(item = "quick_run", value = as.character(quick_run)))
if (exists("comparison_smart_stage_rows")) {
  stage_cmp <- comparison_smart_stage_rows |> left_join(run_info |> select(item, tier2_run = value), by = "item") |>
    mutate(note = "wall-clock seconds of the main process; Tier 2 stages grouped to the SMART run's ten stages (see stage_timings.csv)")
  comparison_with_smart_run <- bind_rows(comparison_with_smart_run, stage_cmp)
  write_table(comparison_with_smart_run, "comparison_with_smart_run.csv")
}
write_table(stage_timings, "stage_timings.csv")
write_table(run_info, "run_info.csv")
# Every run of this script into this folder is appended to models/run_history.csv (kept across cache clears, keyed by
# run_key), so the timings of the run that fitted the models survive later cached re-runs that regenerate the tables.
history_file <- file.path(model_dir, "run_history.csv")
run_history <- bind_rows(
  if (file.exists(history_file)) read_csv(history_file, show_col_types = FALSE, progress = FALSE, col_types = cols(.default = col_character())) else NULL,
  run_info |> transmute(run_id = run_id, run_key = run_key, item, value),
  stage_timings |> transmute(run_id = run_id, run_key = run_key, item = paste0("stage_seconds_detail_", stage), value = as.character(seconds)))
write_csv(run_history, history_file)
write_table(run_history, "run_history.csv")

# ---- Write provenance sidecars ----
# One <output>.manifest.yml per output: input/output SHA256, script hash, git commit, parameters, seed and
# package versions (lite manifests for the RDS files).
git_commit <- tryCatch(system2("git", c("-C", project_dir, "rev-parse", "--short", "HEAD"), stdout = TRUE, stderr = FALSE),
                       error = function(e) "")
manifest_packages <- c("nanoparquet", "dplyr", "tidyr", "stringi", "Matrix", "text2vec", "stopwords", "clue", "readxl", "ps",
                       "ggplot2", "scales", "readr", "digest", "yaml")
package_versions  <- lapply(manifest_packages, function(p) list(name = p, version = as.character(packageVersion(p))))
input_files <- c(lapply(c(comment_files, submission_files), function(p) list(path = p, hash = sha256(p), format = "parquet")),
                 list(list(path = stopword_workbook, hash = workbook_hash, format = "xlsx", rows = nrow(tier2_table), cols = ncol(tier2_raw), sheet = stopword_sheet)),
                 if (smart_available) lapply(unname(smart_paths), function(p) list(path = p, hash = sha256(p), format = "csv", role = "earlier SMART run, read-only, comparison and reproduction check")) else list())
manifest_parameters <- c(
  list(reader = "nanoparquet",
       documents = "one per comment (body) and one per submission (title + ' ' + selftext); doc_key = Reddit fullname",
       text_processing = "lower case; U+2019 -> '; URLs removed; tokens = Unicode letter runs with internal apostrophes; trailing 's stripped; tokens < 2 chars removed; the 310 Tier 2 workbook entries removed; removal-marker-only documents excluded; terms kept when 5 <= df <= 50% of non-marker documents; no stemming",
       stopword_condition = sprintf("Tier 2 cumulative primary raw named-list consensus (n_lists >= 20), 310 entries from sheet %s of the workbook hashed above; replaces the SMART list of the original run", stopword_sheet),
       model = "LDA, WarpLDA collapsed Gibbs sampling (text2vec), symmetric priors alpha = alpha_sum / K and beta",
       heldout = "document-completion perplexity on a seeded 5% held-out sample (half A inferred, half B scored); the whole held-out documents are also transformed with the fixed topics for the document-topic tables",
       topic_numbering = "descending token-weighted prevalence within each fit",
       retention = "lowest mean rank over held-out perplexity, NPMI coherence, exclusivity, matched-topic cosine; plus the best NPMI coherence K and the one-standard-error held-out perplexity K (smallest K within one seed SD of the minimum) if different; identifies models for the full document-level tables, not a final choice of K",
       fits = "every fit's theta over all modeled documents and its topic-word counts are kept in models/ (theta_K###_seed#.rds, fit_K###_seed#.rds); retained models are not re-fitted",
       topic_pair_quantities = "word-based: cosine of smoothed topic-word distributions, Jaccard of top-20 words; document-based: Pearson correlation of document shares, co-activation share at 0.1; cross-seed: cosine of topic-word distributions with Hungarian matching"),
  setNames(as.list(parameters_table$value), parameters_table$parameter))

write_sidecar <- function(output) {
  ext <- tolower(tools::file_ext(output))
  manifest <- list(manifest_version = 1L, output_file = output, output_hash = sha256(output),
                   output_format = switch(ext, csv = "csv", png = "image/png", rds = "rds", ext))
  if (ext == "csv") {
    dims <- table_dims[[output]]
    if (is.null(dims)) { tbl <- read_csv(output, show_col_types = FALSE, guess_max = 1000); dims <- c(rows = nrow(tbl), cols = ncol(tbl)) }
    manifest$output_rows <- unname(dims[["rows"]])
    manifest$output_cols <- unname(dims[["cols"]])
  }
  manifest$input_files    <- input_files
  manifest$transformation <- list(script = script_path, script_hash = sha256(script_path),
                                  parameters = manifest_parameters, git_commit = git_commit)
  manifest$software <- list(language = "R", language_version = paste(R.version$major, R.version$minor, sep = "."),
                            packages = package_versions, os = paste(Sys.info()[["sysname"]], Sys.info()[["release"]]))
  manifest$seed      <- fit_seeds[1]
  manifest$timestamp <- format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")
  manifest$notes     <- if (ext == "rds") "Model object of one Phase 1A LDA fit (Tier 2 stopword condition): theta_*.rds holds the document-topic matrix over all modeled documents (rows = doc_key in modeled order: training then held-out; columns = topics numbered by prevalence) with the sampler index; fit_*.rds holds the topic-word counts, traces, diagnostics, per-document summaries, K x K matrices, representative-document indices, timing and memory. Source Parquet read-only." else
    "Phase 1A LDA topic models of the July 2026 r/politics sample under the Tier 2 (310-word) stopword condition; source Parquet and the Tier 2 workbook read-only; fits use seeds 1-3 (primary seed 1) and the split seed recorded in parameters."
  write_yaml(manifest, paste0(output, ".manifest.yml"))
}
invisible(lapply(list.files(tab_dir,   pattern = "\\.csv$", full.names = TRUE), write_sidecar))
invisible(lapply(list.files(fig_dir,   pattern = "\\.png$", full.names = TRUE), write_sidecar))
invisible(lapply(list.files(model_dir, pattern = "^(theta|fit)_.*\\.rds$", full.names = TRUE), write_sidecar))
mark_stage("manifests")

# ---- Console summary ----
print(as.data.frame(document_accounting))
print(as.data.frame(stopword_accounting[1:40, ]), right = FALSE)
print(as.data.frame(model_comparison_summary[, c("K", "iterations_fit_mean", "heldout_perplexity_mean", "coherence_npmi_mean",
                                                 "exclusivity_mean", "matched_cosine_mean", "ari_dominant_topics_mean",
                                                 "seed1_share_topics_reproduced_in_both_other_seeds", "composite_rank", "retained")]))
for (tag in names(retained)) {
  cat("\n==", tag, "==\n")
  print(as.data.frame(retained[[tag]]$topic_summary[, c("topic", "token_share", "doc_mean_share", "dominant_share", "coherence_npmi", "top_10_words_by_probability")]))
}
print(as.data.frame(run_info))
print(as.data.frame(fit_runtime[, c("K", "seed", "role", "pid", "iterations_fit", "elapsed_seconds", "seconds_fit_transform", "cpu_user_seconds_total",
                                    "process_peak_working_set_mb", "concurrent_fits_at_start")]))
cat("\nDone. Figures ->", fig_dir, "\nTables ->", tab_dir, "\nModels ->", model_dir, "\n")
