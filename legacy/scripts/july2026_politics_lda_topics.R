# July 2026 r/politics sample — Phase 1A: Latent Dirichlet Allocation topic models
#
# Standalone side analysis. Reads the raw Parquet comment and submission files
# (read-only), builds one document per comment and per submission, applies an
# explicit text-processing pipeline in which every step is counted, fits Latent
# Dirichlet Allocation (collapsed Gibbs sampling, WarpLDA as implemented in
# text2vec) at several candidate topic counts with three random seeds each,
# compares the candidates with quantitative diagnostics, measures the stability
# of the topic solutions across seeds, retains models by an explicit rule, and
# writes the topic-word and document-topic distributions with source identifiers
# and metadata preserved. Topics are identified only by number. All outputs go
# under r_analysis_outputs/lda_topics/ and are reproducible by running this
# script top to bottom.

# ---- Setup: packages, paths, parameters, output directories ----
# INPUT : none.
# DOES  : load packages; define source paths and every analytical parameter;
#         create output folders. Setting LDA_QUICK=<folder> in the environment
#         runs a reduced smoke test into that folder (never into the project
#         outputs); the default run uses the parameters as written.
# OUTPUT: parameter objects; figures/ and tables/ directories.
suppressPackageStartupMessages({
  library(nanoparquet); library(dplyr); library(tidyr); library(stringi); library(Matrix)
  library(text2vec); library(stopwords); library(clue); library(parallel)
  library(ggplot2); library(scales); library(readr); library(digest); library(yaml)
})
lgr::get_logger("text2vec")$set_threshold("warn")
run_start   <- Sys.time()
stage_times <- list()
mark_stage  <- function(name) stage_times[[name]] <<- Sys.time()

project_dir     <- Sys.getenv("KC_REPO_ROOT", unset = getwd())
comments_dir    <- file.path(project_dir, "data_sample/comments/2026-07")
submissions_dir <- file.path(project_dir, "data_sample/submissions/2026-07")
script_path     <- file.path(project_dir, "july2026_politics_lda_topics.R")
quick_run       <- nzchar(Sys.getenv("LDA_QUICK"))
out_dir <- if (quick_run) file.path(Sys.getenv("LDA_QUICK"), "lda_topics_quick") else
  file.path(project_dir, "r_analysis_outputs/lda_topics")
fig_dir <- file.path(out_dir, "figures")
tab_dir <- file.path(out_dir, "tables")
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(tab_dir, recursive = TRUE, showWarnings = FALSE)
# Outputs are fully regenerated on every run; remove those of a previous run so no stale file survives.
invisible(file.remove(list.files(c(fig_dir, tab_dir), pattern = "\\.(csv|png|yml)$", full.names = TRUE)))

# Analytical parameters. Every value is used below and recorded in the manifests.
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
stopword_source            <- "smart"         # stopwords::stopwords("en", source = "smart"), 571 terms
n_top_words                <- 20L
coherence_top_n            <- 10L
relevance_lambda           <- 0.6             # Sievert & Shirley (2014) relevance weight for the second word ranking
n_representative           <- 5L
representative_min_tokens  <- 25L
active_topic_threshold     <- 0.1             # a topic is "active" in a document when its share is at least this
stability_cosine_threshold <- 0.8
n_workers                  <- 8L
if (quick_run) { candidate_k <- c(5L, 8L); fit_seeds <- c(1L, 2L); n_iter_max <- 30L; n_workers <- 2L }

comment_files    <- list.files(comments_dir,    pattern = "\\.parquet$", full.names = TRUE)
submission_files <- list.files(submissions_dir, pattern = "\\.parquet$", full.names = TRUE)

# ---- Read source data ----
# INPUT : 7 comment Parquet files + 1 submission Parquet file (read-only).
# DOES  : read each file and stack; keep the columns used below (column
#         selection only; no rows dropped); cast created_utc to double.
# OUTPUT: comments (one row per comment); submissions (one row per submission).
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

# ---- Document table ----
# INPUT : comments; submissions.
# DOES  : one document per source record. Comment text = body exactly as
#         stored; submission text = title, one space, selftext exactly as
#         stored. doc_key is the Reddit fullname ('t1_' + id for comments,
#         't3_' + id for submissions) so the two id spaces cannot collide.
#         submission_id is the thread: link_id without 't3_' for comments, the
#         own id for submissions. parent_id is kept for comments (NA for
#         submissions). Nothing is dropped.
# OUTPUT: docs (one row per source record, source order: comments then submissions).
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
# INPUT : comments (body, author); submissions (title, selftext, author).
# DOES  : classify every value of each text field as missing, empty, one of the
#         deletion/removal markers, or other text; count. Nothing is altered.
# OUTPUT: text_availability (one row per field x state).
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

# ---- Text processing, step by step ----
# INPUT : docs$text.
# DOES  : the transformations that turn stored text into model terms, applied
#         in this order and counted at every step:
#         (1) case folding (Unicode lower case);
#         (2) the curly apostrophe U+2019 becomes ';
#         (3) URLs (http://, https://, www. up to the next whitespace) removed;
#         (4) tokens = maximal runs of Unicode letters, optionally joined by
#             internal apostrophes (don't, o'brien); digits, punctuation and
#             symbols never form tokens;
#         (5) a trailing possessive 's is stripped (trump's -> trump);
#         (6) tokens shorter than min_token_chars are removed;
#         (7) tokens in the SMART English stopword list are removed;
#         (8) documents whose whole text is a removal marker ([deleted],
#             [removed], [ Removed by Reddit ]) are excluded from modelling
#             together with their tokens (they carry no authored text).
#         No stemming or lemmatisation. Reference counts describe what the
#         untaken alternatives would have done; they change nothing.
# OUTPUT: tok / tdoc (candidate token stream with document index); per-document
#         token counts on docs; the counts used by text_processing_report.
removal_markers  <- c("[deleted]", "[removed]", "[ Removed by Reddit ]")
docs$marker_only <- stri_trim_both(docs$text) %in% removal_markers
url_regex   <- "(https?://|www\\.)\\S+"
token_regex <- "\\p{L}+(?:'\\p{L}+)*"
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

is_short <- stri_length(tok) < min_token_chars                                # step 6
stopword_list <- stopwords("en", source = stopword_source)                    # step 7
is_stop  <- !is_short & tok %in% stopword_list
n_stop_snowball_reference <- sum(!is_short & tok %in% stopwords("en", source = "snowball"))
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

# ---- Vocabulary and document-term matrix ----
# INPUT : tok / tdoc (candidate tokens of non-marker documents).
# DOES  : vocabulary before pruning = distinct candidate terms (byte order);
#         df(term) = non-marker documents containing the term. Terms are kept
#         when min_document_frequency <= df <= max_document_share x non-marker
#         documents. The document-term matrix has one row per source document
#         (rows of excluded documents are empty), one column per kept term,
#         cell = count of the term in the document. Documents with no kept
#         term cannot be represented and are labelled, not dropped silently.
# OUTPUT: vocabulary; dtm_all; vocabulary_table; docs$model_status.
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
                                              TRUE ~ "kept")) |>
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

# ---- Training / held-out split and document-completion halves ----
# INPUT : dtm_all; docs$model_status; docs$n_tokens_modeled.
# DOES  : modeled documents with at least 2 modeled tokens are eligible for the
#         held-out set; floor(heldout_share x modeled documents) of them are
#         sampled with split_seed. Models are fitted on the training documents
#         only. For document-completion perplexity every held-out document's
#         token occurrences are put in a seeded random order and alternated
#         into half A (used to infer the document's topic mixture) and half B
#         (scored), so both halves are non-empty.
# OUTPUT: train_rows / heldout_rows; dtm_train; dtm_heldout; dtm_heldout_a;
#         dtm_heldout_b; docs$split.
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
mark_stage("split")

# ---- Model fitting (one worker process per fit) ----
# INPUT : dtm_train; dtm_heldout_a / dtm_heldout_b; parameters.
# DOES  : for every candidate K and seed: set.seed(seed); fit LDA with
#         doc_topic_prior = alpha_sum / K and topic_word_prior = beta by
#         WarpLDA collapsed Gibbs sampling for at most n_iter_max iterations,
#         stopping early when 10 iterations improve the pseudo log-likelihood
#         by less than convergence_tol (relative). The document-topic matrix
#         returned by text2vec is the mean over n_iter_inference post-burn-in
#         samples of the share of the document's tokens assigned to each topic
#         (no prior mass added). The topic-word distribution is the smoothed
#         posterior mean phi = (count + beta) / (topic tokens + V x beta).
#         Perplexity uses the smoothed document mixture
#         (tokens x theta + alpha) / (tokens + K x alpha). Held-out
#         perplexity is document completion: half A of each held-out document
#         is transformed into the fixed topic space, half B is scored.
#         Each worker returns the topic-word counts, prevalence vectors, the
#         per-document dominant topic and its share, the likelihood trace and
#         the perplexities; the full training theta stays in the worker.
# OUTPUT: fit_results (one list per fit, in job order).
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

fit_one_model <- function(K, seed) {
  t_start <- Sys.time()
  alpha   <- alpha_sum / K
  set.seed(seed)
  model <- LDA$new(n_topics = K, doc_topic_prior = alpha, topic_word_prior = beta,
                   n_iter_inference = n_iter_inference)
  theta <- model$fit_transform(dtm_train, n_iter = n_iter_max, convergence_tol = convergence_tol,
                               n_check_convergence = n_check_convergence, progressbar = FALSE)
  trace      <- attr(theta, "likelihood")
  components <- model$components
  phi        <- (components + beta) / (rowSums(components) + ncol(components) * beta)
  train_fit  <- log_likelihood(dtm_train, phi, smooth_theta(theta, train_doc_tokens, alpha))
  theta_a    <- model$transform(dtm_heldout_a, n_iter = n_iter_max, convergence_tol = convergence_tol,
                                n_check_convergence = n_check_convergence, progressbar = FALSE)
  heldout_fit <- log_likelihood(dtm_heldout_b, phi,
                                smooth_theta(theta_a, as.numeric(rowSums(dtm_heldout_a)), alpha))
  # Topic numbering: descending token-weighted prevalence (ties by sampler index). Applied here,
  # before any per-document label is derived, so that ties in a document's shares are broken in
  # the same column order everywhere.
  token_share <- as.numeric(crossprod(theta, train_doc_tokens)) / sum(train_doc_tokens)
  ord         <- order(-token_share, seq_along(token_share))
  theta       <- theta[, ord, drop = FALSE]
  components  <- components[ord, , drop = FALSE]
  dominant    <- max.col(theta, ties.method = "first")
  list(K = K, seed = seed,
       iterations          = max(trace$iter),
       trace               = trace,
       final_pseudo_loglik = trace$loglikelihood[nrow(trace)],
       sampler_index       = ord,
       components          = components,
       token_share         = token_share[ord],
       doc_mean_share      = colMeans(theta),
       dominant            = dominant,
       max_theta           = theta[cbind(seq_len(nrow(theta)), dominant)],
       train_loglik        = train_fit[["loglik"]],
       train_tokens        = train_fit[["tokens"]],
       train_perplexity    = train_fit[["perplexity"]],
       heldout_loglik      = heldout_fit[["loglik"]],
       heldout_tokens      = heldout_fit[["tokens"]],
       heldout_perplexity  = heldout_fit[["perplexity"]],
       heldout_iterations  = max(attr(theta_a, "likelihood")$iter),
       elapsed_seconds     = as.numeric(difftime(Sys.time(), t_start, units = "secs")))
}

jobs <- expand.grid(seed = fit_seeds, K = candidate_k) |> arrange(desc(K), seed)
job_list <- lapply(seq_len(nrow(jobs)), function(i) list(K = jobs$K[i], seed = jobs$seed[i]))

# Optional development aid: LDA_FIT_CACHE=<directory> caches the worker results under a
# content hash of the matrices and every fitting parameter, so an unchanged re-run can
# skip the fits. Unset (the default) means the fits always run. Fits are deterministic
# (checked below), so a cache hit returns exactly what the run would have produced.
fit_cache_dir  <- Sys.getenv("LDA_FIT_CACHE")
fit_cache_file <- if (nzchar(fit_cache_dir)) {
  dir.create(fit_cache_dir, recursive = TRUE, showWarnings = FALSE)
  file.path(fit_cache_dir, paste0("lda_fits_", digest(list(
    dtm_train, dtm_heldout_a, dtm_heldout_b, train_doc_tokens, candidate_k, fit_seeds, alpha_sum, beta,
    n_iter_max, convergence_tol, n_check_convergence, n_iter_inference, as.character(packageVersion("text2vec")))), ".rds"))
} else ""
fits_from_cache <- nzchar(fit_cache_file) && file.exists(fit_cache_file)
if (fits_from_cache) {
  fit_results <- readRDS(fit_cache_file)
} else {
  cl <- makeCluster(min(n_workers, length(job_list)))
  invisible(clusterEvalQ(cl, {
    suppressPackageStartupMessages({library(text2vec); library(Matrix)})
    lgr::get_logger("text2vec")$set_threshold("warn")
    NULL
  }))
  clusterExport(cl, c("dtm_train", "dtm_heldout_a", "dtm_heldout_b", "train_doc_tokens",
                      "fit_one_model", "smooth_theta", "log_likelihood",
                      "alpha_sum", "beta", "n_iter_max", "convergence_tol", "n_check_convergence",
                      "n_iter_inference"))
  fit_results <- parLapplyLB(cl, job_list, function(job) fit_one_model(job$K, job$seed), chunk.size = 1L)
  stopCluster(cl)
  if (nzchar(fit_cache_file)) saveRDS(fit_results, fit_cache_file)
}
mark_stage("fits")

# ---- Per-fit derived quantities ----
# INPUT : fit_results; dtm_train.
# DOES  : every fit arrives with its topics numbered in descending order of
#         token-weighted prevalence (Topic 1 = largest share of training
#         tokens; ties by sampler index; the sampler's original index is kept).
#         Smoothed phi; word rankings by probability and by relevance
#         (lambda log phi + (1 - lambda) log(phi / p_w), p_w = training corpus
#         share of the word); coherence over the top coherence_top_n words:
#         UMass (Mimno et al. 2011, per pair, using training-document
#         co-occurrence) and NPMI (Lau et al. 2014, per pair, document
#         co-occurrence, -1 for never co-occurring pairs); exclusivity = mean
#         over the top words of phi_kw / sum_k' phi_k'w; mean pairwise cosine
#         between topic-word vectors (Cao et al. 2009), mean pairwise symmetric
#         Kullback-Leibler divergence (Deveaud et al. 2014), symmetric KL
#         between the normalised singular values of phi and the token-weighted
#         topic shares (Arun et al. 2010).
# OUTPUT: fits (named list "K_seed"); term_frequency_share.
finalize_fit <- function(fit) {
  fit$phi <- (fit$components + beta) / (rowSums(fit$components) + ncol(fit$components) * beta)
  fit
}
fit_key <- function(K, seed) paste(K, seed, sep = "_")
fits <- lapply(fit_results, finalize_fit)
names(fits) <- vapply(fits, function(f) fit_key(f$K, f$seed), "")
rm(fit_results)

term_frequency_share <- colSums(dtm_train) / n_train_tokens
n_terms_absent_from_train <- sum(term_frequency_share == 0)

top_words_for <- function(phi, n, lambda) {
  score <- if (lambda == 1) log(phi) else {
    p_w <- ifelse(term_frequency_share > 0, term_frequency_share, NA_real_)
    lambda * log(phi) + (1 - lambda) * log(sweep(phi, 2, p_w, "/"))
  }
  score[is.na(score)] <- -Inf
  t(apply(score, 1, function(s) order(-s)[seq_len(n)]))
}

coherence_for <- function(fit, n = coherence_top_n) {
  idx   <- top_words_for(fit$phi, n, 1)
  words <- sort(unique(as.vector(idx)))
  B <- dtm_train[, words, drop = FALSE]
  B@x[] <- 1
  co  <- unname(as.matrix(crossprod(B)))
  dfw <- unname(diag(co))
  pos <- matrix(match(as.vector(idx), words), nrow = nrow(idx))
  n_d <- nrow(dtm_train)
  t(vapply(seq_len(fit$K), function(k) {
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

exclusivity_for <- function(fit, n = coherence_top_n) {
  idx     <- top_words_for(fit$phi, n, 1)
  col_tot <- colSums(fit$phi)
  vapply(seq_len(fit$K), function(k) mean(fit$phi[k, idx[k, ]] / col_tot[idx[k, ]]), numeric(1))
}

pairwise_metrics_for <- function(fit) {
  phi <- fit$phi; K <- fit$K
  cos_mat   <- tcrossprod(phi / sqrt(rowSums(phi^2)))
  log_phi   <- log(phi)
  sym_kl    <- matrix(0, K, K)
  for (i in seq_len(K)) for (j in seq_len(K)) if (i < j)
    sym_kl[i, j] <- 0.5 * sum(phi[i, ] * (log_phi[i, ] - log_phi[j, ])) +
                    0.5 * sum(phi[j, ] * (log_phi[j, ] - log_phi[i, ]))
  cm1 <- svd(phi, nu = 0, nv = 0)$d; cm1 <- cm1 / sum(cm1)
  cm2 <- fit$token_share / sum(fit$token_share)
  c(cao_juan_2009 = mean(cos_mat[upper.tri(cos_mat)]),
    deveaud_2014  = mean(sym_kl[upper.tri(sym_kl)]),
    arun_2010     = sum(cm1 * log(cm1 / cm2)) + sum(cm2 * log(cm2 / cm1)))
}

for (nm in names(fits)) {
  f   <- fits[[nm]]
  coh <- coherence_for(f)
  fits[[nm]]$topic_umass       <- coh[, "umass"]
  fits[[nm]]$topic_npmi        <- coh[, "npmi"]
  fits[[nm]]$topic_exclusivity <- exclusivity_for(f)
  fits[[nm]]$pairwise          <- pairwise_metrics_for(f)
  fits[[nm]]$top_probability   <- top_words_for(f$phi, n_top_words, 1)
  fits[[nm]]$top_relevance     <- top_words_for(f$phi, n_top_words, relevance_lambda)
}

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
topic_prevalence_all_models <- bind_rows(lapply(fits, function(f) tibble(
  K = f$K, seed = f$seed, topic = seq_len(f$K), sampler_index = f$sampler_index,
  token_share = f$token_share, doc_mean_share = f$doc_mean_share,
  dominant_docs = tabulate(f$dominant, nbins = f$K), dominant_share = tabulate(f$dominant, nbins = f$K) / n_train,
  coherence_umass = f$topic_umass, coherence_npmi = f$topic_npmi, exclusivity = f$topic_exclusivity,
  top_10_words_by_probability = paste_top(f, f$top_probability),
  top_10_words_by_relevance = paste_top(f, f$top_relevance))))

fit_traces <- bind_rows(lapply(fits, function(f) tibble(K = f$K, seed = f$seed, iteration = f$trace$iter,
                                                        pseudo_loglikelihood = f$trace$loglikelihood,
                                                        pseudo_loglikelihood_per_token = f$trace$loglikelihood / n_train_tokens)))
mark_stage("diagnostics")

# ---- Stability across seeds ----
# INPUT : fits.
# DOES  : for every K and pair of seeds, topics of the two fits are matched
#         one-to-one by the Hungarian assignment (clue::solve_LSAP) that
#         maximises the total cosine similarity between topic-word vectors.
#         Per matched pair: cosine similarity and the Jaccard overlap of the
#         top n_top_words words. Per seed pair: adjusted Rand index between the
#         dominant-topic labellings of the training documents (label
#         permutation invariant, so no matching is needed).
# OUTPUT: stability_pairs (one row per matched topic pair); stability_summary
#         (one row per K); stability_cos (cosine matrices for seeds 1 vs 2).
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
  if (pair[1] == fit_seeds[1] && pair[2] == fit_seeds[2]) stability_cos[[as.character(K)]] <- list(cos = cos_mat, assignment = assignment)
  topics      <- seq_len(K)
  matched_cos <- cos_mat[cbind(topics, assignment)]
  matched_jac <- vapply(topics, function(k) jaccard(f1$top_probability[k, ], f2$top_probability[assignment[k], ]), numeric(1))
  stability_pairs[[length(stability_pairs) + 1]] <- tibble(
    K = K, seed_a = pair[1], seed_b = pair[2], topic_a = topics, topic_b = assignment,
    cosine = matched_cos, jaccard_top_words = matched_jac,
    token_share_a = f1$token_share, token_share_b = f2$token_share[assignment],
    ari_dominant_topics = adjusted_rand_index(f1$dominant, f2$dominant))
}
stability_pairs <- bind_rows(stability_pairs)
stability_summary <- stability_pairs |>
  group_by(K) |>
  summarise(seed_pairs                 = n_distinct(paste(seed_a, seed_b)),
            matched_cosine_mean        = mean(cosine),
            matched_cosine_median      = median(cosine),
            matched_cosine_min         = min(cosine),
            share_matched_cosine_at_least_threshold = mean(cosine >= stability_cosine_threshold),
            jaccard_top_words_mean     = mean(jaccard_top_words),
            ari_dominant_topics_mean   = mean(ari_dominant_topics[!duplicated(paste(seed_a, seed_b))]),
            .groups = "drop")

# ---- Model comparison and retention rule ----
# INPUT : fits; stability_summary.
# DOES  : one row per fit with its diagnostics; one row per K with the mean
#         and standard deviation over seeds plus the stability measures. Ranks
#         across K (1 = best): held-out perplexity ascending, NPMI coherence
#         descending, exclusivity descending, matched-topic cosine descending;
#         composite = mean of the four ranks. Retained for full inspection: the
#         K with the lowest composite rank (ties: smaller K), the K with the
#         highest NPMI coherence, and the perplexity choice under the
#         one-standard-error rule: the smallest K whose mean held-out perplexity
#         is within one seed standard deviation (at the best K) of the minimum
#         (duplicates collapse to one model). The plain minimum is recorded too.
# OUTPUT: model_comparison_by_fit; model_comparison_summary; retained_k.
model_comparison_by_fit <- bind_rows(lapply(fits, function(f) tibble(
  K = f$K, seed = f$seed, iterations_fit = f$iterations, iterations_heldout_inference = f$heldout_iterations,
  final_pseudo_loglikelihood = f$final_pseudo_loglik,
  final_pseudo_loglikelihood_per_token = f$final_pseudo_loglik / n_train_tokens,
  train_perplexity = f$train_perplexity, train_tokens = f$train_tokens,
  heldout_perplexity = f$heldout_perplexity, heldout_tokens_scored = f$heldout_tokens,
  coherence_npmi = mean(f$topic_npmi), coherence_npmi_median = median(f$topic_npmi), coherence_npmi_min = min(f$topic_npmi),
  coherence_umass = mean(f$topic_umass), exclusivity = mean(f$topic_exclusivity),
  cao_juan_2009 = f$pairwise[["cao_juan_2009"]], deveaud_2014 = f$pairwise[["deveaud_2014"]], arun_2010 = f$pairwise[["arun_2010"]],
  largest_topic_share = max(f$token_share), smallest_topic_share = min(f$token_share),
  elapsed_seconds = f$elapsed_seconds))) |>
  arrange(K, seed)

model_comparison_summary <- model_comparison_by_fit |>
  group_by(K) |>
  summarise(fits = n(),
            across(c(iterations_fit, train_perplexity, heldout_perplexity, coherence_npmi, coherence_umass,
                     exclusivity, cao_juan_2009, deveaud_2014, arun_2010, largest_topic_share, elapsed_seconds),
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

# ---- Retained models: full document-topic distributions and inspection tables ----
# INPUT : retained_k; fits; dtm_train; dtm_heldout; docs.
# DOES  : for each retained K the seed-1 model is fitted again in this process
#         from the same seed; its topic-word counts must equal the worker's
#         (determinism check). Its document-topic matrix covers the training
#         documents; held-out documents are transformed with the fixed topics.
#         Then, over all modeled documents: token-weighted topic shares, mean
#         document shares, dominant-topic counts, shares by document type and
#         by UTC day, topic-topic correlation of document shares, document
#         concentration (largest share, active topics), author concentration
#         (the most frequent stored author string among the documents whose
#         dominant topic is k, and its share of them), representative
#         documents (highest topic share among documents with at least
#         representative_min_tokens modeled tokens, one document per distinct
#         text and per distinct author, ties by doc_key), and the full
#         document-topic table for every source document (NA for documents
#         that are not modeled).
# OUTPUT: retained (list per K of tables); doc-topic matrices.
band_breaks <- c(0, 1, 4, 9, 24, 49, 99, Inf)
band_labels <- c("1", "2-4", "5-9", "10-24", "25-49", "50-99", "100+")
retained <- list()
for (K in retained_k) {
  tag   <- sprintf("K%03d", K)
  fit   <- fits[[fit_key(K, fit_seeds[1])]]
  alpha <- alpha_sum / K
  set.seed(fit_seeds[1])
  model <- LDA$new(n_topics = K, doc_topic_prior = alpha, topic_word_prior = beta, n_iter_inference = n_iter_inference)
  theta_train <- model$fit_transform(dtm_train, n_iter = n_iter_max, convergence_tol = convergence_tol,
                                     n_check_convergence = n_check_convergence, progressbar = FALSE)
  refit_components_identical <- identical(unname(model$components[fit$sampler_index, , drop = FALSE]), unname(fit$components))
  theta_heldout <- model$transform(dtm_heldout, n_iter = n_iter_max, convergence_tol = convergence_tol,
                                   n_check_convergence = n_check_convergence, progressbar = FALSE)
  theta_train   <- theta_train[, fit$sampler_index, drop = FALSE]
  theta_heldout <- theta_heldout[, fit$sampler_index, drop = FALSE]
  refit_dominant_identical <- identical(max.col(theta_train, ties.method = "first"), fit$dominant)
  rm(model)

  theta_full <- matrix(NA_real_, n_docs, K)
  theta_full[train_rows, ]   <- theta_train
  theta_full[heldout_rows, ] <- theta_heldout
  rm(theta_train, theta_heldout)
  theta_m   <- theta_full[modeled_rows, , drop = FALSE]
  n_tok_m   <- as.numeric(docs$n_tokens_modeled[modeled_rows])
  type_m    <- docs$doc_type[modeled_rows]
  day_m     <- as.character(docs$created_date_utc[modeled_rows])
  dominant  <- max.col(theta_m, ties.method = "first")
  max_share <- theta_m[cbind(seq_len(n_modeled), dominant)]
  n_active  <- as.integer(rowSums(theta_m >= active_topic_threshold))
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
  top_author <- tibble(topic = dominant, author = docs$author[modeled_rows]) |>
    count(topic, author, name = "docs") |>
    arrange(topic, desc(docs), author) |>
    distinct(topic, .keep_all = TRUE) |>
    transmute(topic, top_author = author, top_author_dominant_docs = docs)
  topic_summary <- tibble(topic = seq_len(K), topic_label = topic_labels, sampler_index = fit$sampler_index,
                          token_share_train = fit$token_share, token_share = token_share, doc_mean_share = colMeans(theta_m),
                          dominant_docs = tabulate(dominant, nbins = K), dominant_share = tabulate(dominant, nbins = K) / n_modeled,
                          token_share_comments = by_type["comment", ], token_share_submissions = by_type["submission", ],
                          coherence_umass = fit$topic_umass, coherence_npmi = fit$topic_npmi, exclusivity = fit$topic_exclusivity) |>
    left_join(top_author, by = "topic") |>
    mutate(top_author_share_of_dominant_docs = top_author_dominant_docs / dominant_docs) |>
    left_join(stab_wide, by = c("topic" = "topic_a")) |>
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
  topic_correlation <- as_tibble(topic_cor, .name_repair = ~ topic_cols) |> mutate(topic = seq_len(K), .before = 1)

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

  cand <- modeled_rows[docs$n_tokens_modeled[modeled_rows] >= representative_min_tokens]
  representative_documents <- bind_rows(lapply(seq_len(K), function(k) {
    o <- cand[order(-theta_full[cand, k], docs$doc_key[cand])]
    o <- o[!duplicated(docs$text[o]) & !duplicated(docs$author[o])]
    r <- head(o, n_representative)
    tibble(topic = k, topic_label = topic_labels[k], rank = seq_along(r), topic_share_in_document = theta_full[r, k],
           doc_key = docs$doc_key[r], doc_id = docs$doc_id[r], doc_type = docs$doc_type[r], author = docs$author[r],
           created_utc = docs$created_utc[r], submission_id = docs$submission_id[r], link_id = docs$link_id[r],
           parent_id = docs$parent_id[r], split = docs$split[r], n_tokens_modeled = docs$n_tokens_modeled[r],
           dominant_topic = dominant_full[r], dominant_topic_share = max_share_full[r], text = docs$text[r])
  }))

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

  sc <- stability_cos[[as.character(K)]]
  stability_matrix <- tibble(seed1_topic = rep(seq_len(K), times = K), seed2_topic = rep(seq_len(K), each = K),
                             cosine = as.vector(sc$cos), matched = rep(sc$assignment, times = K) == rep(seq_len(K), each = K))

  retained[[tag]] <- list(K = K, tag = tag, fit = fit, theta_full = theta_full, dominant_full = dominant_full,
                          max_share_full = max_share_full, refit_components_identical = refit_components_identical,
                          refit_dominant_identical = refit_dominant_identical,
                          topic_summary = topic_summary, topic_prevalence_by_type = topic_prevalence_by_type,
                          topic_prevalence_by_day = topic_prevalence_by_day, topic_correlation = topic_correlation,
                          concentration = concentration, concentration_summary = concentration_summary,
                          representative_documents = representative_documents,
                          document_topic_table = document_topic_table, topic_word_table = topic_word_table,
                          stability_matrix = stability_matrix, topic_cols = topic_cols, topic_labels = topic_labels)
  rm(theta_m, theta_full, document_topic_table, topic_word_table)
  invisible(gc())
}
mark_stage("retained")

# ---- Accounting and processing-report tables ----
# INPUT : docs; the counts recorded during text processing; vocabulary_table.
# DOES  : document accounting from source to modeled corpus by document type;
#         the text-processing report (rule, count); document length bands.
# OUTPUT: document_accounting; text_processing_report; document_length_bands;
#         stopwords_used; parameters_table.
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
  sprintf("step 7: stopword tokens removed (SMART list, %d terms)", length(stopword_list)), n_tokens_stop_total,
  sprintf("step 7: stopword tokens the Snowball list (%d terms) would remove instead (not applied)", length(stopwords("en", source = "snowball"))), n_stop_snowball_reference,
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

stopwords_used <- tibble(term = stopword_list, source = stopword_source,
                         in_snowball_list = stopword_list %in% stopwords("en", source = "snowball"))

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
  "n_top_words", as.character(n_top_words),
  "coherence_top_n", as.character(coherence_top_n),
  "relevance_lambda", as.character(relevance_lambda),
  "n_representative", as.character(n_representative),
  "representative_min_tokens", as.character(representative_min_tokens),
  "active_topic_threshold", as.character(active_topic_threshold),
  "stability_cosine_threshold", as.character(stability_cosine_threshold),
  "n_workers", as.character(n_workers),
  "k_composite (lowest composite rank)", as.character(k_composite),
  "k_coherence (highest NPMI coherence)", as.character(k_coherence),
  "k_perplexity_minimum (lowest mean held-out perplexity)", as.character(k_perplexity_minimum),
  "k_perplexity (one-standard-error rule)", as.character(k_perplexity),
  "retained_k", paste(retained_k, collapse = ", "))

# ---- Figures ----
# INPUT : the comparison, trace, stability and retained-model tables.
# DOES  : PNG figures with a title, axis titles, tick labels, and a legend
#         where more than one series or a colour scale is shown.
# OUTPUT: figures/*.png.
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

# Figure 1 — candidate comparison: diagnostics against K, one point per seed, line through the seed mean.
metric_labels <- c(heldout_perplexity = "Held-out perplexity, document completion (lower is better)",
                   train_perplexity   = "Training perplexity (lower is better)",
                   coherence_npmi     = "Coherence, NPMI over top 10 words (higher is better)",
                   coherence_umass    = "Coherence, UMass over top 10 words (higher is better)",
                   exclusivity        = "Exclusivity of top 10 words (higher is better)",
                   cao_juan_2009      = "Mean pairwise topic cosine, Cao et al. 2009 (lower is better)",
                   deveaud_2014       = "Mean pairwise symmetric KL, Deveaud et al. 2014 (higher is better)",
                   arun_2010          = "Singular-value divergence, Arun et al. 2010 (lower is better)")
stability_labels <- c(matched_cosine_mean     = "Stability: mean matched-topic cosine across seeds (higher is better)",
                      ari_dominant_topics_mean = "Stability: adjusted Rand index of dominant topics (higher is better)")
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
  labs(title = "LDA candidate topic counts — r/politics, July 2026",
       x = "Number of topics (K)", y = "Diagnostic value") +
  theme_fig + theme(legend.position = "bottom")
ggsave(file.path(fig_dir, "fig1_model_comparison.png"), fig1, width = 11, height = 13, dpi = 150, bg = "white")

# Figure 2 — pseudo log-likelihood per training token along the sampling iterations, one panel per K.
fig2 <- fit_traces |>
  mutate(K_label = factor(paste("K =", K), levels = paste("K =", candidate_k))) |>
  ggplot(aes(x = iteration, y = pseudo_loglikelihood_per_token, colour = factor(seed))) +
  geom_line(linewidth = 0.6) +
  facet_wrap(~ K_label, scales = "free_y", ncol = 4) +
  scale_colour_manual(values = seed_colours, name = "Random seed") +
  labs(title = "Sampling traces of the LDA fits — r/politics, July 2026",
       x = "Sampling iteration", y = "Pseudo log-likelihood per training token") +
  theme_fig + theme(legend.position = "bottom")
ggsave(file.path(fig_dir, "fig2_fit_traces.png"), fig2, width = 12, height = 6.5, dpi = 150, bg = "white")

# Figure 3 — stability: matched-topic cosine similarity between seeds, per K.
fig3 <- stability_pairs |>
  mutate(K_factor = factor(K, levels = candidate_k)) |>
  ggplot(aes(x = K_factor, y = cosine)) +
  geom_boxplot(outlier.shape = NA, colour = ink_sub, fill = NA, width = 0.5, linewidth = 0.4) +
  geom_jitter(position = position_jitter(width = 0.18, height = 0, seed = 1), colour = ink_series, alpha = 0.45, size = 1.2) +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, by = 0.2)) +
  labs(title = "Topic stability across random seeds — r/politics, July 2026",
       x = "Number of topics (K)", y = "Cosine similarity of matched topic-word distributions") +
  theme_fig
ggsave(file.path(fig_dir, "fig3_stability_matched_cosine.png"), fig3, width = 9, height = 6, dpi = 150, bg = "white")

for (tag in names(retained)) {
  r <- retained[[tag]]; K <- r$K
  topic_levels_desc <- rev(r$topic_labels)
  # Figure 4 — topic prevalence, token-weighted and document-mean shares.
  prev_long <- r$topic_summary |>
    select(topic_label, `Share of modeled tokens` = token_share, `Mean share within documents` = doc_mean_share) |>
    pivot_longer(-topic_label, names_to = "measure", values_to = "share") |>
    mutate(topic_label = factor(topic_label, levels = topic_levels_desc),
           measure = factor(measure, levels = c("Share of modeled tokens", "Mean share within documents")))
  fig4 <- ggplot(prev_long, aes(x = share, y = topic_label, fill = measure)) +
    geom_col(position = position_dodge(width = 0.8), width = 0.7) +
    scale_fill_manual(values = c(ink_series, ink_series_2), name = "Measure") +
    scale_x_continuous(labels = label_percent(accuracy = 0.1), expand = expansion(mult = c(0, 0.05))) +
    labs(title = sprintf("Topic prevalence, K = %d — r/politics, July 2026", K), x = "Share", y = NULL) +
    theme_fig + theme(legend.position = "bottom", panel.grid.major.y = element_blank())
  ggsave(file.path(fig_dir, sprintf("fig4_topic_prevalence_%s.png", tag)), fig4,
         width = 9, height = 3 + 0.22 * K, dpi = 150, bg = "white", limitsize = FALSE)

  # Figure 5 — the ten highest-probability words of every topic.
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
    labs(title = sprintf("Highest-probability words per topic, K = %d — r/politics, July 2026", K),
         x = "Probability of the word in the topic", y = NULL) +
    theme_fig + theme(panel.grid.major.y = element_blank(), axis.text.y = element_text(size = 8), axis.text.x = element_text(size = 7))
  ggsave(file.path(fig_dir, sprintf("fig5_top_words_%s.png", tag)), fig5,
         width = 2.6 * fig_cols + 1, height = 2.2 * ceiling(K / fig_cols) + 1, dpi = 130, bg = "white", limitsize = FALSE)

  # Figure 6 — topic prevalence by UTC day.
  fig6 <- r$topic_prevalence_by_day |>
    mutate(topic_label = factor(paste("Topic", topic), levels = topic_levels_desc)) |>
    ggplot(aes(x = created_date_utc, y = topic_label, fill = token_share)) +
    geom_tile() +
    scale_fill_gradientn(colours = blue_ramp, labels = label_percent(accuracy = 1), name = "Share of the day's modeled tokens") +
    scale_x_date(date_breaks = "5 days", date_labels = "%b %d", expand = expansion(mult = 0.01)) +
    labs(title = sprintf("Topic prevalence by day, K = %d — r/politics, July 2026 (UTC)", K), x = "Day (UTC)", y = NULL) +
    theme_fig + theme(legend.position = "bottom", panel.grid.major = element_blank())
  ggsave(file.path(fig_dir, sprintf("fig6_topic_prevalence_by_day_%s.png", tag)), fig6,
         width = 10, height = 3 + 0.2 * K, dpi = 150, bg = "white", limitsize = FALSE)

  # Figure 7 — document concentration: share of the dominant topic, by document length band.
  fig7 <- r$concentration |>
    mutate(token_band = factor(paste(token_band, "modeled tokens"), levels = paste(band_labels, "modeled tokens"))) |>
    ggplot() +
    geom_rect(aes(xmin = share_lower, xmax = share_upper, ymin = 0, ymax = prop_in_band), fill = ink_series) +
    facet_wrap(~ token_band, ncol = 4) +
    scale_x_continuous(breaks = seq(0, 1, by = 0.25), limits = c(0, 1)) +
    scale_y_continuous(labels = label_percent(accuracy = 1), expand = expansion(mult = c(0, 0.05))) +
    labs(title = sprintf("Share of the dominant topic within documents, K = %d — r/politics, July 2026", K),
         x = "Share of the document's tokens in its dominant topic (bins of 0.05)", y = "Share of documents in the length band") +
    theme_fig
  ggsave(file.path(fig_dir, sprintf("fig7_document_concentration_%s.png", tag)), fig7, width = 11, height = 6, dpi = 150, bg = "white")

  # Figure 8 — correlation between topic shares across documents.
  cor_long <- r$topic_correlation |>
    pivot_longer(-topic, names_to = "topic_b", values_to = "correlation") |>
    mutate(topic_b = as.integer(sub("topic_", "", topic_b)),
           label_a = factor(paste("Topic", topic), levels = topic_levels_desc),
           label_b = factor(paste("Topic", topic_b), levels = r$topic_labels))
  cor_limit <- max(abs(cor_long$correlation[cor_long$topic != cor_long$topic_b]))
  fig8 <- ggplot(cor_long |> filter(topic != topic_b), aes(x = label_b, y = label_a, fill = correlation)) +
    geom_tile() +
    scale_fill_gradient2(low = ink_series, mid = mid_grey, high = ink_red, midpoint = 0, limits = c(-cor_limit, cor_limit),
                         name = "Pearson correlation of document shares") +
    labs(title = sprintf("Topic co-occurrence within documents, K = %d — r/politics, July 2026", K), x = NULL, y = NULL) +
    theme_fig + theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1, size = 7), axis.text.y = element_text(size = 7),
                      legend.position = "bottom", panel.grid.major = element_blank())
  ggsave(file.path(fig_dir, sprintf("fig8_topic_correlation_%s.png", tag)), fig8,
         width = max(8, 3 + 0.2 * K), height = max(8, 3.5 + 0.2 * K), dpi = 150, bg = "white", limitsize = FALSE)

  # Figure 9 — stability matrix: topic-word cosine similarity, seed 1 topics against seed 2 topics.
  sc <- stability_cos[[as.character(K)]]
  fig9 <- r$stability_matrix |>
    mutate(label_1 = factor(paste("Topic", seed1_topic), levels = topic_levels_desc),
           label_2 = factor(paste("Topic", seed2_topic), levels = paste("Topic", sc$assignment))) |>
    ggplot(aes(x = label_2, y = label_1, fill = cosine)) +
    geom_tile() +
    scale_fill_gradientn(colours = blue_ramp, limits = c(0, 1), name = "Cosine similarity of topic-word distributions") +
    labs(title = sprintf("Topic matching between seeds 1 and 2, K = %d — r/politics, July 2026", K),
         x = "Seed 2 topics, in matched order", y = "Seed 1 topics") +
    theme_fig + theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1, size = 7), axis.text.y = element_text(size = 7),
                      legend.position = "bottom", panel.grid.major = element_blank())
  ggsave(file.path(fig_dir, sprintf("fig9_stability_matrix_%s.png", tag)), fig9,
         width = max(8, 3 + 0.2 * K), height = max(8, 3.5 + 0.2 * K), dpi = 150, bg = "white", limitsize = FALSE)

  # Figure 10 — topic prevalence by document type.
  fig10 <- r$topic_prevalence_by_type |>
    mutate(topic_label = factor(paste("Topic", topic), levels = topic_levels_desc),
           doc_type = factor(doc_type, levels = c("comment", "submission"), labels = c("Comments", "Submissions"))) |>
    ggplot(aes(x = token_share, y = topic_label, fill = doc_type)) +
    geom_col(position = position_dodge(width = 0.8), width = 0.7) +
    scale_fill_manual(values = c(ink_series, ink_series_2), name = "Document type") +
    scale_x_continuous(labels = label_percent(accuracy = 0.1), expand = expansion(mult = c(0, 0.05))) +
    labs(title = sprintf("Topic prevalence by document type, K = %d — r/politics, July 2026", K),
         x = "Share of the document type's modeled tokens", y = NULL) +
    theme_fig + theme(legend.position = "bottom", panel.grid.major.y = element_blank())
  ggsave(file.path(fig_dir, sprintf("fig10_topic_prevalence_by_type_%s.png", tag)), fig10,
         width = 9, height = 3 + 0.22 * K, dpi = 150, bg = "white", limitsize = FALSE)
}
mark_stage("figures")

# ---- Validation checks ----
# INPUT : the objects above.
# DOES  : reconcile every population and token count, recompute the
#         document-term rows of sampled documents from the stored text by a
#         second code path, confirm the split and the held-out halves, check
#         the probability distributions, the determinism of the re-fits, the
#         perplexity implementation against text2vec::perplexity, the topic
#         matching, the retention rule, and the integrity of every retained
#         model's output tables; stop if any check fails.
# OUTPUT: validation_checks (one row per check).
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
status_counts <- table(factor(docs$model_status, levels = status_levels))
add_check("document accounting: statuses sum to source documents; modeled = train + held-out",
          sprintf("%s = %d; %d = %d + %d", paste(status_counts, collapse = " + "), n_docs, n_modeled, n_train, n_heldout),
          sum(status_counts) == n_docs && n_modeled == n_train + n_heldout && n_docs == n_comments_all + n_submissions_all)
add_check("token accounting: letter tokens = short + stopword + marker-document + rare + modeled",
          sprintf("%d = %d + %d + %d + %d + %d", n_letter_tokens, n_tokens_short_total, n_tokens_stop_total, n_tokens_in_marker_docs,
                  n_tokens_rare_total, n_tokens_modeled_total),
          n_letter_tokens == n_tokens_short_total + n_tokens_stop_total + n_tokens_in_marker_docs + n_tokens_rare_total + n_tokens_modeled_total &&
            n_candidate_tokens == n_letter_tokens - n_tokens_short_total - n_tokens_stop_total)
add_check("vocabulary: kept terms satisfy the df floor and ceiling; dtm columns are the vocabulary; column sums equal term frequencies",
          sprintf("%d terms; df %d..%d; ceiling %.0f", n_vocabulary, min(term_df_all[term_kept]), max(term_df_all[term_kept]), max_document_share * n_nonmarker_docs),
          all(term_df_all[term_kept] >= min_document_frequency) && all(term_df_all[term_kept] <= max_document_share * n_nonmarker_docs) &&
            identical(colnames(dtm_all), vocabulary) && all(colSums(dtm_all) == term_tf_all[term_kept]) && nrow(dtm_all) == n_docs &&
            identical(rownames(dtm_all), docs$doc_key))
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
    f$iterations >= n_check_convergence && f$iterations <= n_iter_max && all(diff(f$token_share) <= 1e-12)
}, logical(1))
add_check("every fit: K x V topic-word counts summing to the training tokens; phi rows, token shares and document-mean shares sum to 1; dominant topics valid; topics ordered by token share",
          sprintf("%d of %d fits pass (%d requested)", sum(fit_ok), length(fits), length(job_list)), all(fit_ok) && length(fits) == length(job_list))

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
add_check("stability: every Hungarian assignment is a permutation; matched cosines in [0, 1]; ARI of a labelling with itself = 1",
          sprintf("%d of %d assignments are permutations; cosine %.3f..%.3f; self-ARI %.6f", sum(stability_assignment_ok), length(stability_assignment_ok),
                  min(stability_pairs$cosine), max(stability_pairs$cosine), adjusted_rand_index(fits[[1]]$dominant, fits[[1]]$dominant)),
          all(stability_assignment_ok) && all(stability_pairs$cosine >= 0 & stability_pairs$cosine <= 1 + 1e-9) &&
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

for (tag in names(retained)) {
  r <- retained[[tag]]; K <- r$K; dt <- r$document_topic_table; tc <- r$topic_cols
  add_check(sprintf("%s: re-fit in the main process reproduces the worker's topic-word counts and dominant topics exactly", tag),
            sprintf("components identical %s; dominant identical %s", r$refit_components_identical, r$refit_dominant_identical),
            r$refit_components_identical && r$refit_dominant_identical)
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
  add_check(sprintf("%s: concentration bins contain every modeled document; correlation matrix is K x K with unit diagonal", tag),
            sprintf("%d docs in bins; %d x %d matrix; diagonal %.6f..%.6f", sum(r$concentration$docs), nrow(r$topic_correlation), length(tc),
                    min(diag(as.matrix(r$topic_correlation[, tc]))), max(diag(as.matrix(r$topic_correlation[, tc])))),
            sum(r$concentration$docs) == n_modeled && nrow(r$topic_correlation) == K && all(abs(diag(as.matrix(r$topic_correlation[, tc])) - 1) < 1e-9))
}
add_check("figure data: comparison rows = fits x metrics; trace rows = sum of checks per fit; stability rows = sum of K x seed pairs",
          sprintf("%d = %d x %d; %d; %d", nrow(comparison_long), length(fits), length(metric_labels), nrow(fit_traces), nrow(stability_pairs)),
          nrow(comparison_long) == length(fits) * length(metric_labels) && nrow(fit_traces) == sum(vapply(fits, function(f) nrow(f$trace), 1L)) &&
            nrow(stability_pairs) == sum(candidate_k) * length(seed_pairs))

validation_checks <- bind_rows(checks)
print(as.data.frame(validation_checks[, c("check", "pass")]), right = FALSE)
mark_stage("validation")

# ---- Write tables ----
# INPUT : the measurement objects above.
# DOES  : write one CSV per table and record its dimensions for the sidecars;
#         stop before writing anything else if a validation check failed.
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

write_table(document_accounting,         "document_accounting.csv")
write_table(text_processing_report,      "text_processing_report.csv")
write_table(text_availability,           "text_availability.csv")
write_table(stopwords_used,              "stopwords_used.csv")
write_table(vocabulary_table,            "vocabulary.csv")
write_table(document_length_bands,       "document_length_bands.csv")
write_table(document_length_summary,     "document_length_summary.csv")
write_table(parameters_table,            "parameters.csv")
write_table(model_comparison_by_fit,     "model_comparison_by_fit.csv")
write_table(model_comparison_summary,    "model_comparison_summary.csv")
write_table(fit_traces,                  "fit_traces.csv")
write_table(stability_pairs,             "stability_pairs.csv")
write_table(stability_summary,           "stability_summary.csv")
write_table(top_words_probability,       "top_words_by_probability_all_models.csv")
write_table(top_words_relevance,         "top_words_by_relevance_all_models.csv")
write_table(topic_prevalence_all_models, "topic_prevalence_all_models.csv")
for (tag in names(retained)) {
  r <- retained[[tag]]
  write_table(r$topic_summary,             sprintf("topic_summary_%s.csv", tag))
  write_table(r$topic_prevalence_by_type,  sprintf("topic_prevalence_by_type_%s.csv", tag))
  write_table(r$topic_prevalence_by_day,   sprintf("topic_prevalence_by_day_%s.csv", tag))
  write_table(r$topic_correlation,         sprintf("topic_correlation_%s.csv", tag))
  write_table(r$concentration,             sprintf("document_concentration_bins_%s.csv", tag))
  write_table(r$concentration_summary,     sprintf("document_concentration_summary_%s.csv", tag))
  write_table(r$representative_documents,  sprintf("representative_documents_%s.csv", tag))
  write_table(r$stability_matrix,          sprintf("stability_matrix_seed1_vs_seed2_%s.csv", tag))
  write_table(r$topic_word_table,          sprintf("topic_word_distribution_%s.csv", tag))
  write_table(r$document_topic_table,      sprintf("document_topic_distribution_%s.csv", tag))
}
mark_stage("tables")

run_end <- Sys.time()
stage_names <- names(stage_times)
stage_durations <- tibble(stage = stage_names,
                          seconds = round(as.numeric(difftime(do.call(c, stage_times), c(run_start, do.call(c, stage_times)[-length(stage_times)]), units = "secs")), 1))
run_info <- bind_rows(
  tibble(item = "run_start_utc", value = format(run_start, "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")),
  tibble(item = "run_end_utc",   value = format(run_end,   "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")),
  tibble(item = "total_seconds", value = as.character(round(as.numeric(difftime(run_end, run_start, units = "secs")), 1))),
  stage_durations |> transmute(item = paste0("stage_seconds_", stage), value = as.character(seconds)),
  tibble(item = "fits_wall_seconds_sum_over_workers", value = as.character(round(sum(model_comparison_by_fit$elapsed_seconds), 1))),
  tibble(item = "workers", value = as.character(min(n_workers, length(job_list)))),
  tibble(item = "fits_loaded_from_cache", value = as.character(fits_from_cache)),
  tibble(item = "R_version", value = R.version.string),
  tibble(item = "text2vec_version", value = as.character(packageVersion("text2vec"))),
  tibble(item = "quick_run", value = as.character(quick_run)))
write_table(run_info, "run_info.csv")

# ---- Write provenance sidecars ----
# INPUT : the CSV and PNG outputs; the 8 source Parquet files.
# DOES  : write one <output>.manifest.yml per output recording input/output
#         SHA256 hashes, this script's hash, git commit, parameters, seed and
#         package versions.
# OUTPUT: *.manifest.yml sidecars next to each output.
sha256 <- function(path) paste0("sha256:", digest(path, algo = "sha256", file = TRUE))
git_commit <- tryCatch(system2("git", c("-C", project_dir, "rev-parse", "--short", "HEAD"), stdout = TRUE, stderr = FALSE),
                       error = function(e) "")
manifest_packages <- c("nanoparquet", "dplyr", "tidyr", "stringi", "Matrix", "text2vec", "stopwords", "clue",
                       "ggplot2", "scales", "readr", "digest", "yaml")
package_versions  <- lapply(manifest_packages, function(p) list(name = p, version = as.character(packageVersion(p))))
input_files <- lapply(c(comment_files, submission_files), function(p) list(path = p, hash = sha256(p), format = "parquet"))
manifest_parameters <- c(
  list(reader = "nanoparquet",
       documents = "one per comment (body) and one per submission (title + ' ' + selftext); doc_key = Reddit fullname",
       text_processing = "lower case; U+2019 -> '; URLs removed; tokens = Unicode letter runs with internal apostrophes; trailing 's stripped; tokens < 2 chars removed; SMART stopwords removed; removal-marker-only documents excluded; terms kept when 5 <= df <= 50% of non-marker documents; no stemming",
       model = "LDA, WarpLDA collapsed Gibbs sampling (text2vec), symmetric priors alpha = alpha_sum / K and beta",
       heldout = "document-completion perplexity on a seeded 5% held-out sample (half A inferred, half B scored)",
       topic_numbering = "descending token-weighted prevalence within each fit",
       retention = "lowest mean rank over held-out perplexity, NPMI coherence, exclusivity, matched-topic cosine; plus the best NPMI coherence K and the one-standard-error held-out perplexity K (smallest K within one seed SD of the minimum) if different"),
  setNames(as.list(parameters_table$value), parameters_table$parameter))

write_sidecar <- function(output) {
  is_csv <- grepl("\\.csv$", output)
  manifest <- list(manifest_version = 1L, output_file = output, output_hash = sha256(output),
                   output_format = if (is_csv) "csv" else "image/png")
  if (is_csv) {
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
  manifest$notes     <- "Phase 1A LDA topic models of the July 2026 r/politics sample; source Parquet read-only; fits use seeds 1-3 (primary seed 1) and the split seed recorded in parameters."
  write_yaml(manifest, paste0(output, ".manifest.yml"))
}
invisible(lapply(list.files(tab_dir, pattern = "\\.csv$", full.names = TRUE), write_sidecar))
invisible(lapply(list.files(fig_dir, pattern = "\\.png$", full.names = TRUE), write_sidecar))

# ---- Console summary ----
# INPUT : the headline tables.
# DOES  : print them.
# OUTPUT: console text.
print(as.data.frame(document_accounting))
print(as.data.frame(text_processing_report))
print(as.data.frame(model_comparison_summary[, c("K", "iterations_fit_mean", "heldout_perplexity_mean", "coherence_npmi_mean",
                                                 "exclusivity_mean", "matched_cosine_mean", "ari_dominant_topics_mean",
                                                 "composite_rank", "retained")]))
for (tag in names(retained)) {
  cat("\n==", tag, "==\n")
  print(as.data.frame(retained[[tag]]$topic_summary[, c("topic", "token_share", "doc_mean_share", "dominant_share", "coherence_npmi", "top_10_words_by_probability")]))
}
print(as.data.frame(run_info))
cat("\nDone. Figures ->", fig_dir, "\nTables ->", tab_dir, "\n")
