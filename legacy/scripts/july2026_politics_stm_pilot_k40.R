# First pilot structural topic model (STM), K = 40, on the July 2026 r/politics sample.
# Uses the text representation of the Tier 2 LDA run (july2026_politics_lda_topics_tier2.R), rebuilt here
# and checked against its saved outputs. Prevalence varies with doc_type and a B-spline of the day of July.
# Writes diagnostics, prevalence tables, linkage checks, figures and the model to r_analysis_outputs/stm_pilot_k40/.
# Env: STM_QUICK=<folder> (smoke test; STM_QUICK_N, STM_QUICK_ITS), STM_MAX_EM_ITS (default 100),
#      STM_MEMORY_SAMPLER=0, STM_REFIT=1.

# ---- Setup: packages, paths, parameters, output directories ----
suppressPackageStartupMessages({
  library(nanoparquet); library(dplyr); library(tidyr); library(stringi); library(Matrix); library(readxl)
  library(stm); library(ps); library(ggplot2); library(scales); library(readr); library(digest); library(yaml); library(patchwork)
})
startup_elapsed_seconds <- proc.time()[["elapsed"]]
run_start    <- Sys.time()
run_id       <- format(run_start, "%Y%m%dT%H%M%SZ", tz = "UTC")
stage_log    <- list()
stage_cursor <- run_start
process_memory <- function() {
  m <- ps::ps_memory_info()
  c(rss_mb = as.numeric(m[["rss"]]) / 2^20, peak_wset_mb = as.numeric(m[["peak_wset"]]) / 2^20)
}
mark_stage <- function(name) {
  now <- Sys.time(); mem <- process_memory()
  stage_log[[length(stage_log) + 1L]] <<- tibble(stage = name, start_utc = stage_cursor, end_utc = now,
                                                 seconds = as.numeric(difftime(now, stage_cursor, units = "secs")),
                                                 rss_mb_at_end = round(mem[["rss_mb"]]), peak_wset_mb_so_far = round(mem[["peak_wset_mb"]]))
  stage_cursor <<- now
}
log_line <- function(...) { cat(format(Sys.time(), "%H:%M:%S"), ..., "\n"); flush.console() }
mat_tibble <- function(m, names) { m <- as.matrix(m); dimnames(m) <- list(NULL, names); as_tibble(m) }

project_dir       <- Sys.getenv("KC_REPO_ROOT", unset = getwd())
comments_dir      <- file.path(project_dir, "data_sample/comments/2026-07")
submissions_dir   <- file.path(project_dir, "data_sample/submissions/2026-07")
script_path       <- file.path(project_dir, "july2026_politics_stm_pilot_k40.R")
tier2_dir         <- file.path(project_dir, "r_analysis_outputs/lda_topics_tier2")            # original Tier 2 run (read-only)
k40_dir           <- file.path(project_dir, "r_analysis_outputs/lda_topics_tier2_k40_k80")    # K = 40 companion package (read-only)
inventory_dir     <- file.path(project_dir, "r_analysis_outputs/metadata_inventory")          # metadata inventory (read-only)
stopword_workbook <- file.path(project_dir, "r_analysis_outputs/stopword_lists/curation/stopword_curation_tier2_cumulative_310_words.xlsx")
stopword_sheet    <- "tier2_cumulative_310"
quick_run <- nzchar(Sys.getenv("STM_QUICK"))
out_dir   <- if (quick_run) file.path(Sys.getenv("STM_QUICK"), "stm_pilot_k40_quick") else file.path(project_dir, "r_analysis_outputs/stm_pilot_k40")
fig_dir   <- file.path(out_dir, "figures")
tab_dir   <- file.path(out_dir, "tables")
model_dir <- file.path(out_dir, "models")
dir.create(fig_dir,   recursive = TRUE, showWarnings = FALSE)
dir.create(tab_dir,   recursive = TRUE, showWarnings = FALSE)
dir.create(model_dir, recursive = TRUE, showWarnings = FALSE)
# Tables and figures are regenerated on every run; models/ is a keyed cache of the fit and its input.
invisible(file.remove(list.files(c(fig_dir, tab_dir), pattern = "\\.(csv|png|yml)$", full.names = TRUE)))

# Tier 2 parameters (unchanged; they reproduce the documents, vocabulary and split).
heldout_share          <- 0.05
split_seed             <- 20260731L
min_token_chars        <- 2L
min_document_frequency <- 5L
max_document_share     <- 0.5
stopword_source        <- "tier2_cumulative_310"
representative_min_tokens <- 25L
n_representative       <- 5L
active_topic_threshold <- 0.1
band_breaks            <- c(0, 1, 4, 9, 24, 49, 99, Inf)
band_labels            <- c("1", "2-4", "5-9", "10-24", "25-49", "50-99", "100+")
# STM pilot settings.
K                <- 40L
stm_seed         <- 1L        # recorded by stm; the spectral initialisation and the EM are deterministic without ngroups, so it has no effect here
init_type        <- "Spectral"
max_em_its       <- as.integer(Sys.getenv("STM_MAX_EM_ITS", "100"))
emtol            <- 1e-5
report_every     <- 10L
spline_df        <- 10L       # s(day_of_july, df = 10): stm's default B-spline basis (cubic, knots at quantiles)
gamma_prior      <- "Pooled"
sigma_prior      <- 0
prevalence_formula <- ~ doc_type + s(day_of_july, df = 10)
day0             <- as.numeric(as.POSIXct("2026-07-01 00:00:00", tz = "UTC"))   # day_of_july = (created_utc - day0) / 86400
effect_nsims     <- 25L       # estimateEffect draws from the variational posterior (Global approximation)
effect_plot_nsims <- 100L     # simulated coefficient draws behind the expected-proportion curves and contrasts
effect_npoints   <- 63L       # grid points over the observed range of day_of_july (about half a day apart)
effect_seed      <- 20260731L
coherence_M      <- 10L
exclusivity_frexw <- 0.7
label_n_words    <- 20L
frex_weight      <- 0.5
correlation_cutoff <- 0.01
sample_stride    <- 5000L     # every sample_stride-th STM document is re-tokenised by a second code path
# Implementation settings (results do not depend on them).
memory_sampler_on <- !identical(Sys.getenv("STM_MEMORY_SAMPLER"), "0")
sampler_interval  <- if (quick_run) 5 else 20
force_refit       <- identical(Sys.getenv("STM_REFIT"), "1")
quick_n           <- as.integer(Sys.getenv("STM_QUICK_N", "20000"))
quick_seed        <- 1L
if (quick_run) { max_em_its <- as.integer(Sys.getenv("STM_QUICK_ITS", "3")); effect_nsims <- 5L; effect_plot_nsims <- 50L }

comment_files    <- sort(list.files(comments_dir,    pattern = "\\.parquet$", full.names = TRUE))
submission_files <- sort(list.files(submissions_dir, pattern = "\\.parquet$", full.names = TRUE))
sha256 <- function(path) paste0("sha256:", digest(path, algo = "sha256", file = TRUE))
manifest_field <- function(path, field) tryCatch({ v <- read_yaml(paste0(path, ".manifest.yml"))[[field]]; if (is.null(v)) NA else v }, error = function(e) NA)
stage_file <- file.path(model_dir, "stage_marker.txt")
begin_stage <- function(name) writeLines(name, stage_file)
log_line("start; quick_run =", quick_run, "; K =", K, "; max EM iterations =", max_em_its, "; run_id", run_id)

# ---- Stopword list: the Tier 2 workbook (read-only) ----
# Reads the 'word' column of the 310-word sheet as stored and records its SHA256.
begin_stage("read")
workbook_hash <- sha256(stopword_workbook)
tier2_raw     <- read_excel(stopword_workbook, sheet = stopword_sheet, col_types = "text")
stopword_list <- tier2_raw$word
token_regex   <- "\\p{L}+(?:'\\p{L}+)*"
log_line("workbook read:", length(stopword_list), "entries; hash", workbook_hash)

# ---- Read source data (read-only) ----
# Reads the 7 comment files and 1 submission file with nanoparquet, keeping the document-table columns plus
# score, num_comments, upvote_ratio and url. No rows are dropped; INT64 columns arrive as double.
read_type <- function(files) bind_rows(lapply(files, function(f) { x <- as_tibble(as.data.frame(read_parquet(f))); x$source_file <- basename(f); x }))
comments_raw    <- read_type(comment_files)
submissions_raw <- read_type(submission_files)
source_files <- tibble(path = c(comment_files, submission_files), file = basename(path),
                       record_type = rep(c("comment", "submission"), c(length(comment_files), length(submission_files))),
                       bytes = file.size(path), sha256 = vapply(path, sha256, character(1))) |>
  left_join(bind_rows(comments_raw |> count(source_file, name = "rows"), submissions_raw |> count(source_file, name = "rows")), by = c("file" = "source_file"))
comments <- comments_raw |>
  select(id, link_id, parent_id, author, created_utc, subreddit, body, score, source_file) |>
  mutate(created_utc = as.numeric(created_utc))
submissions <- submissions_raw |>
  select(id, author, created_utc, subreddit, title, selftext, score, num_comments, upvote_ratio, url, source_file) |>
  mutate(created_utc = as.numeric(created_utc))
rm(comments_raw, submissions_raw)
n_comments_all    <- nrow(comments)
n_submissions_all <- nrow(submissions)
mark_stage("read")
log_line("read:", n_comments_all, "comments,", n_submissions_all, "submissions")

# ---- Document table (as in the Tier 2 run) ----
# One document per record: comment text = body, submission text = title + " " + selftext, doc_key = Reddit
# fullname. Extra metadata is joined afterwards by doc_key so the original construction stays unchanged.
begin_stage("document_table")
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
extra <- bind_rows(
  comments |> transmute(doc_key = paste0("t1_", id), source_file, score, num_comments = NA_real_, upvote_ratio = NA_real_, url = NA_character_,
                        text_source_fields = "body", body_state = case_when(is.na(body) ~ "missing", body == "" ~ "empty", TRUE ~ "text")),
  submissions |> transmute(doc_key = paste0("t3_", id), source_file, score, num_comments, upvote_ratio, url,
                           text_source_fields = "title + ' ' + selftext", body_state = NA_character_))
stopifnot(identical(extra$doc_key, docs$doc_key))
mark_stage("document_table")

# ---- Text processing (Tier 2 steps 1-8; reference-only counts omitted) ----
# Lower case, U+2019 -> ', URLs removed, letter-run tokens, trailing 's stripped, short tokens and the 310
# stopwords removed, marker-only documents excluded.
begin_stage("tokenize")
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
is_short <- stri_length(tok) < min_token_chars                                # step 6
is_stop  <- !is_short & tok %in% stopword_list                                # step 7 (Tier 2)
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

# ---- Vocabulary and document-term matrix (Tier 2 rule) ----
# Terms kept when min_document_frequency <= df <= max_document_share of non-marker documents.
begin_stage("vocabulary_dtm")
vocab_all  <- sort(unique(tok), method = "radix")
term_index <- match(tok, vocab_all)
tf_all <- sparseMatrix(i = tdoc, j = term_index, x = 1, dims = c(n_docs, length(vocab_all)), repr = "C")
rm(tok, tdoc, term_index)
term_df_all      <- diff(tf_all@p)
term_tf_all      <- colSums(tf_all)
n_nonmarker_docs <- sum(!docs$marker_only)
term_below_floor   <- term_df_all < min_document_frequency
term_above_ceiling <- term_df_all > max_document_share * n_nonmarker_docs
term_kept        <- !term_below_floor & !term_above_ceiling
vocabulary       <- vocab_all[term_kept]
n_vocabulary     <- length(vocabulary)
dtm_all <- tf_all[, term_kept, drop = FALSE]
dimnames(dtm_all) <- list(docs$doc_key, vocabulary)
rm(tf_all)
vocabulary_table <- tibble(term = vocab_all, document_frequency = term_df_all, term_frequency = term_tf_all, kept = term_kept,
                           reason = case_when(term_below_floor ~ "below document-frequency floor",
                                              term_above_ceiling ~ "above document-share ceiling", TRUE ~ "kept")) |>
  arrange(desc(document_frequency), term)
n_terms_candidate <- length(vocab_all)
n_tokens_rare     <- sum(term_tf_all[!term_kept])
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

# ---- Tier 2 training / held-out split (carried as a label only) ----
# Reproduces the seeded 5% split for later comparison; the STM is fitted on all modeled documents.
begin_stage("split")
modeled_rows_all <- which(docs$model_status == "modeled")
n_modeled        <- length(modeled_rows_all)
heldout_eligible <- modeled_rows_all[docs$n_tokens_modeled[modeled_rows_all] >= 2L]
n_heldout        <- floor(heldout_share * n_modeled)
set.seed(split_seed)
heldout_rows <- sort(sample(heldout_eligible, n_heldout))
train_rows   <- setdiff(modeled_rows_all, heldout_rows)
n_train      <- length(train_rows)
docs$split   <- NA_character_
docs$split[train_rows]   <- "train"
docs$split[heldout_rows] <- "held-out"
mark_stage("split")

# ---- Reconciliation against the saved Tier 2 outputs and the inventory (read-only) ----
# Hashes every file read against its manifest and compares this run's counts (documents, tokens, vocabulary,
# per-term and per-document values, length summary) with the saved values. Discrepancies are only reported.
begin_stage("reconciliation")
tier2_tab <- file.path(tier2_dir, "tables"); k40_tab <- file.path(k40_dir, "tables"); inv_tab <- file.path(inventory_dir, "tables")
artifact_paths <- c(
  document_accounting        = file.path(tier2_tab, "document_accounting.csv"),
  text_processing_report     = file.path(tier2_tab, "text_processing_report.csv"),
  document_length_summary    = file.path(tier2_tab, "document_length_summary.csv"),
  vocabulary                 = file.path(tier2_tab, "vocabulary.csv"),
  tier2_parameters           = file.path(tier2_tab, "parameters.csv"),
  document_table_K040        = file.path(k40_tab, "document_topic_distribution_K040.csv"),
  theta_K040_seed1           = file.path(tier2_dir, "models/theta_K040_seed1.rds"),
  inventory_source_files     = file.path(inv_tab, "source_files.csv"),
  inventory_linkage_counts   = file.path(inv_tab, "linkage_counts.csv"),
  inventory_parent_linkage   = file.path(inv_tab, "parent_linkage.csv"),
  inventory_model_status     = file.path(inv_tab, "model_status_by_type.csv"))
input_integrity <- tibble(role = names(artifact_paths), path = unname(artifact_paths)) |>
  mutate(exists = file.exists(path), bytes = if_else(exists, file.size(path), NA_real_),
         sha256_now = vapply(path, function(p) if (file.exists(p)) sha256(p) else NA_character_, character(1)),
         sha256_in_manifest = vapply(path, function(p) as.character(manifest_field(p, "output_hash")), character(1)),
         hash_equals_manifest = exists & !is.na(sha256_in_manifest) & sha256_now == sha256_in_manifest)
read_if <- function(role, ...) if (file.exists(artifact_paths[[role]])) read_csv(artifact_paths[[role]], show_col_types = FALSE, progress = FALSE, ...) else NULL
saved_accounting <- read_if("document_accounting")
saved_report     <- read_if("text_processing_report")
saved_length     <- read_if("document_length_summary")
saved_vocabulary <- read_if("vocabulary", col_types = cols(term = col_character(), .default = col_guess()), na = character(), trim_ws = FALSE)
inventory_source     <- read_if("inventory_source_files")
inventory_linkage    <- read_if("inventory_linkage_counts", col_types = cols(.default = col_character()))
inventory_parent     <- read_if("inventory_parent_linkage")
inventory_status     <- read_if("inventory_model_status")
k40_table <- if (file.exists(artifact_paths[["document_table_K040"]])) {
  read_csv(artifact_paths[["document_table_K040"]],
           col_select = c("doc_key", "doc_type", "model_status", "split", "n_tokens_letters", "n_tokens_modeled", "n_chars"),
           col_types = cols(doc_key = col_character(), doc_type = col_character(), model_status = col_character(), split = col_character(),
                            n_tokens_letters = col_integer(), n_tokens_modeled = col_integer(), n_chars = col_integer(), .default = col_double()),
           na = "NA", trim_ws = FALSE, progress = FALSE, show_col_types = FALSE, lazy = FALSE)
} else NULL
theta_k40_keys <- if (file.exists(artifact_paths[["theta_K040_seed1"]])) { th <- readRDS(artifact_paths[["theta_K040_seed1"]]); k <- th$doc_key; rm(th); invisible(gc()); k } else NULL

count_by_type <- function(mask) c(sum(mask & docs$doc_type == "comment"), sum(mask & docs$doc_type == "submission"), sum(mask))
recon <- list()
add_recon <- function(item, tier2_value, this_value, source) {
  recon[[length(recon) + 1L]] <<- tibble(item = item, tier2_value = as.character(tier2_value), this_run_value = as.character(this_value),
                                          equal = !is.na(tier2_value) & !is.na(this_value) & as.character(tier2_value) == as.character(this_value),
                                          source = source)
}
acc_row <- function(stage) if (is.null(saved_accounting)) rep(NA, 3) else unlist(saved_accounting[saved_accounting$stage == stage, c("comments", "submissions", "total")])
status_stage <- c("not modeled: removal marker only" = "not modeled: removal marker only",
                  "not modeled: no letter tokens" = "not modeled: no letter tokens",
                  "not modeled: only single-character tokens or stopwords" = "not modeled: only single-character tokens or stopwords",
                  "not modeled: only terms outside the vocabulary (document-frequency floor or ceiling)" = "not modeled: only terms outside the vocabulary")
for (i in 1:3) {
  lab <- c("comments", "submissions", "total")[i]
  add_recon(sprintf("documents in source (%s)", lab), acc_row("documents in source")[i], count_by_type(rep(TRUE, n_docs))[i], "document_accounting.csv")
  for (s in names(status_stage)) add_recon(sprintf("%s (%s)", status_stage[[s]], lab), acc_row(status_stage[[s]])[i], count_by_type(docs$model_status == s)[i], "document_accounting.csv")
  add_recon(sprintf("modeled documents (%s)", lab), acc_row("modeled documents")[i], count_by_type(docs$model_status == "modeled")[i], "document_accounting.csv")
  add_recon(sprintf("modeled: training set (%s)", lab), acc_row("modeled: training set")[i], count_by_type(docs$split %in% "train")[i], "document_accounting.csv")
  add_recon(sprintf("modeled: held-out set (%s)", lab), acc_row("modeled: held-out set")[i], count_by_type(docs$split %in% "held-out")[i], "document_accounting.csv")
  add_recon(sprintf("letter-run tokens (%s)", lab), acc_row("letter-run tokens (all documents)")[i],
            c(sum(docs$n_tokens_letters[docs$doc_type == "comment"]), sum(docs$n_tokens_letters[docs$doc_type == "submission"]), sum(docs$n_tokens_letters))[i], "document_accounting.csv")
  add_recon(sprintf("modeled tokens (%s)", lab), acc_row("modeled tokens")[i],
            c(sum(docs$n_tokens_modeled[docs$doc_type == "comment"]), sum(docs$n_tokens_modeled[docs$doc_type == "submission"]), sum(docs$n_tokens_modeled))[i], "document_accounting.csv")
}
rep_item <- function(item) if (is.null(saved_report)) NA else saved_report$value[saved_report$item == item]
add_recon("vocabulary size (terms modeled)", rep_item("vocabulary size (terms modeled)"), n_vocabulary, "text_processing_report.csv")
add_recon("distinct candidate terms (vocabulary before pruning)", rep_item("distinct candidate terms in non-marker documents (vocabulary before pruning)"), n_terms_candidate, "text_processing_report.csv")
add_recon("terms below the document-frequency floor", rep_item("step 9: terms in fewer than 5 non-marker documents removed"), sum(term_below_floor), "text_processing_report.csv")
add_recon("tokens removed with rare terms", rep_item("step 9: tokens removed with those terms"), n_tokens_rare, "text_processing_report.csv")
add_recon("terms above the document-share ceiling", rep_item("step 9: terms in more than 50% of non-marker documents removed"), sum(term_above_ceiling), "text_processing_report.csv")
add_recon("candidate tokens in non-marker documents", rep_item("candidate tokens in non-marker documents"), n_candidate_tokens - n_tokens_in_marker_docs, "text_processing_report.csv")
add_recon("tokens modeled", rep_item("tokens modeled"), sum(docs$n_tokens_modeled), "text_processing_report.csv")
if (!is.null(saved_vocabulary)) {
  vj <- saved_vocabulary |> select(term, d_df = document_frequency, d_tf = term_frequency, d_kept = kept) |>
    full_join(vocabulary_table |> select(term, document_frequency, term_frequency, kept), by = "term")
  add_recon("vocabulary.csv: rows", nrow(saved_vocabulary), nrow(vocabulary_table), "vocabulary.csv")
  add_recon("vocabulary.csv: terms present in both", nrow(saved_vocabulary), sum(!is.na(vj$d_df) & !is.na(vj$document_frequency)), "vocabulary.csv")
  add_recon("vocabulary.csv: rows with equal document_frequency, term_frequency and kept", nrow(saved_vocabulary),
            sum(!is.na(vj$d_df) & !is.na(vj$document_frequency) & vj$d_df == vj$document_frequency & vj$d_tf == vj$term_frequency & vj$d_kept == vj$kept), "vocabulary.csv")
  rm(vj)
}
if (!is.null(k40_table)) {
  m <- match(k40_table$doc_key, docs$doc_key)
  add_recon("K = 40 document table: rows", 902144, nrow(k40_table), "document_topic_distribution_K040.csv")
  add_recon("K = 40 document table: doc_keys matched to this run's documents", nrow(k40_table), sum(!is.na(m)), "document_topic_distribution_K040.csv")
  add_recon("K = 40 document table: rows with equal doc_type", nrow(k40_table), sum(k40_table$doc_type == docs$doc_type[m], na.rm = TRUE), "document_topic_distribution_K040.csv")
  add_recon("K = 40 document table: rows with equal model_status", nrow(k40_table), sum(k40_table$model_status == docs$model_status[m], na.rm = TRUE), "document_topic_distribution_K040.csv")
  add_recon("K = 40 document table: rows with equal split (NA = NA)", nrow(k40_table),
            sum((is.na(k40_table$split) & is.na(docs$split[m])) | (!is.na(k40_table$split) & !is.na(docs$split[m]) & k40_table$split == docs$split[m])), "document_topic_distribution_K040.csv")
  add_recon("K = 40 document table: rows with equal n_tokens_modeled", nrow(k40_table), sum(k40_table$n_tokens_modeled == docs$n_tokens_modeled[m], na.rm = TRUE), "document_topic_distribution_K040.csv")
  add_recon("K = 40 document table: rows with equal n_tokens_letters", nrow(k40_table), sum(k40_table$n_tokens_letters == docs$n_tokens_letters[m], na.rm = TRUE), "document_topic_distribution_K040.csv")
  add_recon("K = 40 document table: rows with equal n_chars", nrow(k40_table), sum(k40_table$n_chars == docs$n_chars[m], na.rm = TRUE), "document_topic_distribution_K040.csv")
  rm(m)
}
if (!is.null(theta_k40_keys)) {
  add_recon("theta_K040_seed1.rds: doc_key count", n_modeled, length(theta_k40_keys), "theta_K040_seed1.rds")
  add_recon("theta_K040_seed1.rds: doc_key set equals this run's modeled documents", TRUE, setequal(theta_k40_keys, docs$doc_key[modeled_rows_all]), "theta_K040_seed1.rds")
}
if (!is.null(saved_length)) {
  ql <- function(x, p) quantile(x, p, type = 7, names = FALSE)
  for (tp in c("comment", "submission")) {
    x <- docs$n_tokens_modeled[docs$model_status == "modeled" & docs$doc_type == tp]
    d <- saved_length[saved_length$doc_type == tp, ]
    vals <- c(docs = length(x), tokens = sum(x), min = min(x), p05 = ql(x, 0.05), p25 = ql(x, 0.25), median = ql(x, 0.5), mean = mean(x), p75 = ql(x, 0.75), p95 = ql(x, 0.95), max = max(x))
    for (nm in names(vals)) add_recon(sprintf("document_length_summary.csv: %s %s", tp, nm), as.character(signif(d[[nm]], 12)), as.character(signif(vals[[nm]], 12)), "document_length_summary.csv")
  }
}
if (!is.null(inventory_source)) {
  for (i in seq_len(nrow(source_files))) {
    d <- inventory_source$sha256[inventory_source$file == source_files$file[i]]
    add_recon(sprintf("source file SHA256 unchanged since the inventory: %s", source_files$file[i]), if (length(d)) d else NA, source_files$sha256[i], "metadata_inventory/tables/source_files.csv")
  }
}
if (!is.null(inventory_linkage)) {
  lk <- function(item) { v <- inventory_linkage$value[inventory_linkage$item == item]; if (length(v)) v else NA }
  add_recon("inventory: Tier 2 modelable documents", lk("Tier 2 modelable documents (model_status == 'modeled')"), n_modeled, "metadata_inventory/tables/linkage_counts.csv")
  add_recon("inventory: Tier 2 modelable comments", lk("Tier 2 modelable comments"), count_by_type(docs$model_status == "modeled")[1], "metadata_inventory/tables/linkage_counts.csv")
  add_recon("inventory: Tier 2 modelable submissions", lk("Tier 2 modelable submissions"), count_by_type(docs$model_status == "modeled")[2], "metadata_inventory/tables/linkage_counts.csv")
}
add_recon("expected modelable comments", 846666, count_by_type(docs$model_status == "modeled")[1], "earlier count")
add_recon("expected modelable submissions", 10719, count_by_type(docs$model_status == "modeled")[2], "earlier count")
add_recon("expected modelable total", 857385, n_modeled, "earlier count")
tier2_reconciliation <- bind_rows(recon)
discrepancies <- tier2_reconciliation |> filter(!equal)
rm(k40_table, saved_vocabulary); invisible(gc())
mark_stage("reconciliation")
log_line("reconciliation:", sum(tier2_reconciliation$equal), "of", nrow(tier2_reconciliation), "items equal;", nrow(discrepancies), "discrepancies")

# ---- STM input: documents, vocabulary, covariates ----
# Converts the modeled documents to stm format (seeded subsample under STM_QUICK) and accounts for every
# document, token and term. Covariates: doc_type (comment = reference) and day_of_july, the fractional day
# since 2026-07-01 00:00 UTC, entered as s(day_of_july, df = 10).
begin_stage("stm_input")
modeled_rows <- modeled_rows_all
if (quick_run) { set.seed(quick_seed); modeled_rows <- sort(sample(modeled_rows_all, min(quick_n, n_modeled))) }
dtm_stm <- dtm_all[modeled_rows, , drop = FALSE]
n_stm_in <- nrow(dtm_stm)
tokens_stm_in <- sum(dtm_stm)
terms_stm_in  <- ncol(dtm_stm)
terms_unseen  <- sum(colSums(dtm_stm) == 0)
corp <- withCallingHandlers(asSTMCorpus(dtm_stm), warning = function(w) { log_line("asSTMCorpus warning:", conditionMessage(w)); invokeRestart("muffleWarning") })
documents <- corp$documents
vocab     <- corp$vocab
rm(corp)
doc_tokens_stm <- vapply(documents, function(d) sum(d[2, ]), numeric(1))
doc_terms_stm  <- vapply(documents, ncol, integer(1))
docs_stm <- docs[modeled_rows, ] |> left_join(extra, by = "doc_key") |>
  mutate(stm_row = row_number(), n_tokens_stm = as.integer(doc_tokens_stm), n_distinct_terms_stm = doc_terms_stm,
         day_of_july = (created_utc - day0) / 86400, doc_type_f = factor(doc_type, levels = c("comment", "submission")))
meta_df <- data.frame(doc_type = docs_stm$doc_type_f, day_of_july = docs_stm$day_of_july, stringsAsFactors = FALSE)
spline_basis <- s(meta_df$day_of_july, df = spline_df)
spline_knots <- attr(spline_basis, "knots"); spline_boundary <- attr(spline_basis, "Boundary.knots"); spline_degree <- attr(spline_basis, "degree")
term_df_stm <- diff(dtm_stm@p)[colSums(dtm_stm) > 0]
stm_input_accounting <- tibble(
  stage = c("documents in source", "Tier 2 modelable documents (this run)", "documents entering the STM conversion",
            "STM documents after conversion", "documents lost in conversion", "letter-run tokens (source, all documents)",
            "Tier 2 modeled tokens (this run, all modelable documents)", "tokens entering the STM conversion", "tokens in the STM documents",
            "tokens lost in conversion", "Tier 2 vocabulary terms", "terms entering the STM conversion", "terms unseen in the entering documents (dropped by asSTMCorpus)",
            "STM vocabulary terms", "STM document-term cells (distinct term-document pairs)"),
  comments = c(n_comments_all, count_by_type(docs$model_status == "modeled")[1], sum(docs_stm$doc_type == "comment"), sum(docs_stm$doc_type == "comment"), 0,
               sum(docs$n_tokens_letters[docs$doc_type == "comment"]), sum(docs$n_tokens_modeled[docs$doc_type == "comment" & docs$model_status == "modeled"]),
               sum(docs_stm$n_tokens_modeled[docs_stm$doc_type == "comment"]), sum(docs_stm$n_tokens_stm[docs_stm$doc_type == "comment"]),
               sum(docs_stm$n_tokens_modeled[docs_stm$doc_type == "comment"]) - sum(docs_stm$n_tokens_stm[docs_stm$doc_type == "comment"]), NA, NA, NA, NA,
               sum(docs_stm$n_distinct_terms_stm[docs_stm$doc_type == "comment"])),
  submissions = c(n_submissions_all, count_by_type(docs$model_status == "modeled")[2], sum(docs_stm$doc_type == "submission"), sum(docs_stm$doc_type == "submission"), 0,
                  sum(docs$n_tokens_letters[docs$doc_type == "submission"]), sum(docs$n_tokens_modeled[docs$doc_type == "submission" & docs$model_status == "modeled"]),
                  sum(docs_stm$n_tokens_modeled[docs_stm$doc_type == "submission"]), sum(docs_stm$n_tokens_stm[docs_stm$doc_type == "submission"]),
                  sum(docs_stm$n_tokens_modeled[docs_stm$doc_type == "submission"]) - sum(docs_stm$n_tokens_stm[docs_stm$doc_type == "submission"]), NA, NA, NA, NA,
                  sum(docs_stm$n_distinct_terms_stm[docs_stm$doc_type == "submission"])),
  total = c(n_docs, n_modeled, n_stm_in, length(documents), n_stm_in - length(documents), n_letter_tokens, sum(docs$n_tokens_modeled),
            tokens_stm_in, sum(doc_tokens_stm), tokens_stm_in - sum(doc_tokens_stm), n_vocabulary, terms_stm_in, terms_unseen, length(vocab), sum(doc_terms_stm)))
covariate_summary <- bind_rows(
  tibble(item = "prevalence formula", value = paste(deparse(prevalence_formula), collapse = " ")),
  tibble(item = "doc_type levels (reference first)", value = paste(levels(meta_df$doc_type), collapse = ", ")),
  tibble(item = "doc_type = comment", value = as.character(sum(meta_df$doc_type == "comment"))),
  tibble(item = "doc_type = submission", value = as.character(sum(meta_df$doc_type == "submission"))),
  tibble(item = "day_of_july definition", value = sprintf("(created_utc - %d) / 86400; %d = 2026-07-01 00:00:00 UTC; continuous fractional day", as.integer(day0), as.integer(day0))),
  tibble(item = "day_of_july minimum / median / mean / maximum", value = sprintf("%.6f / %.6f / %.6f / %.6f", min(meta_df$day_of_july), median(meta_df$day_of_july), mean(meta_df$day_of_july), max(meta_df$day_of_july))),
  tibble(item = "day_of_july quantiles p05 / p25 / p75 / p95", value = paste(sprintf("%.4f", quantile(meta_df$day_of_july, c(0.05, 0.25, 0.75, 0.95), type = 7)), collapse = " / ")),
  tibble(item = "day_of_july missing values", value = as.character(sum(is.na(meta_df$day_of_july)))),
  tibble(item = "time basis", value = sprintf("stm::s(day_of_july, df = %d): splines::bs, degree %d, %d interior knots at quantiles of day_of_july", spline_df, spline_degree, length(spline_knots))),
  tibble(item = "spline interior knots (day_of_july)", value = paste(sprintf("%.4f", spline_knots), collapse = ", ")),
  tibble(item = "spline boundary knots (day_of_july)", value = paste(sprintf("%.6f", spline_boundary), collapse = ", ")),
  tibble(item = "design matrix columns", value = as.character(2L + ncol(spline_basis))),
  tibble(item = "covariates deliberately not used", value = "score, num_comments, upvote_ratio, url, author, link_id, parent_id, retrieved_on, subreddit (constant): preserved in the linkage table only"))
usable <- function(x) !is.na(x) & !(is.character(x) & x == "")
marker_strings <- c("[deleted]", "[removed]", "[ Removed by Reddit ]", "[ Removed by moderator ]")
coverage_row <- function(field, x, scope, route) {
  n <- length(x); u <- usable(x)
  tibble(field = field, scope = scope, route = route, n_documents = n, n_usable = sum(u), pct_usable = round(100 * sum(u) / n, 4),
         n_distinct_usable = n_distinct(x[u]), n_missing = sum(is.na(x)), n_empty = sum(!is.na(x) & is.character(x) & x == ""),
         n_marker_valued = if (is.character(x)) sum(x %in% marker_strings) else 0L)
}
sub_by_key <- submissions |> transmute(link_id = paste0("t3_", id), submission_author = author, submission_created_utc = created_utc,
                                       submission_title = title, submission_selftext = selftext, submission_url = url, submission_score = score,
                                       submission_num_comments = num_comments, submission_upvote_ratio = upvote_ratio)
parent_by_key <- docs_stm |> filter(doc_type == "comment") |> transmute(parent_id = doc_key, parent_author = author, parent_created_utc = created_utc, parent_stm_row = stm_row, parent_score = score)
stm_comments <- docs_stm |> filter(doc_type == "comment") |> left_join(sub_by_key, by = "link_id") |> left_join(parent_by_key, by = "parent_id")
stm_subs     <- docs_stm |> filter(doc_type == "submission")
metadata_coverage <- bind_rows(
  lapply(c("doc_key", "doc_id", "author", "created_utc", "score", "link_id", "parent_id", "submission_id"), function(f) coverage_row(f, docs_stm[[f]][docs_stm$doc_type == "comment"], "STM comments", "native field")),
  coverage_row("body", docs$text[modeled_rows][docs_stm$doc_type == "comment"], "STM comments", "native field (the modeled text)"),
  lapply(c("doc_key", "doc_id", "author", "created_utc", "score", "link_id", "submission_id", "num_comments", "upvote_ratio", "url"), function(f) coverage_row(f, docs_stm[[f]][docs_stm$doc_type == "submission"], "STM submissions", "native field")),
  coverage_row("title", submissions$title[match(stm_subs$doc_id, submissions$id)], "STM submissions", "native field (part of the modeled text)"),
  coverage_row("selftext", submissions$selftext[match(stm_subs$doc_id, submissions$id)], "STM submissions", "native field (part of the modeled text)"),
  lapply(c("author", "created_utc", "score"), function(f) coverage_row(f, docs_stm[[f]], "all STM documents", "native field")),
  coverage_row("day_of_july (covariate)", docs_stm$day_of_july, "all STM documents", "derived from created_utc"),
  coverage_row("doc_type (covariate)", docs_stm$doc_type, "all STM documents", "record type"),
  lapply(c("submission_author", "submission_created_utc", "submission_title", "submission_selftext", "submission_url", "submission_score", "submission_num_comments", "submission_upvote_ratio"),
         function(f) coverage_row(f, stm_comments[[f]], "STM comments", "join on link_id = 't3_' + submission id")),
  lapply(c("parent_author", "parent_created_utc", "parent_score", "parent_stm_row"), function(f) coverage_row(f, stm_comments[[f]], "STM comments", "join on parent_id = doc_key of a parent comment that is itself an STM document")))
documents_per_day <- docs_stm |> count(doc_type, created_date_utc, name = "stm_documents") |>
  left_join(docs_stm |> group_by(doc_type, created_date_utc) |> summarise(stm_tokens = sum(n_tokens_stm), mean_tokens = mean(n_tokens_stm), .groups = "drop"), by = c("doc_type", "created_date_utc")) |>
  left_join(docs |> count(doc_type, created_date_utc, name = "source_records"), by = c("doc_type", "created_date_utc")) |>
  mutate(share_of_source_records = stm_documents / source_records) |> arrange(doc_type, created_date_utc)
document_length_summary_stm <- docs_stm |> group_by(doc_type) |>
  summarise(docs = n(), tokens = sum(n_tokens_stm), min = min(n_tokens_stm), p05 = quantile(n_tokens_stm, 0.05, type = 7, names = FALSE), p25 = quantile(n_tokens_stm, 0.25, type = 7, names = FALSE),
            median = median(n_tokens_stm), mean = mean(n_tokens_stm), p75 = quantile(n_tokens_stm, 0.75, type = 7, names = FALSE), p95 = quantile(n_tokens_stm, 0.95, type = 7, names = FALSE), max = max(n_tokens_stm), .groups = "drop")
input_key <- digest(list(dim(dtm_stm), length(dtm_stm@x), sum(dtm_stm), digest(vocab), digest(docs_stm$doc_key), digest(meta_df)), algo = "sha256")
stm_input_path <- file.path(model_dir, "stm_input_K040_pilot.rds")
saveRDS(list(documents = documents, vocab = vocab, meta = docs_stm |> select(stm_row, doc_key, doc_type, created_utc, day_of_july, n_tokens_stm),
             prevalence_formula = paste(deparse(prevalence_formula), collapse = " "), day0 = day0,
             spline = list(df = spline_df, degree = spline_degree, knots = spline_knots, boundary_knots = spline_boundary), input_key = input_key, quick_run = quick_run),
        stm_input_path, compress = FALSE)
rm(dtm_all); invisible(gc())
mark_stage("stm_input")
log_line("stm input:", length(documents), "documents;", length(vocab), "terms;", sum(doc_tokens_stm), "tokens; input key", substr(input_key, 1, 12))

# ---- Background memory sampler ----
# A second Rscript process records this process's memory and CPU and the system's available memory every
# sampler_interval seconds until a stop file appears.
trace_csv  <- file.path(model_dir, sprintf("resource_trace_%s.csv", run_id))
stop_file  <- file.path(model_dir, "sampler_stop.txt")
if (file.exists(stop_file)) invisible(file.remove(stop_file))
sampler_started <- FALSE
if (memory_sampler_on) {
  sampler_path <- file.path(model_dir, "memory_sampler.R")
  writeLines(c(
    'args <- commandArgs(trailingOnly = TRUE)',
    'pid <- as.integer(args[1]); trace_csv <- args[2]; stage_file <- args[3]; stop_file <- args[4]; interval <- as.numeric(args[5])',
    'h <- ps::ps_handle(pid)',
    'repeat {',
    '  if (file.exists(stop_file) || !ps::ps_is_running(h)) break',
    '  ok <- try({',
    '    m <- ps::ps_memory_info(h); sm <- ps::ps_system_memory(); cpu <- ps::ps_cpu_times(h)',
    '    stage <- if (file.exists(stage_file)) readLines(stage_file, warn = FALSE)[1] else NA_character_',
    '    row <- data.frame(time_utc = format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"), stage = stage,',
    '                      rss_mb = as.numeric(m[["rss"]]) / 2^20, wset_mb = as.numeric(m[["wset"]]) / 2^20, peak_wset_mb = as.numeric(m[["peak_wset"]]) / 2^20,',
    '                      private_mb = as.numeric(m[["mem_private"]]) / 2^20, system_total_mb = as.numeric(sm$total) / 2^20, system_avail_mb = as.numeric(sm$avail) / 2^20,',
    '                      cpu_user_s = as.numeric(cpu[["user"]]), cpu_system_s = as.numeric(cpu[["system"]]))',
    '    write.table(row, trace_csv, sep = ",", row.names = FALSE, col.names = !file.exists(trace_csv), append = file.exists(trace_csv))',
    '  }, silent = TRUE)',
    '  Sys.sleep(interval)',
    '}'), sampler_path)
  sampler_log <- file.path(model_dir, "memory_sampler.log")
  system2(file.path(R.home("bin"), "Rscript.exe"), args = c(shQuote(sampler_path), Sys.getpid(), shQuote(trace_csv), shQuote(stage_file), shQuote(stop_file), sampler_interval),
          wait = FALSE, stdout = sampler_log, stderr = sampler_log)
  sampler_started <- TRUE
  log_line("memory sampler started; interval", sampler_interval, "s")
}

# ---- Fit the STM (or load it from the cache) ----
# Spectral initialisation, Pooled gamma prior, console to models/fit_console_K040.log. The model and fit
# record are cached under a run key; STM_REFIT=1 forces a refit.
begin_stage("fit")
run_key <- digest(list(input_key, K, paste(deparse(prevalence_formula), collapse = " "), init_type, stm_seed, max_em_its, emtol, gamma_prior, sigma_prior,
                       as.character(packageVersion("stm")), R.version.string), algo = "sha256")
model_path      <- file.path(model_dir, "stm_K040_pilot.rds")
fit_record_path <- file.path(model_dir, "fit_record_K040.rds")
run_key_file    <- file.path(model_dir, "run_key.txt")
fit_from_cache  <- !force_refit && file.exists(model_path) && file.exists(fit_record_path) && file.exists(run_key_file) && identical(readLines(run_key_file, warn = FALSE)[1], run_key)
if (fit_from_cache) {
  model <- readRDS(model_path)
  fit_record <- readRDS(fit_record_path)
  log_line("model loaded from cache (fitting run", fit_record$run_id, ")")
} else {
  fit_log <- file.path(model_dir, "fit_console_K040.log")
  log_con <- file(fit_log, open = "wt")
  sink(log_con, split = TRUE)
  cat(sprintf("run_id %s; fit start %s\n", run_id, format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")))
  fit_start <- Sys.time(); cpu_before <- proc.time()
  model <- stm(documents, vocab, K = K, prevalence = ~ doc_type + s(day_of_july, df = 10), data = meta_df,
               init.type = init_type, seed = stm_seed, max.em.its = max_em_its, emtol = emtol, verbose = TRUE, reportevery = report_every,
               gamma.prior = gamma_prior, sigma.prior = sigma_prior)
  fit_end <- Sys.time(); cpu_after <- proc.time()
  cat(sprintf("fit end %s\n", format(fit_end, "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")))
  sink(); close(log_con)
  mem_after_fit <- process_memory()
  saveRDS(model, model_path, compress = FALSE)
  fit_record <- list(run_id = run_id, run_key = run_key, fit_start_utc = format(fit_start, "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
                     fit_end_utc = format(fit_end, "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"), wall_seconds = as.numeric(difftime(fit_end, fit_start, units = "secs")),
                     cpu_user_seconds = unname((cpu_after - cpu_before)[["user.self"]]), cpu_system_seconds = unname((cpu_after - cpu_before)[["sys.self"]]),
                     model_time_seconds = model$time, peak_wset_mb_after_fit = unname(mem_after_fit[["peak_wset_mb"]]), rss_mb_after_fit = unname(mem_after_fit[["rss_mb"]]),
                     log_lines = readLines(fit_log, warn = FALSE), trace_csv = trace_csv, stm_version = as.character(packageVersion("stm")),
                     model_file_bytes = file.size(model_path))
  saveRDS(fit_record, fit_record_path)
  writeLines(run_key, run_key_file)
  log_line("fit done:", model$convergence$its, "iterations; converged", model$convergence$converged, ";", round(fit_record$wall_seconds), "s; model saved")
}
mark_stage("fit")

# ---- The fitted model: dimensions, convergence and the EM iteration record ----
# Reads K, N and V and joins per-iteration E- and M-step seconds from the console log to the bound trace.
begin_stage("model_outputs")
theta <- model$theta
beta  <- exp(model$beta$logbeta[[1]])
stopifnot(nrow(theta) == length(documents), ncol(theta) == K, ncol(beta) == length(vocab))
topic_ids    <- seq_len(K)
topic_labels <- sprintf("Topic %d", topic_ids)
topic_cols   <- sprintf("topic_%02d", topic_ids)
n_tokens_total <- sum(model$settings$dim$wcounts$x)
estep_lines <- grep("^Completed E-Step", fit_record$log_lines, value = TRUE)
mstep_lines <- grep("^Completed M-Step", fit_record$log_lines, value = TRUE)
estep_seconds <- as.numeric(sub("^Completed E-Step \\((\\d+) seconds\\).*$", "\\1", estep_lines))
mstep_seconds <- ifelse(grepl("seconds", mstep_lines), as.numeric(sub("^Completed M-Step \\((\\d+) seconds\\).*$", "\\1", mstep_lines)), 0)
its <- model$convergence$its
bound <- model$convergence$bound
em_iterations <- tibble(iteration = seq_along(bound), bound = bound, bound_per_token = bound / n_tokens_total,
                        relative_change = c(NA, diff(bound) / abs(head(bound, -1))),
                        estep_seconds = if (length(estep_seconds) >= length(bound)) estep_seconds[seq_along(bound)] else c(estep_seconds, rep(NA, length(bound) - length(estep_seconds))),
                        mstep_seconds_reported = if (length(mstep_seconds) >= length(bound)) mstep_seconds[seq_along(bound)] else c(mstep_seconds, rep(NA, length(bound) - length(mstep_seconds)))) |>
  mutate(cumulative_estep_seconds = cumsum(coalesce(estep_seconds, 0)))
init_seconds_estimate <- fit_record$wall_seconds - sum(em_iterations$estep_seconds, na.rm = TRUE) - sum(em_iterations$mstep_seconds_reported, na.rm = TRUE)
model_specification <- tibble(
  item = c("implementation", "K", "documents (N)", "vocabulary (V)", "tokens", "prevalence formula", "content covariate", "time representation",
           "initialisation", "seed (recorded by stm)", "max EM iterations", "EM tolerance (relative change of the approximate bound)", "gamma prior (prevalence coefficients)",
           "sigma prior (covariance regularisation)", "kappa prior (content; unused)", "ngroups (memoized inference)", "EM iterations run", "converged",
           "final approximate bound", "final bound per token", "lbound (bound + lfactorial(K))", "last relative change", "fit wall seconds", "fit CPU user seconds",
           "fit CPU system seconds", "initialisation seconds (wall minus E- and M-step seconds; approximate)", "mean E-step seconds", "E-step seconds per document (mean)",
           "process peak working set after the fit (MB)", "model object bytes (uncompressed RDS)", "fitting run id", "loaded from cache in this run", "stm version", "R version"),
  value = c("stm::stm (variational EM; single R process; E-step loops over documents)", K, model$settings$dim$N, model$settings$dim$V, n_tokens_total,
            paste(deparse(prevalence_formula), collapse = " "), "none", sprintf("day_of_july = (created_utc - %d) / 86400, B-spline s(day_of_july, df = %d)", as.integer(day0), spline_df),
            init_type, model$settings$seed, max_em_its, format(emtol, scientific = TRUE), gamma_prior, sigma_prior, "L1 (default; no content covariate, so unused)", model$settings$ngroups,
            its, model$convergence$converged, sprintf("%.3f", tail(bound, 1)), sprintf("%.6f", tail(bound, 1) / n_tokens_total), sprintf("%.3f", tail(bound, 1) + lfactorial(K)),
            if (length(bound) > 1) format(tail(em_iterations$relative_change, 1), digits = 4) else NA, round(fit_record$wall_seconds, 1), round(fit_record$cpu_user_seconds, 1),
            round(fit_record$cpu_system_seconds, 1), round(init_seconds_estimate, 1), round(mean(em_iterations$estep_seconds, na.rm = TRUE), 1),
            format(mean(em_iterations$estep_seconds, na.rm = TRUE) / model$settings$dim$N, digits = 4), round(fit_record$peak_wset_mb_after_fit), fit_record$model_file_bytes,
            fit_record$run_id, fit_from_cache, fit_record$stm_version, R.version.string))
mark_stage("model_outputs")

# ---- Topic-word summaries and STM diagnostics ----
# labelTopics, semanticCoherence and NPMI coherence (top 10 words), exclusivity (frexw = 0.7), checkResiduals,
# topicCorr and full theta correlations, and sigma as a correlation matrix.
begin_stage("diagnostics_labels")
labels <- labelTopics(model, n = label_n_words, frexweight = frex_weight)
word_prob <- function(k, terms) beta[k, match(terms, vocab)]
top_words <- bind_rows(lapply(c("prob", "frex", "lift", "score"), function(type) {
  m <- labels[[type]]
  bind_rows(lapply(topic_ids, function(k) tibble(topic = k, label_type = type, rank = seq_len(ncol(m)), term = m[k, ], probability_in_topic = word_prob(k, m[k, ]))))
}))
top10_prob <- labels$prob[, 1:10, drop = FALSE]
mark_stage("diagnostics_labels")
# The slow model-only diagnostics (semantic coherence, exclusivity, residual dispersion) and the estimateEffect object are
# cached under the run key next to the model, so a cached re-run regenerates the tables and figures without recomputing them.
diag_cache_path <- file.path(model_dir, "diagnostics_cache_K040.rds")
effect_path     <- file.path(model_dir, "estimate_effect_K040.rds")
diag_cache <- if (!force_refit && file.exists(diag_cache_path)) readRDS(diag_cache_path) else NULL
diag_from_cache <- !is.null(diag_cache) && identical(diag_cache$run_key, run_key) && file.exists(effect_path)
diag_cache_run_id <- if (diag_from_cache) diag_cache$computed_run_id else NA_character_
if (diag_from_cache) log_line("slow diagnostics and the estimateEffect object loaded from the cache (computed in run", diag_cache_run_id, ")")
begin_stage("diagnostics_coherence")
t_stage <- Sys.time()
if (diag_from_cache) { coherence_stm <- diag_cache$coherence_stm; excl <- diag_cache$exclusivity } else {
  coherence_stm <- semanticCoherence(model, documents, M = coherence_M)
  excl <- exclusivity(model, M = coherence_M, frexw = exclusivity_frexw)
}
seconds_coherence <- if (diag_from_cache) diag_cache$seconds_coherence else as.numeric(difftime(Sys.time(), t_stage, units = "secs"))
top_idx <- apply(model$beta$logbeta[[1]], 1, function(x) order(x, decreasing = TRUE)[1:coherence_M])   # M x K term indices
union_idx <- sort(unique(as.vector(top_idx)))
vocab_col <- match(vocab, colnames(dtm_stm))
B <- dtm_stm[, vocab_col[union_idx], drop = FALSE]; B@x[] <- 1
co <- as.matrix(crossprod(B)); rm(B)
N_stm <- nrow(dtm_stm)
coherence_npmi <- vapply(topic_ids, function(k) {
  idx <- match(top_idx[, k], union_idx); pr <- expand.grid(a = idx, b = idx); pr <- pr[pr$a < pr$b, ]
  pij <- co[cbind(pr$a, pr$b)] / N_stm; pi <- co[cbind(pr$a, pr$a)] / N_stm; pj <- co[cbind(pr$b, pr$b)] / N_stm
  v <- ifelse(pij > 0, (log(pij) - log(pi) - log(pj)) / (-log(pij)), -1)
  mean(v)
}, numeric(1))
rm(co)
mark_stage("diagnostics_coherence")
begin_stage("diagnostics_residuals")
t_stage <- Sys.time()
resid <- if (diag_from_cache) diag_cache$resid else checkResiduals(model, documents)
seconds_residuals <- if (diag_from_cache) diag_cache$seconds_residuals else as.numeric(difftime(Sys.time(), t_stage, units = "secs"))
mark_stage("diagnostics_residuals")
begin_stage("diagnostics_correlation")
tc <- topicCorr(model, method = "simple", cutoff = correlation_cutoff)
cor_theta <- cor(theta)
sigma_cor <- cov2cor(model$sigma)
topic_correlation <- mat_tibble(cor_theta, topic_cols) |> mutate(topic = topic_ids, quantity = "Pearson correlation of the two topics' document proportions (theta) over all STM documents") |>
  select(topic, quantity, all_of(topic_cols))
positive_edges <- which(tc$posadj == 1 & upper.tri(tc$posadj), arr.ind = TRUE)
topic_correlation_edges <- tibble(topic_a = positive_edges[, 1], topic_b = positive_edges[, 2], correlation = tc$poscor[positive_edges]) |> arrange(desc(correlation))
sigma_table <- mat_tibble(sigma_cor, sprintf("eta_%02d", seq_len(K - 1))) |> mutate(eta = sprintf("eta_%02d", seq_len(K - 1)),
  quantity = sprintf("correlation form of the model's sigma: covariance of the logistic-normal prior over eta_k = log(theta_k / theta_%d), k = 1..%d", K, K - 1)) |>
  select(eta, quantity, everything())
mark_stage("diagnostics_correlation")
log_line("diagnostics: coherence mean", round(mean(coherence_stm), 2), "; exclusivity mean", round(mean(excl), 2), "; residual dispersion", round(resid$dispersion, 3))

# ---- Topic prevalence: overall, by document type, by UTC day ----
# Mean theta, token-weighted share, dominant and active-topic counts by type and UTC day; concentration by
# length band; top author among each topic's dominant documents.
begin_stage("prevalence")
n_tok <- docs_stm$n_tokens_stm
dominant <- max.col(theta, ties.method = "first")
dominant_share <- theta[cbind(seq_len(nrow(theta)), dominant)]
n_active <- as.integer(rowSums(theta >= active_topic_threshold))
expected_proportion <- colMeans(theta)
token_share <- as.numeric(crossprod(theta, n_tok)) / sum(n_tok)
prevalence_rank <- rank(-expected_proportion, ties.method = "first")
type_groups <- as.character(docs_stm$doc_type)
topic_prevalence_by_type <- bind_rows(lapply(c("comment", "submission"), function(tp) {
  sel <- type_groups == tp
  tibble(doc_type = tp, topic = topic_ids, documents = sum(sel), mean_theta = colMeans(theta[sel, , drop = FALSE]),
         token_share = as.numeric(crossprod(theta[sel, , drop = FALSE], n_tok[sel])) / sum(n_tok[sel]),
         documents_dominant = tabulate(dominant[sel], K), share_documents_dominant = tabulate(dominant[sel], K) / sum(sel))
}))
day_key <- paste(type_groups, docs_stm$created_date_utc)
day_sum_theta <- rowsum(theta, day_key); day_sum_tok_theta <- rowsum(theta * n_tok, day_key)
day_n <- tabulate(factor(day_key, levels = rownames(day_sum_theta)), nbins = nrow(day_sum_theta)); day_tok <- rowsum(n_tok, day_key)[rownames(day_sum_theta), 1]
topic_prevalence_by_day <- bind_rows(lapply(seq_len(nrow(day_sum_theta)), function(i) {
  parts <- strsplit(rownames(day_sum_theta)[i], " ", fixed = TRUE)[[1]]
  tibble(doc_type = parts[1], created_date_utc = as.Date(parts[2]), topic = topic_ids, stm_documents = day_n[i], stm_tokens = day_tok[i],
         mean_theta = day_sum_theta[i, ] / day_n[i], token_share = day_sum_tok_theta[i, ] / day_tok[i])
})) |> arrange(doc_type, created_date_utc, topic)
docs_stm$dominant_topic <- dominant; docs_stm$dominant_topic_share <- dominant_share; docs_stm$n_active_topics <- n_active
docs_stm$token_band <- cut(docs_stm$n_tokens_stm, breaks = band_breaks, labels = band_labels, right = TRUE)
document_concentration <- docs_stm |> group_by(doc_type, token_band) |>
  summarise(docs = n(), median_dominant_share = median(dominant_topic_share), p25_dominant_share = quantile(dominant_topic_share, 0.25, names = FALSE),
            p75_dominant_share = quantile(dominant_topic_share, 0.75, names = FALSE), share_docs_dominant_at_least_half = mean(dominant_topic_share >= 0.5),
            mean_active_topics = mean(n_active_topics), median_max_theta_minus_uniform = median(dominant_topic_share) - 1 / K, .groups = "drop")
top_author_by_topic <- docs_stm |> count(dominant_topic, author, name = "n") |> group_by(dominant_topic) |> mutate(share = n / sum(n)) |>
  slice_max(order_by = n, n = 1, with_ties = FALSE) |> ungroup() |> transmute(topic = dominant_topic, top_author_among_dominant = author, top_author_share_among_dominant = share)
mark_stage("prevalence")

# ---- Covariate effects: estimateEffect (method of composition) ----
# 25 Global draws; simulated proportions for comment vs submission at the median day and along a 63-point
# day grid per type, on the expected-proportion scale.
begin_stage("effects")
t_stage <- Sys.time()
effect_formula <- as.formula(sprintf("1:%d ~ doc_type + s(day_of_july, df = %d)", K, spline_df))
if (diag_from_cache) { prep <- readRDS(effect_path) } else {
  set.seed(effect_seed)
  prep <- estimateEffect(effect_formula, model, metadata = meta_df, uncertainty = "Global", nsims = effect_nsims)
  # estimateEffect re-creates the formula inside its own frame, so the returned formula and model-frame terms reference an
  # environment holding the model and simulation matrices; pointing them at the global environment keeps the saved object
  # small (the design matrix is rebuilt from prep$formula and prep$data, which stay intact).
  environment(prep$formula) <- globalenv()
  attr(attr(prep$modelframe, "terms"), ".Environment") <- globalenv()
  saveRDS(prep, effect_path, compress = FALSE)
}
seconds_effects_estimate <- if (diag_from_cache) diag_cache$seconds_effects_estimate else as.numeric(difftime(Sys.time(), t_stage, units = "secs"))
set.seed(effect_seed)
eff_summary <- summary(prep, nsim = 500)
effect_coefficients <- bind_rows(lapply(seq_along(eff_summary$topics), function(i) {
  tb <- eff_summary$tables[[i]]
  tibble(topic = eff_summary$topics[i], term = rownames(tb), estimate = tb[, 1], std_error = tb[, 2], t_value = tb[, 3], p_value = tb[, 4])
}))
pdf(NULL)
set.seed(effect_seed)
pe <- plot(prep, covariate = "doc_type", method = "pointestimate", omit.plot = TRUE, nsims = effect_plot_nsims)
set.seed(effect_seed)
df_ <- plot(prep, covariate = "doc_type", method = "difference", cov.value1 = "submission", cov.value2 = "comment", omit.plot = TRUE, nsims = effect_plot_nsims)
set.seed(effect_seed)
cc <- plot(prep, covariate = "day_of_july", method = "continuous", moderator = "doc_type", moderator.value = "comment", npoints = effect_npoints, omit.plot = TRUE, printlegend = FALSE, nsims = effect_plot_nsims)
set.seed(effect_seed)
cs <- plot(prep, covariate = "day_of_july", method = "continuous", moderator = "doc_type", moderator.value = "submission", npoints = effect_npoints, omit.plot = TRUE, printlegend = FALSE, nsims = effect_plot_nsims)
dev.off()
pe_levels <- as.character(pe$uvals)
effect_doc_type <- bind_rows(lapply(seq_along(pe$topics), function(i) {
  k <- pe$topics[i]
  tibble(topic = k,
         expected_comment = pe$means[[i]][pe_levels == "comment"], comment_ci_low = pe$cis[[i]][1, pe_levels == "comment"], comment_ci_high = pe$cis[[i]][2, pe_levels == "comment"],
         expected_submission = pe$means[[i]][pe_levels == "submission"], submission_ci_low = pe$cis[[i]][1, pe_levels == "submission"], submission_ci_high = pe$cis[[i]][2, pe_levels == "submission"],
         difference_submission_minus_comment = df_$means[[i]], difference_ci_low = df_$cis[[i]][1], difference_ci_high = df_$cis[[i]][2],
         day_of_july_held_at = median(meta_df$day_of_july))
})) |> left_join(effect_coefficients |> filter(term == "doc_typesubmission") |> transmute(topic, coefficient_submission = estimate, coefficient_se = std_error, coefficient_p = p_value), by = "topic")
effect_day <- bind_rows(
  bind_rows(lapply(seq_along(cc$topics), function(i) tibble(doc_type = "comment", topic = cc$topics[i], day_of_july = cc$x, expected_proportion = cc$means[[i]], ci_low = cc$ci[[i]][1, ], ci_high = cc$ci[[i]][2, ]))),
  bind_rows(lapply(seq_along(cs$topics), function(i) tibble(doc_type = "submission", topic = cs$topics[i], day_of_july = cs$x, expected_proportion = cs$means[[i]], ci_low = cs$ci[[i]][1, ], ci_high = cs$ci[[i]][2, ])))) |>
  mutate(date_utc = as.POSIXct(day0 + day_of_july * 86400, origin = "1970-01-01", tz = "UTC"))
effect_day_range <- effect_day |> group_by(doc_type, topic) |>
  summarise(curve_min = min(expected_proportion), curve_max = max(expected_proportion), curve_range = curve_max - curve_min,
            day_of_max = day_of_july[which.max(expected_proportion)], day_of_min = day_of_july[which.min(expected_proportion)], .groups = "drop")
if (!diag_from_cache) {
  saveRDS(list(run_key = run_key, computed_run_id = run_id, coherence_stm = coherence_stm, exclusivity = excl, resid = resid,
               seconds_coherence = seconds_coherence, seconds_residuals = seconds_residuals, seconds_effects_estimate = seconds_effects_estimate),
          diag_cache_path)
}
mark_stage("effects")
log_line("effects: estimated for", length(prep$topics), "topics")

# ---- Representative documents ----
# (a) findThoughts, top five per topic; (b) the Tier 2 rule: highest share among documents with at least 25
# tokens, one per distinct text and author, ties by doc_key. Both include identifiers and the full text.
begin_stage("representative_documents")
texts_stm <- docs$text[modeled_rows]
ft <- findThoughts(model, texts = texts_stm, topics = topic_ids, n = n_representative)
rep_row <- function(k, r, idx, rule) {
  tibble(rule = rule, topic = k, topic_label = topic_labels[k], rank = r, stm_row = idx, topic_share_in_document = theta[idx, k],
         doc_key = docs_stm$doc_key[idx], doc_id = docs_stm$doc_id[idx], doc_type = docs_stm$doc_type[idx], author = docs_stm$author[idx],
         created_utc = docs_stm$created_utc[idx], submission_id = docs_stm$submission_id[idx], link_id = docs_stm$link_id[idx], parent_id = docs_stm$parent_id[idx],
         tier2_split = docs_stm$split[idx], n_tokens_stm = docs_stm$n_tokens_stm[idx], dominant_topic = dominant[idx], dominant_topic_share = dominant_share[idx],
         top_10_words_present = vapply(idx, function(i) sum(vocab[documents[[i]][1, ]] %in% top10_prob[k, ]), integer(1)), text = texts_stm[idx])
}
representative_documents <- bind_rows(lapply(topic_ids, function(k) { idx <- ft$index[[k]]; rep_row(k, seq_along(idx), idx, "findThoughts, n = 5, no restriction") }))
text_id   <- match(texts_stm, unique(texts_stm))
author_id <- match(docs_stm$author, unique(docs_stm$author))
cand <- which(docs_stm$n_tokens_stm >= representative_min_tokens)
representative_documents_tier2rule <- bind_rows(lapply(topic_ids, function(k) {
  o <- cand[order(-theta[cand, k], docs_stm$doc_key[cand])]
  o <- o[!duplicated(text_id[o]) & !duplicated(author_id[o])]
  idx <- head(o, n_representative)
  rep_row(k, seq_along(idx), idx, sprintf("Tier 2 rule: >= %d tokens, one per distinct text and author, ties by doc_key", representative_min_tokens))
}))
mark_stage("representative_documents")

# ---- Linkage verification after fitting ----
# Checks each theta row against its doc_key, source record, author, thread and parent, re-tokenises every
# sample_stride-th document, and compares join counts with the inventory.
begin_stage("linkage")
recompute_tokens <- function(text) {
  x <- stri_replace_all_regex(stri_replace_all_fixed(stri_trans_tolower(text), "\u2019", "'"), url_regex, " ", opts_regex = url_opts)
  t <- stri_extract_all_regex(x, token_regex, omit_no_match = TRUE)[[1]]
  t <- stri_replace_last_regex(t, "'s$", "")
  t <- t[stri_length(t) >= min_token_chars & !(t %in% stopword_list)]
  t[t %in% vocab]
}
sample_idx <- seq(1L, length(documents), by = sample_stride)
recount_matches <- vapply(sample_idx, function(i) {
  counts <- table(recompute_tokens(texts_stm[i]))
  d <- documents[[i]]
  length(counts) == ncol(d) && all(sort(names(counts)) == sort(vocab[d[1, ]])) && all(as.integer(counts[vocab[d[1, ]]]) == d[2, ])
}, logical(1))
source_keys <- c(comments$id, submissions$id)
src_match <- match(docs_stm$doc_id, if (TRUE) c(comments$id, submissions$id) else NULL)
comment_match <- match(docs_stm$doc_key[docs_stm$doc_type == "comment"], paste0("t1_", comments$id))
sub_match     <- match(docs_stm$doc_key[docs_stm$doc_type == "submission"], paste0("t3_", submissions$id))
stm_c <- docs_stm[docs_stm$doc_type == "comment", ]; stm_s <- docs_stm[docs_stm$doc_type == "submission", ]
parent_is_comment <- substr(stm_c$parent_id, 1, 3) == "t1_"
parent_in_sample_comment <- parent_is_comment & stm_c$parent_id %in% paste0("t1_", comments$id)
parent_in_stm <- parent_is_comment & stm_c$parent_id %in% docs_stm$doc_key
parent_is_submission_in_sample <- !parent_is_comment & stm_c$parent_id %in% paste0("t3_", submissions$id)
# The inventory's parent_linkage.csv rows for the "Tier 2 modelable comments" scope; a request naming both parent statuses returns their sum.
inv_parent <- function(record, modelable) {
  if (is.null(inventory_parent)) return(NA_real_)
  sel <- inventory_parent$scope == "Tier 2 modelable comments" & inventory_parent$parent_record == record &
    (is.na(inventory_parent$parent_modelable) == all(is.na(modelable))) & (all(is.na(modelable)) | inventory_parent$parent_modelable %in% modelable)
  v <- inventory_parent$n[sel]
  if (length(v)) sum(v) else NA_real_
}
# One linkage row; a documented value is compared only when it applies (the full document set), and only when it is a single value.
lk_row <- function(item, value, documented = NA, source = "", applies = TRUE) {
  documented <- if (length(documented) == 1L) documented else NA
  tibble(item = item, value = as.character(value), documented_value = as.character(documented), source = source,
         applies_to_this_run = applies, equal_to_documented = if (!isTRUE(applies) || is.na(documented)) NA else as.character(value) == as.character(documented))
}
full <- !quick_run
linkage_checks <- bind_rows(
  lk_row("theta rows", nrow(theta)), lk_row("STM documents (names(documents) = doc_key)", length(documents)),
  lk_row("theta row order equals the STM document order equals docs_stm$doc_key", identical(names(documents), docs_stm$doc_key)),
  lk_row("STM comments matched to exactly one source comment by doc_key", sum(!is.na(comment_match)), nrow(stm_c)),
  lk_row("STM submissions matched to exactly one source submission by doc_key", sum(!is.na(sub_match)), nrow(stm_s)),
  lk_row("distinct doc_key among STM documents", n_distinct(docs_stm$doc_key), nrow(docs_stm)),
  lk_row("STM comments whose author equals the source author", sum(stm_c$author == comments$author[comment_match]), nrow(stm_c)),
  lk_row("STM submissions whose author equals the source author", sum(stm_s$author == submissions$author[sub_match]), nrow(stm_s)),
  lk_row("STM comments whose link_id and parent_id equal the source values", sum(stm_c$link_id == comments$link_id[comment_match] & stm_c$parent_id == comments$parent_id[comment_match]), nrow(stm_c)),
  lk_row("STM comments whose created_utc equals the source value", sum(stm_c$created_utc == comments$created_utc[comment_match]), nrow(stm_c)),
  lk_row("STM submissions whose created_utc equals the source value", sum(stm_s$created_utc == submissions$created_utc[sub_match]), nrow(stm_s)),
  lk_row("STM documents re-tokenised from the stored text by a second code path (every 5,000th) that equal the stm document", sum(recount_matches), length(sample_idx)),
  lk_row("distinct authors, STM comments", n_distinct(stm_c$author), 133602, "metadata_inventory README section 4 (modelable comments)", full),
  lk_row("distinct authors, STM submissions", n_distinct(stm_s$author), 2308, "metadata_inventory README section 4 (modelable submissions)", full),
  lk_row("STM comments with '[deleted]' author", sum(stm_c$author == "[deleted]"), 0, "metadata_inventory README section 5", full),
  lk_row("STM submissions with '[deleted]' author", sum(stm_s$author == "[deleted]"), 337, "metadata_inventory README section 5", full),
  lk_row("STM comments whose thread submission is in the sample (link_id join)", sum(stm_c$submission_in_sample), 839545, "metadata_inventory README section 4", full),
  lk_row("STM comments whose thread submission is itself an STM document", sum(stm_c$link_id %in% stm_s$doc_key)),
  lk_row("distinct threads (link_id) among STM comments", n_distinct(stm_c$link_id), 11021, "metadata_inventory README section 5", full),
  lk_row("STM comments whose parent is a comment in the sample (parent_id join)", sum(parent_in_sample_comment), inv_parent("parent comment in the sample", c("parent comment modelable", "parent comment not modelable")), "metadata_inventory/tables/parent_linkage.csv (modelable in sample, both statuses summed)", full),
  lk_row("STM comments whose parent comment is itself an STM document", sum(parent_in_stm), inv_parent("parent comment in the sample", "parent comment modelable"), "metadata_inventory/tables/parent_linkage.csv", full),
  lk_row("STM comments whose parent is the submission, in the sample", sum(parent_is_submission_in_sample), inv_parent("parent is the submission, in the sample", NA), "metadata_inventory/tables/parent_linkage.csv", full),
  lk_row("STM comments whose parent comment is not in the sample", sum(parent_is_comment & !parent_in_sample_comment), inv_parent("parent comment not in the sample", NA), "metadata_inventory/tables/parent_linkage.csv", full),
  lk_row("STM comments whose parent is the submission, not in the sample", sum(!parent_is_comment & !parent_is_submission_in_sample), inv_parent("parent is the submission, not in the sample", NA), "metadata_inventory/tables/parent_linkage.csv", full),
  lk_row("STM documents with a non-missing score", sum(!is.na(docs_stm$score)), nrow(docs_stm)),
  lk_row("STM submissions with non-missing num_comments and upvote_ratio", sum(!is.na(stm_s$num_comments) & !is.na(stm_s$upvote_ratio)), nrow(stm_s)),
  lk_row("STM submissions with a non-empty url", sum(usable(stm_s$url)), 10382, "metadata_inventory README section 4", full))
mark_stage("linkage")
log_line("linkage:", sum(recount_matches), "of", length(sample_idx), "sampled documents re-tokenised identically")

# ---- Topic summary table ----
# One row per topic: prevalence, coherence, exclusivity, type effects, day-curve range, nearest topic, top words.
nearest_by_cor <- vapply(topic_ids, function(k) { v <- cor_theta[k, ]; v[k] <- -Inf; which.max(v) }, integer(1))
topic_summary <- tibble(topic = topic_ids, topic_label = topic_labels, prevalence_rank = prevalence_rank, expected_proportion = expected_proportion, token_share = token_share,
                        documents_dominant = tabulate(dominant, K), share_documents_dominant = tabulate(dominant, K) / nrow(theta),
                        median_share_when_dominant = vapply(topic_ids, function(k) if (any(dominant == k)) median(dominant_share[dominant == k]) else NA_real_, numeric(1)),
                        mean_theta_comments = topic_prevalence_by_type$mean_theta[topic_prevalence_by_type$doc_type == "comment"],
                        mean_theta_submissions = topic_prevalence_by_type$mean_theta[topic_prevalence_by_type$doc_type == "submission"],
                        semantic_coherence_stm = coherence_stm, coherence_npmi_top10 = coherence_npmi, exclusivity = excl,
                        nearest_topic_by_document_correlation = nearest_by_cor, nearest_topic_correlation = cor_theta[cbind(topic_ids, nearest_by_cor)]) |>
  left_join(effect_doc_type |> select(topic, expected_comment, expected_submission, difference_submission_minus_comment, difference_ci_low, difference_ci_high, coefficient_p), by = "topic") |>
  left_join(effect_day_range |> filter(doc_type == "comment") |> transmute(topic, comment_curve_min = curve_min, comment_curve_max = curve_max, comment_curve_range = curve_range, comment_day_of_max = day_of_max), by = "topic") |>
  left_join(top_author_by_topic, by = "topic") |>
  mutate(top_10_words_by_probability = apply(labels$prob[, 1:10, drop = FALSE], 1, paste, collapse = ", "),
         top_10_words_by_frex = apply(labels$frex[, 1:10, drop = FALSE], 1, paste, collapse = ", "))
model_diagnostics <- tibble(
  item = c("residual dispersion (Taddy 2012; 1 under the model)", "residual dispersion p-value (H0: dispersion = 1)", "residual degrees of freedom",
           "semantic coherence (stm): mean / median / min / max", "NPMI coherence (top 10, all STM documents): mean / median / min / max", "exclusivity (stm, frexw 0.7): mean / median / min / max",
           "expected proportion: max / min topic", "token share: max / min topic", "topics dominant in fewer than 1% of documents", "median dominant-topic share (all documents)",
           "share of documents with dominant share >= 0.5", "mean active topics per document (share >= 0.1)", "largest absolute document-correlation between two topics",
           "positive-correlation edges above cutoff (topicCorr simple)", "EM iterations / converged", "final bound per token", "fit wall seconds"),
  value = c(sprintf("%.4f", resid$dispersion), format(resid$pvalue, digits = 4), sprintf("%.0f", resid$df),
            sprintf("%.3f / %.3f / %.3f / %.3f", mean(coherence_stm), median(coherence_stm), min(coherence_stm), max(coherence_stm)),
            sprintf("%.4f / %.4f / %.4f / %.4f", mean(coherence_npmi), median(coherence_npmi), min(coherence_npmi), max(coherence_npmi)),
            sprintf("%.3f / %.3f / %.3f / %.3f", mean(excl), median(excl), min(excl), max(excl)),
            sprintf("%.4f (Topic %d) / %.4f (Topic %d)", max(expected_proportion), which.max(expected_proportion), min(expected_proportion), which.min(expected_proportion)),
            sprintf("%.4f (Topic %d) / %.4f (Topic %d)", max(token_share), which.max(token_share), min(token_share), which.min(token_share)),
            sum(tabulate(dominant, K) / nrow(theta) < 0.01), sprintf("%.4f", median(dominant_share)), sprintf("%.4f", mean(dominant_share >= 0.5)), sprintf("%.3f", mean(n_active)),
            sprintf("%.4f", max(abs(cor_theta[upper.tri(cor_theta)]))), nrow(topic_correlation_edges), sprintf("%d / %s", its, model$convergence$converged),
            sprintf("%.6f", tail(bound, 1) / n_tokens_total), round(fit_record$wall_seconds, 1)))

# ---- Figures ----
# PNG figures with titles, axis labels and legends where needed.
begin_stage("figures")
ink_series <- "#2a78d6"; ink_series_2 <- "#eb6834"; ink_red <- "#e34948"; ink_text <- "#0b0b0b"; ink_sub <- "#52514e"; ink_muted <- "#898781"
grid_col <- "#e1e0d9"; axis_col <- "#c3c2b7"; mid_grey <- "#f0efec"
blue_ramp <- c("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")
theme_fig <- theme_minimal(base_size = 12) +
  theme(panel.grid.minor = element_blank(), panel.grid.major = element_line(colour = grid_col, linewidth = 0.3),
        axis.line.x = element_line(colour = axis_col, linewidth = 0.4), axis.text = element_text(colour = ink_muted), axis.title = element_text(colour = ink_sub),
        plot.title = element_text(colour = ink_text, face = "bold"), strip.text = element_text(colour = ink_sub, face = "bold", hjust = 0),
        legend.title = element_text(colour = ink_sub), legend.text = element_text(colour = ink_muted), plot.background = element_rect(fill = "white", colour = NA))
type_colours <- c(comment = ink_series, submission = ink_series_2)
save_fig <- function(name, plot, width, height, dpi = 150) ggsave(file.path(fig_dir, name), plot, width = width, height = height, dpi = dpi, bg = "white", limitsize = FALSE)
title_suffix <- sprintf("STM K = %d pilot%s - r/politics, July 2026", K, if (quick_run) " (quick run)" else "")
fig_title <- function(main) paste0(main, "\n", title_suffix)
topic_levels_by_rank <- topic_labels[order(prevalence_rank)]

fig1 <- ggplot(topic_summary, aes(x = semantic_coherence_stm, y = exclusivity)) +
  geom_point(colour = ink_series, size = 2.2) + geom_text(aes(label = topic), colour = ink_sub, size = 3, nudge_y = 0.06) +
  labs(title = fig_title("Semantic coherence and exclusivity per topic"), x = "Semantic coherence (stm, top 10 words)", y = "Exclusivity (stm, top 10 words, frexw = 0.7)") + theme_fig
save_fig("fig1_coherence_exclusivity_stm_K040.png", fig1, 8, 6)

prev_long <- topic_summary |> select(topic_label, `Expected proportion (mean theta)` = expected_proportion, `Share of tokens (token-weighted theta)` = token_share) |>
  pivot_longer(-topic_label, names_to = "measure", values_to = "share") |> mutate(topic_label = factor(topic_label, levels = rev(topic_levels_by_rank)))
fig2 <- ggplot(prev_long, aes(x = share, y = topic_label, fill = measure)) + geom_col(position = position_dodge(width = 0.8), width = 0.7) +
  scale_fill_manual(values = c(ink_series, ink_series_2), name = "Measure") + scale_x_continuous(labels = label_percent(accuracy = 0.1), expand = expansion(mult = c(0, 0.05))) +
  labs(title = fig_title("Topic prevalence"), x = "Share", y = NULL) + theme_fig + theme(legend.position = "bottom", panel.grid.major.y = element_blank())
save_fig("fig2_topic_prevalence_stm_K040.png", fig2, 9, 3 + 0.22 * K)

fig3 <- topic_prevalence_by_day |> mutate(topic_label = factor(sprintf("Topic %d", topic), levels = topic_labels)) |>
  ggplot(aes(x = created_date_utc, y = mean_theta, colour = doc_type)) + geom_line(linewidth = 0.5) +
  facet_wrap(~ topic_label, ncol = 8, scales = "free_y") + scale_colour_manual(values = type_colours, name = "Document type") +
  scale_x_date(date_breaks = "10 days", date_labels = "%b %d") + scale_y_continuous(labels = label_percent(accuracy = 0.1)) +
  labs(title = fig_title("Observed mean topic proportion by UTC day and document type"), x = "UTC day, July 2026", y = "Mean theta") +
  theme_fig + theme(legend.position = "bottom", axis.text.x = element_text(size = 7), axis.text.y = element_text(size = 7))
save_fig("fig3_topic_prevalence_by_day_observed_stm_K040.png", fig3, 16, 2.2 * ceiling(K / 8) + 1.5, dpi = 130)

fig4 <- effect_day |> mutate(topic_label = factor(sprintf("Topic %d", topic), levels = topic_labels)) |>
  ggplot(aes(x = date_utc, y = expected_proportion, colour = doc_type, fill = doc_type)) +
  geom_ribbon(aes(ymin = ci_low, ymax = ci_high), alpha = 0.18, colour = NA) + geom_line(linewidth = 0.5) +
  facet_wrap(~ topic_label, ncol = 8, scales = "free_y") + scale_colour_manual(values = type_colours, name = "Document type") + scale_fill_manual(values = type_colours, name = "Document type") +
  scale_x_datetime(date_breaks = "10 days", date_labels = "%b %d") + scale_y_continuous(labels = label_percent(accuracy = 0.1)) +
  labs(title = fig_title("Expected topic proportion through July by document type (estimateEffect, 95% intervals)"), x = "UTC day, July 2026", y = "Expected topic proportion") +
  theme_fig + theme(legend.position = "bottom", axis.text.x = element_text(size = 7), axis.text.y = element_text(size = 7))
save_fig("fig4_topic_prevalence_by_day_fitted_stm_K040.png", fig4, 16, 2.2 * ceiling(K / 8) + 1.5, dpi = 130)

fig5 <- effect_doc_type |> mutate(topic_label = factor(sprintf("Topic %d", topic), levels = sprintf("Topic %d", order(effect_doc_type$difference_submission_minus_comment)))) |>
  ggplot(aes(x = difference_submission_minus_comment, y = topic_label)) +
  geom_vline(xintercept = 0, colour = axis_col, linewidth = 0.4) + geom_errorbar(aes(xmin = difference_ci_low, xmax = difference_ci_high), width = 0, orientation = "y", colour = ink_sub, linewidth = 0.4) +
  geom_point(colour = ink_series, size = 2) + scale_x_continuous(labels = label_percent(accuracy = 0.1)) +
  labs(title = fig_title("Submission minus comment expected proportion at the median day (95% intervals)"), x = "Difference in expected proportion (submission minus comment)", y = NULL) +
  theme_fig + theme(panel.grid.major.y = element_blank())
save_fig("fig5_doc_type_difference_stm_K040.png", fig5, 9, 3 + 0.22 * K)

fig6a <- ggplot(em_iterations, aes(x = iteration, y = bound_per_token)) + geom_line(colour = ink_series) + geom_point(colour = ink_series, size = 1.4) +
  labs(title = fig_title("Approximate bound per token by EM iteration"), x = "EM iteration", y = "Approximate variational bound per token") + theme_fig
fig6b <- ggplot(em_iterations |> filter(!is.na(relative_change), relative_change > 0), aes(x = iteration, y = relative_change)) + geom_line(colour = ink_series_2) + geom_point(colour = ink_series_2, size = 1.4) +
  scale_y_log10(labels = label_scientific()) + labs(title = "Relative change of the bound by EM iteration (log axis; positive changes)", x = "EM iteration", y = "Relative change") + theme_fig
save_fig("fig6_em_convergence_stm_K040.png", fig6a / fig6b, 9, 8)

tw <- top_words |> filter(label_type == "prob", rank <= 10) |> arrange(topic, rank) |>
  mutate(topic_label = factor(sprintf("Topic %d", topic), levels = topic_labels), key = paste0(term, "  #", topic)) |> mutate(key = factor(key, levels = rev(unique(key))))
fig_cols <- 8
fig7 <- ggplot(tw, aes(x = probability_in_topic, y = key)) + geom_col(fill = ink_series, width = 0.7) + facet_wrap(~ topic_label, scales = "free", ncol = fig_cols) +
  scale_y_discrete(labels = function(x) sub("  #.*$", "", x)) + scale_x_continuous(labels = label_number(accuracy = 0.001), n.breaks = 3, expand = expansion(mult = c(0, 0.05))) +
  labs(title = fig_title("Highest-probability words per topic"), x = "Probability of the word in the topic", y = NULL) +
  theme_fig + theme(panel.grid.major.y = element_blank(), axis.text.y = element_text(size = 8), axis.text.x = element_text(size = 7))
save_fig("fig7_top_words_stm_K040.png", fig7, 2.6 * fig_cols + 1, 2.2 * ceiling(K / fig_cols) + 1, dpi = 130)

cor_long <- mat_tibble(cor_theta, as.character(topic_ids)) |> mutate(topic_a = topic_ids) |> pivot_longer(-topic_a, names_to = "topic_b", values_to = "correlation") |>
  mutate(topic_b = as.integer(topic_b), value = if_else(topic_a == topic_b, NA_real_, correlation))
cor_limit <- max(abs(cor_long$value), na.rm = TRUE)
fig8 <- ggplot(cor_long, aes(x = factor(topic_b, levels = topic_ids), y = factor(topic_a, levels = rev(topic_ids)), fill = value)) + geom_tile() +
  scale_fill_gradient2(low = ink_series, mid = mid_grey, high = ink_red, midpoint = 0, limits = c(-cor_limit, cor_limit), name = "Pearson correlation of document proportions", na.value = "white") +
  labs(title = fig_title("Topic correlation over documents (theta), diagonal blank"), x = "Topic", y = "Topic") +
  theme_fig + theme(legend.position = "bottom", panel.grid.major = element_blank(), axis.text = element_text(size = 7))
save_fig("fig8_topic_correlation_stm_K040.png", fig8, 11, 11)

fig9 <- ggplot(docs_stm, aes(x = token_band, y = dominant_topic_share)) + geom_boxplot(fill = ink_series, colour = ink_sub, outlier.size = 0.4, outlier.alpha = 0.2, width = 0.6) +
  facet_wrap(~ doc_type, ncol = 2) + scale_y_continuous(labels = label_percent(accuracy = 1), limits = c(0, 1)) +
  labs(title = fig_title("Share of the dominant topic within documents by length band"), x = "Tokens in the document (STM representation)", y = "Share of the dominant topic") + theme_fig
save_fig("fig9_document_concentration_stm_K040.png", fig9, 10, 5)

fig11 <- ggplot(documents_per_day, aes(x = created_date_utc, y = stm_documents, colour = doc_type)) + geom_line(linewidth = 0.7) + geom_point(size = 1.6) +
  scale_colour_manual(values = type_colours, name = "Document type") + facet_wrap(~ doc_type, ncol = 1, scales = "free_y") +
  scale_y_continuous(labels = label_comma(), limits = c(0, NA), expand = expansion(mult = c(0, 0.05))) + scale_x_date(date_breaks = "5 days", date_labels = "%b %d") +
  labs(title = fig_title("STM documents per UTC day of created_utc"), x = "UTC date, July 2026", y = "STM documents") + theme_fig + theme(legend.position = "bottom")
save_fig("fig11_documents_per_day_stm_K040.png", fig11, 9, 6)
mark_stage("figures")

# ---- Stop the sampler; runtime and memory tables and figure ----
# Stops the sampler and builds the stage timings, resource trace and runtime figure.
if (sampler_started) { writeLines("stop", stop_file); Sys.sleep(sampler_interval + 2) }
read_trace <- function(p) if (!is.null(p) && file.exists(p)) read_csv(p, show_col_types = FALSE, progress = FALSE, col_types = cols(time_utc = col_character(), stage = col_character(), .default = col_double())) else NULL
resource_trace <- read_trace(trace_csv)
resource_trace_fit <- if (fit_from_cache) read_trace(fit_record$trace_csv) else resource_trace
stage_timings <- bind_rows(stage_log) |>
  mutate(start_utc = format(start_utc, "%Y-%m-%dT%H:%M:%OS3Z", tz = "UTC"), end_utc = format(end_utc, "%Y-%m-%dT%H:%M:%OS3Z", tz = "UTC"), seconds = round(seconds, 3))
stage_group <- c(read = "input", document_table = "input", tokenize = "input", vocabulary_dtm = "input", split = "input", reconciliation = "input", stm_input = "input",
                 fit = "fit", model_outputs = "diagnostics", diagnostics_labels = "diagnostics", diagnostics_coherence = "diagnostics", diagnostics_residuals = "diagnostics",
                 diagnostics_correlation = "diagnostics", prevalence = "diagnostics", effects = "effects", representative_documents = "diagnostics", linkage = "diagnostics",
                 figures = "outputs", validation = "outputs", tables = "outputs", manifests = "outputs")
trace_for_fig <- if (!is.null(resource_trace_fit)) resource_trace_fit |> mutate(time = as.POSIXct(time_utc, format = "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
                                                                              minutes = as.numeric(difftime(time, min(time), units = "mins")),
                                                                              stage_group = factor(coalesce(unname(stage_group[stage]), "other"), levels = c("input", "fit", "diagnostics", "effects", "outputs", "other"))) else NULL
fig10a <- ggplot(em_iterations, aes(x = iteration, y = estep_seconds)) + geom_line(colour = ink_series) + geom_point(colour = ink_series, size = 1.4) +
  scale_y_continuous(limits = c(0, NA)) + labs(title = fig_title("E-step wall seconds by EM iteration"), x = "EM iteration", y = "E-step seconds") + theme_fig
if (!is.null(trace_for_fig)) {
  group_colours <- c(input = ink_muted, fit = ink_series, diagnostics = ink_series_2, effects = "#1baf7a", outputs = ink_sub, other = mid_grey)
  fig10b <- ggplot(trace_for_fig, aes(x = minutes, y = rss_mb / 1024, colour = stage_group)) + geom_point(size = 0.9) +
    scale_colour_manual(values = group_colours, name = "Stage") + scale_y_continuous(limits = c(0, NA)) +
    labs(title = "Working set of the R process through the run (fitting run)", x = "Minutes since the sampler started", y = "Working set (GB)") + theme_fig + theme(legend.position = "bottom")
  save_fig("fig10_runtime_memory_stm_K040.png", fig10a / fig10b, 9, 8)
} else save_fig("fig10_runtime_memory_stm_K040.png", fig10a, 9, 4)

# ---- Validation ----
# Checks that must hold before tables are written; the script stops after validation_checks.csv if any fails.
begin_stage("validation")
checks <- list()
add_check <- function(check, detail, pass) checks[[length(checks) + 1L]] <<- tibble(check = check, detail = detail, pass = isTRUE(pass))
add_check("source files read: 7 comment files and 1 submission file, rows equal the earlier counts", sprintf("%d + %d rows", n_comments_all, n_submissions_all), n_comments_all == 891397 && n_submissions_all == 10747)
add_check("every earlier output read matches the SHA256 in its manifest", sprintf("%d of %d existing files", sum(input_integrity$hash_equals_manifest), sum(input_integrity$exists)), all(input_integrity$hash_equals_manifest[input_integrity$exists]))
add_check("Tier 2 representation reproduced: every reconciliation item equal", sprintf("%d of %d equal", sum(tier2_reconciliation$equal), nrow(tier2_reconciliation)), all(tier2_reconciliation$equal))
add_check("STM documents: every entering document survives the conversion and every token is kept",
          sprintf("%d in, %d out; tokens %d in, %d out", n_stm_in, length(documents), tokens_stm_in, sum(doc_tokens_stm)), length(documents) == n_stm_in && sum(doc_tokens_stm) == tokens_stm_in)
add_check("STM documents: per-document token sums equal the Tier 2 n_tokens_modeled", sprintf("%d of %d", sum(doc_tokens_stm == docs_stm$n_tokens_modeled), length(documents)), all(doc_tokens_stm == docs_stm$n_tokens_modeled))
add_check("STM vocabulary: term indices sequential 1..V, every term used, V equals the Tier 2 vocabulary (full run)",
          sprintf("V = %d; Tier 2 V = %d; unseen dropped = %d", length(vocab), n_vocabulary, terms_unseen),
          length(vocab) == model$settings$dim$V && (quick_run || (length(vocab) == n_vocabulary && terms_unseen == 0)))
add_check("STM documents: no empty document, no duplicate term within a document, integer counts",
          sprintf("min terms %d", min(doc_terms_stm)), min(doc_terms_stm) >= 1 && !any(vapply(documents, function(d) anyDuplicated(d[1, ]) > 0, logical(1))) && all(vapply(documents, function(d) is.integer(d), logical(1))))
basis_values <- unclass(spline_basis); attributes(basis_values) <- attributes(basis_values)["dim"]
basis_max_diff <- max(abs(basis_values - model$settings$covariates$X[, 3:(2 + spline_df)]))
add_check("covariates: no missing value; day_of_july within [0, 31); doc_type has both levels; design matrix has 12 columns and my basis reproduces stm's",
          sprintf("day range %.5f to %.5f; X %d x %d; max |basis difference| %.3g", min(meta_df$day_of_july), max(meta_df$day_of_july), nrow(model$settings$covariates$X), ncol(model$settings$covariates$X), basis_max_diff),
          !anyNA(meta_df) && min(meta_df$day_of_july) >= 0 && max(meta_df$day_of_july) < 31 && nlevels(droplevels(meta_df$doc_type)) == 2 &&
            ncol(model$settings$covariates$X) == 2L + spline_df && basis_max_diff < 1e-10)
add_check("model dimensions: K, N, V as specified; theta rows sum to 1; beta rows sum to 1; no NA",
          sprintf("K %d N %d V %d", model$settings$dim$K, model$settings$dim$N, model$settings$dim$V),
          model$settings$dim$K == K && model$settings$dim$N == length(documents) && max(abs(rowSums(theta) - 1)) < 1e-8 && max(abs(rowSums(beta) - 1)) < 1e-8 && !anyNA(theta) && !anyNA(beta))
add_check("EM record: bound trace length equals iterations; E-step seconds parsed for every iteration; iterations within the ceiling",
          sprintf("%d iterations; %d E-step timings; ceiling %d", its, sum(!is.na(em_iterations$estep_seconds)), max_em_its), length(bound) == its && all(!is.na(em_iterations$estep_seconds)) && its <= max_em_its)
add_check("run key: the cached model (if used) was fitted on this input and these settings", sprintf("from cache %s", fit_from_cache), !fit_from_cache || identical(fit_record$run_key, run_key))
add_check("linkage: theta row order equals the document order equals doc_key; every STM document matches exactly one source record with equal author and identifiers",
          paste(linkage_checks$value[3:11], collapse = " / "), identical(names(documents), docs_stm$doc_key) && all(!is.na(comment_match)) && all(!is.na(sub_match)) &&
            n_distinct(docs_stm$doc_key) == nrow(docs_stm) && all(stm_c$author == comments$author[comment_match]) && all(stm_s$author == submissions$author[sub_match]) &&
            all(stm_c$link_id == comments$link_id[comment_match] & stm_c$parent_id == comments$parent_id[comment_match]))
add_check("linkage: every sampled STM document equals a second tokenisation of its stored text", sprintf("%d of %d", sum(recount_matches), length(sample_idx)), all(recount_matches))
add_check("linkage: documented inventory values reproduced where they apply", sprintf("%d of %d applicable items equal", sum(linkage_checks$equal_to_documented, na.rm = TRUE), sum(!is.na(linkage_checks$equal_to_documented))),
          all(linkage_checks$equal_to_documented[!is.na(linkage_checks$equal_to_documented)]))
add_check("diagnostics: coherence, exclusivity and NPMI have one finite value per topic; residual dispersion finite", sprintf("dispersion %.3f", resid$dispersion),
          length(coherence_stm) == K && length(excl) == K && all(is.finite(c(coherence_stm, excl, coherence_npmi))) && is.finite(resid$dispersion))
add_check("effects: estimateEffect covers every topic; the day grid has the requested points for both types; differences have intervals",
          sprintf("%d topics; %d grid rows", length(prep$topics), nrow(effect_day)), length(prep$topics) == K && nrow(effect_day) == 2 * K * effect_npoints && all(is.finite(effect_doc_type$difference_ci_low)))
add_check("representative documents: five per topic under both rules; texts equal the stored source text; Tier 2 rule respects the token floor and the distinct text/author rule",
          sprintf("%d + %d rows", nrow(representative_documents), nrow(representative_documents_tier2rule)),
          nrow(representative_documents) == K * n_representative && nrow(representative_documents_tier2rule) == K * n_representative &&
            all(representative_documents$text == docs$text[modeled_rows][representative_documents$stm_row]) && min(representative_documents_tier2rule$n_tokens_stm) >= representative_min_tokens &&
            all(tapply(representative_documents_tier2rule$text, representative_documents_tier2rule$topic, n_distinct) == n_representative) &&
            all(tapply(representative_documents_tier2rule$author, representative_documents_tier2rule$topic, n_distinct) == n_representative))
add_check("prevalence tables: expected proportions sum to 1; by-day rows cover every type x day; dominant counts sum to N",
          sprintf("sum %.6f; %d day rows", sum(expected_proportion), nrow(topic_prevalence_by_day)), abs(sum(expected_proportion) - 1) < 1e-8 && sum(topic_summary$documents_dominant) == nrow(theta) &&
            nrow(topic_prevalence_by_day) == K * n_distinct(day_key))
add_check("memory sampler: a trace with at least one row exists for the fitting run (or the sampler was disabled)", sprintf("%s rows", if (is.null(resource_trace_fit)) "0" else nrow(resource_trace_fit)),
          !memory_sampler_on || (!is.null(resource_trace_fit) && nrow(resource_trace_fit) >= 1))
validation_checks <- bind_rows(checks)
mark_stage("validation")
log_line("validation:", sum(validation_checks$pass), "of", nrow(validation_checks), "checks pass")

# ---- Write tables ----
# One CSV per table; stops first if a blocking validation check failed.
begin_stage("tables")
table_dims <- list()
write_table <- function(tbl, name) {
  path <- file.path(tab_dir, name)
  write_csv(tbl, path)
  table_dims[[path]] <<- c(rows = nrow(tbl), cols = ncol(tbl))
  invisible(path)
}
write_table(validation_checks, "validation_checks.csv")
stopifnot(all(validation_checks$pass))
parameters_table <- tibble(
  parameter = c("K", "prevalence_formula", "content_covariate", "time_representation", "spline_df", "spline_degree", "spline_interior_knots", "spline_boundary_knots", "day0_epoch_utc",
                "init_type", "stm_seed", "max_em_its", "emtol", "gamma_prior", "sigma_prior", "ngroups", "effect_nsims", "effect_plot_nsims", "effect_npoints", "effect_seed",
                "coherence_M", "exclusivity_frexw", "label_n_words", "frex_weight", "correlation_cutoff", "n_representative", "representative_min_tokens", "active_topic_threshold",
                "token_bands", "sample_stride", "text_representation", "stopword_source", "stopword_workbook_sha256", "min_token_chars", "min_document_frequency", "max_document_share",
                "heldout_share (Tier 2 split, carried only)", "split_seed (Tier 2)", "documents_entering_stm", "quick_run", "quick_n", "memory_sampler", "sampler_interval_seconds", "fit_from_cache", "fitting_run_id"),
  value = c(K, paste(deparse(prevalence_formula), collapse = " "), "none", sprintf("day_of_july = (created_utc - %d) / 86400, continuous fractional day in [0, 31)", as.integer(day0)), spline_df, spline_degree,
            paste(sprintf("%.4f", spline_knots), collapse = "; "), paste(sprintf("%.6f", spline_boundary), collapse = "; "), as.integer(day0), init_type, stm_seed, max_em_its, format(emtol, scientific = TRUE),
            gamma_prior, sigma_prior, 1L, effect_nsims, effect_plot_nsims, effect_npoints, effect_seed, coherence_M, exclusivity_frexw, label_n_words, frex_weight, correlation_cutoff, n_representative,
            representative_min_tokens, active_topic_threshold, paste(band_labels, collapse = "; "), sample_stride,
            "Tier 2 LDA representation: lower case; U+2019 -> '; URLs removed; Unicode letter-run tokens with internal apostrophes; trailing 's stripped; tokens < 2 chars removed; 310 Tier 2 entries removed; marker-only documents excluded; terms kept when 5 <= df <= 50% of non-marker documents; no stemming",
            stopword_source, workbook_hash, min_token_chars, min_document_frequency, max_document_share, heldout_share, split_seed, "all Tier 2 modelable documents (no split)", quick_run, if (quick_run) quick_n else NA,
            memory_sampler_on, sampler_interval, fit_from_cache, fit_record$run_id))
write_table(parameters_table,            "parameters.csv")
write_table(input_integrity,             "input_integrity.csv")
write_table(source_files |> select(-path), "source_files.csv")
write_table(tier2_reconciliation,        "tier2_reconciliation.csv")
write_table(discrepancies,               "discrepancies.csv")
write_table(stm_input_accounting,        "stm_input_accounting.csv")
write_table(covariate_summary,           "covariate_summary.csv")
write_table(metadata_coverage,           "metadata_coverage_stm.csv")
write_table(documents_per_day,           "documents_per_day_stm.csv")
write_table(document_length_summary_stm, "document_length_summary_stm.csv")
write_table(model_specification,         "model_specification_stm_K040.csv")
write_table(em_iterations,               "em_iterations_stm_K040.csv")
write_table(model_diagnostics,           "model_diagnostics_stm_K040.csv")
write_table(topic_summary,               "topic_summary_stm_K040.csv")
write_table(top_words,                   "top_words_stm_K040.csv")
write_table(topic_prevalence_by_type,    "topic_prevalence_by_type_stm_K040.csv")
write_table(topic_prevalence_by_day,     "topic_prevalence_by_day_stm_K040.csv")
write_table(effect_doc_type,             "topic_effect_doc_type_stm_K040.csv")
write_table(effect_day,                  "topic_effect_day_stm_K040.csv")
write_table(effect_day_range,            "topic_effect_day_range_stm_K040.csv")
write_table(effect_coefficients,         "topic_effect_coefficients_stm_K040.csv")
write_table(topic_correlation,           "topic_correlation_stm_K040.csv")
write_table(topic_correlation_edges,     "topic_correlation_positive_edges_stm_K040.csv")
write_table(sigma_table,                 "topic_sigma_correlation_stm_K040.csv")
write_table(document_concentration,      "document_concentration_stm_K040.csv")
write_table(representative_documents,    "representative_documents_stm_K040.csv")
write_table(representative_documents_tier2rule, "representative_documents_tier2rule_stm_K040.csv")
write_table(linkage_checks,              "linkage_checks_stm_K040.csv")
topic_word_table <- tibble(term = vocab, stm_index = seq_along(vocab), wordcount = model$settings$dim$wcounts$x, document_frequency_stm = as.integer(term_df_stm)) |>
  bind_cols(mat_tibble(t(beta), topic_cols))
write_table(topic_word_table,            "topic_word_distribution_stm_K040.csv")
document_topic_table <- docs_stm |>
  transmute(stm_row, doc_key, doc_id, doc_type, tier2_split = split, author, created_utc, created_date_utc = as.character(created_date_utc), day_of_july, link_id, parent_id,
            submission_id, submission_in_sample, n_tokens_stm, dominant_topic, dominant_topic_share = round(dominant_topic_share, 6), n_active_topics) |>
  bind_cols(mat_tibble(round(theta, 6), topic_cols))
write_table(document_topic_table,        "document_topic_distribution_stm_K040.csv")
rm(document_topic_table)
document_linkage_table <- docs_stm |>
  transmute(stm_row, doc_key, doc_id, doc_type, source_file, author, created_utc, created_date_utc = as.character(created_date_utc), link_id, parent_id, submission_id, submission_in_sample,
            parent_kind = if_else(doc_type == "comment", if_else(substr(parent_id, 1, 3) == "t1_", "comment", "submission"), NA_character_),
            parent_in_sample = if_else(doc_type == "comment", (substr(parent_id, 1, 3) == "t1_" & parent_id %in% paste0("t1_", comments$id)) | (substr(parent_id, 1, 3) == "t3_" & parent_id %in% paste0("t3_", submissions$id)), NA),
            parent_stm_row = if_else(doc_type == "comment", match(parent_id, doc_key), NA_integer_),
            score, num_comments, upvote_ratio, url, text_source_fields, n_chars, n_tokens_letters, n_tokens_modeled, n_tokens_stm, tier2_split = split, model_status)
write_table(document_linkage_table,      "document_linkage_stm_K040.csv")
rm(document_linkage_table)
if (!is.null(resource_trace)) write_table(resource_trace, "resource_trace.csv")
if (fit_from_cache && !is.null(resource_trace_fit)) write_table(resource_trace_fit, "resource_trace_fitting_run.csv")
mark_stage("tables")

run_end <- Sys.time()
mem_end <- process_memory()
stage_timings <- bind_rows(stage_log) |>
  mutate(start_utc = format(start_utc, "%Y-%m-%dT%H:%M:%OS3Z", tz = "UTC"), end_utc = format(end_utc, "%Y-%m-%dT%H:%M:%OS3Z", tz = "UTC"), seconds = round(seconds, 3),
         stage_group = coalesce(unname(stage_group[stage]), "other"),
         computed_seconds = case_when(stage == "fit" & fit_from_cache ~ round(fit_record$wall_seconds, 1),
                                      stage == "diagnostics_coherence" & diag_from_cache ~ round(seconds_coherence, 1),
                                      stage == "diagnostics_residuals" & diag_from_cache ~ round(seconds_residuals, 1),
                                      stage == "effects" & diag_from_cache ~ round(seconds_effects_estimate, 1),
                                      TRUE ~ seconds),
         note = case_when(stage == "fit" & fit_from_cache ~ sprintf("model loaded from the cache; the fitting run %s took the computed seconds", fit_record$run_id),
                          stage %in% c("diagnostics_coherence", "diagnostics_residuals", "effects") & diag_from_cache ~ sprintf("loaded from the diagnostics cache computed in run %s (estimate stage only for effects)", diag_cache_run_id),
                          TRUE ~ ""))
run_info <- tibble(
  item = c("run_id", "run_start_utc", "run_end_utc", "total_seconds", "seconds_before_run_start (R start-up and package loading)", "quick_run", "fit_from_cache", "fitting_run_id",
           "fit_wall_seconds", "fit_cpu_user_seconds", "fit_cpu_system_seconds", "em_iterations", "converged", "mean_estep_seconds", "process_peak_working_set_mb (this run)",
           "process_peak_working_set_mb_after_fit (fitting run)", "sampler_max_rss_mb (fitting run)", "sampler_min_system_avail_mb (fitting run)", "model_file_bytes", "stm_input_file_bytes",
           "cpu_logical_processors", "cpu_physical_cores", "R_version", "stm_version", "documents", "vocabulary", "tokens", "git_commit_at_run", "script_sha256"),
  value = c(run_id, format(run_start, "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"), format(run_end, "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"), sprintf("%.1f", as.numeric(difftime(run_end, run_start, units = "secs"))),
            sprintf("%.1f", startup_elapsed_seconds), quick_run, fit_from_cache, fit_record$run_id, sprintf("%.1f", fit_record$wall_seconds), sprintf("%.1f", fit_record$cpu_user_seconds),
            sprintf("%.1f", fit_record$cpu_system_seconds), its, model$convergence$converged, sprintf("%.1f", mean(em_iterations$estep_seconds, na.rm = TRUE)), round(mem_end[["peak_wset_mb"]]),
            round(fit_record$peak_wset_mb_after_fit), if (is.null(resource_trace_fit)) NA else round(max(resource_trace_fit$rss_mb)), if (is.null(resource_trace_fit)) NA else round(min(resource_trace_fit$system_avail_mb)),
            file.size(model_path), file.size(stm_input_path), parallel::detectCores(logical = TRUE), parallel::detectCores(logical = FALSE), R.version.string, as.character(packageVersion("stm")),
            length(documents), length(vocab), sum(doc_tokens_stm), tryCatch(system2("git", c("-C", project_dir, "rev-parse", "--short", "HEAD"), stdout = TRUE, stderr = FALSE), error = function(e) ""), sha256(script_path)))
write_table(stage_timings, "stage_timings.csv")
write_table(run_info, "run_info.csv")
preservation_status <- tibble(
  item = c("tables (CSV)", "document_topic_distribution_stm_K040.csv", "document_linkage_stm_K040.csv", "figures (PNG)", "models/stm_K040_pilot.rds", "models/stm_input_K040_pilot.rds",
           "models/estimate_effect_K040.rds", "models/fit_record_K040.rds", "models/diagnostics_cache_K040.rds", "models/fit_console_K040.log, run_console.log, resource traces, run_key.txt, memory_sampler.R", "script"),
  status = c("tracked", "git-ignored (large; manifest tracked)", "git-ignored (large; manifest tracked)", "tracked", "git-ignored (model object; manifest tracked)", "git-ignored (stm documents, vocab, covariates; manifest tracked)",
             "git-ignored (estimateEffect object; manifest tracked)", "git-ignored (timings, CPU, memory, console log of the fitting run; manifest tracked)", "git-ignored (coherence, exclusivity, residual dispersion; manifest tracked)", "git-ignored", "tracked"),
  bytes = c(sum(file.size(list.files(tab_dir, pattern = "\\.csv$", full.names = TRUE))), file.size(file.path(tab_dir, "document_topic_distribution_stm_K040.csv")), file.size(file.path(tab_dir, "document_linkage_stm_K040.csv")),
            sum(file.size(list.files(fig_dir, pattern = "\\.png$", full.names = TRUE))), file.size(model_path), file.size(stm_input_path), file.size(effect_path), file.size(fit_record_path), file.size(diag_cache_path),
            sum(file.size(list.files(model_dir, pattern = "\\.(log|csv|txt|R)$", full.names = TRUE))), file.size(script_path)))
write_table(preservation_status, "preservation_status.csv")

# ---- Provenance sidecars ----
# One <output>.manifest.yml per output with input, output and script hashes, parameters and package versions.
begin_stage("manifests")
git_commit <- tryCatch(system2("git", c("-C", project_dir, "rev-parse", "--short", "HEAD"), stdout = TRUE, stderr = FALSE), error = function(e) "")
if (length(git_commit) == 0L) git_commit <- ""
manifest_packages <- c("stm", "lda", "glmnet", "matrixStats", "nanoparquet", "dplyr", "tidyr", "stringi", "Matrix", "readxl", "ps", "ggplot2", "scales", "patchwork", "readr", "digest", "yaml")
package_versions  <- lapply(manifest_packages, function(p) list(name = p, version = as.character(packageVersion(p))))
input_files <- c(
  lapply(seq_len(nrow(source_files)), function(i) list(path = source_files$path[i], hash = source_files$sha256[i], format = "parquet", rows = as.integer(source_files$rows[i]), role = "source data, read-only")),
  list(list(path = stopword_workbook, hash = workbook_hash, format = "xlsx", rows = nrow(tier2_raw), cols = ncol(tier2_raw), sheet = stopword_sheet, role = "Tier 2 stopword list, read-only")),
  lapply(which(input_integrity$exists), function(i) list(path = input_integrity$path[i], hash = input_integrity$sha256_now[i], format = if (grepl("\\.rds$", input_integrity$path[i])) "rds" else "csv",
                                                         role = "earlier output, read-only (reconciliation)")))
manifest_parameters <- c(list(model = "Structural Topic Model, stm::stm variational EM, spectral initialisation, prevalence ~ doc_type + s(day_of_july, df = 10), no content covariate",
                              text_representation = "the Tier 2 LDA representation reproduced with the original code (see parameters.csv)",
                              documents = "all Tier 2 modelable documents (857,385 in the full run), doc_key = Reddit fullname, source order",
                              topic_numbering = "the stm model's own topic index (no renumbering); prevalence_rank is a separate column",
                              retained = "models/stm_K040_pilot.rds (STM object), stm_input_K040_pilot.rds (documents, vocab, covariates), estimate_effect_K040.rds, fit_record_K040.rds"),
                         setNames(as.list(as.character(parameters_table$value)), parameters_table$parameter))
# Dimensions recorded for the RDS objects (the validator treats rds as tabular): the principal matrix of each object.
rds_dims <- list(
  "stm_K040_pilot.rds"         = c(rows = nrow(theta), cols = K),                                # theta: documents x topics
  "stm_input_K040_pilot.rds"   = c(rows = length(documents), cols = length(vocab)),            # documents x vocabulary
  "estimate_effect_K040.rds"   = c(rows = K, cols = effect_nsims),                             # topics x posterior draws (est + vcov each)
  "fit_record_K040.rds"        = c(rows = length(fit_record$log_lines), cols = length(fit_record)),   # console log lines x record fields
  "diagnostics_cache_K040.rds" = c(rows = K, cols = 3L))                                       # topics x (coherence, exclusivity, residual summary)
write_sidecar <- function(output) {
  ext <- tolower(tools::file_ext(output))
  manifest <- list(manifest_version = 1L, output_file = output, output_hash = sha256(output), output_format = switch(ext, csv = "csv", png = "image/png", rds = "rds", ext))
  if (ext == "csv") {
    dims <- table_dims[[output]]
    if (is.null(dims)) { tbl <- read_csv(output, show_col_types = FALSE, guess_max = 1000, progress = FALSE); dims <- c(rows = nrow(tbl), cols = ncol(tbl)) }
    manifest$output_rows <- unname(dims[["rows"]]); manifest$output_cols <- unname(dims[["cols"]])
  }
  if (ext == "rds") {
    dims <- rds_dims[[basename(output)]]
    if (!is.null(dims)) { manifest$output_rows <- unname(dims[["rows"]]); manifest$output_cols <- unname(dims[["cols"]]) }
  }
  manifest$input_files    <- input_files
  manifest$transformation <- list(script = script_path, script_hash = sha256(script_path), parameters = manifest_parameters, git_commit = git_commit)
  manifest$software <- list(language = "R", language_version = paste(R.version$major, R.version$minor, sep = "."), packages = package_versions, os = paste(Sys.info()[["sysname"]], Sys.info()[["release"]]))
  manifest$seed      <- stm_seed
  manifest$timestamp <- format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")
  manifest$notes     <- if (ext == "rds") "Retained object of the K = 40 STM pilot (July 2026 r/politics, Tier 2 text representation): the STM model object (rows x cols = theta documents x topics), its stm-format input (documents x vocabulary), the estimateEffect object (topics x posterior draws), the fit record (console log lines x fields) or the diagnostics cache (topics x measures). Source Parquet and earlier outputs read-only." else
    "K = 40 pilot Structural Topic Model of the July 2026 r/politics sample on the Tier 2 LDA text representation; prevalence varies with document type and a B-spline of the fractional day of July; no content covariate; spectral initialisation; the fit is deterministic on this machine; effect simulations seeded (effect_seed). Source Parquet and earlier outputs read-only."
  write_yaml(manifest, paste0(output, ".manifest.yml"))
}
invisible(lapply(list.files(tab_dir,   pattern = "\\.csv$", full.names = TRUE), write_sidecar))
invisible(lapply(list.files(fig_dir,   pattern = "\\.png$", full.names = TRUE), write_sidecar))
invisible(lapply(list.files(model_dir, pattern = "\\.rds$", full.names = TRUE), write_sidecar))
mark_stage("manifests")

# ---- Console summary ----
print(as.data.frame(stm_input_accounting))
print(as.data.frame(model_specification), right = FALSE)
print(as.data.frame(model_diagnostics), right = FALSE)
print(as.data.frame(topic_summary[, c("topic", "prevalence_rank", "expected_proportion", "semantic_coherence_stm", "exclusivity", "difference_submission_minus_comment", "top_10_words_by_probability")]))
print(as.data.frame(stage_timings[, c("stage", "seconds", "rss_mb_at_end", "peak_wset_mb_so_far")]))
print(as.data.frame(run_info), right = FALSE)
cat("\nDone. Figures ->", fig_dir, "\nTables ->", tab_dir, "\nModels ->", model_dir, "\n")
