# July 2026 r/politics sample - STM K-selection evidence for K = 10, 20 and 30 (searchK methodology)
#
# Purpose: produce the stm package's own K-selection evidence (the searchK() workflow: one common
# held-out construction by document completion, one fit per K, held-out likelihood, residual
# dispersion, variational bound, corrected lower bound, EM iterations) for exactly K = 10, 20 and 30
# on the July 2026 r/politics development corpus, using exactly the delivered Tier 2 LDA text
# representation (the same documents, tokeniser, 310-word Tier 2 stopword list, vocabulary rule and
# modelable-document rule, reproduced with the delivered code and reconciled against the delivered
# artifacts), with the prevalence and content specification of the assignment:
#   prevalence ~ author + s(created_utc) + doc_type + thread_id + parent_id + s(score) +
#                s(num_comments) + s(upvote_ratio) + num_comments_missing + upvote_ratio_missing,
#   L1 (glmnet) prevalence regularisation, content ~ doc_type, spectral initialisation, the same
#   documents, vocabulary, covariates, initialisation and convergence settings for every K.
# searchK() itself discards the fitted models, so its per-K body (stm:::get_statistics) is reproduced
# here step by step with the package's own functions (make.heldout, stm, eval.heldout, checkResiduals,
# bound, lbound = bound + lfactorial(K), em.its) and proven equal to a literal searchK() call on a
# small subset inside every run. Semantic coherence is computed separately with stm's
# content-covariate-aware semanticCoherence(); stm's exclusivity() is not defined for content
# covariate models and is recorded as unavailable, not substituted. No K is chosen and no topic is
# named or interpreted; topics are identified by the model's own index only. The source Parquet
# files and every delivered artifact are read only.
#
# What the script does, in order (main mode):
#   1. rebuilds the Tier 2 representation with the delivered code and reconciles it against the
#      delivered Tier 2 artifacts and the metadata inventory (discrepancies are reported, never
#      repaired);
#   2. converts the modelable documents to the stm format, builds the covariate table from the
#      source records (submission fields inherited through the established link_id join, explicit
#      missingness indicators and technical placeholders where the submission is absent from the
#      sample), and builds the common held-out object with stm::make.heldout under a recorded seed;
#   3. proves the searchK reimplementation against a literal searchK() call on a small subset;
#   4. launches one worker R process per K (this script in worker mode) that fits the model, saves
#      it at once, and computes and saves the held-out likelihood, residual dispersion, semantic
#      coherence and labels, each the moment it exists, with a background memory sampler; waits;
#   5. assembles the K comparison tables and figures, the per-document join tables, validation
#      checks, run records and provenance sidecars.
# Environment switches: STM_QUICK=<folder> runs a reduced smoke test into that folder
# (STM_QUICK_N documents, STM_QUICK_ITS EM iterations); STM_MAX_EM_ITS overrides the EM ceiling;
# STM_SEARCHK_CONCURRENCY sets how many K fits run at once; STM_MEMORY_SAMPLER=0 disables the
# samplers; STM_REFIT=1 ignores the model cache. STM_SEARCHK_WORKER_K is set by the main process
# only. All outputs go under r_analysis_outputs/stm_searchk_k10_k20_k30/ (this script owns that folder
# and clears its tables/ and figures/ at the start of each main run; models/ is a keyed cache).

# ---- Setup: packages, paths, parameters, output directories ----
# INPUT : none.
# DOES  : load packages; define paths, the delivered Tier 2 parameters (unchanged), the searchK
#         settings, the STM specification and the implementation settings; create folders.
# OUTPUT: parameter objects; figures/, tables/ and models/ directories.
suppressPackageStartupMessages({
  library(nanoparquet); library(dplyr); library(tidyr); library(stringi); library(Matrix); library(readxl)
  library(stm); library(ps); library(processx); library(ggplot2); library(scales); library(readr); library(digest); library(yaml); library(patchwork); library(clue)
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
utc_stamp <- function(t = Sys.time()) format(t, "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")
secs_since <- function(t) as.numeric(difftime(Sys.time(), t, units = "secs"))

project_dir       <- Sys.getenv("KC_REPO_ROOT", unset = getwd())
comments_dir      <- file.path(project_dir, "data_sample/comments/2026-07")
submissions_dir   <- file.path(project_dir, "data_sample/submissions/2026-07")
script_path       <- file.path(project_dir, "july2026_politics_stm_searchk_k10_k20_k30.R")
rscript_path      <- file.path(R.home("bin"), "Rscript.exe")
tier2_dir         <- file.path(project_dir, "r_analysis_outputs/lda_topics_tier2")            # delivered run (read-only)
k40_dir           <- file.path(project_dir, "r_analysis_outputs/lda_topics_tier2_k40_k80")    # K = 40 companion package (read-only)
inventory_dir     <- file.path(project_dir, "r_analysis_outputs/metadata_inventory")          # metadata inventory (read-only)
pilot_dir         <- file.path(project_dir, "r_analysis_outputs/stm_pilot_k40")               # K = 40 STM pilot (read-only)
stopword_workbook <- file.path(project_dir, "r_analysis_outputs/stopword_lists/curation/stopword_curation_tier2_cumulative_310_words.xlsx")
stopword_sheet    <- "tier2_cumulative_310"
quick_run   <- nzchar(Sys.getenv("STM_QUICK"))
worker_k    <- Sys.getenv("STM_SEARCHK_WORKER_K")
worker_mode <- nzchar(worker_k)
out_dir   <- if (quick_run) file.path(Sys.getenv("STM_QUICK"), "stm_searchk_k10_k20_k30_quick") else file.path(project_dir, "r_analysis_outputs/stm_searchk_k10_k20_k30")
fig_dir   <- file.path(out_dir, "figures")
tab_dir   <- file.path(out_dir, "tables")
model_dir <- file.path(out_dir, "models")
dir.create(fig_dir,   recursive = TRUE, showWarnings = FALSE)
dir.create(tab_dir,   recursive = TRUE, showWarnings = FALSE)
dir.create(model_dir, recursive = TRUE, showWarnings = FALSE)
# Tables and figures are regenerated on every main run; models/ is a keyed cache of the held-out object, the fits and their diagnostics.
if (!worker_mode) invisible(file.remove(list.files(c(fig_dir, tab_dir), pattern = "\\.(csv|png|yml)$", full.names = TRUE)))

# Delivered Tier 2 parameters (unchanged; they reproduce the documents, vocabulary and split).
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
# searchK settings (the package defaults of searchK / make.heldout, with a recorded seed).
K_values           <- c(10L, 20L, 30L)
heldout_N_share    <- 0.1          # searchK default: N = floor(0.1 * number of documents) partially held out
heldout_proportion <- 0.5          # searchK default: half of the tokens of each held-out document
heldout_seed       <- 20260909L    # make.heldout seed (recorded; the sampling is otherwise random)
coherence_M        <- 10L          # searchK default M (top words) for semantic coherence
# STM specification (identical for every K).
init_type        <- "Spectral"
stm_seed         <- 1L             # recorded by stm; no effect under spectral initialisation with ngroups = 1
max_em_its       <- as.integer(Sys.getenv("STM_MAX_EM_ITS", "25"))   # ceiling chosen from the measured per-iteration cost so that K = 30 completes within the overnight budget; identical for every K; restartable through stm(model = )
emtol            <- 1e-5           # package default: relative change of the approximate bound
report_every     <- 10L
gamma_prior      <- "L1"           # glmnet mgaussian (grouped L1 across topics), AIC selection (gamma.enet = 1, gamma.ic.k = 2: package defaults)
kappa_prior      <- "L1"           # package default for content covariates (distributed Poisson regressions via glmnet)
sigma_prior      <- 0
interactions     <- TRUE           # package default: topic x content-covariate interaction terms
prevalence_formula <- ~ author + s(created_utc) + doc_type + thread_id + parent_id + s(score) + s(num_comments) + s(upvote_ratio) + num_comments_missing + upvote_ratio_missing
content_formula    <- ~ doc_type
prevalence_string  <- paste(deparse(prevalence_formula, width.cutoff = 500L), collapse = " ")
content_string     <- paste(deparse(content_formula), collapse = " ")
smooth_terms       <- c("created_utc", "score", "num_comments", "upvote_ratio")
gamma_pmax       <- as.integer(Sys.getenv("STM_GAMMA_PMAX", "50000"))   # memory cap on glmnet's mgaussian path in the L1 prevalence step (0 = package default, see the worker section)
fit_chunk        <- as.integer(Sys.getenv("STM_FIT_CHUNK", "5"))         # EM iterations per stm() call; the fit continues through stm's documented restart (model = previous), the partial model saved after every chunk
label_n_words    <- 20L
frex_weight      <- 0.5
correlation_cutoff <- 0.01
sample_stride    <- 5000L          # every sample_stride-th STM document is re-tokenised by a second code path
# Self-check of the searchK reimplementation: a literal searchK() call on a small subset in every run.
selfcheck_n      <- 3000L
selfcheck_K      <- c(5L, 6L)
selfcheck_its    <- 2L
selfcheck_seed   <- 11L
# Implementation settings (results do not depend on them).
concurrency       <- as.integer(Sys.getenv("STM_SEARCHK_CONCURRENCY", "3"))
launch_stagger    <- if (quick_run) 2 else 240   # seconds between worker launches so the spectral initialisations (the memory peak) do not coincide
memory_sampler_on <- !identical(Sys.getenv("STM_MEMORY_SAMPLER"), "0")
sampler_interval  <- if (quick_run) 5 else 20
force_refit       <- identical(Sys.getenv("STM_REFIT"), "1")
poll_seconds      <- if (quick_run) 5 else 30
progress_every    <- if (quick_run) 60 else 600
quick_n           <- as.integer(Sys.getenv("STM_QUICK_N", "20000"))
quick_seed        <- 1L
if (quick_run) { max_em_its <- as.integer(Sys.getenv("STM_QUICK_ITS", "2")); selfcheck_n <- 1500L; fit_chunk <- as.integer(Sys.getenv("STM_FIT_CHUNK", "1")) }

comment_files    <- sort(list.files(comments_dir,    pattern = "\\.parquet$", full.names = TRUE))
submission_files <- sort(list.files(submissions_dir, pattern = "\\.parquet$", full.names = TRUE))
sha256 <- function(path) paste0("sha256:", digest(path, algo = "sha256", file = TRUE))
# A sidecar's top-level field; if the YAML does not parse strictly (the K = 40 pilot's sidecars carry a duplicate
# 'text_representation' key that yaml::read_yaml rejects), the top-level line is read directly.
manifest_field <- function(path, field) {
  mf <- paste0(path, ".manifest.yml")
  if (!file.exists(mf)) return(NA)
  v <- tryCatch(read_yaml(mf)[[field]], error = function(e) NULL)
  if (is.null(v)) {
    ln <- grep(sprintf("^%s:", field), readLines(mf, warn = FALSE), value = TRUE)
    v <- if (length(ln)) trimws(sub(sprintf("^%s:[[:space:]]*", field), "", ln[1])) else NA
  }
  v
}
manifest_parses_strictly <- function(path) { mf <- paste0(path, ".manifest.yml"); file.exists(mf) && tryCatch({ read_yaml(mf); TRUE }, error = function(e) FALSE) }
k_tag <- function(K) sprintf("K%03d", K)
common_input_path <- file.path(model_dir, "common_input.rds")
heldout_path      <- file.path(model_dir, "heldout_common.rds")
sampler_path      <- file.path(model_dir, "memory_sampler.R")
worker_status_path <- function(K) file.path(model_dir, sprintf("worker_%s_status.txt", k_tag(K)))
worker_file <- function(K, what, ext = "rds") file.path(model_dir, sprintf("%s_%s.%s", what, k_tag(K), ext))

# ---- Covariate builder (tested by tests/test_stm_searchk_covariates.R) ----
# INPUT : docs_stm (one row per STM document in stm order: doc_key, doc_type, author, created_utc,
#         link_id, parent_id, score, num_comments, upvote_ratio - the record's own stored values, NA
#         for comments); submissions (the source submission records: id, num_comments, upvote_ratio).
# DOES  : thread identity = the established link_id for comments, the submission's own fullname
#         ('t3_' + id, the same string as its comments' link_id) for submissions; parent identity =
#         the stored parent_id for comments (absent parents keep their own level), the explicit level
#         NO_PARENT_SUBMISSION for submissions; num_comments and upvote_ratio = the submission's own
#         stored values, inherited by its comments through the established join comment.link_id =
#         't3_' + submission.id against all source submissions; where the linked submission is absent
#         from the sample the value is not observed: a missingness indicator is 1 and a technical
#         placeholder (the median of the observed values over the modelled documents, a constant
#         whose whole effect the indicator column absorbs) completes the design matrix. Nothing is
#         removed; no substantive value is invented.
# OUTPUT: list(meta = the covariate data.frame passed to stm (factors: author, doc_type with comment
#         as reference, thread_id, parent_id; numerics: created_utc, score, num_comments,
#         upvote_ratio; indicators: num_comments_missing, upvote_ratio_missing), doc_key,
#         placeholders, missingness (reason counts), observed vectors).
build_stm_covariates <- function(docs_stm, submissions) {
  sub_by_link <- tibble(link_id = paste0("t3_", submissions$id), submission_num_comments = as.numeric(submissions$num_comments),
                        submission_upvote_ratio = as.numeric(submissions$upvote_ratio))
  stopifnot(!anyDuplicated(sub_by_link$link_id))
  d <- docs_stm |> select(doc_key, doc_type, author, created_utc, link_id, parent_id, score, num_comments, upvote_ratio) |>
    left_join(sub_by_link, by = "link_id")
  is_sub  <- d$doc_type == "submission"
  linked  <- d$link_id %in% sub_by_link$link_id
  num_comments_observed <- ifelse(is_sub, as.numeric(d$num_comments), d$submission_num_comments)
  upvote_ratio_observed <- ifelse(is_sub, as.numeric(d$upvote_ratio), d$submission_upvote_ratio)
  nc_missing <- as.integer(is.na(num_comments_observed))
  ur_missing <- as.integer(is.na(upvote_ratio_observed))
  placeholders <- c(num_comments = median(num_comments_observed[nc_missing == 0L]), upvote_ratio = median(upvote_ratio_observed[ur_missing == 0L]))
  thread_id    <- ifelse(is_sub, d$doc_key, d$link_id)
  parent_level <- ifelse(is_sub, "NO_PARENT_SUBMISSION", d$parent_id)
  stopifnot(!anyNA(thread_id), !anyNA(parent_level), !anyNA(d$author), !anyNA(d$created_utc), !anyNA(d$score))
  meta <- data.frame(author = factor(d$author), created_utc = as.numeric(d$created_utc), doc_type = factor(d$doc_type, levels = c("comment", "submission")),
                     thread_id = factor(thread_id), parent_id = factor(parent_level), score = as.numeric(d$score),
                     num_comments = ifelse(nc_missing == 1L, placeholders[["num_comments"]], num_comments_observed),
                     upvote_ratio = ifelse(ur_missing == 1L, placeholders[["upvote_ratio"]], upvote_ratio_observed),
                     num_comments_missing = nc_missing, upvote_ratio_missing = ur_missing, stringsAsFactors = FALSE)
  missingness <- bind_rows(lapply(c("num_comments", "upvote_ratio"), function(f) {
    obs <- if (f == "num_comments") num_comments_observed else upvote_ratio_observed
    own <- if (f == "num_comments") d$num_comments else d$upvote_ratio
    tibble(field = f,
           reason = c("submission: own stored value", "comment: inherited from the linked submission (link_id join)",
                      "comment: linked submission absent from the sample", "submission: stored value missing",
                      "comment: linked submission present but its stored value missing"),
           n = c(sum(is_sub & !is.na(own)), sum(!is_sub & linked & !is.na(obs)), sum(!is_sub & !linked), sum(is_sub & is.na(own)), sum(!is_sub & linked & is.na(obs))),
           covariate_value = c("observed", "observed (inherited)", sprintf("placeholder %s with indicator = 1", format(placeholders[[f]])),
                               sprintf("placeholder %s with indicator = 1", format(placeholders[[f]])), sprintf("placeholder %s with indicator = 1", format(placeholders[[f]]))))
  }))
  list(meta = meta, doc_key = d$doc_key, placeholders = placeholders, missingness = missingness, linked_submission_in_sample = linked,
       num_comments_observed = num_comments_observed, upvote_ratio_observed = upvote_ratio_observed)
}

# ---- Fit-log parser ----
# INPUT : the console lines of one stm() fit (verbose = TRUE).
# DOES  : read the E-step and M-step seconds of every iteration and the per-word bound lines.
# OUTPUT: tibble(iteration, estep_seconds, mstep_seconds_reported, per_word_bound_reported,
#         relative_change_reported); the convergence message.
parse_fit_log <- function(lines) {
  estep <- as.numeric(sub("^Completed E-Step \\((\\d+) seconds\\).*$", "\\1", grep("^Completed E-Step", lines, value = TRUE)))
  mlines <- grep("^Completed M-Step", lines, value = TRUE)
  mstep <- ifelse(grepl("seconds", mlines), suppressWarnings(as.numeric(sub("^Completed M-Step \\((\\d+) seconds\\).*$", "\\1", mlines))), 0)
  n <- max(length(estep), length(mstep))
  pad <- function(x) c(x, rep(NA_real_, n - length(x)))
  tibble(iteration = seq_len(n), estep_seconds = pad(estep), mstep_seconds_reported = pad(mstep))
}
fit_log_message <- function(lines) {
  m <- grep("^Model (Converged|Terminated)", lines, value = TRUE)
  if (length(m)) trimws(tail(m, 1)) else NA_character_   # the last stm() call decides (earlier chunks end with the ceiling message)
}

# ---- Memory sampler script (written once; one sampler process per fitting process) ----
sampler_code <- c(
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
  '}')
start_sampler <- function(trace_csv, stage_file, stop_file, log_file) {
  if (!memory_sampler_on) return(FALSE)
  if (file.exists(stop_file)) invisible(file.remove(stop_file))
  system2(rscript_path, args = c(shQuote(sampler_path), Sys.getpid(), shQuote(trace_csv), shQuote(stage_file), shQuote(stop_file), sampler_interval),
          wait = FALSE, stdout = log_file, stderr = log_file)
  TRUE
}
stop_sampler <- function(started, stop_file) { if (started) { writeLines("stop", stop_file); Sys.sleep(sampler_interval + 2) } }

# ---- The L1 prevalence step: stm's opt.mu with a per-call record and an optional memory cap ----
# stm:::opt.mu (mode "L1") calls glmnet::glmnet(x = covar[, -1], y = lambda, family = "mgaussian",
# alpha = enet) with glmnet's defaults and selects the path point minimising dev + ic.k * df
# (stm:::unpack.glmnet). glmnet allocates the whole 100-point coefficient path as a dense
# nx x (K - 1) x 100 array with nx = number of design columns (446,630 here) and copies it several
# times: measured on 2026-09-09, one call at K = 10 took 377 s and raised the process working set to
# 18.7 GB; at K = 30 the same allocation is 3.2 times larger, beyond this machine's free memory.
# With gamma_pmax > 0 the identical call is made with glmnet's `pmax` argument (the path stops once
# more than pmax design columns have nonzero coefficients), which caps nx at pmax. Every path point
# before the stop is computed identically (same lambda sequence, same warm starts), so the selected
# solution is unchanged whenever the information-criterion minimum lies before the stop; every call
# records the path length, the chosen index and whether it was the last computed point
# (chosen_is_last_computed = TRUE would mean the cap could have altered the selection). The
# replacement function reproduces stm's opt.mu line for line; only the `pmax` argument and the
# record are added. STM_GAMMA_PMAX=0 restores the untouched package call.
stm_opt_mu_original <- stm:::opt.mu
gamma_call_counter <- 0L
gamma_log_csv <- NA_character_
opt_mu_recorded <- function(lambda, mode = c("CTM", "Pooled", "L1"), covar = NULL, enet = NULL, ic.k = 2, maxits = 1000) {
  if (mode != "L1") return(stm_opt_mu_original(lambda, mode, covar, enet, ic.k, maxits))
  t1 <- Sys.time()
  out <- if (gamma_pmax > 0) glmnet::glmnet(x = covar[, -1], y = lambda, family = "mgaussian", alpha = enet, pmax = gamma_pmax) else
    glmnet::glmnet(x = covar[, -1], y = lambda, family = "mgaussian", alpha = enet)
  unpack <- stm:::unpack.glmnet(out, ic.k = ic.k)
  gamma <- rbind(unpack$intercept, unpack$coef)
  mu <- t(covar %*% gamma)
  if (!is.matrix(mu)) mu <- as.matrix(mu)
  dev <- (1 - out$dev.ratio) * out$nulldev; df <- colSums(out$dfmat); ic <- dev + ic.k * df; sel <- which.min(ic)
  gamma_call_counter <<- gamma_call_counter + 1L
  row <- data.frame(call = gamma_call_counter, time_utc = utc_stamp(), pmax = gamma_pmax, path_length = length(out$lambda), chosen_index = sel,
                    chosen_is_last_computed = sel == length(out$lambda), df_chosen = df[sel], nonzero_rows_chosen = sum(rowSums(unpack$coef != 0) > 0),
                    nonzero_rows_at_path_end = sum(rowSums(do.call(cbind, lapply(out$beta, function(b) as.numeric(b[, ncol(b)]) != 0))) > 0),
                    dev_ratio_chosen = out$dev.ratio[sel], dev_ratio_end = out$dev.ratio[length(out$lambda)], lambda_chosen = out$lambda[sel],
                    lambda_ratio_end = out$lambda[length(out$lambda)] / out$lambda[1], seconds = secs_since(t1), peak_wset_mb = round(process_memory()[["peak_wset_mb"]]))
  if (!is.na(gamma_log_csv)) write.table(row, gamma_log_csv, sep = ",", row.names = FALSE, col.names = !file.exists(gamma_log_csv), append = file.exists(gamma_log_csv))
  list(mu = mu, gamma = gamma)
}

# ---- Worker mode: fit one K and compute its searchK statistics ----
# INPUT : STM_SEARCHK_WORKER_K, STM_SEARCHK_RUN_KEY; models/common_input.rds (held-out documents,
#         vocabulary, missing tokens, covariate table, settings).
# DOES  : stm(heldout$documents, heldout$vocab, K, prevalence, content, data, init.type = "Spectral",
#         gamma.prior = "L1", kappa.prior = "L1", ...) in this process with the console written to
#         models/fit_console_K.log, the model saved the moment the fit ends; then, in the order of
#         cost, eval.heldout, semanticCoherence (+ the exclusivity attempt), sageLabels/labelTopics,
#         checkResiduals, each saved the moment it exists; a background memory sampler records this
#         process throughout; the status file ends 'done' or 'error: ...'.
# OUTPUT: models/stm_K.rds, fit_record_K.rds, heldout_eval_K.rds, semcoh_K.rds, labels_K.rds,
#         residuals_K.rds, gamma_path_K.csv, fit_console_K.log, resource_trace_K_<run>.csv,
#         worker_K_status.txt.
run_stm_worker <- function(K) {
  run_key <- Sys.getenv("STM_SEARCHK_RUN_KEY")
  status_file <- worker_status_path(K)
  stage_file  <- file.path(model_dir, sprintf("stage_marker_%s.txt", k_tag(K)))
  stop_file   <- file.path(model_dir, sprintf("sampler_stop_%s.txt", k_tag(K)))
  trace_csv   <- file.path(model_dir, sprintf("resource_trace_%s_%s.csv", k_tag(K), run_id))
  set_stage   <- function(name) writeLines(name, stage_file)
  writeLines(c("running", run_key, run_id, utc_stamp()), status_file)
  on_error <- function(e) {
    writeLines(c(paste0("error: ", conditionMessage(e)), run_key, run_id, utc_stamp()), status_file)
    log_line("WORKER ERROR:", conditionMessage(e))
    if (sampler_started) stop_sampler(TRUE, stop_file)
    quit(save = "no", status = 1)
  }
  sampler_started <- FALSE
  withCallingHandlers(tryCatch({
    ci <- readRDS(common_input_path)
    stopifnot(identical(ci$run_key, run_key), K %in% ci$K_values)
    set_stage("load")
    sampler_started <- start_sampler(trace_csv, stage_file, stop_file, file.path(model_dir, sprintf("memory_sampler_%s.log", k_tag(K))))
    log_line("worker K =", K, "started; run key", substr(run_key, 1, 12), "; documents", length(ci$documents), "; vocab", length(ci$vocab), "; sampler", sampler_started)
    model_path      <- worker_file(K, "stm")
    fit_record_path <- worker_file(K, "fit_record")
    partial_path    <- worker_file(K, "stm_partial")
    fit_from_cache  <- !force_refit && file.exists(model_path) && file.exists(fit_record_path) && identical(readRDS(fit_record_path)$run_key, run_key)
    if (fit_from_cache) {
      model <- readRDS(model_path); fit_record <- readRDS(fit_record_path)
      log_line("worker K =", K, ": model loaded from the cache (fitting run", fit_record$run_id, ")")
    } else {
      # The fit runs in chunks of fit_chunk EM iterations: each stm() call after the first continues the previous
      # model through stm's documented restart (argument `model`, with max.em.its = the total so far), and the
      # partial model is saved after every chunk so that a crash costs at most one chunk. The restart carries
      # mu/gamma, sigma, beta, kappa, the variational means (eta) and the bound trace, so the trajectory is the
      # uninterrupted one up to the exp(log(beta)) round trip of the saved topic-word matrices.
      set_stage("fit")
      gamma_pmax <<- ci$settings$gamma_pmax
      gamma_log_csv <<- worker_file(K, "gamma_path", "csv")
      assignInNamespace("opt.mu", opt_mu_recorded, ns = "stm")
      prevalence_formula_w <- as.formula(ci$prevalence_string); environment(prevalence_formula_w) <- globalenv()
      content_formula_w    <- as.formula(ci$content_string);    environment(content_formula_w)    <- globalenv()
      fit_log <- worker_file(K, "fit_console", "log")
      max_its <- ci$settings$max_em_its; chunk <- max(1L, ci$settings$fit_chunk)
      model <- NULL; chunk_records <- list(); fit_start <- Sys.time()
      resumed <- !force_refit && file.exists(partial_path)
      if (resumed) {
        pm <- readRDS(partial_path)
        resumed <- identical(pm$run_key, run_key) && !is.null(pm$model)
        if (resumed) { model <- pm$model; chunk_records <- pm$chunk_records; fit_start <- pm$fit_start; gamma_call_counter <<- pm$gamma_calls
                       log_line("worker K =", K, ": resuming from the partial model after", model$convergence$its, "iterations (saved by run", pm$run_id, ")") }
        rm(pm)
      }
      if (!resumed) { if (file.exists(gamma_log_csv)) invisible(file.remove(gamma_log_csv)); if (file.exists(fit_log)) invisible(file.remove(fit_log)); gamma_call_counter <<- 0L }
      while (is.null(model) || (model$convergence$its < max_its && !model$convergence$converged)) {
        target <- min((if (is.null(model)) 0L else model$convergence$its) + chunk, max_its)
        log_con <- file(fit_log, open = "at")
        sink(log_con, split = TRUE)
        cat(sprintf("run_id %s; K = %d; gamma_pmax %d; chunk to iteration %d; start %s\n", run_id, K, gamma_pmax, target, utc_stamp()))
        t_chunk <- Sys.time(); cpu_before <- proc.time()
        model <- stm(documents = ci$documents, vocab = ci$vocab, K = K, prevalence = prevalence_formula_w, content = content_formula_w, data = ci$meta_df,
                     init.type = ci$settings$init_type, seed = ci$settings$stm_seed, max.em.its = target, emtol = ci$settings$emtol,
                     verbose = TRUE, reportevery = ci$settings$report_every, interactions = ci$settings$interactions,
                     gamma.prior = ci$settings$gamma_prior, sigma.prior = ci$settings$sigma_prior, kappa.prior = ci$settings$kappa_prior, model = model)
        cpu_after <- proc.time()
        cat(sprintf("chunk end %s; iterations %d; converged %s\n", utc_stamp(), model$convergence$its, model$convergence$converged))
        sink(); close(log_con)
        chunk_records[[length(chunk_records) + 1L]] <- tibble(chunk = length(chunk_records) + 1L, target_iterations = target, iterations_after = model$convergence$its, converged = model$convergence$converged,
                                                             wall_seconds = secs_since(t_chunk), cpu_user_seconds = unname((cpu_after - cpu_before)[["user.self"]]), cpu_system_seconds = unname((cpu_after - cpu_before)[["sys.self"]]),
                                                             stm_time_seconds = model$time, peak_wset_mb = unname(process_memory()[["peak_wset_mb"]]), run_id = run_id)
        saveRDS(list(run_key = run_key, K = K, run_id = run_id, model = model, chunk_records = chunk_records, fit_start = fit_start, gamma_calls = gamma_call_counter), partial_path, compress = FALSE)
        log_line("worker K =", K, ": chunk done;", model$convergence$its, "of at most", max_its, "iterations; converged", model$convergence$converged, "; partial model saved")
      }
      fit_end <- Sys.time()
      chunk_table <- bind_rows(chunk_records)
      mem_after_fit <- process_memory()
      saveRDS(model, model_path, compress = FALSE)
      fit_record <- list(K = K, run_id = run_id, run_key = run_key, fit_start_utc = utc_stamp(fit_start), fit_end_utc = utc_stamp(fit_end),
                         wall_seconds = sum(chunk_table$wall_seconds), wall_seconds_including_saves = as.numeric(difftime(fit_end, fit_start, units = "secs")),
                         cpu_user_seconds = sum(chunk_table$cpu_user_seconds), cpu_system_seconds = sum(chunk_table$cpu_system_seconds),
                         model_time_seconds = sum(chunk_table$stm_time_seconds), chunks = nrow(chunk_table), fit_chunk = chunk, chunk_table = chunk_table,
                         peak_wset_mb_after_fit = unname(mem_after_fit[["peak_wset_mb"]]), rss_mb_after_fit = unname(mem_after_fit[["rss_mb"]]),
                         log_lines = readLines(fit_log, warn = FALSE), trace_csv = trace_csv, stm_version = as.character(packageVersion("stm")),
                         glmnet_version = as.character(packageVersion("glmnet")), model_file_bytes = file.size(model_path), pid = Sys.getpid(),
                         gamma_pmax = gamma_pmax, gamma_calls = gamma_call_counter)
      saveRDS(fit_record, fit_record_path)
      if (file.exists(partial_path)) invisible(file.remove(partial_path))
      log_line("worker K =", K, ": fit done;", model$convergence$its, "iterations in", nrow(chunk_table), "chunks; converged", model$convergence$converged, ";", round(fit_record$wall_seconds), "s; model saved")
    }
    cached_ok <- function(p) !force_refit && file.exists(p) && identical(readRDS(p)$run_key, run_key)
    p <- worker_file(K, "heldout_eval")
    if (!cached_ok(p)) {
      set_stage("heldout_eval"); t1 <- Sys.time(); mem0 <- process_memory()
      he <- eval.heldout(model, ci$missing)
      saveRDS(list(run_key = run_key, K = K, computed_run_id = run_id, expected_heldout = he$expected.heldout, doc_heldout = he$doc.heldout, index = he$index, ntokens = he$ntokens,
                   seconds = secs_since(t1), peak_wset_mb = unname(process_memory()[["peak_wset_mb"]])), p)
      log_line("worker K =", K, ": held-out likelihood", round(he$expected.heldout, 4), "in", round(secs_since(t1), 1), "s")
    }
    p <- worker_file(K, "semcoh")
    if (!cached_ok(p)) {
      set_stage("semantic_coherence"); t1 <- Sys.time()
      sc <- semanticCoherence(model, ci$documents, M = ci$settings$coherence_M)
      s_sc <- secs_since(t1); t1 <- Sys.time()
      ex <- tryCatch(list(values = exclusivity(model, M = ci$settings$coherence_M, frexw = 0.7), error = NA_character_),
                     error = function(e) list(values = NULL, error = conditionMessage(e)))
      saveRDS(list(run_key = run_key, K = K, computed_run_id = run_id, semantic_coherence = sc, seconds = s_sc, exclusivity = ex$values, exclusivity_error = ex$error,
                   exclusivity_seconds = secs_since(t1)), p)
      log_line("worker K =", K, ": semantic coherence mean", round(mean(sc), 3), "in", round(s_sc, 1), "s; exclusivity:", if (is.na(ex$error)) "computed" else ex$error)
    }
    p <- worker_file(K, "labels")
    if (!cached_ok(p)) {
      set_stage("labels"); t1 <- Sys.time()
      sl <- sageLabels(model, n = ci$settings$label_n_words)
      lt <- labelTopics(model, n = ci$settings$label_n_words)
      saveRDS(list(run_key = run_key, K = K, computed_run_id = run_id, sage = sl, labels = lt, seconds = secs_since(t1)), p)
      log_line("worker K =", K, ": labels in", round(secs_since(t1), 1), "s")
    }
    p <- worker_file(K, "residuals")
    if (!cached_ok(p)) {
      set_stage("residuals"); t1 <- Sys.time()
      rs <- checkResiduals(model, ci$documents)
      saveRDS(list(run_key = run_key, K = K, computed_run_id = run_id, dispersion = rs$dispersion, pvalue = rs$pvalue, df = rs$df, seconds = secs_since(t1),
                   peak_wset_mb = unname(process_memory()[["peak_wset_mb"]])), p)
      log_line("worker K =", K, ": residual dispersion", round(rs$dispersion, 3), "in", round(secs_since(t1), 1), "s")
    }
    set_stage("done")
    stop_sampler(sampler_started, stop_file)
    writeLines(c("done", run_key, run_id, utc_stamp()), status_file)
    log_line("worker K =", K, "finished; peak working set", round(process_memory()[["peak_wset_mb"]]), "MB")
  }, error = on_error), warning = function(w) { log_line("worker K =", K, "warning:", conditionMessage(w)); invokeRestart("muffleWarning") })
}
if (worker_mode) {
  run_stm_worker(as.integer(worker_k))
  quit(save = "no", status = 0)
}

log_line("start (main); quick_run =", quick_run, "; K =", paste(K_values, collapse = ", "), "; max EM iterations =", max_em_its, "; concurrency =", concurrency, "; run_id", run_id)

# ---- Stopword list: the Tier 2 workbook (read-only; the delivered rule) ----
# INPUT : the curation workbook.
# DOES  : read the 'word' column of the Tier 2 sheet exactly as stored; record its SHA256.
# OUTPUT: stopword_list (character, 310); workbook_hash.
workbook_hash <- sha256(stopword_workbook)
tier2_raw     <- read_excel(stopword_workbook, sheet = stopword_sheet, col_types = "text")
stopword_list <- tier2_raw$word
token_regex   <- "\\p{L}+(?:'\\p{L}+)*"
log_line("workbook read:", length(stopword_list), "entries; hash", workbook_hash)

# ---- Read source data (read-only) ----
# INPUT : 7 comment Parquet files + 1 submission Parquet file.
# DOES  : read every file with nanoparquet and stack; keep the columns the delivered document
#         table uses plus the metadata fields required as covariates or preserved for linkage
#         (score, num_comments, upvote_ratio, url) and the file of origin. Column selection only;
#         no rows dropped. INT64 columns arrive as double.
# OUTPUT: comments; submissions; source_files (bytes, hash, rows per file).
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

# ---- Document table (the delivered rule) ----
# INPUT : comments; submissions.
# DOES  : one document per source record; comment text = body; submission text =
#         title, one space, selftext; doc_key = Reddit fullname ('t1_' + id, 't3_' + id).
#         The preserved metadata fields are attached afterwards by doc_key so that the
#         delivered construction is copied unchanged.
# OUTPUT: docs (source order: comments then submissions); extra (preserved fields by doc_key).
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

# ---- Text processing (the delivered steps 1-8; reference-only counts omitted) ----
# INPUT : docs$text.
# DOES  : case folding; U+2019 -> '; URLs removed; letter-run tokens with internal
#         apostrophes; trailing 's stripped; tokens shorter than min_token_chars removed;
#         the 310 Tier 2 entries removed; tokens of removal-marker-only documents excluded.
# OUTPUT: tok / tdoc; per-document token counts on docs.
removal_markers  <- c("[deleted]", "[removed]", "[ Removed by Reddit ]")
docs$marker_only <- stri_trim_both(docs$text) %in% removal_markers
url_regex   <- "(https?://|www\\.)\\S+"
url_opts    <- stri_opts_regex(case_insensitive = TRUE)
text_step <- stri_trans_tolower(docs$text)                                    # step 1
text_step <- stri_replace_all_fixed(text_step, "’", "'")                 # step 2
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

# ---- Vocabulary and document-term matrix (the delivered rule) ----
# INPUT : tok / tdoc.
# DOES  : terms kept when min_document_frequency <= df <= max_document_share x
#         non-marker documents; one dtm row per source document; model_status.
# OUTPUT: vocabulary; dtm_all; vocabulary_table; docs$model_status; docs$n_tokens_modeled.
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

# ---- Tier 2 training / held-out split (the delivered rule; carried as a label only) ----
# INPUT : docs.
# DOES  : reproduce the seeded 5% held-out sample so every STM document carries the Tier 2
#         'split' label for later comparison. The searchK held-out construction is a different
#         thing (document completion inside documents); the Tier 2 split does not enter the STM.
# OUTPUT: docs$split; n_train; n_heldout.
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

# ---- Reconciliation against the delivered Tier 2 artifacts and the inventory (read-only) ----
# INPUT : the delivered accounting tables, vocabulary table, the K = 40 document table,
#         the cached theta_K040_seed1.rds, three inventory tables and the K = 40 STM pilot's
#         accounting, with their manifests.
# DOES  : hash every artifact read and compare with its manifest; compare every count of
#         this run's representation with the delivered values (documents by status and
#         type, tokens, vocabulary, per-term df/tf, per-document status/split/tokens,
#         length summary, source-file hashes). Discrepancies are reported, not repaired.
# OUTPUT: input_integrity; tier2_reconciliation; discrepancies; delivered tables.
tier2_tab <- file.path(tier2_dir, "tables"); k40_tab <- file.path(k40_dir, "tables"); inv_tab <- file.path(inventory_dir, "tables"); pilot_tab <- file.path(pilot_dir, "tables")
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
  inventory_model_status     = file.path(inv_tab, "model_status_by_type.csv"),
  pilot_stm_input_accounting = file.path(pilot_tab, "stm_input_accounting.csv"))
input_integrity <- tibble(role = names(artifact_paths), path = unname(artifact_paths)) |>
  mutate(exists = file.exists(path), bytes = if_else(exists, file.size(path), NA_real_),
         sha256_now = vapply(path, function(p) if (file.exists(p)) sha256(p) else NA_character_, character(1)),
         sha256_in_manifest = vapply(path, function(p) as.character(manifest_field(p, "output_hash")), character(1)),
         hash_equals_manifest = exists & !is.na(sha256_in_manifest) & sha256_now == sha256_in_manifest,
         manifest_parses_as_strict_yaml = vapply(path, manifest_parses_strictly, logical(1)))
read_if <- function(role, ...) if (file.exists(artifact_paths[[role]])) read_csv(artifact_paths[[role]], show_col_types = FALSE, progress = FALSE, ...) else NULL
delivered_accounting <- read_if("document_accounting")
delivered_report     <- read_if("text_processing_report")
delivered_length     <- read_if("document_length_summary")
delivered_vocabulary <- read_if("vocabulary", col_types = cols(term = col_character(), .default = col_guess()), na = character(), trim_ws = FALSE)
inventory_source     <- read_if("inventory_source_files")
inventory_linkage    <- read_if("inventory_linkage_counts", col_types = cols(.default = col_character()))
inventory_parent     <- read_if("inventory_parent_linkage")
inventory_status     <- read_if("inventory_model_status")
pilot_accounting     <- read_if("pilot_stm_input_accounting")
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
acc_row <- function(stage) if (is.null(delivered_accounting)) rep(NA, 3) else unlist(delivered_accounting[delivered_accounting$stage == stage, c("comments", "submissions", "total")])
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
rep_item <- function(item) if (is.null(delivered_report)) NA else delivered_report$value[delivered_report$item == item]
add_recon("vocabulary size (terms modeled)", rep_item("vocabulary size (terms modeled)"), n_vocabulary, "text_processing_report.csv")
add_recon("distinct candidate terms (vocabulary before pruning)", rep_item("distinct candidate terms in non-marker documents (vocabulary before pruning)"), n_terms_candidate, "text_processing_report.csv")
add_recon("terms below the document-frequency floor", rep_item("step 9: terms in fewer than 5 non-marker documents removed"), sum(term_below_floor), "text_processing_report.csv")
add_recon("tokens removed with rare terms", rep_item("step 9: tokens removed with those terms"), n_tokens_rare, "text_processing_report.csv")
add_recon("terms above the document-share ceiling", rep_item("step 9: terms in more than 50% of non-marker documents removed"), sum(term_above_ceiling), "text_processing_report.csv")
add_recon("candidate tokens in non-marker documents", rep_item("candidate tokens in non-marker documents"), n_candidate_tokens - n_tokens_in_marker_docs, "text_processing_report.csv")
add_recon("tokens modeled", rep_item("tokens modeled"), sum(docs$n_tokens_modeled), "text_processing_report.csv")
if (!is.null(delivered_vocabulary)) {
  vj <- delivered_vocabulary |> select(term, d_df = document_frequency, d_tf = term_frequency, d_kept = kept) |>
    full_join(vocabulary_table |> select(term, document_frequency, term_frequency, kept), by = "term")
  add_recon("vocabulary.csv: rows", nrow(delivered_vocabulary), nrow(vocabulary_table), "vocabulary.csv")
  add_recon("vocabulary.csv: terms present in both", nrow(delivered_vocabulary), sum(!is.na(vj$d_df) & !is.na(vj$document_frequency)), "vocabulary.csv")
  add_recon("vocabulary.csv: rows with equal document_frequency, term_frequency and kept", nrow(delivered_vocabulary),
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
if (!is.null(delivered_length)) {
  ql <- function(x, p) quantile(x, p, type = 7, names = FALSE)
  for (tp in c("comment", "submission")) {
    x <- docs$n_tokens_modeled[docs$model_status == "modeled" & docs$doc_type == tp]
    d <- delivered_length[delivered_length$doc_type == tp, ]
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
if (!is.null(pilot_accounting)) {
  pa <- function(stage) { v <- pilot_accounting$total[pilot_accounting$stage == stage]; if (length(v)) v else NA }
  add_recon("K = 40 STM pilot: STM documents after conversion", pa("STM documents after conversion"), n_modeled, "stm_pilot_k40/tables/stm_input_accounting.csv")
  add_recon("K = 40 STM pilot: tokens in the STM documents", pa("tokens in the STM documents"), sum(docs$n_tokens_modeled), "stm_pilot_k40/tables/stm_input_accounting.csv")
  add_recon("K = 40 STM pilot: STM vocabulary terms", pa("STM vocabulary terms"), n_vocabulary, "stm_pilot_k40/tables/stm_input_accounting.csv")
}
tier2_reconciliation <- bind_rows(recon)
discrepancies <- tier2_reconciliation |> filter(!equal)
rm(k40_table, delivered_vocabulary); invisible(gc())
mark_stage("reconciliation")
log_line("reconciliation:", sum(tier2_reconciliation$equal), "of", nrow(tier2_reconciliation), "items equal;", nrow(discrepancies), "discrepancies")

# ---- STM input: documents, vocabulary, covariates ----
# INPUT : dtm_all; docs; extra; submissions.
# DOES  : restrict the document-term matrix to the modeled documents (all of them, in source
#         order; a seeded subsample under STM_QUICK) and convert it with stm::asSTMCorpus to
#         the list format. Account for every document, token and term across the conversion.
#         Build the covariate table with build_stm_covariates() from the source records; record
#         the spline bases stm's s() will build (df, knots), the design-matrix accounting (columns
#         per term, sparsity) and the coverage of every covariate by document type.
# OUTPUT: documents; vocab; meta_df; docs_stm; stm_input_accounting; covariate tables; X_design.
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
  mutate(stm_row = row_number(), n_tokens_stm = as.integer(doc_tokens_stm), n_distinct_terms_stm = doc_terms_stm)
cov_build <- build_stm_covariates(docs_stm, submissions)
stopifnot(identical(cov_build$doc_key, docs_stm$doc_key))
meta_df <- cov_build$meta
docs_stm <- docs_stm |>
  mutate(thread_id = as.character(meta_df$thread_id), parent_level = as.character(meta_df$parent_id),
         linked_submission_in_sample = cov_build$linked_submission_in_sample,
         num_comments_observed = cov_build$num_comments_observed, upvote_ratio_observed = cov_build$upvote_ratio_observed,
         num_comments_covariate = meta_df$num_comments, upvote_ratio_covariate = meta_df$upvote_ratio,
         num_comments_missing = meta_df$num_comments_missing, upvote_ratio_missing = meta_df$upvote_ratio_missing)
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
# The smooth terms as stm's s() builds them (df defaults to min(10, distinct values - 1)); knots without the placeholder rows for comparison.
basis_of <- function(x) { b <- s(x); list(df = ncol(b), degree = attr(b, "degree"), knots = attr(b, "knots"), boundary = attr(b, "Boundary.knots"), distinct = length(unique(x))) }
spline_info <- lapply(setNames(smooth_terms, smooth_terms), function(v) basis_of(meta_df[[v]]))
spline_info_observed <- list(num_comments = basis_of(meta_df$num_comments[meta_df$num_comments_missing == 0L]), upvote_ratio = basis_of(meta_df$upvote_ratio[meta_df$upvote_ratio_missing == 0L]))
fmt_knots <- function(k) paste(format(k, digits = 10, trim = TRUE), collapse = "; ")
spline_knots <- bind_rows(lapply(smooth_terms, function(v) {
  si <- spline_info[[v]]
  tibble(covariate = v, basis = sprintf("stm::s(%s) = splines::bs(x, df = %d, degree = %d), knots at quantiles", v, si$df, si$degree), df = si$df, degree = si$degree,
         distinct_values = si$distinct, interior_knots = fmt_knots(si$knots), boundary_knots = fmt_knots(si$boundary),
         interior_knots_without_placeholder_rows = if (v %in% names(spline_info_observed)) fmt_knots(spline_info_observed[[v]]$knots) else "not applicable (no placeholder rows)",
         max_abs_knot_shift_from_placeholder_rows = if (v %in% names(spline_info_observed)) max(abs(si$knots - spline_info_observed[[v]]$knots)) else NA_real_)
}))
# The prevalence design matrix as stm builds it (Matrix::sparse.model.matrix on the same formula and data; kept sparse when at least 50% sparse and 50+ columns).
t_design <- Sys.time()
design_terms <- terms(prevalence_formula, data = meta_df)
X_design <- Matrix::sparse.model.matrix(design_terms, data = meta_df)
seconds_design <- secs_since(t_design)
design_assign <- attr(X_design, "assign"); design_term_labels <- c("(Intercept)", attr(design_terms, "term.labels"))
design_columns_by_term <- tibble(term = design_term_labels[design_assign + 1L]) |> count(term, name = "design_columns") |>
  mutate(term = factor(term, levels = design_term_labels)) |> arrange(term) |> mutate(term = as.character(term))
design_prop_sparse <- 1 - Matrix::nnzero(X_design) / prod(dim(X_design))
design_nz_per_row <- tabulate(X_design@i + 1L, nrow(X_design))
reference_level <- function(f) levels(f)[1]
covariate_summary <- bind_rows(
  tibble(item = "prevalence formula", value = prevalence_string),
  tibble(item = "content formula", value = content_string),
  tibble(item = "prevalence regularisation", value = "gamma.prior = 'L1': glmnet mgaussian (grouped lasso across topics, gamma.enet = 1), regularisation parameter chosen by AIC (gamma.ic.k = 2); package defaults; the intercept column is unpenalised"),
  tibble(item = "content regularisation", value = "kappa.prior = 'L1' (package default): distributed Poisson regressions via glmnet, fixed intercept at the background word distribution, topic x aspect interactions"),
  tibble(item = "author", value = sprintf("factor, %d levels (reference level '%s'); comment and submission authors share the level space; '[deleted]' is a level (%d documents)", nlevels(meta_df$author), reference_level(meta_df$author), sum(meta_df$author == "[deleted]"))),
  tibble(item = "created_utc", value = sprintf("numeric epoch seconds as stored, entered as s(created_utc); range %s to %s; %d distinct", format(min(meta_df$created_utc), digits = 12), format(max(meta_df$created_utc), digits = 12), spline_info$created_utc$distinct)),
  tibble(item = "doc_type", value = sprintf("factor: comment (reference, %d), submission (%d); also the content covariate", sum(meta_df$doc_type == "comment"), sum(meta_df$doc_type == "submission"))),
  tibble(item = "thread_id", value = sprintf("factor, %d levels (reference level '%s'): comments = the established link_id ('t3_' + submission id); submissions = own fullname ('t3_' + id), the same string as their comments' link_id; threads whose submission is absent from the sample keep their own level (%d levels); submissions without a modelled comment keep their own level (%d levels)",
                                            nlevels(meta_df$thread_id), reference_level(meta_df$thread_id), sum(!(levels(meta_df$thread_id) %in% paste0("t3_", submissions$id))), sum(!(levels(meta_df$thread_id) %in% meta_df$thread_id[meta_df$doc_type == "comment"])))),
  tibble(item = "parent_id", value = sprintf("factor, %d levels (reference level '%s'): comments = stored parent_id (t1_ comment parents %d levels, t3_ submission parents %d levels; parents absent from the sample keep their own level); submissions = the explicit level NO_PARENT_SUBMISSION (%d documents)",
                                            nlevels(meta_df$parent_id), reference_level(meta_df$parent_id), sum(startsWith(levels(meta_df$parent_id), "t1_")), sum(startsWith(levels(meta_df$parent_id), "t3_")), sum(meta_df$parent_id == "NO_PARENT_SUBMISSION"))),
  tibble(item = "score", value = sprintf("numeric as stored (own value of every record), entered as s(score); range %s to %s; %d distinct", format(min(meta_df$score)), format(max(meta_df$score)), spline_info$score$distinct)),
  tibble(item = "num_comments", value = sprintf("submissions: own stored value; comments: inherited from the linked submission (link_id join, %d comments); not observed for %d comments whose submission is absent from the sample: placeholder %s (median of the observed values over modelled documents) with num_comments_missing = 1; entered as s(num_comments)",
                                               sum(docs_stm$doc_type == "comment" & docs_stm$num_comments_missing == 0L), sum(docs_stm$num_comments_missing == 1L), format(cov_build$placeholders[["num_comments"]]))),
  tibble(item = "upvote_ratio", value = sprintf("submissions: own stored value; comments: inherited from the linked submission (link_id join, %d comments); not observed for %d comments whose submission is absent from the sample: placeholder %s (median of the observed values over modelled documents) with upvote_ratio_missing = 1; entered as s(upvote_ratio)",
                                               sum(docs_stm$doc_type == "comment" & docs_stm$upvote_ratio_missing == 0L), sum(docs_stm$upvote_ratio_missing == 1L), format(cov_build$placeholders[["upvote_ratio"]]))),
  tibble(item = "num_comments_missing, upvote_ratio_missing", value = "0/1 indicator columns; 1 marks the placeholder rows, so the placeholder (a constant) is absorbed by the indicator's coefficient and is distinguishable from genuinely observed values"),
  tibble(item = "smooth terms", value = sprintf("stm::s(x) with its documented default df = min(10, distinct values - 1) = 10 for all four; splines::bs cubic B-spline with 7 interior knots at quantiles of the covariate over all modelled documents (spline_knots.csv); no manual tuning")),
  tibble(item = "design matrix", value = sprintf("%d x %d, %d nonzeros, proportion sparse %.6f, kept sparse by stm (dgCMatrix); nonzeros per row %d to %d (median %d); built by Matrix::sparse.model.matrix in %.1f s; %d MB",
                                                nrow(X_design), ncol(X_design), Matrix::nnzero(X_design), design_prop_sparse, min(design_nz_per_row), max(design_nz_per_row), as.integer(median(design_nz_per_row)), seconds_design, round(as.numeric(object.size(X_design)) / 2^20))),
  tibble(item = "design columns by term", value = paste(sprintf("%s: %d", design_columns_by_term$term, design_columns_by_term$design_columns), collapse = "; ")),
  tibble(item = "covariate values missing in the table passed to stm", value = as.character(sum(is.na(meta_df)))),
  tibble(item = "fields preserved outside the formulas (linkage table only)", value = "doc_key, doc_id, source file, url, original author, original link_id, original parent_id, submission_id, created_date_utc, Tier 2 split, text-field names, token counts, model status"))
usable <- function(x) !is.na(x) & !(is.character(x) & x == "")
marker_strings <- c("[deleted]", "[removed]", "[ Removed by Reddit ]", "[ Removed by moderator ]")
coverage_row <- function(field, x, scope, route) {
  n <- length(x); u <- usable(x)
  tibble(field = field, scope = scope, route = route, n_documents = n, n_usable = sum(u), pct_usable = round(100 * sum(u) / n, 4),
         n_distinct_usable = n_distinct(x[u]), n_missing = sum(is.na(x)), n_empty = sum(!is.na(x) & is.character(x) & x == ""),
         n_marker_valued = if (is.character(x)) sum(x %in% marker_strings) else 0L)
}
stm_c <- docs_stm[docs_stm$doc_type == "comment", ]; stm_s <- docs_stm[docs_stm$doc_type == "submission", ]
covariate_coverage <- bind_rows(
  lapply(c("author", "created_utc", "score", "thread_id", "parent_level"), function(f) coverage_row(f, docs_stm[[f]], "all STM documents", "covariate (native or constructed)")),
  lapply(c("num_comments_observed", "upvote_ratio_observed"), function(f) coverage_row(f, stm_s[[f]], "STM submissions", "own stored value")),
  lapply(c("num_comments_observed", "upvote_ratio_observed"), function(f) coverage_row(f, stm_c[[f]], "STM comments", "inherited from the linked submission (link_id join against all source submissions); missing = submission absent from the sample")),
  lapply(c("num_comments_covariate", "upvote_ratio_covariate"), function(f) coverage_row(f, docs_stm[[f]], "all STM documents", "value entered in the design (observed or placeholder)")),
  lapply(c("doc_key", "doc_id", "author", "created_utc", "score", "link_id", "parent_id", "submission_id"), function(f) coverage_row(f, stm_c[[f]], "STM comments", "preserved linkage field")),
  lapply(c("doc_key", "doc_id", "author", "created_utc", "score", "link_id", "url", "num_comments", "upvote_ratio"), function(f) coverage_row(f, stm_s[[f]], "STM submissions", "preserved linkage field")))
missingness <- cov_build$missingness
document_length_summary_stm <- docs_stm |> group_by(doc_type) |>
  summarise(docs = n(), tokens = sum(n_tokens_stm), min = min(n_tokens_stm), p05 = quantile(n_tokens_stm, 0.05, type = 7, names = FALSE), p25 = quantile(n_tokens_stm, 0.25, type = 7, names = FALSE),
            median = median(n_tokens_stm), mean = mean(n_tokens_stm), p75 = quantile(n_tokens_stm, 0.75, type = 7, names = FALSE), p95 = quantile(n_tokens_stm, 0.95, type = 7, names = FALSE), max = max(n_tokens_stm), .groups = "drop")
rm(dtm_all); invisible(gc())
mark_stage("stm_input")
log_line("stm input:", length(documents), "documents;", length(vocab), "terms;", sum(doc_tokens_stm), "tokens; design", paste(dim(X_design), collapse = " x "), "columns built in", round(seconds_design), "s")

# ---- Common held-out construction (stm::make.heldout with searchK's defaults and a recorded seed) ----
# INPUT : documents; vocab.
# DOES  : hold out, at random under heldout_seed, half of the tokens of floor(0.1 N) documents
#         (documents with a single distinct term are skipped by the package function); every
#         document survives; terms whose every occurrence was held out are dropped from the
#         vocabulary by the package function and the documents re-indexed. The object is saved
#         under the input key so every K (and any re-run) uses exactly the same held-out data.
# OUTPUT: heldout (documents, vocab, missing); heldout_construction table; models/heldout_common.rds.
input_key <- digest(list(dim(dtm_stm), length(dtm_stm@x), sum(dtm_stm), digest(vocab), digest(docs_stm$doc_key), digest(meta_df), heldout_seed, heldout_N_share, heldout_proportion), algo = "sha256")
heldout_N <- floor(heldout_N_share * length(documents))
heldout_key_file <- file.path(model_dir, "heldout_common_key.txt")
heldout_from_cache <- !force_refit && file.exists(heldout_path) && file.exists(heldout_key_file) && identical(readLines(heldout_key_file, warn = FALSE)[1], input_key)
t1 <- Sys.time()
if (heldout_from_cache) {
  heldout <- readRDS(heldout_path)
  seconds_heldout <- NA_real_
  log_line("held-out object loaded from the cache")
} else {
  heldout <- make.heldout(documents, vocab, N = heldout_N, proportion = heldout_proportion, seed = heldout_seed)
  seconds_heldout <- secs_since(t1)
  saveRDS(heldout, heldout_path, compress = FALSE)
  writeLines(input_key, heldout_key_file)
  log_line("held-out object built in", round(seconds_heldout), "s and saved")
}
set.seed(heldout_seed); sampled_index <- sort(sample(seq_along(documents), heldout_N))   # make.heldout draws this index first under the seed
heldout_doc_tokens <- vapply(heldout$documents, function(d) sum(d[2, ]), numeric(1))
missing_tokens_by_doc <- vapply(heldout$missing$docs, function(d) sum(d[2, ]), numeric(1))
heldout_terms_dropped <- setdiff(vocab, heldout$vocab)
term_tf_stm_all <- colSums(dtm_stm)
dropped_term_tokens <- if (length(heldout_terms_dropped)) sum(term_tf_stm_all[heldout_terms_dropped]) else 0   # every occurrence of a dropped term was held out; make.heldout removes them from the held-out set too
ho_type <- docs_stm$doc_type[heldout$missing$index]
heldout_construction <- tibble(
  item = c("function", "seed", "documents (N)", "documents sampled for partial hold-out (floor(0.1 N))", "sampled documents with a single distinct term (skipped by make.heldout)",
           "documents with held-out tokens", "  of which comments", "  of which submissions", "documents after construction (every document survives)", "tokens before", "tokens held out",
           "tokens remaining in the fitted documents", "held-out tokens as a share of the sampled documents' tokens", "held-out tokens per held-out document (min / median / max)",
           "vocabulary before", "vocabulary after (terms whose every occurrence was held out are dropped)", "terms dropped", "held-out tokens of dropped terms (removed from the held-out set as well; not evaluable)",
           "held-out documents dropped because all their held-out terms left the vocabulary", "construction seconds"),
  value = c("stm::make.heldout (the function searchK() calls)", heldout_seed, length(documents), heldout_N, sum(doc_terms_stm[sampled_index] < 2L),
            length(heldout$missing$index), sum(ho_type == "comment"), sum(ho_type == "submission"), length(heldout$documents), sum(doc_tokens_stm), sum(missing_tokens_by_doc),
            sum(heldout_doc_tokens), sprintf("%.4f", sum(missing_tokens_by_doc) / sum(doc_tokens_stm[heldout$missing$index])),
            sprintf("%d / %d / %d", min(missing_tokens_by_doc), as.integer(median(missing_tokens_by_doc)), max(missing_tokens_by_doc)),
            length(vocab), length(heldout$vocab), length(heldout_terms_dropped), dropped_term_tokens, sum(doc_terms_stm[sampled_index] >= 2L) - length(heldout$missing$index),
            if (is.na(seconds_heldout)) "loaded from cache" else sprintf("%.1f", seconds_heldout)))
mark_stage("heldout")

# ---- Common input for the workers ----
# INPUT : heldout; meta_df; the STM settings.
# DOES  : save everything a worker needs (the held-out documents and vocabulary, the missing tokens,
#         the covariate table, the formula strings and settings) under a run key of the input and
#         the settings; write the sampler script.
# OUTPUT: models/common_input.rds; run_key; models/memory_sampler.R.
settings <- list(K_values = K_values, init_type = init_type, stm_seed = stm_seed, max_em_its = max_em_its, emtol = emtol, report_every = report_every,
                 gamma_prior = gamma_prior, kappa_prior = kappa_prior, sigma_prior = sigma_prior, interactions = interactions, coherence_M = coherence_M,
                 label_n_words = label_n_words, prevalence_string = prevalence_string, content_string = content_string, heldout_seed = heldout_seed,
                 heldout_N = heldout_N, heldout_proportion = heldout_proportion, gamma_pmax = gamma_pmax, fit_chunk = fit_chunk)
run_key <- digest(list(input_key, settings, as.character(packageVersion("stm")), as.character(packageVersion("glmnet")), R.version.string), algo = "sha256")
saveRDS(list(run_key = run_key, input_key = input_key, K_values = K_values, documents = heldout$documents, vocab = heldout$vocab, missing = heldout$missing,
             meta_df = meta_df, prevalence_string = prevalence_string, content_string = content_string, settings = settings, doc_key = docs_stm$doc_key,
             quick_run = quick_run, written_by_run = run_id), common_input_path, compress = FALSE)
writeLines(sampler_code, sampler_path)
writeLines(run_key, file.path(model_dir, "run_key.txt"))
mark_stage("common_input")
log_line("common input saved; run key", substr(run_key, 1, 12))

# ---- Self-check: the searchK reimplementation against a literal searchK() call ----
# INPUT : documents; vocab; meta_df.
# DOES  : on a small seeded subset (re-indexed vocabulary, unused factor levels dropped), run
#         stm::searchK() literally with the same specification, then repeat its body step by step
#         (make.heldout under the same seed, stm, eval.heldout, checkResiduals, bound, lbound,
#         em.its) and compare every number; record the package warning searchK emits for content
#         covariate models and which statistics it omits.
# OUTPUT: selfcheck table; selfcheck_equal.
t1 <- Sys.time()
set.seed(selfcheck_seed)
sc_idx <- sort(unique(c(sample(length(documents), min(selfcheck_n, length(documents))), which(meta_df$doc_type == "submission")[seq_len(min(100L, sum(meta_df$doc_type == "submission")))])))
sc_docs <- documents[sc_idx]
sc_present <- sort(unique(unlist(lapply(sc_docs, function(d) d[1, ]))))
sc_map <- integer(length(vocab)); sc_map[sc_present] <- seq_along(sc_present)
sc_docs <- lapply(sc_docs, function(d) { d[1, ] <- sc_map[d[1, ]]; storage.mode(d) <- "integer"; d })
sc_vocab <- vocab[sc_present]
sc_meta <- droplevels(meta_df[sc_idx, ])
sc_warnings <- character(0)
sc_literal <- withCallingHandlers(
  searchK(sc_docs, sc_vocab, K = selfcheck_K, init.type = init_type, heldout.seed = selfcheck_seed, prevalence = prevalence_formula, content = content_formula, data = sc_meta,
          gamma.prior = gamma_prior, kappa.prior = kappa_prior, sigma.prior = sigma_prior, interactions = interactions, max.em.its = selfcheck_its, emtol = emtol, verbose = FALSE, seed = stm_seed),
  warning = function(w) { sc_warnings <<- c(sc_warnings, conditionMessage(w)); invokeRestart("muffleWarning") })
sc_literal_tbl <- as_tibble(lapply(sc_literal$results, function(col) unlist(col)))
sc_ho <- make.heldout(sc_docs, sc_vocab, N = floor(0.1 * length(sc_docs)), proportion = 0.5, seed = selfcheck_seed)
sc_mine <- bind_rows(lapply(selfcheck_K, function(k) {
  m <- stm(sc_ho$documents, sc_ho$vocab, K = k, prevalence = prevalence_formula, content = content_formula, data = sc_meta, init.type = init_type, seed = stm_seed,
           max.em.its = selfcheck_its, emtol = emtol, verbose = FALSE, gamma.prior = gamma_prior, sigma.prior = sigma_prior, kappa.prior = kappa_prior, interactions = interactions)
  tibble(K = k, heldout = eval.heldout(m, sc_ho$missing)$expected.heldout, residual = checkResiduals(m, sc_ho$documents)$dispersion,
         bound = max(m$convergence$bound), lbound = max(m$convergence$bound) + lfactorial(k), em.its = length(m$convergence$bound))
}))
selfcheck_equal <- identical(names(sc_literal_tbl), names(sc_mine)) && isTRUE(all.equal(as.data.frame(sc_literal_tbl), as.data.frame(sc_mine), tolerance = 0, check.attributes = FALSE))
selfcheck <- bind_rows(
  sc_literal_tbl |> mutate(source = "stm::searchK() (literal call)"),
  sc_mine |> mutate(source = "reimplementation (make.heldout, stm, eval.heldout, checkResiduals, bound, lbound, em.its)")) |>
  select(source, everything()) |>
  mutate(documents_in_subset = length(sc_docs), vocabulary_in_subset = length(sc_vocab), max_em_its = selfcheck_its, heldout_seed = selfcheck_seed,
         statistics_returned_by_searchK = paste(names(sc_literal$results), collapse = ", "),
         searchK_warning = paste(unique(sc_warnings), collapse = " | "), all_values_identical = selfcheck_equal, seconds = round(secs_since(t1), 1))
rm(sc_docs, sc_ho, sc_literal, sc_meta); invisible(gc())
mark_stage("selfcheck")
log_line("self-check: reimplementation identical to literal searchK:", selfcheck_equal, "; searchK returned", paste(names(sc_literal_tbl), collapse = ", "), "; warning:", paste(unique(sc_warnings), collapse = " | "))

# ---- Fit K = 10, 20, 30 in worker processes (or load them from the cache) ----
# INPUT : models/common_input.rds; run_key; the worker cache files.
# DOES  : for every K whose results are not cached under this run key, launch this script in
#         worker mode as a separate Rscript process (at most `concurrency` at once, largest K
#         first), record the launch (pid, system memory available), poll until every worker has
#         ended, log progress lines from the fit consoles, and record failures without stopping
#         the other fits. Then load every completed K's model and diagnostics.
# OUTPUT: worker_runs table; fits (list by K of model, fit_record, heldout_eval, semcoh, labels,
#         residuals); K_done.
worker_result_files <- function(K) c(worker_file(K, "stm"), worker_file(K, "fit_record"), worker_file(K, "heldout_eval"), worker_file(K, "semcoh"), worker_file(K, "labels"), worker_file(K, "residuals"))
worker_state <- function(K) { sf <- worker_status_path(K); if (!file.exists(sf)) c("absent", NA, NA, NA) else { x <- readLines(sf, warn = FALSE); c(x, rep(NA, 4 - length(x)))[1:4] } }
worker_done <- function(K) { st <- worker_state(K); !force_refit && identical(st[1], "done") && identical(st[2], run_key) && all(file.exists(worker_result_files(K))) }
launch_worker <- function(K) {
  processx::process$new(rscript_path, args = script_path,
                        env = c("current", STM_SEARCHK_WORKER_K = as.character(K), STM_SEARCHK_RUN_KEY = run_key),
                        stdout = worker_file(K, "worker_console", "log"), stderr = "2>&1", cleanup = FALSE, cleanup_tree = FALSE)
}
last_fit_line <- function(K) {
  fl <- worker_file(K, "fit_console", "log")
  st <- worker_state(K)
  if (!file.exists(fl)) return(sprintf("status %s", st[1]))
  lines <- readLines(fl, warn = FALSE)
  it <- grep("^Completing Iteration|^Model (Converged|Terminated)|^fit end", lines, value = TRUE)
  stage <- { sf <- file.path(model_dir, sprintf("stage_marker_%s.txt", k_tag(K))); if (file.exists(sf)) readLines(sf, warn = FALSE)[1] else NA }
  sprintf("stage %s; %s", stage, if (length(it)) trimws(tail(it, 1)) else sprintf("%d E-steps so far", length(grep("^Completed E-Step", lines))))
}
if (force_refit) for (K in K_values) if (file.exists(worker_status_path(K))) invisible(file.remove(worker_status_path(K)))
pending  <- K_values[!vapply(K_values, worker_done, logical(1))]
pending  <- pending[order(-pending)]            # the longest chain first
cached_K <- setdiff(K_values, pending)
if (length(cached_K)) log_line("cached under this run key: K =", paste(sort(cached_K), collapse = ", "))
procs <- list(); running <- integer(0); launch_rows <- list(); end_rows <- list()
last_progress <- Sys.time(); wait_start <- Sys.time()
while (length(pending) || length(running)) {
  while (length(pending) && length(running) < concurrency) {
    K <- pending[1]; pending <- pending[-1]
    sm <- ps::ps_system_memory()
    p <- launch_worker(K)
    procs[[k_tag(K)]] <- p; running <- c(running, K)
    launch_rows[[k_tag(K)]] <- tibble(K = K, pid = p$get_pid(), launched_utc = utc_stamp(), concurrent_fits_at_launch = length(running),
                                      system_memory_available_mb_at_launch = round(as.numeric(sm$avail) / 2^20), system_memory_total_mb = round(as.numeric(sm$total) / 2^20))
    log_line("worker K =", K, "launched (pid", p$get_pid(), "); running:", paste(running, collapse = ", "), "; system memory available", round(as.numeric(sm$avail) / 2^30, 1), "GB")
    if (length(pending) && length(running) < concurrency) Sys.sleep(launch_stagger)
  }
  Sys.sleep(poll_seconds)
  for (K in running) {
    p <- procs[[k_tag(K)]]
    if (!p$is_alive()) {
      st <- worker_state(K)
      done <- worker_done(K)
      end_rows[[k_tag(K)]] <- tibble(K = K, ended_utc = utc_stamp(), exit_status = p$get_exit_status(), final_status = st[1], completed = done)
      if (done) log_line("worker K =", K, "completed") else log_line("worker K =", K, "ENDED WITHOUT COMPLETING: status", st[1], "; exit status", p$get_exit_status())
      running <- setdiff(running, K)
    }
  }
  if (length(running) && secs_since(last_progress) >= progress_every) {
    for (K in running) log_line("progress K =", K, ":", last_fit_line(K))
    last_progress <- Sys.time()
  }
}
worker_runs <- if (length(launch_rows)) bind_rows(launch_rows) |> left_join(bind_rows(end_rows), by = "K") else
  tibble(K = integer(0), pid = integer(0), launched_utc = character(0), concurrent_fits_at_launch = integer(0), system_memory_available_mb_at_launch = numeric(0),
         system_memory_total_mb = numeric(0), ended_utc = character(0), exit_status = integer(0), final_status = character(0), completed = logical(0))
worker_runs <- bind_rows(worker_runs, tibble(K = cached_K, final_status = rep("done (cached from an earlier run under this run key)", length(cached_K)), completed = rep(TRUE, length(cached_K)))) |> arrange(K)
K_done   <- sort(K_values[vapply(K_values, worker_done, logical(1))])
K_failed <- setdiff(K_values, K_done)
mark_stage("fits")
log_line("fits: completed K =", paste(K_done, collapse = ", "), if (length(K_failed)) paste("; FAILED K =", paste(K_failed, collapse = ", ")) else "", "; waited", round(secs_since(wait_start)), "s")
if (!length(K_done)) stop("No K completed; see models/worker_*_status.txt and models/worker_console_*.log")

load_worker <- function(K) {
  gp <- worker_file(K, "gamma_path", "csv")
  out <- list(K = K, model = readRDS(worker_file(K, "stm")), fit_record = readRDS(worker_file(K, "fit_record")), heldout_eval = readRDS(worker_file(K, "heldout_eval")),
              semcoh = readRDS(worker_file(K, "semcoh")), labels = readRDS(worker_file(K, "labels")), residuals = readRDS(worker_file(K, "residuals")),
              gamma_path = if (file.exists(gp)) read_csv(gp, show_col_types = FALSE, progress = FALSE, col_types = cols(time_utc = col_character(), chosen_is_last_computed = col_logical(), .default = col_double())) |> mutate(K = K) |> select(K, everything()) else NULL)
  stopifnot(identical(out$fit_record$run_key, run_key), identical(out$heldout_eval$run_key, run_key), identical(out$semcoh$run_key, run_key),
            identical(out$labels$run_key, run_key), identical(out$residuals$run_key, run_key), out$model$settings$dim$K == K,
            nrow(out$model$theta) == length(heldout$documents), out$model$settings$dim$V == length(heldout$vocab))
  out
}
fits <- lapply(setNames(K_done, k_tag(K_done)), load_worker)
mark_stage("load_fits")
log_line("fits loaded:", paste(sprintf("K = %d (%d iterations, converged %s)", K_done, vapply(fits, function(f) f$model$convergence$its, numeric(1)), vapply(fits, function(f) f$model$convergence$converged, logical(1))), collapse = "; "))

# ---- Per-K summaries: searchK statistics, convergence, runtime, topics, documents ----
# INPUT : fits; heldout; docs_stm; docs; X_design.
# DOES  : for every completed K read the searchK statistics (held-out likelihood, residual
#         dispersion, bound, lbound, em.its), the separately computed semantic coherence and the
#         exclusivity attempt, the convergence trace with per-iteration E- and M-step seconds
#         parsed from the fit console, the runtime and memory record (fit record and sampler
#         trace), the marginal (aspect-weighted) topic-word distribution, sageLabels/labelTopics
#         word rankings and kappa words, NPMI coherence over the top ten marginal words, topic
#         prevalence by document type, document concentration, representative documents under
#         findThoughts and the Tier 2 rule, the L1 prevalence coefficients by covariate term
#         (gamma sparsity), the content-covariate kappa sparsity, held-out likelihood by document
#         type, and theta correlations.
# OUTPUT: one long table per quantity with a K column; per-K objects kept in `per_k`.
texts_stm <- docs$text[modeled_rows]
vocab_fit <- heldout$vocab
V_fit     <- length(vocab_fit)
n_tok_fit <- heldout_doc_tokens
ho_flag   <- seq_along(heldout$documents) %in% heldout$missing$index
# Binary document-term matrix of the fitted documents for NPMI co-occurrence (shared by every K).
B_fit <- sparseMatrix(i = rep(seq_along(heldout$documents), vapply(heldout$documents, ncol, integer(1))),
                      j = unlist(lapply(heldout$documents, function(d) d[1, ]), use.names = FALSE), x = 1, dims = c(length(heldout$documents), V_fit))
N_fit <- nrow(B_fit)
type_groups <- as.character(docs_stm$doc_type)
text_id   <- match(texts_stm, unique(texts_stm))
author_id <- match(docs_stm$author, unique(docs_stm$author))
cand_rows <- which(docs_stm$n_tokens_stm >= representative_min_tokens)
design_term_of_column <- design_term_labels[design_assign + 1L]
read_trace <- function(p) if (!is.null(p) && file.exists(p)) read_csv(p, show_col_types = FALSE, progress = FALSE, col_types = cols(time_utc = col_character(), stage = col_character(), .default = col_double())) else NULL
cosine_rows <- function(a, b) { an <- a / sqrt(rowSums(a^2)); bn <- b / sqrt(rowSums(b^2)); an %*% t(bn) }

summarise_fit <- function(f) {
  K <- f$K; Kk <- f$K; model <- f$model; fr <- f$fit_record; he <- f$heldout_eval; sc <- f$semcoh; lb <- f$labels; rs <- f$residuals   # Kk: the scalar for use inside tibble()/mutate(), where a K column would shadow it
  tag <- k_tag(K); topic_ids <- seq_len(K); topic_cols <- sprintf("topic_%02d", topic_ids)
  theta <- model$theta
  stopifnot(nrow(theta) == N_fit, ncol(theta) == K)
  betaindex <- model$settings$covariates$betaindex
  aspect_w  <- as.numeric(table(factor(betaindex, levels = seq_along(model$beta$logbeta)))) / length(betaindex)
  beta_list <- lapply(model$beta$logbeta, exp)
  beta_marg <- Reduce("+", Map(function(b, w) b * w, beta_list, aspect_w))
  aspect_names <- model$settings$covariates$yvarlevels
  # searchK statistics and convergence
  bound <- model$convergence$bound; bound_vec <- bound; its <- model$convergence$its; n_tokens_fit_model <- sum(model$settings$dim$wcounts$x)   # bound_vec: the trace for use inside tibble() calls that also create a `bound` column
  log_timing <- parse_fit_log(fr$log_lines)
  conv <- tibble(K = K, iteration = seq_along(bound), bound = bound, bound_per_token = bound / n_tokens_fit_model,
                 relative_change = c(NA, diff(bound) / abs(head(bound, -1)))) |>
    left_join(log_timing, by = "iteration") |>
    mutate(iteration_seconds = coalesce(estep_seconds, 0) + coalesce(mstep_seconds_reported, 0), cumulative_seconds = cumsum(iteration_seconds))
  init_seconds <- fr$wall_seconds - sum(conv$estep_seconds, na.rm = TRUE) - sum(conv$mstep_seconds_reported, na.rm = TRUE)
  searchk_row <- tibble(K = Kk, heldout = he$expected_heldout, residual = rs$dispersion, bound = max(bound_vec), lbound = max(bound_vec) + lfactorial(Kk), em.its = length(bound_vec),
                        converged = model$convergence$converged, convergence_message = fit_log_message(fr$log_lines), max_em_its = model$settings$convergence$max.em.its,
                        emtol = model$settings$convergence$em.converge.thresh, last_relative_change = if (length(bound_vec) > 1) tail(conv$relative_change, 1) else NA_real_,
                        final_bound_per_token = tail(bound_vec, 1) / n_tokens_fit_model, residual_pvalue = rs$pvalue, residual_df = rs$df,
                        exclus = NA_real_, exclusivity_status = if (is.na(sc$exclusivity_error)) "computed (unexpected for a content covariate model)" else paste0("unavailable: stm::exclusivity() stopped with '", sc$exclusivity_error, "'"),
                        semcoh = mean(sc$semantic_coherence), semcoh_status = "computed separately from searchK() with stm::semanticCoherence() (content-covariate-aware: per-aspect coherence weighted by aspect document counts); searchK() omits it for content covariate models",
                        fit_wall_seconds = fr$wall_seconds, fit_cpu_user_seconds = fr$cpu_user_seconds, fitting_run_id = fr$run_id,
                        gamma_pmax = fr$gamma_pmax, gamma_l1_calls = if (is.null(f$gamma_path)) NA_integer_ else nrow(f$gamma_path),
                        gamma_ic_minimum_interior_every_mstep = if (is.null(f$gamma_path)) NA else !any(f$gamma_path$chosen_is_last_computed),
                        gamma_path_length_min = if (is.null(f$gamma_path)) NA_real_ else min(f$gamma_path$path_length), gamma_path_length_max = if (is.null(f$gamma_path)) NA_real_ else max(f$gamma_path$path_length),
                        gamma_chosen_index_min = if (is.null(f$gamma_path)) NA_real_ else min(f$gamma_path$chosen_index), gamma_chosen_index_max = if (is.null(f$gamma_path)) NA_real_ else max(f$gamma_path$chosen_index),
                        gamma_nonzero_rows_chosen_last = if (is.null(f$gamma_path)) NA_real_ else tail(f$gamma_path$nonzero_rows_chosen, 1), gamma_seconds_mean = if (is.null(f$gamma_path)) NA_real_ else mean(f$gamma_path$seconds),
                        fit_chunks = if (is.null(fr$chunks)) 1L else fr$chunks, iterations_per_chunk = if (is.null(fr$fit_chunk)) NA_integer_ else fr$fit_chunk,
                        fit_wall_seconds_including_saves = if (is.null(fr$wall_seconds_including_saves)) fr$wall_seconds else fr$wall_seconds_including_saves)
  fit_chunks <- if (is.null(fr$chunk_table)) tibble(K = K, chunk = 1L, target_iterations = its, iterations_after = its, converged = model$convergence$converged, wall_seconds = fr$wall_seconds, cpu_user_seconds = fr$cpu_user_seconds,
                                                    cpu_system_seconds = fr$cpu_system_seconds, stm_time_seconds = fr$model_time_seconds, peak_wset_mb = fr$peak_wset_mb_after_fit, run_id = fr$run_id) else fr$chunk_table |> mutate(K = K) |> select(K, everything())
  # runtime and memory
  trace <- read_trace(fr$trace_csv)
  stage_mem <- if (!is.null(trace)) trace |> group_by(stage) |> summarise(rows = n(), max_rss_mb = max(rss_mb), median_rss_mb = median(rss_mb), max_wset_mb = max(wset_mb), min_system_avail_mb = min(system_avail_mb), .groups = "drop") |> mutate(K = K) else NULL
  runtime <- tibble(K = K, stage = c("spectral initialisation (wall minus E- and M-step seconds; approximate)", "E-steps (sum)", "M-steps (sum; L1 prevalence glmnet + content-covariate distributed Poisson regressions)", "fit wall", "fit CPU user",
                                     "held-out evaluation (eval.heldout)", "semantic coherence (semanticCoherence)", "exclusivity attempt", "labels (sageLabels, labelTopics)", "residual dispersion (checkResiduals)", "worker total (fit + diagnostics)"),
                    seconds = c(init_seconds, sum(conv$estep_seconds, na.rm = TRUE), sum(conv$mstep_seconds_reported, na.rm = TRUE), fr$wall_seconds, fr$cpu_user_seconds,
                                he$seconds, sc$seconds, sc$exclusivity_seconds, lb$seconds, rs$seconds, fr$wall_seconds + he$seconds + sc$seconds + sc$exclusivity_seconds + lb$seconds + rs$seconds))
  runtime_summary <- tibble(K = K, em_iterations = its, converged = model$convergence$converged, fit_wall_seconds = fr$wall_seconds, fit_cpu_user_seconds = fr$cpu_user_seconds, fit_cpu_system_seconds = fr$cpu_system_seconds,
                            init_seconds_estimate = init_seconds, estep_seconds_mean = mean(conv$estep_seconds, na.rm = TRUE), estep_seconds_min = min(conv$estep_seconds, na.rm = TRUE), estep_seconds_max = max(conv$estep_seconds, na.rm = TRUE),
                            estep_ms_per_document = 1000 * mean(conv$estep_seconds, na.rm = TRUE) / N_fit, mstep_seconds_mean = mean(conv$mstep_seconds_reported, na.rm = TRUE), mstep_seconds_max = max(conv$mstep_seconds_reported, na.rm = TRUE),
                            iteration_seconds_mean = mean(conv$iteration_seconds), heldout_eval_seconds = he$seconds, semcoh_seconds = sc$seconds, labels_seconds = lb$seconds, residuals_seconds = rs$seconds,
                            worker_total_seconds = fr$wall_seconds + he$seconds + sc$seconds + sc$exclusivity_seconds + lb$seconds + rs$seconds,
                            process_peak_working_set_mb_after_fit = fr$peak_wset_mb_after_fit, process_rss_mb_after_fit = fr$rss_mb_after_fit,
                            process_peak_working_set_mb_after_residuals = rs$peak_wset_mb,
                            sampler_max_rss_mb = if (is.null(trace)) NA else max(trace$rss_mb), sampler_max_wset_mb = if (is.null(trace)) NA else max(trace$wset_mb),
                            sampler_median_rss_mb_during_fit = if (is.null(trace)) NA else median(trace$rss_mb[trace$stage %in% "fit"]), sampler_rows = if (is.null(trace)) 0L else nrow(trace),
                            sampler_min_system_avail_mb = if (is.null(trace)) NA else min(trace$system_avail_mb), model_file_bytes = fr$model_file_bytes, fitting_run_id = fr$run_id, pid = fr$pid)
  # semantic coherence per topic and NPMI over the same ten marginal words
  top10 <- apply(beta_marg, 1, function(x) order(x, decreasing = TRUE)[1:coherence_M])   # M x K
  union_idx <- sort(unique(as.vector(top10)))
  co <- as.matrix(crossprod(B_fit[, union_idx, drop = FALSE]))
  npmi <- vapply(topic_ids, function(k) {
    idx <- match(top10[, k], union_idx); pr <- expand.grid(a = idx, b = idx); pr <- pr[pr$a < pr$b, ]
    pij <- co[cbind(pr$a, pr$b)] / N_fit; pi <- co[cbind(pr$a, pr$a)] / N_fit; pj <- co[cbind(pr$b, pr$b)] / N_fit
    mean(ifelse(pij > 0, (log(pij) - log(pi) - log(pj)) / (-log(pij)), -1))
  }, numeric(1))
  coherence <- tibble(K = K, topic = topic_ids, semantic_coherence_stm = sc$semantic_coherence, coherence_npmi_top10 = npmi,
                      top_10_words_marginal = apply(top10, 2, function(i) paste(vocab_fit[i], collapse = ", ")))
  # prevalence and concentration
  dominant <- max.col(theta, ties.method = "first"); dominant_share <- theta[cbind(seq_len(N_fit), dominant)]
  n_active <- as.integer(rowSums(theta >= active_topic_threshold))
  expected_proportion <- colMeans(theta); token_share <- as.numeric(crossprod(theta, n_tok_fit)) / sum(n_tok_fit)
  prevalence_rank <- rank(-expected_proportion, ties.method = "first")
  by_type <- bind_rows(lapply(c("comment", "submission"), function(tp) {
    sel <- type_groups == tp
    tibble(K = K, doc_type = tp, topic = topic_ids, documents = sum(sel), mean_theta = colMeans(theta[sel, , drop = FALSE]),
           token_share = as.numeric(crossprod(theta[sel, , drop = FALSE], n_tok_fit[sel])) / sum(n_tok_fit[sel]),
           documents_dominant = tabulate(dominant[sel], Kk), share_documents_dominant = tabulate(dominant[sel], Kk) / sum(sel))
  }))
  conc_df <- tibble(doc_type = type_groups, token_band = cut(n_tok_fit, breaks = band_breaks, labels = band_labels, right = TRUE), dominant_topic_share = dominant_share, n_active_topics = n_active)
  concentration <- bind_rows(
    conc_df |> group_by(doc_type, token_band) |>
      summarise(docs = n(), median_dominant_share = median(dominant_topic_share), p25_dominant_share = quantile(dominant_topic_share, 0.25, names = FALSE), p75_dominant_share = quantile(dominant_topic_share, 0.75, names = FALSE),
                share_docs_dominant_at_least_half = mean(dominant_topic_share >= 0.5), mean_active_topics = mean(n_active_topics), .groups = "drop") |> mutate(token_band = as.character(token_band)),
    conc_df |> group_by(doc_type) |>
      summarise(docs = n(), median_dominant_share = median(dominant_topic_share), p25_dominant_share = quantile(dominant_topic_share, 0.25, names = FALSE), p75_dominant_share = quantile(dominant_topic_share, 0.75, names = FALSE),
                share_docs_dominant_at_least_half = mean(dominant_topic_share >= 0.5), mean_active_topics = mean(n_active_topics), .groups = "drop") |> mutate(token_band = "all"),
    conc_df |> summarise(docs = n(), median_dominant_share = median(dominant_topic_share), p25_dominant_share = quantile(dominant_topic_share, 0.25, names = FALSE), p75_dominant_share = quantile(dominant_topic_share, 0.75, names = FALSE),
                         share_docs_dominant_at_least_half = mean(dominant_topic_share >= 0.5), mean_active_topics = mean(n_active_topics)) |> mutate(doc_type = "all", token_band = "all")) |>
    mutate(K = Kk, uniform_share = 1 / Kk) |> select(K, doc_type, token_band, everything())
  dominant_share_quantiles <- tibble(K = K, quantity = "dominant-topic share over all documents", p05 = quantile(dominant_share, 0.05, names = FALSE), p25 = quantile(dominant_share, 0.25, names = FALSE), median = median(dominant_share),
                                     p75 = quantile(dominant_share, 0.75, names = FALSE), p95 = quantile(dominant_share, 0.95, names = FALSE), mean = mean(dominant_share), share_at_least_half = mean(dominant_share >= 0.5),
                                     mean_active_topics = mean(n_active), mean_entropy_bits = mean(-rowSums(theta * log2(theta + 1e-12))), max_entropy_bits = log2(Kk))
  top_author <- tibble(dominant_topic = dominant, author = docs_stm$author) |> count(dominant_topic, author, name = "n") |> group_by(dominant_topic) |> mutate(share = n / sum(n)) |>
    slice_max(order_by = n, n = 1, with_ties = FALSE) |> ungroup() |> transmute(topic = dominant_topic, top_author_among_dominant = author, top_author_share_among_dominant = share)
  # word rankings: marginal (sageLabels), per aspect, kappa
  sage <- lb$sage; lab <- lb$labels
  word_prob <- function(k, terms) beta_marg[k, match(terms, vocab_fit)]
  top_words <- bind_rows(
    bind_rows(lapply(c("prob", "frex", "lift", "score"), function(type) { m <- sage$marginal[[type]]; bind_rows(lapply(topic_ids, function(k) tibble(word_set = "marginal (aspect-weighted beta)", ranking = type, topic = k, rank = seq_len(ncol(m)), term = m[k, ], marginal_probability = word_prob(k, m[k, ])))) })),
    bind_rows(lapply(seq_along(aspect_names), function(a) bind_rows(lapply(c("problabels", "frexlabels", "liftlabels", "scorelabels"), function(type) {
      m <- sage$cov.betas[[a]][[type]]
      bind_rows(lapply(topic_ids, function(k) tibble(word_set = sprintf("aspect: %s", aspect_names[a]), ranking = sub("labels$", "", type), topic = k, rank = seq_len(ncol(m)), term = m[k, ],
                                                    marginal_probability = word_prob(k, m[k, ]), aspect_probability = beta_list[[a]][k, match(m[k, ], vocab_fit)])))
    }))))) |> mutate(K = K) |> select(K, everything())
  kp <- model$beta$kappa$params; A <- length(aspect_names)
  kappa_type <- c(rep("topic", K), rep("aspect", A), rep("topic x aspect interaction", K * A))
  kappa_topic <- c(topic_ids, rep(NA, A), rep(topic_ids, A)); kappa_aspect <- c(rep(NA, K), aspect_names, rep(aspect_names, each = K))
  kappa_labels <- bind_rows(lapply(seq_along(kp), function(i) {
    x <- kp[[i]]; o <- order(x, decreasing = TRUE)[seq_len(min(label_n_words, sum(x > 0)))]
    if (!length(o)) return(NULL)
    tibble(K = K, parameter_index = i, parameter_type = kappa_type[i], topic = kappa_topic[i], aspect = kappa_aspect[i], rank = seq_along(o), term = vocab_fit[o], kappa = x[o])
  }))
  kappa_sparsity <- tibble(K = K, parameter_index = seq_along(kp), parameter_type = kappa_type, topic = kappa_topic, aspect = kappa_aspect,
                           nonzero_entries = vapply(kp, function(x) sum(x != 0), integer(1)), positive_entries = vapply(kp, function(x) sum(x > 0), integer(1)),
                           max_kappa = vapply(kp, max, numeric(1)), min_kappa = vapply(kp, min, numeric(1)), vocabulary = V_fit)
  # L1 prevalence coefficients (gamma) by covariate term; row 1 is glmnet's intercept, rows 2..P are the design columns after the intercept
  gamma <- model$mu$gamma
  X_model_cols <- colnames(model$settings$covariates$X)
  design_identical <- identical(X_model_cols, colnames(X_design)) && identical(dim(model$settings$covariates$X), dim(X_design))
  design_values_identical <- design_identical && isTRUE(all.equal(as.numeric(model$settings$covariates$X[1:min(2000L, N_fit), ]@x), as.numeric(X_design[1:min(2000L, N_fit), ]@x)))
  nz_row <- rowSums(gamma != 0) > 0
  gamma_term <- c("(Intercept, glmnet)", design_term_of_column[-1])
  gamma_sparsity <- tibble(term = gamma_term, nonzero = nz_row, max_abs = apply(abs(gamma), 1, max)) |>
    group_by(term) |> summarise(coefficient_rows = n(), rows_with_nonzero_coefficients = sum(nonzero), share_nonzero = sum(nonzero) / n(), max_abs_coefficient = max(max_abs), .groups = "drop") |>
    mutate(K = K, term = factor(term, levels = c("(Intercept, glmnet)", design_term_labels[-1]))) |> arrange(term) |> mutate(term = as.character(term)) |> select(K, everything())
  gamma_overall <- tibble(K = K, gamma_rows = nrow(gamma), gamma_columns = ncol(gamma), rows_with_nonzero_coefficients = sum(nz_row), share_rows_nonzero = mean(nz_row),
                          design_columns_equal_to_this_run = design_identical, design_values_equal_on_first_2000_rows = design_values_identical)
  # held-out likelihood by document type
  ho_types <- type_groups[he$index]
  heldout_by_type <- bind_rows(
    tibble(doc_type = ho_types, ll = he$doc_heldout, ntok = he$ntokens) |> group_by(doc_type) |>
      summarise(heldout_documents = n(), heldout_tokens = sum(ntok), mean_per_document_log_likelihood = mean(ll, na.rm = TRUE), token_weighted_log_likelihood = sum(ll * ntok, na.rm = TRUE) / sum(ntok), median_per_document = median(ll, na.rm = TRUE), .groups = "drop"),
    tibble(doc_type = "all", heldout_documents = length(he$doc_heldout), heldout_tokens = sum(he$ntokens), mean_per_document_log_likelihood = he$expected_heldout,
           token_weighted_log_likelihood = sum(he$doc_heldout * he$ntokens, na.rm = TRUE) / sum(he$ntokens), median_per_document = median(he$doc_heldout, na.rm = TRUE))) |>
    mutate(K = K) |> select(K, everything())
  # correlations
  cor_theta <- cor(theta)
  cos_within <- cosine_rows(beta_marg, beta_marg); diag(cos_within) <- NA
  nearest_cos <- apply(cos_within, 1, which.max); nearest_cos_value <- apply(cos_within, 1, max, na.rm = TRUE)
  nearest_cor <- vapply(topic_ids, function(k) { v <- cor_theta[k, ]; v[k] <- -Inf; which.max(v) }, integer(1))
  topic_correlation <- as_tibble(expand.grid(topic_a = topic_ids, topic_b = topic_ids)) |> mutate(K = K, theta_correlation = as.vector(cor_theta), beta_cosine = as.vector(ifelse(is.na(cos_within), 1, cos_within))) |> filter(topic_a < topic_b) |> select(K, everything())
  # representative documents
  ft <- findThoughts(model, texts = texts_stm, topics = topic_ids, n = n_representative)
  rep_row <- function(k, r, idx, rule) {
    tibble(K = K, rule = rule, topic = k, rank = r, stm_row = idx, topic_share_in_document = theta[idx, k],
           doc_key = docs_stm$doc_key[idx], doc_id = docs_stm$doc_id[idx], doc_type = docs_stm$doc_type[idx], author = docs_stm$author[idx], created_utc = docs_stm$created_utc[idx],
           submission_id = docs_stm$submission_id[idx], link_id = docs_stm$link_id[idx], parent_id = docs_stm$parent_id[idx], thread_id = docs_stm$thread_id[idx], score = docs_stm$score[idx],
           tier2_split = docs_stm$split[idx], n_tokens_stm = docs_stm$n_tokens_stm[idx], n_tokens_fitted = n_tok_fit[idx], tokens_partly_held_out = ho_flag[idx],
           dominant_topic = dominant[idx], dominant_topic_share = dominant_share[idx],
           top_10_marginal_words_present = vapply(idx, function(i) sum(heldout$documents[[i]][1, ] %in% top10[, k]), integer(1)), text = texts_stm[idx])
  }
  representative <- bind_rows(
    bind_rows(lapply(topic_ids, function(k) { idx <- ft$index[[k]]; rep_row(k, seq_along(idx), idx, sprintf("findThoughts, n = %d, no restriction", n_representative)) })),
    bind_rows(lapply(topic_ids, function(k) {
      o <- cand_rows[order(-theta[cand_rows, k], docs_stm$doc_key[cand_rows])]
      o <- o[!duplicated(text_id[o]) & !duplicated(author_id[o])]
      idx <- head(o, n_representative)
      rep_row(k, seq_along(idx), idx, sprintf("Tier 2 rule: >= %d tokens, one per distinct text and author, ties by doc_key", representative_min_tokens))
    })))
  topic_summary <- tibble(K = K, topic = topic_ids, prevalence_rank = prevalence_rank, expected_proportion = expected_proportion, token_share = token_share,
                          documents_dominant = tabulate(dominant, Kk), share_documents_dominant = tabulate(dominant, Kk) / N_fit,
                          median_share_when_dominant = vapply(topic_ids, function(k) if (any(dominant == k)) median(dominant_share[dominant == k]) else NA_real_, numeric(1)),
                          mean_theta_comments = by_type$mean_theta[by_type$doc_type == "comment"], mean_theta_submissions = by_type$mean_theta[by_type$doc_type == "submission"],
                          submission_to_comment_ratio = by_type$mean_theta[by_type$doc_type == "submission"] / by_type$mean_theta[by_type$doc_type == "comment"],
                          semantic_coherence_stm = sc$semantic_coherence, coherence_npmi_top10 = npmi,
                          nearest_topic_by_beta_cosine = nearest_cos, nearest_topic_cosine = nearest_cos_value, nearest_topic_by_theta_correlation = nearest_cor,
                          nearest_topic_correlation = cor_theta[cbind(topic_ids, nearest_cor)],
                          kappa_topic_nonzero_terms = kappa_sparsity$nonzero_entries[kappa_sparsity$parameter_type == "topic"],
                          kappa_interaction_nonzero_terms_comment = kappa_sparsity$nonzero_entries[kappa_sparsity$parameter_type == "topic x aspect interaction" & kappa_sparsity$aspect == aspect_names[1]],
                          kappa_interaction_nonzero_terms_submission = kappa_sparsity$nonzero_entries[kappa_sparsity$parameter_type == "topic x aspect interaction" & kappa_sparsity$aspect == aspect_names[2]]) |>
    left_join(top_author, by = "topic") |>
    mutate(top_10_words_marginal_prob = apply(sage$marginal$prob[, 1:10, drop = FALSE], 1, paste, collapse = ", "),
           top_10_words_marginal_frex = apply(sage$marginal$frex[, 1:10, drop = FALSE], 1, paste, collapse = ", "),
           top_10_words_comment_prob = apply(sage$cov.betas[[1]]$problabels[, 1:10, drop = FALSE], 1, paste, collapse = ", "),
           top_10_words_submission_prob = apply(sage$cov.betas[[2]]$problabels[, 1:10, drop = FALSE], 1, paste, collapse = ", "))
  document_table <- docs_stm |>
    transmute(stm_row, doc_key, doc_id, doc_type, tier2_split = split, author, created_utc, created_date_utc = as.character(created_date_utc), link_id, parent_id, thread_id, parent_level,
              submission_id, n_tokens_stm, n_tokens_fitted = n_tok_fit, tokens_partly_held_out = ho_flag, dominant_topic = dominant, dominant_topic_share = round(dominant_share, 6), n_active_topics = n_active) |>
    bind_cols(mat_tibble(round(theta, 6), topic_cols))
  topic_word_table <- tibble(term = vocab_fit, stm_index = seq_len(V_fit), wordcount_fitted = model$settings$dim$wcounts$x, document_frequency_fitted = as.integer(colSums(B_fit))) |>
    bind_cols(mat_tibble(t(beta_marg), paste0("marginal_", topic_cols)), mat_tibble(t(beta_list[[1]]), paste0(aspect_names[1], "_", topic_cols)), mat_tibble(t(beta_list[[2]]), paste0(aspect_names[2], "_", topic_cols)))
  sigma_cor <- cov2cor(model$sigma)
  model_spec <- tibble(K = K, item = c("implementation", "documents (N, fitted: 10% with half their tokens held out)", "vocabulary (V, after make.heldout)", "tokens fitted", "prevalence formula", "content formula",
                                       "gamma prior (prevalence)", "gamma.enet / gamma.ic.k / gamma.maxits", "kappa prior (content)", "kappa interactions / fixedintercept / contrast", "tau nlambda / lambda.min.ratio / ic.k / enet / tol / maxit",
                                       "gamma pmax (memory cap on glmnet's mgaussian path; 0 = untouched package call)", "L1 prevalence calls: path length (min-max) / chosen index (min-max) / chosen index always interior",
                                       "initialisation", "spectral maxV (terms used in the initialisation)", "seed (recorded by stm)", "max EM iterations", "EM tolerance", "allow.neg.change", "sigma prior", "ngroups",
                                       "design matrix (rows x columns; class)", "aspects (content levels)", "EM iterations run", "converged", "final approximate bound", "lbound", "final bound per token", "last relative change",
                                       "fit wall seconds", "fit CPU user seconds", "fit chunks (stm() calls; restarts through model =)", "model object bytes", "fitting run id", "stm version", "glmnet version", "R version"),
                       value = c("stm::stm (variational EM; one R process per K; E-step loops over documents; M-step = glmnet mgaussian for gamma + one Poisson glmnet per term for kappa)",
                                 model$settings$dim$N, model$settings$dim$V, n_tokens_fit_model, prevalence_string, content_string, model$settings$gamma$mode,
                                 sprintf("%s / %s / %s", model$settings$gamma$enet, model$settings$gamma$ic.k, model$settings$gamma$maxits), model$settings$tau$mode,
                                 sprintf("%s / %s / %s", model$settings$kappa$interactions, model$settings$kappa$fixedintercept, model$settings$kappa$contrast),
                                 sprintf("%s / %s / %s / %s / %s / %s", model$settings$tau$nlambda, model$settings$tau$lambda.min.ratio, model$settings$tau$ic.k, model$settings$tau$enet, model$settings$tau$tol, model$settings$tau$maxit),
                                 fr$gamma_pmax, if (is.null(f$gamma_path)) "no record" else sprintf("%d-%d / %d-%d / %s", min(f$gamma_path$path_length), max(f$gamma_path$path_length), min(f$gamma_path$chosen_index), max(f$gamma_path$chosen_index), !any(f$gamma_path$chosen_is_last_computed)),
                                 model$settings$init$mode, if (is.null(model$settings$init$maxV)) "all terms" else model$settings$init$maxV, model$settings$seed,
                                 sprintf("%d (the ceiling in force for every K; the model object records the last stm() call's target, %d)", max_em_its, model$settings$convergence$max.em.its),
                                 format(model$settings$convergence$em.converge.thresh, scientific = TRUE), model$settings$convergence$allow.neg.change, model$settings$sigma$prior, model$settings$ngroups,
                                 sprintf("%d x %d; %s", nrow(model$settings$covariates$X), ncol(model$settings$covariates$X), class(model$settings$covariates$X)[1]), paste(aspect_names, collapse = ", "),
                                 its, model$convergence$converged, sprintf("%.3f", tail(bound, 1)), sprintf("%.3f", tail(bound, 1) + lfactorial(Kk)), sprintf("%.6f", tail(bound, 1) / n_tokens_fit_model),
                                 if (length(bound) > 1) format(tail(conv$relative_change, 1), digits = 4) else NA, round(fr$wall_seconds, 1), round(fr$cpu_user_seconds, 1),
                                 sprintf("%d (up to %s iterations each)", nrow(fit_chunks), if (is.null(fr$fit_chunk)) "all" else fr$fit_chunk), fr$model_file_bytes, fr$run_id, fr$stm_version, fr$glmnet_version, R.version.string))
  list(K = K, searchk_row = searchk_row, conv = conv, runtime = runtime, runtime_summary = runtime_summary, stage_mem = stage_mem, trace = trace, coherence = coherence, by_type = by_type, fit_chunks = fit_chunks,
       concentration = concentration, dominant_share_quantiles = dominant_share_quantiles, top_words = top_words, kappa_labels = kappa_labels, kappa_sparsity = kappa_sparsity,
       gamma_sparsity = gamma_sparsity, gamma_overall = gamma_overall, heldout_by_type = heldout_by_type, topic_correlation = topic_correlation, representative = representative,
       topic_summary = topic_summary, document_table = document_table, topic_word_table = topic_word_table, model_spec = model_spec, beta_marg = beta_marg, theta = theta,
       sigma_cor = sigma_cor, top10 = top10, dominant = dominant, expected_proportion = expected_proportion)
}
per_k <- lapply(fits, summarise_fit)
bind_k <- function(name) bind_rows(lapply(per_k, function(p) p[[name]]))
searchk_results   <- bind_k("searchk_row")
convergence_trace <- bind_k("conv")
runtime_by_k      <- bind_k("runtime")
runtime_summary   <- bind_k("runtime_summary")
memory_by_stage   <- bind_k("stage_mem")
coherence_by_topic <- bind_k("coherence")
coherence_summary <- coherence_by_topic |> group_by(K) |>
  summarise(topics = n(), semcoh_mean = mean(semantic_coherence_stm), semcoh_median = median(semantic_coherence_stm), semcoh_sd = sd(semantic_coherence_stm), semcoh_min = min(semantic_coherence_stm), semcoh_max = max(semantic_coherence_stm),
            npmi_mean = mean(coherence_npmi_top10), npmi_median = median(coherence_npmi_top10), npmi_min = min(coherence_npmi_top10), npmi_max = max(coherence_npmi_top10), .groups = "drop") |>
  left_join(searchk_results |> select(K, exclusivity_status, semcoh_status), by = "K")
prevalence_by_type <- bind_k("by_type")
concentration      <- bind_k("concentration")
dominant_share_quantiles <- bind_k("dominant_share_quantiles")
top_words          <- bind_k("top_words")
kappa_labels       <- bind_k("kappa_labels")
kappa_sparsity     <- bind_k("kappa_sparsity")
kappa_sparsity_summary <- kappa_sparsity |> group_by(K, parameter_type) |> summarise(parameters = n(), nonzero_entries_total = sum(nonzero_entries), nonzero_entries_mean = mean(nonzero_entries), nonzero_entries_max = max(nonzero_entries), vocabulary = first(vocabulary), .groups = "drop")
gamma_sparsity     <- bind_k("gamma_sparsity")
gamma_overall      <- bind_k("gamma_overall")
heldout_by_type    <- bind_k("heldout_by_type")
topic_correlation  <- bind_k("topic_correlation")
representative_documents <- bind_k("representative")
topic_summary      <- bind_k("topic_summary")
model_specification <- bind_k("model_spec")
gamma_path <- bind_rows(lapply(fits, function(f) f$gamma_path))
fit_chunks <- bind_k("fit_chunks")
exclusivity_status <- searchk_results |> transmute(K, exclusivity = exclus, status = exclusivity_status,
                                                    reason = "stm::exclusivity() is defined for models with a single topic-word distribution; this model has one per content-covariate level (comment, submission); no substitute statistic is computed")
mark_stage("summaries")
log_line("summaries built for K =", paste(K_done, collapse = ", "))

# ---- Topic correspondence across K values (inexpensive) ----
# INPUT : per_k (marginal betas, theta).
# DOES  : for every pair of completed K values, the cosine similarity between the marginal
#         topic-word distributions (all pairs of topics), each topic's best match in the other
#         model in both directions, a one-to-one assignment (Hungarian, maximising total cosine),
#         and the correlation of document-topic proportions across the two models.
# OUTPUT: correspondence_cosine, correspondence_best_match, correspondence_theta.
pairs <- if (length(K_done) >= 2) combn(K_done, 2, simplify = FALSE) else list()
corr_cos_rows <- list(); corr_best_rows <- list(); corr_theta_rows <- list(); corr_summary_rows <- list()
for (pr in pairs) {
  a <- per_k[[k_tag(pr[1])]]; b <- per_k[[k_tag(pr[2])]]
  cs <- cosine_rows(a$beta_marg, b$beta_marg)
  th <- cor(a$theta, b$theta)
  pair_label <- sprintf("K%d vs K%d", pr[1], pr[2])
  corr_cos_rows[[pair_label]] <- as_tibble(expand.grid(topic_a = seq_len(pr[1]), topic_b = seq_len(pr[2]))) |> mutate(pair = pair_label, K_a = pr[1], K_b = pr[2], beta_cosine = as.vector(cs), theta_correlation = as.vector(th)) |> select(pair, K_a, K_b, everything())
  hung <- clue::solve_LSAP(1 - cs)   # rows (smaller K) assigned to distinct columns
  corr_best_rows[[pair_label]] <- bind_rows(
    tibble(pair = pair_label, direction = sprintf("each K%d topic: best K%d match", pr[1], pr[2]), topic = seq_len(pr[1]), best_match = apply(cs, 1, which.max), best_cosine = apply(cs, 1, max),
           second_best_cosine = apply(cs, 1, function(x) sort(x, decreasing = TRUE)[2]), hungarian_match = as.integer(hung), hungarian_cosine = cs[cbind(seq_len(pr[1]), as.integer(hung))],
           theta_correlation_with_best = th[cbind(seq_len(pr[1]), apply(cs, 1, which.max))]),
    tibble(pair = pair_label, direction = sprintf("each K%d topic: best K%d match", pr[2], pr[1]), topic = seq_len(pr[2]), best_match = apply(cs, 2, which.max), best_cosine = apply(cs, 2, max),
           second_best_cosine = apply(cs, 2, function(x) sort(x, decreasing = TRUE)[2]), hungarian_match = NA_integer_, hungarian_cosine = NA_real_,
           theta_correlation_with_best = th[cbind(apply(cs, 2, which.max), seq_len(pr[2]))]))
  corr_summary_rows[[pair_label]] <- tibble(pair = pair_label, K_a = pr[1], K_b = pr[2], mean_best_cosine_a_to_b = mean(apply(cs, 1, max)), mean_best_cosine_b_to_a = mean(apply(cs, 2, max)),
                                            mean_hungarian_cosine = mean(cs[cbind(seq_len(pr[1]), as.integer(hung))]), min_hungarian_cosine = min(cs[cbind(seq_len(pr[1]), as.integer(hung))]),
                                            topics_a_with_best_cosine_at_least_0_8 = sum(apply(cs, 1, max) >= 0.8), topics_b_with_best_cosine_at_least_0_8 = sum(apply(cs, 2, max) >= 0.8),
                                            topics_b_not_best_match_of_any_a = sum(!(seq_len(pr[2]) %in% apply(cs, 1, which.max))), topics_b_claimed_by_two_or_more_a = sum(table(apply(cs, 1, which.max)) >= 2),
                                            max_abs_theta_correlation = max(abs(th)), mean_max_theta_correlation_a = mean(apply(th, 1, max)))
}
correspondence_cosine  <- if (length(corr_cos_rows)) bind_rows(corr_cos_rows) else tibble()
correspondence_best    <- if (length(corr_best_rows)) bind_rows(corr_best_rows) else tibble()
correspondence_summary <- if (length(corr_summary_rows)) bind_rows(corr_summary_rows) else tibble()
within_k_nearest <- topic_summary |> transmute(K, topic, nearest_topic_by_beta_cosine, nearest_topic_cosine, nearest_topic_by_theta_correlation, nearest_topic_correlation)
mark_stage("correspondence")

# ---- Linkage verification after fitting ----
# INPUT : per_k; docs_stm; docs; comments; submissions; heldout; the inventory tables.
# DOES  : confirm that every theta row is the STM document with the same doc_key, that every
#         document matches exactly one source record with the same author, thread and parent
#         identifiers, that the covariate factors carry the expected level counts, that the stm
#         documents equal a second tokenisation of the stored text (every sample_stride-th
#         document, allowing for the held-out tokens), and count the author, thread, parent and
#         submission joins against the inventory's documented values where they apply.
# OUTPUT: linkage_checks; recount_matches.
recompute_tokens <- function(text) {
  x <- stri_replace_all_regex(stri_replace_all_fixed(stri_trans_tolower(text), "’", "'"), url_regex, " ", opts_regex = url_opts)
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
fitted_subset_ok <- vapply(sample_idx, function(i) {
  d <- documents[[i]]; h <- heldout$documents[[i]]
  full <- setNames(d[2, ], vocab[d[1, ]]); part <- setNames(h[2, ], vocab_fit[h[1, ]])
  all(names(part) %in% names(full)) && all(part <= full[names(part)]) && (ho_flag[i] || (length(part) == length(full) && all(part == full[names(part)])))
}, logical(1))
comment_match <- match(docs_stm$doc_key[docs_stm$doc_type == "comment"], paste0("t1_", comments$id))
sub_match     <- match(docs_stm$doc_key[docs_stm$doc_type == "submission"], paste0("t3_", submissions$id))
parent_is_comment <- substr(stm_c$parent_id, 1, 3) == "t1_"
parent_in_sample_comment <- parent_is_comment & stm_c$parent_id %in% paste0("t1_", comments$id)
parent_in_stm <- parent_is_comment & stm_c$parent_id %in% docs_stm$doc_key
parent_is_submission_in_sample <- !parent_is_comment & stm_c$parent_id %in% paste0("t3_", submissions$id)
inv_parent <- function(record, modelable) {
  if (is.null(inventory_parent)) return(NA_real_)
  sel <- inventory_parent$scope == "Tier 2 modelable comments" & inventory_parent$parent_record == record &
    (is.na(inventory_parent$parent_modelable) == all(is.na(modelable))) & (all(is.na(modelable)) | inventory_parent$parent_modelable %in% modelable)
  v <- inventory_parent$n[sel]
  if (length(v)) sum(v) else NA_real_
}
lk_row <- function(item, value, documented = NA, source = "", applies = TRUE) {
  documented <- if (length(documented) == 1L) documented else NA
  tibble(item = item, value = as.character(value), documented_value = as.character(documented), source = source,
         applies_to_this_run = applies, equal_to_documented = if (!isTRUE(applies) || is.na(documented)) NA else as.character(value) == as.character(documented))
}
full <- !quick_run
theta_rows_ok <- all(vapply(per_k, function(p) nrow(p$theta) == nrow(docs_stm), logical(1)))
linkage_checks <- bind_rows(
  lk_row("theta rows equal the STM document count in every fitted model", theta_rows_ok),
  lk_row("STM documents (names(documents) = doc_key)", length(documents)),
  lk_row("theta row order equals the STM document order equals docs_stm$doc_key", identical(names(documents), docs_stm$doc_key) && identical(names(heldout$documents), docs_stm$doc_key)),
  lk_row("STM comments matched to exactly one source comment by doc_key", sum(!is.na(comment_match)), nrow(stm_c)),
  lk_row("STM submissions matched to exactly one source submission by doc_key", sum(!is.na(sub_match)), nrow(stm_s)),
  lk_row("distinct doc_key among STM documents", n_distinct(docs_stm$doc_key), nrow(docs_stm)),
  lk_row("STM comments whose author equals the source author", sum(stm_c$author == comments$author[comment_match]), nrow(stm_c)),
  lk_row("STM submissions whose author equals the source author", sum(stm_s$author == submissions$author[sub_match]), nrow(stm_s)),
  lk_row("STM comments whose link_id and parent_id equal the source values", sum(stm_c$link_id == comments$link_id[comment_match] & stm_c$parent_id == comments$parent_id[comment_match]), nrow(stm_c)),
  lk_row("STM comments whose created_utc equals the source value", sum(stm_c$created_utc == comments$created_utc[comment_match]), nrow(stm_c)),
  lk_row("STM submissions whose created_utc equals the source value", sum(stm_s$created_utc == submissions$created_utc[sub_match]), nrow(stm_s)),
  lk_row("STM documents re-tokenised from the stored text by a second code path (every 5,000th) that equal the stm document", sum(recount_matches), length(sample_idx)),
  lk_row("sampled fitted documents that are the stm document minus held-out tokens (or unchanged when not held out)", sum(fitted_subset_ok), length(sample_idx)),
  lk_row("covariate rows equal STM documents; covariate doc_key order equals the STM order", nrow(meta_df) == length(documents) && identical(cov_build$doc_key, names(documents))),
  lk_row("author levels (comment and submission authors together)", nlevels(meta_df$author)),
  lk_row("distinct authors, STM comments", n_distinct(stm_c$author), 133602, "metadata_inventory README section 4 (modelable comments)", full),
  lk_row("distinct authors, STM submissions", n_distinct(stm_s$author), 2308, "metadata_inventory README section 4 (modelable submissions)", full),
  lk_row("STM comments with '[deleted]' author", sum(stm_c$author == "[deleted]"), 0, "metadata_inventory README section 5", full),
  lk_row("STM submissions with '[deleted]' author", sum(stm_s$author == "[deleted]"), 337, "metadata_inventory README section 5", full),
  lk_row("STM comments whose thread submission is in the sample (link_id join)", sum(stm_c$submission_in_sample), 839545, "metadata_inventory README section 4", full),
  lk_row("STM comments with inherited num_comments and upvote_ratio (linked submission in the sample)", sum(stm_c$num_comments_missing == 0L & stm_c$upvote_ratio_missing == 0L), 839545, "metadata_inventory README section 4 (the same join)", full),
  lk_row("STM comments with the missingness indicator = 1 (linked submission absent from the sample)", sum(stm_c$num_comments_missing == 1L), 846666 - 839545, "846,666 modelable comments minus 839,545 linked (inventory)", full),
  lk_row("STM comments whose thread submission is itself an STM document", sum(stm_c$link_id %in% stm_s$doc_key)),
  lk_row("distinct threads (link_id) among STM comments", n_distinct(stm_c$link_id), 11021, "metadata_inventory README section 5", full),
  lk_row("thread_id levels (comment threads plus submission fullnames)", nlevels(meta_df$thread_id), if (full) 11021 + sum(!(paste0("t3_", stm_s$doc_id) %in% stm_c$link_id)) else NA, "11,021 comment threads plus the STM submissions that no STM comment references", full),
  lk_row("STM submissions whose fullname equals the link_id of at least one STM comment (thread level shared)", sum(stm_s$doc_key %in% stm_c$link_id)),
  lk_row("distinct parent_id among STM comments", n_distinct(stm_c$parent_id), 300429, "metadata_inventory README section 4", full),
  lk_row("parent_id levels (comment parents plus NO_PARENT_SUBMISSION)", nlevels(meta_df$parent_id), 300430, "300,429 + 1", full),
  lk_row("STM comments whose parent is a comment in the sample (parent_id join)", sum(parent_in_sample_comment), inv_parent("parent comment in the sample", c("parent comment modelable", "parent comment not modelable")), "metadata_inventory/tables/parent_linkage.csv (modelable in sample, both statuses summed)", full),
  lk_row("STM comments whose parent comment is itself an STM document", sum(parent_in_stm), inv_parent("parent comment in the sample", "parent comment modelable"), "metadata_inventory/tables/parent_linkage.csv", full),
  lk_row("STM comments whose parent is the submission, in the sample", sum(parent_is_submission_in_sample), inv_parent("parent is the submission, in the sample", NA), "metadata_inventory/tables/parent_linkage.csv", full),
  lk_row("STM comments whose parent comment is not in the sample (kept, own level)", sum(parent_is_comment & !parent_in_sample_comment), inv_parent("parent comment not in the sample", NA), "metadata_inventory/tables/parent_linkage.csv", full),
  lk_row("STM comments whose parent is the submission, not in the sample (kept, own level)", sum(!parent_is_comment & !parent_is_submission_in_sample), inv_parent("parent is the submission, not in the sample", NA), "metadata_inventory/tables/parent_linkage.csv", full),
  lk_row("STM documents with a non-missing score", sum(!is.na(docs_stm$score)), nrow(docs_stm)),
  lk_row("STM submissions with non-missing num_comments and upvote_ratio", sum(!is.na(stm_s$num_comments) & !is.na(stm_s$upvote_ratio)), nrow(stm_s)),
  lk_row("STM submissions with a non-empty url", sum(usable(stm_s$url)), 10382, "metadata_inventory README section 4", full),
  lk_row("held-out documents (partial) that are STM documents", sum(heldout$missing$index %in% seq_along(documents)), length(heldout$missing$index)))
mark_stage("linkage")
log_line("linkage:", sum(recount_matches), "of", length(sample_idx), "sampled documents re-tokenised identically;", sum(linkage_checks$equal_to_documented, na.rm = TRUE), "of", sum(!is.na(linkage_checks$equal_to_documented)), "documented values reproduced")

# ---- Figures ----
# INPUT : the tables above.
# DOES  : PNG figures with a title, axis titles, tick labels and a legend where more than one
#         series or a colour scale is shown; no annotations. K = 10, 20, 30 use the validated
#         categorical slots 1-3 (blue, orange, aqua); document types blue / orange; heatmaps a
#         single blue ramp.
# OUTPUT: figures/*.png.
ink_series <- "#2a78d6"; ink_series_2 <- "#eb6834"; ink_series_3 <- "#1baf7a"; ink_red <- "#e34948"; ink_text <- "#0b0b0b"; ink_sub <- "#52514e"; ink_muted <- "#898781"
grid_col <- "#e1e0d9"; axis_col <- "#c3c2b7"; mid_grey <- "#f0efec"
stage_colours <- c(ink_series, ink_series_2, ink_series_3, "#eda100", "#e87ba4", "#008300", "#4a3aa7")
theme_fig <- theme_minimal(base_size = 12) +
  theme(panel.grid.minor = element_blank(), panel.grid.major = element_line(colour = grid_col, linewidth = 0.3),
        axis.line.x = element_line(colour = axis_col, linewidth = 0.4), axis.text = element_text(colour = ink_muted), axis.title = element_text(colour = ink_sub),
        plot.title = element_text(colour = ink_text, face = "bold"), strip.text = element_text(colour = ink_sub, face = "bold", hjust = 0),
        legend.title = element_text(colour = ink_sub), legend.text = element_text(colour = ink_muted), plot.background = element_rect(fill = "white", colour = NA))
k_levels  <- sprintf("K = %d", K_done)
k_colours <- setNames(c(ink_series, ink_series_2, ink_series_3)[seq_along(K_done)], k_levels)
k_factor  <- function(K) factor(sprintf("K = %d", K), levels = k_levels)
type_colours <- c(comment = ink_series, submission = ink_series_2)
save_fig <- function(name, plot, width, height, dpi = 150) ggsave(file.path(fig_dir, name), plot, width = width, height = height, dpi = dpi, bg = "white", limitsize = FALSE)
title_suffix <- sprintf("STM searchK evidence, K = %s%s - r/politics, July 2026", paste(K_done, collapse = ", "), if (quick_run) " (quick run)" else "")
fig_title <- function(main) paste0(main, "\n", title_suffix)
metric_panel <- function(df, y, ylab, title) ggplot(df, aes(x = k_factor(K), y = .data[[y]], group = 1)) + geom_line(colour = ink_series, linewidth = 0.6) + geom_point(colour = ink_series, size = 2.6) +
  labs(title = title, x = "Number of topics K", y = ylab) + theme_fig
fig1a <- metric_panel(searchk_results, "heldout", "Held-out log-likelihood per token (document average)", fig_title("Held-out likelihood by K"))
fig1b <- metric_panel(searchk_results, "residual", "Residual dispersion (1 under the model)", "Residual dispersion by K")
bound_long <- searchk_results |> select(K, `Approximate bound` = bound, `Corrected lower bound (bound + lfactorial(K))` = lbound) |> pivot_longer(-K, names_to = "quantity", values_to = "value")
fig1c <- ggplot(bound_long, aes(x = k_factor(K), y = value, colour = quantity, group = quantity)) + geom_line(linewidth = 0.6) + geom_point(size = 2.6) +
  scale_colour_manual(values = c(ink_series, ink_series_2), name = "Quantity") + scale_y_continuous(labels = label_comma()) +
  labs(title = "Variational bound and corrected lower bound by K", x = "Number of topics K", y = "Bound") + theme_fig + theme(legend.position = "bottom", legend.direction = "vertical")
fig1d <- ggplot(coherence_by_topic, aes(x = k_factor(K), y = semantic_coherence_stm)) + geom_jitter(width = 0.12, height = 0, colour = ink_series, alpha = 0.55, size = 1.8) +
  stat_summary(fun = mean, geom = "point", colour = ink_series_2, size = 3.4, shape = 18) +
  labs(title = "Semantic coherence per topic by K (stm, top 10 words; diamond = mean)", x = "Number of topics K", y = "Semantic coherence") + theme_fig
save_fig("fig1_searchk_metrics_by_k.png", (fig1a | fig1b) / (fig1c | fig1d), 12, 9)

fig2a <- ggplot(coherence_by_topic, aes(x = k_factor(K), y = semantic_coherence_stm, fill = k_factor(K))) + geom_boxplot(width = 0.55, outlier.size = 1, colour = ink_sub) +
  scale_fill_manual(values = k_colours, guide = "none") + labs(title = fig_title("Semantic coherence per topic (stm, content-covariate-aware)"), x = "Number of topics K", y = "Semantic coherence") + theme_fig
fig2b <- ggplot(coherence_by_topic, aes(x = k_factor(K), y = coherence_npmi_top10, fill = k_factor(K))) + geom_boxplot(width = 0.55, outlier.size = 1, colour = ink_sub) +
  scale_fill_manual(values = k_colours, guide = "none") + labs(title = "NPMI coherence per topic over the same ten marginal words (fitted documents)", x = "Number of topics K", y = "NPMI") + theme_fig
save_fig("fig2_semantic_coherence_by_topic.png", fig2a / fig2b, 8, 8)

fig3a <- ggplot(convergence_trace, aes(x = iteration, y = bound_per_token, colour = k_factor(K))) + geom_line(linewidth = 0.6) + geom_point(size = 1.4) +
  scale_colour_manual(values = k_colours, name = "Model") + labs(title = fig_title("Approximate bound per token by EM iteration"), x = "EM iteration", y = "Approximate variational bound per token") + theme_fig + theme(legend.position = "bottom")
fig3b <- ggplot(convergence_trace |> filter(!is.na(relative_change), relative_change > 0), aes(x = iteration, y = relative_change, colour = k_factor(K))) + geom_line(linewidth = 0.6) + geom_point(size = 1.4) +
  scale_colour_manual(values = k_colours, name = "Model") + scale_y_log10(labels = label_scientific()) + geom_hline(yintercept = emtol, colour = axis_col, linewidth = 0.4) +
  labs(title = "Relative change of the bound by EM iteration (log axis; positive changes)", x = "EM iteration", y = "Relative change") + theme_fig + theme(legend.position = "bottom")
save_fig("fig3_em_convergence_by_k.png", fig3a / fig3b, 9, 9)

stage_order <- c("spectral initialisation (wall minus E- and M-step seconds; approximate)", "E-steps (sum)", "M-steps (sum; L1 prevalence glmnet + content-covariate distributed Poisson regressions)",
                 "held-out evaluation (eval.heldout)", "semantic coherence (semanticCoherence)", "labels (sageLabels, labelTopics)", "residual dispersion (checkResiduals)")
stage_short <- c("initialisation", "E-steps", "M-steps", "held-out evaluation", "semantic coherence", "labels", "residual dispersion")
rt_fig <- runtime_by_k |> filter(stage %in% stage_order) |> mutate(stage = factor(stage_short[match(stage, stage_order)], levels = rev(stage_short)), minutes = seconds / 60)
fig4a <- ggplot(rt_fig, aes(x = k_factor(K), y = minutes, fill = stage)) + geom_col(width = 0.6, colour = "white", linewidth = 0.3) +
  scale_fill_manual(values = setNames(rev(stage_colours), rev(stage_short)), name = "Stage") + scale_y_continuous(expand = expansion(mult = c(0, 0.05))) +
  labs(title = fig_title("Worker wall time by stage and K"), x = "Number of topics K", y = "Minutes") + theme_fig + theme(legend.position = "bottom")
traces_fig <- bind_rows(lapply(per_k, function(p) if (is.null(p$trace)) NULL else p$trace |> mutate(K = p$K, time = as.POSIXct(time_utc, format = "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")) |>
                                 mutate(minutes = as.numeric(difftime(time, min(time), units = "mins")))))
fig4b <- if (nrow(traces_fig)) ggplot(traces_fig, aes(x = minutes, y = wset_mb / 1024, colour = k_factor(K))) + geom_line(linewidth = 0.5) +
  scale_colour_manual(values = k_colours, name = "Model") + scale_y_continuous(limits = c(0, NA)) +
  labs(title = "Working set of the fitting process through the worker run", x = "Minutes since the worker's sampler started", y = "Working set (GB)") + theme_fig + theme(legend.position = "bottom") else NULL
if (is.null(fig4b)) save_fig("fig4_runtime_memory_by_k.png", fig4a, 9, 5) else save_fig("fig4_runtime_memory_by_k.png", fig4a / fig4b, 9, 9)

prev_fig <- prevalence_by_type |> left_join(topic_summary |> select(K, topic, prevalence_rank), by = c("K", "topic")) |>
  mutate(topic_label = sprintf("Topic %d", topic), K_label = k_factor(K)) |> group_by(K) |> mutate(topic_label = factor(topic_label, levels = sprintf("Topic %d", topic[doc_type == "comment"][order(-prevalence_rank[doc_type == "comment"])]))) |> ungroup()
fig5 <- ggplot(prev_fig, aes(x = mean_theta, y = topic_label, fill = doc_type)) + geom_col(position = position_dodge(width = 0.8), width = 0.7) +
  facet_wrap(~ K_label, scales = "free", ncol = length(K_done)) + scale_fill_manual(values = type_colours, name = "Document type") +
  scale_x_continuous(labels = label_percent(accuracy = 1), expand = expansion(mult = c(0, 0.05))) +
  labs(title = fig_title("Expected topic proportion (mean theta) by document type, topics ordered by overall prevalence"), x = "Mean theta", y = NULL) + theme_fig + theme(legend.position = "bottom", panel.grid.major.y = element_blank())
save_fig("fig5_topic_prevalence_by_k.png", fig5, 4 + 3.2 * length(K_done), 9)

conc_fig <- bind_rows(lapply(per_k, function(p) tibble(K = p$K, doc_type = type_groups, dominant_share = p$theta[cbind(seq_len(N_fit), p$dominant)])))
fig6 <- ggplot(conc_fig, aes(x = k_factor(K), y = dominant_share, fill = doc_type)) + geom_boxplot(width = 0.6, outlier.size = 0.3, outlier.alpha = 0.15, colour = ink_sub, position = position_dodge(width = 0.75)) +
  scale_fill_manual(values = type_colours, name = "Document type") + scale_y_continuous(labels = label_percent(accuracy = 1), limits = c(0, 1)) +
  labs(title = fig_title("Share of the dominant topic within documents by K and document type"), x = "Number of topics K", y = "Share of the dominant topic") + theme_fig + theme(legend.position = "bottom")
save_fig("fig6_document_concentration_by_k.png", fig6, 9, 6)

if (nrow(correspondence_cosine)) {
  heat <- function(pair_label) {
    d <- correspondence_cosine |> filter(pair == pair_label)
    ggplot(d, aes(x = factor(topic_b), y = factor(topic_a, levels = rev(sort(unique(topic_a)))), fill = beta_cosine)) + geom_tile(colour = "white", linewidth = 0.3) +
      scale_fill_gradient(low = "#cde2fb", high = "#0d366b", limits = c(0, 1), name = "Cosine of marginal topic-word distributions") +
      labs(title = sprintf("Topic correspondence: %s", pair_label), x = sprintf("Topic (K = %d)", d$K_b[1]), y = sprintf("Topic (K = %d)", d$K_a[1])) +
      theme_fig + theme(panel.grid.major = element_blank(), legend.position = "bottom", axis.text = element_text(size = 7))
  }
  heats <- lapply(unique(correspondence_cosine$pair), heat)
  fig7 <- wrap_plots(heats, ncol = length(heats)) + plot_annotation(title = fig_title("Topic correspondence across K by cosine similarity"), theme = theme(plot.title = element_text(colour = ink_text, face = "bold")))
  save_fig("fig7_topic_correspondence_cosine.png", fig7, 5 + 5.5 * length(heats), 8)
}

for (p in per_k) {
  tw <- top_words |> filter(K == p$K, word_set == "marginal (aspect-weighted beta)", ranking == "prob", rank <= 10) |> arrange(topic, rank) |>
    mutate(topic_label = factor(sprintf("Topic %d", topic), levels = sprintf("Topic %d", seq_len(p$K))), key = paste0(term, "  #", topic)) |> mutate(key = factor(key, levels = rev(unique(key))))
  fig_cols <- min(5L, p$K)
  fig8 <- ggplot(tw, aes(x = marginal_probability, y = key)) + geom_col(fill = ink_series, width = 0.7) + facet_wrap(~ topic_label, scales = "free", ncol = fig_cols) +
    scale_y_discrete(labels = function(x) sub("  #.*$", "", x)) + scale_x_continuous(labels = label_number(accuracy = 0.001), n.breaks = 3, expand = expansion(mult = c(0, 0.05))) +
    labs(title = fig_title(sprintf("Highest-probability words per topic (marginal beta), K = %d", p$K)), x = "Marginal probability of the word in the topic", y = NULL) +
    theme_fig + theme(panel.grid.major.y = element_blank(), axis.text.y = element_text(size = 8), axis.text.x = element_text(size = 7))
  save_fig(sprintf("fig8_top_words_%s.png", k_tag(p$K)), fig8, 2.8 * fig_cols + 1, 2.3 * ceiling(p$K / fig_cols) + 1, dpi = 130)
}

gs_fig <- gamma_sparsity |> filter(term != "(Intercept, glmnet)") |> mutate(term = factor(term, levels = rev(design_term_labels[-1])))
fig9 <- ggplot(gs_fig, aes(x = share_nonzero, y = term, fill = k_factor(K))) + geom_col(position = position_dodge(width = 0.8), width = 0.7) +
  scale_fill_manual(values = k_colours, name = "Model") + scale_x_continuous(labels = label_percent(accuracy = 1), limits = c(0, 1), expand = expansion(mult = c(0, 0.02))) +
  labs(title = fig_title("Share of design columns with a nonzero L1 prevalence coefficient, by covariate term"), x = "Share of the term's design columns selected by the L1 fit", y = NULL) + theme_fig + theme(legend.position = "bottom", panel.grid.major.y = element_blank())
save_fig("fig9_gamma_sparsity_by_k.png", fig9, 9, 6)

fig10 <- ggplot(heldout_by_type |> filter(doc_type != "all"), aes(x = k_factor(K), y = mean_per_document_log_likelihood, colour = doc_type, group = doc_type)) + geom_line(linewidth = 0.6) + geom_point(size = 2.6) +
  scale_colour_manual(values = type_colours, name = "Document type") + labs(title = fig_title("Held-out log-likelihood per token by document type"), x = "Number of topics K", y = "Held-out log-likelihood per token (document average)") + theme_fig + theme(legend.position = "bottom")
save_fig("fig10_heldout_by_doc_type.png", fig10, 7, 5)
mark_stage("figures")

# ---- Validation ----
# INPUT : every object above.
# DOES  : checks that must hold before the tables are written (blocking) plus the completion
#         check (recorded, not blocking, so partial evidence is still written); the script stops
#         after writing validation_checks.csv if a blocking check fails.
# OUTPUT: validation_checks.
checks <- list()
add_check <- function(check, detail, pass, blocking = TRUE) checks[[length(checks) + 1L]] <<- tibble(check = check, detail = detail, pass = isTRUE(pass), blocking = blocking)
add_check("source files read: 7 comment files and 1 submission file, rows equal the delivered counts", sprintf("%d + %d rows", n_comments_all, n_submissions_all), n_comments_all == 891397 && n_submissions_all == 10747)
add_check("every delivered artifact read matches the SHA256 in its manifest", sprintf("%d of %d existing files (%d sidecars parse as strict YAML)", sum(input_integrity$hash_equals_manifest), sum(input_integrity$exists), sum(input_integrity$manifest_parses_as_strict_yaml)), all(input_integrity$hash_equals_manifest[input_integrity$exists]))
add_check("Tier 2 representation reproduced: every reconciliation item equal", sprintf("%d of %d equal", sum(tier2_reconciliation$equal), nrow(tier2_reconciliation)), all(tier2_reconciliation$equal))
add_check("STM documents: every entering document survives the conversion and every token is kept",
          sprintf("%d in, %d out; tokens %d in, %d out", n_stm_in, length(documents), tokens_stm_in, sum(doc_tokens_stm)), length(documents) == n_stm_in && sum(doc_tokens_stm) == tokens_stm_in)
add_check("STM documents: per-document token sums equal the Tier 2 n_tokens_modeled", sprintf("%d of %d", sum(doc_tokens_stm == docs_stm$n_tokens_modeled), length(documents)), all(doc_tokens_stm == docs_stm$n_tokens_modeled))
add_check("STM vocabulary: V equals the Tier 2 vocabulary with no unseen term (full run)", sprintf("V = %d; Tier 2 V = %d; unseen dropped = %d", length(vocab), n_vocabulary, terms_unseen), quick_run || (length(vocab) == n_vocabulary && terms_unseen == 0))
add_check("covariates: no missing value; doc_type has both levels; indicator rows carry the placeholder; indicator count = comments minus linked comments; design columns sum by term",
          sprintf("NA %d; missing indicator %d; design %d x %d", sum(is.na(meta_df)), sum(meta_df$num_comments_missing), nrow(X_design), ncol(X_design)),
          !anyNA(meta_df) && nlevels(droplevels(meta_df$doc_type)) == 2 && all(meta_df$num_comments[meta_df$num_comments_missing == 1L] == cov_build$placeholders[["num_comments"]]) &&
            all(meta_df$upvote_ratio[meta_df$upvote_ratio_missing == 1L] == cov_build$placeholders[["upvote_ratio"]]) &&
            sum(meta_df$num_comments_missing) == sum(docs_stm$doc_type == "comment" & !docs_stm$linked_submission_in_sample) && all(meta_df$num_comments_missing[meta_df$doc_type == "submission"] == 0L) &&
            sum(design_columns_by_term$design_columns) == ncol(X_design) && nrow(X_design) == length(documents))
add_check("covariates: thread and parent identities as specified (submissions share their comments' thread level; NO_PARENT_SUBMISSION for every submission; absent parents kept)",
          sprintf("thread levels %d; parent levels %d", nlevels(meta_df$thread_id), nlevels(meta_df$parent_id)),
          all(as.character(meta_df$thread_id)[meta_df$doc_type == "submission"] == docs_stm$doc_key[docs_stm$doc_type == "submission"]) &&
            all(as.character(meta_df$thread_id)[meta_df$doc_type == "comment"] == docs_stm$link_id[docs_stm$doc_type == "comment"]) &&
            all(as.character(meta_df$parent_id)[meta_df$doc_type == "submission"] == "NO_PARENT_SUBMISSION") &&
            all(as.character(meta_df$parent_id)[meta_df$doc_type == "comment"] == docs_stm$parent_id[docs_stm$doc_type == "comment"]) &&
            nlevels(meta_df$parent_id) == n_distinct(docs_stm$parent_id[docs_stm$doc_type == "comment"]) + 1L)
add_check("held-out construction: every document survives; tokens conserved (fitted + held-out + held-out tokens of dropped terms); vocabulary after within before; held-out index within the sampled index; documents named by doc_key",
          sprintf("%d docs; tokens %d + %d + %d = %d; V %d -> %d", length(heldout$documents), sum(heldout_doc_tokens), sum(missing_tokens_by_doc), dropped_term_tokens, sum(doc_tokens_stm), length(vocab), length(heldout$vocab)),
          length(heldout$documents) == length(documents) && sum(heldout_doc_tokens) + sum(missing_tokens_by_doc) + dropped_term_tokens == sum(doc_tokens_stm) && all(heldout$vocab %in% vocab) &&
            all(heldout$missing$index %in% sampled_index) && identical(names(heldout$documents), docs_stm$doc_key) && length(heldout$missing$docs) == length(heldout$missing$index))
add_check("self-check: the searchK reimplementation equals a literal searchK() call on the subset (every statistic identical)", sprintf("%s; searchK returned %s", selfcheck_equal, paste(names(sc_literal_tbl), collapse = ", ")), selfcheck_equal)
add_check("all three K values completed (fit, held-out likelihood, residual dispersion, coherence, labels)", sprintf("completed K = %s; failed K = %s", paste(K_done, collapse = ", "), if (length(K_failed)) paste(K_failed, collapse = ", ") else "none"), length(K_failed) == 0, blocking = FALSE)
add_check("L1 prevalence step: a path record exists for every M-step of every K, and the information-criterion minimum was interior to the computed path in every call (so the pmax memory cap never truncated the path before the selected point)",
          sprintf("%d calls recorded; interior in all: %s; pmax %s", nrow(gamma_path), if (nrow(gamma_path)) !any(gamma_path$chosen_is_last_computed) else NA, paste(unique(searchk_results$gamma_pmax), collapse = ", ")),
          nrow(gamma_path) > 0 && all(vapply(per_k, function(p) isTRUE(p$searchk_row$gamma_l1_calls == fits[[k_tag(p$K)]]$model$convergence$its), logical(1))) && !any(gamma_path$chosen_is_last_computed), blocking = FALSE)
for (p in per_k) {
  f <- fits[[k_tag(p$K)]]; m <- f$model
  add_check(sprintf("K = %d: model dimensions and the common input (N = held-out documents, V = held-out vocabulary, K); theta and every aspect beta row sum to 1; no NA; run key", p$K),
            sprintf("N %d V %d K %d; from run %s", m$settings$dim$N, m$settings$dim$V, m$settings$dim$K, f$fit_record$run_id),
            m$settings$dim$N == length(heldout$documents) && m$settings$dim$V == length(heldout$vocab) && m$settings$dim$K == p$K && max(abs(rowSums(p$theta) - 1)) < 1e-8 &&
              all(vapply(m$beta$logbeta, function(lb) max(abs(rowSums(exp(lb)) - 1)) < 1e-6, logical(1))) && !anyNA(p$theta) && identical(f$fit_record$run_key, run_key) &&
              length(m$beta$logbeta) == 2L && identical(m$settings$covariates$yvarlevels, c("comment", "submission")))
  add_check(sprintf("K = %d: specification as required (prevalence formula, L1 gamma, content ~ doc_type with L1 kappa, spectral, ceiling %d, emtol %s)", p$K, max_em_its, format(emtol)),
            sprintf("gamma %s; tau %s; init %s; its %d", m$settings$gamma$mode, m$settings$tau$mode, m$settings$init$mode, m$convergence$its),
            m$settings$gamma$mode == "L1" && m$settings$tau$mode == "L1" && m$settings$init$mode == "Spectral" &&
              max(p$fit_chunks$target_iterations) <= max_em_its && (m$convergence$converged || max(p$fit_chunks$target_iterations) == max_em_its) && m$settings$convergence$max.em.its == max(p$fit_chunks$target_iterations) &&
              m$settings$convergence$em.converge.thresh == emtol && identical(paste(deparse(m$settings$covariates$formula, width.cutoff = 500L), collapse = " "), prevalence_string) &&
              ncol(m$settings$covariates$X) == ncol(X_design) && p$gamma_overall$design_columns_equal_to_this_run && p$gamma_overall$design_values_equal_on_first_2000_rows)
  add_check(sprintf("K = %d: EM record (bound trace length = iterations; E-step seconds parsed for every iteration; iterations within the ceiling; lbound = bound + lfactorial(K))", p$K),
            sprintf("%d iterations; %d E-step timings; converged %s", m$convergence$its, sum(!is.na(p$conv$estep_seconds)), m$convergence$converged),
            length(m$convergence$bound) == m$convergence$its && all(!is.na(p$conv$estep_seconds)) && m$convergence$its <= max_em_its && abs(p$searchk_row$lbound - (p$searchk_row$bound + lfactorial(p$K))) < 1e-6 &&
              is.finite(p$searchk_row$heldout) && is.finite(p$searchk_row$residual))
  add_check(sprintf("K = %d: diagnostics (one finite coherence and NPMI per topic; exclusivity unavailable with the package error recorded; held-out per-document values for every held-out document)", p$K),
            sprintf("coherence mean %.3f; exclusivity: %s", mean(p$coherence$semantic_coherence_stm), f$semcoh$exclusivity_error),
            nrow(p$coherence) == p$K && all(is.finite(c(p$coherence$semantic_coherence_stm, p$coherence$coherence_npmi_top10))) && is.null(f$semcoh$exclusivity) && !is.na(f$semcoh$exclusivity_error) &&
              length(f$heldout_eval$doc_heldout) == length(heldout$missing$index))
  add_check(sprintf("K = %d: representative documents (five per topic under both rules; texts equal the stored text; Tier 2 rule respects the floor and the distinct text/author rule); prevalence sums; dominant counts", p$K),
            sprintf("%d rows; expected proportions sum %.6f", nrow(p$representative), sum(p$expected_proportion)),
            nrow(p$representative) == 2L * p$K * n_representative && all(p$representative$text == texts_stm[p$representative$stm_row]) &&
              min(p$representative$n_tokens_stm[grepl("Tier 2", p$representative$rule)]) >= representative_min_tokens &&
              all(tapply(p$representative$text[grepl("Tier 2", p$representative$rule)], p$representative$topic[grepl("Tier 2", p$representative$rule)], n_distinct) == n_representative) &&
              abs(sum(p$expected_proportion) - 1) < 1e-8 && sum(p$topic_summary$documents_dominant) == N_fit)
  add_check(sprintf("K = %d: memory sampler trace exists for the worker (or the sampler was disabled)", p$K), sprintf("%s rows", if (is.null(p$trace)) "0" else nrow(p$trace)), !memory_sampler_on || (!is.null(p$trace) && nrow(p$trace) >= 1))
}
if (nrow(correspondence_best)) add_check("topic correspondence: cosines within [0, 1]; Hungarian assignments distinct within each pair", sprintf("%d pairs", length(pairs)),
                                         all(correspondence_cosine$beta_cosine >= -1e-9 & correspondence_cosine$beta_cosine <= 1 + 1e-9) &&
                                           all(vapply(split(correspondence_best$hungarian_match[!is.na(correspondence_best$hungarian_match)], correspondence_best$pair[!is.na(correspondence_best$hungarian_match)]), function(x) !anyDuplicated(x), logical(1))))
add_check("linkage: theta row order equals the document order equals doc_key; every STM document matches exactly one source record with equal author and identifiers",
          paste(linkage_checks$value[3:11], collapse = " / "), identical(names(documents), docs_stm$doc_key) && all(!is.na(comment_match)) && all(!is.na(sub_match)) &&
            n_distinct(docs_stm$doc_key) == nrow(docs_stm) && all(stm_c$author == comments$author[comment_match]) && all(stm_s$author == submissions$author[sub_match]) &&
            all(stm_c$link_id == comments$link_id[comment_match] & stm_c$parent_id == comments$parent_id[comment_match]))
add_check("linkage: every sampled STM document equals a second tokenisation of its stored text, and the fitted document is that document minus its held-out tokens", sprintf("%d of %d; %d of %d", sum(recount_matches), length(sample_idx), sum(fitted_subset_ok), length(sample_idx)), all(recount_matches) && all(fitted_subset_ok))
add_check("linkage: documented inventory values reproduced where they apply", sprintf("%d of %d applicable items equal", sum(linkage_checks$equal_to_documented, na.rm = TRUE), sum(!is.na(linkage_checks$equal_to_documented))),
          all(linkage_checks$equal_to_documented[!is.na(linkage_checks$equal_to_documented)]))
validation_checks <- bind_rows(checks)
mark_stage("validation")
log_line("validation:", sum(validation_checks$pass), "of", nrow(validation_checks), "checks pass;", sum(!validation_checks$pass & validation_checks$blocking), "blocking failures")

# ---- Write tables ----
# INPUT : the measurement objects above.
# DOES  : write one CSV per table and record its dimensions for the sidecars; stop before
#         writing anything else if a blocking validation check failed.
# OUTPUT: CSV files under tables/.
table_dims <- list()
write_table <- function(tbl, name) {
  path <- file.path(tab_dir, name)
  write_csv(tbl, path)
  table_dims[[path]] <<- c(rows = nrow(tbl), cols = ncol(tbl))
  invisible(path)
}
write_table(validation_checks, "validation_checks.csv")
stopifnot(all(validation_checks$pass[validation_checks$blocking]))
parameters_table <- tibble(
  parameter = c("K_values", "prevalence_formula", "content_formula", "gamma_prior", "gamma_enet", "gamma_ic_k", "gamma_pmax (memory cap on glmnet's path; 0 = untouched package call)", "kappa_prior", "kappa_interactions", "kappa_fixedintercept", "sigma_prior", "init_type", "stm_seed", "max_em_its", "emtol", "ngroups",
                "heldout_function", "heldout_seed", "heldout_N (documents partially held out)", "heldout_proportion", "coherence_M", "exclusivity", "label_n_words", "frex_weight", "correlation_cutoff", "n_representative", "representative_min_tokens", "active_topic_threshold",
                "token_bands", "sample_stride", "selfcheck_n", "selfcheck_K", "selfcheck_its", "selfcheck_seed", "smooth_terms (stm::s default df)", "placeholder_num_comments", "placeholder_upvote_ratio",
                "text_representation", "stopword_source", "stopword_workbook_sha256", "min_token_chars", "min_document_frequency", "max_document_share",
                "heldout_share (Tier 2 split, carried only)", "split_seed (Tier 2)", "documents_entering_stm", "quick_run", "quick_n", "memory_sampler", "sampler_interval_seconds", "concurrency", "workers", "K_completed", "K_failed"),
  value = c(paste(K_values, collapse = ", "), prevalence_string, content_string, gamma_prior, 1, 2, gamma_pmax, kappa_prior, interactions, TRUE, sigma_prior, init_type, stm_seed, max_em_its, format(emtol, scientific = TRUE), 1L,
            "stm::make.heldout (as searchK)", heldout_seed, heldout_N, heldout_proportion, coherence_M, "not computed: stm::exclusivity() stops for content covariate models (recorded)", label_n_words, frex_weight, correlation_cutoff, n_representative, representative_min_tokens, active_topic_threshold,
            paste(band_labels, collapse = "; "), sample_stride, selfcheck_n, paste(selfcheck_K, collapse = ", "), selfcheck_its, selfcheck_seed, paste(sprintf("%s: df %d", smooth_terms, vapply(spline_info, function(s) s$df, integer(1))), collapse = "; "),
            format(cov_build$placeholders[["num_comments"]]), format(cov_build$placeholders[["upvote_ratio"]]),
            "delivered Tier 2 LDA representation: lower case; U+2019 -> '; URLs removed; Unicode letter-run tokens with internal apostrophes; trailing 's stripped; tokens < 2 chars removed; 310 Tier 2 entries removed; marker-only documents excluded; terms kept when 5 <= df <= 50% of non-marker documents; no stemming",
            stopword_source, workbook_hash, min_token_chars, min_document_frequency, max_document_share, heldout_share, split_seed, "all Tier 2 modelable documents (no split), then make.heldout", quick_run, if (quick_run) quick_n else NA,
            memory_sampler_on, sampler_interval, concurrency, "one Rscript process per K (this script in worker mode)", paste(K_done, collapse = ", "), if (length(K_failed)) paste(K_failed, collapse = ", ") else "none"))
write_table(parameters_table,        "parameters.csv")
write_table(input_integrity,         "input_integrity.csv")
write_table(source_files |> select(-path), "source_files.csv")
write_table(tier2_reconciliation,    "tier2_reconciliation.csv")
write_table(discrepancies,           "discrepancies.csv")
write_table(stm_input_accounting,    "stm_input_accounting.csv")
write_table(document_length_summary_stm, "document_length_summary_stm.csv")
write_table(covariate_summary,       "covariate_summary.csv")
write_table(covariate_coverage,      "covariate_coverage.csv")
write_table(missingness,             "covariate_missingness.csv")
write_table(spline_knots,            "spline_knots.csv")
write_table(design_columns_by_term,  "design_columns_by_term.csv")
write_table(heldout_construction,    "heldout_construction.csv")
write_table(selfcheck,               "selfcheck_searchk.csv")
write_table(worker_runs,             "worker_runs.csv")
write_table(searchk_results,         "searchk_results.csv")
write_table(exclusivity_status,      "exclusivity_status.csv")
write_table(coherence_summary,       "semantic_coherence_summary.csv")
write_table(coherence_by_topic,      "semantic_coherence_by_topic.csv")
write_table(convergence_trace,       "convergence_trace.csv")
write_table(fit_chunks,              "fit_chunks.csv")
write_table(model_specification,     "model_specification.csv")
write_table(runtime_by_k,            "runtime_by_k.csv")
write_table(runtime_summary,         "runtime_summary_by_k.csv")
if (!is.null(memory_by_stage) && nrow(memory_by_stage)) write_table(memory_by_stage |> select(K, everything()), "memory_by_stage_by_k.csv")
write_table(heldout_by_type,         "heldout_by_doc_type.csv")
write_table(topic_summary,           "topic_summary.csv")
write_table(prevalence_by_type,      "topic_prevalence_by_type.csv")
write_table(concentration,           "document_concentration.csv")
write_table(dominant_share_quantiles, "dominant_share_quantiles.csv")
write_table(top_words,               "top_words.csv")
write_table(kappa_labels,            "kappa_labels.csv")
write_table(kappa_sparsity,          "kappa_sparsity.csv")
write_table(kappa_sparsity_summary,  "kappa_sparsity_summary.csv")
write_table(gamma_sparsity,          "gamma_sparsity.csv")
write_table(gamma_overall,           "gamma_overall.csv")
if (nrow(gamma_path)) write_table(gamma_path, "gamma_path_by_mstep.csv")
write_table(topic_correlation,       "topic_correlation.csv")
write_table(within_k_nearest,        "within_k_nearest_topic.csv")
if (nrow(correspondence_cosine)) { write_table(correspondence_cosine, "topic_correspondence_cosine.csv"); write_table(correspondence_best, "topic_correspondence_best_match.csv"); write_table(correspondence_summary, "topic_correspondence_summary.csv") }
write_table(representative_documents, "representative_documents.csv")
write_table(linkage_checks,          "linkage_checks.csv")
for (p in per_k) {
  write_table(p$document_table, sprintf("document_topic_distribution_%s.csv", k_tag(p$K)))
  write_table(p$topic_word_table, sprintf("topic_word_distribution_%s.csv", k_tag(p$K)))
  if (!is.null(p$trace)) write_table(p$trace, sprintf("resource_trace_%s.csv", k_tag(p$K)))
}
document_linkage_table <- docs_stm |>
  transmute(stm_row, doc_key, doc_id, doc_type, source_file, author, created_utc, created_date_utc = as.character(created_date_utc), link_id, parent_id, submission_id, submission_in_sample,
            parent_kind = if_else(doc_type == "comment", if_else(substr(parent_id, 1, 3) == "t1_", "comment", "submission"), NA_character_),
            parent_in_sample = if_else(doc_type == "comment", (substr(parent_id, 1, 3) == "t1_" & parent_id %in% paste0("t1_", comments$id)) | (substr(parent_id, 1, 3) == "t3_" & parent_id %in% paste0("t3_", submissions$id)), NA),
            parent_stm_row = if_else(doc_type == "comment", match(parent_id, doc_key), NA_integer_),
            thread_id_covariate = thread_id, parent_id_covariate = parent_level, score, num_comments, upvote_ratio, num_comments_observed, upvote_ratio_observed, num_comments_covariate, upvote_ratio_covariate,
            num_comments_missing, upvote_ratio_missing, linked_submission_in_sample, url, text_source_fields, n_chars, n_tokens_letters, n_tokens_modeled, n_tokens_stm,
            n_tokens_fitted = n_tok_fit, tokens_partly_held_out = ho_flag, tier2_split = split, model_status)
write_table(document_linkage_table,  "document_linkage.csv")
rm(document_linkage_table)
mark_stage("tables")

run_end <- Sys.time()
mem_end <- process_memory()
stage_group <- c(read = "input", document_table = "input", tokenize = "input", vocabulary_dtm = "input", split = "input", reconciliation = "input", stm_input = "input", heldout = "input", common_input = "input",
                 selfcheck = "selfcheck", fits = "fits (workers)", load_fits = "fits (workers)", summaries = "assembly", correspondence = "assembly", linkage = "assembly", figures = "outputs", validation = "outputs", tables = "outputs", manifests = "outputs")
stage_timings <- bind_rows(stage_log) |>
  mutate(start_utc = format(start_utc, "%Y-%m-%dT%H:%M:%OS3Z", tz = "UTC"), end_utc = format(end_utc, "%Y-%m-%dT%H:%M:%OS3Z", tz = "UTC"), seconds = round(seconds, 3),
         stage_group = coalesce(unname(stage_group[stage]), "other"),
         note = case_when(stage == "fits" & length(cached_K) == length(K_values) ~ "every K loaded from the cache (no worker launched in this run)",
                          stage == "fits" ~ sprintf("waited for the worker processes (K = %s launched in this run; K = %s cached)", paste(setdiff(K_values, cached_K), collapse = ", "), if (length(cached_K)) paste(cached_K, collapse = ", ") else "none"),
                          stage == "heldout" & heldout_from_cache ~ "held-out object loaded from the cache",
                          TRUE ~ ""))
run_info <- tibble(
  item = c("run_id", "run_start_utc", "run_end_utc", "total_seconds", "seconds_before_run_start (R start-up and package loading)", "quick_run", "run_key", "input_key", "heldout_from_cache", "K_completed", "K_failed",
           "workers_launched_this_run", "concurrency", "process_peak_working_set_mb (main process, this run)", "cpu_logical_processors", "cpu_physical_cores", "system_memory_total_mb", "R_version", "stm_version", "glmnet_version",
           "documents", "vocabulary (Tier 2)", "vocabulary (fitted, after make.heldout)", "tokens (Tier 2)", "tokens fitted", "tokens held out", "git_commit_at_run", "script_sha256"),
  value = c(run_id, utc_stamp(run_start), utc_stamp(run_end), sprintf("%.1f", as.numeric(difftime(run_end, run_start, units = "secs"))), sprintf("%.1f", startup_elapsed_seconds), quick_run, run_key, input_key, heldout_from_cache,
            paste(K_done, collapse = ", "), if (length(K_failed)) paste(K_failed, collapse = ", ") else "none", if (length(setdiff(K_values, cached_K))) paste(setdiff(K_values, cached_K), collapse = ", ") else "none", concurrency, round(mem_end[["peak_wset_mb"]]),
            parallel::detectCores(logical = TRUE), parallel::detectCores(logical = FALSE), round(as.numeric(ps::ps_system_memory()$total) / 2^20), R.version.string, as.character(packageVersion("stm")), as.character(packageVersion("glmnet")),
            length(documents), length(vocab), length(heldout$vocab), sum(doc_tokens_stm), sum(heldout_doc_tokens), sum(missing_tokens_by_doc),
            tryCatch(system2("git", c("-C", project_dir, "rev-parse", "--short", "HEAD"), stdout = TRUE, stderr = FALSE), error = function(e) ""), sha256(script_path)))
write_table(stage_timings, "stage_timings.csv")
write_table(run_info, "run_info.csv")
model_files <- list.files(model_dir, pattern = "\\.rds$", full.names = TRUE)
preservation_status <- bind_rows(
  tibble(item = "tables (CSV, tracked)", status = "tracked", bytes = sum(file.size(list.files(tab_dir, pattern = "\\.csv$", full.names = TRUE)))),
  tibble(item = sprintf("tables/document_topic_distribution_%s.csv", k_tag(K_done)), status = "git-ignored (large; manifest tracked)", bytes = file.size(file.path(tab_dir, sprintf("document_topic_distribution_%s.csv", k_tag(K_done))))),
  tibble(item = sprintf("tables/topic_word_distribution_%s.csv", k_tag(K_done)), status = "git-ignored (large; manifest tracked; the same distributions are inside the model objects)", bytes = file.size(file.path(tab_dir, sprintf("topic_word_distribution_%s.csv", k_tag(K_done))))),
  tibble(item = "tables/document_linkage.csv", status = "git-ignored (large; manifest tracked)", bytes = file.size(file.path(tab_dir, "document_linkage.csv"))),
  tibble(item = "figures (PNG)", status = "tracked", bytes = sum(file.size(list.files(fig_dir, pattern = "\\.png$", full.names = TRUE)))),
  tibble(item = paste0("models/", basename(model_files)), status = "git-ignored (retained object; manifest tracked)", bytes = file.size(model_files)),
  tibble(item = "models/*.log, *.csv, *.txt, *.R (consoles, resource traces, status files, keys, sampler)", status = "git-ignored", bytes = sum(file.size(list.files(model_dir, pattern = "\\.(log|csv|txt|R)$", full.names = TRUE)))),
  tibble(item = "script", status = "tracked", bytes = file.size(script_path)),
  tibble(item = "tests/test_stm_searchk_covariates.R", status = "tracked", bytes = file.size(file.path(project_dir, "tests/test_stm_searchk_covariates.R"))))
write_table(preservation_status, "preservation_status.csv")

# ---- Provenance sidecars ----
# INPUT : the CSV, PNG and RDS outputs; the source files; the workbook; the delivered artifacts read.
# DOES  : write one <output>.manifest.yml per output (input hashes, output hash and dimensions,
#         script hash, git commit, parameters, seed, package versions; lite manifests with the
#         principal dimensions for the RDS objects).
# OUTPUT: *.manifest.yml next to each output.
git_commit <- tryCatch(system2("git", c("-C", project_dir, "rev-parse", "--short", "HEAD"), stdout = TRUE, stderr = FALSE), error = function(e) "")
if (length(git_commit) == 0L) git_commit <- ""
manifest_packages <- c("stm", "lda", "glmnet", "matrixStats", "nanoparquet", "dplyr", "tidyr", "stringi", "Matrix", "readxl", "ps", "processx", "clue", "ggplot2", "scales", "patchwork", "readr", "digest", "yaml")
package_versions  <- lapply(manifest_packages, function(p) list(name = p, version = as.character(packageVersion(p))))
input_files <- c(
  lapply(seq_len(nrow(source_files)), function(i) list(path = source_files$path[i], hash = source_files$sha256[i], format = "parquet", rows = as.integer(source_files$rows[i]), role = "source data, read-only")),
  list(list(path = stopword_workbook, hash = workbook_hash, format = "xlsx", rows = nrow(tier2_raw), cols = ncol(tier2_raw), sheet = stopword_sheet, role = "Tier 2 stopword list, read-only")),
  lapply(which(input_integrity$exists), function(i) list(path = input_integrity$path[i], hash = input_integrity$sha256_now[i], format = if (grepl("\\.rds$", input_integrity$path[i])) "rds" else "csv",
                                                         role = "delivered artifact, read-only (reconciliation)")))
manifest_parameters <- c(list(model = sprintf("Structural Topic Models (stm::stm variational EM) for K = %s with searchK's held-out construction; prevalence %s (gamma.prior L1); content %s (kappa.prior L1); spectral initialisation", paste(K_values, collapse = ", "), prevalence_string, content_string),
                              text_representation_source = "the delivered Tier 2 LDA representation reproduced with the delivered code (see parameters.csv)",
                              documents = "all Tier 2 modelable documents (857,385 in the full run), doc_key = Reddit fullname, source order; 10% partially held out by stm::make.heldout",
                              topic_numbering = "each model's own topic index (no renumbering); prevalence_rank is a separate column",
                              retained = "models/heldout_common.rds, common_input.rds, stm_K*.rds, fit_record_K*.rds, heldout_eval_K*.rds, semcoh_K*.rds, labels_K*.rds, residuals_K*.rds"),
                         setNames(as.list(as.character(parameters_table$value)), parameters_table$parameter))
rds_dims <- list("heldout_common.rds" = c(rows = length(heldout$documents), cols = length(heldout$vocab)), "common_input.rds" = c(rows = length(heldout$documents), cols = length(heldout$vocab)))
for (p in per_k) {
  f <- fits[[k_tag(p$K)]]
  rds_dims[[sprintf("stm_%s.rds", k_tag(p$K))]]          <- c(rows = N_fit, cols = p$K)
  rds_dims[[sprintf("fit_record_%s.rds", k_tag(p$K))]]   <- c(rows = length(f$fit_record$log_lines), cols = length(f$fit_record))
  rds_dims[[sprintf("heldout_eval_%s.rds", k_tag(p$K))]] <- c(rows = length(f$heldout_eval$doc_heldout), cols = 4L)
  rds_dims[[sprintf("semcoh_%s.rds", k_tag(p$K))]]       <- c(rows = p$K, cols = 2L)
  rds_dims[[sprintf("labels_%s.rds", k_tag(p$K))]]       <- c(rows = p$K, cols = label_n_words)
  rds_dims[[sprintf("residuals_%s.rds", k_tag(p$K))]]    <- c(rows = 1L, cols = 3L)
}
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
  manifest$seed      <- heldout_seed
  manifest$timestamp <- utc_stamp()
  manifest$notes     <- if (ext == "rds") "Retained object of the STM searchK evidence run (July 2026 r/politics, Tier 2 text representation, K = 10, 20, 30): the common held-out object (documents x vocabulary), the worker input, an STM model object (theta documents x topics), a fit record (console log lines x fields), a held-out evaluation (held-out documents x fields), coherence (topics x measures), labels (topics x words) or residual dispersion (1 x 3). Source Parquet and delivered artifacts read-only." else
    "STM searchK evidence for K = 10, 20, 30 on the July 2026 r/politics sample (delivered Tier 2 text representation); one common held-out object (make.heldout, seed recorded); prevalence with author, s(created_utc), doc_type, thread_id, parent_id, s(score), s(num_comments), s(upvote_ratio) and missingness indicators under L1; content ~ doc_type; spectral initialisation; no K chosen. Source Parquet and delivered artifacts read-only."
  write_yaml(manifest, paste0(output, ".manifest.yml"))
}
invisible(lapply(list.files(tab_dir,   pattern = "\\.csv$", full.names = TRUE), write_sidecar))
invisible(lapply(list.files(fig_dir,   pattern = "\\.png$", full.names = TRUE), write_sidecar))
invisible(lapply(model_files, write_sidecar))
mark_stage("manifests")

# ---- Console summary ----
print(as.data.frame(stm_input_accounting))
print(as.data.frame(heldout_construction), right = FALSE)
print(as.data.frame(searchk_results |> select(K, heldout, residual, bound, lbound, em.its, converged, semcoh, fit_wall_seconds)), right = FALSE)
print(as.data.frame(coherence_summary |> select(K, semcoh_mean, semcoh_median, semcoh_min, semcoh_max, npmi_mean)))
print(as.data.frame(runtime_summary |> select(K, em_iterations, converged, fit_wall_seconds, estep_seconds_mean, mstep_seconds_mean, residuals_seconds, worker_total_seconds, process_peak_working_set_mb_after_fit, sampler_max_wset_mb)))
print(as.data.frame(gamma_sparsity |> select(K, term, coefficient_rows, rows_with_nonzero_coefficients, share_nonzero)))
print(as.data.frame(validation_checks |> select(check, pass)), right = FALSE)
print(as.data.frame(run_info), right = FALSE)
cat("\nDone. Figures ->", fig_dir, "\nTables ->", tab_dir, "\nModels ->", model_dir, "\n")
