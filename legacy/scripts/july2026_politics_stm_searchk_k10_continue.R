# Continues the K = 10 STM fit from the searchK run (stopped at its 25-iteration ceiling before converging)
# through stm's restart, stm(model = <saved model>), with the searchK run's settings otherwise unchanged.
# Input: r_analysis_outputs/stm_searchk_k10_k20_k30/models/ and the searchK script, whose helpers are reused.
# Output: the continued models/stm_K010.rds (the stopped model kept as stm_K010_iter25_stopped.rds) and
# tables in r_analysis_outputs/stm_searchk_k10_continuation/. A re-run resumes from the last saved chunk.
# Env: STM_CONTINUE_MAX_EM_ITS (default 125), STM_FIT_CHUNK (default 5), STM_MEMORY_SAMPLER=0,
#      STM_QUICK=<folder>, STM_CONTINUE_ALLOW_SCRIPT_CHANGE=1, STM_CONTINUE_STOP_AFTER_CHUNKS=n (debug).

# ---- Setup: packages, paths, parameters ----
# Paths, continuation settings and switches; clears the continuation tables.
suppressPackageStartupMessages({ library(dplyr); library(tibble); library(Matrix); library(stm); library(ps); library(readr); library(digest); library(yaml) })
startup_elapsed_seconds <- proc.time()[["elapsed"]]
run_start <- Sys.time()
run_id    <- format(run_start, "%Y%m%dT%H%M%SZ", tz = "UTC")
project_dir  <- Sys.getenv("KC_REPO_ROOT", unset = getwd())
main_script  <- file.path(project_dir, "july2026_politics_stm_searchk_k10_k20_k30.R")
script_path  <- file.path(project_dir, "july2026_politics_stm_searchk_k10_continue.R")
rscript_path <- file.path(R.home("bin"), "Rscript.exe")
quick_run    <- nzchar(Sys.getenv("STM_QUICK"))
searchk_dir  <- if (quick_run) file.path(Sys.getenv("STM_QUICK"), "stm_searchk_k10_k20_k30_quick") else file.path(project_dir, "r_analysis_outputs/stm_searchk_k10_k20_k30")
out_dir      <- if (quick_run) file.path(Sys.getenv("STM_QUICK"), "stm_searchk_k10_continuation_quick") else file.path(project_dir, "r_analysis_outputs/stm_searchk_k10_continuation")
model_dir    <- file.path(searchk_dir, "models")
tab_dir      <- file.path(out_dir, "tables")
dir.create(tab_dir, recursive = TRUE, showWarnings = FALSE)
invisible(file.remove(list.files(tab_dir, pattern = "\\.(csv|yml)$", full.names = TRUE)))

continue_K          <- 10L                                                                       # the only K continued
continue_ceiling    <- as.integer(Sys.getenv("STM_CONTINUE_MAX_EM_ITS", if (quick_run) "4" else "125"))   # total EM iterations (the searchK ceiling was 25)
fit_chunk           <- as.integer(Sys.getenv("STM_FIT_CHUNK", if (quick_run) "1" else "5"))    # EM iterations per stm() call, as in the searchK run
memory_sampler_on   <- !identical(Sys.getenv("STM_MEMORY_SAMPLER"), "0")
sampler_interval    <- if (quick_run) 5 else 20
allow_script_change <- identical(Sys.getenv("STM_CONTINUE_ALLOW_SCRIPT_CHANGE"), "1")
stop_after_chunks   <- as.integer(Sys.getenv("STM_CONTINUE_STOP_AFTER_CHUNKS", "0"))
gamma_pmax          <- NA_integer_   # set from the common input before the fit; read by the opt.mu replacement at call time
settings_ignore     <- c("convergence.max.em.its", "call")   # the two settings a restart necessarily changes: the target iteration count and the recorded call

# ---- The searchK script's own functions, extracted from the script ----
# Evaluates only the named helper definitions from the parsed searchK script (logging, fit-log parser, memory
# sampler, the capped opt.mu), so the continuation runs the same code as the searchK run.
main_exprs <- parse(main_script, keep.source = FALSE)
extract_def <- function(name) {
  is_def <- vapply(main_exprs, function(e) is.call(e) && identical(e[[1]], as.name("<-")) && identical(e[[2]], as.name(name)), logical(1))
  if (sum(is_def) != 1L) stop(sprintf("'%s' must be defined exactly once at the top level of %s (found %d)", name, basename(main_script), sum(is_def)))
  eval(main_exprs[[which(is_def)]], envir = globalenv())
}
reused_definitions <- c("process_memory", "log_line", "utc_stamp", "secs_since", "sha256", "k_tag", "worker_file", "parse_fit_log", "fit_log_message",
                        "sampler_code", "start_sampler", "stop_sampler", "stm_opt_mu_original", "gamma_call_counter", "gamma_log_csv", "opt_mu_recorded")
for (nm in reused_definitions) extract_def(nm)
main_script_hash <- sha256(main_script)
stopifnot(identical(stm_opt_mu_original, stm:::opt.mu))   # the package function is still untouched in this process

# ---- Pure functions (tested by tests/test_stm_searchk_k10_continuation.R) ----
# chunk_targets: the planned stm() targets (total iterations) from the current iteration to the ceiling.
chunk_targets <- function(its_now, ceiling, chunk) {
  chunk <- max(1L, as.integer(chunk)); its_now <- as.integer(its_now); ceiling <- as.integer(ceiling)
  if (its_now >= ceiling) return(integer(0))
  first <- its_now + chunk
  t <- if (first > ceiling) integer(0) else seq.int(first, ceiling, by = chunk)
  if (!length(t) || tail(t, 1) < ceiling) t <- c(t, ceiling)
  as.integer(t)
}
# settings_diff: every leaf of two nested settings lists (dotted paths), display values, equality
# (formulas by their text, everything else by identical()), with the expected differences flagged.
flatten_settings <- function(x, prefix = "") {
  if (is.list(x) && !is.data.frame(x) && !inherits(x, "formula")) {
    if (!length(x)) return(setNames(list(x), prefix))
    nms <- names(x); if (is.null(nms)) nms <- rep("", length(x))
    nms[nms == ""] <- as.character(seq_along(x))[nms == ""]
    out <- list()
    for (i in seq_along(x)) out <- c(out, flatten_settings(x[[i]], if (nzchar(prefix)) paste(prefix, nms[i], sep = ".") else nms[i]))
    return(out)
  }
  setNames(list(x), prefix)
}
leaf_display <- function(x) {
  if (is.null(x)) return("NULL")
  if (inherits(x, "formula")) return(paste(deparse(x, width.cutoff = 500L), collapse = " "))
  if (is.language(x)) return(paste(deparse(x), collapse = " "))
  if (is.atomic(x) && length(x) <= 10L) return(paste(format(x), collapse = ", "))
  if (is.atomic(x)) return(sprintf("<%s[%d]>", class(x)[1], length(x)))
  d <- dim(x)
  sprintf("<%s %s>", class(x)[1], if (is.null(d)) sprintf("length %d", length(x)) else paste(d, collapse = " x "))
}
leaf_equal <- function(x, y) {
  if (inherits(x, "formula") || inherits(y, "formula")) return(inherits(x, "formula") && inherits(y, "formula") && identical(leaf_display(x), leaf_display(y)))
  identical(x, y)
}
settings_diff <- function(a, b, ignore = character(0)) {
  fa <- flatten_settings(a); fb <- flatten_settings(b)
  leaves <- union(names(fa), names(fb))
  tibble(leaf = leaves,
         in_a = vapply(leaves, function(l) if (l %in% names(fa)) leaf_display(fa[[l]]) else "<absent>", character(1)),
         in_b = vapply(leaves, function(l) if (l %in% names(fb)) leaf_display(fb[[l]]) else "<absent>", character(1)),
         equal = vapply(leaves, function(l) l %in% names(fa) && l %in% names(fb) && leaf_equal(fa[[l]], fb[[l]]), logical(1)),
         ignored = leaves %in% ignore)
}
# merge_fit_record: the searchK fit record plus the continuation record; totals over both, chunks
# renumbered with a phase column, console lines concatenated, both records kept whole.
merge_fit_record <- function(original, continuation) {
  ct <- bind_rows(original$chunk_table |> mutate(phase = "searchK run"), continuation$chunk_table |> mutate(phase = "continuation")) |>
    mutate(chunk = row_number()) |> select(chunk, phase, everything())
  merged <- original
  merged$fit_end_utc <- continuation$fit_end_utc
  merged$wall_seconds <- original$wall_seconds + continuation$wall_seconds
  merged$wall_seconds_including_saves <- original$wall_seconds_including_saves + continuation$wall_seconds_including_saves
  merged$cpu_user_seconds <- original$cpu_user_seconds + continuation$cpu_user_seconds
  merged$cpu_system_seconds <- original$cpu_system_seconds + continuation$cpu_system_seconds
  merged$model_time_seconds <- original$model_time_seconds + continuation$model_time_seconds
  merged$chunks <- nrow(ct); merged$chunk_table <- ct
  merged$peak_wset_mb_after_fit <- max(original$peak_wset_mb_after_fit, continuation$peak_wset_mb_after_fit)
  merged$rss_mb_after_fit <- continuation$rss_mb_after_fit
  merged$log_lines <- c(original$log_lines, continuation$log_lines)
  merged$model_file_bytes <- continuation$model_file_bytes
  merged$pid <- continuation$pid
  merged$gamma_calls <- original$gamma_calls + continuation$gamma_calls
  merged$continued <- TRUE
  merged$continued_from_iteration <- continuation$its_from
  merged$iterations_after_continuation <- continuation$its_to
  merged$converged <- continuation$converged
  merged$continuation_run_id <- continuation$run_id
  merged$original_run <- original
  merged$continuation <- continuation
  merged
}
# continuation_state: what the retained object currently is, from its hash, its sidecar's hash and the
# continuation record (if any): "pending" (the stopped state as saved), "applied" (the continued
# model this script installed) or "unknown" (neither; nothing is touched).
continuation_state <- function(current_hash, sidecar_hash, record) {
  if (!is.null(record)) return(if (identical(current_hash, record$continued_model_hash)) "applied" else "unknown")
  if (identical(current_hash, sidecar_hash)) "pending" else "unknown"
}
# annotate_manifest: an existing sidecar re-pointed at a renamed output with a note appended.
annotate_manifest <- function(manifest, new_output_file, note) {
  manifest$output_file <- new_output_file
  manifest$notes <- if (is.null(manifest$notes)) note else paste(manifest$notes, note)
  manifest
}

log_line("start; quick_run =", quick_run, "; K =", continue_K, "; ceiling =", continue_ceiling, "; chunk =", fit_chunk, "; run_id", run_id)

# ---- The searchK run's state ----
# Loads the common input, the retained K = 10 model, its fit record and sidecar. In the pending state, checks
# that the model hash and settings match the sidecar; any failure stops before files are touched.
tag <- k_tag(continue_K)
common_input_path <- file.path(model_dir, "common_input.rds")
model_path        <- worker_file(continue_K, "stm")
fit_record_path   <- worker_file(continue_K, "fit_record")
gamma_orig_csv    <- worker_file(continue_K, "gamma_path", "csv")
record_path       <- file.path(model_dir, sprintf("continuation_%s_record.rds", tag))
partial_path      <- file.path(model_dir, sprintf("stm_partial_%s_continuation.rds", tag))
fit_log           <- file.path(model_dir, sprintf("fit_console_%s_continuation.log", tag))
gamma_cont_csv    <- file.path(model_dir, sprintf("gamma_path_%s_continuation.csv", tag))
status_file       <- file.path(model_dir, sprintf("worker_%s_continuation_status.txt", tag))
stage_file        <- file.path(model_dir, sprintf("stage_marker_%s_continuation.txt", tag))
stop_file         <- file.path(model_dir, sprintf("sampler_stop_%s_continuation.txt", tag))
sampler_path      <- file.path(model_dir, "memory_sampler.R")
sampler_log       <- file.path(model_dir, sprintf("memory_sampler_%s_continuation.log", tag))
trace_csv         <- file.path(model_dir, sprintf("resource_trace_%s_continuation_%s.csv", tag, run_id))
set_stage <- function(name) writeLines(name, stage_file)
for (p in c(common_input_path, model_path, fit_record_path, gamma_orig_csv, paste0(model_path, ".manifest.yml"), file.path(model_dir, "run_key.txt"))) if (!file.exists(p)) stop("missing: ", p)
t_load <- Sys.time()
ci <- readRDS(common_input_path)
run_key <- readLines(file.path(model_dir, "run_key.txt"), warn = FALSE)[1]
stopifnot(identical(ci$run_key, run_key), continue_K %in% ci$K_values)
sidecar_orig <- read_yaml(paste0(model_path, ".manifest.yml"))
current_hash <- sha256(model_path)
record <- if (file.exists(record_path)) readRDS(record_path) else NULL
state  <- continuation_state(current_hash, sidecar_orig$output_hash, record)
log_line("retained object", basename(model_path), "hash", current_hash, "; state:", state)
if (state == "unknown") stop(sprintf("The retained object %s is neither the stopped state its sidecar describes (%s) nor the continued model this script recorded (%s); nothing touched. Resolve by hand (restore the object or remove %s).",
                                     model_path, sidecar_orig$output_hash, if (is.null(record)) "no record" else record$continued_model_hash, basename(record_path)))
if (state == "pending") {
  model_orig <- readRDS(model_path)
  its0 <- model_orig$convergence$its
  started_from_hash <- current_hash
  fit_record_orig <- readRDS(fit_record_path)
  inherited_input_files <- sidecar_orig$input_files
} else {
  its0 <- record$its_from
  started_from_hash <- record$started_from_hash
  model_orig <- readRDS(record$iter_model_path)
  fit_record_orig <- readRDS(record$fit_record_iter_path)
  inherited_input_files <- record$inherited_input_files
  model_new <- readRDS(model_path)
  fit_record_merged <- readRDS(fit_record_path)
  stopifnot(identical(fit_record_merged$continuation$run_id, record$run_id), model_new$convergence$its == record$its_to)
}
iter_model_path      <- file.path(model_dir, sprintf("stm_%s_iter%d_stopped.rds", tag, its0))
fit_record_iter_path <- file.path(model_dir, sprintf("fit_record_%s_iter%d_stopped.rds", tag, its0))
gamma_orig <- read_csv(gamma_orig_csv, show_col_types = FALSE, progress = FALSE, col_types = cols(time_utc = col_character(), chosen_is_last_computed = col_logical(), .default = col_double()))
n_tokens_fit <- sum(model_orig$settings$dim$wcounts$x)
verify <- list()
add_verify <- function(item, value, pass) verify[[length(verify) + 1L]] <<- tibble(stage = "starting state", check = item, detail = as.character(value), pass = isTRUE(pass))
so <- model_orig$settings
add_verify("retained object hash equals its sidecar's output_hash (the stopped state as saved) or the recorded continuation", started_from_hash, state == "applied" || identical(current_hash, sidecar_orig$output_hash))
add_verify("stopped state: not converged, stopped by the ceiling, bound trace length equals the iteration count", sprintf("its %d; converged %s; stopits %s; bound length %d", its0, model_orig$convergence$converged, model_orig$convergence$stopits, length(model_orig$convergence$bound)),
           !model_orig$convergence$converged && isTRUE(model_orig$convergence$stopits) && length(model_orig$convergence$bound) == its0)
add_verify("iterations equal the fit record's last chunk and the L1 path record has one row per iteration", sprintf("its %d; last chunk %d; L1 rows %d", its0, tail(fit_record_orig$chunk_table$iterations_after, 1), nrow(gamma_orig)),
           its0 == tail(fit_record_orig$chunk_table$iterations_after, 1) && nrow(gamma_orig) == its0 && identical(fit_record_orig$run_key, run_key) && fit_record_orig$K == continue_K)
add_verify("model dimensions equal the common input (N documents, V terms, K) and the vocabulary is identical", sprintf("N %d V %d K %d A %d", so$dim$N, so$dim$V, so$dim$K, so$dim$A),
           so$dim$N == length(ci$documents) && so$dim$V == length(ci$vocab) && so$dim$K == continue_K && so$dim$A == 2L && identical(model_orig$vocab, ci$vocab) && nrow(ci$meta_df) == so$dim$N)
add_verify("prevalence formula in the model equals the common input's formula string; content levels comment, submission", paste(deparse(so$covariates$formula, width.cutoff = 500L), collapse = " "),
           identical(paste(deparse(so$covariates$formula, width.cutoff = 500L), collapse = " "), ci$prevalence_string) && identical(so$covariates$yvarlevels, c("comment", "submission")) && identical(ci$content_string, "~doc_type"))
add_verify("settings of the stopped state equal the common input's settings (gamma L1, kappa L1, spectral, emtol, seed, sigma prior, interactions, ngroups 1)",
           sprintf("gamma %s; tau %s; init %s; emtol %s; seed %s; sigma %s; interactions %s; ngroups %s", so$gamma$mode, so$tau$mode, so$init$mode, format(so$convergence$em.converge.thresh), so$seed, so$sigma$prior, so$kappa$interactions, so$ngroups),
           so$gamma$mode == ci$settings$gamma_prior && so$tau$mode == ci$settings$kappa_prior && so$init$mode == ci$settings$init_type && so$convergence$em.converge.thresh == ci$settings$emtol && so$seed == ci$settings$stm_seed &&
             so$sigma$prior == ci$settings$sigma_prior && so$kappa$interactions == ci$settings$interactions && so$ngroups == 1L && isTRUE(so$convergence$allow.neg.change))
add_verify("L1 memory cap of the fit record equals the common input's setting (applied again here)", sprintf("pmax %d", fit_record_orig$gamma_pmax), fit_record_orig$gamma_pmax == ci$settings$gamma_pmax && all(gamma_orig$pmax == ci$settings$gamma_pmax))
add_verify("searchK script hash equals the hash recorded in the retained object's sidecar (the extracted functions are the run's)", main_script_hash, identical(main_script_hash, sidecar_orig$transformation$script_hash) || (state == "applied" && identical(main_script_hash, record$main_script_hash)))
add_verify("ceiling above the current iteration count", sprintf("ceiling %d; its %d", continue_ceiling, if (state == "pending") its0 else record$its_to), continue_ceiling > (if (state == "pending") its0 else record$its_to) || state == "applied")
verify_tbl <- bind_rows(verify)
script_hash_ok <- verify_tbl$pass[grepl("^searchK script hash", verify_tbl$check)]
if (!script_hash_ok && allow_script_change) { log_line("WARNING: the searchK script's hash differs from the sidecar's record; continuing because STM_CONTINUE_ALLOW_SCRIPT_CHANGE=1"); verify_tbl$pass[grepl("^searchK script hash", verify_tbl$check)] <- NA }
blocking_fail <- verify_tbl |> filter(!is.na(pass), !pass)
if (nrow(blocking_fail)) { print(as.data.frame(blocking_fail), right = FALSE); stop("starting-state verification failed; nothing touched") }
log_line("starting state verified:", sum(verify_tbl$pass, na.rm = TRUE), "of", nrow(verify_tbl), "checks; its0 =", its0, "; loaded in", round(secs_since(t_load)), "s")

# ---- Continuation of the EM through stm's restart ----
# Installs the capped opt.mu, then calls stm(..., model = model) in chunks until converged or at the ceiling,
# saving the partial model after each chunk. Installs the continued model as stm_K010.rds, renames the
# stopped one, and writes the merged fit record and the continuation record.
fit_needed <- state == "pending" || (!model_new$convergence$converged && continue_ceiling > model_new$convergence$its)
if (state == "applied" && !fit_needed) log_line("continuation already applied (run", record$run_id, "); its", record$its_to, "; converged", record$converged, "; regenerating the tables and sidecars")
if (fit_needed) {
  gamma_pmax <- ci$settings$gamma_pmax
  gamma_log_csv <- gamma_cont_csv
  assignInNamespace("opt.mu", opt_mu_recorded, ns = "stm")
  prevalence_formula_w <- as.formula(ci$prevalence_string); environment(prevalence_formula_w) <- globalenv()
  content_formula_w    <- as.formula(ci$content_string);    environment(content_formula_w)    <- globalenv()
  K <- continue_K
  partial_key <- digest(list(run_key, started_from_hash, gamma_pmax, ci$settings$emtol), algo = "sha256")
  writeLines(c("running", run_key, run_id, utc_stamp()), status_file)
  writeLines(sampler_code, sampler_path)
  set_stage("load")
  sampler_started <- start_sampler(trace_csv, stage_file, stop_file, sampler_log)
  on_error <- function(e) {
    while (sink.number() > 0) sink()
    writeLines(c(paste0("error: ", conditionMessage(e)), run_key, run_id, utc_stamp()), status_file)
    log_line("CONTINUATION ERROR:", conditionMessage(e))
    if (sampler_started) stop_sampler(TRUE, stop_file)
    quit(save = "no", status = 1)
  }
  withCallingHandlers(tryCatch({
    # the starting point: the stopped state, the partial model of an interrupted run, or the continued model when the ceiling was raised
    model <- model_orig; chunk_records <- list(); fit_start <- Sys.time(); resumed <- FALSE; resumed_runs <- character(0); gamma_call_counter <- its0
    if (file.exists(partial_path)) {
      pm <- readRDS(partial_path)
      if (identical(pm$partial_key, partial_key) && !is.null(pm$model) && pm$model$convergence$its > its0) {
        model <- pm$model; chunk_records <- pm$chunk_records; fit_start <- pm$fit_start; gamma_call_counter <- pm$gamma_calls; resumed <- TRUE; resumed_runs <- c(pm$resumed_runs, pm$run_id)
        log_line("resuming from the partial model after", model$convergence$its, "iterations (saved by run", pm$run_id, ")")
      } else log_line("partial model present but not usable (different key or no progress); ignored")
      rm(pm)
    } else if (state == "applied") {
      model <- model_new; chunk_records <- lapply(seq_len(nrow(record$chunk_table)), function(i) record$chunk_table[i, ]); fit_start <- as.POSIXct(record$fit_start_utc, format = "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")
      gamma_call_counter <- its0 + record$gamma_calls; resumed <- TRUE; resumed_runs <- c(record$resumed_runs, record$run_id)
      log_line("continuing the continued model further from iteration", model$convergence$its, "(ceiling raised to", continue_ceiling, ")")
    }
    if (!resumed) { if (file.exists(gamma_cont_csv)) invisible(file.remove(gamma_cont_csv)); if (file.exists(fit_log)) invisible(file.remove(fit_log)) }
    set_stage("fit")
    chunks_this_run <- 0L
    while (model$convergence$its < continue_ceiling && !model$convergence$converged) {
      target <- min(model$convergence$its + max(1L, fit_chunk), continue_ceiling)
      log_con <- file(fit_log, open = "at")
      sink(log_con, split = TRUE)
      cat(sprintf("run_id %s; K = %d; gamma_pmax %d; continuation from iteration %d (started from iteration %d); chunk to iteration %d; start %s\n", run_id, K, gamma_pmax, model$convergence$its, its0, target, utc_stamp()))
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
      saveRDS(list(partial_key = partial_key, run_id = run_id, model = model, chunk_records = chunk_records, fit_start = fit_start, gamma_calls = gamma_call_counter, resumed_runs = resumed_runs), partial_path, compress = FALSE)
      chunks_this_run <- chunks_this_run + 1L
      log_line("chunk done;", model$convergence$its, "of at most", continue_ceiling, "iterations; converged", model$convergence$converged, "; partial model saved")
      if (stop_after_chunks > 0L && chunks_this_run >= stop_after_chunks) {
        log_line("STM_CONTINUE_STOP_AFTER_CHUNKS reached; exiting with the partial model saved (debug)")
        stop_sampler(sampler_started, stop_file); writeLines(c("interrupted (debug switch)", run_key, run_id, utc_stamp()), status_file); quit(save = "no", status = 3)
      }
    }
    fit_end <- Sys.time()
    set_stage("save")
    chunk_table <- bind_rows(chunk_records)
    mem_after_fit <- process_memory()
    tmp_path <- file.path(model_dir, sprintf("stm_%s_continued_tmp.rds", tag))
    saveRDS(model, tmp_path, compress = FALSE)
    continued_hash <- sha256(tmp_path)
    if (state == "pending") {
      if (!identical(sha256(model_path), started_from_hash)) stop("the retained object changed during the fit; not replaced")
      stopifnot(file.rename(model_path, iter_model_path))
      write_yaml(annotate_manifest(sidecar_orig, iter_model_path, sprintf("Renamed on %s by %s: this is the K = %d model as the searchK run stopped it (iteration %d, not converged); the retained object %s was continued from this state through stm(model = ) and replaced (see continuation_%s_record.rds and r_analysis_outputs/stm_searchk_k10_continuation/).",
                                                                       utc_stamp(), basename(script_path), continue_K, its0, basename(model_path), tag)), paste0(iter_model_path, ".manifest.yml"))
      saveRDS(fit_record_orig, fit_record_iter_path)
      fr_sidecar <- paste0(fit_record_path, ".manifest.yml")
      if (file.exists(fr_sidecar)) write_yaml(annotate_manifest(read_yaml(fr_sidecar), fit_record_iter_path, sprintf("Renamed on %s by %s: the fit record of the searchK run's K = %d fit (iteration %d, not converged); %s now holds the merged record after the continuation.",
                                                                                                                utc_stamp(), basename(script_path), continue_K, its0, basename(fit_record_path))), paste0(fit_record_iter_path, ".manifest.yml"))
    } else {
      stopifnot(file.remove(model_path))
    }
    stopifnot(file.rename(tmp_path, model_path))
    if (!identical(sha256(model_path), continued_hash)) stop("the installed model's hash differs from the saved one")
    trace_files <- list.files(model_dir, pattern = sprintf("^resource_trace_%s_continuation_.*\\.csv$", tag), full.names = TRUE)
    continuation <- list(run_id = run_id, resumed_runs = resumed_runs, ceiling = continue_ceiling, fit_chunk = fit_chunk, its_from = its0, its_to = model$convergence$its, converged = model$convergence$converged,
                         fit_start_utc = utc_stamp(fit_start), fit_end_utc = utc_stamp(fit_end), wall_seconds = sum(chunk_table$wall_seconds), wall_seconds_including_saves = as.numeric(difftime(fit_end, fit_start, units = "secs")),
                         cpu_user_seconds = sum(chunk_table$cpu_user_seconds), cpu_system_seconds = sum(chunk_table$cpu_system_seconds), model_time_seconds = sum(chunk_table$stm_time_seconds), chunk_table = chunk_table,
                         peak_wset_mb_after_fit = unname(mem_after_fit[["peak_wset_mb"]]), rss_mb_after_fit = unname(mem_after_fit[["rss_mb"]]), log_lines = readLines(fit_log, warn = FALSE), trace_csv = trace_csv, trace_files = trace_files,
                         model_file_bytes = file.size(model_path), pid = Sys.getpid(), gamma_calls = gamma_call_counter - its0, gamma_pmax = gamma_pmax, stm_version = as.character(packageVersion("stm")), glmnet_version = as.character(packageVersion("glmnet")),
                         started_from_hash = started_from_hash, continued_model_hash = continued_hash, iter_model_path = iter_model_path, fit_record_iter_path = fit_record_iter_path, inherited_input_files = inherited_input_files,
                         main_script_hash = main_script_hash, main_script_hash_equals_sidecar = identical(main_script_hash, sidecar_orig$transformation$script_hash) || state == "applied", continuation_script_hash = sha256(script_path),
                         bound_before = tail(model_orig$convergence$bound, 1), bound_after = tail(model$convergence$bound, 1), emtol = ci$settings$emtol, quick_run = quick_run)
    fit_record_merged <- merge_fit_record(fit_record_orig, continuation)
    saveRDS(fit_record_merged, fit_record_path)
    record <- continuation
    saveRDS(record, record_path)
    if (file.exists(partial_path)) invisible(file.remove(partial_path))
    stop_sampler(sampler_started, stop_file)
    writeLines(c("done", run_key, run_id, utc_stamp()), status_file)
    model_new <- model
    log_line("continuation saved and installed:", its0, "->", model_new$convergence$its, "iterations; converged", model_new$convergence$converged, ";", round(continuation$wall_seconds), "s of fitting;", nrow(chunk_table), "chunks")
  }, error = on_error), warning = function(w) { log_line("warning:", conditionMessage(w)); invokeRestart("muffleWarning") })
}

# ---- Fit-accounting checks on the continued model ----
# Bound trace preserved; iterations and convergence consistent; inputs, design matrix and settings identical
# apart from the ceiling; theta and beta valid; bound increased; K = 20 and K = 30 unchanged.
set_stage("qc")
its_new   <- model_new$convergence$its
bound_new <- model_new$convergence$bound
bound_orig <- model_orig$convergence$bound
emtol     <- model_orig$settings$convergence$em.converge.thresh
rel_change <- c(NA, diff(bound_new) / abs(head(bound_new, -1)))
cont_rel   <- rel_change[(its0 + 1L):its_new]
sd <- settings_diff(model_orig$settings, model_new$settings, ignore = settings_ignore)
X_o <- model_orig$settings$covariates$X; X_n <- model_new$settings$covariates$X
x_dims_equal <- identical(dim(X_o), dim(X_n)) && identical(colnames(X_o), colnames(X_n))
x_max_abs_diff <- if (x_dims_equal && identical(X_o@i, X_n@i) && identical(X_o@p, X_n@p)) max(abs(X_o@x - X_n@x)) else NA_real_
x_identical <- identical(X_o, X_n)
lambda_new <- cbind(model_new$eta, 0)
theta_from_eta_max_diff <- max(abs(exp(lambda_new - log(rowSums(exp(lambda_new)))) - model_new$theta))
rm(lambda_new)
gamma_cont <- if (file.exists(gamma_cont_csv)) read_csv(gamma_cont_csv, show_col_types = FALSE, progress = FALSE, col_types = cols(time_utc = col_character(), chosen_is_last_computed = col_logical(), .default = col_double())) else tibble()
log_cont   <- parse_fit_log(record$log_lines)
log_orig   <- parse_fit_log(fit_record_orig$log_lines)
other_K <- setdiff(ci$K_values, continue_K)
other_untouched <- vapply(other_K, function(k) { p <- worker_file(k, "stm"); mf <- paste0(p, ".manifest.yml"); file.exists(p) && file.exists(mf) && identical(sha256(p), read_yaml(mf)$output_hash) }, logical(1))
hash_model_now <- sha256(model_path); hash_iter_now <- sha256(iter_model_path)
checks <- list()
add_check <- function(check, detail, pass) checks[[length(checks) + 1L]] <<- tibble(stage = "continued model", check = check, detail = as.character(detail), pass = isTRUE(pass))
add_check("saved state reused: the bound trace up to the starting iteration is exactly the stopped state's (no re-initialisation)", sprintf("iterations 1-%d identical: %s", its0, identical(bound_new[seq_len(its0)], bound_orig)), identical(bound_new[seq_len(its0)], bound_orig))
add_check("iteration counts: bound trace length equals the iteration count; more iterations than the start; at most the ceiling", sprintf("its %d -> %d; trace %d; ceiling %d", its0, its_new, length(bound_new), record$ceiling), length(bound_new) == its_new && its_new > its0 && its_new <= record$ceiling)
add_check("stm's convergence rule: converged means the last relative change is below emtol (allow.neg.change TRUE, the package default); otherwise the ceiling was reached", sprintf("converged %s; last relative change %s; emtol %s", model_new$convergence$converged, format(tail(rel_change, 1), digits = 4), format(emtol)),
          if (model_new$convergence$converged) tail(rel_change, 1) < emtol && isTRUE(model_new$convergence$stopits) else its_new == record$ceiling)
add_check("the model's max.em.its records the last stm() call's target, as in the searchK run's chunked fit", sprintf("model %d; last chunk target %d", model_new$settings$convergence$max.em.its, tail(record$chunk_table$target_iterations, 1)), model_new$settings$convergence$max.em.its == tail(record$chunk_table$target_iterations, 1))
add_check("documents, vocabulary and word counts identical (N, V, K, A, wcounts, vocab)", sprintf("N %d V %d K %d A %d", model_new$settings$dim$N, model_new$settings$dim$V, model_new$settings$dim$K, model_new$settings$dim$A),
          identical(model_new$settings$dim, model_orig$settings$dim) && identical(model_new$vocab, model_orig$vocab))
add_check("prevalence design matrix rebuilt by stm identical to the stopped state's (dimensions, column names, every value; the s() bases included)", sprintf("%d x %d; identical %s; max abs value diff %s", nrow(X_n), ncol(X_n), x_identical, format(x_max_abs_diff)), x_identical)
add_check("every setting identical apart from the ceiling (convergence.max.em.its) and the recorded call", sprintf("%d leaves compared; %d differ outside the expected two; expected: %s", nrow(sd), sum(!sd$equal & !sd$ignored), paste(sprintf("%s %s -> %s", sd$leaf[sd$ignored & !sd$equal], sd$in_a[sd$ignored & !sd$equal], sd$in_b[sd$ignored & !sd$equal]), collapse = "; ")),
          sum(!sd$equal & !sd$ignored) == 0L)
add_check("theta rows sum to 1 with no NA and equal the softmax of eta; every aspect beta row sums to 1", sprintf("theta row-sum max dev %.2e; theta vs eta max diff %.2e; beta row-sum max dev %.2e", max(abs(rowSums(model_new$theta) - 1)), theta_from_eta_max_diff, max(vapply(model_new$beta$logbeta, function(lb) max(abs(rowSums(exp(lb)) - 1)), numeric(1)))),
          !anyNA(model_new$theta) && max(abs(rowSums(model_new$theta) - 1)) < 1e-8 && theta_from_eta_max_diff < 1e-8 && all(vapply(model_new$beta$logbeta, function(lb) max(abs(rowSums(exp(lb)) - 1)) < 1e-6, logical(1))) && length(model_new$beta$logbeta) == 2L)
add_check("mu, gamma, sigma, kappa and eta: dimensions unchanged, no NA, sigma symmetric positive definite", sprintf("gamma %d x %d; sigma %d x %d; kappa params %d", nrow(model_new$mu$gamma), ncol(model_new$mu$gamma), nrow(model_new$sigma), ncol(model_new$sigma), length(model_new$beta$kappa$params)),
          identical(dim(model_new$mu$gamma), dim(model_orig$mu$gamma)) && identical(dim(model_new$mu$mu), dim(model_orig$mu$mu)) && identical(dim(model_new$eta), dim(model_orig$eta)) && !anyNA(model_new$mu$gamma) && !anyNA(model_new$sigma) && !anyNA(model_new$eta) &&
            isSymmetric(model_new$sigma) && !inherits(try(chol(model_new$sigma), silent = TRUE), "try-error") && length(model_new$beta$kappa$params) == length(model_orig$beta$kappa$params) && !anyNA(unlist(model_new$beta$kappa$params)))
add_check("the bound increased from the starting iteration to the final one (as expected from EM); negative changes during the continuation counted", sprintf("bound %.3f -> %.3f (difference %.3f, relative %.3e); negative changes %d of %d", bound_orig[its0], bound_new[its_new], bound_new[its_new] - bound_orig[its0], (bound_new[its_new] - bound_orig[its0]) / abs(bound_orig[its0]), sum(cont_rel < 0), length(cont_rel)),
          bound_new[its_new] > bound_orig[its0])
add_check("L1 prevalence record: one call per continuation iteration, numbered after the searchK run's, pmax as before, information-criterion minimum interior in every call", sprintf("%d rows; calls %s-%s; pmax %s; interior in all: %s", nrow(gamma_cont), if (nrow(gamma_cont)) min(gamma_cont$call) else NA, if (nrow(gamma_cont)) max(gamma_cont$call) else NA, paste(unique(gamma_cont$pmax), collapse = ","), if (nrow(gamma_cont)) !any(gamma_cont$chosen_is_last_computed) else NA),
          nrow(gamma_cont) == its_new - its0 && nrow(gamma_cont) > 0 && min(gamma_cont$call) == its0 + 1L && max(gamma_cont$call) == its_new && all(gamma_cont$pmax == record$gamma_pmax) && !any(gamma_cont$chosen_is_last_computed))
add_check("continuation console: one E-step and one M-step timing per continuation iteration; the last stm() message matches the convergence flag", sprintf("%d E-step timings for %d iterations; message '%s'", sum(!is.na(log_cont$estep_seconds)), its_new - its0, fit_log_message(record$log_lines)),
          sum(!is.na(log_cont$estep_seconds)) == its_new - its0 && sum(!is.na(log_cont$mstep_seconds_reported)) == its_new - its0 && identical(fit_log_message(record$log_lines), if (model_new$convergence$converged) "Model Converged" else "Model Terminated Before Convergence Reached"))
add_check("merged fit record: chunks and totals over both runs; run key preserved", sprintf("chunks %d = %d + %d; wall %.0f = %.0f + %.0f", fit_record_merged$chunks, nrow(fit_record_orig$chunk_table), nrow(record$chunk_table), fit_record_merged$wall_seconds, fit_record_orig$wall_seconds, record$wall_seconds),
          fit_record_merged$chunks == nrow(fit_record_orig$chunk_table) + nrow(record$chunk_table) && isTRUE(all.equal(fit_record_merged$wall_seconds, fit_record_orig$wall_seconds + record$wall_seconds)) && identical(fit_record_merged$run_key, run_key) && isTRUE(fit_record_merged$continued) && fit_record_merged$gamma_calls == its_new)
add_check("files: the retained object is the continued model and the renamed object is the stopped state (hashes)", sprintf("%s %s; %s %s", basename(model_path), hash_model_now, basename(iter_model_path), hash_iter_now), identical(hash_model_now, record$continued_model_hash) && identical(hash_iter_now, record$started_from_hash) && identical(hash_iter_now, sidecar_orig_hash <- if (state == "pending") sidecar_orig$output_hash else read_yaml(paste0(iter_model_path, ".manifest.yml"))$output_hash))
add_check(sprintf("K = %s untouched: their retained objects still equal their sidecars", paste(other_K, collapse = ", ")), paste(sprintf("K = %d: %s", other_K, other_untouched), collapse = "; "), all(other_untouched))
add_check("memory sampler trace exists for the continuation (or the sampler was disabled)", sprintf("%d trace file(s)", length(record$trace_files)), !memory_sampler_on || length(record$trace_files) >= 1)
checks_tbl <- bind_rows(verify_tbl, bind_rows(checks))
log_line("checks:", sum(checks_tbl$pass, na.rm = TRUE), "of", nrow(checks_tbl), "pass;", sum(!checks_tbl$pass, na.rm = TRUE), "fail")

# ---- Tables ----
# Continuation summary, convergence trace, chunk table, L1 record, resource trace, checks, settings
# comparison and file inventory.
Kk <- continue_K
trace_tbl <- bind_rows(lapply(record$trace_files, function(p) read_csv(p, show_col_types = FALSE, progress = FALSE, col_types = cols(time_utc = col_character(), stage = col_character(), .default = col_double())) |> mutate(trace_file = basename(p))))
trace_fit <- if (nrow(trace_tbl)) trace_tbl |> filter(stage %in% "fit") else trace_tbl
timing_orig <- log_orig |> mutate(iteration = iteration) |> filter(iteration <= its0)
timing_cont <- log_cont |> mutate(iteration = its0 + iteration)
timing_all  <- bind_rows(timing_orig, timing_cont)
bound_vec <- bound_new
convergence_trace <- tibble(K = Kk, iteration = seq_along(bound_vec), phase = if_else(seq_along(bound_vec) <= its0, "searchK run", "continuation"), bound = bound_vec, bound_per_token = bound_vec / n_tokens_fit, relative_change = rel_change) |>
  left_join(timing_all, by = "iteration") |>
  mutate(iteration_seconds = coalesce(estep_seconds, 0) + coalesce(mstep_seconds_reported, 0), cumulative_seconds = cumsum(iteration_seconds), below_emtol = !is.na(relative_change) & relative_change < emtol & relative_change > 0)
fit_chunks_tbl <- fit_record_merged$chunk_table |> mutate(K = Kk) |> select(K, everything())
gamma_cont_tbl <- gamma_cont |> mutate(K = Kk, phase = "continuation") |> select(K, phase, everything())
cont_iters <- its_new - its0
last_change_positive <- tail(rel_change, 1) > 0
fmt <- function(x, d = 3) format(x, digits = d, nsmall = 0, scientific = FALSE, trim = TRUE)
summary_tbl <- tibble(
  item = c("K", "starting iteration (the stopped state)", "final iteration", "additional EM iterations", "convergence reached", "convergence rule", "final relative change of the bound", "final change positive",
           "relative change at the starting iteration", "bound at the starting iteration", "final bound", "bound difference (final minus starting)", "relative change from the starting bound to the final one",
           "final bound differs from the starting bound in the expected direction (increase)", "negative bound changes during the continuation", "final bound per token", "corrected lower bound (bound + lfactorial(K))",
           "ceiling in force (total iterations)", "iterations per stm() call (chunk)", "stm() calls in the continuation", "additional wall time: fitting (sum of the stm() calls, design rebuild included; s)",
           "fitting wall seconds by continuation run", "additional wall time: first continuation run start to the model saved (s; includes any gap between interrupted runs)", "wall time of this script run (s; this run only, loading and reporting included)",
           "additional CPU time: user (s; proc.time over the stm() calls)", "additional CPU time: system (s)", "additional CPU time: user, sampler cross-check (s; summed over the continuation runs' traces)",
           "E-step seconds per iteration (mean; min-max)", "M-step seconds per iteration (mean; min-max)", "seconds per iteration (mean)", "restart overhead per stm() call (wall minus E- and M-step seconds; mean; s)",
           "peak memory: process peak working set after the fit (MB)", "peak memory: sampler maximum working set during the fit (MB)", "typical memory: sampler median working set during the fit (MB)", "typical memory: sampler median resident set during the fit (MB)", "system memory available, minimum during the fit (MB)", "sampler rows",
           "saved K = 10 model state reused rather than restarted", "restart mechanism", "settings changed", "diagnostics recomputed", "L1 prevalence calls in the continuation", "L1 path: information-criterion minimum interior in every call", "L1 path length (min-max); chosen index (min-max)", "glmnet pmax",
           "nonzero L1 coefficient rows at the last call (searchK run's last call)", "interrupted and resumed", "continuation run id(s)", "searchK fitting run id", "stopped state file", "stopped state hash", "retained object (continued model) file", "retained object hash", "retained object bytes",
           "searchK script hash equals the sidecar's record", "searchK script hash", "continuation script hash", "quick run", "R / stm / glmnet"),
  value = c(Kk, its0, its_new, cont_iters, model_new$convergence$converged, sprintf("stm: relative change (new - old) / |old| of the approximate bound below emtol = %s (allow.neg.change = TRUE, the package default: a negative change also stops the fit)", format(emtol)),
            format(tail(rel_change, 1), digits = 4), last_change_positive, format(rel_change[its0], digits = 4), fmt(bound_orig[its0], 12), fmt(bound_new[its_new], 12), fmt(bound_new[its_new] - bound_orig[its0], 8), format((bound_new[its_new] - bound_orig[its0]) / abs(bound_orig[its0]), digits = 4),
            bound_new[its_new] > bound_orig[its0], sum(cont_rel < 0), fmt(bound_new[its_new] / n_tokens_fit, 8), fmt(bound_new[its_new] + lfactorial(Kk), 12),
            record$ceiling, record$fit_chunk, nrow(record$chunk_table), fmt(record$wall_seconds, 6),
            paste(sprintf("%s: %.0f", names(tapply(record$chunk_table$wall_seconds, record$chunk_table$run_id, sum)), tapply(record$chunk_table$wall_seconds, record$chunk_table$run_id, sum)), collapse = "; "),
            fmt(record$wall_seconds_including_saves, 6), fmt(secs_since(run_start), 6), fmt(record$cpu_user_seconds, 6), fmt(record$cpu_system_seconds, 6),
            if (nrow(trace_tbl)) fmt(sum(tapply(trace_tbl$cpu_user_s, trace_tbl$trace_file, function(x) max(x) - min(x))), 6) else NA,
            sprintf("%.0f; %d-%d", mean(timing_cont$estep_seconds, na.rm = TRUE), min(timing_cont$estep_seconds, na.rm = TRUE), max(timing_cont$estep_seconds, na.rm = TRUE)),
            sprintf("%.0f; %d-%d", mean(timing_cont$mstep_seconds_reported, na.rm = TRUE), min(timing_cont$mstep_seconds_reported, na.rm = TRUE), max(timing_cont$mstep_seconds_reported, na.rm = TRUE)),
            fmt(record$wall_seconds / cont_iters, 5), fmt((record$wall_seconds - sum(timing_cont$estep_seconds, na.rm = TRUE) - sum(timing_cont$mstep_seconds_reported, na.rm = TRUE)) / nrow(record$chunk_table), 4),
            round(record$peak_wset_mb_after_fit), if (nrow(trace_fit)) round(max(trace_fit$wset_mb)) else NA, if (nrow(trace_fit)) round(median(trace_fit$wset_mb)) else NA, if (nrow(trace_fit)) round(median(trace_fit$rss_mb)) else NA, if (nrow(trace_fit)) round(min(trace_fit$system_avail_mb)) else NA, nrow(trace_tbl),
            checks_tbl$pass[grepl("^saved state reused", checks_tbl$check)], "stm::stm(..., model = <saved model>, max.em.its = <target>): mu/gamma, sigma, beta (exp(logbeta)), kappa, eta and the bound trace carried over, iteration counter incremented; no spectral initialisation; the design matrix rebuilt from the same formula and data (checked identical)",
            if (sum(!sd$equal & !sd$ignored) == 0L) "none besides convergence$max.em.its (the ceiling of the last stm() call) and the recorded call; every other setting identical (settings_comparison table)" else sprintf("%d unexpected setting differences (see the settings comparison table)", sum(!sd$equal & !sd$ignored)),
            "none: held-out likelihood, residual dispersion, semantic coherence, labels and estimateEffect were not recomputed; the searchK package's K = 10 diagnostic values remain those of the stopped state",
            nrow(gamma_cont), if (nrow(gamma_cont)) !any(gamma_cont$chosen_is_last_computed) else NA, if (nrow(gamma_cont)) sprintf("%d-%d; %d-%d", min(gamma_cont$path_length), max(gamma_cont$path_length), min(gamma_cont$chosen_index), max(gamma_cont$chosen_index)) else NA, record$gamma_pmax,
            sprintf("%s (%s)", if (nrow(gamma_cont)) tail(gamma_cont$nonzero_rows_chosen, 1) else NA, tail(gamma_orig$nonzero_rows_chosen, 1)), length(record$resumed_runs) > 0, paste(c(record$resumed_runs, record$run_id), collapse = ", "), fit_record_orig$run_id,
            iter_model_path, record$started_from_hash, model_path, record$continued_model_hash, record$model_file_bytes, record$main_script_hash_equals_sidecar, record$main_script_hash, record$continuation_script_hash, quick_run,
            sprintf("%s / %s / %s", R.version.string, record$stm_version, record$glmnet_version))) |>
  mutate(value = as.character(value))
model_files_tbl <- bind_rows(lapply(c(model_path, iter_model_path, fit_record_path, fit_record_iter_path, record_path, fit_log, gamma_cont_csv, record$trace_files, status_file), function(p) tibble(file = file.path("models", basename(p)), bytes = file.size(p), sha256 = sha256(p)))) |>
  mutate(status = case_when(grepl("\\.rds$", file) ~ "git-ignored (retained object; manifest tracked)", TRUE ~ "git-ignored (console, record, trace, status)"),
         role = case_when(file == file.path("models", basename(model_path)) ~ "the retained K = 10 model: the continued model", file == file.path("models", basename(iter_model_path)) ~ "the stopped state the continuation started from (renamed)",
                          file == file.path("models", basename(fit_record_path)) ~ "the merged fit record (searchK run + continuation)", file == file.path("models", basename(fit_record_iter_path)) ~ "the searchK run's fit record (renamed)",
                          file == file.path("models", basename(record_path)) ~ "the continuation record", file == file.path("models", basename(fit_log)) ~ "stm console of the continuation", file == file.path("models", basename(gamma_cont_csv)) ~ "L1 prevalence calls of the continuation",
                          grepl("resource_trace", file) ~ "memory sampler trace", TRUE ~ "status file"))
table_dims <- list()
write_table <- function(tbl, name) { path <- file.path(tab_dir, name); write_csv(tbl, path); table_dims[[path]] <<- c(rows = nrow(tbl), cols = ncol(tbl)); invisible(path) }
write_table(summary_tbl,       sprintf("continuation_%s_summary.csv", tag))
write_table(convergence_trace, sprintf("continuation_%s_convergence_trace.csv", tag))
write_table(fit_chunks_tbl,    sprintf("continuation_%s_fit_chunks.csv", tag))
write_table(gamma_cont_tbl,    sprintf("continuation_%s_gamma_path.csv", tag))
if (nrow(trace_tbl)) write_table(trace_tbl, sprintf("continuation_%s_resource_trace.csv", tag))
write_table(checks_tbl,        sprintf("continuation_%s_checks.csv", tag))
write_table(sd |> mutate(K = Kk, expected_difference = ignored) |> select(K, leaf, stopped_state = in_a, continued_model = in_b, equal, expected_difference), sprintf("continuation_%s_settings_comparison.csv", tag))
write_table(model_files_tbl,   sprintf("continuation_%s_files.csv", tag))

# ---- Provenance sidecars ----
# One <output>.manifest.yml per output and for the replaced model files; inputs include the stopped model,
# the common input, both scripts and the searchK run's own inputs.
set_stage("report")
git_commit <- tryCatch(system2("git", c("-C", project_dir, "rev-parse", "--short", "HEAD"), stdout = TRUE, stderr = FALSE), error = function(e) "")
if (length(git_commit) == 0L) git_commit <- ""
manifest_packages <- c("stm", "lda", "glmnet", "matrixStats", "Matrix", "ps", "dplyr", "tibble", "readr", "digest", "yaml")
package_versions  <- lapply(manifest_packages, function(p) list(name = p, version = as.character(packageVersion(p))))
inputs_model <- c(
  list(list(path = iter_model_path, hash = record$started_from_hash, format = "rds", rows = as.integer(model_orig$settings$dim$N), cols = as.integer(continue_K), role = sprintf("the K = %d model as the searchK run stopped it (iteration %d): the state continued through stm(model = )", continue_K, its0)),
       list(path = common_input_path, hash = sha256(common_input_path), format = "rds", role = "the searchK run's worker input: fitted documents and vocabulary (common held-out construction), covariate table, formula strings, settings"),
       list(path = fit_record_iter_path, hash = sha256(fit_record_iter_path), format = "rds", role = "the searchK run's fit record for K = 10 (merged into the new fit record)"),
       list(path = gamma_orig_csv, hash = sha256(gamma_orig_csv), format = "csv", role = "the searchK run's L1 prevalence call record for K = 10 (the continuation's calls are numbered after it)"),
       list(path = main_script, hash = record$main_script_hash, format = "other", role = "the searchK script: the opt.mu replacement (pmax cap, per-call record), fit-log parser and memory sampler are extracted from it")),
  lapply(inherited_input_files, function(f) { f$role <- paste0(if (is.null(f$role)) "" else paste0(f$role, "; "), "inherited: an input of the searchK run"); f }))
inputs_tables <- c(list(list(path = model_path, hash = record$continued_model_hash, format = "rds", rows = as.integer(model_new$settings$dim$N), cols = as.integer(continue_K), role = "the continued K = 10 model (the retained object)"),
                        list(path = record_path, hash = sha256(record_path), format = "rds", role = "the continuation record")), inputs_model)
manifest_parameters <- list(
  model = sprintf("the K = %d STM of the searchK evidence run (stm::stm variational EM; prevalence %s, gamma.prior L1; content %s, kappa.prior L1; spectral initialisation of the original fit) continued from its saved iteration-%d state through stm(model = ) to iteration %d (%s)",
                  continue_K, ci$prevalence_string, ci$content_string, its0, its_new, if (model_new$convergence$converged) "converged" else "ceiling reached, not converged"),
  continued_from_iteration = its0, iterations_after_continuation = its_new, additional_iterations = cont_iters, converged = model_new$convergence$converged, final_relative_change = format(tail(rel_change, 1), digits = 6),
  ceiling_total_iterations = record$ceiling, fit_chunk = record$fit_chunk, emtol = format(emtol, scientific = TRUE), gamma_pmax = record$gamma_pmax, K = continue_K, prevalence_formula = ci$prevalence_string, content_formula = ci$content_string,
  gamma_prior = ci$settings$gamma_prior, kappa_prior = ci$settings$kappa_prior, kappa_interactions = as.character(ci$settings$interactions), sigma_prior = as.character(ci$settings$sigma_prior), init_type_of_original_fit = ci$settings$init_type,
  stm_seed = as.character(ci$settings$stm_seed), ngroups = "1", documents = as.character(model_new$settings$dim$N), vocabulary = as.character(model_new$settings$dim$V), tokens_fitted = as.character(n_tokens_fit),
  restart = "stm::stm(documents, vocab, K, prevalence, content, data, init.type, seed, max.em.its = target, emtol, verbose, reportevery, interactions, gamma.prior, sigma.prior, kappa.prior, model = <saved model>): mu/gamma, sigma, beta (exp(logbeta)), kappa, eta and the bound trace carried over, iteration counter incremented; no re-initialisation; the design matrix rebuilt from the same formula and data and checked identical",
  settings_preserved = "documents, vocabulary, prevalence and content specification, L1 prevalence regularisation with the same glmnet pmax cap (the searchK script's opt.mu replacement), spline bases (rebuilt identically), initialisation history (bound trace, parameters), convergence tolerance, seed; only convergence$max.em.its (the ceiling) and the recorded call differ",
  diagnostics_recomputed = "none (held-out likelihood, residual dispersion, semantic coherence, labels, estimateEffect not rerun)",
  searchk_script_hash = record$main_script_hash, searchk_script_hash_equals_sidecar = as.character(record$main_script_hash_equals_sidecar), searchk_fitting_run_id = fit_record_orig$run_id, continuation_run_id = record$run_id,
  resumed_from_interrupted_runs = if (length(record$resumed_runs)) paste(record$resumed_runs, collapse = ", ") else "none", quick_run = as.character(quick_run), memory_sampler = as.character(memory_sampler_on), sampler_interval_seconds = as.character(sampler_interval))
rds_dims <- list(); rds_dims[[basename(model_path)]] <- c(rows = model_new$settings$dim$N, cols = continue_K); rds_dims[[basename(fit_record_path)]] <- c(rows = length(fit_record_merged$log_lines), cols = length(fit_record_merged)); rds_dims[[basename(record_path)]] <- c(rows = nrow(record$chunk_table), cols = length(record))
write_sidecar <- function(output, inputs) {
  ext <- tolower(tools::file_ext(output))
  manifest <- list(manifest_version = 1L, output_file = output, output_hash = sha256(output), output_format = switch(ext, csv = "csv", rds = "rds", ext))
  if (ext == "csv") { dims <- table_dims[[output]]; manifest$output_rows <- unname(dims[["rows"]]); manifest$output_cols <- unname(dims[["cols"]]) }
  if (ext == "rds") { dims <- rds_dims[[basename(output)]]; if (!is.null(dims)) { manifest$output_rows <- unname(as.integer(dims[["rows"]])); manifest$output_cols <- unname(as.integer(dims[["cols"]])) } }
  manifest$input_files    <- inputs
  manifest$transformation <- list(script = script_path, script_hash = sha256(script_path), parameters = manifest_parameters, git_commit = git_commit)
  manifest$software <- list(language = "R", language_version = paste(R.version$major, R.version$minor, sep = "."), packages = package_versions, os = paste(Sys.info()[["sysname"]], Sys.info()[["release"]]))
  manifest$seed      <- as.integer(ci$settings$stm_seed)
  manifest$timestamp <- utc_stamp()
  manifest$notes     <- if (ext == "rds") sprintf("Retained object after the continuation of the K = %d STM of the searchK evidence run (July 2026 r/politics, Tier 2 text representation): the model continued from its saved iteration-%d state through stm(model = ) to iteration %d (%s), the merged fit record, or the continuation record. The stopped state is kept as %s with its original sidecar re-pointed. No diagnostic was recomputed. The searchK script's K = 10 diagnostics (held-out likelihood, residual dispersion, coherence) in ../stm_searchk_k10_k20_k30/tables/ are those of the stopped state.",
                                                  continue_K, its0, its_new, if (model_new$convergence$converged) "converged" else "not converged at the ceiling", basename(iter_model_path)) else
    sprintf("Continuation record of the K = %d STM of the searchK evidence run (July 2026 r/politics, Tier 2 text representation): the saved iteration-%d state continued through stm(model = ) under the searchK run's settings to iteration %d (%s); every setting, the documents, vocabulary and design matrix checked identical; no diagnostic recomputed.",
            continue_K, its0, its_new, if (model_new$convergence$converged) "converged" else "not converged at the ceiling")
  write_yaml(manifest, paste0(output, ".manifest.yml"))
}
invisible(lapply(list.files(tab_dir, pattern = "\\.csv$", full.names = TRUE), write_sidecar, inputs = inputs_tables))
write_sidecar(model_path, inputs_model)
write_sidecar(fit_record_path, inputs_model)
write_sidecar(record_path, inputs_model)
set_stage("done")

# ---- Console summary ----
print(as.data.frame(summary_tbl), right = FALSE)
print(as.data.frame(checks_tbl |> select(stage, check, pass)), right = FALSE)
print(as.data.frame(convergence_trace |> filter(phase == "continuation") |> select(iteration, bound, relative_change, estep_seconds, mstep_seconds_reported)), right = FALSE)
cat("\nDone. Tables ->", tab_dir, "\nModels ->", model_dir, "\n")
if (any(!checks_tbl$pass, na.rm = TRUE)) { cat("FAILED CHECKS:\n"); print(as.data.frame(checks_tbl |> filter(!is.na(pass), !pass)), right = FALSE); quit(save = "no", status = 2) }
