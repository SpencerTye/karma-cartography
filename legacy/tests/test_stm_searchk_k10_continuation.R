# Tests for the pure functions of july2026_politics_stm_searchk_k10_continue.R.
# Run: Rscript tests/test_stm_searchk_k10_continuation.R
# The functions are extracted from the script by parsing it, so the script's pipeline does not run.
suppressPackageStartupMessages({ library(dplyr); library(tibble) })
script <- file.path(Sys.getenv("KC_REPO_ROOT", unset = getwd()), "legacy/scripts/july2026_politics_stm_searchk_k10_continue.R")
stopifnot("script file exists" = file.exists(script))
exprs <- parse(script, keep.source = FALSE)
def_of <- function(name) {
  is_def <- vapply(exprs, function(e) is.call(e) && identical(e[[1]], as.name("<-")) && identical(e[[2]], as.name(name)), logical(1))
  stopifnot(sum(is_def) == 1L)
  eval(exprs[[which(is_def)]], envir = globalenv())
}
for (nm in c("chunk_targets", "flatten_settings", "leaf_display", "leaf_equal", "settings_diff", "merge_fit_record", "continuation_state", "annotate_manifest")) def_of(nm)

# ---- chunk_targets: the planned stm() call targets from the current iteration to the ceiling ----
stopifnot(
  "regular chunks up to the ceiling" = identical(chunk_targets(25L, 125L, 5L), as.integer(seq(30L, 125L, by = 5L))),
  "last chunk clipped at the ceiling" = identical(chunk_targets(25L, 33L, 5L), c(30L, 33L)),
  "one short chunk when the ceiling is near" = identical(chunk_targets(23L, 25L, 5L), 25L),
  "nothing to do at or above the ceiling" = identical(chunk_targets(25L, 25L, 5L), integer(0)) && identical(chunk_targets(30L, 25L, 5L), integer(0)),
  "a chunk below one behaves as one iteration per call" = identical(chunk_targets(25L, 27L, 0L), c(26L, 27L)),
  "integer result" = is.integer(chunk_targets(2L, 4L, 1L))
)

# ---- settings_diff: every leaf of two nested settings lists, with expected differences flagged ----
a <- list(dim = list(K = 10, V = 5, wcounts = list(x = 1:3)), convergence = list(max.em.its = 25, em.converge.thresh = 1e-5, allow.neg.change = TRUE),
          gamma = list(mode = "L1", prior = NULL, enet = 1), covariates = list(formula = ~ x + s(z), yvarlevels = c("comment", "submission")), call = quote(f(x)))
b <- a; b$convergence$max.em.its <- 70; b$call <- quote(f(y))
environment(b$covariates$formula) <- new.env()   # same text, a different environment: the same specification
d <- settings_diff(a, b, ignore = c("convergence.max.em.its", "call"))
stopifnot(
  "one row per leaf with the expected columns" = is.data.frame(d) && all(c("leaf", "in_a", "in_b", "equal", "ignored") %in% names(d)),
  "leaves are dotted paths" = all(c("dim.K", "convergence.max.em.its", "gamma.mode", "covariates.formula", "dim.wcounts.x") %in% d$leaf),
  "ignored differences are flagged, not counted" = all(d$equal[!d$ignored]) && sum(!d$equal) == 2L && all(d$leaf[!d$equal] %in% c("convergence.max.em.its", "call")),
  "formulas compare by their text" = d$equal[d$leaf == "covariates.formula"],
  "NULL leaves survive and compare equal" = "gamma.prior" %in% d$leaf && d$equal[d$leaf == "gamma.prior"],
  "display values are character" = is.character(d$in_a) && is.character(d$in_b)
)
b2 <- a; b2$convergence$em.converge.thresh <- 1e-4
d2 <- settings_diff(a, b2, ignore = c("convergence.max.em.its", "call"))
stopifnot("a changed tolerance is a counted difference" = sum(!d2$equal & !d2$ignored) == 1L && d2$leaf[!d2$equal] == "convergence.em.converge.thresh")
a3 <- a; a3$init <- list(mode = "Spectral", maxV = 10000)
d3 <- settings_diff(a3, a, ignore = character(0))
stopifnot("a leaf absent on one side is a counted difference shown as absent" = sum(!d3$equal) == 2L && all(d3$in_b[!d3$equal] == "<absent>"))

# ---- merge_fit_record: the searchK fit record plus a continuation record, totals recomputed ----
orig <- list(K = 10L, run_id = "R1", run_key = "KEY", fit_start_utc = "2026-09-09T10:27:20Z", fit_end_utc = "2026-09-09T13:38:44Z", wall_seconds = 11474.56, wall_seconds_including_saves = 11483.78,
             cpu_user_seconds = 11150.44, cpu_system_seconds = 319.28, model_time_seconds = 10856.59, chunks = 5L, fit_chunk = 5L,
             chunk_table = tibble(chunk = 1:5, target_iterations = seq(5L, 25L, 5L), iterations_after = seq(5L, 25L, 5L), converged = FALSE, wall_seconds = rep(2000, 5), cpu_user_seconds = rep(1900, 5),
                                  cpu_system_seconds = rep(60, 5), stm_time_seconds = rep(1950, 5), peak_wset_mb = 6852.6, run_id = "R1"),
             peak_wset_mb_after_fit = 6852.6, rss_mb_after_fit = 4502.3, log_lines = c("l1", "l2"), trace_csv = "trace_R1.csv", stm_version = "1.3.8", glmnet_version = "4.1.10",
             model_file_bytes = 444640823, pid = 67092L, gamma_pmax = 50000L, gamma_calls = 25L)
cont <- list(run_id = "R2", ceiling = 125L, fit_chunk = 5L, its_from = 25L, its_to = 37L, converged = TRUE, fit_start_utc = "2026-09-10T00:00:00Z", fit_end_utc = "2026-09-10T01:30:00Z",
             wall_seconds = 5000, wall_seconds_including_saves = 5010, cpu_user_seconds = 4900, cpu_system_seconds = 100, model_time_seconds = 4950,
             chunk_table = tibble(chunk = 1:3, target_iterations = c(30L, 35L, 40L), iterations_after = c(30L, 35L, 37L), converged = c(FALSE, FALSE, TRUE), wall_seconds = c(1700, 1700, 1600), cpu_user_seconds = c(1650, 1650, 1600),
                                  cpu_system_seconds = c(30, 30, 40), stm_time_seconds = c(1680, 1680, 1590), peak_wset_mb = 7000, run_id = "R2"),
             peak_wset_mb_after_fit = 7000, rss_mb_after_fit = 4600, log_lines = c("c1", "c2", "c3"), trace_csv = "trace_R2.csv", model_file_bytes = 444700000, pid = 123L, gamma_calls = 12L)
m <- merge_fit_record(orig, cont)
stopifnot(
  "identity and key fields preserved" = identical(m$K, orig$K) && identical(m$run_id, orig$run_id) && identical(m$run_key, orig$run_key) && identical(m$fit_start_utc, orig$fit_start_utc) &&
    identical(m$fit_chunk, orig$fit_chunk) && identical(m$gamma_pmax, orig$gamma_pmax) && identical(m$stm_version, orig$stm_version) && identical(m$trace_csv, orig$trace_csv),
  "totals recomputed over both runs" = isTRUE(all.equal(m$wall_seconds, 16474.56)) && isTRUE(all.equal(m$wall_seconds_including_saves, 16493.78)) && isTRUE(all.equal(m$cpu_user_seconds, 16050.44)) &&
    isTRUE(all.equal(m$cpu_system_seconds, 419.28)) && isTRUE(all.equal(m$model_time_seconds, 15806.59)) && m$gamma_calls == 37L && m$chunks == 8L,
  "end time, memory, file size and pid come from the continuation; peak is the larger" = identical(m$fit_end_utc, cont$fit_end_utc) && m$peak_wset_mb_after_fit == 7000 && m$rss_mb_after_fit == 4600 &&
    m$model_file_bytes == cont$model_file_bytes && m$pid == cont$pid,
  "chunks renumbered consecutively with a phase column" = nrow(m$chunk_table) == 8L && identical(m$chunk_table$chunk, 1:8) && identical(m$chunk_table$phase, c(rep("searchK run", 5), rep("continuation", 3))) &&
    identical(m$chunk_table$target_iterations, c(seq(5L, 25L, 5L), 30L, 35L, 40L)),
  "console lines concatenated in order" = identical(m$log_lines, c("l1", "l2", "c1", "c2", "c3")),
  "the original record and the continuation record are kept whole" = identical(m$original_run, orig) && identical(m$continuation, cont),
  "the merged record says it was continued" = isTRUE(m$continued) && m$continued_from_iteration == 25L && m$iterations_after_continuation == 37L && isTRUE(m$converged)
)

# ---- continuation_state: what the retained object currently is ----
h25 <- "sha256:aaa"; hc <- "sha256:bbb"; rec <- list(started_from_hash = h25, continued_model_hash = hc)
stopifnot(
  "no record and the retained object matches its sidecar: pending" = continuation_state(h25, h25, NULL) == "pending",
  "record present and the retained object is the continued model: applied" = continuation_state(hc, hc, rec) == "applied",
  "record present but the retained object is something else: unknown" = continuation_state(h25, h25, rec) == "unknown" && continuation_state("sha256:ccc", hc, rec) == "unknown",
  "no record and the retained object does not match its sidecar: unknown" = continuation_state(hc, h25, NULL) == "unknown"
)

# ---- annotate_manifest: the old sidecar re-pointed at the renamed object with a note appended ----
old <- list(manifest_version = 1L, output_file = "S:/x/models/stm_K010.rds", output_hash = h25, output_format = "rds", transformation = list(script = "s.R"), timestamp = "2026-09-09T18:31:02Z", notes = "old note")
ann <- annotate_manifest(old, "S:/x/models/stm_K010_iter25_stopped.rds", "superseded")
stopifnot(
  "output_file replaced" = identical(ann$output_file, "S:/x/models/stm_K010_iter25_stopped.rds"),
  "note appended to the existing note" = grepl("^old note", ann$notes) && grepl("superseded", ann$notes),
  "every other field and the field order unchanged" = identical(names(ann), names(old)) && identical(ann[setdiff(names(old), c("output_file", "notes"))], old[setdiff(names(old), c("output_file", "notes"))]),
  "a manifest without notes gets the note" = identical(annotate_manifest(old[names(old) != "notes"], "y.rds", "n")$notes, "n")
)
cat("test_stm_searchk_k10_continuation: all checks passed\n")
