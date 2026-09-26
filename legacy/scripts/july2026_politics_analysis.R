# July 2026 r/politics sample — descriptive analysis
#
# Reads the raw Parquet comment and submission files, reproduces the prior
# headline descriptive measurements, and builds exploratory figures and tables.
# Source Parquet is read-only. All derived outputs are written under
# r_analysis_outputs/ and are reproducible by re-running this script.

# ---- Setup: packages, paths, output directories ----
# INPUT : none.
# DOES  : load packages; define source paths; create output folders.
# OUTPUT: path variables; empty figures/ and tables/ directories.
library(nanoparquet)
library(dplyr)
library(tidyr)
library(lubridate)
library(stringr)
library(ggplot2)
library(scales)
library(readr)
library(digest)
library(yaml)

project_dir     <- Sys.getenv("KC_REPO_ROOT", unset = getwd())
comments_dir    <- file.path(project_dir, "data_sample/comments/2026-07")
submissions_dir <- file.path(project_dir, "data_sample/submissions/2026-07")
script_path     <- file.path(project_dir, "july2026_politics_analysis.R")

out_dir <- file.path(project_dir, "r_analysis_outputs")
fig_dir <- file.path(out_dir, "figures")
tab_dir <- file.path(out_dir, "tables")
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(tab_dir, recursive = TRUE, showWarnings = FALSE)

comment_files    <- list.files(comments_dir, pattern = "\\.parquet$", full.names = TRUE)
submission_files <- list.files(submissions_dir, pattern = "\\.parquet$", full.names = TRUE)

# ---- Read source data ----
# INPUT : 7 comment Parquet files + 1 submission Parquet file.
# DOES  : read each file and stack; then keep only the columns used below
#         (this is column selection only — no rows are dropped).
# OUTPUT: comments, submissions data frames.
comments <- lapply(comment_files, read_parquet) |>
  bind_rows() |>
  select(id, link_id, body, created_utc, subreddit, subreddit_id)

submissions <- lapply(submission_files, read_parquet) |>
  bind_rows() |>
  select(id, created_utc, subreddit, subreddit_id)

# ---- Derived variables (time + thread key) ----
# INPUT : comments, submissions (created_utc as epoch seconds; comment link_id 't3_...').
# DOES  : cast created_utc to double; derive UTC datetime, UTC calendar date, UTC hour;
#         derive thread = link_id with the 't3_' prefix removed.
# OUTPUT: comments, submissions with added columns.
comments <- comments |>
  mutate(
    created_utc = as.numeric(created_utc),
    created_dt  = as_datetime(created_utc, tz = "UTC"),
    date        = as_date(created_dt),
    hour        = hour(created_dt),
    thread      = str_remove(link_id, "^t3_")
  )

submissions <- submissions |>
  mutate(created_utc = as.numeric(created_utc))

# ---- Measurement: totals and single-community check ----
# INPUT : comments, submissions.
# DOES  : count rows; list the community present in each record type.
# OUTPUT: n_comments, n_submissions; community_comments, community_submissions.
n_comments    <- nrow(comments)
n_submissions <- nrow(submissions)
community_comments    <- comments |> count(subreddit, subreddit_id, name = "rows")
community_submissions <- submissions |> count(subreddit, subreddit_id, name = "rows")

# ---- Measurement: exact comment-body frequency (byte-identical) ----
# INPUT : comments (body, thread).
# DOES  : group by exact body value (no trimming, case-folding, or Unicode
#         normalization); count occurrences and distinct threads per value.
# OUTPUT: body_freq (one row per distinct body value).
body_freq <- comments |>
  group_by(body) |>
  summarise(occ = n(), threads = n_distinct(thread), .groups = "drop")

n_distinct_bodies <- nrow(body_freq)
n_bodies_once     <- sum(body_freq$occ == 1)
n_bodies_repeated <- sum(body_freq$occ >= 2)
rows_once     <- sum(body_freq$occ[body_freq$occ == 1])
rows_repeated <- sum(body_freq$occ[body_freq$occ >= 2])

# ---- Measurement: occurrence-frequency distribution ----
# INPUT : body_freq.
# DOES  : bucket distinct body values by occurrence count into fixed bands; also
#         count distinct bodies at each exact occurrence value (for the figure).
# OUTPUT: occ_distribution (banded); occ_ff (frequency-of-frequencies).
occ_breaks <- c(0, 1, 2, 5, 10, 100, 1000, Inf)
occ_labels <- c("1", "2", "3-5", "6-10", "11-100", "101-1000", "1001+")
occ_distribution <- body_freq |>
  mutate(band = cut(occ, breaks = occ_breaks, labels = occ_labels, right = TRUE)) |>
  group_by(band) |>
  summarise(distinct_texts = n(), rows = sum(occ), .groups = "drop") |>
  mutate(pct_rows = round(100 * rows / n_comments, 3))

occ_ff <- body_freq |> count(occ, name = "n_texts")

# ---- Measurement: within-thread vs across-thread recurrence (repeated bodies) ----
# INPUT : body_freq restricted to repeated values (occ >= 2).
#         Population for this section: comment bodies that occur 2+ times.
# DOES  : split repeated values into single-thread vs multi-thread; band by the
#         number of distinct threads a value spans.
# OUTPUT: repeated_freq; within/across counts; thread_distribution (banded).
repeated_freq      <- body_freq |> filter(occ >= 2)
n_within_single    <- sum(repeated_freq$threads == 1)
n_across_multi     <- sum(repeated_freq$threads >= 2)
rows_within_single <- sum(repeated_freq$occ[repeated_freq$threads == 1])
rows_across_multi  <- sum(repeated_freq$occ[repeated_freq$threads >= 2])

thread_breaks <- c(0, 1, 5, 10, 100, 1000, Inf)
thread_labels <- c("1", "2-5", "6-10", "11-100", "101-1000", "1001+")
thread_distribution <- repeated_freq |>
  mutate(band = cut(threads, breaks = thread_breaks, labels = thread_labels, right = TRUE)) |>
  group_by(band) |>
  summarise(text_values = n(), rows = sum(occ),
            avg_occ = round(mean(occ), 1), avg_threads = round(mean(threads), 1),
            .groups = "drop")

# ---- Measurement: concentration among repeated texts ----
# INPUT : repeated_freq ordered by occurrences (descending).
# DOES  : compute the cumulative share of repeated rows across ranked texts
#         (Lorenz-style); summarise the share held by the top-k texts.
# OUTPUT: concentration; concentration_curve (with origin); concentration_summary.
concentration <- repeated_freq |>
  arrange(desc(occ)) |>
  mutate(rank = row_number(),
         cum_share_texts = rank / n(),
         cum_share_rows  = cumsum(occ) / sum(occ))
concentration_summary <- tibble(
  top_n = c(4, 10, 100, 1000),
  share_of_repeated_rows = round(concentration$cum_share_rows[top_n], 4)
)
# prepend the origin (0, 0) so the Lorenz-style curve starts before any text is counted
concentration_curve <- bind_rows(
  tibble(rank = 0, cum_share_texts = 0, cum_share_rows = 0),
  concentration
)

# ---- Measurement: activity by date and by hour (UTC) ----
# INPUT : comments (date, hour).
# DOES  : count comments per UTC calendar day and per UTC clock hour.
# OUTPUT: activity_by_date; activity_by_hour.
activity_by_date <- comments |> count(date, name = "comments")
activity_by_hour <- comments |> count(hour, name = "comments")

# ---- Measurement: comments per thread ----
# INPUT : comments (thread).
# DOES  : count comments per distinct thread; summarise the distribution.
# OUTPUT: per_thread_counts; per_thread_summary.
per_thread_counts  <- comments |> count(thread, name = "comments")
per_thread_summary <- tibble(
  n_threads = nrow(per_thread_counts),
  min  = min(per_thread_counts$comments),
  p50  = quantile(per_thread_counts$comments, 0.50, type = 7, names = FALSE),
  mean = round(mean(per_thread_counts$comments), 1),
  p90  = quantile(per_thread_counts$comments, 0.90, type = 7, names = FALSE),
  p99  = quantile(per_thread_counts$comments, 0.99, type = 7, names = FALSE),
  max  = max(per_thread_counts$comments)
)

# ---- Join: comments to their submission; comment-to-submission timing ----
# INPUT : comments (thread, created_utc); submissions (id, created_utc).
# DOES  : count referenced vs matched threads; inner-join comment.thread =
#         submission.id (matched threads only); compute delta = comment minus
#         submission creation time in seconds.
# OUTPUT: thread-match counts; matched (delta_sec); delta_summary; delta_buckets.
n_threads_referenced <- n_distinct(comments$thread)
n_threads_matched    <- n_distinct(comments$thread[comments$thread %in% submissions$id])

matched <- comments |>
  select(thread, created_utc) |>
  inner_join(submissions |> select(id, sub_created = created_utc),
             by = c("thread" = "id")) |>
  mutate(delta_sec = created_utc - sub_created)

n_comments_matched <- nrow(matched)
n_delta_negative   <- sum(matched$delta_sec < 0)

delta_summary <- tibble(
  n    = n_comments_matched,
  min  = min(matched$delta_sec),
  p1   = quantile(matched$delta_sec, 0.01, type = 7, names = FALSE),
  p25  = quantile(matched$delta_sec, 0.25, type = 7, names = FALSE),
  p50  = quantile(matched$delta_sec, 0.50, type = 7, names = FALSE),
  p90  = quantile(matched$delta_sec, 0.90, type = 7, names = FALSE),
  p99  = quantile(matched$delta_sec, 0.99, type = 7, names = FALSE),
  max  = max(matched$delta_sec),
  mean = round(mean(matched$delta_sec), 0)
)

delta_breaks <- c(-Inf, 3600, 86400, 604800, Inf)
delta_labels <- c("<=1h", "1h-1d", "1d-1w", ">1w")
delta_buckets <- matched |>
  mutate(bucket = cut(delta_sec, breaks = delta_breaks, labels = delta_labels, right = TRUE)) |>
  count(bucket, name = "comments") |>
  mutate(pct = round(100 * comments / n_comments_matched, 2))

# ---- Measurement: most-frequent exact comment bodies (top 300) ----
# INPUT : body_freq.
# DOES  : take the 300 highest-occurrence values; add an md5 of the exact body,
#         its character length, and a one-line preview (newlines -> spaces).
# OUTPUT: top_repeated_bodies.
top_repeated_bodies <- body_freq |>
  arrange(desc(occ)) |>
  slice_head(n = 300) |>
  mutate(
    md5     = vapply(body, function(x) digest(x, algo = "md5", serialize = FALSE), character(1)),
    n_chars = nchar(body),
    preview = str_trunc(str_replace_all(body, "[\r\n]+", " "), 80)
  ) |>
  select(md5, n_chars, occ, threads, preview)

# ---- Reproduction check vs prior reported values ----
# INPUT : the measurements above; prior headline values from sample_analysis.md.
# DOES  : assemble a side-by-side comparison and flag exact agreement.
# OUTPUT: reproduction_check.
reproduction_check <- tribble(
  ~metric,                          ~prior,   ~computed,
  "total_comments",                 891397,   n_comments,
  "total_submissions",              10747,    n_submissions,
  "distinct_comment_bodies",        832861,   n_distinct_bodies,
  "bodies_occurring_once",          826053,   n_bodies_once,
  "bodies_occurring_2plus",         6808,     n_bodies_repeated,
  "rows_in_once_bodies",            826053,   rows_once,
  "rows_in_repeated_bodies",        65344,    rows_repeated,
  "distinct_threads_referenced",    11074,    n_threads_referenced,
  "threads_matched_to_submission",  10308,    n_threads_matched,
  "comments_on_matched_threads",    883733,   n_comments_matched,
  "comments_preceding_submission",  0,        n_delta_negative,
  "within_single_thread_values",    1668,     n_within_single,
  "across_multi_thread_values",     5140,     n_across_multi,
  "rows_within_single_thread",      3706,     rows_within_single,
  "rows_across_multi_thread",       61638,    rows_across_multi,
  "comments_per_thread_median",     14,       per_thread_summary$p50,
  "comments_per_thread_max",        5124,     per_thread_summary$max,
  "delta_median_sec",               7824,     delta_summary$p50,
  "delta_max_sec",                  2508608,  delta_summary$max,
  "delta_le_1h",                    286032,   delta_buckets$comments[delta_buckets$bucket == "<=1h"],
  "delta_1h_1d",                    578774,   delta_buckets$comments[delta_buckets$bucket == "1h-1d"],
  "delta_1d_1w",                    18080,    delta_buckets$comments[delta_buckets$bucket == "1d-1w"],
  "delta_gt_1w",                    847,      delta_buckets$comments[delta_buckets$bucket == ">1w"],
  "max_daily_comments",             45099,    max(activity_by_date$comments),
  "min_daily_comments",             18669,    min(activity_by_date$comments),
  "peak_hour_comments",             58686,    max(activity_by_hour$comments),
  "trough_hour_comments",           9629,     min(activity_by_hour$comments)
) |>
  mutate(difference = computed - prior,
         match = prior == computed)

print(reproduction_check, n = Inf)

# ---- Figures ----
# INPUT : the measurement objects above.
# DOES  : build 5 exploratory plots and save them as PNG. Heavy right-skew is
#         handled with log scales, stated in the axis labels.
# OUTPUT: 5 PNG files under figures/.
ink <- "#2b6cb0"   # single accessible blue for data marks
ref <- "grey55"    # reference lines
theme_set(theme_minimal(base_size = 12))

# Figure 1 — daily comment volume (bars anchored at zero).
fig_daily <- ggplot(activity_by_date, aes(date, comments)) +
  geom_col(fill = ink, width = 0.7) +
  scale_y_continuous(labels = comma, expand = expansion(mult = c(0, 0.05))) +
  scale_x_date(date_breaks = "1 week", date_labels = "%b %d") +
  labs(title = "Daily comment volume — r/politics, July 2026 (UTC)",
       x = NULL, y = "Comments")
ggsave(file.path(fig_dir, "fig1_daily_comment_volume.png"), fig_daily,
       width = 9, height = 5, dpi = 150)

# Figure 2 — exact-text recurrence distribution (frequency-of-frequencies, log-log).
fig_recurrence <- ggplot(occ_ff, aes(occ, n_texts)) +
  geom_point(colour = ink, alpha = 0.7, size = 1.5) +
  scale_x_log10(labels = comma) +
  scale_y_log10(labels = comma) +
  annotation_logticks(sides = "bl", colour = ref) +
  labs(title = "Exact comment-text recurrence distribution",
       x = "Occurrences of a body value (k), log scale",
       y = "Distinct body values occurring exactly k times, log scale")
ggsave(file.path(fig_dir, "fig2_recurrence_distribution.png"), fig_recurrence,
       width = 8, height = 5.5, dpi = 150)

# Figure 3 — concentration among repeated texts (cumulative-share / Lorenz-style).
fig_concentration <- ggplot(concentration_curve, aes(cum_share_texts, cum_share_rows)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = ref) +
  geom_line(colour = ink, linewidth = 1) +
  scale_x_continuous(labels = percent, limits = c(0, 1)) +
  scale_y_continuous(labels = percent, limits = c(0, 1)) +
  coord_equal() +
  labs(title = "Concentration among repeated comment texts",
       x = "Cumulative share of repeated body values (most frequent first)",
       y = "Cumulative share of repeated rows")
ggsave(file.path(fig_dir, "fig3_recurrence_concentration.png"), fig_concentration,
       width = 7, height = 7, dpi = 150)

# Figure 4 — occurrences vs distinct threads per repeated body (log-log).
fig_occ_threads <- ggplot(repeated_freq, aes(occ, threads)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = ref) +
  geom_point(colour = ink, alpha = 0.25, size = 1.3) +
  scale_x_log10(labels = comma) +
  scale_y_log10(labels = comma) +
  annotation_logticks(sides = "bl", colour = ref) +
  labs(title = "Occurrences vs distinct threads — repeated comment texts",
       x = "Total occurrences of a body value, log scale",
       y = "Distinct threads containing it, log scale")
ggsave(file.path(fig_dir, "fig4_occurrences_vs_threads.png"), fig_occ_threads,
       width = 9, height = 6, dpi = 150)

# Figure 5 — elapsed time from submission to comment, matched threads (log scale).
fig_delta <- matched |>
  mutate(log_delta = log10(delta_sec + 1)) |>
  ggplot(aes(log_delta)) +
  geom_histogram(bins = 60, fill = ink) +
  scale_x_continuous(
    breaks = log10(c(0, 60, 3600, 86400, 604800, 2592000) + 1),
    labels = c("0 s", "1 min", "1 h", "1 d", "1 wk", "30 d")) +
  scale_y_continuous(labels = comma) +
  labs(title = "Elapsed time from submission to comment (matched threads)",
       x = "Elapsed time from submission to comment, log10(seconds + 1)",
       y = "Comments")
ggsave(file.path(fig_dir, "fig5_submission_to_comment_delay.png"), fig_delta,
       width = 9, height = 5, dpi = 150)

# ---- Write supporting tables ----
# INPUT : the measurement objects above.
# DOES  : write one CSV per table.
# OUTPUT: CSV files under tables/.
write_csv(reproduction_check,    file.path(tab_dir, "reproduction_check.csv"))
write_csv(occ_distribution,      file.path(tab_dir, "occurrence_frequency_distribution.csv"))
write_csv(occ_ff,                file.path(tab_dir, "occurrence_frequency_of_frequencies.csv"))
write_csv(thread_distribution,   file.path(tab_dir, "thread_recurrence_bands.csv"))
write_csv(concentration_summary, file.path(tab_dir, "recurrence_concentration_summary.csv"))
write_csv(activity_by_date,      file.path(tab_dir, "activity_by_date.csv"))
write_csv(activity_by_hour,      file.path(tab_dir, "activity_by_hour.csv"))
write_csv(per_thread_summary,    file.path(tab_dir, "comments_per_thread_summary.csv"))
write_csv(delta_summary,         file.path(tab_dir, "submission_to_comment_delta_summary.csv"))
write_csv(delta_buckets,         file.path(tab_dir, "submission_to_comment_delta_buckets.csv"))
write_csv(top_repeated_bodies,   file.path(tab_dir, "top_repeated_comment_bodies.csv"))

# ---- Write provenance sidecars ----
# INPUT : the CSV and PNG outputs; the 8 source Parquet files.
# DOES  : write one <output>.manifest.yml per output recording input/output
#         SHA256 hashes, this script's hash, git commit, and package versions.
# OUTPUT: *.manifest.yml sidecars next to each output.
sha256 <- function(path) paste0("sha256:", digest(path, algo = "sha256", file = TRUE))

git_commit <- tryCatch(
  system2("git", c("-C", project_dir, "rev-parse", "--short", "HEAD"),
          stdout = TRUE, stderr = FALSE),
  error = function(e) "")
manifest_packages <- c("nanoparquet", "dplyr", "tidyr", "lubridate",
                       "stringr", "ggplot2", "scales", "readr", "digest", "yaml")
package_versions <- lapply(manifest_packages,
                           function(p) list(name = p, version = as.character(packageVersion(p))))
input_files <- lapply(c(comment_files, submission_files),
                      function(p) list(path = p, hash = sha256(p), format = "parquet"))

write_sidecar <- function(output) {
  is_csv <- grepl("\\.csv$", output)
  manifest <- list(
    manifest_version = 1L,
    output_file = output,
    output_hash = sha256(output),
    output_format = if (is_csv) "csv" else "image/png")
  if (is_csv) {
    tbl <- read_csv(output, show_col_types = FALSE)
    manifest$output_rows <- nrow(tbl)
    manifest$output_cols <- ncol(tbl)
  }
  manifest$input_files <- input_files
  manifest$transformation <- list(
    script = script_path,
    script_hash = sha256(script_path),
    parameters = list(reader = "nanoparquet",
                      exact_match = "byte-identical; no trimming, case-folding, or Unicode normalization",
                      thread_key = "link_id with leading 't3_' removed; joined to submission.id",
                      timezone = "UTC"),
    git_commit = git_commit)
  manifest$software <- list(
    language = "R",
    language_version = paste(R.version$major, R.version$minor, sep = "."),
    packages = package_versions,
    os = paste(Sys.info()[["sysname"]], Sys.info()[["release"]]))
  manifest$timestamp <- format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")
  manifest$notes <- "Descriptive analysis of the July 2026 r/politics sample; source Parquet read-only."
  write_yaml(manifest, paste0(output, ".manifest.yml"))
}

invisible(lapply(list.files(tab_dir, pattern = "\\.csv$", full.names = TRUE), write_sidecar))
invisible(lapply(list.files(fig_dir, pattern = "\\.png$", full.names = TRUE), write_sidecar))

cat("\nDone. Figures ->", fig_dir, "\nTables ->", tab_dir, "\n")
cat("Reproduction: ", sum(reproduction_check$match), "/", nrow(reproduction_check),
    " headline metrics matched exactly.\n", sep = "")
