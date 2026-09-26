# July 2026 r/politics sample — time to first direct reply
#
# Standalone side analysis. Reads the raw Parquet comment files, verifies the
# parent/child reference convention from the data, reconstructs direct
# comment-to-comment replies, and measures the elapsed time from each
# replied-to comment to its earliest observed direct reply. Source Parquet is
# read-only. All outputs are written under r_analysis_outputs/first_direct_reply/
# and are reproducible by re-running this script top to bottom.

# ---- Setup: packages, paths, output directories ----
# INPUT : none.
# DOES  : load packages; define source paths; create output folders.
# OUTPUT: path variables; figures/ and tables/ directories.
library(nanoparquet)
library(dplyr)
library(ggplot2)
library(scales)
library(readr)
library(digest)
library(yaml)

project_dir  <- Sys.getenv("KC_REPO_ROOT", unset = getwd())
comments_dir <- file.path(project_dir, "data_sample/comments/2026-07")
script_path  <- file.path(project_dir, "july2026_politics_first_direct_reply.R")

out_dir <- file.path(project_dir, "r_analysis_outputs/first_direct_reply")
fig_dir <- file.path(out_dir, "figures")
tab_dir <- file.path(out_dir, "tables")
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(tab_dir, recursive = TRUE, showWarnings = FALSE)

comment_files <- list.files(comments_dir, pattern = "\\.parquet$", full.names = TRUE)

# ---- Read source data ----
# INPUT : 7 comment Parquet files (read-only).
# DOES  : read each file and stack; keep id, parent_id, link_id, created_utc
#         (column selection only; no rows dropped); cast created_utc to double;
#         derive the UTC calendar date of creation.
# OUTPUT: comments (one row per comment); n_comments.
comments <- lapply(comment_files, read_parquet) |>
  bind_rows() |>
  select(id, parent_id, link_id, created_utc) |>
  mutate(created_utc  = as.numeric(created_utc),
         created_date = as.Date(as.POSIXct(created_utc, origin = "1970-01-01", tz = "UTC")))

n_comments <- nrow(comments)

# ---- Verify the parent/child reference convention ----
# INPUT : comments.
# DOES  : check that id is unique, non-missing, and carries no type prefix;
#         classify every parent_id by what it resolves to: a 't3_' value equal
#         to the comment's own link_id (reply to the submission); a 't1_' value
#         whose remainder matches a comment id in the sample; a 't1_' value
#         with no matching comment in the sample; a 't3_' value that differs
#         from link_id; or a value not matching the 't1_'/'t3_' + base36 form.
# OUTPUT: comments with parent_key and parent_kind; parent_reference_check.
n_id_distinct   <- n_distinct(comments$id)
n_id_missing    <- sum(is.na(comments$id))
n_id_unprefixed <- sum(grepl("^[0-9a-z]+$", comments$id))

parent_kind_levels <- c("comment_in_sample", "comment_not_in_sample",
                        "submission_own_link_id", "submission_other_link_id",
                        "malformed_or_unknown")
comments <- comments |>
  mutate(
    parent_key  = substr(parent_id, 4, nchar(parent_id)),
    parent_kind = case_when(
      !grepl("^t[13]_[0-9a-z]+$", parent_id)                   ~ "malformed_or_unknown",
      substr(parent_id, 1, 3) == "t3_" & parent_id == link_id  ~ "submission_own_link_id",
      substr(parent_id, 1, 3) == "t3_"                         ~ "submission_other_link_id",
      parent_key %in% id                                       ~ "comment_in_sample",
      TRUE                                                     ~ "comment_not_in_sample"
    ),
    parent_kind = factor(parent_kind, levels = parent_kind_levels)
  )

parent_reference_check <- comments |>
  count(parent_kind, name = "n", .drop = FALSE) |>
  mutate(prop = n / n_comments)

n_parent_not_in_sample      <- sum(comments$parent_kind == "comment_not_in_sample")
n_parent_not_in_sample_day1 <- sum(comments$parent_kind == "comment_not_in_sample" &
                                     comments$created_date == as.Date("2026-07-01"))

# ---- Reconstruct direct comment replies ----
# INPUT : comments.
# DOES  : keep comments whose parent is a comment present in the sample
#         (parent_kind == "comment_in_sample"); join each to its parent by id
#         to attach the parent's created_utc and link_id.
# OUTPUT: replies (one row per direct reply whose parent is in the sample).
replies <- comments |>
  filter(parent_kind == "comment_in_sample") |>
  select(reply_id = id, reply_link_id = link_id,
         reply_created_utc = created_utc, parent_key) |>
  inner_join(
    comments |> select(parent_key = id, parent_link_id = link_id,
                       parent_created_utc = created_utc),
    by = "parent_key"
  )

n_replies_direct      <- nrow(replies)
n_replies_same_thread <- sum(replies$reply_link_id == replies$parent_link_id)
n_replies_self        <- sum(replies$reply_id == replies$parent_key)

# ---- First direct reply per replied-to comment ----
# INPUT : replies.
# DOES  : sort replies by created time within each parent (ties broken by
#         reply id); keep the earliest per parent; count direct replies per
#         parent and how many share the earliest second.
# OUTPUT: first_reply (one row per replied-to comment).
first_reply <- replies |>
  arrange(parent_key, reply_created_utc, reply_id) |>
  group_by(parent_key) |>
  summarise(
    parent_created_utc      = first(parent_created_utc),
    first_reply_id          = first(reply_id),
    first_reply_created_utc = first(reply_created_utc),
    n_direct_replies        = n(),
    n_tied_earliest         = sum(reply_created_utc == first(reply_created_utc)),
    .groups = "drop"
  )

# ---- Elapsed time to first direct reply ----
# INPUT : first_reply.
# DOES  : elapsed_sec = first reply created_utc minus parent created_utc.
#         No row is removed on the basis of its elapsed value; negative,
#         zero-second, and tied cases are counted and kept.
# OUTPUT: first_reply with elapsed_sec; population and anomaly counts.
first_reply <- first_reply |>
  mutate(elapsed_sec = first_reply_created_utc - parent_created_utc)

n_with_reply    <- nrow(first_reply)
n_without_reply <- n_comments - n_with_reply
n_negative      <- sum(first_reply$elapsed_sec < 0)
n_zero_second   <- sum(first_reply$elapsed_sec == 0)
n_tied_earliest <- sum(first_reply$n_tied_earliest > 1)

# ---- Summary statistics ----
# INPUT : first_reply (elapsed_sec); population counts.
# DOES  : quantiles (type 7), mean, extremes, anomaly counts; counts and
#         proportions of first replies in fixed elapsed-time bands and within
#         each band's upper bound (cumulative).
# OUTPUT: reply_summary (long table); reply_bands.
elapsed <- first_reply$elapsed_sec
q <- function(p) quantile(elapsed, p, type = 7, names = FALSE)

reply_summary <- tribble(
  ~measure,                                            ~value,
  "comments_examined",                                 n_comments,
  "comments_with_observed_direct_reply",               n_with_reply,
  "comments_no_direct_reply_observed",                 n_without_reply,
  "prop_with_observed_direct_reply",                   n_with_reply / n_comments,
  "prop_no_direct_reply_observed",                     n_without_reply / n_comments,
  "direct_replies_with_parent_in_sample",              n_replies_direct,
  "replies_whose_parent_is_not_in_sample",             n_parent_not_in_sample,
  "replies_whose_parent_is_not_in_sample_created_jul01", n_parent_not_in_sample_day1,
  "first_reply_min_sec",                               min(elapsed),
  "first_reply_p01_sec",                               q(0.01),
  "first_reply_p05_sec",                               q(0.05),
  "first_reply_p10_sec",                               q(0.10),
  "first_reply_p25_sec",                               q(0.25),
  "first_reply_p50_sec",                               q(0.50),
  "first_reply_mean_sec",                              mean(elapsed),
  "first_reply_p75_sec",                               q(0.75),
  "first_reply_p90_sec",                               q(0.90),
  "first_reply_p95_sec",                               q(0.95),
  "first_reply_p99_sec",                               q(0.99),
  "first_reply_max_sec",                               max(elapsed),
  "first_reply_negative_count",                        n_negative,
  "first_reply_zero_second_count",                     n_zero_second,
  "first_reply_tied_earliest_count",                   n_tied_earliest
)

band_upper_sec <- c(-1, 1, 5, 10, 15, 30, 60, 300, 600, 3600, 86400, Inf)
band_label     <- c("negative (< 0 s)", "0-1 s", "1-5 s", "5-10 s", "10-15 s", "15-30 s",
                    "30 s-1 min", "1-5 min", "5-10 min", "10 min-1 h", "1 h-1 d", "> 1 d")
reply_bands <- first_reply |>
  mutate(band = cut(elapsed_sec, breaks = c(-Inf, band_upper_sec),
                    labels = band_label, right = TRUE)) |>
  count(band, name = "n_in_band", .drop = FALSE) |>
  mutate(upper_bound_sec   = band_upper_sec,
         prop_in_band      = n_in_band / n_with_reply,
         n_within_upper    = cumsum(n_in_band),
         prop_within_upper = n_within_upper / n_with_reply) |>
  select(band, upper_bound_sec, n_in_band, prop_in_band, n_within_upper, prop_within_upper)

# ---- Observed-reply share by comment creation date ----
# INPUT : comments (created_date); first_reply (parent_key).
# DOES  : for each UTC calendar day of comment creation, count comments and
#         how many have a direct reply observed in the sample.
# OUTPUT: reply_by_parent_date.
reply_by_parent_date <- comments |>
  mutate(has_observed_reply = id %in% first_reply$parent_key) |>
  group_by(created_date) |>
  summarise(comments = n(),
            with_observed_direct_reply = sum(has_observed_reply),
            .groups = "drop") |>
  mutate(prop_with_observed_direct_reply = with_observed_direct_reply / comments)

# ---- Figure bins ----
# INPUT : first_reply (elapsed_sec).
# DOES  : bin elapsed seconds with integer-aligned edges: one second wide up to
#         10 s, then growing by about 12% per bin (edges from
#         unique(round(10^seq(0, 6.5, by = 0.05))), offset by -0.5 so every
#         integer second falls inside exactly one bin); count per bin. Also
#         count first replies at each exact second from 0 to 60 s.
# OUTPUT: hist_bins (one row per bin, empty bins included); zoom_bins.
bin_edges_sec <- c(-0.5, unique(round(10^seq(0, 6.5, by = 0.05))) - 0.5)
hist_bins <- first_reply |>
  mutate(bin = cut(elapsed_sec, breaks = bin_edges_sec, right = FALSE)) |>
  count(bin, name = "n", .drop = FALSE) |>
  mutate(edge_lower = head(bin_edges_sec, -1),
         edge_upper = tail(bin_edges_sec, -1),
         first_sec  = edge_lower + 0.5,
         last_sec   = edge_upper - 0.5) |>
  select(first_sec, last_sec, n, edge_lower, edge_upper)

zoom_bins <- first_reply |>
  filter(elapsed_sec <= 60) |>
  count(elapsed_sec, name = "n")
n_within_60 <- sum(zoom_bins$n)

# ---- Figure ----
# INPUT : hist_bins; zoom_bins; summary values.
# DOES  : main panel = count per bin on x = log10(seconds + 1) with tick labels
#         in human units; inset = the 0-60 s region at one bar per second,
#         rendered to a grob on a null device (so no stray Rplots.pdf is
#         written); save as PNG.
# OUTPUT: figures/fig_time_to_first_direct_reply.png.
ink_series <- "#2a78d6"
ink_text   <- "#0b0b0b"
ink_sub    <- "#52514e"
ink_muted  <- "#898781"
grid_col   <- "#e1e0d9"
axis_col   <- "#c3c2b7"

theme_fig <- theme_minimal(base_size = 12) +
  theme(panel.grid.minor = element_blank(),
        panel.grid.major = element_line(colour = grid_col, linewidth = 0.3),
        axis.line.x      = element_line(colour = axis_col, linewidth = 0.4),
        axis.text        = element_text(colour = ink_muted),
        axis.title       = element_text(colour = ink_sub),
        plot.title       = element_text(colour = ink_text, face = "bold"),
        plot.subtitle    = element_text(colour = ink_sub),
        plot.caption     = element_text(colour = ink_muted, hjust = 0, size = 9),
        plot.background  = element_rect(fill = "white", colour = NA))

tick_sec <- c(0, 1, 10, 60, 600, 3600, 86400, 604800, 2592000)
tick_lab <- c("0 s", "1 s", "10 s", "1 min", "10 min", "1 h", "1 d", "1 wk", "30 d")
x_limits <- log10(range(bin_edges_sec) + 1)
y_max    <- max(hist_bins$n)

fig_zoom <- ggplot(zoom_bins, aes(elapsed_sec, n)) +
  geom_col(fill = ink_series, width = 1) +
  scale_x_continuous(limits = c(-0.5, 60.5), breaks = seq(0, 60, 10),
                     labels = function(x) paste0(x, " s"), expand = expansion(0)) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.08))) +
  labs(x = "Time to first direct reply", y = "Comments") +
  theme_fig +
  theme(text            = element_text(size = 8),
        plot.background = element_rect(fill = "white", colour = axis_col, linewidth = 0.3),
        plot.margin     = margin(6, 8, 2, 4))

pdf(NULL)
fig_main <- ggplot(hist_bins) +
  geom_rect(aes(xmin = log10(edge_lower + 1), xmax = log10(edge_upper + 1),
                ymin = 0, ymax = n), fill = ink_series) +
  annotation_custom(ggplotGrob(fig_zoom),
                    xmin = x_limits[1] + 0.05, xmax = 1.95,
                    ymin = 0.34 * y_max, ymax = 0.995 * y_max) +
  scale_x_continuous(breaks = log10(tick_sec + 1), labels = tick_lab,
                     limits = x_limits, expand = expansion(0)) +
  scale_y_continuous(labels = comma, expand = expansion(mult = c(0, 0.04))) +
  labs(title = "Time to first direct reply — r/politics comments, July 2026",
       x = "Time to first direct reply, log10(seconds + 1)",
       y = "Comments") +
  theme_fig

ggsave(file.path(fig_dir, "fig_time_to_first_direct_reply.png"), fig_main,
       width = 10, height = 5.8, dpi = 150, bg = "white")
invisible(dev.off())

# ---- Validation checks ----
# INPUT : the objects above.
# DOES  : recompute key quantities by a second route and compare; reconcile
#         totals; stop if any check fails.
# OUTPUT: validation_checks (one row per check).
earliest_by_parent <- replies |>
  group_by(parent_key) |>
  summarise(min_created = min(reply_created_utc), .groups = "drop")
first_vs_min <- first_reply |>
  inner_join(earliest_by_parent, by = "parent_key")

validation_checks <- tribble(
  ~check, ~detail, ~pass,
  "comment ids unique and non-missing",
    sprintf("%d distinct of %d rows; %d missing", n_id_distinct, n_comments, n_id_missing),
    n_id_distinct == n_comments && n_id_missing == 0,
  "comment ids carry no type prefix (bare base36)",
    sprintf("%d of %d match ^[0-9a-z]+$", n_id_unprefixed, n_comments),
    n_id_unprefixed == n_comments,
  "every parent_id matches ^t[13]_[0-9a-z]+$",
    sprintf("%d malformed or unknown", parent_reference_check$n[parent_reference_check$parent_kind == "malformed_or_unknown"]),
    parent_reference_check$n[parent_reference_check$parent_kind == "malformed_or_unknown"] == 0,
  "every 't3_' parent_id equals the comment's own link_id",
    sprintf("%d 't3_' parents differ from link_id", parent_reference_check$n[parent_reference_check$parent_kind == "submission_other_link_id"]),
    parent_reference_check$n[parent_reference_check$parent_kind == "submission_other_link_id"] == 0,
  "parent categories reconcile to all comments",
    sprintf("%d categorised of %d", sum(parent_reference_check$n), n_comments),
    sum(parent_reference_check$n) == n_comments,
  "each direct reply joins to exactly one parent row",
    sprintf("%d replies filtered, %d rows after join", sum(comments$parent_kind == "comment_in_sample"), n_replies_direct),
    n_replies_direct == sum(comments$parent_kind == "comment_in_sample"),
  "reply and parent share link_id (same thread)",
    sprintf("%d of %d", n_replies_same_thread, n_replies_direct),
    n_replies_same_thread == n_replies_direct,
  "no comment is its own parent",
    sprintf("%d self-references", n_replies_self),
    n_replies_self == 0,
  "one first-reply row per replied-to comment",
    sprintf("%d rows; %d distinct parents among replies", n_with_reply, n_distinct(replies$parent_key)),
    n_with_reply == n_distinct(replies$parent_key) && n_distinct(first_reply$parent_key) == n_with_reply,
  "selected first reply is the earliest direct reply observed (recomputed with min)",
    sprintf("%d of %d match", sum(first_vs_min$first_reply_created_utc == first_vs_min$min_created), n_with_reply),
    nrow(first_vs_min) == n_with_reply && all(first_vs_min$first_reply_created_utc == first_vs_min$min_created),
  "elapsed equals first reply time minus parent time (recomputed)",
    sprintf("%d of %d match", sum(first_reply$elapsed_sec == first_reply$first_reply_created_utc - first_reply$parent_created_utc), n_with_reply),
    all(first_reply$elapsed_sec == first_reply$first_reply_created_utc - first_reply$parent_created_utc),
  "no rows removed after the elapsed calculation (negatives counted, not dropped)",
    sprintf("%d negative; %d zero-second; %d tied earliest; %d rows retained", n_negative, n_zero_second, n_tied_earliest, n_with_reply),
    n_with_reply == n_distinct(replies$parent_key),
  "replied-to plus no-reply comments equal comments examined",
    sprintf("%d + %d = %d", n_with_reply, n_without_reply, n_comments),
    n_with_reply + n_without_reply == n_comments,
  "figure bins contain every first-reply observation",
    sprintf("%d in bins of %d", sum(hist_bins$n), n_with_reply),
    sum(hist_bins$n) == n_with_reply,
  "elapsed-time bands sum to the replied-to population",
    sprintf("%d in bands; cumulative ends at %d", sum(reply_bands$n_in_band), last(reply_bands$n_within_upper)),
    sum(reply_bands$n_in_band) == n_with_reply && last(reply_bands$n_within_upper) == n_with_reply,
  "zoom panel count equals the cumulative count within 1 min",
    sprintf("%d in zoom; %d within 1 min", n_within_60, reply_bands$n_within_upper[reply_bands$band == "30 s-1 min"]),
    n_within_60 == reply_bands$n_within_upper[reply_bands$band == "30 s-1 min"]
)

print(as.data.frame(validation_checks[, c("check", "pass")]), right = FALSE)
stopifnot(all(validation_checks$pass))

# ---- Write supporting tables ----
# INPUT : the measurement objects above.
# DOES  : write one CSV per table.
# OUTPUT: CSV files under tables/.
write_csv(reply_summary,          file.path(tab_dir, "first_direct_reply_summary.csv"))
write_csv(reply_bands,            file.path(tab_dir, "first_direct_reply_bands.csv"))
write_csv(parent_reference_check, file.path(tab_dir, "parent_reference_check.csv"))
write_csv(reply_by_parent_date,   file.path(tab_dir, "first_direct_reply_by_comment_date.csv"))
write_csv(hist_bins,              file.path(tab_dir, "first_direct_reply_histogram_bins.csv"))
write_csv(validation_checks,      file.path(tab_dir, "validation_checks.csv"))

# ---- Write provenance sidecars ----
# INPUT : the CSV and PNG outputs; the 7 source Parquet files.
# DOES  : write one <output>.manifest.yml per output recording input/output
#         SHA256 hashes, this script's hash, git commit, and package versions.
# OUTPUT: *.manifest.yml sidecars next to each output.
sha256 <- function(path) paste0("sha256:", digest(path, algo = "sha256", file = TRUE))

git_commit <- tryCatch(
  system2("git", c("-C", project_dir, "rev-parse", "--short", "HEAD"),
          stdout = TRUE, stderr = FALSE),
  error = function(e) "")
manifest_packages <- c("nanoparquet", "dplyr", "ggplot2", "scales", "readr", "digest", "yaml")
package_versions  <- lapply(manifest_packages,
                            function(p) list(name = p, version = as.character(packageVersion(p))))
input_files <- lapply(comment_files,
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
    parameters = list(
      reader = "nanoparquet",
      direct_reply = "comment whose parent_id is 't1_' + the id of a comment present in the sample",
      first_reply = "earliest direct reply by created_utc; same-second ties broken by reply id (elapsed unaffected)",
      elapsed = "first reply created_utc minus parent created_utc, integer seconds, UTC",
      figure_x_transform = "log10(seconds + 1)",
      figure_bins = "integer-aligned edges: unique(round(10^seq(0, 6.5, by = 0.05))) - 0.5"),
    git_commit = git_commit)
  manifest$software <- list(
    language = "R",
    language_version = paste(R.version$major, R.version$minor, sep = "."),
    packages = package_versions,
    os = paste(Sys.info()[["sysname"]], Sys.info()[["release"]]))
  manifest$timestamp <- format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")
  manifest$notes <- "Time to first direct reply, July 2026 r/politics sample; source Parquet read-only."
  write_yaml(manifest, paste0(output, ".manifest.yml"))
}

invisible(lapply(list.files(tab_dir, pattern = "\\.csv$", full.names = TRUE), write_sidecar))
invisible(lapply(list.files(fig_dir, pattern = "\\.png$", full.names = TRUE), write_sidecar))

# ---- Console summary ----
# INPUT : reply_summary; reply_bands; parent_reference_check.
# DOES  : print the headline tables.
# OUTPUT: console text.
print(as.data.frame(parent_reference_check))
print(as.data.frame(reply_summary))
print(as.data.frame(reply_bands))
cat("\nDone. Figure ->", fig_dir, "\nTables ->", tab_dir, "\n")
