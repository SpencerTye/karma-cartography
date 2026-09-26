# July 2026 r/politics sample - metadata inventory to support later STM design
#
# Purpose: an auditable inventory of the metadata fields available in the source
# Parquet sample (comments and submissions), their observed characteristics, and
# which of them can be attached to the exact documents that were modelable under
# the delivered Tier 2 LDA representation (r_analysis_outputs/lda_topics_tier2/;
# the K = 40 promotion in lda_topics_tier2_k40_k80/ shares the same document set).
# No topic model is fitted, no covariate is chosen, and nothing is filtered,
# recoded, aggregated away or removed: every measurement runs over all records and
# the tables carry the counts from which such decisions can be taken.
#
# What is measured, in order:
#   1. the source files: Parquet footer facts, declared schemas, SHA256 hashes and
#      the ingest manifest sidecars (manifest.jsonl);
#   2. every field of both record types: declared type and R class, missing, empty,
#      distinct, value counts (low cardinality) or top values and examples (high
#      cardinality), numeric quantiles, timestamp coverage, text length;
#   3. identifier structure and the within-sample linkage the existing scripts use
#      (link_id -> submission id; parent_id prefixes);
#   4. the Tier 2 linkage artifacts (the delivered document-topic tables and the
#      cached theta objects): integrity against their manifests, then the link of
#      every Tier 2 document back to its source record by the established doc_key
#      ('t1_' + comment id, 't3_' + submission id) with field-by-field agreement;
#   5. coverage of every field over the Tier 2 modelable documents, separately for
#      comments and submissions and against the full source corpus; the fields that
#      exist only in the Tier 2 tables; the fields reachable through the established
#      joins (submission record via link_id, parent comment via parent_id);
#   6. reconciliation against the counts documented in the existing project
#      artifacts; discrepancies are reported, never repaired.
# The source Parquet files and every delivered artifact are read-only. All outputs
# go under r_analysis_outputs/metadata_inventory/ (this script owns that folder and
# clears its tables/ and figures/ at the start of each run). Run top to bottom with
# Rscript; no environment variables are read.

# ---- Setup: packages, paths, parameters, output directories ----
# INPUT : none.
# DOES  : load packages; define the source paths, the delivered artifacts read, and
#         every inventory parameter; create and clear the output folders.
# OUTPUT: parameter objects; figures/ and tables/ directories.
suppressPackageStartupMessages({
  library(nanoparquet); library(dplyr); library(tidyr); library(stringi); library(readr)
  library(ggplot2); library(scales); library(patchwork); library(digest); library(yaml); library(jsonlite)
})
run_start    <- Sys.time()
stage_log    <- list()
stage_cursor <- run_start
mark_stage <- function(name) {
  now <- Sys.time()
  stage_log[[length(stage_log) + 1L]] <<- tibble(stage = name,
                                                 start_utc = format(stage_cursor, "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
                                                 end_utc   = format(now, "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
                                                 seconds   = as.numeric(difftime(now, stage_cursor, units = "secs")))
  stage_cursor <<- now
}
log_line <- function(...) { cat(format(Sys.time(), "%H:%M:%S"), ..., "\n"); flush.console() }

project_dir     <- Sys.getenv("KC_REPO_ROOT", unset = getwd())
comments_dir    <- file.path(project_dir, "data_sample/comments/2026-07")
submissions_dir <- file.path(project_dir, "data_sample/submissions/2026-07")
script_path     <- file.path(project_dir, "july2026_politics_metadata_inventory.R")
tier2_dir       <- file.path(project_dir, "r_analysis_outputs/lda_topics_tier2")
k40_dir         <- file.path(project_dir, "r_analysis_outputs/lda_topics_tier2_k40_k80")
out_dir         <- file.path(project_dir, "r_analysis_outputs/metadata_inventory")
fig_dir         <- file.path(out_dir, "figures")
tab_dir         <- file.path(out_dir, "tables")
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(tab_dir, recursive = TRUE, showWarnings = FALSE)
invisible(file.remove(list.files(c(fig_dir, tab_dir), pattern = "\\.(csv|png|yml)$", full.names = TRUE)))

# Delivered Tier 2 artifacts read here (all read-only). The two document tables and the
# theta objects are git-ignored regenerable files; the script records their absence
# rather than failing when a cross-check file is missing. The K = 80 document table
# is the linkage artifact this inventory is about and is required.
artifact_paths <- c(
  document_table_K080           = file.path(tier2_dir, "tables/document_topic_distribution_K080.csv"),
  document_table_K040           = file.path(k40_dir,   "tables/document_topic_distribution_K040.csv"),
  theta_K080_seed1              = file.path(tier2_dir, "models/theta_K080_seed1.rds"),
  theta_K040_seed1              = file.path(tier2_dir, "models/theta_K040_seed1.rds"),
  document_accounting           = file.path(tier2_dir, "tables/document_accounting.csv"),
  document_length_summary       = file.path(tier2_dir, "tables/document_length_summary.csv"),
  text_availability             = file.path(tier2_dir, "tables/text_availability.csv"),
  representative_documents_K080 = file.path(tier2_dir, "tables/representative_documents_K080.csv"))
stopifnot(file.exists(artifact_paths[["document_table_K080"]]))

# Inventory parameters (these shape how the tables are laid out, not what is counted).
low_cardinality_max <- 20L    # a field with at most this many distinct non-missing values gets a full value-count listing
n_top_values        <- 10L    # most frequent values listed for a high-cardinality field
n_first_values      <- 3L     # values in source order listed as examples for every field
example_max_chars   <- 80L    # displayed values are cut to this many characters (code points); '...' marks a cut
quantile_probs      <- c(0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1)
quantile_names      <- c("min", "p01", "p05", "p25", "median", "p75", "p95", "p99", "max")
epoch_second_fields <- c("created_utc", "retrieved_on")   # INT64 epoch seconds per the footer schema and the prior reports
# The exact stored strings enumerated by the earlier reports (sample_analysis.md section A.5;
# lda_topics_tier2/tables/text_availability.csv). Counted, never removed or recoded.
marker_strings      <- c("[deleted]", "[removed]", "[ Removed by Reddit ]", "[ Removed by moderator ]")
usable_definition   <- "not missing (NA) and, for text fields, not the empty string; removal-marker strings count as usable and are also counted in n_marker_valued"

comment_files    <- sort(list.files(comments_dir,    pattern = "\\.parquet$", full.names = TRUE))
submission_files <- sort(list.files(submissions_dir, pattern = "\\.parquet$", full.names = TRUE))
source_manifest_files <- c(file.path(comments_dir, "manifest.jsonl"), file.path(submissions_dir, "manifest.jsonl"))
sha256 <- function(path) paste0("sha256:", digest(path, algo = "sha256", file = TRUE))
log_line("start:", length(comment_files), "comment files,", length(submission_files), "submission files")

# ---- Source files: footer metadata, declared schemas, hashes, ingest sidecars ----
# INPUT : the 8 Parquet files and the 2 manifest.jsonl sidecars (read-only).
# DOES  : read footer facts (rows, columns, row groups, writer) and the declared
#         schema of every file without reading data; hash every file; parse the
#         ingest manifests so their row counts, byte sizes and time ranges can be
#         compared with what the files contain.
# OUTPUT: source_files; schema_by_file; source_manifest.
list_to_chr <- function(x) vapply(x, function(v) if (is.null(v) || length(v) == 0L) NA_character_ else paste(unlist(v), collapse = " "), character(1))
parquet_schema_table <- function(path, record_type) {
  s <- as.data.frame(read_parquet_schema(path))
  s <- s[!is.na(s$type), , drop = FALSE]                          # drop the root node
  tibble(record_type = record_type, file = basename(path), column_index = seq_len(nrow(s)), field = as.character(s$name),
         parquet_type = as.character(s$type), parquet_logical_type = list_to_chr(s$logical_type),
         parquet_converted_type = as.character(s$converted_type), parquet_repetition = as.character(s$repetition_type),
         nanoparquet_r_type = as.character(s$r_type))
}
source_file_facts <- function(path, record_type) {
  info <- read_parquet_info(path)
  tibble(record_type = record_type, file = basename(path), path = path, bytes = file.size(path), sha256 = sha256(path),
         footer_rows = as.numeric(info$num_rows), footer_columns = as.integer(info$num_cols),
         footer_row_groups = as.integer(info$num_row_groups), parquet_version = as.integer(info$parquet_version),
         created_by = as.character(info$created_by))
}
schema_by_file <- bind_rows(lapply(comment_files, parquet_schema_table, record_type = "comment"),
                            lapply(submission_files, parquet_schema_table, record_type = "submission"))
source_files   <- bind_rows(lapply(comment_files, source_file_facts, record_type = "comment"),
                            lapply(submission_files, source_file_facts, record_type = "submission"))
source_manifest <- bind_rows(lapply(source_manifest_files, function(f) {
  bind_rows(lapply(readLines(f, warn = FALSE, encoding = "UTF-8"), function(line) {
    j <- fromJSON(line, simplifyVector = TRUE)
    m <- as.data.frame(j$members)
    tibble(manifest_file = f, kind = j$kind, layout = j$layout, generation = as.integer(j$generation),
           hash_prefix = if (is.null(j$hash_prefix)) NA_character_ else as.character(j$hash_prefix),
           prefix_bits = if (is.null(j$prefix_bits)) NA_integer_ else as.integer(j$prefix_bits),
           file = j$path, manifest_bytes = as.numeric(j$bytes), manifest_rows = sum(m$rows),
           manifest_min_created_utc = if ("min_created_utc" %in% names(m)) min(m$min_created_utc) else NA_real_,
           manifest_max_created_utc = if ("max_created_utc" %in% names(m)) max(m$max_created_utc) else NA_real_,
           manifest_members = paste(m$subreddit, m$subreddit_id, sep = "=", collapse = "|"))
  }))
}))
mark_stage("source_files")

# ---- Read source data ----
# INPUT : the 8 Parquet files.
# DOES  : read every file with nanoparquet (all columns, all rows) and stack per
#         record type; tag each row with its file (a helper column for the per-file
#         checks, not a source field). Nothing is cast, trimmed or dropped.
# OUTPUT: comments; submissions; rows_per_file.
read_type <- function(files) bind_rows(lapply(files, function(f) { x <- as_tibble(as.data.frame(read_parquet(f))); x$source_file <- basename(f); x }))
comments    <- read_type(comment_files)
submissions <- read_type(submission_files)
source_fields <- list(comment = setdiff(names(comments), "source_file"), submission = setdiff(names(submissions), "source_file"))
rows_per_file <- bind_rows(comments |> count(source_file, name = "rows_read"), submissions |> count(source_file, name = "rows_read")) |>
  left_join(bind_rows(comments |> group_by(source_file) |> summarise(observed_min_created_utc = min(created_utc), observed_max_created_utc = max(created_utc), .groups = "drop"),
                      submissions |> group_by(source_file) |> summarise(observed_min_created_utc = min(created_utc), observed_max_created_utc = max(created_utc), .groups = "drop")),
            by = "source_file")
source_files <- source_files |>
  left_join(rows_per_file, by = c("file" = "source_file")) |>
  left_join(source_manifest |> select(file, kind, layout, generation, hash_prefix, prefix_bits, manifest_bytes, manifest_rows,
                                      manifest_min_created_utc, manifest_max_created_utc, manifest_members), by = "file") |>
  mutate(rows_read_equal_footer = rows_read == footer_rows, manifest_rows_equal_rows_read = manifest_rows == rows_read,
         manifest_bytes_equal_file_bytes = manifest_bytes == bytes,
         manifest_time_range_equal_observed = (is.na(manifest_min_created_utc) & is.na(manifest_max_created_utc)) |
           (manifest_min_created_utc == observed_min_created_utc & manifest_max_created_utc == observed_max_created_utc))
mark_stage("read")
log_line("read:", nrow(comments), "comments,", nrow(submissions), "submissions")

# ---- Helper functions for the field measurements ----
# INPUT : a vector of values, its population label, record type and field name.
# DOES  : the per-field measurements: missing / empty / usable / distinct / marker
#         counts; numeric quantiles and special values; UTC coverage of epoch fields;
#         text length and repetition; value listings (all values when the field has at
#         most low_cardinality_max distinct values, else the most frequent) and
#         first-in-source-order examples. Displayed values are cut to example_max_chars.
# OUTPUT: functions used by the inventory sections below.
safe_stat <- function(f, x, ...) if (length(x)) f(x, ...) else NA_real_
truncate_chr <- function(x, n = example_max_chars) {
  x <- stri_replace_all_regex(x, "[\\r\\n\\t]+", " ")
  ifelse(!is.na(x) & stri_length(x) > n, paste0(stri_sub(x, 1L, n), "..."), x)
}
display_value <- function(x, field) {
  if (field %in% epoch_second_fields && is.numeric(x)) {
    ifelse(is.na(x), "<NA>", paste0(format(x, scientific = FALSE, trim = TRUE), " (",
                                    format(as.POSIXct(x, origin = "1970-01-01", tz = "UTC"), "%Y-%m-%d %H:%M:%S"), " UTC)"))
  } else if (is.numeric(x)) {
    ifelse(is.na(x), "<NA>", format(x, scientific = FALSE, trim = TRUE, digits = 15))
  } else {
    ifelse(is.na(x), "<NA>", truncate_chr(as.character(x)))
  }
}
field_kind <- function(x, field) {
  if (field %in% epoch_second_fields) "epoch seconds (INT64) - also summarised as UTC date-time"
  else if (is.character(x)) "text"
  else if (is.logical(x)) "logical"
  else if (is.numeric(x)) "numeric"
  else class(x)[1]
}
summarise_field <- function(x, population, record_type, field, origin) {
  n <- length(x); is_chr <- is.character(x)
  miss  <- is.na(x)
  empty <- if (is_chr) !miss & x == "" else rep(FALSE, n)
  usable <- !miss & !empty
  nd <- n_distinct(x[!miss])
  fk <- field_kind(x, field)
  tibble(population = population, record_type = record_type, field = field, field_origin = origin,
         r_class = class(x)[1], field_kind = fk,
         n_records = n, n_missing = sum(miss), pct_missing = 100 * sum(miss) / n,
         n_empty_string = if (is_chr) sum(empty) else NA_integer_,
         pct_empty_string = if (is_chr) 100 * sum(empty) / n else NA_real_,
         n_usable = sum(usable), pct_usable = 100 * sum(usable) / n,
         n_distinct_nonmissing = nd,
         cardinality_class = case_when(nd == 0L ~ "no non-missing values", nd == 1L ~ "constant",
                                       nd <= low_cardinality_max ~ sprintf("low (at most %d distinct)", low_cardinality_max),
                                       TRUE ~ "high"),
         n_marker_valued = if (is_chr) sum(x %in% marker_strings) else NA_integer_,
         n_true = if (is.logical(x)) sum(x %in% TRUE) else NA_integer_,
         n_false = if (is.logical(x)) sum(x %in% FALSE) else NA_integer_)
}
summarise_numeric <- function(x, population, record_type, field) {
  v <- x[!is.na(x)]; f <- v[is.finite(v)]
  q <- if (length(f)) quantile(f, quantile_probs, type = 7, names = FALSE) else rep(NA_real_, length(quantile_probs))
  out <- tibble(population = population, record_type = record_type, field = field,
                n_nonmissing = length(v), n_nonfinite = sum(!is.finite(v)),
                n_zero = sum(f == 0), n_negative = sum(f < 0), n_positive = sum(f > 0),
                all_integer_valued = if (length(f)) all(f == round(f)) else NA,
                n_distinct_nonmissing = n_distinct(v),
                mean = safe_stat(mean, f), sd = if (length(f) > 1L) sd(f) else NA_real_)
  for (i in seq_along(quantile_names)) out[[quantile_names[i]]] <- q[i]
  out
}
summarise_datetime <- function(x, population, record_type, field) {
  v <- x[!is.na(x)]
  if (!length(v)) return(tibble(population = population, record_type = record_type, field = field, n_nonmissing = 0L))
  t <- as.POSIXct(v, origin = "1970-01-01", tz = "UTC"); d <- as.Date(t)
  days <- seq(min(d), max(d), by = "day")
  per_day <- as.integer(table(factor(as.character(d), levels = as.character(days))))
  tibble(population = population, record_type = record_type, field = field,
         stored_as = "INT64 epoch seconds (footer schema); read as double by nanoparquet; converted to UTC here",
         n_nonmissing = length(v), min_utc = format(min(t), "%Y-%m-%d %H:%M:%S"), max_utc = format(max(t), "%Y-%m-%d %H:%M:%S"),
         span_days = as.numeric(difftime(max(t), min(t), units = "days")),
         n_calendar_days_in_range = length(days), n_days_with_records = sum(per_day > 0), n_days_without_records = sum(per_day == 0),
         records_per_day_min = min(per_day), records_per_day_median = median(per_day), records_per_day_max = max(per_day),
         n_distinct_values = n_distinct(v), n_values_not_whole_seconds = sum(v != round(v)),
         n_distinct_utc_hours_of_day = n_distinct(as.integer(format(t, "%H"))))
}
summarise_character <- function(x, population, record_type, field) {
  v <- x[!is.na(x)]
  if (!length(v)) return(tibble(population = population, record_type = record_type, field = field, n_nonmissing = 0L))
  nc <- stri_length(v)
  vc <- tibble(v = v) |> count(v, name = "n")
  tibble(population = population, record_type = record_type, field = field,
         n_nonmissing = length(v), n_empty_string = sum(v == ""),
         n_whitespace_only_nonempty = sum(v != "" & stri_trim_both(v) == ""),
         n_leading_or_trailing_whitespace = sum(v != stri_trim_both(v)),
         n_containing_newline = sum(stri_detect_fixed(v, "\n")),
         nchar_min = min(nc), nchar_p50 = median(nc), nchar_mean = mean(nc),
         nchar_p95 = quantile(nc, 0.95, type = 7, names = FALSE), nchar_max = max(nc),
         n_distinct_nonmissing = nrow(vc), n_values_occurring_once = sum(vc$n == 1L), n_values_repeated = sum(vc$n > 1L),
         n_rows_in_repeated_values = sum(vc$n[vc$n > 1L]),
         n_value_deleted = sum(v == "[deleted]"), n_value_removed = sum(v == "[removed]"),
         n_value_removed_by_reddit = sum(v == "[ Removed by Reddit ]"), n_value_removed_by_moderator = sum(v == "[ Removed by moderator ]"))
}
value_listing <- function(x, population, record_type, field) {
  vc <- tibble(value = x) |> count(value, name = "n", sort = TRUE) |> mutate(pct = 100 * n / length(x))
  nd <- sum(!is.na(vc$value))
  full <- nd <= low_cardinality_max
  listing <- if (full) "all observed values" else sprintf("%d most frequent of %d distinct non-missing values", n_top_values, nd)
  top <- vc |> slice_head(n = if (full) nrow(vc) else n_top_values)
  # Computed before the constructor so the function argument `field` is not shadowed by the same-named column.
  vd <- display_value(top$value, field)
  vn <- if (is.character(top$value)) stri_length(top$value) else rep(NA_integer_, nrow(top))
  vt <- is.character(top$value) & !is.na(top$value) & stri_length(as.character(top$value)) > example_max_chars
  tibble(population = population, record_type = record_type, field = field, listing = listing, rank = seq_len(nrow(top)),
         value_display = vd, value_nchar = vn, value_truncated = vt, n = top$n, pct = top$pct)
}
first_values <- function(x, population, record_type, field) {
  k <- seq_len(min(n_first_values, length(x)))
  vd <- display_value(x[k], field)
  vt <- is.character(x) & !is.na(x[k]) & stri_length(as.character(x[k])) > example_max_chars
  tibble(population = population, record_type = record_type, field = field, source_order_position = k, value_display = vd, value_truncated = vt)
}
inventory_for <- function(df, fields, record_type, population, origin) {
  num_fields <- fields[vapply(fields, function(f) is.numeric(df[[f]]), logical(1))]
  chr_fields <- fields[vapply(fields, function(f) is.character(df[[f]]), logical(1))]
  list(inventory = bind_rows(lapply(fields, function(f) summarise_field(df[[f]], population, record_type, f, origin))),
       numeric   = bind_rows(lapply(num_fields, function(f) summarise_numeric(df[[f]], population, record_type, f))),
       datetime  = bind_rows(lapply(intersect(fields, epoch_second_fields), function(f) summarise_datetime(df[[f]], population, record_type, f))),
       character = bind_rows(lapply(chr_fields, function(f) summarise_character(df[[f]], population, record_type, f))),
       values    = bind_rows(lapply(fields, function(f) value_listing(df[[f]], population, record_type, f))),
       first     = bind_rows(lapply(fields, function(f) first_values(df[[f]], population, record_type, f))))
}

# ---- Source field inventory (questions 1 to 3) ----
# INPUT : comments; submissions; schema_by_file.
# DOES  : run the field measurements over every source field of both record types
#         (population "source (all records)"); attach the declared Parquet types; record
#         whether each field exists for comments, submissions or both; check that the
#         seven comment files declare identical schemas.
# OUTPUT: field_inventory, field_numeric_summary, field_datetime_summary,
#         field_character_summary, field_value_counts, field_value_examples (source part);
#         source_schema; comment_schemas_identical.
pop_source <- "source (all records)"
inv_c <- inventory_for(comments,    source_fields$comment,    "comment",    pop_source, "source Parquet: comments")
inv_s <- inventory_for(submissions, source_fields$submission, "submission", pop_source, "source Parquet: submissions")
source_schema <- schema_by_file |>
  group_by(record_type, field) |>
  summarise(n_files_declaring_field = n_distinct(file), column_index = paste(unique(column_index), collapse = "|"),
            parquet_type = paste(unique(parquet_type), collapse = "|"),
            parquet_logical_type = paste(unique(coalesce(parquet_logical_type, "")), collapse = "|"),
            parquet_converted_type = paste(unique(coalesce(parquet_converted_type, "")), collapse = "|"),
            parquet_repetition = paste(unique(parquet_repetition), collapse = "|"),
            nanoparquet_r_type = paste(unique(nanoparquet_r_type), collapse = "|"), .groups = "drop") |>
  mutate(present_in = case_when(field %in% source_fields$comment & field %in% source_fields$submission ~ "both record types",
                                field %in% source_fields$comment ~ "comments only", TRUE ~ "submissions only"),
         r_class_after_read = vapply(seq_len(n()), function(i) class((if (record_type[i] == "comment") comments else submissions)[[field[i]]])[1], character(1))) |>
  arrange(record_type, as.integer(sub("\\|.*$", "", column_index)))
comment_schema_signatures <- schema_by_file |> filter(record_type == "comment") |> group_by(file) |>
  summarise(signature = paste(field, parquet_type, coalesce(parquet_logical_type, ""), coalesce(parquet_converted_type, ""), parquet_repetition, sep = ":", collapse = ";"), .groups = "drop")
comment_schemas_identical <- n_distinct(comment_schema_signatures$signature) == 1L
field_inventory_source <- bind_rows(inv_c$inventory, inv_s$inventory) |>
  left_join(source_schema |> select(record_type, field, present_in, column_index, parquet_type, parquet_logical_type, parquet_converted_type, parquet_repetition, n_files_declaring_field),
            by = c("record_type", "field"))
mark_stage("source_inventory")
log_line("source inventory:", nrow(field_inventory_source), "field rows")

# ---- Identifier structure and within-sample linkage of the source records ----
# INPUT : comments; submissions.
# DOES  : measure the identifier fields as stored: patterns (bare base36 ids; 't1_' /
#         't3_' / 't5_' prefixes), lengths, uniqueness, overlap of the two id spaces,
#         and the within-sample references: link_id -> submission id (the convention
#         verified by sample_analysis.md and used by the Tier 2 script), parent_id ->
#         comment id or submission id. Helper columns (thread_id, parent kind / bare id)
#         are derived here for the joins below and are not source fields.
# OUTPUT: identifier_structure; thread_linkage_source; helper columns on comments.
thread_id_of <- function(link_id) substr(link_id, 4L, nchar(link_id))                # the Tier 2 script's rule (line 179)
comments <- comments |>
  mutate(thread_id = thread_id_of(link_id),
         parent_kind = case_when(stri_startswith_fixed(parent_id, "t1_") ~ "t1_ (comment)",
                                 stri_startswith_fixed(parent_id, "t3_") ~ "t3_ (submission)", TRUE ~ "other"),
         parent_bare_id = if_else(parent_kind == "other", NA_character_, substr(parent_id, 4L, nchar(parent_id))),
         parent_present = case_when(parent_kind == "t1_ (comment)" ~ parent_bare_id %in% id,
                                    parent_kind == "t3_ (submission)" ~ parent_bare_id %in% submissions$id, TRUE ~ NA),
         submission_in_sample_recomputed = thread_id %in% submissions$id)
id_row <- function(record_type, field, measure, n, denominator, note = "") tibble(record_type = record_type, field = field, measure = measure, n = n, denominator = denominator, pct_of_denominator = 100 * n / denominator, note = note)
n_c <- nrow(comments); n_s <- nrow(submissions)
comments_per_thread <- comments |> count(thread_id, name = "comments_in_sample")
submission_observed <- submissions |> select(id, num_comments) |> left_join(comments_per_thread, by = c("id" = "thread_id")) |>
  mutate(comments_in_sample = coalesce(comments_in_sample, 0L))
identifier_structure <- bind_rows(
  id_row("comment", "id", "matches ^[0-9a-z]+$ (bare base36, no prefix)", sum(grepl("^[0-9a-z]+$", comments$id)), n_c),
  id_row("comment", "id", "distinct values", n_distinct(comments$id), n_c, "equal to the record count = unique key"),
  id_row("comment", "id", "missing", sum(is.na(comments$id)), n_c),
  id_row("comment", "id", sprintf("character length %s", paste(range(stri_length(comments$id)), collapse = "-")), n_c, n_c),
  id_row("submission", "id", "matches ^[0-9a-z]+$ (bare base36, no prefix)", sum(grepl("^[0-9a-z]+$", submissions$id)), n_s),
  id_row("submission", "id", "distinct values", n_distinct(submissions$id), n_s, "equal to the record count = unique key"),
  id_row("submission", "id", "missing", sum(is.na(submissions$id)), n_s),
  id_row("submission", "id", sprintf("character length %s", paste(range(stri_length(submissions$id)), collapse = "-")), n_s, n_s),
  id_row("both", "id", "bare comment ids that also occur as a submission id", length(intersect(comments$id, submissions$id)), n_c,
         "the Tier 2 doc_key adds 't1_' / 't3_' so the two id spaces cannot collide"),
  id_row("comment", "link_id", "matches ^t3_[0-9a-z]+$", sum(grepl("^t3_[0-9a-z]+$", comments$link_id)), n_c),
  id_row("comment", "link_id", "distinct values (threads referenced)", n_distinct(comments$link_id), n_c),
  id_row("comment", "link_id", "records whose thread (link_id without 't3_') is a submission id in the sample", sum(comments$submission_in_sample_recomputed), n_c,
         "join convention: comment.link_id = 't3_' + submission.id"),
  id_row("comment", "link_id", "records whose thread has no submission in the sample", sum(!comments$submission_in_sample_recomputed), n_c),
  id_row("comment", "link_id", "distinct threads with a submission in the sample", n_distinct(comments$thread_id[comments$submission_in_sample_recomputed]), n_distinct(comments$thread_id), "denominator = distinct threads referenced"),
  id_row("comment", "link_id", "distinct threads without a submission in the sample", n_distinct(comments$thread_id[!comments$submission_in_sample_recomputed]), n_distinct(comments$thread_id), "denominator = distinct threads referenced"),
  id_row("submission", "id", "submissions referenced by at least one comment link_id", sum(submissions$id %in% comments$thread_id), n_s),
  id_row("submission", "id", "submissions referenced by no comment in the sample", sum(!submissions$id %in% comments$thread_id), n_s),
  id_row("comment", "parent_id", "prefix t1_ (parent is a comment)", sum(comments$parent_kind == "t1_ (comment)"), n_c),
  id_row("comment", "parent_id", "prefix t3_ (parent is the submission)", sum(comments$parent_kind == "t3_ (submission)"), n_c),
  id_row("comment", "parent_id", "other prefix or missing", sum(comments$parent_kind == "other"), n_c),
  id_row("comment", "parent_id", "t3_ parent equal to the record's link_id", sum(comments$parent_kind == "t3_ (submission)" & comments$parent_id == comments$link_id), sum(comments$parent_kind == "t3_ (submission)"), "denominator = t3_ parents"),
  id_row("comment", "parent_id", "t1_ parent comment present in the sample", sum(comments$parent_kind == "t1_ (comment)" & comments$parent_present), sum(comments$parent_kind == "t1_ (comment)"), "denominator = t1_ parents"),
  id_row("comment", "parent_id", "t1_ parent comment not in the sample", sum(comments$parent_kind == "t1_ (comment)" & !comments$parent_present), sum(comments$parent_kind == "t1_ (comment)"), "denominator = t1_ parents"),
  id_row("comment", "parent_id", "t3_ parent submission present in the sample", sum(comments$parent_kind == "t3_ (submission)" & comments$parent_present), sum(comments$parent_kind == "t3_ (submission)"), "denominator = t3_ parents"),
  id_row("comment", "parent_id", "distinct values", n_distinct(comments$parent_id), n_c),
  id_row("comment", "subreddit_id", "matches ^t5_[0-9a-z]+$", sum(grepl("^t5_[0-9a-z]+$", comments$subreddit_id)), n_c),
  id_row("submission", "subreddit_id", "matches ^t5_[0-9a-z]+$", sum(grepl("^t5_[0-9a-z]+$", submissions$subreddit_id)), n_s),
  id_row("comment", "author", "distinct values", n_distinct(comments$author), n_c),
  id_row("submission", "author", "distinct values", n_distinct(submissions$author), n_s),
  id_row("both", "author", "distinct comment authors that also occur as a submission author", length(intersect(comments$author, submissions$author)), n_distinct(comments$author), "denominator = distinct comment authors"))
thread_linkage_source <- tibble(
  item = c("distinct threads referenced by comments (link_id without 't3_')", "threads with a submission in the sample", "threads without a submission in the sample",
           "comments on threads with a submission in the sample", "comments on threads without a submission in the sample",
           "submissions in the sample", "submissions referenced by at least one comment", "submissions referenced by no comment",
           "stored num_comments minus comments observed in the sample: minimum", "... p25", "... median", "... p75", "... maximum",
           "submissions whose observed comment count exceeds the stored num_comments", "submissions whose observed comment count equals the stored num_comments"),
  value = c(n_distinct(comments$thread_id), n_distinct(comments$thread_id[comments$submission_in_sample_recomputed]), n_distinct(comments$thread_id[!comments$submission_in_sample_recomputed]),
            sum(comments$submission_in_sample_recomputed), sum(!comments$submission_in_sample_recomputed),
            n_s, sum(submissions$id %in% comments$thread_id), sum(!submissions$id %in% comments$thread_id),
            quantile(submission_observed$num_comments - submission_observed$comments_in_sample, c(0, 0.25, 0.5, 0.75, 1), type = 7, names = FALSE),
            sum(submission_observed$comments_in_sample > submission_observed$num_comments), sum(submission_observed$comments_in_sample == submission_observed$num_comments)),
  basis = c(rep("measured on the source records; join convention comment.link_id = 't3_' + submission.id", 8),
            rep("derived: stored num_comments (a point-in-time count of unknown vintage, see sample_analysis.md) minus the comments present in this sample", 7)))
mark_stage("identifiers")

# ---- Tier 2 linkage artifacts: integrity ----
# INPUT : artifact_paths and their .manifest.yml sidecars.
# DOES  : hash every artifact read here and compare with the SHA256 in its delivered
#         manifest; read the two document tables' identifier and metadata columns
#         (the topic columns are not needed) and count their header columns; read the
#         doc_key and split vectors of the cached theta objects; read the delivered
#         accounting tables. Nothing under lda_topics_tier2/ or
#         lda_topics_tier2_k40_k80/ is written.
# OUTPUT: tier2_artifacts; t2 (the K = 80 document table's metadata columns); dt40;
#         theta_keys; delivered_accounting; delivered_length; delivered_text_availability;
#         representative_header.
manifest_field <- function(path, field) tryCatch({ v <- read_yaml(paste0(path, ".manifest.yml"))[[field]]; if (is.null(v)) NA else v }, error = function(e) NA)
tier2_artifacts <- tibble(role = names(artifact_paths), path = unname(artifact_paths)) |>
  mutate(exists = file.exists(path), bytes = if_else(exists, file.size(path), NA_real_),
         sha256_now = if_else(exists, vapply(path, function(p) if (file.exists(p)) sha256(p) else NA_character_, character(1)), NA_character_),
         sha256_in_manifest = vapply(path, function(p) as.character(manifest_field(p, "output_hash")), character(1)),
         hash_equals_manifest = exists & !is.na(sha256_in_manifest) & sha256_now == sha256_in_manifest,
         manifest_rows = vapply(path, function(p) as.numeric(manifest_field(p, "output_rows")), numeric(1)),
         manifest_cols = vapply(path, function(p) as.numeric(manifest_field(p, "output_cols")), numeric(1)),
         manifest_git_commit = vapply(path, function(p) { t <- manifest_field(p, "transformation"); if (is.list(t) && !is.null(t$git_commit)) as.character(t$git_commit) else NA_character_ }, character(1)))
meta_cols <- c("doc_key", "doc_id", "doc_type", "author", "created_utc", "created_date_utc", "submission_id", "link_id", "parent_id",
               "submission_in_sample", "n_chars", "n_tokens_letters", "n_tokens_modeled", "model_status", "split",
               "dominant_topic", "dominant_topic_share", "n_active_topics")
meta_types <- cols(.default = col_double(), doc_key = col_character(), doc_id = col_character(), doc_type = col_character(),
                   author = col_character(), created_utc = col_double(), created_date_utc = col_character(), submission_id = col_character(),
                   link_id = col_character(), parent_id = col_character(), submission_in_sample = col_logical(), n_chars = col_integer(),
                   n_tokens_letters = col_integer(), n_tokens_modeled = col_integer(), model_status = col_character(), split = col_character(),
                   dominant_topic = col_integer(), dominant_topic_share = col_double(), n_active_topics = col_integer())
read_document_table <- function(path) {
  header <- strsplit(readLines(path, n = 1L, encoding = "UTF-8"), ",", fixed = TRUE)[[1]]
  # na = "NA" only: write_csv wrote NA as the bare token NA and empty strings as "" (quoted), so the round trip is exact;
  # no source id or author equals the string "NA" (checked in the reconciliation below). trim_ws = FALSE keeps values as stored.
  tbl <- read_csv(path, col_select = all_of(meta_cols), col_types = meta_types, na = "NA", trim_ws = FALSE,
                  progress = FALSE, show_col_types = FALSE, lazy = FALSE)
  list(header = header, table = tbl)
}
dt80 <- read_document_table(artifact_paths[["document_table_K080"]])
t2   <- dt80$table
dt40 <- if (file.exists(artifact_paths[["document_table_K040"]])) read_document_table(artifact_paths[["document_table_K040"]]) else NULL
theta_keys <- lapply(c("theta_K080_seed1", "theta_K040_seed1"), function(role) {
  p <- artifact_paths[[role]]
  if (!file.exists(p)) return(NULL)
  th <- readRDS(p)
  out <- list(role = role, K = th$K, seed = th$seed, fit_role = th$role, doc_key = th$doc_key, split = th$split,
              n_rows = nrow(th$theta), n_cols = ncol(th$theta), elements = names(th))
  rm(th); invisible(gc())
  out
})
names(theta_keys) <- c("theta_K080_seed1", "theta_K040_seed1")
theta_keys <- theta_keys[!vapply(theta_keys, is.null, logical(1))]
delivered_accounting <- read_csv(artifact_paths[["document_accounting"]], show_col_types = FALSE, progress = FALSE)
delivered_length     <- read_csv(artifact_paths[["document_length_summary"]], show_col_types = FALSE, progress = FALSE)
delivered_text_availability <- read_csv(artifact_paths[["text_availability"]], show_col_types = FALSE, progress = FALSE)
representative_header <- strsplit(readLines(artifact_paths[["representative_documents_K080"]], n = 1L, encoding = "UTF-8"), ",", fixed = TRUE)[[1]]
mark_stage("tier2_artifacts")
log_line("Tier 2 artifacts:", nrow(t2), "document rows;", sum(tier2_artifacts$hash_equals_manifest), "of", sum(tier2_artifacts$exists), "existing files match their manifests")

# ---- Link every source record to its Tier 2 document (question 4) ----
# INPUT : comments; submissions; t2; dt40; theta_keys; delivered_accounting.
# DOES  : form the established doc_key ('t1_' + id for comments, 't3_' + id for
#         submissions; Tier 2 script lines 175-184) and join the Tier 2 document table
#         onto the source records; count matches in both directions; compare every
#         field the Tier 2 table carries with the source value (author, created_utc,
#         link_id, parent_id, submission_id, doc_type, doc_id, created_date_utc,
#         submission_in_sample, n_chars against the stored text); flag the modelable
#         documents (model_status == "modeled"); compare the counts by type and split
#         with the delivered accounting; check the K = 40 table and the cached theta
#         objects describe the same document set.
# OUTPUT: comments_l; submissions_l (source rows with the Tier 2 columns appended);
#         linkage_counts; linkage_field_agreement; modelable_by_type.
comments    <- comments    |> mutate(doc_key = paste0("t1_", id))
submissions <- submissions |> mutate(doc_key = paste0("t3_", id))
t2_pick <- t2 |> rename(t2_doc_id = doc_id, t2_doc_type = doc_type, t2_author = author, t2_created_utc = created_utc,
                        t2_created_date_utc = created_date_utc, t2_submission_id = submission_id, t2_link_id = link_id,
                        t2_parent_id = parent_id, t2_submission_in_sample = submission_in_sample, t2_n_chars = n_chars)
comments_l    <- comments    |> left_join(t2_pick, by = "doc_key") |> mutate(modelable = model_status %in% "modeled")
submissions_l <- submissions |> left_join(t2_pick, by = "doc_key") |> mutate(modelable = model_status %in% "modeled")
source_keys <- c(comments$doc_key, submissions$doc_key)
utc_date_chr <- function(x) as.character(as.Date(as.POSIXct(x, origin = "1970-01-01", tz = "UTC")))
agree_row <- function(field, record_type, a, b, rule) tibble(tier2_field = field, record_type = record_type, comparison_rule = rule,
                                                             n_compared = length(a), n_agree = sum((is.na(a) & is.na(b)) | (!is.na(a) & !is.na(b) & a == b)),
                                                             n_disagree = n_compared - n_agree)
linkage_field_agreement <- bind_rows(
  agree_row("doc_id", "comment", comments_l$t2_doc_id, comments_l$id, "equals source id"),
  agree_row("doc_type", "comment", comments_l$t2_doc_type, rep("comment", n_c), "equals 'comment'"),
  agree_row("author", "comment", comments_l$t2_author, comments_l$author, "equals source author"),
  agree_row("created_utc", "comment", comments_l$t2_created_utc, comments_l$created_utc, "equals source created_utc"),
  agree_row("created_date_utc", "comment", comments_l$t2_created_date_utc, utc_date_chr(comments_l$created_utc), "equals the UTC date of source created_utc"),
  agree_row("link_id", "comment", comments_l$t2_link_id, comments_l$link_id, "equals source link_id"),
  agree_row("parent_id", "comment", comments_l$t2_parent_id, comments_l$parent_id, "equals source parent_id"),
  agree_row("submission_id", "comment", comments_l$t2_submission_id, comments_l$thread_id, "equals source link_id without 't3_'"),
  agree_row("submission_in_sample", "comment", comments_l$t2_submission_in_sample, comments_l$submission_in_sample_recomputed, "equals (link_id without 't3_') %in% submission ids"),
  agree_row("n_chars", "comment", comments_l$t2_n_chars, stri_length(comments_l$body), "equals the code-point length of source body"),
  agree_row("doc_id", "submission", submissions_l$t2_doc_id, submissions_l$id, "equals source id"),
  agree_row("doc_type", "submission", submissions_l$t2_doc_type, rep("submission", n_s), "equals 'submission'"),
  agree_row("author", "submission", submissions_l$t2_author, submissions_l$author, "equals source author"),
  agree_row("created_utc", "submission", submissions_l$t2_created_utc, submissions_l$created_utc, "equals source created_utc"),
  agree_row("created_date_utc", "submission", submissions_l$t2_created_date_utc, utc_date_chr(submissions_l$created_utc), "equals the UTC date of source created_utc"),
  agree_row("link_id", "submission", submissions_l$t2_link_id, paste0("t3_", submissions_l$id), "equals 't3_' + source id"),
  agree_row("parent_id", "submission", submissions_l$t2_parent_id, rep(NA_character_, n_s), "is NA (submissions have no parent)"),
  agree_row("submission_id", "submission", submissions_l$t2_submission_id, submissions_l$id, "equals source id"),
  agree_row("submission_in_sample", "submission", submissions_l$t2_submission_in_sample, rep(TRUE, n_s), "is TRUE"),
  agree_row("n_chars", "submission", submissions_l$t2_n_chars, stri_length(paste(submissions_l$title, submissions_l$selftext, sep = " ")), "equals the code-point length of title + ' ' + selftext"))
modelable_by_type <- bind_rows(comments_l |> transmute(record_type = "comment", model_status, split),
                               submissions_l |> transmute(record_type = "submission", model_status, split)) |>
  count(record_type, model_status, split, name = "n") |>
  group_by(record_type) |> mutate(pct_of_record_type = 100 * n / sum(n)) |> ungroup() |>
  arrange(record_type, desc(n))
n_modelable_c <- sum(comments_l$modelable); n_modelable_s <- sum(submissions_l$modelable); n_modelable <- n_modelable_c + n_modelable_s
t2_modeled_keys <- t2$doc_key[t2$model_status == "modeled"]
theta_check <- function(tk) {
  if (is.null(tk)) return(tibble(item = "cached theta object", value = "not present", note = "cross-check skipped"))
  tibble(item = c(sprintf("%s: rows x columns of theta", tk$role), sprintf("%s: doc_key count", tk$role), sprintf("%s: doc_key set equals the K = 80 table's modeled doc_keys", tk$role),
                  sprintf("%s: split agrees with the K = 80 table for every doc_key", tk$role), sprintf("%s: elements", tk$role)),
         value = c(paste(tk$n_rows, tk$n_cols, sep = " x "), length(tk$doc_key), setequal(tk$doc_key, t2_modeled_keys) && length(tk$doc_key) == length(t2_modeled_keys),
                   all(tk$split == t2$split[match(tk$doc_key, t2$doc_key)]), paste(tk$elements, collapse = ", ")),
         note = "read from the git-ignored model cache; K, seed and role as stored")
}
linkage_counts <- bind_rows(
  tibble(item = "source records (comments + submissions)", value = as.character(n_c + n_s), note = "read from the 8 Parquet files"),
  tibble(item = "Tier 2 document table rows (K = 80, delivered)", value = as.character(nrow(t2)), note = "document_topic_distribution_K080.csv"),
  tibble(item = "Tier 2 rows with a distinct doc_key", value = as.character(n_distinct(t2$doc_key)), note = ""),
  tibble(item = "Tier 2 rows matched to exactly one source record by doc_key", value = as.character(sum(t2$doc_key %in% source_keys)), note = "doc_key = 't1_' + comment id or 't3_' + submission id"),
  tibble(item = "Tier 2 rows with no source record (unmatched)", value = as.character(sum(!t2$doc_key %in% source_keys)), note = ""),
  tibble(item = "source records with no Tier 2 row (unmatched)", value = as.character(sum(is.na(comments_l$model_status)) + sum(is.na(submissions_l$model_status))), note = ""),
  tibble(item = "Tier 2 doc_type agrees with the source record type", value = as.character(sum(comments_l$t2_doc_type %in% "comment") + sum(submissions_l$t2_doc_type %in% "submission")), note = "of the matched rows"),
  tibble(item = "Tier 2 modelable documents (model_status == 'modeled')", value = as.character(n_modelable), note = sprintf("%d comments + %d submissions", n_modelable_c, n_modelable_s)),
  tibble(item = "Tier 2 modelable documents linked to a source record", value = as.character(sum(comments_l$modelable) + sum(submissions_l$modelable)), note = "every modelable document has exactly one source record"),
  tibble(item = "Tier 2 modelable comments", value = as.character(n_modelable_c), note = sprintf("%.4f%% of source comments", 100 * n_modelable_c / n_c)),
  tibble(item = "Tier 2 modelable submissions", value = as.character(n_modelable_s), note = sprintf("%.4f%% of source submissions", 100 * n_modelable_s / n_s)),
  tibble(item = "Tier 2 training split", value = as.character(sum(t2$split %in% "train")), note = ""),
  tibble(item = "Tier 2 held-out split", value = as.character(sum(t2$split %in% "held-out")), note = ""),
  tibble(item = "delivered document_accounting.csv: modeled documents (comments / submissions / total)",
         value = paste(delivered_accounting[delivered_accounting$stage == "modeled documents", c("comments", "submissions", "total")], collapse = " / "), note = "facts from the delivered artifact"),
  tibble(item = "delivered document_accounting.csv: training / held-out",
         value = paste(delivered_accounting$total[delivered_accounting$stage %in% c("modeled: training set", "modeled: held-out set")], collapse = " / "), note = "facts from the delivered artifact"),
  tibble(item = "K = 80 document table header columns", value = as.character(length(dt80$header)), note = paste("non-topic columns:", paste(setdiff(dt80$header, grep("^topic_", dt80$header, value = TRUE)), collapse = " "))),
  if (!is.null(dt40)) bind_rows(
    tibble(item = "K = 40 document table header columns", value = as.character(length(dt40$header)), note = "document_topic_distribution_K040.csv (companion package)"),
    tibble(item = "K = 40 table: doc_key vector identical to the K = 80 table", value = as.character(identical(dt40$table$doc_key, t2$doc_key)), note = ""),
    tibble(item = "K = 40 table: model_status identical to the K = 80 table", value = as.character(identical(dt40$table$model_status, t2$model_status)), note = ""),
    tibble(item = "K = 40 table: split identical to the K = 80 table", value = as.character(identical(dt40$table$split, t2$split)), note = ""),
    tibble(item = "K = 40 table: n_tokens_modeled identical to the K = 80 table", value = as.character(identical(dt40$table$n_tokens_modeled, t2$n_tokens_modeled)), note = ""))
  else tibble(item = "K = 40 document table", value = "not present", note = "cross-check skipped"),
  bind_rows(lapply(theta_keys, theta_check)))
mark_stage("linkage")
log_line("linkage:", n_modelable, "modelable documents;", sum(linkage_field_agreement$n_disagree), "field disagreements")

# ---- Tier 2 document-table fields ----
# INPUT : comments_l; submissions_l (the Tier 2 columns).
# DOES  : the same field measurements over the 18 identifier / metadata columns of
#         the Tier 2 document table, by record type, over all documents and over the
#         modelable documents. These columns are the Tier 2 script's outputs (carried
#         source fields, derived fields, model outputs); their meaning is as documented
#         in lda_topics_tier2/README.md.
# OUTPUT: tier2 parts of the field tables.
t2_fields <- c("doc_key", "doc_id", "doc_type", "author", "created_utc", "created_date_utc", "submission_id", "link_id", "parent_id",
               "submission_in_sample", "n_chars", "n_tokens_letters", "n_tokens_modeled", "model_status", "split",
               "dominant_topic", "dominant_topic_share", "n_active_topics")
t2_view <- function(df) df |> select(doc_key, doc_id = t2_doc_id, doc_type = t2_doc_type, author = t2_author, created_utc = t2_created_utc,
                                    created_date_utc = t2_created_date_utc, submission_id = t2_submission_id, link_id = t2_link_id,
                                    parent_id = t2_parent_id, submission_in_sample = t2_submission_in_sample, n_chars = t2_n_chars,
                                    n_tokens_letters, n_tokens_modeled, model_status, split, dominant_topic, dominant_topic_share, n_active_topics)
pop_t2_all <- "Tier 2 document table (all documents)"
pop_t2_mod <- "Tier 2 modelable documents"
inv_t2 <- list(
  inventory_for(t2_view(comments_l), t2_fields, "comment", pop_t2_all, "Tier 2 document table"),
  inventory_for(t2_view(submissions_l), t2_fields, "submission", pop_t2_all, "Tier 2 document table"),
  inventory_for(t2_view(comments_l |> filter(modelable)), t2_fields, "comment", pop_t2_mod, "Tier 2 document table"),
  inventory_for(t2_view(submissions_l |> filter(modelable)), t2_fields, "submission", pop_t2_mod, "Tier 2 document table"))
# Source fields measured again over the modelable documents only (for the source-versus-modelable comparison).
inv_mod_c <- inventory_for(comments_l |> filter(modelable), source_fields$comment, "comment", pop_t2_mod, "source Parquet: comments")
inv_mod_s <- inventory_for(submissions_l |> filter(modelable), source_fields$submission, "submission", pop_t2_mod, "source Parquet: submissions")
collect <- function(part) bind_rows(inv_c[[part]], inv_s[[part]], inv_mod_c[[part]], inv_mod_s[[part]], lapply(inv_t2, `[[`, part))
field_inventory <- bind_rows(field_inventory_source, inv_mod_c$inventory, inv_mod_s$inventory, lapply(inv_t2, `[[`, "inventory")) |>
  mutate(usable_definition = usable_definition)
field_numeric_summary   <- collect("numeric")
field_datetime_summary  <- collect("datetime")
field_character_summary <- collect("character")
field_value_counts      <- collect("values")
field_value_examples    <- collect("first")
mark_stage("tier2_fields")

# ---- Field provenance: source fields, Tier 2 fields, and how each reaches a modelable document (question 4) ----
# INPUT : source_fields; dt80$header; dt40$header; theta_keys; representative_header;
#         the Tier 2 script's documented construction (lines 175-188, 351-360, 1030-1052).
# DOES  : one row per field name across the source schemas and the Tier 2 artifacts:
#         where it occurs, how the Tier 2 table derived it (from the script), and the
#         route by which it can be attached to a modelable document.
# OUTPUT: field_provenance.
t2_topic_cols80 <- grep("^topic_", dt80$header, value = TRUE)
t2_topic_cols40 <- if (!is.null(dt40)) grep("^topic_", dt40$header, value = TRUE) else character(0)
derivation <- c(
  id = "carried as doc_id; doc_key = 't1_' + id (comments) or 't3_' + id (submissions)",
  subreddit_id = "not read by the Tier 2 script (not in its column selection)",
  subreddit = "read by the Tier 2 script (column selection) but not written to any Tier 2 table",
  author = "carried unchanged",
  created_utc = "carried, cast to double; created_date_utc = its UTC calendar date",
  score = "not read by the Tier 2 script",
  retrieved_on = "not read by the Tier 2 script",
  raw_json = "not read by the Tier 2 script",
  body = "the comment's modeled text (docs$text = body); not written to the document table; full text in the representative-document tables only",
  link_id = "carried for comments; set to 't3_' + id for submissions; submission_id = link_id without 't3_'",
  parent_id = "carried for comments; NA for submissions",
  title = "half of the submission's modeled text (docs$text = title + ' ' + selftext); not written to the document table",
  selftext = "half of the submission's modeled text (docs$text = title + ' ' + selftext); not written to the document table",
  url = "not read by the Tier 2 script",
  num_comments = "not read by the Tier 2 script",
  upvote_ratio = "not read by the Tier 2 script",
  doc_key = "Reddit fullname: 't1_' + comment id or 't3_' + submission id (Tier 2 script lines 177, 181)",
  doc_id = "the source id",
  doc_type = "'comment' or 'submission' by source table",
  created_date_utc = "UTC date of created_utc",
  submission_id = "link_id without 't3_' for comments; own id for submissions",
  submission_in_sample = "submission_id %in% submission ids of the sample",
  n_chars = "code-point length of the modeled text",
  n_tokens_letters = "letter-run tokens after the Tier 2 tokenizer's step 4",
  n_tokens_modeled = "tokens in the modeled vocabulary (row sum of the document-term matrix)",
  model_status = "'modeled' or one of four 'not modeled: ...' reasons (Tier 2 script lines 355-360)",
  split = "'train' / 'held-out' for modeled documents (seeded 5% held-out sample), NA otherwise",
  dominant_topic = "model output: topic with the largest share (ties to the first), NA when not modeled",
  dominant_topic_share = "model output: that largest share",
  n_active_topics = "model output: topics with share >= 0.1",
  topic_XX = "model output: the document's share of each topic (theta), NA when not modeled")
provenance_row <- function(field, tier2_name = field) {
  in_c <- field %in% source_fields$comment; in_s <- field %in% source_fields$submission
  in_t80 <- tier2_name %in% dt80$header; in_t40 <- tier2_name %in% t2_topic_cols40 || tier2_name %in% (if (!is.null(dt40)) dt40$header else character(0))
  in_theta <- tier2_name %in% c("doc_key", "split") || (tier2_name == "topic_XX")
  in_rep <- tier2_name %in% representative_header || (tier2_name %in% c("body", "title", "selftext") && "text" %in% representative_header)
  origin <- if (in_c || in_s) (if (in_t80) "source field carried into the Tier 2 document table" else "source field not carried into any Tier 2 table")
            else if (tier2_name %in% c("dominant_topic", "dominant_topic_share", "n_active_topics", "topic_XX")) "Tier 2 model output"
            else "Tier 2 derived field"
  route <- if (in_c || in_s) "join on doc_key ('t1_' + id / 't3_' + id) to the Tier 2 document table, or already carried in it" else "present in the Tier 2 document table"
  tibble(field = field, tier2_field_name = if (in_t80 || tier2_name %in% c("topic_XX")) tier2_name else NA_character_,
         in_source_comments = in_c, in_source_submissions = in_s,
         in_tier2_document_table_K080 = in_t80 || tier2_name == "topic_XX", in_tier2_document_table_K040 = in_t40 || (tier2_name == "topic_XX" && length(t2_topic_cols40) > 0),
         in_tier2_theta_rds = in_theta, in_tier2_representative_documents = in_rep,
         origin_class = origin, attachment_route_to_modelable_documents = route,
         derivation_in_tier2_script = unname(derivation[if (field %in% names(derivation)) field else tier2_name]))
}
field_provenance <- bind_rows(
  lapply(union(source_fields$comment, source_fields$submission), function(f) provenance_row(f, tier2_name = if (f == "id") "doc_id" else f)),
  lapply(setdiff(t2_fields, c("doc_id", "author", "created_utc", "link_id", "parent_id")), function(f) provenance_row(f)),
  provenance_row("topic_XX")) |>
  mutate(note = case_when(field == "topic_XX" ~ sprintf("%d columns topic_01..topic_%02d in the K = 80 table; %d in the K = 40 table", length(t2_topic_cols80), length(t2_topic_cols80), length(t2_topic_cols40)),
                          field %in% c("author", "created_utc", "link_id", "parent_id") ~ "same name in source and Tier 2 table",
                          field == "id" ~ "named doc_id in the Tier 2 table", TRUE ~ ""))
mark_stage("provenance")

# ---- Coverage over the modelable documents, and source versus modelable (questions 4 and 5) ----
# INPUT : comments_l; submissions_l; source_fields; t2_fields.
# DOES  : for every source field and every Tier 2 field, count the documents with a
#         usable value (definition in usable_definition), the coverage percentage, the
#         distinct usable values, and the missing / empty / marker counts - over all
#         source records and over the Tier 2 modelable documents, for comments, for
#         submissions and for both together. For the combined scope a field native to
#         one record type is also expressed against the documents of that type.
# OUTPUT: modelable_document_coverage; coverage_source_vs_modelable.
coverage_rows <- function(df, fields, scope, origin, route, n_exists = NULL) {
  bind_rows(lapply(fields, function(f) {
    x <- df[[f]]; miss <- is.na(x)
    empty <- if (is.character(x)) !miss & x == "" else rep(FALSE, length(x))
    usable <- !miss & !empty
    n_ex <- if (is.null(n_exists)) length(x) else n_exists[[f]]
    tibble(field = f, field_origin = origin, attachment_route = route, document_scope = scope,
           n_documents_in_scope = length(x), n_documents_where_field_exists = n_ex,
           n_usable = sum(usable), pct_usable_of_scope = 100 * sum(usable) / length(x),
           pct_usable_where_field_exists = 100 * sum(usable) / n_ex,
           n_distinct_usable = n_distinct(x[usable]), n_missing = sum(miss),
           n_empty_string = if (is.character(x)) sum(empty) else NA_integer_,
           n_marker_valued = if (is.character(x)) sum(x %in% marker_strings) else NA_integer_)
  }))
}
mod_c <- comments_l |> filter(modelable); mod_s <- submissions_l |> filter(modelable)
mod_all <- bind_rows(mod_c |> select(all_of(c(source_fields$comment, "doc_key", "modelable", "model_status"))),
                     mod_s |> select(all_of(c(source_fields$submission, "doc_key", "modelable", "model_status"))))
all_fields <- union(source_fields$comment, source_fields$submission)
n_exists_all <- setNames(lapply(all_fields, function(f) (if (f %in% source_fields$comment) n_modelable_c else 0L) + (if (f %in% source_fields$submission) n_modelable_s else 0L)), all_fields)
origin_of <- function(f) if (f %in% source_fields$comment && f %in% source_fields$submission) "source: both record types" else if (f %in% source_fields$comment) "source: comments" else "source: submissions"
route_native <- "native field of the record; joined to the Tier 2 document by doc_key"
coverage_native <- bind_rows(
  coverage_rows(comments, source_fields$comment, "source: all comments", "source: comments or both", route_native),
  coverage_rows(submissions, source_fields$submission, "source: all submissions", "source: submissions or both", route_native),
  coverage_rows(mod_c, source_fields$comment, "Tier 2 modelable comments", "source: comments or both", route_native),
  coverage_rows(mod_s, source_fields$submission, "Tier 2 modelable submissions", "source: submissions or both", route_native),
  coverage_rows(mod_all, all_fields, "Tier 2 modelable documents (all)", "source", route_native, n_exists = n_exists_all)) |>
  mutate(field_origin = vapply(field, origin_of, character(1)))
t2_all <- bind_rows(t2_view(comments_l), t2_view(submissions_l))
coverage_t2 <- bind_rows(
  coverage_rows(t2_view(comments_l), t2_fields, "source: all comments", "Tier 2 document table", "present in the Tier 2 document table"),
  coverage_rows(t2_view(submissions_l), t2_fields, "source: all submissions", "Tier 2 document table", "present in the Tier 2 document table"),
  coverage_rows(t2_view(mod_c), t2_fields, "Tier 2 modelable comments", "Tier 2 document table", "present in the Tier 2 document table"),
  coverage_rows(t2_view(mod_s), t2_fields, "Tier 2 modelable submissions", "Tier 2 document table", "present in the Tier 2 document table"),
  coverage_rows(t2_all |> filter(model_status == "modeled"), t2_fields, "Tier 2 modelable documents (all)", "Tier 2 document table", "present in the Tier 2 document table"))
mark_stage("coverage")

# ---- Join-derived attachments: submission fields via link_id, parent-comment fields via parent_id ----
# INPUT : comments_l; submissions; comments.
# DOES  : attach to every comment the fields of its submission (join on link_id without
#         't3_' = submission id, the convention verified above) and of its parent
#         comment (join on parent_id without 't1_' = comment id; a 't3_' parent is the
#         submission itself); measure coverage over all comments and over the modelable
#         comments, plus whether the parent comment is itself modelable. Each attached
#         field keeps its source name with a 'submission_' or 'parent_' prefix.
# OUTPUT: coverage_joined; parent_linkage.
sub_attach <- submissions |> select(thread_id = id, all_of(setdiff(source_fields$submission, "id"))) |>
  rename_with(~ paste0("submission_", .x), .cols = -thread_id)
par_attach <- comments |> select(parent_bare_id = id, all_of(setdiff(source_fields$comment, "id"))) |>
  rename_with(~ paste0("parent_", .x), .cols = -parent_bare_id) |>
  left_join(comments_l |> transmute(parent_bare_id = id, parent_model_status = model_status), by = "parent_bare_id")
comments_j <- comments_l |>
  left_join(sub_attach, by = "thread_id") |>
  left_join(par_attach |> mutate(parent_kind_key = "t1_ (comment)"), by = c("parent_bare_id", "parent_kind" = "parent_kind_key"))
sub_fields <- setdiff(names(sub_attach), "thread_id"); par_fields <- setdiff(names(par_attach), "parent_bare_id")
route_sub <- "submission record joined on comment.link_id = 't3_' + submission.id"
route_par <- "parent comment joined on comment.parent_id = 't1_' + comment.id (a 't3_' parent is the submission, reached by the link_id join)"
coverage_joined <- bind_rows(
  coverage_rows(comments_j, sub_fields, "source: all comments", "source: submissions (joined)", route_sub),
  coverage_rows(comments_j |> filter(modelable), sub_fields, "Tier 2 modelable comments", "source: submissions (joined)", route_sub),
  coverage_rows(comments_j, par_fields, "source: all comments", "source: comments (parent, joined)", route_par),
  coverage_rows(comments_j |> filter(modelable), par_fields, "Tier 2 modelable comments", "source: comments (parent, joined)", route_par))
parent_linkage <- bind_rows(comments_j |> mutate(scope = "source: all comments"), comments_j |> filter(modelable) |> mutate(scope = "Tier 2 modelable comments")) |>
  mutate(parent_record = case_when(parent_kind == "t1_ (comment)" & parent_present ~ "parent comment in the sample",
                                   parent_kind == "t1_ (comment)" ~ "parent comment not in the sample",
                                   parent_kind == "t3_ (submission)" & parent_present ~ "parent is the submission, in the sample",
                                   parent_kind == "t3_ (submission)" ~ "parent is the submission, not in the sample", TRUE ~ "other"),
         parent_modelable = case_when(parent_kind != "t1_ (comment)" | !parent_present ~ NA_character_,
                                      parent_model_status == "modeled" ~ "parent comment modelable", TRUE ~ "parent comment not modelable")) |>
  count(scope, parent_kind, parent_record, parent_modelable, name = "n") |>
  group_by(scope) |> mutate(pct_of_scope = 100 * n / sum(n)) |> ungroup() |> arrange(scope, desc(n))
modelable_document_coverage <- bind_rows(coverage_native, coverage_t2, coverage_joined) |>
  mutate(usable_definition = usable_definition) |>
  arrange(factor(document_scope, levels = c("source: all comments", "source: all submissions", "Tier 2 modelable comments", "Tier 2 modelable submissions", "Tier 2 modelable documents (all)")),
          factor(attachment_route, levels = c(route_native, "present in the Tier 2 document table", route_sub, route_par)), field)
coverage_source_vs_modelable <- bind_rows(coverage_native, coverage_t2, coverage_joined) |>
  filter(document_scope != "Tier 2 modelable documents (all)") |>
  mutate(record_type = if_else(grepl("comments", document_scope), "comment", "submission"),
         population = if_else(grepl("^source", document_scope), "source", "modelable")) |>
  select(field, field_origin, attachment_route, record_type, population, n_documents_in_scope, n_usable, pct_usable_of_scope, n_distinct_usable, n_marker_valued) |>
  pivot_wider(names_from = population, values_from = c(n_documents_in_scope, n_usable, pct_usable_of_scope, n_distinct_usable, n_marker_valued)) |>
  mutate(pct_point_difference_modelable_minus_source = pct_usable_of_scope_modelable - pct_usable_of_scope_source,
         share_of_source_usable_values_in_modelable = n_usable_modelable / n_usable_source,
         share_of_source_records_modelable = n_documents_in_scope_modelable / n_documents_in_scope_source) |>
  arrange(record_type, factor(attachment_route, levels = c(route_native, "present in the Tier 2 document table", route_sub, route_par)), field)
mark_stage("joins")

# ---- Numeric distributions and daily coverage, source versus modelable ----
# INPUT : comments_l; submissions_l.
# DOES  : quantiles of every numeric source field (and of the derived ingestion lag
#         retrieved_on - created_utc) over all records and over the modelable
#         documents, by record type; records per UTC day of created_utc in both
#         populations, by record type.
# OUTPUT: numeric_distribution_source_vs_modelable; records_per_utc_day.
comments_l    <- comments_l    |> mutate(ingestion_lag_seconds = retrieved_on - created_utc)
submissions_l <- submissions_l |> mutate(ingestion_lag_seconds = retrieved_on - created_utc)
num_c <- c(intersect(source_fields$comment, names(comments_l)[vapply(comments_l, is.numeric, logical(1))]), "ingestion_lag_seconds")
num_s <- c(intersect(source_fields$submission, names(submissions_l)[vapply(submissions_l, is.numeric, logical(1))]), "ingestion_lag_seconds")
numeric_distribution_source_vs_modelable <- bind_rows(
  lapply(num_c, function(f) summarise_numeric(comments_l[[f]], "source (all records)", "comment", f)),
  lapply(num_c, function(f) summarise_numeric(comments_l[[f]][comments_l$modelable], "Tier 2 modelable documents", "comment", f)),
  lapply(num_s, function(f) summarise_numeric(submissions_l[[f]], "source (all records)", "submission", f)),
  lapply(num_s, function(f) summarise_numeric(submissions_l[[f]][submissions_l$modelable], "Tier 2 modelable documents", "submission", f))) |>
  mutate(basis = if_else(field == "ingestion_lag_seconds", "derived by this script: retrieved_on - created_utc (seconds)", "stored field")) |>
  arrange(record_type, field, desc(population))
records_per_utc_day <- bind_rows(comments_l |> transmute(record_type = "comment", date = as.Date(as.POSIXct(created_utc, origin = "1970-01-01", tz = "UTC")), modelable),
                                 submissions_l |> transmute(record_type = "submission", date = as.Date(as.POSIXct(created_utc, origin = "1970-01-01", tz = "UTC")), modelable)) |>
  group_by(record_type, date) |>
  summarise(n_source = n(), n_modelable = sum(modelable), .groups = "drop") |>
  mutate(n_not_modelable = n_source - n_modelable, pct_modelable = 100 * n_modelable / n_source) |>
  arrange(record_type, date)
mark_stage("distributions")

# ---- Reconciliation checks ----
# INPUT : everything above; the counts documented in the existing project artifacts.
# DOES  : structural identities of this inventory (rows read = footer rows = ingest
#         manifest rows; schemas identical across comment files; keys unique; every
#         Tier 2 document links to one source record and back; carried fields agree;
#         accounting identities), integrity of the delivered artifacts against their
#         manifests, and the recomputation of counts documented in
#         sample_reconnaissance.md, sample_analysis.md, r_analysis_outputs/README.md and
#         the delivered Tier 2 tables. A failed check is recorded and reported; nothing
#         is altered to make it pass.
# OUTPUT: reconciliation_checks; documented_value_checks; discrepancies.
checks <- list()
add_check <- function(check, detail, pass) checks[[length(checks) + 1L]] <<- tibble(check = check, detail = detail, pass = isTRUE(pass))
add_check("rows read equal the Parquet footer row counts for every file",
          paste(sprintf("%s: %d read / %d footer", source_files$file, source_files$rows_read, source_files$footer_rows), collapse = "; "), all(source_files$rows_read_equal_footer))
add_check("ingest manifest.jsonl rows, bytes and time ranges equal the files' contents",
          sprintf("%d of %d files: rows equal; %d bytes equal; %d time ranges equal", sum(source_files$manifest_rows_equal_rows_read), nrow(source_files), sum(source_files$manifest_bytes_equal_file_bytes), sum(source_files$manifest_time_range_equal_observed)),
          all(source_files$manifest_rows_equal_rows_read) && all(source_files$manifest_bytes_equal_file_bytes) && all(source_files$manifest_time_range_equal_observed))
add_check("the seven comment files declare identical schemas (names, types, order)", sprintf("%d distinct schema signatures", n_distinct(comment_schema_signatures$signature)), comment_schemas_identical)
add_check("the comment files tile the month without overlap (each file's earliest created_utc follows the previous file's latest)",
          { o <- source_files |> filter(record_type == "comment") |> arrange(observed_min_created_utc); sprintf("%d of %d consecutive pairs ordered", sum(diff(o$observed_min_created_utc) > 0 & o$observed_min_created_utc[-1] > o$observed_max_created_utc[-nrow(o)]), nrow(o) - 1L) },
          { o <- source_files |> filter(record_type == "comment") |> arrange(observed_min_created_utc); all(o$observed_min_created_utc[-1] > o$observed_max_created_utc[-nrow(o)]) })
add_check("comment id and submission id are unique and never missing", sprintf("%d / %d distinct comment ids; %d / %d distinct submission ids", n_distinct(comments$id), n_c, n_distinct(submissions$id), n_s),
          n_distinct(comments$id) == n_c && !anyNA(comments$id) && n_distinct(submissions$id) == n_s && !anyNA(submissions$id))
add_check("no source id or author equals the literal string 'NA' (so the document table's NA token is unambiguous)",
          sprintf("%d ids, %d authors equal 'NA'", sum(comments$id == "NA") + sum(submissions$id == "NA"), sum(comments$author == "NA", na.rm = TRUE) + sum(submissions$author == "NA", na.rm = TRUE)),
          sum(comments$id == "NA") + sum(submissions$id == "NA") + sum(comments$author == "NA", na.rm = TRUE) + sum(submissions$author == "NA", na.rm = TRUE) == 0)
add_check("every delivered artifact read here matches the SHA256 in its manifest", sprintf("%d of %d existing files match", sum(tier2_artifacts$hash_equals_manifest), sum(tier2_artifacts$exists)),
          all(tier2_artifacts$hash_equals_manifest[tier2_artifacts$exists]))
add_check("the K = 80 document table has the row and column counts recorded in its manifest",
          sprintf("%d rows read, manifest %s; %d header columns, manifest %s", nrow(t2), tier2_artifacts$manifest_rows[tier2_artifacts$role == "document_table_K080"], length(dt80$header), tier2_artifacts$manifest_cols[tier2_artifacts$role == "document_table_K080"]),
          nrow(t2) == tier2_artifacts$manifest_rows[tier2_artifacts$role == "document_table_K080"] && length(dt80$header) == tier2_artifacts$manifest_cols[tier2_artifacts$role == "document_table_K080"])
add_check("every Tier 2 document links to exactly one source record by doc_key, and every source record to exactly one Tier 2 document",
          sprintf("%d Tier 2 rows, %d distinct keys, %d matched, %d source records without a row", nrow(t2), n_distinct(t2$doc_key), sum(t2$doc_key %in% source_keys), sum(is.na(comments_l$model_status)) + sum(is.na(submissions_l$model_status))),
          nrow(t2) == n_c + n_s && n_distinct(t2$doc_key) == nrow(t2) && all(t2$doc_key %in% source_keys) && !anyNA(comments_l$model_status) && !anyNA(submissions_l$model_status))
add_check("every field the Tier 2 document table carries agrees with the source record for every document",
          sprintf("%d comparisons, %d disagreements", sum(linkage_field_agreement$n_compared), sum(linkage_field_agreement$n_disagree)), sum(linkage_field_agreement$n_disagree) == 0)
add_check("modelable documents by type and split equal the delivered document_accounting.csv",
          sprintf("measured %d / %d / %d modeled (comments / submissions / total), %d train, %d held-out", n_modelable_c, n_modelable_s, n_modelable, sum(t2$split %in% "train"), sum(t2$split %in% "held-out")),
          n_modelable_c == delivered_accounting$comments[delivered_accounting$stage == "modeled documents"] &&
            n_modelable_s == delivered_accounting$submissions[delivered_accounting$stage == "modeled documents"] &&
            sum(t2$split %in% "train") == delivered_accounting$total[delivered_accounting$stage == "modeled: training set"] &&
            sum(t2$split %in% "held-out") == delivered_accounting$total[delivered_accounting$stage == "modeled: held-out set"])
status_rows_equal <- local({
  m <- modelable_by_type |> group_by(record_type, model_status) |> summarise(n = sum(n), .groups = "drop")
  d <- delivered_accounting |> filter(grepl("^not modeled", stage)) |>
    mutate(stage = sub("only terms outside the vocabulary$", "only terms outside the vocabulary (document-frequency floor or ceiling)", stage))
  cell <- function(type, stage) { v <- m$n[m$record_type == type & m$model_status == stage]; if (length(v)) v else 0L }
  eq <- vapply(seq_len(nrow(d)), function(i) cell("comment", d$stage[i]) == d$comments[i] && cell("submission", d$stage[i]) == d$submissions[i], logical(1))
  list(n_equal = sum(eq), n_rows = nrow(d))
})
add_check("model_status counts by type equal the delivered document_accounting.csv for every not-modeled reason",
          sprintf("%d of %d reason rows equal for both record types", status_rows_equal$n_equal, status_rows_equal$n_rows),
          status_rows_equal$n_equal == status_rows_equal$n_rows)
add_check("modeled token counts by type equal the delivered document_length_summary.csv",
          sprintf("measured %d comment tokens, %d submission tokens", sum(mod_c$n_tokens_modeled), sum(mod_s$n_tokens_modeled)),
          sum(mod_c$n_tokens_modeled) == delivered_length$tokens[delivered_length$doc_type == "comment"] && sum(mod_s$n_tokens_modeled) == delivered_length$tokens[delivered_length$doc_type == "submission"] &&
            nrow(mod_c) == delivered_length$docs[delivered_length$doc_type == "comment"] && nrow(mod_s) == delivered_length$docs[delivered_length$doc_type == "submission"])
if (!is.null(dt40)) add_check("the K = 40 document table describes the same document set as the K = 80 table (doc_key, model_status, split, n_tokens_modeled identical)",
                             sprintf("identical: %s / %s / %s / %s", identical(dt40$table$doc_key, t2$doc_key), identical(dt40$table$model_status, t2$model_status), identical(dt40$table$split, t2$split), identical(dt40$table$n_tokens_modeled, t2$n_tokens_modeled)),
                             identical(dt40$table$doc_key, t2$doc_key) && identical(dt40$table$model_status, t2$model_status) && identical(dt40$table$split, t2$split) && identical(dt40$table$n_tokens_modeled, t2$n_tokens_modeled))
for (tk in theta_keys) add_check(sprintf("the cached %s.rds doc_key set and split equal the K = 80 table's modeled documents", tk$role),
                                 sprintf("%d keys; theta %d x %d", length(tk$doc_key), tk$n_rows, tk$n_cols),
                                 setequal(tk$doc_key, t2_modeled_keys) && length(tk$doc_key) == length(t2_modeled_keys) && all(tk$split == t2$split[match(tk$doc_key, t2$doc_key)]) && tk$n_rows == length(t2_modeled_keys))
add_check("coverage identities: usable + missing + empty = documents in scope for every coverage row",
          sprintf("%d rows", nrow(modelable_document_coverage)), all(with(modelable_document_coverage, n_usable + n_missing + coalesce(n_empty_string, 0L) == n_documents_in_scope)))
add_check("field inventory identities: usable + missing + empty = records for every field row",
          sprintf("%d rows", nrow(field_inventory)), all(with(field_inventory, n_usable + n_missing + coalesce(n_empty_string, 0L) == n_records)))
add_check("value listings: percentages of a full listing sum to 100 and counts to the record count",
          { v <- field_value_counts |> filter(listing == "all observed values") |> group_by(population, record_type, field) |> summarise(s = sum(pct), n = sum(n), .groups = "drop") |> left_join(field_inventory |> select(population, record_type, field, n_records), by = c("population", "record_type", "field")); sprintf("%d full listings", nrow(v)) },
          { v <- field_value_counts |> filter(listing == "all observed values") |> group_by(population, record_type, field) |> summarise(s = sum(pct), n = sum(n), .groups = "drop") |> left_join(field_inventory |> select(population, record_type, field, n_records), by = c("population", "record_type", "field")); all(abs(v$s - 100) < 1e-9) && all(v$n == v$n_records) })
add_check("daily coverage sums to the record counts", sprintf("%d comment-days, %d submission-days", sum(records_per_utc_day$record_type == "comment"), sum(records_per_utc_day$record_type == "submission")),
          sum(records_per_utc_day$n_source[records_per_utc_day$record_type == "comment"]) == n_c && sum(records_per_utc_day$n_source[records_per_utc_day$record_type == "submission"]) == n_s &&
            sum(records_per_utc_day$n_modelable) == n_modelable)
add_check("the submission_in_sample column of the Tier 2 table equals its recomputation from link_id for every comment",
          sprintf("%d of %d agree", sum(comments_l$t2_submission_in_sample == comments_l$submission_in_sample_recomputed), n_c), all(comments_l$t2_submission_in_sample == comments_l$submission_in_sample_recomputed))
reconciliation_checks <- bind_rows(checks)

# Counts documented in the existing project artifacts, recomputed here.
ta <- function(field, state) { v <- delivered_text_availability$n[delivered_text_availability$field == field & delivered_text_availability$state == state]; if (length(v)) v else NA_real_ }
documented_value_checks <- tribble(
  ~item, ~documented_value, ~documented_in, ~measured_value,
  "comment records", 891397, "sample_reconnaissance.md s1; lda_topics_tier2/tables/document_accounting.csv", n_c,
  "submission records", 10747, "sample_reconnaissance.md s1; document_accounting.csv", n_s,
  "comment files", 7, "sample_reconnaissance.md s1", length(comment_files),
  "distinct comment authors", 135371, "sample_reconnaissance.md s8", n_distinct(comments$author),
  "distinct submission authors", 2321, "sample_reconnaissance.md s8", n_distinct(submissions$author),
  "distinct threads referenced by comments", 11074, "sample_analysis.md definitions; r_analysis_outputs/README.md", n_distinct(comments$thread_id),
  "threads with a submission in the sample", 10308, "sample_analysis.md; r_analysis_outputs/README.md", n_distinct(comments$thread_id[comments$submission_in_sample_recomputed]),
  "comments on matched threads", 883733, "sample_analysis.md; r_analysis_outputs/README.md", sum(comments$submission_in_sample_recomputed),
  "threads without a submission in the sample", 766, "sample_reconnaissance.md s9; sample_analysis.md limitations", n_distinct(comments$thread_id[!comments$submission_in_sample_recomputed]),
  "comments on unmatched threads", 7664, "sample_analysis.md limitations", sum(!comments$submission_in_sample_recomputed),
  "submissions with zero comments in the sample", 439, "sample_reconnaissance.md s9", sum(!submissions$id %in% comments$thread_id),
  "comments whose parent_id has the t1_ prefix (share 60.8%)", 60.8, "sample_reconnaissance.md s3 (percentage)", round(100 * mean(comments$parent_kind == "t1_ (comment)"), 1),
  "comment score minimum", -894, "sample_reconnaissance.md s3", min(comments$score),
  "comment score maximum", 32868, "sample_reconnaissance.md s3", max(comments$score),
  "submission score minimum", 0, "sample_reconnaissance.md s3", min(submissions$score),
  "submission score maximum", 42370, "sample_reconnaissance.md s3", max(submissions$score),
  "submission num_comments minimum", 0, "sample_reconnaissance.md s3", min(submissions$num_comments),
  "submission num_comments maximum", 4858, "sample_reconnaissance.md s3", max(submissions$num_comments),
  "submission upvote_ratio minimum", 0.01, "sample_reconnaissance.md s3", min(submissions$upvote_ratio),
  "submission upvote_ratio maximum", 1, "sample_reconnaissance.md s3", max(submissions$upvote_ratio),
  "raw_json missing in every comment", 891397, "sample_reconnaissance.md s5", sum(is.na(comments$raw_json)),
  "raw_json missing in every submission", 10747, "sample_reconnaissance.md s5", sum(is.na(submissions$raw_json)),
  "comment body = '[removed]'", ta("comment body", "[removed]"), "lda_topics_tier2/tables/text_availability.csv", sum(comments$body %in% "[removed]"),
  "comment body = '[deleted]'", ta("comment body", "[deleted]"), "text_availability.csv", sum(comments$body %in% "[deleted]"),
  "comment body = '[ Removed by Reddit ]'", ta("comment body", "[ Removed by Reddit ]"), "text_availability.csv", sum(comments$body %in% "[ Removed by Reddit ]"),
  "comment body empty string", ta("comment body", "empty string"), "text_availability.csv", sum(comments$body %in% ""),
  "comment author = '[deleted]'", ta("comment author", "[deleted]"), "text_availability.csv", sum(comments$author %in% "[deleted]"),
  "submission author = '[deleted]'", ta("submission author", "[deleted]"), "text_availability.csv", sum(submissions$author %in% "[deleted]"),
  "submission selftext empty string", ta("submission selftext", "empty string"), "text_availability.csv", sum(submissions$selftext %in% ""),
  "submission selftext = '[removed]'", ta("submission selftext", "[removed]"), "text_availability.csv", sum(submissions$selftext %in% "[removed]"),
  "submission selftext = '[deleted]'", ta("submission selftext", "[deleted]"), "text_availability.csv", sum(submissions$selftext %in% "[deleted]"),
  "submission title = '[ Removed by Reddit ]'", ta("submission title", "[ Removed by Reddit ]"), "text_availability.csv", sum(submissions$title %in% "[ Removed by Reddit ]"),
  "submission title = '[ Removed by moderator ]'", 368, "sample_analysis.md s A.5", sum(submissions$title %in% "[ Removed by moderator ]"),
  "submission url empty string", 337, "sample_analysis.md s A.6", sum(submissions$url %in% ""),
  "Tier 2 modeled documents", 857385, "lda_topics_tier2/README.md; document_accounting.csv", n_modelable,
  "Tier 2 modeled comments", 846666, "document_accounting.csv", n_modelable_c,
  "Tier 2 modeled submissions", 10719, "document_accounting.csv", n_modelable_s,
  "Tier 2 training documents", 814516, "document_accounting.csv", sum(t2$split %in% "train"),
  "Tier 2 held-out documents", 42869, "document_accounting.csv", sum(t2$split %in% "held-out"),
  "Tier 2 document table rows", 902144, "document_topic_distribution_K080.csv.manifest.yml", nrow(t2),
  "Tier 2 document table columns (K = 80)", 98, "document_topic_distribution_K080.csv.manifest.yml", length(dt80$header),
  "ingestion lag (retrieved_on - created_utc) never negative, comments", 0, "sample_analysis.md s B.0 (rows with retrieved_on < created_utc)", sum(comments_l$ingestion_lag_seconds < 0),
  "ingestion lag never negative, submissions", 0, "sample_analysis.md s B.0", sum(submissions_l$ingestion_lag_seconds < 0),
  "comment ingestion lag minimum (seconds)", 10, "sample_analysis.md s B.0", min(comments_l$ingestion_lag_seconds),
  "comment ingestion lag maximum (seconds)", 129740, "sample_analysis.md s B.0", max(comments_l$ingestion_lag_seconds)) |>
  mutate(equal = !is.na(documented_value) & abs(measured_value - documented_value) < 1e-9)
discrepancies <- bind_rows(
  reconciliation_checks |> filter(!pass) |> transmute(kind = "reconciliation check failed", item = check, detail = detail),
  documented_value_checks |> filter(!equal) |> transmute(kind = "documented value not reproduced", item = item, detail = sprintf("documented %s (%s); measured %s", format(documented_value), documented_in, format(measured_value))),
  linkage_field_agreement |> filter(n_disagree > 0) |> transmute(kind = "carried field disagrees with source", item = paste(record_type, tier2_field), detail = sprintf("%d of %d disagree", n_disagree, n_compared)))
if (nrow(discrepancies) == 0L) discrepancies <- tibble(kind = "none", item = "no discrepancy found", detail = sprintf("%d reconciliation checks passed; %d documented values reproduced; %d carried-field comparisons agree", nrow(reconciliation_checks), nrow(documented_value_checks), nrow(linkage_field_agreement)))
print(as.data.frame(reconciliation_checks[, c("check", "pass")]), right = FALSE)
mark_stage("reconciliation")
log_line("reconciliation:", sum(reconciliation_checks$pass), "of", nrow(reconciliation_checks), "checks pass;", sum(documented_value_checks$equal), "of", nrow(documented_value_checks), "documented values reproduced")

# ---- Figures ----
# INPUT : comments_l; submissions_l; records_per_utc_day.
# DOES  : four figures, each showing a metadata distribution that a quantile table
#         conveys poorly: (1) stored score by record type; (2) submission num_comments
#         and upvote_ratio; (3) the derived ingestion lag by record type; (4) records
#         per UTC day, all source records against the Tier 2 modelable documents.
#         Titles, axis labels, tick labels and (where two series are drawn) a legend
#         only. Nothing is excluded from any panel.
# OUTPUT: PNG files under figures/.
inventory_theme <- theme_minimal(base_size = 11) +
  theme(panel.grid.minor = element_blank(), panel.grid.major = element_line(colour = "#e1e0d9", linewidth = 0.3),
        axis.text = element_text(colour = "#52514e"), axis.title = element_text(colour = "#0b0b0b"),
        plot.title = element_text(colour = "#0b0b0b", size = 12), strip.text = element_text(colour = "#0b0b0b", size = 11),
        plot.background = element_rect(fill = "#fcfcfb", colour = NA), panel.background = element_rect(fill = "#fcfcfb", colour = NA),
        legend.position = "top", legend.title = element_blank())
fill_one    <- "#2a78d6"
palette_two <- c("all source records" = "#2a78d6", "Tier 2 modelable documents" = "#eb6834")
save_fig <- function(name, plot, width, height) ggsave(file.path(fig_dir, name), plot, width = width, height = height, dpi = 150, bg = "#fcfcfb")

score_df <- bind_rows(tibble(record_type = "comment", score = comments_l$score), tibble(record_type = "submission", score = submissions_l$score))
fig1 <- ggplot(score_df, aes(x = score)) +
  geom_histogram(bins = 80, fill = fill_one, colour = NA) +
  scale_x_continuous(transform = transform_pseudo_log(sigma = 1, base = 10), breaks = c(-1000, -100, -10, 0, 10, 100, 1000, 10000), labels = label_comma()) +
  scale_y_continuous(labels = label_comma(), expand = expansion(mult = c(0, 0.05))) +
  facet_wrap(~ record_type, ncol = 2, scales = "free_y") +
  labs(title = "Stored score, all source records", x = "score (pseudo-log axis, base 10)", y = "records") + inventory_theme
save_fig("fig1_score_distribution_by_record_type.png", fig1, 9, 4.2)

fig2a <- ggplot(submissions_l, aes(x = num_comments)) + geom_histogram(bins = 60, fill = fill_one, colour = NA) +
  scale_x_continuous(transform = transform_pseudo_log(sigma = 1, base = 10), breaks = c(0, 10, 100, 1000), labels = label_comma()) +
  scale_y_continuous(labels = label_comma(), expand = expansion(mult = c(0, 0.05))) +
  labs(title = "Submission num_comments, all source records", x = "num_comments (pseudo-log axis, base 10)", y = "submissions") + inventory_theme
fig2b <- ggplot(submissions_l, aes(x = upvote_ratio)) + geom_histogram(binwidth = 0.02, boundary = 0, fill = fill_one, colour = NA) +
  scale_x_continuous(breaks = seq(0, 1, 0.2)) +
  scale_y_continuous(labels = label_comma(), expand = expansion(mult = c(0, 0.05))) +
  labs(title = "Submission upvote_ratio, all source records", x = "upvote_ratio", y = "submissions") + inventory_theme
save_fig("fig2_submission_num_comments_upvote_ratio.png", fig2a + fig2b, 9, 4.2)

lag_df <- bind_rows(tibble(record_type = "comment", lag = comments_l$ingestion_lag_seconds), tibble(record_type = "submission", lag = submissions_l$ingestion_lag_seconds))
fig3 <- ggplot(lag_df, aes(x = lag + 1)) + geom_histogram(bins = 80, fill = fill_one, colour = NA) +
  scale_x_continuous(transform = transform_log10(), breaks = c(10, 60, 600, 3600, 86400) + 1, labels = c("10 s", "1 min", "10 min", "1 h", "1 d")) +
  scale_y_continuous(labels = label_comma(), expand = expansion(mult = c(0, 0.05))) +
  facet_wrap(~ record_type, ncol = 2, scales = "free_y") +
  labs(title = "Ingestion lag: retrieved_on minus created_utc, all source records", x = "seconds (log axis; plotted as lag + 1)", y = "records") + inventory_theme
save_fig("fig3_ingestion_lag_by_record_type.png", fig3, 9, 4.2)

daily_long <- records_per_utc_day |> select(record_type, date, n_source, n_modelable) |>
  pivot_longer(c(n_source, n_modelable), names_to = "population", values_to = "records") |>
  mutate(population = factor(if_else(population == "n_source", "all source records", "Tier 2 modelable documents"), levels = names(palette_two)))
fig4 <- ggplot(daily_long, aes(x = date, y = records, colour = population)) +
  geom_line(linewidth = 0.7) + geom_point(size = 1.6) +
  scale_colour_manual(values = palette_two) +
  scale_y_continuous(labels = label_comma(), limits = c(0, NA), expand = expansion(mult = c(0, 0.05))) +
  scale_x_date(date_breaks = "5 days", date_labels = "%b %d") +
  facet_wrap(~ record_type, ncol = 1, scales = "free_y") +
  labs(title = "Records per UTC day of created_utc: all source records and Tier 2 modelable documents", x = "UTC date, July 2026", y = "records") + inventory_theme
save_fig("fig4_records_per_utc_day_source_vs_modelable.png", fig4, 9, 6)
mark_stage("figures")

# ---- Write tables ----
# INPUT : the measurement objects above.
# DOES  : write one CSV per table and record its dimensions for the sidecars.
# OUTPUT: CSV files under tables/.
table_dims <- list()
write_table <- function(tbl, name) {
  path <- file.path(tab_dir, name)
  write_csv(tbl, path)
  table_dims[[path]] <<- c(rows = nrow(tbl), cols = ncol(tbl))
  invisible(path)
}
parameters_table <- tibble(parameter = c("low_cardinality_max", "n_top_values", "n_first_values", "example_max_chars", "quantile_probs", "epoch_second_fields", "marker_strings", "usable_definition", "doc_key_rule", "thread_join_rule", "parent_join_rule", "parquet_reader"),
                           value = c(low_cardinality_max, n_top_values, n_first_values, example_max_chars, paste(quantile_probs, collapse = ", "), paste(epoch_second_fields, collapse = ", "),
                                     paste(marker_strings, collapse = " | "), usable_definition,
                                     "doc_key = 't1_' + comment id, 't3_' + submission id (Tier 2 script lines 177 and 181)",
                                     "comment.link_id = 't3_' + submission.id, i.e. thread id = link_id without its first three characters (Tier 2 script line 179)",
                                     "comment.parent_id = 't1_' + parent comment id; a 't3_' parent_id names the submission",
                                     "nanoparquet::read_parquet (all columns); INT64 columns arrive as double"))
git_commit <- tryCatch(system2("git", c("-C", project_dir, "rev-parse", "--short", "HEAD"), stdout = TRUE, stderr = FALSE), error = function(e) "")
if (length(git_commit) == 0L) git_commit <- ""
run_end <- Sys.time()
stage_timings <- bind_rows(stage_log)
run_info <- tibble(item = c("run_start_utc", "run_end_utc", "elapsed_seconds", "r_version", "os", "git_commit_at_run", "script", "script_sha256",
                            "comment_records", "submission_records", "tier2_document_rows", "tier2_modelable_documents", "reconciliation_checks_passed", "documented_values_reproduced"),
                   value = c(format(run_start, "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"), format(run_end, "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
                             sprintf("%.1f", as.numeric(difftime(run_end, run_start, units = "secs"))), R.version.string, paste(Sys.info()[["sysname"]], Sys.info()[["release"]]),
                             git_commit, script_path, sha256(script_path), n_c, n_s, nrow(t2), n_modelable,
                             sprintf("%d of %d", sum(reconciliation_checks$pass), nrow(reconciliation_checks)), sprintf("%d of %d", sum(documented_value_checks$equal), nrow(documented_value_checks))))
write_table(reconciliation_checks,                "reconciliation_checks.csv")
write_table(documented_value_checks,              "documented_value_checks.csv")
write_table(discrepancies,                        "discrepancies.csv")
write_table(source_files,                         "source_files.csv")
write_table(source_schema,                        "source_schema.csv")
write_table(field_inventory,                      "field_inventory.csv")
write_table(field_numeric_summary,                "field_numeric_summary.csv")
write_table(field_datetime_summary,               "field_datetime_summary.csv")
write_table(field_character_summary,              "field_character_summary.csv")
write_table(field_value_counts,                   "field_value_counts.csv")
write_table(field_value_examples,                 "field_value_examples.csv")
write_table(identifier_structure,                 "identifier_structure.csv")
write_table(thread_linkage_source,                "thread_linkage.csv")
write_table(parent_linkage,                       "parent_linkage.csv")
write_table(tier2_artifacts,                      "tier2_artifacts.csv")
write_table(linkage_counts,                       "linkage_counts.csv")
write_table(linkage_field_agreement,              "linkage_field_agreement.csv")
write_table(modelable_by_type,                    "model_status_by_type.csv")
write_table(field_provenance,                     "field_provenance.csv")
write_table(modelable_document_coverage,          "modelable_document_coverage.csv")
write_table(coverage_source_vs_modelable,         "coverage_source_vs_modelable.csv")
write_table(numeric_distribution_source_vs_modelable, "numeric_distribution_source_vs_modelable.csv")
write_table(records_per_utc_day,                  "records_per_utc_day.csv")
write_table(parameters_table,                     "parameters.csv")
write_table(stage_timings,                        "stage_timings.csv")
write_table(run_info,                             "run_info.csv")
mark_stage("tables")

# ---- Provenance sidecars ----
# INPUT : the CSV and PNG outputs; the 8 source Parquet files and 2 ingest manifests;
#         the delivered artifacts read.
# DOES  : write one <output>.manifest.yml per output (input hashes, output hash and
#         dimensions, script hash, git commit, parameters, package versions). No
#         stochastic operation runs in this script, so no seed is recorded.
# OUTPUT: *.manifest.yml next to each output.
manifest_packages <- c("nanoparquet", "dplyr", "tidyr", "stringi", "readr", "ggplot2", "scales", "patchwork", "digest", "yaml", "jsonlite")
package_versions  <- lapply(manifest_packages, function(p) list(name = p, version = as.character(packageVersion(p))))
input_files <- c(
  lapply(seq_len(nrow(source_files)), function(i) list(path = source_files$path[i], hash = source_files$sha256[i], format = "parquet", rows = as.integer(source_files$rows_read[i]), cols = as.integer(source_files$footer_columns[i]), role = "source data, read-only")),
  lapply(source_manifest_files, function(p) list(path = p, hash = sha256(p), format = "jsonl", role = "ingest manifest sidecar of the source data, read-only")),
  lapply(which(tier2_artifacts$exists), function(i) list(path = tier2_artifacts$path[i], hash = tier2_artifacts$sha256_now[i],
                                                         format = if (grepl("\\.rds$", tier2_artifacts$path[i])) "rds" else "csv",
                                                         role = if (grepl("\\.rds$", tier2_artifacts$path[i])) "cached fit of the delivered Tier 2 run, read-only (doc_key and split read)" else "delivered Tier 2 table, read-only")))
manifest_parameters <- c(list(reader = "nanoparquet", documents_linked_by = "doc_key = 't1_' + comment id / 't3_' + submission id (the Tier 2 script's rule)",
                              modelable_definition = "Tier 2 document table model_status == 'modeled'",
                              no_model_fitted = "TRUE", no_records_filtered = "TRUE"),
                         setNames(as.list(parameters_table$value), parameters_table$parameter))
write_sidecar <- function(output) {
  ext <- tolower(tools::file_ext(output))
  manifest <- list(manifest_version = 1L, output_file = output, output_hash = sha256(output),
                   output_format = switch(ext, csv = "csv", png = "image/png", ext))
  if (ext == "csv") {
    dims <- table_dims[[output]]
    manifest$output_rows <- unname(dims[["rows"]]); manifest$output_cols <- unname(dims[["cols"]])
  }
  manifest$input_files    <- input_files
  manifest$transformation <- list(script = script_path, script_hash = sha256(script_path), parameters = manifest_parameters, git_commit = git_commit)
  manifest$software <- list(language = "R", language_version = paste(R.version$major, R.version$minor, sep = "."),
                            packages = package_versions, os = paste(Sys.info()[["sysname"]], Sys.info()[["release"]]))
  manifest$timestamp <- format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")
  manifest$notes     <- "Metadata inventory of the July 2026 r/politics sample for later STM design: source fields, their observed characteristics, and their coverage over the Tier 2 LDA modelable documents. Source Parquet and delivered artifacts read-only; no model fitted; no records filtered, recoded or removed."
  write_yaml(manifest, paste0(output, ".manifest.yml"))
}
invisible(lapply(list.files(tab_dir, pattern = "\\.csv$", full.names = TRUE), write_sidecar))
invisible(lapply(list.files(fig_dir, pattern = "\\.png$", full.names = TRUE), write_sidecar))
mark_stage("manifests")

# ---- Console summary ----
# INPUT : the headline tables.
# DOES  : print them.
# OUTPUT: console text.
print(as.data.frame(field_inventory |> filter(population == pop_source) |> select(record_type, field, r_class, n_records, n_missing, n_empty_string, n_usable, n_distinct_nonmissing, cardinality_class)), right = FALSE)
print(as.data.frame(linkage_counts), right = FALSE)
print(as.data.frame(discrepancies), right = FALSE)
log_line("done in", sprintf("%.1f s", as.numeric(difftime(Sys.time(), run_start, units = "secs"))), ";", length(table_dims), "tables,", length(list.files(fig_dir, pattern = "\\.png$")), "figures")
