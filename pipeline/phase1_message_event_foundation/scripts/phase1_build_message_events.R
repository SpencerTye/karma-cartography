# Phase 1: build the canonical Reddit message-event table (one row per submission or comment)
# with doc_key, thread and parent linkage, and provenance back to the source Parquet row, then QC it.
# Inputs: comment and submission Parquet files; optional metadata-inventory tables and Tier 2 document table.
# Outputs (under pipeline/phase1_message_event_foundation/): derived/message_events.parquet, qc/tables/*.csv,
#   qc/figures/*.png and environment/* run records, each with a .manifest.yml sidecar. Source files are read only.
# Env: PHASE1_PROJECT_DIR, PHASE1_COMMENTS_DIR, PHASE1_SUBMISSIONS_DIR, PHASE1_COMPRESSION (zstd),
#   PHASE1_COMPRESSION_LEVEL (3), PHASE1_ROW_GROUP_ROWS, PHASE1_INVENTORY_DIR, PHASE1_TIER2_TABLE.

# ---- Setup: packages, timing, project root, parameters, output folders ----------------
# Resolve paths, create output folders and clear the file types this script writes.
suppressPackageStartupMessages({
  library(nanoparquet); library(dplyr); library(tidyr); library(stringi); library(readr)
  library(ggplot2); library(scales); library(digest); library(yaml); library(jsonlite); library(ps)
})
run_start <- Sys.time()
cpu_start <- proc.time()
invisible(gc(reset = TRUE))          # so that gc()'s "max used" column reports this run's peak

find_project_dir <- function() {
  env <- Sys.getenv("PHASE1_PROJECT_DIR", unset = "")
  if (nzchar(env)) return(normalizePath(env, winslash = "/", mustWork = TRUE))
  arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(arg)) {
    script <- normalizePath(sub("^--file=", "", arg[1]), winslash = "/", mustWork = TRUE)
    return(dirname(dirname(dirname(dirname(script)))))   # scripts/ -> phase folder -> pipeline folder -> project root
  }
  normalizePath(getwd(), winslash = "/", mustWork = TRUE)
}
find_script_path <- function(project_dir) {
  arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(arg)) return(normalizePath(sub("^--file=", "", arg[1]), winslash = "/", mustWork = TRUE))
  file.path(project_dir, "pipeline/phase1_message_event_foundation/scripts/phase1_build_message_events.R")
}
env_or <- function(name, default) { v <- Sys.getenv(name, unset = ""); if (nzchar(v)) v else default }

project_dir     <- find_project_dir()
script_path     <- find_script_path(project_dir)
phase_dir       <- file.path(project_dir, "pipeline/phase1_message_event_foundation")
comments_dir    <- env_or("PHASE1_COMMENTS_DIR",    file.path(project_dir, "data_sample/comments/2026-07"))
submissions_dir <- env_or("PHASE1_SUBMISSIONS_DIR", file.path(project_dir, "data_sample/submissions/2026-07"))
inventory_dir   <- env_or("PHASE1_INVENTORY_DIR",   file.path(project_dir, "r_analysis_outputs/metadata_inventory/tables"))
tier2_table     <- env_or("PHASE1_TIER2_TABLE",     file.path(project_dir, "r_analysis_outputs/lda_topics_tier2/tables/document_topic_distribution_K080.csv"))
compression     <- env_or("PHASE1_COMPRESSION", "zstd")
compression_level <- as.integer(env_or("PHASE1_COMPRESSION_LEVEL", "3"))
row_group_rows  <- env_or("PHASE1_ROW_GROUP_ROWS", "")
derived_dir <- file.path(phase_dir, "derived")
tab_dir     <- file.path(phase_dir, "qc/tables")
fig_dir     <- file.path(phase_dir, "qc/figures")
env_dir     <- file.path(phase_dir, "environment")
for (d in c(derived_dir, tab_dir, fig_dir, env_dir)) dir.create(d, recursive = TRUE, showWarnings = FALSE)
invisible(file.remove(list.files(c(derived_dir, tab_dir, fig_dir, env_dir), pattern = "\\.(parquet|csv|png|txt|yml)$", full.names = TRUE)))
events_path <- file.path(derived_dir, "message_events.parquet")

comment_files    <- sort(list.files(comments_dir,    pattern = "\\.parquet$", full.names = TRUE))
submission_files <- sort(list.files(submissions_dir, pattern = "\\.parquet$", full.names = TRUE))
ingest_manifests <- file.path(c(comments_dir, submissions_dir), "manifest.jsonl")
ingest_manifests <- ingest_manifests[file.exists(ingest_manifests)]
stopifnot("at least one comment Parquet file" = length(comment_files) > 0L, "at least one submission Parquet file" = length(submission_files) > 0L)

# The exact stored strings counted as platform marker values (counted, never removed).
marker_strings <- c("[deleted]", "[removed]", "[ Removed by Reddit ]", "[ Removed by moderator ]", "")

stage_log <- list(); stage_cursor <- run_start
mark_stage <- function(name) {
  now <- Sys.time(); mem <- tryCatch(ps::ps_memory_info(), error = function(e) c(rss = NA_real_, peak_wset = NA_real_))
  stage_log[[length(stage_log) + 1L]] <<- tibble(stage = name,
    start_utc = format(stage_cursor, "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"), end_utc = format(now, "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
    seconds = as.numeric(difftime(now, stage_cursor, units = "secs")),
    working_set_mb_at_end = unname(mem[["rss"]]) / 2^20,
    peak_working_set_mb_so_far = if ("peak_wset" %in% names(mem)) unname(mem[["peak_wset"]]) / 2^20 else NA_real_)
  stage_cursor <<- now
}
log_line <- function(...) { cat(format(Sys.time(), "%H:%M:%S"), ..., "\n"); flush.console() }
sha256 <- function(path) paste0("sha256:", digest(path, algo = "sha256", file = TRUE))
relative_path <- function(path, root) {
  p <- normalizePath(path, winslash = "/", mustWork = FALSE); r <- paste0(normalizePath(root, winslash = "/", mustWork = FALSE), "/")
  ifelse(startsWith(p, r), substring(p, nchar(r) + 1L), p)
}
log_line("Phase 1 start; project root:", project_dir)

# ---- Functions: the transformation and the summaries (pure; unit-tested) --------------
# Defined before any data are read so tests/test_phase1_functions.R can evaluate these
# definitions on a synthetic fixture.

# doc_key: the Reddit "fullname" and the project's established document key
# ('t1_' + comment id, 't3_' + submission id). The two id spaces cannot collide.
make_doc_key <- function(record_type, reddit_id) {
  prefix <- c(comment = "t1_", submission = "t3_")[record_type]
  if (anyNA(prefix)) stop("record_type must be 'comment' or 'submission'")
  paste0(unname(prefix), reddit_id)
}

# Read every Parquet file of one record type with nanoparquet (all columns, all rows, no
# casting) and tag each row with the file it came from and its 1-based row within that file.
read_source_records <- function(files, project_dir) {
  bind_rows(lapply(files, function(f) {
    x <- as_tibble(as.data.frame(read_parquet(f)))
    x$source_file <- relative_path(f, project_dir)
    x$source_row_in_file <- seq_len(nrow(x))
    x
  }))
}

# The canonical message-event table: one row per source record, comments first (in the
# order given) then submissions (in the order given). Every source field is carried
# unchanged; fields native to one record type are NA for the other. Derived fields and
# their rules:
#   doc_key                't1_' + id (comment) / 't3_' + id (submission)
#   record_type            'comment' / 'submission' (from the source table the row came from)
#   reddit_id              the source id, unchanged
#   thread_id              comment: link_id as stored; submission: 't3_' + id. Equal to the
#                          doc_key of the thread's submission, so a submission is the root
#                          of its own thread (is_thread_root = TRUE)
#   thread_root_in_dataset thread_id is the doc_key of a submission present in the dataset
#   parent_id              comment: as stored (a doc_key: 't1_...' a comment, 't3_...' the
#                          submission); submission: NA (no parent)
#   parent_kind            'comment' / 'submission' from the parent_id prefix ('other' if
#                          neither); NA for submissions and for a missing parent_id
#   parent_in_dataset      parent_id is the doc_key of a row of this table; NA for submissions
#                          and for a missing parent_id. Absent parents are kept, not invented
#   created_datetime_utc   created_utc as a UTC date-time; created_date_utc its UTC calendar date
#   text                   comment: body; submission: paste(title, selftext, sep = " ") - the
#                          project's established rule (a submission with empty selftext
#                          therefore ends in one space); the originals stay in their columns
#   text_n_chars           stri_length(text): Unicode code points
build_message_events <- function(comments, submissions) {
  need_c <- c("id", "parent_id", "link_id", "author", "body", "created_utc", "retrieved_on", "score", "subreddit", "subreddit_id", "raw_json", "source_file", "source_row_in_file")
  need_s <- c("id", "title", "selftext", "url", "author", "created_utc", "retrieved_on", "score", "num_comments", "upvote_ratio", "subreddit", "subreddit_id", "raw_json", "source_file", "source_row_in_file")
  miss_c <- setdiff(need_c, names(comments)); miss_s <- setdiff(need_s, names(submissions))
  if (length(miss_c)) stop("comments lack the field(s): ", paste(miss_c, collapse = ", "))
  if (length(miss_s)) stop("submissions lack the field(s): ", paste(miss_s, collapse = ", "))
  if (anyNA(comments$id) || anyNA(submissions$id)) stop("a source id is missing; the record cannot be keyed")
  if (anyDuplicated(comments$id)) stop("duplicate comment id(s) in the source: ", paste(head(unique(comments$id[duplicated(comments$id)]), 5), collapse = ", "))
  if (anyDuplicated(submissions$id)) stop("duplicate submission id(s) in the source: ", paste(head(unique(submissions$id[duplicated(submissions$id)]), 5), collapse = ", "))
  submission_keys <- make_doc_key("submission", submissions$id)
  ev_c <- comments |>
    transmute(doc_key = make_doc_key("comment", id), record_type = "comment", reddit_id = id, author,
              created_utc = as.numeric(created_utc), retrieved_on = as.numeric(retrieved_on), subreddit, subreddit_id,
              thread_id = link_id, is_thread_root = FALSE, link_id, parent_id,
              score = as.numeric(score), num_comments = NA_real_, upvote_ratio = NA_real_,
              url = NA_character_, title = NA_character_, selftext = NA_character_, body,
              text = body, raw_json = as.character(raw_json), source_file, source_row_in_file = as.integer(source_row_in_file))
  ev_s <- submissions |>
    transmute(doc_key = make_doc_key("submission", id), record_type = "submission", reddit_id = id, author,
              created_utc = as.numeric(created_utc), retrieved_on = as.numeric(retrieved_on), subreddit, subreddit_id,
              thread_id = make_doc_key("submission", id), is_thread_root = TRUE, link_id = NA_character_, parent_id = NA_character_,
              score = as.numeric(score), num_comments = as.numeric(num_comments), upvote_ratio = as.numeric(upvote_ratio),
              url, title, selftext, body = NA_character_,
              text = paste(title, selftext, sep = " "), raw_json = as.character(raw_json), source_file, source_row_in_file = as.integer(source_row_in_file))
  ev <- bind_rows(ev_c, ev_s)
  all_keys <- ev$doc_key
  ev |>
    mutate(created_datetime_utc = as.POSIXct(created_utc, origin = "1970-01-01", tz = "UTC"),
           created_date_utc = as.Date(created_datetime_utc, tz = "UTC"),
           thread_root_in_dataset = thread_id %in% submission_keys,
           parent_kind = case_when(record_type == "submission" | is.na(parent_id) ~ NA_character_,
                                   stri_startswith_fixed(parent_id, "t1_") ~ "comment",
                                   stri_startswith_fixed(parent_id, "t3_") ~ "submission",
                                   TRUE ~ "other"),
           parent_in_dataset = case_when(record_type == "submission" | is.na(parent_id) ~ NA, TRUE ~ parent_id %in% all_keys),
           text_n_chars = stri_length(text)) |>
    select(doc_key, record_type, reddit_id, author, created_utc, created_datetime_utc, created_date_utc, retrieved_on,
           subreddit, subreddit_id, thread_id, is_thread_root, thread_root_in_dataset, link_id, parent_id, parent_kind,
           parent_in_dataset, score, num_comments, upvote_ratio, url, title, selftext, body, text, text_n_chars, raw_json,
           source_file, source_row_in_file)
}

# One row per thread represented in the table: every thread_id of a comment plus every
# submission. A thread whose submission is absent from the dataset has root_in_dataset =
# FALSE and NA root fields. Author counts treat every stored author value (including
# '[deleted]') as a value; nothing is excluded.
summarise_threads <- function(events) {
  roots <- events |> filter(record_type == "submission") |>
    transmute(thread_id, root_in_dataset = TRUE, root_doc_key = doc_key, root_author = author, root_created_utc = created_utc,
              root_score = score, root_num_comments = num_comments, root_upvote_ratio = upvote_ratio)
  per_thread <- events |> filter(record_type == "comment") |> group_by(thread_id) |>
    summarise(n_comments = n(), n_comment_authors = n_distinct(author),
              n_comments_replying_to_root = sum(parent_kind %in% "submission"),
              n_comments_parent_in_dataset = sum(parent_in_dataset %in% TRUE),
              first_comment_utc = min(created_utc), last_comment_utc = max(created_utc), .groups = "drop")
  full_join(roots, per_thread, by = "thread_id") |>
    mutate(root_in_dataset = coalesce(root_in_dataset, FALSE),
           n_comments = coalesce(n_comments, 0L), n_comment_authors = coalesce(n_comment_authors, 0L),
           n_comments_replying_to_root = coalesce(n_comments_replying_to_root, 0L),
           n_comments_parent_in_dataset = coalesce(n_comments_parent_in_dataset, 0L),
           n_events = n_comments + as.integer(root_in_dataset)) |>
    select(thread_id, root_in_dataset, n_events, n_comments, n_comment_authors, n_comments_replying_to_root,
           n_comments_parent_in_dataset, first_comment_utc, last_comment_utc, root_doc_key, root_author, root_created_utc,
           root_score, root_num_comments, root_upvote_ratio) |>
    arrange(thread_id)
}

# One row per distinct stored author value ('[deleted]' included, as stored).
summarise_authors <- function(events) {
  events |> group_by(author) |>
    summarise(n_events = n(), n_comments = sum(record_type == "comment"), n_submissions = sum(record_type == "submission"),
              n_threads = n_distinct(thread_id), first_created_utc = min(created_utc), last_created_utc = max(created_utc),
              n_active_days = n_distinct(created_date_utc), .groups = "drop") |>
    arrange(desc(n_events), author)
}

# The linkage facts as counts (comments to submissions, comments to parents, threads, authors).
linkage_summary <- function(events) {
  cm <- events |> filter(record_type == "comment"); sb <- events |> filter(record_type == "submission")
  row <- function(item, value, basis) tibble(item = item, value = as.numeric(value), basis = basis)
  thread_basis <- "thread_id = link_id (comment) / 't3_' + id (submission); present = thread_id is a submission doc_key in the table"
  parent_basis <- "parent_id as stored, joined to doc_key; present = a row with that doc_key exists in the table"
  bind_rows(
    row("comments", nrow(cm), "record_type == 'comment'"),
    row("submissions", nrow(sb), "record_type == 'submission'"),
    row("comments whose submission is in the dataset", sum(cm$thread_root_in_dataset), thread_basis),
    row("comments whose submission is absent from the dataset", sum(!cm$thread_root_in_dataset), thread_basis),
    row("comments whose parent is in the dataset", sum(cm$parent_in_dataset %in% TRUE), parent_basis),
    row("comments whose parent is absent from the dataset", sum(cm$parent_in_dataset %in% FALSE), parent_basis),
    row("comments whose parent_id is missing", sum(is.na(cm$parent_id)), parent_basis),
    row("comments whose parent is a comment (t1_)", sum(cm$parent_kind %in% "comment"), "parent_id prefix"),
    row("comments whose parent is the submission (t3_)", sum(cm$parent_kind %in% "submission"), "parent_id prefix"),
    row("comments whose parent_id has another form", sum(cm$parent_kind %in% "other"), "parent_id prefix"),
    row("comments whose parent is a comment in the dataset", sum(cm$parent_kind %in% "comment" & cm$parent_in_dataset %in% TRUE), parent_basis),
    row("comments whose parent is a comment absent from the dataset", sum(cm$parent_kind %in% "comment" & cm$parent_in_dataset %in% FALSE), parent_basis),
    row("comments whose parent is the submission, in the dataset", sum(cm$parent_kind %in% "submission" & cm$parent_in_dataset %in% TRUE), parent_basis),
    row("comments whose parent is the submission, absent from the dataset", sum(cm$parent_kind %in% "submission" & cm$parent_in_dataset %in% FALSE), parent_basis),
    row("comments whose t3_ parent_id equals their own link_id", sum(cm$parent_kind %in% "submission" & cm$parent_id == cm$link_id), "parent_id == link_id"),
    row("threads represented (distinct thread_id)", n_distinct(events$thread_id), thread_basis),
    row("threads referenced by at least one comment", n_distinct(cm$thread_id), thread_basis),
    row("threads whose root submission is in the dataset", n_distinct(events$thread_id[events$thread_root_in_dataset]), thread_basis),
    row("threads whose root submission is absent from the dataset", n_distinct(events$thread_id[!events$thread_root_in_dataset]), thread_basis),
    row("threads referenced by comments whose root submission is in the dataset", n_distinct(cm$thread_id[cm$thread_root_in_dataset]), thread_basis),
    row("submissions with at least one comment in the dataset", sum(sb$doc_key %in% cm$thread_id), thread_basis),
    row("submissions with no comment in the dataset", sum(!sb$doc_key %in% cm$thread_id), thread_basis),
    row("distinct author values (all events)", n_distinct(events$author), "author as stored; '[deleted]' is a value"),
    row("distinct author values (comments)", n_distinct(cm$author), "author as stored"),
    row("distinct author values (submissions)", n_distinct(sb$author), "author as stored"),
    row("author values occurring in both record types", length(intersect(cm$author, sb$author)), "author as stored"))
}

# Coverage of every canonical field by record type: missing (NA), empty string, usable
# (neither), distinct usable values, and the observed range for numeric and time fields.
field_coverage <- function(events) {
  native <- c(link_id = "comment", parent_id = "comment", parent_kind = "comment", parent_in_dataset = "comment", body = "comment",
              num_comments = "submission", upvote_ratio = "submission", url = "submission", title = "submission", selftext = "submission")
  bind_rows(lapply(c("comment", "submission"), function(t) {
    d <- events[events$record_type == t, ]
    bind_rows(lapply(names(events), function(f) {
      x <- d[[f]]; miss <- is.na(x)
      empty <- if (is.character(x)) !miss & x == "" else rep(FALSE, length(x))
      usable <- !miss & !empty
      rng <- if (is.numeric(x) || inherits(x, c("Date", "POSIXct"))) { if (any(usable)) format(range(x[usable]), trim = TRUE) else c(NA, NA) } else c(NA_character_, NA_character_)
      tibble(field = f, record_type = t, applicable_to = if (f %in% names(native)) unname(native[f]) else "both", r_class = class(x)[1],
             n_records = length(x), n_missing = sum(miss), n_empty = sum(empty), n_usable = sum(usable),
             pct_usable = 100 * sum(usable) / length(x), n_distinct = n_distinct(x[usable]),
             n_true = if (is.logical(x)) sum(x %in% TRUE) else NA_integer_, n_false = if (is.logical(x)) sum(x %in% FALSE) else NA_integer_,
             min_value = rng[1], max_value = rng[2])
    }))
  }))
}

# Exact stored marker strings, counted per text field and record type; the empty string is
# counted as a stored value too. Nothing is excluded or relabelled.
count_platform_markers <- function(events, marker_strings = c("[deleted]", "[removed]", "[ Removed by Reddit ]", "[ Removed by moderator ]", "")) {
  fields <- list(comment = c("author", "body"), submission = c("author", "title", "selftext", "url"))
  bind_rows(lapply(names(fields), function(t) {
    d <- events[events$record_type == t, ]; n_t <- nrow(d)
    bind_rows(lapply(fields[[t]], function(f) {
      counts <- vapply(marker_strings, function(m) sum(d[[f]] %in% m), integer(1))
      tibble(record_type = t, field = f, stored_value = marker_strings,
             value_label = ifelse(marker_strings == "", "(empty string)", marker_strings),
             n = unname(counts), n_records = n_t, pct_of_record_type = 100 * unname(counts) / n_t)
    }))
  }))
}

# ---- Source files before reading: hashes, footer facts, declared schemas, ingest sidecars ----
# Hash every source file (the "before" hash), read footers and declared schemas without
# reading data, and parse the manifest.jsonl ingest sidecars for later comparison.
list_to_chr <- function(x) vapply(x, function(v) if (is.null(v) || length(v) == 0L) NA_character_ else paste(unlist(v), collapse = " "), character(1))
schema_of <- function(path, record_type) {
  s <- as.data.frame(read_parquet_schema(path)); s <- s[!is.na(s$type), , drop = FALSE]
  tibble(record_type = record_type, file = basename(path), column_index = seq_len(nrow(s)), field = as.character(s$name),
         parquet_type = as.character(s$type), parquet_logical_type = list_to_chr(s$logical_type),
         parquet_converted_type = as.character(s$converted_type), parquet_repetition = as.character(s$repetition_type),
         r_type_after_read = as.character(s$r_type))
}
file_facts <- function(path, record_type) {
  info <- read_parquet_info(path)
  tibble(record_type = record_type, file = basename(path), relative_path = relative_path(path, project_dir), bytes = file.size(path),
         sha256_before = sha256(path), footer_rows = as.numeric(info$num_rows), footer_columns = as.integer(info$num_cols),
         footer_row_groups = as.integer(info$num_row_groups), parquet_version = as.integer(info$parquet_version), created_by = as.character(info$created_by))
}
source_files  <- bind_rows(lapply(comment_files, file_facts, record_type = "comment"), lapply(submission_files, file_facts, record_type = "submission"))
source_schema <- bind_rows(lapply(comment_files, schema_of, record_type = "comment"), lapply(submission_files, schema_of, record_type = "submission"))
comment_signatures <- source_schema |> filter(record_type == "comment") |> group_by(file) |>
  summarise(signature = paste(field, parquet_type, coalesce(parquet_logical_type, ""), parquet_repetition, sep = ":", collapse = ";"), .groups = "drop")
comment_schemas_identical <- n_distinct(comment_signatures$signature) == 1L
ingest_manifest <- if (length(ingest_manifests)) bind_rows(lapply(ingest_manifests, function(f) {
  bind_rows(lapply(readLines(f, warn = FALSE, encoding = "UTF-8"), function(line) {
    j <- fromJSON(line, simplifyVector = TRUE); m <- as.data.frame(j$members)
    tibble(manifest_file = relative_path(f, project_dir), kind = j$kind, layout = j$layout, generation = as.integer(j$generation),
           file = j$path, manifest_bytes = as.numeric(j$bytes), manifest_rows = sum(m$rows),
           manifest_min_created_utc = if ("min_created_utc" %in% names(m)) min(m$min_created_utc) else NA_real_,
           manifest_max_created_utc = if ("max_created_utc" %in% names(m)) max(m$max_created_utc) else NA_real_,
           manifest_members = paste(m$subreddit, m$subreddit_id, sep = "=", collapse = "|"))
  }))
})) else tibble(manifest_file = character(), file = character(), manifest_bytes = numeric(), manifest_rows = numeric(),
                manifest_min_created_utc = numeric(), manifest_max_created_utc = numeric())
mark_stage("source_files")
log_line("source files:", nrow(source_files), "Parquet files hashed;", nrow(ingest_manifest), "ingest manifest rows")

# ---- Read the source records ----------------------------------------------------------
# All columns and rows, tagged with file and row number. Nothing is cast or dropped.
comments    <- read_source_records(comment_files, project_dir)
submissions <- read_source_records(submission_files, project_dir)
expected_comment_fields    <- c("id", "parent_id", "link_id", "author", "body", "created_utc", "retrieved_on", "score", "subreddit", "subreddit_id", "raw_json")
expected_submission_fields <- c("id", "title", "selftext", "url", "author", "created_utc", "retrieved_on", "score", "num_comments", "upvote_ratio", "subreddit", "subreddit_id", "raw_json")
source_fields <- list(comment = setdiff(names(comments), c("source_file", "source_row_in_file")), submission = setdiff(names(submissions), c("source_file", "source_row_in_file")))
rows_read <- bind_rows(comments |> group_by(relative_path = source_file) |> summarise(rows_read = n(), observed_min_created_utc = min(created_utc), observed_max_created_utc = max(created_utc), .groups = "drop"),
                       submissions |> group_by(relative_path = source_file) |> summarise(rows_read = n(), observed_min_created_utc = min(created_utc), observed_max_created_utc = max(created_utc), .groups = "drop"))
source_files <- source_files |> left_join(rows_read, by = "relative_path") |>
  left_join(ingest_manifest |> select(file, kind, layout, generation, manifest_bytes, manifest_rows, manifest_min_created_utc, manifest_max_created_utc, manifest_members), by = "file") |>
  mutate(rows_read_equal_footer = rows_read == footer_rows,
         manifest_rows_equal_rows_read = manifest_rows == rows_read, manifest_bytes_equal_file_bytes = manifest_bytes == bytes,
         manifest_time_range_equal_observed = (is.na(manifest_min_created_utc) & is.na(manifest_max_created_utc)) |
           (manifest_min_created_utc == observed_min_created_utc & manifest_max_created_utc == observed_max_created_utc))
n_c <- nrow(comments); n_s <- nrow(submissions)
mark_stage("read")
log_line("read:", n_c, "comments,", n_s, "submissions")

# ---- Build the canonical message-event table ------------------------------------------
# Rules are documented at build_message_events(); one row per source record, 29 columns.
events <- build_message_events(comments, submissions)
n_events <- nrow(events)
mark_stage("build")
log_line("built:", n_events, "events;", ncol(events), "columns")

# ---- Write the canonical table and prove the round trip -------------------------------
# Epoch and count fields are INT64 to match the source types; the key-value metadata is
# constant so identical inputs give identical bytes. The file is read back and compared
# column by column with the in-memory table.
# The codec level is passed explicitly: with the default (NA) level, nanoparquet 0.5.1 declares the zstd
# codec but stores every page uncompressed (verified on this dataset: 381 MB against 14.7 MB per 173k-row
# file at level 3). Levels apply to zstd and gzip; snappy and uncompressed ignore them.
write_options <- if (compression %in% c("zstd", "gzip")) parquet_options(compression_level = compression_level) else parquet_options()
if (nzchar(row_group_rows)) write_options$num_rows_per_row_group <- as.integer(row_group_rows)
write_parquet(events, events_path,
              schema = parquet_schema(created_utc = "INT64", retrieved_on = "INT64", score = "INT64", num_comments = "INT64"),
              compression = compression, options = write_options,
              metadata = c(pipeline = "pipeline, Phase 1: canonical Reddit message-event table",
                           one_row_per = "source record (submission or comment); nothing dropped",
                           doc_key = "'t1_' + comment id / 't3_' + submission id (Reddit fullname)",
                           thread_id = "comment: link_id; submission: 't3_' + id (= doc_key of the thread's submission)",
                           parent_id = "as stored; joins to doc_key",
                           text = "comment: body; submission: paste(title, selftext, sep = ' ')"))
events_back <- as_tibble(as.data.frame(read_parquet(events_path)))
written_schema <- local({ s <- as.data.frame(read_parquet_schema(events_path)); s <- s[!is.na(s$type), , drop = FALSE]
  tibble(field = as.character(s$name), parquet_type = as.character(s$type), parquet_logical_type = list_to_chr(s$logical_type),
         parquet_repetition = as.character(s$repetition_type), r_type_after_read = as.character(s$r_type)) })
compare_column <- function(a, b) {
  same_na <- identical(is.na(a), is.na(b)); ok <- !is.na(a) & !is.na(b)
  n_differ <- if (same_na) sum(a[ok] != b[ok]) else NA_integer_
  tibble(same_missing_pattern = same_na, n_compared = sum(ok), n_differ = n_differ, values_equal = same_na && n_differ == 0L)
}
parquet_roundtrip <- bind_rows(lapply(names(events), function(f) {
  bind_cols(tibble(field = f, r_class_written = class(events[[f]])[1], r_class_read = class(events_back[[f]])[1],
                   storage_mode_written = typeof(events[[f]]), storage_mode_read = typeof(events_back[[f]])),
            compare_column(events[[f]], events_back[[f]]))
})) |> left_join(written_schema, by = "field") |>
  mutate(same_class = r_class_written == r_class_read, pass = same_class & values_equal)
roundtrip_ok <- identical(names(events_back), names(events)) && nrow(events_back) == n_events && all(parquet_roundtrip$pass)
parquet_bytes <- file.size(events_path)
rm(events_back); invisible(gc())
mark_stage("write_and_roundtrip")
log_line("written:", relative_path(events_path, project_dir), sprintf("(%.1f MB);", parquet_bytes / 2^20), "round trip", if (roundtrip_ok) "exact" else "DIFFERS")

# ---- QC: counts, mapping, uniqueness, coverage, markers, linkage, threads, authors, time ----
# The QC measurements listed in the README, each over all records.
cm <- events |> filter(record_type == "comment"); sb <- events |> filter(record_type == "submission")
record_counts <- bind_rows(
  source_files |> transmute(item = paste0("source rows: ", file), record_type, value = rows_read),
  events |> count(record_type, source_file, name = "value") |> transmute(item = paste0("canonical events from: ", basename(source_file)), record_type, value = as.numeric(value)),
  tibble(item = c("source rows: all comment files", "source rows: all submission files", "source rows: total",
                  "canonical events: comments", "canonical events: submissions", "canonical events: total"),
         record_type = c("comment", "submission", "both", "comment", "submission", "both"),
         value = c(n_c, n_s, n_c + n_s, nrow(cm), nrow(sb), n_events)))
per_file_map <- source_files |> select(relative_path, record_type, rows_read) |>
  left_join(events |> group_by(source_file) |> summarise(events_from_file = n(), min_row = min(source_row_in_file), max_row = max(source_row_in_file), .groups = "drop"),
            by = c("relative_path" = "source_file")) |>
  mutate(events_from_file = coalesce(events_from_file, 0L), equal = rows_read == events_from_file,
         rows_within_bounds = !is.na(min_row) & min_row >= 1L & max_row <= rows_read)
n_distinct_pairs <- nrow(distinct(events, source_file, source_row_in_file))
doc_key_rule_holds <- events$doc_key == make_doc_key(events$record_type, events$reddit_id)
source_event_mapping <- tibble(
  check = c("canonical events equal source rows (comments + submissions)", "every source file contributes exactly its row count of events",
            "every (source_file, source_row_in_file) pair occurs exactly once", "every source_row_in_file lies within 1..rows of its file",
            "doc_key is unique", "doc_key equals 't1_' + id for every comment and 't3_' + id for every submission", "record_type is 'comment' or 'submission' for every event"),
  detail = c(sprintf("%d events, %d + %d source rows", n_events, n_c, n_s), sprintf("%d of %d files equal", sum(per_file_map$equal), nrow(per_file_map)),
             sprintf("%d distinct pairs of %d events", n_distinct_pairs, n_events),
             sprintf("row indices within bounds for %d of %d files", sum(per_file_map$rows_within_bounds), nrow(per_file_map)),
             sprintf("%d distinct doc_keys of %d events", n_distinct(events$doc_key), n_events),
             sprintf("%d of %d agree", sum(doc_key_rule_holds), n_events),
             paste(sort(unique(events$record_type)), collapse = ", ")),
  pass = c(n_events == n_c + n_s, all(per_file_map$equal), n_distinct_pairs == n_events, all(per_file_map$rows_within_bounds),
           n_distinct(events$doc_key) == n_events, all(doc_key_rule_holds), all(events$record_type %in% c("comment", "submission"))))
duplicate_id_checks <- tibble(
  check = c("comment reddit_id: distinct values equal comment count", "submission reddit_id: distinct values equal submission count",
            "no comment reddit_id also occurs as a submission reddit_id", "doc_key: no duplicates", "no missing reddit_id", "no missing doc_key"),
  n = c(n_distinct(cm$reddit_id), n_distinct(sb$reddit_id), length(intersect(cm$reddit_id, sb$reddit_id)), sum(duplicated(events$doc_key)), sum(is.na(events$reddit_id)), sum(is.na(events$doc_key))),
  denominator = c(nrow(cm), nrow(sb), nrow(cm), n_events, n_events, n_events),
  pass = c(n_distinct(cm$reddit_id) == nrow(cm), n_distinct(sb$reddit_id) == nrow(sb), length(intersect(cm$reddit_id, sb$reddit_id)) == 0L,
           !anyDuplicated(events$doc_key), !anyNA(events$reddit_id), !anyNA(events$doc_key)))
coverage <- field_coverage(events)
markers  <- count_platform_markers(events, marker_strings)
# Every fully bracketed stored value (whole value = '[' ... ']') occurring at least twice, per field: a
# factual listing of marker-like values as stored, without deciding what any of them means.
bracketed_values <- bind_rows(lapply(list(c("comment", "author"), c("comment", "body"), c("submission", "author"), c("submission", "title"), c("submission", "selftext")), function(p) {
  x <- events[[p[2]]][events$record_type == p[1]]; x <- x[!is.na(x) & stri_startswith_fixed(x, "[") & stri_endswith_fixed(x, "]")]
  if (!length(x)) return(NULL)
  tibble(value = x) |> count(value, name = "n") |> filter(n >= 2L) |> arrange(desc(n), value) |> transmute(record_type = p[1], field = p[2], stored_value = value, n)
}))
linkage <- linkage_summary(events)
parent_linkage <- cm |> count(parent_kind, parent_in_dataset, thread_root_in_dataset, name = "n") |>
  mutate(pct_of_comments = 100 * n / nrow(cm)) |> arrange(desc(n))
threads <- summarise_threads(events)
n_threads_total <- nrow(threads)   # scalar taken outside mutate(): inside it, the column named 'threads' would shadow the data frame
size_breaks <- c(-1, 0, 1, 2, 5, 10, 100, 1000, Inf); size_labels <- c("0", "1", "2", "3-5", "6-10", "11-100", "101-1000", "1001+")
thread_size_distribution <- threads |> mutate(band = cut(n_comments, breaks = size_breaks, labels = size_labels, right = TRUE)) |>
  group_by(band, root_in_dataset) |> summarise(threads = n(), comments = sum(n_comments), .groups = "drop") |>
  mutate(pct_of_threads = 100 * threads / n_threads_total) |> arrange(band, desc(root_in_dataset))
qtl <- function(x, p) quantile(x, p, type = 7, names = FALSE)
thread_size_summary <- bind_rows(
  tibble(population = "all threads represented", n_threads = nrow(threads), min = min(threads$n_comments), p25 = qtl(threads$n_comments, .25), median = qtl(threads$n_comments, .5), mean = mean(threads$n_comments), p75 = qtl(threads$n_comments, .75), p90 = qtl(threads$n_comments, .9), p99 = qtl(threads$n_comments, .99), max = max(threads$n_comments)),
  local({ t <- threads |> filter(n_comments > 0); tibble(population = "threads with at least one comment", n_threads = nrow(t), min = min(t$n_comments), p25 = qtl(t$n_comments, .25), median = qtl(t$n_comments, .5), mean = mean(t$n_comments), p75 = qtl(t$n_comments, .75), p90 = qtl(t$n_comments, .9), p99 = qtl(t$n_comments, .99), max = max(t$n_comments)) }),
  local({ t <- threads |> filter(root_in_dataset); tibble(population = "threads whose submission is in the dataset", n_threads = nrow(t), min = min(t$n_comments), p25 = qtl(t$n_comments, .25), median = qtl(t$n_comments, .5), mean = mean(t$n_comments), p75 = qtl(t$n_comments, .75), p90 = qtl(t$n_comments, .9), p99 = qtl(t$n_comments, .99), max = max(t$n_comments)) }),
  local({ t <- threads |> filter(!root_in_dataset); tibble(population = "threads whose submission is absent from the dataset", n_threads = nrow(t), min = min(t$n_comments), p25 = qtl(t$n_comments, .25), median = qtl(t$n_comments, .5), mean = mean(t$n_comments), p75 = qtl(t$n_comments, .75), p90 = qtl(t$n_comments, .9), p99 = qtl(t$n_comments, .99), max = max(t$n_comments)) }))
authors <- summarise_authors(events)
n_authors_total <- nrow(authors)   # scalar taken outside mutate(), for the same shadowing reason as n_threads_total
act_breaks <- c(0, 1, 2, 5, 10, 100, 1000, Inf); act_labels <- c("1", "2", "3-5", "6-10", "11-100", "101-1000", "1001+")
author_activity_distribution <- authors |> mutate(band = cut(n_events, breaks = act_breaks, labels = act_labels, right = TRUE)) |>
  group_by(band) |> summarise(authors = n(), events = sum(n_events), .groups = "drop") |>
  mutate(pct_of_authors = 100 * authors / n_authors_total, pct_of_events = 100 * events / n_events)
utc_chr <- function(x) format(as.POSIXct(x, origin = "1970-01-01", tz = "UTC"), "%Y-%m-%d %H:%M:%S")
timestamp_coverage <- bind_rows(lapply(list(list("comment", cm), list("submission", sb), list("all events", events)), function(p) {
  x <- p[[2]]$created_utc; d <- p[[2]]$created_date_utc; days <- seq(min(d), max(d), by = "day")
  tibble(record_type = p[[1]], field = "created_utc", n_records = length(x), n_missing = sum(is.na(x)),
         min_epoch = min(x), min_utc = utc_chr(min(x)), max_epoch = max(x), max_utc = utc_chr(max(x)),
         span_days = (max(x) - min(x)) / 86400, n_calendar_days_in_range = length(days), n_days_with_records = n_distinct(d),
         n_days_without_records = length(days) - n_distinct(d), all_whole_seconds = all(x == round(x)), n_distinct_values = n_distinct(x),
         retrieved_on_min_utc = utc_chr(min(p[[2]]$retrieved_on)), retrieved_on_max_utc = utc_chr(max(p[[2]]$retrieved_on)),
         n_retrieved_before_created = sum(p[[2]]$retrieved_on < x))
}))
records_by_day <- events |> count(created_date_utc, record_type, name = "n") |>
  pivot_wider(names_from = record_type, values_from = n, values_fill = 0L) |>
  mutate(total = comment + submission) |> rename(date_utc = created_date_utc) |> arrange(date_utc)
mark_stage("qc")
log_line("qc: coverage", nrow(coverage), "rows;", nrow(threads), "threads;", nrow(authors), "authors")

# ---- Reconciliation with the verified project artifacts -------------------------------
# Compare with the metadata inventory, the values in earlier reports (cited literals below)
# and the existing Tier 2 document table. Missing inputs are marked "not applicable" and
# differences are reported as found.
inventory_files <- c(identifier_structure = "identifier_structure.csv", thread_linkage = "thread_linkage.csv", parent_linkage = "parent_linkage.csv",
                     records_per_utc_day = "records_per_utc_day.csv", source_files = "source_files.csv")
inventory_paths <- setNames(file.path(inventory_dir, inventory_files), names(inventory_files))
inventory_present <- setNames(file.exists(inventory_paths), names(inventory_paths))   # file.exists() drops names
read_inv <- function(role) read_csv(inventory_paths[[role]], show_col_types = FALSE, progress = FALSE, trim_ws = FALSE)
inventory_reconciliation <- list()
if (inventory_present[["identifier_structure"]]) {
  inv <- read_inv("identifier_structure")
  m <- function(record_type, field, measure, value) tibble(record_type = record_type, field = field, measure = measure, measured_value = as.numeric(value))
  measured <- bind_rows(
    m("comment", "id", "matches ^[0-9a-z]+$ (bare base36, no prefix)", sum(grepl("^[0-9a-z]+$", cm$reddit_id))),
    m("comment", "id", "distinct values", n_distinct(cm$reddit_id)), m("comment", "id", "missing", sum(is.na(cm$reddit_id))),
    m("comment", "id", sprintf("character length %s", paste(range(stri_length(cm$reddit_id)), collapse = "-")), nrow(cm)),
    m("submission", "id", "matches ^[0-9a-z]+$ (bare base36, no prefix)", sum(grepl("^[0-9a-z]+$", sb$reddit_id))),
    m("submission", "id", "distinct values", n_distinct(sb$reddit_id)), m("submission", "id", "missing", sum(is.na(sb$reddit_id))),
    m("submission", "id", sprintf("character length %s", paste(range(stri_length(sb$reddit_id)), collapse = "-")), nrow(sb)),
    m("both", "id", "bare comment ids that also occur as a submission id", length(intersect(cm$reddit_id, sb$reddit_id))),
    m("comment", "link_id", "matches ^t3_[0-9a-z]+$", sum(grepl("^t3_[0-9a-z]+$", cm$link_id))),
    m("comment", "link_id", "distinct values (threads referenced)", n_distinct(cm$link_id)),
    m("comment", "link_id", "records whose thread (link_id without 't3_') is a submission id in the sample", sum(cm$thread_root_in_dataset)),
    m("comment", "link_id", "records whose thread has no submission in the sample", sum(!cm$thread_root_in_dataset)),
    m("comment", "link_id", "distinct threads with a submission in the sample", n_distinct(cm$thread_id[cm$thread_root_in_dataset])),
    m("comment", "link_id", "distinct threads without a submission in the sample", n_distinct(cm$thread_id[!cm$thread_root_in_dataset])),
    m("submission", "id", "submissions referenced by at least one comment link_id", sum(sb$doc_key %in% cm$thread_id)),
    m("submission", "id", "submissions referenced by no comment in the sample", sum(!sb$doc_key %in% cm$thread_id)),
    m("comment", "parent_id", "prefix t1_ (parent is a comment)", sum(cm$parent_kind %in% "comment")),
    m("comment", "parent_id", "prefix t3_ (parent is the submission)", sum(cm$parent_kind %in% "submission")),
    m("comment", "parent_id", "other prefix or missing", sum(!cm$parent_kind %in% c("comment", "submission"))),
    m("comment", "parent_id", "t3_ parent equal to the record's link_id", sum(cm$parent_kind %in% "submission" & cm$parent_id == cm$link_id)),
    m("comment", "parent_id", "t1_ parent comment present in the sample", sum(cm$parent_kind %in% "comment" & cm$parent_in_dataset %in% TRUE)),
    m("comment", "parent_id", "t1_ parent comment not in the sample", sum(cm$parent_kind %in% "comment" & cm$parent_in_dataset %in% FALSE)),
    m("comment", "parent_id", "t3_ parent submission present in the sample", sum(cm$parent_kind %in% "submission" & cm$parent_in_dataset %in% TRUE)),
    m("comment", "parent_id", "distinct values", n_distinct(cm$parent_id)),
    m("comment", "subreddit_id", "matches ^t5_[0-9a-z]+$", sum(grepl("^t5_[0-9a-z]+$", cm$subreddit_id))),
    m("submission", "subreddit_id", "matches ^t5_[0-9a-z]+$", sum(grepl("^t5_[0-9a-z]+$", sb$subreddit_id))),
    m("comment", "author", "distinct values", n_distinct(cm$author)), m("submission", "author", "distinct values", n_distinct(sb$author)),
    m("both", "author", "distinct comment authors that also occur as a submission author", length(intersect(cm$author, sb$author))))
  inventory_reconciliation$identifier_structure <- inv |> transmute(source_table = "identifier_structure.csv", record_type, field, measure, inventory_value = as.numeric(n)) |>
    full_join(measured, by = c("record_type", "field", "measure"))
}
if (inventory_present[["thread_linkage"]]) {
  inv <- read_inv("thread_linkage")
  diffs <- threads |> filter(root_in_dataset) |> mutate(d = root_num_comments - n_comments)
  measured <- tibble(measure = c("distinct threads referenced by comments (link_id without 't3_')", "threads with a submission in the sample", "threads without a submission in the sample",
                                 "comments on threads with a submission in the sample", "comments on threads without a submission in the sample", "submissions in the sample",
                                 "submissions referenced by at least one comment", "submissions referenced by no comment",
                                 "stored num_comments minus comments observed in the sample: minimum", "... p25", "... median", "... p75", "... maximum",
                                 "submissions whose observed comment count exceeds the stored num_comments", "submissions whose observed comment count equals the stored num_comments"),
                     measured_value = c(n_distinct(cm$thread_id), n_distinct(cm$thread_id[cm$thread_root_in_dataset]), n_distinct(cm$thread_id[!cm$thread_root_in_dataset]),
                                        sum(cm$thread_root_in_dataset), sum(!cm$thread_root_in_dataset), nrow(sb), sum(sb$doc_key %in% cm$thread_id), sum(!sb$doc_key %in% cm$thread_id),
                                        qtl(diffs$d, 0), qtl(diffs$d, .25), qtl(diffs$d, .5), qtl(diffs$d, .75), qtl(diffs$d, 1), sum(diffs$d < 0), sum(diffs$d == 0)))
  inventory_reconciliation$thread_linkage <- inv |> transmute(source_table = "thread_linkage.csv", record_type = NA_character_, field = NA_character_, measure = item, inventory_value = as.numeric(value)) |>
    full_join(measured, by = "measure")
}
if (inventory_present[["parent_linkage"]]) {
  inv <- read_inv("parent_linkage") |> filter(scope == "source: all comments") |> group_by(parent_record) |> summarise(inventory_value = sum(n), .groups = "drop")
  measured <- tibble(measure = c("parent comment in the sample", "parent comment not in the sample", "parent is the submission, in the sample", "parent is the submission, not in the sample"),
                     measured_value = c(sum(cm$parent_kind %in% "comment" & cm$parent_in_dataset %in% TRUE), sum(cm$parent_kind %in% "comment" & cm$parent_in_dataset %in% FALSE),
                                        sum(cm$parent_kind %in% "submission" & cm$parent_in_dataset %in% TRUE), sum(cm$parent_kind %in% "submission" & cm$parent_in_dataset %in% FALSE)))
  inventory_reconciliation$parent_linkage <- inv |> transmute(source_table = "parent_linkage.csv (scope: source, all comments)", record_type = "comment", field = "parent_id", measure = parent_record, inventory_value = as.numeric(inventory_value)) |>
    full_join(measured, by = "measure")
}
if (inventory_present[["records_per_utc_day"]]) {
  inv <- read_inv("records_per_utc_day")
  measured <- events |> count(record_type, created_date_utc, name = "measured_value") |> transmute(record_type, measure = paste("records on", format(created_date_utc)), measured_value = as.numeric(measured_value))
  inventory_reconciliation$records_per_utc_day <- inv |> transmute(source_table = "records_per_utc_day.csv", record_type, field = "created_utc", measure = paste("records on", format(as.Date(date))), inventory_value = as.numeric(n_source)) |>
    full_join(measured, by = c("record_type", "measure"))
}
if (inventory_present[["source_files"]]) {
  inv <- read_inv("source_files")
  measured <- bind_rows(source_files |> transmute(record_type, field = "file", measure = paste("footer rows of", file), measured_value = footer_rows),
                        source_files |> transmute(record_type, field = "file", measure = paste("sha256 equal to the inventory's for", file), measured_value = NA_real_, measured_text = sha256_before))
  inv_long <- bind_rows(inv |> transmute(record_type, field = "file", measure = paste("footer rows of", file), inventory_value = as.numeric(footer_rows), inventory_text = NA_character_),
                        inv |> transmute(record_type, field = "file", measure = paste("sha256 equal to the inventory's for", file), inventory_value = NA_real_, inventory_text = sha256))
  inventory_reconciliation$source_files <- inv_long |> mutate(source_table = "source_files.csv") |> full_join(measured, by = c("record_type", "field", "measure")) |>
    mutate(measured_value = if_else(is.na(inventory_value), as.numeric(measured_text == inventory_text), measured_value), inventory_value = if_else(is.na(inventory_value), 1, inventory_value)) |>
    select(-inventory_text, -measured_text)
}
inventory_reconciliation <- if (length(inventory_reconciliation)) bind_rows(inventory_reconciliation) |>
  mutate(equal = !is.na(inventory_value) & !is.na(measured_value) & abs(inventory_value - measured_value) < 1e-9) |>
  select(source_table, record_type, field, measure, inventory_value, measured_value, equal) else
  tibble(source_table = "metadata inventory tables not present", record_type = NA_character_, field = NA_character_, measure = NA_character_, inventory_value = NA_real_, measured_value = NA_real_, equal = NA)

# Values written in the earlier reports and READMEs, recomputed from the canonical table.
daily_c <- records_by_day$comment; daily_s <- records_by_day$submission
tpt <- threads |> filter(n_comments > 0)
documented_value_checks <- tribble(
  ~item, ~documented_value, ~documented_in, ~measured_value,
  "comment records", 891397, "sample_reconnaissance.md s1; sample_analysis.md; r_analysis_outputs/README.md", nrow(cm),
  "submission records", 10747, "sample_reconnaissance.md s1; sample_analysis.md", nrow(sb),
  "comment Parquet files", 7, "sample_reconnaissance.md s1", length(comment_files),
  "distinct comment authors", 135371, "sample_reconnaissance.md s8; metadata_inventory/README.md s3.1", n_distinct(cm$author),
  "distinct submission authors", 2321, "sample_reconnaissance.md s8; metadata_inventory/README.md s3.1", n_distinct(sb$author),
  "distinct threads referenced by comments", 11074, "sample_analysis.md definitions; r_analysis_outputs/README.md", n_distinct(cm$thread_id),
  "threads with a submission in the sample", 10308, "sample_analysis.md; r_analysis_outputs/README.md", n_distinct(cm$thread_id[cm$thread_root_in_dataset]),
  "comments on matched threads", 883733, "sample_analysis.md; r_analysis_outputs/README.md", sum(cm$thread_root_in_dataset),
  "threads without a submission in the sample", 766, "sample_reconnaissance.md s9; sample_analysis.md limitations", n_distinct(cm$thread_id[!cm$thread_root_in_dataset]),
  "comments on unmatched threads", 7664, "sample_analysis.md limitations; metadata_inventory/README.md s3.5", sum(!cm$thread_root_in_dataset),
  "submissions with zero comments in the sample", 439, "sample_reconnaissance.md s9", sum(!sb$doc_key %in% cm$thread_id),
  "comments with a t1_ parent", 541931, "metadata_inventory/README.md s3.1", sum(cm$parent_kind %in% "comment"),
  "comments with a t3_ parent", 349466, "metadata_inventory/README.md s3.1", sum(cm$parent_kind %in% "submission"),
  "t1_ parents present in the sample", 538485, "metadata_inventory/README.md s3.5", sum(cm$parent_kind %in% "comment" & cm$parent_in_dataset %in% TRUE),
  "t3_ parent equals the record's own link_id, every comment", 349466, "metadata_inventory/README.md s3.5", sum(cm$parent_kind %in% "submission" & cm$parent_id == cm$link_id),
  "comment body = '[removed]'", 27501, "sample_analysis.md s A.5", sum(cm$body %in% "[removed]"),
  "comment body = '[deleted]'", 3496, "sample_analysis.md s A.5", sum(cm$body %in% "[deleted]"),
  "comment body = '[ Removed by Reddit ]'", 1009, "sample_analysis.md s A.5", sum(cm$body %in% "[ Removed by Reddit ]"),
  "comment body empty string", 1, "sample_analysis.md s A.5", sum(cm$body %in% ""),
  "comment author = '[deleted]'", 30996, "sample_analysis.md s A.5", sum(cm$author %in% "[deleted]"),
  "submission author = '[deleted]'", 337, "sample_analysis.md s A.5", sum(sb$author %in% "[deleted]"),
  "submission title = '[ Removed by moderator ]'", 368, "sample_analysis.md s A.5", sum(sb$title %in% "[ Removed by moderator ]"),
  "submission title = '[ Removed by Reddit ]'", 17, "sample_analysis.md s A.5", sum(sb$title %in% "[ Removed by Reddit ]"),
  "submission selftext empty string", 10335, "sample_analysis.md s A.5", sum(sb$selftext %in% ""),
  "submission selftext = '[removed]'", 339, "sample_analysis.md s A.5", sum(sb$selftext %in% "[removed]"),
  "submission selftext = '[deleted]'", 47, "sample_analysis.md s A.5", sum(sb$selftext %in% "[deleted]"),
  "submission url empty string", 337, "sample_analysis.md s A.6", sum(sb$url %in% ""),
  "raw_json missing in every comment", 891397, "sample_reconnaissance.md s5", sum(is.na(cm$raw_json)),
  "raw_json missing in every submission", 10747, "sample_reconnaissance.md s5", sum(is.na(sb$raw_json)),
  "earliest comment created_utc (2026-07-01 00:00:02 UTC)", 1782864002, "sample_reconnaissance.md s4; comments manifest.jsonl", min(cm$created_utc),
  "latest comment created_utc (2026-07-31 23:59:59 UTC)", 1785542399, "sample_reconnaissance.md s4; comments manifest.jsonl", max(cm$created_utc),
  "earliest submission created_utc (2026-07-01 00:02:20 UTC)", 1782864140, "sample_reconnaissance.md s4; metadata_inventory source_files.csv", min(sb$created_utc),
  "latest submission created_utc (2026-07-31 23:58:02 UTC)", 1785542282, "sample_reconnaissance.md s4; metadata_inventory source_files.csv", max(sb$created_utc),
  "comment score minimum", -894, "sample_reconnaissance.md s3", min(cm$score),
  "comment score maximum", 32868, "sample_reconnaissance.md s3", max(cm$score),
  "submission score minimum", 0, "sample_reconnaissance.md s3", min(sb$score),
  "submission score maximum", 42370, "sample_reconnaissance.md s3", max(sb$score),
  "submission num_comments maximum", 4858, "sample_reconnaissance.md s3", max(sb$num_comments),
  "submission upvote_ratio minimum", 0.01, "sample_reconnaissance.md s3", min(sb$upvote_ratio),
  "submission upvote_ratio maximum", 1, "sample_reconnaissance.md s3", max(sb$upvote_ratio),
  "comments per day: maximum", 45099, "sample_analysis.md s B.1; r_analysis_outputs/README.md", max(daily_c),
  "comments per day: minimum", 18669, "sample_analysis.md s B.1; r_analysis_outputs/README.md", min(daily_c),
  "submissions per day: minimum", 210, "sample_analysis.md s B.1; metadata_inventory/README.md s3.3", min(daily_s),
  "submissions per day: maximum", 420, "sample_analysis.md s B.1; metadata_inventory/README.md s3.3", max(daily_s),
  "comments per thread (threads with comments): median", 14, "sample_analysis.md s B.2; r_analysis_outputs/README.md", qtl(tpt$n_comments, .5),
  "comments per thread (threads with comments): maximum", 5124, "sample_analysis.md s B.2; r_analysis_outputs/README.md", max(tpt$n_comments),
  "rows with retrieved_on before created_utc, comments", 0, "sample_analysis.md s B.0", sum(cm$retrieved_on < cm$created_utc),
  "rows with retrieved_on before created_utc, submissions", 0, "sample_analysis.md s B.0", sum(sb$retrieved_on < sb$created_utc),
  "UTC calendar days with comments", 31, "metadata_inventory/README.md s3.3", n_distinct(cm$created_date_utc),
  "UTC calendar days with submissions", 31, "metadata_inventory/README.md s3.3", n_distinct(sb$created_date_utc)) |>
  mutate(equal = abs(measured_value - documented_value) < 1e-9)

# Full-population cross-check against the Tier 2 document table from the earlier run (git-ignored,
# optional): it keyed the same 902,144 records by the same doc_key and recorded author,
# created_utc, link_id, parent_id, submission_id, submission_in_sample and n_chars.
tier2_present <- file.exists(tier2_table)
tier2_crosscheck <- if (tier2_present) {
  t2 <- read_csv(tier2_table, col_select = all_of(c("doc_key", "doc_id", "doc_type", "author", "created_utc", "submission_id", "link_id", "parent_id", "submission_in_sample", "n_chars")),
                 col_types = cols(doc_key = col_character(), doc_id = col_character(), doc_type = col_character(), author = col_character(), created_utc = col_double(),
                                  submission_id = col_character(), link_id = col_character(), parent_id = col_character(), submission_in_sample = col_logical(), n_chars = col_integer()),
                 na = "NA", trim_ws = FALSE, progress = FALSE, show_col_types = FALSE, lazy = FALSE)
  j <- events |> select(doc_key, record_type, reddit_id, author, created_utc, thread_id, link_id, parent_id, thread_root_in_dataset, text_n_chars) |>
    inner_join(t2, by = "doc_key", suffix = c("", "_t2"))
  agree <- function(a, b) sum((is.na(a) & is.na(b)) | (!is.na(a) & !is.na(b) & a == b))
  bind_rows(
    tibble(item = "Tier 2 document rows", value = nrow(t2), expected = n_events),
    tibble(item = "doc_keys in both tables", value = nrow(j), expected = n_events),
    tibble(item = "Tier 2 doc_keys absent from the canonical table", value = sum(!t2$doc_key %in% events$doc_key), expected = 0),
    tibble(item = "canonical doc_keys absent from the Tier 2 table", value = sum(!events$doc_key %in% t2$doc_key), expected = 0),
    tibble(item = "record_type equals Tier 2 doc_type", value = agree(j$record_type, j$doc_type), expected = nrow(j)),
    tibble(item = "reddit_id equals Tier 2 doc_id", value = agree(j$reddit_id, j$doc_id), expected = nrow(j)),
    tibble(item = "author equals Tier 2 author", value = agree(j$author, j$author_t2), expected = nrow(j)),
    tibble(item = "created_utc equals Tier 2 created_utc", value = agree(j$created_utc, j$created_utc_t2), expected = nrow(j)),
    tibble(item = "thread_id equals Tier 2 link_id (comments: as stored; submissions: 't3_' + id)", value = agree(j$thread_id, j$link_id_t2), expected = nrow(j)),
    tibble(item = "thread_id without 't3_' equals Tier 2 submission_id", value = agree(substring(j$thread_id, 4L), j$submission_id), expected = nrow(j)),
    tibble(item = "parent_id equals Tier 2 parent_id (NA for submissions in both)", value = agree(j$parent_id, j$parent_id_t2), expected = nrow(j)),
    tibble(item = "thread_root_in_dataset equals Tier 2 submission_in_sample", value = agree(j$thread_root_in_dataset, j$submission_in_sample), expected = nrow(j)),
    tibble(item = "text_n_chars equals Tier 2 n_chars (same text rule; proves the texts are unchanged)", value = agree(j$text_n_chars, j$n_chars), expected = nrow(j))) |>
    mutate(equal = value == expected)
} else tibble(item = "Tier 2 document table not present; cross-check not applicable", value = NA_real_, expected = NA_real_, equal = NA)
if (tier2_present) { rm(t2, j); invisible(gc()) }
mark_stage("reconciliation")
log_line("reconciliation:", sum(inventory_reconciliation$equal, na.rm = TRUE), "of", sum(!is.na(inventory_reconciliation$equal)), "inventory values equal;",
         sum(documented_value_checks$equal), "of", nrow(documented_value_checks), "documented values reproduced;",
         if (tier2_present) sprintf("Tier 2 cross-check %d of %d equal", sum(tier2_crosscheck$equal), nrow(tier2_crosscheck)) else "Tier 2 table absent")

# ---- Figures --------------------------------------------------------------------------
# Four QC figures under qc/figures/.
qc_theme <- theme_minimal(base_size = 11) +
  theme(panel.grid.minor = element_blank(), panel.grid.major = element_line(colour = "#e1e0d9", linewidth = 0.3),
        axis.text = element_text(colour = "#52514e"), axis.title = element_text(colour = "#0b0b0b"),
        plot.title = element_text(colour = "#0b0b0b", size = 12), strip.text = element_text(colour = "#0b0b0b", size = 11),
        plot.background = element_rect(fill = "#fcfcfb", colour = NA), panel.background = element_rect(fill = "#fcfcfb", colour = NA))
ink <- "#2a78d6"
save_fig <- function(name, plot, width, height) ggsave(file.path(fig_dir, name), plot, width = width, height = height, dpi = 150, bg = "#fcfcfb")

daily_long <- records_by_day |> select(date_utc, comment, submission) |> pivot_longer(c(comment, submission), names_to = "record_type", values_to = "records")
fig1 <- ggplot(daily_long, aes(x = date_utc, y = records)) + geom_col(fill = ink, width = 0.8) +
  scale_y_continuous(labels = label_comma(), expand = expansion(mult = c(0, 0.05))) +
  scale_x_date(date_breaks = "5 days", date_labels = "%b %d") +
  facet_wrap(~ record_type, ncol = 1, scales = "free_y") +
  labs(title = "Canonical events per UTC day of created_utc, by record type", x = "UTC date, July 2026", y = "events") + qc_theme
save_fig("fig1_events_per_utc_day_by_record_type.png", fig1, 9, 6)

fig2 <- ggplot(threads, aes(x = n_comments + 1)) + geom_histogram(bins = 60, fill = ink, colour = NA) +
  scale_x_continuous(transform = transform_log10(), breaks = c(0, 1, 10, 100, 1000) + 1, labels = c("0", "1", "10", "100", "1,000")) +
  scale_y_continuous(labels = label_comma(), expand = expansion(mult = c(0, 0.05))) +
  labs(title = "Comments per thread, all threads represented", x = "comments in the thread (log axis; plotted as count + 1)", y = "threads") + qc_theme
save_fig("fig2_thread_size_distribution.png", fig2, 8, 4.5)

author_ff <- authors |> count(n_events, name = "authors")
fig3 <- ggplot(author_ff, aes(x = n_events, y = authors)) + geom_point(colour = ink, size = 1.6, alpha = 0.8) +
  scale_x_log10(labels = label_comma()) + scale_y_log10(labels = label_comma()) +
  labs(title = "Events per author value, all events", x = "events by the author value (log axis)", y = "author values with exactly that many events (log axis)") + qc_theme
save_fig("fig3_author_activity_distribution.png", fig3, 8, 5)

link_df <- tibble(relation = factor(rep(c("submission of the comment", "parent of the comment"), each = 2), levels = c("submission of the comment", "parent of the comment")),
                  status = factor(rep(c("in the dataset", "absent from the dataset"), 2), levels = c("in the dataset", "absent from the dataset")),
                  comments = c(sum(cm$thread_root_in_dataset), sum(!cm$thread_root_in_dataset), sum(cm$parent_in_dataset %in% TRUE), sum(cm$parent_in_dataset %in% FALSE)))
fig4 <- ggplot(link_df, aes(x = status, y = comments)) + geom_col(fill = ink, width = 0.6) +
  scale_y_continuous(labels = label_comma(), expand = expansion(mult = c(0, 0.05))) +
  facet_wrap(~ relation, ncol = 2) +
  labs(title = "Linkage coverage of the comments", x = NULL, y = "comments") + qc_theme
save_fig("fig4_linkage_coverage.png", fig4, 8, 4.2)
mark_stage("figures")

# ---- Source files after the run: hashes unchanged -------------------------------------
# Hash the source files again; compare with the "before" hashes and, when present, the
# metadata inventory's hashes.
raw_data_integrity <- source_files |> transmute(record_type, file, relative_path, bytes, sha256_before, sha256_after = vapply(file.path(project_dir, relative_path), sha256, character(1)),
                                                unchanged_during_run = sha256_before == sha256_after,
                                                sha256_in_metadata_inventory = if (inventory_present[["source_files"]]) { inv <- read_inv("source_files"); inv$sha256[match(file, inv$file)] } else NA_character_) |>
  mutate(equal_to_metadata_inventory = if (inventory_present[["source_files"]]) sha256_after == sha256_in_metadata_inventory else NA)
raw_unchanged <- all(raw_data_integrity$unchanged_during_run)
mark_stage("raw_hash_after")

# ---- Reconciliation checks: every requirement as a pass / fail row ----------------------
# A failed check is also written to discrepancies.csv.
checks <- list()
add_check <- function(check, detail, pass, applicable = TRUE) checks[[length(checks) + 1L]] <<- tibble(check = check, detail = detail, status = if (!applicable) "not applicable" else if (isTRUE(pass)) "pass" else "fail", pass = if (!applicable) NA else isTRUE(pass))
add_check("rows read equal the Parquet footer row counts for every source file", paste(sprintf("%s: %d read / %d footer", source_files$file, source_files$rows_read, source_files$footer_rows), collapse = "; "), all(source_files$rows_read_equal_footer))
add_check("ingest manifest.jsonl rows, bytes and time ranges equal the files' contents", sprintf("%d of %d files with a manifest row: rows equal %d, bytes equal %d, time ranges equal %d", sum(!is.na(source_files$manifest_rows)), nrow(source_files), sum(source_files$manifest_rows_equal_rows_read, na.rm = TRUE), sum(source_files$manifest_bytes_equal_file_bytes, na.rm = TRUE), sum(source_files$manifest_time_range_equal_observed, na.rm = TRUE)),
          all(source_files$manifest_rows_equal_rows_read, na.rm = TRUE) && all(source_files$manifest_bytes_equal_file_bytes, na.rm = TRUE) && all(source_files$manifest_time_range_equal_observed, na.rm = TRUE), applicable = any(!is.na(source_files$manifest_rows)))
add_check("the comment files declare identical schemas; both record types have the expected fields", sprintf("%d distinct comment schema signatures; comment fields %s; submission fields %s", n_distinct(comment_signatures$signature), if (setequal(source_fields$comment, expected_comment_fields)) "as expected" else paste(source_fields$comment, collapse = ","), if (setequal(source_fields$submission, expected_submission_fields)) "as expected" else paste(source_fields$submission, collapse = ",")),
          comment_schemas_identical && setequal(source_fields$comment, expected_comment_fields) && setequal(source_fields$submission, expected_submission_fields))
add_check("every source record maps to exactly one canonical event and every event to one source record", paste(source_event_mapping$detail[c(1, 2, 3)], collapse = "; "), all(source_event_mapping$pass))
add_check("identifiers: unique reddit_id per record type, no cross-type overlap, unique doc_key, nothing missing", paste(sprintf("%s: %d / %d", duplicate_id_checks$check, duplicate_id_checks$n, duplicate_id_checks$denominator), collapse = "; "), all(duplicate_id_checks$pass))
add_check("thread identity: every submission's thread_id equals its doc_key; every comment's thread_id equals its stored link_id", sprintf("%d of %d submissions; %d of %d comments", sum(sb$thread_id == sb$doc_key), nrow(sb), sum(cm$thread_id == cm$link_id), nrow(cm)), all(sb$thread_id == sb$doc_key) && all(cm$thread_id == cm$link_id))
thread_join <- cm |> select(doc_key, thread_id, thread_root_in_dataset) |> left_join(sb |> transmute(thread_id = doc_key, root_found = TRUE), by = "thread_id") |> mutate(root_found = coalesce(root_found, FALSE))
add_check("thread_root_in_dataset equals the result of joining thread_id to the submissions' doc_key, for every comment", sprintf("%d of %d agree; %d comments have their submission, %d do not", sum(thread_join$root_found == thread_join$thread_root_in_dataset), nrow(cm), sum(thread_join$root_found), sum(!thread_join$root_found)), all(thread_join$root_found == thread_join$thread_root_in_dataset))
parent_join <- cm |> select(doc_key, parent_id, parent_kind, parent_in_dataset) |> left_join(events |> transmute(parent_id = doc_key, parent_record_type = record_type), by = "parent_id") |>
  mutate(found = !is.na(parent_record_type), kind_consistent = !found | parent_record_type == parent_kind)
add_check("parent_in_dataset equals the result of joining parent_id to doc_key, and the found parent's record type matches parent_kind, for every comment", sprintf("%d of %d presence flags agree; %d of %d found parents have the kind their prefix says", sum(parent_join$found == parent_join$parent_in_dataset), nrow(cm), sum(parent_join$kind_consistent & parent_join$found), sum(parent_join$found)), all(parent_join$found == parent_join$parent_in_dataset) && all(parent_join$kind_consistent))
add_check("every t3_ parent_id equals the comment's own link_id", sprintf("%d of %d", sum(cm$parent_kind %in% "submission" & cm$parent_id == cm$link_id), sum(cm$parent_kind %in% "submission")), all(cm$parent_id[cm$parent_kind %in% "submission"] == cm$link_id[cm$parent_kind %in% "submission"]))
add_check("original text fields are identical to the source, record by record (body; title; selftext)", sprintf("body %d of %d; title %d of %d; selftext %d of %d", sum(cm$body == comments$body), n_c, sum(sb$title == submissions$title), n_s, sum(sb$selftext == submissions$selftext), n_s), identical(cm$body, comments$body) && identical(sb$title, submissions$title) && identical(sb$selftext, submissions$selftext))
add_check("author, created_utc, retrieved_on, score, url, num_comments, upvote_ratio, subreddit fields are identical to the source, record by record",
          sprintf("comments: author %d, created_utc %d, retrieved_on %d, score %d of %d; submissions: author %d, created_utc %d, score %d, url %d, num_comments %d, upvote_ratio %d of %d", sum(cm$author == comments$author), sum(cm$created_utc == comments$created_utc), sum(cm$retrieved_on == comments$retrieved_on), sum(cm$score == comments$score), n_c, sum(sb$author == submissions$author), sum(sb$created_utc == submissions$created_utc), sum(sb$score == submissions$score), sum(sb$url == submissions$url), sum(sb$num_comments == submissions$num_comments), sum(sb$upvote_ratio == submissions$upvote_ratio), n_s),
          identical(cm$author, comments$author) && all(cm$created_utc == comments$created_utc) && all(cm$retrieved_on == comments$retrieved_on) && all(cm$score == comments$score) && identical(cm$link_id, comments$link_id) && identical(cm$parent_id, comments$parent_id) && identical(cm$subreddit, comments$subreddit) && identical(cm$subreddit_id, comments$subreddit_id) &&
            identical(sb$author, submissions$author) && all(sb$created_utc == submissions$created_utc) && all(sb$retrieved_on == submissions$retrieved_on) && all(sb$score == submissions$score) && identical(sb$url, submissions$url) && all(sb$num_comments == submissions$num_comments) && all(sb$upvote_ratio == submissions$upvote_ratio) && identical(sb$subreddit, submissions$subreddit) && identical(sb$subreddit_id, submissions$subreddit_id))
add_check("text follows the stated rule for every event and text_n_chars is its code-point length", sprintf("%d of %d comments text == body; %d of %d submissions text == title + ' ' + selftext; %d of %d lengths agree", sum(cm$text == cm$body), nrow(cm), sum(sb$text == paste(sb$title, sb$selftext, sep = " ")), nrow(sb), sum(events$text_n_chars == stri_length(events$text)), n_events),
          all(cm$text == cm$body) && all(sb$text == paste(sb$title, sb$selftext, sep = " ")) && all(events$text_n_chars == stri_length(events$text)))
add_check("derived time fields agree with created_utc for every event", sprintf("%d of %d date-times; %d of %d dates", sum(as.numeric(events$created_datetime_utc) == events$created_utc), n_events, sum(events$created_date_utc == as.Date(as.POSIXct(events$created_utc, origin = "1970-01-01", tz = "UTC"), tz = "UTC")), n_events),
          all(as.numeric(events$created_datetime_utc) == events$created_utc) && all(events$created_date_utc == as.Date(as.POSIXct(events$created_utc, origin = "1970-01-01", tz = "UTC"), tz = "UTC")))
add_check("the written Parquet file reads back identical to the in-memory table in every column (names, order, classes, missing pattern, values)", sprintf("%d of %d columns pass; %d rows", sum(parquet_roundtrip$pass), nrow(parquet_roundtrip), n_events), roundtrip_ok)
add_check("coverage identities: usable + missing + empty = records for every field row", sprintf("%d rows", nrow(coverage)), all(coverage$n_usable + coverage$n_missing + coverage$n_empty == coverage$n_records))
add_check("daily counts sum to the record counts", sprintf("%d comments, %d submissions over %d UTC days", sum(records_by_day$comment), sum(records_by_day$submission), nrow(records_by_day)), sum(records_by_day$comment) == nrow(cm) && sum(records_by_day$submission) == nrow(sb))
add_check("thread and author summaries account for every event", sprintf("threads: %d events; authors: %d events", sum(threads$n_events), sum(authors$n_events)), sum(threads$n_events) == n_events && sum(authors$n_events) == n_events && sum(threads$n_comments) == nrow(cm))
add_check("no reddit_id or author equals the literal string 'NA' (so NA in the CSV outputs is unambiguous)", sprintf("%d ids, %d authors equal 'NA'", sum(events$reddit_id == "NA"), sum(events$author == "NA")), sum(events$reddit_id == "NA") + sum(events$author == "NA") == 0)
add_check("every value in the metadata-inventory tables that this phase can recompute is reproduced", sprintf("%d of %d compared values equal; %d inventory rows without a measured counterpart", sum(inventory_reconciliation$equal, na.rm = TRUE), sum(!is.na(inventory_reconciliation$equal)), sum(is.na(inventory_reconciliation$measured_value) & !is.na(inventory_reconciliation$inventory_value))),
          all(inventory_reconciliation$equal) && !any(is.na(inventory_reconciliation$measured_value) & !is.na(inventory_reconciliation$inventory_value)), applicable = any(inventory_present))
add_check("every value documented in the earlier reports is reproduced", sprintf("%d of %d", sum(documented_value_checks$equal), nrow(documented_value_checks)), all(documented_value_checks$equal))
add_check("the existing Tier 2 document table keys the same records and agrees on every carried field and on text length", if (tier2_present) sprintf("%d of %d items equal", sum(tier2_crosscheck$equal), nrow(tier2_crosscheck)) else "table not present", all(tier2_crosscheck$equal), applicable = tier2_present)
add_check("source files unchanged during the run (SHA256 before = after)", sprintf("%d of %d files unchanged", sum(raw_data_integrity$unchanged_during_run), nrow(raw_data_integrity)), raw_unchanged)
add_check("source file hashes equal those recorded by the metadata inventory", sprintf("%d of %d equal", sum(raw_data_integrity$equal_to_metadata_inventory, na.rm = TRUE), nrow(raw_data_integrity)), all(raw_data_integrity$equal_to_metadata_inventory), applicable = inventory_present[["source_files"]])
reconciliation_checks <- bind_rows(checks)
discrepancies <- bind_rows(
  reconciliation_checks |> filter(status == "fail") |> transmute(kind = "check failed", item = check, detail = detail),
  inventory_reconciliation |> filter(!is.na(equal) & !equal) |> transmute(kind = "inventory value not reproduced", item = paste(source_table, coalesce(record_type, ""), coalesce(field, ""), measure), detail = sprintf("inventory %s; measured %s", format(inventory_value), format(measured_value))),
  documented_value_checks |> filter(!equal) |> transmute(kind = "documented value not reproduced", item = item, detail = sprintf("documented %s (%s); measured %s", format(documented_value), documented_in, format(measured_value))),
  tier2_crosscheck |> filter(!is.na(equal) & !equal) |> transmute(kind = "Tier 2 cross-check differs", item = item, detail = sprintf("%s of %s", format(value), format(expected))))
if (nrow(discrepancies) == 0L) discrepancies <- tibble(kind = "none", item = "no discrepancy found", detail = sprintf("%d checks passed, %d not applicable; %d inventory values and %d documented values reproduced", sum(reconciliation_checks$status == "pass"), sum(reconciliation_checks$status == "not applicable"), sum(inventory_reconciliation$equal, na.rm = TRUE), sum(documented_value_checks$equal)))
print(as.data.frame(reconciliation_checks[, c("check", "status")]), right = FALSE)
mark_stage("checks")

# ---- Environment, runtime and resource record ------------------------------------------
# What was run, with which R and packages, stage timings, and memory and CPU use.
git_commit <- tryCatch(system2("git", c("-C", project_dir, "rev-parse", "--short", "HEAD"), stdout = TRUE, stderr = FALSE), error = function(e) "")
if (length(git_commit) == 0L) git_commit <- ""
used_packages <- c("nanoparquet", "dplyr", "tidyr", "stringi", "readr", "ggplot2", "scales", "digest", "yaml", "jsonlite", "ps")
r_environment <- bind_rows(tibble(component = "R", version = R.version.string, source = "R.version.string"),
                           tibble(component = paste0("package: ", used_packages), version = vapply(used_packages, function(p) as.character(packageVersion(p)), character(1)), source = "packageVersion()"),
                           tibble(component = "OS", version = paste(Sys.info()[["sysname"]], Sys.info()[["release"]], Sys.info()[["version"]]), source = "Sys.info()"),
                           tibble(component = "platform", version = R.version$platform, source = "R.version$platform"),
                           tibble(component = "locale (LC_CTYPE)", version = Sys.getlocale("LC_CTYPE"), source = "Sys.getlocale()"),
                           tibble(component = "logical CPUs", version = as.character(parallel::detectCores()), source = "parallel::detectCores()"))
writeLines(c(capture.output(print(sessionInfo())), "", "Packages used by the script:", paste0("  ", used_packages, " ", vapply(used_packages, function(p) as.character(packageVersion(p)), character(1)))), file.path(env_dir, "session_info.txt"))
run_end <- Sys.time(); cpu <- proc.time() - cpu_start
gc_now <- gc(); mem_now <- tryCatch(ps::ps_memory_info(), error = function(e) NULL)
sys_mem <- tryCatch(ps::ps_system_memory(), error = function(e) NULL)
resource_use <- tibble(
  measure = c("wall clock seconds (whole script)", "CPU seconds, user (R process and children)", "CPU seconds, system", "peak working set of the R process, MB (Windows PeakWorkingSetSize via ps)",
              "working set at the end, MB", "R allocator peak, MB (gc() max used, Vcells + Ncells, reset at start)", "source Parquet bytes", "canonical Parquet bytes", "canonical Parquet rows", "canonical Parquet columns",
              "logical CPUs on this machine", "physical memory on this machine, MB", "records processed per wall-clock second"),
  value = c(round(as.numeric(difftime(run_end, run_start, units = "secs")), 1), round(sum(cpu[c("user.self", "user.child")], na.rm = TRUE), 1), round(sum(cpu[c("sys.self", "sys.child")], na.rm = TRUE), 1),
            if (!is.null(mem_now) && "peak_wset" %in% names(mem_now)) round(unname(mem_now[["peak_wset"]]) / 2^20, 1) else NA_real_,
            if (!is.null(mem_now)) round(unname(mem_now[["rss"]]) / 2^20, 1) else NA_real_, round(sum(gc_now[, "max used"] * c(56, 8) / 2^20), 1),
            sum(source_files$bytes), parquet_bytes, n_events, ncol(events), parallel::detectCores(), if (!is.null(sys_mem)) round(sys_mem$total / 2^20) else NA_real_,
            round(n_events / as.numeric(difftime(run_end, run_start, units = "secs")))),
  note = c("run_start to the end of the environment stage", "proc.time()", "proc.time()", "single-process run; no parallel workers", "", "gc() reports cells; converted at 56 bytes per Ncell and 8 bytes per Vcell",
           "8 files", "derived/message_events.parquet", "", "", "", "ps::ps_system_memory()", "canonical events / wall seconds"))
stage_timings <- bind_rows(stage_log)
run_info <- tibble(item = c("pipeline", "phase", "script", "script_sha256", "run_start_utc", "run_end_utc", "elapsed_seconds", "git_commit_at_run", "project_dir", "comments_dir", "submissions_dir",
                            "compression", "compression_level", "row_group_rows", "inventory_dir", "inventory_tables_present", "tier2_table", "tier2_table_present",
                            "source_records", "canonical_events", "comments", "submissions", "threads_represented", "distinct_authors", "checks_passed", "checks_not_applicable", "checks_failed"),
                   value = c("pipeline", "1: message-event foundation", relative_path(script_path, project_dir), sha256(script_path),
                             format(run_start, "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"), format(run_end, "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"), sprintf("%.1f", as.numeric(difftime(run_end, run_start, units = "secs"))),
                             git_commit, project_dir, relative_path(comments_dir, project_dir), relative_path(submissions_dir, project_dir), compression, as.character(compression_level), if (nzchar(row_group_rows)) row_group_rows else "nanoparquet default (one row group)",
                             relative_path(inventory_dir, project_dir), paste(names(inventory_present)[inventory_present], collapse = ", "), relative_path(tier2_table, project_dir), as.character(tier2_present),
                             n_c + n_s, n_events, nrow(cm), nrow(sb), nrow(threads), nrow(authors), sum(reconciliation_checks$status == "pass"), sum(reconciliation_checks$status == "not applicable"), sum(reconciliation_checks$status == "fail")))
parameters <- tibble(parameter = c("doc_key_rule", "record_type_rule", "thread_id_rule", "thread_root_in_dataset_rule", "parent_id_rule", "parent_kind_rule", "parent_in_dataset_rule", "text_rule", "text_n_chars_rule", "time_rules", "row_order", "marker_strings", "parquet_reader", "parquet_writer", "compression", "compression_level", "int64_fields"),
                     value = c("'t1_' + comment id; 't3_' + submission id (Reddit fullname; the project's established document key)", "'comment' for rows from the comment files; 'submission' for rows from the submission file",
                               "comment: link_id as stored; submission: 't3_' + id; equal to the doc_key of the thread's submission", "thread_id is the doc_key of a submission in the table (TRUE for every submission by construction)",
                               "comment: parent_id as stored (a doc_key); submission: NA", "'comment' if parent_id starts with 't1_', 'submission' if 't3_', 'other' otherwise; NA for submissions", "parent_id is the doc_key of a row of the table; NA for submissions",
                               "comment: body; submission: paste(title, selftext, sep = ' ') (the project's established rule; originals kept)", "stringi::stri_length(text): Unicode code points",
                               "created_datetime_utc = as.POSIXct(created_utc, tz = 'UTC'); created_date_utc = its UTC calendar date", "comment files in sorted file-name order, rows in file order; then the submission file",
                               paste(marker_strings, collapse = " | "), "nanoparquet::read_parquet (all columns; INT64 read as double)", "nanoparquet::write_parquet", compression,
                               if (compression %in% c("zstd", "gzip")) as.character(compression_level) else "not applicable to this codec", "created_utc, retrieved_on, score, num_comments"))
mark_stage("environment")

# ---- Write the tables ------------------------------------------------------------------
# One CSV per table under qc/tables/ (run records under environment/).
table_dims <- list()
write_table <- function(tbl, name, dir = tab_dir) {
  path <- file.path(dir, name); write_csv(tbl, path, na = "NA")
  table_dims[[normalizePath(path, winslash = "/")]] <<- c(rows = nrow(tbl), cols = ncol(tbl))
  invisible(path)
}
write_table(reconciliation_checks,       "reconciliation_checks.csv")
write_table(discrepancies,               "discrepancies.csv")
write_table(record_counts,               "record_counts.csv")
write_table(source_event_mapping,        "source_to_event_mapping.csv")
write_table(duplicate_id_checks,         "duplicate_id_checks.csv")
write_table(source_files |> select(-sha256_before), "source_files.csv")
write_table(source_schema,               "source_schema.csv")
write_table(written_schema,              "canonical_parquet_schema.csv")
write_table(parquet_roundtrip,           "parquet_roundtrip_check.csv")
write_table(coverage,                    "field_coverage.csv")
write_table(markers,                     "platform_marker_counts.csv")
write_table(bracketed_values,            "bracketed_value_counts.csv")
write_table(linkage,                     "linkage_summary.csv")
write_table(parent_linkage,              "parent_linkage.csv")
write_table(threads,                     "thread_summary.csv")
write_table(thread_size_distribution,    "thread_size_distribution.csv")
write_table(thread_size_summary,         "thread_size_summary.csv")
write_table(authors,                     "author_activity_summary.csv")
write_table(author_activity_distribution,"author_activity_distribution.csv")
write_table(timestamp_coverage,          "timestamp_coverage.csv")
write_table(records_by_day,              "records_by_day.csv")
write_table(inventory_reconciliation,    "inventory_reconciliation.csv")
write_table(documented_value_checks,     "documented_value_checks.csv")
write_table(tier2_crosscheck,            "tier2_crosscheck.csv")
write_table(raw_data_integrity,          "raw_data_integrity.csv")
write_table(parameters,                  "parameters.csv")
write_table(run_info,                    "run_info.csv",      env_dir)
write_table(stage_timings,               "stage_timings.csv", env_dir)
write_table(resource_use,                "resource_use.csv",  env_dir)
write_table(r_environment,               "r_environment.csv", env_dir)

# ---- Provenance sidecars ---------------------------------------------------------------
# One <output>.manifest.yml per output (manifest_version 1): output hash and size, inputs
# with SHA256 and role, script hash, parameters, git commit and package versions. The
# canonical table's sidecar also records the schema difference from the source.
input_entries <- c(
  lapply(seq_len(nrow(source_files)), function(i) list(path = source_files$relative_path[i], hash = raw_data_integrity$sha256_after[match(source_files$file[i], raw_data_integrity$file)], format = "parquet",
                                                      rows = as.integer(source_files$rows_read[i]), cols = as.integer(source_files$footer_columns[i]), role = "source data, read-only")),
  lapply(ingest_manifests, function(f) list(path = relative_path(f, project_dir), hash = sha256(f), format = "jsonl", role = "ingest manifest sidecar of the source data, read-only")),
  lapply(names(inventory_paths)[inventory_present], function(r) list(path = relative_path(inventory_paths[[r]], project_dir), hash = sha256(inventory_paths[[r]]), format = "csv", role = "earlier metadata-inventory table, read for reconciliation only")),
  if (tier2_present) list(list(path = relative_path(tier2_table, project_dir), hash = sha256(tier2_table), format = "csv", role = "Tier 2 document table from the earlier run, read for the cross-check only")) else list())
package_versions <- lapply(used_packages, function(p) list(name = p, version = as.character(packageVersion(p))))
manifest_parameters <- c(setNames(as.list(parameters$value), parameters$parameter), list(row_group_rows = if (nzchar(row_group_rows)) row_group_rows else "nanoparquet default"))
source_columns <- source_schema |> distinct(field, r_type_after_read) |> group_by(field) |> summarise(type = paste(unique(r_type_after_read), collapse = "|"), .groups = "drop")
canonical_schema_diff <- list(
  added = lapply(setdiff(names(events), c(source_columns$field, "reddit_id")), function(f) list(name = f, type = class(events[[f]])[1])),
  removed = list(),
  renamed = list(list(from = "id", to = "reddit_id", inferred = FALSE)),
  retyped = list())
write_sidecar <- function(output) {
  ext <- tolower(tools::file_ext(output))
  fmt <- switch(ext, parquet = "parquet", csv = "csv", png = "image/png", txt = "other", "other")
  manifest <- list(manifest_version = 1L, output_file = relative_path(output, project_dir), output_hash = sha256(output), output_format = fmt)
  if (fmt == "parquet") { manifest$output_rows <- n_events; manifest$output_cols <- ncol(events) }
  dims <- table_dims[[normalizePath(output, winslash = "/")]]
  if (fmt == "csv" && !is.null(dims)) { manifest$output_rows <- as.integer(dims[["rows"]]); manifest$output_cols <- as.integer(dims[["cols"]]) }
  manifest$input_files <- input_entries
  manifest$transformation <- list(script = relative_path(script_path, project_dir), script_hash = sha256(script_path), parameters = manifest_parameters, git_commit = git_commit)
  manifest$software <- list(language = "R", language_version = paste(R.version$major, R.version$minor, sep = "."), packages = package_versions, os = paste(Sys.info()[["sysname"]], Sys.info()[["release"]]))
  if (fmt == "parquet") manifest$schema_diff <- canonical_schema_diff
  manifest$timestamp <- format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")
  manifest$notes <- "pipeline, Phase 1: canonical Reddit message-event foundation. One row per source record; every source field preserved; doc_key, thread and parent linkage with presence flags; source Parquet read-only; no filtering, no text transformation beyond the stated combined-text rule, no interpretation."
  write_yaml(manifest, paste0(output, ".manifest.yml"))
}
invisible(lapply(c(events_path, list.files(c(tab_dir, env_dir), pattern = "\\.csv$", full.names = TRUE), list.files(fig_dir, pattern = "\\.png$", full.names = TRUE), file.path(env_dir, "session_info.txt")), write_sidecar))
mark_stage("manifests")

# ---- Console summary --------------------------------------------------------------------
cat("\nPhase 1 complete.\n")
cat(sprintf("  canonical events: %d (%d comments + %d submissions) from %d source records\n", n_events, nrow(cm), nrow(sb), n_c + n_s))
cat(sprintf("  threads represented: %d (%d with their submission in the dataset, %d without); distinct author values: %d\n", nrow(threads), sum(threads$root_in_dataset), sum(!threads$root_in_dataset), nrow(authors)))
cat(sprintf("  comments whose submission is present / absent: %d / %d; whose parent is present / absent: %d / %d\n", sum(cm$thread_root_in_dataset), sum(!cm$thread_root_in_dataset), sum(cm$parent_in_dataset %in% TRUE), sum(cm$parent_in_dataset %in% FALSE)))
cat(sprintf("  checks: %d pass, %d not applicable, %d fail; discrepancies: %s\n", sum(reconciliation_checks$status == "pass"), sum(reconciliation_checks$status == "not applicable"), sum(reconciliation_checks$status == "fail"), if (discrepancies$kind[1] == "none") "none" else as.character(nrow(discrepancies))))
cat(sprintf("  raw source files unchanged: %s; elapsed %.1f s\n", raw_unchanged, as.numeric(difftime(Sys.time(), run_start, units = "secs"))))
cat("  outputs:", relative_path(phase_dir, project_dir), "\n")
