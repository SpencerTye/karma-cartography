# Unit test for the pure functions in phase1_build_message_events.R.
# Parses the Phase 1 script and evaluates only its top-level function definitions (the pipeline
# does not run), then checks them on a synthetic fixture covering every linkage case.
# Run: Rscript "pipeline/phase1_message_event_foundation/tests/test_phase1_functions.R"
# Needs dplyr, tidyr, stringi and tibble. PHASE1_SCRIPT overrides the script path.

suppressPackageStartupMessages({ library(dplyr); library(tibble); library(stringi) })

this_file <- local({
  arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(arg)) normalizePath(sub("^--file=", "", arg[1]), winslash = "/", mustWork = FALSE) else NA_character_
})
script <- Sys.getenv("PHASE1_SCRIPT", unset = "")
if (!nzchar(script)) {
  script <- if (!is.na(this_file)) file.path(dirname(dirname(this_file)), "scripts", "phase1_build_message_events.R")
            else "pipeline/phase1_message_event_foundation/scripts/phase1_build_message_events.R"
}
stopifnot("Phase 1 script file exists" = file.exists(script))
exprs <- parse(script, keep.source = FALSE)
is_fun_def <- vapply(exprs, function(e) {
  is.call(e) && identical(e[[1]], as.name("<-")) && is.name(e[[2]]) &&
    is.call(e[[3]]) && identical(e[[3]][[1]], as.name("function"))
}, logical(1))
for (e in exprs[is_fun_def]) eval(e, envir = globalenv())
needed <- c("make_doc_key", "build_message_events", "summarise_threads", "summarise_authors",
            "linkage_summary", "field_coverage", "count_platform_markers")
missing_fun <- needed[!vapply(needed, exists, logical(1), mode = "function")]
stopifnot("every function under test is defined in the script" = length(missing_fun) == 0L)

# ---- Fixture -------------------------------------------------------------------------
# Submissions: A (self-post with empty selftext, url present), B (marker title, marker
# selftext, url present), Q (no comments, deleted author, real selftext, empty url).
# Comments: c1 thread A replies to the submission; c2 thread A replies to c1; c3 thread Z
# (submission absent) replies to the absent submission, deleted author, removed body;
# c4 thread B replies to an absent comment x, empty body; c5 thread B replies to c4.
submissions <- tibble(
  id = c("A", "B", "Q"), title = c("Title A", "[ Removed by moderator ]", "Title Q"),
  selftext = c("", "[removed]", "Body of Q\nsecond line"), url = c("https://a.example/x", "https://b.example/y", ""),
  author = c("u9", "u2", "[deleted]"), created_utc = c(1782864000, 1782864050, 1782864060),
  retrieved_on = c(1782864020, 1782864070, 1782864080), score = c(40, 7, 0), num_comments = c(5, 8, 0),
  upvote_ratio = c(0.9, 0.7, 1), subreddit = "politics", subreddit_id = "t5_2cneq", raw_json = NA_character_,
  source_file = "data_sample/submissions/2026-07/s.parquet", source_row_in_file = 1:3)
comments <- tibble(
  id = c("c1", "c2", "c3", "c4", "c5"), parent_id = c("t3_A", "t1_c1", "t3_Z", "t1_x", "t1_c4"),
  link_id = c("t3_A", "t3_A", "t3_Z", "t3_B", "t3_B"), author = c("u1", "u2", "[deleted]", "u1", "u2"),
  body = c("hello\nworld", "reply to c1 \u2019 caf\u00e9", "[removed]", "", "reply to c4"),
  created_utc = c(1782864100, 1782864200, 1782864300, 1782864400, 1782950800), retrieved_on = c(1782864130, 1782864230, 1782864330, 1782864430, 1782950830),
  score = c(3, -2, 1, 0, 5), subreddit = "politics", subreddit_id = "t5_2cneq", raw_json = NA_character_,
  source_file = c(rep("data_sample/comments/2026-07/c0.parquet", 3), rep("data_sample/comments/2026-07/c1.parquet", 2)),
  source_row_in_file = c(1L, 2L, 3L, 1L, 2L))

# ---- make_doc_key ----------------------------------------------------------------------
stopifnot(
  "doc_key prefixes" = identical(make_doc_key(c("comment", "submission"), c("c1", "A")), c("t1_c1", "t3_A")),
  "unknown record type is an error" = inherits(tryCatch(make_doc_key("other", "z"), error = function(e) e), "error"))

# ---- build_message_events --------------------------------------------------------------
ev <- build_message_events(comments, submissions)
expected_cols <- c("doc_key", "record_type", "reddit_id", "author", "created_utc", "created_datetime_utc", "created_date_utc",
                   "retrieved_on", "subreddit", "subreddit_id", "thread_id", "is_thread_root", "thread_root_in_dataset",
                   "link_id", "parent_id", "parent_kind", "parent_in_dataset", "score", "num_comments", "upvote_ratio",
                   "url", "title", "selftext", "body", "text", "text_n_chars", "raw_json", "source_file", "source_row_in_file")
stopifnot(
  "one event per source record, comments first then submissions, source order kept" =
    nrow(ev) == 8L && identical(ev$reddit_id, c("c1", "c2", "c3", "c4", "c5", "A", "B", "Q")),
  "exact column set and order" = identical(names(ev), expected_cols),
  "doc_key is the Reddit fullname and unique" =
    identical(ev$doc_key, c("t1_c1", "t1_c2", "t1_c3", "t1_c4", "t1_c5", "t3_A", "t3_B", "t3_Q")) && !anyDuplicated(ev$doc_key),
  "record_type values" = identical(ev$record_type, c(rep("comment", 5), rep("submission", 3))),
  "author preserved exactly, including [deleted]" = identical(ev$author, c("u1", "u2", "[deleted]", "u1", "u2", "u9", "u2", "[deleted]")),
  "created_utc preserved as numeric epoch seconds" = identical(ev$created_utc, c(comments$created_utc, submissions$created_utc)),
  "created_datetime_utc is POSIXct in UTC equal to created_utc" =
    inherits(ev$created_datetime_utc, "POSIXct") && identical(attr(ev$created_datetime_utc, "tzone"), "UTC") &&
    all(as.numeric(ev$created_datetime_utc) == ev$created_utc),
  "created_date_utc is the UTC calendar date" = inherits(ev$created_date_utc, "Date") &&
    identical(as.character(ev$created_date_utc), c(rep("2026-07-01", 4), "2026-07-02", rep("2026-07-01", 3))),
  "thread_id: link_id for comments, own fullname for submissions" =
    identical(ev$thread_id, c("t3_A", "t3_A", "t3_Z", "t3_B", "t3_B", "t3_A", "t3_B", "t3_Q")),
  "submissions are the root of their own thread" = identical(ev$is_thread_root, c(rep(FALSE, 5), rep(TRUE, 3))) &&
    all(ev$thread_id[ev$is_thread_root] == ev$doc_key[ev$is_thread_root]),
  "thread_root_in_dataset: absent submission Z recorded, not invented" =
    identical(ev$thread_root_in_dataset, c(TRUE, TRUE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE)),
  "link_id and parent_id kept as stored for comments, NA for submissions" =
    identical(ev$link_id, c(comments$link_id, rep(NA_character_, 3))) && identical(ev$parent_id, c(comments$parent_id, rep(NA_character_, 3))),
  "parent_kind decoded from the prefix, NA for submissions" =
    identical(ev$parent_kind, c("submission", "comment", "submission", "comment", "comment", NA, NA, NA)),
  "parent_in_dataset: absent parents (t3_Z, t1_x) recorded" =
    identical(ev$parent_in_dataset, c(TRUE, TRUE, FALSE, FALSE, TRUE, NA, NA, NA)),
  "score carried for both types; submission-only fields NA for comments" =
    identical(ev$score, c(comments$score, submissions$score)) && all(is.na(ev$num_comments[1:5])) && all(is.na(ev$upvote_ratio[1:5])) &&
    all(is.na(ev$url[1:5])) && all(is.na(ev$title[1:5])) && all(is.na(ev$selftext[1:5])) && identical(ev$num_comments[6:8], submissions$num_comments),
  "comment-only fields NA for submissions" = all(is.na(ev$body[6:8])),
  "original text fields preserved byte for byte" =
    identical(ev$body[1:5], comments$body) && identical(ev$title[6:8], submissions$title) && identical(ev$selftext[6:8], submissions$selftext),
  "text rule: body for comments; title + one space + selftext for submissions" =
    identical(ev$text, c(comments$body, "Title A ", "[ Removed by moderator ] [removed]", "Title Q Body of Q\nsecond line")),
  "text_n_chars is the code-point length of text" = is.integer(ev$text_n_chars) && identical(ev$text_n_chars, stri_length(ev$text)),
  "raw_json carried (all NA in the fixture)" = all(is.na(ev$raw_json)),
  "provenance columns carried" = identical(ev$source_file[1:5], comments$source_file) && identical(ev$source_row_in_file, c(1L, 2L, 3L, 1L, 2L, 1L, 2L, 3L)),
  "retrieved_on and subreddit fields carried" = identical(ev$retrieved_on, c(comments$retrieved_on, submissions$retrieved_on)) && all(ev$subreddit_id == "t5_2cneq"))

# A duplicated id in the input must be rejected, never silently merged.
dup <- tryCatch(build_message_events(bind_rows(comments, comments[1, ]), submissions), error = function(e) e)
stopifnot("duplicate source ids are an error" = inherits(dup, "error"))

# ---- summarise_threads -----------------------------------------------------------------
th <- summarise_threads(ev)
th <- th[order(th$thread_id), ]
stopifnot(
  "one row per thread represented (comments' threads plus every submission)" = identical(th$thread_id, c("t3_A", "t3_B", "t3_Q", "t3_Z")),
  "root presence per thread" = identical(th$root_in_dataset, c(TRUE, TRUE, TRUE, FALSE)),
  "comments per thread" = identical(as.integer(th$n_comments), c(2L, 2L, 0L, 1L)),
  "events per thread count the root when present" = identical(as.integer(th$n_events), c(3L, 3L, 1L, 1L)),
  "distinct comment authors per thread ([deleted] counted as a stored value)" = identical(as.integer(th$n_comment_authors), c(2L, 2L, 0L, 1L)),
  "first and last comment times" = identical(th$first_comment_utc, c(1782864100, 1782864400, NA, 1782864300)) &&
    identical(th$last_comment_utc, c(1782864200, 1782950800, NA, 1782864300)),
  "root fields when present" = identical(th$root_created_utc, c(1782864000, 1782864050, 1782864060, NA)) &&
    identical(th$root_num_comments, c(5, 8, 0, NA)) && identical(th$root_author, c("u9", "u2", "[deleted]", NA)))

# ---- summarise_authors -----------------------------------------------------------------
au <- summarise_authors(ev)
au <- au[order(au$author), ]
stopifnot(
  "one row per distinct stored author value" = identical(au$author, c("[deleted]", "u1", "u2", "u9")),
  "event, comment and submission counts" = identical(as.integer(au$n_events), c(2L, 2L, 3L, 1L)) &&
    identical(as.integer(au$n_comments), c(1L, 2L, 2L, 0L)) && identical(as.integer(au$n_submissions), c(1L, 0L, 1L, 1L)),
  "distinct threads per author" = identical(as.integer(au$n_threads), c(2L, 2L, 2L, 1L)),
  "first and last activity" = identical(au$first_created_utc, c(1782864060, 1782864100, 1782864050, 1782864000)) &&
    identical(au$last_created_utc, c(1782864300, 1782864400, 1782950800, 1782864000)),
  "active days" = identical(as.integer(au$n_active_days), c(1L, 1L, 2L, 1L)))

# ---- linkage_summary -------------------------------------------------------------------
lk <- linkage_summary(ev)
val <- function(item) lk$value[lk$item == item]
stopifnot(
  "comments with / without their submission" = val("comments whose submission is in the dataset") == 4 && val("comments whose submission is absent from the dataset") == 1,
  "comments with / without their parent" = val("comments whose parent is in the dataset") == 3 && val("comments whose parent is absent from the dataset") == 2,
  "parent kinds" = val("comments whose parent is a comment (t1_)") == 3 && val("comments whose parent is the submission (t3_)") == 2,
  "threads" = val("threads represented (distinct thread_id)") == 4 && val("threads whose root submission is in the dataset") == 3 &&
    val("threads whose root submission is absent from the dataset") == 1 && val("submissions with no comment in the dataset") == 1,
  "authors" = val("distinct author values (all events)") == 4)

# ---- field_coverage --------------------------------------------------------------------
fc <- field_coverage(ev)
row <- function(f, t) fc[fc$field == f & fc$record_type == t, ]
stopifnot(
  "every canonical field x record type" = nrow(fc) == 2L * length(expected_cols),
  "author coverage" = row("author", "comment")$n_usable == 5 && row("author", "comment")$n_distinct == 3,
  "num_comments not applicable to comments" = row("num_comments", "comment")$n_missing == 5 && row("num_comments", "submission")$n_usable == 3,
  "empty strings counted separately from missing" = row("body", "comment")$n_empty == 1 && row("body", "comment")$n_usable == 4 &&
    row("selftext", "submission")$n_empty == 1 && row("url", "submission")$n_empty == 1,
  "identities hold" = all(fc$n_usable + fc$n_missing + fc$n_empty == fc$n_records))

# ---- count_platform_markers ------------------------------------------------------------
mk <- count_platform_markers(ev)
cnt <- function(t, f, v) { r <- mk[mk$record_type == t & mk$field == f & mk$stored_value == v, ]; if (nrow(r)) r$n else 0L }
stopifnot(
  "comment body markers" = cnt("comment", "body", "[removed]") == 1 && cnt("comment", "body", "") == 1 && cnt("comment", "body", "[deleted]") == 0,
  "author markers by type" = cnt("comment", "author", "[deleted]") == 1 && cnt("submission", "author", "[deleted]") == 1,
  "submission title and selftext markers" = cnt("submission", "title", "[ Removed by moderator ]") == 1 &&
    cnt("submission", "selftext", "[removed]") == 1 && cnt("submission", "selftext", "") == 1 && cnt("submission", "url", "") == 1,
  "marker rows carry percentages of the record type" = all(mk$pct_of_record_type >= 0 & mk$pct_of_record_type <= 100))

cat("test_phase1_functions: all checks passed\n")
