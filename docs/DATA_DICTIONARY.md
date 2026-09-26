# Data dictionary: `derived/message_events.parquet`

One row per Reddit record of the collected dataset (a submission or a comment), 902,144
rows, 29 columns, written by `scripts/phase1_build_message_events.R`. Columns are listed in
the order they are stored. "Applies to" says which record type carries a value; a field
native to one record type is `NA` (Parquet null) for the other. Row order: the seven comment
files in sorted file-name order, rows in file order, then the submission file.

Types: "R" is the class after `nanoparquet::read_parquet()`; "Parquet" is the declared
physical (and logical) type in the file (`qc/tables/canonical_parquet_schema.csv`).

| # | Field | Meaning | Source or derivation | R / Parquet | Applies to | Known limitations |
|---|---|---|---|---|---|---|
| 1 | `doc_key` | The record's Reddit "fullname" and the project's established document key: unique across both record types | derived: `'t1_' + id` for comments, `'t3_' + id` for submissions (the same rule as every earlier project artifact) | character / STRING | both | none; the two id spaces are disjoint (verified) |
| 2 | `record_type` | `comment` or `submission` | derived from the source table the row was read from (comment files / submission file); equals `doc_type` in the earlier artifacts | character / STRING | both | only these two record types exist in the source |
| 3 | `reddit_id` | The bare base36 Reddit id of the record, exactly as stored | source `id` (renamed; unchanged) | character / STRING | both | unique within a record type, not across types: use `doc_key` as the key |
| 4 | `author` | The account name exactly as stored, including the platform value `[deleted]` | source `author` (unchanged) | character / STRING | both | `[deleted]` is one stored value covering many accounts (30,996 comments, 337 submissions); no field says which; nothing is normalised or classified here |
| 5 | `created_utc` | Creation time, seconds since 1970-01-01 00:00:00 UTC | source `created_utc` (unchanged; INT64 in the source, read as double, written back as INT64) | numeric (double) / INT64 | both | whole seconds; all values in July 2026 UTC |
| 6 | `created_datetime_utc` | `created_utc` as a date-time in UTC | derived: `as.POSIXct(created_utc, tz = "UTC")` | POSIXct (UTC) / INT64 TIMESTAMP(MICROS, UTC) | both | convenience only; carries no information beyond `created_utc` |
| 7 | `created_date_utc` | The UTC calendar date of `created_utc` | derived: `as.Date(created_datetime_utc, tz = "UTC")` | Date / INT32 DATE | both | UTC dates, not local dates |
| 8 | `retrieved_on` | Ingestion time recorded by the collector, epoch seconds | source `retrieved_on` (unchanged) | numeric (double) / INT64 | both | 10 s to about 36 h after `created_utc`; whether it dates the mutable counters (`score`, `num_comments`, `upvote_ratio`) is not established by any field |
| 9 | `subreddit` | Community name | source `subreddit` (unchanged) | character / STRING | both | constant `politics` in this dataset |
| 10 | `subreddit_id` | Community id | source `subreddit_id` (unchanged) | character / STRING | both | constant `t5_2cneq` in this dataset |
| 11 | `thread_id` | The thread the record belongs to, as the fullname of the thread's submission; equal to the `doc_key` of that submission | derived: comments `link_id` as stored; submissions `'t3_' + id` | character / STRING | both | a comment's thread may have no submission row in the dataset (see `thread_root_in_dataset`); the id is still kept |
| 12 | `is_thread_root` | Whether the record is the submission that starts its thread | derived: `record_type == "submission"` | logical / BOOLEAN | both | none |
| 13 | `thread_root_in_dataset` | Whether the thread's submission is present as a row of this table | derived: `thread_id %in% doc_key[record_type == "submission"]` | logical / BOOLEAN | both | TRUE for every submission by construction; FALSE for 7,664 comments on 766 threads whose submission was not collected |
| 14 | `link_id` | The thread reference exactly as stored (`t3_` + submission id) | source `link_id` (unchanged) | character / STRING | comment | NA for submissions; identical to `thread_id` for comments |
| 15 | `parent_id` | What the comment replies to, exactly as stored: `t1_` + comment id or `t3_` + submission id; directly joinable to `doc_key` | source `parent_id` (unchanged) | character / STRING | comment | NA for submissions (no parent); the parent may be absent from the dataset (see `parent_in_dataset`) |
| 16 | `parent_kind` | `comment` if `parent_id` starts with `t1_`, `submission` if `t3_`, `other` otherwise | derived from the `parent_id` prefix | character / STRING | comment | NA for submissions; no `other` value occurs in this dataset |
| 17 | `parent_in_dataset` | Whether the parent record is present as a row of this table | derived: `parent_id %in% doc_key` | logical / BOOLEAN | comment | NA for submissions; FALSE for 5,351 comments (3,446 parent comments and 1,905 parent submissions not collected) |
| 18 | `score` | Net vote score as stored | source `score` (unchanged) | numeric (double) / INT64 | both | a point-in-time value of unknown vintage; comment scores go negative, submission scores do not in this dataset |
| 19 | `num_comments` | Comment count stored on the submission | source `num_comments` (unchanged) | numeric (double) / INT64 | submission | NA for comments; a point-in-time value that does not equal the comments collected (the dataset holds more comments than the stored count for 5,262 submissions) |
| 20 | `upvote_ratio` | Fraction of votes that were upvotes, as stored | source `upvote_ratio` (unchanged) | numeric (double) / DOUBLE | submission | NA for comments; 100 distinct two-decimal values; vintage unknown |
| 21 | `url` | The submission's link URL as stored | source `url` (unchanged) | character / STRING | submission | NA for comments; empty string for 337 submissions |
| 22 | `title` | Submission title as stored | source `title` (unchanged) | character / STRING | submission | NA for comments; the platform values `[ Removed by moderator ]` (368) and `[ Removed by Reddit ]` (17) occur |
| 23 | `selftext` | Submission body text as stored | source `selftext` (unchanged) | character / STRING | submission | NA for comments; empty for 10,335 submissions (link posts); `[removed]` 339, `[deleted]` 47 |
| 24 | `body` | Comment text as stored | source `body` (unchanged) | character / STRING | comment | NA for submissions; the platform values `[removed]` (27,501), `[deleted]` (3,496), `[ Removed by Reddit ]` (1,009) and one empty string occur; the original text of such records is not in the source |
| 25 | `text` | One text per record for downstream use | derived: comments `body`; submissions `paste(title, selftext, sep = " ")` (the project's established rule) | character / STRING | both | a submission with empty `selftext` ends in one space; no other transformation; the originals remain in `body`, `title`, `selftext` |
| 26 | `text_n_chars` | Length of `text` in Unicode code points | derived: `stringi::stri_length(text)` | integer / INT32 | both | code points, not bytes or grapheme clusters |
| 27 | `raw_json` | Raw payload column declared by the collector | source `raw_json` (unchanged) | character / STRING | both | NULL in every record of this dataset; carried so that the canonical schema is a superset of both source schemas |
| 28 | `source_file` | Path of the Parquet file the record was read from, relative to the project root | provenance, set at read time | character / STRING | both | none |
| 29 | `source_row_in_file` | 1-based row position of the record within `source_file` | provenance, set at read time | integer / INT32 | both | none; (`source_file`, `source_row_in_file`) is unique and identifies the original record |

## Keys and joins

- **Primary key:** `doc_key` (unique; equal to `'t1_' + reddit_id` or `'t3_' + reddit_id`).
- **Thread:** `thread_id` equals the `doc_key` of the thread's submission. All rows of a thread:
  `filter(thread_id == "t3_xxxxxxx")`. The root: the row with `doc_key == thread_id`.
- **Reply:** `parent_id` equals the `doc_key` of the parent. Replies to a record:
  `filter(parent_id == that_doc_key)`. The parent of a comment: the row with
  `doc_key == parent_id` (absent from the table when `parent_in_dataset` is FALSE).
- **Source record:** (`source_file`, `source_row_in_file`), or `reddit_id` within the file's
  record type.
- **Earlier project artifacts:** their `doc_key` is the same key; `doc_type` = `record_type`,
  `doc_id` = `reddit_id`, `submission_id` = `thread_id` without its first three characters,
  `submission_in_sample` = `thread_root_in_dataset`, `n_chars` = `text_n_chars`.

## Missing-value conventions

- `NA` (Parquet null) means the field does not exist for that record type (for example
  `body` on a submission) or, for `raw_json`, that the source stored null.
- The empty string `""` is a stored value (kept as stored): comment `body` once, submission
  `selftext` 10,335 times, `url` 337 times.
- Platform marker strings (`[deleted]`, `[removed]`, `[ Removed by Reddit ]`,
  `[ Removed by moderator ]`) are stored values, kept as stored and counted in
  `qc/tables/platform_marker_counts.csv`; nothing in this table flags or removes them.

## Example queries (R, dplyr)

```r
library(nanoparquet); library(dplyr)
ev <- read_parquet("pipeline/phase1_message_event_foundation/derived/message_events.parquet")

ev |> filter(author == "some_account")                                   # all messages by an author
ev |> filter(thread_id == "t3_1uk5ysg") |> distinct(author)             # all authors in a thread
ev |> filter(parent_id == "t1_owfaptx")                                  # all replies to a message
ev |> filter(author == "some_account") |> arrange(created_utc)           # an author's chronology
ev |> count(author, subreddit)                                           # cross-subreddit activity (one subreddit here)
ev |> filter(record_type == "comment", !thread_root_in_dataset)          # comments whose submission was not collected
```

In Python: `pandas.read_parquet(path)` or `duckdb.sql("SELECT ... FROM 'path'")` read the same
file; `created_utc`, `retrieved_on`, `score` and `num_comments` arrive as 64-bit integers,
`created_datetime_utc` as a UTC timestamp, `created_date_utc` as a date.
