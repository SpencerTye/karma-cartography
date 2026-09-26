# Phase 1: message-event table

Turns the raw Reddit files (seven comment files and one submission file) into one table
with one row per message. Every submission and comment gets the same 29 columns. Every
source field is kept exactly as stored, and three things are added:

1. `doc_key`, one key per message: the Reddit fullname (`t1_` + id for a comment, `t3_` + id
   for a submission).
2. Thread and reply structure: `thread_id` (the submission that starts the thread),
   `parent_id` (what a comment replies to, joinable to `doc_key`), and two flags for whether
   that submission or parent is in the data set.
3. The source file and row each message came from.

Nothing is removed or filtered, and the text is not cleaned or tokenized. The later phases
all start from this table, so any result can be traced back to the original record.

The script then checks the table against the source files and against itself, writes QC
tables and figures, and records its run time and memory use.

## Inputs

| Input | Location (from the repository root) |
|---|---|
| 7 comment Parquet files (891,397 rows, 11 columns) | `data_sample/comments/2026-07/*.parquet` |
| 1 submission Parquet file (10,747 rows, 13 columns) | `data_sample/submissions/2026-07/*.parquet` |
| Ingest manifests, compared against the files | `data_sample/{comments,submissions}/2026-07/manifest.jsonl` |
| Optional: earlier metadata-inventory tables, for cross-checking counts | `r_analysis_outputs/metadata_inventory/tables/` |
| Optional: the earlier LDA document table, for cross-checking keys and text | `r_analysis_outputs/lda_topics_tier2/tables/document_topic_distribution_K080.csv` |

The data are one community (`r/politics`) over July 2026 (UTC). The source files are only
read, and their SHA256 hashes are taken before and after the run.

## Outputs

Written under this folder. The script clears and rewrites these folders on every run.

| Path | Content |
|---|---|
| `derived/message_events.parquet` | The table: 902,144 rows, 29 columns (135 MB, zstd level 3) |
| `qc/tables/*.csv` | 26 QC tables: counts, linkage, field coverage, marker strings, threads, authors, cross-checks, read-back and source-integrity checks |
| `qc/figures/*.png` | Messages per day, thread sizes, author activity, linkage coverage |
| `environment/` | Run info, stage timings, memory use, R package versions |

Each output has a `.manifest.yml` next to it with the input, output and script hashes,
parameters and package versions. Every column of the table is described in
[`docs/DATA_DICTIONARY.md`](../../docs/DATA_DICTIONARY.md).

## Running it

Needs R 4.5 with nanoparquet, dplyr, tidyr, stringi, readr, ggplot2, scales, digest, yaml,
jsonlite and ps. Use nanoparquet to read the files: arrow 25.0.1 crashed on these comment
files.

```bash
Rscript pipeline/phase1_message_event_foundation/scripts/phase1_build_message_events.R
```

The script finds the repository root from its own location. Optional settings:

| Environment variable | Default | Meaning |
|---|---|---|
| `PHASE1_PROJECT_DIR` | four folders above the script | repository root |
| `PHASE1_COMMENTS_DIR` | `<root>/data_sample/comments/2026-07` | comment files |
| `PHASE1_SUBMISSIONS_DIR` | `<root>/data_sample/submissions/2026-07` | submission files |
| `PHASE1_COMPRESSION` | `zstd` | Parquet codec (`zstd`, `gzip`, `snappy`, `uncompressed`) |
| `PHASE1_COMPRESSION_LEVEL` | `3` | codec level; set explicitly because nanoparquet 0.5.1 writes zstd pages uncompressed without one |
| `PHASE1_ROW_GROUP_ROWS` | one row group | rows per Parquet row group |
| `PHASE1_INVENTORY_DIR` | `<root>/r_analysis_outputs/metadata_inventory/tables` | inventory tables for cross-checks; skipped if missing |
| `PHASE1_TIER2_TABLE` | `<root>/r_analysis_outputs/lda_topics_tier2/tables/document_topic_distribution_K080.csv` | LDA document table for cross-checks; skipped if missing |

It takes under a minute on this data set.

## Test

```bash
Rscript pipeline/phase1_message_event_foundation/tests/test_phase1_functions.R
```

The test loads only the script's functions and runs them on a small made-up data set that
covers every linkage case: threads with and without their submission, a reply to a missing
comment, `[deleted]` authors, empty and marker texts, and a newline in a body. It prints
`test_phase1_functions: all checks passed`.

## Checking a run

- `qc/tables/reconciliation_checks.csv`: 23 checks, all `pass`. The three that need the
  optional inputs can be `not applicable`. `discrepancies.csv` has the single row `none`.
- `qc/tables/record_counts.csv` and `source_to_event_mapping.csv`: 891,397 + 10,747 =
  902,144 rows, every source row used once, `doc_key` unique.
- `qc/tables/raw_data_integrity.csv`: all 8 source files unchanged.
- `qc/tables/parquet_roundtrip_check.csv`: all 29 columns read back identical.
- The console ends with `Phase 1 complete.` and the counts below. Two runs on the same
  inputs give the same file hash.

## Results (run of 2026-09-09)

| Quantity | Value |
|---|---:|
| Messages (comments + submissions) | 891,397 + 10,747 = 902,144 |
| Threads | 11,513 (766 of them without their submission in the data) |
| Submissions with no comments | 439 |
| Comments whose submission is present / missing | 883,733 / 7,664 |
| Comments whose parent is present / missing | 886,046 / 5,351 |
| Distinct author names | 136,479 |
| Messages by `[deleted]` authors | 31,333 |
| Comments per day | 18,669 (Jul 26) to 45,099 (Jul 7) |
| Comments per thread | median 14, mean 80.5, max 5,124 |
| Checks | 23 pass, 0 fail |
| Run time / peak memory | 51.7 s / 2.8 GB (Ryzen 9 3900X, Windows 10, R 4.5.3) |

## Limitations

- `score`, `num_comments` and `upvote_ratio` change over time on Reddit; the data don't say
  exactly when they were captured.
- `raw_json` is empty in every record. It is kept so the schema matches the source.
- 766 threads have no submission in the data, 5,351 comments have no parent in it, and
  `num_comments` disagrees with the collected comments for 5,404 submissions, so the
  collection is not a closed set.
- Only the direct parent is stored. Longer reply chains can be followed through `parent_id`
  where every message in the chain is present.
- Text lengths are counted in Unicode code points. One comment has 10,479 characters, more
  than Reddit's stated limit, and is kept as stored.
