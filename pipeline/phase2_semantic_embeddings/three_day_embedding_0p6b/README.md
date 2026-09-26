# Three-day embedding run

A trial of the embedding setup on three full days in the middle of the month (July 15 to
17, 2026) before the [full run](../full_corpus_embedding_0p6b/). Same method and settings
as the full run.

## Results (run of 2026-09-10)

| Measure | Value |
|---|---|
| Documents / tokens | 101,722 / 3,732,319 |
| Time (timed batches) | 1 min 14 s: 1,035 documents/s, 37,383 tokens/s |
| Peak GPU memory (whole card) | 18.48 GiB, under the 19 GiB limit for this run |
| Time estimate made at 52.7% of tokens | 1 min 15 s (actual 1 min 14 s) |
| Checks | every `doc_key` has one finite 1,024-dimension embedding; only the three days were embedded |

## Method

- TEI 1.9.3 with `--max-batch-tokens 327680`, `--auto-truncate false`, float16, last-token
  pooling, L2 normalization, no truncation, no prompt and no memory cap.
- A short memory check with real batches runs first. The batch size would have been reduced
  if the peak had gone over 19 GiB. It didn't.
- Documents are filtered to the three days by `created_date_utc` (and checked against the
  timestamps), sorted by `(created_utc, doc_key)`, and packed into requests of up to 327,680
  tokens. The first 3 are untimed warm-up.

## Running it

```bash
python scripts/embed_three_days.py --stage prepare   # select and order the three days
python scripts/embed_three_days.py --stage embed     # start TEI, embed, measure, verify
python scripts/embed_three_days.py --stage report    # write REPORT.md
python tests/test_embed_three_days_functions.py      # unit test, no GPU needed
```

Outputs go to `results/`: `embeddings.npy` (101,722 × 1,024), `embedding_index.parquet`,
`run.json`, `batches.csv` and `verification.json`.
