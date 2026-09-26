# Three-day embedding run — Qwen3-Embedding-0.6B on the RTX 3090 (TEI, float16)

A **real** embedding run (not a benchmark): every document of three complete UTC days —
**2026-07-15, 2026-07-16, 2026-07-17** — of the canonical Phase 1 table is embedded with the
established Phase 2 TEI setup, and the run is measured to give an early time-to-completion
estimate and the actual totals. These three days sit 14–16 days from either end of the July
dataset (2026-07-01 … 2026-07-31).

The full evidence is in `REPORT.md`. Headline: **101,722 documents / 3,732,319 Qwen tokens**
embedded; whole-card GPU peak **18.48 GiB** (under the 19 GiB ceiling, no shim, no allocator
cap); timed processing **0:01:14** end-to-end (37,383 tokens/s, 1,035 docs/s); the first ETA —
made at 52.7 % of tokens — predicted 0:01:15 total, the actual was 0:01:14; every expected
`doc_key` has a finite 1024-d embedding and only the three dates were embedded.

## Inputs (read-only; Phase 1 is never modified)

| Input | Location | Role |
|---|---|---|
| canonical message-event table (902,144 rows) | `../../phase1_message_event_foundation/derived/message_events.parquet`, verified against its sidecar | the documents; field `text` (comment `body`; submission `title + " " + selftext`), restricted to the three UTC days by `created_date_utc` (cross-checked against the `created_utc` epoch window) |
| per-document Qwen token counts | `../feasibility_qwen3_embedding_worst_case/results/token_counts_all_documents.parquet` (`n_tokens_4B`; the 0.6B tokenizer is identical to the 4B on every document), verified against its sidecar | batch packing and the reported token totals |
| `Qwen/Qwen3-Embedding-0.6B` @ `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` | the WSL2 Hugging Face cache | model |
| TEI release image | `ghcr.io/huggingface/text-embeddings-inference:86-1.9.3` | GPU server |

Reused, unchanged, from `../benchmark_qwen3_4b_0p6b_throughput/scripts/throughput_benchmark.py`:
the TEI `/embed` client, the NVML peak sampler, the token packer, and the provenance manifest writer.

## Method

- TEI `86-1.9.3`, `--model-id` the local 0.6B snapshot, **`--max-batch-tokens 327680`**
  (fixed in advance), `--auto-truncate false`, no `--max-batch-requests`. **No `vram_cap` shim and
  no allocator cap** (unlike the throughput benchmark). float16 — TEI's only precision for its
  flash Qwen3 kernel — last-token pooling, L2-normalized, no truncation, no prompt.
- A short real-batch **memory check** runs first (TEI's own start-up warm-up plus 3 untimed
  warm-up batches, whole-card NVML sampled every 50 ms). The budget would have been reduced only
  if the measured whole-card peak exceeded 19 GiB; it did not (18.48 GiB), so the run proceeded at
  327,680.
- Documents ordered by `(created_utc, doc_key)`; packed into client requests of up to 327,680
  tokens (TEI's non-padded rule). The first 3 batches are untimed warm-up (their embeddings are
  kept); the rest are timed. The reported rate is end-to-end wall-clock (GPU inference + JSON
  transfer/parse); `REPORT.md` also gives TEI's backend-only rate (the GPU ceiling).

## Outputs (`results/`; each has a `.manifest.yml` provenance sidecar)

| Path | Content |
|---|---|
| `embeddings.npy` | float32 `[101722, 1024]`, L2-normalized; row *i* ↔ `embedding_index.parquet` row *i* (**git-ignored**, ~400 MB) |
| `embedding_index.parquet` | `row → doc_key`, with `record_type`, `created_utc`, `created_date_utc`, `n_tokens_4B` |
| `subset.parquet` | the ordered three-day subset built by `prepare` (the embedding row order) |
| `run.json` | every required-evidence field: date range, counts, tokens, memory, warm-up, first ETA, final rates, verification |
| `batches.csv` | one row per client request: documents, planned tokens, TEI `x-compute-tokens`, `x-inference-time`, wall time, device memory |
| `verification.json` | every `doc_key` embedded exactly once, all finite, only the three dates |
| `subset_summary.json`, `environment.json` | subset facts and input checks; image/driver/package record |
| `container_final.log` | the TEI container's own log for the run |
| `../REPORT.md` | the generated run report (the evidence table and the ETA-vs-actual comparison) |

## How to run (inside WSL2, in the Phase 2 venv)

```
PY=/home/spence/venvs/socialmediadgg-phase2/bin/python
BASE='$REPO/pipeline/phase2_semantic_embeddings/three_day_embedding_0p6b'
$PY "$BASE/tests/test_embed_three_days_functions.py"     # pure-function unit test (no GPU)
$PY "$BASE/scripts/embed_three_days.py" --stage prepare  # build the ordered subset (read-only)
$PY "$BASE/scripts/embed_three_days.py" --stage embed    # launch TEI, warm up, embed, measure
$PY "$BASE/scripts/embed_three_days.py" --stage report   # render REPORT.md from run.json
```

`--stage all` runs the three in order. The embed stage removes any prior `tei_three_day_0p6b`
container, launches its own, and removes it at the end.

## Status

Complete (2026-09-10). All 101,722 documents embedded and verified; `REPORT.md` written.
