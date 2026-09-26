# Full-corpus embedding run — Qwen3-Embedding-0.6B on the RTX 3090 (TEI, float16)

The **production** semantic-embedding run for the complete canonical dataset: **every** document
of the canonical Phase 1 message-event table — **all 902,144 rows**, the full July window
**2026-07-01 … 2026-07-31** — is embedded with the established Phase 2 TEI setup, kept linked to
its original `doc_key`, and measured (an early time-to-completion estimate plus the actual totals).

This is the full-corpus sibling of `../three_day_embedding_0p6b/`. It uses the same embedding
behaviour proven there and does not read, modify, or overwrite that run. The full evidence table is
in `REPORT.md`.

**Headline:** **902,144 documents / 35,388,381 Qwen tokens** embedded; whole-card GPU peak
**18.43 GiB** (under the 19 GiB ceiling, no shim, no allocator cap); timed processing **0:14:59**
end-to-end (**38,250 tokens/s**, 977 docs/s), ready→done **0:16:56**; the first ETA — made at 5.6 %
of tokens — predicted 0:15:23 of timed processing, the actual was 0:14:59. Every expected `doc_key`
has a finite, L2-unit-norm 1024-d embedding; the embedded set equals the canonical corpus exactly
(0 missing, 0 extra).

## Inputs (read-only; Phase 1 is never modified)

| Input | Location | Role |
|---|---|---|
| canonical message-event table (902,144 rows) | `../../phase1_message_event_foundation/derived/message_events.parquet`, verified against its sidecar | the documents; field `text` (comment `body`; submission `title + " " + selftext`), all rows, ordered by `(created_utc, doc_key)` |
| per-document Qwen token counts | `../feasibility_qwen3_embedding_worst_case/results/token_counts_all_documents.parquet` (`n_tokens_4B`; the 0.6B tokenizer is identical to the 4B on every document), verified against its sidecar | batch packing and the reported token totals — **reused, not recomputed** |
| `Qwen/Qwen3-Embedding-0.6B` @ `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` | the WSL2 Hugging Face cache | model |
| TEI release image | `ghcr.io/huggingface/text-embeddings-inference:86-1.9.3` | GPU server |

Reused, unchanged, from `../benchmark_qwen3_4b_0p6b_throughput/scripts/throughput_benchmark.py`:
the TEI `/embed` client, the NVML peak sampler, the token packer, and the provenance manifest writer.

## Method

- TEI `86-1.9.3`, `--model-id` the local 0.6B snapshot, **`--max-batch-tokens 327680`**
  (fixed from the three-day run measured this at an 18.48 GiB whole-card peak), `--auto-truncate
  false`, no `--max-batch-requests`. **No `vram_cap` shim and no allocator cap** — the same launch as
  the three-day run. float16 (TEI's only precision for its flash Qwen3 kernel), last-token pooling,
  L2-normalized, no truncation, no prompt.
- A short real-batch **memory check** runs first (TEI's own start-up warm-up plus 3 untimed warm-up
  batches, whole-card NVML sampled every 50 ms). The budget would have been reduced to the single
  documented fallback (262,144) only if the measured whole-card peak exceeded 19 GiB; it did not
  (18.43 GiB), so the run proceeded at 327,680.
- Documents ordered by `(created_utc, doc_key)`; packed into 109 client requests of up to 327,680
  tokens (TEI's non-padded rule). The first 3 batches are untimed warm-up (their embeddings are
  kept); the remaining 106 are timed. The reported rate is end-to-end wall-clock (GPU inference +
  JSON transfer/parse); `REPORT.md` also gives TEI's backend-only rate (the GPU ceiling).
- **Empty text:** the canonical corpus contains exactly one empty-text comment (`t1_ovhv2aj`). TEI
  1.9.3 rejects an empty input with HTTP 400, so that one document is embedded from a minimal
  single-space `" "` placeholder to keep exactly one embedding per `doc_key`. The substitution is
  recorded in `run.json`/`verification.json`. It leaves the planned/budgeted token totals unchanged
  and adds exactly 1 token to TEI's own `x-compute-tokens` total (an empty string counts as 1 token,
  the placeholder as 2).

## Outputs (`results/`; each has a `.manifest.yml` provenance sidecar)

| Path | Content |
|---|---|
| `embeddings.npy` | float32 `[902144, 1024]`, L2-normalized (every row unit-norm); row *i* ↔ `embedding_index.parquet` row *i* (**git-ignored**, ~3.44 GiB) |
| `embedding_index.parquet` | `row → doc_key`, with `record_type`, `created_utc`, `created_date_utc`, `n_tokens_4B` |
| `corpus_order.parquet` | the ordered full-corpus stream built by `prepare` (the embedding row order; metadata only, no text — it does not duplicate the canonical text) |
| `run.json` | every required-evidence field: scope, date range, counts, tokens, memory, warm-up, first ETA, final rates, verification, the empty-text substitution, TEI `docker run` command |
| `batches.csv` | one row per client request: documents, planned tokens, TEI `x-compute-tokens`, `x-inference-time`, wall time, device memory |
| `verification.json` | every `doc_key` embedded exactly once, all finite, embedded set == canonical, token totals reconciled |
| `corpus_summary.json`, `environment.json` | corpus facts and input checks; image/driver/package record |
| `container_final.log` | the TEI container's own log for the run |
| `../REPORT.md` | the generated run report (the evidence table and the ETA-vs-actual comparison) |

## How to run (inside WSL2 Ubuntu-24.04, in the Phase 2 venv)

```
PY=/home/spence/venvs/socialmediadgg-phase2/bin/python
BASE='$REPO/pipeline/phase2_semantic_embeddings/full_corpus_embedding_0p6b'
$PY "$BASE/tests/test_embed_full_corpus_functions.py"     # pure-function unit test (no GPU)
$PY "$BASE/scripts/embed_full_corpus.py" --stage prepare   # build the ordered full-corpus stream (read-only)
$PY "$BASE/scripts/embed_full_corpus.py" --stage embed     # launch TEI, warm up, embed all, measure, verify
$PY "$BASE/scripts/embed_full_corpus.py" --stage report    # render REPORT.md from run.json
```

`--stage all` runs the three in order. The embed stage removes any prior `tei_full_corpus_0p6b`
container, launches its own, and removes it at the end.

## Status

Complete (2026-09-10). All 902,144 documents embedded and verified (independently confirmed on the
saved `embeddings.npy`: shape `[902144, 1024]`, all values finite, every row L2-unit-norm; the index
matches the canonical corpus exactly); `REPORT.md` written.
