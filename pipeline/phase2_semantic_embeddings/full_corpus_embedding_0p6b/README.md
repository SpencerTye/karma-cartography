# Full-corpus embedding run

Embeds all 902,144 messages from the Phase 1 table (July 1 to 31, 2026) with
Qwen3-Embedding-0.6B, served by Text Embeddings Inference (TEI) on the RTX 3090. Each
embedding stays linked to its `doc_key`. The settings are the ones tested in the
[three-day run](../three_day_embedding_0p6b/).

## Results (run of 2026-09-10)

| Measure | Value |
|---|---|
| Documents / tokens | 902,144 / 35,388,381 |
| Time (timed batches) | 14 min 59 s: 977 documents/s, 38,250 tokens/s |
| Time from TEI ready to done | 16 min 56 s |
| Peak GPU memory (whole card) | 18.43 GiB, under the 19 GiB limit for this run |
| Time estimate made at 5.6% of tokens | 15 min 23 s (actual 14 min 59 s) |
| Checks | every `doc_key` has one finite, unit-length 1,024-dimension embedding; the embedded set matches the corpus exactly (0 missing, 0 extra) |

## Method

- TEI 1.9.3 with `--max-batch-tokens 327680`, `--auto-truncate false`, float16, last-token
  pooling, L2 normalization, no truncation and no prompt. No memory cap is used; the batch
  size was set from the three-day run, which peaked at 18.48 GiB.
- A short memory check with real batches runs first. The batch size would have dropped to
  262,144 if the peak had gone over 19 GiB. It didn't.
- Documents are sorted by `(created_utc, doc_key)` and packed into 109 requests of up to
  327,680 tokens each. The first 3 are untimed warm-up (their embeddings are kept) and the
  other 106 are timed. The rate is end to end: GPU time plus sending and parsing the results.
- The token counts come from the [worst-case test](../feasibility_qwen3_embedding_worst_case/).
  The 0.6B and 4B tokenizers give identical tokens on every document.
- One comment (`t1_ovhv2aj`) has empty text, which TEI rejects, so it is embedded from a
  single space. This is recorded in the outputs.
- The TEI client, memory sampler, token packer and manifest writer are imported from
  `../benchmark_qwen3_4b_0p6b_throughput/scripts/throughput_benchmark.py`.

## Running it

Inside WSL2, with the Phase 2 environment
([`environment/requirements-embeddings.txt`](../../../environment/requirements-embeddings.txt))
and the image `ghcr.io/huggingface/text-embeddings-inference:86-1.9.3`:

```bash
python scripts/embed_full_corpus.py --stage prepare   # order the documents (reads only)
python scripts/embed_full_corpus.py --stage embed     # start TEI, embed, measure, verify
python scripts/embed_full_corpus.py --stage report    # write REPORT.md
python tests/test_embed_full_corpus_functions.py      # unit test, no GPU needed
```

`--stage all` runs the three stages in order. The embed stage starts its own TEI container
and removes it at the end.

## Outputs

In `results/`, each with a `.manifest.yml`:

- `embeddings.npy`: float32, 902,144 × 1,024 (3.4 GB)
- `embedding_index.parquet`: maps each row to its `doc_key`, type, timestamp and token count
- `run.json`, `batches.csv`, `verification.json`: timings, memory and checks for each batch
  and for the whole run
