# Phase 2: semantic embeddings

Every message is embedded with Qwen3-Embedding-0.6B (1,024 dimensions, L2-normalised),
served by Hugging Face Text Embeddings Inference on the GPU. The folders run in this order:

| Folder | Purpose |
|---|---|
| `feasibility_qwen3_embedding_worst_case/` | Do the 4B and 8B models fit the longest documents in a 21 GiB budget? |
| `feasibility_tei_qwen3_4b_batch_tokens/` | Largest safe batch size in TEI, enforced with the `vram_cap` library in `scripts/vram_cap/` |
| `benchmark_qwen3_4b_0p6b_throughput/` | GPU vs CPU speed, 0.6B vs 4B; holds the shared helpers (`throughput_benchmark.py`) |
| `three_day_embedding_0p6b/` | Trial run on three days of data |
| `full_corpus_embedding_0p6b/` | The production run: all 902,144 messages |
| `faiss_gpu_exact_search_benchmark/` | Speed of exact nearest-neighbour search over the result |

Each folder has its own README with the exact command, outputs and success checks.
Results are summarised in [`docs/METHODS.md`](../../docs/METHODS.md).
