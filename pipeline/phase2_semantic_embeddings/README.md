# Phase 2: embeddings

Each message is embedded with Qwen3-Embedding-0.6B (1,024 dimensions, L2-normalized) using
Hugging Face Text Embeddings Inference on the GPU. The folders, in the order they were run:

| Folder | What it does |
|---|---|
| `feasibility_qwen3_embedding_worst_case/` | Tests whether the 4B and 8B models fit the longest documents in 21 GiB of GPU memory |
| `feasibility_tei_qwen3_4b_batch_tokens/` | Finds the largest batch size that stays under the limit, using the `vram_cap` library in `scripts/vram_cap/` |
| `benchmark_qwen3_4b_0p6b_throughput/` | Compares GPU and CPU speed for the 0.6B and 4B models; `throughput_benchmark.py` here also holds helper functions the embedding runs use |
| `three_day_embedding_0p6b/` | Trial run on three days of data |
| `full_corpus_embedding_0p6b/` | The full run on all 902,144 messages |
| `faiss_gpu_exact_search_benchmark/` | Times exact nearest-neighbor search over the embeddings |

Each folder's README has the command to run it, its outputs and how to check them. Results
are in [`docs/METHODS.md`](../../docs/METHODS.md).
