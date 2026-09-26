# Methods and results

A condensed record of what each stage did and what it measured. All numbers below come from
the runs' own reports; nothing is estimated unless marked. Data: every submission and
comment posted to `r/politics` in July 2026 (UTC), from publicly available Reddit data.
Hardware: one NVIDIA RTX 3090 (24 GiB) under WSL2 Ubuntu 24.04.

## Phase 1: message-event foundation (R)

`pipeline/phase1_message_event_foundation/`

- Combines 7 comment files (891,397 rows) and 1 submission file (10,747 rows) into one
  canonical table: **902,144 rows, 29 columns**, one row per message.
- Adds a stable key (`doc_key`, the Reddit fullname), explicit thread and parent links with
  flags for whether the referenced message is present, and row-level provenance.
- Removes, filters and rewrites nothing. The source files are hashed before and after the
  run to prove they were only read.
- Every field is documented in [`DATA_DICTIONARY.md`](DATA_DICTIONARY.md).

## Phase 2: semantic embeddings (Python)

`pipeline/phase2_semantic_embeddings/`

Before the full run, four smaller studies settled the model, the batch size and the memory
limits on a fixed 21 GiB GPU budget:

| Study | Question | Result |
|---|---|---|
| `feasibility_qwen3_embedding_worst_case` | Can Qwen3-Embedding-4B and 8B embed the longest documents (up to 5,823 tokens) within budget? | Both handled the longest single document and a batch of the three longest; on the largest batch (eight documents) the 4B fit and the 8B ran out of memory. The two tokenizers are byte-identical, so one token count serves both |
| `feasibility_tei_qwen3_4b_batch_tokens` | Largest safe `max_batch_tokens` for the 4B model in Text Embeddings Inference | **92,160** (peak 20.8 GiB; the next value up, 92,672, went over). Enforced with `vram_cap`, below |
| `benchmark_qwen3_4b_0p6b_throughput` | GPU vs CPU speed on the same three documents | 0.6B model: **GPU 88.6× faster** than CPU (11,925 vs 135 tokens/s) |
| `faiss_gpu_exact_search_benchmark` | Is exact (brute-force) nearest-neighbour search practical? | Index built in 1.3 s; exact search of all 902,144 documents projected at **~1.5 min**; exactness verified independently |

**`vram_cap`** (`feasibility_tei_qwen3_4b_batch_tokens/scripts/vram_cap/`): Text Embeddings
Inference has no setting to cap its GPU memory. `vram_cap.c` is a drop-in replacement for
the CUDA driver library that forwards every call to the real driver but refuses
allocations that would push the process past a set limit, the same guarantee PyTorch gives
with `set_per_process_memory_fraction`. `gen_trampolines.py` generates forwarding stubs for
every other driver function.

**Full-corpus run** (`full_corpus_embedding_0p6b`), after a three-day trial run:

- Qwen3-Embedding-0.6B, float16, last-token pooling, L2-normalised, no truncation.
- **902,144 documents / 35,388,381 tokens in 15 minutes** (38,250 tokens/s end to end),
  peak GPU memory 18.4 GiB.
- Verified: one finite 1,024-dimension embedding per `doc_key`, no missing or extra keys,
  token totals reconciled to the token count table (the single empty document is
  embedded from a one-space placeholder, and that is recorded).

## Phase 3: structure discovery (Python, GPU)

`pipeline/phase3_blind_semantic_structure_discovery/`

- RAPIDS cuML UMAP (1,024 → 5 dimensions, cosine) followed by HDBSCAN on the GPU, plus a
  separate 2-D UMAP for visualisation only.
- **All three fits in 121 s** on the full corpus, peak GPU memory 4.9 GiB.
- **5,167 clusters**; 457,880 documents (50.8%) assigned, 444,264 left as noise. Cluster
  size: median 26, largest 24,862.
- A trial run on three days (101,722 documents) and a sensitivity/stability study came first.
- Verified: row alignment with `doc_key` preserved end to end, every coordinate finite, the
  input embeddings unchanged (hashes before and after).
- Clusters are measured, not named or interpreted, at this stage.

![2-D UMAP of all 902,144 documents](images/umap_density.png)

## Phase 4: describing the clusters (R)

`pipeline/phase4_semantic_structure_description_qc/`

- **4a:** reconnects every cluster to the original messages; computes c-TF-IDF keywords
  per cluster and picks representative documents (nearest the cluster centroid, and
  highest membership probability), with the parent comment for context.
- **4b:** per-cluster metadata (authors, timing, threads, links, reply context) and
  automated quality checks. Fully deterministic, with explicit tie-breaks.
- **4c:** selects a sample of clusters (the largest plus a random set) and builds a
  package for manual review. The manual review itself is the next step.

Nothing in Phases 1 to 4 infers coordination, disinformation or intent. Those questions
come after the clusters are described and checked by hand.

## Reproducibility practices used throughout

- Inputs are read-only and hashed before and after every run.
- Every output gets a `.manifest.yml` sidecar: input and output hashes, script hash,
  parameters, git commit and package versions.
- Fixed seeds where randomness is used (and recorded where it is not, e.g. unseeded UMAP).
- Each script records its wall-clock time and peak memory.
- Pure functions are unit-tested separately from the pipeline run.
