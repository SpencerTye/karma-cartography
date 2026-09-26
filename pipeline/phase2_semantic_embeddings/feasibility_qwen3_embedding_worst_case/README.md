# Worst-case test: Qwen3-Embedding 4B and 8B on the longest documents

Two questions about the two candidate models, `Qwen/Qwen3-Embedding-4B` and `-8B`, on one
RTX 3090 and the Phase 1 table (902,144 messages):

1. How long is every document in the models' own tokens?
2. How much GPU memory and time does it take to embed the longest documents, with a limit of
   21 GiB of total GPU memory?

For question 2, each model (unquantized, bfloat16) embeds the longest document alone (case
A), the three longest in one batch (case B), and the eight longest in one batch (case C),
each in a single forward pass with no truncation.

## Results (run of 2026-09-09)

Token lengths: median 22 tokens, 99th percentile 253, maximum 5,823. No document is longer
than 8,192 tokens, far under the models' 32,768-token limit. The two models' tokenizers are
identical (same files, vocabulary and special tokens, and the same ids on a 2,000-document
sample), so one tokenization pass (38 s for all documents) serves both.

| Case | 4B: peak GPU memory, time | 8B: peak GPU memory, time |
|---|---|---|
| Loaded, before any case | 10,200 MiB | 17,087 MiB |
| A: longest document | 10,871 MiB, 1.22 s | 17,777 MiB, 1.53 s |
| B: three longest | 12,173 MiB, 2.76 s | 19,153 MiB, 3.86 s |
| C: eight longest | 15,570 MiB, 5.92 s | out of memory after 0.35 s |

Peak memory is for the whole card, including about 2 GB used by the Windows desktop. The
embedding dimension is 2,560 for the 4B and 4,096 for the 8B. The same document embedded
alone, in the batch of three and in the batch of eight gives embeddings with cosine
similarity of at least 0.9997.

Two things learned from reading the transformers 5.17 source, then confirmed in the runs:

- Unless `use_cache=False` is passed, the Qwen3 model builds a key-value cache it doesn't
  need for embedding (819 MiB for the longest document). All cases pass `use_cache=False`.
- With `attn_implementation="flash_attention_2"`, padded batches are unpadded and run
  through flash attention's variable-length kernel. The profiler showed only the flash
  attention kernel, called once per layer (36 times per forward pass).

## Method

- The model card's usage: left padding, flash attention 2, bfloat16, last-token pooling and
  L2 normalization, with no truncation. Each model runs in its own process.
- The 21 GiB limit covers the whole card. Once the CUDA context exists, PyTorch's allocator
  is capped with `set_per_process_memory_fraction` at the limit minus what is already in use,
  so an allocation past the limit raises an out-of-memory error instead of spilling into
  system memory. That error counts as the result. A thread samples total GPU memory with NVML
  every 20 ms.
- Before case A a short warm-up pass initializes the GPU kernels. Memory statistics are reset
  between cases.

## Running it

This runs under WSL2 (Ubuntu 24.04) because flash-attention only publishes prebuilt wheels
for Linux. `scripts/setup_wsl2_env.sh` builds the environment (Python 3.12, torch 2.8,
flash-attn 2.8.3, transformers 5.17) and downloads both models at pinned revisions, about
30 GB. The models and environment need at least 30 GB free on the drive that holds the WSL2
virtual disk.

```bash
bash scripts/setup_wsl2_env.sh
python scripts/phase2_feasibility_qwen3_worst_case.py --stage all
```

The stages can also be run one at a time: `--stage tokens`, `--stage bench --model 4B`,
`--stage bench --model 8B`, `--stage report`. `--help` lists the options (for example
`--vram-ceiling-mib 21504` and `--top-n 8`). The whole run takes under four minutes.

The unit test uses made-up inputs and needs no data or GPU:

```bash
python tests/test_phase2_feasibility_functions.py
```

## Outputs

Written to `results/` and `environment/`: the token count of every document, a token-length
summary, the eight longest documents per model, every measurement for each case and model,
the tokenizer comparison, package versions and `nvidia-smi` output. A `REPORT.md` with the
side-by-side table is generated from them.

## Limitations

- Memory used by other programs counts against the limit and changes between runs. Each
  result records the GPU memory in use before the model loaded.
- Under WSL2, NVML doesn't report per-process GPU memory, so the process's share is the card
  total minus what was in use before it started. PyTorch's out-of-memory message shows a
  meaningless per-process figure for the same reason.
- Times are single measurements on a desktop machine, not repeated benchmarks.
- Only the eight longest documents are embedded.
