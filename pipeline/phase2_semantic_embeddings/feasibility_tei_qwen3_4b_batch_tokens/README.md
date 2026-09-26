# Batch-size limit: Qwen3-Embedding-4B in Text Embeddings Inference

What is the largest `--max-batch-tokens` at which Hugging Face Text Embeddings Inference
(TEI) can serve `Qwen/Qwen3-Embedding-4B` on the longest real documents without the GPU
ever holding more than 21 GiB? TEI groups requests into batches up to this many tokens, so
this one setting decides how much GPU memory a TEI deployment uses.

**Answer: 92,160 tokens.** That value ran as a single batch of 42 documents, peaking at
21,294.5 MiB, and again in a confirmation run at 21,389.0 MiB, both under the 21,504 MiB
limit. The next value tested, 92,672, went over.

## Why a custom memory cap was needed

TEI has no setting to limit its GPU memory. Under WSL2 the driver also doesn't stop a program
at the card's physical memory: a test program was allowed to allocate 23.5 GB on the 24 GB
card while the driver quietly paged the rest to system memory. So the limit had to be
enforced inside TEI's own process.

`scripts/vram_cap/vram_cap.c` does that. It is built as `libcuda.so.1` and placed first on
`LD_LIBRARY_PATH` inside the container, so TEI loads it in place of the real CUDA driver
library. It passes every call through to the real driver and keeps count of the GPU memory
the process holds:

- memory from the ordinary allocation calls (`cuMemAlloc`, `cuMemAllocPitch`,
  `cuMemAllocManaged`, `cuMemCreate`), minus what has been freed, and
- the memory reserved by the stream-ordered memory pool. TEI's backend allocates every
  tensor through `cuMemAllocAsync`, and freed blocks stay reserved in the pool until the next
  synchronization, so the pool's reserved size is what actually occupies the GPU.

An allocation that would take the process past the cap gets `CUDA_ERROR_OUT_OF_MEMORY`, the
same error TEI would see if the card were full. If a pool allocation grows the pool past the
cap, the growth is undone within milliseconds and the call is refused. The library also
intercepts `cuGetProcAddress`, so the CUDA runtime and cuBLAS go through the same checks.
`gen_trampolines.py` generates pass-through stubs for every other driver function from the
real library's export table, and `build_vram_cap.sh` compiles it all.

The cap for each run is the 21 GiB limit, minus the GPU memory already in use before the
container starts, minus the memory the CUDA context takes, which no allocation call reports
(measured once in a calibration run). As a second safeguard, a thread samples the card's
total memory with NVML every 20 ms and stops the container if it ever passes the limit.

## Method

- Every value is tested in a fresh container. TEI starts by embedding a synthetic batch of
  exactly `max_batch_tokens` tokens, which is its own warm-up. Then a batch of the longest
  real documents, filled to exactly that many tokens, is sent.
- TEI would normally split one request across several batches on an idle server. To force a
  single batch, the script first sends two one-document requests that keep the backend busy,
  then the test documents in one request. TEI's Prometheus metrics must then show exactly
  three batches (1, 1 and all the test documents), and TEI's own token count for the request
  must equal the planned total. Otherwise the test is repeated.
- Search: start at 40,960 tokens (TEI's minimum for this model with truncation off), double
  until a value fails, then bisect on a 512-token grid, then rerun the best safe value in a
  fresh container to confirm it.
- A value counts as safe only if TEI started, the batch ran and was verified as one batch,
  and every memory sample stayed under the limit.

TEI settings: `--auto-truncate false`, float16 (the only precision TEI's flash Qwen3
implementation accepts), last-token pooling from the model's config, L2-normalized output,
no prompt, no quantization.

## Results (run of 2026-09-10)

| Tokens | Result | Peak GPU memory (MiB) |
|---:|---|---:|
| 32,768 | TEI refuses to start: below its minimum with truncation off | |
| 40,960 | safe | 14,679.9 |
| 81,920 | safe | 20,229.2 |
| 163,840, 122,880, 102,400, 97,280 | refused by the cap during warm-up | |
| 92,160 | safe | 21,294.5 |
| 94,720 | over the limit | 21,607.6 |
| 93,184 | safe | 21,461.6 |
| 93,696 | over the limit | 21,516.3 |
| 93,184 (confirmation) | over the limit | 21,760.0 |
| 92,672 | over the limit | 21,648.7 |
| 92,160 (confirmation) | safe | 21,389.0 |

93,184 succeeded once but failed its confirmation, so it was not accepted. Repeated runs of
the same value varied by up to about 300 MiB, mostly because the Windows desktop's share of
GPU memory moves, which is why the answer had to succeed twice. The search used 14
containers and took 6.5 minutes. With the 92,160-token batch, TEI took about 12 s to embed
42 documents.

## Running it

Under WSL2 (Ubuntu 24.04), with Docker Engine and the NVIDIA Container Toolkit installed in
the distro (`scripts/setup_docker_wsl2.sh`, run once as root), and the image
`ghcr.io/huggingface/text-embeddings-inference:86-1.9.3`. Building `vram_cap` needs gcc.

```bash
docker pull ghcr.io/huggingface/text-embeddings-inference:86-1.9.3
python scripts/tei_batch_token_limit.py --stage all   # about 25 minutes
```

The stages can also be run one at a time: `--stage prepare` (checks the inputs and builds
`vram_cap`), `--stage search`, `--stage report`. `--help` lists the options (for example
`--ceiling-mib 21504` and `--resolution 512`).

The unit test uses made-up inputs and needs no data, Docker or GPU:

```bash
python tests/test_tei_batch_token_limit_functions.py
```

## Limitations

- The cap counts memory from the allocation calls. The CUDA context and module code are
  outside it and are covered by the calibration and the NVML check.
- Memory used by other programs counts against the limit and changes between runs. It is
  recorded for every test.
- TEI's warm-up uses synthetic sequences, but it is part of starting TEI at a given setting,
  so a value whose warm-up doesn't fit counts as a failure.
- The result applies to TEI 1.9.3 on this GPU and driver.
