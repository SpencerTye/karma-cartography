# Phase 2 benchmark: steady-state throughput of Qwen3-Embedding-4B and 0.6B on GPU and CPU

## Corrected comparison (2026-09-10): CPU versus GPU on the same three documents

I re-specified the benchmark as a direct comparison of four conditions: Qwen3-Embedding-0.6B and 4B, each on the CPU and on the GPU, every batch made of
the same three real documents in the same order, with the same untimed warm-up and timed batches
everywhere. The original stages below were not run again; their results stay as measured (see
"Status" at the end).

- **Script:** `scripts/cpu_vs_gpu_three_documents.py` (stages `run --model M --device D` and
  `report`; helpers imported from `throughput_benchmark.py`); unit test
  `tests/test_cpu_vs_gpu_three_documents_functions.py`.
- **Documents:** Case B of `../feasibility_qwen3_embedding_worst_case/results/cases_4B.csv`, the
  three documents the Phase 2 work had already selected: `t1_oxogwqy` (5,823 tokens),
  `t1_oxwqnx9` (4,863) and `t1_ozvubg1` (3,123), in that order; their canonical `text`,
  unchanged.
- **Same code on both devices:** transformers `AutoModel`, `attn_implementation="sdpa"` with
  keys and values repeated to the query heads, `use_cache=False`, left padding (the three
  documents padded to 5,823 tokens), the model card's last-token pooling, L2 normalisation, no
  truncation, no prompt, no quantisation. Only the device and the precision differ: CPU float32,
  GPU bfloat16 (the format of the published weights; this CPU has no fast 16-bit arithmetic).
  TEI is not used here: its start-up warm-up embeds synthetic tokens, and it splits a single
  three-document request into several batches unless other requests keep it busy.
- **Batches:** 3 untimed warm-up batches (the first under the PyTorch profiler, which records the
  attention kernels), then 5 timed batches. Batch time = texts in to normalised float32
  embeddings in host memory (tokenisation, copies, forward pass, pooling, normalisation; the GPU
  synchronised).
- **Clock:** batch times are taken on the Windows host's performance counter (QPC). Just before
  and just after each batch the worker reads it through a small PowerShell process over WSL
  interop (a round trip of about 0.15 ms; half of each round trip is subtracted). As a
  cross-check the worker also writes a marker file before and after each batch on the Windows
  drive, and the run stage reads the files' Windows modification times
  (`host_clock_markers.csv`; the file writes add about 6 ms per batch). The VM's own clock
  (`perf_counter`) is recorded beside both but is not used for timing: measured against the
  Windows counter it ran 4.6% slow when idle, 5.1% slow with all eight CPUs busy and about 10%
  slow during a GPU smoke run. A first pass timed on it alone is kept, superseded, in
  `results/cpu_vs_gpu_three_documents_first_pass_vm_clock/`, with the Windows-side check that
  found the problem in its `vm_clock_check/`. The Windows clock itself was checked: its system
  time kept pace with its hardware counter to 1 part in 10 million over 30 s, and its offset
  from time.windows.com held at +0.86 s within 3 ms.
- **Envelope, both devices:** one worker process in a systemd scope with `MemoryMax` 21 GiB and
  no swap, 8 threads, one batch at a time; CUDA hidden from the CPU runs. **GPU ceiling:** 21 GiB
  of the card's total memory, other processes included, enforced by PyTorch's allocator cap
  (the ceiling minus the device total once the CUDA context exists) and the 20 ms NVML watchdog
  that kills the worker above the ceiling.
- **Outputs:** `results/cpu_vs_gpu_three_documents/` (per condition `run_<model>_<device>/`, with
  `host_clock_markers.csv`;
  combined `comparison.csv`, `speed_ratios.csv`, `precision_agreement.csv`,
  `timings_all_batches.csv`, `documents.csv`, `environment.json`, each with a sidecar) and the
  generated `CPU_VS_GPU_REPORT.md`.
- **Status (2026-09-10):** 0.6B GPU, 4B GPU and 0.6B CPU completed (3 untimed warm-up and 5
  timed batches each). The 4B CPU run was stopped by choice after its 3 untimed
  warm-up batches (411.1, 395.1 and 407.7 s on the Windows clock), so it has no timed result and
  the two comparisons that need it (4B GPU vs CPU, 4B vs 0.6B on the CPU) are not measured. The
  report is `CPU_VS_GPU_REPORT.md`.
- **Run** inside WSL2, one condition at a time (never two at once), then the report:

  ```
  wsl -d Ubuntu-24.04 -- bash -lc "~/venvs/socialmediadgg-phase2/bin/python '$REPO/pipeline/phase2_semantic_embeddings/benchmark_qwen3_4b_0p6b_throughput/scripts/cpu_vs_gpu_three_documents.py' --stage run --model 0.6B --device gpu"
  ```

  and the same with `--model 0.6B --device cpu`, `--model 4B --device gpu` and
  `--model 4B --device cpu`, then `--stage report`.

## What this benchmark does

It measures, on real documents of the canonical Phase 1 table (902,144 Reddit messages), how
fast two embedding models embed documents once they are warmed up, and what memory and
processors they use doing it, in four configurations:

| Configuration | How it runs | Precision | Batch budget |
|---|---|---|---|
| 4B GPU | Hugging Face Text Embeddings Inference (TEI) 1.9.3 release image on the RTX 3090, 21 GiB device ceiling | float16 | TEI `max_batch_tokens` = 92,160 (the confirmed safe value of the TEI 4B test; not searched again) |
| 0.6B GPU | the same | float16 | the largest `max_batch_tokens` directly measured as safe under the same ceiling (searched here) |
| 4B CPU | transformers + PyTorch on the 8 CPUs of the WSL2 VM, inside a 21 GiB memory scope, no GPU | float32 | the best point of a short batching sweep, never above the GPU budget of the same model |
| 0.6B CPU | the same | float32 | the same |

For every configuration it runs 3 warm-up batches (excluded from the statistics) and then 10
measured batches with the model loaded throughout, and reports the total, mean, median, minimum,
maximum and standard deviation of the batch time, the documents and tokens processed, documents
per second and tokens per second (the primary comparison), the documents and tokens of every
batch as TEI (or the CPU worker) processed them, and the memory, VRAM and CPU use. For the 0.6B
GPU it first finds the largest safe `max_batch_tokens`: bracket by doubling, bisection on a
512-token grid, and a confirmation rerun in a fresh container. It also compares, document by
document, the float16 GPU embedding with the float32 CPU embedding of the same documents (a
document is compared only with itself).

## What it does not do

No full-corpus embedding, no extrapolation of full-corpus runtime, no similarity analysis between
documents, no clustering or indexing, no choice between the two models. The 8B model is not
tested. Nothing under `../../../data_sample/`, `../../../legacy/scripts/` or Phase 1 is
written to.

## Inputs

| Input | Location | Role |
|---|---|---|
| canonical message-event table (902,144 rows) | `../../phase1_message_event_foundation/derived/message_events.parquet` (git-ignored; regenerated by the Phase 1 script), verified against its sidecar | the documents; field `text` (comment `body`; submission `title + " " + selftext`) |
| exact token count of every document | `../feasibility_qwen3_embedding_worst_case/results/token_counts_all_documents.parquet` (`n_tokens_4B`, including the appended `<\|endoftext\|>`), verified against its sidecar | batch packing; the 0.6B uses the same counts only if its tokenizer produces identical token ids on every document (checked in `prepare`) |
| the TEI 4B search summary | `../feasibility_tei_qwen3_4b_batch_tokens/results/search_summary.json`, verified against its sidecar | the 4B GPU budget (92,160, confirmed twice) |
| `Qwen/Qwen3-Embedding-4B` @ `5cf2132abc99cad020ac570b19d031efec650f2b` | the WSL2 Hugging Face cache (downloaded by the first feasibility test) | model |
| `Qwen/Qwen3-Embedding-0.6B` @ `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` | the WSL2 Hugging Face cache, downloaded and verified by `scripts/download_qwen3_embedding_0p6b.py` (every file equal to the Hub's SHA256 or git blob id; `environment/model_snapshot_0p6b.json`) | model |
| TEI release image | `ghcr.io/huggingface/text-embeddings-inference:86-1.9.3`, index digest `sha256:a7d82dfe...` | GPU server |
| allocation-cap shim | `../feasibility_tei_qwen3_4b_batch_tokens/scripts/vram_cap/` (rebuilt into `~/vram_cap/build` by `prepare`) | the 21 GiB VRAM ceiling inside TEI |

## Outputs

Everything under this folder; every output has a `.manifest.yml` provenance sidecar (input
hashes, output hash, the hash of the script as it was when the writing process started,
parameters, package versions, git commit). The measured embeddings (`*.npy`, hundreds of MB) are
git-ignored; their sidecars are tracked.

| Path | Stage | Content |
|---|---|---|
| `REPORT.md` | report | the answers, the comparison table, the speed ratios, the 0.6B GPU budget, every batch, the resources, the CPU sweeps, precision (explanation and measurement), the tokenizer check, the assumptions, the environment, the file list (generated; never edited by hand) |
| `results/prepare.json`, `results/tokenizer_identity.csv` | prepare | input checks; the 0.6B-versus-4B tokenizer comparison (files, `tokenizer.json` sections, `tokenizer_config.json` keys, vocabulary, special tokens, post-processor, token ids of all 902,144 documents); the benchmark stream |
| `results/gpu_calibration_4b.json`, `results/gpu_calibration_0p6b.json` (+ `_tokenize_check.csv`, `container_logs/calibration_*.log`) | gpu_calibrate | per model: the untracked device overhead, TEI's `/info`, TEI's own token count of the longest documents and of a stream sample against the stored counts, the embedding dimension |
| `results/gpu_search_0p6b/` | gpu_search | `search_summary.json` (largest safe value, confirmation, nearest failing value above it), `tested_values.csv` (one row per tested value: documents, tokens, peak VRAM, outcome), `batch_compositions.csv`, `tests/`, `container_logs/`, `shim_reports/` |
| `results/gpu_throughput_4b/`, `results/gpu_throughput_0p6b/` | gpu_throughput | `run.json` (steady-state statistics, resources, attempts), `attempt_<n>.json` (everything measured in one container), `batches.csv` (one row per batch: planned and TEI-counted documents and tokens, `x-compute-tokens`, verification, time), `batch_documents.csv`, `container_attempt_<n>.log`, `shim_attempt_<n>.json`, `measured_embeddings.npy` + `measured_embedding_doc_keys.csv` |
| `results/cpu_sweep_4b/`, `results/cpu_sweep_0p6b/` | cpu_sweep | `sweep.json`, `points.csv` (one row per budget tried), `batches.csv`, `worker_record.json`, `worker.log` |
| `results/cpu_throughput_4b/`, `results/cpu_throughput_0p6b/` | cpu_throughput | `run.json`, `batches.csv` (documents, real and padded tokens, forward passes, time, tokenisation time, busy CPUs, memory), `batch_documents.csv`, `worker_record.json`, `worker.log`, `measured_embeddings.npy` + `measured_embedding_doc_keys.csv` |
| `results/precision_agreement.csv`, `results/precision_agreement_documents_<model>.csv` | compare | per model and per document: cosine and largest absolute difference between the GPU (float16) and CPU (float32) embedding of the same document |
| `results/comparison_table.csv` | report | the comparison table |
| `environment/` | prepare, report | `tei_image.json`, `docker_versions.txt`, `nvidia_smi.txt`, `vram_cap_build.txt`, `model_snapshots.json`, `model_snapshot_0p6b.json`, `host.json` (VM CPUs, memory, swap, `.wslconfig`), `runtime_record.csv`, `python_packages.csv` |

## Environment

| Component | Value |
|---|---|
| distro | WSL2 Ubuntu-24.04 (kernel 6.18), user `spence`; the VM is limited by `.wslconfig` to 8 logical processors and 56 GB of memory (16 GB swap); not changed by this benchmark |
| GPU server | TEI `86-1.9.3` in Docker Engine 29.8 with the NVIDIA Container Toolkit inside the distro, `--gpus all --network host`, the Hugging Face cache mounted read-only, `HF_HUB_OFFLINE=1` |
| CPU worker | `~/venvs/socialmediadgg-phase2` (Python 3.12.3, torch 2.8.0+cu128, transformers 5.17.0, tokenizers 0.23.2), run through `systemd-run --user --scope -p MemoryMax=22548578304 -p MemorySwapMax=0` |
| hardware | NVIDIA GeForce RTX 3090 (24 GiB, driver 596.49), AMD Ryzen 9 3900X (12 cores, 24 threads; the VM sees 8), 64 GB RAM |

## How to run

Every command runs inside WSL2 in the venv; the script finds the project root from its own
location. Download and verify the 0.6B model once:

```
wsl -d Ubuntu-24.04 -- bash -lc "~/venvs/socialmediadgg-phase2/bin/python '$REPO/pipeline/phase2_semantic_embeddings/benchmark_qwen3_4b_0p6b_throughput/scripts/download_qwen3_embedding_0p6b.py'"
```

Run the whole benchmark (the GPU stages take about half an hour; the CPU stages take much longer,
mostly the 4B):

```
wsl -d Ubuntu-24.04 -- bash -lc "~/venvs/socialmediadgg-phase2/bin/python '$REPO/pipeline/phase2_semantic_embeddings/benchmark_qwen3_4b_0p6b_throughput/scripts/throughput_benchmark.py' --stage all"
```

Or stage by stage, in this order: `--stage prepare`; `--stage gpu_calibrate --model 4B` and
`--model 0.6B`; `--stage gpu_search`; `--stage gpu_throughput --model 4B` and `--model 0.6B`;
`--stage cpu_sweep --model 0.6B`, `--stage cpu_throughput --model 0.6B`, the same for `4B`;
`--stage compare`; `--stage report`. `python throughput_benchmark.py --help` lists every
parameter with its default (ceiling 21,504 MiB for both VRAM and the CPU scope, 3 warm-up and
10 measured batches, 8 CPU threads, sweep from 1,024 tokens with 1 warm-up and 3 timed batches per
point and a 5% improvement threshold, 512-token search grid, 5,000 search candidates, NVML
sampling every 20 ms). GPU and CPU stages must not run at the same time: they share the VM's
CPUs.

Unit test of the pure functions (synthetic inputs, no data, no Docker, no GPU, about a second):

```
wsl -d Ubuntu-24.04 -- bash -lc "~/venvs/socialmediadgg-phase2/bin/python '$REPO/pipeline/phase2_semantic_embeddings/benchmark_qwen3_4b_0p6b_throughput/tests/test_throughput_benchmark_functions.py'"
```

It prints `test_throughput_benchmark_functions: all checks passed`. A smoke test of every stage
with reduced settings writes to another folder: add `--quick --results-dir <dir>
--environment-dir <dir> --report-path <file>` and small settings (`--budget`, `--n-warmup 1
--n-measured 2`, `--cpu-sweep-start 256 --cpu-max-budget 512 --cpu-sweep-timed 1`,
`--ceiling-mib 7000` for the search).

## Measurement protocol

- **Documents and batches.** The benchmark stream holds the canonical documents in a seeded
  order: each document's position is the BLAKE2b-64 hash of `20260910:<doc_key>`, so the order
  depends only on the seed and the key. The one empty document (`t1_ovhv2aj`, no text) is left
  out. Warm-up batches are packed from the start of the stream and measured batches from its
  midpoint, so the measured documents of all four configurations begin at the same document.
  Every batch is packed by TEI's own rule: take documents while the real-token sum stays within
  the budget; the first document that does not fit starts the next batch. The same stream and rule
  serve every configuration; a different budget gives different batch boundaries. Texts are the
  canonical `text`, unchanged: nothing synthetic, nothing duplicated.
- **GPU (TEI).** One container per run: `--model-id <local snapshot>`, `--max-batch-tokens
  <budget>`, `--auto-truncate false`, `--max-client-batch-size 100000`,
  `--max-concurrent-requests 200000`, `--payload-limit 100000000`, no `--max-batch-requests`, so
  documents per batch are limited only by the token budget. float16 (TEI's only precision for its
  flash Qwen3 implementation), last-token pooling from the model's configuration, requests with
  `truncate: false, normalize: true`, no prompt, no quantisation. The 21 GiB ceiling (21,504 MiB of
  total device memory, other processes included) is enforced exactly as in the TEI 4B test: the
  `vram_cap` shim caps TEI's physical device memory at ceiling minus the device total before the
  start minus the calibrated untracked overhead of the model, and a 20 ms NVML watchdog kills the
  container if a sample exceeds the ceiling. A run counts only if every sample stayed at or below
  the ceiling.
- **GPU batch release.** Two single-document blockers (the two longest documents) are sent
  first, then batch 1. Each later batch is sent as soon as TEI's `/metrics`, polled every 5 ms,
  show that every earlier document has been batched. At that moment the GPU is still running an
  earlier batch, so the new batch waits in TEI's queue and TEI forms it as one batch the moment the
  backend takes the batch before it: the GPU runs the planned batches back to back. For every
  batch, the Prometheus deltas of `te_batch_next_size` and `te_batch_next_tokens` must show exactly
  one new batch (the first interval: the two blockers and batch 1) with exactly the planned
  documents and tokens, and the request's `x-compute-tokens` must equal the planned tokens. A run
  whose measured batches are not all verified is repeated in a fresh container, up to three
  attempts. The time of a batch is TEI's backend duration for it (`x-inference-time`): the forward
  pass, pooling and normalisation plus the copy of the embeddings to the host. Tokenisation,
  queueing, HTTP and JSON are excluded, and so are model loading and TEI's own start-up warm-up.
- **0.6B GPU budget search.** The search follows the TEI 4B test's protocol with a fresh
  container per value: TEI's start-up warm-up of `max_batch_tokens` synthetic tokens, then a batch
  of the longest real documents filled first-fit-decreasing to exactly the value, sent behind the
  two blockers and verified as one TEI batch. It starts at 32,768 (TEI's minimum for the 0.6B with
  truncation disabled) and doubles until the first failure, then bisects on a 512-token grid, then
  reruns the safe value in a fresh container. A failed confirmation becomes the failing bound and
  the search continues. Longest-first batches make the budget safe for any batch of the corpus.
- **CPU.** transformers `AutoModel` in float32 with `attn_implementation="sdpa"`,
  `use_cache=False`, left padding, the model card's last-token pooling, L2 normalisation, no
  truncation, no prompt. transformers' `use_gqa_in_sdpa` is forced to False, so keys and values
  are repeated to the query heads and PyTorch's memory-efficient CPU attention kernel runs instead
  of the kernel that materialises the attention matrix; the embeddings are unchanged. TEI's CPU
  image is not used because its CPU backend limits every batch to 4 documents. A CPU batch is a
  planned batch of real tokens within the budget. Because the CPU model pads, it runs in forward
  passes of at most the budget in padded tokens, documents sorted by length; a document longer
  than the budget runs alone. The batch time is the forward passes plus pooling and
  normalisation; tokenisation is timed separately.
- **CPU resources equal to the GPU runs'.** One worker process in a transient systemd scope with
  `MemoryMax` = 21 GiB and `MemorySwapMax` = 0 (the kernel kills the worker at the ceiling: a
  failure at the ceiling, like a refusal on the GPU). It uses 8 PyTorch compute threads on the
  VM's 8 logical CPUs, the CPUs the TEI container had. It processes one batch at a time, as TEI's
  backend does, with the model loaded once per worker. `CUDA_VISIBLE_DEVICES` is empty, and the
  GPU's memory is sampled throughout to show the worker used none.
- **CPU sweep.** The budget starts at 1,024 tokens and doubles, never above the GPU budget of the
  same model (the last point is that budget). Each point runs 1 warm-up and 3 timed batches from
  the start of the stream, with the model loaded once for the whole sweep. The sweep stops at the
  first point whose tokens per second are not at least 5% above the best earlier point
  (improvement stopped or throughput worsened), at a failure at the ceiling, or at the GPU budget.
  A fresh worker then runs the 3 warm-up and 10 measured batches at the point with the highest
  tokens per second.
- **Resources.** GPU: the device total before the container starts (other processes), after load
  and TEI's start-up warm-up (the model-loaded baseline), and its peak during start-up, the warm-up
  batches, the measured batches and overall, sampled every 20 ms; TEI's own physical device memory
  (shim); the container's RAM (cgroup `memory.peak`, anonymous memory) and the VM's memory in use.
  CPU: the scope's memory peak (cgroup, page cache included), the worker's resident and anonymous
  memory (sampled, plus the kernel's per-batch high-water mark), the scope's CPU time per batch
  (busy CPUs = CPU seconds / batch seconds), the compute threads and the process's OS threads.
- **Precision check.** The GPU run keeps the embeddings of the first 30,000 documents of its
  measured batches; the CPU run keeps all of its measured documents, which are the first documents
  of the GPU's measured region. For every document in both, the cosine and the largest absolute
  difference between its float16 (GPU) and float32 (CPU) embedding are recorded. Non-finite
  values are counted in every measured batch.

## What to check

1. `results/prepare.json`: `input_checks.*.hash_equal_to_sidecar` true; `tokenizer` (files,
   sections, `n_token_ids_equal` of 902,144); `stream.excluded_doc_keys` = `["t1_ovhv2aj"]`.
2. `results/gpu_calibration_*.json`: `tokenize_check.all_equal` true; `info_checks`
   (`auto_truncate_false`, `dtype_float16`, `pooling_last_token`, `max_batch_requests_unset` true).
3. `results/gpu_search_0p6b/search_summary.json`: `largest_safe`, `confirmation_succeeded` true,
   `nearest_failing_above`, `aborted` null; `tested_values.csv`: `verified_single_batch` true for
   every started test, `overall_peak_under_ceiling` true for every success.
4. `results/gpu_throughput_*/batches.csv`: `verified` true, `tei_documents` = `n_documents`,
   `tei_tokens` = `x_compute_tokens` = `n_tokens` for every measured batch; `run.json`:
   `accepted_attempt` set, `resources.overall_peak_device_bytes` at or below 22,548,578,304.
5. `results/cpu_throughput_*/run.json`: `complete` true, `outcome` success; `worker_threads`
   (8 threads, `cuda_available` false, `memory_max` 22548578304, `memory_swap_max` 0);
   `batches.csv`: `counts_equal_attention_mask` and `embeddings_finite` true.
6. `results/precision_agreement.csv`: documents compared, cosines, `n_nonfinite` 0.
7. `REPORT.md`: the answers and the comparison table.

## Status of the 2026-09-10 run: incomplete, to be redone

The benchmark was not completed and has no `REPORT.md`. I redid it.
The committed `results/` hold:

- `prepare`: the 0.6B tokenizer gives the same token ids as the 4B on all 902,144 documents.
- Both GPU calibrations.
- The 0.6B search: 370,688 confirmed safe and 371,200 failing, found with Windows' own VRAM
  (about 1.74 GB) counted inside the 21 GiB.
- The 4B CPU sweep and run: 2,048 tokens per batch, 31.4 tokens/s, 0.85 documents/s.
- The 0.6B CPU sweep and run: 1,024 tokens per batch, 185.1 tokens/s, 5.11 documents/s.
- The GPU throughput attempts, all refused at the cap. The 4B at 92,160 and the 0.6B at 370,688
  were refused in three attempts each, because the ceiling was applied to the whole card while
  Windows programs held 1.6 to 1.8 GB.

The redo is the corrected comparison at the top of this README.
It ran under the stricter reading of the 21 GiB ceiling (the whole card, other processes
included).

## Known limitations

1. Timings are measurements on a shared desktop machine (the Windows desktop and other WSL
   distros run at the same time). The CPU runs are sensitive to other load on the host: the VM's 8
   virtual CPUs are scheduled on the host's 24 logical processors.
2. The GPU and CPU software differ by necessity: TEI's candle implementation on the GPU (float16,
   varlen flash attention, no padding), PyTorch on the CPU (float32, padded batches, a
   memory-efficient attention kernel). The configurations share the model files, the tokenizer,
   the pooling, the normalisation, the documents, the batching rule and the resource envelope.
3. TEI reports `x-inference-time` in whole milliseconds.
4. The 4B GPU budget (92,160) was established with the desktop holding about 1.1 to 1.4 GB of
   VRAM. A larger desktop share at run time lowers the cap left for TEI; the share is recorded
   with every run.
5. The sidecar hashes are of the files as the script wrote them in WSL2 (LF line endings). The
   repository's `core.autocrlf=true` rewrites text outputs to CRLF on a fresh checkout, so their
   hashes match only in this working copy or after a checkout with `core.autocrlf=false`. The
   scripts are pinned to LF by the folder's `.gitattributes`.
