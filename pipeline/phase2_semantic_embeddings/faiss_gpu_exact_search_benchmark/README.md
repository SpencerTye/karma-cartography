# FAISS GPU exact nearest-neighbor search benchmark

A performance/feasibility test of **exact, exhaustive** nearest-neighbor search
over the full-corpus Qwen3-Embedding-0.6B embeddings (902,144 × 1024, float32,
L2-normalized) on the RTX 3090. See `REPORT.md` for results; this file covers
method, environment, and reproduction.

The benchmark reads the existing embeddings **read-only** (it never rebuilds or
alters them) and writes only small evidence files — no all-corpus neighbor
table.

## Inputs (used directly, not rediscovered)

- Embeddings: `../full_corpus_embedding_0p6b/results/embeddings.npy`
  (float32 `[902144, 1024]`, 3.44 GiB, git-ignored)
- Row→doc_key map: `../full_corpus_embedding_0p6b/results/embedding_index.parquet`

## What the index is

`faiss.GpuIndexFlatIP` — a **flat** (brute-force) inner-product index on the
GPU. Each query is compared against every one of the 902,144 stored vectors;
nothing is pruned. The stored vectors are L2-normalized, so inner product =
cosine similarity and the ranking is a cosine ranking.

**Deliberately excluded** (per the task): IVF, HNSW, PQ, any quantization,
vector compression, clustering, UMAP, HDBSCAN, and any approximate search. The
vectors stay **float32**. Exactness is then *verified independently* rather than
assumed from the index name (see "Exactness check").

## Isolated environment (nothing else was touched)

FAISS GPU was not present in the project's Phase 2 venv. A **separate** venv was
created so no unrelated environment or package was modified:

```bash
uv venv /home/spence/venvs/faiss-bench --python 3.12
uv pip install --python /home/spence/venvs/faiss-bench/bin/python \
    faiss-gpu-cu12 numpy pyarrow psutil nvidia-ml-py
```

This installed **FAISS 1.14.1** (`faiss-gpu-cu12==1.14.1.post1`, compile options
`OPTIMIZE GPU`) with a bundled CUDA 12.9 runtime (cuBLAS 12.9.2.10, cudart
12.9.79, nvrtc 12.9.86). `faiss.get_num_gpus()` returns 1 on the RTX 3090
(driver 596.49) under WSL2 Ubuntu-24.04. The project's
`socialmediadgg-phase2` venv was left unchanged. Frozen list:
`environment/pip_freeze.txt`.

## How to run

From Windows PowerShell (the script runs in WSL2 and reads `S:` via `/mnt/s`):

```powershell
wsl.exe -d Ubuntu-24.04 -- /home/spence/venvs/faiss-bench/bin/python \
  "$REPO/pipeline/phase2_semantic_embeddings/faiss_gpu_exact_search_benchmark/scripts/faiss_gpu_exact_search_benchmark.py"
```

- `--quick` runs a tiny smoke version (still loads the full index; exercises
  every code path) for validation.
- Key flags (defaults in parentheses): `--n-queries` (10000), `--k` (100
  *other* docs kept), `--seed` (20260910), `--batch-sizes` (256,1024,4096,8192),
  `--neighbor-ks` (20,100,500), `--n-exact` (6), `--exact-k` (110),
  `--nvml-interval-ms` (20), `--main-batch` (0 = auto-pick the fastest swept
  size).

The whole run takes ~40 s (index load ~16 s + build ~1.3 s + fast query work).

## Method / stages (single process, index loaded once)

1. **Load + build.** Read `embeddings.npy` into host RAM (float32, contiguous);
   verify shape/dtype/finiteness/L2-norms; build `GpuIndexFlatIP` and `add()`
   all vectors. Record load/build time, whole-card GPU memory before/after
   (NVML), host RAM (RSS), and `ntotal`/`d`.
2. **Query selection.** A fixed random sample of 10,000 real corpus rows
   (`numpy.random.default_rng(seed)`), so the queries are real embeddings drawn
   from the same 902,144 documents (no synthetic vectors).
3. **Warm-up.** Untimed searches (covering k up to 500) before any timing.
4. **Batch-size sweep.** A few sizes (256/1024/4096/8192), each searching all
   10,000 queries; pick the fastest (the plateau is flat, so this is not a
   memory search). Peak memory recorded per size.
5. **Main timed test.** 10,000 queries, request k+1 = 101, drop the query's own
   row, keep the 100 closest *others*; consecutive timed batches; per-batch and
   aggregate timing + peak whole-card memory.
6. **Neighbor-count comparison.** Same queries at k = 20 / 100 / 500 to measure
   k's effect on throughput.
7. **Exactness check** (below).
8. **Full-corpus extrapolation** from the measured sustained rate (the
   902,144-query search itself is **not** run).

## Self-match handling

Every query is in the database, so its own row is retrieved and removed by id
(not by rank), then the 100 closest others are kept. This is robust to the
corpus's exact-duplicate texts, which tie at cosine 1.0 and can put a duplicate
ahead of — or in place of — the query's own row (see `REPORT.md` §9).

## Exactness check

For a few real queries, similarities against the entire float32 matrix are
recomputed with an independent numpy fp32 matmul (`xb @ q`) and compared to
FAISS. The tie-robust criterion is agreement of the **sorted score vectors**;
identity/order differences are only accepted where the scores are equal (ties).
Verified: score agreement (~1e-7, float32 rounding), that all identity
differences are tie-breaking, and that self-similarity is the global maximum.
Result: **EXACT** (`results/exactness_validation.json`).

## GPU-memory measurement

Whole-card `used` bytes via NVML (nvidia-ml-py), sampled every 20 ms by a
background thread that tracks the peak; cross-checked once against `nvidia-smi`.
This counts all processes on the card (including the Windows desktop apps ~1.6
GiB), matching the project's whole-card 21 GiB ceiling convention. The
`vram_cap` allocation-cap shim used elsewhere in Phase 2 was **not** needed
here: the index sits at ~7 GiB, far under the ceiling, so there was no boundary
to enforce.

## Output inventory

```
faiss_gpu_exact_search_benchmark/
├── README.md                     # this file
├── REPORT.md                     # results / evidence (13 report-back items)
├── scripts/
│   └── faiss_gpu_exact_search_benchmark.py
├── results/
│   ├── run.json                  # master record (config + all sections)
│   ├── index_load.json           # load/build time, GPU mem before/after, host RAM, presence
│   ├── batch_sweep.csv / .json    # query-batch-size measurements
│   ├── main_batches.csv          # per-batch timing (main test)
│   ├── main_summary.json         # main-test totals + self-match stats
│   ├── neighbor_count_comparison.csv / .json   # 20/100/500 neighbors
│   ├── exactness_validation.json # exactness (summary + per-query)
│   └── full_corpus_estimate.json # extrapolation
└── environment/
    ├── environment.json          # versions, GPU, packages (machine-readable)
    ├── pip_freeze.txt            # frozen faiss-bench venv
    ├── faiss_build.txt           # FAISS version / compile options / num_gpus
    └── nvidia_smi.txt            # card state at capture
```

## Caveats

- The `embeddings.npy` load time (~16 s) is dominated by the cross-mount read
  from the Windows `S:` drive into WSL2; a same-filesystem copy would load
  faster. It is a one-time cost and separate from search time.
- The 10,000-query timing spans ~1 s; repeated runs varied within ~±6%
  (projected full-corpus time ~90–96 s). Treat the headline figures as
  approximate rates, not exact constants.
- Throughput is nearly flat across batch sizes 256–8192, so the auto-picked
  batch (fastest) can differ run-to-run without materially changing the result.
```

*This benchmark makes no recommendation about exact vs approximate search; it
reports measurements so that decision can be made from the evidence.*
