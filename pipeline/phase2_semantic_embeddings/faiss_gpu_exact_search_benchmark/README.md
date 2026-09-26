# Exact nearest-neighbor search with FAISS on the GPU

Measures how long exact nearest-neighbor search takes over all 902,144 embeddings
(1,024 dimensions, float32, L2-normalized) on the RTX 3090. The embeddings are only read.

## Method

The index is `faiss.GpuIndexFlatIP`, a brute-force inner-product index: every query is
compared with every stored vector. Because the vectors are L2-normalized, inner product is
cosine similarity. No approximate index (IVF, HNSW, product quantization) or compression is
used.

The script, in one process:

1. Loads `embeddings.npy`, checks its shape, type, finiteness and norms, and builds the
   index on the GPU.
2. Picks 10,000 real documents as queries with a fixed seed.
3. Runs untimed warm-up searches.
4. Times the 10,000 queries at batch sizes of 256, 1,024, 4,096 and 8,192, and uses the
   fastest.
5. Times the main search: for each query, the 100 nearest other documents. Each query is
   also in the index, so its own row is removed by id. Removing it by rank would fail when
   duplicate texts tie at similarity 1.0.
6. Repeats the search with 20, 100 and 500 neighbors.
7. Checks exactness: for a few queries it recomputes the similarity to every document with
   numpy and compares the sorted scores with FAISS's. They agree to about 1e-7, and every
   difference in order is between tied scores.
8. Projects the time for all 902,144 queries from the measured rate. The full search is
   not run.

GPU memory is sampled every 20 ms with NVML for the whole card, other programs included.

## Results

| Measure | Value |
|---|---|
| Build the index on the GPU | 1.3 s |
| Load the embeddings from disk | 16 s (reading across from the Windows drive into WSL2) |
| GPU memory used by the index | about 7 GiB |
| Projected search of all 902,144 documents | about 1.5 minutes (90 to 96 s across repeated runs) |
| Exactness check | exact |

Speed barely changes across batch sizes from 256 to 8,192.

## Running it

It uses its own environment:
[`environment/requirements-search.txt`](../../../environment/requirements-search.txt)
(FAISS 1.14.1 for CUDA 12, under WSL2 Ubuntu 24.04).

```bash
python scripts/faiss_gpu_exact_search_benchmark.py
```

`--quick` runs a small version that still loads the full index. Other options, with their
defaults: `--n-queries 10000`, `--k 100`, `--seed 20260910`,
`--batch-sizes 256,1024,4096,8192`, `--neighbor-ks 20,100,500`, `--nvml-interval-ms 20`.
The whole run takes about 40 seconds.

It writes timing, memory, exactness and projection files to `results/`, and package
versions to `environment/`.
