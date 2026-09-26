# GPU vs CPU benchmark: Qwen3-Embedding 0.6B and 4B

Measures how fast the two embedding models run on the GPU and on the CPU once they are
warmed up, and how much memory they use. This folder also holds `throughput_benchmark.py`,
whose helper functions (the TEI client, NVML memory sampler, token packer and manifest
writer) are imported by the embedding runs.

There are two versions of the benchmark here.

## 1. Same three documents on every device (the finished comparison)

Script: `scripts/cpu_vs_gpu_three_documents.py`. Test:
`tests/test_cpu_vs_gpu_three_documents_functions.py`.

Every batch is the same three long documents from the corpus (5,823, 4,863 and 3,123
tokens), in the same order, on both devices. Both devices run the same code: transformers
`AutoModel` with SDPA attention, left padding, last-token pooling and L2 normalization. Only
the device and the precision differ (CPU float32, GPU bfloat16). Each condition runs 3
untimed warm-up batches and then 5 timed batches, inside the same limits: a 21 GiB memory
cap, no swap and 8 CPU threads.

Results (from the report that the `report` stage writes; outputs are not in the repository):

| Condition | Mean batch time | Tokens/s | Peak GPU memory |
|---|---:|---:|---:|
| 0.6B GPU | 1.16 s | 11,925 | 4.3 GiB |
| 0.6B CPU | 102.6 s | 135 | none |
| 4B GPU | 4.06 s | 3,405 | 11.9 GiB |
| 4B CPU | stopped after the warm-up batches (about 400 s each) | | |

For the 0.6B model the GPU is 88.6 times faster than the CPU.

Timing note: the clock inside the WSL2 virtual machine turned out to run 4.6% to 10% slow
compared with the Windows host's clock, depending on load. Batch times are therefore taken
from the Windows performance counter, read through a small PowerShell call before and after
each batch, and cross-checked with file timestamps on the Windows drive. A first pass timed
with the VM clock was discarded.

Run inside WSL2, one condition at a time, then the report:

```bash
python scripts/cpu_vs_gpu_three_documents.py --stage run --model 0.6B --device gpu
python scripts/cpu_vs_gpu_three_documents.py --stage run --model 0.6B --device cpu
python scripts/cpu_vs_gpu_three_documents.py --stage run --model 4B --device gpu
python scripts/cpu_vs_gpu_three_documents.py --stage report
```

## 2. Steady-state throughput on the real document stream (incomplete)

Script: `scripts/throughput_benchmark.py`. Test: `tests/test_throughput_benchmark_functions.py`.

The first design compared four configurations (0.6B and 4B, each on GPU and CPU) on batches
drawn from the whole corpus in a seeded order, with the GPU served by Text Embeddings
Inference. It also searched for the largest safe batch size for the 0.6B model on the GPU,
using the same method as the [4B batch-size test](../feasibility_tei_qwen3_4b_batch_tokens/).

What finished:

- The 0.6B tokenizer gives the same token ids as the 4B on all 902,144 documents.
- The 0.6B batch-size search: 370,688 tokens safe, 371,200 over the limit.
- CPU runs: 4B at 31.4 tokens/s, 0.6B at 185.1 tokens/s.

The GPU throughput runs were refused by the memory cap every time. The 21 GiB limit was
applied to the whole card, and Windows programs were already using 1.6 to 1.8 GB of it.
Version 1 above replaced this comparison.

```bash
python scripts/download_qwen3_embedding_0p6b.py      # download and verify the 0.6B model once
python scripts/throughput_benchmark.py --stage all   # or one stage at a time; see --help
```

## Requirements

WSL2 (Ubuntu 24.04) with Python 3.12, torch 2.8 and transformers 5.17
([`environment/requirements-embeddings.txt`](../../../environment/requirements-embeddings.txt)).
The TEI runs need Docker with the NVIDIA Container Toolkit and the image
`ghcr.io/huggingface/text-embeddings-inference:86-1.9.3`. The models are pinned to fixed
revisions and loaded from the local Hugging Face cache.

Both tests use made-up inputs and need no data, Docker or GPU. They print
`all checks passed`.

## Limitations

- The timings were measured on a desktop machine that was also running other programs. The
  CPU runs are the most sensitive to this.
- TEI reports inference time in whole milliseconds.
- The report also records, per document, how closely the GPU (float16) and CPU (float32)
  embeddings agree.
