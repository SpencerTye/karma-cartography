# karma-cartography

[![tests](https://github.com/SpencerTye/karma-cartography/actions/workflows/tests.yml/badge.svg)](https://github.com/SpencerTye/karma-cartography/actions/workflows/tests.yml)

Code for analyzing Reddit text by embedding every message and clustering the embeddings.
The current data set is one month of `r/politics` (July 2026): 902,144 posts and comments.

I wrote this as a volunteer analyst for [Digital Ground Game](https://digitalgroundgame.org),
a nonprofit that holds the full historical Reddit corpus (about 40 TB). This month is the
development set; the pipeline will be scaled to the full corpus once it is validated. The
longer-term aim is to follow how narratives and disinformation spread between communities.
The methods are adapted from the ones I use to integrate multi-omics data in my PhD work.

![2-D UMAP of all 902,144 messages](docs/images/umap_density.png)

*2-D UMAP of all 902,144 messages, colored by density (log scale).*

## Phases

1. **[Message-event table](pipeline/phase1_message_event_foundation/)** (R, done). Combines
   the raw Parquet files into one table with one row per message, a stable key, thread and
   reply links, and the source file and row of each message.
2. **[Embeddings](pipeline/phase2_semantic_embeddings/)** (Python, done). Qwen3-Embedding-0.6B
   served by Text Embeddings Inference on an RTX 3090. All 902,144 messages took 15 minutes.
3. **[Clustering](pipeline/phase3_blind_semantic_structure_discovery/)** (Python, done).
   UMAP and HDBSCAN on the GPU with RAPIDS cuML: 5,167 clusters in about two minutes.
4. **[Cluster description and QC](pipeline/phase4_semantic_structure_description_qc/)** (R,
   in progress). Keywords, representative messages and metadata for each cluster, automated
   checks, and a sample for manual review.
5. Change over time, threads, links and coordination (planned).

Numbers and methods for each phase are in [`docs/METHODS.md`](docs/METHODS.md).

Before the full embedding run, separate tests measured whether each model fit in GPU memory,
the largest batch size that stayed under a 21 GiB limit, and GPU versus CPU speed. Text
Embeddings Inference has no setting to limit its GPU memory, so for the batch-size test I
wrote [`vram_cap.c`](pipeline/phase2_semantic_embeddings/feasibility_tei_qwen3_4b_batch_tokens/scripts/vram_cap/vram_cap.c),
a small library that sits in front of the CUDA driver and refuses allocations past a set limit.

Every script hashes its inputs before and after it runs, and writes a manifest next to each
output with the input and output hashes, parameters and package versions.

## Layout

```
pipeline/       the four phases; each step has scripts/, tests/ and a README
legacy/         my first approach (LDA and STM topic models in R), kept for reference
environment/    Python requirements for each stage, and an R package installer
docs/           methods and results, data dictionary
```

Data, results and model files are not included (see `.gitignore`).

## Running it

Phases 1 and 4 need R 4.5. Phases 2 and 3 need Python 3.12 with CUDA 12; I ran them under
WSL2 (Ubuntu 24.04) on an RTX 3090. Text Embeddings Inference runs in Docker with the NVIDIA
Container Toolkit.

```bash
Rscript environment/install_r_packages.R
pip install -r environment/requirements-embeddings.txt
pip install -r environment/requirements-clustering.txt
```

Put the Reddit Parquet files under `data_sample/` (the Phase 1 README gives the layout) and
run the phases in order. Each step's README has its command, inputs, outputs and the checks
that show it worked. Scripts find the repository root themselves; set `KC_REPO_ROOT` to
override it.

## Tests

The unit tests don't need a GPU, the data or the models:

```bash
pip install -r environment/requirements-tests.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
python pipeline/phase2_semantic_embeddings/full_corpus_embedding_0p6b/tests/test_embed_full_corpus_functions.py
Rscript pipeline/phase1_message_event_foundation/tests/test_phase1_functions.R
```

GitHub runs all of them on every push.

## License

MIT. Written by Spencer Tye ([spencertye.me](https://spencertye.me)).
