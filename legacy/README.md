# Legacy: the first approach (topic models)

Before the embedding pipeline in `../pipeline/`, I analysed the same July 2026 `r/politics`
sample with classical text-as-data methods in R: curated stop-word lists, TF-IDF similarity,
LDA topic models across several values of K and structural topic models (STM) with
document covariates.

This was a first pass at the structure of the data before moving to embeddings. The
scripts are kept for reference; the pipeline does not use them.

- `scripts/`: the analysis scripts, one per step (metadata inventory, stop-word
  curation and audit, TF-IDF similarity, LDA fitting and review, STM pilot and `searchK`).
- `tests/`: unit tests for the STM helper functions. Run from the repository root:
  `Rscript legacy/tests/test_stm_searchk_covariates.R`.

Paths default to the working directory; set `KC_REPO_ROOT` to run from elsewhere.
