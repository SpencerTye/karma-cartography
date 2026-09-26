# Legacy: topic models

My first approach to the same July 2026 `r/politics` data used text-as-data methods in R:
curated stop-word lists, TF-IDF similarity, LDA topic models for several values of K, and
structural topic models (STM) with document covariates. It was a first look at the
structure of the data before I moved to embeddings. The pipeline does not use these scripts.

- `scripts/`: one script per step (metadata inventory, stop-word curation and audit,
  TF-IDF similarity, LDA fitting and review, STM pilot and `searchK`).
- `tests/`: unit tests for the STM helper functions. Run them from the repository root, for
  example `Rscript legacy/tests/test_stm_searchk_covariates.R`.

Paths are relative to the working directory; set `KC_REPO_ROOT` to run from somewhere else.
