# Installs the CRAN packages used by the R stages (Phases 1 and 4) and the legacy scripts.
# Tested with R 4.5.3.
pkgs <- c(
  # pipeline
  "nanoparquet", "data.table", "dplyr", "tidyr", "tibble", "stringi", "stringr",
  "Matrix", "ggplot2", "scales", "digest", "jsonlite", "yaml", "lubridate", "readr",
  "purrr", "patchwork", "processx", "ps",
  # legacy topic-model approach
  "quanteda", "stm", "text2vec", "tidytext", "stopwords", "lexicon", "qdapDictionaries",
  "lsa", "glmnet", "clue", "ggupset", "openxlsx", "readxl", "curl", "xml2", "zip", "lgr"
)
missing <- setdiff(pkgs, rownames(installed.packages()))
if (length(missing)) install.packages(missing, repos = "https://cloud.r-project.org")
