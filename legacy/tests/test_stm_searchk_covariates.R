# Test for build_stm_covariates() in july2026_politics_stm_searchk_k10_k20_k30.R.
# Run: Rscript tests/test_stm_searchk_covariates.R
# The function is extracted from the script by parsing it, so the script's pipeline does not run.
suppressPackageStartupMessages({ library(dplyr) })
script <- file.path(Sys.getenv("KC_REPO_ROOT", unset = getwd()), "legacy/scripts/july2026_politics_stm_searchk_k10_k20_k30.R")
stopifnot("script file exists" = file.exists(script))
exprs <- parse(script, keep.source = FALSE)
is_def <- vapply(exprs, function(e) is.call(e) && identical(e[[1]], as.name("<-")) && identical(e[[2]], as.name("build_stm_covariates")), logical(1))
stopifnot("build_stm_covariates is defined once in the script" = sum(is_def) == 1L)
eval(exprs[[which(is_def)]])

# Synthetic modelled documents: two submissions (A, B) and four comments.
#   c1: thread A, replies to the submission;  c2: thread A, replies to c1;
#   c3: thread Z whose submission is absent from the sample; c4: thread B, parent comment absent.
docs_stm <- tibble::tibble(
  doc_key      = c("t1_c1", "t1_c2", "t1_c3", "t1_c4", "t3_A", "t3_B"),
  doc_type     = c("comment", "comment", "comment", "comment", "submission", "submission"),
  author       = c("u1", "u2", "u1", "[deleted]", "u9", "u2"),
  created_utc  = c(1782864100, 1782864200, 1782864300, 1782864400, 1782864000, 1782864050),
  link_id      = c("t3_A", "t3_A", "t3_Z", "t3_B", "t3_A", "t3_B"),
  parent_id    = c("t3_A", "t1_c1", "t3_Z", "t1_x", NA, NA),
  score        = c(3, -2, 1, 0, 40, 7),
  num_comments = c(NA, NA, NA, NA, 5, 8),
  upvote_ratio = c(NA, NA, NA, NA, 0.9, 0.7))
submissions <- tibble::tibble(id = c("A", "B", "Q"), num_comments = c(5, 8, 100), upvote_ratio = c(0.9, 0.7, 0.5))

out <- build_stm_covariates(docs_stm, submissions)
meta <- out$meta
stopifnot(
  "returns meta and placeholders" = all(c("meta", "placeholders", "missingness") %in% names(out)),
  "one row per document, in order" = nrow(meta) == 6L && identical(out$doc_key, docs_stm$doc_key),
  "doc_type factor with comment as reference" = is.factor(meta$doc_type) && identical(levels(meta$doc_type), c("comment", "submission")),
  "author is a factor with [deleted] as a level" = is.factor(meta$author) && "[deleted]" %in% levels(meta$author),
  "thread: comments keep link_id, submissions get t3_ + own id, shared with their comments" =
    identical(as.character(meta$thread_id), c("t3_A", "t3_A", "t3_Z", "t3_B", "t3_A", "t3_B")),
  "thread of an absent submission is kept as its own level" = "t3_Z" %in% levels(meta$thread_id),
  "parent: comments keep parent_id (absent parents kept), submissions get NO_PARENT_SUBMISSION" =
    identical(as.character(meta$parent_id), c("t3_A", "t1_c1", "t3_Z", "t1_x", "NO_PARENT_SUBMISSION", "NO_PARENT_SUBMISSION")),
  "created_utc and score pass through unchanged" = identical(meta$created_utc, docs_stm$created_utc) && identical(meta$score, docs_stm$score),
  "num_comments: own for submissions, inherited for linked comments" = identical(meta$num_comments[c(1, 2, 4, 5, 6)], c(5, 5, 8, 5, 8)),
  "upvote_ratio: own for submissions, inherited for linked comments" = identical(meta$upvote_ratio[c(1, 2, 4, 5, 6)], c(0.9, 0.9, 0.7, 0.9, 0.7)),
  "missingness indicator marks only the comment whose submission is absent" =
    identical(meta$num_comments_missing, c(0L, 0L, 1L, 0L, 0L, 0L)) && identical(meta$upvote_ratio_missing, c(0L, 0L, 1L, 0L, 0L, 0L)),
  "placeholder is the median of the observed values over modelled documents" =
    identical(unname(out$placeholders[["num_comments"]]), 5) && identical(unname(out$placeholders[["upvote_ratio"]]), 0.9),
  "placeholder fills the missing row" = meta$num_comments[3] == 5 && meta$upvote_ratio[3] == 0.9,
  "no NA anywhere in the covariate table" = !anyNA(meta),
  "missingness table counts the absent-submission comments" =
    out$missingness$n[out$missingness$field == "num_comments" & out$missingness$reason == "comment: linked submission absent from the sample"] == 1L
)
cat("test_stm_searchk_covariates: all checks passed\n")
