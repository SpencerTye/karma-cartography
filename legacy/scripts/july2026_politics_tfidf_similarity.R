# TF-IDF cosine similarity between each r/politics comment and its own submission (July 2026 sample).
# Input: comment and submission Parquet files under data_sample/.
# Tokens are whitespace-delimited; the submission is represented as title only and as title + selftext.
# Output: tables, figures and manifest sidecars under r_analysis_outputs/tfidf_similarity/.

# ---- Setup: packages, paths, output directories ----
library(nanoparquet)
library(dplyr)
library(tidyr)
library(stringi)
library(Matrix)
library(ggplot2)
library(scales)
library(readr)
library(digest)
library(yaml)

project_dir     <- Sys.getenv("KC_REPO_ROOT", unset = getwd())
comments_dir    <- file.path(project_dir, "data_sample/comments/2026-07")
submissions_dir <- file.path(project_dir, "data_sample/submissions/2026-07")
script_path     <- file.path(project_dir, "july2026_politics_tfidf_similarity.R")

out_dir <- file.path(project_dir, "r_analysis_outputs/tfidf_similarity")
fig_dir <- file.path(out_dir, "figures")
tab_dir <- file.path(out_dir, "tables")
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(tab_dir, recursive = TRUE, showWarnings = FALSE)

comment_files    <- list.files(comments_dir,    pattern = "\\.parquet$", full.names = TRUE)
submission_files <- list.files(submissions_dir, pattern = "\\.parquet$", full.names = TRUE)

# ---- Read source data ----
# 7 comment files and 1 submission file; keep only the columns used below.
comments <- lapply(comment_files, read_parquet) |>
  bind_rows() |>
  select(id, link_id, parent_id, author, created_utc, subreddit, body) |>
  mutate(created_utc = as.numeric(created_utc))

submissions <- lapply(submission_files, read_parquet) |>
  bind_rows() |>
  select(id, author, created_utc, subreddit, title, selftext) |>
  mutate(created_utc = as.numeric(created_utc))

n_comments_all    <- nrow(comments)
n_submissions_all <- nrow(submissions)

# ---- Text-field availability ----
# Count each text field's values as missing, empty, a deletion/removal marker, or text.
text_state <- function(x) case_when(
  is.na(x)                     ~ "missing (NA)",
  x == ""                      ~ "empty string",
  x == "[deleted]"             ~ "[deleted]",
  x == "[removed]"             ~ "[removed]",
  x == "[ Removed by Reddit ]" ~ "[ Removed by Reddit ]",
  TRUE                         ~ "other text")

text_availability <- bind_rows(
  tibble(field = "comment body",        state = text_state(comments$body)),
  tibble(field = "comment author",      state = text_state(comments$author)),
  tibble(field = "submission title",    state = text_state(submissions$title)),
  tibble(field = "submission selftext", state = text_state(submissions$selftext)),
  tibble(field = "submission author",   state = text_state(submissions$author))) |>
  count(field, state, name = "n") |>
  group_by(field) |>
  mutate(rows_in_field = sum(n), prop = n / rows_in_field) |>
  ungroup()

n_text_missing <- sum(is.na(comments$body)) + sum(is.na(submissions$title)) + sum(is.na(submissions$selftext))

# ---- Verify the comment-to-submission relationship ----
# submission_id = link_id without 't3_'. A comment is eligible when its submission is
# in the sample; a submission is modeled when an eligible comment points to it.
n_comment_id_distinct    <- n_distinct(comments$id)
n_comment_id_missing     <- sum(is.na(comments$id))
n_submission_id_distinct <- n_distinct(submissions$id)
n_submission_id_missing  <- sum(is.na(submissions$id))
n_submission_id_base36   <- sum(grepl("^[0-9a-z]+$", submissions$id))
n_link_id_wellformed     <- sum(grepl("^t3_[0-9a-z]+$", comments$link_id))

comments <- comments |>
  mutate(submission_id = substr(link_id, 4, nchar(link_id)),
         matched       = submission_id %in% submissions$id)

n_eligible           <- sum(comments$matched)
n_unmatched          <- sum(!comments$matched)
n_threads_referenced <- n_distinct(comments$submission_id)
n_threads_unmatched  <- n_distinct(comments$submission_id[!comments$matched])

submissions <- submissions |>
  mutate(modeled = id %in% comments$submission_id[comments$matched])
n_modeled     <- sum(submissions$modeled)
n_not_modeled <- sum(!submissions$modeled)

pair_lookup <- match(comments$submission_id[comments$matched], submissions$id)
n_pairs_same_subreddit          <- sum(comments$subreddit[comments$matched] == submissions$subreddit[pair_lookup])
n_pairs_comment_before_submission <- sum(comments$created_utc[comments$matched] < submissions$created_utc[pair_lookup])

# ---- Eligible population, modeled submissions, text construction ----
# Comment text is the stored body. Submission text is the title, or title + " " + selftext.
eligible <- comments |>
  filter(matched) |>
  select(comment_id = id, submission_id, link_id, parent_id,
         comment_author = author, comment_created_utc = created_utc, body)

modeled <- submissions |>
  filter(modeled) |>
  select(submission_id = id, submission_author = author,
         submission_created_utc = created_utc, title, selftext) |>
  mutate(title_text = title,
         full_text  = paste(title, selftext, sep = " "))

eligible <- eligible |>
  mutate(submission_row = match(submission_id, modeled$submission_id))

# ---- Tokenization (the only text transformation) ----
# Split on runs of Unicode whitespace (ICU \s+). Case, punctuation, URLs and markup
# are kept, so a term is an exact whitespace-delimited string.
tokens_comment <- stri_split_regex(eligible$body,       "\\s+", omit_empty = TRUE)
tokens_title   <- stri_split_regex(modeled$title_text,  "\\s+", omit_empty = TRUE)
tokens_full    <- stri_split_regex(modeled$full_text,   "\\s+", omit_empty = TRUE)

eligible <- eligible |>
  mutate(comment_n_chars  = stri_length(body),
         comment_n_tokens = lengths(tokens_comment))
modeled <- modeled |>
  mutate(title_n_tokens = lengths(tokens_title),
         full_n_tokens  = lengths(tokens_full))

# ---- Vocabulary and document collection ----
# Collection = eligible comments plus modeled submissions (title + selftext).
# The title-only form uses the same vocabulary.
n_docs_comment    <- length(tokens_comment)
n_docs_submission <- length(tokens_full)
n_docs            <- n_docs_comment + n_docs_submission

collection_tokens <- c(unlist(tokens_comment, use.names = FALSE),
                       unlist(tokens_full,    use.names = FALSE))
collection_doc    <- c(rep(seq_len(n_docs_comment), lengths(tokens_comment)),
                       rep(n_docs_comment + seq_len(n_docs_submission), lengths(tokens_full)))
vocabulary        <- unique(collection_tokens)
n_vocabulary      <- length(vocabulary)

# ---- Term-frequency matrices ----
# Sparse raw counts, documents x terms (sparseMatrix sums repeated pairs).
tf_collection <- sparseMatrix(i = collection_doc,
                              j = match(collection_tokens, vocabulary),
                              x = 1, dims = c(n_docs, n_vocabulary), repr = "C")
tf_comment <- tf_collection[seq_len(n_docs_comment), , drop = FALSE]
tf_full    <- tf_collection[n_docs_comment + seq_len(n_docs_submission), , drop = FALSE]

title_tokens <- unlist(tokens_title, use.names = FALSE)
tf_title <- sparseMatrix(i = rep(seq_len(n_docs_submission), lengths(tokens_title)),
                         j = match(title_tokens, vocabulary),
                         x = 1, dims = c(n_docs_submission, n_vocabulary), repr = "C")

# ---- Document frequency and IDF ----
# idf = ln(n_docs / df), no smoothing, no terms dropped.
term_df  <- colSums(tf_collection > 0)
term_idf <- log(n_docs / term_df)

df_in_comments    <- colSums(tf_comment > 0)
df_in_submissions <- colSums(tf_full > 0)
n_terms_comments_only    <- sum(df_in_comments > 0 & df_in_submissions == 0)
n_terms_submissions_only <- sum(df_in_comments == 0 & df_in_submissions > 0)
n_terms_both             <- sum(df_in_comments > 0 & df_in_submissions > 0)
n_terms_df_one           <- sum(term_df == 1)

vocabulary_top_df <- tibble(term = vocabulary, df = term_df, idf = term_idf,
                            df_in_comments = df_in_comments, df_in_submissions = df_in_submissions,
                            token_occurrences = colSums(tf_collection)) |>
  arrange(desc(df), term) |>
  slice_head(n = 50)

# ---- Tokenization report ----
# The "distinct terms if ..." rows show what the vocabulary would shrink to under
# normalisations that were not applied.
vocab_lower      <- stri_trans_tolower(vocabulary)
vocab_edge_strip <- stri_replace_all_regex(vocabulary, "^[\\p{P}\\p{S}]+|[\\p{P}\\p{S}]+$", "")

tokenization_report <- tribble(
  ~item,                                                         ~value,
  "rule: split on runs of Unicode whitespace (ICU \\s+); drop empty pieces; nothing else",  NA_real_,
  "eligible comments tokenized",                                  n_docs_comment,
  "modeled submissions tokenized (title form)",                   n_docs_submission,
  "modeled submissions tokenized (title + selftext form)",        n_docs_submission,
  "tokens in eligible comments",                                  sum(lengths(tokens_comment)),
  "tokens in submission titles",                                  sum(lengths(tokens_title)),
  "tokens in submission title + selftext",                        sum(lengths(tokens_full)),
  "comments with zero tokens (zero vector)",                      sum(lengths(tokens_comment) == 0),
  "submission titles with zero tokens (zero vector)",             sum(lengths(tokens_title) == 0),
  "submission title + selftext with zero tokens (zero vector)",   sum(lengths(tokens_full) == 0),
  "documents in the IDF collection",                              n_docs,
  "vocabulary size (distinct exact terms in the collection)",     n_vocabulary,
  "vocabulary terms occurring in exactly one document",           n_terms_df_one,
  "vocabulary terms in comments only",                            n_terms_comments_only,
  "vocabulary terms in submissions only",                         n_terms_submissions_only,
  "vocabulary terms in both comments and submissions",            n_terms_both,
  "vocabulary terms beginning with http:// or https://",          sum(stri_detect_regex(vocabulary, "^https?://")),
  "vocabulary terms with leading or trailing punctuation/symbol", sum(stri_detect_regex(vocabulary, "^[\\p{P}\\p{S}]|[\\p{P}\\p{S}]$")),
  "vocabulary terms containing an upper-case letter",             sum(stri_detect_regex(vocabulary, "\\p{Lu}")),
  "distinct terms if case-folded (not applied)",                  n_distinct(vocab_lower),
  "distinct terms if edge punctuation stripped (not applied)",    n_distinct(vocab_edge_strip),
  "distinct terms if both (not applied)",                         n_distinct(stri_trans_tolower(vocab_edge_strip))
)

# ---- TF-IDF weighting ----
# weight = count x idf, same idf for all three matrices; Euclidean norms.
idf_diagonal <- Diagonal(x = term_idf)
w_comment <- tf_comment %*% idf_diagonal
w_title   <- tf_title   %*% idf_diagonal
w_full    <- tf_full    %*% idf_diagonal
norm_comment <- sqrt(rowSums(w_comment^2))
norm_title   <- sqrt(rowSums(w_title^2))
norm_full    <- sqrt(rowSums(w_full^2))

# ---- Cosine similarity, comment to its own submission ----
# cosine = dot / (comment norm x submission norm). A zero norm gives NA, with the
# reason in a status column.
row_s <- eligible$submission_row
dot_title <- rowSums(w_comment * w_title[row_s, , drop = FALSE])
dot_full  <- rowSums(w_comment * w_full[row_s, , drop = FALSE])

eligible <- eligible |>
  mutate(
    comment_norm          = norm_comment,
    title_norm            = norm_title[row_s],
    full_norm             = norm_full[row_s],
    cosine_title          = if_else(comment_norm > 0 & title_norm > 0,
                                    dot_title / (comment_norm * title_norm), NA_real_),
    cosine_title_selftext = if_else(comment_norm > 0 & full_norm > 0,
                                    dot_full / (comment_norm * full_norm), NA_real_),
    status_title = case_when(
      comment_norm == 0 & title_norm == 0 ~ "undefined: comment and submission have no tokens",
      comment_norm == 0                   ~ "undefined: comment has no tokens",
      title_norm == 0                     ~ "undefined: submission has no tokens",
      TRUE                                ~ "valid"),
    status_title_selftext = case_when(
      comment_norm == 0 & full_norm == 0  ~ "undefined: comment and submission have no tokens",
      comment_norm == 0                   ~ "undefined: comment has no tokens",
      full_norm == 0                      ~ "undefined: submission has no tokens",
      TRUE                                ~ "valid"))

n_zero_vector_comments          <- sum(norm_comment == 0)
n_zero_vector_submissions_title <- sum(norm_title == 0)
n_zero_vector_submissions_full  <- sum(norm_full == 0)
n_cosine_above_one_title        <- sum(eligible$cosine_title > 1 + 1e-12, na.rm = TRUE)
n_cosine_above_one_full         <- sum(eligible$cosine_title_selftext > 1 + 1e-12, na.rm = TRUE)

# ---- Derived similarity table ----
# One row per eligible comment, ordered by creation time then id.
similarity_table <- eligible |>
  left_join(modeled |> select(submission_id, submission_author, submission_created_utc,
                              title_n_tokens, full_n_tokens),
            by = "submission_id") |>
  transmute(comment_id, submission_id, link_id, parent_id,
            comment_author, submission_author,
            comment_created_utc, submission_created_utc,
            comment_n_chars, comment_n_tokens,
            submission_title_n_tokens          = title_n_tokens,
            submission_title_selftext_n_tokens = full_n_tokens,
            cosine_title, cosine_title_selftext,
            status_title, status_title_selftext) |>
  arrange(comment_created_utc, comment_id)

representation_labels <- c(title          = "Submission title",
                           title_selftext = "Submission title + selftext")

similarity_long <- similarity_table |>
  select(comment_id, submission_id, comment_n_tokens, cosine_title, cosine_title_selftext) |>
  pivot_longer(c(cosine_title, cosine_title_selftext), names_to = "representation",
               names_prefix = "cosine_", values_to = "cosine") |>
  filter(!is.na(cosine))

# ---- Summary measurements ----
# Population counts, valid/undefined counts, exact zeros, quantiles (type 7), mean,
# min and max per representation.
similarity_stats <- function(cosine, representation) {
  valid <- cosine[!is.na(cosine)]
  tibble(representation = representation,
         measure = c("valid_cosine_measurements", "undefined_cosine_cases",
                     "exactly_zero_cosine_count", "exactly_zero_cosine_prop",
                     "min", "p01", "p05", "p10", "p25", "median", "mean",
                     "p75", "p90", "p95", "p99", "max"),
         value = c(length(valid), sum(is.na(cosine)), sum(valid == 0), mean(valid == 0),
                   min(valid), quantile(valid, c(0.01, 0.05, 0.10, 0.25, 0.50), type = 7, names = FALSE),
                   mean(valid), quantile(valid, c(0.75, 0.90, 0.95, 0.99), type = 7, names = FALSE),
                   max(valid)))
}

similarity_summary <- bind_rows(
  tribble(
    ~measure,                                            ~value,
    "comments_in_sample",                                n_comments_all,
    "eligible_matched_comments",                         n_eligible,
    "unmatched_comments",                                n_unmatched,
    "threads_referenced_by_comments",                    n_threads_referenced,
    "threads_without_a_submission_in_sample",            n_threads_unmatched,
    "submissions_in_sample",                             n_submissions_all,
    "modeled_submissions",                               n_modeled,
    "submissions_without_eligible_comment",              n_not_modeled,
    "documents_represented",                             n_docs,
    "documents_comments",                                n_docs_comment,
    "documents_submissions",                             n_docs_submission,
    "vocabulary_size",                                   n_vocabulary,
    "zero_vector_comments",                              n_zero_vector_comments,
    "zero_vector_submissions_title",                     n_zero_vector_submissions_title,
    "zero_vector_submissions_title_selftext",            n_zero_vector_submissions_full,
    "pairs_comment_created_before_submission",           n_pairs_comment_before_submission) |>
    mutate(representation = "collection"),
  similarity_stats(similarity_table$cosine_title,          "title"),
  similarity_stats(similarity_table$cosine_title_selftext, "title_selftext")) |>
  select(representation, measure, value)

# ---- Representative pairs ----
# For p05, p25, p50, p75 and p95 of the non-zero cosines, the 3 comments nearest
# each target (ties by id), plus 3 exact-zero comments with the lowest ids.
position_probs     <- c("very low" = 0.05, "lower-middle" = 0.25, "middle" = 0.50,
                        "upper-middle" = 0.75, "very high" = 0.95)
position_levels    <- c(names(position_probs), "zero (no shared term)")
pairs_per_position <- 3

position_targets <- similarity_long |>
  filter(cosine > 0) |>
  group_by(representation) |>
  reframe(position = names(position_probs),
          target_cosine = quantile(cosine, position_probs, type = 7, names = FALSE))

selected_positions <- similarity_long |>
  inner_join(position_targets, by = "representation", relationship = "many-to-many") |>
  mutate(distance_to_target = abs(cosine - target_cosine)) |>
  group_by(representation, position) |>
  arrange(distance_to_target, comment_id, .by_group = TRUE) |>
  slice_head(n = pairs_per_position) |>
  ungroup()

selected_zero <- similarity_long |>
  filter(cosine == 0) |>
  group_by(representation) |>
  arrange(comment_id, .by_group = TRUE) |>
  slice_head(n = pairs_per_position) |>
  ungroup() |>
  mutate(position = "zero (no shared term)", target_cosine = 0, distance_to_target = 0)

representative_pairs <- bind_rows(selected_positions, selected_zero) |>
  left_join(similarity_table |> select(comment_id, link_id, comment_author, submission_author,
                                       comment_created_utc, submission_created_utc),
            by = "comment_id") |>
  left_join(eligible |> select(comment_id, comment_text = body), by = "comment_id") |>
  left_join(modeled  |> select(submission_id, title_text, full_text), by = "submission_id") |>
  mutate(submission_text_used = if_else(representation == "title", title_text, full_text),
         position = factor(position, levels = position_levels)) |>
  arrange(representation, position, distance_to_target, comment_id) |>
  select(representation, position, target_cosine, cosine, comment_id, submission_id,
         thread_id = link_id, submission_text_used, comment_text,
         submission_author, comment_author, comment_created_utc, submission_created_utc,
         comment_n_tokens)

# ---- Shared weighted terms for the representative pairs ----
# Per-term contribution to each pair's dot product; the 10 largest per pair are kept.
shared_weighted_terms <- lapply(seq_len(nrow(representative_pairs)), function(k) {
  i <- match(representative_pairs$comment_id[k],    eligible$comment_id)
  s <- match(representative_pairs$submission_id[k], modeled$submission_id)
  count_c <- as.numeric(tf_comment[i, ])
  count_s <- if (representative_pairs$representation[k] == "title") as.numeric(tf_title[s, ]) else as.numeric(tf_full[s, ])
  shared  <- which(count_c > 0 & count_s > 0)
  tibble(representation  = representative_pairs$representation[k],
         position        = representative_pairs$position[k],
         comment_id      = representative_pairs$comment_id[k],
         submission_id   = representative_pairs$submission_id[k],
         cosine          = representative_pairs$cosine[k],
         comment_norm    = norm_comment[i],
         submission_norm = if (representative_pairs$representation[k] == "title") norm_title[s] else norm_full[s],
         n_shared_terms  = length(shared),
         term            = vocabulary[shared],
         count_comment   = count_c[shared],
         count_submission = count_s[shared],
         df              = term_df[shared],
         idf             = term_idf[shared],
         contribution    = count_c[shared] * term_idf[shared] * count_s[shared] * term_idf[shared])
}) |>
  bind_rows() |>
  group_by(representation, position, comment_id) |>
  mutate(dot_product = sum(contribution), share_of_dot_product = contribution / dot_product) |>
  arrange(desc(contribution), term, .by_group = TRUE) |>
  slice_head(n = 10) |>
  ungroup() |>
  arrange(representation, position, comment_id, desc(contribution), term)

# ---- Similarity by comment length ----
length_breaks <- c(0, 1, 2, 5, 10, 20, 50, 100, 200, Inf)
length_labels <- c("1", "2", "3-5", "6-10", "11-20", "21-50", "51-100", "101-200", "201+")

similarity_by_comment_length_bands <- similarity_long |>
  mutate(comment_length_band = cut(comment_n_tokens, breaks = length_breaks,
                                   labels = length_labels, right = TRUE)) |>
  group_by(representation, comment_length_band) |>
  summarise(comments        = n(),
            prop_exactly_zero = mean(cosine == 0),
            p25             = quantile(cosine, 0.25, type = 7, names = FALSE),
            median          = quantile(cosine, 0.50, type = 7, names = FALSE),
            mean            = mean(cosine),
            p75             = quantile(cosine, 0.75, type = 7, names = FALSE),
            p95             = quantile(cosine, 0.95, type = 7, names = FALSE),
            max             = max(cosine),
            .groups = "drop")

# ---- Figure data: bins and grids ----
# Exact 0 is its own bin, then 50 right-closed bins of 0.02 on (0, 1]; values just
# above 1 from floating-point noise go in the top bin. Length edges are integer-aligned.
cosine_bin <- function(x) if_else(x == 0, 0L, pmin(as.integer(ceiling(x / 0.02)), 50L))

histogram_bins <- similarity_long |>
  mutate(bin = cosine_bin(cosine)) |>
  count(representation, bin, name = "n") |>
  complete(representation, bin = 0:50, fill = list(n = 0)) |>
  mutate(bin_label   = if_else(bin == 0L, "exactly 0", sprintf("(%.2f, %.2f]", (bin - 1) * 0.02, bin * 0.02)),
         cosine_lower = if_else(bin == 0L, 0, (bin - 1) * 0.02),
         cosine_upper = bin * 0.02,
         x_left       = if_else(bin == 0L, -0.02, cosine_lower),
         x_right      = cosine_upper) |>
  group_by(representation) |>
  mutate(prop = n / sum(n)) |>
  ungroup() |>
  arrange(representation, bin)

ecdf_grid <- similarity_long |>
  group_by(representation) |>
  reframe(cosine_threshold = seq(0, 1, by = 0.001),
          cumulative_share = ecdf(cosine)(cosine_threshold))

comparison_bins <- similarity_table |>
  filter(!is.na(cosine_title), !is.na(cosine_title_selftext)) |>
  mutate(x_bin = cosine_bin(cosine_title), y_bin = cosine_bin(cosine_title_selftext)) |>
  count(x_bin, y_bin, name = "n") |>
  mutate(title_lower = if_else(x_bin == 0L, 0, (x_bin - 1) * 0.02), title_upper = x_bin * 0.02,
         full_lower  = if_else(y_bin == 0L, 0, (y_bin - 1) * 0.02), full_upper  = y_bin * 0.02,
         x_left = if_else(x_bin == 0L, -0.02, title_lower), x_right = title_upper,
         y_low  = if_else(y_bin == 0L, -0.02, full_lower),  y_high  = full_upper)
n_valid_both <- sum(!is.na(similarity_table$cosine_title) & !is.na(similarity_table$cosine_title_selftext))
n_identical_both <- sum(abs(similarity_table$cosine_title - similarity_table$cosine_title_selftext) < 1e-12, na.rm = TRUE)

token_edges <- unique(round(10^seq(0, log10(max(similarity_table$comment_n_tokens)) + 0.1, by = 0.1))) - 0.5
length_bins <- similarity_long |>
  mutate(x_bin = as.integer(cut(comment_n_tokens, breaks = token_edges, right = FALSE)),
         y_bin = cosine_bin(cosine)) |>
  count(representation, x_bin, y_bin, name = "n") |>
  mutate(tokens_first = token_edges[x_bin] + 0.5, tokens_last = token_edges[x_bin + 1] - 0.5,
         x_left = token_edges[x_bin], x_right = token_edges[x_bin + 1],
         cosine_lower = if_else(y_bin == 0L, 0, (y_bin - 1) * 0.02), cosine_upper = y_bin * 0.02,
         y_low = if_else(y_bin == 0L, -0.02, cosine_lower), y_high = cosine_upper) |>
  arrange(representation, x_bin, y_bin)

# ---- Figures ----
# Four PNGs; log10 count scales are noted in the axis or legend title.
ink_series   <- "#2a78d6"
ink_series_2 <- "#eb6834"
ink_text     <- "#0b0b0b"
ink_sub      <- "#52514e"
ink_muted    <- "#898781"
grid_col     <- "#e1e0d9"
axis_col     <- "#c3c2b7"
ref_col      <- "grey55"
blue_ramp    <- c("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")

theme_fig <- theme_minimal(base_size = 12) +
  theme(panel.grid.minor = element_blank(),
        panel.grid.major = element_line(colour = grid_col, linewidth = 0.3),
        axis.line.x      = element_line(colour = axis_col, linewidth = 0.4),
        axis.text        = element_text(colour = ink_muted),
        axis.title       = element_text(colour = ink_sub),
        plot.title       = element_text(colour = ink_text, face = "bold"),
        strip.text       = element_text(colour = ink_sub, face = "bold", hjust = 0),
        legend.title     = element_text(colour = ink_sub),
        legend.text      = element_text(colour = ink_muted),
        plot.background  = element_rect(fill = "white", colour = NA))

# Figure 1: distribution of cosine similarity, one panel per representation.
fig1 <- histogram_bins |>
  filter(n > 0) |>
  mutate(representation = representation_labels[representation]) |>
  ggplot() +
  geom_rect(aes(xmin = x_left, xmax = x_right, ymin = 1, ymax = n), fill = ink_series) +
  facet_wrap(~ representation, ncol = 1) +
  scale_x_continuous(breaks = seq(0, 1, by = 0.1), limits = c(-0.02, 1), expand = expansion(mult = 0.01)) +
  scale_y_log10(breaks = 10^(0:6), labels = label_comma(), expand = expansion(mult = c(0, 0.08))) +
  labs(title = "Comment-to-submission lexical similarity — r/politics, July 2026",
       x = "TF-IDF cosine similarity, comment vs its submission (bin left of 0 = exactly 0)",
       y = "Comments, log10 scale") +
  theme_fig
ggsave(file.path(fig_dir, "fig1_similarity_distribution.png"), fig1,
       width = 10, height = 7, dpi = 150, bg = "white")

# Figure 2: empirical cumulative distribution, both representations.
fig2 <- ecdf_grid |>
  mutate(representation = representation_labels[representation]) |>
  ggplot(aes(cosine_threshold, cumulative_share, colour = representation, linetype = representation)) +
  geom_step(linewidth = 0.9) +
  scale_colour_manual(values = unname(c(ink_series, ink_series_2)), breaks = unname(representation_labels),
                      name = "Submission text") +
  scale_linetype_manual(values = c("solid", "dashed"), breaks = unname(representation_labels),
                        name = "Submission text") +
  scale_x_continuous(breaks = seq(0, 1, by = 0.1)) +
  scale_y_continuous(labels = label_percent(), limits = c(0, 1), breaks = seq(0, 1, by = 0.1)) +
  labs(title = "Cumulative lexical similarity — r/politics comments, July 2026",
       x = "TF-IDF cosine similarity, comment vs its submission",
       y = "Share of eligible comments at or below") +
  theme_fig +
  theme(legend.position = "bottom")
ggsave(file.path(fig_dir, "fig2_similarity_ecdf.png"), fig2,
       width = 9, height = 6, dpi = 150, bg = "white")

# Figure 3: title-only vs title + selftext similarity, 2-D bin counts.
fig3 <- ggplot(comparison_bins) +
  geom_rect(aes(xmin = x_left, xmax = x_right, ymin = y_low, ymax = y_high, fill = n)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = ref_col) +
  scale_fill_gradientn(colours = blue_ramp, transform = "log10", labels = label_comma(),
                       name = "Comments, log10 scale") +
  scale_x_continuous(breaks = seq(0, 1, by = 0.2), limits = c(-0.02, 1), expand = expansion(mult = 0.01)) +
  scale_y_continuous(breaks = seq(0, 1, by = 0.2), limits = c(-0.02, 1), expand = expansion(mult = 0.01)) +
  coord_equal() +
  labs(title = "Submission-text representation comparison — r/politics, July 2026",
       x = "Cosine similarity to submission title (bin left of 0 = exactly 0)",
       y = "Cosine similarity to submission title + selftext") +
  theme_fig
ggsave(file.path(fig_dir, "fig3_representation_comparison.png"), fig3,
       width = 8, height = 7.5, dpi = 150, bg = "white")

# Figure 4: comment length vs similarity, 2-D bin counts, one panel per representation.
fig4 <- length_bins |>
  mutate(representation = representation_labels[representation]) |>
  ggplot() +
  geom_rect(aes(xmin = x_left, xmax = x_right, ymin = y_low, ymax = y_high, fill = n)) +
  facet_wrap(~ representation, ncol = 2) +
  scale_fill_gradientn(colours = blue_ramp, transform = "log10", labels = label_comma(),
                       name = "Comments, log10 scale") +
  scale_x_log10(breaks = c(1, 10, 100, 1000), labels = label_comma(), expand = expansion(mult = 0.01)) +
  scale_y_continuous(breaks = seq(0, 1, by = 0.2), limits = c(-0.02, 1), expand = expansion(mult = 0.01)) +
  labs(title = "Lexical similarity and comment length — r/politics, July 2026",
       x = "Comment length, whitespace tokens (log10 scale)",
       y = "TF-IDF cosine similarity to submission (bin below 0 = exactly 0)") +
  theme_fig +
  theme(legend.position = "bottom")
ggsave(file.path(fig_dir, "fig4_similarity_vs_comment_length.png"), fig4,
       width = 11, height = 6.5, dpi = 150, bg = "white")

# ---- Validation checks ----
# Recompute key quantities a second way on fixed samples, reconcile counts, and
# stop on any failure.
sample_docs  <- seq(1, n_eligible, by = 10000)   # every 10,000th eligible comment (source order)
sample_pairs <- seq(1, n_eligible, by = 1000)    # every 1,000th eligible comment (source order)

tf_row_matches <- vapply(sample_docs, function(i) {
  counts <- table(tokens_comment[[i]])
  row    <- as.numeric(tf_comment[i, ])
  sum(row > 0) == length(counts) && all(row[match(names(counts), vocabulary)] == as.numeric(counts))
}, logical(1))

tfidf_row_matches <- vapply(sample_docs, function(i) {
  isTRUE(all.equal(as.numeric(w_comment[i, ]), as.numeric(tf_comment[i, ]) * term_idf))
}, logical(1))

cosine_from_tokens <- function(tokens_a, tokens_b) {
  count_a <- table(tokens_a); count_b <- table(tokens_b)
  idf_a   <- term_idf[match(names(count_a), vocabulary)]
  idf_b   <- term_idf[match(names(count_b), vocabulary)]
  shared  <- intersect(names(count_a), names(count_b))
  dot     <- sum(as.numeric(count_a[shared]) * as.numeric(count_b[shared]) * term_idf[match(shared, vocabulary)]^2)
  dot / (sqrt(sum((as.numeric(count_a) * idf_a)^2)) * sqrt(sum((as.numeric(count_b) * idf_b)^2)))
}
recomputed <- lapply(sample_pairs, function(i) {
  s <- match(eligible$submission_id[i], modeled$submission_id)   # lookup by id, not by stored row
  tibble(comment_id = eligible$comment_id[i],
         cosine_title_recomputed = cosine_from_tokens(tokens_comment[[i]], tokens_title[[s]]),
         cosine_full_recomputed  = cosine_from_tokens(tokens_comment[[i]], tokens_full[[s]]))
}) |>
  bind_rows() |>
  inner_join(similarity_table |> select(comment_id, cosine_title, cosine_title_selftext), by = "comment_id")
recompute_ok <- with(recomputed,
  (is.na(cosine_title) == is.na(cosine_title_recomputed) | is.nan(cosine_title_recomputed)) &
  (is.na(cosine_title_selftext) == is.na(cosine_full_recomputed) | is.nan(cosine_full_recomputed)) &
  (is.na(cosine_title) | abs(cosine_title - cosine_title_recomputed) < 1e-9) &
  (is.na(cosine_title_selftext) | abs(cosine_title_selftext - cosine_full_recomputed) < 1e-9))

source_pairs <- comments |>
  filter(matched) |>
  select(comment_id = id, source_submission_id = submission_id, source_link_id = link_id,
         source_parent_id = parent_id, source_author = author, source_created = created_utc)
table_vs_source <- similarity_table |>
  inner_join(source_pairs, by = "comment_id") |>
  inner_join(submissions |> select(source_submission_id = id, sub_author = author, sub_created = created_utc),
             by = "source_submission_id")
n_table_source_agree <- with(table_vs_source,
  sum(submission_id == source_submission_id & link_id == source_link_id & parent_id == source_parent_id &
      comment_author == source_author & comment_created_utc == source_created &
      submission_author == sub_author & submission_created_utc == sub_created))

pairs_vs_table <- representative_pairs |>
  inner_join(similarity_table |> select(comment_id, cosine_title, cosine_title_selftext), by = "comment_id") |>
  inner_join(eligible |> select(comment_id, body), by = "comment_id") |>
  inner_join(modeled  |> select(submission_id, title, selftext), by = "submission_id") |>
  mutate(cosine_in_table = if_else(representation == "title", cosine_title, cosine_title_selftext),
         text_expected   = if_else(representation == "title", title, paste(title, selftext, sep = " ")))
n_pairs_consistent <- with(pairs_vs_table,
  sum(cosine == cosine_in_table & comment_text == body & submission_text_used == text_expected))

shared_totals <- shared_weighted_terms |>
  distinct(representation, position, comment_id, cosine, comment_norm, submission_norm, dot_product, n_shared_terms)
n_shared_dot_ok <- sum(abs(shared_totals$dot_product / (shared_totals$comment_norm * shared_totals$submission_norm) -
                             shared_totals$cosine) < 1e-9)
n_zero_pairs_with_terms <- sum(representative_pairs$cosine == 0 &
                                 representative_pairs$comment_id %in% shared_weighted_terms$comment_id)

validation_checks <- tribble(
  ~check, ~detail, ~pass,
  "comment ids unique and non-missing",
    sprintf("%d distinct of %d rows; %d missing", n_comment_id_distinct, n_comments_all, n_comment_id_missing),
    n_comment_id_distinct == n_comments_all && n_comment_id_missing == 0,
  "submission ids unique, non-missing, bare base36",
    sprintf("%d distinct of %d rows; %d missing; %d base36", n_submission_id_distinct, n_submissions_all,
            n_submission_id_missing, n_submission_id_base36),
    n_submission_id_distinct == n_submissions_all && n_submission_id_missing == 0 && n_submission_id_base36 == n_submissions_all,
  "every link_id matches ^t3_[0-9a-z]+$",
    sprintf("%d of %d", n_link_id_wellformed, n_comments_all),
    n_link_id_wellformed == n_comments_all,
  "no missing (NA) text in body, title, selftext",
    sprintf("%d missing values", n_text_missing),
    n_text_missing == 0,
  "matched plus unmatched comments equal comments in sample",
    sprintf("%d + %d = %d", n_eligible, n_unmatched, n_comments_all),
    n_eligible + n_unmatched == n_comments_all,
  "modeled plus not-modeled submissions equal submissions in sample",
    sprintf("%d + %d = %d", n_modeled, n_not_modeled, n_submissions_all),
    n_modeled + n_not_modeled == n_submissions_all,
  "every eligible comment resolves to exactly one modeled submission",
    sprintf("%d of %d resolve; %d distinct submissions", sum(!is.na(eligible$submission_row)), n_eligible,
            n_distinct(eligible$submission_id)),
    !anyNA(eligible$submission_row) && n_distinct(eligible$submission_id) == n_modeled,
  "paired comment and submission share the same subreddit",
    sprintf("%d of %d", n_pairs_same_subreddit, n_eligible),
    n_pairs_same_subreddit == n_eligible,
  "no comment created before its submission",
    sprintf("%d pairs with comment before submission", n_pairs_comment_before_submission),
    n_pairs_comment_before_submission == 0,
  "term counts reconcile to token totals (collection and title matrices)",
    sprintf("collection %d = %d tokens; title %d = %d tokens", sum(tf_collection), length(collection_tokens),
            sum(tf_title), length(title_tokens)),
    sum(tf_collection) == length(collection_tokens) && sum(tf_title) == length(title_tokens),
  "title term counts never exceed title + selftext term counts",
    sprintf("minimum of (title+selftext - title) counts = %g", min(tf_full - tf_title)),
    min(tf_full - tf_title) >= 0,
  "sparse term counts equal table() of tokens on sampled comments",
    sprintf("%d of %d sampled documents match", sum(tf_row_matches), length(sample_docs)),
    all(tf_row_matches),
  "df within [1, n_docs]; idf finite and non-negative; one column per term",
    sprintf("df %d..%d; idf %.4f..%.4f; %d columns", min(term_df), max(term_df), min(term_idf), max(term_idf), ncol(tf_collection)),
    min(term_df) >= 1 && max(term_df) <= n_docs && all(is.finite(term_idf)) && min(term_idf) >= 0 && ncol(tf_collection) == n_vocabulary,
  "TF-IDF weights equal count x idf on sampled comments",
    sprintf("%d of %d sampled documents match", sum(tfidf_row_matches), length(sample_docs)),
    all(tfidf_row_matches),
  "cosine recomputed from token tables (submission looked up by id) matches",
    sprintf("%d of %d sampled pairs within 1e-9, both representations", sum(recompute_ok), length(sample_pairs)),
    all(recompute_ok),
  "valid cosines lie in [0, 1] (tolerance 1e-12)",
    sprintf("%d above 1 (title), %d above 1 (title+selftext); min %.4f, %.4f",
            n_cosine_above_one_title, n_cosine_above_one_full,
            min(similarity_table$cosine_title, na.rm = TRUE), min(similarity_table$cosine_title_selftext, na.rm = TRUE)),
    n_cosine_above_one_title == 0 && n_cosine_above_one_full == 0 &&
      min(similarity_table$cosine_title, na.rm = TRUE) >= 0 && min(similarity_table$cosine_title_selftext, na.rm = TRUE) >= 0,
  "valid plus undefined equal eligible comments; undefined equals zero-norm cases",
    sprintf("title %d + %d; title+selftext %d + %d; eligible %d",
            sum(!is.na(similarity_table$cosine_title)), sum(is.na(similarity_table$cosine_title)),
            sum(!is.na(similarity_table$cosine_title_selftext)), sum(is.na(similarity_table$cosine_title_selftext)), n_eligible),
    sum(!is.na(similarity_table$cosine_title)) + sum(is.na(similarity_table$cosine_title)) == n_eligible &&
      sum(is.na(similarity_table$cosine_title)) == sum(eligible$comment_norm == 0 | eligible$title_norm == 0) &&
      sum(is.na(similarity_table$cosine_title_selftext)) == sum(eligible$comment_norm == 0 | eligible$full_norm == 0),
  "similarity table has one row per eligible comment with distinct ids",
    sprintf("%d rows; %d distinct comment ids", nrow(similarity_table), n_distinct(similarity_table$comment_id)),
    nrow(similarity_table) == n_eligible && n_distinct(similarity_table$comment_id) == n_eligible,
  "similarity table ids and metadata agree with the source files (joined by id)",
    sprintf("%d of %d rows agree on submission, link, parent, authors, timestamps", n_table_source_agree, n_eligible),
    n_table_source_agree == n_eligible,
  "representative pairs agree with the table and the source texts",
    sprintf("%d of %d pairs consistent", n_pairs_consistent, nrow(representative_pairs)),
    n_pairs_consistent == nrow(representative_pairs) && nrow(representative_pairs) == length(position_levels) * 2 * pairs_per_position,
  "shared-term contributions sum to the pair's dot product; zero pairs have no shared term",
    sprintf("%d of %d pairs reconcile; %d zero pairs with shared terms", n_shared_dot_ok, nrow(shared_totals), n_zero_pairs_with_terms),
    n_shared_dot_ok == nrow(shared_totals) && n_zero_pairs_with_terms == 0,
  "figure bins and grids contain every valid observation",
    sprintf("histogram %d; length bins %d; valid %d; comparison %d of %d valid under both; ecdf ends at %.3f",
            sum(histogram_bins$n), sum(length_bins$n), nrow(similarity_long), sum(comparison_bins$n), n_valid_both,
            min(ecdf_grid$cumulative_share[ecdf_grid$cosine_threshold == 1])),
    sum(histogram_bins$n) == nrow(similarity_long) && sum(length_bins$n) == nrow(similarity_long) &&
      sum(comparison_bins$n) == n_valid_both && all(ecdf_grid$cumulative_share[ecdf_grid$cosine_threshold == 1] == 1),
  "summary table reconciles eligible + unmatched to comments in sample",
    sprintf("%g + %g = %g",
            similarity_summary$value[similarity_summary$measure == "eligible_matched_comments"],
            similarity_summary$value[similarity_summary$measure == "unmatched_comments"],
            similarity_summary$value[similarity_summary$measure == "comments_in_sample"]),
    similarity_summary$value[similarity_summary$measure == "eligible_matched_comments"] +
      similarity_summary$value[similarity_summary$measure == "unmatched_comments"] ==
      similarity_summary$value[similarity_summary$measure == "comments_in_sample"]
)

print(as.data.frame(validation_checks[, c("check", "pass")]), right = FALSE)
stopifnot(all(validation_checks$pass))

# ---- Write tables ----
write_csv(similarity_table,                   file.path(tab_dir, "comment_submission_similarity.csv"))
write_csv(similarity_summary,                 file.path(tab_dir, "similarity_summary.csv"))
write_csv(text_availability,                  file.path(tab_dir, "text_availability.csv"))
write_csv(tokenization_report,                file.path(tab_dir, "tokenization_report.csv"))
write_csv(vocabulary_top_df,                  file.path(tab_dir, "vocabulary_top_document_frequency.csv"))
write_csv(representative_pairs,               file.path(tab_dir, "representative_pairs.csv"))
write_csv(shared_weighted_terms,              file.path(tab_dir, "shared_weighted_terms.csv"))
write_csv(similarity_by_comment_length_bands, file.path(tab_dir, "similarity_by_comment_length_bands.csv"))
write_csv(histogram_bins,                     file.path(tab_dir, "similarity_histogram_bins.csv"))
write_csv(ecdf_grid,                          file.path(tab_dir, "similarity_ecdf_grid.csv"))
write_csv(comparison_bins,                    file.path(tab_dir, "representation_comparison_bins.csv"))
write_csv(length_bins,                        file.path(tab_dir, "similarity_by_comment_length_bins.csv"))
write_csv(validation_checks,                  file.path(tab_dir, "validation_checks.csv"))

# ---- Write provenance sidecars ----
# One <output>.manifest.yml per output: input/output SHA256, script hash,
# git commit and package versions.
sha256 <- function(path) paste0("sha256:", digest(path, algo = "sha256", file = TRUE))

git_commit <- tryCatch(
  system2("git", c("-C", project_dir, "rev-parse", "--short", "HEAD"),
          stdout = TRUE, stderr = FALSE),
  error = function(e) "")
manifest_packages <- c("nanoparquet", "dplyr", "tidyr", "stringi", "Matrix",
                       "ggplot2", "scales", "readr", "digest", "yaml")
package_versions  <- lapply(manifest_packages,
                            function(p) list(name = p, version = as.character(packageVersion(p))))
input_files <- lapply(c(comment_files, submission_files),
                      function(p) list(path = p, hash = sha256(p), format = "parquet"))

write_sidecar <- function(output) {
  is_csv <- grepl("\\.csv$", output)
  manifest <- list(
    manifest_version = 1L,
    output_file = output,
    output_hash = sha256(output),
    output_format = if (is_csv) "csv" else "image/png")
  if (is_csv) {
    tbl <- read_csv(output, show_col_types = FALSE, guess_max = 1000)
    manifest$output_rows <- nrow(tbl)
    manifest$output_cols <- ncol(tbl)
  }
  manifest$input_files <- input_files
  manifest$transformation <- list(
    script = script_path,
    script_hash = sha256(script_path),
    parameters = list(
      reader = "nanoparquet",
      eligible_comment = "comment whose link_id ('t3_' + id) matches a submission id in the submissions file",
      document_collection = "eligible comments + modeled submissions in title + selftext form, one document each",
      tokenizer = "split on runs of Unicode whitespace (ICU \\s+), empty pieces dropped; no case folding, punctuation handling, stopword removal, stemming, or pruning",
      term_frequency = "raw count of the term in the document",
      inverse_document_frequency = "ln(n_docs / df), df = collection documents containing the term",
      submission_representations = "title; title + ' ' + selftext (same vocabulary and idf)",
      similarity = "cosine = dot(tfidf_comment, tfidf_submission) / (norm_comment * norm_submission); NA when either norm is 0",
      representative_pairs = "targets p05/p25/p50/p75/p95 of valid non-zero cosines per representation; 3 nearest comments per target (ties by comment id); plus 3 exactly-zero comments with lowest ids",
      figure_bins = "cosine: exactly 0, then width 0.02 right-closed on (0, 1]; comment length: unique(round(10^seq(0, ., by = 0.1))) - 0.5; ecdf grid step 0.001"),
    git_commit = git_commit)
  manifest$software <- list(
    language = "R",
    language_version = paste(R.version$major, R.version$minor, sep = "."),
    packages = package_versions,
    os = paste(Sys.info()[["sysname"]], Sys.info()[["release"]]))
  manifest$timestamp <- format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")
  manifest$notes <- "TF-IDF cosine similarity between each eligible comment and its submission, July 2026 r/politics sample; source Parquet read-only."
  write_yaml(manifest, paste0(output, ".manifest.yml"))
}

invisible(lapply(list.files(tab_dir, pattern = "\\.csv$", full.names = TRUE), write_sidecar))
invisible(lapply(list.files(fig_dir, pattern = "\\.png$", full.names = TRUE), write_sidecar))

# ---- Console summary ----
print(as.data.frame(text_availability))
print(as.data.frame(tokenization_report))
print(as.data.frame(similarity_summary))
print(as.data.frame(similarity_by_comment_length_bands))
print(as.data.frame(representative_pairs[, c("representation", "position", "cosine", "comment_id", "submission_id")]))
cat("\nDone. Figures ->", fig_dir, "\nTables ->", tab_dir, "\n")
