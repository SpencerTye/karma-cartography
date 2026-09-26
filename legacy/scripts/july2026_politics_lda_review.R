# July 2026 r/politics sample — Phase 1A: read-only review of the retained LDA models (K = 10, K = 60)
#
# Reads the tables delivered by july2026_politics_lda_topics.R under
# r_analysis_outputs/lda_topics/ and the source Parquet text (read-only, needed
# only to display documents and to count a few whole-word occurrences). Fits no
# model, changes no preprocessing, alters no delivered file. Writes review-only
# tables and figures under r_analysis_outputs/lda_topics_review/. Topics keep
# their delivered numbers; nothing here names or interprets a topic.

# ---- Setup ----
suppressPackageStartupMessages({
  library(nanoparquet); library(data.table); library(dplyr); library(tidyr); library(stringi)
  library(ggplot2); library(scales); library(readr); library(digest); library(yaml)
})
run_start <- Sys.time()
log_line  <- function(...) cat(format(Sys.time(), "%H:%M:%S"), ..., "\n")

project_dir     <- Sys.getenv("KC_REPO_ROOT", unset = getwd())
lda_dir         <- file.path(project_dir, "r_analysis_outputs/lda_topics")
in_tab          <- file.path(lda_dir, "tables")
in_fig          <- file.path(lda_dir, "figures")
out_dir         <- file.path(project_dir, "r_analysis_outputs/lda_topics_review")
fig_dir         <- file.path(out_dir, "figures")
tab_dir         <- file.path(out_dir, "tables")
script_path     <- file.path(project_dir, "july2026_politics_lda_review.R")
lda_script_path <- file.path(project_dir, "july2026_politics_lda_topics.R")
comments_dir    <- file.path(project_dir, "data_sample/comments/2026-07")
submissions_dir <- file.path(project_dir, "data_sample/submissions/2026-07")
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(tab_dir, recursive = TRUE, showWarnings = FALSE)
invisible(file.remove(list.files(c(fig_dir, tab_dir), pattern = "\\.(csv|png|yml)$", full.names = TRUE)))

# Review parameters (every value is recorded in the manifests).
retained_k          <- c(10L, 60L)
review_seed         <- 20260906L   # seeds the "typical document" sample (rule C)
n_top_share         <- 10L         # rule A: delivered rule (top share, >= 25 tokens, distinct text and author) extended from 5 to 10 documents
rep_min_tokens      <- 25L
n_typical           <- 10L         # rule C: seeded random sample of documents dominated by the topic
typical_min_share   <- 0.5
typical_min_tokens  <- 10L
typical_max_tokens  <- 100L
narrow_min_tokens   <- 10L         # narrowness measures use documents with at least this many modeled tokens
n_top_narrow        <- 100L        # ... and the 100 highest-share documents (>= 25 tokens) per topic
n_words_shown       <- 15L
relevance_lambda    <- 0.6         # as delivered
n_words_other_seed  <- 8L
peak_ratio_flag     <- 3           # temporal concentration flags (measured thresholds, no inference)
top3_share_flag     <- 0.25
moderator_accounts  <- c("AutoModerator", "politics-ModTeam", "PoliticsModeratorBot")
tracked_words <- c(right = "removed by SMART", new = "removed by SMART", us = "removed by SMART", old = "removed by SMART",
                   well = "removed by SMART", left = "kept", back = "kept", state = "kept", rights = "kept", united = "kept")
partner_terms <- tribble(
  ~term, ~removed_partner,
  "wing", "right", "winger", "right", "wingers", "right", "far", "right", "alt", "right", "leaning", "right",
  "rights", "right", "civil", "right", "human", "right",
  "york", "new", "jersey", "new", "hampshire", "new", "mexico", "new", "orleans", "new", "deal", "new", "nyc", "new",
  "usa", "us", "united", "us", "america", "us", "american", "us", "americans", "us",
  "age", "old", "older", "old", "aged", "old", "elderly", "old")
tag_of  <- function(K) sprintf("K%03d", K)
iso_utc <- function(x) format(as.POSIXct(x, origin = "1970-01-01", tz = "UTC"), "%Y-%m-%dT%H:%M:%SZ")
preview <- function(x, n = 240L) {
  y <- stri_trim_both(stri_replace_all_regex(x, "\\s+", " "))
  ifelse(stri_length(y) > n, paste0(stri_sub(y, 1L, n - 1L), "\u2026"), y)
}
sha256 <- function(path) paste0("sha256:", digest(path, algo = "sha256", file = TRUE))
checks <- list()
add_check <- function(group, check, measured, pass) checks[[length(checks) + 1L]] <<- tibble(group = group, check = check, measured = measured, pass = isTRUE(pass))

# ---- Integrity of the delivered outputs (hash on disk vs. the delivered manifests) ----
# INPUT : every *.manifest.yml under lda_topics/tables and lda_topics/figures.
# DOES  : re-hash the file each manifest describes; compare; record script hash and git commit.
# OUTPUT: delivered_outputs_integrity (one row per delivered output); the list of reviewed files.
log_line("hashing delivered outputs")
manifest_files <- list.files(c(in_tab, in_fig), pattern = "\\.manifest\\.yml$", full.names = TRUE)
delivered_outputs_integrity <- bind_rows(lapply(manifest_files, function(mf) {
  mfy  <- read_yaml(mf)
  path <- mfy$output_file
  tibble(file = basename(path), kind = if (grepl("\\.csv$", path)) "table" else "figure",
         size_bytes = file.size(path), manifest_hash = mfy$output_hash, current_hash = sha256(path),
         manifest_rows = if (is.null(mfy$output_rows)) NA_integer_ else mfy$output_rows,
         manifest_cols = if (is.null(mfy$output_cols)) NA_integer_ else mfy$output_cols,
         manifest_script_hash = mfy$transformation$script_hash, manifest_git_commit = mfy$transformation$git_commit,
         manifest_timestamp = mfy$timestamp)
})) |> mutate(hash_matches_manifest = manifest_hash == current_hash) |> arrange(kind, file)
lda_script_hash <- sha256(lda_script_path)
add_check("integrity", "every delivered table and figure on disk has the hash recorded in its manifest",
          sprintf("%d of %d match", sum(delivered_outputs_integrity$hash_matches_manifest), nrow(delivered_outputs_integrity)),
          all(delivered_outputs_integrity$hash_matches_manifest) && nrow(delivered_outputs_integrity) == 55)
add_check("integrity", "the LDA script on disk has the script hash recorded in every delivered manifest",
          sprintf("%s; %d distinct manifest hashes", lda_script_hash, n_distinct(delivered_outputs_integrity$manifest_script_hash)),
          all(delivered_outputs_integrity$manifest_script_hash == lda_script_hash))

# ---- Read the delivered tables ----
# INPUT : lda_topics/tables/*.csv (read-only).
# OUTPUT: in-memory copies; the two per-document tables are read with data.table::fread.
log_line("reading delivered tables")
read_tab <- function(name) read_csv(file.path(in_tab, name), show_col_types = FALSE, guess_max = 100000)
per_k <- function(stem, reader = read_tab) setNames(lapply(retained_k, function(K) reader(sprintf("%s_%s.csv", stem, tag_of(K)))), tag_of(retained_k))
topic_summary       <- per_k("topic_summary")
prev_day            <- per_k("topic_prevalence_by_day")
prev_type           <- per_k("topic_prevalence_by_type")
rep_delivered       <- per_k("representative_documents")
stab_matrix         <- per_k("stability_matrix_seed1_vs_seed2")
conc_summary        <- per_k("document_concentration_summary")
topic_correlation   <- per_k("topic_correlation")
topic_word          <- per_k("topic_word_distribution", function(name) fread(file.path(in_tab, name), encoding = "UTF-8"))
doc_topic           <- per_k("document_topic_distribution", function(name) fread(file.path(in_tab, name), encoding = "UTF-8", na.strings = c("NA", "")))
top_prob_all        <- read_tab("top_words_by_probability_all_models.csv") |> filter(K %in% retained_k)
top_rel_all         <- read_tab("top_words_by_relevance_all_models.csv")   |> filter(K %in% retained_k)
stability_pairs     <- read_tab("stability_pairs.csv") |> filter(K %in% retained_k)
stopwords_used      <- read_tab("stopwords_used.csv")
model_comparison    <- read_tab("model_comparison_summary.csv")
parameters_delivered <- read_tab("parameters.csv")
validation_delivered <- read_tab("validation_checks.csv")
text_processing_report <- read_tab("text_processing_report.csv")
input_tables_read <- c("topic_summary", "topic_prevalence_by_day", "topic_prevalence_by_type", "representative_documents",
                       "stability_matrix_seed1_vs_seed2", "document_concentration_summary", "topic_correlation",
                       "topic_word_distribution", "document_topic_distribution")
input_table_files <- c(unlist(lapply(input_tables_read, function(s) sprintf("%s_%s.csv", s, tag_of(retained_k)))),
                       "top_words_by_probability_all_models.csv", "top_words_by_relevance_all_models.csv", "stability_pairs.csv",
                       "stopwords_used.csv", "model_comparison_summary.csv", "parameters.csv", "validation_checks.csv", "text_processing_report.csv")
add_check("delivered", "the retained K recorded in parameters.csv are the models reviewed here",
          parameters_delivered$value[parameters_delivered$parameter == "retained_k"],
          parameters_delivered$value[parameters_delivered$parameter == "retained_k"] == paste(retained_k, collapse = ", "))
add_check("delivered", "all delivered validation checks are TRUE",
          sprintf("%d of %d", sum(validation_delivered$pass), nrow(validation_delivered)), all(validation_delivered$pass))

# ---- Source text (read-only; used only to display documents and count whole words) ----
# INPUT : the 8 Parquet files.
# DOES  : text per doc_key exactly as the LDA script built it; a letters-only
#         normalised form for near-duplicate detection (lower case, URLs removed,
#         every non-letter run collapsed to one space); whole-word counts of the
#         tracked words on the lower-cased, URL-stripped text, using the LDA
#         script's token boundaries (a word is a maximal run of letters,
#         apostrophes join). Nothing is written back.
# OUTPUT: src_text (doc_key, text, text_norm, norm_count_corpus, w_<word> ...).
log_line("reading source text")
comment_files    <- list.files(comments_dir,    pattern = "\\.parquet$", full.names = TRUE)
submission_files <- list.files(submissions_dir, pattern = "\\.parquet$", full.names = TRUE)
src_text <- rbindlist(c(
  lapply(comment_files, function(p) { d <- read_parquet(p); data.table(doc_key = paste0("t1_", d$id), text = d$body) }),
  lapply(submission_files, function(p) { d <- read_parquet(p); data.table(doc_key = paste0("t3_", d$id), text = paste(d$title, d$selftext, sep = " ")) })))
url_regex <- "(https?://|www\\.)\\S+"
url_opts  <- stri_opts_regex(case_insensitive = TRUE)
text_lc   <- stri_replace_all_regex(stri_replace_all_fixed(stri_trans_tolower(src_text$text), "\u2019", "'"), url_regex, " ", opts_regex = url_opts)
src_text[, text_norm := stri_trim_both(stri_replace_all_regex(text_lc, "[^\\p{L}]+", " "))]
src_text[, norm_count_corpus := .N, by = text_norm]
for (w in names(tracked_words)) {
  set(src_text, j = paste0("w_", w), value = stri_count_regex(text_lc, sprintf("(?<![\\p{L}'])%s(?![\\p{L}'])", w)))
}
rm(text_lc)
setkey(src_text, doc_key)
wcols <- paste0("w_", names(tracked_words))
add_check("source", "source text: one row per document, keys unique", sprintf("%d rows, %d distinct keys", nrow(src_text), uniqueN(src_text$doc_key)),
          nrow(src_text) == 902144L && uniqueN(src_text$doc_key) == nrow(src_text))

# ---- Figure style (as in the LDA script: title, axis titles, tick labels, legend only) ----
ink_series <- "#2a78d6"; ink_series_2 <- "#eb6834"; ink_series_3 <- "#1baf7a"
ink_text <- "#0b0b0b"; ink_sub <- "#52514e"; ink_muted <- "#898781"; grid_col <- "#e1e0d9"; axis_col <- "#c3c2b7"
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

# ---- Per-model review ----
out_tables <- list()
review <- list()
for (K in retained_k) {
  tag <- tag_of(K)
  log_line("reviewing", tag)
  topic_cols <- sprintf("topic_%0*d", nchar(K), seq_len(K))
  ts <- topic_summary[[tag]] |> arrange(topic)
  dt <- merge(doc_topic[[tag]], src_text, by = "doc_key", all.x = TRUE, sort = TRUE)
  add_check(tag, "every row of the document-topic table found its source text by doc_key",
            sprintf("%d rows, %d without text", nrow(dt), sum(is.na(dt$text))), nrow(dt) == 902144L && !anyNA(dt$text))
  m       <- dt[model_status == "modeled"]
  theta_m <- as.matrix(m[, topic_cols, with = FALSE])
  n_modeled <- nrow(m)
  rm(dt)

  # -- 1. Topic-word evidence: both delivered rankings, mass concentration, shared vocabulary --
  tp <- top_prob_all |> filter(K == !!K, seed == 1L) |> arrange(topic, rank)
  tr <- top_rel_all  |> filter(K == !!K, seed == 1L) |> arrange(topic, rank)
  words_long <- bind_rows(tp |> mutate(ranking = "probability"), tr |> mutate(ranking = "relevance")) |>
    mutate(lift = phi / term_share_of_training_tokens,
           relevance_score = relevance_lambda * log(phi) + (1 - relevance_lambda) * log(lift)) |>
    transmute(K, topic, ranking, rank, term, phi, term_share_of_training_tokens, lift, relevance_score)
  tw      <- topic_word[[tag]]
  phi_mat <- as.matrix(tw[, topic_cols, with = FALSE])
  mass <- t(vapply(seq_len(K), function(k) {
    s <- sort(phi_mat[, k], decreasing = TRUE); cs <- cumsum(s)
    c(phi_mass_top10 = cs[10], phi_mass_top50 = cs[50], phi_mass_top200 = cs[200], terms_to_half_of_mass = which(cs >= 0.5)[1])
  }, numeric(4)))
  word_topic_count <- tp |> filter(rank <= 10) |> count(term, name = "n_topics_with_word_in_top10")
  generic <- tp |> filter(rank <= 10) |> left_join(word_topic_count, by = "term") |>
    group_by(topic) |>
    summarise(top10_words_shared_with_2plus_other_topics = sum(n_topics_with_word_in_top10 >= 3),
              top10_words_unique_to_topic = sum(n_topics_with_word_in_top10 == 1), .groups = "drop")
  paste_top <- function(d, n_words) d |> filter(rank <= n_words) |> group_by(topic) |> summarise(words = paste(term, collapse = ", "), .groups = "drop")
  topic_word_evidence <- ts |>
    transmute(topic, topic_label, token_share, doc_mean_share, dominant_share, coherence_npmi, exclusivity) |>
    left_join(paste_top(tp, n_words_shown) |> rename(top_words_by_probability = words), by = "topic") |>
    left_join(paste_top(tr, n_words_shown) |> rename(top_words_by_relevance = words), by = "topic") |>
    bind_cols(as_tibble(mass)) |>
    left_join(generic, by = "topic") |>
    mutate(K = K, .before = 1)
  shared_top10_words <- tp |> filter(rank <= 10) |> group_by(term) |>
    summarise(n_topics_with_word_in_top10 = n(), topics = paste(topic, collapse = ", "),
              term_share_of_training_tokens = first(term_share_of_training_tokens), .groups = "drop") |>
    filter(n_topics_with_word_in_top10 >= 2) |> arrange(desc(n_topics_with_word_in_top10), term) |> mutate(K = K, .before = 1)

  # -- 2. Representative documents: rule A (delivered rule, 10 per topic) and rule C (seeded typical sample) --
  cand_idx <- which(m$n_tokens_modeled >= rep_min_tokens)
  meta_cols <- c("doc_key", "doc_id", "doc_type", "author", "created_utc", "submission_id", "link_id", "parent_id", "split",
                 "n_tokens_modeled", "n_tokens_letters", "dominant_topic", "dominant_topic_share", "norm_count_corpus", wcols)
  rep_top <- rbindlist(lapply(seq_len(K), function(k) {
    o <- cand_idx[order(-theta_m[cand_idx, k], m$doc_key[cand_idx])]
    o <- o[!duplicated(m$text[o]) & !duplicated(m$author[o])]
    r <- head(o, n_top_share)
    data.table(topic = k, rank = seq_along(r), topic_share_in_document = theta_m[r, k], m[r, meta_cols, with = FALSE],
               text_norm = m$text_norm[r], text = m$text[r])
  }))
  rep_top[, near_duplicate_of_higher_rank := duplicated(text_norm), by = topic]
  rep_top[, created_utc_iso := iso_utc(created_utc)]
  rep_top[, text_preview := preview(text)]
  rep_top[, text_norm := NULL]
  setcolorder(rep_top, c("topic", "rank", "topic_share_in_document", "doc_key", "doc_id", "doc_type", "author", "created_utc", "created_utc_iso",
                         "submission_id", "link_id", "parent_id", "split", "n_tokens_modeled", "n_tokens_letters", "dominant_topic",
                         "dominant_topic_share", "norm_count_corpus", "near_duplicate_of_higher_rank", wcols, "text_preview", "text"))
  del <- as.data.table(rep_delivered[[tag]])
  cmp <- merge(del[, .(topic, rank, doc_key_delivered = doc_key, share_delivered = topic_share_in_document)],
               rep_top[rank <= 5L, .(topic, rank, doc_key, topic_share_in_document)], by = c("topic", "rank"))
  add_check(tag, "rule A reproduces the delivered representative documents in its first five ranks (doc_key and share)",
            sprintf("%d of %d rows agree; max share difference %g", sum(cmp$doc_key == cmp$doc_key_delivered), nrow(del),
                    max(abs(cmp$share_delivered - cmp$topic_share_in_document))),
            nrow(cmp) == nrow(del) && all(cmp$doc_key == cmp$doc_key_delivered) && max(abs(cmp$share_delivered - cmp$topic_share_in_document)) < 1e-6)

  set.seed(review_seed)
  rep_typical <- rbindlist(lapply(seq_len(K), function(k) {
    c2 <- which(m$dominant_topic == k & m$dominant_topic_share >= typical_min_share &
                  m$n_tokens_modeled >= typical_min_tokens & m$n_tokens_modeled <= typical_max_tokens)
    n_eligible <- length(c2)
    if (n_eligible == 0L) return(NULL)
    o <- c2[sample.int(n_eligible)]
    o <- o[!duplicated(m$text_norm[o]) & !duplicated(m$author[o])]
    r <- head(o, n_typical)
    data.table(topic = k, rank = seq_along(r), eligible_docs = n_eligible, topic_share_in_document = theta_m[r, k],
               m[r, meta_cols, with = FALSE], text = m$text[r])
  }))
  rep_typical[, created_utc_iso := iso_utc(created_utc)]
  rep_typical[, text_preview := preview(text)]
  setcolorder(rep_typical, c("topic", "rank", "eligible_docs", "topic_share_in_document", "doc_key", "doc_id", "doc_type", "author", "created_utc",
                             "created_utc_iso", "submission_id", "link_id", "parent_id", "split", "n_tokens_modeled", "n_tokens_letters",
                             "dominant_topic", "dominant_topic_share", "norm_count_corpus", wcols, "text_preview", "text"))
  add_check(tag, "rule C: every sampled document is dominated by its topic with share >= 0.5 and 10-100 modeled tokens; texts and authors distinct within topic",
            sprintf("%d rows over %d topics; min share %.3f; tokens %d..%d", nrow(rep_typical), n_distinct(rep_typical$topic),
                    min(rep_typical$dominant_topic_share), min(rep_typical$n_tokens_modeled), max(rep_typical$n_tokens_modeled)),
            all(rep_typical$dominant_topic == rep_typical$topic) && all(rep_typical$dominant_topic_share >= typical_min_share) &&
              all(rep_typical$n_tokens_modeled >= typical_min_tokens & rep_typical$n_tokens_modeled <= typical_max_tokens) &&
              all(rep_typical[, .(ok = !anyDuplicated(author)), by = topic]$ok))

  # -- 3. Narrowness: is a topic carried by many distinct texts or by a few repeated ones? --
  md <- m[n_tokens_modeled >= narrow_min_tokens, .(dominant_topic, dominant_topic_share, author, text_norm, text, submission_id)]
  md[, norm_count_in_topic := .N, by = .(dominant_topic, text_norm)]
  narrow <- md[, .(dominant_docs_ge10_tokens = .N,
                   distinct_norm_texts = uniqueN(text_norm),
                   share_docs_with_repeated_norm_text = mean(norm_count_in_topic >= 2L),
                   top_norm_text_docs = max(norm_count_in_topic),
                   distinct_authors = uniqueN(author),
                   distinct_submissions = uniqueN(submission_id),
                   docs_share_ge_0.5 = sum(dominant_topic_share >= 0.5),
                   share_docs_automoderator = mean(author == "AutoModerator"),
                   share_docs_politics_modteam = mean(author == "politics-ModTeam")), by = .(topic = dominant_topic)]
  narrow[, top_norm_text_share := top_norm_text_docs / dominant_docs_ge10_tokens]
  top_text <- md[order(dominant_topic, -norm_count_in_topic, text_norm)][, head(.SD, 1L), by = .(topic = dominant_topic)][
    , .(topic, top_norm_text_preview = preview(text, 160L), top_norm_text_author_example = author)]
  top_auth <- md[, .N, by = .(topic = dominant_topic, author)][order(topic, -N, author)][, head(.SD, 3L), by = topic]
  top_auth_w <- top_auth[, .(top_author_ge10_tokens = author[1], top_author_docs_ge10_tokens = N[1],
                             authors_2_and_3 = paste(sprintf("%s (%d)", author[-1], N[-1]), collapse = "; ")), by = topic]
  all_dom <- m[, .(dominant_docs_all_lengths = .N, share_automoderator_all_lengths = mean(author == "AutoModerator"),
                   share_politics_modteam_all_lengths = mean(author == "politics-ModTeam")), by = .(topic = dominant_topic)]
  top100 <- rbindlist(lapply(seq_len(K), function(k) {
    o <- head(cand_idx[order(-theta_m[cand_idx, k], m$doc_key[cand_idx])], n_top_narrow)
    a <- sort(table(m$author[o]), decreasing = TRUE)
    data.table(topic = k, top100_min_share = min(theta_m[o, k]), top100_distinct_norm_texts = uniqueN(m$text_norm[o]),
               top100_distinct_authors = uniqueN(m$author[o]), top100_distinct_submissions = uniqueN(m$submission_id[o]),
               top100_top_author = names(a)[1], top100_top_author_docs = as.integer(a[1]))
  }))
  narrowness <- tibble(topic = seq_len(K)) |>
    left_join(ts |> transmute(topic, token_share, dominant_docs, top_author_delivered = top_author,
                              top_author_share_delivered = top_author_share_of_dominant_docs), by = "topic") |>
    left_join(all_dom, by = "topic") |> left_join(narrow, by = "topic") |> left_join(top_auth_w, by = "topic") |>
    left_join(top_text, by = "topic") |> left_join(top100, by = "topic") |>
    mutate(top_author_share_ge10_tokens = top_author_docs_ge10_tokens / dominant_docs_ge10_tokens, K = K, .before = 1)
  add_check(tag, "dominant-document counts recomputed from the document-topic table equal the delivered topic summary",
            sprintf("%d of %d topics agree", sum(narrowness$dominant_docs == narrowness$dominant_docs_all_lengths), K),
            all(narrowness$dominant_docs == narrowness$dominant_docs_all_lengths))
  top_auth_all <- m[, .N, by = .(topic = dominant_topic, author)][order(topic, -N, author)][, head(.SD, 1L), by = topic]
  add_check(tag, "top author per topic recomputed over all dominant documents equals the delivered top_author",
            sprintf("%d of %d topics agree", sum(top_auth_all$author == ts$top_author[top_auth_all$topic]), K),
            all(top_auth_all$author == ts$top_author[top_auth_all$topic]))

  # -- 4. Cross-seed stability of every retained (seed-1) topic --
  words_other <- top_prob_all |> filter(K == !!K, seed %in% c(2L, 3L), rank <= n_words_other_seed) |>
    group_by(seed, topic) |> summarise(words = paste(term, collapse = ", "), .groups = "drop")
  sp <- stability_pairs |> filter(K == !!K, seed_a == 1L) |>
    left_join(words_other, by = c("seed_b" = "seed", "topic_b" = "topic")) |>
    transmute(topic = topic_a, seed_b, topic_b, cosine, jaccard_top20 = jaccard_top_words, token_share_b, words) |>
    pivot_wider(names_from = seed_b, values_from = c(topic_b, cosine, jaccard_top20, token_share_b, words), names_glue = "{.value}_seed{seed_b}")
  sm <- stab_matrix[[tag]] |> group_by(seed1_topic) |> arrange(desc(cosine), .by_group = TRUE) |>
    summarise(best_cosine_seed2 = cosine[1], best_seed2_topic = seed2_topic[1],
              second_cosine_seed2 = cosine[2], second_seed2_topic = seed2_topic[2],
              matched_cosine_seed2_from_matrix = cosine[matched][1], hungarian_match_is_row_max = seed2_topic[matched][1] == seed2_topic[1],
              .groups = "drop")
  stability <- sp |> left_join(sm, by = c("topic" = "seed1_topic")) |>
    mutate(min_cosine = pmin(cosine_seed2, cosine_seed3), mean_cosine = (cosine_seed2 + cosine_seed3) / 2,
           both_seeds_ge_0.8 = min_cosine >= 0.8, any_seed_below_0.5 = min_cosine < 0.5,
           gap_best_to_second_seed2 = best_cosine_seed2 - second_cosine_seed2) |>
    left_join(ts |> transmute(topic, token_share, coherence_npmi, words_seed1 = top_10_words_by_probability), by = "topic") |>
    arrange(desc(min_cosine)) |> mutate(rank_by_min_cosine = row_number()) |>
    select(topic, rank_by_min_cosine, min_cosine, mean_cosine, cosine_seed2, cosine_seed3, jaccard_top20_seed2, jaccard_top20_seed3,
           both_seeds_ge_0.8, any_seed_below_0.5, topic_b_seed2, topic_b_seed3, token_share, token_share_b_seed2, token_share_b_seed3,
           best_cosine_seed2, best_seed2_topic, second_cosine_seed2, second_seed2_topic, gap_best_to_second_seed2,
           hungarian_match_is_row_max, matched_cosine_seed2_from_matrix, coherence_npmi, words_seed1, words_seed2, words_seed3) |>
    mutate(K = K, .before = 1)
  matched_ids <- stab_matrix[[tag]] |> filter(matched) |> arrange(seed1_topic)
  ids_agree   <- sum(stability$topic_b_seed2[order(stability$topic)] == matched_ids$seed2_topic)
  add_check(tag, "matched cosine in stability_pairs equals the matched cell of the seed 1 x seed 2 matrix; seed-2 counterpart ids agree",
            sprintf("max difference %g; %d of %d ids agree", max(abs(stability$cosine_seed2 - stability$matched_cosine_seed2_from_matrix)), ids_agree, K),
            max(abs(stability$cosine_seed2 - stability$matched_cosine_seed2_from_matrix)) < 1e-9 && ids_agree == K && nrow(matched_ids) == K)
  add_check(tag, "matched cosine to seeds 2 and 3 in the delivered topic summary equals stability_pairs",
            sprintf("max difference %g", max(abs(c(stability$cosine_seed2 - ts$matched_cosine_vs_seed_2[stability$topic],
                                                    stability$cosine_seed3 - ts$matched_cosine_vs_seed_3[stability$topic])))),
            max(abs(c(stability$cosine_seed2 - ts$matched_cosine_vs_seed_2[stability$topic],
                      stability$cosine_seed3 - ts$matched_cosine_vs_seed_3[stability$topic]))) < 1e-9)

  # -- 5. Prevalence and time structure --
  pd <- prev_day[[tag]] |> mutate(topic_tokens = token_share * modeled_tokens)
  day_share <- pd |> distinct(created_date_utc, modeled_tokens) |> mutate(day_share = modeled_tokens / sum(modeled_tokens)) |> arrange(desc(day_share))
  time_structure <- pd |> group_by(topic) |> arrange(desc(topic_tokens), .by_group = TRUE) |>
    summarise(daily_share_min = min(token_share), daily_share_median = median(token_share), daily_share_max = max(token_share),
              peak_day = created_date_utc[1], peak_to_median_ratio = daily_share_max / daily_share_median,
              share_of_topic_tokens_on_peak_day = topic_tokens[1] / sum(topic_tokens),
              share_of_topic_tokens_top3_days = sum(topic_tokens[1:3]) / sum(topic_tokens),
              days_to_half_of_topic_tokens = which(cumsum(topic_tokens) / sum(topic_tokens) >= 0.5)[1],
              n_days_at_least_2x_median = sum(token_share >= 2 * median(token_share)),
              top3_days = paste(sprintf("%s (%.2f%%)", format(created_date_utc[1:3], "%b %d"), 100 * token_share[1:3]), collapse = "; "),
              .groups = "drop") |>
    left_join(ts |> transmute(topic, token_share, doc_mean_share, dominant_share, token_share_comments, token_share_submissions), by = "topic") |>
    mutate(uniform_reference_peak_day_share = day_share$day_share[1],
           uniform_reference_top3_days_share = sum(day_share$day_share[1:3]),
           concentration_flag = peak_to_median_ratio >= peak_ratio_flag | share_of_topic_tokens_top3_days >= top3_share_flag) |>
    arrange(desc(peak_to_median_ratio)) |> mutate(rank_by_peak_ratio = row_number()) |>
    select(topic, rank_by_peak_ratio, token_share, doc_mean_share, dominant_share, token_share_comments, token_share_submissions,
           daily_share_min, daily_share_median, daily_share_max, peak_day, peak_to_median_ratio, n_days_at_least_2x_median,
           share_of_topic_tokens_on_peak_day, share_of_topic_tokens_top3_days, days_to_half_of_topic_tokens,
           uniform_reference_peak_day_share, uniform_reference_top3_days_share, concentration_flag, top3_days) |>
    mutate(K = K, .before = 1)
  add_check(tag, "daily token shares sum to 1 on every day and topic tokens over days reproduce the delivered all-document token share",
            sprintf("day sums %.6f..%.6f; max share difference %g",
                    min(tapply(pd$token_share, pd$created_date_utc, sum)), max(tapply(pd$token_share, pd$created_date_utc, sum)),
                    max(abs(tapply(pd$topic_tokens, pd$topic, sum) / sum(pd$topic_tokens) - ts$token_share))),
            max(abs(tapply(pd$topic_tokens, pd$topic, sum) / sum(pd$topic_tokens) - ts$token_share)) < 1e-6)

  # -- 6. Preprocessing signals: tracked whole words by dominant topic; partner terms; representative-document counts --
  wc_topic <- m[, c(lapply(.SD, sum), .(letter_tokens = sum(n_tokens_letters), modeled_tokens = sum(n_tokens_modeled), docs = .N)),
                by = .(topic = dominant_topic), .SDcols = wcols]
  wc_docs  <- m[, lapply(.SD, function(x) mean(x > 0)), by = .(topic = dominant_topic), .SDcols = wcols]
  stopword_use <- bind_rows(lapply(names(tracked_words), function(w) {
    tibble(topic = wc_topic$topic, word = w, status = tracked_words[[w]], docs = wc_topic$docs,
           occurrences = wc_topic[[paste0("w_", w)]],
           per_100_letter_tokens = 100 * wc_topic[[paste0("w_", w)]] / wc_topic$letter_tokens,
           share_docs_containing = wc_docs[[paste0("w_", w)]][match(wc_topic$topic, wc_docs$topic)])
  })) |> arrange(word, topic) |> mutate(K = K, .before = 1)
  stopword_use_by_topic_wide <- stopword_use |> filter(word %in% c("right", "new", "us", "old", "well", "left")) |>
    select(topic, word, per_100_letter_tokens) |> pivot_wider(names_from = word, values_from = per_100_letter_tokens, names_prefix = "per100_") |>
    left_join(stopword_use |> filter(word %in% c("right", "new", "us")) |> select(topic, word, share_docs_containing) |>
                pivot_wider(names_from = word, values_from = share_docs_containing, names_prefix = "docs_with_"), by = "topic") |>
    left_join(ts |> transmute(topic, token_share, top_10_words_by_probability), by = "topic") |>
    mutate(share_letter_tokens_not_modeled = 1 - wc_topic$modeled_tokens[match(topic, wc_topic$topic)] / wc_topic$letter_tokens[match(topic, wc_topic$topic)]) |>
    mutate(K = K, .before = 1) |>
    arrange(desc(per100_right))
  rep_word_use <- rep_top[, c(.(docs = .N, docs_with_right_new_or_us = sum(w_right + w_new + w_us > 0)), lapply(.SD, sum)), by = topic, .SDcols = wcols]
  setnames(rep_word_use, wcols, paste0("occurrences_", names(tracked_words)))
  rep_word_use[, K := K]; setcolorder(rep_word_use, "K")
  vocab_terms <- tw$term
  partner <- partner_terms |> mutate(in_vocabulary = term %in% vocab_terms) |>
    rowwise() |>
    mutate(term_frequency_train = if (in_vocabulary) tw$term_frequency_train[match(term, vocab_terms)] else NA_real_,
           top_topic = if (in_vocabulary) which.max(phi_mat[match(term, vocab_terms), ]) else NA_integer_,
           top_topic_phi = if (in_vocabulary) max(phi_mat[match(term, vocab_terms), ]) else NA_real_,
           top_topic_share_of_term_mass = if (in_vocabulary) max(phi_mat[match(term, vocab_terms), ]) / sum(phi_mat[match(term, vocab_terms), ]) else NA_real_) |>
    ungroup() |>
    left_join(words_long |> filter(term %in% partner_terms$term) |> group_by(term) |>
                summarise(n_topics_top20_probability = sum(ranking == "probability"), n_topics_top20_relevance = sum(ranking == "relevance"),
                          topics_top20 = paste(sprintf("Topic %d (%s rank %d)", topic, ranking, rank), collapse = "; "), .groups = "drop"), by = "term") |>
    mutate(across(c(n_topics_top20_probability, n_topics_top20_relevance), ~ coalesce(.x, 0L))) |>
    mutate(K = K, .before = 1)

  # -- 7. Moderation-account topics: the evidence behind the reported AutoModerator concentration --
  mod_topics <- ts$topic[ts$top_author %in% moderator_accounts]
  mod_texts <- md[dominant_topic %in% mod_topics][order(dominant_topic, -norm_count_in_topic, text_norm)][
    , head(.SD, 1L), by = .(dominant_topic, text_norm)][order(dominant_topic, -norm_count_in_topic)][, head(.SD, 3L), by = dominant_topic][
      , .(topic = dominant_topic, text_rank = seq_len(.N), docs = norm_count_in_topic, author_example = author, text_preview = preview(text, 200L)), by = dominant_topic][, dominant_topic := NULL]
  mod_summary <- ts |> filter(topic %in% mod_topics) |>
    transmute(K = K, topic, token_share, dominant_docs, top_author, top_author_dominant_docs, top_author_share_of_dominant_docs,
              matched_cosine_vs_seed_2, matched_cosine_vs_seed_3, coherence_npmi, exclusivity, top_10_words_by_probability, top_10_words_by_relevance) |>
    left_join(narrowness |> select(topic, dominant_docs_ge10_tokens, distinct_norm_texts, share_docs_with_repeated_norm_text, top_norm_text_docs,
                                   share_automoderator_all_lengths, share_politics_modteam_all_lengths), by = "topic")
  mod_account_by_topic <- m[author %in% moderator_accounts, .(docs = .N), by = .(author, topic = dominant_topic)][order(author, -docs)]
  mod_account_by_topic[, share_of_account_docs := docs / sum(docs), by = author]
  mod_account_by_topic[, K := K]; setcolorder(mod_account_by_topic, "K")
  automod_50_99 <- m[n_tokens_modeled >= 50L & n_tokens_modeled <= 99L, .(docs = .N, share_automoderator = mean(author == "AutoModerator"))]

  review[[tag]] <- list(K = K, ts = ts, words_long = words_long, topic_word_evidence = topic_word_evidence, shared_top10_words = shared_top10_words,
                        rep_top = rep_top, rep_typical = rep_typical, narrowness = narrowness, stability = stability,
                        time_structure = time_structure, pd = pd, stopword_use = stopword_use, stopword_use_wide = stopword_use_by_topic_wide,
                        rep_word_use = rep_word_use, partner = partner, mod_summary = mod_summary, mod_texts = mod_texts |> mutate(K = K, .before = 1),
                        mod_account_by_topic = mod_account_by_topic, automod_50_99 = automod_50_99, n_modeled = n_modeled)
  out_tables[[sprintf("topic_words_long_%s.csv", tag)]]                 <- words_long
  out_tables[[sprintf("topic_word_evidence_%s.csv", tag)]]              <- topic_word_evidence
  out_tables[[sprintf("shared_top10_words_%s.csv", tag)]]               <- shared_top10_words
  out_tables[[sprintf("representative_documents_top_share_%s.csv", tag)]] <- rep_top
  out_tables[[sprintf("representative_documents_typical_%s.csv", tag)]] <- rep_typical
  out_tables[[sprintf("topic_narrowness_%s.csv", tag)]]                 <- narrowness
  out_tables[[sprintf("topic_stability_%s.csv", tag)]]                  <- stability
  out_tables[[sprintf("topic_time_structure_%s.csv", tag)]]             <- time_structure
  out_tables[[sprintf("tracked_words_by_topic_%s.csv", tag)]]           <- stopword_use
  out_tables[[sprintf("tracked_words_by_topic_wide_%s.csv", tag)]]      <- stopword_use_by_topic_wide
  out_tables[[sprintf("tracked_words_in_top_share_documents_%s.csv", tag)]] <- rep_word_use
  out_tables[[sprintf("partner_terms_%s.csv", tag)]]                    <- partner
  out_tables[[sprintf("moderation_topics_summary_%s.csv", tag)]]        <- mod_summary
  out_tables[[sprintf("moderation_topics_texts_%s.csv", tag)]]          <- mod_texts |> mutate(K = K, .before = 1)
  out_tables[[sprintf("moderator_accounts_by_topic_%s.csv", tag)]]      <- mod_account_by_topic
  rm(m, theta_m, md, phi_mat, tw); invisible(gc())
}

# ---- Corpus-level counts of the tracked words (all source documents; reference only) ----
n_letter_tokens_all <- text_processing_report$value[text_processing_report$item == "step 4: letter-run tokens extracted"]
n_tokens_modeled_all <- text_processing_report$value[text_processing_report$item == "tokens modeled"]
tracked_words_corpus <- tibble(word = names(tracked_words), status = unname(tracked_words),
                               in_smart_list = names(tracked_words) %in% stopwords_used$term,
                               occurrences_all_documents = vapply(wcols, function(cn) sum(src_text[[cn]]), numeric(1)),
                               documents_containing = vapply(wcols, function(cn) sum(src_text[[cn]] > 0), numeric(1))) |>
  mutate(share_of_all_documents = documents_containing / nrow(src_text),
         occurrences_per_100_letter_tokens_all_documents = 100 * occurrences_all_documents / n_letter_tokens_all,
         occurrences_relative_to_modeled_tokens = occurrences_all_documents / n_tokens_modeled_all)
add_check("source", "tracked words: SMART membership on disk matches the status assumed here",
          paste(sprintf("%s=%s", tracked_words_corpus$word, tracked_words_corpus$in_smart_list), collapse = ", "),
          all(tracked_words_corpus$in_smart_list == (tracked_words_corpus$status != "kept")))
out_tables[["tracked_words_corpus.csv"]] <- tracked_words_corpus

# ---- README consistency: claims in lda_topics/README.md against the delivered tables ----
# INPUT : the delivered tables and the values this review computed.
# DOES  : recompute each numeric claim; pass when the rounded value agrees.
# OUTPUT: readme_consistency_checks.
rc <- list()
add_rc <- function(section, claim, measured, pass) rc[[length(rc) + 1L]] <<- tibble(section = section, readme_claim = claim, measured = measured, agrees = isTRUE(pass))
ts10 <- review$K010$ts; ts60 <- review$K060$ts
st10 <- review$K010$stability; st60 <- review$K060$stability
tt60 <- review$K060$time_structure; pd60 <- review$K060$pd
cs10 <- conc_summary$K010; cs60 <- conc_summary$K060
r2 <- function(x, d = 1) round(x, d)
add_rc("K = 10", "Topic 10 is 70% AutoModerator among the documents it dominates (table: 69.8%)",
       sprintf("%.1f%%", 100 * ts10$top_author_share_of_dominant_docs[10]), r2(100 * ts10$top_author_share_of_dominant_docs[10]) == 69.8)
add_rc("K = 10", "Topic 10 coherence 0.78, cosine 1.00 / 0.98", sprintf("%.2f, %.2f / %.2f", ts10$coherence_npmi[10], ts10$matched_cosine_vs_seed_2[10], ts10$matched_cosine_vs_seed_3[10]),
       r2(ts10$coherence_npmi[10], 2) == 0.78 && r2(ts10$matched_cosine_vs_seed_2[10], 2) == 1 && r2(ts10$matched_cosine_vs_seed_3[10], 2) == 0.98)
add_rc("K = 10", "Topics 2, 3, 4, 5 and 10 recur in both other seeds with cosine >= 0.8 (and no other topic does)",
       paste(st10$topic[st10$both_seeds_ge_0.8], collapse = ", "), setequal(st10$topic[st10$both_seeds_ge_0.8], c(2, 3, 4, 5, 10)))
add_rc("K = 10", "Topics 1 and 6 have a counterpart at 0.44 and 0.38 against one seed",
       sprintf("Topic 1 min %.2f; Topic 6 min %.2f", st10$min_cosine[st10$topic == 1], st10$min_cosine[st10$topic == 6]),
       r2(st10$min_cosine[st10$topic == 1], 2) == 0.44 && r2(st10$min_cosine[st10$topic == 6], 2) == 0.38)
add_rc("K = 10", "Topic 5 carries 20.3% of submission tokens vs 10.4% of comment tokens; Topic 1 5.2% vs 12.9%",
       sprintf("%.1f%% vs %.1f%%; %.1f%% vs %.1f%%", 100 * ts10$token_share_submissions[5], 100 * ts10$token_share_comments[5],
               100 * ts10$token_share_submissions[1], 100 * ts10$token_share_comments[1]),
       r2(100 * ts10$token_share_submissions[5]) == 20.3 && r2(100 * ts10$token_share_comments[5]) == 10.4 &&
         r2(100 * ts10$token_share_submissions[1]) == 5.2 && r2(100 * ts10$token_share_comments[1]) == 12.9)
add_rc("K = 10", "Token shares nearly even, 7.0-12.8%", sprintf("%.1f%%..%.1f%%", 100 * min(ts10$token_share), 100 * max(ts10$token_share)),
       r2(100 * min(ts10$token_share)) == 7.0 && r2(100 * max(ts10$token_share)) == 12.8)
add_rc("K = 10", "Median dominant share 0.55 (2-4 tokens), 0.50 (5-9), 0.46 (10-24), 0.45 (25-49); 42% of 10-24-token documents half in one topic; 2.7-2.9 active topics",
       sprintf("%s; %.0f%%; %.1f-%.1f", paste(sprintf("%.2f", cs10$dominant_share_median[cs10$token_band %in% c("2-4", "5-9", "10-24", "25-49")]), collapse = ", "),
               100 * cs10$share_docs_dominant_at_least_0.5[cs10$token_band == "10-24"],
               min(cs10$active_topics_mean[cs10$token_band %in% c("2-4", "5-9", "10-24", "25-49")]), max(cs10$active_topics_mean[cs10$token_band %in% c("2-4", "5-9", "10-24", "25-49")])),
       all(r2(cs10$dominant_share_median[match(c("2-4", "5-9", "10-24", "25-49"), cs10$token_band)], 2) == c(0.55, 0.50, 0.46, 0.45)) &&
         r2(100 * cs10$share_docs_dominant_at_least_0.5[cs10$token_band == "10-24"], 0) == 42 &&
         r2(min(cs10$active_topics_mean[cs10$token_band %in% c("2-4", "5-9", "10-24", "25-49")])) == 2.7 &&
         r2(max(cs10$active_topics_mean[cs10$token_band %in% c("2-4", "5-9", "10-24", "25-49")])) == 2.9)
add_rc("K = 10", "50-99-token documents: median dominant share 0.57 (K = 10) / 0.39 (K = 60)",
       sprintf("%.2f / %.2f", cs10$dominant_share_median[cs10$token_band == "50-99"], cs60$dominant_share_median[cs60$token_band == "50-99"]),
       r2(cs10$dominant_share_median[cs10$token_band == "50-99"], 2) == 0.57 && r2(cs60$dominant_share_median[cs60$token_band == "50-99"], 2) == 0.39)
add_rc("K = 10", "the identical AutoModerator notice sits in the 50-99-token band (measured: share of that band's documents authored by AutoModerator)",
       sprintf("%.1f%% of %d documents", 100 * review$K010$automod_50_99$share_automoderator, review$K010$automod_50_99$docs), review$K010$automod_50_99$share_automoderator > 0.25)
add_rc("K = 10", "a 27-token comment repeating one word reaches a 0.996 share of Topic 5",
       sprintf("rank-1 document of Topic 5: %d tokens, share %.3f", rep_delivered$K010$n_tokens_modeled[rep_delivered$K010$topic == 5][1],
               rep_delivered$K010$topic_share_in_document[rep_delivered$K010$topic == 5][1]),
       rep_delivered$K010$n_tokens_modeled[rep_delivered$K010$topic == 5][1] == 27 && r2(rep_delivered$K010$topic_share_in_document[rep_delivered$K010$topic == 5][1], 3) == 0.996)
add_rc("K = 60", "Topic 1: 6.4% of tokens; AutoModerator is top author of 68.6% of its dominant documents; cosine 1.00 / 1.00",
       sprintf("%.2f%%; %.1f%%; %.2f / %.2f", 100 * ts60$token_share[1], 100 * ts60$top_author_share_of_dominant_docs[1], ts60$matched_cosine_vs_seed_2[1], ts60$matched_cosine_vs_seed_3[1]),
       r2(100 * ts60$token_share[1]) == 6.4 && r2(100 * ts60$top_author_share_of_dominant_docs[1]) == 68.6 &&
         r2(ts60$matched_cosine_vs_seed_2[1], 2) == 1 && r2(ts60$matched_cosine_vs_seed_3[1], 2) == 1)
add_rc("K = 60", "Topic 35: AutoModerator 42.3%, cosine 1.00 / 1.00",
       sprintf("%.1f%%; %.2f / %.2f", 100 * ts60$top_author_share_of_dominant_docs[35], ts60$matched_cosine_vs_seed_2[35], ts60$matched_cosine_vs_seed_3[35]),
       r2(100 * ts60$top_author_share_of_dominant_docs[35]) == 42.3 && r2(ts60$matched_cosine_vs_seed_2[35], 2) == 1 && r2(ts60$matched_cosine_vs_seed_3[35], 2) == 1)
add_rc("K = 60", "politics-ModTeam is the top author for Topics 51 and 59 (3-4%)",
       paste(sprintf("Topic %d (%.1f%%)", ts60$topic[ts60$top_author == "politics-ModTeam"], 100 * ts60$top_author_share_of_dominant_docs[ts60$top_author == "politics-ModTeam"]), collapse = ", "),
       setequal(ts60$topic[ts60$top_author == "politics-ModTeam"], c(51, 59)))
add_rc("K = 60", "Token shares are 0.8-2.9% except Topic 1",
       sprintf("%.2f%%..%.2f%% over Topics 2-60", 100 * min(ts60$token_share[-1]), 100 * max(ts60$token_share[-1])),
       r2(100 * min(ts60$token_share[-1])) == 0.8 && r2(100 * max(ts60$token_share[-1])) == 2.9)
add_rc("K = 60", "twenty-one topics recur in both other seeds with cosine >= 0.8", sprintf("%d", sum(st60$both_seeds_ge_0.8)), sum(st60$both_seeds_ge_0.8) == 21)
add_rc("K = 60", "twenty topics have a counterpart below 0.5 in at least one seed", sprintf("%d", sum(st60$any_seed_below_0.5)), sum(st60$any_seed_below_0.5) == 20)
add_rc("K = 60", "Topic 40 has no counterpart in either seed (0.02 / 0.07)",
       sprintf("%.2f / %.2f", ts60$matched_cosine_vs_seed_2[40], ts60$matched_cosine_vs_seed_3[40]),
       r2(ts60$matched_cosine_vs_seed_2[40], 2) == 0.02 && r2(ts60$matched_cosine_vs_seed_3[40], 2) == 0.07)
cor60 <- as.matrix(topic_correlation$K060[, -1]); diag(cor60) <- 0
add_rc("K = 60", "within-document correlations between topics are all within +/-0.04", sprintf("max |r| off-diagonal %.4f", max(abs(cor60))), max(abs(cor60)) <= 0.04)
peak_of <- function(k) tt60 |> filter(topic == k)
day_val <- function(k, d) 100 * pd60$token_share[pd60$topic == k & pd60$created_date_utc == as.Date(d)]
add_rc("K = 60", "Topic 23 rises from a 0.7% floor to 7.7% on July 7 (4.3-4.9% on July 6 and 8)",
       sprintf("min %.2f%%; peak %.2f%% on %s; Jul 6 %.2f%%, Jul 8 %.2f%%", 100 * peak_of(23)$daily_share_min, 100 * peak_of(23)$daily_share_max, format(peak_of(23)$peak_day, "%b %d"),
               day_val(23, "2026-07-06"), day_val(23, "2026-07-08")),
       r2(100 * peak_of(23)$daily_share_min) == 0.7 && r2(100 * peak_of(23)$daily_share_max) == 7.7 && peak_of(23)$peak_day == as.Date("2026-07-07") &&
         all(r2(c(day_val(23, "2026-07-06"), day_val(23, "2026-07-08"))) %in% c(4.3, 4.9)))
add_rc("K = 60", "Topic 42 rises from 0.5% to 5.0% on July 7",
       sprintf("min %.2f%%; peak %.2f%% on %s", 100 * peak_of(42)$daily_share_min, 100 * peak_of(42)$daily_share_max, format(peak_of(42)$peak_day, "%b %d")),
       r2(100 * peak_of(42)$daily_share_min) == 0.5 && r2(100 * peak_of(42)$daily_share_max) == 5.0 && peak_of(42)$peak_day == as.Date("2026-07-07"))
add_rc("K = 60", "Topic 25 reaches 6.4% on July 17", sprintf("peak %.2f%% on %s", 100 * peak_of(25)$daily_share_max, format(peak_of(25)$peak_day, "%b %d")),
       r2(100 * peak_of(25)$daily_share_max) == 6.4 && peak_of(25)$peak_day == as.Date("2026-07-17"))
add_rc("K = 60", "Topic 29 reaches 4.0-4.5% on July 12-13", sprintf("Jul 12 %.2f%%, Jul 13 %.2f%%", day_val(29, "2026-07-12"), day_val(29, "2026-07-13")),
       all(r2(c(day_val(29, "2026-07-12"), day_val(29, "2026-07-13"))) %in% c(4.0, 4.5)))
add_rc("K = 60", "Median dominant share 0.43 (5-9 tokens), 0.37 (10-24), 0.33 (25-49); 3.1-3.3 active topics; 22% of 10-24-token documents half in one topic",
       sprintf("%s; %.1f-%.1f; %.0f%%", paste(sprintf("%.2f", cs60$dominant_share_median[match(c("5-9", "10-24", "25-49"), cs60$token_band)]), collapse = ", "),
               min(cs60$active_topics_mean[cs60$token_band %in% c("5-9", "10-24", "25-49")]), max(cs60$active_topics_mean[cs60$token_band %in% c("5-9", "10-24", "25-49")]),
               100 * cs60$share_docs_dominant_at_least_0.5[cs60$token_band == "10-24"]),
       all(r2(cs60$dominant_share_median[match(c("5-9", "10-24", "25-49"), cs60$token_band)], 2) == c(0.43, 0.37, 0.33)) &&
         r2(min(cs60$active_topics_mean[cs60$token_band %in% c("5-9", "10-24", "25-49")])) == 3.1 &&
         r2(max(cs60$active_topics_mean[cs60$token_band %in% c("5-9", "10-24", "25-49")])) == 3.3 &&
         r2(100 * cs60$share_docs_dominant_at_least_0.5[cs60$token_band == "10-24"], 0) == 22)
generic_topics <- c(2, 7, 9, 16, 26, 27, 36, 53, 57, 60)
add_rc("K = 60", "general-vocabulary topics 2, 7, 9, 16, 26, 27, 36, 53, 57, 60 have NPMI 0.05-0.17",
       sprintf("%.2f..%.2f", min(ts60$coherence_npmi[generic_topics]), max(ts60$coherence_npmi[generic_topics])),
       r2(min(ts60$coherence_npmi[generic_topics]), 2) == 0.05 && r2(max(ts60$coherence_npmi[generic_topics]), 2) == 0.17)
mc <- model_comparison
add_rc("comparison", "held-out perplexity 2,625.1 (SD 10.5) at K = 60, 2,627.7 at K = 80, 2,610.8 (SD 15.2) at K = 100; band reaches 2,626.0; K = 10 composite 2.75",
       sprintf("%.1f (%.1f); %.1f; %.1f (%.1f); %.1f; %.2f", mc$heldout_perplexity_mean[mc$K == 60], mc$heldout_perplexity_sd[mc$K == 60], mc$heldout_perplexity_mean[mc$K == 80],
               mc$heldout_perplexity_mean[mc$K == 100], mc$heldout_perplexity_sd[mc$K == 100], mc$heldout_perplexity_mean[mc$K == 100] + mc$heldout_perplexity_sd[mc$K == 100],
               mc$composite_rank[mc$K == 10]),
       r2(mc$heldout_perplexity_mean[mc$K == 60]) == 2625.1 && r2(mc$heldout_perplexity_sd[mc$K == 60]) == 10.5 && r2(mc$heldout_perplexity_mean[mc$K == 80]) == 2627.7 &&
         r2(mc$heldout_perplexity_mean[mc$K == 100]) == 2610.8 && r2(mc$heldout_perplexity_sd[mc$K == 100]) == 15.2 &&
         r2(mc$heldout_perplexity_mean[mc$K == 100] + mc$heldout_perplexity_sd[mc$K == 100]) == 2626.0 && mc$composite_rank[mc$K == 10] == 2.75)
add_rc("files", "17 PNG figures and 38 CSV tables, each with a manifest",
       sprintf("%d figures, %d tables, %d manifests", sum(delivered_outputs_integrity$kind == "figure"), sum(delivered_outputs_integrity$kind == "table"), length(manifest_files)),
       sum(delivered_outputs_integrity$kind == "figure") == 17 && sum(delivered_outputs_integrity$kind == "table") == 38)
add_rc("files", "manifests record the LDA script's hash and a git commit (the commit is the HEAD at run time, before the outputs were committed)",
       sprintf("script hash matches: %s; git_commit values: %s", all(delivered_outputs_integrity$manifest_script_hash == lda_script_hash),
               paste(unique(delivered_outputs_integrity$manifest_git_commit), collapse = ", ")),
       all(delivered_outputs_integrity$manifest_script_hash == lda_script_hash))
readme_consistency_checks <- bind_rows(rc)
out_tables[["readme_consistency_checks.csv"]] <- readme_consistency_checks

# ---- Figures ----
log_line("figures")
seed_colours <- c(`2` = ink_series, `3` = ink_series_2)
for (tag in names(review)) {
  r <- review[[tag]]; K <- r$K
  topic_labels <- paste("Topic", seq_len(K)); topic_levels_desc <- rev(topic_labels)
  fig_cols <- if (K <= 12) 3 else 6

  # rfig1 — the ten words ranked highest by relevance (lambda = 0.6) per topic; bar = probability of the word in the topic.
  twr <- r$words_long |> filter(ranking == "relevance", rank <= 10) |> arrange(topic, rank) |>
    mutate(topic_label = factor(paste("Topic", topic), levels = topic_labels), key = paste0(term, "  #", topic)) |>
    mutate(key = factor(key, levels = rev(unique(key))))
  rfig1 <- ggplot(twr, aes(x = phi, y = key)) +
    geom_col(fill = ink_series, width = 0.7) +
    facet_wrap(~ topic_label, scales = "free", ncol = fig_cols) +
    scale_y_discrete(labels = function(x) sub("  #.*$", "", x)) +
    scale_x_continuous(labels = label_number(accuracy = 0.001), n.breaks = 3, expand = expansion(mult = c(0, 0.05))) +
    labs(title = sprintf("Words ranked by relevance (lambda = 0.6) per topic, K = %d \u2014 r/politics, July 2026", K),
         x = "Probability of the word in the topic", y = NULL) +
    theme_fig + theme(panel.grid.major.y = element_blank(), axis.text.y = element_text(size = 8), axis.text.x = element_text(size = 7))
  ggsave(file.path(fig_dir, sprintf("rfig1_top_words_by_relevance_%s.png", tag)), rfig1,
         width = 2.6 * fig_cols + 1, height = 2.2 * ceiling(K / fig_cols) + 1, dpi = 130, bg = "white", limitsize = FALSE)

  # rfig2 — cross-seed cosine of every retained topic against its matched seed-2 and seed-3 topic, ordered by the weaker match.
  stab_long <- r$stability |> select(topic, rank_by_min_cosine, cosine_seed2, cosine_seed3) |>
    pivot_longer(c(cosine_seed2, cosine_seed3), names_to = "seed", values_to = "cosine") |>
    mutate(seed = sub("cosine_seed", "", seed), topic_label = factor(paste("Topic", topic), levels = rev(paste("Topic", r$stability$topic))))
  rfig2 <- ggplot(stab_long, aes(x = cosine, y = topic_label, colour = seed)) +
    geom_line(aes(group = topic_label), colour = grid_col, linewidth = 0.6) +
    geom_point(size = 2.4) +
    scale_colour_manual(values = seed_colours, name = "Compared seed") +
    scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, by = 0.2)) +
    labs(title = sprintf("Cross-seed cosine of each retained topic, K = %d \u2014 r/politics, July 2026", K),
         x = "Cosine similarity of the matched topic-word distributions", y = NULL) +
    theme_fig + theme(legend.position = "bottom", panel.grid.major.y = element_blank())
  ggsave(file.path(fig_dir, sprintf("rfig2_cross_seed_cosine_%s.png", tag)), rfig2, width = 8, height = 3 + 0.22 * K, dpi = 150, bg = "white", limitsize = FALSE)

  # rfig3 — daily share of every topic on one fixed scale, panels ordered by peak-to-median ratio.
  order_topics <- r$time_structure$topic
  pdf_ <- r$pd |> mutate(topic_label = factor(paste("Topic", topic), levels = paste("Topic", order_topics)))
  rfig3 <- ggplot(pdf_, aes(x = created_date_utc, y = token_share)) +
    geom_line(colour = ink_series, linewidth = 0.6) +
    facet_wrap(~ topic_label, ncol = fig_cols) +
    scale_x_date(date_breaks = "10 days", date_labels = "%b %d", expand = expansion(mult = 0.02)) +
    scale_y_continuous(labels = label_percent(accuracy = 1), limits = c(0, NA), expand = expansion(mult = c(0, 0.05))) +
    labs(title = sprintf("Topic share by day, K = %d \u2014 r/politics, July 2026 (UTC)", K),
         x = "Day (UTC)", y = "Share of the day's modeled tokens") +
    theme_fig + theme(axis.text.x = element_text(size = 7), axis.text.y = element_text(size = 7))
  ggsave(file.path(fig_dir, sprintf("rfig3_daily_share_by_topic_%s.png", tag)), rfig3,
         width = 2.4 * fig_cols + 1, height = 1.7 * ceiling(K / fig_cols) + 1.2, dpi = 130, bg = "white", limitsize = FALSE)

  # rfig4 — narrowness: share of a topic's dominant documents (>= 10 tokens) whose normalised text recurs, and the top author's share.
  nar_long <- r$narrowness |>
    select(topic, `Documents whose normalised text recurs within the topic` = share_docs_with_repeated_norm_text,
           `Documents by the topic's most frequent author` = top_author_share_ge10_tokens) |>
    pivot_longer(-topic, names_to = "measure", values_to = "share") |>
    mutate(measure = factor(measure, levels = c("Documents whose normalised text recurs within the topic", "Documents by the topic's most frequent author")),
           topic_label = factor(paste("Topic", topic), levels = rev(paste("Topic", r$narrowness$topic[order(-r$narrowness$share_docs_with_repeated_norm_text)]))))
  rfig4 <- ggplot(nar_long, aes(x = share, y = topic_label, fill = measure)) +
    geom_col(position = position_dodge(width = 0.8), width = 0.7) +
    scale_fill_manual(values = c(ink_series, ink_series_2), name = "Measure") +
    scale_x_continuous(labels = label_percent(accuracy = 1), expand = expansion(mult = c(0, 0.05))) +
    labs(title = sprintf("Repeated texts and top authors per topic, K = %d \u2014 r/politics, July 2026", K),
         x = "Share of the topic's dominant documents with at least 10 modeled tokens", y = NULL) +
    theme_fig + theme(legend.position = "bottom", legend.direction = "vertical", panel.grid.major.y = element_blank())
  ggsave(file.path(fig_dir, sprintf("rfig4_topic_narrowness_%s.png", tag)), rfig4, width = 9, height = 3.5 + 0.22 * K, dpi = 150, bg = "white", limitsize = FALSE)

  # rfig5 — tracked words removed by the SMART list, per 100 letter tokens of the documents each topic dominates.
  sw <- r$stopword_use |> filter(word %in% c("right", "new", "us")) |>
    mutate(word = factor(word, levels = c("right", "new", "us")), topic_label = factor(paste("Topic", topic), levels = topic_levels_desc))
  rfig5 <- ggplot(sw, aes(x = per_100_letter_tokens, y = topic_label, fill = word)) +
    geom_col(position = position_dodge(width = 0.8), width = 0.7) +
    scale_fill_manual(values = c(ink_series, ink_series_2, ink_series_3), name = "Word removed by the SMART list") +
    scale_x_continuous(expand = expansion(mult = c(0, 0.05))) +
    labs(title = sprintf("'right', 'new' and 'us' by dominant topic, K = %d \u2014 r/politics, July 2026", K),
         x = "Occurrences per 100 letter-run tokens", y = NULL) +
    theme_fig + theme(legend.position = "bottom", panel.grid.major.y = element_blank())
  ggsave(file.path(fig_dir, sprintf("rfig5_tracked_words_by_topic_%s.png", tag)), rfig5, width = 9, height = 3.5 + 0.22 * K, dpi = 150, bg = "white", limitsize = FALSE)
}

# ---- Review checks, tables, sidecars ----
review_checks <- bind_rows(checks)
print(as.data.frame(review_checks[, c("group", "check", "pass")]), right = FALSE)
out_tables[["review_checks.csv"]] <- review_checks
out_tables[["delivered_outputs_integrity.csv"]] <- delivered_outputs_integrity
run_end <- Sys.time()
out_tables[["review_run_info.csv"]] <- tibble(item = c("run_start_utc", "run_end_utc", "total_seconds", "R_version", "review_seed", "git_commit"),
                                              value = c(format(run_start, "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"), format(run_end, "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
                                                        as.character(round(as.numeric(difftime(run_end, run_start, units = "secs")), 1)), R.version.string,
                                                        as.character(review_seed), tryCatch(system2("git", c("-C", project_dir, "rev-parse", "--short", "HEAD"), stdout = TRUE, stderr = FALSE), error = function(e) "")))

table_dims <- list()
for (name in names(out_tables)) {
  path <- file.path(tab_dir, name)
  write_csv(as_tibble(out_tables[[name]]), path)
  table_dims[[path]] <- c(rows = nrow(out_tables[[name]]), cols = ncol(out_tables[[name]]))
}

git_commit <- tryCatch(system2("git", c("-C", project_dir, "rev-parse", "--short", "HEAD"), stdout = TRUE, stderr = FALSE), error = function(e) "")
manifest_packages <- c("nanoparquet", "data.table", "dplyr", "tidyr", "stringi", "ggplot2", "scales", "readr", "digest", "yaml")
package_versions  <- lapply(manifest_packages, function(p) list(name = p, version = as.character(packageVersion(p))))
input_files <- c(
  lapply(file.path(in_tab, input_table_files), function(p) list(path = p, hash = sha256(p), format = "csv")),
  lapply(c(comment_files, submission_files), function(p) list(path = p, hash = sha256(p), format = "parquet")),
  list(list(path = lda_script_path, hash = lda_script_hash, format = "r-script"),
       list(path = file.path(lda_dir, "README.md"), hash = sha256(file.path(lda_dir, "README.md")), format = "markdown")))
manifest_parameters <- list(
  purpose = "read-only review of the retained Phase 1A LDA models; no model fitted, no delivered file altered",
  retained_k = paste(retained_k, collapse = ", "),
  rule_a = sprintf("per topic the %d documents with the highest topic share among modeled documents with >= %d modeled tokens, one per distinct stored text and per distinct author, ties by doc_key (delivered rule extended from 5)", n_top_share, rep_min_tokens),
  rule_c = sprintf("per topic %d documents drawn at random (seed %d) from modeled documents whose dominant topic is the topic with share >= %.1f and %d-%d modeled tokens, one per distinct normalised text and per distinct author", n_typical, review_seed, typical_min_share, typical_min_tokens, typical_max_tokens),
  normalised_text = "lower case, U+2019 -> ', URLs removed, every run of non-letters collapsed to one space, trimmed",
  narrowness = sprintf("over documents whose dominant topic is the topic with >= %d modeled tokens; plus the %d highest-share documents (>= %d tokens)", narrow_min_tokens, n_top_narrow, rep_min_tokens),
  tracked_words = paste(sprintf("%s (%s)", names(tracked_words), tracked_words), collapse = "; "),
  tracked_word_counting = "whole-word matches on the lower-cased, URL-stripped text, with the LDA script's letter-run token boundaries",
  relevance_lambda = as.character(relevance_lambda),
  temporal_flags = sprintf("peak-to-median daily share >= %g or >= %.0f%% of the topic's July tokens on its three largest days", peak_ratio_flag, 100 * top3_share_flag),
  moderator_accounts = paste(moderator_accounts, collapse = ", "))
write_sidecar <- function(output) {
  is_csv <- grepl("\\.csv$", output)
  manifest <- list(manifest_version = 1L, output_file = output, output_hash = sha256(output), output_format = if (is_csv) "csv" else "image/png")
  if (is_csv) { dims <- table_dims[[output]]; manifest$output_rows <- unname(dims[["rows"]]); manifest$output_cols <- unname(dims[["cols"]]) }
  manifest$input_files    <- input_files
  manifest$transformation <- list(script = script_path, script_hash = sha256(script_path), parameters = manifest_parameters, git_commit = git_commit)
  manifest$software <- list(language = "R", language_version = paste(R.version$major, R.version$minor, sep = "."), packages = package_versions,
                            os = paste(Sys.info()[["sysname"]], Sys.info()[["release"]]))
  manifest$seed      <- review_seed
  manifest$timestamp <- format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")
  manifest$notes     <- "Review-only output: derived from the delivered Phase 1A LDA tables (read-only) and the source Parquet text (read-only, for display and whole-word counts). Nothing was refitted."
  write_yaml(manifest, paste0(output, ".manifest.yml"))
}
invisible(lapply(list.files(tab_dir, pattern = "\\.csv$", full.names = TRUE), write_sidecar))
invisible(lapply(list.files(fig_dir, pattern = "\\.png$", full.names = TRUE), write_sidecar))

print(as.data.frame(readme_consistency_checks[, c("section", "readme_claim", "measured", "agrees")]), right = FALSE)
cat("\nDone in", round(as.numeric(difftime(run_end, run_start, units = "mins")), 1), "min. Figures ->", fig_dir, "\nTables ->", tab_dir, "\n")
