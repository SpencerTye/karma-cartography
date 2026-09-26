# English stopword lists: audit of the evidence structure behind the consensus rankings
#
# Read-only over the outputs of english_stopword_lists.R (r_analysis_outputs/stopword_lists/).
# Reads the delivered tables and the extracted lists (lists/*.txt), re-derives the consensus
# variables and tiers from the delivered master membership table as a reproduction check, and
# writes audit tables under r_analysis_outputs/stopword_lists_audit/. Nothing under
# stopword_lists/ is modified; no source list is re-acquired; no family assignment is changed;
# no keep/remove decision is made; the Reddit corpus is not read; no topic model is run.

# ---- Setup ----
suppressPackageStartupMessages({
  library(dplyr); library(tidyr); library(tibble); library(purrr); library(stringi); library(readr); library(digest); library(yaml)
})
run_start <- Sys.time()
invisible(Sys.setlocale("LC_COLLATE", "C"))   # reproducible string ordering (same as the source script)
log_line  <- function(...) cat(format(Sys.time(), "%H:%M:%S"), ..., "\n")

project_dir <- Sys.getenv("KC_REPO_ROOT", unset = getwd())
src_dir     <- file.path(project_dir, "r_analysis_outputs/stopword_lists")      # delivered analysis (read-only)
src_tab     <- file.path(src_dir, "tables")
src_lists   <- file.path(src_dir, "lists")
out_dir     <- file.path(project_dir, "r_analysis_outputs/stopword_lists_audit")
tab_dir     <- file.path(out_dir, "tables")
script_path <- file.path(project_dir, "english_stopword_lists_audit.R")
source_script_path <- file.path(project_dir, "english_stopword_lists.R")
dir.create(tab_dir, recursive = TRUE, showWarnings = FALSE)
invisible(file.remove(list.files(tab_dir, pattern = "[.](csv|yml)$", full.names = TRUE)))

# Parameters (recorded in the manifests). The first three repeat the source script's values and are
# verified against its outputs below; the last two are presentation thresholds of this audit only.
near_duplicate_jaccard <- 0.90
tier_ratio             <- c(1, 2, 4)
family_review_jaccard  <- 0.50
boundary_window        <- 20L     # words shown immediately above and below each tier boundary
large_rank_move_share  <- 0.05    # a min-rank move of at least this share of the table counts as large

sha256 <- function(path) paste0("sha256:", digest(path, algo = "sha256", file = TRUE))
checks <- list()
add_check <- function(group, check, measured, pass) checks[[length(checks) + 1L]] <<- tibble(group = group, check = check, measured = as.character(measured), pass = isTRUE(pass))
`%||%` <- function(a, b) if (is.null(a) || length(a) == 0) b else a
collapse_words <- function(x) paste(sort(x), collapse = " ")

# ---- Inputs (delivered tables, read-only) ----
log_line("reading delivered tables")
input_tables <- c("master_membership", "provenance_table", "list_summary", "family_definitions", "family_review_flags", "duplicate_groups",
                  "near_duplicates", "jaccard_pairs_all", "jaccard_matrix", "secondary_crosschecks", "normalisation_log", "acquisition_log",
                  "excluded_candidates", "tier_summary", "run_info", "checks")
rd <- function(name, ...) read_csv(file.path(src_tab, paste0(name, ".csv")), show_col_types = FALSE, progress = FALSE, ...)
master      <- rd("master_membership", col_types = cols(.default = col_integer(), entry = col_character(), entry_class = col_character(), families_any = col_character(), families_all = col_character()))
provenance  <- rd("provenance_table", col_types = cols(.default = col_character(), n_raw_entries = col_integer(), n_unique_entries = col_integer(), is_family_root = col_logical()))
list_summary <- rd("list_summary", col_types = cols(.default = col_character(), n_raw_entries = col_integer(), n_unique_entries = col_integer(), n_duplicates_within = col_integer(),
                                                    n_case_changed = col_integer(), n_apostrophe_changed = col_integer(), n_trimmed = col_integer(), n_word = col_integer(), n_apostrophe = col_integer(),
                                                    n_other_characters = col_integer(), n_punctuation_or_symbol = col_integer(), n_hyphenated = col_integer(), n_numeric = col_integer(), is_family_root = col_logical()))
family_definitions <- rd("family_definitions", col_types = cols(.default = col_character(), n_lists = col_integer(), min_within_family_jaccard = col_double(), n_review_flags = col_integer()))
family_review <- rd("family_review_flags", col_types = cols(.default = col_character(), is_family_root = col_logical(), size = col_integer(), within_family_max_jaccard = col_double(), outside_family_max_jaccard = col_double()))
duplicate_groups <- rd("duplicate_groups", col_types = cols(.default = col_character(), distinct_set_id = col_integer(), n_lists = col_integer(), size = col_integer()))
near_dups   <- rd("near_duplicates", col_types = cols(.default = col_character(), jaccard = col_double(), size_a = col_integer(), size_b = col_integer(), only_in_a = col_integer(), only_in_b = col_integer()))
pairs_all   <- rd("jaccard_pairs_all", col_types = cols(.default = col_character(), jaccard = col_double(), size_a = col_integer(), size_b = col_integer(), intersection = col_integer(), union = col_integer(), only_in_a = col_integer(), only_in_b = col_integer(), same_family = col_logical()))
jaccard_tab <- rd("jaccard_matrix", col_types = cols(.default = col_double(), list_id = col_character()))
crosschecks <- rd("secondary_crosschecks", col_types = cols(.default = col_character(), n_a = col_integer(), n_b = col_integer(), identical_sets = col_logical()))
norm_log    <- rd("normalisation_log", col_types = cols(.default = col_character(), position = col_integer()))
acq_log     <- rd("acquisition_log", col_types = cols(.default = col_character(), cache_hit = col_logical(), bytes = col_double()))
excluded    <- rd("excluded_candidates", col_types = cols(.default = col_character()))
tier_summary_src <- rd("tier_summary", col_types = cols(.default = col_character(), tier = col_integer(), threshold = col_integer(), cumulative_size = col_integer(), size_ratio_to_tier1 = col_double()))
run_info_src <- rd("run_info", col_types = cols(.default = col_character()))
checks_src  <- rd("checks", col_types = cols(.default = col_character(), pass = col_logical()))
for (col in c("families_any", "families_all")) master[[col]] <- coalesce(master[[col]], "")
for (col in c("changes", "dropped")) norm_log[[col]] <- coalesce(norm_log[[col]], "")

# Membership matrix of the 45 included lists, exactly as delivered (normalised entries, 0/1).
included_ids <- sub("^in_", "", grep("^in_", names(master), value = TRUE))
Mx <- as.matrix(master[, paste0("in_", included_ids)]); dimnames(Mx) <- list(master$entry, included_ids); storage.mode(Mx) <- "integer"
prov_inc  <- provenance |> filter(list_id %in% included_ids)
family_of <- setNames(prov_inc$family, prov_inc$list_id)[included_ids]
status_of <- setNames(prov_inc$inclusion_status, prov_inc$list_id)[included_ids]
root_ids  <- prov_inc$list_id[prov_inc$is_family_root]
firm_ids  <- included_ids[status_of == "included"]
provisional_ids <- included_ids[status_of == "included_provisional"]
fam_names <- sort(unique(family_of))
set_of    <- function(id) rownames(Mx)[Mx[, id] == 1L]
sz        <- colSums(Mx)
add_check("inputs", "45 included lists in master_membership.csv, 32 firm and 13 provisional", sprintf("%d / %d / %d", length(included_ids), length(firm_ids), length(provisional_ids)), length(included_ids) == 45 && length(firm_ids) == 32 && length(provisional_ids) == 13)
add_check("inputs", "11 families with one root list each", sprintf("%d families, %d roots", length(fam_names), length(root_ids)), length(fam_names) == 11 && length(root_ids) == 11 && setequal(family_of[root_ids], fam_names))
add_check("inputs", "list sizes in the membership matrix equal n_unique_entries in list_summary.csv", "", all(sz[included_ids] == list_summary$n_unique_entries[match(included_ids, list_summary$list_id)]))

# Pairwise Jaccard recomputed from the membership matrix (must equal the delivered matrix).
inter <- crossprod(Mx); uni <- outer(sz, sz, "+") - inter; J <- inter / uni
Jd <- as.matrix(jaccard_tab[, -1]); rownames(Jd) <- jaccard_tab$list_id
add_check("inputs", "Jaccard matrix recomputed from master_membership equals jaccard_matrix.csv (4 dp)", max(abs(round(J[rownames(Jd), colnames(Jd)], 4) - Jd)), max(abs(round(J[rownames(Jd), colnames(Jd)], 4) - Jd)) < 1e-9)
relation <- function(a, b) {
  A <- set_of(a); B <- set_of(b); i <- length(intersect(A, B)); j <- i / length(union(A, B))
  r <- if (i == length(A) && i == length(B)) "identical" else if (i == length(A)) "a_subset_of_b" else if (i == length(B)) "b_subset_of_a" else if (j >= near_duplicate_jaccard) "near_duplicate" else "distinct"
  list(jaccard = round(j, 4), relationship = r, n_a = length(A), n_b = length(B), intersection = i, only_a = sort(setdiff(A, B)), only_b = sort(setdiff(B, A)))
}

# ---- Reproduction of the consensus variables and tiers from the delivered matrix ----
# DOES  : re-implements lines 541-578 of english_stopword_lists.R on the delivered 0/1 matrix, for any
#         subset of lists. With all 45 lists the result must equal the delivered columns exactly.
consensus_stats <- function(ids) {
  X <- Mx[, ids, drop = FALSE]; fams <- family_of[ids]; fn <- sort(unique(fams))
  set_hash <- vapply(ids, function(id) digest(sort(rownames(X)[X[, id] == 1L]), algo = "sha256"), character(1))
  rep_ids  <- ids[!duplicated(set_hash)]                                    # first list of each identical-set group, in list order
  fam_any  <- sapply(fn, function(f) as.integer(rowSums(X[, ids[fams == f], drop = FALSE]) > 0))
  fam_all  <- sapply(fn, function(f) as.integer(rowSums(X[, ids[fams == f], drop = FALSE]) == sum(fams == f)))
  if (is.null(dim(fam_any))) { fam_any <- matrix(fam_any, ncol = length(fn)); fam_all <- matrix(fam_all, ncol = length(fn)) }
  colnames(fam_any) <- fn; colnames(fam_all) <- fn
  roots <- intersect(root_ids, ids)
  list(stats = tibble(entry = rownames(X), n_lists = as.integer(rowSums(X)), n_distinct_lists = as.integer(rowSums(X[, rep_ids, drop = FALSE])),
                      n_families_any = as.integer(rowSums(fam_any)), n_families_all = as.integer(rowSums(fam_all)), n_family_roots = as.integer(rowSums(X[, roots, drop = FALSE])),
                      families_any = apply(fam_any, 1, function(r) paste(fn[r == 1], collapse = ";")), families_all = apply(fam_all, 1, function(r) paste(fn[r == 1], collapse = ";"))),
       set_hash = set_hash, rep_ids = rep_ids, fam_any = fam_any, fam_all = fam_all, families = fn, roots = roots)
}
choose_tiers <- function(cnt, ratio = tier_ratio) {     # verbatim logic of the source script (lines 565-575)
  vals <- sort(unique(cnt), decreasing = TRUE); vals <- vals[vals >= 2]
  if (length(vals) < 3) return(list(thresholds = rep(NA, 3), sizes = rep(NA, 3), obj = NA))
  best <- NULL
  for (a in seq_along(vals)) for (b in seq_along(vals)) for (cc in seq_along(vals)) if (a < b && b < cc) {
    N <- c(sum(cnt >= vals[a]), sum(cnt >= vals[b]), sum(cnt >= vals[cc]))
    obj <- abs(log(N[2] / N[1]) - log(ratio[2] / ratio[1])) + abs(log(N[3] / N[1]) - log(ratio[3] / ratio[1]))
    if (is.null(best) || obj < best$obj - 1e-12) best <- list(obj = obj, thresholds = vals[c(a, b, cc)], sizes = N)
  }
  best
}
tier_of  <- function(cnt, th) case_when(cnt >= th[1] ~ 1L, cnt >= th[2] ~ 2L, cnt >= th[3] ~ 3L, TRUE ~ NA_integer_)
tier_obj <- function(N, ratio = tier_ratio) abs(log(N[2] / N[1]) - log(ratio[2] / ratio[1])) + abs(log(N[3] / N[1]) - log(ratio[3] / ratio[1]))
sizes_at <- function(cnt, th) vapply(th, function(t) sum(cnt >= t), integer(1))

s45 <- consensus_stats(included_ids)
st45 <- s45$stats
add_check("reproduction", "n_lists, n_distinct_lists, n_families_any, n_families_all, n_family_roots recomputed from the matrix equal the delivered columns", "",
          all(st45$n_lists == master$n_lists) && all(st45$n_distinct_lists == master$n_distinct_lists) && all(st45$n_families_any == master$n_families_any) &&
          all(st45$n_families_all == master$n_families_all) && all(st45$n_family_roots == master$n_family_roots))
add_check("reproduction", "families_any and families_all strings recomputed equal the delivered columns", "", all(st45$families_any == master$families_any) && all(st45$families_all == master$families_all))
add_check("reproduction", "30 distinct sets among the 45 lists (7 identical-set groups)", length(s45$rep_ids), length(s45$rep_ids) == 30 && nrow(duplicate_groups) == 7)
tp45 <- choose_tiers(master$n_lists); tf45 <- choose_tiers(master$n_families_any)
add_check("reproduction", "primary tier thresholds 29/20/7 and sizes 155/310/627 reproduced", paste(paste(tp45$thresholds, collapse = "/"), paste(tp45$sizes, collapse = "/")),
          identical(as.integer(tp45$thresholds), c(29L, 20L, 7L)) && identical(as.integer(tp45$sizes), c(155L, 310L, 627L)))
add_check("reproduction", "family-aware tier thresholds 9/6/3 and sizes 127/253/454 reproduced", paste(paste(tf45$thresholds, collapse = "/"), paste(tf45$sizes, collapse = "/")),
          identical(as.integer(tf45$thresholds), c(9L, 6L, 3L)) && identical(as.integer(tf45$sizes), c(127L, 253L, 454L)))
add_check("reproduction", "tier assignments recomputed equal the delivered tier_primary and tier_family columns", "",
          identical(tier_of(master$n_lists, tp45$thresholds), master$tier_primary) && identical(tier_of(master$n_families_any, tf45$thresholds), master$tier_family))
add_check("reproduction", "delivered tier_summary.csv carries the same thresholds and sizes", "",
          all(tier_summary_src$threshold == c(tp45$thresholds, tf45$thresholds)) && all(tier_summary_src$cumulative_size == c(tp45$sizes, tf45$sizes)))
add_check("reproduction", "master rank equals the row order n_lists descending then entry ascending (C locale)", "", identical(master$rank, seq_len(nrow(master))) && !is.unsorted(order(-master$n_lists, master$entry, method = "radix")))

# Family coverage string per word: "family k/m" for every family with at least one member containing it.
fam_size <- table(family_of)
fam_count_mat <- sapply(fam_names, function(f) rowSums(Mx[, included_ids[family_of == f], drop = FALSE]))
family_coverage <- apply(fam_count_mat, 1, function(r) { k <- which(r > 0); paste(sprintf("%s %d/%d", fam_names[k], r[k], fam_size[fam_names[k]]), collapse = "; ") })
lists_containing <- apply(Mx, 1, function(r) paste(included_ids[r == 1L], collapse = " "))
roots_containing <- apply(Mx[, root_ids, drop = FALSE], 1, function(r) paste(root_ids[r == 1L], collapse = " "))
n_firm_lists <- as.integer(rowSums(Mx[, firm_ids, drop = FALSE])); n_prov_lists <- as.integer(rowSums(Mx[, provisional_ids, drop = FALSE]))
provisional_lists_containing <- apply(Mx[, provisional_ids, drop = FALSE], 1, function(r) paste(provisional_ids[r == 1L], collapse = " "))

# ---- Lineage register (hand-maintained interpretation; every measured fact is recomputed below) ----
# INPUT : the documented_parent / evidence fields of provenance_table.csv, read list by list.
# DOES  : records, for every included list, the immediate parent list proposed by the delivered family
#         assignment, and the BASIS of that single link:
#           root          - no parent claimed (family root);
#           documented    - a source statement names the parent (file header, package help, README, Javadoc);
#           exact_identity- no source statement; the normalised sets are identical;
#           near_identity - no source statement; Jaccard >= 0.90 with the parent;
#           similarity    - no source statement; Jaccard < 0.90 with the parent (containment or overlap only).
#         documentation_strength qualifies documented links: explicit statement / name only / via an
#         intermediate that was not acquired. The chain to the family root is then followed mechanically.
lineage <- tribble(
  ~list_id, ~immediate_parent, ~link_basis, ~documentation_strength, ~link_evidence, ~unresolved_issues,
  "snowball_original", "none", "root", "n/a", "stop.txt header: 'An English stop word list ...'; no parent claimed", "pinned to a 2025 website commit; the list's own date is not stated in the file",
  "tm_english", "snowball_original", "documented", "explicit statement", "tm help for stopwords(): lists from the Snowball stemmer project (svn.tartarus.org ... stop.txt)", "",
  "tidytext_snowball", "tm_english", "documented", "explicit statement", "tidytext help: 'The snowball and SMART sets are pulled from the tm package'", "",
  "stopwords_pkg_snowball", "snowball_original", "documented", "explicit statement", "stopwords help for data_stopwords_snowball: taken from snowball_all.tgz", "the extra entry 'will' (commented out in stop.txt) is not explained by the package documentation",
  "quanteda_snowball", "stopwords_pkg_snowball", "documented", "explicit statement", "quanteda help: lists moved to the stopwords package; identical(quanteda::stopwords, stopwords::stopwords)", "",
  "nltk_english", "snowball_original", "documented", "via an intermediate not acquired", "NLTK corpus README: obtained from the PostgreSQL snowball stopwords directory; 'The English list has been augmented' (nltk_data issue 22)", "the documented ancestor is PostgreSQL english.stop (not acquired); the dates of the intermediate NLTK states are not documented (see version audit)",
  "stopwords_pkg_nltk", "nltk_english", "documented", "explicit statement", "stopwords help: source nltk_data stopwords.zip", "the date of the nltk_data snapshot embedded in stopwords 2.3 is not stated by the package",
  "spark_ml", "nltk_english", "documented", "explicit statement", "Spark README in the stopwords directory repeats NLTK's README (PostgreSQL snowball; English augmented per nltk_data issue 22)", "documented as NLTK-derived, but the content matches no acquired NLTK snapshot: it equals the 175-entry Snowball variant plus can, don, just, now, s, t (see version audit)",
  "dkpro", "nltk_english", "documented", "explicit statement", "DKPro stopwords README: 'Files copied from NLTK.' (quoted in provenance_table.csv)", "the DKPro README is quoted but not preserved under raw/ (only english.txt.gz was acquired); the 2013 NLTK state it copies is not otherwise acquired",
  "dl4j", "stopwords_pkg_snowball", "near_identity", "none", "no header or source statement; measured superset of the 175-entry Snowball variant (Jaccard 0.9067) plus punctuation, malformed tokens and 7 words", "no source statement; malformed entries (----s, \"the); family by measurement only",
  "marimo_upstream", "snowball_original", "documented", "explicit statement", "koheiw/marimo README: 'Marimo extends the Snowball stopword list'; YAML comments mark the additions", "",
  "marimo_pkg", "marimo_upstream", "documented", "explicit statement", "stopwords help for data_stopwords_marimo: adopted from the Snowball collection, then extended (built from the marimo repository)", "",
  "python_stop_words_2014", "snowball_original", "exact_identity", "none", "no source statement in the Alir3z4/stop-words repository; normalised set identical to Snowball", "no source statement; identity with Snowball is measured only",
  "lexicon_python", "python_stop_words_2014", "documented", "explicit statement", "lexicon help for sw_python: 'Python Stopword List', credited to Alireza Savand (pypi stop-words)", "",
  "smart_lextek", "none", "root", "n/a", "lextek page header: 'stopword list generated by Chris Buckley and Gerard Salton at Cornell University'", "the Cornell original (ftp.cs.cornell.edu/pub/smart/english.stop) is unreachable with no Wayback capture, so the root is itself a secondary publication; the page states 571 words but lists 'would' twice",
  "rake_smart", "smart_lextek", "documented", "explicit statement", "SmartStoplist.txt header: 'stop word list from SMART (Salton,1971). Available at ftp://ftp.cs.cornell.edu/pub/smart/english.stop'", "the documented parent is the Cornell file, not the lextek page; identity with the lextek copy is measured",
  "tm_smart", "smart_lextek", "documented", "explicit statement", "tm help: SMART list 'as documented in Appendix 11 of' Lewis et al. (2004)", "the documented parent is the JMLR appendix (URL returns 404 in 2026); identity with the lextek copy is measured",
  "tidytext_smart", "tm_smart", "documented", "explicit statement", "tidytext help: pulled from the tm package", "",
  "stopwords_pkg_smart", "smart_lextek", "documented", "explicit statement", "stopwords help for data_stopwords_smart: 'taken from the online appendix 11 of Lewis et al. (2004)'", "the documented parent is the JMLR appendix; identity with the lextek copy is measured",
  "quanteda_smart", "stopwords_pkg_smart", "documented", "explicit statement", "quanteda re-exports stopwords::stopwords (identical function object)", "",
  "buckley_salton_qdap", "smart_lextek", "documented", "explicit statement", "qdapDictionaries help quotes the Onix page (Salton and Buckley, SMART); Note: 'Reduced from the original 571 words to 546'", "documented as a reduction from 571 to 546 words; measured as apostrophes stripped inside the contractions and the single letters b-z removed, not a word-level reduction (see version audit)",
  "bow_rainbow", "smart_lextek", "near_identity", "none", "stopwords.c carries a libbow copyright header and no source statement; measured: exactly SMART without its 47 apostrophe forms (subset, Jaccard 0.9175)", "no source statement; the SMART relation is measured only",
  "mallet", "bow_rainbow", "exact_identity", "none", "MALLET documentation calls it 'a standard list of English stopwords' without naming a source; normalised set identical to Bow", "no source named by the documentation; identity with Bow is measured only",
  "lexicon_mallet", "mallet", "documented", "explicit statement", "lexicon help for sw_mallet: 'From MAchine Learning for LanguagE Toolkit'", "",
  "weka_rainbow", "bow_rainbow", "documented", "explicit statement", "Rainbow.java Javadoc: 'Stopwords list based on Rainbow: http://www.cs.cmu.edu/~mccallum/bow/rainbow/'", "the additions ll and ve are not documented",
  "yake", "smart_lextek", "near_identity", "none", "no source statement in the repository; measured: SMART plus dr, dra, mr, ms (Jaccard 0.993)", "no source statement; the four title additions are undocumented",
  "glasgow", "none", "root", "n/a", "live page ir.dcs.gla.ac.uk/resources/linguistic_utils/stop_words; no parent documented", "unversioned live page; the attribution to van Rijsbergen's group is not verified",
  "sklearn", "glasgow", "documented", "explicit statement", "_stop_words.py header: 'This list of English stop words is taken from the \"Glasgow Information Retrieval Group\"' with the URL", "differs from the live Glasgow page by computer and fify (absent) and fifty (present); which side changed is not documented",
  "taporware", "glasgow", "documented", "name only", "the file is named glasgowstoplist.txt (Voyant's header cites it under that name); no statement inside the file", "parent documented by file name only; Wayback copy of a defunct site; adds numbers, years, single letters and regex-escaped punctuation",
  "voyant_en", "taporware", "documented", "explicit statement", "stop.en.txt header: 'see http://taporware.mcmaster.ca/~taporware/cgi-bin/prototype/glasgowstoplist.txt'", "",
  "spacy", "glasgow", "similarity", "none", "no source statement (current file comment is '# Stop words'); igorbrigadir attributes it to Stone, Denis & Kwantes (2010), unverified; Jaccard 0.8812 with gensim and 0.86 with scikit-learn, both below 0.90", "no source statement; the Glasgow relation rests on overlap below the near-duplicate threshold; the 2010 attribution could not be checked",
  "gensim", "sklearn", "near_identity", "none", "no source statement in preprocessing.py; measured: strict superset of scikit-learn's Glasgow copy (all 318 words) plus 19 (Jaccard 0.9436); igorbrigadir's 'same as spaCy' is contradicted by measurement", "no source statement; the 19 additions are undocumented; the collection's claim about spaCy is wrong for the current sets",
  "fox_1989", "none", "root", "n/a", "RAKE FoxStoplist.txt header: '#From \"A stop list for general text\" Fox 1989'", "the root is RAKE's copy: 425 entries against the 421 reported in the paper; the paper itself was not consulted",
  "onix_lextek", "fox_1989", "near_identity", "none", "the lextek page does not credit Fox; measured: Fox without numbered and numbering (subset, Jaccard 0.9953)", "no source statement; the page states 429 words but lists 6 twice (423 unique)",
  "lsa_stopwords_en", "onix_lextek", "exact_identity", "none", "lsa help gives no source; normalised set identical to the ONIX list", "no source statement; identity with ONIX (hence the Fox relation) is measured only",
  "tidytext_onix", "onix_lextek", "documented", "explicit statement", "tidytext help cites lextek stopwords1.html as the onix source", "the ONIX-to-Fox link is measured only; the removal of the single letters b-z and 'so' is not documented",
  "qdap_onix", "onix_lextek", "documented", "explicit statement", "qdapDictionaries help Note: 'Reduced from the original 429 words to 404'", "the ONIX-to-Fox link is measured only",
  "pattern_clips", "tidytext_onix", "similarity", "none", "single-line file without header; measured: contains every entry of the tidytext/qdap ONIX list plus contraction fragments and about 165 words (Jaccard 0.7069)", "no source statement; containment of a Fox derivative is the only evidence; the 165 additions have no documented origin",
  "gate_kea", "none", "root", "n/a", "StopwordsEnglish.java header: 'Copyright (C) 2001 Eibe Frank' (KEA); no parent claimed", "unversioned live source-browser page",
  "corenlp_patterns", "none", "root", "n/a", "no header; kept as its own family because the content differs materially from every Snowball copy (nearest list DL4J, Jaccard 0.6842)", "no source statement; the separation from the Snowball family is a judgment, not a documented fact",
  "lingpipe", "none", "root", "n/a", "Javadoc: 'The built-in stoplist consists of the following words'; no parent claimed", "source taken from a third-party clone of LingPipe 4.1.0, confirmed against the alias-i Javadoc",
  "choi_c99", "none", "root", "n/a", "igorbrigadir table: stop list of Freddy Choi's C99 / TextTiling implementations; no parent claimed", "only a secondary copy exists; the original code is not machine-readable",
  "cook_1988", "none", "root", "n/a", "page text: list compiled as data for a parser of student English; no parent claimed", "Wayback copy; IPA weak forms removed by the parser; the igorbrigadir copy lacks 's",
  "qdap_function_words", "none", "root", "n/a", "qdapDictionaries help: function words from John and Muriel Higgins's ECLIPSE list plus qdap's contractions", "the Higgins list itself was not acquired and qdap's contraction additions are not itemised",
  "okapi_framework", "none", "root", "n/a", "file header: '# Stop words default list for English'; no parent claimed", "unpinned (bitbucket master, retrieved at run time); no source statement"
)
stopifnot(setequal(lineage$list_id, included_ids), all(lineage$immediate_parent %in% c("none", included_ids)))
basis_rank <- c(similarity = 1L, near_identity = 2L, exact_identity = 3L, documented = 4L, root = 5L)
lineage_chain <- function(id) {
  chain <- id; bases <- character(); cur <- id
  repeat { p <- lineage$immediate_parent[lineage$list_id == cur]; b <- lineage$link_basis[lineage$list_id == cur]
           if (p == "none") break; bases <- c(bases, b); chain <- c(chain, p); cur <- p; if (length(chain) > 10) stop("cycle at ", id) }
  list(chain = chain, bases = bases)
}
chain_info <- bind_rows(lapply(included_ids, function(id) {
  ci <- lineage_chain(id); root <- ci$chain[length(ci$chain)]
  chain_str <- if (length(ci$bases) == 0) id else paste0(paste0(ci$chain[-length(ci$chain)], " -[", ci$bases, "]-> ", collapse = ""), root)
  weakest <- if (length(ci$bases) == 0) "root" else names(basis_rank)[basis_rank == min(basis_rank[ci$bases])][1]
  undocumented <- if (length(ci$bases) == 0) "" else paste(sprintf("%s -> %s (%s)", ci$chain[-length(ci$chain)][ci$bases != "documented"], ci$chain[-1][ci$bases != "documented"], ci$bases[ci$bases != "documented"]), collapse = "; ")
  support <- case_when(weakest == "root" ~ "root list (no parent claimed)", weakest == "documented" ~ "documented lineage (every link to the root has a source statement)",
                       weakest == "exact_identity" ~ "exact identity without documentation on at least one link", weakest == "near_identity" ~ "near-identity (Jaccard >= 0.90) without documentation on at least one link",
                       TRUE ~ "measured similarity only (Jaccard < 0.90) on at least one link")
  judgment <- case_when(weakest %in% c("root", "documented") ~ "none", weakest == "exact_identity" ~ "low", weakest == "near_identity" ~ "medium", TRUE ~ "high")
  tibble(list_id = id, chain_root = root, chain_depth = length(ci$bases), chain_to_root = chain_str, weakest_link_basis = weakest, undocumented_links = undocumented,
         assignment_support = support, human_judgment_level = judgment, requires_human_judgment = judgment != "none")
}))
add_check("lineage", "every lineage chain ends at the root list of the list's delivered family", "", all(chain_info$chain_root == root_ids[match(family_of[chain_info$list_id], family_of[root_ids])]))

# Measured relations to the immediate parent and to the family root (from the delivered matrix).
measured_links <- bind_rows(lapply(included_ids, function(id) {
  p <- lineage$immediate_parent[lineage$list_id == id]; root <- chain_info$chain_root[chain_info$list_id == id]
  rp <- if (p == "none") NULL else relation(id, p); rr <- if (id == root) NULL else relation(id, root)
  tibble(list_id = id,
         jaccard_to_parent = if (is.null(rp)) NA_real_ else rp$jaccard, relationship_to_parent = if (is.null(rp)) "" else rp$relationship,
         n_only_in_list_vs_parent = if (is.null(rp)) NA_integer_ else length(rp$only_a), n_only_in_parent = if (is.null(rp)) NA_integer_ else length(rp$only_b),
         words_only_in_list_vs_parent = if (is.null(rp)) "" else collapse_words(rp$only_a), words_only_in_parent = if (is.null(rp)) "" else collapse_words(rp$only_b),
         jaccard_to_root = if (is.null(rr)) NA_real_ else rr$jaccard, relationship_to_root = if (is.null(rr)) "" else rr$relationship,
         n_only_in_list_vs_root = if (is.null(rr)) NA_integer_ else length(rr$only_a), n_only_in_root = if (is.null(rr)) NA_integer_ else length(rr$only_b))
}))

# ---- 1. Included-list audit ----
log_line("included-list audit")
dup_membership <- duplicate_groups |> mutate(list_id = stri_split_fixed(lists, " ")) |> unnest(list_id) |>
  group_by(distinct_set_id) |> mutate(identical_to = map_chr(list_id, function(x) paste(setdiff(list_id, x), collapse = " "))) |> ungroup() |>
  transmute(list_id, exact_duplicate_group = distinct_set_id, n_identical_lists = n_lists, identical_to)
near_partners <- near_dups |> filter(relationship != "identical") |> select(list_a, list_b, jaccard, relationship) |>
  bind_rows(near_dups |> filter(relationship != "identical") |> transmute(list_a = list_b, list_b = list_a, jaccard, relationship = recode(relationship, a_subset_of_b = "b_subset_of_a", b_subset_of_a = "a_subset_of_b"))) |>
  group_by(list_a) |> arrange(desc(jaccard), list_b, .by_group = TRUE) |>
  summarise(n_near_duplicates = n(), max_near_duplicate_jaccard = max(jaccard), near_duplicate_partners = paste(sprintf("%s (%.4f, %s)", list_b, jaccard, relationship), collapse = "; "), .groups = "drop") |> rename(list_id = list_a)
included_list_audit <- tibble(list_id = included_ids) |>
  left_join(prov_inc |> select(list_id, list_name, inclusion_status, inclusion_reason, intended_use, upstream_source, url, pin, version_date, documented_parent, family, is_family_root, evidence), by = "list_id") |>
  mutate(status = ifelse(inclusion_status == "included", "firm", "provisional"), url = coalesce(url, "")) |>
  left_join(list_summary |> select(list_id, n_raw_entries, n_unique_entries, n_duplicates_within, n_word, n_apostrophe, n_numeric, n_punctuation_or_symbol, n_other_characters, n_hyphenated), by = "list_id") |>
  left_join(dup_membership, by = "list_id") |> left_join(near_partners, by = "list_id") |>
  mutate(exact_duplicate_status = ifelse(is.na(exact_duplicate_group), "no identical list", sprintf("identical set to %s (group %d)", identical_to, exact_duplicate_group)),
         near_duplicate_status = ifelse(is.na(n_near_duplicates), sprintf("no non-identical list at Jaccard >= %.2f", near_duplicate_jaccard), near_duplicate_partners)) |>
  left_join(family_review |> select(list_id, within_family_max_jaccard, most_similar_in_family, outside_family_max_jaccard, most_similar_outside_family, outside_family, review_flag), by = "list_id") |>
  left_join(lineage, by = "list_id") |> left_join(chain_info, by = "list_id") |> left_join(measured_links, by = "list_id") |>
  mutate(family_assignment_basis = case_when(link_basis == "root" ~ sprintf("root of the %s family: %s", family, link_evidence),
                                             TRUE ~ sprintf("parent %s [%s; %s]: %s", immediate_parent, link_basis, documentation_strength, link_evidence)),
         documented_parent_list = ifelse(link_basis == "documented", immediate_parent, "none documented"),
         measured_parent_list = ifelse(link_basis %in% c("exact_identity", "near_identity", "similarity"), immediate_parent, "")) |>
  select(list_id, list_name, status, inclusion_status, inclusion_reason, intended_use, upstream_source, url, pin, version_date,
         documented_parent_field = documented_parent, documented_parent_list, measured_parent_list, link_basis, documentation_strength, link_evidence,
         n_raw_entries, n_unique_entries, n_duplicates_within, n_word, n_apostrophe, n_numeric, n_punctuation_or_symbol, n_other_characters, n_hyphenated,
         exact_duplicate_status, exact_duplicate_group, n_identical_lists, near_duplicate_status, max_near_duplicate_jaccard,
         family, is_family_root, jaccard_to_parent, relationship_to_parent, jaccard_to_root, relationship_to_root, within_family_max_jaccard, most_similar_in_family,
         outside_family_max_jaccard, most_similar_outside_family, outside_family, delivered_review_flag = review_flag,
         family_assignment_basis, chain_to_root, weakest_link_basis, undocumented_links, assignment_support, human_judgment_level, requires_human_judgment, unresolved_issues, evidence)
add_check("audit", "included-list audit has one row per included list", nrow(included_list_audit), nrow(included_list_audit) == 45 && !anyNA(included_list_audit$n_unique_entries))

# ---- 2. Family / lineage audit ----
log_line("family audit")
family_lineage_audit <- included_list_audit |>
  transmute(family, family_root = chain_info$chain_root[match(list_id, chain_info$list_id)], list_id, status, is_family_root, n_unique_entries,
            immediate_parent = lineage$immediate_parent[match(list_id, lineage$list_id)], link_basis, documentation_strength, link_evidence,
            jaccard_to_parent, relationship_to_parent, words_only_in_list_vs_parent = measured_links$words_only_in_list_vs_parent[match(list_id, measured_links$list_id)],
            words_only_in_parent = measured_links$words_only_in_parent[match(list_id, measured_links$list_id)],
            jaccard_to_root, relationship_to_root, within_family_max_jaccard, most_similar_in_family, outside_family_max_jaccard, most_similar_outside_family, outside_family,
            chain_to_root, weakest_link_basis, undocumented_links, assignment_support, human_judgment_level, requires_human_judgment, delivered_review_flag, unresolved_issues) |>
  arrange(family, desc(is_family_root), chain_to_root, list_id)
family_lineage_summary <- family_lineage_audit |> group_by(family, family_root) |>
  summarise(n_members = n(), n_firm = sum(status == "firm"), n_provisional = sum(status == "provisional"), members = paste(list_id, collapse = " "),
            n_documented_chain = sum(weakest_link_basis == "documented"), n_exact_identity_only = sum(weakest_link_basis == "exact_identity"),
            n_near_identity_only = sum(weakest_link_basis == "near_identity"), n_similarity_only = sum(weakest_link_basis == "similarity"),
            members_needing_judgment = paste(list_id[requires_human_judgment], collapse = " "),
            members_similarity_only = paste(list_id[weakest_link_basis == "similarity"], collapse = " "),
            min_jaccard_to_root = suppressWarnings(min(jaccard_to_root, na.rm = TRUE)), min_within_family_max_jaccard = suppressWarnings(min(within_family_max_jaccard, na.rm = TRUE)),
            max_outside_family_jaccard = max(outside_family_max_jaccard), .groups = "drop") |>
  mutate(across(c(min_jaccard_to_root, min_within_family_max_jaccard), ~ ifelse(is.finite(.x), .x, NA_real_))) |>
  left_join(family_definitions |> select(family, delivered_documented_parent_of_root = documented_parent, delivered_min_within_family_jaccard = min_within_family_jaccard, delivered_n_review_flags = n_review_flags), by = "family") |>
  arrange(desc(n_members), family)
# Pairwise values relevant to the non-documented assignments and to the flagged singleton families.
pair_register <- tribble(
  ~list_a, ~list_b, ~why,
  "spacy", "glasgow", "spaCy placed in the Glasgow family on overlap only",
  "spacy", "sklearn", "spaCy vs the documented Glasgow copy in scikit-learn",
  "spacy", "gensim", "spaCy's most similar list in its family; igorbrigadir claims gensim is 'same as spaCy'",
  "gensim", "sklearn", "gensim placed in the Glasgow family on containment of scikit-learn's copy",
  "gensim", "glasgow", "gensim vs the Glasgow root",
  "onix_lextek", "fox_1989", "ONIX placed in the Fox family on near-identity (page does not credit Fox)",
  "lsa_stopwords_en", "fox_1989", "lsa placed in the Fox family via identity with ONIX",
  "lsa_stopwords_en", "onix_lextek", "lsa identical to ONIX",
  "tidytext_onix", "onix_lextek", "documented ONIX derivative (letters b-z and 'so' removed)",
  "pattern_clips", "tidytext_onix", "Pattern placed in the Fox family on containment of the tidytext/qdap ONIX list",
  "pattern_clips", "onix_lextek", "Pattern vs the ONIX page",
  "pattern_clips", "fox_1989", "Pattern vs the Fox root",
  "bow_rainbow", "smart_lextek", "Bow placed in the SMART family on near-identity (no statement in stopwords.c)",
  "mallet", "bow_rainbow", "MALLET identical to Bow (documentation names no source)",
  "yake", "smart_lextek", "YAKE placed in the SMART family on near-identity",
  "buckley_salton_qdap", "smart_lextek", "documented SMART derivative below the near-duplicate threshold",
  "dl4j", "stopwords_pkg_snowball", "DL4J placed in the Snowball family on near-identity",
  "python_stop_words_2014", "snowball_original", "Python stop-words 2014 identical to Snowball (no statement)",
  "spark_ml", "nltk_english", "Spark's documented parent",
  "spark_ml", "stopwords_pkg_snowball", "Spark's most similar list",
  "dkpro", "nltk_english", "DKPro's documented parent (2013 copy)",
  "corenlp_patterns", "dl4j", "singleton family CoreNLP vs its nearest list (Snowball family)",
  "corenlp_patterns", "snowball_original", "singleton family CoreNLP vs the Snowball root",
  "choi_c99", "spacy", "singleton family C99 vs its nearest list (Glasgow family)",
  "cook_1988", "spark_ml", "singleton family Cook vs its nearest list (Snowball family)",
  "qdap_function_words", "choi_c99", "singleton family Higgins vs its nearest list",
  "gate_kea", "choi_c99", "singleton family KEA vs its nearest list",
  "lingpipe", "dkpro", "singleton family LingPipe vs its nearest list",
  "okapi_framework", "tidytext_onix", "singleton family Okapi vs its nearest list"
)
family_lineage_pairs <- bind_rows(lapply(seq_len(nrow(pair_register)), function(i) {
  a <- pair_register$list_a[i]; b <- pair_register$list_b[i]; r <- relation(a, b)
  tibble(list_a = a, list_b = b, family_a = family_of[a], family_b = family_of[b], same_family = family_of[a] == family_of[b], status_a = ifelse(a %in% firm_ids, "firm", "provisional"), status_b = ifelse(b %in% firm_ids, "firm", "provisional"),
         jaccard = r$jaccard, relationship = r$relationship, size_a = r$n_a, size_b = r$n_b, intersection = r$intersection, n_only_in_a = length(r$only_a), n_only_in_b = length(r$only_b),
         words_only_in_a = collapse_words(r$only_a), words_only_in_b = collapse_words(r$only_b),
         link_basis_of_a = lineage$link_basis[lineage$list_id == a], why_relevant = pair_register$why[i])
}))
add_check("audit", "pair register values agree with jaccard_pairs_all.csv", "", all(vapply(seq_len(nrow(family_lineage_pairs)), function(i) {
  a <- family_lineage_pairs$list_a[i]; b <- family_lineage_pairs$list_b[i]
  abs(pairs_all$jaccard[(pairs_all$list_a == a & pairs_all$list_b == b) | (pairs_all$list_a == b & pairs_all$list_b == a)] - family_lineage_pairs$jaccard[i]) < 1e-9 }, logical(1))))

# ---- 3. Consensus-variable audit ----
log_line("consensus-variable audit")
src_lines <- readLines(source_script_path, warn = FALSE)
code_line <- function(pattern) { k <- grep(pattern, src_lines, fixed = TRUE)[1]; sprintf("line %d: %s", k, stri_trim_both(src_lines[k])) }
n_dup_groups <- nrow(duplicate_groups)
consensus_variable_definitions <- tribble(
  ~variable, ~unit_counted, ~exact_rule, ~source_code, ~copies_counted_separately, ~provisional_lists_contribute, ~used_for_tiers, ~notes,
  "n_lists", "included named list (45: 32 firm + 13 provisional)", "row sum of the 0/1 membership matrix over all 45 included lists; every list counts 1 whether or not it is an exact copy of another list", code_line("n_lists = as.integer(rowSums(Mx))"), "yes", "yes", "primary tiers (thresholds 29 / 20 / 7)", sprintf("the 45 lists form %d identical-set groups (%d lists) plus %d singletons", n_dup_groups, sum(duplicate_groups$n_lists), 45 - sum(duplicate_groups$n_lists)),
  "n_distinct_lists", "distinct normalised set (30)", "row sum over one representative list per identical-set group (the first list of the group in list order); a word in all six SMART copies counts 1", code_line("n_distinct_lists = as.integer(rowSums(Mx[, distinct_rep$list_id, drop = FALSE]))"), "no (identical sets collapsed; near-identical sets still count separately)", "yes", "no", "identical sets are detected by a hash of the sorted normalised entries (line 542); the representative choice does not affect the sum",
  "n_families_any", "family (11)", "number of families with at least one member list containing the word", code_line("n_families_any = as.integer(rowSums(fam_any))"), "no (a family counts 1 however many members contain the word)", "yes (four families consist of one provisional list each: c99, cook, higgins, okapi_framework)", "family-aware tiers (thresholds 9 / 6 / 3)", "fam_any[, f] = rowSums(Mx[, members of f]) > 0 (line 548)",
  "n_families_all", "family (11)", "number of families in which every member list contains the word", code_line("n_families_all = as.integer(rowSums(fam_all))"), "no", "yes (a provisional member missing the word removes its whole family from the count, e.g. dkpro for Snowball, yake for SMART)", "no", "fam_all[, f] = rowSums(Mx[, members of f]) == number of members of f (line 549); singleton families give n_families_all = n_families_any",
  "n_family_roots", "root list (11)", "row sum over the 11 family root lists only", code_line("n_family_roots = as.integer(rowSums(Mx[, root_ids, drop = FALSE]))"), "no (copies ignored entirely)", "yes (four roots are provisional lists: choi_c99, cook_1988, qdap_function_words, okapi_framework)", "no", "roots: snowball_original, smart_lextek, glasgow, fox_1989, gate_kea, corenlp_patterns, lingpipe, choi_c99, cook_1988, qdap_function_words, okapi_framework"
)
ws <- master |> select(rank, entry, entry_class, n_lists, n_distinct_lists, n_families_any, n_families_all, n_family_roots, tier_primary, tier_family, families_any, families_all) |>
  mutate(n_firm_lists = n_firm_lists, n_provisional_lists = n_prov_lists, family_coverage = unname(family_coverage[entry]), lists_containing = unname(lists_containing[entry]), roots_containing = unname(roots_containing[entry]))
pick <- function(df, why, n = 1L) df |> arrange(desc(n_lists), entry) |> slice_head(n = n) |> mutate(why_chosen = why)
spread <- ws |> group_by(n_distinct_lists) |> filter(n() > 1) |> summarise(lo = min(n_lists), hi = max(n_lists), .groups = "drop") |> mutate(d = hi - lo) |> arrange(desc(d), desc(n_distinct_lists)) |> slice(1)
gap <- ws |> mutate(g = n_families_any - n_families_all) |> arrange(desc(g), desc(n_lists), entry) |> slice(1)
examples <- bind_rows(
  pick(ws |> filter(n_lists == 45), "in every included list: every measure at its maximum"),
  pick(ws |> filter(n_families_any == 1, families_any == "smart"), "SMART family only: 12 raw lists collapse to a few distinct sets, one family, one root"),
  pick(ws |> filter(n_families_any == 1, families_any == "snowball"), "Snowball family only"),
  pick(ws |> filter(n_families_any == 1, families_any == "glasgow"), "Glasgow family only"),
  pick(ws |> filter(n_families_any == 1, families_any == "fox"), "Fox family only"),
  pick(ws |> filter(n_distinct_lists == spread$n_distinct_lists, n_lists == spread$hi), sprintf("same n_distinct_lists (%d) as the next example but the highest n_lists: copies inflate the raw count", spread$n_distinct_lists)),
  pick(ws |> filter(n_distinct_lists == spread$n_distinct_lists, n_lists == spread$lo), sprintf("same n_distinct_lists (%d) as the previous example but the lowest n_lists", spread$n_distinct_lists)),
  pick(ws |> filter(entry == gap$entry), "largest gap between n_families_any and n_families_all: families only partly covered"),
  pick(ws |> filter(n_family_roots == 0, n_lists >= 2), "in no root list: the raw count comes only from derived lists"),
  pick(ws |> filter(n_family_roots >= 1, n_lists == n_family_roots, n_lists >= 1), "only in root lists: n_families_all is 0 for every family that has copies"),
  pick(ws |> filter(n_firm_lists == 0), "only in provisional lists: disappears from the 32-list calculation"),
  pick(ws |> mutate(r = n_families_any / n_lists) |> filter(n_lists >= 3) |> arrange(desc(r), desc(n_lists), entry) |> slice(1) |> select(-r), "many families for few lists: each list is its own family"),
  pick(ws |> filter(n_lists >= 10) |> mutate(r = n_families_any / n_lists) |> arrange(r, desc(n_lists), entry) |> slice(1) |> select(-r), "many lists for few families: the family count corrects the copy inflation"),
  pick(ws |> filter(entry %in% c("don't", "dont", "don")), "one contraction in three spellings: normalisation keeps them as distinct entries", n = 3L)
) |> distinct(entry, .keep_all = TRUE)
consensus_variable_examples <- examples |> mutate(example_no = row_number(), .before = 1) |>
  mutate(distinct_set_representatives_containing = vapply(entry, function(w) paste(s45$rep_ids[Mx[w, s45$rep_ids] == 1L], collapse = " "), character(1))) |>
  select(example_no, entry, entry_class, why_chosen, n_lists, n_distinct_lists, n_families_any, n_families_all, n_family_roots, tier_primary, tier_family, rank,
         n_firm_lists, n_provisional_lists, lists_containing, distinct_set_representatives_containing, family_coverage, families_any, families_all, roots_containing)
set_group_of <- setNames(match(s45$set_hash, unique(s45$set_hash)), included_ids)   # identical-set group number in list order
consensus_variable_examples_long <- bind_rows(lapply(consensus_variable_examples$entry, function(w) {
  ids <- included_ids[Mx[w, ] == 1L]
  tibble(entry = w, list_id = ids, family = family_of[ids], status = ifelse(ids %in% firm_ids, "firm", "provisional"), is_family_root = ids %in% root_ids,
         identical_set_group = set_group_of[ids], is_distinct_set_representative = ids %in% s45$rep_ids, family_size = as.integer(fam_size[family_of[ids]]))
})) |> group_by(entry, family) |> mutate(family_members_containing = n(), family_fully_covered = family_members_containing == family_size) |> ungroup()
provisional_contribution <- tribble(
  ~measure, ~provisional_lists_included, ~how,
  "n_lists", TRUE, "all 45 included lists enter the matrix (included_ids = inclusion_status in included, included_provisional; line 533)",
  "n_distinct_lists", TRUE, "the identical-set groups are formed over all 45 lists; provisional copies collapse into their groups (python_stop_words_2014 with Snowball; lsa with ONIX; tidytext_onix with qdap_onix)",
  "n_families_any", TRUE, "families are defined over all 45 lists; c99, cook, higgins and okapi_framework exist only because of provisional lists",
  "n_families_all", TRUE, "a provisional member that lacks a word removes its family from the count",
  "n_family_roots", TRUE, "four of the 11 roots (choi_c99, cook_1988, qdap_function_words, okapi_framework) are provisional lists",
  "tier_primary", TRUE, "computed from n_lists over all 45 lists (thresholds chosen on that distribution)",
  "tier_family", TRUE, "computed from n_families_any over all 11 families"
)

# ---- 4. Firm-versus-provisional influence ----
log_line("firm vs provisional")
s32 <- consensus_stats(firm_ids); st32 <- s32$stats
tp32 <- choose_tiers(st32$n_lists); tf32 <- choose_tiers(st32$n_families_any)
n_words <- nrow(master)
# Size-matched tiers: thresholds cut only at count values, chosen so that each cumulative size is closest to the
# delivered size (155/310/627 or 127/253/454). This isolates re-ordering from the shrinkage of the re-derived tiers.
size_matched_thresholds <- function(cnt, target_sizes) {
  vals <- sort(unique(cnt[cnt >= 2]), decreasing = TRUE); th <- integer(0)
  for (S in target_sizes) { cand <- vals[if (length(th)) vals < th[length(th)] else TRUE]; if (!length(cand)) return(rep(NA_integer_, 3)); cum <- vapply(cand, function(t) sum(cnt >= t), integer(1)); th <- c(th, cand[which.min(abs(cum - S))]) }
  th
}
tp32_size <- size_matched_thresholds(st32$n_lists, tp45$sizes); tf32_size <- size_matched_thresholds(st32$n_families_any, tf45$sizes)
cmp <- master |> select(entry, entry_class, rank_45 = rank, n_lists_45 = n_lists, n_distinct_lists_45 = n_distinct_lists, n_families_any_45 = n_families_any, n_families_all_45 = n_families_all, n_family_roots_45 = n_family_roots, tier_primary_45 = tier_primary, tier_family_45 = tier_family, families_any_45 = families_any) |>
  left_join(st32 |> rename(n_lists_32 = n_lists, n_distinct_lists_32 = n_distinct_lists, n_families_any_32 = n_families_any, n_families_all_32 = n_families_all, n_family_roots_32 = n_family_roots, families_any_32 = families_any) |> select(-families_all), by = "entry") |>
  mutate(provisional_only = n_lists_32 == 0L, n_provisional_lists = n_lists_45 - n_lists_32, provisional_lists_containing = unname(provisional_lists_containing[entry]),
         rank_min_45 = rank(-n_lists_45, ties.method = "min"),
         rank_min_32 = ifelse(provisional_only, NA_integer_, rank(-n_lists_32, ties.method = "min")),          # over the entries that survive (count >= 1)
         rank_pct_45 = round(rank_min_45 / n_words, 4), rank_pct_32 = round(rank_min_32 / sum(!provisional_only), 4), rank_pct_change = round(rank_pct_32 - rank_pct_45, 4),
         tier_primary_32_logic = tier_of(n_lists_32, tp32$thresholds), tier_family_32_logic = tier_of(n_families_any_32, tf32$thresholds),
         tier_primary_32_size = tier_of(n_lists_32, tp32_size), tier_family_32_size = tier_of(n_families_any_32, tf32_size),
         tier_primary_changed_logic = coalesce(tier_primary_45, 0L) != coalesce(tier_primary_32_logic, 0L), tier_family_changed_logic = coalesce(tier_family_45, 0L) != coalesce(tier_family_32_logic, 0L),
         tier_primary_changed_size = coalesce(tier_primary_45, 0L) != coalesce(tier_primary_32_size, 0L), tier_family_changed_size = coalesce(tier_family_45, 0L) != coalesce(tier_family_32_size, 0L),
         large_rank_move = !provisional_only & abs(rank_pct_change) >= large_rank_move_share,
         material_change = provisional_only | tier_primary_changed_size | tier_family_changed_size | large_rank_move) |>
  mutate(change_kind = pmap_chr(list(provisional_only, tier_primary_changed_size, tier_family_changed_size, large_rank_move, tier_primary_changed_logic, tier_family_changed_logic),
                                function(a, b, c, d, e, f) paste(c("only in provisional lists", "primary tier (size-matched)", "family-aware tier (size-matched)", "rank percentile moves >= 5 points", "primary tier (re-derived thresholds)", "family-aware tier (re-derived thresholds)")[c(a, b, c, d, e, f)], collapse = "; "))) |>
  arrange(rank_45)
cmp$rank_32 <- order(order(-cmp$n_lists_32, cmp$entry, method = "radix"))   # row position when sorted by the 32-list count then entry
firm_vs_provisional_words <- cmp |> select(entry, entry_class, n_lists_45, n_lists_32, n_provisional_lists, provisional_lists_containing, n_distinct_lists_45, n_distinct_lists_32,
                                           n_families_any_45, n_families_any_32, n_families_all_45, n_families_all_32, n_family_roots_45, n_family_roots_32,
                                           rank_45, rank_32, rank_min_45, rank_min_32, rank_pct_45, rank_pct_32, rank_pct_change,
                                           tier_primary_45, tier_primary_32_size, tier_primary_32_logic, tier_family_45, tier_family_32_size, tier_family_32_logic,
                                           provisional_only, tier_primary_changed_size, tier_family_changed_size, large_rank_move, tier_primary_changed_logic, tier_family_changed_logic, material_change, change_kind, families_any_45, families_any_32)
firm_vs_provisional_changes <- firm_vs_provisional_words |> filter(material_change) |> arrange(desc(provisional_only), desc(tier_primary_changed_size | tier_family_changed_size), desc(abs(rank_pct_change)), rank_45)
tier_sizes_tbl <- function(th, sizes, obj, ordering, label, note) tibble(calculation = label, ordering = ordering, tier = 1:3, threshold = th, cumulative_size = sizes, size_ratio_to_tier1 = round(sizes / sizes[1], 3), objective = round(obj, 5), note = note)
firm_vs_provisional_tier_sizes <- bind_rows(
  tier_sizes_tbl(tp45$thresholds, tp45$sizes, tp45$obj, "primary (n_lists)", "45 lists (delivered)", "delivered thresholds"),
  tier_sizes_tbl(tf45$thresholds, tf45$sizes, tf45$obj, "family-aware (n_families_any)", "45 lists (delivered)", "delivered thresholds"),
  tier_sizes_tbl(tp32$thresholds, tp32$sizes, tp32$obj, "primary (n_lists)", "32 firm lists, re-derived thresholds", "same construction logic re-run on the 32-list counts"),
  tier_sizes_tbl(tf32$thresholds, tf32$sizes, tf32$obj, "family-aware (n_families_any)", "32 firm lists, re-derived thresholds", "same construction logic re-run on the 7 remaining families"),
  tier_sizes_tbl(tp32_size, sizes_at(st32$n_lists, tp32_size), tier_obj(sizes_at(st32$n_lists, tp32_size)), "primary (n_lists)", "32 firm lists, size-matched thresholds", "cut at the count values whose cumulative sizes are closest to 155 / 310 / 627"),
  tier_sizes_tbl(tf32_size, sizes_at(st32$n_families_any, tf32_size), tier_obj(sizes_at(st32$n_families_any, tf32_size)), "family-aware (n_families_any)", "32 firm lists, size-matched thresholds", "cut at the count values whose cumulative sizes are closest to 127 / 253 / 454"))
transitions <- function(a, b, label) tibble(system = label, tier_45 = coalesce(as.character(a), "none"), tier_32 = coalesce(as.character(b), "none"), entry = cmp$entry, rank_45 = cmp$rank_45) |>
  group_by(system, tier_45, tier_32) |> summarise(n_words = n(), example_words = paste(head(entry[order(rank_45)], 25), collapse = " "), .groups = "drop") |> mutate(changed = tier_45 != tier_32)
firm_vs_provisional_tier_transitions <- bind_rows(
  transitions(cmp$tier_primary_45, cmp$tier_primary_32_size, "primary (n_lists), size-matched 32-list cut"), transitions(cmp$tier_family_45, cmp$tier_family_32_size, "family-aware (n_families_any), size-matched 32-list cut"),
  transitions(cmp$tier_primary_45, cmp$tier_primary_32_logic, "primary (n_lists), re-derived 32-list thresholds"), transitions(cmp$tier_family_45, cmp$tier_family_32_logic, "family-aware (n_families_any), re-derived 32-list thresholds")) |>
  arrange(system, tier_45, tier_32)
rho_all <- cor(cmp$n_lists_45, cmp$n_lists_32, method = "spearman"); rho_surv <- cor(cmp$n_lists_45[!cmp$provisional_only], cmp$n_lists_32[!cmp$provisional_only], method = "spearman")
rho_fam <- cor(cmp$n_families_any_45[!cmp$provisional_only], cmp$n_families_any_32[!cmp$provisional_only], method = "spearman")
q <- quantile(abs(cmp$rank_pct_change[!cmp$provisional_only]), c(0.5, 0.9, 0.99, 1))
firm_vs_provisional_summary <- tribble(
  ~item, ~value_45_lists, ~value_32_firm_lists, ~note,
  "lists in the calculation", "45", "32", "13 provisional lists removed: dkpro, dl4j, python_stop_words_2014, yake, lsa_stopwords_en, pattern_clips, onix_lextek, tidytext_onix, qdap_onix, choi_c99, cook_1988, qdap_function_words, okapi_framework",
  "families present", as.character(length(fam_names)), as.character(length(s32$families)), paste("families lost:", paste(setdiff(fam_names, s32$families), collapse = ", ")),
  "root lists present", as.character(length(root_ids)), as.character(length(s32$roots)), paste("roots lost:", paste(setdiff(root_ids, s32$roots), collapse = ", ")),
  "distinct normalised sets", as.character(length(s45$rep_ids)), as.character(length(s32$rep_ids)), "identical-set groups recomputed over the firm lists only",
  "entries with count >= 1", as.character(n_words), as.character(sum(!cmp$provisional_only)), sprintf("%d entries occur only in provisional lists and leave the union", sum(cmp$provisional_only)),
  "entries with unchanged n_lists", "", as.character(sum(cmp$n_lists_45 == cmp$n_lists_32)), "entries in no provisional list",
  "Spearman correlation of n_lists (45 vs 32)", "", sprintf("%.4f (all %d entries); %.4f (the %d surviving entries)", rho_all, n_words, rho_surv, sum(!cmp$provisional_only)), "rank agreement of the two raw counts",
  "Spearman correlation of n_families_any (45 vs 32)", "", sprintf("%.4f (surviving entries)", rho_fam), "",
  "primary tiers, re-derived thresholds (cumulative 1 / 2 / 3)", paste(tp45$sizes, collapse = " / "), paste(tp32$sizes, collapse = " / "), sprintf("n_lists >= %s versus n_lists >= %s: the same construction logic on 32 lists gives much smaller tiers (fewer count values, same 1:2:4 objective)", paste(tp45$thresholds, collapse = " / "), paste(tp32$thresholds, collapse = " / ")),
  "family-aware tiers, re-derived thresholds (cumulative 1 / 2 / 3)", paste(tf45$sizes, collapse = " / "), paste(tf32$sizes, collapse = " / "), sprintf("n_families_any >= %s versus >= %s over %d families", paste(tf45$thresholds, collapse = " / "), paste(tf32$thresholds, collapse = " / "), length(s32$families)),
  "primary tiers, size-matched thresholds (cumulative 1 / 2 / 3)", paste(tp45$sizes, collapse = " / "), paste(sizes_at(st32$n_lists, tp32_size), collapse = " / "), sprintf("n_lists >= %s on the 32-list counts, cut at the count values closest to the delivered sizes", paste(tp32_size, collapse = " / ")),
  "family-aware tiers, size-matched thresholds (cumulative 1 / 2 / 3)", paste(tf45$sizes, collapse = " / "), paste(sizes_at(st32$n_families_any, tf32_size), collapse = " / "), sprintf("n_families_any >= %s on the 32-list counts", paste(tf32_size, collapse = " / ")),
  "entries whose primary tier changes (size-matched cut)", "", as.character(sum(cmp$tier_primary_changed_size)), "membership in tiers of about the delivered size, so this counts re-ordering rather than shrinkage",
  "entries whose family-aware tier changes (size-matched cut)", "", as.character(sum(cmp$tier_family_changed_size)), "",
  "entries whose primary tier changes (re-derived thresholds)", "", as.character(sum(cmp$tier_primary_changed_logic)), "dominated by the smaller tiers; see firm_vs_provisional_tier_transitions.csv",
  "entries whose family-aware tier changes (re-derived thresholds)", "", as.character(sum(cmp$tier_family_changed_logic)), "",
  "entries whose rank percentile moves by >= 5 points", "", as.character(sum(cmp$large_rank_move)), "min-rank divided by the number of ranked entries (1352 versus the surviving entries); provisional-only entries excluded",
  "abs rank-percentile change: median / p90 / p99 / max (surviving entries)", "", paste(sprintf("%.3f", q), collapse = " / "), "",
  "entries with any material change", "", as.character(sum(cmp$material_change)), "leaves the union, or a size-matched tier change, or a rank-percentile move of at least 5 points"
)

# ---- 5. Tier-boundary audit ----
log_line("tier boundaries")
order_primary <- order(-master$n_lists, master$entry, method = "radix")
order_family  <- order(-master$n_families_any, -master$n_lists, master$entry, method = "radix")
boundary_tables <- function(cnt, th, sizes, obj, ordering, var, label) {
  vals <- sort(unique(cnt), decreasing = TRUE)
  stair <- tibble(system = label, variable = var, count_value = vals, n_words = vapply(vals, function(v) sum(cnt == v), integer(1))) |>
    mutate(cumulative_words = cumsum(n_words), is_chosen_threshold = count_value %in% th, tier = ifelse(is_chosen_threshold, match(count_value, th), NA_integer_),
           ratio_to_tier1 = round(cumulative_words / sizes[1], 3), target_size_for_ratio_1_2_4 = ifelse(is_chosen_threshold, sizes[1] * tier_ratio[tier], NA_real_),
           eligible_as_threshold = count_value >= 2, cumulative_below_this_value = cumulative_words - n_words)
  summ <- bind_rows(lapply(1:3, function(k) {
    t <- th[k]; g <- sum(cnt == t); lower <- vals[vals < t & vals >= 2]; upper <- vals[vals > t]
    down <- if (length(lower)) max(lower) else NA_integer_; up <- if (length(upper)) min(upper) else NA_integer_
    alt <- function(newt) { th2 <- th; th2[k] <- newt; if (any(diff(th2) >= 0)) return(list(sizes = NA, obj = NA)); N <- sizes_at(cnt, th2); list(sizes = N, obj = tier_obj(N)) }
    a_down <- if (is.na(down)) list(sizes = NA, obj = NA) else alt(down); a_up <- if (is.na(up)) list(sizes = NA, obj = NA) else alt(up)
    ideal <- if (k == 1) NA_real_ else sizes[1] * tier_ratio[k]
    tibble(system = label, tier = k, variable = var, cutoff_rule = sprintf("%s >= %d", var, t), threshold = t, cumulative_size = sizes[k], size_ratio_to_tier1 = round(sizes[k] / sizes[1], 3),
           target_size_1_2_4 = ideal, tied_group_at_threshold = g, cumulative_before_tied_group = sizes[k] - g,
           ideal_falls_inside_tied_group = if (is.na(ideal)) NA else (ideal > sizes[k] - g & ideal < sizes[k]),
           next_lower_value = down, tied_group_at_next_lower_value = if (is.na(down)) NA_integer_ else sum(cnt == down), cumulative_if_cut_one_value_lower = if (is.na(down)) NA_integer_ else sum(cnt >= down),
           next_higher_value = up, cumulative_if_cut_one_value_higher = if (is.na(up)) NA_integer_ else sum(cnt >= up),
           objective_chosen = round(obj, 5), objective_if_one_value_lower = round(a_down$obj, 5), sizes_if_one_value_lower = paste(a_down$sizes, collapse = "/"),
           objective_if_one_value_higher = round(a_up$obj, 5), sizes_if_one_value_higher = paste(a_up$sizes, collapse = "/"),
           ideal_falls_inside_next_lower_group = if (is.na(ideal) || is.na(down)) NA else (ideal > sizes[k] & ideal < sum(cnt >= down)),
           nearest_achievable_sizes = if (is.na(ideal)) "" else sprintf("%d (cut at >= %d) or %d (cut at >= %d): distance to target %+d or %+d", sizes[k], t, sum(cnt >= down), down, sizes[k] - as.integer(ideal), sum(cnt >= down) - as.integer(ideal)),
           effect_of_no_split_rule = if (is.na(ideal)) sprintf("anchor tier: %d words at %s >= %d; the %d words tied at %d enter together", sizes[k], var, t, g, t) else
             sprintf("the cut can only fall between count values: the %d words at %s = %d enter together (cumulative %d -> %d) and the next %d words at %s = %d would enter together too (-> %d); the 1:2:4 target %d is %s", g, var, t, sizes[k] - g, sizes[k], sum(cnt == down), var, down, sum(cnt >= down), as.integer(ideal),
                     ifelse(ideal > sizes[k] - g & ideal < sizes[k], "inside the tied group at the threshold", ifelse(ideal > sizes[k] & ideal < sum(cnt >= down), "inside the next lower tied group, so it cannot be reached", "outside both groups"))))
  }))
  ord <- ordering; pos <- seq_along(ord)
  words <- bind_rows(lapply(1:3, function(k) {
    t <- th[k]; cut <- sizes[k]   # positions 1..cut are inside the tier
    above <- ord[max(1, cut - boundary_window + 1):cut]; below <- ord[(cut + 1):min(length(ord), cut + boundary_window)]
    bind_rows(tibble(side = "above (last words inside the tier)", position = max(1, cut - boundary_window + 1):cut, idx = above),
              tibble(side = "below (first words outside the tier)", position = (cut + 1):min(length(ord), cut + boundary_window), idx = below)) |>
      mutate(system = label, tier = k, boundary = sprintf("%s >= %d", var, t), .before = 1)
  })) |> mutate(entry = master$entry[idx], entry_class = master$entry_class[idx], n_lists = master$n_lists[idx], n_distinct_lists = master$n_distinct_lists[idx], n_families_any = master$n_families_any[idx],
                n_families_all = master$n_families_all[idx], n_family_roots = master$n_family_roots[idx], n_firm_lists = n_firm_lists[idx], n_provisional_lists = n_prov_lists[idx],
                family_coverage = unname(family_coverage[entry]), lists_containing = unname(lists_containing[entry])) |> select(-idx)
  tied <- bind_rows(lapply(1:3, function(k) {
    t <- th[k]; lower <- vals[vals < t]; down <- if (length(lower)) max(lower) else NA_integer_
    bind_rows(tibble(group = "at threshold (inside the tier)", count_value = t, idx = ord[cnt[ord] == t]),
              if (!is.na(down)) tibble(group = "next lower value (outside the tier)", count_value = down, idx = ord[cnt[ord] == down]) else tibble()) |>
      mutate(system = label, tier = k, boundary = sprintf("%s >= %d", var, t), .before = 1)
  })) |> mutate(position = match(idx, ord), entry = master$entry[idx], entry_class = master$entry_class[idx], n_lists = master$n_lists[idx], n_families_any = master$n_families_any[idx],
                n_family_roots = master$n_family_roots[idx], n_firm_lists = n_firm_lists[idx], n_provisional_lists = n_prov_lists[idx], family_coverage = unname(family_coverage[entry])) |> select(-idx)
  list(stair = stair, summary = summ, words = words, tied = tied)
}
bp <- boundary_tables(master$n_lists, tp45$thresholds, tp45$sizes, tp45$obj, order_primary, "n_lists", "primary")
bf <- boundary_tables(master$n_families_any, tf45$thresholds, tf45$sizes, tf45$obj, order_family, "n_families_any", "family-aware")
tier_boundary_summary <- bind_rows(bp$summary, bf$summary)
tier_count_staircase  <- bind_rows(bp$stair, bf$stair)
tier_boundary_words   <- bind_rows(bp$words, bf$words)
tier_boundary_tied_groups <- bind_rows(bp$tied, bf$tied)
add_check("tiers", "the chosen thresholds minimise the objective among the one-step alternatives listed", "", all(tier_boundary_summary$objective_chosen <= coalesce(tier_boundary_summary$objective_if_one_value_lower, Inf) + 1e-9) && all(tier_boundary_summary$objective_chosen <= coalesce(tier_boundary_summary$objective_if_one_value_higher, Inf) + 1e-9))
add_check("tiers", "family-aware tier 3 target 4 x 127 = 508 lies between the cumulative sizes at n_families_any >= 3 and >= 2", sprintf("%d < 508 < %d", tf45$sizes[3], sum(master$n_families_any >= 2)), tf45$sizes[3] < 508 && sum(master$n_families_any >= 2) > 508)

# ---- 6. Version and normalisation audit ----
log_line("version and normalisation audit")
# Raw entries as found (lists/*.txt, entries in file order, original case, duplicates kept) and the normalisation
# of the source script (lines 407-416) re-applied here, for all 70 registered sources.
normalise_entries <- function(e0) {
  e1 <- stri_trans_nfc(e0); e2 <- stri_replace_all_regex(e1, "[\\u2018\\u2019\\u02BC\\u2032]", "'"); e3 <- stri_trim_both(e2)
  e4 <- stri_replace_all_regex(e3, "\\s+", " "); e5 <- stri_trans_tolower(e4); e5 <- e5[e5 != ""]; unique(e5)
}
list_files <- list.files(src_lists, pattern = "[.]txt$", full.names = TRUE)
raw_lists  <- setNames(lapply(list_files, function(f) readLines(f, encoding = "UTF-8", warn = FALSE)), sub("[.]txt$", "", basename(list_files)))
norm_lists <- lapply(raw_lists, normalise_entries)
add_check("normalisation", "70 extracted lists read from lists/", length(raw_lists), length(raw_lists) == 70)
add_check("normalisation", "raw entry counts of the included lists equal n_raw_entries in list_summary.csv", "", all(vapply(included_ids, function(id) length(raw_lists[[id]]), integer(1)) == list_summary$n_raw_entries[match(included_ids, list_summary$list_id)]))
add_check("normalisation", "re-applied normalisation of lists/*.txt reproduces the membership matrix for all 45 included lists", "", all(vapply(included_ids, function(id) setequal(norm_lists[[id]], set_of(id)), logical(1))))
extra_norm <- function(x, how) switch(how, none = x, strip_apostrophes = unique(stri_replace_all_fixed(x, "'", "")), drop_apostrophe_entries = x[!stri_detect_fixed(x, "'")],
                                      drop_non_word_entries = x[stri_detect_regex(x, "^[a-z]+$")], strip_apostrophes_and_drop_non_word = { y <- unique(stri_replace_all_fixed(x, "'", "")); y[stri_detect_regex(y, "^[a-z]+$")] }, stop("unknown extra normalisation ", how))
raw_hash_of <- function(id) {
  rows <- acq_log |> filter(list_id == !!id)
  f <- switch(id, nltk_english = "nltk_stopwords_english.txt", bow_rainbow = "bow_stopwords.c", python_stop_words_2025 = "stop_words-2025.11.4_english.txt", rows$file[1])
  rows$sha256[rows$file == f][1] %||% NA_character_
}
equality_register <- tribble(
  ~list_a, ~list_b, ~reported_as, ~extra_normalisation,
  "smart_lextek", "rake_smart", "exact-duplicate group (SMART x 6)", "none",
  "smart_lextek", "tm_smart", "exact-duplicate group (SMART x 6)", "none",
  "smart_lextek", "tidytext_smart", "exact-duplicate group (SMART x 6)", "none",
  "smart_lextek", "stopwords_pkg_smart", "exact-duplicate group (SMART x 6)", "none",
  "smart_lextek", "quanteda_smart", "exact-duplicate group (SMART x 6)", "none",
  "smart_lextek", "sec_smart_lextek_2011", "2011 Wayback snapshot of the same page", "none",
  "snowball_original", "tm_english", "exact-duplicate group (Snowball x 5)", "none",
  "snowball_original", "tidytext_snowball", "exact-duplicate group (Snowball x 5)", "none",
  "snowball_original", "python_stop_words_2014", "exact-duplicate group (Snowball x 5)", "none",
  "snowball_original", "lexicon_python", "exact-duplicate group (Snowball x 5)", "none",
  "stopwords_pkg_snowball", "quanteda_snowball", "exact-duplicate group (stopwords/quanteda snowball x 2)", "none",
  "stopwords_pkg_snowball", "snowball_original", "Snowball plus will", "none",
  "bow_rainbow", "mallet", "exact-duplicate group (Bow/MALLET/lexicon x 3)", "none",
  "bow_rainbow", "lexicon_mallet", "exact-duplicate group (Bow/MALLET/lexicon x 3)", "none",
  "bow_rainbow", "smart_lextek", "Bow = SMART without its 47 apostrophe forms", "drop_apostrophe_entries",
  "weka_rainbow", "bow_rainbow", "Weka = Bow plus ll, ve", "none",
  "buckley_salton_qdap", "smart_lextek", "qdap Buckley-Salton = SMART with apostrophes stripped (help: reduced from 571 to 546)", "strip_apostrophes",
  "buckley_salton_qdap", "bow_rainbow", "qdap Buckley-Salton vs Bow (both apostrophe-free)", "strip_apostrophes",
  "yake", "smart_lextek", "YAKE = SMART plus dr, dra, mr, ms", "none",
  "onix_lextek", "lsa_stopwords_en", "exact-duplicate group (ONIX/lsa x 2)", "none",
  "onix_lextek", "fox_1989", "ONIX = Fox without numbered, numbering", "none",
  "tidytext_onix", "qdap_onix", "exact-duplicate group (tidytext/qdap onix x 2)", "none",
  "tidytext_onix", "onix_lextek", "tidytext onix = ONIX without b-z and so", "none",
  "marimo_upstream", "marimo_pkg", "exact-duplicate group (marimo x 2)", "none",
  "sklearn", "glasgow", "scikit-learn = Glasgow without computer, fify, plus fifty", "none",
  "voyant_en", "taporware", "Voyant = TAPoRware plus thee, thou, thy", "none",
  "taporware", "glasgow", "TAPoRware = Glasgow minus 20 words plus numbers, letters, punctuation", "drop_non_word_entries",
  "gensim", "sklearn", "gensim = scikit-learn plus 19 words", "none",
  "gensim", "spacy", "collection claim: gensim 'same as spaCy'", "drop_apostrophe_entries",
  "spacy", "sklearn", "spaCy as a curated Glasgow variant", "drop_apostrophe_entries",
  "nltk_english", "stopwords_pkg_nltk", "NLTK 198 vs the 179-entry snapshot in the stopwords package", "none",
  "nltk_english", "sec_igor_nltk", "NLTK 198 vs the 153-entry igorbrigadir copy", "none",
  "nltk_english", "dkpro", "NLTK 198 vs the 127-entry DKPro copy", "none",
  "spark_ml", "nltk_english", "Spark (documented NLTK derivative) vs current NLTK", "none",
  "spark_ml", "stopwords_pkg_snowball", "Spark vs the 175-entry Snowball variant", "none",
  "dkpro", "snowball_original", "DKPro vs Snowball", "drop_apostrophe_entries",
  "dl4j", "stopwords_pkg_snowball", "DL4J superset of the 175-entry Snowball variant", "drop_non_word_entries",
  "corenlp_patterns", "snowball_original", "CoreNLP vs Snowball (punctuation tokens, apostrophe-stripped contractions)", "strip_apostrophes_and_drop_non_word",
  "sec_igor_cook", "cook_1988", "igorbrigadir Cook copy vs the parsed Wayback page", "none",
  "sec_spacy_v2", "spacy", "spaCy v2.0.0 word block vs the current runtime set", "none"
)
raw_vs_normalised_equality <- bind_rows(lapply(seq_len(nrow(equality_register)), function(i) {
  a <- equality_register$list_a[i]; b <- equality_register$list_b[i]; how <- equality_register$extra_normalisation[i]
  ra <- raw_lists[[a]]; rb <- raw_lists[[b]]; na <- norm_lists[[a]]; nb <- norm_lists[[b]]
  ea <- extra_norm(na, how); eb <- extra_norm(nb, how)
  ha <- raw_hash_of(a); hb <- raw_hash_of(b)
  tibble(list_a = a, list_b = b, reported_as = equality_register$reported_as[i],
         raw_file_hash_identical = !is.na(ha) && !is.na(hb) && ha == hb,
         n_raw_entries_a = length(ra), n_raw_entries_b = length(rb),
         raw_sequence_identical = identical(ra, rb), raw_multiset_identical = identical(sort(ra), sort(rb)), raw_set_identical_case_sensitive = setequal(ra, rb),
         n_normalised_a = length(na), n_normalised_b = length(nb), normalised_set_identical = setequal(na, nb),
         jaccard_normalised = round(length(intersect(na, nb)) / length(union(na, nb)), 4),
         n_only_in_a_normalised = length(setdiff(na, nb)), n_only_in_b_normalised = length(setdiff(nb, na)),
         extra_normalisation_tested = how, identical_after_extra_normalisation = if (how == "none") NA else setequal(ea, eb),
         n_only_in_a_after_extra = if (how == "none") NA_integer_ else length(setdiff(ea, eb)), n_only_in_b_after_extra = if (how == "none") NA_integer_ else length(setdiff(eb, ea)),
         words_only_in_a_after_extra = if (how == "none") "" else collapse_words(setdiff(ea, eb)), words_only_in_b_after_extra = if (how == "none") "" else collapse_words(setdiff(eb, ea)),
         words_only_in_a_normalised = collapse_words(setdiff(na, nb)), words_only_in_b_normalised = collapse_words(setdiff(nb, na)))
}))
# NLTK snapshots: nesting of the acquired NLTK-labelled lists.
nltk_ids <- c("dkpro", "sec_igor_nltk", "stopwords_pkg_nltk", "nltk_english")
nltk_snapshots <- bind_rows(lapply(seq_along(nltk_ids), function(k) {
  id <- nltk_ids[k]; x <- norm_lists[[id]]; prev <- if (k > 1) norm_lists[[nltk_ids[k - 1]]] else character(); nxt <- if (k < length(nltk_ids)) norm_lists[[nltk_ids[k + 1]]] else NULL
  tibble(snapshot_order = k, list_id = id, source = list_summary$list_name[list_summary$list_id == id], pin = list_summary$pin[list_summary$list_id == id],
         n_entries = length(x), n_word_entries = sum(stri_detect_regex(x, "^[a-z]+$")), n_apostrophe_entries = sum(stri_detect_fixed(x, "'")),
         subset_of_next_snapshot = if (is.null(nxt)) NA else all(x %in% nxt), superset_of_previous_snapshot = if (k == 1) NA else all(prev %in% x),
         n_added_vs_previous = length(setdiff(x, prev)), words_added_vs_previous = if (k == 1) "" else collapse_words(setdiff(x, prev)), n_removed_vs_previous = if (k == 1) NA_integer_ else length(setdiff(prev, x)),
         documented_date_or_pin = c("DKPro commit 2013-11-15 (README: 'Files copied from NLTK.')", "igorbrigadir/stopwords commit 2019-08-20 (date of the copy, not of the NLTK state)", "stopwords 2.3 (CRAN); snapshot date not stated", "nltk_data gh-pages commit 2026-01-09")[k])
})) |> bind_rows(tibble(snapshot_order = NA_integer_, list_id = "spark_ml", source = list_summary$list_name[list_summary$list_id == "spark_ml"], pin = list_summary$pin[list_summary$list_id == "spark_ml"],
                        n_entries = length(norm_lists$spark_ml), n_word_entries = sum(stri_detect_regex(norm_lists$spark_ml, "^[a-z]+$")), n_apostrophe_entries = sum(stri_detect_fixed(norm_lists$spark_ml, "'")),
                        subset_of_next_snapshot = NA, superset_of_previous_snapshot = NA, n_added_vs_previous = NA_integer_, words_added_vs_previous = "", n_removed_vs_previous = NA_integer_,
                        documented_date_or_pin = "Spark tag v4.2.0; README repeats NLTK's README; content matches no acquired NLTK snapshot (see raw_vs_normalised_equality.csv)"))
nltk_nested <- all(nltk_snapshots$subset_of_next_snapshot[1:3])
add_check("versions", "the four acquired NLTK-labelled lists are strictly nested 127 < 153 < 179 < 198", paste(nltk_snapshots$n_entries[1:4], collapse = " < "), nltk_nested && all(diff(nltk_snapshots$n_entries[1:4]) > 0))
spark_in_chain <- any(vapply(nltk_ids, function(id) setequal(norm_lists$spark_ml, norm_lists[[id]]), logical(1)))
add_check("versions", "Spark's list equals none of the four acquired NLTK snapshots and equals the 175-entry Snowball variant plus 6 entries", collapse_words(setdiff(norm_lists$spark_ml, norm_lists$stopwords_pkg_snowball)),
          !spark_in_chain && all(norm_lists$stopwords_pkg_snowball %in% norm_lists$spark_ml) && length(setdiff(norm_lists$spark_ml, norm_lists$stopwords_pkg_snowball)) == 6)
# Normalisation rules as implemented (source script lines 401-431) with their measured effect on the included lists.
norm_inc <- norm_log |> filter(list_id %in% included_ids)
class_counts <- master |> count(entry_class) |> arrange(desc(n))
normalisation_rules <- tribble(
  ~step, ~rule_as_implemented, ~source_code, ~effect_on_included_lists, ~what_is_not_done,
  "raw preservation", "every upstream file is stored byte-exact under raw/ (download mode wb; package files copied); every extracted list is written to lists/<list_id>.txt in file order with original case and duplicates", "lines 240-253 (download), 276-297 (package exports), 399 (lists/)", sprintf("%d raw files hashed in acquisition_log.csv; %d list files", nrow(acq_log), length(raw_lists)), "no character is altered before normalisation",
  "parsing", "comment lines and code syntax removed by an explicit parser per source (Snowball '|' comments; '#' + text comments where declared; Python, Java, C and HTML wrappers stripped); a line holding only '#' is an entry", "lines 310-388", "entries such as '#', '\\?' (TAPoRware), '-lrb-' (CoreNLP) and '\"the' (DL4J) survive parsing as entries", "no token is split, joined or filtered on character class",
  "NFC", "Unicode canonical composition", "line 409", sprintf("%d included-list entries changed by NFC", sum(stri_detect_fixed(norm_inc$changes, "nfc"))), "",
  "apostrophes", "U+2018, U+2019, U+02BC and U+2032 replaced by the ASCII apostrophe U+0027; the apostrophe itself is never removed", "line 410", sprintf("%d entries changed (all in spacy, whose runtime set adds curly variants of its contraction tokens); after the change they collapse onto the ASCII forms", sum(stri_detect_fixed(norm_inc$changes, "apostrophe"))), "contractions are not expanded or stripped: don't (Snowball), dont (qdap, CoreNLP) and don (NLTK, gensim) stay three distinct entries",
  "trim and whitespace", "leading and trailing whitespace removed; internal runs of whitespace collapsed to one space", "lines 411-412", sprintf("%d entries trimmed (dl4j 'ours ', corenlp 'ours ')", sum(stri_detect_fixed(norm_inc$changes, "trim"))), "",
  "case", "lower case", "line 413", sprintf("%d entries lower-cased (dl4j '\"The', cook 'I', qdap function words I'd I'll I'm I've)", sum(stri_detect_fixed(norm_inc$changes, "case"))), "",
  "empty entries", "entries that become empty are dropped", "line 416", sprintf("%d dropped as empty", sum(norm_inc$dropped == "empty")), "",
  "within-list duplicates", "the first occurrence is kept, later identical normalised entries in the same list are dropped and logged", "line 416", sprintf("%d within-list duplicates collapsed (e.g. SMART's second 'would', ONIX's 6 repeats, spacy's 15 apostrophe variants and repeated 'very')", sum(norm_inc$dropped == "duplicate_within_list")), "",
  "non-word entries", "numbers, punctuation, regex-escaped tokens and malformed tokens stay as distinct entries and are classified in entry_class", "lines 419-427", paste(sprintf("%s %d", class_counts$entry_class, class_counts$n), collapse = "; "), "they are not removed from the union; they inflate n_lists = 1 rows only (they come from taporware, voyant_en, corenlp_patterns, dl4j)",
  "comparison", "set operations (Jaccard, duplicates, consensus counts) use the normalised unique entries; nothing else is folded", "lines 537-560, 592", "identity is normalised-set identity: Bow = MALLET = lexicon_mallet although the raw sequences differ (see raw_vs_normalised_equality.csv)", "no stemming, no apostrophe stripping, no removal of single letters or numbers"
)
# Registered sources, excluded candidates and manifests: how 70, 26 and 109 relate.
kind_counts <- list_summary |> count(kind) |> arrange(kind)
n_manifests <- c(tables = length(list.files(src_tab, pattern = "[.]manifest[.]yml$")), figures = length(list.files(file.path(src_dir, "figures"), pattern = "[.]manifest[.]yml$")), lists = length(list.files(src_lists, pattern = "[.]manifest[.]yml$")))
source_accounting <- tribble(
  ~count, ~value, ~composition, ~where_measured,
  "registered sources", nrow(list_summary), paste(sprintf("%s %d", kind_counts$kind, kind_counts$n), collapse = "; "), "list_summary.csv rows (one per registry entry of english_stopword_lists.R lines 56-128)",
  "named lists in the comparison", length(included_ids), sprintf("%d included (firm) + %d included_provisional", length(firm_ids), length(provisional_ids)), "provenance_table.csv inclusion_status",
  "provenance rows", nrow(provenance), "45 named + 3 aggregate conditions + 1 acquired-but-excluded (jockers); secondary copies have no provenance row", "provenance_table.csv",
  "raw files in the acquisition log", nrow(acq_log), sprintf("%d non-secondary files (49 sources + 6 archive members + 1 tidytext export) + %d secondary copies", sum(!grepl("^secondary/", acq_log$file)), sum(grepl("^secondary/", acq_log$file))), "acquisition_log.csv",
  "documented but not acquired candidates", nrow(excluded), "rows of excluded_candidates.csv; one row (Alir3z4 2018) was in fact acquired as a secondary copy and one (Snowball commented-out words) was extracted into a table from an acquired file", "excluded_candidates.csv",
  "manifests", sum(n_manifests), sprintf("%d tables + %d figures + %d extracted lists (one per registered source)", n_manifests[["tables"]], n_manifests[["figures"]], n_manifests[["lists"]]), "*.manifest.yml sidecars under stopword_lists/"
)
add_check("versions", "70 registered sources = 45 named + 3 aggregate + 1 excluded + 21 secondary; 26 excluded candidates; 109 manifests", sprintf("%d / %d / %d", nrow(list_summary), nrow(excluded), sum(n_manifests)), nrow(list_summary) == 70 && nrow(excluded) == 26 && sum(n_manifests) == 109)
eq <- function(a, b) raw_vs_normalised_equality |> filter(list_a == a, list_b == b)
bs <- eq("buckley_salton_qdap", "smart_lextek"); bw <- eq("bow_rainbow", "smart_lextek"); bm <- eq("bow_rainbow", "mallet"); bl <- eq("bow_rainbow", "lexicon_mallet"); sm6 <- raw_vs_normalised_equality |> filter(list_a == "smart_lextek", reported_as == "exact-duplicate group (SMART x 6)")
sn5 <- raw_vs_normalised_equality |> filter(list_a == "snowball_original", grepl("Snowball x 5", reported_as)); tg <- eq("taporware", "glasgow"); gs <- eq("gensim", "spacy"); cs <- eq("corenlp_patterns", "snowball_original"); dk <- eq("dkpro", "snowball_original")
version_normalisation_audit <- tribble(
  ~topic, ~question, ~measured_facts, ~documentary_evidence, ~assessment, ~status,
  "NLTK 127 versus 198", "Does the 127-versus-198 difference reflect a documented historical version change, different embedded snapshots across tools, or something else?",
    sprintf("The four acquired NLTK-labelled lists are %s nested: DKPro %d < igorbrigadir copy %d < stopwords package %d < current NLTK %d. Each step only adds entries: +%d bare contraction fragments (%s), +%d apostrophe forms, +%d pronoun contractions. Spark's 181-entry list matches none of them (it equals the 175-entry Snowball variant plus can don just now s t).", ifelse(nltk_nested, "strictly", "not"), nltk_snapshots$n_entries[1], nltk_snapshots$n_entries[2], nltk_snapshots$n_entries[3], nltk_snapshots$n_entries[4], nltk_snapshots$n_added_vs_previous[2], nltk_snapshots$words_added_vs_previous[2], nltk_snapshots$n_added_vs_previous[3], nltk_snapshots$n_added_vs_previous[4]),
    "NLTK corpus README (raw/nltk_stopwords_README.txt): lists 'obtained from' the PostgreSQL snowball stopwords directory; 'The English list has been augmented https://github.com/nltk/nltk_data/issues/22'. DKPro README (quoted in provenance_table.csv, file not preserved): 'Files copied from NLTK.' The stopwords package help cites nltk_data stopwords.zip without a date.",
    "Consistent with successive states of one nltk_data file embedded by different tools at different times: the nesting is measured, and one augmentation is documented (issue 22). What the evidence does not establish: the nltk_data commit or date of the 153- and 179-entry states (the igorbrigadir commit dates the copy, not the NLTK state; the stopwords package states no snapshot date), whether DKPro's 127 equals PostgreSQL's english.stop (not acquired), and where Spark's 181-entry content comes from. The '2015' date attached to the 153-entry copy in the delivered README is not supported by any acquired file.",
    "partly resolved: nesting measured and one change documented; intermediate dates, the PostgreSQL ancestor and the Spark snapshot unresolved",
  "Bow / MALLET / SMART", "Is the reported equality raw equality or equality after normalisation?",
    sprintf("Bow vs MALLET: raw sequences identical = %s (both %d raw entries with 'would' twice), normalised sets identical = %s. Bow vs lexicon sw_mallet: raw sequences identical = %s (%d vs %d raw entries: the package export has no duplicate), normalised sets identical = %s. Bow vs SMART: not equal at any level (Jaccard %.4f); after dropping SMART's apostrophe entries the sets are %s (%d / %d entries differ).", bm$raw_sequence_identical, bm$n_raw_entries_a, bm$normalised_set_identical, bl$raw_sequence_identical, bl$n_raw_entries_a, bl$n_raw_entries_b, bl$normalised_set_identical, bw$jaccard_normalised, ifelse(bw$identical_after_extra_normalisation, "identical", "not identical"), bw$n_only_in_a_after_extra, bw$n_only_in_b_after_extra),
    "MALLET and Weka document Rainbow as their source; stopwords.c has no source statement; no file states that Bow's list is SMART's.",
    "Bow = MALLET holds for the raw entry sequence, not only after normalisation; lexicon's copy differs only by the dropped duplicate. Bow = SMART minus apostrophe forms is an exact set relation after normalisation, but it is a subset relation (47 entries), not equality, and it is undocumented.",
    "resolved as stated; the Bow-to-SMART derivation remains undocumented",
  "qdap Buckley-Salton", "Is BuckleySaltonSWL a reduced SMART list (help: 571 to 546) or a re-formatted one?",
    sprintf("Raw: %d entries (%d unique) vs SMART %d (%d unique); normalised Jaccard to SMART %.4f, to Bow %.4f. After stripping apostrophes inside SMART's entries the sets are %s: %d entries only in Buckley-Salton (%s), %d only in stripped SMART (%s).", list_summary$n_raw_entries[list_summary$list_id == "buckley_salton_qdap"], list_summary$n_unique_entries[list_summary$list_id == "buckley_salton_qdap"], list_summary$n_raw_entries[list_summary$list_id == "smart_lextek"], list_summary$n_unique_entries[list_summary$list_id == "smart_lextek"], bs$jaccard_normalised, eq("buckley_salton_qdap", "bow_rainbow")$jaccard_normalised, ifelse(bs$identical_after_extra_normalisation, "identical", "not identical"), bs$n_only_in_a_after_extra, bs$words_only_in_a_after_extra, bs$n_only_in_b_after_extra, bs$words_only_in_b_after_extra),
    "qdapDictionaries help: quotes the Onix page attribution and notes 'Reduced from the original 571 words to 546'.",
    "The difference from SMART is formatting (apostrophes removed inside contractions, which also merges some forms into existing words) plus the removal of the single letters listed; it is not a curated word-level reduction. The delivered comparison keeps it as a distinct set (Jaccard 0.8863 to Bow) because the normalisation never strips apostrophes.",
    "resolved",
  "SMART six copies and Snowball five copies", "Are the exact-duplicate groups identical as raw files or only as normalised sets?",
    sprintf("SMART: raw sequences identical to the lextek page for %d of %d copies (%s); the 2011 snapshot of the page is %s in raw sequence. Snowball: raw sequences identical to stop.txt for %d of %d copies (%s). Raw file hashes are never equal across copies because the files differ in format (HTML page, .dat, package export).", sum(sm6$raw_sequence_identical), nrow(sm6), paste(sprintf("%s=%s", sm6$list_b, sm6$raw_sequence_identical), collapse = ", "), ifelse(eq("smart_lextek", "sec_smart_lextek_2011")$raw_sequence_identical, "identical", "not identical"), sum(sn5$raw_sequence_identical), nrow(sn5), paste(sprintf("%s=%s", sn5$list_b, sn5$raw_sequence_identical), collapse = ", ")),
    "tm, tidytext, stopwords and quanteda document their SMART and Snowball sources (provenance_table.csv); RAKE's header cites the Cornell file.",
    "See raw_vs_normalised_equality.csv for the per-pair levels (file hash, raw sequence, raw multiset, normalised set).",
    "resolved",
  "Other reported near-equalities", "Which reported relations hold at which level?",
    sprintf("Weka = Bow + ll ve; YAKE = SMART + dr dra mr ms; ONIX = Fox - numbered numbering; tidytext onix = ONIX - b..z so; stopwords snowball = Snowball + will; scikit-learn = Glasgow - computer fify + fifty; Voyant = TAPoRware + thee thou thy: all are normalised-set relations verified in raw_vs_normalised_equality.csv (none is a raw-file identity). TAPoRware vs Glasgow after dropping non-word entries: %s (%d / %d differ). gensim vs spaCy after dropping apostrophe entries: %s (%d / %d differ). CoreNLP vs Snowball after stripping apostrophes and dropping non-word entries: %d / %d differ. DKPro vs Snowball after dropping apostrophe entries: %d / %d differ.", ifelse(tg$identical_after_extra_normalisation, "identical", "not identical"), tg$n_only_in_a_after_extra, tg$n_only_in_b_after_extra, ifelse(gs$identical_after_extra_normalisation, "identical", "not identical"), gs$n_only_in_a_after_extra, gs$n_only_in_b_after_extra, cs$n_only_in_a_after_extra, cs$n_only_in_b_after_extra, dk$n_only_in_a_after_extra, dk$n_only_in_b_after_extra),
    "Weka (Javadoc: based on Rainbow), scikit-learn (header: Glasgow), Voyant (header: TAPoRware), tidytext onix (help: lextek), qdap onix (help: reduced from 429 to 404) are documented; YAKE, ONIX-to-Fox and the stopwords-package 'will' are not.",
    "Reported equalities in the delivered README are normalised-set statements; the word 'identical' there always means identical normalised sets.",
    "resolved",
  "Apostrophes and non-word entries", "How were they treated in raw preservation versus comparison?",
    sprintf("Raw files and lists/ keep every character. Normalisation maps four curly apostrophe code points to U+0027 (%d entries, all spacy) and never removes an apostrophe. Entry classes in the union: %s. Contractions therefore exist in up to three spellings (don't / dont / don) that never merge.", sum(stri_detect_fixed(norm_inc$changes, "apostrophe")), paste(sprintf("%s %d", class_counts$entry_class, class_counts$n), collapse = ", ")),
    "english_stopword_lists.R lines 401-431 and README 'Method notes'.",
    "Rules and counts are itemised in normalisation_rules.csv. Consequence for the consensus counts: apostrophe-free copies (Bow, MALLET, Weka, qdap Buckley-Salton, CoreNLP, DKPro, the 153-entry NLTK state) never vote for the apostrophe forms, and the apostrophe-carrying lists never vote for the bare forms.",
    "resolved",
  "70 sources, 26 candidates, 109 manifests", "How do the three headline counts relate?",
    sprintf("%d registered sources (%s) yield %d raw files and %d extracted lists; %d provenance rows cover the non-secondary sources; %d excluded-candidate rows are documented without acquisition; %d manifests = %d tables + %d figures + %d lists.", nrow(list_summary), paste(sprintf("%s %d", kind_counts$kind, kind_counts$n), collapse = ", "), nrow(acq_log), length(raw_lists), nrow(provenance), nrow(excluded), sum(n_manifests), n_manifests[["tables"]], n_manifests[["figures"]], n_manifests[["lists"]]),
    "Registry and exclusion tables inside english_stopword_lists.R; acquisition_log.csv.",
    "The 26 candidates are not a subset of the 70 sources, except that the Alir3z4 2018 list appears in both (as a secondary copy) and the Snowball commented-out words come from an acquired file. See source_accounting.csv.",
    "resolved",
  "Fox 425 versus 421 and ONIX 429 versus 423", "Do the entry counts match the documented sizes?",
    sprintf("Fox (RAKE copy): %d raw, %d unique; the paper reports 421. ONIX page: %d raw, %d unique (6 repeats: down, high x2, new, right, still); page text says 429.", list_summary$n_raw_entries[list_summary$list_id == "fox_1989"], list_summary$n_unique_entries[list_summary$list_id == "fox_1989"], list_summary$n_raw_entries[list_summary$list_id == "onix_lextek"], list_summary$n_unique_entries[list_summary$list_id == "onix_lextek"]),
    "RAKE header '#From \"A stop list for general text\" Fox 1989'; the paper was not consulted; the ONIX page states 429.",
    "Whether RAKE's copy adds four words to Fox's published list, or the paper's count excludes something, cannot be decided from the acquired files. ONIX's 429 is a count of lines including the six repeats.",
    "unresolved (Fox); resolved (ONIX)",
  "spaCy runtime set", "Which spaCy list is compared?",
    sprintf("The runtime set (words block + contraction tokens + curly variants) normalises to %d entries; the v2.0.0 word block alone gives %d, differing by the 7 contraction tokens %s.", length(norm_lists$spacy), length(norm_lists$sec_spacy_v2), eq("sec_spacy_v2", "spacy")$words_only_in_b_normalised),
    "spacy/lang/en/stop_words.py (words block unchanged since v2.0.0; contractions appended at import).",
    "The comparison uses spaCy's runtime set, so 7 of its 312 entries are tokens ('s, n't, ...) that no other list except Pattern and CoreNLP-style lists carry.",
    "resolved",
  "Unpinned sources", "Which included lists have no version pin?",
    "glasgow (live page), gate_kea (live source-browser page), okapi_framework (bitbucket master): retrieved at run time, SHA256 and retrieval time in acquisition_log.csv.",
    "provenance_table.csv version_date; acquisition_log.csv.",
    "A later re-download could change these three lists; the delivered hashes define the compared content.",
    "open by nature"
)

# ---- Checks, run info, write tables ----
log_line("writing tables")
table_dims <- list()
save_table <- function(df, name) { p <- file.path(tab_dir, paste0(name, ".csv")); write_csv(df, p, na = ""); table_dims[[p]] <<- c(rows = nrow(df), cols = ncol(df)); invisible(p) }
save_table(included_list_audit, "included_list_audit")
save_table(family_lineage_audit, "family_lineage_audit")
save_table(family_lineage_summary, "family_lineage_summary")
save_table(family_lineage_pairs, "family_lineage_pairs")
save_table(consensus_variable_definitions, "consensus_variable_definitions")
save_table(consensus_variable_examples, "consensus_variable_examples")
save_table(consensus_variable_examples_long, "consensus_variable_examples_long")
save_table(provisional_contribution, "provisional_contribution")
save_table(firm_vs_provisional_summary, "firm_vs_provisional_summary")
save_table(firm_vs_provisional_tier_sizes, "firm_vs_provisional_tier_sizes")
save_table(firm_vs_provisional_tier_transitions, "firm_vs_provisional_tier_transitions")
save_table(firm_vs_provisional_changes, "firm_vs_provisional_changes")
save_table(firm_vs_provisional_words, "firm_vs_provisional_words")
save_table(tier_boundary_summary, "tier_boundary_summary")
save_table(tier_count_staircase, "tier_count_staircase")
save_table(tier_boundary_words, "tier_boundary_words")
save_table(tier_boundary_tied_groups, "tier_boundary_tied_groups")
save_table(version_normalisation_audit, "version_normalisation_audit")
save_table(raw_vs_normalised_equality, "raw_vs_normalised_equality")
save_table(nltk_snapshots, "nltk_snapshots")
save_table(normalisation_rules, "normalisation_rules")
save_table(source_accounting, "source_accounting")
run_end <- Sys.time()
git_commit <- tryCatch(system2("git", c("-C", shQuote(project_dir), "rev-parse", "--short", "HEAD"), stdout = TRUE, stderr = FALSE), error = function(e) "")
checks_tbl <- bind_rows(checks)
audit_run_info <- tibble(item = c("run_start_utc", "run_end_utc", "runtime_minutes", "git_commit_at_start", "r_version", "source_script", "source_script_sha256", "source_run_id", "n_included_lists", "n_firm_lists", "n_provisional_lists", "n_entries", "boundary_window", "large_rank_move_share", "delivered_checks_passing"),
                        value = c(format(run_start, "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"), format(run_end, "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"), round(as.numeric(difftime(run_end, run_start, units = "mins")), 2), git_commit, R.version.string,
                                  source_script_path, sha256(source_script_path), paste(run_info_src$value[run_info_src$item == "run_end_utc"], run_info_src$value[run_info_src$item == "git_commit_at_start"]), length(included_ids), length(firm_ids), length(provisional_ids), nrow(master), boundary_window, large_rank_move_share,
                                  sprintf("%d of %d", sum(checks_src$pass), nrow(checks_src))))
save_table(audit_run_info, "audit_run_info")
save_table(checks_tbl, "audit_checks")

# ---- Manifests (same structure as the source analysis: raw inputs with SHA256, script hash, git commit, parameters, package versions) ----
package_versions <- lapply(c("dplyr", "tidyr", "tibble", "purrr", "stringi", "readr", "digest", "yaml"), function(p) list(name = p, version = as.character(packageVersion(p))))
input_files <- c(lapply(file.path(src_tab, paste0(input_tables, ".csv")), function(f) list(path = f, hash = sha256(f), format = "csv")),
                 lapply(list_files, function(f) list(path = f, hash = sha256(f), format = "other")),
                 list(list(path = source_script_path, hash = sha256(source_script_path), format = "r-script"), list(path = script_path, hash = sha256(script_path), format = "r-script")))
manifest_parameters <- list(
  purpose = "audit of the evidence structure behind the delivered stopword consensus rankings: lineage bases, consensus-variable reproduction, firm-versus-provisional influence, tier boundaries, version and normalisation questions; read-only over stopword_lists/; no family assignment changed, no keep/remove decision, no project list, no corpus access",
  included_lists = paste(included_ids, collapse = ", "), firm_lists = paste(firm_ids, collapse = ", "), provisional_lists = paste(provisional_ids, collapse = ", "),
  near_duplicate_jaccard = as.character(near_duplicate_jaccard), tier_ratio = paste(tier_ratio, collapse = ":"), family_review_jaccard = as.character(family_review_jaccard),
  boundary_window = as.character(boundary_window), large_rank_move_share = as.character(large_rank_move_share),
  lineage_basis_vocabulary = "root; documented (explicit statement / name only / via an intermediate not acquired); exact_identity; near_identity (Jaccard >= 0.90); similarity (Jaccard < 0.90)")
write_sidecar <- function(output) {
  manifest <- list(manifest_version = 1L, output_file = output, output_hash = sha256(output), output_format = "csv",
                   output_rows = unname(table_dims[[output]][["rows"]]), output_cols = unname(table_dims[[output]][["cols"]]), input_files = input_files,
                   transformation = list(script = script_path, script_hash = sha256(script_path), parameters = manifest_parameters, git_commit = git_commit),
                   software = list(language = "R", language_version = paste(R.version$major, R.version$minor, sep = "."), packages = package_versions, os = paste(Sys.info()[["sysname"]], Sys.info()[["release"]])),
                   timestamp = format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
                   notes = "Derived read-only from the delivered tables and extracted lists of r_analysis_outputs/stopword_lists/ (hashed above). Interpretation columns (lineage basis, assignment support, unresolved issues) come from the hand-maintained lineage register inside this script; every measured column is recomputed from the delivered membership matrix or the lists/ files.")
  write_yaml(manifest, paste0(output, ".manifest.yml"))
}
invisible(lapply(list.files(tab_dir, pattern = "[.]csv$", full.names = TRUE), write_sidecar))

print(as.data.frame(checks_tbl), right = FALSE)
cat(sprintf("\n%d audit tables written to %s in %.1f min.\n", length(table_dims), tab_dir, as.numeric(difftime(run_end, run_start, units = "mins"))))
