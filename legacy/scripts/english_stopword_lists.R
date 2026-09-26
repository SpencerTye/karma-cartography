# English stopword lists: acquisition, provenance, comparison and organisation for manual curation
#
# Standalone. Acquires candidate English stopword lists from pinned upstream sources
# (GitHub commits / release tags, Wayback Machine snapshots, PyPI, CRAN packages
# installed locally), preserves every acquired file untouched under raw/, writes every
# extracted list separately under lists/, documents provenance and the inclusion
# decision for every candidate, and compares the included lists (membership table,
# Jaccard similarities, duplicates, larger-list additions, lineage-aware consensus,
# curation tiers). Makes no keep/remove decision and creates no project stopword list.
# Nothing here reads the Reddit corpus.

# ---- Setup ----
suppressPackageStartupMessages({
  library(dplyr); library(tidyr); library(tibble); library(purrr); library(stringi); library(readr)
  library(ggplot2); library(scales); library(digest); library(yaml); library(curl); library(xml2); library(ggupset)
})
run_start <- Sys.time()
invisible(Sys.setlocale("LC_COLLATE", "C"))   # reproducible string ordering
log_line  <- function(...) cat(format(Sys.time(), "%H:%M:%S"), ..., "\n")

project_dir <- Sys.getenv("KC_REPO_ROOT", unset = getwd())
out_dir     <- file.path(project_dir, "r_analysis_outputs/stopword_lists")
raw_dir     <- file.path(out_dir, "raw")                    # byte-exact upstream files (cache; never rewritten)
sec_dir     <- file.path(raw_dir, "secondary")              # secondary copies used only for cross-checks
pkg_dir     <- file.path(raw_dir, "package_exports")        # lists as exposed by installed R packages
lists_dir   <- file.path(out_dir, "lists")                  # one extracted list per file, entries as found
tab_dir     <- file.path(out_dir, "tables")
fig_dir     <- file.path(out_dir, "figures")
script_path <- file.path(project_dir, "english_stopword_lists.R")
for (d in c(raw_dir, sec_dir, pkg_dir, lists_dir, tab_dir, fig_dir)) dir.create(d, recursive = TRUE, showWarnings = FALSE)
invisible(file.remove(list.files(c(lists_dir, tab_dir, fig_dir), pattern = "\\.(csv|png|txt|yml)$", full.names = TRUE)))

# Parameters (all recorded in the manifests)
near_duplicate_jaccard <- 0.90          # pairs at or above this Jaccard (and not identical) are reported as near-duplicates
tier_ratio             <- c(1, 2, 4)    # target size ratio of the three nested curation tiers
family_review_jaccard  <- 0.50          # family assignment is flagged for review below this within-family similarity
user_agent             <- "karma-cartography stopword-list acquisition (R curl)"
igor_commit            <- "ab85d86c3fac0360020a921b91ccf9d697b54757"   # igorbrigadir/stopwords HEAD, 2019-08-20
igor_raw               <- function(f) sprintf("https://raw.githubusercontent.com/igorbrigadir/stopwords/%s/en/%s", igor_commit, f)
gh_raw                 <- function(repo, ref, path) sprintf("https://raw.githubusercontent.com/%s/%s/%s", repo, ref, path)

sha256 <- function(path) paste0("sha256:", digest(path, algo = "sha256", file = TRUE))
checks <- list()
add_check <- function(group, check, measured, pass) checks[[length(checks) + 1L]] <<- tibble(group = group, check = check, measured = as.character(measured), pass = isTRUE(pass))
required_pkgs <- c("stopwords", "tm", "tidytext", "quanteda", "lexicon", "qdapDictionaries", "lsa")
missing_pkgs  <- required_pkgs[!vapply(required_pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_pkgs)) stop("Install the data packages first: ", paste(missing_pkgs, collapse = ", "))
pkg_version <- function(p) as.character(packageVersion(p))

# ---- Source registry ----
# INPUT : none (hand-maintained). One row per acquired file. `kind`: named = a named list that enters
#         the raw comparison if included; aggregate = comparison condition only; secondary = copy used for
#         cross-checks; excluded = acquired for documentation, kept out of every comparison.
# DOES  : fixes the URL, the pin (commit / tag / Wayback timestamp / package version), the parser and
#         the raw file name for every source.
sources <- tribble(
  ~list_id, ~list_name, ~kind, ~acquisition, ~url, ~pin, ~parser, ~parser_arg, ~file_name,
  "snowball_original", "Snowball English stop list (stop.txt, default words)", "named", "download", gh_raw("snowballstem/snowball-website", "ce80856f7574e040b191328062433247c67690e8", "algorithms/english/stop.txt"), "commit ce80856f, 2025-03-03", "snowball", NA, "snowball_stop.txt",
  "nltk_english", "NLTK stopwords corpus, English", "named", "archive", gh_raw("nltk/nltk_data", "98e14261ea", "packages/corpora/stopwords.zip"), "gh-pages commit 98e14261, 2026-01-09", "lines", NA, "nltk_stopwords.zip",
  "stopwords_pkg_nltk", "R stopwords package, source nltk", "named", "package", NA, paste("stopwords", pkg_version("stopwords")), "lines", NA, "package_exports/stopwords_pkg_nltk.txt",
  "spark_ml", "Apache Spark ML StopWordsRemover, english.txt", "named", "download", gh_raw("apache/spark", "v4.2.0", "mllib/src/main/resources/org/apache/spark/ml/feature/stopwords/english.txt"), "tag v4.2.0 (PyPI pyspark 4.2.0, 2026-07-14)", "lines", NA, "spark_v4.2.0_english.txt",
  "dkpro", "DKPro Toolbox corpus stopwords, english.txt.gz", "named", "download", gh_raw("dkpro/dkpro-toolbox", "154eb5d6ac31be2e07d35cdb0f4c3c36d48cac13", "dkpro.toolbox.corpus-asl/src/main/resources/corpus/stopwords/english.txt.gz"), "commit 154eb5d6, 2013-11-15", "gz_lines", NA, "dkpro_english.txt.gz",
  "dl4j", "Deeplearning4j NLP stopwords.txt", "named", "download", gh_raw("deeplearning4j/deeplearning4j", "b5f0ec072f3fd0da566e32f82c0e43ca36553f39", "deeplearning4j/deeplearning4j-nlp-parent/deeplearning4j-nlp/src/main/resources/stopwords.txt"), "commit b5f0ec07, 2019-06-06", "lines", NA, "dl4j_stopwords.txt",
  "tm_english", "R tm package, stopwords('english') (english.dat)", "named", "package", NA, paste("tm", pkg_version("tm")), "lines", NA, "package_exports/tm_english.dat",
  "tidytext_snowball", "R tidytext stop_words, lexicon snowball", "named", "package", NA, paste("tidytext", pkg_version("tidytext")), "lines", NA, "package_exports/tidytext_snowball.txt",
  "stopwords_pkg_snowball", "R stopwords package, source snowball", "named", "package", NA, paste("stopwords", pkg_version("stopwords")), "lines", NA, "package_exports/stopwords_pkg_snowball.txt",
  "quanteda_snowball", "R quanteda stopwords('en', source = 'snowball')", "named", "package", NA, paste("quanteda", pkg_version("quanteda")), "lines", NA, "package_exports/quanteda_snowball.txt",
  "marimo_upstream", "Marimo multilingual stopwords, English YAML (koheiw/marimo)", "named", "download", gh_raw("koheiw/marimo", "f3e26e376dedd2efda0c3df596a3fb3391bfa048", "yaml/stopwords_en.yml"), "commit f3e26e37, 2020-03-27", "marimo_yaml", NA, "marimo_stopwords_en.yml",
  "marimo_pkg", "R stopwords package, source marimo", "named", "package", NA, paste("stopwords", pkg_version("stopwords")), "lines", NA, "package_exports/stopwords_pkg_marimo.txt",
  "python_stop_words_2014", "Python stop-words package, original English list (Alir3z4/stop-words, 2014)", "named", "download", gh_raw("Alir3z4/stop-words", "c2ba7cd744", "english.txt"), "commit c2ba7cd7, 2014-05-06 ('Add English')", "lines", NA, "alir3z4_2014_english.txt",
  "lexicon_python", "R lexicon package, sw_python", "named", "package", NA, paste("lexicon", pkg_version("lexicon")), "lines", NA, "package_exports/lexicon_sw_python.txt",
  "smart_lextek", "SMART stop list (Onix Text Retrieval Toolkit reference, Stop Word List 2)", "named", "download", "http://web.archive.org/web/20080209105654id_/http://www.lextek.com/manuals/onix/stopwords2.html", "Wayback snapshot 2008-02-09 (lextek.com now a parking page)", "html_pre", NA, "lextek_stopwords2_wayback20080209.html",
  "rake_smart", "RAKE (aneesha/RAKE) SmartStoplist.txt", "named", "download", gh_raw("aneesha/RAKE", "7aef80f7ebc814bd3fa9f5b078c49ce11c734a10", "SmartStoplist.txt"), "commit 7aef80f7, 2012-12-17", "lines", "#", "rake_SmartStoplist.txt",
  "tm_smart", "R tm package, stopwords('SMART') (SMART.dat)", "named", "package", NA, paste("tm", pkg_version("tm")), "lines", NA, "package_exports/tm_SMART.dat",
  "tidytext_smart", "R tidytext stop_words, lexicon SMART", "named", "package", NA, paste("tidytext", pkg_version("tidytext")), "lines", NA, "package_exports/tidytext_SMART.txt",
  "stopwords_pkg_smart", "R stopwords package, source smart", "named", "package", NA, paste("stopwords", pkg_version("stopwords")), "lines", NA, "package_exports/stopwords_pkg_smart.txt",
  "quanteda_smart", "R quanteda stopwords('en', source = 'smart')", "named", "package", NA, paste("quanteda", pkg_version("quanteda")), "lines", NA, "package_exports/quanteda_smart.txt",
  "buckley_salton_qdap", "R qdapDictionaries BuckleySaltonSWL", "named", "package", NA, paste("qdapDictionaries", pkg_version("qdapDictionaries")), "lines", NA, "package_exports/qdap_BuckleySaltonSWL.txt",
  "bow_rainbow", "Bow / Rainbow toolkit built-in stoplist (libbow stopwords.c)", "named", "archive", "http://www.cs.cmu.edu/~mccallum/bow/src/bow-20020213.tar.gz", "bow-20020213 (last release, 2002-02-13)", "c_array", NA, "bow-20020213.tar.gz",
  "mallet", "MALLET default stoplist (stoplists/en.txt)", "named", "download", gh_raw("mimno/Mallet", "23ee9bcebfd69f6df87a14cb70d0831dfce897af", "stoplists/en.txt"), "commit 23ee9bce, 2010-04-26", "lines", NA, "mallet_en.txt",
  "lexicon_mallet", "R lexicon package, sw_mallet", "named", "package", NA, paste("lexicon", pkg_version("lexicon")), "lines", NA, "package_exports/lexicon_sw_mallet.txt",
  "weka_rainbow", "Weka weka.core.stopwords.Rainbow", "named", "download", gh_raw("Waikato/weka-3.8", "4cc176bb8e4e2b7672deeff2bbf1679c0651e0fd", "weka/src/main/java/weka/core/stopwords/Rainbow.java"), "weka-3.8 mirror commit 4cc176bb, 2016-04-11", "java_add", "m_Words\\.add\\(\"([^\"]*)\"\\)", "weka_Rainbow.java",
  "yake", "YAKE keyword extractor, stopwords_en.txt", "named", "download", gh_raw("INESCTEC/yake", "bf0856c9bcda800642008cdccb5d0fb0810ca365", "yake/core/StopwordsList/stopwords_en.txt"), "commit bf0856c9, 2025-05-27 (PyPI yake 0.7.3)", "lines", NA, "yake_stopwords_en.txt",
  "glasgow", "Glasgow Information Retrieval Group stop word list", "named", "download", "http://ir.dcs.gla.ac.uk/resources/linguistic_utils/stop_words", "unversioned live page, retrieved at run time", "lines", NA, "glasgow_stop_words.txt",
  "sklearn", "scikit-learn ENGLISH_STOP_WORDS", "named", "download", gh_raw("scikit-learn/scikit-learn", "1.9.0", "sklearn/feature_extraction/_stop_words.py"), "tag 1.9.0 (2026-06-02)", "frozenset", "ENGLISH_STOP_WORDS", "sklearn_1.9.0_stop_words.py",
  "taporware", "TAPoRware glasgowstoplist.txt", "named", "download", "http://web.archive.org/web/20181105213841id_/http://taporware.ualberta.ca:80/~taporware/cgi-bin/prototype/glasgowstoplist.txt", "Wayback snapshot 2018-11-05 (site defunct)", "lines", NA, "taporware_glasgowstoplist_wayback20181105.txt",
  "voyant_en", "Voyant Tools (trombone) default English stop list, stop.en.txt", "named", "download", gh_raw("voyanttools/trombone", "8b1ad4ff594933d5b553432b6f8db020f0929d74", "src/main/resources/org/voyanttools/trombone/stopwords/stop.en.txt"), "commit 8b1ad4ff, 2023-04-24", "lines", "#", "voyant_stop.en.txt",
  "spacy", "spaCy English STOP_WORDS (words, contractions and curly-apostrophe variants)", "named", "download", gh_raw("explosion/spaCy", "32c4b638ae0e042264fadc3ceb3ab67147c7b6f5", "spacy/lang/en/stop_words.py"), "master commit 32c4b638, 2026-03-21 (PyPI spacy 3.8.16)", "spacy", NA, "spacy_stop_words.py",
  "gensim", "gensim parsing.preprocessing STOPWORDS", "named", "download", gh_raw("piskvorky/gensim", "4.4.0", "gensim/parsing/preprocessing.py"), "tag 4.4.0 (2025-10-18)", "frozenset", "STOPWORDS", "gensim_4.4.0_preprocessing.py",
  "fox_1989", "Fox (1989) stop list for general text (copy shipped with RAKE)", "named", "download", gh_raw("aneesha/RAKE", "7aef80f7ebc814bd3fa9f5b078c49ce11c734a10", "FoxStoplist.txt"), "commit 7aef80f7, 2012-12-17", "lines", "#", "rake_FoxStoplist.txt",
  "lsa_stopwords_en", "R lsa package, stopwords_en", "named", "package", NA, paste("lsa", pkg_version("lsa")), "lines", NA, "package_exports/lsa_stopwords_en.txt",
  "pattern_clips", "Pattern (CLiPS) pattern.vector stopwords-en.txt", "named", "download", gh_raw("clips/pattern", "f660ad4a17772a8800382c8831500e081740abe1", "pattern/vector/stopwords-en.txt"), "commit f660ad4a, 2013-05-15", "comma", NA, "pattern_stopwords-en.txt",
  "onix_lextek", "ONIX stop list (Onix Text Retrieval Toolkit reference, Stop Word List 1)", "named", "download", "http://web.archive.org/web/20071117122640id_/http://www.lextek.com/manuals/onix/stopwords1.html", "Wayback snapshot 2007-11-17 (lextek.com now a parking page)", "html_pre", NA, "lextek_stopwords1_wayback20071117.html",
  "tidytext_onix", "R tidytext stop_words, lexicon onix", "named", "package", NA, paste("tidytext", pkg_version("tidytext")), "lines", NA, "package_exports/tidytext_onix.txt",
  "qdap_onix", "R qdapDictionaries OnixTxtRetToolkitSWL1", "named", "package", NA, paste("qdapDictionaries", pkg_version("qdapDictionaries")), "lines", NA, "package_exports/qdap_OnixTxtRetToolkitSWL1.txt",
  "gate_kea", "GATE Keyphrase Extraction Algorithm plugin (KEA) StopwordsEnglish.java", "named", "download", "https://gate.ac.uk/gate/plugins/Keyphrase_Extraction_Algorithm/src/kea/StopwordsEnglish.java", "unversioned live source-browser page, retrieved at run time", "gate_html", NA, "gate_kea_StopwordsEnglish.java.html",
  "corenlp_patterns", "Stanford CoreNLP patterns/surface/stopwords.txt", "named", "download", gh_raw("stanfordnlp/CoreNLP", "ceefe81491183e557cd047755a5a88a90bfeb2af", "data/edu/stanford/nlp/patterns/surface/stopwords.txt"), "commit ceefe814, 2016-01-22", "lines", NA, "corenlp_patterns_stopwords.txt",
  "lingpipe", "LingPipe EnglishStopTokenizerFactory built-in stop list", "named", "download", gh_raw("hvtuananh/lingpipe", "70331b9f9f95be76a6ba2d2e5010b6924069e054", "src/com/aliasi/tokenizer/EnglishStopTokenizerFactory.java"), "LingPipe 4.1.0 source clone, commit 70331b9f, 2013-11-18", "java_add", "STOP_SET\\.add\\(\"([^\"]*)\"\\)", "lingpipe_EnglishStopTokenizerFactory.java",
  "choi_c99", "C99 / TextTiling stop list (Choi 2000), copy in igorbrigadir/stopwords", "named", "download", igor_raw("choi_2000naacl.txt"), "igorbrigadir/stopwords commit ab85d86c, 2019-08-20", "lines", NA, "igor_choi_2000naacl.txt",
  "cook_1988", "Vivian Cook, list of English structure (function) words", "named", "download", "http://web.archive.org/web/20191225212407id_/http://www.viviancook.uk/Words/StructureWordsList.htm", "Wayback snapshot 2019-12-25 (live page returns 403)", "cook_html", NA, "viviancook_StructureWordsList_wayback20191225.htm",
  "qdap_function_words", "R qdapDictionaries function.words", "named", "package", NA, paste("qdapDictionaries", pkg_version("qdapDictionaries")), "lines", NA, "package_exports/qdap_function.words.txt",
  "okapi_framework", "Okapi Framework term extraction step, stopWords_en.txt", "named", "download", "https://bitbucket.org/okapiframework/okapi/raw/master/okapi/steps/termextraction/src/main/resources/net/sf/okapi/steps/termextraction/stopWords_en.txt", "bitbucket master, retrieved at run time", "lines", "#", "okapiframework_stopWords_en.txt",
  "stopwords_iso", "stopwords-iso English (stopwords-en.txt)", "aggregate", "download", gh_raw("stopwords-iso/stopwords-en", "472b42fd0a3fdd9c5126db073cd1e6ca67ebc83c", "stopwords-en.txt"), "commit 472b42fd, 2016-10-10", "lines", NA, "stopwords_iso_en.txt",
  "stopwords_pkg_iso", "R stopwords package, source stopwords-iso", "aggregate", "package", NA, paste("stopwords", pkg_version("stopwords")), "lines", NA, "package_exports/stopwords_pkg_stopwords-iso.txt",
  "python_stop_words_2025", "Python stop-words package 2025.11.4, English (merged list)", "aggregate", "archive", "https://files.pythonhosted.org/packages/b7/cb/27ee3d3e0b7b1169269e83331c075b2dd3c4bcc1a005821174c32a273dc4/stop_words-2025.11.4.tar.gz", "PyPI sdist 2025.11.4 (2025-11-03), sha256 0459072b...", "lines", NA, "stop_words-2025.11.4.tar.gz",
  "jockers", "R lexicon package, sw_jockers (Jockers topic-modelling stoplist)", "excluded", "package", NA, paste("lexicon", pkg_version("lexicon")), "lines", NA, "package_exports/lexicon_sw_jockers.txt",
  "sec_python_stop_words_2018", "Alir3z4/stop-words english.txt, 2018 version", "secondary", "download", gh_raw("Alir3z4/stop-words", "6742db2304", "english.txt"), "commit 6742db23, 2018-12-02", "lines", NA, "secondary/alir3z4_2018_english.txt",
  "sec_igor_smart", "igorbrigadir copy: smart.txt", "secondary", "download", igor_raw("smart.txt"), igor_commit, "lines", NA, "secondary/igor_smart.txt",
  "sec_igor_glasgow", "igorbrigadir copy: glasgow_stop_words.txt", "secondary", "download", igor_raw("glasgow_stop_words.txt"), igor_commit, "lines", NA, "secondary/igor_glasgow_stop_words.txt",
  "sec_igor_taporware", "igorbrigadir copy: taporware.txt", "secondary", "download", igor_raw("taporware.txt"), igor_commit, "lines", NA, "secondary/igor_taporware.txt",
  "sec_igor_voyant", "igorbrigadir copy: voyant_taporware.txt", "secondary", "download", igor_raw("voyant_taporware.txt"), igor_commit, "lines", NA, "secondary/igor_voyant_taporware.txt",
  "sec_igor_mallet", "igorbrigadir copy: mallet.txt", "secondary", "download", igor_raw("mallet.txt"), igor_commit, "lines", NA, "secondary/igor_mallet.txt",
  "sec_igor_weka", "igorbrigadir copy: weka.txt", "secondary", "download", igor_raw("weka.txt"), igor_commit, "lines", NA, "secondary/igor_weka.txt",
  "sec_igor_lingpipe", "igorbrigadir copy: lingpipe.txt", "secondary", "download", igor_raw("lingpipe.txt"), igor_commit, "lines", NA, "secondary/igor_lingpipe.txt",
  "sec_igor_gate", "igorbrigadir copy: gate_keyphrase.txt", "secondary", "download", igor_raw("gate_keyphrase.txt"), igor_commit, "lines", NA, "secondary/igor_gate_keyphrase.txt",
  "sec_igor_onix", "igorbrigadir copy: onix.txt", "secondary", "download", igor_raw("onix.txt"), igor_commit, "lines", NA, "secondary/igor_onix.txt",
  "sec_igor_cook", "igorbrigadir copy: cook1988_function_words.txt", "secondary", "download", igor_raw("cook1988_function_words.txt"), igor_commit, "lines", NA, "secondary/igor_cook1988_function_words.txt",
  "sec_igor_snowball", "igorbrigadir copy: snowball_original.txt", "secondary", "download", igor_raw("snowball_original.txt"), igor_commit, "lines", NA, "secondary/igor_snowball_original.txt",
  "sec_igor_spacy", "igorbrigadir copy: spacy.txt", "secondary", "download", igor_raw("spacy.txt"), igor_commit, "lines", NA, "secondary/igor_spacy.txt",
  "sec_igor_gensim", "igorbrigadir copy: gensim.txt", "secondary", "download", igor_raw("gensim.txt"), igor_commit, "lines", NA, "secondary/igor_gensim.txt",
  "sec_igor_sklearn", "igorbrigadir copy: scikitlearn.txt", "secondary", "download", igor_raw("scikitlearn.txt"), igor_commit, "lines", NA, "secondary/igor_scikitlearn.txt",
  "sec_igor_nltk", "igorbrigadir copy: nltk.txt", "secondary", "download", igor_raw("nltk.txt"), igor_commit, "lines", NA, "secondary/igor_nltk.txt",
  "sec_igor_corenlp", "igorbrigadir copy: corenlp_stopwords.txt", "secondary", "download", igor_raw("corenlp_stopwords.txt"), igor_commit, "lines", NA, "secondary/igor_corenlp_stopwords.txt",
  "sec_voyant_test_taporware", "voyanttools/trombone test copy stop.en.taporware.txt", "secondary", "download", gh_raw("voyanttools/trombone", "8b1ad4ff594933d5b553432b6f8db020f0929d74", "src/test/resources/org/voyanttools/trombone/texts/keywords/stop.en.taporware.txt"), "commit 8b1ad4ff", "lines", "#", "secondary/voyant_test_stop.en.taporware.txt",
  "sec_lingpipe_javadoc", "LingPipe 4.1 Javadoc of EnglishStopTokenizerFactory (alias-i.com, Wayback)", "secondary", "download", "http://web.archive.org/web/20220303110100id_/http://www.alias-i.com/lingpipe/docs/api/com/aliasi/tokenizer/EnglishStopTokenizerFactory.html", "Wayback snapshot 2022-03-03", "lingpipe_javadoc", NA, "secondary/lingpipe_javadoc_wayback20220303.html",
  "sec_smart_lextek_2011", "SMART stop list, lextek page Wayback 2011-08-05", "secondary", "download", "http://web.archive.org/web/20110805122456id_/http://www.lextek.com/manuals/onix/stopwords2.html", "Wayback snapshot 2011-08-05", "html_pre", NA, "secondary/lextek_stopwords2_wayback20110805.html",
  "sec_spacy_v2", "spaCy v2.0.0 stop_words.py (word block only)", "secondary", "download", gh_raw("explosion/spaCy", "v2.0.0", "spacy/lang/en/stop_words.py"), "tag v2.0.0 (2017-11)", "spacy", NA, "secondary/spacy_v2.0.0_stop_words.py"
)
# Archive members: which file inside an archive holds the list (and companion documentation).
archive_members <- tribble(
  ~list_id, ~member, ~extracted_name,
  "nltk_english", "stopwords/english", "nltk_stopwords_english.txt",
  "nltk_english", "stopwords/README", "nltk_stopwords_README.txt",
  "bow_rainbow", "bow-20020213/stopwords.c", "bow_stopwords.c",
  "bow_rainbow", "bow-20020213/README", "bow_README.txt",
  "python_stop_words_2025", "stop_words-2025.11.4/src/stop_words/stop-words/english.txt", "stop_words-2025.11.4_english.txt",
  "python_stop_words_2025", "stop_words-2025.11.4/README.rst", "stop_words-2025.11.4_README.rst"
)
stopifnot(!anyDuplicated(sources$list_id))

# Provenance and inclusion decisions (hand-maintained; the evidence column cites what was read).
provenance <- tribble(
  ~list_id, ~upstream_source, ~version_date, ~intended_use, ~documented_parent, ~family, ~is_family_root, ~inclusion_status, ~inclusion_reason, ~evidence,
  "snowball_original", "Snowball stemming project, algorithms/english/stop.txt (snowballstem.org; GitHub snowballstem/snowball-website)", "commit ce80856f, 2025-03-03; list content dates from the early 2000s", "Stop list published alongside the Snowball English stemmer for text processing / retrieval; the file's comments discuss which homonyms to keep (will, can, may, must, might, us)", "none (original list)", "snowball", TRUE, "included", "Original list behind the most widely reused English stopword family (tm, tidytext, quanteda, stopwords, NLTK via PostgreSQL)", "File header: 'An English stop word list. Comments begin with vertical bar. Each stop word is at the start of a line.' Commented-out candidates (e.g. '|will', '| us') preserved in snowball_commented_out_words.csv",
  "nltk_english", "NLTK data, packages/corpora/stopwords.zip, member stopwords/english (GitHub nltk/nltk_data, gh-pages)", "commit 98e14261, 2026-01-09 (NLTK 3.10.3 current on PyPI)", "NLTK stopwords corpus: 'high-frequency grammatical words which are usually ignored in text retrieval applications' (corpus README)", "PostgreSQL copy of the Snowball English list, augmented by NLTK (README: obtained from anoncvs.postgresql.org snowball/stopwords; 'The English list has been augmented', nltk_data issue 22)", "snowball", FALSE, "included", "Default NLTK English list; Snowball-derived with documented augmentation", "Corpus README quoted in the evidence; the R stopwords package and DKPro carry older snapshots of the same list (179 and 127 entries)",
  "stopwords_pkg_nltk", "R package stopwords (data_stopwords_nltk), built from nltk_data stopwords.zip", "stopwords 2.3 (CRAN); package documentation cites nltk_data", "Same as NLTK: stopword lists for text analysis in R", "nltk_english (older snapshot)", "snowball", FALSE, "included", "Named R implementation of the NLTK list (an older 179-entry snapshot)", "Package help: 'Stopword lists for 23 languages from the Python NLTK library', source https://github.com/nltk/nltk_data/blob/gh-pages/packages/corpora/stopwords.zip",
  "spark_ml", "Apache Spark, mllib resources org/apache/spark/ml/feature/stopwords/english.txt", "tag v4.2.0 (PyPI pyspark 4.2.0, 2026-07-14); file unchanged since it was added", "Default stop words of Spark ML StopWordsRemover", "NLTK English list (Spark's README repeats NLTK's: obtained from PostgreSQL snowball stopwords; English augmented per nltk_data issue 22)", "snowball", FALSE, "included", "ML-library default with documented NLTK lineage", "Spark README in the same directory: 'They were obtained from: http://anoncvs.postgresql.org/cvsweb.cgi/pgsql/src/backend/snowball/stopwords/ ... The English list has been augmented https://github.com/nltk/nltk_data/issues/22'",
  "dkpro", "DKPro Toolbox, dkpro.toolbox.corpus-asl resources corpus/stopwords/english.txt.gz", "commit 154eb5d6, 2013-11-15", "Stopword resource of the DKPro Toolbox NLP framework", "NLTK English list, 2013 version (README: 'Files copied from NLTK.'); measured: the Snowball list without its 50 apostrophe forms and without cannot, could, ought, would, plus can, don, just, now, s, t, will (secondary_crosschecks.csv)", "snowball", FALSE, "included_provisional", "NLP-framework copy of an early NLTK snapshot (no contractions); kept to show the lineage, adds almost no new content", "README in the stopwords directory: 'Files copied from NLTK.' followed by NLTK's own README text",
  "dl4j", "Deeplearning4j, deeplearning4j-nlp resources stopwords.txt", "commit b5f0ec07, 2019-06-06 ('Eclipse Migration Initial Commit')", "Default stop words of the Deeplearning4j NLP tokenization pipeline", "undocumented (no header); measured: contains the whole stopwords-package Snowball list (175) plus punctuation, malformed tokens and act, also, put, somebody, something, take, without", "snowball", FALSE, "included_provisional", "NLP-library default, but provenance undocumented and the file holds malformed entries (----s, \"the, -, ;, :); family assignment is by comparison only", "File inspected: 194 lines, no comment header; entries such as '----s', '\"the', '\"The', '-', ';', ':'",
  "tm_english", "R package tm, inst/stopwords/english.dat", "tm 0.7-19 (CRAN)", "stopwords('english') for R text mining (tm)", "Snowball stop.txt (tm help: lists 'from the Snowball stemmer project ... obtained from http://svn.tartarus.org/snowball/trunk/website/algorithms/*/stop.txt')", "snowball", FALSE, "included", "Named R implementation of the Snowball list", "tm help page for stopwords(); raw english.dat copied byte-for-byte from the installed package",
  "tidytext_snowball", "R package tidytext, stop_words dataset, lexicon 'snowball'", "tidytext 0.4.3 (CRAN)", "Stop words for tidy text mining in R", "tm (help: 'The snowball and SMART sets are pulled from the tm package. Note that words with non-ASCII characters have been removed.')", "snowball", FALSE, "included", "Named R implementation (copy of tm's list)", "tidytext help page for stop_words; sources listed there: lextek stopwords1.html, JMLR Lewis et al. 2004, snowball stop.txt",
  "stopwords_pkg_snowball", "R package stopwords, data_stopwords_snowball$en", "stopwords 2.3 (CRAN)", "Multilingual stopword lists for R text analysis (used by quanteda)", "Snowball project lists (help: taken from snowball_all.tgz)", "snowball", FALSE, "included", "Named R implementation; differs from tm's copy by one entry ('will'), see secondary_crosschecks.csv", "Package help for data_stopwords_snowball",
  "quanteda_snowball", "R package quanteda, stopwords() re-exported from the stopwords package", "quanteda 4.5.0 (CRAN)", "Stop words for quanteda corpus analysis", "stopwords_pkg_snowball (identical function object; quanteda help: 'Stopword lists were formerly built into quanteda, but have been moved to the stopwords package')", "snowball", FALSE, "included", "Named implementation the user asked for; an exact re-export", "identical(quanteda::stopwords, stopwords::stopwords) is TRUE in this session",
  "marimo_upstream", "Marimo multilingual stopwords collection, yaml/stopwords_en.yml (Kohei Watanabe)", "commit f3e26e37, 2020-03-27", "Cross-lingual quantitative (social-scientific) text analysis: 'Marimo extends the Snowball stopword list ... English words are extended and translated into functionally equivalent words in respective languages manually' (README)", "Snowball English list, extended (YAML comments mark additions: reporting verbs, time units, months, days, cardinal numbers, whose, many, little, less)", "snowball", FALSE, "included", "Provenance supports general text analysis; documented Snowball extension", "README of koheiw/marimo; YAML field structure preserved in marimo_upstream_fields.csv",
  "marimo_pkg", "R package stopwords, data_stopwords_marimo$en flattened by stopwords(source = 'marimo')", "stopwords 2.3 (CRAN)", "As marimo_upstream, exposed in R", "marimo_upstream (help: 'The English version was adopted from the Snowball collection, and then extended')", "snowball", FALSE, "included", "Named R implementation of the marimo list", "Package help for data_stopwords_marimo",
  "python_stop_words_2014", "Alir3z4/stop-words english.txt as first added (the list bundled by the Python 'stop-words' package until 2018)", "commit c2ba7cd7, 2014-05-06", "Stop word lists for the python stop-words package (general-purpose)", "undocumented in the repository; by comparison a copy of the Snowball list (see secondary_crosschecks.csv)", "snowball", FALSE, "included_provisional", "Provenance established only by comparison (no source statement); later versions of the same file are aggregates and are treated as comparison conditions", "Repository history: 'Add English' 2014-05-06; 'Update english.txt' 2018-12-02 replaced it by a 1,298-entry list identical to stopwords-iso; 'Merge word lists with rival lists' 2025-10-28",
  "lexicon_python", "R package lexicon, sw_python ('Python Stopword List', credited to Alireza Savand, https://pypi.org/project/stop-words/)", "lexicon 1.3.2 (CRAN)", "Stopword data for R text analysis", "python_stop_words_2014", "snowball", FALSE, "included", "Named R implementation of the Python stop-words English list (174 entries)", "lexicon help page for sw_python",
  "smart_lextek", "SMART information retrieval system stop list (Salton & Buckley, Cornell), as published in the Onix Text Retrieval Toolkit API reference 'Stop Word List 2' (lextek.com)", "Wayback snapshot 2008-02-09; page states 571 words; list dates from the SMART system (1960s-1990s)", "IR system stop list: 'built by Gerard Salton and Chris Buckley for the experimental SMART information retrieval system at Cornell University' (page text); adopted as the SMART lexicon of tm, tidytext, stopwords, quanteda", "none (original list)", "smart", TRUE, "included", "Root of the SMART family; the list the user's project currently uses", "Page header inside the list: '# Freely available stopword list generated by Chris Buckley and Gerard Salton at Cornell University.' The Cornell ftp original (ftp.cs.cornell.edu/pub/smart/english.stop) and the GNOME mirror are no longer reachable and have no Wayback capture",
  "rake_smart", "RAKE reference implementation (aneesha/RAKE) SmartStoplist.txt", "commit 7aef80f7, 2012-12-17", "Stop list for RAKE keyword extraction (Rose et al. 2010)", "SMART (header: 'stop word list from SMART (Salton,1971). Available at ftp://ftp.cs.cornell.edu/pub/smart/english.stop')", "smart", FALSE, "included", "Keyphrase-extraction implementation of SMART", "File header quoted in documented_parent",
  "tm_smart", "R package tm, inst/stopwords/SMART.dat", "tm 0.7-19 (CRAN)", "stopwords('SMART') for R text mining", "SMART list 'as documented in Appendix 11 of https://jmlr.csail.mit.edu/papers/volume5/lewis04a/' (tm help), which coincides with the MC toolkit stoplist", "smart", FALSE, "included", "Named R implementation of SMART", "tm help page for stopwords(); the JMLR appendix URL returns 404 in 2026",
  "tidytext_smart", "R package tidytext, stop_words dataset, lexicon 'SMART'", "tidytext 0.4.3 (CRAN)", "Stop words for tidy text mining", "tm_smart (pulled from tm)", "smart", FALSE, "included", "Named R implementation (copy of tm's list)", "tidytext help page for stop_words",
  "stopwords_pkg_smart", "R package stopwords, data_stopwords_smart$en", "stopwords 2.3 (CRAN)", "Stop words for R text analysis", "SMART list 'taken from the online appendix 11 of Lewis et al. (2004)' (package help)", "smart", FALSE, "included", "Named R implementation of SMART", "Package help for data_stopwords_smart",
  "quanteda_smart", "R package quanteda, stopwords(source = 'smart') re-exported from stopwords", "quanteda 4.5.0 (CRAN)", "Stop words for quanteda corpus analysis", "stopwords_pkg_smart (exact re-export)", "smart", FALSE, "included", "Named implementation the user asked for", "identical(quanteda::stopwords, stopwords::stopwords) is TRUE",
  "buckley_salton_qdap", "R package qdapDictionaries, BuckleySaltonSWL ('Buckley & Salton Stopword List')", "qdapDictionaries 1.0.7 (CRAN, 2018)", "Stopword list for the qdap discourse-analysis package", "SMART (help quotes the Onix page: 'built by Gerard Salton and Chris Buckley for the experimental SMART information retrieval system'; Note: 'Reduced from the original 571 words to 546'); measured: the SMART words with the apostrophes stripped from the contraction forms (aint, arent, cmon, couldnt ...), not a word-level reduction", "smart", FALSE, "included", "The 'Buckley-Salton' name resolves to a re-formatted copy of SMART; included as a named derivative", "qdapDictionaries help page for BuckleySaltonSWL; secondary_crosschecks.csv lists the apostrophe-stripped forms",
  "bow_rainbow", "Bow / Rainbow toolkit (Andrew McCallum, CMU), libbow stopwords.c array _bow_builtin_stopwords", "bow-20020213 tarball (2002-02-13); file copyright 1997-1998", "Built-in stoplist of the Bow toolkit for statistical text classification, retrieval and clustering (rainbow, arrow, crossbow)", "SMART (undocumented in the file; MALLET and Weka document their lists as Rainbow's); measured: exactly the SMART list without its 47 apostrophe forms (a's, ain't, can't ...)", "smart", FALSE, "included", "Parent of the MALLET and Weka lists", "stopwords.c header: 'Copyright (C) 1997, 1998 Andrew McCallum ... This file is part of the Bag-Of-Words Library, libbow'; secondary_crosschecks.csv",
  "mallet", "MALLET, stoplists/en.txt (mimno/Mallet)", "commit 23ee9bce, 2010-04-26 ('multilingual stoplists added')", "Default English stoplist of MALLET text import (topic modelling, classification): 'a standard list of English stopwords that is included in the compiled Java code ... the contents of this list are included in the Mallet distribution in the file stoplists/en.txt' (MALLET docs)", "Bow/Rainbow list (identical set, see secondary_crosschecks.csv)", "smart", FALSE, "included", "Topic-modelling toolkit default", "MALLET documentation page 'Data Import - Stopwords' (mimno.github.io/Mallet/import-stoplist)",
  "lexicon_mallet", "R package lexicon, sw_mallet ('From MAchine Learning for LanguagE Toolkit')", "lexicon 1.3.2 (CRAN)", "Stopword data for R text analysis", "mallet (523 vs 524 entries; difference listed in secondary_crosschecks.csv)", "smart", FALSE, "included", "Named R implementation of the MALLET list", "lexicon help page for sw_mallet",
  "weka_rainbow", "Weka, weka.core.stopwords.Rainbow (Waikato/weka-3.8 mirror)", "mirror commit 4cc176bb, 2016-04-11; class copyright 2014", "Default stopwords handler of Weka's StringToWordVector text filter", "Rainbow (class Javadoc: 'Stopwords list based on Rainbow: http://www.cs.cmu.edu/~mccallum/bow/rainbow/'); measured: the Bow list plus ll and ve", "smart", FALSE, "included", "Machine-learning toolkit default with documented Rainbow lineage", "Rainbow.java Javadoc quoted in documented_parent",
  "yake", "YAKE keyword extractor, yake/core/StopwordsList/stopwords_en.txt (INESCTEC/yake, formerly LIAAD/yake)", "commit bf0856c9, 2025-05-27 (PyPI yake 0.7.3)", "Stop words for YAKE unsupervised keyword extraction (Campos et al. 2020)", "undocumented in the repository; measured: exactly the SMART list plus dr, dra, mr, ms", "smart", FALSE, "included_provisional", "Keyphrase-extraction default; family assigned by comparison only", "File inspected: 575 lines beginning 'dr dra mr ms a a's able'; secondary_crosschecks.csv",
  "glasgow", "University of Glasgow Information Retrieval Group, linguistic utilities: stop_words (ir.dcs.gla.ac.uk)", "unversioned; live in 2026-09 (Wayback captures back to 2003)", "IR research resource (Glasgow IR group's 'linguistic utils'); adopted by scikit-learn, TAPoRware, Voyant, and (by comparison) gensim and spaCy", "none documented (often attributed to van Rijsbergen's Glasgow group; not verified)", "glasgow", TRUE, "included", "Root of the Glasgow family; documented parent of scikit-learn's list", "scikit-learn header cites this URL; contains collection-specific words (computer, system, thick, mill, fire, bill) and the typo fify",
  "sklearn", "scikit-learn, sklearn/feature_extraction/_stop_words.py ENGLISH_STOP_WORDS", "tag 1.9.0 (2026-06-02)", "stop_words='english' of CountVectorizer / TfidfVectorizer", "Glasgow (file header: 'This list of English stop words is taken from the \"Glasgow Information Retrieval Group\". The original list can be found at http://ir.dcs.gla.ac.uk/resources/linguistic_utils/stop_words'); measured: Glasgow without computer and the typo fify, with fifty added", "glasgow", FALSE, "included", "ML-library default with documented lineage", "File header quoted in documented_parent; secondary_crosschecks.csv",
  "taporware", "TAPoRware text-analysis tools (McMaster University), glasgowstoplist.txt", "Wayback snapshot 2018-11-05 of taporware.ualberta.ca (site defunct)", "Default stop list of the TAPoRware digital-humanities text-analysis tools", "Glasgow (file name); measured: Glasgow without 20 words (became, become, becomes, becoming, behind, below, beyond, bill, computer, cry, describe, detail, empty, interest, show, side, sincere, thick, thin, top) plus the numbers 0-100 (79 absent, 78 listed twice), the years 1990-2020, the single letters a-z and 24 punctuation marks written as regex-escaped tokens (\\?, \\*, \\( ...)", "glasgow", FALSE, "included", "Digital-humanities tool default", "Voyant's stop.en.txt header cites taporware.mcmaster.ca glasgowstoplist.txt; secondary_crosschecks.csv",
  "voyant_en", "Voyant Tools (trombone), stopwords/stop.en.txt", "commit 8b1ad4ff, 2023-04-24", "Default English stop list of Voyant Tools (web-based text analysis)", "TAPoRware (file header: 'see http://taporware.mcmaster.ca/~taporware/cgi-bin/prototype/glasgowstoplist.txt'); measured: TAPoRware plus thee, thou, thy", "glasgow", FALSE, "included", "Digital-humanities tool default with documented lineage", "File header quoted in documented_parent; identical to the test-resource copy stop.en.taporware.txt",
  "spacy", "spaCy, spacy/lang/en/stop_words.py (words block, contraction tokens, curly-apostrophe variants added at import)", "master commit 32c4b638, 2026-03-21 (PyPI spacy 3.8.16, 2026-08-24); word block unchanged since v2.0.0", "spaCy's is_stop lexical attribute for English", "undocumented in the current file; the igorbrigadir collection attributes it to 'Improved list from Stone, Denis, Kwantes (2010)' (a paragraph-similarity study), unverified here; measured: a curated variant of the Glasgow list (Jaccard 0.86 with scikit-learn's copy: drops the collection-specific words bill, cry, fire, mill, sincere, system, adds contraction tokens and a few words)", "glasgow", FALSE, "included", "NLP-library default; assigned to the Glasgow family on the measured overlap because its documented parent could not be verified (see family_review_flags.csv)", "Current file has only the comment '# Stop words'; v2.0.0 word block identical (secondary check); secondary_crosschecks.csv",
  "gensim", "gensim, gensim/parsing/preprocessing.py STOPWORDS", "tag 4.4.0 (2025-10-18)", "remove_stopwords() in gensim's preprocessing (topic-modelling library)", "undocumented in the file; the igorbrigadir collection says 'Same as spaCy'; measured: a strict superset of scikit-learn's Glasgow copy (all 318 words) plus 19 words (computer, did, didn, does, doesn, doing, don, just, kg, km, make, quite, really, regarding, say, unless, used, using, various), not the spaCy list", "glasgow", FALSE, "included", "Topic-modelling library default; family assigned on the measured containment of the scikit-learn/Glasgow list", "STOPWORDS frozenset inspected; secondary_crosschecks.csv",
  "fox_1989", "Fox, C. (1989) 'A stop list for general text', ACM SIGIR Forum 24(1-2), 19-21; machine-readable copy shipped with RAKE (aneesha/RAKE FoxStoplist.txt)", "commit 7aef80f7, 2012-12-17 (paper 1989)", "Published general-text stop list built from Brown-corpus frequencies; used by RAKE keyword extraction", "none (original list)", "fox", TRUE, "included", "Published general-text list; RAKE's copy is the only machine-readable source found; measured root of the ONIX, lsa and Pattern lists", "File header: '#From \"A stop list for general text\" Fox 1989'; the paper reports 421 words, the file holds 425 (the RAKE copy may carry additions; numbered and numbering are absent from every other copy)",
  "lsa_stopwords_en", "R package lsa, stopwords_en (Fridolin Wild)", "lsa 0.73.4 (CRAN)", "Stop list for latent semantic analysis: 'very common lists of words that want to be ignored when building up a document-term matrix' (help)", "undocumented; measured: identical to the ONIX list, i.e. Fox (1989) without numbered and numbering", "fox", FALSE, "included_provisional", "Semantic-analysis package default; family assigned by comparison only", "lsa help page for stopwords_en; secondary_crosschecks.csv",
  "pattern_clips", "Pattern web-mining module (CLiPS, De Smedt & Daelemans 2012), pattern/vector/stopwords-en.txt", "commit f660ad4a, 2013-05-15", "Stop words for pattern.vector (document vectors, LSA, clustering, classification)", "undocumented; measured: contains the whole tidytext/qdap ONIX list (Fox without single letters) plus contraction fragments and about 165 further words (aboard, alongside, amid ...)", "fox", FALSE, "included_provisional", "Text-mining toolkit default; family assigned by comparison only (extension of the Fox/ONIX list)", "Single-line comma-separated file, no header; secondary_crosschecks.csv",
  "onix_lextek", "Onix Text Retrieval Toolkit API reference, 'Stop Word List 1' (lextek.com)", "Wayback snapshot 2007-11-17; page states 429 words", "Stopword list of a commercial text-retrieval toolkit: 'probably the most widely used stopword list. It covers a wide number of stopwords without getting too aggressive and including too many words which a user might search upon' (page text); adopted by tidytext and qdap", "none documented on the page; measured: identical to Fox (1989) without numbered and numbering (423 unique entries, 6 duplicates in the page)", "fox", FALSE, "included_provisional", "Retrieval-toolkit origin and a near-copy of the published Fox list; kept as a named implementation because tidytext and qdap cite it as their source", "Page header inside the list: '# Freely available stopword list. This stopword list provides a nice balance between coverage and size.'; secondary_crosschecks.csv",
  "tidytext_onix", "R package tidytext, stop_words dataset, lexicon 'onix'", "tidytext 0.4.3 (CRAN)", "Stop words for tidy text mining", "onix_lextek (help cites lextek stopwords1.html); measured: ONIX without the single letters b-z and 'so' (404 rows, 398 unique)", "fox", FALSE, "included_provisional", "Named R implementation of ONIX (a Fox derivative)", "tidytext help page for stop_words; secondary_crosschecks.csv",
  "qdap_onix", "R package qdapDictionaries, OnixTxtRetToolkitSWL1", "qdapDictionaries 1.0.7 (CRAN)", "Stopword list for qdap discourse analysis", "onix_lextek (help Note: 'Reduced from the original 429 words to 404'); measured: identical to tidytext's onix lexicon", "fox", FALSE, "included_provisional", "Named R implementation of ONIX (a Fox derivative)", "qdapDictionaries help page for OnixTxtRetToolkitSWL1; secondary_crosschecks.csv",
  "gate_kea", "GATE plugin Keyphrase_Extraction_Algorithm, kea/StopwordsEnglish.java (KEA by Eibe Frank, University of Waikato)", "class copyright 2001; page retrieved from gate.ac.uk at run time", "Stopwords for KEA automatic keyphrase extraction as packaged in GATE", "KEA (Witten et al. 1999) - the plugin is KEA's code", "kea", TRUE, "included", "Keyphrase-extraction list with clear provenance", "Source header: 'StopwordsEnglish.java Copyright (C) 2001 Eibe Frank'; 452 put() calls",
  "corenlp_patterns", "Stanford CoreNLP, data/edu/stanford/nlp/patterns/surface/stopwords.txt", "commit ceefe814, 2016-01-22", "Stop words for CoreNLP's pattern-based entity extraction (patterns.surface, SPIED); includes punctuation tokens", "none documented; measured: closest to the Snowball family (Jaccard 0.68 with the Snowball list) with punctuation tokens and apostrophe-stripped contractions (arent, cant ...) added", "corenlp", TRUE, "included", "The general stopword file shipped with CoreNLP (the coref and acronym word lists are special-purpose and excluded); kept as its own family because it is materially different from any Snowball copy", "File inspected: 257 lines starting with punctuation tokens (!!, ?!, -lrb-, ...); family_review_flags.csv",
  "lingpipe", "LingPipe 4.1.0, com.aliasi.tokenizer.EnglishStopTokenizerFactory (built-in STOP_SET)", "LingPipe 4.1.0 (2011); source from the hvtuananh/lingpipe clone, commit 70331b9f, 2013-11-18; confirmed against the alias-i Javadoc (Wayback 2022-03-03)", "Built-in English stop list of LingPipe's tokenizer (NLP toolkit)", "none documented", "lingpipe", TRUE, "included", "NLP-toolkit default with clear provenance", "Javadoc: 'The built-in stoplist consists of the following words: ...' (76 words); identical to the clone's source",
  "choi_c99", "Stop list of Freddy Choi's C99 / TextTiling segmentation implementations (Choi 2000), preserved in igorbrigadir/stopwords en/choi_2000naacl.txt", "igorbrigadir commit ab85d86c, 2019-08-20; original code (Google Code archive uima-text-segmenter) not machine-readable", "Stop list used by text-segmentation algorithms (topic boundary detection)", "none documented", "c99", TRUE, "included_provisional", "Text-analysis use is documented, but only a secondary copy exists", "igorbrigadir table: 'UIMA wrapper for the java implementations of the segmentation algorithms C99 and TextTiling, written by Freddy Choi'",
  "cook_1988", "Vivian Cook, 'List of English structure (function) words' (viviancook.uk/Words/StructureWordsList.htm)", "Wayback snapshot 2019-12-25; list compiled for Cook (1988)", "Function-word list 'compiled for practical purposes some time ago as data for a computer parser for student English' (page text); applied linguistics", "none (original list)", "cook", TRUE, "included_provisional", "A linguistic function-word list rather than a stoplist built for text mining; kept provisionally as a general-English reference for the user's review", "Page text quoted in intended_use; IPA weak forms removed by the parser (documented in list_summary.csv)",
  "qdap_function_words", "R package qdapDictionaries, function.words ('function words from John and Muriel Higgins's list used for the text game ECLIPSE', augmented with contractions)", "qdapDictionaries 1.0.7 (CRAN)", "Function-word list distributed with the qdap discourse-analysis package", "Higgins & Higgins ECLIPSE function-word list (language-teaching game), plus qdap's contractions table", "higgins", TRUE, "included_provisional", "Function-word list from a language-teaching context, exposed by an R text-analysis package; provisional", "qdapDictionaries help page for function.words",
  "okapi_framework", "Okapi Framework (localisation toolkit), steps/termextraction resource stopWords_en.txt", "bitbucket master, retrieved at run time (no version pin available without authentication)", "Stop words for the Okapi Framework term-extraction step", "none documented", "okapi_framework", TRUE, "included_provisional", "Term extraction is close to concept extraction; provenance thin (unpinned, no source statement)", "File header: '# Stop words default list for English'; not the Okapi BM25 search engine",
  "stopwords_iso", "stopwords-iso/stopwords-en, stopwords-en.txt", "commit 472b42fd, 2016-10-10", "'The most comprehensive collection of stopwords for the english language' (README): an aggregate of 30+ sources including SEO sites, Lextek Onix, Snowball, Glasgow, PubMed", "aggregate of many lists (CREDITS.md of stopwords-iso)", "aggregate", FALSE, "aggregate_condition", "Aggregate comparison condition, not an independent source family", "CREDITS.md lists Link Assistant, Lextek Onix, Snowball Tartarus, Webconfs, 99Webtools, Glasgow, SEO Book, NCBI, Ranks NL, ...",
  "stopwords_pkg_iso", "R package stopwords, data_stopwords_stopwordsiso$en", "stopwords 2.3 (CRAN)", "As stopwords_iso, exposed in R", "stopwords_iso", "aggregate", FALSE, "aggregate_condition", "Aggregate comparison condition", "Package help for data_stopwords_stopwordsiso",
  "python_stop_words_2025", "Python stop-words package 2025.11.4 (PyPI sdist), src/stop_words/stop-words/english.txt", "PyPI release 2025-11-03; upstream commit b338be6c 'Merge word lists with rival lists' 2025-10-28", "Stop word lists for the python stop-words package", "merge of the 2018 list (a stopwords-iso copy) with 'rival lists' (commit message); README: 'Stop word lists compiled from various open sources'", "aggregate", FALSE, "aggregate_condition", "Provenance of the merged list is not itemised; treated as an aggregate condition like stopwords-iso", "PyPI README credits and the repository commit history",
  "jockers", "R package lexicon, sw_jockers ('Matthew Jocker's Expanded Topic Modeling Stopword List')", "lexicon 1.3.2 (CRAN); Jockers 2013 (Macroanalysis; UWM 2013 materials)", "Stoplist for topic modelling 19th-century novels, expanded with character names", "none documented", "excluded", FALSE, "excluded_acquired", "Topic-modelling origin, but corpus-specific: 5,902 entries dominated by first names (aaron, abbey, abbie, ...); not a general-English list", "lexicon help page for sw_jockers; entries inspected"
)
stopifnot(setequal(provenance$list_id, sources$list_id[sources$kind != "secondary"]))

# Candidates investigated but not acquired.
excluded_candidates <- tribble(
  ~candidate, ~source_url, ~intended_use, ~exclusion_reason, ~evidence,
  "Lucene / Solr / Elasticsearch EnglishAnalyzer stop set (33 words)", "https://github.com/apache/lucene/blob/main/lucene/analysis/common/src/java/org/apache/lucene/analysis/en/EnglishAnalyzer.java", "Search-engine index analysis", "Search-engine indexing list", "igorbrigadir table: 'common English words that are not usually useful for searching'; also lexicon::sw_lucene",
  "MySQL InnoDB (36) and MyISAM (543) full-text stopword tables", "https://dev.mysql.com/doc/refman/8.0/en/innodb-ft-default-stopword-table.html", "Database full-text indexing", "Database indexing list (MyISAM's is a modified SMART)", "MySQL reference manual",
  "PostgreSQL english.stop (127)", "https://www.postgresql.org/docs/current/textsearch-dictionaries.html", "Database full-text search", "Database indexing list; it is the Snowball list without contractions and is the documented ancestor of NLTK's list", "PostgreSQL documentation; NLTK README",
  "MongoDB stop_words_english.txt (174)", "https://github.com/mongodb/mongo/blob/master/src/mongo/db/fts/stop_words_english.txt", "Database full-text search", "Database indexing copy of Snowball", "igorbrigadir table: 'Changed stop words files to the snowball stop lists'",
  "Xapian english.txt (174)", "https://github.com/xapian/xapian/blob/master/xapian-core/languages/stopwords/english.txt", "Search-engine library", "Search-engine copy of Snowball", "igorbrigadir table",
  "Sphinx / Sphinx Search Ultimate (Mirasvit) stopwords", "http://sphinxsearch.com/docs/current.html#conf-stopwords", "Search server", "Search-engine list", "igorbrigadir table",
  "Terrier stopword list (733)", "http://terrier.org/docs/v4.1/javadoc/org/terrier/terms/Stopwords.html", "IR platform", "Search-engine / IR-platform list", "igorbrigadir table",
  "Zettair stop list (469)", "http://www.seg.rmit.edu.au/zettair/", "Text search engine", "Search-engine list", "igorbrigadir table",
  "ATIRE stop_word.c (Puurula 988; NCBI Medline 313)", "http://www.atire.org/", "IR engine; Medline retrieval", "Search-engine list; biomedical retrieval list", "igorbrigadir table",
  "Okapi BM25 GSL stop lists (cacm 108/339; sample 222/474)", "http://www.staff.city.ac.uk/~andym/OKAPI-PACK/appendix-d.html", "IR system (Okapi BM25)", "Search-engine list, collection-specific (CACM)", "igorbrigadir table",
  "Galago / Indri / Lemur INQUERY stoplist (418) and rmstop (565)", "https://sourceforge.net/p/lemur/", "IR research engines", "Search-engine lists", "igorbrigadir table",
  "Ranks NL default (174), large (667), old Google (32)", "http://www.ranks.nl/stopwords", "SEO page analyser", "SEO list (default is a Snowball copy)", "igorbrigadir table",
  "TextFixer (119), 99webtools (183), xpo6 (319), SEO Book, Webconfs, Link Assistant", "http://www.textfixer.com/resources/common-english-words.txt", "SEO / web utilities", "SEO lists (xpo6 is a Glasgow copy with typos)", "igorbrigadir table; stopwords-iso CREDITS",
  "Azure Gallery stopword list (310)", "https://gallery.azure.ai/Experiment/How-to-modify-default-stopword-list-1", "Cloud ML demo", "Demo copy ('slightly modified glasgow list'), not an established list", "igorbrigadir table",
  "bbalet Go stopwords (317), Kevin Bouge lists (571), tonybsk_1/6, t101 minimal (85), vw_lda (83), DataScienceDojo (250)", "https://github.com/igorbrigadir/stopwords", "Library / blog / demo lists", "Copies of Glasgow or SMART, unknown origin, or demo lists without an established provenance", "igorbrigadir table ('Unknown origin - I lost the reference' for tonybsk)",
  "PubMed (133), Ovid (39), EBSCOhost MEDLINE/CINAHL (24), Reuters Web of Science (211)", "https://www.ncbi.nlm.nih.gov/books/NBK3827/", "Bibliographic / biomedical database search", "Biomedical or bibliographic retrieval lists", "igorbrigadir table",
  "LexisNexis noise words (100)", "http://help.lexisnexis.com/", "Legal / patent database search", "Database search list", "igorbrigadir table",
  "ROUGE 1.5.5 smart_common_words.txt (598)", "https://github.com/andersjo/pyrouge/blob/master/tools/ROUGE-1.5.5/data/smart_common_words.txt", "Summarisation evaluation", "Summarisation-evaluation list (extended SMART)", "igorbrigadir table",
  "Loughran-McDonald stopword lists (lexicon::sw_loughran_mcdonald_long/short)", "https://sraf.nd.edu/textual-analysis/", "Financial text analysis", "Finance-specific lists", "lexicon package documentation",
  "Fry instant words (lexicon::sw_fry_25/100/200/1000; qdapDictionaries Top25/100/200Words, Fry_1000) and Dolch sight words (lexicon::sw_dolch, qdapDictionaries Dolch)", "https://cran.r-project.org/package=lexicon", "Reading instruction (sight-word lists)", "Educational frequency lists, not stopword lists (documented as Fry 1997 / Dolch reading lists)", "lexicon and qdapDictionaries help pages",
  "Stanford CoreNLP hard-coded coreference stop set (WordLists.java, 28) and AcronymMatcher stop set (150)", "https://github.com/stanfordnlp/CoreNLP", "Coreference resolution; acronym matching", "Special-purpose sets inside specific components, not general stopword lists", "igorbrigadir table; CoreNLP source",
  "Voyant alternative lists stop.en.glasgow.txt and stop.en.smart.txt; Voyant name lists (stop.en.people, stop.en.common-names)", "https://github.com/voyanttools/trombone", "Optional Voyant stop lists", "Copies of Glasgow and SMART already represented; name lists are not stopword lists", "trombone repository tree",
  "Snowball 'expanded' list (commented-out words in stop.txt)", "https://snowballstem.org/algorithms/english/stop.txt", "Words the Snowball authors discuss but exclude", "Not a published list; the commented-out words are preserved separately in snowball_commented_out_words.csv", "stop.txt comments",
  "Stone, Denis & Kwantes (2010) list (attributed parent of spaCy's list)", "n/a", "Paragraph-similarity study", "No machine-readable copy found; documented only as an attribution", "igorbrigadir table",
  "van Rijsbergen (1979) stop list", "n/a", "IR textbook list", "No machine-readable copy found; mentioned as a possible ancestor of the Glasgow and NLTK lists", "igorbrigadir NLTK note",
  "Alir3z4/stop-words english.txt, 2018 version (1,298)", "https://github.com/Alir3z4/stop-words", "Python stop-words package", "Acquired as a secondary copy only: identical in size to stopwords-iso (checked in secondary_crosschecks.csv), hence an aggregate", "Repository history"
)

# ---- Acquisition ----
# INPUT : the registry; the network (first run) or the raw/ cache (later runs).
# DOES  : downloads every URL byte-for-byte (curl, mode wb) into raw/ unless the file already exists;
#         extracts archive members; exports lists from the installed R packages; hashes everything.
# OUTPUT: acquisition_log (one row per raw file).
log_line("acquiring sources")
acq_rows <- list()
record_acq <- function(list_id, file_rel, url, pin, status, cache_hit) {
  path <- file.path(raw_dir, file_rel)
  acq_rows[[length(acq_rows) + 1L]] <<- tibble(list_id = list_id, file = file_rel, url = ifelse(is.na(url), "", url), pin = pin, status = status,
    cache_hit = cache_hit, retrieved_at = if (file.exists(path)) format(file.info(path)$mtime, "%Y-%m-%dT%H:%M:%SZ", tz = "UTC") else NA_character_,
    bytes = if (file.exists(path)) file.size(path) else NA_real_, sha256 = if (file.exists(path)) sha256(path) else NA_character_)
}
download_raw <- function(url, dest) {
  if (file.exists(dest) && file.size(dest) > 0) return(list(status = "cached", cache_hit = TRUE))
  h <- new_handle(useragent = user_agent, followlocation = TRUE, timeout = 300)
  st <- tryCatch({ curl_download(url, dest, mode = "wb", handle = h); "downloaded" }, error = function(e) { if (file.exists(dest)) file.remove(dest); paste("ERROR:", conditionMessage(e)) })
  list(status = st, cache_hit = FALSE)
}
extract_member <- function(archive, member, dest) {
  if (file.exists(dest) && file.size(dest) > 0) return("cached")
  tmp <- tempfile("extract_"); dir.create(tmp)
  if (grepl("\\.zip$", archive)) unzip(archive, files = member, exdir = tmp, junkpaths = FALSE) else untar(archive, files = member, exdir = tmp)
  src <- file.path(tmp, member)
  if (!file.exists(src)) stop("member not found in archive: ", member)
  file.copy(src, dest, overwrite = FALSE); "extracted"
}
for (i in seq_len(nrow(sources))) {
  s <- sources[i, ]
  if (s$acquisition %in% c("download", "archive")) {
    dest <- file.path(raw_dir, s$file_name)
    res  <- download_raw(s$url, dest)
    record_acq(s$list_id, s$file_name, s$url, s$pin, res$status, res$cache_hit)
    if (!file.exists(dest)) { if (s$kind == "secondary") next else stop("Primary source could not be acquired: ", s$list_id, " (", res$status, ")") }
    if (s$acquisition == "archive") {
      for (j in which(archive_members$list_id == s$list_id)) {
        m <- archive_members[j, ]
        st <- extract_member(dest, m$member, file.path(raw_dir, m$extracted_name))
        record_acq(s$list_id, m$extracted_name, s$url, paste0(s$pin, "; member ", m$member), st, st == "cached")
      }
    }
  }
}
# Package exports: byte-exact package files where they exist, otherwise one entry per line as the package returns them.
write_export <- function(x, file_rel) {
  dest <- file.path(raw_dir, file_rel)
  writeLines(enc2utf8(x), dest, useBytes = TRUE)
  dest
}
tm_dir <- system.file("stopwords", package = "tm")
file.copy(file.path(tm_dir, "english.dat"), file.path(pkg_dir, "tm_english.dat"), overwrite = TRUE)
file.copy(file.path(tm_dir, "SMART.dat"),   file.path(pkg_dir, "tm_SMART.dat"),   overwrite = TRUE)
write_export(stopwords::stopwords("en", source = "nltk"),          "package_exports/stopwords_pkg_nltk.txt")
write_export(stopwords::stopwords("en", source = "snowball"),      "package_exports/stopwords_pkg_snowball.txt")
write_export(stopwords::stopwords("en", source = "smart"),         "package_exports/stopwords_pkg_smart.txt")
write_export(stopwords::stopwords("en", source = "marimo"),        "package_exports/stopwords_pkg_marimo.txt")
write_export(stopwords::stopwords("en", source = "stopwords-iso"), "package_exports/stopwords_pkg_stopwords-iso.txt")
write_export(quanteda::stopwords("en", source = "snowball"),       "package_exports/quanteda_snowball.txt")
write_export(quanteda::stopwords("en", source = "smart"),          "package_exports/quanteda_smart.txt")
sw <- tidytext::stop_words
write_export(sw$word[sw$lexicon == "snowball"], "package_exports/tidytext_snowball.txt")
write_export(sw$word[sw$lexicon == "SMART"],    "package_exports/tidytext_SMART.txt")
write_export(sw$word[sw$lexicon == "onix"],     "package_exports/tidytext_onix.txt")
write_csv(sw, file.path(pkg_dir, "tidytext_stop_words.csv"))
write_export(lexicon::sw_python,  "package_exports/lexicon_sw_python.txt")
write_export(lexicon::sw_mallet,  "package_exports/lexicon_sw_mallet.txt")
write_export(lexicon::sw_jockers, "package_exports/lexicon_sw_jockers.txt")
write_export(qdapDictionaries::BuckleySaltonSWL,     "package_exports/qdap_BuckleySaltonSWL.txt")
write_export(qdapDictionaries::OnixTxtRetToolkitSWL1, "package_exports/qdap_OnixTxtRetToolkitSWL1.txt")
write_export(qdapDictionaries::function.words,        "package_exports/qdap_function.words.txt")
write_export(lsa::stopwords_en, "package_exports/lsa_stopwords_en.txt")
for (i in which(sources$acquisition == "package")) record_acq(sources$list_id[i], sources$file_name[i], NA, sources$pin[i], "exported from installed package", FALSE)
record_acq("tidytext_all", "package_exports/tidytext_stop_words.csv", NA, paste("tidytext", pkg_version("tidytext")), "exported from installed package", FALSE)
acquisition_log <- bind_rows(acq_rows)
add_check("acquisition", "every primary (non-secondary) raw file exists and is non-empty",
          sprintf("%d files", sum(!grepl("^secondary/", acquisition_log$file))),
          all(!is.na(acquisition_log$bytes[!grepl("^secondary/", acquisition_log$file)]) & acquisition_log$bytes[!grepl("^secondary/", acquisition_log$file)] > 0))
add_check("acquisition", "no download error recorded for a primary source", sum(grepl("^ERROR", acquisition_log$status) & !grepl("^secondary/", acquisition_log$file)), !any(grepl("^ERROR", acquisition_log$status) & !grepl("^secondary/", acquisition_log$file)))

# ---- Parsers ----
# INPUT : a raw file. OUTPUT: the entries in file order, exactly as written (comment lines, blank lines,
#         code syntax and HTML removed; nothing lower-cased, trimmed or deduplicated here).
read_utf8 <- function(path) { x <- readLines(path, encoding = "UTF-8", warn = FALSE); x <- stri_replace_all_regex(x, "\\r$", ""); stri_replace_first_regex(x, "^\\uFEFF", "") }
parse_lines <- function(path, comment = NA) {
  x <- read_utf8(path); x <- x[stri_trim_both(x) != ""]
  # a comment line starts with the marker and carries text after it; a line holding only the marker (e.g. "#") is an entry
  if (!is.na(comment)) { t <- stri_trim_both(x); x <- x[!(stri_startswith_fixed(t, comment) & stri_length(t) > stri_length(comment))] }
  x
}
parse_gz_lines <- function(path, comment = NA) { con <- gzfile(path, encoding = "UTF-8"); x <- readLines(con, warn = FALSE); close(con); x <- stri_replace_all_regex(x, "\\r$", ""); x[stri_trim_both(x) != ""] }
parse_snowball <- function(path) { x <- read_utf8(path); w <- stri_trim_both(stri_replace_first_regex(x, "\\|.*$", "")); w[w != ""] }
snowball_commented_words <- function(path) {
  x <- read_utf8(path); out <- list()
  for (k in which(stri_detect_regex(x, "^\\s*\\|"))) {
    parts <- stri_split_fixed(stri_replace_first_regex(x[k], "^\\s*\\|", ""), "|")[[1]]
    toks  <- stri_split_regex(stri_trim_both(parts[1]), "\\s+")[[1]]; toks <- toks[toks != ""]
    ok <- length(toks) >= 1 && all(stri_detect_regex(toks, "^[a-z']+$")) && (length(toks) == 1 || all(stri_detect_fixed(toks, "'")))
    if (ok) out[[length(out) + 1L]] <- tibble(line = k, word = toks, note = if (length(parts) > 1) stri_trim_both(paste(parts[-1], collapse = "|")) else "", raw_line = x[k])
  }
  bind_rows(out)
}
parse_spacy <- function(path) {
  txt   <- paste(read_utf8(path), collapse = "\n")
  block <- stri_match_first_regex(txt, "(?s)\"\"\"(.*?)\"\"\"")[, 2]
  words <- stri_split_regex(block, "\\s+")[[1]]; words <- words[words != ""]
  cline <- stri_extract_first_regex(txt, "contractions\\s*=\\s*\\[[^\\]]*\\]")
  contr <- if (is.na(cline)) character() else stri_match_all_regex(cline, "\"([^\"]*)\"")[[1]][, 2]
  aline <- stri_extract_first_regex(txt, "for apostrophe in \\[[^\\]]*\\]")
  apos  <- if (is.na(aline)) character() else stri_match_all_regex(aline, "\"([^\"]*)\"")[[1]][, 2]
  variants <- unlist(lapply(apos, function(a) stri_replace_all_fixed(contr, "'", a)))
  c(words, contr, variants)
}
parse_frozenset <- function(path, varname) {
  txt   <- paste(read_utf8(path), collapse = "\n")
  block <- stri_extract_first_regex(txt, paste0("(?s)", varname, "\\s*=\\s*frozenset\\(.*?\\]\\s*\\)"))
  m <- stri_match_all_regex(block, "'([^']*)'|\"([^\"]*)\"")[[1]]
  ifelse(is.na(m[, 2]), m[, 3], m[, 2])
}
parse_java_add <- function(path, pattern) { txt <- paste(read_utf8(path), collapse = "\n"); stri_match_all_regex(txt, pattern)[[1]][, 2] }
parse_c_array <- function(path) {
  txt   <- paste(read_utf8(path), collapse = "\n")
  block <- stri_extract_first_regex(txt, "(?s)_bow_builtin_stopwords\\[\\]\\s*=\\s*\\{.*?\\};")
  stri_match_all_regex(block, "\"([^\"]*)\"")[[1]][, 2]
}
parse_html_pre <- function(path) {
  doc <- read_html(path, encoding = "UTF-8"); txt <- xml_text(xml_find_first(doc, "//pre"))
  x <- stri_split_regex(txt, "\\r?\\n")[[1]]; x <- x[!stri_startswith_fixed(stri_trim_both(x), "#")]
  toks <- stri_split_regex(paste(x, collapse = " "), "\\s+")[[1]]; toks[toks != ""]
}
parse_gate_html <- function(path) { doc <- read_html(path, encoding = "UTF-8"); txt <- xml_text(xml_find_first(doc, "//pre")); stri_match_all_regex(txt, "m_Stopwords\\.put\\(\"([^\"]*)\"")[[1]][, 2] }
parse_lingpipe_javadoc <- function(path) {
  doc <- read_html(path, encoding = "UTF-8"); txt <- stri_replace_all_regex(xml_text(doc), "\\s+", " ")
  seg <- stri_match_first_regex(txt, "consists of the following words:\\s*(.*?)\\s*Note that the stoplist entries")[, 2]
  stri_trim_both(stri_split_fixed(seg, ",")[[1]])
}
parse_cook_html <- function(path) {
  doc <- read_html(path, encoding = "UTF-8"); txt <- stri_replace_all_regex(xml_text(doc), "\\s+", " ")
  seg <- stri_match_first_regex(txt, "click PRINT\\s*(.*?)\\s*Content and structure words")[, 2]
  seg <- stri_replace_all_regex(seg, "\\([^()]*\\)", " ")                      # IPA weak forms and glosses in parentheses
  toks <- stri_split_regex(seg, "\\s+")[[1]]
  toks[stri_detect_regex(toks, "^[A-Za-z']+$")]                                  # drop stray IPA fragments (contain / : ~)
}
flatten_yaml <- function(x, path = character()) {
  if (is.character(x) || is.numeric(x) || is.logical(x)) return(tibble(field = paste(path, collapse = "/"), entry = as.character(x)))
  if (is.list(x)) { nms <- names(x); if (is.null(nms)) return(bind_rows(lapply(x, flatten_yaml, path = path))); return(bind_rows(lapply(seq_along(x), function(i) flatten_yaml(x[[i]], c(path, nms[i]))))) }
  tibble()
}
parse_marimo_yaml <- function(path) flatten_yaml(read_yaml(path))$entry
parse_comma <- function(path) { txt <- paste(read_utf8(path), collapse = " "); x <- stri_trim_both(stri_split_fixed(txt, ",")[[1]]); x[x != ""] }
list_file_of <- function(list_id) {
  s <- sources[sources$list_id == list_id, ]
  f <- switch(list_id, nltk_english = "nltk_stopwords_english.txt", bow_rainbow = "bow_stopwords.c", python_stop_words_2025 = "stop_words-2025.11.4_english.txt", s$file_name)
  file.path(raw_dir, f)
}
parse_source <- function(list_id) {
  s <- sources[sources$list_id == list_id, ]; p <- list_file_of(list_id)
  switch(s$parser,
    lines = parse_lines(p, s$parser_arg), gz_lines = parse_gz_lines(p), snowball = parse_snowball(p), spacy = parse_spacy(p),
    frozenset = parse_frozenset(p, s$parser_arg), java_add = parse_java_add(p, s$parser_arg), c_array = parse_c_array(p),
    html_pre = parse_html_pre(p), gate_html = parse_gate_html(p), lingpipe_javadoc = parse_lingpipe_javadoc(p),
    cook_html = parse_cook_html(p), marimo_yaml = parse_marimo_yaml(p), comma = parse_comma(p), stop("unknown parser ", s$parser))
}
log_line("parsing")
raw_entries <- bind_rows(lapply(sources$list_id, function(id) {
  if (!file.exists(list_file_of(id))) return(tibble())
  e <- parse_source(id); tibble(list_id = id, position = seq_along(e), raw_entry = e)
}))
add_check("parsing", "every acquired source yields at least one entry", n_distinct(raw_entries$list_id), setequal(unique(raw_entries$list_id), sources$list_id[file.exists(vapply(sources$list_id, list_file_of, character(1)))]))
snowball_commented <- snowball_commented_words(list_file_of("snowball_original"))
marimo_fields      <- flatten_yaml(read_yaml(list_file_of("marimo_upstream"))) |> mutate(position = row_number())

# Preserve every extracted list separately (entries as found, file order, no curation).
for (id in unique(raw_entries$list_id)) writeLines(enc2utf8(raw_entries$raw_entry[raw_entries$list_id == id]), file.path(lists_dir, paste0(id, ".txt")), useBytes = TRUE)

# ---- Normalisation (formatting only) ----
# DOES  : NFC; curly apostrophes (U+2018, U+2019, U+02BC, U+2032) to '; trim; internal whitespace to one
#         space; lower case; drop entries that become empty; keep the first of within-list duplicates.
#         Every change is logged per entry. No character is removed, no punctuation stripped.
# OUTPUT: entries (one row per raw entry with its normalised form), members (unique normalised entries per list).
log_line("normalising")
normalise_table <- function(df) {
  e0 <- df$raw_entry
  e1 <- stri_trans_nfc(e0)
  e2 <- stri_replace_all_regex(e1, "[\\u2018\\u2019\\u02BC\\u2032]", "'")
  e3 <- stri_trim_both(e2)
  e4 <- stri_replace_all_regex(e3, "\\s+", " ")
  e5 <- stri_trans_tolower(e4)
  chg <- paste0(ifelse(e1 != e0, "nfc;", ""), ifelse(e2 != e1, "apostrophe;", ""), ifelse(e3 != e2, "trim;", ""), ifelse(e4 != e3, "whitespace;", ""), ifelse(e5 != e4, "case;", ""))
  df |> mutate(entry = e5, changes = stri_replace_last_fixed(chg, ";", "")) |>
    group_by(list_id) |> mutate(dropped = case_when(entry == "" ~ "empty", duplicated(entry) ~ "duplicate_within_list", TRUE ~ "")) |> ungroup()
}
entries <- normalise_table(raw_entries)
entry_class <- function(e) case_when(
  stri_detect_regex(e, "\\s") ~ "multiword",
  stri_detect_regex(e, "^[a-z]+$") ~ "word",
  stri_detect_regex(e, "^[a-z]+(-[a-z]+)+$") ~ "hyphenated",
  stri_detect_regex(e, "^[a-z']+$") ~ "apostrophe",
  stri_detect_regex(e, "^[0-9]+$") ~ "numeric",
  stri_detect_regex(e, "[a-z]") & stri_detect_regex(e, "[0-9]") ~ "alphanumeric",
  !stri_detect_regex(e, "[a-z0-9]") ~ "punctuation_or_symbol",
  TRUE ~ "other_characters")
normalisation_log <- entries |> filter(changes != "" | dropped != "") |> select(list_id, position, raw_entry, entry, changes, dropped)
members <- entries |> filter(dropped == "") |> mutate(entry_class = entry_class(entry)) |> select(list_id, entry, entry_class)
add_check("normalisation", "no empty normalised entry survives", sum(members$entry == ""), all(members$entry != ""))
add_check("normalisation", "entries are unique within each list", nrow(members) - nrow(distinct(members, list_id, entry)), !anyDuplicated(members[, c("list_id", "entry")]))

# ---- List summary ----
sizes <- members |> count(list_id, name = "n_unique_entries")
list_summary <- sources |> select(list_id, list_name, kind, pin, parser, file_name) |>
  left_join(raw_entries |> count(list_id, name = "n_raw_entries"), by = "list_id") |>
  left_join(sizes, by = "list_id") |>
  left_join(entries |> group_by(list_id) |> summarise(n_duplicates_within = sum(dropped == "duplicate_within_list"), n_case_changed = sum(stri_detect_fixed(changes, "case")),
            n_apostrophe_changed = sum(stri_detect_fixed(changes, "apostrophe")), n_trimmed = sum(stri_detect_fixed(changes, "trim")), .groups = "drop"), by = "list_id") |>
  left_join(members |> count(list_id, entry_class) |> pivot_wider(names_from = entry_class, values_from = n, values_fill = 0L, names_prefix = "n_"), by = "list_id") |>
  left_join(provenance |> select(list_id, family, is_family_root, inclusion_status), by = "list_id") |>
  mutate(inclusion_status = ifelse(kind == "secondary", "secondary_copy", inclusion_status), formatting_only_changes = paste0(
    ifelse(n_case_changed > 0, sprintf("lower-cased %d; ", n_case_changed), ""), ifelse(n_apostrophe_changed > 0, sprintf("curly apostrophe to ' in %d; ", n_apostrophe_changed), ""),
    ifelse(n_trimmed > 0, sprintf("trimmed %d; ", n_trimmed), ""), ifelse(n_duplicates_within > 0, sprintf("%d within-list duplicate(s) collapsed; ", n_duplicates_within), ""),
    "parser: ", parser, ifelse(is.na(sources$parser_arg[match(list_id, sources$list_id)]), "", paste0(" (", sources$parser_arg[match(list_id, sources$list_id)], ")")))) |>
  arrange(kind, desc(n_unique_entries))

# ---- Cross-checks between primary lists, package copies and secondary copies ----
# DOES  : set equality and differences for pairs whose identity is claimed or expected.
set_of <- function(id) members$entry[members$list_id == id]
compare_sets <- function(a, b, claim) {
  A <- set_of(a); B <- set_of(b)
  tibble(list_a = a, list_b = b, claim = claim, n_a = length(A), n_b = length(B), identical_sets = setequal(A, B),
         only_in_a = paste(sort(setdiff(A, B)), collapse = " "), only_in_b = paste(sort(setdiff(B, A)), collapse = " "))
}
crosschecks <- bind_rows(
  compare_sets("tm_english", "snowball_original", "tm english.dat is the Snowball default list"),
  compare_sets("tidytext_snowball", "tm_english", "tidytext snowball is pulled from tm"),
  compare_sets("stopwords_pkg_snowball", "snowball_original", "stopwords package snowball vs the Snowball file"),
  compare_sets("quanteda_snowball", "stopwords_pkg_snowball", "quanteda re-exports the stopwords package"),
  compare_sets("sec_igor_snowball", "snowball_original", "igorbrigadir snowball_original copy"),
  compare_sets("stopwords_pkg_nltk", "nltk_english", "stopwords package nltk snapshot vs current NLTK"),
  compare_sets("sec_igor_nltk", "nltk_english", "igorbrigadir 2019 NLTK copy vs current NLTK"),
  compare_sets("spark_ml", "nltk_english", "Spark english.txt vs current NLTK"),
  compare_sets("spark_ml", "stopwords_pkg_nltk", "Spark english.txt vs the stopwords package NLTK snapshot"),
  compare_sets("dkpro", "nltk_english", "DKPro (copied from NLTK, 2013) vs current NLTK"),
  compare_sets("python_stop_words_2014", "snowball_original", "python stop-words 2014 English list vs Snowball"),
  compare_sets("lexicon_python", "python_stop_words_2014", "lexicon sw_python vs the 2014 python stop-words list"),
  compare_sets("sec_python_stop_words_2018", "stopwords_iso", "python stop-words 2018 English list vs stopwords-iso"),
  compare_sets("marimo_pkg", "marimo_upstream", "stopwords package marimo vs upstream YAML"),
  compare_sets("tm_smart", "smart_lextek", "tm SMART.dat vs the lextek SMART page"),
  compare_sets("tidytext_smart", "tm_smart", "tidytext SMART is pulled from tm"),
  compare_sets("stopwords_pkg_smart", "smart_lextek", "stopwords package smart vs the lextek SMART page"),
  compare_sets("quanteda_smart", "stopwords_pkg_smart", "quanteda re-exports the stopwords package"),
  compare_sets("rake_smart", "smart_lextek", "RAKE SmartStoplist vs the lextek SMART page"),
  compare_sets("sec_igor_smart", "smart_lextek", "igorbrigadir smart copy vs the lextek SMART page"),
  compare_sets("sec_smart_lextek_2011", "smart_lextek", "lextek SMART page 2011 snapshot vs 2008 snapshot"),
  compare_sets("buckley_salton_qdap", "smart_lextek", "qdap Buckley-Salton is a reduced SMART"),
  compare_sets("mallet", "bow_rainbow", "MALLET en.txt vs the Bow built-in list"),
  compare_sets("sec_igor_mallet", "mallet", "igorbrigadir mallet copy"),
  compare_sets("lexicon_mallet", "mallet", "lexicon sw_mallet vs MALLET en.txt"),
  compare_sets("weka_rainbow", "bow_rainbow", "Weka Rainbow vs the Bow built-in list"),
  compare_sets("sec_igor_weka", "weka_rainbow", "igorbrigadir weka copy"),
  compare_sets("bow_rainbow", "smart_lextek", "Bow built-in list vs SMART"),
  compare_sets("yake", "smart_lextek", "YAKE vs SMART"),
  compare_sets("sklearn", "glasgow", "scikit-learn vs the Glasgow list"),
  compare_sets("sec_igor_sklearn", "sklearn", "igorbrigadir scikit-learn copy"),
  compare_sets("sec_igor_glasgow", "glasgow", "igorbrigadir glasgow copy"),
  compare_sets("taporware", "glasgow", "TAPoRware vs the Glasgow list"),
  compare_sets("sec_igor_taporware", "taporware", "igorbrigadir taporware copy"),
  compare_sets("voyant_en", "taporware", "Voyant stop.en.txt vs TAPoRware"),
  compare_sets("sec_voyant_test_taporware", "voyant_en", "Voyant test-resource copy vs stop.en.txt"),
  compare_sets("sec_igor_voyant", "voyant_en", "igorbrigadir voyant copy"),
  compare_sets("sec_igor_spacy", "spacy", "igorbrigadir spacy copy (2019) vs current"),
  compare_sets("sec_spacy_v2", "spacy", "spaCy v2.0.0 word block vs current runtime set"),
  compare_sets("sec_igor_gensim", "gensim", "igorbrigadir gensim copy (2019) vs 4.4.0"),
  compare_sets("gensim", "spacy", "gensim vs spaCy (collection claims 'same as spaCy')"),
  compare_sets("gensim", "sklearn", "gensim vs scikit-learn"),
  compare_sets("lsa_stopwords_en", "fox_1989", "lsa stopwords_en vs Fox"),
  compare_sets("onix_lextek", "fox_1989", "lextek ONIX page vs Fox"),
  compare_sets("onix_lextek", "lsa_stopwords_en", "lextek ONIX page vs lsa stopwords_en"),
  compare_sets("pattern_clips", "fox_1989", "Pattern vs Fox"),
  compare_sets("pattern_clips", "qdap_onix", "Pattern vs qdap Onix"),
  compare_sets("spacy", "sklearn", "spaCy vs scikit-learn (Glasgow copy)"),
  compare_sets("corenlp_patterns", "snowball_original", "CoreNLP patterns stopwords vs Snowball"),
  compare_sets("dkpro", "snowball_original", "DKPro (NLTK 2013) vs Snowball"),
  compare_sets("dl4j", "stopwords_pkg_snowball", "DL4J vs the stopwords package Snowball list"),
  compare_sets("cook_1988", "snowball_original", "Cook function words vs Snowball"),
  compare_sets("okapi_framework", "tidytext_onix", "Okapi Framework vs tidytext onix (its most similar list)"),
  compare_sets("tidytext_onix", "onix_lextek", "tidytext onix vs the lextek ONIX page"),
  compare_sets("qdap_onix", "onix_lextek", "qdap Onix vs the lextek ONIX page"),
  compare_sets("qdap_onix", "tidytext_onix", "qdap Onix vs tidytext onix"),
  compare_sets("sec_igor_onix", "onix_lextek", "igorbrigadir onix copy"),
  compare_sets("sec_igor_gate", "gate_kea", "igorbrigadir gate copy"),
  compare_sets("sec_igor_lingpipe", "lingpipe", "igorbrigadir lingpipe copy"),
  compare_sets("sec_lingpipe_javadoc", "lingpipe", "LingPipe Javadoc words vs the source clone"),
  compare_sets("sec_igor_cook", "cook_1988", "igorbrigadir cook copy vs the parsed Wayback page"),
  compare_sets("sec_igor_corenlp", "corenlp_patterns", "igorbrigadir corenlp copy"),
  compare_sets("stopwords_pkg_iso", "stopwords_iso", "stopwords package stopwords-iso vs upstream"),
  compare_sets("python_stop_words_2025", "stopwords_iso", "python stop-words 2025 vs stopwords-iso")
)
add_check("crosscheck", "quanteda snowball and smart equal the stopwords package lists", "", all(crosschecks$identical_sets[crosschecks$list_a %in% c("quanteda_snowball", "quanteda_smart")]))
add_check("crosscheck", "tm english.dat equals the Snowball default list", crosschecks$identical_sets[crosschecks$list_a == "tm_english"], crosschecks$identical_sets[crosschecks$list_a == "tm_english"])
add_check("crosscheck", "LingPipe source clone equals the alias-i Javadoc list", crosschecks$identical_sets[crosschecks$list_a == "sec_lingpipe_javadoc"], crosschecks$identical_sets[crosschecks$list_a == "sec_lingpipe_javadoc"])
add_check("crosscheck", "Voyant test-resource copy equals stop.en.txt", crosschecks$identical_sets[crosschecks$list_a == "sec_voyant_test_taporware"], crosschecks$identical_sets[crosschecks$list_a == "sec_voyant_test_taporware"])
add_check("parsing", "documented sizes reproduced: Snowball 174, SMART 571 raw, ONIX 429, KEA 452, LingPipe 76, MALLET 524, Weka 526, Glasgow 319, scikit-learn 318",
          paste(sapply(c("snowball_original", "smart_lextek", "onix_lextek", "gate_kea", "lingpipe", "mallet", "weka_rainbow", "glasgow", "sklearn"), function(i) sum(raw_entries$list_id == i)), collapse = ", "),
          all(sapply(c("snowball_original", "smart_lextek", "onix_lextek", "gate_kea", "lingpipe", "mallet", "weka_rainbow", "glasgow", "sklearn"), function(i) sum(raw_entries$list_id == i)) == c(174, 571, 429, 452, 76, 524, 526, 319, 318)))

# ---- Comparison set ----
# DOES  : the raw named comparison uses every named list whose inclusion status is included or
#         included_provisional. Aggregates, secondary copies and the excluded list stay out.
included_ids <- provenance$list_id[provenance$inclusion_status %in% c("included", "included_provisional")]
included_ids <- included_ids[included_ids %in% members$list_id]
fam <- provenance |> filter(list_id %in% included_ids) |> select(list_id, family, is_family_root)
mem_inc <- members |> filter(list_id %in% included_ids)
M <- mem_inc |> mutate(v = 1L) |> pivot_wider(id_cols = entry, names_from = list_id, values_from = v, values_fill = 0L)
Mx <- as.matrix(M[, included_ids]); rownames(Mx) <- M$entry
add_check("comparison", "membership matrix covers the union of the included lists", nrow(Mx), nrow(Mx) == n_distinct(mem_inc$entry))

# Exact-duplicate groups (identical normalised sets) and family-aware counts.
set_hash <- vapply(included_ids, function(id) digest(sort(set_of(id)), algo = "sha256"), character(1))
dup_groups <- tibble(list_id = included_ids, set_hash = set_hash) |> group_by(set_hash) |> mutate(distinct_set_id = cur_group_id(), n_in_group = n()) |> ungroup()
distinct_rep <- dup_groups |> group_by(distinct_set_id) |> slice(1) |> ungroup()
family_of <- setNames(fam$family, fam$list_id)
root_ids  <- fam$list_id[fam$is_family_root]
fam_names <- sort(unique(fam$family))
fam_any <- sapply(fam_names, function(f) as.integer(rowSums(Mx[, names(family_of)[family_of == f], drop = FALSE]) > 0))
fam_all <- sapply(fam_names, function(f) as.integer(rowSums(Mx[, names(family_of)[family_of == f], drop = FALSE]) == sum(family_of == f)))
colnames(fam_any) <- fam_names; colnames(fam_all) <- fam_names
word_stats <- tibble(entry = rownames(Mx),
  n_lists = as.integer(rowSums(Mx)),
  n_distinct_lists = as.integer(rowSums(Mx[, distinct_rep$list_id, drop = FALSE])),
  n_families_any = as.integer(rowSums(fam_any)),
  n_families_all = as.integer(rowSums(fam_all)),
  n_family_roots = as.integer(rowSums(Mx[, root_ids, drop = FALSE])),
  families_any = apply(fam_any, 1, function(r) paste(fam_names[r == 1], collapse = ";")),
  families_all = apply(fam_all, 1, function(r) paste(fam_names[r == 1], collapse = ";"))) |>
  left_join(mem_inc |> distinct(entry, entry_class), by = "entry")
add_check("comparison", "raw list count equals the row sums of the membership matrix", "", all(word_stats$n_lists == rowSums(Mx)))

# ---- Tiers ----
# DOES  : nested tiers cut at count thresholds (never inside a tied group), chosen to bring the cumulative
#         sizes closest to 1:2:4 on the log scale. Primary tiers use n_lists; supplementary tiers use n_families_any.
choose_tiers <- function(cnt, ratio = tier_ratio) {
  vals <- sort(unique(cnt), decreasing = TRUE); vals <- vals[vals >= 2]   # the widest tier still requires agreement of at least two lists / families
  if (length(vals) < 3) return(list(thresholds = rep(NA, 3), sizes = rep(NA, 3)))
  best <- NULL
  for (a in seq_along(vals)) for (b in seq_along(vals)) for (cc in seq_along(vals)) if (a < b && b < cc) {
    N <- c(sum(cnt >= vals[a]), sum(cnt >= vals[b]), sum(cnt >= vals[cc]))
    obj <- abs(log(N[2] / N[1]) - log(ratio[2] / ratio[1])) + abs(log(N[3] / N[1]) - log(ratio[3] / ratio[1]))
    if (is.null(best) || obj < best$obj - 1e-12) best <- list(obj = obj, thresholds = vals[c(a, b, cc)], sizes = N)
  }
  best
}
tp <- choose_tiers(word_stats$n_lists); tf <- choose_tiers(word_stats$n_families_any)
tier_of <- function(cnt, th) case_when(cnt >= th[1] ~ 1L, cnt >= th[2] ~ 2L, cnt >= th[3] ~ 3L, TRUE ~ NA_integer_)
word_stats <- word_stats |> mutate(tier_primary = tier_of(n_lists, tp$thresholds), tier_family = tier_of(n_families_any, tf$thresholds))
tier_summary <- bind_rows(
  tibble(ordering = "primary (n_lists)", tier = 1:3, threshold = tp$thresholds, threshold_rule = sprintf("n_lists >= %d", tp$thresholds), cumulative_size = tp$sizes, size_ratio_to_tier1 = round(tp$sizes / tp$sizes[1], 2)),
  tibble(ordering = "family-aware (n_families_any)", tier = 1:3, threshold = tf$thresholds, threshold_rule = sprintf("n_families_any >= %d", tf$thresholds), cumulative_size = tf$sizes, size_ratio_to_tier1 = round(tf$sizes / tf$sizes[1], 2)))
add_check("tiers", "primary tiers are nested and cut only at count thresholds", paste(tp$sizes, collapse = " < "), all(diff(tp$sizes) > 0) && all(diff(tp$thresholds) < 0))

# ---- Master membership table (primary sort: most lists first, then alphabetical in the C locale) ----
master <- word_stats |> left_join(M, by = "entry") |> arrange(desc(n_lists), entry) |> mutate(rank = row_number(), .before = 1)
names(master)[names(master) %in% included_ids] <- paste0("in_", names(master)[names(master) %in% included_ids])
master <- master |> relocate(rank, entry, entry_class, n_lists, n_distinct_lists, n_families_any, n_families_all, n_family_roots, tier_primary, tier_family, families_any, families_all)
master_curation <- master |> mutate(decision = "", notes = "", .after = entry)
add_check("master", "master table sorted by n_lists descending then entry ascending", "", !is.unsorted(rev(master$n_lists)) && all(diff(order(-master$n_lists, master$entry, method = "radix")) == 1))

# ---- Pairwise Jaccard ----
inter <- crossprod(Mx); sz <- colSums(Mx); uni <- outer(sz, sz, "+") - inter; J <- inter / uni
jaccard_matrix <- as_tibble(round(J, 4), rownames = "list_id")
pairs <- as_tibble(as.table(J), .name_repair = "minimal"); names(pairs) <- c("list_a", "list_b", "jaccard")
pairs <- pairs |> filter(as.character(list_a) < as.character(list_b)) |> mutate(list_a = as.character(list_a), list_b = as.character(list_b),
  size_a = sz[list_a], size_b = sz[list_b], intersection = inter[cbind(list_a, list_b)], union = uni[cbind(list_a, list_b)],
  only_in_a = size_a - intersection, only_in_b = size_b - intersection, family_a = family_of[list_a], family_b = family_of[list_b], same_family = family_a == family_b,
  relationship = case_when(intersection == size_a & intersection == size_b ~ "identical", intersection == size_a ~ "a_subset_of_b", intersection == size_b ~ "b_subset_of_a",
                           jaccard >= near_duplicate_jaccard ~ "near_duplicate", TRUE ~ "distinct"), jaccard = round(jaccard, 4)) |> arrange(desc(jaccard), list_a, list_b)
words_diff <- function(a, b) paste(sort(setdiff(set_of(a), set_of(b))), collapse = " ")
duplicates <- pairs |> filter(relationship == "identical") |> select(list_a, list_b, size_a, family_a, family_b)
near_duplicates <- pairs |> filter(relationship %in% c("near_duplicate", "a_subset_of_b", "b_subset_of_a")) |> filter(jaccard >= near_duplicate_jaccard) |>
  mutate(words_only_in_a = map2_chr(list_a, list_b, words_diff), words_only_in_b = map2_chr(list_b, list_a, words_diff)) |>
  select(list_a, list_b, jaccard, relationship, size_a, size_b, only_in_a, only_in_b, words_only_in_a, words_only_in_b, family_a, family_b)
duplicate_groups <- dup_groups |> filter(n_in_group > 1) |> group_by(distinct_set_id) |> summarise(n_lists = n(), lists = paste(list_id, collapse = " "), size = sz[list_id[1]], family = paste(unique(family_of[list_id]), collapse = ";"), .groups = "drop")
most_similar_pairs  <- pairs |> slice_head(n = 40)
least_similar_pairs <- pairs |> arrange(jaccard, list_a, list_b) |> slice_head(n = 40)
add_check("jaccard", "Jaccard matrix is symmetric with unit diagonal", "", isTRUE(all.equal(J, t(J))) && all(abs(diag(J) - 1) < 1e-12))

# ---- Larger lists: additions beyond every smaller list, and differences from the most similar smaller list ----
order_ids <- names(sz)[order(sz, names(sz), method = "radix")]
larger <- bind_rows(lapply(order_ids, function(id) {
  smaller <- names(sz)[sz < sz[id]]
  added <- if (length(smaller)) sort(setdiff(set_of(id), unique(unlist(lapply(smaller, set_of))))) else sort(set_of(id))
  if (length(smaller)) { js <- J[id, smaller]; best <- smaller[order(-js, smaller, method = "radix")][1] } else best <- NA_character_
  tibble(list_id = id, size = sz[id], family = family_of[id], n_smaller_lists = length(smaller), n_added_beyond_smaller_union = length(added),
         words_added_beyond_smaller_union = paste(added, collapse = " "), most_similar_smaller_list = best,
         jaccard_to_most_similar_smaller = if (is.na(best)) NA_real_ else round(J[id, best], 4),
         n_in_list_not_in_most_similar_smaller = if (is.na(best)) NA_integer_ else length(setdiff(set_of(id), set_of(best))),
         words_in_list_not_in_most_similar_smaller = if (is.na(best)) "" else paste(sort(setdiff(set_of(id), set_of(best))), collapse = " "),
         n_in_most_similar_smaller_not_in_list = if (is.na(best)) NA_integer_ else length(setdiff(set_of(best), set_of(id))),
         words_in_most_similar_smaller_not_in_list = if (is.na(best)) "" else paste(sort(setdiff(set_of(best), set_of(id))), collapse = " "))
}))
larger_long <- bind_rows(lapply(order_ids, function(id) {
  smaller <- names(sz)[sz < sz[id]]
  added <- if (length(smaller)) sort(setdiff(set_of(id), unique(unlist(lapply(smaller, set_of))))) else sort(set_of(id))
  if (length(added) == 0) return(tibble())
  tibble(list_id = id, size = sz[id], word = added) |> left_join(word_stats |> select(entry, n_lists, entry_class), by = c("word" = "entry"))
}))
smaller_diff_long <- bind_rows(lapply(order_ids, function(id) {
  smaller <- names(sz)[sz < sz[id]]; if (!length(smaller)) return(tibble())
  best <- larger$most_similar_smaller_list[larger$list_id == id]
  bind_rows(tibble(list_id = id, most_similar_smaller_list = best, direction = "in_list_not_in_smaller", word = sort(setdiff(set_of(id), set_of(best)))),
            tibble(list_id = id, most_similar_smaller_list = best, direction = "in_smaller_not_in_list", word = sort(setdiff(set_of(best), set_of(id))))) |>
    left_join(word_stats |> select(entry, n_lists, entry_class), by = c("word" = "entry"))
}))

# ---- Family verification (documented families against measured similarity) ----
hc <- hclust(as.dist(1 - J), method = "average")
family_review <- bind_rows(lapply(included_ids, function(id) {
  f <- family_of[id]; same <- setdiff(names(family_of)[family_of == f], id); other <- names(family_of)[family_of != f]
  within_max <- if (length(same)) max(J[id, same]) else NA_real_; within_best <- if (length(same)) same[which.max(J[id, same])] else NA_character_
  other_max <- max(J[id, other]); other_best <- other[which.max(J[id, other])]
  flag <- case_when(is.na(within_max) & other_max >= family_review_jaccard ~ "singleton family but similar to another family",
                    !is.na(within_max) & other_max > within_max ~ "more similar to a list outside its documented family",
                    !is.na(within_max) & within_max < family_review_jaccard ~ "weak similarity to its documented family", TRUE ~ "")
  tibble(list_id = id, family = f, is_family_root = id %in% root_ids, size = sz[id], within_family_max_jaccard = round(within_max, 4), most_similar_in_family = within_best,
         outside_family_max_jaccard = round(other_max, 4), most_similar_outside_family = other_best, outside_family = family_of[other_best], review_flag = flag)
})) |> arrange(desc(review_flag != ""), family, list_id)
family_definitions <- fam |> group_by(family) |> summarise(n_lists = n(), root_list = paste(list_id[is_family_root], collapse = " "), member_lists = paste(list_id, collapse = " "), .groups = "drop") |>
  left_join(provenance |> filter(list_id %in% included_ids, is_family_root) |> select(family, documented_parent, upstream_source), by = "family") |>
  left_join(family_review |> group_by(family) |> summarise(min_within_family_jaccard = suppressWarnings(min(within_family_max_jaccard, na.rm = TRUE)), n_review_flags = sum(review_flag != ""), .groups = "drop"), by = "family") |>
  mutate(min_within_family_jaccard = ifelse(is.finite(min_within_family_jaccard), min_within_family_jaccard, NA))
word_family <- tibble(entry = rownames(Mx)) |> bind_cols(as_tibble(fam_any) |> rename_with(~ paste0("any_", .x))) |> bind_cols(as_tibble(fam_all) |> rename_with(~ paste0("all_", .x))) |>
  left_join(word_stats |> select(entry, n_lists, n_families_any, n_families_all), by = "entry") |> arrange(desc(n_families_any), desc(n_lists), entry)

# ---- Aggregate conditions (stopwords-iso, python stop-words 2025) against the included lists ----
agg_ids <- provenance$list_id[provenance$inclusion_status == "aggregate_condition"]
union_included <- rownames(Mx)
aggregate_coverage <- bind_rows(lapply(agg_ids, function(a) {
  A <- set_of(a)
  bind_rows(tibble(aggregate = a, list_id = "(union of included lists)", size = length(union_included), n_in_aggregate = sum(union_included %in% A), share_in_aggregate = round(mean(union_included %in% A), 4),
                   words_not_in_aggregate = paste(sort(setdiff(union_included, A)), collapse = " ")),
            bind_rows(lapply(included_ids, function(id) { S <- set_of(id); tibble(aggregate = a, list_id = id, size = length(S), n_in_aggregate = sum(S %in% A), share_in_aggregate = round(mean(S %in% A), 4), words_not_in_aggregate = paste(sort(setdiff(S, A)), collapse = " ")) })))
}))
aggregate_only_words <- bind_rows(lapply(agg_ids, function(a) { A <- set_of(a); tibble(aggregate = a, aggregate_size = length(A), n_not_in_any_included_list = sum(!A %in% union_included), words_not_in_any_included_list = paste(sort(setdiff(A, union_included)), collapse = " ")) }))
jockers_note <- { Jk <- set_of("jockers"); tibble(list_id = "jockers", size = length(Jk), n_in_union_of_included = sum(Jk %in% union_included), share_in_union_of_included = round(mean(Jk %in% union_included), 4), sample_entries_not_in_any_included_list = paste(head(sort(setdiff(Jk, union_included)), 60), collapse = " ")) }

# ---- Write tables ----
log_line("writing tables")
table_dims <- list()
save_table <- function(df, name) { p <- file.path(tab_dir, paste0(name, ".csv")); write_csv(df, p, na = ""); table_dims[[p]] <<- c(rows = nrow(df), cols = ncol(df)); invisible(p) }
provenance_table <- sources |> filter(kind != "secondary") |> select(list_id, list_name, kind, acquisition, url, pin, file_name, parser) |>
  left_join(provenance, by = "list_id") |> left_join(list_summary |> select(list_id, n_raw_entries, n_unique_entries, formatting_only_changes), by = "list_id") |>
  left_join(dup_groups |> select(list_id, distinct_set_id, n_in_group), by = "list_id") |>
  left_join(pairs |> filter(relationship == "identical") |> select(list_a, list_b) |> pivot_longer(everything(), values_to = "list_id") |> select(list_id) |> distinct() |> mutate(has_exact_duplicate = TRUE), by = "list_id") |>
  left_join(near_duplicates |> select(list_a, list_b, jaccard) |> pivot_longer(c(list_a, list_b), values_to = "list_id") |> group_by(list_id) |> summarise(max_near_duplicate_jaccard = max(jaccard), .groups = "drop"), by = "list_id") |>
  mutate(has_exact_duplicate = coalesce(has_exact_duplicate, FALSE), exact_or_near_copy = case_when(has_exact_duplicate ~ "exact copy of another named list (see duplicate_groups.csv)",
         !is.na(max_near_duplicate_jaccard) ~ sprintf("near copy (Jaccard >= %.2f with another named list, see near_duplicates.csv)", near_duplicate_jaccard), TRUE ~ "no exact or near copy among the included lists")) |>
  select(list_id, list_name, kind, inclusion_status, inclusion_reason, upstream_source, url, pin, version_date, intended_use, n_raw_entries, n_unique_entries, documented_parent, family, is_family_root,
         exact_or_near_copy, formatting_only_changes, evidence, acquisition, file_name, parser)
save_table(provenance_table, "provenance_table")
save_table(excluded_candidates, "excluded_candidates")
save_table(acquisition_log, "acquisition_log")
save_table(list_summary, "list_summary")
save_table(normalisation_log, "normalisation_log")
save_table(crosschecks, "secondary_crosschecks")
save_table(master, "master_membership")
save_table(master_curation, "master_membership_curation")
save_table(word_family, "word_family_membership")
save_table(family_definitions, "family_definitions")
save_table(family_review, "family_review_flags")
save_table(jaccard_matrix, "jaccard_matrix")
save_table(pairs, "jaccard_pairs_all")
save_table(most_similar_pairs, "most_similar_pairs")
save_table(least_similar_pairs, "least_similar_pairs")
save_table(duplicate_groups, "duplicate_groups")
save_table(near_duplicates, "near_duplicates")
save_table(larger, "larger_list_additions")
save_table(larger_long, "larger_list_additions_long")
save_table(smaller_diff_long, "most_similar_smaller_list_differences_long")
save_table(tier_summary, "tier_summary")
save_table(master |> select(rank, entry, entry_class, n_lists, n_families_any, tier_primary, tier_family) |> filter(!is.na(tier_primary)), "tiers")
save_table(aggregate_coverage, "aggregate_coverage")
save_table(aggregate_only_words, "aggregate_only_words")
save_table(jockers_note, "jockers_excluded_evidence")
save_table(snowball_commented, "snowball_commented_out_words")
save_table(marimo_fields, "marimo_upstream_fields")
save_table(tibble(list_id = included_ids, size = sz[included_ids], family = family_of[included_ids], dendrogram_order = match(included_ids, included_ids[hc$order])), "similarity_dendrogram_order")

# ---- Figures (title, axis titles and tick labels only) ----
log_line("writing figures")
fam_levels <- fam_names
p1 <- ggplot(tibble(list_id = included_ids, size = sz[included_ids], family = family_of[included_ids]) |> mutate(list_id = factor(list_id, levels = names(sort(sz)))),
             aes(x = list_id, y = size, fill = family)) + geom_col() + coord_flip() + scale_fill_viridis_d(option = "turbo") +
  labs(title = "Unique entries per included list", x = "List", y = "Unique entries", fill = "Family") + theme_minimal(base_size = 11)
ggsave(file.path(fig_dir, "fig1_list_sizes.png"), p1, width = 9, height = 10, dpi = 150, bg = "white")
ord <- included_ids[hc$order]
heat <- as_tibble(as.table(J), .name_repair = "minimal"); names(heat) <- c("a", "b", "jaccard")
heat <- heat |> mutate(a = factor(as.character(a), levels = ord), b = factor(as.character(b), levels = ord))
p2 <- ggplot(heat, aes(x = a, y = b, fill = jaccard)) + geom_tile() + scale_fill_viridis_c(limits = c(0, 1)) +
  labs(title = "Jaccard similarity between included lists", x = "List", y = "List", fill = "Jaccard") + theme_minimal(base_size = 10) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5), panel.grid = element_blank()) + coord_equal()
ggsave(file.path(fig_dir, "fig2_jaccard_heatmap.png"), p2, width = 12, height = 11, dpi = 150, bg = "white")
png(file.path(fig_dir, "fig3_similarity_dendrogram.png"), width = 1500, height = 1400, res = 150)
par(mar = c(5, 4, 4, 14)); plot(as.dendrogram(hc), horiz = TRUE, main = "Average-linkage clustering of included lists", xlab = "1 - Jaccard"); invisible(dev.off())
p4 <- ggplot(word_stats |> count(n_lists), aes(x = n_lists, y = n)) + geom_col() + scale_x_continuous(breaks = pretty_breaks(10)) +
  labs(title = "Words by number of included lists containing them", x = "Number of included lists", y = "Words") + theme_minimal(base_size = 11)
ggsave(file.path(fig_dir, "fig4_words_by_list_count.png"), p4, width = 9, height = 5, dpi = 150, bg = "white")
p5 <- ggplot(word_stats |> count(n_families_any), aes(x = n_families_any, y = n)) + geom_col() + scale_x_continuous(breaks = 0:20) +
  labs(title = "Words by number of families containing them", x = "Number of families (any member)", y = "Words") + theme_minimal(base_size = 11)
ggsave(file.path(fig_dir, "fig5_words_by_family_count.png"), p5, width = 9, height = 5, dpi = 150, bg = "white")
p6 <- ggplot(tier_summary |> mutate(tier = factor(tier)), aes(x = tier, y = cumulative_size, fill = ordering)) + geom_col(position = position_dodge()) +
  labs(title = "Cumulative size of the three curation tiers", x = "Tier", y = "Words", fill = "Ordering") + theme_minimal(base_size = 11)
ggsave(file.path(fig_dir, "fig6_tier_sizes.png"), p6, width = 8, height = 5, dpi = 150, bg = "white")
p7 <- ggplot(larger |> mutate(list_id = factor(list_id, levels = order_ids)), aes(x = list_id, y = n_added_beyond_smaller_union, fill = family)) + geom_col() + coord_flip() + scale_fill_viridis_d(option = "turbo") +
  labs(title = "Entries added beyond the union of all smaller included lists", x = "List (smallest to largest)", y = "Entries added", fill = "Family") + theme_minimal(base_size = 11)
ggsave(file.path(fig_dir, "fig7_additions_beyond_smaller_union.png"), p7, width = 9, height = 10, dpi = 150, bg = "white")
upset_df <- mem_inc |> filter(list_id %in% root_ids) |> group_by(entry) |> summarise(lists = list(sort(list_id)), .groups = "drop")
p8 <- ggplot(upset_df, aes(x = lists)) + geom_bar() + scale_x_upset(n_intersections = 30) +
  labs(title = "Intersections among family root lists", x = "Root lists containing the word", y = "Words") + theme_minimal(base_size = 11)
ggsave(file.path(fig_dir, "fig8_family_root_intersections.png"), p8, width = 12, height = 7, dpi = 150, bg = "white")
p9 <- ggplot(word_stats |> count(entry_class), aes(x = reorder(entry_class, n), y = n)) + geom_col() + coord_flip() +
  labs(title = "Entries in the union of included lists by entry class", x = "Entry class", y = "Entries") + theme_minimal(base_size = 11)
ggsave(file.path(fig_dir, "fig9_entry_classes.png"), p9, width = 8, height = 4.5, dpi = 150, bg = "white")

# ---- Checks, run info, manifests ----
run_end <- Sys.time()
checks_tbl <- bind_rows(checks)
git_commit <- tryCatch(system2("git", c("-C", shQuote(project_dir), "rev-parse", "--short", "HEAD"), stdout = TRUE, stderr = FALSE), error = function(e) "")
run_info <- tibble(item = c("run_start_utc", "run_end_utc", "runtime_minutes", "git_commit_at_start", "r_version", "n_sources_registered", "n_included_lists", "n_union_entries", "n_families", "near_duplicate_jaccard", "tier_ratio", "family_review_jaccard"),
                   value = c(format(run_start, "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"), format(run_end, "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"), round(as.numeric(difftime(run_end, run_start, units = "mins")), 2), git_commit, R.version.string,
                             nrow(sources), length(included_ids), nrow(Mx), length(fam_names), near_duplicate_jaccard, paste(tier_ratio, collapse = ":"), family_review_jaccard))
save_table(run_info, "run_info")
save_table(checks_tbl, "checks")
package_versions <- lapply(c("dplyr", "tidyr", "tibble", "purrr", "stringi", "readr", "ggplot2", "scales", "digest", "yaml", "curl", "xml2", "ggupset", required_pkgs), function(p) list(name = p, version = pkg_version(p)))
raw_files <- list.files(raw_dir, recursive = TRUE, full.names = TRUE)
input_files <- c(lapply(raw_files, function(f) list(path = f, hash = sha256(f), format = "other")), list(list(path = script_path, hash = sha256(script_path), format = "r-script")))
manifest_parameters <- list(
  purpose = "acquire, preserve, document and compare English stopword lists for manual curation; no keep/remove decision, no project list",
  included_lists = paste(included_ids, collapse = ", "), aggregate_conditions = paste(agg_ids, collapse = ", "), excluded_acquired = "jockers",
  normalisation = "NFC; U+2018/U+2019/U+02BC/U+2032 to '; trim; internal whitespace to one space; lower case; empty entries dropped; within-list duplicates collapsed (first kept); every change logged",
  primary_sort = "n_lists descending, then entry ascending in the C locale",
  family_counts = "n_families_any: families with at least one member containing the word; n_families_all: families whose every member contains it; n_family_roots: count over root lists; n_distinct_lists: count after collapsing identical sets",
  tiers = sprintf("nested tiers cut at count thresholds closest to %s on the log scale, never inside a tied group; primary on n_lists (thresholds %s), family-aware on n_families_any (thresholds %s)", paste(tier_ratio, collapse = ":"), paste(tp$thresholds, collapse = "/"), paste(tf$thresholds, collapse = "/")),
  near_duplicate_jaccard = as.character(near_duplicate_jaccard), family_review_jaccard = as.character(family_review_jaccard),
  families = paste(sprintf("%s: %s", family_definitions$family, family_definitions$member_lists), collapse = " | "))
write_sidecar <- function(output) {
  ext <- tolower(tools::file_ext(output))
  manifest <- list(manifest_version = 1L, output_file = output, output_hash = sha256(output), output_format = switch(ext, csv = "csv", png = "image/png", "other"))
  if (ext == "csv" && !is.null(table_dims[[output]])) { manifest$output_rows <- unname(table_dims[[output]][["rows"]]); manifest$output_cols <- unname(table_dims[[output]][["cols"]]) }
  manifest$input_files    <- input_files
  manifest$transformation <- list(script = script_path, script_hash = sha256(script_path), parameters = manifest_parameters, git_commit = git_commit)
  manifest$software <- list(language = "R", language_version = paste(R.version$major, R.version$minor, sep = "."), packages = package_versions, os = paste(Sys.info()[["sysname"]], Sys.info()[["release"]]))
  manifest$timestamp <- format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")
  manifest$notes <- "Derived from the raw source files under raw/ (byte-exact upstream copies and installed-package exports). Raw files are never modified; every formatting-only change is listed in normalisation_log.csv."
  write_yaml(manifest, paste0(output, ".manifest.yml"))
}
invisible(lapply(list.files(tab_dir, pattern = "\\.csv$", full.names = TRUE), write_sidecar))
invisible(lapply(list.files(fig_dir, pattern = "\\.png$", full.names = TRUE), write_sidecar))
invisible(lapply(list.files(lists_dir, pattern = "\\.txt$", full.names = TRUE), write_sidecar))

print(as.data.frame(checks_tbl), right = FALSE)
cat(sprintf("\n%d included lists, %d entries in their union, %d families; tiers %s (primary) and %s (family-aware).\n", length(included_ids), nrow(Mx), length(fam_names), paste(tp$sizes, collapse = "/"), paste(tf$sizes, collapse = "/")))
cat("Done in", round(as.numeric(difftime(run_end, run_start, units = "mins")), 1), "min. Outputs ->", out_dir, "\n")
