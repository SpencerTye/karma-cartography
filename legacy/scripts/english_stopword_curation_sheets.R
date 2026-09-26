# English stopword lists: manual-curation spreadsheets for the primary raw named-list consensus tiers
#
# Reads, read-only, three delivered tables of english_stopword_lists.R under
# r_analysis_outputs/stopword_lists/tables/ (master_membership.csv, list_summary.csv,
# tier_summary.csv) and writes three .xlsx curation workbooks under
# r_analysis_outputs/stopword_lists/curation/, one per cumulative primary tier
# (155 / 310 / 627 entries). Rows are ordered by the raw number of included named lists
# containing the entry (descending), then by the entry (ascending, C locale); every one of
# the 45 included named lists keeps its own 1/0 membership column; decision and notes are
# blank. Nothing under tables/, lists/, raw/ or figures/ is modified; no list is
# re-acquired; no audit, provenance, lineage, Jaccard or topic-model step is run; no
# keep/remove decision or recommendation is made; the Reddit corpus is not read.

# ---- Setup ----
suppressPackageStartupMessages({
  library(dplyr); library(tibble); library(readr); library(openxlsx); library(readxl); library(digest); library(yaml)
})
run_start <- Sys.time()
invisible(Sys.setlocale("LC_COLLATE", "C"))   # reproducible string ordering (same as the source and audit scripts)
log_line  <- function(...) cat(format(Sys.time(), "%H:%M:%S"), ..., "\n")

project_dir <- Sys.getenv("KC_REPO_ROOT", unset = getwd())
src_dir     <- file.path(project_dir, "r_analysis_outputs/stopword_lists")   # delivered analysis (read-only)
src_tab     <- file.path(src_dir, "tables")
out_dir     <- file.path(src_dir, "curation")
script_path <- file.path(project_dir, "english_stopword_curation_sheets.R")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# The workbooks are the curator's working files once decisions are typed into them, so an
# existing workbook is never overwritten unless regeneration is requested explicitly.
overwrite <- identical(Sys.getenv("STOPWORD_CURATION_OVERWRITE"), "1")

# Parameters (recorded in the manifests)
expected_tier_sizes <- c(155L, 310L, 627L)   # cumulative primary tiers, as assigned (tier_summary.csv)
membership_indicator <- c(present = 1L, absent = 0L)
width_rank <- 7; width_word <- 22; width_n_lists <- 16; width_membership <- 5; width_decision <- 16; width_notes <- 50

sha256 <- function(path) paste0("sha256:", digest(path, algo = "sha256", file = TRUE))
checks <- list()
add_check <- function(file, check, measured, pass) checks[[length(checks) + 1L]] <<- tibble(file = file, check = check, measured = as.character(measured), pass = isTRUE(pass))
radix_sorted <- function(n_lists, words) identical(order(-n_lists, words, method = "radix"), seq_along(words))   # C-locale byte order

# ---- Inputs (delivered tables, read-only) ----
log_line("reading delivered tables")
master <- read_csv(file.path(src_tab, "master_membership.csv"), show_col_types = FALSE, progress = FALSE,
                   col_types = cols(.default = col_integer(), entry = col_character(), entry_class = col_character(), families_any = col_character(), families_all = col_character()))
list_summary <- read_csv(file.path(src_tab, "list_summary.csv"), show_col_types = FALSE, progress = FALSE, col_types = cols(.default = col_character()))
tier_summary <- read_csv(file.path(src_tab, "tier_summary.csv"), show_col_types = FALSE, progress = FALSE,
                         col_types = cols(.default = col_character(), tier = col_integer(), threshold = col_integer(), cumulative_size = col_integer(), size_ratio_to_tier1 = col_double()))
input_dims <- list(master_membership = dim(master), list_summary = dim(list_summary), tier_summary = dim(tier_summary))

# Included named lists (32 included + 13 included_provisional) and their membership columns, in the delivered column order.
included_ids    <- list_summary$list_id[list_summary$kind == "named" & list_summary$inclusion_status %in% c("included", "included_provisional")]
membership_cols <- grep("^in_", names(master), value = TRUE)
stopifnot(length(included_ids) == 45L, setequal(membership_cols, paste0("in_", included_ids)), !anyDuplicated(master$entry))
stopifnot(all(as.matrix(master[membership_cols]) %in% membership_indicator), identical(rowSums(master[membership_cols]), as.numeric(master$n_lists)))

# Primary raw named-list consensus tiers, exactly as delivered (tier_primary in master_membership.csv; thresholds in tier_summary.csv).
primary <- tier_summary %>% filter(ordering == "primary (n_lists)") %>% arrange(tier)
stopifnot(identical(primary$tier, 1:3), identical(primary$cumulative_size, expected_tier_sizes))
for (k in 1:3) stopifnot(identical(which(!is.na(master$tier_primary) & master$tier_primary <= k), which(master$n_lists >= primary$threshold[k])))

# ---- Curation table: rank, word, number_of_lists, one column per included named list, blank decision and notes ----
base <- master %>% transmute(word = entry, number_of_lists = n_lists, across(all_of(membership_cols)), tier_primary, delivered_rank = rank)
base <- base[order(-base$number_of_lists, base$word, method = "radix"), ]
tier_tables <- lapply(1:3, function(k) {
  base %>% filter(!is.na(tier_primary), tier_primary <= k) %>%
    mutate(rank = row_number(), decision = NA_character_, notes = NA_character_) %>%
    select(rank, word, number_of_lists, all_of(membership_cols), decision, notes, delivered_rank)
})
tier_files <- tibble(tier = 1:3,
                     file  = sprintf("stopword_curation_tier%d%s_%d_words.xlsx", 1:3, c("", "_cumulative", "_cumulative"), expected_tier_sizes),
                     sheet = sprintf("tier%d%s_%d", 1:3, c("", "_cumulative", "_cumulative"), expected_tier_sizes),
                     path  = NA_character_, rows = NA_integer_, cols = NA_integer_)
tier_files$path <- file.path(out_dir, tier_files$file)
if (!overwrite && any(file.exists(tier_files$path))) {
  stop("Curation workbook(s) already exist under ", out_dir, " and may hold the curator's decisions. ",
       "Set STOPWORD_CURATION_OVERWRITE=1 to regenerate them from the delivered tables.")
}

# ---- Workbooks ----
# openxlsx 4.2.8.1 writes, for every sheet, relationships to a drawing part and a VML drawing part plus a
# content-type override for the drawing, without adding either part to the archive (verified on a bare
# one-column workbook). Excel tolerates the dangling entries; strict readers such as openpyxl reject the
# file. After saving, every relationship or content-type override whose target part is absent is removed
# and the package is re-zipped with [Content_Types].xml first. Nothing else in the package is touched.
resolve_part <- function(base_dir, target) {
  out <- character(0)
  for (p in strsplit(paste(base_dir, target, sep = "/"), "/", fixed = TRUE)[[1]]) {
    if (p == "..") out <- out[-length(out)] else if (nzchar(p) && p != ".") out <- c(out, p)
  }
  paste(out, collapse = "/")
}
rels_base_dir <- function(rels) if (grepl("/_rels/", rels, fixed = TRUE)) sub("/_rels/[^/]*$", "", rels) else ""
required_parts <- c("[Content_Types].xml", "_rels/.rels", "xl/workbook.xml", "xl/_rels/workbook.xml.rels", "xl/worksheets/sheet1.xml", "xl/worksheets/_rels/sheet1.xml.rels")
dangling_parts <- function(read_member, members) {   # read_member(name) returns a member's text; returns the missing parts and unresolved targets
  problems <- sprintf("missing part %s", setdiff(required_parts, members))
  if (length(problems)) return(problems)
  for (rels in grep("_rels/[^/]*[.]rels$", members, value = TRUE)) {
    xml <- read_member(rels)
    for (el in regmatches(xml, gregexpr("<Relationship [^>]*/>", xml))[[1]]) {
      if (grepl('TargetMode="External"', el, fixed = TRUE)) next
      target <- resolve_part(rels_base_dir(rels), sub('.*Target="([^"]*)".*', "\\1", el))
      if (!target %in% members) problems <- c(problems, sprintf("%s -> %s", rels, target))
    }
  }
  xml <- read_member("[Content_Types].xml")
  for (el in regmatches(xml, gregexpr("<Override [^>]*/>", xml))[[1]]) {
    part <- sub('.*PartName="/([^"]*)".*', "\\1", el)
    if (!part %in% members) problems <- c(problems, sprintf("[Content_Types].xml -> %s", part))
  }
  problems
}
archive_members <- function(path) { m <- zip::zip_list(path)$filename; m[!endsWith(m, "/")] }
strip_dangling_parts <- function(path) {
  exdir <- tempfile("xlsx_parts_"); dir.create(exdir); on.exit(unlink(exdir, recursive = TRUE), add = TRUE)
  zip::unzip(path, exdir = exdir)
  members <- archive_members(path)
  drop_dangling <- function(member, pattern, resolve) {
    f <- file.path(exdir, member); xml <- paste(readLines(f, warn = FALSE, encoding = "UTF-8"), collapse = "\n")
    for (el in regmatches(xml, gregexpr(pattern, xml))[[1]]) if (!resolve(el) %in% members) xml <- sub(el, "", xml, fixed = TRUE)
    writeLines(xml, f, useBytes = TRUE)
  }
  for (rels in grep("_rels/[^/]*[.]rels$", members, value = TRUE)) {
    drop_dangling(rels, "<Relationship [^>]*/>", function(el) if (grepl('TargetMode="External"', el, fixed = TRUE)) el else resolve_part(rels_base_dir(rels), sub('.*Target="([^"]*)".*', "\\1", el)))
  }
  drop_dangling("[Content_Types].xml", "<Override [^>]*/>", function(el) sub('.*PartName="/([^"]*)".*', "\\1", el))
  rebuilt <- tempfile(fileext = ".xlsx")
  zip::zip(rebuilt, files = c("[Content_Types].xml", setdiff(members, "[Content_Types].xml")), root = exdir, mode = "mirror", include_directories = FALSE)   # mirror keeps the member paths
  stopifnot(file.copy(rebuilt, path, overwrite = TRUE)); unlink(rebuilt)
}
package_integrity <- function(path) {   # unresolved relationship targets / content-type parts of the saved workbook
  members <- archive_members(path)
  dangling_parts(function(m) paste(readLines(unz(path, m), warn = FALSE, encoding = "UTF-8"), collapse = "\n"), members)
}

header_height <- round(max(nchar(membership_cols)) * 5.5 + 18)   # rotated membership headers need a tall header row
write_curation_workbook <- function(df, path, sheet) {
  n <- nrow(df)
  member_idx <- match(membership_cols, names(df))
  right_idx  <- match(c("decision", "notes"), names(df))
  header_plain <- createStyle(textDecoration = "bold", wrapText = TRUE, valign = "bottom", halign = "left", border = "bottom")
  header_rot   <- createStyle(textDecoration = "bold", textRotation = 90, valign = "bottom", halign = "center", border = "bottom")
  body_center  <- createStyle(halign = "center")
  wb <- createWorkbook(creator = "english_stopword_curation_sheets.R")   # fixed creator: no user name in docProps
  addWorksheet(wb, sheet)
  writeData(wb, sheet, df, startRow = 1, startCol = 1, withFilter = TRUE, headerStyle = header_plain, keepNA = FALSE)   # NA -> blank cell
  addStyle(wb, sheet, header_rot, rows = 1, cols = member_idx, gridExpand = TRUE)
  addStyle(wb, sheet, body_center, rows = 1 + seq_len(n), cols = member_idx, gridExpand = TRUE)
  setColWidths(wb, sheet, cols = 1:3, widths = c(width_rank, width_word, width_n_lists))
  setColWidths(wb, sheet, cols = member_idx, widths = width_membership)
  setColWidths(wb, sheet, cols = right_idx, widths = c(width_decision, width_notes))
  setRowHeights(wb, sheet, rows = 1, heights = header_height)
  freezePane(wb, sheet, firstActiveRow = 2, firstActiveCol = 4)   # header row plus rank, word, number_of_lists stay visible
  saveWorkbook(wb, path, overwrite = TRUE)
  strip_dangling_parts(path)
}
for (k in 1:3) {
  df <- tier_tables[[k]] %>% select(-delivered_rank)
  log_line(sprintf("writing %s (%d rows, %d columns)", tier_files$file[k], nrow(df), ncol(df)))
  write_curation_workbook(df, tier_files$path[k], tier_files$sheet[k])
  tier_files$rows[k] <- nrow(df); tier_files$cols[k] <- ncol(df)
}

# ---- Verification: every workbook is read back with readxl (an independent reader) and checked ----
log_line("verifying written workbooks")
expected_names <- c("rank", "word", "number_of_lists", membership_cols, "decision", "notes")
sheet_xml <- function(path) paste(readLines(unz(path, "xl/worksheets/sheet1.xml"), warn = FALSE), collapse = "")
read_back <- lapply(1:3, function(k) {
  f <- tier_files$file[k]
  x <- readxl::read_excel(tier_files$path[k], sheet = tier_files$sheet[k])
  xml <- sheet_xml(tier_files$path[k])
  n <- nrow(x)
  add_check(f, "column names and order: rank, word, number_of_lists, one in_<list> column per included named list, decision, notes", paste(ncol(x), "columns"), identical(names(x), expected_names))
  add_check(f, sprintf("exactly %d word rows", expected_tier_sizes[k]), n, n == expected_tier_sizes[k])
  add_check(f, "number_of_lists never increases down the rows", paste(range(x$number_of_lists), collapse = "-"), !is.unsorted(rev(x$number_of_lists)))
  add_check(f, "words alphabetically ascending (C locale) within every tied number_of_lists group; no duplicate word", n, radix_sorted(x$number_of_lists, x$word) && !anyDuplicated(x$word))
  add_check(f, "rank equals the row position 1..n", n, identical(as.integer(x$rank), seq_len(n)))
  add_check(f, "rank equals the delivered master_membership.csv rank for the same word", n, identical(as.integer(x$rank), tier_tables[[k]]$delivered_rank[match(x$word, tier_tables[[k]]$word)]))
  add_check(f, "membership rule: rows are exactly the entries with n_lists >= primary threshold", primary$threshold[k], setequal(x$word, master$entry[master$n_lists >= primary$threshold[k]]))
  add_check(f, "every included named stopword list has its own membership column", length(membership_cols), setequal(grep("^in_", names(x), value = TRUE), paste0("in_", included_ids)) && length(included_ids) == 45L)
  add_check(f, "membership values are 1/0 and agree with master_membership.csv for every word and list", n * length(membership_cols),
            all(as.matrix(x[membership_cols]) %in% membership_indicator) && identical(as.matrix(x[membership_cols]) * 1, as.matrix(master[match(x$word, master$entry), membership_cols]) * 1))
  add_check(f, "number_of_lists equals the row sum of the membership columns", n, identical(as.numeric(rowSums(x[membership_cols])), as.numeric(x$number_of_lists)))
  add_check(f, "decision column completely blank", sum(!is.na(x$decision)), all(is.na(x$decision)))
  add_check(f, "notes column completely blank", sum(!is.na(x$notes)), all(is.na(x$notes)))
  pane <- regmatches(xml, regexpr("<pane [^>]*/>", xml))
  add_check(f, "header row and columns A:C frozen (pane at D2)", if (length(pane)) pane else "no pane element",
            length(pane) == 1L && all(vapply(c('xSplit="3"', 'ySplit="1"', 'topLeftCell="D2"', 'state="frozen"'), grepl, logical(1), x = pane, fixed = TRUE)))
  add_check(f, "autofilter over the header row of every column", sprintf("A1:%s%d", int2col(length(expected_names)), n + 1L), grepl(sprintf('<autoFilter ref="A1:%s%d"', int2col(length(expected_names)), n + 1L), xml, fixed = TRUE))
  dangling <- package_integrity(tier_files$path[k])
  add_check(f, "package integrity: every relationship target and content-type override resolves to a part in the archive", if (length(dangling)) paste(dangling, collapse = "; ") else "no dangling part", length(dangling) == 0L)
  x
})
add_check("tier2 vs tier1", "tier 1 words are a subset of tier 2 (and its first 155 rows)", nrow(read_back[[1]]), all(read_back[[1]]$word %in% read_back[[2]]$word) && identical(read_back[[2]]$word[seq_len(nrow(read_back[[1]]))], read_back[[1]]$word))
add_check("tier3 vs tier2", "tier 2 words are a subset of tier 3 (and its first 310 rows)", nrow(read_back[[2]]), all(read_back[[2]]$word %in% read_back[[3]]$word) && identical(read_back[[3]]$word[seq_len(nrow(read_back[[2]]))], read_back[[2]]$word))
checks_tbl <- bind_rows(checks)
print(as.data.frame(checks_tbl), right = FALSE)
if (!all(checks_tbl$pass)) stop(sum(!checks_tbl$pass), " verification check(s) failed; no manifest written.")

# ---- Manifests (same structure as the source and audit analyses: inputs with SHA256, script hash, git commit, parameters, package versions) ----
run_end <- Sys.time()
git_commit <- tryCatch(system2("git", c("-C", shQuote(project_dir), "rev-parse", "--short", "HEAD"), stdout = TRUE, stderr = FALSE), error = function(e) "")
package_versions <- lapply(c("dplyr", "tibble", "readr", "openxlsx", "zip", "readxl", "digest", "yaml"), function(p) list(name = p, version = as.character(packageVersion(p))))
input_files <- c(lapply(names(input_dims), function(nm) { f <- file.path(src_tab, paste0(nm, ".csv")); list(path = f, hash = sha256(f), format = "csv", rows = input_dims[[nm]][1], cols = input_dims[[nm]][2]) }),
                 list(list(path = script_path, hash = sha256(script_path), format = "r-script")))
manifest_parameters <- list(
  purpose = "manual stopword curation workbooks: one per cumulative primary raw named-list consensus tier; rows ranked by number_of_lists descending then word ascending; blank decision and notes; no keep/remove decision, no recommendation, no re-acquisition, no audit or comparison step re-run",
  tier_system = sprintf("primary raw named-list consensus (n_lists over the 45 included named lists, copies counted): tier 1 n_lists >= %d (%d entries); tier 2 cumulative n_lists >= %d (%d); tier 3 cumulative n_lists >= %d (%d); the family-aware 127 / 253 / 454 tiers are not used",
                        primary$threshold[1], expected_tier_sizes[1], primary$threshold[2], expected_tier_sizes[2], primary$threshold[3], expected_tier_sizes[3]),
  row_order = "number_of_lists descending, then word ascending in C-locale byte order (order(method = 'radix')); rank = row position, identical to the delivered master_membership.csv rank",
  included_lists = paste(included_ids, collapse = ", "),
  membership_indicator = "1 = the list contains the word, 0 = it does not; one in_<list_id> column per included named list, in the column order of master_membership.csv",
  workbook_layout = sprintf("one sheet; header row and columns A:C (rank, word, number_of_lists) frozen at D2; autofilter on the header row; column widths rank %g, word %g, number_of_lists %g, membership %g (headers rotated 90 degrees, header row %d points), decision %g, notes %g",
                            width_rank, width_word, width_n_lists, width_membership, header_height, width_decision, width_notes),
  overwrite_guard = "an existing workbook is never overwritten unless STOPWORD_CURATION_OVERWRITE=1, because the workbooks become the curator's working files",
  package_cleanup = "after saving, the dangling drawing relationships and content-type override that openxlsx 4.2.8.1 writes for every sheet (parts absent from the archive) are removed and the package is re-zipped; verified by the package-integrity check")
schema_diff <- list(
  added   = lapply(c("decision", "notes"), function(n) list(name = n, type = "character")),
  removed = lapply(setdiff(names(master), c("rank", "entry", "n_lists", membership_cols)), function(n) list(name = n, type = class(master[[n]])[1])),
  renamed = list(list(from = "entry", to = "word", inferred = FALSE), list(from = "n_lists", to = "number_of_lists", inferred = FALSE)))
write_sidecar <- function(k) {
  output <- tier_files$path[k]
  manifest <- list(manifest_version = 1L, output_file = output, output_hash = sha256(output), output_format = "xlsx",
                   output_rows = tier_files$rows[k], output_cols = tier_files$cols[k], input_files = input_files,
                   transformation = list(script = script_path, script_hash = sha256(script_path), parameters = c(manifest_parameters, list(tier = tier_files$tier[k], sheet = tier_files$sheet[k], n_lists_threshold = primary$threshold[k])), git_commit = git_commit),
                   software = list(language = "R", language_version = paste(R.version$major, R.version$minor, sep = "."), packages = package_versions, os = paste(Sys.info()[["sysname"]], Sys.info()[["release"]])),
                   schema_diff = schema_diff,
                   timestamp = format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
                   notes = "Derived read-only from the delivered master_membership.csv, list_summary.csv and tier_summary.csv (hashed above): a projection of the master membership table onto rank, word, number_of_lists and the 45 membership columns, restricted to the delivered primary tier, with blank decision and notes columns for manual curation. Every check in the generating script passed on the written file.")
  write_yaml(manifest, paste0(output, ".manifest.yml"))
}
invisible(lapply(1:3, write_sidecar))

cat(sprintf("\n%s: %d rows x %d columns\n", tier_files$file, tier_files$rows, tier_files$cols), sep = "")
cat(sprintf("%d of %d checks pass; written to %s in %.2f min.\n", sum(checks_tbl$pass), nrow(checks_tbl), out_dir, as.numeric(difftime(run_end, run_start, units = "mins"))))
