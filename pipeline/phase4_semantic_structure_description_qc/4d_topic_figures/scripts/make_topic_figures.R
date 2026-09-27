# Figures for the largest topic clusters.
#
# 1. topic_structure.png: the full-corpus 2-D UMAP from Phase 3 with the separate islands
#    circled, the core zoomed with the largest topics numbered, and a table of those topics.
# 2. top_topics_umap.png: the largest topics re-embedded on their own
#    (output of run_top_topics_umap.py, min_dist 0.5).
#
# Topic and island names were written by hand from each cluster's c-TF-IDF keywords (4a).
# Inputs are only read. Env: KC_REPO_ROOT (default: working directory). Each input path can
# also be overridden: KC_P3_RESULTS, KC_4D_RESULTS. Figures go to KC_4D_FIGURES.

suppressMessages({
  library(arrow); library(ggplot2); library(ggforce); library(ggrepel); library(patchwork); library(jsonlite)
})

BASE <- Sys.getenv("KC_REPO_ROOT", unset = getwd())
STEP <- file.path(BASE, "pipeline/phase4_semantic_structure_description_qc/4d_topic_figures")
P3 <- Sys.getenv("KC_P3_RESULTS",
  unset = file.path(BASE, "pipeline/phase3_blind_semantic_structure_discovery/full_corpus_gpu_umap_hdbscan_baseline/results"))
R4D <- Sys.getenv("KC_4D_RESULTS", unset = file.path(STEP, "results"))
FIG <- Sys.getenv("KC_4D_FIGURES", unset = file.path(STEP, "figures"))
dir.create(FIG, showWarnings = FALSE, recursive = TRUE)

# Names for the largest topic clusters, keyed by cluster id.
TOPIC_NAMES <- c(
  "655" = "Trump and the MAGA movement", "338" = "World Cup soccer", "2355" = "Health insurance and Medicare",
  "2909" = "American identity", "3136" = "Graham Platner's Senate campaign", "2169" = "Religion and Christianity",
  "679" = "Daylight saving time", "1950" = "Tattoos and extremist symbols", "5166" = "Donald Trump (general)",
  "4451" = "Democratic Socialists of America", "2313" = "Canada-US trade", "1640" = "War with Iran",
  "3777" = "Sexual violence", "5148" = "The Republican Party", "2428" = "Midterm elections")

# Islands outside the core, named from their clusters' keywords.
ISLANDS <- list(
  "Removed-post placeholders" = c(160, 380, 46, 17),
  "Moderator bot notices" = c(85, 566, 333, 564, 559),
  "Removed by Reddit\n(content policy)" = c(157, 158),
  "Deleted posts" = c(454),
  "Data centers and AI" = c(1919))

FULL_X <- c(-18.5, 17); FULL_Y <- c(-19.5, 21)   # central viewport of the full map
ZOOM_X <- c(-6.6, 8.3); ZOOM_Y <- c(-4.6, 6.6)   # the core
INK <- "#1f2328"; MUTED <- "#57606a"

u <- as.data.frame(read_parquet(file.path(P3, "umap_2d_visualization_coordinates.parquet")))
l <- as.data.frame(read_parquet(file.path(P3, "hdbscan_labels.parquet")))
stopifnot(identical(u$doc_key, l$doc_key))
u$label <- l$label

tids <- as.integer(fromJSON(file.path(R4D, "summary.json"))$topics)
stopifnot(all(as.character(tids) %in% names(TOPIC_NAMES)))
sizes <- table(u$label[u$label >= 0])
topics <- data.frame(rank = seq_along(tids), id = tids, name = unname(TOPIC_NAMES[as.character(tids)]),
                     n = as.integer(sizes[as.character(tids)]))

theme_map <- function() theme_minimal(base_size = 11) +
  theme(panel.grid = element_blank(), axis.text = element_blank(), legend.position = "bottom",
        legend.key.width = unit(14, "mm"), plot.title = element_text(face = "bold", colour = INK),
        plot.caption = element_text(colour = MUTED, hjust = 0))
fill_scale <- function() scale_fill_viridis_c(trans = "log10", name = "Posts per bin", labels = scales::label_comma())
in_box <- function(d, xl, yl) d[d$umap_0 > xl[1] & d$umap_0 < xl[2] & d$umap_1 > yl[1] & d$umap_1 < yl[2], ]

# --- Figure 1a: full map with circled islands --------------------------------------------
set.seed(1)
isl <- do.call(rbind, lapply(names(ISLANDS), function(nm) {
  s <- u[u$label %in% ISLANDS[[nm]], c("umap_0", "umap_1")]
  cx <- median(s$umap_0); cy <- median(s$umap_1)
  s <- s[sqrt((s$umap_0 - cx)^2 + (s$umap_1 - cy)^2) < 3, ]   # dense core only, for the ellipse
  if (nrow(s) > 1500) s <- s[sample(nrow(s), 1500), ]
  s$grp <- nm
  s$lab <- paste0(nm, "\n", format(sum(u$label %in% ISLANDS[[nm]]), big.mark = ","), " posts")
  s
}))
p_full <- ggplot(in_box(u, FULL_X, FULL_Y), aes(umap_0, umap_1)) +
  geom_bin_2d(bins = 260) + fill_scale() +
  geom_mark_ellipse(data = isl, aes(group = grp, label = lab), expand = unit(3, "mm"),
    colour = INK, linewidth = 0.5, label.fontsize = 9, label.fontface = "plain", label.colour = INK,
    con.colour = INK, label.fill = "white", label.buffer = unit(6, "mm"), label.minwidth = unit(30, "mm")) +
  annotate("rect", xmin = ZOOM_X[1], xmax = ZOOM_X[2], ymin = ZOOM_Y[1], ymax = ZOOM_Y[2],
           fill = NA, colour = INK, linewidth = 0.4, linetype = "22") +
  coord_equal(xlim = FULL_X, ylim = FULL_Y, expand = FALSE) +
  labs(x = "UMAP 1", y = "UMAP 2", title = sprintf("All %s posts and comments", format(nrow(u), big.mark = ",")),
       caption = "UMAP: cosine distance, n_neighbors = 15, min_dist = 0") +
  theme_map()

# --- Figure 1b: zoomed core with numbered topics -----------------------------------------
tm <- do.call(rbind, lapply(seq_along(tids), function(k) {
  s <- u[u$label == tids[k], ]
  data.frame(k = k, x = median(s$umap_0), y = median(s$umap_1), spread = IQR(s$umap_0) > 1)
}))
spread_note <- paste(sprintf("#%d (%s)", tm$k[tm$spread], topics$name[tm$spread]), collapse = ", ")
tm <- tm[!tm$spread, ]
p_zoom <- ggplot(in_box(u, ZOOM_X, ZOOM_Y), aes(umap_0, umap_1)) +
  geom_bin_2d(binwidth = c(0.06, 0.06)) + fill_scale() +
  geom_point(data = tm, aes(x, y), inherit.aes = FALSE, shape = 21, size = 6.5, stroke = 0.6, colour = INK, fill = NA) +
  geom_label_repel(data = tm, aes(x, y, label = k), inherit.aes = FALSE, size = 3.4, colour = INK, fill = "white",
    label.size = 0.2, label.padding = unit(0.12, "lines"), box.padding = 0.5, point.padding = 0.6,
    min.segment.length = 0, segment.colour = INK, segment.size = 0.3, seed = 3, max.overlaps = Inf) +
  coord_equal(xlim = ZOOM_X, ylim = ZOOM_Y, expand = FALSE) +
  labs(x = "UMAP 1", y = "UMAP 2", title = "Core, zoomed",
       caption = if (nzchar(spread_note)) sprintf("Topic numbers match the table; %s is spread across the map", spread_note)
                 else "Topic numbers match the table") +
  theme_map() + theme(panel.border = element_rect(fill = NA, colour = INK, linetype = "22", linewidth = 0.4))

# --- Figure 1c: topic table and cluster statistics ---------------------------------------
n_clusters <- length(sizes); n_in <- sum(u$label >= 0); n_noise <- sum(u$label < 0)
stats <- sprintf("HDBSCAN clusters: %s\nPosts in a cluster: %s (%.1f%%)\nUnclustered posts: %s\nLargest cluster: %s removed-post placeholders",
  format(n_clusters, big.mark = ","), format(n_in, big.mark = ","), 100 * n_in / nrow(u),
  format(n_noise, big.mark = ","), format(max(sizes), big.mark = ","))
p_tab <- ggplot(topics, aes(y = rank)) +
  geom_text(aes(x = 0, label = rank), hjust = 0, size = 3.7, colour = INK) +
  geom_text(aes(x = 0.07, label = name), hjust = 0, size = 3.7, colour = INK) +
  geom_text(aes(x = 1, label = format(n, big.mark = ",")), hjust = 1, size = 3.7, colour = INK) +
  annotate("text", x = c(0, 0.07, 1), y = 0, label = c("#", "Topic", "Posts"), hjust = c(0, 0, 1),
           fontface = "bold", size = 3.7, colour = INK) +
  annotate("segment", x = 0, xend = 1, y = 0.5, yend = 0.5, colour = "#d0d7de") +
  annotate("text", x = 0, y = nrow(topics) + 1.2, hjust = 0, vjust = 1, size = 3.3, colour = INK,
           lineheight = 1.15, label = stats) +
  scale_y_reverse(limits = c(nrow(topics) + 12, -0.6)) + scale_x_continuous(limits = c(0, 1.02)) +
  labs(title = "Largest topic clusters") +
  theme_void(base_size = 11) +
  theme(plot.title = element_text(face = "bold", colour = INK, margin = margin(b = 8)),
        plot.margin = margin(10, 20, 10, 10))

ggsave(file.path(FIG, "topic_structure.png"), p_full + p_zoom + p_tab + plot_layout(widths = c(1, 1.15, 0.72)),
       width = 22, height = 9.5, dpi = 150, bg = "white")

# --- Figure 2: the largest topics re-embedded on their own -------------------------------
d <- as.data.frame(read_parquet(file.path(R4D, "top_topics_umap_nn15_md0.5.parquet")))
d$name <- TOPIC_NAMES[as.character(d$label)]
lab <- aggregate(cbind(umap_0, umap_1) ~ name, d, median)
lab$txt <- paste0(lab$name, " (", format(as.integer(table(d$name)[lab$name]), big.mark = ","), ")")
set.seed(7)
obs <- d[sample(nrow(d), min(4000, nrow(d))), c("umap_0", "umap_1")]   # labels avoid these points
obs$txt <- ""
rep_df <- rbind(lab[, c("umap_0", "umap_1", "txt")], obs)
p_top <- ggplot(d, aes(umap_0, umap_1)) +
  geom_point(size = 0.5, alpha = 0.45, colour = "#21667a", stroke = 0) +
  geom_text_repel(data = rep_df, aes(label = txt), size = 3.5, colour = INK, bg.colour = "white", bg.r = 0.12,
    box.padding = 0.5, point.padding = 0.2, force = 3, min.segment.length = 0.2, segment.colour = "#8c959f",
    segment.size = 0.3, seed = 11, max.overlaps = Inf, max.time = 5, max.iter = 50000) +
  coord_equal(clip = "off") +
  labs(title = sprintf("The %d largest topic clusters (n = %s posts)", length(tids), format(nrow(d), big.mark = ",")),
       x = "UMAP 1", y = "UMAP 2",
       caption = "UMAP: cosine distance, n_neighbors = 15, min_dist = 0.5, random_state = 42") +
  theme_minimal(base_size = 11) +
  theme(panel.grid = element_blank(), axis.text = element_blank(), axis.title = element_text(colour = INK),
        panel.border = element_rect(fill = NA, colour = "#8c959f"), plot.caption = element_text(colour = MUTED, hjust = 0),
        plot.title = element_text(face = "bold", colour = INK), plot.margin = margin(12, 30, 12, 30))
ggsave(file.path(FIG, "top_topics_umap.png"), p_top, width = 12, height = 11, dpi = 150, bg = "white")
cat("figures written to", FIG, "\n")
