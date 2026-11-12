# Figure S4: Test 1 sensitivity and false-positive rate by query-length bin,
# for every method.
#
# Input:  figures/data/fig4_length.csv
# Output: figures/out/figS8_length_all.{pdf,png}
# Run:    Rscript figures/R/figS8_length_all.R   (from the repository root)
#
# Methods are grouped into three columns (alignment arms / other embeddings /
# ESM family) to keep each panel readable; the two metrics are rows. All bins
# use the same pooled threshold.

if (!exists("theme_ktn")) {
  source(if (file.exists("00_theme.R")) "00_theme.R" else "figures/R/00_theme.R")
}

d <- read_fig("fig4_length.csv") |>
  mutate(short = method |>
           str_replace(" kNN$", "") |>
           str_replace("^sequence .*", "MMseqs2")) |>
  mutate(grp = case_when(
    short %in% c("MMseqs2", "Foldseek", "ProstT5-Foldseek") ~ "alignment arms",
    grepl("^ESM", short) ~ "ESM family",
    TRUE ~ "other embeddings"),
    grp = factor(grp, levels = c("alignment arms", "other embeddings",
                                 "ESM family")),
    bin = factor(bin, levels = c("1-49", "50-99", "100-199", "200-400"))) |>
  tidyr::pivot_longer(c(tpr, fpr), names_to = "metric", values_to = "value") |>
  mutate(metric = factor(metric, levels = c("tpr", "fpr"),
                         labels = c("sensitivity (TPR)",
                                    "false-positive rate (D6)")))

pal <- c("MMseqs2" = "#000000", "Foldseek" = "#0072B2",
         "ProstT5-Foldseek" = "#56B4E9",
         "gLM2-embed" = "#009E73", "ProtT5" = "#E69F00",
         "ProstT5" = "#8B7500", "ProtTucker/EAT" = "#004D40",
         "Ankh-base" = "#CC6677", "Ankh-large" = "#882255",
         "ESM-2 150M" = "#FDBF6F", "ESM-2 650M" = "#D55E00",
         "ESM-2 3B" = "#A63603", "ESM-C 300M" = "#F1B6DA",
         "ESM-C 6B" = "#CC79A7", "ESM-C 600M" = "#DE77AE")

p <- ggplot(d, aes(x = bin, y = value, colour = short, group = short)) +
  geom_line(linewidth = 0.55) +
  geom_point(size = 1.4) +
  facet_grid(metric ~ grp, scales = "free_y") +
  scale_colour_manual(values = pal, name = NULL) +
  scale_y_continuous(labels = pct) +
  guides(colour = guide_legend(nrow = 3)) +
  theme(panel.spacing = unit(8, "pt")) +
  labs(
    x = "query length (aa)", y = NULL,
    caption = NULL  # notes are in the paper's figure caption
  )

save_fig(p, "figS8_length_all", width = 6.9, height = 5.2)
