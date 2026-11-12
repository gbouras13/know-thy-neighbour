# Figure 2: best-hit score densities on Test 1 for real remote homologs,
# natural hard negatives and shuffled decoys, for six representative methods.
#
# Input:  figures/data/fig2_densities.csv
# Output: figures/out/fig2_densities.{pdf,png}
# Run:    Rscript figures/R/fig2_densities.R   (from the repository root)
#
# Densities are drawn on the score each method ranks by. Score scales differ
# per method (E-value vs cosine), so each panel has its own x axis; no single
# cutoff means the same thing across methods.

if (!exists("theme_ktn")) {
  source(if (file.exists("00_theme.R")) "00_theme.R" else "figures/R/00_theme.R")
}

d <- read_fig("fig2_densities.csv") |>
  mutate(
    class = factor(class, levels = names(PAL_CLASS)),
    # E-values are plotted as -log10(E), with the display capped at 60 so the
    # strongest hits do not flatten the panel. Nothing is dropped: capped values
    # pile up at the right edge. Cosines are plotted unchanged.
    x = ifelse(scale == "E-value", pmin(-log10(pmax(score, 1e-200)), 60), score),
    method = factor(method, levels = c("sequence (MMseqs2)", "ProstT5-Foldseek",
                                       "gLM2-embed", "Ankh-base kNN",
                                       "ESM-2 650M kNN", "ESM-C 6B kNN"))
  ) |>
  # the csv carries all fourteen methods (see Fig S1); keep six here
  filter(!is.na(method)

  )

p <- ggplot(d, aes(x = x, fill = class, colour = class)) +
  geom_density(alpha = 0.35, linewidth = 0.4, adjust = 1.1) +
  # 2 rows x 3 columns keeps the figure short on the page.
  facet_wrap(~ method, scales = "free", nrow = 2) +
  scale_fill_manual(values = PAL_CLASS, name = NULL) +
  scale_colour_manual(values = PAL_CLASS, name = NULL) +
  guides(fill = guide_legend(nrow = 1), colour = guide_legend(nrow = 1)) +
  labs(
    x = "score  (alignment arms: -log10 E-value; embeddings: cosine to nearest neighbour)",
    y = "density",
    # Explanatory notes are in the paper's figure caption, not on the plot.
    caption = NULL
  )

# 6.9 x 4.0 in (about 3.2 in tall at the 5.5 in text width).
p <- p + theme_ktn(base_size = 10) + theme(legend.position = "top")
save_fig(p, "fig2_densities", width = 6.9, height = 4.0)
