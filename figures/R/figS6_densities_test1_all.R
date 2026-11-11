# Figure S1: Figure 2 extended to every Test 1 method.
#
# Input:  figures/data/fig2_densities.csv
# Output: figures/out/figS6_densities_test1_all.{pdf,png}
# Run:    Rscript figures/R/figS6_densities_test1_all.R   (from the repository root)
#
# Same three populations as Figure 2: real remote homologs, D6 natural hard
# negatives (real clan-absent proteins) and D2 per-query shuffles; best score
# per query, all three length arms pooled. Foldseek shows two of the three
# classes: its D6 negatives were searched, but synthetic decoys were not.

if (!exists("theme_nnm")) {
  source(if (file.exists("00_theme.R")) "00_theme.R" else "figures/R/00_theme.R")
}

ORDER <- c("sequence (MMseqs2)", "Foldseek", "ProstT5-Foldseek",
           "gLM2-embed", "ProstT5 kNN", "ProtT5 kNN", "ProtTucker/EAT",
           "Ankh-base kNN", "Ankh-large kNN", "ESM-2 150M kNN",
           "ESM-2 650M kNN", "ESM-2 3B kNN", "ESM-C 300M kNN",
           "ESM-C 600M kNN", "ESM-C 6B kNN")

d <- read_fig("fig2_densities.csv") |>
  mutate(class = factor(class, levels = names(PAL_CLASS)),
         x = ifelse(scale == "E-value",
                    pmin(-log10(pmax(score, 1e-200)), 60), score),
         method = factor(method, levels = ORDER)) |>
  filter(!is.na(method))

p <- ggplot(d, aes(x = x, fill = class, colour = class)) +
  geom_density(alpha = 0.35, linewidth = 0.4, adjust = 1.1) +
  facet_wrap(~ method, scales = "free", nrow = 5) +
  scale_fill_manual(values = PAL_CLASS, name = NULL) +
  scale_colour_manual(values = PAL_CLASS, name = NULL) +
  guides(fill = guide_legend(nrow = 1), colour = guide_legend(nrow = 1)) +
  theme(strip.text = element_text(size = 6.4),
        axis.text = element_text(size = 5.4)) +
  labs(
    x = "best-hit score  (alignment arms: -log10 E-value, capped at 60; embeddings: cosine; ProtTucker: negative Euclidean)",
    y = "density",
    caption = NULL  # notes are in the paper's figure caption
  )

save_fig(p, "figS6_densities_test1_all", width = 6.9, height = 8.6)
