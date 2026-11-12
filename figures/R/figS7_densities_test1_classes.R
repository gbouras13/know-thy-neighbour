# Figure S2: Test 1 best-hit score densities for every method and all eight
# populations (real homologs plus the full D0-D6 decoy ladder).
#
# Input:  figures/data/figS7_densities_all_classes.csv
# Output: figures/out/figS7_densities_test1_classes.{pdf,png}
# Run:    Rscript figures/R/figS7_densities_test1_classes.R   (from the repository root)
#
# Extends Fig S1 (real / D6 / D2) with D0 uniform, D1 composition, D3 reverse
# and D4/D5 Markov decoys. Curves are unfilled because eight populations
# overplot.

if (!exists("theme_ktn")) {
  source(if (file.exists("00_theme.R")) "00_theme.R" else "figures/R/00_theme.R")
}

PAL_LADDER <- c("real"             = "#009E73",
                "D6 natural hard"  = "#D55E00",
                "D0 uniform"       = "#999999",
                "D1 composition"   = "#B8A03E",
                "D2 shuffle"       = "#CC79A7",
                "D3 reverse"       = "#8B5AA8",
                "D4 markov2"       = "#5D9FC7",
                "D5 markov3"       = "#1B587C")

ORDER <- c("sequence (MMseqs2)", "Foldseek", "ProstT5-Foldseek",
           "gLM2-embed", "ProstT5 kNN", "ProtT5 kNN", "ProtTucker/EAT",
           "Ankh-base kNN", "Ankh-large kNN", "ESM-2 150M kNN",
           "ESM-2 650M kNN", "ESM-2 3B kNN", "ESM-C 300M kNN",
           "ESM-C 600M kNN", "ESM-C 6B kNN")

d <- read_fig("figS7_densities_all_classes.csv") |>
  mutate(class = factor(class, levels = names(PAL_LADDER)),
         x = ifelse(scale == "E-value",
                    pmin(-log10(pmax(score, 1e-200)), 60), score),
         method = factor(method, levels = ORDER)) |>
  filter(!is.na(method), !is.na(class))

p <- ggplot(d, aes(x = x, colour = class)) +
  geom_density(linewidth = 0.45, adjust = 1.1, fill = NA) +
  facet_wrap(~ method, scales = "free", nrow = 5) +
  scale_colour_manual(values = PAL_LADDER, name = NULL) +
  guides(colour = guide_legend(nrow = 1, override.aes = list(linewidth = 1.4))) +
  theme(strip.text = element_text(size = 6.4),
        axis.text = element_text(size = 5.4),
        legend.text = element_text(size = 6.6)) +
  labs(
    x = "best-hit score  (alignment arms: -log10 E-value, capped at 60; embeddings: cosine; ProtTucker: negative Euclidean)",
    y = "density"
  )

save_fig(p, "figS7_densities_test1_classes", width = 6.9, height = 8.6)
