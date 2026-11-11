# Figure S11: per-query distribution of Test 3 sensitivity up to the first
# false positive, at family, superfamily and fold level.
#
# Input:  figures/data/fig9_test3_cdf.csv
# Output: figures/out/fig9_scope_cdf.{pdf,png}
# Run:    Rscript figures/R/fig9_scope_cdf.R   (from the repository root)
#
# Uses the same all-vs-all Test 3 protocol and 5,003-query common set as
# Fig S10, whose bars show the means of these distributions.
#
# y = fraction of queries achieving at least x sensitivity. The left edge shows
# how often a method returns nothing useful, the right edge how often it is
# perfect; a mean cannot distinguish these shapes.
#
# Ten series are shown for readability: both structure arms, the sequence
# baseline, the supervised reference (ProtTucker), and three pLMs (Ankh-base,
# ESM-2 650M, ESM-C 6B) with and without whitening.

if (!exists("theme_nnm")) {
  source(if (file.exists("00_theme.R")) "00_theme.R" else "figures/R/00_theme.R")
}

d <- read_fig("fig9_test3_cdf.csv") |>
  mutate(
    level = factor(level, levels = c("family", "superfamily", "fold")),
    whitened = grepl("\\+whiten", method),
    base = str_replace(method, " \\+whiten$", ""),
    base = factor(base, levels = c("sequence (MMseqs2)", "Foldseek",
                                   "ProstT5-Foldseek", "ProtTucker/EAT",
                                   "Ankh-base", "ESM-2 650M", "ESM-C 6B")))

pal <- c("sequence (MMseqs2)" = "#000000", "Foldseek" = "#0072B2",
         "ProstT5-Foldseek" = "#56B4E9", "ProtTucker/EAT" = "#009E73",
         "Ankh-base" = "#CC79A7", "ESM-2 650M" = "#D55E00",
         "ESM-C 6B" = "#8B5A2B")

p <- ggplot(d, aes(x = sens, y = frac_queries, colour = base,
                   linetype = whitened, group = method)) +
  geom_step(linewidth = 0.55, direction = "hv") +
  facet_wrap(~ level, nrow = 1) +
  scale_colour_manual(values = pal, name = NULL) +
  scale_linetype_manual(values = c(`FALSE` = "solid", `TRUE` = "22"),
                        labels = c(`FALSE` = "baseline", `TRUE` = "+whiten"),
                        name = NULL) +
  scale_x_continuous(labels = scales::label_percent(accuracy = 1),
                     breaks = c(0, 0.25, 0.5, 0.75, 1)) +
  scale_y_continuous(labels = scales::label_percent(accuracy = 1),
                     limits = c(0, 1)) +
  guides(colour = guide_legend(order = 1, nrow = 2),
         linetype = guide_legend(order = 2,
                                 override.aes = list(colour = "grey30"))) +
  theme(panel.spacing.x = unit(10, "pt"),
        legend.box = "vertical", legend.spacing.y = unit(1, "pt")) +
  labs(
    x = "sensitivity up to the first false positive",
    y = "fraction of queries reaching at least x"
  )

save_fig(p, "fig9_scope_cdf", width = 6.9, height = 4.6)
