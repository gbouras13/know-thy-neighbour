# Figure 1: sensitivity (TPR) at 1% FPR for every method on both benchmarks.
#
# Input:  figures/data/fig1_headline.csv
# Output: figures/out/fig1_headline.{pdf,png}
# Run:    Rscript figures/R/fig1_headline.R   (from the repository root)
#
# Test 1 bars carry query-level bootstrap intervals. Test 2 has no intervals of
# the same construction, so its bars are drawn without them.

if (!exists("theme_nnm")) {
  source(if (file.exists("00_theme.R")) "00_theme.R" else "figures/R/00_theme.R")
}

d <- read_fig("fig1_headline.csv")

# ProtTucker is marked with an asterisk: it is trained on CATH, which classifies
# the same PDB structures as SCOPe, so its Test 2 result is near-circular.
d <- d |>
  mutate(method = if_else(method == "ProtTucker/EAT", "ProtTucker/EAT *", method))

# Label any method missing from the Test 2 panel as "not evaluated", so a gap
# reads as deliberate rather than as a missing bar. Empty for the current data.
t1_only <- d |>
  group_by(method) |>
  filter(!any(benchmark == "Test 2 (SCOPe)")) |>
  ungroup() |> distinct(method) |>
  mutate(benchmark = "Test 2 (SCOPe)", tpr = 0, label = "not evaluated")

# Order by Test 1 where available, else by Test 2, so a method missing from
# either panel still gets a position rather than being dropped by the factor.
ord <- d |>
  group_by(method) |>
  summarise(key = if (any(benchmark == "Test 1 (Swiss-Prot)"))
                    tpr[benchmark == "Test 1 (Swiss-Prot)"][1]
                  else tpr[benchmark == "Test 2 (SCOPe)"][1] / 3,
            .groups = "drop") |>
  arrange(key) |> pull(method)
d <- d |> mutate(method = factor(method, levels = ord))
t1_only <- t1_only |> mutate(method = factor(method, levels = ord)) |> filter(!is.na(method))

p <- ggplot(d, aes(x = tpr, y = method, fill = family)) +
  geom_col(width = 0.72) +
  geom_errorbar(aes(xmin = lo, xmax = hi), orientation = "y", width = 0.28,
                colour = "grey25", linewidth = 0.35, na.rm = TRUE) +
  # Place the label clear of the whisker, not on top of it.
  geom_text(aes(x = pmax(tpr, ifelse(is.na(hi), tpr, hi)),
                label = sprintf("%.1f", tpr)),
            hjust = -0.25, size = 2.7, colour = "grey25") +
  geom_text(data = t1_only,
            aes(x = 1.5, y = method, label = label), inherit.aes = FALSE,
            hjust = 0, size = 2.4, colour = "grey55", fontface = "italic") +
  facet_wrap(~ benchmark, nrow = 1) +
  scale_fill_manual(values = PAL_FAMILY, name = NULL,
                    breaks = c("sequence", "structure", "embedding")) +
  scale_x_continuous(labels = pct, expand = expansion(mult = c(0, 0.16)),
                     limits = c(0, 100)) +
  labs(
    x = "Sensitivity at 1% false-positive rate", y = NULL,
    # Explanatory notes are in the paper's figure caption, not on the plot.
    caption = NULL
  )

# 6.9 x 3.5 in; scaled to the 5.5 in text width, base_size 10 renders at ~8 pt.
p <- p + theme_nnm(base_size = 10) + theme(legend.position = "top")
save_fig(p, "fig1_headline", width = 6.9, height = 3.5)
