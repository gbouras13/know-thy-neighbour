# Figure S6: transform ladder on Test 2 (SCOPe). Delivered sensitivity at
# 1% FPR for each embedding model under baseline, +center, +diag, +ABTT
# (remove dominant directions) and +whiten (ZCA).
#
# Input:  figures/data/fig3b_ceiling.csv
# Output: figures/out/fig6_ladder.{pdf,png}
# Run:    Rscript figures/R/fig6_ladder.R   (from the repository root)

if (!exists("theme_ktn")) {
  source(if (file.exists("00_theme.R")) "00_theme.R" else "figures/R/00_theme.R")
}

d <- read_fig("fig3b_ceiling.csv") |>
  filter(transform %in% TRANSFORM_LEVELS, family == "embedding") |>
  mutate(transform = factor(transform, levels = TRANSFORM_LEVELS,
                            labels = TRANSFORM_LABELS[TRANSFORM_LEVELS]),
         short = str_replace(method, " kNN$", ""))

# Keep only models that actually have the full ladder.
full <- d |> count(short) |> filter(n == length(TRANSFORM_LEVELS)) |> pull(short)
d <- d |> filter(short %in% full)

# Order panels by baseline sensitivity, lowest first.
ord <- d |> filter(transform == "baseline") |> arrange(delivered) |> pull(short)
d <- d |> mutate(short = factor(short, levels = ord))

p <- ggplot(d, aes(x = transform, y = delivered, fill = transform)) +
  geom_col(width = 0.78) +
  geom_text(aes(label = sprintf("%.1f", delivered)), vjust = -0.45,
            size = 1.8, colour = "grey30") +
  # Fixed 0-60% y axis across all panels so heights are comparable between
  # models as well as within them; every value fits on this scale.
  facet_wrap(~ short, nrow = 2) +
  scale_fill_manual(values = setNames(PAL_TRANSFORM[TRANSFORM_LEVELS],
                                      TRANSFORM_LABELS[TRANSFORM_LEVELS]),
                    guide = "none") +
  scale_y_continuous(labels = pct, limits = c(0, 60), breaks = seq(0, 60, 20),
                     expand = expansion(mult = c(0, 0.02))) +
  # Transform names are angled to fit under the narrow panels.
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 6.4),
        panel.spacing.x = unit(7, "pt")) +
  labs(
    x = NULL, y = "sensitivity at 1% FPR"
  )

save_fig(p, "fig6_ladder", width = 6.9, height = 4.4)
