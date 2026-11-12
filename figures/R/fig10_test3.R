# Figure S10: Test 3, the Foldseek SCOP benchmark protocol, for sequence,
# structure, embedding and whitened-embedding arms at family, superfamily and
# fold level.
#
# Input:  figures/data/fig10_test3.csv
# Output: figures/out/fig10_test3.{pdf,png}
# Run:    Rscript figures/R/fig10_test3.R   (from the repository root)
#
# All-vs-all, no split, self-hits dropped, full ranking, scored as the
# fraction of true positives found before the first different-fold hit. Every
# arm is scored on the same 13,822 domains, with the allowlist applied to the
# denominators too, so no arm is charged for true positives absent from its
# own database.

if (!exists("theme_ktn")) {
  source(if (file.exists("00_theme.R")) "00_theme.R" else "figures/R/00_theme.R")
}

d <- read_fig("fig10_test3.csv") |>
  mutate(level = factor(level, levels = c("family", "superfamily", "fold")),
         arm = factor(arm, levels = c("sequence", "structure",
                                      "embedding", "embedding +wh")))

# Order by family sensitivity, so the panels read in one direction.
ord <- d |> filter(level == "family") |> arrange(mean_sens) |> pull(method)
d <- d |> mutate(method = factor(method, levels = ord))

# MMseqs2 is the reference line every arm is measured against.
mm <- d |> filter(method == "sequence (MMseqs2)") |> select(level, mmv = mean_sens)
d <- d |> left_join(mm, by = "level")

p <- ggplot(d, aes(x = mean_sens, y = method, fill = arm)) +
  geom_vline(aes(xintercept = mmv), linetype = "22", colour = "grey55",
             linewidth = 0.35) +
  geom_col(width = 0.74) +
  geom_text(aes(label = ifelse(mean_sens < 0.01,
                               sprintf("%.4f", mean_sens),
                               sprintf("%.3f", mean_sens))),
            hjust = -0.12, size = 1.9, colour = "grey30") +
  facet_wrap(~ level, nrow = 1, scales = "free_x") +
  theme(panel.spacing.x = unit(13, "pt")) +
  scale_fill_manual(values = c("sequence" = "#000000", "structure" = "#0072B2",
                               "embedding" = "#D55E00",
                               "embedding +wh" = "#012F52"), name = NULL,
                    labels = c("embedding +wh" = "embedding +whiten")) +
  # Right-hand expansion and panel spacing keep the value label on the longest
  # bar in each facet clear of the next facet's axis.
  scale_x_continuous(expand = expansion(mult = c(0, 0.34))) +
  labs(
    x = "sensitivity up to the first false positive", y = NULL
  )

save_fig(p, "fig10_test3", width = 6.9, height = 7.2)
