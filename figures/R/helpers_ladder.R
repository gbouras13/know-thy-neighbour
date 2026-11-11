# Shared helpers for the transform-ladder figures (fig6_ladder.R,
# figS9_ladder_test1.R, figS11_ladder_top1_test1.R, figS10_ladder_top1_test2.R).
# Kept separate so no figure script re-executes another's save_fig.
#
# One builder covers {Test 1, Test 2} x {delivered, top-1}. The metrics are not
# interchangeable across benchmarks: Test 2 checks that the top-1 hit is in the
# same superfamily, whereas Test 1 "delivered" does not check correctness
# (see scripts/43_test1_top1.py).

load_ladder <- function(file) {
  read_fig(file) |>
    filter(transform %in% TRANSFORM_LEVELS) |>
    mutate(transform = factor(transform, levels = TRANSFORM_LEVELS,
                              labels = TRANSFORM_LABELS[TRANSFORM_LEVELS]),
           short = model)
}

# keep only models carrying the full five-rung ladder, ordered by baseline
ladder_complete <- function(d, value) {
  full <- d |> group_by(short) |> filter(n() == 5) |> ungroup()
  ord <- full |> filter(transform == "baseline") |>
    arrange(.data[[value]]) |> pull(short)
  full |> mutate(short = factor(short, levels = ord))
}

draw_ladder <- function(d, value, ylab, ymax) {
  ggplot(d, aes(x = transform, y = .data[[value]], fill = transform)) +
    geom_col(width = 0.78) +
    geom_text(aes(label = sprintf("%.1f", .data[[value]])), vjust = -0.45,
              size = 1.8, colour = "grey30") +
    facet_wrap(~ short, nrow = 2) +
    scale_fill_manual(values = setNames(PAL_TRANSFORM[TRANSFORM_LEVELS],
                                        TRANSFORM_LABELS[TRANSFORM_LEVELS]),
                      guide = "none") +
    scale_y_continuous(labels = pct, limits = c(0, ymax),
                       breaks = scales::pretty_breaks(4),
                       expand = expansion(mult = c(0, 0.02))) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 6.4),
          panel.spacing.x = unit(7, "pt")) +
    labs(x = NULL, y = ylab)
}
