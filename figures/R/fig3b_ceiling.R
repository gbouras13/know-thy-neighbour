# Figure 3: ranking ceiling (top-1) vs delivered sensitivity at 1% FPR on
# Test 2 (SCOPe), per method, as run and after ZCA whitening.
#
# Input:  figures/data/fig3b_ceiling.csv
# Output: figures/out/fig3b_ceiling.{pdf,png}
# Run:    Rscript figures/R/fig3b_ceiling.R   (from the repository root)
#
# Dumbbell layout: model names sit on the y axis, and each segment runs from
# delivered sensitivity (filled dot) to top-1 accuracy (hollow dot), so the gap
# between ranking and delivered answers is drawn as a length. Where a whitened
# variant exists, each model has two rows: as run (upper) and after ZCA
# whitening (lower).

if (!exists("theme_ktn")) {
  source(if (file.exists("00_theme.R")) "00_theme.R" else "figures/R/00_theme.R")
}

d <- read_fig("fig3b_ceiling.csv") |>
  mutate(short = method |>
           str_replace(" kNN$", "") |>
           # Use plain ASCII in labels: the prime U+2032 has no glyph in the
           # pdf device's default encoding.
           str_replace("^sequence .*", "MMseqs2"))

# The whitened series needs a hue distinct from the three family colours
# (black / blue / orange); Okabe-Ito reddish purple is distinct and
# colour-blind safe.
WHITEN_COL <- "#CC79A7"

base <- d |> filter(transform == "none")
whit <- d |> filter(transform == "whiten")

# Order by delivered sensitivity, best at top.
lev <- base |> arrange(delivered) |> pull(short)
NUDGE <- 0.21
mk <- function(df, dy, tag) {
  df |>
    mutate(short = factor(short, levels = lev),
           y = as.numeric(short) + dy,
           run = tag)
}
b <- mk(base, NUDGE, "baseline")
w <- mk(whit, -NUDGE, "+ZCA whitening")
both <- bind_rows(b, w) |>
  mutate(run = factor(run, levels = c("baseline", "+ZCA whitening")))

p <- ggplot(both) +
  geom_segment(aes(x = delivered, xend = top1, y = y, yend = y, colour = run),
               linewidth = 0.9, lineend = "round", alpha = 0.55) +
  # hollow = what the representation can find; filled = what a user is given
  geom_point(aes(x = top1, y = y, colour = run), shape = 21, fill = "white",
             size = 1.9, stroke = 0.7) +
  geom_point(data = b, aes(x = delivered, y = y, fill = family),
             shape = 21, size = 2.5, stroke = 0.5, colour = "white") +
  geom_point(data = w, aes(x = delivered, y = y),
             shape = 21, size = 2.5, stroke = 0.5, colour = "white",
             fill = WHITEN_COL) +
  # Delivered value printed to the left of each filled dot so it never sits on
  # its own segment. The x limit extends to -11 so small labels do not clip.
  geom_text(data = both,
            aes(x = delivered, y = y, label = sprintf("%.1f", delivered),
                colour = run),
            hjust = 1.3, size = 2.35, show.legend = FALSE) +
  # top-1 value printed at each hollow dot, since nearby dots are hard to read
  # off the axis.
  geom_text(data = both, aes(x = top1, y = y, label = sprintf("%.1f", top1),
                             colour = run),
            hjust = -0.35, size = 2.35, show.legend = FALSE) +
  scale_y_continuous(breaks = seq_along(lev), labels = lev,
                     expand = expansion(add = 0.55)) +
  # Headroom for the top-1 labels; breaks stop at 100 so the axis still reads
  # as a percentage scale.
  scale_x_continuous(labels = pct, limits = c(-11, 104),
                     breaks = seq(0, 100, 25),
                     expand = expansion(mult = c(0.01, 0))) +
  # Grey for the as-run series, WHITEN_COL for the whitened series.
  scale_colour_manual(values = c("baseline" = "grey55",
                                 "+ZCA whitening" = WHITEN_COL),
                      name = NULL) +
  scale_fill_manual(values = PAL_FAMILY, name = NULL,
                    breaks = c("sequence", "structure", "embedding"),
                    guide = guide_legend(order = 1)) +
  guides(colour = guide_legend(order = 2, override.aes = list(shape = NA))) +
  labs(
    x = "percentage of answerable queries",
    y = NULL
  )

# 6.9 x 4.3 in (about 3.4 in tall at the 5.5 in text width). The legend label
# "baseline" matches the transform-ladder figures.
p <- p + theme_ktn(base_size = 10) + theme(legend.position = "top")
save_fig(p, "fig3b_ceiling", width = 6.9, height = 4.3)
