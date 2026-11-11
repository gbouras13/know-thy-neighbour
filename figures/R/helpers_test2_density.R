# Shared helpers for the Test 2 score-density figures
# (e.g. figS5_densities_test2_all.R). Kept separate so no figure script
# re-executes another's save_fig.

PAL_T2CLASS <- c("top-1 correct" = "#009E73",
                 "top-1 wrong fold" = "#D55E00",
                 "shuffled query" = "#CC79A7")

load_t2dens <- function(keep = NULL) {
  d <- read_fig("fig12_densities_test2.csv") |>
    mutate(class = factor(class, levels = names(PAL_T2CLASS)),
           x = ifelse(scale == "E-value",
                      pmin(-log10(pmax(score, 1e-200)), 60), score))
  if (!is.null(keep)) d <- d |> filter(method %in% keep) |>
      mutate(method = factor(method, levels = keep))
  d
}

draw_t2dens <- function(d, nrow) {
  ggplot(d, aes(x = x, fill = class, colour = class)) +
    geom_density(alpha = 0.35, linewidth = 0.4, adjust = 1.1) +
    facet_wrap(~ method, scales = "free", nrow = nrow) +
    scale_fill_manual(values = PAL_T2CLASS, name = NULL) +
    scale_colour_manual(values = PAL_T2CLASS, name = NULL) +
    guides(fill = guide_legend(nrow = 1), colour = guide_legend(nrow = 1)) +
    labs(x = paste("best-hit score  (alignment arms: -log10 E-value,",
                   "capped at 60; embeddings: cosine; ProtTucker: negative Euclidean)"),
         y = "density")
}

