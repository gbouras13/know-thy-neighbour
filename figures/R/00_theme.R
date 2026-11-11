# Shared theme, palette and helpers for every figure.
#
# Design constraints:
#   * method family (sequence / structure / embedding) drives colour, so the
#     three tiers are readable without the legend;
#   * intervals are drawn wherever they exist, and their absence is visible;
#   * colours are colour-blind-safe (Okabe-Ito) and survive greyscale printing.
#
# Sourced by every fig*.R. Contains no data loading.

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(readr)
  library(scales)
  library(forcats)
  library(stringr)
})

`%||%` <- function(a, b) if (is.null(a)) b else a

# Resolve the repository root by walking up from the working directory until a
# directory containing figures/R/00_theme.R is found, so scripts work from the
# repository root (`Rscript figures/R/<script>.R`), from inside figures/R/ and
# from RStudio.
PROJ <- local({
  d <- normalizePath(getwd(), mustWork = TRUE)
  repeat {
    if (file.exists(file.path(d, "figures", "R", "00_theme.R"))) return(d)
    parent <- dirname(d)
    if (identical(parent, d)) break
    d <- parent
  }
  stop("cannot find the repository root above ", getwd(),
       " -- run from inside the repository.", call. = FALSE)
})
DATA <- file.path(PROJ, "figures", "data")
OUT  <- file.path(PROJ, "figures", "out")
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

# --- palette ---------------------------------------------------------------
# Okabe-Ito. Sequence is the reference method and gets the strongest colour.
PAL_FAMILY <- c(
  "sequence"  = "#000000",
  "structure" = "#0072B2",
  "embedding" = "#D55E00"
)

# Transform ladder: ordered light -> dark so the ordering of the transforms
# reads as a gradient.
PAL_TRANSFORM <- c(
  "none"   = "#CCCCCC",
  "center" = "#9AD0F5",
  "diag"   = "#56B4E9",
  "abtt"   = "#0072B2",
  "whiten" = "#012F52"
)

PAL_CLASS <- c(
  "real remote homolog"      = "#009E73",
  "D6 natural hard negative" = "#D55E00",
  "D2 shuffled decoy"        = "#CC79A7"
)

TRANSFORM_LEVELS <- c("none", "center", "diag", "abtt", "whiten")
TRANSFORM_LABELS <- c(none = "baseline", center = "+center", diag = "+diag",
                      abtt = "+ABTT", whiten = "+whiten")

# --- theme -----------------------------------------------------------------
theme_nnm <- function(base_size = 9, base_family = "") {
  theme_minimal(base_size = base_size, base_family = base_family) +
    theme(
      panel.grid.minor   = element_blank(),
      panel.grid.major.y = element_line(linewidth = 0.25, colour = "grey92"),
      panel.grid.major.x = element_line(linewidth = 0.25, colour = "grey92"),
      axis.title         = element_text(size = rel(1.0)),
      axis.text          = element_text(colour = "grey20"),
      plot.title         = element_text(face = "bold", size = rel(1.15),
                                        margin = margin(b = 3)),
      plot.subtitle      = element_text(colour = "grey35", size = rel(0.95),
                                        margin = margin(b = 8)),
      plot.caption       = element_text(colour = "grey40", size = rel(0.8),
                                        hjust = 0, margin = margin(t = 8)),
      plot.caption.position = "plot",
      plot.title.position   = "plot",
      strip.text         = element_text(face = "bold", size = rel(0.95)),
      legend.position    = "top",
      legend.title       = element_text(size = rel(0.9)),
      legend.key.height  = unit(9, "pt"),
      legend.margin      = margin(b = 0)
    )
}

theme_set(theme_nnm())

# --- helpers ---------------------------------------------------------------

read_fig <- function(name) {
  p <- file.path(DATA, name)
  # Large inputs are shipped gzipped; readr reads either form.
  if (!file.exists(p) && file.exists(paste0(p, ".gz"))) p <- paste0(p, ".gz")
  if (!file.exists(p)) {
    stop(sprintf(
      "%s is missing. Regenerate the figure inputs first:\n  python3 scripts/27_figure_data.py",
      name), call. = FALSE)
  }
  readr::read_csv(p, show_col_types = FALSE, progress = FALSE)
}

#' Save one figure at a fixed physical size.
#'
#' Writes both PDF and PNG to figures/out/. Width defaults to a single
#' column; pass width = 6.9 for full width.
save_fig <- function(plot, name, width = 3.4, height = 3.0, dpi = 400) {
  for (ext in c("pdf", "png")) {
    ggsave(file.path(OUT, paste0(name, ".", ext)), plot,
           width = width, height = height, dpi = dpi,
           device = ext, bg = "white")
  }
  message(sprintf("  wrote %s.{pdf,png}  (%.1f x %.1f in)", name, width, height))
  invisible(plot)
}

#' Order a method factor by a value column, descending.
by_value <- function(df, method_col = "method", value_col = "tpr") {
  df[[method_col]] <- forcats::fct_reorder(df[[method_col]], df[[value_col]],
                                           .fun = max, .desc = FALSE)
  df
}

#' Percent axis for values ALREADY on a 0-100 scale (e.g. tpr = 34.7).
#' It does NOT multiply by 100. For 0-1 proportions use
#' scales::label_percent() instead -- passing proportions here silently
#' renders every value below 0.5 as "0%".
pct <- function(...) scales::label_number(accuracy = 1, suffix = "%")(...)
