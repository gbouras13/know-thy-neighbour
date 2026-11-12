# Figure S5: hubness for every method. Share of the searched reference set that
# appears in at least one query's top-50 list, for real queries and for decoy
# queries, on Test 1 (three length arms pooled) and Test 2 (SCOPe), with no
# embedding transform.
#
# Input:  figures/data/figS2_hubness.csv
#         (built by scripts/27_figure_data.py from results/hubness.tsv and
#         results/hubness_alignment.tsv)
# Output: figures/out/figS9_hubness_all.{pdf,png}
# Run:    Rscript figures/R/figS9_hubness_all.R   (from the repository root)
#
# The alignment arms (MMseqs2, Foldseek, ProstT5-Foldseek) sit left of a
# separator under their own header, because their lists differ from kNN lists:
#   * a kNN list always has exactly 50 entries; an alignment list holds whatever
#     cleared the prefilter and E <= 1000, so its top 50 by bit score is a rank
#     list, and some queries receive fewer than 50 hits;
#   * the denominator is the database each arm searched: 296,807 sequences for
#     MMseqs2, 289,206 AFDB models for both Foldseek arms on Test 1, and 9,787
#     domains with coordinates for Foldseek on Test 2. Foldseek has no synthetic
#     decoys (a shuffled sequence has no structure) and no Test 2 decoys; that
#     slot is labelled rather than left blank.
# (A facet column would not work for this split: facet_grid shares one x scale
# down a column, so the two benchmark rows could not have their own labels.)
#
# Within each group, methods are ordered by Test 2 decoy coverage, most
# concentrated first, and the same order is used in the Test 1 row. The kNN
# decoy population excludes the bare D6_natural_hard class. This figure is a
# measurement only and makes no causal claim.

if (!exists("theme_ktn")) {
  source(if (file.exists("00_theme.R")) "00_theme.R" else "figures/R/00_theme.R")
}

# slug in the hubness tables -> method name as used in fig1_headline.csv
SLUG <- c(
  mmseqs2 = "sequence (MMseqs2)", foldseek = "Foldseek",
  prostt5_foldseek = "ProstT5-Foldseek",
  esm2_150m = "ESM-2 150M kNN", esm2_650m = "ESM-2 650M kNN",
  esm2_3b = "ESM-2 3B kNN", esmc_300m = "ESM-C 300M kNN",
  esmc_600m = "ESM-C 600M kNN", esmc_6b = "ESM-C 6B kNN",
  ankh_base = "Ankh-base kNN", ankh_large = "Ankh-large kNN",
  prott5 = "ProtT5 kNN", prostt5 = "ProstT5 kNN",
  glm2_650m_embed = "gLM2-embed", prottucker = "ProtTucker/EAT")
ALIGN <- c("MMseqs2", "Foldseek", "ProstT5-Foldseek")
BENCH <- c(test1_pooled = "Test 1 (Swiss-Prot)", test2_scope = "Test 2 (SCOPe)")
GRP <- c("alignment arms", "embedding kNN")
short <- function(x) x |> str_replace(" kNN$", "") |> str_replace("^sequence .*", "MMseqs2")
# Fifteen slots across the page: the two longest names and the six ESM names
# break over two lines so neighbouring labels do not collide.
wrap_name <- function(x) c("ProstT5-Foldseek" = "ProstT5-\nFoldseek",
                           "ProtTucker/EAT" = "ProtTucker/\nEAT")[x] |>
  (\(w) ifelse(is.na(w), str_replace(x, "^(ESM-[2C]) ", "\\1\n"), w))()

hub <- read_fig("figS2_hubness.csv") |>
  filter(transform == "none", benchmark %in% names(BENCH),
         population %in% c("real", "decoy"))

# Fail on an unmapped slug (it would otherwise draw as an unlabelled bar) and
# report any model missing from the input.
unknown <- setdiff(unique(hub$model), names(SLUG))
if (length(unknown)) {
  stop("unmapped model slug(s) in figS2_hubness.csv: ",
       paste(unknown, collapse = ", "), call. = FALSE)
}
absent <- setdiff(names(SLUG), unique(hub$model))
if (length(absent)) {
  message("  NOTE: not in figS2_hubness.csv, so not drawn (rerun 26_hubness.py / ",
          "43_hubness_alignment.py, then 27_figure_data.py): ",
          paste(absent, collapse = ", "))
}

d <- hub |>
  transmute(short = short(SLUG[model]), benchmark = BENCH[benchmark],
            population, coverage = 100 * as.numeric(coverage))

# Foldseek has no Test 2 decoys (no structure for a shuffled sequence). Give the
# slot a zero-height bar labelled as such, so the dodge geometry stays regular
# and the absence reads as deliberate rather than as a broken bar.
d <- d |>
  tidyr::complete(tidyr::nesting(short, benchmark), population) |>
  mutate(missing = is.na(coverage),
         coverage = ifelse(missing, 0, coverage))
if (any(d$missing & !(d$short == "Foldseek" & d$benchmark == BENCH["test2_scope"]))) {
  stop("a population is missing where none was expected: ",
       paste(unique(d$short[d$missing]), collapse = ", "), call. = FALSE)
}

# Alignment arms first, then the kNN methods; within each group by Test 2
# decoy coverage, most concentrated first. Foldseek has no Test 2 decoys and
# leads its group.
ord <- d |>
  filter(benchmark == BENCH["test2_scope"], population == "decoy") |>
  mutate(grp = ifelse(short %in% ALIGN, 1, 2),
         cov = ifelse(missing, -Inf, coverage)) |>
  arrange(grp, cov) |> pull(short)
ord <- c(ord, setdiff(unique(d$short), ord))
n_align <- sum(ord %in% ALIGN)

# Labels are the method names only.
d <- d |> mutate(xlab = wrap_name(short))

# The x variable is keyed on benchmark and model, with levels in `ord` within
# each benchmark block; the display label is supplied through the scale, so
# identical labels in the two panels never produce a duplicated factor level.
# With free x scales each panel drops the keys it does not use, so both rows
# show the same positions with their own labels.
d <- d |>
  mutate(bkey = names(BENCH)[match(benchmark, BENCH)],
         key = paste(bkey, short, sep = "|"))
lv <- as.vector(outer(ord, names(BENCH),
                      function(m, b) paste(b, m, sep = "|")))
LAB <- d |> distinct(key, xlab) |> (\(x) setNames(x$xlab, x$key))()
if (any(duplicated(names(LAB)))) stop("x key maps to two labels", call. = FALSE)
d <- d |>
  mutate(key = factor(key, levels = lv),
         benchmark = factor(benchmark, levels = BENCH),
         population = factor(population, levels = c("real", "decoy"),
                             labels = c("real queries", "decoy queries")),
         # "100.0" is one character too wide for its slot; 100 is exact anyway.
         vlab = ifelse(missing, "no decoys",
                       ifelse(coverage >= 99.95, "100", sprintf("%.1f", coverage))))

empty_slot <- d |> filter(missing) |>
  transmute(benchmark, x = match(short, ord) + 0.76 / 4, label = vlab)

# Group headers and the separator sit at fixed discrete positions: the
# alignment arms occupy 1..n_align, the kNN methods the rest. Both panels have
# the same layout, so one annotation serves both.
n_knn <- length(ord) - n_align
hdr <- tibble(x = c((1 + n_align) / 2, n_align + (1 + n_knn) / 2),
              y = 111, label = GRP)

# Foldseek's Test 1 decoys are the 3,344 D6 naturals only, whose ~148k list
# slots can reach at most half of its 289,206 references. The bar is kept and
# annotated above the pair.
note <- d |>
  filter(short == "Foldseek", benchmark == BENCH["test1_pooled"]) |>
  summarise(key = key[1], y = max(coverage) + 9, .by = benchmark) |>
  mutate(label = "decoys:\nD6 only")

p <- ggplot(d, aes(x = key, y = coverage, fill = population)) +
  geom_vline(xintercept = n_align + 0.5, colour = "grey75", linetype = "22",
             linewidth = 0.35) +
  geom_col(position = position_dodge(width = 0.76), width = 0.70) +
  geom_text(data = d |> filter(!missing), aes(label = vlab),
            position = position_dodge(width = 0.76),
            vjust = -0.45, size = 1.7, colour = "grey30") +
  # the one empty decoy slot: the note runs upward inside it. Dodging a layer
  # that holds one row per x centres it on the pair, so the slot offset is
  # given explicitly.
  geom_text(data = empty_slot, aes(x = x, y = 1.5, label = label),
            inherit.aes = FALSE, angle = 90, hjust = 0,
            size = 1.7, colour = "grey45", fontface = "italic") +
  geom_text(data = note, aes(x = key, y = y, label = label), inherit.aes = FALSE,
            size = 1.6, colour = "grey45", fontface = "italic", lineheight = 0.9) +
  geom_text(data = hdr, aes(x = x, y = y, label = label), inherit.aes = FALSE,
            size = 2.6, fontface = "bold", colour = "grey30") +
  facet_wrap(~ benchmark, nrow = 2, scales = "free_x") +
  scale_x_discrete(labels = function(k) unname(LAB[k])) +
  scale_fill_manual(values = c("real queries" = "#009E73",
                               "decoy queries" = "#CC79A7"), name = NULL) +
  scale_y_continuous(labels = pct, breaks = seq(0, 100, 25),
                     limits = c(0, 116),
                     expand = expansion(mult = c(0, 0))) +
  labs(
    x = "within each group, ordered by Test 2 decoy coverage",
    y = "share of the searched reference set ever retrieved"
  ) +
  theme(axis.text.x = element_text(size = 5.0, lineheight = 0.95,
                                   colour = "grey20"),
        axis.title.x = element_text(size = rel(0.8), colour = "grey35",
                                    margin = margin(t = 5), lineheight = 1.1),
        panel.spacing.y = unit(10, "pt"),
        panel.grid.major.x = element_blank())

save_fig(p, "figS9_hubness_all", width = 6.9, height = 5.6)
