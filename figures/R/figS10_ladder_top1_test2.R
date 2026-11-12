# Figure S9: transform ladder on Test 2 (SCOPe) by top-1 accuracy.
#
# Input:  figures/data/fig6_ladder.csv
# Output: figures/out/figS10_ladder_top1_test2.{pdf,png}
# Run:    Rscript figures/R/figS10_ladder_top1_test2.R   (from the repository root)
#
# A top-1 hit counts as correct only if it is in the query's superfamily; the
# query's own family is held out. ProtTucker is excluded: it is a supervised
# head rather than a raw pLM.
if (!exists("theme_ktn")) {
  source(if (file.exists("00_theme.R")) "00_theme.R" else "figures/R/00_theme.R")
}
source(file.path(PROJ, "figures", "R", "helpers_ladder.R"))

d <- ladder_complete(load_ladder("fig6_ladder.csv"), "top1")

p <- draw_ladder(d, "top1", "top-1 accuracy", 100) +
  labs()

save_fig(p, "figS10_ladder_top1_test2", width = 6.9, height = 4.6)
