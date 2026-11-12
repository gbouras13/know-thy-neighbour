# Figure S8: transform ladder on Test 1 (Swiss-Prot) by top-1 accuracy.
#
# Input:  figures/data/figS9_ladder_test1.csv
# Output: figures/out/figS11_ladder_top1_test1.{pdf,png}
# Run:    Rscript figures/R/figS11_ladder_top1_test1.R   (from the repository root)
#
# A top-1 hit counts as correct only if query and target share a Pfam family
# (see scripts/43_test1_top1.py); Test 1 delivered sensitivity does not check
# correctness. ProtTucker is excluded: it is a supervised head rather than a
# raw pLM.
if (!exists("theme_ktn")) {
  source(if (file.exists("00_theme.R")) "00_theme.R" else "figures/R/00_theme.R")
}
source(file.path(PROJ, "figures", "R", "helpers_ladder.R"))

d <- ladder_complete(load_ladder("figS9_ladder_test1.csv"), "top1")

p <- draw_ladder(d, "top1", "top-1 accuracy", 100) +
  labs()

save_fig(p, "figS11_ladder_top1_test1", width = 6.9, height = 4.6)
