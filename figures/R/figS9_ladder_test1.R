# Figure S7: transform ladder on Test 1 (Swiss-Prot). Delivered sensitivity at
# 1% FPR for each embedding model under each transform.
#
# Input:  figures/data/figS9_ladder_test1.csv
# Output: figures/out/figS9_ladder_test1.{pdf,png}
# Run:    Rscript figures/R/figS9_ladder_test1.R   (from the repository root)
#
# Values are in-sample (one threshold fit per variant on the full data), so no
# intervals are drawn. ProtTucker is excluded: it is a supervised head rather
# than a raw pLM.
if (!exists("theme_ktn")) {
  source(if (file.exists("00_theme.R")) "00_theme.R" else "figures/R/00_theme.R")
}
source(file.path(PROJ, "figures", "R", "helpers_ladder.R"))

d <- ladder_complete(load_ladder("figS9_ladder_test1.csv"), "delivered")

p <- draw_ladder(d, "delivered", "sensitivity at 1% FPR", 60) +
  labs()

save_fig(p, "figS9_ladder_test1", width = 6.9, height = 4.6)
