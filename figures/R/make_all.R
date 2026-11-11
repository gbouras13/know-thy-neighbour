# Build every figure whose script is present in figures/R. Scripts listed below
# but absent from the directory are skipped with a message. Run from the
# repository root:
#
#   Rscript figures/R/make_all.R
#
# Inputs are read from figures/data/, written by:
#
#   python3 scripts/26_hubness.py            (slow, 20-30 min; kNN hubness)
#   python3 scripts/43_hubness_alignment.py  (~10 min; alignment-arm hubness)
#   python3 scripts/27_figure_data.py        (all other inputs; also stacks the two above)
#
# Outputs are written to figures/out/ as matching .pdf and .png files. Each
# figure script can also be run on its own.

source(if (file.exists("figures/R/00_theme.R")) "figures/R/00_theme.R"
       else "00_theme.R")

scripts <- c(
  "fig1_headline.R",                 # Fig 1
  "fig2_densities.R",                # Fig 2
  "fig3b_ceiling.R",                 # Fig 3
  "figS6_densities_test1_all.R",     # Fig S1
  "figS7_densities_test1_classes.R", # Fig S2
  "figS5_densities_test2_all.R",     # Fig S3
  "figS8_length_all.R",              # Fig S4
  "figS9_hubness_all.R",             # Fig S5
  "fig6_ladder.R",                   # Fig S6
  "figS9_ladder_test1.R",            # Fig S7
  "figS11_ladder_top1_test1.R",      # Fig S8
  "figS10_ladder_top1_test2.R",      # Fig S9
  "fig10_test3.R",                   # Fig S10
  "fig9_scope_cdf.R"                 # Fig S11
)

root <- file.path(PROJ, "figures", "R")
message("building ", length(scripts), " figures into figures/out/")
for (s in scripts) {
  p <- file.path(root, s)
  if (!file.exists(p)) {
    message("  [skip] ", s, " not present")
    next
  }
  message("- ", s)
  ok <- tryCatch({ source(p, local = new.env()); TRUE },
                 error = function(e) { message("  [FAILED] ", conditionMessage(e)); FALSE })
}
message("done.")
