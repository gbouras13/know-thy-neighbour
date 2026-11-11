# Figure S3: Test 2 (SCOPe) best-hit score densities for every method:
# top-1 correct, top-1 wrong fold, and shuffled queries.
#
# Input:  figures/data/fig12_densities_test2.csv (via helpers_test2_density.R)
# Output: figures/out/figS5_densities_test2_all.{pdf,png}
# Run:    Rscript figures/R/figS5_densities_test2_all.R   (from the repository root)
#
# Foldseek and ProstT5-Foldseek show no shuffle class: shuffled sequences have
# no structures.

if (!exists("theme_nnm")) {
  source(if (file.exists("00_theme.R")) "00_theme.R" else "figures/R/00_theme.R")
}
source(file.path(PROJ, "figures", "R", "helpers_test2_density.R"))

ORDER <- c("sequence (MMseqs2)", "Foldseek", "ProstT5-Foldseek",
           "ProtTucker/EAT", "gLM2-embed", "ProstT5 kNN", "ProtT5 kNN",
           "Ankh-base kNN", "Ankh-large kNN", "ESM-2 150M kNN",
           "ESM-2 650M kNN", "ESM-2 3B kNN", "ESM-C 300M kNN",
           "ESM-C 600M kNN", "ESM-C 6B kNN")

dS <- load_t2dens(keep = ORDER)

pS <- draw_t2dens(dS, nrow = 5) +
  theme(strip.text = element_text(size = 6.4),
        axis.text = element_text(size = 5.4)) +
  labs()

save_fig(pS, "figS5_densities_test2_all", width = 6.9, height = 8.6)
