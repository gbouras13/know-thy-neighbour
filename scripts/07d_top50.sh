#!/usr/bin/env bash
# Test 1, structure arms: keep the 50 best hits per query.
#
# Every downstream analysis uses best-hit-per-query or top-50 lists, so the
# full Foldseek tables (tens of GB) are reduced to the top 50 hits per query.
# Foldseek writes hits grouped by query in score order, so a running counter
# keeps the best 50 without sorting.
#
# Input:  results/structure_C_<arm>.tsv, results/structure_Cprime_<arm>.tsv (07b)
# Output: results/structure_C_<arm>_top50.tsv.gz, results/structure_Cprime_<arm>_top50.tsv.gz
# Run:    ./scripts/07d_top50.sh            (from the repository root)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
for f in results/structure_C_*.tsv results/structure_Cprime_*.tsv; do
  [[ -s "$f" ]] || continue
  out="${f%.tsv}_top50.tsv.gz"
  [[ -s "$out" ]] && { echo "skip $(basename "$out") (exists)"; continue; }
  awk -F'\t' '$1!=prev{prev=$1; n=0} ++n<=50' "$f" | gzip > "$out"
  echo "wrote $(basename "$out")"
done
