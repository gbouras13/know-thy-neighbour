#!/usr/bin/env bash
# Test 1: build the combined structure-search query FASTAs (real queries +
# every decoy class) for ProstT5 3Di prediction (Arm C').
#
# This is the first of a two-stage alternative to 07_run_structure.sh: 3Di
# prediction is GPU-bound and can run on a separate GPU node, while the Foldseek
# search (07b_search_structure.sh) is CPU work that runs where the AFDB database
# lives. These FASTAs are the handoff between the two.
#
# Usage:  scripts/07a_build_query_fastas.sh   (env REFNAME, default "joint")
# Output: data/armC_queries/{small,mid,control}_armC_queries.fasta
#
# Headers are tagged "<class>|<id>" as in Arms A and B, so a single analysis
# path serves all arms.

set -euo pipefail

BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REFNAME="${REFNAME:-joint}"
OUT="$BASE/data/armC_queries"
mkdir -p "$OUT"

for ARM in small mid control; do
  QRY="$BASE/data/queries/${REFNAME}_${ARM}_query.fasta"
  DEST="$OUT/${ARM}_armC_queries.fasta"

  awk '/^>/{print ">real|" substr($1,2); next}{print}' "$QRY" > "$DEST"
  for f in "$BASE"/data/decoys/${ARM}_D*.fasta; do
    # _matched is a strict subset of _all; including both would predict 3Di
    # twice for the same sequences under two different tags.
    case "$f" in *_D6_natural_hard_matched.fasta) continue ;; esac
    tag=$(basename "$f" .fasta | sed "s/^${ARM}_//")
    awk -v t="$tag" '/^>/{print ">" t "|" substr($1,2); next}{print}' "$f" >> "$DEST"
  done

  n=$(grep -c '^>' "$DEST")
  res=$(grep -v '^>' "$DEST" | tr -d '\n' | wc -c)
  printf "  %-8s %7d sequences  %10d residues  %s\n" \
      "$ARM" "$n" "$res" "$(du -h "$DEST" | cut -f1)"
done

echo
echo "Predict 3Di for $OUT on a GPU node (e.g. phold proteins-predict),"
echo "then place {arm}_aa.fasta and {arm}_3di.fasta in data/armC_3di/{arm}/ for 07b_search_structure.sh."
