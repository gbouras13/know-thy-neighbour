#!/usr/bin/env bash
# 3Di prediction for the Test 2 SHUFFLED queries with phold (ProstT5), using
# the same tool, model and flags that produced the real queries' 3Di
# (data/scope_3di/scope_3di.fasta), so any difference between real and shuffle
# 3Di is not an artefact of a different prediction pipeline:
#
#   --threads 16 --force --database <phold_db> --batch_size 10
#   --mask_threshold 0.0   (half precision, GPU)
#
# Note: real and shuffled queries may be predicted on different GPU hardware.
# 3Di output is an argmax over a 20-letter alphabet, so fp16 numeric
# differences matter only if they flip a token's top-1.
#
# Inputs:  data/scope/scope_D2_shuffle_seed{1..5}.fasta (from 12_scope_search.sh)
# Outputs: data/scope_3di_shuffle/shuffle_{aa,3di}.fasta
# Env:     PHOLD (default: phold on PATH), PHOLD_DB (required), THREADS,
#          BATCH_SIZE
#
#   bash scripts/41a_predict_shuffle_3di.sh
set -euo pipefail
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PHOLD="${PHOLD:-phold}"
PHOLD_DB="${PHOLD_DB:?set PHOLD_DB to the phold database directory}"
THREADS="${THREADS:-16}"; BATCH="${BATCH_SIZE:-10}"
OUT="$BASE/data/scope_3di_shuffle"
# The input must live OUTSIDE the output dir: phold -f recreates -o.
IN="$BASE/data/scope/scope_D2_shuffle_all_tagged.fasta"
mkdir -p "$OUT"

if [[ ! -s "$IN" ]]; then
  echo "[in] building tagged shuffle input from the decoy FASTAs"
  : > "$IN"
  for s in 1 2 3 4 5; do
    f="$BASE/data/scope/scope_D2_shuffle_seed${s}.fasta"
    [[ -s "$f" ]] || { echo "  missing $f" >&2; exit 1; }
    awk -v tag="D2_shuffle_seed${s}" \
        '/^>/{print ">" tag "|" substr($1,2); next}{print}' "$f" >> "$IN"
  done
  echo "[in] $(grep -c '^>' "$IN") sequences"
fi

if [[ -s "$OUT/shuffle_3di.fasta" ]]; then
  echo "[skip] shuffle_3di.fasta exists"; exit 0
fi

"$PHOLD" proteins-predict \
    -i "$IN" -o "$OUT" -p shuffle -d "$PHOLD_DB" \
    -t "$THREADS" -f --mask_threshold 0 --batch_size "$BATCH"

echo "[done] $(grep -c '^>' "$OUT/shuffle_3di.fasta") 3Di records"
