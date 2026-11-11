#!/usr/bin/env bash
# ProtTucker / EAT: embedding-based annotation transfer using the authors' own
# pipeline (github.com/Rostlab/EAT).
#
# ProtTucker is a contrastive projection head over ProtT5 embeddings, trained
# on the CATH hierarchy; annotations are transferred to the nearest neighbour
# by Euclidean distance. It is the retrieval-trained form of embedding-based
# annotation transfer.
#
# Leakage caveat: ProtTucker is trained on CATH, and CATH and SCOPe classify
# largely the same PDB structures, so Test 2 (SCOPe/ASTRAL-40) results for
# ProtTucker risk train/test contamination. Test 1 (Swiss-Prot/Pfam) is the
# primary evaluation for this method.
#
# Runs in EAT's own pinned environment (see its requirements.txt) on a GPU node.
# 21_prottucker_direct.py applies the same head to existing ProtT5 embeddings
# without re-embedding.
#
# Usage:  scripts/20_prottucker.sh [scope|small|mid|control]
# Output: results/ table written by 20b_parse_eat.py
set -euo pipefail

BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EAT_DIR="${EAT_DIR:-$BASE/tools/EAT}"
BENCH="${1:-scope}"          # scope | small | mid | control
NUM_NN="${NUM_NN:-50}"
OUT="$BASE/data/interim/prottucker_$BENCH"
mkdir -p "$OUT" "$BASE/results" "$(dirname "$EAT_DIR")"

if [[ ! -d "$EAT_DIR" ]]; then
  echo "[get] cloning EAT"
  git clone --depth 1 https://github.com/Rostlab/EAT "$EAT_DIR"
fi

# --- inputs -------------------------------------------------------------------
if [[ "$BENCH" == "scope" ]]; then
  LOOKUP="$BASE/data/scope/scope_reference.fasta"
  QUERY="$BASE/data/scope/scope_query.fasta"
  QTAG="real"
else
  LOOKUP="$BASE/data/reference/joint_reference.fasta"
  QUERY="$OUT/query_all.fasta"
  # Test 1 searches the decoys as well, so the method has negatives for
  # calibration. Each class is tagged in the header so the analysis can tell
  # them apart.
  if [[ ! -s "$QUERY" ]]; then
    awk '/^>/{print ">real|" substr($1,2); next} {print}' \
        "$BASE/data/queries/joint_${BENCH}_query.fasta" > "$QUERY"
    for f in "$BASE/data/decoys/${BENCH}_"*.fasta; do
      [[ -s "$f" ]] || continue
      cls=$(basename "$f" .fasta); cls=${cls#${BENCH}_}
      awk -v c="$cls" '/^>/{print ">" c "|" substr($1,2); next} {print}' "$f" >> "$QUERY"
    done
  fi
  QTAG=""
fi
echo "[in] lookup=$(grep -c '^>' "$LOOKUP")  query=$(grep -c '^>' "$QUERY")"

# --- run EAT ------------------------------------------------------------------
# --use_tucker 1 applies the ProtTucker head (without it this is plain ProtT5
# kNN). No --threshold, so every query returns its top NUM_NN and thresholding
# is done downstream at matched FPR, as for every other method.
# eat.py looks for the head at <output_dir>/tucker_weights.pt and downloads it
# if absent. For compute nodes without internet, seed it from a copy fetched
# beforehand (TUCKER_WEIGHTS, default tools/tucker_weights.pt).
WEIGHTS="${TUCKER_WEIGHTS:-$BASE/tools/tucker_weights.pt}"
if [[ -s "$WEIGHTS" && ! -s "$OUT/tucker_weights.pt" ]]; then
  cp "$WEIGHTS" "$OUT/tucker_weights.pt"
  echo "[w] seeded tucker_weights.pt"
elif [[ ! -s "$WEIGHTS" && ! -s "$OUT/tucker_weights.pt" ]]; then
  echo "[w] WARNING: no local tucker weights; eat.py will try to download them," >&2
  echo "    which fails without internet. Fetch tools/tucker_weights.pt first." >&2
fi

if [[ ! -s "$OUT/eat_result.txt" ]]; then
  python3 "$EAT_DIR/eat.py" \
      --lookup "$LOOKUP" --queries "$QUERY" --output "$OUT" \
      --use_tucker 1 --num_NN "$NUM_NN"
  # EAT's output filename has varied between releases; normalise it.
  if [[ ! -s "$OUT/eat_result.txt" ]]; then
    cand=$(ls -S "$OUT"/*.txt "$OUT"/*.csv 2>/dev/null | head -1 || true)
    [[ -n "$cand" ]] && cp "$cand" "$OUT/eat_result.txt"
  fi
fi
[[ -s "$OUT/eat_result.txt" ]] || { echo "EAT produced no output" >&2; exit 1; }

# --- convert to our table format ---------------------------------------------
python3 "$BASE/scripts/20b_parse_eat.py" \
    --eat "$OUT/eat_result.txt" --bench "$BENCH" --qtag "$QTAG"
