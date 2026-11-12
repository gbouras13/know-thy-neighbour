#!/usr/bin/env bash
# Test 1, Arm A: MMseqs2 sequence search for real queries and every decoy class
# (D0-D5 synthetic, D6 natural hard negatives) against the joint reference.
#
# Usage:  scripts/04_run_sequence.sh [ARM=small|mid|control] [THREADS=8]
#         (env REFNAME, default "joint"; MMSEQS, default mmseqs on PATH)
# Inputs: data/queries/${REFNAME}_${ARM}_query.fasta,
#         data/reference/${REFNAME}_reference.fasta, data/decoys/${ARM}_D*.fasta
# Output: results/sequence_${ARM}_vs_${REFNAME}.tsv (MMseqs2 tabular; query ids
#         are prefixed "real|" or "<decoy tag>|")
#
# Design choices:
#
#  1. One combined query set. Real queries and all decoy files are searched in
#     a single pass against the same reference, so database size, parameters
#     and E-value denominator are identical across them.
#
#  2. A permissive reporting threshold (-e 1000). E-values are recorded, not
#     applied; thresholds are swept in analysis.

set -euo pipefail

BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MMSEQS="${MMSEQS:-$(command -v mmseqs || true)}"
if [[ -z "$MMSEQS" ]] || ! command -v "$MMSEQS" >/dev/null 2>&1; then
  echo "mmseqs not found: put it on PATH or set MMSEQS=/path/to/mmseqs" >&2
  exit 1
fi
ARM="${1:-small}"
THREADS="${2:-8}"
# All length arms search the same reference, so database size (the E-value
# denominator) is constant across arms.
REFNAME="${REFNAME:-joint}"

WORK="$BASE/data/interim/seqsearch_${ARM}_vs_${REFNAME}"
OUT="$BASE/results/sequence_${ARM}_vs_${REFNAME}.tsv"
mkdir -p "$WORK"

REF="$BASE/data/reference/${REFNAME}_reference.fasta"
QRY="$BASE/data/queries/${REFNAME}_${ARM}_query.fasta"

# The "skip if it already exists" branches below reuse cached results, so the
# inputs are fingerprinted and the working directory is cleared when they
# change. Sequence and byte counts suffice: these files are only ever
# rewritten wholesale by an upstream stage, never edited in place.
fingerprint="ref:$(grep -c '^>' "$REF") qry:$(grep -c '^>' "$QRY")"
fingerprint="$fingerprint decoys:$(cat "$BASE"/data/decoys/${ARM}_D*.fasta 2>/dev/null | wc -c)"
stamp="$WORK/.inputs"
if [[ -f "$stamp" && "$(cat "$stamp")" != "$fingerprint" ]]; then
  echo "[stale] inputs changed since last run -- clearing $WORK"
  rm -rf "$WORK"
  mkdir -p "$WORK"
fi
printf '%s' "$fingerprint" > "$stamp"

# ---- combined query fasta: real + all decoys, source encoded in the id ----
COMBINED="$WORK/combined_queries.fasta"
if [[ ! -s "$COMBINED" ]]; then
  echo "[build] combined query set"
  awk '/^>/{print ">real|" substr($1,2); next}{print}' "$QRY" > "$COMBINED"
  for f in "$BASE"/data/decoys/${ARM}_D*.fasta; do
    # D6_natural_hard_matched is a strict subset of D6_natural_hard_all, so
    # searching both would run the same sequences twice under two tags. Search
    # _all; recover the matched-set result by filtering in analysis.
    case "$f" in *_D6_natural_hard_matched.fasta) continue ;; esac
    tag=$(basename "$f" .fasta | sed "s/^${ARM}_//")
    awk -v t="$tag" '/^>/{print ">" t "|" substr($1,2); next}{print}' "$f" >> "$COMBINED"
  done
fi
echo "[info]  $(grep -c '^>' "$COMBINED") combined queries vs $(grep -c '^>' "$REF") reference"

# ---- databases ----
[[ -s "$WORK/refDB.dbtype" ]]   || "$MMSEQS" createdb "$REF" "$WORK/refDB" -v 1
[[ -s "$WORK/queryDB.dbtype" ]] || "$MMSEQS" createdb "$COMBINED" "$WORK/queryDB" -v 1

# ---- search ----
if [[ ! -s "$WORK/resultDB.dbtype" ]]; then
  echo "[search] mmseqs2 -s 7.5 -e 1000 (permissive: E-values recorded, not applied)"
  "$MMSEQS" search "$WORK/queryDB" "$WORK/refDB" "$WORK/resultDB" "$WORK/tmp" \
    -s 7.5 -e 1000 --max-seqs 1000 --threads "$THREADS" -v 1
fi

"$MMSEQS" convertalis "$WORK/queryDB" "$WORK/refDB" "$WORK/resultDB" "$OUT" \
  --format-output "query,target,evalue,bits,pident,alnlen,qlen,tlen,qcov,tcov" \
  --threads "$THREADS" -v 1

rm -rf "$WORK/tmp"
echo "[done]  $OUT  ($(wc -l < "$OUT") alignment rows)"
