#!/usr/bin/env bash
# Test 2 (SCOPe/ASTRAL-40): shuffle decoys + MMseqs2 search (arm A).
#
# Inputs:  data/scope/scope_{query,reference}.fasta (from 11_scope_build.py)
# Outputs: data/scope/scope_D2_shuffle_seed{1..5}.fasta,
#          results/scope_sequence.tsv
# Run:     bash scripts/12_scope_search.sh [threads]   (MMSEQS=/path/to/mmseqs)
#
# Permissive -e as in Test 1: E-values are recorded, not applied, so every
# threshold can be swept post hoc.
set -euo pipefail
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# mmseqs from $MMSEQS, else PATH.
MMSEQS="${MMSEQS:-$(command -v mmseqs || true)}"
THREADS="${1:-8}"
S="$BASE/data/scope"; W="$BASE/data/interim/scope"; mkdir -p "$W"

# --- shuffle decoys: same composition, no evolutionary order ------------------
python3 - "$S" <<'PY'
import random, sys
from pathlib import Path
S = Path(sys.argv[1])
ids, seqs, cur, cid = [], [], [], None
for line in open(S / "scope_query.fasta"):
    line = line.rstrip("\n")
    if line.startswith(">"):
        if cid: ids.append(cid); seqs.append("".join(cur))
        cid, cur = line[1:].split()[0], []
    else: cur.append(line.strip())
if cid: ids.append(cid); seqs.append("".join(cur))
for seed in range(1, 6):
    rng = random.Random(1000 * seed + 7)
    with (S / f"scope_D2_shuffle_seed{seed}.fasta").open("w") as fh:
        for i, s in zip(ids, seqs):
            c = list(s); rng.shuffle(c)
            fh.write(f">{i}\n{''.join(c)}\n")
print(f"[decoys] 5 x {len(ids)} shuffles")
PY

# --- combined query set: real + decoys, class-tagged as in Test 1 -------------
C="$W/combined.fasta"
awk '/^>/{print ">real|" substr($1,2); next}{print}' "$S/scope_query.fasta" > "$C"
for f in "$S"/scope_D2_shuffle_seed*.fasta; do
  tag="D2_shuffle_$(basename "$f" .fasta | sed 's/.*_//')"
  awk -v t="$tag" '/^>/{print ">" t "|" substr($1,2); next}{print}' "$f" >> "$C"
done
echo "[info] $(grep -c '^>' "$C") queries vs $(grep -c '^>' "$S/scope_reference.fasta") reference"

# --- search -------------------------------------------------------------------
# The decoys (needed by the embedding arms) are written by this point. If
# mmseqs is unavailable the search is skipped with a message rather than
# failing, so the decoys can be generated on their own.
OUT="$BASE/results/scope_sequence.tsv"
if [[ -z "$MMSEQS" ]] || ! command -v "$MMSEQS" >/dev/null 2>&1; then
  echo "[skip] mmseqs not found (set MMSEQS or put mmseqs on PATH)."
  echo "[skip] decoys are written; sequence baseline must be run where mmseqs exists."
  exit 0
fi
if [[ ! -s "$OUT" ]]; then
  "$MMSEQS" easy-search "$C" "$S/scope_reference.fasta" "$W/hits.tsv" "$W/tmp" \
    -s 7.5 -e 1000 --max-seqs 1000 --threads "$THREADS" -v 1 \
    --format-output "query,target,evalue,bits,pident,alnlen,qlen,tlen,qcov,tcov"
  mv "$W/hits.tsv" "$OUT"; rm -rf "$W/tmp"
fi
echo "[done] $OUT ($(wc -l < "$OUT") rows)"
