#!/usr/bin/env bash
# Test 1, structure-based search with Foldseek in two forms:
#
#   Arm C   AFDB structures on both sides                -> real queries only
#   Arm C'  ProstT5-predicted 3Di on the query side      -> real queries AND decoys
#           ("ProstT5-Foldseek"); reference uses AFDB structures
#
# Decoys have no real structure, so only C' can be evaluated against them. C'
# is also the deployment-relevant case for a novel protein without a structure.
# C bounds how much of C''s behaviour is due to 3Di prediction error, on the
# real queries where both are defined.
#
# Usage:  scripts/07_run_structure.sh [ARM=small|mid|control] [THREADS=8]
#         (env REFNAME default "joint"; FOLDSEEK, STRUCTDB override paths)
# Output: results/structure_C_${ARM}.tsv, results/structure_Cprime_${ARM}.tsv,
#         results/afdb_covered_reference_${ARM}.txt
#
# -e is permissive: E-values are recorded, not applied here.

set -euo pipefail

BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FOLDSEEK="${FOLDSEEK:-$HOME/miniforge3/envs/foldseek/bin/foldseek}"
ARM="${1:-small}"
THREADS="${2:-8}"
REFNAME="${REFNAME:-joint}"

WORK="$BASE/data/interim/structure_$ARM"
DBDIR="${STRUCTDB:-$BASE/data/structdb}"
mkdir -p "$WORK" "$DBDIR" "$BASE/results"

REF_FASTA="$BASE/data/reference/${REFNAME}_reference.fasta"
QRY_FASTA="$BASE/data/queries/${REFNAME}_${ARM}_query.fasta"

# Cached Foldseek databases must be invalidated when their inputs change, or a
# rebuilt split silently reuses structures belonging to a different query set.
fingerprint="ref:$(grep -c '^>' "$REF_FASTA") qry:$(grep -c '^>' "$QRY_FASTA")"
fingerprint="$fingerprint decoys:$(cat "$BASE"/data/decoys/${ARM}_D*.fasta 2>/dev/null | wc -c)"
stamp="$WORK/.inputs"
if [[ -f "$stamp" && "$(cat "$stamp")" != "$fingerprint" ]]; then
  echo "[stale] inputs changed since last run -- clearing $WORK"
  rm -rf "$WORK"; mkdir -p "$WORK"
fi
printf '%s' "$fingerprint" > "$stamp"

# --- 1. AFDB Swiss-Prot ------------------------------------------------------
if [[ ! -s "$DBDIR/afdb_sp.dbtype" ]]; then
  echo "[db] downloading AFDB Swiss-Prot into $DBDIR"
  "$FOLDSEEK" databases Alphafold/Swiss-Prot "$DBDIR/afdb_sp" "$DBDIR/tmp" --threads "$THREADS"
  rm -rf "$DBDIR/tmp"
fi

# --- 2. accession -> internal key, EXACTLY ------------------------------------
# Two naming pitfalls:
#
# 1. Substring matching (e.g. `grep -F -f`) can match a short accession inside
#    an unrelated identifier, so accessions are matched exactly.
#
# 2. AFDB Swiss-Prot contains isoforms as well as canonical entries:
#       AF-A0A061FKM4-F1-model_v6     canonical
#       AF-A0A061FKM4-2-F1-model_v6   isoform 2
#    Splitting on '-' maps both to the same accession, which would attach an
#    isoform's structure (a different sequence) to the canonical accession.
#    Only the canonical 4-field form is accepted.
if [[ ! -s "$DBDIR/acc2key.tsv" ]]; then
  echo "[db] indexing $(wc -l < "$DBDIR/afdb_sp.lookup") AFDB entries by accession"
  awk -F'\t' '{n=split($2,a,"-"); if (n==4 && a[1]=="AF" && a[3]=="F1") print a[2] "\t" $1}' \
      "$DBDIR/afdb_sp.lookup" > "$DBDIR/acc2key.tsv"
  echo "[db] canonical entries indexed: $(wc -l < "$DBDIR/acc2key.tsv")"
fi

# --- 3. coverage: which of our reference accessions have a structure? ---------
# Arms A and B search the full reference. If AFDB covers fewer sequences, Arm C
# searches a smaller database (different E-value denominator, fewer findable
# homologs). The covered subset is written out so every arm can be restricted
# to the same intersection at analysis time.
awk '/^>/{print substr($1,2)}' "$REF_FASTA" > "$WORK/ref_accs.txt"
awk 'NR==FNR{want[$1];next} ($1 in want){print $2}' \
    "$WORK/ref_accs.txt" "$DBDIR/acc2key.tsv" > "$WORK/ref_keys.txt"
awk 'NR==FNR{want[$1];next} ($1 in want){print $1}' \
    "$WORK/ref_accs.txt" "$DBDIR/acc2key.tsv" > "$WORK/ref_covered.txt"

n_ref=$(wc -l < "$WORK/ref_accs.txt")
n_cov=$(wc -l < "$WORK/ref_covered.txt")
echo "[cov] reference: $n_cov / $n_ref have an AFDB structure ($((100*n_cov/n_ref))%)"
cp "$WORK/ref_covered.txt" "$BASE/results/afdb_covered_reference_${ARM}.txt"

# --- 4. subset the structure DB ----------------------------------------------
# Subset the aux databases with the SAME key list and --subdb-mode 1 (preserve
# original keys) so main/_ss/_h stay aligned. _ca holds coordinates and is only
# needed for TM-align mode, which is not used; skipping it saves most of the
# disk.
if [[ ! -s "$WORK/refDB.dbtype" ]]; then
  echo "[db] subsetting AFDB to $n_cov reference structures"
  "$FOLDSEEK" createsubdb "$WORK/ref_keys.txt" "$DBDIR/afdb_sp"    "$WORK/refDB"    --subdb-mode 1
  "$FOLDSEEK" createsubdb "$WORK/ref_keys.txt" "$DBDIR/afdb_sp_ss" "$WORK/refDB_ss" --subdb-mode 1
  "$FOLDSEEK" createsubdb "$WORK/ref_keys.txt" "$DBDIR/afdb_sp_h"  "$WORK/refDB_h"  --subdb-mode 1
  echo "[db] reference DB entries: $(wc -l < "$WORK/refDB.index")"
fi

# --- 5. ProstT5 weights ------------------------------------------------------
if [[ ! -s "$DBDIR/prostt5/model.safetensors" && ! -d "$DBDIR/prostt5" ]]; then
  echo "[db] fetching ProstT5 weights for Foldseek"
  "$FOLDSEEK" databases ProstT5 "$DBDIR/prostt5" "$DBDIR/tmp_p"
  rm -rf "$DBDIR/tmp_p"
fi

# --- 6. combined query fasta: real + every decoy ------------------------------
COMBINED="$WORK/combined_queries.fasta"
if [[ ! -s "$COMBINED" ]]; then
  awk '/^>/{print ">real|" substr($1,2); next}{print}' "$QRY_FASTA" > "$COMBINED"
  for f in "$BASE"/data/decoys/${ARM}_D*.fasta; do
    # _matched is a strict subset of _all; searching both runs the same
    # sequences twice under two tags.
    case "$f" in *_D6_natural_hard_matched.fasta) continue ;; esac
    tag=$(basename "$f" .fasta | sed "s/^${ARM}_//")
    awk -v t="$tag" '/^>/{print ">" t "|" substr($1,2); next}{print}' "$f" >> "$COMBINED"
  done
fi
echo "[info] $(grep -c '^>' "$COMBINED") combined queries (real + decoys)"

# --- 7. C': ProstT5-predicted 3Di, real queries AND decoys -------------------
if [[ ! -s "$WORK/queryDB_prostt5.dbtype" ]]; then
  echo "[C'] predicting 3Di with ProstT5 (GPU strongly preferred)"
  "$FOLDSEEK" createdb "$COMBINED" "$WORK/queryDB_prostt5" \
      --prostt5-model "$DBDIR/prostt5" --threads "$THREADS"
fi

if [[ ! -s "$BASE/results/structure_Cprime_${ARM}.tsv" ]]; then
  echo "[C'] search"
  "$FOLDSEEK" search "$WORK/queryDB_prostt5" "$WORK/refDB" "$WORK/resC_prime" "$WORK/tmp_cp" \
      -e 1000 -s 9.5 --max-seqs 1000 --threads "$THREADS"
  "$FOLDSEEK" convertalis "$WORK/queryDB_prostt5" "$WORK/refDB" "$WORK/resC_prime" \
      "$BASE/results/structure_Cprime_${ARM}.tsv" \
      --format-output "query,target,evalue,bits,fident,alnlen,qlen,tlen,qcov,tcov" \
      --threads "$THREADS"
  rm -rf "$WORK/tmp_cp"
fi

# --- 8. C: real AFDB structures, real queries only ---------------------------
awk '/^>/{print substr($1,2)}' "$QRY_FASTA" > "$WORK/query_accs.txt"
awk 'NR==FNR{want[$1];next} ($1 in want){print $2}' \
    "$WORK/query_accs.txt" "$DBDIR/acc2key.tsv" > "$WORK/query_keys.txt"
n_q=$(wc -l < "$WORK/query_accs.txt"); n_qc=$(wc -l < "$WORK/query_keys.txt")
echo "[cov] queries: $n_qc / $n_q have an AFDB structure ($((100*n_qc/n_q))%)"

if [[ ! -s "$WORK/queryDB_real.dbtype" ]]; then
  "$FOLDSEEK" createsubdb "$WORK/query_keys.txt" "$DBDIR/afdb_sp"    "$WORK/queryDB_real"    --subdb-mode 1
  "$FOLDSEEK" createsubdb "$WORK/query_keys.txt" "$DBDIR/afdb_sp_ss" "$WORK/queryDB_real_ss" --subdb-mode 1
  "$FOLDSEEK" createsubdb "$WORK/query_keys.txt" "$DBDIR/afdb_sp_h"  "$WORK/queryDB_real_h"  --subdb-mode 1
fi

if [[ ! -s "$BASE/results/structure_C_${ARM}.tsv" ]]; then
  echo "[C]  search (real structures, true-positive side only)"
  "$FOLDSEEK" search "$WORK/queryDB_real" "$WORK/refDB" "$WORK/resC" "$WORK/tmp_c" \
      -e 1000 -s 9.5 --max-seqs 1000 --threads "$THREADS"
  "$FOLDSEEK" convertalis "$WORK/queryDB_real" "$WORK/refDB" "$WORK/resC" \
      "$BASE/results/structure_C_${ARM}.tsv" \
      --format-output "query,target,evalue,bits,fident,alnlen,qlen,tlen,qcov,tcov" \
      --threads "$THREADS"
  rm -rf "$WORK/tmp_c"
fi

echo "[done] results/structure_C_${ARM}.tsv and results/structure_Cprime_${ARM}.tsv"
