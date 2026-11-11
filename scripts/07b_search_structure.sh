#!/usr/bin/env bash
# Test 1, structure search stage 2: Foldseek search using externally predicted
# ProstT5 3Di (see 07a_build_query_fastas.sh). CPU-bound; run where the AFDB
# Swiss-Prot Foldseek database (data/structdb) lives.
#
#   Arm C'  ProstT5-predicted 3Di on the query side  -> real queries AND decoys
#   Arm C   AFDB structures on both sides            -> real queries only
#
# Usage:  scripts/07b_search_structure.sh [ARM=small|mid|control] [THREADS=8]
#         (env REFNAME, MAX_SEQS, FOLDSEEK, STRUCTDB)
# Inputs: data/armC_3di/$ARM/{$ARM_aa.fasta,$ARM_3di.fasta},
#         data/structdb/acc2key.tsv and AFDB DB (from 07_run_structure.sh)
# Output: results/structure_Cprime_$ARM.tsv, results/structure_C_$ARM.tsv
#
# Query DB construction follows phold's recipe (create_foldseek_db.py): write
# index-keyed TSVs for AA, 3Di and headers, then `foldseek tsv2db` with dbtypes
# 0, 0 and 12. Replicated directly rather than shelling out to phold so this
# step has no dependency beyond foldseek itself.

set -euo pipefail

BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Prefer foldseek on PATH (e.g. inside a container); fall back to a local
# conda install.
FOLDSEEK="${FOLDSEEK:-$(command -v foldseek || true)}"
FOLDSEEK="${FOLDSEEK:-$HOME/miniforge3/envs/foldseek/bin/foldseek}"
if ! command -v "$FOLDSEEK" >/dev/null 2>&1; then
  echo "foldseek not found (tried PATH, then $FOLDSEEK)" >&2
  exit 1
fi
ARM="${1:-small}"
THREADS="${2:-8}"
REFNAME="${REFNAME:-joint}"

# Matches the sequence arm's --max-seqs so the two are directly comparable.
#
# Output is large: at -s 9.5 --max-seqs 1000 the small arm's C' produces
# ~130M alignment rows (~10 GB); all arms together are roughly 60 GB. Plan for
# scratch space, and reduce (e.g. to top hits per query) before transfer.
MAX_SEQS="${MAX_SEQS:-1000}"

DBDIR="${STRUCTDB:-$BASE/data/structdb}"
WORK="$BASE/data/interim/structure_$ARM"
IN3DI="$BASE/data/armC_3di/$ARM"
mkdir -p "$WORK" "$BASE/results"

REF_FASTA="$BASE/data/reference/${REFNAME}_reference.fasta"
QRY_FASTA="$BASE/data/queries/${REFNAME}_${ARM}_query.fasta"

for f in "$IN3DI/${ARM}_aa.fasta" "$IN3DI/${ARM}_3di.fasta" "$DBDIR/acc2key.tsv"; do
  [[ -s "$f" ]] || { echo "missing $f" >&2; exit 1; }
done

# Foldseek reports whatever ids its databases carry. The reference DB is
# subsetted from AFDB, so target ids come back as "AF-<acc>-F1-model_v6" -- and
# for arm C the query DB is AFDB-derived too, so query ids do as well. The
# analysis keys on plain accessions and on a "<class>|<acc>" query tag, so both
# columns are normalised here.
#
# C' queries already carry the class tag (from header.tsv) and are left alone;
# only their targets need stripping.
normalise_ids () {         # $1 = tsv, $2 = query prefix ("" leaves as-is)
  local f="$1" qpfx="$2"
  awk -F'\t' -v OFS='\t' -v qpfx="$qpfx" '
    function norm(x,  n, a) {
      if (x ~ /^AF-.+-F1-model_v[0-9]+$/) { n = split(x, a, "-"); return a[2] }
      return x
    }
    { $1 = (qpfx == "" ? $1 : qpfx norm($1)); $2 = norm($2); print }
  ' "$f" > "$f.norm" && mv "$f.norm" "$f"
}

# --- 1. drop the legacy D6 class ----------------------------------------------
# Only the D6_natural_hard_all class is used. If the 3Di input also contains a
# legacy untagged "D6_natural_hard|" class (an older single D6 file built
# before the _all/_matched sets), it is removed here before the search. Also
# checks that the AA and 3Di FASTAs agree in order and length.
python3 - "$IN3DI" "$ARM" "$WORK" <<'PY'
import sys
from pathlib import Path
in3di, arm, work = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])

def read_fa(p):
    ids, seqs, cur, cid = [], [], [], None
    for line in open(p):
        line = line.rstrip("\n")
        if line.startswith(">"):
            if cid is not None:
                ids.append(cid); seqs.append("".join(cur))
            cid, cur = line[1:].split()[0], []
        else:
            cur.append(line.strip())
    if cid is not None:
        ids.append(cid); seqs.append("".join(cur))
    return ids, seqs

aid, aa = read_fa(in3di / f"{arm}_aa.fasta")
tid, td = read_fa(in3di / f"{arm}_3di.fasta")
assert aid == tid, "aa and 3di id order differ"

STALE = "D6_natural_hard|"        # exact-prefix: _all must survive
keep = [i for i, x in enumerate(aid) if not x.startswith(STALE)]
dropped = len(aid) - len(keep)

bad = [i for i in keep if len(aa[i]) != len(td[i])]
if bad:
    raise SystemExit(f"{len(bad)} sequences have aa/3di length mismatch")

with open(work / "aa.tsv", "w") as fa, \
     open(work / "3di.tsv", "w") as f3, \
     open(work / "header.tsv", "w") as fh:
    for n, i in enumerate(keep, start=1):
        fa.write(f"{n}\t{aa[i]}\n")
        f3.write(f"{n}\t{td[i]}\n")
        fh.write(f"{n}\t{aid[i]}\n")

print(f"[filter] {arm}: kept {len(keep)}, dropped {dropped} legacy D6_natural_hard")
PY

# --- 2. query DB from AA + 3Di ------------------------------------------------
if [[ ! -s "$WORK/queryDB.dbtype" ]]; then
  echo "[C'] building query DB from predicted 3Di"
  "$FOLDSEEK" tsv2db "$WORK/aa.tsv"     "$WORK/queryDB"    --output-dbtype 0
  "$FOLDSEEK" tsv2db "$WORK/3di.tsv"    "$WORK/queryDB_ss" --output-dbtype 0
  "$FOLDSEEK" tsv2db "$WORK/header.tsv" "$WORK/queryDB_h"  --output-dbtype 12
fi
rm -f "$WORK/aa.tsv" "$WORK/3di.tsv" "$WORK/header.tsv"

# --- 3. reference: AFDB structures for our reference accessions ---------------
if [[ ! -s "$WORK/refDB.dbtype" ]]; then
  awk '/^>/{print substr($1,2)}' "$REF_FASTA" > "$WORK/ref_accs.txt"
  awk 'NR==FNR{want[$1];next} ($1 in want){print $2}' \
      "$WORK/ref_accs.txt" "$DBDIR/acc2key.tsv" > "$WORK/ref_keys.txt"
  awk 'NR==FNR{want[$1];next} ($1 in want){print $1}' \
      "$WORK/ref_accs.txt" "$DBDIR/acc2key.tsv" > "$WORK/ref_covered.txt"
  n_ref=$(wc -l < "$WORK/ref_accs.txt"); n_cov=$(wc -l < "$WORK/ref_covered.txt")
  echo "[cov] reference: $n_cov / $n_ref have an AFDB structure ($((100*n_cov/n_ref))%)"
  cp "$WORK/ref_covered.txt" "$BASE/results/afdb_covered_reference.txt"

  echo "[db] subsetting AFDB to $n_cov structures"
  "$FOLDSEEK" createsubdb "$WORK/ref_keys.txt" "$DBDIR/afdb_sp"    "$WORK/refDB"    --subdb-mode 1
  "$FOLDSEEK" createsubdb "$WORK/ref_keys.txt" "$DBDIR/afdb_sp_ss" "$WORK/refDB_ss" --subdb-mode 1
  "$FOLDSEEK" createsubdb "$WORK/ref_keys.txt" "$DBDIR/afdb_sp_h"  "$WORK/refDB_h"  --subdb-mode 1
fi

# --- 4. C': predicted 3Di vs real structures ---------------------------------
OUT_CP="$BASE/results/structure_Cprime_${ARM}.tsv"
if [[ ! -s "$OUT_CP" ]]; then
  echo "[C'] search ($(wc -l < "$WORK/queryDB.index") queries)"
  "$FOLDSEEK" search "$WORK/queryDB" "$WORK/refDB" "$WORK/resCp" "$WORK/tmp_cp" \
      -e 1000 -s 9.5 --max-seqs "$MAX_SEQS" --threads "$THREADS"
  "$FOLDSEEK" convertalis "$WORK/queryDB" "$WORK/refDB" "$WORK/resCp" "$WORK/Cprime.tsv" \
      --format-output "query,target,evalue,bits,fident,alnlen,qlen,tlen,qcov,tcov" \
      --threads "$THREADS"
  normalise_ids "$WORK/Cprime.tsv" ""  # queries already tagged; targets are AFDB
  mv "$WORK/Cprime.tsv" "$OUT_CP"
  rm -rf "$WORK/tmp_cp" "$WORK/resCp"*
fi

# --- 5. C: real structures both sides, real queries only ----------------------
OUT_C="$BASE/results/structure_C_${ARM}.tsv"
if [[ ! -s "$OUT_C" ]]; then
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

  echo "[C]  search (real structures, true-positive side only)"
  "$FOLDSEEK" search "$WORK/queryDB_real" "$WORK/refDB" "$WORK/resC" "$WORK/tmp_c" \
      -e 1000 -s 9.5 --max-seqs "$MAX_SEQS" --threads "$THREADS"
  "$FOLDSEEK" convertalis "$WORK/queryDB_real" "$WORK/refDB" "$WORK/resC" "$WORK/C.tsv" \
      --format-output "query,target,evalue,bits,fident,alnlen,qlen,tlen,qcov,tcov" \
      --threads "$THREADS"
  normalise_ids "$WORK/C.tsv" "real|"  # both sides are AFDB-derived here
  mv "$WORK/C.tsv" "$OUT_C"
  rm -rf "$WORK/tmp_c" "$WORK/resC"*
fi

echo "[done] $OUT_C"
echo "[done] $OUT_CP"
