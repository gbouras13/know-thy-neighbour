#!/usr/bin/env bash
# Test 2 (SCOPe/ASTRAL-40): Foldseek searches, two variants as in Test 1.
#
#   C'  ProstT5-predicted 3Di on BOTH sides ("ProstT5-Foldseek"): sequence in,
#       no structure assumed, directly comparable to the pLM kNN methods. In
#       Test 1 the reference side came from AFDB; ASTRAL domains are PDB
#       fragments with no clean UniProt mapping, so here the reference is
#       predicted too.
#
#   C   real experimental ASTRAL coordinates on both sides. SCOPe labels were
#       assigned from these structures, so this is an upper bound rather than
#       a competitor to the sequence-only methods.
#
# The Test 2 negative class ("top hit in a different fold") is defined for real
# queries, so no synthetic decoys are needed here.
#
# Inputs:  data/scope/scope_{query,reference}.fasta; data/scope_3di/
#          scope_{aa,3di}.fasta (ProstT5 3Di from phold) for C'; ASTRAL-40
#          pdbstyle coordinates for C (downloaded if absent).
# Outputs: results/scope_structure_Cprime.tsv, results/scope_structure_C.tsv
#          (top 50 hits per query).
# Run:     bash scripts/16b_scope_build_search.sh [threads]
#          (env: FOLDSEEK, IN3DI, PDBDIR, SCOPE_WORK, MAX_SEQS, SKIP_ARM_C_REAL=1)
set -euo pipefail

BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FOLDSEEK="${FOLDSEEK:-$(command -v foldseek || true)}"
FOLDSEEK="${FOLDSEEK:-$HOME/miniforge3/envs/foldseek/bin/foldseek}"
command -v "$FOLDSEEK" >/dev/null 2>&1 || { echo "foldseek not found" >&2; exit 1; }
THREADS="${1:-8}"

S="$BASE/data/scope"; W="${SCOPE_WORK:-$BASE/data/interim/scope_struct}"
mkdir -p "$W" "$BASE/results"
MAX_SEQS="${MAX_SEQS:-1000}"

convert () {   # $1=qdb $2=tdb $3=resdb $4=out
  "$FOLDSEEK" convertalis "$1" "$2" "$3" "$4" \
      --format-output "query,target,evalue,bits,fident,alnlen,qlen,tlen,qcov,tcov" \
      --threads "$THREADS"
}

# ---------------------------------------------------------------- C' ----------
# 3Di is predicted by phold (proteins-predict) over a combined FASTA in which
# queries are headed "real|<id>" and reference entries bare "<id>".
IN3DI="${IN3DI:-$BASE/data/scope_3di}"
OUT_CP="$BASE/results/scope_structure_Cprime.tsv"
if [[ -s "$IN3DI/scope_3di.fasta" && ! -s "$OUT_CP" ]]; then
  echo "[C'] splitting predicted 3Di into query / reference"
  python3 - "$IN3DI" "$W" <<'PY'
import sys
from pathlib import Path
in3di, work = Path(sys.argv[1]), Path(sys.argv[2])

def read_fa(p):
    ids, seqs, cur, cid = [], [], [], None
    for line in open(p):
        line = line.rstrip("\n")
        if line.startswith(">"):
            if cid is not None: ids.append(cid); seqs.append("".join(cur))
            cid, cur = line[1:].split()[0], []
        else: cur.append(line.strip())
    if cid is not None: ids.append(cid); seqs.append("".join(cur))
    return ids, seqs

aid, aa = read_fa(in3di / "scope_aa.fasta")
tid, td = read_fa(in3di / "scope_3di.fasta")
assert aid == tid, "aa and 3di id order differ"
bad = [i for i in range(len(aid)) if len(aa[i]) != len(td[i])]
if bad: raise SystemExit(f"{len(bad)} aa/3di length mismatches")

for side, keep in (("query",  lambda x: x.startswith("real|")),
                   ("reference", lambda x: not x.startswith("real|"))):
    idx = [i for i, x in enumerate(aid) if keep(x)]
    with open(work / f"{side}_aa.tsv", "w") as fa, \
         open(work / f"{side}_3di.tsv", "w") as f3, \
         open(work / f"{side}_h.tsv", "w") as fh:
        for n, i in enumerate(idx, start=1):
            fa.write(f"{n}\t{aa[i]}\n"); f3.write(f"{n}\t{td[i]}\n")
            fh.write(f"{n}\t{aid[i]}\n")
    print(f"[C'] {side}: {len(idx)}")
PY
  for side in query reference; do
    db="$W/cp_${side}DB"
    if [[ ! -s "$db.dbtype" ]]; then
      "$FOLDSEEK" tsv2db "$W/${side}_aa.tsv"  "$db"     --output-dbtype 0
      "$FOLDSEEK" tsv2db "$W/${side}_3di.tsv" "${db}_ss" --output-dbtype 0
      "$FOLDSEEK" tsv2db "$W/${side}_h.tsv"   "${db}_h"  --output-dbtype 12
    fi
  done
  echo "[C'] search"
  "$FOLDSEEK" search "$W/cp_queryDB" "$W/cp_referenceDB" "$W/resCp" "$W/tmp_cp" \
      -e 1000 -s 9.5 --max-seqs "$MAX_SEQS" --threads "$THREADS"
  convert "$W/cp_queryDB" "$W/cp_referenceDB" "$W/resCp" "$W/Cp.tsv"
  # Headers already carry the real| tag on queries and bare ids on targets.
  awk -F'\t' '$1!=p{p=$1;n=0} ++n<=50' "$W/Cp.tsv" > "$OUT_CP"
  rm -rf "$W/tmp_cp" "$W/resCp"* "$W/Cp.tsv"
  echo "[C'] -> $OUT_CP"
fi

# ----------------------------------------------------------------- C ----------
PDBDIR="${PDBDIR:-$S/pdbstyle}"
TGZ="$S/pdbstyle-sel-gs-bib-40-2.08.tgz"
URL="https://scop.berkeley.edu/downloads/pdbstyle/pdbstyle-sel-gs-bib-40-2.08.tgz"
OUT_C="$BASE/results/scope_structure_C.tsv"

if [[ "${SKIP_ARM_C_REAL:-0}" == "1" ]]; then
  echo "[C]  skipped (SKIP_ARM_C_REAL=1)"; exit 0
fi

# The 40%-filtered coordinate set, matching the ASTRAL-40 sequences exactly.
# The download can be slow from some networks, so it is resumable and retried.
# Failure is non-fatal: C' above does not depend on it.
if [[ ! -d "$PDBDIR/pdbstyle-sel-gs-bib-40-2.08" ]]; then
  mkdir -p "$PDBDIR"
  if [[ ! -s "$TGZ" ]]; then
    echo "[dl] $URL (~1.0 GB, resumable)"
    curl -sS --fail --retry 20 --retry-delay 10 --retry-all-errors \
         -C - -o "$TGZ.part" "$URL" || { echo "[C] download failed; skipping arm C" >&2; exit 0; }
    mv "$TGZ.part" "$TGZ"
  fi
  tar -xzf "$TGZ" -C "$PDBDIR" && rm -f "$TGZ"
fi

if [[ ! -s "$W/allDB.dbtype" ]]; then
  echo "[db] createdb over $(find "$PDBDIR" -name '*.ent' | wc -l | tr -d ' ') structures"
  "$FOLDSEEK" createdb "$PDBDIR" "$W/allDB" --threads "$THREADS"
fi
# Foldseek names entries from the filename (d1dlwa_.ent -> d1dlwa_), which is
# already the SCOPe id; strip any suffix defensively rather than assume.
awk -F'\t' '{n=$2; sub(/\.ent$/,"",n); sub(/ .*/,"",n); print $1"\t"n}' \
    "$W/allDB.lookup" > "$W/key2id.tsv"

for side in query reference; do
  ids="$W/c_${side}_ids.txt"; keys="$W/c_${side}_keys.txt"
  awk '/^>/{print substr($1,2)}' "$S/scope_${side}.fasta" > "$ids"
  awk 'NR==FNR{want[$1];next} ($2 in want){print $1}' "$ids" "$W/key2id.tsv" > "$keys"
  echo "[cov] $side: $(wc -l < "$keys") / $(wc -l < "$ids") have coordinates"
  if [[ ! -s "$W/c_${side}DB.dbtype" ]]; then
    for sfx in "" _ss _ca _h; do
      "$FOLDSEEK" createsubdb "$keys" "$W/allDB$sfx" "$W/c_${side}DB$sfx" --subdb-mode 1 2>/dev/null || true
    done
  fi
done

if [[ ! -s "$OUT_C" ]]; then
  echo "[C]  search"
  "$FOLDSEEK" search "$W/c_queryDB" "$W/c_referenceDB" "$W/resC" "$W/tmp_c" \
      -e 1000 -s 9.5 --max-seqs "$MAX_SEQS" --threads "$THREADS"
  convert "$W/c_queryDB" "$W/c_referenceDB" "$W/resC" "$W/C.tsv"
  awk -F'\t' -v OFS='\t' '{sub(/\.ent$/,"",$1); sub(/\.ent$/,"",$2); $1="real|"$1;
                           if ($1!=p){p=$1;n=0} if (++n<=50) print}' "$W/C.tsv" > "$OUT_C"
  rm -rf "$W/tmp_c" "$W/resC"* "$W/C.tsv"
fi
echo "[done] $OUT_C"
