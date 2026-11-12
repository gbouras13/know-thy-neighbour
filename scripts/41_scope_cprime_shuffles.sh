#!/usr/bin/env bash
# Test 2 arm C' (ProstT5-Foldseek): search the SHUFFLED decoy queries.
#
# Foldseek on real structures (arm C) cannot score shuffles, which have no
# experimental structure, but C' predicts 3Di from sequence, so the shuffle
# class can be searched like the embedding arms.
#
# Mirrors 16b_scope_build_search.sh: same foldseek settings (-e 1000 -s 9.5
# --max-seqs), same convertalis columns, same reference DB; only the query
# side changes.
#
# Inputs:  data/scope_3di_shuffle/shuffle_{aa,3di}.fasta (from 41a),
#          reference DB built by 16b.
# Output:  results/scope_structure_Cprime_shuffles.tsv
#
#   bash scripts/41_scope_cprime_shuffles.sh [threads]
set -euo pipefail
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FOLDSEEK="${FOLDSEEK:-$(command -v foldseek || true)}"
if [[ -z "$FOLDSEEK" ]] || ! command -v "$FOLDSEEK" >/dev/null 2>&1; then
  echo "foldseek not found: put it on PATH or set FOLDSEEK=/path/to/foldseek" >&2
  exit 1
fi
THREADS="${1:-8}"; MAX_SEQS="${MAX_SEQS:-1000}"
W="${SCOPE_WORK:-$BASE/data/interim/scope_struct}"; mkdir -p "$W"
IN="$BASE/data/scope_3di_shuffle"
OUT="$BASE/results/scope_structure_Cprime_shuffles.tsv"
[[ -s "$OUT" ]] && { echo "[skip] $OUT exists"; exit 0; }
[[ -s "$IN/shuffle_3di.fasta" ]] || { echo "run 41a (3Di prediction) first" >&2; exit 1; }
[[ -s "$W/cp_referenceDB.dbtype" ]] || { echo "reference DB missing: run 16b first" >&2; exit 1; }

python3 - "$IN" "$W" <<'PY'
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
aid, aa = read_fa(in3di / "shuffle_aa.fasta")
tid, td = read_fa(in3di / "shuffle_3di.fasta")
assert aid == tid, "aa and 3di id order differ"
bad = [i for i in range(len(aid)) if len(aa[i]) != len(td[i])]
if bad: raise SystemExit(f"{len(bad)} aa/3di length mismatches")
with open(work/"shuf_aa.tsv","w") as fa, open(work/"shuf_3di.tsv","w") as f3, \
     open(work/"shuf_h.tsv","w") as fh:
    for n, i in enumerate(range(len(aid)), start=1):
        fa.write(f"{n}\t{aa[i]}\n"); f3.write(f"{n}\t{td[i]}\n")
        fh.write(f"{n}\t{aid[i]}\n")
print(f"[C'-shuf] queries: {len(aid)}")
PY

db="$W/cp_shufDB"
if [[ ! -s "$db.dbtype" ]]; then
  "$FOLDSEEK" tsv2db "$W/shuf_aa.tsv"  "$db"      --output-dbtype 0
  "$FOLDSEEK" tsv2db "$W/shuf_3di.tsv" "${db}_ss" --output-dbtype 0
  "$FOLDSEEK" tsv2db "$W/shuf_h.tsv"   "${db}_h"  --output-dbtype 12
fi
echo "[C'-shuf] search"
"$FOLDSEEK" search "$db" "$W/cp_referenceDB" "$W/resCpShuf" "$W/tmp_cpshuf" \
    -e 1000 -s 9.5 --max-seqs "$MAX_SEQS" --threads "$THREADS"
"$FOLDSEEK" convertalis "$db" "$W/cp_referenceDB" "$W/resCpShuf" "$OUT" \
    --format-output "query,target,evalue,bits,fident,alnlen,qlen,tlen,qcov,tcov" \
    --threads "$THREADS"
echo "[C'-shuf] wrote $OUT ($(wc -l < "$OUT") rows)"
