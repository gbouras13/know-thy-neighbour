#!/usr/bin/env python3
"""Measure how far each Test 1 query sits from the reference.

Clustering does not guarantee remote-homology separation, so this script
searches every query against the reference permissively (MMseqs2, -s 7.5,
-e 10000) and records, per query, the maximum sequence identity attained at
several query-coverage requirements.

Coverage matters because identity alone is meaningless on a short local
alignment: 40% identity over 12 residues is noise, 40% over 90% of the query is
a homolog. Reporting max identity at qcov >= 0, 0.3, 0.5 and 0.8 separates the
two instead of collapsing them.

The result is written back onto each query record as max_pident_qcov{0,30,50,80}
plus a boolean "remote" (<=30% identity at qcov >= 0.8). Analysis can then
stratify true-positive performance by measured identity rather than relying on
a single cut. Synthetic decoys have no relationship to the reference by
construction, so they are unaffected by this step.

Input:  data/queries/{prefix}_{arm}_query.{fasta,jsonl},
        data/reference/{prefix}_reference.fasta (from 02c_joint_split.py)
Output: query JSONL updated in place; alignments cached under
        data/interim/separation_{prefix}/

Usage: python scripts/02d_measure_separation.py [--prefix joint] [--threads N]
"""

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
MMSEQS = Path.home() / "miniforge3/envs/mmseqs2/bin/mmseqs"
COV_LEVELS = [0.0, 0.3, 0.5, 0.8]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="joint", help="split prefix written by 02c")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--sensitivity", type=float, default=7.5)
    args = ap.parse_args()

    ref_fa = BASE / "data/reference" / f"{args.prefix}_reference.fasta"
    work = BASE / "data/interim" / f"separation_{args.prefix}"
    work.mkdir(parents=True, exist_ok=True)
    if not ref_fa.exists():
        raise SystemExit(f"missing {ref_fa} -- run 02c_joint_split.py first")

    n_ref = sum(1 for line in open(ref_fa) if line.startswith(">"))

    for arm in ["small", "mid", "control"]:
        q_fa = BASE / "data/queries" / f"{args.prefix}_{arm}_query.fasta"
        q_js = BASE / "data/queries" / f"{args.prefix}_{arm}_query.jsonl"

        # The cache name includes the query and reference counts so a rebuilt
        # split cannot reuse stale alignments. Queries missing from a stale
        # file would default to 0% identity and be mislabelled "remote".
        n_q = sum(1 for line in open(q_fa) if line.startswith(">"))
        aln = work / f"{arm}_q{n_q}_vs_ref{n_ref}.tsv"

        if not aln.exists():
            cmd = [str(MMSEQS), "easy-search", str(q_fa), str(ref_fa), str(aln),
                   str(work / f"tmp_{arm}"),
                   "-s", str(args.sensitivity), "-e", "10000",
                   "--max-seqs", "2000", "--threads", str(args.threads), "-v", "1",
                   "--format-output", "query,target,pident,alnlen,qcov,tcov,evalue"]
            print(f"[search] {arm} vs {args.prefix} reference", flush=True)
            subprocess.run(cmd, check=True)
            subprocess.run(["rm", "-rf", str(work / f"tmp_{arm}")], check=False)

        best = defaultdict(lambda: {c: 0.0 for c in COV_LEVELS})
        with aln.open() as fh:
            for line in fh:
                p = line.rstrip("\n").split("\t")
                if len(p) < 7:
                    continue
                q, pid, qcov = p[0], float(p[2]), float(p[4])
                # mmseqs reports pident as a percentage in this format.
                for c in COV_LEVELS:
                    if qcov >= c and pid > best[q][c]:
                        best[q][c] = pid

        rows = [json.loads(l) for l in open(q_js) if l.strip()]
        for r in rows:
            b = best.get(r["acc"])
            for c in COV_LEVELS:
                key = f"max_pident_qcov{int(c * 100)}"
                r[key] = round(b[c], 2) if b else 0.0
            # "Remote": no reference sequence exceeds 30% identity over 80%
            # of the query.
            r["remote"] = r["max_pident_qcov80"] <= 30.0
        with open(q_js, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")

        n = len(rows)
        ans = [r for r in rows if r.get("answerable")]
        rem = [r for r in ans if r["remote"]]
        print(f"[{arm}] {n} queries, {len(ans)} answerable, "
              f"{len(rem)} of those remote (<=30% id at qcov>=0.8)", flush=True)
        for c in COV_LEVELS:
            key = f"max_pident_qcov{int(c * 100)}"
            hi = sum(1 for r in ans if r[key] > 30.0)
            print(f"    qcov>={c:.1f}: {hi:6d} answerable queries "
                  f"({100 * hi / max(len(ans), 1):5.1f}%) exceed 30% identity")

    print("\nStratify results by max_pident_qcov80 rather than filtering on it.")


if __name__ == "__main__":
    main()
