#!/usr/bin/env python3
"""Score an all-vs-all hit list under the Foldseek SCOP benchmark protocol.

Companion to `30_test3_allvall.py`, which does the same for embeddings. This one
takes a TSV of `query, target, evalue` from an alignment arm (MMseqs2, Foldseek
on real structures, ProstT5-Foldseek), so every Test 3 arm is scored by the
same protocol.

THE PROTOCOL, as stated in the Foldseek paper's methods:

  "we compare the domains all-versus-all. Per domain, we measured the fraction
   of detected TPs up to the first FP. For family-level, superfamily-level and
   fold-level recognition, TPs were defined as same family, same superfamily and
   not same family, and same fold and not same superfamily, respectively. Hits
   from different folds are FPs."

which is what `bench.noselfhit.awk` implements and what both scoring scripts
do.

--restrict: the arms do not cover the same domains. Embeddings have a vector
for all 15,176 domains; Foldseek on real structures has usable coordinates for
13,822, because NMR multi-model entries are dropped rather than deduplicated.
Denominators over all 15,176 would charge Foldseek for true positives absent
from its database. `--restrict` takes an allowlist and applies it to BOTH the
hits and the denominators, so every arm is scored on an identical domain set.
Use the intersection of what every arm covers, and pass the same file to this
script and to `30_test3_allvall.py --restrict`.

Inputs:  --hits TSV (query, target, evalue; plain or .gz), SCOPe split JSONLs.
Output:  one row per level appended to --out (default results/test3_scored.csv);
         optionally per-query sensitivities (--per-query).
Run:     python scripts/31_test3_score_tsv.py --hits HITS.tsv --name NAME \
             --restrict common_ids.txt --out results/test3_common_NAME.csv

Stdlib only.
"""

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
S = BASE / "data/scope"


def smart(p):
    p = Path(p)
    if p.suffix == ".gz" and p.exists():
        return gzip.open(p, "rt")
    if p.exists():
        return open(p)
    g = p.with_name(p.name + ".gz")
    return gzip.open(g, "rt") if g.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hits", required=True, help="TSV: query, target, evalue")
    ap.add_argument("--name", required=True)
    ap.add_argument("--restrict", help="file of domain ids, one per line")
    ap.add_argument("--lower-is-better", action="store_true", default=True)
    ap.add_argument("--out", default="results/test3_scored.csv")
    ap.add_argument("--per-query", help="also write every query's sensitivity, "
                    "for the per-query distribution (Fig. S11). See the same flag on "
                    "30_test3_allvall.py.")
    args = ap.parse_args()

    sccs = {}
    for f in ("scope_reference.jsonl", "scope_query.jsonl"):
        for r in map(json.loads, open(S / f)):
            sccs[r["id"]] = r["sccs"]

    if args.restrict:
        keep = {l.strip() for l in open(args.restrict) if l.strip()}
        sccs = {k: v for k, v in sccs.items() if k in keep}
    print(f"[{args.name}] scoring over {len(sccs)} domains")

    fam = dict(sccs)
    sfam = {i: s.rsplit(".", 1)[0] for i, s in sccs.items()}
    fold = {i: s.rsplit(".", 2)[0] for i, s in sccs.items()}
    fc, sc, oc = defaultdict(int), defaultdict(int), defaultdict(int)
    for i in sccs:
        fc[fam[i]] += 1
        sc[sfam[i]] += 1
        oc[fold[i]] += 1

    hits = defaultdict(list)
    fh = smart(BASE / args.hits)
    if fh is None:
        raise SystemExit(f"missing {args.hits}")
    with fh:
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) < 3:
                continue
            # Query ids may carry the "real|" tag from the Test 2 pipeline.
            q = p[0].split("|", 1)[-1]
            t = p[1].split("|", 1)[-1]
            if q not in sccs or t not in sccs or q == t:
                continue
            try:
                e = float(p[2])
            except ValueError:
                continue
            hits[q].append((e if args.lower_is_better else -e, t))

    acc = {"family": [], "superfamily": [], "fold": []}
    qids = []
    for q in sccs:
        # The awk END-block filter: the query needs a partner at every level.
        if not (fc[fam[q]] > 1 and sc[sfam[q]] - fc[fam[q]] > 0
                and oc[fold[q]] - sc[sfam[q]] > 0):
            continue
        ff = fs = fo = 0
        for _e, t in sorted(hits.get(q, [])):
            if fold[t] != fold[q]:          # first different-fold hit is the FP
                break
            if fam[t] == fam[q]:
                ff += 1
            elif sfam[t] == sfam[q]:
                fs += 1
            else:
                fo += 1
        qids.append(q)
        acc["family"].append(ff / (fc[fam[q]] - 1))
        acc["superfamily"].append(fs / (sc[sfam[q]] - (fc[fam[q]] - 1)))
        acc["fold"].append(fo / (oc[fold[q]] - (sc[sfam[q]] - 1)))

    if args.per_query:
        pq = BASE / args.per_query
        with pq.open("w") as fh:
            fh.write("model,query,level,sens\n")
            for lvl in ("family", "superfamily", "fold"):
                for qid, v in zip(qids, acc[lvl]):
                    fh.write(f"{args.name},{qid},{lvl},{v:.6f}\n")
        print(f"-> {pq.relative_to(BASE)}")

    n = len(acc["family"])
    out = BASE / args.out
    new = not out.exists()
    with out.open("a") as fh:
        if new:
            fh.write("model,level,n_queries,mean_sens\n")
        for lvl in ("family", "superfamily", "fold"):
            m = sum(acc[lvl]) / n if n else float("nan")
            fh.write(f"{args.name},{lvl},{n},{round(m, 5)}\n")
            print(f"  {lvl:<12} {m:.4f}")
    print(f"  n={n} queries -> {out.relative_to(BASE)}")


if __name__ == "__main__":
    main()
