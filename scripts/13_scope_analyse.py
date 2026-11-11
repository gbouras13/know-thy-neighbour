#!/usr/bin/env python3
"""Test 2 analysis: SCOPe superfamily-level correctness by query length.

Scoring is per-HIT, using SCOPe's structural hierarchy rather than any
sequence-derived label:

  TP         top hit shares the query's SUPERFAMILY (and, by construction of the
             split, comes from a different family)
  FP-fold    top hit is in a DIFFERENT FOLD -- non-homologous by expert
             structural assessment
  ambiguous  same fold, different superfamily. SCOPe folds can be analogous
             rather than homologous, so these are reported and excluded from
             both rates instead of being quietly scored as one or the other
  FP-decoy   any hit at all for a shuffled sequence

Ambiguous hits are reported separately rather than counted as TP or FP.

Inputs:  data/scope/scope_{query,reference}.jsonl and one search result table
         (--results, query in col 0, target in col 1, score in --score-col).
Outputs: results/scope_hits_<label>.tsv (per length bin x threshold) and a
         printed summary at E<1e-3 (or cosine>=0.95).
Run:     python3 scripts/13_scope_analyse.py --results results/scope_sequence.tsv
         --label sequence [--score-col 2 --better lower] [--header]

Stdlib only.
"""
import argparse, gzip, json
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
BINS = [(20, 60), (61, 100), (101, 150), (151, 250), (251, 400), (401, 10000)]


def bin_of(L):
    for lo, hi in BINS:
        if lo <= L <= hi:
            return f"{lo}-{hi}" if hi < 10000 else "401+"
    return None


def smart_open(p):
    p = Path(p)
    if p.suffix == ".gz": return gzip.open(p, "rt")
    if p.exists(): return open(p)
    gz = p.with_name(p.name + ".gz")
    if gz.exists(): return gzip.open(gz, "rt")
    raise SystemExit(f"missing {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/scope_sequence.tsv")
    ap.add_argument("--label", default="sequence")
    ap.add_argument("--score-col", type=int, default=2)
    ap.add_argument("--better", choices=["lower", "higher"], default="lower")
    ap.add_argument("--header", action="store_true")
    args = ap.parse_args()

    S = BASE / "data/scope"
    ref = {r["id"]: r for r in map(json.loads, open(S / "scope_reference.jsonl"))}
    qry = {r["id"]: r for r in map(json.loads, open(S / "scope_query.jsonl"))}

    # best hit per query
    best = {}
    with smart_open(BASE / args.results if not Path(args.results).is_absolute()
                    else Path(args.results)) as fh:
        if args.header: fh.readline()
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) <= args.score_col: continue
            try: s = float(p[args.score_col])
            except ValueError: continue
            cur = best.get(p[0])
            if cur is None or (s < cur[0] if args.better == "lower" else s > cur[0]):
                best[p[0]] = (s, p[1])

    thresholds = ([1e-10, 1e-5, 1e-3, 1e-2, 1e-1, 1.0] if args.better == "lower"
                  else [0.99, 0.975, 0.95, 0.90, 0.85, 0.80])
    passes = (lambda s, t: s < t) if args.better == "lower" else (lambda s, t: s >= t)

    n_ans = defaultdict(int); n_dec = defaultdict(int)
    for q in qry.values():
        b = bin_of(q["length"])
        if b and q["answerable"]:
            n_ans[b] += 1
    for qid in best:
        pass

    # decoy denominators: 5 seeds x every query
    for q in qry.values():
        b = bin_of(q["length"])
        if b: n_dec[b] += 5

    tp = defaultdict(lambda: defaultdict(int))
    amb = defaultdict(lambda: defaultdict(int))
    fpf = defaultdict(lambda: defaultdict(int))
    fpd = defaultdict(lambda: defaultdict(int))

    for qid, (score, tgt) in best.items():
        tag, acc = qid.split("|", 1)
        q = qry.get(acc); t = ref.get(tgt)
        if q is None or t is None: continue
        b = bin_of(q["length"])
        if not b: continue
        for th in thresholds:
            if not passes(score, th): continue
            if tag == "real":
                if not q["answerable"]: continue
                if q["superfamily"] == t["superfamily"]: tp[b][th] += 1
                elif q["fold"] == t["fold"]: amb[b][th] += 1
                else: fpf[b][th] += 1
            else:
                fpd[b][th] += 1

    labs = ["20-60", "61-100", "101-150", "151-250", "251-400", "401+"]
    out = BASE / f"results/scope_hits_{args.label}.tsv"
    with out.open("w") as fh:
        fh.write("length_bin\tn_answerable\tn_decoy\tthreshold\ttpr_superfamily"
                 "\tambiguous_samefold\tfpr_difffold\tfpr_shuffle\n")
        for b in labs:
            for th in thresholds:
                na, nd = n_ans[b], n_dec[b]
                if not na: continue
                fh.write(f"{b}\t{na}\t{nd}\t{th:g}\t{tp[b][th]/na:.4f}"
                         f"\t{amb[b][th]/na:.4f}\t{fpf[b][th]/na:.4f}"
                         f"\t{(fpd[b][th]/nd if nd else 0):.5f}\n")

    T = 1e-3 if args.better == "lower" else 0.95
    print(f"SCOPe/ASTRAL-40, structure-derived labels, {args.label} "
          f"@ {'E<' if args.better=='lower' else 'cos>='}{T:g}\n")
    print(f"{'bin':<10}{'n':>6}{'TP superfam':>13}{'amb samefold':>14}"
          f"{'FP difffold':>13}{'FP shuffle':>12}")
    for b in labs:
        na = n_ans[b]
        if not na: continue
        print(f"{b:<10}{na:>6}{100*tp[b][T]/na:>12.1f}%{100*amb[b][T]/na:>13.1f}%"
              f"{100*fpf[b][T]/na:>12.1f}%{100*fpd[b][T]/max(n_dec[b],1):>11.2f}%")
    print(f"\n[out] {out.relative_to(BASE)}")


if __name__ == "__main__":
    main()
