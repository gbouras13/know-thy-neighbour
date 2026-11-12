#!/usr/bin/env python3
"""Test 1 top-1 accuracy, per model per transform.

Test 1's delivered TPR counts a query as recovered when its best-scoring hit
clears the threshold, whatever that hit is (the target is not checked). That is defensible (the queries are answerable AND remote, and the D6
decoys establish what a score means when there is no answer), but it is weaker
than "found the right homolog". This script adds the correctness check that
Test 2 has: is the rank-1 hit a true homolog? On the control arm, 93.6% of
MMseqs2's above-threshold hits are correct against 68.8% of ESM-C 6B's, so the
TPR metric, if anything, favours the embeddings relative to sequence search.

Correctness matches Test 1's own positive definition: query and target share
at least one Pfam family. Queries are the answerable AND remote subset, pooled
over all three length arms (the same 22,076 used by 17_stratify.py). Queries
with no hit count as misses.

Inputs:  results/knn_{arm}_<model>[_<transform>].tsv[.gz], query and reference
         JSONLs.
Output:  results/test1_top1.csv (input to Figs. S7 and S8 via
         44_ladder_figure_data.py).

  python3 scripts/43_test1_top1.py --models esm2_650m ... --transforms none whiten
"""
import argparse
import csv
import gzip
import json
from pathlib import Path

from tableio import find_table

BASE = Path(__file__).resolve().parent.parent
ARMS = ("small", "mid", "control")


def labels():
    ref = {}
    for line in open(BASE / "data/reference/joint_reference.jsonl"):
        r = json.loads(line)
        ref[r["acc"]] = set(r["pfam"])
    q = {}
    for arm in ARMS:
        for line in open(BASE / f"data/queries/joint_{arm}_query.jsonl"):
            r = json.loads(line)
            if r.get("answerable") and r.get("remote"):
                q[r["acc"]] = set(r["pfam"])
    return ref, q


def top1_for(stem_tmpl, ref, q, col=3):
    """Fraction of answerable+remote queries whose rank-1 hit shares a family.
    Queries with no hit at all count as misses, matching every other metric."""
    hit = seen = 0
    for arm in ARMS:
        p = find_table(BASE / stem_tmpl.format(arm=arm))   # .tsv or .tsv.gz
        if p is None:
            return None
        op = gzip.open if str(p).endswith(".gz") else open
        best = {}
        with op(p, "rt") as fh:
            first = fh.readline()
            if not first.startswith(("real|", "D")):
                pass
            else:
                fh.seek(0)
            for line in fh:
                a = line.rstrip("\n").split("\t")
                if len(a) <= col or not a[0].startswith("real|"):
                    continue
                try:
                    s = float(a[col])
                except ValueError:
                    continue
                acc = a[0][5:]
                cur = best.get(acc)
                if cur is None or s > cur[0]:
                    best[acc] = (s, a[1])
        for acc, (_s, t) in best.items():
            if acc in q:
                seen += 1
                if q[acc] & ref.get(t, set()):
                    hit += 1
    return hit, seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--transforms", nargs="+",
                    default=["none", "center", "diag", "abtt", "whiten"])
    ap.add_argument("--out", default="results/test1_top1.csv")
    args = ap.parse_args()

    ref, q = labels()
    print(f"  {len(q):,} answerable+remote queries; {len(ref):,} reference\n")
    rows = []
    for m in args.models:
        for t in args.transforms:
            tmpl = (f"results/knn_{{arm}}_{m}.tsv.gz" if t == "none"
                    else f"results/knn_{{arm}}_{m}_{t}.tsv")
            r = top1_for(tmpl, ref, q)
            if r is None:
                print(f"  [skip] {m} {t}")
                continue
            hit, seen = r
            pct = 100 * hit / max(seen, 1)
            rows.append({"model": m, "transform": t, "top1": round(pct, 2),
                         "n_hit": hit, "n_queries": seen})
            print(f"  {m:<18}{t:<8}top-1 {pct:6.2f}%   ({hit:,}/{seen:,})",
                  flush=True)
    out = BASE / args.out
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["model", "transform", "top1",
                                           "n_hit", "n_queries"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n  wrote {out.relative_to(BASE)}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
