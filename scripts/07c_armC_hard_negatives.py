#!/usr/bin/env python3
"""Add a negative class to Test 1 Arm C (Foldseek on AFDB structures) by
re-tagging existing results.

Arm C uses real AFDB structures on both sides, so synthetic decoys (D0-D5),
which have no structure, cannot serve as negatives for it. The D6 natural hard
negatives can: 03b_natural_negatives.py draws them from the query set (queries
that are remote and have no clan-mate in the reference). Positives and
negatives are disjoint by answerability, so a D6 member is never also counted
as a positive.

Arm C already searched every real query, including every D6 member, under the
tag "real|<acc>". Re-running Foldseek would reproduce the same rows, so this
script copies each D6 member's rows with the tag "<neg class>|<acc>".

Negatives with no row (no AFDB structure, or no hit) are left absent; the
downstream evaluation counts every issued negative, so an absent negative
scores as a non-hit (the conservative direction).

Input:  results/structure_C_{arm}_top50.tsv.gz,
        data/decoys/{arm}_D6_natural_hard_all.fasta
Output: results/structure_Cn_{arm}.tsv.gz

Usage: python scripts/07c_armC_hard_negatives.py --arm small|mid|control
"""
import argparse
import gzip
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent


def accs(fasta):
    return {l[1:].split()[0] for l in open(fasta) if l.startswith(">")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="small")
    ap.add_argument("--neg", default="D6_natural_hard_all")
    args = ap.parse_args()

    src = BASE / f"results/structure_C_{args.arm}_top50.tsv.gz"
    if not src.exists():
        raise SystemExit(f"missing {src}")
    neg = accs(BASE / f"data/decoys/{args.arm}_{args.neg}.fasta")
    out = BASE / f"results/structure_Cn_{args.arm}.tsv.gz"

    n_in = n_neg_rows = 0
    seen = set()
    with gzip.open(src, "rt") as fh, gzip.open(out, "wt") as w:
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) < 3:
                continue
            n_in += 1
            w.write(line)                      # keep the real| row as-is
            acc = p[0].split("|", 1)[1] if "|" in p[0] else p[0]
            if acc in neg:
                p[0] = f"{args.neg}|{acc}"
                w.write("\t".join(p) + "\n")
                n_neg_rows += 1
                seen.add(acc)

    print(f"[{args.arm}] rows in={n_in}  negative rows added={n_neg_rows}")
    print(f"[{args.arm}] negatives with results: {len(seen)}/{len(neg)} "
          f"({len(neg) - len(seen)} absent -> scored as non-hits)")
    print(f"[{args.arm}] -> {out.relative_to(BASE)}")


if __name__ == "__main__":
    main()
