#!/usr/bin/env python3
"""Per-query Test 3 sensitivity as a CDF (input to Fig. S11).

Fig. S10 reports mean sensitivity per method; this shows the per-query
distribution behind the same all-vs-all measurement. The mean hides the shape:
a method that is uniformly mediocre and one that is perfect on half its queries
and empty on the rest report the same mean. For each method and level, the
output gives the fraction of queries reaching at least each sensitivity value
on a grid, plus a console summary of mean / zero / perfect at family level.

Inputs (per-query dumps from the two Test 3 scorers):

  results/test3_perquery_embed.csv    30_test3_allvall.py --per-query
  results/test3_perquery_{mmseqs,fsC,fsCp}.csv
                                      31_test3_score_tsv.py --per-query

Output: figures/data/fig9_test3_cdf.csv
Run:    python3 scripts/37_test3_cdf.py [--grid 100]

Stdlib only.
"""
import argparse
import bisect
import csv
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

# A subset of the Test 3 methods: both structure arms, the sequence baseline,
# the supervised reference, and three pLMs, each with and without whitening.
# All methods would be unreadable as curves.
DISPLAY = {
    "sequence (MMseqs2)": "sequence (MMseqs2)",
    "Foldseek": "Foldseek",
    "ProstT5-Foldseek": "ProstT5-Foldseek",
    "prottucker": "ProtTucker/EAT",
    "ankh_base": "Ankh-base", "ankh_base:whiten": "Ankh-base +whiten",
    "esm2_650m": "ESM-2 650M", "esm2_650m:whiten": "ESM-2 650M +whiten",
    "esmc_6b": "ESM-C 6B", "esmc_6b:whiten": "ESM-C 6B +whiten",
}
SOURCES = ["results/test3_perquery_embed.csv", "results/test3_perquery_mmseqs.csv",
           "results/test3_perquery_fsC.csv", "results/test3_perquery_fsCp.csv"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures/data/fig9_test3_cdf.csv")
    ap.add_argument("--grid", type=int, default=100)
    args = ap.parse_args()

    by = defaultdict(list)
    for src in SOURCES:
        p = BASE / src
        if not p.exists():
            print(f"  [warn] missing {src}")
            continue
        for r in csv.DictReader(open(p)):
            name = DISPLAY.get(r["model"])
            if name:
                by[(name, r["level"])].append(float(r["sens"]))
    if not by:
        raise SystemExit("no per-query input found; see the docstring")

    grid = [i / args.grid for i in range(args.grid + 1)]
    out = []
    for (m, lvl), v in by.items():
        v.sort()
        n = len(v)
        for g in grid:
            # fraction of queries reaching AT LEAST g
            out.append({"method": m, "level": lvl, "sens": round(g, 4),
                        "frac_queries": round((n - bisect_left(v, g)) / n, 6),
                        "n": n})

    f = BASE / args.out
    with f.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["method", "level", "sens",
                                           "frac_queries", "n"])
        w.writeheader()
        w.writerows(out)
    print(f"  wrote {f.relative_to(BASE)}  ({len(out)} rows, "
          f"{len(set(k[0] for k in by))} methods)")
    print(f"\n  {'method':<22}{'mean':>7}{'zero':>8}{'perfect':>9}   (family level)")
    for (m, lvl), v in sorted(by.items()):
        if lvl == "family":
            z = 100 * sum(1 for x in v if x == 0) / len(v)
            pf = 100 * sum(1 for x in v if x >= 0.999) / len(v)
            print(f"  {m:<22}{sum(v)/len(v):7.3f}{z:7.1f}%{pf:8.1f}%")


if __name__ == "__main__":
    main()
