#!/usr/bin/env python3
"""Assemble transform-ladder figure inputs for both benchmarks and both metrics.

Writes one CSV per benchmark, each with delivered sensitivity and top-1
accuracy per model and transform (none, center, diag, abtt, whiten):

  fig6_ladder.csv          Test 2   <- fig3b_ceiling.csv
                           (Fig. S6 delivered; Fig. S9 top-1)
  figS9_ladder_test1.csv   Test 1   <- threshold_ladder_pooled.csv
                                       + test1_top1.csv
                           (Fig. S7 delivered; Fig. S8 top-1)

The two benchmarks measure different things and are NOT interchangeable:

  Test 2 delivered  matched 1% different-fold FPR, correctness verified
                    (top-1 hit must be same-superfamily)
  Test 1 delivered  TPR at a pooled 1%-FPR threshold against D6 natural
                    negatives, IN-SAMPLE (one fit, no resampling); does NOT
                    verify the hit is correct
  Test 1 top-1      from 43_test1_top1.py; the correctness check Test 1's TPR
                    omits

  python3 scripts/44_ladder_figure_data.py
"""
import csv
import re
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
ORDER = ["none", "center", "diag", "abtt", "whiten"]
DISP = {"esm2_150m": "ESM-2 150M", "esm2_650m": "ESM-2 650M", "esm2_3b": "ESM-2 3B",
        "esmc_300m": "ESM-C 300M", "esmc_600m": "ESM-C 600M", "esmc_6b": "ESM-C 6B",
        "ankh_base": "Ankh-base", "ankh_large": "Ankh-large", "prott5": "ProtT5",
        "prostt5": "ProstT5", "glm2_650m_embed": "gLM2-embed",
        "prottucker": "ProtTucker/EAT"}


def test2():
    """Test 2: delivered and top-1 straight from the ceiling table.

    ProtTucker is deliberately EXCLUDED from the ladder figures. It is a
    supervised contrastive head, not a raw pLM, so an anisotropy correction has
    no premise there; measurement agrees: run under its native Euclidean
    metric the whole ladder moves it by at most 1.8 points (58.8 baseline ->
    59.9 diag, 59.2 ABTT, 57.0 whiten, and centering is a mathematical no-op
    under Euclidean, reproducing 58.8 exactly). It keeps its baseline row in
    Fig. 3, where the comparison is to other methods rather than to
    transforms of itself.
    """
    rows = []
    for r in csv.DictReader(open(BASE / "figures/data/fig3b_ceiling.csv")):
        if r["transform"] not in ORDER or r["method"] == "ProtTucker/EAT":
            continue
        rows.append({"model": r["method"].replace(" kNN", ""),
                     "transform": r["transform"], "benchmark": "Test 2 (SCOPe)",
                     "delivered": r["delivered"], "top1": r["top1"]})
    return rows


def test1():
    """Test 1: delivered from the threshold ladder, top-1 from 43_."""
    deliv = {}
    p = BASE / "results/threshold_ladder_pooled.csv"
    for r in csv.DictReader(open(p)):
        name = r["method"].replace(" kNN", "").strip()
        m = re.match(r"^(.*?)(?: \+(center|diag|abtt|whiten))?$", name)
        deliv[(m.group(1).strip(), m.group(2) or "none")] = float(r["tpr"])
    top1 = {}
    q = BASE / "results/test1_top1.csv"
    if q.exists():
        for r in csv.DictReader(open(q)):
            top1[(DISP.get(r["model"], r["model"]), r["transform"])] = float(r["top1"])
    rows = []
    for (mod, t), dv in sorted(deliv.items()):
        if t not in ORDER or mod == "ProtTucker/EAT":
            continue
        rows.append({"model": mod, "transform": t, "benchmark": "Test 1 (Swiss-Prot)",
                     "delivered": round(dv, 2),
                     "top1": (round(top1[(mod, t)], 2) if (mod, t) in top1 else "")})
    return rows


def write(path, rows):
    out = BASE / path
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["model", "transform", "benchmark",
                                           "delivered", "top1"])
        w.writeheader()
        w.writerows(rows)
    full = sum(1 for m in {r["model"] for r in rows}
               if len([r for r in rows if r["model"] == m]) == 5)
    print(f"  {out.relative_to(BASE)}: {len(rows)} rows, "
          f"{len({r['model'] for r in rows})} models, {full} with the full ladder")


def main():
    write("figures/data/fig6_ladder.csv", test2())
    write("figures/data/figS9_ladder_test1.csv", test1())


if __name__ == "__main__":
    main()
