#!/usr/bin/env python3
"""Test 1: recovery and false-positive rate stratified by query properties.

For every answerable and remote query, a method "recovers" it if its best hit
passes the method's own threshold, calibrated once to 1% FPR on the D6 natural
hard negatives of all three arms pooled (the same operating point as
25_bootstrap_test1.py). Recovery and the within-stratum D6 false-positive rate
are then tabulated by query properties, including length (Fig. S4), so that
failure can be located without hand-picked case studies.

Input:  data/queries/joint_<arm>_query.jsonl, data/decoys/<arm>_D6_*.fasta,
        results/ hit tables for each method
Output: printed tables; the length block is parsed by 27_figure_data.py
Run:    python3 scripts/17_stratify.py > results/stratified_recovery.txt

Stdlib only.
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from tableio import find_table, open_table

BASE = Path(__file__).resolve().parent.parent
ARMS = ("small", "mid", "control")
# Calibration window. The default pools every negative regardless of length, so
# all methods sit at the same achieved FPR; a single bin such as (321, 400) can
# be set instead, but its threshold does not hold 1% FPR in other bins.
CALIB_BIN = (1, 10**9)

METHODS = [
    ("sequence (MMseqs2)", "sequence_{arm}_vs_joint.tsv.gz",   2, "lower"),
    ("Foldseek",         "structure_Cn_{arm}.tsv.gz",        2, "lower"),
    ("ProstT5-Foldseek",        "structure_Cprime_{arm}_top50.tsv.gz", 2, "lower"),
    ("gLM2-embed",         "knn_{arm}_glm2_650m_embed.tsv.gz", 3, "higher"),
    ("ProtT5",             "knn_{arm}_prott5.tsv.gz",          3, "higher"),
    ("ProstT5",            "knn_{arm}_prostt5.tsv.gz",         3, "higher"),
    ("ProtTucker/EAT",     "knn_{arm}_prottucker.tsv.gz",      3, "higher"),
    ("Ankh-base",          "knn_{arm}_ankh_base.tsv.gz",       3, "higher"),
    ("Ankh-large",         "knn_{arm}_ankh_large.tsv.gz",      3, "higher"),
    ("ESM-2 150M",         "knn_{arm}_esm2_150m.tsv.gz",       3, "higher"),
    ("ESM-2 650M",         "knn_{arm}_esm2_650m.tsv.gz",       3, "higher"),
    ("ESM-2 3B",           "knn_{arm}_esm2_3b.tsv.gz",         3, "higher"),
    ("ESM-C 300M",         "knn_{arm}_esmc_300m.tsv.gz",       3, "higher"),
    ("ESM-C 600M",         "knn_{arm}_esmc_600m.tsv.gz",       3, "higher"),
    ("ESM-C 6B",           "knn_{arm}_esmc_6b.tsv.gz",         3, "higher"),
]


def best_scores(path, col, better):
    """query id -> best score. Skips a header line if present."""
    best = {}
    with open_table(path) as fh:
        first = fh.readline()
        if first and not first.startswith(("real|", "D0", "D1", "D2", "D3",
                                           "D4", "D5", "D6")):
            pass                      # header consumed
        else:
            fh.seek(0)
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) <= col:
                continue
            try:
                s = float(p[col])
            except ValueError:
                continue
            cur = best.get(p[0])
            if cur is None or (s > cur if better == "higher" else s < cur):
                best[p[0]] = s
    return best


def load_negatives(neg):
    """D6 negatives. They are real proteins and carry the same covariates as
    the queries, which makes a within-stratum FPR possible: at a fixed
    threshold the FPR can drift with length, so recovery alone is not
    interpretable."""
    out = {}
    for arm in ARMS:
        f = BASE / "data/decoys" / f"{arm}_{neg}.jsonl"
        if not f.exists():
            continue
        for line in open(f):
            r = json.loads(line)
            r["arm"] = arm
            out[r["acc"]] = r
    return out


def load_queries():
    q = {}
    for arm in ARMS:
        f = BASE / "data/queries" / f"joint_{arm}_query.jsonl"
        if not f.exists():
            continue
        for line in open(f):
            r = json.loads(line)
            r["arm"] = arm
            q[r["acc"]] = r
    return q


def decoy_lengths(arm, neg):
    """decoy id -> length, from the issued FASTA."""
    fa = BASE / "data/decoys" / f"{arm}_{neg}.fasta"
    out, cid, n = {}, None, 0
    if not fa.exists():
        return out
    for line in open(fa):
        if line.startswith(">"):
            if cid:
                out[cid] = n
            cid, n = line[1:].split()[0], 0
        else:
            n += len(line.strip())
    if cid:
        out[cid] = n
    return out


def threshold(scores, arm_neg_lengths, neg, better, target_fpr):
    """Score achieving `target_fpr` on D6 negatives in the calibration window.

    With the default CALIB_BIN the window is every negative (pooled).
    """
    vals = []
    for cid, L in arm_neg_lengths.items():
        if not (CALIB_BIN[0] <= L <= CALIB_BIN[1]):
            continue
        s = scores.get(f"{neg}|{cid}")
        if s is not None:
            vals.append(s)
    n_issued = sum(1 for L in arm_neg_lengths.values()
                   if CALIB_BIN[0] <= L <= CALIB_BIN[1])
    if not n_issued:
        return None
    vals.sort(reverse=(better == "higher"))
    k = int(target_fpr * n_issued)
    if k < len(vals):
        return vals[k]
    return vals[-1] if vals else None


# ---------------------------------------------------------------- strata -----
def strata_of(r):
    """Yield (factor, level) pairs for one query."""
    L = r["length"]
    yield "length", ("1-49" if L < 50 else "50-99" if L < 100 else
                     "100-199" if L < 200 else "200-400")
    yield "transmembrane", "TM" if r.get("transmembrane") else "not TM"
    lc = r.get("low_complexity") or 0.0
    yield "low_complexity", ("0" if lc <= 0.0 else
                             "0-10%" if lc < 0.10 else ">=10%")
    g = r.get("gravy")
    if g is not None:
        yield "hydrophobicity", ("hydrophilic (<-0.5)" if g < -0.5 else
                                 "neutral" if g < 0.5 else
                                 "hydrophobic (>=0.5)")
    yield "evidence", "experimental" if r.get("experimental") else "inferred"
    p = r.get("max_pident_qcov80")
    if p is not None:
        yield "max identity", ("<=10%" if p <= 10 else
                               "10-20%" if p <= 20 else "20-30%")
    yield "n Pfam families", ("1" if len(r.get("answerable_families") or []) <= 1
                              else ">1")


def wilson(k, n):
    """95% Wilson interval -- behaves at k=0 and small n, unlike normal approx."""
    if not n:
        return (0.0, 0.0)
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-fpr", type=float, default=0.01)
    ap.add_argument("--neg", default="D6_natural_hard_all")
    ap.add_argument("--min-n", type=int, default=30,
                    help="suppress strata with fewer positives than this")
    args = ap.parse_args()

    queries = load_queries()
    negs = load_negatives(args.neg)
    pos = {a: r for a, r in queries.items()
           if r.get("answerable") and r.get("remote")}
    print(f"Stratified recovery of {len(pos)} remote, answerable queries.")
    print(f"'Recovered' = best hit passes that method's own threshold, "
          f"calibrated to {args.target_fpr:.0%} FPR on "
          + ("ALL lengths pooled" if CALIB_BIN == (1, 10**9)
             else f"{CALIB_BIN[0]}-{CALIB_BIN[1]} aa") + ".\n")

    results = {}
    for name, tmpl, col, better in METHODS:
        # The threshold is calibrated ONCE over the negatives of all three
        # arms pooled, then held fixed across every arm, matching the operating
        # point of 25_bootstrap_test1.py.
        cal_scores, cal_lengths = {}, {}
        for a in ARMS:
            f = find_table(BASE / "results" / tmpl.format(arm=a))   # .tsv or .tsv.gz
            if f is None:
                continue
            # Qualify both sides by arm so accessions cannot collide between
            # arms, and so threshold()'s f"{neg}|{cid}" lookup still resolves.
            for c, L in decoy_lengths(a, args.neg).items():
                cal_lengths[f"{a}|{c}"] = L
            for k, v in best_scores(f, col, better).items():
                if k.startswith(f"{args.neg}|"):
                    cal_scores[f"{args.neg}|{a}|{k.split('|', 1)[1]}"] = v
        if not cal_lengths:
            continue
        thr = threshold(cal_scores, cal_lengths, args.neg, better,
                        args.target_fpr)
        if thr is None:
            continue

        rec = {}
        for arm in ARMS:
            f = find_table(BASE / "results" / tmpl.format(arm=arm))   # .tsv or .tsv.gz
            if f is None:
                continue
            sc = best_scores(f, col, better)
            for acc, r in pos.items():
                if r["arm"] != arm:
                    continue
                s = sc.get(f"real|{acc}")
                rec[acc] = bool(s is not None and
                                (s >= thr if better == "higher" else s < thr))
        negrec = {}
        for arm in ARMS:
            f = find_table(BASE / "results" / tmpl.format(arm=arm))   # .tsv or .tsv.gz
            if f is None:
                continue
            sc = best_scores(f, col, better)
            for acc, r in negs.items():
                if r["arm"] != arm:
                    continue
                s = sc.get(f"{args.neg}|{acc}")
                negrec[acc] = bool(s is not None and
                                   (s >= thr if better == "higher" else s < thr))
        if rec:
            results[name] = (rec, negrec)

    factors = defaultdict(set)
    for r in pos.values():
        for f, lv in strata_of(r):
            factors[f].add(lv)

    order = {
        "length": ["1-49", "50-99", "100-199", "200-400"],
        "transmembrane": ["not TM", "TM"],
        "low_complexity": ["0", "0-10%", ">=10%"],
        "hydrophobicity": ["hydrophilic (<-0.5)", "neutral", "hydrophobic (>=0.5)"],
        "evidence": ["experimental", "inferred"],
        "max identity": ["<=10%", "10-20%", "20-30%"],
        "n Pfam families": ["1", ">1"],
    }

    names = list(results)
    for fac in order:
        levels = [lv for lv in order[fac] if lv in factors.get(fac, ())]
        if len(levels) < 2:
            continue
        print(f"=== {fac} ===   cells are  TPR% / FPR%  at the same fixed threshold")
        print(f"{'level':<22}{'nPos':>6}{'nNeg':>6}" +
              "".join(f"{n[:18]:>20}" for n in names))
        for lv in levels:
            accs = [a for a, r in pos.items() if (fac, lv) in set(strata_of(r))]
            nacc = [a for a, r in negs.items() if (fac, lv) in set(strata_of(r))]
            if len(accs) < args.min_n:
                continue
            cells = []
            for n in names:
                rec, negrec = results[n]
                sub = [a for a in accs if a in rec]
                nsub = [a for a in nacc if a in negrec]
                if not sub:
                    cells.append(f"{'-':>20}")
                    continue
                tpr = 100 * sum(rec[a] for a in sub) / len(sub)
                if nsub:
                    fpr = 100 * sum(negrec[a] for a in nsub) / len(nsub)
                    cells.append(f"{tpr:>6.1f} /{fpr:>5.1f}".rjust(20))
                else:
                    cells.append(f"{tpr:>6.1f} /  n/a".rjust(20))
            print(f"{lv:<22}{len(accs):>6}{len(nacc):>6}" + "".join(cells))
        print()


if __name__ == "__main__":
    main()
