#!/usr/bin/env python3
"""Test 2 (SCOPe/ASTRAL-40): held-out threshold calibration with paired
resampling intervals.

Selecting a threshold on the same queries used to report sensitivity is
in-sample and optimistic. Because the quantity under study is calibration, this
script fits the threshold on one half of the queries and evaluates on the other.

Procedure, repeated R times:
  1. split answerable queries at random into calibration / evaluation halves;
  2. choose the threshold on the CALIBRATION half only, as the most permissive
     score holding different-fold FPR at or below the target;
  3. report TPR and the ACHIEVED FPR on the untouched evaluation half.

The achieved FPR is reported alongside TPR because a threshold fitted on half
the data need not deliver the nominal FPR on the other half.

Paired throughout -- every method sees the same split on every repeat -- so
differences from the baseline get intervals that account for shared queries.

Inputs:  data/scope/scope_{query,reference}.jsonl, results/scope_*.tsv[.gz].
Outputs: printed table of median TPR, achieved FPR and difference from the
         baseline, each with a 95% interval over repeats.
Run:     python3 scripts/24_heldout.py [--target-fpr 0.01] [--repeats 200]

Stdlib + numpy.
"""
import argparse
import gzip
import json
import random
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
S = BASE / "data/scope"


def smart(p):
    p = Path(p)
    if p.exists():
        return gzip.open(p, "rt") if p.suffix == ".gz" else open(p)
    g = p.with_name(p.name + ".gz")
    return gzip.open(g, "rt") if g.exists() else None


def outcomes(path, col, better, header, qry, ref):
    """query -> (score, outcome) with outcome 1=TP, 0=ambiguous, -1=FP-fold."""
    fh = smart(BASE / path)
    if fh is None:
        return None
    best = {}
    with fh:
        if header:
            fh.readline()
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) <= col:
                continue
            try:
                s = float(p[col])
            except ValueError:
                continue
            cur = best.get(p[0])
            if cur is None or (s > cur[0] if better == "higher" else s < cur[0]):
                best[p[0]] = (s, p[1])
    out = {}
    for qid, (s, t) in best.items():
        tag, acc = (qid.split("|", 1) if "|" in qid else ("real", qid))
        if tag != "real":
            continue
        q, tt = qry.get(acc), ref.get(t)
        if q is None or tt is None or not q["answerable"]:
            continue
        if q["superfamily"] == tt["superfamily"]:
            o = 1
        elif q["fold"] == tt["fold"]:
            o = 0
        else:
            o = -1
        # Sign-normalise so higher is always better; thresholds are then ">=".
        out[acc] = (s if better == "higher" else -s, o)
    return out


def pick_threshold(items, target):
    """Most permissive score with different-fold FPR <= target, on `items`."""
    if not items:
        return None
    arr = sorted(items, key=lambda x: -x[0])
    nF = sum(1 for _, o in arr if o == -1)
    if nF == 0:
        return arr[-1][0]
    fp = 0
    thr = arr[0][0]
    for s, o in arr:
        if o == -1:
            fp += 1
            if fp / nF > target:
                break
        thr = s
    return thr


def evaluate(items, thr, n_ans_eval):
    """(TPR over all evaluation queries, achieved different-fold FPR)."""
    tp = fp = nF = 0
    for s, o in items:
        if o == -1:
            nF += 1
        if s >= thr:
            if o == 1:
                tp += 1
            elif o == -1:
                fp += 1
    return (tp / max(n_ans_eval, 1), fp / nF if nF else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-fpr", type=float, default=0.01)
    ap.add_argument("--repeats", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--baseline", default="sequence (MMseqs2)")
    args = ap.parse_args()

    qry = {r["id"]: r for r in map(json.loads, open(S / "scope_query.jsonl"))}
    ref = {r["id"]: r for r in map(json.loads, open(S / "scope_reference.jsonl"))}

    methods = [("sequence (MMseqs2)", "results/scope_sequence.tsv", 2, "lower", False),
               ("Foldseek C'", "results/scope_structure_Cprime.tsv", 2, "lower", False)]
    for m in ["prott5", "prostt5", "glm2_650m_embed", "esm2_650m", "esmc_6b"]:
        methods.append((f"{m}", f"results/scope_knn_{m}.tsv.gz", 3, "higher", True))
    # Whitened variants, including the full ESM-C size series, so the ordering
    # across model sizes after correction gets intervals.
    for m, t in [("esm2_650m", "whiten"), ("glm2_650m_embed", "whiten"),
                 ("esm2_650m", "abtt"), ("esmc_300m", "whiten"),
                 ("esmc_600m", "whiten"), ("esmc_6b", "whiten")]:
        methods.append((f"{m} +{t}", f"results/scope_knn_{m}_{t}.tsv.gz", 3, "higher", True))
    for m, v in [("glm2_650m_embed", "nulldecoy"), ("esm2_650m", "nullself"),
                 ("glm2_650m_embed", "evdgumbel")]:
        methods.append((f"{m} +{v}", f"results/scope_knn_{m}_{v}.tsv.gz", 3, "higher", True))

    data = {}
    for name, path, col, better, header in methods:
        o = outcomes(path, col, better, header, qry, ref)
        if o:
            data[name] = o
    if not data:
        raise SystemExit("no result tables found")

    # Evaluate every method on exactly the same queries, so differences are
    # paired and not driven by differing coverage.
    common = set.intersection(*(set(v) for v in data.values()))
    common = sorted(common)
    print(f"Held-out threshold calibration, {args.repeats} random 50/50 splits.")
    print(f"Threshold chosen on the calibration half only; TPR and achieved FPR")
    print(f"reported on the untouched half. Target FPR {args.target_fpr:.0%}.")
    print(f"Queries scored by every method (paired): {len(common)}\n")

    rng = random.Random(args.seed)
    tprs = {n: [] for n in data}
    fprs = {n: [] for n in data}
    diffs = {n: [] for n in data if n != args.baseline}

    for _ in range(args.repeats):
        shuf = common[:]
        rng.shuffle(shuf)
        half = len(shuf) // 2
        cal, ev = set(shuf[:half]), shuf[half:]
        per = {}
        for n, o in data.items():
            thr = pick_threshold([o[a] for a in o if a in cal], args.target_fpr)
            if thr is None:
                continue
            t, f = evaluate([o[a] for a in ev if a in o], thr, len(ev))
            tprs[n].append(t)
            fprs[n].append(f)
            per[n] = t
        if args.baseline in per:
            for n, v in per.items():
                if n != args.baseline:
                    diffs[n].append(v - per[args.baseline])

    def ci(v):
        a = np.asarray([x for x in v if np.isfinite(x)])
        if a.size == 0:
            return (float("nan"),) * 3
        return (float(np.median(a)), float(np.percentile(a, 2.5)),
                float(np.percentile(a, 97.5)))

    print(f"{'method':<28}{'TPR (95% CI)':>24}{'achieved FPR':>20}"
          f"{'vs baseline':>22}")
    print("-" * 94)
    for n in data:
        t, tlo, thi = ci(tprs[n])
        f, flo, fhi = ci(fprs[n])
        cell = f"{100*t:6.1f} [{100*tlo:5.1f},{100*thi:5.1f}]"
        fcell = f"{100*f:5.2f} [{100*flo:4.2f},{100*fhi:4.2f}]"
        if n == args.baseline:
            d = "  (baseline)"
        else:
            dm, dlo, dhi = ci(diffs[n])
            sig = "" if (dlo < 0 < dhi) else " *"
            d = f"{100*dm:+6.1f} [{100*dlo:+5.1f},{100*dhi:+5.1f}]{sig}"
        print(f"{n:<28}{cell:>24}{fcell:>20}{d:>22}")
    print("\n* = paired 95% interval excludes zero.")
    print("Achieved FPR is measured on the evaluation half; the gap from the")
    print("nominal target is the cost of fitting the threshold on finite data.")


if __name__ == "__main__":
    main()
