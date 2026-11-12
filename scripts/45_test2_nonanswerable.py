#!/usr/bin/env python3
"""Test 2: do the 1% thresholds hold on an INDEPENDENT negative set?

Test 2's FPR is defined on wrong-fold top hits among the 3,503 answerable
queries, so the same query pool supplies positives and negatives. The 566
non-answerable queries (superfamily absent from the reference) are otherwise
unused and form an independent negative set: a same-superfamily hit is
impossible by construction, so any accepted top hit in a DIFFERENT fold is a
false positive. Same-fold top hits are ambiguous and excluded, as in Test 2
scoring.

Procedure, per method, on the full data (no resampling):
  1. fit the published threshold: most permissive score holding different-fold
     FPR <= target over the answerable queries (24_heldout.pick_threshold);
  2. check it reproduces the published TP rate (fig3b_ceiling.csv "delivered");
  3. apply it to the non-answerable queries and report the achieved FPR there,
     with an exact (Clopper-Pearson) 95% interval.

Inputs:  results/scope_*.tsv[.gz] hit tables, SCOPe split JSONLs,
         figures/data/fig3b_ceiling.csv (for the reproduction check).
Output:  results/test2_nonanswerable.tsv, plus a printed table.
Run:     python scripts/45_test2_nonanswerable.py [--target-fpr 0.01]

Stdlib only.
"""
import argparse
import csv
import gzip
import json
import math
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
S = BASE / "data/scope"


def smart(p):
    p = Path(p)
    if p.exists():
        return gzip.open(p, "rt") if p.suffix == ".gz" else open(p)
    g = p.with_name(p.name + ".gz")
    if g.exists():
        return gzip.open(g, "rt")
    raw = p.with_suffix("") if p.suffix == ".gz" else None
    return open(raw) if raw is not None and raw.exists() else None


def best_hits(path, col, better, header):
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
    return best


def beta_ppf(q, a, b):
    """Inverse regularised incomplete beta by bisection (no scipy needed)."""
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if betainc(a, b, mid) < q:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def betainc(a, b, x):
    """Regularised incomplete beta I_x(a, b), continued fraction (NR 6.4)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbt = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
           + a * math.log(x) + b * math.log(1 - x))
    if x < (a + 1) / (a + b + 2):
        return math.exp(lbt) * _cf(a, b, x) / a
    return 1 - math.exp(lbt) * _cf(b, a, 1 - x) / b


def _cf(a, b, x):
    tiny = 1e-300
    c, d = 1.0, 1 - (a + b) * x / (a + 1)
    d = 1 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((a + m2 - 1) * (a + m2))
        d = 1 + aa * d; d = 1 / (d if abs(d) > tiny else tiny)
        c = 1 + aa / c if abs(c) > tiny else 1 + aa / tiny
        h *= d * c
        aa = -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1))
        d = 1 + aa * d; d = 1 / (d if abs(d) > tiny else tiny)
        c = 1 + aa / c if abs(c) > tiny else 1 + aa / tiny
        de = d * c
        h *= de
        if abs(de - 1) < 1e-12:
            break
    return h


def clopper_pearson(k, n, alpha=0.05):
    if n == 0:
        return (float("nan"), float("nan"))
    lo = 0.0 if k == 0 else beta_ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta_ppf(1 - alpha / 2, k + 1, n - k)
    return lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-fpr", type=float, default=0.01)
    ap.add_argument("--out", default="results/test2_nonanswerable.tsv")
    args = ap.parse_args()

    qry = {r["id"]: r for r in map(json.loads, open(S / "scope_query.jsonl"))}
    ref = {r["id"]: r for r in map(json.loads, open(S / "scope_reference.jsonl"))}
    n_ans = sum(1 for q in qry.values() if q["answerable"])
    n_non = sum(1 for q in qry.values() if not q["answerable"])

    specs = [("sequence (MMseqs2)", "results/scope_sequence.tsv", 2, "lower", False),
             ("ProstT5-Foldseek", "results/scope_structure_Cprime.tsv", 2, "lower", False),
             ("Foldseek", "results/scope_structure_C.tsv", 2, "lower", False)]
    disp = {"esm2_150m": "ESM-2 150M", "esm2_650m": "ESM-2 650M", "esm2_3b": "ESM-2 3B",
            "esmc_300m": "ESM-C 300M", "esmc_600m": "ESM-C 600M", "esmc_6b": "ESM-C 6B",
            "ankh_base": "Ankh-base", "ankh_large": "Ankh-large", "prott5": "ProtT5",
            "prostt5": "ProstT5", "glm2_650m_embed": "gLM2-embed",
            "prottucker": "ProtTucker/EAT"}
    for k, name in disp.items():
        specs.append((name, f"results/scope_knn_{k}.tsv.gz", 3, "higher", True))
        if k != "prottucker":
            # _whiten is exact ZCA (18_center.py --shrink 0), as reported in
            # the paper; the shrinkage variant is _whiten_shrunk.
            specs.append((f"{name} +whiten", f"results/scope_knn_{k}_whiten.tsv.gz",
                          3, "higher", True))

    # published delivered sensitivity, for the reproduction check
    pub = {}
    fc = BASE / "figures/data/fig3b_ceiling.csv"
    if fc.exists():
        for r in csv.DictReader(open(fc)):
            nm = r["method"].replace(" kNN", "")
            nm = {"sequence (MMseqs2)": "sequence (MMseqs2)"}.get(nm, nm)
            key = nm if r["transform"] == "none" else f"{nm} +{r['transform']}"
            pub[key] = float(r["delivered"])

    print(f"Test 2 non-answerable queries as independent negatives "
          f"(n = {n_non}; answerable n = {n_ans}); target FPR {args.target_fpr:.0%}.")
    print("Threshold fitted on answerable wrong-fold top hits, as published; then")
    print("applied unchanged to non-answerable queries. Same-fold top hits excluded.\n")
    hdr = (f"{'method':<24}{'TP@thr':>8}{'pub':>7}{'ans FPR':>9}"
           f"{'non-ans FP/n':>14}{'non-ans FPR [95% CI]':>26}{'ambig':>7}")
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for name, path, col, better, header in specs:
        best = best_hits(path, col, better, header)
        if not best:
            print(f"{name:<24}  [skip] {path} not found")
            continue
        sgn = 1.0 if better == "higher" else -1.0
        ans, non = [], []
        for qid, (s, t) in best.items():
            tag, acc = (qid.split("|", 1) if "|" in qid else ("real", qid))
            if tag != "real":
                continue
            q, tt = qry.get(acc), ref.get(t)
            if q is None or tt is None:
                continue
            sc = sgn * s
            if q["answerable"]:
                o = 1 if q["superfamily"] == tt["superfamily"] else (
                    0 if q["fold"] == tt["fold"] else -1)
                ans.append((sc, o))
            else:
                o = 0 if q["fold"] == tt["fold"] else -1
                non.append((sc, o))
        # threshold: identical rule to 24_heldout.pick_threshold / 27_figure_data
        arr = sorted(ans, key=lambda x: -x[0])
        nF = sum(1 for _, o in arr if o == -1)
        thr, fp = arr[0][0], 0
        for s, o in arr:
            if o == -1:
                fp += 1
                if fp / nF > args.target_fpr:
                    break
            thr = s
        tp = sum(1 for s, o in ans if s >= thr and o == 1)
        afp = sum(1 for s, o in ans if s >= thr and o == -1)
        neg = [s for s, o in non if o == -1]
        amb = sum(1 for _, o in non if o == 0)
        k = sum(1 for s in neg if s >= thr)
        lo, hi = clopper_pearson(k, len(neg))
        fpr = k / len(neg) if neg else float("nan")
        p = pub.get(name)
        print(f"{name:<24}{100*tp/n_ans:>7.1f}%{(f'{p:.1f}' if p is not None else '--'):>7}"
              f"{100*afp/max(nF,1):>8.2f}%{f'{k}/{len(neg)}':>14}"
              f"{100*fpr:>10.2f}% [{100*lo:5.2f},{100*hi:6.2f}]{amb:>7}")
        rows.append({"method": name, "tp_rate": round(100 * tp / n_ans, 2),
                     "published": p, "answerable_fpr": round(100 * afp / max(nF, 1), 3),
                     "nonans_fp": k, "nonans_n": len(neg),
                     "nonans_fpr": round(100 * fpr, 3), "ci_lo": round(100 * lo, 3),
                     "ci_hi": round(100 * hi, 3), "nonans_ambiguous": amb,
                     "nonans_missing": n_non - len(non)})
    out = BASE / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["method"],
                           delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(f"\n'pub' = delivered sensitivity in fig3b_ceiling.csv; TP@thr must match it.")
    print(f"Non-answerable queries with no hit, or missing from a method, are not "
          f"counted as negatives (column nonans_missing).\n-> {out}")


if __name__ == "__main__":
    main()
