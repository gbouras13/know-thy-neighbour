#!/usr/bin/env python3
"""Hubness: do a few reference entries absorb a disproportionate share of kNN slots?

Computes standard hubness statistics from the embedding kNN tables, per model,
transform and query population.

Measures:

  N_k         k-occurrence: how many times each reference appears in some
              query's top-k list (Radovanovic, Nanopoulos & Ivanovic 2010).
  skew(N_k)   third standardised moment of N_k, the conventional hubness
              statistic: 0 for a uniform space, large and positive when a few
              entries are everyone's neighbour.
  top1pct     share of all kNN slots taken by the busiest 1% of references.
  coverage    distinct references retrieved at all, over the reference size.
              Low coverage is the same pathology seen from the other side.

Populations. Hubness over real queries answers "is retrieval concentrated?".
Hubness with decoys included answers "do featureless inputs pile onto the same
few references?", which is the failure mode under study (decoys retrieve
*something* confidently). Both are reported and should not be conflated.

Notes on the kNN tables:
1. The Test 1 kNN tables also contain an obsolete `D6_natural_hard`
   generation alongside the canonical `D6_natural_hard_all`. It is dropped by
   exact prefix (`D6_natural_hard|`, so `_all` survives); the number of
   skipped slots is printed per table. Test 2 (SCOPe) has only D2 shuffles.
2. ProtTucker/EAT is also a kNN method (Euclidean, top-50, same table layout
   with `score` in place of `cosine`) and is included; it has no transform
   ladder, so only its `none` rows exist. Missing tables are skipped.

Inputs:  results/knn_{small,mid,control}_<model>[_whiten].tsv.gz (Test 1),
         results/scope_knn_<model>[_whiten].tsv.gz (Test 2), and the reference
         JSONL files (for reference set sizes).
Output:  results/hubness.tsv (source data for Fig. S5).
Run:     python scripts/26_hubness.py [--out results/hubness.tsv]

Stdlib only.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from tableio import open_table

BASE = Path(__file__).resolve().parent.parent

# Obsolete natural-negative generation present in the Test 1 kNN tables. Exact prefix INCLUDING the pipe so that `D6_natural_hard_all|` is kept.
STALE = "D6_natural_hard|"


def smart(p):
    return open_table(p)                     # .tsv or .tsv.gz, else None


def k_occurrence(paths, population):
    """Counter over reference ids, the number of slots counted, and the number
    of slots dropped because they belong to the obsolete D6 class."""
    c, n, stale = Counter(), 0, 0
    for path in paths:
        fh = smart(path)
        if fh is None:
            continue
        with fh:
            fh.readline()                      # kNN tables carry a header
            for line in fh:
                p = line.rstrip("\n").split("\t")
                if len(p) < 2:
                    continue
                if p[0].startswith(STALE):
                    stale += 1
                    continue
                is_real = p[0].startswith("real|")
                if population == "real" and not is_real:
                    continue
                if population == "decoy" and is_real:
                    continue
                c[p[1]] += 1
                n += 1
    return c, n, stale


def stats(c, n, n_ref):
    """skew(N_k), top-1% absorption, coverage. N_k includes the zeros."""
    if not n:
        return None
    # Every reference has an N_k, including the many that are never retrieved.
    counts = list(c.values()) + [0] * max(0, n_ref - len(c))
    m = sum(counts) / len(counts)
    var = sum((x - m) ** 2 for x in counts) / len(counts)
    sd = var ** 0.5
    skew = (sum((x - m) ** 3 for x in counts) / len(counts) / sd ** 3
            if sd > 0 else 0.0)
    top = max(1, n_ref // 100)
    absorbed = sum(v for _, v in c.most_common(top))
    return {"slots": n, "distinct": len(c), "coverage": len(c) / n_ref,
            "skew_Nk": skew, "top1pct_share": absorbed / n,
            "max_Nk": max(counts) if counts else 0, "mean_Nk": m}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/hubness.tsv")
    args = ap.parse_args()

    models = ["esm2_150m", "esm2_650m", "esm2_3b", "esmc_300m", "esmc_600m",
              "esmc_6b", "ankh_base", "ankh_large", "prott5", "prostt5",
              "glm2_650m_embed", "prottucker"]
    transforms = ["", "_whiten"]

    n_ref_t1 = sum(1 for _ in open(BASE / "data/reference/joint_reference.jsonl"))
    n_ref_t2 = sum(1 for _ in open(BASE / "data/scope/scope_reference.jsonl"))

    rows = []
    for m in models:
        for t in transforms:
            # Test 1: three arms pooled, and the small arm alone.
            for label, paths, n_ref in [
                ("test1_pooled",
                 [BASE / f"results/knn_{a}_{m}{t}.tsv.gz"
                  for a in ("small", "mid", "control")], n_ref_t1),
                ("test1_small",
                 [BASE / f"results/knn_small_{m}{t}.tsv.gz"], n_ref_t1),
                ("test2_scope",
                 [BASE / f"results/scope_knn_{m}{t}.tsv.gz"], n_ref_t2),
            ]:
                for pop in ("real", "decoy", "all"):
                    c, n, stale = k_occurrence(paths, pop)
                    st = stats(c, n, n_ref)
                    if st and stale and pop == "all":
                        print(f"[hubness] {m}{t} {label}: dropped {stale} "
                              f"slots of the obsolete {STALE[:-1]} class")
                    if st:
                        rows.append({"model": m,
                                     "transform": t.lstrip("_") or "none",
                                     "benchmark": label, "population": pop,
                                     "n_ref": n_ref, **st})

    out = BASE / args.out
    cols = ["model", "transform", "benchmark", "population", "n_ref", "slots",
            "distinct", "coverage", "skew_Nk", "top1pct_share", "max_Nk",
            "mean_Nk"]
    with out.open("w") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join(
                f"{r[c]:.6g}" if isinstance(r[c], float) else str(r[c])
                for c in cols) + "\n")
    print(f"[hubness] {len(rows)} rows -> {out.relative_to(BASE)}")

    # Console summary: top-1% share for the strongest and weakest models.
    print("\nTop-1% share of kNN slots, ESM-C 6B vs gLM2-embed "
          "(no transform),\nper benchmark and population. "
          "Ratio = larger / smaller:\n")
    print(f"{'benchmark':<14}{'population':<11}{'ESM-C 6B':>12}{'gLM2-embed':>13}"
          f"{'ratio':>9}")
    print("-" * 59)
    for bench in ("test1_small", "test1_pooled", "test2_scope"):
        for pop in ("real", "decoy", "all"):
            a = next((r for r in rows if r["model"] == "esmc_6b"
                      and r["transform"] == "none" and r["benchmark"] == bench
                      and r["population"] == pop), None)
            b = next((r for r in rows if r["model"] == "glm2_650m_embed"
                      and r["transform"] == "none" and r["benchmark"] == bench
                      and r["population"] == pop), None)
            if a and b:
                ra = 100 * a["top1pct_share"]
                rb = 100 * b["top1pct_share"]
                print(f"{bench:<14}{pop:<11}{ra:>11.1f}%{rb:>12.1f}%"
                      f"{max(ra, rb) / max(min(ra, rb), 1e-9):>8.1f}x")
    print("\nFull table written to the output TSV.")


if __name__ == "__main__":
    main()
