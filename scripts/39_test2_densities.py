#!/usr/bin/env python3
"""Test 2 score densities for all methods (input to Fig. S3).

Test 2's natural negative is not an input class but an OUTCOME: a real
answerable query whose top-1 hit landed in a different fold. Each method's
panel therefore shows three populations of best scores:

  top-1 correct     answerable query, top-1 same superfamily
  top-1 wrong fold  answerable query, top-1 in a different fold (the population
                    Test 2's 1% FPR is measured on)
  shuffled query    the D2 shuffles (5 seeds, 20,345 queries), no homolog by
                    construction
  (same fold / different superfamily is ambiguous and dropped, as in scoring)

Foldseek on real structures has no shuffle rows (a shuffled sequence has no
coordinates), so its panel carries the two real classes. ProstT5-Foldseek
shuffles are read from a separate file and merged.

Output: figures/data/fig12_densities_test2.csv (one row per query per method).

  python3 scripts/39_test2_densities.py
"""
import csv
import gzip
import json
import math
import statistics as st
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

# (display, file, score col, better, family, scale)
SPECS = [
    ("sequence (MMseqs2)", "results/scope_sequence.tsv", 2, "lower", "sequence", "E-value"),
    ("Foldseek", "results/scope_structure_C.tsv.gz", 2, "lower", "structure", "E-value"),
    # ProstT5-Foldseek shuffles are in a separate file (3Di predicted for the
    # shuffles in a separate run with the same phold + foldseek recipe).
    # Merged here so the panel shows all three classes from one method row.
    ("ProstT5-Foldseek", ["results/scope_structure_Cprime.tsv.gz",
                          "results/scope_structure_Cprime_shuffles.tsv.gz"],
     2, "lower", "structure", "E-value"),
    ("ProtTucker/EAT", "results/scope_knn_prottucker.tsv.gz", 3, "higher", "embedding", "neg. Euclidean"),
    ("gLM2-embed", "results/scope_knn_glm2_650m_embed.tsv.gz", 3, "higher", "embedding", "cosine"),
    ("ProstT5 kNN", "results/scope_knn_prostt5.tsv.gz", 3, "higher", "embedding", "cosine"),
    ("ProtT5 kNN", "results/scope_knn_prott5.tsv.gz", 3, "higher", "embedding", "cosine"),
    ("Ankh-base kNN", "results/scope_knn_ankh_base.tsv.gz", 3, "higher", "embedding", "cosine"),
    ("Ankh-large kNN", "results/scope_knn_ankh_large.tsv.gz", 3, "higher", "embedding", "cosine"),
    ("ESM-2 150M kNN", "results/scope_knn_esm2_150m.tsv.gz", 3, "higher", "embedding", "cosine"),
    ("ESM-2 650M kNN", "results/scope_knn_esm2_650m.tsv.gz", 3, "higher", "embedding", "cosine"),
    ("ESM-2 3B kNN", "results/scope_knn_esm2_3b.tsv.gz", 3, "higher", "embedding", "cosine"),
    ("ESM-C 300M kNN", "results/scope_knn_esmc_300m.tsv.gz", 3, "higher", "embedding", "cosine"),
    ("ESM-C 600M kNN", "results/scope_knn_esmc_600m.tsv.gz", 3, "higher", "embedding", "cosine"),
    ("ESM-C 6B kNN", "results/scope_knn_esmc_6b.tsv.gz", 3, "higher", "embedding", "cosine"),
]


def labels():
    qry, ref = {}, {}
    for f, d in (("scope_query.jsonl", qry), ("scope_reference.jsonl", ref)):
        for r in map(json.loads, open(BASE / "data/scope" / f)):
            r["fold"] = r["sccs"].rsplit(".", 2)[0]
            d[r["id"]] = r
    return qry, ref


def best_hits(path, col, better):
    op = gzip.open if str(path).endswith(".gz") else open
    best = {}
    with op(path, "rt") as fh:
        first = fh.readline()
        if not first.startswith(("real|", "D2_")):
            pass  # header consumed
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
            q = p[0]
            cur = best.get(q)
            if cur is None or (s > cur[0] if better == "higher" else s < cur[0]):
                best[q] = (s, p[1])
    return best


def main():
    qry, ref = labels()
    rows = []
    for name, path, col, better, fam, scale in SPECS:
        paths = [BASE / x for x in ([path] if isinstance(path, str) else path)]
        missing = [q for q in paths if not q.exists()]
        if missing == paths:
            print(f"  [skip] {missing[0].name}")
            continue
        for q in missing:
            print(f"  [warn] {q.name} absent — that class will be empty")
        paths = [q for q in paths if q.exists()]
        best = {}
        for one in paths:
            best.update(best_hits(one, col, better))
        cls_scores = {"top-1 correct": [], "top-1 wrong fold": [], "shuffled query": []}
        for q, (s, t) in best.items():
            if q.startswith("D2_shuffle"):
                cls_scores["shuffled query"].append(s)
                continue
            acc = q.split("|", 1)[1] if "|" in q else q
            qr, tr = qry.get(acc), ref.get(t)
            if not qr or not tr or not qr.get("answerable"):
                continue
            if qr["superfamily"] == tr["superfamily"]:
                cls_scores["top-1 correct"].append(s)
            elif qr["fold"] != tr["fold"]:
                cls_scores["top-1 wrong fold"].append(s)
        for cls, v in cls_scores.items():
            for s in v:
                rows.append({"method": name, "family": fam, "scale": scale,
                             "class": cls, "score": s})
        meds = {c: (st.median(v) if v else None) for c, v in cls_scores.items()}
        disp = {c: ("—" if m is None
                    else f"{-math.log10(max(m,1e-200)):.1f}" if scale == "E-value"
                    else f"{m:.4f}")
                for c, m in meds.items()}
        print(f"  {name:<18}"
              f"correct(md) {disp['top-1 correct']:>8}  "
              f"wrongfold {disp['top-1 wrong fold']:>8}  "
              f"shuffle {disp['shuffled query']:>8}   "
              f"n={len(cls_scores['top-1 correct'])}/{len(cls_scores['top-1 wrong fold'])}/{len(cls_scores['shuffled query'])}")

    out = BASE / "figures/data/fig12_densities_test2.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["method", "family", "scale", "class", "score"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n  wrote {out.relative_to(BASE)} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
