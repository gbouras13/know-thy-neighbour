#!/usr/bin/env python3
"""Test 2 (SCOPe/ASTRAL-40 family hold-out): cross-method comparison at
matched false-positive rate.

A fixed cosine cutoff is a different operating point for every model (ESM-2
cosines sit near 0.98, gLM2's near 0.77), so methods are compared either
threshold-free (AUROC) or at a per-method threshold matched on FPR.

Positives: top hit shares the query's SCOPe superfamily.
Negatives: top hit is in a different FOLD (non-homologous by structure).
Same-fold/different-superfamily hits are excluded from both, since SCOPe folds
can be analogous rather than homologous.

Inputs:  data/scope/scope_{query,reference}.jsonl, results/scope_*.tsv[.gz].
Outputs: printed table of TPR at target FPR, AUROC, threshold and ambiguous
         fraction per method. Missing result files are shown as [pending].
Run:     python3 scripts/15_scope_compare.py [--target-fpr 0.01]
         [--fpr-denominator fold|answerable]
"""
import argparse, gzip, json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
S = BASE / "data/scope"


def smart_open(p):
    p = Path(p)
    return gzip.open(p, "rt") if p.suffix == ".gz" else open(p)


def best_hits(path, col, better, header):
    best = {}
    with smart_open(path) as fh:
        if header: fh.readline()
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) <= col: continue
            try: s = float(p[col])
            except ValueError: continue
            cur = best.get(p[0])
            if cur is None or (s > cur[0] if better == "higher" else s < cur[0]):
                best[p[0]] = (s, p[1])
    return best


def curve(best, qry, ref, better):
    """-> sorted list of (score, is_TP, is_FPfold) for real answerable queries."""
    pts = []
    for qid, (s, t) in best.items():
        tag, acc = qid.split("|", 1)
        if tag != "real": continue
        q, tt = qry.get(acc), ref.get(t)
        if q is None or tt is None or not q["answerable"]: continue
        if q["superfamily"] == tt["superfamily"]: pts.append((s, 1, 0))
        elif q["fold"] == tt["fold"]: pts.append((s, 0, 0))      # ambiguous
        else: pts.append((s, 0, 1))
    pts.sort(key=lambda x: -x[0] if better == "higher" else x[0])
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-fpr", type=float, default=0.01)
    ap.add_argument("--fpr-denominator", default="fold",
                    choices=["fold", "answerable"],
                    help="'fold' (default): FPR over queries whose top hit is in "
                         "a different fold -- a CONDITIONAL rate whose denominator "
                         "shrinks as a method improves, so a better method is held "
                         "to a stricter absolute budget. 'answerable': FPR over all "
                         "answerable queries, fixed at 3,503 for every method, at "
                         "the cost of mixing same-fold ambiguous hits into the "
                         "negatives.")
    args = ap.parse_args()

    qry = {r["id"]: r for r in map(json.loads, open(S / "scope_query.jsonl"))}
    ref = {r["id"]: r for r in map(json.loads, open(S / "scope_reference.jsonl"))}
    n_ans = sum(1 for q in qry.values() if q["answerable"])

    methods = [("sequence (MMseqs2)", "results/scope_sequence.tsv", 2, "lower", False)]
    for m in ["esm2_150m", "esm2_650m", "ankh_base", "ankh_large", "prott5", "prostt5",
              "glm2_650m_embed", "esmc_300m", "esmc_600m", "esmc_6b", "prottucker"]:
        methods.append((f"{m} kNN", f"results/scope_knn_{m}.tsv.gz", 3, "higher", True))
    # Anisotropy transforms, same models, same everything else.
    for m in ["esm2_150m", "esm2_650m", "esmc_300m", "esmc_600m", "esmc_6b",
              "ankh_base", "prott5", "prostt5", "glm2_650m_embed"]:
        for t in ["center", "diag", "abtt", "whiten"]:
            methods.append((f"{m} +{t}", f"results/scope_knn_{m}_{t}.tsv.gz",
                            3, "higher", True))
    # Structure arms. C = Foldseek on real ASTRAL structures; C' = Foldseek on
    # ProstT5-predicted 3Di ("ProstT5-Foldseek"). The Test 2 negative class
    # ("top hit in a different fold") is defined for real queries, so no
    # synthetic decoys are needed. A query whose domain lacks coordinates
    # scores as a miss against the same n_ans denominator.
    # C' uses sequence input only, with 3Di predicted on both sides -- the
    # same input the pLM kNN methods get.
    methods.append(("Foldseek C' (ProstT5)", "results/scope_structure_Cprime.tsv",
                    2, "lower", False))
    # C is an upper bound rather than a competitor: SCOPe labels were
    # assigned from these same coordinates.
    methods.append(("Foldseek C (real struct)", "results/scope_structure_C.tsv",
                    2, "lower", False))

    print(f"SCOPe/ASTRAL-40 -- structure-derived labels, {n_ans} answerable queries")
    dlabel = ("different-fold" if args.fpr_denominator == "fold"
              else "all-answerable")
    print(f"Threshold set per method to {args.target_fpr:.0%} {dlabel} FPR.")
    print(f"FPR denominator: {args.fpr_denominator}"
          + ("  (method-dependent -- shrinks as a method improves)"
             if args.fpr_denominator == "fold"
             else "  (fixed at every scored query, for every method)") + "\n")
    print(f"{'method':<24}{'TP@1%FPR':>10}{'AUROC':>9}{'thr':>12}{'ambig':>8}")
    print("-" * 63)
    for name, path, col, better, header in methods:
        p = BASE / path
        if not p.exists():
            p = p.with_name(p.name + ".gz")
            if not p.exists():
                print(f"{name:<24}{'[pending]':>10}"); continue
        pts = curve(best_hits(p, col, better, header), qry, ref, better)
        if not pts:
            print(f"{name:<24}{'no data':>10}"); continue

        # sweep: walk down the ranked list accumulating TP and FP-fold
        tp = fp = 0
        nP = sum(x[1] for x in pts)
        # See --fpr-denominator. 'fold' counts only different-fold top hits;
        # 'answerable' fixes the denominator at every scored query so it cannot
        # move with the method.
        nF = (sum(x[2] for x in pts) if args.fpr_denominator == "fold"
              else len(pts))
        best_tp, thr = 0.0, None
        roc = []
        for s, isTP, isFP in pts:
            tp += isTP; fp += isFP
            roc.append((fp / max(nF, 1), tp / max(nP, 1)))
            if nF and fp / nF <= args.target_fpr:
                best_tp, thr = tp / n_ans, s
        # trapezoidal AUROC over the swept curve
        auc = 0.0; px = py = 0.0
        for x, y in roc:
            auc += (x - px) * (y + py) / 2; px, py = x, y
        auc += (1 - px) * (1 + py) / 2
        amb = sum(1 for x in pts if x[1] == 0 and x[2] == 0) / len(pts)
        print(f"{name:<24}{100*best_tp:>9.1f}%{auc:>9.3f}{thr if thr is None else round(thr,4):>12}"
              f"{100*amb:>7.1f}%")


if __name__ == "__main__":
    main()
