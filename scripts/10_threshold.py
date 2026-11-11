#!/usr/bin/env python3
"""Test 1: does a single fixed score threshold hold its FPR across query length?

An embedding kNN search has no null model, so a user must choose one score
threshold without knowing the query's length. This script asks whether that
threshold's realised false-positive rate moves with length.

For each method the threshold is fitted to a target FPR on the negative class
(by default pooled over every length bin; optionally on a single bin via
--calib-bin), then held fixed and applied to every length bin. Per-bin FPR is
reported with 95% bootstrap CIs, since several short bins have few queries.

Inputs:  results/<method>_{small,mid,control}*.tsv[.gz] search outputs,
         data/queries/joint_<arm>_query.jsonl, data/decoys/<arm>_<negative>.fasta
Outputs: printed per-bin FPR table; results/threshold_ladder_pooled.csv
         (method, threshold, TPR at calibration).
Run:     python3 scripts/10_threshold.py [--negative D6_natural_hard_all]
         [--target-fpr 0.01] [--calib-bin pooled]

Stdlib only.
"""

import argparse
import gzip
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
BINS = ["1-30", "31-50", "51-75", "76-99", "100-130", "131-165",
        "166-199", "200-250", "251-320", "321-400"]
ARMS = {"small": BINS[:4], "mid": BINS[4:7], "control": BINS[7:]}


def bin_of(L):
    for b in BINS:
        lo, hi = (int(x) for x in b.split("-"))
        if lo <= L <= hi:
            return b
    return None


def best_scores(path, col, better):
    b = {}
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as fh:
        fh.readline() if better == "higher" else None
        for line in fh:
            p = line.rstrip("\n").split("\t")
            try:
                s = float(p[col])
            except (ValueError, IndexError):
                continue
            cur = b.get(p[0])
            if cur is None or (s > cur if better == "higher" else s < cur):
                b[p[0]] = s
    return b


def collect(method_file, col, better, negative):
    """-> {bin: (list_of_positive_scores, n_pos, list_of_neg_scores, n_neg)}"""
    out = {}
    for arm, arm_bins in ARMS.items():
        path = BASE / "results" / method_file.format(arm=arm)
        if not path.exists():
            # Some tables (e.g. the embedding transforms) may be stored
            # un-gzipped; fall back to the uncompressed path.
            raw = path.with_suffix("")
            if raw.exists():
                path = raw
            else:
                continue
        scores = best_scores(path, col, better)
        q = {}
        for line in open(BASE / "data/queries" / f"joint_{arm}_query.jsonl"):
            r = json.loads(line)
            q[r["acc"]] = r

        pos, neg = defaultdict(list), defaultdict(list)
        npos, nneg = defaultdict(int), defaultdict(int)
        for acc, r in q.items():
            if r.get("answerable") and r.get("remote"):
                b = bin_of(r["length"])
                if b:
                    npos[b] += 1
                    if f"real|{acc}" in scores:
                        pos[b].append(scores[f"real|{acc}"])

        negfa = BASE / "data/decoys" / f"{arm}_{negative}.fasta"
        if negfa.exists():
            cid, L = None, 0
            for line in open(negfa):
                if line.startswith(">"):
                    if cid:
                        b = bin_of(L)
                        if b:
                            nneg[b] += 1
                            k = f"{negative}|{cid}"
                            if k in scores:
                                neg[b].append(scores[k])
                    cid, L = line[1:].split()[0], 0
                else:
                    L += len(line.strip())
            if cid:
                b = bin_of(L)
                if b:
                    nneg[b] += 1
                    k = f"{negative}|{cid}"
                    if k in scores:
                        neg[b].append(scores[k])

        for b in arm_bins:
            out[b] = (pos[b], npos[b], neg[b], nneg[b])
    return out


def rate(scores, n, thr, better):
    if not n:
        return None
    hit = sum(1 for s in scores if (s >= thr if better == "higher" else s < thr))
    return hit / n


def boot_ci(scores, n, thr, better, iters=400, seed=0):
    """Bootstrap CI over the full issued set (misses included as non-hits)."""
    if not n:
        return (None, None)
    rng = random.Random(seed)
    flags = [1 if (s >= thr if better == "higher" else s < thr) else 0
             for s in scores] + [0] * (n - len(scores))
    if not flags:
        return (None, None)
    vals = []
    for _ in range(iters):
        vals.append(sum(rng.choices(flags, k=len(flags))) / len(flags))
    vals.sort()
    return (vals[int(0.025 * iters)], vals[int(0.975 * iters)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--negative", default="D6_natural_hard_all")
    ap.add_argument("--target-fpr", type=float, default=0.01)
    ap.add_argument("--calib-bin", default="pooled",
                    help='"pooled" (default) fits the threshold on every query '
                         "at once, as in 25_bootstrap_test1.py; a bin name "
                         "(e.g. 321-400) fits it on that length bin only.")
    args = ap.parse_args()

    # Missing result files are skipped.
    methods = [
        ("sequence (MMseqs2)",  "sequence_{arm}_vs_joint.tsv.gz",    2, "lower"),
        ("ESM-2 150M kNN",      "knn_{arm}_esm2_150m.tsv.gz",        3, "higher"),
        ("ESM-2 650M kNN",      "knn_{arm}_esm2_650m.tsv.gz",        3, "higher"),
        ("ESM-2 3B kNN",        "knn_{arm}_esm2_3b.tsv.gz",          3, "higher"),
        ("ProstT5 kNN",         "knn_{arm}_prostt5.tsv.gz",          3, "higher"),
        ("Ankh-base kNN",       "knn_{arm}_ankh_base.tsv.gz",        3, "higher"),
        ("Ankh-large kNN",      "knn_{arm}_ankh_large.tsv.gz",       3, "higher"),
        ("ProtT5 kNN",          "knn_{arm}_prott5.tsv.gz",           3, "higher"),
        ("ESM-C 300M kNN",      "knn_{arm}_esmc_300m.tsv.gz",        3, "higher"),
        ("ESM-C 600M kNN",      "knn_{arm}_esmc_600m.tsv.gz",        3, "higher"),
        ("ESM-C 6B kNN",        "knn_{arm}_esmc_6b.tsv.gz",          3, "higher"),
        ("gLM2-embed",          "knn_{arm}_glm2_650m_embed.tsv.gz",  3, "higher"),
        ("ESM-2 650M +center", "knn_{arm}_esm2_650m_center.tsv.gz", 3, "higher"),
        ("ESM-2 650M +diag", "knn_{arm}_esm2_650m_diag.tsv.gz", 3, "higher"),
        ("ESM-2 650M +abtt", "knn_{arm}_esm2_650m_abtt.tsv.gz", 3, "higher"),
        ("ESM-2 650M +whiten", "knn_{arm}_esm2_650m_whiten.tsv.gz", 3, "higher"),
        ("ESM-C 6B +center", "knn_{arm}_esmc_6b_center.tsv.gz", 3, "higher"),
        ("ESM-C 6B +diag", "knn_{arm}_esmc_6b_diag.tsv.gz", 3, "higher"),
        ("ESM-C 6B +abtt", "knn_{arm}_esmc_6b_abtt.tsv.gz", 3, "higher"),
        ("ESM-C 6B +whiten", "knn_{arm}_esmc_6b_whiten.tsv.gz", 3, "higher"),
        ("ProtT5 +center", "knn_{arm}_prott5_center.tsv.gz", 3, "higher"),
        ("ProtT5 +diag", "knn_{arm}_prott5_diag.tsv.gz", 3, "higher"),
        ("ProtT5 +abtt", "knn_{arm}_prott5_abtt.tsv.gz", 3, "higher"),
        ("ProtT5 +whiten", "knn_{arm}_prott5_whiten.tsv.gz", 3, "higher"),
        ("gLM2-embed +center", "knn_{arm}_glm2_650m_embed_center.tsv.gz", 3, "higher"),
        ("gLM2-embed +diag", "knn_{arm}_glm2_650m_embed_diag.tsv.gz", 3, "higher"),
        ("gLM2-embed +abtt", "knn_{arm}_glm2_650m_embed_abtt.tsv.gz", 3, "higher"),
        ("gLM2-embed +whiten", "knn_{arm}_glm2_650m_embed_whiten.tsv.gz", 3, "higher"),
        ("ESM-C 300M +center", "knn_{arm}_esmc_300m_center.tsv.gz", 3, "higher"),
        ("ESM-C 300M +diag", "knn_{arm}_esmc_300m_diag.tsv.gz", 3, "higher"),
        ("ESM-C 300M +abtt", "knn_{arm}_esmc_300m_abtt.tsv.gz", 3, "higher"),
        ("ESM-C 300M +whiten", "knn_{arm}_esmc_300m_whiten.tsv.gz", 3, "higher"),
        ("ESM-C 600M +center", "knn_{arm}_esmc_600m_center.tsv.gz", 3, "higher"),
        ("ESM-C 600M +diag", "knn_{arm}_esmc_600m_diag.tsv.gz", 3, "higher"),
        ("ESM-C 600M +abtt", "knn_{arm}_esmc_600m_abtt.tsv.gz", 3, "higher"),
        ("ESM-C 600M +whiten", "knn_{arm}_esmc_600m_whiten.tsv.gz", 3, "higher"),
        ("ESM-2 150M +center", "knn_{arm}_esm2_150m_center.tsv.gz", 3, "higher"),
        ("ESM-2 150M +diag", "knn_{arm}_esm2_150m_diag.tsv.gz", 3, "higher"),
        ("ESM-2 150M +abtt", "knn_{arm}_esm2_150m_abtt.tsv.gz", 3, "higher"),
        ("ESM-2 150M +whiten", "knn_{arm}_esm2_150m_whiten.tsv.gz", 3, "higher"),
        ("ESM-2 3B +center", "knn_{arm}_esm2_3b_center.tsv.gz", 3, "higher"),
        ("ESM-2 3B +diag", "knn_{arm}_esm2_3b_diag.tsv.gz", 3, "higher"),
        ("ESM-2 3B +abtt", "knn_{arm}_esm2_3b_abtt.tsv.gz", 3, "higher"),
        ("ESM-2 3B +whiten", "knn_{arm}_esm2_3b_whiten.tsv.gz", 3, "higher"),
        ("Ankh-base +center", "knn_{arm}_ankh_base_center.tsv.gz", 3, "higher"),
        ("Ankh-base +diag", "knn_{arm}_ankh_base_diag.tsv.gz", 3, "higher"),
        ("Ankh-base +abtt", "knn_{arm}_ankh_base_abtt.tsv.gz", 3, "higher"),
        ("Ankh-base +whiten", "knn_{arm}_ankh_base_whiten.tsv.gz", 3, "higher"),
        ("Ankh-large +center", "knn_{arm}_ankh_large_center.tsv.gz", 3, "higher"),
        ("Ankh-large +diag", "knn_{arm}_ankh_large_diag.tsv.gz", 3, "higher"),
        ("Ankh-large +abtt", "knn_{arm}_ankh_large_abtt.tsv.gz", 3, "higher"),
        ("Ankh-large +whiten", "knn_{arm}_ankh_large_whiten.tsv.gz", 3, "higher"),
        ("ProstT5 +center", "knn_{arm}_prostt5_center.tsv.gz", 3, "higher"),
        ("ProstT5 +diag", "knn_{arm}_prostt5_diag.tsv.gz", 3, "higher"),
        ("ProstT5 +abtt", "knn_{arm}_prostt5_abtt.tsv.gz", 3, "higher"),
        ("ProstT5 +whiten", "knn_{arm}_prostt5_whiten.tsv.gz", 3, "higher"),
        ("ProtTucker/EAT",      "knn_{arm}_prottucker.tsv.gz",       3, "higher"),
        ("ProstT5-Foldseek",         "structure_Cprime_{arm}_top50.tsv.gz", 2, "lower"),
        ("Foldseek",          "structure_Cn_{arm}.tsv.gz",         2, "lower"),
    ]
    calib_bin = args.calib_bin
    where = ("every query pooled" if calib_bin == "pooled" else f"{calib_bin}")
    print(f"Threshold calibrated to {args.target_fpr:.0%} FPR on {where}, "
          f"then held FIXED across every length bin.")
    print(f"Negative class: {args.negative}. Cells are FPR% [95% bootstrap CI].\n")

    csv_rows = []
    for name, tmpl, col, better in methods:
        data = collect(tmpl, col, better, args.negative)
        if calib_bin == "pooled":
            pc  = [x for b in BINS for x in data.get(b, ([], 0, [], 0))[0]]
            npc = sum(data.get(b, ([], 0, [], 0))[1] for b in BINS)
            nc  = [x for b in BINS for x in data.get(b, ([], 0, [], 0))[2]]
            nnc = sum(data.get(b, ([], 0, [], 0))[3] for b in BINS)
        else:
            pc, npc, nc, nnc = data.get(calib_bin, ([], 0, [], 0))
        if not nnc:
            print(f"{name:<20} no data in calibration bin")
            continue

        # threshold achieving target FPR on the calibration bin
        ordered = sorted(nc, reverse=(better == "higher"))
        k = int(args.target_fpr * nnc)
        thr = ordered[k] if k < len(ordered) else (
            ordered[-1] if ordered else (0.0 if better == "higher" else 1e-300))

        tpr_c = rate(pc, npc, thr, better)
        # One decimal place: low-TPR methods would otherwise all print "0%".
        print(f"{name:<20} threshold={thr:.6g}  "
              f"(TPR at calibration = {100*(tpr_c or 0):.1f}%)")
        csv_rows.append({"method": name, "threshold": f"{thr:.6g}",
                         "tpr": round(100 * (tpr_c or 0), 3)})
        cells = []
        for b in BINS:
            p, np_, n, nn = data.get(b, ([], 0, [], 0))
            f = rate(n, nn, thr, better)
            if f is None:
                cells.append(f"    {b:>8}:      -")
                continue
            lo, hi = boot_ci(n, nn, thr, better)
            cells.append(f"    {b:>8}: {100*f:6.2f}%  "
                         f"[{100*lo:5.2f},{100*hi:5.2f}]  n={nn}")
        print("\n".join(cells))
        print()
    _write_csv(csv_rows)


def _write_csv(rows):
    """Write the per-method threshold and calibration TPR to CSV for plotting."""
    import csv as _csv
    out = BASE / "results/threshold_ladder_pooled.csv"
    with out.open("w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=["method", "threshold", "tpr"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n-> {out.relative_to(BASE)}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
