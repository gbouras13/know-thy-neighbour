#!/usr/bin/env python3
"""Are the D6 negatives that pass each method's 1% threshold really negatives?

D6 "natural hard negatives" have no DETECTED clan-mate in the reference, so
some may be unannotated remote homologs, which would penalise exactly the
methods that find them. For each method, take the D6 proteins its pooled 1%
threshold accepts (the ~34 highest-scoring of 3,344, full data, no resampling)
and ask whether structure search (Foldseek) finds a significant hit for them.
Compare against the base rate over all D6.

A structural-hit rate among a method's accepted D6 negatives well above the D6
base rate means its "false positives" are partly unannotated homologs and its
Test 1 sensitivity is understated. A rate near the base rate means the accepted
negatives look like ordinary D6 proteins.

Inputs:  results/<table> as in 25_bootstrap_test1.py. The Foldseek table is
         read from structure_C_<arm>.tsv(.gz) if present, else
         structure_Cn_<arm>.tsv(.gz).
Outputs: results/d6_structural_check.tsv (per method) and
         results/d6_structural_check_proteins.tsv (per accepted protein).
Run:     python scripts/46_d6_structural_check.py [--target-fpr 0.01] [--evalue 1e-3]
"""
import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
ARMS = ("small", "mid", "control")
NEG = "D6_natural_hard_all"


def smart(path):
    path = Path(path)
    for p in (path, path.with_name(path.name + ".gz"),
              path.with_suffix("") if path.suffix == ".gz" else None):
        if p is not None and p.exists():
            return gzip.open(p, "rt") if p.suffix == ".gz" else open(p)
    return None


def best(path, col, better):
    """qid -> (best score, target). Cosine tables carry a header line."""
    fh = smart(path)
    if fh is None:
        return None
    b = {}
    with fh:
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) <= col:
                continue
            try:
                s = float(p[col])
            except ValueError:
                continue                       # header
            cur = b.get(p[0])
            if cur is None or (s > cur[0] if better == "higher" else s < cur[0]):
                b[p[0]] = (s, p[1])
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-fpr", type=float, default=0.01)
    ap.add_argument("--evalue", type=float, default=1e-3,
                    help="structure-hit significance cutoff")
    args = ap.parse_args()

    d6 = {}
    for arm in ARMS:
        p = BASE / "data/decoys" / f"{arm}_{NEG}.jsonl"
        for line in smart(p):
            r = json.loads(line)
            d6[r["acc"]] = (arm, r)
    print(f"D6 negatives: {len(d6)}")

    def merged(tmpls, col, better):
        out = {}
        for arm in ARMS:
            for t in tmpls:
                b = best(BASE / "results" / t.format(arm=arm), col, better)
                if b:
                    out.update(b)
                    break
        return out

    fs = merged(["structure_C_{arm}.tsv.gz", "structure_Cn_{arm}.tsv.gz"], 2, "lower")
    mm = merged(["sequence_{arm}_vs_joint.tsv.gz"], 2, "lower")
    if not fs:
        raise SystemExit("no Foldseek table found (structure_C_* / structure_Cn_*)")

    def fs_e(acc):
        v = fs.get(f"{NEG}|{acc}")
        return v if v else (None, None)

    covered = [a for a in d6 if f"{NEG}|{a}" in fs]
    base_hits = sum(1 for a in covered if fs[f"{NEG}|{a}"][0] < args.evalue)
    base_rate = base_hits / max(len(covered), 1)
    print(f"Foldseek reports a hit for {len(covered)} D6 proteins; "
          f"{base_hits} ({100*base_rate:.1f}%) at E < {args.evalue:g} -- the base rate.\n")

    methods = [
        ("sequence (MMseqs2)", "sequence_{arm}_vs_joint.tsv.gz", 2, "lower"),
        ("ProstT5-Foldseek", "structure_Cprime_{arm}_top50.tsv.gz", 2, "lower"),
        ("gLM2-embed", "knn_{arm}_glm2_650m_embed.tsv.gz", 3, "higher"),
        ("ProtTucker/EAT", "knn_{arm}_prottucker.tsv.gz", 3, "higher"),
        ("ProtT5 kNN", "knn_{arm}_prott5.tsv.gz", 3, "higher"),
        ("ProstT5 kNN", "knn_{arm}_prostt5.tsv.gz", 3, "higher"),
        ("Ankh-base kNN", "knn_{arm}_ankh_base.tsv.gz", 3, "higher"),
        ("Ankh-large kNN", "knn_{arm}_ankh_large.tsv.gz", 3, "higher"),
        ("ESM-2 150M kNN", "knn_{arm}_esm2_150m.tsv.gz", 3, "higher"),
        ("ESM-2 650M kNN", "knn_{arm}_esm2_650m.tsv.gz", 3, "higher"),
        ("ESM-2 3B kNN", "knn_{arm}_esm2_3b.tsv.gz", 3, "higher"),
        ("ESM-C 300M kNN", "knn_{arm}_esmc_300m.tsv.gz", 3, "higher"),
        ("ESM-C 600M kNN", "knn_{arm}_esmc_600m.tsv.gz", 3, "higher"),
        ("ESM-C 6B kNN", "knn_{arm}_esmc_6b.tsv.gz", 3, "higher"),
        ("ESM-2 650M +whiten", "knn_{arm}_esm2_650m_whiten.tsv.gz", 3, "higher"),
        ("ESM-C 6B +whiten", "knn_{arm}_esmc_6b_whiten.tsv.gz", 3, "higher"),
    ]

    hdr = (f"{'method':<22}{'accepted':>9}{'with FS hit':>12}"
           f"{'FS E<cut':>10}{'rate':>8}{'vs base':>9}{'MMseqs E<cut':>14}")
    print(hdr)
    print("-" * len(hdr))
    rows, prot_rows = [], []
    for name, tmpl, col, better in methods:
        sc = merged([tmpl], col, better)
        if not sc:
            print(f"{name:<22}  [skip] no table")
            continue
        sgn = 1.0 if better == "higher" else -1.0
        neg = sorted(((sgn * sc[f'{NEG}|{a}'][0] if f'{NEG}|{a}' in sc else float('-inf')), a)
                     for a in d6)[::-1]
        k = int(args.target_fpr * len(neg))
        thr = neg[k][0] if k < len(neg) else neg[-1][0]
        acc = [a for s, a in neg if s >= thr and s != float("-inf")]
        with_fs = [a for a in acc if fs_e(a)[0] is not None]
        sig = [a for a in with_fs if fs_e(a)[0] < args.evalue]
        mms = [a for a in acc if f"{NEG}|{a}" in mm and mm[f"{NEG}|{a}"][0] < args.evalue]
        rate = len(sig) / max(len(with_fs), 1)
        print(f"{name:<22}{len(acc):>9}{len(with_fs):>12}{len(sig):>10}"
              f"{100*rate:>7.1f}%{rate / base_rate if base_rate else float('nan'):>8.1f}x"
              f"{len(mms):>14}")
        rows.append({"method": name, "accepted": len(acc), "with_foldseek_hit": len(with_fs),
                     "foldseek_sig": len(sig), "rate": round(100 * rate, 2),
                     "base_rate": round(100 * base_rate, 2), "mmseqs_sig": len(mms)})
        for a in acc:
            e, t = fs_e(a)
            arm, r = d6[a]
            prot_rows.append({"method": name, "acc": a, "length": r.get("length"),
                              "pfam": ";".join(r.get("pfam") or []),
                              "protein": r.get("protein", ""),
                              "foldseek_E": e, "foldseek_target": t,
                              "mmseqs_E": mm.get(f"{NEG}|{a}", (None,))[0]})
    for fn, data in (("d6_structural_check.tsv", rows),
                     ("d6_structural_check_proteins.tsv", prot_rows)):
        if data:
            with open(BASE / "results" / fn, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(data[0]), delimiter="\t")
                w.writeheader()
                w.writerows(data)
    print(f"\nA rate well above the base rate means a method's accepted D6 'false "
          f"positives' are\npartly undetected homologs. -> results/d6_structural_check*.tsv")


if __name__ == "__main__":
    main()
