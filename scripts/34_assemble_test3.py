#!/usr/bin/env python3
"""Assemble figures/data/fig10_test3.csv (Fig. S10) from the scored Test 3 arms.

Reads every `results/test3_common_*.csv` (embedding arms from
30_test3_allvall.py, alignment arms from 31_test3_score_tsv.py, all scored with
the same --restrict allowlist), maps model ids to display names and arm labels,
checks that every arm was scored on the same number of queries, and writes one
long-form CSV (method, arm, level, mean_sens).

The MMseqs2 arm is an all-vs-all search (mmseqs 18.8cc5c, -s 7.5
--max-seqs 4000 -e 10000 over all 15,176 domains) rescored through
31_test3_score_tsv.py into `results/test3_common_mmseqs.csv`.

Run:  python3 scripts/34_assemble_test3.py [--out figures/data/fig10_test3.csv]
"""
import argparse
import csv
import glob
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

# Scored-arm model id -> (display name, arm label used for facetting).
# ":whiten" is how 30_test3_allvall.py names a transformed variant.
NAMES = {
    "esm2_150m": "ESM-2 150M", "esm2_650m": "ESM-2 650M", "esm2_3b": "ESM-2 3B",
    "esmc_300m": "ESM-C 300M", "esmc_600m": "ESM-C 600M", "esmc_6b": "ESM-C 6B",
    "ankh_base": "Ankh-base", "ankh_large": "Ankh-large",
    "prott5": "ProtT5", "prostt5": "ProstT5", "glm2_650m_embed": "gLM2-embed",
    "prottucker": "ProtTucker/EAT",
    "Foldseek C (real struct)": "Foldseek",
    "Foldseek C-prime (ProstT5 3Di)": "ProstT5-Foldseek",
    "sequence (MMseqs2)": "sequence (MMseqs2)",
}


def arm_of(model, display):
    if "Foldseek" in display:
        return "structure"
    if "MMseqs2" in display:
        return "sequence"
    if display == "ProtTucker/EAT":
        return "embedding"
    return "embedding +wh" if ":whiten" in model else "embedding"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures/data/fig10_test3.csv")
    args = ap.parse_args()

    rows, seen = [], set()
    for f in sorted(glob.glob(str(BASE / "results/test3_common_*.csv"))):
        for r in csv.DictReader(open(f)):
            model, lvl = r["model"], r["level"]
            base = model.split(":")[0]
            if base not in NAMES:
                print(f"  [warn] no display name for {model!r} ({Path(f).name})")
                continue
            disp = NAMES[base] + (" +whiten" if ":whiten" in model else "")
            if (disp, lvl) in seen:
                continue
            seen.add((disp, lvl))
            rows.append({"method": disp, "arm": arm_of(model, disp),
                         "level": lvl, "mean_sens": r["mean_sens"]})

    # All arms must share one query set, or they are not comparable.
    nq = {r["n_queries"] for f in glob.glob(str(BASE / "results/test3_common_*.csv"))
          for r in csv.DictReader(open(f))}
    if len(nq) != 1:
        raise SystemExit(f"arms scored on DIFFERENT query counts: {nq}")

    order = {"family": 0, "superfamily": 1, "fold": 2}
    rows.sort(key=lambda r: (r["method"], order.get(r["level"], 9)))
    out = BASE / args.out
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["method", "arm", "level", "mean_sens"])
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {out}  ({len(rows)} rows, "
          f"{len(set(r['method'] for r in rows))} methods, n={nq.pop()} queries)")


if __name__ == "__main__":
    main()
