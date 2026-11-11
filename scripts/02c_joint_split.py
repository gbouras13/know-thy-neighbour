#!/usr/bin/env python3
"""Build the Test 1 joint query/reference split across all three length arms.

All sequences from the small, mid and control arms are clustered together in
one MMseqs2 run (30% identity, 80% coverage), and whole clusters are assigned
to the query or reference side. Clustering the arms jointly means a short
protein and a longer homolog in another arm fall in the same cluster and
cannot end up on opposite sides of the split.

Clustering alone does not guarantee separation: a short protein contained as a
domain in a longer one can escape bidirectional coverage clustering while still
aligning at high identity. 02d_measure_separation.py therefore measures each
query's maximum identity to the reference and stores it, so remoteness is a
measured per-query property and results can be stratified by it.

Input:  data/{queries,reference}/{arm}_{query,reference}.jsonl from
        02_build_splits.py, for all three arms.
Output: data/queries/joint_{arm}_query.{fasta,jsonl},
        data/reference/joint_reference.{fasta,jsonl},
        results/joint_split_manifest.json

Usage: python scripts/02c_joint_split.py [--threads N]

Standard library only.
"""

import argparse
import json
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
MMSEQS = Path.home() / "miniforge3/envs/mmseqs2/bin/mmseqs"
ARMS = ["small", "mid", "control"]


def read_jsonl(p):
    with open(p) as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query-frac", type=float, default=0.20)
    ap.add_argument("--min-seq-id", type=float, default=0.3)
    ap.add_argument("--coverage", type=float, default=0.8)
    # --cov-mode 5 is a length-ratio condition (shorter >= c of the longer),
    # which prevents domain containment from clustering, so it is not used.
    # No cov-mode reliably catches containment; the standard bidirectional
    # mode 0 is used and remoteness is measured downstream in 02d.
    ap.add_argument("--cov-mode", type=int, default=0,
                    help="0 = bidirectional coverage (standard)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    interim = BASE / "data/interim"
    interim.mkdir(parents=True, exist_ok=True)

    # ---- gather every sequence from both arms -----------------------------
    entries, per_arm = {}, Counter()
    for arm in ARMS:
        for side in ("queries", "reference"):
            name = f"{arm}_query.jsonl" if side == "queries" else f"{arm}_reference.jsonl"
            path = BASE / "data" / side / name
            if not path.exists():
                sys.exit(f"missing {path} -- run 02_build_splits.py for both arms first")
            for r in read_jsonl(path):
                r["arm"] = arm
                entries[r["acc"]] = r
                per_arm[arm] += 1
    print(f"[load] {len(entries)} unique sequences ({dict(per_arm)})", flush=True)

    # ---- joint clustering -------------------------------------------------
    all_fa = interim / "joint_all.fasta"
    with all_fa.open("w") as fh:
        for acc, r in sorted(entries.items()):
            fh.write(f">{acc}\n{r['seq']}\n")

    # Include the input size in the cache name, so the "skip if exists" branch
    # cannot reuse a clustering built from a different input.
    prefix = interim / (f"joint_clu{args.min_seq_id}_cov{args.cov_mode}"
                        f"_n{len(entries)}")
    cluster_tsv = prefix.with_name(prefix.name + "_cluster.tsv")
    if not cluster_tsv.exists():
        tmp = interim / "joint_tmp"
        cmd = [str(MMSEQS), "easy-cluster", str(all_fa), str(prefix), str(tmp),
               "--min-seq-id", str(args.min_seq_id),
               "-c", str(args.coverage),
               "--cov-mode", str(args.cov_mode),
               "--threads", str(args.threads), "-v", "1"]
        print(f"[cluster] {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True)
        subprocess.run(["rm", "-rf", str(tmp)], check=False)

    members = defaultdict(list)
    with cluster_tsv.open() as fh:
        for line in fh:
            rep, mem = line.rstrip("\n").split("\t")[:2]
            members[rep].append(mem)
    print(f"[cluster] {len(members)} joint clusters from {len(entries)} sequences", flush=True)

    mixed = sum(1 for m in members.values()
                if len({entries[a]["arm"] for a in m if a in entries}) > 1)
    print(f"[cluster] {mixed} clusters span both length arms "
          f"(cross-arm relationships captured by joint clustering)", flush=True)

    # ---- split whole clusters --------------------------------------------
    rng = random.Random(args.seed)
    reps = sorted(members)
    rng.shuffle(reps)
    query_reps = set(reps[:int(round(len(reps) * args.query_frac))])

    q_accs, r_accs = [], []
    for rep, mems in members.items():
        (q_accs if rep in query_reps else r_accs).extend(m for m in mems if m in entries)

    ref_families = set()
    for acc in r_accs:
        ref_families.update(entries[acc]["pfam"])

    for acc in q_accs:
        shared = set(entries[acc]["pfam"]) & ref_families
        entries[acc]["answerable"] = bool(shared)
        entries[acc]["answerable_families"] = sorted(shared)

    # ---- write ------------------------------------------------------------
    outq = BASE / "data/queries"
    outr = BASE / "data/reference"
    summary = {"clusters": len(members), "mixed_arm_clusters": mixed,
               "params": vars(args), "arms": {}}

    for arm in ARMS:
        accs = [a for a in q_accs if entries[a]["arm"] == arm]
        with (outq / f"joint_{arm}_query.fasta").open("w") as fa, \
             (outq / f"joint_{arm}_query.jsonl").open("w") as js:
            for acc in sorted(accs):
                fa.write(f">{acc}\n{entries[acc]['seq']}\n")
                js.write(json.dumps(entries[acc]) + "\n")
        summary["arms"][arm] = {
            "n_query": len(accs),
            "answerable": sum(entries[a]["answerable"] for a in accs),
            "answerable_and_experimental": sum(
                entries[a]["answerable"] and entries[a]["experimental"] for a in accs),
        }
        print(f"[qry] {arm}: {len(accs)} queries, "
              f"{summary['arms'][arm]['answerable']} answerable", flush=True)

    with (outr / "joint_reference.fasta").open("w") as fa, \
         (outr / "joint_reference.jsonl").open("w") as js:
        for acc in sorted(r_accs):
            fa.write(f">{acc}\n{entries[acc]['seq']}\n")
            js.write(json.dumps(entries[acc]) + "\n")
    summary["n_reference"] = len(r_accs)
    summary["n_reference_families"] = len(ref_families)
    print(f"[ref] {len(r_accs)} sequences, {len(ref_families)} families", flush=True)

    out = BASE / "results" / "joint_split_manifest.json"
    out.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(f"[done] {out.relative_to(BASE)}")
    print("\nNEXT: run 02d_measure_separation.py to measure each query's "
          "identity to the reference.")


if __name__ == "__main__":
    main()
