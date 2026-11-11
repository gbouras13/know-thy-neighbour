#!/usr/bin/env python3
"""Build a per-arm query / reference split for one Test 1 length arm.

Input:  data/raw/swissprot_{arm}.jsonl.gz (from 01_fetch_swissprot.py)
Output: data/queries/{arm}_query.{fasta,jsonl},
        data/reference/{arm}_reference.{fasta,jsonl},
        results/split_manifest_{arm}.json
The per-record JSONL produced here is the input to 02c_joint_split.py, which
builds the final joint split.

Usage: python scripts/02_build_splits.py --arm small|mid|control

Design notes
------------
Clusters, not sequences, are assigned to the query or reference side. Because
clustering runs at 30% identity, a query has no >=30%-identical relative left
in the reference -- which is what makes the retrieval task *remote* homology
rather than trivial lookup. (MMseqs2 clustering is greedy set-cover, so this
is the usual approximation, not a hard guarantee.)

A query is only *answerable* if at least one reference protein shares one of
its Pfam accessions. Unanswerable queries are kept and flagged: they are a
second, weaker class of negative control -- real proteins whose family is
absent from the reference, where the correct behaviour is also to abstain.

Evidence level is carried through rather than silently filtered. Swiss-Prot
entries at "Inferred from homology" / "Predicted" had their annotation
produced *by homology transfer in the first place*, so scoring a homology
method against them is partly circular. The primary analysis should use the
experimental subset; the full set is a sensitivity check.

Standard library only.
"""

import argparse
import gzip
import json
import random
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
MMSEQS = Path.home() / "miniforge3/envs/mmseqs2/bin/mmseqs"

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")

# Kyte-Doolittle hydropathy, for the composition-shortcut diagnostic.
KD = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
    "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
    "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

EXPERIMENTAL = {"Evidence at protein level", "Evidence at transcript level"}


def gravy(seq):
    vals = [KD[c] for c in seq if c in KD]
    return round(sum(vals) / len(vals), 4) if vals else 0.0


def low_complexity_fraction(seq, window=12, threshold=1.9):
    """Fraction of windows whose Shannon entropy (bits) falls below `threshold`.

    A crude stand-in for SEG. Sequence tools mask these regions; embedding
    methods do not, so this is recorded for the composition diagnostic.
    """
    import math

    if len(seq) < window:
        windows = [seq]
    else:
        windows = [seq[i:i + window] for i in range(len(seq) - window + 1)]
    lo = 0
    for w in windows:
        counts = Counter(w)
        n = len(w)
        ent = -sum((c / n) * math.log2(c / n) for c in counts.values())
        if ent < threshold:
            lo += 1
    return round(lo / len(windows), 4)


def parse_pfam(field):
    return tuple(sorted(p for p in field.strip().strip(";").split(";") if p))


def load(jsonl_gz):
    """Yield dict rows written by 01_fetch_swissprot.py (one JSON object per line)."""
    with gzip.open(jsonl_gz, "rt") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def col(row, *candidates):
    """UniProt TSV column names vary by field; take the first that exists."""
    for c in candidates:
        if c in row:
            return row[c]
    return ""


def run_mmseqs_cluster(fasta, prefix, min_seq_id, coverage, threads):
    tmp = prefix.parent / f"{prefix.name}_tmp"
    cmd = [
        str(MMSEQS), "easy-cluster", str(fasta), str(prefix), str(tmp),
        "--min-seq-id", str(min_seq_id),
        "-c", str(coverage),
        "--cov-mode", "0",
        "--threads", str(threads),
        "-v", "1",
    ]
    print(f"[cluster] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    subprocess.run(["rm", "-rf", str(tmp)], check=False)
    return prefix.with_name(prefix.name + "_cluster.tsv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["small", "mid", "control"])
    ap.add_argument("--query-frac", type=float, default=0.20)
    ap.add_argument("--min-seq-id", type=float, default=0.3)
    ap.add_argument("--coverage", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    raw = BASE / "data/raw" / f"swissprot_{args.arm}.jsonl.gz"
    if not raw.exists():
        sys.exit(f"missing {raw} -- run 01_fetch_swissprot.py first")

    interim = BASE / "data/interim"
    interim.mkdir(parents=True, exist_ok=True)

    # ---- load and filter -------------------------------------------------
    kept, stats = {}, Counter()
    for row in load(raw):
        stats["read"] += 1
        acc = row["Entry"]
        seq = col(row, "Sequence").strip().upper()
        pfam = parse_pfam(col(row, "Pfam"))

        if not seq:
            stats["drop_no_seq"] += 1
            continue
        if set(seq) - STANDARD_AA:
            stats["drop_nonstandard_aa"] += 1
            continue
        if not pfam:
            stats["drop_no_pfam"] += 1
            continue

        evidence = col(row, "Protein existence")
        kept[acc] = {
            "acc": acc,
            "name": col(row, "Entry Name"),
            "protein": col(row, "Protein names"),
            "organism": col(row, "Organism"),
            "length": len(seq),
            "seq": seq,
            "pfam": list(pfam),
            "go": [g for g in col(row, "Gene Ontology IDs").replace(" ", "").split(";") if g],
            "evidence": evidence,
            "experimental": evidence in EXPERIMENTAL,
            "transmembrane": "TRANSMEM" in col(row, "Transmembrane"),
            "gravy": gravy(seq),
            "low_complexity": low_complexity_fraction(seq),
        }
        stats["kept"] += 1

    print(f"[{args.arm}] filter: {json.dumps(dict(stats), indent=None)}", flush=True)
    if not kept:
        sys.exit("nothing survived filtering")

    # ---- cluster ---------------------------------------------------------
    all_fa = interim / f"{args.arm}_all.fasta"
    with all_fa.open("w") as fh:
        for acc, r in kept.items():
            fh.write(f">{acc}\n{r['seq']}\n")

    cluster_tsv = run_mmseqs_cluster(
        all_fa, interim / f"{args.arm}_clu{args.min_seq_id}",
        args.min_seq_id, args.coverage, args.threads,
    )

    members = defaultdict(list)
    with cluster_tsv.open() as fh:
        for line in fh:
            rep, mem = line.rstrip("\n").split("\t")[:2]
            members[rep].append(mem)
    print(f"[{args.arm}] {len(members)} clusters from {len(kept)} sequences", flush=True)

    # ---- split whole clusters -------------------------------------------
    rng = random.Random(args.seed)
    reps = sorted(members)
    rng.shuffle(reps)
    n_query_clusters = int(round(len(reps) * args.query_frac))
    query_reps = set(reps[:n_query_clusters])

    query_accs, ref_accs = [], []
    for rep, mems in members.items():
        (query_accs if rep in query_reps else ref_accs).extend(mems)

    # ---- answerability ---------------------------------------------------
    ref_families = set()
    for acc in ref_accs:
        ref_families.update(kept[acc]["pfam"])

    for acc in query_accs:
        shared = set(kept[acc]["pfam"]) & ref_families
        kept[acc]["answerable"] = bool(shared)
        kept[acc]["answerable_families"] = sorted(shared)

    n_answerable = sum(kept[a]["answerable"] for a in query_accs)

    # ---- write -----------------------------------------------------------
    def dump(accs, side):
        d = BASE / "data" / ("queries" if side == "query" else "reference")
        d.mkdir(parents=True, exist_ok=True)
        fa = d / f"{args.arm}_{side}.fasta"
        meta = d / f"{args.arm}_{side}.jsonl"
        with fa.open("w") as f_fa, meta.open("w") as f_meta:
            for acc in sorted(accs):
                r = kept[acc]
                f_fa.write(f">{acc}\n{r['seq']}\n")
                f_meta.write(json.dumps(r) + "\n")
        return fa, meta

    q_fa, q_meta = dump(query_accs, "query")
    r_fa, r_meta = dump(ref_accs, "reference")

    manifest = {
        "arm": args.arm,
        "source": str(raw.relative_to(BASE)),
        "params": {
            "query_frac": args.query_frac,
            "min_seq_id": args.min_seq_id,
            "coverage": args.coverage,
            "seed": args.seed,
        },
        "filter": dict(stats),
        "clusters": len(members),
        "query_clusters": n_query_clusters,
        "n_query": len(query_accs),
        "n_reference": len(ref_accs),
        "n_query_answerable": n_answerable,
        "n_query_unanswerable": len(query_accs) - n_answerable,
        "n_query_experimental": sum(kept[a]["experimental"] for a in query_accs),
        "n_reference_families": len(ref_families),
        "length_min": min(kept[a]["length"] for a in kept),
        "length_max": max(kept[a]["length"] for a in kept),
        "outputs": [str(p.relative_to(BASE)) for p in (q_fa, q_meta, r_fa, r_meta)],
    }
    man_path = BASE / "results" / f"split_manifest_{args.arm}.json"
    man_path.parent.mkdir(parents=True, exist_ok=True)
    man_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
