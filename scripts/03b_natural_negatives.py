#!/usr/bin/env python3
"""Natural hard negatives (D6) for Test 1: real proteins with no homolog in
the reference.

Synthetic decoys (D0-D5) are non-homologous by construction, but a method might
separate them from real queries by recognising "natural protein" versus
"synthetic peptide" rather than "homolog" versus "non-homolog".

Natural hard negatives address this. They are real Swiss-Prot proteins --
natural composition, natural low-complexity structure, natural transmembrane
content -- for which the reference contains no family-mate. A confident
family-level assignment for one of them is a false positive that cannot be
explained by synthetic-versus-natural discrimination.

Two filters define them, and both are needed:

  1. Clan-level family absence. Not merely "no shared Pfam family" -- two
     families inside one Pfam clan may be genuine remote homologs, so
     family-level absence would silently admit true positives into the negative
     set. Families without a clan are treated as their own singleton clan.

  2. Measured sequence remoteness. No reference sequence above 30% identity
     over 80% of the query, using the values written by 02d. Clan absence is a
     statement about annotation; this is a statement about the sequence itself,
     and neither implies the other.

Negatives are then matched to the remote positives on the nuisance covariates
that could otherwise drive a spurious difference: length, hydrophobicity,
low-complexity fraction, and transmembrane status.

Limitation: this is weaker than the synthetic guarantee. A synthetic decoy
cannot have a homolog; a natural hard negative merely has no detected
clan-mate. The two negative classes have complementary weaknesses, so both are
used.

Input:  data/queries/{prefix}_{arm}_query.jsonl (with 02d fields),
        data/reference/{prefix}_reference.jsonl, data/raw/Pfam-A.clans.tsv.gz
Output: data/decoys/{arm}_D6_natural_hard_{matched,all}.{fasta,jsonl},
        results/natural_negatives_manifest.json

Usage: python scripts/03b_natural_negatives.py [--prefix joint] [--max-pident 30]

Standard library only.
"""

import argparse
import gzip
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
CLANS = BASE / "data/raw/Pfam-A.clans.tsv.gz"

LENGTH_BINS_SMALL = [(1, 30), (31, 50), (51, 75), (76, 99)]
LENGTH_BINS_MID = [(100, 130), (131, 165), (166, 199)]
LENGTH_BINS_CONTROL = [(200, 250), (251, 320), (321, 400)]


def load_clan_map():
    """pfam accession -> clan accession, with unclanned families as singletons."""
    m = {}
    with gzip.open(CLANS, "rt") as fh:
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) < 2 or not p[0]:
                continue
            m[p[0]] = p[1] if p[1] else p[0]
    return m


def clans_of(pfams, clan_map):
    return {clan_map.get(p, p) for p in pfams}


def read_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def bin_of(L, bins):
    for lo, hi in bins:
        if lo <= L <= hi:
            return f"{lo}-{hi}"
    return None


def match(positives, negatives, bins, rng, gravy_tol=0.5, lc_tol=0.15):
    """Greedy 1:1 matching within (length bin, transmembrane) strata.

    Matching is on the covariates that could produce a difference having
    nothing to do with homology: hydrophobicity and low-complexity content
    especially, since a hydrophobic 40-mer resembles every other hydrophobic
    40-mer regardless of ancestry.
    """
    pools = defaultdict(list)
    for n in negatives:
        b = bin_of(n["length"], bins)
        if b:
            pools[(b, n["transmembrane"])].append(n)
    for k in pools:
        rng.shuffle(pools[k])

    matched, unmatched = [], 0
    for p in positives:
        b = bin_of(p["length"], bins)
        if not b:
            continue
        pool = pools.get((b, p["transmembrane"]), [])
        best, best_i, best_d = None, None, None
        for i, n in enumerate(pool):
            dg = abs(n["gravy"] - p["gravy"])
            dl = abs(n["low_complexity"] - p["low_complexity"])
            if dg > gravy_tol or dl > lc_tol:
                continue
            d = (dg / gravy_tol) ** 2 + (dl / lc_tol) ** 2
            if best_d is None or d < best_d:
                best, best_i, best_d = n, i, d
        if best is None:
            unmatched += 1
            continue
        pool.pop(best_i)
        best = dict(best)
        best["matched_to"] = p["acc"]
        matched.append(best)
    return matched, unmatched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="joint")
    ap.add_argument("--max-pident", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    if not CLANS.exists():
        raise SystemExit(f"missing {CLANS} -- fetch Pfam-A.clans.tsv.gz first")
    clan_map = load_clan_map()
    print(f"[clans] {len(clan_map)} Pfam families, "
          f"{len(set(clan_map.values()))} distinct clans/singletons", flush=True)

    ref = read_jsonl(BASE / "data/reference" / f"{args.prefix}_reference.jsonl")
    ref_clans = set()
    for r in ref:
        ref_clans |= clans_of(r["pfam"], clan_map)
    print(f"[ref] {len(ref)} sequences covering {len(ref_clans)} clans", flush=True)

    rng = random.Random(args.seed)
    summary = {"max_pident": args.max_pident, "arms": {}}

    for arm, bins in [("small", LENGTH_BINS_SMALL),
                      ("mid", LENGTH_BINS_MID),
                      ("control", LENGTH_BINS_CONTROL)]:
        qpath = BASE / "data/queries" / f"{args.prefix}_{arm}_query.jsonl"
        rows = read_jsonl(qpath)
        funnel = Counter(total=len(rows))

        positives, candidates = [], []
        for r in rows:
            remote = r.get("max_pident_qcov80", 100.0) <= args.max_pident
            if r.get("answerable") and remote:
                positives.append(r)
                funnel["remote_positives"] += 1
            if r.get("answerable"):
                continue
            funnel["family_absent"] += 1
            if clans_of(r["pfam"], clan_map) & ref_clans:
                funnel["rejected_clan_present"] += 1
                continue
            funnel["clan_absent"] += 1
            if not remote:
                funnel["rejected_not_remote"] += 1
                continue
            funnel["clan_absent_and_remote"] += 1
            candidates.append(r)

        matched, unmatched = match(positives, candidates, bins, rng)

        # Two negative sets are emitted, because covariate matching and
        # statistical power pull in opposite directions on a small pool:
        #
        #   _matched  1:1 covariate-matched on length, GRAVY, low-complexity
        #             and transmembrane status.
        #   _all      every clan-absent, remote candidate.
        #
        # Greedy matching can exhaust the transmembrane stratum and drop the
        # remaining TM positives, so _matched is not necessarily better
        # balanced than _all. _all is the primary set and _matched a
        # sensitivity check; covariate balance is reported for both.
        d = BASE / "data/decoys"
        for name, rows_out in (("matched", matched), ("all", candidates)):
            fa = d / f"{arm}_D6_natural_hard_{name}.fasta"
            js = d / f"{arm}_D6_natural_hard_{name}.jsonl"
            with fa.open("w") as f_fa, js.open("w") as f_js:
                for r in rows_out:
                    f_fa.write(f">{r['acc']} len={r['length']}\n{r['seq']}\n")
                    f_js.write(json.dumps(r) + "\n")

        def dist(rs, key):
            vals = [x[key] for x in rs]
            return round(sum(vals) / len(vals), 3) if vals else None

        summary["arms"][arm] = {
            "funnel": dict(funnel),
            "n_matched_negatives": len(matched),
            "unmatched_positives": unmatched,
            "balance": {
                "positives": {k: dist(positives, k)
                              for k in ("length", "gravy", "low_complexity")},
                "negatives": {k: dist(matched, k)
                              for k in ("length", "gravy", "low_complexity")},
                "tm_frac_pos": round(sum(p["transmembrane"] for p in positives)
                                     / max(len(positives), 1), 3),
                "tm_frac_neg": round(sum(n["transmembrane"] for n in matched)
                                     / max(len(matched), 1), 3),
            },
            "n_all_negatives": len(candidates),
            "balance_all": {
                "negatives": {k: dist(candidates, k)
                              for k in ("length", "gravy", "low_complexity")},
                "tm_frac_neg": round(sum(n["transmembrane"] for n in candidates)
                                     / max(len(candidates), 1), 3),
            },
            "per_bin": {b: sum(1 for n in matched if bin_of(n["length"], bins) == b)
                        for b in [f"{lo}-{hi}" for lo, hi in bins]},
            "per_bin_all": {b: sum(1 for n in candidates if bin_of(n["length"], bins) == b)
                            for b in [f"{lo}-{hi}" for lo, hi in bins]},
            "per_bin_positives": {b: sum(1 for p in positives if bin_of(p["length"], bins) == b)
                                  for b in [f"{lo}-{hi}" for lo, hi in bins]},
        }

        print(f"\n[{arm}] funnel: {dict(funnel)}")
        print(f"[{arm}] matched natural hard negatives: {len(matched)} "
              f"({unmatched} positives left unmatched)")
        print(f"[{arm}] per bin  neg: {summary['arms'][arm]['per_bin']}")
        print(f"[{arm}]          pos: {summary['arms'][arm]['per_bin_positives']}")
        print(f"[{arm}] balance: {json.dumps(summary['arms'][arm]['balance'])}")

    out = BASE / "results" / "natural_negatives_manifest.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\n[done] {out.relative_to(BASE)}")


if __name__ == "__main__":
    main()
