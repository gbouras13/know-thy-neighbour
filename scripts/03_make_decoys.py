#!/usr/bin/env python3
"""Generate the synthetic decoy ladder (D0-D5) for Test 1: inputs with no
homologs by construction.

  D0  uniform      i.i.d. uniform over the 20 standard residues
  D1  composition  i.i.d. from the reference-set background composition
  D2  shuffle      per-sequence shuffle of a real query
  D3  reverse      reversal of a real query
  D4  markov2      order-2 Markov chain trained on the reference set
  D5  markov3      order-3 Markov chain trained on the reference set

The ladder separates the kind of signal a false positive exploits. D0/D1 carry
no order. D2/D3 preserve each query's exact amino-acid composition while
destroying evolutionary order, so a hit there is composition-driven. D4/D5
additionally preserve local k-mer statistics. Natural hard negatives (D6) are
built separately by 03b_natural_negatives.py.

Every class is length-matched to the real query distribution, so the
false-positive rate can be read against length without a confound. Rejecting
decoys is necessary but not sufficient; true-positive performance is measured
separately on the real queries.

Input:  data/queries/{prefix}_{arm}_query.jsonl, data/reference/{prefix}_reference.jsonl
Output: data/decoys/{arm}_{D0..D5}_{class}_seed{k}.fasta,
        results/decoy_manifest_{arm}.json

Usage: python scripts/03_make_decoys.py --arm small|mid|control [--n 5000] [--seeds 5]

Standard library only.
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
AA = "ACDEFGHIKLMNPQRSTVWY"

CLASSES = ["uniform", "composition", "shuffle", "reverse", "markov2", "markov3"]
CLASS_ID = {"uniform": "D0", "composition": "D1", "shuffle": "D2",
            "reverse": "D3", "markov2": "D4", "markov3": "D5"}


def read_jsonl(path):
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def weighted_alphabet(seqs):
    """Background composition of the reference set, as (letters, cumulative weights)."""
    counts = Counter()
    for s in seqs:
        counts.update(s)
    letters = [a for a in AA if counts[a] > 0]
    weights = [counts[a] for a in letters]
    return letters, weights


class Markov:
    """Order-k residue model with backoff to the order-0 background."""

    def __init__(self, seqs, order, rng):
        self.order = order
        self.rng = rng
        self.trans = defaultdict(Counter)
        self.starts = Counter()
        bg = Counter()
        for s in seqs:
            bg.update(s)
            if len(s) <= order:
                continue
            self.starts[s[:order]] += 1
            for i in range(len(s) - order):
                self.trans[s[i:i + order]][s[i + order]] += 1
        self.bg_letters = [a for a in AA if bg[a] > 0]
        self.bg_weights = [bg[a] for a in self.bg_letters]
        self._start_keys = list(self.starts)
        self._start_w = [self.starts[k] for k in self._start_keys]

    def _next(self, ctx):
        tbl = self.trans.get(ctx)
        if not tbl:
            return self.rng.choices(self.bg_letters, weights=self.bg_weights, k=1)[0]
        letters = list(tbl)
        return self.rng.choices(letters, weights=[tbl[c] for c in letters], k=1)[0]

    def generate(self, length):
        if length <= self.order:
            return "".join(self.rng.choices(self.bg_letters, weights=self.bg_weights, k=length))
        out = list(self.rng.choices(self._start_keys, weights=self._start_w, k=1)[0])
        while len(out) < length:
            out.append(self._next("".join(out[-self.order:])))
        return "".join(out[:length])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="small", choices=["small", "mid", "control"])
    ap.add_argument("--n", type=int, default=5000, help="decoys per class per seed")
    ap.add_argument("--seeds", type=int, default=5)
    # Decoys must be length-matched to the query set they will be compared
    # against, and the background composition must come from the reference
    # actually being searched -- so both follow whichever split is in use.
    ap.add_argument("--prefix", default="joint",
                    help="split prefix; 'joint' uses the joint split, '' the per-arm one")
    args = ap.parse_args()

    pre = f"{args.prefix}_" if args.prefix else ""
    qpath = BASE / "data/queries" / f"{pre}{args.arm}_query.jsonl"
    rpath = (BASE / "data/reference" / f"{args.prefix}_reference.jsonl") if args.prefix \
        else (BASE / "data/reference" / f"{args.arm}_reference.jsonl")
    for p in (qpath, rpath):
        if not p.exists():
            raise SystemExit(f"missing {p} -- run 02c_joint_split.py first")

    queries = list(read_jsonl(qpath))
    ref_seqs = [r["seq"] for r in read_jsonl(rpath)]
    q_seqs = [q["seq"] for q in queries]
    lengths = [q["length"] for q in queries]

    letters, weights = weighted_alphabet(ref_seqs)

    outdir = BASE / "data/decoys"
    outdir.mkdir(parents=True, exist_ok=True)

    manifest = {"arm": args.arm, "n_per_class_per_seed": args.n,
                "seeds": args.seeds, "source_queries": len(queries), "files": []}

    for seed in range(1, args.seeds + 1):
        rng = random.Random(1000 * seed + 7)
        # Length-match every class to the real query length distribution.
        target_lengths = rng.choices(lengths, k=args.n)
        # Real sequences to derive D2/D3 from, sampled with the same rng stream.
        donors = rng.choices(q_seqs, k=args.n)

        m2 = Markov(ref_seqs, 2, rng)
        m3 = Markov(ref_seqs, 3, rng)

        for cls in CLASSES:
            path = outdir / f"{args.arm}_{CLASS_ID[cls]}_{cls}_seed{seed}.fasta"
            n_written = 0
            with path.open("w") as fh:
                for i in range(args.n):
                    L = target_lengths[i]
                    donor = donors[i]
                    if cls == "uniform":
                        seq = "".join(rng.choices(AA, k=L))
                    elif cls == "composition":
                        seq = "".join(rng.choices(letters, weights=weights, k=L))
                    elif cls == "shuffle":
                        chars = list(donor)
                        rng.shuffle(chars)
                        seq = "".join(chars)
                    elif cls == "reverse":
                        seq = donor[::-1]
                    elif cls == "markov2":
                        seq = m2.generate(L)
                    else:
                        seq = m3.generate(L)
                    fh.write(f">{CLASS_ID[cls]}_{cls}_s{seed}_{i:05d} len={len(seq)}\n{seq}\n")
                    n_written += 1
            manifest["files"].append({
                "class": cls, "id": CLASS_ID[cls], "seed": seed,
                "path": str(path.relative_to(BASE)), "n": n_written,
            })
            print(f"[{CLASS_ID[cls]:>2} {cls:<11} seed {seed}]  {n_written:5d} -> {path.name}",
                  flush=True)

    man = BASE / "results" / f"decoy_manifest_{args.arm}.json"
    man.parent.mkdir(parents=True, exist_ok=True)
    man.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nwrote {len(manifest['files'])} decoy files; manifest -> {man.relative_to(BASE)}")


if __name__ == "__main__":
    main()
