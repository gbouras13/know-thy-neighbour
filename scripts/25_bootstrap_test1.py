#!/usr/bin/env python3
"""Test 1 (Swiss-Prot/Pfam): query-level bootstrap for TPR at a target FPR.

Design requirements (`10_threshold.py`'s `boot_ci` resamples items with a fixed
threshold and is not used for intervals here):

  1. derivatives of one query stay in the same resampling unit;
  2. the threshold is refitted inside every replicate, so threshold-estimation
     uncertainty is propagated into the intervals;
  3. methods share queries, so differences are paired.

DEPENDENCE STRUCTURE
--------------------
How `03_make_decoys.py` generates decoys determines the grouping. Per arm and
per seed the generator draws

    target_lengths = rng.choices(lengths, k=n)      # n length draws
    donors         = rng.choices(q_seqs,  k=n)      # n donor draws, WITH replacement

and then

    D0 uniform      sequence from the flat alphabet, length from target_lengths
    D1 composition  sequence from the reference background, length likewise
    D4 markov2      sequence from an order-2 chain fitted to the reference
    D5 markov3      sequence from an order-3 chain fitted to the reference
    D2 shuffle      a permutation OF THE DONOR      <- inherits donor length + composition
    D3 reverse      the donor, reversed             <- DETERMINISTIC given the donor

So:

  * D0/D1/D4/D5 inherit nothing from any individual query except a length drawn
    from the query length distribution. They are exchangeable items, not query
    derivatives, and are resampled as their own units. (They do share a fitted
    generative model across all seeds; that weaker, global dependence is not
    modelled here.)
  * D2/D3 belong to a donor query and are grouped with it.
  * Donors are sampled WITH REPLACEMENT and independently per seed, so a query
    donates a random number of times -- 0 to 9 in the small arm -- not exactly
    once per seed. Group sizes are therefore variable and must be measured.
  * D3 being deterministic in the donor means a donor collision produces an
    EXACT duplicate sequence. In the small arm 62.3% of the 25,000 D3 decoys
    repeat a sequence already present. The effective D3 sample size is the
    number of distinct donors, not the number of FASTA records.
  * D6 natural hard negatives are real, distinct Swiss-Prot proteins. They are
    independent and are resampled individually.

RECOVERING THE DONOR MAPPING
----------------------------
The decoy FASTA headers are positional (`D2_shuffle_s1_00000`) and the donor is
recorded nowhere. It is recovered by replaying the generator's RNG: the seeding
(`random.Random(1000*seed + 7)`) and the two `rng.choices` calls are
deterministic, and `random.choices` consumes the same number of draws whatever
the population holds, so replaying with `range(len(queries))` yields the donor
INDICES exactly.

`--verify` (on by default) reconstructs every D3 sequence as `donor[::-1]` and
compares it to the FASTA (25,000/25,000 in each arm); a mismatch aborts the
run rather than mis-grouping.

PROCEDURE
---------
One replicate:
  1. resample real-query units with replacement, stratified by arm (arms cover
     different length ranges and were built separately);
  2. carry each drawn query's D2/D3 derivatives with it;
  3. resample D6, and D0/D1/D4/D5, with replacement as their own units;
  4. REFIT the threshold inside the replicate: the most permissive score holding
     FPR at or below the target against the resampled negatives of the
     calibration class (D6 by default), pooled over all arms and length bins;
  5. apply it and record TPR over resampled positives and achieved FPR per class.

Every method sees the identical resampled unit indices on every replicate, so
differences are paired.

Intervals are query-level bootstrap percentile intervals, not standard errors.

Options
  --calib-bin BIN     fit the threshold on one length bin and transfer it to
                      all bins (alternative to the pooled default).
  --by-stratum        report TPR per coarse length stratum.
  --pos-filter        robustness analysis on positives: single_pfam keeps
                      single-domain queries; arch_match keeps queries whose
                      exact Pfam architecture occurs in the reference.
  --length-weighted   robustness analysis: fit the threshold on D6 reweighted
                      to the positives' length distribution.

Inputs:  data/queries/joint_<arm>_query.jsonl, data/decoys/*.fasta,
         data/reference/joint_reference.jsonl (arch_match only),
         results/<method>_<arm>*.tsv[.gz].
Outputs: printed tables (TPR, achieved FPR, paired differences, threshold
         spread, per-class FPR).
Run:     python3 scripts/25_bootstrap_test1.py [--replicates 1000]
         [--target-fpr 0.01] [--only "ProtT5 kNN" ...]

Stdlib + numpy. Scores are projected onto integer unit indices once, so a
replicate is a handful of vectorised gathers rather than a Python loop over
~220,000 records per method.
"""

import argparse
import gzip
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent

BINS = ["1-30", "31-50", "51-75", "76-99", "100-130", "131-165",
        "166-199", "200-250", "251-320", "321-400"]
ARMS = {"small": BINS[:4], "mid": BINS[4:7], "control": BINS[7:]}

# Classes whose members are derived from a specific donor query.
DERIVED = ("D2_shuffle", "D3_reverse")
# Classes generated from global statistics; only the length comes from the
# query distribution.
SYNTHETIC = ("D0_uniform", "D1_composition", "D4_markov2", "D5_markov3")

N_PER_CLASS_PER_SEED = 5000
N_SEEDS = 5


def smart_open(path):
    """Open a table, or return None if it is absent.

    Callers treat None as "this method has no data, skip it", so a missing
    table never aborts the run.
    """
    path = Path(path)
    if path.suffix == ".gz":
        if path.exists():
            return gzip.open(path, "rt")
        # Fall back to the un-gzipped table (some tables are stored as .tsv).
        raw = path.with_suffix("")
        return open(raw) if raw.exists() else None
    if path.exists():
        return open(path)
    gz = path.with_name(path.name + ".gz")
    return gzip.open(gz, "rt") if gz.exists() else None


def bin_of(L):
    for b in BINS:
        lo, hi = (int(x) for x in b.split("-"))
        if lo <= L <= hi:
            return b
    return None


def read_fasta(path):
    """-> [(id, length)], streaming."""
    out, cid, n = [], None, 0
    fh = smart_open(path)
    if fh is None:
        return out
    with fh:
        for line in fh:
            if line.startswith(">"):
                if cid is not None:
                    out.append((cid, n))
                cid, n = line[1:].split()[0], 0
            else:
                n += len(line.strip())
    if cid is not None:
        out.append((cid, n))
    return out


def replay_donors(queries, n=N_PER_CLASS_PER_SEED, seeds=N_SEEDS):
    """seed -> [donor query index per decoy position], by replaying the RNG.

    Mirrors 03_make_decoys.py exactly: one `choices` over lengths, then one over
    the query sequences. Substituting `range(len(queries))` for the sequences is
    safe because `random.choices` draws the same number of variates whatever the
    population contains.
    """
    lengths = [q["length"] for q in queries]
    nq = len(queries)
    out = {}
    for seed in range(1, seeds + 1):
        rng = random.Random(1000 * seed + 7)
        rng.choices(lengths, k=n)                      # target_lengths, discarded
        out[seed] = rng.choices(range(nq), k=n)        # donor indices
    return out


def verify_donors(arm, queries, donors):
    """Reconstruct every D3 sequence from its putative donor and compare."""
    seqs = [q["seq"] for q in queries]
    ok = tot = 0
    for seed, idx in donors.items():
        path = BASE / "data/decoys" / f"{arm}_D3_reverse_seed{seed}.fasta"
        cid, buf = None, []
        recs = []
        fh = smart_open(path)
        if fh is None:
            return None
        with fh:
            for line in fh:
                if line.startswith(">"):
                    if cid is not None:
                        recs.append("".join(buf))
                    cid, buf = line, []
                else:
                    buf.append(line.strip())
        if cid is not None:
            recs.append("".join(buf))
        for i, s in enumerate(recs):
            tot += 1
            if i < len(idx) and s == seqs[idx[i]][::-1]:
                ok += 1
    return ok, tot


def best_scores(path, col, better):
    """query id -> best score. Missing queries are simply absent.

    Header convention follows 10_threshold.py: cosine tables (higher-is-better)
    carry a `query\ttarget\trank\tcosine` header, E-value tables do not.
    """
    fh = smart_open(path)
    if fh is None:
        return None
    b = {}
    with fh:
        if better == "higher":
            fh.readline()
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) <= col:
                continue
            try:
                s = float(p[col])
            except ValueError:
                continue
            cur = b.get(p[0])
            if cur is None or (s > cur if better == "higher" else s < cur):
                b[p[0]] = s
    return b


def build_units(arm, negative_class, verify=False, pos_filter="none", ref_arch=None):
    """Assemble resampling units for one arm.

    Returns
      q_units   [(kind, payload)] one entry per real query:
                ('acc', is_positive_eligible, bin, [derived decoy ids])
      neg_units {class: [(decoy_id, bin)]} for classes resampled independently
    """
    qpath = BASE / "data/queries" / f"joint_{arm}_query.jsonl"
    queries = [json.loads(l) for l in smart_open(qpath)]

    donors = replay_donors(queries)
    if verify:
        ok, tot = verify_donors(arm, queries, donors)
        status = "OK" if ok == tot else "*** MISMATCH ***"
        print(f"[verify] {arm:<8} D3 reconstruction {ok}/{tot}  {status}")
        if ok != tot:
            raise SystemExit(
                "donor replay does not reproduce D3; 03_make_decoys.py has "
                "changed and the grouping would be wrong. Fix before using.")

    # derived decoys, attached to their donor query index
    derived = defaultdict(list)          # query index -> [(decoy id, bin)]
    for cls in DERIVED:
        cid, name = cls.split("_", 1)
        for seed in range(1, N_SEEDS + 1):
            fa = BASE / "data/decoys" / f"{arm}_{cls}_seed{seed}.fasta"
            recs = read_fasta(fa)
            idx = donors[seed]
            for i, (did, L) in enumerate(recs):
                if i >= len(idx):
                    break
                b = bin_of(L)
                if b:
                    derived[idx[i]].append((f"{cls}_seed{seed}|{did}", b, cls))

    q_units = []
    for i, r in enumerate(queries):
        b = bin_of(r["length"])
        pos = bool(r.get("answerable") and r.get("remote")) and b is not None
        # Optional positive subsets (--pos-filter). Default "none" keeps all
        # positives. Negatives are never filtered: D6 is already 98.5%
        # single-Pfam.
        if pos and pos_filter == "single_pfam":
            pos = len(r.get("pfam") or []) == 1
        elif pos and pos_filter == "arch_match":
            pos = tuple(sorted(r.get("pfam") or [])) in ref_arch
        q_units.append((r["acc"], pos, b, derived.get(i, ())))

    # independently-resampled negative classes
    neg_units = {}
    for cls in SYNTHETIC:
        items = []
        for seed in range(1, N_SEEDS + 1):
            fa = BASE / "data/decoys" / f"{arm}_{cls}_seed{seed}.fasta"
            for did, L in read_fasta(fa):
                b = bin_of(L)
                if b:
                    items.append((f"{cls}_seed{seed}|{did}", b))
        if items:
            neg_units[cls] = items
    fa = BASE / "data/decoys" / f"{arm}_{negative_class}.fasta"
    items = [(f"{negative_class}|{did}", b)
             for did, L in read_fasta(fa) if (b := bin_of(L))]
    if items:
        neg_units[negative_class] = items
    return q_units, neg_units


def pick_threshold(neg_scores, n_neg, target, better):
    """Most permissive score holding FPR <= target over the ISSUED negatives.

    n_neg is the issued count, so negatives with no hit at all count as
    correctly rejected rather than being dropped from the denominator.
    """
    if not n_neg:
        return None
    k = int(target * n_neg)
    ordered = sorted(neg_scores, reverse=(better == "higher"))
    if not ordered:
        return float("-inf") if better == "higher" else float("inf")
    if k >= len(ordered):
        # target FPR exceeds what the negatives can produce; accept everything
        return ordered[-1]
    return ordered[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicates", type=int, default=1000)
    ap.add_argument("--target-fpr", type=float, default=0.01)
    ap.add_argument("--calib-bin", default="pooled",
                    help='"pooled" (default) fits the threshold on every query '
                         'at once, so every method lands at the same achieved '
                         'FPR. A single bin name (e.g. "321-400") fits on '
                         'that bin and transfers it to all bins.')
    ap.add_argument("--negative", default="D6_natural_hard_all")
    ap.add_argument("--baseline", default="sequence (MMseqs2)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verify", action="store_true", default=True)
    ap.add_argument("--by-stratum", action="store_true",
                    help="report TPR per length stratum with intervals. Strata "
                         "match 17_stratify.py so the numbers are comparable.")
    ap.add_argument("--no-verify", dest="verify", action="store_false")
    ap.add_argument("--pos-filter", default="none",
                    choices=["none", "single_pfam", "arch_match"],
                    help="restrict POSITIVES (robustness): single_pfam = one "
                         "Pfam domain; arch_match = the query's exact Pfam set "
                         "occurs in some reference protein. Negatives unchanged.")
    ap.add_argument("--length-weighted", action="store_true",
                    help="robustness: fit the threshold on D6 reweighted by "
                         "length bin to the POSITIVES' length distribution "
                         "(w = p_pos(bin)/p_D6(bin)), pooled calibration only.")
    ap.add_argument("--only", nargs="*", default=None,
                    help="restrict to these method display names (speed)")
    args = ap.parse_args()

    methods = [
        ("sequence (MMseqs2)", "sequence_{arm}_vs_joint.tsv.gz", 2, "lower"),
        # Foldseek output may be named structure_Cn_* or structure_C_*; see
        # below. Missing methods are listed in the [skip] summary.
        ("Foldseek", "structure_Cn_{arm}.tsv.gz", 2, "lower"),
        ("ProstT5-Foldseek", "structure_Cprime_{arm}_top50.tsv.gz", 2, "lower"),
        ("gLM2-embed", "knn_{arm}_glm2_650m_embed.tsv.gz", 3, "higher"),
        ("ProtT5 kNN", "knn_{arm}_prott5.tsv.gz", 3, "higher"),
        ("ProstT5 kNN", "knn_{arm}_prostt5.tsv.gz", 3, "higher"),
        ("ESM-2 150M kNN", "knn_{arm}_esm2_150m.tsv.gz", 3, "higher"),
        ("ESM-2 650M kNN", "knn_{arm}_esm2_650m.tsv.gz", 3, "higher"),
        ("ESM-2 3B kNN", "knn_{arm}_esm2_3b.tsv.gz", 3, "higher"),
        ("Ankh-base kNN", "knn_{arm}_ankh_base.tsv.gz", 3, "higher"),
        ("Ankh-large kNN", "knn_{arm}_ankh_large.tsv.gz", 3, "higher"),
        ("ProtTucker/EAT", "knn_{arm}_prottucker.tsv.gz", 3, "higher"),
        ("ESM-2 650M +whiten", "knn_{arm}_esm2_650m_whiten.tsv.gz", 3, "higher"),
        ("ESM-2 150M +whiten", "knn_{arm}_esm2_150m_whiten.tsv.gz", 3, "higher"),
        ("ESM-2 3B +whiten", "knn_{arm}_esm2_3b_whiten.tsv.gz", 3, "higher"),
        ("Ankh-base +whiten", "knn_{arm}_ankh_base_whiten.tsv.gz", 3, "higher"),
        ("Ankh-large +whiten", "knn_{arm}_ankh_large_whiten.tsv.gz", 3, "higher"),
        ("ProstT5 +whiten", "knn_{arm}_prostt5_whiten.tsv.gz", 3, "higher"),
        ("ESM-C 300M kNN", "knn_{arm}_esmc_300m.tsv.gz", 3, "higher"),
        ("ESM-C 300M +whiten", "knn_{arm}_esmc_300m_whiten.tsv.gz", 3, "higher"),
        ("ESM-C 600M kNN", "knn_{arm}_esmc_600m.tsv.gz", 3, "higher"),
        ("ESM-C 600M +whiten", "knn_{arm}_esmc_600m_whiten.tsv.gz", 3, "higher"),
        ("ESM-C 6B kNN", "knn_{arm}_esmc_6b.tsv.gz", 3, "higher"),
        ("ESM-C 6B +whiten", "knn_{arm}_esmc_6b_whiten.tsv.gz", 3, "higher"),
        ("gLM2 +whiten", "knn_{arm}_glm2_650m_embed_whiten.tsv.gz", 3, "higher"),
        ("ProtT5 +whiten", "knn_{arm}_prott5_whiten.tsv.gz", 3, "higher"),
    ]

    # Use structure_C_* for Foldseek if structure_Cn_* is absent.
    if not any((BASE / "results" / f"structure_Cn_small{x}").exists()
               for x in (".tsv", ".tsv.gz")):
        methods = [(n, "structure_C_{arm}.tsv.gz" if n == "Foldseek" else t, c, b)
                   for n, t, c, b in methods]
    if args.only:
        methods = [m for m in methods if m[0] in set(args.only)]
        if not methods:
            raise SystemExit("--only matched no method names")
    ref_arch = None
    if args.pos_filter == "arch_match":
        ref_arch = set()
        for line in smart_open(BASE / "data/reference/joint_reference.jsonl"):
            ref_arch.add(tuple(sorted(json.loads(line).get("pfam") or [])))
    if args.length_weighted and args.calib_bin not in ("pooled", "all"):
        raise SystemExit("--length-weighted requires pooled calibration")

    print("Test 1 (Swiss-Prot/Pfam) -- query-level bootstrap")
    print(f"positive filter: {args.pos_filter};  length-weighted threshold: "
          f"{args.length_weighted}")
    where = ("EVERY query pooled (all arms, all length bins)"
             if args.calib_bin in ("pooled", "all")
             else f"the {args.calib_bin} bin alone")
    print(f"{args.replicates} replicates; resampling unit = one original query, "
          f"carrying its D2/D3\nderivatives. Threshold REFITTED inside every "
          f"replicate on {where}\nagainst {args.negative}, "
          f"target {args.target_fpr:.0%} FPR, then applied to every bin.")
    print("All methods see identical resampled units, so differences are paired.\n")

    # ---- assemble units -----------------------------------------------------
    units, negs = {}, {}
    for arm in ARMS:
        units[arm], negs[arm] = build_units(arm, args.negative, verify=args.verify,
                                            pos_filter=args.pos_filter,
                                            ref_arch=ref_arch)

    npos_tot = sum(1 for a_ in ARMS for u in units[a_] if u[1])
    print(f"positive queries after filter '{args.pos_filter}': {npos_tot}")

    # group-size diagnostics, reported because they set the effective n
    print("\nresampling units and derived-decoy group sizes:")
    for arm in ARMS:
        sizes = [len(u[3]) for u in units[arm]]
        withd = [s for s in sizes if s]
        print(f"  {arm:<8} {len(units[arm]):>6} query units; "
              f"{len(withd):>5} donate D2/D3 "
              f"({100*len(withd)/len(sizes):.1f}%); "
              f"group size max {max(sizes)}, mean over donors "
              f"{sum(withd)/max(len(withd),1):.2f}")

    # ---- load scores, pre-projected onto unit indices -----------------------
    # Everything below is numpy over integer index arrays. Scores are
    # sign-normalised so higher is always better, and a query with no reported
    # hit becomes -inf, i.e. never accepted at any finite threshold. That keeps
    # missing queries as misses against a fixed denominator, as in the
    # non-bootstrap analyses.
    NEG_INF = -np.inf
    # POOLED calibration (default) fits the threshold on all D6 negatives at
    # once, across arms and length bins, so every method is compared at the
    # same achieved overall FPR. A single-bin fit (--calib-bin, e.g. 321-400)
    # transferred to other bins does not hold that FPR elsewhere; it is kept
    # as an alternative for sensitivity analysis.
    POOLED = args.calib_bin in ("pooled", "all")
    calib_arm = (None if POOLED else
                 next(a for a, bs in ARMS.items() if args.calib_bin in bs))

    n_units = {a: len(units[a]) for a in ARMS}
    neg_classes = {a: sorted(negs[a]) for a in ARMS}

    # per-arm static index structures, shared by every method
    is_pos = {a: np.array([u[1] for u in units[a]], dtype=bool) for a in ARMS}

    # Coarse length strata, matching 17_stratify.py so the two are comparable.
    STRATA = ["1-49", "50-99", "100-199", "200-400"]

    def stratum_of(b):
        if b is None:
            return -1
        lo = int(b.split("-")[0])
        if lo < 50:
            return 0
        if lo < 100:
            return 1
        if lo < 200:
            return 2
        return 3

    strat = {a: np.array([stratum_of(u[2]) for u in units[a]], dtype=np.int64)
             for a in ARMS}
    owner, der_cls = {}, {}
    for a in ARMS:
        own, cls = [], []
        for j, u in enumerate(units[a]):
            for _did, _b, c in u[3]:
                own.append(j)
                cls.append(c)
        owner[a] = np.array(own, dtype=np.int64)
        der_cls[a] = np.array(cls)
    # Length weights for D6 (--length-weighted): positives' bin distribution over
    # D6's. Bins holding positives but no D6 cannot be represented; their share
    # of positives is reported rather than silently renormalised away.
    neg_w = None
    if args.length_weighted:
        pos_bins = defaultdict(int)
        for a in ARMS:
            for u in units[a]:
                if u[1]:
                    pos_bins[u[2]] += 1
        neg_bins = defaultdict(int)
        for a in ARMS:
            for _d, b in negs[a].get(args.negative, []):
                neg_bins[b] += 1
        npos_all, nneg_all = sum(pos_bins.values()), sum(neg_bins.values())
        wbin = {b: (pos_bins[b] / npos_all) / (neg_bins[b] / nneg_all)
                for b in neg_bins if neg_bins[b]}
        lost = sum(v for b, v in pos_bins.items() if not neg_bins.get(b))
        print("\nlength weights per bin (p_pos / p_D6):")
        for b in BINS:
            print(f"  {b:>8}  pos {pos_bins.get(b, 0):>6}  D6 {neg_bins.get(b, 0):>5}"
                  f"  w {wbin.get(b, float('nan')):6.2f}")
        print(f"  positives in bins with no D6: {lost} "
              f"({100 * lost / max(npos_all, 1):.2f}%)")
        neg_w = {a: np.array([wbin.get(b, 0.0) for _d, b in
                              negs[a].get(args.negative, [])]) for a in ARMS}
    wfprs = defaultdict(list)

    calib_mask = (None if POOLED else
                  {a: {c: np.array([b == args.calib_bin
                                    for _d, b in negs[a][c]], dtype=bool)
                       for c in neg_classes[a]} for a in ARMS})

    def project(sc, better, a):
        """dict of raw scores -> (positive vector, derived vector, {cls: vector})."""
        sgn = 1.0 if better == "higher" else -1.0
        pv = np.full(n_units[a], NEG_INF)
        for j, u in enumerate(units[a]):
            v = sc.get(f"real|{u[0]}")
            if v is not None:
                pv[j] = sgn * v
        dv = np.full(owner[a].shape[0], NEG_INF)
        k = 0
        for u in units[a]:
            for did, _b, _c in u[3]:
                v = sc.get(did)
                if v is not None:
                    dv[k] = sgn * v
                k += 1
        nv = {}
        for c in neg_classes[a]:
            arr = np.full(len(negs[a][c]), NEG_INF)
            for i, (did, _b) in enumerate(negs[a][c]):
                v = sc.get(did)
                if v is not None:
                    arr[i] = sgn * v
            nv[c] = arr
        return pv, dv, nv

    proj, sgn_of = {}, {}
    for name, tmpl, col, better in methods:
        per_arm = {}
        for a in ARMS:
            sc = best_scores(BASE / "results" / tmpl.format(arm=a), col, better)
            if sc:
                per_arm[a] = project(sc, better, a)
        if per_arm:
            proj[name] = per_arm
            sgn_of[name] = 1.0 if better == "higher" else -1.0
    missing = [n for n, _, _, _ in methods if n not in proj]
    if missing:
        print(f"\n[skip] NO TEST 1 TABLES FOR: {', '.join(missing)}")
        print("[skip] Those methods are ABSENT from every table below. If one was "
              "expected,\n[skip] check the result file name and location.")

    rng = np.random.default_rng(args.seed)
    tprs = defaultdict(list)
    fprs = defaultdict(list)
    diffs = defaultdict(list)
    thrs = defaultdict(list)
    per_class = defaultdict(list)
    strata_tpr = defaultdict(list)
    strata_n = {}
    n_cal_seen = {}

    # counts per unit, per derived class -- static, so precompute
    der_total = {a: {c: np.bincount(owner[a][der_cls[a] == c],
                                    minlength=n_units[a])
                     for c in DERIVED} for a in ARMS}

    for _ in range(args.replicates):
        # ONE shared resample per replicate, reused by every method: that is
        # what makes the differences paired.
        draw_q = {a: rng.integers(0, n_units[a], n_units[a]) for a in ARMS}
        draw_n = {a: {c: rng.integers(0, len(negs[a][c]), len(negs[a][c]))
                      for c in neg_classes[a]} for a in ARMS}

        per_method = {}
        for name, per_arm in proj.items():
            # --- refit the threshold inside this replicate -------------------
            if POOLED:
                # every arm, every bin: one operating point for all methods
                parts = [per_arm[a][2][args.negative][draw_n[a][args.negative]]
                         for a in ARMS
                         if a in per_arm and args.negative in neg_classes[a]]
                if not parts:
                    continue
                cal = np.concatenate(parts)
            else:
                if (calib_arm not in per_arm
                        or args.negative not in neg_classes[calib_arm]):
                    continue
                _pv, _dv, nv = per_arm[calib_arm]
                d = draw_n[calib_arm][args.negative]
                m = calib_mask[calib_arm][args.negative][d]
                cal = nv[args.negative][d][m]
            n_cal = int(cal.size)
            if not n_cal:
                continue
            n_cal_seen.setdefault(name, []).append(n_cal)
            if neg_w is not None:
                wparts = [neg_w[a][draw_n[a][args.negative]] for a in ARMS
                          if a in per_arm and args.negative in neg_classes[a]]
                wcal = np.concatenate(wparts)
                o = np.argsort(-cal, kind="stable")
                cw = np.cumsum(wcal[o]) / max(wcal.sum(), 1e-12)
                # same convention as the unweighted rule: with equal weights
                # this returns index int(target * n), i.e. identical output.
                k = int(np.searchsorted(cw, args.target_fpr + 1e-12, side="right"))
                thr = cal[o][k] if k < cal.size else cal[o][-1]
                acc = cal >= thr
                wfprs[name].append(float(wcal[acc].sum() / max(wcal.sum(), 1e-12)))
            else:
                k = int(args.target_fpr * n_cal)
                ordered = np.sort(cal)[::-1]
                thr = ordered[k] if k < ordered.size else ordered[-1]
            thrs[name].append(float(thr))

            # --- TPR over resampled positive units ---------------------------
            tp = npos = 0
            s_tp = np.zeros(len(STRATA)); s_n = np.zeros(len(STRATA))
            for a in ARMS:
                if a not in per_arm:
                    continue
                pv, _dv, _nv = per_arm[a]
                dq = draw_q[a]
                sel = is_pos[a][dq]
                npos += int(sel.sum())
                hit = (pv[dq] >= thr) & sel
                tp += int(hit.sum())
                if args.by_stratum:
                    sv = strat[a][dq]
                    for k in range(len(STRATA)):
                        m = sel & (sv == k)
                        s_n[k] += int(m.sum())
                        s_tp[k] += int((hit & (sv == k)).sum())
            if not npos:
                continue
            per_method[name] = tp / npos
            if args.by_stratum:
                for k, lab in enumerate(STRATA):
                    if s_n[k]:
                        strata_tpr[(name, lab)].append(s_tp[k] / s_n[k])
                        strata_n[(name, lab)] = int(s_n[k])

            # --- achieved FPR per class -------------------------------------
            # D2/D3 ride along with the resampled QUERY that donated them --
            # the whole point of the query-level unit. The others are their
            # own units.
            cls_fp, cls_n = defaultdict(int), defaultdict(int)
            for a in ARMS:
                if a not in per_arm:
                    continue
                _pv, dv, nv = per_arm[a]
                for c in neg_classes[a]:
                    d = draw_n[a][c]
                    cls_n[c] += d.size
                    cls_fp[c] += int((nv[c][d] >= thr).sum())
                hit_by_unit = np.bincount(owner[a][dv >= thr],
                                          minlength=n_units[a])
                dq = draw_q[a]
                for c in DERIVED:
                    cls_n[c] += int(der_total[a][c][dq].sum())
                # split the hit counts by class
                for c in DERIVED:
                    sel = der_cls[a] == c
                    hb = np.bincount(owner[a][sel & (dv >= thr)],
                                     minlength=n_units[a])
                    cls_fp[c] += int(hb[dq].sum())
            for c, n_ in cls_n.items():
                if n_:
                    per_class[(name, c)].append(cls_fp[c] / n_)
            if cls_n.get(args.negative):
                fprs[name].append(cls_fp[args.negative] / cls_n[args.negative])

        for n, v in per_method.items():
            tprs[n].append(v)
            if args.baseline in per_method:
                diffs[n].append(v - per_method[args.baseline])

    def ci(v):
        if not v:
            return None
        v = sorted(v)
        return (v[int(0.025 * len(v))], v[min(len(v) - 1, int(0.975 * len(v)))],
                sum(v) / len(v))

    print(f"\n{'method':<22}{'TPR (95% CI)':>26}{'achieved FPR':>20}"
          f"{'vs baseline':>22}")
    print("-" * 90)
    for name, _, _, _ in methods:
        if name not in tprs:
            continue
        lo, hi, mean = ci(tprs[name])
        row = f"{name:<22}{100*mean:>13.1f} [{100*lo:>5.1f},{100*hi:>5.1f}]"
        f = ci(fprs.get(name))
        row += (f"{100*f[2]:>12.2f} [{100*f[0]:.2f},{100*f[1]:.2f}]"
                if f else f"{'--':>20}")
        d = ci(diffs.get(name))
        if d and name != args.baseline:
            star = " *" if (d[0] > 0) == (d[1] > 0) else "  "
            row += f"{100*d[2]:>+11.1f} [{100*d[0]:>+5.1f},{100*d[1]:>+5.1f}]{star}"
        elif name == args.baseline:
            row += f"{'(baseline)':>22}"
        print(row)

    # --- per-stratum TPR -----------------------------------------------------
    if args.by_stratum:
        print("\n\nTPR BY LENGTH STRATUM, mean % [95% percentile interval].")
        print("Strata match 17_stratify.py. The 1-49 stratum is EXPLORATORY: the")
        print("boundary was chosen after seeing the stratified table, and it is thin.")
        hdr = f"{'method':<22}" + "".join(f"{s_:>20}" for s_ in STRATA)
        print("\n" + hdr)
        print("-" * len(hdr))
        for name, _, _, _ in methods:
            if name not in tprs:
                continue
            row = f"{name:<22}"
            for lab in STRATA:
                v = strata_tpr.get((name, lab))
                if not v:
                    row += f"{'--':>20}"
                    continue
                v = sorted(v)
                lo = v[int(0.025 * len(v))]
                hi = v[min(len(v) - 1, int(0.975 * len(v)))]
                row += f"{100*sum(v)/len(v):>8.1f}[{100*lo:.1f},{100*hi:.1f}]"
            print(row)
        ns = "  ".join(f"{lab} n~{strata_n.get((methods[0][0], lab), 0)}"
                       for lab in STRATA)
        print(f"\nmean positives per replicate:  {ns}")

    # --- the threshold's own instability, which drives everything above -----
    _v = next(iter(n_cal_seen.values()), [0])
    ncal = int(round(sum(_v) / max(len(_v), 1)))
    print(f"\nTHE REFITTED THRESHOLD ITSELF, across replicates "
          f"(calibration {args.calib_bin},")
    print(f"mean {ncal} {args.negative} negatives per replicate, so a "
          f"{args.target_fpr:.0%} target is set by the")
    rank = max(1, int(args.target_fpr * ncal))
    suffix = {1: "st", 2: "nd", 3: "rd"}.get(rank if rank < 20 else 0, "th")
    print(f"{rank}{suffix}-ranked false positive):\n")
    print("Thresholds are shown in each method's NATIVE direction (E-value for "
          "the\nalignment arms, cosine for the kNN arms).\n")
    print(f"{'method':<22}{'median thr':>14}{'95% range':>30}"
          f"{'fold / width':>14}")
    print("-" * 78)
    for name, _, _, _ in methods:
        v = thrs.get(name)
        if not v:
            continue
        g = sgn_of.get(name, 1.0)
        v = sorted(g * x for x in v)
        lo, med, hi = (v[int(0.025 * len(v))], v[len(v) // 2],
                       v[min(len(v) - 1, int(0.975 * len(v)))])
        a, b = abs(lo), abs(hi)
        # A fold range is only interpretable on an unbounded score. Cosine is
        # bounded above by 1.0 and every kNN threshold sits within 0.01 of it,
        # so a "1.0x fold range" there means the boundary is tightly located in
        # cosine units, NOT that the decision is stable -- the score density is
        # compressed against the ceiling. Report width for those and let the
        # TPR interval carry the consequence.
        if g < 0 and min(a, b) > 0:
            spread = f"{max(a, b) / min(a, b):>12.1f}x"
        else:
            spread = f"{hi - lo:>13.4f}"
        rng_s = "[" + format(lo, ".4g") + ", " + format(hi, ".4g") + "]"
        print(f"{name:<22}{med:>14.4g}{rng_s:>30}{spread:>14}")

    # --- per-class achieved FPR, with the grouping that matters -------------
    classes = [c for c in ("D0_uniform", "D1_composition", "D2_shuffle",
                           "D3_reverse", "D4_markov2", "D5_markov3",
                           args.negative)
               if any((n, c) in per_class for n, _, _, _ in methods)]
    print("\n\nACHIEVED FPR PER NEGATIVE CLASS at the transferred threshold, "
          "all bins pooled.")
    print("D2/D3 are resampled WITH their donor query; D0/D1/D4/D5 and D6 as "
          "their own units.")
    print("Cells are mean % [95% percentile interval].\n")
    head = f"{'method':<22}" + "".join(f"{c.split('_')[0]:>18}" for c in classes)
    print(head)
    print("-" * len(head))
    for name, _, _, _ in methods:
        if name not in tprs:
            continue
        row = f"{name:<22}"
        for c in classes:
            v = per_class.get((name, c))
            if not v:
                row += f"{'--':>18}"
                continue
            v = sorted(v)
            lo = v[int(0.025 * len(v))]
            hi = v[min(len(v) - 1, int(0.975 * len(v)))]
            row += f"{100*sum(v)/len(v):>8.2f}[{100*lo:.2f},{100*hi:.2f}]"
        print(row)

    if wfprs:
        print("\nLENGTH-WEIGHTED achieved D6 FPR (the quantity the weighted "
              "threshold targets):")
        for name, _, _, _ in methods:
            v = wfprs.get(name)
            if v:
                v = sorted(v)
                print(f"  {name:<22}{100*sum(v)/len(v):6.2f} "
                      f"[{100*v[int(0.025*len(v))]:.2f},"
                      f"{100*v[min(len(v)-1, int(0.975*len(v)))]:.2f}]")

    print("\n* = paired 95% percentile interval excludes zero.")
    print("Intervals are query-level bootstrap percentile intervals: the unit is")
    print("one original biological query, carrying its D2/D3 derivatives, drawn")
    print("WITH replacement, with the threshold refitted inside every replicate.")
    print("D0/D1/D4/D5 share a generative model fitted once to the reference set;")
    print("that global dependence is not modelled here.")


if __name__ == "__main__":
    main()
