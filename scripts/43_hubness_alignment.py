#!/usr/bin/env python3
"""Hubness for the alignment arms: top-50 lists from MMseqs2, Foldseek and
ProstT5-Foldseek, so hubness (Fig. S5) can be compared across every method.

Companion to 26_hubness.py, which covers the twelve kNN methods. The object is
the same: for every query a list of up to k = 50 reference entries; from the
lists, the k-occurrence N_k of every reference (Radovanovic, Nanopoulos &
Ivanovic 2010), its skew, the share of slots absorbed by the busiest 1%, and
coverage = distinct references retrieved / references searched. stats() is
imported from 26_hubness.py so the definitions stay identical.

Comparability with kNN
----------------------
Every alignment search was run with a permissive reporting threshold, so that
scores are recorded rather than applied:

    MMseqs2                     search -s 7.5 -e 1000 --max-seqs 1000  (04_, 12_)
    Foldseek, ProstT5-Foldseek  search -s 9.5 -e 1000 --max-seqs 1000  (07b_, 16b_, 41_)

A query's 50 best alignments are therefore a RANK list, like the kNN top-50,
not a significance-filtered set. "Best" means highest BIT SCORE: the order
convertalis emits for both tools, and the order used for the top-50 cut of the
Test 1 structure tables. For MMseqs2 the 50 highest bit scores are exactly the
50 lowest E-values (E is monotone in the score within a query). For Foldseek
they can differ, because its E-value also depends on query and target lengths,
so e.g. the Test 2 Foldseek rows are bit-ordered, not E-ordered. A block not in
descending bit order is sorted before the cut, and such blocks are counted.

Rows are grouped by query in every table except results/structure_Cn_{arm}
(07c_), which interleaves each real row with its D6-tagged copy. Foldseek
Test 1 is therefore read from results/structure_C_{arm}_top50.tsv.gz and the
D6 population is formed as 07c forms it: a real query whose accession is in
data/decoys/{arm}_D6_natural_hard_all.fasta contributes its list to the decoy
population as well. This mirrors the kNN tables, where the same D6 proteins
appear once tagged real| and once tagged D6_natural_hard_all|.

Two differences from kNN are reported as columns:

  1. A list can be short or empty. kNN gives every query exactly 50
     neighbours; an alignment search returns only what passed the prefilter
     and E <= 1000. Per population: n_queries_issued, n_queries_listed (>= 1
     hit), n_lists_full (exactly 50), mean_list_len, mean_hits_before_cut.
     Low coverage with many empty lists is abstention; low coverage with full
     lists is collapse onto hubs.

  2. The searched reference differs by arm, so coverage is over the database
     the arm actually searched:

       MMseqs2           Test 1  296,807 joint reference       Test 2  11,107
       Foldseek          Test 1  289,206 with an AFDB model    Test 2   9,787 with pdbstyle coordinates
       ProstT5-Foldseek  Test 1  289,206 (same AFDB refDB)     Test 2  11,107 (3Di predicted for every domain)

     Sources: results/afdb_covered_reference.txt (written by 07b) and the
     coordinate coverage of the SCOPe set (9,787 of 11,107 reference domains;
     3,720 of 4,069 queries). Test 1 Foldseek searched only queries with an
     AFDB model; n_queries_issued for Foldseek is the issued count in the
     classes it could search at all (real, D6), and its Test 2 real
     denominator is the 3,720 queries with coordinates.

Classes: `real|` is real; every other tag is decoy. Foldseek carries
D6_natural_hard_all only (a synthetic sequence has no structure) and no Test 2
shuffles. The obsolete bare D6_natural_hard tag is dropped by exact prefix if
present. Every retrieved target is checked for membership of the searched
reference, and the script fails if any is not.

A fourth population, `d6`, for a like-for-like decoy comparison
---------------------------------------------------------------
Coverage depends on how many lists there are: 3,344 D6 queries hold at most
167,200 slots and cannot reach more than 58% of 289,206 references however
dispersed they are, while 453,344 synthetic + natural decoys saturate any
reference. Foldseek's Test 1 decoys are the D6 naturals only, so every method
also gets a `d6` population (D6_natural_hard_all queries alone) on Test 1. For
the twelve kNN methods this pass streams the same kNN tables 26_hubness.py
reads (real/decoy/all stay in 26_'s output; only `d6` is written here, to
results/hubness_knn_d6.tsv). `--skip-knn` skips that pass.

Output: results/hubness_alignment.tsv (+ results/hubness_knn_d6.tsv);
27_figure_data.py stacks them under the kNN table into
figures/data/figS2_hubness.csv.
Run:    python scripts/43_hubness_alignment.py [--skip-knn]

Stdlib only.
"""
import glob
import gzip
import importlib.util
import sys
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
K = 50
D6TAG = "D6_natural_hard_all|"
KNN_MODELS = ["esm2_150m", "esm2_650m", "esm2_3b", "esmc_300m", "esmc_600m",
              "esmc_6b", "ankh_base", "ankh_large", "prott5", "prostt5",
              "glm2_650m_embed", "prottucker"]

# Definitions shared with the kNN script, imported rather than copied.
_spec = importlib.util.spec_from_file_location("hub26", BASE / "scripts/26_hubness.py")
hub26 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hub26)
stats, STALE = hub26.stats, hub26.STALE

ARMS = ("small", "mid", "control")

# Test 2 Foldseek reference: domains with usable pdbstyle coordinates. The
# Foldseek database is not part of the local results, so these counts are
# fixed constants, checked below against the distinct targets actually
# retrieved (which cannot exceed them).
N_REF_SCOPE_COORDS = 9787
N_QRY_SCOPE_COORDS = 3720


def fasta_ids(path):
    return [l[1:].split()[0] for l in open(path) if l.startswith(">")]


def count_fasta(paths):
    return sum(1 for p in paths for l in open(p) if l.startswith(">"))


def opener(p):
    p = Path(p)
    return gzip.open(p, "rt") if p.suffix == ".gz" else open(p)


class Acc:
    __slots__ = ("c", "slots", "n_listed", "n_full", "hits", "issued")

    def __init__(self):
        self.c, self.slots, self.n_listed, self.n_full, self.hits, self.issued = \
            Counter(), 0, 0, 0, 0, 0

    def add(self, other):
        self.c.update(other.c)
        self.slots += other.slots
        self.n_listed += other.n_listed
        self.n_full += other.n_full
        self.hits += other.hits
        self.issued += other.issued
        return self


def process(path, d6_accs=None):
    """One hit table -> {'real': Acc, 'decoy': Acc}, plus diagnostics.

    Rows are consumed as query blocks. Each block is cut to its K highest bit
    scores (ties broken by lower E-value) and every target in the cut list
    gets one k-occurrence. If `d6_accs` is given, a real| block whose accession
    is in it is ALSO counted into the decoy population (07c's construction of
    the Foldseek D6 negatives).
    """
    acc = {"real": Acc(), "decoy": Acc(), "d6": Acc()}
    d6_hit = set()
    diag = {"rows": 0, "stale": 0, "unordered_blocks": 0, "reappearances": 0,
            "bad_rows": 0}
    seen, cur_q, cur = set(), None, []

    def add_list(a, top, n_all):
        a.n_listed += 1
        a.slots += len(top)
        a.n_full += len(top) == K
        a.hits += n_all
        for _, _, t in top:
            a.c[t] += 1

    def flush():
        if cur_q is None or not cur:
            return
        lst = cur
        if any(lst[i][0] > lst[i + 1][0] for i in range(len(lst) - 1)):
            diag["unordered_blocks"] += 1
            lst = sorted(lst)
        top = lst[:K]
        is_real = cur_q.startswith("real|")
        add_list(acc["real" if is_real else "decoy"], top, len(cur))
        if cur_q.startswith(D6TAG):
            add_list(acc["d6"], top, len(cur))
        if is_real and d6_accs is not None:
            accn = cur_q.split("|", 1)[1]
            if accn in d6_accs:
                add_list(acc["decoy"], top, len(cur))
                add_list(acc["d6"], top, len(cur))
                d6_hit.add(accn)

    with opener(path) as fh:
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) < 4:
                diag["bad_rows"] += 1
                continue
            q = p[0]
            if q.startswith(STALE):
                diag["stale"] += 1
                continue
            try:
                ev, bits = float(p[2]), float(p[3])
            except ValueError:            # a header line, if a table has one
                diag["bad_rows"] += 1
                continue
            diag["rows"] += 1
            if q != cur_q:
                flush()
                if q in seen:
                    diag["reappearances"] += 1
                seen.add(q)
                cur_q, cur = q, []
            # sort key: highest bit score first, then lowest E-value
            cur.append((-bits, ev, p[1]))
        flush()
    diag["d6_with_list"] = len(d6_hit)
    return acc, diag


def write_tsv(path, cols, rows):
    with path.open("w") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join(f"{r[c]:.6g}" if isinstance(r[c], float) else str(r[c])
                               for c in cols) + "\n")


COLS = ["model", "transform", "benchmark", "population", "n_ref", "slots",
        "distinct", "coverage", "skew_Nk", "top1pct_share", "max_Nk", "mean_Nk",
        "n_queries_issued", "n_queries_listed", "n_lists_full", "mean_list_len",
        "mean_hits_before_cut"]


def main():
    skip_knn = "--skip-knn" in sys.argv
    out = BASE / "results/hubness_alignment.tsv"
    out_knn = BASE / "results/hubness_knn_d6.tsv"

    # --- reference sets, for the denominators and the membership check --------
    ref_seq = set(fasta_ids(BASE / "data/reference/joint_reference.fasta"))
    ref_afdb = set(l.strip() for l in open(BASE / "results/afdb_covered_reference.txt") if l.strip())
    ref_scope = set(fasta_ids(BASE / "data/scope/scope_reference.fasta"))
    assert len(ref_seq) == 296807, len(ref_seq)
    assert len(ref_scope) == 11107, len(ref_scope)
    assert ref_afdb <= ref_seq, "AFDB-covered list is not a subset of the joint reference"
    REF = {"seq": (ref_seq, len(ref_seq)),
           "afdb": (ref_afdb, len(ref_afdb)),
           "scope": (ref_scope, len(ref_scope)),
           "scope_coords": (ref_scope, N_REF_SCOPE_COORDS)}
    print(f"[ref] joint {len(ref_seq)}  afdb-covered {len(ref_afdb)}  "
          f"scope {len(ref_scope)}  scope-with-coordinates {N_REF_SCOPE_COORDS} (fixed)")

    # --- issued queries, per arm and class group ------------------------------
    D = BASE / "data/decoys"
    issued_t1 = {}
    for arm in ARMS:
        issued_t1[arm] = {
            "real": count_fasta([BASE / f"data/queries/joint_{arm}_query.fasta"]),
            "synthetic": count_fasta(sorted(glob.glob(str(D / f"{arm}_D[0-5]_*_seed*.fasta")))),
            "d6": count_fasta([D / f"{arm}_D6_natural_hard_all.fasta"]),
        }
    S = BASE / "data/scope"
    issued_t2 = {"real": count_fasta([S / "scope_query.fasta"]),
                 "shuffle": count_fasta(sorted(glob.glob(str(S / "scope_D2_shuffle_seed*.fasta"))))}
    print(f"[issued] Test 1 {issued_t1}\n[issued] Test 2 {issued_t2}")

    SPECS = [
        {"slug": "mmseqs2",
         "t1": lambda arm: [BASE / f"results/sequence_{arm}_vs_joint.tsv.gz"],
         "t1_ref": "seq", "t1_decoy": ("synthetic", "d6"),
         "t2": [BASE / "results/scope_sequence.tsv"],
         "t2_ref": "scope", "t2_decoy": True, "t2_real_issued": issued_t2["real"]},
        {"slug": "foldseek",
         "t1": lambda arm: [BASE / f"results/structure_C_{arm}_top50.tsv.gz"],
         "t1_ref": "afdb", "t1_decoy": ("d6",), "t1_d6_from_real": True,
         "t2": [BASE / "results/scope_structure_C.tsv.gz"],
         "t2_ref": "scope_coords", "t2_decoy": False, "t2_real_issued": N_QRY_SCOPE_COORDS},
        {"slug": "prostt5_foldseek",
         "t1": lambda arm: [BASE / f"results/structure_Cprime_{arm}_top50.tsv.gz"],
         "t1_ref": "afdb", "t1_decoy": ("synthetic", "d6"),
         "t2": [BASE / "results/scope_structure_Cprime.tsv.gz",
                BASE / "results/scope_structure_Cprime_shuffles.tsv.gz"],
         "t2_ref": "scope", "t2_decoy": True, "t2_real_issued": issued_t2["real"]},
    ]

    rows, problems = [], []

    def emit(slug, bench, accs, refkey, only=None):
        refset, n_ref = REF[refkey]
        both = Acc().add(accs["real"]).add(accs["decoy"])
        pops = [("real", accs["real"]), ("decoy", accs["decoy"]), ("all", both),
                ("d6", accs["d6"])]
        if only:
            pops = [(p, a) for p, a in pops if p in only]
        for pop, a in pops:
            if not a.slots:
                continue
            outside = [t for t in a.c if t not in refset]
            if outside:
                problems.append(f"{slug} {bench} {pop}: {len(outside)} targets outside the "
                                f"reference, e.g. {outside[:3]}")
            if len(a.c) > n_ref:
                problems.append(f"{slug} {bench} {pop}: {len(a.c)} distinct targets > n_ref {n_ref}")
            st = stats(a.c, a.slots, n_ref)
            rows.append({"model": slug, "transform": "none", "benchmark": bench,
                         "population": pop, "n_ref": n_ref, **st,
                         "n_queries_issued": a.issued, "n_queries_listed": a.n_listed,
                         "n_lists_full": a.n_full,
                         "mean_list_len": a.slots / a.n_listed,
                         "mean_hits_before_cut": a.hits / a.n_listed})

    for sp in SPECS:
        slug = sp["slug"]
        # Test 1: one pass per arm, then pooled = sum of the three.
        per_arm = {}
        for arm in ARMS:
            accs = {"real": Acc(), "decoy": Acc(), "d6": Acc()}
            d6 = None
            if sp.get("t1_d6_from_real"):
                d6 = set(fasta_ids(D / f"{arm}_D6_natural_hard_all.fasta"))
            for path in sp["t1"](arm):
                if not Path(path).exists():
                    print(f"[{slug}] MISSING {Path(path).relative_to(BASE)}")
                    continue
                a, dg = process(path, d6)
                for k in accs:
                    accs[k].add(a[k])
                print(f"[{slug}] {Path(path).name}: {dg['rows']:,} rows, "
                      f"real listed {a['real'].n_listed:,}, decoy listed {a['decoy'].n_listed:,}, "
                      f"stale dropped {dg['stale']}, unordered blocks {dg['unordered_blocks']}, "
                      f"re-appearances {dg['reappearances']}, skipped rows {dg['bad_rows']}"
                      + (f", D6 with a list {dg['d6_with_list']}/{len(d6)}" if d6 else ""),
                      flush=True)
                if dg["reappearances"]:
                    problems.append(f"{slug} {Path(path).name}: {dg['reappearances']} query "
                                    f"re-appearances -- table is not grouped by query")
            accs["real"].issued = issued_t1[arm]["real"]
            accs["decoy"].issued = sum(issued_t1[arm][g] for g in sp["t1_decoy"])
            accs["d6"].issued = issued_t1[arm]["d6"]
            per_arm[arm] = accs
        pooled = {"real": Acc(), "decoy": Acc(), "d6": Acc()}
        for arm in ARMS:
            for k in pooled:
                pooled[k].add(per_arm[arm][k])
        emit(slug, "test1_pooled", pooled, sp["t1_ref"])
        emit(slug, "test1_small", per_arm["small"], sp["t1_ref"])

        # Test 2
        accs = {"real": Acc(), "decoy": Acc(), "d6": Acc()}
        for path in sp["t2"]:
            if not Path(path).exists():
                print(f"[{slug}] MISSING {Path(path).relative_to(BASE)}")
                continue
            a, dg = process(path)
            for k in accs:
                accs[k].add(a[k])
            print(f"[{slug}] {Path(path).name}: {dg['rows']:,} rows, "
                  f"real listed {a['real'].n_listed:,}, decoy listed {a['decoy'].n_listed:,}, "
                  f"stale dropped {dg['stale']}, unordered blocks {dg['unordered_blocks']}, "
                  f"re-appearances {dg['reappearances']}, skipped rows {dg['bad_rows']}",
                  flush=True)
            if dg["reappearances"]:
                problems.append(f"{slug} {Path(path).name}: {dg['reappearances']} query "
                                f"re-appearances -- table is not grouped by query")
        accs["real"].issued = sp["t2_real_issued"]
        accs["decoy"].issued = issued_t2["shuffle"] if sp["t2_decoy"] else 0
        if not sp["t2_decoy"] and accs["decoy"].slots:
            problems.append(f"{slug} test2: decoy rows present although none were expected")
        emit(slug, "test2_scope", accs, sp["t2_ref"])

    if problems:
        print("\nPROBLEMS -- table NOT written:")
        for p in problems:
            print("  " + p)
        sys.exit(1)
    write_tsv(out, COLS, rows)
    print(f"\n[hubness_alignment] {len(rows)} rows -> {out.relative_to(BASE)}\n")

    # --- the kNN methods' d6 population, Test 1 ------------------------------
    if not skip_knn:
        knn_rows, aln_rows = [], rows
        rows = knn_rows
        for m in KNN_MODELS:
            per_arm = {}
            for arm in ARMS:
                path = BASE / f"results/knn_{arm}_{m}.tsv.gz"
                if not path.exists():
                    print(f"[{m}] MISSING {path.relative_to(BASE)}")
                    continue
                a, dg = process(path)
                a["d6"].issued = issued_t1[arm]["d6"]
                per_arm[arm] = a
                print(f"[{m}] {path.name}: {dg['rows']:,} rows, d6 listed "
                      f"{a['d6'].n_listed:,}/{issued_t1[arm]['d6']}, stale dropped "
                      f"{dg['stale']}, re-appearances {dg['reappearances']}", flush=True)
                if dg["reappearances"]:
                    problems.append(f"{m} {path.name}: {dg['reappearances']} re-appearances")
            if not per_arm:
                continue
            pooled = {"real": Acc(), "decoy": Acc(), "d6": Acc()}
            for arm in per_arm:
                for k in pooled:
                    pooled[k].add(per_arm[arm][k])
            emit(m, "test1_pooled", pooled, "seq", only=("d6",))
            if "small" in per_arm:
                emit(m, "test1_small", per_arm["small"], "seq", only=("d6",))
        if problems:
            print("\nPROBLEMS -- kNN d6 table NOT written:")
            for p in problems:
                print("  " + p)
            sys.exit(1)
        write_tsv(out_knn, COLS, knn_rows)
        print(f"[hubness_knn_d6] {len(knn_rows)} rows -> {out_knn.relative_to(BASE)}\n")
        rows = aln_rows + knn_rows

    print(f"{'method':<18}{'benchmark':<14}{'pop':<7}{'coverage':>9}{'listed/issued':>16}"
          f"{'full lists':>12}{'mean len':>10}")
    print("-" * 86)
    for r in rows:
        if r["population"] == "all" or r["benchmark"] == "test1_small":
            continue
        print(f"{r['model']:<18}{r['benchmark']:<14}{r['population']:<7}"
              f"{100*r['coverage']:>8.1f}%"
              f"{r['n_queries_listed']:>8,}/{r['n_queries_issued']:<7,}"
              f"{100*r['n_lists_full']/max(r['n_queries_listed'],1):>11.1f}%"
              f"{r['mean_list_len']:>10.1f}")


if __name__ == "__main__":
    main()
