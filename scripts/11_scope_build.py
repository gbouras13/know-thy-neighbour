#!/usr/bin/env python3
"""Test 2: build the SCOPe/ASTRAL-40 family hold-out remote-homology benchmark.

Test 1 (Swiss-Prot/Pfam) defines positives by shared Pfam family, and Pfam
families are built from sequence profiles, so its ground truth is
sequence-derived and may favour sequence-based retrieval. SCOPe labels come
from expert assessment of 3D structure, giving an independent check.

Design (standard SCOP remote-homology setup)
--------------------------------------------
sccs is class.fold.superfamily.family, e.g. a.1.1.1. Homology is asserted at the
SUPERFAMILY level; different FOLD means non-homologous.

  positive   top hit in the same superfamily, DIFFERENT family
  negative   top hit in a different fold
  ambiguous  same fold, different superfamily -- folds can be analogous rather
             than homologous, so these are counted separately and excluded from
             both rates

Whole FAMILIES are held out, so a query's own family is absent from the
reference while its superfamily remains: that is what makes the task remote
rather than lookup. ASTRAL-40 is already redundancy-filtered at 40% identity.
Labels are per hit (pair), so no separate per-query answerability step is
needed beyond checking the superfamily survives in the reference.

Inputs:  ASTRAL-40 FASTA and dir.cla (downloaded to data/scope/ if absent).
Outputs: data/scope/scope_{query,reference}.{fasta,jsonl},
         results/scope_split_manifest.json.
Run:     python3 scripts/11_scope_build.py [--query-frac 0.25] [--seed 1]

Stdlib only.
"""
import argparse, json, random, re, urllib.request
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
SRC = BASE / "data/scope"
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


# Source files are small and fetched automatically if not already present.
SOURCES = {
    "astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.fa":
        "https://scop.berkeley.edu/downloads/scopeseq-2.08/"
        "astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.fa",
    "dir.cla.scope.2.08-stable.txt":
        "https://scop.berkeley.edu/downloads/parse/dir.cla.scope.2.08-stable.txt",
}


def fetch_sources():
    """Ensure the ASTRAL FASTA is present; return its path."""
    SRC.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        dest = SRC / name
        if dest.exists() and dest.stat().st_size > 0:
            print(f"[src] have {name} ({dest.stat().st_size/1e6:.1f} MB)")
            continue
        print(f"[src] downloading {name}", flush=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        try:
            urllib.request.urlretrieve(url, tmp)
            tmp.replace(dest)
        except Exception as exc:
            tmp.unlink(missing_ok=True)
            raise SystemExit(
                f"could not download {name} from {url}: {exc}\n"
                f"Fetch it manually into {SRC} and re-run.")
        print(f"[src] {name} ({dest.stat().st_size/1e6:.1f} MB)")

    hits = sorted(SRC.glob("astral-scopedom-seqres-*.fa"))
    if not hits:
        raise SystemExit(f"no ASTRAL FASTA in {SRC} after download")
    return hits[0]


def parse_astral(path):
    """(id, sccs, seq) from ASTRAL headers: '>d1dlwa_ a.1.1.1 (A:) ...'"""
    out, cid, sccs, cur = [], None, None, []
    for line in open(path):
        line = line.rstrip("\n")
        if line.startswith(">"):
            if cid:
                out.append((cid, sccs, "".join(cur)))
            parts = line[1:].split()
            cid, sccs, cur = parts[0], (parts[1] if len(parts) > 1 else ""), []
        else:
            cur.append(line.strip())
    if cid:
        out.append((cid, sccs, "".join(cur)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query-frac", type=float, default=0.25,
                    help="fraction of FAMILIES held out as queries")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--min-len", type=int, default=20)
    args = ap.parse_args()

    fa = fetch_sources()
    recs = parse_astral(fa)
    print(f"[load] {len(recs)} ASTRAL domains from {fa.name}")

    kept, drop = [], Counter()
    for cid, sccs, seq in recs:
        seq = seq.upper().replace("X", "")
        if not re.fullmatch(r"[a-z]\.\d+\.\d+\.\d+", sccs or ""):
            drop["bad_sccs"] += 1; continue
        if len(seq) < args.min_len:
            drop["too_short"] += 1; continue
        if set(seq) - STANDARD_AA:
            drop["nonstandard_aa"] += 1; continue
        c, f, sf, fam = sccs.split(".")
        kept.append({"id": cid, "sccs": sccs, "seq": seq, "length": len(seq),
                     "cls": c, "fold": f"{c}.{f}",
                     "superfamily": f"{c}.{f}.{sf}", "family": sccs})
    print(f"[filter] kept {len(kept)}  dropped {dict(drop)}")

    by_fam = defaultdict(list)
    for r in kept:
        by_fam[r["family"]].append(r)
    print(f"[groups] {len(by_fam)} families, "
          f"{len({r['superfamily'] for r in kept})} superfamilies, "
          f"{len({r['fold'] for r in kept})} folds")

    # Hold out whole families. A query's superfamily must survive in the
    # reference via a DIFFERENT family, or the query is unanswerable.
    rng = random.Random(args.seed)
    fams = sorted(by_fam)
    rng.shuffle(fams)
    qfams = set(fams[:int(round(len(fams) * args.query_frac))])

    qry = [r for f in qfams for r in by_fam[f]]
    ref = [r for f in fams if f not in qfams for r in by_fam[f]]
    ref_sf = {r["superfamily"] for r in ref}
    for r in qry:
        r["answerable"] = r["superfamily"] in ref_sf

    out = BASE / "data/scope"
    for name, rows in (("query", qry), ("reference", ref)):
        with (out / f"scope_{name}.fasta").open("w") as f_fa, \
             (out / f"scope_{name}.jsonl").open("w") as f_js:
            for r in sorted(rows, key=lambda x: x["id"]):
                f_fa.write(f">{r['id']}\n{r['seq']}\n")
                f_js.write(json.dumps(r) + "\n")

    n_ans = sum(r["answerable"] for r in qry)
    man = {"source": fa.name, "params": vars(args),
           "n_query": len(qry), "n_reference": len(ref),
           "n_query_answerable": n_ans,
           "n_families": len(by_fam), "n_query_families": len(qfams),
           "n_reference_superfamilies": len(ref_sf),
           "length": {"min": min(r["length"] for r in kept),
                      "median": sorted(r["length"] for r in kept)[len(kept)//2],
                      "max": max(r["length"] for r in kept)}}
    (BASE / "results/scope_split_manifest.json").write_text(json.dumps(man, indent=2) + "\n")
    print(json.dumps(man, indent=2))


if __name__ == "__main__":
    main()
