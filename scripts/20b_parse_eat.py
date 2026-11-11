#!/usr/bin/env python3
"""Convert EAT/ProtTucker output into this project's kNN table format.

EAT's output layout varies between releases, so no fixed column order is
assumed: the parser sniffs the delimiter and columns, and if it cannot identify
query / neighbour / distance it raises with a sample of the file rather than
emitting a wrong table.

ProtTucker ranks by Euclidean distance (lower is better). Downstream analysis
expects higher-is-better scores, so distance is negated. The transform is
monotone, so AUROC, matched-FPR thresholds and rankings are unaffected; the
values are not similarities and should not be compared numerically with the
cosine scores of other methods.

Input:  EAT result file (from 20_prottucker.sh)
Output: results/scope_knn_prottucker.tsv.gz or results/knn_<bench>_prottucker.tsv.gz
        (query, target, rank, score)

Usage:
    python3 scripts/20b_parse_eat.py --eat eat_result.txt --bench scope --qtag real
"""
import argparse
import csv
import gzip
import re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

Q_HINTS = ("query", "q_id", "qid", "query_id", "protein")
N_HINTS = ("nn", "hit", "target", "lookup", "neighbour", "neighbor", "subject")
D_HINTS = ("dist", "eat_dist", "euclidean", "score", "similarity")


def pick(header):
    """-> (query_col, target_col, dist_col) or None."""
    low = [h.strip().lower() for h in header]

    def find(hints, exclude=()):
        for i, h in enumerate(low):
            if i in exclude:
                continue
            if any(k in h for k in hints):
                return i
        return None

    q = find(Q_HINTS)
    n = find(N_HINTS, exclude={q} if q is not None else ())
    d = find(D_HINTS, exclude={i for i in (q, n) if i is not None})
    if q is not None and n is not None and d is not None:
        return q, n, d
    return None


def parse(path):
    """Yield (query, target, distance)."""
    text = Path(path).read_text().splitlines()
    text = [l for l in text if l.strip()]
    if not text:
        raise SystemExit(f"{path} is empty")

    delim = "\t" if text[0].count("\t") >= text[0].count(",") else ","
    rows = list(csv.reader(text, delimiter=delim))

    cols = pick(rows[0]) if len(rows[0]) >= 3 else None
    if cols:
        q, n, d = cols
        for r in rows[1:]:
            if len(r) <= max(q, n, d):
                continue
            try:
                yield r[q].strip(), r[n].strip(), float(r[d])
            except ValueError:
                continue
        return

    # Headerless table: assume the first float-parsable column is the distance
    # and the two preceding string columns are query and neighbour.
    if len(rows[0]) >= 3:
        d = next((i for i, v in enumerate(rows[0])
                  if _isfloat(v)), None)
        if d is not None and d >= 2:
            for r in rows:
                if len(r) <= d:
                    continue
                try:
                    yield r[0].strip(), r[d - 1].strip(), float(r[d])
                except ValueError:
                    continue
            return

    # Free-text "Query: X  NN: Y  Distance: Z". Anchored on explicit keywords
    # so incidental prose (e.g. "total hits: 5") does not match.
    pat = re.compile(
        r"quer\w*\W+(\S+).*?(?:nn|hit|target|neighbou?r)\W+(\S+)"
        r".*?([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$", re.I)
    hits = [m for m in (pat.search(l) for l in text) if m]
    # Require the pattern to explain most of the file, not a stray line.
    if len(hits) >= max(1, int(0.8 * len(text))):
        for m in hits:
            yield m.group(1), m.group(2), float(m.group(3))
        return

    sample = "\n".join(text[:5])
    raise SystemExit(
        "could not identify query / neighbour / distance columns in EAT output.\n"
        f"first lines of {path}:\n{sample}\n"
        "Extend 20b_parse_eat.py for this format rather than guessing -- a "
        "wrong column here yields a meaningless score.")


def _isfloat(v):
    try:
        float(v)
        return True
    except ValueError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eat", required=True)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--qtag", default="",
                    help="class tag to prepend when headers do not carry one")
    args = ap.parse_args()

    name = (f"scope_knn_prottucker" if args.bench == "scope"
            else f"knn_{args.bench}_prottucker")
    out = BASE / "results" / f"{name}.tsv.gz"

    rows, last, rank = 0, None, 0
    classes = set()
    with gzip.open(out, "wt") as fh:
        fh.write("query\ttarget\trank\tscore\n")
        for q, t, d in parse(args.eat):
            qid = f"{args.qtag}|{q}" if args.qtag else q
            if "|" not in qid:
                raise SystemExit(
                    f"query id {qid!r} carries no class tag; the analysis needs "
                    f"'<class>|<acc>'. Pass --qtag or tag the FASTA headers.")
            classes.add(qid.split("|", 1)[0])
            if qid != last:
                last, rank = qid, 0
            rank += 1
            # Negated distance: higher is better downstream (monotone, so
            # rank-based metrics are unchanged).
            fh.write(f"{qid}\t{t}\t{rank}\t{-d:.6f}\n")
            rows += 1

    if not rows:
        raise SystemExit("parsed 0 rows from EAT output")
    # Few rows from a large file means the format was not understood and
    # something incidental matched.
    n_lines = sum(1 for l in Path(args.eat).read_text().splitlines() if l.strip())
    if rows < 0.5 * (n_lines - 1):
        raise SystemExit(
            f"parsed only {rows} rows from {n_lines} non-empty lines in "
            f"{args.eat}. The format was probably not understood; inspect it "
            f"and extend 20b_parse_eat.py for this format.")
    print(f"[eat] {rows} rows -> {out.relative_to(BASE)}")
    print(f"[eat] query classes: {sorted(classes)}")
    print("[eat] score column is NEGATED Euclidean distance, not a similarity; "
          "use --score cosine (higher is better) downstream.")


if __name__ == "__main__":
    main()
