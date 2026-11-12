#!/usr/bin/env python3
"""Export tidy CSVs for the R figure code.

R should never parse the kNN tables: they are millions of rows, gzipped, and
several GB in total. Everything the figures need is small once aggregated, so
the split is: Python owns the data, R owns the drawing. Each CSV below is one
figure's input, already in long form.

Output: `figures/data/*.csv`. Re-runnable locally from the results tables.
Run:    python scripts/27_figure_data.py [--skip-slow]
        (--skip-slow skips the score-density export, which streams the full
        Test 1 tables.)

Sources, so every number can be traced back:
  Test 1 point estimates + intervals   results/bootstrap_test1.txt  (25_)
  Test 2 matched-FPR + AUROC           recomputed here from the kNN tables
  ceiling vs delivered                 recomputed here
  length strata                        results/stratified_recovery.txt (17_)
  hubness, kNN methods                 results/hubness.tsv (26_)
  hubness, alignment arms              results/hubness_alignment.tsv (43_)

Stdlib only.
"""

import json
import re
from pathlib import Path

from tableio import open_table

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "figures/data"
S = BASE / "data/scope"

# Display name -> (family, params in millions). Families drive the colour scale
# and give the three-tier structure (sequence / structure / embedding) of Fig. 1.
FAMILY = {
    "sequence (MMseqs2)": ("sequence", None),
    "Foldseek": ("structure", None),
    "ProstT5-Foldseek": ("structure", None),
    "ProstT5-Foldseek": ("structure", None),
    "gLM2-embed": ("embedding", 650),
    "ProtT5 kNN": ("embedding", 3000),
    "ProstT5 kNN": ("embedding", 3000),
    "ProtTucker/EAT": ("embedding", 3000),
    "ESM-2 150M kNN": ("embedding", 150),
    "ESM-2 650M kNN": ("embedding", 650),
    "ESM-2 3B kNN": ("embedding", 3000),
    "Ankh-base kNN": ("embedding", 450),
    "Ankh-large kNN": ("embedding", 1150),
    "ESM-C 300M kNN": ("embedding", 300),
    "ESM-C 600M kNN": ("embedding", 600),
    "ESM-C 6B kNN": ("embedding", 6000),
}


def smart(p):
    return open_table(p)                     # .tsv or .tsv.gz, else None


def write_csv(name, cols, rows):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name

    def cell(v):
        if v is None:
            return "NA"
        s = str(v)
        return f'"{s}"' if ("," in s or '"' in s) else s

    with path.open("w") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join(cell(r.get(c)) for c in cols) + "\n")
    print(f"  {name:<28} {len(rows):>5} rows")


# --------------------------------------------------------------------------
# Test 2 recomputation (fast: the SCOPe tables are small)
# --------------------------------------------------------------------------
def scope_curve(path, col, better, header):
    fh = smart(BASE / path)
    if fh is None:
        return None
    best = {}
    with fh:
        if header:
            fh.readline()
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) <= col:
                continue
            try:
                s = float(p[col])
            except ValueError:
                continue
            cur = best.get(p[0])
            if cur is None or (s > cur[0] if better == "higher" else s < cur[0]):
                best[p[0]] = (s, p[1])
    return best


def scope_metrics(best, qry, ref, better, n_ans):
    pts, top1 = [], 0
    for qid, (s, t) in best.items():
        tag, acc = (qid.split("|", 1) if "|" in qid else ("real", qid))
        if tag != "real":
            continue
        q, tt = qry.get(acc), ref.get(t)
        if q is None or tt is None or not q["answerable"]:
            continue
        same = q["superfamily"] == tt["superfamily"]
        top1 += same
        pts.append((s, 1, 0) if same
                   else ((s, 0, 0) if q["fold"] == tt["fold"] else (s, 0, 1)))
    if not pts:
        return None
    pts.sort(key=lambda x: -x[0] if better == "higher" else x[0])
    tp = fp = 0
    nP = sum(x[1] for x in pts)
    nF = sum(x[2] for x in pts)
    deliv, roc = 0.0, []
    for s, a, b in pts:
        tp += a
        fp += b
        roc.append((fp / max(nF, 1), tp / max(nP, 1)))
        if nF and fp / nF <= 0.01:
            deliv = tp / n_ans
    auc, px, py = 0.0, 0.0, 0.0
    for x, y in roc:
        auc += (x - px) * (y + py) / 2
        px, py = x, y
    auc += (1 - px) * (1 + py) / 2
    t1 = top1 / n_ans
    return {"top1": t1, "delivered": deliv, "auroc": auc,
            "kept": (deliv / t1 if t1 else 0.0),
            "n_pos": nP, "n_fold": nF, "n_scored": len(pts)}


# --------------------------------------------------------------------------
# Test 1 bootstrap table parser
# --------------------------------------------------------------------------
def parse_bootstrap(path):
    """-> {method: (mean, lo, hi, fpr_mean, diff, diff_lo, diff_hi, sig)}"""
    txt = Path(path).read_text().splitlines()
    try:
        i = next(i for i, l in enumerate(txt) if l.startswith("method"))
    except StopIteration:
        return {}
    out = {}
    pat = re.compile(
        r"^(?P<name>.+?)\s{2,}(?P<tpr>[\d.]+)\s+\[\s*(?P<lo>[\d.]+),\s*(?P<hi>[\d.]+)\]"
        r"\s+(?P<fpr>[\d.]+)\s+\[(?P<flo>[\d.]+),(?P<fhi>[\d.]+)\]"
        r"(?:\s+(?P<rest>.*))?$")
    for line in txt[i + 2:]:
        if not line.strip() or line.startswith(("THE ", "-", "*")):
            if line.startswith("THE "):
                break
            continue
        m = pat.match(line.rstrip())
        if not m:
            continue
        d = {"method": m.group("name").strip(),
             "tpr": float(m.group("tpr")), "lo": float(m.group("lo")),
             "hi": float(m.group("hi")), "fpr": float(m.group("fpr"))}
        rest = (m.group("rest") or "").strip()
        dm = re.match(r"([-+][\d.]+)\s+\[\s*([-+][\d.]+),\s*([-+][\d.]+)\]\s*(\*?)",
                      rest)
        if dm:
            d["diff"] = float(dm.group(1))
            d["diff_lo"] = float(dm.group(2))
            d["diff_hi"] = float(dm.group(3))
            d["sig"] = bool(dm.group(4))
        out[d["method"]] = d
    return out


def export_hubness():
    """figS2_hubness.csv, the source data for Fig. S5.

    Three producers, one schema: results/hubness.tsv (26_, the twelve kNN
    methods), results/hubness_alignment.tsv (43_, the three alignment arms) and
    results/hubness_knn_d6.tsv (43_, the kNN methods on the D6-only population).
    The alignment table carries five list-completeness columns that the kNN
    table has no reason to store -- every kNN list is exactly 50 long -- so
    they are filled in for the kNN rows here (queries = slots / 50) rather than
    left NA, and R sees one table. Either producer may be absent; what is
    missing is printed, never silently dropped.
    """
    EXTRA = ["n_queries_issued", "n_queries_listed", "n_lists_full",
             "mean_list_len", "mean_hits_before_cut"]

    def read_tsv(path):
        lines = path.read_text().strip().split("\n")
        cols = lines[0].split("\t")
        return cols, [dict(zip(cols, l.split("\t"))) for l in lines[1:]]

    hub = BASE / "results/hubness.tsv"
    if not hub.exists():
        print("  figS2_hubness.csv          [skipped: run 26_hubness.py first]")
        return
    cols, rows = read_tsv(hub)
    for r in rows:
        n = int(r["slots"]) // 50
        r.update(n_queries_issued=n, n_queries_listed=n, n_lists_full=n,
                 mean_list_len=50, mean_hits_before_cut=50)
    for name, what in (("hubness_alignment.tsv", "alignment arms"),
                       ("hubness_knn_d6.tsv", "kNN d6-only population")):
        extra = BASE / "results" / name
        if extra.exists():
            acols, arows = read_tsv(extra)
            if set(acols) != set(cols + EXTRA):
                raise SystemExit(f"{name} columns {acols} do not match {cols + EXTRA}")
            rows += arows
        else:
            print(f"  figS2_hubness.csv          [{what} absent: run 43_hubness_alignment.py]")
    write_csv("figS2_hubness.csv", cols + EXTRA, rows)


def main():
    import sys
    skip_slow = "--skip-slow" in sys.argv
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"writing to {OUT.relative_to(BASE)}/")

    qry = {r["id"]: r for r in map(json.loads, open(S / "scope_query.jsonl"))}
    ref = {r["id"]: r for r in map(json.loads, open(S / "scope_reference.jsonl"))}
    n_ans = sum(1 for q in qry.values() if q["answerable"])

    # ---- Test 2, every method and every transform -------------------------
    t2_specs = [("sequence (MMseqs2)", "results/scope_sequence.tsv", 2, "lower", False, "none"),
                ("ProstT5-Foldseek", "results/scope_structure_Cprime.tsv", 2, "lower", False, "none"),
                ("Foldseek", "results/scope_structure_C.tsv", 2, "lower", False, "none")]
    disp = {"esm2_150m": "ESM-2 150M kNN", "esm2_650m": "ESM-2 650M kNN",
            "esm2_3b": "ESM-2 3B kNN", "esmc_300m": "ESM-C 300M kNN",
            "esmc_600m": "ESM-C 600M kNN", "esmc_6b": "ESM-C 6B kNN",
            "ankh_base": "Ankh-base kNN", "ankh_large": "Ankh-large kNN",
            "prott5": "ProtT5 kNN", "prostt5": "ProstT5 kNN",
            "glm2_650m_embed": "gLM2-embed", "prottucker": "ProtTucker/EAT"}
    for k, name in disp.items():
        t2_specs.append((name, f"results/scope_knn_{k}.tsv.gz", 3, "higher", True, "none"))
        for t in ("center", "diag", "abtt", "whiten"):
            t2_specs.append((name, f"results/scope_knn_{k}_{t}.tsv.gz", 3,
                             "higher", True, t))

    t2 = []
    for name, path, col, better, header, transform in t2_specs:
        b = scope_curve(path, col, better, header)
        if not b:
            continue
        met = scope_metrics(b, qry, ref, better, n_ans)
        if not met:
            continue
        fam, params = FAMILY.get(name, ("embedding", None))
        t2.append({"method": name, "transform": transform, "family": fam,
                   "params_m": params, **{k: round(v, 6) for k, v in met.items()}})

    # ---- Fig. 1: headline, both benchmarks --------------------------------
    boot = parse_bootstrap(BASE / "results/bootstrap_test1.txt")
    # display name -> the name used inside results/bootstrap_test1.txt, which
    # labels the Foldseek arms "Foldseek C" / "Foldseek C'"
    alias = {"ProstT5-Foldseek": "Foldseek C'", "Foldseek": "Foldseek C"}
    rows = []
    for name in [s[0] for s in t2_specs if s[5] == "none"]:
        fam, params = FAMILY.get(name, ("embedding", None))
        t2r = next((r for r in t2 if r["method"] == name
                    and r["transform"] == "none"), None)
        b = boot.get(name) or boot.get(alias.get(name, name))
        if t2r:
            rows.append({"method": name, "family": fam, "benchmark": "Test 2 (SCOPe)",
                         "tpr": round(100 * t2r["delivered"], 2),
                         "lo": None, "hi": None, "auroc": round(t2r["auroc"], 4)})
        if b:
            rows.append({"method": name, "family": fam,
                         "benchmark": "Test 1 (Swiss-Prot)",
                         "tpr": b["tpr"], "lo": b["lo"], "hi": b["hi"],
                         "auroc": None})
    write_csv("fig1_headline.csv",
              ["method", "family", "benchmark", "tpr", "lo", "hi", "auroc"], rows)

    # ---- Fig. 3: ceiling vs delivered, all transforms ---------------------
    write_csv("fig3b_ceiling.csv",
              ["method", "transform", "family", "params_m", "top1", "delivered",
               "kept", "auroc", "n_pos", "n_fold", "n_scored"],
              [{**r, "top1": round(100 * r["top1"], 2),
                "delivered": round(100 * r["delivered"], 2),
                "kept": round(100 * r["kept"], 1),
                "auroc": round(r["auroc"], 4)} for r in t2])

    # ---- Fig. 2 / Fig. S1: score densities --------------------------------
    # Best score per query, real remote answerables vs D6 vs D2. Subsampled: a density needs a few thousand
    # points, not a few million, and the CSV has to stay openable.
    import random
    rng = random.Random(0)
    if skip_slow:
        print("  fig2_densities.csv         [skipped: --skip-slow]")
    # All Test 1 methods are exported (Fig. S1); Fig. 2 selects its panels in
    # R. Foldseek on real structures searched only real queries and the D6
    # naturals (synthetic decoys have no structure), so its D2 class is empty.
    dens_specs = [("sequence (MMseqs2)", "results/sequence_{arm}_vs_joint.tsv.gz", 2, "lower", False),
                  ("Foldseek", "results/structure_Cn_{arm}.tsv.gz", 2, "lower", False),
                  ("ProstT5-Foldseek", "results/structure_Cprime_{arm}_top50.tsv.gz", 2, "lower", False),
                  ("gLM2-embed", "results/knn_{arm}_glm2_650m_embed.tsv.gz", 3, "higher", True),
                  ("ProstT5 kNN", "results/knn_{arm}_prostt5.tsv.gz", 3, "higher", True),
                  ("ProtT5 kNN", "results/knn_{arm}_prott5.tsv.gz", 3, "higher", True),
                  ("ProtTucker/EAT", "results/knn_{arm}_prottucker.tsv.gz", 3, "higher", True),
                  ("Ankh-base kNN", "results/knn_{arm}_ankh_base.tsv.gz", 3, "higher", True),
                  ("Ankh-large kNN", "results/knn_{arm}_ankh_large.tsv.gz", 3, "higher", True),
                  ("ESM-2 150M kNN", "results/knn_{arm}_esm2_150m.tsv.gz", 3, "higher", True),
                  ("ESM-2 650M kNN", "results/knn_{arm}_esm2_650m.tsv.gz", 3, "higher", True),
                  ("ESM-2 3B kNN", "results/knn_{arm}_esm2_3b.tsv.gz", 3, "higher", True),
                  ("ESM-C 300M kNN", "results/knn_{arm}_esmc_300m.tsv.gz", 3, "higher", True),
                  ("ESM-C 600M kNN", "results/knn_{arm}_esmc_600m.tsv.gz", 3, "higher", True),
                  ("ESM-C 6B kNN", "results/knn_{arm}_esmc_6b.tsv.gz", 3, "higher", True)]
    remote = {}
    for arm in ("small", "mid", "control"):
        for line in open(BASE / f"data/queries/joint_{arm}_query.jsonl"):
            r = json.loads(line)
            if r.get("answerable") and r.get("remote"):
                remote[r["acc"]] = True

    CAP = 4000
    dens = []
    dens_all = []
    for name, tmpl, col, better, header in (dens_specs if not skip_slow else []):
        buckets = {"real remote homolog": [], "D6 natural hard negative": [],
                   "D2 shuffled decoy": []}
        # every class, for Fig. S2 (same pass -- these files are too big to
        # read twice). Keys are canonical class names, seeds stripped.
        buckets_all = {c: [] for c in
                       ("real", "D0 uniform", "D1 composition", "D2 shuffle",
                        "D3 reverse", "D4 markov2", "D5 markov3",
                        "D6 natural hard")}
        CANON = {"D0_uniform": "D0 uniform", "D1_composition": "D1 composition",
                 "D2_shuffle": "D2 shuffle", "D3_reverse": "D3 reverse",
                 "D4_markov2": "D4 markov2", "D5_markov3": "D5 markov3",
                 "D6_natural_hard_all": "D6 natural hard"}
        for arm in ("small", "mid", "control"):
            fh = smart(BASE / tmpl.format(arm=arm))
            if fh is None:
                continue
            best = {}
            with fh:
                if header:
                    fh.readline()
                for line in fh:
                    p_ = line.rstrip("\n").split("\t")
                    if len(p_) <= col:
                        continue
                    tag = p_[0].split("|", 1)[0]
                    # EXACT class names (seeds stripped to canonical form). The
                    # obsolete bare D6_natural_hard is excluded by construction:
                    # it is not in CANON. Do not use prefix matching here.
                    base_tag = tag.rsplit("_seed", 1)[0]
                    if not (tag == "real" or base_tag in CANON):
                        continue
                    try:
                        sc = float(p_[col])
                    except ValueError:
                        continue
                    cur = best.get(p_[0])
                    if cur is None or (sc > cur if better == "higher" else sc < cur):
                        best[p_[0]] = sc
            for qid, sc in best.items():
                tag, acc = qid.split("|", 1)
                if tag == "real":
                    if acc in remote:
                        buckets["real remote homolog"].append(sc)
                        buckets_all["real"].append(sc)
                elif tag == "D6_natural_hard_all":
                    buckets["D6 natural hard negative"].append(sc)
                    buckets_all["D6 natural hard"].append(sc)
                elif tag.startswith("D2_shuffle"):
                    buckets["D2 shuffled decoy"].append(sc)
                    buckets_all["D2 shuffle"].append(sc)
                else:
                    c = CANON.get(tag.rsplit("_seed", 1)[0])
                    if c:
                        buckets_all[c].append(sc)
        for cls, vals in buckets.items():
            if len(vals) > CAP:
                vals = rng.sample(vals, CAP)
            for v in vals:
                # MMseqs2 reports E-values that underflow to exactly 0. A log
                # axis cannot show those, so floor at 1e-200 and flag the row
                # rather than silently dropping the strongest hits.
                floored = (better == "lower" and v <= 0)
                if floored:
                    v = 1e-200
                dens.append({"method": name, "class": cls,
                             "score": v if better == "lower" else round(v, 6),
                             "floored": int(floored),
                             "scale": ("E-value" if better == "lower" else
                                       "neg. Euclidean" if name == "ProtTucker/EAT"
                                       else "cosine")})
        for cls, vals in buckets_all.items():
            if len(vals) > CAP:
                vals = rng.sample(vals, CAP)
            for v in vals:
                floored = (better == "lower" and v <= 0)
                if floored:
                    v = 1e-200
                dens_all.append({"method": name, "class": cls,
                                 "score": v if better == "lower" else round(v, 6),
                                 "floored": int(floored),
                                 "scale": ("E-value" if better == "lower" else
                                       "neg. Euclidean" if name == "ProtTucker/EAT"
                                       else "cosine")})
    if not skip_slow:
        write_csv("fig2_densities.csv",
                  ["method", "class", "scale", "score", "floored"], dens)
        write_csv("figS7_densities_all_classes.csv",
                  ["method", "class", "scale", "score", "floored"], dens_all)

    # ---- Fig. S4: length strata --------------------------------------------
    strat = BASE / "results/stratified_recovery.txt"
    if strat.exists():
        lines = strat.read_text().split("\n")
        rows4 = []
        try:
            i = next(i for i, l in enumerate(lines) if l.startswith("=== length ==="))
            hdr = lines[i + 1]
            names = [h.strip() for h in re.findall(r"\s{2,}(\S.*?)(?=\s{2,}|$)",
                                                  hdr[hdr.index("nNeg") + 4:])]
            for l in lines[i + 2:]:
                if not l.strip() or l.startswith("==="):
                    break
                m = re.match(r"^(\S+)\s+(\d+)\s+(\d+)\s+(.*)$", l)
                if not m:
                    continue
                cells = re.findall(r"([\d.]+)\s*/\s*([\d.]+)", m.group(4))
                # 17_stratify.py truncates names to the column width
                # ("sequence (MM"), so restore them rather than shipping a
                # truncated factor level into the plots.
                untrunc = {"sequence (MM": "sequence (MMseqs2)",
                           # older arm labels, accepted for compatibility
                           "Foldseek C": "Foldseek",
                           "Foldseek C'": "ProstT5-Foldseek",
                           # 12-char truncation of "ProstT5-Foldseek"
                           "ProstT5-Fold": "ProstT5-Foldseek"}
                for nm, (tpr, fpr) in zip(names, cells):
                    nm = nm.strip()
                    rows4.append({"bin": m.group(1), "n_pos": int(m.group(2)),
                                  "n_neg": int(m.group(3)),
                                  "method": untrunc.get(nm, nm),
                                  "tpr": float(tpr), "fpr": float(fpr)})
            write_csv("fig4_length.csv",
                      ["bin", "n_pos", "n_neg", "method", "tpr", "fpr"], rows4)
        except StopIteration:
            print("  fig4_length.csv            [no '=== length ===' block]")

    # ---- Fig. S5: hubness ------------------------------------------------
    export_hubness()

    print("\ndone. R reads from figures/data/.")


if __name__ == "__main__":
    main()
