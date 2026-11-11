#!/usr/bin/env python3
"""Test 3: the Foldseek SCOPe all-vs-all benchmark protocol, on embeddings.

Reimplements `steineggerlab/foldseek-analysis/scopbenchmark/scripts/
bench.noselfhit.awk` exactly, so the numbers follow that protocol.

Why all-vs-all: Test 2 holds out whole families, so the family level is empty
(0 of 3,503 queries have a same-family target). The Foldseek benchmark uses no
split: every domain is queried against every other, self-hits dropped. This
reuses the vectors already computed for all 15,176 domains (reference + query
matrices) and populates all three levels.

THE PROTOCOL, transcribed from the awk
----------------------------------------
For each query, walk its hits best-first. The FIRST different-fold hit is the
false positive and stops all further counting for that query. Before it:

    famVal  = same-family hits          / (famCnt - 1)
    sfamVal = same-superfamily, diff fam / (sfamCnt - (famCnt - 1))
    foldVal = same-fold, diff superfam   / (foldCnt - (sfamCnt - 1))

The -1 terms remove the query itself. A query is reported only if it has at
least one partner at every level:

    famCnt > 1  and  sfamCnt - famCnt > 0  and  foldCnt - sfamCnt > 0

Differences from the published Foldseek run:
  * dataset  ASTRAL-40 SCOPe 2.08, 15,176 domains, vs 11,211 domains
             (scop_lookup.fix.tsv, SCOPe 2.01). Absolute values are not
             directly comparable to the published table; the protocol is.
  * arms     this script scores embedding arms only. Alignment arms are scored
             from their search output by 31_test3_score_tsv.py.
  * depth    full ranking, as in the original (not top-50 capped).

Inputs:  data/embeddings/<model>/scope_{reference,real}.n*.f16.npy (+ .ids.txt),
         data/scope/scope_{reference,query}.jsonl.
Outputs: results/test3_allvall.csv (mean sensitivity per model and level);
         optionally a per-query CSV (--per-query).
Run:     python scripts/30_test3_allvall.py --models esmc_6b esmc_6b:whiten prottucker

numpy only (torch for the ProtTucker head).
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
S = BASE / "data/scope"


def zca_from_reference(X, ids):
    """ZCA fitted on the TEST 2 REFERENCE domains only, applied to everything.

    Test 3 is all-vs-all, so there is no reference/query division to fit on.
    Test 2's reference set (11,107 domains) is reused as the fitting
    population, so the transform is a property of a fixed lookup database,
    never of the queries being scored. Raw ZCA (no shrinkage), matching the
    estimator used for Test 2.
    """
    ref_ids = {r["id"] for r in map(json.loads, open(S / "scope_reference.jsonl"))}
    fit = np.array([i in ref_ids for i in ids])
    if fit.sum() < 100:
        raise SystemExit("too few reference domains to fit the transform")
    C = X[fit].astype(np.float64)
    mu = C.mean(0, keepdims=True)
    C = C - mu
    cov = (C.T @ C) / max(1, C.shape[0] - 1)
    w, V = np.linalg.eigh(cov)
    w = np.clip(w, 1e-8, None)
    W = (V @ np.diag(w ** -0.5) @ V.T).astype(np.float32)
    print(f"  [whiten] fitted on {int(fit.sum())} reference domains, "
          f"cond={w.max()/w.min():.3g}")
    return (X.astype(np.float32) - mu.astype(np.float32)) @ W


def apply_prottucker(X):
    """The ProtTucker head over ProtT5 vectors: 1024 -> 256 -> tanh -> 128."""
    import torch
    from torch import nn
    wpath = BASE / "tools/tucker_weights.pt"
    if not wpath.exists():
        raise SystemExit(f"missing {wpath}")
    sd = torch.load(wpath, map_location="cpu", weights_only=False)["state_dict"]
    net = nn.Sequential(nn.Linear(1024, 256), nn.Tanh(), nn.Linear(256, 128))
    net.load_state_dict({k.replace("tucker.", ""): v for k, v in sd.items()})
    net.eval()
    with torch.no_grad():
        out = net(torch.from_numpy(X.astype(np.float32))).numpy()
    print(f"  [tucker] {X.shape[1]} -> {out.shape[1]}")
    return out


def load_all(model):
    """All 15,176 domain vectors, from the reference and query matrices."""
    d = BASE / "data/embeddings" / model
    ids, mats = [], []
    for stem in ("scope_reference", "scope_real"):
        hits = sorted(d.glob(f"{stem}.n*.f16.npy"))
        if not hits:
            return None, None
        if len(hits) > 1:
            raise SystemExit(f"{model}/{stem}: {len(hits)} generations present")
        npy = hits[0]
        idf = npy.with_name(npy.name.replace(".f16.npy", ".ids.txt"))
        ids += idf.read_text().split()
        mats.append(np.load(npy).astype(np.float32))
    X = np.vstack(mats)
    if len(ids) != X.shape[0]:
        raise SystemExit(f"{model}: {len(ids)} ids vs {X.shape[0]} rows")
    return ids, X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--out", default="results/test3_allvall.csv")
    ap.add_argument("--per-query", help="also write every query's sensitivity "
                    "to this CSV. The mean hides the shape -- a method can be "
                    "uniformly mediocre or bimodal (perfect on some queries, "
                    "zero on the rest) and report the same mean. Fig. S11 "
                    "plots the resulting distribution.")
    ap.add_argument("--restrict", help="file of domain ids, one per line. Applied "
                    "to BOTH the searchable set and the denominators, so every "
                    "arm is scored on an identical domain set -- see the note in "
                    "31_test3_score_tsv.py. Without it, the structure arm is "
                    "charged for true positives absent from its database.")
    args = ap.parse_args()

    # sccs for every domain, from both split files (they partition the set).
    sccs = {}
    for f in ("scope_reference.jsonl", "scope_query.jsonl"):
        for r in map(json.loads, open(S / f)):
            sccs[r["id"]] = r["sccs"]

    if args.restrict:
        keep = {l.strip() for l in open(args.restrict) if l.strip()}
        sccs = {k: v for k, v in sccs.items() if k in keep}
        print(f"[restrict] scoring over {len(sccs)} domains")

    fam = {i: s for i, s in sccs.items()}
    sfam = {i: s.rsplit(".", 1)[0] for i, s in sccs.items()}
    fold = {i: s.rsplit(".", 2)[0] for i, s in sccs.items()}
    famCnt, sfamCnt, foldCnt = defaultdict(int), defaultdict(int), defaultdict(int)
    for i in sccs:
        famCnt[fam[i]] += 1
        sfamCnt[sfam[i]] += 1
        foldCnt[fold[i]] += 1

    rows = []
    per_query = {}
    for spec in args.models:
        # "<model>" | "<model>:whiten" | "prottucker" (ProtT5 + the EAT head)
        model, _, tf = spec.partition(":")
        src = "prott5" if model == "prottucker" else model
        ids, X = load_all(src)
        if ids is None:
            print(f"[skip] {spec}: no SCOPe vectors for {src}")
            continue
        if model == "prottucker":
            X = apply_prottucker(X)
        if args.restrict:
            sel = [k for k, i in enumerate(ids) if i in sccs]
            ids = [ids[k] for k in sel]
            X = X[sel]
        if tf == "whiten":
            X = zca_from_reference(X, ids)
        elif tf:
            raise SystemExit(f"unknown transform {tf!r}")
        n = len(ids)
        # Order must match the label arrays below.
        f_ = np.array([fam[i] for i in ids])
        s_ = np.array([sfam[i] for i in ids])
        o_ = np.array([fold[i] for i in ids])
        U = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)

        acc = {"family": [], "superfamily": [], "fold": []}
        qids = []
        for lo in range(0, n, args.chunk):
            hi = min(n, lo + args.chunk)
            sim = U[lo:hi] @ U.T                       # (chunk, n)
            for r in range(hi - lo):
                qi = lo + r
                # The awk END-block filter: the query needs a partner at each level.
                if not (famCnt[f_[qi]] > 1
                        and sfamCnt[s_[qi]] - famCnt[f_[qi]] > 0
                        and foldCnt[o_[qi]] - sfamCnt[s_[qi]] > 0):
                    continue
                sim[r, qi] = -np.inf                   # drop the self-hit
                order = np.argsort(-sim[r], kind="stable")
                nf = 0
                ff = fs = fo = 0
                for j in order:
                    if sim[r, j] == -np.inf:
                        break
                    if o_[j] != o_[qi]:                # different fold => the FP
                        nf = 1
                        break                          # first FP stops counting
                    if f_[j] == f_[qi]:
                        ff += 1
                    elif s_[j] == s_[qi]:
                        fs += 1
                    else:
                        fo += 1
                if args.per_query:
                    qids.append(ids[qi])
                acc["family"].append(ff / (famCnt[f_[qi]] - 1))
                acc["superfamily"].append(
                    fs / (sfamCnt[s_[qi]] - (famCnt[f_[qi]] - 1)))
                acc["fold"].append(
                    fo / (foldCnt[o_[qi]] - (sfamCnt[s_[qi]] - 1)))
            del sim
        if args.per_query:
            per_query[spec] = {"ids": qids, **acc}
        line = f"{spec:<22}"
        for lvl in ("family", "superfamily", "fold"):
            v = acc[lvl]
            m = sum(v) / len(v) if v else float("nan")
            rows.append({"model": spec, "level": lvl, "n_queries": len(v),
                         "mean_sens": round(m, 5)})
            line += f"  {lvl[:4]}={m:.3f}"
        print(line + f"   (n={len(acc['family'])})", flush=True)

    if args.per_query:
        pq = BASE / args.per_query
        with pq.open("w") as fh:
            fh.write("model,query,level,sens\n")
            for spec, per in per_query.items():
                for lvl in ("family", "superfamily", "fold"):
                    for qid, v in zip(per["ids"], per[lvl]):
                        fh.write(f"{spec},{qid},{lvl},{v:.6f}\n")
        print(f"-> {pq.relative_to(BASE)}")

    out = BASE / args.out
    with out.open("w") as fh:
        fh.write("model,level,n_queries,mean_sens\n")
        for r in rows:
            fh.write(f"{r['model']},{r['level']},{r['n_queries']},{r['mean_sens']}\n")
    print(f"\n-> {out.relative_to(BASE)}")


if __name__ == "__main__":
    main()
