#!/usr/bin/env python3
"""What do the dominant embedding directions encode?

Do the leading principal directions encode sequence length and/or amino-acid
composition (which would explain why length- and composition-matched shuffles
score so high), and what happens when those covariates are regressed out?
Test 2 (SCOPe) only: n = 11,107 reference domains.

Per model:
  1. PCA of the centred REFERENCE embeddings. For PC1..PC5 report the share of
     variance, Spearman rho with log length, with GRAVY (Kyte-Doolittle), the
     strongest single amino-acid fraction, and the R^2 of a linear model of the
     PC score on [log length + composition].
  2. The share of TOTAL embedding variance explained by those covariates.
  3. Regress the covariates out, fitting the linear map on the reference ONLY
     (as for every other transform), apply it to reference, real queries
     and D2 shuffles, and rescore with cosine kNN. Report, raw vs residualised
     (and optionally residualised then ZCA-whitened): Top-1 same-superfamily,
     TP at 1% different-fold FPR (published rule), and AUROC of real
     answerable queries' top-1 cosine against D2 shuffles' top-1 cosine.
     If the top directions carry composition, residualising should weaken the
     shuffle inversion (AUROC rises) because shuffles preserve composition.

Interpretation: an AUROC rise and a strong PC-composition correlation
support "the dominant directions encode composition"; no change refutes it.

Embeddings: data/embeddings/<model>/<stem>.n*.f16.npy + .ids.txt, as written by
05_embed.py. Writes results/pc_covariates.json and, with --write-knn, for each
model results/scope_knn_<model>_lcresid.tsv (same format as 18_center.py).

Run: python scripts/47_pc_covariates.py --models esm2_650m esmc_6b [--write-knn]
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
S = BASE / "data/scope"
AA = "ACDEFGHIKLMNPQRSTVWY"
KD = dict(A=1.8, R=-4.5, N=-3.5, D=-3.5, C=2.5, Q=-3.5, E=-3.5, G=-0.4, H=-3.2,
          I=4.5, L=3.8, K=-3.9, M=1.9, F=2.8, P=-1.6, S=-0.8, T=-0.7, W=-0.9,
          Y=-1.3, V=4.2)


def load(model, stem):
    d = BASE / "data/embeddings" / model
    hits = sorted(d.glob(f"{stem}.n*.f16.npy"))
    if not hits:
        return None, None
    if len(hits) > 1:
        raise SystemExit(f"{model}/{stem}: {len(hits)} generations; delete stale ones")
    ids = hits[0].with_name(hits[0].name.replace(".f16.npy", ".ids.txt")).read_text().split()
    X = np.load(hits[0]).astype(np.float64)
    if len(ids) != X.shape[0]:
        raise SystemExit(f"{model}/{stem}: ids/rows mismatch")
    return ids, X


def read_fasta(p):
    out, cid, buf = {}, None, []
    for line in open(p):
        if line.startswith(">"):
            if cid:
                out[cid] = "".join(buf)
            cid, buf = line[1:].split()[0], []
        else:
            buf.append(line.strip())
    if cid:
        out[cid] = "".join(buf)
    return out


def covariates(seqs):
    """[log length, 20 amino-acid fractions], GRAVY. design() drops A (the
    fractions sum to 1)."""
    Z = np.zeros((len(seqs), 1 + 20))
    g = np.zeros(len(seqs))
    for i, s in enumerate(seqs):
        L = max(len(s), 1)
        Z[i, 0] = math.log(L)
        for c in s:
            j = AA.find(c)
            if j >= 0:
                Z[i, 1 + j] += 1.0 / L
        g[i] = sum(KD.get(c, 0.0) for c in s) / L
    return Z, g


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    den = math.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den else float("nan")


def design(Z):
    """Intercept + log length + 19 composition columns (drop A: collinear)."""
    return np.hstack([np.ones((Z.shape[0], 1)), Z[:, [0]], Z[:, 2:]])


def unit(X):
    return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)


def topk(Q, R, k=50, chunk=2048):
    """-> (idx [n,k], sims [n,k]) sorted by descending cosine."""
    I = np.zeros((Q.shape[0], k), dtype=np.int64)
    V = np.zeros((Q.shape[0], k))
    for s in range(0, Q.shape[0], chunk):
        sims = Q[s:s + chunk] @ R.T
        idx = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        rows = np.arange(idx.shape[0])[:, None]
        o = np.argsort(-sims[rows, idx], axis=1)
        I[s:s + chunk] = idx[rows, o]
        V[s:s + chunk] = sims[rows, idx[rows, o]]
    return I, V


def zca(Xref):
    mu = Xref.mean(0, keepdims=True)
    C = Xref - mu
    w, V = np.linalg.eigh((C.T @ C) / max(1, C.shape[0] - 1))
    w = np.clip(w, 1e-8, None)
    W = V @ np.diag(w ** -0.5) @ V.T
    return lambda X: (X - mu) @ W


def score(tag, ref_ids, R, real_ids, Qr, dec, qry, ref, target=0.01):
    """Top-1, TP@1% different-fold FPR (published rule), AUROC real vs D2."""
    I, V = topk(unit(Qr), unit(R), k=1)
    items, top1 = [], 0
    n_ans = sum(1 for q in qry.values() if q["answerable"])
    real_s = []
    for qi, acc in enumerate(real_ids):
        q = qry.get(acc)
        t = ref.get(ref_ids[I[qi, 0]])
        if q is None or t is None or not q["answerable"]:
            continue
        s = float(V[qi, 0])
        real_s.append(s)
        o = 1 if q["superfamily"] == t["superfamily"] else (0 if q["fold"] == t["fold"] else -1)
        top1 += o == 1
        items.append((s, o))
    arr = sorted(items, key=lambda x: -x[0])
    nF = sum(1 for _, o in arr if o == -1)
    thr, fp = arr[0][0], 0
    for s, o in arr:
        if o == -1:
            fp += 1
            if fp / nF > target:
                break
        thr = s
    tp = sum(1 for s, o in items if s >= thr and o == 1)
    auroc = float("nan")
    if dec is not None and len(dec):
        _, Vd = topk(unit(dec), unit(R), k=1)
        d = Vd[:, 0]
        r = np.asarray(real_s)
        allv = np.concatenate([r, d])
        ranks = np.argsort(np.argsort(allv)).astype(float) + 1
        auroc = float((ranks[:len(r)].sum() - len(r) * (len(r) + 1) / 2) / (len(r) * len(d)))
    return {"variant": tag, "top1": round(100 * top1 / n_ans, 2),
            "tp_at_1pct": round(100 * tp / n_ans, 2), "auroc_real_vs_D2": round(auroc, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--n-pcs", type=int, default=5)
    ap.add_argument("--write-knn", action="store_true",
                    help="also write results/scope_knn_<model>_lcresid.tsv (top-50)")
    ap.add_argument("--out", default="results/pc_covariates.json")
    args = ap.parse_args()

    qry = {r["id"]: r for r in map(json.loads, open(S / "scope_query.jsonl"))}
    ref = {r["id"]: r for r in map(json.loads, open(S / "scope_reference.jsonl"))}
    seq = {i: r["seq"] for i, r in list(qry.items()) + list(ref.items())}
    dec_seq = {}
    for k in range(1, 6):
        p = S / f"scope_D2_shuffle_seed{k}.fasta"
        if p.exists():
            dec_seq[f"scope_D2_shuffle_seed{k}"] = read_fasta(p)

    results = []
    for m in args.models:
        ref_ids, Xr = load(m, "scope_reference")
        real_ids, Xq = load(m, "scope_real")
        if Xr is None or Xq is None:
            print(f"[skip] {m}: no SCOPe embeddings")
            continue
        dec_X, dec_s = [], []
        for stem, sq in dec_seq.items():
            ids, X = load(m, stem)
            if X is None:
                continue
            miss = [i for i in ids if i not in sq]
            if miss:
                raise SystemExit(f"{m}/{stem}: {len(miss)} ids not in FASTA, e.g. {miss[0]}")
            dec_X.append(X); dec_s += [sq[i] for i in ids]
        Xd = np.vstack(dec_X) if dec_X else None

        Zr, gr = covariates([seq[i] for i in ref_ids])
        Zq, _ = covariates([seq[i] for i in real_ids])
        Zd = covariates(dec_s)[0] if Xd is not None else None

        # --- 1. what the leading directions track ------------------------------
        mu = Xr.mean(0, keepdims=True)
        C = Xr - mu
        U, sv, Vt = np.linalg.svd(C, full_matrices=False)
        var = sv ** 2 / (sv ** 2).sum()
        P = C @ Vt[:args.n_pcs].T
        D = design(Zr)
        pcs = []
        for j in range(min(args.n_pcs, P.shape[1])):
            y = P[:, j]
            beta, *_ = np.linalg.lstsq(D, y, rcond=None)
            r2 = 1 - ((y - D @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()
            aa_rho = [spearman(y, Zr[:, 1 + a]) for a in range(20)]
            ja = int(np.nanargmax(np.abs(aa_rho)))
            pcs.append({"pc": j + 1, "var_share": round(float(var[j]), 4),
                        "rho_loglen": round(spearman(y, Zr[:, 0]), 3),
                        "rho_gravy": round(spearman(y, gr), 3),
                        "top_aa": AA[ja], "rho_top_aa": round(aa_rho[ja], 3),
                        "r2_len_comp": round(float(r2), 3)})
        # --- 2. total variance explained by the covariates ---------------------
        B, *_ = np.linalg.lstsq(D, Xr, rcond=None)
        resid = Xr - D @ B
        frac = 1 - (resid - resid.mean(0)).var(0).sum() / C.var(0).sum()

        # --- 3. residualise (fit on reference only) and rescore ----------------
        f_res = lambda X, Z: X - design(Z) @ B
        Rr, Qr2 = f_res(Xr, Zr), f_res(Xq, Zq)
        Dr = f_res(Xd, Zd) if Xd is not None else None
        rows = [score("raw", ref_ids, Xr, real_ids, Xq, Xd, qry, ref),
                score("len+comp regressed out", ref_ids, Rr, real_ids, Qr2, Dr, qry, ref)]
        z = zca(Xr)
        rows.append(score("ZCA whiten (for comparison)", ref_ids, z(Xr), real_ids, z(Xq),
                          z(Xd) if Xd is not None else None, qry, ref))
        z2 = zca(Rr)
        rows.append(score("regressed out, then ZCA", ref_ids, z2(Rr), real_ids, z2(Qr2),
                          z2(Dr) if Dr is not None else None, qry, ref))

        print(f"\n=== {m}  (d={Xr.shape[1]}, n_ref={Xr.shape[0]}, "
              f"D2 decoys={0 if Xd is None else Xd.shape[0]})")
        print(f"{'PC':>3}{'var':>8}{'rho logL':>10}{'rho GRAVY':>11}{'top aa':>8}"
              f"{'rho aa':>8}{'R2 L+comp':>11}")
        for p in pcs:
            print(f"{p['pc']:>3}{100*p['var_share']:>7.1f}%{p['rho_loglen']:>10.2f}"
                  f"{p['rho_gravy']:>11.2f}{p['top_aa']:>8}{p['rho_top_aa']:>8.2f}"
                  f"{p['r2_len_comp']:>11.2f}")
        print(f"share of total embedding variance explained by length + composition: "
              f"{100*frac:.1f}%")
        print(f"{'variant':<30}{'Top-1':>8}{'TP@1%':>8}{'AUROC vs D2':>13}")
        for r in rows:
            print(f"{r['variant']:<30}{r['top1']:>7.1f}%{r['tp_at_1pct']:>7.1f}%"
                  f"{r['auroc_real_vs_D2']:>13.3f}")
        results.append({"model": m, "pcs": pcs, "var_explained_len_comp": round(float(frac), 4),
                        "rescoring": rows})

        if args.write_knn:
            out = BASE / "results" / f"scope_knn_{m}_lcresid.tsv"
            Ru = unit(Rr)
            with open(out, "w") as fh:
                fh.write("query\ttarget\trank\tcosine\n")
                sets = [("real", real_ids, Qr2)]
                if Xd is not None:
                    off = 0
                    for stem, sq in dec_seq.items():
                        ids, _ = load(m, stem)
                        if ids is None:
                            continue
                        sets.append((stem[len("scope_"):], ids, Dr[off:off + len(ids)]))
                        off += len(ids)
                for tag, ids, Q in sets:
                    I, V = topk(unit(Q), Ru, k=50)
                    for i, qid in enumerate(ids):
                        for rnk in range(50):
                            fh.write(f"{tag}|{qid}\t{ref_ids[I[i, rnk]]}\t{rnk+1}\t{V[i, rnk]:.6f}\n")
            print(f"[knn] -> {out.relative_to(BASE)}")

    (BASE / args.out).write_text(json.dumps(results, indent=2) + "\n")
    print(f"\n-> {args.out}")
    print("Check: the 'raw' and 'ZCA whiten' rows must reproduce fig3b_ceiling.csv "
          "(TP and Top-1) for each model; if not, the vectors differ from the published run.")


if __name__ == "__main__":
    main()
