#!/usr/bin/env python3
"""Anisotropy: measure how cone-shaped an embedding space is, and recompute
the kNN after a corrective transform.

Background
----------
Nearest-neighbour cosines from mean-pooled pLM embeddings sit near 0.97-0.98
regardless of content, the signature of an anisotropic ("cone") space. Writing
v = mu + delta, with mu the large shared mean and delta the small
content-specific residual, cos(v1, v2) is dominated by |mu|^2. Cosine to
anything then largely measures how central a vector is; a featureless random
peptide lands near the mean and is close to everything, which can let decoys
out-score real proteins.

Modes:

  --diagnostic  Measures centrality directly: cos(v, mu) and ||v - mu|| for
                real queries vs each decoy class. Writes
                results/anisotropy_<arm|scope>_<model>.json.

  (default)     Applies a transform and recomputes the kNN, so the downstream
                analysis can be re-run unchanged on the corrected space.
                Writes results/knn_<arm>_<model>_<suffix>.tsv or
                results/scope_knn_<model>_<suffix>.tsv.

Transforms, in increasing aggressiveness:
  none    baseline, reproduces 06_knn
  center  v - mu
  abtt    center, then project out the top-k principal directions
          ("all-but-the-top", Mu & Viswanath) -- these usually encode
          length/frequency rather than content
  diag    center, then divide each channel by its own standard deviation
          (diagonal whitening: fixes per-channel scale, ignores correlations).
          As applied by Lu et al. (Patterns 2025) to ESMFold latents; much
          cheaper than full ZCA (no d x d matrix).
  whiten  center, then rescale by the inverse square root of the (shrunk)
          covariance; see --shrink (the shrinkage ablation is Table S4)

mu and the PCA/whitening transforms are fitted on the reference set only.
Fitting on queries and decoys would use test-set information and pull the mean
toward the decoys; reference-only also mirrors deployment, where only the
lookup database is available in advance.

Usage:
    python3 scripts/18_center.py --model esm2_650m --arm small --diagnostic
    python3 scripts/18_center.py --model esm2_650m --scope --transform whiten
    python3 scripts/18_center.py --model prottucker --scope --metric euclidean
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "scripts"))


def _tucker_head():
    """ProtTucker's projection, loaded once and applied to every stem."""
    import torch
    from torch import nn
    w = BASE / "tools/tucker_weights.pt"
    if not w.exists():
        raise SystemExit(f"missing {w} -- see 21_prottucker_direct.py")
    sd = torch.load(w, map_location="cpu", weights_only=False)["state_dict"]
    net = nn.Sequential(nn.Linear(1024, 256), nn.Tanh(), nn.Linear(256, 128))
    net.load_state_dict({k.replace("tucker.", ""): v for k, v in sd.items()})
    net.eval()

    def f(X):
        with torch.no_grad():
            return net(torch.from_numpy(X.astype(np.float32))).numpy()
    return f


def load(outdir, stem, head=None):
    """Load one embedding matrix, resolving 05_embed.py's .n<count> suffix."""
    hits = sorted(outdir.glob(f"{stem}.n*.f16.npy"))
    if not hits:
        present = sorted(q.name.split(".n")[0] for q in outdir.glob("*.f16.npy"))
        raise SystemExit(f"no embeddings for {stem!r} in {outdir}\n"
                         f"  stems present: {present or '(none)'}")
    if len(hits) > 1:
        raise SystemExit(f"{stem}: {len(hits)} generations present; "
                         f"delete stale ones rather than guessing")
    npy = hits[0]
    ids = npy.with_name(npy.name.replace(".f16.npy", ".ids.txt")).read_text().split()
    mat = np.load(npy)
    if len(ids) != mat.shape[0]:
        raise SystemExit(f"{stem}: {len(ids)} ids vs {mat.shape[0]} rows")
    return ids, (head(mat) if head is not None else mat)


def fit_transform(ref, kind, k, shrink=None):
    """Fit on the REFERENCE matrix only. Returns f(X) -> X'."""
    ref = ref.astype(np.float32)
    mu = ref.mean(axis=0, keepdims=True)
    if kind == "none":
        return (lambda X: X.astype(np.float32)), mu
    if kind == "center":
        return (lambda X: X.astype(np.float32) - mu), mu
    C = ref - mu
    if kind == "diag":
        sd = C.std(axis=0, keepdims=True)
        sd = np.clip(sd, 1e-8, None)
        return (lambda X: (X.astype(np.float32) - mu) / sd), mu
    if kind == "abtt":
        # Top-k principal directions of the CENTERED reference, projected out.
        _, _, Vt = np.linalg.svd(C, full_matrices=False)
        V = Vt[:k]                                    # (k, d)
        return (lambda X: (X.astype(np.float32) - mu)
                - ((X.astype(np.float32) - mu) @ V.T) @ V), mu
    if kind == "whiten":
        n, d = C.shape
        cov = (C.T @ C) / max(1, n - 1)
        cov = cov.astype(np.float64)

        # Shrinkage. Whitening divides by the square root of the eigenvalues,
        # so the smallest (worst-estimated) are amplified most. With few
        # samples per dimension (e.g. ESM-C 6B on SCOPe: d=2560, n=11,107,
        # 4.3 samples/dim) raw ZCA is unstable or amplifies noise.
        #
        # Ridge toward a scaled identity:
        #     cov' = (1-a)*cov + a*(trace(cov)/d)*I
        # a=0 recovers exact ZCA. The default a = d/n, capped at 0.5, is set
        # from the sample ratio rather than tuned: negligible when n >> d and
        # substantial when the estimate is thin.
        a = (min(0.5, d / max(n, 1)) if shrink is None else float(shrink))
        if a > 0:
            cov = (1.0 - a) * cov + a * (np.trace(cov) / d) * np.eye(d)
        w, V = np.linalg.eigh(cov)
        w = np.clip(w, 1e-8, None)
        cond = float(w.max() / w.min())
        print(f"[whiten] n={n} d={d} samples/dim={n/d:.1f} "
              f"shrink={a:.3f} cond={cond:.3g}")
        if not np.all(np.isfinite(w)):
            raise SystemExit("non-finite eigenvalues; raise --shrink")
        W = (V @ np.diag(w ** -0.5) @ V.T).astype(np.float32)
        if not np.all(np.isfinite(W)):
            raise SystemExit("non-finite whitening matrix; raise --shrink")
        return (lambda X: (X.astype(np.float32) - mu) @ W), mu
    raise SystemExit(f"unknown transform {kind}")


def unit(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.clip(n, 1e-12, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="esm2_650m")
    ap.add_argument("--arm", default="small")
    ap.add_argument("--prefix", default="joint")
    ap.add_argument("--transform", default="center",
                    choices=["none", "center", "diag", "abtt", "whiten"])
    ap.add_argument("--abtt-k", type=int, default=0,
                    help="components to remove; 0 => max(1, d//100)")
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--suffix", default=None,
                    help="override the output suffix. Existing outputs are "
                         "never overwritten (a re-run skips), so use this for "
                         "variants, "
                         "e.g. --transform whiten --shrink 0 --suffix whiten_raw")
    ap.add_argument("--shrink", type=float, default=None,
                    help="covariance shrinkage for --transform whiten; "
                         "default d/n capped at 0.5, 0 disables")
    ap.add_argument("--diagnostic", action="store_true",
                    help="report centrality only; write no kNN")
    ap.add_argument("--metric", default="cosine", choices=["cosine", "euclidean"],
                    help="cosine (default, L2-normalised dot product) as for "
                         "the pLM embeddings. euclidean writes NEGATED L2 "
                         "distance with no normalisation, matching eat.py and "
                         "21_prottucker_direct.py (ProtTucker's native metric). "
                         "Centering "
                         "is a mathematical no-op under Euclidean "
                         "(||(x-mu)-(y-mu)|| = ||x-y||), so that transform "
                         "equals the baseline exactly.")
    ap.add_argument("--scope", action="store_true",
                    help="Test 2 (SCOPe) instead of Test 1")
    args = ap.parse_args()

    # ProtTucker is a contrastive head over ProtT5 (1024 -> 256 -> Tanh -> 128),
    # so ProtT5's vectors are loaded and projected before fitting anything, as
    # in 28_spectrum.py and 30_test3_allvall.py.
    src_model = "prott5" if args.model == "prottucker" else args.model
    outdir = BASE / "data/embeddings" / src_model
    if not outdir.is_dir():
        raise SystemExit(f"missing {outdir} -- run 05_embed.py / "
                         f"14_scope_embed.py --save-embeddings first")
    project_head = _tucker_head() if args.model == "prottucker" else None

    if args.scope:
        # 14_scope_embed.py tags the query set "real", so the persisted stem
        # is scope_real (not scope_query, which is the FASTA name).
        ref_stem = "scope_reference"
        q_stems = ["scope_real"] + sorted(
            {p.name.split(".n")[0] for p in outdir.glob("scope_D*.f16.npy")})
        tagger = (lambda s: "real" if s == "scope_real" else s[len("scope_"):])
    else:
        ref_stem = f"{args.prefix}_reference"
        q_stems = [f"{args.arm}_real"] + sorted(
            {p.name.split(".n")[0] for p in outdir.glob(f"{args.arm}_D*.f16.npy")})
        tagger = (lambda s: "real" if s.endswith("_real")
                  else s[len(args.arm) + 1:])

    ref_ids, ref_mat = load(outdir, ref_stem, project_head)
    d = ref_mat.shape[1]
    k = args.abtt_k or max(1, d // 100)
    f, mu = fit_transform(ref_mat, args.transform, k, args.shrink)
    print(f"[fit] {args.model} d={d} n_ref={len(ref_ids)} "
          f"transform={args.transform}"
          f"{f' k={k}' if args.transform == 'abtt' else ''}  (reference-only)")

    # ---- diagnostic: how central is each class? -----------------------------
    if args.diagnostic:
        mu_u = unit(mu.astype(np.float32))[0]
        print(f"\n{'class':<24}{'n':>8}{'cos(v,mu)':>12}{'||v-mu||':>12}")
        rows = []
        for stem in [ref_stem] + q_stems:
            ids, mat = load(outdir, stem, project_head)
            X = mat.astype(np.float32)
            c = unit(X) @ mu_u
            nrm = np.linalg.norm(X - mu, axis=1)
            tag = "REFERENCE" if stem == ref_stem else tagger(stem)
            print(f"{tag:<24}{len(ids):>8}{np.median(c):>12.4f}{np.median(nrm):>12.3f}")
            rows.append({"class": tag, "n": len(ids),
                         "median_cos_to_mean": float(np.median(c)),
                         "median_dist_to_mean": float(np.median(nrm))})
        real = next((r for r in rows if r["class"] == "real"), None)
        dec = [r for r in rows if r["class"].startswith("D")]
        if real and dec:
            worst = max(dec, key=lambda r: r["median_cos_to_mean"])
            gap = worst["median_cos_to_mean"] - real["median_cos_to_mean"]
            print(f"\n[cone] most-central decoy class: {worst['class']}")
            print(f"[cone] centrality gap (decoy - real) = {gap:+.4f}")
            print("[cone] positive gap => decoys sit nearer the centroid, which "
                  "is\n       what predicts decoys out-scoring real queries.")
        tag = "scope" if args.scope else args.arm
        (BASE / "results" / f"anisotropy_{tag}_{args.model}.json").write_text(
            json.dumps(rows, indent=2) + "\n")
        return

    # ---- transformed kNN ----------------------------------------------------
    # cosine normalises; euclidean does not (eat.py scores raw L2 distance).
    euclid = args.metric == "euclidean"
    R = f(ref_mat).astype(np.float32) if euclid else unit(f(ref_mat))
    if euclid:
        Rsq = (R ** 2).sum(1)
    suffix = args.suffix or args.transform
    tag = "scope" if args.scope else args.arm
    name = (f"scope_knn_{args.model}_{suffix}" if args.scope
            else f"knn_{args.arm}_{args.model}_{suffix}")
    out = BASE / "results" / f"{name}.tsv"
    if out.exists() or out.with_suffix(".tsv.gz").exists():
        print(f"[skip] {out.name} exists")
        return

    tmp = out.with_name(out.name + ".part")
    n_rows = 0
    with tmp.open("w") as fh:
        fh.write(f"query\ttarget\trank\t{'negeuclid' if euclid else 'cosine'}\n")
        for stem in q_stems:
            q_ids, q_mat = load(outdir, stem, project_head)
            Q = f(q_mat).astype(np.float32) if euclid else unit(f(q_mat))
            t = tagger(stem)
            kk = min(args.topk, len(ref_ids))
            for s in range(0, len(q_ids), args.chunk):
                Qb = Q[s:s + args.chunk]
                if euclid:
                    # -||q-r|| via the expanded square; negated so higher is
                    # better downstream.
                    d2 = (Qb ** 2).sum(1)[:, None] - 2.0 * (Qb @ R.T) + Rsq[None, :]
                    sims = -np.sqrt(np.maximum(d2, 0.0))
                else:
                    sims = Qb @ R.T
                idx = np.argpartition(-sims, kk - 1, axis=1)[:, :kk]
                for r in range(idx.shape[0]):
                    o = idx[r][np.argsort(-sims[r, idx[r]])]
                    qid = f"{t}|{q_ids[s + r]}"
                    for rank, j in enumerate(o, 1):
                        fh.write(f"{qid}\t{ref_ids[j]}\t{rank}\t{sims[r, j]:.6f}\n")
                        n_rows += 1
            print(f"[knn] {t}: {len(q_ids)} queries", flush=True)
    tmp.replace(out)
    print(f"[done] {out.relative_to(BASE)}  rows={n_rows}")


if __name__ == "__main__":
    main()
