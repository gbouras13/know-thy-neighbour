#!/usr/bin/env python3
"""Covariance eigenspectrum and effective dimensionality of each embedding space.

Measures anisotropy directly on the reference set: how many dimensions each
space actually uses. Produces the numbers in Tables S2/S3.

  participation ratio  PR = (sum L)^2 / sum(L^2)   -- effective dimensions
  PR / d               fraction of the nominal dimensionality actually used
  top-k share          fraction of total variance in the leading k directions

  mean_cos_random_pair mean cosine between random pairs of raw vectors

Hakim et al. report that ProtBERT/ProtXLNet use only a handful of effective
dimensions; this applies the same measurement to the models studied here.

Inputs:  data/embeddings/<model>/<stem>.n<count>.f16.npy (from 05_embed.py or
         14_scope_embed.py --save-embeddings); "prottucker" projects ProtT5.
Output:  results/spectrum.json (one row per model)

Usage:
    python3 scripts/28_spectrum.py --models esm2_650m prott5 prottucker \
        [--stem joint_reference --out results/spectrum_test1.json]
"""
import argparse, json
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parent.parent


def apply_prottucker(X):
    """ProtTucker's 4-tensor head over ProtT5: 1024 -> 256 -> Tanh -> 128.

    Lets the contrastively trained (CATH-supervised) space be measured on the
    same axes as the general-purpose pLM spaces. Same head as in
    21_prottucker_direct.py.
    """
    import torch
    from torch import nn
    w = BASE / "tools/tucker_weights.pt"
    if not w.exists():
        raise SystemExit(f"missing {w} -- see 21_prottucker_direct.py")
    sd = torch.load(w, map_location="cpu", weights_only=False)["state_dict"]
    net = nn.Sequential(nn.Linear(1024, 256), nn.Tanh(), nn.Linear(256, 128))
    net.load_state_dict({k.replace("tucker.", ""): v for k, v in sd.items()})
    net.eval()
    with torch.no_grad():
        return net(torch.from_numpy(X.astype(np.float32))).numpy().astype(np.float64)


def load(model, stem="scope_reference"):
    # ProtTucker is a head over ProtT5: load ProtT5's vectors and project them.
    src = "prott5" if model == "prottucker" else model
    d = BASE / "data/embeddings" / src
    hits = sorted(d.glob(f"{stem}.n*.f16.npy"))
    if not hits:
        return None
    if len(hits) > 1:
        raise SystemExit(f"{src}: {len(hits)} generations; delete stale ones")
    X = np.load(hits[0]).astype(np.float64)
    return apply_prottucker(X) if model == "prottucker" else X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--out", default="results/spectrum.json")
    ap.add_argument("--stem", default="scope_reference",
                    help="embedding stem to analyse: scope_reference (Test 2, "
                         "n=11,107) or joint_reference (Test 1, n=296,807). "
                         "Anisotropy is a property of the model, so the two "
                         "should agree up to sampling; comparing them is a "
                         "consistency check on the embeddings.")
    args = ap.parse_args()

    rows = []
    for m in args.models:
        X = load(m, args.stem)
        if X is None:
            print(f"[skip] {m}: no {args.stem} vectors")
            continue
        n, d = X.shape
        mu = X.mean(0, keepdims=True)
        C = X - mu
        # Eigenvalues of the covariance, descending.
        L = np.linalg.eigvalsh((C.T @ C) / max(1, n - 1))[::-1]
        L = np.clip(L, 0, None)
        tot = L.sum()
        pr = (tot ** 2) / (L ** 2).sum()          # participation ratio
        # How many leading directions to reach 50% / 90% of the variance.
        cs = np.cumsum(L) / tot
        k50 = int(np.searchsorted(cs, 0.50) + 1)
        k90 = int(np.searchsorted(cs, 0.90) + 1)
        # Anisotropy on the raw vectors: mean cosine between random pairs.
        rng = np.random.default_rng(0)
        idx = rng.choice(n, size=(2, min(20000, n)), replace=True)
        U = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)
        cos_pair = float(np.mean(np.sum(U[idx[0]] * U[idx[1]], axis=1)))
        r = {"model": m, "stem": args.stem, "n": n, "d": d,
             "participation_ratio": float(pr), "pr_over_d": float(pr / d),
             "k_for_50pct_var": k50, "k_for_90pct_var": k90,
             "top1_var_share": float(L[0] / tot),
             "top10_var_share": float(L[:10].sum() / tot),
             "cond_number": float(L[0] / max(L[-1], 1e-30)),
             "mean_cos_random_pair": cos_pair}
        rows.append(r)
        print(f"{m:<18} d={d:<5} PR={pr:8.1f} ({100*pr/d:5.2f}% of d)  "
              f"k50={k50:<4} k90={k90:<5} top1={100*L[0]/tot:5.1f}%  "
              f"cos(random pair)={cos_pair:.3f}")

    out = BASE / args.out
    out.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"\n-> {out.relative_to(BASE)}")


if __name__ == "__main__":
    main()
