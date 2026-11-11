#!/usr/bin/env python3
"""ProtTucker kNN retrieval from existing ProtT5 embeddings.

ProtTucker is a projection head trained contrastively on the CATH hierarchy,
applied to mean-pooled ProtT5 embeddings:

    Linear(1024, 256) -> Tanh -> Linear(256, 128)

with no normalisation; nearest neighbours are taken by Euclidean distance on
the 128-d output (as in EAT's eat.py, torch.cdist(..., p=2)). The published
checkpoint has 4 tensors of shapes (256,1024) (256,) (128,256) (128,).

Because mean-pooled ProtT5 embeddings already exist (05_embed.py,
14_scope_embed.py --save-embeddings), the head is applied directly on CPU.

Note: EAT embeds with `Rostlab/prot_t5_xl_half_uniref50-enc`; this pipeline
uses `Rostlab/prot_t5_xl_uniref50`. The former is the encoder-only,
half-precision export of the latter, so pooled vectors should agree to
rounding. 20_prottucker.sh runs the original EAT pipeline as a cross-check.

Weights: Zenodo record 14675997 (ProtTucker_ProtT5.pt), downloaded on first
use if absent. The URL hardcoded in eat.py is no longer reachable.

Inputs:  data/embeddings/prott5/<stem>.n<count>.f16.npy (+ .ids.txt)
Outputs: results/scope_knn_prottucker.tsv (--arm scope) or
         results/knn_<arm>_prottucker.tsv; columns query, target, rank, score
         (score = negated Euclidean distance, higher is better).

Usage:
    python3 scripts/21_prottucker_direct.py --arm scope
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

BASE = Path(__file__).resolve().parent.parent


class Tucker(nn.Module):
    """Same architecture as the ProtTucker head in eat.py."""

    def __init__(self):
        super().__init__()
        self.tucker = nn.Sequential(
            nn.Linear(1024, 256), nn.Tanh(), nn.Linear(256, 128))

    def forward(self, x):
        return self.tucker(x)


def load_emb(outdir, stem):
    hits = sorted(outdir.glob(f"{stem}.n*.f16.npy"))
    if not hits:
        raise SystemExit(f"no embeddings for {stem!r} in {outdir}")
    if len(hits) > 1:
        raise SystemExit(f"{stem}: {len(hits)} generations present; "
                         f"delete stale ones rather than guessing")
    npy = hits[0]
    ids = npy.with_name(npy.name.replace(".f16.npy", ".ids.txt")).read_text().split()
    mat = np.load(npy)
    if len(ids) != mat.shape[0]:
        raise SystemExit(f"{stem}: {len(ids)} ids vs {mat.shape[0]} rows")
    return ids, mat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(BASE / "tools/tucker_weights.pt"))
    ap.add_argument("--arm", default="scope",
                    help="scope | small | mid | control")
    ap.add_argument("--prefix", default="joint")
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--chunk", type=int, default=2048)
    args = ap.parse_args()

    outdir = BASE / "data/embeddings/prott5"
    if not outdir.is_dir():
        raise SystemExit(f"missing {outdir} -- run 05_embed.py / "
                         f"14_scope_embed.py with --model prott5 first")

    wpath = Path(args.weights)
    if not wpath.is_file():
        # Fetch the 4.7 MB checkpoint from the ProtTucker Zenodo deposit.
        url = ("https://zenodo.org/api/records/14675997/files/"
               "ProtTucker_ProtT5.pt/content")
        wpath.parent.mkdir(parents=True, exist_ok=True)
        try:
            import urllib.request
            print(f"[get] {wpath.name} from Zenodo 14675997")
            urllib.request.urlretrieve(url, str(wpath) + ".part")
            Path(str(wpath) + ".part").replace(wpath)
        except Exception as exc:
            raise SystemExit(
                f"ProtTucker weights not found at {wpath} and the download "
                f"failed ({exc}).\n"
                f"Compute nodes have no internet -- fetch on a login node:\n"
                f"  mkdir -p {wpath.parent}\n"
                f"  curl -fSL -o {wpath} \\\n    {url}")

    state = torch.load(args.weights, map_location="cpu", weights_only=False)
    if "state_dict" not in state:
        raise SystemExit(f"{wpath} has no state_dict -- wrong file?")
    shapes = [tuple(v.shape) for v in state["state_dict"].values()]
    if shapes != [(256, 1024), (256,), (128, 256), (128,)]:
        raise SystemExit(f"unexpected ProtTucker architecture: {shapes}")
    sd = state["state_dict"]
    model = Tucker()
    model.load_state_dict(sd)
    model.eval()
    print(f"[tucker] loaded {len(sd)} tensors from {Path(args.weights).name} "
          f"(epoch {state.get('epoch', '?')})")

    if args.arm == "scope":
        # 14_scope_embed.py tags the query set "real" -> stem scope_real.
        # Shuffle decoy stems are projected too, so ProtTucker has a shuffle
        # class on Test 2 like every other method.
        ref_stem = "scope_reference"
        q_stems = ["scope_real"] + sorted(
            {p.name.split(".n")[0]
             for p in outdir.glob("scope_D2_shuffle_seed*.f16.npy")})
        tag_of = (lambda s: "real" if s == "scope_real" else s[len("scope_"):])
        out = BASE / "results" / "scope_knn_prottucker.tsv"
    else:
        ref_stem = f"{args.prefix}_reference"
        q_stems = [f"{args.arm}_real"] + sorted(
            {p.name.split(".n")[0] for p in outdir.glob(f"{args.arm}_D*.f16.npy")})
        tag_of = (lambda s: "real" if s.endswith("_real")
                  else s[len(args.arm) + 1:])
        out = BASE / "results" / f"knn_{args.arm}_prottucker.tsv"

    if out.exists() or out.with_name(out.name + ".gz").exists():
        print(f"[skip] {out.name} exists")
        return

    @torch.no_grad()
    def project(mat):
        # Run the head in float32; the stored embeddings are float16.
        x = torch.from_numpy(mat.astype(np.float32))
        return model(x)

    ref_ids, ref_mat = load_emb(outdir, ref_stem)
    R = project(ref_mat)
    print(f"[tucker] reference {R.shape[0]} x {R.shape[1]}")

    tmp = out.with_name(out.name + ".part")
    n_rows = 0
    with tmp.open("w") as fh:
        fh.write("query\ttarget\trank\tscore\n")
        for stem in q_stems:
            q_ids, q_mat = load_emb(outdir, stem)
            Q = project(q_mat)
            tag = tag_of(stem)
            k = min(args.topk, len(ref_ids))
            for s in range(0, len(q_ids), args.chunk):
                # Euclidean, matching eat.py's torch.cdist(p=2). Negated on
                # write so the score is higher-is-better like every other
                # method; the transform is monotone, so ranks and AUROC are
                # unchanged.
                d = torch.cdist(Q[s:s + args.chunk], R, p=2)
                vals, idx = torch.topk(d, k=k, dim=1, largest=False)
                v, ix = vals.numpy(), idx.numpy()
                for r in range(ix.shape[0]):
                    qid = f"{tag}|{q_ids[s + r]}"
                    for rank in range(k):
                        fh.write(f"{qid}\t{ref_ids[ix[r, rank]]}\t{rank + 1}"
                                 f"\t{-v[r, rank]:.6f}\n")
                        n_rows += 1
            print(f"[knn] {tag}: {len(q_ids)} queries", flush=True)
    tmp.replace(out)
    print(f"[done] {out.relative_to(BASE)}  rows={n_rows}")
    print("[note] score = NEGATED Euclidean distance; use --score cosine "
          "(higher is better) downstream.")


if __name__ == "__main__":
    main()
