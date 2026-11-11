#!/usr/bin/env python3
"""Embedding kNN arm (Arm B), step two: cosine-similarity kNN retrieval.

Retrieval always returns a neighbour: unlike a search with a significance
model, there is no "nothing found" state, so abstention has to be imposed by a
similarity threshold. Nothing is thresholded here. The top-k neighbours and
their raw cosine similarities are written out and every threshold is applied
post hoc in analysis.

Inputs:  data/embeddings/<model>/*.n<count>.f16.npy (+ .ids.txt) from 05_embed.py
Outputs: results/knn_<arm>_<model>.tsv  (query, target, rank, cosine)
         results/knn_meta_<arm>_<model>.json

Usage:
    python3 scripts/06_knn.py --arm small --model esm2_650m [--device auto]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

BASE = Path(__file__).resolve().parent.parent


def pick_device(requested):
    """Resolve --device. DEFAULTS TO CPU, deliberately.

    This step is a matrix product over a reference already on disk, and CPU is
    fast enough. `torch.cuda.is_available()` reports whether a driver is
    present, not whether the card is free, so a GPU is not chosen
    automatically. Pass --device auto (or cuda) on a dedicated GPU node.
    """
    if requested not in ("auto", None):
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_device_or_cpu(t, device):
    """Move a tensor to `device`, falling back to CPU if the GPU refuses.

    A CUDA context can fail at allocation time even when the device reports as
    available (busy card, exhausted memory). The CPU result is identical, only
    slower.
    """
    # Cast to float32 on every path: the stored matrices are float16 and the
    # query blocks are float32, and the matmul needs matching dtypes.
    if device.type == "cpu":
        return t.to(dtype=torch.float32), device
    try:
        return t.to(device=device, dtype=torch.float32), device
    except RuntimeError as exc:
        print(f"[warn] {device} unusable ({type(exc).__name__}: "
              f"{str(exc).splitlines()[0]}); falling back to CPU", flush=True)
        cpu = torch.device("cpu")
        return t.to(device=cpu, dtype=torch.float32), cpu


def load(outdir, stem):
    """Load one embedding matrix by stem, resolving the .n<count> suffix.

    05_embed.py keys its output on the input sequence count so that a rebuilt
    split cannot reuse embeddings of different sequences. The count is not
    known here, so glob for it, and refuse to guess if more than one is present.
    """
    hits = sorted(outdir.glob(f"{stem}.n*.f16.npy"))
    if not hits:
        present = sorted(q.name.split(".n")[0] for q in outdir.glob("*.f16.npy"))
        raise SystemExit(
            f"no embeddings for {stem!r} in {outdir}\n"
            f"  stems present: {present or '(none)'}\n"
            f"  run 05_embed.py first, and check --prefix matches between the "
            f"two scripts.")
    if len(hits) > 1:
        names = ", ".join(h.name for h in hits)
        raise SystemExit(
            f"{stem}: {len(hits)} embedding generations present ({names}). "
            f"Delete the stale ones -- refusing to guess which matches the "
            f"current split.")
    npy = hits[0]
    idsf = npy.with_name(npy.name.replace(".f16.npy", ".ids.txt"))
    mat = np.load(npy)
    ids = idsf.read_text().split()
    if len(ids) != mat.shape[0]:
        raise SystemExit(f"{stem}: {len(ids)} ids vs {mat.shape[0]} rows")
    return ids, mat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="small", choices=["small", "mid", "control"])
    ap.add_argument("--model", default="esm2_650m")
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--device", default="cpu",
                    help="cpu (default; see pick_device), auto, cuda, mps. "
                         "Use 'auto' on a dedicated GPU node.")
    ap.add_argument("--prefix", default="joint", help="split prefix")
    args = ap.parse_args()

    outdir = BASE / "data/embeddings" / args.model
    if not outdir.exists():
        raise SystemExit(f"missing {outdir} -- run 05_embed.py first")

    device = pick_device(args.device)
    print(f"[setup] device={device} model={args.model} arm={args.arm} "
          f"prefix={args.prefix!r}", flush=True)

    ref_ids, ref_mat = load(outdir, f"{args.prefix}_reference"
                            if args.prefix else f"{args.arm}_reference")
    ref, device = to_device_or_cpu(torch.from_numpy(ref_mat), device)
    ref = torch.nn.functional.normalize(ref, dim=1)
    ref_t = ref.T.contiguous()
    print(f"[ref]   {len(ref_ids)} x {ref.shape[1]}", flush=True)

    # Stems as written by 05_embed.py, minus the .n<count>.f16 suffix.
    def stem_of(path):
        return path.name.split(".n")[0]

    stems = [f"{args.arm}_real"] + sorted(
        {stem_of(p) for p in outdir.glob(f"{args.arm}_D*.f16.npy")}
    )

    out_path = BASE / "results" / f"knn_{args.arm}_{args.model}.tsv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Skip completed outputs so the job can be resubmitted until it finishes.
    # This is safe because the write below is atomic: a killed job leaves a
    # .part file, never a truncated final one.
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"[skip]  {out_path.name} already complete", flush=True)
        return
    gz = out_path.with_name(out_path.name + ".gz")
    if gz.exists() and gz.stat().st_size > 0:
        print(f"[skip]  {gz.name} already complete", flush=True)
        return

    tmp_path = out_path.with_name(out_path.name + ".part")
    n_rows = 0

    with tmp_path.open("w") as fh:
        fh.write("query\ttarget\trank\tcosine\n")
        for stem in stems:
            tag = "real" if stem.endswith("_real") else stem[len(args.arm) + 1:]
            q_ids, q_mat = load(outdir, stem)
            k = min(args.topk, len(ref_ids))
            for start in range(0, len(q_ids), args.chunk):
                blk = torch.from_numpy(q_mat[start:start + args.chunk]).to(
                    device=device, dtype=torch.float32)
                blk = torch.nn.functional.normalize(blk, dim=1)
                sims = blk @ ref_t                       # (chunk, n_ref)
                vals, idx = torch.topk(sims, k=k, dim=1)
                vals = vals.cpu().numpy()
                idx = idx.cpu().numpy()
                for r in range(idx.shape[0]):
                    qid = f"{tag}|{q_ids[start + r]}"
                    for rank in range(k):
                        fh.write(f"{qid}\t{ref_ids[idx[r, rank]]}\t{rank + 1}\t{vals[r, rank]:.6f}\n")
                        n_rows += 1
            print(f"[knn]   {tag}: {len(q_ids)} queries", flush=True)

    tmp_path.replace(out_path)

    meta = {"arm": args.arm, "model": args.model, "topk": args.topk,
            "n_reference": len(ref_ids), "rows": n_rows,
            "note": "raw cosine similarities; no threshold applied"}
    (BASE / "results" / f"knn_meta_{args.arm}_{args.model}.json").write_text(
        json.dumps(meta, indent=2) + "\n")
    print(f"[done]  {out_path.relative_to(BASE)} ({n_rows} rows)")


if __name__ == "__main__":
    main()
