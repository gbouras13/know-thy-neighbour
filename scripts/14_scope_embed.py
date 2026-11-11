#!/usr/bin/env python3
"""Test 2 (SCOPe/ASTRAL-40 family hold-out), embedding kNN arm: embed the
SCOPe benchmark and run cosine kNN retrieval.

Imports 05_embed.py for cache configuration, node-local staging, dtype policy,
special-token masking, the ESM-C architecture guard and the batch-invariance
check, so the embedding path is identical to Test 1.

Inputs:  data/scope/scope_reference.fasta, scope_query.fasta,
         scope_D2_shuffle_seed{1..5}.fasta (from 11_scope_build.py)
Outputs: results/scope_knn_<model>.tsv  (query, target, rank, cosine)
         optionally data/embeddings/<model>/scope_<tag>.n<count>.f16.npy

Usage:
    python3 scripts/14_scope_embed.py --model esm2_650m [--save-embeddings]
"""
import argparse, importlib.util, sys
from pathlib import Path
import numpy as np, torch

BASE = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("e5", BASE / "scripts/05_embed.py")
E = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(E)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="esm2_650m", choices=list(E.MODELS))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-tokens", type=int, default=16384)
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--no-stage", action="store_true")
    ap.add_argument("--no-check-batching", action="store_true")
    ap.add_argument("--stage-dir", default=None)
    ap.add_argument("--hf-home", default=None)
    ap.add_argument("--save-embeddings", action="store_true",
                    help="also persist the matrices under data/embeddings/"
                         "<model>/ using 05_embed.py's stem.n<count> naming")
    args = ap.parse_args()

    out = BASE / f"results/scope_knn_{args.model}.tsv"
    embdir = BASE / "data/embeddings" / args.model
    # --save-embeddings persists the matrices for downstream steps that need
    # the vectors (centering, ProtTucker's projection head). A finished kNN
    # table does not count as complete if embeddings were requested and are
    # absent.
    def emb_done():
        return all(any(embdir.glob(f"{st}.n*.f16.npy"))
                   for st in ("scope_reference", "scope_real"))
    done = ((out.exists() and out.stat().st_size > 0) or
            (out.with_name(out.name + ".gz").exists()
             and out.with_name(out.name + ".gz").stat().st_size > 0))
    if done and (emb_done() or not args.save_embeddings):
        print(f"[skip] {out.name} already complete"); return
    if done:
        print(f"[note] kNN exists but embeddings are missing; "
              f"re-running to persist them")

    E.configure_hf_cache(args.hf_home)
    cache_dir = E.resolve_cache(E.MODELS[args.model]["hf"], args)
    device = E.pick_device(args.device)
    tok, model, mspec, dtype = E.load_model(args.model, device, args.fp32,
                                            cache_dir=cache_dir)
    print(f"[setup] {args.model} {device} {dtype}", flush=True)

    S = BASE / "data/scope"
    targets = [("reference", S / "scope_reference.fasta"),
               ("real", S / "scope_query.fasta")]
    targets += [(f"D2_shuffle_seed{i}", S / f"scope_D2_shuffle_seed{i}.fasta")
                for i in range(1, 6)]
    for _, p in targets:
        if not p.exists():
            raise SystemExit(f"missing {p} -- run 11_scope_build.py and "
                             f"12_scope_search.sh first")

    if not args.no_check_batching:
        _, probe = E.read_fasta(S / "scope_query.fasta")
        E.check_batching(probe[:2000], tok, model, mspec, device)

    ids_all, mats = {}, {}
    for tag, path in targets:
        ids, seqs = E.read_fasta(path)
        print(f"[embed] {tag}: {len(seqs)}", flush=True)
        try:
            mats[tag] = E.embed_all(seqs, tok, model, mspec, device,
                                    args.batch_tokens)
        except E.NonFiniteEmbeddings as exc:
            if args.fp32:
                raise
            print(f"[warn] {exc}; reloading in fp32", flush=True)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            args.fp32 = True
            tok, model, mspec, dtype = E.load_model(args.model, device, True,
                                                    cache_dir=cache_dir)
            mats[tag] = E.embed_all(seqs, tok, model, mspec, device,
                                    args.batch_tokens)
        ids_all[tag] = ids
        if args.save_embeddings:
            # Same naming convention as 05_embed.py (keyed on sequence count).
            embdir.mkdir(parents=True, exist_ok=True)
            stem = f"scope_{tag}"
            npy = embdir / f"{stem}.n{len(ids)}.f16.npy"
            if not npy.exists():
                tmp_npy = npy.with_name(npy.name + ".part")
                with open(tmp_npy, "wb") as fh_:
                    np.save(fh_, mats[tag])
                (embdir / f"{stem}.n{len(ids)}.ids.txt").write_text(
                    "\n".join(ids) + "\n")
                tmp_npy.replace(npy)          # rename last: no half-written .npy
                print(f"[save] {npy.name}", flush=True)

    ref = torch.from_numpy(mats["reference"]).to(device=device, dtype=torch.float32)
    ref = torch.nn.functional.normalize(ref, dim=1).T.contiguous()
    rid = ids_all["reference"]
    k = min(args.topk, len(rid))

    tmp = out.with_name(out.name + ".part")
    n = 0
    with tmp.open("w") as fh:
        fh.write("query\ttarget\trank\tcosine\n")
        for tag in [t for t, _ in targets if t != "reference"]:
            q = torch.from_numpy(mats[tag]).to(device=device, dtype=torch.float32)
            q = torch.nn.functional.normalize(q, dim=1)
            for s in range(0, q.shape[0], 4096):
                vals, idx = torch.topk(q[s:s + 4096] @ ref, k=k, dim=1)
                vals, idx = vals.cpu().numpy(), idx.cpu().numpy()
                for r in range(idx.shape[0]):
                    qid = f"{tag}|{ids_all[tag][s + r]}"
                    for j in range(k):
                        fh.write(f"{qid}\t{rid[idx[r, j]]}\t{j+1}\t{vals[r, j]:.6f}\n")
                        n += 1
            print(f"[knn] {tag}", flush=True)
    tmp.replace(out)
    print(f"[done] {out.relative_to(BASE)} ({n} rows)")


if __name__ == "__main__":
    sys.exit(main())
