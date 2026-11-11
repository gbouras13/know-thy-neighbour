#!/usr/bin/env python3
"""Embedding kNN arm (Arm B): mean-pooled protein-language-model embeddings.

Embeds the reference, the real queries and every decoy FASTA with one pLM and
writes, per input, a float16 matrix plus a parallel id list:

    data/embeddings/<model>/<stem>.n<count>.f16.npy
    data/embeddings/<model>/<stem>.n<count>.ids.txt
    results/embed_manifest_<arm>_<model>.json

Mean pooling is the object of study: it is the recipe used for embedding-based
annotation transfer (goPredSim, ProtT5 kNN transfer), and its 1/L variance
scaling is one of the mechanisms under test. Special tokens and padding are
excluded from the mean so that they do not introduce a length artefact.

Runs on CUDA or ROCm GPUs (ROCm torch builds expose the torch.cuda API), MPS or
CPU. Weights are downloaded on first use.

Usage:
    python3 scripts/05_embed.py --model esm2_650m --arm all
    python3 scripts/05_embed.py --model all --smoke-test
    python3 scripts/05_embed.py --model all --download-only
"""

import argparse
import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

BASE = Path(__file__).resolve().parent.parent


def configure_hf_cache(path=None):
    """Point the Hugging Face cache somewhere with room, before transformers loads.

    On most HPC systems $HOME has a small quota, and the default
    ~/.cache/huggingface fills partway through a download (ESM-2 650M is
    ~2.6 GB, ProtT5-XL ~11 GB). The default here is inside the project
    directory.

    Must run before transformers/huggingface_hub are imported: they resolve
    their cache paths at import time. Model imports are therefore lazy, inside
    load_model().
    """
    if "transformers" in sys.modules or "huggingface_hub" in sys.modules:
        raise RuntimeError("configure_hf_cache() called after transformers import")

    root = Path(path or os.environ.get("HF_HOME") or (BASE / "data/hf_cache"))
    root.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(root)
    os.environ["HF_HUB_CACHE"] = str(root / "hub")
    # Older transformers releases read TRANSFORMERS_CACHE instead; set both.
    os.environ["TRANSFORMERS_CACHE"] = str(root / "hub")

    # Parallel chunked downloads. Only enable when the package is actually
    # present: huggingface_hub raises if the flag is set without it.
    if importlib.util.find_spec("hf_transfer") is not None:
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
        xfer = "hf_transfer on"
    else:
        xfer = "hf_transfer NOT installed (single-stream downloads)"

    # Unauthenticated Hub requests are rate-limited. Only report whether a token
    # is present; huggingface_hub reads it from the environment itself.
    tok_set = any(os.environ.get(v) for v in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"))

    free_gb = shutil.disk_usage(root).free / 1e9
    print(f"[hf] cache -> {root}  ({free_gb:.1f} GB free)", flush=True)
    print(f"[hf] {xfer}; token {'set' if tok_set else 'NOT set (rate-limited)'}",
          flush=True)
    if free_gb < 15:
        print(f"[hf] WARNING: {free_gb:.1f} GB free; ESM-2 650M needs ~2.6 GB "
              f"and ProtT5-XL ~11 GB", flush=True)
    return root

# Many protein-LM repos carry the same weights in several formats (TensorFlow,
# PyTorch .bin, safetensors). ProtT5's repo totals 33.8 GB but only ~11 GB is
# loaded, so fetch safetensors and metadata only.
_WEIGHT_PATTERNS = ["*.safetensors", "*.safetensors.index.json", "*.json",
                    "*.txt", "*.model", "*.py"]


def ensure_cached(repo_id, cache_dir, allow_all=False, revision=None):
    """Download the repo into `cache_dir` if absent. Returns the snapshot path.

    Prefers safetensors and skips the duplicate TF/.bin copies. Falls back to a
    full download if the repo has no safetensors at all -- SaProt and the older
    Ankh releases ship only pytorch_model.bin.
    """
    from huggingface_hub import snapshot_download
    # Announce before starting so a long download is distinguishable from a hang.
    print(f"[hf] fetching {repo_id}"
          f"{' @ ' + revision[:12] if revision else ' @ main (UNPINNED)'}"
          f" ... (progress on stderr)", flush=True)
    t0 = time.time()
    if allow_all:
        path = snapshot_download(repo_id, cache_dir=cache_dir, revision=revision)
    else:
        try:
            path = snapshot_download(repo_id, cache_dir=cache_dir,
                                     revision=revision,
                                     allow_patterns=_WEIGHT_PATTERNS)
            if not any(Path(path).rglob("*.safetensors")):
                raise FileNotFoundError("no safetensors in repo")
        except Exception as exc:
            print(f"[hf] {repo_id}: safetensors-only fetch insufficient "
                  f"({type(exc).__name__}); pulling full repo", flush=True)
            path = snapshot_download(repo_id, cache_dir=cache_dir,
                                     revision=revision)
    size = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
    print(f"[hf] {repo_id} cached, {size/1e9:.1f} GB "
          f"({time.time() - t0:.0f}s)", flush=True)
    return path


def pick_stage_root(explicit=None):
    """Node-local scratch to stage model weights into, or None."""
    if explicit:
        return Path(explicit)
    for env in ("TMPDIR", "LOCALDIR"):
        v = os.environ.get(env)
        if v and Path(v).is_dir():
            return Path(v) / "hf_stage"
    shm = Path("/dev/shm")
    if shm.is_dir() and os.access(shm, os.W_OK):
        return shm / f"hf_stage_{os.environ.get('USER', 'u')}"
    return None


def stage_model(repo_id, stage_root):
    """Copy a cached model to node-local storage and return the new cache root.

    safetensors loads tensors by mmap, which on a parallel filesystem such as
    Lustre becomes many small high-latency random reads and can take hours for
    a multi-GB model. Copying the directory first is one sequential read, and
    the subsequent load from local disk or RAM takes seconds.

    Returns None when staging is not possible; the caller then loads from the
    shared cache (slower, but correct).
    """
    if stage_root is None:
        return None
    folder = "models--" + repo_id.replace("/", "--")
    src = Path(os.environ["HF_HUB_CACHE"]) / folder  # shared scratch cache
    if not src.exists():
        print(f"[stage] {repo_id} not in cache yet; loading from shared cache",
              flush=True)
        return None

    dst_root = Path(stage_root) / "hub"
    dst = dst_root / folder
    if dst.exists():
        print(f"[stage] reusing staged copy at {dst_root}", flush=True)
        return str(dst_root)

    size = sum(f.stat().st_size for f in src.rglob("*") if f.is_file())
    try:
        free = shutil.disk_usage(stage_root if Path(stage_root).exists()
                                 else Path(stage_root).parent).free
    except OSError:
        free = 0
    if free < size * 1.2:
        print(f"[stage] skipping: need {size/1e9:.1f} GB, {free/1e9:.1f} GB free "
              f"at {stage_root}", flush=True)
        return None

    dst_root.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    # symlinks=False dereferences the snapshot->blob links so the staged copy
    # is self-contained and never reaches back to Lustre during the load.
    shutil.copytree(src, dst, symlinks=False)
    dt = time.time() - t0
    print(f"[stage] copied {size/1e9:.2f} GB to {dst_root} in {dt:.0f}s "
          f"({size/1e6/max(dt,1e-9):.0f} MB/s)", flush=True)
    return str(dst_root)


# ---------------------------------------------------------------------------
# CHECKPOINT REVISIONS
#
# A repo id is not a version: `facebook/esm2_t33_650M_UR50D` resolves to
# whatever `main` holds at download time. Pin a 40-hex commit sha here and every
# load (weights and tokenizer) resolves to exactly that tree. None means
# unpinned: the run still works, the manifest records it as unpinned, and
# `--require-pinned` refuses it.
#
# The HF cache stores each repo under
#     $HF_HUB_CACHE/models--<org>--<name>/snapshots/<commit-sha>/
# so `python3 scripts/05_embed.py --print-revisions` recovers the sha used on a
# machine that ran the job, checked against the embedding file mtimes (see
# mtime_verdict). An OK verdict means "not contradicted", not "proven".
#
# Filled entries are revisions that passed that check; None entries could not
# be verified.
REVISIONS = {
    "esm2_150m":       None,  # not verified
    "esm2_650m":       None,  # not verified
    "esm2_3b":         "476b639933c8baad5ad09a60ac1a87f987b656fc",
    "prott5":          None,  # not verified
    "prostt5":         None,  # not verified
    "ankh_base":       "d99cb6b966530dfc2ae96bc69d9255c2a07308b0",
    "ankh_large":      "74b371dbfa3ee0a05d32ae74df0c2e0b82d6b9a6",
    "esmc_300m":       "a59b831785f907e96e6a246b1d142bfb76df31ee",
    "esmc_600m":       "a7e82012c83126b9eedb055fea9fa84b6c02f094",
    "esmc_6b":         "45b0fa5d7fb06faefbd5e3b89bdcef35d564e79a",
    "glm2_650m_embed": None,  # not verified
}


def embedding_window(name):
    """(earliest, latest) mtime of this model's embedding matrices, or None.

    The model must have been loaded BEFORE the first matrix it produced, so a
    cache snapshot older than `earliest` is consistent with having been the one
    used, and a snapshot newer than it cannot have been.
    """
    d = BASE / "data/embeddings" / name
    if not d.is_dir():
        return None
    ts = [f.stat().st_mtime for f in d.glob("*.npy")]
    return (min(ts), max(ts)) if ts else None


def snapshot_mtime(repo_id, cache_dir, sha):
    """mtime of one snapshot directory, or None."""
    d = (Path(cache_dir) / ("models--" + repo_id.replace("/", "--"))
         / "snapshots" / sha)
    return d.stat().st_mtime if d.is_dir() else None


def mtime_verdict(name, cache_dir, sha):
    """Is the cached sha consistent with having produced this model's output?

    Returns (verdict, detail). Verdicts:
      OK          snapshot predates the first embedding written -- consistent
      RE-FETCHED  snapshot postdates it -- the cache was refreshed after the
                  run, so this sha is NOT evidence of what the job loaded
      NO-OUTPUT   no embeddings on this machine, so nothing to check against
      UNKNOWN     snapshot directory missing or unreadable
    """
    win = embedding_window(name)
    st = snapshot_mtime(MODELS[name]["hf"], cache_dir, sha) if sha else None
    if st is None:
        return "UNKNOWN", "no snapshot dir"
    if win is None:
        return "NO-OUTPUT", "no embeddings here"
    first, _last = win
    fmt = lambda t: time.strftime("%Y-%m-%d %H:%M", time.localtime(t))
    if st <= first:
        return "OK", f"snapshot {fmt(st)} <= first embedding {fmt(first)}"
    return "RE-FETCHED", f"snapshot {fmt(st)} > first embedding {fmt(first)}"


def revision_of(name):
    """Pinned commit sha for a model key, or None. Validated, not trusted."""
    rev = REVISIONS.get(name)
    if rev is None:
        return None
    rev = str(rev).strip()
    if len(rev) != 40 or any(c not in "0123456789abcdef" for c in rev.lower()):
        raise SystemExit(
            f"REVISIONS[{name!r}] = {rev!r} is not a 40-hex commit sha. "
            f"Use the full sha from the cache snapshot directory, not a tag or "
            f"a short hash -- tags move, and a short hash is ambiguous.")
    return rev.lower()


def cached_revision(repo_id, cache_dir):
    """Commit sha actually present in `cache_dir` for `repo_id`, or None.

    Reads refs/main first (what a plain unpinned download resolved to), then
    falls back to the snapshots directory when exactly one snapshot is present.
    Two or more snapshots and no ref is genuinely ambiguous, so it says so
    rather than guessing which one a past job loaded.
    """
    root = Path(cache_dir) / ("models--" + repo_id.replace("/", "--"))
    ref = root / "refs" / "main"
    if ref.is_file():
        v = ref.read_text().strip()
        if v:
            return v
    snaps = sorted(d.name for d in (root / "snapshots").glob("*") if d.is_dir())
    if len(snaps) == 1:
        return snaps[0]
    if len(snaps) > 1:
        return f"AMBIGUOUS:{','.join(snaps)}"
    return None


MODELS = {
    # --- general-purpose pLMs, mean-pooled (the recipe under test) ----------
    "esm2_150m":  {"hf": "facebook/esm2_t30_150M_UR50D", "kind": "esm",  "dim": 640},
    "esm2_650m":  {"hf": "facebook/esm2_t33_650M_UR50D", "kind": "esm",  "dim": 1280},
    "esm2_3b":    {"hf": "facebook/esm2_t36_3B_UR50D",   "kind": "esm",  "dim": 2560},
    "prott5":     {"hf": "Rostlab/prot_t5_xl_uniref50",  "kind": "t5",   "dim": 1024},
    # ProstT5's encoder in amino-acid mode; the same model that predicts the 3Di
    # strings used by ProstT5-Foldseek.
    "prostt5":    {"hf": "Rostlab/ProstT5",              "kind": "prostt5", "dim": 1024},
    "ankh_base":  {"hf": "ElnaggarLab/ankh-base",        "kind": "ankh", "dim": 768},
    "ankh_large": {"hf": "ElnaggarLab/ankh-large",       "kind": "ankh", "dim": 1536},
    # ESM-C. Repo names and dims follow biotrainer's ESMC loader. These are
    # transformers checkpoints, not the "esmc-*-2024-12" repos (bare .pth in
    # the esm SDK layout). ESM-C 6B is ~12 GB in bf16.
    #
    # Requires the Biohub transformers fork (ESM-C is not in released
    # transformers). Install into an overlay rather than over the environment:
    #   pip install --target $HOME/tfoverlay \
    #       "transformers @ git+https://github.com/Biohub/transformers.git@main"
    #   PYTHONPATH=$HOME/tfoverlay python3 scripts/05_embed.py ...
    #
    # Pooling is identical to ESM-2 (last_hidden_state, BOS/EOS stripped), which
    # the generic special-tokens-mask path already handles.
    "esmc_300m":  {"hf": "biohub/ESMC-300M", "kind": "esmc", "dim": 960},
    "esmc_600m":  {"hf": "biohub/ESMC-600M", "kind": "esmc", "dim": 1152},
    "esmc_6b":    {"hf": "biohub/ESMC-6B",   "kind": "esmc", "dim": 2560},

    # --- purpose-built retrieval embeddings ---------------------------------
    # Trained for retrieval rather than masked-token prediction; contrasts with
    # the mean-pooled general-purpose models above.
    "glm2_650m_embed": {"hf": "tattabio/gLM2_650M_embed", "kind": "glm2",
                        "dim": 512, "trust_remote_code": True},
}

# --- models not loaded by this script ---------------------------------------
#
#   ProtTucker    Contrastive head over ProtT5 trained on the CATH hierarchy.
#                 Not on HF; embedded separately (20_prottucker.sh and
#                 21_prottucker_direct.py).
#
#   TM-Vec        Small transformer head over ProtT5 per-residue embeddings
#                 (scikit-bio/tmvec on HF; needs `pip install tmvec`).
#
#   Protein-Vec   Mixture-of-experts embedding (github.com/tymor22/protein-vec);
#                 weights hosted outside HF.
#
#   SaProt        westlake-repl/SaProt_650M_AF2. Structure-aware (fused AA+3Di
#                 tokens) rather than retrieval-trained.
#
#   ESM3          EvolutionaryScale/esm3-sm-open-v1. Gated; needs the esm SDK.


def pick_device(requested):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():                       # true for ROCm builds too
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_dtype(device, kind, force_fp32):
    if force_fp32 or device.type in ("cpu", "mps"):
        return torch.float32
    # T5-family encoders (ProtT5, ProstT5, Ankh) overflow fp16 and can emit NaN
    # embeddings, so they use bfloat16, which has fp32's exponent range. gLM2
    # and ESM-C use bfloat16 as a precaution.
    if kind in ("t5", "prostt5", "ankh", "glm2", "esmc"):
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    return torch.float16


def read_fasta(path):
    ids, seqs, cur, cid = [], [], [], None
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if cid is not None:
                    ids.append(cid)
                    seqs.append("".join(cur))
                cid, cur = line[1:].split()[0], []
            else:
                cur.append(line.strip())
    if cid is not None:
        ids.append(cid)
        seqs.append("".join(cur))
    return ids, seqs


def load_model(name, device, force_fp32, cache_dir=None):
    """Load tokenizer + encoder.

    cache_dir is passed explicitly rather than left to the environment:
    huggingface_hub resolves its cache constants at import time, so once
    anything has imported it, changing HF_HUB_CACHE has no effect. Passing the
    directory per call is order-independent and therefore actually reliable.
    """
    spec = MODELS[name]
    cd = {"cache_dir": cache_dir} if cache_dir else {}
    # Every from_pretrained below splats **cd, so a pinned revision applies to
    # both weights and tokenizer (the tokenizer determines the special-token
    # mask and therefore what is mean-pooled).
    rev = revision_of(name)
    if rev:
        cd["revision"] = rev
    print(f"[load] {name} <- {spec['hf']} @ "
          f"{rev[:12] if rev else 'main (UNPINNED -- not reproducible)'}",
          flush=True)
    dtype = pick_dtype(device, spec["kind"], force_fp32)

    # Initialise on the meta device and stream weights in, rather than building
    # a full fp32 model in RAM and then casting. Needs accelerate, so it is
    # opt-in on availability rather than assumed.
    extra = {}
    if importlib.util.find_spec("accelerate") is not None:
        extra["low_cpu_mem_usage"] = True

    t0 = time.time()
    if spec["kind"] == "esm":
        from transformers import AutoTokenizer, EsmModel
        tok = AutoTokenizer.from_pretrained(spec["hf"], **cd)
        model = EsmModel.from_pretrained(spec["hf"], torch_dtype=dtype, **extra, **cd)
    elif spec["kind"] == "esmc":
        # Plain AutoTokenizer/AutoModel -- no trust_remote_code; the support
        # lives in the transformers fork, not in the repo.
        from transformers import AutoModel, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(spec["hf"], **cd)
        try:
            model = AutoModel.from_pretrained(spec["hf"], torch_dtype=dtype,
                                              **extra, **cd)
        except (ValueError, KeyError) as exc:
            import transformers as _tf
            raise SystemExit(
                f"{spec['hf']} failed to load under transformers "
                f"{_tf.__version__}: {exc}\n"
                f"ESM-C is in no released transformers. Install the Biohub fork "
                f"into an overlay and put it first on PYTHONPATH:\n"
                f"  pip install --target $HOME/tfoverlay "
                f'"transformers @ git+https://github.com/Biohub/transformers.git@main"')

        # Without the fork, AutoModel can resolve the config to a different
        # architecture that runs but emits meaningless vectors. Fail instead.
        mname, tname = type(model).__name__, type(tok).__name__
        if "ESMC" not in mname.upper() or "ESMC" not in tname.upper():
            raise SystemExit(
                f"{spec['hf']} resolved to {mname}/{tname}, not an ESM-C pair. "
                f"transformers lacks ESM-C support here -- see the install note "
                f"above. Refusing to embed with the wrong architecture.")
    elif spec["kind"] == "glm2":
        # Custom architecture loaded via AutoModel + trust_remote_code. The
        # esmc check below is unreachable in this branch (esmc is handled
        # above); embed_batch falls back to hidden_states[-1] when
        # last_hidden_state is absent.
        from transformers import AutoModel, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(spec["hf"], trust_remote_code=True, **cd)
        kw = dict(trust_remote_code=True, torch_dtype=dtype, **extra, **cd)
        if spec["kind"] == "esmc":
            kw["output_hidden_states"] = True
        model = AutoModel.from_pretrained(spec["hf"], **kw)
    elif spec["kind"] == "ankh":
        # Ankh is T5-based but uses its own tokenizer conventions, so let Auto*
        # resolve it rather than forcing T5Tokenizer.
        from transformers import AutoTokenizer, T5EncoderModel
        tok = AutoTokenizer.from_pretrained(spec["hf"], **cd)
        model = T5EncoderModel.from_pretrained(spec["hf"], torch_dtype=dtype,
                                               **extra, **cd)
    else:
        from transformers import T5EncoderModel, T5Tokenizer
        tok = T5Tokenizer.from_pretrained(spec["hf"], do_lower_case=False, legacy=True, **cd)
        model = T5EncoderModel.from_pretrained(spec["hf"], torch_dtype=dtype, **extra, **cd)
    print(f"[load] {name} weights in {time.time() - t0:.0f}s "
          f"(low_cpu_mem_usage={'low_cpu_mem_usage' in extra})", flush=True)

    t0 = time.time()
    model = model.to(device).eval()
    print(f"[load] {name} -> {device} in {time.time() - t0:.0f}s", flush=True)
    return tok, model, spec, dtype


def prep(seqs, kind):
    if kind == "t5":
        # ProtT5 expects space-separated residues, rare amino acids mapped to X.
        return [" ".join("X" if c in "UZOB" else c for c in s) for s in seqs]
    if kind == "prostt5":
        # ProstT5 is bilingual: the <AA2fold> prefix selects amino-acid mode.
        return ["<AA2fold> " + " ".join("X" if c in "UZOB" else c for c in s)
                for s in seqs]
    if kind == "ankh":
        # Pre-split residues; the tokenizer is called with is_split_into_words.
        return [list(s) for s in seqs]
    if kind == "glm2":
        # gLM2 expects the strand prefix used in its pretraining.
        return ["<+>" + s for s in seqs]
    return list(seqs)


def make_batches(seqs, batch_tokens):
    """Length-sorted batches, capped by padded token count to bound memory."""
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    batches, cur, cur_max = [], [], 0
    for i in order:
        L = len(seqs[i]) + 2
        if cur and max(cur_max, L) * (len(cur) + 1) > batch_tokens:
            batches.append(cur)
            cur, cur_max = [], 0
        cur.append(i)
        cur_max = max(cur_max, L)
    if cur:
        batches.append(cur)
    return batches


class NonFiniteEmbeddings(RuntimeError):
    """Raised when a model emits NaN/Inf, so the caller can retry in fp32."""


@torch.no_grad()
def embed_batch(raw, tok, model, spec, device):
    """Pool one batch to (B, D) float16 on CPU."""
    kw = {"return_tensors": "pt", "padding": True, "add_special_tokens": True,
          "return_special_tokens_mask": True}
    if spec["kind"] == "ankh":
        # Ankh expects pre-split residue lists, not a joined string.
        kw["is_split_into_words"] = True
    enc = tok(prep(raw, spec["kind"]), **kw)
    enc = {k: v.to(device) for k, v in enc.items()}

    special = enc.pop("special_tokens_mask", None)

    if spec["kind"] == "glm2":
        # gLM2 exposes its own pooled representation. Pass the full encoding so
        # padding is masked; fall back to input_ids alone if the custom forward
        # rejects it (check_batching detects any resulting padding leak).
        try:
            pooled = model(**enc).pooler_output
        except TypeError:
            pooled = model(enc["input_ids"]).pooler_output
    else:
        out = model(**enc)
        hidden = getattr(out, "last_hidden_state", None)
        if hidden is None:                      # e.g. *ForMaskedLM wrappers
            hidden = out.hidden_states[-1]

        # Keep real residues only. Tokenizers differ (ESM adds <cls> and <eos>,
        # T5 only </s>), so use the tokenizer's own special-tokens mask.
        mask = enc["attention_mask"].clone()
        if special is not None:
            mask = mask * (1 - special)
        if spec["kind"] == "prostt5":
            # The <AA2fold> prefix is part of the input text, not an added
            # special token, so the tokenizer does not flag it. Drop the first
            # surviving position explicitly.
            first = mask.argmax(dim=1)
            mask[torch.arange(mask.size(0), device=mask.device), first] = 0

        m = mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)

    return pooled.float().cpu().numpy().astype(np.float16)


def check_batching(seqs, tok, model, spec, device, n=8, tol=2e-2):
    """Verify batched pooling equals single-sequence pooling.

    If padding leaks into the pooled vector, the error scales with how much
    padding a batch carried, which correlates with sequence length -- the axis
    this study measures. The check guards against such a length artefact.
    """
    sample = sorted(seqs, key=len)[::max(1, len(seqs) // n)][:n]
    if len(sample) < 2:
        return
    singly = np.vstack([embed_batch([s], tok, model, spec, device) for s in sample])
    batched = embed_batch(sample, tok, model, spec, device)   # heavy padding

    a = singly.astype(np.float32)
    b = batched.astype(np.float32)
    cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)
    worst = float(cos.min())
    print(f"[check] batched vs single pooling: min cosine {worst:.6f} "
          f"over {len(sample)} sequences (len {len(sample[0])}-{len(sample[-1])})",
          flush=True)
    if worst < 1.0 - tol:
        raise RuntimeError(
            f"batching changes embeddings (min cosine {worst:.4f}). Padding is "
            f"leaking into the pooled vector, which would produce a "
            f"length-correlated artifact. Re-run with --batch-tokens set to the "
            f"longest sequence so every batch holds one sequence.")


@torch.no_grad()
def embed_all(seqs, tok, model, spec, device, batch_tokens):
    # The width is taken from the model's actual output, not from the MODELS
    # table, so a wrong table entry cannot truncate or zero-pad vectors.
    out = None
    batches = make_batches(seqs, batch_tokens)
    done, t0 = 0, time.time()

    for bi, idxs in enumerate(batches):
        pooled = embed_batch([seqs[i] for i in idxs], tok, model, spec, device)
        if out is None:
            dim = pooled.shape[1]
            if spec.get("dim") not in (None, dim):
                print(f"[warn] {spec['hf']} emits dim {dim}, table says "
                      f"{spec['dim']} -- using {dim}", flush=True)
            spec["dim"] = dim
            out = np.zeros((len(seqs), dim), dtype=np.float16)
        for row, i in enumerate(idxs):
            out[i] = pooled[row]

        done += len(idxs)
        if bi % 200 == 0 or bi == len(batches) - 1:
            print(f"    {done}/{len(seqs)}  "
                  f"{done / max(time.time() - t0, 1e-9):.0f} seq/s", flush=True)

    if not np.isfinite(out.astype(np.float32)).all():
        n_bad = int((~np.isfinite(out.astype(np.float32))).any(axis=1).sum())
        raise NonFiniteEmbeddings(
            f"{n_bad}/{len(seqs)} sequences produced non-finite embeddings")
    return out


def targets(arm, prefix="joint"):
    """(output_stem, fasta_path) for everything this arm needs embedded.

    Here "arm" is a query set (small, mid, control). The reference is shared
    across all of them under the joint split, so its output stem omits the arm
    and it is embedded once. Query and decoy stems keep the arm.
    """
    if arm == "all":
        # "all" must be expanded by the caller into concrete arm names.
        raise ValueError("targets() needs a concrete arm; expand 'all' first")
    pre = f"{prefix}_" if prefix else ""
    if prefix:
        ref_fa = BASE / "data/reference" / f"{prefix}_reference.fasta"
        ref_stem = f"{prefix}_reference"
    else:
        ref_fa = BASE / "data/reference" / f"{arm}_reference.fasta"
        ref_stem = f"{arm}_reference"

    items = [
        (ref_stem, ref_fa),
        (f"{arm}_real", BASE / "data/queries" / f"{pre}{arm}_query.fasta"),
    ]
    for p in sorted((BASE / "data/decoys").glob(f"{arm}_D*.fasta")):
        # _matched is a strict subset of _all; embedding both duplicates work.
        if p.stem.endswith("_matched"):
            continue
        items.append((p.stem, p))
    return items


def resolve_cache(repo, args, revision=None):
    """Ensure `repo` is cached, stage it node-local, and return the cache root.

    Both the smoke test and the real run use this, so both benefit from
    node-local staging.
    """
    shared = os.environ["HF_HUB_CACHE"]
    if args.no_stage:
        ensure_cached(repo, shared, revision=revision)
        return shared
    ensure_cached(repo, shared, revision=revision)
    staged = stage_model(repo, pick_stage_root(args.stage_dir))
    return staged or shared


def main():
    ap = argparse.ArgumentParser()
    # --arm all runs every query set in one invocation, so the model is loaded
    # once.
    ap.add_argument("--arm", default="small",
                    choices=["small", "mid", "control", "all"])
    ap.add_argument("--model", default="esm2_650m",
                    choices=list(MODELS) + ["all"])
    ap.add_argument("--download-only", action="store_true",
                    help="populate the weight cache and exit without using a "
                         "GPU (e.g. on a login node; the full roster is ~60 GB).")
    ap.add_argument("--smoke-test", action="store_true",
                    help="load the model, embed a handful of real sequences, "
                         "report dim/throughput/finiteness, and exit.")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-tokens", type=int, default=16384)
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--prefix", default="joint", help="split prefix")
    ap.add_argument("--no-check-batching", action="store_true",
                    help="skip the batched-vs-single pooling equivalence check")
    ap.add_argument("--stage-dir",
                    help="node-local dir to copy weights into before loading. "
                         "Defaults to $TMPDIR or /dev/shm. Loading safetensors "
                         "by mmap from a parallel filesystem is much slower.")
    ap.add_argument("--no-stage", action="store_true",
                    help="load straight from the shared cache")
    ap.add_argument("--hf-home",
                    help="Hugging Face cache root. Defaults to $HF_HOME, else "
                         "<project>/data/hf_cache. Keep this off $HOME on HPC.")
    ap.add_argument("--print-revisions", action="store_true",
                    help="report the commit sha each model resolves to in the "
                         "local HF cache, checked against embedding mtimes, "
                         "and exit. Paste OK shas into REVISIONS.")
    ap.add_argument("--require-pinned", action="store_true",
                    help="refuse to embed a model whose REVISIONS entry is "
                         "None.")
    args = ap.parse_args()

    # Before any transformers import -- see configure_hf_cache().
    configure_hf_cache(args.hf_home)

    if args.print_revisions:
        shared = os.environ["HF_HUB_CACHE"]
        print(f"# HF cache:   {shared}")
        print(f"# Embeddings: {BASE / 'data/embeddings'}")
        print("#")
        print("# Each sha below is checked against this machine's own output:")
        print("#   OK          snapshot predates the first embedding written,")
        print("#               so it is consistent with having been the weights used")
        print("#   RE-FETCHED  snapshot is NEWER than the embeddings -- the cache was")
        print("#               refreshed after the run. NOT evidence. Do not pin it.")
        print("#   NO-OUTPUT   no embeddings on this machine; nothing to check against")
        print("#")
        print("# Only paste OK lines into REVISIONS. Caveats that this check cannot")
        print("# see: snapshot_download leaves mtimes untouched when files are already")
        print("# present, and any rsync/cp of either tree rewrites mtimes. So OK means")
        print("# 'not contradicted', not 'proven'.")
        print("REVISIONS = {")
        summary = []
        for name in MODELS:
            got = cached_revision(MODELS[name]["hf"], shared)
            pin = REVISIONS.get(name)
            if got is None:
                val, note, verdict = "None", "  # NOT CACHED on this machine", "UNCACHED"
            elif str(got).startswith("AMBIGUOUS"):
                val, note, verdict = "None", f"  # {got} -- resolve by hand", "AMBIGUOUS"
            else:
                verdict, detail = mtime_verdict(name, shared, got)
                if verdict == "OK":
                    val, note = f'"{got}"', ""
                else:
                    val, note = "None", f"  # {verdict}: {detail}"
                if pin not in (None, got):
                    note += f"  # DIFFERS from current pin {pin}"
            summary.append((name, verdict))
            key = f'"{name}":'
            print(f"    {key:<22}{val},{note}")
        print("}")
        print("\n# verdicts:")
        for name, v in summary:
            print(f"#   {name:<18} {v}")
        ok = sum(1 for _, v in summary if v == "OK")
        print(f"#\n# {ok}/{len(summary)} usable from this machine. Run this on every "
              f"machine that\n# produced embeddings and merge the OK lines.")
        return

    if args.require_pinned:
        unpinned = [n for n in ([args.model] if args.model != "all" else list(MODELS))
                    if revision_of(n) is None]
        if unpinned:
            raise SystemExit(
                "--require-pinned: no REVISIONS entry for "
                + ", ".join(unpinned)
                + ".\nRun --print-revisions on the machine that holds the cache "
                  "and paste the shas in.")

    if args.download_only:
        # Smallest first, so auth or disk problems surface quickly.
        order = ["glm2_650m_embed", "esm2_150m", "ankh_base", "esm2_650m",
                 "ankh_large", "prostt5", "prott5", "esm2_3b", "esmc_6b"]
        names = ([n for n in order if n in MODELS] +
                 [n for n in MODELS if n not in order]) \
            if args.model == "all" else [args.model]
        total = 0
        for name in names:
            try:
                path = ensure_cached(MODELS[name]["hf"], os.environ["HF_HUB_CACHE"],
                                     revision=revision_of(name))
                total += sum(f.stat().st_size for f in Path(path).rglob("*")
                             if f.is_file())
            except Exception as exc:
                print(f"[hf] {name:16} FAILED  {type(exc).__name__}: "
                      f"{str(exc)[:200]}", flush=True)
        print(f"[hf] cache now holds {total/1e9:.1f} GB across {len(names)} model(s)")
        return

    if args.smoke_test:
        # Quick per-model check of tokenisation, output dim, finiteness and
        # batching before a full run.
        names = list(MODELS) if args.model == "all" else [args.model]
        _, probe = read_fasta(BASE / "data/queries" / f"{args.prefix}_small_query.fasta")
        sample = sorted(probe, key=len)[::max(1, len(probe) // 8)][:8]
        print(f"[smoke] {len(sample)} sequences, "
              f"len {len(sample[0])}-{len(sample[-1])}\n", flush=True)
        dev = pick_device(args.device)
        for name in names:
            try:
                cd = resolve_cache(MODELS[name]["hf"], args)
                tk, md, sp, dt = load_model(name, dev, args.fp32, cache_dir=cd)
                t0 = time.time()
                vecs = embed_batch(sample, tk, md, sp, dev)
                fin = bool(np.isfinite(vecs.astype(np.float32)).all())
                nrm = np.linalg.norm(vecs.astype(np.float32), axis=1)
                print(f"[smoke] {name:16} OK  dim={vecs.shape[1]:<5} {dt}  "
                      f"{time.time()-t0:.1f}s  finite={fin}  "
                      f"|v| {nrm.min():.2f}-{nrm.max():.2f}", flush=True)
                if not fin:
                    print(f"[smoke] {name}: non-finite output -- try --fp32")
                check_batching(sample, tk, md, sp, dev)
                del md
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
            except Exception as exc:
                print(f"[smoke] {name:16} FAILED  {type(exc).__name__}: "
                      f"{str(exc)[:220]}", flush=True)
        return

    if args.model == "all":
        raise SystemExit("--model all is only valid with --smoke-test")

    cache_dir = resolve_cache(MODELS[args.model]["hf"], args,
                              revision=revision_of(args.model))

    device = pick_device(args.device)
    tok, model, spec, dtype = load_model(args.model, device, args.fp32,
                                         cache_dir=cache_dir)
    print(f"[setup] model={args.model} device={device} dtype={dtype}", flush=True)
    print(f"[setup] arm={args.arm} prefix={args.prefix!r} "
          f"(script {Path(__file__).resolve()})", flush=True)
    if device.type == "cuda":
        print(f"[setup] gpu={torch.cuda.get_device_name(0)} "
              f"hip={getattr(torch.version, 'hip', None)}", flush=True)

    # Warn about embedding files without the .n<count> tag. They are ignored
    # (06_knn only globs the .n<count> form).
    legacy = [q for q in (BASE / "data/embeddings" / args.model).glob("*.f16.npy")
              if ".n" not in q.name.replace(".f16.npy", "")]
    if legacy:
        print(f"[warn] {len(legacy)} embedding file(s) without a sequence-count "
              f"tag found (e.g. {legacy[0].name}). These do not match the joint "
              f"split; they are ignored, not "
              f"reused. Delete them to reclaim space.", flush=True)

    outdir = BASE / "data/embeddings" / args.model
    outdir.mkdir(parents=True, exist_ok=True)
    # Record both the pin and the sha the cache actually resolved to, so an
    # unpinned run is still self-documenting.
    _pin = revision_of(args.model)
    _resolved = cached_revision(spec["hf"], cache_dir or os.environ["HF_HUB_CACHE"])
    manifest = {"arm": args.arm, "model": args.model, "hf": spec["hf"],
                "revision_pinned": _pin,
                "revision_resolved": _resolved,
                "revision_is_pinned": bool(_pin),
                "dim": spec["dim"], "pooling": "mean_over_residues",
                "layer": "last_hidden_state",
                "dtype": str(dtype), "files": []}
    if _pin and _resolved and not str(_resolved).startswith("AMBIGUOUS") \
            and _resolved != _pin:
        raise SystemExit(
            f"{spec['hf']}: pinned {_pin} but the cache resolved {_resolved}. "
            f"Refusing to embed -- a staged cache from an unpinned run "
            f"is the usual cause. Clear the staged copy and re-fetch.")

    # Expand "all" into concrete arms and build the work list before touching
    # any path; the batching probe below then draws from a checked file.
    arms = ["small", "mid", "control"] if args.arm == "all" else [args.arm]
    # Deduplicate across arms: the reference is shared, so it appears in every
    # arm's target list but must only be embedded once per invocation.
    work, seen = [], set()
    for a in arms:
        for stem, path in targets(a, args.prefix):
            if stem not in seen:
                seen.add(stem)
                work.append((stem, path))

    for stem, path in work:
        if not path.exists():
            avail = sorted(q.name for q in (BASE / "data/queries").glob("*.fasta"))
            raise SystemExit(
                f"missing {path}\n"
                f"  resolved from arm={args.arm!r} prefix={args.prefix!r}\n"
                f"  files present in data/queries: {avail}\n"
                f"  if the names differ only by prefix, this script is out of "
                f"sync with the split on disk.")

    if not args.no_check_batching:
        # Probe on a real query set: it spans the length range of interest, and
        # a padding leak is length-dependent.
        probe_path = next((pa for st, pa in work if st.endswith("_real")),
                          work[0][1])
        _, probe = read_fasta(probe_path)
        print(f"[check] probing with {probe_path.name}", flush=True)
        check_batching(probe[:2000], tok, model, spec, device)

    for stem, path in work:
        ids, seqs = read_fasta(path)
        # Key the cached matrix on the input sequence count as well as the
        # name, so rebuilding the split never reuses embeddings computed for a
        # different set of sequences.
        npy = outdir / f"{stem}.n{len(ids)}.f16.npy"
        idsf = outdir / f"{stem}.n{len(ids)}.ids.txt"
        tag = stem
        if npy.exists() and idsf.exists():
            print(f"[skip] {stem}")
            manifest["files"].append({"tag": tag, "npy": str(npy.relative_to(BASE))})
            continue
        print(f"[embed] {stem}: {len(seqs)} sequences", flush=True)
        try:
            mat = embed_all(seqs, tok, model, spec, device, args.batch_tokens)
        except NonFiniteEmbeddings as exc:
            # Reduced precision overflowed: reload this model in fp32 once and
            # continue. Matrices already written stay valid; they are stored as
            # float16 either way.
            if args.fp32:
                raise
            print(f"[warn] {stem}: {exc}\n"
                  f"[warn] reloading {args.model} in fp32 and retrying",
                  flush=True)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            args.fp32 = True
            tok, model, spec, dtype = load_model(args.model, device, True,
                                                 cache_dir=cache_dir)
            manifest["dtype"] = str(dtype)
            manifest["fp32_fallback"] = True
            mat = embed_all(seqs, tok, model, spec, device, args.batch_tokens)

        # Write atomically so a killed job never leaves a truncated matrix that
        # the skip-if-exists check would accept. Both files go to .part first
        # and the .npy is renamed last, so a crash leaves at most the ids file,
        # which fails the two-file skip condition.
        tmp_npy = npy.parent / (npy.name + ".part")
        tmp_ids = idsf.parent / (idsf.name + ".part")
        with open(tmp_npy, "wb") as fh:      # open handle: np.save would
            np.save(fh, mat)                 # otherwise append another .npy
        tmp_ids.write_text("\n".join(ids) + "\n")
        tmp_ids.replace(idsf)
        tmp_npy.replace(npy)
        manifest["files"].append({"tag": tag, "n": len(ids),
                                  "npy": str(npy.relative_to(BASE))})

    man = BASE / "results" / f"embed_manifest_{args.arm}_{args.model}.json"
    man.parent.mkdir(parents=True, exist_ok=True)
    man.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[done] manifest -> {man.relative_to(BASE)}")


if __name__ == "__main__":
    sys.exit(main())
