"""The frozen teacher document/query encoder -- whichever one is selected.

Which model, how it pools, and what prompt it wants all come from `encoders.py`; set M7_ENCODER
to switch. Pooling used to be hardcoded to CLS here, so a mean-pooled teacher (stella) could not
be run at all, and the cache key asserted "cls-l2" as a literal regardless of what was run.

Encodes are cached as sharded fp16 .npy under work/enc/<key>/, keyed on (model, revision, dtype,
tokenizer identity, POOLING, prefix, max_length, corpus hash); shards make a long corpus encode
resumable after a crash. The key for bge-base is byte-identical to the pre-registry one so the
existing ~22 GB of encodes stay valid -- `m7src/test_encoders.py` pins that.
"""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

import encoders
from _paths import WORK

SPEC = encoders.active()
# Module-level names kept so the dozen existing call sites need no change.
TEACHER = SPEC.repo
TEACHER_REV = SPEC.revision
QUERY_PREFIX = SPEC.query_prefix
DOC_PREFIX = SPEC.doc_prefix
ENC = WORK / "enc"
SHARD = 50_000  # texts per shard file


def sha_texts(texts):
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8", "surrogatepass"))
        h.update(b"\x00")
    return h.hexdigest()


_CACHE = {}
_DENSE = {}


def load_post_dense(spec, device="cuda"):
    """Load the post-pooling Dense module a Spec declares, or None.

    stella's published pipeline is Transformer -> Pooling(mean) -> Dense_1024 -> normalize. Encoding
    without that Dense yields a different model from the one its MTEB score describes, so this is
    not optional polish -- it is loader fidelity, the same failure class as the M6 gate's BLOCKER 1.
    """
    if not spec.post_dense:
        return None
    key = (spec.repo, spec.revision, spec.post_dense, device)
    if key not in _DENSE:
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        cfg = json.loads(Path(hf_hub_download(spec.repo, f"{spec.post_dense}/config.json",
                                              revision=spec.revision)).read_text())
        if cfg.get("activation_function", "").rsplit(".", 1)[-1] not in ("Identity",):
            raise NotImplementedError(f"{spec.name} Dense uses a non-identity activation "
                                      f"{cfg['activation_function']!r}; add it before encoding")
        w = load_file(hf_hub_download(spec.repo, f"{spec.post_dense}/model.safetensors",
                                      revision=spec.revision))
        W = w["linear.weight"].to(device).float()
        b = w.get("linear.bias")
        _DENSE[key] = (W, None if b is None else b.to(device).float())
    return _DENSE[key]


def load_teacher(model_id=TEACHER, revision=TEACHER_REV, dtype=torch.float32, device="cuda"):
    key = (model_id, revision, str(dtype), device)
    if key not in _CACHE:
        sp = encoders.by_repo(model_id)
        kw = {"trust_remote_code": True} if sp.trust_remote_code else {}
        tok = AutoTokenizer.from_pretrained(model_id, revision=revision, **kw)
        # config_kwargs belong on the CONFIG, not on from_pretrained: transformers 4.57 forwards
        # unknown from_pretrained kwargs straight to the model's __init__, which raises.
        if sp.config_kwargs:
            cfg = AutoConfig.from_pretrained(model_id, revision=revision, **kw,
                                             **sp.config_kwargs)
            model = AutoModel.from_pretrained(model_id, revision=revision, config=cfg,
                                              dtype=dtype, **kw).to(device).eval()
        else:
            model = AutoModel.from_pretrained(model_id, revision=revision, dtype=dtype,
                                              **kw).to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        _CACHE[key] = (tok, model)
    return _CACHE[key]


@torch.no_grad()
def encode_batch(tok, model, texts, max_length=512, device="cuda", pooling="cls", dense=None):
    """Pool per the encoder's own convention, apply its post-pooling Dense if it has one, then
    L2 normalize. Returned fp32 on CPU."""
    b = tok(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
    h = model(**b).last_hidden_state
    if pooling == "cls":
        v = h[:, 0]
    elif pooling == "mean":
        m = b["attention_mask"].unsqueeze(-1).to(h.dtype)
        v = (h * m).sum(1) / m.sum(1).clamp(min=1e-6)
    else:
        raise ValueError(f"unknown pooling {pooling!r}")
    v = v.float()
    if dense is not None:
        W, bias = dense
        v = v @ W.T + (0.0 if bias is None else bias)
    return torch.nn.functional.normalize(v, dim=-1).cpu().numpy()


def _order_by_length(tok, texts, max_length):
    n_tok = [min(len(tok(t, add_special_tokens=True, truncation=True, max_length=max_length)["input_ids"]), max_length)
             for t in texts]
    order = np.argsort(np.array(n_tok), kind="stable")
    return order, n_tok


@torch.no_grad()
def encode(texts, prefix="", max_length=512, batch_tokens=16384, model_id=TEACHER,
           revision=TEACHER_REV, dtype=torch.float32, device="cuda", verbose=False):
    """Length-bucketed dynamic batching; returns (N, dim) fp32 normalized in input order."""
    tok, model = load_teacher(model_id, revision, dtype, device)
    spec = encoders.by_repo(model_id)
    pooling, dense = spec.pooling, load_post_dense(spec, device)
    order, n_tok = _order_by_length(tok, [prefix + t for t in texts], max_length)
    # The output width is the Dense's out_features when there is one, NOT the backbone's hidden
    # size. They coincide for stella's square 1024->1024 head, but an MRL head (e.g. 1024->256)
    # would silently write 256 values into 1024-wide rows.
    dim = model.config.hidden_size if dense is None else int(dense[0].shape[0])
    if dim != spec.dim:
        raise AssertionError(f"{spec.name}: encode width {dim} != Spec.dim {spec.dim}; the "
                             f"registry and the loaded modules disagree")
    out = np.empty((len(texts), dim), dtype=np.float32)
    i, t0, done = 0, time.time(), 0
    while i < len(order):
        # grow the batch until padded token count would exceed batch_tokens
        longest, j = 0, i
        while j < len(order):
            L = max(longest, n_tok[order[j]])
            if (j - i + 1) * L > batch_tokens and j > i:
                break
            longest, j = L, j + 1
        idx = order[i:j]
        out[idx] = encode_batch(tok, model, [prefix + texts[k] for k in idx], max_length, device,
                                pooling=pooling, dense=dense)
        done += len(idx)
        if verbose and (done % 20000 < len(idx)):
            print(f"    {done}/{len(texts)} @ {done/(time.time()-t0):.0f} texts/s", flush=True)
        i = j
    return out


DT = {torch.float32: "fp32", torch.float16: "fp16", torch.bfloat16: "bf16"}


def cache_key(name, prefix, max_length, model_id, revision, corpus_sha, dtype):
    # The key must change whenever the vectors would, so POOLING and the tokenizer identity come
    # from the registry rather than being asserted as bge's literals. The pre-registry version
    # hardcoded "cls-l2"/"bert-wordpiece-30522", so a mean-pooled encode would have been stored
    # under a key claiming CLS. For bge-base the resulting blob is unchanged.
    spec = encoders.by_repo(model_id)
    blob = json.dumps({"name": name, "prefix": prefix, "max_length": max_length, "model": model_id,
                       "revision": revision, "pooling": spec.pooling_key,
                       "corpus_sha256": corpus_sha,
                       "encode_dtype": DT[dtype], "store_dtype": "fp16",
                       "tokenizer": spec.tokenizer_id,
                       # Added CONDITIONALLY so bge-base's blob stays byte-identical and the ~22 GB
                       # of existing encodes remain valid. post_dense also covers a future MRL
                       # truncation, since that is a different Dense directory and hence a
                       # different output dimension -- which is what would otherwise collide.
                       **({"post_dense": spec.post_dense} if spec.post_dense else {})},
                      sort_keys=True)
    return f"{name}-{DT[dtype]}-{hashlib.sha256(blob.encode()).hexdigest()[:12]}", blob


def encode_cached(name, texts, prefix="", max_length=512, model_id=TEACHER, revision=TEACHER_REV,
                  batch_tokens=32768, device="cuda", verbose=True, dtype=torch.float16):
    """Shard-resumable cached encode. Returns (N, dim) fp16 memmap-backed array."""
    key, blob = cache_key(name, prefix, max_length, model_id, revision, sha_texts(texts), dtype)
    d = ENC / key
    d.mkdir(parents=True, exist_ok=True)
    (d / "meta.json").write_text(blob)
    n_shards = (len(texts) + SHARD - 1) // SHARD
    for s in range(n_shards):
        p = d / f"shard_{s:05d}.npy"
        if p.exists():
            continue
        lo, hi = s * SHARD, min((s + 1) * SHARD, len(texts))
        if verbose:
            print(f"  [{name}] shard {s+1}/{n_shards} ({hi-lo} texts)", flush=True)
        v = encode(texts[lo:hi], prefix, max_length, batch_tokens, model_id, revision,
                   dtype=dtype, device=device, verbose=verbose)
        np.save(p.with_suffix(".tmp.npy"), v.astype(np.float16))
        p.with_suffix(".tmp.npy").rename(p)
    return _combined(d, n_shards, len(texts))


def _combined(d, n_shards, n_rows):
    """Return the encode as a read-only memmap over one contiguous file.

    np.concatenate over the shards would materialize the whole encode in RAM -- 8 GB for the
    5.23M-doc HotpotQA dev component, on every call. The shards stay as the resumable unit; this
    is a one-time stitch, and the combined file is what every reader mmaps afterwards.
    """
    parts = [np.load(d / f"shard_{s:05d}.npy", mmap_mode="r") for s in range(n_shards)]
    if n_shards == 1:
        return parts[0]
    dim = parts[0].shape[1]
    comb = d / "combined.f16"
    if not comb.exists() or comb.stat().st_size != n_rows * dim * 2:
        tmp = d / "combined.tmp"
        mm = np.memmap(tmp, dtype=np.float16, mode="w+", shape=(n_rows, dim))
        off = 0
        for p in parts:
            mm[off:off + len(p)] = p
            off += len(p)
        mm.flush()
        del mm
        tmp.rename(comb)
    return np.memmap(comb, dtype=np.float16, mode="r", shape=(n_rows, dim))
