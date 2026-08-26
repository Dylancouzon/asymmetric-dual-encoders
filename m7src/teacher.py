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
from transformers import AutoModel, AutoTokenizer

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


def load_teacher(model_id=TEACHER, revision=TEACHER_REV, dtype=torch.float32, device="cuda"):
    key = (model_id, revision, str(dtype), device)
    if key not in _CACHE:
        kw = {"trust_remote_code": True} if encoders.by_repo(model_id).trust_remote_code else {}
        tok = AutoTokenizer.from_pretrained(model_id, revision=revision, **kw)
        model = AutoModel.from_pretrained(model_id, revision=revision, dtype=dtype,
                                          **kw).to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        _CACHE[key] = (tok, model)
    return _CACHE[key]


@torch.no_grad()
def encode_batch(tok, model, texts, max_length=512, device="cuda", pooling="cls"):
    """Pool per the encoder's own convention, then L2 normalize. Returned fp32 on CPU."""
    b = tok(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
    h = model(**b).last_hidden_state
    if pooling == "cls":
        v = h[:, 0]
    elif pooling == "mean":
        m = b["attention_mask"].unsqueeze(-1).to(h.dtype)
        v = (h * m).sum(1) / m.sum(1).clamp(min=1e-6)
    else:
        raise ValueError(f"unknown pooling {pooling!r}")
    return torch.nn.functional.normalize(v.float(), dim=-1).cpu().numpy()


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
    pooling = encoders.by_repo(model_id).pooling
    order, n_tok = _order_by_length(tok, [prefix + t for t in texts], max_length)
    dim = model.config.hidden_size
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
                                pooling=pooling)
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
                       "tokenizer": spec.tokenizer_id}, sort_keys=True)
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
