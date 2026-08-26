"""Frozen teacher document/query encoder (default BAAI/bge-base-en-v1.5, revision-pinned).

bge convention: CLS pooling, L2 normalize, optional query prefix. Encodes are cached as
sharded fp16 .npy under work/enc/<key>/ where the key covers (model, revision, dtype,
tokenizer hash, prefix, max_length, corpus hash) per the M7 ops rules; shards make a long
corpus encode resumable after a crash.
"""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from _paths import WORK

TEACHER = "BAAI/bge-base-en-v1.5"
TEACHER_REV = "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
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
        tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
        model = AutoModel.from_pretrained(model_id, revision=revision, dtype=dtype).to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        _CACHE[key] = (tok, model)
    return _CACHE[key]


@torch.no_grad()
def encode_batch(tok, model, texts, max_length=512, device="cuda"):
    """CLS pooling + L2 normalize, returned fp32 on CPU."""
    b = tok(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
    h = model(**b).last_hidden_state[:, 0]
    return torch.nn.functional.normalize(h.float(), dim=-1).cpu().numpy()


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
        out[idx] = encode_batch(tok, model, [prefix + texts[k] for k in idx], max_length, device)
        done += len(idx)
        if verbose and (done % 20000 < len(idx)):
            print(f"    {done}/{len(texts)} @ {done/(time.time()-t0):.0f} texts/s", flush=True)
        i = j
    return out


DT = {torch.float32: "fp32", torch.float16: "fp16", torch.bfloat16: "bf16"}


def cache_key(name, prefix, max_length, model_id, revision, corpus_sha, dtype):
    blob = json.dumps({"name": name, "prefix": prefix, "max_length": max_length, "model": model_id,
                       "revision": revision, "pooling": "cls-l2", "corpus_sha256": corpus_sha,
                       "encode_dtype": DT[dtype], "store_dtype": "fp16",
                       "tokenizer": "bert-wordpiece-30522"}, sort_keys=True)
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
