"""The frozen document-vector pool: one contiguous fp16 memmap over every TRAIN doc store.

Frozen doc vectors are what make very large negative pools nearly free, so the pool is built
once and everything downstream (the InfoNCE bank, teacher-mined hard negatives, the KL term's
candidate set) indexes into it.

hotpotqa-corpus reuses the dev HotpotQA encode -- identical texts, identical prefix, identical
teacher revision -- so the 5.23M-vector encode is paid for once.

The index is per-store and lazy. A single dict over all 6.2M (store, docid) keys costs ~1.2 GB
and its JSON dump costs several more, which is how the first version of this file would have
reproduced the OOM incident logged in m7/LEDGER.md.
"""
import json

import numpy as np
import torch

import mix
from _paths import WORK
from teacher import encode_cached

POOL = WORK / "pool"
POOL.mkdir(parents=True, exist_ok=True)
STORE_CACHE_NAME = {"hotpotqa-corpus": "dev-hotpotqa-docs"}  # reuse the dev encode
DIM = 768


class PoolIndex:
    """(store, docid) -> global row. Per-store maps are built on first use and cached."""

    def __init__(self, spans):
        self.spans = spans
        self._maps = {}

    def _map(self, store):
        m = self._maps.get(store)
        if m is None:
            ids, _ = mix.load_store(store)
            m = {d: i for i, d in enumerate(ids)}
            self._maps[store] = m
        return m

    def get(self, store, docid):
        j = self._map(store).get(docid)
        return None if j is None else self.spans[store][0] + j

    def drop(self, store=None):
        self._maps.pop(store, None) if store else self._maps.clear()

    def __contains__(self, key):
        return self.get(*key) is not None

    def __getitem__(self, key):
        v = self.get(*key)
        if v is None:
            raise KeyError(key)
        return v


def store_vecs(store):
    ids, texts = mix.load_store(store)
    v = encode_cached(STORE_CACHE_NAME.get(store, f"train-{store}"), texts, prefix="",
                      dtype=torch.float16, verbose=True)
    return ids, v


def build(dim=DIM):
    """-> (PoolIndex, memmap (N,dim) fp16, meta). Cached on disk."""
    vec_p, meta_p = POOL / "vecs.f16", POOL / "meta.json"
    stores = sorted({mix.load_source(s)["docstore"] for s in mix.available_sources()})
    if vec_p.exists() and meta_p.exists():
        meta = json.loads(meta_p.read_text())
        if meta["stores"] == stores and vec_p.stat().st_size == meta["n"] * dim * 2:
            return (PoolIndex(meta["spans"]),
                    np.memmap(vec_p, dtype=np.float16, mode="r", shape=(meta["n"], dim)), meta)

    counts = {}
    for s in stores:
        ids, _ = mix.load_store(s)
        counts[s] = len(ids)
    total = sum(counts.values())
    spans, off = {}, 0
    for s in stores:
        spans[s] = [off, off + counts[s]]
        off += counts[s]
    print(f"  pool: {total:,} docs x {dim} fp16 = {total*dim*2/1e9:.2f} GB", flush=True)

    tmp = POOL / "vecs.tmp"
    mm = np.memmap(tmp, dtype=np.float16, mode="w+", shape=(total, dim))
    for s in stores:
        ids, v = store_vecs(s)
        assert len(ids) == counts[s], f"{s}: store changed under us"
        lo, hi = spans[s]
        step = 250_000                       # chunked so nothing materializes a whole store
        for a in range(0, len(ids), step):
            b = min(a + step, len(ids))
            mm[lo + a:lo + b] = v[a:b]
        del ids, v
        print(f"  wrote {s}: rows {lo:,}-{hi:,}", flush=True)
    mm.flush()
    del mm
    tmp.rename(vec_p)
    meta = {"n": total, "dim": dim, "stores": stores, "spans": spans, "counts": counts}
    meta_p.write_text(json.dumps(meta, indent=1))
    return (PoolIndex(spans),
            np.memmap(vec_p, dtype=np.float16, mode="r", shape=(total, dim)), meta)


if __name__ == "__main__":
    index, vecs, meta = build()
    print(f"pool: {vecs.shape} fp16 = {vecs.nbytes/1e9:.2f} GB")
    print(json.dumps(meta["spans"], indent=1))
