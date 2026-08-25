"""The frozen document-vector pool: one contiguous fp16 memmap over every TRAIN doc store.

Frozen doc vectors are what make very large negative pools nearly free, so the pool is built
once and everything downstream (InfoNCE bank, teacher-mined hard negatives, the KL term's
candidate set) indexes into it.

hotpotqa-corpus reuses the dev HotpotQA encode -- identical texts, identical prefix, identical
teacher revision -- so the 5.23M-vector encode is paid for once.
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


def store_vecs(store):
    ids, texts = mix.load_store(store)
    return ids, encode_cached(STORE_CACHE_NAME.get(store, f"train-{store}"), texts,
                              prefix="", dtype=torch.float16, verbose=True)


def build(dim=768):
    """-> (index: {(store,docid): global_idx}, memmap (N,dim) fp16). Cached on disk."""
    idx_p, vec_p, meta_p = POOL / "index.json", POOL / "vecs.f16", POOL / "meta.json"
    stores = sorted({mix.load_source(s)["docstore"] for s in mix.available_sources()})
    if idx_p.exists() and vec_p.exists() and meta_p.exists():
        meta = json.loads(meta_p.read_text())
        if meta["stores"] == stores:
            index = {tuple(k.split("\t", 1)): v for k, v in json.loads(idx_p.read_text()).items()}
            return index, np.memmap(vec_p, dtype=np.float16, mode="r",
                                    shape=(meta["n"], meta["dim"])), meta
    per = {}
    total = 0
    for s in stores:
        ids, v = store_vecs(s)
        per[s] = (ids, v)
        total += len(ids)
        print(f"  pool store {s}: {len(ids):,} docs (running total {total:,})", flush=True)
    mm = np.memmap(vec_p, dtype=np.float16, mode="w+", shape=(total, dim))
    index, off, spans = {}, 0, {}
    for s in stores:
        ids, v = per[s]
        mm[off:off + len(ids)] = v
        for i, d in enumerate(ids):
            index[(s, d)] = off + i
        spans[s] = [off, off + len(ids)]
        off += len(ids)
    mm.flush()
    meta = {"n": total, "dim": dim, "stores": stores, "spans": spans}
    meta_p.write_text(json.dumps(meta, indent=1))
    idx_p.write_text(json.dumps({f"{k[0]}\t{k[1]}": v for k, v in index.items()}))
    return index, np.memmap(vec_p, dtype=np.float16, mode="r", shape=(total, dim)), meta


if __name__ == "__main__":
    index, vecs, meta = build()
    print(f"pool: {vecs.shape} fp16 = {vecs.nbytes/1e9:.2f} GB")
    print(json.dumps(meta["spans"], indent=1))
