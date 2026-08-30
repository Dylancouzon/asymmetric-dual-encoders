"""M9 training-text pools.

Query side: M8's extended-filter survivor list (`work/m8_trainq_texts.json`, 337,981 texts,
screened against the six + dev + untouched-final + the reserved four + the LoTTE shadow + the
M9-reserve) re-labelled by source through a two-pointer alignment against the same
`train.build_arrays` derivation it was cut from, then filtered to drop `fever-train`
(FEVER is reserved AND stella-disclosed -- instructions-m9.md Data).

Doc side: the frozen 6.17M-row stella document pool, minus `fever-pos`, minus the
`banned_pool_rows` mask. Vectors already exist, so doc targets cost zero teacher compute.

Nothing here reads a protected path: the extended filter's output is a plain text list and the
pool is M7's own frozen artifact.
"""
import hashlib
import json

import numpy as np

import m9base
from m9base import REPO, WORK, RESULTS

FEVER_SOURCES = ("fever-train",)
FEVER_STORE = "fever-pos"


def _sha(obj):
    return hashlib.sha256(json.dumps(obj).encode()).hexdigest()


def labelled_query_pool():
    """-> (texts, sources, meta). M8's filtered TRAIN query texts with a source label each."""
    import pool as poolmod
    import train
    from train import Cfg

    index, _vecs, pmeta = poolmod.build()
    q_texts, _pos, _hn, src_id, srcs = train.build_arrays(Cfg(), index)
    q_texts = list(q_texts)

    kept = json.loads((WORK / "m8_trainq_texts.json").read_text())
    man = json.loads((RESULTS / "m8_trainq_manifest.json").read_text())
    assert man["n_derived"] == len(q_texts), (
        f"the M8 fit list was cut from {man['n_derived']:,} derived texts, this derivation "
        f"produced {len(q_texts):,} -- the TRAIN pool moved under the filter, refuse to align")
    assert man["n_kept"] == len(kept)

    # Two-pointer: `kept` is an order-preserving subsequence of `q_texts` (build_fitlist keeps
    # `[q for i, q in enumerate(q_texts) if i not in hits]`), so a greedy walk recovers the index
    # of every kept text even when texts repeat.
    idx, j = [], 0
    for i, t in enumerate(q_texts):
        if j < len(kept) and kept[j] == t:
            idx.append(i)
            j += 1
    assert j == len(kept), f"alignment consumed {j:,} of {len(kept):,} kept texts"
    assert len(idx) == len(kept)

    texts = [q_texts[i] for i in idx]
    sources = [srcs[int(src_id[i])] for i in idx]
    meta = {
        "derived_from": "train.build_arrays(Cfg(), pool.build()) + work/m8_trainq_texts.json",
        "n_derived": len(q_texts), "n_kept": len(texts),
        "m8_manifest_sha256": man["sha256"],
        "by_source": {s: sources.count(s) for s in sorted(set(sources))},
        "pool_encoder": pmeta["encoder"], "pool_n": pmeta["n"],
    }
    return texts, sources, meta


def screen_query_pool():
    """-> (texts, rows, meta). The M9.0-locked screen query pool: labelled pool minus
    fever-train. `rows` are the positions of these texts inside the 337,981-row M8 fit list,
    which is exactly the row order of the cached stella target matrix `trainq-337981` -- so
    targets are a fancy-index away and cost zero teacher compute."""
    texts, sources, meta = labelled_query_pool()
    keep = [i for i, s in enumerate(sources) if s not in FEVER_SOURCES]
    out = [texts[i] for i in keep]
    rows = np.array(keep, dtype=np.int64)
    meta = dict(meta)
    meta.update({
        "excluded_sources": list(FEVER_SOURCES),
        "n_screen_pool": len(out),
        "screen_pool_sha256": _sha(out),
        "screen_rows_sha256": hashlib.sha256(rows.tobytes()).hexdigest(),
        "by_source_kept": {s: sources.count(s) for s in sorted(set(sources))
                           if s not in FEVER_SOURCES},
    })
    return out, rows, meta


def stella_query_targets():
    """-> memmap (337981, 1024) fp16, the cached stella s2p query targets in fit-list order.
    Verifies the cache was built from the identical text list before returning it."""
    import teacher

    kept = json.loads((WORK / "m8_trainq_texts.json").read_text())
    return teacher.encode_cached("trainq-337981", kept, prefix=teacher.QUERY_PREFIX,
                                 max_length=512, verbose=False)


def doc_pool_rows(n, seed):
    """-> (global pool row indices, meta). A fixed subsample of the frozen stella doc pool,
    excluding the fever-pos span and every banned row. Row -> vector is pool_vecs[row];
    row -> text is store text at the row's offset inside its span."""
    import pool as poolmod

    _index, _vecs, pmeta = poolmod.build()
    spans = pmeta["spans"]
    lo, hi = spans[FEVER_STORE]
    banned = np.load(WORK / "decontam" / "banned_pool_rows.npy")
    mask = np.ones(pmeta["n"], dtype=bool)
    mask[lo:hi] = False
    mask[banned] = False
    eligible = np.flatnonzero(mask)
    rng = np.random.default_rng(seed)
    rows = np.sort(rng.choice(eligible, size=n, replace=False))
    meta = {"n_eligible": int(eligible.size), "n_drawn": int(rows.size), "seed": seed,
            "excluded_store": FEVER_STORE, "n_banned": int(banned.size),
            "rows_sha256": hashlib.sha256(rows.tobytes()).hexdigest()}
    return rows, meta


def row_texts(rows):
    """-> list[str] document texts for global pool rows, in the given order."""
    import mix
    import pool as poolmod

    _index, _vecs, pmeta = poolmod.build()
    spans = pmeta["spans"]
    out = [None] * len(rows)
    for store, (lo, hi) in spans.items():
        sel = np.flatnonzero((rows >= lo) & (rows < hi))
        if sel.size == 0:
            continue
        _ids, texts = mix.load_store(store)
        for k in sel:
            out[int(k)] = texts[int(rows[int(k)]) - lo]
    assert all(t is not None for t in out)
    return out


def main():
    texts, rows, meta = screen_query_pool()
    lens = np.array([len(t.split()) for t in texts])
    meta["word_len"] = {"mean": round(float(lens.mean()), 3), "p50": int(np.percentile(lens, 50)),
                        "p95": int(np.percentile(lens, 95)), "max": int(lens.max())}
    tgt = stella_query_targets()
    meta["stella_target_cache"] = {"shape": list(tgt.shape), "dtype": str(tgt.dtype),
                                   "verified": "teacher.encode_cached re-derived the same key"}
    dv = np.asarray(tgt[rows[:64]], dtype=np.float32)
    meta["target_norm_check"] = {"min": round(float(np.linalg.norm(dv, axis=1).min()), 6),
                                 "max": round(float(np.linalg.norm(dv, axis=1).max()), 6)}
    (WORK / "m9_screen_queries.json").write_text(json.dumps(texts))
    np.save(WORK / "m9_screen_rows.npy", rows)
    (RESULTS / "m9_screen_pool.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
