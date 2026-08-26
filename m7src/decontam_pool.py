"""Codex blocker B2: decontaminate the NEGATIVES surface, not just the positives.

R2 fingerprinted the ~855K TRAIN positives against the six, but training touches all 6.17M pool
rows -- as the random InfoNCE bank, as mined hard negatives, and as the KL term's candidate set.
"TRAIN <-> KNOWN-TEST decontaminated" was therefore false for the actual training surface until
this pass: a six-set document sitting in the pool shapes the loss even if it is nobody's positive.

Direction is inverted relative to decontam.py, for the same RAM reason: here the PROTECTED side is
the small one (272K six docs + a few thousand protected queries), so it is indexed, and the 6.17M
pool rows are streamed against it, store by store in global row order.

Banned (written to work/decontam/banned_pool_rows.npy, sorted int64 global rows):
  - a pool row that exact- or near-duplicates (sketch share >= 8/32) a document of the six;
  - a pool row that exact-matches or shares >= 1 word-8-gram with a query of the six or of
    UNTOUCHED-FINAL (a test query's text entering the loss as a negative is query leakage,
    the same class R1 removes on the TRAIN side).
Measured and disclosed, not banned: dev-query matches (heldout-* queries overlap the pool by
construction -- they are held-out TRAIN queries over the pool's own corpora; banning would alter
the training surface dev was pinned to measure).

Known limitation, disclosed: queries under 8 normalized words fingerprint as one whole-text gram,
which can never equal a document's rolling 8-gram, so short-query leakage into pool docs is only
caught by exact whole-document match (M-decontam-short applies here exactly as it does to R1).

Consumers: train.py loads banned_pool_rows.npy and masks the bank, both miners, and
dataset-provided hard negatives (the KL set draws only from those, so it is clean by
construction). Enforcement points are listed in results/m7_decontam_pool.json.
"""
import json
import sys
import time

import numpy as np

from _paths import REPO, WORK
from decontam import (Inverted, contains_short, exact_u64, ngram_hashes, norm_words,
                      protected_queries, query_grams, short_whole_index, stream_six_docs,
                      DUP_SHARE)
from hashing import sha_stream_list

OUT = WORK / "decontam"
RES = REPO / "results" / "m7_decontam_pool.json"


def _grams_and_sketch(text):
    g = np.unique(ngram_hashes(norm_words(text)))
    return g, g[:32]


def _query_index(texts):
    ex = set(int(exact_u64(t)) for t in texts)
    gr = (np.unique(np.concatenate([query_grams(t) for t in texts]))
          if texts else np.zeros(0, np.uint64))
    return ex, gr, short_whole_index(texts)


def _in_sorted(sorted_arr, vals):
    if vals.size == 0 or sorted_arr.size == 0:
        return False
    i = np.minimum(np.searchsorted(sorted_arr, vals, "left"), sorted_arr.size - 1)
    return bool((sorted_arr[i] == vals).any())


def stream_store_texts(store, expect_sha):
    """Yield a store's doc texts in pool row order, verifying id identity streamingly.

    hotpotqa-corpus streams from the HF arrow cache (trainmix built it as a byte-identical copy
    of BeIR/hotpotqa) so its 1.6 GB JSON is never parsed; everything else goes through
    mix.load_store one store at a time.
    """
    if store == "hotpotqa-corpus":
        from datasets import load_dataset
        from core import doc_text
        corpus = load_dataset("BeIR/hotpotqa", "corpus")["corpus"]
        assert sha_stream_list([str(x) for x in corpus["_id"]]) == expect_sha, \
            f"{store}: HF arrow ids do not match the pool meta -- refuse to stream"
        for r in corpus:
            yield doc_text(r)
    else:
        import mix
        ids, texts = mix.load_store(store)
        assert sha_stream_list(ids) == expect_sha, f"{store}: ids drifted under the pool meta"
        yield from texts


def run(limit_per_store=None):
    t0 = time.time()
    meta = json.loads((WORK / "pool" / "meta.json").read_text())
    spans = meta["spans"]

    print("indexing the protected side ...", flush=True)
    docs = list(stream_six_docs())
    inv = Inverted([_grams_and_sketch(t)[1] for t in docs], [exact_u64(t) for t in docs])
    pq = protected_queries()
    q_idx = {cls: _query_index(v) for cls, v in pq.items()}
    print(f"  six docs {len(docs):,} ({inv.nbytes/1e6:.0f} MB) | queries "
          f"{ {k: len(v) for k, v in pq.items()} } in {time.time()-t0:.0f}s", flush=True)
    del docs

    banned, counts = [], {}
    for store in sorted(spans):
        lo, hi = spans[store]
        c = {"sixdoc_exact": 0, "sixdoc_near": 0, "sixq": 0, "untq": 0, "devq_measured": 0}
        t1, n = time.time(), 0
        for j, text in enumerate(stream_store_texts(store, meta["id_sha256"][store])):
            if limit_per_store and j >= limit_per_store:
                break
            n = j + 1
            g, sk = _grams_and_sketch(text)
            w = norm_words(text)
            if len(w) < 8:                 # short pool rows also emit 4-grams on the query check
                g = np.unique(np.concatenate([g, query_grams(text)]))
            row, bad = lo + j, False
            k = exact_u64(text)
            a, b = np.searchsorted(inv.ek, k, "left"), np.searchsorted(inv.ek, k, "right")
            if b > a:
                c["sixdoc_exact"] += 1
                bad = True
            elif sk.size and inv.h.size:
                x = np.searchsorted(inv.h, sk, "left")
                y = np.searchsorted(inv.h, sk, "right")
                if (y - x).sum() >= DUP_SHARE:
                    idxs = np.concatenate([inv.i[p:q] for p, q in zip(x, y) if q > p])
                    _, cnt = np.unique(idxs, return_counts=True)
                    if cnt.size and cnt.max() >= DUP_SHARE:
                        c["sixdoc_near"] += 1
                        bad = True
            for cls, key in (("six", "sixq"), ("untouched-final", "untq")):
                ex, gr, whole = q_idx[cls]
                # gram share, or a protected 4-7-word query VERBATIM inside this row (review #2
                # BLOCKER 4: rolling k-grams vs the query's whole-hash, same polynomial)
                if int(k) in ex or _in_sorted(gr, g) or contains_short(w, whole):
                    c[key] += 1
                    bad = True
            ex, gr, whole = q_idx["dev"]
            if int(k) in ex or _in_sorted(gr, g) or contains_short(w, whole):
                c["devq_measured"] += 1        # disclosed, not banned -- see module docstring
            if bad:
                banned.append(row)
            if (j + 1) % 500_000 == 0:
                print(f"  {store} {j+1:,}/{hi-lo:,} @ {int((j+1)/(time.time()-t1))} docs/s",
                      flush=True)
        counts[store] = {**c, "rows": n, "seconds": round(time.time() - t1, 1)}
        print(f"  {store}: {c} ({n:,} rows, {time.time()-t1:.0f}s)", flush=True)

    banned = np.array(sorted(banned), dtype=np.int64)
    np.save(OUT / "banned_pool_rows.npy", banned)
    # bind the mask to the pool identity it was computed against (review #2 MAJOR 11)
    (OUT / "banned_pool_rows.meta.json").write_text(json.dumps(
        {"pool_id_sha256": meta["id_sha256"], "n_banned": int(banned.size),
         "rules": "sixdoc exact/near + six/untouched query grams + short-query containment"}))
    report = {
        "_note": "B2 fix: pool rows banned as negatives/KL candidates. Ban rule and the "
                 "short-query limitation are in m7src/decontam_pool.py's docstring. devq hits "
                 "are measured and disclosed, never banned.",
        "params": {"ngram": 8, "sketch": 32, "dup_share": DUP_SHARE},
        "pool_rows": int(sum(v["rows"] for v in counts.values())),
        "banned_rows": int(banned.size),
        "per_store": counts,
        "enforcement": ["train.py bank_ids filter", "mine_hard_negatives cand filter + sig",
                        "mine_bm25_negatives picked filter + sig", "build_arrays hn_idx filter"],
        "seconds_total": round(time.time() - t0, 1),
    }
    if not limit_per_store:
        RES.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "per_store"}, indent=1), flush=True)
    return report


if __name__ == "__main__":
    run(limit_per_store=int(sys.argv[1]) if len(sys.argv) > 1 else None)
