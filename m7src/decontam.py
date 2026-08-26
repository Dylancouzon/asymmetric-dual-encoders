"""Fingerprint decontamination: TRAIN vs DEV, KNOWN-TEST (the six), and UNTOUCHED-FINAL.

Two fingerprints, both deterministic and process-stable (blake2b, never Python's salted hash):
  exact     64-bit blake2b of the normalized text (lowercased, alphanumeric words, single-spaced)
  near-dup  word-8-gram rolling hashes reduced to a bottom-32 sketch; two texts are
            near-duplicates when their sketches share >= 8 of 32 values (est. Jaccard >= 0.25)

Direction matters for memory. The index is built over the TRAIN side (queries, and the ~1M
documents that are actually positives) and the protected corpora are STREAMED against it, so
peak RAM is one train index (~0.4 GB) regardless of whether the protected corpus is 70K
CQADupStack documents or 4.6M DBpedia abstracts. The first version of this file built indexes
over the protected side instead and was killed by the OOM killer on DBpedia.

Pre-registered rules -- see m7/LEDGER.md for the reasoning, which is part of the protocol:
  R1  REMOVE a training pair whose QUERY exactly matches or shares >=1 word-8-gram with any
      protected query (six + dev + untouched-final). Query overlap is the real leakage.
  R2  REMOVE a training pair whose POSITIVE DOCUMENT exact- or near-duplicates a document of
      the six. This enforces the contamination map at fingerprint level rather than by name.
  R3  MEASURE AND DISCLOSE, do not remove, document overlap with DEV and UNTOUCHED-FINAL
      corpora. Those corpora are Wikipedia and StackExchange; hotpotqa-corpus IS the dev
      HotpotQA corpus and fever-pos comes from the untouched FEVER corpus, so removal would
      delete the sources rather than decontaminate them. What removal protects -- the test
      QUERIES and QRELS -- is enforced by R1 and by the final-scorer ledger. Every comparator
      in the M4 matrix has the same property, so the comparison stays like-for-like; the
      report states the measured rates.
"""
import hashlib
import json
import sys
import time

import numpy as np

from _paths import REPO, WORK

OUT = WORK / "decontam"
OUT.mkdir(parents=True, exist_ok=True)
NGRAM = 8
SKETCH = 32
DUP_SHARE = 8
_BASE = np.uint64(1_000_003)
_wcache = {}


def _wh(w):
    v = _wcache.get(w)
    if v is None:
        if len(_wcache) > 8_000_000:      # bounded: never let the memo table become the leak
            _wcache.clear()
        v = np.uint64(int.from_bytes(hashlib.blake2b(w.encode(), digest_size=8).digest(), "little"))
        _wcache[w] = v
    return v


def norm_words(text):
    out, cur = [], []
    for ch in text.lower():
        if ch.isalnum():
            cur.append(ch)
        elif cur:
            out.append("".join(cur))
            cur = []
    if cur:
        out.append("".join(cur))
    return out


def exact_u64(text):
    return np.uint64(int.from_bytes(
        hashlib.blake2b(" ".join(norm_words(text)).encode(), digest_size=8).digest(), "little"))


def ngram_hashes(words):
    """Rolling polynomial word-8-gram hashes. Texts shorter than 8 words hash as one whole gram."""
    if not words:
        return np.zeros(0, dtype=np.uint64)
    wh = np.fromiter((_wh(w) for w in words), dtype=np.uint64, count=len(words))
    if len(wh) < NGRAM:
        h = np.uint64(0)
        for i, x in enumerate(wh):
            h = h + x * (_BASE ** np.uint64(i))
        return np.array([h], dtype=np.uint64)
    n = len(wh) - NGRAM + 1
    acc = np.zeros(n, dtype=np.uint64)
    for j in range(NGRAM):
        acc += wh[j:j + n] * (_BASE ** np.uint64(j))
    return acc


def sketch(text):
    return np.unique(ngram_hashes(norm_words(text)))[:SKETCH]


def all_grams(text):
    return np.unique(ngram_hashes(norm_words(text)))


class Inverted:
    """hash -> train-item ids, as two sorted parallel arrays. Built once, streamed against."""

    def __init__(self, per_item_hashes, exact_keys):
        h, i = [], []
        for k, hs in enumerate(per_item_hashes):
            if hs.size:
                h.append(hs)
                i.append(np.full(hs.size, k, dtype=np.int32))
        self.h = np.concatenate(h) if h else np.zeros(0, np.uint64)
        self.i = np.concatenate(i) if i else np.zeros(0, np.int32)
        o = np.argsort(self.h, kind="stable")
        self.h, self.i = self.h[o], self.i[o]
        self.n = len(per_item_hashes)
        ek = np.asarray(exact_keys, dtype=np.uint64)
        eo = np.argsort(ek, kind="stable")
        self.ek, self.ei = ek[eo], eo.astype(np.int32)

    @property
    def nbytes(self):
        return self.h.nbytes + self.i.nbytes + self.ek.nbytes + self.ei.nbytes

    def match(self, text, min_share, want_sketch=True):
        """-> (exact_hits, near_hits) as arrays of train-item ids for ONE protected text."""
        k = exact_u64(text)
        lo, hi = np.searchsorted(self.ek, k, "left"), np.searchsorted(self.ek, k, "right")
        ex = self.ei[lo:hi]
        hs = sketch(text) if want_sketch else all_grams(text)
        if hs.size == 0 or self.h.size == 0:
            return ex, np.zeros(0, np.int32)
        a = np.searchsorted(self.h, hs, "left")
        b = np.searchsorted(self.h, hs, "right")
        span = b - a
        if span.sum() == 0:
            return ex, np.zeros(0, np.int32)
        idxs = np.concatenate([self.i[x:y] for x, y in zip(a, b) if y > x])
        u, c = np.unique(idxs, return_counts=True)
        return ex, u[c >= min_share]


# ---- protected-side streams (never materialized) -------------------------------------

def stream_six_docs():
    import os
    os.environ.setdefault("BENCH_DATASETS", "scifact,nfcorpus,fiqa,arguana,scidocs,trec-covid")
    from datasets import load_dataset
    from core import DATASETS, doc_text
    for ds in DATASETS:
        for r in load_dataset(f"BeIR/{ds}", "corpus")["corpus"]:
            yield doc_text(r)


def stream_beir_docs(ds):
    from datasets import load_dataset
    from core import doc_text
    for r in load_dataset(f"BeIR/{ds}", "corpus")["corpus"]:
        yield doc_text(r)


def stream_cqa_dev_docs():
    import devsuite
    for c in ("cqadup-programmers", "cqadup-physics"):
        _, doc_texts, *_ = devsuite.load(c)
        yield from doc_texts


def protected_queries():
    """The six (from the vendored frozen_eval text), dev, and untouched-final query strings."""
    import os
    os.environ.setdefault("BENCH_DATASETS", "scifact,nfcorpus,fiqa,arguana,scidocs,trec-covid")
    from datasets import load_dataset
    import devsuite
    from core import DATASETS
    qs = {}
    six = []
    for ds in DATASETS:
        six += list(json.loads((REPO / "results" / "frozen_eval" / f"{ds}.json").read_text())["queries"].values())
    qs["six"] = six
    dev = []
    for c in devsuite.COMPONENTS:
        dev += devsuite.load(c)[3]
    qs["dev"] = dev
    unt = []
    for ds in ("fever", "dbpedia-entity"):
        keep = {str(r["query-id"]) for r in load_dataset(f"BeIR/{ds}-qrels", split="test")}
        q = load_dataset(f"BeIR/{ds}", "queries")["queries"]
        unt += [t for i, t in zip(list(q["_id"]), list(q["text"])) if str(i) in keep]
    qs["untouched-final"] = unt
    return qs


# ---- runner --------------------------------------------------------------------------

def run():
    import mix
    t0 = time.time()
    tr, _ = mix.split_pairs()
    print(f"[train] {len(tr):,} TRAIN pairs (held-out slices already excluded)", flush=True)

    # --- R1: protected queries indexed forward (they are few); check each train query -----
    pq = protected_queries()
    prot_q = [q for v in pq.values() for q in v]
    print(f"[R1] protected queries: " + ", ".join(f"{k} {len(v):,}" for k, v in pq.items()), flush=True)
    q_ex = set(int(exact_u64(q)) for q in prot_q)
    q_gram = np.unique(np.concatenate([all_grams(q) for q in prot_q]))
    print(f"  query index: {len(q_ex):,} exact, {q_gram.size:,} 8-grams ({time.time()-t0:.0f}s)", flush=True)

    r1 = {}
    survive = []
    for src, qid, query, pos, hneg in tr:
        if int(exact_u64(query)) in q_ex:
            r1[src] = r1.get(src, {"exact": 0, "near": 0})
            r1[src]["exact"] += 1
            continue
        g = all_grams(query)
        if g.size and np.isin(g, q_gram, assume_unique=True).any():
            r1[src] = r1.get(src, {"exact": 0, "near": 0})
            r1[src]["near"] += 1
            continue
        survive.append((src, qid, pos))
    print(f"[R1] removed {sum(v['exact']+v['near'] for v in r1.values()):,} pairs; "
          f"{len(survive):,} survive ({time.time()-t0:.0f}s)", flush=True)

    # --- index the TRAIN positive documents (the direction that bounds memory) ------------
    stores, doc_key, doc_text_of = {}, {}, {}
    for src, qid, pos in survive:
        st = mix.load_source(src)["docstore"]
        if st not in stores:
            ids, texts = mix.load_store(st)
            stores[st] = dict(zip(ids, texts))
        for d in pos:
            t = stores[st].get(d)
            if t is not None and (st, d) not in doc_key:
                doc_key[(st, d)] = len(doc_key)
                doc_text_of[(st, d)] = t
    keys = list(doc_key)
    print(f"[index] {len(keys):,} unique TRAIN positive documents ({time.time()-t0:.0f}s)", flush=True)
    sk = [sketch(doc_text_of[k]) for k in keys]
    ex = [exact_u64(doc_text_of[k]) for k in keys]
    inv = Inverted(sk, ex)
    del sk, ex, doc_text_of, stores
    print(f"  inverted index {inv.h.size:,} sketch hashes, {inv.nbytes/1e9:.2f} GB "
          f"({time.time()-t0:.0f}s)", flush=True)

    # --- stream each protected corpus against it ------------------------------------------
    def sweep(label, it, total=None):
        flag = np.zeros(inv.n, dtype=bool)
        exact_flag = np.zeros(inv.n, dtype=bool)
        n = 0
        for text in it:
            e, near = inv.match(text, DUP_SHARE, want_sketch=True)
            if e.size:
                exact_flag[e] = True
            if near.size:
                flag[near] = True
            n += 1
            if n % 500_000 == 0:
                print(f"    {label} {n:,}{'/' + format(total, ',') if total else ''} "
                      f"({time.time()-t0:.0f}s)", flush=True)
        res = {"protected_docs_scanned": n,
               "train_docs_exact_dup": int(exact_flag.sum()),
               "train_docs_near_dup": int(flag.sum()),
               "train_docs_total": inv.n,
               "near_dup_rate": round(float((flag | exact_flag).mean()), 5)}
        print(f"  [{label}] {json.dumps(res)}", flush=True)
        return res, (flag | exact_flag)

    six_res, six_hit = sweep("six", stream_six_docs(), 272_117)
    cqa_res, _ = sweep("cqadupstack-dev", stream_cqa_dev_docs(), 70_492)
    dbp_res, _ = sweep("dbpedia-untouched", stream_beir_docs("dbpedia-entity"), 4_635_922)

    # --- R2 removal (the six only); R3 is disclosure --------------------------------------
    bad = {keys[i] for i in np.nonzero(six_hit)[0]}
    r2, kept = {}, {}
    for src, qid, pos in survive:
        st = mix.load_source(src)["docstore"]
        if any((st, d) in bad for d in pos):
            r2[src] = r2.get(src, 0) + 1
            continue
        kept.setdefault(src, []).append(qid)
    n_kept = sum(len(v) for v in kept.values())
    (OUT / "kept.json").write_text(json.dumps({k: sorted(v) for k, v in kept.items()}))

    summary = {
        "n_train_pairs_in": len(tr), "n_train_pairs_kept": n_kept,
        "R1_query_removals": r1, "R1_total": sum(v["exact"] + v["near"] for v in r1.values()),
        "R2_six_doc_removals": r2, "R2_total": sum(r2.values()),
        "R3_disclosed_overlap": {"six (also the R2 removal basis)": six_res,
                                 "cqadupstack-dev": cqa_res,
                                 "dbpedia-entity-untouched": dbp_res,
                                 "hotpotqa-dev": "100% by construction: hotpotqa-corpus IS the dev corpus",
                                 "fever-untouched": "fever-pos is drawn from the untouched FEVER corpus"},
        "protected_queries": {k: len(v) for k, v in pq.items()},
        "params": {"ngram": NGRAM, "sketch": SKETCH, "dup_share": DUP_SHARE,
                   "hash": "blake2b-64 word hashes, polynomial rolling n-gram, bottom-k sketch"},
        "rules": {"R1": "remove on query overlap (all partitions)",
                  "R2": "remove on positive-document overlap with the six",
                  "R3": "measure and disclose document overlap with dev and untouched-final; "
                        "see the module docstring and m7/LEDGER.md for why removal there would "
                        "delete the sources rather than decontaminate them"},
        "seconds": round(time.time() - t0, 1),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    (REPO / "results" / "m7_decontam.json").write_text(json.dumps(summary, indent=1))
    print("\n" + json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    run()
