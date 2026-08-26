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
        # same modular sum as the array path, done as one vectorized reduction: the scalar loop
        # produced identical values but emitted a uint64-overflow RuntimeWarning per call, and the
        # wraparound is deliberate
        pw = _BASE ** np.arange(len(wh), dtype=np.uint64)
        return np.array([(wh * pw).sum(dtype=np.uint64)], dtype=np.uint64)
    n = len(wh) - NGRAM + 1
    acc = np.zeros(n, dtype=np.uint64)
    for j in range(NGRAM):
        acc += wh[j:j + n] * (_BASE ** np.uint64(j))
    return acc


SHORT_NGRAM = 4


def short_grams(words):
    """Word-4-grams for short texts (4 <= len < 8): the M-decontam-short fix. The 8-gram rule
    degenerates to normalized exact match below 8 words -- the dominant NQ/FEVER query regime --
    so short queries get rolling 4-grams ON THE QUERY PATHS ONLY (R1 and the pool pass's
    query-leak check). Document fingerprints for >= 8-word texts are untouched, so R2/R3/pool
    doc-vs-doc results stay bit-identical. Adopted 2026-08-26 on a measured dry run: >= 1 shared
    4-gram removes 5,126 of 571,329 TRAIN queries (0.9%%) -- conservative, affordable."""
    if not (SHORT_NGRAM <= len(words) < NGRAM):
        return np.zeros(0, dtype=np.uint64)
    wh = np.fromiter((_wh(w) for w in words), dtype=np.uint64, count=len(words))
    n = len(wh) - SHORT_NGRAM + 1
    acc = np.zeros(n, dtype=np.uint64)
    for j in range(SHORT_NGRAM):
        acc += wh[j:j + n] * (_BASE ** np.uint64(j))
    return np.unique(acc)


def query_grams(text):
    """All grams a QUERY contributes on the R1 path: 8-grams (or the whole-text hash) plus the
    short-text 4-grams. Query-side only; never use for documents."""
    w = norm_words(text)
    return np.unique(np.concatenate([ngram_hashes(w), short_grams(w)]))


def rolling_kgrams(words, k):
    """Rolling word-k-gram hashes, same polynomial as ngram_hashes -- so a k-word text's
    whole-hash equals a k-gram window over the same words inside any longer text."""
    if len(words) < k:
        return np.zeros(0, dtype=np.uint64)
    wh = np.fromiter((_wh(w) for w in words), dtype=np.uint64, count=len(words))
    n = len(wh) - k + 1
    acc = np.zeros(n, dtype=np.uint64)
    for j in range(k):
        acc += wh[j:j + n] * (_BASE ** np.uint64(j))
    return acc


def short_whole_index(texts):
    """{k: sorted array of whole-hashes} over the 4-7-word texts. With rolling_kgrams this makes
    'protected short query appears VERBATIM inside a longer text' detectable (review #2 BLOCKER 4:
    the 4-gram short-short fix still missed a 5-word query embedded in a 20-word document).
    1-3-word queries stay exact-whole-text only, disclosed: banning every text that merely
    contains a 2-word entity name would remove topical overlap, not leaked content."""
    idx = {}
    for t in texts:
        w = norm_words(t)
        if SHORT_NGRAM <= len(w) < NGRAM:
            idx.setdefault(len(w), []).append(ngram_hashes(w)[0])
    return {k: np.unique(np.asarray(v, dtype=np.uint64)) for k, v in idx.items()}


def contains_short(words, whole_idx):
    """True if any protected short query occurs verbatim (word-level) inside `words`."""
    for k, arr in whole_idx.items():
        g = rolling_kgrams(words, k)
        if g.size == 0:
            continue
        i = np.minimum(np.searchsorted(arr, g, "left"), arr.size - 1)
        if bool((arr[i] == g).any()):
            return True
    return False


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


def stream_dev_component_docs(comp):
    import devsuite
    _, doc_texts, *_ = devsuite.load(comp)
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
    # untouched-final repair (LEDGER 2026-08-26): two unused CQADupStack subforums, chosen by a
    # rule fixed before any candidate number (alphabetically first two outside dev's pair).
    import devsuite
    for c in ("cqadup-android", "cqadup-english"):
        unt += devsuite.load(c)[3]
    qs["untouched-final"] = unt
    return qs


def protected_query_index():
    """(exact-key set, sorted 8-gram array) over every protected query: the six + dev +
    untouched-final. Shared by decontam.run, decontam_querytext, decontam_heldout and pseudoq so
    rule R1 has exactly one implementation."""
    pq = protected_queries()
    prot = [q for v in pq.values() for q in v]
    q_ex = set(int(exact_u64(q)) for q in prot)
    q_gram = np.unique(np.concatenate([query_grams(q) for q in prot]))
    return q_ex, q_gram, short_whole_index(prot), {k: len(v) for k, v in pq.items()}


def query_hits(text, q_ex, q_gram, q_whole=None):
    """R1 test for one candidate TRAIN query. -> 'exact' | 'near' | None.

    searchsorted, not np.isin: isin re-sorts q_gram on every call, which turned a 353K-pair scan
    into hours (see m7/CODEMAP.md)."""
    if int(exact_u64(text)) in q_ex:
        return "exact"
    g = query_grams(text)
    if g.size and q_gram.size:
        i = np.minimum(np.searchsorted(q_gram, g, "left"), q_gram.size - 1)
        if bool((q_gram[i] == g).any()):
            return "near"
    if q_whole and contains_short(norm_words(text), q_whole):
        return "contains"
    return None


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
    q_gram = np.unique(np.concatenate([query_grams(q) for q in prot_q]))  # sorted by np.unique
    q_whole = short_whole_index(prot_q)
    print(f"  query index: {len(q_ex):,} exact, {q_gram.size:,} 8-grams ({time.time()-t0:.0f}s)", flush=True)

    def shares_gram(g):
        """searchsorted, not np.isin: isin re-sorts the 345K-gram array on every call, which
        turned this 353K-pair scan into hours."""
        if g.size == 0:
            return False
        i = np.searchsorted(q_gram, g, "left")
        i = np.minimum(i, q_gram.size - 1)
        return bool((q_gram[i] == g).any())

    r1 = {}
    survive = []
    for n, (src, qid, query, pos, hneg) in enumerate(tr):
        if n and n % 100_000 == 0:
            print(f"    R1 {n:,}/{len(tr):,} ({time.time()-t0:.0f}s)", flush=True)
        if int(exact_u64(query)) in q_ex:
            r1[src] = r1.get(src, {"exact": 0, "near": 0})
            r1[src]["exact"] += 1
            continue
        if shares_gram(query_grams(query)) or contains_short(norm_words(query), q_whole):
            r1[src] = r1.get(src, {"exact": 0, "near": 0})
            r1[src]["near"] += 1
            continue
        survive.append((src, qid, pos))
    print(f"[R1] removed {sum(v['exact']+v['near'] for v in r1.values()):,} pairs; "
          f"{len(survive):,} survive ({time.time()-t0:.0f}s)", flush=True)

    # --- index the TRAIN positive documents (the direction that bounds memory) ------------
    store_of = {s: mix.load_source(s)["docstore"] for s in mix.available_sources()}
    stores, doc_key, doc_text_of = {}, {}, {}
    for n, (src, qid, pos) in enumerate(survive):
        if n and n % 100_000 == 0:
            print(f"    collecting positives {n:,}/{len(survive):,} "
                  f"({len(doc_key):,} unique docs, {time.time()-t0:.0f}s)", flush=True)
        st = store_of[src]
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
    sk, ex = [], []
    for n, k in enumerate(keys):
        if n and n % 200_000 == 0:
            print(f"    sketching {n:,}/{len(keys):,} ({time.time()-t0:.0f}s)", flush=True)
        sk.append(sketch(doc_text_of[k]))
        ex.append(exact_u64(doc_text_of[k]))
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
    nq_res, _ = sweep("nq-250k-dev", stream_dev_component_docs("nq-250k"), 250_000)
    dbp_res, _ = sweep("dbpedia-untouched", stream_beir_docs("dbpedia-entity"), 4_635_922)
    fev_res, _ = sweep("fever-untouched", stream_beir_docs("fever"), 5_416_568)

    def stream_cqa_untouched_docs():
        import devsuite
        for c in ("cqadup-android", "cqadup-english"):
            yield from devsuite.load(c)[1]
    cqa_unt_res, _ = sweep("cqadupstack-untouched", stream_cqa_untouched_docs())

    # --- R2 removal (the six only); R3 is disclosure --------------------------------------
    bad = {keys[i] for i in np.nonzero(six_hit)[0]}
    r2, kept = {}, {}
    for src, qid, pos in survive:
        st = store_of[src]
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
                                 "nq-250k-dev": nq_res,
                                 "dbpedia-entity-untouched": dbp_res,
                                 "fever-untouched": fev_res,
                                 "cqadupstack-untouched (android+english)": cqa_unt_res,
                                 "hotpotqa-dev": "100% by construction: hotpotqa-corpus IS the dev corpus"},
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
