"""Fingerprint decontamination: TRAIN vs DEV, KNOWN-TEST (the six), and UNTOUCHED-FINAL.

Two fingerprints, both deterministic and process-stable (blake2b, never Python's salted hash):
  exact       sha of the normalized text (lowercased, alphanumeric words, single-spaced)
  near-dup    word-8-gram rolling hashes, reduced to a bottom-64 sketch per text;
              two texts are near-duplicates when their sketches share >= 16 of 64 values
              (estimated Jaccard >= 0.25)

Pre-registered removal rules (counts logged to m7/LEDGER.md):
  R1  drop a training pair whose QUERY exactly matches, or shares >=1 word-8-gram with,
      any protected query (six + dev + untouched-final). Queries are the real leakage risk.
  R2  drop a training pair whose POSITIVE DOCUMENT exactly matches or near-duplicates any
      document of the six (KNOWN-TEST). This is the contamination map enforced at fingerprint
      level rather than by source name.
  R3  drop a training pair whose positive near-duplicates a CQADupStack dev document, and any
      DBpedia-entity (untouched-final) document.

Reported, NOT removed, with the rationale in the ledger: hotpotqa-corpus IS the dev HotpotQA
corpus and fever-pos comes from the untouched-final FEVER corpus, so document overlap there is
inherent to using those training sets at all. Removing it would delete the sources rather than
decontaminate them; the protection that matters -- their test QUERIES and QRELS -- is enforced
by R1 and by the final-scorer ledger. Counts are disclosed in the report.
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
SKETCH = 64
DUP_SHARE = 16
_BASE = np.uint64(1_000_003)
_wcache = {}


def _wh(w):
    v = _wcache.get(w)
    if v is None:
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


def exact_key(text):
    return hashlib.blake2b(" ".join(norm_words(text)).encode(), digest_size=16).digest()


def ngram_hashes(words):
    """Rolling polynomial word-8-gram hashes, vectorized. Short texts hash as one whole-text gram."""
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
    h = np.unique(ngram_hashes(norm_words(text)))
    return h[:SKETCH]


def build_protected(texts, want_sketch=True, label=""):
    """-> (set of exact keys, sorted uint64 array of all n-gram hashes or sketch hashes)."""
    ex = set()
    grams = []
    t0 = time.time()
    for i, t in enumerate(texts):
        ex.add(exact_key(t))
        grams.append(sketch(t) if want_sketch else np.unique(ngram_hashes(norm_words(t))))
        if label and i and i % 250_000 == 0:
            print(f"    {label} {i:,}/{len(texts):,} @ {i/(time.time()-t0):.0f}/s", flush=True)
    g = np.unique(np.concatenate(grams)) if grams else np.zeros(0, dtype=np.uint64)
    return ex, g


def hits(text, ex_set, gram_sorted, want_sketch, min_share):
    if exact_key(text) in ex_set:
        return "exact"
    h = sketch(text) if want_sketch else np.unique(ngram_hashes(norm_words(text)))
    if h.size == 0 or gram_sorted.size == 0:
        return None
    n = int(np.isin(h, gram_sorted, assume_unique=True).sum())
    return "near" if n >= min_share else None


# ---- protected sets ------------------------------------------------------------------

def six_queries_and_docs():
    """Six-set QUERIES come from the vendored results/frozen_eval/ (the only authorized reader
    of six-set labels is the final scorer; queries here are text, not qrels). Documents come
    from the HF corpora that eval_manifest.json already pinned on this machine."""
    import os
    os.environ.setdefault("BENCH_DATASETS", "scifact,nfcorpus,fiqa,arguana,scidocs,trec-covid")
    from core import DATASETS, load_beir
    qs, docs = [], []
    for ds in DATASETS:
        froz = json.loads((REPO / "results" / "frozen_eval" / f"{ds}.json").read_text())
        qs += list(froz["queries"].values())
        _, doc_texts, *_ = load_beir(ds)
        docs += doc_texts
    return qs, docs


def dev_queries_and_cqa_docs():
    import devsuite
    qs, docs = [], []
    for c in devsuite.COMPONENTS:
        _, doc_texts, _, q_texts, _ = devsuite.load(c)
        qs += q_texts
        if c.startswith("cqadup-"):
            docs += doc_texts
    return qs, docs


def untouched_queries_and_dbpedia_docs():
    from datasets import load_dataset
    from core import doc_text
    qs, docs = [], []
    for ds in ("fever", "dbpedia-entity"):
        qrels = {str(r["query-id"]) for r in load_dataset(f"BeIR/{ds}-qrels", split="test")}
        q = load_dataset(f"BeIR/{ds}", "queries")["queries"]
        qs += [t for i, t in zip(list(q["_id"]), list(q["text"])) if str(i) in qrels]
        if ds == "dbpedia-entity":
            docs += [doc_text(r) for r in load_dataset(f"BeIR/{ds}", "corpus")["corpus"]]
    return qs, docs


# ---- runner --------------------------------------------------------------------------

def run():
    import mix
    t0 = time.time()
    print("[protected] six", flush=True)
    six_q, six_d = six_queries_and_docs()
    print("[protected] dev", flush=True)
    dev_q, cqa_d = dev_queries_and_cqa_docs()
    print("[protected] untouched-final", flush=True)
    unt_q, dbp_d = untouched_queries_and_dbpedia_docs()

    print(f"[index] queries: six {len(six_q):,} + dev {len(dev_q):,} + untouched {len(unt_q):,}", flush=True)
    q_ex, q_gram = build_protected(six_q + dev_q + unt_q, want_sketch=False, label="queries")
    print(f"  query index: {len(q_ex):,} exact keys, {q_gram.size:,} 8-grams  ({time.time()-t0:.0f}s)", flush=True)

    doc_sets = {}
    for name, texts in (("six", six_d), ("cqadupstack-dev", cqa_d), ("dbpedia-untouched", dbp_d)):
        print(f"[index] docs {name}: {len(texts):,}", flush=True)
        doc_sets[name] = build_protected(texts, want_sketch=True, label=f"docs {name}")
        print(f"  {name}: {len(doc_sets[name][0]):,} exact keys, {doc_sets[name][1].size:,} sketch hashes"
              f"  ({time.time()-t0:.0f}s)", flush=True)

    tr, ho = mix.split_pairs()
    stores = {}
    removed = {"R1_query": {}, "R2_six_doc": {}, "R3_dev_untouched_doc": {}}
    keep_pairs, drop_log = [], []
    seen_doc = {}
    print(f"[scan] {len(tr):,} TRAIN pairs", flush=True)
    for i, (src, qid, query, pos, hneg) in enumerate(tr):
        if i and i % 50_000 == 0:
            print(f"    {i:,}/{len(tr):,} ({time.time()-t0:.0f}s)", flush=True)
        r1 = hits(query, q_ex, q_gram, want_sketch=False, min_share=1)
        if r1:
            removed["R1_query"][src] = removed["R1_query"].get(src, 0) + 1
            drop_log.append({"rule": "R1", "kind": r1, "source": src, "qid": qid, "query": query[:200]})
            continue
        store = mix.load_source(src)["docstore"]
        if store not in stores:
            ids, texts = mix.load_store(store)
            stores[store] = dict(zip(ids, texts))
        smap = stores[store]
        bad = None
        for d in pos:
            t = smap.get(d)
            if t is None:
                continue
            k = (store, d)
            if k not in seen_doc:
                seen_doc[k] = {n: hits(t, *doc_sets[n], want_sketch=True, min_share=DUP_SHARE)
                               for n in doc_sets}
            v = seen_doc[k]
            if v["six"]:
                bad = ("R2_six_doc", v["six"], d)
                break
            if v["cqadupstack-dev"] or v["dbpedia-untouched"]:
                bad = ("R3_dev_untouched_doc", v["cqadupstack-dev"] or v["dbpedia-untouched"], d)
                break
        if bad:
            removed[bad[0]][src] = removed[bad[0]].get(src, 0) + 1
            drop_log.append({"rule": bad[0], "kind": bad[1], "source": src, "qid": qid, "docid": bad[2]})
            continue
        keep_pairs.append({"source": src, "qid": qid})

    kept = {}
    for p in keep_pairs:
        kept.setdefault(p["source"], set()).add(p["qid"])
    (OUT / "kept.json").write_text(json.dumps({k: sorted(v) for k, v in kept.items()}))
    (OUT / "dropped.json").write_text(json.dumps(drop_log[:20000]))
    summary = {
        "n_train_pairs_in": len(tr), "n_train_pairs_kept": len(keep_pairs),
        "removed": removed,
        "removed_total": {k: sum(v.values()) for k, v in removed.items()},
        "protected": {"six_queries": len(six_q), "dev_queries": len(dev_q),
                      "untouched_queries": len(unt_q), "six_docs": len(six_d),
                      "cqadupstack_dev_docs": len(cqa_d), "dbpedia_untouched_docs": len(dbp_d)},
        "params": {"ngram": NGRAM, "sketch": SKETCH, "dup_share": DUP_SHARE,
                   "hash": "blake2b-64 word hashes, polynomial rolling n-gram"},
        "not_removed_by_construction": {
            "hotpotqa-corpus_is_dev_hotpotqa_corpus": True,
            "fever-pos_from_untouched_fever_corpus": True,
            "rationale": "same-corpus document overlap is inherent to using these training sets; "
                         "their test queries and qrels stay protected by R1 and the final-scorer ledger",
        },
        "seconds": round(time.time() - t0, 1),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    (REPO / "results" / "m7_decontam.json").write_text(json.dumps(summary, indent=1))
    print("\n" + json.dumps({k: summary[k] for k in ("n_train_pairs_in", "n_train_pairs_kept",
                                                     "removed_total")}, indent=1), flush=True)
    print(json.dumps(removed, indent=1), flush=True)


if __name__ == "__main__":
    run()
