"""Dev-suite evaluation: candidate tables and the pinned reference rows.

Macro = equal weight per component (instructions-m7.md). Per-query vectors are returned so
every dev comparison can be paired-bootstrapped with the same machinery as the final run.
"""
import json

import numpy as np
import torch

import devsuite
from _paths import WORK
from evalkit import macro, score
from table import Preproc
from teacher import QUERY_PREFIX, encode_cached

DEVRES = WORK / "devres"
DEVRES.mkdir(parents=True, exist_ok=True)
CHUNK = {"hotpotqa": 250_000}


def dev_components():
    """The pinned dev suite: the four text-backed components plus whichever held-out training
    slices exist. heldout-longq is absent if the training mix has no queries that long -- which
    is itself a reportable fact about the mix, not a silent omission."""
    import heldout
    out = list(devsuite.COMPONENTS)
    for c in heldout.COMPONENTS:
        if (WORK / "dev" / f"{c}.json").exists():
            out.append(c)
    return out


_HELD_CACHE = {}


def doc_vecs(comp):
    if comp.startswith("heldout-"):
        import heldout
        if comp not in _HELD_CACHE:
            doc_ids, _, q_ids, q_texts, qrels, _ = heldout.load(comp)
            _HELD_CACHE[comp] = (doc_ids, None, q_ids, q_texts, qrels, heldout.doc_vectors(comp))
        return _HELD_CACHE[comp]
    doc_ids, doc_texts, q_ids, q_texts, qrels = devsuite.load(comp)
    dv = encode_cached(f"dev-{comp}-docs", doc_texts, prefix="", dtype=torch.float16, verbose=False)
    return doc_ids, doc_texts, q_ids, q_texts, qrels, dv


def eval_query_vecs(comp, qv):
    doc_ids, _, q_ids, _, qrels, dv = doc_vecs(comp)
    return score(qv, q_ids, dv, doc_ids, qrels, chunk=CHUNK.get(comp, 200_000))


def eval_table(model, pre: Preproc, components=None, tok=None):
    out = {}
    for c in (components or dev_components()):
        _, _, _, q_texts, _, _ = doc_vecs(c)
        out[c] = eval_query_vecs(c, model.encode(q_texts, pre, tok=tok))
    return out


# ---- reference rows ------------------------------------------------------------------

def ref_bge_base(comp, prefix=True):
    _, _, _, q_texts, _, _ = doc_vecs(comp)
    if comp.startswith("heldout-"):
        return eval_query_vecs(comp, np.asarray(encode_cached(
            f"dev-{comp}-q-{'pfx' if prefix else 'nopfx'}", q_texts,
            prefix=QUERY_PREFIX if prefix else "", dtype=torch.float16, verbose=False),
            dtype=np.float32))
    qv = encode_cached(f"dev-{comp}-queries-{'pfx' if prefix else 'nopfx'}", q_texts,
                       prefix=QUERY_PREFIX if prefix else "", dtype=torch.float16, verbose=False)
    return eval_query_vecs(comp, np.asarray(qv, dtype=np.float32))


def ref_bm25(comp):
    """BM25 needs document TEXT, which the held-out slices do not carry (their corpora are pool
    row indices). Those components therefore have no BM25 reference row; the gate's BM25
    comparison runs on the four text-backed components, and the report says so."""
    import Stemmer
    import bm25s
    from evalkit import per_query_ndcg
    doc_ids, doc_texts, q_ids, q_texts, qrels, _ = doc_vecs(comp)
    if doc_texts is None:
        return None
    st = Stemmer.Stemmer("english")
    r = bm25s.BM25(method="lucene", k1=1.2, b=0.75)
    r.index(bm25s.tokenize(doc_texts, stopwords="en", stemmer=st, show_progress=False), show_progress=False)
    ids, sc = r.retrieve(bm25s.tokenize(q_texts, stopwords="en", stemmer=st, show_progress=False),
                         k=min(1000, len(doc_ids)), show_progress=False)
    run = {qid: {doc_ids[d]: float(s) for d, s in zip(ids[qi], sc[qi]) if doc_ids[d] != qid}
           for qi, qid in enumerate(q_ids)}
    return per_query_ndcg(run, qrels)


def ref_potion(comp, chunk_docs=250_000):
    """Chunked into a preallocated fp16 buffer: the 5.23M-doc HotpotQA component would be
    10.7 GB as one fp32 array, and this box has 25 GB total. Returns None for the held-out
    slices, whose corpora are pool row indices and carry no document text."""
    from model2vec import StaticModel
    doc_ids, doc_texts, q_ids, q_texts, qrels, _ = doc_vecs(comp)
    if doc_texts is None:
        return None
    m = StaticModel.from_pretrained("minishlab/potion-retrieval-32M")
    nrm = lambda v: v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
    qv = nrm(m.encode(q_texts, show_progress_bar=False).astype(np.float32))
    dim = qv.shape[1]
    dv = np.empty((len(doc_texts), dim), dtype=np.float16)
    for lo in range(0, len(doc_texts), chunk_docs):
        hi = min(lo + chunk_docs, len(doc_texts))
        dv[lo:hi] = nrm(m.encode(doc_texts[lo:hi], show_progress_bar=False).astype(np.float32))
    return score(qv, q_ids, dv, doc_ids, qrels, chunk=CHUNK.get(comp, 200_000))


REFS = {"bm25": ref_bm25, "potion-retrieval-32M": ref_potion,
        "bge-base-symmetric": lambda c: ref_bge_base(c, True),
        "bge-base-symmetric-nopfx": lambda c: ref_bge_base(c, False)}


def reference_rows(components=None, names=None, cache=True):
    """Computed once, cached to work/devres/refs.json (per-query vectors kept for pairing)."""
    p = DEVRES / "refs.json"
    blob = json.loads(p.read_text()) if p.exists() else {}
    for name in (names or REFS):
        for c in (components or dev_components()):
            if name in blob and c in blob[name]:
                continue
            pq = REFS[name](c)
            if pq is None:
                print(f"  ref {name}/{c}: not applicable (no document text)", flush=True)
                continue
            blob.setdefault(name, {})[c] = {k: round(v, 6) for k, v in pq.items()}
            print(f"  ref {name}/{c}: {np.mean(list(pq.values())):.4f}", flush=True)
            if cache:
                p.write_text(json.dumps(blob))
    return blob


def report(per_component, label=""):
    m, means = macro(per_component)
    print(f"{label:34s} macro {m:.4f}  " +
          "  ".join(f"{k.replace('cqadup-','cqa-')}={v:.4f}" for k, v in means.items()), flush=True)
    return m, means


if __name__ == "__main__":
    import sys
    comps = sys.argv[1:] or None
    blob = reference_rows(components=comps)
    for name, per in blob.items():
        report({c: v for c, v in per.items() if comps is None or c in comps}, f"[ref] {name}")
