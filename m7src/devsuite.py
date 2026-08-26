"""The M7 dev suite, pinned by instructions-m7.md. Construction is deterministic (seed 0).

Components (macro = equal weight per component):
  nq-250k             BEIR NQ, all qrels-positive docs + random distractors to 250K, all test queries
  hotpotqa            BEIR HotpotQA, full corpus, all test queries
  cqadup-programmers  mteb/cqadupstack-programmers (non-Wikipedia, real qrels)
  cqadup-physics      mteb/cqadupstack-physics
  heldout-*           held-out training slices, built by trainmix.py (sha256(pair_id) % 50 == 0)

Touche is banned (args.me is ArguAna's source family); Quora is banned (no license).
Six-set qrels are never read here.
"""
import hashlib
import json

import numpy as np
from datasets import load_dataset

from _paths import REPO, WORK
from core import doc_text

CACHE = WORK / "dev"
CACHE.mkdir(parents=True, exist_ok=True)
NQ_TARGET_DOCS = 250_000
SEED = 0
COMPONENTS = ["nq-250k", "hotpotqa", "cqadup-programmers", "cqadup-physics"]


def sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def _beir(name, split="test"):
    corpus = load_dataset(f"BeIR/{name}", "corpus")["corpus"]
    queries = load_dataset(f"BeIR/{name}", "queries")["queries"]
    qrels_rows = load_dataset(f"BeIR/{name}-qrels", split=split)
    qrels = {}
    for r in qrels_rows:
        qrels.setdefault(str(r["query-id"]), {})[str(r["corpus-id"])] = int(r["score"])
    doc_ids = [str(x) for x in corpus["_id"]]
    doc_texts = [doc_text(r) for r in corpus]
    q_ids, q_texts = [], []
    for qid, text in zip(queries["_id"], queries["text"]):
        if str(qid) in qrels:
            q_ids.append(str(qid))
            q_texts.append(text)
    return doc_ids, doc_texts, q_ids, q_texts, qrels


def _mteb_cqa(sub):
    corpus = load_dataset(f"mteb/cqadupstack-{sub}", "corpus")["corpus"]
    queries = load_dataset(f"mteb/cqadupstack-{sub}", "queries")["queries"]
    qrels = {}
    for r in load_dataset(f"mteb/cqadupstack-{sub}", "default", split="test"):
        # every row in both subsets is score 1 (verified), so no filter is applied and the
        # component hashes in results/m7_dev_manifest.json stay valid
        qrels.setdefault(str(r["query-id"]), {})[str(r["corpus-id"])] = int(r["score"])
    doc_ids = [str(x) for x in corpus["_id"]]
    doc_texts = [doc_text(r) for r in corpus]
    q_ids, q_texts = [], []
    for qid, text in zip(queries["_id"], queries["text"]):
        if str(qid) in qrels:
            q_ids.append(str(qid))
            q_texts.append(text)
    return doc_ids, doc_texts, q_ids, q_texts, qrels


def _build(name):
    if name == "nq-250k":
        doc_ids, doc_texts, q_ids, q_texts, qrels = _beir("nq")
        pos = {d for v in qrels.values() for d in v}
        keep_pos = [i for i, d in enumerate(doc_ids) if d in pos]
        rest = np.array([i for i, d in enumerate(doc_ids) if d not in pos])
        n_extra = max(0, NQ_TARGET_DOCS - len(keep_pos))
        extra = np.random.default_rng(SEED).choice(rest, size=min(n_extra, len(rest)), replace=False)
        keep = sorted(set(keep_pos) | set(int(x) for x in extra))
        return ([doc_ids[i] for i in keep], [doc_texts[i] for i in keep], q_ids, q_texts, qrels)
    if name == "hotpotqa":
        return _beir("hotpotqa")
    if name.startswith("cqadup-"):
        return _mteb_cqa(name.split("-", 1)[1])
    raise KeyError(name)


def load(name):
    """Cached loader: returns (doc_ids, doc_texts, q_ids, q_texts, qrels)."""
    p = CACHE / f"{name}.json"
    if p.exists():
        b = json.loads(p.read_text())
        return b["doc_ids"], b["doc_texts"], b["q_ids"], b["q_texts"], b["qrels"]
    doc_ids, doc_texts, q_ids, q_texts, qrels = _build(name)
    p.write_text(json.dumps({"doc_ids": doc_ids, "doc_texts": doc_texts, "q_ids": q_ids,
                             "q_texts": q_texts, "qrels": qrels}))
    return doc_ids, doc_texts, q_ids, q_texts, qrels


def manifest_entry(name):
    doc_ids, doc_texts, q_ids, q_texts, qrels = load(name)
    return {"n_docs": len(doc_ids), "n_queries": len(q_ids),
            "corpus_ids_sha256": sha(doc_ids), "corpus_text_sha256": sha(doc_texts),
            "qids_sha256": sha(sorted(q_ids)), "qrels_sha256": sha(qrels),
            "construction": {"nq-250k": f"BEIR NQ: all qrels-positive docs + rng(seed={SEED}) distractors to {NQ_TARGET_DOCS}",
                             "hotpotqa": "BEIR HotpotQA test split, full corpus",
                             "cqadup-programmers": "mteb/cqadupstack-programmers, full corpus + test qrels",
                             "cqadup-physics": "mteb/cqadupstack-physics, full corpus + test qrels"}[name]}


if __name__ == "__main__":
    out = {}
    for c in COMPONENTS:
        out[c] = manifest_entry(c)
        print(f"{c:20s} {out[c]['n_docs']:>9,} docs  {out[c]['n_queries']:>6,} queries", flush=True)
    (REPO / "results" / "m7_dev_manifest.json").write_text(json.dumps(out, indent=1))
    print("wrote results/m7_dev_manifest.json")
