"""Bring-up step 3 — authorized six-set access class (a): harness validation, pre-freeze.

Reproduces the three named cells from instructions-m7.md to <=0.003:
  bge-small ArguAna 0.6034 · bge-small SciFact 0.7127 · bm25 FiQA 0.2532
Writes nothing to results/quality.json. Logged to m7/LEDGER.md by the caller.
"""
import json
import sys

import numpy as np
import torch
import transformers
from sentence_transformers import SentenceTransformer

from _paths import REPO  # noqa: F401
from core import evaluate, load_beir, score_run  # noqa: E402

TARGETS = {("bge-small-en-v1.5", "arguana"): 0.6034,
           ("bge-small-en-v1.5", "scifact"): 0.7127,
           ("bm25", "fiqa"): 0.2532}
TOL = 0.003
BGE_Q = "Represent this sentence for searching relevant passages: "


def bge_small(ds):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = SentenceTransformer("BAAI/bge-small-en-v1.5", device=dev, model_kwargs={"dtype": torch.float32})
    doc_ids, doc_texts, q_ids, q_texts, qrels = load_beir(ds)
    enc = lambda ts, pre: m.encode([pre + t for t in ts], batch_size=256, normalize_embeddings=True,
                                   show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
    return evaluate(doc_ids, enc(doc_texts, ""), q_ids, enc(q_texts, BGE_Q), qrels)


def bm25(ds):
    import Stemmer
    import bm25s
    st = Stemmer.Stemmer("english")
    doc_ids, doc_texts, q_ids, q_texts, qrels = load_beir(ds)
    r = bm25s.BM25(method="lucene", k1=1.2, b=0.75)
    r.index(bm25s.tokenize(doc_texts, stopwords="en", stemmer=st, show_progress=False), show_progress=False)
    ids, sc = r.retrieve(bm25s.tokenize(q_texts, stopwords="en", stemmer=st, show_progress=False),
                         k=min(1000, len(doc_ids)), show_progress=False)
    run = {qid: {doc_ids[d]: float(s) for d, s in zip(ids[qi], sc[qi]) if doc_ids[d] != qid}
           for qi, qid in enumerate(q_ids)}
    return score_run(run, qrels)


RUNNERS = {"bge-small-en-v1.5": bge_small, "bm25": bm25}

out, fail = {}, []
for (sysname, ds), want in TARGETS.items():
    got = RUNNERS[sysname](ds)["ndcg@10"]
    ok = abs(got - want) <= TOL
    out[f"{sysname}/{ds}"] = {"got": round(got, 4), "expected": want, "delta": round(got - want, 4), "pass": ok}
    print(f"{'PASS' if ok else 'FAIL'} {sysname}/{ds}: {got:.4f} vs {want:.4f} (d={got-want:+.4f})", flush=True)
    if not ok:
        fail.append(f"{sysname}/{ds}")

out["_meta"] = {"tol": TOL, "torch": torch.__version__, "transformers": transformers.__version__,
                "device": "cuda" if torch.cuda.is_available() else "cpu"}
(REPO / "results" / "m7_harness_validation.json").write_text(json.dumps(out, indent=1))
print("\n" + ("HARNESS VALIDATION FAILED: " + ", ".join(fail) if fail else "OK: harness validated on the new machine."))
sys.exit(1 if fail else 0)
