"""The held-out training slices as dev retrieval components.

The mandate pins "held-out training slices (pairs with sha256(pair_id) mod 50 == 0), including a
long-query slice (held-out queries >= 64 WordPiece tokens)". The mod-50 rule is applied at QUERY
granularity (see m7src/trainmix.py) so a held-out query never appears in TRAIN.

A held-out pair alone is not a retrieval task, so each slice needs a corpus. The corpus is the
ENTIRE frozen doc pool (6.17M documents), not a sample.

The first version sampled ~200K random distractors, and the teacher scored 0.8383 on
heldout-train and 0.9915 on heldout-longq: a random distractor drawn from 6M documents is almost
never confusable with the true positive, so both components were near-saturated and could not
discriminate between candidates. A dev component that every candidate passes contributes noise to
the macro, not signal -- and with equal per-component weighting it would have made the go/no-go
gate easier to pass for no reason. Using the whole pool makes the task as hard as a real 6M-document
retrieval task, with no teacher-derived bias (mining hard distractors with the teacher would have
biased the component toward the teacher's own ranking, which is the thing being measured).

It is also cheaper: document vectors are the pool memmap itself, so nothing is copied.
"""
import json

import numpy as np

import mix
import pool as poolmod
from _paths import WORK
from table import get_tokenizer
from trainmix import heldout

HELD = WORK / "dev"
SEED = 0
LONG_TOKENS = 64
COMPONENTS = ["heldout-train", "heldout-longq"]


def _build():
    index, pool_vecs, meta = poolmod.build()
    tok = get_tokenizer()
    rows = []
    for src in mix.available_sources():
        st = mix.load_source(src)["docstore"]
        for p in mix.load_source(src)["pairs"]:
            if not heldout(src, p["qid"]):
                continue
            ps = [j for j in (index.get(st, d) for d in p["pos"]) if j is not None]
            if ps:
                rows.append({"src": src, "qid": f"{src}:{p['qid']}", "text": p["query"], "pos": ps})
        index.drop(st)
    n_tok = [len(tok(r["text"], add_special_tokens=True, truncation=True, max_length=512)["input_ids"])
             for r in rows]
    for r, n in zip(rows, n_tok):
        r["n_tokens"] = n
    print(f"  held-out queries: {len(rows):,}  token length p50={np.percentile(n_tok,50):.0f} "
          f"p90={np.percentile(n_tok,90):.0f} max={max(n_tok)}  "
          f">= {LONG_TOKENS} tokens: {sum(1 for n in n_tok if n >= LONG_TOKENS):,}", flush=True)

    out = {}
    pool_n = len(pool_vecs)
    for name, sel in (("heldout-train", rows),
                      ("heldout-longq", [r for r in rows if r["n_tokens"] >= LONG_TOKENS])):
        if not sel:
            out[name] = None
            continue
        qrels = {r["qid"]: {str(g): 1 for g in r["pos"]} for r in sel}
        blob = {"corpus": "full-pool", "n_docs": pool_n,
                "q_ids": [r["qid"] for r in sel], "q_texts": [r["text"] for r in sel],
                "qrels": qrels, "n_tokens": [r["n_tokens"] for r in sel],
                "by_source": {s: sum(1 for r in sel if r["src"] == s)
                              for s in sorted({r["src"] for r in sel})}}
        (HELD / f"{name}.json").write_text(json.dumps(blob))
        print(f"  {name}: corpus = the full pool ({pool_n:,} docs), {len(sel):,} queries, "
              f"sources {blob['by_source']}", flush=True)
        out[name] = blob
    return out


def load(name):
    p = HELD / f"{name}.json"
    if not p.exists():
        _build()
    if not p.exists():
        return None
    b = json.loads(p.read_text())
    doc_ids = [str(i) for i in range(b["n_docs"])]
    return doc_ids, None, b["q_ids"], b["q_texts"], b["qrels"], None


def doc_vectors(name):
    """The pool memmap itself: the corpus IS the pool, so nothing is copied."""
    _, pool_vecs, _ = poolmod.build()
    return pool_vecs


if __name__ == "__main__":
    _build()
