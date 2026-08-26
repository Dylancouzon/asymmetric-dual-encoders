"""The held-out training slices as dev retrieval components.

The mandate pins "held-out training slices (pairs with sha256(pair_id) mod 50 == 0), including a
long-query slice (held-out queries >= 64 WordPiece tokens)". The mod-50 rule is applied at QUERY
granularity (see m7src/trainmix.py) so a held-out query never appears in TRAIN.

A held-out pair alone is not a retrieval task, so each slice gets a corpus: every held-out
positive plus random distractors drawn from the frozen pool, to TARGET_DOCS, seed 0. Document
vectors come straight out of the pool -- no new encode.
"""
import json

import numpy as np

import mix
import pool as poolmod
from _paths import WORK
from table import get_tokenizer
from trainmix import heldout

HELD = WORK / "dev"
TARGET_DOCS = 200_000
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

    rng = np.random.default_rng(SEED)
    out = {}
    for name, sel in (("heldout-train", rows),
                      ("heldout-longq", [r for r in rows if r["n_tokens"] >= LONG_TOKENS])):
        if not sel:
            out[name] = None
            continue
        pos = sorted({j for r in sel for j in r["pos"]})
        need = max(0, TARGET_DOCS - len(pos))
        pool_n = len(pool_vecs)
        extra = rng.choice(pool_n, size=min(need, pool_n), replace=False)
        keep = sorted(set(pos) | {int(x) for x in extra})
        pos_at = {g: i for i, g in enumerate(keep)}
        doc_ids = [str(g) for g in keep]
        qrels = {r["qid"]: {str(g): 1 for g in r["pos"]} for r in sel}
        blob = {"doc_pool_idx": keep, "doc_ids": doc_ids,
                "q_ids": [r["qid"] for r in sel], "q_texts": [r["text"] for r in sel],
                "qrels": qrels, "n_tokens": [r["n_tokens"] for r in sel],
                "by_source": {s: sum(1 for r in sel if r["src"] == s)
                              for s in sorted({r["src"] for r in sel})}}
        (HELD / f"{name}.json").write_text(json.dumps(blob))
        print(f"  {name}: {len(doc_ids):,} docs, {len(sel):,} queries, sources {blob['by_source']}",
              flush=True)
        out[name] = blob
    return out


def load(name):
    p = HELD / f"{name}.json"
    if not p.exists():
        _build()
    if not p.exists():
        return None
    b = json.loads(p.read_text())
    return b["doc_ids"], None, b["q_ids"], b["q_texts"], b["qrels"], b["doc_pool_idx"]


def doc_vectors(name, chunk=100_000):
    """Materialize this slice's corpus vectors out of the pool (200K x 768 fp16 = 0.31 GB)."""
    _, pool_vecs, _ = poolmod.build()
    _, _, _, _, _, idx = load(name)
    idx = np.asarray(idx)
    out = np.empty((len(idx), pool_vecs.shape[1]), dtype=np.float16)
    for lo in range(0, len(idx), chunk):
        out[lo:lo + chunk] = pool_vecs[idx[lo:lo + chunk]]
    return out


if __name__ == "__main__":
    _build()
