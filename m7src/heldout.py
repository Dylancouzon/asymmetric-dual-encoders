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


def _warn_if_pinned(name):
    """A rebuild that reproduces the pinned bytes is fine and common (the build is deterministic);
    one that does NOT is a protocol event, so say so loudly at the moment it happens rather than
    leaving it to whoever next calls verify_pinned()."""
    import hashlib

    from _paths import REPO
    man_p = REPO / "results" / "m7_dev_manifest.json"
    want = (json.loads(man_p.read_text()).get(name) or {}).get("json_sha256") \
        if man_p.exists() else None
    if want and hashlib.sha256((HELD / f"{name}.json").read_bytes()).hexdigest() != want:
        print(f"  !! {name} REBUILT WITH DIFFERENT BYTES than the pinned manifest -- every dev "
              f"number computed before this point used a different component", flush=True)


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
        _warn_if_pinned(name)
        print(f"  {name}: corpus = the full pool ({pool_n:,} docs), {len(sel):,} queries, "
              f"sources {blob['by_source']}", flush=True)
        out[name] = blob
    return out


_VERIFIED = False


def verify_pinned():
    """Refuse to serve a held-out component whose bytes differ from the pinned manifest.

    These two JSONs are DERIVED from the training mix and the doc pool, so without this a changed
    mix, a reordered pool or a regenerated file would move the dev macro while every hash
    `freeze.py` checks stayed valid (Codex review #3 BLOCKER 2). Checked once per process."""
    global _VERIFIED
    if _VERIFIED:
        return
    import hashlib

    from _paths import REPO
    man_p = REPO / "results" / "m7_dev_manifest.json"
    man = json.loads(man_p.read_text()) if man_p.exists() else {}
    for name in COMPONENTS:
        want = (man.get(name) or {}).get("json_sha256")
        if not want:
            continue                                   # not pinned yet: freeze_heldout.py does that
        p = HELD / f"{name}.json"
        if not p.exists():
            raise SystemExit(f"pinned dev component {name} is missing ({p})")
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        if got != want:
            raise SystemExit(
                f"PINNED dev component {name} changed: {p} hashes {got[:16]}..., manifest says "
                f"{want[:16]}.... A dev component may not change under a selection. Restore it, "
                f"or re-pin deliberately with freeze_heldout.py and disclose it in m7/LEDGER.md.")
    _VERIFIED = True


def load(name):
    p = HELD / f"{name}.json"
    if not p.exists():
        _build()
    if not p.exists():
        return None
    verify_pinned()
    b = json.loads(p.read_text())
    return pool_doc_ids(b["n_docs"]), None, b["q_ids"], b["q_texts"], b["qrels"], None


_DOC_IDS = {}
_POOL_VECS = None


def pool_doc_ids(n):
    """ONE shared list of pool row ids. Both held-out components address the same 6.17M-row pool,
    and a per-component copy is ~400 MB of Python strings each -- and, worse, makes two callers
    that are looking at the identical corpus unable to prove it (multieval's shared-pass check)."""
    if n not in _DOC_IDS:
        _DOC_IDS[n] = [str(i) for i in range(n)]
    return _DOC_IDS[n]


def doc_vectors(name):
    """The pool memmap itself: the corpus IS the pool, so nothing is copied. Memoized so every
    held-out component gets the SAME memmap object, not an equal one."""
    global _POOL_VECS
    if _POOL_VECS is None:
        _, _POOL_VECS, _ = poolmod.build()
    return _POOL_VECS


if __name__ == "__main__":
    _build()
