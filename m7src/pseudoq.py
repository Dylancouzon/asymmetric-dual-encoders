"""Pseudo-queries for vocabulary-coverage distillation.

The pre-registered headwind is that no approved source supplies in-domain relevance pairs for
the six (scientific, biomedical, financial, argumentative) -- every candidate is on the
contamination map: S2ORC feeds SciFact and SciDocs, PubMed and NutritionFacts feed NFCorpus,
CORD-19 IS TREC-COVID's corpus, StackExchange-finance feeds FiQA, args.me feeds ArguAna. That
gap is unmitigable on the document and relevance side, and the report says so.

But objective B needs no labels -- only query text and the teacher's own embedding of it. So the
token->direction map can be taught over far more vocabulary than the 352K real queries cover, at
no licensing cost, by drawing short spans out of the TRAIN doc stores we already hold. This is a
VOCABULARY mitigation, not a domain mitigation: it supplies neither in-domain documents nor
relevance structure, and it is labeled that way everywhere it appears.
"""
import json
import re

import numpy as np
import torch

import mix
from _paths import WORK
from teacher import QUERY_PREFIX, encode_cached

OUT = WORK / "pseudoq"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 0
MAX_WORDS = 32
_SENT = re.compile(r"(?<=[.!?])\s+")


def _span(text):
    """First sentence, capped at MAX_WORDS -- query-shaped in length, not in grammar."""
    s = _SENT.split(text.strip(), 1)[0]
    w = s.split()
    return " ".join(w[:MAX_WORDS]) if len(w) >= 4 else None


def build(n, stores=None, seed=SEED):
    """Deterministic sample of n pseudo-queries, spread evenly over the doc stores."""
    p = OUT / f"pseudoq-{n}-{seed}.json"
    if p.exists():
        return json.loads(p.read_text())
    stores = stores or sorted({mix.load_source(s)["docstore"] for s in mix.available_sources()})
    rng = np.random.default_rng(seed)
    per = n // len(stores) + 1
    out = []
    for st in stores:
        _, texts = mix.load_store(st)
        idx = rng.choice(len(texts), size=min(per * 2, len(texts)), replace=False)
        got = 0
        for i in idx:
            s = _span(texts[int(i)])
            if s:
                out.append(s)
                got += 1
                if got >= per:
                    break
        del texts
        print(f"  pseudoq {st}: {got:,}", flush=True)
    out = out[:n]
    p.write_text(json.dumps(out))
    return out


def targets(texts, tag):
    return np.asarray(encode_cached(f"pseudoq-{tag}-{len(texts)}", texts, prefix=QUERY_PREFIX,
                                    dtype=torch.float16, verbose=True), dtype=np.float32)


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
    qs = build(n)
    print(f"{len(qs):,} pseudo-queries; sample:")
    for q in qs[:5]:
        print("   ", q[:110])
