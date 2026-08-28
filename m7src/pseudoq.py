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
# Long spans (kind="mixed"), capacity lever #7. The budget brackets ArguAna's ~250-word queries,
# which are 1 of the 6 confirmatory sets and the architecture's pre-identified worst case; the
# lower end keeps the long half clearly separated from the <=32-word short half.
LONG_MIN_WORDS, LONG_MAX_WORDS = 64, 320
KINDS = ("short", "mixed")
_SENT = re.compile(r"(?<=[.!?])\s+")


def _span(text):
    """First sentence, capped at MAX_WORDS -- query-shaped in length, not in grammar."""
    s = _SENT.split(text.strip(), 1)[0]
    w = s.split()
    return " ".join(w[:MAX_WORDS]) if len(w) >= 4 else None


def _long_span(text, rng):
    """A contiguous run of whole sentences from a random start, to a word budget in
    [LONG_MIN_WORDS, LONG_MAX_WORDS].

    Sentence-aligned rather than a raw word window so the span reads like the argumentative
    passages it is standing in for, and started at a random sentence rather than the first so the
    long half is not just "the short half plus more of the same lead paragraph". Returns None when
    the tail from that start cannot reach the budget's floor -- short documents contribute to the
    short half instead of contributing a truncated long span.
    """
    sents = [s for s in _SENT.split(text.strip()) if s.strip()]
    if not sents:
        return None
    budget = int(rng.integers(LONG_MIN_WORDS, LONG_MAX_WORDS + 1))
    start = int(rng.integers(0, len(sents)))
    out, n = [], 0
    for s in sents[start:]:
        out.append(s)
        n += len(s.split())
        if n >= budget:
            break
    if n < LONG_MIN_WORDS:
        return None
    return " ".join(" ".join(out).split()[:budget])


def path(n, seed=SEED, kind="short"):
    """The pool's filename. `short` keeps its historical name so every committed cache, every
    `kept-*.json` and every encode key stays valid; a new kind gets a new name, which is also what
    makes `decontam_querytext.py`'s `pseudoq-*.json` glob pick it up with no change."""
    if kind not in KINDS:
        raise KeyError(f"unknown pseudo-query kind {kind!r}; known {KINDS}")
    return OUT / (f"pseudoq-{n}-{seed}.json" if kind == "short"
                  else f"pseudoq-{kind}{n}-{seed}.json")


def build(n, stores=None, seed=SEED, kind="short"):
    """Deterministic sample of n pseudo-queries, spread evenly over the doc stores.

    kind="short": first-sentence spans capped at 32 words -- the historical behaviour.
    kind="mixed": half those, half long sentence-aligned spans (lever #7). Half and half rather
    than all-long because the probe found a length GAP, not that short spans are useless; swapping
    them out would trade a measured strength for an unmeasured one.
    """
    p = path(n, seed, kind)
    if p.exists():
        return json.loads(p.read_text())
    stores = stores or sorted({mix.load_source(s)["docstore"] for s in mix.available_sources()})
    rng = np.random.default_rng(seed)
    per = n // len(stores) + 1
    long_per = per // 2 if kind == "mixed" else 0
    out = []
    for st in stores:
        _, texts = mix.load_store(st)
        # 4x rather than 2x oversampling for the long half: _long_span rejects any document whose
        # tail from the drawn start cannot reach 64 words, which is a much higher reject rate than
        # _span's four-word floor.
        over = 4 if kind == "mixed" else 2
        idx = rng.choice(len(texts), size=min(per * over, len(texts)), replace=False)
        got = got_long = 0
        for i in idx:
            want_long = got_long < long_per
            s = _long_span(texts[int(i)], rng) if want_long else _span(texts[int(i)])
            if s is None and want_long:
                s = _span(texts[int(i)])          # fall back to the short half, never to nothing
            elif s is None:
                continue
            if s is None:
                continue
            out.append(s)
            got += 1
            got_long += bool(want_long and len(s.split()) >= LONG_MIN_WORDS)
            if got >= per:
                break
        del texts
        print(f"  pseudoq[{kind}] {st}: {got:,} ({got_long:,} long)", flush=True)
    out = out[:n]
    p.write_text(json.dumps(out))
    return out


def targets(texts, tag):
    return np.asarray(encode_cached(f"pseudoq-{tag}-{len(texts)}", texts, prefix=QUERY_PREFIX,
                                    dtype=torch.float16, verbose=True), dtype=np.float32)


def build_decontaminated(n, seed=SEED, kind="short"):
    """The R1-filtered pool. Pseudo-queries are spans of TRAIN documents and are TRAIN queries
    under the mandate's "all partitions" wording, so they go through rule R1 like every other
    training query. Raises rather than falling back to the unfiltered pool.

    A long span carries more word-8-grams than a short one and therefore matches the R1 index more
    often, so the mixed pool is expected to lose a larger share than the short pool did; the counts
    are logged in m7/LEDGER.md alongside every other source's.
    """
    qs = build(n, seed=seed, kind=kind)
    p = path(n, seed, kind)
    kept = OUT / f"kept-{p.stem}.json"
    if not kept.exists():
        raise RuntimeError(f"{kept} missing: run decontam_querytext.py after building the "
                           f"pseudo-query pool. Pseudo-queries are never used unfiltered.")
    idx = json.loads(kept.read_text())
    return [qs[i] for i in idx]


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
    qs = build(n)
    print(f"{len(qs):,} pseudo-queries; sample:")
    for q in qs[:5]:
        print("   ", q[:110])
