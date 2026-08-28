"""Codex B5 regression: the BM25 run that gets fused must be ONE function everywhere.

Selection dropped bm25s's zero-score padding rows and the final run kept them; convex fusion
min-max normalises over what is in the run, so the two paths scored two different functions and
the parameter frozen on dev would not have been the function applied at test. fusion.bm25_run /
fusion._to_run is now the single builder; this file makes a re-fork loud.

Run after touching fusion.py, select_fusion.py, or final_run.py's scoring path.
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

import fusion

FAILS = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {name}" + (f" -- {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


def main():
    # Fixture: 30 docs, 5 queries; corpus < DEPTH so bm25s pads every query with zero-score
    # rows (the exact divergence trigger); q4's text also exists as a doc with id == qid
    # (the ArguAna self-hit shape).
    rng = np.random.default_rng(0)
    words = ["retrieval", "lookup", "table", "quantum", "market", "protein", "tax", "argue",
             "covid", "vector", "sparse", "dense", "edge", "query", "token"]
    doc_ids = [f"d{i}" for i in range(29)] + ["q4"]
    doc_texts = [" ".join(rng.choice(words, size=8)) for _ in range(29)]
    doc_texts.append("argue tax market lookup")                    # == q4's text, id == qid
    q_ids = [f"q{i}" for i in range(5)]
    q_texts = ["retrieval lookup table", "quantum protein", "covid vector sparse",
               "zzzz yyyy xxxx",                                    # matches nothing: all padding
               "argue tax market lookup"]

    with tempfile.TemporaryDirectory() as td:
        cp = Path(td) / "bm25-fixture.npz"
        fresh = fusion.bm25_run(doc_ids, doc_texts, q_ids, q_texts, cache_path=cp)
        check("cache file written", cp.exists())
        cached = fusion.bm25_run(doc_ids, doc_texts, q_ids, q_texts, cache_path=cp)
        check("cached == fresh, byte-identical",
              json.dumps(fresh, sort_keys=True) == json.dumps(cached, sort_keys=True))
        nocache = fusion.bm25_run(doc_ids, doc_texts, q_ids, q_texts, cache_path=None)
        check("uncached path == cached path, byte-identical",
              json.dumps(fresh, sort_keys=True) == json.dumps(nocache, sort_keys=True))

        # ---- the cache is keyed on CONTENT, not on the pathname -------------------------
        # Codex one-shot-path review 2026-08-28, MAJOR 2: the cached arrays are integer doc
        # POSITIONS; `_to_run` re-attaches whatever id lists the caller passes, so a corpus of
        # the same shape used to be accepted silently and a parameter selected on one lexical
        # run applied to another.
        check("the cache stores its key", "key" in np.load(cp, allow_pickle=False).files)

        other_texts = list(doc_texts)
        other_texts[0] = "tax tax tax argue argue market"      # same shape, different corpus
        other = fusion.bm25_run(doc_ids, other_texts, q_ids, q_texts, cache_path=cp)
        check("a different corpus of the same shape does NOT reuse the cache",
              json.dumps(other, sort_keys=True) != json.dumps(fresh, sort_keys=True))
        check("the rebuilt run equals the uncached run on the new corpus",
              json.dumps(other, sort_keys=True) ==
              json.dumps(fusion.bm25_run(doc_ids, other_texts, q_ids, q_texts), sort_keys=True))

        for label, kw in (("doc ids", {"doc_ids": [f"x{i}" for i in range(30)]}),
                          ("query texts", {"q_texts": q_texts[:-1] + ["protein market"]}),
                          ("query ids", {"q_ids": [f"z{i}" for i in range(5)]})):
            args = {"doc_ids": doc_ids, "doc_texts": doc_texts, "q_ids": q_ids, "q_texts": q_texts}
            args.update(kw)
            k = fusion.cache_key(args["doc_ids"], args["doc_texts"], args["q_ids"], args["q_texts"])
            check(f"changing the {label} changes the cache key",
                  k != fusion.cache_key(doc_ids, doc_texts, q_ids, q_texts))

        # a keyless cache -- everything written before 2026-08-28 -- must be rebuilt, not trusted
        legacy = Path(td) / "bm25-legacy.npz"
        z = np.load(cp, allow_pickle=False)
        np.savez_compressed(legacy, ids=z["ids"], scores=z["scores"])
        _, why = fusion._read_cache(legacy, fusion.cache_key(doc_ids, other_texts, q_ids, q_texts))
        check("a keyless legacy cache is rejected", why is not None and "key" in why, str(why))
        relegacy = fusion.bm25_run(doc_ids, doc_texts, q_ids, q_texts, cache_path=legacy)
        check("a keyless legacy cache is rebuilt correctly",
              json.dumps(relegacy, sort_keys=True) == json.dumps(fresh, sort_keys=True))
        check("the rebuilt legacy cache now carries a key",
              "key" in np.load(legacy, allow_pickle=False).files)

    # ---- an unknown fusion family must be fatal, never a silent convex ------------------
    try:
        fusion.apply_frozen({"family": "convexx", "param": 0.5}, {}, {})
        check("apply_frozen refuses an unknown family", False, "it returned instead of raising")
    except SystemExit as e:
        check("apply_frozen refuses an unknown family", "convexx" in str(e))
    check("every family in FAMILIES is applicable",
          all(fusion.apply_frozen({"family": f, "param": 0.5 if f != "rrf" else 60},
                                  {"q": {"d": 1.0}}, {"q": {"d": 2.0}}) for f in fusion.FAMILIES))

    check("no zero/negative score survives (padding dropped)",
          all(s > 0 for docs in fresh.values() for s in docs.values()))
    check("no-match query yields an empty run entry, not padding", fresh["q3"] == {})
    check("self-hit dropped (doc_id == qid)", "q4" not in fresh["q4"])

    got = {qid: len(d) for qid, d in fresh.items()}
    check("padded queries are shorter than the corpus", all(v < len(doc_ids) for v in got.values()),
          str(got))

    # convex's lo must be the min POSITIVE bm25 score once padding is gone
    dense = {qid: {d: float(rng.uniform(0.2, 0.9)) for d in list(fresh[qid]) + ["d0", "d1"]}
             for qid in q_ids}
    fused = fusion.convex([dense, fresh], w=0.5)
    q0 = fresh["q0"]
    lo, hi = min(q0.values()), max(q0.values())
    top_bm = max(q0, key=q0.get)
    expect = 0.5 * (q0[top_bm] - lo) / (hi - lo + 1e-9)
    check("convex normalises over positive scores only",
          abs((fused["q0"][top_bm] - 0.5 * (dense["q0"][top_bm] - min(dense["q0"].values()))
               / (max(dense["q0"].values()) - min(dense["q0"].values()) + 1e-9)) - expect) < 1e-12)

    # the re-fork guard: both call sites must go through fusion.bm25_run and define no local copy
    here = Path(__file__).parent
    fr = (here / "final_run.py").read_text()
    sf = (here / "select_fusion.py").read_text()
    check("final_run.py defines no local bm25 builder", "def bm25_run(" not in fr)
    check("final_run.py calls fusion.bm25_run", "fusion.bm25_run(" in fr)
    check("select_fusion.py routes through fusion.bm25_run", "fusion.bm25_run(" in sf)

    print(f"\n{'PASS' if not FAILS else 'FAIL: ' + ', '.join(FAILS)}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
