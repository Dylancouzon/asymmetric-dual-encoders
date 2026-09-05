"""M10.0-d — the COV fingerprint screen (§Surfaces admission requirement).

Builds an `Inverted` index over each candidate COV component's queries and documents, then
streams the SIX's documents and the protected query index against it and counts which COV items
match. That is M7's direction of travel (candidate-side index, protected-side stream), so the
thresholds and the hash functions are exactly the ones M7 and M9 screened training text with.

**What this does NOT do, deliberately.** §Surfaces asks for a screen "against the six and the
reserved four". Reserved-set DOCUMENT fingerprints do not exist, and creating them would open the
reserved corpora — the same reasoning that ruled FineWeb out of M10 on 2026-09-01
(`m9/LEDGER.md` §1.3, CLAUDE.md key decisions). So the reserved four are covered here on the QUERY
side only, through `decontam.protected_query_index()`, which already includes `untouched-final`.
The gap is recorded, not papered over: `m10/LEDGER.md` §3 W4.
"""
import json, os, sys, time
from pathlib import Path

os.environ.pop("HF_HUB_OFFLINE", None)
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "m7src"))
OUT = REPO / "work" / "m10cov"

import numpy as np
import decontam
from cov_admit import COMPONENTS, BRIGHT_SLICES

MIN_SHARE = 8          # >= 8/32 sketch agreement, the registered near-match threshold


def load_component(name, repo, rev):
    """-> (query_texts, doc_texts). BRIGHT is six slices under one family id."""
    from datasets import load_dataset
    if name == "LEDGER":
        import cov_ledger
        qs, _qrels, _ids, texts, _rep = cov_ledger.load(verbose=False)
        return qs, texts
    if name == "BRIGHT":
        qs, ds = [], []
        for sl in BRIGHT_SLICES:
            e = load_dataset(repo, "examples", revision=rev, split=sl)
            d = load_dataset(repo, "documents", revision=rev, split=sl)
            qs += [r for r in e["query"]]
            ds += [r for r in d["content"]]
        return qs, ds
    q = load_dataset(repo, "queries", revision=rev, split="queries")
    c = load_dataset(repo, "corpus", revision=rev, split="corpus")
    qt = [r for r in q["text"]]
    dt = [(f"{t} {x}".strip() if t else x)
          for t, x in zip(c["title"], c["text"])] if "title" in c.column_names else list(c["text"])
    return qt, dt


def screen(texts, label):
    """-> dict of hit counts for one list of candidate texts."""
    t0 = time.time()
    per = [decontam.all_grams(t) for t in texts]
    inv = decontam.Inverted(per, [decontam.exact_u64(t) for t in texts])
    hit_doc_ex, hit_doc_near = set(), set()
    n_streamed = 0
    for d in decontam.stream_six_docs():
        n_streamed += 1
        ex, near = inv.match(d, MIN_SHARE)
        hit_doc_ex.update(ex.tolist()); hit_doc_near.update(near.tolist())
    pq = decontam.protected_queries()
    hit_q_ex, hit_q_near = set(), set()
    n_pq = 0
    for grp, qs in pq.items():
        for q in qs:
            n_pq += 1
            ex, near = inv.match(q, MIN_SHARE)
            hit_q_ex.update(ex.tolist()); hit_q_near.update(near.tolist())
    r = dict(n_candidates=len(texts), index_mb=round(inv.nbytes / 2**20, 1),
             six_docs_streamed=n_streamed, protected_queries_streamed=n_pq,
             hits_vs_six_docs_exact=len(hit_doc_ex), hits_vs_six_docs_near=len(hit_doc_near),
             hits_vs_protected_queries_exact=len(hit_q_ex),
             hits_vs_protected_queries_near=len(hit_q_near),
             seconds=round(time.time() - t0, 1))
    r["total_flagged"] = len(hit_doc_ex | hit_doc_near | hit_q_ex | hit_q_near)
    r["flagged_frac"] = round(r["total_flagged"] / max(len(texts), 1), 5)
    print(f"  {label:28s} n={len(texts):7d}  six-doc ex/near {r['hits_vs_six_docs_exact']}/"
          f"{r['hits_vs_six_docs_near']}  prot-q ex/near {r['hits_vs_protected_queries_exact']}/"
          f"{r['hits_vs_protected_queries_near']}  flagged {r['flagged_frac']:.4%}  "
          f"{r['seconds']:.0f}s", flush=True)
    return r


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    for family, comps in COMPONENTS.items():
        for name, repo, rev in comps:
            print(f"\n=== {family} / {name}", flush=True)
            qs, ds = load_component(name, repo, rev)
            res[name] = dict(family=family, repo=repo, revision=rev,
                             n_queries=len(qs), n_docs=len(ds),
                             queries=screen(qs, "queries"),
                             documents=screen(ds, "documents"))
            (OUT / "screen.json").write_text(json.dumps(res, indent=1))
    (OUT / "screen.json").write_text(json.dumps(res, indent=1))
    print("\nwrote", OUT / "screen.json")


if __name__ == "__main__":
    main()
