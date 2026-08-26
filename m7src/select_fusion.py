"""Select and freeze the fusion parameter on dev, before any test access.

One family (RRF or convex, picked here), one parameter, no per-dataset weights, no routing.
BM25 runs at `fusion.DEPTH` -- the same depth the final run applies -- because both fusion
families are depth-sensitive and a parameter chosen at one depth is not valid at another.

Fusion selection runs on the TEXT-BACKED dev components only: the held-out slices' corpora are
pool row indices and carry no document text, so BM25 has no run on them. Same restriction the
gate's BM25 comparison carries, and disclosed the same way.

Output: work/runs/<run_id>.fusion.json, to be copied into m7/FREEZE.json by freeze.write().
"""
import json
import sys

import numpy as np

import dev_eval
import fusion
from _paths import REPO, WORK
from evalkit import topk_ids_scores
from table import Preproc, load_table, read_meta

CACHE = WORK / "fusionruns"
CACHE.mkdir(parents=True, exist_ok=True)


def bm25_run_cached(comp):
    """BM25 at fusion.DEPTH. Cached: indexing HotpotQA's 5.23M documents is the single most
    expensive repeated step on this box."""
    p = CACHE / f"bm25-{comp}-d{fusion.DEPTH}.npz"
    doc_ids, doc_texts, q_ids, q_texts, qrels, _ = dev_eval.doc_vecs(comp)
    if p.exists():
        z = np.load(p, allow_pickle=False)
        ids, sc = z["ids"], z["scores"]
        return {q_ids[i]: {doc_ids[int(d)]: float(s) for d, s in zip(ids[i], sc[i]) if s > 0
                           and doc_ids[int(d)] != q_ids[i]} for i in range(len(q_ids))}
    import Stemmer
    import bm25s
    st = Stemmer.Stemmer("english")
    r = bm25s.BM25(method="lucene", k1=1.2, b=0.75)
    r.index(bm25s.tokenize(doc_texts, stopwords="en", stemmer=st, show_progress=False),
            show_progress=False)
    ids, sc = r.retrieve(bm25s.tokenize(q_texts, stopwords="en", stemmer=st, show_progress=False),
                         k=min(fusion.DEPTH, len(doc_ids)), show_progress=False)
    np.savez_compressed(p, ids=ids.astype(np.int32), scores=sc.astype(np.float32))
    # The `s > 0` filter MUST match the cache-read path above. It did not: the fresh build let
    # zero-score padding rows into the run, which shifts every min-max normalisation's `lo`, so
    # selection and application scored two different functions. Measured impact was <5e-4, but a
    # frozen fusion parameter is only frozen if the function it was frozen against is fixed.
    return {q_ids[i]: {doc_ids[int(d)]: float(s) for d, s in zip(ids[i], sc[i]) if s > 0
                       and doc_ids[int(d)] != q_ids[i]} for i in range(len(q_ids))}


def dense_run(comp, model, pre):
    doc_ids, _, q_ids, q_texts, _, dv = dev_eval.doc_vecs(comp)
    return topk_ids_scores(model.encode(q_texts, pre), dv, doc_ids, k=fusion.DEPTH,
                           chunk=dev_eval.CHUNK.get(comp, 250_000), qids=q_ids)


def main(run_id):
    npz = WORK / "runs" / f"{run_id}.npz"
    meta = read_meta(npz)
    pre = Preproc(**meta["preproc"])
    comps = [c for c in dev_eval.dev_components() if not c.startswith("heldout-")]
    print(f"fusion selection on the text-backed dev components: {comps} at depth {fusion.DEPTH}")

    # the released artifact is the int8 table, so fusion is fitted against int8, not fp16
    model = load_table(npz, variant="int8")
    dense, bm25, qrels = {}, {}, {}
    for c in comps:
        dense[c] = dense_run(c, model, pre)
        bm25[c] = bm25_run_cached(c)
        qrels[c] = dev_eval.doc_vecs(c)[4]
        print(f"  runs built for {c}", flush=True)
    del model

    from evalkit import per_query_ndcg
    for label, runs in (("int8-table alone", dense), ("bm25 alone", bm25)):
        m = float(np.mean([np.mean(list(per_query_ndcg(runs[c], qrels[c]).values())) for c in comps]))
        print(f"  {label}: dev macro(text-backed) {m:.4f}")

    spec, _ = fusion.select_on_dev(dense, bm25, qrels)
    spec["depth"] = fusion.DEPTH
    spec["components"] = comps
    spec["fitted_against"] = "int8 table (the released artifact)"
    (WORK / "runs" / f"{run_id}.fusion.json").write_text(json.dumps(spec, indent=1))
    (REPO / "results" / f"m7_fusion_{run_id}.json").write_text(json.dumps(spec, indent=1))
    print(f"\nfrozen fusion spec -> work/runs/{run_id}.fusion.json")
    return spec


if __name__ == "__main__":
    main(sys.argv[1])
