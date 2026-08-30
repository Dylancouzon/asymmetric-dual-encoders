"""Dev evaluation for a nano student against a given teacher's document space.

Two locked surfaces (m9/LEDGER.md §4.1): DEV-6 (all six pinned M7 components, equal weight) for
student / prompt / mix / batch / seed-floor, and SCREEN-3 (family-weighted) for the teacher, which
is the only surface a challenger teacher can afford. Every stella arm is scored on BOTH so no
contrast crosses surfaces.

For stella this reuses `m7src/dev_eval.doc_vecs`, which already knows how to serve the two
pool-backed held-out slices; a challenger goes through `m9src/teacher9`, and only on SCREEN-3.
"""
import json

import numpy as np

import m9base
from m9base import SCREEN_DEV, DEV_FULL, RESULTS

INCUMBENT = "stella-400M-v5"
_DOCS = {}


def surfaces_for(teacher_key):
    return ("SCREEN3", "DEV6") if teacher_key == INCUMBENT else ("SCREEN3",)


def components(surface):
    return list(DEV_FULL) if surface == "DEV6" else list(SCREEN_DEV)


def doc_vecs(comp, teacher_key):
    """-> (doc_ids, q_ids, q_texts, qrels, doc_vecs fp16) in `teacher_key`'s space."""
    ck = (comp, teacher_key)
    if ck in _DOCS:
        return _DOCS[ck]
    if teacher_key == INCUMBENT:
        import dev_eval
        doc_ids, _dt, q_ids, q_texts, qrels, dv = dev_eval.doc_vecs(comp)
    else:
        assert comp in SCREEN_DEV, (
            f"{comp} is not on SCREEN-3; a challenger teacher has no document vectors for it "
            f"and M9.0 forbids proxying one (LEDGER §0)")
        import devsuite
        import teacher9
        doc_ids, doc_texts, q_ids, q_texts, qrels = devsuite.load(comp)
        dv = teacher9.encode_cached(teacher_key, f"dev-{comp}-docs", doc_texts, "doc",
                                    batch_size=32, max_length=512)
    assert dv.shape[0] == len(doc_ids)
    _DOCS[ck] = (doc_ids, q_ids, q_texts, qrels, dv)
    return _DOCS[ck]


CHUNK = {"hotpotqa": 250_000, "heldout-train": 250_000, "heldout-longq": 250_000}


def eval_student(student, teacher_key, comps=None):
    """-> {component: {qid: nDCG@10}}. `student.encode_queries(texts)` returns (n, dim) fp32
    unit-norm rows aligned to `texts`."""
    import evalkit

    comps = comps or sorted(set().union(*[set(components(s))
                                          for s in surfaces_for(teacher_key)]))
    out = {}
    # heldout-longq's 55 queries are a verified SUBSET of heldout-train's 7,325 over the identical
    # 6.17M-row pool corpus, so scoring it separately would read 12.6 GB twice per checkpoint.
    # Score it by subsetting instead, and assert the subset relation rather than assuming it.
    pair = {"heldout-train", "heldout-longq"}
    subset_longq = pair <= set(comps)
    if subset_longq:
        comps = [c for c in comps if c != "heldout-longq"]
    for comp in comps:
        doc_ids, q_ids, q_texts, qrels, dv = doc_vecs(comp, teacher_key)
        qv = student.encode_queries(list(q_texts))
        assert qv.shape == (len(q_ids), dv.shape[1]), f"{comp}: {qv.shape} vs {dv.shape}"
        run = evalkit.topk_ids_scores(qv, dv, doc_ids, k=100,
                                      chunk=CHUNK.get(comp, 200_000), qids=q_ids)
        out[comp] = evalkit.per_query_ndcg(run, qrels)
    if subset_longq:
        tr_ids, _tq, _tt, _tr, tr_dv = doc_vecs("heldout-train", teacher_key)
        lq_di, lq_ids, _qt, lq_rels, lq_dv = doc_vecs("heldout-longq", teacher_key)
        # Subsetting is exact only if both components rank against the identical corpus under the
        # identical self-hit and tie-break path. Assert it; do not assume it (Codex pass 2,
        # MAJOR-11). Same object identity for the memmap is the strongest available check.
        import hashlib

        def _h(seq):
            hh = hashlib.sha256()
            for x in seq:
                hh.update(str(x).encode()); hh.update(b"\x00")
            return hh.hexdigest()

        assert _h(lq_di) == _h(tr_ids), \
            "heldout-longq and heldout-train no longer share a corpus -- score longq separately"
        assert lq_dv.shape == tr_dv.shape, "shared-corpus vector shapes differ"
        assert lq_dv is tr_dv or bytes(np.asarray(lq_dv[:64])) == bytes(np.asarray(tr_dv[:64])), \
            "shared-corpus document vectors differ"
        lq = [str(q) for q in lq_ids]
        assert set(lq) <= set(out["heldout-train"]), (
            "heldout-longq is no longer a subset of heldout-train -- score it separately")
        assert all(lq_rels[q] == _train_rels(teacher_key)[q] for q in lq_rels), (
            "heldout-longq qrels differ from heldout-train's for the shared queries")
        # a query dropped by heldout-train's own scoring cannot be recovered by subsetting
        assert set(lq) <= set(out["heldout-train"])
        out["heldout-longq"] = {q: out["heldout-train"][q] for q in lq}
    return out


_TR = {}


def _train_rels(teacher_key):
    if teacher_key not in _TR:
        _TR[teacher_key] = doc_vecs("heldout-train", teacher_key)[3]
    return _TR[teacher_key]


def macros(per_component, teacher_key):
    import screen_stats
    out = {}
    for s in surfaces_for(teacher_key):
        if all(c in per_component for c in components(s)):
            m, means = screen_stats.macro(per_component, s)
            out[s] = {"macro": round(m, 6), "means": {k: round(v, 6) for k, v in means.items()}}
    return out


def teacher_symmetric(teacher_key, comps=None):
    """The ceiling row: the teacher encoding its own queries into its own document space."""
    class _Sym:
        def encode_queries(self, texts):
            if teacher_key == INCUMBENT:
                import torch
                import teacher
                v = teacher.encode(texts, prefix=teacher.QUERY_PREFIX, max_length=512,
                                   dtype=torch.float32)
                return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)
            import teacher9
            return teacher9.encode(teacher_key, texts, "query")

    return eval_student(_Sym(), teacher_key, comps)


def cached_symmetric(teacher_key):
    p = RESULTS / f"m9_dev_symmetric_{teacher_key}.json"
    if p.exists():
        return json.loads(p.read_text())["per_component"]
    per = teacher_symmetric(teacher_key)
    p.write_text(json.dumps({"teacher": teacher_key, "macros": macros(per, teacher_key),
                             "per_component": per}, indent=1))
    return per
