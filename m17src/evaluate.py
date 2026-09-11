"""M17 evaluation: exact dense retrieval, per-domain nDCG@10/Recall@10, paired family
bootstrap, DBSF@100 through `m12src/qfusion.py`, and the held-out alias test.

Quality numbers come from EXACT dense retrieval over a declared corpus. ANN measurements
establish deployment behavior and must never be mixed into these numbers (CLAUDE.md).

Uncertainty is a paired bootstrap over QUERY FAMILIES, not over queries: two views of the same
intent and near-duplicate variants move together, so resampling them independently would
understate the interval.

The alias test reports top-10 overlap and rank correlation between the two views of each
held-out judged pair, plus each view's nDCG@10 where judged. It is descriptive: it appears in no
selection predicate (`registry.training.decision_protocol.panel_and_alias_test_role`).

**Surfaces are gated.** The default surface is `synthetic`. The pinned development suite and the
M17 panel each require their explicit flag AND an executable registry, and neither is to be run
during implementation. Nothing here can reach `results/frozen_eval/untouched-*`, the reserved
qrels caches, `work/m9reserve` or any six-set/LoTTE payload: those paths are refused by name.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common import admit_read, registry, require_executable, write_json

FORBIDDEN = ("frozen_eval/untouched-", "m9reserve", "reserved_qrels", "lotte")


def _check_path(p):
    """Spelling check plus `common.admit_read`, which resolves symlinks before refusing."""
    s = str(p).replace("\\", "/").lower()
    for bad in FORBIDDEN:
        if bad in s:
            raise SystemExit(f"M17 EVAL REFUSED: {p} names a protected surface ({bad!r}). "
                             "M17 has no protected access (registry.protected_access).")
    return admit_read(p)


# ---- exact dense retrieval -----------------------------------------------------------------

DOC_BLOCK = 65536       # 4096 queries x 65536 documents of float32 is 1 GiB per score block


def search(query_vecs, doc_vecs, k=10, doc_ids=None, block=4096, doc_block=DOC_BLOCK,
           query_ids=None):
    """Exact inner product over L2-normalized vectors. Returns a run dict {qkey: {docid: score}}.

    Blocked over queries AND over documents: a 4096 x 5.2M score matrix is 85 GB, so the corpus
    is walked in `doc_block` slices and each slice's top-k is merged. The merge is exact — each
    block keeps its WHOLE cutoff tie (every document scoring at least the block's k-th largest
    score) and then selects by (-score, ascending document id), the same rule the global merge
    and the cache builder use. Taking only `argpartition`'s arbitrary k would discard smaller
    document ids from a tie wider than k and make the answer depend on the block size.
    Document ids must therefore be unique.

    `query_ids` keys the run by the caller's own query ids instead of positional integers; the
    panel's qrels/domains/families are keyed that way (`results/m17_panel_manifest.json`
    reader_contract).
    """
    q = np.asarray(query_vecs, dtype=np.float32)
    d = np.asarray(doc_vecs, dtype=np.float32)
    ids = list(doc_ids) if doc_ids is not None else list(range(d.shape[0]))
    if len(set(ids)) != len(ids):
        raise ValueError("doc_ids contains duplicates; the ascending-id tie rule needs one key "
                         "per document, and a duplicate key would silently drop a document")
    if query_ids is not None and len(query_ids) != q.shape[0]:
        raise ValueError(f"{len(query_ids)} query_ids for {q.shape[0]} query vectors")
    kk = min(k, d.shape[0])
    run = {}
    for lo in range(0, q.shape[0], block):
        qb = q[lo:lo + block]
        cand = [[] for _ in range(qb.shape[0])]
        for dlo in range(0, d.shape[0], doc_block):
            s = qb @ d[dlo:dlo + doc_block].T
            m = min(kk, s.shape[1])
            if m <= 0:
                continue
            top = np.argpartition(-s, m - 1, axis=1)[:, :m]
            for r in range(s.shape[0]):
                cut = float(s[r, top[r]].min())          # the block's k-th largest score
                tied = np.flatnonzero(s[r] >= cut)       # the whole cutoff tie, not a subset
                block = sorted(((float(s[r, j]), ids[dlo + int(j)]) for j in tied),
                               key=lambda t: (-t[0], t[1]))
                cand[r].extend(block[:m])
        for r in range(qb.shape[0]):
            key = query_ids[lo + r] if query_ids is not None else lo + r
            best = sorted(cand[r], key=lambda t: (-t[0], t[1]))[:kk]
            run[key] = {doc: sc for sc, doc in best}
    return run


def ndcg_at_k(run, qrels, k=10):
    """Binary-or-graded nDCG@10 with the standard log2 discount, per query."""
    out = {}
    for qi, docs in run.items():
        rel = qrels.get(qi, {})
        ranked = sorted(docs.items(), key=lambda kv: -kv[1])[:k]
        dcg = sum(float(rel.get(d, 0)) / np.log2(i + 2) for i, (d, _) in enumerate(ranked))
        ideal = sorted((float(v) for v in rel.values()), reverse=True)[:k]
        idcg = sum(v / np.log2(i + 2) for i, v in enumerate(ideal))
        out[qi] = float(dcg / idcg) if idcg > 0 else 0.0
    return out


def recall_at_k(run, qrels, k=10):
    out = {}
    for qi, docs in run.items():
        rel = {d for d, v in qrels.get(qi, {}).items() if float(v) > 0}
        if not rel:
            out[qi] = 0.0
            continue
        ranked = [d for d, _ in sorted(docs.items(), key=lambda kv: -kv[1])[:k]]
        out[qi] = len(rel & set(ranked)) / len(rel)
    return out


def per_domain(values, domains):
    """Mean per domain plus the macro over domains. `domains` maps query key -> domain."""
    by = {}
    for qi, v in values.items():
        by.setdefault(domains.get(qi, "general"), []).append(v)
    means = {d: float(np.mean(v)) for d, v in sorted(by.items())}
    return {"per_domain": means, "macro": float(np.mean(list(means.values()))) if means else 0.0,
            "n_per_domain": {d: len(v) for d, v in sorted(by.items())}}


# ---- paired family bootstrap ------------------------------------------------------------

def paired_family_bootstrap(a, b, families, replicates=10000, confidence=0.95, seed=0,
                            domains=None):
    """Interval on the reported statistic's difference, resampling FAMILIES with replacement.

    With `domains`, the statistic is the EQUAL-WEIGHT DOMAIN MACRO — the same quantity
    `per_domain()` reports — recomputed inside every replicate, not the query mean. The two
    differ badly whenever domains have unequal sizes: 100 queries improving by 1.0 in one domain
    and one query worsening by 1.0 in another is a macro difference of 0 and a query mean of
    0.98. The query-weighted estimate is kept beside it, labelled.

    `domains` also stratifies the resample, so a domain does not vanish from a replicate.
    A wide interval here is a statement about this panel's size, not evidence of equivalence.
    """
    if set(a) != set(b):
        miss, extra = sorted(set(a) - set(b))[:3], sorted(set(b) - set(a))[:3]
        raise ValueError(f"baseline keys do not match the candidate's: missing {miss}, "
                         f"unexpected {extra}. Intersecting would silently drop queries from "
                         "one side and report the paired interval as if they were compared.")
    keys = sorted(a)
    if not keys:
        return {"delta": 0.0, "ci": [None, None], "n_families": 0}
    delta = np.asarray([a[k] - b[k] for k in keys], dtype=np.float64)
    if domains:
        dom_names = sorted({domains.get(k, "general") for k in keys})
        dom_ix = {d: i for i, d in enumerate(dom_names)}
        dom_of = np.asarray([dom_ix[domains.get(k, "general")] for k in keys], dtype=np.int64)

        def statistic(idx):
            sums = np.bincount(dom_of[idx], weights=delta[idx], minlength=len(dom_names))
            cnts = np.bincount(dom_of[idx], minlength=len(dom_names))
            hit = cnts > 0
            return float((sums[hit] / cnts[hit]).mean())
    else:
        def statistic(idx):
            return float(delta[idx].mean())
    fam = np.asarray([families.get(k, k) for k in keys], dtype=object)
    groups = {}
    for i, f in enumerate(fam):
        groups.setdefault(f, []).append(i)
    strata = {}
    for f, idx in groups.items():
        s = (domains or {}).get(keys[idx[0]], "_all") if domains else "_all"
        strata.setdefault(s, []).append(np.asarray(idx))
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=np.float64)
    for r in range(replicates):
        picked = []
        for fams in strata.values():
            sel = rng.integers(0, len(fams), size=len(fams))
            picked.extend(fams[i] for i in sel)
        draws[r] = statistic(np.concatenate(picked))
    lo = float(np.percentile(draws, 100 * (1 - confidence) / 2))
    hi = float(np.percentile(draws, 100 * (1 + confidence) / 2))
    all_idx = np.arange(len(keys))
    return {"delta": statistic(all_idx),
            "statistic": "equal-weight domain macro" if domains else "query mean",
            "delta_query_mean": float(delta.mean()),
            "ci": [lo, hi], "confidence": confidence,
            "replicates": replicates, "n_families": len(groups), "n_queries": len(keys),
            "_note": "paired over query families; excludes training-seed variation"}


# ---- fusion -------------------------------------------------------------------------------

def dbsf_at(dense_run, bm25_run, prefetch=100):
    """Fixed DBSF over the two prefetches, using Qdrant's operator from `m12src/qfusion.py`."""
    import sys
    from common import REPO
    if str(REPO / "m12src") not in sys.path:
        sys.path.insert(0, str(REPO / "m12src"))
    import qfusion
    return qfusion.dbsf([qfusion.truncate(dense_run, prefetch),
                         qfusion.truncate(bm25_run, prefetch)])


# ---- alias test ---------------------------------------------------------------------------

def alias_test(run_a, run_b, k=10):
    """Top-10 overlap and rank correlation between the two views of each held-out pair."""
    if set(run_a) != set(run_b):
        miss, extra = sorted(set(run_a) - set(run_b))[:3], sorted(set(run_b) - set(run_a))[:3]
        raise ValueError(f"the two alias views do not cover the same pairs: missing {miss}, "
                         f"unexpected {extra}. Every declared pair needs one result per view.")
    overlaps, rhos = [], []
    for key in sorted(run_a):
        ra = [d for d, _ in sorted(run_a[key].items(), key=lambda kv: -kv[1])[:k]]
        rb = [d for d, _ in sorted(run_b[key].items(), key=lambda kv: -kv[1])[:k]]
        overlaps.append(len(set(ra) & set(rb)) / max(1, k))
        shared = [d for d in ra if d in rb]
        if len(shared) >= 2:
            x = np.asarray([ra.index(d) for d in shared], dtype=np.float64)
            y = np.asarray([rb.index(d) for d in shared], dtype=np.float64)
            rhos.append(_spearman(x, y))
    return {"n_pairs": len(overlaps),
            "top10_overlap_mean": float(np.mean(overlaps)) if overlaps else 0.0,
            "rank_correlation_mean": float(np.mean(rhos)) if rhos else None,
            "n_pairs_with_rank_correlation": len(rhos),
            "_role": "descriptive; appears in no selection predicate"}


def _spearman(x, y):
    rx, ry = _rankdata(x), _rankdata(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    den = float(np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))
    return float((rx * ry).sum() / den) if den > 0 else 0.0


def _rankdata(a):
    order = np.argsort(a, kind="stable")
    r = np.empty(len(a), dtype=np.float64)
    r[order] = np.arange(1, len(a) + 1)
    return r


# ---- one evaluation ------------------------------------------------------------------------

def evaluate(query_vecs, doc_vecs, qrels, domains, families=None, doc_ids=None, bm25_run=None,
             baseline_ndcg=None, reg=None, k=10, seed=0, query_ids=None):
    """One artifact, one declared corpus. Returns the full descriptive block.

    `query_ids` (the panel's own `query_id` strings) keys the run, so qrels/domains/families
    keyed that way line up. A key-set mismatch raises here rather than silently scoring every
    query as unjudged and every domain as 'general'.
    """
    reg = reg or registry()
    prefetch = int(reg["serving"]["prefetch"])
    if query_ids is not None:
        keys = list(query_ids)
        if len(set(keys)) != len(keys):
            raise ValueError("query_ids contains duplicates; each query needs one key")
        supplied = [("qrels", qrels), ("domains", domains)]
        if families:
            supplied.append(("families", families))
        for name, m in supplied:
            if set(m) != set(keys):
                miss, extra = sorted(set(keys) - set(m))[:3], sorted(set(m) - set(keys))[:3]
                raise ValueError(f"{name} keys do not match query_ids: missing {miss}, "
                                 f"unexpected {extra}")
    run = search(query_vecs, doc_vecs, k=max(k, prefetch), doc_ids=doc_ids, query_ids=query_ids)
    nd = ndcg_at_k(run, qrels, k)
    rc = recall_at_k(run, qrels, k)
    out = {"n_queries": len(run), "n_docs": int(np.asarray(doc_vecs).shape[0]),
           "ndcg@10": per_domain(nd, domains), "recall@10": per_domain(rc, domains),
           "per_query_ndcg@10": nd}
    if bm25_run is not None:
        if set(bm25_run) != set(run):
            miss = sorted(set(run) - set(bm25_run), key=str)[:3]
            extra = sorted(set(bm25_run) - set(run), key=str)[:3]
            raise ValueError(f"bm25_run keys do not match the dense run: missing {miss}, "
                             f"unexpected {extra}. Positional keys against string query ids "
                             "would fuse unrelated queries and report them as fused.")
        fused = dbsf_at(run, bm25_run, prefetch)
        fnd = ndcg_at_k(fused, qrels, k)
        out["fused_dbsf@100"] = {**per_domain(fnd, domains), "prefetch": prefetch}
    if baseline_ndcg is not None:
        prefs = reg["screening_preferences_not_release_bars"]
        out["vs_baseline"] = paired_family_bootstrap(
            nd, baseline_ndcg, families or {}, replicates=int(prefs["audit_bootstrap_replicates"]),
            confidence=float(prefs["audit_interval_confidence"]), seed=seed, domains=domains)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fixtures", required=True,
                    help="directory of synthetic .npy/.json fixtures to evaluate")
    ap.add_argument("--surface", choices=("synthetic", "dev-suite", "panel"), default="synthetic")
    ap.add_argument("--allow-dev-suite", action="store_true")
    ap.add_argument("--allow-panel", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    reg = registry()
    if args.surface != "synthetic":
        flag = {"dev-suite": args.allow_dev_suite, "panel": args.allow_panel}[args.surface]
        if not flag:
            raise SystemExit(f"M17 EVAL REFUSED: surface {args.surface!r} needs its explicit "
                             f"--allow-{args.surface} flag; it is a registered read, not a "
                             "development convenience.")
        require_executable(reg, rehearsal=False, what=f"a {args.surface} read")
        raise SystemExit(f"M17 EVAL REFUSED: the {args.surface} reader is not wired up in this "
                         "pre-clock implementation. It is registered work for the execution "
                         "session (m17/STATUS.md steps 5-6), after the lock.")
    d = _check_path(Path(args.fixtures))
    fx = json.loads((d / "fixtures.json").read_text())
    rep = evaluate(np.load(d / "query_vecs.npy"), np.load(d / "doc_vecs.npy"),
                   {int(k): v for k, v in fx["qrels"].items()},
                   {int(k): v for k, v in fx["domains"].items()},
                   {int(k): v for k, v in fx.get("families", {}).items()},
                   doc_ids=fx.get("doc_ids"), reg=reg)
    rep.pop("per_query_ndcg@10", None)
    print(json.dumps(rep, indent=1, sort_keys=True))
    if args.out:
        write_json(args.out, rep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
