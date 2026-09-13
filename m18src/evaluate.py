"""Fresh M18 exact dense, BM25, Qdrant DBSF and paired-bootstrap evaluation."""
from __future__ import annotations

import math
import re
import sys
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from common import REPO, atomic_write_bytes, sha_json

DOC_BLOCK = 65536
LEX_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_./:+#-]*")


def _ordered(docs):
    return sorted(docs.items(), key=lambda kv: (-kv[1], str(kv[0])))


def search(query_vecs, doc_vecs, k=100, doc_ids=None, query_ids=None,
           query_block=256, doc_block=DOC_BLOCK):
    """Exact normalized inner product with deterministic score/id tie breaking."""
    q, d = np.asarray(query_vecs, np.float32), np.asarray(doc_vecs, np.float32)
    dids = list(doc_ids or range(len(d)))
    qids = list(query_ids or range(len(q)))
    if len(set(dids)) != len(dids) or len(set(qids)) != len(qids):
        raise ValueError("query and document ids must be unique")
    kk = min(int(k), len(dids))
    out = {}
    for qlo in range(0, len(q), query_block):
        qb = q[qlo:qlo + query_block]
        candidates = [[] for _ in range(len(qb))]
        for dlo in range(0, len(d), doc_block):
            scores = qb @ d[dlo:dlo + doc_block].T
            m = min(kk, scores.shape[1])
            if not m:
                continue
            top = np.argpartition(-scores, m - 1, axis=1)[:, :m]
            for i in range(len(qb)):
                cut = float(scores[i, top[i]].min())
                tied = np.flatnonzero(scores[i] >= cut)
                block_rows = sorted(((float(scores[i, j]), dids[dlo + int(j)]) for j in tied),
                                    key=lambda x: (-x[0], str(x[1])))[:m]
                candidates[i].extend(block_rows)
        for i, rows in enumerate(candidates):
            rows.sort(key=lambda x: (-x[0], str(x[1])))
            out[qids[qlo + i]] = {doc: score for score, doc in rows[:kk]}
    return out


def lex_tokens(text):
    return [m.group(0).lower() for m in LEX_RE.finditer(str(text))]


class BM25Index:
    def __init__(self, doc_ids, documents, k1=1.2, b=0.75):
        import Stemmer
        import bm25s
        self.doc_ids = list(doc_ids)
        self.k1, self.b = float(k1), float(b)
        if len(set(self.doc_ids)) != len(self.doc_ids):
            raise ValueError("duplicate BM25 document ids")
        self._stemmer = Stemmer.Stemmer("english")
        self._bm25 = bm25s.BM25(method="lucene", k1=self.k1, b=self.b)
        tokens = bm25s.tokenize(list(documents), stopwords="en", stemmer=self._stemmer,
                                show_progress=False)
        self._bm25.index(tokens, show_progress=False)

    def query(self, text, k=100):
        import bm25s
        tokens = bm25s.tokenize([str(text)], stopwords="en", stemmer=self._stemmer,
                                show_progress=False)
        ids, scores = self._bm25.retrieve(tokens, k=min(int(k), len(self.doc_ids)),
                                          show_progress=False)
        rows = [(float(s), self.doc_ids[int(i)]) for i, s in zip(ids[0], scores[0]) if s > 0]
        rows.sort(key=lambda x: (-x[0], str(x[1])))
        return {d: s for s, d in rows}

    def run(self, query_ids, texts, k=100):
        return {qid: self.query(text, k) for qid, text in zip(query_ids, texts)}

    def save(self, path):
        from common import admit_write, write_json
        path = Path(admit_write(path))
        if path.exists():
            raise SystemExit(f"M18 BM25 REFUSED: immutable index exists: {path}")
        stage = path.with_name(path.name + f".building-{os.getpid()}")
        stage.mkdir(parents=True)
        try:
            self._bm25.save(stage, show_progress=False)
            write_json(stage / "m18.json", {"doc_ids": self.doc_ids, "k1": self.k1,
                                             "b": self.b, "method": "lucene",
                                             "stopwords": "en", "stemmer": "english"})
            os.replace(stage, path)
        except BaseException:
            shutil.rmtree(stage, ignore_errors=True)
            raise

    @classmethod
    def load(cls, path):
        import Stemmer
        import bm25s
        import json
        from common import admit_read
        path = Path(admit_read(path))
        obj = json.loads(admit_read(path / "m18.json").read_text())
        self = cls.__new__(cls)
        self.doc_ids, self.k1, self.b = obj["doc_ids"], obj["k1"], obj["b"]
        self._stemmer = Stemmer.Stemmer("english")
        self._bm25 = bm25s.BM25.load(path, load_corpus=False, mmap=True,
                                     show_progress=False)
        return self


def dbsf_at(dense_run, bm25_run, prefetch=100):
    if str(REPO / "m12src") not in sys.path:
        sys.path.insert(0, str(REPO / "m12src"))
    import qfusion
    return qfusion.dbsf([qfusion.truncate(dense_run, prefetch),
                         qfusion.truncate(bm25_run, prefetch)])


def metric_per_query(run, qrels, k=10):
    out = {"ndcg@10": {}, "recall@10": {}, "mrr@10": {}}
    for qid in qrels:
        ranked = [d for d, _ in _ordered(run.get(qid, {}))[:k]]
        rel = qrels[qid]
        gains = [float(rel.get(d, 0)) for d in ranked]
        dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
        ideal = sorted((float(x) for x in rel.values()), reverse=True)[:k]
        idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
        relevant = {d for d, v in rel.items() if float(v) > 0}
        out["ndcg@10"][qid] = dcg / idcg if idcg else 0.0
        out["recall@10"][qid] = len(relevant & set(ranked)) / max(1, len(relevant))
        ranks = [i + 1 for i, d in enumerate(ranked) if d in relevant]
        out["mrr@10"][qid] = 1.0 / min(ranks) if ranks else 0.0
    return out


def aggregate(per_query, strata):
    result = {}
    for metric, values in per_query.items():
        by = defaultdict(list)
        for qid, value in values.items():
            by[strata[qid]].append(float(value))
        per = {s: {"mean": float(np.mean(v)), "n": len(v)} for s, v in sorted(by.items())}
        result[metric] = {"query_mean": float(np.mean(list(values.values()))) if values else 0.0,
                          "stratum_macro": float(np.mean([x["mean"] for x in per.values()])) if per else 0.0,
                          "per_stratum": per}
    return result


def paired_bootstrap(candidate, baseline, families, strata, replicates=10000, seed=18018):
    """Stratified family bootstrap of equal-weight five-stratum macro deltas."""
    if set(candidate) != set(baseline):
        raise ValueError("paired metric keys differ")
    keys = sorted(candidate)
    by_sf = defaultdict(lambda: defaultdict(list))
    for q in keys:
        by_sf[strata[q]][families[q]].append(q)
    rng = np.random.default_rng(seed)

    def stat(sample_groups=None):
        means = []
        for stratum in sorted(by_sf):
            groups = list(by_sf[stratum].values()) if sample_groups is None else sample_groups[stratum]
            vals = [candidate[q] - baseline[q] for group in groups for q in group]
            if vals:
                means.append(float(np.mean(vals)))
        return float(np.mean(means)) if means else 0.0

    draws = []
    for _ in range(replicates):
        sampled = {}
        for stratum, fams in by_sf.items():
            groups = list(fams.values())
            sampled[stratum] = [groups[i] for i in rng.integers(0, len(groups), len(groups))]
        draws.append(stat(sampled))
    per_stratum = {}
    for stratum, fams in sorted(by_sf.items()):
        qs = [q for group in fams.values() for q in group]
        delta = np.asarray([candidate[q] - baseline[q] for q in qs])
        # Query bootstrap is sufficient within a stratum because protocol families contribute
        # one evaluation query after near-duplicate union; family count is still disclosed.
        local = [float(np.mean(delta[rng.integers(0, len(delta), len(delta))]))
                 for _ in range(replicates)] if len(delta) else []
        per_stratum[stratum] = {"delta": float(delta.mean()) if len(delta) else 0.0,
                               "ci95": [float(np.percentile(local, 2.5)), float(np.percentile(local, 97.5))]
                               if local else [None, None], "n": len(qs), "families": len(fams)}
    return {"delta": stat(), "ci95": [float(np.percentile(draws, 2.5)),
                                        float(np.percentile(draws, 97.5))],
            "replicates": replicates, "seed": seed, "unit": "query family, stratified",
            "statistic": "equal-weight stratum macro", "per_stratum": per_stratum}


def evaluate_run(run, qrels, strata, families):
    pq = metric_per_query(run, qrels)
    return {"aggregate": aggregate(pq, strata), "per_query": pq}


def compare_models(model_runs, bm25_run, qrels, query_rows, prefetch=100,
                   replicates=10000, seed=18018, baseline="zero_v1"):
    qids = [q["query_id"] for q in query_rows]
    if set(qids) != set(qrels):
        raise ValueError("query/qrels keys differ")
    strata = {q["query_id"]: q["stratum"] for q in query_rows}
    families = {q["query_id"]: q["near_duplicate_family"] or q["family"] for q in query_rows}
    runs = {"bm25": bm25_run}
    runs.update(model_runs)
    for name, dense in model_runs.items():
        runs[name + "+dbsf"] = dbsf_at(dense, bm25_run, prefetch)
    results = {name: evaluate_run(run, qrels, strata, families) for name, run in runs.items()}
    comparisons = {}
    if baseline in model_runs:
        for name in model_runs:
            if name == baseline:
                continue
            comparisons[name] = {}
            for route in ("dense", "fused"):
                a = results[name if route == "dense" else name + "+dbsf"]["per_query"]["ndcg@10"]
                bname = baseline if route == "dense" else baseline + "+dbsf"
                b = results[bname]["per_query"]["ndcg@10"]
                comparisons[name][route] = paired_bootstrap(a, b, families, strata,
                                                             replicates, seed)
    identity = {"qids_sha256": sha_json(qids), "qrels_sha256": sha_json(qrels),
                "prefetch": int(prefetch)}
    return {"_schema": "m18-evaluation-v1", "identity": identity,
            "results": results, "comparisons_vs_zero_v1": comparisons}
