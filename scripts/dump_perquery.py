"""Freeze per-query nDCG@10 vectors for the M7 tier comparators (Opus review B2, 2026-08-25).

Writes results/perquery.json so the M7 session can pair-bootstrap against the exact
published comparator runs without rebuilding LightRetriever/OpenSearch on the new machine.
"""
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
os.environ["BENCH_DATASETS"] = "scifact,nfcorpus,fiqa,arguana,scidocs,trec-covid"
sys.path.insert(0, str(REPO / "bench"))

from core import DATASETS, load_beir  # noqa: E402
from significance import SYSTEMS, per_query_ndcg  # noqa: E402

KEEP = [
    "opensearch-doc-v3-gte",  # Tier 1 comparator
    "lr-dense-pertask",       # Tier 2 comparator (LR strongest dense config, fp16 per-task)
    "lr-dense-websearch",     # like-for-like single-table LR row
    "bm25",                   # Tier 3 comparator + dense-only headline condition
    "potion-retrieval-32M",   # statics reference (Tier 4 line, Stage-0 dev analog)
    "bge-small-en-v1.5",      # harness-validation reference + M8 release comparator
    "leaf-ir-asym",           # M8 reference: MongoDB's own distilled query tower
    "mdbr-leaf-ir",           # M8 reference: LEAF student used symmetrically
    "arctic-embed-m-v1.5",    # M8 reference: LEAF's teacher
]

out = {
    "_note": "per-query nDCG@10, exact brute-force search, frozen 2026-08-25 from the M4 artifact caches. "
    "M7 tier decisions pair against these vectors; qids sorted lexicographically per dataset.",
    "datasets": {},
}
for ds in DATASETS:
    *_, qrels = load_beir(ds)
    per, qids = {}, None
    for name in KEEP:
        pq = per_query_ndcg(SYSTEMS[name](ds, qrels), qrels)
        if qids is None:
            qids = sorted(pq)
        assert sorted(pq) == qids, f"{name}/{ds}: query set mismatch"
        per[name] = [round(pq[q], 6) for q in qids]
    out["datasets"][ds] = {"qids": qids, "systems": per}
    print(ds, {k: round(sum(v) / len(v), 4) for k, v in per.items()}, flush=True)

(REPO / "results" / "perquery.json").write_text(json.dumps(out))
print("wrote results/perquery.json")

# self-check: macro averages must reproduce FINAL_MATRIX.md rows to 4 decimals
expect = {"opensearch-doc-v3-gte": 0.4868, "lr-dense-pertask": 0.4583, "lr-dense-websearch": 0.4320,
          "bm25": 0.4174, "potion-retrieval-32M": 0.3601, "bge-small-en-v1.5": 0.5042,
          "leaf-ir-asym": 0.5155, "mdbr-leaf-ir": 0.5123, "arctic-embed-m-v1.5": 0.5264}
for name, want in expect.items():
    got = sum(sum(out["datasets"][ds]["systems"][name]) / len(out["datasets"][ds]["qids"]) for ds in DATASETS) / len(DATASETS)
    status = "OK" if abs(got - want) < 5e-4 else "MISMATCH"
    print(f"{status} {name}: avg-6 {got:.4f} vs matrix {want:.4f}", flush=True)
