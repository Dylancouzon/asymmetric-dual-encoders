"""Qdrant Edge two-collection prototype of the LightRetriever edge architecture.

Shard 1 (token_table): 151,666 points, id = token id, vector = lookup-table row (websearch table).
Shard 2 (docs): FiQA corpus, LightRetriever dense doc vectors, HNSW.
Query path: tokenize -> retrieve token vectors by id -> mean -> normalize -> query docs shard.
No transformer anywhere at query time.

  python bench/edge_prototype.py build     # create + fill + optimize both shards
  python bench/edge_prototype.py measure   # cold start, latency breakdown, nDCG vs brute force
"""
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from core import REPO, load_beir, load_vecs, score_run, topk_run

EDGE_DIR = REPO / "edge_data"


def point_id(doc_id):
    """Numeric ids pass through; others map to a stable uuid5 (original kept in payload)."""
    import uuid

    return int(doc_id) if doc_id.isdigit() else str(uuid.uuid5(uuid.NAMESPACE_URL, doc_id))
TABLE_SHARD = str(EDGE_DIR / "token_table")
DOCS_SHARD = str(EDGE_DIR / "docs_fiqa")
LR = REPO / "artifacts/lightretriever-qwen2.5-1.5b"
DATASET = "fiqa"


def build():
    from qdrant_edge import Distance, EdgeConfig, EdgeShard, EdgeVectorParams, Point, UpdateOperation

    table = np.load(LR / "table_websearch.npy").astype(np.float32)
    Path(TABLE_SHARD).mkdir(parents=True, exist_ok=True)
    shard = EdgeShard.create(TABLE_SHARD, EdgeConfig(vectors={"tok": EdgeVectorParams(size=table.shape[1], distance=Distance.Dot)}))
    t0 = time.time()
    for start in range(0, table.shape[0], 4096):
        pts = [Point(id=i, vector={"tok": table[i].tolist()}, payload=None) for i in range(start, min(start + 4096, table.shape[0]))]
        shard.update(UpdateOperation.upsert_points(pts))
    shard.optimize()
    shard.flush()
    shard.close()
    print(f"token_table: {table.shape[0]} points in {time.time()-t0:.0f}s", flush=True)

    doc_ids, doc_vecs = load_vecs("lightretriever-qwen2.5-1.5b", DATASET, "doc")
    doc_vecs = doc_vecs.astype(np.float32)
    Path(DOCS_SHARD).mkdir(parents=True, exist_ok=True)
    shard = EdgeShard.create(DOCS_SHARD, EdgeConfig(vectors={"dense": EdgeVectorParams(size=doc_vecs.shape[1], distance=Distance.Dot)}))
    t0 = time.time()
    for start in range(0, len(doc_ids), 4096):
        pts = [
            Point(id=point_id(doc_ids[i]), vector={"dense": doc_vecs[i].tolist()}, payload={"doc_id": doc_ids[i]})
            for i in range(start, min(start + 4096, len(doc_ids)))
        ]
        shard.update(UpdateOperation.upsert_points(pts))
    shard.optimize()
    shard.flush()
    shard.close()
    print(f"docs: {len(doc_ids)} points in {time.time()-t0:.0f}s", flush=True)


def measure():
    from qdrant_edge import EdgeShard, Query, QueryRequest
    from transformers import AutoTokenizer

    out = {}
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained("lightretriever/lightretriever-qwen2.5-1.5b")
    out["tokenizer_load_s"] = round(time.perf_counter() - t0, 3)
    t0 = time.perf_counter()
    table_shard = EdgeShard.load(TABLE_SHARD)
    docs_shard = EdgeShard.load(DOCS_SHARD)
    out["shards_load_s"] = round(time.perf_counter() - t0, 3)
    out["shards_load_note"] = "warm OS page cache; true cold start (fresh download) is bounded by bundle size / bandwidth"

    _, _, q_ids, q_texts, qrels = load_beir(DATASET)
    lat_lookup, lat_search = [], []
    run = {}
    for qid, text in zip(q_ids, q_texts):
        t1 = time.perf_counter()  # timer includes tokenization, matching how baselines are timed
        ids = tok(text, add_special_tokens=False, truncation=True, max_length=512)["input_ids"]
        recs = table_shard.retrieve(point_ids=ids, with_payload=False, with_vector=True)
        vecs = np.array([r.vector["tok"] for r in recs], dtype=np.float32)
        # retrieve() returns unique ids; re-expand to count repeated tokens per occurrence
        by_id = {r.id: i for i, r in enumerate(recs)}
        v = vecs[[by_id[i] for i in ids if i in by_id]].mean(0)
        v = v / (np.linalg.norm(v) + 1e-12)
        t2 = time.perf_counter()
        res = docs_shard.query(QueryRequest(query=Query.Nearest(v.tolist(), using="dense"), limit=100, with_payload=True, with_vector=False))
        t3 = time.perf_counter()
        lat_lookup.append((t2 - t1) * 1000)
        lat_search.append((t3 - t2) * 1000)
        # same self-hit drop as core.topk_run, so edge_metrics and exact_metrics measure the same thing
        run[qid] = {p.payload["doc_id"]: float(p.score) for p in res if p.payload["doc_id"] != qid}
    out["lookup_ms_p50"] = round(statistics.median(lat_lookup), 2)
    out["search_ms_p50"] = round(statistics.median(lat_search), 2)
    out["total_ms_p50"] = round(statistics.median([a + b for a, b in zip(lat_lookup, lat_search)]), 2)
    out["edge_metrics"] = score_run(run, qrels)

    # brute-force reference on the same vectors (ANN recall confound check)
    doc_ids, doc_vecs = load_vecs("lightretriever-qwen2.5-1.5b", DATASET, "doc")
    table = np.load(LR / "table_websearch.npy").astype(np.float32)
    qv = np.zeros((len(q_ids), table.shape[1]), dtype=np.float32)
    for i, text in enumerate(q_texts):
        ids = tok(text, add_special_tokens=False, truncation=True, max_length=512)["input_ids"]
        v = table[ids].mean(0)
        qv[i] = v / (np.linalg.norm(v) + 1e-12)
    sims = qv @ doc_vecs.astype(np.float32).T
    out["exact_metrics"] = score_run(topk_run(doc_ids, sims, q_ids), qrels)

    def du_mb(p):
        return round(sum(f.stat().st_size for f in Path(p).rglob("*") if f.is_file()) / 1e6, 1)

    out["table_shard_mb"] = du_mb(TABLE_SHARD)
    out["docs_shard_mb"] = du_mb(DOCS_SHARD)
    (REPO / "results" / "edge_prototype.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"build": build, "measure": measure}[sys.argv[1]]()
