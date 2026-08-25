"""Edge prototype, deployment-tuned variant: fp16 + on-disk storage, hnsw_ef sweep.
Same query path as edge_prototype.py measure; writes results/edge_variant.json."""
import json
import statistics
import time
from pathlib import Path

import numpy as np
from qdrant_edge import (Distance, EdgeConfig, EdgeShard, EdgeVectorParams, Point, Query, QueryRequest,
                         SearchParams, UpdateOperation, VectorStorageDatatype)
from transformers import AutoTokenizer

from core import REPO, load_beir, load_vecs, score_run

EDGE_DIR = REPO / "edge_data"
TABLE_SHARD = str(EDGE_DIR / "token_table_f16")
DOCS_SHARD = str(EDGE_DIR / "docs_fiqa_f16")
LR = REPO / "artifacts/lightretriever-qwen2.5-1.5b"


def build():
    table = np.load(LR / "table_websearch.npy").astype(np.float32)
    Path(TABLE_SHARD).mkdir(parents=True, exist_ok=True)
    p = EdgeVectorParams(size=table.shape[1], distance=Distance.Dot, datatype=VectorStorageDatatype.Float16, on_disk=True)
    shard = EdgeShard.create(TABLE_SHARD, EdgeConfig(vectors={"tok": p}))
    for start in range(0, table.shape[0], 4096):
        pts = [Point(id=i, vector={"tok": table[i].tolist()}, payload={}) for i in range(start, min(start + 4096, table.shape[0]))]
        shard.update(UpdateOperation.upsert_points(pts))
    shard.optimize()
    shard.flush()
    shard.close()
    doc_ids, doc_vecs = load_vecs("lightretriever-qwen2.5-1.5b", "fiqa", "doc")
    doc_vecs = doc_vecs.astype(np.float32)
    Path(DOCS_SHARD).mkdir(parents=True, exist_ok=True)
    p = EdgeVectorParams(size=doc_vecs.shape[1], distance=Distance.Dot, datatype=VectorStorageDatatype.Float16, on_disk=True)
    shard = EdgeShard.create(DOCS_SHARD, EdgeConfig(vectors={"dense": p}))
    for start in range(0, len(doc_ids), 4096):
        pts = [Point(id=int(doc_ids[i]), vector={"dense": doc_vecs[i].tolist()}, payload={"doc_id": doc_ids[i]})
               for i in range(start, min(start + 4096, len(doc_ids)))]
        shard.update(UpdateOperation.upsert_points(pts))
    shard.optimize()
    shard.flush()
    shard.close()


def measure():
    tok = AutoTokenizer.from_pretrained("lightretriever/lightretriever-qwen2.5-1.5b")
    table_shard = EdgeShard.load(TABLE_SHARD)
    docs_shard = EdgeShard.load(DOCS_SHARD)
    _, _, q_ids, q_texts, qrels = load_beir("fiqa")
    out = {}
    for ef in [None, 128, 256, 512]:
        params = SearchParams(hnsw_ef=ef) if ef else None
        lat, run = [], {}
        for qid, text in zip(q_ids, q_texts):
            t1 = time.perf_counter()
            ids = tok(text, add_special_tokens=False, truncation=True, max_length=512)["input_ids"]
            recs = table_shard.retrieve(point_ids=ids, with_payload=False, with_vector=True)
            by_id = {r.id: np.array(r.vector["tok"], dtype=np.float32) for r in recs}
            v = np.mean([by_id[i] for i in ids if i in by_id], axis=0)
            v = v / (np.linalg.norm(v) + 1e-12)
            kw = {"params": params} if params else {}
            res = docs_shard.query(QueryRequest(query=Query.Nearest(v.tolist(), using="dense"), limit=100, with_payload=True, with_vector=False, **kw))
            lat.append((time.perf_counter() - t1) * 1000)
            run[qid] = {p.payload["doc_id"]: float(p.score) for p in res if p.payload["doc_id"] != qid}
        m = score_run(run, qrels)
        out[f"ef={ef or 'default'}"] = {"ndcg@10": round(m["ndcg@10"], 4), "recall@100": round(m["recall@100"], 4), "total_ms_p50": round(statistics.median(lat), 2)}
        print(f"ef={ef}: {out[f'ef={ef or 'default'}']}", flush=True)

    def du_mb(p):
        return round(sum(f.stat().st_size for f in Path(p).rglob("*") if f.is_file()) / 1e6, 1)

    out["table_shard_mb"] = du_mb(TABLE_SHARD)
    out["docs_shard_mb"] = du_mb(DOCS_SHARD)
    out["exact_ndcg@10"] = 0.4099  # from edge_prototype.json exact_metrics
    (REPO / "results" / "edge_variant.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    import sys

    {"build": build, "measure": measure}[sys.argv[1]]()
