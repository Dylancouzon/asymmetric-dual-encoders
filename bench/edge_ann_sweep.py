"""Does the exact-search ranking survive ANN? ef sweeps per system/corpus on Qdrant Edge.

Systems: bge-small (384d) and LightRetriever websearch-table dense (1536d);
corpora: fiqa (57.6K) and trec-covid (171K). LR/fiqa numbers come from edge_variant.py.
Writes results/ann_sweep.json.
"""
import json
import statistics
import time
from pathlib import Path

import numpy as np
from qdrant_edge import (Distance, EdgeConfig, EdgeShard, EdgeVectorParams, Point, Query, QueryRequest,
                         SearchParams, UpdateOperation, VectorStorageDatatype)

from core import REPO, evaluate, load_beir, load_vecs, score_run
from edge_prototype import point_id

EDGE_DIR = REPO / "edge_data"
LR = REPO / "artifacts/lightretriever-qwen2.5-1.5b"
CASES = [("bge-small-en-v1.5", "fiqa"), ("bge-small-en-v1.5", "trec-covid"), ("lightretriever-qwen2.5-1.5b", "trec-covid")]


def query_vecs_for(slug, ds):
    if slug.startswith("lightretriever"):
        from transformers import AutoTokenizer

        from run_lightretriever import queries_from_table, query_token_ids

        tok = AutoTokenizer.from_pretrained("lightretriever/lightretriever-qwen2.5-1.5b")
        _, _, q_ids, q_texts, qrels = load_beir(ds)
        table = np.load(LR / "table_websearch.npy").astype(np.float32)
        return q_ids, queries_from_table(table, query_token_ids(tok, q_texts)), qrels
    q_ids, qv = load_vecs(slug, ds, "query")
    *_, qrels = load_beir(ds)
    return q_ids, qv.astype(np.float32), qrels


out = {}
for slug, ds in CASES:
    doc_ids, doc_vecs = load_vecs(slug, ds, "doc")
    doc_vecs = doc_vecs.astype(np.float32)
    q_ids, q_vecs, qrels = query_vecs_for(slug, ds)
    shard_dir = str(EDGE_DIR / f"sweep_{slug[:12]}_{ds}")
    if not Path(shard_dir, "config.json").exists() and not list(Path(shard_dir).glob("**/*")):
        Path(shard_dir).mkdir(parents=True, exist_ok=True)
        p = EdgeVectorParams(size=doc_vecs.shape[1], distance=Distance.Dot, datatype=VectorStorageDatatype.Float16, on_disk=True)
        shard = EdgeShard.create(shard_dir, EdgeConfig(vectors={"v": p}))
        for s in range(0, len(doc_ids), 4096):
            pts = [Point(id=point_id(doc_ids[i]), vector={"v": doc_vecs[i].tolist()}, payload={"doc_id": doc_ids[i]})
                   for i in range(s, min(s + 4096, len(doc_ids)))]
            shard.update(UpdateOperation.upsert_points(pts))
        shard.optimize()
        shard.flush()
    else:
        shard = EdgeShard.load(shard_dir)
    key = f"{slug}/{ds}"
    out[key] = {"exact": evaluate(doc_ids, doc_vecs, q_ids, q_vecs, qrels)}
    for ef in [None, 128, 256, 512]:
        run, lat = {}, []
        for qi, qid in enumerate(q_ids):
            kw = {"params": SearchParams(hnsw_ef=ef)} if ef else {}
            t0 = time.perf_counter()
            res = shard.query(QueryRequest(query=Query.Nearest(q_vecs[qi].tolist(), using="v"), limit=100, with_payload=True, with_vector=False, **kw))
            lat.append((time.perf_counter() - t0) * 1000)
            run[qid] = {p.payload["doc_id"]: float(p.score) for p in res if p.payload["doc_id"] != qid}
        m = score_run(run, qrels)
        out[key][f"ef={ef or 'default'}"] = {"ndcg@10": round(m["ndcg@10"], 4), "recall@100": round(m["recall@100"], 4), "ms_p50": round(statistics.median(lat), 2)}
        print(key, f"ef={ef}", out[key][f"ef={ef or 'default'}"], flush=True)
    shard.close()
(REPO / "results" / "ann_sweep.json").write_text(json.dumps(out, indent=1))
