"""The two-collection architecture running our table, on Qdrant.

  collection `token_table`  one point per vocab token, vector = that token's table row.
                            Created with HNSW disabled (m=0): it is retrieve-by-id only, and
                            M5 found that indexing it inflated the shard from 466 MB of raw
                            fp16 to 1.82 GB for no benefit.
  collection `docs`         document vectors from the frozen teacher, HNSW indexed.

Query path: tokenize -> retrieve rows by id -> weighted average -> normalize -> ANN search.
No transformer anywhere in it.

M5 measured this on Qdrant Edge shards in-process (0.9 ms/query). `qdrant-edge` is not
installable on this box (see m7/STATUS.md), so timings here run against the standalone v1.19.0
server and therefore include client/transport overhead -- they are labeled `server_mode` and are
NOT comparable to M5's in-process numbers. The comparable edge number is the in-process
encode latency in results/m7_costs.json.
"""
import json
import statistics
import time

import numpy as np

import ann_sweep
import dev_eval
from _paths import REPO, WORK
from evalkit import per_query_ndcg
from table import NO_PREFIX, WITH_PREFIX, get_tokenizer, load_table, tokenize

PRE = {"noprefix": NO_PREFIX, "prefix": WITH_PREFIX}


def build_token_collection(client, name, rows, batch=4096):
    from qdrant_client import models
    client.recreate_collection(
        collection_name=name,
        vectors_config=models.VectorParams(size=rows.shape[1], distance=models.Distance.COSINE,
                                           datatype=models.Datatype.FLOAT16),
        hnsw_config=models.HnswConfigDiff(m=0),      # retrieve-by-id only: no graph to build
    )
    for lo in range(0, len(rows), batch):
        hi = min(lo + batch, len(rows))
        client.upsert(collection_name=name, wait=(hi == len(rows)), points=models.Batch(
            ids=list(range(lo, hi)), vectors=rows[lo:hi].astype(np.float32).tolist()))


def encode_via_qdrant(client, token_coll, ids_list, weights, dim):
    """The edge query path, with the rows fetched from Qdrant rather than a local array."""
    out = np.zeros((len(ids_list), dim), dtype=np.float32)
    for i, ids in enumerate(ids_list):
        uniq = sorted(set(ids))
        pts = client.retrieve(collection_name=token_coll, ids=uniq, with_vectors=True,
                              with_payload=False)
        vec = {p.id: np.asarray(p.vector, dtype=np.float32) for p in pts}
        num = np.zeros(dim, dtype=np.float32)
        den = 0.0
        for t in ids:                                  # multiplicity kept, as in the frozen rule
            w = 1.0 if weights is None else float(weights[t])
            num += w * vec[t]
            den += w
        v = num / max(den, 1e-6)
        n = np.linalg.norm(v)
        out[i] = v / n if n > 1e-6 else v
    return out


def run(run_id, preproc="noprefix", component="cqadup-physics", ef=512, n_timed=100):
    pre = PRE[preproc]
    npz = WORK / "runs" / f"{run_id}.npz"
    z = np.load(npz)
    rows = z["rows_int8"].astype(np.float32) * z["int8_scale"][:, None]
    w = z["token_weights"]
    weights = w if w.size else None
    tok = get_tokenizer()

    doc_ids, _, q_ids, q_texts, qrels, dv = dev_eval.doc_vecs(component)
    ids_list = tokenize(tok, q_texts, pre)
    local = load_table(npz, variant="int8", device="cpu")
    local_qv = local.encode(q_texts, pre, tok=tok, device="cpu")

    out = {"_note": "server_mode timings include client/transport overhead and are NOT comparable "
                    "to M5's in-process Qdrant Edge numbers; see the module docstring.",
           "run_id": run_id, "component": component, "n_docs": len(doc_ids),
           "n_queries": len(q_ids), "ef": ef}
    with ann_sweep.Server() as client:
        t0 = time.time()
        build_token_collection(client, "token_table", rows)
        out["token_collection_build_s"] = round(time.time() - t0, 2)
        out["doc_collection_build_s"] = ann_sweep.index(client, "docs", dv, dv.shape[1])

        qv = encode_via_qdrant(client, "token_table", ids_list, weights, rows.shape[1])
        out["max_abs_diff_vs_local_encode"] = float(np.abs(qv - local_qv).max())

        lat_lookup, lat_search, lat_total = [], [], []
        from qdrant_client import models
        params = models.SearchParams(hnsw_ef=ef)
        for i in range(min(n_timed, len(q_texts))):
            t = time.perf_counter()
            v = encode_via_qdrant(client, "token_table", [ids_list[i]], weights, rows.shape[1])[0]
            t1 = time.perf_counter()
            client.query_points(collection_name="docs", query=v.tolist(), limit=10,
                                params=params, with_payload=False)
            t2 = time.perf_counter()
            lat_lookup.append((t1 - t) * 1000)
            lat_search.append((t2 - t1) * 1000)
            lat_total.append((t2 - t) * 1000)
        out["server_mode_ms"] = {"lookup_median": round(statistics.median(lat_lookup), 3),
                                 "search_median": round(statistics.median(lat_search), 3),
                                 "total_median": round(statistics.median(lat_total), 3)}

        run_ = {}
        from qdrant_client import models as M
        res = client.query_batch_points(collection_name="docs", requests=[
            M.QueryRequest(query=v.tolist(), limit=1000, params=M.SearchParams(hnsw_ef=ef),
                           with_payload=False) for v in qv])
        for qid, r in zip(q_ids, res):
            run_[qid] = {doc_ids[p.id]: float(p.score) for p in r.points if doc_ids[p.id] != qid}
        out["ndcg@10_two_collection_ann"] = round(float(np.mean(list(per_query_ndcg(run_, qrels).values()))), 4)

    from evalkit import score
    out["ndcg@10_exact_same_vectors"] = round(float(np.mean(list(
        score(local_qv, q_ids, dv, doc_ids, qrels, chunk=dev_eval.CHUNK.get(component, 200_000)).values()))), 4)
    (REPO / "results" / "m7_edge_demo.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    import sys
    run(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "noprefix")
