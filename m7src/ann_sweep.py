"""ANN behaviour of the released lookup table, on a real Qdrant HNSW index.

M5 found lookup-query vectors are harder for HNSW than transformer-query vectors (-2.1 nDCG at
default ef on FiQA vs -0.7 for bge-small, mostly recovered at ef=512), so the released table
gets its own sweep. It runs on DEV components: scoring a model against six-set qrels outside the
final run is forbidden by the protocol, so the six-set ANN row is produced by the final scorer.

Qdrant runs as the standalone v1.19.0 binary (~/qdrant-bin/qdrant) -- no Docker, so no host-side
change was needed on the Windows box.
"""
import json
import os
import shutil
import signal
import socket
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np

from _paths import REPO, WORK
from evalkit import per_query_ndcg

QDRANT = Path.home() / "qdrant-bin" / "qdrant"
STORAGE = WORK / "qdrant"
EFS = [None, 64, 128, 256, 512]


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class Server:
    def __init__(self):
        self.http = free_port()
        self.grpc = free_port()
        if STORAGE.exists():
            shutil.rmtree(STORAGE)
        STORAGE.mkdir(parents=True)
        env = {**os.environ, "QDRANT__SERVICE__HTTP_PORT": str(self.http),
               "QDRANT__SERVICE__GRPC_PORT": str(self.grpc),
               "QDRANT__STORAGE__STORAGE_PATH": str(STORAGE / "storage"),
               "QDRANT__STORAGE__SNAPSHOTS_PATH": str(STORAGE / "snapshots"),
               "QDRANT__TELEMETRY_DISABLED": "true"}
        self.log = open(WORK / "qdrant.log", "w")
        self.p = subprocess.Popen([str(QDRANT)], env=env, cwd=str(STORAGE),
                                  stdout=self.log, stderr=subprocess.STDOUT)

    def __enter__(self):
        from qdrant_client import QdrantClient
        for _ in range(120):
            try:
                c = QdrantClient(host="127.0.0.1", port=self.http, timeout=120)
                c.get_collections()
                self.client = c
                return c
            except Exception:
                time.sleep(0.5)
        raise RuntimeError("qdrant did not come up; see work/qdrant.log")

    def __exit__(self, *a):
        self.p.send_signal(signal.SIGTERM)
        try:
            self.p.wait(30)
        except subprocess.TimeoutExpired:
            self.p.kill()
        self.log.close()


def index(client, name, vecs, dim, batch=2048):
    from qdrant_client import models
    client.recreate_collection(
        collection_name=name,
        vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE,
                                           datatype=models.Datatype.FLOAT16),
        hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100),
        optimizers_config=models.OptimizersConfigDiff(default_segment_number=2),
    )
    t0 = time.time()
    for lo in range(0, len(vecs), batch):
        hi = min(lo + batch, len(vecs))
        client.upsert(collection_name=name, wait=False, points=models.Batch(
            ids=list(range(lo, hi)), vectors=np.asarray(vecs[lo:hi], dtype=np.float32).tolist()))
    client.update_collection(collection_name=name,
                            optimizer_config=models.OptimizersConfigDiff(indexing_threshold=1))
    for _ in range(3600):
        info = client.get_collection(name)
        if info.status == models.CollectionStatus.GREEN and info.indexed_vectors_count:
            break
        time.sleep(2)
    return round(time.time() - t0, 1)


def search(client, name, qv, doc_ids, q_ids, k=10, ef=None, timed_n=50):
    from qdrant_client import models
    params = models.SearchParams(hnsw_ef=ef) if ef else None
    reqs = [models.QueryRequest(query=v.tolist(), limit=k, params=params, with_payload=False)
            for v in np.asarray(qv, dtype=np.float32)]
    t0 = time.time()
    res = client.query_batch_points(collection_name=name, requests=reqs)
    total = time.time() - t0
    run = {}
    for qid, r in zip(q_ids, res):
        run[qid] = {doc_ids[p.id]: float(p.score) for p in r.points if doc_ids[p.id] != qid}
    lat = []
    for v in np.asarray(qv[:timed_n], dtype=np.float32):
        t = time.perf_counter()
        client.query_points(collection_name=name, query=v.tolist(), limit=k, params=params,
                            with_payload=False)
        lat.append((time.perf_counter() - t) * 1000)
    return run, round(total / len(q_ids) * 1000, 3), round(statistics.median(lat), 3)


def sweep(components=("cqadup-physics", "nq-250k"), table_npz=None, preproc="noprefix",
          out="m7_ann_sweep.json"):
    import dev_eval
    from evalkit import score
    from table import NO_PREFIX, WITH_PREFIX, Preproc, load_table, read_meta
    # The artifact's OWN metadata is the authority on how it is queried; a name-keyed lookup
    # silently serves the default pooling rule to a table whose frozen rule is something else.
    pre = (Preproc(**read_meta(table_npz)["preproc"]) if table_npz
           else {"noprefix": NO_PREFIX, "prefix": WITH_PREFIX}[preproc])
    print(f"  ann sweep query rule: {pre}", flush=True)
    results = {}
    with Server() as client:
        for comp in components:
            doc_ids, _, q_ids, q_texts, qrels, dv = dev_eval.doc_vecs(comp)
            build_s = index(client, comp.replace("-", "_"), dv, dv.shape[1])
            row = {"n_docs": len(doc_ids), "index_build_s": build_s, "systems": {}}
            # candidate query vectors: the table (int8, the released artifact) + the teacher
            qvs = {}
            if table_npz:
                m = load_table(table_npz, variant="int8")
                qvs["int8-table"] = m.encode(q_texts, pre)
                del m
            import torch
            from teacher import QUERY_PREFIX, encode_cached
            qvs["teacher-query"] = np.asarray(
                encode_cached(f"dev-{comp}-queries-pfx", q_texts, prefix=QUERY_PREFIX,
                              dtype=torch.float16, verbose=False), dtype=np.float32)
            for sysname, qv in qvs.items():
                exact = float(np.mean(list(score(qv, q_ids, dv, doc_ids, qrels,
                                                 chunk=dev_eval.CHUNK.get(comp, 200_000)).values())))
                rows = {"exact": {"ndcg@10": round(exact, 4)}}
                for ef in EFS:
                    run, amort_ms, med_ms = search(client, comp.replace("-", "_"), qv, doc_ids, q_ids, ef=ef)
                    n = float(np.mean(list(per_query_ndcg(run, qrels).values())))
                    rows[f"ef={ef or 'default'}"] = {"ndcg@10": round(n, 4),
                                                     "delta_vs_exact": round(n - exact, 4),
                                                     "amortized_ms": amort_ms, "median_ms": med_ms}
                    print(f"  {comp} {sysname} ef={ef or 'default'}: {n:.4f} "
                          f"({n-exact:+.4f} vs exact) {med_ms:.2f} ms", flush=True)
                row["systems"][sysname] = rows
            results[comp] = row
    (REPO / "results" / out).write_text(json.dumps(results, indent=1))
    print(f"wrote results/{out}")
    return results


if __name__ == "__main__":
    import sys
    t = WORK / "runs" / f"{sys.argv[1]}.npz" if len(sys.argv) > 1 else None
    sweep(table_npz=t, preproc=sys.argv[2] if len(sys.argv) > 2 else "noprefix")
