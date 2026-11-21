"""Qdrant Edge demo index and benchmark for the 2026-11-23 talk (exploratory).

One in-process Edge shard holds Stella document vectors (fp16, binary quantization in RAM) plus a
BM25 sparse vector. A second shard holds Zero's token table, so Zero's whole query path is two
Edge calls: retrieve the query's token rows, then search.

  .venv/bin/python m15src/talk_edge.py build fiqa|nq
  .venv/bin/python m15src/talk_edge.py parity fiqa|nq     # Edge Zero == released Zero
  .venv/bin/python m15src/talk_edge.py bench fiqa|nq      # writes results/m15_talk_edge_<ds>.json
"""
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from common import REPO, load_public, receipt, sha_file, utc_now, write_result

sys.path.insert(0, str(REPO / "bench"))
EDGE = REPO / "work/m15/edge"
NQ_VECS = REPO / "work/m15/vecs/nq_stella"
EFS = (32, 64, 128, 256, 512, 1024)
OVERSAMPLING = (1.0, 2.0, 4.0, 8.0)
TEXT_CHARS = 700


def doc_vectors(dataset, data):
    if dataset == "nq":
        ids = json.loads((NQ_VECS / "doc_ids.json").read_text())
        if ids != data["doc_ids"]:
            raise RuntimeError("NQ vector ids differ from the pinned corpus")
        shards = sorted(NQ_VECS.glob("shard_*.npy"))
        vecs = np.concatenate([np.load(p) for p in shards])
        if len(vecs) != len(ids):
            raise RuntimeError(f"NQ has {len(vecs)} vectors for {len(ids)} docs")
        return vecs, {"source": str(NQ_VECS.relative_to(REPO)), "n_shards": len(shards)}
    import vectors15
    return vectors15.doc_vectors(dataset, data)


def avg_words(texts):
    return float(sum(len(t.split()) for t in texts) / len(texts))


def build(dataset):
    import qdrant_edge as E
    from datasets import load_dataset
    data = load_public(dataset)
    vecs, prov = doc_vectors(dataset, data)
    avg_len = avg_words(data["doc_texts"])
    corpus = load_dataset(data["pins"]["source"], "corpus", revision=data["pins"]["revision"])["corpus"]
    titles, bodies = corpus["title"], corpus["text"]
    bm25 = E.Bm25(E.Bm25Config(avg_len=avg_len))
    path = EDGE / dataset
    if path.exists():
        raise SystemExit(f"{path} exists; delete it to rebuild")
    path.mkdir(parents=True)
    cfg = E.EdgeConfig(
        vectors={"stella": E.EdgeVectorParams(
            size=vecs.shape[1], distance=E.Distance.Cosine, datatype=E.VectorStorageDatatype.Float16,
            on_disk=True, hnsw_config=E.HnswIndexConfig(m=16, ef_construct=100, full_scan_threshold=10_000),
            quantization_config=E.BinaryQuantizationConfig(always_ram=True))},
        sparse_vectors={"bm25": E.EdgeSparseVectorParams(modifier=E.Modifier.Idf)},
        on_disk_payload=True)
    shard = E.EdgeShard.create(str(path), cfg)
    t0 = time.time()
    for lo in range(0, len(vecs), 2048):
        hi = min(lo + 2048, len(vecs))
        pts = []
        for i in range(lo, hi):
            text = data["doc_texts"][i]
            pts.append(E.Point(id=i, vector={"stella": vecs[i].astype(np.float32),
                                             "bm25": bm25.embed_document(text)},
                               payload={"doc_id": data["doc_ids"][i], "title": titles[i] or "",
                                        "text": (bodies[i] or "")[:TEXT_CHARS]}))
        shard.update(E.UpdateOperation.upsert_points(pts))
        if lo % 204800 == 0:
            print(f"upserted {hi}/{len(vecs)} in {time.time() - t0:.0f}s", flush=True)
    print("optimizing (HNSW + quantization)", flush=True)
    t1 = time.time()
    shard.optimize()
    shard.flush()
    shard.close()
    meta = {"dataset": dataset, "n_docs": len(vecs), "bm25_avg_len": avg_len, "vectors": prov,
            "upsert_s": round(t1 - t0), "optimize_s": round(time.time() - t1),
            "hnsw": {"m": 16, "ef_construct": 100}, "datatype": "float16",
            "quantization": "binary, always_ram"}
    (path.parent / f"{dataset}.meta.json").write_text(json.dumps(meta, indent=1))
    build_zero_table()
    print(json.dumps(meta), flush=True)


def build_zero_table():
    import qdrant_edge as E
    import encoders15
    path = EDGE / "zero_table"
    if path.exists():
        return
    zero = encoders15.ZeroEncoder().model     # released int8 table, dequantized exactly as served
    path.mkdir(parents=True)
    shard = E.EdgeShard.create(str(path), E.EdgeConfig(vectors={"row": E.EdgeVectorParams(
        size=zero.dim, distance=E.Distance.Dot)}))
    for lo in range(0, zero.rows.shape[0], 4096):
        shard.update(E.UpdateOperation.upsert_points([
            E.Point(id=i, vector={"row": zero.rows[i]}, payload=None)
            for i in range(lo, min(lo + 4096, zero.rows.shape[0]))]))
    shard.flush()
    shard.close()


class EdgeZero:
    """Zero's query path with the table stored in an Edge shard: tokenize, retrieve rows, pool."""

    def __init__(self):
        import qdrant_edge as E
        import encoders15
        self.ref = encoders15.ZeroEncoder().model
        self.tok = self.ref.tokenizer
        self.table = E.EdgeShard.load(str(EDGE / "zero_table"))
        self.name = "zero"

    def encode_one(self, text):
        ids = self.tok.encode(text).ids
        if not ids:
            return self.ref._fallback
        uniq, counts = np.unique(np.asarray(ids, dtype=np.int64), return_counts=True)
        recs = self.table.retrieve(point_ids=[int(u) for u in uniq], with_payload=False,
                                   with_vector=True)
        by_id = {r.id: np.asarray(r.vector["row"], dtype=np.float32) for r in recs}
        rows = np.stack([by_id[int(u)] for u in uniq])
        w = np.sqrt(counts).astype(np.float32)
        v = (rows * w[:, None]).sum(0) / max(float(w.sum()), 1e-6)
        n = float(np.linalg.norm(v))
        return self.ref._fallback if n <= 1e-6 else (v / n).astype(np.float32)

    def encode(self, texts):
        return np.stack([self.encode_one(t) for t in texts])


def parity(dataset):
    data = load_public(dataset, with_corpus=False)
    z = EdgeZero()
    qs = data["q_texts"][:500]
    diff = float(np.abs(z.encode(qs) - z.ref.encode(qs)).max())
    print(f"Edge Zero vs released Zero, {len(qs)} queries: max abs diff {diff:.2e}")
    if diff > 1e-5:
        raise SystemExit("parity FAILED")


def exact_topk(q, vecs, k=10, chunk=262_144):
    best_s = np.full((len(q), k), -np.inf, dtype=np.float32)
    best_i = np.zeros((len(q), k), dtype=np.int64)
    for lo in range(0, len(vecs), chunk):
        s = q @ vecs[lo:lo + chunk].astype(np.float32).T
        cand_s = np.concatenate([best_s, s], 1)
        cand_i = np.concatenate([best_i, np.arange(lo, lo + s.shape[1])[None].repeat(len(q), 0)], 1)
        top = np.argpartition(-cand_s, k - 1, axis=1)[:, :k]
        best_s = np.take_along_axis(cand_s, top, 1)
        best_i = np.take_along_axis(cand_i, top, 1)
    order = np.argsort(-best_s, axis=1)
    return np.take_along_axis(best_i, order, 1), np.take_along_axis(best_s, order, 1)


def per_query_ndcg(run, qrels, q_ids):
    import pytrec_eval
    s = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut.10"}).evaluate(run)
    return [round(s[q]["ndcg_cut_10"], 4) if q in s else 0.0 for q in q_ids]


def bench(dataset):
    import qdrant_edge as E
    import encoders15
    from core import score_run
    started = utc_now()
    data = load_public(dataset)
    vecs, prov = doc_vectors(dataset, data)
    meta = json.loads((EDGE / f"{dataset}.meta.json").read_text())
    shard = E.EdgeShard.load(str(EDGE / dataset))
    q_ids, q_texts, qrels, ids = data["q_ids"], data["q_texts"], data["qrels"], data["doc_ids"]

    def run_of(rows):
        return {q: {ids[i]: float(s) for i, s in r if ids[i] != q} for q, r in zip(q_ids, rows)}

    # "zero" is the released in-memory path; "zero-edge" reads the same rows from an Edge shard.
    encoders = {"stella": encoders15.stella_query(), "nano": encoders15.nano(),
                "zero": encoders15.ZeroEncoder(), "zero-edge": EdgeZero()}
    out = {}
    for name, enc in encoders.items():
        enc.encode(q_texts[:20])                                   # warm
        enc_ms, qv = [], []
        for t in q_texts:
            t0 = time.perf_counter()
            v = enc.encode([t])[0]
            enc_ms.append((time.perf_counter() - t0) * 1000)
            qv.append(v)
        qv = np.stack(qv).astype(np.float32)
        ex_i, ex_s = exact_topk(qv, vecs)
        exact = score_run(run_of([zip(a, b) for a, b in zip(ex_i, ex_s)]), qrels)["ndcg@10"]
        rows = {"encode_ms_p50": statistics.median(enc_ms), "exact_ndcg10": exact, "settings": []}
        raw = {}
        for ef in EFS:
            for os_ in OVERSAMPLING:
                params = E.SearchParams(hnsw_ef=ef, quantization=E.QuantizationSearchParams(
                    rescore=True, oversampling=os_))
                for v in qv[:20]:
                    shard.query(E.QueryRequest(query=E.Query.Nearest(v, using="stella"), limit=10,
                                               params=params))
                srch_ms, res, recov = [], [], []
                for qi, v in enumerate(qv):
                    t0 = time.perf_counter()
                    hits = shard.query(E.QueryRequest(query=E.Query.Nearest(v, using="stella"),
                                                      limit=10, params=params))
                    srch_ms.append((time.perf_counter() - t0) * 1000)
                    res.append([(h.id, h.score) for h in hits])
                    recov.append(len({h.id for h in hits} & set(ex_i[qi].tolist())) / 10)
                tot = [a + b for a, b in zip(enc_ms, srch_ms)]
                raw[(ef, os_)] = (srch_ms, res)
                rows["settings"].append({
                    "ef": ef, "oversampling": os_,
                    "ndcg10": score_run(run_of(res), qrels)["ndcg@10"],
                    "own_exact_top10_recovery": float(np.mean(recov)),
                    "search_ms_p50": statistics.median(srch_ms),
                    "total_ms_p50": statistics.median(tot),
                    "total_ms_p95": float(np.percentile(tot, 95))})
                print(name, rows["settings"][-1], flush=True)
        ok = [s for s in rows["settings"] if s["ndcg10"] >= 0.99 * exact]
        rows["selected"] = min(ok, key=lambda s: s["total_ms_p50"]) if ok else None
        pick = rows["selected"] or max(rows["settings"], key=lambda s: s["ndcg10"])
        srch_ms, res = raw[(pick["ef"], pick["oversampling"])]
        # Per-query arrays for the demo's race replay, at the selected (else best) setting.
        rows["per_query"] = {"ef": pick["ef"], "oversampling": pick["oversampling"],
                             "encode_ms": [round(x, 4) for x in enc_ms],
                             "search_ms": [round(x, 4) for x in srch_ms],
                             "ndcg10": per_query_ndcg(run_of(res), qrels, q_ids)}
        out[name] = rows
    bm = E.Bm25(E.Bm25Config(avg_len=meta["bm25_avg_len"]))
    for t in q_texts[:20]:
        shard.query(E.QueryRequest(query=E.Query.Nearest(bm.embed_query(t), using="bm25"), limit=10))
    tot, res = [], []
    for t in q_texts:
        t0 = time.perf_counter()
        hits = shard.query(E.QueryRequest(query=E.Query.Nearest(bm.embed_query(t), using="bm25"),
                                          limit=10))
        tot.append((time.perf_counter() - t0) * 1000)
        res.append([(h.id, h.score) for h in hits])
    out["bm25"] = {"ndcg10": score_run(run_of(res), qrels)["ndcg@10"],
                   "total_ms_p50": statistics.median(tot), "total_ms_p95": float(np.percentile(tot, 95)),
                   "per_query": {"total_ms": [round(x, 4) for x in tot],
                                 "ndcg10": per_query_ndcg(run_of(res), qrels, q_ids)}}
    write_result(REPO / f"results/m15_talk_edge_{dataset}.json", {
        "status": "COMPLETE", "analysis": "exploratory", "dataset": dataset,
        "n_queries": len(q_ids), "n_docs": len(ids), "index": meta, "results": out,
        "protocol": "Warmed, sequential, one process on the Apple M5 Pro. Encoding and search timed "
                    "separately per query; totals are per-query sums. Transformers run ONNX Runtime "
                    "CPU, 4 threads. zero-edge retrieves Zero's rows from an Edge shard. Selected = fastest "
                    "median total within 1% of the encoder's own exact nDCG@10.",
        "receipt": receipt(__file__, {"doc_vectors": prov}, started,
                           ("onnxruntime", "tokenizers", "qdrant-edge"))})


if __name__ == "__main__":
    {"build": build, "parity": parity, "bench": bench}[sys.argv[1]](sys.argv[2])
