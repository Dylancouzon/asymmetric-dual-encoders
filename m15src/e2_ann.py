"""E2: does the query-compute saving survive the search? (m15/MEASUREMENTS.md, E2)

    e2_ann.py fiqa | msmarco1m [--smoke]

--smoke runs 150 queries, one ef and one oversampling, and writes to work/m15/ (not results/).

Starts the native Qdrant binary (work/m15/qdrant/qdrant, v1.19.1) with its own storage, builds
one collection per quantization setting, sweeps hnsw_ef and oversampling for Zero, Nano and the
Stella query path, and writes results/m15_e2_ann_<dataset>.json. Run only on an idle machine.
"""
import json
import subprocess
import sys
import time
import urllib.request

import numpy as np

from common import REPO, SEED, load_public, receipt, sha_file, utc_now, write_result
import encoders15 as E
import vectors15 as V

QDRANT = REPO / "work" / "m15" / "qdrant"
ENCODERS = ("zero", "nano", "stella-query")
EFS = (16, 32, 64, 128, 256, 512)
OVERSAMPLING = (1.0, 2.0, 4.0)
QUANT = ("none", "int8", "binary1", "turbo4")
TARGETS = (0.01, 0.02, 0.05)
WARMUP = 100
EXACT_CHECK = 200          # queries per encoder searched with exact=true against numpy
FUSED_EF = 128
MSMARCO_REV = "a918e0d11a77ed33f42f29d98340b655593b96ad"


# ---------------------------------------------------------------- data

def load(dataset):
    if dataset == "fiqa":
        data = load_public("fiqa")
        dv, prov = V.doc_vectors("fiqa", data)
        return data, dv, prov
    if dataset != "msmarco1m":
        raise SystemExit(f"unknown E2 dataset {dataset}")
    from datasets import load_dataset
    ids = json.loads((REPO / "work" / "m15" / "msmarco1m_ids.json").read_text())
    data = load_public("msmarco", with_corpus=False)
    corpus = load_dataset("BeIR/msmarco", "corpus", revision=MSMARCO_REV)["corpus"]
    sub = corpus.select(ids["corpus_rows"])
    data["doc_ids"] = ids["doc_ids"]
    data["doc_texts"] = [f"{(t or '').strip()} {(x or '').strip()}".strip()
                         for t, x in zip(sub["title"], sub["text"])]
    shards = sorted((REPO / "work" / "m15" / "vecs" / "msmarco1m").glob("shard_*.npy"))
    dv = np.concatenate([np.load(p) for p in shards])
    if len(dv) != len(data["doc_ids"]):
        raise SystemExit(f"msmarco1m: {len(dv)} vectors for {len(data['doc_ids'])} ids")
    prov = {"ids": "work/m15/msmarco1m_ids.json",
            "ids_sha256": sha_file(REPO / "work" / "m15" / "msmarco1m_ids.json"),
            "n_positives": ids["n_positives"], "vectors": "work/m15/vecs/msmarco1m (Mac MPS)",
            "role": "positive-preserving 1M subset diagnostic; MS MARCO is validation only"}
    return data, dv, prov


# ---------------------------------------------------------------- Qdrant

def start_qdrant(tag):
    storage = QDRANT / f"storage-{tag}"
    log = open(QDRANT / f"qdrant-{tag}.log", "w")
    proc = subprocess.Popen([str(QDRANT / "qdrant")], cwd=QDRANT, stdout=log, stderr=log,
                            env={"QDRANT__STORAGE__STORAGE_PATH": str(storage),
                                 "QDRANT__TELEMETRY_DISABLED": "true",
                                 "PATH": "/usr/bin:/bin"})
    for _ in range(120):
        try:
            urllib.request.urlopen("http://127.0.0.1:6333/readyz", timeout=1)
            return proc
        except Exception:
            time.sleep(0.5)
    proc.kill()
    raise SystemExit("Qdrant did not start")


def quant_config(name):
    from qdrant_client import models as m
    return {"none": None,
            "int8": m.ScalarQuantization(scalar=m.ScalarQuantizationConfig(
                type=m.ScalarType.INT8, quantile=0.99, memory=m.Memory.PINNED)),
            "binary1": m.BinaryQuantization(binary=m.BinaryQuantizationConfig(
                encoding=m.BinaryQuantizationEncoding.ONE_BIT, memory=m.Memory.PINNED)),
            "turbo4": m.TurboQuantization(turbo=m.TurboQuantQuantizationConfig(
                bits=m.TurboQuantBitSize.BITS4, memory=m.Memory.PINNED))}[name]


def build(client, name, dv, quant="none", texts=None, avg_len=None, batch=1024):
    """One collection; with `texts`, also a server-side BM25 sparse vector (IDF modifier)."""
    from qdrant_client import models as m
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(
        name, vectors_config={"dense": m.VectorParams(size=dv.shape[1],
                                                      distance=m.Distance.COSINE)},
        sparse_vectors_config={"bm25": m.SparseVectorParams(modifier=m.Modifier.IDF)}
        if texts else None,
        hnsw_config=m.HnswConfigDiff(m=16, ef_construct=100),
        # Index every segment, so no vector is served by a plain full scan (default 10,000 KB).
        optimizers_config=m.OptimizersConfigDiff(indexing_threshold=1),
        quantization_config=quant_config(quant))
    t0 = time.time()
    for lo in range(0, len(dv), batch):
        hi = min(lo + batch, len(dv))
        vecs = dv[lo:hi].astype(np.float32)
        points = []
        for i in range(lo, hi):
            vector = {"dense": vecs[i - lo].tolist()}
            if texts:
                vector["bm25"] = m.Document(text=texts[i], model="qdrant/bm25",
                                            options={"avg_len": avg_len})
            points.append(m.PointStruct(id=i, vector=vector))
        client.upsert(name, points=points, wait=False)
    while True:            # the index is fully built before any timing
        info = client.get_collection(name)
        if (info.status == m.CollectionStatus.GREEN and info.points_count == len(dv) and
                (info.indexed_vectors_count or 0) >= len(dv)):
            break
        time.sleep(5)
    return {"seconds_to_green": round(time.time() - t0, 1), "points": info.points_count,
            "indexed_vectors": info.indexed_vectors_count}


def search_all(client, name, qvecs, params, limit=11):
    from qdrant_client import models as m
    for v in qvecs[:WARMUP]:
        client.query_points(name, query=v.tolist(), using="dense", limit=limit,
                            search_params=params)
    ids, lat = [], []
    for v in qvecs:
        s = time.perf_counter()
        r = client.query_points(name, query=v.tolist(), using="dense", limit=limit,
                                search_params=params)
        lat.append((time.perf_counter() - s) * 1000)
        ids.append([p.id for p in r.points])
    return ids, np.array(lat)


# ---------------------------------------------------------------- metrics

def top10(id_lists, doc_ids, q_ids):
    """Rank lists -> run of 10 docs, with the registered self-hit drop (evalkit.run_from_arrays)."""
    run = {}
    for q, lst in zip(q_ids, id_lists):
        kept = [doc_ids[i] for i in lst if doc_ids[i] != q][:10]
        run[q] = {d: float(10 - r) for r, d in enumerate(kept)}
    return run


def recovery(ann, exact):
    return float(np.mean([len(set(ann[q]) & set(exact[q])) / max(1, len(exact[q]))
                          for q in exact]))


def decide(rows, exact_ndcg):
    """Per target e: the lowest end-to-end p50 setting with ANN nDCG >= (1 - e) exact."""
    out = {}
    for e in TARGETS:
        ok = [r for r in rows if r["ann_ndcg10"] >= (1 - e) * exact_ndcg]
        best = min(ok, key=lambda r: r["e2e_p50_ms"]) if ok else None
        out[str(e)] = None if best is None else {k: best[k] for k in (
            "quant", "hnsw_ef", "oversampling", "e2e_p50_ms", "search_p50_ms", "ann_ndcg10")}
    return out


# ---------------------------------------------------------------- main

def main(dataset, smoke=False):
    from qdrant_client import QdrantClient, models as m
    global EFS, OVERSAMPLING
    started = utc_now()
    out_path = REPO / "results" / f"m15_e2_ann_{dataset}.json"
    if smoke:
        EFS, OVERSAMPLING = (64,), (2.0,)
        out_path = REPO / "work" / "m15" / f"e2_smoke_{dataset}.json"
        out_path.unlink(missing_ok=True)
    if out_path.exists():
        raise SystemExit(f"{out_path} exists")
    data, dv, prov = load(dataset)
    if smoke:
        keep = data["q_ids"][:150]
        data["q_ids"], data["q_texts"] = keep, data["q_texts"][:150]
        data["qrels"] = {q: data["qrels"][q] for q in keep}
    q_ids, qrels = data["q_ids"], data["qrels"]
    enc = {n: E.make(n) for n in ENCODERS}
    qv, enc_ms, exact_ids, exact, geometry = {}, {}, {}, {}, {}
    for n in ENCODERS:
        vecs, lat = [], []
        for t in data["q_texts"]:          # E1 protocol: batch one, timed in this process
            s = time.perf_counter()
            vecs.append(enc[n].encode([t])[0])
            lat.append((time.perf_counter() - s) * 1000)
        qv[n], enc_ms[n] = np.stack(vecs), np.array(lat)
        from evalkit import topk_arrays
        bi, bs = topk_arrays(qv[n], dv, k=100, chunk=250_000, device="cpu")
        exact_ids[n] = [list(map(int, row)) for row in bi]
        run = top10(exact_ids[n], data["doc_ids"], q_ids)
        exact[n] = {"run": run, "ndcg10": float(np.mean(list(
            V.ndcg10(run, qrels, q_ids).values())))}
        geometry[n] = {"top1_cos_quantiles": np.quantile(bs[:, 0], [.1, .5, .9]).tolist(),
                       "top1_minus_top10_quantiles":
                           np.quantile(bs[:, 0] - bs[:, 9], [.1, .5, .9]).tolist()}
        print(f"{dataset} {n}: exact nDCG@10 {exact[n]['ndcg10']:.4f}, encode p50 "
              f"{np.median(lat):.3f} ms", flush=True)
    proc = start_qdrant(dataset)
    try:
        client = QdrantClient(host="127.0.0.1", grpc_port=6334, prefer_grpc=True, timeout=600)
        rows, builds, exact_check = [], {}, {}
        for qname in QUANT:
            name = f"{dataset}-{qname}"
            builds[qname] = build(client, name, dv, quant=qname)
            print(f"built {name}: {builds[qname]}", flush=True)
            if qname == "none":
                for n in ENCODERS:          # the collection agrees with numpy exact search
                    ids, _ = search_all(client, name, qv[n][:EXACT_CHECK],
                                        m.SearchParams(exact=True), limit=10)
                    exact_check[n] = float(np.mean([len(set(a) & set(b[:10])) / 10 for a, b in
                                                    zip(ids, exact_ids[n][:EXACT_CHECK])]))
                    if exact_check[n] < 0.999:
                        raise SystemExit(f"E2 STOP: Qdrant exact disagrees with numpy for {n}")
            settings = [(ef, None) for ef in EFS] if qname == "none" else \
                [(ef, os) for ef in EFS for os in OVERSAMPLING]
            for ef, os in settings:
                params = m.SearchParams(hnsw_ef=ef, quantization=None if os is None else
                                        m.QuantizationSearchParams(rescore=True, oversampling=os))
                for n in ENCODERS:
                    ids, lat = search_all(client, name, qv[n], params)
                    run = top10(ids, data["doc_ids"], q_ids)
                    ndcg = float(np.mean(list(V.ndcg10(run, qrels, q_ids).values())))
                    e2e = enc_ms[n] + lat
                    rows.append({"encoder": n, "quant": qname, "hnsw_ef": ef,
                                 "oversampling": os, "ann_ndcg10": ndcg,
                                 "ann_ndcg_loss_rel": 1 - ndcg / exact[n]["ndcg10"],
                                 "ann_recovery_at10": recovery(run, exact[n]["run"]),
                                 "search_p50_ms": float(np.median(lat)),
                                 "search_p95_ms": float(np.quantile(lat, .95)),
                                 "e2e_p50_ms": float(np.median(e2e)),
                                 "e2e_p95_ms": float(np.quantile(e2e, .95))})
                    print(f"  {qname} ef={ef} os={os} {n}: nDCG {ndcg:.4f} "
                          f"rec {rows[-1]['ann_recovery_at10']:.3f} "
                          f"p50 {rows[-1]['search_p50_ms']:.2f} ms", flush=True)
            client.delete_collection(name)
        fused = fused_rows(client, dataset, data, dv, qv, enc_ms)
    finally:
        proc.terminate()
        proc.wait(timeout=60)
    decision = {n: decide([r for r in rows if r["encoder"] == n], exact[n]["ndcg10"])
                for n in ENCODERS}
    survives = {}
    for e in TARGETS:
        z, nn = decision["zero"][str(e)], decision["nano"][str(e)]
        survives[str(e)] = None if not (z and nn) else {
            "zero_e2e_p50_ms": z["e2e_p50_ms"], "nano_e2e_p50_ms": nn["e2e_p50_ms"],
            "nano_over_zero": nn["e2e_p50_ms"] / z["e2e_p50_ms"],
            "saving_survives": z["e2e_p50_ms"] < nn["e2e_p50_ms"]}
    write_result(out_path, {
        "status": "COMPLETE", "measurement": "E2", "dataset": dataset,
        "scope": "ANN and latency are deployment behavior; quality numbers are the exact rows",
        "data": prov, "n_queries": len(q_ids), "n_docs": len(data["doc_ids"]),
        "qdrant": {"version": "1.19.1", "binary_sha256": sha_file(QDRANT / "qdrant"),
                   "hnsw": {"m": 16, "ef_construct": 100}, "builds": builds},
        "exact": {n: exact[n]["ndcg10"] for n in ENCODERS}, "qdrant_exact_check": exact_check,
        "encode_ms": {n: {"p50": float(np.median(enc_ms[n])),
                          "p95": float(np.quantile(enc_ms[n], .95))} for n in ENCODERS},
        "geometry": geometry, "rows": rows, "decision": decision, "saving_survives": survives,
        "fused": fused,
        "receipt": receipt(__file__, [prov], started, ("qdrant-client", "onnxruntime"))})


def fused_rows(client, dataset, data, dv, qv, enc_ms):
    """Zero/Nano + Qdrant server-side BM25, DBSF over two prefetches of 100 (unquantized)."""
    from qdrant_client import models as m
    name = f"{dataset}-fused"
    avg_len = float(np.mean([len(t.split()) for t in data["doc_texts"]]))
    build_info = build(client, name, dv, texts=data["doc_texts"], avg_len=avg_len)
    out = {"avg_len": avg_len, "build": build_info, "hnsw_ef": FUSED_EF}
    q_ids, qrels = data["q_ids"], data["qrels"]

    def run_queries(make_query):
        for i in range(min(WARMUP, len(q_ids))):
            make_query(i)
        ids, lat = [], []
        for i in range(len(q_ids)):
            s = time.perf_counter()
            r = make_query(i)
            lat.append((time.perf_counter() - s) * 1000)
            ids.append([p.id for p in r.points])
        run = top10(ids, data["doc_ids"], q_ids)
        return float(np.mean(list(V.ndcg10(run, qrels, q_ids).values()))), np.array(lat)

    bm25 = lambda i: m.Document(text=data["q_texts"][i], model="qdrant/bm25",
                                options={"avg_len": avg_len})
    ndcg, lat = run_queries(lambda i: client.query_points(name, query=bm25(i), using="bm25",
                                                         limit=11))
    out["qdrant_bm25"] = {"ndcg10": ndcg, "p50_ms": float(np.median(lat))}
    for n in ("zero", "nano"):
        params = m.SearchParams(hnsw_ef=FUSED_EF)
        ndcg, lat = run_queries(lambda i: client.query_points(
            name, prefetch=[m.Prefetch(query=qv[n][i].tolist(), using="dense", limit=100,
                                       params=params),
                            m.Prefetch(query=bm25(i), using="bm25", limit=100)],
            query=m.FusionQuery(fusion=m.Fusion.DBSF), limit=11))
        e2e = enc_ms[n] + lat
        out[f"{n}+qdrant_bm25_dbsf@100"] = {"ndcg10": ndcg, "search_p50_ms": float(np.median(lat)),
                                            "e2e_p50_ms": float(np.median(e2e)),
                                            "e2e_p95_ms": float(np.quantile(e2e, .95))}
        print(f"  fused {n}: nDCG {ndcg:.4f} e2e p50 {np.median(e2e):.2f} ms", flush=True)
    client.delete_collection(name)
    return out


if __name__ == "__main__":
    main(sys.argv[1], smoke="--smoke" in sys.argv)
