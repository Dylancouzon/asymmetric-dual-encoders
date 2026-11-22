"""Exploratory matched-request benchmark on E2's million-passage diagnostic.

Run from the repository root with the M15 virtual environment:
    .venv/bin/python m15src/bm25_zero_msmarco1m.py [--smoke]

Both paths use one local gRPC client, sequential requests, and the same passages
and judged queries. Each clock runs from input text to ranked point IDs.
"""
import argparse
from contextlib import closing, contextmanager
import csv
import json
import os
import socket
import subprocess
import tempfile
import time
import urllib.request

import numpy as np
from qdrant_client import QdrantClient, models as M

import e2_ann as E2
import encoders15
import vectors15
from common import M20_SCORES, REPO, receipt, sha_file, utc_now, write_result


ORIGINAL = REPO / "results/m15_e2_ann_msmarco1m.json"
OUTPUT = REPO / "results/m15_bm25_zero_msmarco1m.json"
WORK = REPO / "work/m15"
ENGINE = WORK / "qdrant/qdrant"
ZERO = "zero_binary1"
BM25 = "qdrant_bm25"
LIMIT = 11
BATCH_SIZE = 1024
REPEATS = 3


def load_cached_inputs():
    """Read the pinned public snapshots directly so the run stays offline."""
    from huggingface_hub import hf_hub_download
    from pyarrow.parquet import ParquetFile, read_table

    reference = json.loads((M20_SCORES / "msmarco/zero-dense.json").read_text())
    pins = {key: reference[key] for key in ("source", "revision", "qrels_source",
                                           "qrels_revision", "split")}

    def cached(source, revision, filename):
        return hf_hub_download(source, filename, repo_type="dataset", revision=revision,
                               local_files_only=True)

    labels_path = cached(pins["qrels_source"], pins["qrels_revision"], "dev.tsv")
    qrels = {}
    with open(labels_path, encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            qrels.setdefault(row["query-id"], {})[row["corpus-id"]] = int(row["score"])
    queries = read_table(cached(pins["source"], pins["revision"],
                                "queries/queries-00000-of-00001.parquet"))
    judged = [(str(qid), text) for qid, text in zip(queries["_id"].to_pylist(),
                                                   queries["text"].to_pylist())
              if str(qid) in qrels]
    q_ids, q_texts = map(list, zip(*judged))
    if set(q_ids) != set(reference["scores"]):
        raise RuntimeError("Judged query IDs differ from the pinned M20 row")
    ids_path = WORK / "msmarco1m_ids.json"
    ids = json.loads(ids_path.read_text())
    if (ids["corpus_revision"], ids["qrels_revision"]) != (pins["revision"], pins["qrels_revision"]):
        raise RuntimeError("Passage and query revisions disagree")
    corpus = ParquetFile(cached(pins["source"], pins["revision"],
                                "corpus/corpus-00000-of-00001.parquet"))
    rows = np.asarray(ids["corpus_rows"])
    doc_texts = []
    offset = 0
    for batch in corpus.iter_batches(batch_size=80_000, columns=["_id", "title", "text"]):
        low, high = np.searchsorted(rows, [offset, offset + batch.num_rows])
        picked = batch.take(rows[low:high] - offset)
        if picked["_id"].to_pylist() != ids["doc_ids"][low:high]:
            raise RuntimeError("Corpus rows do not map to the recorded passage IDs")
        doc_texts.extend(f"{(title or '').strip()} {(text or '').strip()}".strip()
                         for title, text in zip(picked["title"].to_pylist(),
                                                picked["text"].to_pylist()))
        offset += batch.num_rows
    shards = sorted((WORK / "vecs/msmarco1m").glob("shard_*.npy"))
    vectors = [np.load(path, mmap_mode="r") for path in shards]
    data = {"q_ids": q_ids, "q_texts": q_texts, "qrels": qrels, "doc_ids": ids["doc_ids"],
            "doc_texts": doc_texts, "pins": pins}
    source = {"ids": str(ids_path.relative_to(REPO)), "ids_sha256": sha_file(ids_path),
              "n_positives": ids["n_positives"], "vectors": "work/m15/vecs/msmarco1m (Mac MPS)",
              "vector_shards": len(shards)}
    return data, vectors, source


@contextmanager
def engine(log_path):
    with tempfile.TemporaryDirectory(prefix="bm25-zero-", dir=WORK) as temp:
        with socket.socket() as http, socket.socket() as grpc:
            for sock in (http, grpc):
                sock.bind(("127.0.0.1", 0))
            http_port, grpc_port = http.getsockname()[1], grpc.getsockname()[1]
        env = dict(os.environ, QDRANT__SERVICE__HOST="127.0.0.1",
                   QDRANT__SERVICE__HTTP_PORT=str(http_port),
                   QDRANT__SERVICE__GRPC_PORT=str(grpc_port),
                   QDRANT__STORAGE__STORAGE_PATH=f"{temp}/storage",
                   QDRANT__TELEMETRY_DISABLED="true")
        with log_path.open("x") as log:
            proc = subprocess.Popen([str(ENGINE)], cwd=temp, env=env, stdout=log, stderr=log)
            try:
                for _ in range(120):
                    if proc.poll() is not None:
                        raise RuntimeError(f"Qdrant exited with {proc.returncode}; see {log_path}")
                    try:
                        with urllib.request.urlopen(f"http://127.0.0.1:{http_port}/readyz",
                                                    timeout=1):
                            break
                    except OSError:
                        time.sleep(0.5)
                else:
                    raise TimeoutError(f"Qdrant failed readiness; see {log_path}")
                with closing(QdrantClient(host="127.0.0.1", port=http_port, grpc_port=grpc_port,
                                          prefer_grpc=True, timeout=180)) as client:
                    yield client
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()


def build(client, name, data, vectors, avg_len):
    dense = name == ZERO
    client.create_collection(
        name,
        vectors_config={"dense": M.VectorParams(size=vectors[0].shape[1], distance=M.Distance.COSINE)}
        if dense else {},
        sparse_vectors_config=None if dense else {"bm25": M.SparseVectorParams(modifier=M.Modifier.IDF)},
        hnsw_config=M.HnswConfigDiff(m=16, ef_construct=100) if dense else None,
        optimizers_config=M.OptimizersConfigDiff(indexing_threshold=1),
        quantization_config=E2.quant_config("binary1") if dense else None,
    )
    started = time.monotonic()
    n_docs = len(data["doc_ids"])
    def dense_points():
        for shard in vectors:
            for row in shard:
                yield {"dense": row.astype(np.float32).tolist()}

    points = dense_points() if dense else (
        {"bm25": M.Document(text=text, model="qdrant/bm25", options={"avg_len": avg_len})}
        for text in data["doc_texts"])
    from itertools import batched
    for index, batch in enumerate(batched(points, BATCH_SIZE)):
        client.upsert(name, [M.PointStruct(id=point_id, vector=vector)
                            for point_id, vector in enumerate(batch, index * BATCH_SIZE)], wait=False)
        if index % 100 == 0:
            print(f"{name}: uploaded {index * BATCH_SIZE + len(batch):,}/{n_docs:,}", flush=True)
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        info = client.get_collection(name)
        if info.status == M.CollectionStatus.GREEN and info.points_count == n_docs and (
            not dense or info.indexed_vectors_count >= n_docs
        ):
            build_info = {"seconds_to_green": time.monotonic() - started,
                          "points": info.points_count, "indexed_vectors": info.indexed_vectors_count,
                          "segments": info.segments_count, "config": info.config.model_dump(mode="json")}
            print(f"{name}: ready in {build_info['seconds_to_green']:.1f}s", flush=True)
            return build_info
        time.sleep(2)
    raise TimeoutError(f"{name} failed to index {n_docs:,} points: {info}")


def measure(client, data, zero, avg_len, repeats):
    binary = M.SearchParams(hnsw_ef=512, quantization=M.QuantizationSearchParams(
        rescore=True, oversampling=2.0))

    def query(name, text):
        if name == ZERO:
            result = client.query_points(name, query=zero.encode([text])[0].tolist(), using="dense",
                                         search_params=binary, limit=LIMIT, with_payload=False)
        else:
            document = M.Document(text=text, model="qdrant/bm25", options={"avg_len": avg_len})
            result = client.query_points(name, query=document, using="bm25", limit=LIMIT,
                                         with_payload=False)
        return [point.id for point in result.points]

    series = {name: [] for name in (ZERO, BM25)}
    for rep in range(repeats):
        for text in data["q_texts"][:E2.WARMUP]:
            for name in series:
                query(name, text)
        indices = {name: [] for name in series}
        latencies = {name: [] for name in series}
        for i, text in enumerate(data["q_texts"]):
            order = (ZERO, BM25) if (i + rep) % 2 == 0 else (BM25, ZERO)
            for name in order:
                start = time.perf_counter_ns()
                ranked_ids = query(name, text)
                elapsed = (time.perf_counter_ns() - start) / 1e6
                latencies[name].append(elapsed)
                indices[name].append(ranked_ids)
        for name, runs in series.items():
            run = E2.top10(indices[name], data["doc_ids"], data["q_ids"])
            ndcg = np.mean(list(vectors15.ndcg10(run, data["qrels"], data["q_ids"]).values()))
            runs.append({"ndcg10": float(ndcg), "p50_ms": float(np.median(latencies[name])),
                         "p95_ms": float(np.quantile(latencies[name], 0.95))})
            print(f"pass {rep + 1} {name}: {runs[-1]}", flush=True)
    return {name: {**{key: float(np.median([row[key] for row in rows])) for key in rows[0]},
                   "passes": rows} for name, rows in series.items()}


def main(smoke):
    started = utc_now()
    output = WORK / "bm25_zero_msmarco1m_smoke.json" if smoke else OUTPUT
    if output.exists():
        raise FileExistsError(output)
    original = json.loads(ORIGINAL.read_text())
    if sha_file(ENGINE) != original["qdrant"]["binary_sha256"]:
        raise RuntimeError("Qdrant binary does not match the original E2 benchmark")
    data, vectors, source = load_cached_inputs()
    n_docs = sum(len(shard) for shard in vectors)
    if source["ids_sha256"] != original["data"]["ids_sha256"] or (
        len(data["q_ids"]) != original["n_queries"] or len(data["doc_texts"]) != len(data["doc_ids"])
        or n_docs != original["n_docs"]
    ):
        raise RuntimeError("Public benchmark inputs do not match the E2 receipt")
    avg_len = original["fused"]["avg_len"]
    if not np.isclose(sum(len(t.split()) for t in data["doc_texts"]) / n_docs, avg_len,
                      rtol=0, atol=1e-12):
        raise RuntimeError("Document length disagrees with the original BM25 receipt")
    if smoke:
        for key in ("q_ids", "q_texts"):
            data[key] = data[key][:150]
        for key in ("doc_ids", "doc_texts"):
            data[key] = data[key][:20_000]
        vectors = [vectors[0][:20_000]]
    zero = encoders15.make("zero")
    repeats = 1 if smoke else REPEATS
    started_tag = started.replace(":", "").replace("-", "")
    with engine(WORK / f"bm25-zero-{started_tag}.log") as client:
        builds = {name: build(client, name, data, vectors, avg_len) for name in (ZERO, BM25)}
        measured = measure(client, data, zero, avg_len, repeats)
    zero_freeze = json.loads((REPO / "m7/FREEZE.json").read_text())["table_sha256"]
    run_receipt = receipt(__file__, [source], started, ("qdrant-client", "pyarrow", "tokenizers"))
    run_receipt["method"] = "Text-to-ranked-IDs measurement defined in the benchmark script"
    run_receipt.pop("method_sha256")
    write_result(output, {
        "status": "COMPLETE", "analysis": "exploratory", "smoke": smoke, "dataset": "msmarco1m",
        "n_queries": len(data["q_ids"]), "n_docs": len(data["doc_ids"]), "inputs": data["pins"],
        "qdrant_version": original["qdrant"]["version"], "qdrant_binary_sha256": sha_file(ENGINE),
        "original_e2_receipt_sha256": sha_file(ORIGINAL), "zero_int8_table_sha256": zero_freeze,
        "bm25_avg_doc_length": avg_len, "top_k_requested": LIMIT, "top_k_scored": 10,
        "zero_search": {"hnsw_ef": 512, "oversampling": 2.0, "rescore": True},
        "protocol": {"clock": "text input to ranked IDs, single sequential gRPC client",
                     "order": "alternating which system sees each query first, flipped each pass",
                     "warmup_queries_per_path_per_pass": E2.WARMUP, "passes": repeats,
                     "summary": "median across pass-level statistics"},
        "indexes": builds, "results": measured, "receipt": run_receipt,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    main(parser.parse_args().smoke)
