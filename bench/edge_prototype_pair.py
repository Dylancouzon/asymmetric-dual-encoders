"""Qdrant Edge prototype for the M9 PAIR: two query paths, one document index.

  zero: tokenize -> gather rows from the int8 token->vector table -> average -> normalize -> ANN
  nano: tokenize -> ONNX forward (backbone + head, already normalized) -> ANN

Latency/architecture only (m9/EDGE_PROTOTYPE_MAC.md). Vectors are SYNTHETIC:
  * docs: 1M random unit vectors, 1024d, fp16 on-disk -- a realistic edge shard size. HNSW search
    time is a function of n/dim/ef, not of what the vectors mean.
  * zero's table: 30,522 x 1024 int8 + one global scale, matching the released artifact's shape
    (m7/FREEZE.json) -- the real table.npz is not on this machine.
Recall is NOT measured here; that needs the real stella index and runs on the training box.

  python bench/edge_prototype_pair.py build     # docs shard + token_table shard (idempotent)
  python bench/edge_prototype_pair.py measure   # zero vs nano, ef sweep, per-length-bucket timing
"""
import json
import platform
import random
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from qdrant_edge import (Distance, EdgeConfig, EdgeShard, EdgeVectorParams, HnswIndexConfig, Point,
                          Query, QueryRequest, SearchParams, UpdateOperation, VectorStorageDatatype)

from core import REPO

EDGE_DIR = REPO / "edge_data" / "m9_pair"
DOCS_SHARD = str(EDGE_DIR / "docs_synth")
TABLE_SHARD = str(EDGE_DIR / "token_table_synth")
NANO_DIR = REPO / "work" / "m9onnx" / "nano-minilm-l6"
NANO_ONNX = NANO_DIR / "model_fp16.onnx"

N_DOCS = 1_000_000
DIM = 1024
VOCAB = 30522
TABLE_SCALE = 0.02  # arbitrary; latency of dequant/gather doesn't depend on its value
BUCKETS = ((1, 5), (6, 10), (11, 20), (21, 50), (51, 120))
EF_SWEEP = (None, 128, 512)
WORDS = ("alpha beta gamma delta epsilon zeta eta theta kappa lambda retrieval embedding vector "
         "index document query passage neural token model ranking corpus semantic").split()


def synth_texts(nwords, n, seed=0):
    r = random.Random(seed)
    return [" ".join(r.choice(WORDS) for _ in range(nwords)) for _ in range(n)]


def du_mb(p):
    return round(sum(f.stat().st_size for f in Path(p).rglob("*") if f.is_file()) / 1e6, 1)


def host():
    info = {"platform": platform.platform(), "python": platform.python_version()}
    info["cpu"] = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True).stdout.strip()
    info["cores"] = subprocess.run(["sysctl", "-n", "hw.ncpu"], capture_output=True, text=True).stdout.strip()
    return info


def build():
    rng = np.random.default_rng(0)
    Path(EDGE_DIR).mkdir(parents=True, exist_ok=True)

    if not Path(DOCS_SHARD, "config.json").exists():
        Path(DOCS_SHARD).mkdir(parents=True, exist_ok=True)
        p = EdgeVectorParams(size=DIM, distance=Distance.Dot, datatype=VectorStorageDatatype.Float16, on_disk=True)
        shard = EdgeShard.create(DOCS_SHARD, EdgeConfig(vectors={"dense": p}))
        t0 = time.time()
        for start in range(0, N_DOCS, 8192):
            n = min(8192, N_DOCS - start)
            v = rng.standard_normal((n, DIM)).astype(np.float32)
            v /= np.linalg.norm(v, axis=1, keepdims=True)
            pts = [Point(id=start + i, vector={"dense": v[i].tolist()}, payload=None) for i in range(n)]
            shard.update(UpdateOperation.upsert_points(pts))
            if start % 200_000 == 0:
                print(f"docs_synth: {start}/{N_DOCS} ({time.time()-t0:.0f}s)", flush=True)
        shard.optimize()
        shard.flush()
        shard.close()
        print(f"docs_synth: {N_DOCS} points, optimized, {time.time()-t0:.0f}s total", flush=True)
    else:
        print("docs_synth: already built", flush=True)

    if not Path(TABLE_SHARD, "config.json").exists():
        # retrieve-only shard: no HNSW graph needed (m=0), matches the M5 finding that a default
        # HNSW index bloats a lookup table that is never ANN-searched (1.82 GB vs 466 MB raw).
        Path(TABLE_SHARD).mkdir(parents=True, exist_ok=True)
        hnsw = HnswIndexConfig(m=0, ef_construct=0, full_scan_threshold=1)
        p = EdgeVectorParams(size=DIM, distance=Distance.Dot, datatype=VectorStorageDatatype.Uint8,
                              on_disk=True, hnsw_config=hnsw)
        shard = EdgeShard.create(TABLE_SHARD, EdgeConfig(vectors={"tok": p}))
        t0 = time.time()
        for start in range(0, VOCAB, 4096):
            n = min(4096, VOCAB - start)
            raw = rng.integers(0, 256, size=(n, DIM), dtype=np.uint8)
            pts = [Point(id=start + i, vector={"tok": raw[i].tolist()}, payload=None) for i in range(n)]
            shard.update(UpdateOperation.upsert_points(pts))
        shard.optimize()
        shard.flush()
        shard.close()
        (EDGE_DIR / "table_scale.json").write_text(json.dumps({"scale": TABLE_SCALE, "zero_point": 128}))
        print(f"token_table_synth: {VOCAB} points, {time.time()-t0:.0f}s", flush=True)
    else:
        print("token_table_synth: already built", flush=True)


def zero_encode(table_shard, tok, ids, scale):
    recs = table_shard.retrieve(point_ids=list(set(ids)), with_payload=False, with_vector=True)
    by_id = {r.id: (np.array(r.vector["tok"], dtype=np.float32) - 128.0) * scale for r in recs}
    v = np.mean([by_id[i] for i in ids if i in by_id], axis=0)
    return v / (np.linalg.norm(v) + 1e-12)


def nano_encode(sess, tok, text):
    b = tok([text], return_tensors="np", truncation=True, max_length=512)
    out = sess.run(None, {"input_ids": b["input_ids"].astype("int64"),
                          "attention_mask": b["attention_mask"].astype("int64")})[0]
    return out[0].astype(np.float32)  # already head-projected + L2-normalized by the export graph


def measure_path(name, encode_fn, docs_shard, texts, threads):
    out = {}
    for ef in EF_SWEEP:
        params = SearchParams(hnsw_ef=ef) if ef else None
        enc_lat, search_lat = [], []
        for t in texts:
            t0 = time.perf_counter()
            v = encode_fn(t)
            t1 = time.perf_counter()
            kw = {"params": params} if params else {}
            docs_shard.query(QueryRequest(query=Query.Nearest(v.tolist(), using="dense"),
                                          limit=100, with_payload=False, with_vector=False, **kw))
            t2 = time.perf_counter()
            enc_lat.append((t1 - t0) * 1000)
            search_lat.append((t2 - t1) * 1000)
        out[f"ef={ef or 'default'}"] = {
            "encode_ms_p50": round(statistics.median(enc_lat), 3),
            "encode_ms_p95": round(sorted(enc_lat)[int(0.95 * len(enc_lat)) - 1], 3),
            "search_ms_p50": round(statistics.median(search_lat), 3),
            "search_ms_p95": round(sorted(search_lat)[int(0.95 * len(search_lat)) - 1], 3),
            "total_ms_p50": round(statistics.median([a + b for a, b in zip(enc_lat, search_lat)]), 3),
        }
    return out


def measure():
    from transformers import AutoTokenizer

    threads = 4
    out = {"_what": "M9 pair edge prototype: zero vs nano query paths, one synthetic doc index. "
                    "Latency/architecture only -- see file docstring.",
           "host": host(), "threads": threads, "n_docs": N_DOCS, "dim": DIM}

    t0 = time.perf_counter()
    docs_shard = EdgeShard.load(DOCS_SHARD)
    out["docs_shard_load_s"] = round(time.perf_counter() - t0, 3)
    out["docs_shard_mb"] = du_mb(DOCS_SHARD)

    scale = json.loads((EDGE_DIR / "table_scale.json").read_text())["scale"]
    t0 = time.perf_counter()
    zero_tok = AutoTokenizer.from_pretrained("NovaSearch/stella_en_400M_v5",
                                             revision="ffeb2b7ee715c226d4ffe5e4619f7dbb48624c20")
    table_shard = EdgeShard.load(TABLE_SHARD)
    out["zero_load_s"] = round(time.perf_counter() - t0, 3)
    out["table_shard_mb"] = du_mb(TABLE_SHARD)

    t0 = time.perf_counter()
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(NANO_ONNX), so, providers=["CPUExecutionProvider"])
    nano_tok = AutoTokenizer.from_pretrained(str(NANO_DIR))
    out["nano_load_s"] = round(time.perf_counter() - t0, 3)
    out["nano_onnx_mb"] = round(NANO_ONNX.stat().st_size / 1e6, 1)

    out["buckets"] = {}
    for lo, hi in BUCKETS:
        texts = synth_texts((lo + hi) // 2, 60)
        peak0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        def zero_fn(t, _tok=zero_tok, _shard=table_shard, _scale=scale):
            ids = _tok(t, add_special_tokens=True, truncation=True, max_length=512)["input_ids"]
            return zero_encode(_shard, _tok, ids, _scale)

        def nano_fn(t, _sess=sess, _tok=nano_tok):
            return nano_encode(_sess, _tok, t)

        bucket_key = f"{lo}-{hi}w"
        out["buckets"][bucket_key] = {
            "zero": measure_path("zero", zero_fn, docs_shard, texts, threads),
            "nano": measure_path("nano", nano_fn, docs_shard, texts, threads),
        }
        print(f"{bucket_key}: zero default {out['buckets'][bucket_key]['zero']['ef=default']} "
              f"nano default {out['buckets'][bucket_key]['nano']['ef=default']}", flush=True)

    out["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 1)
    docs_shard.close()
    table_shard.close()

    tag = out["host"]["cpu"].replace(" ", "_")
    dest = REPO / "results" / f"m9_edge_prototype_{tag}.json"
    dest.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {dest.relative_to(REPO)}")


if __name__ == "__main__":
    {"build": build, "measure": measure}[sys.argv[1]]()
