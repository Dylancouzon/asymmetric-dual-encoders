"""Qdrant Edge prototype for the M9 PAIR: two query paths, one document index.

  zero: tokenize -> gather rows from the int8 token->vector table -> average -> normalize -> ANN
  nano: tokenize -> ONNX forward (backbone + head, already normalized) -> ANN

Latency/architecture only (m9/EDGE_PROTOTYPE_MAC.md). Vectors are SYNTHETIC:
  * docs: 1M random unit vectors, 1024d -- a realistic edge shard size. HNSW search time is a
    function of n/dim/ef/quantization, not of what the vectors mean.
  * zero's table: 30,522 x 1024 int8 + one global scale, matching the released artifact's shape
    (m7/FREEZE.json) -- the real table.npz is not on this machine.
Recall is NOT measured here; that needs the real stella index and runs on the training box.

Round 2 (m9/EDGE_PROTOTYPE_MAC.md): fixed the round-1 cold-start artifact with an explicit
warm-up pass per (path, ef, doc config), and swept the DOCUMENT index -- fp16 / scalar-int8 /
binary, each with the original vectors mmap'd (on_disk) or fully in RAM -- since round 1 found
the doc index is 8-48x the size of either query-side asset and nobody had quantized it yet.

  python bench/edge_prototype_pair.py build     # token_table + all 6 doc-index configs (idempotent)
  python bench/edge_prototype_pair.py measure   # zero vs nano, ef sweep, per doc-index config
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
from qdrant_edge import (BinaryQuantizationConfig, Distance, EdgeConfig, EdgeShard,
                          EdgeVectorParams, HnswIndexConfig, Point, Query, QueryRequest,
                          ScalarQuantizationConfig, ScalarType, SearchParams, UpdateOperation,
                          VectorStorageDatatype)

from core import REPO

EDGE_DIR = REPO / "edge_data" / "m9_pair"
TABLE_SHARD = str(EDGE_DIR / "token_table_synth")
NANO_DIR = REPO / "work" / "m9onnx" / "nano-minilm-l6"
NANO_ONNX = NANO_DIR / "model_fp16.onnx"

N_DOCS = 1_000_000
DIM = 1024
VOCAB = 30522
TABLE_SCALE = 0.02  # arbitrary; latency of dequant/gather doesn't depend on its value
BUCKETS = ((1, 5), (6, 10), (11, 20), (21, 50), (51, 120))
EF_SWEEP = (None, 128, 512)
WARMUP_SEARCHES = 200
N_TIMED = 60
WORDS = ("alpha beta gamma delta epsilon zeta eta theta kappa lambda retrieval embedding vector "
         "index document query passage neural token model ranking corpus semantic").split()

# name -> (quantization_config, on_disk for the ORIGINAL vectors). Quantized data is always_ram
# (it's what search actually touches); on_disk toggles whether the original fp16 vectors are
# mmap'd or fully resident, which is the RAM-vs-disk knob an edge device actually cares about.
DOC_CONFIGS = {
    "fp16_mmap": (None, True),
    "fp16_ram": (None, False),
    "int8_mmap": (ScalarQuantizationConfig(type=ScalarType.Int8, always_ram=True), True),
    "int8_ram": (ScalarQuantizationConfig(type=ScalarType.Int8, always_ram=True), False),
    "binary_mmap": (BinaryQuantizationConfig(always_ram=True), True),
    "binary_ram": (BinaryQuantizationConfig(always_ram=True), False),
}


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


def docs_shard_path(name):
    return str(EDGE_DIR / f"docs_{name}")


def build_docs(name, quant, on_disk):
    path = docs_shard_path(name)
    if Path(path, "config.json").exists():
        print(f"docs_{name}: already built", flush=True)
        return
    Path(path).mkdir(parents=True, exist_ok=True)
    p = EdgeVectorParams(size=DIM, distance=Distance.Dot, datatype=VectorStorageDatatype.Float16,
                          on_disk=on_disk, quantization_config=quant)
    shard = EdgeShard.create(path, EdgeConfig(vectors={"dense": p}))
    rng = np.random.default_rng(0)  # same seed every config -> byte-identical docs across the sweep
    t0 = time.time()
    for start in range(0, N_DOCS, 8192):
        n = min(8192, N_DOCS - start)
        v = rng.standard_normal((n, DIM)).astype(np.float32)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        pts = [Point(id=start + i, vector={"dense": v[i].tolist()}, payload=None) for i in range(n)]
        shard.update(UpdateOperation.upsert_points(pts))
        if start % 400_000 == 0:
            print(f"docs_{name}: {start}/{N_DOCS} ({time.time()-t0:.0f}s)", flush=True)
    shard.optimize()
    shard.flush()
    shard.close()
    print(f"docs_{name}: {N_DOCS} points, optimized, {time.time()-t0:.0f}s total", flush=True)


def build():
    Path(EDGE_DIR).mkdir(parents=True, exist_ok=True)
    for name, (quant, on_disk) in DOC_CONFIGS.items():
        build_docs(name, quant, on_disk)

    if not Path(TABLE_SHARD, "config.json").exists():
        # retrieve-only shard: no HNSW graph needed (m=0), matches the M5 finding that a default
        # HNSW index bloats a lookup table that is never ANN-searched (1.82 GB vs 466 MB raw).
        Path(TABLE_SHARD).mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(0)
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


def zero_encode(table_shard, ids, scale):
    recs = table_shard.retrieve(point_ids=list(set(ids)), with_payload=False, with_vector=True)
    by_id = {r.id: (np.array(r.vector["tok"], dtype=np.float32) - 128.0) * scale for r in recs}
    v = np.mean([by_id[i] for i in ids if i in by_id], axis=0)
    return v / (np.linalg.norm(v) + 1e-12)


def nano_encode(sess, tok, text):
    b = tok([text], return_tensors="np", truncation=True, max_length=512)
    out = sess.run(None, {"input_ids": b["input_ids"].astype("int64"),
                          "attention_mask": b["attention_mask"].astype("int64")})[0]
    return out[0].astype(np.float32)  # already head-projected + L2-normalized by the export graph


def timed_search(docs_shard, v, params):
    kw = {"params": params} if params else {}
    t0 = time.perf_counter()
    docs_shard.query(QueryRequest(query=Query.Nearest(v.tolist(), using="dense"),
                                  limit=100, with_payload=False, with_vector=False, **kw))
    return (time.perf_counter() - t0) * 1000


def measure_path(encode_fn, docs_shard, bucket_texts, warmup_seed):
    """Per ef: warm up WARMUP_SEARCHES times (discarded), then time N_TIMED queries per bucket,
    with bucket order randomized so a residual warming effect can't land in one bucket again."""
    out = {b: {} for b in bucket_texts}
    bucket_order_log = {}
    for ef in EF_SWEEP:
        ef_key = f"ef={ef or 'default'}"
        params = SearchParams(hnsw_ef=ef) if ef else None

        warm_texts = synth_texts(8, WARMUP_SEARCHES, seed=warmup_seed)
        for t in warm_texts:
            v = encode_fn(t)
            timed_search(docs_shard, v, params)

        order = list(bucket_texts)
        random.Random(warmup_seed * 1000 + (ef or 0)).shuffle(order)
        bucket_order_log[ef_key] = order
        for bkey in order:
            enc_lat, search_lat = [], []
            for t in bucket_texts[bkey]:
                t0 = time.perf_counter()
                v = encode_fn(t)
                t1 = time.perf_counter()
                search_lat.append(timed_search(docs_shard, v, params))
                enc_lat.append((t1 - t0) * 1000)
            out[bkey][ef_key] = {
                "encode_ms_p50": round(statistics.median(enc_lat), 3),
                "encode_ms_p95": round(sorted(enc_lat)[int(0.95 * len(enc_lat)) - 1], 3),
                "search_ms_p50": round(statistics.median(search_lat), 3),
                "search_ms_p95": round(sorted(search_lat)[int(0.95 * len(search_lat)) - 1], 3),
                "total_ms_p50": round(statistics.median([a + b for a, b in zip(enc_lat, search_lat)]), 3),
            }
    return out, bucket_order_log


def measure():
    from transformers import AutoTokenizer

    threads = 4
    out = {"_what": "M9 pair edge prototype: zero vs nano query paths against a swept document "
                    "index (fp16/int8/binary x mmap/ram). Latency/architecture only, see docstring.",
           "round": 2, "warmup_searches": WARMUP_SEARCHES, "host": host(), "threads": threads,
           "n_docs": N_DOCS, "dim": DIM}

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

    def zero_fn(t, _tok=zero_tok, _shard=table_shard, _scale=scale):
        ids = _tok(t, add_special_tokens=True, truncation=True, max_length=512)["input_ids"]
        return zero_encode(_shard, ids, _scale)

    def nano_fn(t, _sess=sess, _tok=nano_tok):
        return nano_encode(_sess, _tok, t)

    bucket_texts = {f"{lo}-{hi}w": synth_texts((lo + hi) // 2, N_TIMED) for lo, hi in BUCKETS}

    out["configs"] = {}
    for name, (quant, on_disk) in DOC_CONFIGS.items():
        path = docs_shard_path(name)
        t0 = time.perf_counter()
        docs_shard = EdgeShard.load(path)
        load_s = round(time.perf_counter() - t0, 3)

        zero_buckets, zero_order = measure_path(zero_fn, docs_shard, bucket_texts, warmup_seed=1)
        nano_buckets, nano_order = measure_path(nano_fn, docs_shard, bucket_texts, warmup_seed=2)
        buckets = {b: {"zero": zero_buckets[b], "nano": nano_buckets[b]} for b in bucket_texts}

        cfg_out = {
            "quantization": "none" if quant is None else type(quant).__name__,
            "on_disk_original": on_disk,
            "load_s": load_s,
            "bytes_mb": du_mb(path),
            "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 1),
            "bucket_order": {"zero": zero_order, "nano": nano_order},
            "buckets": buckets,
        }
        out["configs"][name] = cfg_out
        docs_shard.close()
        d10 = buckets["6-10w"]
        print(f"{name}: bytes {cfg_out['bytes_mb']} MB · load {load_s}s · "
              f"zero default {d10['zero']['ef=default']} · nano default {d10['nano']['ef=default']}",
              flush=True)

    table_shard.close()

    tag = out["host"]["cpu"].replace(" ", "_")
    dest = REPO / "results" / f"m9_edge_prototype_{tag}.json"
    dest.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {dest.relative_to(REPO)}")


if __name__ == "__main__":
    {"build": build, "measure": measure}[sys.argv[1]]()
