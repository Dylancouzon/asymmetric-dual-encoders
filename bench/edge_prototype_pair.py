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
                          EdgeVectorParams, HnswIndexConfig, Point, Query, QuantizationSearchParams,
                          QueryRequest, ScalarQuantizationConfig, ScalarType, SearchParams,
                          UpdateOperation, VectorStorageDatatype)

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


def measure_path(encode_fn, docs_shard, bucket_texts, warmup_seed, ef_sweep=None, quant_params=None):
    """Per ef: warm up WARMUP_SEARCHES times (discarded), then time N_TIMED queries per bucket,
    with bucket order randomized so a residual warming effect can't land in one bucket again.
    quant_params (round 3): a QuantizationSearchParams applied at every ef, e.g. rescore=False."""
    ef_sweep = ef_sweep or EF_SWEEP
    out = {b: {} for b in bucket_texts}
    bucket_order_log = {}
    for ef in ef_sweep:
        ef_key = f"ef={ef or 'default'}"
        params = SearchParams(hnsw_ef=ef, quantization=quant_params) if (ef or quant_params) else None

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


ROUND3_EF_SWEEP = (None, 128)
# Configs already built with the round-3-correct storage shape: originals on_disk (mmap'd, not
# RAM), quantized copy always_ram (the hot path). fp16 has no quantized copy, included as baseline.
ROUND3_RUNS = [("fp16_mmap", False), ("int8_mmap", False), ("int8_mmap", True),
              ("binary_mmap", False), ("binary_mmap", True)]


def vector_bytes_split(path):
    """original (matrix.dat) vs quantized (quantized.data) bytes, summed across segments."""
    orig = sum(f.stat().st_size for f in Path(path).rglob("vector_storage-dense/matrix.dat"))
    quant = sum(f.stat().st_size for f in Path(path).rglob("vector_storage-dense/quantized.data"))
    return orig, quant


def measure_one():
    """Round 3: ONE doc-index config, measured in its OWN process, invoked by measure3() below.
    Round 2 measured all 6 configs in one process, so `ru_maxrss` (a high-water mark for the whole
    process, never reset) reported roughly the same ~7GB for every config -- whatever the first,
    heaviest config touched stayed in the number for every config measured after it. Isolating
    each config in its own process is what makes peak RSS mean anything per-config."""
    import argparse

    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--no-rescore", action="store_true")
    a = ap.parse_args(sys.argv[2:])

    threads = 4
    scale = json.loads((EDGE_DIR / "table_scale.json").read_text())["scale"]
    zero_tok = AutoTokenizer.from_pretrained("NovaSearch/stella_en_400M_v5",
                                             revision="ffeb2b7ee715c226d4ffe5e4619f7dbb48624c20")
    table_shard = EdgeShard.load(TABLE_SHARD)

    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(NANO_ONNX), so, providers=["CPUExecutionProvider"])
    nano_tok = AutoTokenizer.from_pretrained(str(NANO_DIR))

    def zero_fn(t, _tok=zero_tok, _shard=table_shard, _scale=scale):
        ids = _tok(t, add_special_tokens=True, truncation=True, max_length=512)["input_ids"]
        return zero_encode(_shard, ids, _scale)

    def nano_fn(t, _sess=sess, _tok=nano_tok):
        return nano_encode(_sess, _tok, t)

    path = docs_shard_path(a.name)
    t0 = time.perf_counter()
    docs_shard = EdgeShard.load(path)
    load_s = round(time.perf_counter() - t0, 3)
    rss_after_load_mb = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 1)

    bucket_texts = {f"{lo}-{hi}w": synth_texts((lo + hi) // 2, N_TIMED) for lo, hi in BUCKETS}
    qp = QuantizationSearchParams(rescore=False) if a.no_rescore else None
    zero_buckets, zero_order = measure_path(zero_fn, docs_shard, bucket_texts, warmup_seed=1,
                                            ef_sweep=ROUND3_EF_SWEEP, quant_params=qp)
    nano_buckets, nano_order = measure_path(nano_fn, docs_shard, bucket_texts, warmup_seed=2,
                                            ef_sweep=ROUND3_EF_SWEEP, quant_params=qp)
    buckets = {b: {"zero": zero_buckets[b], "nano": nano_buckets[b]} for b in bucket_texts}

    orig_bytes, quant_bytes = vector_bytes_split(path)
    out = {
        "load_s": load_s,
        "bytes_mb": du_mb(path),
        "orig_vector_bytes_mb": round(orig_bytes / 1e6, 1),
        "quantized_vector_bytes_mb": round(quant_bytes / 1e6, 1),
        "rss_after_load_mb": rss_after_load_mb,
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 1),
        "no_rescore": a.no_rescore,
        "bucket_order": {"zero": zero_order, "nano": nano_order},
        "buckets": buckets,
    }
    docs_shard.close()
    table_shard.close()
    print("RESULT_JSON:" + json.dumps(out))


def measure3():
    """Orchestrator: runs each ROUND3_RUNS entry in its own subprocess (see measure_one) and
    assembles the combined report. Writes round: 3, keeping rounds 1-2 recoverable from git."""
    out = {"_what": "M9 pair edge prototype ROUND 3: storage-configured quantization "
                    "(originals on_disk/mmap, quantized copy always_ram) with each config measured "
                    "in its own process for a clean peak-RSS reading, plus rescore=false rows for "
                    "int8/binary. Latency/architecture only, see file docstring.",
           "round": 3, "warmup_searches": WARMUP_SEARCHES, "host": host(), "threads": 4,
           "n_docs": N_DOCS, "dim": DIM, "table_shard_mb": du_mb(TABLE_SHARD),
           "nano_onnx_mb": round(NANO_ONNX.stat().st_size / 1e6, 1)}

    out["configs"] = {}
    for name, no_rescore in ROUND3_RUNS:
        key = name + ("_norescore" if no_rescore else "")
        cmd = [sys.executable, str(Path(__file__).resolve()), "measure_one", name]
        if no_rescore:
            cmd.append("--no-rescore")
        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"{key}: FAILED\n{proc.stderr[-4000:]}", flush=True)
            out["configs"][key] = {"error": proc.stderr[-2000:]}
            continue
        line = next(l for l in proc.stdout.splitlines() if l.startswith("RESULT_JSON:"))
        cfg_out = json.loads(line[len("RESULT_JSON:"):])
        out["configs"][key] = cfg_out
        d10 = cfg_out["buckets"]["6-10w"]
        print(f"{key}: rss_after_load {cfg_out['rss_after_load_mb']} MB · peak_rss {cfg_out['peak_rss_mb']} MB"
              f" · orig {cfg_out['orig_vector_bytes_mb']} MB · quant {cfg_out['quantized_vector_bytes_mb']} MB · "
              f"zero default {d10['zero']['ef=default']} · nano default {d10['nano']['ef=default']}"
              f" · ({time.time()-t0:.0f}s)", flush=True)

    tag = out["host"]["cpu"].replace(" ", "_")
    dest = REPO / "results" / f"m9_edge_prototype_{tag}.json"
    dest.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {dest.relative_to(REPO)}")


DOCKER_IMAGE = "qdrant/qdrant:v1.19.0"
DOCKER_STORAGE = {"binary": EDGE_DIR / "docker_storage_binary", "fp16": EDGE_DIR / "docker_storage_fp16"}
MEM_LIMITS = ("256m", "512m", "1g", "2g")
HTTP_PORT, GRPC_PORT = 16333, 16334


def _docker_run(name, storage_dir, mem_limit=None):
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    cmd = ["docker", "run", "-d", "--name", name]
    if mem_limit:
        cmd += ["-m", mem_limit]
    cmd += ["-p", f"{HTTP_PORT}:6333", "-p", f"{GRPC_PORT}:6334",
            "-v", f"{storage_dir}:/qdrant/storage", DOCKER_IMAGE]
    subprocess.run(cmd, capture_output=True)


def _wait_healthy(timeout_s=30):
    import urllib.request
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            with urllib.request.urlopen(f"http://localhost:{HTTP_PORT}/healthz", timeout=2) as r:
                if r.status == 200:
                    return round(time.time() - t0, 3)
        except Exception:
            pass
        time.sleep(0.5)
    return None


def _docker_stop_rm(name):
    subprocess.run(["docker", "stop", name], capture_output=True)
    subprocess.run(["docker", "rm", name], capture_output=True)


def _docker_mem_used_mb(name):
    """cgroup memory.current via `docker stats`, the constrained truth round 4 needs -- unlike RSS
    it can't include evictable page cache the container was never charged for."""
    out = subprocess.run(["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", name],
                         capture_output=True, text=True).stdout.strip()
    try:
        used = out.split("/")[0].strip()
        val = float(used[:-3])
        mult = {"KiB": 1 / 1024, "MiB": 1.0, "GiB": 1024.0}[used[-3:]]
        return round(val * mult, 1)
    except Exception:
        return None


def build_docker_collection(kind):
    """kind: 'binary' (originals on_disk, quantized copy always_ram, matches round 3's winning
    config) or 'fp16' (on_disk, unquantized -- the default-reach-for baseline). Real Qdrant, not
    qdrant_edge: round 4 needs an actual memory-capped container, which only a real server has."""
    from qdrant_client import QdrantClient, models

    storage_dir = DOCKER_STORAGE[kind]
    name = f"m9_r4_build_{kind}"
    _docker_run(name, storage_dir)  # no -m: generous memory for the one-time build/index
    load_s = _wait_healthy(60)
    print(f"build {kind}: container healthy after {load_s}s", flush=True)
    c = QdrantClient(url=f"http://localhost:{HTTP_PORT}", grpc_port=GRPC_PORT, prefer_grpc=True, timeout=120)
    if c.collection_exists("docs"):
        print(f"build {kind}: collection already built", flush=True)
    else:
        quant = (models.BinaryQuantization(binary=models.BinaryQuantizationConfig(always_ram=True))
                 if kind == "binary" else None)
        c.create_collection("docs", vectors_config=models.VectorParams(
            size=DIM, distance=models.Distance.DOT, on_disk=True, quantization_config=quant))
        rng = np.random.default_rng(0)  # same seed as every other round -> byte-identical docs
        t0 = time.time()
        for start in range(0, N_DOCS, 500):
            n = min(500, N_DOCS - start)
            v = rng.standard_normal((n, DIM)).astype(np.float32)
            v /= np.linalg.norm(v, axis=1, keepdims=True)
            c.upsert("docs", points=[models.PointStruct(id=start + i, vector=v[i].tolist()) for i in range(n)])
            if start % 200_000 == 0:
                print(f"build {kind}: {start}/{N_DOCS} ({time.time()-t0:.0f}s)", flush=True)
        while c.get_collection("docs").status != models.CollectionStatus.GREEN:
            time.sleep(2)
        print(f"build {kind}: {N_DOCS} points, green, {time.time()-t0:.0f}s total", flush=True)
    _docker_stop_rm(name)


def build4():
    build_docker_collection("binary")
    build_docker_collection("fp16")


def serve_one(kind, mem_limit):
    """Start a container at `mem_limit`, see if it serves the pre-built `docs` collection, and if
    so run round 3's warm-up + randomized-bucket-order sweep at ef=default only (round 4 keeps the
    sweep small; the point here is the memory limit, not another ef table)."""
    from qdrant_client import QdrantClient, models

    from transformers import AutoTokenizer
    import onnxruntime as ort

    name = f"m9_r4_serve_{kind}_{mem_limit}"
    _docker_run(name, DOCKER_STORAGE[kind], mem_limit=mem_limit)
    load_s = _wait_healthy(30)
    if load_s is None:
        inspect = subprocess.run(
            ["docker", "inspect", name, "--format", "OOMKilled={{.State.OOMKilled}} status={{.State.Status}} exit={{.State.ExitCode}}"],
            capture_output=True, text=True).stdout.strip()
        logs = subprocess.run(["docker", "logs", "--tail", "20", name], capture_output=True, text=True).stderr
        _docker_stop_rm(name)
        return {"served": False, "reason": inspect, "logs_tail": logs[-1500:]}

    c = QdrantClient(url=f"http://localhost:{HTTP_PORT}", grpc_port=GRPC_PORT, prefer_grpc=True, timeout=30)
    try:
        info = c.get_collection("docs")
        if info.points_count != N_DOCS:
            raise RuntimeError(f"points_count={info.points_count}, expected {N_DOCS}")
    except Exception as e:
        inspect = subprocess.run(
            ["docker", "inspect", name, "--format", "OOMKilled={{.State.OOMKilled}} status={{.State.Status}}"],
            capture_output=True, text=True).stdout.strip()
        _docker_stop_rm(name)
        return {"served": False, "reason": f"collection not ready after healthz: {e!r} / {inspect}"}

    mem_after_load_mb = _docker_mem_used_mb(name)

    scale = json.loads((EDGE_DIR / "table_scale.json").read_text())["scale"]
    zero_tok = AutoTokenizer.from_pretrained("NovaSearch/stella_en_400M_v5",
                                             revision="ffeb2b7ee715c226d4ffe5e4619f7dbb48624c20")
    table_shard = EdgeShard.load(TABLE_SHARD)
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(NANO_ONNX), so, providers=["CPUExecutionProvider"])
    nano_tok = AutoTokenizer.from_pretrained(str(NANO_DIR))

    def zero_fn(t):
        ids = zero_tok(t, add_special_tokens=True, truncation=True, max_length=512)["input_ids"]
        return zero_encode(table_shard, ids, scale)

    def nano_fn(t):
        return nano_encode(sess, nano_tok, t)

    sp = models.SearchParams(quantization=models.QuantizationSearchParams(rescore=False)) if kind == "binary" else None

    def timed_search(v):
        t0 = time.perf_counter()
        c.query_points("docs", query=v.tolist(), limit=100, search_params=sp,
                       with_payload=False, with_vectors=False)
        return (time.perf_counter() - t0) * 1000

    bucket_texts = {f"{lo}-{hi}w": synth_texts((lo + hi) // 2, N_TIMED) for lo, hi in BUCKETS}

    def run_path(encode_fn, warmup_seed):
        for t in synth_texts(8, WARMUP_SEARCHES, seed=warmup_seed):
            timed_search(encode_fn(t))
        order = list(bucket_texts)
        random.Random(warmup_seed).shuffle(order)
        res = {}
        for bkey in order:
            enc_lat, search_lat = [], []
            for t in bucket_texts[bkey]:
                t0 = time.perf_counter()
                v = encode_fn(t)
                t1 = time.perf_counter()
                search_lat.append(timed_search(v))
                enc_lat.append((t1 - t0) * 1000)
            res[bkey] = {
                "encode_ms_p50": round(statistics.median(enc_lat), 3),
                "search_ms_p50": round(statistics.median(search_lat), 3),
                "search_ms_p95": round(sorted(search_lat)[int(0.95 * len(search_lat)) - 1], 3),
                "total_ms_p50": round(statistics.median([a + b for a, b in zip(enc_lat, search_lat)]), 3),
            }
        return res, order

    zero_buckets, zero_order = run_path(zero_fn, 11)
    nano_buckets, nano_order = run_path(nano_fn, 12)
    mem_after_queries_mb = _docker_mem_used_mb(name)
    table_shard.close()
    _docker_stop_rm(name)
    return {
        "served": True,
        "load_s": load_s,
        "container_mem_after_load_mb": mem_after_load_mb,
        "container_mem_after_queries_mb": mem_after_queries_mb,
        "bucket_order": {"zero": zero_order, "nano": nano_order},
        "buckets": {b: {"zero": zero_buckets[b], "nano": nano_buckets[b]} for b in bucket_texts},
    }


def measure4():
    out = {"_what": "M9 pair edge prototype ROUND 4: does the winning round-3 config (binary "
                    "quantization, originals on_disk, quantized copy always_ram, rescore=false) "
                    "actually serve a 1M x 1024 index under a real memory-capped container? An "
                    "fp16 row runs as the default-reach-for contrast. Real Qdrant in Docker, not "
                    "qdrant_edge -- RSS isn't a real constraint, a cgroup memory limit is.",
           "round": 4, "warmup_searches": WARMUP_SEARCHES, "host": host(), "n_docs": N_DOCS,
           "dim": DIM, "mem_limits": list(MEM_LIMITS)}
    out["configs"] = {}
    for kind in ("binary", "fp16"):
        out["configs"][kind] = {}
        for limit in MEM_LIMITS:
            print(f"{kind} @ {limit}: starting", flush=True)
            r = serve_one(kind, limit)
            out["configs"][kind][limit] = r
            if r["served"]:
                d10 = r["buckets"]["6-10w"]
                print(f"{kind} @ {limit}: SERVED · load {r['load_s']}s · "
                      f"mem_after_load {r['container_mem_after_load_mb']} MiB · "
                      f"mem_after_queries {r['container_mem_after_queries_mb']} MiB · "
                      f"zero {d10['zero']} · nano {d10['nano']}", flush=True)
            else:
                print(f"{kind} @ {limit}: DIED -- {r['reason']}", flush=True)

    tag = out["host"]["cpu"].replace(" ", "_")
    dest = REPO / "results" / f"m9_edge_prototype_{tag}.json"
    dest.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {dest.relative_to(REPO)}")


if __name__ == "__main__":
    {"build": build, "measure": measure, "measure_one": measure_one, "measure3": measure3,
     "build4": build4, "measure4": measure4}[sys.argv[1]]()
