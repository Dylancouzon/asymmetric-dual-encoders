"""Sustained throughput for the talk's cost slide (exploratory).

encode: W processes, each with its own encoder (ONNX Runtime 1 thread), closed loop for D seconds.
e2e:    one process, W threads sharing one encoder and one Qdrant Edge shard (Edge locks its
        directory, so one process), encode + search at each encoder's selected (ef, oversampling)
        from results/m15_talk_edge_<dataset>.json.

  .venv/bin/python m15src/talk_throughput.py --dataset nq --label c4-standard-8 \
      --price-per-hour 0.40 --workers 1,2,4,8 --seconds 30
"""
import argparse
import json
import multiprocessing as mp
import os
import platform
import statistics
import subprocess
import threading
import time

import numpy as np

from common import REPO, load_public, receipt, utc_now, write_result

ENCODERS = ("stella", "nano", "zero")


def make_encoder(name):
    import encoders15
    from huggingface_hub import snapshot_download
    if name == "zero":
        return encoders15.ZeroEncoder()
    if name == "nano":
        d = snapshot_download(encoders15.NANO_REPO, revision=encoders15.NANO_REVISION)
        return encoders15.OnnxEncoder("nano", d, "mean", expect_sha=encoders15.NANO_ONNX_SHA, threads=1)
    import roster_ids as I
    d = snapshot_download(encoders15.STELLA_ONNX_REPO, revision=encoders15.STELLA_ONNX_REVISION)
    return encoders15.OnnxEncoder("stella-query", d, "mean", prefix=I.STELLA_QUERY_PROMPT, threads=1)


def _encode_worker(args):
    name, queries, seconds, start_at, offset = args
    enc = make_encoder(name)
    for q in queries[:10]:
        enc.encode([q])
    if time.time() >= start_at:
        raise RuntimeError(f"{name} worker loaded after the shared start; pre-download models")
    while time.time() < start_at:          # all workers start together
        time.sleep(0.001)
    lat, i, end = [], offset, start_at + seconds
    while time.time() < end:
        t0 = time.perf_counter()
        enc.encode([queries[i % len(queries)]])
        lat.append((time.perf_counter() - t0) * 1000)
        i += 1
    return lat


def run_encode(name, queries, workers, seconds):
    start_at = time.time() + 25 + 4 * workers       # time for every worker to load its model
    with mp.get_context("spawn").Pool(workers) as pool:
        lats = pool.map(_encode_worker, [(name, queries, seconds, start_at, k * 997)
                                         for k in range(workers)])
    return summarize(lats, seconds)


def run_e2e(name, enc, shard, params, queries, workers, seconds):
    import qdrant_edge as E
    lats = [[] for _ in range(workers)]
    stop = threading.Event()

    def loop(k):
        i = k * 997
        while not stop.is_set():
            t0 = time.perf_counter()
            v = enc.encode([queries[i % len(queries)]])[0].astype(np.float32)
            shard.query(E.QueryRequest(query=E.Query.Nearest(v, using="stella"), limit=10,
                                       params=params))
            lats[k].append((time.perf_counter() - t0) * 1000)
            i += 1

    threads = [threading.Thread(target=loop, args=(k,)) for k in range(workers)]
    for t in threads:
        t.start()
    time.sleep(seconds)
    stop.set()
    for t in threads:
        t.join()
    return summarize(lats, seconds)


def summarize(lats, seconds):
    flat = [x for l in lats for x in l]
    return {"completed": len(flat), "qps": len(flat) / seconds,
            "p50_ms": statistics.median(flat), "p95_ms": float(np.percentile(flat, 95)),
            "p99_ms": float(np.percentile(flat, 99))}


def machine():
    info = {"platform": platform.platform(), "cpus": os.cpu_count()}
    try:
        info["lscpu"] = subprocess.run(["lscpu"], capture_output=True, text=True).stdout[:2000]
    except OSError:
        info["chip"] = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                      capture_output=True, text=True).stdout.strip()
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="nq")
    ap.add_argument("--label", required=True)
    ap.add_argument("--price-per-hour", type=float, required=True)
    ap.add_argument("--workers", default="1,2,4,8")
    ap.add_argument("--seconds", type=int, default=30)
    ap.add_argument("--modes", default="encode,e2e")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    started = utc_now()
    workers = [int(w) for w in a.workers.split(",")]
    queries = load_public(a.dataset, with_corpus=False)["q_texts"]
    out = {"encode": {}, "e2e": {}}
    if "encode" in a.modes:
        for name in ENCODERS:
            for w in workers:
                r = run_encode(name, queries, w, a.seconds)
                out["encode"].setdefault(name, {})[str(w)] = r
                print("encode", name, w, r, flush=True)
    if "e2e" in a.modes:
        import qdrant_edge as E
        bench = json.loads((REPO / f"results/m15_talk_edge_{a.dataset}.json").read_text())
        shard = E.EdgeShard.load(str(REPO / "work/m15/edge" / a.dataset))
        for name in ENCODERS:
            sel = bench["results"][name]["selected"] or bench["results"][name]["per_query"]
            params = E.SearchParams(hnsw_ef=sel["ef"], quantization=E.QuantizationSearchParams(
                rescore=True, oversampling=sel["oversampling"]))
            enc = make_encoder(name)
            for q in queries[:20]:
                enc.encode([q])
            for w in workers:
                r = run_e2e(name, enc, shard, params, queries, w, a.seconds)
                r["ef"], r["oversampling"] = sel["ef"], sel["oversampling"]
                out["e2e"].setdefault(name, {})[str(w)] = r
                print("e2e", name, w, r, flush=True)
    for mode in out.values():
        for rows in mode.values():
            for r in rows.values():
                r["usd_per_million"] = a.price_per_hour / (r["qps"] * 3600) * 1e6
    blob = {"status": "COMPLETE", "analysis": "exploratory", "label": a.label,
                        "price_per_hour_usd": a.price_per_hour, "seconds": a.seconds,
                        "machine": machine(), "results": out,
                        "protocol": "Closed loop, batch 1. encode: processes, ORT 1 thread each. "
                                    "e2e: threads sharing one encoder and one Edge shard. "
                                    "usd_per_million = price_per_hour / (qps * 3600) * 1e6.",
                        "receipt": receipt(__file__, {"dataset": a.dataset}, started,
                                           ("onnxruntime", "tokenizers", "qdrant-edge"))}
    if a.out:
        open(a.out, "w").write(json.dumps(blob, indent=2))
    else:
        write_result(REPO / f"results/m15_talk_throughput_{a.dataset}_{a.label}_{'-'.join(sorted(a.modes.split(',')))}.json", blob)


if __name__ == "__main__":
    main()
