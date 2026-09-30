"""E1: per-query encode latency on the M5 Pro CPU under one protocol (m15/MEASUREMENTS.md, E1).

  e1_latency.py --prepare   draw queries, export bge-small and LEAF to ONNX, run parity gates
  e1_latency.py --measure   three fresh worker processes per encoder, sequentially
"""
import os
import sys
os.environ.update(OMP_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false")
if "--prepare" not in sys.argv:   # timing never touches the network
    os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
import argparse
import json
import resource
import subprocess
import time
from pathlib import Path

import numpy as np

from common import EVAL12, REPO, SEED, load_public, receipt, sha_file, utc_now, write_result

WORK = REPO / "work" / "m15"
QUERIES = WORK / "e1_queries.json"
PARITY = WORK / "e1_parity.json"
OUT = REPO / "results" / "m15_e1_latency.json"
ENCODERS = ("zero", "nano", "stella-query", "bge-small", "leaf-query")
BUCKETS = {"short": (1, 4), "medium": (5, 12), "long": (13, 64)}
PER_BUCKET = 100


def draw_queries():
    pool, sources = [], []
    for dataset in EVAL12:
        data = load_public(dataset, with_corpus=False)
        pool += [(dataset, q, t) for q, t in zip(data["q_ids"], data["q_texts"])]
        sources.append({dataset: data["pins"]})
    rng = np.random.default_rng(SEED)
    drawn = {}
    for bucket, (lo, hi) in BUCKETS.items():
        members = [p for p in pool if lo <= len(p[2].split()) <= hi]
        pick = rng.choice(len(members), PER_BUCKET, replace=False)
        drawn[bucket] = [{"dataset": members[i][0], "qid": members[i][1], "text": members[i][2]}
                         for i in sorted(pick)]
    return drawn, sources


def reference(name, texts):
    """The registered torch path for parity (Nano: hash gate instead, see the amendment)."""
    import torch
    import roster_ids as I
    if name == "stella-query":
        sys.path.insert(0, str(REPO / "m7src"))
        import teacher
        return teacher.encode(texts, prefix=I.STELLA_QUERY_PROMPT, max_length=512,
                              model_id=I.STELLA_MODEL, revision=I.STELLA_REVISION,
                              dtype=torch.float32, device="cpu")
    from sentence_transformers import SentenceTransformer
    repo, rev = {"bge-small": (I.BGE_MODEL, I.BGE_REVISION),
                 "leaf-query": (I.LEAF_MODEL, I.LEAF_REVISION)}[name]
    st = SentenceTransformer(repo, revision=rev, device="cpu",
                             model_kwargs={"dtype": torch.float32})
    if name == "leaf-query":
        return st.encode(texts, prompt_name="query", normalize_embeddings=True)
    return st.encode([I.BGE_PREFIX + t for t in texts], normalize_embeddings=True)


def prepare():
    import encoders15 as E
    import roster_ids as I
    WORK.mkdir(parents=True, exist_ok=True)
    if not QUERIES.exists():
        drawn, sources = draw_queries()
        QUERIES.write_text(json.dumps({"seed": SEED, "buckets": drawn, "sources": sources}))
    drawn = json.loads(QUERIES.read_text())["buckets"]
    E.export_sentence_transformer("bge-small", I.BGE_MODEL, I.BGE_REVISION)
    E.export_sentence_transformer("leaf-query", I.LEAF_MODEL, I.LEAF_REVISION)
    rng = np.random.default_rng(SEED)
    everything = [q["text"] for b in drawn.values() for q in b]
    sample = [everything[i] for i in sorted(rng.choice(len(everything), 32, replace=False))]
    parity = {}
    for name in ("stella-query", "bge-small", "leaf-query"):
        got = E.make(name).encode(sample)
        ref = np.asarray(reference(name, sample), dtype=np.float32)
        cos = (got * ref).sum(1)
        parity[name] = {"min_cos": float(cos.min()), "n": len(sample),
                        "passed": bool(cos.min() >= 0.9999)}
        print(name, parity[name], flush=True)
    parity["nano"] = {"gate": "model.onnx sha256 equals the M13 freeze", "passed": True,
                      "sha256": E.NANO_ONNX_SHA}
    PARITY.write_text(json.dumps(parity, indent=2))
    if not all(v["passed"] for v in parity.values()):
        raise SystemExit("parity gate failed; see work/m15/e1_parity.json")


def fertility_cost(tok, texts):
    lat = []
    for t in texts:
        s = time.perf_counter()
        enc = tok.encode(t, add_special_tokens=False)
        _ = len(enc.ids) / max(1, len(t.split()))
        lat.append((time.perf_counter() - s) * 1000)
    return float(np.median(lat))


def worker(name):
    started = time.perf_counter()
    import encoders15 as E
    enc = E.make(name)
    hydration = time.perf_counter() - started
    drawn = json.loads(QUERIES.read_text())["buckets"]
    first = time.perf_counter()
    enc.encode([drawn["medium"][0]["text"]])
    cold_ms = (time.perf_counter() - first) * 1000
    result = {}
    for bucket, queries in drawn.items():
        texts = [q["text"] for q in queries]
        for t in texts[:5]:
            enc.encode([t])
        lat = []
        for t in texts:
            s = time.perf_counter()
            v = enc.encode([t])
            lat.append((time.perf_counter() - s) * 1000)
            if not np.isfinite(v).all():
                raise RuntimeError("nonfinite vector")
        result[bucket] = {"p50_ms": float(np.quantile(lat, .5)),
                          "p95_ms": float(np.quantile(lat, .95)), "samples_ms": lat}
    row = {"encoder": name, "hydration_s": hydration, "first_query_ms": cold_ms,
           "by_bucket": result,
           # macOS reports ru_maxrss in bytes.
           "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    if name == "zero":
        texts = [q["text"] for b in drawn.values() for q in b]
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(enc.dir / "tokenizer.json"))
        row["fertility_feature_p50_ms"] = fertility_cost(tok, texts)
    return row


def asset_bytes(name):
    import encoders15 as E
    from huggingface_hub import snapshot_download
    d = {"zero": lambda: snapshot_download(E.I.ZERO_HUB_REPO, revision=E.I.ZERO_HUB_REVISION),
         "nano": lambda: snapshot_download(E.NANO_REPO, revision=E.NANO_REVISION),
         "stella-query": lambda: snapshot_download(E.STELLA_ONNX_REPO,
                                                   revision=E.STELLA_ONNX_REVISION),
         "bge-small": lambda: E.EXPORTS / "bge-small",
         "leaf-query": lambda: E.EXPORTS / "leaf-query"}[name]()
    keep = {"zero": ("model.npz", "tokenizer.json", "config.json")}.get(name)
    files = [p for p in Path(d).iterdir() if p.is_file() and
             (p.name in keep if keep else p.suffix in (".onnx", ".json", ".txt"))]
    return {p.name: p.resolve().stat().st_size for p in files}, {
        p.name: sha_file(p) for p in files if p.suffix in (".onnx", ".npz")}


def measure():
    started = utc_now()
    if OUT.exists():
        raise SystemExit(f"{OUT} exists")
    parity = json.loads(PARITY.read_text())
    if not all(v["passed"] for v in parity.values()):
        raise SystemExit("parity gate has not passed")
    rows = []
    for name in ENCODERS:
        for trial in range(3):
            proc = subprocess.run([sys.executable, __file__, "--worker", name], cwd=REPO,
                                  capture_output=True, text=True,
                                  env={**os.environ, "PYTHONPATH": str(REPO / "m15src")})
            if proc.returncode:
                raise RuntimeError(f"{name} trial {trial} failed:\n{proc.stderr[-2000:]}")
            row = json.loads(proc.stdout.strip().splitlines()[-1])
            row["trial"] = trial
            rows.append(row)
            print("measured", name, trial, {b: round(v["p50_ms"], 3)
                                           for b, v in row["by_bucket"].items()}, flush=True)
    summary = {}
    for name in ENCODERS:
        mine = [r for r in rows if r["encoder"] == name]
        sizes, hashes = asset_bytes(name)
        summary[name] = {
            "p50_ms_median_of_trials": {b: float(np.median([r["by_bucket"][b]["p50_ms"]
                                                            for r in mine])) for b in BUCKETS},
            "p95_ms_median_of_trials": {b: float(np.median([r["by_bucket"][b]["p95_ms"]
                                                            for r in mine])) for b in BUCKETS},
            "hydration_s": float(np.median([r["hydration_s"] for r in mine])),
            "first_query_ms": float(np.median([r["first_query_ms"] for r in mine])),
            "peak_rss_bytes": int(np.median([r["peak_rss_bytes"] for r in mine])),
            "asset_bytes": sizes, "asset_total_bytes": sum(sizes.values()),
            "model_sha256": hashes}
    zero = [r for r in rows if r["encoder"] == "zero"]
    summary["zero"]["fertility_feature_p50_ms"] = float(np.median(
        [r["fertility_feature_p50_ms"] for r in zero]))
    import onnxruntime
    write_result(OUT, {
        "status": "COMPLETE", "measurement": "E1",
        "protocol": "3 fresh processes per encoder, sequential, idle machine; batch 1; ORT CPU "
                    "4 intra-op / 1 inter-op threads, ORT_ENABLE_ALL, fp32, dynamic padding; "
                    "Zero on its NumPy encoder; 5 warm-ups per bucket; 100 real test queries "
                    "per bucket; OS file cache not flushed",
        "buckets_words": BUCKETS, "onnxruntime": onnxruntime.__version__,
        "parity": parity, "queries_sha256": sha_file(QUERIES),
        "queries": json.loads(QUERIES.read_text())["buckets"],
        "summary": summary, "rows": rows,
        "receipt": receipt(__file__, [{"queries": str(QUERIES.relative_to(REPO))}], started,
                           ("onnxruntime", "tokenizers", "torch", "sentence-transformers"))})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--prepare", action="store_true")
    g.add_argument("--measure", action="store_true")
    g.add_argument("--worker", choices=ENCODERS)
    a = ap.parse_args()
    if a.prepare:
        prepare()
    elif a.measure:
        measure()
    else:
        print(json.dumps(worker(a.worker)))
