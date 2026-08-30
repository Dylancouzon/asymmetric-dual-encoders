"""The edge cost table: what a query actually costs on a CPU (m9/LEDGER.md §Costs).

Runs anywhere, needs no GPU, and is deliberately kept off the machine doing the training. It
measures the SERVING path — ONNX Runtime, batch 1, fixed thread count, tokenizer included — for
nano's two candidate backbones and for the `mdbr-leaf-ir` comparator, so the frontier table
compares like with like.

Two things make it valid to run before nano is trained:
  * latency depends on the ARCHITECTURE and the tokenizer, not on the weights; and
  * it feeds a cost row, never a quality decision, so it may come from a different machine than
    the one producing the vectors. Nothing here is bit-compared against anything.

The host's CPU is recorded and must be named wherever a number from it is quoted.

    python m9src/edge_cost.py                 # export + measure everything
    python m9src/edge_cost.py --threads 4     # a specific thread budget
"""
import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "m9src"), str(REPO / "m7src"), str(REPO)]

# The bucket edges the cost table reports against (m9/registry.json bins.length_words).
BUCKETS = ((1, 5), (6, 10), (11, 20), (21, 50), (51, 120))
WORDS = ("alpha beta gamma delta epsilon zeta eta theta kappa lambda retrieval embedding vector "
         "index document query passage neural token model ranking corpus semantic").split()

MODELS = {
    "nano-bge-small": {"repo": "BAAI/bge-small-en-v1.5",
                       "revision": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
                       "head": 1024, "what": "nano's anchor backbone + Linear(384,1024)"},
    "nano-minilm-l6": {"repo": "sentence-transformers/all-MiniLM-L6-v2",
                       "revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
                       "head": 1024, "what": "nano's challenger backbone + Linear(384,1024)"},
    "mdbr-leaf-ir": {"repo": "MongoDB/mdbr-leaf-ir", "revision": None,
                     "head": None, "what": "the LEAF query tower, comparator only -- vendor rule "
                                           "keeps it out of any release"},
}


def synth(nwords, n, seed=0):
    import random
    r = random.Random(seed)
    return [" ".join(r.choice(WORDS) for _ in range(nwords)) for _ in range(n)]


def host():
    info = {"platform": platform.platform(), "machine": platform.machine(),
            "processor": platform.processor(), "python": platform.python_version()}
    try:
        if sys.platform == "darwin":
            info["cpu"] = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                         capture_output=True, text=True).stdout.strip()
            info["cores"] = subprocess.run(["sysctl", "-n", "hw.ncpu"],
                                           capture_output=True, text=True).stdout.strip()
        else:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    info["cpu"] = line.split(":", 1)[1].strip()
                    break
            info["cores"] = str(len(
                [l for l in Path("/proc/cpuinfo").read_text().splitlines()
                 if l.startswith("processor")]))
    except Exception as e:
        info["cpu_error"] = repr(e)[:120]
    return info


def export(name, out_dir, opset=17):
    """Export the serving graph (backbone → mean pool → head → L2) at fp32 and fp16."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    spec = MODELS[name]
    out_dir.mkdir(parents=True, exist_ok=True)
    p32, p16 = out_dir / "model.onnx", out_dir / "model_fp16.onnx"
    kw = {"revision": spec["revision"]} if spec["revision"] else {}
    tok = AutoTokenizer.from_pretrained(spec["repo"], **kw)
    if p32.exists() and p16.exists():
        return p32, p16, tok, None

    backbone = AutoModel.from_pretrained(spec["repo"], **kw).eval()
    hid = backbone.config.hidden_size
    head = torch.nn.Linear(hid, spec["head"]) if spec["head"] else None

    class Serve(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.b, self.h = backbone, head

        def forward(self, input_ids, attention_mask):
            hs = self.b(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            m = attention_mask.unsqueeze(-1).to(hs.dtype)
            v = (hs * m).sum(1) / m.sum(1).clamp(min=1e-9)
            if self.h is not None:
                v = self.h(v)
            return torch.nn.functional.normalize(v, dim=-1, eps=1e-12)

    ex = tok(["a short query", "a longer example sentence for the export trace"],
             padding=True, truncation=True, max_length=64, return_tensors="pt")
    torch.onnx.export(Serve().eval(), (ex["input_ids"], ex["attention_mask"]), str(p32),
                      input_names=["input_ids", "attention_mask"], output_names=["embedding"],
                      dynamic_axes={"input_ids": {0: "b", 1: "s"},
                                    "attention_mask": {0: "b", 1: "s"},
                                    "embedding": {0: "b"}},
                      opset_version=opset, do_constant_folding=True, dynamo=False)
    tok.save_pretrained(out_dir)
    import onnx
    from onnxruntime.transformers.float16 import convert_float_to_float16
    import copy
    g = onnx.load(str(p32))
    onnx.save(convert_float_to_float16(copy.deepcopy(g), keep_io_types=True), str(p16))
    n_params = sum(q.numel() for q in Serve().parameters())
    return p32, p16, tok, n_params


def measure(path, tok, threads, warmup=20, reps=100):
    """Batch-1 warm p50/p95 per length bucket, plus cold load. Tokenizer time IS included:
    a served query pays for it."""
    import onnxruntime as ort
    import resource

    t0 = time.perf_counter()
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    cold_load_ms = (time.perf_counter() - t0) * 1000

    out = {"cold_load_ms": round(cold_load_ms, 1), "threads": threads, "buckets": {}}
    for lo, hi in BUCKETS:
        texts = synth((lo + hi) // 2, warmup + reps)
        lat = []
        for i, t in enumerate(texts):
            s = time.perf_counter()
            b = tok([t], return_tensors="np", truncation=True, max_length=512)
            sess.run(None, {"input_ids": b["input_ids"].astype("int64"),
                            "attention_mask": b["attention_mask"].astype("int64")})
            if i >= warmup:
                lat.append((time.perf_counter() - s) * 1000)
        lat.sort()
        out["buckets"][f"{lo}-{hi}w"] = {
            "p50_ms": round(statistics.median(lat), 3),
            "p95_ms": round(lat[int(0.95 * len(lat)) - 1], 3),
            "mean_ms": round(statistics.fmean(lat), 3), "n": len(lat)}
    out["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                               / (1e6 if sys.platform == "darwin" else 1e3), 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4,
                    help="intra-op threads; an edge device is not a 16-core desktop")
    ap.add_argument("--only", default=None, help="one model name")
    a = ap.parse_args()

    root = REPO / "work" / "m9onnx"
    blob = {"_what": "edge serving cost, batch 1, CPU only. Feeds a COST row and never a quality "
                     "decision, which is why it may be measured on a different machine from the "
                     "one producing vectors. Latency depends on architecture and tokenizer, not "
                     "on weights, so it is valid before nano is trained.",
            "host": host(), "threads": a.threads, "buckets": [list(b) for b in BUCKETS],
            "models": {}}
    print(json.dumps(blob["host"], indent=1), flush=True)

    for name in ([a.only] if a.only else list(MODELS)):
        try:
            p32, p16, tok, n_params = export(name, root / name)
            row = {"what": MODELS[name]["what"], "repo": MODELS[name]["repo"],
                   "params": n_params,
                   "onnx_fp32_bytes": p32.stat().st_size, "onnx_fp16_bytes": p16.stat().st_size,
                   "shipped_fp16_bytes": sum(f.stat().st_size for f in (root / name).iterdir()
                                             if f.is_file() and f.name != "model.onnx"),
                   "fp32": measure(p32, tok, a.threads),
                   "fp16": measure(p16, tok, a.threads)}
            row["shipped_fp16_MB_decimal"] = round(row["shipped_fp16_bytes"] / 1e6, 3)
            blob["models"][name] = row
            q = row["fp16"]["buckets"]["6-10w"]
            print(f"{name}: fp16 6-10w p50 {q['p50_ms']} ms / p95 {q['p95_ms']} ms · "
                  f"cold load {row['fp16']['cold_load_ms']} ms · "
                  f"shipped {row['shipped_fp16_MB_decimal']} MB", flush=True)
        except Exception as e:
            blob["models"][name] = {"error": repr(e)[:400]}
            print(f"{name}: FAILED {repr(e)[:200]}", flush=True)

    tag = (blob["host"].get("cpu") or platform.machine()).replace(" ", "_")[:40]
    dest = REPO / "results" / f"m9_edge_cost_{tag}.json"
    dest.write_text(json.dumps(blob, indent=2))
    print(f"\nwrote {dest.relative_to(REPO)}")


if __name__ == "__main__":
    main()
