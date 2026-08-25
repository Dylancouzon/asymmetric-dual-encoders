"""Bring-up step 4: teacher encode throughput + peak VRAM on 10K docs, and dtype agreement.

Extrapolates wall-clock per approved corpus and records the per-stage RAM/disk budget
the M7 ops rules require before any full encode.
"""
import json
import time

import numpy as np
import torch

from _paths import REPO
from core import load_beir
from teacher import TEACHER, TEACHER_REV, encode, load_teacher

doc_ids, doc_texts, *_ = load_beir("fiqa")
sample = doc_texts[:10_000]
tok, _ = load_teacher()
n_tok = [len(tok(t, truncation=True, max_length=512)["input_ids"]) for t in sample]
print(f"10K FiQA docs: mean {np.mean(n_tok):.1f} tok, p95 {np.percentile(n_tok,95):.0f}, max {max(n_tok)}")

out = {"sample": {"n": len(sample), "mean_tokens": round(float(np.mean(n_tok)), 1),
                  "p95_tokens": int(np.percentile(n_tok, 95))},
       "teacher": {"model": TEACHER, "revision": TEACHER_REV}, "runs": {}}
ref = None
for label, dtype, bt in [("fp32", torch.float32, 16384), ("fp16", torch.float16, 32768),
                         ("bf16", torch.bfloat16, 32768)]:
    load_teacher(dtype=dtype)  # warm
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    encode(sample[:512], dtype=dtype, batch_tokens=bt)  # warmup
    torch.cuda.synchronize(); t0 = time.time()
    v = encode(sample, dtype=dtype, batch_tokens=bt)
    torch.cuda.synchronize(); dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9
    if ref is None:
        ref = v
        agree = 1.0
    else:
        agree = float((ref * v).sum(1).mean())
    out["runs"][label] = {"texts_per_s": round(len(sample)/dt, 1), "tokens_per_s": round(sum(n_tok)/dt),
                          "peak_vram_gb": round(peak, 2), "cos_vs_fp32": round(agree, 6),
                          "batch_tokens": bt}
    print(f"{label}: {len(sample)/dt:7.1f} texts/s  {sum(n_tok)/dt:9.0f} tok/s  "
          f"peak {peak:.2f} GB  cos-vs-fp32 {agree:.6f}", flush=True)

r = out["runs"]["fp32"]["texts_per_s"]
out["extrapolation_fp32_hours"] = {k: round(v / r / 3600, 2) for k, v in
    {"the six (269K docs)": 269_117, "miracl-en (32.9M passages)": 32_900_000,
     "nq-dev-250K": 250_000, "hotpotqa (5.2M)": 5_233_329, "esci (1.8M products)": 1_800_000,
     "fever-wiki (5.4M)": 5_400_000}.items()}
out["storage_gb_fp16_768d"] = {k: round(v * 768 * 2 / 1e9, 2) for k, v in
    {"the six (269K)": 269_117, "nq-dev-250K": 250_000, "hotpotqa (5.2M)": 5_233_329,
     "esci (1.8M)": 1_800_000, "fever-wiki (5.4M)": 5_400_000, "miracl-en (32.9M)": 32_900_000}.items()}
print("\nfp32 extrapolated hours:", json.dumps(out["extrapolation_fp32_hours"]))
print("fp16 storage GB:", json.dumps(out["storage_gb_fp16_768d"]))
(REPO / "results" / "m7_throughput.json").write_text(json.dumps(out, indent=1))
