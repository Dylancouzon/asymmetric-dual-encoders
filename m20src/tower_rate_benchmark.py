#!/usr/bin/env python3
"""Relative document-encode rate of the three M20 document towers, on one GPU.

Why this exists.  `results/m13_encode_benchmark.json` priced the reserved allowance from the
STELLA tower alone: 10M passages at the slowest measured rate, doubled, gives the registered
55.2-hour `reserved_batch_allowance`.  The reserved batch also needs bge-small and Arctic-M
document vectors for the same 10.1M passages, and those encodes were never in that arithmetic.
M20 adds ~18M more passages on top.  A cap that ignores two of three towers is a cap that ends a
paid run in the middle, so M20 registers its own hour cap from a measurement that includes them.

This is a rate measurement on SQuAD TRAINING passages.  No evaluation data, no protected path, no
vector cache, no gradients.  The absolute numbers are device-specific; the RATIO between towers is
what M20's cap uses, applied to the A100 stella rate already measured in M13.

  .venv/bin/python m20src/tower_rate_benchmark.py --device cuda
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _p in ("m7src", "bench"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

SOURCE = REPO / "work" / "train" / "stores" / "squad-ctx.json"
RESULT = REPO / "results" / "m20_tower_rate_benchmark.json"
SIZES = (1000, 5000)
TOWERS = {
    "stella-400M-v5": {"repo": "NovaSearch/stella_en_400M_v5",
                       "revision": "ffeb2b7ee715c226d4ffe5e4619f7dbb48624c20",
                       "backend": "teacher", "dim": 1024},
    "bge-small-en-v1.5": {"repo": "BAAI/bge-small-en-v1.5",
                          "revision": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
                          "backend": "sentence-transformers", "dim": 384},
    "arctic-m-v1.5": {"repo": "Snowflake/snowflake-arctic-embed-m-v1.5",
                      "revision": "e58a8f756156a1293d763f17e3aae643474e9b8a",
                      "backend": "sentence-transformers", "dim": 768},
}


def select_texts(payload, n):
    texts = payload["texts"]
    return [texts[i * len(texts) // n] for i in range(n)]


_LOADED = {}


def encode(name, texts, device):
    """The same kernel `m8src/pre_encode.py` uses for this tower, at the same settings."""
    import torch

    spec = TOWERS[name]
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
    if spec["backend"] == "teacher":
        import teacher

        return teacher.encode(list(texts), prefix="", max_length=512, batch_tokens=32768,
                              model_id=spec["repo"], revision=spec["revision"],
                              dtype=torch.float32, device=device, verbose=False)
    from sentence_transformers import SentenceTransformer

    if name not in _LOADED:
        model = SentenceTransformer(spec["repo"], revision=spec["revision"], device=device,
                                    model_kwargs={"dtype": torch.float32})
        model.max_seq_length = 512
        _LOADED[name] = model
    return _LOADED[name].encode(list(texts), batch_size=256, normalize_embeddings=True,
                                show_progress_bar=False, convert_to_numpy=True)


def release(name, device):
    """Free this tower before timing the next one.

    Production encodes one tower per process (`m8src/pre_encode.py --system`), so a tower never
    competes for VRAM with another.  Leaving them resident here made Arctic-M measure 18x slower
    than it is on a 10 GB card, purely from allocator pressure -- a measurement artifact that would
    have gone straight into the hour cap.
    """
    import torch

    if TOWERS[name]["backend"] == "teacher":
        import teacher

        teacher.release_teacher(TOWERS[name]["repo"], TOWERS[name]["revision"],
                                torch.float32, device)
    _LOADED.pop(name, None)
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default=str(RESULT))
    args = parser.parse_args(argv)

    payload = json.loads(SOURCE.read_text())
    rows = {}
    for name in TOWERS:
        release(name, args.device)
        measurements = []
        for n in SIZES:
            texts = select_texts(payload, n)
            encode(name, texts[:32], args.device)          # warm the loader and the kernel
            started = time.monotonic()
            values = encode(name, texts, args.device)
            seconds = time.monotonic() - started
            values = np.asarray(values, dtype=np.float32)
            if values.shape != (n, TOWERS[name]["dim"]) or not np.isfinite(values).all():
                raise RuntimeError(f"{name}: bad encode output {values.shape}")
            measurements.append({"passages": n, "seconds": seconds,
                                 "passages_per_second": n / seconds})
            print(f"  {name:<20} {n:>6} passages  {n / seconds:8.1f}/s", flush=True)
        slowest = max(m["seconds"] / m["passages"] for m in measurements)
        rows[name] = {"measurements": measurements, "slowest_seconds_per_passage": slowest}
        release(name, args.device)

    base = rows["stella-400M-v5"]["slowest_seconds_per_passage"]
    for name, row in rows.items():
        row["cost_relative_to_stella"] = row["slowest_seconds_per_passage"] / base
    record = {
        "status": "PASSED",
        "purpose": "relative document-encode cost of the three M20 towers; input to the M20 "
                   "cloud-hour cap in m20/REGISTRATION.md",
        "protected_evaluation_access": False,
        "source": str(SOURCE.relative_to(REPO)),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "selection": f"evenly spaced store positions, sizes {list(SIZES)}",
        "device": args.device,
        "sizes": list(SIZES),
        "towers": rows,
        "sum_cost_relative_to_stella": sum(r["cost_relative_to_stella"] for r in rows.values()),
        "limitation": "SQuAD training passages are a surrogate and this device is not the A100. "
                      "Only the RATIO between towers is carried into the cap; the absolute stella "
                      "rate comes from results/m13_encode_benchmark.json on the A100.",
    }
    Path(args.out).write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({name: round(r["cost_relative_to_stella"], 4) for name, r in rows.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
