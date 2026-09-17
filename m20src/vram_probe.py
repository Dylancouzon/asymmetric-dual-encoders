#!/usr/bin/env python3
"""Peak VRAM of each document tower on WORST-CASE 512-token inputs.

`results/m20_tower_rate_benchmark.json` measured throughput on SQuAD passages averaging 153
tokens.  That says nothing about a batch made entirely of 512-token documents, which BEIR corpora
do produce.  Stella is token-budgeted (32,768 padded tokens per batch) and so is bounded by
construction; the two SentenceTransformer towers use a fixed `batch_size=256`, which at 512 tokens
is 131,072 padded tokens -- four times Stella's budget -- on a 10 GB card.

This measures the real peak so the batch size is chosen from a measurement rather than from hope.
Synthetic text only: no corpus, no query, no protected path.

  PYTHONPATH=m20src:m7src .venv/bin/python -u m20src/vram_probe.py --device cuda
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
for _p in ("m7src", "m20src"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

RESULT = REPO / "results" / "m20_vram_probe.json"
# ~700 words of filler tokenizes well past 512 wordpieces, so every row truncates to the cap.
LONG = ("retrieval augmented generation storage index vector quantization latency throughput "
        * 120)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[256, 128, 64, 32])
    args = parser.parse_args(argv)

    import torch
    from sentence_transformers import SentenceTransformer

    import roster

    torch.backends.cuda.matmul.allow_tf32 = False
    total = torch.cuda.get_device_properties(0).total_memory if args.device == "cuda" else 0
    rows = {}

    for name in ("bge-small-en-v1.5", "arctic-m-v1.5"):
        spec = roster.TOWERS[name]
        model = SentenceTransformer(spec["model"], revision=spec["revision"], device=args.device,
                                    model_kwargs={"dtype": torch.float32})
        model.max_seq_length = 512
        tokens = len(model.tokenizer(LONG, truncation=True, max_length=512)["input_ids"])
        rows[name] = {"tokens_per_document": tokens, "by_batch_size": {}}
        for batch in args.batch_sizes:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            try:
                model.encode([LONG] * max(batch, 8), batch_size=batch, normalize_embeddings=True,
                             show_progress_bar=False, convert_to_numpy=True)
                peak = int(torch.cuda.max_memory_allocated())
                rows[name]["by_batch_size"][batch] = {"peak_bytes": peak, "ok": True}
                print(f"  {name:<20} batch {batch:>4}: peak {peak / 1e9:5.2f} GB", flush=True)
            except torch.cuda.OutOfMemoryError as error:
                rows[name]["by_batch_size"][batch] = {"ok": False, "error": str(error)[:200]}
                print(f"  {name:<20} batch {batch:>4}: OOM", flush=True)
        del model
        torch.cuda.empty_cache()

    # Stella is token-budgeted, so measure it at its registered budget rather than by batch size.
    import teacher

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    spec = roster.TOWERS["stella-400M-v5"]
    teacher.encode([LONG] * 128, prefix="", max_length=512, batch_tokens=32768,
                   model_id=spec["model"], revision=spec["revision"], dtype=torch.float32,
                   device=args.device, verbose=False)
    rows["stella-400M-v5"] = {"batch_tokens": 32768,
                              "peak_bytes": int(torch.cuda.max_memory_allocated()), "ok": True}
    print(f"  {'stella-400M-v5':<20} batch_tokens 32768: "
          f"peak {rows['stella-400M-v5']['peak_bytes'] / 1e9:5.2f} GB", flush=True)

    record = {"status": "MEASURED", "device": args.device, "total_vram_bytes": total,
              "document_text": "synthetic filler truncated to the 512-token cap; no corpus",
              "protected_evaluation_access": False, "towers": rows}
    RESULT.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"total_vram_gb": round(total / 1e9, 1)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
