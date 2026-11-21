"""Encode the BEIR NQ corpus with Stella's registered document path for the talk demo (exploratory).

Same path as m15src/vectors15.encode_docs: no prompt, max length 512, fp32 compute, normalized,
stored fp16. NQ pins come from results/m20_beir15_scores/nq/zero-dense.json. Resumable by shard.

Run on the RTX box: M7_ENCODER=stella PYTHONPATH=m7src python m15src/talk_encode_nq.py OUT_DIR
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

NQ_REV = "b7253e6c379163d024ddb1d6948152a91a2e3b46"
N_DOCS = 2681468
SHARD = 50_000


def main(out):
    import teacher
    from datasets import load_dataset
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    corpus = load_dataset("BeIR/nq", "corpus", revision=NQ_REV)["corpus"]
    if len(corpus) != N_DOCS:
        raise SystemExit(f"NQ has {len(corpus)} docs, M20 had {N_DOCS}")
    if not (out / "doc_ids.json").exists():
        (out / "doc_ids.json").write_text(json.dumps([str(x) for x in corpus["_id"]]))
    t0 = time.time()
    for s, lo in enumerate(range(0, N_DOCS, SHARD)):
        p = out / f"shard_{s:05d}.npy"
        if p.exists():
            continue
        rows = corpus.select(range(lo, min(lo + SHARD, N_DOCS)))
        texts = [f"{(t or '').strip()} {(x or '').strip()}".strip()
                 for t, x in zip(rows["title"], rows["text"])]
        v = teacher.encode(texts, prefix="", max_length=512, batch_tokens=16384,
                           model_id=teacher.TEACHER, revision=teacher.TEACHER_REV,
                           dtype=torch.float32, device="cuda")
        np.save(p.with_suffix(".tmp.npy"), v.astype(np.float16))
        p.with_suffix(".tmp.npy").rename(p)
        done = min(lo + SHARD, N_DOCS)
        print(f"shard {s} done: {done}/{N_DOCS} at {done / (time.time() - t0):.0f} docs/s",
              flush=True)
    print("NQ COMPLETE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
