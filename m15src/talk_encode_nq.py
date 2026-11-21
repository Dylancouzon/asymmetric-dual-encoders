"""Encode the BEIR NQ corpus with Stella's registered document path for the talk demo (exploratory).

Same path as m15src/vectors15.encode_docs: no prompt, max length 512, fp32 compute, normalized,
stored fp16. NQ pins come from results/m20_beir15_scores/nq/zero-dense.json. Resumable by shard.

Run on the RTX box: M7_ENCODER=stella PYTHONPATH=m7src python m15src/talk_encode_nq.py OUT_DIR
Parity on the Mac (MPS, 256 seeded passages vs the box vectors; receipt results/m15_talk_nq_parity.json):
  PYTHONPATH=m7src:m15src .venv-mac/bin/python m15src/talk_encode_nq.py --parity work/m15/vecs/nq_stella
"""
import json
import os
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
    t0, new = time.time(), 0
    for s, lo in enumerate(range(0, N_DOCS, SHARD)):
        p = out / f"shard_{s:05d}.npy"
        if p.exists():
            continue
        rows = corpus.select(range(lo, min(lo + SHARD, N_DOCS)))
        texts = [f"{(t or '').strip()} {(x or '').strip()}".strip()
                 for t, x in zip(rows["title"], rows["text"])]
        # 16384 filled the 10 GB card and spilled into WSL shared memory (shard 1: 58 min vs 13).
        v = teacher.encode(texts, prefix="", max_length=512,
                           batch_tokens=int(os.environ.get("BATCH_TOKENS", "8192")),
                           model_id=teacher.TEACHER, revision=teacher.TEACHER_REV,
                           dtype=torch.float32, device="cuda")
        np.save(p.with_suffix(".tmp.npy"), v.astype(np.float16))
        p.with_suffix(".tmp.npy").rename(p)
        done, new = min(lo + SHARD, N_DOCS), new + len(texts)
        # Rate over this run only; resumed shards are not re-encoded.
        print(f"shard {s} done: {done}/{N_DOCS} at {new / (time.time() - t0):.0f} docs/s",
              flush=True)
    print("NQ COMPLETE", flush=True)


def parity(vec_dir, n=256):
    from datasets import load_dataset
    from common import REPO, SEED, receipt, utc_now, write_result
    import vectors15 as V
    started, vec_dir = utc_now(), Path(vec_dir)
    corpus = load_dataset("BeIR/nq", "corpus", revision=NQ_REV)["corpus"]
    pick = np.sort(np.random.default_rng(SEED).choice(N_DOCS, n, replace=False))
    rows = corpus.select(pick.tolist())
    ids = json.loads((vec_dir / "doc_ids.json").read_text())
    assert [ids[i] for i in pick] == [str(x) for x in rows["_id"]], "doc_ids misaligned"
    texts = [f"{(t or '').strip()} {(x or '').strip()}".strip()
             for t, x in zip(rows["title"], rows["text"])]
    got = V.encode_docs(texts, REPO / "work/m15/vecs/nq-parity256", shard=n)
    box = np.stack([np.load(vec_dir / f"shard_{i // SHARD:05d}.npy", mmap_mode="r")[i % SHARD]
                    for i in pick])
    cos = (got.astype(np.float32) * box.astype(np.float32)).sum(1)
    blob = {"status": "COMPLETE", "analysis": "exploratory", "n": n, "seed": SEED,
            "min_cos": float(cos.min()), "median_cos": float(np.median(cos)),
            "passed": bool(cos.min() >= 0.9999), "box": "RTX 3080 CUDA fp32", "mac": "M5 Pro MPS fp32",
            "receipt": receipt(__file__, {}, started, ("torch", "transformers"))}
    print({k: blob[k] for k in ("n", "min_cos", "median_cos", "passed")})
    write_result(REPO / "results/m15_talk_nq_parity.json", blob)
    if not blob["passed"]:
        raise SystemExit("NQ parity FAILED")


if __name__ == "__main__":
    parity(sys.argv[2]) if sys.argv[1] == "--parity" else main(sys.argv[1])
