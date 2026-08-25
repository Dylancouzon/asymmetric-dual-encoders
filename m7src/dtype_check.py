"""One-off: does fp16 teacher encoding change dev nDCG vs fp32? Decides the dev encode dtype."""
import numpy as np
import torch

import devsuite
from evalkit import score
from teacher import QUERY_PREFIX, encode

for comp in ["cqadup-programmers", "cqadup-physics"]:
    doc_ids, doc_texts, q_ids, q_texts, qrels = devsuite.load(comp)
    res = {}
    for label, dt in [("fp32", torch.float32), ("fp16", torch.float16)]:
        dv = encode(doc_texts, dtype=dt, batch_tokens=16384 if dt == torch.float32 else 32768)
        qv = encode(q_texts, prefix=QUERY_PREFIX, dtype=dt)
        res[label] = float(np.mean(list(score(qv, q_ids, dv, doc_ids, qrels).values())))
    print(f"{comp}: bge-base symmetric fp32 {res['fp32']:.6f}  fp16 {res['fp16']:.6f}  "
          f"delta {res['fp16']-res['fp32']:+.6f}", flush=True)
