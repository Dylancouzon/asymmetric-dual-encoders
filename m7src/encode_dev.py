"""Encode every dev component's corpus + queries with the frozen teacher (fp16; see LEDGER)."""
import sys
import time

import torch

import devsuite
from teacher import QUERY_PREFIX, encode_cached

comps = sys.argv[1:] or devsuite.COMPONENTS
for c in comps:
    doc_ids, doc_texts, q_ids, q_texts, qrels = devsuite.load(c)
    t0 = time.time()
    dv = encode_cached(f"dev-{c}-docs", doc_texts, prefix="", dtype=torch.float16)
    qv = encode_cached(f"dev-{c}-queries-pfx", q_texts, prefix=QUERY_PREFIX, dtype=torch.float16)
    qv0 = encode_cached(f"dev-{c}-queries-nopfx", q_texts, prefix="", dtype=torch.float16)
    print(f"{c}: docs {dv.shape} queries {qv.shape} in {time.time()-t0:.0f}s", flush=True)
