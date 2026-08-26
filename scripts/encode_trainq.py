"""Encode the TRAIN query texts with a candidate encoder, WITHOUT building that encoder's doc pool.

The learnability probe (see m7/STATUS.md) fits a closed-form table per candidate against that
candidate's own TRAIN **query** vectors. Those need no documents at all -- but the only existing
path to them is inside `train.run`, which calls `pool.build()` first. Under a new encoder that
builds a 6.17M-doc pool: three hours of GPU to obtain a list of strings.

So this splits the two. `dump` resolves the TRAIN query text list using the EXISTING pool index
(the doc-id set is teacher-independent -- same corpus, same ids -- so the surviving pair list is
too), and `encode` reads that list back under whatever M7_ENCODER is set and calls the ordinary
`encode_cached`. The cache key it lands on is the one `train.run` will later ask for, because the
name, prefix and corpus hash are computed the same way.

    ../.venv/bin/python scripts/encode_trainq.py dump                      # once, default encoder
    M7_ENCODER=arctic-embed-l ../.venv/bin/python scripts/encode_trainq.py encode
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "m7src"))

from _paths import WORK

OUT = WORK / "trainq_texts.json"


def dump():
    import pool as poolmod
    import train
    from train import Cfg
    index, _, _ = poolmod.build()
    q_texts, *_ = train.build_arrays(Cfg(), index)
    OUT.write_text(json.dumps(q_texts))
    print(f"wrote {OUT} with {len(q_texts):,} query texts")


def encode():
    import encoders
    from teacher import QUERY_PREFIX, encode_cached
    if not OUT.exists():
        raise SystemExit(f"{OUT} missing -- run `encode_trainq.py dump` first (default encoder)")
    q_texts = json.loads(OUT.read_text())
    sp = encoders.active()
    print(f"{sp.name}: encoding {len(q_texts):,} TRAIN queries, prefix={QUERY_PREFIX!r}", flush=True)
    v = encode_cached(f"trainq-{len(q_texts)}", q_texts, prefix=QUERY_PREFIX, dtype=torch.float16,
                      verbose=True)
    v = np.asarray(v)
    print(f"{sp.name}: {v.shape} {v.dtype}")


if __name__ == "__main__":
    {"dump": dump, "encode": encode}[sys.argv[1]]()
