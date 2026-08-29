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
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "m7src"))

from _paths import REPO, WORK

OUT = WORK / "trainq_texts.json"
MANIFEST = REPO / "results" / "m7_trainq_manifest.json"
# Side-branch-only transfer copy, for a machine that cannot re-derive the list. Re-deriving needs
# the 6.17M-row pool index AND the decontamination outputs, both gitignored, so a second machine
# has no honest path to the identical list without this.
TRANSFER = REPO / "transfer" / "trainq_texts.json.gz"


def load_texts():
    """The TRAIN query list, VERIFIED against its committed hash.

    It was an unpinned input to every teacher-learnability probe -- the most consequential
    comparison left -- and a probe fitted on a silently different query set would rank candidates
    against an incumbent row that saw different data. Restores from the gzipped transfer copy when
    the working file is absent, then checks the hash either way and refuses on mismatch.
    """
    if not OUT.exists() and TRANSFER.exists():
        import gzip
        print(f"restoring {OUT.name} from {TRANSFER.relative_to(REPO)}", flush=True)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(TRANSFER, "rb") as f:
            OUT.write_bytes(f.read())
    if not OUT.exists():
        raise SystemExit(f"{OUT} missing and no {TRANSFER.relative_to(REPO)} to restore it from. "
                         "Run `encode_trainq.py dump` on a machine that holds the pool index.")
    raw = OUT.read_bytes()
    if MANIFEST.exists():
        m = json.loads(MANIFEST.read_text())
        got = hashlib.sha256(raw).hexdigest()
        if got != m["sha256"]:
            raise SystemExit(f"{OUT} does not match results/m7_trainq_manifest.json\n"
                             f"  expected {m['sha256']}\n  got      {got}\n"
                             "Refusing: a probe fitted on a different TRAIN query set is not "
                             "comparable to the committed incumbent row.")
        print(f"trainq verified: {m['n_queries']:,} queries, sha256 {got[:16]}", flush=True)
    return json.loads(raw)


def dump():
    import pool as poolmod
    import train
    from train import Cfg
    index, _, _ = poolmod.build()
    q_texts, *_ = train.build_arrays(Cfg(), index)
    OUT.write_text(json.dumps(q_texts))
    raw = OUT.read_bytes()
    MANIFEST.write_text(json.dumps(
        {"_what": "provenance pin for work/trainq_texts.json, the TRAIN query list every "
                  "teacher-learnability probe is fitted on",
         "n_queries": len(q_texts), "sha256": hashlib.sha256(raw).hexdigest(),
         "bytes": len(raw), "produced_by": "scripts/encode_trainq.py dump"}, indent=1))
    print(f"wrote {OUT} with {len(q_texts):,} query texts; pinned in {MANIFEST.name}")


def encode():
    import encoders
    from teacher import QUERY_PREFIX, encode_cached
    q_texts = load_texts()
    sp = encoders.active()
    print(f"{sp.name}: encoding {len(q_texts):,} TRAIN queries, prefix={QUERY_PREFIX!r}", flush=True)
    v = encode_cached(f"trainq-{len(q_texts)}", q_texts, prefix=QUERY_PREFIX, dtype=torch.float16,
                      verbose=True)
    v = np.asarray(v)
    print(f"{sp.name}: {v.shape} {v.dtype}")


if __name__ == "__main__":
    {"dump": dump, "encode": encode}[sys.argv[1]]()
