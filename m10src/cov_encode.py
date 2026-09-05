"""Encode the admitted COV surfaces with the frozen teacher, so the screen can score them.

Uses `m9src/teacher9.encode_cached` unchanged — the same chunk-resumable fp16 cache, the same
role prompts, keyed on the exact text list — so COV vectors are produced by exactly the path
that produced M9's. No new encoder, no new cache format.
"""
import json, os, sys, time
from pathlib import Path

os.environ.pop("HF_HUB_OFFLINE", None)
REPO = Path(__file__).resolve().parents[1]
for p in ("m7src", "m9src", "m10src"):
    sys.path.insert(0, str(REPO / p))
OUT = REPO / "work" / "m10cov"

import teacher9
from cov_admit import COMPONENTS
from cov_screen import load_component

KEY = "stella-400M-v5"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rec = {}
    for family, comps in COMPONENTS.items():
        for name, repo, rev in comps:
            qs, ds = load_component(name, repo, rev)
            t0 = time.time()
            print(f"\n=== {name}: {len(qs):,} queries, {len(ds):,} documents", flush=True)
            qv = teacher9.encode_cached(KEY, f"m10cov-{name}-q", qs, "query")
            print(f"  queries done {time.time()-t0:.0f}s", flush=True)
            dv = teacher9.encode_cached(KEY, f"m10cov-{name}-d", ds, "doc")
            rec[name] = dict(family=family, n_queries=len(qs), n_docs=len(ds),
                             q_shape=list(qv.shape), d_shape=list(dv.shape),
                             seconds=round(time.time() - t0, 1))
            print(f"  {name} done in {rec[name]['seconds']:.0f}s", flush=True)
            (OUT / "encode.json").write_text(json.dumps(rec, indent=1))
    print("\nwrote", OUT / "encode.json")


if __name__ == "__main__":
    main()
