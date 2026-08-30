"""The fp16 target-cache acceptance gate (m9/LEDGER.md §3.3).

Codex pass 2, BLOCKER-5: the lock said fp16 targets were *conditional* and the code used them
unconditionally, with no sample, no acceptance artifact and no fp32 fallback. This module is the
missing gate, and `screen.query_targets()` refuses to hand out fp16 targets until it has passed.

The check is numerical only, on the locked sample: 10,000 screen-pool texts stratified into ten
equal-count WordPiece-length deciles, seed 11, ids materialized and hashed at M9.0. Live fp32
encoding through the frozen teacher path is compared to the cached fp16 rows for the same texts.
"""
import hashlib
import json

import numpy as np
import torch

import m9base
from m9base import RESULTS, WORK

import data as m9data   # noqa: E402
import guard9           # noqa: E402

ARTIFACT = RESULTS / "m9_fp16_gate.json"


def sample_ids():
    """-> (indices into the screen pool, sha256). Deterministic; also written into the lock
    constants so the sample is pinned rather than re-derived on trust."""
    r = guard9.registry()["validation_samples"]["fp16_target_cache"]
    texts = json.loads((WORK / "m9_screen_queries.json").read_text())
    from transformers import AutoTokenizer
    st = guard9.registry()["models"]["students"]["bge-small-en-v1.5"]
    tok = AutoTokenizer.from_pretrained(st["repo"], revision=st["revision"])
    lens = np.array([len(x) for x in
                     tok(texts, truncation=True, max_length=512)["input_ids"]], dtype=np.int64)
    order = np.argsort(lens, kind="stable")
    rng = np.random.default_rng(r["seed"])
    per, out = r["n"] // 10, []
    for d in range(10):
        lo, hi = d * len(order) // 10, (d + 1) * len(order) // 10
        pool = order[lo:hi]
        out.append(rng.choice(pool, size=min(per, pool.size), replace=False))
    idx = np.sort(np.concatenate(out))
    return idx, hashlib.sha256(idx.tobytes()).hexdigest()


def run():
    r = guard9.registry()["validation_samples"]["fp16_target_cache"]
    guard9.begin_run("m9-fp16-gate")
    texts = json.loads((WORK / "m9_screen_queries.json").read_text())
    rows = np.load(WORK / "m9_screen_rows.npy")
    idx, ish = sample_ids()

    import teacher
    live = teacher.encode([texts[i] for i in idx], prefix=teacher.QUERY_PREFIX, max_length=512,
                          dtype=torch.float32)
    live = live / np.linalg.norm(live, axis=1, keepdims=True)
    cached = np.asarray(m9data.stella_query_targets()[rows[idx]], dtype=np.float32)

    cos = (live * cached).sum(1)
    max_abs = float(np.abs(live - cached).max())
    out = {"n": int(idx.size), "seed": r["seed"], "sample_sha256": ish,
           "min_cos": float(cos.min()), "mean_cos": float(cos.mean()), "max_abs": max_abs,
           "thresholds": {"min_cos": r["min_cos"], "max_abs": r["max_abs"]},
           "pass_min_cos": bool(cos.min() >= r["min_cos"]),
           "pass_max_abs": bool(max_abs <= r["max_abs"])}
    out["pass"] = bool(out["pass_min_cos"] and out["pass_max_abs"])
    out["_effect"] = ("PASS -> every arm trains on the fp16 target cache. FAIL -> the lock "
                      "requires fp32 targets for every arm, which means re-encoding 337,981 "
                      "texts through the teacher and holding 1.38 GB per teacher.")
    guard9.write_result(ARTIFACT, out, "m9-fp16-gate")
    print(json.dumps(out, indent=1))
    return out


def passed():
    """The predicate `screen.query_targets()` consults before returning fp16 rows."""
    if not ARTIFACT.exists():
        return False, "results/m9_fp16_gate.json does not exist -- run m9src/fp16_gate.py first"
    blob = json.loads(ARTIFACT.read_text())
    if not guard9.eligible(blob):
        return False, "the fp16 gate artifact is diagnostic or was written under a different lock"
    if not blob.get("pass"):
        return False, f"the fp16 gate FAILED: {json.dumps({k: blob[k] for k in ('min_cos','max_abs')})}"
    _idx, ish = sample_ids()
    if blob.get("sample_sha256") != ish:
        return False, "the fp16 gate ran on a different sample than the lock now specifies"
    return True, blob


if __name__ == "__main__":
    run()
