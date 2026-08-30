"""LEDGER G6 -- benchmark every new code path, then publish the serial schedule.

The rule this discharges (m7/CODEMAP.md, CLAUDE.md "Long runs must be watched"): never trust a
docstring's cost estimate. Both "a few minutes for the whole set" claims in M7's train.py were
wrong by two orders of magnitude, in the same file, on the same day. And a corpus with non-uniform
composition has no single rate -- M7's objective-B encode ran 1,511 -> 55 texts/s across its own
shards -- so this reports tokens/s as well as texts/s, and the schedule is computed from TOKENS.

What it measures, on NON-protected corpora only (dev components; the reserved four are not
downloaded, let alone read -- their pre-encode is a post-freeze step):
  * teacher encode throughput and peak VRAM at the box's batch settings;
  * the token-length distribution of Wikipedia-shaped and forum-shaped text, which is what the
    reserved-4 estimate has to be projected through;
  * the closed-form ridge solve at the 30,522-row control vocabulary (the B7 baseline).

Then it writes the serial GPU/RAM/disk schedule to results/m8_schedule.json, including the two
lines that decide whether the calendar is feasible at all:
  * the reserved-4 document pre-encode, ~10.12M docs, for EVERY scored system;
  * the E12 comparator pre-encode, whose LR-dense arm is a 1.5B model over the same 10.12M docs.
"""
import argparse
import json
import sys
import time

import numpy as np
import torch

import m8base
import devsuite
import encoders
import teacher

RESULTS = m8base.RESULTS
OUT = RESULTS / "m8_schedule.json"

# LEDGER 2.1. Token counts per corpus are ESTIMATED by projecting the measured dev distributions;
# the estimate is refined from the real corpora at pre-encode time, which is the only honest
# order (the reserved corpora are not downloaded tonight).
RESERVED_DOCS = {"fever": 5_416_568, "dbpedia-entity": 4_635_922,
                 "cqadup-android": 22_998, "cqadup-english": 40_221}
SHAPE = {"fever": "wikipedia", "dbpedia-entity": "wikipedia",
         "cqadup-android": "forum", "cqadup-english": "forum"}


def measure(sample, label, dtype=torch.float16, batch_tokens=32768):
    tok, _ = teacher.load_teacher(dtype=dtype)
    n_tok = [len(tok(t, truncation=True, max_length=512)["input_ids"]) for t in sample]
    teacher.encode(sample[:256], dtype=dtype, batch_tokens=batch_tokens)      # warm
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.time()
    v = teacher.encode(sample, dtype=dtype, batch_tokens=batch_tokens)
    torch.cuda.synchronize()
    dt = time.time() - t0
    return {
        "label": label, "n": len(sample), "seconds": round(dt, 2),
        "texts_per_s": round(len(sample) / dt, 1),
        "tokens_per_s": round(sum(n_tok) / dt),
        "mean_tokens": round(float(np.mean(n_tok)), 1),
        "p50_tokens": int(np.percentile(n_tok, 50)),
        "p95_tokens": int(np.percentile(n_tok, 95)),
        "max_tokens": int(max(n_tok)),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "dim": int(v.shape[1]), "dtype": str(dtype), "batch_tokens": batch_tokens,
    }


def ridge_timing(dim=1024, vocab=30522, n=200_000, seed=0):
    """The closed-form flat-table solve at the control vocabulary: the operation B7's block-CG
    curve is measured against. Timed as the Gram build + Cholesky solve, which is what dominates."""
    rng = np.random.default_rng(seed)
    # A sparse bag matrix with a realistic 12 nonzeros per row. Build the Gram as one sparse
    # matmul, not a Python loop over rows -- the first version of this function did the loop and
    # took longer than the thing it was timing, which is m7/CODEMAP.md pitfall 6 wearing a hat.
    from scipy import sparse
    t0 = time.time()
    nnz = 12
    rows = np.repeat(np.arange(n), nnz)
    cols = rng.integers(0, vocab, size=n * nnz)
    x = sparse.csr_matrix((np.ones(n * nnz, dtype=np.float64), (rows, cols)), shape=(n, vocab))
    g = np.asarray((x.T @ x).todense())
    t_gram = time.time() - t0
    g[np.diag_indices_from(g)] += 1e-3
    b = rng.standard_normal((vocab, dim))
    t0 = time.time()
    np.linalg.solve(g, b)
    t_solve = time.time() - t0
    return {"vocab": vocab, "dim": dim, "n_rows": n,
            "gram_seconds": round(t_gram, 2), "solve_seconds": round(t_solve, 2),
            "gram_bytes_fp64": vocab * vocab * 8,
            "note": ("at 64K the fp64 Gram is %.1f GB and at 128K it is %.1f GB -- above this "
                     "box's 18 GB peak budget, which is exactly why B7 measures a block-CG solver "
                     "instead. M7 closed granite-r2 and gte-modernbert on this arithmetic "
                     "(50,368-vocab fp64 Gram = 20.3 GB), not on merit."
                     % (65536 ** 2 * 8 / 1e9, 131072 ** 2 * 8 / 1e9))}


def schedule(runs):
    """Serial GPU/RAM/disk plan, computed from measured tokens/s."""
    by_shape = {r["label"]: r for r in runs}
    wiki = by_shape.get("nq-250k (wikipedia-shaped)")
    forum = by_shape.get("cqadup-physics (forum-shaped)")
    lines, total = [], 0.0
    for ds, n_docs in RESERVED_DOCS.items():
        ref = wiki if SHAPE[ds] == "wikipedia" else forum
        if ref is None:
            continue
        est_tokens = n_docs * ref["mean_tokens"]
        hours = est_tokens / ref["tokens_per_s"] / 3600
        lines.append({"stage": f"reserved pre-encode: {ds}", "docs": n_docs,
                      "shape": SHAPE[ds], "est_tokens": int(est_tokens),
                      "est_hours_per_system": round(hours, 2),
                      "vectors_gb_fp16": round(n_docs * 1024 * 2 / 1e9, 2)})
        total += hours
    return {
        "reserved_pre_encode": lines,
        "reserved_total_hours_per_teacher_dim_system": round(total, 2),
        "reserved_total_vectors_gb_fp16": round(sum(RESERVED_DOCS.values()) * 1024 * 2 / 1e9, 1),
        "systems_to_pre_encode": {
            "m8_candidate": "shares the teacher's document vectors with frozen M7 IF the teacher "
                            "does not change and no doc-side head (D1) ships. Otherwise its own "
                            "full pass.",
            "frozen_m7": "one pass with the frozen stella teacher.",
            "bge-small-en-v1.5 (E12, descriptive)": "384-d, 33M params: cheap, and its vectors "
                                                    "are ~0.4x the bytes.",
            "LR-dense-websearch (E12, descriptive)": "a 1.5B Qwen2.5 over the same 10.12M docs at "
                                                     "1536-d. THIS IS THE CALENDAR RISK: it is "
                                                     "plausibly larger than all of Stage R's "
                                                     "training combined, purchased for a "
                                                     "descriptive row. instructions-m8.md line 39 "
                                                     "already sanctions published numbers as "
                                                     "labelled context; the fallback is "
                                                     "pre-agreed rather than discovered.",
        },
        "hard_rules": [
            "Strictly sequential: one GPU/memory job at a time, enforced by flock (m7 drivers).",
            "18 GB peak RAM budget; nothing materializes a whole corpus.",
            "setsid nohup for anything over 10 minutes -- a harness interrupt killed M7's first "
            "final run 40 minutes in.",
            "Write the wall-clock estimate BEFORE launch; kill any job exceeding it 2x.",
            "Take the rate check IN the slow region, not on the first batches.",
        ],
        "disk": {"free_gb_at_transcription": 781,
                 "reserved_vectors_per_system_gb": round(
                     sum(RESERVED_DOCS.values()) * 1024 * 2 / 1e9, 1)},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10_000)
    ap.add_argument("--skip-encode", action="store_true")
    a = ap.parse_args()

    runs = []
    if not a.skip_encode:
        for comp, label in (("nq-250k", "nq-250k (wikipedia-shaped)"),
                            ("cqadup-physics", "cqadup-physics (forum-shaped)")):
            doc_ids, doc_texts, *_ = devsuite.load(comp)
            runs.append(measure(doc_texts[:a.n], label))
            print(json.dumps(runs[-1], indent=2))
            m8base.empty_cache()

    out = {
        "_note": "LEDGER G6. Measured on dev corpora only; the reserved four are not downloaded.",
        "encoder": encoders.by_repo(teacher.TEACHER).__dict__
        if hasattr(encoders.by_repo(teacher.TEACHER), "__dict__") else str(teacher.TEACHER),
        "throughput": runs,
        "ridge_control": ridge_timing(),
        "schedule": schedule(runs),
    }
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out["schedule"], indent=2, default=str))
    print(f"\nwrote {OUT}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
