"""Stella document vectors on the Mac, exact search, and the reproduction gate (E2, E4).

Method: `m15/MEASUREMENTS.md`, "Reproduction gate". Document vectors are the registered Stella
document path (no prompt, max length 512, fp32 compute), normalized, stored fp16.
"""
import json
import os
import sys

import numpy as np

from common import REPO, m20_rows, sha_file

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.7")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.5")
for p in ("m7src", "m12src", "m20src"):
    if str(REPO / p) not in sys.path:
        sys.path.append(str(REPO / p))

LOCAL = REPO / "artifacts" / "demo-stella"
VECS = REPO / "work" / "m15" / "vecs"
GATE_SYSTEMS = {"stella-query": "stella-query", "zero": "zero-dense", "nano": "nano-dense",
                "bm25": "bm25"}


def encode_docs(texts, out_dir, device="mps", shard=20_000, batch_tokens=8192):
    """Resumable shard encode with the registered Stella document path; returns an fp16 array."""
    import torch
    import roster_ids as I
    import teacher
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = []
    for s, lo in enumerate(range(0, len(texts), shard)):
        p = out_dir / f"shard_{s:05d}.npy"
        if not p.exists():
            v = teacher.encode(texts[lo:lo + shard], prefix="", max_length=512,
                               batch_tokens=batch_tokens, model_id=I.STELLA_MODEL,
                               revision=I.STELLA_REVISION, dtype=torch.float32, device=device)
            np.save(p.with_suffix(".tmp.npy"), v.astype(np.float16))
            p.with_suffix(".tmp.npy").rename(p)
            print(f"  shard {s} done ({lo + len(v)}/{len(texts)})", flush=True)
            if device == "mps":
                torch.mps.empty_cache()
        parts.append(np.load(p))
    return np.concatenate(parts)


def doc_vectors(dataset, data):
    """(vectors fp16, provenance) aligned to data['doc_ids']."""
    local = LOCAL / dataset
    if (local / "doc_vecs.npy").exists():
        ids = json.loads((local / "doc_ids.json").read_text())
        vecs = np.load(local / "doc_vecs.npy")
        if ids != data["doc_ids"]:
            pos = {d: i for i, d in enumerate(ids)}
            vecs = vecs[[pos[d] for d in data["doc_ids"]]]
        return vecs, {"source": str((local / "doc_vecs.npy").relative_to(REPO)),
                      "sha256": sha_file(local / "doc_vecs.npy")}
    d = VECS / dataset
    vecs = encode_docs(data["doc_texts"], d)
    return vecs, {"source": str(d.relative_to(REPO)), "encoded": "Mac MPS, this run"}


def exact_run(qv, dv, doc_ids, q_ids, k=100):
    from evalkit import topk_ids_scores
    return topk_ids_scores(qv, dv, doc_ids, k=k, chunk=250_000, device="cpu", qids=q_ids)


def ndcg10(run, qrels, q_ids):
    from evalkit import per_query_ndcg
    got = {str(q): float(v) for q, v in per_query_ndcg(run, qrels).items()}
    return {q: got.get(q, 0.0) for q in q_ids}


def bm25(data, q_texts=None):
    import roster as R
    return R.bm25_run(data["doc_ids"], data["doc_texts"], data["q_ids"],
                      q_texts or data["q_texts"])


def gate(dataset, data, dv, encoders):
    """Reproduce M20's per-query rows for Stella, Zero, Nano and BM25 on full queries."""
    out, ok = {}, True
    for name, system in GATE_SYSTEMS.items():
        if name == "bm25":
            run = bm25(data)
        else:
            run = exact_run(encoders[name].encode(data["q_texts"]), dv, data["doc_ids"],
                            data["q_ids"])
        mine = ndcg10(run, data["qrels"], data["q_ids"])
        ref, meta = m20_rows(dataset, system)
        a = np.array([mine[q] for q in data["q_ids"]])
        b = np.array([ref[q] for q in data["q_ids"]])
        row = {"mine": float(a.mean()), "m20": float(b.mean()),
               "abs_mean_diff": float(abs(a.mean() - b.mean())),
               "share_within_1e-6": float((np.abs(a - b) <= 1e-6).mean()), "m20_row": meta}
        row["passed"] = row["abs_mean_diff"] <= 5e-4 and row["share_within_1e-6"] >= 0.98
        ok &= row["passed"]
        out[name] = row
        print(f"  gate {dataset}/{name}: {row['mine']:.6f} vs {row['m20']:.6f} "
              f"within1e-6 {row['share_within_1e-6']:.3f} {'PASS' if row['passed'] else 'FAIL'}",
              flush=True)
    return ok, out
