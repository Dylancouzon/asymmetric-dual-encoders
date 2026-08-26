"""Does our encode path reproduce the model its authors published? Check it before trusting it.

The M2 gate's first blocker was a loader mismatch (potion encoded through the wrong wrapper), and
this session's review found the same class again: stella's published pipeline is
Transformer -> Pooling(mean) -> Dense_1024, and our Spec initially omitted the Dense, so we would
have encoded a different model from the one whose MTEB score justified shortlisting it.

Adding the Dense makes the loader *plausible*. This makes it *validated*, with no hardcoded
numbers: it compares our `teacher.encode` against `sentence_transformers`, which implements each
repo's own `modules.json` faithfully. Both run fp32 so the comparison isolates loader fidelity
from dtype.

The number that matters is not per-vector cosine but the PAIRWISE SIMILARITY matrix, since that is
what ranking consumes -- a shared affine map can leave cosines high while moving rankings.

    ../.venv/bin/python validate_encoder.py                 # every registered encoder
    ../.venv/bin/python validate_encoder.py stella-400M-v5  # just one

Writes results/m7_encoder_validation.json. Any new Spec must pass before it may enter
teacher_probe.py or a corpus encode.
"""
import json
import sys

import numpy as np
import torch

import encoders
import teacher
from _paths import REPO

# Deliberately mixed: short/long, question/statement, near-duplicates (to stress the pairwise
# matrix), and repeated tokens (to exercise multiplicity handling).
TEXTS = [
    "What is the capital of France?",
    "Paris is the capital and most populous city of France.",
    "the the the the the",
    "A frozen document encoder pairs with a lookup-table query encoder for edge retrieval.",
    "How do I reset my password?",
    "Password reset instructions are available in the account settings page.",
    "Vector databases index high-dimensional embeddings for approximate nearest neighbour search.",
    "capital of France",
]
COS_BAR = 0.9999          # per-vector agreement
PAIRWISE_BAR = 1e-3       # max abs difference in the similarity matrix -- the ranking-relevant one


def st_encode(spec, texts):
    from sentence_transformers import SentenceTransformer
    kw = {"trust_remote_code": True} if spec.trust_remote_code else {}
    if spec.config_kwargs:
        kw["config_kwargs"] = dict(spec.config_kwargs)
    m = SentenceTransformer(spec.repo, revision=spec.revision, device="cuda",
                            model_kwargs={"dtype": torch.float32}, **kw)
    v = m.encode(texts, normalize_embeddings=True, batch_size=8, show_progress_bar=False)
    del m
    torch.cuda.empty_cache()
    return np.asarray(v, dtype=np.float64)


def ours(spec, texts):
    v = teacher.encode(texts, prefix="", max_length=spec.max_length, model_id=spec.repo,
                       revision=spec.revision, dtype=torch.float32, device="cuda")
    teacher._CACHE.clear()
    torch.cuda.empty_cache()
    return np.asarray(v, dtype=np.float64)


def main(names):
    out, failed = {}, []
    for name in names:
        spec = encoders.get(name)
        print(f"\n{name}  ({spec.repo} @ {spec.revision[:12]}, {spec.pooling} pooling, "
              f"post_dense={spec.post_dense})", flush=True)
        try:
            a, b = ours(spec, TEXTS), st_encode(spec, TEXTS)
        except Exception as e:
            print(f"  ERROR {type(e).__name__}: {e}")
            out[name] = {"status": "error", "error": f"{type(e).__name__}: {e}"}
            failed.append(name)
            continue
        if a.shape != b.shape:
            print(f"  FAIL shape {a.shape} vs sentence-transformers {b.shape}")
            out[name] = {"status": "fail", "reason": "shape",
                         "ours": list(a.shape), "st": list(b.shape)}
            failed.append(name)
            continue
        cos = (a * b).sum(1)
        dp = np.abs(a @ a.T - b @ b.T).max()
        ok = bool(cos.min() >= COS_BAR and dp <= PAIRWISE_BAR)
        print(f"  per-vector cosine  min {cos.min():.8f}  (bar {COS_BAR})")
        print(f"  pairwise sim  max|delta| {dp:.2e}  (bar {PAIRWISE_BAR:.0e})   "
              f"{'PASS' if ok else 'FAIL'}")
        out[name] = {"status": "pass" if ok else "fail", "dim": int(a.shape[1]),
                     "min_cosine_vs_sentence_transformers": float(cos.min()),
                     "max_abs_pairwise_sim_delta": float(dp),
                     "pooling": spec.pooling, "post_dense": spec.post_dense,
                     "revision": spec.revision}
        if not ok:
            failed.append(name)

    (REPO / "results" / "m7_encoder_validation.json").write_text(json.dumps(
        {"_note": "Our teacher.encode path vs sentence-transformers, which implements each repo's "
                  "own modules.json. Both fp32, so this isolates loader fidelity from dtype. The "
                  "load-bearing number is the PAIRWISE similarity delta, since ranking consumes "
                  "similarities, not vectors -- a shared affine map can keep cosines high while "
                  "moving rankings. Any new Spec must pass before entering teacher_probe.py or a "
                  "corpus encode. Bars: min cosine >= 0.9999, max|pairwise delta| <= 1e-3.",
         "bars": {"min_cosine": COS_BAR, "max_pairwise_delta": PAIRWISE_BAR},
         "n_texts": len(TEXTS), "results": out}, indent=1))
    print(f"\nwrote results/m7_encoder_validation.json — {len(failed)} failure(s): {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or list(encoders.REGISTRY)))
