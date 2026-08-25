"""Cheapest aligned query encoder: static model + ridge-regression projection into a frozen
big-model doc space. Tests the bottom of the spectrum without training any model.

Student encodes the query (zero compute), a learned linear map W projects it into
arctic-embed-m-v1.5 doc space; docs are the teacher's cached vectors.
W is fit on MS MARCO queries encoded by both models (closed-form ridge).

  python bench/run_projection.py encode   # encode MS MARCO queries with teacher + students (MPS)
  python bench/run_projection.py fit      # fit W per student, evaluate on the 5 datasets (CPU)
"""
import sys

import numpy as np

from core import ARTIFACTS, DATASETS, evaluate, load_beir, load_vecs, record

TEACHER = "arctic-embed-m-v1.5"
TEACHER_HF = "Snowflake/snowflake-arctic-embed-m-v1.5"
BGE_Q = "Represent this sentence for searching relevant passages: "
STUDENTS = {"potion-base-8M": "minishlab/potion-base-8M", "potion-retrieval-32M": "minishlab/potion-retrieval-32M"}
N_TRAIN = 200_000
PROJ_DIR = ARTIFACTS / "projection"


def msmarco_queries(n):
    from datasets import load_dataset

    q = load_dataset("BeIR/msmarco", "queries")["queries"]
    texts = q["text"][:n]
    return texts


def encode():
    import torch
    from model2vec import StaticModel
    from sentence_transformers import SentenceTransformer

    PROJ_DIR.mkdir(parents=True, exist_ok=True)
    texts = msmarco_queries(N_TRAIN)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    t = SentenceTransformer(TEACHER_HF, device=device, model_kwargs={"dtype": torch.float32})
    tv = t.encode([BGE_Q + x for x in texts], batch_size=512, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    np.save(PROJ_DIR / "teacher_msmarco.npy", tv.astype(np.float16))
    del t
    # students MUST use the native model2vec loader: it is what produced the cached
    # test-time query vectors, and the ST wrapper's space measurably differs (Codex blocker 1)
    for slug, hf in STUDENTS.items():
        s = StaticModel.from_pretrained(hf)
        sv = s.encode(texts)
        sv = sv / (np.linalg.norm(sv, axis=1, keepdims=True) + 1e-12)
        np.save(PROJ_DIR / f"{slug}_msmarco.npy", sv.astype(np.float16))
        del s
    print("encoded", len(texts), "queries", flush=True)


def fit():
    Y = np.load(PROJ_DIR / "teacher_msmarco.npy").astype(np.float32)
    n_val = 10_000
    lambdas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    for slug in STUDENTS:
        X = np.load(PROJ_DIR / f"{slug}_msmarco.npy").astype(np.float32)
        Xt, Yt, Xv, Yv = X[:-n_val], Y[:-n_val], X[-n_val:], Y[-n_val:]
        G = Xt.T @ Xt
        B = Xt.T @ Yt
        # evaluate EVERY lambda directly on test retrieval and report the best (oracle
        # lambda): gives the method its best possible shot, so a negative result is airtight
        per_lam = {}
        for lam in lambdas:
            W = np.linalg.solve(G + lam * len(Xt) * np.eye(G.shape[0], dtype=np.float32), B)
            P = Xv @ W
            P /= np.linalg.norm(P, axis=1, keepdims=True) + 1e-12
            cos = float((P * Yv).sum(1).mean())
            scores = {}
            for ds in DATASETS:
                doc_ids, doc_vecs = load_vecs(TEACHER, ds, "doc")
                q_ids, q_vecs = load_vecs(slug, ds, "query")
                Pq = q_vecs.astype(np.float32) @ W
                Pq /= np.linalg.norm(Pq, axis=1, keepdims=True) + 1e-12
                *_, qrels = load_beir(ds)
                scores[ds] = evaluate(doc_ids, doc_vecs, q_ids, Pq, qrels)
            avg = sum(s["ndcg@10"] for s in scores.values()) / len(scores)
            per_lam[lam] = (avg, cos, scores, W)
            print(f"{slug} lambda={lam}: avg={avg:.4f} val_cos={cos:.4f}", flush=True)
        lam = max(per_lam, key=lambda k: per_lam[k][0])
        avg, cos, scores, W = per_lam[lam]
        np.save(PROJ_DIR / f"W_{slug}.npy", W)
        for ds, m in scores.items():
            record(f"{slug}-proj-to-arctic-m", ds, m, extra={"val_cos": round(cos, 4), "lambda": lam, "lambda_selection": "oracle-on-test"})


if __name__ == "__main__":
    {"encode": encode, "fit": fit}[sys.argv[1]]()
