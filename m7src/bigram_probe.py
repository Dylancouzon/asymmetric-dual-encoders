"""Capacity lever #1, cheapest decisive form: do BIGRAM ROWS buy retrieval quality?

Per results/m7_absorb_check.json, n-gram rows and multiplicity-dependent pooling are the ONLY
query-side changes that add capacity to a token-lookup table — everything else is absorbable.
This probe answers the bigram question closed-form, before any table.py surgery or training: the
augmented model is q = normalize(sum unigram rows + sum bigram rows), fitted by the same ridge as
stage0 (unigram init = teacher rows, bigram init = zeros, so lam shrinks bigram rows toward
"no correction"), scored on the proxy-3 dev components against the cached stella doc vectors,
and paired-bootstrapped against the matched unigram-only table.

Feature convention: top-K adjacent WordPiece id pairs by TRAIN-query frequency, [CLS]/[SEP]
excluded (near-constant features). Count / len, like the unigram bag — any per-query scalar is
absorbed by the final normalize. Artifact cost if adopted: K x 1024 fp16 = 2 MB per 1,000 rows.

Peak RAM is the dense Gram: (30522+K)^2 fp64 = 10.1 GB at K=5,000, 13.1 GB at K=10,000 — the
run checks available RAM and refuses a K it cannot afford (the M4 OOM lesson).
"""
import json
import sys
import time
from collections import Counter

import numpy as np
import scipy.sparse as sp
import torch

import dev_eval
import mix
from _paths import REPO
from evalkit import per_query_ndcg, topk_ids_scores
from init_table import get_init, spec_tag
from stage0_ridge import bag_matrix, solve_ridge
from table import NO_PREFIX, get_tokenizer, tokenize
from teacher import QUERY_PREFIX, encode_cached
import boot

SPECIALS = {101, 102, 0}
COMPONENTS = ("nq-250k", "cqadup-programmers", "cqadup-physics")


def top_bigrams(tok, texts, pre, k):
    c = Counter()
    B = 4096
    for lo in range(0, len(texts), B):
        for ids in tokenize(tok, texts[lo:lo + B], pre):
            c.update((a, b) for a, b in zip(ids, ids[1:])
                     if a not in SPECIALS and b not in SPECIALS)
    return [bg for bg, _ in c.most_common(k)]


def aug_matrix(tok, texts, pre, V, bmap):
    X = bag_matrix(tok, texts, pre, V)
    indptr, indices, data = [0], [], []
    B = 4096
    for lo in range(0, len(texts), B):
        for ids in tokenize(tok, texts[lo:lo + B], pre):
            cols = Counter(bmap[bg] for bg in zip(ids, ids[1:]) if bg in bmap)
            indices.append(np.fromiter(cols.keys(), dtype=np.int64, count=len(cols)))
            data.append(np.fromiter(cols.values(), dtype=np.float64,
                                    count=len(cols)) / max(1, len(ids)))
            indptr.append(indptr[-1] + len(cols))
    Xb = sp.csr_matrix((np.concatenate(data) if data else np.zeros(0),
                        np.concatenate(indices) if indices else np.zeros(0, np.int64),
                        np.array(indptr)), shape=(len(texts), len(bmap)))
    return sp.hstack([X, Xb], format="csr")


def eval_w(W, tok, pre, V, bmap):
    per = {}
    for comp in COMPONENTS:
        doc_ids, _, q_ids, q_texts, qrels, dv = dev_eval.doc_vecs(comp)
        Xq = aug_matrix(tok, q_texts, pre, V, bmap)
        qv = np.asarray(Xq @ W, dtype=np.float32)
        n = np.linalg.norm(qv, axis=1, keepdims=True)
        qv = qv / np.clip(n, 1e-9, None)
        run = topk_ids_scores(qv, dv, doc_ids, k=100,
                              chunk=dev_eval.CHUNK.get(comp, 250_000), qids=q_ids)
        per[comp] = per_query_ndcg(run, qrels)
    return per


def main(k=5000, lam=0.01, n_queries=None):
    t0 = time.time()
    tok = get_tokenizer()
    V = tok.vocab_size
    pre = NO_PREFIX
    avail_gb = int(open("/proc/meminfo").readlines()[2].split()[1]) / 1e6
    need_gb = (V + k) ** 2 * 8 / 1e9
    if need_gb > avail_gb - 3:
        raise SystemExit(f"REFUSED: K={k} needs a {need_gb:.1f} GB Gram, only "
                         f"{avail_gb:.1f} GB available")
    qs = mix.query_texts(train_only=True)
    if n_queries and len(qs) > n_queries:
        rng = np.random.default_rng(0)
        qs = [qs[i] for i in rng.choice(len(qs), size=n_queries, replace=False)]
    print(f"bigram probe: K={k} lam={lam} on {len(qs):,} TRAIN queries "
          f"(Gram {need_gb:.1f} GB, {avail_gb:.0f} GB free)", flush=True)

    bigrams = top_bigrams(tok, qs, pre, k)
    bmap = {bg: j for j, bg in enumerate(bigrams)}
    print(f"  top-{k} bigrams selected ({time.time()-t0:.0f}s)", flush=True)

    Y = np.asarray(encode_cached(f"stage0-qtargets-pfx-{len(qs)}", qs, prefix=QUERY_PREFIX,
                                 dtype=torch.float16), dtype=np.float32)
    X = aug_matrix(tok, qs, pre, V, bmap)
    print(f"  augmented bag {X.shape} nnz={X.nnz:,} ({time.time()-t0:.0f}s)", flush=True)

    W0u = get_init("teacher", pre, vocab=V)
    W0 = np.vstack([W0u, np.zeros((k, W0u.shape[1]), dtype=np.float32)])
    W = solve_ridge(X, Y, W0, lam)
    del X, Y, W0

    per_aug = eval_w(W, tok, pre, V, bmap)
    per_uni = eval_w(W[:V], tok, pre, V, {})
    mac = lambda per: float(np.mean([np.mean(list(per[c].values())) for c in COMPONENTS]))
    r = boot.paired(per_aug, per_uni, alternative="greater")
    out = {"k": k, "lam": lam, "n_queries": len(qs),
           "macro_bigram_augmented": round(mac(per_aug), 4),
           "macro_unigram_rows_of_same_solve": round(mac(per_uni), 4),
           "aug_minus_uni_same_solve": {q: r[q] for q in ("delta", "ci95", "resolved")},
           "artifact_cost_mb_fp16": round(k * W.shape[1] * 2 / 1e6, 1),
           "encoder": spec_tag(), "seconds": round(time.time() - t0, 1),
           "_note": "closed-form capacity probe; the unigram comparator is the SAME solve's "
                    "unigram block (isolates the bigram rows' contribution at eval); compare "
                    "also against the committed unigram-only ridge macro in "
                    "results/m7_stage0_ridge_stella.json"}
    (REPO / "results" / f"m7_bigram_probe_k{k}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1), flush=True)


if __name__ == "__main__":
    main(k=int(sys.argv[1]) if len(sys.argv) > 1 else 5000,
         lam=float(sys.argv[2]) if len(sys.argv) > 2 else 0.01,
         n_queries=int(sys.argv[3]) if len(sys.argv) > 3 else None)
