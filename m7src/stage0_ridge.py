"""Stage 0, closed form: the MSE-optimal flat-weight bag-of-tokens approximation of the
frozen teacher's query encoder.

This is an exact upper bound on what flat-weight distillation (objective B under squared
loss) can achieve, and it costs one linear solve. It answers the central structural question
-- can a bag of token vectors land where a frozen bge-base query lands? -- before any
gradient step is spent.

  minimize_W  || X W - Y ||_F^2 + lambda || W - W0 ||_F^2
  X  (n x V) row-normalized token counts (the flat-weight mean operator)
  Y  (n x d) teacher query vectors for the same queries
  W0 the init (regularizing toward it is also the low-update-row rule from the mandate)

lambda is selected on the dev macro only.
"""
import json
import sys
import time

import numpy as np
import scipy.sparse as sp
import torch

import dev_eval
import mix
from _paths import WORK
from init_table import get_init
from table import NO_PREFIX, WITH_PREFIX, Preproc, QueryTable, get_tokenizer, tokenize
from teacher import QUERY_PREFIX, encode_cached

OUT = WORK / "stage0"
OUT.mkdir(parents=True, exist_ok=True)
LAMBDAS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]


def bag_matrix(tok, texts, pre: Preproc, V):
    """Row-normalized count matrix: row i is the flat-weight mean operator for query i."""
    indptr, indices, data = [0], [], []
    B = 4096
    for lo in range(0, len(texts), B):
        for ids in tokenize(tok, texts[lo:lo + B], pre):
            u, c = np.unique(np.array(ids, dtype=np.int64), return_counts=True)
            indices.append(u)
            data.append(c.astype(np.float64) / max(1, len(ids)))
            indptr.append(indptr[-1] + len(u))
    X = sp.csr_matrix((np.concatenate(data), np.concatenate(indices), np.array(indptr)),
                      shape=(len(texts), V))
    return X


def solve_ridge(X, Y, W0, lam):
    """Normal equations in float64. V=30522 -> the Gram is 7.45 GB fp64, so it is built once
    per lambda and factored in place (assume_a='pos', overwrite_a) to avoid a second copy --
    the M4 session lost a machine to a peak-RAM surprise, so peak here is one Gram, not two."""
    import scipy.linalg as sla
    Gs = X.T @ X
    # (i,j) is nonzero only when tokens i and j co-occur in some query, so this stays far from
    # the 931M-entry dense worst case -- but check, because densifying a dense-ish sparse Gram
    # would need ~19 GB on a 25 GB box.
    print(f"    gram nnz={Gs.nnz:,} ({Gs.nnz/ (X.shape[1]**2):.3%} dense, "
          f"{Gs.nnz*12/1e9:.2f} GB sparse -> {X.shape[1]**2*8/1e9:.2f} GB dense fp64)", flush=True)
    G = Gs.toarray().astype(np.float64)
    del Gs
    rhs = (X.T @ Y).astype(np.float64) + lam * W0.astype(np.float64)
    G[np.diag_indices_from(G)] += lam
    W = sla.solve(G, rhs, assume_a="pos", overwrite_a=True, overwrite_b=True)
    del G, rhs
    return np.ascontiguousarray(W, dtype=np.float32)


def chunked_train_cos(X, W, Y, chunk=50_000):
    """Mean cosine between the fitted bag vector and the teacher target, without materializing
    a (n_queries x 768) float64 product."""
    tot, n = 0.0, X.shape[0]
    Xf = X.astype(np.float32)
    for lo in range(0, n, chunk):
        f = (Xf[lo:lo + chunk] @ W).astype(np.float32)
        f /= np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-12)
        tot += float((f * Y[lo:lo + chunk]).sum())
    return tot / n


def overlap_at_10(model, pre, tok, components):
    """Retrieval agreement with the teacher: mean |top-10 student ∩ top-10 teacher| / 10.
    Reported separately from embedding agreement (train_cos), as the mandate requires."""
    from evalkit import topk_ids_scores
    out = {}
    for c in components:
        doc_ids, _, q_ids, q_texts, _, dv = dev_eval.doc_vecs(c)
        tqv = np.asarray(encode_cached(f"dev-{c}-queries-pfx", q_texts, prefix=QUERY_PREFIX,
                                       dtype=torch.float16, verbose=False), dtype=np.float32)
        chunk = dev_eval.CHUNK.get(c, 200_000)
        a = topk_ids_scores(model.encode(q_texts, pre, tok=tok), dv, doc_ids, k=10, chunk=chunk, qids=q_ids)
        b = topk_ids_scores(tqv, dv, doc_ids, k=10, chunk=chunk, qids=q_ids)
        out[c] = float(np.mean([len(set(a[q]) & set(b[q])) / 10.0 for q in q_ids]))
    return out


def main(n_queries=1_000_000, init_kind="teacher", pre=None, target_prefix=True,
         components=("nq-250k", "cqadup-programmers", "cqadup-physics")):
    pre = pre or NO_PREFIX
    tok = get_tokenizer()
    V = tok.vocab_size
    qs = mix.query_texts(train_only=True)
    rng = np.random.default_rng(0)
    if len(qs) > n_queries:
        qs = [qs[i] for i in rng.choice(len(qs), size=n_queries, replace=False)]
    print(f"ridge fit on {len(qs):,} TRAIN queries | init={init_kind} | bag-preproc={pre} | "
          f"target={'prefixed' if target_prefix else 'bare'} teacher query vectors", flush=True)

    t0 = time.time()
    Y = np.asarray(encode_cached(f"stage0-qtargets-{'pfx' if target_prefix else 'nopfx'}-{len(qs)}",
                                 qs, prefix=QUERY_PREFIX if target_prefix else "",
                                 dtype=torch.float16), dtype=np.float32)
    print(f"  teacher targets: {Y.shape} in {time.time()-t0:.0f}s", flush=True)
    X = bag_matrix(tok, qs, pre, V)
    cov = float((X.getnnz(axis=0) > 0).mean())
    print(f"  bag matrix {X.shape} nnz={X.nnz:,} | vocab coverage on TRAIN queries {cov:.3f}", flush=True)
    ov = None
    W0 = get_init(init_kind, pre, vocab=V)

    results = {}
    for lam in LAMBDAS:
        t0 = time.time()
        wp = OUT / f"ridge-{init_kind}-{pre.fingerprint()}-lam{lam}.npy"
        if wp.exists():
            print(f"  lam={lam:<7g} reusing the solved table on disk", flush=True)
            W = np.load(wp)
        else:
            W = solve_ridge(X, Y, W0, lam)
        cos = chunked_train_cos(X, W, Y)
        m = QueryTable(W, learned_weights=False).to("cuda").eval()
        per = dev_eval.eval_table(m, pre, components=list(components), tok=tok)
        ov = overlap_at_10(m, pre, tok, list(components))
        macro, means = dev_eval.report(
            per, f"  lam={lam:<7g} train-cos={cos:.4f} ov@10={np.mean(list(ov.values())):.3f}")
        results[str(lam)] = {"train_cos": cos, "macro": macro, "per_component": means,
                            "overlap_at_10": ov, "solve_s": round(time.time() - t0, 1)}
        np.save(wp, W)
        del m, W
        torch.cuda.empty_cache()
    (OUT / f"ridge-{init_kind}-{pre.fingerprint()}.json").write_text(json.dumps(
        {"n_queries": len(qs), "init": init_kind, "preproc": pre.fingerprint(),
         "target_prefix": target_prefix, "vocab_coverage_train_queries": cov,
         "results": results}, indent=1))
    best = max(results, key=lambda k: results[k]["macro"])
    print(f"\nbest lambda={best}: dev macro {results[best]['macro']:.4f}", flush=True)
    return results


if __name__ == "__main__":
    kind = sys.argv[1] if len(sys.argv) > 1 else "teacher"
    pre = {"noprefix": NO_PREFIX, "prefix": WITH_PREFIX}[sys.argv[2] if len(sys.argv) > 2 else "noprefix"]
    main(init_kind=kind, pre=pre)
