"""E14-PRE -- does the document tower hold reachable structure its read-out discards?

WHY THIS RUNS BEFORE THE FLOOR. `E14-LORA` costs six full chains plus a joint-training path that
does not exist yet -- twelve hours -- and its case rests on an ANALOGY: LightRetriever trains its
document encoder and beat us, we do not. Before paying for that, ask the cheap version of the same
question. `D2-PRE` just saved five chains this way.

PROBE 1 -- READ-OUT. stella is `Transformer -> Pooling(mean) -> Dense_1024(identity) -> normalize`.
Vary that read-out over four pre-named arms, all 1024-d and therefore comparable, and make the WHOLE
system consistent for each: teacher query targets re-encoded under that read-out, the table
re-solved in closed form against them, dev documents re-encoded under the same read-out. Varying the
document read-out ALONE would be a confound, not a probe -- queries and documents would simply be in
different spaces, and the arm would lose for a reason that has nothing to do with reachability.
Every arm uses `W0 = 0` so none inherits the incumbent's init.

PROBE 2 -- ORACLE LINEAR DOC-SIDE MAP. Score documents as `normalize(M d)` with R0's shipped table
fixed. `M` is fitted so that the TABLE's query vectors behave like the TEACHER's -- least squares
`M^T q_table ~ q_teacher` -- honestly on train queries, and again as an ORACLE on the scored dev
queries themselves. The renormalization is load-bearing: without it `(M d).q = d.(M^T q)` is a
linear map of the query and therefore ABSORBABLE into the table rows, so it would measure nothing
new (CODEMAP pitfall 20 / `m7_absorb_check`: rank agreement 1.000 unnormalized, 0.000 normalized).

THE SCOPE LIMIT, STATED BEFORE THE RUN AND BINDING. **A NO SURVIVOR HERE DOES NOT CLOSE
`E14-LORA`.** A LoRA changes what the tower COMPUTES; probe 1 changes only how its output is READ
and probe 2 only how that output is MAPPED. That is exactly why this ledger refuses to let
`E14-HEAD` close `E14-LORA`, and it applies here with equal force. This screen decides whether to
SPEND, not what is true.
"""
import argparse
import gc
import json
import sys
import time

import numpy as np

import m8base
import blockcg
import probe_guard
from d2_pre import bag_matrix, counts_of, encode, ids_of, load_incumbent, r0_denominator, tok_pre

RESULTS = m8base.RESULTS
OUT = RESULTS / "m8_e14_pre.json"
OOD = ("cqadup-programmers", "cqadup-physics")
LAMBDAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
BAR = 0.0040

# The four read-outs, named before anything was encoded. `dense` selects stella's published
# post-pooling Dense_1024; dropping it is a different model, which is the point of the arm.
READOUTS = (("mean+dense", "mean", True),      # the incumbent: stella's published pipeline
            ("cls+dense", "cls", True),
            ("mean_no_dense", "mean", False),
            ("cls_no_dense", "cls", False))
INCUMBENT = "mean+dense"


def encode_readout(texts, pooling, use_dense, prefix="", batch_tokens=16384, verbose=False,
                   dtype=None):
    """`teacher.encode` with the read-out passed in rather than read off the Spec.

    m7src is frozen (G3), and `teacher.encode` takes its pooling from `encoders.active()`, so this
    mirrors its length-bucketed loop and reuses its own helpers -- `load_teacher`, `encode_batch`,
    `_order_by_length`, `load_post_dense` -- rather than reimplementing the arithmetic that decides
    what a document vector IS.
    """
    import torch
    import encoders
    from teacher import (TEACHER, TEACHER_REV, _order_by_length, encode_batch, load_post_dense,
                         load_teacher)
    spec = encoders.active()
    device = m8base.device()
    # fp16 WEIGHTS, matching every cached encode in this project. `encode_cached` is called with
    # dtype=torch.float16 everywhere -- dev document vectors, pool vectors, query targets -- and
    # that dtype goes to `load_teacher`, so it is the dtype the INCUMBENT document index was built
    # with. Loading fp32 here would make the "incumbent" read-out arm not the incumbent, and it
    # also halved throughput and pushed a 10 GB card to 9,970 MiB. `conformance()` checks the
    # match rather than trusting this comment.
    tok, model = load_teacher(TEACHER, TEACHER_REV, dtype or torch.float16, device)
    dense = load_post_dense(spec, device) if use_dense else None
    order, n_tok = _order_by_length(tok, [prefix + t for t in texts], 512)
    dim = model.config.hidden_size if dense is None else int(dense[0].shape[0])
    out = np.empty((len(texts), dim), dtype=np.float32)
    i, t0, done = 0, time.time(), 0
    while i < len(order):
        longest, j = 0, i
        while j < len(order):
            L = max(longest, n_tok[order[j]])
            if (j - i + 1) * L > batch_tokens and j > i:
                break
            longest, j = L, j + 1
        idx = order[i:j]
        out[idx] = encode_batch(tok, model, [prefix + texts[k] for k in idx], 512, device,
                                pooling=pooling, dense=dense)
        done += len(idx)
        if verbose and done % 50_000 < len(idx):
            print(f"      {done:,}/{len(texts):,} @ {done/(time.time()-t0):.0f}/s", flush=True)
        i = j
    return out


def conformance(n=2000):
    """`encode_readout` at the INCUMBENT read-out must reproduce the cached document vectors that
    the shipped index is built from. Without this the whole screen could compare four arms none of
    which is the incumbent -- a wrong number rather than a crash, which is the class that has cost
    this project the most (CODEMAP pitfalls 15, 19)."""
    import dev_eval
    c = OOD[0]
    _, doc_texts, _, _, _, dv = dev_eval.doc_vecs(c)
    mine = encode_readout(list(doc_texts)[:n], "mean", True)
    ref = np.asarray(dv[:n], dtype=np.float32)
    d = float(np.abs(mine - ref).max())
    cos = float(np.mean((mine * ref).sum(1) / np.maximum(
        np.linalg.norm(mine, axis=1) * np.linalg.norm(ref, axis=1), 1e-12)))
    print(f"  conformance vs cached {c} docs: max|d|={d:.2e} mean cos={cos:.8f}", flush=True)
    return {"max_abs": d, "mean_cosine": cos, "n": n, "pass": bool(cos > 0.9999)}


def score_against(qv, comp, doc_vecs):
    """nDCG@10 for given query vectors against given document vectors."""
    import dev_eval
    from evalkit import score
    doc_ids, _, q_ids, _, qrels, _ = dev_eval.doc_vecs(comp)
    return score(qv, list(q_ids), doc_vecs, doc_ids, qrels,
                 chunk=dev_eval.CHUNK.get(comp, 200_000))


def probe1(tok, pre, fit_texts, X, dev_texts, dev_ids, cache):
    """Four read-outs; for each, a consistent system solved and scored end to end."""
    import dev_eval
    out = {}
    for name, pooling, use_dense in READOUTS:
        t0 = time.time()
        Y = cache.get(("q", name))
        if Y is None:
            print(f"  [{name}] encoding {len(fit_texts):,} train-query targets", flush=True)
            from teacher import QUERY_PREFIX
            Y = encode_readout(fit_texts, pooling, use_dense, prefix=QUERY_PREFIX, verbose=True)
            cache[("q", name)] = Y
        dv = {}
        for c in OOD:
            k = ("d", name, c)
            if k not in cache:
                print(f"  [{name}] encoding {c} documents", flush=True)
                _, doc_texts, _, _, _, _ = dev_eval.doc_vecs(c)
                cache[k] = encode_readout(list(doc_texts), pooling, use_dense, verbose=True)
                del doc_texts
                gc.collect()
            dv[c] = cache[k]
        W0 = np.zeros((X.shape[1], Y.shape[1]), dtype=np.float32)
        best = (None, -1.0)
        curve = {}
        for lam in LAMBDAS:
            W, _ = blockcg.block_cg_ridge(X, Y, W0, lam, device=m8base.device())
            W = W.astype(np.float32)
            m = float(np.mean([
                np.mean(list(score_against(encode([counts_of(x) for x in dev_ids[c]], W),
                                           c, dv[c]).values())) for c in OOD]))
            curve[lam] = m
            if m > best[1]:
                best = (lam, m)
            del W
            m8base.empty_cache()
        out[name] = {"argmax_lambda": best[0], "ood_macro": best[1], "lambda_curve": curve,
                     "argmax_at_grid_boundary": best[0] in (min(LAMBDAS), max(LAMBDAS)),
                     "dim": int(Y.shape[1]), "seconds": round(time.time() - t0, 1)}
        print(f"  [{name:14s}] ood={best[1]:.6f} at lam={best[0]:g}"
              f"{'  BOUNDARY' if out[name]['argmax_at_grid_boundary'] else '  interior'} "
              f"({time.time()-t0:.0f}s)", flush=True)
    return out


def probe2(tok, pre, fit_texts, X, dev_texts, dev_ids, cache):
    """The oracle linear doc-side map, fitted honestly and then as an oracle."""
    import dev_eval
    from teacher import QUERY_PREFIX

    W_inc, _ = load_incumbent()
    Y_train = cache[("q", INCUMBENT)]                       # teacher query vecs, incumbent read-out
    q_tab_train = encode([counts_of(x) for x in ids_of(tok, fit_texts, pre)], W_inc)

    def fit_map(q_tab, q_tea, lam=1e-2):
        """least squares  M^T q_tab ~ q_tea  ->  returns M (d x d)."""
        A = q_tab.astype(np.float64)
        B = q_tea.astype(np.float64)
        G = A.T @ A + lam * np.eye(A.shape[1])
        Mt = np.linalg.solve(G, A.T @ B)                    # (d, d): M^T
        return Mt.T.astype(np.float32)

    res = {}
    base = {}
    for c in OOD:
        _, _, _, _, _, dv = dev_eval.doc_vecs(c)
        qv = encode([counts_of(x) for x in dev_ids[c]], W_inc)
        base[c] = float(np.mean(list(score_against(qv, c, np.asarray(dv, dtype=np.float32))
                                     .values())))
    res["baseline_ood_macro"] = float(np.mean(list(base.values())))
    res["baseline_per_component"] = base

    fits = {"honest_train_fit": (q_tab_train, Y_train)}
    # the ORACLE fit: the same regression fitted on the DEV queries it is scored on
    q_tab_dev, q_tea_dev = [], []
    for c in OOD:
        _, _, _, q_texts, _, _ = dev_eval.doc_vecs(c)
        q_tab_dev.append(encode([counts_of(x) for x in dev_ids[c]], W_inc))
        q_tea_dev.append(encode_readout(list(q_texts), "mean", True, prefix=QUERY_PREFIX))
    fits["oracle_dev_fit"] = (np.concatenate(q_tab_dev), np.concatenate(q_tea_dev))

    for tag, (a, b) in fits.items():
        M = fit_map(a, b)
        per = {}
        for c in OOD:
            _, _, _, _, _, dv = dev_eval.doc_vecs(c)
            d2 = np.asarray(dv, dtype=np.float32) @ M.T
            d2 /= np.maximum(np.linalg.norm(d2, axis=1, keepdims=True), 1e-12)
            qv = encode([counts_of(x) for x in dev_ids[c]], W_inc)
            per[c] = float(np.mean(list(score_against(qv, c, d2).values())))
            del d2
            gc.collect()
        m = float(np.mean(list(per.values())))
        res[tag] = {"ood_macro": m, "per_component": per,
                    "delta_vs_baseline": m - res["baseline_ood_macro"]}
        print(f"  [probe2 {tag:16s}] ood={m:.6f} delta={m - res['baseline_ood_macro']:+.6f}",
              flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-fit", type=int, default=None)
    a = ap.parse_args()
    t0 = time.time()

    import dev_eval
    tok, pre = tok_pre()
    V = len(tok)
    fit_texts = json.loads((m8base.WORK / "m8_trainq_texts.json").read_text())
    if a.smoke or a.max_fit:
        n = a.max_fit or 4000
        sel = np.random.default_rng(0).choice(len(fit_texts), n, replace=False)
        fit_texts = [fit_texts[i] for i in sel]
    fit_ids = ids_of(tok, fit_texts, pre)
    X = bag_matrix([counts_of(x) for x in fit_ids], r0_denominator(fit_ids), V)
    dev_texts, dev_ids = {}, {}
    for c in OOD:
        _, _, _, q_texts, _, _ = dev_eval.doc_vecs(c)
        dev_texts[c] = list(q_texts)
        dev_ids[c] = ids_of(tok, dev_texts[c], pre)
    print(f"fit {len(fit_texts):,} queries, X nnz={X.nnz:,} [{time.time()-t0:.0f}s]", flush=True)

    conf = conformance()
    if not conf["pass"]:
        raise SystemExit(f"encode_readout does not reproduce the cached incumbent index: {conf}. "
                         "Every arm would be measured in a frame that is not the shipped one.")
    cache = {}
    p1 = probe1(tok, pre, fit_texts, X, dev_texts, dev_ids, cache)
    p2 = probe2(tok, pre, fit_texts, X, dev_texts, dev_ids, cache)

    inc = p1[INCUMBENT]["ood_macro"]
    d1 = {k: v["ood_macro"] - inc for k, v in p1.items() if k != INCUMBENT}
    best_read = max(d1, key=lambda k: d1[k])
    d2 = max(p2["honest_train_fit"]["delta_vs_baseline"], p2["oracle_dev_fit"]["delta_vs_baseline"])
    authorise = bool(d1[best_read] >= BAR or d2 >= BAR)
    out = {
        "_note": __doc__.strip().splitlines()[0],
        "probe1_readout": p1, "probe1_delta_vs_incumbent": d1, "probe1_best": best_read,
        "probe2_oracle_map": p2, "probe2_best_delta": d2,
        "bar": BAR,
        "authorise_E14_LORA": authorise,
        "verdict": ("ROUTE -- a measured mechanism supports spending E14-LORA's chains"
                    if authorise else
                    "NO MEASURED MECHANISM -- do not spend E14-LORA's twelve hours"),
        "scope_limit_binding": (
            "A NO SURVIVOR HERE DOES NOT CLOSE E14-LORA. A LoRA changes what the tower COMPUTES; "
            "probe 1 changes only how its output is READ and probe 2 only how that output is "
            "MAPPED. That is the identical limitation for which this ledger refuses to let "
            "E14-HEAD close E14-LORA. This screen decides whether to SPEND, not what is true."),
        "conformance": conf,
        "smoke": bool(a.smoke), "n_fit": len(fit_texts),
        "seconds": round(time.time() - t0, 1),
    }
    print(json.dumps({"probe1_delta_vs_incumbent": d1, "probe2_best_delta": d2,
                      "verdict": out["verdict"]}, indent=2))
    dest = (RESULTS / "m8_e14_pre.SMOKE.json") if a.smoke else OUT
    if a.smoke:
        dest.write_text(json.dumps(out, indent=2, default=str))
    else:
        probe_guard.write_result(dest, out, "E14-PRE")
    print(f"wrote {dest}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
