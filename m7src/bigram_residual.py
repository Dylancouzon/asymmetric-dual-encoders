"""Capacity lever #1, adoption run: bigram rows residual-fitted onto the TRAINED winner.

The probe (bigram_probe.py) established bigram capacity on a closed-form ridge table, proxy-3.
Adoption is judged per the pre-registered protocol in m7/LEDGER.md (2026-08-27): freeze the
winner's released unigram rows, fit only the K bigram rows by residual ridge on the TRAIN
queries (one global scalar s absorbs the trained table's scale — a global scalar is absorbable,
so this adds no capacity by itself), then score the augmented table against the identical
winner WITHOUT bigram rows on the FULL pinned dev suite, release shape, paired.

Adopt iff signflip p < 0.05 AND paired CI > 0; the int8 candidate must independently clear
CI > 0 against the int8 winner. lam = 0.01 carried from the probe, not tuned.

The Gram here is only K x K (0.8 GB fp64 at K=10,000) — the V x V solve never happens.
"""
import json
import sys
import time

import numpy as np
import scipy.sparse as sp
import torch

import dev_eval
import mix
from _paths import REPO, WORK
from bigram_probe import SPECIALS, aug_matrix, top_bigrams
from evalkit import per_query_ndcg, topk_ids_scores
from stage0_ridge import bag_matrix
from table import NO_PREFIX, dequantize_int8, get_tokenizer, load_table, quantize_int8, read_meta
from teacher import QUERY_PREFIX, encode_cached
import boot

WINNER = WORK / "runs" / "s2w-1e3-s1000.release.npz"


def rss_gb():
    return int(open("/proc/self/status").read().split("VmRSS:")[1].split()[0]) / 1e6


def eval_variants(variants, tok, pre, V, components):
    """All row-matrix variants scored per component, components OUTER: each component's corpus
    JSON is parsed once total instead of once per variant — the four-pass version of this loop
    ratcheted RSS until the kernel OOM-killed it (2026-08-27). `variants` maps
    tag -> (W, bmap); returns tag -> {comp: {qid: ndcg}}. The bag/aug matrix path with an
    empty bmap is exactly the released table's forward (sum/len then normalize; folded
    weights live in the rows), verified against the gate numbers."""
    per = {tag: {} for tag in variants}
    for comp in components:
        doc_ids, _, q_ids, q_texts, qrels, dv = dev_eval.doc_vecs(comp)
        Xu = bag_matrix(tok, q_texts, pre, V)
        Xa = None
        for tag, (W, bmap) in variants.items():
            if bmap and Xa is None:
                Xa = aug_matrix(tok, q_texts, pre, V, bmap)
            Xq = Xa if bmap else Xu
            qv = np.asarray(Xq @ W[:Xq.shape[1]], dtype=np.float32)
            qv /= np.clip(np.linalg.norm(qv, axis=1, keepdims=True), 1e-9, None)
            run = topk_ids_scores(qv, dv, doc_ids, k=100,
                                  chunk=dev_eval.CHUNK.get(comp, 200_000), qids=q_ids)
            per[tag][comp] = per_query_ndcg(run, qrels)
        print(f"    eval {comp} done ({', '.join(variants)}), rss {rss_gb():.1f} GB", flush=True)
    return per


def macro(per):
    return float(np.mean([np.mean(list(v.values())) for v in per.values()]))


def stats(per_a, per_b, alternative="greater"):
    r = boot.paired(per_a, per_b, alternative=alternative)
    sf = boot.signflip(per_a, per_b, alternative=alternative)
    return {"delta": r["delta"], "ci95": r["ci95"], "resolved": r["resolved"],
            "signflip_p": sf["p"], "signflip_p_str": sf["p_str"]}


FIT = WORK / "runs" / "bigram_residual_fit.npz"


def main(k, lam=0.01, smoke=False, proxy=False):
    t0 = time.time()
    tok = get_tokenizer()
    V = tok.vocab_size
    meta = read_meta(WINNER)
    pre = NO_PREFIX
    assert meta["preproc"]["prefix"] == pre.prefix and meta["weights_folded"], meta
    Wu = load_table(WINNER, variant="fp16", device="cpu").rows.detach().numpy().astype(np.float32)

    qs = mix.query_texts(train_only=True)
    if smoke:
        rng = np.random.default_rng(0)
        qs = [qs[i] for i in rng.choice(len(qs), size=20000, replace=False)]
    print(f"bigram residual fit: K={k} lam={lam} on {len(qs):,} TRAIN queries, "
          f"winner={WINNER.name}", flush=True)
    if FIT.exists() and not smoke and lam == 0.01:
        z = np.load(FIT, allow_pickle=True)
        assert int(z["k"]) == k and float(z["lam"]) == lam, dict(k=int(z["k"]), lam=float(z["lam"]))
        Wb, s = z["wb"].astype(np.float32), float(z["s"])
        bmap = {tuple(bg): j for j, bg in enumerate(z["bigrams"])}
        print(f"  fit loaded from {FIT.name}: K={k} s={s:.4f}", flush=True)
    else:
        bigrams = top_bigrams(tok, qs, pre, k)
        bmap = {bg: j for j, bg in enumerate(bigrams)}
        Y = np.asarray(encode_cached(f"stage0-qtargets-pfx-{len(qs)}", qs, prefix=QUERY_PREFIX,
                                     dtype=torch.float16), dtype=np.float32)
        Xu = bag_matrix(tok, qs, pre, V)
        U = np.asarray(Xu @ Wu, dtype=np.float32)
        s = float((U * Y).sum() / np.maximum((U * U).sum(), 1e-12))
        del Xu
        Xb = aug_matrix(tok, qs, pre, V, bmap)[:, V:].tocsr()
        G = (Xb.T @ Xb).toarray()
        G[np.diag_indices_from(G)] += lam
        rhs = Xb.T @ (Y - s * U).astype(np.float64)
        import scipy.linalg as sla
        posv, = sla.get_lapack_funcs(("posv",), (G, rhs))
        _, Wb, info = posv(G, np.asfortranarray(rhs), lower=1, overwrite_a=True, overwrite_b=True)
        assert info == 0, info
        Wb = Wb.astype(np.float32)
        del G, rhs, Xb, U, Y
        if not smoke and lam == 0.01:
            np.savez(FIT, wb=Wb, s=s, k=k, lam=lam, bigrams=np.array(list(bmap), dtype=np.int64))
        print(f"  fitted K={k} bigram rows, global scale s={s:.4f} ({time.time()-t0:.0f}s), "
              f"rss {rss_gb():.1f} GB", flush=True)
    n_train = len(qs)
    del qs  # 561K strings; the eval phase must not carry them (OOM-killed 2026-08-27)

    # The shipped shape: relative scale between blocks is what matters; fold s into the
    # unigram block (rows are where absorbable scalars go).
    Waug = np.vstack([s * Wu, Wb])
    comps = ["nq-250k", "cqadup-programmers", "cqadup-physics"] if (smoke or proxy) \
        else dev_eval.dev_components()
    per = eval_variants(
        {"winner": (Wu, {}), "aug": (Waug, bmap),
         "winner-int8": (dequantize_int8(*quantize_int8(Wu)), {}),
         "aug-int8": (dequantize_int8(*quantize_int8(Waug)), bmap)},
        tok, pre, V, comps)
    per_base, per_aug = per["winner"], per["aug"]
    per_base8, per_aug8 = per["winner-int8"], per["aug-int8"]

    out = {"k": k, "lam": lam, "n_queries": n_train, "winner": WINNER.name,
           "global_scale_s": round(s, 6),
           "macro_winner": round(macro(per_base), 4),
           "macro_winner_plus_bigrams": round(macro(per_aug), 4),
           "macro_winner_int8": round(macro(per_base8), 4),
           "macro_winner_plus_bigrams_int8": round(macro(per_aug8), 4),
           "per_component_macro": {c: {"winner": round(float(np.mean(list(per_base[c].values()))), 4),
                                       "aug": round(float(np.mean(list(per_aug[c].values()))), 4)}
                                   for c in comps},
           "aug_vs_winner": stats(per_aug, per_base),
           "aug_vs_winner_int8": stats(per_aug8, per_base8),
           "artifact_cost_mb_fp16": round(k * Waug.shape[1] * 2 / 1e6, 1),
           "components": comps, "seconds": round(time.time() - t0, 1),
           "_protocol": "pre-registered in m7/LEDGER.md 2026-08-27: adopt iff signflip p<0.05 "
                        "AND paired CI>0 on the full pinned dev suite, release shape; int8 "
                        "independently CI>0 vs int8 winner; lam fixed at 0.01"}
    if smoke or proxy:
        out["_smoke"] = ("20k-query subsample, " if smoke else "full-query fit, ") + \
            "proxy components only — NOT an adoption number"
        print(json.dumps(out, indent=1), flush=True)
        return
    (REPO / "results" / f"m7_bigram_residual_k{k}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1), flush=True)


if __name__ == "__main__":
    main(k=int(sys.argv[1]), lam=float(sys.argv[2]) if len(sys.argv) > 2 else 0.01,
         smoke=len(sys.argv) > 3 and sys.argv[3] == "--smoke",
         proxy=len(sys.argv) > 3 and sys.argv[3] == "--proxy")
