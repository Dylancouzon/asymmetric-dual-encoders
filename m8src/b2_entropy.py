"""B2: is the distillation KL term degenerate? (LEDGER §9, hypothesis H2.)

THE HYPOTHESIS. Objective A's KL term asks the student's distribution over a candidate set to match
the teacher's. The candidate set is the query's own positive plus `kl_k - 1 = 31` distractors drawn
UNIFORMLY from a 2M-row bank, and the temperature is 0.02 (`m7src/train.py`, the `cfg.kl_weight >
0` branch). A uniform draw from two million documents will essentially never produce anything
competitive with the true positive, and dividing by 0.02 multiplies every score gap by fifty. If
the resulting teacher distribution is one-hot to within ~1e-4 nats, the KL term is carrying almost
no information: it is asking the student to reproduce a delta function it already reproduces, and
the gradient it contributes is nearly zero for reasons that have nothing to do with the student.

WHAT THIS MEASURES, and what it is allowed to do. The entropy of the teacher's candidate
distribution, per query, under the recipe's own sampler and under a harder one (distractors drawn
from the teacher's own top-200 for that query). It is REGISTERED AS A DIAGNOSTIC: its bar is
"descriptive; adopts nothing and may not adopt anything", and its only registered consequence is
that it may or may not trigger the separately registered `R-LIST` arm. It cannot change the recipe
by itself, and a low entropy here is not evidence that a listwise objective would be better -- only
that the current term is not doing the work its presence implies.

Entropy is reported in NATS, against the ceiling ln(32) = 3.466 that a uniform distribution over
the candidate set would give, so "degenerate" has a scale rather than a vibe.
"""
import argparse
import json
import sys
import time

import numpy as np

import m8base
import probe_guard

RESULTS = m8base.RESULTS
OUT = RESULTS / "m8_b2_entropy.json"
KL_K = 32
TEMP = 0.02
TOPK_POOL = 200


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=int, default=4000)
    ap.add_argument("--bank", type=int, default=2_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    import torch
    import pool as poolmod
    import train
    from train import Cfg

    from teacher import QUERY_PREFIX, encode_cached

    t0 = time.time()
    # pool.build -> (PoolIndex, memmap (N, dim) fp16, meta); build_arrays -> (q_texts, pos_idx,
    # hn_idx, src_id, srcs). Teacher query vectors are NOT among them -- train.run encodes them
    # separately, and the cache key is the one it will ask for.
    index, pool_vecs, _meta = poolmod.build()
    cfg = Cfg()
    q_texts, pos_idx, _hn, _src, _srcs = train.build_arrays(cfg, index)
    tq = np.asarray(encode_cached(f"trainq-{len(q_texts)}", q_texts, prefix=QUERY_PREFIX,
                                  dtype=torch.float16, verbose=False), dtype=np.float32)
    print(f"setup: {len(q_texts):,} train queries, pool {pool_vecs.shape}, tq {tq.shape} "
          f"({time.time()-t0:.0f}s)", flush=True)

    rng = np.random.default_rng(a.seed)
    n = min(a.queries, len(q_texts))
    qsel = rng.choice(len(q_texts), n, replace=False)
    dev = m8base.device()

    # The bank, exactly as training builds it: a contiguous prefix of the pool.
    nb = min(a.bank, pool_vecs.shape[0])
    bank = torch.from_numpy(np.ascontiguousarray(pool_vecs[:nb])).to(dev).half()
    tqs = torch.from_numpy(np.ascontiguousarray(tq[qsel])).to(dev).float()
    p_i = np.array([pos_idx[i][0] for i in qsel])
    pos_v = torch.from_numpy(np.ascontiguousarray(pool_vecs[p_i])).to(dev).float()

    def entropies(cand):
        """cand: (n, K, d). -> per-query entropy of softmax(teacher . cand / TEMP), in nats."""
        logits = torch.einsum("bd,bkd->bk", tqs, cand) / TEMP
        p = torch.softmax(logits, 1)
        h = -(p * torch.log(p.clamp_min(1e-30))).sum(1)
        return h.detach().cpu().numpy(), p.max(1).values.detach().cpu().numpy()

    out = {"_note": __doc__.strip().splitlines()[0],
           "status": "DIAGNOSTIC. Adopts nothing and may not adopt anything (LEDGER 9). Its only "
                     "registered consequence is whether it triggers the separately registered "
                     "R-LIST arm.",
           "setting": {"n_queries": int(n), "kl_k": KL_K, "temp": TEMP, "bank_rows": int(nb),
                       "entropy_ceiling_nats": float(np.log(KL_K)), "seed": a.seed},
           "arms": {}}

    # ---- arm 1: the recipe's own sampler -- uniform from the bank -------------------------
    sel = torch.from_numpy(rng.integers(0, nb, n * (KL_K - 1))).to(dev)
    dist = bank.index_select(0, sel).float().view(n, KL_K - 1, -1)
    cand = torch.cat([pos_v.unsqueeze(1), dist], 1)
    h_u, pmax_u = entropies(cand)
    del dist, cand
    m8base.empty_cache()

    # ---- arm 2: distractors from the teacher's own top-200 --------------------------------
    # Chunked over queries: the full (n x nb) score matrix would be 4,000 x 2,000,000 floats.
    # The scoring matmul stays in HALF. `bank.float()` inside this loop would have materialized
    # an 8 GB fp32 copy of the 2M-row bank on every chunk -- on a 10 GB card, once. Half is ample
    # for choosing a top-200 set, and the chunk is sized so the (CH x 2M) score block is ~0.5 GB.
    top = np.empty((n, TOPK_POOL), dtype=np.int64)
    tqs_h = tqs.half()
    CH = 128
    for lo in range(0, n, CH):
        hi = min(lo + CH, n)
        sc = tqs_h[lo:hi] @ bank.T
        top[lo:hi] = torch.topk(sc, TOPK_POOL, dim=1).indices.cpu().numpy()
        del sc
        if lo == 0:
            print(f"  top-200 scoring: {CH}/{n} in {time.time()-t0:.0f}s from start", flush=True)
    pick = np.array([rng.choice(TOPK_POOL, KL_K - 1, replace=False) for _ in range(n)])
    hard_idx = np.take_along_axis(top, pick, 1)
    dist_h = bank.index_select(0, torch.from_numpy(hard_idx.ravel()).to(dev)) \
                 .float().view(n, KL_K - 1, -1)
    cand_h = torch.cat([pos_v.unsqueeze(1), dist_h], 1)
    h_t, pmax_t = entropies(cand_h)

    for name, h, pm in (("uniform_bank_the_recipe", h_u, pmax_u),
                        ("teacher_top200", h_t, pmax_t)):
        q = {f"p{p}": float(np.percentile(h, p)) for p in (1, 5, 25, 50, 75, 95, 99)}
        out["arms"][name] = {
            "entropy_nats": {"mean": float(h.mean()), **q},
            "entropy_as_fraction_of_ceiling": float(h.mean() / np.log(KL_K)),
            "p_max_of_teacher_distribution": {"mean": float(pm.mean()),
                                              "p50": float(np.percentile(pm, 50)),
                                              "p95": float(np.percentile(pm, 95))},
            "share_of_queries_below_1e-4_nats": float((h < 1e-4).mean()),
            "share_of_queries_below_1e-2_nats": float((h < 1e-2).mean()),
        }

    u, t = out["arms"]["uniform_bank_the_recipe"], out["arms"]["teacher_top200"]
    out["reading"] = (
        f"Under the recipe's own uniform sampler the teacher's candidate distribution carries "
        f"{u['entropy_nats']['mean']:.2e} nats on average "
        f"({u['entropy_as_fraction_of_ceiling']:.2%} of the ln(32)={np.log(KL_K):.3f} ceiling), "
        f"and {u['share_of_queries_below_1e-4_nats']:.1%} of queries are below 1e-4 nats. "
        f"With distractors drawn from the teacher's own top-200 it carries "
        f"{t['entropy_nats']['mean']:.2e} nats "
        f"({t['entropy_as_fraction_of_ceiling']:.2%} of ceiling). "
        f"H2 predicted the first number would be ~1e-4.")
    out["seconds"] = round(time.time() - t0, 1)

    if a.smoke:
        (RESULTS / "m8_b2_entropy.SMOKE.json").write_text(json.dumps(out, indent=2, default=str))
    else:
        probe_guard.write_result(OUT, out, "B2")
    print(json.dumps({"arms": out["arms"], "reading": out["reading"]}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
