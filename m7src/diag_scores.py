"""Why did contrastive collapse? Measure the score geometry instead of ablating blindly.

Objective A declined monotonically with random negatives. Three mechanisms fit that curve and
they call for different fixes, so this measures which one is real:

  1  negatives too easy      -> positive and random-negative score distributions barely overlap
  2  temperature too sharp   -> at tau=0.02 the softmax mass sits on the top few random negatives,
                               whose scores are high because bge's space is anisotropic rather
                               than because they are semantically close
  3  fn_margin masks the good ones -> the teacher-margin filter removes a large share of the
                               HARDEST negatives, leaving an easy, uninformative pool

Reports, for a query sample: positive-score distribution, random-negative score distribution,
teacher-mined-negative score distribution, the softmax mass concentration at several
temperatures, and the fraction of negatives the fn_margin filter removes at each rank band.
"""
import json
import sys

import numpy as np
import torch

import mix
import pool as poolmod
from _paths import REPO, WORK
from teacher import QUERY_PREFIX, encode_cached
from train import build_arrays, kept_pairs

Cfg = None


def main(n_q=2000, n_neg=32768, seed=0):
    from train import Cfg as _C
    cfg = _C(hard_neg_k=0)
    index, pool_vecs, meta = poolmod.build()
    q_texts, pos_idx, hn_idx, src_id, srcs = build_arrays(cfg, index)
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(q_texts), size=min(n_q, len(q_texts)), replace=False)
    qs = [q_texts[i] for i in pick]
    tq = np.asarray(encode_cached(f"diagq-{len(qs)}", qs, prefix=QUERY_PREFIX,
                                  dtype=torch.float16, verbose=False), dtype=np.float32)
    p_i = np.array([pos_idx[i][0] for i in pick])
    pos = np.ascontiguousarray(pool_vecs[p_i]).astype(np.float32)
    s_pos = (tq * pos).sum(1)

    neg_i = np.sort(rng.choice(len(pool_vecs), size=n_neg, replace=False))
    neg = torch.from_numpy(np.ascontiguousarray(pool_vecs[neg_i])).cuda().float()
    Q = torch.from_numpy(tq).cuda()
    s_neg = (Q @ neg.T).cpu().numpy()
    del neg, Q
    torch.cuda.empty_cache()

    out = {"n_queries": len(qs), "n_random_negatives": n_neg,
           "positive_scores": {"mean": float(s_pos.mean()), "p5": float(np.percentile(s_pos, 5)),
                               "p50": float(np.percentile(s_pos, 50)),
                               "p95": float(np.percentile(s_pos, 95))},
           "random_negative_scores": {"mean": float(s_neg.mean()),
                                      "p50": float(np.percentile(s_neg, 50)),
                                      "p99": float(np.percentile(s_neg, 99)),
                                      "p99.9": float(np.percentile(s_neg, 99.9)),
                                      "max": float(s_neg.max())}}
    # mechanism 1: how often does a RANDOM negative outscore the positive?
    beats = (s_neg > s_pos[:, None])
    out["random_negatives_outscoring_the_positive"] = {
        "mean_count_per_query": float(beats.sum(1).mean()),
        "frac_queries_with_at_least_one": float((beats.sum(1) > 0).mean()),
        "note": "if this is near zero the negatives are trivially separable (mechanism 1)"}
    # mechanism 2: where does the softmax mass sit, per temperature?
    conc = {}
    for temp in (0.005, 0.01, 0.02, 0.05, 0.1):
        z = s_neg / temp
        z = z - z.max(1, keepdims=True)
        w = np.exp(z)
        w /= w.sum(1, keepdims=True)
        srt = -np.sort(-w, axis=1)
        conc[str(temp)] = {"top1_mass": float(srt[:, 0].mean()),
                           "top10_mass": float(srt[:, :10].sum(1).mean()),
                           "top100_mass": float(srt[:, :100].sum(1).mean()),
                           "effective_negatives": float((1.0 / (srt ** 2).sum(1)).mean())}
    out["softmax_mass_by_temperature"] = conc
    out["softmax_note"] = ("effective_negatives is the inverse participation ratio: how many "
                           "negatives actually receive gradient. If it is single digits at "
                           "tau=0.02, the objective is chasing the anisotropy tail (mechanism 2)")
    # mechanism 3: what does the fn_margin filter remove?
    filt = {}
    order = np.argsort(-s_neg, axis=1)
    for m in (0.0, 0.01, 0.02, 0.05):
        if m == 0.0:
            filt["0.0"] = {"frac_removed_overall": 0.0, "frac_removed_in_top100": 0.0}
            continue
        mask = s_neg > (s_pos[:, None] - m)
        top100 = np.take_along_axis(mask, order[:, :100], axis=1)
        filt[str(m)] = {"frac_removed_overall": float(mask.mean()),
                        "frac_removed_in_top100": float(top100.mean())}
    out["fn_margin_filter"] = filt
    out["fn_margin_note"] = ("frac_removed_in_top100 is the one that matters: the filter is meant "
                             "to drop false negatives, but if it removes most of the hardest "
                             "negatives it leaves an uninformative pool (mechanism 3)")
    (REPO / "results" / "m7_diag_scores.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2000)
