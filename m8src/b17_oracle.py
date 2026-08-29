"""B17: does the lookup-table class cap IN DOMAIN? (LEDGER §9, registry probe `B17`.)

THE QUESTION. M7's table retained 0.755 of its teacher on the six. Two very different explanations
fit that: the class is expressive enough and the SUPERVISION is what fell short, or a bag of token
vectors simply cannot land where a contextual query encoder lands and no amount of supervision
fixes it. They imply opposite milestones -- the first says R1 (objective, data, training frame) is
the centre of gravity, the second says only added CAPACITY (D2's vocabulary, D1's doc-side head)
can move anything.

THE MEASUREMENT. Split the dev CQADupStack components' queries 50/50, fit a closed-form table on
one half's queries against the teacher's own query vectors for them, and score the OTHER half.
That is in-domain supervision, generalising only across queries, on the exact task -- about as
favourable a setting as the class will ever see. The reference is the teacher's own symmetric
score on the same held-out half (0.4806 over the two components, `work/devres/refs-*.json`).

"ORACLE" means lambda is chosen on the HELD-OUT half, which is selection on the thing being
reported and is deliberately so: it is the strongest possible shot for the class, in the same
spirit as M7's oracle-lambda projection reruns. A ceiling measured with the scales tipped toward
the method is a ceiling you can believe when it is LOW. The honestly-selected number (lambda
chosen on the fit half) is reported beside it, and the gap between them is the size of the favour.

THE REGISTERED ROUTING RULE, fixed before this number existed, and amended once (also before)
to require out-of-domain corroboration on the upper branch:
    held-out >= 0.45  -> supervision/objective is the story AND R1 takes the majority budget
                         ONLY IF B3-template out-of-domain corroboration is also present;
                         without it the branch reads "both".
    held-out <= 0.40  -> the class caps in domain and D2/D1/D4' carry the milestone.
    in between        -> both, budget split 50/50.

THE CAVEAT THAT MUST TRAVEL WITH THE NUMBER. The fit half is roughly 950 queries against
30,522 x 1024 parameters, so the ridge is enormously underdetermined and the solution stays close
to its initialization. This measures what ~950 in-domain queries buy on top of the teacher-vector
init -- it is NOT the class's asymptotic ceiling, and a low number here is evidence about that
supervision budget, not proof of a hard limit. Reported at every use.
"""
import argparse
import json
import sys
import time

import numpy as np

import m8base
import probe_guard

RESULTS = m8base.RESULTS
OUT = RESULTS / "m8_b17_oracle.json"
COMPONENTS = ("cqadup-programmers", "cqadup-physics")
LAMBDAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
TEACHER_CEILING = 0.4806          # stella-400M-v5 symmetric, CQA-2 (work/devres/refs-*.json)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lambdas", type=float, nargs="*", default=list(LAMBDAS))
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    import torch
    import blockcg
    import dev_eval
    import encoders
    import stage0_ridge as sr
    from evalkit import per_query_ndcg, topk_ids_scores
    from init_table import get_init
    from table import Preproc, QueryTable, get_tokenizer
    from teacher import QUERY_PREFIX, encode_cached

    spec = encoders.active()
    pre, tok = Preproc(), get_tokenizer()
    dev = m8base.device()
    rng = np.random.default_rng(a.seed)
    t0 = time.time()

    fit_texts, held = [], {}
    for comp in COMPONENTS:
        doc_ids, _dt, q_ids, q_texts, qrels, dv = dev_eval.doc_vecs(comp)
        order = rng.permutation(len(q_ids))
        half = len(order) // 2
        fit_i, held_i = order[:half], order[half:]
        fit_texts += [q_texts[i] for i in fit_i]
        held[comp] = {"doc_ids": doc_ids, "dv": dv, "qrels": qrels,
                      "q_ids": [q_ids[i] for i in held_i],
                      "q_texts": [q_texts[i] for i in held_i]}
        print(f"{comp}: {len(fit_i)} fit / {len(held_i)} held-out queries", flush=True)

    X = sr.bag_matrix(tok, fit_texts, pre, spec.vocab)
    Y = np.asarray(encode_cached(f"b17-fit-{len(fit_texts)}", fit_texts, prefix=QUERY_PREFIX,
                                 dtype=torch.float16, verbose=False), dtype=np.float32)
    W0 = get_init("teacher", pre)
    print(f"fit: {len(fit_texts)} queries, X {X.shape} ({X.nnz:,} nnz), "
          f"{X.shape[1] * Y.shape[1]:,} parameters  ({time.time()-t0:.0f}s)", flush=True)

    def score(W):
        model = QueryTable(W, weight_init=None, learned_weights=False,
                           fallback_id=spec.cls_id).to(dev)
        per = {}
        for comp, h in held.items():
            qv = model.encode(h["q_texts"], pre, tok=tok)
            run = topk_ids_scores(qv, h["dv"], h["doc_ids"], k=100, qids=h["q_ids"])
            per[comp] = float(np.mean(list(per_query_ndcg(run, h["qrels"]).values())))
        del model
        m8base.empty_cache()
        return float(np.mean(list(per.values()))), per

    rows = {}
    for lam in a.lambdas:
        W, info = blockcg.block_cg_ridge(X, Y, W0, lam, device=dev, verbose=False)
        macro, per = score(W)
        rows[str(lam)] = {"held_out_macro": macro, "per_component": per,
                          "cg_iterations": info["iterations"],
                          "cg_converged": info["converged"]}
        print(f"  lam={lam:g}: held-out {macro:.4f}  {per}", flush=True)
        del W
        m8base.empty_cache()

    # The init itself, scored on the same held-out half: the floor this experiment must beat to
    # have measured anything at all. If ~950 queries move nothing, the reading is about the
    # supervision budget, not about the class.
    init_macro, init_per = score(W0)
    oracle_lam = max(rows, key=lambda k: rows[k]["held_out_macro"])
    oracle = rows[oracle_lam]["held_out_macro"]

    branch = ("class caps in domain -> D2/D1/D4' carry the milestone" if oracle <= 0.40 else
              "supervision/objective is the story -> R1 takes the majority budget ONLY IF "
              "B3-template out-of-domain corroboration is present; without it, BOTH"
              if oracle >= 0.45 else "BOTH, budget split 50/50")

    out = {
        "_note": __doc__.strip().splitlines()[0],
        "components": list(COMPONENTS), "seed": a.seed,
        "n_fit_queries": len(fit_texts),
        "n_parameters": int(X.shape[1] * Y.shape[1]),
        "teacher_symmetric_ceiling_cqa2": TEACHER_CEILING,
        "per_lambda": rows,
        "init_only_held_out_macro": init_macro, "init_only_per_component": init_per,
        "oracle_lambda": oracle_lam, "oracle_held_out_macro": oracle,
        "oracle_as_fraction_of_teacher": oracle / TEACHER_CEILING,
        "gain_over_init": oracle - init_macro,
        "routing_rule": {"registered": ">=0.45 / <=0.40 / between", "value": oracle,
                         "branch": branch},
        "caveat": (f"the fit half is {len(fit_texts)} queries against "
                   f"{X.shape[1] * Y.shape[1]:,} parameters, so the ridge is enormously "
                   f"underdetermined and stays near its initialization. This measures what "
                   f"~{len(fit_texts)} in-domain queries buy ON TOP OF the teacher-vector init; "
                   f"it is NOT the class's asymptotic ceiling, and a low value is evidence about "
                   f"that supervision budget rather than proof of a hard limit."),
        "oracle_note": "lambda is chosen on the HELD-OUT half -- selection on the reported "
                       "quantity, deliberately, as the strongest shot for the class. The "
                       "honestly-selected value is per_lambda at the fit-half argmax, and the "
                       "difference is the size of the favour.",
        "seconds": round(time.time() - t0, 1),
    }
    if a.smoke:
        (RESULTS / "m8_b17_oracle.SMOKE.json").write_text(json.dumps(out, indent=2, default=str))
    else:
        probe_guard.write_result(OUT, out, "B17")
    print(json.dumps({k: out[k] for k in
                      ("n_fit_queries", "init_only_held_out_macro", "oracle_lambda",
                       "oracle_held_out_macro", "oracle_as_fraction_of_teacher",
                       "gain_over_init", "routing_rule")}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
