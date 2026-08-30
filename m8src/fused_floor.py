"""The FUSED half of the noise floor (LEDGER §4.7, §4.4 gap list).

`noise_floor.py` measures the floor on DENSE endpoints. B3, B13, R1-ASSEMBLY, D-SYNTH and
D-FINEWEB all register "dense **AND fused**" endpoints, and §4.7 says a multi-endpoint probe's bar
is the max over its endpoints' bars -- so without a fused floor those bars are not computable as
registered, and `probe_guard` correctly refuses them. This closes that.

Same design as the dense floor and for the same reasons: K = 3 arms differing ONLY in training
seed (a true null), floor = max of the three pairwise |delta|, bar = max(0.0040, 2 x floor).

THE FUSION OPERATOR IS FROZEN, NOT FITTED. §7 requires the operator to be fixed before Stage R and
applied identically at every fused read; the final invocation instantiates parameters only. So
this applies M7's frozen spec (`results/m7_fusion_p35w-2m-s2500.json`) rather than re-selecting
per arm -- re-selecting would measure the floor of a fitting procedure, which is a different and
much larger quantity, and would quietly make the fused floor depend on a choice each arm got to
make for itself.

LOOP ORDER IS LOAD-BEARING. `dev_eval.doc_vecs` re-parses its corpus cache on EVERY call, and
HotpotQA's is 5.23M documents peaking around 14 GB (m7/CODEMAP.md pitfall 6). Calling it once per
(arm x precision x component) would be 40 loads of that corpus. Components are therefore the OUTER
loop: each corpus is loaded once and every arm is scored against it before it is released. This is
the same reversal that turned a 3.6-hour job into 165 seconds in M7 (pitfall 7).
"""
import argparse
import gc
import json
import sys
import time

import numpy as np

import m8base
import probe_guard

RESULTS = m8base.RESULTS
OUT = RESULTS / "m8_noise_floor_fused.json"
# The frozen fusion spec's own components -- BM25 needs document text, and the two held-out dev
# slices carry pool row indices rather than text, so they have no fused read by construction.
COMPONENTS = ("nq-250k", "hotpotqa", "cqadup-programmers", "cqadup-physics")
ARMS = ("m8nf-seed0", "m8nf-seed1", "m8nf-seed2", "m8nf-steps2250", "m8nf-steps2750")
SEED_ARMS = ARMS[:3]


def frozen_spec():
    d = json.loads((RESULTS / "m7_fusion_p35w-2m-s2500.json").read_text())
    spec = {"family": d["family"], "param": d["param"]}
    print(f"frozen fusion operator: {spec} (dev macro at selection {d['dev_macro']:.4f})",
          flush=True)
    return spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    # `--arms` only controlled which arms were LOADED; the floor was computed over the module
    # constant SEED_ARMS regardless, so passing a different null measured the wrong thing while
    # looking like it worked. The null is now named explicitly and defaults to the A-leg one.
    ap.add_argument("--seed-arms", nargs="*", default=None,
                    help="the arms forming the null (default: the A-leg seed arms)")
    ap.add_argument("--out", default=None,
                    help="artifact filename under results/ (default: the A-leg fused floor). The "
                         "B-leg floor MUST use its own name -- overwriting the A-leg artifact "
                         "would redefine a floor that frozen bars already cite.")
    ap.add_argument("--modes", nargs="*", default=["mean", "sqrt"],
                    help="pool modes to load. A probe that reads only its release pooling can pass "
                         "just that one and halve the variants scored.")
    ap.add_argument("--precisions", nargs="*", default=["fp16", "int8"],
                    help="precisions to keep. Same reason.")
    ap.add_argument("--per-query-out", default=None,
                    help="also write the per-query fused scores to this .json.gz")
    ap.add_argument("--components", nargs="*", default=list(COMPONENTS))
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    import torch
    import compare_full
    import dev_eval
    import fusion
    import select_fusion
    from evalkit import per_query_ndcg, topk_ids_scores

    spec = frozen_spec()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load every arm's table ONCE. Small (31 MB int8 each); the corpora are the expensive side.
    loaded = {}
    for rid in a.arms:
        for mode in a.modes:
            # compare_full.load takes the BARE run id plus an optional pool-mode override; the
            # "rid:mode" form is its COMMAND-LINE spelling, split before it ever reaches load().
            rel, pre, models = compare_full.load(rid, mode if mode != "mean" else None,
                                                device=device)
            for prec, m in models.items():
                if prec in a.precisions:
                    loaded[f"{rid}:{mode}|{prec}"] = (m, pre)
    print(f"{len(loaded)} arm variants loaded", flush=True)

    per_query = {k: {} for k in loaded}
    for comp in a.components:
        t0 = time.time()
        # ONE load of this corpus; every arm is scored against it before it is released.
        # dev_eval.doc_vecs returns (doc_ids, doc_texts, q_ids, q_texts, qrels, doc_vectors) --
        # vectors LAST. Unpacking it in any other order shifts every argument downstream.
        b_run, _ = select_fusion.bm25_run_and_key(comp)
        doc_ids, doc_texts, q_ids, q_texts, qrels, dv = dev_eval.doc_vecs(comp)
        print(f"{comp}: {len(doc_ids):,} docs, {len(q_ids):,} queries "
              f"({time.time()-t0:.0f}s to load)", flush=True)
        for key, (model, pre) in loaded.items():
            d_run = topk_ids_scores(model.encode(q_texts, pre), dv, doc_ids,
                                    k=fusion.DEPTH, qids=q_ids)
            fused = fusion.apply_frozen(spec, d_run, b_run)
            # evalkit's own scorer, not a hand-rolled RelevanceEvaluator: it restricts qrels to
            # the run's keys, which is what every other decision path in this repo does.
            per_query[key][comp] = {q: float(v)
                                    for q, v in per_query_ndcg(fused, qrels).items()}
        del dv, doc_ids, doc_texts, b_run, q_texts
        gc.collect()
        m8base.empty_cache()
        print(f"{comp}: {len(loaded)} variants fused and scored "
              f"({time.time()-t0:.0f}s total)", flush=True)

    # Floor, per (precision, pool_mode), on the fused macro over these components.
    def macro(pq):
        return float(np.mean([np.mean(list(pq[c].values())) for c in a.components]))

    import itertools
    null_arms = list(a.seed_arms) if a.seed_arms is not None else list(SEED_ARMS)
    missing = [r for r in null_arms if r not in a.arms]
    if missing:
        raise SystemExit(f"--seed-arms names {missing}, which --arms did not load")
    floor, bars, rows = {}, {}, {}
    for prec in ("fp16", "int8"):
        for mode in a.modes:
            sel = {r: macro(per_query[f"{r}:{mode}|{prec}"]) for r in null_arms
                   if f"{r}:{mode}|{prec}" in per_query}
            if null_arms and len(sel) not in (0, len(null_arms)):
                raise SystemExit(f"{prec}.{mode}: null needs {sorted(null_arms)} but only "
                                 f"{sorted(sel)} were scored; a floor from a partial arm set "
                                 f"understates the floor and loosens every bar reading it.")
            if len(sel) < 2:
                continue
            key = f"{prec}.{mode}.fused_macro"
            ds = [{"pair": [x, y], "abs_delta": abs(sel[x] - sel[y])}
                  for x, y in itertools.combinations(sorted(sel), 2)]
            floor[key] = max(d["abs_delta"] for d in ds)
            bars[key] = max(0.0040, 2 * floor[key])
            rows[key] = {"arm_macros": sel, "pairwise": ds}

    out = {
        "_note": __doc__.strip().splitlines()[0],
        "fusion_operator": spec,
        "_operator_note": "FROZEN, not re-fitted per arm: re-selecting would measure the floor of "
                          "a fitting procedure, a different and much larger quantity (LEDGER 7).",
        "components": list(a.components),
        "_components_note": "the two held-out dev slices carry pool row indices rather than "
                            "document text, so BM25 -- and therefore any fused read -- does not "
                            "exist for them by construction.",
        "arm_macros": {k: macro(v) for k, v in per_query.items()},
        "floor": floor, "bars": bars, "detail": rows,
        "bar_formula": "bar = max(0.0040, 2 x floor); floor = max pairwise |delta| over the "
                       "arms naming the null",
        "null_arms": null_arms,
    }
    if a.per_query_out:
        import gzip
        with gzip.open(RESULTS / a.per_query_out, "wt") as fh:
            json.dump({"per_query": per_query, "components": list(a.components)}, fh)
        print(f"wrote per-query fused scores to {a.per_query_out}", flush=True)
    if a.smoke:
        (RESULTS / "m8_noise_floor_fused.SMOKE.json").write_text(json.dumps(out, indent=2,
                                                                           default=str))
    else:
        probe_guard.write_result((RESULTS / a.out) if a.out else OUT, out, "NF")
    print(json.dumps({"floor": floor, "bars": bars}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
