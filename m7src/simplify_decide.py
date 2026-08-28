"""Apply the pre-registered recipe-simplification rule, mechanically.

m7/LEDGER.md, "Recipe simplification", fixed before any number: accept the simplified recipe iff
the dependence-preserving RAW paired 95% CI lower bound for (simple - complex) is **> -0.0040**,
in fp16 AND int8. Non-inferiority, not a two-sided band -- the question is whether a loss larger
than the margin can be ruled out, so a win is an acceptance too.

The margin is anchored to the smallest effect this project has actually adopted (lever #4 `sqrt`,
+0.0040 fp16, before it was un-adopted on the new candidate). Replay noise is ~5e-6 on the dev
macro (the ablation replay's raw delta is 4.47e-06 -- reproducible, not bit-identical), far too
small to calibrate a band from, which is why an adopted effect is the anchor instead.

This is a reproducibility and over-engineering fix, NOT a quality lever, and the script prints it
that way: a win here is not evidence the simplification improves anything, only that the removed
components were not carrying the number.

Usage: simplify_decide.py <compare_tag> <simplified_key> [<baseline_key>]
"""
import json
import sys

from _paths import REPO

OOD = ("cqadup-programmers", "cqadup-physics")
MARGIN = 0.0040

REMOVED = {
    "init=teacher -> input_emb": "30,522 teacher forward passes to build the init rows",
    "b_pseudo_queries request 2m -> 500k": "2.85x less pseudo-query generation and teacher "
                                           "encoding -- the REALISED pools are 924,704 and "
                                           "324,704 spans, not 2m and 500k; build() saturates",
    "idf_init_weights True -> False": "an IDF pass over the training queries",
    "reg_init 1e-3 -> 0.0": "a penalty term and the W0 anchor it needs",
}


def main(tag, simple_key, base_key=None):
    d = json.loads((REPO / "results" / f"m7_compare_full_{tag}.json").read_text())
    base_key = base_key or d["baseline"]
    rows = {}
    for q in ("fp16", "int8"):
        c = d["comparisons"][f"{simple_key}_vs_{base_key}|{q}"]["dependence_preserving"]
        rows[q] = {"delta_raw": c["paired"]["delta_raw"], "ci95_raw": c["paired"]["ci95_raw"],
                   "signflip_p": c["signflip"]["p"],
                   "non_inferior": c["paired"]["ci95_raw"][0] > -MARGIN}
    accept = all(rows[q]["non_inferior"] for q in rows)

    def ood(key):
        pc = d["per_component_unrounded"][f"{key}|fp16"]
        return sum(pc[c] for c in OOD) / len(OOD)

    out = {
        "_what": "recipe simplification, judged by the pre-registered non-inferiority rule",
        "_protocol": "m7/LEDGER.md 'Recipe simplification', fixed before any number: accept iff "
                     f"the dependence-preserving raw paired 95% CI lower bound > -{MARGIN} in "
                     "fp16 AND int8. One arm, no ladder of fallbacks -- backing off component by "
                     "component until something passes would be adaptive dev search.",
        "_framing": "A reproducibility and over-engineering fix, not a quality change. Acceptance "
                    "means the removed components were not carrying the number; it is not "
                    "evidence that removing them helps.",
        "_status": "exploratory dev SELECTION evidence",
        "_ood_disclosure": "cqadup-programmers + cqadup-physics, the only dev components outside "
                           "the TRAIN mix and its Wikipedia family. Disclosure only; not in the bar.",
        "compare_artifact": f"m7_compare_full_{tag}.json",
        "simplified": simple_key, "baseline": base_key, "margin": MARGIN,
        "macro_fp16": {simple_key: d["macros_unrounded"][f"{simple_key}|fp16"],
                       base_key: d["macros_unrounded"][f"{base_key}|fp16"]},
        "macro_int8": {simple_key: d["macros_unrounded"][f"{simple_key}|int8"],
                       base_key: d["macros_unrounded"][f"{base_key}|int8"]},
        "ood_macro_fp16": {simple_key: ood(simple_key), base_key: ood(base_key)},
        "per_component_fp16": {k: d["per_component_unrounded"][f"{k}|fp16"]
                               for k in (simple_key, base_key)},
        "stats": rows,
        "removed_and_what_it_buys": REMOVED,
        "accepted": accept,
        "consequence": ("the simplified artifact is the candidate; lever #4 is re-adjudicated on "
                        "it and fusion is selected on it" if accept else
                        "the measured recipe ships unchanged. WHY it failed is NOT established: "
                        "single-knob main effects sum to about +0.0015 against this joint "
                        "-0.0048, a gap the size of the recipe-perturbation band, so interaction "
                        "and one unlucky draw are not separable. Do not report this as evidence "
                        "that inert components interact."),
    }
    (REPO / "results" / "m7_simplify_decision.json").write_text(json.dumps(out, indent=1))
    print(f"{simple_key} vs {base_key}, margin -{MARGIN}")
    for q in ("fp16", "int8"):
        r = rows[q]
        print(f"  {q}: {out['macro_' + q][simple_key]:.4f} vs {out['macro_' + q][base_key]:.4f}  "
              f"delta {r['delta_raw']:+.4f} ci {[round(x, 4) for x in r['ci95_raw']]} "
              f"p={r['signflip_p']:.4f}  non-inferior={r['non_inferior']}")
    print(f"  ood fp16: {out['ood_macro_fp16'][simple_key]:.4f} vs "
          f"{out['ood_macro_fp16'][base_key]:.4f}")
    print(f"ACCEPTED={accept} -- {out['consequence']}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
