"""T1's swap bar, executed (LEDGER §10, registry probe `T1`).

A pre-registered rule a session can re-read in its own favour is not a pre-registration, so the
bar runs as code against the committed screen artifacts and writes its own verdict.

THE BAR, all four conditions, none waivable here:
  1. the challenger's closed-form table beats the incumbent's, CI-RESOLVED -- raw two-sided 95%
     `paired_dep` CI excluding 0 AND `signflip_dep` p < 0.05, on per-query nDCG@10 over
     cqadup-programmers and cqadup-physics, strict alignment;
  2. the point-estimate margin EXCEEDS the swap's CI-widening penalty, fixed per challenger
     BEFORE any result was read: 0.0050 near-sibling / 0.0096 dissimilar, where near-sibling means
     sharing the incumbent's tokenizer identity AND dimension. Every T1 challenger is dissimilar;
  3. the off-family read (nq-250k, hotpotqa) does not reverse the sign on EITHER component;
  4. Dylan signs off.
Multiplicity: Holm across the challenger set at alpha = 0.05.

Condition 1 is evaluated first because the bar is ordered: if it fails, 2, 3 and the tie-break
never arise, and the off-family read -- which is expensive, hotpotqa being 5.23M documents per
candidate -- is never bought. That is the same structure M7's screen used and the same reason.
"""
import json
import sys

import numpy as np

import m8base
import boot
import probe_guard

RESULTS = m8base.RESULTS
OUT = RESULTS / "m8_t1_decision.json"
INCUMBENT = "stella-400M-v5"
COMPONENTS = ("cqadup-programmers", "cqadup-physics")


def load(name):
    p = RESULTS / f"m8_t1_{name}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def per_query_at_best(d):
    return d["lambdas"][d["best_lambda"]]["per_query"]


def main():
    reg = probe_guard.registry()["probes"]["T1"]
    cands = reg["candidates"]
    inc = load(INCUMBENT)
    if inc is None:
        raise SystemExit(f"the incumbent {INCUMBENT} has not been screened; T1's bar requires it "
                         f"re-probed in the IDENTICAL frame, not compared against a recorded "
                         f"number (LEDGER §10).")
    inc_pq = per_query_at_best(inc)

    rows, pvals = {}, {}
    for name in cands:
        # the registry keys candidates by REPO identity; `spec_name` says which encoder Spec
        # implements each. Without the mapping a screened candidate silently reads as unscreened,
        # which is a wrong verdict rather than an error.
        spec = cands[name].get("spec_name", name)
        if spec == INCUMBENT:
            continue
        d = load(spec)
        if d is None:
            rows[name] = {"screened": False, "spec_name": spec,
                          "why": cands[name].get("blockers") or "not screened this session"}
            continue
        pq = per_query_at_best(d)
        sf = boot.signflip_dep(pq, inc_pq, alternative="greater", strict=True)
        pr = boot.paired_dep(pq, inc_pq, alternative="two-sided", strict=True)
        penalty = cands[name]["penalty"]
        delta = pr["delta_raw"]
        ci_resolved = bool(pr["ci95_raw"][0] > 0 or pr["ci95_raw"][1] < 0) and sf["p"] < 0.05
        rows[name] = {
            "screened": True, "spec_name": spec,
            "best_lambda": d["best_lambda"], "macro": d["best_macro"],
            "incumbent_macro": inc["best_macro"],
            "delta_vs_incumbent": delta,
            "ci95_raw": pr["ci95_raw"], "signflip_dep_p": sf["p"],
            "penalty_class": cands[name]["penalty_class"], "penalty": penalty,
            "cond1_beats_incumbent_ci_resolved": bool(delta > 0 and ci_resolved),
            "cond2_margin_exceeds_penalty": bool(delta > penalty),
            "cond3_off_family": "NOT EVALUATED -- condition 1 failed, so the bar's ordering "
                                "means it is never bought (hotpotqa is 5.23M docs per candidate)"
                                if not (delta > 0 and ci_resolved) else "REQUIRED",
            "int8_table_mb": d["int8_table_mb"],
            "vocab": d["encoder"]["vocab"], "dim": d["encoder"]["dim"],
        }
        pvals[name] = sf["p"]

    holm = boot.holm(pvals, alpha=0.05) if pvals else {}
    for n, h in holm.items():
        rows[n]["holm"] = h

    survivors = [n for n, r in rows.items()
                 if r.get("screened") and r.get("cond1_beats_incumbent_ci_resolved")
                 and r.get("cond2_margin_exceeds_penalty") and holm.get(n, {}).get("reject")]

    out = {
        "_note": __doc__.strip().splitlines()[0],
        "incumbent": {"name": INCUMBENT, "best_lambda": inc["best_lambda"],
                      "macro": inc["best_macro"], "int8_table_mb": inc["int8_table_mb"],
                      "fit_list": inc["fit_list"]},
        "components": list(COMPONENTS),
        "candidates": rows,
        "holm_alpha_0.05": holm,
        "survivors": survivors,
        "verdict": ("NO SWAP -- the incumbent stands. Same-teacher is the registered default "
                    "(LEDGER §10)." if not survivors else
                    f"CANDIDATE(S) CLEAR CONDITIONS 1-2: {survivors}. Conditions 3 and 4 now "
                    f"arise: the off-family read must run, and Dylan signs."),
        "not_screened_this_session": {n: r.get("why") for n, r in rows.items()
                                      if not r.get("screened")},
    }
    probe_guard.write_result(OUT, out, "T1")
    print(json.dumps({"incumbent": out["incumbent"], "verdict": out["verdict"],
                      "candidates": {n: {k: r.get(k) for k in
                                         ("macro", "delta_vs_incumbent", "ci95_raw",
                                          "signflip_dep_p", "penalty",
                                          "cond1_beats_incumbent_ci_resolved",
                                          "cond2_margin_exceeds_penalty")}
                                     for n, r in rows.items() if r.get("screened")}},
                     indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
