"""Apply the pre-registered negatives rule to the step-rule-corrected arms, mechanically.

The rule is in m7/LEDGER.md, "Negatives ablation", and was fixed before any arm's result. This
script executes it rather than restating it, so the decision is auditable and a session cannot
quietly re-read a tie-break in its own favour:

  1. PROMOTION -- an arm reaches the full-suite comparison only if its proxy macro exceeds the
     `bank` control's. Checked here against the committed run JSONs, not assumed.
  2. BAR -- dependence-preserving signflip p<0.05 AND raw paired CI > 0, in fp16 AND int8,
     against the candidate the bar was written against.
  3. HOLM at alpha=0.05 across the promoted arms, since more than one is promoted.
  4. TIE-BREAK, three levels: largest full-suite fp16 macro; then, within the ~0.0007 band,
     fewer negatives and a single mining source; then, if that also ties, the teacher-mined arm,
     because mining with the teacher is a by-product of a document encode this system performs
     anyway while the BM25 arm needs a second retrieval system stood up to reproduce the recipe.

Disclosure, per LEDGER.md's biased-estimator rule: the out-of-domain subset
(cqadup-programmers + cqadup-physics) is reported for every arm alongside the six-macro. It does
not enter the bar.

Usage: negatives_decide.py [<compare_tag>]     default tag: steprule
"""
import json
import re
import sys

import boot
from _paths import REPO, WORK

OOD = ("cqadup-programmers", "cqadup-physics")
CONTROL = "p4n-bank-a"
BAND = 0.0007          # the parsimony band, as written in LEDGER.md
# k negatives and how many mining sources, per arm suffix -- the parsimony tie-break's inputs.
SHAPE = {"teacher16": (16, 1, "teacher"), "bm2516": (16, 1, "bm25"), "mixed32": (32, 2, "mixed")}
# An arm re-run at its own best proxy step, per the step-selection rule: p4n-<suffix>-s<steps>-a.
CORRECTED = re.compile(r"^p4n-[a-z0-9]+-s\d+-a$")


def proxy(run_id):
    return json.loads((WORK / "runs" / f"{run_id}.json").read_text())["final_macro"]


def main(tag="steprule"):
    p = REPO / "results" / f"m7_compare_full_{tag}.json"
    d = json.loads(p.read_text())
    base = d["baseline"]
    arms = [k for k in d["artifacts"] if k != base]

    control_proxy = proxy(CONTROL)
    rows = {}
    for key in arms:
        rid = d["artifacts"][key]["run_id"]
        suffix = next((s for s in SHAPE if rid.startswith(f"p4n-{s}")), None)
        pm = proxy(rid)
        comp = {q: d["comparisons"][f"{key}_vs_{base}|{q}"]["dependence_preserving"]
                for q in ("fp16", "int8")}
        bar = all(comp[q]["signflip"]["p"] < 0.05 and comp[q]["paired"]["ci95_raw"][0] > 0
                  for q in ("fp16", "int8"))
        pc = d["per_component_unrounded"][f"{key}|fp16"]
        rows[key] = {
            "run_id": rid, "arm": suffix,
            "proxy_macro": pm, "promoted_on_proxy": pm > control_proxy,
            "macro_fp16": d["macros_unrounded"][f"{key}|fp16"],
            "macro_int8": d["macros_unrounded"][f"{key}|int8"],
            "ood_macro_fp16": sum(pc[c] for c in OOD) / len(OOD),
            "per_component_fp16": pc,
            "p_fp16": comp["fp16"]["signflip"]["p"], "p_int8": comp["int8"]["signflip"]["p"],
            "ci95_raw_fp16": comp["fp16"]["paired"]["ci95_raw"],
            "ci95_raw_int8": comp["int8"]["paired"]["ci95_raw"],
            "delta_raw_fp16": comp["fp16"]["paired"]["delta_raw"],
            "bar_before_holm": bar,
        }

    # Holm over the arms the rule promotes. The uncorrected 2500-step arm is in the comparison
    # only so the step-rule correction's own effect is visible; it is DESCRIPTIVE and must not
    # consume a Holm slot, or the correction would silently cost the corrected arms power.
    family = [k for k, r in rows.items()
              if r["arm"] and r["promoted_on_proxy"] and CORRECTED.match(r["run_id"])]
    holm = {q: boot.holm({k: rows[k][f"p_{q}"] for k in family}, alpha=0.05)
            for q in ("fp16", "int8")}
    survivors = [k for k in family
                 if all(holm[q][k]["reject"] and rows[k][f"ci95_raw_{q}"][0] > 0
                        for q in ("fp16", "int8"))]

    winner, why = None, None
    if survivors:
        best = max(survivors, key=lambda k: rows[k]["macro_fp16"])
        near = [k for k in survivors
                if rows[best]["macro_fp16"] - rows[k]["macro_fp16"] <= BAND]
        if len(near) == 1:
            winner, why = best, f"largest full-suite fp16 macro, no other arm within {BAND}"
        else:
            fewest = min(SHAPE[rows[k]["arm"]][:2] for k in near)
            simple = [k for k in near if SHAPE[rows[k]["arm"]][:2] == fewest]
            if len(simple) == 1:
                winner, why = simple[0], (f"within the {BAND} band of the best; fewest negatives "
                                          f"and fewest mining sources {fewest}")
            else:
                tm = [k for k in simple if SHAPE[rows[k]["arm"]][2] == "teacher"]
                winner = tm[0] if tm else sorted(simple)[0]
                why = ("parsimony also ties; teacher-mined preferred -- mining with the teacher "
                       "is a by-product of a document encode this system already performs, while "
                       "the BM25 arm needs a second retrieval system to reproduce"
                       if tm else "parsimony ties and no teacher-mined arm survives; "
                                  "alphabetical, recorded as arbitrary")

    out = {"_what": "the pre-registered negatives rule, executed on the step-rule-corrected arms",
           "_protocol": "m7/LEDGER.md 'Negatives ablation' (bar, Holm, three tie-break levels) and "
                        "'The step-selection rule was NOT applied to the negatives arms'",
           "_status": "exploratory dev SELECTION evidence; the only confirmatory comparisons are "
                      "the three frozen-test ones in the final run",
           "_ood_disclosure": "ood_macro_fp16 covers cqadup-programmers and cqadup-physics, the "
                              "only dev components outside the TRAIN mix and its Wikipedia family. "
                              "Mandatory disclosure per LEDGER.md; it does not enter the bar.",
           "compare_artifact": p.name, "baseline": base,
           "control": CONTROL, "control_proxy_macro": control_proxy,
           "arms": rows, "holm_family": family,
           "holm_alpha0.05_per_precision": holm, "survivors": survivors,
           "winner": winner, "tie_break_reason": why,
           "baseline_macro_fp16": d["macros_unrounded"][f"{base}|fp16"]}
    (REPO / "results" / "m7_negatives_decision.json").write_text(json.dumps(out, indent=1))

    print(f"baseline {base}: {out['baseline_macro_fp16']:.4f}   control proxy {control_proxy:.4f}")
    for k, r in sorted(rows.items(), key=lambda kv: -kv[1]["macro_fp16"]):
        mark = "*" if k == winner else (" " if k in survivors else "x")
        print(f" {mark} {k:26s} fp16 {r['macro_fp16']:.4f} (ood {r['ood_macro_fp16']:.4f})  "
              f"d {r['delta_raw_fp16']:+.4f} ci {[round(x, 4) for x in r['ci95_raw_fp16']]} "
              f"p={r['p_fp16']:.4f}/{r['p_int8']:.4f}  proxy {r['proxy_macro']:.4f}"
              f"{'' if r['arm'] else '  [descriptive]'}")
    print(f"\nHolm family {family}\nsurvivors {survivors}\nWINNER {winner} -- {why}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "steprule")
