"""G8: how many times has M8 looked at the dev suite? Count it from the artifacts, never recall it.

§14 G8 promises `results/m8_dev_reuse_count.json` "from evaluation #1" and it was absent from HEAD
for the whole milestone (§15, "Recorded, unfixed"). This closes that.

WHY IT MATTERS HERE and not only at freeze time. M8's nested selection protects the RESERVED four:
`D2` selects its vocabulary on the wikipedia+heldout groups and is barred on the out-of-domain
macro, so the bar never reads the selection endpoint. That is a real protection and it is the one
the registry claims. It is NOT a claim that the dev suite is fresh: every M8 probe so far has been
adjudicated on the same six dev components, M7 handed over 322 in-training dev evaluations before
M8 began, and the effects M8 is now chasing are at the 0.005 scale -- the scale at which accumulated
adaptive reuse stops being a footnote. The number belongs beside every dev-read verdict, so this
counts M7's inherited reuse and M8's added reuse separately and reports the total.

WHAT IS COUNTED, and the unit for each:
  * TRAINED ARMS -- `work/runs/*.json` whose run id is M8's (`m8*`). Smokes are excluded BY NAME.
  * IN-TRAINING EVALUATIONS -- every phase A/B/C eval row in those arms' histories. This is the
    honest unit: an arm evaluated every 500 steps for 2,500 steps looked at dev five times.
  * EVAL-ONLY VARIANTS -- artifacts that trained nothing and scored existing tables on the suite
    (the crossed floor's nine cells, the fused floor's twenty variants, the E14 head arms). Read
    off each artifact's own keys, so the count follows what ran rather than what this file recalls.
  * CLOSED-FORM DEV READS -- `D2-PRE` scores four arms plus a comparator on five folds. No training,
    but each is a look at the dev suite and none of them would be visible in `work/runs`.

WHAT IS NOT COUNTED, said plainly so the number is not mistaken for something it is not: the
RESERVED four have never been scored and their single access is unspent. This counts DEV only.
"""
import json
import re
import sys

import m8base

RESULTS = m8base.RESULTS
OUT = RESULTS / "m8_dev_reuse_count.json"
M8_RUN = re.compile(r"^m8[a-z0-9]*[-_]")
SMOKE = re.compile(r"(^|[-_])smoke([-_]|$)")


def _arms():
    m8_arms, m7_arms, skipped = {}, {}, []
    runs = sorted((m8base.WORK / "runs").glob("*.json"))
    if not runs:
        raise SystemExit("no run artifacts under work/runs -- a counter that returns 0 because it "
                         "found nothing is not a count (CODEMAP pitfall 17).")
    for p in runs:
        rid = p.stem
        if rid.endswith((".meta", ".fusion", ".head", ".release")):
            continue
        if SMOKE.search(rid):
            skipped.append(rid)
            continue
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        if "history" not in d:
            continue
        n = sum(1 for e in d["history"] if e.get("phase") in ("A", "B", "C"))
        (m8_arms if M8_RUN.match(rid) else m7_arms)[rid] = n
    return m8_arms, m7_arms, sorted(skipped)


def _eval_only():
    """Variants scored without training, counted from each artifact's own keys."""
    out, missing = {}, []
    for name in ("m7_compare_full_m8xfloor.json", "m7_compare_full_m8nfbfloor.json",
                 "m7_compare_full_m8noise.json", "m7_compare_full_m8b3.json",
                 "m8_noise_floor_fused.json"):
        p = RESULTS / name
        if not p.exists():
            missing.append(name)
            continue
        d = json.loads(p.read_text())
        keys = list(d.get("macros_unrounded", {})) or list(d.get("arm_macros", {}))
        out[name] = len(keys)
    for p in sorted(RESULTS.glob("m7_compare_full_m8e14head-*.json")):
        d = json.loads(p.read_text())
        out[p.name] = len(d.get("macros_unrounded", {}))
    if not out:
        raise SystemExit("no eval-only artifacts found; refusing to report 0 (CODEMAP pitfall 17).")
    return out, missing


def _closed_form():
    p = RESULTS / "m8_d2_pre.json"
    if not p.exists():
        return {"D2-PRE": 0, "_note": "not yet run"}
    d = json.loads(p.read_text())
    n_arms, n_folds = len(d.get("oof_macro", {})), d.get("n_folds", 0)
    # every arm on every fold, plus the shared comparator once per fold, times two components
    return {"D2-PRE": (n_arms * n_folds + n_folds) * 2,
            "_note": f"{n_arms} arms x {n_folds} folds plus the comparator, over the two "
                     f"out-of-domain components"}


def main():
    m8_arms, m7_arms, smokes = _arms()
    evalonly, missing = _eval_only()
    closed = _closed_form()
    m7 = json.loads((RESULTS / "m7_dev_reuse_count.json").read_text())
    m8_evals = sum(m8_arms.values())
    total_m8 = m8_evals + sum(evalonly.values()) + closed["D2-PRE"]
    out = {
        "_what": __doc__.strip().splitlines()[0],
        "_scope": "DEV only. The reserved four have never been scored; their one access is unspent.",
        "m8": {"trained_arms": len(m8_arms), "in_training_dev_evaluations": m8_evals,
               "eval_only_variants_by_artifact": evalonly,
               "eval_only_variants_total": sum(evalonly.values()),
               "closed_form_dev_reads": closed,
               "total_dev_looks": total_m8, "per_arm": m8_arms},
        "inherited_from_m7": {"trained_arms": m7["trained_arms"],
                              "in_training_dev_evaluations": m7["in_training_dev_evaluations"],
                              "eval_only_variants_total": m7["eval_only_variants_total"]},
        "cumulative_in_training_dev_evaluations":
            m7["in_training_dev_evaluations"] + m8_evals,
        "smokes_excluded_by_name": smokes,
        "artifacts_expected_but_absent": missing,
        "m7_era_arms_on_disk_not_counted_as_m8": len(m7_arms),
        "_caveat": "These are counts of LOOKS, not of independent hypotheses: many arms share a "
                   "checkpoint or differ in one knob, and the pre-registered bars applied Holm "
                   "only within a named family. NO multiplicity correction spans the search and "
                   "none is claimed. What the nested split buys is narrower than it reads: it "
                   "keeps D2's vocabulary SELECTION off the endpoint its bar reads, and it keeps "
                   "the RESERVED four untouched. It does not make the out-of-domain macro a fresh "
                   "surface -- by this count it has been read hundreds of times across M7 and M8, "
                   "and the effects M8 is chasing are at the 0.005 scale.",
    }
    OUT.write_text(json.dumps(out, indent=1))
    print(f"M8: {len(m8_arms)} trained arms, {m8_evals} in-training dev evaluations, "
          f"{sum(evalonly.values())} eval-only variants, {closed['D2-PRE']} closed-form reads "
          f"-> {total_m8} dev looks")
    print(f"cumulative in-training dev evaluations (M7+M8): "
          f"{out['cumulative_in_training_dev_evaluations']}")
    if missing:
        print(f"absent artifacts (not counted): {missing}")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
