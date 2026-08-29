"""How many times has the dev suite been looked at? Count it, do not estimate it.

Written for the pre-freeze adversarial review, which asks how many adaptive decisions the dev
suite has absorbed. Every honest answer this project can give about generalization depends on
that number, and "a lot" is not an answer a reviewer can check.

What is counted, and why each category counts:
  * TRAINED ARMS -- every artifact in work/runs. Each was scored on the in-training proxy at
    least once, and the arm that survived was chosen partly on that score. Smokes and the
    per-artifact dev sidecars are excluded by name: a 90-step smoke is a code check, not an
    experiment.
  * IN-TRAINING EVALUATIONS -- every eval row inside those arms' histories. This is the honest
    unit for the step-selection rule: an arm evaluated every 500 steps for 4,000 steps looked at
    dev eight times, and the rule then picks the best of the eight.
  * EVAL-ONLY ARMS -- the lever arms that trained nothing but scored a variant of an existing
    table on the suite (pooling modes, shrinkage taus, quantizations, path-equivalence variants).
    Read off the committed comparison artifacts' own `macros_unrounded` keys, so the count follows
    what was actually run rather than what this file remembers.

What is NOT counted, stated so the number is not mistaken for something it is not: the six
confirmatory datasets have been accessed the three times m7/LEDGER.md enumerates, none of them a
model comparison, and the final run will be the fourth. This file counts DEV reuse only.

Usage: dev_reuse.py
"""
import json
import re

from _paths import REPO, WORK

# Not experiments: 90-step code checks, and the per-artifact dev-cache sidecars.
EXCLUDE = re.compile(r"^(smoke|smoke-chain-[ab]|dev)$")


def main():
    arms, evals = {}, 0
    for p in sorted((WORK / "runs").glob("*.json")):
        rid = p.stem
        if rid.endswith(".meta") or rid.endswith(".fusion") or EXCLUDE.match(rid):
            continue
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        if "history" not in d:
            continue
        # the "final" row repeats the last eval; counting it would inflate every arm by one
        n = sum(1 for e in d["history"] if e.get("phase") in ("A", "B", "C"))
        arms[rid] = {"in_training_evals": n, "final_proxy_macro": d.get("final_macro")}
        evals += n

    # Eval-only variants, read from the committed comparison artifacts rather than listed here.
    evalonly, sources = {}, []
    for name in ("m7_dev_audit_full.json", "m7_lever4_pooling_p4n-teacher16-a.json",
                 "m7_lever4_pooling_p35w-2m-s2500.json", "m7_lever5_shrinkage.json",
                 "m7_compare_full_postabl.json", "m7_compare_full_lever6.json",
                 "m7_compare_full_steprule.json", "m7_compare_full_simplify.json",
                 "m7_compare_full_lever7.json"):
        p = REPO / "results" / name
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        keys = list(d.get("macros_unrounded", {}))
        if not keys and "arms" in d:                    # the lever artifacts' own shape
            keys = [f"{d.get('adjudicated_on', d.get('candidate', '?'))}|{m}|{q}"
                    for m in d["arms"] for q in ("fp16", "int8")]
        evalonly[name] = len(keys)
        sources.append(name)

    out = {
        "_what": "DEV reuse, counted from the artifacts rather than recalled. Written for the "
                 "pre-freeze review's question: how many adaptive decisions has the dev suite "
                 "absorbed?",
        "_scope": "Dev only. The six confirmatory datasets have the three accesses m7/LEDGER.md "
                  "enumerates, none a model comparison; the final run is the fourth.",
        "trained_arms": len(arms),
        "in_training_dev_evaluations": evals,
        "eval_only_variants_by_artifact": evalonly,
        "eval_only_variants_total": sum(evalonly.values()),
        "sources_read": sources,
        "per_arm": arms,
        "_caveat": "These are counts of LOOKS, not of independent hypotheses -- many arms share a "
                   "checkpoint or differ in one knob, and the pre-registered bars applied Holm "
                   "only within a named family. No multiplicity correction spans the whole "
                   "search, and none is claimed. The number is here so the report states the "
                   "scale of adaptive dev reuse instead of implying it was small.",
    }
    (REPO / "results" / "m7_dev_reuse_count.json").write_text(json.dumps(out, indent=1))
    print(f"trained arms {len(arms)}, in-training dev evaluations {evals}, "
          f"eval-only variants {out['eval_only_variants_total']} over {len(sources)} artifacts")
    print("wrote results/m7_dev_reuse_count.json")


if __name__ == "__main__":
    main()
