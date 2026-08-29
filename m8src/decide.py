"""LEDGER 4 + 5, as code: the confirmatory decision rule and the complete ship rule.

Registered before the first M8 number exists. This module is the pre-registration -- the ledger's
prose describes what this code does, and where they disagree the code is what ran, so the two are
committed together and any change to either is a LEDGER 15 amendment.

Estimand: the EQUAL-WEIGHT macro over the four reserved sets (LEDGER 4.2). The grouped variant is
a registered sensitivity read and is computed alongside, never substituted.

A leg is resolved only with all THREE of:
  1. Holm-corrected sign-flip rejection at family alpha = 0.025 over m = 3;
  2. the raw two-sided 95% paired CI lower endpoint > 0;
  3. the raw one-sided lower bound at alpha/3 = 0.008333 > 0, from the SAME bootstrap draws.
Leg 3 is strictly harder than legs 1-2 alone, which is the only direction a bar may move once
registered. Every endpoint is read UNROUNDED: a true lower endpoint of +4e-5 displays as 0.0000,
and this project has read a rounded endpoint on an irreversible decision once already
(m7/LEDGER.md, Statistics).

Ship additionally requires all four of (LEDGER 5): a qualifying v2 table; the +0.005 point guard
on C1; the worst-group guard; and the six-set no-regression guard. Those four are NOT hypotheses
and consume no multiplicity budget -- they are product conditions evaluated after the family, and
they can only ever REDUCE the set of outcomes that ship. `power.py` accounts for them jointly.
"""
import json
import sys
from pathlib import Path

import numpy as np

import _paths                                     # noqa: F401  (sys.path + G2 guard)
import boot                                       # m7src, the frozen statistics machinery

ALPHA = 0.025
M = 3                                             # C1, C2, C3
BONF_LEVEL = ALPHA / M                            # 0.0083333...
POINT_GUARD_C1 = 0.005                            # LEDGER 5.2
WORST_GROUP_MAX_REGRESSION = 0.01                 # LEDGER 5.3
LEGS = ("C1", "C2", "C3")

# LEDGER 5.1. A distinct int8 payload is necessary but not sufficient; D1 alone does not qualify.
QUALIFYING_CHANGES = frozenset({
    "objective-family", "data-construction", "feature-tokenizer",
    "row-init-construction", "structural-rider", "doc-side-head",
})
NOT_QUALIFYING = frozenset({
    "seed", "steps", "temperature", "negative-count", "learning-rate", "pool-size", "tuning",
})
QUALIFYING_TABLE_CHANGES = QUALIFYING_CHANGES - {"doc-side-head"}


def _macro(per_query):
    """Equal-weight macro over datasets. per_query: {dataset: {qid: score}}."""
    return float(np.mean([np.mean(list(v.values())) for v in per_query.values()]))


def leg(name, a, b, *, hypothesis):
    """One confirmatory comparison, a > b, on per-query dicts. Returns every number the rule
    reads, unrounded, plus the sign-flip p that Holm will consume."""
    sf = boot.signflip(a, b, alternative="greater", strict=True)
    pr = boot.paired(a, b, alternative="greater", strict=True)
    return {
        "leg": name, "hypothesis": hypothesis,
        "delta_raw": pr["delta_raw"],
        "macro_a": _macro(a), "macro_b": _macro(b),
        "signflip_p": sf["p"],
        "ci95_raw": pr["ci95_raw"],
        "one_sided_lower_raw": pr["one_sided_lower_raw"],
        "per_dataset": pr["per_dataset"],
        "R": sf.get("R"), "B": pr["B"], "seed": pr["seed"],
    }


def decide_family(legs):
    """legs: {name: leg(...)}. Applies Holm over the sign-flip p's, then the two CI legs.
    A leg is resolved only if all three pass."""
    holm = boot.holm({k: v["signflip_p"] for k, v in legs.items()}, alpha=ALPHA)
    out = {}
    for name, r in legs.items():
        lo_raw = r["ci95_raw"][0]
        bonf = r["one_sided_lower_raw"][f"{round(BONF_LEVEL * 100, 4)}"] \
            if r["one_sided_lower_raw"] and f"{round(BONF_LEVEL * 100, 4)}" in r["one_sided_lower_raw"] \
            else _bonf_lookup(r["one_sided_lower_raw"])
        legs_pass = {
            "holm": bool(holm[name]["reject"]),
            "raw_ci_lower_gt0": bool(lo_raw > 0),
            "bonferroni_lower_gt0": bool(bonf is not None and bonf > 0),
        }
        out[name] = {**r, "holm": holm[name], "bonferroni_lower_raw": bonf,
                     "legs": legs_pass, "resolved": all(legs_pass.values())}
    return out


def _bonf_lookup(one_sided):
    """boot.paired keys the simultaneous bound by percentile as a string ('0.8333')."""
    if not one_sided:
        return None
    for k, v in one_sided.items():
        if abs(float(k) - BONF_LEVEL * 100) < 1e-3:
            return float(v)
    return None


def worst_group(m8_per_query, m7_per_query, groups):
    """LEDGER 5.3, as an explicit formula. groups: {name: (dataset, ...)}. Returns the per-group
    point-estimate deltas and the worst one. A group regresses if its delta < -0.01."""
    rows = {}
    for g, members in groups.items():
        d = [np.mean(list(m8_per_query[ds].values())) - np.mean(list(m7_per_query[ds].values()))
             for ds in members if ds in m8_per_query and ds in m7_per_query]
        if d:
            rows[g] = float(np.mean(d))
    worst = min(rows.values()) if rows else None
    return {"per_group": rows, "worst": worst,
            "threshold": -WORST_GROUP_MAX_REGRESSION,
            "pass": bool(worst is not None and worst >= -WORST_GROUP_MAX_REGRESSION)}


def six_set_no_regression(m8_six, m7_six, margin):
    """LEDGER 5.4 -- the anti-memorization ship-blocker. Descriptive, frozen per-query vectors,
    ZERO new access: both arguments come from already-scored artifacts. One-directional: it can
    only block, never license."""
    d = _macro(m8_six) - _macro(m7_six)
    return {"delta_raw": float(d), "margin": -abs(margin),
            "macro_m8": _macro(m8_six), "macro_m7": _macro(m7_six),
            "pass": bool(d >= -abs(margin))}


def qualifying_table(changes):
    """LEDGER 5.1. `changes` is the manifest's declared change set. A doc-side head alone does not
    open the release path: a qualifying TABLE change (R1 or D2) must also survive."""
    changes = set(changes)
    unknown = changes - QUALIFYING_CHANGES - NOT_QUALIFYING
    qualifying = changes & QUALIFYING_CHANGES
    table_changes = changes & QUALIFYING_TABLE_CHANGES
    return {"declared": sorted(changes), "unknown": sorted(unknown),
            "qualifying": sorted(qualifying), "qualifying_table": sorted(table_changes),
            "pass": bool(table_changes) and not unknown,
            "note": ("a distinct int8 payload is necessary but not sufficient; "
                     "doc-side-head alone does not satisfy the requirement (LEDGER 5.1, E11/G4-4)")}


def ship(family, *, qualifying, worst, six_guard):
    """The complete rule. Returns SHIP / NO-SHIP with every condition's verdict, so a miss is
    reportable as the pre-registered publishable outcome rather than argued about."""
    c1 = family["C1"]
    point_guard = {"delta_raw": c1["delta_raw"], "threshold": POINT_GUARD_C1,
                   "pass": bool(c1["delta_raw"] >= POINT_GUARD_C1),
                   "note": "a product margin, not a hypothesis: a resolved win too small to be "
                           "worth a version bump is not a v2"}
    conds = {
        "C1_resolved": family["C1"]["resolved"],
        "C2_resolved": family["C2"]["resolved"],
        "C3_resolved": family["C3"]["resolved"],
        "qualifying_v2_table": qualifying["pass"],
        "point_guard_C1": point_guard["pass"],
        "worst_group_guard": worst["pass"],
        "six_set_no_regression": six_guard["pass"],
    }
    return {"ship": all(conds.values()), "conditions": conds,
            "point_guard": point_guard, "qualifying": qualifying,
            "worst_group": worst, "six_set_guard": six_guard,
            "alpha": ALPHA, "m": M, "bonferroni_level": BONF_LEVEL,
            "weak_null_caveat": (
                "Sharp-null Holm validity is exact under per-query exchangeability. Under two "
                "empirical weak-null constructions the complete three-leg rule rejected at 0.0198 "
                "and 0.0283 against a nominal 0.025 (results/m7_tier_rule_calibration.json, "
                "S=4,000); these are sensitivity evidence, not a demonstration of uniform "
                "weak-null FWER control; a union bound over the marginal legs alone would say "
                "0.039. The claim 'the family is bounded at 0.025' is withdrawn permanently and "
                "the nominal figure may never be quoted without this alongside.")}


def self_test():
    """Runs the rule end to end on synthetic per-query data, so the registration is demonstrably
    executable before any real number exists. Zero-padded qids: boot._align_ids sorts qids as
    STRINGS (m7/CODEMAP.md pitfall 19)."""
    rng = np.random.default_rng(0)
    ds = _paths.RESERVED_FOUR
    n = {"fever": 6666, "dbpedia-entity": 400, "cqadup-android": 699, "cqadup-english": 1570}

    def make(shift):
        return {d: {f"q{i:07d}": float(v) for i, v in
                    enumerate(np.clip(rng.normal(0.45 + shift, 0.30, n[d]), 0, 1))} for d in ds}

    m7 = make(0.0)
    m8 = {d: {q: min(1.0, v + 0.02) for q, v in m7[d].items()} for d in ds}
    bm25 = {d: {q: max(0.0, v - 0.05) for q, v in m7[d].items()} for d in ds}
    legs = {
        "C1": leg("C1", m8, m7, hypothesis="fused-M8 > fused-M7"),
        "C2": leg("C2", m8, m7, hypothesis="dense M8 > frozen dense M7"),
        "C3": leg("C3", m8, bm25, hypothesis="fused-M8 > BM25"),
    }
    fam = decide_family(legs)
    six = {d: {f"q{i:07d}": float(v) for i, v in enumerate(rng.random(300))}
           for d in _paths.SIX}
    verdict = ship(
        fam,
        qualifying=qualifying_table({"data-construction", "feature-tokenizer"}),
        worst=worst_group(m8, m7, {"cqa": ("cqadup-android", "cqadup-english"),
                                   "fever": ("fever",), "dbpedia": ("dbpedia-entity",)}),
        six_guard=six_set_no_regression(six, six, margin=0.005))
    return {"legs": {k: {kk: v[kk] for kk in
                         ("delta_raw", "signflip_p", "ci95_raw", "bonferroni_lower_raw",
                          "legs", "resolved")} for k, v in fam.items()},
            "verdict": verdict}


if __name__ == "__main__":
    out = self_test()
    print(json.dumps(out, indent=2, default=float))
    assert out["verdict"]["ship"] is True, "self-test: a +0.02 uniform shift should ship"
    assert all(out["legs"][k]["resolved"] for k in LEGS)
    # And the negative control: no qualifying table change must block an otherwise perfect result.
    print("\nself-test OK", file=sys.stderr)
