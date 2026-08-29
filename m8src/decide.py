"""LEDGER 4 + 5, as code: the confirmatory decision rule and the complete ship rule.

Registered before the first M8 number exists. **`m8/registry.json` is the authority** for every
constant this module applies -- the ledger's prose renders it and this module reads it. It does
NOT restate them: a restatement went stale within hours once (this file carried its own
qualifying-key vocabulary while the registry carried a different one, so a manifest written in
either vocabulary would have failed the other), which is the precise disaster mode section 5.4
exists to prevent. Any change to the registry or to this module is a LEDGER 15 amendment.

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

import m8base                                    # noqa: F401  (sys.path + G2 guard)
import boot                                       # m7src, the frozen statistics machinery
import probe_guard                                # the registry's one change classifier


def _registry():
    return json.loads((m8base.REPO / "m8" / "registry.json").read_text())

ALPHA = 0.025
M = 3                                             # C1, C2, C3
BONF_LEVEL = ALPHA / M                            # 0.0083333...
POINT_GUARD_C1 = 0.005                            # LEDGER 5.5, mirrors registry.point_guard_c1
LEGS = ("C1", "C2", "C3")

# The qualifying-key vocabulary lives in m8/registry.json and is read through
# probe_guard.classify_change. There is deliberately no copy of it here.


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


def decide_family(legs, per_query=None):
    """legs: {name: leg(...)}. Applies Holm over the sign-flip p's, then the two CI legs.
    A leg is resolved only if all three pass.

    `per_query` (optional): {name: (a, b)} so the alpha/3 bound can be recomputed at the EXACT
    level and cross-checked against boot.paired's rounded percentile (LEDGER 4.1)."""
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
        exact = None
        if per_query and name in per_query:
            exact = exact_bonferroni_lower(*per_query[name])
            if bonf is not None and abs(exact - bonf) > 1e-6:
                raise AssertionError(
                    f"{name}: the exact alpha/3 bound {exact!r} and boot.paired's rounded "
                    f"percentile {bonf!r} differ by {abs(exact - bonf):.2e} > 1e-6. A decision "
                    f"may not read a bound whose level is uncertain (LEDGER 4.1).")
            legs_pass["bonferroni_lower_gt0"] = bool(exact > 0)
        out[name] = {**r, "holm": holm[name], "bonferroni_lower_raw": bonf,
                     "bonferroni_lower_exact": exact,
                     "bonferroni_level_pct": BONF_LEVEL * 100,
                     "legs": legs_pass, "resolved": all(legs_pass.values())}
    return out


def _bonf_lookup(one_sided):
    """boot.paired keys the simultaneous bound by percentile as the ROUNDED string '0.8333'."""
    if not one_sided:
        return None
    for k, v in one_sided.items():
        if abs(float(k) - BONF_LEVEL * 100) < 1e-3:
            return float(v)
    return None


def exact_bonferroni_lower(a, b, B=boot.B, seed=boot.SEED):
    """The alpha/3 simultaneous lower bound at the EXACT level, not boot.paired's rounded
    '0.8333' percentile string.

    The ledger requires the bound to be computed at 100 x 0.025/3 = 0.83333...%, and requires the
    two to be checked against each other rather than assumed equal. This reproduces boot.paired's
    bootstrap draw for draw -- same RNG, same seed, same per-dataset loop order over
    sorted(set(a) & set(b)), same accumulation -- and takes the exact quantile. If this ever stops
    matching boot.paired's rounded value to 1e-6, that is a real divergence and the assertion in
    decide_family() will surface it instead of a decision quietly reading the wrong number.
    """
    pairs = boot._align(a, b, strict=True)
    rng = np.random.default_rng(seed)
    k = len(pairs)
    deltas = np.zeros(B)
    for _ds, (da, db) in pairs.items():
        n = len(da)
        idx = rng.integers(0, n, size=(B, n))
        deltas += (da[idx].mean(1) - db[idx].mean(1)) / k
    return float(np.percentile(deltas, BONF_LEVEL * 100))


def worst_group(m8_per_query, m7_per_query):
    """LEDGER 5.6, with NO grouping choice left open: the groups ARE the four reserved datasets,
    individually, read from the registry.

    Aborts on a missing dataset rather than shrinking the guard -- a guard that quietly covers
    three datasets where the rule says four is worse than no guard, and silently scoring the
    intersection is the failure mode section 4.1 bans on every confirmatory path."""
    ship = _registry()["ship_rule"]
    members, thr = ship["worst_group_members"], ship["worst_group_threshold"]
    missing = [ds for ds in members if ds not in m8_per_query or ds not in m7_per_query]
    if missing:
        raise AssertionError(f"worst-group guard: {missing} absent from one of the inputs. It is "
                             f"defined over exactly {members} (LEDGER 5.6) and may not be "
                             f"evaluated over a subset.")
    rows = {ds: float(np.mean(list(m8_per_query[ds].values()))
                      - np.mean(list(m7_per_query[ds].values()))) for ds in members}
    worst = min(rows.values())
    return {"per_dataset": rows, "worst": worst, "threshold": thr, "members": members,
            "pass": bool(worst >= thr)}


def six_set_no_regression(m8_six, m7_six, margin=None):
    """LEDGER 5.7 -- the anti-memorization ship-blocker. One-directional: it can only block, never
    license. The margin comes from the registry, where its provenance (2.4x the measured
    near-sibling six-macro SE) is recorded.

    Strict alignment, like every other decision path: `boot._align(strict=True)` aborts on a
    missing dataset or qid rather than silently comparing two different statistics."""
    if margin is None:
        margin = _registry()["ship_rule"]["six_set_no_regression_margin"]
    boot._align(m8_six, m7_six, strict=True)          # aborts on any mismatch
    d = _macro(m8_six) - _macro(m7_six)
    return {"delta_raw": float(d), "margin": -abs(margin),
            "macro_m8": _macro(m8_six), "macro_m7": _macro(m7_six),
            "n_datasets": len(m8_six),
            "pass": bool(d >= -abs(margin))}


def qualifying_table(changes):
    """LEDGER 5.4. `changes` is the manifest's declared CONFIG-KEY diff against R0. Classification
    is delegated to the registry (probe_guard.classify_change) so there is exactly one vocabulary.

    An UNKNOWN key fails the condition: a key that was never classified cannot be argued into a
    category after a number exists. A qualifying_non_table change (doc_side_head) alone does not
    open the release path -- a qualifying TABLE change must also survive (E11 + G4-4)."""
    reg = _registry()["ship_rule"]
    changes = set(changes)
    kinds = {k: probe_guard.classify_change(k) for k in changes}
    unknown = sorted(k for k, v in kinds.items() if v == "unknown")
    table = sorted(k for k, v in kinds.items() if v == "qualifying_table")
    non_table = sorted(k for k, v in kinds.items() if v == "qualifying_non_table")
    # A teacher swap flips tokenizer_id/vocab as a SIDE EFFECT. Those are qualifying-table keys,
    # but a swap is not an E11-sense v2 lever, so it may not satisfy condition 4 on its own.
    swap = bool(changes & set(reg.get("neutral_keys", {}).get("keys", []))
                & {"teacher", "teacher_id", "encoder_spec"})
    side = set(reg.get("swap_side_effect_keys", []))
    substantive = [k for k in table if not (swap and k in side)]
    return {"declared": sorted(changes), "classified": kinds, "unknown": unknown,
            "qualifying_table": table, "qualifying_non_table": non_table,
            "teacher_swap_in_diff": swap, "qualifying_table_net_of_swap": substantive,
            "pass": bool(substantive) and not unknown,
            "note": ("a distinct int8 payload is necessary but not sufficient; a "
                     "qualifying_non_table change alone does not satisfy the requirement "
                     "(LEDGER 5.4, E11/G4-4). An unknown key FAILS.")}


def ship(family, *, qualifying, worst, six_guard):
    """The complete rule. Returns SHIP / NO-SHIP with every condition's verdict, so a miss is
    reportable as the pre-registered publishable outcome rather than argued about."""
    c1 = family["C1"]
    point_guard = {"delta_raw": c1["delta_raw"], "threshold": POINT_GUARD_C1,
                   "pass": bool(c1["delta_raw"] > POINT_GUARD_C1),      # STRICT, as registered
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
    ds = m8base.RESERVED_FOUR
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
    fam = decide_family(legs, per_query={"C1": (m8, m7), "C2": (m8, m7), "C3": (m8, bm25)})
    six = {d: {f"q{i:07d}": float(v) for i, v in enumerate(rng.random(300))}
           for d in m8base.SIX}
    verdict = ship(
        fam,
        qualifying=qualifying_table({"pool_composition", "tokenizer_id", "steps_a"}),
        worst=worst_group(m8, m7),
        six_guard=six_set_no_regression(six, six))
    return {"legs": {k: {kk: v[kk] for kk in
                         ("delta_raw", "signflip_p", "ci95_raw", "bonferroni_lower_raw",
                          "bonferroni_lower_exact", "legs", "resolved")}
                     for k, v in fam.items()},
            "verdict": verdict}


if __name__ == "__main__":
    out = self_test()
    print(json.dumps(out, indent=2, default=float))
    assert out["verdict"]["ship"] is True, "self-test: a +0.02 uniform shift should ship"
    assert all(out["legs"][k]["resolved"] for k in LEGS)
    # And the negative control: no qualifying table change must block an otherwise perfect result.
    print("\nself-test OK", file=sys.stderr)
