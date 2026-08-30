"""B3's bar, executed (registry probe `B3`, LEDGER §9).

A pre-registered rule a session can re-read in its own favour is not a pre-registration. This
project has already paid for that lesson once -- M7's step-selection rule sat in prose and went
unapplied for four arms -- so B3's verdict runs as code against committed artifacts and writes
itself out, and the constants come from `m8/registry.json` rather than being restated here.

THE TWO SCALARS, both at int8 / sqrt (the release format and R0's adopted pooling rule):
  DENSE  = out-of-domain macro: equal-weight nDCG@10 over cqadup-programmers and cqadup-physics.
  FUSED  = the 4-component `fused_macro` (nq-250k, hotpotqa, cqadup-programmers, cqadup-physics)
           under the FROZEN fusion operator.
The fused scalar is deliberately NOT the group-vector median: the two held-out dev components
carry pool row indices rather than document text, so no fused read exists for them at all, and the
fused floor this bar cites was measured on the 4-component macro. An earlier version of the
registration named a quantity that does not exist while citing a floor for a different one.

THE ORDER IS LOAD-BEARING, because each step can end the probe:
  0. NEGATIVE-DOSE -- 1.00 vs 0.25 mean gain <= -bar on BOTH scalars. More distinct pairs made the
     table worse at fixed compute; a pool-quality finding in its own right.
  1. MANIPULATION CHECK -- 1.00 vs 0.25 must reach >= bar on BOTH scalars, else UNINFORMATIVE.
  2. PRIMARY -- 1.00 vs 0.50 mean-over-seeds gain >= bar on BOTH scalars -> PASS.
  3. otherwise FAIL, worded ONLY as the registry's `fail_wording` allows.

Seed sign agreement and the log-pairs slope are computed and REPORTED, and neither gates anything.
"""
import argparse
import gzip
import json
import sys

import numpy as np

import m8base
import b3_pool
import noise_floor
import probe_guard

RESULTS = m8base.RESULTS
OUT = RESULTS / "m8_b3_decision.json"
PREC, MODE = "int8", "sqrt"
DENSE_ENDPOINT = "out_of_domain_macro"


def _dense_scalars(dump_path):
    """{run_id: {endpoint: value}} at the pinned precision and pooling mode."""
    raw = json.loads(gzip.open(dump_path).read() if str(dump_path).endswith(".gz")
                     else open(dump_path).read())
    pq = raw["per_query"] if "per_query" in raw else raw
    out = {}
    for key, comps in pq.items():
        parts = key.split("|")
        if len(parts) < 2 or parts[-1] != PREC:
            continue
        rid, _, mode = parts[0].partition(":")
        if (mode or "mean") != MODE:
            continue
        out[rid] = noise_floor._group_vector(comps)
    return out


def _fused_scalars(fused_path):
    """{run_id: fused_macro} at the pinned precision and pooling mode."""
    d = json.loads(open(fused_path).read())
    out = {}
    for key, v in d["arm_macros"].items():
        rid, _, rest = key.partition(":")
        mode, _, prec = rest.partition("|")
        if prec == PREC and mode == MODE:
            out[rid] = float(v)
    return out


def _contrast(vals, frac_lo, seeds):
    """Mean over seeds of (value at 1.00) - (value at frac_lo), plus the per-seed gains."""
    gains = {}
    for s in seeds:
        hi, lo = b3_pool.arm_id(1.00, s), b3_pool.arm_id(frac_lo, s)
        if hi not in vals or lo not in vals:
            return None, {}, [hi if hi not in vals else lo]
        gains[s] = vals[hi] - vals[lo]
    return float(np.mean(list(gains.values()))), gains, []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, help="compare_full per-query dump over B3's arms")
    ap.add_argument("--fused", required=True, help="fused artifact carrying B3's arm_macros")
    a = ap.parse_args()

    reg = probe_guard.registry()["probes"]["B3"]
    bar = float(reg["bar_frozen"]["bar"])
    seeds = list(b3_pool.SEEDS)

    curve = b3_pool.collect()
    if curve["problems"]:
        raise SystemExit("B3's arms are not a dose curve; refusing to score them:\n  "
                         + "\n  ".join(curve["problems"]))

    dense_all = _dense_scalars(a.dump)
    dense = {r: v[DENSE_ENDPOINT] for r, v in dense_all.items()}
    fused = _fused_scalars(a.fused)
    scal = {"dense": dense, "fused": fused}

    contrasts, missing = {}, []
    for name, lo in (("manipulation_1.00_vs_0.25", 0.25), ("primary_1.00_vs_0.50", 0.50),
                     ("descriptive_1.00_vs_0.75", 0.75)):
        contrasts[name] = {}
        for which, vals in scal.items():
            mean, gains, miss = _contrast(vals, lo, seeds)
            missing += miss
            contrasts[name][which] = {
                "mean_gain": mean, "per_seed_gain": gains,
                "seed_signs_agree": (bool(len({np.sign(g) for g in gains.values()}) == 1)
                                     if gains else None),
                "meets_bar": (mean is not None and mean >= bar),
                "at_or_below_negative_bar": (mean is not None and mean <= -bar),
            }
    if missing:
        raise SystemExit(f"arms absent from the scored artifacts: {sorted(set(missing))}")

    def both(name, field):
        return all(contrasts[name][w][field] for w in ("dense", "fused"))

    man, pri = "manipulation_1.00_vs_0.25", "primary_1.00_vs_0.50"
    if both(man, "at_or_below_negative_bar"):
        verdict, headline = "NEGATIVE-DOSE", (
            "More distinct pairs made the table WORSE at fixed compute, by at least the bar on "
            "both scalars. This is a finding about pool quality, not a failed lever, and it is "
            "reported as such rather than folded into FAIL or UNINFORMATIVE.")
    elif not both(man, "meets_bar"):
        verdict, headline = "UNINFORMATIVE", (
            "A 4x dose (0.25 -> 1.00 of the pool) moved neither scalar by the bar. "
            + reg["uninformative_action"])
    elif both(pri, "meets_bar"):
        verdict, headline = "PASS", (
            "Phase A is pair-starved at the frozen recipe: doubling the pool from half to all of "
            "it buys at least the bar on both scalars.")
    else:
        verdict, headline = "FAIL", reg["fail_wording"]

    # DESCRIPTIVE ONLY. Slope of each scalar against log2(pairs), pooled over seeds.
    slopes = {}
    counts = curve["pair_counts_by_fraction"]
    for which, vals in scal.items():
        xs, ys = [], []
        for f in b3_pool.FRACTIONS:
            for s in seeds:
                rid = b3_pool.arm_id(f, s)
                if rid in vals and f in counts:
                    xs.append(np.log2(counts[f])); ys.append(vals[rid])
        slopes[which] = (float(np.polyfit(xs, ys, 1)[0]) if len(set(xs)) > 1 else None)

    doublings = {w: (bar / sl if sl and sl > 0 else None)
                 for w, sl in slopes.items()}
    out = {
        "_note": __doc__.strip().splitlines()[0],
        "what_would_reach_the_bar": {
            "doublings_of_the_pool_needed": doublings,
            "multiple_of_the_current_pool": {w: (2 ** d if d else None)
                                             for w, d in doublings.items()},
            "_note": "extrapolating the fitted log-linear slope, which the data do not license "
                     "beyond the measured range -- this is a magnitude, not a forecast. It is "
                     "here because 'the lever is too small to see' is only actionable alongside "
                     "how much more pool it would take to see it.",
        },
        "read_at": f"{PREC} / {MODE}",
        "dense_endpoint": DENSE_ENDPOINT,
        "fused_endpoint": "4-component fused_macro under the frozen operator",
        "bar": bar,
        "pair_counts_by_fraction": counts,
        "arm_scalars": {w: {r: v for r, v in sorted(vals.items())} for w, vals in scal.items()},
        "contrasts": contrasts,
        "verdict": verdict,
        "headline": headline,
        "descriptive": {
            "slope_per_doubling_of_pairs": slopes,
            "_slope_note": "gain in the scalar per doubling of distinct pairs at FIXED compute; "
                           "descriptive, gates nothing.",
            "group_vector_median": {r: v.get("group_vector_median")
                                    for r, v in sorted(dense_all.items())},
            "seed_sign_agreement": {n: {w: contrasts[n][w]["seed_signs_agree"]
                                        for w in ("dense", "fused")} for n in contrasts},
            "_sign_note": "reported, never a gate: at n=2 it was nearly vacuous evidence and cost "
                          "real power, so the registration dropped it as a requirement.",
        },
        "known_biases": reg["known_biases"],
        "scope": reg["no_survivor"],
    }
    probe_guard.write_result(OUT, out, "B3")
    print(json.dumps({"verdict": verdict, "headline": headline, "bar": bar,
                      "contrasts": {n: {w: {"mean_gain": contrasts[n][w]["mean_gain"],
                                            "meets_bar": contrasts[n][w]["meets_bar"]}
                                        for w in ("dense", "fused")} for n in contrasts},
                      "slope_per_doubling": slopes}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
