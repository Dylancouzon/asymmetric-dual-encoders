"""LEDGER 4.7 / G4 -- measure M8's noise floor, so every bar is set against a MEASURED quantity.

Why this runs before any bar is frozen. M7 adopted effects of +0.0040, +0.0065, +0.0038, +0.0023
and then measured, late, that changing a parameter nobody would report -- the step count -- moved
the same dev macro by 0.0027 to 0.0078. Every effect the project had adopted sat inside that band.
A bar chosen from intuition is a bar chosen from nothing; this file replaces the intuition with a
number.

THE DESIGN, and the correction the 2026-08-29 gate forced. The plan's original floor was "two
matched null replicates: a seed change and +/-10% A-steps". A step change is a TREATMENT, not a
null -- it can have a real effect, and calling its magnitude "noise" would license a bar that a
real effect could clear. So:

  * the FLOOR is measured from TRUE NULLS only: K = 3 arms, identical recipe, identical data,
    identical B checkpoint, differing ONLY in training seed;
  * floor(endpoint) = the MAX of the three pairwise |delta| on that endpoint -- the max, not the
    mean, because a bar must survive the unlucky pair;
  * bar(endpoint) = max(planning_minimum, 2 x floor(endpoint)), planning_minimum >= 0.0040;
  * the +/-10% A-steps perturbation IS still run, and reported BESIDE the floor as a
    recipe-sensitivity number -- never as the floor.

FRAME, disclosed. The floor is measured in the INCUMBENT teacher frame and the M7 data mix, from
the M7 candidate's own B checkpoint. LEDGER 6 step 5 already requires re-measurement if the
teacher swaps; the same applies if the data mix changes materially. What this measures is the
magnitude of seed-to-seed variation in the A phase that produces the shipped rows, with the B
checkpoint held fixed -- which is the shape of nearly every probe arm. **An arm that differs in
its B leg has a larger floor, and no bar may read such an arm until that floor is measured too.**

Endpoints, all from the full pinned dev suite through the released `QueryTable` path (never the
fast proxy): the registered group vector's median, the worst group, the out-of-domain macro, and
the all-six macro. Bars are per-endpoint; a multi-endpoint probe takes the max.
"""
import argparse
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import m8base
import probe_guard

REPO = m8base.REPO
WORK = REPO / "work"
OUT = REPO / "results" / "m8_noise_floor.json"

# The artifact whose recipe defines R0's A leg. Its cfg is read from disk and reused verbatim --
# an arm's base recipe must come from the artifact it varies against, never from a snapshot
# (m7/CODEMAP.md pitfall 15).
BASE_RUN = "p35w-2m-s2500"
B_BASE_RUN = "p35b-2m"          # the B checkpoint the A-leg floor holds fixed
SEEDS = (0, 1, 2)
STEP_PERTURBATIONS = (2250, 2750)          # +/-10% of 2500, the recipe-sensitivity arms
GROUPS = m8base.DEV_GROUPS


def base_cfg(run=BASE_RUN):
    meta = json.loads((WORK / "runs" / f"{run}.meta.json").read_text())["cfg"]
    return {k: v for k, v in meta.items() if k != "run_id"}


def bleg_plan():
    """The B-LEG floor: full chains (B then A) differing ONLY in seed.

    §4.7's floor holds the B checkpoint fixed, which is the shape of nearly every probe arm -- but
    R-PHASE restructures the B->A phases, and any pool or init change flows through the B leg, so
    those arms have a LARGER floor that the A-leg measurement does not bound. This measures it.

    Seed 0's chain already exists end to end (`p35b-2m` -> `m8nf-seed0`), so only seeds 1 and 2
    need training. And the expensive part is free: `pseudoq.build_decontaminated` does NOT take
    the cfg seed -- it uses the module default -- so the ~925K-span objective-B text set is
    identical across seeds and `encode_cached` hits M7's existing cache. A B-leg arm therefore
    costs its training, not a re-encode.
    """
    bcfg, acfg = base_cfg(B_BASE_RUN), base_cfg(BASE_RUN)
    arms = {}
    for s_ in SEEDS[1:]:
        b_id, a_id = f"m8nfb-seed{s_}-b", f"m8nfb-seed{s_}-a"
        arms[b_id] = {**bcfg, "seed": s_, "_kind": "bleg", "_value": s_, "_leg": "b"}
        arms[a_id] = {**acfg, "seed": s_, "init": f"run:{b_id}",
                      "_kind": "bleg", "_value": s_, "_leg": "a", "_after": b_id}
    for rid, a in arms.items():
        a["_exists"] = (WORK / "runs" / f"{rid}.npz").exists()
    return arms


# ---- the crossed B x A design -------------------------------------------------------------
#
# THE STATED MOTIVE WAS WRONG, AND CHECKING IT IS WHY THIS IS WORTH RUNNING ANYWAY. The B-leg
# floor's own artifact says the aliasing -- one seed driving both legs -- makes that floor an
# UNDER-estimate, "the anti-conservative direction". That does not follow. A diagonal cell is
# B_s + A_s + e; if the two leg effects are independent draws its variance is exactly
# sigma_B^2 + sigma_A^2 + sigma_e^2, which IS the chain variance the floor is trying to estimate.
# Simulated at 200,000 replicates: the aliased diagonal and an independent chain have the same SD
# to four digits (0.003202 vs 0.003202) and the same E[range] (0.005425 vs 0.005416). Aliasing
# would bias the floor downward only under NEGATIVE correlation between the two legs' effects at a
# shared seed, and no mechanism for that has been named. So the diagonal is unbiased-but-noisy,
# not anti-conservative, and the claim is withdrawn here rather than inherited.
#
# WHAT THE CROSSED GRID ACTUALLY BUYS, which is still worth the four arms:
#   1. DECOMPOSITION. Is a B-leg-varying arm's floor really larger than an A-leg arm's? That is
#      an assertion the ledger makes and no measurement supports. If sigma_B is small, the
#      measured A-leg floor already covers R-PHASE and every pool-or-init lever, and a whole
#      gap-list entry closes. If it is large, the size of the gap is finally a number.
#   2. PRECISION. Nine cells with 4 residual df instead of a K=3 sample range whose CV is 0.525
#      (m8/CODEMAP.md pitfall 18: one range pins sigma only to a 12x span).
#   3. It TESTS the negative-correlation escape hatch directly: that is what the residual
#      (interaction) term measures, and the diagonal cells sit in the same grid as the rest.
#
# It is also nearly free: the three chains on disk are the DIAGONAL of the 3x3, and the A-leg
# floor arms are its FIRST ROW (all three init from `p35b-2m`, the seed-0 B checkpoint), so five
# of the nine cells already exist and only four A legs have to be trained.
B_CHECKPOINTS = {0: "p35b-2m", 1: "m8nfb-seed1-b", 2: "m8nfb-seed2-b"}

# Cells already trained. Row b=0 is the A-leg floor (three A seeds over the seed-0 B checkpoint);
# the diagonal is the B-leg floor (one seed driving both legs).
EXISTING_CELLS = {
    (0, 0): "m8nf-seed0", (0, 1): "m8nf-seed1", (0, 2): "m8nf-seed2",
    (1, 1): "m8nfb-seed1-a", (2, 2): "m8nfb-seed2-a",
}


def crossed_cells():
    """(B-checkpoint seed, A seed) -> run id, for the full 3x3."""
    return {(b, a): EXISTING_CELLS.get((b, a), f"m8nfx-b{b}a{a}")
            for b in SEEDS for a in SEEDS}


def crossed_plan():
    """The four A legs the 3x3 is missing. Same A recipe as every other cell; only the seed and
    the B checkpoint it inits from vary."""
    acfg = base_cfg(BASE_RUN)
    arms = {}
    for (b, a), rid in crossed_cells().items():
        if (b, a) in EXISTING_CELLS:
            continue
        arms[rid] = {**acfg, "seed": a, "init": f"run:{B_CHECKPOINTS[b]}",
                     "_kind": "crossed", "_value": f"b{b}a{a}", "_b": b, "_a": a}
    for rid, x in arms.items():
        x["_exists"] = (WORK / "runs" / f"{rid}.npz").exists()
    return arms


def arm_id(kind, value):
    return f"m8nf-{kind}{value}"


def plan():
    """Every arm this needs, and whether it already exists on disk."""
    cfg = base_cfg()
    arms = {}
    for s in SEEDS:
        arms[arm_id("seed", s)] = {**cfg, "seed": s, "_kind": "seed", "_value": s}
    for st in STEP_PERTURBATIONS:
        arms[arm_id("steps", st)] = {**cfg, "seed": SEEDS[0], "steps_a": st,
                                     "_kind": "steps", "_value": st}
    for rid, a in arms.items():
        a["_exists"] = (WORK / "runs" / f"{rid}.npz").exists()
    return cfg, arms


def train(rids=None, smoke=False, plan_fn=None):
    """Train the arms. ONE PROCESS PER ARM (m7/CODEMAP.md pitfall 14: a driver that runs a night
    of arms in one process accumulates every memoized cache and the third arm thrashes)."""
    arms = plan_fn() if plan_fn else plan()[1]
    # B legs before the A legs that init from them -- dict order is insertion order, and bleg_plan
    # inserts each pair in that order, so this is already correct; asserted rather than assumed.
    todo = [(r, a) for r, a in arms.items() if (rids is None or r in rids) and not a["_exists"]]
    for i, (r, a) in enumerate(todo):
        dep = a.get("_after")
        if dep and dep in {x for x, _ in todo[i + 1:]}:
            raise SystemExit(f"{r} inits from {dep}, which is scheduled AFTER it")
    if not todo:
        print("every arm already on disk")
        return
    print(f"{len(todo)} arms to train: {[r for r, _ in todo]}", flush=True)
    for rid, a in todo:
        over = {k: v for k, v in a.items() if not k.startswith("_")}
        if smoke:
            # A smoke must NOT occupy the real run id. The first version wrote 90-step artifacts
            # to `m8nf-seed0` etc, so plan()'s `_exists` check would then have reported the real
            # arms as already trained and the floor would have been measured on 90-step tables.
            over["steps_a"] = 90
            rid = rid + "-smoke"
        # `sweep.one` CATCHES the exception, appends a FAILED row and returns None -- so a broken
        # arm exits 0, and a driver that only checks the return code marches on and measures a
        # floor from whatever survived. `sys.exit(2)` on None is what makes the check real.
        code = (
            "import os, sys\n"
            "os.environ.setdefault('M7_ENCODER', 'stella-400M-v5')\n"
            "sys.path.insert(0, %r); sys.path.insert(0, %r)\n"
            "import program, sweep\n"
            "d = sweep.one(%r, program.BASE, **%r)\n"
            "print('DEVPROXY', %r, d, flush=True)\n"
            "sys.exit(0 if d is not None else 2)\n"
            % (str(REPO / "m7src"), str(REPO / "bench"), rid, over, rid))
        t0 = time.time()
        print(f"[{rid}] launching (steps_a={over.get('steps_a')}, seed={over.get('seed')})",
              flush=True)
        r = subprocess.run([sys.executable, "-u", "-c", code], cwd=str(REPO))
        print(f"[{rid}] exit {r.returncode} in {time.time()-t0:.0f}s", flush=True)
        if r.returncode != 0:
            raise SystemExit(f"{rid} failed (exit {r.returncode}); stopping rather than measuring "
                             f"a floor from a partial arm set. Its FAILED row is in "
                             f"m7/RESULTS.md.")


def _group_vector(per_q):
    """per_q: {component: {qid: score}} -> the registered group vector and its summaries.

    ABORTS on a missing component. Quietly averaging over whichever components happen to be
    present is the silent-intersection failure the ledger bans on every decision path, and a floor
    computed over five of six components is a different number wearing the same name."""
    want = {c for members in GROUPS.values() for c in members}
    missing = sorted(want - set(per_q))
    if missing:
        raise AssertionError(f"noise floor: components {missing} absent. The registered group "
                             f"vector is defined over {sorted(want)} (LEDGER 8) and may not be "
                             f"computed over a subset.")
    means = {c: float(np.mean(list(v.values()))) for c, v in per_q.items()}
    gm = {}
    for g, members in GROUPS.items():
        vals = [means[c] for c in members if c in means]
        if vals:
            gm[g] = float(np.mean(vals))
    return {
        "component_means": means,
        "group_means": gm,
        "group_vector_median": float(np.median(list(gm.values()))) if gm else None,
        "worst_group": float(min(gm.values())) if gm else None,
        "out_of_domain_macro": gm.get("out-of-domain"),
        "all_component_macro": float(np.mean(list(means.values()))) if means else None,
    }


BLEG_ARMS = ("m8nf-seed0", "m8nfb-seed1-a", "m8nfb-seed2-a")


def measure(dump_path, arm_ids=None):
    """Read a compare_full per-query dump and compute the floor on every endpoint.

    `arm_ids` names the null explicitly. The default (None) discovers the A-LEG floor's arms by
    their `m8nf-seed` prefix -- arms that share one Phase-B checkpoint and vary only the Phase-A
    seed. Passing `BLEG_ARMS` instead measures the B-LEG floor: three FULL chains (B then A)
    varying only the seed, which is the shape R-PHASE and every pool-or-init change actually has.
    The A-leg floor does not bound those, because it holds constant the very leg they perturb.
    """
    import gzip
    raw = json.loads(gzip.open(dump_path).read() if str(dump_path).endswith(".gz")
                     else Path(dump_path).read_text())
    pq = raw["per_query"] if "per_query" in raw else raw
    # compare_full keys its dump `<run_id>[:<pool_mode>]|<precision>`. BOTH precisions are kept:
    # int8 is the release format and the C2 identity, but B10 registers "raw CI > 0 in BOTH
    # precisions", so a bar reads fp16 too and an fp16 floor must exist for it (2026-08-29
    # review finding -- the first version filtered fp16 out by construction).
    # The key keeps the POOL MODE. `cfg.pool_mode` is None for these arms -- they train and
    # self-evaluate under `mean` -- while the M7 release is served under `sqrt`, adopted after the
    # fact by adopt_pool_mode.py. A floor measured under a rule the artifact is not served under
    # is a floor for a different function, and stripping ":mode" from the key would silently
    # collapse the two variants of the same arm onto each other. So the floor is reported per
    # (precision, pool_mode) and a bar reads the one its candidate is actually served under.
    arms = {"int8": {}, "fp16": {}}
    for key, comps in pq.items():
        parts = key.split("|")
        if len(parts) < 2 or parts[-1] not in arms:
            continue
        arms[parts[-1]][parts[0]] = _group_vector(comps)
    if not arms["int8"]:
        raise SystemExit(f"no int8 arms found in {dump_path}; keys look like "
                         f"{sorted(pq)[:4]}")

    endpoints = ("group_vector_median", "worst_group", "out_of_domain_macro",
                 "all_component_macro")
    floor, pairs, bars, sensitivity, seeds_seen, steps_seen = {}, {}, {}, {}, {}, {}
    for prec, a in arms.items():
        if not a:
            continue
        modes = sorted({(r.split(":", 1) + ["mean"])[1] for r in a})
        for mode in modes:
            sel = {r: v for r, v in a.items() if (r.split(":", 1) + ["mean"])[1] == mode}
            if arm_ids is None:
                seed_arms = {r: v for r, v in sel.items() if r.startswith("m8nf-seed")}
            else:
                want = set(arm_ids)
                seed_arms = {r: v for r, v in sel.items() if r.split(":", 1)[0] in want}
                # a floor from a PARTIAL arm set is a smaller floor, so it silently loosens every
                # bar that reads it. Refuse rather than under-report.
                if seed_arms and len(seed_arms) != len(want):
                    raise SystemExit(
                        f"{prec}.{mode}: expected arms {sorted(want)} but the dump has "
                        f"{sorted(r.split(':', 1)[0] for r in seed_arms)}. A floor computed from a "
                        f"partial arm set understates the floor and loosens every bar reading it.")
            step_arms = {r: v for r, v in sel.items() if r.startswith("m8nf-steps")}
            if not seed_arms:
                continue
            seeds_seen[f"{prec}.{mode}"] = sorted(seed_arms)
            steps_seen[f"{prec}.{mode}"] = sorted(step_arms)
            ref = next((v for r, v in seed_arms.items()
                        if r.split(":", 1)[0] == arm_id("seed", SEEDS[0])), None)
            for e in endpoints:
                key = f"{prec}.{mode}.{e}"
                ds = []
                for x, y in itertools.combinations(sorted(seed_arms), 2):
                    va, vb = seed_arms[x][e], seed_arms[y][e]
                    if va is not None and vb is not None:
                        ds.append({"pair": [x, y], "abs_delta": abs(va - vb)})
                pairs[key] = ds
                floor[key] = max((d["abs_delta"] for d in ds), default=None)
                bars[key] = None if floor[key] is None else max(0.0040, 2 * floor[key])
                sensitivity[key] = [{"arm": r, "delta_vs_seed0": v[e] - ref[e]}
                                    for r, v in sorted(step_arms.items())
                                    if ref and v[e] is not None and ref[e] is not None]
    return {
        "dump": str(dump_path),
        "seed_arms_by_precision": seeds_seen, "step_arms_by_precision": steps_seen,
        "_dump_path_note": ("the dump lands under a results/m7_devperquery_*.json.gz name because "
                            "compare_full.py is frozen M7 code and M8 does not edit m7src (G3)"),
        "arms": arms,
        "pairwise": pairs, "floor": floor, "bars": bars,
        "recipe_sensitivity_steps": sensitivity,
        "bar_formula": "bar(endpoint) = max(0.0040, 2 x floor(endpoint)); "
                       "floor = max of the pairwise |delta| over the 3 seed arms",
        "precision": "BOTH int8 (the release format and the C2 identity) and fp16; keys are "
                     "'<precision>.<endpoint>'",
        "endpoints_NOT_covered": {
            "fused": "B3, B13, R1-ASSEMBLY, D-SYNTH and D-FINEWEB register dense AND FUSED "
                     "endpoints. No fused floor is measured here, so their bars are NOT yet "
                     "computable as registered and remain refused. Measuring it needs a fused "
                     "read of the same three seed arms -- the arms exist, only the scoring pass "
                     "widens. LEDGER 4.4 gap list.",
            "B-leg-varying arms": "this floor holds the B checkpoint fixed. R-PHASE restructures "
                                  "the B->A phases, and any pool or init change flowing through "
                                  "the B leg has a larger floor. No bar may read such an arm "
                                  "until a B-leg null pair is measured. LEDGER 4.4 gap list.",
        },
        "frame": {
            "teacher": "stella_en_400M_v5 (incumbent)",
            "data_mix": "the M7 mix, from the M7 candidate's own B checkpoint",
            "legs_varied": "A leg only; the B checkpoint is held fixed",
            "disclosure": "an arm that differs in its B leg has a LARGER floor, and no bar may "
                          "read such an arm until that floor is measured too. LEDGER 6 step 5 "
                          "already voids this floor on a teacher swap; the same applies to a "
                          "material data-mix change.",
        },
        "why_steps_are_not_the_floor": (
            "a +/-10% step change is a TREATMENT, not a null: calling its magnitude 'noise' would "
            "license a bar that a real effect could clear. It is reported here as recipe "
            "sensitivity, never as the floor (LEDGER 4.7)."),
    }


def _twoway(x):
    """Two-way layout without replication on a 3x3 of one endpoint. Returns the variance
    components and what they imply for a fresh null difference between two independent chains.

    x[i][j] is the endpoint for B-checkpoint seed i, A seed j. With one observation per cell the
    residual mean square IS the interaction, so sigma_e here means 'A-given-B variation plus any
    B x A interaction' -- they are not separable at this design and the artifact says so rather
    than pretending otherwise.
    """
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    g, rm, cm = x.mean(), x.mean(1), x.mean(0)
    ss_b = n * float(((rm - g) ** 2).sum())
    ss_a = n * float(((cm - g) ** 2).sum())
    resid = x - rm[:, None] - cm[None, :] + g
    ss_e = float((resid ** 2).sum())
    df_b = df_a = n - 1
    df_e = (n - 1) ** 2
    ms_b, ms_a, ms_e = ss_b / df_b, ss_a / df_a, ss_e / df_e
    # random-effects moment estimators; negative estimates are clipped to 0, which biases the
    # chain SD DOWNWARD, so the clip is disclosed beside the number rather than hidden in it.
    v_b, v_a = max(0.0, (ms_b - ms_e) / n), max(0.0, (ms_a - ms_e) / n)
    v_chain = v_b + v_a + ms_e
    sd_diff = float(np.sqrt(2 * v_chain))
    from math import erfc, sqrt
    p_exceed = float(erfc(0.0040 / (sd_diff * sqrt(2)))) if sd_diff > 0 else 0.0
    diag = [float(x[i, i]) for i in range(n)]
    return {
        "cells": x.tolist(),
        "row_means_by_B_seed": rm.tolist(), "col_means_by_A_seed": cm.tolist(),
        "ms": {"B": ms_b, "A_given_B_plus_interaction": ms_e, "A": ms_a},
        "df": {"B": df_b, "A": df_a, "residual": df_e},
        "sd": {"B": float(np.sqrt(v_b)), "A": float(np.sqrt(v_a)),
               "residual": float(np.sqrt(ms_e)), "chain": float(np.sqrt(v_chain))},
        "clipped_to_zero": {"B": (ms_b - ms_e) < 0, "A": (ms_a - ms_e) < 0},
        "sd_of_fresh_null_difference": sd_diff,
        "p_null_difference_exceeds_0.0040": p_exceed,
        "range_all_9_cells": float(np.ptp(x)),
        "range_on_the_diagonal": float(np.ptp(diag)),
        "range_within_rows": [float(np.ptp(r)) for r in x],
        "diagonal_understates_by": (float(np.ptp(x) / np.ptp(diag))
                                    if np.ptp(diag) > 0 else None),
    }


def measure_crossed(dump_path, prec="int8", mode="sqrt"):
    """The crossed floor: a 3x3 of (B checkpoint seed) x (A seed), read at one precision/mode.

    Reads at int8/sqrt only -- the release format and R0's adopted pooling rule, which is what
    every frozen bar actually reads. Widening to the other three variants would quadruple the
    scoring cost for numbers no bar consults (the same argument b3_score.sh makes).
    """
    import gzip
    raw = json.loads(gzip.open(dump_path).read() if str(dump_path).endswith(".gz")
                     else Path(dump_path).read_text())
    pq = raw["per_query"] if "per_query" in raw else raw
    want = crossed_cells()
    got = {}
    for key, comps in pq.items():
        run, _, p = key.rpartition("|")
        rid, _, m = run.partition(":")
        if p == prec and (m or "mean") == mode:
            got[rid] = _group_vector(comps)
    missing = sorted({r for r in want.values()} - set(got))
    if missing:
        raise SystemExit(f"{prec}/{mode}: the 3x3 is missing {missing}. A crossed floor from a "
                         f"partial grid is not a crossed floor -- the whole point is that the "
                         f"aliased diagonal understates, and dropping cells re-introduces that.")
    endpoints = ("group_vector_median", "worst_group", "out_of_domain_macro",
                 "all_component_macro")
    out = {}
    for e in endpoints:
        grid = [[got[want[(b, a)]][e] for a in SEEDS] for b in SEEDS]
        if any(v is None for row in grid for v in row):
            continue
        out[f"{prec}.{mode}.{e}"] = _twoway(grid)
    return {
        "dump": str(dump_path),
        "design": "3x3, (B checkpoint seed) x (A seed), one observation per cell",
        "cells": {f"b{b}a{a}": want[(b, a)] for b in SEEDS for a in SEEDS},
        "cells_reused": {f"b{b}a{a}": EXISTING_CELLS[(b, a)] for (b, a) in EXISTING_CELLS},
        "read_at": f"{prec} / {mode}",
        "components": out,
        "what_this_answers": (
            "(1) is a B-leg-varying arm's floor actually larger than an A-leg arm's? `sd.B` "
            "against `sd.A` answers it, and no measurement supported that assertion before. "
            "(2) `sd_of_fresh_null_difference` is the actionable number: the SD of the difference "
            "between two independent full chains, which is the shape a B-leg-varying probe arm "
            "has. (3) whether the aliased diagonal is biased at all -- the residual term is where "
            "a leg-to-leg interaction at a shared seed would show up."),
        "withdrawn_claim": (
            "the B-leg artifact says its aliasing makes that floor an UNDER-estimate, 'the "
            "anti-conservative direction'. It does not follow: a diagonal cell is B_s + A_s + e "
            "and, for independent leg effects, has exactly the chain variance. Simulated at "
            "200,000 replicates the aliased diagonal and an independent chain match to four "
            "digits in SD (0.003202 vs 0.003202) and in E[range] (0.005425 vs 0.005416). The "
            "diagonal is unbiased-but-noisy. Only a NEGATIVE correlation between the two legs at "
            "a shared seed would bias it down, and no mechanism for that was ever named."),
        "estimator_bias": (
            "the variance components are moment estimators with negative values clipped to zero, "
            "and sd = sqrt(var-hat) is a concave function of it, so both push the reported chain "
            "SD DOWN. Simulated at this exact design: truth 0.00320 -> mean estimate 0.00295, "
            "about 8% low. Read the chain SD as a floor on the floor, not as a tight estimate."),
        "what_this_does_NOT_answer": (
            "the pool is still held FIXED -- `pseudoq.build_decontaminated` draws with a seed "
            "independent of the training seed, so all three B legs distil on the identical text "
            "set. This bounds seed variability in a B-leg-varying arm; it does not bound a "
            "pool-varying lever. It is also still the incumbent teacher frame and the M7 data "
            "mix, and LEDGER 6 step 5 voids it on a teacher swap."),
        "adopts": ("nothing. The registered bar formula is unchanged. This is reported as the "
                   "error rate the standing 0.0040 convention actually carries for a "
                   "B-leg-varying arm, which is a disclosure, not a new rule."),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["plan", "train", "measure", "bleg", "crossed",
                                     "measure-crossed"])
    ap.add_argument("--dump", default=None, help="compare_full per-query dump for `measure`")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--bleg-arms", action="store_true",
                    help="measure the B-LEG floor from the three full chains, not the A-leg arms")
    a = ap.parse_args()

    if a.step == "plan":
        cfg, arms = plan()
        print(json.dumps({"base_run": BASE_RUN, "base_cfg": cfg,
                          "arms": {r: {k: v for k, v in x.items() if k.startswith("_")
                                       or k in ("seed", "steps_a")}
                                   for r, x in arms.items()}}, indent=2, default=str))
    elif a.step == "bleg":
        arms = bleg_plan()
        print(json.dumps({r: {k: v for k, v in x.items() if k.startswith("_")
                              or k in ("seed", "steps_a", "steps_b", "init")}
                          for r, x in arms.items()}, indent=2, default=str))
        train(smoke=a.smoke, plan_fn=bleg_plan)
    elif a.step == "crossed":
        arms = crossed_plan()
        print(json.dumps({r: {k: v for k, v in x.items() if k.startswith("_")
                              or k in ("seed", "steps_a", "init")}
                          for r, x in arms.items()}, indent=2, default=str))
        train(smoke=a.smoke, plan_fn=crossed_plan)
    elif a.step == "measure-crossed":
        if not a.dump:
            raise SystemExit("--dump is required: point at the compare_full per-query dump")
        out = measure_crossed(a.dump)
        out["leg"] = "crossed"
        dest = OUT.parent / "m8_noise_floor_crossed.json"
        probe_guard.write_result(dest, out, "NF", strict_commit=not a.smoke)
        print(json.dumps({k: {kk: v[kk] for kk in
                              ("sd", "sd_of_fresh_null_difference",
                               "p_null_difference_exceeds_0.0040", "range_all_9_cells",
                               "range_on_the_diagonal")}
                          for k, v in out["components"].items()}, indent=2))
        print(f"\nwrote {dest}", file=sys.stderr)
    elif a.step == "train":
        train(smoke=a.smoke)
    else:
        if not a.dump:
            raise SystemExit("--dump is required: point at the compare_full per-query dump")
        # The B-leg floor gets its OWN artifact. Writing it to `OUT` would overwrite the A-leg
        # floor that B3's frozen bar and every other registered bar already cite -- a silent
        # redefinition of a number that has already been read.
        out = measure(a.dump, arm_ids=BLEG_ARMS if a.bleg_arms else None)
        out["leg"] = "B" if a.bleg_arms else "A"
        out["arms_measured"] = list(BLEG_ARMS) if a.bleg_arms else None
        if a.bleg_arms:
            # `frame` is written by the A-leg path and describes THAT null. Left alone it made the
            # B-leg artifact say "A leg only; the B checkpoint is held fixed" and assert that a
            # B-differing arm "has a LARGER floor" -- a file contradicting its own contents, and
            # asserting the very claim this measurement does not support.
            out["frame"] = {
                "teacher": "stella_en_400M_v5 (incumbent)",
                "data_mix": "the M7 mix",
                "legs_varied": "BOTH: three full B->A chains, one seed driving both legs",
                "aliasing": "because ONE seed drives both legs, B-seed and A-seed effects cannot "
                            "be separated and may partially cancel, which would make this floor an "
                            "UNDER-estimate. Crossing the three B checkpoints against several A "
                            "seeds is the fix (LEDGER 15, 2026-08-29).",
                "pool": "held FIXED -- pseudoq draws with a seed independent of the training seed, "
                        "so all three chains distil on the identical text set. This is a "
                        "conditional seed null and does NOT bound pool-varying levers.",
                "disclosure": "K = 3 makes this statistic the sample RANGE (CV 0.525). It is a "
                              "pre-registered decision CONVENTION and a magnitude yardstick, not a "
                              "statistical bound; do not read it as showing the B leg adds no "
                              "variability.",
            }
        dest = (OUT.parent / "m8_noise_floor_bleg.json") if a.bleg_arms else OUT
        probe_guard.write_result(dest, out, "NF", strict_commit=not a.smoke)
        print(json.dumps({"leg": out["leg"], "floor": out["floor"], "bars": out["bars"]}, indent=2))
        print(f"\nwrote {dest}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
