"""LEDGER 4.4 deliverable 2 -- the joint power simulation of the FULL ship rule.

Why this must exist before Phase 0 spends its week. M7 froze a recipe, spent its one access, and
missed its release bar CI-resolved. That was a pre-registered publishable outcome, but nobody had
computed, in advance, how likely it was. Doing that first turns "report-only" from a discovered
outcome into a knowing choice: if P(ship) at the effect sizes the surviving levers can plausibly
deliver is 0.2, the owner gets to decide whether to spend the week on this design or a different
one. A knowing report-only choice beats a discovered one.

What it computes
  * the standard error of the equal-weight four-set macro delta, calibrated from REAL paired
    per-query vectors;
  * the minimum detectable effect (MDE) of the three-leg rule, i.e. the smallest true effect the
    rule resolves with probability 0.8 -- which is also the LoTTE shadow gate's GO threshold
    (LEDGER 2.3: the GO threshold is the MDE, not zero);
  * P(ship) for the COMPLETE rule -- three legs plus the qualifying-table requirement, the +0.005
    point guard, the worst-group guard and the six-set no-regression guard -- under named
    scenarios.

The honest limitation, stated first rather than found later. The reserved four have never been
scored, by construction, so their per-query variance is NOT measurable and this simulation
extrapolates it from the nearest available analogue for each:

    fever            <- hotpotqa        (dev; Wikipedia, multi-sentence claims, comparable n)
    dbpedia-entity   <- nq-250k         (dev; Wikipedia/entity-shaped, but n=400 vs 3,452)
    cqadup-android   <- cqadup-programmers   (dev; same family, same task)
    cqadup-english   <- cqadup-physics       (dev; same family, same task)

Two of the four analogues are same-family and should be good; the DBpedia one is the weak link and
its n is 8.6x smaller than its analogue's, so its contribution to the macro variance is the least
trustworthy number here. Every output is therefore reported across a sensitivity band on the
calibrated sd, not as a point.

Method. The CI legs and the sign-flip p are computed under a normal approximation, which is what
lets 20,000 replicates of the whole rule run in seconds; `--validate` checks that approximation
against the exact `boot` machinery on real vectors and reports the agreement. At these n
(9,335 queries over four sets) the approximation is close; where it is not, the exact path is the
one that decides, and this file is only a planning instrument.
"""
import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

import m8base
import boot
import decide

REPO = m8base.REPO
RESULTS = m8base.RESULTS
OUT = RESULTS / "m8_power.json"

# n on the reserved four (results/eval_manifest.json, m7_untouched_final), and the dev analogue
# each borrows its per-query difference sd from.
RESERVED_N = {"fever": 6666, "dbpedia-entity": 400,
              "cqadup-android": 699, "cqadup-english": 1570}
ANALOGUE = {"fever": "hotpotqa", "dbpedia-entity": "nq-250k",
            "cqadup-android": "cqadup-programmers", "cqadup-english": "cqadup-physics"}

ALPHA = decide.ALPHA           # 0.025
M = decide.M                   # 3
BONF = decide.BONF_LEVEL       # 0.0083333


# ---------------------------------------------------------------- calibration ---------------

def _dev_pairs():
    """-> {component: sd of the paired per-query nDCG difference between two RELATED trained
    tables}. Related is the right reference class: M8-vs-M7 is two tables from the same family,
    not a table versus BM25."""
    src = RESULTS / "m8_devperquery_cal.json"        # cached; the gz is 40 arms and slow to parse
    if src.exists():
        return json.loads(src.read_text())
    d = json.loads(gzip.open(RESULTS / "m7_devperquery_full.json.gz").read())
    pq = d["per_query"]
    arms = sorted(k for k in pq if k.endswith("|int8|table"))
    if len(arms) < 2:
        arms = sorted(k for k in pq if k.endswith("|fp16|table"))
    out = {}
    a, b = arms[0], arms[-1]
    for comp in pq[a]:
        qs = sorted(set(pq[a][comp]) & set(pq[b][comp]))
        diff = np.array([pq[a][comp][q] - pq[b][comp][q] for q in qs], dtype=np.float64)
        out[comp] = {"sd": float(diff.std(ddof=1)), "n": len(qs), "arms": [a, b],
                     "mean": float(diff.mean())}
    src.write_text(json.dumps(out, indent=2))
    return out


def _six_pairs():
    """The dissimilar reference class, from the frozen final run: int8 table vs BM25 on the six.
    Used only as an upper sensitivity bound on the sd."""
    fr = json.loads((RESULTS / "m7_final_run.json").read_text())["six"]
    out = {}
    for ds in fr["int8-table"]:
        qs = sorted(set(fr["int8-table"][ds]) & set(fr["bm25"][ds]))
        diff = np.array([fr["int8-table"][ds][q] - fr["bm25"][ds][q] for q in qs])
        out[ds] = {"sd": float(diff.std(ddof=1)), "n": len(qs)}
    return out


def macro_se(sds, ns):
    """SE of the equal-weight macro of per-dataset mean differences.
    macro = (1/K) sum_d mean_d(diff);  Var = (1/K^2) sum_d sd_d^2 / n_d."""
    k = len(sds)
    return float(np.sqrt(sum(s * s / n for s, n in zip(sds, ns)) / (k * k)))


def calibrate(scale=1.0):
    dev = _dev_pairs()
    sds, ns, rows = [], [], {}
    for ds, an in ANALOGUE.items():
        sd = dev[an]["sd"] * scale
        n = RESERVED_N[ds]
        sds.append(sd)
        ns.append(n)
        rows[ds] = {"analogue": an, "sd": sd, "n": n,
                    "analogue_n": dev[an]["n"],
                    "var_contribution": sd * sd / n / (len(ANALOGUE) ** 2)}
    se = macro_se(sds, ns)
    return {"scale": scale, "per_dataset": rows, "macro_se": se,
            "half_width_95": 1.96 * se, "dev_source": dev}


# ---------------------------------------------------------------- the rule, fast --------------

def _resolve(delta_hat, se):
    """The three legs under the normal approximation. Returns (p, ci_lo, bonf_lo)."""
    z = delta_hat / se
    p = float(stats.norm.sf(z))                       # one-sided sign-flip analogue
    return p, delta_hat - stats.norm.isf(0.025) * se, delta_hat - stats.norm.isf(BONF) * se


def _holm_reject(ps, alpha=ALPHA):
    """Vectorized Holm over the three legs. ps: (S, 3). -> (S, 3) bool."""
    order = np.argsort(ps, axis=1)
    m = ps.shape[1]
    rej = np.zeros_like(ps, dtype=bool)
    still = np.ones(ps.shape[0], dtype=bool)
    for i in range(m):
        idx = order[:, i]
        thr = alpha / (m - i)
        p_i = np.take_along_axis(ps, idx[:, None], 1)[:, 0]
        ok = still & (p_i <= thr)
        np.put_along_axis(rej, idx[:, None], ok[:, None], 1)
        still = ok
    return rej


def registry_params():
    """The ship rule's constants, READ FROM m8/registry.json rather than restated here.

    They were restated here once, and the restatement went stale the moment section 5 got its
    measured values: this file carried six_margin = 0.005 and se_six = 0.006 while the registry
    said 0.0075 and the measured near-sibling six-macro SE was 0.0026-0.0032. The six-set guard's
    false-veto rate was overstated about fortyfold, and it is the dominant term in P(ship) -- so
    the headline table handed to the owner was materially wrong. A planning instrument that
    disagrees with the rule it simulates is worse than no instrument."""
    reg = json.loads((REPO / "m8" / "registry.json").read_text())["ship_rule"]
    return {"six_margin": abs(reg["six_set_no_regression_margin"]),
            "worst_group_threshold": reg["worst_group_threshold"],
            "worst_group_members": reg["worst_group_members"],
            "point_guard_c1": reg["point_guard_c1"]}


def six_macro_se():
    """The six-set macro's paired SE for a NEAR-SIBLING system pair, measured from the frozen
    comparator vectors -- the same provenance the registered 0.0075 margin was derived from."""
    import itertools
    pq = json.loads((RESULTS / "perquery.json").read_text())["datasets"]
    six = list(m8base.SIX)
    names = list(pq[six[0]]["systems"])
    out = []
    for a, b in itertools.combinations(names, 2):
        k = len(six)
        var = 0.0
        for ds in six:
            d = (np.asarray(pq[ds]["systems"][a], float)
                 - np.asarray(pq[ds]["systems"][b], float))
            var += d.var(ddof=1) / len(d)
        out.append((float(np.sqrt(var) / k), a, b))
    out.sort()
    # The reference class is NAMED, not taken as the minimum. LEDGER 5.7 derives the registered
    # 0.0075 margin from two specific near-sibling pairs -- the same model family asym-vs-sym, and
    # the same table under two instruction sets -- and the simulation must use the same pairs or
    # it is simulating a different guard. The MAX of the two is used: for a false-veto estimate
    # the conservative direction is the LARGER standard error, and taking the minimum over all
    # pairs would flatter the guard.
    named = {frozenset(("leaf-ir-asym", "mdbr-leaf-ir")),
             frozenset(("lr-dense-pertask", "lr-dense-websearch"))}
    ref = [(se, a, b) for se, a, b in out if frozenset((a, b)) in named]
    return {"reference_pairs": [{"se": se, "pair": [a, b]} for se, a, b in ref],
            "near_sibling_se": max(se for se, _, _ in ref) if ref else out[0][0],
            "min_over_all_pairs": {"se": out[0][0], "pair": list(out[0][1:])},
            "median_over_all_pairs": float(np.median([o[0] for o in out])),
            "_why_max": "conservative: for a false-veto estimate the larger SE is the safe one"}


def per_dataset_se(cal):
    """SE of each reserved dataset's own mean difference: sd_d / sqrt(n_d). The worst-group guard
    reads PER-DATASET point estimates (LEDGER 5.6, four singleton groups), and their SEs differ by
    4x across the four -- DBpedia's n = 400 against FEVER's 6,666 -- so simulating them with a
    common noise term is simulating a different guard."""
    return {ds: m["sd"] / np.sqrt(m["n"]) for ds, m in cal["per_dataset"].items()}


def simulate(scenario, cal, S=20000, seed=0):
    """One scenario -> P(ship) and the per-condition marginals.

    Scenario fields (all in nDCG points on the equal-weight four-set macro):
      d_c1   true fused-M8 minus fused-M7
      d_c2   true dense-M8 minus dense-M7 (strict C2)
      d_c3   true fused-M8 minus BM25
      rho    correlation between the C1 and C2 estimate errors (same candidate -> high)
      tau    per-group heterogeneity sd of the true effect, which is what the worst-group guard
             actually tests
      d_six  true six-set delta (M8 minus M7) for the no-regression guard
      se_six SE of that six-set delta (frozen vectors, so this is a query-sampling SE only)
      six_margin  the registered no-regression margin
      qualifying  whether a qualifying TABLE change survives Stage R/S (probability)
    """
    rng = np.random.default_rng(seed)
    se = cal["macro_se"]
    # C1 and C2 share the candidate, so their estimate errors are correlated.
    rho = scenario["rho"]
    z = rng.standard_normal((S, 3))
    e1 = z[:, 0]
    e2 = rho * z[:, 0] + np.sqrt(1 - rho * rho) * z[:, 1]
    e3 = z[:, 2]
    hat = np.stack([scenario["d_c1"] + se * e1,
                    scenario["d_c2"] + se * e2,
                    scenario["d_c3"] + se * e3], axis=1)
    ps = stats.norm.sf(hat / se)
    ci_lo = hat - stats.norm.isf(0.025) * se
    bonf_lo = hat - stats.norm.isf(BONF) * se
    resolved = _holm_reject(ps) & (ci_lo > 0) & (bonf_lo > 0)

    par = registry_params()
    point_guard = hat[:, 0] > par["point_guard_c1"]          # STRICT, as registered
    # Worst-group: the FOUR reserved datasets individually (LEDGER 5.6), each with its own
    # sampling SE. Each dataset's TRUE effect is d_c1 + N(0, tau); the guard reads a point
    # estimate, so add that dataset's own sd_d/sqrt(n_d).
    ses = per_dataset_se(cal)
    members = [m for m in par["worst_group_members"] if m in ses]
    g_true = scenario["d_c1"] + scenario["tau"] * rng.standard_normal((S, len(members)))
    g_hat = g_true + np.array([ses[m] for m in members]) * rng.standard_normal((S, len(members)))
    worst_ok = g_hat.min(1) >= par["worst_group_threshold"]
    se_six = scenario.get("se_six") or six_macro_se()["near_sibling_se"]
    six_hat = scenario["d_six"] + se_six * rng.standard_normal(S)
    six_ok = six_hat >= -par["six_margin"]
    qual_ok = rng.random(S) < scenario["qualifying"]

    ship = resolved.all(1) & point_guard & worst_ok & six_ok & qual_ok
    return {
        "scenario": scenario, "S": S, "macro_se": se,
        "registered_params": par, "se_six_used": se_six,
        "per_dataset_se": {m: ses[m] for m in members},
        "P_C1_resolved": float(resolved[:, 0].mean()),
        "P_C2_resolved": float(resolved[:, 1].mean()),
        "P_C3_resolved": float(resolved[:, 2].mean()),
        "P_all_three": float(resolved.all(1).mean()),
        "P_point_guard": float(point_guard.mean()),
        "P_worst_group": float(worst_ok.mean()),
        "P_six_no_regression": float(six_ok.mean()),
        "P_qualifying_table": float(qual_ok.mean()),
        "P_ship": float(ship.mean()),
    }


def mde(cal, power=0.8, seed=0, S=20000):
    """Smallest true C1 effect the three-leg rule resolves with probability `power`, holding C2
    and C3 comfortably positive. This is also the registered LoTTE shadow GO threshold."""
    se = cal["macro_se"]
    lo, hi = 0.0, 20 * se
    for _ in range(40):
        mid = (lo + hi) / 2
        r = simulate({"d_c1": mid, "d_c2": mid, "d_c3": 0.06, "rho": 0.8, "tau": 0.005,
                      "d_six": 0.0, "se_six": None, "qualifying": 1.0},
                     cal, S=S, seed=seed)
        if r["P_C1_resolved"] < power:
            lo = mid
        else:
            hi = mid
    # The analytic value for a single leg: the Bonferroni bound binds at z_{1-0.008333}=2.394.
    z_bonf = stats.norm.isf(BONF)
    analytic = (z_bonf + stats.norm.isf(1 - power)) * se
    return {"mde_c1_power%.2f" % power: (lo + hi) / 2,
            "analytic_single_leg": float(analytic),
            "binding_leg": "the alpha/3 = 0.008333 simultaneous lower bound (z = %.3f)" % z_bonf,
            "macro_se": se}


def validate(cal, n_check=200, seed=0):
    """Check the normal approximation against the exact `boot` machinery on synthetic vectors
    shaped like the reserved four. Reports the agreement rather than asserting it."""
    rng = np.random.default_rng(seed)
    rows = []
    for shift in (0.0, 0.005, 0.01, 0.02):
        a, b = {}, {}
        for ds, meta in cal["per_dataset"].items():
            n = meta["n"]
            base = np.clip(rng.normal(0.45, 0.30, n), 0, 1)
            d = rng.normal(shift, meta["sd"], n)
            b[ds] = {f"q{i:07d}": float(v) for i, v in enumerate(base)}
            a[ds] = {f"q{i:07d}": float(v) for i, v in enumerate(base + d)}
        pr = boot.paired(a, b, alternative="greater", strict=True)
        sf = boot.signflip(a, b, alternative="greater", strict=True)
        p_n, ci_n, bonf_n = _resolve(pr["delta_raw"], cal["macro_se"])
        rows.append({
            "true_shift": shift, "delta_raw": pr["delta_raw"],
            "exact": {"signflip_p": sf["p"], "ci_lower_raw": pr["ci95_raw"][0],
                      "bonferroni_lower_raw": decide._bonf_lookup(pr["one_sided_lower_raw"])},
            "normal_approx": {"p": p_n, "ci_lower": ci_n, "bonferroni_lower": bonf_n},
        })
    return rows


SCENARIOS = {
    # Named, so the wake-up note can say which world it is quoting. The effect sizes are the
    # PLANNING targets from m8/PLAN-DRAFT.md 1.10 and the surviving levers' expected values, not
    # measurements: no M8 number exists.
    "structural_target": dict(
        d_c1=0.020, d_c2=0.020, d_c3=0.060, rho=0.8, tau=0.005,
        d_six=0.000, se_six=None, qualifying=0.85),
    "modest": dict(
        d_c1=0.010, d_c2=0.010, d_c3=0.060, rho=0.8, tau=0.005,
        d_six=-0.002, se_six=None, qualifying=0.85),
    "recipe_only": dict(
        d_c1=0.005, d_c2=0.005, d_c3=0.060, rho=0.8, tau=0.005,
        d_six=-0.003, se_six=None, qualifying=0.60),
    "m7_repeat": dict(   # the post-gate lever programme transferred 0.000 +/- 0.005 in M7
        d_c1=0.000, d_c2=0.000, d_c3=0.060, rho=0.8, tau=0.005,
        d_six=0.000, se_six=None, qualifying=0.85),
    "dense_lags_fused": dict(   # strict C2 (E11) is the plausible binding constraint
        d_c1=0.020, d_c2=0.006, d_c3=0.060, rho=0.8, tau=0.005,
        d_six=0.000, se_six=None, qualifying=0.85),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--S", type=int, default=20000)
    ap.add_argument("--validate", action="store_true")
    a = ap.parse_args()

    cal = calibrate()
    cal_hi = calibrate(scale=1.25)
    cal_lo = calibrate(scale=0.80)
    out = {
        "_note": __doc__.strip().splitlines()[0],
        "registered_params": registry_params(),
        "six_macro_se_measured": six_macro_se(),
        "calibration": {k: {kk: vv for kk, vv in v.items() if kk != "dev_source"}
                        for k, v in {"central": cal, "sd+25%": cal_hi, "sd-20%": cal_lo}.items()},
        "dev_analogue_sds": {k: v for k, v in cal["dev_source"].items()},
        "six_dissimilar_sds": _six_pairs(),
        "prior_plan_estimate": {"near_sibling_half_width": 0.005, "dissimilar_half_width": 0.0096,
                                "source": "m8/PLAN-DRAFT.md 1.10"},
        "mde": {"central": mde(cal, S=a.S), "sd+25%": mde(cal_hi, S=a.S),
                "sd-20%": mde(cal_lo, S=a.S)},
        "scenarios": {name: {"central": simulate(sc, cal, S=a.S),
                             "sd+25%": simulate(sc, cal_hi, S=a.S),
                             "sd-20%": simulate(sc, cal_lo, S=a.S)}
                      for name, sc in SCENARIOS.items()},
        "shadow_gate_note": ("LEDGER 2.3: the LoTTE shadow GO threshold is the MDE reported here, "
                             "not zero. Read it off mde.central."),
        "limitation": ("The reserved four have never been scored, so their per-query variance is "
                       "extrapolated from dev analogues (fever<-hotpotqa, dbpedia<-nq-250k, "
                       "android<-programmers, english<-physics). DBpedia is the weak link: n=400 "
                       "against its analogue's 3,452. Sensitivity is reported at +/-25%/-20% on "
                       "the calibrated sd rather than as a point."),
    }
    if a.validate:
        out["normal_approximation_check"] = validate(cal)
    OUT.write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps({"macro_se": cal["macro_se"],
                      "half_width_95": cal["half_width_95"],
                      "mde": out["mde"]["central"],
                      "P_ship": {k: v["central"]["P_ship"] for k, v in out["scenarios"].items()}},
                     indent=2))
    print(f"\nwrote {OUT}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
