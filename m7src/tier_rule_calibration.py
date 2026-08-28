"""What the tier rule's family-wise error rate ACTUALLY is under the weak null.

Why this exists (Codex one-shot-path review 2026-08-28, MAJOR 5). The ledger claims the tier rule
bounds the family at alpha = 0.025. Two facts sit against that claim:

  * `boot.signflip` is exact under the SHARP null (per-query differences symmetric about 0), not
    under the weak null the report means (macro mean <= 0). `m7_signflip_weaknull.json` measures
    its actual rate at the Bonferroni level 0.008333 as 0.013 and 0.008.
  * A union bound over three such marginal procedures gives 3 x 0.013 = 0.039, not 0.025.

The union bound is what is available WITHOUT measurement, and it is loose in two ways that both
matter here: the rule is a CONJUNCTION of three legs (its rate is at most the smallest leg's, and
the two bootstrap legs have never been measured under the weak null), and the three comparisons
share one system and one query sample, so their rejections are positively dependent and the union
bound over-counts. This script measures the thing itself.

CONSTRUCTION. The same weak null as `test_signflip_weaknull.py`, extended to the family: take a
candidate stand-in system A and the three real comparators, center each (comparison, dataset)
per-query difference vector to mean zero -- so all three weak nulls are true BY CONSTRUCTION while
skew, boundedness and per-dataset n are the real ones -- and resample queries within each dataset.
The SAME resampled query indices feed all three comparisons, which is what makes them dependent,
exactly as in the final run where one candidate faces three comparators on one query sample.

FIDELITY, stated rather than assumed. The rule is replayed as `final_run.py` implements it:
Holm(sign-flip) at family alpha=0.025 AND raw paired CI lower > 0 AND raw one-sided lower at
0.025/3 > 0. Two deviations, both for cost: R=2,000 sign-flip replicates instead of 100,000 (the
setting the committed weak-null suite already uses) and one shared randomization draw per dataset
reused across the three comparisons -- which is not an approximation at all, since `boot` seeds
every call with SEED=0 and the real run therefore reuses the same draws too.

This is a TYPE-I calibration of a procedure, not a result: it reads only the frozen comparator
per-query nDCG vectors already committed in results/perquery.json, never a qrel, a corpus or any
number produced by our own system. Same access class as the two committed sign-flip suites.

    M7_ENCODER=stella-400M-v5 PYTHONPATH=m7src .venv/bin/python m7src/tier_rule_calibration.py
"""
import json
import time

import numpy as np

import boot
from _paths import REPO

# Stand-ins for C1/C2/C3: one candidate system facing the three real comparators, the shape the
# final run has. Two candidates, because the answer should not depend on which stand-in is used --
# bge-small is a small symmetric transformer, lr-dense-websearch is a zero-query-compute lookup
# table like ours and is the closer architectural analogue.
CANDIDATES = ["bge-small-en-v1.5", "lr-dense-websearch"]
COMPARATORS = ["lr-dense-pertask", "bm25", "opensearch-doc-v3-gte"]
ALPHA = 0.025
# M7_TIER_SMOKE=1 shrinks every budget so the path can be exercised end to end in seconds; a
# smoke run writes to its own file so it can never be mistaken for the calibration.
import os
SMOKE = bool(os.environ.get("M7_TIER_SMOKE"))
S = 4 if SMOKE else 1000    # simulations; binomial SE at p=0.02 is 0.0044
R = 200 if SMOKE else 2_000     # sign-flip replicates per test (as in test_signflip_weaknull.py)
B = 500 if SMOKE else 10_000    # bootstrap replicates per test (as in boot.B, the real rule)
LEVELS = (2.5, 100 * ALPHA / len(COMPARATORS))     # 2.5% and the Bonferroni 0.8333%


def centered_diffs(blob, a_name, b_name):
    """{dataset: centered per-query difference array}, weak null true by construction."""
    A = boot.from_perquery_json(blob, a_name)
    B_ = boot.from_perquery_json(blob, b_name)
    pairs = boot._align(A, B_)
    return {ds: (da - db) - (da - db).mean() for ds, (da, db) in pairs.items()}


def legs(diffs, signs, bidx):
    """The three legs of the tier rule for ONE comparison, on pre-drawn randomization matrices.

    diffs/signs/bidx are keyed by dataset. Returns (p_signflip, p_studentized, lower_2.5,
    lower_bonferroni). Vectorized restatement of boot.signflip + boot.paired(alternative=
    "greater"); `selfcheck()` asserts it agrees with those functions rather than trusting it.
    """
    k = len(diffs)
    t_obs = sum(float(v.mean()) for v in diffs.values()) / k
    t = np.zeros(next(iter(signs.values())).shape[0])
    q_obs = 0.0                                     # sum_ds mean(d^2)/n_ds, for studentization
    q_rep = np.zeros_like(t)
    for ds, v in diffs.items():
        flipped = (signs[ds] * v).mean(1)
        t += flipped / k
        q_obs += float((v ** 2).mean()) / v.size
        # var(s.d) = mean(d^2) - mean(s.d)^2, so the studentized denominator costs one square
        q_rep += (float((v ** 2).mean()) - flipped ** 2) / v.size
    p = (1 + int((t >= t_obs).sum())) / (1 + len(t))
    v_obs = (q_obs - sum(float(v.mean()) ** 2 / v.size for v in diffs.values())) / k ** 2
    T_obs = t_obs / np.sqrt(max(v_obs, 1e-300))
    T_rep = t / np.sqrt(np.maximum(q_rep / k ** 2, 1e-300))
    p_stud = (1 + int((T_rep >= T_obs).sum())) / (1 + len(t))
    deltas = np.zeros(next(iter(bidx.values())).shape[0])
    for ds, v in diffs.items():
        deltas += v[bidx[ds]].mean(1) / k
    lo = {lv: float(np.percentile(deltas, lv)) for lv in LEVELS}
    return p, p_stud, lo[LEVELS[0]], lo[LEVELS[1]]


def selfcheck(blob):
    """The vectorized legs must reproduce boot.signflip and boot.paired, or this whole file is
    calibrating a different procedure than the one that ships."""
    d = centered_diffs(blob, CANDIDATES[0], COMPARATORS[0])
    # shift it off the null so the comparison is not all at the same trivial value
    d = {ds: v + 0.01 for ds, v in d.items()}
    rng = np.random.default_rng(boot.SEED)
    signs = {ds: (rng.integers(0, 2, size=(R, v.size)) * 2 - 1).astype(np.int8)
             for ds, v in d.items()}
    rng2 = np.random.default_rng(boot.SEED)
    bidx = {ds: rng2.integers(0, v.size, size=(B, v.size)) for ds, v in d.items()}
    p, _, lo25, lo_b = legs(d, signs, bidx)
    # zero-padded qids: `boot._align_ids` sorts qids as STRINGS, so "q10" < "q2" would permute
    # the arrays relative to `d` and the shared draw matrices would index a different order --
    # which is exactly how the first version of this self-check failed.
    A = {ds: {f"q{i:07d}": float(x) for i, x in enumerate(v)} for ds, v in d.items()}
    Z = {ds: {f"q{i:07d}": 0.0 for i in range(v.size)} for ds, v in d.items()}
    ref_p = boot.signflip(A, Z, R=R, seed=boot.SEED, alternative="greater")["p"]
    ref = boot.paired(A, Z, B=B, seed=boot.SEED, alternative="greater", strict=True)
    ok = {"signflip p": (p, ref_p, p == ref_p),
          "lower 2.5%": (lo25, ref["ci95_raw"][0], abs(lo25 - ref["ci95_raw"][0]) < 5e-5),
          f"lower {LEVELS[1]:.4f}%": (lo_b, ref["one_sided_lower_raw"][f"{LEVELS[1]:.4f}"],
                                      True)}
    # the Bonferroni level is keyed as "0.8333" in boot.paired; compare on the value
    ref_b = ref["one_sided_lower_raw"].get("0.8333")
    ok[f"lower {LEVELS[1]:.4f}%"] = (lo_b, ref_b, abs(lo_b - ref_b) < 5e-5)
    for name, (mine, theirs, agree) in ok.items():
        print(f"  {'ok  ' if agree else 'FAIL'} {name}: vectorized {mine!r} vs boot {theirs!r}")
    assert all(v[2] for v in ok.values()), "the vectorized legs do not reproduce boot.*"
    return True


def simulate(blob, candidate, seed=7):
    diffs = {b: centered_diffs(blob, candidate, b) for b in COMPARATORS}
    comps = sorted(next(iter(diffs.values())))
    n = {ds: diffs[COMPARATORS[0]][ds].size for ds in comps}
    rng = np.random.default_rng(seed)
    counts = {k: 0 for k in ("any_reject", "any_signflip_holm", "any_ci25", "any_bonf",
                             "any_reject_studentized")}
    per_comp = {b: {"reject": 0, "signflip_holm": 0, "ci25": 0, "bonf": 0,
                    "signflip_raw_0.025": 0, "signflip_raw_bonf": 0,
                    "studentized_raw_bonf": 0} for b in COMPARATORS}
    for _ in range(S):
        # one resample of queries, shared by all three comparisons (they face one query sample)
        idx = {ds: rng.integers(0, n[ds], n[ds]) for ds in comps}
        drawn = {b: {ds: diffs[b][ds][idx[ds]] for ds in comps} for b in COMPARATORS}
        # one randomization draw per dataset, reused across comparisons -- as in the real run,
        # where every boot.* call is seeded with SEED=0
        signs = {ds: (rng.integers(0, 2, size=(R, n[ds])) * 2 - 1).astype(np.int8) for ds in comps}
        bidx = {ds: rng.integers(0, n[ds], size=(B, n[ds])) for ds in comps}
        res = {b: legs(drawn[b], signs, bidx) for b in COMPARATORS}
        holm = boot.holm({b: res[b][0] for b in COMPARATORS}, alpha=ALPHA)
        holm_s = boot.holm({b: res[b][1] for b in COMPARATORS}, alpha=ALPHA)
        rej, rej_s = {}, {}
        for b in COMPARATORS:
            p, p_s, lo25, lob = res[b]
            r = bool(holm[b]["reject"] and lo25 > 0 and lob > 0)
            rej[b] = r
            rej_s[b] = bool(holm_s[b]["reject"] and lo25 > 0 and lob > 0)
            pc = per_comp[b]
            pc["reject"] += r
            pc["signflip_holm"] += bool(holm[b]["reject"])
            pc["ci25"] += bool(lo25 > 0)
            pc["bonf"] += bool(lob > 0)
            pc["signflip_raw_0.025"] += bool(p <= ALPHA)
            pc["signflip_raw_bonf"] += bool(p <= ALPHA / len(COMPARATORS))
            pc["studentized_raw_bonf"] += bool(p_s <= ALPHA / len(COMPARATORS))
        counts["any_reject"] += any(rej.values())
        counts["any_reject_studentized"] += any(rej_s.values())
        counts["any_signflip_holm"] += any(holm[b]["reject"] for b in COMPARATORS)
        counts["any_ci25"] += any(res[b][2] > 0 for b in COMPARATORS)
        counts["any_bonf"] += any(res[b][3] > 0 for b in COMPARATORS)
    return ({k: round(v / S, 4) for k, v in counts.items()},
            {b: {k: round(v / S, 4) for k, v in d.items()} for b, d in per_comp.items()})


def main():
    t0 = time.time()
    blob = json.loads((REPO / "results" / "perquery.json").read_text())
    print("self-check: the vectorized legs vs boot.signflip / boot.paired")
    selfcheck(blob)
    out = {}
    for cand in CANDIDATES:
        print(f"\nsimulating the family under the weak null, candidate stand-in {cand}", flush=True)
        fam, per = simulate(blob, cand)
        print(f"  family-wise rejection rate (the tier rule as written): {fam['any_reject']}")
        print(f"  per-leg family rates: Holm(sign-flip) {fam['any_signflip_holm']}, "
              f"CI>0 {fam['any_ci25']}, Bonferroni lower>0 {fam['any_bonf']}")
        print(f"  with a studentized sign-flip leg instead: {fam['any_reject_studentized']}")
        for b, d in per.items():
            print(f"    vs {b:26s} conjunction {d['reject']}  legs: holm {d['signflip_holm']} "
                  f"ci {d['ci25']} bonf {d['bonf']}  |  raw signflip@0.00833 "
                  f"{d['signflip_raw_bonf']} studentized@0.00833 {d['studentized_raw_bonf']}")
        out[cand] = {"family": fam, "per_comparison": per}
    se = round(float(np.sqrt(0.025 * 0.975 / S)), 4)
    res = {"S": S, "R": R, "B": B, "alpha": ALPHA,
           "bonferroni_level": ALPHA / len(COMPARATORS),
           "binomial_se_at_alpha": se,
           "candidates": CANDIDATES, "comparators": COMPARATORS,
           "_null": "each (comparison, dataset) per-query difference vector centered to mean zero; "
                    "all three weak nulls true by construction; queries resampled within dataset, "
                    "the SAME resample feeding all three comparisons",
           "_rule": "Holm(sign-flip) at family alpha AND raw paired CI lower > 0 AND raw one-sided "
                    "lower at alpha/3 > 0 -- final_run.py as written",
           "_deviations": "R=2,000 sign-flip replicates instead of the run's 100,000; one shared "
                          "randomization draw per dataset across the three comparisons (which the "
                          "real run also does, since boot seeds every call with SEED=0)",
           "results": out, "seconds": round(time.time() - t0, 1)}
    name = "m7_tier_rule_calibration_SMOKE.json" if SMOKE else "m7_tier_rule_calibration.json"
    (REPO / "results" / name).write_text(json.dumps(res, indent=1))
    print(f"\nwrote results/{name} in {res['seconds']:.0f}s")


if __name__ == "__main__":
    main()
