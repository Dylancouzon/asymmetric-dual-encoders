"""How well does MTEB-Retrieval predict OUR six-set? Fit it on the nine models we measured.

The M7 teacher projection was anchored on a SINGLE model measured both ways (bge-small,
ratio 0.976) and used to claim a teacher swap clears Tier 1. That anchor is one point, so it
carries no uncertainty and no check on bias. We have measured nine transformers on the six
(results/FINAL_MATRIX.md) whose official MTEB v1 English Retrieval (15-task BEIR) scores are
recorded in research/landscape.md sec.1 -- enough to fit the mapping and quote a residual.

Statics are excluded from the fit on purpose: their published numbers come from the Model2Vec
"MTEB Ret" column, a different harness and subset, so mixing them would corrupt the slope.
They are reported separately as a consistency check.

Writes results/m7_calibration.json. Pure arithmetic over already-committed numbers: no model
is run, no eval is touched, and in particular the six-set is NOT read -- these are the M4
numbers already in the repo.
"""
import json

import numpy as np

from _paths import REPO

# (name, MTEB v1 English Retrieval avg-15, our measured avg-6)
TRANSFORMERS = [
    ("all-MiniLM-L6-v2", 41.95, 0.4201),
    ("e5-small-v2", 49.04, 0.4541),
    ("gte-small", 49.46, 0.4837),
    ("arctic-embed-xs", 50.15, 0.4662),
    ("granite-small-r2", 50.90, 0.4947),
    ("bge-small-en-v1.5", 51.68, 0.5042),
    ("arctic-embed-s", 51.98, 0.4993),
    ("mdbr-leaf-ir", 53.55, 0.5123),
    ("arctic-embed-m-v1.5", 55.14, 0.5264),
]
STATICS = [  # Model2Vec "MTEB Ret" column -- different harness, never mixed into the fit
    ("potion-base-8M", 31.11, 0.3193),
    ("static-retrieval-mrl-en-v1", 34.95, 0.3527),
    ("potion-retrieval-32M", 35.06, 0.3601),
]
# Retention measured on the pinned dev suite, text-backed components (the six are all
# text-backed, so this is the right analogue). results/m7_gate_p1-objB.json.
RETENTION_TODAY = 0.7853
BARS = {"bm25": 0.4174, "tier2_release": 0.4583, "tier1_aim": 0.4868}


def fit(rows):
    M = np.array([m for _, m, _ in rows])
    O = np.array([o for _, _, o in rows])
    ratios = O / (M / 100)
    a, b = np.polyfit(M, O, 1)
    resid = O - (a * M + b)
    return {
        "n": len(rows), "mteb_range": [float(M.min()), float(M.max())],
        "ratio": {"mean": float(ratios.mean()), "sd": float(ratios.std(ddof=1)),
                  "min": float(ratios.min()), "max": float(ratios.max())},
        "affine": {"slope": float(a), "intercept": float(b),
                   "pearson_r": float(np.corrcoef(M, O)[0, 1]),
                   "resid_sd": float(resid.std(ddof=1)),
                   "max_abs_resid": float(np.abs(resid).max())},
        "per_model": {n: {"mteb": m, "our6": o, "ratio": float(o / (m / 100)),
                          "affine_resid": float(o - (a * m + b))} for (n, m, o) in rows},
    }


def main():
    tx, st = fit(TRANSFORMERS), fit(STATICS)
    a, b = tx["affine"]["slope"], tx["affine"]["intercept"]
    sd = tx["affine"]["resid_sd"]
    hi_anchor = 0.5042 / 51.68  # the single-point anchor the prior projection used

    cands = {}
    for name, mteb in [("bge-base-en-v1.5 (current)", 53.25), ("bge-large-en-v1.5", 54.29),
                       ("gte-base-en-v1.5", 54.09), ("gte-large-en-v1.5", 57.0),
                       ("stella_en_400M_v5", 58.97)]:
        six = a * mteb + b
        row = {"mteb_v1_retrieval": mteb, "six_est_affine": round(six, 4),
               "six_est_ratio_mean": round(tx["ratio"]["mean"] * mteb / 100, 4),
               "six_est_bge_small_anchor_PRIOR_METHOD": round(hi_anchor * mteb, 4),
               "extrapolation_beyond_fit_range": round(mteb - tx["mteb_range"][1], 2),
               "at_retention": {}}
        for r in (RETENTION_TODAY, 0.82, 0.85, 0.88, 0.91):
            v = six * r
            # +-2 resid sd on the teacher estimate, scaled by retention
            lo, hi = (six - 2 * sd) * r, (six + 2 * sd) * r
            row["at_retention"][f"{r:.4f}"] = {
                "point": round(v, 4), "ci_from_calibration_resid": [round(lo, 4), round(hi, 4)],
                "clears": {k: bool(lo > bar) for k, bar in BARS.items()},
                "point_clears": {k: bool(v > bar) for k, bar in BARS.items()}}
        cands[name] = row

    out = {"_note": "Calibration of MTEB v1 English Retrieval (15-task BEIR avg) onto our "
                    "six-set avg-6, fit on the nine transformers we measured ourselves. "
                    "Supersedes the single-anchor projection: that anchor (bge-small, ratio "
                    "0.976) is the 3rd-HIGHEST of nine, so it biased every teacher estimate "
                    "high. 'clears' uses the LOWER end of the calibration interval; "
                    "'point_clears' uses the point estimate. Retention is itself measured on "
                    "dev and assumed to transfer to the six -- an assumption, not a result.",
           "bars": BARS, "retention_today_text_backed": RETENTION_TODAY,
           "transformers_fit": tx, "statics_fit_separate_harness": st,
           "candidates": cands}
    (REPO / "results" / "m7_calibration.json").write_text(json.dumps(out, indent=1))

    print(f"transformers n={tx['n']} MTEB {tx['mteb_range']}")
    print(f"  ratio  mean {tx['ratio']['mean']:.4f} sd {tx['ratio']['sd']:.4f} "
          f"range [{tx['ratio']['min']:.4f},{tx['ratio']['max']:.4f}]  <- NOT tight")
    print(f"  affine slope {a:.5f} intercept {b:+.4f} r {tx['affine']['pearson_r']:.4f} "
          f"resid sd {sd:.4f}")
    print(f"  statics (other harness) ratio mean {st['ratio']['mean']:.4f}\n")
    print(f"{'candidate':28s} {'MTEB':>5s} {'six':>6s} | "
          + " ".join(f"x{r:.0%}".rjust(7) for r in (RETENTION_TODAY, 0.85, 0.88, 0.91)))
    for n, r in cands.items():
        cells = " ".join(f"{r['at_retention'][f'{x:.4f}']['point']:7.4f}"
                         for x in (RETENTION_TODAY, 0.85, 0.88, 0.91))
        print(f"{n:28s} {r['mteb_v1_retrieval']:5.2f} {r['six_est_affine']:6.4f} | {cells}")
    print(f"\nbars: bm25 {BARS['bm25']}  tier2 {BARS['tier2_release']}  "
          f"tier1 {BARS['tier1_aim']}")
    for n in cands:
        for r in ("0.8500", "0.8800"):
            c = cands[n]["at_retention"][r]
            print(f"  {n:28s} x{r[:4]}: {c['point']:.4f} "
                  f"CI{c['ci_from_calibration_resid']} tier1 point={c['point_clears']['tier1_aim']} "
                  f"lower={c['clears']['tier1_aim']}")
    print("wrote results/m7_calibration.json")


if __name__ == "__main__":
    main()
