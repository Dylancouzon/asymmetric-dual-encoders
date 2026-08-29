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


# t(0.975, df) for the small dfs this fit can have. Hardcoded so the module needs no scipy.
T975 = {5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def fit(rows):
    M = np.array([m for _, m, _ in rows])
    O = np.array([o for _, _, o in rows])
    ratios = O / (M / 100)
    a, b = np.polyfit(M, O, 1)
    resid = O - (a * M + b)
    n = len(rows)
    # Regression sigma uses n-2 (two fitted parameters), NOT ddof=1. An earlier version quoted
    # +-2 * resid.std(ddof=1) and called it a CI. That was wrong three times over: wrong dof, 2sd
    # is not 95% at n=9 (t(7)=2.365), and it omitted the prediction-interval widening term, which
    # matters most exactly where we use it -- extrapolating past the fit range.
    sigma = float(np.sqrt((resid ** 2).sum() / (n - 2)))
    return {
        "sigma_regression": sigma, "df": n - 2, "t975": T975.get(n - 2, 1.96),
        "mteb_mean": float(M.mean()), "Sxx": float(((M - M.mean()) ** 2).sum()),
        "n": len(rows), "mteb_range": [float(M.min()), float(M.max())],
        "ratio": {"mean": float(ratios.mean()), "sd": float(ratios.std(ddof=1)),
                  "min": float(ratios.min()), "max": float(ratios.max())},
        "affine": {"slope": float(a), "intercept": float(b),
                   "pearson_r": float(np.corrcoef(M, O)[0, 1]),
                   "resid_sd_ddof1_DEPRECATED": float(resid.std(ddof=1)),
                   "max_abs_resid": float(np.abs(resid).max())},
        "per_model": {n: {"mteb": m, "our6": o, "ratio": float(o / (m / 100)),
                          "affine_resid": float(o - (a * m + b))} for (n, m, o) in rows},
    }


def pred_interval(tx, mteb):
    """95% PREDICTION interval half-width for ONE new model's six-set score at this MTEB value.

    Prediction, not confidence: the interval is on a single new model, so it must carry the
    model-to-model residual (the leading 1.0), not only uncertainty in the fitted line. The
    widening term is what punishes extrapolation -- stella sits 3.8 MTEB beyond the fit range.
    """
    n = tx["n"]
    w = np.sqrt(1.0 + 1.0 / n + (mteb - tx["mteb_mean"]) ** 2 / tx["Sxx"])
    return float(tx["t975"] * tx["sigma_regression"] * w)


def main():
    tx, st = fit(TRANSFORMERS), fit(STATICS)
    a, b = tx["affine"]["slope"], tx["affine"]["intercept"]
    hi_anchor = 0.5042 / 51.68  # the single-point anchor the prior projection used

    # Verified 2026-08-26 by four parallel sweeps. Every row PASSES the hard filters
    # (licence permits commercial derived weights; vocab <= ~50K; dim <= 1024; runs on a 10 GB
    # 3080). MTEB figures are MTEB v1 English Retrieval (15-task BEIR avg) -- the only scale our
    # nine-model fit is valid on. Vendor tier per CLAUDE.md's relaxed rule.
    CANDIDATES = [
        # name, MTEB v1 Ret, vocab, dim, params M, licence, vendor tier
        ("stella_en_400M_v5", 58.97, 30528, 1024, 435, "MIT", "clean"),
        ("gte-large-en-v1.5", 57.91, 30528, 1024, 434, "Apache-2.0", "justify-alibaba"),
        ("arctic-embed-l", 55.98, 30522, 1024, 335, "Apache-2.0", "justify-snowflake-max"),
        ("gte-modernbert-base", 55.33, 50368, 768, 149, "Apache-2.0", "justify-alibaba"),
        ("arctic-embed-m-v1.5", 55.14, 30522, 768, 109, "Apache-2.0", "justify-snowflake-max"),
        ("bge-large-en-v1.5", 54.29, 30522, 1024, 335, "MIT", "clean"),
        ("gte-base-en-v1.5", 54.09, 30528, 768, 137, "Apache-2.0", "justify-alibaba"),
        ("bge-base-en-v1.5 (current)", 53.25, 30522, 768, 109, "MIT", "clean"),
        ("granite-embedding-english-r2", 53.10, 50368, 768, 149, "Apache-2.0", "justify-ibm"),
    ]
    cands = {}
    for name, mteb, vocab, dim, prm, lic, tier in CANDIDATES:
        six = a * mteb + b
        row = {"mteb_v1_retrieval": mteb, "vocab": vocab, "dim": dim, "params_m": prm,
               "licence": lic, "vendor_tier": tier,
               "table_mb_fp16": round(vocab * dim * 2 / 1e6, 1),
               "table_mb_int8": round(vocab * dim / 1e6, 1),
               "six_est_affine": round(six, 4),
               "pred_interval_95_halfwidth": round(pred_interval(tx, mteb), 4),
               "extrapolating_beyond_fit": bool(mteb > tx["mteb_range"][1]),
               "six_est_ratio_mean": round(tx["ratio"]["mean"] * mteb / 100, 4),
               "six_est_bge_small_anchor_PRIOR_METHOD": round(hi_anchor * mteb, 4),
               "extrapolation_beyond_fit_range": round(mteb - tx["mteb_range"][1], 2),
               "at_retention": {}}
        hw = pred_interval(tx, mteb)
        for r in (RETENTION_TODAY, 0.82, 0.85, 0.88, 0.91):
            v = six * r
            lo, hi = (six - hw) * r, (six + hw) * r
            row["at_retention"][f"{r:.4f}"] = {
                "point": round(v, 4), "pi95_teacher_only": [round(lo, 4), round(hi, 4)],
                "clears_lower_bound": {k: bool(lo > bar) for k, bar in BARS.items()},
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
          f"sigma {tx['sigma_regression']:.5f} (df {tx['df']}, t {tx['t975']})")
    for m in (53.25, 55.14, 57.91, 58.97):
        print(f"    95% PI half-width at MTEB {m}: +-{pred_interval(tx, m):.4f}"
              f"{'   <- EXTRAPOLATION' if m > tx['mteb_range'][1] else ''}")
    print(f"  statics (other harness) ratio mean {st['ratio']['mean']:.4f}\n")
    print(f"{'candidate':30s} {'MTEB':>5s} {'six':>6s} {'int8MB':>7s} {'tier':>22s} | "
          + " ".join(f"x{r:.0%}".rjust(7) for r in (RETENTION_TODAY, 0.85, 0.88, 0.91)))
    for n, r in cands.items():
        cells = " ".join(f"{r['at_retention'][f'{x:.4f}']['point']:7.4f}"
                         for x in (RETENTION_TODAY, 0.85, 0.88, 0.91))
        print(f"{n:30s} {r['mteb_v1_retrieval']:5.2f} {r['six_est_affine']:6.4f} "
              f"{r['table_mb_int8']:7.1f} {r['vendor_tier']:>22s} | {cells}")
    print(f"\nbars: bm25 {BARS['bm25']}  tier2 {BARS['tier2_release']}  "
          f"tier1 {BARS['tier1_aim']}")
    for n in ("stella_en_400M_v5", "gte-large-en-v1.5", "bge-base-en-v1.5 (current)"):
        for r in ("0.8500", "0.8800", "0.9100"):
            c = cands[n]["at_retention"][r]
            print(f"  {n:28s} x{r[:4]}: {c['point']:.4f} PI{c['pi95_teacher_only']} | tier1 "
                  f"point={str(c['point_clears']['tier1_aim']):5s} "
                  f"lower={str(c['clears_lower_bound']['tier1_aim']):5s} | tier2 "
                  f"lower={c['clears_lower_bound']['tier2_release']}")
    print("wrote results/m7_calibration.json")


if __name__ == "__main__":
    main()
