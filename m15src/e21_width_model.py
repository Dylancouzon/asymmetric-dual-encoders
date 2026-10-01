"""E21: conditional teacher-selection analysis on existing data (m15/MEASUREMENTS.md, E21).

    e21_width_model.py            writes results/m15_e21_width_model.json
    e21_width_model.py selftest   synthetic check of the nested fit and leave-family-out logic

Inputs: results/m15_e19_head_screen.json (26 pooled checkpoints with teacher score, table and head
six-set scores, dev scores, width, family). No encoding, scoring, or protected access.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "m15src"))
from common import receipt, utc_now, write_result  # noqa: E402

E19 = REPO / "results" / "m15_e19_head_screen.json"
RESULT = REPO / "results" / "m15_e21_width_model.json"
CONTROL = "arctic-embed-l-mean"
BACKBONE = "bge-small-en-v1.5"
SEED = 20260930
DRAWS = 10_000
RECIPES = {"head": ("six_head_macro_all6", "dev_head", "six_head_macro_clean4"),
           "table": ("recipe1_six_table_macro_all6", "recipe1_dev_table", "recipe1_six_table_macro_clean4")}


def ols(X, y):
    X1 = np.column_stack([np.ones(len(y)), X])
    beta = np.linalg.lstsq(X1, y, rcond=None)[0]
    pred = X1 @ beta
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return beta, pred, float(r2)


def standardized(beta, X, y):
    return [float(b * np.std(X[:, j]) / np.std(y)) for j, b in enumerate(beta[1:])]


def boot_beta(X, y, draws=DRAWS, seed=SEED):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(draws):
        i = rng.integers(0, len(y), len(y))
        if len(set(i)) <= X.shape[1] + 1:
            continue
        try:
            out.append(standardized(ols(X[i], y[i])[0], X[i], y[i]))
        except np.linalg.LinAlgError:
            continue
    out = np.array(out)
    return [[float(np.quantile(out[:, j], .025)), float(np.quantile(out[:, j], .975))]
            for j in range(out.shape[1])]


def leave_one_family_out(X, y, fam):
    pred = np.empty(len(y))
    for f in set(fam):
        m = np.array([g != f for g in fam])
        beta = ols(X[m], y[m])[0]
        pred[~m] = np.column_stack([np.ones((~m).sum()), X[~m]]) @ beta
    return pred


def leave_one_out(X, y):
    pred = np.empty(len(y))
    for i in range(len(y)):
        m = np.ones(len(y), bool)
        m[i] = False
        beta = ols(X[m], y[m])[0]
        pred[i] = np.concatenate([[1.0], X[i]]) @ beta
    return pred


def partial_spearman(y, x, z):
    Z = np.column_stack([np.ones(len(y)), z])
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    return float(spearmanr(ry, rx)[0])


def boot_spearman(a, b, draws=DRAWS, seed=SEED):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    vals = []
    for _ in range(draws):
        i = rng.integers(0, len(a), len(a))
        if len(set(i)) > 2:
            v = spearmanr(a[i], b[i])[0]
            if np.isfinite(v):
                vals.append(v)
    return [float(np.quantile(vals, .025)), float(np.quantile(vals, .975))]


def analyse(rows, names, recipe, outcome_key, dev_key):
    y = np.array([rows[n][outcome_key] for n in names])
    teach = np.array([rows[n]["teacher_six_macro_all6"] for n in names])
    width = np.log2(np.array([rows[n]["dim"] for n in names], dtype=float))
    dev = np.array([rows[n][dev_key] for n in names])
    fam = [rows[n]["family"] for n in names]
    fams = sorted(set(fam))
    famX = np.column_stack([[1.0 if f == g else 0.0 for f in fam] for g in fams[1:]]) if len(fams) > 1 else None
    models = {"M1_teacher": np.column_stack([teach]),
              "M2_teacher_width": np.column_stack([teach, width]),
              "M4_teacher_width_dev": np.column_stack([teach, width, dev])}
    if famX is not None:
        models["M3_teacher_width_family"] = np.column_stack([teach, width, famX])
        models["M5_teacher_width_family_dev"] = np.column_stack([teach, width, famX, dev])
    out = {"n": len(names), "outcome": outcome_key, "models": {}}
    for name, X in models.items():
        beta, pred, r2 = ols(X, y)
        labels = ["teacher", "width"][:X.shape[1]] if "dev" not in name else None
        entry = {"r2_in_sample": r2, "std_beta": standardized(beta, X, y)}
        if "family" not in name:
            entry["std_beta_bootstrap95"] = boot_beta(X, y)
            lofo = leave_one_family_out(X, y, fam)
            entry["leave_one_family_out"] = {"spearman_pred_vs_actual": float(spearmanr(lofo, y)[0]),
                                             "mae": float(np.mean(np.abs(lofo - y)))}
            loo = leave_one_out(X, y)
            entry["leave_one_checkpoint_out"] = {"spearman_pred_vs_actual": float(spearmanr(loo, y)[0]),
                                                 "mae": float(np.mean(np.abs(loo - y)))}
        out["models"][name] = entry
    out["std_beta_order"] = {"M1_teacher": ["teacher"], "M2_teacher_width": ["teacher", "width"],
                             "M4_teacher_width_dev": ["teacher", "width", "dev"],
                             "M3_teacher_width_family": ["teacher", "width"] + [f"family={g}" for g in fams[1:]],
                             "M5_teacher_width_family_dev": ["teacher", "width"] + [f"family={g}" for g in fams[1:]] + ["dev"]}
    out["spearman"] = {
        "student_vs_teacher": float(spearmanr(y, teach)[0]),
        "student_vs_teacher_bootstrap95": boot_spearman(y, teach),
        "student_vs_width": float(spearmanr(y, width)[0]),
        "teacher_vs_width": float(spearmanr(teach, width)[0]),
        "partial_student_vs_teacher_given_width": partial_spearman(y, teach, width),
        "dev_vs_student": float(spearmanr(dev, y)[0])}
    out["within_width"] = {}
    for w in sorted(set(width)):
        m = width == w
        if m.sum() > 3:
            out["within_width"][str(int(2 ** w))] = {"n": int(m.sum()),
                                                      "spearman_student_vs_teacher": float(spearmanr(y[m], teach[m])[0]),
                                                      "bootstrap95": boot_spearman(y[m], teach[m])}
    # Selection regret per rule.
    best = float(y.max())
    loo_m2 = leave_one_out(models["M2_teacher_width"], y)
    rules = {"strongest_teacher": int(np.argmax(teach)),
             "strongest_teacher_widest_band": int(np.argmax(np.where(width == width.max(), teach, -np.inf))),
             "M2_prediction_leave_one_out": int(np.argmax(loo_m2)),
             "dev_screen": int(np.argmax(dev))}
    out["selection_regret"] = {r: {"pick": names[i], "student": float(y[i]), "regret": best - float(y[i])}
                               for r, i in rules.items()}
    out["best_student"] = {"name": names[int(np.argmax(y))], "student": best}
    return out


def main():
    d = json.loads(E19.read_text())
    rows = {n: r for n, r in d["configs"].items() if n != CONTROL}
    registered = [n for n in d["rosters"]["registered_ten"]["names"] if n in rows]
    pooled = list(rows)
    result = {"status": "COMPLETE", "measurement": "E21 (exploratory, pre-specified before running; existing data)",
              "scope": "26 related checkpoints, three width levels, two closed-form student recipes; "
                       "hypothesis about width, not a mechanism; no new scoring",
              "rosters": {}, "receipt": None}
    for roster_name, names in (("pooled", pooled), ("pooled_without_backbone", [n for n in pooled if n != BACKBONE]),
                               ("registered_ten", registered)):
        result["rosters"][roster_name] = {}
        for recipe, (ok, dk, ck) in RECIPES.items():
            result["rosters"][roster_name][recipe] = {
                "all6": analyse(rows, names, recipe, ok, dk),
                "clean4": analyse(rows, names, recipe, ck, dk)}
    # Registered-ten fit predicting the 16 exploratory checkpoints (M2), both recipes.
    expl = [n for n in pooled if n not in registered]
    result["registered_fit_predicts_exploratory"] = {}
    for recipe, (ok, dk, ck) in RECIPES.items():
        tr_y = np.array([rows[n][ok] for n in registered])
        tr_X = np.column_stack([[rows[n]["teacher_six_macro_all6"] for n in registered],
                                np.log2([rows[n]["dim"] for n in registered])])
        beta = ols(tr_X, tr_y)[0]
        te_X = np.column_stack([np.ones(len(expl)), [rows[n]["teacher_six_macro_all6"] for n in expl],
                                np.log2([rows[n]["dim"] for n in expl])])
        pred = te_X @ beta
        te_y = np.array([rows[n][ok] for n in expl])
        te_t = np.array([rows[n]["teacher_six_macro_all6"] for n in expl])
        result["registered_fit_predicts_exploratory"][recipe] = {
            "n_train": len(registered), "n_test": len(expl),
            "spearman_pred_vs_actual": float(spearmanr(pred, te_y)[0]),
            "spearman_pred_vs_actual_bootstrap95": boot_spearman(pred, te_y),
            "spearman_teacher_only_vs_actual": float(spearmanr(te_t, te_y)[0]),
            "mae": float(np.mean(np.abs(pred - te_y)))}
    result["receipt"] = receipt(__file__, [{"e19_result": str(E19.relative_to(REPO))}], utc_now(), ("scipy",))
    write_result(RESULT, result)


def selftest():
    rng = np.random.default_rng(1)
    n = 40
    teach = rng.uniform(0.3, 0.6, n)
    width = rng.choice([8.585, 9.585, 10.0], n)       # log2 of 384, 768, 1024
    y = 0.8 * teach - 0.15 * (width - 9) + rng.normal(0, 0.01, n)
    X = np.column_stack([teach, width])
    beta, _, r2 = ols(X, y)
    sb = standardized(beta, X, y)
    assert r2 > 0.9 and sb[0] > 0 and sb[1] < 0, (r2, sb)
    fam = list(rng.choice(["a", "b", "c", "d"], n))
    pred = leave_one_family_out(X, y, fam)
    assert spearmanr(pred, y)[0] > 0.9
    assert partial_spearman(y, teach, width) > 0.8
    print(f"selftest ok: r2 {r2:.3f}, std betas {sb[0]:+.2f} {sb[1]:+.2f}, lofo rho {spearmanr(pred, y)[0]:.2f}")


if __name__ == "__main__":
    selftest() if "selftest" in sys.argv else main()
