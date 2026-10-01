"""E22: predictors of the cross-space recovery gap (m15/MEASUREMENTS.md, E22). Runpod, CPU/GPU.

    e22_gap_predictors.py features <name>   one space: declared features from the E20 cached vectors
    e22_gap_predictors.py assemble          results/m15_e22_gap_predictors.json
    e22_gap_predictors.py all               every space, then assemble
    e22_gap_predictors.py selftest          synthetic check of the feature and validation code

Features are computed per space and workload from `work/m15/e20/<name>/` (document vectors,
teacher and table query vectors, exact top-100 ids and scores) and, for the fit-query cloud, the
teacher's cached `trainq-337981` vectors (a seeded 20,000-row sample). The gap itself comes from
the committed E20 result. No new encoding; `features` for a teacher needs M7_ENCODER set only for
the cached fit-query read.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for p in ("m15src", "m8src", "m7src"):
    if str(REPO / p) not in sys.path:
        sys.path.insert(0, str(REPO / p))
from common import receipt, utc_now, write_result  # noqa: E402

E20_DIR = REPO / "work" / "m15" / "e20"
OUT_DIR = REPO / "work" / "m15" / "e22"
E20_RESULT = REPO / "results" / "m15_e20_ann_spaces.json"
RESULT = REPO / "results" / "m15_e22_gap_predictors.json"
WORKLOADS = ("fiqa", "scidocs", "trec-covid")
SEED = 20260930
DOC_SAMPLE = 5000
FITQ_SAMPLE = 20000
FEATURES = ("qdoc_ratio_table", "qdoc_ratio_teacher", "qdoc_ratio_delta", "hubness_delta",
            "top10_overlap", "effrank_fitq_teacher", "effrank_q_table_over_teacher",
            "top1_cos_delta", "margin_delta", "log2_width", "log10_ndocs")


def eligible():
    d = json.loads(E20_RESULT.read_text())
    return [n for n in d["spaces"] if n != "arctic-embed-l-mean"]


# ---------------------------------------------------------------- features

def nn_dist_docs(dv, rng):
    """Mean nearest-neighbour cosine distance among a document sample (self excluded)."""
    idx = rng.choice(len(dv), size=min(DOC_SAMPLE, len(dv)), replace=False)
    S = dv[idx] @ dv.T
    S[np.arange(len(idx)), idx] = -np.inf
    return float(np.mean(1 - S.max(axis=1)))


def qdoc_ratio(scores_top1, doc_nn):
    return float(np.mean(1 - scores_top1) / doc_nn)


def gini(counts):
    c = np.sort(np.asarray(counts, dtype=np.float64))
    n = len(c)
    if n == 0 or c.sum() == 0:
        return 0.0
    return float((2 * np.sum((np.arange(1, n + 1)) * c) / (n * c.sum())) - (n + 1) / n)


def hubness(exact_ids, n_docs):
    counts = np.bincount(exact_ids[:, :10].ravel(), minlength=n_docs)
    return gini(counts[counts > 0])


def effective_rank(X):
    Xc = X - X.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False) ** 2
    return float(s.sum() ** 2 / (s ** 2).sum())


def space_features(name):
    out = OUT_DIR / f"{name}.json"
    if out.exists():
        print(f"{name}: features already done", flush=True)
        return
    rng = np.random.default_rng(SEED)
    vec = json.loads((E20_DIR / name / "vectors.json").read_text())
    feats = {}
    # Fit-query cloud of the teacher: a seeded 20k sample of the cached trainq vectors.
    import e8_towers as E8
    from teacher import QUERY_PREFIX, encode_cached
    import torch
    texts = E8.fit_list()
    Y = encode_cached(f"trainq-{len(texts)}", texts, prefix=QUERY_PREFIX, dtype=torch.float16,
                      verbose=False)
    sample = np.sort(rng.choice(len(texts), size=FITQ_SAMPLE, replace=False))
    effrank_fitq = effective_rank(np.asarray(Y[sample], dtype=np.float32))
    for ds in WORKLOADS:
        d = E20_DIR / name
        dv = np.load(d / f"{ds}-docs.npy").astype(np.float32)
        q = {p: np.load(d / f"{ds}-{p}-q.npy") for p in ("teacher", "table")}
        ids = {p: np.load(d / f"{ds}-{p}-exact-ids.npy") for p in ("teacher", "table")}
        top1 = {p: np.einsum("ij,ij->i", q[p], dv[ids[p][:, 0]]) for p in q}
        doc_nn = nn_dist_docs(dv, np.random.default_rng(SEED))
        g = vec["workloads"][ds]["paths"]
        f = {"qdoc_ratio_table": qdoc_ratio(top1["table"], doc_nn),
             "qdoc_ratio_teacher": qdoc_ratio(top1["teacher"], doc_nn),
             "hubness_delta": hubness(ids["table"], len(dv)) - hubness(ids["teacher"], len(dv)),
             "top10_overlap": float(np.mean([len(set(a[:10]) & set(b[:10])) / 10
                                             for a, b in zip(ids["table"], ids["teacher"])])),
             "effrank_fitq_teacher": effrank_fitq,
             "effrank_q_table_over_teacher": effective_rank(q["table"]) / effective_rank(q["teacher"]),
             "top1_cos_delta": g["table"]["geometry"]["top1_cos_quantiles"][1]
                               - g["teacher"]["geometry"]["top1_cos_quantiles"][1],
             "margin_delta": g["table"]["geometry"]["top1_minus_top10_quantiles"][1]
                             - g["teacher"]["geometry"]["top1_minus_top10_quantiles"][1],
             "log10_ndocs": float(np.log10(len(dv))), "doc_nn_distance": doc_nn}
        f["qdoc_ratio_delta"] = f["qdoc_ratio_table"] - f["qdoc_ratio_teacher"]
        feats[ds] = f
        print(f"  {name} {ds}: ratio_delta {f['qdoc_ratio_delta']:+.3f} overlap {f['top10_overlap']:.3f} "
              f"hub_delta {f['hubness_delta']:+.3f}", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    E8.atomic_json(out, {"name": name, "features": feats})


# ---------------------------------------------------------------- analysis

def ridge_fit(X, y, lam=1.0):
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-12
    Z = (X - mu) / sd
    Z1 = np.column_stack([np.ones(len(y)), Z])
    P = lam * np.eye(Z1.shape[1])
    P[0, 0] = 0
    beta = np.linalg.solve(Z1.T @ Z1 + P, Z1.T @ y)
    return mu, sd, beta


def ridge_predict(model, X):
    mu, sd, beta = model
    return np.column_stack([np.ones(len(X)), (X - mu) / sd]) @ beta


def held_out(X, y, groups, feature_idx=None):
    from scipy.stats import spearmanr
    Xs = X if feature_idx is None else X[:, feature_idx]
    pred = np.empty(len(y))
    for g in set(groups):
        m = np.array([h != g for h in groups])
        pred[~m] = ridge_predict(ridge_fit(Xs[m], y[m]), Xs[~m])
    return {"spearman": float(spearmanr(pred, y)[0]), "mae": float(np.mean(np.abs(pred - y)))}


def boot_spearman(a, b, draws=10_000, seed=SEED):
    from scipy.stats import spearmanr
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


def assemble():
    from scipy.stats import spearmanr
    e20 = json.loads(E20_RESULT.read_text())
    names = eligible()
    rows = []
    for n in names:
        f = json.loads((OUT_DIR / f"{n}.json").read_text())["features"]
        for ds in WORKLOADS:
            w = e20["spaces"][n]["workloads"][ds]
            r = {"space": n, "family": e20["spaces"][n]["family"], "workload": ds,
                 "gap": 100 * float(np.mean(w["table_minus_teacher_recovery_at_ref_ef"])),
                 "log2_width": float(np.log2(json.loads((E20_DIR / n / "vectors.json").read_text())
                                             .get("dim", 0) or 1)) if False else None}
            r.update(f[ds])
            rows.append(r)
    # width from the E19 result (dims per teacher)
    dims = {k: v["dim"] for k, v in json.loads((REPO / "results" / "m15_e19_head_screen.json").read_text())["configs"].items()}
    for r in rows:
        r["log2_width"] = float(np.log2(dims[r["space"]]))
    y = np.array([r["gap"] for r in rows])
    X = np.column_stack([[r[k] for r in rows] for k in FEATURES])
    fam = [r["family"] for r in rows]
    wl = [r["workload"] for r in rows]
    uni = {}
    for j, k in enumerate(FEATURES):
        uni[k] = {"spearman_all75": float(spearmanr(X[:, j], y)[0]),
                  "bootstrap95": boot_spearman(X[:, j], y),
                  "per_workload": {ds: float(spearmanr(X[[i for i, w in enumerate(wl) if w == ds], j],
                                                       y[[i for i, w in enumerate(wl) if w == ds]])[0])
                                   for ds in WORKLOADS}}
    const_lofo = {"spearman": None, "mae": float(np.mean(np.abs(y - y.mean())))}
    models = {"all_features": held_out(X, y, fam),
              "all_features_leave_workload_out": held_out(X, y, wl),
              "covariates_only_width_ndocs": held_out(X, y, fam, [FEATURES.index("log2_width"), FEATURES.index("log10_ndocs")]),
              "placement_only": held_out(X, y, fam, [FEATURES.index(k) for k in
                                                     ("qdoc_ratio_delta", "hubness_delta", "top10_overlap",
                                                      "effrank_q_table_over_teacher")])}
    for k in FEATURES:
        models[f"single_{k}"] = held_out(X, y, fam, [FEATURES.index(k)])
    write_result(RESULT, {
        "status": "COMPLETE", "measurement": "E22 (exploratory, pre-specified before computing features)",
        "scope": "75 space-by-workload points, 25 spaces; families and workloads are not independent; "
                 "ridge lambda 1 on standardized features; declared features only",
        "features": list(FEATURES), "n_points": len(rows), "univariate": uni,
        "held_out_models": models, "constant_predictor_mae": const_lofo["mae"], "rows": rows,
        "receipt": receipt(__file__, [{"e20_result": str(E20_RESULT.relative_to(REPO))}], utc_now(), ("scipy",))})


def all_steps():
    py = sys.executable
    env = {**os.environ, "PYTHONPATH": f"{REPO / 'm15src'}:{REPO / 'm8src'}:{REPO / 'm7src'}"}
    for name in eligible():
        subprocess.run([py, __file__, "features", name], check=True, cwd=REPO,
                       env={**env, "M7_ENCODER": name})
    subprocess.run([py, __file__, "assemble"], check=True, env=env, cwd=REPO)


def selftest():
    rng = np.random.default_rng(0)
    assert abs(gini([1, 1, 1, 1])) < 1e-9 and gini([0, 0, 0, 10]) > 0.7
    X = rng.normal(size=(500, 8)); X[:, 0] *= 10
    er = effective_rank(X)
    assert 1.0 < er < 8.0, er
    assert effective_rank(rng.normal(size=(500, 8))) > 7.0
    dv = rng.normal(size=(300, 16)); dv /= np.linalg.norm(dv, axis=1, keepdims=True)
    assert 0 < nn_dist_docs(dv, rng) < 1
    y = X[:, 1] * 2 + rng.normal(0, 0.1, 500); groups = list(rng.choice(["a", "b", "c"], 500))
    h = held_out(X, y, groups)
    assert h["spearman"] > 0.9, h
    print(f"selftest ok: gini, effective rank {er:.2f}, held-out rho {h['spearman']:.2f}")


if __name__ == "__main__":
    cmd, *rest = sys.argv[1:] or ["all"]
    {"assemble": assemble, "all": all_steps, "selftest": selftest,
     "features": lambda: space_features(rest[0])}[cmd]()
