"""Pre-registered paired statistics (instructions-m7.md + Codex B3 fix, 2026-08-26).

Division of labor: `paired` gives the delta and percentile INTERVALS; `signflip` gives the
P-VALUE (paired sign-flip randomization). The bootstrap tail mass `paired` used to call "p" is a
CI-inversion heuristic, not P(T >= t_obs | H0), so Holm over it controlled nothing (Codex B3).

Paired on per-query vectors; resample queries WITHIN each dataset; recompute the macro per
replicate; B=10,000; fixed logged seed; per-dataset CIs alongside the macro.

Confirmatory decisions are exactly three final-run comparisons -- int8 table vs
lr-dense-pertask, int8 table vs BM25, released system vs OpenSearch -- one-sided, Holm
step-down at family alpha = 0.025. The dev int8-equivalence gate is a separate dev-stage
decision. Everything else produced here is exploratory and must be labeled as such.
"""
import numpy as np

B = 10_000
SEED = 0


def _align(a, b, strict=False):
    """-> {dataset: (arr_a, arr_b)} over the intersection of datasets, aligned by sorted qid.

    strict=True refuses any silent shrinkage: a confirmatory comparison must never quietly drop
    a dataset or a query, because nDCG over the intersection is a different statistic than the
    pre-registered one (Codex M-perquery). Exploratory callers keep the permissive default."""
    if strict:
        if set(a) != set(b):
            raise ValueError(f"dataset sets differ: only-a={sorted(set(a)-set(b))} "
                             f"only-b={sorted(set(b)-set(a))}")
        for ds in a:
            if set(a[ds]) != set(b[ds]):
                raise ValueError(f"{ds}: qid sets differ (a-only {len(set(a[ds])-set(b[ds]))}, "
                                 f"b-only {len(set(b[ds])-set(a[ds]))})")
    out = {}
    for ds in sorted(set(a) & set(b)):
        qs = sorted(set(a[ds]) & set(b[ds]))
        if not qs:
            continue
        out[ds] = (np.array([a[ds][q] for q in qs], dtype=np.float64),
                   np.array([b[ds][q] for q in qs], dtype=np.float64))
    return out


def signflip(a, b, R=100_000, seed=SEED, alternative="greater", strict=True):
    """Paired sign-flip randomization test on the macro delta (a - b). THE p-value for any
    confirmatory decision; `paired` supplies intervals only (its tail mass is not a p-value —
    Codex B3, and Holm controls nothing over an invalid p).

    Null: a and b exchangeable per query, so each per-query difference is symmetric about 0.
    Each query's sign flips independently; the macro is recomputed per replicate as the mean of
    per-dataset means — flipped diffs average within their dataset first, never pooled, because
    the pre-registered statistic weights a TREC-COVID query ~13x a FiQA query.
    p = (1 + #{T_r >= T_obs}) / (1 + R): a valid p-value for any n (P(p<=α|H0) <= α), so Holm
    step-down over these controls family error. Min attainable p = 1/(R+1) — report as a bound.
    """
    pairs = _align(a, b, strict=strict)
    if not pairs:
        raise ValueError("no overlapping datasets/queries")
    d = {ds: da - db for ds, (da, db) in pairs.items()}
    k = len(d)
    t_obs = sum(float(v.mean()) for v in d.values()) / k
    rng = np.random.default_rng(seed)
    t = np.zeros(R)
    chunk = 10_000                     # caps the sign matrix at ~10k x n_ds doubles (~120 MB max)
    for ds, v in d.items():
        for a0 in range(0, R, chunk):
            b0 = min(a0 + chunk, R)
            signs = rng.integers(0, 2, size=(b0 - a0, v.size)) * 2 - 1
            t[a0:b0] += (signs * v).mean(1) / k
    if alternative == "greater":
        p = (1 + int((t >= t_obs).sum())) / (1 + R)
    elif alternative == "less":
        p = (1 + int((t <= t_obs).sum())) / (1 + R)
    else:
        p = (1 + int((np.abs(t) >= abs(t_obs)).sum())) / (1 + R)
    return {"delta": round(t_obs, 4), "p": float(p),
            "p_str": (f"{p:.2e} (MC floor 1/(R+1))" if p == 1 / (1 + R) else f"{p:.5f}"),
            "R": R, "seed": seed, "alternative": alternative,
            "method": "paired sign-flip randomization, macro-structured",
            "per_dataset_n": {ds: int(v.size) for ds, v in d.items()},
            "per_dataset_nonzero": {ds: int((v != 0).sum()) for ds, v in d.items()}}


def paired(a, b, B=B, seed=SEED, alternative="two-sided"):
    """a, b: {dataset: {qid: score}}. Returns the macro delta (a-b), 95% CI, p, per-dataset rows."""
    pairs = _align(a, b)
    if not pairs:
        raise ValueError("no overlapping datasets/queries")
    rng = np.random.default_rng(seed)
    k = len(pairs)
    deltas = np.zeros(B)
    base = 0.0
    per = {}
    for ds, (da, db) in pairs.items():
        n = len(da)
        idx = rng.integers(0, n, size=(B, n))
        d_ds = da[idx].mean(1) - db[idx].mean(1)
        deltas += d_ds / k
        base += (da.mean() - db.mean()) / k
        lo, hi = np.percentile(d_ds, [2.5, 97.5])
        per[ds] = {"n": int(n), "delta": round(float(da.mean() - db.mean()), 4),
                   "ci95": [round(float(lo), 4), round(float(hi), 4)]}
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    if alternative == "greater":       # H1: a > b
        p = float((deltas <= 0).mean())
        one_sided_lo = float(np.percentile(deltas, 2.5))
    elif alternative == "less":
        p = float((deltas >= 0).mean())
        one_sided_lo = None
    else:
        p = 2 * float(min((deltas < 0).mean(), (deltas > 0).mean()))
        one_sided_lo = None
    return {"delta": round(base, 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "boot_tail": p, "boot_tail_str": (f"<{1/B}" if p == 0 else f"{p:.4f}"),
            "_boot_tail_note": "bootstrap tail mass, NOT a p-value; use signflip() for p",
            "alternative": alternative, "B": B, "seed": seed,
            "one_sided_lower_2.5": None if one_sided_lo is None else round(one_sided_lo, 4),
            "per_dataset": per,
            "resolved": bool(lo > 0 or hi < 0)}


def upper_bound_one_sided(a, b, B=B, seed=SEED, level=0.975):
    """One-sided upper bound on the macro (a - b). Used by the int8-equivalence dev gate:
    the 97.5% upper bound of (fp16 - int8) must be below 0.005."""
    pairs = _align(a, b)
    rng = np.random.default_rng(seed)
    k = len(pairs)
    deltas = np.zeros(B)
    base = 0.0
    for ds, (da, db) in pairs.items():
        n = len(da)
        idx = rng.integers(0, n, size=(B, n))
        deltas += (da[idx].mean(1) - db[idx].mean(1)) / k
        base += (da.mean() - db.mean()) / k
    return {"delta": round(base, 5), "upper": round(float(np.percentile(deltas, level * 100)), 5),
            "level": level, "B": B, "seed": seed}


def holm(pvals, alpha=0.025):
    """Holm step-down. pvals: {name: p}. Returns {name: {p, adj_threshold, reject}}."""
    order = sorted(pvals, key=lambda k: pvals[k])
    m = len(order)
    out, still = {}, True
    for i, name in enumerate(order):
        thr = alpha / (m - i)
        rej = still and pvals[name] <= thr
        if not rej:
            still = False
        out[name] = {"p": pvals[name], "threshold": thr, "reject": bool(rej), "rank": i + 1}
    return out


def from_perquery_json(blob, system, datasets=None):
    """results/perquery.json -> {dataset: {qid: score}} for one frozen comparator system."""
    out = {}
    for ds, d in blob["datasets"].items():
        if datasets and ds not in datasets:
            continue
        if system not in d["systems"]:
            continue
        out[ds] = dict(zip(d["qids"], d["systems"][system]))
    return out
