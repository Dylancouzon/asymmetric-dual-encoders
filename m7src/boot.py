"""Pre-registered paired bootstrap (instructions-m7.md).

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


def _align(a, b):
    """-> {dataset: (arr_a, arr_b)} over the intersection of datasets, aligned by sorted qid."""
    out = {}
    for ds in sorted(set(a) & set(b)):
        qs = sorted(set(a[ds]) & set(b[ds]))
        if not qs:
            continue
        out[ds] = (np.array([a[ds][q] for q in qs], dtype=np.float64),
                   np.array([b[ds][q] for q in qs], dtype=np.float64))
    return out


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
            "p": p, "p_str": (f"<{1/B}" if p == 0 else f"{p:.4f}"),
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
