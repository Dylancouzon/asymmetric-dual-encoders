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


def _align_ids(a, b, strict=False):
    """-> {dataset: (qids, arr_a, arr_b)} over the intersection of datasets, aligned by sorted qid.

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
        out[ds] = (qs, np.array([a[ds][q] for q in qs], dtype=np.float64),
                   np.array([b[ds][q] for q in qs], dtype=np.float64))
    return out


def _align(a, b, strict=False):
    """The dependence-blind view: {dataset: (arr_a, arr_b)}. Kept byte-identical in behaviour
    because every committed number was produced through it."""
    return {ds: (x, y) for ds, (_, x, y) in _align_ids(a, b, strict=strict).items()}


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


def paired(a, b, B=B, seed=SEED, alternative="two-sided", strict=False):
    """a, b: {dataset: {qid: score}}. Returns the macro delta (a-b), 95% CI, p, per-dataset rows.

    `strict=True` refuses to silently score the intersection -- the ledger requires it on every
    confirmatory path, and a gate condition is confirmatory-shaped even though it is a dev
    decision."""
    pairs = _align(a, b, strict=strict)
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
    # ci95 is ROUNDED FOR DISPLAY. Never decide on it: a true lower endpoint of +4e-5 rounds to
    # 0.0000 and would read as unresolved (Codex review #3b MAJOR 2). `ci95_raw`/`delta_raw` are
    # what an adoption bar must compare against.
    return {"delta": round(base, 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "delta_raw": float(base), "ci95_raw": [float(lo), float(hi)],
            "boot_tail": p, "boot_tail_str": (f"<{1/B}" if p == 0 else f"{p:.4f}"),
            "_boot_tail_note": "bootstrap tail mass, NOT a p-value; use signflip() for p",
            "alternative": alternative, "B": B, "seed": seed,
            "one_sided_lower_2.5": None if one_sided_lo is None else round(one_sided_lo, 4),
            "per_dataset": per,
            "resolved": bool(lo > 0 or hi < 0)}


def upper_bound_one_sided(a, b, B=B, seed=SEED, level=0.975, dep=False, unit_of=None,
                          strict=False):
    """One-sided upper bound on the macro (a - b). Used by the int8-equivalence dev gate:
    the 97.5% upper bound of (fp16 - int8) must be below 0.005.

    `dep=True` uses the same stratified, shared-draw resampling as `paired_dep`, which the gate
    needs because the two held-out components are nested (review #3: "recheck int8 equivalence
    with dependence-preserving handling of the nested held-out components")."""
    if dep:
        r = paired_dep(a, b, B=B, seed=seed, alternative="two-sided", strict=strict,
                       unit_of=unit_of or unit_key)   # unit_key is defined below in this module
        # paired_dep already reports the percentile interval; the equivalence bound is its upper
        # endpoint at `level`, which for the default 0.975 is exactly ci95's upper end.
        if level != 0.975:
            raise ValueError("dependence-preserving bound is defined here at level 0.975 only")
        return {"delta": round(r["delta_raw"], 5), "upper": round(r["ci95_raw"][1], 5),
                "delta_raw": r["delta_raw"], "upper_raw": r["ci95_raw"][1],
                "level": level, "B": B, "seed": seed, "method": r["method"],
                "strata": r["strata"]}
    pairs = _align(a, b, strict=strict)
    rng = np.random.default_rng(seed)
    k = len(pairs)
    deltas = np.zeros(B)
    base = 0.0
    for ds, (da, db) in pairs.items():
        n = len(da)
        idx = rng.integers(0, n, size=(B, n))
        deltas += (da[idx].mean(1) - db[idx].mean(1)) / k
        base += (da.mean() - db.mean()) / k
    up = float(np.percentile(deltas, level * 100))
    return {"delta": round(base, 5), "upper": round(up, 5), "delta_raw": float(base),
            "upper_raw": up, "level": level, "B": B, "seed": seed,
            "method": "componentwise paired bootstrap (dependence-blind)"}


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


# ---- dependence-preserving variants (Codex review #3 BLOCKER 1, 2026-08-27) ---------------
#
# `heldout-longq` is literally a subset of `heldout-train` (m7src/heldout.py): the same 55 queries,
# the same corpus, the same qrels, hence bit-identical per-query nDCG (verified: max |diff| = 0.0).
# The macro weights them 1/6 as a component AND 55/7325 inside another component, so they are two
# observations of ONE underlying query. `signflip`/`paired` above give them independent signs and
# independent bootstrap draws, which discards that covariance. These variants do not change the
# statistic (the macro is defined with equal component weights and is unchanged); they change only
# its null/resampling distribution, i.e. the p-value and the interval.

def unit_key(ds, qid):
    """Which underlying query a (component, qid) observation is OF. Held-out components share one
    query space by construction; every other component is its own."""
    return ("heldout", qid) if ds.startswith("heldout-") else (ds, qid)


def strata(aligned, unit_of=unit_key):
    """Partition units by which components they appear in; that signature IS the stratum.

    Returns [{sig, n, idx: {ds: positions}}]. A unit appears at most once per component, so a
    stratum's units index each of its components with one aligned position array. With no shared
    units this degenerates to one stratum per component and every method below reduces to the
    dependence-blind one (up to RNG stream)."""
    pos = {}
    for ds, (qids, _, _) in aligned.items():
        for i, q in enumerate(qids):
            d = pos.setdefault(unit_of(ds, q), {})
            if ds in d:
                raise ValueError(f"duplicate qid {q!r} within component {ds}")
            d[ds] = i
    groups = {}
    for u, d in pos.items():
        groups.setdefault(tuple(sorted(d)), []).append(u)
    out = []
    for sig in sorted(groups):
        units = sorted(groups[sig], key=str)
        out.append({"sig": sig, "n": len(units),
                    "idx": {ds: np.array([pos[u][ds] for u in units], dtype=np.int64) for ds in sig}})
    return out


def _macro_of(pairs_or_diff, k):
    return sum(float(v.mean()) for v in pairs_or_diff.values()) / k


def signflip_dep(a, b, R=100_000, seed=SEED, alternative="greater", strict=True,
                 unit_of=unit_key, chunk=2000):
    """`signflip` with ONE shared sign per underlying query, applied in every component that
    query appears in. Same observed statistic, dependence-preserving null."""
    aligned = _align_ids(a, b, strict=strict)
    if not aligned:
        raise ValueError("no overlapping datasets/queries")
    d = {ds: (x - y) for ds, (_, x, y) in aligned.items()}
    k = len(d)
    n_ds = {ds: v.size for ds, v in d.items()}
    t_obs = _macro_of(d, k)
    st = strata(aligned, unit_of)
    rng = np.random.default_rng(seed)
    t = np.zeros(R)
    for a0 in range(0, R, chunk):
        m = min(chunk, R - a0)
        sums = {ds: np.zeros(m) for ds in d}
        for s in st:
            signs = rng.integers(0, 2, size=(m, s["n"])).astype(np.int8) * 2 - 1
            for ds, idx in s["idx"].items():
                sums[ds] += (signs * d[ds][idx]).sum(1)
        t[a0:a0 + m] = sum(sums[ds] / n_ds[ds] for ds in d) / k
    if alternative == "greater":
        p = (1 + int((t >= t_obs).sum())) / (1 + R)
    elif alternative == "less":
        p = (1 + int((t <= t_obs).sum())) / (1 + R)
    else:
        p = (1 + int((np.abs(t) >= abs(t_obs)).sum())) / (1 + R)
    return {"delta": round(t_obs, 4), "p": float(p),
            "p_str": (f"{p:.2e} (MC floor 1/(R+1))" if p == 1 / (1 + R) else f"{p:.5f}"),
            "R": R, "seed": seed, "alternative": alternative,
            "method": "paired sign-flip randomization, macro-structured, one sign per shared query",
            "strata": [{"components": list(s["sig"]), "n_units": s["n"]} for s in st],
            "shared_units": sum(s["n"] for s in st if len(s["sig"]) > 1),
            "per_dataset_n": {ds: int(v.size) for ds, v in d.items()}}


def paired_dep(a, b, B=B, seed=SEED, alternative="two-sided", unit_of=unit_key, strict=False,
               chunk=1000, share=True):
    """`paired` with a STRATIFIED bootstrap: resample units once per stratum and reuse that draw
    in every component the stratum feeds, so a shared query moves together everywhere it counts.

    Two things change at once versus `paired`, and the report must separate them (Codex review
    #3b MAJOR 1). (i) CONDITIONING: stratifying heldout-train into long/non-long and holding both
    stratum sizes fixed estimates a conditional variance -- the ordinary bootstrap lets the
    long-query fraction fluctuate, this one always draws exactly 55 long and 7,270 non-long.
    That is defensible because the suite and its component sizes are frozen, but it is not the
    dependence fix. (ii) DEPENDENCE: reusing one draw across every component a stratum feeds.
    `share=False` gives the middle term -- fixed strata, independent draws -- so
    ordinary -> share=False isolates the conditioning and share=False -> share=True isolates the
    covariance."""
    aligned = _align_ids(a, b, strict=strict)
    if not aligned:
        raise ValueError("no overlapping datasets/queries")
    k = len(aligned)
    n_ds = {ds: x.size for ds, (_, x, _) in aligned.items()}
    base = sum(float(x.mean() - y.mean()) for _, x, y in aligned.values()) / k
    st = strata(aligned, unit_of)
    rng = np.random.default_rng(seed)
    deltas = np.zeros(B)
    per_rep = {ds: np.zeros(B) for ds in aligned}
    for a0 in range(0, B, chunk):
        m = min(chunk, B - a0)
        sums = {ds: np.zeros(m) for ds in aligned}
        for s in st:
            draw = rng.integers(0, s["n"], size=(m, s["n"]))
            for ds, idx in s["idx"].items():
                _, x, y = aligned[ds]
                dv = (x - y)[idx]
                # share=False deliberately DISCARDS `draw` and redraws per component:
                # that is the whole difference between the two variants. The wasted
                # draw keeps the two RNG streams aligned; do not 'optimize' it away.
                d_ = draw if share else rng.integers(0, s["n"], size=(m, s["n"]))
                sums[ds] += dv[d_].sum(1)
        for ds in aligned:
            per_rep[ds][a0:a0 + m] = sums[ds] / n_ds[ds]
        deltas[a0:a0 + m] = sum(sums[ds] / n_ds[ds] for ds in aligned) / k
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    per = {}
    for ds, (_, x, y) in aligned.items():
        dlo, dhi = np.percentile(per_rep[ds], [2.5, 97.5])
        per[ds] = {"n": int(n_ds[ds]), "delta": round(float(x.mean() - y.mean()), 4),
                   "ci95": [round(float(dlo), 4), round(float(dhi), 4)],
                   "delta_raw": float(x.mean() - y.mean()),
                   "ci95_raw": [float(dlo), float(dhi)]}
    if alternative == "greater":
        tail = float((deltas <= 0).mean())
    elif alternative == "less":
        tail = float((deltas >= 0).mean())
    else:
        tail = 2 * float(min((deltas < 0).mean(), (deltas > 0).mean()))
    return {"delta": round(base, 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "delta_raw": float(base), "ci95_raw": [float(lo), float(hi)],
            "boot_tail": tail, "_boot_tail_note": "bootstrap tail mass, NOT a p-value",
            "alternative": alternative, "B": B, "seed": seed,
            "one_sided_lower_2.5": round(float(np.percentile(deltas, 2.5)), 4),
            "method": ("stratified paired bootstrap, shared draw across components"
                       if share else
                       "stratified paired bootstrap, INDEPENDENT draws (conditioning only, "
                       "not the dependence fix)"),
            "shared_draw": bool(share),
            "strata": [{"components": list(s["sig"]), "n_units": s["n"]} for s in st],
            "per_dataset": per, "resolved": bool(lo > 0 or hi < 0)}


def both_ways(a, b, alternative="greater"):
    """The side-by-side the review asks for, in three bootstrap steps so the two effects separate:
    ordinary -> `stratified_independent` isolates CONDITIONING on fixed stratum sizes;
    `stratified_independent` -> `dependence_preserving` isolates the shared-query COVARIANCE.
    Dev comparisons are SELECTION evidence in every column (review #3 MAJOR 1)."""
    return {"ordinary": {"paired": paired(a, b, alternative="two-sided"),
                         "signflip": signflip(a, b, alternative=alternative)},
            "stratified_independent": {
                "paired": paired_dep(a, b, alternative="two-sided", share=False),
                "_note": "fixed stratum sizes, independent draws: the conditioning step only"},
            "dependence_preserving": {"paired": paired_dep(a, b, alternative="two-sided"),
                                      "signflip": signflip_dep(a, b, alternative=alternative)},
            "_note": "exploratory selection evidence; the only confirmatory comparisons are the "
                     "three frozen-test ones in the final run"}


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
