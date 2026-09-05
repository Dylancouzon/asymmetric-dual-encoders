"""The COV macro and its paired stratified bootstrap — the contrast rule of §Screen.

`m9src/final_stats.bootstrap` is the registered M9 implementation and is hardcoded to six equally
weighted datasets (`k != 6` raises). The COV surface is **family-weighted**: four family IDs at
equal weight, units averaged within family, so BRIGHT's six slices carry 1/24 each and MedicalQA
carries 1/4 on its own. That is a different estimator, so it is a separate module rather than a
loosened one — M9's gate keeps its refusal.

Registered constants (§Screen): B = 200,000, seed 0, `inverted_cdf`, one-sided lower bound at
0.025/13. Nothing here reads them from prose; the caller passes them and `record()` stamps them.

The draw plan is materialised per unit from `SeedSequence(seed).spawn()` in **sorted unit order**,
consumed in fixed `chunk`-sized blocks, so it is reproducible from (seed, B, chunk, unit list) and
its digest is recorded. B = 200,000 x 13,416 queries is 21 GB as one array; chunking is what makes
it runnable, and it is therefore part of the plan definition, not an implementation detail.
"""
from __future__ import annotations

import hashlib
import numpy as np

FAMILIES = ("BRIGHT", "consumer-health", "finance", "legal")


def weights(unit_family):
    """{unit: family} -> {unit: weight}. Families equal, units equal within family."""
    fams = sorted(set(unit_family.values()))
    if not set(fams) <= set(FAMILIES):
        raise ValueError(f"unknown family in {fams}; registered are {FAMILIES}")
    n_in = {f: sum(1 for v in unit_family.values() if v == f) for f in fams}
    return {u: 1.0 / (len(fams) * n_in[f]) for u, f in unit_family.items()}


def macro(per_unit, unit_family):
    """{unit: {qid: score}} -> (macro, {family: mean}, {unit: mean})."""
    w = weights(unit_family)
    um = {u: float(np.mean(list(v.values()))) for u, v in per_unit.items()}
    if set(um) != set(w):
        raise ValueError(f"unit mismatch: scored {sorted(um)} vs weighted {sorted(w)}")
    fm = {f: float(np.mean([um[u] for u, g in unit_family.items() if g == f]))
          for f in sorted(set(unit_family.values()))}
    return float(sum(w[u] * um[u] for u in um)), fm, um


def align(a, b, unit_family):
    """Two {unit: {qid: score}} -> {unit: (qids, x, y)}, strict: same units, same qids."""
    if set(a) != set(b) or set(a) != set(unit_family):
        raise ValueError(f"unit sets differ: {sorted(a)} / {sorted(b)} / {sorted(unit_family)}")
    out = {}
    for u in sorted(a):
        qa, qb = set(a[u]), set(b[u])
        if qa != qb:
            raise ValueError(f"{u}: {len(qa)} vs {len(qb)} qids, {len(qa ^ qb)} unshared")
        if not qa:
            raise ValueError(f"{u}: empty")
        qids = sorted(qa)
        out[u] = (qids, np.array([a[u][q] for q in qids], dtype=np.float64),
                  np.array([b[u][q] for q in qids], dtype=np.float64))
    return out


def contrast(aligned, unit_family, B=200_000, seed=0, quantile=0.025 / 13,
             method="inverted_cdf", chunk=5_000):
    """Paired stratified bootstrap of the family-weighted macro difference (a - b).

    Returns the point estimate, the one-sided lower bound at `quantile`, and the DISTANCE
    between them, which is the quantity §Surfaces registers as the resolution number.
    """
    w = weights(unit_family)
    units = sorted(aligned)
    diffs = {u: aligned[u][1] - aligned[u][2] for u in units}
    point = float(sum(w[u] * d.mean() for u, d in diffs.items()))
    seeds = np.random.SeedSequence(seed).spawn(len(units))
    rngs = {u: np.random.default_rng(s) for u, s in zip(units, seeds)}
    draws = np.empty(B, dtype=np.float64)
    h = hashlib.sha256(f"B={B};seed={seed};chunk={chunk};units={units}".encode())
    done = 0
    while done < B:
        m = min(chunk, B - done)
        acc = np.zeros(m, dtype=np.float64)
        for u in units:
            n = diffs[u].size
            idx = rngs[u].integers(0, n, size=(m, n), dtype=np.int64)
            h.update(idx[:1].tobytes())            # first row of every block, cheap and pinning
            acc += w[u] * diffs[u][idx].mean(axis=1)
        draws[done:done + m] = acc
        done += m
    lower = float(np.quantile(draws, quantile, method=method))
    return {
        "delta_raw": point,
        "lower_bound_raw": lower,
        "distance_raw": point - lower,
        "quantile": quantile, "quantile_method": method, "B": int(B), "seed": int(seed),
        "chunk": int(chunk),
        "plan_sha256": h.hexdigest(),
        "draws_sha256": hashlib.sha256(np.ascontiguousarray(draws).tobytes()).hexdigest(),
        "draws_sd": float(draws.std(ddof=1)),
        "per_unit_delta_raw": {u: float(d.mean()) for u, d in diffs.items()},
        "n_by_unit": {u: int(d.size) for u, d in diffs.items()},
        "weights": w,
        "_note": "distance_raw = delta_raw - lower_bound_raw; a contrast resolves when "
                 "delta_raw >= MDE and lower_bound_raw > 0 (sign stability checked elsewhere)",
    }
