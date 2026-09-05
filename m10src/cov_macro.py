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

# The admitted COV surface, locked (`m10/LEDGER.md` §2). This is the M9 `_assert_six` lesson
# transplanted: checking only that observed labels are *known* lets a dropped unit or a dropped
# family renormalise silently -- a three-family macro, or a five-slice BRIGHT, would still report
# as a clean number. A Codex pass caught exactly that hole here (2026-09-05).
SURFACE = {
    "MedicalQARetrieval": "consumer-health",
    "BRIGHT/biology": "BRIGHT", "BRIGHT/earth_science": "BRIGHT",
    "BRIGHT/economics": "BRIGHT", "BRIGHT/psychology": "BRIGHT",
    "BRIGHT/robotics": "BRIGHT", "BRIGHT/sustainable_living": "BRIGHT",
    "LegalBenchCorporateLobbying": "legal", "LegalBenchConsumerContractsQA": "legal",
    "LEDGER": "finance",
}


def assert_surface(unit_family):
    """The macro is defined on exactly the admitted surface, or not at all."""
    if dict(unit_family) != SURFACE:
        missing = sorted(set(SURFACE) - set(unit_family))
        extra = sorted(set(unit_family) - set(SURFACE))
        relabelled = sorted(u for u in set(SURFACE) & set(unit_family)
                            if unit_family[u] != SURFACE[u])
        raise ValueError(f"COV surface does not match the admitted lock: missing={missing} "
                         f"extra={extra} relabelled={relabelled}")


def weights(unit_family):
    """{unit: family} -> {unit: weight}. Families equal, units equal within family.

    No bypass flag: an escape hatch on a surface lock is the lock's own defect.
    """
    assert_surface(unit_family)
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
            # First row of every block only: hashing all B x n indices is 21 GB of sha256 for
            # a plan whose reproducibility the draws digest already establishes. Named for what
            # it is -- a SAMPLE of the plan, not the plan (Codex 2026-09-05).
            h.update(idx[:1].tobytes())
            acc += w[u] * diffs[u][idx].mean(axis=1)
        draws[done:done + m] = acc
        done += m
    lower = float(np.quantile(draws, quantile, method=method))
    # §Screen names BOTH `inverted_cdf` and "the 384th order statistic", and they differ by one
    # observation: `inverted_cdf` at B = 200,000 takes 0-based index 384, i.e. the 385th. The
    # method name is the operative constant and is what decides; the neighbouring order statistic
    # is returned so the size of the discrepancy is on the record (Codex 2026-09-05).
    i0 = int(np.ceil(quantile * B)) - 1
    prev = float(np.partition(draws, i0 - 1)[i0 - 1]) if i0 >= 1 else lower
    return {
        "delta_raw": point,
        "lower_bound_raw": lower,
        "distance_raw": point - lower,
        "lower_bound_prev_order_statistic": prev,
        "order_statistic_index0": i0,
        "quantile": quantile, "quantile_method": method, "B": int(B), "seed": int(seed),
        "chunk": int(chunk),
        "plan_sample_sha256": h.hexdigest(),
        "draws_sha256": hashlib.sha256(np.ascontiguousarray(draws).tobytes()).hexdigest(),
        "draws_sd": float(draws.std(ddof=1)),
        "per_unit_delta_raw": {u: float(d.mean()) for u, d in diffs.items()},
        "n_by_unit": {u: int(d.size) for u, d in diffs.items()},
        "weights": w,
        "_note": "distance_raw = delta_raw - lower_bound_raw; a contrast resolves when "
                 "delta_raw >= MDE and lower_bound_raw > 0 (sign stability checked elsewhere)",
    }
