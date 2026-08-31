"""M9.4 final-run statistics — the gate the release decision actually reads.

Why this module exists at all. `m7src/boot.py` is the hardened M7 machinery and M9 reuses it for
the sign-flip, Holm, and strata. But the mandate's bootstrap gate is the **0.0125 quantile**, and
`boot.py` exposes no such field: `paired_dep` computes only the [2.5, 97.5] percentiles and returns
`one_sided_lower_2.5` **rounded to 4 decimals**. A scorer reaching for the nearest-looking field
would decide release on the wrong tail at reduced precision -- the same class of defect a review
caught in M7's `final_run.py`. So the bootstrap is reimplemented here to return the full-precision
draw vector and one explicitly named decision field, and nothing else may decide.

Constants come from `m9/registry.json -> final_run`. Prose is not authoritative; the registry is.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from m7src import boot                                    # noqa: E402

REPO = Path(__file__).resolve().parents[1]
REGISTRY = REPO / "m9" / "registry.json"

SIX = ("scifact", "nfcorpus", "fiqa", "arguana", "scidocs", "trec-covid")


def cfg():
    """The registry is the single source of decision constants (BLOCKER 5)."""
    return json.loads(REGISTRY.read_text())["final_run"]


def _assert_six(aligned):
    """A five-dataset macro must be IMPOSSIBLE, not merely unlikely.

    `boot.paired_dep` defaults to strict=False and divides by `len(aligned)`, so a dataset absent
    from BOTH inputs silently yields a 1/5 macro that still reports as a pass. `strict=True` only
    requires the two inputs to agree with each other, so this check is separate and additional
    (MAJOR 4).
    """
    got = tuple(sorted(aligned))
    if got != tuple(sorted(SIX)):
        raise ValueError(f"final run requires exactly the six named datasets; got {got}")
    for ds, (qids, x, y) in aligned.items():
        if not (len(qids) == len(x) == len(y)) or len(qids) == 0:
            raise ValueError(f"{ds}: ragged or empty aligned arrays")


def align(a, b):
    """{dataset: {qid: score}} x2 -> {dataset: (qids, x, y)}, via boot's hardened aligner.

    `strict=True` refuses any silent shrinkage of datasets or queries; `_assert_six` then refuses
    anything that is not exactly the six. Reusing boot's aligner rather than re-deriving it keeps
    one implementation of the pairing that every committed number went through.
    """
    aligned = boot._align_ids(a, b, strict=True)
    _assert_six(aligned)
    return aligned


def draw_plan(aligned, B, seed):
    """ONE frozen resample plan, generated once and reused byte-identically by C1 and C2.

    The mandate requires shared resample indices across the two contrasts. `boot.paired_dep`
    cannot express that -- its `share=True` shares draws across COMPONENTS within one contrast,
    not across contrasts -- so the plan is materialized here and its digest serialized, making the
    sharing checkable after the fact rather than merely asserted (MAJOR 5).

    With the six datasets and `boot.unit_key` no query is shared between components, so each
    dataset is its own stratum and stratified resampling reduces exactly to the mandate's
    "resample n_d queries with replacement within each dataset".
    """
    rng = np.random.default_rng(seed)
    plan, h = {}, hashlib.sha256()
    h.update(f"B={B};seed={seed}".encode())
    for ds in sorted(aligned):                       # sorted: the order fixes the RNG stream
        n = len(aligned[ds][0])
        idx = rng.integers(0, n, size=(B, n), dtype=np.int64)
        plan[ds] = idx
        h.update(ds.encode())
        h.update(idx.tobytes())
    return plan, h.hexdigest()


def bootstrap(aligned, plan, quantile):
    """Equal-weight macro of per-query differences, with the FULL draw vector retained.

    The decision field is `lower_q0125_raw`: the only number permitted to decide, never rounded.
    """
    k = len(aligned)
    if k != 6:
        raise ValueError(f"expected 6 datasets, got {k}")
    diffs = {ds: (x - y) for ds, (_, x, y) in aligned.items()}
    if set(plan) != set(diffs):
        raise ValueError("draw plan does not cover exactly the aligned datasets")
    for ds, d in diffs.items():
        if plan[ds].shape[1] != d.size:
            raise ValueError(f"{ds}: plan width {plan[ds].shape[1]} != n {d.size}")
    B = next(iter(plan.values())).shape[0]
    draws = np.zeros(B, dtype=np.float64)
    for ds, d in diffs.items():
        draws += d[plan[ds]].mean(axis=1)             # (B, n_d) -> (B,)
    draws /= k
    point = sum(float(d.mean()) for d in diffs.values()) / k
    return {
        "delta_raw": point,
        "lower_q0125_raw": float(np.quantile(draws, quantile, method="linear")),   # THE gate
        "quantile": quantile,
        "quantile_method": "linear",
        "B": int(B),
        "draws_sha256": hashlib.sha256(np.ascontiguousarray(draws).tobytes()).hexdigest(),
        "ci95_raw_reporting_only": [float(np.quantile(draws, 0.025, method="linear")),
                                    float(np.quantile(draws, 0.975, method="linear"))],
        "per_dataset_delta_raw": {ds: float(d.mean()) for ds, d in diffs.items()},
        "n_by_dataset": {ds: int(d.size) for ds, d in diffs.items()},
        "_gate_note": "lower_q0125_raw is the ONLY field that decides; ci95_raw is a 2.5% "
                      "endpoint and is reporting-only",
    }


def run_contrasts(rows, c1, c2, conf=None):
    """The two registered contrasts, sharing one draw plan and one sign-plan seed.

    `rows` maps system -> {dataset: {qid: score}}. Returns the decision record; it writes no
    verdict text and takes no action -- the caller applies the locked claim table.
    """
    conf = conf or cfg()
    q = conf["bootstrap"].get("quantile", 0.0125)
    Bb, sb = conf["bootstrap"]["B"], conf["bootstrap"]["seed"]
    Rs, ss = conf["signflip"]["B"], conf["signflip"]["seed"]
    alpha = conf["holm"]["alpha_family"]
    pairs = {"C1": c1, "C2": c2}

    al = {name: align(rows[a], rows[b]) for name, (a, b) in pairs.items()}
    shapes = {name: {ds: len(v[0]) for ds, v in a.items()} for name, a in al.items()}
    if shapes["C1"] != shapes["C2"]:
        raise ValueError(f"C1 and C2 differ in shape; a shared plan would be invalid: {shapes}")
    qids = {name: {ds: v[0] for ds, v in a.items()} for name, a in al.items()}
    if qids["C1"] != qids["C2"]:
        raise ValueError("C1 and C2 are aligned on different qids; sharing a plan is invalid")

    plan, plan_digest = draw_plan(al["C1"], Bb, sb)

    out = {"plan_sha256": plan_digest, "alpha_family": alpha,
           "quantile": q, "contrasts": {}}
    pvals = {}
    for name, (a, b) in pairs.items():
        boots = bootstrap(al[name], plan, q)
        sf = boot.signflip_dep(rows[a], rows[b], R=Rs, seed=ss, alternative="greater",
                               strict=True, unit_of=boot.unit_key)
        pvals[name] = sf["p"]
        out["contrasts"][name] = {"a": a, "b": b, "bootstrap": boots, "signflip": sf}

    hol = boot.holm(pvals, alpha=alpha)               # dict in, dict out
    for name in pairs:
        c = out["contrasts"][name]
        boot_ok = c["bootstrap"]["lower_q0125_raw"] > 0
        holm_ok = bool(hol[name]["reject"])
        c["bootstrap_rejects"] = bool(boot_ok)
        c["holm_rejects"] = holm_ok
        # BOTH must reject. The sign-flip is a required sensitivity conjunct, not evidence that
        # its weak-null assumptions hold.
        c["passes"] = bool(boot_ok and holm_ok)
    out["holm"] = hol
    return out
