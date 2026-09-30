"""E5: per-query oracle headroom between Zero and Nano on the 12 non-reserved BEIR-15 datasets.

Reads committed M20 per-query rows only. Method: `m15/MEASUREMENTS.md`, E5.
"""
import numpy as np

from common import EVAL12, REPO, m20_rows, receipt, utc_now, write_result

OUT = REPO / "results" / "m15_e5_oracle.json"
FRACTIONS = [round(0.05 * i, 2) for i in range(21)]


def aligned(dataset):
    rows, inputs = {}, []
    for system in ("zero-dense", "nano-dense", "stella-query"):
        scores, meta = m20_rows(dataset, system)
        rows[system] = scores
        inputs.append(meta)
    qids = sorted(rows["zero-dense"])
    if any(sorted(r) != qids for r in rows.values()):
        raise RuntimeError(f"{dataset}: query sets differ across systems")
    arr = {s: np.array([rows[s][q] for q in qids]) for s in rows}
    return qids, arr, inputs


def frontier(zero, nano):
    """Best mean when Nano serves a fraction b of queries, taking the largest gains first."""
    gain = np.sort(nano - zero)[::-1]
    out = []
    for b in FRACTIONS:
        k = int(round(b * len(zero)))
        out.append(float((zero.sum() + gain[:k].sum()) / len(zero)))
    return out


def summarize(zero, nano, stella):
    oracle = np.maximum(zero, nano)
    return {"n_queries": int(len(zero)), "zero": float(zero.mean()), "nano": float(nano.mean()),
            "oracle": float(oracle.mean()), "stella_query": float(stella.mean()),
            "share_zero_above_nano": float((zero > nano).mean()),
            "share_equal": float((zero == nano).mean()),
            "share_zero_below_nano": float((zero < nano).mean()),
            "min_nano_fraction_at_oracle_max": float((nano > zero).mean()),
            "share_zero_at_or_above_stella": float((zero >= stella).mean()),
            "frontier": frontier(zero, nano)}


def build():
    per, inputs = {}, []
    for dataset in EVAL12:
        _, arr, meta = aligned(dataset)
        per[dataset] = summarize(arr["zero-dense"], arr["nano-dense"], arr["stella-query"])
        inputs += meta
    keys = ("zero", "nano", "oracle", "stella_query", "share_zero_above_nano", "share_equal",
            "share_zero_below_nano", "min_nano_fraction_at_oracle_max",
            "share_zero_at_or_above_stella")
    macro = {k: float(np.mean([per[d][k] for d in EVAL12])) for k in keys}
    macro["frontier"] = [float(np.mean([per[d]["frontier"][i] for d in EVAL12]))
                         for i in range(len(FRACTIONS))]
    return per, macro, inputs


def selfcheck():
    z, n = np.array([0.0, 1.0, 0.5]), np.array([1.0, 0.0, 0.5])
    f = frontier(z, n)
    assert f[0] == z.mean() and abs(max(f) - np.maximum(z, n).mean()) < 1e-12
    assert f[-1] == n.mean()


if __name__ == "__main__":
    selfcheck()
    started = utc_now()
    per, macro, inputs = build()
    write_result(OUT, {"status": "COMPLETE", "measurement": "E5",
                       "scope": "descriptive; an upper bound on per-query routing, not a router",
                       "datasets": list(EVAL12), "fractions": FRACTIONS,
                       "macro_over_12": macro, "per_dataset": per,
                       "receipt": receipt(__file__, inputs, started)})
