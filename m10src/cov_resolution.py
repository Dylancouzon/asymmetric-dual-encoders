"""The COV resolution number — the surface's power disclosure (§Surfaces, M10.0-d).

Runs the contrast rule of §Screen (family-weighted COV macro, paired stratified bootstrap over
queries within unit, B = 200,000, seed 0, `inverted_cdf`, one-sided 0.025/13) on two models that
are candidates in no M10 family, and records the DISTANCE between the point estimate and the
lower bound. It is a power quantity, not a selection: **the sign is discarded here**, so the
artifact cannot say which probe led, and nothing downstream can read a direction out of it.

MDE 0.0056 and alpha are fixed by §Screen and are NOT touched by this number (amendment A4's
sizing was struck by the Codex pass). The number is reported beside every contrast so a reader
can tell an unresolved verdict from an invisible one.
"""
import json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "m10src"))

import numpy as np
import cov_macro

SRC = REPO / "work" / "m10cov" / "probe_scores.json"
OUT = REPO / "results" / "m10_cov_resolution.json"
MDE = 0.0056
B, SEED, QUANTILE = 200_000, 0, 0.025 / 13


def main():
    d = json.loads(SRC.read_text())
    per, units = d["per_query"], d["units"]
    uf = {u: v["family"] for u, v in units.items()}
    keys = sorted(per)
    if len(keys) != 2:
        raise SystemExit(f"expected exactly two probes, got {keys}")
    a, b = (per[k] for k in keys)
    aligned = cov_macro.align(a, b, uf)
    r = cov_macro.contrast(aligned, uf, B=B, seed=SEED, quantile=QUANTILE)
    ma, fa, _ = cov_macro.macro(a, uf)
    mb, fb, _ = cov_macro.macro(b, uf)
    dist = r["distance_raw"]
    # every per-unit and per-family quantity is absolutised: the sign is the one thing the
    # mandate forbids recording, and a signed per-unit table reconstructs it immediately.
    rec = {
        "what": "COV resolution number (power disclosure only; sign discarded by design)",
        "surface": {u: dict(family=v["family"], n_queries=v["n_queries"], n_docs=v["n_docs"])
                    for u, v in sorted(units.items())},
        "n_families": len(set(uf.values())), "n_units": len(uf),
        "n_queries_total": int(sum(v["n_queries"] for v in units.values())),
        "weights": r["weights"],
        "abs_macro_delta": abs(r["delta_raw"]),
        "resolution_distance": dist,
        "MDE_registered": MDE,
        "mde_below_resolution": bool(MDE < dist),
        "macro_level_mean_of_probes": (ma + mb) / 2,
        "abs_family_delta": {f: abs(fa[f] - fb[f]) for f in sorted(fa)},
        "abs_unit_delta": {u: abs(v) for u, v in sorted(r["per_unit_delta_raw"].items())},
        "n_by_unit": r["n_by_unit"],
        "bootstrap": {k: r[k] for k in ("B", "seed", "chunk", "quantile", "quantile_method",
                                        "plan_sha256", "draws_sha256", "draws_sd")},
        "implied_paired_se": r["draws_sd"],
        "z_implied": dist / r["draws_sd"] if r["draws_sd"] else None,
        "interpretation":
            "A contrast on this surface resolves only if its point estimate reaches "
            f"{dist:.4f} (and >= the MDE {MDE}). The registered MDE is "
            f"{'BELOW' if MDE < dist else 'AT OR ABOVE'} that, so a contrast landing at the MDE "
            f"{'cannot' if MDE < dist else 'can'} resolve. Measured between unrelated models, so "
            "it over-estimates the width of a same-init contrast; recorded, not corrected.",
    }
    OUT.write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: rec[k] for k in ("n_families", "n_units", "n_queries_total",
                                          "abs_macro_delta", "resolution_distance",
                                          "implied_paired_se", "mde_below_resolution")}, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
