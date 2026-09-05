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
# `m10/LEDGER.md` §2, the admitted surface's per-unit query counts. BRIGHT's slice sizes are the
# published ones; MedicalQA, the two LegalBench components and LEDGER are §2's rows.
ADMITTED_N = {"MedicalQARetrieval": 2048, "BRIGHT/biology": 103, "BRIGHT/earth_science": 116,
              "BRIGHT/economics": 103, "BRIGHT/psychology": 101, "BRIGHT/robotics": 101,
              "BRIGHT/sustainable_living": 108, "LegalBenchCorporateLobbying": 340,
              "LegalBenchConsumerContractsQA": 396, "LEDGER": 10000}


def main():
    d = json.loads(SRC.read_text())
    per, units = d["per_query"], d["units"]
    uf = {u: v["family"] for u, v in units.items()}
    keys = sorted(per)
    if len(keys) != 2:
        raise SystemExit(f"expected exactly two probes, got {keys}")
    a, b = (per[k] for k in keys)
    # ORIENTATION, and it is a correctness fix, not a cosmetic one (Codex 2026-09-05).
    # `distance = point - lower` is orientation-sensitive on a skewed draw distribution, and the
    # draws digest of a signed vector is a two-candidate direction oracle for anyone who can
    # recompute both orderings. Orienting on the sign of the point estimate makes every recorded
    # quantity -- distance, digest, draws SD -- identical under either ordering, so the artifact
    # carries no direction at all, and it is also the orientation a screen contrast has (an arm
    # difference is read winner-minus-default).
    if cov_macro.macro(a, uf)[0] < cov_macro.macro(b, uf)[0]:
        a, b = b, a
    aligned = cov_macro.align(a, b, uf)
    r = cov_macro.contrast(aligned, uf, B=B, seed=SEED, quantile=QUANTILE)
    if r["delta_raw"] <= 0:
        raise SystemExit("orientation failed: the macro difference is not positive")
    # The scored query counts must match the admitted surface (`m10/LEDGER.md` §2), or a unit
    # that silently lost queries would reweight nothing and misreport everything.
    for u, n in ADMITTED_N.items():
        got = r["n_by_unit"][u]
        if got != n:
            raise SystemExit(f"{u}: scored {got} queries, §2 admits {n}")
    # Where the width comes from, analytically, from the same paired diffs: w_u^2 * Var_u / n_u.
    # Direction-free (a variance carries no sign) and decision-free -- it says which family the
    # surface's power is lost in, which is the whole content of W5.
    w = r["weights"]
    vc = {}
    for u, (_q, x, y) in aligned.items():
        d = x - y
        vc.setdefault(uf[u], 0.0)
        vc[uf[u]] += w[u] ** 2 * float(np.var(d, ddof=1)) / d.size
    tot = sum(vc.values())
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
                                        "plan_sample_sha256", "draws_sha256", "draws_sd")},
        # §Screen names both `inverted_cdf` and "the 384th order statistic". They differ by one
        # observation: `inverted_cdf` at B = 200,000 takes 0-based index 384, the 385th. The
        # method name is the operative constant and is what runs; the alternative reading is
        # computed here so the size of the discrepancy is on the record, not argued about.
        "order_statistic_disclosure": {
            "inverted_cdf_index0": r["order_statistic_index0"],
            "prose_384th_index0": r["order_statistic_index0"] - 1,
            "distance_under_prose_reading":
                r["delta_raw"] - r["lower_bound_prev_order_statistic"]},
        "implied_paired_se": r["draws_sd"],
        "analytic_paired_se": tot ** 0.5,
        "variance_share_by_family": {f: v / tot for f, v in sorted(vc.items())},
        # The mandate records that this over-estimates a same-init contrast's width and is not
        # corrected. It does not say by how much, so the scaling is given as arithmetic: two arms
        # of one screen share an init and score the same queries, so their paired per-query
        # differences are far more correlated than two unrelated models' are, and the distance
        # falls with the paired SD. This is a sensitivity row, NOT a claim about any contrast.
        "distance_if_paired_sd_scaled": {str(c): dist * c for c in (1.0, 0.75, 0.5, 0.25)},
        "z_implied": dist / r["draws_sd"] if r["draws_sd"] else None,
        "interpretation":
            "The rule resolves a contrast when its point estimate is >= the MDE AND the lower "
            f"bound is > 0; since lower = point - distance, that needs point > {dist:.4f} "
            f"STRICTLY, as well as >= {MDE}. The registered MDE is "
            f"{'BELOW' if MDE < dist else 'AT OR ABOVE'} the distance, so a contrast landing at "
            f"the MDE {'cannot' if MDE < dist else 'can'} resolve. The mandate records the "
            "EXPECTATION that a distance measured between unrelated models over-estimates a "
            "same-init contrast's width; that is an expectation about correlated per-query "
            "differences, not a guarantee -- paired variance can move either way -- and it is "
            "recorded, not corrected. The sensitivity row prices it as arithmetic.",
    }
    OUT.write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: rec[k] for k in ("n_families", "n_units", "n_queries_total",
                                          "abs_macro_delta", "resolution_distance",
                                          "implied_paired_se", "mde_below_resolution")}, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
