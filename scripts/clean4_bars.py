"""M10's registered bars AND pass points on both partitions, from the frozen comparator vectors.

Amendment A3 registers C1/C2 on clean-4 as well as avg-6, so the clean-4 bars are numbers a
rule reads and must be reproducible. Comparator rows only: `results/perquery.json` holds
per-query nDCG@10 for nine frozen systems, none of them nano.

It also reports a *planning proxy* for each conjunct — the bar plus the width between the point
estimate and the one-sided bootstrap lower bound at the REGISTERED quantile — because the Fable
review of 2026-09-04 showed clean-4's width is ~36% larger than avg-6's (trec-covid contributes
25% of the clean-4 macro on 50 queries). The quantile is 0.025: fixed-sequence gatekeeping tests
each conjunct at the full one-sided 0.025 (amendment A3/B2); the Codex pass of 2026-09-04 caught
this script still using M9's Holm-2 quantile 0.0125. The width is estimated on the leaf-vs-bge
comparator pair, so these are PROXIES: nano's own interval depends on its per-query differences
and will differ. Nothing here is a pass/fail rule; the rule is the final run's own bound.

Writes results/m10_bars.json.
"""
import json, pathlib

import numpy as np

REPO = pathlib.Path(__file__).resolve().parent.parent
ALL6 = ("scifact", "nfcorpus", "fiqa", "arguana", "scidocs", "trec-covid")
CLEAN4 = ("nfcorpus", "scidocs", "scifact", "trec-covid")  # no disclosed teacher overlap
SYSTEMS = ("bge-small-en-v1.5", "leaf-ir-asym", "lr-dense-pertask",
           "opensearch-doc-v3-gte", "bm25")


def macro(datasets, system, sets):
    per_set = [sum(datasets[s]["systems"][system]) / len(datasets[s]["systems"][system])
               for s in sets]
    return sum(per_set) / len(per_set)


def boot_width(datasets, a_sys, b_sys, sets, n_boot=20000, seed=0, q=0.025):
    """-> (point, lower bound, width) for macro(a) - macro(b), paired, stratified per dataset."""
    rng = np.random.default_rng(seed)
    arrs = [(np.array(datasets[s]["systems"][a_sys]), np.array(datasets[s]["systems"][b_sys]))
            for s in sets]
    point = float(np.mean([a.mean() - b.mean() for a, b in arrs]))
    draws = np.empty(n_boot)
    for i in range(n_boot):
        draws[i] = np.mean([float((a[idx] - b[idx]).mean())
                            for a, b in arrs
                            for idx in (rng.integers(0, len(a), len(a)),)])
    lb = float(np.quantile(draws, q))
    return point, lb, point - lb


def main():
    d = json.loads((REPO / "results/perquery.json").read_text())["datasets"]
    assert tuple(sorted(d)) == tuple(sorted(ALL6)), sorted(d)
    rows = {s: {"all6": round(macro(d, s, ALL6), 4), "clean4": round(macro(d, s, CLEAN4), 4)}
            for s in SYSTEMS}
    widths, ceiling = {}, {"all6": 0.5744, "clean4": 0.5640}
    for name, sets in (("all6", ALL6), ("clean4", CLEAN4)):
        p, lb, w = boot_width(d, "leaf-ir-asym", "bge-small-en-v1.5", sets)
        widths[name] = {"proxy_point": round(p, 4), "proxy_lower_bound": round(lb, 4),
                        "width": round(w, 4),
                        "n_queries": {s: len(d[s]["systems"]["bm25"]) for s in sets}}
    out = {
        "_what": "M10 bars on both registered partitions, from the frozen comparator vectors "
                 "(results/perquery.json, per-query nDCG@10). Comparator rows only.",
        "date": "2026-09-04",
        "clean4": list(CLEAN4),
        "all6": list(ALL6),
        "rows": rows,
        "registered": {
            "C1a_release_avg6": rows["bge-small-en-v1.5"]["all6"],
            "C1b_release_clean4": rows["bge-small-en-v1.5"]["clean4"],
            "C2a_aim_avg6": rows["leaf-ir-asym"]["all6"],
            "C2b_aim_clean4": rows["leaf-ir-asym"]["clean4"],
        },
        "bootstrap_widths_proxy": widths,
        "planning_proxies": {
            k: {"bar": bar, "width": widths[part]["width"],
                "nano_must_reach": round(bar + widths[part]["width"], 4),
                "retention_of_ceiling": round((bar + widths[part]["width"]) / ceiling[part], 4)}
            for k, bar, part in (
                ("C1a_release_avg6", rows["bge-small-en-v1.5"]["all6"], "all6"),
                ("C1b_release_clean4", rows["bge-small-en-v1.5"]["clean4"], "clean4"),
                ("C2a_aim_avg6", rows["leaf-ir-asym"]["all6"], "all6"),
                ("C2b_aim_clean4", rows["leaf-ir-asym"]["clean4"], "clean4"))
        },
    }
    (REPO / "results/m10_bars.json").write_text(json.dumps(out, indent=1) + "\n")
    for s, r in rows.items():
        print(f"{s:26s} all6 {r['all6']:.4f}  clean4 {r['clean4']:.4f}")
    for k, v in out["planning_proxies"].items():
        print(f"{k:22s} bar {v['bar']:.4f} + width {v['width']:.4f} -> nano must reach "
              f"{v['nano_must_reach']:.4f} = {v['retention_of_ceiling']*100:.1f}% of ceiling")


if __name__ == "__main__":
    main()
