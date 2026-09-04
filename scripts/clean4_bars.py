"""Recompute M10's registered bars on both partitions from the frozen comparator vectors.

Amendment A3 registers C1/C2 on clean-4 as well as avg-6, so the clean-4 bars are numbers a
rule reads and must be reproducible. Comparator rows only: `results/perquery.json` holds
per-query nDCG@10 for nine frozen systems, none of them nano. Writes results/m10_bars.json.
"""
import json, pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent
ALL6 = ("scifact", "nfcorpus", "fiqa", "arguana", "scidocs", "trec-covid")
CLEAN4 = ("nfcorpus", "scidocs", "scifact", "trec-covid")  # no disclosed teacher overlap
SYSTEMS = ("bge-small-en-v1.5", "leaf-ir-asym", "lr-dense-pertask",
           "opensearch-doc-v3-gte", "bm25")


def macro(datasets, system, sets):
    per_set = [sum(datasets[s]["systems"][system]) / len(datasets[s]["systems"][system])
               for s in sets]
    return sum(per_set) / len(per_set)


def main():
    d = json.loads((REPO / "results/perquery.json").read_text())["datasets"]
    assert tuple(sorted(d)) == tuple(sorted(ALL6)), sorted(d)
    rows = {s: {"all6": round(macro(d, s, ALL6), 4), "clean4": round(macro(d, s, CLEAN4), 4)}
            for s in SYSTEMS}
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
    }
    (REPO / "results/m10_bars.json").write_text(json.dumps(out, indent=1) + "\n")
    for s, r in rows.items():
        print(f"{s:26s} all6 {r['all6']:.4f}  clean4 {r['clean4']:.4f}")


if __name__ == "__main__":
    main()
