"""What each M10 conjunct demands PER DATASET, from the frozen comparator rows (2026-09-04 review).

Comparator-only read: `results/perquery.json` (bge-small, leaf-ir-asym) and the stella symmetric
ceiling from `results/m7_final_run.json` (`six.teacher-symmetric`), both observed in M7. No nano
number exists. Three quantities a reader of the plan needs and the mandate did not state:

1. the retention of the ceiling nano needs on EACH dataset merely to equal each comparator;
2. the UNIFORM retention that reaches each registered pass point (`results/m10_bars.json`);
3. how much of nano's avg-6 margin over bge-small would come from fiqa, a disclosed stella
   training set, at a given uniform retention -- the reason clean-4 is the headline.

Writes results/m10_conjunct_arithmetic.json. Read by no rule.
"""
import json, pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent
ALL6 = ("scifact", "nfcorpus", "fiqa", "arguana", "scidocs", "trec-covid")
CLEAN4 = ("nfcorpus", "scidocs", "scifact", "trec-covid")


def mean(xs):
    return sum(xs) / len(xs)


def main():
    pq = json.loads((REPO / "results/perquery.json").read_text())["datasets"]
    six = json.loads((REPO / "results/m7_final_run.json").read_text())["six"]
    bars = json.loads((REPO / "results/m10_bars.json").read_text())
    ceil = {s: mean(list(six["teacher-symmetric"][s].values())) for s in ALL6}
    bge = {s: mean(pq[s]["systems"]["bge-small-en-v1.5"]) for s in ALL6}
    leaf = {s: mean(pq[s]["systems"]["leaf-ir-asym"]) for s in ALL6}
    per_ds = {s: {"ceiling": round(ceil[s], 4), "bge_small": round(bge[s], 4),
                  "leaf_ir_asym": round(leaf[s], 4),
                  "retention_to_equal_bge": round(bge[s] / ceil[s], 4),
                  "retention_to_equal_leaf": round(leaf[s] / ceil[s], 4)} for s in ALL6}
    cm = {"all6": mean([ceil[s] for s in ALL6]), "clean4": mean([ceil[s] for s in CLEAN4])}
    pp = bars["pass_points"]
    uniform = {k: round(v["nano_must_reach"] / cm["all6" if k.endswith("avg6") else "clean4"], 4)
               for k, v in pp.items()}
    scen = {}
    for r in (0.90, 0.92, 0.94, 0.96):
        a6 = r * cm["all6"]; c4 = r * cm["clean4"]
        margin = {s: round(r * ceil[s] - bge[s], 4) for s in ALL6}
        scen[f"{r:.2f}"] = {
            "avg6": round(a6, 4), "clean4": round(c4, 4),
            "passes": [k for k, v in pp.items() if (a6 if k.endswith("avg6") else c4) >= v["nano_must_reach"]],
            "nano_minus_bge_per_dataset": margin,
            "fiqa_share_of_avg6_margin": round((margin["fiqa"] / 6) / max(a6 - mean([bge[s] for s in ALL6]), 1e-9), 3)}
    out = {"_what": __doc__.strip(), "date": "2026-09-04",
           "ceiling_macro": {k: round(v, 4) for k, v in cm.items()},
           "per_dataset": per_ds,
           "uniform_retention_reaching_pass_point": uniform,
           "uniform_retention_scenarios": scen}
    (REPO / "results/m10_conjunct_arithmetic.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "_what"}, indent=1))


if __name__ == "__main__":
    main()
