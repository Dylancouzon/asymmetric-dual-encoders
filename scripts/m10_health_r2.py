"""`health` prompt revision 2 (the LAST of the two registered revisions): its 200-query smoke and
its 2,000-query diversity pilot.

Two measurements, on two populations, deliberately:
  * the **smoke** re-runs decision 15's registered 200-query gate on the SAME 40 `hotpotqa-corpus`
    seeds r1 was approved on, so the contract rate and the judge sample are comparable to r1's;
  * the **pilot** runs on the BUILD population (`wikipedia-body`, non-held-out), so the A8 curve
    is the one the 143,000 will be drawn from -- the same instrument the six approved forms got.

This script does NOT judge on-form and does NOT start generation. It writes the judge sample; the
lead runs the independent judge and opens the veto window.
"""
import json, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "m10src"))
OUT = REPO / "work" / "m10gen" / "smoke"

import m10_gen_build as B
import forms, gen, qfilter, corpus10, smoke

N_SMOKE_SEEDS, N_PER_SEED, JUDGE_N = 40, 5, 50

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    gen.health(base=B.BASE)
    # --- the registered 200-query smoke, on r1's own 40 seeds
    blob = json.loads((REPO / "work/m10gen/seeds/seeds-be3fb88f10f18d8c.json").read_text())
    sseeds = [tuple(x) for x in blob["seeds"]["health"]][:N_SMOKE_SEEDS]
    g = gen.generate("health", sseeds, n=N_PER_SEED, base=B.BASE, workers=64, label="health r2 smoke")
    sample = smoke.judge_sample(g["queries"], JUDGE_N)
    kept, qrep = qfilter.filter_queries(g["queries"])
    smoke_rep = {k: v for k, v in g.items() if k != "queries"}
    smoke_rep.update({"queries": [q["query"] for q in g["queries"]],
                      "judge_sample": sample, "rubric_range_filter": qrep.get("health"),
                      "seeds": "hotpotqa-corpus, the same 40 seeds r1 was approved on",
                      "contract_gate": 0.90, "PASS_contract": g["contract_rate"] >= 0.90})
    # --- the build-population pilot (wikipedia-body, non-held-out)
    pilot, texts = B.pilot_form("health")
    out = {"_what": "health prompt revision 2 of 2 -- smoke (contract) and diversity pilot",
           "when": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "prompt_sha256": B.prompt_sha256("health"),
           "prompt_hash8_registered": B.prompt_hash8("health"),
           "prompt_r2": forms.FORMS["health"].format(n="N"),
           "rubric_frozen": forms.RUBRIC["health"].format(n="N"),
           "revision": "2 of 2 (the last)",
           "smoke": smoke_rep, "pilot": pilot,
           "_not_done_here": "on-form judging and the veto window are the lead's; generation is "
                             "NOT started"}
    (OUT / "health_r2.json").write_text(json.dumps(out, indent=1))
    (OUT / "judge_input_health_r2.json").write_text(json.dumps(
        {"health": {"registered_description": forms.RUBRIC["health"].format(n="N"),
                    "sample": sample}}, indent=1))
    print(json.dumps({"contract": g["contract_rate"], "PASS_contract": smoke_rep["PASS_contract"],
                      "a8_curve": {k: v["near_dup_rate"] for k, v in pilot["a8_amended_curve"].items()},
                      "top10_opener": pilot["top10_opening_4gram_share"],
                      "distinct_2": pilot["distinct_2"]}, indent=1), flush=True)
