"""What the studio's auto router plans as traffic grows (exploratory planner output for the talk).

Runs allocator.plan() from m15/talk/demo/studio with the four studio search presets. Quality and search time per
preset come from the committed NQ Edge sweep; encode capacity is the M5 Pro figure the allocator self-check uses
(Stella 80, Nano 1,400, Zero 17,000 encodes/s). Floor 0.50 and headroom 0.8 are the studio defaults. This is the
planner's decision rule on fixed costs, not a load measurement.

Run: .venv/bin/python m15src/talk_router_plan.py
"""
import json
import sys

from common import REPO, receipt, sha_file, utc_now, write_result

sys.path.insert(0, str(REPO / "m15/talk/demo/studio"))
import allocator as A  # noqa: E402

SWEEP = REPO / "results/m15_talk_edge_nq.json"
OUT = REPO / "results/m15_talk_router_plan.json"
PRESETS = ((128, 4.0), (256, 4.0), (512, 4.0), (1024, 8.0))
ENCODE_PER_S = {"stella": 80, "nano": 1400, "zero": 17000}
RATES = (40, 60, 100, 150, 200, 300, 450, 600, 800, 1000, 1200, 1500, 1800, 2000, 2250, 2500)


def main():
    started = utc_now()
    r = json.loads(SWEEP.read_text())["results"]
    pick = lambda n, ef, os_: next(s for s in r[n]["settings"] if s["ef"] == ef and s["oversampling"] == os_)
    presets = {k: [A.EncoderCost(n, pick(n, *k)["ndcg10"], ENCODE_PER_S[n], pick(n, *k)["search_ms_p50"])
                   for n in ("stella", "nano", "zero")] for k in PRESETS}
    plans = []
    for rate in RATES:
        p = A.plan(rate, presets, floor=0.50, headroom=0.8)
        plans.append({"rate": rate, "ef": p.preset[0], "oversampling": p.preset[1], "shares": p.shares,
                      "quality": p.quality, "status": p.status})
        print(rate, p.preset, {k: round(v, 2) for k, v in p.shares.items()}, round(p.quality, 3), p.status)
    write_result(OUT, {"status": "COMPLETE", "analysis": "exploratory (planner output on fixed costs)", "dataset": "nq",
                       "floor": 0.50, "headroom": 0.8, "encode_per_s": ENCODE_PER_S, "plans": plans,
                       "receipt": receipt(__file__, {"edge_nq": {"path": str(SWEEP.relative_to(REPO)),
                                                                 "sha256": sha_file(SWEEP)}}, started, ("numpy",))})


if __name__ == "__main__":
    main()
