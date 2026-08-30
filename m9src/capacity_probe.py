"""Is the ≤35M student cap what binds retention, or is it the dose?

Authorised by Dylan 2026-08-30, before three days of compute are committed. The anchor reaches
73.2% of the teacher ceiling and the aim needs ~90%; the curve says more epochs on 242,786 queries
asymptotes near 74%, so either (a) the data volume is the wall — fixable with the 6.15M-row
document pool — or (b) 33M parameters cannot represent a 400M teacher's query space well enough,
in which case no amount of compute helps. This probe separates the two by running the identical
recipe and the identical dose on a **109M** student.

**DIAGNOSTIC, and out of M9's scope for selection.** The mandate caps nano at 35M and reserves
anything larger for M10 on Dylan's call. The artifact is written under a `-diag` id, is stamped
decision-ineligible, and lands in `work/m9smoke/` where the decision loader cannot address it.
It answers one question — *is capacity binding?* — and selects nothing.

It registers its student by mutating `nano.STUDENTS` at run time rather than editing `nano.py`,
so it changes no fingerprint scope and cannot invalidate a single trained arm.
"""
import json
import time

import m9base
from m9base import RESULTS

import eval9      # noqa: E402
import guard9     # noqa: E402
import nano       # noqa: E402
import screen     # noqa: E402

RUN_ID = "m9cap-diag"


def spec_and_cfg():
    r = guard9.registry()
    row = dict(next(a for a in r["arms"] if a["id"] == RUN_ID))
    diag = r["models"]["diagnostic_students"][row["student"]]
    nano.STUDENTS[row["student"]] = {"repo": diag["repo"], "revision": diag["revision"],
                                     "params": diag["params_approx"]}
    return row, screen.arm_cfg(row), diag


def run():
    t0 = time.time()
    spec, cfg, diag = spec_and_cfg()
    guard9.begin_run(RUN_ID)
    print(f"=== {RUN_ID}: {json.dumps(spec)} | {json.dumps(cfg)}", flush=True)

    stub = nano.Nano(spec["student"])
    tok, n_params = stub.tok, stub.n_params()
    del stub
    plan, meta = screen.build_plan(spec, cfg, tok)

    def eval_fn(model, step):
        # SCREEN-3 only: this probe answers a capacity question, and DEV-6 would triple its cost
        # for a number no decision reads.
        per = eval9.eval_student(model, spec["teacher"], comps=eval9.components("SCREEN3"))
        return {"macros": eval9.macros(per, spec["teacher"]), "per_component": per}

    rec, model = nano.train_arm(RUN_ID, spec["student"], plan, cfg, eval_fn=eval_fn)
    del model
    anchor = json.loads((RESULTS / "m9_screen_m9s1.json").read_text())
    ceil3 = guard9.registry()["ceilings"]["stella-400M-v5"]["SCREEN3"]
    mine = rec["history"][-1]["macros"]["SCREEN3"]["macro"]
    theirs = anchor["final"]["SCREEN3"]["macro"]
    rec.update({
        "spec": spec, "cfg": cfg, "plan": meta, "n_params": n_params,
        "anchor_params": anchor["n_params"], "final_screen3": mine,
        "anchor_screen3": theirs, "delta_vs_anchor": round(mine - theirs, 5),
        "retention": round(mine / ceil3, 4),
        "anchor_retention": round(theirs / ceil3, 4),
        "reading": ("a LARGE positive delta means the 35M cap is binding and more compute alone "
                    "cannot reach the aim; a delta near zero means capacity is not the wall at "
                    "this dose and the lever is unique text plus compute"),
        "_status": ("DIAGNOSTIC. Out of M9 scope for selection -- the mandate caps nano at 35M. "
                    "No M9 rule may cite this artifact."),
        "seconds_total": round(time.time() - t0, 1)})
    guard9.write_result(RESULTS / "m9_capacity_probe.json", rec, RUN_ID)
    print(f"=== {RUN_ID} DONE  {n_params/1e6:.1f}M student SCREEN-3 {mine:.5f} "
          f"(retention {rec['retention']}) vs anchor {theirs:.5f} "
          f"({rec['anchor_retention']}) -> delta {rec['delta_vs_anchor']:+.5f}", flush=True)
    return rec


if __name__ == "__main__":
    run()
