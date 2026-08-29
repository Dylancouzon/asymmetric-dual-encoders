"""E14-HEAD's arm runner: one process per arm, four patches installed before the first step.

Registered design (`m8/registry.json["probes"]["E14-HEAD"]`, amended 2026-08-29 after the Codex
review). What this file schedules, and why in this order:

  1. `selftest`  -- `e14_head.self_test()` must pass before any arm. A copied loss that has drifted
                    from its original makes every number a comparison between two objectives, and
                    the drift looks exactly like a result.
  2. `ladder`    -- the head learning rate, chosen on a held-out 2% slice of the TRAINING pairs, on
                    a DISJOINT tuning seed (3), in a process that cannot load a dev corpus at all.
  3. `adequacy`  -- the step-adequacy continuation. Pre-registered, and it GATES THE NULL: if the
                    winning arm has not plateaued by 2500 steps, the primary reports
                    OPTIMIZATION-INADEQUATE, not a method null. A null at an inadequate step budget
                    is evidence about the configuration, not about the method.
  4. `reported`  -- the nine arms that are read: R0N, LIN, MLP x seeds (0, 1, 2), full pair pool.

R0N IS THE COMPARATOR, NOT R0. At `W = 0` the head emits `normalize(d)` while R0 scores the raw
cached fp16 vectors, and those are only approximately unit-norm (0.36% exactly 1 over 100,000 pool
rows, max |norm-1| 4.8e-05). Renormalization shifts Phase-A logits and therefore the training
trajectory, so rescoring R0 is not a repair. R0N is this same patched path with the head frozen at
identity. R0N against the existing R0 arms is reported separately, as an end-to-end null on the
whole patch stack -- a check the design did not previously have.

THE ARMS ARE PAIRED BY SEED against the same Phase-B checkpoint (`p35b-2m`), differing only in the
Phase-A seed, which is what makes the seed-paired read legitimate.

Cost, measured on this box for arms of exactly this shape (`logs/m8_b3_train.log`,
`logs/m8_nf_train.log`): 202-570 s per 2500-step Phase-A arm, the spread dominated by pool load and
the four dev evaluations rather than by the steps. The head adds a 1024x1024 matmul over ~33K
negatives per step; budget accordingly and CHECK THE FIRST RATE LINE rather than trusting this.
"""
import argparse
import json
import subprocess
import sys
import time

import numpy as np

import m8base
import probe_guard

REPO = m8base.REPO
WORK = REPO / "work"
RUNS = WORK / "runs"
SIDE = WORK / "e14_sidecars"

PROBE = "E14-HEAD"
BASE_RUN = "p35w-2m-s2500"      # the artifact whose recipe defines R0's A leg (see noise_floor)
SEEDS = (0, 1, 2)
TUNE_SEED = 3                   # disjoint from every reported seed, so selection is not made on one
HEADS = ("lin", "mlp")          # `lin` PRIMARY, `mlp` its nonlinearity control
LADDER = (("3e4", 3e-4), ("1e3", 1e-3), ("3e3", 3e-3))
HOLDOUT_FRAC = 0.02
ADEQUACY_STEPS = 5000
ADEQUACY_EVAL_EVERY = 1250      # gives the 1250 / 2500 / 5000 readings the plateau rule needs


def base_cfg(run=BASE_RUN):
    """R0's recipe, read from the artifact rather than from a snapshot (m7/CODEMAP.md pitfall 15)."""
    meta = json.loads((RUNS / f"{run}.meta.json").read_text())["cfg"]
    return {k: v for k, v in meta.items() if k != "run_id"}


def _arm(rid, over, *, head_kind, head_lr, trainable, arm_kind, kind_tag, seed):
    return {rid: {**over, "_head_kind": head_kind, "_head_lr": head_lr,
                  "_trainable": trainable, "_arm_kind": arm_kind, "_tag": kind_tag,
                  "_seed": seed, "_exists": (RUNS / f"{rid}.npz").exists()}}


def plan_ladder():
    cfg = base_cfg()
    arms = {}
    for hk in HEADS:
        for tag, lr in LADDER:
            arms.update(_arm(f"m8e14-lad-{hk}-lr{tag}", {**cfg, "seed": TUNE_SEED},
                             head_kind=hk, head_lr=lr, trainable=True, arm_kind="ladder",
                             kind_tag=f"lad-{hk}", seed=TUNE_SEED))
    return arms


def plan_adequacy(winners):
    """The winning lr per head, continued to 5,000 steps and read on the holdout only.

    Implemented as ONE 5,000-step arm read at 1250 / 2500 / 5000 rather than as a restart from the
    2,500-step checkpoint, because `set_lr` parameterizes the warmup/decay by the phase's TOTAL
    step count: a restart would change the schedule half way through the curve and the plateau
    rule would then be reading two different schedules against each other. The rule's arithmetic
    is unchanged; it is applied within a single consistent schedule.
    """
    cfg = base_cfg()
    arms = {}
    for hk, lr in winners.items():
        arms.update(_arm(f"m8e14-step-{hk}",
                         {**cfg, "seed": TUNE_SEED, "steps_a": ADEQUACY_STEPS,
                          "eval_every": ADEQUACY_EVAL_EVERY},
                         head_kind=hk, head_lr=lr, trainable=True, arm_kind="ladder",
                         kind_tag=f"step-{hk}", seed=TUNE_SEED))
    return arms


def plan_reported(winners):
    cfg = base_cfg()
    arms = {}
    for s in SEEDS:
        arms.update(_arm(f"m8e14-r0n-s{s}", {**cfg, "seed": s},
                         head_kind="lin", head_lr=0.0, trainable=False, arm_kind="reported",
                         kind_tag="r0n", seed=s))
    for hk in HEADS:
        for s in SEEDS:
            arms.update(_arm(f"m8e14-{hk}-s{s}", {**cfg, "seed": s},
                             head_kind=hk, head_lr=winners[hk], trainable=True,
                             arm_kind="reported", kind_tag=hk, seed=s))
    return arms


def _code(rid, over, a, sidecar, smoke=False):
    patch_kw = {"head_kind": a["_head_kind"], "head_lr": a["_head_lr"],
                "trainable": a["_trainable"], "arm_kind": a["_arm_kind"],
                "holdout_frac": HOLDOUT_FRAC if a["_arm_kind"] == "ladder" else 0.0,
                "temp": over.get("temp", 0.02)}
    return (
        "import os, sys, json\n"
        "os.environ.setdefault('M7_ENCODER', 'stella-400M-v5')\n"
        "sys.path.insert(0, %r); sys.path.insert(0, %r); sys.path.insert(0, %r)\n"
        # m8base installs the G2 protected-path guard process-wide. The process that does the
        # actual training is the one that must be guarded, not the launcher.
        "import m8base\n"
        "import e14_patch\n"
        "h = e14_patch.install(**%r)\n"
        "import program, sweep\n"
        "d = sweep.one(%r, program.BASE, **%r)\n"
        "if d is None:\n"
        "    print('ARM FAILED (sweep.one returned None); its FAILED row is in m7/RESULTS.md')\n"
        "    sys.exit(2)\n"
        "h.assert_fired()\n"
        "rec = h.persist(%r, %r, extra={'smoke': %r})\n"
        "hist = json.loads(open(%r).read())['history'] if os.path.exists(%r) else []\n"
        "json.dump({'run_id': %r, 'tag': %r, 'seed': %r, 'head_kind': %r, 'head_lr': %r,\n"
        "           'trainable': %r, 'arm_kind': %r, 'dev_proxy': d, 'history': hist,\n"
        "           'provenance': rec}, open(%r, 'w'), indent=1)\n"
        "print('DEVPROXY', %r, d, flush=True)\n"
        "sys.exit(0)\n"
        % (str(REPO / "m7src"), str(REPO / "bench"), str(REPO / "m8src"),
           patch_kw, rid, over, rid, a["_seed"], smoke,
           str(RUNS / f"{rid}.json"), str(RUNS / f"{rid}.json"),
           rid, a["_tag"], a["_seed"], a["_head_kind"], a["_head_lr"],
           a["_trainable"], a["_arm_kind"], str(sidecar), rid))


def run_arms(arms, smoke=False, only=None):
    SIDE.mkdir(parents=True, exist_ok=True)
    todo = [(r, a) for r, a in arms.items()
            if (only is None or r in only) and (smoke or not a["_exists"])]
    if not todo:
        print("every arm already on disk")
        return
    print(f"{len(todo)} arms to run: {[r for r, _ in todo]}", flush=True)
    for rid, a in todo:
        over = {k: v for k, v in a.items() if not k.startswith("_")}
        run_id = rid
        if smoke:
            # A smoke must NOT occupy the real run id (m8/CODEMAP.md pitfall 11): 90-step
            # artifacts under the real id would make the next plan() report the arm as trained.
            over = {**over, "steps_a": 90, "eval_every": 45}
            run_id = rid + "-smoke"
        sidecar = SIDE / f"{run_id}.json"
        code = _code(run_id, over, a, sidecar, smoke=smoke)
        t0 = time.time()
        print(f"[{run_id}] launching (head={a['_head_kind']} lr={a['_head_lr']} "
              f"trainable={a['_trainable']} kind={a['_arm_kind']} seed={a['_seed']} "
              f"steps_a={over.get('steps_a')})", flush=True)
        r = subprocess.run([sys.executable, "-u", "-c", code], cwd=str(REPO))
        print(f"[{run_id}] exit {r.returncode} in {time.time()-t0:.0f}s", flush=True)
        if r.returncode != 0:
            raise SystemExit(f"{run_id} failed (exit {r.returncode}); stopping rather than "
                             f"reading a probe from a partial arm set.")


def holdout_curve(rid):
    """The ladder/adequacy statistic per step, from the run's own history."""
    p = RUNS / f"{rid}.json"
    if not p.exists():
        return None
    hist = json.loads(p.read_text())["history"]
    # `macro` carries the NEGATED holdout InfoNCE (higher is better everywhere in this harness)
    out = {}
    for h in hist:
        if h["phase"] == "final":
            continue
        out[int(h["step"])] = float(h["macro"])
    return out


def select(verbose=True):
    """The winning head lr per head, on the holdout only. Refuses if the ladder is incomplete."""
    winners, table = {}, {}
    for hk in HEADS:
        rows = {}
        for tag, lr in LADDER:
            rid = f"m8e14-lad-{hk}-lr{tag}"
            sc = SIDE / f"{rid}.json"
            if not sc.exists():
                raise SystemExit(f"ladder arm {rid} has no sidecar; run `ladder` first")
            d = json.loads(sc.read_text())
            rows[lr] = {"run_id": rid, "final_holdout": d["dev_proxy"],
                        "curve": holdout_curve(rid)}
        best = max(rows, key=lambda lr: rows[lr]["final_holdout"])
        winners[hk], table[hk] = best, rows
        if verbose:
            print(f"[{hk}] " + "  ".join(f"lr={lr:g}:{rows[lr]['final_holdout']:+.4f}"
                                         for lr in sorted(rows)) + f"  -> {best:g}")
    return winners, table


def adequacy_verdict():
    """The pre-registered plateau rule, as code.

    improvement(2500->5000) must be under 25% of improvement(1250->2500). If it is not, the primary
    reports OPTIMIZATION-INADEQUATE / UNINFORMATIVE and NOT a method null.
    """
    out = {}
    for hk in HEADS:
        rid = f"m8e14-step-{hk}"
        c = holdout_curve(rid)
        if not c or not {1250, 2500, 5000} <= set(c):
            out[hk] = {"run_id": rid, "verdict": "MISSING",
                       "note": "needs holdout readings at 1250, 2500 and 5000"}
            continue
        early, late = c[2500] - c[1250], c[5000] - c[2500]
        ratio = (late / early) if early > 0 else float("inf")
        out[hk] = {"run_id": rid, "holdout": {k: c[k] for k in (1250, 2500, 5000)},
                   "improvement_1250_2500": early, "improvement_2500_5000": late,
                   "ratio": ratio, "threshold": 0.25,
                   "verdict": "ADEQUATE" if ratio < 0.25 else "OPTIMIZATION-INADEQUATE",
                   "_what": ("a null at an inadequate step budget is evidence about the "
                             "configuration, not about the method")}
    return out


def collect():
    """Every arm's sidecar, plus the checks that make the set readable as a probe."""
    arms = {}
    for p in sorted(SIDE.glob("*.json")):
        if p.stem.endswith("-smoke"):
            continue
        arms[p.stem] = json.loads(p.read_text())
    problems = []
    for rid, d in arms.items():
        pr = d.get("provenance", {})
        moved = pr.get("head_max_abs_move_from_init", 0.0)
        if d["trainable"] and moved == 0.0:
            problems.append(f"{rid}: a TRAINABLE head never moved from its initialization")
        if not d["trainable"] and moved != 0.0:
            problems.append(f"{rid}: a FROZEN head moved ({moved:.3g}) -- R0N is not frozen")
        if pr.get("sha256", {}).get("phase_b_checkpoint") is None:
            problems.append(f"{rid}: no Phase-B checkpoint hash bound")
    for kind in ("r0n",) + HEADS:
        have = [r for r, d in arms.items() if d["tag"] == kind]
        if have and len(have) != len(SEEDS):
            problems.append(f"{kind}: {len(have)} arms, expected {len(SEEDS)} paired seeds")
    return arms, problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["selftest", "smoke", "ladder", "adequacy", "reported",
                                      "select", "collect"])
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--no-guard", action="store_true",
                    help="skip the registry gate. For a smoke on an uncommitted tree ONLY; a "
                         "reported arm must never be run this way.")
    a = ap.parse_args()

    if a.stage == "selftest":
        import e14_head
        out = e14_head.self_test()
        print(json.dumps(out, indent=2))
        return 0 if out["pass"] else 1

    if a.stage == "select":
        w, t = select()
        print(json.dumps({"winners": {k: v for k, v in w.items()}}, indent=1))
        return 0

    if a.stage == "collect":
        arms, problems = collect()
        print(json.dumps({"n_arms": len(arms), "arms": sorted(arms),
                          "problems": problems}, indent=1))
        return 1 if problems else 0

    if not a.no_guard:
        stamp = probe_guard.assert_registered(PROBE)
        print(f"[guard] {PROBE} registered under registry {stamp['registry_sha256'][:12]}",
              flush=True)

    if a.stage == "smoke":
        # Smoke the path with no execution history: a ladder arm exercises the holdout split, the
        # cache-subset encode and the dev-blindness refusals; a reported arm exercises the headed
        # scorer view. Both patch stacks, 90 steps each.
        arms = {}
        arms.update({k: v for k, v in plan_ladder().items() if k.endswith("lin-lr1e3")})
        arms.update({k: v for k, v in plan_reported({"lin": 1e-3, "mlp": 1e-3}).items()
                     if k in ("m8e14-r0n-s0", "m8e14-lin-s0")})
        run_arms(arms, smoke=True, only=a.only)
        return 0

    if a.stage == "ladder":
        run_arms(plan_ladder(), only=a.only)
        return 0

    winners, _ = select(verbose=True)
    if a.stage == "adequacy":
        run_arms(plan_adequacy(winners), only=a.only)
        print(json.dumps(adequacy_verdict(), indent=1))
        return 0
    if a.stage == "reported":
        v = adequacy_verdict()
        for hk, r in v.items():
            if r["verdict"] == "MISSING":
                raise SystemExit(f"step-adequacy has not run for {hk}; it gates the null and must "
                                 f"be measured before the reported arms are read.")
        run_arms(plan_reported(winners), only=a.only)
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
