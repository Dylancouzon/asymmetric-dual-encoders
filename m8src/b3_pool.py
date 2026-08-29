"""B3: is Phase A pair-starved? Real-pair pool scaling at fixed compute (registry probe `B3`).

WHAT THIS MEASURES, and why it is not what B3 was originally registered to do. B3's first lever was
synthetic ICT augmentation. An adversarial review of the ARM DEFINITION -- run before any arm
existed -- established that it could not answer the question (LEDGER §15, registry row `B3-ICT`):
"equal updates AND equal exposure" over-constrains a fixed batch, and adding synthetic pairs at
fixed compute measures whether spending Phase-A budget on ICT helps, not whether Phase A is short
of pairs. So the lever is now the real pair pool itself.

THE DESIGN. Four nested subsets of the Phase-A pair pool -- {0.25, 0.50, 0.75, 1.00} of the
340,850 pairs `train.kept_pairs()` returns, realising the 338,076 a run records once banned
positives drop -- at THREE seeds each, with updates (2,500), batch (512), negatives (32,768),
temperature, learning rate and the Phase-B checkpoint ALL held. Total draws are therefore 1,280,000 in every arm. Compute is fixed;
the only thing that varies is how many DISTINCT pairs those draws are spread over, which is
diversity against repetition at constant budget. That is the honest form of "pair-starved": if
quality is still climbing where the real pool runs out, more pairs would buy something.

WHAT A NULL DOES NOT LICENSE. `p35b-2m` ran 16,000 objective-B steps over the full pair query set,
so every arm -- 0.25 included -- starts from a table that has already seen all ~338K training
queries and their teacher vectors. This probe measures the marginal value of distinct pairs IN
PHASE A GIVEN B absorbed them. A B-side pair lever flows through the leg held fixed here, so it is
not covered by this result; the registry's `no_survivor` is scoped accordingly.

NESTING IS THE POINT. The subsets are prefixes of ONE fixed permutation, drawn from a pool seed
that is deliberately INDEPENDENT of the training seed. So 0.25 is a subset of 0.50 is a subset of
0.75, and no between-arm sampling noise enters the dose contrast -- the arms differ by which pairs
were ADDED, never by which were swapped.

THE TREATMENT DOES NOT APPEAR IN `cfg`, WHICH IS THE HAZARD HERE. G3 forbids editing `m7src`, and
`Cfg` has no pair-fraction knob, so the fraction is applied by patching `train.kept_pairs` inside
each arm's subprocess. Nothing in the run's `meta.json` would record it. Two things therefore
compensate, and neither is optional: the fraction is in the RUN ID, and every arm writes a stamped
sidecar naming its fraction, its realised pair count and the pool seed. See CODEMAP 16 -- an
unrecorded treatment is the dangerous case, not a crashing one.
"""
import argparse
import json
import subprocess
import sys
import time

import m8base
import probe_guard

REPO = m8base.REPO
WORK = m8base.WORK
RESULTS = m8base.RESULTS
OUT = RESULTS / "m8_b3_pool.json"

B_CHECKPOINT = "p35b-2m"          # the ONE Phase-B checkpoint every arm inits from
A_BASE_RUN = "p35w-2m-s2500"      # the frozen Phase-A recipe every arm inherits
FRACTIONS = (0.25, 0.50, 0.75, 1.00)
SEEDS = (0, 1, 2)
# Fraction 1.00 is a no-op patch, so the A-leg noise-floor arms ARE the 1.00 arms: same Phase-B
# checkpoint, same frozen Phase-A recipe, full pool (n_train_pairs 338,076), differing only in
# seed. Verified config-identical before being adopted. Retraining them would burn ~20 minutes to
# reproduce three artifacts that already exist.
EXISTING_AT_1_00 = {0: "m8nf-seed0", 1: "m8nf-seed1", 2: "m8nf-seed2"}
POOL_SEED = 20260829              # fixes the nesting; independent of the training seed BY DESIGN


def arm_id(frac, seed):
    if frac >= 1.0 and seed in EXISTING_AT_1_00:
        return EXISTING_AT_1_00[seed]
    return f"m8b3-p{int(round(frac * 100)):03d}-s{seed}"


def base_cfg(run=A_BASE_RUN):
    meta = json.loads((WORK / "runs" / f"{run}.meta.json").read_text())["cfg"]
    return {k: v for k, v in meta.items() if k != "run_id"}


def plan():
    cfg = base_cfg()
    arms = {}
    for f in FRACTIONS:
        for s in SEEDS:
            rid = arm_id(f, s)
            arms[rid] = {**cfg, "seed": s, "init": f"run:{B_CHECKPOINT}",
                         "steps_b": 0, "_frac": f, "_seed": s,
                         "_exists": (WORK / "runs" / f"{rid}.npz").exists()}
    return arms


def _subprocess_code(rid, over, frac, sidecar):
    """The arm, as a standalone process. `train.kept_pairs` is patched BEFORE `sweep.one` runs."""
    return (
        "import os, sys, json\n"
        "os.environ.setdefault('M7_ENCODER', 'stella-400M-v5')\n"
        "sys.path.insert(0, %r); sys.path.insert(0, %r); sys.path.insert(0, %r)\n"
        # m8base installs the G2 protected-path guard. Without it the process that does the actual
        # training -- the one that touches data -- runs unguarded.
        "import m8base\n"
        "import numpy as np, program, sweep, train\n"
        "FRAC = %r; POOL_SEED = %r\n"
        "_orig = train.kept_pairs\n"
        "_seen = {}\n"
        "def _patched(sources=None):\n"
        "    ps = _orig(sources)\n"
        "    _seen['full'] = len(ps)\n"
        "    if FRAC >= 1.0:\n"
        "        _seen['kept'] = len(ps)\n"
        "        return ps\n"
        "    # ONE fixed permutation, prefix-sliced -> the fractions are nested by construction.\n"
        "    rng = np.random.default_rng(POOL_SEED)\n"
        "    n = int(round(FRAC * len(ps)))\n"
        "    keep = sorted(rng.permutation(len(ps))[:n].tolist())\n"
        "    _seen['kept'] = len(keep)\n"
        "    return [ps[i] for i in keep]\n"
        "train.kept_pairs = _patched\n"
        "d = sweep.one(%r, program.BASE, **%r)\n"
        "if not _seen:\n"
        "    print('POOL PATCH NEVER FIRED', flush=True); sys.exit(3)\n"
        "json.dump({'run_id': %r, 'pair_fraction': FRAC, 'pool_seed': POOL_SEED,\n"
        "           'pairs_full': _seen.get('full'), 'pairs_kept': _seen.get('kept'),\n"
        # `sweep.one` returns the dev proxy as a FLOAT, not a dict -- the first version called
        # .get() on it and every arm died after training, at the very last line.
        "           'dev_proxy': (d if isinstance(d, (int, float)) else None)},\n"
        "          open(%r, 'w'), indent=1)\n"
        "print('DEVPROXY', %r, d, flush=True)\n"
        "sys.exit(0 if d is not None else 2)\n"
        % (str(REPO / "m7src"), str(REPO / "bench"), str(REPO / "m8src"), frac, POOL_SEED,
           rid, over, rid, str(sidecar), rid))


def train(rids=None, smoke=False):
    probe_guard.assert_registered("B3")
    arms = plan()
    sc_dir = WORK / "b3_sidecars"
    sc_dir.mkdir(parents=True, exist_ok=True)
    todo = [(r, a) for r, a in arms.items()
            if (rids is None or r in rids) and not a["_exists"]]
    if not todo:
        print("every arm already on disk")
        return
    print(f"{len(todo)} arms to train: {[r for r, _ in todo]}", flush=True)
    for rid, a in todo:
        over = {k: v for k, v in a.items() if not k.startswith("_")}
        run_id = rid
        if smoke:
            # never occupy a real run id with a 90-step artifact: `plan()`'s `_exists` check would
            # then read the real arm as trained (noise_floor.py learned this the hard way).
            over["steps_a"] = 90
            run_id = rid + "-smoke"
        sidecar = sc_dir / f"{run_id}.json"
        code = _subprocess_code(run_id, over, a["_frac"], sidecar)
        t0 = time.time()
        print(f"[{run_id}] launching (frac={a['_frac']}, seed={a['_seed']}, "
              f"steps_a={over.get('steps_a')}, init={over.get('init')})", flush=True)
        r = subprocess.run([sys.executable, "-u", "-c", code], cwd=str(REPO))
        print(f"[{run_id}] exit {r.returncode} in {time.time()-t0:.0f}s", flush=True)
        if r.returncode != 0:
            raise SystemExit(f"{run_id} failed (exit {r.returncode}); stopping rather than "
                             f"measuring a dose curve from a partial arm set.")
        if sidecar.exists():
            print(f"  sidecar: {json.dumps(json.loads(sidecar.read_text()))}", flush=True)


def collect():
    """Gather the sidecars and verify the arms are a dose curve, BEFORE anything is scored.

    The first version claimed to "check the nesting" and did not: a missing sidecar was silently
    dropped, `pairs_full` was never compared across arms, and `pool_seed` was written but never
    read. All three matter. The dangerous window is a child killed between `save_table` and the
    sidecar write -- that leaves the .npz on disk, so `_exists` reads True forever, with no record
    that any fraction was applied.
    """
    sc_dir = WORK / "b3_sidecars"
    rows, problems = {}, []
    for f in FRACTIONS:
        for s_ in SEEDS:
            rid = arm_id(f, s_)
            p = sc_dir / f"{rid}.json"
            if p.exists():
                rows[rid] = json.loads(p.read_text())
            elif f >= 1.0:
                # the adopted floor arms predate this probe and have no sidecar; synthesize one
                # from the run's own record, which carries the pair count that matters.
                rp = WORK / "runs" / f"{rid}.json"
                if rp.exists():
                    n = json.loads(rp.read_text()).get("n_train_pairs")
                    rows[rid] = {"run_id": rid, "pair_fraction": 1.0, "pool_seed": POOL_SEED,
                                 "pairs_full": None, "pairs_kept": n, "adopted": True}
                else:
                    rows[rid] = None
            else:
                rows[rid] = None
            if rows[rid] is None:
                problems.append(f"{rid}: NO sidecar and no adoptable run record -- its fraction is "
                                f"unrecorded, so it cannot enter the curve")

    got = {r: v for r, v in rows.items() if v}
    fulls = {v["pairs_full"] for v in got.values() if v.get("pairs_full") is not None}
    if len(fulls) > 1:
        problems.append(f"arms disagree on the FULL pool size {sorted(fulls)}: the permutation is "
                        f"taken over that length, so a change breaks nesting while leaving the "
                        f"per-arm counts looking monotone")
    for rid, v in got.items():
        if v.get("pool_seed") != POOL_SEED:
            problems.append(f"{rid}: pool_seed {v.get('pool_seed')} != {POOL_SEED}")
        full = v.get("pairs_full")
        if full is not None and not v.get("adopted"):
            want = full if v["pair_fraction"] >= 1.0 else int(round(v["pair_fraction"] * full))
            if v.get("pairs_kept") != want:
                problems.append(f"{rid}: kept {v.get('pairs_kept')} but frac {v['pair_fraction']} "
                                f"of {full} is {want}")

    by_frac = {}
    for rid, v in got.items():
        by_frac.setdefault(v["pair_fraction"], set()).add(v["pairs_kept"])
    problems += [f"fraction {f}: seeds disagree on pair count {sorted(c)}"
                 for f, c in by_frac.items() if len(c) > 1]
    counts = [(f, sorted(c)[0]) for f, c in sorted(by_frac.items())]
    if any(b <= a for (_, a), (_, b) in zip(counts, counts[1:])):
        problems.append(f"pair counts are not increasing in the fraction: {counts}")
    return {"arms": rows, "pair_counts_by_fraction": dict(counts), "problems": problems}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["plan", "train", "collect"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--only", default=None, help="comma-separated run ids")
    a = ap.parse_args()
    rids = a.only.split(",") if a.only else None
    if a.step == "plan":
        print(json.dumps({r: {k: v for k, v in x.items() if k.startswith("_")
                              or k in ("seed", "init", "steps_a", "steps_b", "batch")}
                          for r, x in plan().items()}, indent=1, default=str))
    elif a.step == "train":
        train(rids=rids, smoke=a.smoke)
    else:
        d = collect()
        print(json.dumps(d, indent=1, default=str))
        if d["problems"]:
            raise SystemExit("B3's arms are not a dose curve: " + "; ".join(d["problems"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
