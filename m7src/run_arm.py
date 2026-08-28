"""Run exactly ONE training arm, then exit.

The ablation driver used to run every arm inside one long-lived python process. That process
accumulated this repo's deliberate module-level caches (`mix.load_source`, `heldout._DOC_IDS`,
`dev_eval._HELD_CACHE`, encode memmaps) on top of each arm's own ~4 GB of pseudo-query targets and
~4 GB negative bank, and the third chain reached 24.7 GB RSS on a 25 GB box and thrashed. Running
each arm in a fresh process is the fix, and it also makes the arms comparable: every one starts
from the same memory state instead of from whatever its predecessors left behind.

Usage: run_arm.py <phase> <suffix> <leg>      leg is "b" or "a"
       run_arm.py p4 base b
"""
import json
import sys

import program
import sweep
from _paths import WORK


def main(phase, suffix, leg):
    surv, b_base, a_base = program.ablation_recipe()
    arms = program.ARMS[phase]
    if suffix not in arms:
        raise SystemExit(f"unknown arm {phase}-{suffix}; known: {sorted(arms)}")
    spec = arms[suffix]
    rid = f"{phase}-{suffix}"

    if leg == "b":
        if "init" in spec:
            raise SystemExit(f"{rid} is an A-only arm (it names its own init); there is no B leg")
        over = {**b_base, **spec.get("b", {})}
        print(f"[{rid}-b] {json.dumps(spec.get('b', {}))}", flush=True)
        dev = sweep.one(f"{rid}-b", program.BASE, **over)
    elif leg == "a":
        over = {**a_base, **spec.get("a", {})}
        init = spec.get("init")
        if init == "@candidate_b":
            # the checkpoint the surviving candidate itself was initialized from
            init = json.loads((WORK / "runs" / f"{surv}.json").read_text())["cfg"]["init"]
        elif init is None:
            init = f"run:{rid}-b"
            if not (WORK / "runs" / f"{rid}-b.npz").exists():
                raise SystemExit(f"{rid}-b has not been trained; run its B leg first")
        print(f"[{rid}-a] init={init} {json.dumps(spec.get('a', {}))}", flush=True)
        dev = sweep.one(f"{rid}-a", program.BASE, init=init, **over)
    else:
        raise SystemExit(f"leg must be 'b' or 'a', got {leg!r}")
    print(f"[{rid}-{leg}] {'—' if dev is None else f'{dev:.4f}'}", flush=True)
    return 0 if dev is not None else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2], sys.argv[3]))
