"""Sequential experiment driver. Appends every run -- including stopped, failed and OOM ones --
to m7/RESULTS.md, which is the experiment ledger the mandate requires.
"""
import json
import sys
import traceback
from dataclasses import asdict, replace

import torch

from _paths import REPO, WORK
from train import Cfg, run

RESULTS = REPO / "m7" / "RESULTS.md"


def append_row(run_id, cfg, dev, verdict, extra=""):
    with open(RESULTS, "a") as f:
        f.write(f"| {run_id} | `work/runs/{run_id}.json` | "
                f"{'—' if dev is None else f'{dev:.4f}'} | {verdict}{(' — ' + extra) if extra else ''} |\n")


_LAST_STATUS = None


def _set_status(v):
    global _LAST_STATUS
    _LAST_STATUS = v


def last_status():
    """Status of the most recent `one()`: "ok", "OOM", or "FAILED". `one` returns None for both
    OOM and exception, and a grid needs to tell those apart to decide whether to keep going."""
    return _LAST_STATUS


def one(run_id, base=None, **over):
    cfg = replace(base or Cfg(), run_id=run_id, **over)
    (WORK / "runs").mkdir(parents=True, exist_ok=True)
    try:
        dev, model, hist = run(cfg)
        _set_status("ok")
        append_row(run_id, cfg, dev, "ok")
        del model
        torch.cuda.empty_cache()
        return dev
    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        _set_status("OOM")
        append_row(run_id, cfg, None, "OOM", str(e)[:120])
        print(f"[{run_id}] OOM: {e}", flush=True)
    except Exception as e:
        _set_status("FAILED")
        append_row(run_id, cfg, None, "FAILED", f"{type(e).__name__}: {str(e)[:120]}")
        print(f"[{run_id}] FAILED\n{traceback.format_exc()}", flush=True)
    return None


def smoke(base, over, steps_b=60, steps_a=30):
    """Run one arm at trivial step counts to prove the CODE PATH before a grid commits hours to it.

    Earned the hard way: the phase-2 screen crashed four consecutive arms on the same shape error
    in the KL candidate set, because no arm with mined hard negatives had ever executed. A 3-minute
    smoke would have caught it before the first 3-hour mining pass. Not written to RESULTS.md --
    it is a code check, not an experiment, and its dev number is meaningless at 90 steps.
    """
    cfg = replace(base, run_id="smoke", **over)
    cfg = replace(cfg, steps_b=min(cfg.steps_b, steps_b), steps_a=min(cfg.steps_a, steps_a),
                  eval_every=10 ** 9)
    print(f"\n{'='*80}\nSMOKE {json.dumps(over)}\n{'='*80}", flush=True)
    dev, model, _ = run(cfg)
    del model
    torch.cuda.empty_cache()
    print(f"SMOKE ok (dev {dev:.4f} at {cfg.steps_b}+{cfg.steps_a} steps -- not a result)",
          flush=True)
    return dev


def grid(name, base, variants, fail_fast=True):
    """variants: {suffix: {field: value}}. Runs them in order, returns {run_id: dev macro}.

    fail_fast stops after the first arm that raises, because arms in a grid share a code path: the
    phase-2 screen repeated one identical shape error four times, spending a mining pass each time,
    and reported it as four dashes in a summary line. An OOM does NOT trip it -- that is a real
    per-arm resource result and the remaining arms may be smaller.
    """
    out, failed = {}, []
    for suffix, over in variants.items():
        rid = f"{name}-{suffix}"
        print(f"\n{'='*80}\n{rid}: {json.dumps(over)}\n{'='*80}", flush=True)
        out[rid] = one(rid, base, **over)
        if out[rid] is None and last_status() == "FAILED":
            failed.append(rid)
            if fail_fast:
                print(f"\n[{name}] STOPPING: {rid} raised, and the remaining "
                      f"{len(variants) - len(out)} arm(s) share its code path. Fix and rerun; "
                      f"pass fail_fast=False to grind through failures deliberately.", flush=True)
                break
    print(f"\n[{name}] " + "  ".join(f"{k}={'—' if v is None else f'{v:.4f}'}" for k, v in out.items()),
          flush=True)
    if failed:
        print(f"[{name}] FAILED ARMS: {failed}", flush=True)
    return out


if __name__ == "__main__":
    print("import this module and call grid()/one(); see m7/RESULTS.md for the ledger")
