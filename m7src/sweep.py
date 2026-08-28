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
    # A smoke must exercise every phase the OBJECTIVE has, regardless of the base's step counts:
    # min(steps_a, 30) on a steps_a=0 base silently skipped the A phase of an objective-C smoke
    # (2026-08-26), which is precisely the unexecuted-path bug this function exists to catch.
    sb = steps_b if cfg.objective in ("B", "C") else 0
    sa = steps_a if cfg.objective in ("A", "C") else 0
    cfg = replace(cfg, steps_b=sb, steps_a=sa, eval_every=10 ** 9)
    print(f"\n{'='*80}\nSMOKE {json.dumps(over)}\n{'='*80}", flush=True)
    dev, model, _ = run(cfg)
    del model
    torch.cuda.empty_cache()
    print(f"SMOKE ok (dev {dev:.4f} at {cfg.steps_b}+{cfg.steps_a} steps -- not a result)",
          flush=True)
    return dev


def smoke_chain(base, b_over, a_over, steps_b=60, steps_a=30):
    """Prove the TWO-RUN chain path before a night of them: a tiny B run, then a fresh A run that
    actually loads that B artifact. The B->A handoff is the part with no execution history --
    `init="run:<id>"` has to find the checkpoint, match its preprocessing fingerprint and its
    teacher, and restore its token weights -- and it is now also the path where `init_preproc`
    lets the runtime rule differ from the rule the rows were built under. Not written to
    RESULTS.md; the numbers are meaningless at 90 steps."""
    bid, aid = "smoke-chain-b", "smoke-chain-a"
    print(f"\n{'='*80}\nSMOKE CHAIN B {json.dumps(b_over)}\n{'='*80}", flush=True)
    cfg_b = replace(base, run_id=bid, **{**b_over, "steps_b": steps_b, "steps_a": 0,
                                         "eval_every": 10 ** 9})
    devb, mb, _ = run(cfg_b)
    del mb
    torch.cuda.empty_cache()
    print(f"\n{'='*80}\nSMOKE CHAIN A {json.dumps(a_over)}\n{'='*80}", flush=True)
    cfg_a = replace(base, run_id=aid, init=f"run:{bid}",
                    **{**a_over, "steps_b": 0, "steps_a": steps_a, "eval_every": 10 ** 9})
    deva, ma, _ = run(cfg_a)
    del ma
    torch.cuda.empty_cache()
    print(f"SMOKE CHAIN ok (B {devb:.4f} at {steps_b} steps -> A {deva:.4f} at {steps_a} steps "
          f"-- not results)", flush=True)
    return devb, deva


def chain(name, base, b_over, a_over, skip_b_if_exists=True):
    """One ablation arm as TWO separate runs: a B run, then a fresh A run initialized from that
    exact B artifact. Returns the A run's dev macro (None if either leg failed).

    This is not a convenience wrapper -- it is the only way to reproduce the winning recipe
    (Codex review #3 BLOCKER 4). A single objective-C run carries ONE learning rate, ONE schedule,
    and one Adam state with cumulative update counts across both phases, whereas the winner is
    B at 3e-3 constant followed by A at 1e-3 warmup-linear with a fresh optimizer and a reset
    update counter (which also rescales the 1/(1+updates) init penalty). An objective-C "full
    chain" would therefore differ from the candidate in four ways that have nothing to do with
    the variable the arm is meant to isolate.

    `skip_b_if_exists` reuses a B artifact that is already on disk -- arms that do not vary the B
    phase share one, and re-running it would spend an hour to reproduce it up to CUDA
    nondeterminism. The reuse is by run id, so a B arm and its variants can never collide.
    """
    bid, aid = f"{name}-b", f"{name}-a"
    if skip_b_if_exists and (WORK / "runs" / f"{bid}.npz").exists():
        # Reuse only if that artifact was trained under the overrides this arm asks for. Without
        # this, editing an arm's B definition and re-running silently produces an ablation that is
        # not the ablation it claims to be -- the worst failure mode this driver has, because the
        # number looks fine.
        prev = json.loads((WORK / "runs" / f"{bid}.json").read_text())["cfg"]
        want = replace(base, run_id=bid, **b_over)
        drift = {k: (prev.get(k), getattr(want, k)) for k in asdict(want)
                 if k != "run_id" and prev.get(k) != getattr(want, k)
                 and not (isinstance(getattr(want, k), (list, tuple))
                          and list(prev.get(k) or []) == list(getattr(want, k) or []))}
        if drift:
            raise ValueError(f"{bid} exists but was trained with different settings {drift}; "
                             f"delete work/runs/{bid}.* to retrain, or fix the arm")
        print(f"[{name}] reusing existing B artifact {bid} (config verified)", flush=True)
    else:
        if one(bid, base, **b_over) is None:
            print(f"[{name}] B leg failed; A leg skipped", flush=True)
            return None
    return one(aid, base, init=f"run:{bid}", **a_over)


def chains(name, base, arms, b_base, a_base, fail_fast=True):
    """arms: {suffix: {"b": {...}, "a": {...}, "share_b": <suffix or None>}}.

    `share_b` points an arm at another arm's B artifact when the arm varies only the A phase --
    the honest alternative to silently re-running an identical B leg.
    """
    out = {}
    for suffix, spec in arms.items():
        rid = f"{name}-{suffix}"
        b_over = {**b_base, **spec.get("b", {})}
        a_over = {**a_base, **spec.get("a", {})}
        share = spec.get("share_b")
        print(f"\n{'='*80}\n{rid}: B {json.dumps(spec.get('b', {}))} | "
              f"A {json.dumps(spec.get('a', {}))}"
              f"{f' | shares B with {share}' if share else ''}\n{'='*80}", flush=True)
        if share:
            bid = f"{name}-{share}-b"
            if not (WORK / "runs" / f"{bid}.npz").exists():
                print(f"[{rid}] SKIPPED: shared B artifact {bid} does not exist", flush=True)
                out[rid] = None
                continue
            out[rid] = one(f"{rid}-a", base, init=f"run:{bid}", **a_over)
        else:
            out[rid] = chain(rid, base, b_over, a_over)
        if out[rid] is None and last_status() == "FAILED":
            if fail_fast:
                print(f"\n[{name}] STOPPING: {rid} raised and the remaining arms share its code "
                      f"path.", flush=True)
                break
    print(f"\n[{name}] " + "  ".join(f"{k}={'—' if v is None else f'{v:.4f}'}"
                                     for k, v in out.items()), flush=True)
    return out


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
