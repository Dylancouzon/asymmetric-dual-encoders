"""The M10 training loop: the step state machine, checkpointing, and resume.

Data is supplied by a caller-owned `batch_fn(step, kind) -> (input_ids, attention_mask, targets)`
so the loop can be tested without the corpus. What lives here is everything that decides an arm's
trajectory and everything a crash can corrupt:

- the 4-step mix window (family B) and the 3-cycle schedule, both from `m10src/nano10`;
- the registered evaluation cadence — COV at every cycle end, plus midpoints — and the kill and
  plateau rules read off those evaluations;
- an examples/s counter, because §0b records measured rates and a docstring's estimate has been
  wrong by two orders of magnitude in this repo before;
- checkpoint and resume, which is the part a seven-day build cannot get wrong: the checkpoint
  carries the model, the optimizer, the step, AND the RNG states, and `test_trainer10` proves a
  resumed run reproduces an uninterrupted one step for step rather than merely "looking similar".

`torch.compile` binds the training STEP only (`m10/HEADROOM.md` §T): checkpoints save the eager
module via `_orig_mod`, and export, parity, encode and evaluation all run eager.
"""
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

import nano10 as N


def eager(model):
    """The eager module behind a possibly-compiled wrapper. §T: no compiled state_dict, ever."""
    return getattr(model, "_orig_mod", model)


def param_groups(model, wd=0.01):
    """§Recipe: "AdamW beta=(0.9, 0.999), eps 1e-8, wd 0.01 on dim>1". Biases and LayerNorm gains
    are 1-D and take NO decay; torch's `AdamW(model.parameters())` decays them too (Codex runner
    review 2026-09-07, finding 2)."""
    ps = [p for p in model.parameters() if p.requires_grad]
    return [{"params": [p for p in ps if p.dim() > 1], "weight_decay": float(wd)},
            {"params": [p for p in ps if p.dim() <= 1], "weight_decay": 0.0}]


def autocast_for(device, dtype=torch.bfloat16):
    """§Recipe: "fp32 loss, bf16 autocast" — the form `rate_bench_real` measured the 910 ex/s on.

    Enabled on CUDA only: bf16 autocast on CPU is a different (and on this box slower) kernel set,
    and the screens' CPU smokes are a path check, not a rate. The loss is computed OUTSIDE this
    context on `.float()` predictions, so only the forward is reduced precision.
    """
    cuda = torch.device(device).type == "cuda"
    return torch.autocast("cuda", dtype=dtype, enabled=cuda), cuda


def save(path, model, opt, step, extra=None):
    """Atomic: a temp file, fsynced, then `os.replace`.

    The checkpoint is the ONLY recovery point of a multi-day build, and `torch.save` writing
    straight onto it means a crash mid-write destroys the run rather than costing it one interval
    (Codex 2026-09-05 finding 10).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f".tmp{os.getpid()}")
    with open(tmp, "wb") as fh:
        torch.save({"model": eager(model).state_dict(), "opt": opt.state_dict(),
                    "step": int(step), "torch_rng": torch.get_rng_state(),
                    "cuda_rng": (torch.cuda.get_rng_state_all()
                                 if torch.cuda.is_available() else None),
                    "extra": extra or {}}, fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, p)
    return str(p)


def load(path, model, opt):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    eager(model).load_state_dict(ck["model"])
    opt.load_state_dict(ck["opt"])
    torch.set_rng_state(ck["torch_rng"])
    if ck.get("cuda_rng") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(ck["cuda_rng"])
    return int(ck["step"]), ck.get("extra", {})


def train_arm(model, batch_fn, total_steps, *, pattern="75/25", cycles=3, peak=1e-4, final=1e-5,
              loss_name="squared_l2", sigma=None, eval_fn=None, evals_per_cycle=2,
              ckpt_path=None, ckpt_every=0, resume_from=None, seed=0, log_every=0,
              device="cpu", batch_size=32, read_steps=(), cycle_ckpt_fmt=None,
              eval_state=None, fingerprint=None, wd=0.01):
    """Run one arm. -> a record: losses, evaluations, rates, and why it stopped.

    `eval_fn(model, step, kind)` returns the arm's COV macro at a scheduled evaluation; `kind` is
    'end' at a cycle end and 'mid' otherwise, which is the distinction §Kill compares within.

    `read_steps` are EXTRA reading points (family F is read at 5M/10M/20M inside a 20M schedule,
    and 5M is neither a cycle end nor a midpoint). They get `kind='read'` and are recorded in
    `read_evals`, **outside** the kill/plateau state: §Kill compares "midpoint against midpoints,
    cycle end against cycle ends", so a third kind entering that sequence would change a
    registered rule rather than implement it.

    `cycle_ckpt_fmt` is a path template (`.../cycle{cycle}.pt`) saved at every cycle end and
    RETAINED — the rolling `ckpt_path` is a resume point that gets overwritten, while the final
    cycle end is the arm's final checkpoint and its sha256 goes in the arm's record. It is written
    before the kill/plateau test, so an arm that stops AT a cycle end still has that checkpoint.

    `eval_state` is the CALLER'S evaluator (`run_arm.CovEval`), whose `.state()`/`.restore()` go
    into the checkpoint beside the loop's own state: its per-family macros and per-query score
    paths are part of the run and a resume that loses them loses the contrast step's inputs
    (Codex runner review 2026-09-07, finding 7).

    `fingerprint` binds a checkpoint to the recipe that wrote it. A resume whose fingerprint
    differs is REFUSED rather than continuing one arm's schedule into another arm's weights
    (finding 8). Checkpoints written without one never satisfy a fingerprinted resume.
    """
    loss_fn = N.LOSSES[loss_name]
    opt = torch.optim.AdamW(param_groups(model, wd), lr=peak, betas=(0.9, 0.999), eps=1e-8)
    amp, amp_on = autocast_for(device)
    start = 0
    losses, evals, kinds, cycle_end_evals, read_evals = [], [], [], [], []
    gn_stats = {"n": 0, "sum": 0.0, "max": 0.0, "min": float("inf"), "n_clipped": 0}
    n_examples, stopped = 0, None
    if resume_from:
        # The EVALUATION history is part of the run state. Without it a resumed arm restarts
        # `cycle_end_evals` empty, so the plateau rule reads one cycle where it needs three and
        # the registered kill/plateau decision cannot fire at all (Codex 2026-09-05 finding 4).
        start, ex = load(resume_from, model, opt)
        if fingerprint is not None and ex.get("fingerprint") != fingerprint:
            raise SystemExit(
                f"REFUSED: {resume_from} was written by a different recipe "
                f"(fingerprint {ex.get('fingerprint')!r} vs {fingerprint!r}). A resume continues "
                f"ONE arm; move the checkpoint aside deliberately.")
        if eval_state is not None and ex.get("eval_state") is not None:
            eval_state.restore(ex["eval_state"])
        losses = list(ex.get("losses", []))
        evals = list(ex.get("evals", []))
        kinds = list(ex.get("eval_kinds", []))
        cycle_end_evals = list(ex.get("cycle_end_evals", []))
        read_evals = list(ex.get("read_evals", []))
        n_examples = int(ex.get("examples", 0))
        stopped = ex.get("stopped")
    else:
        torch.manual_seed(seed)

    model.train()               # the warm start's `pooled_features` left it in eval() (finding 1)

    def extra():
        e = {"losses": losses, "evals": evals, "eval_kinds": kinds,
             "cycle_end_evals": cycle_end_evals, "read_evals": read_evals,
             "examples": n_examples, "stopped": stopped, "fingerprint": fingerprint}
        if eval_state is not None:
            e["eval_state"] = eval_state.state()
        return e

    ends = set(N.cycle_ends(total_steps, cycles))
    reads = {int(s) for s in read_steps} - ends
    per = max(total_steps // cycles, 1)
    mids = {c * per + per // 2 for c in range(cycles)}
    t0, run_examples, run_steps = time.time(), 0, 0

    for step in range(start, total_steps if stopped is None else start):
        kind = N.mix_window(pattern, step)
        ids, mask, tgt = batch_fn(step, kind)
        lr = N.lr_at(step, total_steps, cycles, peak, final, N.warmup_steps_for(batch_size))
        for g in opt.param_groups:
            g["lr"] = lr
        with amp:                                       # bf16 forward on CUDA, no-op on CPU
            pred = model(ids.to(device), mask.to(device))
        pred = pred.float()                             # the LOSS is fp32 (§Recipe)
        loss = loss_fn(pred, tgt.to(device)) if sigma is None \
            else loss_fn(pred, tgt.to(device), sigma)
        if not torch.isfinite(loss):
            stopped = f"non-finite loss at step {step}"
            losses.append(float("nan"))
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gn):
            stopped = f"non-finite grad norm at step {step}"
            break
        # RECORD the pre-clip norm, do not just guard on it. The clip threshold is 1.0 and
        # objectives differ enormously in gradient scale -- measured 2026-09-09, D-COV's
        # unit-trace covariance loss is 1024x smaller than squared_l2's and its gradients 512x
        # smaller (`results/m10_dcov_gradient_audit.json`), so squared_l2 can clip while D-COV
        # never can. That asymmetry is exactly what a D1/D2 reading needs to rule out, and it was
        # unrecoverable from the screen's arms because this norm was computed and thrown away.
        g = float(gn)
        gn_stats["n"] += 1
        gn_stats["sum"] += g
        gn_stats["max"] = max(gn_stats["max"], g)
        gn_stats["min"] = min(gn_stats["min"], g)
        if g > 1.0:
            gn_stats["n_clipped"] += 1
        opt.step()
        losses.append(float(loss.detach()))
        n_examples += len(ids)
        run_examples += len(ids)
        run_steps += 1

        if eval_fn is not None and (step in ends or step in mids or step in reads):
            k = "end" if step in ends else ("mid" if step in mids else "read")
            m = eval_fn(model, step, k)
            model.train()               # every evaluator encodes in eval(); restore train mode
            if k == "read":
                # a READING point, not a scheduled evaluation: recorded, and deliberately not
                # part of the kill/plateau state (see the docstring).
                read_evals.append({"step": int(step), "macro": m})
            else:
                evals.append(m); kinds.append(k)
                if k == "end":
                    cycle_end_evals.append(m)
                    if cycle_ckpt_fmt:
                        save(str(cycle_ckpt_fmt).format(cycle=len(cycle_end_evals)), model, opt,
                             step + 1, extra=extra())
                fired, why = N.kill_fires(evals, lambda i: kinds[i])
                if fired:
                    stopped = f"kill: {why}"
                    break
                pf, at = N.plateau_fires(cycle_end_evals)
                if pf:
                    stopped = f"plateau at cycle {at}"
                    break
        if ckpt_path and ckpt_every and (step + 1) % ckpt_every == 0:
            save(ckpt_path, model, opt, step + 1, extra=extra())
        if log_every and (step + 1) % log_every == 0:
            el = time.time() - t0
            print(f"  step {step + 1}/{total_steps} loss {np.mean(losses[-log_every:]):.4f} "
                  f"lr {lr:.2e} {run_examples / max(el, 1e-9):.0f} ex/s", flush=True)

    if stopped and ckpt_path:
        # the STOPPING evaluation is part of the arm's record: every `stopped` path breaks before
        # the interval save, so without this the last checkpoint knows nothing about the kill or
        # the plateau that ended the run (Codex re-review 2026-09-05). This must fire even when
        # `run_steps == 0` -- a RESUMED arm whose very first step is non-finite stops before
        # taking one, and the checkpoint must still record why (Codex 2026-09-05, third pass).
        save(ckpt_path, model, opt, start + run_steps, extra=extra())
    el = time.time() - t0
    # `losses`/`evals` are the WHOLE arm's history (restored on resume); `steps_run` and the rate
    # are this process's, because a rate measured over another machine's steps is not a rate.
    n = gn_stats["n"] or 1
    grad_norm_report = {
        "n_steps": gn_stats["n"],
        "mean": round(gn_stats["sum"] / n, 6),
        "max": round(gn_stats["max"], 6),
        "min": (round(gn_stats["min"], 6) if gn_stats["n"] else None),
        "n_clipped_at_1.0": gn_stats["n_clipped"],
        "clip_rate": round(gn_stats["n_clipped"] / n, 6),
        "_what": ("PRE-clip gradient norms. The clip threshold is 1.0, and objectives differ hugely "
                  "in gradient scale (D-COV's are 512x smaller than squared_l2's), so an objective "
                  "comparison must be able to show whether one arm clipped and the other could not. "
                  "The screen's arms discarded this, which is why D1/D2 carry that caveat."),
    }
    return {"grad_norm": grad_norm_report,
            "steps_run": run_steps,
            "start_step": start, "total_steps": total_steps, "pattern": pattern,
            "loss": loss_name, "losses": losses, "evals": evals, "eval_kinds": kinds,
            "cycle_end_evals": cycle_end_evals, "read_evals": read_evals, "stopped": stopped,
            "examples": n_examples, "examples_this_run": run_examples, "seconds": round(el, 2),
            "examples_per_s": round(run_examples / max(el, 1e-9), 1),
            "bf16_autocast": bool(amp_on), "weight_decay_on_dim_gt_1": float(wd),
            "warmup_steps": int(N.warmup_steps_for(batch_size)),
            "warmup_examples": int(N.WARMUP_EXAMPLES),
            "mix": N.window_shares(pattern, max(len(losses), 1))}
