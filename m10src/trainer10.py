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
import time
from pathlib import Path

import numpy as np
import torch

import nano10 as N


def eager(model):
    """The eager module behind a possibly-compiled wrapper. §T: no compiled state_dict, ever."""
    return getattr(model, "_orig_mod", model)


def save(path, model, opt, step, extra=None):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": eager(model).state_dict(), "opt": opt.state_dict(), "step": int(step),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": (torch.cuda.get_rng_state_all()
                             if torch.cuda.is_available() else None),
                "extra": extra or {}}, p)
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
              device="cpu", batch_size=32):
    """Run one arm. -> a record: losses, evaluations, rates, and why it stopped.

    `eval_fn(model, step, kind)` returns the arm's COV macro at a scheduled evaluation; `kind` is
    'end' at a cycle end and 'mid' otherwise, which is the distinction §Kill compares within.
    """
    loss_fn = N.LOSSES[loss_name]
    opt = torch.optim.AdamW(model.parameters(), lr=peak)
    start = 0
    if resume_from:
        start, _ = load(resume_from, model, opt)
    else:
        torch.manual_seed(seed)

    ends = set(N.cycle_ends(total_steps, cycles))
    per = max(total_steps // cycles, 1)
    mids = {c * per + per // 2 for c in range(cycles)}
    losses, evals, kinds, cycle_end_evals = [], [], [], []
    n_examples, t0 = 0, time.time()
    stopped = None

    for step in range(start, total_steps):
        kind = N.mix_window(pattern, step)
        ids, mask, tgt = batch_fn(step, kind)
        lr = N.lr_at(step, total_steps, cycles, peak, final)
        for g in opt.param_groups:
            g["lr"] = lr
        pred = model(ids.to(device), mask.to(device))
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
        opt.step()
        losses.append(float(loss.detach()))
        n_examples += len(ids)

        if eval_fn is not None and (step in ends or step in mids):
            k = "end" if step in ends else "mid"
            m = eval_fn(model, step, k)
            evals.append(m); kinds.append(k)
            if k == "end":
                cycle_end_evals.append(m)
            fired, why = N.kill_fires(evals, lambda i: kinds[i])
            if fired:
                stopped = f"kill: {why}"
                break
            pf, at = N.plateau_fires(cycle_end_evals)
            if pf:
                stopped = f"plateau at cycle {at}"
                break
        if ckpt_path and ckpt_every and (step + 1) % ckpt_every == 0:
            save(ckpt_path, model, opt, step + 1)
        if log_every and (step + 1) % log_every == 0:
            el = time.time() - t0
            print(f"  step {step + 1}/{total_steps} loss {np.mean(losses[-log_every:]):.4f} "
                  f"lr {lr:.2e} {n_examples / max(el, 1e-9):.0f} ex/s", flush=True)

    el = time.time() - t0
    return {"steps_run": len(range(start, min(total_steps, start + len(losses)))),
            "start_step": start, "total_steps": total_steps, "pattern": pattern,
            "loss": loss_name, "losses": losses, "evals": evals, "eval_kinds": kinds,
            "cycle_end_evals": cycle_end_evals, "stopped": stopped,
            "examples": n_examples, "seconds": round(el, 2),
            "examples_per_s": round(n_examples / max(el, 1e-9), 1),
            "mix": N.window_shares(pattern, max(len(losses), 1))}
