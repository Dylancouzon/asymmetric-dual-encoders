"""Split-run equivalence: does `N` steps equal `N1` + kill + resume + `N2`, exactly?

Codex review #5's item 7, and the one check that decides whether a crash on day four of a
seven-day build costs an hour or the whole run. Everything else about resume can be argued from the
code; this measures it.

It compares, after both paths reach the same step:

* every model parameter, bit for bit;
* every optimizer state tensor (Adam's exp_avg / exp_avg_sq — a resume that reloads weights but not
  moments looks fine for a hundred steps and is a different run by day two);
* stream positions per source, and the cumulative token/example ledger;
* phase, and the learning rate the schedule would next produce.

Run against a scratch config and a scratch checkpoint directory so it can never touch a real build.
CUDA determinism is enforced in this test process so exact equality is a meaningful gate even when
the production training process intentionally uses the faster default kernel selection.

    python m9src/test_resume.py --n1 12 --n2 8
"""
import argparse
import json
import os
import shutil
import time
from pathlib import Path

# Must be set before torch initializes CUDA. Deterministic algorithms will raise instead of falling
# back if this model reaches an operation for which PyTorch has no deterministic implementation.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch

import m9base

import longrun     # noqa: E402
import make_config  # noqa: E402

SCRATCH = Path("/tmp/m9long_test")


def enforce_determinism():
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def scratch_cfg(total_steps):
    # The REAL screen decision, not a hard-coded recipe: if the screen picked MiniLM or policy
    # (a), a hard-coded config would fail the corpus checks -- or worse, prove resume for a
    # recipe the build will not run (Fable review, M1). build() fails closed without a decision.
    cfg = make_config.build()
    cfg.update({"run_id": "m9-build-test", "tokens_per_step": 1024, "warmup_steps": 5,
                "log_every": 10 ** 9, "ckpt_every": 10 ** 9, "eval_every": 10 ** 9,
                "stable_token_cap": 10 ** 15})
    cfg["_hash"] = longrun.canon({k: v for k, v in cfg.items() if not k.startswith("_")})
    return cfg


def _run(cfg, steps, fresh):
    if fresh and longrun.CKPT.exists():
        shutil.rmtree(longrun.CKPT)
    longrun.CKPT.mkdir(parents=True, exist_ok=True)
    longrun.LOCKFILE.unlink(missing_ok=True)
    return longrun.train(cfg, max_steps=steps)


def compare(a, b):
    """-> (ok, report). `a` is the uninterrupted checkpoint, `b` the split one."""
    rep, ok = {}, True

    da, db = a["model"], b["model"]
    assert set(da) == set(db)
    worst, worst_k = 0.0, None
    for k in da:
        d = (da[k].float() - db[k].float()).abs().max().item()
        if d > worst:
            worst, worst_k = d, k
    rep["model_max_abs_diff"] = worst
    rep["model_worst_tensor"] = worst_k
    ok &= worst == 0.0

    sa = a["opt"]["state"]
    sb = b["opt"]["state"]
    rep["optimizer_states"] = len(sa)
    ok &= len(sa) == len(sb)
    oworst = 0.0
    for i in sa:
        for f in ("exp_avg", "exp_avg_sq"):
            if f in sa[i] and f in sb[i]:
                oworst = max(oworst, (sa[i][f].float() - sb[i][f].float()).abs().max().item())
        ok &= sa[i].get("step") == sb[i].get("step") if "step" in sa[i] else True
    rep["optimizer_max_abs_diff"] = oworst
    ok &= oworst == 0.0

    rep["streams_match"] = a["streams"] == b["streams"]
    rep["streams_a"] = a["streams"]
    rep["cum_match"] = a["cum"] == b["cum"]
    rep["cum_a"], rep["cum_b"] = a["cum"], b["cum"]
    rep["phase_match"] = a["phase"] == b["phase"]
    rep["step_match"] = a["step"] == b["step"]
    ok &= rep["streams_match"] and rep["cum_match"] and rep["phase_match"] and rep["step_match"]

    la = longrun.lr_at(a["step"], a["cfg"], a["phase"])
    lb = longrun.lr_at(b["step"], b["cfg"], b["phase"])
    rep["next_lr_a"], rep["next_lr_b"] = la, lb
    ok &= la == lb
    return ok, rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n1", type=int, default=12)
    ap.add_argument("--n2", type=int, default=8)
    a = ap.parse_args()
    n = a.n1 + a.n2

    enforce_determinism()

    real_run, real_ckpt = longrun.RUN, longrun.CKPT
    longrun.RUN = SCRATCH
    longrun.CKPT = SCRATCH / "ckpt"
    longrun.HISTORY = SCRATCH / "history.jsonl"
    longrun.LOCKFILE = SCRATCH / "trainer.lock"
    # TERMINAL and HEARTBEAT too: leaving them real meant the test's own registered stop blocked
    # the launch and its heartbeat could confuse a live watchdog.
    longrun.TERMINAL = SCRATCH / "terminal.json"
    longrun.HEARTBEAT = SCRATCH / "heartbeat.json"
    SCRATCH.mkdir(parents=True, exist_ok=True)
    try:
        cfg = scratch_cfg(n)
        t0 = time.time()
        print(f"=== uninterrupted: {n} steps", flush=True)
        _run(cfg, n, fresh=True)
        whole = torch.load(longrun.CKPT / "last.pt", map_location="cpu", weights_only=False)
        shutil.move(str(longrun.CKPT / "last.pt"), str(SCRATCH / "whole.pt"))

        print(f"=== split: {a.n1} steps, resume, {a.n2} more", flush=True)
        _run(cfg, a.n1, fresh=True)
        _run(cfg, n, fresh=False)               # resumes from last.pt
        split = torch.load(longrun.CKPT / "last.pt", map_location="cpu", weights_only=False)

        ok, rep = compare(whole, split)
        rep["n1"], rep["n2"], rep["steps"] = a.n1, a.n2, n
        rep["seconds"] = round(time.time() - t0, 1)
        rep["EQUIVALENT"] = ok
        print(json.dumps(rep, indent=1, default=str))
        print("RESUME EQUIVALENCE: " + ("PASS" if ok else "FAIL"))
        if not ok:
            raise SystemExit(2)
    finally:
        longrun.RUN, longrun.CKPT = real_run, real_ckpt


if __name__ == "__main__":
    main()
