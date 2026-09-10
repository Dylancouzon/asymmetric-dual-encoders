"""The M13 200M BUILD CONTROLLER: one nano build, end to end, and its provenance record.

    .venv/bin/python m13src/build13.py --config m13/build_config.json --plan
    .venv/bin/python m13src/build13.py --config ... --benchmark --rate 910 --price 1.90
    .venv/bin/python m13src/build13.py --config ... [--device cuda] [--resume] [--smoke-steps N]

It COMPOSES; it decides nothing. The recipe comes from `m10/screen_registry.json` through
`m13/build_config.json` (validated by `m13src/build_lock`), the schedule and the stop rules from
`m10src/nano10`, the loop/checkpoint/resume from `m10src/trainer10`, the corpus from
`corpus_loader.assemble_arm` — the ONLY corpus path — and the evaluators, the warm start, the
record writer and the recipe fingerprint from `m10src/run_arm`. What is NEW here, and what
`run_arm` cannot express, is only this:

  * **the 200M pin.** `total_steps = dose // batch` must divide EXACTLY and the three cycles must
    sum to exactly 200,000,000 (`build_lock.check_dose`). A screen arm floors its dose; the build
    does not ("Do not silently train 200.1M", `m10/M102_LOCK.md`).
  * **extension cycles.** After every annealed cycle end k >= 3: kill -> terminal FAILED; gain
    < 0.003 -> PLATEAU freeze; gain >= 0.003 with cap and budget left -> ONE further cycle of
    66,700,000 examples (linear 1e-4 -> 1e-5, `warmup=0` passed explicitly, warm-loaded from the
    previous cycle's retained checkpoint, its data position continued from the GLOBAL example
    count); else CAPPED freeze.
  * **`max_extension_cycles`**, which is not a constant: it is fixed at the day-one benchmark from
    the BILLED $/h and the MEASURED examples/s, written into `state.json`, and refused if absent.
  * **spend accounting** against the recorded $1,000 ceiling: a cycle whose projected cost plus
    the spend to date would exceed it does not start.
  * **a wall-clock rolling checkpoint** (default 30 min, converted to a step interval with the
    benchmark's rate), because `total // 20` is ~10 GPU-hours of exposure at this dose, and a
    `losses.jsonl` sidecar, because a 6.25M-element loss list in every checkpoint is not free.
  * **the freeze**: DEV-6 once, `nano10.export_onnx`, ORT and fastembed parity, and
    `results/m13_build_record.json`.

**Refusals** mirror `run_arm`'s, and for the same reason — a registered run reads its recipe from
the registry or does not run: `--compile` and every recipe knob are `--smoke-steps` only; non-CUDA
outside a smoke; an invalid screen lock; a PENDING E1 batch; a missing LoTTE gate record; a
missing cap; a dose that is not the registered 200M or does not divide by the batch; a terminal
record that already exists.

On disk: `work/m13build/<arm>/{state.json, receipt.json, ckpt.pt, cycle{k}.pt, ext{j}.pt,
cov_*.json, losses.jsonl, onnx/}` (`work/m13build/smoke/<arm>/` under `--smoke-steps`).
No six-set, reserved or LoTTE surface is touched anywhere in this file.
"""
import argparse
import hashlib
import json
import math
import os
import time
import traceback
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for p in ("m7src", "m9src", "m10src", "m13src"):
    if str(REPO / p) not in sys.path:
        sys.path.insert(0, str(REPO / p))

import torch                                    # noqa: E402

import build_lock as BL                         # noqa: E402
import corpus_loader as CL                      # noqa: E402
import data10 as D                              # noqa: E402
import nano10 as N                              # noqa: E402
import run_arm as R                             # noqa: E402
import screen_lock as SL                        # noqa: E402
import trainer10 as Tr                          # noqa: E402

WORK = REPO / "work" / "m13build"
SMOKE_WORK = WORK / "smoke"
RESULTS = REPO / "results"
RECORD_NAME = "m13_build_record.json"
CYCLES = BL.CYCLES
PEAK, FINAL = R.PEAK, R.FINAL
MAX_LEN = R.MAX_LEN
DEFAULT_CKPT_MINUTES = 30
# Every knob that would change the registered recipe from the command line, exactly as
# `run_arm.SMOKE_ONLY_KNOBS` does. `--batch` joins them because the batch is the E1 verdict's,
# and `--lotte-gate` because the gate record is stage 3's artifact and not a launcher's argument.
SMOKE_ONLY_KNOBS = tuple(R.SMOKE_ONLY_KNOBS) + ("batch", "lotte_gate")

refuse = R.refuse
sha256_file = R.sha256_file
rel = R.rel


# ------------------------------------------------------------------------------ paths, state ----

def run_paths(arm, smoke):
    """-> (out_dir, receipt path, record path, published results path or None). A smoke gets its
    OWN tree (§Hazards: smokes have overwritten real artifacts in this milestone twice)."""
    d = (SMOKE_WORK if smoke else WORK) / R.slug(arm)
    return d, d / "receipt.json", d / "record.json", (None if smoke else RESULTS / RECORD_NAME)


def write_json(path, obj):
    """Atomic, fsynced. `state.json` is read back by a resume; a truncated one is a lost run."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f".tmp{os.getpid()}")
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1, default=str)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, p)
    return p


def read_json(path):
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


# ------------------------------------------------------------------------------- the schedule ----

def continued_batch_fn(q_stream, d_stream, pattern, q_offset=0, d_offset=0):
    """`data10.batch_fn` with the stream positions CONTINUED from a global offset.

    An extension cycle is a fresh `train_arm` whose local step starts at 0, but its data must go
    on where the build left off (`m13/STAGE1_DESIGN.md` §1, `m10/CODEMAP.md` pitfall 11). The
    offsets are per-stream batch counts, not steps, so no alignment with the 4-step mix window is
    required — the local step still decides the KIND (the extension's own mix is exactly the
    arm's), and the position inside each stream is `already consumed + kind_index(local step)`.
    With both offsets 0 this is `data10.batch_fn` exactly.
    """
    def f(step, kind):
        want = N.mix_window(pattern, step)
        if kind != want:
            raise ValueError(f"step {step} is a {want!r} step under pattern {pattern!r}, asked "
                             f"for {kind!r}: batch_fn's pattern and the loop's disagree")
        s, off = (q_stream, q_offset) if kind == "Q" else (d_stream, d_offset)
        return s.batch(off + D.kind_index(pattern, step))
    return f


def stream_offsets(pattern, global_steps):
    """-> (query batches, document batches) consumed by `global_steps` steps of `pattern`."""
    sh = N.window_shares(pattern, int(global_steps)) if global_steps else {"Q": 0, "D": 0}
    return int(sh["Q"]), int(sh["D"])


def build_plan(cfg, reg, batch, *, smoke_dose=None):
    """Everything `--plan` prints and the controller needs, all of it from the registry and the
    validated configuration. Shaped like `run_arm.arm_plan`'s output so `run_arm.fingerprint`
    reads it unchanged."""
    arm = cfg["arm"]
    entry = dict((reg.get("arms") or {}).get(arm) or {})
    k = cfg["data_knobs_equal_to_anchor"]
    pattern, pattern_rep = CL.resolve_arm_pattern(arm, entry, reg)
    dose = int(smoke_dose if smoke_dose is not None else entry["dose_examples"])
    total = dose // int(batch)
    return {"arm": arm, "family": entry.get("family"), "dose_examples": dose, "batch": int(batch),
            "pattern": pattern, "pattern_source": pattern_rep,
            "student": k["student"], "n_layers": int(k["feature_layers"]), "head": k["head"],
            "objective": k["objective"], "warm_start": k["warm_start"]["kind"],
            "cut_corpus": bool(CL.is_cut_corpus(arm, reg)),
            "sources": list(CL.arm_sources(arm, reg)),
            "n_docs": CL.arm_doc_count(entry, pattern, int(batch)),
            "documents": CL.arm_document_policy(entry),
            "total_steps": total, "cycle_end_steps": N.cycle_ends(total, CYCLES),
            "seed": int(entry.get("seed", 0))}


def fingerprint(plan, man, seed, smoke_steps, device, phase, cfg_sha):
    """`run_arm.fingerprint` (arm, recipe, registry, corpus manifest, device, optimizer, code)
    plus the build's own identity: the configuration's bytes, `build13.py`'s bytes, and the PHASE
    — so an extension's checkpoint can never satisfy the base schedule's resume, or another
    extension's."""
    base = R.fingerprint(plan["arm"], plan, man, seed, smoke_steps, device)
    body = {"base": base, "phase": phase, "config_sha256": cfg_sha,
            "build13_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "build_lock_sha256": sha256_file(REPO / "m13src" / "build_lock.py")}
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()


# ---------------------------------------------------------------------------- the day-one cap ----

def benchmark(cfg, out_dir, rate, price, *, state=None, note=None, smoke=False):
    """Fix `max_extension_cycles` from the MEASURED examples/s and the BILLED $/h, and write it
    into `state.json` before cycle 1. Refused later if absent — the cap is not a constant."""
    cap = BL.cap_arithmetic(cfg, rate, price, smoke=smoke)
    st = dict(state or {})
    st["benchmark"] = {**cap, "when": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "note": note}
    st["max_extension_cycles"] = int(cap["max_extension_cycles"])
    st["spent_usd"] = float(cap["committed_usd"])
    st["_spent_usd"] = ("the mandatory lines and the fixed disk/egress, committed at the "
                        "benchmark; every extension cycle adds its billed cost")
    write_json(out_dir / "state.json", st)
    return st


def print_plan(cfg, batch, rate=None, price=None):
    """The cycle table and the cap arithmetic. Writes nothing, trains nothing, rents nothing."""
    print(f"build arm {cfg['arm']}  dose {cfg['arm_entry']['dose_examples']:,}  "
          f"batch {batch if batch else 'PENDING (E1 unread)'}")
    batches = [batch] if batch else [32, 128]
    for b in batches:
        p = BL.cycle_plan(cfg["arm_entry"]["dose_examples"], b)
        e = BL.extension_plan(cfg["extension_examples"], b)
        print(f"\n  bs{b}: {p['total_steps']:,} steps, divides exactly: {p['divides_exactly']}")
        for row in p["cycles"]:
            print(f"    cycle {row['cycle']}  steps {row['steps']:>9,}  "
                  f"[{row['start_step']:>9,}..{row['end_step']:>9,}]  "
                  f"examples {row['examples']:>12,}")
        print(f"    sum {p['sum_examples']:,} (= dose: {p['sum_examples'] == p['dose_examples']})")
        print(f"    extension cycle: {e['steps']:,} steps = {e['examples']:,} examples "
              f"(dropped {e['examples_dropped']})")
        rr = rate or R.PLAN_RATES.get(b)
        for pr in ([price] if price else [1.5, 2.5]):
            try:
                cap = BL.cap_arithmetic(cfg, rr, pr)
            except SystemExit as exc:
                print(f"    cap at {rr:.0f} ex/s, ${pr}/h: {exc}")
                continue
            print(f"    cap at {rr:.0f} ex/s, ${pr}/h: mandatory {cap['mandatory_hours_total']} h "
                  f"= ${cap['mandatory_usd']}, +${cap['fixed_usd']} fixed, "
                  f"extension ${cap['extension_cycle_usd']}/cycle -> "
                  f"max_extension_cycles {cap['max_extension_cycles']}")
        if not (rate and price):
            print("    (a PROJECTION: run_arm.PLAN_RATES and assumed prices. The cap is fixed at "
                  "the day-one benchmark from the measured rate and the billed price.)")
    return 0


# ------------------------------------------------------------------------------- the evaluator ----

class BuildEval:
    """`run_arm.CovEval` (or `StubEval` in a smoke) plus the build's own bookkeeping.

    The evaluation history is what the extension, plateau and kill rules read, and it has to
    survive both an abrupt process loss BETWEEN phases (`state.json` carries it) and a resume
    INSIDE one (the checkpoint carries it, through `trainer10`'s `eval_state` hook). The
    checkpointed copy is authoritative on `restore`, so replaying a few steps after a crash
    cannot append the same evaluation twice — the state file is rewritten from it.
    """

    def __init__(self, inner, state_path, state):
        self.inner, self.state_path, self.st = inner, state_path, state
        self.evals = list(state.get("evals") or [])
        self.kinds = list(state.get("eval_kinds") or [])
        self.labels = list(state.get("eval_labels") or [])
        saved = state.get("eval_state")
        if saved:
            self.inner.restore(saved.get("inner") or {})

    @property
    def records(self):
        return self.inner.records

    def __call__(self, step, label, kind):
        m = self.inner(step, label)
        self.evals.append(float(m))
        self.kinds.append(kind)
        self.labels.append(label)
        self._sync()
        return m

    def _sync(self):
        self.st["evals"] = list(self.evals)
        self.st["eval_kinds"] = list(self.kinds)
        self.st["eval_labels"] = list(self.labels)
        self.st["cycle_macros"] = [m for m, k in zip(self.evals, self.kinds) if k == "end"]
        self.st["eval_state"] = self.state_payload()
        write_json(self.state_path, self.st)

    # the trainer checkpoints the evaluator's own state beside its own (finding 7)
    def state_payload(self):
        return {"inner": self.inner.state(), "evals": list(self.evals),
                "kinds": list(self.kinds), "labels": list(self.labels)}

    def state(self):
        return self.state_payload()

    def restore(self, d):
        self.inner.restore(d.get("inner") or {})
        self.evals = list(d.get("evals") or [])
        self.kinds = list(d.get("kinds") or [])
        self.labels = list(d.get("labels") or [])
        self._sync()


# ------------------------------------------------------------------------------------ the run ----

def _record_failure(ctx, exc, verbose=True):
    """A crash, an OOM or a registered kill is an OUTCOME (`rules.arm_failure`), so it is
    recorded — with NO final checkpoint — and the exception re-raised."""
    if ctx.get("record_path") is None:
        return None
    rec = {"_what": "the M13 200M nano build, FAILED, recorded by m13src/build13.py",
           "arm": ctx.get("arm"), "status": "failed", "complete": False, "terminal": True,
           "smoke": bool(ctx.get("smoke")),
           "stopped": f"{type(exc).__name__}: {exc}",
           "failure": {"type": type(exc).__name__, "message": str(exc)[:2000],
                       "traceback_tail": traceback.format_exc()[-4000:]},
           "state": ctx.get("state"), "recipe": ctx.get("plan"),
           "final_checkpoint": None, "final_checkpoint_sha256": None,
           "_final_checkpoint_note": "a failed build has NO final checkpoint; any cycle or "
                                     "extension checkpoint on disk is a partial",
           "registry_sha256": ctx.get("registry_sha256"), "git_head": R.git_head(),
           "_rule": "rules.arm_failure — reported, never silently re-run"}
    try:
        R.write_record(rec, ctx["record_path"], ctx.get("results_path"))
        write_json(ctx["receipt_path"], {**(read_json(ctx["receipt_path"]) or {}),
                                         "status": "failed", "terminal": True, "complete": False})
        if verbose:
            print(f"BUILD FAILED ({type(exc).__name__}): wrote {ctx['record_path']}", flush=True)
    except Exception as e:                                              # pragma: no cover
        print(f"could not write the failure record: {type(e).__name__}: {e}", flush=True)
    return rec


def run(config=None, *, device="cuda", resume=False, smoke_steps=None, batch=None, max_len=None,
        ckpt_every=None, ckpt_minutes=DEFAULT_CKPT_MINUTES, n_fit=None, compile_step=False,
        real_eval=False, lotte_gate=None, verbose=True, benchmark_rate=None,
        benchmark_price=None, benchmark_only=False):
    ctx = {"started": False}
    try:
        return _run(ctx, config, device=device, resume=resume, smoke_steps=smoke_steps,
                    batch=batch, max_len=max_len, ckpt_every=ckpt_every,
                    ckpt_minutes=ckpt_minutes, n_fit=n_fit, compile_step=compile_step,
                    real_eval=real_eval, lotte_gate=lotte_gate, verbose=verbose,
                    benchmark_rate=benchmark_rate, benchmark_price=benchmark_price,
                    benchmark_only=benchmark_only)
    except SystemExit:
        if ctx.get("started"):
            _record_failure(ctx, sys.exc_info()[1], verbose=verbose)
        raise
    except BaseException as e:                                          # noqa: BLE001
        _record_failure(ctx, e, verbose=verbose)
        raise


def _preflight(ctx, cfg, cfg_path, *, smoke, device, batch, lotte_gate, compile_step, real_eval,
               max_len, ckpt_every, n_fit):
    """Everything that must be true before a GPU-hour is spent. Every failure here writes
    NOTHING: the build never started."""
    passed = {"max_len": max_len, "ckpt_every": ckpt_every, "n_fit": n_fit,
              "compile_step": compile_step or None, "real_eval": real_eval or None,
              "batch": batch, "lotte_gate": lotte_gate}
    given = sorted(k for k in SMOKE_ONLY_KNOBS if passed.get(k))
    if given and not smoke:
        refuse(f"{given} may only be passed with --smoke-steps: the build's recipe comes from "
               f"m10/screen_registry.json and m13/build_config.json, not from the command line")
    if not smoke and device != "cuda":
        refuse(f"a registered build requires --device cuda (got {device!r}) — the recipe is bf16 "
               f"autocast and CPU silently means fp32. Pass --smoke-steps for a path check.")
    reg = SL.cfg()
    report = BL.validate(cfg, reg, smoke=smoke)
    b, batch_source = BL.resolve_batch(cfg, smoke=smoke, override=batch)
    gate_path = Path(lotte_gate or (REPO / cfg["lotte_gate"]))
    gate = read_json(gate_path)
    if gate is None:
        refuse(f"no LoTTE gate record at {gate_path}. The pre-build LoTTE handling is registered "
               f"before the build (m10/M102_LOCK.md 'Before training' 3); the record may say "
               f"`skipped`, but it must exist — a bs128 selection can still be vetoed.")
    ctx["registry_sha256"] = report["registry_sha256"]
    return reg, b, batch_source, {"path": rel(gate_path), "sha256": sha256_file(gate_path),
                                  "decision": gate.get("decision", gate.get("status"))}, report


def _run(ctx, config, *, device, resume, smoke_steps, batch, max_len, ckpt_every, ckpt_minutes,
         n_fit, compile_step, real_eval, lotte_gate, verbose, benchmark_rate, benchmark_price,
         benchmark_only):
    smoke = smoke_steps is not None
    cfg, cfg_path = BL.load(config)
    cfg_sha = sha256_file(cfg_path)
    reg, batch, batch_source, gate, report = _preflight(
        ctx, cfg, cfg_path, smoke=smoke, device=device, batch=batch, lotte_gate=lotte_gate,
        compile_step=compile_step, real_eval=real_eval, max_len=max_len, ckpt_every=ckpt_every,
        n_fit=n_fit)
    arm = cfg["arm"]
    ctx["arm"] = arm
    ctx["smoke"] = smoke
    max_len = int(max_len or MAX_LEN)

    # the 200M pin (or the smoke's own tiny dose, which gets its own tree and its own record)
    if smoke:
        dose = int(smoke_steps) * int(batch)
        smoke_docs = max(int(math.ceil(dose * CL.DOC_SHARE["75/25"])), int(batch))
    else:
        dose, smoke_docs = int(cfg["arm_entry"]["dose_examples"]), None
        BL.check_dose(cfg, batch)
    merged = BL.merged_registry(cfg, reg, batch=batch, smoke_documents=smoke_docs)
    if smoke:
        merged["arms"][arm]["dose_examples"] = dose
    plan = build_plan(cfg, merged, batch, smoke_dose=dose if smoke else None)
    ctx["plan"] = plan
    seed = plan["seed"]

    out_dir, receipt_path, record_path, results_path = run_paths(arm, smoke)
    ctx.update(receipt_path=receipt_path, record_path=record_path, results_path=results_path)
    # `rules.arm_failure`: a build is reported, never silently re-run. Any record at either
    # published path blocks a bare re-run; a TERMINAL one blocks `--resume` too.
    for path in (receipt_path, record_path, results_path):
        st = R._record_status_at(path)
        if st is None:
            continue
        if not resume and not (benchmark_only and path is receipt_path):
            refuse(f"{path} already exists. Pass --resume to continue a non-terminal run, or "
                   f"move the record aside deliberately.")
        if st["parseable"] and st["terminal"]:
            refuse(f"{path} already carries a TERMINAL record (complete or failed); there is "
                   f"nothing to continue. Move the record aside deliberately.")
        if not st["parseable"]:
            refuse(f"{path} is unparseable, so whether the build already finished is UNKNOWABLE.")

    state = read_json(out_dir / "state.json") or {}
    if benchmark_only:
        if benchmark_rate is None or benchmark_price is None:
            refuse("--benchmark needs --rate EX_S and --price USD_H: the cap is fixed from the "
                   "MEASURED rate and the BILLED price, never from a projection")
        st = benchmark(cfg, out_dir, benchmark_rate, benchmark_price, state=state,
                       note="smoke" if smoke else None, smoke=smoke)
        print(json.dumps(st["benchmark"], indent=1))
        return st
    if state.get("max_extension_cycles") is None:
        refuse(f"{out_dir / 'state.json'} carries no `max_extension_cycles`. It is fixed at the "
               f"day-one benchmark from the billed price and the measured rate, not now: run "
               f"`--benchmark --rate EX_S --price USD_H` first.")

    if resume:
        if R._record_status_at(receipt_path) is None:
            refuse(f"--resume: no receipt at {receipt_path}; there is no run to continue.")
    elif (out_dir / "ckpt.pt").exists():
        refuse(f"{out_dir / 'ckpt.pt'} already exists without a running receipt; refusing to "
               f"overwrite an orphaned checkpoint with a fresh run.")

    t_start = time.time()
    print(f"=== build {arm} on {device}: {dose:,} examples, batch {batch}, "
          f"{plan['total_steps']:,} steps, pattern {plan['pattern']}, student {plan['student']}, "
          f"objective {plan['objective']}, seed {seed}, cap "
          f"{state['max_extension_cycles']} extension cycle(s)"
          + (f"  [SMOKE {smoke_steps} steps]" if smoke else ""), flush=True)

    torch.manual_seed(seed)
    model = N.Nano10(plan["student"], n_layers=plan["n_layers"], head=plan["head"]).to(device)
    if not model.under_cap():
        refuse(f"{arm}: {model.n_params():,} parameters exceeds the {N.CAP:,} cap")
    print(f"  student {plan['student']}: d_in {model.d_in}, {model.n_params():,} params",
          flush=True)

    base_fn, man = CL.assemble_arm(arm, model.tok, plan["student"], batch_size=batch, seed=seed,
                                   max_len=max_len, verbose=verbose, registry=merged)
    q_stream, d_stream = R.streams_of(base_fn)
    fp = fingerprint(plan, man, seed, smoke_steps, device, "cycles1-3", cfg_sha)

    identity = {"arm": arm, "smoke": bool(smoke), "recipe_fingerprint": fp,
                "config_sha256": cfg_sha, "registry_sha256": ctx["registry_sha256"]}
    if resume:
        rec = read_json(receipt_path) or {}
        if rec.get("status") != "running" or any(rec.get(k) != v for k, v in identity.items()):
            refuse(f"--resume: {receipt_path} has missing or mismatched running identity; "
                   f"refusing to continue another recipe or an unauthenticated record.")
    else:
        try:
            R.write_record({"_what": "M13 build started; abrupt process loss may resume it",
                            **identity, "status": "running", "complete": False,
                            "terminal": False, "git_head": R.git_head()},
                           receipt_path, None, create_only=True)
        except FileExistsError:
            refuse(f"{receipt_path} already exists; refusing to replace another run's receipt.")

    ctx["started"] = True
    state.update({"arm": arm, "smoke": bool(smoke), "config_sha256": cfg_sha,
                  "registry_sha256": ctx["registry_sha256"], "batch": batch,
                  "batch_source": batch_source, "lotte_gate": gate,
                  "dose_examples": dose, "total_steps": plan["total_steps"],
                  "cycle_end_steps": plan["cycle_end_steps"],
                  "recipe_fingerprint": fp, "validation": report})
    state.setdefault("cycle_macros", [])
    state.setdefault("evals", [])
    state.setdefault("eval_kinds", [])
    state.setdefault("extensions", [])
    state.setdefault("decisions", [])
    ctx["state"] = state
    write_json(out_dir / "state.json", state)

    ws_cfg = {"n_fit": int(n_fit or (R.SMOKE_N_FIT if smoke
                                     else cfg["data_knobs_equal_to_anchor"]["warm_start"]["n_fit"])),
              "seed": int(cfg["data_knobs_equal_to_anchor"]["warm_start"]["fit_seed"])}
    ws = R.warm_start(model, {"warm_start": plan["warm_start"]}, q_stream, ws_cfg, verbose=verbose)

    inner = R.StubEval() if smoke else R.CovEval(model, out_dir, verbose=verbose)
    ev = BuildEval(inner, out_dir / "state.json", state)
    ctx["cov_records"] = ev.records
    # ONE sidecar per PHASE: a fresh `train_arm` truncates the log it is given (so a resume does
    # not double-log), which would erase cycles 1-3's losses when extension 1 started.
    loss_log = out_dir / "losses.jsonl"

    def ext_loss_log(j):
        return out_dir / f"losses_ext{j}.jsonl"

    def cadence(total):
        """The rolling checkpoint interval in STEPS, from the wall-clock cadence and the
        benchmark's measured rate. Without a rate it falls back to `run_arm`'s `total // 20`,
        which at the build's dose is ~10 GPU-hours of exposure — hence the cadence."""
        if ckpt_every:
            return int(ckpt_every)
        rate = ((state.get("benchmark") or {}).get("rate_ex_per_s"))
        if rate:
            return max(int(float(rate) * 60 * float(ckpt_minutes) / batch), 1)
        return max(total // 20, 1)

    # ---- cycles 1..3, ONE schedule -------------------------------------------------------
    ends = plan["cycle_end_steps"]

    def base_eval(_m, step, kind):
        label = f"cycle{ends.index(step) + 1}" if kind == "end" else f"mid{step}"
        return ev(step, label, kind)

    base_ck = out_dir / "ckpt.pt"
    base_done = bool(state.get("base_done"))
    train_model = torch.compile(model) if compile_step else model
    if not base_done:
        r = Tr.train_arm(train_model, base_fn, total_steps=plan["total_steps"],
                         pattern=plan["pattern"], cycles=CYCLES, peak=PEAK, final=FINAL,
                         loss_name=plan["objective"], eval_fn=base_eval, ckpt_path=base_ck,
                         ckpt_every=cadence(plan["total_steps"]),
                         resume_from=(str(base_ck) if resume and base_ck.exists() else None),
                         seed=seed, log_every=max(plan["total_steps"] // 50, 1), device=device,
                         batch_size=batch, cycle_ckpt_fmt=str(out_dir / "cycle{cycle}.pt"),
                         eval_state=ev, fingerprint=fp, loss_log=str(loss_log))
        state["base"] = {k: r[k] for k in ("steps_run", "start_step", "total_steps", "stopped",
                                           "examples", "seconds", "examples_per_s", "mix",
                                           "cycle_end_evals", "n_losses", "grad_norm")}
        state["base_done"] = True
        state["examples_run"] = int(r["examples"])
        state["steps_run"] = int(plan["total_steps"])
        state["stopped"] = r["stopped"]
        write_json(out_dir / "state.json", state)
    else:
        r = state["base"]

    # ---- the extension / plateau / kill loop ---------------------------------------------
    # A SMOKE's extension is the SMOKE's dose. The registered 66,700,000 is a cloud cycle
    # (~30 GPU-hours); a path check that started one would be a runaway, which is exactly what the
    # first box smoke of this controller did before this line existed.
    ext_examples = BL.extension_plan(dose if smoke else cfg["extension_examples"], batch)
    min_gain = float(cfg["extension_min_gain"])
    cap = int(state["max_extension_cycles"])
    ceiling = float(cfg["budget_ceiling_usd"])
    cycle_usd = float((state.get("benchmark") or {}).get("extension_cycle_usd") or 0.0)
    outcome, final_ck = None, None
    while True:
        fired, why = N.kill_fires(state["evals"], lambda i: state["eval_kinds"][i])
        if fired or (state.get("stopped") and not str(state["stopped"]).startswith("plateau")):
            outcome = {"outcome": "FAILED",
                       "why": f"kill: {why}" if fired else str(state.get("stopped"))}
            break
        macros = state["cycle_macros"]
        if len(macros) < CYCLES:
            outcome = {"outcome": "FAILED",
                       "why": f"only {len(macros)} annealed cycle end(s) were read"}
            break
        gain = macros[-1] - max(macros[:-1])
        n_ext = len(state["extensions"])
        projected = state.get("spent_usd", 0.0) + cycle_usd
        decision = {"after_cycle": len(macros), "macro": macros[-1],
                    "best_previous": max(macros[:-1]), "gain": round(gain, 6),
                    "min_gain": min_gain, "extensions_run": n_ext, "max_extension_cycles": cap,
                    "spent_usd": round(state.get("spent_usd", 0.0), 2),
                    "projected_cycle_usd": round(cycle_usd, 2),
                    "budget_ceiling_usd": ceiling}
        if gain < min_gain:
            decision["decision"] = "PLATEAU"
            outcome = {"outcome": "PLATEAU", "why": f"gain {gain:.6f} < {min_gain} at cycle "
                                                    f"{len(macros)}"}
        elif n_ext >= cap:
            decision["decision"] = "CAPPED"
            outcome = {"outcome": "CAPPED", "why": f"{n_ext} extension cycle(s) run, cap {cap}"}
        elif projected > ceiling:
            decision["decision"] = "CAPPED_BUDGET"
            outcome = {"outcome": "CAPPED", "why": f"${projected:.2f} projected exceeds the "
                                                   f"${ceiling:.0f} ceiling"}
        else:
            decision["decision"] = "EXTEND"
        # a resume re-derives the decision it already recorded; record it once (the inputs are a
        # pure function of the macro history and the spend, so a repeat is the same decision).
        if not any(d["after_cycle"] == decision["after_cycle"]
                   and d["extensions_run"] == decision["extensions_run"]
                   for d in state["decisions"]):
            state["decisions"].append(decision)
        write_json(out_dir / "state.json", state)
        if decision["decision"] != "EXTEND":
            break
        j = n_ext + 1
        from_ck = out_dir / (f"ext{n_ext}.pt" if n_ext else f"cycle{CYCLES}.pt")
        if not from_ck.exists():
            refuse(f"an extension cycle warm-loads from {from_ck}, which does not exist")
        global_steps = int(state["steps_run"])
        q_off, d_off = stream_offsets(plan["pattern"], global_steps)
        ext_fn = continued_batch_fn(q_stream, d_stream, plan["pattern"], q_off, d_off)
        fp_ext = fingerprint(plan, man, seed, smoke_steps, device,
                             {"extension": j, "steps": ext_examples["steps"],
                              "from": from_ck.name, "warmup_steps": 0,
                              "global_steps_before": global_steps}, cfg_sha)
        ext_ck = out_dir / f"ext{j}_ckpt.pt"
        in_flight = ext_ck.exists() and torch.load(
            ext_ck, map_location="cpu", weights_only=False)["extra"].get("fingerprint") == fp_ext
        if not in_flight:
            # warm-load the previous cycle end into the model; the fresh 1-cycle schedule then
            # owns the optimizer, exactly as a new cycle does.
            ck = torch.load(from_ck, map_location="cpu", weights_only=False)
            Tr.eager(model).load_state_dict(ck["model"])
        print(f"  extension {j}: {ext_examples['steps']:,} steps = {ext_examples['examples']:,} "
              f"examples from {from_ck.name}, data continued at step {global_steps:,}", flush=True)

        def ext_eval(_m, step, kind, _j=j):
            return ev(step, f"ext{_j}" if kind == "end" else f"ext{_j}mid{step}", kind)

        re = Tr.train_arm(train_model, ext_fn, total_steps=ext_examples["steps"],
                          pattern=plan["pattern"], cycles=1, peak=PEAK, final=FINAL,
                          loss_name=plan["objective"], eval_fn=ext_eval, ckpt_path=ext_ck,
                          ckpt_every=cadence(ext_examples["steps"]),
                          resume_from=(str(ext_ck) if in_flight else None), seed=seed,
                          log_every=max(ext_examples["steps"] // 50, 1), device=device,
                          batch_size=batch, cycle_ckpt_fmt=str(out_dir / f"ext{j}.pt"),
                          eval_state=ev, fingerprint=fp_ext, loss_log=str(ext_loss_log(j)),
                          warmup_steps=0)
        state["extensions"].append({"j": j, "from": from_ck.name, **ext_examples,
                                    "stopped": re["stopped"], "examples": int(re["examples"]),
                                    "start_step": int(re["start_step"]),
                                    "macro": (state["cycle_macros"][-1]
                                              if state["cycle_macros"] else None),
                                    "q_offset": q_off, "d_offset": d_off,
                                    "examples_per_s": re["examples_per_s"]})
        state["steps_run"] = global_steps + ext_examples["steps"]
        state["examples_run"] = int(state.get("examples_run", 0)) + ext_examples["examples"]
        state["spent_usd"] = round(state.get("spent_usd", 0.0) + cycle_usd, 4)
        state["stopped"] = re["stopped"]
        write_json(out_dir / "state.json", state)

    state["outcome"] = outcome
    write_json(out_dir / "state.json", state)
    ok = outcome["outcome"] in ("PLATEAU", "CAPPED")
    if ok:
        final_ck = out_dir / (f"ext{len(state['extensions'])}.pt" if state["extensions"]
                              else f"cycle{CYCLES}.pt")
        if not final_ck.exists():
            ok, outcome = False, {"outcome": "FAILED", "why": f"no final checkpoint at {final_ck}"}

    # ---- freeze --------------------------------------------------------------------------
    freeze = {}
    if ok:
        freeze = freeze_checkpoint(model, out_dir, final_ck, smoke=smoke, verbose=verbose)
    else:
        print(f"BUILD FAILED ({outcome['why']}): no final checkpoint, so no DEV-6, no export and "
              f"no parity. `rules.arm_failure`: reported, not re-run at different settings.",
              flush=True)

    cks = {}
    for pth in sorted(out_dir.glob("*.pt")):
        cks[pth.stem] = {"path": rel(pth), "sha256": sha256_file(pth)}
    rec = {
        "_what": "the M13 200M nano build, trained end to end by m13src/build13.py",
        "arm": arm, "status": ("complete" if ok else "failed"), "complete": ok, "terminal": True,
        "outcome": outcome, "smoke": smoke, "smoke_steps": smoke_steps, "device": device,
        "max_len": max_len, "seed": seed,
        "config": rel(cfg_path), "config_sha256": cfg_sha, "validation": report,
        "lotte_gate": gate, "batch": batch, "batch_source": batch_source,
        "recipe": plan, "warm_start": ws,
        "schedule": {"cycles": CYCLES, "peak": PEAK, "final": FINAL,
                     "cycle_plan": BL.cycle_plan(dose, batch),
                     "extension": ext_examples,
                     "warmup_examples": int(N.WARMUP_EXAMPLES),
                     "warmup_steps_cycle1": int(N.warmup_steps_for(batch)),
                     "extension_warmup_steps": 0},
        "params": model.n_params(), "under_cap": model.under_cap(),
        "evaluation_mode": "stub" if smoke else "cov",
        "cov": {"per_checkpoint": ev.records, "cycle_macros": state["cycle_macros"],
                "final_macro": (state["cycle_macros"][-1] if state["cycle_macros"] else None),
                "_scope": "COV only, at every annealed cycle end and the schedule midpoints the "
                          "kill rule reads. No six-set, reserved or LoTTE surface."},
        "extensions": state["extensions"], "decisions": state["decisions"],
        "budget": {"benchmark": state.get("benchmark"), "spent_usd": state.get("spent_usd"),
                   "ceiling_usd": ceiling,
                   "max_extension_cycles": state["max_extension_cycles"]},
        "dose_run_examples": state.get("examples_run"), "steps_run": state.get("steps_run"),
        "training": state.get("base"), "loss_logs": [rel(p) for p in sorted(out_dir.glob("losses*.jsonl"))],
        "checkpoints": cks,
        "final_checkpoint": rel(final_ck) if ok else None,
        "final_checkpoint_sha256": (sha256_file(final_ck) if ok else None),
        "freeze": freeze,
        "recipe_fingerprint": fp,
        "assemble_manifest": man,
        "data_manifest_sha256": (sha256_file(RESULTS / "m10_data_manifest.json")
                                 if (RESULTS / "m10_data_manifest.json").exists() else None),
        "registry_sha256": ctx["registry_sha256"],
        "environment": environment(device),
        "git_head": R.git_head(),
        "wall_seconds": round(time.time() - t_start, 1),
    }
    for w in R.write_record(rec, record_path, results_path):
        print(f"wrote {w}", flush=True)
    write_json(receipt_path, {**(read_json(receipt_path) or {}), "status": rec["status"],
                              "complete": ok, "terminal": True})
    print(f"{arm}: {rec['status']} ({outcome['outcome']})  COV final {rec['cov']['final_macro']}  "
          f"{rec['wall_seconds']:.0f}s", flush=True)
    return rec


def environment(device):
    env = {"torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
           "device": str(device), "cuda": getattr(torch.version, "cuda", None)}
    try:
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
            env["gpu_total_bytes"] = int(torch.cuda.get_device_properties(0).total_memory)
    except Exception as e:                                              # pragma: no cover
        env["gpu"] = f"unavailable: {type(e).__name__}"
    return env


def freeze_checkpoint(model, out_dir, final_ck, *, smoke=False, verbose=True):
    """DEV-6 once, the ONNX export, and the serving-parity reads. A smoke skips DEV-6 (~13 GB of
    reads) exactly as `run_arm`'s does, and says so in the record."""
    ck = torch.load(final_ck, map_location="cpu", weights_only=False)
    Tr.eager(model).load_state_dict(ck["model"])
    out = {"from_checkpoint": rel(final_ck), "dev6": None, "onnx": None, "ort_parity": None,
           "fastembed_parity": None}
    if smoke:
        out["dev6"] = {"skipped": "a smoke never reads DEV-6 (run_arm.dev6's ~13 GB)"}
    else:
        out["dev6"] = R.dev6(model, verbose=verbose)          # on the training device
    d = out_dir / "onnx"
    try:
        # Export, parity and encoding run EAGER and on CPU (`m10/HEADROOM.md` §T). `export_onnx`
        # builds its trace inputs on the CPU, so a CUDA model raises "Expected all tensors to be
        # on the same device" — which is how the first box smoke of this controller ended.
        m = Tr.eager(model).to("cpu")
        out["onnx"] = N.export_onnx(m, d, max_len=MAX_LEN)
        out["onnx"]["sha256"] = sha256_file(d / "model.onnx")
        texts = ["a short query", "what is the capital of france", "a " * 300,
                 "The quick brown fox jumps over the lazy dog."]
        out["ort_parity"] = N.export_parity(m, d, texts)
        out["fastembed_parity"] = fastembed_parity(m, d, texts)
    except Exception as e:                                              # pragma: no cover
        out["export_error"] = f"{type(e).__name__}: {e}"
    return out


def fastembed_parity(model, out_dir, texts, name="qdrant/nano-m13-build"):
    """min-cos between the training form and fastembed's own serving of the export. Reported
    `served: False` with the error when fastembed is unavailable — the ORT parity above is the
    one this build always has (`m10/CODEMAP.md` pitfalls 2 and 5)."""
    d = Path(out_dir)
    try:
        import numpy as np
        (d / "special_tokens_map.json").write_text(json.dumps(
            {k: (v if isinstance(v, str) else str(v))
             for k, v in model.tok.special_tokens_map.items()}, indent=1))
        from fastembed import TextEmbedding
        from fastembed.common.model_description import ModelSource, PoolingType
        TextEmbedding.add_custom_model(model=name, pooling=PoolingType.MEAN, normalization=True,
                                       sources=ModelSource(hf=name), dim=N.OUT_DIM,
                                       model_file="model.onnx",
                                       description="M13 nano build serving parity",
                                       license="mit",
                                       size_in_gb=round((d / "model.onnx").stat().st_size / 1e9, 4))
        fe = TextEmbedding(model_name=name, specific_model_path=str(d), threads=4)
        got = np.stack(list(fe.embed(texts, batch_size=32)))
        with torch.inference_mode():
            b = model.tok(texts, padding=True, truncation=True, max_length=MAX_LEN,
                          return_tensors="pt")
            ref = model(b["input_ids"], b["attention_mask"]).cpu().numpy()
        cos = (ref * got).sum(1)
        import fastembed as _fe
        return {"served": True, "version": _fe.__version__, "n_texts": len(texts),
                "min_cos": float(cos.min()), "max_abs": float(abs(ref - got).max()),
                "pass_min_cos_1e-4": bool(cos.min() >= 1 - 1e-4)}
    except Exception as e:
        return {"served": False, "error": repr(e)[:500]}


# ------------------------------------------------------------------------------------- the CLI ----

def build_argparser():
    ap = argparse.ArgumentParser(description="the M13 200M nano build controller")
    ap.add_argument("--config", default=str(BL.CONFIG))
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--plan", action="store_true",
                    help="print the cycle table and the cap arithmetic; write nothing")
    ap.add_argument("--benchmark", action="store_true",
                    help="fix max_extension_cycles from --rate and --price and write state.json")
    ap.add_argument("--rate", type=float, default=None, help="MEASURED examples/s")
    ap.add_argument("--price", type=float, default=None, help="BILLED $/h")
    ap.add_argument("--resume", action="store_true",
                    help="continue the run in work/m13build/<arm>/ from its receipt and checkpoint")
    ap.add_argument("--smoke-steps", type=int, default=None,
                    help="a SMOKE: N steps, its own dose, stub evaluations and its own tree under "
                         "work/m13build/smoke/ — never the published record path")
    ap.add_argument("--ckpt-minutes", type=float, default=DEFAULT_CKPT_MINUTES,
                    help="wall-clock rolling-checkpoint cadence (operational, not a recipe knob)")
    # SMOKE-ONLY, every one of them, exactly as run_arm.SMOKE_ONLY_KNOBS
    ap.add_argument("--batch", type=int, default=None,
                    help="SMOKE ONLY; the batch is the E1 verdict's")
    ap.add_argument("--max-len", type=int, default=None, help="SMOKE ONLY; a build runs at %d"
                    % MAX_LEN)
    ap.add_argument("--ckpt-every", type=int, default=None, help="SMOKE ONLY; steps")
    ap.add_argument("--n-fit", type=int, default=None, help="SMOKE ONLY warm-start fit sample")
    ap.add_argument("--lotte-gate", default=None, help="SMOKE ONLY gate record path")
    ap.add_argument("--compile", action="store_true",
                    help="SMOKE ONLY: torch.compile the training step (checkpoints eager, §T)")
    ap.add_argument("--quiet", action="store_true")
    return ap


def main(argv=None):
    a = build_argparser().parse_args(argv)
    cfg, _ = BL.load(a.config)
    if a.plan:
        try:
            batch, _src = BL.resolve_batch(cfg, smoke=a.batch is not None, override=a.batch)
        except SystemExit as e:
            print(f"batch: {e}")
            batch = None
        return print_plan(cfg, batch, a.rate, a.price)
    run(a.config, device=a.device, resume=a.resume, smoke_steps=a.smoke_steps, batch=a.batch,
        max_len=a.max_len, ckpt_every=a.ckpt_every, ckpt_minutes=a.ckpt_minutes, n_fit=a.n_fit,
        compile_step=a.compile, lotte_gate=a.lotte_gate, verbose=not a.quiet,
        benchmark_rate=a.rate, benchmark_price=a.price, benchmark_only=a.benchmark)
    return 0


if __name__ == "__main__":
    sys.exit(main())
