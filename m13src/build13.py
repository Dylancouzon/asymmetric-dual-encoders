"""The M13 200M BUILD CONTROLLER: one nano build, end to end, and its provenance record.

    .venv/bin/python m13src/build13.py --config m13/build_config.json --plan [--rate R --price P]
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
  * **the stop rules read as the whole build.** Ruling R13 (Dylan, 2026-09-10) drops extension
    cycles: the build is these three cycles, with `nano10.kill_fires` (terminal FAILED, no final
    checkpoint) and `nano10.plateau_fires` (a freeze at the last cycle end) the only stop rules.
    There is no cap, no spend accounting and no second schedule.
  * **the LoTTE gate as a precondition** (`check_gate`): a gate record that says what it executed,
    on which two checkpoints, against which E1 verdict — and whose `veto` selects bs32 whatever
    the E1 verdict says.
  * **a wall-clock rolling checkpoint** (default 30 min, converted to a step interval with the
    measured rate `--rate` reports), because `total // 20` is ~10 GPU-hours of exposure at this
    dose, and a `losses.jsonl` sidecar, because a 6.25M-element loss list in every checkpoint is
    not free.
  * **the freeze**: DEV-6 once, `nano10.export_onnx`, ORT and fastembed parity, and
    `results/m13_build_record.json`. A failed export or a failed parity ends the build
    `FROZEN_UNVERIFIED` — the checkpoint retained, `complete` false, the record non-terminal —
    and re-running with `--resume` retries the freeze without retraining.

**Refusals** mirror `run_arm`'s, and for the same reason — a registered run reads its recipe from
the registry or does not run: `--compile` and every recipe knob are `--smoke-steps` only; non-CUDA
outside a smoke; an invalid screen lock; a PENDING E1 batch; screen verdicts not bound to the live
registry; a missing or incomplete LoTTE gate record; a mandatory allocation above the $1,000
ceiling; a dose that is not the registered 200M or does not divide by the batch; a terminal record
that already exists.

On disk: `work/m13build/<arm>/{state.json, receipt.json, ckpt.pt, cycle{k}.pt, cov_*.json,
losses.jsonl, onnx/}` (`work/m13build/smoke/<arm>/` under `--smoke-steps`).
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


# --------------------------------------------------------------------------- the LoTTE gate ----

GATE_DECISIONS = ("veto", "no_veto", "skipped")
GATE_BRANCHES = ("bs32", "bs128")


def _is_sha(x):
    return isinstance(x, str) and len(x) == 64 and all(c in "0123456789abcdef" for c in x.lower())


ARM_RECORDS_DIR = None      # None -> RESULTS; tests point it at a directory of fake E arm records


def _registered_e_checkpoints(records_dir, *, smoke):
    """-> {arm: cycle-3 checkpoint sha} for both E arms from their published records, or None in a
    smoke when a record is missing. Outside a smoke a missing record refuses: the gate cannot have
    read a checkpoint that was never published (Codex re-check 2026-09-10, B6)."""
    d = Path(records_dir) if records_dir is not None else (ARM_RECORDS_DIR or RESULTS)
    out = {}
    for arm in ("E-bs32", "E-bs128"):
        rec = read_json(d / f"m10_arm_{arm}.json")
        sha = None
        if rec:
            sha = ((rec.get("checkpoints") or {}).get("cycle3") or {}).get("sha256") or \
                  rec.get("final_checkpoint_sha256")
        if not _is_sha(sha):
            if smoke:
                return None
            refuse(f"no published record with a cycle-3 checkpoint for {arm} under {d} — the LoTTE "
                   f"gate compares the two A100 E checkpoints, so both records must exist before it "
                   f"can have run (m13/LOTTE_GATE_REGISTRATION.json).")
        out[arm] = sha
    return out


def check_gate(gate_path, *, verdicts_path=None, e1_batch=None, arm_records_dir=None, smoke=False):
    """-> the gate summary the record carries, or refuses.

    `m13/LOTTE_GATE_REGISTRATION.json` registers the contract: the gate record is written by the
    executor that PERFORMS read #1, and "a record with executed false or a missing decision must
    be refused by build13". The version this replaces read `gate.get("decision")` off whatever
    JSON was at the path, so `{}` passed and a recorded VETO changed nothing (Codex 2026-09-10,
    B6). Enforced here: the execution outcome, the branch, both checkpoint identities, and that
    the E1 verdict the gate read is the one on disk now.
    """
    p = Path(gate_path)
    g = read_json(p)
    if g is None:
        refuse(f"no LoTTE gate record at {p}. The pre-build LoTTE handling is registered before "
               f"the build (m10/M102_LOCK.md 'Before training' 3); the record may say `skipped`, "
               f"but it must exist — a bs128 selection can still be vetoed.")
    if g.get("executed") is not True:
        refuse(f"{p} does not carry `executed: true`. m13/LOTTE_GATE_REGISTRATION.json is the "
               f"REGISTRATION, not the gate record: the record is written by the executor that "
               f"performs read #1 and reports what it did.")
    dec, branch = g.get("decision"), g.get("branch")
    if dec not in GATE_DECISIONS:
        refuse(f"{p}: `decision` is {dec!r}, not one of {list(GATE_DECISIONS)}")
    if branch not in GATE_BRANCHES:
        refuse(f"{p}: `branch` is {branch!r}, not one of {list(GATE_BRANCHES)}")
    if not _is_sha(g.get("candidate_sha256")):
        refuse(f"{p}: `candidate_sha256` is {g.get('candidate_sha256')!r}, not a sha256 — the "
               f"gate must name the checkpoint it actually read")
    comp = g.get("comparator_sha256")
    if dec == "skipped":
        if comp is not None and not _is_sha(comp):
            refuse(f"{p}: a `skipped` gate carries `comparator_sha256: null` (the bs32 branch has "
                   f"no comparator: identical recipe and action), not {comp!r}")
    elif not _is_sha(comp):
        refuse(f"{p}: a {dec!r} decision compares two checkpoints; `comparator_sha256` is "
               f"{comp!r}")
    # The gate must be the one E1's selection calls for (Codex re-check 2026-09-10, B6): in the
    # bs128 branch the veto RUNS, so `skipped` is the bypass the veto exists to prevent; in the
    # bs32 branch the selected recipe IS the anchor recipe and the veto is SKIPPED and forfeited
    # (m10/LOTTE_LOCK.md). A gate for the other branch is a gate for a different build.
    if e1_batch is not None:
        want = "bs128" if int(e1_batch) == 128 else "bs32"
        if branch != want:
            refuse(f"{p}: the gate is for branch {branch!r} but E1 selected {want}; the gate that "
                   f"was read is not the gate this build needs.")
        if want == "bs128" and dec == "skipped":
            refuse(f"{p}: E1 selected bs128, so the veto RUNS; a `skipped` gate cannot license a "
                   f"bs128 build (m10/LOTTE_LOCK.md, m13/LOTTE_GATE_REGISTRATION.json).")
        if want == "bs32" and dec != "skipped":
            refuse(f"{p}: E1 selected bs32, where the veto is registered SKIPPED (identical recipe "
                   f"and action); a {dec!r} decision means the gate compared something else.")
    expected = _registered_e_checkpoints(arm_records_dir, smoke=smoke)
    if expected is not None:
        cand_arm = "E-bs128" if branch == "bs128" else "E-bs32"
        if g["candidate_sha256"] != expected[cand_arm]:
            refuse(f"{p}: `candidate_sha256` {g['candidate_sha256'][:12]} is not {cand_arm}'s "
                   f"published cycle-3 checkpoint {expected[cand_arm][:12]}.")
        if dec != "skipped" and comp != expected["E-bs32"]:
            refuse(f"{p}: `comparator_sha256` {str(comp)[:12]} is not E-bs32's published cycle-3 "
                   f"checkpoint {expected['E-bs32'][:12]}.")
    live = sha256_file(Path(verdicts_path or BL.VERDICTS))
    if g.get("e1_verdict_sha256") != live:
        refuse(f"{p}: `e1_verdict_sha256` is {g.get('e1_verdict_sha256')!r} but "
               f"results/m10_screen_verdicts.json is {live!r}. The gate read one E1 verdict and "
               f"the build would run under another.")
    return {"path": rel(p), "sha256": sha256_file(p), "executed": True, "decision": dec,
            "branch": branch, "candidate_sha256": g["candidate_sha256"],
            "comparator_sha256": comp, "e1_verdict_sha256": live,
            "read_at": g.get("read_at"), "code_identity": g.get("code_identity"),
            "e1_batch": e1_batch, "arm_records_checked": expected is not None}


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


LOADER_FILES = ("corpus_loader.py", "data10.py")


def loader_identity():
    """sha256 of the SAMPLER's bytes. `run_arm.code_identity` covers run_arm/trainer10/nano10 but
    not the corpus path, so a change to `EpochShuffledStream`'s tail policy or to
    `length_buckets` left an unchanged assemble manifest and a resumable checkpoint while the
    data order had moved (Codex 2026-09-10, B10)."""
    h = hashlib.sha256()
    for name in LOADER_FILES:
        h.update((REPO / "m10src" / name).read_bytes())
    return h.hexdigest()


def fingerprint(plan, man, seed, smoke_steps, device, cfg_sha, *, gate=None):
    """`run_arm.fingerprint` (arm, recipe, registry, corpus manifest, device, optimizer, code)
    plus the build's own identity: the configuration's bytes, `build13.py`'s and
    `build_lock.py`'s bytes, the SAMPLER's bytes (`loader_identity`) and the LoTTE gate record's
    sha — so a resume cannot continue this build under a different recipe, a different sampler or
    a different gate outcome."""
    base = R.fingerprint(plan["arm"], plan, man, seed, smoke_steps, device)
    body = {"base": base, "config_sha256": cfg_sha,
            "build13_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "build_lock_sha256": sha256_file(REPO / "m13src" / "build_lock.py"),
            "loader_sha256": loader_identity(),
            "lotte_gate_sha256": (gate or {}).get("sha256")}
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()


# ------------------------------------------------------------------------ the allocation table ----

def print_plan(cfg, batch, rate=None, price=None):
    """The cycle table and the allocation table. Writes nothing, trains nothing, rents nothing.

    Ruling R13 leaves no cap to compute: the allocation is a fixed list of mandatory lines priced
    at the measured rate and the billed price, and the only judgement in it is whether the total
    fits under the recorded ceiling. `build_lock.allocation` REFUSES a plan that does not.
    """
    print(f"build arm {cfg['arm']}  dose {cfg['arm_entry']['dose_examples']:,}  "
          f"batch {batch if batch else 'PENDING (E1 unread)'}")
    batches = [batch] if batch else [32, 128]
    for b in batches:
        p = BL.cycle_plan(cfg["arm_entry"]["dose_examples"], b)
        print(f"\n  bs{b}: {p['total_steps']:,} steps, divides exactly: {p['divides_exactly']}")
        for row in p["cycles"]:
            print(f"    cycle {row['cycle']}  steps {row['steps']:>9,}  "
                  f"[{row['start_step']:>9,}..{row['end_step']:>9,}]  "
                  f"examples {row['examples']:>12,}")
        print(f"    sum {p['sum_examples']:,} (= dose: {p['sum_examples'] == p['dose_examples']})")
        rr = rate or R.PLAN_RATES.get(b)
        for pr in ([price] if price else [1.5, 2.5]):
            try:
                al = BL.allocation(cfg, rr, pr)
            except SystemExit as exc:
                print(f"    allocation at {rr:.0f} ex/s, ${pr}/h: {exc}")
                continue
            print(f"    allocation at {rr:.0f} ex/s, ${pr}/h: "
                  f"{al['mandatory_hours_total']} h = ${al['mandatory_usd']} "
                  f"+ ${al['fixed_usd']} fixed = ${al['committed_usd']} of "
                  f"${al['budget_ceiling_usd']:.0f} (headroom ${al['headroom_usd']})")
            for k, v in sorted(al["mandatory_hours"].items()):
                print(f"      {k:<38} {v:>8.3f} h  ${pr * v:>8.2f}")
        if not (rate and price):
            print("    (a PROJECTION: run_arm.PLAN_RATES and assumed prices. The table that "
                  "governs is priced from the MEASURED rate and the BILLED price on day one.)")
    return 0


# ------------------------------------------------------------------------------- the evaluator ----

class BuildEval:
    """`run_arm.CovEval` (or `StubEval` in a smoke) plus the build's own bookkeeping.

    The evaluation history is what the plateau and kill rules read, and it has to
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
           "_final_checkpoint_note": "a failed build has NO final checkpoint; any cycle "
                                     "checkpoint on disk is a partial",
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
        real_eval=False, lotte_gate=None, verbose=True, rate=None, price=None):
    ctx = {"started": False}
    try:
        return _run(ctx, config, device=device, resume=resume, smoke_steps=smoke_steps,
                    batch=batch, max_len=max_len, ckpt_every=ckpt_every,
                    ckpt_minutes=ckpt_minutes, n_fit=n_fit, compile_step=compile_step,
                    real_eval=real_eval, lotte_gate=lotte_gate, verbose=verbose,
                    rate=rate, price=price)
    except SystemExit:
        if ctx.get("started"):
            _record_failure(ctx, sys.exc_info()[1], verbose=verbose)
        raise
    except BaseException as e:                                          # noqa: BLE001
        _record_failure(ctx, e, verbose=verbose)
        raise


def _preflight(ctx, cfg, cfg_path, *, smoke, device, batch, lotte_gate, compile_step, real_eval,
               max_len, ckpt_every, n_fit, rate=None, price=None):
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
    gate = check_gate(lotte_gate or (REPO / cfg["lotte_gate"]), e1_batch=b, smoke=smoke)
    if gate["decision"] == "veto":
        # `m10/LOTTE_LOCK.md`: "the comparator's recipe (bs32) is what the 200M build trains".
        # The veto OVERRIDES the E1 verdict — that is the whole point of a veto, and the version
        # this replaces read the decision into the record and then trained bs128 anyway (B6).
        b, batch_source = 32, (f"m13 LoTTE gate VETO ({gate['path']}): the comparator's bs32 "
                               f"recipe, overriding {batch_source}")
    if rate is not None and price is not None:
        # the plain affordability check: a mandatory plan above the ceiling refuses here, before
        # a GPU-hour is spent (B7 as narrowed by R13 — there is no cap to compute).
        report["allocation"] = BL.allocation(cfg, rate, price, smoke=smoke)
    ctx["registry_sha256"] = report["registry_sha256"]
    return reg, b, batch_source, gate, report


def _run(ctx, config, *, device, resume, smoke_steps, batch, max_len, ckpt_every, ckpt_minutes,
         n_fit, compile_step, real_eval, lotte_gate, verbose, rate, price):
    smoke = smoke_steps is not None
    cfg, cfg_path = BL.load(config)
    cfg_sha = sha256_file(cfg_path)
    reg, batch, batch_source, gate, report = _preflight(
        ctx, cfg, cfg_path, smoke=smoke, device=device, batch=batch, lotte_gate=lotte_gate,
        compile_step=compile_step, real_eval=real_eval, max_len=max_len, ckpt_every=ckpt_every,
        n_fit=n_fit, rate=rate, price=price)
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
        if not resume:
            refuse(f"{path} already exists. Pass --resume to continue a non-terminal run, or "
                   f"move the record aside deliberately.")
        if st["parseable"] and st["terminal"]:
            refuse(f"{path} already carries a TERMINAL record (complete or failed); there is "
                   f"nothing to continue. Move the record aside deliberately.")
        if not st["parseable"]:
            refuse(f"{path} is unparseable, so whether the build already finished is UNKNOWABLE.")

    state = read_json(out_dir / "state.json") or {}
    if resume:
        if R._record_status_at(receipt_path) is None:
            refuse(f"--resume: no receipt at {receipt_path}; there is no run to continue.")
    elif (out_dir / "ckpt.pt").exists():
        refuse(f"{out_dir / 'ckpt.pt'} already exists without a running receipt; refusing to "
               f"overwrite an orphaned checkpoint with a fresh run.")

    t_start = time.time()
    print(f"=== build {arm} on {device}: {dose:,} examples, batch {batch}, "
          f"{plan['total_steps']:,} steps, pattern {plan['pattern']}, student {plan['student']}, "
          f"objective {plan['objective']}, seed {seed}, no extension cycles (R13)"
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
    fp = fingerprint(plan, man, seed, smoke_steps, device, cfg_sha, gate=gate)

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
    ctx["state"] = state
    write_json(out_dir / "state.json", state)

    ws_cfg = {"n_fit": int(n_fit or (R.SMOKE_N_FIT if smoke
                                     else cfg["data_knobs_equal_to_anchor"]["warm_start"]["n_fit"])),
              "seed": int(cfg["data_knobs_equal_to_anchor"]["warm_start"]["fit_seed"])}
    ws = R.warm_start(model, {"warm_start": plan["warm_start"]}, q_stream, ws_cfg, verbose=verbose)

    inner = R.StubEval() if smoke else R.CovEval(model, out_dir, verbose=verbose)
    ev = BuildEval(inner, out_dir / "state.json", state)
    ctx["cov_records"] = ev.records
    loss_log = out_dir / "losses.jsonl"

    def cadence(total):
        """The rolling checkpoint interval in STEPS, from the wall-clock cadence and the MEASURED
        rate `--rate` reports. Without a rate it falls back to `run_arm`'s `total // 20`, which at
        the build's dose is ~10 GPU-hours of exposure — hence the cadence."""
        if ckpt_every:
            return int(ckpt_every)
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

    # ---- the stop rules, read over the whole build ---------------------------------------
    # Ruling R13 (Dylan, 2026-09-10): NO extension cycles. The build is these three cycles, and
    # `nano10.kill_fires` / `nano10.plateau_fires` are the only stop rules. The plateau rule can
    # only fire at cycle end k >= 3, which for a 3-cycle schedule IS its last step, so a plateau
    # here is the ordinary end of the build and not a decision about what to run next.
    fired, why = N.kill_fires(state["evals"], lambda i: state["eval_kinds"][i])
    macros = state["cycle_macros"]
    stopped = state.get("stopped")
    if fired:
        outcome = {"outcome": "FAILED", "why": f"kill: {why}"}
    elif stopped and not str(stopped).startswith("plateau"):
        outcome = {"outcome": "FAILED", "why": str(stopped)}
    elif len(macros) < CYCLES:
        outcome = {"outcome": "FAILED",
                   "why": f"only {len(macros)} annealed cycle end(s) were read"}
    else:
        pf, at = N.plateau_fires(macros)
        outcome = {"outcome": "COMPLETE", "plateau_at": at,
                   "why": (f"plateau at cycle {at}, the schedule's last cycle end" if pf else
                           f"the registered {dose:,}-example schedule ran to its last cycle end"),
                   "final_gain": (round(macros[-1] - max(macros[:-1]), 6) if len(macros) > 1
                                  else None),
                   "_rule": "nano10.plateau_fires / kill_fires; no extension cycles (R13)"}
    state["outcome"] = outcome
    write_json(out_dir / "state.json", state)
    ok = outcome["outcome"] == "COMPLETE"
    final_ck = out_dir / f"cycle{CYCLES}.pt"
    if ok and not final_ck.exists():
        ok, outcome = False, {"outcome": "FAILED", "why": f"no final checkpoint at {final_ck}"}
        state["outcome"] = outcome
        write_json(out_dir / "state.json", state)

    # ---- freeze --------------------------------------------------------------------------
    freeze, verified = {}, False
    if ok:
        freeze = freeze_checkpoint(model, out_dir, final_ck, smoke=smoke, verbose=verbose)
        verified = bool(freeze.get("verified"))
        if not verified:
            # B8: an export that raised, or a parity read that did not meet the bar, is NOT a
            # complete build. The checkpoint is retained and the record is NON-terminal, so
            # re-running with --resume retries the freeze alone: training is already done and
            # `state["base_done"]` skips it.
            print(f"FREEZE UNVERIFIED: {freeze.get('unverified_why')}. The checkpoint is kept "
                  f"and the record is not `complete`; fix the export environment and re-run "
                  f"with --resume to retry the freeze without retraining.", flush=True)
    else:
        print(f"BUILD FAILED ({outcome['why']}): no final checkpoint, so no DEV-6, no export and "
              f"no parity. `rules.arm_failure`: reported, not re-run at different settings.",
              flush=True)

    cks = {}
    for pth in sorted(out_dir.glob("*.pt")):
        cks[pth.stem] = {"path": rel(pth), "sha256": sha256_file(pth)}
    status = ("complete" if verified else "frozen_unverified") if ok else "failed"
    rec = {
        "_what": "the M13 200M nano build, trained end to end by m13src/build13.py",
        "arm": arm, "status": status, "complete": bool(ok and verified),
        # a FROZEN_UNVERIFIED build is not terminal: its finalization is resumable (B8).
        "terminal": not (ok and not verified),
        "outcome": outcome, "smoke": smoke, "smoke_steps": smoke_steps, "device": device,
        "max_len": max_len, "seed": seed,
        "config": rel(cfg_path), "config_sha256": cfg_sha, "validation": report,
        "lotte_gate": gate, "batch": batch, "batch_source": batch_source,
        "recipe": plan, "warm_start": ws,
        "schedule": {"cycles": CYCLES, "peak": PEAK, "final": FINAL,
                     "cycle_plan": BL.cycle_plan(dose, batch),
                     "warmup_examples": int(N.WARMUP_EXAMPLES),
                     "warmup_steps_cycle1": int(N.warmup_steps_for(batch)),
                     "extension_cycles": "none (ruling R13, Dylan 2026-09-10)"},
        "params": model.n_params(), "under_cap": model.under_cap(),
        "evaluation_mode": "stub" if smoke else "cov",
        "cov": {"per_checkpoint": ev.records, "cycle_macros": state["cycle_macros"],
                "final_macro": (state["cycle_macros"][-1] if state["cycle_macros"] else None),
                "_scope": "COV only, at every annealed cycle end and the schedule midpoints the "
                          "kill rule reads. No six-set, reserved or LoTTE surface."},
        "budget": {"allocation": report.get("allocation"),
                   "ceiling_usd": float(cfg["budget_ceiling_usd"]),
                   "_what": "the fixed allocation table (m13/build_config.json `budget`) priced "
                            "at --rate and --price when they were given. R13 leaves no cap "
                            "formula and no in-run spend accounting: the bill is reconciled "
                            "against this table in the allocation record, not by the controller."},
        "dose_run_examples": state.get("examples_run"), "steps_run": state.get("steps_run"),
        "training": state.get("base"), "loss_logs": [rel(p) for p in sorted(out_dir.glob("losses*.jsonl"))],
        "checkpoints": cks,
        "final_checkpoint": rel(final_ck) if ok else None,
        "final_checkpoint_sha256": (sha256_file(final_ck) if ok else None),
        "_final_checkpoint": (None if verified or not ok else
                              "RETAINED but UNVERIFIED: the export or the serving parity failed, "
                              "so this checkpoint has not been shown to serve what it trained. "
                              "It is not a release candidate until a --resume freeze passes."),
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
    # the receipt stays `running` while the freeze is unverified, because that is exactly what
    # `--resume` requires to continue the run and retry it.
    write_json(receipt_path, {**(read_json(receipt_path) or {}),
                              "status": ("running" if (ok and not verified) else rec["status"]),
                              "freeze": (None if verified else "unverified"),
                              "complete": rec["complete"], "terminal": rec["terminal"]})
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


PARITY_MIN_COS = 1 - 1e-4


def freeze_checkpoint(model, out_dir, final_ck, *, smoke=False, verbose=True):
    """DEV-6 once, the ONNX export, and the serving-parity reads. A smoke skips DEV-6 (~13 GB of
    reads) exactly as `run_arm`'s does, and says so in the record.

    Returns `verified: False` with `unverified_why` when the export RAISED or the ORT parity did
    not reach `PARITY_MIN_COS`. The version this replaces swallowed the export exception into an
    `export_error` field that nothing read, so a build whose ONNX never existed still published
    `complete: true` and a final checkpoint (Codex 2026-09-10, B8). fastembed serving parity IS part of the bar (Codex re-check 2026-09-10, B8): `pass_min_cos_1e-4` must be true, so a box without fastembed leaves the freeze unverified until it is re-run where fastembed serves it.md` pitfalls 2 and 5); the ORT parity is the one this
    build always has.
    """
    ck = torch.load(final_ck, map_location="cpu", weights_only=False)
    Tr.eager(model).load_state_dict(ck["model"])
    out = {"from_checkpoint": rel(final_ck),
           "from_checkpoint_sha256": sha256_file(final_ck),
           "dev6": None, "onnx": None, "ort_parity": None,
           "fastembed_parity": None, "verified": False, "unverified_why": None}
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
    except Exception as e:
        out["export_error"] = f"{type(e).__name__}: {e}"
        out["unverified_why"] = f"the ONNX export or its parity read raised: {out['export_error']}"
        return out
    cos = (out["ort_parity"] or {}).get("min_cos")
    if cos is None or float(cos) < PARITY_MIN_COS:
        out["unverified_why"] = (f"ORT serving parity min-cos {cos!r} < {PARITY_MIN_COS}: the "
                                 f"exported graph does not reproduce the trained model")
        return out
    fe = out.get("fastembed_parity") or {}
    if not fe.get("served") or not fe.get("pass_min_cos_1e-4"):
        # fastembed IS the shipped serving path (M11's release contract), so it is part of the
        # bar (Codex re-check 2026-09-10, B8): an unavailable or failing fastembed read leaves the
        # freeze unverified until the freeze is re-run where fastembed serves it.
        out["unverified_why"] = ("fastembed serving parity did not pass: "
                                 + json.dumps({k: fe.get(k) for k in ("served", "error", "min_cos",
                                                                     "pass_min_cos_1e-4")}))
        return out
    out["verified"] = True
    out["parity_bar_min_cos"] = PARITY_MIN_COS
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
                    help="print the cycle table and the allocation table; write nothing")
    ap.add_argument("--rate", type=float, default=None,
                    help="MEASURED examples/s: prices the allocation table and sets the rolling "
                         "checkpoint's step interval. Recorded, never a recipe knob.")
    ap.add_argument("--price", type=float, default=None,
                    help="BILLED $/h: with --rate, prices the allocation table. A mandatory plan "
                         "above the $1,000 ceiling is REFUSED.")
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
        rate=a.rate, price=a.price)
    return 0


if __name__ == "__main__":
    sys.exit(main())
