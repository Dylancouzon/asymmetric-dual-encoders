"""The registered screen-arm RUNNER: one arm, end to end, and its record.

    .venv/bin/python m10src/run_arm.py <arm> [--device cuda|cpu] [--resume] [--smoke-steps N]
    .venv/bin/python m10src/run_arm.py --plan

It COMPOSES; it decides nothing. Every data-affecting and schedule-affecting constant is read
from `m10/screen_registry.json` (§0a) through the module that owns it — `corpus_loader`
(`assemble_arm`, the ONLY corpus path), `nano10` (student, head, losses, cycle ends, LR),
`trainer10` (loop, checkpoint, resume, kill/plateau), `cov_eval10`/`cov_macro` (the COV macro),
`eval9` (the DEV-6 read) — and `arm_smoke.SHAPES` is the one place an arm's *shape* is encoded, so
it is imported rather than retyped.

**What it refuses to start on** (`rules.arm_failure`: an arm is never silently re-run):
`screen_lock.validate` reporting anything; an arm that is not `trained: true`, or is `cut`; a
cut-corpus arm while `data_cut.unique_text_count` is unregistered (enforced by `assemble_arm`;
checked here only to fail before the corpus is read); a record that already exists.

**Evaluation, exactly as `evaluation` in the registry says and nothing else:** the COV macro at
every cycle end (plus the schedule midpoints the trainer already reads for the kill rule), the
family-F `read_at` curve points, and DEV-6 ONCE at the final checkpoint. No six-set, reserved or
LoTTE surface is touched, and per-query COV scores are written per checkpoint because the
CONTRAST step — which is not this file — needs aligned per-query vectors to bootstrap.

**Contrasts are NOT computed here.** `statistics.bootstrap._runner_contract` (the per-contrast
`quantile`) binds the contrast runner, not this one: no bootstrap is run and no quantile is read.

**Two hooks `trainer10` did not have**, added there additively because the eval and checkpoint
schedules are computed inside the loop and no parameter reached them:
  * `read_steps` — family F is read at 5M/10M/20M inside a 20M schedule, and 5M is neither a
    cycle end nor a midpoint. Read evals are recorded SEPARATELY and are deliberately kept out of
    the kill/plateau state: §Kill compares "midpoint against midpoints, cycle end against cycle
    ends", and a third kind entering that sequence would change a registered rule.
  * `cycle_ckpt_fmt` — a RETAINED checkpoint per cycle end. The rolling checkpoint is a resume
    point that gets overwritten; the final cycle end is the arm's final checkpoint and its sha256
    goes in the record.

**Success is narrow** (`classify`): `stopped is None`, or exactly `plateau at cycle 3` —
`PLATEAU_FROM_CYCLE` is 3 and a screen arm runs 3 cycles, so that plateau means the arm finished
its dose and stopped one step short. Every KILL and every non-finite stop is FAILED, the final
cycle end included, and a failed arm gets no DEV-6 read and no checkpoint labelled final.

**A crash or an OOM is an outcome, not silence**: it writes a terminal FAILED record and re-raises
(`rules.arm_failure`). A REFUSAL writes nothing — the arm never started.
Before training, a durable running receipt records the arm's identity. Abrupt process loss that
cannot write a terminal outcome can resume from that receipt and the rolling checkpoint.

**Post-F arms take their student from F's verdict**, `results/m10_F_verdict.json`
(`anchor.init: "F's winner backbone"`, §Recipe amendment A6), validated against the current
registry and F's own arm records. `arm_smoke.SHAPES` holds shapes; it does not decide the student.

**Every recipe knob is SMOKE-ONLY** (`--max-len`, `--ckpt-every`, `--n-fit`, `--compile`): a
registered arm reads its recipe from the registry or does not run.

**`--dev6 defer` is not a recipe knob.** DEV-6 is still read once at the final checkpoint; the
runner records that the read did not happen on this machine and which checkpoint bytes it must
consume, and `m13src/dev6_from_checkpoint.py` fills the record's `dev6` field on the box from the
identical checkpoint. For the cloud E arms, whose ~35 GB of DEV-6 caches are not shipped.
"""
import argparse
import copy
import hashlib
import json
import os
import subprocess
import traceback
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for p in ("m7src", "m9src", "m10src"):
    sys.path.insert(0, str(REPO / p))

import numpy as np
import torch

import arm_smoke as AS
import corpus_loader as CL
import nano10 as N
import screen_lock as SL
import trainer10 as Tr

WORK = REPO / "work" / "m10arms"
SMOKE_WORK = WORK / "smoke"
RESULTS = REPO / "results"
REGISTRY = REPO / "m10" / "screen_registry.json"

# The W8-band-1 order (`m10/STATUS.md` §Screen design, settled; `_w8_band`). Both E arms are in
# the list and marked CLOUD: `E-bs128` does not run on this box at realistic sequence lengths, and
# `E-bs32` runs beside it on the A100 so E1 carries no hardware difference. The registry's `order`
# is by FAMILY ("F, A, G, B, E, C, D") and names no arm, and `_w8_band` names no per-arm order
# either, so `E-bs32` is placed immediately before `E-bs128` — the E pair, kept adjacent, in the
# tail this list already gives family E.
BAND1_ORDER = ["F-bge-small", "F-MiniLM-L6", "ANCHOR", "A1", "A2", "A3",
               "G-384", "G-1536", "G-MLP", "B-100/0", "B-50/50", "D-NORM", "D-COV",
               "E-bs32", "E-bs128"]
CLOUD_ONLY = dict(AS.CLOUD_ONLY)

# Projected-hours rates for `--plan` ONLY. A projection, never a measurement: every arm records
# its OWN ex/s.
#
# **bs32 corrected 2026-09-07 from 890 to 512, measured in THIS runner.** 890 came from
# `results/m10_rate_bench_real_box.json`, whose fastest row (959 ex/s) is the `fixed_bucket_compile`
# path — and `--compile` is SMOKE-ONLY here, so a registered arm runs EAGER. On the real corpus,
# eager, at the registered max_len 512, a 300-step F-bge-small smoke with checkpointing out of the
# way holds **~512 ex/s** (steady, still creeping up: 507 -> 512). That is 10.9 h per 20M arm, not
# 6.24, and ~52 box hours for the 14-arm band rather than 30. The benchmark's own header always
# said it bounds the hardware and not the pipeline; this is the pipeline.
# bs128 stays as-is and is CLOUD-only: the box cannot reproduce it in the trainer (LEDGER §E-bs128).
PLAN_RATES = {32: 512.0, 128: 1517.0}

CYCLES = 3
PEAK, FINAL = 1e-4, 1e-5
MAX_LEN = 512
# Every arm after family F trains on F's winner backbone (`anchor.init: "F's winner backbone"`,
# §Recipe amendment A6). The verdict file is written by the CONTRAST step, not here.
F_VERDICT_NAME = "m10_F_verdict.json"
F_VERDICT_KEYS = ("winner", "contrast", "registry_sha256", "sha256_of_F_records")
# The registry names n_fit/seed only under `warm_start.G-MLP`; `warm_start.all_arms` names M9's
# ridge warm start without a sample size. Read from the registry's G-MLP block, defaulting to
# `m9/registry.json` `warm_start` (n_fit 60,000, seed 21) — the sample `calib.py` also used.
WS_DEFAULTS = {"n_fit": 60_000, "seed": 21}
# A SMOKE's warm start fits 256 texts, as `arm_smoke.N_WS_FIT` does: 60,000 pooled forwards on CPU
# is not a smoke. The lambda it selects is meaningless at that size (`nano10.select_lambda`) --
# the PATH is what is being smoked.
SMOKE_N_FIT = 256
# registry `student` strings -> `nano10.REPOS`/`arm_smoke.SHAPES` keys
STUDENT_ALIAS = {"bge-small": "bge-small", "MiniLM-L6-v2": "MiniLM-L6",
                 "MiniLM-L12-v2": "MiniLM-L12", "MiniLM-L6": "MiniLM-L6",
                 "MiniLM-L12": "MiniLM-L12"}


def refuse(msg):
    raise SystemExit(f"REFUSED: {msg}")


def slug(name):
    """`B-100/0` is an arm name and not a path. The record keeps the arm name verbatim."""
    return name.replace("/", "-")


def rel(p):
    """Repo-relative when it is inside the repo, absolute otherwise (a sandboxed test path)."""
    p = Path(p)
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


def sha256_file(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def git_head():
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception as e:                                          # pragma: no cover
        return f"unavailable: {type(e).__name__}"


def cfg():
    return SL.cfg()


def record_paths(arm, smoke):
    """-> (out_dir, record path, published results path or None). A smoke gets its OWN tree
    (§Hazards: smokes have overwritten real artifacts in this milestone twice)."""
    if smoke:
        d = SMOKE_WORK / slug(arm)
        return d, d / "record.json", None
    d = WORK / slug(arm)
    return d, d / "record.json", RESULTS / f"m10_arm_{slug(arm)}.json"


def write_record(rec, rec_path, results_path, *, create_only=False):
    """ONE canonical record, written atomically to each published path (finding 13).

    `Path.write_text` on the real path leaves a truncated JSON behind if the process dies
    mid-write, and a half-written record is indistinguishable from an arm that reported nothing.
    """
    blob = json.dumps(rec, indent=1, default=str)
    written = []
    for p in [rec_path] + ([results_path] if results_path is not None else []):
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + f".tmp{os.getpid()}")
        with open(tmp, "w") as fh:
            fh.write(blob)
            fh.flush()
            os.fsync(fh.fileno())
        if create_only:
            # Publish a complete receipt without replacing an existing run's identity.
            try:
                os.link(tmp, p)
            finally:
                tmp.unlink()
        else:
            os.replace(tmp, p)
        dfd = os.open(p.parent, os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
        written.append(p)
    return written


def _record_status_at(p):
    """-> None if no file exists at `p`, else {"parseable", "terminal", "raw"}.

    A record is TERMINAL once it carries `complete: true` (a successful arm) or `terminal: true`
    (a FAILED arm, item E: a failed record carries `complete: false, terminal: true` so a failure
    is never mistaken for a completion). An unparseable file is treated as existing but
    non-terminal -- it still blocks a bare re-run (below), but does not itself block `--resume`.
    """
    if p is None or not p.exists():
        return None
    try:
        raw = json.loads(p.read_text())
    except Exception:
        return {"parseable": False, "terminal": False, "raw": None}
    if not isinstance(raw, dict):
        return {"parseable": False, "terminal": False, "raw": None}
    return {"parseable": True, "terminal": bool(raw.get("complete")) or bool(raw.get("terminal")),
            "raw": raw}


# ------------------------------------------------------------------------------- F's verdict ----

def f_arms(reg):
    """The trained family-F arms, from the registry — never a hand-copied list."""
    return sorted(n for n, e in (reg.get("arms") or {}).items()
                  if e.get("family") == "F" and e.get("trained") is True)


def f_verdict(reg, path=None):
    """-> {"student", "winner", "path", ...} for `anchor.init: "F's winner backbone"`.

    Family F runs FIRST and every later family is screened on its winner (§Recipe amendment A6),
    so an arm that took its student from `arm_smoke.SHAPES` would silently screen the wrong
    backbone (Codex runner review 2026-09-07, finding 11). The verdict FILE is produced by the
    contrast step — not by this runner, which only validates and reads it:

      {"winner": "<student key>", "contrast": {...}, "registry_sha256": "...",
       "sha256_of_F_records": ["...", ...]}

    Refused if it is absent, if it does not name the registry it was decided under, or if its
    record hashes are not exactly the current F arm records' — a verdict is only as good as the
    two arms it was read off.
    """
    p = Path(path) if path is not None else RESULTS / F_VERDICT_NAME
    if not p.exists():
        refuse(f"every arm after family F trains on F's WINNER backbone "
               f"(anchor.init {(reg.get('anchor') or {}).get('init')!r}), and {p} does not exist. "
               f"Run family F and its contrast first; this runner never guesses the student.")
    try:
        v = json.loads(p.read_text())
    except Exception as e:
        refuse(f"{p} is not readable JSON: {type(e).__name__}: {e}")
    missing = [k for k in F_VERDICT_KEYS if k not in v]
    if missing:
        refuse(f"{p} is missing {missing}; the F verdict schema is {list(F_VERDICT_KEYS)}")
    # item F: the `contrast` block's SCHEMA, not its semantics — no quantile is recomputed here,
    # only that the fields the contrast runner registered are actually present.
    contrast = v.get("contrast")
    contrast_keys = ("rule", "point", "lower", "resolved")
    if not isinstance(contrast, dict) or any(k not in contrast for k in contrast_keys):
        got = sorted(contrast) if isinstance(contrast, dict) else type(contrast).__name__
        refuse(f"{p}'s contrast block must carry {list(contrast_keys)}; got {got}")
    reg_sha = sha256_file(REGISTRY)
    if v["registry_sha256"] != reg_sha:
        refuse(f"{p} was decided under registry {v['registry_sha256'][:12]}… and the registry is "
               f"now {reg_sha[:12]}…; re-read F's contrast under the current registry")
    want = []
    for n in f_arms(reg):
        rp = RESULTS / f"m10_arm_{slug(n)}.json"
        if not rp.exists():
            refuse(f"{p} claims an F verdict but {rp} does not exist — F's own arm records are "
                   f"what the verdict is read off")
        try:
            rrec = json.loads(rp.read_text())
        except Exception as e:
            refuse(f"{rp} is not readable JSON: {type(e).__name__}: {e}")
        # item F: only a COMPLETED F arm with a final checkpoint can back a verdict — a failed
        # arm's record hash is just as reproducible, so the hash check alone would not catch it.
        if rrec.get("status") != "complete":
            refuse(f"{rp} has status {rrec.get('status')!r}, not 'complete'; F's verdict can only "
                   f"be read off a completed arm")
        if not rrec.get("final_checkpoint_sha256"):
            refuse(f"{rp} carries no final_checkpoint_sha256; F's verdict can only be read off an "
                   f"arm with a final checkpoint")
        want.append(sha256_file(rp))
    got = list(v["sha256_of_F_records"])
    if sorted(got) != sorted(want):
        refuse(f"{p} names F records {[h[:12] for h in sorted(got)]} and the records on disk "
               f"hash to {[h[:12] for h in sorted(want)]}; the verdict is stale")
    student = STUDENT_ALIAS.get(v["winner"])
    if student is None:
        refuse(f"{p} names winner {v['winner']!r}, which is not a known nano10 student "
               f"({sorted(set(STUDENT_ALIAS))})")
    # item F: the winner must be the student of a family-F arm that is `trained: true` and not
    # `cut` in the CURRENT registry — a verdict naming e.g. MiniLM-L12-v2 (cut, never trained) is
    # refused even if some stale contrast once named it.
    f_reg_entries = {n: e for n, e in (reg.get("arms") or {}).items() if e.get("family") == "F"}
    named_by = [n for n, e in f_reg_entries.items() if e.get("student") == v["winner"]]
    eligible = [n for n in named_by
                if f_reg_entries[n].get("trained") is True and not f_reg_entries[n].get("cut")]
    if not eligible:
        refuse(f"{p} names winner {v['winner']!r}, but no family-F arm in the registry is both "
               f"`trained: true` and uncut with that student (candidates: {named_by or 'none'}); "
               f"a verdict can only name a TRAINED, uncut F arm's student")
    return {"student": student, "winner": v["winner"], "path": rel(p),
            "registry_sha256": reg_sha, "sha256_of_F_records": sorted(want)}


# ------------------------------------------------------------------------------ arm resolution --

def shape_for(arm):
    """The arm's registered SHAPE, from `arm_smoke.SHAPES` — the one hand-copy of the registry
    that is already validated against it (`arm_smoke.main` refuses a trained arm it does not
    cover). `COVERS` maps the arms that differ only in data or dose onto the shape they share."""
    if arm in AS.SHAPES:
        return dict(AS.SHAPES[arm])
    for k, covered in AS.COVERS.items():
        if arm in covered:
            return dict(AS.SHAPES[k])
    refuse(f"arm {arm!r} has no shape in arm_smoke.SHAPES/COVERS")


def resolved_recipe(name, entry, reg, pattern, batch):
    """The arm's recipe as the REGISTRY resolves it — every field defaulted from `anchor` where
    the arm's own entry is silent, which is what `arm_smoke.SHAPES` is a hand copy OF.

    `student` is None for every arm after family F: the registry says `anchor.init: "F's winner
    backbone"` and names no student, so there is nothing to cross-check and the student comes
    from F's verdict instead (`f_verdict`).
    """
    a = reg.get("anchor") or {}
    return {
        "student": entry.get("student"),
        "objective": entry.get("objective") or a.get("objective") or "squared_l2",
        "feature_layers": int(entry.get("feature_layers")
                              if entry.get("feature_layers") is not None
                              else a.get("feature_layers", 3)),
        # G-MLP is the one arm carrying its own `head` (a formula string); every other arm is the
        # anchor's linear head. C-M9init is the one arm carrying `head_init`, so its warm start is
        # M9's checkpoint and every other arm's is the ridge (`warm_start.all_arms`).
        "head": "mlp" if entry.get("head") else "linear",
        "warm_start": ("m9" if entry.get("head_init") else
                       ("mlp" if entry.get("head") else "linear")),
        "pattern": pattern,
        "batch": int(batch),
    }


def check_shape(arm, recipe, spec):
    """Cross-check the UNTOUCHED `arm_smoke.SHAPES` entry against the fully resolved registry
    recipe, BEFORE anything is copied into it.

    Finding 6 (Codex runner review 2026-09-07): `arm_plan` used to write the registry's batch INTO
    `spec` and then compare the two, so the batch check could not fail; head, mix pattern and warm
    start were never compared at all. Every field the two representations share is compared here.
    """
    bad = []
    if recipe["student"] is not None:
        want = STUDENT_ALIAS.get(recipe["student"])
        if want is None:
            bad.append(f"registry student {recipe['student']!r} is not a known nano10 student")
        elif want != spec["student"]:
            bad.append(f"student: registry {recipe['student']!r} -> {want!r} vs shape "
                       f"{spec['student']!r}")
    for field, in_spec in (("objective", "loss"), ("feature_layers", "n_layers"),
                           ("head", "head"), ("pattern", "pattern"), ("batch", "batch")):
        if recipe[field] != spec[in_spec]:
            bad.append(f"{field}: registry {recipe[field]!r} vs shape {spec[in_spec]!r}")
    if recipe["warm_start"] != spec.get("warm_start", "linear"):
        bad.append(f"warm_start: registry {recipe['warm_start']!r} vs shape "
                   f"{spec.get('warm_start', 'linear')!r}")
    if bad:
        refuse(f"arm {arm!r}: registry and arm_smoke.SHAPES disagree — " + "; ".join(bad))


def schedule(dose, batch, entry):
    """-> (total_steps, cycle_end_steps, [(examples, step)] read points, rounding report).

    The schedule is 3 cycles over THE ARM'S OWN dose (F's 20M arms therefore anneal over 20M and
    are merely READ at 5M/10M/20M), so `total_steps = dose // batch` and the cycle ends are
    `nano10.cycle_ends`, which is what `lr_at` anneals to.

    **Registry ambiguity, floored.** `E-bs128`'s registered dose 5,000,000 is not a whole number
    of batches of 128 (39,062.5), so family E's "equal examples and identical schedule" cannot be
    exact at both batch sizes. The dose is treated as a CAP: 39,062 steps = 4,999,936 examples, 64
    fewer than bs32's 5,000,000 (0.0013%), because the other rounding spends unregistered compute.
    Recorded in the arm's record, never silently.
    """
    total = dose // batch
    rounding = {"dose_registered": int(dose), "examples_run": int(total * batch),
                "examples_dropped": int(dose - total * batch), "rule": "floor (the dose is a cap)"}
    ends = N.cycle_ends(total, CYCLES)
    reads = []
    for r in entry.get("read_at") or []:
        r = int(r)
        if r > dose:
            refuse(f"read_at {r:,} exceeds the arm's dose {dose:,}")
        step = r // batch - 1
        reads.append((r, step))
        if r % batch:
            rounding.setdefault("read_at_floored", {})[str(r)] = int((step + 1) * batch)
    return total, ends, reads, rounding


def arm_plan(name, reg, verdict=None):
    """Everything `--plan` prints and the runner needs, all of it read from the registry.

    `verdict` is F's verdict (`f_verdict`) when one has been read: for every arm after family F it
    is where the STUDENT comes from, `arm_smoke.SHAPES` holding only that arm's shape. Without one
    the plan still prints, marked — a plan is not a launch.
    """
    entry = dict((reg.get("arms") or {}).get(name) or {})
    resolved = CL.resolve_arm_name(name, reg) if entry.get("trained") else name
    spec = shape_for(resolved)                          # UNTOUCHED until the check has passed
    pattern, pattern_rep = CL.resolve_arm_pattern(resolved, entry, reg)
    batch = int(CL.arm_batch(entry, reg))
    recipe = resolved_recipe(name, entry, reg, pattern, batch)
    check_shape(name, recipe, spec)
    student, student_source = spec["student"], "registry `arms.%s.student`" % name
    if recipe["student"] is None:                       # a post-F arm: F's winner backbone
        if verdict is not None:
            student, student_source = verdict["student"], verdict["path"]
        else:
            student_source = "PENDING: F's verdict has not been read (shape's placeholder shown)"
    dose = int(entry["dose_examples"])
    total, ends, reads, rounding = schedule(dose, batch, entry)
    rate = PLAN_RATES.get(batch)
    return {"arm": name, "resolved": resolved, "family": entry.get("family"),
            "dose_examples": dose, "batch": batch, "pattern": pattern,
            "pattern_source": pattern_rep, "student": student,
            "student_source": student_source, "f_verdict": verdict,
            # the registry is authoritative for everything but the student; `check_shape` has
            # already proved `arm_smoke.SHAPES` agrees field by field
            "n_layers": recipe["feature_layers"], "head": recipe["head"],
            "objective": recipe["objective"], "warm_start": recipe["warm_start"],
            "cut_corpus": bool(CL.is_cut_corpus(resolved, reg)),
            "sources": list(CL.arm_sources(resolved, reg)),
            "n_docs": CL.arm_doc_count(entry, pattern, batch),
            "total_steps": total, "cycle_end_steps": ends, "dose_rounding": rounding,
            "read_points": [{"examples": e, "step": s} for e, s in reads],
            "projected_hours": (round(rounding["examples_run"] / rate / 3600, 2)
                                if rate else None),
            "projected_at_ex_per_s": rate,
            "cloud_only": CLOUD_ONLY.get(name)}


def plan(reg, order=None):
    """The band-1 plan. Post-F arms print `arm_smoke.SHAPES`'s placeholder student marked `*`
    unless F's verdict is on disk: only a RUN reads the verdict, and only a run refuses without
    one (finding 11)."""
    verdict = f_verdict(reg) if (RESULTS / F_VERDICT_NAME).exists() else None
    rows = []
    for name in (order or BAND1_ORDER):
        rows.append(arm_plan(name, reg, verdict=verdict))
    cut = CL.data_cut_count(reg)
    print(f"W8 band-1 order, {len(rows)} arms. data_cut.unique_text_count = "
          f"{cut if cut is not None else 'UNREGISTERED (§0b open: every cut arm refuses)'}")
    hdr = (f"{'arm':13s} {'dose':>10s} {'bs':>4s} {'pattern':>8s} {'student':11s} "
           f"{'objective':28s} {'warm':6s} {'cut':4s} {'steps':>8s} {'h':>6s}  where")
    print(hdr)
    print("-" * len(hdr))
    tot = 0.0
    for r in rows:
        where = r["cloud_only"] and "CLOUD" or "box"
        if r["projected_hours"] and not r["cloud_only"]:
            tot += r["projected_hours"]
        st = r["student"] + ("*" if str(r["student_source"]).startswith("PENDING") else "")
        print(f"{r['arm']:13s} {r['dose_examples']:>10,} {r['batch']:>4d} {r['pattern']:>8s} "
              f"{st:11s} {r['objective']:28s} {r['warm_start']:6s} "
              f"{'yes' if r['cut_corpus'] else 'no':4s} {r['total_steps']:>8,} "
              f"{r['projected_hours']:>6.2f}  {where}")
    if any(str(r["student_source"]).startswith("PENDING") for r in rows):
        print("\n* the student is F's WINNER backbone and F's verdict "
              f"({rel(RESULTS / F_VERDICT_NAME)}) has not been written: the shape's placeholder "
              "is shown and every one of these arms REFUSES to run until it exists")
    print(f"\nprojected box hours (excluding CLOUD arms) {tot:.1f} at "
          f"{PLAN_RATES[32]:.0f}/{PLAN_RATES[128]:.0f} ex/s (bs32/bs128); a PROJECTION — every "
          f"arm records its own rate")
    return rows


# ------------------------------------------------------------------------------- the evaluators --

class CovEval:
    """The COV macro at a scheduled checkpoint, and the per-query scores the contrast step needs.

    `cov_probe.units()` is loaded ONCE: it is 13,416 queries over four families of corpora and
    reloading it per cycle end would cost more than the encode.
    """

    def __init__(self, model, out_dir, batch_size=256, verbose=True):
        self.model, self.out_dir, self.bs, self.verbose = model, out_dir, batch_size, verbose
        self._units = None
        self.records = []

    def units(self):
        if self._units is None:
            import cov_probe
            self._units = cov_probe.units()
        return self._units

    def __call__(self, step, label):
        import cov_eval10
        us = self.units()
        was_training = self.model.training
        t0 = time.time()
        per = cov_eval10.score_student(
            lambda t: self.model.encode_queries(t, batch_size=self.bs), units=us,
            verbose=self.verbose)
        macro, by_family, by_unit = cov_eval10.macro(per, units=us)
        if was_training:
            self.model.train()          # encode_queries calls .eval(); the loop must go on training
        p = self.out_dir / f"cov_{label}.json"
        p.write_text(json.dumps({"label": label, "step": int(step), "macro": macro,
                                 "by_family": by_family, "by_unit": by_unit,
                                 "per_unit_query": per}))
        # item C: the evidence file's own hash goes beside its path, so a resume can tell a
        # per-query score file that has moved, been truncated or been silently rewritten from the
        # one the run actually produced.
        rec = {"label": label, "step": int(step), "macro": macro, "by_family": by_family,
               "by_unit": by_unit, "per_query_scores": rel(p),
               "per_query_scores_sha256": sha256_file(p),
               "seconds": round(time.time() - t0, 1)}
        self.records.append(rec)
        if self.verbose:
            print(f"  COV {label} step {step:,}: macro {macro:.4f} "
                  f"{ {k: round(v, 4) for k, v in by_family.items()} }", flush=True)
        return macro

    # The evaluator's records — per-family macros, the per-query score paths and their hashes the
    # contrast step bootstraps over — are part of the run state and go in every checkpoint
    # (finding 7). The final record carries them too, since `records` IS `cov.per_checkpoint`.
    def state(self):
        return {"records": self.records}

    def restore(self, d):
        """Item C: every evidence file a restored record references must still exist and hash to
        what the run wrote. A resume that silently trusted a moved or altered `cov_*.json` would
        hand the contrast step per-query scores that no longer match the arm's own history."""
        recs = list(d.get("records") or [])
        for r in recs:
            rel_path, want = r.get("per_query_scores"), r.get("per_query_scores_sha256")
            if rel_path is None:
                continue
            fp = Path(rel_path)
            fp = fp if fp.is_absolute() else REPO / fp
            if not fp.exists():
                refuse(f"resume: the evidence file for {r.get('label')!r} ({fp}) is missing. "
                       f"A resume must not report an evaluation whose evidence has vanished.")
            got = sha256_file(fp)
            if want is not None and got != want:
                refuse(f"resume: the evidence file for {r.get('label')!r} ({fp}) hashes to "
                       f"{got[:12]}… but the checkpoint recorded {want[:12]}…. The arm's own "
                       f"evidence no longer matches its history.")
        self.records = recs


class StubEval:
    """The smoke's evaluator. The real COV read is 13,416 queries against 452,757 cached stella
    document vectors and DEV-6 is ~13 GB of reads per pass; neither is a CPU smoke's business.
    Returns a deterministic increasing series so the plateau rule cannot fire spuriously."""

    def __init__(self):
        self.records = []

    def __call__(self, step, label):
        m = 0.4 + 0.01 * len(self.records)
        self.records.append({"label": label, "step": int(step), "macro": m, "stub": True})
        print(f"  [stub] COV {label} step {step:,}: {m:.4f}", flush=True)
        return m

    def state(self):
        return {"records": self.records}

    def restore(self, d):
        self.records = list(d.get("records") or [])


def dev6(model, verbose=True):
    """DEV-6, ONCE, at the final checkpoint (`evaluation.DEV-6`), never selection-bearing.

    Imported LAZILY: `m9base` installs `paths_guard` on import (CODEMAP pitfall 14), so it must
    not be loaded before the corpus path has read its cached masks. Query prefix is "" — M10's
    query-role examples are raw bytes (prompt policy (b)), which is what the student trained on.
    """
    import m9base                                                   # noqa: F401  (installs guard)
    import eval9
    import guard9
    comps = eval9.components("DEV6")
    if verbose:
        print(f"DEV-6 (once, final checkpoint): {comps}", flush=True)
    per = eval9.eval_student(model, eval9.INCUMBENT, comps=comps, query_prefix="")
    m = eval9.macros(per, eval9.INCUMBENT)["DEV6"]
    ceil = guard9.registry()["ceilings"][eval9.INCUMBENT]["DEV6"]
    return {"teacher": eval9.INCUMBENT, "components": list(comps), "per_component": m["means"],
            "macro": m["macro"], "ceiling_DEV6": ceil,
            "retention": round(m["macro"] / ceil, 4) if ceil else None,
            "_scope": "DEV-6 only; no six-set, reserved or LoTTE surface was read"}


DEV6_MODES = ("inline", "defer")


def dev6_deferred(final_ck, arm):
    """`--dev6 defer`: the runner records that it did NOT read DEV-6 and which checkpoint bytes the
    read must consume; `m13src/dev6_from_checkpoint.py` fills this field on the box from the
    identical final checkpoint, once. Registered use: the two cloud E arms, whose ~35 GB of DEV-6
    document caches are not shipped to the instance (m13/SHIP_LIST.md). DEV-6 stays a read ONCE at
    the final checkpoint and never selection-bearing; only the machine changes."""
    return {"deferred": True, "macro": None, "per_component": None,
            "checkpoint_sha256": (final_ck or {}).get("sha256"),
            "fill_with": f".venv/bin/python m13src/dev6_from_checkpoint.py {arm}",
            "_scope": "DEV-6 NOT read by the runner (deferred to the box); no six-set, reserved or "
                      "LoTTE surface was read either"}


# ------------------------------------------------------------------------------------- the arm --

def streams_of(batch_fn):
    """-> (query_stream, document_stream) behind `assemble_arm`'s batch_fn.

    `assemble_arm` is the ONLY corpus path a launcher may use and it returns the batch function
    alone, but two registered things need the corpus itself: the ridge/G-MLP warm start needs the
    arm's own fit texts and teacher targets, and D-COV's Sigma is the covariance of the pool's
    frozen stella DOCUMENT vectors. Re-opening either pool here would be a second corpus path —
    exactly what that function exists to prevent — so the streams it built are read off its
    closure instead. Asserted by type, never guessed.
    """
    import data10 as D
    q = d = None
    for cell in getattr(batch_fn, "__closure__", None) or ():
        v = cell.cell_contents
        if isinstance(v, CL.FormBalancedStream):
            q = v
        elif isinstance(v, D.Stream):
            d = v
    if q is None or d is None:
        refuse("could not recover the query/document streams from assemble_arm's batch_fn; "
               "corpus_loader.assemble_arm's shape has changed and this runner must be updated")
    return q, d


def fit_sample(q_stream, n_fit, seed):
    """-> (texts, targets) for the warm start, from the arm's own assembled corpus.

    `TargetView` keeps the segments, so the texts behind a global row index are recoverable
    without a second read of the corpus. Draw is `default_rng(seed).choice(n, n_fit)`, sorted —
    the registered sample (`warm_start`, `m9/registry.json` n_fit 60,000 / seed 21).
    """
    tv = q_stream.T
    n = len(tv)
    k = min(int(n_fit), n)
    rows = np.sort(np.random.default_rng(int(seed)).choice(n, size=k, replace=False))
    which = np.searchsorted(tv.bounds, rows, side="right") - 1
    texts = []
    for r, b in zip(rows, which):
        s = tv.segs[int(b)]
        texts.append(s.texts[int(r - tv.bounds[int(b)])])
    return texts, tv[rows]


def warm_start(model, spec, q_stream, ws_cfg, verbose=True):
    """The arm's REGISTERED warm start, actually run. It MUST have run: an arm that silently
    trains from a fresh head is the confound `arm_smoke` exists to prevent, and G3's direction
    depends on G-MLP starting AT the anchor's fitted head."""
    kind = spec.get("warm_start", "linear")
    t0 = time.time()
    texts, Y = fit_sample(q_stream, ws_cfg["n_fit"], ws_cfg["seed"])
    try:
        if kind == "mlp":
            rec = N.warm_start_mlp(model, texts, Y, verbose=verbose)
        elif kind == "m9":
            rec = N.warm_start_from_m9(model, AS.M9_CANDIDATE, verbose=verbose)
        else:
            X = N.pooled_features(model, texts)
            lam, rows = N.select_lambda(X, Y)
            rec = N.warm_start_linear(model, X, Y, lam=lam)
            rec["lambda_rows"] = rows
    except Exception as e:
        import traceback
        print(traceback.format_exc()[-1500:], flush=True)
        refuse(f"the registered warm start ({kind!r}) FAILED: {type(e).__name__}: {e}. An arm "
               f"never trains from an un-warm-started head.")
    rec.update(registered=kind, n_fit_requested=ws_cfg["n_fit"], fit_seed=ws_cfg["seed"],
               n_fit_used=len(texts), seconds=round(time.time() - t0, 1))
    if verbose:
        print(f"  warm start {kind}: n_fit {len(texts):,} lambda {rec.get('lambda')} "
              f"train objective {rec.get('train_objective')} ({rec['seconds']:.0f}s)", flush=True)
    return rec


def classify(stopped, n_cycle_ends, cycles=CYCLES):
    """-> (status, ok, plateau_at_last). SUCCESS is `stopped is None` or exactly "plateau at
    cycle <CYCLES>", with all `cycles` cycle ends read.

    Finding 4: `finished = n_ends >= CYCLES` alone made a KILL at the final cycle end — an arm
    whose curve collapsed at exactly the wrong moment — indistinguishable from a clean finish, and
    it got a DEV-6 read and a checkpoint labelled final. `rules.arm_failure` says the opposite.
    """
    plateau_at_last = stopped == f"plateau at cycle {cycles}"
    ok = bool(n_cycle_ends >= cycles and (stopped is None or plateau_at_last))
    return ("complete" if ok else "failed"), ok, plateau_at_last


# The three files that decide a checkpoint's semantics (item B: code identity). Any edit to any
# one of them must invalidate a resume, the same as an edited registry or manifest.
CODE_IDENTITY_FILES = ("run_arm.py", "trainer10.py", "nano10.py")


def code_identity():
    """sha256 of `run_arm.py` + `trainer10.py` + `nano10.py`'s bytes, concatenated in that order.
    Part of the fingerprint (item B): a code change between a checkpoint and its resume is exactly
    as much a different recipe as a registry or manifest change, and none of the other fingerprint
    fields would catch it."""
    h = hashlib.sha256()
    for name in CODE_IDENTITY_FILES:
        h.update((REPO / "m10src" / name).read_bytes())
    return h.hexdigest()


def fingerprint(arm, p, man, seed, smoke_steps, device):
    """The recipe a checkpoint belongs to (finding 8): the arm, its resolved recipe, the registry,
    the assembled corpus manifest, the device/precision the step ran under, the optimizer settings
    and the code that ran it (item B). A resume under any other is refused by `trainer10`."""
    cuda = str(device) == "cuda"
    body = {"arm": arm, "seed": int(seed), "smoke_steps": smoke_steps,
            "device": str(device), "autocast_dtype": "bf16" if cuda else "fp32",
            "recipe": {k: p[k] for k in ("dose_examples", "batch", "pattern", "student",
                                         "n_layers", "head", "objective", "warm_start",
                                         "total_steps", "cycle_end_steps")},
            # batch-aware since 2026-09-09 (`rules.E_warmup_parity`). At the screen batch of 32
            # this is 2,000, identical to the old constant, so EVERY arm run so far fingerprints
            # unchanged; only `E-bs128`, which has not run, sees a different value.
            "warmup_steps": int(N.warmup_steps_for(p["batch"])),
            "optimizer": {"peak_lr": PEAK, "final_lr": FINAL, "betas": [0.9, 0.999], "eps": 1e-8,
                          "weight_decay_groups": {"dim_gt_1": 0.01, "dim_le_1": 0.0}},
            "registry_sha256": sha256_file(REGISTRY),
            "assemble_manifest_sha256": hashlib.sha256(
                json.dumps(man, sort_keys=True, default=str).encode()).hexdigest(),
            "code_sha256": code_identity()}
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()


# Knobs that would change a registered arm's recipe from the command line. They exist for the
# SMOKE (a 60-step CPU path check needs a short sequence and a tiny warm-start fit) and for
# nothing else: a registered arm reads its recipe from the registry or does not run (finding 5).
SMOKE_ONLY_KNOBS = ("max_len", "ckpt_every", "n_fit", "compile_step", "real_eval")


def run(arm, *, device="cpu", resume=False, smoke_steps=None, max_len=None, ckpt_every=None,
        n_fit=None, real_eval=False, compile_step=False, verbose=True, f_verdict_path=None,
        dev6_mode="inline"):
    """Train one registered arm and write its record. -> the record dict.

    A crash, an OOM or a kill is an OUTCOME (`rules.arm_failure`), so it is recorded and the
    exception is re-raised — never swallowed, and never with a partial checkpoint labelled final.
    """
    ctx = {"started": False}
    try:
        return _run(ctx, arm, device=device, resume=resume, smoke_steps=smoke_steps,
                    max_len=max_len, ckpt_every=ckpt_every, n_fit=n_fit, real_eval=real_eval,
                    compile_step=compile_step, verbose=verbose, f_verdict_path=f_verdict_path,
                    dev6_mode=dev6_mode)
    except SystemExit as e:
        # item D: a REFUSAL raised BEFORE any work starts (screen_lock, an unregistered/untrained/
        # cut arm, a record that already exists, a missing/stale F verdict, ...) means the arm
        # never started, so there is nothing to report and a record here would block the
        # legitimate re-run after the refusal is fixed. But `warm_start`'s own `refuse()` is
        # reached only AFTER the model and corpus are built (`ctx["started"]` is set immediately
        # before it runs) -- that is a FAILURE, not a refusal, and must be recorded like a crash.
        if ctx.get("started"):
            _record_failure(ctx, arm, e, verbose=verbose)
        raise
    except BaseException as e:                                          # noqa: BLE001
        _record_failure(ctx, arm, e, verbose=verbose)
        raise


def _record_failure(ctx, arm, exc, verbose=True):
    """`rules.arm_failure`: "an arm that crashes, OOMs or trips the kill rule has NO final
    checkpoint: its contrasts are reported UNRESOLVED and revert to default, the failure is
    reported, and the arm is not silently re-run". Without this a crashed arm left no record at
    all and the contrast step could not tell it apart from an arm nobody launched (finding 9).
    """
    rec_path, results_path = ctx.get("rec_path"), ctx.get("results_path")
    if rec_path is None:
        return None
    rec = {"_what": "one registered M10 screen arm that FAILED, recorded by m10src/run_arm.py",
           "arm": arm, "family": (ctx.get("plan") or {}).get("family"),
           # item E: a failed record is NOT `complete` -- `terminal` is the flag the re-run
           # protection reads to know the arm already ran to an outcome and blocks a bare re-run.
           "status": "failed", "complete": False, "terminal": True,
           "smoke": bool(ctx.get("smoke")),
           "stopped": f"{type(exc).__name__}: {exc}",
           "stopped_is_completion": False,
           "failure": {"type": type(exc).__name__, "message": str(exc)[:2000],
                       "traceback_tail": traceback.format_exc()[-4000:]},
           "recipe": ctx.get("plan"), "warm_start": ctx.get("warm_start"),
           "cov": {"per_checkpoint": ctx.get("cov_records") or []},
           "dev6": None,
           "final_checkpoint": None, "final_checkpoint_sha256": None,
           "_final_checkpoint_note": "a failed arm has NO final checkpoint; any cycle checkpoint "
                                     "on disk is a partial and is never labelled final",
           "registry_sha256": sha256_file(REGISTRY), "git_head": git_head(),
           "_rule": "rules.arm_failure — reported UNRESOLVED, not re-run at different settings"}
    try:
        write_record(rec, rec_path, results_path)
        if verbose:
            print(f"ARM FAILED ({type(exc).__name__}): wrote {rec_path}", flush=True)
    except Exception as e:                                              # pragma: no cover
        print(f"could not write the failure record: {type(e).__name__}: {e}", flush=True)
    return rec


def _run(ctx, arm, *, device, resume, smoke_steps, max_len, ckpt_every, n_fit, real_eval,
         compile_step, verbose, f_verdict_path=None, dev6_mode="inline"):
    smoke = smoke_steps is not None
    passed = {"max_len": max_len, "ckpt_every": ckpt_every, "n_fit": n_fit,
              "compile_step": compile_step or None, "real_eval": real_eval or None}
    given = sorted(k for k in SMOKE_ONLY_KNOBS if passed[k])
    if given and not smoke:
        refuse(f"{given} may only be passed with --smoke-steps: a registered arm's "
               f"recipe comes from m10/screen_registry.json, not from the command line "
               f"(`rules.arm_failure`: an arm is never re-run at different settings)")
    if real_eval and smoke:
        # findings 5 and 10 together: the real COV/DEV-6 read is never caller-selectable. A smoke
        # must not touch the 13,416-query surface or DEV-6's ~13 GB, and a real arm always does.
        refuse("--real-eval is prohibited under --smoke-steps: a smoke never touches the COV "
               "surface or DEV-6. Run the arm itself for a real evaluation.")
    if dev6_mode not in DEV6_MODES:
        refuse(f"--dev6 {dev6_mode!r}: choose from {list(DEV6_MODES)}")
    if dev6_mode == "defer" and smoke:
        refuse("--dev6 defer is meaningless under --smoke-steps: a smoke never reads DEV-6, so "
               "there is nothing to defer")
    max_len = int(max_len or MAX_LEN)
    problems = SL.validate(SL.cfg())
    if problems:
        refuse(f"screen_lock.validate reports {len(problems)} problem(s), so §0a is not coherent "
               f"and no arm may run: {problems}")
    reg = SL.cfg()
    entry = (reg.get("arms") or {}).get(arm)
    if entry is None:
        refuse(f"{arm!r} is not a registered arm (m10/screen_registry.json `arms`)")
    if entry.get("trained") is not True:
        refuse(f"arm {arm!r} is not `trained: true` — {entry.get('note') or 'not a trained arm'}")
    if entry.get("cut"):
        refuse(f"arm {arm!r} is CUT: {entry['cut']}")

    base = arm_plan(arm, reg)               # batch and CLOUD note only; the student comes below
    if base["cloud_only"] and device == "cuda" and not smoke:
        print(f"NOTE {arm}: {base['cloud_only']}", flush=True)

    # a smoke gets its OWN registry copy, so `assemble_arm` derives a tiny document count from a
    # tiny dose rather than 1.25M documents, and its OWN output tree (§Hazards: smokes have
    # overwritten real artifacts in this milestone twice).
    if smoke:
        reg = copy.deepcopy(reg)
        reg["arms"][arm]["dose_examples"] = int(smoke_steps) * base["batch"]
        reg["arms"][arm].pop("read_at", None)
    out_dir, rec_path, results_path = record_paths(arm, smoke)
    ctx.update(rec_path=rec_path, results_path=results_path, smoke=smoke)
    # item E: ANY record file at EITHER published path blocks a bare re-run, parseable or not,
    # complete or not -- `rules.arm_failure` is "reported, not silently re-run", and a half-written
    # or in-progress record is exactly as much a re-run hazard as a complete one. `--resume`
    # continues a NON-terminal record; a TERMINAL one (success or FAILED) still refuses -- there
    # is nothing left for `--resume` to continue.
    resume_records = []
    for path in (rec_path, results_path):
        st = _record_status_at(path)
        if st is None:
            continue
        if not resume:
            refuse(f"{path} already exists. `rules.arm_failure`: an arm is reported, not "
                   f"silently re-run. Pass --resume to continue a non-terminal run, or move "
                   f"the record aside deliberately.")
        if not st["parseable"]:
            refuse(f"{path} is unparseable, so whether the arm already finished is UNKNOWABLE; "
                   f"`--resume` refuses rather than guess. Inspect and move it aside deliberately.")
        if st["terminal"]:
            refuse(f"{path} already carries a TERMINAL record (complete or failed); `--resume` "
                   f"only continues a non-terminal run. Move the record aside deliberately.")
        resume_records.append((path, st["raw"]))
    if resume and not smoke:
        # Codex pass 3: `--resume` must CONTINUE a run, never silently become a fresh one. It
        # needs both a non-terminal work record (proof an arm started) and the rolling checkpoint.
        if _record_status_at(rec_path) is None:
            refuse(f"--resume: no record at {rec_path}; there is no run to continue. Start the "
                   f"arm without --resume.")
        if not (out_dir / "ckpt.pt").exists():
            refuse(f"--resume: no rolling checkpoint at {out_dir / 'ckpt.pt'}; a resume never "
                   f"falls back to a fresh run.")
    elif not smoke and (out_dir / "ckpt.pt").exists():
        refuse(f"{out_dir / 'ckpt.pt'} already exists without a running receipt; "
               "refusing to overwrite an orphaned checkpoint with a fresh run.")
    p = arm_plan(arm, reg)                      # re-read: a smoke changed the dose
    ctx["plan"] = p
    if p["cut_corpus"] and CL.data_cut_count(reg) is None:
        # `assemble_arm` is the enforcer (it refuses with no `allow_uncut` escape); failing here
        # first only avoids reading a 5M-row corpus to learn it.
        refuse(f"arm {arm!r} trains on the registered CUT corpus and "
               f"`data_cut.unique_text_count` is unregistered (§0b open). Register the count.")

    # Every arm after family F trains on F's WINNER backbone, read from the verdict file and not
    # from `arm_smoke.SHAPES` (finding 11). A SMOKE is exempt — it is a path check on a shape, the
    # same exemption `arm_smoke` itself has — and its record says which student it actually ran.
    verdict = None
    if entry.get("family") != "F" and not smoke:
        verdict = f_verdict(reg, f_verdict_path)
        p = arm_plan(arm, reg, verdict=verdict)
        ctx["plan"] = p
    out_dir.mkdir(parents=True, exist_ok=True)

    # A registered run REFUSES anything but CUDA (item A): §Recipe is bf16 autocast, and CPU
    # silently trains fp32 — a confound, not a smoke. A smoke is exempt, same as every other
    # recipe-shaping check in this file.
    if not smoke and device != "cuda":
        refuse(f"arm {arm!r}: a registered (non-smoke) run requires --device cuda (got {device!r})"
               f" — the recipe is bf16 autocast and CPU silently means fp32. Pass --smoke-steps "
               f"for a CPU path check.")

    ws_cfg = dict(WS_DEFAULTS)
    ws_cfg.update({k: v for k, v in ((reg.get("warm_start") or {}).get("G-MLP") or {}).items()
                   if k in ("n_fit", "seed")})
    if smoke and not n_fit:
        ws_cfg["n_fit"] = SMOKE_N_FIT
    if n_fit:
        ws_cfg["n_fit"] = int(n_fit)
    # `seed_rule`: "one seed, seed 0, for every screen arm". An ARM may carry its own `seed` and
    # exactly one does -- `ANCHOR-seed1`, the descriptive seed-sensitivity run (astra's decision 2,
    # 2026-09-09). Read per-arm so a seed can never be passed in at a launch: it is a registered
    # property of the arm, like its dose.
    entry_seed = (reg.get("arms") or {}).get(arm, {}).get("seed")
    seed = (int(entry_seed) if entry_seed is not None
            else (0 if "seed" not in reg.get("anchor", {}) else int(reg["anchor"]["seed"])))

    t_start = time.time()
    print(f"=== arm {arm} ({p['family']}) on {device}: {p['dose_examples']:,} examples, "
          f"batch {p['batch']}, {p['total_steps']:,} steps, pattern {p['pattern']}, "
          f"student {p['student']}, objective {p['objective']}, "
          f"warm start {p['warm_start']}, seed {seed}"
          + (f"  [SMOKE {smoke_steps} steps]" if smoke else ""), flush=True)

    torch.manual_seed(seed)
    model = N.Nano10(p["student"], n_layers=p["n_layers"], head=p["head"]).to(device)
    if not model.under_cap():
        refuse(f"{arm}: {model.n_params():,} parameters exceeds the {N.CAP:,} cap")
    print(f"  student {p['student']}: d_in {model.d_in}, {model.n_params():,} params", flush=True)

    batch_fn, man = CL.assemble_arm(arm, model.tok, p["student"], batch_size=p["batch"],
                                    seed=seed, max_len=max_len, verbose=verbose, registry=reg)
    q_stream, d_stream = streams_of(batch_fn)
    fp = fingerprint(arm, p, man, seed, smoke_steps, device)
    if not smoke:
        identity = {"arm": arm, "smoke": False, "recipe_fingerprint": fp,
                    "registry_sha256": sha256_file(REGISTRY)}
        if resume:
            for path, receipt in resume_records:
                if receipt.get("status") != "running" or any(
                        receipt.get(k) != v for k, v in identity.items()):
                    refuse(f"--resume: {path} has missing or mismatched running identity; "
                           "refusing to continue another recipe or an unauthenticated record.")
        else:
            receipt = {"_what": "M10 arm started; abrupt process loss may resume its checkpoint",
                       **identity, "status": "running", "complete": False, "terminal": False,
                       "git_head": git_head()}
            try:
                write_record(receipt, rec_path, None, create_only=True)
            except FileExistsError:
                refuse(f"{rec_path} already exists; refusing to replace another run's receipt.")
    # item D: everything from here on is WORK -- a warm-start failure (or anything after) is a
    # FAILURE to be recorded, not a pre-flight refusal.
    ctx["started"] = True
    ws = warm_start(model, {"warm_start": p["warm_start"]}, q_stream, ws_cfg, verbose=verbose)

    sigma = None
    if p["objective"] == "document_covariance_weighted":
        sigma = torch.from_numpy(np.asarray(N.cov_matrix(d_stream.T), dtype=np.float32)).to(device)

    total, ends, reads = p["total_steps"], p["cycle_end_steps"], p["read_points"]
    read_steps = {int(r["step"]): int(r["examples"]) for r in reads}
    extra_reads = sorted(s for s in read_steps if s not in set(ends))
    ev = StubEval() if smoke else CovEval(model, out_dir, verbose=verbose)
    ctx["cov_records"] = ev.records

    def eval_fn(_m, step, kind):
        if kind == "end":
            label = f"cycle{ends.index(step) + 1}"
        elif kind == "mid":
            label = f"mid{step}"
        else:
            label = f"read{read_steps[step]}"
        if kind != "read" and step in read_steps:
            # a read point that coincides with a cycle end (20M) or a midpoint (10M): one eval,
            # labelled as both, so the record's `reads` list finds it.
            label = f"{label}_read{read_steps[step]}"
        return ev(step, label)

    ck = out_dir / "ckpt.pt"
    every = int(ckpt_every or max(total // 20, 1))
    train_model = torch.compile(model) if compile_step else model
    r = Tr.train_arm(train_model, batch_fn, total_steps=total, pattern=p["pattern"],
                     cycles=CYCLES, peak=PEAK, final=FINAL, loss_name=p["objective"],
                     sigma=sigma, eval_fn=eval_fn, ckpt_path=ck, ckpt_every=every,
                     resume_from=(str(ck) if resume else None), seed=seed,
                     log_every=max(total // 50, 1), device=device, batch_size=p["batch"],
                     read_steps=extra_reads, cycle_ckpt_fmt=str(out_dir / "cycle{cycle}.pt"),
                     eval_state=ev, fingerprint=fp)

    status, ok, plateau_at_last = classify(r["stopped"], len(r["cycle_end_evals"]))
    stopped = r["stopped"]
    final_ck = out_dir / f"cycle{CYCLES}.pt"
    cks = {}
    for c in range(1, CYCLES + 1):
        pth = out_dir / f"cycle{c}.pt"
        if pth.exists():
            cks[f"cycle{c}"] = {"path": rel(pth), "sha256": sha256_file(pth)}
    if ck.exists():
        cks["rolling"] = {"path": rel(ck), "sha256": sha256_file(ck)}

    d6 = None
    if ok and not smoke:
        if dev6_mode == "defer":
            d6 = dev6_deferred(cks.get(f"cycle{CYCLES}"), arm)
            print(f"DEV-6 DEFERRED to the box (--dev6 defer): fill it with `{d6['fill_with']}` "
                  f"from checkpoint {str(d6['checkpoint_sha256'])[:12]}", flush=True)
        else:
            d6 = dev6(model, verbose=verbose)
    elif not ok:
        print(f"ARM FAILED ({stopped}): no final checkpoint, so no DEV-6 read. "
              f"`rules.arm_failure`: its contrasts are reported UNRESOLVED and revert to default; "
              f"the arm is NOT re-run at different settings.", flush=True)

    rec = {
        "_what": "one registered M10 screen arm, trained end to end by m10src/run_arm.py",
        # item E: `complete` tracks the OUTCOME (only a "complete" status is complete); `terminal`
        # is always true once the record is written -- either outcome blocks a bare re-run.
        "arm": arm, "family": p["family"], "status": status, "complete": ok, "terminal": True,
        "smoke": smoke,
        "smoke_steps": smoke_steps, "device": device, "max_len": max_len,
        "seed": seed, "seed_rule": reg.get("seed_rule"),
        "recipe": {k: p[k] for k in ("dose_examples", "batch", "pattern", "pattern_source",
                                     "student", "n_layers", "head", "objective", "warm_start",
                                     "cut_corpus", "sources", "n_docs", "total_steps",
                                     "cycle_end_steps", "read_points", "dose_rounding")},
        "schedule": {"cycles": CYCLES, "peak": PEAK, "final": FINAL,
                     "_what": "3 cycles of equal example count over the ARM'S OWN dose, each "
                              "linear 1e-4 -> 1e-5 (§Recipe)"},
        "params": model.n_params(), "under_cap": model.under_cap(),
        "warm_start": ws,
        "evaluation_mode": "stub" if isinstance(ev, StubEval) else "cov",
        "cov": {"per_checkpoint": ev.records,
                "cycle_end_macros": r["cycle_end_evals"],
                "final_macro": (r["cycle_end_evals"][-1] if r["cycle_end_evals"] else None),
                "reads": [x for x in ev.records if "read" in x["label"]],
                "_scope": "COV only at every cycle end (plus the schedule midpoints the kill rule "
                          "reads) and family F's read_at points"},
        "dev6": d6,
        "training": {k: r[k] for k in ("steps_run", "start_step", "total_steps", "stopped",
                                       "examples", "examples_this_run", "seconds",
                                       "examples_per_s", "mix", "evals", "eval_kinds",
                                       "cycle_end_evals", "read_evals")},
        "throughput_ex_per_s": r["examples_per_s"],
        "loss_tail": r["losses"][-200:],
        "stopped": stopped,
        "stopped_is_completion": plateau_at_last,
        "_stopped_note": ("SUCCESS is `stopped is None` or exactly `plateau at cycle 3`: "
                          "PLATEAU_FROM_CYCLE is 3 and a screen arm runs 3 cycles, so that "
                          "plateau means the arm finished its dose and stopped one step short "
                          "(LEDGER §M10.0-e trace). EVERY kill and every non-finite stop is "
                          "FAILED even at the final cycle end (finding 4)."),
        "checkpoints": cks,
        # A FAILED arm has no final checkpoint, whatever is on disk: a kill at the last cycle end
        # still leaves `cycle3.pt`, and labelling that "final" is how a killed arm gets exported
        # (finding 4). The cycle checkpoints are listed under their own cycle keys either way.
        "final_checkpoint": rel(final_ck) if ok and final_ck.exists() else None,
        "final_checkpoint_sha256": ((cks.get(f"cycle{CYCLES}") or {}).get("sha256")
                                    if ok else None),
        "recipe_fingerprint": fp,
        "f_verdict": p.get("f_verdict"), "student_source": p.get("student_source"),
        "assemble_manifest": man,
        "registry_sha256": sha256_file(REGISTRY),
        "registry_dose_overridden_for_smoke": (reg["arms"][arm]["dose_examples"] if smoke else None),
        "git_head": git_head(),
        "wall_seconds": round(time.time() - t_start, 1),
        "_contrasts": "NOT computed here — the contrast step reads the per-query COV files and "
                      "passes each contrast's own registered quantile",
    }
    for w in write_record(rec, rec_path, results_path):
        print(f"wrote {w}", flush=True)
    print(f"{arm}: {status}  COV final {rec['cov']['final_macro']}  "
          f"{r['examples_per_s']:.0f} ex/s  {rec['wall_seconds']:.0f}s", flush=True)
    return rec


def build_argparser():
    """Factored out of `main()` so a test can check CLI defaults (item A) without launching a
    run."""
    ap = argparse.ArgumentParser(description="run one registered M10 screen arm")
    ap.add_argument("arm", nargs="?", default=None)
    # A registered (non-smoke) run REFUSES anything but cuda (Codex runner review 2026-09-07,
    # item A): the recipe is bf16 autocast and CPU silently means fp32. A smoke may still pass
    # `--device cpu`. The default reflects that a bare invocation is presumed registered.
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--resume", action="store_true",
                    help="continue from work/m10arms/<arm>/ckpt.pt (evaluation state included)")
    ap.add_argument("--smoke-steps", type=int, default=None,
                    help="a SMOKE: N steps, its own dose, stub evals, and its own output tree "
                         "under work/m10arms/smoke/ — never the real record path")
    ap.add_argument("--plan", action="store_true", help="print the W8-band-1 plan; train nothing")
    # SMOKE-ONLY, every one of them (finding 5): a registered arm's recipe comes from the
    # registry. `--real-eval` is gone entirely — findings 5 and 10 together make the real COV and
    # DEV-6 reads non-selectable: a smoke never takes them and a real arm always does.
    ap.add_argument("--max-len", type=int, default=None,
                    help="SMOKE ONLY; a registered arm runs at %d" % MAX_LEN)
    ap.add_argument("--ckpt-every", type=int, default=None,
                    help="SMOKE ONLY; default total_steps // 20")
    ap.add_argument("--n-fit", type=int, default=None,
                    help="SMOKE ONLY warm-start fit sample; a registered arm uses 60,000")
    ap.add_argument("--compile", action="store_true",
                    help="SMOKE ONLY: torch.compile the training step (checkpoints eager, §T)")
    # Not a recipe knob: DEV-6 is read ONCE at the final checkpoint either way. `defer` records
    # that the read did not happen HERE and names the checkpoint bytes it must consume;
    # `m13src/dev6_from_checkpoint.py` performs it on the box (the cloud E arms, whose DEV-6
    # caches are not shipped — m13/SHIP_LIST.md). Meaningless under --smoke-steps and refused there.
    ap.add_argument("--dev6", default="inline", choices=list(DEV6_MODES),
                    help="inline (default): read DEV-6 at the final checkpoint here; defer: record "
                         "the deferral and fill it later with m13src/dev6_from_checkpoint.py")
    ap.add_argument("--quiet", action="store_true")
    return ap


def main(argv=None):
    ap = build_argparser()
    a = ap.parse_args(argv)
    if a.plan:
        plan(SL.cfg())
        return 0
    if not a.arm:
        ap.error("an arm name is required (or --plan)")
    run(a.arm, device=a.device, resume=a.resume, smoke_steps=a.smoke_steps, max_len=a.max_len,
        ckpt_every=a.ckpt_every, n_fit=a.n_fit, compile_step=a.compile, verbose=not a.quiet,
        dev6_mode=a.dev6)
    return 0


if __name__ == "__main__":
    sys.exit(main())
