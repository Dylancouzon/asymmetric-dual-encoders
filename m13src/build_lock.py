"""The build configuration's validator: `m13/build_config.json` against the LIVE screen registry.

`m13src/build13.py` refuses to do anything before this passes. What it exists to catch is the one
failure the 200M build cannot recover from: training a recipe that is not the screened one. The
lock (`m10/M102_LOCK.md`) selects a recipe by pointing at `m10/screen_registry.json`'s anchor, so
"the build arm equals ANCHOR except dose, batch, documents and cut" is checkable — and it is
checked against the registry that is on disk NOW, not against a copy in this file.

The registry sha is RECORDED, not pinned: ruling R9 (strip `pending` from both E arms and re-issue
the F verdict) changes it before the build starts, and a pin would refuse the registry the build
is meant to run under. What is pinned is the content — every data knob, field by field.

Nothing here trains, reads a corpus, or touches a protected surface.
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for p in ("m7src", "m9src", "m10src", "m13src"):
    if str(REPO / p) not in sys.path:
        sys.path.insert(0, str(REPO / p))

import corpus_loader as CL           # noqa: E402
import nano10 as N                   # noqa: E402
import run_arm as R                  # noqa: E402
import screen_lock as SL             # noqa: E402

CONFIG = REPO / "m13" / "build_config.json"
VERDICTS = REPO / "results" / "m10_screen_verdicts.json"
DOSE = 200_000_000
CYCLES = 3
PENDING = "PENDING"

# The only fields the build arm's registry entry may carry. Anything else would be a data or
# recipe knob smuggled past the "equal to ANCHOR" check.
ENTRY_KEYS = {"family", "trained", "dose_examples", "batch", "seed", "data_cut",
              "require_all_forms", "documents", "note"}
ALLOWED_DIFFS = ("dose_examples", "batch", "documents", "data_cut")


def refuse(msg):
    raise SystemExit(f"REFUSED: {msg}")


def load(path=None):
    p = Path(path or CONFIG)
    if not p.exists():
        refuse(f"no build configuration at {p}")
    try:
        return json.loads(p.read_text()), p
    except Exception as e:
        refuse(f"{p} is not readable JSON: {type(e).__name__}: {e}")


def verdicts(path=None):
    p = Path(path or VERDICTS)
    if not p.exists():
        refuse(f"no screen verdicts at {p}; the build's batch and student come from the screen")
    return json.loads(p.read_text())


def resolve_batch(cfg, *, smoke=False, override=None, verdicts_path=None):
    """-> (batch, source). The batch is the E1 verdict's, never the configuration's own value.

    `rules.E_cost` is read for the build by `contrasts` and recorded in
    `results/m10_screen_verdicts.json:selected.batch`; while family E has not run that field is
    'PENDING' and a registered build cannot start. A SMOKE may pass `--batch`, recorded as such.
    """
    if override is not None:
        if not smoke:
            refuse("--batch may only be passed with --smoke-steps: the build's batch is the E1 "
                   "verdict's (results/m10_screen_verdicts.json:selected.batch)")
        return int(override), "--batch (SMOKE ONLY)"
    sel = (verdicts(verdicts_path).get("selected") or {}).get("batch")
    if sel is None:
        refuse("results/m10_screen_verdicts.json carries no `selected.batch`")
    if str(sel).upper() == PENDING:
        refuse("the E1 verdict is PENDING: `selected.batch` in results/m10_screen_verdicts.json "
               "is not a batch size. Both E arms run on the cloud GPU and `rules.E_cost` is read "
               "BEFORE the build starts (m10/M102_LOCK.md). Pass --smoke-steps with --batch for a "
               "path check.")
    try:
        return int(sel), "results/m10_screen_verdicts.json:selected.batch"
    except (TypeError, ValueError):
        refuse(f"`selected.batch` is {sel!r}, not a batch size")


def cycle_plan(dose, batch, cycles=CYCLES):
    """-> the cycle table. `total_steps = dose // batch` must divide EXACTLY for the build: the
    lock's "three x 66.7M" is rounded and 200.1M is a different dose, not a rounding (M102_LOCK
    "Do not silently train 200.1M")."""
    dose, batch = int(dose), int(batch)
    total = dose // batch
    ends = N.cycle_ends(total, cycles)
    starts = [0] + [e + 1 for e in ends[:-1]]
    rows = [{"cycle": i + 1, "start_step": s, "end_step": e, "steps": e - s + 1,
             "examples": (e - s + 1) * batch} for i, (s, e) in enumerate(zip(starts, ends))]
    return {"dose_examples": dose, "batch": batch, "total_steps": total,
            "divides_exactly": dose == total * batch, "cycle_end_steps": ends, "cycles": rows,
            "sum_examples": sum(r["examples"] for r in rows)}


def extension_plan(extension_examples, batch):
    """-> one extension cycle in steps. bs128 floors 66,700,000 to 521,093 steps = 66,699,904
    examples; the 96 dropped examples are recorded, never silent (the same rule as
    `run_arm.schedule`: the dose is a cap)."""
    steps = int(extension_examples) // int(batch)
    ran = steps * int(batch)
    return {"registered_examples": int(extension_examples), "steps": steps, "examples": ran,
            "examples_dropped": int(extension_examples) - ran, "rule": "floor (the dose is a cap)"}


def cap_arithmetic(cfg, rate_ex_s, price_usd_h, *, smoke=False):
    """The day-one cap (`archive .../M102_LOCK.md:126`): mandatory lines first at the measured
    rate and the BILLED price, then whole extension cycles out of what is left."""
    rate, price = float(rate_ex_s), float(price_usd_h)
    if rate <= 0 or price <= 0:
        refuse(f"the benchmark needs a positive rate and price (got {rate!r} ex/s, ${price!r}/h)")
    b = cfg["budget"]
    fixed_h = dict(b["mandatory_hours_fixed"])
    missing = sorted(k for k, v in fixed_h.items() if v is None)
    if missing:
        if not smoke:
            refuse(f"the M13 allocation has no line for {missing} (ruling R10: the conditional "
                   f"reserved batch is unpriced and must be a NAMED line before the build "
                   f"starts). Register the hours in m13/build_config.json "
                   f"`budget.mandatory_hours_fixed`.")
        # a SMOKE prices an unregistered line at zero and SAYS SO, so the path can be exercised
        # before the owner ruling lands. A registered benchmark refuses instead.
        fixed_h.update({k: 0.0 for k in missing})
    rate_h = {k: v / rate / 3600 for k, v in b["mandatory_examples"].items()}
    hours = {**fixed_h, **rate_h}
    fixed_usd = sum(b["fixed_usd"].values())
    mandatory_usd = price * sum(hours.values())
    cycle_h = float(cfg["extension_examples"]) / rate / 3600
    cycle_usd = price * cycle_h
    remaining = float(cfg["budget_ceiling_usd"]) - fixed_usd - mandatory_usd
    cap = max(int(remaining // cycle_usd), 0) if cycle_usd > 0 else 0
    return {"rate_ex_per_s": rate, "price_usd_per_h": price,
            "mandatory_hours": {k: round(v, 3) for k, v in hours.items()},
            "mandatory_hours_total": round(sum(hours.values()), 3),
            "mandatory_usd": round(mandatory_usd, 2), "fixed_usd": round(fixed_usd, 2),
            "committed_usd": round(mandatory_usd + fixed_usd, 2),
            "extension_cycle_hours": round(cycle_h, 3),
            "extension_cycle_usd": round(cycle_usd, 2),
            "remaining_usd": round(remaining, 2),
            "budget_ceiling_usd": float(cfg["budget_ceiling_usd"]),
            "max_extension_cycles": cap,
            "unpriced_lines_zeroed_for_smoke": missing if smoke else [],
            "formula": cfg["budget"]["formula"]}


def _entry_of(cfg):
    return dict(cfg["arm_entry"])


def merged_registry(cfg, reg=None, *, batch, smoke_documents=None):
    """-> a COPY of the screen registry with the build arm added. `m10/screen_registry.json` is
    never written: the screen verdicts and the F verdict are bound to its bytes."""
    reg = json.loads(json.dumps(reg or SL.cfg()))
    entry = _entry_of(cfg)
    entry["batch"] = int(batch)
    if smoke_documents is not None:
        entry["documents"] = {**entry["documents"], "n": int(smoke_documents)}
    reg["arms"][cfg["arm"]] = entry
    return reg


def validate(cfg, reg=None, *, smoke=False, verdicts_path=None, registry_path=None):
    """-> the validation report. Raises SystemExit on the first refusal.

    Checks, in order: the screen lock validates; the arm entry carries no unexpected knob; every
    data knob equals ANCHOR's resolved value; the differences are exactly the four registered
    ones; the dose is 200M and divides by the batch; the cycle example counts sum to the dose.
    """
    reg = reg or SL.cfg()
    problems = SL.validate(reg)
    if problems:
        refuse(f"screen_lock.validate reports {len(problems)} problem(s), so §0a is not coherent "
               f"and no build may run: {problems}")
    arm = cfg["arm"]
    if arm in (reg.get("arms") or {}):
        refuse(f"{arm!r} is already a registered screen arm; the build arm is added to an "
               f"in-memory copy of the registry and must not collide with one")
    entry = _entry_of(cfg)
    extra = sorted(set(entry) - ENTRY_KEYS)
    if extra:
        refuse(f"the build arm entry carries {extra}, which are not among {sorted(ENTRY_KEYS)}: "
               f"every other knob must be the anchor's, not the configuration's")
    anchor = reg.get("anchor") or {}
    k = cfg["data_knobs_equal_to_anchor"]
    bad = []
    for field, got in (("mix", k["mix"]), ("window_pattern", k["window_pattern"]),
                       ("objective", k["objective"]), ("feature_layers", k["feature_layers"]),
                       ("feature_dim", k["feature_dim"]), ("seed", k["seed"])):
        if anchor.get(field) != got:
            bad.append(f"{field}: config {got!r} vs anchor {anchor.get(field)!r}")
    if entry.get("seed") != anchor.get("seed"):
        bad.append(f"arm_entry.seed {entry.get('seed')!r} vs anchor {anchor.get('seed')!r}")
    if anchor.get("init") != "F's winner backbone":
        bad.append(f"anchor.init is {anchor.get('init')!r}; the build's student is F's winner")
    sel = (verdicts(verdicts_path).get("selected") or {})
    if sel.get("student") != k["student"]:
        bad.append(f"student: config {k['student']!r} vs screen verdict {sel.get('student')!r}")
    if tuple(k["sources"]) != tuple(CL.ARM_SOURCES["ANCHOR"]):
        bad.append(f"sources: config {tuple(k['sources'])} vs ANCHOR "
                   f"{tuple(CL.ARM_SOURCES['ANCHOR'])}")
    if k["head"] != "linear":
        bad.append(f"head: config {k['head']!r}; the anchor's head is the 1152-wide linear one")
    want_layers = N.LAYERS[k["student"]][int(k["feature_layers"])]
    if tuple(k["head_layers"]) != tuple(want_layers):
        bad.append(f"head_layers: config {tuple(k['head_layers'])} vs nano10.LAYERS {want_layers}")
    ws = dict(R.WS_DEFAULTS)
    ws.update({a: b for a, b in ((reg.get("warm_start") or {}).get("G-MLP") or {}).items()
               if a in ("n_fit", "seed")})
    if (int(k["warm_start"]["n_fit"]), int(k["warm_start"]["fit_seed"])) != (ws["n_fit"],
                                                                            ws["seed"]):
        bad.append(f"warm_start: config {k['warm_start']} vs registry {ws}")
    if k["warm_start"]["kind"] != "linear":
        bad.append(f"warm_start.kind {k['warm_start']['kind']!r}; `warm_start.all_arms` is the "
                   f"closed-form ridge head")
    if bad:
        refuse(f"the build arm's data knobs differ from ANCHOR's — " + "; ".join(bad))
    if tuple(cfg["differs_from_anchor"]) != ALLOWED_DIFFS:
        refuse(f"`differs_from_anchor` is {cfg['differs_from_anchor']}, not {list(ALLOWED_DIFFS)}")

    if int(entry["dose_examples"]) != DOSE:
        refuse(f"the registered build dose is {DOSE:,}; the configuration says "
               f"{int(entry['dose_examples']):,}. Any dose above 200M is an EXTENSION CYCLE under "
               f"the extension rule, never a silent increase (M102_LOCK).")
    pol = CL.arm_document_policy(entry)
    if pol is None:
        refuse("the build arm carries no `documents` policy, so corpus_loader.arm_doc_count would "
               "demand ceil(200,000,000 x 0.25) = 50,000,000 unique documents from a ~6.15M pool")
    if pol.get("n") is not None and not smoke:
        refuse(f"`documents.n` = {pol['n']!r} is a SMOKE-ONLY override; a registered build draws "
               f"every eligible re-screened document")
    if str(entry.get("data_cut", "")).lower() != "none":
        refuse("the build trains the FULL uncut A4 (`arm_entry.data_cut` must be \"none\"); the "
               "screen's cut applies only to data_cut.applies_to")
    if str(cfg.get("torch_compile")) != "smoke_only":
        refuse(f"`torch_compile` is {cfg.get('torch_compile')!r}; it is registered SMOKE-ONLY")
    if float(cfg["budget_ceiling_usd"]) != 1000:
        refuse(f"the recorded ceiling is $1,000; the configuration says "
               f"{cfg['budget_ceiling_usd']!r}")
    if abs(float(cfg["extension_min_gain"]) - N.PLATEAU_MIN_GAIN) > 1e-12:
        refuse(f"`extension_min_gain` {cfg['extension_min_gain']!r} != the registered "
               f"{N.PLATEAU_MIN_GAIN} (nano10.PLATEAU_MIN_GAIN)")
    reg_path = Path(registry_path or R.REGISTRY)
    return {"config_arm": arm, "registry": R.rel(reg_path),
            "registry_sha256": R.sha256_file(reg_path),
            "registry_sha256_pinned": cfg.get("registry_sha256"),
            "anchor_fields_checked": ["mix", "window_pattern", "objective", "feature_layers",
                                      "feature_dim", "seed", "init", "sources", "head",
                                      "head_layers", "warm_start", "student"],
            "differs_from_anchor": list(ALLOWED_DIFFS),
            "dose_examples": DOSE, "document_policy": pol, "data_cut": "none",
            "smoke": bool(smoke)}


def check_dose(cfg, batch):
    """The 200M pin, as arithmetic. Refuses a dose that does not divide by the batch and a cycle
    table that does not sum to exactly the dose."""
    plan = cycle_plan(cfg["arm_entry"]["dose_examples"], batch)
    if not plan["divides_exactly"]:
        refuse(f"the dose {plan['dose_examples']:,} is not a whole number of batches of "
               f"{plan['batch']} ({plan['dose_examples'] / plan['batch']:.2f} steps); the build's "
               f"200M pin is exact, unlike a screen arm's floored dose")
    if plan["sum_examples"] != plan["dose_examples"]:
        refuse(f"the cycle example counts sum to {plan['sum_examples']:,}, not "
               f"{plan['dose_examples']:,}")
    return plan
