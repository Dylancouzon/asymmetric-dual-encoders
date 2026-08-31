"""Generate `work/m9long/config.json` — the M9.3 build's every numeric knob, in one place.

Kept out of `m9src/longrun.py` deliberately: the trainer lives in the guarded `build` scope, so
iterating on the generator here does not churn the hash that a running build is pinned to. The
config it writes IS in that scope, which is the correct place for the binding.

Everything that came from a screen decision is read from `results/m9_screen_decisions.json`; every
dose figure is derived from measured quantities and shown with its arithmetic, so a reviewer can
check the numbers rather than take them.
"""
import hashlib
import json
import math
from numbers import Real
import subprocess

import m9base
from m9base import RESULTS, WORK

import guard9   # noqa: E402
import longrun  # noqa: E402

# Measured on this box by the anchor arm: 59,507,872 non-pad tokens in 2,216 s.
TOK_PER_S = 26_854
HOURS = 168                                   # the horizon we size for; we stop on evidence

# Codex review #5's registration, adopted. At a true combined-example mean, queries are short
# enough that a 5% TOKEN share is ~23% of the objective, while cutting real-query repetition from
# ~438 presentations to ~109. The 20/10/70 draft was rejected for the repetition, not the weight.
SHARES = {"queries": 0.05, "spans": 0.05, "documents": 0.90}

# ~113 examples a step at the corpus means. LEAF found batch 32 beat 256 because dense supervision
# favours more updates; this is the same order once documents dominate, and it keeps the GPU busy.
TOKENS_PER_STEP = 8192

# The only annealing evidence this project owns is the anchor's own run: 59.5M tokens. Codex #5 was
# right that the 4,000-step default was five minutes and unsupported by anything.
COOLDOWN_TOKENS = 59_507_872

SHARE_KEYS = {"queries", "spans", "documents"}
RULING_DECISION = "results/m9_screen_decisions.json"
# The ruling is post-screen authorization, not a mutable config slot. Pinning the whole canonical
# record here means changed shares/rationale/caveat require an explicit build-scope code change.
M9S6_RULING_SHA256 = "fcab490443f400809631e57a9eee0ff52c73ea3ad47aecc97bb3fbf510677d35"


class DecisionIntegrityError(ValueError):
    """A final screen artifact is not the immutable, registered lineage it claims to be."""


def _refuse(message):
    raise DecisionIntegrityError(message)


def _scope_sha(scope):
    """Reproduce guard9's hash for one current scope without hashing unrelated scopes."""
    h = hashlib.sha256()
    for name in guard9.SCOPES[scope]:
        path = m9base.REPO / name
        digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "MISSING"
        h.update(name.encode())
        h.update(digest.encode())
    return h.hexdigest()


def _git_blob(commit, name):
    proc = subprocess.run(
        ["git", "show", f"{commit}:{name}"], cwd=m9base.REPO, capture_output=True)
    return proc.stdout if proc.returncode == 0 else None


def _historical_scope_sha(commit, scope):
    h = hashlib.sha256()
    for name in guard9.SCOPES[scope]:
        raw = _git_blob(commit, name)
        digest = hashlib.sha256(raw).hexdigest() if raw is not None else "MISSING"
        h.update(name.encode())
        h.update(digest.encode())
    return h.hexdigest()


def _verify_record_only_protocol_drift(registration):
    """Allow post-screen records, but no post-screen change to a rule the screen read.

    LEDGER may only have bytes appended. registry may only gain owner-ruling entries; everything
    else in it, including thresholds and arm definitions, must remain byte-for-JSON identical.
    Every other protocol dependency must remain byte-identical to the registered commit.
    """
    commit = registration.get("commit")
    if not isinstance(commit, str) or len(commit) != 40:
        _refuse("decision registration has no full 40-character commit")
    recorded = registration.get("scoped", {}).get("protocol")
    historical = _historical_scope_sha(commit, "protocol")
    if historical != recorded:
        _refuse(
            f"decision protocol fingerprint does not match registered commit {commit[:12]}: "
            f"artifact records {recorded}, commit reconstructs {historical}")

    for name in guard9.SCOPES["protocol"]:
        old = _git_blob(commit, name)
        path = m9base.REPO / name
        current = path.read_bytes() if path.exists() else None
        if name == "m9/LEDGER.md":
            if old is None or current is None or not current.startswith(old):
                _refuse("m9/LEDGER.md changed before its registered end; only appends are allowed")
        elif name == "m9/registry.json":
            try:
                before = json.loads(old) if old is not None else None
                now = json.loads(current) if current is not None else None
            except (TypeError, json.JSONDecodeError) as exc:
                _refuse(f"m9/registry.json cannot be compared with the registered commit: {exc}")
            if not isinstance(before, dict) or not isinstance(now, dict):
                _refuse("m9/registry.json was not an object at both registered and current states")
            before_rulings = before.pop("owner_rulings", {})
            now_rulings = now.pop("owner_rulings", {})
            if before != now:
                _refuse("m9/registry.json changed outside owner_rulings after the screen")
            if not isinstance(before_rulings, dict) or not isinstance(now_rulings, dict):
                _refuse("m9/registry.json owner_rulings must be an object")
            changed = [key for key, value in before_rulings.items()
                       if now_rulings.get(key) != value]
            if changed:
                _refuse(f"pre-existing owner ruling(s) changed or disappeared: {changed}")
        elif old != current:
            _refuse(f"protocol dependency {name} changed after the screen")


def _verify_token_registration(payload, expected_run_id):
    if not isinstance(payload, dict):
        _refuse(f"{expected_run_id}: artifact root must be an object")
    reg = payload.get("_registration")
    if not isinstance(reg, dict):
        _refuse(f"{expected_run_id}: registration is missing")
    if (reg.get("run_id") != expected_run_id or reg.get("diagnostic")
            or reg.get("eligible_for_decision") is not True):
        _refuse(
            f"{expected_run_id}: registration must be non-diagnostic, decision-eligible, "
            "and name the artifact's run id")
    token_path = guard9.TOKENS / f"{expected_run_id}.json"
    if not token_path.exists():
        _refuse(f"{expected_run_id}: consumed run token is missing")
    try:
        token = json.loads(token_path.read_text())
    except json.JSONDecodeError as exc:
        _refuse(f"{expected_run_id}: run token is not valid JSON: {exc}")
    if token.get("run_id") != expected_run_id or token.get("consumed") is not True:
        _refuse(f"{expected_run_id}: run token is not the consumed token for this artifact")
    for key in ("commit", "branch", "session_sha256", "scopes", "scoped", "opened_at", "extra"):
        if token.get(key) != reg.get(key):
            _refuse(f"{expected_run_id}: registration/token mismatch in {key}")
    actual_payload = guard9.payload_sha(payload)
    if token.get("payload_sha256") != actual_payload:
        _refuse(
            f"{expected_run_id}: token payload hash mismatch: token has "
            f"{token.get('payload_sha256')}, artifact has {actual_payload}")
    session_path = guard9._session_path(expected_run_id)
    if not session_path.exists():
        _refuse(f"{expected_run_id}: frozen session manifest is missing")
    try:
        session = json.loads(session_path.read_text())
    except json.JSONDecodeError as exc:
        _refuse(f"{expected_run_id}: session manifest is not valid JSON: {exc}")
    session_sha = guard9.fp_sha(session.get("fingerprint", {}))
    if session_sha != token.get("session_sha256"):
        _refuse(
            f"{expected_run_id}: token/session fingerprint mismatch: token has "
            f"{token.get('session_sha256')}, manifest has {session_sha}")
    for key in ("commit", "branch"):
        if session.get(key) != reg.get(key):
            _refuse(f"{expected_run_id}: registration/session mismatch in {key}")
    if session.get("stage") != reg.get("stage"):
        _refuse(f"{expected_run_id}: registration/session mismatch in stage")
    frozen = session.get("fingerprint", {})
    for scope in reg.get("scopes", ()):
        if reg.get("scoped", {}).get(scope) != frozen.get(scope):
            _refuse(f"{expected_run_id}: registration/session mismatch in {scope} scope")
    return reg


def _verify_screen_decision(blob, results_dir=None):
    """Verify the frozen decision and every artifact/token in its registered lineage.

    Unlike guard9.eligible(strict=True), this deliberately permits only the two kinds of protocol
    recording that must happen after a verdict. All non-protocol ambient scopes remain strict.
    """
    results_dir = RESULTS if results_dir is None else results_dir
    decision_reg = _verify_token_registration(blob, "m9-decisions")
    expected_scopes = set(guard9.deps_for("m9-decisions"))
    if set(decision_reg.get("scopes", ())) != expected_scopes:
        _refuse("m9-decisions: registration does not contain the registered dependency scopes")
    for scope in expected_scopes - {"protocol"}:
        current = _scope_sha(scope)
        recorded = decision_reg.get("scoped", {}).get(scope)
        if current != recorded:
            _refuse(
                f"m9-decisions: ambient {scope} scope changed: registered {recorded}, "
                f"current {current}")
    _verify_record_only_protocol_drift(decision_reg)

    mandatory = blob.get("mandatory_arms")
    present = blob.get("arms_present")
    hashes = blob.get("arm_artifact_sha256")
    if (blob.get("complete") is not True or blob.get("missing_arms") != []
            or not isinstance(mandatory, list) or not mandatory
            or not isinstance(present, list) or not isinstance(hashes, dict)):
        _refuse("m9-decisions: completion/arm metadata is missing or inconsistent")
    if not all(isinstance(arm, str) for arm in mandatory + present):
        _refuse("m9-decisions: every arm id must be a string")
    if len(mandatory) != len(set(mandatory)) or len(present) != len(set(present)):
        _refuse("m9-decisions: arm metadata contains duplicates")
    if not set(mandatory) <= set(present) or set(present) != set(hashes):
        _refuse("m9-decisions: mandatory, present, and hashed arm sets are inconsistent")
    known = {row["id"] for row in guard9.registry()["arms"]}
    unknown = sorted(set(present) - known)
    if unknown:
        _refuse(f"m9-decisions: unknown arm id(s) in lineage: {unknown}")

    for arm in present:
        path = results_dir / f"m9_screen_{arm}.json"
        if not path.exists():
            _refuse(f"{arm}: arm artifact is missing at {path}")
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_hash != hashes[arm]:
            _refuse(f"{arm}: arm file hash mismatch: decision has {hashes[arm]}, file has "
                    f"{actual_hash}")
        try:
            arm_blob = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            _refuse(f"{arm}: arm artifact is not valid JSON: {exc}")
        arm_reg = _verify_token_registration(arm_blob, arm)
        for key in ("commit", "branch", "session_sha256", "stage"):
            if arm_reg.get(key) != decision_reg.get(key):
                _refuse(f"{arm}: arm and decision registrations disagree on {key}")
        expected_arm_scopes = set(guard9.deps_for(arm))
        if set(arm_reg.get("scopes", ())) != expected_arm_scopes:
            _refuse(f"{arm}: registration does not contain its registered dependency scopes")
        for scope in expected_arm_scopes:
            if arm_reg.get("scoped", {}).get(scope) != decision_reg["scoped"].get(scope):
                _refuse(f"{arm}: arm and decision registrations disagree on {scope} scope")
    return True


def _validated_shares(value, where):
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be an object with exactly {sorted(SHARE_KEYS)}")
    keys = set(value)
    if keys != SHARE_KEYS:
        raise ValueError(
            f"{where} keys must be exactly {sorted(SHARE_KEYS)}; got {sorted(keys)}")
    for key, share in value.items():
        if isinstance(share, bool) or not isinstance(share, Real):
            raise TypeError(f"{where}.{key} must be a real number, got {share!r}")
        if not math.isfinite(float(share)) or not 0.0 <= share <= 1.0:
            raise ValueError(f"{where}.{key} must be finite and in [0, 1], got {share!r}")
    if not math.isclose(math.fsum(value.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"{where} must sum to 1.0 within 1e-9; got {math.fsum(value.values())}")
    return {key: value[key] for key in ("queries", "spans", "documents")}


def _validated_override(ruling, mix, decision):
    if not isinstance(ruling, dict):
        raise ValueError("owner_rulings.m9s6_mix_override must be an object")
    required_text = ("ruled_by", "date", "why", "caveat", "overrides_verdict",
                     "recorded_before")
    empty = [key for key in required_text
             if not isinstance(ruling.get(key), str) or not ruling[key].strip()]
    if empty:
        raise ValueError(f"m9s6_mix_override requires non-empty fields: {empty}")
    if ruling["overrides_verdict"] != mix:
        raise ValueError(
            f"m9s6_mix_override names verdict {ruling['overrides_verdict']!r}, not {mix!r}")

    artifact = ruling.get("decision_artifact")
    expected_sha = ruling.get("decision_sha256")
    if artifact != RULING_DECISION:
        raise ValueError(
            f"m9s6_mix_override.decision_artifact must be {RULING_DECISION!r}, got {artifact!r}")
    if not isinstance(expected_sha, str) or not expected_sha.strip():
        raise ValueError("m9s6_mix_override.decision_sha256 must be non-empty")
    path = m9base.REPO / artifact
    raw = path.read_bytes()
    actual_sha = hashlib.sha256(raw).hexdigest()
    if actual_sha != expected_sha:
        raise ValueError(
            f"m9s6_mix_override applies to decision sha256 {expected_sha}, but {artifact} "
            f"currently hashes {actual_sha}; a ruling for one screen cannot authorize another")
    bound = json.loads(raw)
    if not bound.get("complete") or bound.get("selected") != decision:
        raise ValueError(
            "m9s6_mix_override is not bound to this complete screen decision; refusing override")
    shares = _validated_shares(ruling.get("build_shares"),
                               "m9s6_mix_override.build_shares")
    ruling_sha = hashlib.sha256(json.dumps(
        ruling, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if ruling_sha != M9S6_RULING_SHA256:
        raise ValueError(
            f"m9s6_mix_override canonical sha256 is {ruling_sha}, not the build-pinned "
            f"{M9S6_RULING_SHA256}; ruling text or shares were altered")
    return shares


def build():
    r = guard9.registry()
    p = RESULTS / "m9_screen_decisions.json"
    if not p.exists():
        raise SystemExit(f"{p} does not exist -- there is no screen decision. A seven-day "
                         f"build does not run on invented defaults (Codex #7/#8). Run the "
                         f"screen chain through `decide` first.")
    blob = json.loads(p.read_text())
    try:
        _verify_screen_decision(blob)
    except DecisionIntegrityError as exc:
        raise SystemExit(f"m9_screen_decisions.json integrity failure: {exc}") from exc
    if not blob.get("complete"):
        raise SystemExit(f"the screen decision is provisional; missing {blob['missing_arms']}")
    dec = blob["selected"]

    missing = [k for k in ("student", "teacher", "prompt", "mix") if k not in dec]
    if missing:
        raise SystemExit(f"the screen decision lacks {missing}; refusing to fill defaults")
    student, teacher, prompt, mix = dec["student"], dec["teacher"], dec["prompt"], dec["mix"]
    if teacher != "stella-400M-v5":
        raise SystemExit(f"teacher {teacher!r}: a challenger win STOPS M9 (LEDGER §0), it does not "
                         f"start a seven-day build")
    # Registered mapping (M92_LOCK §4, written before m9s6 ran): "70/30" confirms document text
    # helps at matched dose -> the registered 5/5/90 build. "query-only" means documents failed
    # their only direct test, so a 90%-documents seven-day bet is unsupported by its own screen --
    # that is an owner decision, not a formula.
    ruling = r.get("owner_rulings", {}).get("m9s6_mix_override")
    if mix != "70/30":
        # The stop is real: only an explicit owner ruling bound to this exact screen decision
        # lifts it, and the shares then come from the ruling, not from this file's constant.
        if not ruling:
            raise SystemExit(f"the screen selected mix {mix!r}. The registered document-dominant "
                             f"build (5/5/90) rests on m9s6 confirming documents help; without it, "
                             f"STOP and get Dylan's ruling on the build shares (M92_LOCK §4).")
        shares = _validated_override(ruling, mix, dec)
        print(f"OWNER RULING ({ruling['date']}, {ruling['ruled_by']}): mix verdict {mix!r} "
              f"overridden; shares {shares}", flush=True)
    else:
        shares = _validated_shares(SHARES, "SHARES")

    total_tokens = int(HOURS * 3600 * TOK_PER_S)
    decay_steps = math.ceil(COOLDOWN_TOKENS / TOKENS_PER_STEP)
    stable_cap = total_tokens - COOLDOWN_TOKENS
    steps_total = total_tokens // TOKENS_PER_STEP
    eval_every = 20_000                                    # ~164M tokens, ~1.7 h
    cfg = {
        "run_id": "m9-build",
        "_what": "M9.3: the seven-day build. Stops on evidence, not on this horizon.",
        "student": student, "teacher": teacher, "prompt_policy": prompt,
        "student_query_prefix": (r["templates"]["query_policy_a_student"] if prompt == "a"
                                 else r["templates"]["query_policy_b_student"]),
        "seed": 0,
        "shares": shares,
        "tokens_per_step": TOKENS_PER_STEP,
        "lr_peak": 1e-4, "lr_final": 1e-5, "warmup_steps": 2000,
        "decay_steps": decay_steps,
        "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.01, "grad_clip": 1.0,
        "log_every": 500, "ckpt_every": 5000, "eval_every": eval_every,
        "stable_token_cap": stable_cap,
        # kill envelope, all in the units the dose is registered in
        "regression_thresh": 0.0056,        # the MDE: two evals this far below best -> stop
        "plateau_tokens": 1_000_000_000,    # judged over a billion tokens, never over steps
        "plateau_gain": 0.001,
        "throughput_floor_frac": 0.5,
        "throughput_window_s": 600,          # rolling, never the cumulative session mean
        "throughput_baseline_after_s": 900,  # frozen once after warmup, persisted across restarts
        # The anchor reached 0.50004 on 59.5M tokens; the first build evaluation lands at ~164M
        # with 15x the unique text, so anything below 0.45 means something is broken rather than
        # merely slow.
        "first_eval_floor": 0.45,          # ADVISORY: logged, never a hard stop
        "first_eval_regression": 0.02,     # HARD: below this run's own step-0 baseline
        "_arithmetic": {
            "measured_tokens_per_s": TOK_PER_S,
            "horizon_hours": HOURS,
            "total_tokens_if_never_stopped": total_tokens,
            "steps_if_never_stopped": steps_total,
            "evals_if_never_stopped": steps_total // eval_every,
            "cooldown_tokens": COOLDOWN_TOKENS,
            "cooldown_steps": decay_steps,
            "cooldown_provenance": "the anchor arm's entire dose -- the only annealing scale this "
                                   "project has measured",
        },
    }

    per_step, epochs = {}, {}
    for name in longrun.QUERY_SOURCES + longrun.SPAN_SOURCES + longrun.DOC_SOURCES:
        _f, offs, meta = longrun.load_corpus(name)
        if name in longrun.QUERY_SOURCES and meta.get("prefix") != cfg["student_query_prefix"]:
            raise SystemExit(
                f"corpus {name!r} is tokenized with prefix {meta.get('prefix')!r} but the screen "
                f"selected prompt policy {prompt!r} ({cfg['student_query_prefix']!r}). Re-run "
                f"`longrun.py prepare --prompt-policy {prompt}`, then `targets`, `manifest`, "
                f"`verify` -- documents are skipped automatically (Codex #7, blocker 1).")
        grp = longrun._grp(name)
        sibs = [n for n in (longrun.QUERY_SOURCES + longrun.SPAN_SOURCES + longrun.DOC_SOURCES)
                if longrun._grp(n) == grp]
        tot = sum(longrun.load_corpus(s)[2]["n_tokens"] for s in sibs)
        share = shares[grp] * meta["n_tokens"] / tot
        per_step[name] = max(1, int(round(TOKENS_PER_STEP * share / meta["mean_tokens"])))
        epochs[name] = round(total_tokens * share / meta["n_tokens"], 1)
    cfg["_arithmetic"]["examples_per_step"] = per_step
    cfg["_arithmetic"]["examples_per_step_total"] = sum(per_step.values())
    cfg["_arithmetic"]["epochs_at_horizon"] = epochs
    return cfg


def main():
    cfg = build()
    WORK.joinpath("m9long").mkdir(parents=True, exist_ok=True)
    longrun.CONFIG.write_text(json.dumps(cfg, indent=2))
    print(json.dumps(cfg["_arithmetic"], indent=1))
    print(f"\nwrote {longrun.CONFIG}")


if __name__ == "__main__":
    main()
