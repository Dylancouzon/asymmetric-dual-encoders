"""Generate `work/m9long/config.json` — the M9.3 build's every numeric knob, in one place.

Kept out of `m9src/longrun.py` deliberately: the trainer lives in the guarded `build` scope, so
iterating on the generator here does not churn the hash that a running build is pinned to. The
config it writes IS in that scope, which is the correct place for the binding.

Everything that came from a screen decision is read from `results/m9_screen_decisions.json`; every
dose figure is derived from measured quantities and shown with its arithmetic, so a reviewer can
check the numbers rather than take them.
"""
import json
import math

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


def build(decisions=None):
    r = guard9.registry()
    dec = decisions or {}
    p = RESULTS / "m9_screen_decisions.json"
    if not dec:
        if not p.exists():
            raise SystemExit(f"{p} does not exist -- there is no screen decision. A seven-day "
                             f"build does not run on invented defaults (Codex #7/#8). Run the "
                             f"screen chain through `decide` first.")
        blob = json.loads(p.read_text())
        if not guard9.eligible(blob):
            raise SystemExit("m9_screen_decisions.json is not decision-eligible")
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
        # The stop is real: only an explicit owner ruling that NAMES this verdict lifts it, and
        # the shares then come from the ruling itself, not from this file's constant.
        if not (ruling and ruling.get("overrides_verdict") == mix):
            raise SystemExit(f"the screen selected mix {mix!r}. The registered document-dominant "
                             f"build (5/5/90) rests on m9s6 confirming documents help; without it, "
                             f"STOP and get Dylan's ruling on the build shares (M92_LOCK §4).")
        print(f"OWNER RULING ({ruling['date']}, {ruling['ruled_by']}): mix verdict {mix!r} "
              f"overridden; shares {ruling['build_shares']}", flush=True)

    shares = dict(ruling["build_shares"]) if (ruling and mix != "70/30") else dict(SHARES)
    assert abs(sum(shares.values()) - 1.0) < 1e-9, shares
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
