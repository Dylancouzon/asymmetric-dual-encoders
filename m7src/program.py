"""The post-gate experiment program, phase by phase. Every phase is selected on dev only.

Phases 1-3 find a base config; phase 4 runs the ablations the mandate makes mandatory whatever
they say; phase 5 settles whether FEVER-train stays in the mix (which decides whether BEIR
FEVER is an in-domain or a clean untouched-final row); phase 6 freezes the fusion.

Call one phase at a time from a driver script -- runs are cheap (a few minutes each) but the
GPU must not be shared, per the OOM incident in m7/LEDGER.md.
"""
import json
from dataclasses import replace

from _paths import REPO, WORK
from sweep import chain, chains, grid, one
from train import Cfg

BASE = Cfg(objective="C", init="teacher", preproc="noprefix", learned_weights=True,
           steps_b=4000, steps_a=8000, batch=512, n_neg=32768, temp=0.02, lr=3e-3,
           hard_neg_k=0, reg_init=1e-3)
# Kill criterion, set BEFORE running phase 2 so it cannot be moved afterwards: if no arm of the
# phase-2 screen beats p1-objB's 0.4548 dev proxy, contrastive training is closed as an avenue and
# the program pivots to vocabulary-coverage distillation (phase35) plus fusion. Distillation
# already passed the gate; grinding a broken objective past a clean negative is how budgets die.
CONTRASTIVE_KILL_BAR = 0.4548
# RESTATED 2026-08-26, before the screen ran, because the arms it was to be judged on were
# themselves misconfigured. The bar is unchanged; what changed is which arms may trip it.
# Phase 1 collapsed at lr=3e-3 with no warmup -- 10-300x above every published frozen-tower recipe
# (NV-Retriever 1e-5, DAFT 1e-5, BGE/E5/GTE 1e-5..5e-5) -- and arXiv 2110.09348 gives an analytic
# mechanism for exactly that symptom. The two competing suspects are now dead by measurement
# (results/m7_diag_scores.json: fn_margin=0.02 removes only 4.3% of the top-100 hardest negatives;
# random negatives are not separable). The screen as originally written had its "decisive" arm at
# lr=1e-3 and its lr arm at 3e-4, i.e. still above the published range -- so it could have tripped
# the kill bar on a configuration we already believe is broken. CLAUDE.md's standing directive is
# explicit that a kill criterion exists to stop grinding a DIAGNOSED dead end, never to license
# abandoning an undiagnosed one. So:
#   the kill criterion may only be invoked once at least one arm has run at a published lr
#   (<= 1e-4) WITH warmup and mined hard negatives, and that arm has failed the bar.
KILL_REQUIRES = {"lr_at_most": 1e-4, "warmup": True, "hard_negatives": True}


def may_invoke_contrastive_kill(runs, scores=None, bar=CONTRASTIVE_KILL_BAR):
    """Enforce KILL_REQUIRES against COMMITTED RESULTS, not against a comment.

    `runs` maps run_id -> Cfg-like dict; `scores` maps run_id -> dev macro (None = unknown/failed).

    Codex's review found the first version enforced only that a qualifying CONFIGURATION existed --
    not what it scored, and not whether some other arm had already cleared the bar. Both are now
    conditions: a kill requires a qualifying arm AND that every arm, qualifying or not, failed the
    bar. An avenue where any arm beats the bar is not dead, whatever the qualifying arm did.
    """
    scores = scores or {}
    qualifying = [r for r, c in runs.items()
                  if c.get("lr", 1) <= KILL_REQUIRES["lr_at_most"]
                  and c.get("warmup_steps", 0) > 0 and c.get("hard_neg_k", 0) > 0]
    if not qualifying:
        return False, ("no arm has run at lr <= 1e-4 WITH warmup and mined hard negatives, so the "
                       "avenue is not yet diagnosed and the kill criterion may not be invoked")
    missing = [r for r in qualifying if scores.get(r) is None]
    if missing:
        return False, f"qualifying arms have no committed score: {sorted(missing)}"
    # A zero-step arm IS the bar (it re-scores the checkpoint the bar was set from), so it must not
    # count as an arm that "beat" it -- it exceeded 0.4548 only by the rounding in the bar itself.
    # Tolerance for the same reason: the bar is a rounded number.
    def trained(r):
        c = runs.get(r, {})
        return (c.get("steps_a", 0) + c.get("steps_b", 0)) > 0
    passed = {r: v for r, v in scores.items()
              if v is not None and v > bar + 1e-4 and trained(r)}
    if passed:
        return False, (f"the kill criterion may not be invoked: {len(passed)} arm(s) beat the "
                       f"{bar} bar -- " + ", ".join(f"{r}={v:.4f}" for r, v in sorted(passed.items())))
    return True, (f"qualifying arms {sorted(qualifying)} all failed the {bar} bar, and no other arm "
                  f"passed it either")


def contrastive_verdict(screen_name="phase2_screen"):
    """Read the screen's committed results and the runs' own cfgs, and record the verdict.

    Exists so the kill decision is a function of files on disk rather than of a session's memory of
    what it saw. Writes results/m7_contrastive_verdict.json.
    """
    scores = json.loads((REPO / "results" / f"m7_program_{screen_name}.json").read_text())
    runs = {}
    for rid in scores:
        f = WORK / "runs" / f"{rid}.json"
        runs[rid] = json.loads(f.read_text())["cfg"] if f.exists() else {}
    allowed, reason = may_invoke_contrastive_kill(runs, scores)
    best = max((r for r in scores if scores[r] is not None), key=lambda r: scores[r], default=None)
    out = {"_note": "Whether the pre-registered contrastive kill criterion may fire, computed from "
                    "committed screen results and each run's own cfg. A kill needs a qualifying arm "
                    "(lr <= 1e-4, warmup, mined hard negatives) AND every arm failing the bar.",
           "bar": CONTRASTIVE_KILL_BAR, "kill_requires": KILL_REQUIRES,
           "scores": scores, "kill_allowed": allowed, "reason": reason,
           "best_arm": best, "best_score": scores.get(best)}
    (REPO / "results" / "m7_contrastive_verdict.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("kill_allowed", "reason", "best_arm", "best_score")},
                     indent=1))
    return out

ALL_SOURCES = ("hotpotqa-train", "fever-train", "squad-train", "esci-us", "mrtydi-en")
NO_FEVER = tuple(s for s in ALL_SOURCES if s != "fever-train")


def phase1_objective():
    """A vs B vs C. B alone is also the Stage-0 distilled table the gate's G1 is judged on."""
    return grid("p1", BASE, {
        "objB": {"objective": "B", "steps_b": 8000, "steps_a": 0},
        "objA": {"objective": "A", "steps_b": 0, "steps_a": 12000},
        "objC": {"objective": "C"},
    })


def phase2_screen(base):
    """The CHEAP DECISIVE SCREEN: does the CONTRASTIVE PHASE degrade a good table, and does the
    learning rate explain it?

    REDESIGNED 2026-08-26, mid-screen, after seeing the first arm's B-phase curve and BEFORE any
    arm's A-phase result -- so this is not a change made to chase an outcome. The original design
    ran objective C (B then A) per arm at a matched step budget across a 60x lr range. That does not
    isolate the learning rate, because the B phase is also run at that lr: at 5e-5 B reaches 0.2731
    after 4,000 steps and is still climbing steeply, where 3e-3 reaches 0.4449 at the same count.
    Every arm would therefore enter its contrastive phase from a different table, and an A-phase
    delta would be confounded with how far B got. Codex's review of the plan reached the same
    conclusion independently.

    So: every arm is objective A ONLY, initialised from the SAME p1-objB checkpoint (dev 0.4548,
    rows and trained token weights both restored), and the arms vary the contrastive learning rate.
    That is the quantity the kill criterion is about, and phase 1 already established what happens
    at lr 3e-3 from this exact starting table -- p1-objC degraded it monotonically to 0.3721.

    Cheaper as well as cleaner: no arm re-runs the B phase, so each is 2,000 steps rather than
    6,000, and the mined negatives are already cached.
    """
    CKPT = "run:p1-objB"

    def arm(lr, **kw):
        return {"objective": "A", "init": CKPT, "steps_b": 0, "steps_a": 2000, "eval_every": 250,
                "hard_neg_k": 16, "hard_neg_source": "teacher", "fn_margin": 0.05,
                "lr": lr, "lr_weights": lr * 10, "warmup_steps": 200,
                "lr_schedule": "warmup_linear", **kw}

    return grid("p2s", base, {
        # step 0 with no training: pins the starting point in THIS harness, so every arm's delta is
        # measured against a number produced by the same code path rather than against a ledger row.
        "start":         arm(5e-5, steps_a=0, eval_every=1),
        # the decisive arms: published frozen-tower learning rates, with warmup and mined negatives
        "sane-1e5":      arm(1e-5),          # NV-Retriever / DAFT's exact value
        "sane-5e5":      arm(5e-5),
        "sane-1e4":      arm(1e-4),          # top of the published range; satisfies KILL_REQUIRES
        # controls, one variable each
        "old-lr-3e3":    arm(3e-3),          # phase 1's lr, now WITH warmup and hard negatives
        "sane-randneg":  arm(5e-5, hard_neg_k=0),   # the mandate's premise, at a published lr
    })


def phase2_screen_ext(base):
    """Where does the learning rate turn over?

    The screen left the story open at both ends: 1e-4 was the BEST arm and the top of the published
    range, so the trend was still rising where the evidence stopped, while 3e-3 was flat. Somewhere
    between them the objective stops helping and starts destroying, and "we simply never pushed the
    lr far enough" is otherwise an unanswered objection to reading 1e-4 as the answer.

    Same fixed checkpoint, same everything else, and hard_neg_k=0 -- the screen resolved that mined
    negatives HURT at matched lr (+0.0034 for random-only, CI [0.0019, 0.0049]), so the extension
    varies one thing.
    """
    def arm(lr):
        return {"objective": "A", "init": "run:p1-objB", "steps_b": 0, "steps_a": 2000,
                "eval_every": 250, "hard_neg_k": 0, "lr": lr, "lr_weights": lr * 10,
                "warmup_steps": 200, "lr_schedule": "warmup_linear"}

    return grid("p2x", base, {"rn-3e4": arm(3e-4), "rn-1e3": arm(1e-3), "rn-3e3": arm(3e-3)})


def stella_confirm(base):
    """ONE confirmation of the phase-2 result under the stella teacher, not a re-sweep: the
    5e-5..3e-4 band and hard_neg_k=0 were established on bge-base from a fixed B checkpoint;
    three arms + a zero-step pin check that the finding transfers. Pre-registered in LEDGER
    (2026-08-26): eval every 500 steps, best-eval selection uniformly across arms, and the winner
    is re-run at its best step so the selected step count is part of the frozen config.
    """
    b = one("s1-objB", base, objective="B", steps_b=8000, steps_a=0)
    if b is None:
        return None

    def arm(lr):
        return {"objective": "A", "init": "run:s1-objB", "steps_b": 0, "steps_a": 2000,
                "eval_every": 500, "hard_neg_k": 0, "lr": lr, "lr_weights": lr * 10,
                "warmup_steps": 200, "lr_schedule": "warmup_linear"}

    return grid("s2", base, {
        "start":  {**arm(5e-5), "steps_a": 0, "eval_every": 1},   # pins the s1-objB point in-harness
        "rn-5e5": arm(5e-5),
        "rn-1e4": arm(1e-4),
        "rn-3e4": arm(3e-4),
    })


def phase2_negatives(base):
    """The mandated negatives ablation: BM25-mined vs teacher-mined vs mixed, against the
    random-bank baseline. Objective A collapsed with random-only negatives (p1-objA declined
    monotonically to 0.3248 -- RESULTS.md and the run JSON agree on 0.3248; this docstring said
    0.3366 until 2026-08-28), so this is the load-bearing phase, not a tuning sweep."""
    return grid("p2", base, {
        "bank": {"hard_neg_k": 0},
        "teacher16": {"hard_neg_k": 16, "hard_neg_source": "teacher"},
        "teacher32": {"hard_neg_k": 32, "hard_neg_source": "teacher"},
        "bm2516": {"hard_neg_k": 16, "hard_neg_source": "bm25"},
        "mixed32": {"hard_neg_k": 32, "hard_neg_source": "mixed"},
        "teacher16-nofn": {"hard_neg_k": 16, "hard_neg_source": "teacher", "fn_margin": 0.0},
        "teacher16-noreg": {"hard_neg_k": 16, "hard_neg_source": "teacher", "reg_init": 0.0},
    })


def phase3_hparams(base):
    return grid("p3", base, {
        "lr1e-3": {"lr": 1e-3}, "lr3e-3": {"lr": 3e-3}, "lr1e-2": {"lr": 1e-2},
        "t005": {"temp": 0.005}, "t01": {"temp": 0.01}, "t05": {"temp": 0.05},
        "neg8k": {"n_neg": 8192}, "neg128k": {"n_neg": 131072},
        "long": {"steps_b": 8000, "steps_a": 24000},
    })


# ---- the mandatory ablations, rebuilt after Codex review #3 ----------------------------------
#
# The recipe every chain reproduces, taken from the winning runs' own cfgs (work/runs/*.json):
# objective B at 3e-3 constant for 16,000 steps with the pseudo-query mix, then a SEPARATE
# objective-A run from that exact checkpoint at 1e-3 warmup-linear. It has to be two runs: one
# objective-C Cfg carries one learning rate, one schedule, and one Adam state whose cumulative
# update counts also rescale the 1/(1+updates) init penalty, so a C-shaped "full chain" would
# differ from the candidate in four ways unrelated to the arm's variable (review #3 BLOCKER 4).
ABLATION_B = {"objective": "B", "init": "teacher", "preproc": "noprefix", "learned_weights": True,
              "idf_init_weights": True, "lr": 3e-3, "lr_weights": 1e-2, "warmup_steps": 0,
              "lr_schedule": "constant", "steps_b": 16000, "steps_a": 0, "batch": 512,
              "n_neg": 32768, "temp": 0.02, "hard_neg_k": 0, "fn_margin": 0.02, "kl_weight": 1.0,
              "kl_k": 32, "cos_weight": 1.0, "reg_init": 1e-3, "bank_size": 2_000_000,
              "eval_every": 2000, "b_query_sources_all": True, "b_pseudo_queries": 2_000_000,
              "b_pseudo_frac": 0.5}
ABLATION_A = {"objective": "A", "preproc": "noprefix", "steps_b": 0, "steps_a": 2500, "lr": 1e-3,
              "lr_weights": 1e-2, "warmup_steps": 200, "lr_schedule": "warmup_linear",
              "hard_neg_k": 0, "eval_every": 500, "b_pseudo_queries": 0, "reg_init": 1e-3}
# steps_a per surviving candidate: p35w-2m-s2500 -> 2500, p35a-2m-1e3 -> 2000. Which one survives
# is decided by the dependence recompute (results/m7_dev_audit_full.json), never by choosing here.
SURVIVOR_STEPS_A = {"p35w-2m-s2500": 2500, "p35a-2m-1e3": 2000, "p35w-500k-s1500": 1500,
                    "s2w-1e3-s1000": 1000}


def ablation_recipe():
    """The (B, A) overrides for the ablation chains, read from the SURVIVING candidate's own
    committed config and that of the checkpoint it was initialized from.

    Derived from disk rather than typed here for two reasons. Which artifact survives the
    dependence recompute is not known when this code is written, and each candidate carries a
    different B phase (8,000 steps for the pre-lever winner, 16,000 with the pseudo mix for the 2M
    arms). And an ablation whose "unchanged" baseline drifts from the candidate by a
    transcription slip measures the slip, not the ablated variable. ABLATION_B/ABLATION_A below
    are the expected values and are asserted against what is read.

    **`M7_RECIPE_FROM` overrides which artifact is read.** The dev-audit survivor was the right
    source while the lever chain ended at the candidate; it no longer does -- the negatives arm
    and the simplification each moved the candidate past it. A lever that has to isolate one knob
    must build on whatever recipe is current, and hardcoding the survivor would silently measure
    that knob PLUS every change made since. Set it to the current candidate's run id.
    """
    import os
    over = os.environ.get("M7_RECIPE_FROM")
    if over:
        surv = over
        print(f"[recipe] M7_RECIPE_FROM={surv}: deriving the base recipe from that artifact, not "
              f"from the dev-audit survivor", flush=True)
    else:
        p = REPO / "results" / "m7_dev_audit_full.json"
        if not p.exists():
            raise SystemExit("run dev_audit.py first, or set M7_RECIPE_FROM: an arm must reproduce "
                             "the recipe of the artifact it is varying one knob against")
        surv = json.loads(p.read_text())["surviving_candidate"]
    a_cfg = json.loads((WORK / "runs" / f"{surv}.json").read_text())["cfg"]
    init = a_cfg.get("init", "")
    if not init.startswith("run:"):
        raise SystemExit(f"{surv} was not initialized from a checkpoint (init={init!r}); it is not "
                         "a two-phase chain and the ablation design does not apply to it")
    bid = init.split(":", 1)[1]
    b_cfg = json.loads((WORK / "runs" / f"{bid}.json").read_text())["cfg"]
    b = {k: v for k, v in b_cfg.items() if k != "run_id"}
    a = {k: v for k, v in a_cfg.items() if k not in ("run_id", "init")}
    drift = {k: (b.get(k), v) for k, v in ABLATION_B.items() if b.get(k) != v}
    if drift:
        print(f"[p4] NOTE: surviving B phase ({bid}) differs from the documented ABLATION_B on "
              f"{drift} -- the chains follow the ARTIFACT, which is correct, but update the "
              f"constants so the file stops describing a different recipe.", flush=True)
    return surv, b, a


# Arm definitions as DATA, so a driver can run exactly one arm in its own process.
#
# Why that matters, learned at 00:05 on 2026-08-28: running every arm inside ONE python process
# accumulated the module-level caches this repo memoizes on purpose (`mix.load_source`,
# `heldout._DOC_IDS`, `dev_eval._HELD_CACHE`, the encode memmaps) on top of each arm's own
# ~4 GB of pseudo-query targets and ~4 GB negative bank. The first two chains fit; the third
# reached 24.7 GB RSS on a 25 GB box and sat burning one core with zero disk I/O -- the thrash
# signature from the LEDGER's OOM incident. A fresh process per arm gives every arm the same
# starting memory, which is also what makes the arms comparable.
P4_ARMS = {
    "base":      {},
    "input-emb": {"b": {"init": "input_emb"}},
    "random":    {"b": {"init": "random"}},
    # RUNTIME prefix only: the teacher rows stay the no-prefix ones in both legs, so the arm
    # varies query tokenization and nothing else (review #3 BLOCKER 3). Prefix-CONDITIONED rows
    # are a separate, exploratory arm below.
    "prefix":    {"b": {"preproc": "prefix", "init_preproc": "noprefix"},
                  "a": {"preproc": "prefix", "init_preproc": "noprefix"}},
    "flat":      {"b": {"learned_weights": False}, "a": {"learned_weights": False}},
    "uniform-w": {"b": {"idf_init_weights": False}, "a": {"idf_init_weights": False}},
    "reg0":      {"b": {"reg_init": 0.0}, "a": {"reg_init": 0.0}},
}
P4X_ARMS = {
    "nopseudo":   {"b": {"b_pseudo_queries": 0}},
    "pseudo500k": {"b": {"b_pseudo_queries": 500_000}},
}
P4E_ARMS = {
    "prefix-init": {"b": {"preproc": "prefix", "init_preproc": "prefix"},
                    "a": {"preproc": "prefix", "init_preproc": "prefix"}},
}
# A-only arms from the CANDIDATE's own B checkpoint: "@candidate_b" is resolved at run time.
P4N_ARMS = {
    "bank":      {"a": {"hard_neg_k": 0}, "init": "@candidate_b"},
    "teacher16": {"a": {"hard_neg_k": 16, "hard_neg_source": "teacher"}, "init": "@candidate_b"},
    "bm2516":    {"a": {"hard_neg_k": 16, "hard_neg_source": "bm25"}, "init": "@candidate_b"},
    "mixed32":   {"a": {"hard_neg_k": 32, "hard_neg_source": "mixed"}, "init": "@candidate_b"},
    # STEP-RULE CORRECTIONS. The four arms above were promoted and full-suite-compared at the
    # inherited steps_a=2500, but the pre-registered rule is that an arm's step count is its best
    # proxy eval, "implemented by re-running to that step". Their proxy curves peak at 1500, 1500
    # and 1000; `bank`'s peaks at 2500, so the control needs no correction and has no entry here.
    # `warmup_linear` decays over steps_a, so these are genuinely different runs and not prefixes
    # of the 2500-step ones. Pre-registered in LEDGER.md, "The step-selection rule was NOT applied
    # to the negatives arms" -- including the part that says the proxy picks the step and the
    # full-suite number does not get a vote.
    "teacher16-s1500": {"a": {"hard_neg_k": 16, "hard_neg_source": "teacher", "steps_a": 1500},
                        "init": "@candidate_b"},
    "bm2516-s1500":    {"a": {"hard_neg_k": 16, "hard_neg_source": "bm25", "steps_a": 1500},
                        "init": "@candidate_b"},
    "mixed32-s1000":   {"a": {"hard_neg_k": 32, "hard_neg_source": "mixed", "steps_a": 1000},
                        "init": "@candidate_b"},
}
# The recipe SIMPLIFICATION arm: every ablation-inert component of the shipping recipe removed at
# once. One arm, no ladder of fallbacks -- backing off component by component until something
# passes would be adaptive dev search. Tested for non-inferiority at margin 0.0040, not adopted on
# a point estimate. Full rationale, including why `input_emb` rather than `random`, is in
# LEDGER.md "Recipe simplification". `steps_a` is left at the base recipe's value here and the
# step-selection rule is applied to the arm's own proxy curve afterwards, like any other arm.
#
# Three A-leg variants share ONE B leg, because `hard_neg_k` is 0 in the B recipe and the
# negatives only enter the A phase. EXACTLY ONE of them is run: whichever matches the negatives
# source the step-rule-corrected comparison selects, so the simplification is measured against a
# like-for-like unsimplified arm. This is NOT a search over negatives -- that decision is made
# independently, on the unsimplified arms, under its own pre-registered bar and tie-break. Listing
# all three here before that decision is known is what keeps it from becoming one.
P5S_ARMS = {
    "simple": {"b": {"init": "input_emb", "b_pseudo_queries": 500_000,
                     "idf_init_weights": False, "reg_init": 0.0},
               "a": {"idf_init_weights": False, "reg_init": 0.0,
                     "hard_neg_k": 16, "hard_neg_source": "teacher"}},
}
# THE ARM THAT FACES THE BAR, after the negatives avenue closed on 2026-08-28. The baseline is
# `p35w-2m-s2500`, which trains with no mined negatives, so the simplification of it must too --
# `p5s-simple` (k=16) is now a simplification of a recipe that is not the candidate, and is kept
# as a labelled off-baseline arm rather than deleted. Same B leg, so this costs one A phase.
P5S_ARMS["simple-nohn"] = {"init": "run:p5s-simple-b",
                           "a": {"idf_init_weights": False, "reg_init": 0.0, "hard_neg_k": 0}}
for _name, _k, _src in (("bm25", 16, "bm25"), ("mixed", 32, "mixed")):
    # A-only: they name `p5s-simple-b` as their init rather than carrying a B leg of their own,
    # so switching the negatives source costs ~5 minutes instead of a second 16,000-step B phase.
    P5S_ARMS[f"simple-{_name}"] = {
        "init": "run:p5s-simple-b",
        "a": {**P5S_ARMS["simple"]["a"], "hard_neg_k": _k, "hard_neg_source": _src},
    }
# Capacity lever #7: ONE knob, `b_pseudo_kind`, against whatever recipe is current. Deliberately
# not a copy of the simplification arm's overrides: run it with `M7_RECIPE_FROM=<candidate>` and
# the base comes from that artifact's own config, so the arm is "the candidate plus long spans"
# whichever way the simplification test and the negatives tie-break went. Copying the overrides
# here would silently measure the span distribution PLUS every difference between this file's
# snapshot and the artifact.
P7_ARMS = {
    "longspan": {"b": {"b_pseudo_kind": "mixed"}},
}
ARMS = {"p4": P4_ARMS, "p4x": P4X_ARMS, "p4e": P4E_ARMS, "p4n": P4N_ARMS,
        "p5s": P5S_ARMS, "p7": P7_ARMS}


def phase4_mandatory(base):
    """The ablations the mandate requires reported whatever they say -- seven chains, one variable
    each, every one a B run plus a fresh A run.

    `base` is the replay: it is both the nondeterminism estimate and the regularization-ON control
    at 1e-3, which is why there is no separate reg arm on that side (review #3 MINOR: the old grid
    ran the identical baseline three times under three names, and its "reg on/off" control was
    0 vs 1e-2 while the winner uses 1e-3).
    """
    surv, b, a = ablation_recipe()
    print(f"[p4] chains reproduce {surv}: B {b['steps_b']} steps -> A {a['steps_a']} steps",
          flush=True)
    return chains("p4", base, {
        "base":      {},
        "input-emb": {"b": {"init": "input_emb"}},
        "random":    {"b": {"init": "random"}},
        # RUNTIME prefix only: the teacher rows stay the no-prefix ones in both legs, so the arm
        # varies query tokenization and nothing else (review #3 BLOCKER 3). Prefix-CONDITIONED
        # rows are a separate, exploratory arm below.
        "prefix":    {"b": {"preproc": "prefix", "init_preproc": "noprefix"},
                      "a": {"preproc": "prefix", "init_preproc": "noprefix"}},
        "flat":      {"b": {"learned_weights": False}, "a": {"learned_weights": False}},
        "uniform-w": {"b": {"idf_init_weights": False}, "a": {"idf_init_weights": False}},
        "reg0":      {"b": {"reg_init": 0.0}, "a": {"reg_init": 0.0}},
    }, b, a)


def phase4_attribution(base):
    """Can the +0.0126 be attributed to pseudo-query COVERAGE at all?

    It cannot, from the search record alone: the sequence changed the pseudo pool size AND the B
    step count AND the A step count together (review #3 MAJOR 2). These two matched controls hold
    everything else at the surviving recipe and move only the pseudo-query pool, so `nopseudo` is
    the attribution control and `pseudo500k` is the dose control.
    """
    surv, b, a = ablation_recipe()
    return chains("p4x", base, {
        "nopseudo":   {"b": {"b_pseudo_queries": 0}},
        "pseudo500k": {"b": {"b_pseudo_queries": 500_000}},
    }, b, a)


def phase4_negatives(base):
    """The mandated negatives ablation -- BM25-mined vs teacher-mined vs mixed vs the random bank
    -- which `phase2_negatives` was written for and which NEVER RAN.

    `hard_neg_k=0` has been hard-coded into every arm since the phase-2 screen on the strength of
    a single bge-era pair at lr 5e-5 (+0.0034 for random-only). That is one point, under a
    different teacher, at a learning rate 20x below the one that ships, with no mechanism -- and
    it was never written down as a closed avenue, so it never faced the bar every other closure
    here had to clear. These arms vary ONLY the negatives, from the surviving candidate's own B
    checkpoint at its own A recipe, so `bank` is the candidate itself and is the control.
    """
    surv, b, a = ablation_recipe()
    bid = json.loads((WORK / "runs" / f"{surv}.json").read_text())["cfg"]["init"]
    return grid("p4n", base, {
        "bank":      {**a, "init": bid, "hard_neg_k": 0},
        "teacher16": {**a, "init": bid, "hard_neg_k": 16, "hard_neg_source": "teacher"},
        "bm2516":    {**a, "init": bid, "hard_neg_k": 16, "hard_neg_source": "bm25"},
        "mixed32":   {**a, "init": bid, "hard_neg_k": 32, "hard_neg_source": "mixed"},
    })


def phase4_exploratory(base):
    """Prefix-CONDITIONED teacher rows, i.e. every vocab token embedded inside the query prefix.
    The mandate calls this exploratory, so it is labelled and separate from the mandatory prefix
    arm, which shares the no-prefix rows."""
    surv, b, a = ablation_recipe()
    return chains("p4e", base, {
        "prefix-init": {"b": {"preproc": "prefix", "init_preproc": "prefix"},
                        "a": {"preproc": "prefix", "init_preproc": "prefix"}},
    }, b, a)


def phase35_coverage(base):
    """Vocabulary-coverage distillation. A VOCABULARY mitigation for the pre-registered domain
    gap, not a domain one -- it supplies no in-domain documents and no relevance structure.
    Selected on dev like everything else; dev is Wikipedia + StackExchange, so it can only
    speak to whether broader token coverage helps at all, not to the six."""
    return grid("p35", base, {
        "none": {"b_pseudo_queries": 0},
        "500k": {"b_pseudo_queries": 500_000, "steps_b": 8000},
        "2m": {"b_pseudo_queries": 2_000_000, "steps_b": 16000},
    })


def phase5_fever(base):
    """Does FEVER-train earn its place? If not, dropping it buys a clean untouched-final row."""
    return grid("p5", base, {"with-fever": {"sources": ALL_SOURCES},
                             "no-fever": {"sources": NO_FEVER}})


def save(name, res):
    p = REPO / "results" / f"m7_program_{name}.json"
    p.write_text(json.dumps(res, indent=1))
    print(f"wrote {p}")
