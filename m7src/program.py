"""The post-gate experiment program, phase by phase. Every phase is selected on dev only.

Phases 1-3 find a base config; phase 4 runs the ablations the mandate makes mandatory whatever
they say; phase 5 settles whether FEVER-train stays in the mix (which decides whether BEIR
FEVER is an in-domain or a clean untouched-final row); phase 6 freezes the fusion.

Call one phase at a time from a driver script -- runs are cheap (a few minutes each) but the
GPU must not be shared, per the OOM incident in m7/LEDGER.md.
"""
import json
from dataclasses import replace

from _paths import REPO
from sweep import grid, one
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


def may_invoke_contrastive_kill(runs):
    """Enforce KILL_REQUIRES rather than trusting a comment. `runs` maps run_id -> Cfg-like dict.

    Returns (allowed, reason). A future session must call this before recording a kill; the review
    that added it found the criterion was documented and enforced nowhere.
    """
    ok = [r for r, c in runs.items()
          if c.get("lr", 1) <= KILL_REQUIRES["lr_at_most"]
          and c.get("warmup_steps", 0) > 0 and c.get("hard_neg_k", 0) > 0]
    if not ok:
        return False, ("no arm has run at lr <= 1e-4 WITH warmup and mined hard negatives, so the "
                       "avenue is not yet diagnosed and the kill criterion may not be invoked")
    return True, f"qualifying arms: {sorted(ok)}"

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
    """The CHEAP DECISIVE SCREEN, to run before any wide negatives sweep.

    Two of the three original suspects are now dead by measurement, not ablation
    (results/m7_diag_scores.json): fn_margin=0.02 removes only 4.3% of the top-100 hardest
    negatives, and random negatives are not trivially separable. Temperature is kept LOW on
    purpose -- 0.01-0.02 is the published norm (BGE uses 0.01) and Wang & Liu (arXiv 2012.09740)
    argue low temperature makes the loss more hardness-aware, so re-tuning it would spend budget
    on the one knob the literature says to leave alone.

    That leaves the learning rate, and the screen is now built around it: three arms across the
    published range with warmup, plus one-variable controls that make a pass or a fail
    attributable (old lr WITH warmup; correct lr WITHOUT hard negatives; phase-1 verbatim).
    Every arm starts from the B checkpoint, not the teacher init. Evals every 500 steps, and each
    one now logs the collapse diagnostics, so a degenerating representation is observed rather
    than inferred from the dev curve.
    """
    # Warmup + a published lr, and lr_weights kept at 10x the row lr rather than a fixed 1e-2
    # (at lr=5e-5 the old default would have been 200x the row lr).
    def sane(lr, **kw):
        return {"hard_neg_k": 16, "hard_neg_source": "teacher", "fn_margin": 0.05,
                "lr": lr, "lr_weights": lr * 10, "warmup_steps": 500,
                "lr_schedule": "warmup_linear", **kw}

    out = {}
    for tag, over in {
        # the decisive arm: a published lr, warmup, teacher-mined negatives, NV-Retriever's tuned
        # 0.05 margin. If this still degrades a 0.4449 table, the avenue is diagnosed and dead.
        "sane-5e5":      sane(5e-5),
        "sane-1e5":      sane(1e-5),          # NV-Retriever / DAFT's exact value
        "sane-1e4":      sane(1e-4),          # top of the published range
        # one-variable controls, so a pass or fail is attributable
        "warmup-only":   sane(3e-3),          # old lr WITH warmup: does warmup alone rescue it?
        "sane-randneg":  sane(5e-5, hard_neg_k=0),   # the mandate's premise, at the correct lr
        "baseline":      {"hard_neg_k": 0},    # phase-1 config verbatim, for the reference curve
    }.items():
        out.update(grid("p2s", base, {tag: {"objective": "C", "steps_b": 4000, "steps_a": 2000,
                                            "eval_every": 500, **over}}))
    return out


def phase2_negatives(base):
    """The mandated negatives ablation: BM25-mined vs teacher-mined vs mixed, against the
    random-bank baseline. Objective A collapsed with random-only negatives (p1-objA declined
    monotonically to 0.3366), so this is the load-bearing phase, not a tuning sweep."""
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


def phase4_mandatory(base):
    """The ablations the mandate requires reported whatever they say."""
    out = {}
    out.update(grid("p4-init", base, {"teacher": {"init": "teacher"},
                                      "input_emb": {"init": "input_emb"},
                                      "random": {"init": "random"}}))
    out.update(grid("p4-prefix", base, {"noprefix": {"preproc": "noprefix"},
                                        "prefix": {"preproc": "prefix"}}))
    out.update(grid("p4-weights", base, {"learned": {"learned_weights": True},
                                         "flat": {"learned_weights": False},
                                         "learned-noidf": {"idf_init_weights": False}}))
    out.update(grid("p4-reg", base, {"reg0": {"reg_init": 0.0}, "reg1e-2": {"reg_init": 1e-2}}))
    return out


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
