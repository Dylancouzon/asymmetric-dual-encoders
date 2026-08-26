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

    The phase-1 collapse leaves three suspects (fn_margin deleting the hardest negatives; tau
    concentrating softmax mass on bge's anisotropy tail; Adam lr on a weak signal). A negatives-only
    sweep with all three held at their suspect values could "conclude" that negatives do not help
    when temperature or the filter was the killer. So screen the knobs jointly and briefly,
    starting from the B checkpoint rather than the teacher init.

    One arm is the single most informative run in the program: teacher-mined-16, fn_margin=0,
    tau=0.05, lr=1e-3, 2k steps from B. If that still degrades a 0.4449 table, contrastive
    training against a frozen tower is structurally hostile and the wide ablation is a formality.
    """
    out = {}
    for tag, over in {
        "decisive":      {"hard_neg_k": 16, "fn_margin": 0.0, "temp": 0.05, "lr": 1e-3},
        "fnmargin-only": {"hard_neg_k": 16, "fn_margin": 0.0},
        "temp-only":     {"hard_neg_k": 16, "temp": 0.05},
        "lr-only":       {"hard_neg_k": 16, "lr": 3e-4},
        "hard-only":     {"hard_neg_k": 16},
        "baseline":      {"hard_neg_k": 0},
    }.items():
        out.update(grid("p2s", base, {tag: {"objective": "C", "steps_b": 4000, "steps_a": 2000,
                                            "eval_every": 1000, **over}}))
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
