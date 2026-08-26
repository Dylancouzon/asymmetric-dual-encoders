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
ALL_SOURCES = ("hotpotqa-train", "fever-train", "squad-train", "esci-us", "mrtydi-en")
NO_FEVER = tuple(s for s in ALL_SOURCES if s != "fever-train")


def phase1_objective():
    """A vs B vs C. B alone is also the Stage-0 distilled table the gate's G1 is judged on."""
    return grid("p1", BASE, {
        "objB": {"objective": "B", "steps_b": 8000, "steps_a": 0},
        "objA": {"objective": "A", "steps_b": 0, "steps_a": 12000},
        "objC": {"objective": "C"},
    })


def phase2_negatives(base):
    """BM25-mined negatives are added by phase2b once the BM25 mining cache exists."""
    return grid("p2", base, {
        "bank": {"hard_neg_k": 0},
        "mined16": {"hard_neg_k": 16},
        "mined32": {"hard_neg_k": 32},
        "mined16-nofn": {"hard_neg_k": 16, "fn_margin": 0.0},
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


def phase5_fever(base):
    """Does FEVER-train earn its place? If not, dropping it buys a clean untouched-final row."""
    return grid("p5", base, {"with-fever": {"sources": ALL_SOURCES},
                             "no-fever": {"sources": NO_FEVER}})


def save(name, res):
    p = REPO / "results" / f"m7_program_{name}.json"
    p.write_text(json.dumps(res, indent=1))
    print(f"wrote {p}")
