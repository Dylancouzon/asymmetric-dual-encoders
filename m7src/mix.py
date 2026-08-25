"""Loader over the built training mix. Splits TRAIN vs the held-out dev slices."""
import json

import numpy as np

from _paths import WORK
from trainmix import heldout

TRAIN = WORK / "train"
PAIR_SOURCES = ["hotpotqa-train", "fever-train", "squad-train", "esci-us", "mrtydi-en"]
QUERYTEXT_SOURCES = ["nqopen", "triviaqa"]


def load_source(name):
    return json.loads((TRAIN / "sources" / f"{name}.json").read_text())


def load_store(name):
    b = json.loads((TRAIN / "stores" / f"{name}.json").read_text())
    return b["ids"], b["texts"]


def available_sources():
    return [s for s in PAIR_SOURCES if (TRAIN / "sources" / f"{s}.json").exists()]


def split_pairs(sources=None):
    """-> (train_pairs, heldout_pairs); each item is (source, qid, query, [pos_docids], [hardneg])."""
    tr, ho = [], []
    for s in (sources or available_sources()):
        blob = load_source(s)
        for p in blob["pairs"]:
            item = (s, p["qid"], p["query"], p["pos"], p.get("hardneg", []))
            (ho if heldout(s, p["qid"]) else tr).append(item)
    return tr, ho


def query_texts(sources=None, include_querytext=True, train_only=True):
    """All query strings usable for objective B (distillation), TRAIN partition only by default."""
    out = []
    tr, ho = split_pairs(sources)
    out += [q for _, _, q, _, _ in (tr if train_only else tr + ho)]
    if include_querytext:
        for s in QUERYTEXT_SOURCES:
            p = TRAIN / "querytext" / f"{s}.json"
            if not p.exists():
                continue
            qs = json.loads(p.read_text())
            out += [q for i, q in enumerate(qs) if not heldout(s, str(i)) or not train_only]
    return out
