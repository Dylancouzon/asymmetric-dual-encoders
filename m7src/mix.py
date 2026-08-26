"""Loader over the built training mix. Splits TRAIN vs the held-out dev slices."""
import json
from functools import lru_cache

import numpy as np

from _paths import WORK
from trainmix import heldout

TRAIN = WORK / "train"
PAIR_SOURCES = ["hotpotqa-train", "fever-train", "squad-train", "esci-us", "mrtydi-en"]
QUERYTEXT_SOURCES = ["nqopen", "triviaqa"]


@lru_cache(maxsize=None)
def load_source(name):
    """Memoized: these files are 2-21 MB of JSON and callers reach for them inside loops.
    An un-memoized version re-parsed a 16 MB file once per training pair in decontam.py --
    352,190 times -- and the step never finished. Bounded: five sources, ~63 MB of JSON.
    load_store is deliberately NOT cached; hotpotqa-corpus alone is 5.23M strings."""
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


def query_texts(sources=None, include_querytext=True, train_only=True, decontaminated=True):
    """Query strings usable for objective B (distillation).

    decontaminated=True is the default and the only setting any reported number may use: it
    restricts to the pairs and query-text rows that survived fingerprint decontamination.
    """
    from _paths import WORK as _W
    out = []
    tr, ho = split_pairs(sources)
    if decontaminated and train_only:
        kp = _W / "decontam" / "kept.json"
        if not kp.exists():
            raise RuntimeError(f"{kp} missing: run decontam.py before any training or probe. "
                               "This never falls back to undecontaminated text.")
        allow = {k: set(v) for k, v in json.loads(kp.read_text()).items()}
        tr = [p for p in tr if p[1] in allow.get(p[0], set())]
    out += [q for _, _, q, _, _ in (tr if train_only else tr + ho)]
    if include_querytext:
        kq = None
        if decontaminated and train_only:
            p = _W / "decontam" / "kept_querytext.json"
            if not p.exists():
                raise RuntimeError(f"{p} missing: run decontam_querytext.py before using "
                                   "query-text-only sources. No silent fallback.")
            kq = json.loads(p.read_text())
        for s in QUERYTEXT_SOURCES:
            p = TRAIN / "querytext" / f"{s}.json"
            if not p.exists():
                continue
            qs = json.loads(p.read_text())
            if kq is None:
                out += [q for i, q in enumerate(qs) if not heldout(s, str(i)) or not train_only]
            elif s in kq:
                out += [qs[i] for i in kq[s]]
            else:
                raise RuntimeError(f"kept_querytext.json has no entry for {s}: re-run "
                                   "decontam_querytext.py")
    return out
