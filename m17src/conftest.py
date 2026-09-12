"""Shared synthetic fixtures for the M17 checks.

Everything is built in `tmp_path`. No test reads `results/`, the real work tree, a development
component, the M17 panel or any protected surface, and no test writes a registered artifact.
The one session-scoped fixture is the tiny rehearsal world, built once and reused read-only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "m7src", REPO / "m17src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import rehearse17                                                          # noqa: E402


@pytest.fixture(scope="session")
def tok_and_vocab():
    return rehearse17.build_tokenizer()


@pytest.fixture(scope="session")
def reg():
    r, _ = rehearse17.rehearsal_registry()
    return r


@pytest.fixture
def tiny_world(tmp_path):
    """Queries, a bank and teacher/v1 vectors — the inputs the cache builder takes."""
    import cache
    tok, n = rehearse17.build_tokenizer()
    rng = np.random.default_rng(7)
    queries, doc_ids = rehearse17.fixture_queries(n_general=12, n_coverage=6, n_pairs=3,
                                                  n_heldout=4, seed=7)
    vecs = rehearse17.unit(rng.normal(size=(len(doc_ids), rehearse17.DIM))).astype(np.float32)
    bank = cache.Bank(doc_ids, vecs, ["synthetic"] * len(doc_ids), seed=7)
    tq = rehearse17.unit(rng.normal(size=(len(queries), rehearse17.DIM)))
    vq = rehearse17.unit(rng.normal(size=(len(queries), rehearse17.DIM)))
    return {"tok": tok, "base_vocab": n, "queries": queries, "bank": bank,
            "teacher_q": tq, "v1_q": vq, "dir": tmp_path}


@pytest.fixture(scope="session")
def rehearsal(tmp_path_factory):
    """One full synthetic end-to-end run, shared read-only by the integration checks."""
    root = tmp_path_factory.mktemp("m17rehearsal") / "world"
    rec = rehearse17.build(root, seed=0, device="cpu", log=lambda *a, **k: None)
    return {"record": rec, "root": root}
