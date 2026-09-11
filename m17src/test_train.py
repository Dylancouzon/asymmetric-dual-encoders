"""The driver's registered behaviours, and the refusals that protect them.

The refusals are the point: a status gate that can be bypassed silently, a warm start that
accepts a folded release, or a resume that accepts a different tokenizer would each let an
unregistered artifact through wearing the right filename.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

import common
import train as T


def test_registry_gate_refuses_a_draft_and_names_the_bypass():
    with pytest.raises(SystemExit) as e:
        common.require_executable({"status": "DRAFT_NOT_EXECUTABLE"}, rehearsal=False)
    assert "--rehearsal" in str(e.value)
    assert common.require_executable({"status": "DRAFT_NOT_EXECUTABLE"}, rehearsal=True) \
        == "DRAFT_NOT_EXECUTABLE"
    assert common.require_executable({"status": "EXECUTABLE"}, rehearsal=False) == "EXECUTABLE"


def test_the_real_registry_is_still_a_draft():
    """If this ever fails, the lock landed — and step 6, not this test, is what changed."""
    assert common.registry()["status"] == "DRAFT_NOT_EXECUTABLE"


def test_cli_refuses_a_real_arm_while_the_registry_is_a_draft():
    with pytest.raises(SystemExit, match="registry status"):
        T.main(["--arm", "VL-A", "--data", "nowhere"])


def test_batch_composition_matches_the_registry(reg):
    cfg = T.RunCfg.from_registry(common.registry(), "VL-A")
    g, c, p = cfg.batch_shape()
    assert (g, c, p) == (192, 32, 16)          # 0.75 / 0.25 with 0.125 of the batch as views
    assert g + c + 2 * p == cfg.batch
    small = T.RunCfg.from_registry(reg, "VL-A")
    assert sum(small.batch_shape()[:2]) + 2 * small.batch_shape()[2] == small.batch


def test_arms_match_the_registry():
    reg_arms = {a["id"]: a for a in common.registry()["training"]["arms"]}
    assert set(reg_arms) == set(T.ARMS)
    for aid, spec in reg_arms.items():
        for k in ("vocab_extension", "listwise", "alias_consistency"):
            assert T.ARMS[aid][k] == spec[k]


def test_stream_is_without_replacement_and_counts_passes_per_bucket():
    s = T.Stream("general", np.arange(10), seed=0)
    first = s.draw(10)
    assert sorted(first.tolist()) == list(range(10))
    assert s.passes == 1
    s.draw(5)
    assert s.passes == 1
    # a resumed stream continues from the same position in the same order
    t = T.Stream("general", np.arange(10), seed=0)
    t.load_state(s.state())
    assert np.array_equal(t.draw(5), s.draw(5))


def test_stream_order_is_stable_across_processes():
    """Not `hash(str)`: that is salted per process and would break resume."""
    a = T.Stream("coverage", np.arange(20), seed=1).draw(20)
    b = T.Stream("coverage", np.arange(20), seed=1).draw(20)
    assert np.array_equal(a, b)


def test_heldout_bucket_is_never_sampled():
    buckets = ["general"] * 5 + ["coverage"] * 3 + ["heldout"] * 4
    streams, pairs, incomplete = T.build_streams(buckets, [""] * len(buckets), seed=0)
    drawn = set(streams["general"].draw(20).tolist()) | set(streams["coverage"].draw(20).tolist())
    assert drawn.isdisjoint(range(8, 12))
    assert len(pairs) == 0 and incomplete == []


def test_alias_pairs_are_paired_and_incomplete_pairs_are_reported():
    buckets = ["coverage"] * 5
    pids = ["p0", "p0", "p1", "p1", "p2"]
    streams, pairs, incomplete = T.build_streams(buckets, pids, seed=0)
    assert pairs.shape == (2, 2) and incomplete == ["p2"]
    assert len(streams["coverage"].population) == 0


def test_warm_start_refuses_a_folded_release(tmp_path):
    from table import Preproc, QueryTable, save_release, save_table
    rng = np.random.default_rng(0)
    # Full base vocabulary: `save_release` self-verifies through the real WordPiece tokenizer,
    # so a toy row count would index off the end before the gate under test is reached.
    V = 30522
    m = QueryTable(rng.normal(size=(V, 4)).astype(np.float32),
                   weight_init=np.ones(V, dtype=np.float32))
    pre = Preproc(pool_mode="sqrt")
    save_release(tmp_path / "rel.npz", m, pre, device="cpu")
    with pytest.raises(SystemExit, match="FOLDED release"):
        T.load_warm_start(tmp_path / "rel.npz", device="cpu", rehearsal=True)
    save_table(tmp_path / "ck.npz", m, pre, meta={"weights_folded": False})
    model, lineage = T.load_warm_start(tmp_path / "ck.npz", expect_vocab=V, device="cpu",
                                       rehearsal=True)
    assert model.rows.shape[0] == V and lineage["vocab"] == V
    with pytest.raises(SystemExit, match="expected 99 base rows"):
        T.load_warm_start(tmp_path / "ck.npz", expect_vocab=99, device="cpu", rehearsal=True)


def test_extend_model_appends_rows_with_unit_scalars():
    from table import QueryTable
    rng = np.random.default_rng(0)
    m = QueryTable(rng.normal(size=(6, 4)).astype(np.float32),
                   weight_init=np.full(6, 2.0, dtype=np.float32))
    new = rng.normal(size=(2, 4)).astype(np.float32)
    ext = T.extend_model(m, new, "cpu")
    assert ext.rows.shape == (8, 4)
    assert np.allclose(ext.rows.detach().numpy()[6:], new, atol=1e-6)
    assert np.allclose(ext.token_weights().detach().numpy()[6:], 1.0, atol=1e-5)
    assert np.allclose(ext.rows.detach().numpy()[:6], m.rows.detach().numpy(), atol=1e-6)


def test_alias_term_is_batch_normalized():
    """`(2/B) * sum_pairs (1 - cos)` — a paired view's alias coefficient is the weight times
    its own per-view cosine coefficient, which the earlier per-pair mean was not."""
    from table import QueryTable
    B = 8
    q = torch.zeros(B, 4)
    q[:, 0] = 1.0
    q[1] = torch.tensor([0.0, 1.0, 0.0, 0.0])           # one pair fully disagrees
    slots = torch.tensor([[0, 1], [2, 3]])
    model = QueryTable(np.zeros((3, 4), dtype=np.float32), learned_weights=False)
    cfg = T.RunCfg(alias_weight=1.0, cosine_weight=0.0, init_anchor_weight=0.0)
    total, parts = T.losses(model, q, q, None, None, None, None, slots, cfg,
                            {"listwise": False, "alias_consistency": True})
    assert parts["alias"] == pytest.approx(2.0 / B * 1.0)


def test_divergence_flag_needs_two_consecutive_rises():
    state = {"history": [], "flags": []}
    for h, t in ((1.0, 5.0), (0.9, 4.0), (0.8, 3.0)):
        state["history"].append({"step": len(state["history"]), "heldout": h,
                                 "train_running_mean": t})
        T._flag_divergence(state, lambda *a: None)
    assert state["flags"] == []
    for h, t in ((0.9, 2.0), (1.0, 1.0)):
        state["history"].append({"step": len(state["history"]), "heldout": h,
                                 "train_running_mean": t})
        T._flag_divergence(state, lambda *a: None)
    assert len(state["flags"]) == 1 and "after lock" in state["flags"][0]["note"]


def test_run_record_and_snapshots(rehearsal):
    rec = json.loads((rehearsal["root"] / "run" / "run_record.json").read_text())
    assert rec["arm"] == "VL-A" and rec["rehearsal"] is True
    assert sorted(int(s) for s in rec["snapshots"]) == [15, 18, 20]
    assert rec["identity"]["tokenizer_sha256"] and rec["identity"]["candidate_cache_sha256"]
    assert rec["bucket_passes"]["general"]["n"] > 0
    for s in rec["snapshots"].values():
        assert (rehearsal["root"] / "run" / s).exists()


def test_resume_refuses_a_different_tokenizer(rehearsal, tmp_path):
    """A resume across tokenizers would continue a schedule for a different table."""
    from table import QueryTable
    ck = rehearsal["root"] / "run" / "recovery.pt"
    saved = torch.load(ck, map_location="cpu", weights_only=False)
    model = QueryTable(np.zeros((saved["vocab"], 16), dtype=np.float32),
                       weight_init=np.ones(saved["vocab"], dtype=np.float32))
    cfg = T.RunCfg(**{**saved["cfg"], "tokenizer_sha256": "deadbeef",
                      "snapshot_steps": tuple(saved["cfg"]["snapshot_steps"])})
    opt = torch.optim.Adam([{"params": [model.rows]}, {"params": [model.w_raw]}])
    with pytest.raises(SystemExit, match="tokenizer"):
        T._resume(ck, model, opt, {}, cfg, lambda *a: None)
