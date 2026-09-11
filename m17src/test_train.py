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


def test_admit_read_refuses_protected_paths_including_symlinks(tmp_path):
    protected = tmp_path / "results" / "frozen_eval" / "untouched-final"
    protected.mkdir(parents=True)
    (protected / "x.json").write_text("{}")
    with pytest.raises(common.ProtectedRead, match="protected content"):
        common.admit_read(protected / "x.json")
    link = tmp_path / "innocent.json"
    link.symlink_to(protected / "x.json")
    with pytest.raises(common.ProtectedRead, match="protected content"):
        common.admit_read(link)                    # the spelling is benign, the target is not
    for bad in ("work/m9reserve/a.json", "caches/reserved_qrels/b.json", "lotte/c.json",
                "results/frozen_eval/fever.json", "work/qrels/cqadup-android.json"):
        p = tmp_path / bad
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}")
        with pytest.raises(common.ProtectedRead):
            common.admit_read(p)
    # admitted FEVER TRAINING material stays admitted
    ok = tmp_path / "work" / "train" / "stores" / "fever-train.json"
    ok.parent.mkdir(parents=True)
    ok.write_text("{}")
    assert common.admit_read(ok) == ok.resolve()


def test_writers_refuse_the_frozen_destinations(tmp_path):
    for bad in ("results/perquery.json", "results/frozen_eval/untouched-final.json",
                "m7/FREEZE.json"):
        with pytest.raises(common.ProtectedWrite, match="immutable evidence"):
            common.write_json(common.REPO / bad, {"x": 1})
    assert common.write_json(tmp_path / "ok.json", {"x": 1}).exists()


def test_the_rehearsal_deletes_only_its_own_output(tmp_path):
    import rehearse17
    someone_elses = tmp_path / "m13work"
    someone_elses.mkdir()
    (someone_elses / "precious.json").write_text("{}")
    with pytest.raises(SystemExit, match="carries no .m17_rehearsal marker"):
        rehearse17._clear_rehearsal_dir(someone_elses)
    assert (someone_elses / "precious.json").exists()
    mine = tmp_path / "rehearsal"
    mine.mkdir()
    (mine / rehearse17.MARKER).write_text("x")
    (mine / "old.json").write_text("{}")
    rehearse17._clear_rehearsal_dir(mine)
    assert not mine.exists()
    rehearse17._clear_rehearsal_dir(tmp_path / "empty")        # absent: nothing to refuse


def test_the_real_registry_is_still_a_draft():
    """If this ever fails, the lock landed — and step 6, not this test, is what changed."""
    assert common.registry()["status"] == "DRAFT_NOT_EXECUTABLE"


def test_cli_refuses_a_real_arm_while_the_registry_is_a_draft():
    with pytest.raises(SystemExit, match="registry status"):
        T.main(["--arm", "VL-A", "--data", "nowhere"])


def test_batch_composition_is_the_measured_dose(reg):
    """`data.measured_dose_after_pre_lock_rule`, not the superseded fractions."""
    real = common.registry()
    dose = real["data"]["measured_dose_after_pre_lock_rule"]
    cfg = T.RunCfg.from_registry(real, "VL-A")
    g, c, p = cfg.batch_shape()
    assert (g, c, p) == (204, 32, 10) == (dose["general_views_per_batch"],
                                          dose["unpaired_coverage_views_per_batch"],
                                          dose["alias_pairs_per_batch"])
    assert g + c + 2 * p == cfg.batch == dose["batch"]
    # the pre-lock fallback batch derives proportionally, floors applied, sum exact
    half = T.RunCfg.from_registry(real, "VL-A", batch=128)
    g2, c2, p2 = half.batch_shape()
    assert (g2, c2, p2) == (102, 16, 5) and g2 + c2 + 2 * p2 == 128
    small = T.RunCfg.from_registry(reg, "VL-A")
    sg, sc, sp = small.batch_shape()
    assert sg + sc + 2 * sp == small.batch


def test_batch_shape_refuses_a_composition_that_does_not_sum(reg):
    cfg = T.RunCfg.from_registry(common.registry(), "VL-A")
    cfg.general_views = 200
    with pytest.raises(ValueError, match="does not sum to batch"):
        cfg.batch_shape()


def test_run_id_carries_the_phase(reg):
    real = common.registry()
    screen = T.RunCfg.from_registry(real, "VL", steps=real["training"]["screen_steps"])
    final = T.RunCfg.from_registry(real, "VL")
    assert screen.phase == "screen" and final.phase == "final"
    assert screen.run_id != final.run_id and "screen" in screen.run_id


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


def _streams(buckets, pids=None, views=None, fams=None, **kw):
    n = len(buckets)
    return T.build_streams(buckets, pids or [""] * n, views or [""] * n,
                           fams or [f"f{i}" for i in range(n)], seed=0, **kw)


def test_heldout_bucket_is_never_sampled():
    buckets = ["general"] * 5 + ["coverage"] * 3 + ["heldout"] * 4
    streams, pairs, report = _streams(buckets)
    drawn = set(streams["general"].draw(5).tolist()) | set(streams["coverage"].draw(3).tolist())
    assert drawn.isdisjoint(range(8, 12))
    assert len(pairs) == 0 and report["n_pairs"] == 0


def test_heldout_alias_pairs_never_enter_training():
    """Two `heldout` rows sharing a pair id used to be sampled for gradients."""
    buckets = ["coverage", "coverage", "heldout", "heldout"]
    with pytest.raises(SystemExit, match="not training rows"):
        _streams(buckets, pids=["p0", "p0", "p1", "p1"], views=["a", "b", "a", "b"],
                 fams=["f0", "f0", "f1", "f1"])


def test_alias_pairs_must_be_two_views_of_one_family():
    with pytest.raises(SystemExit, match="expected exactly two views"):
        _streams(["coverage"] * 3, pids=["p0", "p0", "p0"], views=["a", "b", "a"],
                 fams=["f0"] * 3)
    with pytest.raises(SystemExit, match=r"expected \['a', 'b'\]"):
        _streams(["coverage"] * 2, pids=["p0", "p0"], views=["a", "a"], fams=["f0", "f0"])
    with pytest.raises(SystemExit, match="expected one"):
        _streams(["coverage"] * 2, pids=["p0", "p0"], views=["a", "b"], fams=["f0", "f1"])
    streams, pairs, _ = _streams(["coverage"] * 2, pids=["p0", "p0"], views=["b", "a"],
                                 fams=["f0", "f0"])
    assert pairs.tolist() == [[1, 0]], "view a first, then view b"


def test_an_undersupplied_bucket_is_refused_not_shortened():
    with pytest.raises(SystemExit, match="one batch needs"):
        _streams(["general"] * 3 + ["coverage"] * 2, need={"general": 8, "coverage": 1})
    s = T.Stream("coverage", np.zeros(0, dtype=np.int64), seed=0)
    with pytest.raises(SystemExit, match="is empty but the registered batch"):
        s.draw(4)


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


def _resume_fixture(rehearsal, **over):
    from table import QueryTable
    ck = rehearsal["root"] / "run" / "recovery.pt"
    saved = torch.load(ck, map_location="cpu", weights_only=False)
    model = QueryTable(np.zeros((saved["vocab"], saved["dim"]), dtype=np.float32),
                       weight_init=np.ones(saved["vocab"], dtype=np.float32))
    cfg = T.RunCfg(**{**saved["cfg"],
                      "snapshot_steps": tuple(saved["cfg"]["snapshot_steps"]), **over})
    opt = torch.optim.Adam([{"params": [model.rows]}, {"params": [model.w_raw]}])
    eff = torch.zeros(saved["vocab"], saved["dim"])
    return ck, model, opt, cfg, eff, saved["heldout_sha256"]


def test_resume_refuses_a_different_tokenizer(rehearsal):
    """A resume across tokenizers would continue a schedule for a different table."""
    ck, model, opt, cfg, eff, ho = _resume_fixture(rehearsal, tokenizer_sha256="deadbeef")
    with pytest.raises(SystemExit, match="tokenizer"):
        T._resume(ck, model, opt, {}, cfg, eff, lambda *a: None, ho)


@pytest.mark.parametrize("field,value", [("arm", "C"), ("seed", 1), ("alias_pairs", 3),
                                         ("temperature", 0.1), ("rows_lr", 1e-4),
                                         ("cache_sha256", "0" * 64)])
def test_resume_refuses_a_different_experiment(rehearsal, field, value):
    """Row count, tokenizer and step budget alone let another arm, seed or dose resume."""
    ck, model, opt, cfg, eff, ho = _resume_fixture(rehearsal, **{field: value})
    with pytest.raises(SystemExit, match="configuration differs"):
        T._resume(ck, model, opt, {}, cfg, eff, lambda *a: None, ho)


def test_resume_refuses_a_different_heldout_slice(rehearsal):
    """Same length, different queries: the divergence history would be incomparable."""
    ck, model, opt, cfg, eff, ho = _resume_fixture(rehearsal)
    with pytest.raises(SystemExit, match="held-out slice hashes to"):
        T._resume(ck, model, opt, {}, cfg, eff, lambda *a: None, "0" * 64)


@pytest.mark.parametrize("field,value", [("check_every", 7), ("heldout_queries", 3),
                                         ("snapshot_steps", (1, 2, 3))])
def test_resume_binds_the_divergence_and_snapshot_protocol(rehearsal, field, value):
    ck, model, opt, cfg, eff, ho = _resume_fixture(rehearsal, **{field: value})
    with pytest.raises(SystemExit, match="configuration differs"):
        T._resume(ck, model, opt, {}, cfg, eff, lambda *a: None, ho)


def test_every_arm_requires_the_registered_alias_supply(tmp_path):
    """Every arm DRAWS the alias pairs; only VL-A adds their loss. A C run with too few pairs
    used to wrap the same handful inside one batch instead of refusing the dose."""
    from table import QueryTable
    n = 8
    model = QueryTable(np.zeros((4, 3), dtype=np.float32),
                       weight_init=np.ones(4, dtype=np.float32))
    data = {"model": model, "ids": [[0]] * n,
            "teacher_q": np.zeros((n, 3), dtype=np.float32),
            "bank": np.zeros((2, 3), dtype=np.float32),
            "candidate_ids": np.zeros((n, 2), dtype=np.int64),
            "teacher_scores": np.zeros((n, 2), dtype=np.float32),
            "buckets": ["general"] * 6 + ["coverage"] * 2,
            "alias_pair_ids": [""] * n, "alias_views": [""] * n,
            "families": [f"f{i}" for i in range(n)], "heldout_idx": [0]}
    cfg = T.RunCfg(arm="C", batch=8, general_views=4, coverage_views=2, alias_pairs=1,
                   rehearsal=True, device="cpu")
    with pytest.raises(SystemExit, match="bucket 'alias' holds 0 rows"):
        T.run(cfg, data, tmp_path / "out", resume=False, log=lambda *a: None)


def test_resume_restores_the_anchor_initialization(rehearsal):
    """The anchor must measure drift from where the RUN started, not from the resumed rows."""
    ck, model, opt, cfg, eff, ho = _resume_fixture(rehearsal)
    saved = torch.load(ck, map_location="cpu", weights_only=False)
    state, restored = T._resume(ck, model, opt, {}, cfg, eff, lambda *a: None, ho)
    assert torch.allclose(restored, saved["eff_init"])
    assert not torch.allclose(restored, eff), "a zero anchor would have been silently accepted"
    assert state["step"] and "train_running" in state


def test_resume_refuses_a_changed_stream_population():
    s = T.Stream("general", np.arange(10), seed=0)
    st = s.state()
    with pytest.raises(SystemExit, match="would select different queries"):
        T.Stream("general", np.arange(9), seed=0).load_state(st)


def test_heldout_slice_must_be_unique_disjoint_and_non_empty():
    streams, pairs, _ = _streams(["general"] * 4 + ["coverage"] * 2)
    cfg = T.RunCfg(rehearsal=True, heldout_queries=2000)
    with pytest.raises(SystemExit, match="held-out slice is empty"):
        T._check_heldout(np.zeros(0, dtype=np.int64), streams, pairs, cfg)
    with pytest.raises(SystemExit, match="repeats queries"):
        T._check_heldout(np.array([7, 7]), streams, pairs, cfg)
    with pytest.raises(SystemExit, match="also in a training bucket"):
        T._check_heldout(np.array([0, 9]), streams, pairs, cfg)
    T._check_heldout(np.array([9, 10]), streams, pairs, cfg)          # rehearsal: any size
    with pytest.raises(SystemExit, match="the registry pins 2000"):
        T._check_heldout(np.array([9, 10]), streams, pairs, T.RunCfg(heldout_queries=2000))


def test_divergence_monitors_the_same_components_on_both_sides(rehearsal):
    rec = json.loads((rehearsal["root"] / "run" / "run_record.json").read_text())
    h = rec["history"][-1]
    for side in ("train_components", "heldout_components"):
        assert set(h[side]) == {"cosine", "listwise", "monitored"}
        assert h[side]["monitored"] == pytest.approx(h[side]["cosine"] + h[side]["listwise"])
    assert "anchor" not in h["train_components"] and "alias" not in h["heldout_components"]
    assert rec["heldout"]["n"] > 0


def test_listwise_kl_matches_a_hand_computed_value():
    """Three candidates, one masked. KL(teacher || student) over the two live slots only."""
    from table import QueryTable
    import math
    T_ = 0.5
    q_s = torch.tensor([[1.0, 0.0]])
    cand = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]])       # third slot is masked
    mask = torch.tensor([[True, True, False]])
    tsc = torch.tensor([[0.5, 0.1, 9.0]])                              # 9.0 must not count
    model = QueryTable(np.zeros((2, 2), dtype=np.float32), learned_weights=False)
    cfg = T.RunCfg(cosine_weight=0.0, init_anchor_weight=0.0, listwise_weight=1.0,
                   temperature=T_)
    _, parts = T.losses(model, q_s, q_s, cand, mask, tsc, None, None, cfg,
                        {"listwise": True, "alias_consistency": False})

    def softmax2(a, b):
        m = max(a, b)
        ea, eb = math.exp(a - m), math.exp(b - m)
        return ea / (ea + eb), eb / (ea + eb)

    pt = softmax2(0.5 / T_, 0.1 / T_)                                  # teacher scores / T
    ps = softmax2(1.0 / T_, 0.0 / T_)                                  # dot(q_s, d) / T
    want = sum(p * (math.log(p) - math.log(s)) for p, s in zip(pt, ps))
    assert parts["listwise_kl"] == pytest.approx(want, rel=1e-6)


def test_anchor_is_the_mean_square_deviation_of_effective_rows():
    """Effective rows are softplus(w) * rows; the anchor is their MSE against the init."""
    from table import QueryTable
    rows = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    model = QueryTable(rows, weight_init=np.array([2.0, 0.5], dtype=np.float32))
    eff = (model.token_weights().detach().unsqueeze(1) * model.rows.detach())
    assert torch.allclose(eff, torch.tensor([[2.0, 0.0], [0.0, 1.0]]), atol=1e-5)
    eff_init = eff - torch.tensor([[1.0, 0.0], [0.0, 3.0]])            # deviations 1 and 3
    q = torch.tensor([[1.0, 0.0]])
    cfg = T.RunCfg(cosine_weight=0.0, init_anchor_weight=1.0)
    _, parts = T.losses(model, q, q, None, None, None, eff_init, None, cfg,
                        {"listwise": False, "alias_consistency": False})
    assert parts["anchor"] == pytest.approx((1.0 ** 2 + 3.0 ** 2) / 4, rel=1e-5)


def test_warm_start_verifies_the_freeze_hash_and_explicit_metadata(tmp_path):
    """Matching teacher strings are not lineage: only the frozen bytes are p35w-2m-s2500."""
    from table import Preproc, QueryTable, save_table
    rng = np.random.default_rng(0)
    V = 30522
    m = QueryTable(rng.normal(size=(V, 4)).astype(np.float32),
                   weight_init=np.ones(V, dtype=np.float32))
    pre = Preproc(pool_mode="sqrt")
    save_table(tmp_path / "ck.npz", m, pre, meta={"weights_folded": False})
    with pytest.raises(SystemExit, match="training_checkpoint_sha256"):
        T.load_warm_start(tmp_path / "ck.npz", expect_vocab=V, device="cpu", rehearsal=False)
    # unmarked fold state is refused outright
    meta = json.loads((tmp_path / "ck.meta.json").read_text())
    meta.pop("weights_folded")
    (tmp_path / "ck.meta.json").write_text(json.dumps(meta))
    with pytest.raises(SystemExit, match="does not state"):
        T.load_warm_start(tmp_path / "ck.npz", device="cpu", rehearsal=True)


def test_warm_start_refuses_missing_or_misshapen_scalars(tmp_path):
    from table import Preproc, QueryTable, save_table
    m = QueryTable(np.ones((8, 4), dtype=np.float32), learned_weights=False)
    save_table(tmp_path / "nw.npz", m, Preproc(pool_mode="sqrt"),
               meta={"weights_folded": False})
    with pytest.raises(SystemExit, match="no learned scalars"):
        T.load_warm_start(tmp_path / "nw.npz", device="cpu", rehearsal=True)
    m2 = QueryTable(np.ones((8, 4), dtype=np.float32),
                    weight_init=np.ones(8, dtype=np.float32))
    save_table(tmp_path / "bad.npz", m2, Preproc(pool_mode="sqrt"),
               meta={"weights_folded": False})
    z = dict(np.load(tmp_path / "bad.npz"))
    z["token_weights"] = np.ones(3, dtype=np.float32)
    np.savez(tmp_path / "bad.npz", **z)
    with pytest.raises(SystemExit, match="one positive scalar per row"):
        T.load_warm_start(tmp_path / "bad.npz", device="cpu", rehearsal=True)


def test_load_prepared_is_arm_aware(tmp_path):
    """A control arm pointed at expanded prepared data must not get the extended table."""
    reg = common.registry()
    cfg = T.RunCfg.from_registry(reg, "C", rehearsal=True)
    with pytest.raises(SystemExit, match="carries `new_rows`"):
        T._load_prepared(tmp_path, {"new_rows": "new_rows.npy", "warm_start": "w.npz",
                                    "tokenizer": "t.json"}, cfg, reg)
    cfg_v = T.RunCfg.from_registry(reg, "VL", rehearsal=True)
    with pytest.raises(SystemExit, match="carries no `new_rows`"):
        T._load_prepared(tmp_path, {"warm_start": "w.npz", "tokenizer": "t.json"}, cfg_v, reg)


def test_a_real_run_refuses_an_unregistered_dose_or_seed():
    reg = common.registry()
    cfg = T.RunCfg.from_registry(reg, "VL", steps=123)
    with pytest.raises(SystemExit, match="neither the registered screen"):
        T._check_locked_config(cfg, reg)
    with pytest.raises(SystemExit, match="not the locked batch"):
        T._check_locked_config(T.RunCfg.from_registry(reg, "VL", batch=64), reg)
    with pytest.raises(SystemExit, match="not a registered seed"):
        T._check_locked_config(T.RunCfg.from_registry(reg, "VL", seed=7), reg)
    T._check_locked_config(T.RunCfg.from_registry(reg, "VL", seed=1), reg)


def test_run_refuses_a_draft_registry_even_when_called_directly():
    """`train.run()` is a driver, not a helper: it meets the same bar as the CLI."""
    cfg = T.RunCfg.from_registry(common.registry(), "VL-A")
    with pytest.raises(SystemExit, match="registry status"):
        T.run(cfg, {}, "/tmp/does-not-matter", resume=False, log=lambda *a: None)
