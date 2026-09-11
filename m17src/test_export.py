"""Folding, averaging and the M17 gates — each gate must refuse what it exists to refuse.

Modeled on `m11/release/test_gates.py`: passing gates prove nothing on their own, so every
check below breaks the bundle in a way that could plausibly happen and asserts the refusal,
with a positive control that the untouched bundle still passes everything.

None of these touch M11's gates or `m7/FREEZE.json`.
"""
from __future__ import annotations

import json
import shutil

import numpy as np
import pytest

import common
import export


@pytest.fixture
def bundle(rehearsal, tmp_path):
    d = tmp_path / "bundle"
    shutil.copytree(rehearsal["root"] / "bundle-endpoint", d)
    return d


def test_positive_control_all_gates_pass(bundle):
    rep = export.run_gates(bundle, log=lambda *a: None)
    assert set(rep) == {g.__name__ for g in export.GATES}


def test_fold_twice_is_refused(rehearsal, tmp_path):
    src = rehearsal["root"] / "run" / "endpoint.npz"
    meta_p = src.parent / (src.stem + ".meta.json")
    shutil.copy(src, tmp_path / "e.npz")
    meta = json.loads(meta_p.read_text())
    meta["weights_folded"] = True
    (tmp_path / "e.meta.json").write_text(json.dumps(meta))
    with pytest.raises(SystemExit, match="already folded"):
        export.effective_rows(tmp_path / "e.npz")


def test_averaging_refuses_a_missing_snapshot(rehearsal, reg):
    run = rehearsal["root"] / "run"
    paths = [run / f"snapshot_{s:06d}.npz" for s in (15, 18)]
    with pytest.raises(SystemExit, match="2 snapshots supplied"):
        export.average_snapshots(paths, reg)


def test_averaging_refuses_a_fourth_snapshot_and_a_missing_step(rehearsal, reg, tmp_path):
    """The registered mean is of exactly three snapshots with three distinct registered steps.

    A fourth table whose identity matches but whose `m17_step` is absent used to be dropped
    from the window check and still averaged in.
    """
    run = rehearsal["root"] / "run"
    paths = []
    for s in (15, 18, 20):
        for ext in (".npz", ".meta.json"):
            shutil.copy(run / f"snapshot_{s:06d}{ext}", tmp_path / f"snapshot_{s:06d}{ext}")
        paths.append(tmp_path / f"snapshot_{s:06d}.npz")
    for ext in (".npz", ".meta.json"):
        shutil.copy(run / f"snapshot_{s:06d}{ext}", tmp_path / f"extra{ext}")
    m = json.loads((tmp_path / "extra.meta.json").read_text())
    m.pop("m17_step")
    (tmp_path / "extra.meta.json").write_text(json.dumps(m))
    with pytest.raises(SystemExit, match="4 snapshots supplied"):
        export.average_snapshots(paths + [tmp_path / "extra.npz"], reg)
    # and a three-input set with one unstepped snapshot is refused too
    with pytest.raises(SystemExit, match="records no `m17_step`"):
        export.average_snapshots(paths[:2] + [tmp_path / "extra.npz"], reg)


def test_averaging_refuses_an_incomplete_identity(rehearsal, reg, tmp_path):
    run = rehearsal["root"] / "run"
    paths = []
    for s in (15, 18, 20):
        for ext in (".npz", ".meta.json"):
            shutil.copy(run / f"snapshot_{s:06d}{ext}", tmp_path / f"snapshot_{s:06d}{ext}")
        paths.append(tmp_path / f"snapshot_{s:06d}.npz")
    for s in (15, 18, 20):                       # a missing field must not read as "equal"
        m = json.loads((tmp_path / f"snapshot_{s:06d}.meta.json").read_text())
        m.pop("vocabulary_sha256")
        (tmp_path / f"snapshot_{s:06d}.meta.json").write_text(json.dumps(m))
    with pytest.raises(SystemExit, match="does not record its run, tokenizer"):
        export.average_snapshots(paths, reg)


def test_averaging_refuses_mixed_runs(rehearsal, reg, tmp_path):
    run = rehearsal["root"] / "run"
    paths = []
    for s in (15, 18, 20):
        for ext in (".npz", ".meta.json"):
            shutil.copy(run / f"snapshot_{s:06d}{ext}", tmp_path / f"snapshot_{s:06d}{ext}")
        paths.append(tmp_path / f"snapshot_{s:06d}.npz")
    m = json.loads((tmp_path / "snapshot_000018.meta.json").read_text())
    m["m17_run_id"] = "some-other-run"
    (tmp_path / "snapshot_000018.meta.json").write_text(json.dumps(m))
    with pytest.raises(SystemExit, match="AVERAGING REFUSED"):
        export.average_snapshots(paths, reg)


def test_average_is_the_equal_mean_of_effective_rows(rehearsal, reg):
    run = rehearsal["root"] / "run"
    paths = [run / f"snapshot_{s:06d}.npz" for s in (15, 18, 20)]
    mean, diag = export.average_snapshots(paths, reg)
    want = np.mean([export.effective_rows(p)[0] for p in paths], axis=0)
    assert np.allclose(mean, want, atol=1e-6)
    assert [d["step"] for d in diag["snapshots"]] == [15, 18, 20, "mean_last_three"]
    assert all(d["rms"] > 0 for d in diag["snapshots"])


def test_a_wrong_fallback_id_is_no_longer_a_cap_bypass(rehearsal, reg, tmp_path):
    """`fallback_id != frozen CLS` used to mark the bundle a fixture and skip the dim / added
    row / 35M checks. Only the explicit `fixture=True` provenance flag may do that."""
    from tokenizers import Tokenizer
    rows, _ = export.effective_rows(rehearsal["root"] / "run" / "endpoint.npz")
    tok = Tokenizer.from_file(str(rehearsal["root"] / "data" / "tokenizer_ext.json"))
    with pytest.raises(SystemExit, match="dimensions"):
        export.build_bundle(tmp_path / "b", rows, tok, {}, reg, fallback_id=2)
    with pytest.raises(SystemExit, match="outside a table of"):
        export.build_bundle(tmp_path / "n", rows, tok, {}, reg, fallback_id=-1, fixture=True)


def test_endpoint_export_needs_the_identity_averaging_needs():
    with pytest.raises(SystemExit, match="does not record its run, tokenizer"):
        export.snapshot_identity({"m17_step": 3, "m17_run_id": "r"}, "endpoint.npz")
    with pytest.raises(SystemExit, match="records no `m17_step`"):
        export.snapshot_identity({f: "x" for f in export.IDENTITY_FIELDS}, "endpoint.npz")


def test_export_main_refuses_a_tokenizer_that_is_not_the_trained_one(rehearsal, tmp_path):
    """Conformance compares the bundle tokenizer against itself, so a same-sized tokenizer
    with different ids passed every gate."""
    src = rehearsal["root"] / "data" / "tokenizer_ext.json"
    bad = tmp_path / "tokenizer.json"
    bad.write_text(src.read_text() + " ")
    with pytest.raises(SystemExit, match="--tokenizer hashes to"):
        export.main(["--endpoint", str(rehearsal["root"] / "run" / "endpoint.npz"),
                     "--tokenizer", str(bad), "--out", str(tmp_path / "b")])


def test_gate_artifact_catches_a_stale_staging_dir(bundle):
    z = dict(np.load(bundle / "model.npz"))
    z["rows_int8"] = np.zeros_like(z["rows_int8"])
    np.savez(bundle / "model.npz", **z)
    with pytest.raises(SystemExit, match="staged model_npz_sha256"):
        export.gate_artifact(bundle)


def test_gate_artifact_checks_the_tokenizer_and_config_hashes_too(bundle):
    """A same-size tokenizer with different ids used to pass every gate."""
    raw = json.loads((bundle / "tokenizer.json").read_text())
    raw["model"]["vocab"] = {k: v for k, v in raw["model"]["vocab"].items()}
    raw["added_tokens"] = raw.get("added_tokens", [])
    (bundle / "tokenizer.json").write_text(json.dumps(raw) + " ")     # same tokens, new bytes
    with pytest.raises(SystemExit, match="staged tokenizer_sha256"):
        export.gate_artifact(bundle)


def test_gate_encoder_spec_checks_config_kwargs_and_the_actual_dimension(bundle):
    assert "config_kwargs" in export.SPEC_FIELDS
    cfg = json.loads((bundle / "config.json").read_text())
    cfg["document_encoder"]["config_kwargs"] = {"unpad_inputs": True}
    (bundle / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(SystemExit, match="encoder_spec fields"):
        export.gate_encoder_spec(bundle)


def test_table_limits_refuse_a_wrong_dimension_and_an_oversized_table():
    reg = common.registry()
    with pytest.raises(SystemExit, match="dimensions"):
        export.check_table_limits(np.zeros((reg["base_vocab"], 16), dtype=np.float32), reg)
    too_many = reg["base_vocab"] + reg["added_rows_max"] + 1
    with pytest.raises(SystemExit, match="added rows"):
        export.check_table_limits(np.zeros((too_many, reg["dim"]), dtype=np.float32), reg)
    ok = export.check_table_limits(
        np.zeros((reg["base_vocab"] + reg["added_rows_max"], reg["dim"]), dtype=np.float32), reg)
    assert ok["parameters"] == 34433850 <= reg["student_parameter_cap"]
    small = {**reg, "student_parameter_cap": 1000}
    with pytest.raises(SystemExit, match="student cap"):
        export.check_table_limits(np.zeros((reg["base_vocab"], reg["dim"]), dtype=np.float32),
                                  small)


def test_gate_conformance_refuses_non_finite_rows(bundle):
    """`nan > tol` is False: a NaN table must be rejected before the comparison."""
    z = dict(np.load(bundle / "model.npz"))
    z["rows_fp16"] = np.full_like(z["rows_fp16"], np.nan)
    np.savez(bundle / "model.npz", **z)
    with pytest.raises(SystemExit, match="non-finite"):
        export.gate_conformance(bundle)


def test_gate_conformance_refuses_a_bad_scale(bundle):
    z = dict(np.load(bundle / "model.npz"))
    z["int8_scale"] = np.zeros_like(z["int8_scale"])
    np.savez(bundle / "model.npz", **z)
    with pytest.raises(SystemExit, match="finite and strictly positive"):
        export.gate_conformance(bundle)
    z["int8_scale"] = np.ones(3, dtype=np.float32)
    np.savez(bundle / "model.npz", **z)
    with pytest.raises(SystemExit, match="one positive scale per row"):
        export.gate_conformance(bundle)


def test_conformance_tolerance_is_recorded_in_provenance(bundle):
    prov = json.loads((bundle / "provenance.json").read_text())
    assert prov["conformance_tolerance_max_abs"] == export.CONFORMANCE_TOL == 1e-5
    assert "loader-vs-loader" in prov["conformance_tolerance_note"]
    rep = export.gate_conformance(bundle)
    assert rep["tolerance"] == export.CONFORMANCE_TOL


def test_gate_encoder_spec_catches_a_drifted_document_space(bundle):
    cfg = json.loads((bundle / "config.json").read_text())
    cfg["document_encoder"]["revision"] = "0" * 40
    (bundle / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(SystemExit, match="encoder_spec fields"):
        export.gate_encoder_spec(bundle)


def test_gate_encoder_spec_catches_a_changed_preproc(bundle):
    cfg = json.loads((bundle / "config.json").read_text())
    cfg["preproc"]["pool_mode"] = "mean"
    (bundle / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(SystemExit, match="preproc"):
        export.gate_encoder_spec(bundle)


def test_gate_tokenizer_catches_unsanitised_padding(bundle):
    raw = json.loads((bundle / "tokenizer.json").read_text())
    raw["padding"] = {"strategy": {"Fixed": 512}, "direction": "Right", "pad_id": 0,
                      "pad_type_id": 0, "pad_token": "[PAD]", "pad_to_multiple_of": None}
    (bundle / "tokenizer.json").write_text(json.dumps(raw))
    with pytest.raises(SystemExit, match="padding is enabled"):
        export.gate_tokenizer(bundle)


def test_gate_files_catches_an_extra_or_missing_file(bundle):
    (bundle / "stray.txt").write_text("x")
    with pytest.raises(SystemExit, match="unexpected"):
        export.gate_files(bundle)
    (bundle / "stray.txt").unlink()
    (bundle / "config.json").unlink()
    with pytest.raises(SystemExit, match="missing"):
        export.gate_files(bundle)


def test_gate_conformance_catches_a_wrong_loader_rule(bundle, monkeypatch):
    """A loader that drops repeated-token saturation must fail, not pass quietly."""
    import loader_np

    def broken(self, ids):
        if not ids:
            return self._fallback
        uniq = np.unique(np.asarray(ids, dtype=np.int64))
        v = self._gather(uniq).sum(0)
        return self._normalize(v).astype(np.float32)

    monkeypatch.setattr(loader_np.M17QueryEncoder, "_encode_ids", broken)
    with pytest.raises(SystemExit, match="does not reproduce the torch query path"):
        export.gate_conformance(bundle)


def test_kubernetes_attribution_ships_with_the_bundle(rehearsal, reg, tmp_path):
    """CC BY 4.0: the attribution notice ships WITH any weights derived from k8s-docs-en."""
    from tokenizers import Tokenizer
    rows, _ = export.effective_rows(rehearsal["root"] / "run" / "endpoint.npz")
    out = export.build_bundle(
        tmp_path / "attr", rows,
        Tokenizer.from_file(str(rehearsal["root"] / "data" / "tokenizer_ext.json")),
        {"vocabulary_sources": ["k8s-docs-en", "synthetic-general"], "rehearsal": True},
        reg, fallback_id=2, fixture=True)
    assert (out / export.ATTRIBUTION_NAME).exists()
    assert (out / export.ATTRIBUTION_NAME).read_text() == \
        (common.REPO / export.ATTRIBUTION_SRC).read_text()
    export.run_gates(out, log=lambda *a: None)
    (out / export.ATTRIBUTION_NAME).unlink()
    with pytest.raises(SystemExit, match="missing"):
        export.gate_files(out)


def test_fp32_snapshots_average_differently_from_their_fp16_truncations(rehearsal, reg,
                                                                        tmp_path):
    """FP16 rounding before folding can collapse genuinely different snapshots.

    Three snapshots differing by 1e-4 near 1.0 — below FP16's ~1e-3 spacing there — must
    average to something the FP16-only tables cannot reproduce.
    """
    import shutil as sh
    run = rehearsal["root"] / "run"
    src = run / "snapshot_000015.npz"
    z = dict(np.load(src))
    fp32, fp16 = [], []
    for k, s in enumerate((15, 18, 20)):
        rows = np.full(z["rows_fp16"].shape, 1.0, dtype=np.float32) + k * 1e-4
        for tag, rows_fp32 in (("fp32", rows), ("fp16", None)):
            d = tmp_path / tag
            d.mkdir(exist_ok=True)
            arrs = {**z, "rows_fp16": rows.astype(np.float16),
                    "token_weights": np.ones(rows.shape[0], dtype=np.float32)}
            if rows_fp32 is not None:
                arrs["rows_fp32"] = rows
            np.savez(d / f"snapshot_{s:06d}.npz", **arrs)
            sh.copy(run / f"snapshot_{s:06d}.meta.json", d / f"snapshot_{s:06d}.meta.json")
        fp32.append(tmp_path / "fp32" / f"snapshot_{s:06d}.npz")
        fp16.append(tmp_path / "fp16" / f"snapshot_{s:06d}.npz")
    a, _ = export.average_snapshots(fp32, reg)
    b, _ = export.average_snapshots(fp16, reg)
    assert np.allclose(a, 1.0 + 1e-4)                     # the true FP32 mean
    assert not np.allclose(a, b), "FP16 truncation collapsed three distinct snapshots"


def test_bundle_carries_the_frozen_encoder_spec(bundle):
    cfg = json.loads((bundle / "config.json").read_text())
    fz = common.freeze()["encoder_spec"]
    for f in export.SPEC_FIELDS:
        assert cfg["document_encoder"][f] == fz[f]
    prov = json.loads((bundle / "provenance.json").read_text())
    assert prov["bank"]["doc_ids_sha256"] and prov["bank"]["vector_bytes_sha256"]
    assert "corpus" not in prov, "a production corpus hash is deliberately NOT required"
