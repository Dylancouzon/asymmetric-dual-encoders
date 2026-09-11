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
    with pytest.raises(SystemExit, match="snapshots at steps"):
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


def test_gate_artifact_catches_a_stale_staging_dir(bundle):
    z = dict(np.load(bundle / "model.npz"))
    z["rows_int8"] = np.zeros_like(z["rows_int8"])
    np.savez(bundle / "model.npz", **z)
    with pytest.raises(SystemExit, match="staged model.npz hashes"):
        export.gate_artifact(bundle)


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


def test_bundle_carries_the_frozen_encoder_spec(bundle):
    cfg = json.loads((bundle / "config.json").read_text())
    fz = common.freeze()["encoder_spec"]
    for f in export.SPEC_FIELDS:
        assert cfg["document_encoder"][f] == fz[f]
    prov = json.loads((bundle / "provenance.json").read_text())
    assert prov["bank"]["doc_ids_sha256"] and prov["bank"]["vector_bytes_sha256"]
    assert "corpus" not in prov, "a production corpus hash is deliberately NOT required"
