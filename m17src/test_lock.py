"""Step-6 lock checks on a synthetic prepared directory and a copy of the registry.

No test reads `results/`, a development component, the panel or a protected surface, and none
writes the real registry: the fixed manifests are tiny stand-ins under `tmp_path`.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

import lock
from common import registry, sha_json


def _fake_build(root: Path, screened: bool, status="DRAFT_NOT_EXECUTABLE", seed_hash="a",
                seed=0, size=None):
    hashes = {k: sha_json([k, seed_hash]) for k in lock.EXECUTED_HASHES + lock.INVARIANT_HASHES}
    (root / "ext").mkdir(parents=True)
    (root / "protected_receipt.json").write_text('{"admitted_documents": []}')
    receipt_sha = lock.sha_file(root / "protected_receipt.json")
    scr = ({"state": "complete", "screened": 10, "dropped": 1, "dropped_by_text": 1,
            "dropped_by_document_receipt": 0, "documents_screened": 3,
            "receipt": {"path": str(root / "protected_receipt.json"),   # the builder's shape
                        "sha256": receipt_sha}} if screened else
           {"state": "deferred_to_clock", "screened": 0, "dropped": 0, "receipt": None,
            "unscreened_new_source_rows": {"bank": 5, "pool": 2}})
    for form in ("ext", "base"):
        (root / form).mkdir(parents=True, exist_ok=True)
        man = {"hashes": dict(hashes), "queries": 600000, "seed": seed, "size": size, "form": form,
               "registry_status_at_build": status, "protected_screen": scr,
               "stages": {"vocab": {"selected": 3}}}
        if form == "base":
            man["hashes"]["tokenizer"] = "base-tok"
            man["hashes"]["new_rows"] = None
        (root / form / "prepared.json").write_text(json.dumps(man))
    (root / "build_record.json").write_text(json.dumps(
        {"complete_build": True, "wall_clock_kind": "full build", "wall_clock_seconds": 5400.0,
         "rss_high_water_gib": 11.0, "gpu_peak_gib": 3.0, "size": None,
         "memory_gib": {"RssAnon": 7.0}}))
    return root


@pytest.fixture
def fixed(tmp_path):
    files = {}
    for k in lock.FIXED_FILES:
        p = tmp_path / f"{k}.json"
        body = ({"status": "FINAL — sealed; cloud-software relevance is MODEL-JUDGED (ruling A6)",
                 "panel_jsonl": {"sha256": "d1b0" + "0" * 60}}       # the real manifest's shape
                if k == "panel_manifest" else {k: 1})
        p.write_text(json.dumps(body))
        files[k] = p
    return files


def test_pre_half_binds_protocol_allocation_and_flips_status(tmp_path, fixed):
    reg = copy.deepcopy(registry())
    assert reg["status"] == "DRAFT_NOT_EXECUTABLE"
    build = _fake_build(tmp_path / "full", screened=False)
    out = lock.lock_pre(reg, build, date="2026-09-11", fixed_files=fixed)
    lk = out["lock"]
    assert out["status"] == "EXECUTABLE"
    assert lk["protocol"]["decision_protocol_sha256"] == sha_json(reg["training"]["decision_protocol"])
    assert lk["panel_sha256"].startswith("d1b0")
    assert lk["measured_allocation"]["full_build_hours"] == 1.5
    assert lk["measured_allocation"]["phase_ceiling_hours"] == 3
    row = [r for r in out["allocation_hours"] if r["phase"] == lock.PREPARE_PHASE][0]
    assert row["hours"] == 3 and row["placeholder_hours_before_measurement"] == 16
    assert lk["allocation_hours_total"] + lk["allocation_unallocated_hours"] == 72
    assert set(lk["pre_screen_build"]["ext_hashes"]) == set(lock.EXECUTED_HASHES)
    # never twice, never on a screened build, never on a non-draft registry
    with pytest.raises(SystemExit, match="already exists|applies to"):
        lock.lock_pre(copy.deepcopy(out), build, fixed_files=fixed)
    reg2 = copy.deepcopy(registry())
    with pytest.raises(SystemExit, match="UNSCREENED"):
        lock.lock_pre(reg2, _fake_build(tmp_path / "scr", screened=True), fixed_files=fixed)
    # a complete --size build and a seed-7 build are not the full pool at the locked seed
    with pytest.raises(SystemExit, match="subsample"):
        lock.lock_pre(copy.deepcopy(registry()), _fake_build(tmp_path / "sz", False, size=50000),
                      fixed_files=fixed)
    with pytest.raises(SystemExit, match="seed 7"):
        lock.lock_pre(copy.deepcopy(registry()), _fake_build(tmp_path / "sd", False, seed=7),
                      fixed_files=fixed)


def test_pre_half_refuses_a_partial_rebuild_and_a_non_final_panel(tmp_path, fixed):
    build = _fake_build(tmp_path / "full", screened=False)
    rec = json.loads((build / "build_record.json").read_text())
    rec["wall_clock_kind"], rec["complete_build"] = "partial rebuild", False
    (build / "build_record.json").write_text(json.dumps(rec))
    with pytest.raises(SystemExit, match="complete full build"):
        lock.lock_pre(copy.deepcopy(registry()), build, fixed_files=fixed)
    rec["wall_clock_kind"], rec["complete_build"] = "full build", True
    (build / "build_record.json").write_text(json.dumps(rec))
    fixed["panel_manifest"].write_text(json.dumps({"status": "SEALED_PENDING",
                                                    "panel_jsonl": {"sha256": "x"}}))
    with pytest.raises(SystemExit, match="FINAL"):
        lock.lock_pre(copy.deepcopy(registry()), build, fixed_files=fixed)


def _fake_export(bundle_root):
    def export(build, out, reg, log=print):
        out = Path(out); out.mkdir(parents=True, exist_ok=True)
        (out / "provenance.json").write_text(json.dumps(
            {"model_npz_sha256": "m" * 64, "tokenizer_sha256": "t" * 64, "config_sha256": "c" * 64}))
        return out, {"gate_files": {"files": []}}
    return export


def test_executed_half_needs_the_screened_build_under_the_same_protocol(tmp_path, fixed):
    reg = lock.lock_pre(copy.deepcopy(registry()), _fake_build(tmp_path / "pre", screened=False),
                        date="2026-09-11", fixed_files=fixed)
    # not screened -> refused
    with pytest.raises(SystemExit, match="SCREENED"):
        lock.lock_executed(copy.deepcopy(reg), _fake_build(tmp_path / "u", screened=False,
                                                            status="EXECUTABLE"),
                           tmp_path / "v0", "2026-09-12T09:00:00", export=_fake_export(tmp_path), fixed_files=fixed)
    # screened but built under the draft status -> refused
    with pytest.raises(SystemExit, match="ran under status"):
        lock.lock_executed(copy.deepcopy(reg), _fake_build(tmp_path / "d", screened=True),
                           tmp_path / "v0", "2026-09-12T09:00:00", export=_fake_export(tmp_path), fixed_files=fixed)
    # an invariant input moved -> refused
    with pytest.raises(SystemExit, match="invariant build inputs changed"):
        lock.lock_executed(copy.deepcopy(reg), _fake_build(tmp_path / "m", screened=True,
                                                            status="EXECUTABLE", seed_hash="b"),
                           tmp_path / "v0", "2026-09-12T09:00:00", export=_fake_export(tmp_path), fixed_files=fixed)
    # the protocol moved since the pre half -> refused
    moved = copy.deepcopy(reg)
    moved["training"]["decision_protocol"]["eligibility"] += " (amended)"
    with pytest.raises(SystemExit, match="protocol/recipe identity moved"):
        lock.lock_executed(moved, _fake_build(tmp_path / "p", screened=True, status="EXECUTABLE"),
                           tmp_path / "v0", "2026-09-12T09:00:00", export=_fake_export(tmp_path), fixed_files=fixed)
    # an operative setting outside the decision_protocol block moved -> refused
    band = copy.deepcopy(reg)
    band["screening_preferences_not_release_bars"]["simple_arm_tie_band"] = 0
    with pytest.raises(SystemExit, match="protocol/recipe identity moved"):
        lock.lock_executed(band, _fake_build(tmp_path / "tb", screened=True, status="EXECUTABLE"),
                           tmp_path / "v0", "x", export=_fake_export(tmp_path), fixed_files=fixed)
    # a fixed manifest edited after the pre half -> refused
    fixed["support_manifest"].write_text('{"edited": true}')
    with pytest.raises(SystemExit, match="fixed manifests changed"):
        lock.lock_executed(copy.deepcopy(reg), _fake_build(tmp_path / "fm", screened=True,
                                                            status="EXECUTABLE"),
                           tmp_path / "v0", "x", export=_fake_export(tmp_path), fixed_files=fixed)
    fixed["support_manifest"].write_text(json.dumps({"support_manifest": 1}))
    # a receipt whose bytes do not hash to the manifest's record -> refused
    bad = _fake_build(tmp_path / "rc", screened=True, status="EXECUTABLE")
    (bad / "protected_receipt.json").write_text('{"admitted_documents": ["swapped"]}')
    with pytest.raises(SystemExit, match="manifest records"):
        lock.lock_executed(copy.deepcopy(reg), bad, tmp_path / "v0", "x",
                           export=_fake_export(tmp_path), fixed_files=fixed)
    # the real thing: screened, same inputs, executed under EXECUTABLE
    good = _fake_build(tmp_path / "g", screened=True, status="EXECUTABLE")
    ext = json.loads((good / "ext" / "prepared.json").read_text())
    ext["hashes"]["vocabulary"] = "v" * 64                       # the screen changed the list
    (good / "ext" / "prepared.json").write_text(json.dumps(ext))
    out = lock.lock_executed(copy.deepcopy(reg), good, tmp_path / "v0", "2026-09-12T09:00:00",
                             date="2026-09-12", export=_fake_export(tmp_path), fixed_files=fixed)
    ex = out["lock"]["executed"]
    assert out["status"] == "LOCKED_EXECUTABLE"
    assert ex["changed_since_pre_screen"] == ["vocabulary"]
    assert ex["base_hashes"]["tokenizer"] == "base-tok" and ex["base_hashes"]["new_rows"] is None
    assert ex["v0_export"]["model_npz_sha256"] == "m" * 64 and ex["v0_export"]["read"] is False
    assert ex["protected_screen"]["dropped"] == 1 and len(ex["protected_receipt_sha256"]) == 64
    with pytest.raises(SystemExit, match="already exists|needs the pre half"):
        lock.lock_executed(out, good, tmp_path / "v0", "x", export=_fake_export(tmp_path), fixed_files=fixed)


def test_measured_allocation_accepts_a_resumed_build_only_with_prior_seconds_for_the_gap(tmp_path):
    import prepare_data
    build = _fake_build(tmp_path / "b", screened=False)
    rec = json.loads((build / "build_record.json").read_text())
    rec.update({"complete_build": False, "wall_clock_kind": "partial rebuild",
                "stages_run_this_invocation": list(prepare_data.STAGES[3:]), "wall_clock_seconds": 5240.0})
    (build / "build_record.json").write_text(json.dumps(rec))
    with pytest.raises(SystemExit, match="did not run"):
        lock.measured_allocation(build)
    with pytest.raises(SystemExit, match="did not run"):          # the wrong gap
        lock.measured_allocation(build, {"seconds": 146.0, "stages": ["pool", "domain"], "source": "log"})
    a = lock.measured_allocation(build, {"seconds": 146.0, "stages": ["pool", "domain", "protected"],
                                         "source": "attempt-1 log"})
    assert a["full_build_wall_clock_seconds"] == 5386.0 and a["phase_ceiling_hours"] == 3
    assert a["summed_across_invocations"]["prior_stages"] == ["pool", "domain", "protected"]
    assert a["summed_across_invocations"]["prior_source"] == "attempt-1 log"


def test_measured_allocation_rounds_the_ceiling_up_and_never_below_one_hour(tmp_path):
    build = _fake_build(tmp_path / "b", screened=False)
    rec = json.loads((build / "build_record.json").read_text())
    for secs, want in ((5400.0, 3), (1800.0, 1), (3601.0, 3), (600.0, 1)):
        rec["wall_clock_seconds"] = secs
        (build / "build_record.json").write_text(json.dumps(rec))
        assert lock.measured_allocation(build)["phase_ceiling_hours"] == want


def test_training_needs_the_executed_half_and_the_locked_directory(tmp_path):
    import common
    import train
    reg = copy.deepcopy(registry())
    reg["status"] = "EXECUTABLE"
    assert common.require_executable(reg, rehearsal=False) == "EXECUTABLE"      # preparation
    with pytest.raises(SystemExit, match="real training needs 'LOCKED_EXECUTABLE'"):
        common.require_executable(reg, rehearsal=False, training=True)
    assert common.require_executable(reg, rehearsal=True, training=True) == "EXECUTABLE"
    reg["status"] = "LOCKED_EXECUTABLE"
    assert common.require_executable(reg, rehearsal=False, training=True) == "LOCKED_EXECUTABLE"
    man = {"form": "ext", "hashes": {k: "h" for k in lock.EXECUTED_HASHES}}
    with pytest.raises(SystemExit, match="no `lock.executed`"):
        train._check_executed_lock(man, reg, tmp_path)
    reg["lock"] = {"executed": {"ext_hashes": {k: "h" for k in lock.EXECUTED_HASHES}}}
    train._check_executed_lock(man, reg, tmp_path)
    man["hashes"]["cache_artifact"] = "other"
    with pytest.raises(SystemExit, match="cache_artifact"):
        train._check_executed_lock(man, reg, tmp_path)
    with pytest.raises(SystemExit, match="no `base_hashes`"):
        train._check_executed_lock({"form": "base", "hashes": {}}, reg, tmp_path)
