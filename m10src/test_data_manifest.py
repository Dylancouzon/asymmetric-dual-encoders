"""The provenance file the recipe lock cites. Its failure mode is being INCOMPLETE while looking
complete, so that is what these test.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import data_manifest as DM


def test_the_manifest_on_disk_is_complete_and_hashes_real_bytes():
    m = json.loads(DM.OUT.read_text())
    assert m["complete"] is True and m["holes"] == []
    for sect in ("corpus", "generated_by_form"):
        for name, e in m[sect].items():
            assert not e.get("MISSING"), f"{sect}/{name}"
            assert len(e["sha256"]) == 64 and e["bytes"] > 0, name
    for name, e in m["cited_measurements"].items():
        assert not e.get("MISSING"), name
        assert len(e["sha256"]) == 64


def test_it_hashes_the_RAW_pools_not_only_the_draw():
    """21,087,043 rows were harvested and 1,250,000 drawn. Hashing only the draw proves what was
    used but not what it was drawn FROM, which is half of what provenance means."""
    m = json.loads(DM.OUT.read_text())
    for k in ("harvest_pool_wikipedia", "harvest_pool_arxiv", "harvest_pool_licensed"):
        assert k in m["corpus"] and m["corpus"][k]["bytes"] > 0
    assert m["corpus"]["harvest_pool_wikipedia"]["bytes"] > m["corpus"]["harvest_train"]["bytes"]


def test_a_hole_is_a_REFUSAL_not_a_missing_key(monkeypatch, tmp_path):
    """A provenance file whose gaps are invisible is worse than none: the lock cites it as whole."""
    monkeypatch.setitem(DM.CORPUS, "invented", "work/m10harvest/does-not-exist.jsonl")
    monkeypatch.setattr(DM, "OUT", tmp_path / "m.json")
    with pytest.raises(SystemExit, match="REFUSED"):
        DM.build(verbose=False)
    written = json.loads((tmp_path / "m.json").read_text())
    assert written["complete"] is False and "corpus/invented" in written["holes"], \
        "the refusal must still leave the hole ON THE RECORD, not vanish"


def test_the_disclosures_the_lock_owes_are_present_and_say_NOT_RUN_plainly():
    """W11's own-source 5-gram screen and the unexecuted CUREv1 admission. A disclosure that
    hedges is not a disclosure."""
    d = json.loads(DM.OUT.read_text())["disclosures"]
    assert "NOT RUN" in d["own_source_5gram_screen"]
    assert "results/m10_rescreen10.json" in d["own_source_5gram_screen"], \
        "it must name the screen that DID run, or the disclosure reads as no screening at all"
    assert "NEVER EXECUTED" in d["cure_v1"] and "re-screen" in d["cure_v1"]
    assert "834,463" in d["generated_count"]


def test_it_does_not_restate_a_number_it_could_read():
    """`_not_a_measurement`: if a figure appears here and in a cited artifact, the artifact wins."""
    m = json.loads(DM.OUT.read_text())
    reg = json.loads((DM.REPO / "m10" / "screen_registry.json").read_text())
    asm = json.loads((DM.RESULTS / "m10_assemble10.json").read_text())
    assert m["provenance"]["unique_text_count"] == reg["data_cut"]["unique_text_count"]
    assert m["provenance"]["generated_rows_realized"] == asm["total_rows"]
