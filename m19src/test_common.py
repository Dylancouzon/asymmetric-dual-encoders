from pathlib import Path

import pytest

from m19src import common


def test_admits_only_named_inheritance_and_m19_outputs(tmp_path):
    assert common.admit_read(common.REPO / "results/m18_index_manifest.json").name.endswith(".json")
    with pytest.raises(common.ProtectedRead):
        common.admit_read(common.REPO / "results/m18_development_baselines.json")
    assert common.admit_write(common.REPO / "results/m19_fixture.json").name == "m19_fixture.json"
    with pytest.raises(common.ProtectedWrite):
        common.admit_write(common.REPO / "results/m18_index_manifest.json")


@pytest.mark.parametrize("relative", [
    "results/perquery.json",
    "results/frozen_eval/untouched-fever.json",
    "work/m9reserve/qrels.json",
    "work/m18/derived/protocol/confirmation_queries.jsonl",
    "work/m18/derived/protocol/confirmation_qrels.json",
])
def test_refuses_protected_paths(relative):
    with pytest.raises(common.ProtectedRead):
        common.admit_read(common.REPO / relative)


def test_resolved_symlink_cannot_escape(tmp_path):
    link = common.WORK / "guard-test-link"
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(common.REPO / "CLAUDE.md")
    try:
        with pytest.raises(common.ProtectedRead):
            common.admit_read(link)
        with pytest.raises(common.ProtectedWrite):
            common.admit_write(link)
    finally:
        link.unlink()
