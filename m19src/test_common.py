import pytest

from m19src import common


def test_admits_only_named_inheritance_and_m19_outputs(tmp_path):
    assert common.admit_read(common.REPO / "results/m18_index_manifest.json").name.endswith(".json")
    with pytest.raises(common.ProtectedRead):
        common.admit_read(common.REPO / "results/m18_development_baselines.json")
    with pytest.raises(common.ProtectedRead):
        common.admit_read(common.M18_WORK / "derived/index/unregistered_payload.json")
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
    link = tmp_path / "guard-test-link"
    link.symlink_to(common.REPO / "README.md")
    with pytest.raises(common.ProtectedRead):
        common.admit_read(link)
    with pytest.raises(common.ProtectedWrite):
        common.admit_write(link)


def test_fresh_confirmation_requires_claimed_exact_bytes(tmp_path, monkeypatch):
    confirmation = tmp_path / "confirmation"
    confirmation.mkdir()
    query_file = confirmation / "queries.jsonl"
    query_file.write_text('{"query_id":"sealed"}\n')
    monkeypatch.setattr(common, "CONFIRMATION_WORK", confirmation)
    digest = common.sha_file_unchecked(query_file)
    with pytest.raises(common.ProtectedRead):
        common.admit_read(query_file)
    with pytest.raises(common.ProtectedRead):
        common.admit_confirmation_read(query_file, state="locked", claimed_files={str(query_file): digest})
    assert common.admit_confirmation_read(
        query_file, state="claimed", claimed_files={str(query_file.resolve()): digest}
    ) == query_file.resolve()
    query_file.write_text('{"query_id":"changed"}\n')
    with pytest.raises(common.ProtectedRead):
        common.admit_confirmation_read(
            query_file, state="claimed", claimed_files={str(query_file.resolve()): digest}
        )
