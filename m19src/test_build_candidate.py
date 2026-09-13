import json

import pytest

from m19src import build_candidate


def test_complete_teacher_snapshot_lock_refuses_mutation_and_extra_file(tmp_path, monkeypatch):
    snapshot = tmp_path / ("f" * 40)
    snapshot.mkdir()
    (snapshot / "modeling.py").write_text("trusted code\n")
    lock = {
        "_schema": "m19-teacher-snapshot-lock-v1", "snapshot_path": str(snapshot),
        "revision": snapshot.name,
        "files": {"modeling.py": build_candidate.sha_file_unchecked(snapshot / "modeling.py")},
    }
    lock_path = tmp_path / "lock.json"
    lock_path.write_text(json.dumps(lock))
    monkeypatch.setattr(build_candidate, "TEACHER_SNAPSHOT_LOCK", lock_path)
    monkeypatch.setattr(build_candidate, "load_json", lambda path: json.loads(path.read_text()))
    assert build_candidate.verify_teacher_snapshot(snapshot) == lock
    (snapshot / "modeling.py").write_text("changed\n")
    with pytest.raises(SystemExit, match="file differs"):
        build_candidate.verify_teacher_snapshot(snapshot)
    (snapshot / "modeling.py").write_text("trusted code\n")
    (snapshot / "unlocked.json").write_text("{}")
    with pytest.raises(SystemExit, match="file set differs"):
        build_candidate.verify_teacher_snapshot(snapshot)
