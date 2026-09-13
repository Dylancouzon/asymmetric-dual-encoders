import hashlib
import json
import tarfile
from pathlib import Path

import pytest

from m19src.portable_handoff import (archive_path, export_archive, load_manifest,
                                    selected_files, verify_files)


def _record(path: str, payload: bytes, source: str = "repo", archive=None):
    record = {
        "source": source,
        "path": path,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    if archive:
        record["archive_path"] = archive
    return record


def test_verify_and_export_explicit_payload(tmp_path: Path):
    repo = tmp_path / "repo"
    release = tmp_path / "release"
    (repo / "work/m18").mkdir(parents=True)
    release.mkdir()
    (repo / "work/m18/index.bin").write_bytes(b"index")
    (release / "model.bin").write_bytes(b"model")
    manifest = {
        "_schema": "m18-m19-portable-artifacts-v1",
        "profiles": {"system": {"files": ["index", "model"]}},
        "files": {
            "index": _record("work/m18/index.bin", b"index"),
            "model": _record("model.bin", b"model", "release",
                             "work/release/zero-v1/model.bin"),
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    loaded = load_manifest(manifest_path)
    verified = verify_files(selected_files(loaded, "system"), repo, release)
    output = tmp_path / "handoff.tar.gz"
    export_archive(verified, output)

    with tarfile.open(output, "r:gz") as archive:
        assert archive.getnames() == [
            "work/m18/index.bin",
            "work/release/zero-v1/model.bin",
        ]
        assert archive.extractfile("work/m18/index.bin").read() == b"index"


def test_refuses_bad_hash_and_unsafe_archive_path(tmp_path: Path):
    repo = tmp_path / "repo"
    (repo / "work/m18").mkdir(parents=True)
    (repo / "work/m18/file").write_bytes(b"expEcted")
    record = _record("work/m18/file", b"expected")
    with pytest.raises(SystemExit, match="sha256"):
        verify_files([("bad", record)], repo, None)
    with pytest.raises(ValueError, match="unsafe archive path"):
        archive_path({"path": "../protected"})
    with pytest.raises(ValueError, match="ignored work"):
        archive_path({"path": "results/private.json"})


def test_export_rechecks_bytes_and_never_publishes_changed_input(tmp_path: Path):
    source = tmp_path / "input"
    source.write_bytes(b"expEcted")
    record = _record("work/m18/file", b"expected")
    output = tmp_path / "handoff.tar.gz"

    with pytest.raises(SystemExit, match="changed during export"):
        export_archive([("changed", record, source)], output)
    assert not output.exists()
