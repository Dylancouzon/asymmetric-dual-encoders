"""Recompute and bind the existing M19 teacher array to the complete snapshot lock."""
from __future__ import annotations

import json

import numpy as np

from m19src.build_candidate import (_encode_teachers, _json_bytes, _publish,
                                     TEACHER_ARRAY, TEACHER_RECEIPT,
                                     TEACHER_SNAPSHOT_LOCK, verify_teacher_snapshot)
from m19src.common import M19, RESULTS, admit_read, load_json, sha_array, sha_file, sha_json

OUT = RESULTS / "m19_teacher_snapshot_verification.json"


def verify():
    registry = load_json(M19 / "registry.json")
    roster = load_json(M19 / "term-roster-lock-v3.json")
    inheritance = load_json(M19 / "inheritance-lock.json")
    config = load_json(inheritance["inherited_data"]["zero_v1_config"]["path"])
    lock = verify_teacher_snapshot(load_json(TEACHER_RECEIPT)["runtime"]["snapshot_path"])
    terms = [row["term"] for row in roster["terms"]]
    regenerated, runtime = _encode_teachers(
        lock["snapshot_path"], terms, registry["candidate"]["query_prefix"],
        config["document_encoder"]["config_kwargs"], "cuda")
    stored = np.load(admit_read(TEACHER_ARRAY))
    if not np.array_equal(regenerated, stored):
        raise SystemExit("M19 TEACHER REFUSED: locked snapshot does not reproduce stored vectors")
    result = {
        "_schema": "m19-teacher-snapshot-verification-v1", "state": "complete",
        "snapshot_lock_sha256": sha_file(TEACHER_SNAPSHOT_LOCK),
        "snapshot_files_identity_sha256": sha_json(lock["files"]),
        "teacher_receipt_sha256": sha_file(TEACHER_RECEIPT),
        "teacher_array_file_sha256": sha_file(TEACHER_ARRAY),
        "teacher_vectors_identity_sha256": sha_array(stored),
        "regenerated_vectors_identity_sha256": sha_array(regenerated),
        "runtime": runtime,
    }
    _publish(OUT, _json_bytes(result))
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    verify()
