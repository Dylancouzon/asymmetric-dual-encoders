"""Small immutable execution lock for the realized M18 preparation artifacts."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from common import M18, REPO, admit_read, registry, sha_file, sha_json

LOCK = M18 / "execution-lock.json"
REQUIRED = {
    "source": REPO / "results/m18_source_manifest.json",
    "corpus": REPO / "results/m18_corpus_manifest.json",
    "protocol": REPO / "results/m18_protocol_manifest.json",
    "index": REPO / "results/m18_index_manifest.json",
    "prepare": REPO / "results/m18_prepare_manifest.json",
}


def recipe(reg):
    return {k: reg[k] for k in ("models", "versions", "split", "strata", "vocabulary",
                                 "variants", "training", "retrieval", "evaluation", "serving")}


def create(path=LOCK):
    reg = registry()
    if reg.get("status") != "EXECUTABLE_PREPARATION":
        raise SystemExit("M18 LOCK REFUSED: create the lock from EXECUTABLE_PREPARATION")
    missing = [str(p) for p in REQUIRED.values() if not p.exists()]
    if missing:
        raise SystemExit(f"M18 LOCK REFUSED: preparation artifacts absent: {missing}")
    artifacts = {name: {"path": str(p.relative_to(REPO)), "sha256": sha_file(p)}
                 for name, p in REQUIRED.items()}
    payload = {"_schema": "m18-execution-lock-v1", "state": "locked",
               "recipe_sha256": sha_json(recipe(reg)), "artifacts": artifacts}
    path = Path(path)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as e:
        raise SystemExit(f"M18 LOCK REFUSED: immutable lock exists at {path}") from e
    with os.fdopen(fd, "w") as f:
        json.dump(payload, f, indent=1, sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
    print(f"lock_sha256={sha_file(path)}")
    return payload


def verify(reg=None, path=LOCK):
    reg = reg or registry(); path = Path(path)
    if reg.get("status") != "LOCKED_EXECUTABLE" or not path.exists():
        raise SystemExit("M18 TRAINING REFUSED: execution lock/status is absent")
    if sha_file(path) != reg.get("execution_lock_sha256"):
        raise SystemExit("M18 TRAINING REFUSED: execution lock hash differs from registry")
    payload = json.loads(admit_read(path).read_text())
    if payload.get("state") != "locked" or payload.get("recipe_sha256") != sha_json(recipe(reg)):
        raise SystemExit("M18 TRAINING REFUSED: locked recipe changed")
    for rec in payload["artifacts"].values():
        if sha_file(REPO / rec["path"]) != rec["sha256"]:
            raise SystemExit(f"M18 TRAINING REFUSED: locked artifact changed: {rec['path']}")
    return payload


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    create()
