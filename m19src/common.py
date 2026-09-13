"""M19 paths, hashes, atomic writes and strict protected-input boundaries.

Forked from ``m18src/common.py`` at M18 source commit bfa7257. M19 admits only its own state,
the explicit immutable inheritance inputs and the pinned redacted corpus/index. It additionally
refuses every M18 confirmation query/qrel spelling. Callers resolve a data path through
``admit_read`` or ``admit_write`` before opening it.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
M19 = REPO / "m19"
WORK = REPO / "work" / "m19"
RESULTS = REPO / "results"
REGISTRY_PATH = M19 / "registry.json"
RELEASE_BUNDLE = Path("/home/dylan/asymetric-dual-encoders/work/release/zero-v1")
M18_WORK = REPO / "work" / "m18"


class ProtectedRead(SystemExit):
    """A requested data path is outside M19's explicit read boundary."""


class ProtectedWrite(SystemExit):
    """A requested output path is outside M19-owned destinations."""


FORBIDDEN_PARTS = (
    "results/perquery.json",
    "results/frozen_eval/untouched-",
    "work/m9reserve",
    "reserved_qrels",
    "lotte",
    "confirmation_queries",
    "confirmation_qrels",
    "m18_confirmation",
)

EXACT_INHERITED_FILES = {
    (REPO / "m18" / "registry.json").resolve(),
    (REPO / "m18" / "execution-lock.json").resolve(),
    (RESULTS / "m18_source_manifest.json").resolve(),
    (RESULTS / "m18_corpus_manifest.json").resolve(),
    (RESULTS / "m18_index_manifest.json").resolve(),
    (RESULTS / "m18_system_manifest.json").resolve(),
}


def _resolved(path) -> Path:
    try:
        return Path(path).resolve()
    except OSError:  # pragma: no cover - exotic filesystems
        return Path(path).absolute()


def _under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _m19_result(path: Path) -> bool:
    return path.parent == RESULTS.resolve() and path.name.startswith("m19_")


def admit_read(path):
    """Resolve symlinks, reject protected spellings, then enforce the M19 allowlist."""
    rp = _resolved(path)
    low = rp.as_posix().lower()
    if any(part in low for part in FORBIDDEN_PARTS):
        raise ProtectedRead(f"M19 READ REFUSED: {rp} is protected or spent evaluation content")
    allowed = (
        _under(rp, M19)
        or _under(rp, REPO / "m19src")
        or _under(rp, WORK)
        or _m19_result(rp)
        or rp in EXACT_INHERITED_FILES
        or _under(rp, M18_WORK / "derived" / "index")
        or rp == (M18_WORK / "derived" / "corpus.jsonl").resolve()
        or _under(rp, RELEASE_BUNDLE)
    )
    if not allowed:
        raise ProtectedRead(f"M19 READ REFUSED: {rp} is outside the explicit M19 input allowlist")
    return rp


def admit_write(path):
    """Allow writes only to m19/, work/m19/, and results/m19_* (after resolution)."""
    rp = _resolved(path)
    allowed = _under(rp, M19) or _under(rp, WORK) or _m19_result(rp)
    if not allowed:
        raise ProtectedWrite(f"M19 WRITE REFUSED: {rp} is outside M19-owned output paths")
    return rp


def load_json(path):
    return json.loads(admit_read(path).read_text())


def registry():
    return load_json(REGISTRY_PATH)


def sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha_json(obj) -> str:
    return sha_bytes(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode())


def sha_file(path, chunk=1 << 22) -> str:
    h = hashlib.sha256()
    with open(admit_read(path), "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def atomic_write_bytes(path, payload: bytes):
    path = admit_write(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        with open(tmp, "xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return path


def write_json(path, obj, indent=2):
    payload = (json.dumps(obj, indent=indent, sort_keys=True) + "\n").encode()
    return atomic_write_bytes(path, payload)
