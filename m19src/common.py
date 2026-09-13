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
CONFIRMATION_WORK = WORK / "confirmation"


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
    "m18_confirmation",
)

EXACT_INHERITED_FILES = {
    (REPO / "CLAUDE.md").resolve(),
    (REPO / "instructions-m19.md").resolve(),
    (REPO / "m11" / "release" / "zero_encoder.py").resolve(),
    (REPO / "m18" / "registry.json").resolve(),
    (REPO / "m18" / "execution-lock.json").resolve(),
    (RESULTS / "m18_source_manifest.json").resolve(),
    (RESULTS / "m18_corpus_manifest.json").resolve(),
    (RESULTS / "m18_index_manifest.json").resolve(),
    (RESULTS / "m18_system_manifest.json").resolve(),
}

EXACT_INHERITED_DATA = {
    (M18_WORK / "derived/corpus.jsonl").resolve(),
    (M18_WORK / "derived/index/corpus.jsonl").resolve(),
    (M18_WORK / "derived/index/doc_ids.json").resolve(),
    (M18_WORK / "derived/index/documents/vectors.f16.npy").resolve(),
    (M18_WORK / "derived/index/bm25/data.csc.index.npy").resolve(),
    (M18_WORK / "derived/index/bm25/indices.csc.index.npy").resolve(),
    (M18_WORK / "derived/index/bm25/indptr.csc.index.npy").resolve(),
    (M18_WORK / "derived/index/bm25/m18.json").resolve(),
    (M18_WORK / "derived/index/bm25/params.index.json").resolve(),
    (M18_WORK / "derived/index/bm25/vocab.index.json").resolve(),
    (RELEASE_BUNDLE / "model.npz").resolve(),
    (RELEASE_BUNDLE / "tokenizer.json").resolve(),
    (RELEASE_BUNDLE / "config.json").resolve(),
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


def _refuse_permanent(path: Path):
    low = path.as_posix().lower()
    if any(part in low for part in FORBIDDEN_PARTS):
        raise ProtectedRead(f"M19 READ REFUSED: {path} is protected or spent evaluation content")


def admit_read(path):
    """Resolve symlinks, reject protected spellings, then enforce the M19 allowlist."""
    rp = _resolved(path)
    _refuse_permanent(rp)
    if _under(rp, CONFIRMATION_WORK):
        raise ProtectedRead(
            f"M19 READ REFUSED: {rp} is sealed confirmation content; use the claimed "
            "confirmation transaction reader"
        )
    allowed = (
        _under(rp, M19)
        or _under(rp, REPO / "m19src")
        or _under(rp, WORK)
        or _m19_result(rp)
        or rp in EXACT_INHERITED_FILES
        or rp in EXACT_INHERITED_DATA
    )
    if not allowed:
        raise ProtectedRead(f"M19 READ REFUSED: {rp} is outside the explicit M19 input allowlist")
    return rp


def admit_confirmation_read(path, *, state, claimed_files):
    """Admit one exact fresh-confirmation file after a validated transaction claim.

    ``claimed_files`` is the lock-bound mapping of resolved path strings to SHA-256 values. The
    state machine supplies it only after verifying its claim/decision lock; this helper still
    checks namespace, state, exact membership and bytes before returning the path.
    """
    rp = _resolved(path)
    _refuse_permanent(rp)
    if _resolved(CONFIRMATION_WORK) != _resolved(WORK / "confirmation"):
        raise ProtectedRead("M19 CONFIRMATION READ REFUSED: confirmation root was redirected")
    if not _under(rp, CONFIRMATION_WORK):
        raise ProtectedRead(f"M19 CONFIRMATION READ REFUSED: {rp} is outside confirmation work")
    allowed_states = {
        "claimed", "pools-frozen", "judgments-in-progress", "qrels-frozen", "scored", "complete"
    }
    if state not in allowed_states:
        raise ProtectedRead(f"M19 CONFIRMATION READ REFUSED: state {state!r} is not claimed")
    expected = claimed_files.get(str(rp))
    if expected is None:
        raise ProtectedRead(f"M19 CONFIRMATION READ REFUSED: {rp} is absent from the claim")
    if not rp.is_file() or sha_file_unchecked(rp) != expected:
        raise ProtectedRead(f"M19 CONFIRMATION READ REFUSED: {rp} bytes differ from the claim")
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


def sha_texts(texts) -> str:
    """M18's order-sensitive identity for a sequence of strings."""
    h = hashlib.sha256()
    for value in texts:
        h.update(value.encode("utf-8", "surrogatepass"))
        h.update(b"\x00")
    return h.hexdigest()


def sha_array(array) -> str:
    import numpy as np
    value = np.ascontiguousarray(array)
    return sha_bytes(value.tobytes() + str(value.dtype).encode() + str(value.shape).encode())


def sha_file(path, chunk=1 << 22) -> str:
    return sha_file_unchecked(admit_read(path), chunk=chunk)


def sha_file_unchecked(path, chunk=1 << 22) -> str:
    """Hash a path already admitted by a transaction-specific boundary."""
    h = hashlib.sha256()
    with open(path, "rb") as handle:
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


def atomic_create_bytes(path, payload: bytes):
    """Publish new immutable bytes atomically without ever replacing an existing name."""
    path = admit_write(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-create-{os.getpid()}")
    try:
        with open(tmp, "xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(tmp, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if tmp.exists():
            tmp.unlink()
    return path


def write_json(path, obj, indent=2):
    payload = (json.dumps(obj, indent=indent, sort_keys=True) + "\n").encode()
    return atomic_write_bytes(path, payload)


def create_json(path, obj, indent=2):
    payload = (json.dumps(obj, indent=indent, sort_keys=True) + "\n").encode()
    return atomic_create_bytes(path, payload)
