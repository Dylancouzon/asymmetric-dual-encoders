"""Shared M17 plumbing: paths, the registry, hashes and the execution guard.

Small on purpose. Everything here is used by at least two of `cache.py`, `vocab.py`,
`train.py`, `export.py`, `loader_np.py` and `evaluate.py`; nothing here is a framework.

The registry (`m17/registry.json`) is the single source of the constants. Modules read it
rather than restating numbers, so a locked constant cannot drift between the cache builder,
the driver and the exporter.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# Order matters and must be re-established even for entries Python already put on the path:
# run as a script, `m17src` is sys.path[0], and inserting `m7src` in front of it would make
# `import train` resolve to the LEGACY M7 driver.
for _p in (REPO, REPO / "m7src", REPO / "m17src"):
    if str(_p) in sys.path:
        sys.path.remove(str(_p))
    sys.path.insert(0, str(_p))

# Legacy teacher modules default to bge-base; M17 lives in stella's document space only
# (m17/CODEMAP.md, "Reuse hazards"). Set before anything imports `teacher`.
os.environ.setdefault("M7_ENCODER", "stella-400M-v5")

M17 = REPO / "m17"
REGISTRY_PATH = M17 / "registry.json"
FREEZE_PATH = REPO / "m7" / "FREEZE.json"
WORK = REPO / "work" / "m17"
RESULTS = REPO / "results"

EXECUTABLE_STATUSES = ("EXECUTABLE", "LOCKED_EXECUTABLE")


class NotExecutable(SystemExit):
    """The registry is still a draft and no `--rehearsal` flag was given."""


class ProtectedRead(SystemExit):
    """An M17 module tried to open a protected evaluation surface."""


class ProtectedWrite(SystemExit):
    """An M17 module tried to write outside its own output boundary."""


# Substrings that are never admitted, whatever the caller believes it is opening
# (CLAUDE.md, "Evidence and protocol"; m17/CODEMAP.md, last reuse hazard).
FORBIDDEN_READ_SUBSTRINGS = (
    "frozen_eval/untouched-",
    "m9reserve",
    "reserved_qrels",
    "lotte",
)
# The reserved four by dataset name. These names also spell legitimately admitted TRAINING
# material (`work/train/stores/fever-train.json`), so they are refused only where they can
# only mean the reserved evaluation surface: under `results/frozen_eval` or in a qrels path.
RESERVED_DATASET_NAMES = ("fever", "dbpedia", "cqadup-android", "cqadup-english")
RESERVED_NAME_CONTEXTS = ("results/frozen_eval", "qrels")


def admit_read(path):
    """Read admission. `p = common.admit_read(p)` before opening anything, everywhere.

    Resolves symlinks first: a benign-looking link into a protected cache must not pass
    because its own spelling is innocent. Returns the resolved `Path`.
    """
    try:
        rp = Path(path).resolve()
    except OSError:                                   # pragma: no cover - exotic filesystems
        rp = Path(path).absolute()
    s = rp.as_posix().lower()
    for bad in FORBIDDEN_READ_SUBSTRINGS:
        if bad in s:
            raise ProtectedRead(
                f"M17 READ REFUSED: {rp} resolves into protected content ({bad!r}). "
                "No six-set, reserved-four or LoTTE payload is an M17 development surface.")
    if any(ctx in s for ctx in RESERVED_NAME_CONTEXTS):
        for name in RESERVED_DATASET_NAMES:
            if name in s:
                raise ProtectedRead(
                    f"M17 READ REFUSED: {rp} names reserved set {name!r} inside an evaluation "
                    "path. Reserved qrels stay inside their registered transaction; admitted "
                    "training stores are unaffected.")
    return rp


# The three destinations no M17 writer may ever land on: the frozen comparator vectors, the
# frozen-eval caches and the M7 freeze itself (CLAUDE.md, "Evidence and protocol").
IMMUTABLE_WRITE_TARGETS = ("results/perquery.json", "results/frozen_eval/", "m7/freeze.json")


def admit_write(path):
    """Refuse the three immutable destinations. Returns the resolved path."""
    rp = Path(path).absolute()
    low = rp.as_posix().lower()
    for bad in IMMUTABLE_WRITE_TARGETS:
        if bad in low or low.endswith(bad.rstrip("/")):
            raise ProtectedWrite(
                f"M17 WRITE REFUSED: {rp} is immutable evidence ({bad}); its frozen vectors "
                "cannot be rebuilt from the remaining caches.")
    return rp


def load_json(path):
    return json.loads(admit_read(path).read_text())


def registry(path=None):
    return load_json(path or REGISTRY_PATH)


def freeze(path=None):
    return load_json(path or FREEZE_PATH)


def sha_bytes(b) -> str:
    return hashlib.sha256(b).hexdigest()


def sha_text(s: str) -> str:
    return sha_bytes(s.encode("utf-8", "surrogatepass"))


def sha_json(obj) -> str:
    return sha_text(json.dumps(obj, sort_keys=True, separators=(",", ":")))


def sha_file(path, chunk=1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def sha_texts(texts) -> str:
    """Order-sensitive hash of a text list; same construction as `m7src/teacher.sha_texts`."""
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8", "surrogatepass"))
        h.update(b"\x00")
    return h.hexdigest()


def sha_array(a) -> str:
    import numpy as np
    a = np.ascontiguousarray(a)
    return sha_bytes(a.tobytes() + str(a.dtype).encode() + str(a.shape).encode())


def require_executable(reg, rehearsal: bool, what="this run"):
    """The registry status gate. `--rehearsal` is the only bypass, and it is not silent.

    A rehearsal writes under `work/m17/rehearsal` with synthetic inputs; it may never be
    pointed at a development component, the M17 panel or any protected surface.
    """
    status = reg.get("status")
    if status in EXECUTABLE_STATUSES:
        return status
    if rehearsal:
        print(f"[m17] registry status {status!r}: {what} runs in REHEARSAL mode "
              "(synthetic fixtures, disposable outputs, no registered observation)")
        return status
    raise NotExecutable(
        f"M17 REFUSED: registry status is {status!r}, not one of {EXECUTABLE_STATUSES}. "
        "The lock (m17/STATUS.md step 6) must be committed before a real run. Pass "
        "--rehearsal for the synthetic end-to-end rehearsal.")


def write_json(path, obj, indent=1):
    path = admit_write(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=indent, sort_keys=True))
    tmp.replace(path)
    return path


def quantiles(values, qs=(1, 5, 25, 50, 75, 95, 99)):
    """`results/m8_b2_entropy.json`'s quantile block, same key spelling (`p1` .. `p99`)."""
    import numpy as np
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return {f"p{q}": None for q in qs}
    return {f"p{q}": float(np.percentile(v, q)) for q in qs}
