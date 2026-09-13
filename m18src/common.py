"""Shared M18 plumbing: paths, hashes, atomic publication and execution guards.

Small on purpose. Everything here is used by at least two of `cache.py`, `vocab.py`,
`train.py`, `export.py`, `loader_np.py` and `evaluate.py`; nothing here is a framework.

Forked from ``m17src/common.py`` at commit 4213920. M18 deliberately has fresh paths/statuses,
forbids reads of every historical evaluation payload (M17 only guarded some writes), and offers
atomic byte/NPZ publication for the new pipeline. It never consults M17's closed registry/lock.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# Order matters and must be re-established even for entries Python already put on the path:
# run as a script, `m18src` is sys.path[0], and inserting `m7src` in front of it would make
# `import train` resolve to the LEGACY M7 driver.
def reassert_path_order():
    """Put `m18src` back in front of `m7src`. Call after importing any legacy module that
    inserts `m7src` at sys.path[0] (`m10src/protected10` does), or a later lazy `import train`
    resolves to the legacy driver (the 2026-09-11 screened full build died this way at parity)."""
    for _p in (REPO, REPO / "m7src", REPO / "m18src"):
        if str(_p) in sys.path:
            sys.path.remove(str(_p))
        sys.path.insert(0, str(_p))
    # A legacy `train` already cached in sys.modules would win over the path order (Astra 6b P2).
    cached = sys.modules.get("train")
    if cached is not None and str(getattr(cached, "__file__", "")).startswith(str(REPO / "m7src")):
        del sys.modules["train"]


reassert_path_order()

# Legacy teacher modules default to bge-base; M18 lives in stella's document space only
# (m18/CODEMAP.md, "Reuse hazards"). Set before anything imports `teacher`.
os.environ.setdefault("M7_ENCODER", "stella-400M-v5")

M18 = REPO / "m18"
REGISTRY_PATH = M18 / "registry.json"
FREEZE_PATH = REPO / "m7" / "FREEZE.json"
WORK = REPO / "work" / "m18"
RESULTS = REPO / "results"

EXECUTABLE_STATUSES = ("EXECUTABLE_PREPARATION", "LOCKED_EXECUTABLE")


class NotExecutable(SystemExit):
    """The registry is still a draft and no `--rehearsal` flag was given."""


class ProtectedRead(SystemExit):
    """An M18 module tried to open a protected evaluation surface."""


class ProtectedWrite(SystemExit):
    """An M18 module tried to write outside its own output boundary."""


# Substrings that are never admitted, whatever the caller believes it is opening
# (CLAUDE.md, "Evidence and protocol"; m18/CODEMAP.md, last reuse hazard).
FORBIDDEN_READ_SUBSTRINGS = (
    "results/perquery.json",
    "frozen_eval/untouched-",
    "m9reserve",
    "reserved_qrels",
    "lotte",
    "/m17/panel/",
    "results/m17_panel_",
    "results/m17_v0_read",
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
                f"M18 READ REFUSED: {rp} resolves into protected or historical evaluation "
                f"content ({bad!r}). No M7-M13 spent surface or M17 panel is an M18 input.")
    if any(ctx in s for ctx in RESERVED_NAME_CONTEXTS):
        for name in RESERVED_DATASET_NAMES:
            if name in s:
                raise ProtectedRead(
                    f"M18 READ REFUSED: {rp} names reserved set {name!r} inside an evaluation "
                    "path. Reserved qrels stay inside their registered transaction; admitted "
                    "training stores are unaffected.")
    return rp


# The three destinations no M18 writer may ever land on: the frozen comparator vectors, the
# frozen-eval caches and the M7 freeze itself (CLAUDE.md, "Evidence and protocol").
IMMUTABLE_WRITE_TARGETS = ("results/perquery.json", "results/frozen_eval/", "m7/freeze.json")


def admit_write(path):
    """Refuse the three immutable destinations. Returns the resolved path."""
    rp = Path(path).absolute()
    low = rp.as_posix().lower()
    for bad in IMMUTABLE_WRITE_TARGETS:
        if bad in low or low.endswith(bad.rstrip("/")):
            raise ProtectedWrite(
                f"M18 WRITE REFUSED: {rp} is immutable evidence ({bad}); its frozen vectors "
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
    with open(admit_read(path), "rb") as f:
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


def require_executable(reg, rehearsal: bool, what="this run", training=False):
    """The registry status gate. `--rehearsal` is the only bypass, and it is not silent.

    A rehearsal writes under `work/m18/rehearsal` with synthetic inputs; it may never be
    pointed at a development component, the M18 panel or any protected surface.

    `EXECUTABLE` (the pre-clock lock half) admits PREPARATION — the on-clock protected screen
    and rebuild. Real TRAINING needs `LOCKED_EXECUTABLE`: the executed identities and the V0
    hash committed (`m18src/lock.py --phase executed`; Astra lock review P1-2).
    """
    status = reg.get("status")
    if status in EXECUTABLE_STATUSES:
        if training and not rehearsal and status != EXECUTABLE_STATUSES[-1]:
            raise NotExecutable(
                f"M18 REFUSED: registry status is {status!r}; real training needs "
                f"{EXECUTABLE_STATUSES[-1]!r} (the executed lock half, m18src/lock.py). "
                f"{status!r} admits the on-clock preparation only.")
        if training and not rehearsal:
            import lock
            lock.verify(reg)
        return status
    if rehearsal:
        print(f"[m18] registry status {status!r}: {what} runs in REHEARSAL mode "
              "(synthetic fixtures, disposable outputs, no registered observation)")
        return status
    raise NotExecutable(
        f"M18 REFUSED: registry status is {status!r}, not one of {EXECUTABLE_STATUSES}. "
        "The lock (m18/STATUS.md step 6) must be committed before a real run. Pass "
        "--rehearsal for the synthetic end-to-end rehearsal.")


def write_json(path, obj, indent=1):
    path = admit_write(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=indent, sort_keys=True))
    tmp.replace(path)
    return path


def atomic_write_bytes(path, payload: bytes):
    """Publish one complete byte string beside its destination, then rename atomically."""
    path = admit_write(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        with open(tmp, "xb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return path


def atomic_save_npz(path, **arrays):
    """Write an NPZ as a complete transaction; NumPy otherwise writes live paths in-place."""
    import numpy as np
    path = admit_write(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}.npz")
    try:
        with open(tmp, "xb") as f:
            np.savez(f, **arrays)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return path


def quantiles(values, qs=(1, 5, 25, 50, 75, 95, 99)):
    """`results/m8_b2_entropy.json`'s quantile block, same key spelling (`p1` .. `p99`)."""
    import numpy as np
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return {f"p{q}": None for q in qs}
    return {f"p{q}": float(np.percentile(v, q)) for q in qs}
