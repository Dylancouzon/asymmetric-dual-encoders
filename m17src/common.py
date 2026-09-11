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
for _p in (REPO, REPO / "m7src", REPO / "m17src"):
    if str(_p) not in sys.path:
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


def load_json(path):
    return json.loads(Path(path).read_text())


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
    path = Path(path)
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
