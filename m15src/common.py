"""Shared M15 helpers: the reserved-name guard, receipts and no-overwrite result writes.

Method: `m15/MEASUREMENTS.md`. Every M15 script imports `refuse_reserved` before it opens data.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SEED = 20260930
RESERVED = {"fever", "dbpedia-entity", "cqadup-android", "cqadup-english",
            "cqadupstack-android", "cqadupstack-english"}
# BEIR-15 minus FEVER, DBpedia-entity and CQADupStack (E5, E6 evaluation).
EVAL12 = ("scifact", "nfcorpus", "scidocs", "trec-covid", "fiqa", "arguana", "msmarco", "nq",
          "hotpotqa", "webis-touche2020", "quora", "climate-fever")
FIT_FORUMS = ("cqadup-physics", "cqadup-programmers")   # M7 dev forums, E6 fit only
M20_SCORES = REPO / "results" / "m20_beir15_scores"


def refuse_reserved(name):
    if name.lower() in RESERVED or "android" in name.lower() or "english" in name.lower():
        raise SystemExit(f"refusing reserved dataset {name!r}")
    return name


def sha_file(path, block=8 << 20):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def m20_rows(dataset, system):
    """Committed per-query nDCG@10 rows for one non-reserved dataset and system."""
    refuse_reserved(dataset)
    path = M20_SCORES / dataset / f"{system.replace(' ', '_')}.json"
    blob = json.loads(path.read_text())
    if blob["status"] != "COMPLETE" or blob["dataset"] != dataset:
        raise RuntimeError(f"{path}: not a complete row for {dataset}")
    return blob["scores"], {"path": str(path.relative_to(REPO)), "sha256": sha_file(path)}


def _versions(packages):
    out = {}
    for p in packages:
        try:
            out[p] = version(p)
        except PackageNotFoundError:
            out[p] = None
    return out


def receipt(script, inputs, started, extra_packages=()):
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True)
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "m15src"], cwd=REPO,
                           capture_output=True, text=True).stdout.strip()
    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                          text=True).stdout.strip()
    return {"script": str(Path(script).resolve().relative_to(REPO)),
            "script_sha256": sha_file(script), "git_commit": git.stdout.strip(),
            "m15src_dirty": bool(dirty), "started_utc": started,
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "machine": {"chip": chip, "platform": platform.platform(),
                        "python": sys.version.split()[0]},
            "packages": _versions(("numpy",) + tuple(extra_packages)),
            "seed": SEED, "inputs": inputs, "method": "m15/MEASUREMENTS.md",
            "method_sha256": sha_file(REPO / "m15" / "MEASUREMENTS.md")}


def utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_result(path, blob):
    path = Path(path)
    if path.exists():
        raise SystemExit(f"{path} exists; M15 results are never overwritten")
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(blob, indent=2) + "\n")
    tmp.replace(path)
    print(f"wrote {path.relative_to(REPO)}", flush=True)


def bootstrap_mean(values, n=10_000, seed=SEED):
    """Query-resampling 95% interval of the mean of a per-query array."""
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n, len(values)))
    means = values[idx].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]
