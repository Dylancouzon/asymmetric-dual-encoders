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


def _six_access_trail(name):
    sys.path.insert(0, str(REPO / "bench"))
    from core import _m7_access_trail   # appends to m7/SIX_ACCESS.log for the six M7 sets
    _m7_access_trail(name)


def load_public(dataset, with_corpus=True):
    """Corpus, test queries and qrels of one non-reserved dataset at the M20 pinned revisions.

    Pins come from the committed M20 zero-dense row, so E2/E4 score exactly what M20 scored.
    Missing queries follow M20: a query with qrels but no run scores 0.
    """
    from datasets import load_dataset
    refuse_reserved(dataset)
    row = json.loads((M20_SCORES / dataset / "zero-dense.json").read_text())
    _six_access_trail(dataset)
    source, revision = row["source"], row["revision"]
    split = {"dev": "validation"}.get(row["split"], row["split"])
    if dataset.startswith("cqadup-"):
        labels = load_dataset(source, "default", revision=revision, split=split)
    else:
        labels = load_dataset(row["qrels_source"], revision=row["qrels_revision"], split=split)
    qrels = {}
    for r in labels:
        qrels.setdefault(str(r["query-id"]), {})[str(r["corpus-id"])] = int(r["score"])
    queries = load_dataset(source, "queries", revision=revision)["queries"]
    q_ids, q_texts = [], []
    for qid, text in zip(queries["_id"], queries["text"]):
        if str(qid) in qrels:
            q_ids.append(str(qid))
            q_texts.append(text)
    if set(q_ids) != set(row["scores"]):
        raise RuntimeError(f"{dataset}: query set differs from the committed M20 row")
    out = {"q_ids": q_ids, "q_texts": q_texts, "qrels": {q: qrels[q] for q in q_ids},
           "pins": {k: row[k] for k in ("source", "revision", "qrels_source", "qrels_revision",
                                         "split")}}
    if with_corpus:
        corpus = load_dataset(source, "corpus", revision=revision)["corpus"]
        out["doc_ids"] = [str(x) for x in corpus["_id"]]
        out["doc_texts"] = [(f"{(r.get('title') or '').strip()} {(r.get('text') or '').strip()}"
                             .strip()) for r in corpus]
        if len(out["doc_ids"]) != row["n_docs"]:
            raise RuntimeError(f"{dataset}: {len(out['doc_ids'])} docs, M20 had {row['n_docs']}")
    return out
