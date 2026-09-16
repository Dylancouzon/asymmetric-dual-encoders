#!/usr/bin/env python3
"""Corpus-only, resumable document pre-encode for M13's reserved four and M20's BEIR-15.

This entry point deliberately cannot read queries or qrels.  It claims the corpus-only G2
allowlist entry, authenticates each public corpus, and writes hash-recorded fp16 shards.  The
later reserved transaction re-hashes these shards before opening the one-shot payload and scores
only the systems registered in M10's final-run registry.

M20 (owner rulings R20/R22) extends it in exactly two ways and changes nothing about the encoding
contract:

  * The three "systems" here were always the three DOCUMENT TOWERS.  M20's eight-system roster adds
    `zero-dense` and `stella-query`, both of which REUSE the Stella shards written under
    ``nano-dense``; no new tower and no second corpus-scale Stella encode.  `bm25` and the two
    DBSF rows have no document vectors at all.  So this file's `SYSTEMS` table is unchanged.
  * ``--datasets`` selects the reserved four (the default) or a BEIR-15 corpus.  Reserved corpora
    are still authenticated against ``results/eval_manifest.json``.  A BEIR-15 corpus has no prior
    frozen hash, so it is pinned by HuggingFace revision and its computed hashes are recorded in
    ``results/m20_corpus_pins.json`` on first encode and re-verified on every later one.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _p in ("bench", "m7src", "m8src"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

import paths_guard  # noqa: E402

GUARD_ENTRY = "m8src.pre_encode"
GUARD_RECEIPT = paths_guard.claim(
    GUARD_ENTRY, note="M13 triggered reserved batch: public corpora only; no query or qrel access")
paths_guard.install()

RESERVED_DATASETS = ("fever", "dbpedia-entity", "cqadup-android", "cqadup-english")
DATASETS = RESERVED_DATASETS            # the historical name, still the default batch
REGISTRY = REPO / "m20" / "beir15_registry.json"
CORPUS_PINS = REPO / "results" / "m20_corpus_pins.json"
BEIR15_SUMMARY = REPO / "results" / "m20_beir15_preencode.json"
# The BEIR-15 corpora this file may encode, by cache name -> (hub repo, pinned revision).  Built
# from the pushed pre-observation registration so the executor cannot drift from it.  The reserved
# four are present because BEIR-15 reuses their shards; they are never re-encoded from here under
# a BEIR-15 invocation, and their rows come from the tagged transaction.
SYSTEMS = {
    "nano-dense": {
        "repo": "NovaSearch/stella_en_400M_v5",
        "revision": "ffeb2b7ee715c226d4ffe5e4619f7dbb48624c20",
        "backend": "stella",
        "dim": 1024,
        "doc_prefix": "",
    },
    "bge-small-en-v1.5": {
        "repo": "BAAI/bge-small-en-v1.5",
        "revision": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        "backend": "sentence-transformers",
        "dim": 384,
        "doc_prefix": "",
    },
    "leaf-ir-asym": {
        "repo": "Snowflake/snowflake-arctic-embed-m-v1.5",
        "revision": "e58a8f756156a1293d763f17e3aae643474e9b8a",
        "backend": "sentence-transformers",
        "dim": 768,
        "doc_prefix": "",
    },
}
SHARD_ROWS = 50_000
ROOT = REPO / "work" / "m13-reserved-enc"
SUMMARY = REPO / "results" / "m13_reserved_preencode.json"


def sha_file(path, block=8 << 20):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def write_atomic(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _json_list(path, values):
    """Write a large string list without materializing its JSON serialization."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as stream:
        stream.write("[")
        for i, value in enumerate(values):
            if i:
                stream.write(", ")
            stream.write(json.dumps(str(value)))
        stream.write("]\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def _json_list_digest(values):
    """SHA256 of the exact bytes `_json_list` writes, without creating another file."""
    h = hashlib.sha256()
    h.update(b"[")
    for i, value in enumerate(values):
        if i:
            h.update(b", ")
        h.update(json.dumps(str(value)).encode())
    h.update(b"]\n")
    return h.hexdigest()


def _doc_text(row):
    title = (row.get("title") or "").strip()
    text = (row.get("text") or "").strip()
    return f"{title} {text}".strip() if title else text


def corpus_sources():
    """cache name -> (hub repo, pinned revision), built from the pushed M20 registration.

    CQADupStack is one BEIR dataset of twelve forums, each its own `mteb/` repository, so it
    expands here into twelve cache names of the historical `cqadup-<forum>` shape.
    """
    registry = json.loads(REGISTRY.read_text())
    out = {}
    for row in registry["datasets"]:
        if row["key"] == "cqadupstack":
            continue
        out[row["key"]] = (row["source"], row["revision"])
    for forum, row in registry["cqadupstack"]["forums"].items():
        out[f"cqadup-{forum}"] = (row["source"], row["revision"])
    return out


def load_corpus(dataset):
    """Load only the named corpus configuration, at its pinned revision; the guard refuses every
    other config.  The revision pin is belt-and-braces for the reserved four, whose corpora are
    additionally hash-checked against the frozen manifest, and it is the ONLY identity a BEIR-15
    corpus has on its first encode."""
    from datasets import load_dataset

    paths_guard.ensure_loader_guard()
    sources = corpus_sources()
    if dataset not in sources:
        raise ValueError(f"{dataset!r} is not a registered M20 corpus")
    source, revision = sources[dataset]
    return load_dataset(source, "corpus", revision=revision)["corpus"], source, revision


def _corpus_identity(corpus):
    from hashing import sha_stream_list

    return {
        "n_docs": len(corpus),
        "corpus_ids_sha256": sha_stream_list(str(x) for x in corpus["_id"]),
        "corpus_text_sha256": sha_stream_list(_doc_text(row) for row in corpus),
    }


def authenticate_corpus(dataset, corpus, revision=None):
    """Reserved corpora must match the frozen manifest.  A BEIR-15 corpus has no prior frozen
    hash: it is pinned by revision, and its computed hashes are RECORDED on first encode and
    re-verified on every later one, so a silently republished dataset cannot slip through."""
    got = _corpus_identity(corpus)
    if dataset in RESERVED_DATASETS:
        expected = json.loads((REPO / "results" / "eval_manifest.json").read_text())[
            "m7_untouched_final"][dataset]
        bad = [f"{key}: {got[key]} != {expected.get(key)}" for key in got
               if got[key] != expected.get(key)]
        if bad:
            raise RuntimeError(f"{dataset}: public corpus differs from the frozen manifest: "
                               + "; ".join(bad))
        return got
    pins = json.loads(CORPUS_PINS.read_text()) if CORPUS_PINS.exists() else {}
    want = {**got, "hf_revision": revision}
    if dataset in pins:
        bad = [key for key in want if pins[dataset].get(key) != want[key]]
        if bad:
            raise RuntimeError(f"{dataset}: corpus changed since it was pinned on {bad}")
    else:
        pins[dataset] = want
        write_atomic(CORPUS_PINS, json.dumps(pins, indent=2, sort_keys=True) + "\n")
    return got


def identity(system, dataset, corpus_identity, source, revision=None):
    spec = SYSTEMS[system]
    return {
        "schema_version": 1,
        "system": system,
        "dataset": dataset,
        "source": source,
        "source_revision": revision,
        "model_id": spec["repo"],
        "revision": spec["revision"],
        "backend": spec["backend"],
        "dim": spec["dim"],
        "inference_dtype": "fp32",
        "storage_dtype": "fp16",
        "normalized": True,
        "doc_prefix": spec["doc_prefix"],
        "max_length": 512,
        "shard_rows": SHARD_ROWS,
        **corpus_identity,
    }


class Encoder:
    def __init__(self, system, device):
        self.system, self.device = system, device
        self.spec = SYSTEMS[system]
        self.model = None

    def encode(self, texts):
        import torch

        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False

        if self.spec["backend"] == "stella":
            import teacher

            values = teacher.encode(
                list(texts), prefix=self.spec["doc_prefix"], max_length=512,
                batch_tokens=32768, model_id=self.spec["repo"],
                revision=self.spec["revision"], dtype=torch.float32,
                device=self.device, verbose=True)
        else:
            if self.model is None:
                from sentence_transformers import SentenceTransformer

                self.model = SentenceTransformer(
                    self.spec["repo"], revision=self.spec["revision"], device=self.device,
                    model_kwargs={"dtype": torch.float32})
                self.model.max_seq_length = 512
            values = self.model.encode(
                [self.spec["doc_prefix"] + text for text in texts], batch_size=256,
                normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
        values = np.asarray(values, dtype=np.float32)
        if values.shape != (len(texts), self.spec["dim"]):
            raise RuntimeError(f"{self.system}: encoded shape {values.shape}, expected "
                               f"({len(texts)}, {self.spec['dim']})")
        if not np.isfinite(values).all():
            raise RuntimeError(f"{self.system}: non-finite document vector")
        norms = np.linalg.norm(values, axis=1)
        if np.max(np.abs(norms - 1.0), initial=0.0) > 2e-3:
            raise RuntimeError(f"{self.system}: document vectors are not unit-normalized")
        return values


def encode_dataset(encoder, dataset, projection=None):
    corpus, source, revision = load_corpus(dataset)
    corpus_identity = authenticate_corpus(dataset, corpus, revision)
    out = ROOT / encoder.system / dataset
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    want = identity(encoder.system, dataset, corpus_identity, source, revision)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        existing_identity = {key: manifest.get(key) for key in want}
        if existing_identity != want:
            raise RuntimeError(f"{encoder.system}/{dataset}: cache identity changed; preserve "
                               "the existing directory and inspect it")
    else:
        manifest = {**want, "status": "RUNNING", "shards": {},
                    "started_utc": time.time(), "guard": GUARD_RECEIPT}
        write_atomic(manifest_path, json.dumps(manifest, indent=2) + "\n")

    ids_path = ROOT / "corpora" / dataset / "doc_ids.json"
    ids_path.parent.mkdir(parents=True, exist_ok=True)
    expected_ids_file_sha = _json_list_digest(corpus["_id"])
    if not ids_path.exists():
        _json_list(ids_path, corpus["_id"])
    got_ids_file_sha = sha_file(ids_path)
    if got_ids_file_sha != expected_ids_file_sha:
        raise RuntimeError(f"{dataset}: shared document-id file does not encode the pinned corpus")
    if manifest.get("doc_ids_file_sha256", got_ids_file_sha) != got_ids_file_sha:
        raise RuntimeError(f"{dataset}: shared document-id file changed after this cache began")
    manifest["doc_ids_path"] = str(ids_path.relative_to(REPO))
    manifest["doc_ids_file_sha256"] = got_ids_file_sha

    n_shards = (len(corpus) + SHARD_ROWS - 1) // SHARD_ROWS
    for shard in range(n_shards):
        sid = f"{shard:05d}"
        path = out / f"shard_{sid}.npy"
        recorded = manifest["shards"].get(sid)
        if path.exists():
            if recorded is None:
                raise RuntimeError(f"{path}: an unrecorded shard exists; do not trust it")
            if sha_file(path) != recorded["sha256"]:
                raise RuntimeError(f"{path}: shard hash changed")
            continue
        lo, hi = shard * SHARD_ROWS, min((shard + 1) * SHARD_ROWS, len(corpus))
        texts = [_doc_text(row) for row in corpus.select(range(lo, hi))]
        started = time.monotonic()
        vecs = encoder.encode(texts).astype(np.float16)
        tmp = path.with_suffix(".tmp.npy")
        np.save(tmp, vecs)
        os.replace(tmp, path)
        seconds = time.monotonic() - started
        manifest["shards"][sid] = {
            "rows": hi - lo,
            "bytes": path.stat().st_size,
            "sha256": sha_file(path),
            "seconds": seconds,
        }
        write_atomic(manifest_path, json.dumps(manifest, indent=2) + "\n")
        print(f"[{encoder.system}/{dataset}] shard {shard + 1}/{n_shards} "
              f"({hi - lo:,} rows, {(hi - lo) / max(seconds, 1e-9):.0f}/s)", flush=True)
        if projection is not None:
            projection.observe(encoder.system, hi - lo, seconds)

    manifest.update(status="COMPLETE", n_shards=n_shards, completed_utc=time.time())
    write_atomic(manifest_path, json.dumps(manifest, indent=2) + "\n")
    return manifest_path


class Projection:
    """Stop a stage that is going to blow its registered cap, instead of discovering it at the cap.

    A cap alone does not protect a paid run: a job five times slower than planned spends the whole
    allowance before anyone sees it.  After each shard this re-projects the stage from the rate
    actually observed for this tower, and refuses at a shard boundary -- the only place where
    stopping costs nothing, because the shard just written is hash-recorded and resumable.
    """

    def __init__(self, remaining_docs, cap_hours, label, elapsed_hours=0.0, min_rows=100_000):
        self.remaining = dict(remaining_docs)          # system -> documents still to encode
        self.cap_hours = float(cap_hours)
        self.label = label
        self.elapsed_hours = float(elapsed_hours)
        self.min_rows = int(min_rows)
        self.rows, self.seconds = {}, {}

    def observe(self, system, rows, seconds):
        self.rows[system] = self.rows.get(system, 0) + int(rows)
        self.seconds[system] = self.seconds.get(system, 0.0) + float(seconds)
        self.remaining[system] = max(0, self.remaining.get(system, 0) - int(rows))
        self.elapsed_hours += float(seconds) / 3600.0
        if self.rows[system] < self.min_rows:
            return
        rate = self.rows[system] / max(self.seconds[system], 1e-9)
        projected = self.elapsed_hours + sum(
            self.remaining.get(name, 0) / (self.rows[name] / max(self.seconds[name], 1e-9))
            for name in self.remaining if self.rows.get(name)) / 3600.0
        unmeasured = [name for name in self.remaining
                      if self.remaining[name] and not self.rows.get(name)]
        if projected > self.cap_hours:
            raise RuntimeError(
                f"{self.label}: projected {projected:.1f} h exceeds the registered cap "
                f"{self.cap_hours:.1f} h at the measured rate ({system} {rate:.0f}/s). "
                f"Stopping at a shard boundary; every shard written so far is hash-recorded and "
                f"resumable. Towers not yet measured: {unmeasured or 'none'}.")


def _summary(datasets, summary_path):
    rows = {}
    for system in SYSTEMS:
        rows[system] = {}
        for dataset in datasets:
            path = ROOT / system / dataset / "manifest.json"
            if path.exists():
                record = json.loads(path.read_text())
                rows[system][dataset] = {
                    "status": record.get("status"),
                    "manifest": str(path.relative_to(REPO)),
                    "manifest_sha256": sha_file(path),
                    "n_docs": record.get("n_docs"),
                    "n_shards": record.get("n_shards"),
                    "source_revision": record.get("source_revision"),
                    "vector_bytes": sum(v.get("bytes", 0) for v in record.get("shards", {}).values()),
                }
    complete = all(rows.get(system, {}).get(dataset, {}).get("status") == "COMPLETE"
                   for system in SYSTEMS for dataset in datasets)
    blob = {"status": "COMPLETE" if complete else "RUNNING", "datasets": list(datasets),
            "systems": rows, "guard": GUARD_RECEIPT, "updated_utc": time.time()}
    write_atomic(summary_path, json.dumps(blob, indent=2) + "\n")
    return complete


def update_summary():
    """The reserved receipt the tagged transaction authenticates. Its shape is unchanged."""
    return _summary(RESERVED_DATASETS, SUMMARY)


def update_beir15_summary(datasets):
    return _summary(datasets, BEIR15_SUMMARY)


def beir15_encode_datasets():
    """BEIR-15 corpora that this stage encodes: every registered corpus except the reserved four,
    whose shards the tagged transaction produces and BEIR-15 reuses."""
    return tuple(name for name in corpus_sources() if name not in RESERVED_DATASETS)


def main(system, device, datasets=RESERVED_DATASETS, cap_hours=None, label="pre-encode"):
    if system not in SYSTEMS:
        raise ValueError(f"unknown system {system!r}")
    reserved_batch = tuple(datasets) == tuple(RESERVED_DATASETS)
    projection = None
    if cap_hours:
        pins = json.loads(CORPUS_PINS.read_text()) if CORPUS_PINS.exists() else {}
        frozen = json.loads((REPO / "results" / "eval_manifest.json").read_text())[
            "m7_untouched_final"]
        known = {name: (frozen[name]["n_docs"] if name in frozen
                        else pins.get(name, {}).get("n_docs", 0)) for name in datasets}
        projection = Projection({system: sum(known.values())}, cap_hours, label)
    encoder = Encoder(system, device)
    for dataset in datasets:
        encode_dataset(encoder, dataset, projection=projection)
        if reserved_batch:
            update_summary()
        else:
            update_beir15_summary(datasets)
    if reserved_batch:
        update_summary()
    else:
        update_beir15_summary(datasets)


def preflight(datasets=RESERVED_DATASETS):
    """Exercise the real guarded corpus route without hashing or encoding any passage."""
    rows = {}
    expected = json.loads((REPO / "results" / "eval_manifest.json").read_text())[
        "m7_untouched_final"]
    for dataset in datasets:
        corpus, source, revision = load_corpus(dataset)
        rows[dataset] = {"source": source, "revision": revision, "n_docs": len(corpus)}
        if dataset in expected:
            rows[dataset]["expected_n_docs"] = expected[dataset]["n_docs"]
            if rows[dataset]["n_docs"] != rows[dataset]["expected_n_docs"]:
                raise RuntimeError(f"{dataset}: corpus count changed during preflight")
    print(json.dumps({"status": "PASSED", "guard": GUARD_RECEIPT, "datasets": rows}))
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=tuple(SYSTEMS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--batch", choices=("reserved", "beir15"), default="reserved",
                        help="reserved: the registered reserved four. beir15: every other "
                             "registered BEIR-15 corpus; the reserved four are never re-encoded "
                             "here and their rows come from the tagged transaction.")
    parser.add_argument("--corpora", nargs="+", default=None,
                        help="restrict the batch to these corpus names; used for smoke runs and "
                             "for splitting a long stage across processes")
    parser.add_argument("--cap-hours", type=float, default=None,
                        help="registered stage cap; the projection gate stops at a shard boundary "
                             "if the measured rate cannot finish inside it")
    args = parser.parse_args()
    batch = RESERVED_DATASETS if args.batch == "reserved" else beir15_encode_datasets()
    if args.corpora:
        unknown = [name for name in args.corpora if name not in batch]
        if unknown:
            parser.error(f"not in the --batch {args.batch} corpus list: {unknown}")
        batch = tuple(args.corpora)
    if args.preflight_only:
        preflight(batch)
    elif args.system:
        main(args.system, args.device, batch, args.cap_hours, f"pre-encode/{args.batch}")
    else:
        parser.error("--system is required unless --preflight-only is used")
