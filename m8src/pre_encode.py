#!/usr/bin/env python3
"""Corpus-only, resumable document pre-encode for M13's reserved four.

This entry point deliberately cannot read queries or qrels.  It claims the corpus-only G2
allowlist entry, authenticates each public corpus against ``results/eval_manifest.json``, and
writes hash-recorded fp16 shards.  The later reserved transaction re-hashes these shards before
opening the one-shot payload and scores only the systems registered in M10's final-run registry.
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

DATASETS = ("fever", "dbpedia-entity", "cqadup-android", "cqadup-english")
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


def load_corpus(dataset):
    """Load only the named corpus configuration; the guard refuses every other config."""
    from datasets import load_dataset

    paths_guard.ensure_loader_guard()
    if dataset.startswith("cqadup-"):
        source = f"mteb/cqadupstack-{dataset.split('-', 1)[1]}"
    else:
        source = f"BeIR/{dataset}"
    return load_dataset(source, "corpus")["corpus"], source


def authenticate_corpus(dataset, corpus):
    from hashing import sha_stream_list

    expected = json.loads((REPO / "results" / "eval_manifest.json").read_text())[
        "m7_untouched_final"][dataset]
    got = {
        "n_docs": len(corpus),
        "corpus_ids_sha256": sha_stream_list(str(x) for x in corpus["_id"]),
        "corpus_text_sha256": sha_stream_list(_doc_text(row) for row in corpus),
    }
    bad = [f"{key}: {got[key]} != {expected.get(key)}" for key in got
           if got[key] != expected.get(key)]
    if bad:
        raise RuntimeError(f"{dataset}: public corpus differs from the frozen manifest: "
                           + "; ".join(bad))
    return got


def identity(system, dataset, corpus_identity, source):
    spec = SYSTEMS[system]
    return {
        "schema_version": 1,
        "system": system,
        "dataset": dataset,
        "source": source,
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


def encode_dataset(encoder, dataset):
    corpus, source = load_corpus(dataset)
    corpus_identity = authenticate_corpus(dataset, corpus)
    out = ROOT / encoder.system / dataset
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    want = identity(encoder.system, dataset, corpus_identity, source)
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
        manifest["shards"][sid] = {
            "rows": hi - lo,
            "bytes": path.stat().st_size,
            "sha256": sha_file(path),
            "seconds": time.monotonic() - started,
        }
        write_atomic(manifest_path, json.dumps(manifest, indent=2) + "\n")
        print(f"[{encoder.system}/{dataset}] shard {shard + 1}/{n_shards} "
              f"({hi - lo:,} rows)", flush=True)

    manifest.update(status="COMPLETE", n_shards=n_shards, completed_utc=time.time())
    write_atomic(manifest_path, json.dumps(manifest, indent=2) + "\n")
    return manifest_path


def update_summary():
    rows = {}
    for system in SYSTEMS:
        rows[system] = {}
        for dataset in DATASETS:
            path = ROOT / system / dataset / "manifest.json"
            if path.exists():
                record = json.loads(path.read_text())
                rows[system][dataset] = {
                    "status": record.get("status"),
                    "manifest": str(path.relative_to(REPO)),
                    "manifest_sha256": sha_file(path),
                    "n_docs": record.get("n_docs"),
                    "n_shards": record.get("n_shards"),
                    "vector_bytes": sum(v.get("bytes", 0) for v in record.get("shards", {}).values()),
                }
    complete = all(rows.get(system, {}).get(dataset, {}).get("status") == "COMPLETE"
                   for system in SYSTEMS for dataset in DATASETS)
    blob = {"status": "COMPLETE" if complete else "RUNNING", "systems": rows,
            "guard": GUARD_RECEIPT, "updated_utc": time.time()}
    write_atomic(SUMMARY, json.dumps(blob, indent=2) + "\n")
    return complete


def main(system, device):
    if system not in SYSTEMS:
        raise ValueError(f"unknown system {system!r}")
    encoder = Encoder(system, device)
    for dataset in DATASETS:
        encode_dataset(encoder, dataset)
        update_summary()
    update_summary()


def preflight():
    """Exercise the real guarded corpus route without hashing or encoding any passage."""
    rows = {}
    expected = json.loads((REPO / "results" / "eval_manifest.json").read_text())[
        "m7_untouched_final"]
    for dataset in DATASETS:
        corpus, source = load_corpus(dataset)
        rows[dataset] = {"source": source, "n_docs": len(corpus),
                         "expected_n_docs": expected[dataset]["n_docs"]}
        if rows[dataset]["n_docs"] != rows[dataset]["expected_n_docs"]:
            raise RuntimeError(f"{dataset}: corpus count changed during preflight")
    print(json.dumps({"status": "PASSED", "guard": GUARD_RECEIPT, "datasets": rows}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=tuple(SYSTEMS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.preflight_only:
        preflight()
    elif args.system:
        main(args.system, args.device)
    else:
        parser.error("--system is required unless --preflight-only is used")
