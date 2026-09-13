"""Build and query the local M18 Stella/BM25/DBSF project-memory system."""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

from common import (REPO, WORK, admit_read, atomic_write_bytes, registry, sha_array, sha_file,
                    sha_json, sha_texts, write_json)
from evaluate import BM25Index, dbsf_at, search


def _teacher_module():
    p = str(REPO / "m7src")
    if p not in sys.path:
        sys.path.insert(0, p)
    import teacher
    return teacher


def document_text(row):
    metadata = " ".join(str(x) for x in (row.get("path"), row.get("symbol"), row.get("title"))
                        if x)
    return (metadata + "\n" + row["text"]).strip()


def read_corpus(path):
    rows = []
    with open(admit_read(path), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def encode_stella_sharded(name, texts, out_dir, prefix="", device="cuda", shard_size=4096,
                           batch_tokens=16384, verbose=True):
    """Fresh M18 sharded encode with immutable shard hashes and resumable receipts."""
    out = Path(out_dir) / name
    out.mkdir(parents=True, exist_ok=True)
    reg = registry()
    spec = reg["models"]["teacher"]
    identity = {"texts_sha256": sha_texts(texts), "n": len(texts), "prefix": prefix,
                "model": spec["repo"], "revision": spec["revision"], "dim": spec["dim"],
                "max_length": spec["max_length"], "store_dtype": "fp16",
                "shard_size": int(shard_size)}
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(admit_read(manifest_path).read_text())
        if manifest.get("identity_sha256") != sha_json(identity):
            raise SystemExit(f"M18 ENCODE REFUSED: {out} belongs to another input identity")
    else:
        manifest = {"_schema": "m18-stella-encode-v1", "identity": identity,
                    "identity_sha256": sha_json(identity), "shards": {}, "complete": False}
        write_json(manifest_path, manifest)
    started = time.time()
    expected_ids = {f"{lo // shard_size:06d}" for lo in range(0, len(texts), shard_size)}
    if set(manifest.get("shards", {})) - expected_ids:
        raise SystemExit(f"M18 ENCODE REFUSED: unexpected shards in {out}")
    teacher = None
    for lo in range(0, len(texts), shard_size):
        hi = min(len(texts), lo + shard_size)
        sid = f"{lo // shard_size:06d}"
        path = out / f"shard_{sid}.npy"
        rec = manifest["shards"].get(sid)
        if path.exists() and rec:
            arr = np.load(admit_read(path), mmap_mode="r")
            valid_meta = (rec.get("lo") == lo and rec.get("hi") == hi
                          and rec.get("shape") == [hi - lo, int(spec["dim"])]
                          and arr.shape == (hi - lo, int(spec["dim"])) and arr.dtype == np.float16)
            if (not valid_meta or path.stat().st_size != rec["bytes"]
                    or sha_file(path) != rec["sha256"] or not np.isfinite(arr).all()
                    or (len(arr) and float(np.max(np.abs(np.linalg.norm(arr.astype(np.float32), axis=1) - 1))) > 0.01)):
                raise SystemExit(f"M18 ENCODE REFUSED: existing shard {path} failed its receipt")
            continue
        if path.exists() != bool(rec):
            raise SystemExit(f"M18 ENCODE REFUSED: partial shard transaction {path}")
        if verbose:
            print(f"[{name}] encode {lo:,}:{hi:,}/{len(texts):,}", flush=True)
        teacher = teacher or _teacher_module()
        vecs = teacher.encode(texts[lo:hi], prefix=prefix, max_length=int(spec["max_length"]),
                              batch_tokens=batch_tokens, model_id=spec["repo"],
                              revision=spec["revision"], dtype=torch.float16, device=device,
                              verbose=verbose).astype(np.float16)
        tmp = path.with_name(path.name + f".tmp-{os.getpid()}.npy")
        try:
            np.save(tmp, vecs)
            os.replace(tmp, path)
        finally:
            if tmp.exists():
                tmp.unlink()
        manifest["shards"][sid] = {"lo": lo, "hi": hi, "shape": list(vecs.shape),
                                    "bytes": path.stat().st_size, "sha256": sha_file(path)}
        write_json(manifest_path, manifest)
    combined = out / "vectors.f16.npy"
    if combined.exists():
        rec = manifest.get("combined")
        arr = np.load(admit_read(combined), mmap_mode="r")
        if (not manifest.get("complete") or not rec or rec.get("path") != combined.name
                or rec.get("bytes") != combined.stat().st_size
                or rec.get("sha256") != sha_file(combined)
                or arr.shape != (len(texts), int(spec["dim"])) or arr.dtype != np.float16
                or not np.isfinite(arr).all()
                or (len(arr) and float(np.max(np.abs(np.linalg.norm(arr.astype(np.float32), axis=1) - 1))) > 0.01)):
            raise SystemExit(f"M18 ENCODE REFUSED: combined vectors failed receipt in {out}")
    if not combined.exists():
        tmp = out / f"vectors.f16.tmp-{os.getpid()}.npy"
        mm = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16,
                                      shape=(len(texts), int(spec["dim"])))
        for rec in sorted(manifest["shards"].values(), key=lambda x: x["lo"]):
            mm[rec["lo"]:rec["hi"]] = np.load(out / f"shard_{rec['lo']//shard_size:06d}.npy",
                                                mmap_mode="r")
        mm.flush(); del mm
        os.replace(tmp, combined)
    manifest.update({"complete": True, "combined": {"path": combined.name,
                     "bytes": combined.stat().st_size, "sha256": sha_file(combined)},
                     "elapsed_seconds_this_invocation": round(time.time() - started, 3),
                     "rss_high_water_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                     "peak_vram_bytes": (int(torch.cuda.max_memory_allocated(device))
                                           if str(device).startswith("cuda") and torch.cuda.is_available()
                                           else 0)})
    write_json(manifest_path, manifest)
    return np.load(combined, mmap_mode="r"), manifest


def build_index(corpus_path=None, out_root=None, device="cuda", shard_size=4096,
                batch_tokens=16384):
    reg = registry()
    corpus_path = Path(corpus_path or WORK / "derived" / "corpus.jsonl")
    out = Path(out_root or WORK / "derived" / "index")
    out.mkdir(parents=True, exist_ok=True)
    all_rows = read_corpus(corpus_path)
    rows = [r for r in all_rows if r.get("indexable", True)]
    ids = [r["doc_id"] for r in rows]
    texts = [document_text(r) for r in rows]
    vectors, enc = encode_stella_sharded("documents", texts, out, prefix="", device=device,
                                          shard_size=shard_size, batch_tokens=batch_tokens)
    ids_path = out / "doc_ids.json"
    if not ids_path.exists():
        atomic_write_bytes(ids_path, (json.dumps(ids, ensure_ascii=False) + "\n").encode())
    elif json.loads(admit_read(ids_path).read_text()) != ids:
        raise SystemExit("M18 INDEX REFUSED: existing document ids differ from corpus")
    corpus_copy = out / "corpus.jsonl"
    corpus_bytes = admit_read(corpus_path).read_bytes()
    if corpus_copy.exists():
        if corpus_copy.read_bytes() != corpus_bytes:
            raise SystemExit("M18 INDEX REFUSED: embedded corpus differs from input")
    else:
        atomic_write_bytes(corpus_copy, corpus_bytes)
    bm_path = out / "bm25"
    if bm_path.exists():
        bm = BM25Index.load(bm_path)
        if bm.doc_ids != ids:
            raise SystemExit("M18 INDEX REFUSED: BM25 ids differ from corpus")
    else:
        BM25Index(ids, texts, **{k: reg["retrieval"]["bm25"][k] for k in ("k1", "b")}).save(bm_path)
    manifest = {"_schema": "m18-project-memory-index-v1", "documents": len(ids),
                "corpus_sha256": sha_file(corpus_path), "doc_ids_sha256": sha_texts(ids),
                "indexed_text_sha256": sha_texts(texts), "stella": enc,
                "bm25": reg["retrieval"]["bm25"], "fusion": reg["retrieval"]["fusion"],
                "vector_shape": list(vectors.shape), "vector_dtype": str(vectors.dtype)}
    manifest["sha256"] = sha_json(manifest)
    write_json(out / "index_manifest.json", manifest)
    if out_root is None:
        write_json(REPO / "results/m18_index_manifest.json", manifest)
    return manifest


def encode_teacher_queries(texts, out_root, name="development", device="cuda"):
    spec = registry()["models"]["teacher"]
    return encode_stella_sharded(name, texts, out_root, prefix=spec["query_prefix"],
                                  device=device, shard_size=1024, batch_tokens=16384)


def _query_encoder(bundle):
    cfg = json.loads(admit_read(Path(bundle) / "config.json").read_text())
    if "query_preprocessing" in cfg:
        from loader_np import M18QueryEncoder
        return M18QueryEncoder(bundle, variant=cfg["variant"])
    p = str(REPO / "m11" / "release")
    if p not in sys.path:
        sys.path.insert(0, p)
    from zero_encoder import ZeroQueryEncoder
    return ZeroQueryEncoder(bundle, variant="int8")


class ProjectMemory:
    def __init__(self, index_root, bundle):
        self.root = Path(index_root)
        self.ids = json.loads(admit_read(self.root / "doc_ids.json").read_text())
        self.vectors = np.load(admit_read(self.root / "documents/vectors.f16.npy"), mmap_mode="r")
        self.bm25 = BM25Index.load(self.root / "bm25")
        manifest = json.loads(admit_read(self.root / "index_manifest.json").read_text())
        if (manifest.get("doc_ids_sha256") != sha_texts(self.ids)
                or manifest.get("corpus_sha256") != sha_file(self.root / "corpus.jsonl")
                or manifest.get("stella", {}).get("combined", {}).get("sha256")
                   != sha_file(self.root / "documents/vectors.f16.npy")
                or list(self.vectors.shape) != manifest.get("vector_shape")
                or self.bm25.doc_ids != self.ids):
            raise SystemExit("M18 PROJECT MEMORY REFUSED: index manifest integrity check failed")
        self.encoder = _query_encoder(bundle)
        corpus = read_corpus(self.root / "corpus.jsonl")
        self.metadata = {r["doc_id"]: r for r in corpus}

    def query(self, text, limit=10, prefetch=100):
        t0 = time.perf_counter()
        q = self.encoder.encode([text])
        dense = search(q, self.vectors, prefetch, self.ids, ["query"], query_block=1)["query"]
        lexical = self.bm25.query(text, prefetch)
        fused = dbsf_at({"query": dense}, {"query": lexical}, prefetch)["query"]
        ranked = sorted(fused.items(), key=lambda x: (-x[1], str(x[0])))[:limit]
        return {"query": text, "latency_ms": (time.perf_counter() - t0) * 1000,
                "results": [{"doc_id": d, "score": float(s), **self.metadata.get(d, {})}
                            for d, s in ranked]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--corpus", default=None); b.add_argument("--out", default=None)
    b.add_argument("--device", default="cuda"); b.add_argument("--shard-size", type=int, default=4096)
    q = sub.add_parser("query")
    q.add_argument("text"); q.add_argument("--index", default=str(WORK / "derived/index"))
    q.add_argument("--bundle", required=True); q.add_argument("--limit", type=int, default=10)
    args = ap.parse_args(argv)
    if args.command == "build":
        print(json.dumps(build_index(args.corpus, args.out, args.device, args.shard_size),
                         indent=1, sort_keys=True))
    else:
        print(json.dumps(ProjectMemory(args.index, args.bundle).query(args.text, args.limit),
                         indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
