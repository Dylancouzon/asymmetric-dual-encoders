#!/usr/bin/env python3
"""M20 deliverable 5: hash-manifested archive of every evaluated corpus and document vector.

Registered in `m20/REGISTRATION.md` under owner ruling R22.  The point is that any future
Stella-space encoder can be re-evaluated on BEIR-15 without paying for another corpus encode, so
the archive holds the fp16 document vectors as well as the raw payloads.  Stopped Runpod volumes
are not the archive.

**Reserved queries and qrels are not copied here, by design.**  They already live in the
repository at `results/frozen_eval/untouched-*.json`, hash-pinned in `results/eval_manifest.json`
and durable in git, so archiving them would be a fresh protected read that buys nothing.  This
module installs the corpus-only guard and records a pointer plus the pinned hashes instead.  Every
reserved CORPUS and its document vectors are archived in full, which is what a future re-evaluation
actually needs.

Three phases, run where the data is:

  --build   on the pod, from work/m13-reserved-enc and the pinned corpora, into --root
  --verify  anywhere, re-hashing every file in --root against the manifest
  --upload  from the local box, rclone to --remote, then --verify the remote by re-hash

  .venv/bin/python m20src/archive.py --build --root work/m20-archive
  .venv/bin/python m20src/archive.py --verify --root /mnt/d/constella-archive/beir15
  .venv/bin/python m20src/archive.py --upload --root /mnt/d/constella-archive/beir15 \\
      --remote constella:constella-archive/beir15
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
for _p in ("bench", "m7src", "m8src", "m20src"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

import pre_encode as P            # claims the corpus-only guard entry and installs the guard
import roster as R                # noqa: E402

MANIFEST = REPO / "results" / "m20_archive_manifest.json"
ENC_ROOT = REPO / "work" / "m13-reserved-enc"
TOWER_ID = {"stella-400M-v5": "stella-ffeb2b7e",
            "bge-small-en-v1.5": "bge-small-5c38ec7c",
            "arctic-m-v1.5": "arctic-m-e58a8f75"}


def _dump_jsonl(path, rows):
    """Write one gzipped JSON-lines payload atomically.

    gzip rather than zstd: it is in the standard library on every host this runs on, and the
    archive's size is dominated by ~57 GB of fp16 vectors that do not compress anyway. Adding a
    third-party codec to a step whose whole job is durability is the wrong trade.
    `mtime=0` keeps the bytes, and therefore the manifest hash, reproducible.
    """
    import gzip

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as stream:
        for row in rows:
            stream.write((json.dumps(row, sort_keys=True) + "\n").encode())
    os.replace(tmp, path)
    return path


def build_corpus(root, corpus_name):
    """corpus.jsonl.gz for any corpus; queries/qrels too when the corpus is public."""
    out = Path(root) / "datasets" / corpus_name
    files = {}
    corpus, source, revision = P.load_corpus(corpus_name)
    P.authenticate_corpus(corpus_name, corpus, revision)
    path = out / "corpus.jsonl.gz"
    if not path.exists():
        _dump_jsonl(path, ({"_id": str(row["_id"]), "title": row.get("title") or "",
                            "text": row.get("text") or ""} for row in corpus))
    files["corpus.jsonl.gz"] = path
    if corpus_name in P.RESERVED_DATASETS:
        return files, {"source": source, "revision": revision,
                       "queries_and_qrels": "results/frozen_eval/untouched-%s.json in git; hashes "
                                            "pinned in results/eval_manifest.json. Not copied "
                                            "here: archiving them would be a fresh protected read "
                                            "that buys nothing." % corpus_name}
    import beir15

    payload = beir15.load_public(corpus_name)
    qpath = out / "queries.jsonl.gz"
    if not qpath.exists():
        _dump_jsonl(qpath, ({"_id": qid, "text": text}
                            for qid, text in zip(payload["q_ids"], payload["q_texts"])))
    files["queries.jsonl.gz"] = qpath
    rpath = out / "qrels.jsonl.gz"
    if not rpath.exists():
        _dump_jsonl(rpath, ({"query-id": qid, "corpus-id": did, "score": score}
                            for qid, row in payload["qrels"].items()
                            for did, score in sorted(row.items())))
    files["qrels.jsonl.gz"] = rpath
    return files, {"source": source, "revision": revision, "split": payload["split"],
                   "n_queries": len(payload["q_ids"])}


def build_vectors(root, corpus_name):
    """Copy the hash-recorded fp16 shards and their manifest, one directory per tower."""
    files = {}
    for tower, directory in R.TOWER_DIR.items():
        source = ENC_ROOT / directory / corpus_name
        if not (source / "manifest.json").exists():
            continue
        target = Path(root) / "vectors" / TOWER_ID[tower] / corpus_name
        target.mkdir(parents=True, exist_ok=True)
        for item in sorted(source.iterdir()):
            if item.suffix not in (".npy", ".json") or item.name.endswith(".tmp.npy"):
                continue
            destination = target / item.name
            if not destination.exists() or destination.stat().st_size != item.stat().st_size:
                shutil.copy2(item, destination)
            files[f"vectors/{TOWER_ID[tower]}/{corpus_name}/{item.name}"] = destination
    return files


def build(root, corpora=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    corpora = list(corpora or P.corpus_sources())
    entries, datasets = {}, {}
    for corpus_name in corpora:
        files, meta = build_corpus(root, corpus_name)
        for name, path in files.items():
            entries[f"datasets/{corpus_name}/{name}"] = path
        entries.update(build_vectors(root, corpus_name))
        datasets[corpus_name] = meta
        print(f"[archive] {corpus_name}: staged", flush=True)
    manifest = {
        "status": "BUILT",
        "purpose": "M20 deliverable 5 (R22): raw corpora, public queries/qrels and fp16 document "
                   "vectors for every evaluated dataset, so a future Stella-space encoder can be "
                   "re-evaluated without another corpus encode.",
        "reserved_labels": "results/frozen_eval/untouched-*.json in git; hashes in "
                           "results/eval_manifest.json. Deliberately not copied here.",
        "towers": TOWER_ID,
        "datasets": datasets,
        "files": {name: {"bytes": path.stat().st_size, "sha256": R.sha_file(path)}
                  for name, path in sorted(entries.items())},
    }
    manifest["total_bytes"] = sum(row["bytes"] for row in manifest["files"].values())
    manifest["n_files"] = len(manifest["files"])
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    (root / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": "BUILT", "files": manifest["n_files"],
                      "bytes": manifest["total_bytes"]}))
    return manifest


def verify(root, manifest_path=MANIFEST):
    """Re-hash every manifested file under `root`. This is the archive's acceptance test."""
    manifest = json.loads(Path(manifest_path).read_text())
    root = Path(root)
    problems = []
    for name, row in manifest["files"].items():
        path = root / name
        if not path.is_file():
            problems.append(f"{name}: missing")
            continue
        if path.stat().st_size != row["bytes"]:
            problems.append(f"{name}: size {path.stat().st_size} != {row['bytes']}")
            continue
        if R.sha_file(path) != row["sha256"]:
            problems.append(f"{name}: sha256 changed")
    print(json.dumps({"status": "VERIFIED" if not problems else "FAILED", "root": str(root),
                      "files": len(manifest["files"]), "problems": problems[:20]}))
    return problems


def upload(root, remote):
    """rclone copy to the registered bucket/prefix. Credentials stay in the local rclone config."""
    if ":" not in remote:
        raise ValueError("--remote must be an rclone target such as `constella:bucket/prefix`")
    subprocess.run(["rclone", "copy", "--checksum", "--progress", str(root), remote], check=True)
    print(json.dumps({"status": "UPLOADED", "remote": remote}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--corpora", nargs="*", default=None)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--remote", default=None)
    args = parser.parse_args(argv)
    if not (args.build or args.verify or args.upload):
        parser.error("one of --build, --verify or --upload is required")
    if args.build:
        build(args.root, args.corpora)
    if args.upload:
        if not args.remote:
            parser.error("--upload needs --remote")
        upload(args.root, args.remote)
    if args.verify:
        return 1 if verify(args.root) else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
