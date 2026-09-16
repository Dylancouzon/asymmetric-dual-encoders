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


def _place(origin, destination):
    """Stage one shard beside the archive without a second copy of its bytes.

    The vectors are about 147 GB across the three towers. Copying them into the archive on the
    same 500 GB volume that already holds them would need 294 GB for the vectors alone, which the
    pod does not have. A hard link is the same bytes under a second name, so re-hashing the
    destination still hashes the real content, and rsync to `D:` and rclone to object storage both
    read and transfer real bytes. Falls back to a copy across filesystems.
    """
    if destination.exists():
        destination.unlink()
    try:
        os.link(origin, destination)
    except OSError:
        shutil.copy2(origin, destination)


def build_corpus(root, corpus_name):
    """corpus.jsonl.gz for any corpus; queries and qrels too.

    For a PUBLIC corpus the queries and qrels are dumped here. For a RESERVED one they were
    already written, into this same layout, by the tagged transaction
    (`reserved_support.export_reserved_payload_archive`), from payloads it had legitimately open.
    They are REQUIRED here: R22 asks for queries and qrels for every evaluated dataset, and a
    missing file is a failure rather than something to substitute a pointer for.
    """
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
        missing = [name for name in ("queries.jsonl.gz", "qrels.jsonl.gz")
                   if not (out / name).is_file()]
        if missing:
            raise RuntimeError(
                f"{corpus_name}: {missing} were not exported by the reserved transaction. They "
                f"can only be written from inside it; re-running the archive cannot produce them "
                f"and this module must never reopen a reserved payload to fill the gap.")
        for name in ("queries.jsonl.gz", "qrels.jsonl.gz"):
            files[name] = out / name
        return files, {"source": source, "revision": revision,
                       "queries_and_qrels": "exported by the tagged reserved transaction from "
                                            "payloads it already held open; the frozen originals "
                                            "remain in results/frozen_eval/, hash-pinned in "
                                            "results/eval_manifest.json"}

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
                   "qrels_source": payload["qrels_source"],
                   "qrels_revision": payload["qrels_revision"],
                   "n_queries": len(payload["q_ids"])}


def build_vectors(root, corpus_name):
    """Stage every tower's shards for one corpus, enumerated FROM the source manifest.

    Three things this must not do, each of which would let a verified archive be wrong:
    skip a tower whose shards are absent, accept a staged file on size alone, or hash whatever
    happens to be on disk into a fresh manifest. The source manifest is the authority: it must be
    COMPLETE, and every shard it records must hash to what it records, both at the source and at
    the destination.
    """
    files = {}
    for tower, directory in R.TOWER_DIR.items():
        source = ENC_ROOT / directory / corpus_name
        manifest_path = source / "manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError(f"{corpus_name}/{tower}: no pre-encode manifest at {manifest_path}; "
                               f"the archive may not omit an evaluated tower")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("status") != "COMPLETE":
            raise RuntimeError(f"{corpus_name}/{tower}: pre-encode manifest is "
                               f"{manifest.get('status')!r}, not COMPLETE")
        target = Path(root) / "vectors" / TOWER_ID[tower] / corpus_name
        target.mkdir(parents=True, exist_ok=True)
        wanted = {f"shard_{sid}.npy": row["sha256"] for sid, row in manifest["shards"].items()}
        n_shards = int(manifest.get("n_shards", len(wanted)))
        if len(wanted) != n_shards:
            raise RuntimeError(f"{corpus_name}/{tower}: manifest records {len(wanted)} shards but "
                               f"claims {n_shards}")
        for name, digest in sorted(wanted.items()):
            origin = source / name
            if not origin.is_file() or R.sha_file(origin) != digest:
                raise RuntimeError(f"{origin}: missing or does not match its recorded hash")
            destination = target / name
            if not destination.is_file() or R.sha_file(destination) != digest:
                _place(origin, destination)
                if R.sha_file(destination) != digest:
                    raise RuntimeError(f"{destination}: staged copy does not match the recorded "
                                       f"hash")
            files[f"vectors/{TOWER_ID[tower]}/{corpus_name}/{name}"] = destination
        destination = target / "manifest.json"
        if not destination.is_file() or R.sha_file(destination) != R.sha_file(manifest_path):
            if destination.exists():
                destination.unlink()
            shutil.copy2(manifest_path, destination)
        files[f"vectors/{TOWER_ID[tower]}/{corpus_name}/manifest.json"] = destination
    return files


def build(root, corpora=None):
    """Stage the whole registered inventory. A partial archive is a failure, not a smaller archive."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    required = list(P.corpus_sources())
    corpora = list(corpora or required)
    complete = set(corpora) == set(required)
    entries, datasets = {}, {}
    for corpus_name in corpora:
        files, meta = build_corpus(root, corpus_name)
        for name, path in files.items():
            entries[f"datasets/{corpus_name}/{name}"] = path
        entries.update(build_vectors(root, corpus_name))
        datasets[corpus_name] = meta
        print(f"[archive] {corpus_name}: staged", flush=True)
    manifest = {
        "status": "BUILT" if complete else "PARTIAL",
        "purpose": "M20 deliverable 5 (R22): raw corpora, queries, qrels and fp16 document "
                   "vectors for every evaluated dataset, so a future Stella-space encoder can be "
                   "re-evaluated without another corpus encode.",
        "required_corpora": required,
        "corpora": corpora,
        "reserved_labels": "exported by the tagged reserved transaction; the frozen originals "
                           "stay in results/frozen_eval/, hash-pinned in "
                           "results/eval_manifest.json",
        "towers": TOWER_ID,
        "datasets": datasets,
        "files": {name: {"bytes": path.stat().st_size, "sha256": R.sha_file(path)}
                  for name, path in sorted(entries.items())},
    }
    manifest["total_bytes"] = sum(row["bytes"] for row in manifest["files"].values())
    manifest["n_files"] = len(manifest["files"])
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    (root / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": manifest["status"], "files": manifest["n_files"],
                      "bytes": manifest["total_bytes"]}))
    if not complete:
        raise RuntimeError("archive is PARTIAL: "
                           f"missing {sorted(set(required) - set(corpora))}")
    return manifest


def verify(root, manifest_path=MANIFEST):
    """Re-hash every manifested file under `root`. This is the archive's acceptance test."""
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest.get("status") != "BUILT":
        raise RuntimeError(f"manifest status is {manifest.get('status')!r}; only a complete "
                           f"archive can be accepted")
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


def verify_remote(remote, manifest_path=MANIFEST):
    """Re-hash the OBJECT-STORAGE copy against the manifest, which `rclone copy` does not do.

    The registration promises re-hash verification at BOTH targets. `rclone hashsum sha256` asks
    the destination for its own checksums, so this compares stored objects rather than trusting
    the transfer.
    """
    manifest = json.loads(Path(manifest_path).read_text())
    out = subprocess.run(["rclone", "hashsum", "sha256", remote], check=True, text=True,
                         capture_output=True).stdout
    remote_hashes = {}
    for line in out.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2:
            remote_hashes[parts[1].strip()] = parts[0].strip().lower()
    problems = []
    for name, row in manifest["files"].items():
        got = remote_hashes.get(name)
        if got is None:
            problems.append(f"{name}: absent at {remote}")
        elif got != row["sha256"]:
            problems.append(f"{name}: sha256 differs at {remote}")
    print(json.dumps({"status": "VERIFIED" if not problems else "FAILED", "remote": remote,
                      "files": len(manifest["files"]), "problems": problems[:20]}))
    return problems


def upload(root, remote):
    """rclone copy to the registered bucket/prefix. Credentials stay in the local rclone config."""
    if ":" not in remote:
        raise ValueError("--remote must be an rclone target such as `constella:bucket/prefix`")
    subprocess.run(["rclone", "copy", "--checksum", "--progress", str(root), remote], check=True)
    print(json.dumps({"status": "UPLOADED", "remote": remote,
                      "next": "--verify-remote re-hashes the stored objects; `rclone copy` alone "
                              "is not the registered verification"}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--corpora", nargs="*", default=None)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--remote", default=None)
    parser.add_argument("--verify-remote", action="store_true",
                        help="re-hash the object-storage copy against the manifest")
    args = parser.parse_args(argv)
    if not (args.build or args.verify or args.upload or args.verify_remote):
        parser.error("one of --build, --verify, --upload or --verify-remote is required")
    problems = []
    if args.build:
        build(args.root, args.corpora)
    if args.upload:
        if not args.remote:
            parser.error("--upload needs --remote")
        upload(args.root, args.remote)
    if args.verify:
        problems += verify(args.root)
    if args.verify_remote:
        if not args.remote:
            parser.error("--verify-remote needs --remote")
        problems += verify_remote(args.remote)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
