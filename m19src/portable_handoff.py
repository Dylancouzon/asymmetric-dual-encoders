"""Verify or export the explicit, non-evaluation M18/M19 portability payload."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tarfile
import tempfile
from pathlib import Path, PurePosixPath


REPO = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO / "m19" / "portable-artifacts-v1.json"


def sha256_file(path: Path, chunk_size: int = 4 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("_schema") != "m18-m19-portable-artifacts-v1":
        raise ValueError(f"unsupported portability manifest: {path}")
    return manifest


def selected_files(manifest: dict, profile: str) -> list[tuple[str, dict]]:
    if profile == "all":
        names = [name for item in manifest["profiles"].values() for name in item["files"]]
    else:
        try:
            names = manifest["profiles"][profile]["files"]
        except KeyError as exc:
            raise ValueError(f"unknown profile {profile!r}") from exc
    unique = list(dict.fromkeys(names))
    return [(name, manifest["files"][name]) for name in unique]


def archive_path(record: dict) -> PurePosixPath:
    path = PurePosixPath(record.get("archive_path", record["path"]))
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"unsafe archive path: {path}")
    if not str(path).startswith("work/"):
        raise ValueError(f"payload path must remain under ignored work/: {path}")
    return path


def source_path(record: dict, repo_root: Path, release_root: Path | None) -> Path:
    source = record["source"]
    if source == "repo":
        root = repo_root
    elif source == "release":
        root = release_root or repo_root / "work" / "release" / "zero-v1"
    else:
        raise ValueError(f"unknown source root {source!r}")
    path = (root / record["path"]).resolve()
    expected_root = root.resolve()
    if path != expected_root and expected_root not in path.parents:
        raise ValueError(f"source path escapes {source} root: {path}")
    return path


def verify_files(entries: list[tuple[str, dict]], repo_root: Path,
                 release_root: Path | None) -> list[tuple[str, dict, Path]]:
    verified = []
    failures = []
    for name, record in entries:
        path = source_path(record, repo_root, release_root)
        if not path.is_file():
            failures.append(f"{name}: missing {path}")
            continue
        size = path.stat().st_size
        if size != record["bytes"]:
            failures.append(f"{name}: size {size}, expected {record['bytes']} ({path})")
            continue
        actual_hash = sha256_file(path)
        if actual_hash != record["sha256"]:
            failures.append(
                f"{name}: sha256 {actual_hash}, expected {record['sha256']} ({path})"
            )
            continue
        archive_path(record)
        verified.append((name, record, path))
    if failures:
        raise SystemExit("PORTABLE HANDOFF REFUSED:\n" + "\n".join(failures))
    return verified


class _HashingReader:
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()
        self.bytes_read = 0

    def read(self, size: int) -> bytes:
        payload = self.stream.read(size)
        self.digest.update(payload)
        self.bytes_read += len(payload)
        return payload


def export_archive(entries: list[tuple[str, dict, Path]], output: Path) -> None:
    output = output.resolve()
    if output.exists():
        raise SystemExit(f"PORTABLE HANDOFF REFUSED: output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=output.name + ".tmp-", dir=output.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with tarfile.open(temporary, "w:gz", format=tarfile.PAX_FORMAT) as archive:
            for _, record, source in entries:
                target = str(archive_path(record))
                info = archive.gettarinfo(str(source), arcname=target)
                info.size = record["bytes"]
                info.mtime = 0
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                info.mode = 0o644
                with source.open("rb") as stream:
                    checked = _HashingReader(stream)
                    archive.addfile(info, checked)
                if (checked.bytes_read != record["bytes"]
                        or checked.digest.hexdigest() != record["sha256"]):
                    raise SystemExit(
                        f"PORTABLE HANDOFF REFUSED: {source} changed during export"
                    )
        try:
            os.link(temporary, output)
        except FileExistsError as exc:
            raise SystemExit(
                f"PORTABLE HANDOFF REFUSED: output appeared during export: {output}"
            ) from exc
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "export"))
    parser.add_argument("--profile", choices=("system", "m19-candidates", "all"),
                        default="system")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--repo-root", type=Path, default=REPO)
    parser.add_argument("--release-root", type=Path)
    parser.add_argument("--output", type=Path,
                        help="new .tar.gz path; required only for export")
    args = parser.parse_args(argv)
    if args.command == "export" and args.output is None:
        parser.error("export requires --output")
    manifest = load_manifest(args.manifest)
    entries = selected_files(manifest, args.profile)
    verified = verify_files(entries, args.repo_root, args.release_root)
    total = sum(record["bytes"] for _, record, _ in verified)
    if args.command == "export":
        export_archive(verified, args.output)
        print(f"exported {len(verified)} files ({total} bytes) to {args.output.resolve()}")
    else:
        print(f"verified {len(verified)} files ({total} bytes) for profile {args.profile}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
