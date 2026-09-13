"""Pinned, resumable M18 source acquisition.

This module deliberately stores GitHub responses before any parsing or redaction.  A page is
complete only when its immutable JSON payload and its receipt agree.  The command line uses the
authenticated ``gh api`` client; tests pass small in-memory callables instead.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import inspect
import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import parse_qs, urlencode, urlsplit, urlunsplit

try:  # script execution from m18src and package-style test imports are both supported
    from common import REPO, require_executable
except ImportError:  # pragma: no cover - exercised when imported as m18src.source
    from m18src.common import REPO, require_executable


ACCEPT = "application/vnd.github+json"
SOURCE_SCHEMA = "m18-source-manifest-v1"


class SourceError(RuntimeError):
    """A source snapshot is incomplete, inconsistent, mutable, or unsafe to publish."""


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _atomic_json(path: Path, value: Any) -> None:
    """Write a whole JSON file beside its final name, then atomically publish it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _json_bytes(value) + b"\n"
    fd, name = tempfile.mkstemp(prefix=path.name + ".tmp-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass


def _immutable_json(path: Path, value: Any) -> None:
    """Atomically create, but never replace, one evidence file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _json_bytes(value) + b"\n"
    fd, name = tempfile.mkstemp(prefix=path.name + ".tmp-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        try:
            os.link(name, path)  # link creation is atomic and refuses an existing destination
        except FileExistsError as e:
            raise SourceError(f"refusing to replace immutable evidence: {path}") from e
    finally:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass


def _owned_root(root: Path) -> Path:
    """Return the resolved M18 source output root, refusing symlink escapes."""
    root = Path(root).resolve()
    owned = (root / "work" / "m18" / "source").resolve(strict=False)
    try:
        owned.relative_to(root)
    except ValueError as e:  # a symlinked work directory points outside the requested root
        raise SourceError(f"M18 source output escapes --root: {owned}") from e
    return owned


def _safe_remote(remote: str) -> str:
    """Keep a useful remote identity while never retaining URL credentials or query secrets."""
    parts = urlsplit(remote)
    if parts.scheme:
        host = parts.hostname or ""
        port = f":{parts.port}" if parts.port else ""
        return urlunsplit((parts.scheme, host + port, parts.path, "", ""))
    # git's SCP syntax has no query component; remove the only credential-bearing form we accept.
    if "@" in remote and ":" in remote:
        return remote.split("@", 1)[1]
    return remote.split("?", 1)[0]


def _redact(value: Any) -> Any:
    """Defence in depth for anything persisted in a receipt or manifest."""
    secret_words = ("token", "authorization", "password", "secret", "cookie", "credential")
    if isinstance(value, Mapping):
        return {str(k): ("[REDACTED]" if any(w in str(k).lower() for w in secret_words) else _redact(v))
                for k, v in value.items()}
    if isinstance(value, list):
        return [_redact(v) for v in value]
    if isinstance(value, str):
        # GH tokens have a stable prefix; do not needlessly alter ordinary source text.
        for prefix in ("ghp_", "github_pat_", "gho_", "ghs_", "ghu_"):
            if prefix in value:
                return "[REDACTED]"
        if any(word + "=" in value.lower() for word in secret_words):
            return "[REDACTED]"
    return value


def _parse_headers(stdout: str) -> tuple[dict[str, str], str]:
    """Split ``gh api --include`` output, including possible redirect header blocks."""
    normalized = stdout.replace("\r\n", "\n")
    blocks = normalized.split("\n\n")
    header: dict[str, str] = {}
    body = normalized
    for i, block in enumerate(blocks[:-1]):
        lines = block.splitlines()
        if lines and lines[0].startswith("HTTP/"):
            header = {}
            for line in lines[1:]:
                if ":" in line:
                    k, v = line.split(":", 1)
                    header[k.lower()] = v.strip()
            body = "\n\n".join(blocks[i + 1:])
    return header, body


def gh_api_page(path: str, params: Mapping[str, Any], headers: Mapping[str, str],
                run: Callable[..., Any] = subprocess.run) -> tuple[bytes, Mapping[str, str]]:
    """Fetch exactly one API page through the user's authenticated gh CLI session.

    Explicit ``page``/``per_page`` parameters are intentional: ``gh api --paginate`` combines
    pages in stdout and cannot give each page its own immutable receipt.
    """
    query = urlencode([(str(k), str(v)) for k, v in params.items()])
    target = path + ("?" + query if query else "")
    command = ["gh", "api", "--method", "GET", "--include", target]
    for key, value in headers.items():
        command.extend(["-H", f"{key}: {value}"])
    completed = None
    for attempt in range(4):
        completed = run(command, check=False, capture_output=True, text=True)
        if not completed.returncode:
            break
        if attempt < 3:
            time.sleep(2 ** attempt)
    if completed is None or completed.returncode:
        # stderr can contain a URL copied from an environment/proxy.  Do not echo it.
        code = None if completed is None else completed.returncode
        raise SourceError(f"gh api failed for {path} page={params.get('page')} after 4 attempts (exit {code})")
    response_headers, body = _parse_headers(completed.stdout)
    status = response_headers.get(":status")
    if status is None:
        for line in completed.stdout.replace("\r\n", "\n").splitlines()[:1]:
            pieces = line.split()
            if len(pieces) >= 2 and pieces[0].startswith("HTTP/"):
                status = pieces[1]
    if status and (not status.isdigit() or not 200 <= int(status) < 300):
        raise SourceError(f"GitHub API returned HTTP {status} for {path} page={params.get('page')}")
    return body.encode("utf-8"), response_headers


def gh_api_all_pages(path: str, params: Mapping[str, Any], headers: Mapping[str, str],
                     run: Callable[..., Any] = subprocess.run) -> list[list[dict[str, Any]]]:
    """Use one gh process to follow Link cursors for the read-only reconciliation pass."""
    target = path + "?" + urlencode([(str(k), str(v)) for k, v in params.items()])
    command = ["gh", "api", "--method", "GET", "--paginate", target]
    for key, value in headers.items():
        command.extend(["-H", f"{key}: {value}"])
    completed = None
    for attempt in range(4):
        completed = run(command, check=False, capture_output=True, text=False)
        if not completed.returncode:
            break
        if attempt < 3:
            time.sleep(2 ** attempt)
    if completed is None or completed.returncode:
        raise SourceError(f"gh paginated reconciliation failed for {path} after 4 attempts")
    try:
        stream = completed.stdout.decode("utf-8")
        decoder, pages, pos = json.JSONDecoder(), [], 0
        while pos < len(stream):
            while pos < len(stream) and stream[pos].isspace():
                pos += 1
            if pos >= len(stream):
                break
            value, pos = decoder.raw_decode(stream, pos)
            pages.append(value)
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise SourceError(f"gh paginated reconciliation returned invalid JSON for {path}") from e
    if not isinstance(pages, list) or any(not isinstance(page, list) for page in pages):
        raise SourceError(f"gh paginated reconciliation returned an invalid page list for {path}")
    return pages


def _call_api(api_call: Callable[..., Any], path: str, params: Mapping[str, Any],
              headers: Mapping[str, str]) -> tuple[bytes, Mapping[str, str]]:
    """Invoke injectable fixtures accepting (path, params[, headers])."""
    try:
        arity = len(inspect.signature(api_call).parameters)
    except (TypeError, ValueError):  # callable objects without an inspectable signature
        arity = 3
    answer = api_call(path, params, headers) if arity >= 3 else api_call(path, params)
    response_headers: Mapping[str, str] = {}
    body = answer
    if isinstance(answer, tuple) and len(answer) == 2:
        body, response_headers = answer
    elif isinstance(answer, Mapping) and "body" in answer:
        body = answer["body"]
        response_headers = answer.get("headers", {})
    if isinstance(body, bytes):
        raw = body
    elif isinstance(body, str):
        raw = body.encode("utf-8")
    else:
        raw = _json_bytes(body)
    if not isinstance(response_headers, Mapping):
        raise SourceError("API fixture returned non-mapping headers")
    return raw, {str(k).lower(): str(v) for k, v in response_headers.items()}


def _decode_page(raw: bytes, where: Path | str) -> list[dict[str, Any]]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise SourceError(f"corrupt GitHub JSON page: {where}") from e
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise SourceError(f"GitHub API page is not a JSON array of objects: {where}")
    return value


def _request_params(endpoint: Mapping[str, Any], page: int, per_page: int,
                    after: str | None = None) -> dict[str, Any]:
    params = dict(endpoint.get("params", {}))
    params.update({"page": page, "per_page": per_page})
    if after:
        params["after"] = after
    return params


def _next_cursor(link: str) -> str | None:
    for part in str(link or "").split(","):
        if 'rel="next"' not in part and "rel=next" not in part:
            continue
        m = re.match(r"\s*<([^>]+)>", part)
        if not m:
            continue
        values = parse_qs(urlsplit(m.group(1)).query).get("after")
        if values:
            return values[0]
    return None


def _source_identity(source: Mapping[str, Any]) -> dict[str, Any]:
    """Every resume-affecting registry field, excluding credentials and prose-only settings."""
    return {
        "repository": source["repository"], "commit": source["commit"],
        "github_cutoff_utc": source["github_cutoff_utc"], "api_version": source["api_version"],
        "per_page": source["per_page"],
        "endpoints": [{"name": e["name"], "path": e["path"], "params": e.get("params", {})}
                      for e in source["endpoints"]],
    }


def _validate_or_publish_config(owned: Path, source: Mapping[str, Any]) -> None:
    """A resume may not silently apply a different cutoff or endpoint configuration."""
    path = owned / "github" / "source-config.json"
    identity = _redact(_source_identity(source))
    expected = {"source_identity": identity, "sha256": _sha(_json_bytes(identity))}
    if path.exists():
        if _load_receipt(path) != expected:
            raise SourceError("existing GitHub snapshot has a different source identity")
        return
    _immutable_json(path, expected)


def _receipt_path(owned: Path, name: str, page: int) -> Path:
    return _within_owned(owned, owned / "github" / name / "receipts" / f"{page:06d}.json")


def _page_path(owned: Path, name: str, page: int) -> Path:
    return _within_owned(owned, owned / "github" / name / "pages" / f"{page:06d}.json")


def _within_owned(owned: Path, candidate: Path) -> Path:
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(owned.resolve())
    except ValueError as e:
        raise SourceError(f"source path escapes M18-owned output: {candidate}") from e
    return candidate


def _endpoint_name(endpoint: Mapping[str, Any]) -> str:
    name = endpoint.get("name")
    if not isinstance(name, str) or not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
        raise SourceError("endpoint name must be a simple non-empty identifier")
    return name


def _load_receipt(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise SourceError(f"corrupt page receipt: {path}") from e
    if not isinstance(value, dict):
        raise SourceError(f"page receipt is not an object: {path}")
    return value


def _validate_page(owned: Path, endpoint: Mapping[str, Any], page: int, per_page: int,
                   api_version: str | None = None) -> list[dict[str, Any]]:
    name = _endpoint_name(endpoint)
    raw_path, receipt_path = _page_path(owned, name, page), _receipt_path(owned, name, page)
    if raw_path.exists() != receipt_path.exists():
        raise SourceError(f"partial immutable page transaction for {name} page {page}")
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)
    try:
        raw = raw_path.read_bytes()
    except OSError as e:
        raise SourceError(f"cannot read immutable page {raw_path}") from e
    receipt = _load_receipt(receipt_path)
    after = None
    if page > 1:
        previous = _load_receipt(_receipt_path(owned, name, page - 1))
        after = _next_cursor(previous.get("link_header", ""))
    expected = _request_params(endpoint, page, per_page, after)
    legacy_expected = _request_params(endpoint, page, per_page)
    if (receipt.get("complete") is not True or receipt.get("endpoint") != name or
            receipt.get("path") != endpoint.get("path") or receipt.get("params") not in (expected, legacy_expected) or
            receipt.get("sha256") != _sha(raw) or
            receipt.get("bytes") != len(raw) or receipt.get("raw_path") != f"pages/{page:06d}.json" or
            (api_version is not None and receipt.get("api_version") != api_version)):
        raise SourceError(f"immutable page validation failed for {name} page {page}")
    if _time(receipt.get("acquired_at_utc")) is None:
        raise SourceError(f"immutable page has no valid acquisition time: {name} page {page}")
    rows = _decode_page(raw, raw_path)
    if receipt.get("object_count") != len(rows):
        raise SourceError(f"immutable page count mismatch for {name} page {page}")
    return rows


def _existing_pages(owned: Path, endpoint: Mapping[str, Any], per_page: int,
                    api_version: str | None = None) -> list[list[dict[str, Any]]]:
    """Validate every existing page before a resume makes any request."""
    name = _endpoint_name(endpoint)
    page_dir = owned / "github" / name / "pages"
    receipt_dir = owned / "github" / name / "receipts"
    raw_numbers = {p.stem for p in page_dir.glob("*.json")} if page_dir.exists() else set()
    receipt_numbers = {p.stem for p in receipt_dir.glob("*.json")} if receipt_dir.exists() else set()
    if raw_numbers != receipt_numbers:
        raise SourceError(f"partial immutable receipt set for endpoint {name}")
    try:
        numbers = sorted(int(n) for n in raw_numbers)
    except ValueError as e:
        raise SourceError(f"non-numeric page name in endpoint {name}") from e
    if numbers and numbers != list(range(1, numbers[-1] + 1)):
        raise SourceError(f"non-contiguous immutable page set for endpoint {name}")
    pages = [_validate_page(owned, endpoint, n, per_page, api_version) for n in numbers]
    for index, rows in enumerate(pages[:-1], 1):
        if len(rows) != per_page:
            raise SourceError(f"page after terminal page for endpoint {name} (page {index})")
    return pages


def _write_new_page(owned: Path, endpoint: Mapping[str, Any], page: int, per_page: int,
                    raw: bytes, headers: Mapping[str, str], api_version: str,
                    acquired_at_utc: str, request_params: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Create a new immutable page transaction.  Existing paths are never replaced."""
    name = _endpoint_name(endpoint)
    raw_path, receipt_path = _page_path(owned, name, page), _receipt_path(owned, name, page)
    if raw_path.exists() or receipt_path.exists():
        # A concurrent/retried writer is safe only if it produced precisely the same transaction.
        rows = _validate_page(owned, endpoint, page, per_page, api_version)
        if raw_path.read_bytes() != raw:
            raise SourceError(f"refusing to overwrite changed immutable page {name} page {page}")
        return rows
    rows = _decode_page(raw, f"{name} page {page}")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    # Write beside the destination, then link it into place.  No reader can observe partial JSON.
    fd, temp_name = tempfile.mkstemp(prefix=raw_path.name + ".tmp-", dir=raw_path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())
        try:
            os.link(temp_name, raw_path)
        except FileExistsError as e:
            raise SourceError(f"concurrent immutable page collision for {name} page {page}") from e
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
    receipt = {
        "complete": True,
        "endpoint": name,
        "path": endpoint["path"],
        "params": dict(request_params),
        "page": page,
        "object_count": len(rows),
        "bytes": len(raw), "sha256": _sha(raw), "raw_path": f"pages/{page:06d}.json",
        "link_header": _redact(headers.get("link", "")),
        "api_version": api_version,
        "acquired_at_utc": acquired_at_utc,
    }
    _immutable_json(receipt_path, receipt)
    return rows


def canonical_key(row: Mapping[str, Any], endpoint_name: str) -> str:
    """GitHub-global node identity, with a stable kind/id fallback for older API objects."""
    node = row.get("node_id")
    if isinstance(node, str) and node:
        return "node:" + node
    kind = str(row.get("type") or ("pull_request" if row.get("pull_request") else endpoint_name))
    ident = row.get("id")
    if ident is None:
        raise SourceError(f"object without node_id or id in endpoint {endpoint_name}")
    return f"fallback:{kind}:{ident}"


def _time(value: Any) -> dt.datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(dt.timezone.utc)
    except ValueError as e:
        raise SourceError(f"invalid GitHub timestamp {value!r}") from e


def _cutoff_counts(pages: Mapping[str, list[list[dict[str, Any]]]], cutoff: str) -> tuple[dict[str, Any], int]:
    cutoff_time = _time(cutoff)
    if cutoff_time is None:
        raise SourceError("registry source.github_cutoff_utc is required")
    summary: dict[str, Any] = {}
    unique: set[str] = set()
    endpoint_unique_total = 0
    for name, endpoint_pages in pages.items():
        rows = [row for page in endpoint_pages for row in page]
        if any(_time(row.get("created_at")) is None for row in rows):
            raise SourceError(f"object has no valid created_at in endpoint {name}")
        included = [row for row in rows if _time(row.get("created_at")) <= cutoff_time]
        post_created = len(rows) - len(included)
        post_updated = sum(1 for row in included if (_time(row.get("updated_at")) or cutoff_time) > cutoff_time)
        endpoint_unique = {canonical_key(row, name) for row in included}
        endpoint_unique_total += len(endpoint_unique)
        unique.update(endpoint_unique)
        summary[name] = {
            "pages": len(endpoint_pages), "raw_objects": len(rows),
            "created_at_included": len(included), "created_after_cutoff": post_created,
            "updated_after_cutoff": post_updated, "canonical_unique_in_endpoint": len(endpoint_unique),
            "duplicates_in_endpoint": len(included) - len(endpoint_unique),
            "raw_pages_sha256": _sha(_json_bytes([
                _sha(_json_bytes(page)) for page in endpoint_pages])),
        }
    return summary, endpoint_unique_total - len(unique)


def _fetch_endpoint(owned: Path, endpoint: Mapping[str, Any], source: Mapping[str, Any],
                    api_call: Callable[..., Any], now: Callable[[], dt.datetime],
                    existing: list[list[dict[str, Any]]] | None = None) -> list[list[dict[str, Any]]]:
    per_page = int(source["per_page"])
    name = _endpoint_name(endpoint)
    pages = existing if existing is not None else _existing_pages(owned, endpoint, per_page, str(source["api_version"]))
    if pages and len(pages[-1]) < per_page:
        receipt = _load_receipt(_receipt_path(owned, name, len(pages)))
        link = str(receipt.get("link_header", "")).lower()
        if 'rel="next"' not in link and "rel=next" not in link:
            return pages
    if pages:
        link = str(_load_receipt(_receipt_path(owned, name, len(pages))).get("link_header", ""))
        if link and 'rel="next"' not in link.lower() and "rel=next" not in link.lower():
            return pages
    page = len(pages) + 1
    after = None
    if pages:
        after = _next_cursor(_load_receipt(_receipt_path(owned, name, len(pages))).get("link_header", ""))
    headers = {"Accept": ACCEPT, "X-GitHub-Api-Version": str(source["api_version"])}
    while True:
        params = _request_params(endpoint, page, per_page, after)
        raw, response_headers = _call_api(api_call, str(endpoint["path"]),
                                           params, headers)
        rows = _write_new_page(owned, endpoint, page, per_page, raw, response_headers,
                               str(source["api_version"]), now().astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
                               params)
        pages.append(rows)
        link = str(response_headers.get("link", "")).lower()
        has_next = 'rel="next"' in link or "rel=next" in link
        if (response_headers.get("link") and not has_next) or (len(rows) < per_page and not has_next):
            return pages
        after = _next_cursor(response_headers.get("link", ""))
        page += 1


def _admitted(rows: list[dict[str, Any]], endpoint_name: str, cutoff: str) -> dict[str, dict[str, Any]]:
    cutoff_time = _time(cutoff)
    if cutoff_time is None:
        raise SourceError("registry source.github_cutoff_utc is required")
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        created = _time(row.get("created_at"))
        if created is None:
            raise SourceError(f"object has no valid created_at in endpoint {endpoint_name}")
        if created <= cutoff_time:
            key = canonical_key(row, endpoint_name)
            if key in out:
                raise SourceError(f"duplicate canonical identity in endpoint {endpoint_name}: {key}")
            out[key] = row
    return out


def _reconcile_endpoint(owned: Path, endpoint: Mapping[str, Any], source: Mapping[str, Any],
                        api_call: Callable[..., Any]) -> dict[str, Any]:
    """Reconcile the cutoff-admitted identity set, tolerating mutable bodies and a growing tail.

    GitHub issue/comment representations are mutable.  The first pass is the immutable payload we
    parse; the second pass proves that the same cutoff-admitted objects were observed.  Edits are
    recorded, not allowed to make a completed snapshot permanently unresumable.
    """
    per_page = int(source["per_page"])
    stored = _existing_pages(owned, endpoint, per_page, str(source["api_version"]))
    headers = {"Accept": ACCEPT, "X-GitHub-Api-Version": str(source["api_version"])}
    if api_call is gh_api_page:
        observed_pages = gh_api_all_pages(str(endpoint["path"]),
                                          _request_params(endpoint, 1, per_page), headers)
    else:
        observed_pages = []
        number = 1
        after = None
        while True:
            params = _request_params(endpoint, number, per_page, after)
            raw, response_headers = _call_api(api_call, str(endpoint["path"]), params, headers)
            observed = _decode_page(raw, f"reconciliation {endpoint['name']} page {number}")
            observed_pages.append(observed)
            link = response_headers.get("link", "")
            has_next = 'rel="next"' in link.lower() or "rel=next" in link.lower()
            if (link and not has_next) or (len(observed) < per_page and not has_next):
                break
            after = _next_cursor(link)
            number += 1
            if number > max(len(stored) + 1000, 10000):
                raise SourceError(f"unbounded reconciliation pagination for {endpoint['name']}")
    name = str(endpoint["name"])
    cutoff = str(source["github_cutoff_utc"])
    original = _admitted([r for p in stored for r in p], name, cutoff)
    observed = _admitted([r for p in observed_pages for r in p], name, cutoff)
    if set(original) != set(observed):
        missing = len(set(original) - set(observed))
        added = len(set(observed) - set(original))
        raise SourceError(f"GitHub cutoff-admitted identity set changed during reconciliation: "
                          f"{name} missing={missing} added={added}")
    mutations = sum(_sha(_json_bytes(original[k])) != _sha(_json_bytes(observed[k]))
                    for k in original)
    return {"stored_pages": len(stored), "observed_pages": len(observed_pages),
            "admitted_identities": len(original), "mutable_payload_changes": mutations,
            "observed_objects": sum(map(len, observed_pages)),
            "observed_sha256": _sha(_json_bytes(observed_pages))}


def _acquisition_interval(owned: Path, endpoint_pages: Mapping[str, list[list[dict[str, Any]]]]) -> dict[str, str | None]:
    times: list[str] = []
    for name, pages in endpoint_pages.items():
        for page in range(1, len(pages) + 1):
            times.append(str(_load_receipt(_receipt_path(owned, name, page))["acquired_at_utc"]))
    return {"first_page_utc": min(times) if times else None, "last_page_utc": max(times) if times else None}


def acquire_github(root: Path | str = REPO, *, registry_data: Mapping[str, Any] | None = None,
                   api_call: Callable[..., Any] | None = None, reconcile: bool = True,
                   now: Callable[[], dt.datetime] | None = None) -> dict[str, Any]:
    """Acquire every registry endpoint and publish an atomic, parser-ready source manifest."""
    root = Path(root).resolve()
    registry_data = dict(registry_data or json.loads((root / "m18" / "registry.json").read_text()))
    require_executable(registry_data, rehearsal=False, what="M18 source acquisition")
    source = registry_data.get("source", {})
    if not isinstance(source, Mapping):
        raise SourceError("registry source block is missing")
    for key in ("repository", "commit", "github_cutoff_utc", "api_version", "per_page", "endpoints"):
        if key not in source:
            raise SourceError(f"registry source.{key} is required")
    if int(source["per_page"]) < 1:
        raise SourceError("source.per_page must be positive")
    if not reconcile and api_call is None:
        raise SourceError("unreconciled GitHub acquisition is allowed only with an injected fixture API")
    api_call = api_call or gh_api_page
    now = now or (lambda: dt.datetime.now(dt.timezone.utc))
    owned = _owned_root(root)
    _validate_or_publish_config(owned, source)
    endpoint_pages: dict[str, list[list[dict[str, Any]]]] = {}
    endpoints: list[Mapping[str, Any]] = []
    for endpoint in source["endpoints"]:
        if not isinstance(endpoint, Mapping) or not endpoint.get("name") or not endpoint.get("path"):
            raise SourceError("each registry endpoint needs name and path")
        name = _endpoint_name(endpoint)
        if name in endpoint_pages:
            raise SourceError(f"duplicate endpoint name: {name}")
        endpoints.append(endpoint)
        # Validate *all* existing evidence before mutating any endpoint on this run.
        endpoint_pages[name] = _existing_pages(owned, endpoint, int(source["per_page"]), str(source["api_version"]))
    for endpoint in endpoints:
        name = _endpoint_name(endpoint)
        endpoint_pages[name] = _fetch_endpoint(owned, endpoint, source, api_call, now, endpoint_pages[name])
    reconciliation = {}
    if reconcile:
        for endpoint in endpoints:
            reconciliation[_endpoint_name(endpoint)] = _reconcile_endpoint(
                owned, endpoint, source, api_call)
    counts, cross_endpoint_duplicates = _cutoff_counts(endpoint_pages, str(source["github_cutoff_utc"]))
    manifest = _redact({
        "schema": SOURCE_SCHEMA,
        "complete": True,
        "repository": source["repository"], "default_branch": source.get("default_branch"),
        "commit": source["commit"], "commit_time_utc": source.get("commit_time_utc"),
        "github_cutoff_utc": source["github_cutoff_utc"], "api_version": source["api_version"],
        "per_page": source["per_page"], "endpoints": counts,
        "acquisition_interval_utc": _acquisition_interval(owned, endpoint_pages),
        "canonical_unique_after_cutoff": sum(v["canonical_unique_in_endpoint"] for v in counts.values()) - cross_endpoint_duplicates,
        "cross_endpoint_duplicates_after_cutoff": cross_endpoint_duplicates,
        "reconciled_second_pass": bool(reconcile),
        "reconciliation": reconciliation,
        "consistency_limit": "GitHub bodies may have been edited after cutoff; updated_after_cutoff records this.",
    })
    _atomic_json(owned / "github-manifest.json", manifest)
    if root == REPO.resolve():
        _atomic_json(root / "results" / "m18_source_manifest.json", manifest)
    return manifest


def ensure_git_snapshot(root: Path | str = REPO, *, registry_data: Mapping[str, Any] | None = None,
                        run: Callable[..., Any] = subprocess.run) -> dict[str, Any]:
    """Clone/fetch and detach-checkout the registered commit, validating the resulting HEAD."""
    root = Path(root).resolve()
    registry_data = dict(registry_data or json.loads((root / "m18" / "registry.json").read_text()))
    require_executable(registry_data, rehearsal=False, what="M18 repository acquisition")
    source = registry_data["source"]
    remote, commit = str(source["git_remote"]), str(source["commit"])
    owned = _owned_root(root)
    checkout = owned / "repository"
    fresh_clone = False
    def call(args: list[str]) -> str:
        result = run(args, check=False, capture_output=True, text=True)
        if result.returncode:
            raise SourceError("git pinned snapshot command failed")
        return result.stdout.strip()
    if not checkout.exists():
        checkout.parent.mkdir(parents=True, exist_ok=True)
        call(["git", "clone", "--no-checkout", remote, str(checkout)])
        fresh_clone = True
    if not (checkout / ".git").exists():
        raise SourceError(f"repository destination is not a git checkout: {checkout}")
    # --no-checkout deliberately leaves a fresh clone's worktree empty (shown as deletions by
    # status).  Only a pre-existing checkout can represent user edits that need protecting.
    if not fresh_clone and call(["git", "-C", str(checkout), "status", "--porcelain"]):
        raise SourceError("refusing to alter a dirty pinned repository checkout")
    observed_remote = call(["git", "-C", str(checkout), "remote", "get-url", "origin"])
    if _safe_remote(observed_remote) != _safe_remote(remote):
        raise SourceError("existing repository origin does not match registered remote")
    call(["git", "-C", str(checkout), "fetch", "--force", "origin", commit])
    call(["git", "-C", str(checkout), "checkout", "--detach", commit])
    head = call(["git", "-C", str(checkout), "rev-parse", "HEAD"])
    if head.lower() != commit.lower():
        raise SourceError(f"pinned checkout validation failed: expected {commit}, got {head}")
    if call(["git", "-C", str(checkout), "status", "--porcelain"]):
        raise SourceError("pinned checkout is dirty after checkout")
    tree = call(["git", "-C", str(checkout), "rev-parse", "HEAD^{tree}"])
    manifest = _redact({"repository": source["repository"], "remote": _safe_remote(remote),
                        "commit": commit, "head": head, "tree": tree,
                        "checkout": "repository", "complete": True})
    git_manifest = owned / "git-manifest.json"
    if git_manifest.exists():
        if _load_receipt(git_manifest) != manifest:
            raise SourceError("existing git snapshot manifest does not match the registered pin")
    else:
        _immutable_json(git_manifest, manifest)
    return manifest


def acquire_sources(root: Path | str = REPO, *, registry_data: Mapping[str, Any] | None = None,
                    api_call: Callable[..., Any] | None = None, git_run: Callable[..., Any] = subprocess.run,
                    reconcile: bool = True) -> dict[str, Any]:
    """Acquire both immutable API evidence and the pinned repository snapshot."""
    git = ensure_git_snapshot(root, registry_data=registry_data, run=git_run)
    github = acquire_github(root, registry_data=registry_data, api_call=api_call, reconcile=reconcile)
    combined = {"schema": "m18-combined-source-manifest-v1", "complete": True,
                "git": git, "github": github}
    combined["sha256"] = _sha(_json_bytes(combined))
    root = Path(root).resolve()
    _atomic_json(_owned_root(root) / "source-manifest.json", combined)
    if root == REPO.resolve():
        _atomic_json(root / "results" / "m18_source_manifest.json", combined)
    return combined


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Acquire the pinned M18 Qdrant source snapshot")
    parser.add_argument("--root", type=Path, default=REPO, help="repository root (fixtures may use a temp root)")
    parser.add_argument("--github-only", action="store_true", help="skip the pinned git checkout")
    args = parser.parse_args(argv)
    if args.github_only:
        acquire_github(args.root, reconcile=True)
    else:
        acquire_sources(args.root, reconcile=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
