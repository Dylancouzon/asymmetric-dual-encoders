"""Network-free contract tests for the M18 source snapshot adapter."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import source


def _registry(*endpoints, per_page=2, cutoff="2026-01-10T00:00:00Z"):
    return {
        "status": "EXECUTABLE_PREPARATION",
        "source": {
            "repository": "qdrant/qdrant", "default_branch": "master", "commit": "a" * 40,
            "git_remote": "https://github.com/qdrant/qdrant.git", "github_cutoff_utc": cutoff,
            "api_version": "2022-11-28", "per_page": per_page, "endpoints": list(endpoints),
        },
    }


def _endpoint(name="issues", path="/repos/qdrant/qdrant/issues"):
    return {"name": name, "path": path, "params": {"state": "all", "sort": "created"}}


def _row(node, created="2026-01-01T00:00:00Z", updated=None, ident=1):
    row = {"node_id": node, "id": ident, "created_at": created}
    if updated:
        row["updated_at"] = updated
    return row


class Pages:
    def __init__(self, values):
        self.values = {key: list(value) for key, value in values.items()}
        self.calls = []

    def __call__(self, path, params, headers):
        self.calls.append((path, dict(params), dict(headers)))
        key = (path, params["page"])
        if not self.values.get(key):
            raise AssertionError(f"unexpected request {key}")
        return self.values[key].pop(0), {}


def test_resume_uses_valid_completed_pages_without_network(tmp_path):
    endpoint = _endpoint()
    reg = _registry(endpoint)
    first = Pages({
        (endpoint["path"], 1): [[_row("A", ident=1), _row("B", ident=2)]],
        (endpoint["path"], 2): [[_row("C", ident=3)]],
    })
    before = source.acquire_github(tmp_path, registry_data=reg, api_call=first, reconcile=False)
    assert [call[1]["page"] for call in first.calls] == [1, 2]
    assert first.calls[0][2] == {"Accept": source.ACCEPT, "X-GitHub-Api-Version": "2022-11-28"}

    def no_network(*_):
        raise AssertionError("a completed receipt must resume offline")

    after = source.acquire_github(tmp_path, registry_data=reg, api_call=no_network, reconcile=False)
    assert before == after
    assert after["endpoints"]["issues"]["raw_objects"] == 3


@pytest.mark.parametrize("which", ("raw", "receipt"))
def test_partial_transaction_is_refused_before_resume(tmp_path, which):
    endpoint = _endpoint()
    root = tmp_path / "work" / "m18" / "source" / "github" / "issues"
    raw = root / "pages" / "000001.json"
    receipt = root / "receipts" / "000001.json"
    (raw if which == "raw" else receipt).parent.mkdir(parents=True)
    (raw if which == "raw" else receipt).write_text("[]")
    with pytest.raises(source.SourceError, match="partial"):
        source.acquire_github(tmp_path, registry_data=_registry(endpoint), api_call=lambda *_: [])


def test_corrupt_or_changed_immutable_page_is_refused(tmp_path):
    endpoint = _endpoint()
    reg = _registry(endpoint, per_page=10)
    source.acquire_github(tmp_path, registry_data=reg,
                          api_call=Pages({(endpoint["path"], 1): [[_row("A")]]}), reconcile=False)
    raw = tmp_path / "work" / "m18" / "source" / "github" / "issues" / "pages" / "000001.json"
    raw.write_text("[not-json]")
    with pytest.raises(source.SourceError, match="immutable page validation failed|corrupt"):
        source.acquire_github(tmp_path, registry_data=reg, api_call=lambda *_: [])


def test_node_dedupe_cutoff_and_post_cutoff_update_counts(tmp_path):
    issues, comments = _endpoint(), _endpoint("issue_comments", "/repos/qdrant/qdrant/issues/comments")
    reg = _registry(issues, comments, per_page=10)
    api = Pages({
        (issues["path"], 1): [[
            _row("shared", updated="2026-01-11T00:00:00Z"),
            _row("new", created="2026-01-11T00:00:00Z", ident=2),
        ]],
        (comments["path"], 1): [[_row("shared", ident=3)]],
    })
    manifest = source.acquire_github(tmp_path, registry_data=reg, api_call=api, reconcile=False)
    assert manifest["canonical_unique_after_cutoff"] == 1
    assert manifest["cross_endpoint_duplicates_after_cutoff"] == 1
    assert manifest["endpoints"]["issues"]["created_after_cutoff"] == 1
    assert manifest["endpoints"]["issues"]["updated_after_cutoff"] == 1


def test_second_pass_reconciliation_accepts_body_edit_and_post_cutoff_tail(tmp_path):
    endpoint = _endpoint()
    reg = _registry(endpoint, per_page=10)
    first = _row("A")
    first["body"] = "original"
    edited = dict(first, body="edited")
    tail = _row("later", created="2026-01-11T00:00:00Z", ident=2)
    api = Pages({
        (endpoint["path"], 1): [
            [first],                       # acquisition
            [edited, tail],                # same admitted ID, mutable edit, growing tail
        ],
    })
    manifest = source.acquire_github(tmp_path, registry_data=reg, api_call=api, reconcile=True)
    assert manifest["reconciliation"]["issues"]["mutable_payload_changes"] == 1
    raw = tmp_path / "work/m18/source/github/issues/pages/000001.json"
    assert json.loads(raw.read_text())[0]["body"] == "original"


def test_second_pass_reconciliation_rejects_changed_admitted_ids(tmp_path):
    endpoint = _endpoint()
    reg = _registry(endpoint, per_page=10)
    api = Pages({(endpoint["path"], 1): [[_row("A")], [_row("changed")]]})
    with pytest.raises(source.SourceError, match="admitted identity set changed"):
        source.acquire_github(tmp_path, registry_data=reg, api_call=api, reconcile=True)


def test_fallback_identity_keeps_endpoint_kinds_separate():
    assert source.canonical_key({"id": 7}, "issues") != source.canonical_key({"id": 7}, "issue_comments")
    assert source.canonical_key({"id": 7, "node_id": "N"}, "issues") == "node:N"


def test_receipts_never_persist_token_bearing_headers(tmp_path):
    endpoint = _endpoint()
    reg = _registry(endpoint, per_page=10)
    def api(_path, _params, _headers):
        if _params["page"] == 1:
            return [_row("A")], {"Link": "<https://api.github.com/x?access_token=not-a-token>; rel=\"next\""}
        return [], {}
    source.acquire_github(tmp_path, registry_data=reg, api_call=api, reconcile=False)
    receipt = tmp_path / "work" / "m18" / "source" / "github" / "issues" / "receipts" / "000001.json"
    text = receipt.read_text()
    assert "Authorization" not in text and "ghp_" not in text and "not-a-token" not in text
    assert json.loads(text)["link_header"] == "[REDACTED]"


def test_git_snapshot_clones_fetches_and_detaches_at_registered_pin(tmp_path):
    endpoint = _endpoint()
    reg = _registry(endpoint)
    commit = reg["source"]["commit"]
    calls = []

    def git(args, **_kwargs):
        calls.append(args)
        if args[1:3] == ["clone", "--no-checkout"]:
            (Path(args[-1]) / ".git").mkdir(parents=True)
            output = ""
        elif args[-3:] == ["remote", "get-url", "origin"]:
            output = reg["source"]["git_remote"]
        elif args[-2:] == ["status", "--porcelain"]:
            output = ""
        elif args[-2:] == ["rev-parse", "HEAD"]:
            output = commit
        elif args[-2:] == ["rev-parse", "HEAD^{tree}"]:
            output = "b" * 40
        else:
            output = ""
        return SimpleNamespace(returncode=0, stdout=output, stderr="")

    manifest = source.ensure_git_snapshot(tmp_path, registry_data=reg, run=git)
    assert manifest["head"] == commit and manifest["tree"] == "b" * 40
    assert ["git", "-C", str(tmp_path / "work" / "m18" / "source" / "repository"), "fetch", "--force", "origin", commit] in calls
    checkout = next(command for command in calls if command[-3:] == ["checkout", "--detach", commit])
    assert "--force" not in checkout


def test_changed_source_identity_is_refused_on_resume(tmp_path):
    endpoint = _endpoint()
    reg = _registry(endpoint, per_page=10)
    source.acquire_github(tmp_path, registry_data=reg,
                          api_call=Pages({(endpoint["path"], 1): [[_row("A")]]}), reconcile=False)
    changed = _registry(endpoint, per_page=10, cutoff="2026-01-09T00:00:00Z")
    with pytest.raises(source.SourceError, match="different source identity"):
        source.acquire_github(tmp_path, registry_data=changed, api_call=lambda *_: [])


def test_fresh_no_checkout_clone_reaches_detached_pin_with_local_git(tmp_path):
    """A real local clone catches the deleted-worktree status from --no-checkout."""
    origin = tmp_path / "origin"
    subprocess.run(["git", "init", "-q", str(origin)], check=True)
    subprocess.run(["git", "-C", str(origin), "config", "user.email", "m18@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(origin), "config", "user.name", "M18 test"], check=True)
    (origin / "README.md").write_text("pinned fixture\n")
    subprocess.run(["git", "-C", str(origin), "add", "README.md"], check=True)
    subprocess.run(["git", "-C", str(origin), "commit", "-qm", "fixture"], check=True)
    commit = subprocess.run(["git", "-C", str(origin), "rev-parse", "HEAD"], check=True,
                            capture_output=True, text=True).stdout.strip()
    reg = _registry(_endpoint())
    reg["source"]["commit"] = commit
    reg["source"]["git_remote"] = str(origin)
    manifest = source.ensure_git_snapshot(tmp_path / "snapshot", registry_data=reg)
    assert manifest["head"] == commit
