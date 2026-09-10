"""Shared fixtures for the M13 executor tests.

The synthetic world is built ONCE per session (`m13src/rehearse13.build`) and copied per test, with
the copy's own bare origin, so a test that spends the tag or pushes a commit cannot affect another
test — or, ever, the real repository. Nothing here touches `results/`, the real registry or the real
origin, and no protected payload exists anywhere in the fixture.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "m7src", REPO / "m13src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import rehearse13                                                          # noqa: E402


def git(root, *a):
    r = subprocess.run(("git",) + a, cwd=root, capture_output=True, text=True)
    assert r.returncode == 0, f"git {' '.join(a)}: {r.stderr}"
    return r.stdout.strip()


@pytest.fixture(scope="session")
def base_fixture(tmp_path_factory):
    root = tmp_path_factory.mktemp("m13base")
    rehearse13.build(root / "world", n_docs=60, n_queries=20, dim=64)
    return root / "world"


@pytest.fixture
def world(base_fixture, tmp_path):
    """A private copy of the fixture world, re-pointed at its own bare origin. -> Config."""
    dst = tmp_path / "world"
    shutil.copytree(base_fixture, dst)
    repo, origin = dst / "repo", dst / "origin.git"
    reg = repo / "m10" / "final_run_registry.json"
    conf = json.loads(reg.read_text())
    conf["origin_url"] = str(origin)
    reg.write_text(json.dumps(conf, indent=1))
    git(repo, "remote", "set-url", "origin", str(origin))
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "fixture: re-point origin")
    git(repo, "push", "-q", "origin", "main")
    return rehearse13._reopen(dst)
