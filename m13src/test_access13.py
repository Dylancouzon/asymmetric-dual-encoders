"""M13 access-machinery checks. Everything runs against the synthetic fixture (`conftest.world`)."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import access13
import score13

REPO = access13.REPO


def _conf(cfg):
    return json.loads(Path(cfg.registry_path).read_text())


def _write_pq(cfg, pq):
    """Rewrite the fixture comparator AND its registered digest, so a test that means to perturb
    qids does not fail on the sha check it is not testing."""
    cfg.perquery_path.write_text(json.dumps(pq, indent=1))
    conf = _conf(cfg)
    conf["comparator_source"]["sha256"] = access13.sha256_file(cfg.perquery_path)
    Path(cfg.registry_path).write_text(json.dumps(conf, indent=1))
    subprocess.run(["git", "commit", "-qam", "fixture: comparator"], cwd=cfg.repo, check=True)
    subprocess.run(["git", "push", "-q", "origin", "main"], cwd=cfg.repo, check=True)


def test_comparator_sha_mismatch_is_refused_before_the_tag(world, capsys):
    pq = json.loads(world.perquery_path.read_text())
    pq["_note"] = "edited after the digest was registered"
    world.perquery_path.write_text(json.dumps(pq, indent=1))
    subprocess.run(["git", "commit", "-qam", "drift"], cwd=world.repo, check=True)
    subprocess.run(["git", "push", "-q", "origin", "main"], cwd=world.repo, check=True)

    assert score13.run(world) == score13.EXIT_REFUSED
    out = capsys.readouterr().out
    assert "perquery.json sha256" in out and "!= registered" in out
    # nothing was spent
    assert access13.sh(world, "git", "tag", "-l", world.spent_tag) == ""
    assert not any(Path(world.scores_dir).glob("*.json")) if world.scores_dir.exists() else True


def test_manifest_only_qid_check_catches_a_dropped_qid(world):
    conf = _conf(world)
    ds = conf["partitions"]["all6"][0]
    pq = json.loads(world.perquery_path.read_text())
    pq["datasets"][ds]["qids"].pop()
    for name in pq["datasets"][ds]["systems"]:
        pq["datasets"][ds]["systems"][name].pop()
    _write_pq(world, pq)

    problems = access13.preflight(world, _conf(world))
    assert any("qid-set sha" in p for p in problems), problems
    assert any(f"`{ds}`" in p for p in problems)


def test_manifest_only_qid_check_catches_a_reordered_comparator(world):
    conf = _conf(world)
    ds = conf["partitions"]["all6"][1]
    pq = json.loads(world.perquery_path.read_text())
    q = pq["datasets"][ds]["qids"]
    q[0], q[1] = q[1], q[0]
    _write_pq(world, pq)
    problems = access13.preflight(world, _conf(world))
    # the SET hash is unchanged by a reorder; the ordering check is what sees it
    assert any("not in sorted order" in p for p in problems), problems


def test_manifest_only_qid_check_catches_int_keys(world):
    conf = _conf(world)
    ds = conf["partitions"]["all6"][2]
    pq = json.loads(world.perquery_path.read_text())
    pq["datasets"][ds]["qids"] = list(range(len(pq["datasets"][ds]["qids"])))
    _write_pq(world, pq)
    problems = access13.preflight(world, _conf(world))
    assert any("non-string qids" in p for p in problems), problems


def test_the_manifest_hashing_convention_is_the_producers(world):
    """`scripts/freeze_eval_assets.py`:44 -> sha(sorted(q_ids)) with sha at :22."""
    man = json.loads(world.manifest_path.read_text())["datasets"]
    pq = json.loads(world.perquery_path.read_text())["datasets"]
    for ds, entry in man.items():
        assert entry["qids_sha256"] == access13.sha_json(sorted(pq[ds]["qids"]))


def test_seal_protected_paths_failure_is_fatal(monkeypatch):
    monkeypatch.setitem(sys.modules, "paths_guard", None)
    with pytest.raises(SystemExit) as e:
        access13.seal_protected_paths(fatal=True)
    assert "paths_guard" in str(e.value)
    assert access13.seal_protected_paths(fatal=False) is False


def test_spent_tag_check_fails_closed_when_origin_is_unreachable(world):
    subprocess.run(["git", "remote", "set-url", "origin", str(world.repo / "no-such-origin.git")],
                   cwd=world.repo, check=True)
    conf = _conf(world)
    conf["origin_url"] = str(world.repo / "no-such-origin.git")
    world.origin_url = conf["origin_url"]
    with pytest.raises(SystemExit) as e:
        access13.spent_tag_exists(world, conf)
    assert "Refusing to assume the access is unspent" in str(e.value)


def test_origin_is_identity_pinned(world):
    conf = _conf(world)
    conf["origin_url"] = "https://example.invalid/other.git"
    world.origin_url = None
    with pytest.raises(SystemExit) as e:
        access13.spent_tag_exists(world, conf)
    assert "not the registered" in str(e.value)


def test_ratification_and_free_space_and_parity_artifacts_are_required(world):
    conf = _conf(world)
    conf["ratified_by_owner"] = False
    world.min_free_gb = 1e9
    (Path(world.repo) / "results" / "m10_student_parity_box.json").unlink()
    problems = access13.preflight(world, conf)
    assert any("ratified_by_owner" in p for p in problems)
    assert any("GB free" in p for p in problems)
    assert any("serving-parity artifact" in p for p in problems)


def test_a_missing_document_cache_is_refused(world):
    conf = _conf(world)
    ds = conf["partitions"]["all6"][0]
    for d in access13._doc_cache_dirs(world, ds):
        shutil.rmtree(d)
    problems = access13.preflight(world, conf)
    assert any("no document cache" in p and ds in p for p in problems), problems


def test_an_unrecorded_shard_is_refused(world):
    conf = _conf(world)
    ds = conf["partitions"]["all6"][0]
    d = access13._doc_cache_dirs(world, ds)[0]
    (d / "shards.json").unlink()
    problems = access13.preflight(world, conf)
    assert any("no shards.json" in p for p in problems), problems


def test_an_existing_result_is_never_overwritten(world):
    world.result_path.write_text("{}")
    problems = access13.preflight(world, _conf(world))
    assert any("already exists" in p for p in problems)


def test_the_lock_is_exclusive(world):
    access13.acquire_lock(world)
    code = (f"import sys; sys.path[:0]={[str(REPO), str(REPO / 'm7src'), str(REPO / 'm13src')]};"
            f"import access13, rehearse13;"
            f"cfg = rehearse13._reopen({str(Path(world.repo).parent)!r});"
            f"access13.acquire_lock(cfg)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO)
    assert r.returncode != 0
    assert "flock-held" in (r.stdout + r.stderr)


def test_write_atomic_leaves_no_partial_file(tmp_path):
    p = tmp_path / "x.json"
    access13.write_atomic(p, '{"a": 1}')
    access13.write_atomic(p, '{"a": 2}')
    assert json.loads(p.read_text()) == {"a": 2}
    assert not list(tmp_path.glob("*.tmp"))
