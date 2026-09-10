"""M13 scoring-transaction checks: the six, the bridge, the crash modes, the reserved trigger.

Every test runs the PRODUCTION executor against the synthetic fixture. The two crash tests kill a
real subprocess with SIGKILL, because a `try/finally` cannot prove what an unclean death leaves on
disk.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import access13
import score13

REPO = access13.REPO
ENV_PATHS = [str(REPO), str(REPO / "m7src"), str(REPO / "m13src")]

# A driver that runs the transaction and SIGKILLs itself before the Nth per-dataset write.
DRIVER = """
import os, signal, sys
sys.path[:0] = {paths!r}
import rehearse13, score13
cfg = rehearse13._reopen({root!r})
kill_before = {n}
orig, seen = score13.score_dataset, [0]
def patched(*a, **k):
    if seen[0] >= kill_before:
        os.kill(os.getpid(), signal.SIGKILL)
    seen[0] += 1
    return orig(*a, **k)
score13.score_dataset = patched
sys.exit(score13.run(cfg))
"""


def _kill_run(world, n):
    root = str(Path(world.repo).parent)
    code = DRIVER.format(paths=ENV_PATHS, root=root, n=n)
    return subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True)


def _conf(cfg):
    return json.loads(Path(cfg.registry_path).read_text())


def _commit_push(cfg, msg):
    subprocess.run(["git", "add", "-A"], cwd=cfg.repo, check=True)
    subprocess.run(["git", "commit", "-qm", msg], cwd=cfg.repo, check=True)
    subprocess.run(["git", "push", "-q", "origin", "main"], cwd=cfg.repo, check=True)


# --------------------------------------------------------------- the happy path

def test_five_dataset_env_still_scores_six(world, monkeypatch, capsys):
    """`bench/core.DATASETS` defaults to FIVE. The transaction iterates `partitions.all6`."""
    monkeypatch.setenv("BENCH_DATASETS", "scifact,nfcorpus,fiqa,arguana,scidocs")
    assert score13.run(world) == score13.EXIT_OK
    conf = _conf(world)
    assert sorted(p.stem for p in Path(world.scores_dir).glob("*.json")
                  if p.stem != "run_state") == sorted(conf["partitions"]["all6"])
    blob = json.loads(world.result_path.read_text())
    assert len(blob["datasets"]) == 6
    # every persisted row is string-keyed and carries the run-time registry sha
    for ds in blob["datasets"]:
        rec = json.loads(world.score_path(ds).read_text())
        assert all(isinstance(k, str) for k in rec["scores"])
        assert rec["registry_sha256"] == access13.sha256_file(world.registry_path)
    assert "VERDICT" in capsys.readouterr().out


def test_the_anchor_row_is_discarded_and_movement_is_reported(world):
    assert score13.run(world) == score13.EXIT_OK
    for ds in _conf(world)["partitions"]["all6"]:
        rec = json.loads(world.score_path(ds).read_text())
        assert set(rec["bridge"]) >= {"mean_delta", "max_abs_delta", "changed_queries", "n"}
        assert "anchor_scores" not in rec and rec["system"] == "nano-dense"


# --------------------------------------------------------------- qid alignment

def test_a_dropped_qrel_makes_the_scored_set_short_and_is_refused(world):
    conf = _conf(world)
    ds = conf["partitions"]["all6"][0]
    p = Path(world.frozen_eval_dir) / f"{ds}.json"
    froz = json.loads(p.read_text())
    froz["qrels"].pop(sorted(froz["qrels"])[0])          # pytrec_eval will silently omit that qid
    p.write_text(json.dumps(froz))
    _commit_push(world, "fixture: drop one qrel")
    with pytest.raises(SystemExit) as e:
        score13.run(world)
    assert "scored qid set is not the frozen list" in str(e.value)


def test_payload_checks_refuse_a_missing_query(world):
    conf = _conf(world)
    ds = conf["partitions"]["all6"][0]
    p = Path(world.frozen_eval_dir) / f"{ds}.json"
    froz = json.loads(p.read_text())
    froz["queries"].pop(sorted(froz["queries"])[0])
    p.write_text(json.dumps(froz))
    problems = score13.payload_checks(world, conf, conf["partitions"]["all6"])
    assert any("frozen payload qids != comparator qids" in x for x in problems), problems


def test_payload_checks_refuse_relabelled_qids(world):
    """A JSON round-trip stringifies every key, so a frozen payload can never ARRIVE with int qids
    — relabelling them is what such a mistake looks like on disk, and the set check catches it."""
    conf = _conf(world)
    ds = conf["partitions"]["all6"][0]
    p = Path(world.frozen_eval_dir) / f"{ds}.json"
    froz = json.loads(p.read_text())
    froz["queries"] = {str(i): t for i, t in enumerate(froz["queries"].values())}
    p.write_text(json.dumps(froz))
    problems = score13.payload_checks(world, conf, [ds])
    assert any("frozen payload qids != comparator qids" in x for x in problems), problems


def test_assert_qid_set_refuses_int_keys_and_a_comparator_mismatch():
    # int-vs-str: the scored dict came back with int keys (a non-JSON producer), the frozen list
    # is strings. The sets must not be allowed to "match" by coercion.
    with pytest.raises(SystemExit) as e:
        score13.assert_qid_set("scifact", {1: 0.5}, ["1"], ["1"])
    assert "scored qid set is not the frozen list" in str(e.value)
    with pytest.raises(SystemExit) as e:
        score13.assert_qid_set("scifact", {"a": 1.0}, ["a"], ["a", "b"])
    assert "not the comparator" in str(e.value)


# --------------------------------------------------------------- the bridge

def _row(v, n=40):
    return {f"q{i}": v for i in range(n)}


def test_bridge_dataset_mean_0004_hard_fails_and_0002_passes():
    frozen = _row(0.5)
    ok = score13.bridge_row("scifact", _row(0.502), frozen, 0.003)
    assert ok["passes"] and ok["changed_queries"] == 40
    assert ok["max_abs_delta"] == pytest.approx(0.002)
    assert ok["mean_delta"] == pytest.approx(0.002)
    with pytest.raises(SystemExit) as e:
        score13.bridge_row("scifact", _row(0.504), frozen, 0.003)
    assert "dataset |mean delta|" in str(e.value)


def test_bridge_reports_per_query_movement_without_failing():
    """Per-query movement alone NEVER fails (`bridge.failure`): a big single-query move whose
    dataset mean stays inside the bound passes and is reported."""
    frozen = _row(0.5, n=200)
    moved = dict(frozen)
    moved["q0"] = 0.9
    row = score13.bridge_row("scifact", moved, frozen, 0.003)
    assert row["passes"] and row["changed_queries"] == 1
    assert row["max_abs_delta"] == pytest.approx(0.4)


def test_bridge_hard_fails_on_qid_set_inequality():
    with pytest.raises(SystemExit) as e:
        score13.bridge_row("scifact", _row(0.5, 39), _row(0.5, 40), 0.003)
    assert "qid set differs" in str(e.value)


# --------------------------------------------------------------- crash modes

def test_sigkill_before_the_first_write_loses_the_access(world, capsys):
    r = _kill_run(world, 0)
    assert r.returncode == -9, r.stdout + r.stderr
    assert access13.sh(world, "git", "ls-remote", "origin", f"refs/tags/{world.spent_tag}")
    assert not list(Path(world.scores_dir).glob("*.json")) or \
        sorted(p.name for p in Path(world.scores_dir).glob("*.json")) == ["run_state.json"]
    assert score13.run(world, recover=True) == score13.EXIT_ACCESS_LOST
    assert "LOSS of the six-set access" in capsys.readouterr().out


def test_sigkill_after_three_datasets_continues_with_only_the_rest(world, capsys):
    conf = _conf(world)
    r = _kill_run(world, 3)
    assert r.returncode == -9
    done = sorted(p.stem for p in Path(world.scores_dir).glob("*.json") if p.stem != "run_state")
    assert len(done) == 3

    assert score13.run(world) == score13.EXIT_OK
    out = capsys.readouterr().out
    assert "post-tag continuation: opening ONLY" in out
    remaining = [ds for ds in conf["partitions"]["all6"] if ds not in done]
    for ds in remaining:
        assert ds in out.split("opening ONLY")[1].split("\n")[0]
    assert json.loads(world.result_path.read_text())["mode"] == "continuation"


def test_continuation_refuses_a_moved_head(world, capsys):
    _kill_run(world, 3)
    (Path(world.repo) / "note.txt").write_text("a later commit\n")
    _commit_push(world, "fixture: HEAD moves after BEGIN")
    assert score13.run(world) == score13.EXIT_REFUSED
    assert "is not the FINAL-RUN-BEGIN commit" in capsys.readouterr().out


def test_continuation_refuses_a_moved_registry(world, capsys):
    _kill_run(world, 3)
    conf = _conf(world)
    conf["_touched"] = "the decision constants moved"
    Path(world.registry_path).write_text(json.dumps(conf, indent=1))
    assert score13.run(world) == score13.EXIT_REFUSED
    assert "live registry sha" in capsys.readouterr().out


def test_continuation_refuses_undeclared_output_drift(world, capsys):
    _kill_run(world, 3)
    (Path(world.repo) / "results" / "stray_output.json").write_text("{}")
    assert score13.run(world) == score13.EXIT_REFUSED
    assert "undeclared output drift" in capsys.readouterr().out


def test_recover_refuses_registry_drift_outside_decide(world, capsys):
    assert score13.run(world) == score13.EXIT_OK
    conf = _conf(world)
    conf["_touched"] = "after the run"
    Path(world.registry_path).write_text(json.dumps(conf, indent=1))
    assert score13.run(world, recover=True) == score13.EXIT_REFUSED
    assert "the live registry is" in capsys.readouterr().out


def test_recover_reproduces_the_verdict_from_persisted_scores(world):
    assert score13.run(world) == score13.EXIT_OK
    first = json.loads(world.result_path.read_text())["decision_record"]["verdict"]["conjuncts"]
    assert score13.run(world, recover=True) == score13.EXIT_OK
    blob = json.loads(world.result_path.read_text())
    assert blob["mode"] == "recover"
    again = blob["decision_record"]["verdict"]["conjuncts"]
    assert {c: v["status"] for c, v in again.items()} == {c: v["status"] for c, v in first.items()}


# --------------------------------------------------------------- the decision layer

def test_reserved_batch_runs_iff_a_conjunct_rejects(world):
    assert score13.run(world) == score13.EXIT_OK
    blob = json.loads(world.result_path.read_text())
    rec = blob["decision_record"]
    assert rec["reserved_batch_runs"] is True and rec["verdict"]["rejected"]
    # ... and it refuses precisely, because the reserved document vectors do not exist
    assert blob["reserved"]["status"] == "NOT RUN"
    assert "RESERVED BATCH NOT RUNNABLE" in blob["reserved"]["reason"]

    # the same machinery with a candidate that does NOT beat the bar: no rejection, no batch
    conf = _conf(world)
    scored = score13.persisted(world, conf["partitions"]["all6"])
    for ds, p in scored.items():
        p["scores"] = {q: 0.0 for q in p["scores"]}
    rows = score13.build_rows(world, conf, scored, "nano-dense")
    rec2 = score13.decision_record(world, conf, rows, "nano-dense")
    assert rec2["reserved_batch_runs"] is False
    assert rec2["verdict"]["conjuncts"]["C1b"]["status"] == "NOT_REJECTED"
    assert rec2["verdict"]["conjuncts"]["C1a"]["status"] == "NOT_TESTED"


def test_the_candidate_may_not_shadow_a_frozen_comparator_row(world):
    conf = _conf(world)
    assert score13.run(world) == score13.EXIT_OK
    scored = score13.persisted(world, conf["partitions"]["all6"])
    with pytest.raises(SystemExit) as e:
        score13.build_rows(world, conf, scored, "bge-small-en-v1.5")
    assert "already exists in the frozen comparator" in str(e.value)


def test_decisions_use_final10_against_the_configured_registry(world):
    final10 = score13.bind_final10(world)
    assert Path(final10.REGISTRY) == Path(world.registry_path)
    assert final10.registry_sha256() == access13.sha256_file(world.registry_path)
