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


def _enter_phase(cfg):
    """The cheap stand-in for "inside the transaction": a run manifest and a local spent tag, which
    is what `score13.assert_transaction_phase` requires before any frozen payload is opened."""
    access13.write_atomic(cfg.state_path, json.dumps({"_test": "phase stand-in"}))
    subprocess.run(["git", "tag", "-f", cfg.spent_tag], cwd=cfg.repo, check=True,
                   capture_output=True)


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
    assert _score_files(world) == sorted(conf["partitions"]["all6"])
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

def test_a_dropped_qrel_is_refused_before_anything_is_scored(world):
    """pytrec_eval would silently omit that qid. `assert_qid_set` is the backstop (below); the
    qrels hash now catches the edit first, on the objects about to be scored (review A1)."""
    conf = _conf(world)
    ds = conf["partitions"]["all6"][0]
    p = Path(world.frozen_eval_dir) / f"{ds}.json"
    froz = json.loads(p.read_text())
    froz["qrels"].pop(sorted(froz["qrels"])[0])
    p.write_text(json.dumps(froz))
    _commit_push(world, "fixture: drop one qrel")
    with pytest.raises(SystemExit) as e:
        score13.run(world)
    assert "qrels_sha256 mismatch" in str(e.value)
    assert _score_files(world) == []


def test_payload_checks_refuse_a_missing_query(world):
    conf = _conf(world)
    ds = conf["partitions"]["all6"][0]
    p = Path(world.frozen_eval_dir) / f"{ds}.json"
    froz = json.loads(p.read_text())
    froz["queries"].pop(sorted(froz["queries"])[0])
    p.write_text(json.dumps(froz))
    _enter_phase(world)
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
    _enter_phase(world)
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


# --------------------------------------------------------------- crash modes (ruling R14)

def _score_files(cfg):
    return sorted(p.stem for p in Path(cfg.scores_dir).glob("*.json") if p.stem != "run_manifest")


def test_sigkill_after_three_datasets_is_a_loss_and_nothing_is_reopened(world, capsys):
    """R14: there is NO post-tag continuation. The rerun reports and exits nonzero."""
    r = _kill_run(world, 3)
    assert r.returncode == -9, r.stdout + r.stderr
    done = _score_files(world)
    assert len(done) == 3
    assert access13.sh(world, "git", "ls-remote", "origin", f"refs/tags/{world.spent_tag}")

    assert score13.run(world) == score13.EXIT_ACCESS_LOST
    out = capsys.readouterr().out
    assert "ACCESS SPENT" in out and "no post-tag continuation" in out
    assert "never scored:" in out
    assert _score_files(world) == done            # nothing further was opened or written
    assert not world.result_path.exists()


def test_recover_refuses_an_incomplete_run(world, capsys):
    _kill_run(world, 3)
    assert score13.run(world, recover=True) == score13.EXIT_REFUSED
    assert "have no persisted scores" in capsys.readouterr().out


def test_sigkill_before_the_first_write_loses_the_access(world, capsys):
    r = _kill_run(world, 0)
    assert r.returncode == -9, r.stdout + r.stderr
    assert access13.sh(world, "git", "ls-remote", "origin", f"refs/tags/{world.spent_tag}")
    assert _score_files(world) == []
    assert score13.run(world, recover=True) == score13.EXIT_ACCESS_LOST
    assert "LOSS of the six-set access" in capsys.readouterr().out
    # and the plain rerun says the same thing rather than re-opening anything
    assert score13.run(world) == score13.EXIT_ACCESS_LOST
    assert _score_files(world) == []


def test_preflight_only_after_a_crash_opens_nothing_and_exits_zero(world, capsys, monkeypatch):
    """Review B1: observational in every state — including the post-tag state."""
    _kill_run(world, 3)
    opened = []
    real = score13.open_frozen_six
    monkeypatch.setattr(score13, "open_frozen_six", lambda cfg, ds: opened.append(ds) or real(cfg, ds))
    monkeypatch.setattr(score13, "score_dataset",
                        lambda *a, **k: pytest.fail("scored during --preflight-only"))
    before = _score_files(world)
    assert score13.run(world, preflight_only=True) == score13.EXIT_OK
    out = capsys.readouterr().out
    assert "--preflight-only REPORT" in out
    assert "already spent" in out                       # the report SAYS the state it found
    assert opened == [] and _score_files(world) == before


def test_the_run_manifest_is_written_before_the_tag(world, monkeypatch):
    """Review B2/B3: BEGIN-bound identity lands before the receipt does."""
    seen = {}
    real_push = access13.push_spent_tag

    def spy(cfg, freeze_sha):
        seen["manifest_exists"] = Path(cfg.state_path).exists()
        seen["man"] = json.loads(Path(cfg.state_path).read_text())
        seen["tag_before"] = access13.sh(cfg, "git", "tag", "-l", cfg.spent_tag)
        seen["head"] = access13.sh(cfg, "git", "rev-parse", "HEAD")
        return real_push(cfg, freeze_sha)

    monkeypatch.setattr(access13, "push_spent_tag", spy)
    assert score13.run(world) == score13.EXIT_OK
    assert seen["manifest_exists"] is True and seen["tag_before"] == ""
    man = seen["man"]
    assert man["begin_commit"] == seen["head"]
    assert man["code_sha256"] == score13.code_identity()
    assert man["comparator_sha256"] == access13.sha256_file(world.perquery_path)
    assert man["datasets"] == _conf(world)["partitions"]["all6"]
    for ds in man["datasets"]:                          # every row re-presents the identity
        row = json.loads(world.score_path(ds).read_text())
        assert score13.row_identity_problems(ds, row, man) == []


def test_a_row_from_a_foreign_checkpoint_is_refused(world, capsys):
    assert score13.run(world) == score13.EXIT_OK
    ds = _conf(world)["partitions"]["all6"][2]
    row = json.loads(world.score_path(ds).read_text())
    row["freeze_sha256"] = "0" * 64                     # same registry, different checkpoint
    world.score_path(ds).write_text(json.dumps(row))
    assert score13.run(world, recover=True) == score13.EXIT_REFUSED
    assert "freeze_sha256 000000000000" in capsys.readouterr().out


def test_a_bridge_failure_is_persisted_as_terminal_and_never_rescored(world, capsys):
    """Review B3. The frozen anchor row is moved past the 0.003 dataset-mean bound."""
    conf = _conf(world)
    ds = conf["partitions"]["all6"][-1]                 # last, so every other row is persisted
    pq = json.loads(world.perquery_path.read_text())
    anchor = conf["bridge"]["anchor"]
    pq["datasets"][ds]["systems"][anchor] = [max(0.0, v - 0.05)
                                             for v in pq["datasets"][ds]["systems"][anchor]]
    world.perquery_path.write_text(json.dumps(pq, indent=1))
    c = _conf(world)
    c["comparator_source"]["sha256"] = access13.sha256_file(world.perquery_path)
    Path(world.registry_path).write_text(json.dumps(c, indent=1))
    _commit_push(world, "fixture: move the frozen anchor row")

    with pytest.raises(SystemExit) as e:
        score13.run(world)
    assert "BRIDGE HARD FAIL" in str(e.value)
    rec = json.loads(world.score_path(ds).read_text())
    assert rec["bridge_ok"] is False and rec["terminal"] is True and rec["scores"] is None
    assert "BRIDGE HARD FAIL" in rec["error"]

    # a rerun neither re-scores it nor produces a verdict, and --recover refuses
    assert score13.run(world) == score13.EXIT_ACCESS_LOST
    out = capsys.readouterr().out
    assert f"terminal bridge failure: ['{ds}']" in out
    assert json.loads(world.score_path(ds).read_text())["bridge_ok"] is False
    assert score13.run(world, recover=True) == score13.EXIT_REFUSED
    assert "bridge_ok is False" in capsys.readouterr().out


# --------------------------------------------------------------- inside-the-transaction checks

def test_edited_query_text_is_refused_inside(world):
    """Review A1: the qids are untouched, so only a hash of the TEXT catches this."""
    conf = _conf(world)
    ds = conf["partitions"]["all6"][0]
    p = Path(world.frozen_eval_dir) / f"{ds}.json"
    froz = json.loads(p.read_text())
    q0 = sorted(froz["queries"])[0]
    froz["queries"][q0] = froz["queries"][q0] + " tampered"
    p.write_text(json.dumps(froz))
    _commit_push(world, "fixture: edit one query's text")
    with pytest.raises(SystemExit) as e:
        score13.run(world)
    assert "qtexts_sha256 mismatch" in str(e.value)


def test_edited_qrels_are_refused_inside(world):
    conf = _conf(world)
    ds = conf["partitions"]["all6"][0]
    p = Path(world.frozen_eval_dir) / f"{ds}.json"
    froz = json.loads(p.read_text())
    q0 = sorted(froz["qrels"])[0]
    froz["qrels"][q0] = {k: 1 for k in froz["qrels"][q0]}
    p.write_text(json.dumps(froz))
    _commit_push(world, "fixture: relabel one query")
    with pytest.raises(SystemExit) as e:
        score13.run(world)
    assert "qrels_sha256 mismatch" in str(e.value)


def test_a_missing_manifest_field_refuses_rather_than_skipping(world):
    conf = _conf(world)
    ds = conf["partitions"]["all6"][0]
    man = json.loads(world.manifest_path.read_text())
    man["datasets"][ds].pop("qtexts_sha256")
    world.manifest_path.write_text(json.dumps(man, indent=1))
    _commit_push(world, "fixture: drop qtexts_sha256")
    with pytest.raises(SystemExit) as e:
        score13.run(world)
    assert "has no `qtexts_sha256`" in str(e.value)


def test_the_frozen_six_cannot_be_opened_before_the_tag(world):
    """Review A4: one loader, and it asserts the phase."""
    ds = _conf(world)["partitions"]["all6"][0]
    with pytest.raises(SystemExit) as e:
        score13.open_frozen_six(world, ds)
    assert "no run manifest" in str(e.value)
    assert score13.run(world) == score13.EXIT_OK
    assert "queries" in score13.open_frozen_six(world, ds)


def test_a_missing_shard_is_refused_before_any_encode(world):
    """Review A2: `encode_cached` would ENCODE the gap; the cache is checked first."""
    ds = _conf(world)["partitions"]["all6"][0]
    doc_texts = world.corpus_reader(ds)[1]
    assert score13.cache_record(world, ds, doc_texts)["n_shards"] == 1
    d = access13._doc_cache_dirs(world, ds)[0]
    (d / "shard_00000.npy").unlink()
    with pytest.raises(SystemExit) as e:
        score13.cache_record(world, ds, doc_texts)
    assert "REFUSED before any encode" in str(e.value)
    assert "missing on disk" in str(e.value)
    with pytest.raises(SystemExit) as e:
        score13._refuse_to_encode(["text"])
    assert "asked to ENCODE" in str(e.value)


def test_a_foreign_teacher_is_refused(world):
    ds = _conf(world)["partitions"]["all6"][0]
    world.teacher_model_id = "some-other/encoder"
    with pytest.raises(SystemExit) as e:
        score13.cache_record(world, ds, world.corpus_reader(ds)[1])
    assert "not the pinned" in str(e.value)


def test_the_consumed_cache_identity_is_persisted(world):
    assert score13.run(world) == score13.EXIT_OK
    for ds in _conf(world)["partitions"]["all6"]:
        cache = json.loads(world.score_path(ds).read_text())["doc_cache"]
        assert cache["cache_key"].endswith(tuple("0123456789abcdef"))
        assert cache["teacher"]["dtype"] == "fp32"
        assert len(cache["shard_sha256"]) == cache["n_shards"]
        assert all(v and len(v) == 64 for v in cache["shard_sha256"].values())


# --------------------------------------------------------------- the decision layer

def test_reserved_batch_runs_iff_a_conjunct_rejects(world):
    assert score13.run(world) == score13.EXIT_OK
    blob = json.loads(world.result_path.read_text())
    rec = blob["decision_record"]
    assert rec["reserved_batch_runs"] is True and rec["verdict"]["rejected"]
    assert blob["end_status"] == "COMPLETE" and blob["reserved"]["status"] == "complete"

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


def test_a_required_reserved_batch_that_cannot_run_ends_incomplete_and_nonzero(world, capsys):
    """Review B13. Production has no reserved encoder, so this is the production shape."""
    world.reserved_encoder = None
    assert score13.run(world) == score13.EXIT_INCOMPLETE_RESERVED
    blob = json.loads(world.result_path.read_text())
    assert blob["end_status"] == "INCOMPLETE_RESERVED"
    assert blob["reserved"]["status"] == "INCOMPLETE_RESERVED"
    assert "RESERVED BATCH NOT RUNNABLE" in blob["reserved"]["reason"]
    # the six-set decisions stay durable
    assert blob["decision_record"]["verdict"]["rejected"]
    assert len(blob["datasets"]) == 6
    assert "INCOMPLETE_RESERVED" in capsys.readouterr().out
    assert "INCOMPLETE_RESERVED" in Path(world.ledger_path).read_text()


def test_an_evidence_failure_on_a_reached_conjunct_is_unscorable_and_earlier_verdicts_stand(world):
    """The case the design review named: C1b REJECTED, then a later conjunct cannot be scored."""
    import final10
    conf = _conf(world)
    assert score13.run(world) == score13.EXIT_OK
    scored = score13.persisted(world, conf["partitions"]["all6"])
    rows = score13.build_rows(world, conf, scored, "nano-dense")
    real = final10.evidence_for

    def flaky(cid, rows_, *a, **k):
        if cid == "C1a":
            raise ValueError("a NaN in the comparator row")
        return real(cid, rows_, *a, **k)

    final10.evidence_for = flaky
    try:
        rec = score13.decision_record(world, conf, rows, "nano-dense")
    finally:
        final10.evidence_for = real
    v = rec["verdict"]["conjuncts"]
    assert v["C1b"]["status"] == "REJECTED"            # established before the failure, stands
    assert v["C1a"]["status"] == "UNSCORABLE"
    assert "NaN in the comparator row" in v["C1a"]["reason"]
    assert v["C2a"]["status"] == "NOT_TESTED" and v["C2b"]["status"] == "NOT_TESTED"
    assert "C1a" in rec["evidence_errors"]             # the error is persisted either way


def test_an_evidence_failure_on_an_unreachable_conjunct_is_omitted(world):
    """A conjunct the sequence never reaches must not abort the decisions (review A12)."""
    import final10
    conf = _conf(world)
    assert score13.run(world) == score13.EXIT_OK
    scored = score13.persisted(world, conf["partitions"]["all6"])
    for p in scored.values():                           # nothing rejects: the sequence stops at C1b
        p["scores"] = {q: 0.0 for q in p["scores"]}
    rows = score13.build_rows(world, conf, scored, "nano-dense")
    real = final10.evidence_for

    def flaky(cid, rows_, *a, **k):
        if cid == "C2b":
            raise ValueError("unreachable and broken")
        return real(cid, rows_, *a, **k)

    final10.evidence_for = flaky
    try:
        rec = score13.decision_record(world, conf, rows, "nano-dense")
    finally:
        final10.evidence_for = real
    assert rec["verdict"]["conjuncts"]["C1b"]["status"] == "NOT_REJECTED"
    assert rec["verdict"]["conjuncts"]["C2b"]["status"] == "NOT_TESTED"
    assert rec["verdict"]["unscorable"] == []
    assert rec["evidence_errors"]["C2b"].startswith("ValueError")


def test_the_m9_layer_refuses_and_names_what_is_missing(world):
    world.decision_layer = "m9"
    with pytest.raises(SystemExit) as e:
        score13.run(world)
    msg = str(e.value)
    for owed in ("DATASETS", "BRIDGE", "CRASH POLICY", "M9Student"):
        assert owed in msg
    assert score13.M9Student is not None                 # the adapter interface is kept


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
