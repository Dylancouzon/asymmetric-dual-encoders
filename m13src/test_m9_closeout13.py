import json
from pathlib import Path

import pytest

import access13 as A
import m9_closeout13 as M
import score13 as S


@pytest.fixture
def m9world(world):
    original = json.loads(world.registry_path.read_text())
    conf = json.loads((A.REPO / 'm9/final_run_registry.json').read_text())
    conf.update(ratified_by_owner=True, origin_url=original['origin_url'])
    conf['comparator_source'] = original['comparator_source']
    world.registry_path.write_text(json.dumps(conf))
    world.spent_tag = 'm9-six-spent'
    world.ledger_prefix = 'm9'
    world.decision_layer = 'm9'
    world.build_record_path = None
    for args in [('add', '-A'), ('commit', '-qm', 'fixture: M9 protocol'),
                 ('push', '-q', 'origin', 'main')]:
        A.sh(world, 'git', *args)
    return world


def test_m9_uses_own_statistics_and_can_recover_without_scoring(m9world, monkeypatch):
    assert M.run(m9world) == 0
    before = json.loads(m9world.result_path.read_text())
    assert before['decision_record']['quantile'] == 0.0125
    assert before['end_status'] == 'COMPLETE_SIX_ONLY'
    assert len(before['datasets']) == 6
    def refuse(*a, **kw):
        raise AssertionError('Recovery attempted a protected read or encoding')
    monkeypatch.setattr(S, 'load_six', refuse)
    monkeypatch.setattr(S, '_anchor', refuse)
    monkeypatch.setattr(M, 'load_student', refuse)
    assert M.run(m9world, recover=True) == 0
    after = json.loads(m9world.result_path.read_text())
    assert before['decision_record'] == after['decision_record']


def test_m9_post_tag_failure_cannot_resume(m9world, monkeypatch):
    def fail(*a, **kw):
        raise RuntimeError('simulated post-tag failure')
    monkeypatch.setattr(S, 'payload_checks', fail)
    with pytest.raises(RuntimeError):
        M.run(m9world)
    with pytest.raises(ValueError, match='already spent'):
        M.run(m9world)
    with pytest.raises(ValueError, match='Incomplete'):
        M.run(m9world, recover=True)


def test_m9_recovery_refuses_moved_remote_tag(m9world, monkeypatch):
    assert M.run(m9world) == 0
    monkeypatch.setattr(A, 'remote_tag_commit', lambda cfg: 'wrong-begin')
    with pytest.raises(ValueError, match='BEGIN'):
        M.run(m9world, recover=True)


@pytest.mark.parametrize('tamper', ['registry', 'identity', 'qid', 'bridge', 'freeze'])
def test_m9_recovery_refuses_drift(m9world, tamper):
    assert M.run(m9world) == 0
    if tamper == 'registry':
        p = m9world.registry_path
        value = json.loads(p.read_text()); value['bootstrap']['seed'] += 1
    elif tamper == 'freeze':
        p = m9world.freeze_path
        value = json.loads(p.read_text()); value['query_prefix'] = 'changed'
    else:
        p = m9world.score_path('scifact')
        value = json.loads(p.read_text())
        if tamper == 'identity': value['system'] = 'wrong-model'
        elif tamper == 'qid': value['scores'].pop(next(iter(value['scores'])))
        else: value['bridge_ok'] = False
    p.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        M.run(m9world, recover=True)
