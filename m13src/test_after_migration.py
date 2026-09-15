"""Automatic handoff publication in an isolated Git repo and bare origin."""
from pathlib import Path
import subprocess
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import m13_after_migration as handoff


@pytest.fixture
def repo(tmp_path,monkeypatch):
    root=tmp_path/'repo';root.mkdir();remote=tmp_path/'origin.git'
    def run(*args):return subprocess.run(['git',*args],cwd=root,check=True,capture_output=True,text=True)
    run('init','-b','m13-stage1-execution-prep');run('config','user.name','Test');run('config','user.email','test@example.invalid')
    (root/'base.txt').write_text('baseline');run('add','base.txt');run('commit','-m','baseline')
    run('init','--bare',str(remote));run('remote','add','origin',str(remote));run('push','-u','origin','m13-stage1-execution-prep')
    monkeypatch.setattr(handoff,'REPO',root)
    return root,run


def test_publish_only_named_receipt_leaves_untracked_evidence(repo):
    root,run=repo
    (root/'receipt.json').write_text('{"status":"PASSED"}')
    (root/'other-evidence.txt').write_text('preserve')
    handoff.publish(['receipt.json'],'verified receipt')
    assert run('show','--format=','--name-only','HEAD').stdout.strip()=='receipt.json'
    assert (root/'other-evidence.txt').exists()
    handoff.require_pushed()


def test_unrelated_tracked_change_refuses_before_commit(repo):
    root,run=repo;before=run('rev-parse','HEAD').stdout
    (root/'base.txt').write_text('unrelated edit')
    (root/'receipt.json').write_text('{}')
    with pytest.raises(RuntimeError,match='Unrelated'):handoff.publish(['receipt.json'],'must refuse')
    assert run('rev-parse','HEAD').stdout==before


def test_unrelated_staged_new_file_refuses(repo):
    root,run=repo
    (root/'unrelated.txt').write_text('staged');run('add','unrelated.txt')
    (root/'receipt.json').write_text('{}')
    with pytest.raises(RuntimeError,match='Unrelated'):handoff.publish(['receipt.json'],'must refuse')


def test_unconfirmed_child_is_dead_before_stop(tmp_path,monkeypatch):
    import io
    import os
    import signal
    (tmp_path/'api_key').write_text('test-only')
    monkeypatch.setattr(handoff.recovery,'KEYS',tmp_path)
    child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],start_new_session=True)
    requests=[]
    def urlopen(request,timeout):
        assert child.poll() is not None, 'STOP cannot race a late paid child start'
        requests.append((request.full_url,request.get_method()))
        return io.BytesIO(b'{"desiredStatus":"EXITED"}')
    monkeypatch.setattr(handoff.urllib.request,'urlopen',urlopen)
    try:
        assert handoff.cancel_unconfirmed(child)=='EXITED'
        assert child.poll()==-signal.SIGTERM
        assert requests==[(f'https://rest.runpod.io/v1/pods/{handoff.admission.TARGET}/stop','POST'),
                          (f'https://rest.runpod.io/v1/pods/{handoff.admission.TARGET}','GET')]
    finally:
        if child.poll() is None:os.killpg(child.pid,signal.SIGKILL);child.wait()


def test_unresponsive_child_killed_before_stop(tmp_path,monkeypatch):
    import io
    import signal
    (tmp_path/'api_key').write_text('test-only')
    monkeypatch.setattr(handoff.recovery,'KEYS',tmp_path)
    events=[]
    class Child:
        pid=12345
        def poll(self):return None
        def wait(self,timeout):
            events.append(('wait',timeout))
            if timeout==600:raise subprocess.TimeoutExpired('fake-child',timeout)
            return -signal.SIGKILL
    monkeypatch.setattr(handoff.os,'killpg',lambda pid,sig:events.append(('signal',sig)))
    def urlopen(request,timeout):
        assert events==[('signal',signal.SIGTERM),('wait',600),('signal',signal.SIGKILL),('wait',30)]
        return io.BytesIO(b'{"desiredStatus":"EXITED"}')
    monkeypatch.setattr(handoff.urllib.request,'urlopen',urlopen)
    assert handoff.cancel_unconfirmed(Child())=='EXITED'


def test_stop_still_attempted_after_termination_error(monkeypatch):
    events=[]
    class Child:
        pid=12345
        def poll(self):return None
    def denied(*args):events.append('kill');raise PermissionError('injected')
    monkeypatch.setattr(handoff.os,'killpg',denied)
    monkeypatch.setattr(handoff,'stop_failed_dispatch',lambda:events.append('stop') or 'EXITED')
    with pytest.raises(RuntimeError,match='child termination unconfirmed'):
        handoff.cancel_unconfirmed(Child())
    assert events==['kill','kill','stop']

