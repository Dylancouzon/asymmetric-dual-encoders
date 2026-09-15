"""Exercise exact remote verification on synthetic files, never real research payloads."""
import json
from pathlib import Path
import subprocess
import sys
import hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import m13_verify_uploaded as verify


def run_verifier(tmp_path,corrupt=False,wrong_size=False):
    entries=[]
    for i in range(394):
        p=tmp_path/str(i);data=('synthetic '+str(i)).encode();p.write_bytes(data)
        entries.append({'destination':str(p),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
    if corrupt:
        p=tmp_path/'200';p.write_bytes(b'x'*p.stat().st_size)
    if wrong_size:(tmp_path/'200').write_bytes(b'x')
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps({'entries':entries}))
    script=verify.REMOTE_CODE.replace('/tmp/m13-verify-manifest.json',str(manifest))
    return subprocess.run([sys.executable,'-c',script],capture_output=True,text=True,timeout=15)


def test_all_394_hashes_verified(tmp_path):
    result=run_verifier(tmp_path)
    assert result.returncode==0,result.stderr
    lines=[json.loads(line) for line in result.stdout.splitlines()]
    assert lines[-1]['status']=='PASSED' and lines[-1]['verified_files']==394
    assert [x['verified_file'] for x in lines[:-1]]==list(range(1,395))


def test_same_size_corruption_cannot_pass(tmp_path):
    result=run_verifier(tmp_path,corrupt=True)
    assert result.returncode!=0 and 'Checksum mismatch' in result.stderr
    assert '"status": "PASSED"' not in result.stdout


def test_wrong_size_cannot_pass(tmp_path):
    result=run_verifier(tmp_path,wrong_size=True)
    assert result.returncode!=0 and 'Size mismatch' in result.stderr
