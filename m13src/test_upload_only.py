"""Synthetic upload manifest refusal checks without reading registered data."""
import importlib.util
from pathlib import Path
import sys
import pytest

scripts=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(scripts))
import m13_upload_only as upload


def test_exact_manifest_and_duplicate_destinations(tmp_path, monkeypatch):
    monkeypatch.setattr(upload,'REPO',tmp_path)
    monkeypatch.setattr(upload.budget,'sha',lambda path:'a'*64)
    entries=[]
    for i in range(394):
        source=tmp_path/str(i);source.write_bytes(b'fixture')
        entries.append({'source':str(source),'destination':upload.REMOTE+'/'+str(i),'bytes':7,'sha256':'a'*64})
    manifest={'files':394,'entries':entries}
    groups,expected=upload.inventory(manifest)
    assert len(expected)==394 and len(groups['tree'])==394 and not groups['root']
    entries[-1]['destination']=entries[0]['destination']
    with pytest.raises(RuntimeError,match='duplicate'):upload.inventory(manifest)


def test_protected_payload_refuses_before_stat_or_hash(tmp_path,monkeypatch):
    monkeypatch.setattr(upload,'REPO',tmp_path)
    row={'source':str(tmp_path/'perquery.json'),'destination':upload.REMOTE+'/results/perquery.json','bytes':0,'sha256':'a'*64}
    with pytest.raises(RuntimeError,match='Protected'):upload.inventory({'files':394,'entries':[row]*394})
