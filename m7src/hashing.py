"""Streaming equivalents of the M4 hash convention, for corpora too large to json.dumps whole.

The six's manifest entries were written as sha256(json.dumps(obj, sort_keys=True)). For a list
of strings that serialization is '["a", "b", ...]', so it can be fed to the hasher piece by
piece -- byte-identical, without a 4 GB intermediate string. Verified against the frozen
manifest by `verify_stream_matches()`.
"""
import hashlib
import json


def sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def sha_stream_list(it):
    h = hashlib.sha256()
    h.update(b"[")
    first = True
    for x in it:
        if not first:
            h.update(b", ")
        first = False
        h.update(json.dumps(x).encode())
    h.update(b"]")
    return h.hexdigest()


def verify_stream_matches():
    for case in ([], ["a"], ["a", "b"], ["xé", 'q"uote', "tab\tnl\n"]):
        assert sha(case) == sha_stream_list(case), case
    return True
