"""The encode cache must authenticate its own bytes on the one-shot path.

Codex one-shot-path review 2026-08-28, MAJOR 4: the final run's document vectors came from a
gitignored, mutable work/enc tree that nothing verified. `encode_cached` skipped any shard that
merely EXISTED, and `_combined` accepted `combined.f16` on its byte SIZE alone -- which cannot
tell one 10.7 GB file from another. The cache KEY binds the inputs; nothing bound the outputs.

This suite runs with a stubbed encoder and a 4-text shard size, so it needs no GPU and no model.

    ../.venv/bin/python test_encode_cache.py
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

import teacher

FAILS = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {name}" + (f" -- {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


def refuses(name, fn, must_mention):
    try:
        fn()
    except SystemExit as e:
        check(name, must_mention.lower() in str(e).lower(),
              f"refused but did not mention {must_mention!r}: {str(e)[:200]}")
        return
    check(name, False, "did NOT refuse")


TEXTS = [f"document number {i}" for i in range(10)]      # 3 shards at SHARD=4


def fake_encode(texts, prefix="", *a, **k):
    """Deterministic stand-in for the teacher: row i is (hash of the text) broadcast to 8 dims."""
    return np.array([[float(abs(hash(prefix + t)) % 1000) / 1000.0] * 8 for t in texts],
                    dtype=np.float32)


def call(**kw):
    return teacher.encode_cached("smoke", TEXTS, prefix="", verbose=False,
                                 dtype=torch.float16, **kw)


def main():
    orig = (teacher.ENC, teacher.SHARD, teacher.encode)
    with tempfile.TemporaryDirectory() as td:
        teacher.ENC, teacher.SHARD, teacher.encode = Path(td), 4, fake_encode
        try:
            v = call()
            key = teacher.PROVENANCE["smoke"]["cache_key"]
            d = Path(td) / key
            man = json.loads((d / "shards.json").read_text())
            check("three shards written", len(man["shards"]) == 3, str(sorted(man["shards"])))
            check("every shard is hashed",
                  all(len(r["sha256"]) == 64 for r in man["shards"].values()))
            check("nothing is trust-on-first-use on a cache we wrote",
                  not any(r["trusted_on_first_use"] for r in man["shards"].values()))
            check("the stitch records the shard hashes it was built from",
                  man["combined"]["from_shard_sha256"] ==
                  [man["shards"][f"{s:05d}"]["sha256"] for s in range(3)])
            check("the vectors round-trip", np.allclose(np.asarray(v, dtype=np.float32),
                                                        fake_encode(TEXTS), atol=1e-3))

            p = teacher.PROVENANCE["smoke"]
            check("provenance reports what was written", p["shards_written_now"] == 3
                  and p["shards_trusted_on_first_use"] == 0, json.dumps(p)[:200])

            # a second call verifies rather than re-encodes
            teacher.encode = lambda *a, **k: (_ for _ in ()).throw(
                AssertionError("re-encoded a shard that was already cached"))
            call(verify=True)
            check("verify=True on an intact cache passes and re-encodes nothing",
                  teacher.PROVENANCE["smoke"]["shards_verified"] == 3,
                  json.dumps(teacher.PROVENANCE["smoke"])[:200])

            # mutate one shard: size-preserving, so the old size-only check could never see it
            sp = d / "shard_00001.npy"
            arr = np.load(sp)
            arr[0, 0] = np.float16(0.123)
            np.save(sp, arr)
            refuses("a mutated shard is refused under verify", lambda: call(verify=True),
                    "does not match the hash recorded")

            # Without verify the same mutation is NOT detected -- nothing re-hashes a shard it did
            # not write, which is what keeps the dev loop free of a full re-read. Asserted rather
            # than left implicit, so the limitation is a documented property and not a surprise:
            # dev numbers rest on unauthenticated cache bytes, confirmatory ones do not.
            before = json.loads((d / "shards.json").read_text())["combined"]["sha256"]
            call()
            check("without verify a size-preserving mutation is undetected (documented cost)",
                  json.loads((d / "shards.json").read_text())["combined"]["sha256"] == before)

            # A shard legitimately re-encoded (deleted, then rebuilt to DIFFERENT bytes) must
            # restitch: this is the case the old byte-size check could never see, since the new
            # combined.f16 has exactly the size of the old one.
            teacher.encode = lambda texts, prefix="", *a, **k: fake_encode(texts, prefix) + 0.5
            sp.unlink()
            call()
            man2 = json.loads((d / "shards.json").read_text())
            check("a re-encoded shard updates its hash and restitches combined.f16",
                  man2["combined"]["sha256"] != before
                  and man2["combined"]["from_shard_sha256"] ==
                  [man2["shards"][f"{s:05d}"]["sha256"] for s in range(3)],
                  f"before={before[:12]} after={man2['combined']['sha256'][:12]}")
            check("the restitched combined.f16 carries the NEW shard's rows",
                  abs(float(np.asarray(call(), dtype=np.float32)[4, 0]
                            - fake_encode(TEXTS[4:5])[0, 0]) - 0.5) < 1e-3)
            check("and the untouched shards are unchanged",
                  np.allclose(np.asarray(call(), dtype=np.float32)[0], fake_encode(TEXTS[:1])[0],
                              atol=1e-3))
            call(verify=True)
            check("verify passes again once the cache is consistent",
                  teacher.PROVENANCE["smoke"]["shards_verified"] == 3)

            # a cache with no manifest at all -- every cache written before 2026-08-28
            (d / "shards.json").unlink()
            call()
            man = json.loads((d / "shards.json").read_text())
            check("a keyless cache is adopted trust-on-first-use, not re-encoded",
                  all(r["trusted_on_first_use"] for r in man["shards"].values())
                  and man["combined"]["trusted_on_first_use"] is True)
            check("provenance says so rather than claiming verification",
                  teacher.PROVENANCE["smoke"]["shards_trusted_on_first_use"] == 3
                  and "NOT verified" in teacher.PROVENANCE["smoke"]["_note"])
            refuses("verify=True refuses a trust-on-first-use cache instead of trusting it",
                    lambda: call(verify=True), "predate hash recording")
        finally:
            teacher.ENC, teacher.SHARD, teacher.encode = orig

    print(f"\n{'PASS' if not FAILS else 'FAIL: ' + ', '.join(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
