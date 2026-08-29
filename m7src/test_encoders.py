"""Pin the encoder registry against the encode cache that already exists on disk.

The registry made model, pooling, prompt and tokenizer identity configurable. Two things could go
silently wrong as a result, and both would be expensive:

  1. The bge-base cache key could drift, orphaning ~22 GB of encodes and quietly re-encoding
     everything (or worse, reading a stale directory).
  2. Two encoders that produce DIFFERENT vectors could collide on one key. The pre-registry
     cache_key hardcoded "pooling": "cls-l2", so a mean-pooled model would have been stored under a
     key claiming CLS -- exactly the collision that makes a comparison silently wrong.

This replays every meta.json under work/enc/ through the current cache_key() and requires the
directory name to come back identical. Run it after touching encoders.py or teacher.py.

    ../.venv/bin/python test_encoders.py
"""
import json
import sys

import torch

import encoders
from _paths import WORK
from teacher import DT, cache_key

FAIL = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        FAIL.append(msg)


def main():
    print("registry:")
    for name, s in sorted(encoders.REGISTRY.items()):
        print(f"  {name:20s} {s.repo:38s} {s.pooling:4s} dim {s.dim:5d} "
              f"vocab {s.vocab} int8 {s.vocab*s.dim/1e6:.1f} MB")

    # These are properties of the DEFAULT spec, not of whatever M7_ENCODER happens to select.
    # Checking encoders.active() made this gate fail spuriously the moment it ran under another
    # encoder -- which is exactly when a swap most needs the cache keys verified.
    print(f"\nactive encoder is {encoders.active().name!r}; the DEFAULT must still be the M7 teacher "
          f"(so nothing changes unless M7_ENCODER is set):")
    a = encoders.get(encoders.DEFAULT)
    check(a.repo == "BAAI/bge-base-en-v1.5", f"default repo is bge-base, got {a.repo}")
    check(a.pooling == "cls" and a.pooling_key == "cls-l2",
          f"bge pooling_key is the legacy literal 'cls-l2', got {a.pooling_key!r}")
    check(a.tokenizer_id == "bert-wordpiece-30522",
          f"bge tokenizer_id is the legacy literal, got {a.tokenizer_id!r}")

    print("\nreplaying every existing encode cache key:")
    dirs = sorted(p for p in (WORK / "enc").glob("*") if (p / "meta.json").exists())
    if not dirs:
        check(False, "found no existing encode caches to replay -- cannot verify key stability")
    inv = {v: k for k, v in DT.items()}
    for d in dirs:
        m = json.loads((d / "meta.json").read_text())
        # Resolve the Spec from the meta's own POOLING, not from the repo: one repo can carry two
        # Specs (arctic CLS and arctic mean), and this replay runs under whichever encoder happens
        # to be active. The meta records what the encode actually was.
        cands = [sp for sp in encoders.specs_for_repo(m["model"])
                 if sp.pooling_key == m["pooling"]]
        if len(cands) != 1:
            check(False, f"{d.name}: meta pooling {m['pooling']!r} matches {len(cands)} Specs for "
                         f"{m['model']} -- cannot replay unambiguously")
            continue
        key, _ = cache_key(m["name"], m["prefix"], m["max_length"], m["model"], m["revision"],
                           m["corpus_sha256"], inv[m["encode_dtype"]], spec=cands[0])
        check(key == d.name, f"{d.name}" if key == d.name else
              f"{d.name} -> recomputed {key} (KEY DRIFT: existing encode would be orphaned)")

    print("\ndifferent pooling must not collide on one key:")
    args = ("probe", "", 512, None, None, "deadbeef", torch.float16)
    keys = {}
    for name in ("bge-base-en-v1.5", "stella-400M-v5"):
        s = encoders.get(name)
        k, _ = cache_key(args[0], args[1], args[2], s.repo, s.revision, args[5], args[6])
        keys[name] = k
    check(len(set(keys.values())) == 2,
          f"cls and mean pooled encoders get distinct keys: {keys}")
    check(encoders.get("stella-400M-v5").pooling_key == "mean-l2",
          "stella's pooling_key says mean-l2")

    print("\nSpec.cls_id must match the real tokenizer (skipped where not cached offline):")
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoTokenizer
    for name, sp in sorted(encoders.REGISTRY.items()):
        try:
            kw = {"trust_remote_code": True} if sp.trust_remote_code else {}
            tk = AutoTokenizer.from_pretrained(sp.repo, revision=sp.revision, **kw)
        except Exception:
            print(f"  skip {name} (tokenizer not cached locally)")
            continue
        check(tk.cls_token_id == sp.cls_id,
              f"{name} cls_id {sp.cls_id} == tokenizer {tk.cls_token_id}")
        check(tk.vocab_size == sp.vocab or len(tk) == sp.vocab,
              f"{name} vocab {sp.vocab} == tokenizer {tk.vocab_size}/{len(tk)}")

    print("\nevery Spec must be revision-pinned:")
    check(all(s.revision for s in encoders.REGISTRY.values()),
          "no unpinned revisions in the registry")

    print("\ntwo Specs sharing one repo must not be resolved by registry order:")
    shared = [r for r in {sp.repo for sp in encoders.REGISTRY.values()}
              if len(encoders.specs_for_repo(r)) > 1]
    for r in shared:
        names = [sp.name for sp in encoders.specs_for_repo(r)]
        if encoders.active().repo == r:
            check(encoders.by_repo(r) is encoders.active(),
                  f"{r}: the ACTIVE spec wins ({encoders.active().name} of {names})")
        else:
            try:
                encoders.by_repo(r)
                check(False, f"{r} is ambiguous ({names}) and by_repo did NOT raise")
            except KeyError:
                check(True, f"{r} is ambiguous ({names}) and by_repo raises")
    if not shared:
        print("  (no repo is shared by two Specs right now)")

    print("\nan unregistered repo must be refused, not silently defaulted:")
    try:
        encoders.by_repo("some/unregistered-model")
        check(False, "by_repo raised on an unknown repo")
    except KeyError:
        check(True, "by_repo raised on an unknown repo")

    print(f"\n{len(dirs)} caches replayed; {len(FAIL)} failure(s)")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
