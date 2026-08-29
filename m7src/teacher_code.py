"""Pin the teacher's `trust_remote_code` payload, and vendor it.

The doc side of every third-party reproduction of this artifact runs the teacher's OWN Python:
stella declares `auto_map` to `configuration.NewConfig` / `modeling.NewModel`, and we load it with
`trust_remote_code=True`. `AutoModel.from_pretrained(..., revision=REV, trust_remote_code=True)`
does fetch that module from the SAME revision -- the auto_map is same-repo, with no `repo--module`
cross-reference, which is the case that would silently resolve at `main` -- but nothing in this
repo recorded WHICH bytes ran, and a Hub repo can be rewritten, gated or deleted.

So two things, neither of which is a code change to the model:

  1. `results/m7_teacher_code_pin.json` records the sha256 of every file in the pinned snapshot,
     including `modeling.py`, `configuration.py` and the weights. `verify()` re-hashes the local
     snapshot against it and names any file that differs.
  2. `vendor/<name>/` holds a byte-identical copy of the two Python modules, so the code survives
     the Hub. stella's card is MIT and its modeling code carries Apache-2.0 headers (GTE / Alibaba
     Group), both of which permit redistribution with attribution; `vendor/<name>/PROVENANCE.md`
     carries it.

The pin is written once and committed. `freeze.write` verifies it and records the result, so a
freeze cannot claim a teacher whose remote code has moved under it.

    PYTHONPATH=m7src .venv/bin/python m7src/teacher_code.py            # verify
    PYTHONPATH=m7src .venv/bin/python m7src/teacher_code.py --write    # (re)write the pin + vendor
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

import encoders
from _paths import REPO

PIN = REPO / "results" / "m7_teacher_code_pin.json"
VENDOR = REPO / "vendor"
# The remote-code modules an `auto_map` can name. Vendored verbatim.
CODE_FILES = ("modeling.py", "configuration.py")
# Files whose bytes change what the teacher computes. `README.md` is excluded: it is a 170 KB
# model card that upstream edits without touching the model, and pinning it would produce a
# failure that means nothing.
PINNED_GLOBS = ("config.json", "configuration.py", "modeling.py", "modules.json",
                "config_sentence_transformers.json", "sentence_bert_config.json",
                "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt",
                "model.safetensors", "1_Pooling/config.json",
                "2_Dense_1024/config.json", "2_Dense_1024/model.safetensors")


def sha256_file(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def snapshot_dir(spec=None):
    """The local snapshot for the ACTIVE encoder at its pinned revision, without downloading."""
    from huggingface_hub import snapshot_download
    sp = spec or encoders.active()
    return Path(snapshot_download(sp.repo, revision=sp.revision, local_files_only=True))


def fingerprint(spec=None):
    sp = spec or encoders.active()
    d = snapshot_dir(sp)
    files = {}
    for rel in PINNED_GLOBS:
        p = d / rel
        if p.exists():
            files[rel] = {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
    return {"encoder": sp.name, "repo": sp.repo, "revision": sp.revision,
            "trust_remote_code": bool(sp.trust_remote_code),
            "auto_map": (json.loads((d / "config.json").read_text()).get("auto_map")
                         if (d / "config.json").exists() else None),
            "files": files}


def verify(spec=None, strict=True):
    """-> (ok, problems). `strict` also fails when the pin has no entry for the active encoder."""
    sp = spec or encoders.active()
    if not PIN.exists():
        return (not strict), ([f"{PIN.relative_to(REPO)} does not exist; write it with "
                               "`teacher_code.py --write`"] if strict else [])
    pinned = json.loads(PIN.read_text()).get("encoders", {}).get(sp.name)
    if pinned is None:
        return (not strict), ([f"no remote-code pin for encoder {sp.name!r}"] if strict else [])
    problems = []
    if pinned.get("revision") != sp.revision:
        problems.append(f"pin is for revision {str(pinned.get('revision'))[:12]}, active spec "
                        f"wants {str(sp.revision)[:12]}")
    live = fingerprint(sp)
    # A cross-repo auto_map ("other/repo--modeling.Class") resolves at that repo's MAIN, which our
    # revision pin does not cover. stella's is same-repo; assert it, so a future encoder that is
    # not cannot slip through this check silently.
    for k, v in (live.get("auto_map") or {}).items():
        if "--" in str(v):
            problems.append(f"auto_map {k}={v!r} points at ANOTHER repo, which `revision` does not "
                            "pin: that code resolves at main and is not reproducible")
    for rel, rec in pinned.get("files", {}).items():
        got = live["files"].get(rel)
        if got is None:
            problems.append(f"{rel} is missing from the local snapshot")
        elif got["sha256"] != rec["sha256"]:
            problems.append(f"{rel} sha256 {got['sha256'][:12]} != pinned {rec['sha256'][:12]}")
    for rel in live["files"]:
        if rel not in pinned.get("files", {}):
            problems.append(f"{rel} is present locally but not in the pin")
    # the vendored copy must be the same bytes as what actually runs
    for rel in CODE_FILES:
        vp = VENDOR / sp.name / rel
        if rel in live["files"]:
            if not vp.exists():
                problems.append(f"vendor/{sp.name}/{rel} is missing")
            elif sha256_file(vp) != live["files"][rel]["sha256"]:
                problems.append(f"vendor/{sp.name}/{rel} differs from the snapshot's copy")
    return (not problems), problems


def write(spec=None):
    sp = spec or encoders.active()
    fp = fingerprint(sp)
    blob = json.loads(PIN.read_text()) if PIN.exists() else {
        "_what": "sha256 of every file that changes what the teacher computes, at its pinned "
                 "revision, including the trust_remote_code modules. `teacher_code.verify()` "
                 "re-hashes the local snapshot against this and freeze.write records the result.",
        "encoders": {}}
    blob["encoders"][sp.name] = fp
    PIN.write_text(json.dumps(blob, indent=1, sort_keys=True))
    d = snapshot_dir(sp)
    out = VENDOR / sp.name
    out.mkdir(parents=True, exist_ok=True)
    for rel in CODE_FILES:
        if (d / rel).exists():
            shutil.copy2(d / rel, out / rel)
    (out / "PROVENANCE.md").write_text(
        f"# Vendored remote code: `{sp.repo}`\n\n"
        f"Copied byte-identically from the Hugging Face snapshot at revision\n"
        f"`{sp.revision}` on the date of this commit. These are the modules the model's\n"
        f"`auto_map` names, i.e. the code that runs under `trust_remote_code=True`.\n\n"
        "**Why vendored.** The document side of this artifact's every reproduction executes this\n"
        "code. A pinned revision does pin it today, but a Hub repo can be rewritten, gated or\n"
        "deleted, and nothing here should depend on that not happening.\n\n"
        "**Licence.** The model card declares MIT; `modeling.py` and `configuration.py` carry\n"
        "Apache-2.0 headers (Copyright 2024 The GTE Team Authors and Alibaba Group; portions\n"
        "NVIDIA Corporation). Both permit redistribution with attribution, which this file is.\n"
        "No modifications have been made; hashes are in `results/m7_teacher_code_pin.json`.\n\n"
        "**Not imported by this repo.** Loading still goes through `transformers` at the pinned\n"
        "revision; this copy exists so a third party can reconstruct the doc tower if the Hub\n"
        "copy ever changes. `teacher_code.verify()` asserts the two are byte-identical.\n")
    print(json.dumps({k: v["sha256"][:16] for k, v in fp["files"].items()}, indent=1))
    print(f"\npinned {len(fp['files'])} files for {sp.name}; vendored {list(CODE_FILES)} to "
          f"{out.relative_to(REPO)}")
    return fp


if __name__ == "__main__":
    if "--write" in sys.argv:
        write()
    else:
        ok, probs = verify()
        print("teacher remote-code pin: " + ("OK" if ok else "MISMATCH"))
        for p in probs:
            print("  - " + p)
        sys.exit(0 if ok else 1)
