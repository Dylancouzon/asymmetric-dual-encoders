"""M13 — fill an arm record's DEFERRED DEV-6 read from its final checkpoint, on the box.

    .venv/bin/python m13src/dev6_from_checkpoint.py E-bs32 [--device cuda|cpu]

`m10src/run_arm.py --dev6 defer` trains an arm without DEV-6's ~35 GB of document caches on the
machine (the cloud E arms; `m13/SHIP_LIST.md`) and writes `dev6: {"deferred": true, ...}` naming
the final checkpoint's sha. This script performs that one read here, from the identical bytes:

  1. both published copies of the record (`work/m10arms/<arm>/record.json` and
     `results/m10_arm_<arm>.json`) exist, agree, are `complete`, not a smoke, and carry a deferred
     `dev6` — a record that already holds a DEV-6 row refuses: DEV-6 is read ONCE;
  2. the checkpoint the record names is on disk and hashes to `final_checkpoint_sha256`
     (= `checkpoints.cycle3.sha256` = `dev6.checkpoint_sha256`);
  3. the student is rebuilt from the record's own recipe (`student`, `n_layers`, `head`), the
     checkpoint's `model` state is loaded, the parameter count must match the record's;
  4. `run_arm.dev6(model)` — the same function the inline read uses — and the record is rewritten
     atomically to both paths with `dev6` filled and a `filled_from_checkpoint` provenance block
     (host, device, git HEAD, time, checkpoint sha, code sha).

Nothing else in the record changes. No six-set, reserved or LoTTE surface is touched: DEV-6 has no
protected kind. The arm records are not sha-bound anywhere (`contrasts` hashes family F's records
for the F verdict; the E arms feed E1 through their per-query COV files), so filling `dev6` after
the fact invalidates no binding.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for _p in ("m7src", "m9src", "m10src", "m13src"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

import nano10 as N                              # noqa: E402
import run_arm as R                             # noqa: E402

refuse = R.refuse
sha256_file = R.sha256_file
CODE_IDENTITY_FILES = ("m13src/dev6_from_checkpoint.py", "m10src/run_arm.py", "m10src/nano10.py")


def code_identity():
    h = hashlib.sha256()
    for name in CODE_IDENTITY_FILES:
        h.update((REPO / name).read_bytes())
    return h.hexdigest()


def _is_sha(x):
    return isinstance(x, str) and len(x) == 64 and all(c in "0123456789abcdef" for c in x)


def load_records(arm):
    """-> (record, work path, published path). Both copies must exist and agree."""
    _out, rec_path, results_path = R.record_paths(arm, False)
    for p in (rec_path, results_path):
        if not p.exists():
            refuse(f"no arm record at {p}; a deferred DEV-6 can only fill a record that exists at "
                   f"both published paths (copy the arm's work directory back from the instance)")
    a, b = json.loads(rec_path.read_text()), json.loads(results_path.read_text())
    if a != b:
        refuse(f"{rec_path} and {results_path} differ; the runner writes one record to both paths, "
               f"so one of them was edited. Resolve that before filling DEV-6.")
    return a, rec_path, results_path


def check_record(rec, arm):
    if rec.get("arm") != arm:
        refuse(f"the record is for arm {rec.get('arm')!r}, not {arm!r}")
    if rec.get("status") != "complete" or rec.get("complete") is not True:
        refuse(f"{arm}: status {rec.get('status')!r}, complete {rec.get('complete')!r}; a failed "
               f"arm has no final checkpoint and gets no DEV-6 read (rules.arm_failure)")
    if rec.get("smoke"):
        refuse(f"{arm}: a SMOKE record never reads DEV-6")
    d6 = rec.get("dev6")
    if not (isinstance(d6, dict) and d6.get("deferred") is True):
        refuse(f"{arm}: `dev6` is not a deferral ({str(d6)[:80]!r}); DEV-6 is read ONCE at the "
               f"final checkpoint and this record already carries its read (or never deferred it)")
    ck, sha = rec.get("final_checkpoint"), rec.get("final_checkpoint_sha256")
    c3 = ((rec.get("checkpoints") or {}).get("cycle3") or {}).get("sha256")
    if not ck or not _is_sha(sha) or c3 != sha:
        refuse(f"{arm}: final_checkpoint {ck!r} / sha {str(sha)[:12]!r} / cycle3 sha "
               f"{str(c3)[:12]!r} do not name one final checkpoint")
    if d6.get("checkpoint_sha256") not in (None, sha):
        refuse(f"{arm}: the deferral names checkpoint {str(d6.get('checkpoint_sha256'))[:12]} but "
               f"the record's final checkpoint is {sha[:12]}")
    for f in ("student", "n_layers", "head"):
        if f not in (rec.get("recipe") or {}):
            refuse(f"{arm}: recipe lacks {f!r}; the student cannot be rebuilt from the record")
    return ck, sha


def verify_checkpoint(rec, arm):
    """-> (path, bytes). The bytes are read ONCE and hashed; `rebuild_student` deserialises that
    same buffer, so a file replaced between hashing and loading cannot be scored (Astra
    2026-09-10, finding 3)."""
    ck, sha = rec["final_checkpoint"], rec["final_checkpoint_sha256"]
    p = REPO / ck
    if not p.exists():
        refuse(f"{arm}: checkpoint {p} is not on this machine; copy the arm's `cycle3.pt` back to "
               f"the same relative path first")
    data = p.read_bytes()
    got = hashlib.sha256(data).hexdigest()
    if got != sha:
        refuse(f"{arm}: {p} hashes {got[:12]}, the record says {sha[:12]}; refusing to read DEV-6 "
               f"on bytes that are not the published final checkpoint")
    return p, data


def note_attempt(rec_path, arm, sha):
    """A DEV-6 attempt line beside the record, written BEFORE the read: a crash between the read
    and the rewrite leaves the deferral in place, and the next fill discloses that it is the
    second attempt rather than silently looking like the first (finding 7). DEV-6 is a
    development surface, so a repeated read is disclosed, not refused."""
    p = Path(rec_path).with_name("dev6_attempts.jsonl")
    with open(p, "a") as fh:
        fh.write(json.dumps({"arm": arm, "checkpoint_sha256": sha, "host": platform.node(),
                             "started_at": datetime.now(timezone.utc).strftime(
                                 "%Y-%m-%dT%H:%M:%S%z")}) + "\n")
    with open(p) as fh:
        return sum(1 for _ in fh)


def rebuild_student(rec, data, device):
    import io
    import torch
    r = rec["recipe"]
    model = N.Nano10(r["student"], n_layers=int(r["n_layers"]), head=r["head"])
    blob = torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
    sd = blob.get("model", blob) if isinstance(blob, dict) else blob
    model.load_state_dict(sd)
    if not model.under_cap():
        refuse(f"{rec['arm']}: {model.n_params():,} parameters, over the registered cap")
    if rec.get("params") is not None and int(rec["params"]) != model.n_params():
        refuse(f"{rec['arm']}: the record says {rec['params']:,} parameters, the rebuilt student "
               f"has {model.n_params():,}; the recipe in the record does not describe it")
    model.eval()
    return model.to(device)


def run(arm, *, device="cuda", verbose=True):
    """-> the rewritten record."""
    import torch
    if device == "cuda" and not torch.cuda.is_available():
        refuse("--device cuda but no CUDA device is visible; a CUDA torch installation is not a GPU")
    rec, rec_path, results_path = load_records(arm)
    check_record(rec, arm)
    ck_path, data = verify_checkpoint(rec, arm)
    deferral = rec["dev6"]
    t0 = time.time()
    model = rebuild_student(rec, data, device)
    del data
    if verbose:
        print(f"{arm}: student {rec['recipe']['student']} rebuilt from {ck_path.name} "
              f"({rec['final_checkpoint_sha256'][:12]}), {model.n_params():,} params on {device}",
              flush=True)
    attempts = note_attempt(rec_path, arm, rec["final_checkpoint_sha256"])
    if attempts > 1 and verbose:
        print(f"{arm}: DEV-6 attempt {attempts}; an earlier attempt did not rewrite the record",
              flush=True)
    d6 = R.dev6(model, verbose=verbose)
    d6["filled_from_checkpoint"] = {
        "deferred_by_runner": deferral,
        "script": "m13src/dev6_from_checkpoint.py",
        "attempts_including_this": attempts,
        "host": platform.node(), "device": device,
        "git_head": R.git_head(),
        "read_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S%z"),
        "checkpoint": rec["final_checkpoint"], "checkpoint_sha256": rec["final_checkpoint_sha256"],
        "code_sha256": code_identity(), "seconds": round(time.time() - t0, 1),
        "_note": "DEV-6 read ONCE, on the box, from the identical final checkpoint bytes the "
                 "runner published; the runner itself read no DEV-6 (`--dev6 defer`)."}
    rec["dev6"] = d6
    for w in R.write_record(rec, rec_path, results_path):
        if verbose:
            print(f"wrote {w}", flush=True)
    if verbose:
        print(f"{arm}: DEV-6 macro {d6.get('macro')} (retention {d6.get('retention')}) in "
              f"{time.time() - t0:.0f}s", flush=True)
    return rec


def build_argparser():
    ap = argparse.ArgumentParser(description="fill a deferred DEV-6 read from an arm's final "
                                             "checkpoint (m10src/run_arm.py --dev6 defer)")
    ap.add_argument("arm")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--quiet", action="store_true")
    return ap


def main(argv=None):
    a = build_argparser().parse_args(argv)
    run(a.arm, device=a.device, verbose=not a.quiet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
