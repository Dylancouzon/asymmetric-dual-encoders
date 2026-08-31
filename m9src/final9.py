"""M9.4 — the single six-set transaction.

Runs ONCE. The data it reads can never be re-read, so the ordering below is not stylistic: the
spent receipt is pushed to origin BEFORE the first protected byte is opened, which makes
"was the access consumed?" a question with a durable external answer rather than a judgement call.

Spec: `m9/FINAL_LOCK.md`. Constants: `m9/final_run_registry.json` (NOT m9/registry.json -- the
build adapter permits only owner_rulings there). Statistics: `m9src/final_stats.py`.

State machine (FINAL_LOCK "Access state machine"):
  1 acquire process lock; verify guard, freeze hashes, clean tree, HEAD pushed
  2 append FINAL-RUN-BEGIN to the ledger, commit, PUSH
  3 create and PUSH the annotated `m9-six-spent` tag -- push failure aborts HERE, before any read
  4 open six-set queries/qrels; bridge; score; write results/m9_final_run.json
  5 compute decisions; append FINAL-RUN-END + digest; commit; push

Any failure at or after step 3 consumes the access. `--infra-retry` is admissible ONLY when the
tag is absent from origin; that is a fact, not a judgement. `--recover` recomputes decisions from
an already-written result and NEVER re-reads the six.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "m9src"))
sys.path.insert(0, str(REPO / "m7src"))

from m9src import final_stats                                   # noqa: E402

LOCK = REPO / "work" / "m9final.lock"
RESULT = REPO / "results" / "m9_final_run.json"
LEDGER = REPO / "m9" / "LEDGER.md"
FREEZE = REPO / "m9" / "FREEZE.json"
PERQUERY = REPO / "results" / "perquery.json"
SPENT_TAG = "m9-six-spent"
BEGIN = "FINAL-RUN-BEGIN"
END = "FINAL-RUN-END"


def sh(*a):
    return subprocess.run(a, cwd=REPO, capture_output=True, text=True).stdout.strip()


def sh_ok(*a):
    r = subprocess.run(a, cwd=REPO, capture_output=True, text=True)
    return r.returncode == 0, (r.stderr or r.stdout).strip()


def acquire_lock():
    """One process at a time. A stale lock from a dead pid is reclaimed once; a live one refuses."""
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    for attempt in (1, 2):
        try:
            fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{os.getpid()}\n".encode())
            os.close(fd)
            return
        except FileExistsError:
            try:
                pid = int(LOCK.read_text().split()[0])
            except (ValueError, IndexError, OSError):
                pid = None
            alive = False
            if pid is not None:
                try:
                    os.kill(pid, 0)
                    alive = True
                except (ProcessLookupError, PermissionError):
                    alive = False
            if alive:
                raise SystemExit(f"REFUSED: pid {pid} holds {LOCK}. Two concurrent runs could "
                                 f"both score the six.")
            if attempt == 1:
                print(f"[final9] removing stale lock (pid {pid} is gone)")
                LOCK.unlink(missing_ok=True)
    raise SystemExit(f"REFUSED: could not acquire {LOCK}")


def spent_tag_exists():
    """-> (exists, where). origin is the EXTERNAL witness and is checked first: a local-only tag
    could be deleted, but the pushed one is what makes the spend durable."""
    remote = sh("git", "ls-remote", "origin", f"refs/tags/{SPENT_TAG}",
                f"refs/tags/{SPENT_TAG}^{{}}")
    if remote:
        return True, "origin"
    if sh("git", "tag", "-l", SPENT_TAG):
        return True, "local"
    return False, ""


def write_atomic(path, text):
    """write_text truncates first, so a kill mid-write destroys the sole confirmatory result."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- step 1: preflight

def preflight(conf, infra_retry):
    """Everything static that must hold BEFORE the six are touched.

    It must NOT open six-set queries or qrels -- a failing preflight that read labels could be
    repeated indefinitely, reading protected data without ever consuming the access
    (Codex final-lock review, blocker 4). Paths, hashes and sizes only.
    """
    problems = []

    exists, where = spent_tag_exists()
    if exists and not infra_retry:
        problems.append(f"{SPENT_TAG} exists on {where}: the six-set access is already spent.")
    if exists and infra_retry:
        problems.append(f"--infra-retry is inadmissible: {SPENT_TAG} exists on {where}, so the "
                        f"failure did NOT precede the spend.")
    if RESULT.exists() and not infra_retry:
        problems.append(f"{RESULT} already exists; use --recover to recompute decisions from it.")

    if sh("git", "status", "--porcelain"):
        problems.append("working tree is dirty; the freeze commit must be clean.")
    if sh("git", "rev-parse", "HEAD") != sh("git", "rev-parse", "@{u}"):
        problems.append("HEAD is not pushed to its upstream.")

    if not FREEZE.exists():
        problems.append(f"{FREEZE} is missing: the candidate is not frozen.")
    else:
        fz = json.loads(FREEZE.read_text())
        ckpt = REPO / fz["checkpoint"]
        if not ckpt.exists():
            problems.append(f"frozen checkpoint {ckpt} is missing.")
        elif sha256_file(ckpt) != fz["checkpoint_sha256"]:
            problems.append(f"checkpoint sha256 does not match {FREEZE.name}.")

    # The comparator snapshot is the SOURCE of C1/C2's b-side. Opening it read-only does not pin
    # its contents, so the digest is verified explicitly (final-lock blocker 6).
    want = conf["comparator_source"]["sha256"]
    got = sha256_file(PERQUERY)
    if got != want:
        problems.append(f"{PERQUERY.name} sha256 {got[:12]} != registered {want[:12]}.")

    if conf.get("ratified_by_owner") is not True:
        problems.append("final_run_registry.ratified_by_owner is not true: the final-lock "
                        "amendment has not been ratified by the owner.")
    return problems


# ---------------------------------------------------------------- steps 2-3: spend, durably

def ledger_append(line):
    with open(LEDGER, "a") as f:
        f.write(line + "\n")


def spend_access(freeze_sha):
    """Push the receipt BEFORE the first protected read. Push failure aborts before any access."""
    stamp = datetime.now(timezone.utc).isoformat()
    ledger_append(f"\n- {stamp} — **{BEGIN}** freeze `{freeze_sha[:12]}` "
                  f"pid {os.getpid()} host `{os.uname().nodename}`")
    ok, err = sh_ok("git", "add", str(LEDGER))
    ok2, err2 = sh_ok("git", "commit", "-q", "-m", f"m9: {BEGIN} {freeze_sha[:12]}")
    ok3, err3 = sh_ok("git", "push", "-q", "origin", "HEAD")
    if not ok3:
        raise SystemExit(f"ABORT before any protected read: could not push the {BEGIN} ledger "
                         f"entry ({err3}). No access consumed.")
    ok4, err4 = sh_ok("git", "tag", "-a", SPENT_TAG, "-m",
                      f"M9 six-set access spent {stamp} freeze {freeze_sha[:12]}")
    if not ok4:
        raise SystemExit(f"ABORT before any protected read: could not create {SPENT_TAG} "
                         f"({err4}). No access consumed.")
    ok5, err5 = sh_ok("git", "push", "-q", "origin", f"refs/tags/{SPENT_TAG}")
    if not ok5:
        sh("git", "tag", "-d", SPENT_TAG)
        raise SystemExit(f"ABORT before any protected read: could not PUSH {SPENT_TAG} ({err5}). "
                         f"Local tag removed; no access consumed. The receipt must be durable on "
                         f"origin before the six are opened.")
    print(f"[final9] access SPENT and pushed: {SPENT_TAG}. Everything from here is irreversible.")


# ---------------------------------------------------------------- step 5: decide

def decide(rows, conf):
    """Apply the locked claim table. Computes no scores; reads only the contrast record."""
    c1 = (conf["contrasts"]["C1"]["a"], conf["contrasts"]["C1"]["b"])
    c2 = (conf["contrasts"]["C2"]["a"], conf["contrasts"]["C2"]["b"])
    rec = final_stats.run_contrasts(rows, c1, c2, conf=conf)
    p1 = rec["contrasts"]["C1"]["passes"]
    p2 = rec["contrasts"]["C2"]["passes"]
    if p1 and p2:
        outcome, release, claim = "release+aim", True, "verbatim headline permitted"
    elif p1:
        outcome, release, claim = "release", True, "did not resolve above the LEAF system"
    elif p2:
        # C2 does not gate the ship and C1 does not gate the claim (final-lock review, major 1).
        outcome, release, claim = "aim-only", False, "aim claim permitted under the qualification"
    else:
        outcome, release, claim = "neither", False, "measurement; no claim"
    rec["decision"] = {"outcome": outcome, "release": release, "claim": claim,
                       "C1_passes": p1, "C2_passes": p2,
                       "headline_verbatim": conf["headline_verbatim"],
                       "forbidden_words": conf["forbidden_words"]}
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infra-retry", action="store_true",
                    help="admissible ONLY when the spent tag is absent from origin")
    ap.add_argument("--recover", action="store_true",
                    help="recompute decisions from an existing result; never re-reads the six")
    ap.add_argument("--preflight-only", action="store_true")
    a = ap.parse_args()

    conf = final_stats.cfg()
    acquire_lock()
    try:
        if a.recover:
            if not RESULT.exists():
                raise SystemExit("--recover needs an existing results/m9_final_run.json")
            blob = json.loads(RESULT.read_text())
            rec = decide(blob["rows"], conf)
            blob["decision_record"] = rec
            write_atomic(RESULT, json.dumps(blob, indent=1))
            print(json.dumps(rec["decision"], indent=1))
            return 0

        problems = preflight(conf, a.infra_retry)
        if problems:
            print("FINAL RUN REFUSED:\n  " + "\n  ".join(problems))
            return 2
        print("[final9] preflight clean.")
        if a.preflight_only:
            return 0

        raise SystemExit(
            "SCORING PATH NOT IMPLEMENTED. Steps 4's encoding is deliberately unwritten until the "
            "GPU is free and it can be validated: encode the six with stella (documents) and the "
            "frozen nano checkpoint (queries), score the bge-small anchor for the bridge, require "
            "zero missing/extra/reordered qids and max per-query |dNDCG| <= 3e-4 vs the frozen "
            "row, DISCARD the anchor row (validation only), then build `rows` from "
            "results/perquery.json for the b-sides and call decide(). See m9/FINAL_LOCK.md and "
            "reuse m7src/final_run.py verify_and_load/score_set.")
    finally:
        LOCK.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
