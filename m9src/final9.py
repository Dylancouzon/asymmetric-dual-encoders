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


def seal_protected_paths():
    """A real boundary, not a promise. `--recover` must not be able to open six-set data even if
    something downstream tried; a comment asserting `final_stats` does no such I/O cannot
    establish that from the code under review (Codex final9 re-review, B4)."""
    try:
        sys.path.insert(0, str(REPO / "m8src"))
        import paths_guard
        paths_guard.install()
        return True
    except Exception as e:
        print(f"WARNING: could not install paths_guard ({e!r}); recovery proceeds WITHOUT an "
              f"enforced boundary. Treat its provenance claim as unverified.")
        return False

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


def branch_name():
    return sh("git", "rev-parse", "--abbrev-ref", "HEAD")


def sh_ok(*a):
    r = subprocess.run(a, cwd=REPO, capture_output=True, text=True)
    return r.returncode == 0, (r.stderr or r.stdout).strip()


_LOCK_FH = None


def acquire_lock():
    """flock, NOT O_EXCL-plus-staleness.

    The O_EXCL form had the classic race: two processes both see a dead holder's lock, A replaces
    it, B unlinks A's new lock and takes its own -- two processes scoring the six. It also had to
    guess liveness from `kill(pid, 0)`, where PermissionError means "alive but not ours" and was
    being read as dead. The kernel drops an flock the instant the holder dies, so there is no
    stale state to reclaim and nothing to guess (Codex final9 review, blocker 5).
    """
    global _LOCK_FH
    import fcntl
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    fh = open(LOCK, "a+")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        raise SystemExit(f"REFUSED: {LOCK} is flock-held by a live final9 process. Two concurrent "
                         f"runs could both score the six.")
    fh.truncate(0)
    fh.write(f"{os.getpid()}\n")
    fh.flush()
    _LOCK_FH = fh          # held for the process lifetime; never unlinked


def spent_tag_exists(conf):
    """-> (exists, where). FAILS CLOSED.

    The first version ran `git ls-remote` and treated empty stdout as "absent" -- so an auth
    failure, a network drop or a renamed remote all read as "the access is unspent", which would
    permit a SECOND scoring of the six (Codex final9 review, blocker 2). A non-zero exit is now an
    unknown, and unknown is refused. `origin` is identity-pinned so repointing it cannot
    manufacture an unspent verdict.
    """
    want_url = conf.get("origin_url")
    if not want_url:
        raise SystemExit("REFUSED: final_run_registry.origin_url is not set. The spend receipt's "
                         "witness must be a pinned remote.")
    got_url = sh("git", "remote", "get-url", "origin")
    if got_url != want_url:
        raise SystemExit(f"REFUSED: origin is {got_url!r}, not the registered {want_url!r}. "
                         f"The spend receipt's witness must be the registered remote.")
    r = subprocess.run(["git", "ls-remote", "origin", f"refs/tags/{SPENT_TAG}",
                        f"refs/tags/{SPENT_TAG}^{{}}"], cwd=REPO, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"REFUSED: cannot reach origin to check {SPENT_TAG} "
                         f"({(r.stderr or '').strip()[:200]}). Refusing to assume the access is "
                         f"unspent. Fix connectivity and retry.")
    if r.stdout.strip():
        return True, "origin"
    if sh("git", "tag", "-l", SPENT_TAG):
        return True, "local-only"
    return False, ""


def write_atomic(path, text):
    """write_text truncates first, so a kill mid-write destroys the sole confirmatory result."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    dfd = os.open(str(path.parent), os.O_RDONLY)      # the rename itself must be durable
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


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

    exists, where = spent_tag_exists(conf)
    if exists and where == "origin":
        problems.append(f"{SPENT_TAG} exists on origin: the six-set access is already spent."
                        + (" --infra-retry is inadmissible: the failure did NOT precede the spend."
                           if infra_retry else ""))
    elif exists and where == "local-only":
        # origin absence is positively established at this point (ls-remote exited 0 and empty),
        # so the crash landed between tag creation and push, i.e. BEFORE the durable spend.
        if not infra_retry:
            problems.append(f"a LOCAL-ONLY {SPENT_TAG} exists while origin has none: a crash "
                            f"between tag creation and push. Rerun with --infra-retry, which "
                            f"clears the remnant; the access was never durably spent.")
        else:
            ok, err = sh_ok("git", "tag", "-d", SPENT_TAG)
            print(f"[final9] cleared local-only {SPENT_TAG} remnant ({'ok' if ok else err})")
    # Checked unconditionally: ignoring it under --infra-retry accepted an inconsistent state and
    # would permit rescoring (Codex final9 review, major).
    if RESULT.exists():
        problems.append(f"{RESULT} already exists; a scored result must never be overwritten. "
                        f"Use --recover to recompute decisions from it.")

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
    for cmd in (("git", "add", str(LEDGER)),
                ("git", "commit", "-q", "-m", f"m9: {BEGIN} {freeze_sha[:12]}"),
                ("git", "push", "-q", "origin", "HEAD")):
        ok, err = sh_ok(*cmd)
        if not ok:
            raise SystemExit(f"ABORT before any protected read: {' '.join(cmd)} failed ({err}). "
                             f"No access consumed.")
    # Positively verify the BEGIN entry reached origin: a successful `push` of an UNCHANGED HEAD
    # would otherwise satisfy the check while the ledger entry existed nowhere durable
    # (Codex final9 review, blocker 1).
    if sh("git", "rev-parse", "HEAD") != sh("git", "rev-parse", "origin/" + branch_name()):
        raise SystemExit("ABORT before any protected read: HEAD is not the pushed origin tip "
                         "after the BEGIN commit. No access consumed.")
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
            # Resolves the wedge: a crash between scoring and the ledger digest leaves a spent
            # access whose C1 verdict was never established, blocking the reserved conditional.
            # It performs NO six-set I/O -- it opens only RESULT and the registry, and
            # `final_stats` reads nothing else (verified by inspection; there is no capability
            # boundary enforcing it, which is stated as a limitation).
            sealed = seal_protected_paths()
            exists, where = spent_tag_exists(conf)
            if not RESULT.exists():
                # The access can be spent with no scores on disk (a crash between the tag push and
                # the first durable write). No decision can be reconstructed -- say so plainly
                # rather than refusing with a misleading message (Codex final9 re-review).
                if exists and where == "origin":
                    raise SystemExit(
                        f"ACCESS SPENT, NO RESULT. {SPENT_TAG} is on origin but {RESULT.name} does "
                        f"not exist: the run died between spending the access and writing any "
                        f"score. No decision can be established and none may be invented. This is "
                        f"a documented loss of the six-set access -- disclose it and stop.")
                raise SystemExit("--recover needs an existing results/m9_final_run.json")
            if not (exists and where == "origin"):
                raise SystemExit(f"--recover REFUSED: {SPENT_TAG} is not on origin ({where or 'absent'}). "
                                 f"A result without a durable spend receipt has unverified "
                                 f"provenance and must not be turned into a decision.")
            blob = json.loads(RESULT.read_text())
            fz = json.loads(FREEZE.read_text())
            if blob.get("freeze_sha256") != fz["checkpoint_sha256"]:
                raise SystemExit("--recover REFUSED: the result was produced under a different "
                                 "frozen checkpoint than m9/FREEZE.json names.")
            rec = decide(blob["rows"], conf)
            blob["decision_record"] = rec
            write_atomic(RESULT, json.dumps(blob, indent=1))
            digest = sha256_file(RESULT)
            ledger_append(f"- {datetime.now(timezone.utc).isoformat()} — **{END}** (recover) "
                          f"result sha256 `{digest[:16]}` outcome `{rec['decision']['outcome']}`")
            for cmd in (("git", "add", str(LEDGER), str(RESULT)),
                        ("git", "commit", "-q", "-m", f"m9: {END} (recover) {digest[:12]}"),
                        ("git", "push", "-q", "origin", "HEAD")):
                ok, err = sh_ok(*cmd)
                if not ok:
                    print(json.dumps(rec["decision"], indent=1))
                    print(f"FAILED: {' '.join(cmd)} ({err}). The decision is on disk but NOT "
                          f"durable on origin; it must not be acted on until pushed.")
                    return 3
            print(json.dumps(rec["decision"], indent=1))
            print(f"[final9] recovery durable on origin; paths_guard={'on' if sealed else 'OFF'}")
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
        # The lock file is deliberately NOT unlinked: unlinking it while our descriptor is still
        # open would let another process lock a fresh inode and score concurrently (Codex final9
        # re-review). The kernel releases the flock when this process dies.
        pass


if __name__ == "__main__":
    sys.exit(main())
