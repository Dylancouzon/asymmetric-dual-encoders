"""The one-shot guard, tested. Codex review #4 found there were no tests for it at all.

`final_run.guard` is the only thing standing between a session and a second confirmatory reading
of the six. It is pure bookkeeping over git output, the ledger and the result file, so it can be
tested by stubbing `sh`/`sh_raw` -- no repo state, no scoring, no GPU.

    ../.venv/bin/python test_final_guard.py
"""
import json
import sys
import tempfile
from pathlib import Path

import final_run as F

FAILS = []
HEAD = "a" * 40
TAG_OBJ = "b" * 40          # an ANNOTATED tag's own object hash, which is NOT the commit
TABLE = "c" * 64


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {name}" + (f" -- {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


class Git:
    """Stub of the two git shells `guard` uses. `annotated` decides whether the pushed tag is an
    annotated object (peeled ref present) or a lightweight one. `spent` / `spent_local` model the
    m7-six-spent receipt on origin / in local tags."""

    def __init__(self, dirty=(), head=HEAD, remote=HEAD, tag=HEAD, annotated=True, diff=(),
                 spent=False, spent_local=False):
        self.dirty, self.head, self.remote = list(dirty), head, remote
        self.tag, self.annotated, self.diff = tag, annotated, list(diff)
        self.spent, self.spent_local = spent, spent_local

    def raw(self, *a):
        if a[:2] == ("git", "status"):
            return "".join(f" M {p}\n" for p in self.dirty)
        if a[:2] == ("git", "ls-remote"):
            out = ""
            for ref in a[3:]:
                if F.SPENT_TAG in ref:      # the spent receipt is a different tag namespace
                    if self.spent:
                        out += f"{'9' * 40}\t{ref}\n"
                elif ref.endswith("^{}"):
                    if self.annotated and self.tag:
                        out += f"{self.tag}\trefs/tags/{F.FREEZE_TAG}^{{}}\n"
                elif ref.startswith("refs/tags/") and self.tag:
                    out += f"{TAG_OBJ if self.annotated else self.tag}\trefs/tags/{F.FREEZE_TAG}\n"
                elif ref.startswith("refs/heads/"):
                    out += f"{self.remote}\t{ref}\n"
            return out
        return ""

    def stripped(self, *a):
        if a[:2] == ("git", "rev-parse"):
            return self.head
        if a[:3] == ("git", "diff", "--name-only"):
            return "\n".join(self.diff)
        if a[:3] == ("git", "tag", "-l"):
            return F.SPENT_TAG if self.spent_local and a[3] == F.SPENT_TAG else ""
        return self.raw(*a).strip()


def run_guard(tmp, git, ledger_text="", result=None, **kw):
    """-> (ok, message). Never lets a refusal escape as SystemExit(2)."""
    F.LEDGER = Path(tmp) / "LEDGER.md"
    F.LEDGER.write_text(ledger_text)
    F.OUT = Path(tmp) / "m7_final_run.json"
    if result is None:
        F.OUT.unlink(missing_ok=True)
    else:
        F.OUT.write_text(json.dumps(result))
    F.sh, F.sh_raw = git.stripped, git.raw
    fz = {"table_sha256": TABLE}
    out = {}
    real_print = __builtins__["print"] if isinstance(__builtins__, dict) else __builtins__.print
    import builtins
    builtins.print = lambda *a, **k: out.setdefault("msg", " ".join(map(str, a)))
    try:
        F.guard(kw.pop("freeze_hash", HEAD), kw.pop("infra_retry", False), "m7-query-encoder", fz,
                untouched_only=kw.pop("untouched_only", False))
        return True, ""
    except SystemExit:
        return False, out.get("msg", "")
    finally:
        builtins.print = real_print


BEGIN = f"- FINAL-RUN-BEGIN freeze={HEAD} table={TABLE}\n"
DONE = "- FINAL-RUN complete in 10s\n"


def main():
    saved = (F.LEDGER, F.OUT, F.sh, F.sh_raw)
    with tempfile.TemporaryDirectory() as td:
        try:
            print("the happy path")
            ok, msg = run_guard(td, Git())
            check("a clean tree at the tagged commit passes", ok, msg)

            print("\nthe annotated tag must PEEL to the commit")
            # This is the one that would have stopped the final run dead: `git ls-remote
            # refs/tags/X` returns the TAG OBJECT for an annotated tag, and the documented
            # procedure is `git tag -a`.
            ok, _ = run_guard(td, Git(annotated=True))
            check("an ANNOTATED tag is accepted (peeled ref used)", ok)
            ok, _ = run_guard(td, Git(annotated=False))
            check("a lightweight tag is accepted too", ok)
            ok, msg = run_guard(td, Git(tag=""))
            check("no pushed tag is refused", not ok and "no pushed tag" in msg, msg)
            ok, msg = run_guard(td, Git(tag="d" * 40, annotated=True))
            check("a tag pointing elsewhere is refused", not ok and "the tag is authoritative" in msg,
                  msg)

            print("\nthe access is spent when the RESULT exists, not when the ledger says so")
            spent = {"six": {"int8-table": {}}, "freeze": {"table_sha256": TABLE}}
            ok, msg = run_guard(td, Git(), ledger_text=BEGIN + DONE, result=spent)
            check("a second plain run is refused", not ok and "SPENT" in msg, msg)
            # the exact attack: delete the COMPLETE line, keep one BEGIN, pass --infra-retry
            ok, msg = run_guard(td, Git(dirty=["m7/LEDGER.md"]), ledger_text=BEGIN, result=spent,
                                infra_retry=True)
            check("deleting the ledger's COMPLETE line does not buy a retry",
                  not ok and "SPENT" in msg, msg)
            # a result file that exists but is unparseable is also spent, not ignored
            F.OUT = Path(td) / "m7_final_run.json"
            F.OUT.write_text("{truncated")
            F.sh, F.sh_raw = Git().stripped, Git().raw
            check("an unparseable result counts as spent",
                  F.six_already_scored({"table_sha256": TABLE})[0])

            print("\ninfra-retry")
            ok, msg = run_guard(td, Git(dirty=["m7/LEDGER.md"]), ledger_text=BEGIN,
                                infra_retry=True)
            check("one retry after an aborted run is allowed", ok, msg)
            ok, msg = run_guard(td, Git(dirty=["m7/LEDGER.md"]), ledger_text=BEGIN + BEGIN,
                                infra_retry=True)
            check("a THIRD begin is refused (the cap is two)",
                  not ok and "cap is" in msg, msg)
            ok, msg = run_guard(td, Git(dirty=["m7/LEDGER.md"], head="e" * 40),
                                ledger_text=BEGIN, infra_retry=True, freeze_hash="e" * 40)
            check("a retry on a different commit is refused", not ok and "same commit" in msg, msg)
            ok, msg = run_guard(td, Git(dirty=["m7/LEDGER.md"], diff=["m7src/train.py"]),
                                ledger_text=BEGIN, infra_retry=True)
            check("a retry with a code change is refused",
                  not ok and "infrastructure only" in msg, msg)
            ok, msg = run_guard(td, Git(dirty=["m7src/train.py"]), ledger_text=BEGIN,
                                infra_retry=True)
            check("a dirty source file is refused even under retry",
                  not ok and "not clean beyond" in msg, msg)

            print("\nuntouched-only")
            digest = F.sha({"six": spent["six"], "confirmatory": None, "holm": None})
            full = dict(spent, confirmatory=None, holm=None)
            led = BEGIN + DONE + f"- FINAL-RUN-SIX-SHA256 {digest}\n"
            ok, msg = run_guard(td, Git(dirty=["results/m7_final_run.json"]), ledger_text=led,
                                result=full, untouched_only=True)
            check("resuming the tail is allowed", ok, msg)
            # ... and it does NOT need the completion marker, so a crash between the result write
            # and the ledger append cannot wedge the run
            ok, msg = run_guard(td, Git(dirty=["results/m7_final_run.json"]),
                                ledger_text=BEGIN + f"- FINAL-RUN-SIX-SHA256 {digest}\n",
                                result=full, untouched_only=True)
            check("resuming works without the COMPLETE marker (no wedge)", ok, msg)
            edited = dict(full, six={"int8-table": {"scifact": {"q1": 0.99}}})
            ok, msg = run_guard(td, Git(dirty=["results/m7_final_run.json"]), ledger_text=led,
                                result=edited, untouched_only=True)
            check("an edited confirmatory block is refused", not ok and "has been edited" in msg,
                  msg)
            ok, msg = run_guard(td, Git(dirty=["results/m7_final_run.json"]),
                                ledger_text=BEGIN + DONE, result=full, untouched_only=True)
            check("resuming with no recorded digest is refused",
                  not ok and "FINAL-RUN-SIX-SHA256" in msg, msg)
            ok, msg = run_guard(td, Git(), ledger_text=BEGIN + DONE, untouched_only=True)
            check("resuming with no result file is refused", not ok and "have not been scored" in msg,
                  msg)
            ok, msg = run_guard(td, Git(dirty=["results/m7_final_run.json"]), ledger_text=led,
                                result=dict(full, freeze={"table_sha256": "f" * 64}),
                                untouched_only=True)
            check("resuming against a different table is refused",
                  not ok and "different table" in msg, msg)
            ok, msg = run_guard(td, Git(dirty=["results/m7_final_run.json"]), ledger_text=led,
                                result=full, untouched_only=True, infra_retry=True)
            check("--untouched-only and --infra-retry together are refused",
                  not ok and "mutually exclusive" in msg, msg)

            print("\nthe spent receipt (the tag survives a deleted result and an edited ledger)")
            ok, msg = run_guard(td, Git(spent=True))
            check("a pushed spent tag refuses a plain run", not ok and F.SPENT_TAG in msg, msg)
            # the exact BLOCKER-2 attack: result deleted, COMPLETE line trimmed, --infra-retry --
            # the receipt on origin still refuses it
            ok, msg = run_guard(td, Git(dirty=["m7/LEDGER.md"], spent=True), ledger_text=BEGIN,
                                infra_retry=True)
            check("deleting the result + trimming the ledger does not beat the receipt",
                  not ok and F.SPENT_TAG in msg, msg)
            ok, msg = run_guard(td, Git(spent_local=True))
            check("a local-only spent tag refuses too", not ok and F.SPENT_TAG in msg, msg)
            digest2 = F.sha({"six": spent["six"], "confirmatory": None, "holm": None})
            ok, msg = run_guard(td, Git(dirty=["results/m7_final_run.json"], spent=True),
                                ledger_text=BEGIN + DONE + f"- FINAL-RUN-SIX-SHA256 {digest2}\n",
                                result=dict(spent, confirmatory=None, holm=None),
                                untouched_only=True)
            check("--untouched-only still works after the receipt exists", ok, msg)

            print("\nthe exclusive lock")
            F.LOCK = Path(td) / ".final-run.lock"
            F.LOCK.unlink(missing_ok=True)
            F.acquire_lock()
            check("the lock is created", F.LOCK.exists())
            import os as _os
            F.LOCK.write_text(f"{_os.getpid()}\n")     # a LIVE pid holds it
            try:
                F.acquire_lock()
                check("a live holder refuses a second process", False)
            except SystemExit as e:
                check("a live holder refuses a second process", "holds" in str(e))
            F.LOCK.write_text("999999999\n")            # a dead pid: stale, self-clears
            try:
                F.acquire_lock()
                check("a stale lock (dead pid) self-clears", F.LOCK.exists())
            except SystemExit as e:
                check("a stale lock (dead pid) self-clears", False, str(e))
            F.LOCK.unlink(missing_ok=True)

            print("\nmiscellaneous refusals")
            ok, msg = run_guard(td, Git(dirty=["m7src/train.py"]))
            check("a dirty tree is refused", not ok and "not clean" in msg, msg)
            ok, msg = run_guard(td, Git(remote="9" * 40))
            check("an unpushed freeze commit is refused", not ok and "not pushed" in msg, msg)
            ok, msg = run_guard(td, Git(head="9" * 40))
            check("HEAD != freeze commit is refused", not ok and "!= freeze commit" in msg, msg)
        finally:
            F.LEDGER, F.OUT, F.sh, F.sh_raw = saved

    print(f"\n{'PASS' if not FAILS else 'FAIL: ' + ', '.join(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
