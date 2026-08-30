"""M9 registration gate: a session manifest, per-run tokens, and verification at read time.

Three Codex passes shaped this. Pass 1 (BLOCKER-3): a write-time-only check lets a session start
under lock A, watch the console, amend and push lock B, and write the artifact under B. Pass 2
(BLOCKER-3 again): per-run tokens fix one process interval but not the *experiment* — arms from
different commits could still be combined, a boolean in a JSON file was trusted at read time, and
`-smoke` bypassed every check and could write anywhere.

So there are three layers now:

1. **A session manifest** (`work/m9tokens/SESSION.json`) written before the first arm: the lock
   commit and the complete fingerprint of every file a result may depend on. Every later arm must
   match it exactly, so the whole screen is one frozen experiment, not nine independent ones.
2. **A per-run token**, one use, consumed atomically at write time. The session is keyed on the
   FINGERPRINT, never on HEAD: committing an arm's own result moves HEAD, and that must not void
   the arm that produced it.
3. **Read-time verification**: `eligible()` recomputes rather than trusting the recorded boolean,
   and requires the artifact's session to be the current one.

Diagnostics (`-smoke` / `-diag`) skip the state checks — they exist to be run against a dirty
tree — but they are confined to a separate filename namespace that the decision loader cannot
address, and they are stamped ineligible.

Note the asymmetry that makes the launch check worth having at all (m8/CODEMAP.md pitfall 24): the
launch check fails fast and cheap, the write check fails after the entire job has run.
"""
import hashlib
import json
import subprocess
import time

import m9base
from m9base import REPO, M9, WORK

GUARDED = ("m9/LEDGER.md", "m9/registry.json")
BRANCH = "m9-work"
CODE = tuple(f"m9src/{n}.py" for n in
             ("m9base", "data", "eval9", "fp16_gate", "guard9", "nano", "port", "screen",
              "screen_stats", "teacher9", "warmfit", "lock_constants", "bridge_dryrun")) + \
       ("m7src/evalkit.py", "m7src/teacher.py", "m7src/devsuite.py", "m7src/dev_eval.py",
        "m7src/pool.py", "m7src/heldout.py", "m7src/train.py", "m7src/mix.py",
        "m8src/paths_guard.py")
DATA = ("work/m9_screen_queries.json", "work/m9_screen_rows.npy",
        "work/decontam/banned_pool_rows.npy", "results/m9_lock_constants.json")
TOKENS = WORK / "m9tokens"
SESSION = TOKENS / "SESSION.json"
SMOKE_DIR = WORK / "m9smoke"


class NotLocked(RuntimeError):
    pass


def _sha(p):
    p = REPO / p
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def _git(*args):
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True).stdout.strip()


def registry():
    return json.loads((M9 / "registry.json").read_text())


def fingerprint():
    """Every file a result is allowed to depend on, hashed."""
    return {"lock": {f: _sha(f) for f in GUARDED},
            "code": {f: _sha(f) for f in CODE},
            "data": {f: _sha(f) for f in DATA}}


def fp_sha(fp):
    return hashlib.sha256(json.dumps(fp, sort_keys=True).encode()).hexdigest()


def check_state():
    """-> (problems, head, branch). Includes comparing the starting DATA files against the hashes
    the registry itself pins, so a modified pool cannot simply be blessed as 'current'."""
    problems = []
    for f in GUARDED + DATA:
        if not (REPO / f).exists():
            problems.append(f"{f} is missing -- M9.0 is not locked")
    if problems:
        return problems, "", ""
    r = registry()
    expect = {"work/m9_screen_queries.json": r["data"]["screen_query_pool_sha256"]}
    for f, want in expect.items():
        got = _sha(f)
        if want and got != want:
            problems.append(f"{f} hashes {got[:12]}, the registry pins {want[:12]}")
    dirty = _git("status", "--porcelain", "--", *GUARDED, *CODE)
    if dirty:
        problems.append("uncommitted lock or code files:\n" + dirty)
    head = _git("rev-parse", "HEAD")
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    if branch != BRANCH:
        problems.append(f"HEAD is on {branch!r}, the lock requires {BRANCH!r}")
    if f"origin/{BRANCH}" not in _git("branch", "-r", "--contains", head):
        problems.append(f"HEAD {head[:12]} is not on origin/{BRANCH} -- push before launching")
    return problems, head, branch


def _is_diagnostic(run_id):
    return run_id.endswith("-smoke") or run_id.endswith("-diag")


def open_session(force=False):
    """Freeze the whole screen once. Every arm afterwards must match this manifest."""
    problems, head, branch = check_state()
    if problems:
        raise NotLocked("M9.0 lock is not in a runnable state:\n  " + "\n  ".join(problems))
    fp = fingerprint()
    TOKENS.mkdir(parents=True, exist_ok=True)
    if SESSION.exists() and not force:
        sess = json.loads(SESSION.read_text())
        # The session is bound to the FINGERPRINT, not to HEAD. Binding it to the commit was wrong
        # in a way that only shows up in use: committing an arm's own RESULT moves HEAD and would
        # then void every arm already run, making "commit frequently" and "run a multi-arm screen"
        # mutually exclusive. What must not move is the lock, the code and the input data -- which
        # is exactly what the fingerprint covers. `check_state()` still requires the guarded files
        # to be clean and HEAD to be pushed on `m9-work`.
        if sess["fingerprint_sha256"] != fp_sha(fp):
            raise NotLocked(
                f"the screen session was opened at fingerprint "
                f"{sess['fingerprint_sha256'][:12]} and the tree is now {fp_sha(fp)[:12]}. Arms "
                "from different lock states may not be combined. Either restore the tree or start "
                "a new session with force=True, which invalidates every arm already run under it.")
        sess["head_at_last_check"] = head
        return sess
    sess = {"opened_at": time.time(), "commit": head, "branch": branch,
            "stage": registry()["stage"], "fingerprint": fp, "fingerprint_sha256": fp_sha(fp)}
    SESSION.write_text(json.dumps(sess, indent=1))
    return sess


def begin_run(run_id, extra=None):
    """Open a one-use run token bound to the session. Raises unless the lock is clean, committed,
    pushed on `m9-work`, and identical to the session manifest."""
    if _is_diagnostic(run_id):
        return {"run_id": run_id, "diagnostic": True,
                "_note": "diagnostic: no state check, ineligible for any decision"}
    sess = open_session()
    old = TOKENS / f"{run_id}.json"
    if old.exists() and json.loads(old.read_text()).get("consumed"):
        raise NotLocked(f"{run_id!r} already produced a result under this lock. Delete its "
                        f"artifact and token deliberately if you mean to re-run it.")
    tok = {"run_id": run_id, "diagnostic": False, "commit": sess["commit"],
           "branch": sess["branch"], "session_sha256": sess["fingerprint_sha256"],
           "opened_at": time.time(), "consumed": False, "extra": extra or {}}
    (TOKENS / f"{run_id}.json").write_text(json.dumps(tok, indent=1))
    return tok


def write_result(path, payload, run_id):
    """The only sanctioned write path. Consumes the token and re-verifies the session."""
    payload = dict(payload)
    if _is_diagnostic(run_id):
        SMOKE_DIR.mkdir(parents=True, exist_ok=True)
        if path.parent != SMOKE_DIR:
            path = SMOKE_DIR / path.name
        payload["_registration"] = {"run_id": run_id, "diagnostic": True,
                                    "eligible_for_decision": False}
        path.write_text(json.dumps(payload, indent=2, default=str))
        return payload["_registration"]

    tp = TOKENS / f"{run_id}.json"
    if not tp.exists():
        raise NotLocked(f"no run token for {run_id!r} -- call begin_run() before the work")
    tok = json.loads(tp.read_text())
    if tok.get("consumed"):
        raise NotLocked(f"the run token for {run_id!r} was already consumed; a token is one-use")
    sess = open_session()
    if sess["fingerprint_sha256"] != tok["session_sha256"]:
        raise NotLocked(f"{run_id}: the lock, code or data changed while the run was in flight "
                        f"({tok['session_sha256'][:12]} -> {sess['fingerprint_sha256'][:12]}). "
                        f"This run is void.")
    payload["_registration"] = {
        "run_id": run_id, "diagnostic": False, "commit": tok["commit"], "branch": tok["branch"],
        "session_sha256": tok["session_sha256"], "stage": sess["stage"],
        "opened_at": tok["opened_at"], "written_at": time.time(), "extra": tok["extra"],
        "eligible_for_decision": True}
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    tok["consumed"] = True
    tp.write_text(json.dumps(tok, indent=1))       # consume BEFORE publishing
    tmp.replace(path)
    return payload["_registration"]


def eligible(payload, strict=True):
    """Recomputed, never trusted: an artifact counts only if it is non-diagnostic AND was written
    under the session manifest that describes the tree right now."""
    reg = payload.get("_registration", {})
    if not reg or reg.get("diagnostic") or not reg.get("eligible_for_decision"):
        return False
    if not strict:
        return True
    try:
        sess = open_session()
    except NotLocked:
        return False
    if reg.get("session_sha256") != sess["fingerprint_sha256"]:
        return False
    # and it must correspond to a token THIS session actually issued and consumed, so a
    # hand-written registration block cannot vouch for itself (Codex pass 3, B3)
    tp = TOKENS / f"{reg.get('run_id')}.json"
    if not tp.exists():
        return False
    tok = json.loads(tp.read_text())
    return bool(tok.get("consumed") and tok.get("session_sha256") == reg.get("session_sha256"))


def self_test():
    r = registry()
    d = r["dose"]
    n = r["data"]["n_screen_queries"]
    assert d["examples"] == d["epochs_query_only"] * n
    assert d["steps"] == -(-d["examples"] // d["batch_size"])
    # checkpoints must be the step at which each quarter-epoch boundary COMPLETES
    want = [-(-(k * d["examples"] // 4) // d["batch_size"]) for k in (1, 2, 3, 4)]
    assert d["checkpoints"] == want, (d["checkpoints"], want)
    assert isinstance(r["rules"]["mde"]["value"], float)
    p, head, branch = check_state()
    print(json.dumps({"registry_ok": True, "head": head[:12], "branch": branch,
                      "problems": p, "fingerprint": fp_sha(fingerprint())[:12]}, indent=1))


if __name__ == "__main__":
    self_test()
