"""M9 registration gate — the M8 G1 pattern, plus the thing G1 did not have: a RUN TOKEN.

Codex BLOCKER-3: a write-time-only check lets a session start under lock A, watch the console,
amend and push lock B, and write the artifact successfully under B. So an arm opens a token at
launch recording the lock commit, the branch, the lock-file hashes, the code hashes and the data
hashes; `write_result` refuses unless every one of those is byte-identical at write time.
Amending the lock mid-run therefore kills the run instead of blessing it.

Note the asymmetry that makes the launch check worth having at all (m8/CODEMAP.md pitfall 24):
the launch check fails fast and cheap, the write check fails after the entire job has run.
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
             ("m9base", "data", "eval9", "guard9", "nano", "screen", "screen_stats",
              "teacher9", "port", "lock_constants")) + \
       ("m7src/evalkit.py", "m7src/teacher.py", "m7src/devsuite.py", "m7src/dev_eval.py",
        "m7src/pool.py", "m7src/heldout.py", "m8src/paths_guard.py")
DATA = ("work/m9_screen_queries.json", "work/m9_screen_rows.npy")
TOKENS = WORK / "m9tokens"


class NotLocked(RuntimeError):
    pass


def _sha(p):
    p = REPO / p if not str(p).startswith("/") else p
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def _git(*args):
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True).stdout.strip()


def registry():
    return json.loads((M9 / "registry.json").read_text())


def fingerprint():
    """Everything a result is allowed to depend on, hashed."""
    return {"lock": {f: _sha(f) for f in GUARDED},
            "code": {f: _sha(f) for f in CODE},
            "data": {f: _sha(f) for f in DATA}}


def check_state():
    problems = []
    for f in GUARDED + DATA:
        if not (REPO / f).exists():
            problems.append(f"{f} is missing -- M9.0 is not locked")
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


def begin_run(run_id, extra=None):
    """Open a run token. Raises unless the lock is clean, committed and pushed on `m9-work`."""
    if run_id.endswith("-smoke"):
        return {"run_id": run_id, "diagnostic": True,
                "_note": "diagnostic smoke: ineligible for any decision"}
    problems, head, branch = check_state()
    if problems:
        raise NotLocked("M9.0 lock is not in a runnable state:\n  " + "\n  ".join(problems))
    TOKENS.mkdir(parents=True, exist_ok=True)
    tok = {"run_id": run_id, "diagnostic": False, "commit": head, "branch": branch,
           "stage": registry()["stage"], "opened_at": time.time(),
           "fingerprint": fingerprint(), "extra": extra or {}}
    (TOKENS / f"{run_id}.json").write_text(json.dumps(tok, indent=1))
    return tok


def write_result(path, payload, run_id):
    """The only sanctioned write path. Verifies the run token still describes reality."""
    payload = dict(payload)
    if run_id.endswith("-smoke"):
        payload["_registration"] = {"run_id": run_id, "diagnostic": True,
                                    "eligible_for_decision": False}
        path.write_text(json.dumps(payload, indent=2, default=str))
        return payload["_registration"]

    tp = TOKENS / f"{run_id}.json"
    if not tp.exists():
        raise NotLocked(f"no run token for {run_id!r} -- call begin_run() before the work")
    tok = json.loads(tp.read_text())
    problems, head, branch = check_state()
    if problems:
        raise NotLocked("lock state broke during the run:\n  " + "\n  ".join(problems))
    now = fingerprint()
    drift = [f"{k}/{f}" for k in now for f in now[k] if now[k][f] != tok["fingerprint"][k].get(f)]
    if drift:
        raise NotLocked(f"{run_id}: the lock, code or data changed while the run was in flight: "
                        + ", ".join(drift) + " -- this run is void")
    if head != tok["commit"]:
        raise NotLocked(f"{run_id}: HEAD moved {tok['commit'][:12]} -> {head[:12]} mid-run")
    payload["_registration"] = {**{k: v for k, v in tok.items() if k != "fingerprint"},
                                "verified_at_write": True,
                                "fingerprint_sha256": hashlib.sha256(
                                    json.dumps(now, sort_keys=True).encode()).hexdigest(),
                                "eligible_for_decision": True}
    path.write_text(json.dumps(payload, indent=2, default=str))
    return payload["_registration"]


def eligible(payload):
    return bool(payload.get("_registration", {}).get("eligible_for_decision"))


def self_test():
    r = registry()
    d = r["dose"]
    assert d["examples"] == 16 * r["data"]["n_screen_queries"] == 3884576
    assert d["steps"] == -(-d["examples"] // d["batch_size"]) == 30349
    mx = d["mix_arm"]
    assert mx["query_token_target"] + mx["doc_token_target"] == d["T_base_nonpad_tokens"]
    assert r["rules"]["teacher_swap"]["margin"] > 0.005
    ids = [a["id"] for a in r["arms"]]
    assert ids[0] == "m9p-bs128" and "m9s1b" in ids and len(ids) == 9
    p, head, branch = check_state()
    print(json.dumps({"registry_ok": True, "head": head[:12], "branch": branch,
                      "problems": p}, indent=1))


if __name__ == "__main__":
    self_test()
