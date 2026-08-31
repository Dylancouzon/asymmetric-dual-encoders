"""Out-of-process watchdog for the seven-day build.

The trainer already stops itself for non-finite loss, quality regression, plateau and throughput
collapse. Every one of those rules runs *inside* the training process, so none of them can fire if
that process is dead, wedged, or writing to a full disk. This is the part that watches from
outside, on a fixed timer, and it exists because of a measured incident rather than a worry:
`m9s2` fell from ~2,000 examples/s to ~400 when a teacher model stayed resident on the GPU, and
**nothing reported it** — the run would have delivered a fifth of its dose and looked healthy.

What it checks, every `--period` seconds:

| check | why it is here |
|---|---|
| heartbeat age | a wedged process keeps its PID and stops writing. Liveness is not a PID |
| step progress | the heartbeat can be written by a loop that is making no progress |
| throughput floor | the m9s2 failure: still training, at a fifth of the rate, silently |
| checkpoint freshness | if `last.pt` stops advancing, a crash costs everything since it |
| disk headroom | a checkpoint that half-writes on a full disk is worse than one not written |
| GPU present | a card that falls off the bus takes the run with it |
| eval cadence | training without evaluating means the kill rules cannot fire either |

On death or a stall it **restarts the trainer**, which is safe precisely because resume is exact
and tested (`m9src/test_resume.py`). Restarts are counted and bounded: a crash loop is a different
problem from a crash, and it stops rather than thrashing for three days.

It also writes `m9/RUN_STATUS.md` and pushes it, so the run is legible from anywhere while nobody
is at the machine.

    python m9src/watchdog.py --hours 168 --period 60
"""
import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import m9base
from m9base import REPO, WORK

import longrun   # noqa: E402

RUN = longrun.RUN
HEARTBEAT = RUN / "heartbeat.json"
INCIDENTS = RUN / "watchdog.jsonl"
DEADLINE = RUN / "deadline.json"
STATUS_MD = REPO / "m9" / "RUN_STATUS.md"

OPERATOR_PROCEDURE = """1. Stop safely: `touch work/m9long/ckpt/STOP`. Keep the watchdog running;
   it supervises until `terminal.json` confirms the trainer exited.
2. Cool down: after that terminal marker appears, run
   `setsid nohup .venv/bin/python m9src/watchdog.py --cooldown --hours 4 >> logs/m9_watchdog.log 2>&1 &`.
   The cooldown command safely consumes the acknowledged STOP and terminal markers, resumes
   `last.pt` in decay, and supervises it through `cooldown complete`.
3. Restart after a crash: if the watchdog is alive, do nothing; it restarts the trainer exactly.
   If the watchdog died, rerun the original watchdog launch command. It reuses `deadline.json`,
   attaches to a live trainer or resumes `last.pt`, and never resets the seven-day horizon."""


def log(rec):
    rec["wall"] = time.time()
    rec["at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    INCIDENTS.parent.mkdir(parents=True, exist_ok=True)
    with open(INCIDENTS, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(f"[{rec['at']}] {rec.get('event')}: {rec.get('detail','')}", flush=True)


WD_LOCK = RUN / "watchdog.lock"
_WD_LOCK_FH = None    # held open for the watchdog's lifetime; the kernel releases it on death


def acquire_wd_lock():
    """Two watchdogs do not compose: each SIGTERMs trainers the other just started and both
    inflate restart counters until one (or both) gives up (Fable review, M3). flock, not
    O_EXCL-plus-staleness: the kernel drops the lock the instant the holder dies, so there is no
    stale-lock state and therefore no takeover race (Codex #10/#11). The file is NEVER unlinked --
    unlinking would let a second watchdog lock a fresh inode while the first still runs."""
    global _WD_LOCK_FH
    import fcntl
    RUN.mkdir(parents=True, exist_ok=True)
    fh = open(WD_LOCK, "a+")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        raise SystemExit(f"{WD_LOCK} is flock-held by a live watchdog. Two watchdogs must never "
                         f"supervise one trainer.")
    fh.truncate(0)
    fh.write(json.dumps({"pid": os.getpid(), "at": time.time()}))
    fh.flush()
    _WD_LOCK_FH = fh


def trainer_alive():
    out = subprocess.run(["pgrep", "-af", "longrun[.]py (train|decay)"],
                         capture_output=True, text=True, timeout=30)
    pids = []
    for line in out.stdout.splitlines():
        pid, _, args = line.partition(" ")
        if str(REPO) in args or "m9src/longrun.py" in args:   # this repo's trainer, not any
            pids.append(int(pid))
    return pids


def wait_gone(pids, timeout=180, escalate=True):
    """Do not start a replacement until the old process is CONFIRMED gone. A SIGTERM plus a
    15-second nap is not confirmation, and a trainer stuck in uninterruptible I/O that comes back
    would give two writers the same `last.tmp` -- which atomic replace does not protect against."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        if not any(os.path.exists(f"/proc/{p}") for p in pids):
            return True
        time.sleep(2)
    if escalate:
        for p in pids:
            try:
                os.kill(p, 9)
            except ProcessLookupError:
                pass
        time.sleep(5)
    return not any(os.path.exists(f"/proc/{p}") for p in pids)


def terminal_state():
    """A registered stop. The watchdog must never restart after one."""
    return read_json(longrun.TERMINAL)


def start_trainer(deadline, cooldown=False):
    # The dead trainer's heartbeat must go first, or its old wall clock gives the fresh process
    # only ~80s to finish a possibly cold import before being declared stale (Fable review, M2).
    # Only called with no live trainer (wait_gone confirmed), so there is no competing writer.
    HEARTBEAT.unlink(missing_ok=True)
    mode = "decay" if cooldown else "train"
    anneal = "" if cooldown else " --anneal-before-deadline"
    cmd = (f"cd {REPO} && setsid nohup .venv/bin/python m9src/longrun.py {mode} "
           f"--deadline-wall {deadline:.6f}{anneal} >> logs/m9_build.log 2>&1 &")
    subprocess.run(["bash", "-lc", cmd], check=False, timeout=60)
    time.sleep(20)
    return trainer_alive()


def gpu_ok():
    """ADVISORY ONLY, and it must never raise. A wedged driver makes `nvidia-smi` hang until the
    timeout; `TimeoutExpired` used to escape here into the loop's outer `except Exception`, which
    logged and started the next iteration -- so a wedged GPU disabled restart, terminal-state and
    deadline supervision for the rest of the run (Codex unattended review, blocker 1)."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=60)
        return r.returncode == 0 and bool(r.stdout.strip())
    except Exception:
        return False


def read_json(p, default=None):
    try:
        return json.loads(p.read_text())
    except Exception:
        return default


def write_watchdog_terminal(reason, action):
    """Record a terminal state even when a wedged trainer cannot write its own marker."""
    hb = read_json(HEARTBEAT, {}) or {}
    rec = {"reason": f"watchdog giving up: {reason}", "step": hb.get("step", 0),
           "tokens": hb.get("tokens", 0), "examples": hb.get("examples", 0),
           "phase": hb.get("phase", "unknown"), "watchdog_action": action,
           "at": time.strftime("%Y-%m-%dT%H:%M:%S"), "wall": time.time()}
    longrun.TERMINAL.parent.mkdir(parents=True, exist_ok=True)
    tmp = longrun.TERMINAL.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, indent=1))
    os.replace(tmp, longrun.TERMINAL)
    return rec


def give_up_safely(reason, clean_grace=120):
    """Leave no silently-running trainer behind on any supervision give-up path."""
    longrun.CKPT.mkdir(parents=True, exist_ok=True)
    stop = longrun.CKPT / "STOP"
    stop.write_text(f"watchdog giving up: {reason}\n")
    pids = trainer_alive()
    log({"event": "give_up_stop_requested",
         "detail": f"{reason}; wrote {stop}; waiting up to {clean_grace}s for a clean terminal "
                   f"stop from pids {pids or 'none'}"})
    until = time.time() + clean_grace
    while pids and time.time() < until:
        time.sleep(2)
        pids = trainer_alive()
    term = terminal_state()
    if term and not trainer_alive():
        action = (f"requested STOP and trainer exited cleanly with terminal reason "
                  f"{term.get('reason')!r}")
        log({"event": "giving_up", "detail": f"{reason}; {action}. Nothing remains running."})
        return

    pids = trainer_alive()
    actions = ["wrote STOP"]
    if pids:
        for pid in pids:
            try:
                os.kill(pid, 15)
            except ProcessLookupError:
                pass
        actions.append(f"sent SIGTERM to {pids}")
        log({"event": "give_up_sigterm", "detail": f"{reason}; sent SIGTERM to pids {pids}"})
        wait_gone(pids, timeout=30, escalate=False)
    pids = trainer_alive()
    if pids:
        for pid in pids:
            try:
                os.kill(pid, 9)
            except ProcessLookupError:
                pass
        actions.append(f"sent SIGKILL to {pids}")
        log({"event": "give_up_sigkill", "detail": f"{reason}; sent SIGKILL to pids {pids}"})
        wait_gone(pids, timeout=10, escalate=False)
    remaining = trainer_alive()
    action = "; ".join(actions) + f"; remaining trainer pids {remaining or 'none'}"
    write_watchdog_terminal(reason, action)
    log({"event": "giving_up", "detail": f"{reason}; {action}; wrote terminal marker. "
                                            "No trainer will be restarted."})


def shared_deadline(hours, cooldown=False):
    """Create once, then reuse after watchdog crashes so restarts cannot extend the run."""
    if not cooldown:
        saved = read_json(DEADLINE)
        if saved and isinstance(saved.get("wall"), (int, float)):
            return float(saved["wall"]), "reused"
        # FAIL CLOSED. read_json() maps every parse/read error to None, so a truncated or
        # corrupt deadline.json used to look identical to "no deadline yet" and a watchdog
        # restart minted a brand-new seven-day horizon, silently extending the run past the
        # registered budget (Codex unattended review, major 6).
        if DEADLINE.exists():
            raise SystemExit(
                f"{DEADLINE} exists but is unreadable or has no numeric 'wall'. Refusing to "
                f"create a new {hours} h horizon, which would extend the registered run. "
                f"Repair or delete it deliberately, recording the decision.")
    wall = time.time() + hours * 3600
    tmp = DEADLINE.with_suffix(".tmp")
    with open(tmp, "w") as fh:
        json.dump({"wall": wall, "hours": hours,
                   "mode": "cooldown" if cooldown else "train",
                   "created_at": time.strftime("%Y-%m-%dT%H:%M:%S")}, fh, indent=1)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, DEADLINE)
    dfd = os.open(str(DEADLINE.parent), os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)
    return wall, "created"


def read_incidents(k=40):
    """Per-line tolerant: one torn line in watchdog.jsonl must not send every tick into the
    exception handler and disable supervision for as long as it stays in the tail window
    (Codex #8, MAJOR 3)."""
    if not INCIDENTS.exists():
        return []
    out = []
    for line in INCIDENTS.read_text().splitlines()[-k:]:
        try:
            out.append(json.loads(line))
        except (json.JSONDecodeError, ValueError):
            continue
    return out


def write_status(hb, rows, incidents):
    """A file a human can read from a phone, refreshed on the timer and pushed."""
    best = max((r["screen3"] for r in rows), default=None)
    try:
        ceil = longrun.guard9.registry()["ceilings"][
            read_json(longrun.CONFIG)["teacher"]]["SCREEN3"]
    except Exception:
        ceil = 0.68223
    lines = ["# M9.3 build — live status", "",
             f"_Updated {time.strftime('%Y-%m-%d %H:%M:%S')} by `m9src/watchdog.py`._", ""]
    if hb:
        # Heartbeats from non-train states (`verify`, `model`, `eval`, `stopped`) carry no
        # step/throughput fields; rendering must not assume them (Codex review #7, blocker 4).
        age = time.time() - hb["wall"]
        if hb.get("step") is not None and hb.get("stable_token_cap"):
            lines += [f"**step {hb['step']:,}** · **{hb.get('tokens', 0)/1e9:.3f} B tokens** "
                      f"({hb.get('tokens', 0)/hb['stable_token_cap']:.1%} of the cap) · "
                      f"{(hb.get('tok_per_s') or 0):,.0f} tok/s · "
                      f"phase **{hb.get('phase', '?')}** · heartbeat {age:.0f}s old", ""]
        else:
            lines += [f"state **{hb.get('state', '?')}** · heartbeat {age:.0f}s old"
                      + (f" · {hb.get('reason')}" if hb.get("reason") else ""), ""]
    if rows:
        lines += [f"**Best SCREEN-3 {best:.5f} — retention {best/ceil:.3f}** of the "
                  f"{ceil} teacher ceiling.", "",
                  "| step | B tokens | SCREEN-3 | retention |", "|---|---|---|---|"]
        for r in rows[-15:]:
            lines.append(f"| {r['step']:,} | {r['tokens']/1e9:.3f} | {r['screen3']:.5f} | "
                         f"{r['retention']} |")
        lines.append("")
    if incidents:
        lines += ["## Incidents", "", "| when | event | detail |", "|---|---|---|"]
        for i in incidents[-12:]:
            lines.append(f"| {i['at']} | {i.get('event')} | {str(i.get('detail',''))[:90]} |")
        lines.append("")
    lines += ["## Stop, cool down, restart", "", OPERATOR_PROCEDURE, ""]
    STATUS_MD.write_text("\n".join(lines))


STATUS_WT = WORK / "m9status"
STATUS_BRANCH = "m9-status"


def push_status():
    """Publish the status file on its OWN branch, through a separate worktree.

    Committing on `m9-work` was unsafe in three ways Codex named: `git commit` would sweep up any
    human changes already staged; a failed push would leave HEAD unpushed, which fails the build
    guard and makes the next restart give up; and every failure was swallowed. A dedicated
    worktree touches neither the index nor HEAD of the working branch.
    """
    try:
        if not STATUS_WT.exists():
            subprocess.run(["bash", "-lc",
                            f"cd {REPO} && git worktree add -B {STATUS_BRANCH} {STATUS_WT} "
                            f"origin/{STATUS_BRANCH} 2>/dev/null || "
                            f"git worktree add -B {STATUS_BRANCH} {STATUS_WT}"],
                           check=True, capture_output=True, text=True, timeout=120)
        shutil.copy2(STATUS_MD, STATUS_WT / "RUN_STATUS.md")
        # timeout: a hung push (WSL network flap mid-transfer) would otherwise freeze the
        # single-threaded watchdog indefinitely -- no restarts, no checks (Fable review, M4)
        r = subprocess.run(
            ["bash", "-lc",
             f"cd {STATUS_WT} && git add RUN_STATUS.md && "
             f"(git diff --cached --quiet || git commit -q -m 'm9.3 build status') && "
             # `timeout` bounds git ITSELF: killing the bash wrapper left a hung push alive,
             # holding index.lock and blocking every later push (Codex review, minor 9).
             f"timeout -k 10 120 git push -q origin {STATUS_BRANCH}"],
            capture_output=True, text=True, timeout=180, start_new_session=True)
        if r.returncode != 0:
            log({"event": "status_push_failed", "detail": (r.stderr or r.stdout)[:200]})
    except Exception as e:
        log({"event": "status_push_failed", "detail": repr(e)[:200]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=168)
    ap.add_argument("--cooldown", action="store_true",
                    help="supervise the STOP -> decay transition (normally use --hours 4)")
    ap.add_argument("--period", type=int, default=60, help="the consistent timer, in seconds")
    ap.add_argument("--stale", type=int, default=300, help="heartbeat age that means wedged")
    ap.add_argument("--startup-deadline", type=int, default=1800,
                    help="a fresh trainer must produce its first heartbeat within this")
    ap.add_argument("--ckpt-stale", type=int, default=2 * 3600)
    ap.add_argument("--eval-stale", type=int, default=5 * 3600)
    ap.add_argument("--eval-grace", type=int, default=3600,
                    help="heartbeat staleness allowance while the trainer is inside an evaluation")
    ap.add_argument("--max-restarts", type=int, default=8)
    ap.add_argument("--max-restarts-6h", type=int, default=3)
    ap.add_argument("--min-disk-gb", type=float, default=25.0)
    ap.add_argument("--status-every", type=int, default=1800)
    ap.add_argument("--no-push", action="store_true")
    a = ap.parse_args()

    restarts, failed_starts, last_status, recent = 0, 0, 0.0, []
    launched_at = time.time()
    last_step, last_step_at, last_digest = None, time.time(), time.time()
    # Checkpoint/eval advancement is judged over time THIS watchdog has observed, never over raw
    # mtimes: a resume after downtime starts with an old `last.pt`, and acting on its age directly
    # would restart a healthy trainer before its first checkpoint boundary -- a crash loop made of
    # nothing (Codex review #7, blocker 5).
    last_ckpt_mtime, last_ckpt_at = None, time.time()
    last_eval_n, last_eval_at = None, time.time()
    for req in (longrun.CONFIG, longrun.MANIFEST):
        if not req.exists():
            raise SystemExit(f"{req} is missing -- the build is not ready to be watched. Run "
                             f"`longrun.py prepare`, `targets`, `manifest`, then generate the "
                             f"config. A watchdog over a run that cannot start is theatre.")
    acquire_wd_lock()
    if a.cooldown:
        if trainer_alive():
            raise SystemExit("cannot start cooldown watchdog while a trainer is still alive")
        if not longrun.TERMINAL.exists() or not (longrun.CKPT / "last.pt").exists():
            raise SystemExit("cooldown requires an acknowledged terminal stop and ckpt/last.pt")
    prior_terminal_wall = (read_json(longrun.TERMINAL, {}) or {}).get("wall") if a.cooldown else None
    deadline, deadline_action = shared_deadline(a.hours, cooldown=a.cooldown)
    log({"event": "watchdog_start", "detail": f"period {a.period}s, mode "
                                                f"{'cooldown' if a.cooldown else 'train'}, "
                                                f"absolute deadline {deadline:.3f} "
                                                f"({deadline_action})"})
    # The initial launch is deliberate, not a "restart" of a dead trainer: counting it against
    # max_restarts spent supervision budget on a non-incident (Fable review, minor). Guarded like
    # a tick: a TimeoutExpired here must not kill the watchdog before its loop even starts
    # (Codex #10) -- the loop's dead-trainer path retries the launch.
    try:
        if (a.cooldown or not terminal_state()) and not trainer_alive():
            pids = start_trainer(deadline, cooldown=a.cooldown)
            log({"event": "launch", "detail": f"initial trainer start; pids {pids}"})
            launched_at = time.time()
    except Exception as e:
        log({"event": "watchdog_error", "detail": f"initial launch: {e!r}"[:300]})

    stop_seen = False
    while True:
        time.sleep(a.period)
        # One bad iteration must never kill the only supervisor of a seven-day run: nothing
        # supervises the watchdog itself (Codex review #7, blocker 4).
        try:
            hb = read_json(HEARTBEAT)
            rows = longrun.read_history()
            incidents = read_incidents()

            stop_requested = (longrun.CKPT / "STOP").exists()
            if stop_requested and not stop_seen:
                log({"event": "stop_file", "detail": "clean halt requested; watchdog remains "
                                                    "until the trainer records a terminal state"})
                stop_seen = True

            free_gb = shutil.disk_usage(longrun.CKPT if longrun.CKPT.exists() else REPO).free / 1e9
            if free_gb < a.min_disk_gb:
                (longrun.CKPT).mkdir(parents=True, exist_ok=True)
                (longrun.CKPT / "STOP").write_text("watchdog: disk headroom")
                log({"event": "disk_low", "detail": f"{free_gb:.1f} GB free < {a.min_disk_gb}; "
                                                    f"asked the trainer to stop cleanly"})
                stop_requested = stop_seen = True
            if not gpu_ok():
                log({"event": "gpu_missing", "detail": "nvidia-smi failed"})

            term = terminal_state()
            # During cooldown launch the trainer removes the acknowledged prior marker under its
            # lock. Do not mistake that old marker for completion while the new process imports.
            if a.cooldown and term and term.get("wall") == prior_terminal_wall:
                term = None
            if term:
                log({"event": "terminal", "detail": f"the trainer stopped deliberately: "
                                                    f"{term.get('reason')} (step "
                                                    f"{term.get('step', 0):,}, "
                                                    f"{term.get('tokens', 0)/1e9:.3f}B tokens). Not "
                                                    f"restarting -- a registered stop is a "
                                                    f"decision."})
                break

            alive = trainer_alive()
            # A fresh start has no heartbeat yet, so `hb and ...` was False for both staleness
            # checks and a wedge during verify/targets/warm-start was invisible forever (Codex,
            # blocker 4). Non-train states (verify hashes gigabytes, an eval holds the beat for
            # its whole read) legitimately beat slower than a training step, so they get the
            # startup allowance, not the train one.
            state = (hb or {}).get("state", "train")
            # An evaluation writes one beat at its start and none inside; the five-hour overdue
            # rule only runs while state is "train", so a slow (page-cache-thrashed) eval must
            # get ITS grace here, not the 30-minute startup one -- killing it mid-read replays
            # 15,000 steps and meets the same eval again: a restart loop (Codex #8, MAJOR 4).
            thresh = (a.stale if state == "train"
                      else a.eval_grace if state in ("eval", "eval0")
                      else max(a.stale, a.startup_deadline))
            started = hb["wall"] if hb else launched_at
            stale = (time.time() - started) > (thresh if hb else a.startup_deadline)
            step_now = hb.get("step") if hb else None
            no_progress = (step_now is not None and step_now == last_step
                           and (time.time() - last_step_at) > a.stale
                           and state == "train")
            if step_now != last_step:
                last_step, last_step_at = step_now, time.time()

            # Checkpoint/eval wedges RESTART, they do not just log (Codex review #7, blocker 5).
            ck = longrun.CKPT / "last.pt"
            ck_m = ck.stat().st_mtime if ck.exists() else None
            if ck_m != last_ckpt_mtime:
                last_ckpt_mtime, last_ckpt_at = ck_m, time.time()
            ckpt_wedged = (bool(alive) and state == "train" and ck_m is not None
                           and time.time() - last_ckpt_at > a.ckpt_stale)
            if len(rows) != last_eval_n:
                last_eval_n, last_eval_at = len(rows), time.time()
            eval_wedged = (bool(alive) and state == "train" and bool(rows)
                           and time.time() - last_eval_at > a.eval_stale)

            # Both processes own this exact absolute deadline. The trainer stops at its next safe
            # step boundary; the watchdog stays beyond it until terminal state exists.
            past_deadline = time.time() >= deadline
            if past_deadline and not alive:
                give_up_safely("shared deadline passed and trainer exited without terminal state",
                               clean_grace=0)
                break
            if past_deadline and alive and time.time() > deadline + a.startup_deadline:
                give_up_safely("trainer remained alive past the shared deadline grace")
                break

            if not alive or stale or no_progress or ckpt_wedged or eval_wedged:
                why = ("dead" if not alive else
                       ("no heartbeat within the startup deadline" if not hb
                        else "stale heartbeat") if stale else
                       "no step progress" if no_progress else
                       f"no new checkpoint for {(time.time()-last_ckpt_at)/60:.0f} min -- a crash "
                       f"now costs everything since the last one" if ckpt_wedged else
                       f"no new evaluation for {(time.time()-last_eval_at)/3600:.1f} h -- the "
                       f"trainer's own quality kill rules cannot fire without them")
                if stop_requested:
                    give_up_safely(f"trainer failed while a clean STOP was pending: {why}")
                    break
                if past_deadline:
                    give_up_safely(f"trainer failed at the shared deadline: {why}")
                    break
                if restarts >= a.max_restarts:
                    give_up_safely(f"{why}; {restarts} restarts already; crash-loop limit reached")
                    break
                if alive:
                    for pid in alive:
                        try:
                            os.kill(pid, 15)
                        except ProcessLookupError:
                            pass
                    if not wait_gone(alive):
                        give_up_safely(f"pids {alive} did not exit after restart termination; "
                                       "refusing to start a second writer", clean_grace=0)
                        break
                time.sleep(2 ** min(restarts, 5))    # backoff; a crash loop should not thrash
                pids = start_trainer(deadline, cooldown=a.cooldown)
                restarts += 1
                recent.append(time.time())
                recent[:] = [t for t in recent if t > time.time() - 6 * 3600]
                if len(recent) > a.max_restarts_6h:
                    give_up_safely(f"{len(recent)} restarts in six hours; crash-loop limit reached")
                    break
                if not pids:
                    # A restart that starts nothing is not a restart. Retrying a launch that
                    # cannot launch just burns the budget silently -- exactly the class this
                    # watchdog exists for. Two in a row and it stops and says so.
                    failed_starts += 1
                    log({"event": "restart_failed", "detail": f"{why}; nothing came up. See "
                                                              f"logs/m9_build.log. "
                                                              f"{failed_starts} consecutive."})
                    if failed_starts >= 2:
                        give_up_safely("two consecutive launches produced no process; "
                                       "configuration failure", clean_grace=0)
                        break
                else:
                    failed_starts = 0
                    log({"event": "restart", "detail": f"{why}; restart "
                                                       f"{restarts}/{a.max_restarts}; "
                                                       f"pids {pids}"})
                last_step, last_step_at = None, time.time()
                last_ckpt_at = last_eval_at = launched_at = time.time()

            if hb and hb.get("tok_per_s") and hb.get("floor") and hb["tok_per_s"] < hb["floor"]:
                log({"event": "throughput_low",
                     "detail": f"{hb['tok_per_s']:,.0f} tok/s below the floor "
                               f"{hb['floor']:,.0f} -- the m9s2 failure mode; the trainer's own "
                               f"rule should also fire"})

            if time.time() - last_digest > 86400:
                r0 = [r for r in rows if r["wall"] >= time.time() - 86400]
                if r0:
                    log({"event": "daily", "detail":
                         f"{len(r0)} evals; tokens {r0[0]['tokens']/1e9:.2f}B -> "
                         f"{r0[-1]['tokens']/1e9:.2f}B; SCREEN-3 {r0[0]['screen3']:.5f} -> "
                         f"{r0[-1]['screen3']:.5f} ({r0[-1]['screen3']-r0[0]['screen3']:+.5f}); "
                         f"best {max(r['screen3'] for r in rows):.5f}"})
                last_digest = time.time()

            if time.time() - last_status > a.status_every:
                write_status(hb, rows, incidents)
                if not a.no_push:
                    push_status()
                last_status = time.time()
        except Exception as e:
            try:
                log({"event": "watchdog_error", "detail": repr(e)[:300]})
            except Exception:
                print(f"watchdog_error (unloggable): {e!r}", flush=True)

    try:
        write_status(read_json(HEARTBEAT), longrun.read_history(), read_incidents())
        if not a.no_push:
            push_status()
    except Exception as e:
        print(f"final status write failed: {e!r}", flush=True)
    log({"event": "watchdog_stop", "detail": f"{restarts} restarts"})


if __name__ == "__main__":
    main()
