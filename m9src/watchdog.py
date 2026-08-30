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
STATUS_MD = REPO / "m9" / "RUN_STATUS.md"


def log(rec):
    rec["wall"] = time.time()
    rec["at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    INCIDENTS.parent.mkdir(parents=True, exist_ok=True)
    with open(INCIDENTS, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(f"[{rec['at']}] {rec.get('event')}: {rec.get('detail','')}", flush=True)


def trainer_alive():
    out = subprocess.run(["pgrep", "-af", "longrun[.]py (train|decay)"],
                         capture_output=True, text=True)
    pids = []
    for line in out.stdout.splitlines():
        pid, _, args = line.partition(" ")
        if str(REPO) in args or "m9src/longrun.py" in args:   # this repo's trainer, not any
            pids.append(int(pid))
    return pids


def wait_gone(pids, timeout=180):
    """Do not start a replacement until the old process is CONFIRMED gone. A SIGTERM plus a
    15-second nap is not confirmation, and a trainer stuck in uninterruptible I/O that comes back
    would give two writers the same `last.tmp` -- which atomic replace does not protect against."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        if not any(os.path.exists(f"/proc/{p}") for p in pids):
            return True
        time.sleep(2)
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


def start_trainer(hours):
    cmd = (f"cd {REPO} && setsid nohup .venv/bin/python m9src/longrun.py train "
           f"--hours {hours} >> logs/m9_build.log 2>&1 &")
    subprocess.run(["bash", "-lc", cmd], check=False)
    time.sleep(20)
    return trainer_alive()


def gpu_ok():
    r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"],
                       capture_output=True, text=True)
    return r.returncode == 0 and bool(r.stdout.strip())


def read_json(p, default=None):
    try:
        return json.loads(p.read_text())
    except Exception:
        return default


def write_status(hb, rows, incidents):
    """A file a human can read from a phone, refreshed on the timer and pushed."""
    best = max((r["screen3"] for r in rows), default=None)
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
    lines += ["## To stop it",
              "", "```bash", f"touch {longrun.CKPT}/STOP        # clean halt at the next step",
              "python m9src/longrun.py decay      # cooldown -> a servable model", "```", ""]
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
                           check=True, capture_output=True, text=True)
        shutil.copy2(STATUS_MD, STATUS_WT / "RUN_STATUS.md")
        r = subprocess.run(
            ["bash", "-lc",
             f"cd {STATUS_WT} && git add RUN_STATUS.md && "
             f"(git diff --cached --quiet || git commit -q -m 'm9.3 build status') && "
             f"git push -q origin {STATUS_BRANCH}"],
            capture_output=True, text=True)
        if r.returncode != 0:
            log({"event": "status_push_failed", "detail": (r.stderr or r.stdout)[:200]})
    except Exception as e:
        log({"event": "status_push_failed", "detail": repr(e)[:200]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=168)
    ap.add_argument("--period", type=int, default=60, help="the consistent timer, in seconds")
    ap.add_argument("--stale", type=int, default=300, help="heartbeat age that means wedged")
    ap.add_argument("--startup-deadline", type=int, default=1800,
                    help="a fresh trainer must produce its first heartbeat within this")
    ap.add_argument("--ckpt-stale", type=int, default=5400)
    ap.add_argument("--eval-stale", type=int, default=4 * 3600)
    ap.add_argument("--max-restarts", type=int, default=8)
    ap.add_argument("--max-restarts-6h", type=int, default=3)
    ap.add_argument("--min-disk-gb", type=float, default=25.0)
    ap.add_argument("--status-every", type=int, default=1800)
    ap.add_argument("--no-push", action="store_true")
    a = ap.parse_args()

    deadline = time.time() + a.hours * 3600
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
    log({"event": "watchdog_start", "detail": f"period {a.period}s, horizon {a.hours}h"})

    while time.time() < deadline:
        time.sleep(a.period)
        # One bad iteration must never kill the only supervisor of a seven-day run: nothing
        # supervises the watchdog itself (Codex review #7, blocker 4).
        try:
            hb = read_json(HEARTBEAT)
            rows = longrun.read_history()
            incidents = [json.loads(l) for l in INCIDENTS.read_text().splitlines()[-40:]
                         if l.strip()] if INCIDENTS.exists() else []

            if (longrun.CKPT / "STOP").exists():
                log({"event": "stop_file", "detail": "clean halt requested; watchdog exiting"})
                break

            free_gb = shutil.disk_usage(longrun.CKPT if longrun.CKPT.exists() else REPO).free / 1e9
            if free_gb < a.min_disk_gb:
                (longrun.CKPT).mkdir(parents=True, exist_ok=True)
                (longrun.CKPT / "STOP").write_text("watchdog: disk headroom")
                log({"event": "disk_low", "detail": f"{free_gb:.1f} GB free < {a.min_disk_gb}; "
                                                    f"asked the trainer to stop cleanly"})
                break
            if not gpu_ok():
                log({"event": "gpu_missing", "detail": "nvidia-smi failed"})

            term = terminal_state()
            if term:
                log({"event": "terminal", "detail": f"the trainer stopped deliberately: "
                                                    f"{term['reason']} (step {term['step']:,}, "
                                                    f"{term['tokens']/1e9:.3f}B tokens). Not "
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
            thresh = a.stale if state == "train" else max(a.stale, a.startup_deadline)
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

            if not alive or stale or no_progress or ckpt_wedged or eval_wedged:
                why = ("dead" if not alive else
                       ("no heartbeat within the startup deadline" if not hb
                        else "stale heartbeat") if stale else
                       "no step progress" if no_progress else
                       f"no new checkpoint for {(time.time()-last_ckpt_at)/60:.0f} min -- a crash "
                       f"now costs everything since the last one" if ckpt_wedged else
                       f"no new evaluation for {(time.time()-last_eval_at)/3600:.1f} h -- the "
                       f"trainer's own quality kill rules cannot fire without them")
                if restarts >= a.max_restarts:
                    log({"event": "giving_up", "detail": f"{why}; {restarts} restarts already. A "
                                                         f"crash loop is a different problem from "
                                                         f"a crash."})
                    break
                if alive:
                    for pid in alive:
                        try:
                            os.kill(pid, 15)
                        except ProcessLookupError:
                            pass
                    if not wait_gone(alive):
                        log({"event": "will_not_restart",
                             "detail": f"pids {alive} did not exit even after SIGKILL; refusing to "
                                       f"start a second writer"})
                        continue
                time.sleep(2 ** min(restarts, 5))    # backoff; a crash loop should not thrash
                pids = start_trainer(max(0.5, (deadline - time.time()) / 3600))
                restarts += 1
                recent.append(time.time())
                recent[:] = [t for t in recent if t > time.time() - 6 * 3600]
                if len(recent) > a.max_restarts_6h:
                    log({"event": "giving_up", "detail": f"{len(recent)} restarts in six hours; "
                                                         f"that is a crash loop, not a crash"})
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
                        log({"event": "giving_up", "detail": "two consecutive launches produced "
                                                             "no process; this is a configuration "
                                                             "failure, not a crash"})
                        break
                else:
                    failed_starts = 0
                    log({"event": "restart", "detail": f"{why}; restart "
                                                       f"{restarts}/{a.max_restarts}; "
                                                       f"pids {pids}"})
                last_step, last_step_at = None, time.time()
                last_ckpt_at = last_eval_at = time.time()

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
        write_status(read_json(HEARTBEAT), longrun.read_history(),
                     [json.loads(l) for l in INCIDENTS.read_text().splitlines()[-40:] if l.strip()]
                     if INCIDENTS.exists() else [])
        if not a.no_push:
            push_status()
    except Exception as e:
        print(f"final status write failed: {e!r}", flush=True)
    log({"event": "watchdog_stop", "detail": f"{restarts} restarts"})


if __name__ == "__main__":
    main()
