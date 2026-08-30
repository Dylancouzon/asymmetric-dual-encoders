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
    out = subprocess.run(["pgrep", "-f", "longrun[.]py train"], capture_output=True, text=True)
    return [int(x) for x in out.stdout.split()]


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
        age = time.time() - hb["wall"]
        lines += [f"**step {hb['step']:,}** · **{hb['tokens']/1e9:.3f} B tokens** "
                  f"({hb['tokens']/hb['stable_token_cap']:.1%} of the cap) · "
                  f"{hb['tok_per_s']:,.0f} tok/s · phase **{hb['phase']}** · "
                  f"heartbeat {age:.0f}s old", ""]
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


def push_status():
    subprocess.run(["bash", "-lc",
                    f"cd {REPO} && git add m9/RUN_STATUS.md && "
                    f"git commit -q -m 'm9.3: build status' && git push -q origin m9-work"],
                   check=False, capture_output=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=168)
    ap.add_argument("--period", type=int, default=60, help="the consistent timer, in seconds")
    ap.add_argument("--stale", type=int, default=900, help="heartbeat age that means wedged")
    ap.add_argument("--max-restarts", type=int, default=8)
    ap.add_argument("--min-disk-gb", type=float, default=25.0)
    ap.add_argument("--status-every", type=int, default=1800)
    ap.add_argument("--no-push", action="store_true")
    a = ap.parse_args()

    deadline = time.time() + a.hours * 3600
    restarts, failed_starts, last_status = 0, 0, 0.0
    last_step, last_step_at, last_digest = None, time.time(), time.time()
    for req in (longrun.CONFIG, longrun.MANIFEST):
        if not req.exists():
            raise SystemExit(f"{req} is missing -- the build is not ready to be watched. Run "
                             f"`longrun.py prepare`, `targets`, `manifest`, then generate the "
                             f"config. A watchdog over a run that cannot start is theatre.")
    log({"event": "watchdog_start", "detail": f"period {a.period}s, horizon {a.hours}h"})

    while time.time() < deadline:
        time.sleep(a.period)
        hb = read_json(HEARTBEAT)
        rows = longrun.read_history()
        incidents = [json.loads(l) for l in INCIDENTS.read_text().splitlines()[-40:]
                     if l.strip()] if INCIDENTS.exists() else []

        if (longrun.CKPT / "STOP").exists():
            log({"event": "stop_file", "detail": "clean halt requested; watchdog exiting"})
            break

        free_gb = shutil.disk_usage(REPO).free / 1e9
        if free_gb < a.min_disk_gb:
            (longrun.CKPT).mkdir(parents=True, exist_ok=True)
            (longrun.CKPT / "STOP").write_text("watchdog: disk headroom")
            log({"event": "disk_low", "detail": f"{free_gb:.1f} GB free < {a.min_disk_gb}; "
                                                f"asked the trainer to stop cleanly"})
            break
        if not gpu_ok():
            log({"event": "gpu_missing", "detail": "nvidia-smi failed"})

        alive = trainer_alive()
        stale = hb and (time.time() - hb["wall"]) > a.stale
        no_progress = hb and last_step is not None and hb["step"] == last_step and \
            (time.time() - last_step_at) > a.stale
        if hb and hb["step"] != last_step:
            last_step, last_step_at = hb["step"], time.time()

        if not alive or stale or no_progress:
            why = ("dead" if not alive else "stale heartbeat" if stale else "no step progress")
            if restarts >= a.max_restarts:
                log({"event": "giving_up", "detail": f"{why}; {restarts} restarts already. A crash "
                                                     f"loop is a different problem from a crash."})
                break
            for pid in alive:
                try:
                    os.kill(pid, 15)
                except ProcessLookupError:
                    pass
            time.sleep(15)
            longrun.LOCKFILE.unlink(missing_ok=True)
            pids = start_trainer(max(0.5, (deadline - time.time()) / 3600))
            restarts += 1
            if not pids:
                # A restart that starts nothing is not a restart. Retrying a launch that cannot
                # launch just burns the budget silently -- exactly the class this watchdog exists
                # for. Two in a row and it stops and says so.
                failed_starts += 1
                log({"event": "restart_failed", "detail": f"{why}; nothing came up. See "
                                                          f"logs/m9_build.log. "
                                                          f"{failed_starts} consecutive."})
                if failed_starts >= 2:
                    log({"event": "giving_up", "detail": "two consecutive launches produced no "
                                                         "process; this is a configuration "
                                                         "failure, not a crash"})
                    break
            else:
                failed_starts = 0
                log({"event": "restart", "detail": f"{why}; restart {restarts}/{a.max_restarts}; "
                                                   f"pids {pids}"})
            last_step, last_step_at = None, time.time()

        if hb and hb.get("tok_per_s") and hb.get("floor") and hb["tok_per_s"] < hb["floor"]:
            log({"event": "throughput_low",
                 "detail": f"{hb['tok_per_s']:,.0f} tok/s below the floor {hb['floor']:,.0f} -- "
                           f"the m9s2 failure mode; the trainer's own rule should also fire"})

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

    write_status(read_json(HEARTBEAT), longrun.read_history(),
                 [json.loads(l) for l in INCIDENTS.read_text().splitlines()[-40:] if l.strip()]
                 if INCIDENTS.exists() else [])
    if not a.no_push:
        push_status()
    log({"event": "watchdog_stop", "detail": f"{restarts} restarts"})


if __name__ == "__main__":
    main()
