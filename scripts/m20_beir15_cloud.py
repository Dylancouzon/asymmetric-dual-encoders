#!/usr/bin/env python3
"""M20 stages C and D on the retained A100: BEIR-15 descriptive validation, then the archive.

Registered in `m20/REGISTRATION.md` under owner ruling R22. Deliberately a SEPARATE controller from
`scripts/m13_reserved_cloud.py`: the pod stops between the two, so the irreplaceable reserved
receipt is durable and pushed before this much longer, entirely repeatable work begins. Nothing
here is protected access -- BEIR-15 never opens a reserved payload -- so a crash costs time only.

It refuses to start unless the reserved transaction has completed, which is what makes the BEIR-15
table able to carry the reserved rows at all.

  cd /home/dylan/asymetric-dual-encoders/work/m13cloud
  .venv/bin/python scripts/m20_beir15_cloud.py            # stage C, then stage D build + pull
  .venv/bin/python scripts/m20_beir15_cloud.py --archive-only
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shlex
import signal
import subprocess
import time

REPO = Path(__file__).resolve().parents[1]
import sys                                                                    # noqa: E402

sys.path.insert(0, str(REPO / "scripts"))
from m13_reserved_cloud import (BRANCH, Cloud, KEYS, OTHER_PODS, POD, REMOTE,  # noqa: E402
                                SSH_ALIAS, SSH_CONFIG, MAX_TOTAL_HOURLY, m20_budget,
                                run, sha, ssh, storage_hourly, update_ssh)

RESULT = REPO / "results" / "m20_beir15_cloud.json"
LOG = REPO / "logs" / "m20-beir15-cloud.log"
M20_REGISTRY = REPO / "m20" / "beir15_registry.json"
RESERVED_RESULT = REPO / "results" / "m13_reserved_run.json"
SIX_RESULT = REPO / "results" / "m10_final_run.json"
BEIR15_RESULT = REPO / "results" / "m20_beir15_run.json"
ARCHIVE_MANIFEST = REPO / "results" / "m20_archive_manifest.json"
REMOTE_ARCHIVE = "work/m20-archive"
LOCAL_ARCHIVE = Path("/mnt/d/constella-archive/beir15")
# Stage C writes fp16 vectors for BEIR-15's ~23.74M new documents across three towers:
# 23,744,806 x (1024 + 384 + 768) x 2 bytes = 103 GB. Stage D adds about 25 GB of gzipped
# payloads; the archive's vectors are HARD LINKS to the shards, not a second 147 GB copy. Raw
# corpus downloads and the datasets cache account for the rest.
MIN_REMOTE_FREE_BYTES = 170_000_000_000


def save(record):
    record["updated_utc"] = datetime.now(timezone.utc).isoformat()
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    pending = RESULT.with_suffix(".pending.json")
    pending.write_text(json.dumps(record, indent=2) + "\n")
    pending.replace(RESULT)


def stage_caps():
    caps = m20_budget()["stage_caps_hours"]
    return caps["C_beir15"], caps["D_archive"]


def remote_command(stage_c_deadline_epoch, cap_c, cap_d):
    """Stage C and stage D under SEPARATE wall clocks, like the reserved controller.

    `timeout` is the hard backstop; the projection gate inside the pre-encode is the soft one that
    stops at a shard boundary. Neither stage is protected access, so a kill costs only time.
    """
    env = (
        "export HF_HOME=/home/dylan/.cache/huggingface "
        "HF_HUB_CACHE=/home/dylan/.cache/huggingface/hub "
        "HF_DATASETS_CACHE=/home/dylan/.cache/huggingface/datasets "
        "XDG_CACHE_HOME=/home/dylan/.cache M7_DEVICE=cuda M7_ENCODER=stella-400M-v5 "
        "OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 "
        "TOKENIZERS_PARALLELISM=false "
        "PYTHONPATH=m20src:m13src:m12src:m8src:m7src:bench; "
    )
    towers = "nano-dense,bge-small-en-v1.5,leaf-ir-asym"
    checks = [
        ".venv/bin/python -m py_compile m20src/roster.py m20src/beir15.py m20src/archive.py",
        ".venv/bin/python -c 'import roster; print(roster.assert_registered_identities())'",
    ]
    stage_c = "; ".join(
        [f".venv/bin/python -u m8src/pre_encode.py --system {tower} --device cuda --batch beir15 "
         f"--stage-deadline {stage_c_deadline_epoch:.0f} --tower-order {towers}"
         for tower in towers.split(",")]
        + [".venv/bin/python -u m20src/beir15.py --device cuda"])
    stage_d = "; ".join([
        f".venv/bin/python -u m20src/archive.py --build --root {REMOTE_ARCHIVE}",
        f".venv/bin/python -u m20src/archive.py --verify --root {REMOTE_ARCHIVE}",
    ])
    body = "; ".join(checks) + "; " \
        + f"timeout {int(cap_c * 3600)} bash -c " + shlex.quote("set -e; " + stage_c) + "; " \
        + f"timeout {int(cap_d * 3600)} bash -c " + shlex.quote("set -e; " + stage_d)
    return "cd " + shlex.quote(REMOTE) + "; " + env + "set -e; " + body


def require_reserved_complete():
    if not RESERVED_RESULT.is_file():
        raise RuntimeError("results/m13_reserved_run.json is missing: stage B has not completed, "
                           "and BEIR-15 cannot carry the reserved rows without it")
    six = json.loads(SIX_RESULT.read_text())
    if six.get("end_status") != "COMPLETE":
        raise RuntimeError("results/m10_final_run.json does not end COMPLETE; finish stage B first")
    if subprocess.run(["git", "ls-remote", "--exit-code", "origin",
                       "refs/tags/m8-reserved-spent"], cwd=REPO, capture_output=True).returncode:
        raise RuntimeError("m8-reserved-spent is not on origin; the reserved receipt is not durable")


def pull_results(record, timeout_seconds):
    """Bring BEIR-15's results back, check they are COMPLETE, and make them durable.

    The pod writes these files and never pushes them. Without this the controller could report
    PASSED while the entire descriptive table existed only on a machine about to be stopped.
    """
    for relative in ("results/m20_beir15_run.json", "results/m20_corpus_pins.json",
                     "results/m20_beir15_preencode.json"):
        run(["scp", "-F", str(SSH_CONFIG), f"{SSH_ALIAS}:{REMOTE}/{relative}",
             str(REPO / relative)], timeout=1800)
    (REPO / "results" / "m20_beir15_scores").mkdir(parents=True, exist_ok=True)
    run(["rsync", "-a", "--partial", "-e", f"ssh -F {SSH_CONFIG}",
         f"{SSH_ALIAS}:{REMOTE}/results/m20_beir15_scores/",
         str(REPO / "results" / "m20_beir15_scores") + "/"], timeout=timeout_seconds)
    table = json.loads(BEIR15_RESULT.read_text())
    if table.get("status") != "COMPLETE":
        raise RuntimeError(f"BEIR-15 table is {table.get('status')}: missing {table.get('missing')}")
    if table.get("registry_sha256") != sha(M20_REGISTRY):
        raise RuntimeError("BEIR-15 table was produced against a different registration")
    record["beir15_result_sha256"] = sha(BEIR15_RESULT)
    record["beir15_datasets"] = len(table.get("datasets", []))
    record["beir15_systems"] = len(table.get("systems", []))


def pull_archive(record, timeout_seconds):
    """Copy the built archive to the local D: target, then re-hash it there."""
    LOCAL_ARCHIVE.mkdir(parents=True, exist_ok=True)
    run(["scp", "-F", str(SSH_CONFIG),
         f"{SSH_ALIAS}:{REMOTE}/results/m20_archive_manifest.json", str(ARCHIVE_MANIFEST)],
        timeout=1800)
    run(["rsync", "-a", "--partial", "--info=progress2", "-e", f"ssh -F {SSH_CONFIG}",
         f"{SSH_ALIAS}:{REMOTE}/{REMOTE_ARCHIVE}/", str(LOCAL_ARCHIVE) + "/"],
        timeout=timeout_seconds)
    verify = subprocess.run(
        [str(REPO / ".venv/bin/python"), str(REPO / "m20src/archive.py"),
         "--verify", "--root", str(LOCAL_ARCHIVE)], cwd=REPO, text=True, capture_output=True,
        timeout=timeout_seconds)
    record["local_archive_verify"] = verify.stdout.strip()[-2000:]
    if verify.returncode:
        raise RuntimeError(f"local archive verification failed: {verify.stdout.strip()[-500:]}")
    record["local_archive_root"] = str(LOCAL_ARCHIVE)
    record["archive_manifest_sha256"] = sha(ARCHIVE_MANIFEST)


def commit_results(record):
    paths = ["results/m20_beir15_run.json", "results/m20_beir15_scores",
             "results/m20_corpus_pins.json", "results/m20_beir15_preencode.json",
             "results/m20_archive_manifest.json"]
    existing = [p for p in paths if (REPO / p).exists()]
    run(["git", "add", "--"] + existing, cwd=REPO)
    run(["git", "commit", "-m",
         f"m20: BEIR-15 descriptive results and archive manifest "
         f"({record.get('beir15_result_sha256', '')[:12]})"], cwd=REPO)
    run(["git", "push", "origin", f"HEAD:{BRANCH}"], cwd=REPO, timeout=1800)
    record["results_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO,
                                                       text=True).strip()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-only", action="store_true",
                        help="skip stage C; build and pull the archive from what is already there")
    args = parser.parse_args(argv)

    if RESULT.exists():
        raise RuntimeError(f"preserve existing controller receipt {RESULT}")
    LOG.parent.mkdir(parents=True, exist_ok=True)
    if LOG.exists():
        raise RuntimeError(f"preserve existing cloud log {LOG}")
    require_reserved_complete()
    dirty = subprocess.check_output(["git", "status", "--porcelain=v1", "--untracked-files=all"],
                                    cwd=REPO, text=True)
    if dirty:
        raise RuntimeError("checkout is not clean: " + repr(dirty.splitlines()[:10]))
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    run(["git", "fetch", "origin", BRANCH], cwd=REPO, timeout=300)
    upstream = subprocess.check_output(["git", "rev-parse", f"origin/{BRANCH}"], cwd=REPO,
                                       text=True).strip()
    if head != upstream:
        raise RuntimeError(f"HEAD must equal pushed origin/{BRANCH}")

    cap_c, cap_d = stage_caps()
    max_hours = cap_c + cap_d
    cloud = Cloud()
    pods = [cloud.get(POD), *(cloud.get(pod) for pod in OTHER_PODS)]
    if any(pod.get("desiredStatus") != "EXITED" for pod in pods):
        raise RuntimeError("all retained pods must be stopped before launch")
    target = pods[0]
    if (target.get("volumeInGb") != 500 or target.get("containerDiskInGb") != 30 or
            float(target.get("costPerHr", math.inf)) > 1.59):
        raise RuntimeError("retained A100 storage or quote changed")
    target_total_hourly = float(target["costPerHr"]) + storage_hourly(target)
    if target_total_hourly > MAX_TOTAL_HOURLY + 1e-9:
        raise RuntimeError("target compute plus its storage exceeds the registered hourly rate")
    all_retained_hourly = target_total_hourly + sum(storage_hourly(p) for p in pods[1:])
    balance_start = cloud.balance()
    if balance_start < max_hours * all_retained_hourly:
        raise RuntimeError(f"wallet cannot cover {max_hours} h at ${all_retained_hourly:.4f}/h")

    record = {"status": "RUNNING", "stage": "resuming", "pod_id": POD,
              "implementation_commit": head, "archive_only": bool(args.archive_only),
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "balance_start_usd": balance_start,
              "stage_cap_hours": {"C_beir15": cap_c, "D_archive": cap_d},
              "max_hours": max_hours, "all_retained_hourly_usd": all_retained_hourly,
              "m20_registry_sha256": sha(M20_REGISTRY),
              "reserved_result_sha256": sha(RESERVED_RESULT)}
    save(record)
    started = False

    def interrupted(signum, _frame):
        raise KeyboardInterrupt(f"controller interrupted by signal {signum}")

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupted)
    deadline = time.monotonic() + max_hours * 3600
    try:
        cloud.resume()
        started = True
        record["stage"] = "waiting-for-ssh"
        save(record)
        ready_deadline = time.monotonic() + 900
        while time.monotonic() < ready_deadline:
            pod = cloud.get()
            if float(pod.get("costPerHr", math.inf)) + storage_hourly(pod) > MAX_TOTAL_HOURLY + 1e-9:
                raise RuntimeError("live target hourly cost exceeds the registered rate")
            if pod.get("desiredStatus") == "RUNNING" and pod.get("gpuCount") == 1 \
                    and update_ssh(pod):
                break
            time.sleep(5)
        else:
            raise RuntimeError("SSH readiness timeout")

        record["stage"] = "deploying"
        save(record)
        ssh("cd " + shlex.quote(REMOTE) + "; set -e; git fetch origin " + BRANCH +
            "; git checkout -B " + BRANCH + " origin/" + BRANCH +
            "; git branch --set-upstream-to=origin/" + BRANCH + " " + BRANCH, timeout=600)
        remote_head = ssh("cd " + shlex.quote(REMOTE) + "; git rev-parse HEAD", timeout=120,
                          capture_output=True).stdout.strip()
        if remote_head != head:
            raise RuntimeError(f"remote HEAD {remote_head} differs from {head}")
        free = int(ssh("python3 -c " + shlex.quote(
            "import shutil; print(shutil.disk_usage('/home/dylan').free)"),
            capture_output=True).stdout.strip())
        record["remote_free_bytes"] = free
        save(record)
        if free < MIN_REMOTE_FREE_BYTES:
            raise RuntimeError(f"remote has {free} free bytes; BEIR-15 vectors plus the staged "
                               f"archive need at least {MIN_REMOTE_FREE_BYTES}")

        remote_log, remote_exit = "/tmp/m20-beir15.log", "/tmp/m20-beir15.exit"
        stage_c_deadline = time.time() + cap_c * 3600
        record["stage_c_deadline_epoch"] = stage_c_deadline
        command = (remote_command(stage_c_deadline, cap_c, cap_d) if not args.archive_only else
                   "cd " + shlex.quote(REMOTE) +
                   "; set -e; PYTHONPATH=m20src:m13src:m12src:m8src:m7src:bench "
                   f"timeout {int(cap_d * 3600)} .venv/bin/python -u m20src/archive.py "
                   f"--build --verify --root {REMOTE_ARCHIVE}")
        inner = "( " + command + " ); rc=$?; echo $rc > " + remote_exit
        launch = ("rm -f " + remote_exit + " " + remote_log + "; nohup bash -lc " +
                  shlex.quote(inner) + " > " + remote_log + " 2>&1 < /dev/null & echo $!")
        remote_pid = int(ssh(launch, capture_output=True).stdout.strip())
        record.update(stage="running", remote_pid=remote_pid, remote_log=remote_log)
        save(record)
        last_tail = ""
        while time.monotonic() < deadline:
            poll = ssh("if kill -0 " + str(remote_pid) + " 2>/dev/null; then echo RUNNING; "
                       "elif test -f " + remote_exit + "; then echo EXIT:$(cat " + remote_exit +
                       "); else echo LOST; fi; tail -n 1 " + remote_log + " 2>/dev/null || true",
                       capture_output=True)
            rows = poll.stdout.splitlines()
            state = rows[0] if rows else "LOST"
            tail = rows[-1] if len(rows) > 1 else ""
            if tail and tail != last_tail:
                print(tail, flush=True)
                last_tail = tail
                record["last_remote_line"] = tail[-500:]
                save(record)
            if state.startswith("EXIT:"):
                code = int(state.split(":", 1)[1])
                if code != 0:
                    raise RuntimeError(f"remote BEIR-15 runner exited {code}: {tail}")
                break
            if state == "LOST":
                raise RuntimeError("remote process disappeared without an exit receipt")
            time.sleep(30)
        else:
            raise TimeoutError(f"stage C/D exceeded its {max_hours}-hour allowance")

        run(["scp", "-F", str(SSH_CONFIG), SSH_ALIAS + ":" + remote_log, str(LOG)], timeout=600)
        transfer_budget = max(600, int(deadline - time.monotonic()))
        record["stage"] = "pulling-results"
        save(record)
        if not args.archive_only:
            pull_results(record, transfer_budget)
        record["stage"] = "pulling-archive"
        save(record)
        pull_archive(record, max(600, int(deadline - time.monotonic())))
        record["stage"] = "publishing"
        save(record)
        commit_results(record)
        record.update(status="PASSED", stage="finished")
    except BaseException as error:
        record.update(status="FAILED", stage="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if started:
            try:
                record["stop_response"] = cloud.post("stop")
                stop_deadline = time.monotonic() + 600
                while time.monotonic() < stop_deadline:
                    if cloud.get().get("desiredStatus") == "EXITED":
                        record["pod_final_status"] = "EXITED"
                        break
                    time.sleep(5)
                else:
                    record["pod_final_status"] = "STOP_TIMEOUT"
            except Exception as stop_error:
                record["pod_final_status"] = "STOP_FAILED"
                record["stop_error"] = f"{type(stop_error).__name__}: {stop_error}"
        try:
            record["balance_end_usd"] = cloud.balance()
            record["wallet_delta_usd"] = record["balance_start_usd"] - record["balance_end_usd"]
        except Exception as balance_error:
            record["balance_error"] = f"{type(balance_error).__name__}: {balance_error}"
        save(record)

    if record.get("pod_final_status") != "EXITED":
        raise RuntimeError("stage C/D finished but the paid pod did not confirm STOP")
    print(json.dumps({key: record.get(key) for key in
                      ("status", "pod_final_status", "wallet_delta_usd",
                       "beir15_result_sha256", "archive_manifest_sha256")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
