#!/usr/bin/env python3
"""WSL-side initial upload/validation job; no registered training or protected evaluation.

Requires the provisioned Pod and completed bootstrap. Credentials stay on WSL.
Always requests STOP (never termination) after success, failure, or a six-hour deadline.
Keep WSL running: this local process cannot stop a Pod if the whole host powers off.
"""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time
import urllib.request

REPO = Path(__file__).resolve().parents[1]
REMOTE = "/home/dylan/asymetric-dual-encoders"
CONFIG = Path.home() / ".config/runpod"
SSH = ["ssh", "-F", str(CONFIG / "m13_ssh_config"), "m13-runpod"]
RESULT = REPO / "results/m13_cloud_stage0.json"
LOGS = REPO / "logs"


def main():
    setup = json.loads((REPO / "results/m13_cloud_setup.json").read_text())
    pod_id = setup["pod"]["id"]
    if RESULT.exists():
        raise SystemExit("Stage-0 receipt already exists; inspect it before any retry.")
    LOGS.mkdir(exist_ok=True)
    deadline = time.monotonic() + 6 * 3600
    receipt = {"pod_id": pod_id, "started_utc": utc(), "stage": "starting",
               "registered_training": False, "protected_evaluation": False,
               "steps": [], "status": "RUNNING"}
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    receipt["code_commit"] = head
    # Exclusive receipt reservation prevents two local launchers owning the same Pod.
    with RESULT.open("x") as initial:
        json.dump(receipt, initial)

    def interrupted(signum, frame):
        raise InterruptedError(f"Stage-0 interrupted by signal {signum}")

    for signum in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(signum, interrupted)

    def save():
        temp = RESULT.with_suffix(".tmp")
        temp.write_text(json.dumps(receipt, indent=2) + "\n")
        os.replace(temp, RESULT)

    def run(name, command, timeout):
        receipt["stage"] = name
        save()
        print(utc(), name, flush=True)
        start = time.monotonic()
        log = LOGS / ("m13-cloud-" + name + ".log")
        with log.open("w") as output:
            child = subprocess.Popen(command, cwd=REPO, stdout=output,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = child.wait(timeout=min(timeout, max(1, deadline-start)))
            finally:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=30)
        receipt["steps"].append({"name": name, "exit_code": code,
                                  "elapsed_seconds": round(time.monotonic()-start, 2),
                                  "log": str(log.relative_to(REPO))})
        save()
        if code:
            raise RuntimeError(f"{name} failed with exit {code}; inspect {log}")

    def remote(name, command, timeout):
        # timeout also bounds the remote child if the SSH connection disappears.
        # Noninteractive SSH does not inherit the Pod container's configured env.
        command = ("export HF_HOME=/home/dylan/.cache/huggingface "
                   "HF_HUB_CACHE=/home/dylan/.cache/huggingface/hub "
                   "HF_DATASETS_CACHE=/home/dylan/.cache/huggingface/datasets "
                   "XDG_CACHE_HOME=/home/dylan/.cache; " + command)
        run(name, SSH + ["timeout --kill-after=30s " + str(timeout) + "s bash -c "
                        + shlex.quote(command)], timeout+60)

    try:
        save()
        local_inputs = ["m13/cloud_transfer_files.txt", "m13/cloud_transfer_sha256.txt",
                        "scripts/m13_cloud_stage0.py", "scripts/m13_cloud_smoke.py"]
        subprocess.run(["git", "diff", "--exit-code", "HEAD", "--"] + local_inputs,
                       cwd=REPO, check=True, stdout=subprocess.DEVNULL)
        remote("preflight", "set -eu; "
               "command -v rsync >/dev/null; "
               "mountpoint -q /home/dylan; "
               "findmnt -rn -o SOURCE /home/dylan | grep -F -- " + shlex.quote(pod_id) + "; "
               "grep -q 'Pinned packages, imports and CUDA bf16 allocation passed.' "
               "/home/dylan/setup-logs/bootstrap.log; "
               "cd " + REMOTE + "; test \"$(git rev-parse HEAD)\" = " + shlex.quote(head) + "; "
               "git diff --quiet HEAD; "
               "test -f scripts/m13_cloud_smoke.py", 60)
        run("upload", ["rsync", "-a", "--no-owner", "--no-group", "--no-perms",
                       "--partial", "--append-verify", "--compress",
                       "--compress-choice=zstd", "--compress-level=3",
                       "--info=progress2", "--stats",
                       "--files-from=" + str(REPO / "m13/cloud_transfer_files.txt"),
                       "-e", "ssh -F " + shlex.quote(str(CONFIG / "m13_ssh_config")),
                       "/", "m13-runpod:/"], 4*3600)
        remote("checksums", "cd / && sha256sum -c " + REMOTE
               + "/m13/cloud_transfer_sha256.txt", 1800)
        remote("checks", "cd " + REMOTE + " && bash run_checks.sh", 1200)
        remote("plan", "cd " + REMOTE + " && .venv/bin/python m10src/run_arm.py --plan", 120)
        remote("shapes", "cd " + REMOTE + " && "
               "OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 "
               "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 "
               ".venv/bin/python -u m10src/arm_smoke.py "
               "--only E-bs32 E-bs128 --device cuda --max-len 512 "
               "--out results/m13_cloud_arm_smoke.json", 1200)
        remote("resume", "cd " + REMOTE + " && "
               "OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 "
               "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 "
               ".venv/bin/python -u scripts/m13_cloud_smoke.py", 1800)
        receipt["status"] = "PASSED"
    except BaseException as error:
        receipt["status"] = "FAILED"
        receipt["error"] = f"{type(error).__name__}: {error}"
        print(receipt["error"], flush=True)
    finally:
        # Repeated termination signals must not interrupt the bounded STOP cleanup.
        for signum in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
            signal.signal(signum, signal.SIG_IGN)
        # Copy only this job's evidence, including failures, before stopping compute.
        try:
            backup = REPO / "work/m13cloud-evidence"
            backup.mkdir(exist_ok=True)
            run("evidence", ["rsync", "-a", "--no-owner", "--no-group", "--no-perms",
                             "--ignore-missing-args", "-e",
                             "ssh -F " + shlex.quote(str(CONFIG / "m13_ssh_config")),
                             "m13-runpod:" + REMOTE + "/results/m13_cloud_arm_smoke.json",
                             "m13-runpod:" + REMOTE + "/results/m13_cloud_resume_smoke.json",
                             "m13-runpod:" + REMOTE + "/work/m13cloud-smoke",
                             "m13-runpod:" + REMOTE + "/work/m10arms/smoke/E-bs32",
                             "m13-runpod:" + REMOTE + "/work/m10arms/smoke/E-bs128",
                             "m13-runpod:/home/dylan/setup-logs",
                             str(backup) + "/"], 600)
            smoke_result = backup / "m13_cloud_resume_smoke.json"
            if receipt["status"] == "PASSED" and not smoke_result.is_file():
                raise RuntimeError("Missing completed smoke result in local backup")
            if smoke_result.exists():
                summary = json.loads(smoke_result.read_text())
                if receipt["status"] == "PASSED":
                    if summary.get("status") != "complete" or set(summary.get("arms", {})) != {"E-bs32", "E-bs128"}:
                        raise RuntimeError("Local backup lacks both completed E smoke arms")
                for arm, row in summary.get("arms", {}).items():
                    if "interrupted" in row:
                        verify(backup / "m13cloud-smoke" / (arm + ".interrupted.pt"),
                               row["interrupted"]["sha256"])
                    if row.get("passed"):
                        record = backup / arm / "record.json"
                        verify(record, row["record_sha256"])
                        for checkpoint in json.loads(record.read_text())["checkpoints"].values():
                            verify(backup / arm / Path(checkpoint["path"]).name,
                                   checkpoint["sha256"])
            receipt["backup_verified"] = True
        except BaseException as error:
            receipt["backup_error"] = f"{type(error).__name__}: {error}"
            receipt["status"] = "FAILED"
        try:
            key = (CONFIG / "api_key").read_text().strip()
            url = "https://rest.runpod.io/v1/pods/" + pod_id
            headers = {"Authorization": "Bearer " + key, "Content-Type": "application/json",
                       "User-Agent": "m13-stage0/1.0"}
            stop_deadline = time.monotonic() + 120
            for attempt in range(6):
                try:
                    request = urllib.request.Request(url + "/stop", data=b"{}", headers=headers)
                    with urllib.request.urlopen(request, timeout=10) as response:
                        receipt["stop_http_status"] = response.status
                    time.sleep(2)
                    with urllib.request.urlopen(urllib.request.Request(url, headers=headers),
                                                timeout=10) as response:
                        state = json.load(response)["desiredStatus"]
                    receipt["pod_final_status"] = state
                    if state == "EXITED":
                        break
                except Exception as error:
                    receipt["last_stop_attempt_error"] = type(error).__name__
                if time.monotonic() >= stop_deadline:
                    break
                time.sleep(3)
            if receipt.get("pod_final_status") != "EXITED":
                raise RuntimeError("STOP could not be confirmed")
            print(utc(), "STOP confirmed; persistent Pod disk retained", flush=True)
        except BaseException as error:
            # Exception strings must not expose the API key or response headers.
            receipt["stop_error"] = type(error).__name__
            receipt["status"] = "FAILED"
            print("STOP FAILED: inspect Pod immediately", flush=True)
        receipt["finished_utc"] = utc()
        receipt["stage"] = "finished"
        save()
    return 0 if receipt["status"] == "PASSED" else 1


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def verify(path, expected):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise RuntimeError(f"Backup checksum mismatch: {path}")


if __name__ == "__main__":
    raise SystemExit(main())
