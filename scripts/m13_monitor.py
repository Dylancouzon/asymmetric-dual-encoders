#!/usr/bin/env python3
"""Read-only job watchdog. Paths are relative to config root (default: cwd).

Only health_path and alerts_path are written; workloads are never modified.
Run with --config FILE [--once]. Logs are explicit paths, never recursive scans.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time


def read_json(path):
    return json.loads(path.read_text())


def process_identity(pid):
    """Linux boot ID and process start ticks protect against PID reuse."""
    text = Path(f"/proc/{pid}/stat").read_text()
    fields = text[text.rfind(")") + 2:].split()
    if fields[0] == "Z":
        raise ProcessLookupError("controller is a zombie")
    return f"{Path('/proc/sys/kernel/random/boot_id').read_text().strip()}:{pid}:{fields[19]}"


def check_job(job, root, previous, now):
    result = {"name": job["name"], "state": "pending", "detail": "receipt absent"}
    if previous.get("pid_identity"):
        result["pid_identity"] = previous["pid_identity"]
    receipt = root / job["receipt"]
    try:
        raw = receipt.read_bytes()
    except FileNotFoundError:
        return result
    except OSError as exc:
        result.update(state="error", detail=f"unreadable receipt: {type(exc).__name__}")
        return result
    try:
        data = json.loads(raw)
        status = data.get("status", "").upper()
        result.update(stage=data.get("stage", ""), updated=receipt.stat().st_mtime)
        if status == "PASSED":
            result.update(state="passed", detail="receipt passed")
            return result
        if status == "FAILED":
            result.update(state="failed", detail="receipt failed")
            recovery = job.get("recovery_receipt")
            if recovery and (root / recovery).exists():
                recovered = read_json(root / recovery)
                digest = recovered.get("artifact_sha256", {}).get(job["receipt"])
                if recovered.get("status") == "PASSED" and digest == hashlib.sha256(raw).hexdigest():
                    result.update(state="recovered", detail="recovery binds exact failed receipt")
                    result["updated"] = (root / recovery).stat().st_mtime
            return result
        if status not in ("RUNNING", "WAITING", "PENDING"):
            raise ValueError(f"unknown receipt status {status!r}")
        result.update(state="running", detail="recent activity")
        if job.get("pid_file"):
            try:
                pid = int((root / job["pid_file"]).read_text().strip())
                if pid <= 0:
                    raise ValueError("PID must be positive")
                identity = process_identity(pid)
                expected = previous.get("pid_identity")
                result["pid_identity"] = expected or identity
                if expected and identity != expected:
                    result.update(state="dead", detail="controller PID identity changed")
                    return result
            except (OSError, ValueError) as exc:
                result.update(state="dead", detail=f"controller unavailable: {type(exc).__name__}")
                return result
        paths = [root / item for item in job.get("logs", [])]
        if job.get("log"):
            paths.append(root / job["log"])
        updated = [receipt.stat().st_mtime]
        for path in paths:
            try:
                updated.append(path.stat().st_mtime)
                with path.open("rb") as stream:
                    stream.seek(max(0, path.stat().st_size - 65536))
                    tail = stream.read().decode("utf-8", errors="replace")
                if re.search(r"Traceback \(most recent call last\):|CUDA out of memory|"
                             r"torch\.OutOfMemoryError|^[\w.]*Error:|^(?:FAILED|Killed)\b",
                             tail, re.I | re.M):
                    result.update(state="error", detail=f"failure marker in {path.relative_to(root)}")
            except FileNotFoundError:
                pass  # A later arm's log need not exist yet.
        result["updated"] = max(updated)
        result["stale_seconds"] = job.get("stale_seconds", 900)
        if result["state"] == "running" and now - max(updated) > result["stale_seconds"]:
            result.update(state="stale", detail="no recent receipt or log activity")
        return result
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        result.update(state="error", detail=f"unreadable evidence: {type(exc).__name__}")
        return result


def poll(config, now=None):
    now = time.time() if now is None else now
    root = Path(config.get("root", ".")).resolve()
    health_path = root / config.get("health_path", "work/m13-monitor/health.json")
    alerts_path = root / config.get("alerts_path", "work/m13-monitor/alerts.jsonl")
    try:
        previous = {item["name"]: item for item in read_json(health_path)["jobs"]}
    except FileNotFoundError:
        previous = {}
    jobs = config["jobs"]
    command = config.get("notification_command")
    if command is not None and (not isinstance(command, list) or not command or
                                not all(isinstance(arg, str) and arg for arg in command)):
        raise ValueError("notification_command must be a nonempty argv list")
    names = [job["name"] for job in jobs]
    if len(set(names)) != len(names):
        raise ValueError("duplicate job names")
    results = {job["name"]: check_job(job, root, previous.get(job["name"], {}), now) for job in jobs}
    for job in jobs:
        result = results[job["name"]]
        dependency = job.get("wait_for")
        if dependency and dependency not in results:
            raise ValueError(f"unknown dependency {dependency}")
        if dependency and result.get("stage") == "waiting" and result["state"] in ("running", "stale"):
            upstream = results[dependency]
            if upstream["state"] in ("running", "waiting"):
                result.update(state="waiting", detail=f"waiting for {dependency}")
            elif upstream["state"] in ("passed", "recovered"):
                if now - max(result["updated"], upstream["updated"]) <= result["stale_seconds"]:
                    result.update(state="running", detail=f"awaiting handoff from {dependency}")
            else:
                result.update(state="error", detail=f"dependency {dependency} is {upstream['state']}")
    transitions = []
    for name, result in results.items():
        old = previous.get(name, {})
        keys = ("state", "detail", "stage")
        if any(old.get(key) != result.get(key) for key in keys):
            transitions.append({"timestamp": now, "previous_state": old.get("state"), **result})
    for event in transitions:
        if config.get("notification_command"):
            try:
                subprocess.run(config["notification_command"], input=json.dumps(event),
                               text=True, timeout=15, check=True, cwd=root, capture_output=True)
            except (OSError, subprocess.SubprocessError) as exc:
                event["notification_error"] = f"{type(exc).__name__}: {exc}"
                results[event["name"]]["notification_error"] = event["notification_error"]
    for name, result in results.items():
        if not any(event["name"] == name for event in transitions) and previous.get(name, {}).get("notification_error"):
            result["notification_error"] = previous[name]["notification_error"]
    health = {"timestamp": now, "jobs": list(results.values())}
    for path in (health_path, alerts_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    # Publish alerts first: interruption may duplicate an alert but cannot silently lose it.
    with alerts_path.open("a") as stream:
        for transition in transitions:
            stream.write(json.dumps(transition, sort_keys=True) + "\n")
    temporary = health_path.with_name(health_path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(health, indent=2) + "\n")
    os.replace(temporary, health_path)
    for transition in transitions:
        print(json.dumps(transition, sort_keys=True), flush=True)
    return health


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    config = read_json(args.config)
    while True:
        poll(config)
        if args.once:
            break
        time.sleep(300)


if __name__ == "__main__":
    main()
