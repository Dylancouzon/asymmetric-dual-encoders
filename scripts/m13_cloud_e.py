#!/usr/bin/env python3
"""Bounded WSL controller for the two registered E arms, with DEV-6 deferred.

Operate only the existing Pod. Never retries an arm, opens protected evaluation,
starts a Pod, or terminates storage. Keep WSL awake until STOP is confirmed.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time
import urllib.request
import uuid

from m13_cloud_stage0 import CONFIG, LOGS, REMOTE, REPO, SSH, utc, verify

ARMS = ("E-bs32", "E-bs128")
RESULT = REPO / "results/m13_cloud_e.json"
ALLOCATION = REPO / "results/m13_cloud_allocation.json"
BACKUP = REPO / "work/m13cloud-e-backup"
ENV = ("HF_HOME=/home/dylan/.cache/huggingface "
       "HF_HUB_CACHE=/home/dylan/.cache/huggingface/hub "
       "HF_DATASETS_CACHE=/home/dylan/.cache/huggingface/datasets "
       "XDG_CACHE_HOME=/home/dylan/.cache HF_HUB_OFFLINE=1 "
       "TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 "
       "OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def validate_allocation(a, setup, stage0):
    def number(key):
        value = a[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Invalid allocation number: {key}")
        return value
    if a["status"] != "READY_FOR_E" or a["pod_id"] != setup["pod"]["id"]:
        raise ValueError("Allocation not ready for this Pod")
    if number("budget_ceiling_usd") != 1000 or not 0 < number("projected_total_usd") <= 1000:
        raise ValueError("Allocation exceeds registered budget")
    if number("total_price_usd_per_hour") < setup["pod"]["costPerHr"]:
        raise ValueError("Allocation price is below provisioned GPU price")
    for key, path in (("config_sha256", REPO / "m13/build_config.json"),
                      ("registry_sha256", REPO / "m10/screen_registry.json"),
                      ("stage0_sha256", REPO / "results/m13_cloud_stage0.json"),
                      ("e_preflight_sha256", REPO / "results/m13_e_preflight.json"),
                      ("benchmark_sha256", REPO / "results/m13_encode_benchmark.json")):
        verify(path, a[key])
    if (stage0.get("status") != "PASSED" or not stage0.get("backup_verified")
            or stage0.get("pod_final_status") != "EXITED" or stage0.get("pod_id") != a["pod_id"]):
        raise ValueError("Prior smoke, backup and STOP must all have passed on this Pod")
    for name in ("m13_e_preflight.json", "m13_encode_benchmark.json"):
        evidence = json.loads((REPO / "results" / name).read_text())
        if evidence.get("status") != "PASSED":
            raise ValueError(f"Required readiness evidence not passed: {name}")
        if name == "m13_e_preflight.json" and evidence.get("registry_sha256") != a["registry_sha256"]:
            raise ValueError("E preflight does not match current config and registry")
    if set(a["e_arms"]) != set(ARMS):
        raise ValueError("Allocation must cover exactly both E arms")
    limits = [a["e_arms"][arm]["timeout_seconds"] for arm in ARMS]
    if any(type(x) is not int or not 0 < x <= 43200 for x in limits):
        raise ValueError("Each E arm requires a positive timeout <=12h")
    total = number("e_total_timeout_seconds")
    if not sum(limits) + 1800 <= total <= 86400:
        raise ValueError("Total E cap must cover arm timeouts plus cleanup, within 24h")


def verify_backup(root, head, registry_sha, require_complete):
    """Verify every copied byte against remote hashes, then record-referenced evidence."""
    manifest = json.loads((root / "manifest.json").read_text())
    for name, expected in manifest.items():
        path = Path(name)
        if (path.is_absolute() or ".." in path.parts or not (
                any(path.is_relative_to(Path("work/m10arms") / arm) for arm in ARMS)
                or str(path) in {f"results/m10_arm_{arm}.json" for arm in ARMS})):
            raise ValueError("Unexpected backup manifest path")
        verify(root / path, expected)
    outcomes = {}
    for arm in ARMS:
        directory = Path("work/m10arms") / arm
        record_path = root / directory / "record.json"
        if not record_path.exists():
            if require_complete:
                raise ValueError(f"Missing {arm} record")
            continue
        record = json.loads(record_path.read_text())
        published = root / "results" / f"m10_arm_{arm}.json"
        if published.exists():
            verify(published, digest(record_path))
        elif require_complete or record.get("terminal") or record.get("status") != "running":
            raise ValueError(f"Missing published terminal record: {arm}")
        if record["arm"] != arm or record.get("smoke") or record["git_head"] != head:
            raise ValueError(f"Wrong registered identity: {arm}")
        if record["registry_sha256"] != registry_sha:
            raise ValueError(f"Wrong registry: {arm}")
        outcomes[arm] = record["status"]
        if require_complete and (record["status"] != "complete" or not record.get("complete")
                                 or (record.get("dev6") or {}).get("deferred") is not True
                                 or not record.get("final_checkpoint")
                                 or not record.get("final_checkpoint_sha256")):
            raise ValueError(f"{arm} not complete with DEV-6 deferred")
        references = [(v["path"], v["sha256"]) for v in record.get("checkpoints", {}).values()]
        references += [(v["per_query_scores"], v["per_query_scores_sha256"])
                       for v in record.get("cov", {}).get("per_checkpoint", [])]
        if record.get("final_checkpoint"):
            references.append((record["final_checkpoint"], record["final_checkpoint_sha256"]))
        for name, expected in references:
            path = Path(name)
            if path.is_absolute() or ".." in path.parts or not path.is_relative_to(directory):
                raise ValueError(f"Evidence escapes arm directory: {name}")
            if name not in manifest:
                raise ValueError(f"Referenced evidence missing from manifest: {name}")
            verify(root / path, expected)
    return outcomes


def main():
    setup = json.loads((REPO / "results/m13_cloud_setup.json").read_text())
    stage0 = json.loads((REPO / "results/m13_cloud_stage0.json").read_text())
    allocation = json.loads(ALLOCATION.read_text())
    validate_allocation(allocation, setup, stage0)
    # Refuse before taking ownership if earlier records or backup evidence exist.
    if RESULT.exists() or BACKUP.exists():
        raise SystemExit("Registered E receipt/backup already exists; preserve it, never rerun.")
    for arm in ARMS:
        if ((REPO / "work/m10arms" / arm).exists()
                or (REPO / "results" / f"m10_arm_{arm}.json").exists()):
            raise SystemExit(f"Local registered evidence already exists for {arm}")
    subprocess.run(["git", "diff", "--quiet", "HEAD"], cwd=REPO, check=True)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    subprocess.run(["git", "ls-files", "--error-unmatch", str(ALLOCATION.relative_to(REPO)),
                    "scripts/m13_cloud_e.py"], cwd=REPO, check=True, stdout=subprocess.DEVNULL)
    pod_id = setup["pod"]["id"]
    job = uuid.uuid4().hex
    pidfile = "/tmp/m13-e-" + job + ".pid"
    receipt = {"status": "RUNNING", "started_utc": utc(), "pod_id": pod_id,
               "code_commit": head, "allocation_sha256": digest(ALLOCATION),
               "registered_training": True, "protected_evaluation": False, "steps": []}
    with RESULT.open("x") as stream:
        json.dump(receipt, stream)
    deadline = time.monotonic() + allocation["e_total_timeout_seconds"]
    LOGS.mkdir(exist_ok=True)

    def save():
        temp = RESULT.with_suffix(".tmp")
        temp.write_text(json.dumps(receipt, indent=2) + "\n")
        os.replace(temp, RESULT)

    def interrupt(signum, frame):
        raise InterruptedError(f"Interrupted by signal {signum}")

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupt)

    def run(name, command, seconds, cleanup=False):
        remaining = deadline - time.monotonic()
        if remaining <= 0 and not cleanup:
            raise TimeoutError("Overall E deadline reached")
        limit = seconds if cleanup else min(seconds, remaining)
        receipt["stage"] = name
        save()
        print(utc(), name, flush=True)
        started = time.monotonic()
        row = {"name": name, "log": f"logs/m13-cloud-e-{name}.log"}
        try:
            with (REPO / row["log"]).open("x") as output:
                child = subprocess.Popen(command, cwd=REPO, stdout=output,
                                         stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    row["exit_code"] = child.wait(timeout=limit)
                finally:
                    if child.poll() is None:
                        os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=30)
            if row["exit_code"]:
                raise RuntimeError(f"{name} failed: exit {row['exit_code']}")
        finally:
            row["elapsed_seconds"] = round(time.monotonic() - started, 2)
            receipt["steps"].append(row)
            save()

    def remote(name, command, seconds, training=False, cleanup=False):
        command = "set -eu; cd " + shlex.quote(REMOTE) + "; " + command
        timed = "timeout --kill-after=30s " + str(seconds) + "s bash -c " + shlex.quote(command)
        if training:
            timed = ("env M13_E_JOB_ID=" + job + " setsid bash -c " + shlex.quote(
                "echo $$ > " + pidfile + "; exec " + timed))
        run(name, SSH + [timed], seconds + 60, cleanup=cleanup)

    try:
        # Read the current price/state without placing credentials on the Pod.
        key = (CONFIG / "api_key").read_text().strip()
        headers = {"Authorization": "Bearer " + key, "User-Agent": "m13-cloud-e/1.0"}
        with urllib.request.urlopen(urllib.request.Request(
                "https://rest.runpod.io/v1/pods/" + pod_id, headers=headers), timeout=15) as response:
            live = json.load(response)
        live_price = live.get("adjustedCostPerHr") or live.get("costPerHr")
        if (live.get("desiredStatus") != "RUNNING" or not isinstance(live_price, (int, float))
                or not math.isfinite(live_price) or live_price <= 0
                or live_price > allocation["total_price_usd_per_hour"]):
            raise ValueError("Live Pod state/price does not fit approved allocation")
        receipt["live_gpu_price_usd_h"] = live_price
        remote("preflight", "command -v rsync >/dev/null; mountpoint -q /home/dylan; "
               "findmnt -rn -o SOURCE /home/dylan | grep -F -- " + shlex.quote(pod_id) + "; "
               "grep -q 'Pinned packages, imports and CUDA bf16 allocation passed.' "
               "/home/dylan/setup-logs/bootstrap.log; "
               "test -d /home/dylan/.cache/huggingface/hub; "
               "test \"$(git rev-parse HEAD)\" = " + shlex.quote(head) + "; git diff --quiet HEAD; "
               + " ".join("test ! -e " + shlex.quote(path) + ";" for arm in ARMS for path in
                          (f"work/m10arms/{arm}", f"results/m10_arm_{arm}.json")), 60)
        receipt["arm_outcomes"] = {}
        for arm in ARMS:
            remote(arm, "env " + ENV + " .venv/bin/python -u m10src/run_arm.py "
                   + arm + " --device cuda --dev6 defer",
                   allocation["e_arms"][arm]["timeout_seconds"], training=True)
            outcome_code = ("import json; from pathlib import Path; "
                            "r=json.loads(Path('work/m10arms/" + arm + "/record.json').read_text()); "
                            "assert r.get('terminal') is True; "
                            "assert r['status'] in ('complete','failed'); "
                            "print(json.dumps({'status':r['status']}))")
            remote(arm + "-outcome", ".venv/bin/python -c " + shlex.quote(outcome_code), 30)
            outcome = json.loads((LOGS / ("m13-cloud-e-" + arm + "-outcome.log")).read_text())
            receipt["arm_outcomes"][arm] = outcome["status"]
            print(utc(), arm, outcome["status"], flush=True)
            save()
        receipt["status"] = ("PASSED" if all(x == "complete" for x in receipt["arm_outcomes"].values())
                             else "FAILED")
    except BaseException as error:
        receipt["status"] = "FAILED"
        receipt["error"] = f"{type(error).__name__}: {error}"
        print(receipt["error"], flush=True)
    finally:
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, signal.SIG_IGN)
        try:
            # Only this controller's session can be killed; a reused PID is not trusted.
            stop_code = f'''import os, pathlib, signal, time
p = pathlib.Path({pidfile!r})
if p.exists():
    pid = int(p.read_text())
    proc = pathlib.Path('/proc') / str(pid)
    if proc.exists():
        marker = {('M13_E_JOB_ID=' + job).encode()!r}
        if marker not in (proc / 'environ').read_bytes().split(bytes([0])) or os.getpgid(pid) != pid:
            raise RuntimeError('Refusing to signal an unowned process')
        os.killpg(pid, signal.SIGTERM)
        time.sleep(3)
        try: os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError: pass
'''
            remote("quiesce", ".venv/bin/python -c " + shlex.quote(stop_code), 30, cleanup=True)
            # Hash only explicitly admitted E output trees, including partial failure files.
            manifest_code = '''import hashlib, json, pathlib
root = pathlib.Path('.')
paths = []
for arm in ('E-bs32', 'E-bs128'):
    d = root / 'work/m10arms' / arm
    if d.exists(): paths.extend(p for p in d.rglob('*') if p.is_file())
    r = root / 'results' / ('m10_arm_' + arm + '.json')
    if r.exists(): paths.append(r)
manifest = {}
for p in paths:
    if p.is_symlink(): raise RuntimeError('Symlink in E evidence')
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''): h.update(block)
    manifest[str(p)] = h.hexdigest()
pathlib.Path('/tmp/m13-e-MARKER-manifest.json').write_text(json.dumps(manifest))
'''.replace("MARKER", job)
            remote("manifest", ".venv/bin/python -c " + shlex.quote(manifest_code), 180, cleanup=True)
            BACKUP.mkdir()
            rsync = ["rsync", "-aR", "--no-owner", "--no-group", "--no-perms",
                     "--ignore-missing-args", "-e", "ssh -F " + shlex.quote(str(CONFIG / "m13_ssh_config"))]
            sources = ["m13-runpod:" + REMOTE + "/./" + path for arm in ARMS for path in
                       (f"work/m10arms/{arm}", f"results/m10_arm_{arm}.json")]
            run("backup", rsync + sources + [str(BACKUP) + "/"], 900, cleanup=True)
            run("backup-manifest", ["scp", "-F", str(CONFIG / "m13_ssh_config"),
                "m13-runpod:/tmp/m13-e-" + job + "-manifest.json", str(BACKUP / "manifest.json")], 60, cleanup=True)
            receipt["arm_outcomes"] = verify_backup(BACKUP, head, allocation["registry_sha256"],
                                                    receipt["status"] == "PASSED")
            receipt["backup_verified"] = True
            receipt["backup_manifest_sha256"] = digest(BACKUP / "manifest.json")
        except BaseException as error:
            receipt["status"] = "FAILED"
            receipt["backup_error"] = f"{type(error).__name__}: {error}"
        try:
            key = (CONFIG / "api_key").read_text().strip()
            url = "https://rest.runpod.io/v1/pods/" + pod_id
            headers = {"Authorization": "Bearer " + key, "Content-Type": "application/json",
                       "User-Agent": "m13-cloud-e/1.0"}
            for attempt in range(6):
                try:
                    with urllib.request.urlopen(urllib.request.Request(url + "/stop", data=b"{}", headers=headers), timeout=10) as response:
                        receipt["stop_http_status"] = response.status
                    time.sleep(2)
                    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=10) as response:
                        receipt["pod_final_status"] = json.load(response)["desiredStatus"]
                    if receipt["pod_final_status"] == "EXITED":
                        break
                except Exception as error:
                    receipt["last_stop_attempt_error"] = type(error).__name__
                time.sleep(3)
            if receipt.get("pod_final_status") != "EXITED":
                raise RuntimeError("STOP unconfirmed")
            print(utc(), "STOP confirmed; persistent disk retained", flush=True)
        except BaseException as error:
            receipt["status"] = "FAILED"
            receipt["stop_error"] = type(error).__name__
            print("STOP FAILED: inspect Pod immediately", flush=True)
        receipt["finished_utc"] = utc()
        receipt["stage"] = "finished"
        save()
    return 0 if receipt["status"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
