#!/usr/bin/env python3
"""One-shot local follow-on: verified E backup -> deferred DEV-6 -> E1 selection.

No cloud calls, retries, LoTTE access or full build. Leave every failed artifact in place.
"""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

from m13_cloud_e import ALLOCATION, ARMS, BACKUP, REPO, digest, verify_backup
from m13_cloud_stage0 import utc

RESULT = REPO / "results/m13_after_e.json"
E_RESULT = REPO / "results/m13_cloud_e.json"
PYTHON = str(REPO / ".venv/bin/python")
ARM_RECORDS = [f"results/m10_arm_{arm}.json" for arm in ARMS]


def main():
    receipt = {"status": "RUNNING", "stage": "waiting", "started_utc": utc(),
               "protected_evaluation": False, "steps": []}
    with RESULT.open("x") as stream:
        json.dump(receipt, stream)

    def save():
        temp = RESULT.with_suffix(".tmp")
        temp.write_text(json.dumps(receipt, indent=2) + "\n")
        os.replace(temp, RESULT)

    def interrupted(signum, frame):
        raise InterruptedError(f"Interrupted by signal {signum}")

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupted)

    def run(name, command, seconds=120):
        receipt["stage"] = name
        save()
        log = REPO / "logs" / f"m13-after-e-{name}.log"
        log.parent.mkdir(exist_ok=True)
        print(utc(), name, flush=True)
        with log.open("x") as output:
            child = subprocess.Popen(command, cwd=REPO, stdout=output,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = child.wait(timeout=seconds)
            finally:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=30)
        receipt["steps"].append({"name": name, "exit_code": code, "log": str(log.relative_to(REPO))})
        save()
        if code:
            raise RuntimeError(f"{name} failed: exit {code}; see {log}")

    def clean():
        subprocess.run(["git", "diff", "--quiet", "HEAD"], cwd=REPO, check=True, timeout=30)

    def collisions():
        for arm in ARMS:
            for path in (REPO / "work/m10arms" / arm, REPO / "results" / f"m10_arm_{arm}.json"):
                if path.exists() or path.is_symlink():
                    raise RuntimeError(f"Canonical E evidence exists; refusing overwrite: {path}")

    def publish(label, paths, message):
        # Refuse any unrelated staged or unstaged tracked change before touching the index.
        changed = subprocess.check_output(["git", "diff", "HEAD", "--name-only"], cwd=REPO,
                                          text=True, timeout=30).splitlines()
        if set(changed) - set(paths):
            raise RuntimeError("Unrelated tracked changes block publication")
        run(label + "-add", ["git", "add", "--"] + paths)
        staged = subprocess.check_output(["git", "diff", "--cached", "--name-only"], cwd=REPO,
                                         text=True, timeout=30).splitlines()
        if set(staged) != set(paths):
            raise RuntimeError("Index must contain exactly the intended publication files")
        run(label + "-commit", ["git", "commit", "-m", message, "--"] + paths)
        run(label + "-push", ["git", "push"], 180)
        clean()

    def dev6_ready():
        marker_path = REPO / "results/m13_dev6_preflight.json"
        if digest(marker_path) != receipt["dev6_preflight_sha256"]:
            raise RuntimeError("DEV-6 preflight changed while waiting")
        marker = json.loads(marker_path.read_text())
        if marker.get("status") != "PASSED" or marker["registry_sha256"] != digest(REPO / "m10/screen_registry.json"):
            raise RuntimeError("DEV-6 content preflight is missing or stale")
        required_sources = {"m13src/dev6_from_checkpoint.py", "m10src/run_arm.py", "m10src/nano10.py",
            "m9src/eval9.py", "m7src/dev_eval.py", "m7src/teacher.py", "m7src/heldout.py", "m7src/devsuite.py"}
        required_components = {"nq-250k", "hotpotqa", "cqadup-programmers", "cqadup-physics", "heldout-train", "heldout-longq"}
        if not required_sources <= set(marker.get("code_sha256", {})) or set(marker.get("components", {})) != required_components:
            raise RuntimeError("DEV-6 preflight lacks required source and component identities")
        for name, expected in marker["code_sha256"].items():
            path = Path(name)
            if path.is_absolute() or ".." in path.parts or digest(REPO / path) != expected:
                raise RuntimeError(f"DEV-6 source identity changed: {name}")

    def selection_ready():
        for name, expected in receipt["selection_source_sha256"].items():
            if digest(REPO / name) != expected:
                raise RuntimeError(f"Selection code changed while waiting: {name}")
        if digest(REPO / "m10/screen_registry.json") != allocation["registry_sha256"]:
            raise RuntimeError("Live registry differs from cloud allocation")

    try:
        clean()
        receipt["selection_source_sha256"] = {name: digest(REPO / name) for name in
            ("m10src/contrasts.py", "m10src/cov_macro.py", "m10src/decision_rules.py")}
        marker_path = REPO / "results/m13_dev6_preflight.json"
        receipt["dev6_preflight_sha256"] = digest(marker_path)
        dev6_ready()
        collisions()
        deadline = time.monotonic() + 26 * 3600
        while True:
            if E_RESULT.exists():
                cloud = json.loads(E_RESULT.read_text())
                if cloud.get("stage") == "finished":
                    receipt["cloud_receipt_sha256"] = digest(E_RESULT)
                    break
            if time.monotonic() >= deadline:
                raise TimeoutError("E controller did not finish within 26 hours")
            time.sleep(30)
        if (cloud.get("status") != "PASSED" or cloud.get("backup_verified") is not True
                or cloud.get("pod_final_status") != "EXITED"):
            raise RuntimeError("E success, verified backup and confirmed STOP are required")
        allocation = json.loads(ALLOCATION.read_text())
        if digest(ALLOCATION) != cloud["allocation_sha256"]:
            raise RuntimeError("Allocation changed after E launch")
        selection_ready()
        dev6_ready()
        if digest(BACKUP / "manifest.json") != cloud["backup_manifest_sha256"]:
            raise RuntimeError("Backup manifest differs from verified cloud snapshot")
        verify_backup(BACKUP, cloud["code_commit"], allocation["registry_sha256"], True)
        if digest(E_RESULT) != receipt["cloud_receipt_sha256"]:
            raise RuntimeError("Cloud receipt changed during backup validation")
        clean()
        collisions()
        for arm in ARMS:
            destination = REPO / "work/m10arms" / arm
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(BACKUP / "work/m10arms" / arm, destination)
            source = BACKUP / "results" / f"m10_arm_{arm}.json"
            with source.open("rb") as src, (REPO / "results" / source.name).open("xb") as dst:
                shutil.copyfileobj(src, dst)
        for name, expected in json.loads((BACKUP / "manifest.json").read_text()).items():
            if digest(REPO / name) != expected:
                raise RuntimeError(f"Restored copy differs from verified backup: {name}")
        if digest(E_RESULT) != receipt["cloud_receipt_sha256"]:
            raise RuntimeError("Cloud receipt changed during restoration")
        publish("trained", ARM_RECORDS, "Record both registered cloud E arms with deferred DEV-6")
        receipt["local_code_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True, timeout=30).strip()
        for arm in ARMS:
            dev6_ready()
            changed = subprocess.check_output(["git", "diff", "HEAD", "--name-only"],
                cwd=REPO, text=True, timeout=30).splitlines()
            if set(changed) - set(ARM_RECORDS):
                raise RuntimeError("Unrelated changes block DEV-6")
            run(arm + "-cuda", [PYTHON, "-c", "import torch; assert torch.cuda.is_available()"], 60)
            processes = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                text=True, timeout=30).strip()
            if processes:
                raise RuntimeError("Local GPU is busy; DEV-6 pending manual continuation")
            run(arm + "-dev6", [PYTHON, "m13src/dev6_from_checkpoint.py", arm, "--device", "cuda"], 3600)
        publish("dev6", ARM_RECORDS, "Fill deferred DEV-6 for both cloud E checkpoints")
        selection_ready()
        contrast_path = REPO / "results/m10_contrast_E1.json"
        if contrast_path.exists():
            previous = json.loads(contrast_path.read_text())
            if not previous.get("not_computed") or previous.get("registry_sha256") != allocation["registry_sha256"]:
                raise RuntimeError("Existing E1 is already computed or belongs to another registry")
        run("contrast", [PYTHON, "m10src/contrasts.py", "E1"], 3600)
        contrast = json.loads((REPO / "results/m10_contrast_E1.json").read_text())
        if contrast.get("not_computed") or contrast.get("registry_sha256") != allocation["registry_sha256"]:
            raise RuntimeError("E1 did not produce a computed contrast under the cloud registry")
        selection_ready()
        run("select", [PYTHON, "m10src/contrasts.py", "--select"], 300)
        selected = json.loads((REPO / "results/m10_screen_verdicts.json").read_text())
        if selected.get("selected", {}).get("batch") not in ("bs32", "bs128") or selected.get("registry_sha256") != allocation["registry_sha256"]:
            raise RuntimeError("Selection did not resolve the registered E batch")
        publish("selection", ["results/m10_contrast_E1.json", "results/m10_screen_verdicts.json"],
                "Resolve E1 and apply the registered recipe selection")
        receipt["status"] = "PASSED"
    except BaseException as error:
        receipt["status"] = "FAILED"
        receipt["error"] = f"{type(error).__name__}: {error}"
        print(receipt["error"], flush=True)
    finally:
        receipt["stage"] = "finished"
        receipt["finished_utc"] = utc()
        save()
    return 0 if receipt["status"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
