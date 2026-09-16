#!/usr/bin/env python3
"""Run M13's triggered reserved work on the retained A100, then STOP it.

M20 (owner rulings R20, R22, R23) owns the execution. This controller covers the two stages that
belong to the reserved access and nothing else:

  A. the unprotected corpus pre-encode for the reserved four, three document towers, pre-tag;
  B. the tagged reserved transaction over the eight-system roster.

BEIR-15 and the archive are stages C and D and live in `scripts/m20_beir15_cloud.py`. Splitting
them is deliberate: the pod stops between the two, so the irreplaceable reserved receipt is durable
before the much longer and entirely repeatable broad validation begins.

The inherited `reserved_batch_allowance` of 55.2 hours is UNCHANGED and is applied to stage B,
where its derivation is sufficient. It was computed from the Stella tower alone and never priced
bge-small or Arctic-M document vectors, so it cannot also be the cap on stage A; M20 registers
stage caps in `m20/beir15_registry.json` §budget and this controller enforces them.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shlex
import signal
import subprocess
import time
import urllib.request

REPO = Path(__file__).resolve().parents[1]
KEYS = Path("/home/dylan/.config/runpod")
SSH_CONFIG = KEYS / "m13_ssh_config"
SSH_ALIAS = "m13-runpod"
POD = "k3aee2m68765em"
OTHER_PODS = ("exulxoxelug5um", "wnzk8eeqrrkw4m")
REMOTE = "/home/dylan/asymetric-dual-encoders"
# M13 is merged and its topic branches are cleaned before M14 starts.  The reserved run must be
# bound to the terminal, pushed integration commit rather than to a branch that no longer exists.
BRANCH = "main"
RESULT = REPO / "results" / "m13_reserved_cloud.json"
LOG = REPO / "logs" / "m13-reserved-cloud.log"
BUILD_CONFIG = REPO / "m13" / "build_config.json"
BUILD_RECORD = REPO / "results" / "m13_build_record.json"
EVAL_MANIFEST = REPO / "results" / "eval_manifest.json"
M20_REGISTRY = REPO / "m20" / "beir15_registry.json"
DBSF_RECEIPT = REPO / "results" / "m20_dbsf_reproduction.json"
INHERITED_RESERVED_ALLOWANCE = 55.2      # m13/build_config.json; unchanged, stage B's cap
MAX_TOTAL_HOURLY = 1.6636111111111112
STORAGE_USD_PER_GB_MONTH = 0.10
BILLING_MONTH_HOURS = 720
MIN_POST_PREENCODE_FREE_BYTES = 120_000_000_000
PREENCODE_OPERATIONAL_HEADROOM_BYTES = 2_000_000_000
RESERVED_DATASETS = ("fever", "dbpedia-entity", "cqadup-android", "cqadup-english")
RESERVED_DIMS = (1024, 384, 768)


def m20_budget():
    return json.loads(M20_REGISTRY.read_text())["budget"]


def stage_caps():
    caps = m20_budget()["stage_caps_hours"]
    return caps["A_reserved_pre_encode_unprotected"], caps["B_tagged_reserved_transaction"]


def sha(path, block=8 << 20):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def storage_hourly(pod):
    """Conservative persistent + container storage quote for a retained pod."""
    volume = float(pod.get("volumeInGb") or 0)
    container = float(pod.get("containerDiskInGb") or 0)
    if volume < 0 or container < 0:
        raise RuntimeError("Runpod reported a negative storage size")
    return (volume + container) * STORAGE_USD_PER_GB_MONTH / BILLING_MONTH_HOURS


def expected_vector_bytes(eval_manifest):
    rows = eval_manifest["m7_untouched_final"]
    total_docs = sum(int(rows[dataset]["n_docs"]) for dataset in RESERVED_DATASETS)
    return total_docs * sum(RESERVED_DIMS) * 2  # normalized fp16 arrays at rest


def save(record):
    record["updated_utc"] = datetime.now(timezone.utc).isoformat()
    pending = RESULT.with_suffix(".pending.json")
    pending.write_text(json.dumps(record, indent=2) + "\n")
    pending.replace(RESULT)


class Cloud:
    def __init__(self):
        self.headers = {"Authorization": "Bearer " + (KEYS / "api_key").read_text().strip(),
                        "User-Agent": "m13-reserved/1.0"}

    def get(self, pod=POD):
        request = urllib.request.Request(f"https://rest.runpod.io/v1/pods/{pod}",
                                         headers=self.headers)
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)

    def post(self, suffix):
        request = urllib.request.Request(f"https://rest.runpod.io/v1/pods/{POD}/{suffix}",
                                         headers=self.headers, method="POST")
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read()
            return json.loads(raw) if raw else {}

    def balance(self):
        body = json.dumps({"query": "query { myself { clientBalance } }"}).encode()
        request = urllib.request.Request("https://api.runpod.io/graphql",
                                         headers={**self.headers, "Content-Type": "application/json"},
                                         data=body)
        with urllib.request.urlopen(request, timeout=30) as response:
            return float(json.load(response)["data"]["myself"]["clientBalance"])

    def resume(self):
        query = ("mutation { podResume(input: { podId: \"" + POD +
                 "\", gpuCount: 1 }) { id desiredStatus gpuCount } }")
        request = urllib.request.Request(
            "https://api.runpod.io/graphql",
            headers={**self.headers, "Content-Type": "application/json"},
            data=json.dumps({"query": query}).encode())
        with urllib.request.urlopen(request, timeout=30) as response:
            body = json.load(response)
        resumed = (body.get("data") or {}).get("podResume") or {}
        if resumed.get("id") != POD or resumed.get("gpuCount") != 1:
            raise RuntimeError(f"Runpod did not resume the exact one-GPU target: {body}")


def run(command, **kwargs):
    return subprocess.run(command, check=True, **kwargs)


def ssh(command, timeout=60, capture_output=False):
    return run(["ssh", "-n", "-F", str(SSH_CONFIG), SSH_ALIAS, command], timeout=timeout,
               text=True, capture_output=capture_output)


def update_ssh(pod):
    mappings = pod.get("portMappings") or {}
    if not pod.get("publicIp") or not mappings.get("22"):
        return False
    lines = []
    for line in SSH_CONFIG.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("HostName "):
            line = "  HostName " + pod["publicIp"]
        elif stripped.startswith("Port "):
            line = "  Port " + str(mappings["22"])
        lines.append(line)
    SSH_CONFIG.write_text("\n".join(lines) + "\n")
    try:
        ssh("true", timeout=20)
    except Exception:
        return False
    return True


def remote_command():
    env = (
        "export HF_HOME=/home/dylan/.cache/huggingface "
        "HF_HUB_CACHE=/home/dylan/.cache/huggingface/hub "
        "HF_DATASETS_CACHE=/home/dylan/.cache/huggingface/datasets "
        "XDG_CACHE_HOME=/home/dylan/.cache M7_DEVICE=cuda M7_ENCODER=stella-400M-v5 "
        "OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 "
        "TOKENIZERS_PARALLELISM=false; "
    )
    cap_a, _cap_b = stage_caps()
    commands = [
        ".venv/bin/python -m py_compile m8src/pre_encode.py m13src/reserved_support.py "
        "m13src/reserved_transaction.py m13src/score13.py m20src/roster.py m20src/beir15.py",
        "PYTHONPATH=m13src:m20src .venv/bin/python -m pytest -q m13src/test_reserved_support.py",
        # The executable roster and the pushed registration must name the same models, prompts,
        # revisions, BM25 parameters and depths before anything is encoded.
        "PYTHONPATH=m20src .venv/bin/python -c "
        "'import roster; print(roster.assert_registered_identities())'",
        ".venv/bin/python m8src/pre_encode.py --preflight-only",
        f".venv/bin/python -u m8src/pre_encode.py --system nano-dense --device cuda "
        f"--cap-hours {cap_a}",
        f".venv/bin/python -u m8src/pre_encode.py --system bge-small-en-v1.5 --device cuda "
        f"--cap-hours {cap_a}",
        f".venv/bin/python -u m8src/pre_encode.py --system leaf-ir-asym --device cuda "
        f"--cap-hours {cap_a}",
        # The zero tower resolves its pinned Hub revision, so its snapshot is fetched here rather
        # than under HF_HUB_OFFLINE; everything after this point is offline.
        "PYTHONPATH=m13src:m20src .venv/bin/python -m reserved_support "
        "--preflight-models --device cuda",
        "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 "
        ".venv/bin/python -u m13src/score13.py --reserved-only",
    ]
    return "cd " + shlex.quote(REMOTE) + "; " + env + "set -e; " + "; ".join(commands)


def checkpoint_prep_command(checkpoint_rel, build_checkpoint_rel, digest, remote=REMOTE):
    checkpoint_arg = shlex.quote(str(checkpoint_rel))
    source_arg = shlex.quote(str(build_checkpoint_rel))
    pending_arg = shlex.quote(str(checkpoint_rel) + ".pending")
    parent_arg = shlex.quote(str(Path(checkpoint_rel).parent))
    expected = shlex.quote(str(digest))
    return (
        "cd " + shlex.quote(str(remote)) + "; set -e; "
        "if test -e " + checkpoint_arg + "; then test -f " + checkpoint_arg + "; "
        "printf '%s  %s\\n' " + expected + " " + checkpoint_arg +
        " | sha256sum -c --status -; else test -f " + source_arg + "; "
        "printf '%s  %s\\n' " + expected + " " + source_arg +
        " | sha256sum -c --status -; mkdir -p " + parent_arg + "; "
        "cp --reflink=auto " + source_arg + " " + pending_arg + "; "
        "printf '%s  %s\\n' " + expected + " " + pending_arg +
        " | sha256sum -c --status -; mv " + pending_arg + " " + checkpoint_arg + "; fi")


def main():
    if RESULT.exists():
        raise RuntimeError(f"preserve existing controller receipt {RESULT}")
    LOG.parent.mkdir(parents=True, exist_ok=True)
    if LOG.exists():
        raise RuntimeError(f"preserve existing cloud log {LOG}")
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=REPO, text=True)
    if dirty:
        raise RuntimeError("reserved implementation checkout is not clean: " +
                           repr(dirty.splitlines()[:10]))
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    run(["git", "fetch", "origin", BRANCH], cwd=REPO, timeout=300)
    upstream = subprocess.check_output(
        ["git", "rev-parse", f"origin/{BRANCH}"], cwd=REPO, text=True).strip()
    if head != upstream:
        raise RuntimeError(f"reserved implementation HEAD must equal pushed origin/{BRANCH}")
    six = json.loads((REPO / "results" / "m10_final_run.json").read_text())
    if six.get("end_status") != "INCOMPLETE_RESERVED" or not six.get(
            "decision_record", {}).get("reserved_batch_runs"):
        raise RuntimeError("durable six-set result does not require the reserved batch")
    build_config = json.loads(BUILD_CONFIG.read_text())
    build_record = json.loads(BUILD_RECORD.read_text())
    eval_manifest = json.loads(EVAL_MANIFEST.read_text())
    vector_bytes = expected_vector_bytes(eval_manifest)
    remote_free_required = (MIN_POST_PREENCODE_FREE_BYTES + vector_bytes +
                            PREENCODE_OPERATIONAL_HEADROOM_BYTES)
    freeze = json.loads((REPO / "m10" / "FREEZE.json").read_text())
    checkpoint_rel = Path(str(freeze.get("checkpoint", "")))
    if (checkpoint_rel.is_absolute() or ".." in checkpoint_rel.parts or
            not checkpoint_rel.parts or checkpoint_rel.parts[0] != "work"):
        raise RuntimeError("frozen checkpoint path is not a bounded work/ path")
    checkpoint = REPO / checkpoint_rel
    if not checkpoint.is_file() or sha(checkpoint) != freeze.get("checkpoint_sha256"):
        raise RuntimeError("local frozen checkpoint is missing or changed")
    build_checkpoint_rel = Path(str(build_record.get("final_checkpoint", "")))
    if (build_checkpoint_rel.is_absolute() or ".." in build_checkpoint_rel.parts or
            not build_checkpoint_rel.parts or build_checkpoint_rel.parts[0] != "work" or
            build_record.get("final_checkpoint_sha256") != freeze.get("checkpoint_sha256")):
        raise RuntimeError("build record cannot authenticate the frozen checkpoint source")
    registered_hours = build_config["budget"]["mandatory_hours_fixed"].get(
        "reserved_batch_allowance")
    allocation = build_record.get("budget", {}).get("allocation", {})
    if (registered_hours != INHERITED_RESERVED_ALLOWANCE or
            allocation.get("mandatory_hours", {}).get("reserved_batch_allowance")
            != INHERITED_RESERVED_ALLOWANCE or
            allocation.get("price_usd_per_h") != MAX_TOTAL_HOURLY or
            allocation.get("budget_ceiling_usd") != 1000.0):
        raise RuntimeError("reserved controller differs from the registered M13 allocation")
    budget = m20_budget()
    cap_a, cap_b = stage_caps()
    max_hours = cap_a + cap_b
    if cap_b != INHERITED_RESERVED_ALLOWANCE:
        raise RuntimeError("stage B must carry the inherited 55.2-hour reserved allowance")
    if budget["project_ceiling_usd"] != allocation.get("budget_ceiling_usd") \
            or budget["committed_usd_before_m20"] != allocation.get("committed_usd"):
        raise RuntimeError("M20 registered budget disagrees with the M13 allocation record")
    # The registered DBSF reproduction is a run prerequisite: the extended roster code must have
    # been shown to compute M12's operator before it computes anything new.
    if not DBSF_RECEIPT.is_file() or json.loads(DBSF_RECEIPT.read_text()).get("status") != "PASSED":
        raise RuntimeError("results/m20_dbsf_reproduction.json is missing or did not pass")
    if subprocess.run(["git", "ls-remote", "--exit-code", "origin", "refs/tags/m8-reserved-spent"],
                      cwd=REPO, capture_output=True).returncode == 0:
        raise RuntimeError("reserved spent tag already exists; use the bounded continuation manually")

    cloud = Cloud()
    pods = [cloud.get(POD), *(cloud.get(pod) for pod in OTHER_PODS)]
    if any(pod.get("desiredStatus") != "EXITED" for pod in pods):
        raise RuntimeError("all retained M13 pods must be stopped before reserved launch")
    target = pods[0]
    if (target.get("volumeInGb") != 500 or target.get("containerDiskInGb") != 30 or
            float(target.get("costPerHr", math.inf)) > 1.59):
        raise RuntimeError("retained A100 storage or quote changed")
    target_total_hourly = float(target["costPerHr"]) + storage_hourly(target)
    if target_total_hourly > MAX_TOTAL_HOURLY + 1e-9:
        raise RuntimeError("target compute plus its storage exceeds the registered hourly rate")
    other_storage_hourly = sum(storage_hourly(pod) for pod in pods[1:])
    all_retained_hourly = target_total_hourly + other_storage_hourly
    extra_retained_storage_usd = other_storage_hourly * max_hours
    # Price the WHOLE registered M20 plan against the ceiling, not just this controller's share:
    # stages C and D follow on the same wallet and the same budget line.
    if (float(allocation.get("committed_usd", math.inf)) + float(budget["new_spend_at_cap_usd"]) >
            float(allocation.get("budget_ceiling_usd", -math.inf))):
        raise RuntimeError("the registered M20 stage caps exceed the project budget ceiling")
    balance_start = cloud.balance()
    if balance_start < max_hours * all_retained_hourly:
        raise RuntimeError("current wallet cannot cover this controller's conservative allowance "
                           f"of {max_hours} h at ${all_retained_hourly:.4f}/h")

    record = {"status": "RUNNING", "stage": "resuming", "pod_id": POD,
              "implementation_commit": head, "started_utc": datetime.now(timezone.utc).isoformat(),
              "balance_start_usd": balance_start, "max_hours": max_hours,
              "stage_cap_hours": {"A_pre_encode": cap_a, "B_tagged_transaction": cap_b},
              "m20_registry_sha256": sha(M20_REGISTRY),
              "dbsf_reproduction_sha256": sha(DBSF_RECEIPT),
              "max_total_hourly_usd": MAX_TOTAL_HOURLY,
              "target_total_hourly_usd": target_total_hourly,
              "other_retained_storage_hourly_usd": other_storage_hourly,
              "all_retained_hourly_usd": all_retained_hourly,
              "max_all_retained_interval_usd": max_hours * all_retained_hourly,
              "extra_retained_storage_interval_usd": extra_retained_storage_usd,
              "expected_reserved_vector_bytes": vector_bytes,
              "remote_free_required_before_preencode_bytes": remote_free_required,
              "build_config_sha256": sha(BUILD_CONFIG),
              "build_record_sha256": sha(BUILD_RECORD),
              "checkpoint": str(checkpoint_rel),
              "checkpoint_sha256": freeze["checkpoint_sha256"],
              "six_result_sha256": sha(REPO / "results" / "m10_final_run.json")}
    save(record)
    started = False
    remote_pid = None

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
            actual = float(pod.get("costPerHr", math.inf)) + storage_hourly(pod)
            actual_all_retained = actual + other_storage_hourly
            if actual > MAX_TOTAL_HOURLY + 1e-9:
                raise RuntimeError(f"live target hourly cost {actual} exceeds allocation")
            if (float(allocation["committed_usd"]) +
                    max(0.0, actual_all_retained - MAX_TOTAL_HOURLY) * max_hours >
                    float(allocation["budget_ceiling_usd"])):
                raise RuntimeError("live all-retained hourly cost exceeds project budget")
            if pod.get("desiredStatus") == "RUNNING" and pod.get("gpuCount") == 1 \
                    and update_ssh(pod):
                record["actual_target_total_hourly_usd"] = actual
                record["actual_all_retained_hourly_usd"] = actual_all_retained
                break
            time.sleep(5)
        else:
            raise RuntimeError("SSH readiness timeout")

        record["stage"] = "deploying"
        save(record)
        status = ssh("cd " + shlex.quote(REMOTE) +
                     "; git status --porcelain --untracked-files=all", capture_output=True)
        lines = [line for line in status.stdout.splitlines() if line]
        if lines:
            raise RuntimeError(f"remote checkout is not clean: {lines}")
        ssh("cd " + shlex.quote(REMOTE) + "; set -e; git fetch origin " + BRANCH +
            "; git checkout -B " + BRANCH + " origin/" + BRANCH +
            "; git branch --set-upstream-to=origin/" + BRANCH + " " + BRANCH,
            timeout=300)
        # The cloud build retains its artifact at the build-record path.  Later freeze bookkeeping
        # names an identical copy under work/m13-final.  Materialize that copy before any query
        # model or protected payload is touched, and never replace an existing mismatched target.
        ssh(checkpoint_prep_command(checkpoint_rel, build_checkpoint_rel,
                                    freeze["checkpoint_sha256"]), timeout=600)
        checkpoint_arg = shlex.quote(str(checkpoint_rel))
        remote_identity = ssh(
            "cd " + shlex.quote(REMOTE) + "; git rev-parse HEAD; test -f " + checkpoint_arg +
            "; sha256sum " + checkpoint_arg + " | cut -d' ' -f1",
            timeout=300, capture_output=True).stdout.splitlines()
        if remote_identity != [head, freeze["checkpoint_sha256"]]:
            raise RuntimeError("remote implementation/checkpoint identity differs: " +
                               repr(remote_identity))
        remote_free = int(ssh(
            "python3 -c " + shlex.quote(
                "import shutil; print(shutil.disk_usage('/home/dylan').free)"),
            capture_output=True).stdout.strip())
        record["remote_free_before_preencode_bytes"] = remote_free
        save(record)
        if remote_free < remote_free_required:
            raise RuntimeError(f"remote has {remote_free} free bytes; pre-encode requires "
                               f"{remote_free_required} to preserve the 120 GB scoring floor")

        remote_log = "/tmp/m13-reserved.log"
        remote_exit = "/tmp/m13-reserved.exit"
        inner = "( " + remote_command() + " ); rc=$?; echo $rc > " + remote_exit
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
                    raise RuntimeError(f"remote reserved runner exited {code}: {tail}")
                break
            if state == "LOST":
                raise RuntimeError("remote reserved process disappeared without an exit receipt")
            time.sleep(30)
        else:
            raise TimeoutError(f"reserved run exceeded its conservative {max_hours}-hour "
                               f"allowance (stage A {cap_a} h + stage B {cap_b} h)")

        run(["scp", "-F", str(SSH_CONFIG), SSH_ALIAS + ":" + remote_log, str(LOG)], timeout=180)
        run(["git", "fetch", "origin", BRANCH], cwd=REPO, timeout=300)
        run(["git", "merge", "--ff-only", f"origin/{BRANCH}"], cwd=REPO, timeout=300)
        reserved = REPO / "results" / "m13_reserved_run.json"
        final = json.loads((REPO / "results" / "m10_final_run.json").read_text())
        if not reserved.exists() or final.get("end_status") != "COMPLETE":
            raise RuntimeError("remote runner exited zero without durable complete reserved results")
        record.update(status="PASSED", stage="finished", reserved_result_sha256=sha(reserved),
                      final_result_sha256=sha(REPO / "results" / "m10_final_run.json"))
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
        raise RuntimeError("reserved work finished but the paid pod did not confirm STOP")
    run(["git", "add", str(RESULT)], cwd=REPO)
    run(["git", "commit", "-m", "ops(m13): record reserved cloud execution"], cwd=REPO)
    run(["git", "push", "origin", f"HEAD:{BRANCH}"], cwd=REPO)
    print(json.dumps({key: record.get(key) for key in
                      ("status", "pod_final_status", "wallet_delta_usd",
                       "reserved_result_sha256")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    main()
