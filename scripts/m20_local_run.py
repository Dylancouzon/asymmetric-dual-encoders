#!/usr/bin/env python3
"""Run M20's stages A, B and C on the local GPU, under ruling R24.

`scripts/m13_reserved_cloud.py` is the cloud controller. Most of what it does is cloud plumbing --
resume a pod, price it, cover it from the wallet, deploy a checkout, STOP it -- but a handful of
its checks are real readiness gates that have nothing to do with a pod, and invoking the two stage
commands by hand would silently drop them. This launcher keeps exactly those, and nothing else.

Kept from the controller (Astra's condition 3, 2026-09-17):

  * clean working tree at the exact pushed `origin/main`;
  * the six-set prerequisite: `INCOMPLETE_RESERVED` and the reserved batch actually triggered;
  * `results/m20_dbsf_reproduction.json` must be `PASSED` -- the transaction preflight does not
    enforce this, and it is the evidence that the extended roster computes M12's operator;
  * compile, the reserved-support tests, the roster identity assertion and the BM25 version gate,
    all BEFORE anything is encoded, because a roster mismatch caught at scoring time is caught
    after the tag;
  * the corpus-loader preflight online, and again offline after stage A, because completed vector
    shards do not prove BM25 can load its corpus through the offline path;
  * the query-tower preflight, which resolves the pinned Zero snapshot and validates the Nano
    dependency while the access is still unspent;
  * a disk floor before the pre-encode that preserves the registered 120 GB scoring floor;
  * one shared absolute stage-A deadline across all three tower processes, and a separate hard
    timeout per stage;
  * explicit cache, device, thread and offline settings for the scorer;
  * a durable local execution receipt.

Dropped, because there is no pod and no bill: pod identity, storage and price checks, wallet
cover, cloud allocation checks, resume and SSH handling, remote transfer, STOP confirmation.
Retained storage keeps billing and the retained pods stay STOP-only; neither is this script's job.

Stage C is the same shape for BEIR-15, and is here for the same reason: `m20src/beir15.py` is a
scorer. Its document-vector path loads existing shards and fails on a missing one, it never
schedules an encode, and it carries no deadline, timeout or projection gate, so running it alone
would neither do the ~23.7M documents of missing corpus work nor enforce the registered single
120 h cap (Astra review, 2026-09-19). Stage C pre-encodes BEIR-15's own corpora on all three
towers and then scores, under one retained deadline, with the projection gate armed against the
registered volume. It reuses the reserved four's shards rather than re-encoding them, and it
inverts two preflight gates rather than skipping them: by then the access must be SPENT and the
six-set result must read COMPLETE.

  .venv/bin/python -u scripts/m20_local_run.py             # preflight only, changes nothing
  .venv/bin/python -u scripts/m20_local_run.py --execute   # stage A then stage B
  .venv/bin/python -u scripts/m20_local_run.py --stage-c   # stage C preflight only
  .venv/bin/python -u scripts/m20_local_run.py --stage-c --execute   # stage C
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
PY = str(REPO / ".venv" / "bin" / "python")
RECEIPT = REPO / "results" / "m20_local_run.json"
STAGE_C_RECEIPT = REPO / "results" / "m20_stage_c_run.json"
LOG_DIR = REPO / "work" / "m20" / "logs"
REGISTRY = REPO / "m20" / "beir15_registry.json"
DBSF_RECEIPT = REPO / "results" / "m20_dbsf_reproduction.json"
SIX_RESULT = REPO / "results" / "m10_final_run.json"
BRANCH = "main"
TOWERS = ("nano-dense", "bge-small-en-v1.5", "leaf-ir-asym")
# The controller's floor: the registered 120 GB scoring floor plus the reserved vectors plus
# operational headroom. 10,115,709 documents x (1024 + 384 + 768) x 2 bytes = 44.02 GB.
MIN_FREE_BYTES = 120_000_000_000 + 44_023_565_568 + 2_000_000_000

ENV = {
    "HF_HOME": os.path.expanduser("~/.cache/huggingface"),
    "HF_HUB_CACHE": os.path.expanduser("~/.cache/huggingface/hub"),
    "HF_DATASETS_CACHE": os.path.expanduser("~/.cache/huggingface/datasets"),
    "XDG_CACHE_HOME": os.path.expanduser("~/.cache"),
    "M7_DEVICE": "cuda", "M7_ENCODER": "stella-400M-v5",
    "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4",
    "TOKENIZERS_PARALLELISM": "false",
    # A 50,000-document `teacher.encode` call is length-sorted, so one call sweeps batch shapes
    # from many short documents to 64x512 and the caching allocator accumulates segments it cannot
    # reuse: 23.06 GiB reserved on this 10.24 GiB card for 1.76 GiB of live tensors. Past the
    # device ceiling WSL's driver falls back to host memory over PCIe rather than raising OOM, so
    # `num_alloc_retries` stays 0 and throughput collapses silently -- 74 then 25 docs/s, which is
    # what tripped the projection gate on 2026-09-17. Expandable segments bound the reservation to
    # 4.40 GiB and restore 179 docs/s. It changes fragmentation handling only: not batch
    # composition, not fp32/TF32-disabled, not max_length, not the stored dtype. Verified by
    # re-encoding FEVER 0:50000 and reproducing the already-written shard byte for byte.
    # Receipt: results/m20_allocator_probe.json.
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}


def sh(*command, timeout=600, env=None, check=True):
    result = subprocess.run(command, cwd=REPO, text=True, capture_output=True, timeout=timeout,
                            env={**os.environ, **ENV, **(env or {})})
    if check and result.returncode:
        raise RuntimeError(f"{' '.join(command[:4])}… exited {result.returncode}: "
                           f"{(result.stderr or result.stdout).strip()[-800:]}")
    return result


def stage_caps():
    caps = json.loads(REGISTRY.read_text())["budget"]["stage_caps_hours"]
    return caps["A_reserved_pre_encode_unprotected"], caps["B_tagged_reserved_transaction"]


def stage_c_cap():
    return json.loads(REGISTRY.read_text())["budget"]["stage_caps_hours"]["C_beir15"]


def stage_c_scorer_reserve_hours():
    """Hours of the stage-C cap held back for scoring, derived from the registration's own figures.

    The projection gate only ever projects encoding: its own tower's remaining documents plus the
    towers after it. Nothing reserved time for the scorer that runs afterwards under the same
    deadline, so encoding could be allowed to consume essentially the whole cap and the stage would
    discover it had no time to score only after five days of encoding (Sol re-review, 2026-09-19,
    P1). This is not a new constant: the registration expects stage C to take 113 h of its 120 h
    cap, and its own registered volume and A100 rate imply the encode share, so the remainder is
    what it budgeted for download and scoring.
    """
    budget = json.loads(REGISTRY.read_text())["budget"]
    docs = budget["document_volumes"]["beir15_new_docs_expected"]
    # `all_three` is the registered combined ratio and sits in the same dict as the three towers;
    # summing the dict double-counts it.
    ratio = budget["tower_cost_relative_to_stella"]["all_three"]
    encode_hours = docs * ratio / budget["a100_stella_fp32_passages_per_second"] / 3600
    return max(1.0, budget["expected_hours"]["C_beir15"] - encode_hours)


def stage_c_disk_floor():
    """The registered 120 GB scoring floor plus BEIR-15's own fp16 vectors, plus headroom."""
    docs = json.loads(REGISTRY.read_text())["budget"]["document_volumes"][
        "beir15_new_docs_expected"]
    return 120_000_000_000 + docs * (1024 + 384 + 768) * 2 + 2_000_000_000


def preflight(stage="AB"):
    """Every gate that must hold before a stage runs. Opens no protected payload.

    Stage C inverts two of stage A/B's gates rather than skipping them: by then the access must be
    SPENT and the six-set result must read COMPLETE, because stage C runs after the transaction,
    not before it. Everything else -- clean pushed tree, GPU, compile, tests, roster and BM25
    identity, corpus route -- applies to both, and stage C checks the corpus route for its own
    batch and holds a disk floor sized for its own vectors.
    """
    problems, facts = [], {}

    dirty = sh("git", "status", "--porcelain=v1", "--untracked-files=all").stdout
    if dirty.strip():
        problems.append(f"working tree is not clean: {dirty.splitlines()[:5]}")
    sh("git", "fetch", "origin", BRANCH, timeout=300)
    head = sh("git", "rev-parse", "HEAD").stdout.strip()
    upstream = sh("git", "rev-parse", f"origin/{BRANCH}").stdout.strip()
    facts["head"] = head
    if head != upstream:
        problems.append(f"HEAD {head[:12]} is not the pushed origin/{BRANCH} {upstream[:12]}")

    six = json.loads(SIX_RESULT.read_text())
    facts["six_end_status"] = six.get("end_status")
    want_six = "COMPLETE" if stage == "C" else "INCOMPLETE_RESERVED"
    if six.get("end_status") != want_six:
        problems.append(f"six-set result ends {six.get('end_status')!r}, not {want_six}")
    if stage == "C" and "reserved" not in six:
        problems.append("the six-set result carries no reserved report; stage B has not completed")
    if not (six.get("decision_record") or {}).get("reserved_batch_runs"):
        problems.append("the frozen six-set decision did not trigger the reserved batch")

    if not DBSF_RECEIPT.is_file() or json.loads(DBSF_RECEIPT.read_text()).get("status") != "PASSED":
        problems.append("results/m20_dbsf_reproduction.json is missing or did not pass")

    spent = subprocess.run(["git", "ls-remote", "--exit-code", "origin",
                            "refs/tags/m8-reserved-spent"], cwd=REPO, capture_output=True)
    facts["reserved_access_unspent"] = bool(spent.returncode)
    if stage == "C":
        if spent.returncode != 0:
            problems.append("m8-reserved-spent is not on origin; stage C runs after the "
                            "transaction, not instead of it")
    elif spent.returncode == 0:
        problems.append("m8-reserved-spent already exists on origin; the access is spent")

    floor = stage_c_disk_floor() if stage == "C" else MIN_FREE_BYTES
    free = shutil.disk_usage(REPO).free
    facts["free_bytes"] = free
    facts["disk_floor_bytes"] = floor
    if free < floor:
        problems.append(f"{free} free bytes; this stage's vectors plus the registered 120 GB "
                        f"scoring floor need {floor}")

    try:
        gpu = json.loads(sh(PY, "-c",
                            "import json, torch; "
                            "print(json.dumps({'available': torch.cuda.is_available(), "
                            "'name': torch.cuda.get_device_name(0) "
                            "if torch.cuda.is_available() else None, "
                            "'vram': torch.cuda.get_device_properties(0).total_memory "
                            "if torch.cuda.is_available() else 0}))").stdout)
        facts["gpu"] = gpu
        if not gpu["available"]:
            problems.append("no CUDA device: a CUDA torch installation is not a GPU")
    except Exception as error:
        problems.append(f"GPU check failed: {type(error).__name__}: {error}")

    sh(PY, "-m", "py_compile", "m8src/pre_encode.py", "m13src/reserved_support.py",
       "m13src/reserved_transaction.py", "m13src/score13.py", "m20src/roster.py",
       "m20src/beir15.py")
    tests = sh(PY, "-m", "pytest", "-q", "m13src/test_reserved_support.py",
               env={"PYTHONPATH": "m13src:m20src"}, timeout=900, check=False)
    facts["reserved_support_tests"] = tests.stdout.strip().splitlines()[-1] if tests.stdout else ""
    if tests.returncode:
        problems.append(f"reserved-support tests failed: {facts['reserved_support_tests']}")

    identity = sh(PY, "-c", "import roster; print(roster.assert_registered_identities()); "
                            "print(roster.assert_registered_bm25_versions())",
                  env={"PYTHONPATH": "m20src"}, check=False)
    facts["roster_identity"] = identity.stdout.strip().replace("\n", " ")
    if identity.returncode:
        problems.append(f"roster or BM25 version gate failed: "
                        f"{(identity.stderr or identity.stdout).strip()[-400:]}")

    corpora = sh(PY, "m8src/pre_encode.py", "--preflight-only",
                 "--batch", "beir15" if stage == "C" else "reserved", timeout=7200, check=False)
    if corpora.returncode:
        problems.append(f"corpus preflight failed: "
                        f"{(corpora.stderr or corpora.stdout).strip()[-400:]}")
    else:
        facts["corpus_preflight"] = json.loads(corpora.stdout.strip().splitlines()[-1])["status"]
    return problems, facts


def run_stage(label, command, cap_hours, env=None, log=None):
    log = log or (LOG_DIR / f"{label}.log")
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    print(f"[m20local] {label}: starting, cap {cap_hours} h, log {log}", flush=True)
    with log.open("ab") as stream:
        result = subprocess.run(command, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT,
                                timeout=cap_hours * 3600,
                                env={**os.environ, **ENV, **(env or {})})
    elapsed = (time.time() - started) / 3600
    if result.returncode:
        raise RuntimeError(f"{label} exited {result.returncode} after {elapsed:.1f} h; see {log}")
    print(f"[m20local] {label}: complete in {elapsed:.1f} h", flush=True)
    return elapsed


def run_stage_c(cap_c, record):
    """Stage C: encode BEIR-15's own corpora, then score, under ONE retained deadline.

    `m20src/beir15.py` is a scorer: its document-vector path loads existing shards and fails on a
    missing one, it never schedules an encode, and it holds no deadline, timeout or projection
    gate. Running it alone would neither do the ~23.7M documents of missing corpus work nor enforce
    the registered single 120 h cap (Astra review, 2026-09-19, P1). This does both, with the same
    primitives stage A used: one absolute deadline handed to every tower so a slow one shortens the
    next rather than each getting the full cap, the pre-encode projection gate armed against the
    registered BEIR-15 volume, and the scorer bounded by whatever time is left.
    """
    deadline = record["stage_c_deadline_epoch"]
    # The scorer runs after the towers under the same clock, and the projection gate only ever
    # projects encoding. Hand the towers a deadline short of the real one by the registered
    # download-and-score allowance, so encoding cannot eat the time scoring needs.
    reserve = stage_c_scorer_reserve_hours()
    encode_deadline = deadline - reserve * 3600
    record["stage_c_scorer_reserve_hours"] = reserve

    encode_hours = 0.0
    for tower in TOWERS:
        encode_hours += run_stage(
            f"stageC-preencode-{tower}",
            [PY, "-u", "m8src/pre_encode.py", "--batch", "beir15", "--system", tower,
             "--device", "cuda", "--stage-deadline", f"{encode_deadline:.0f}",
             "--tower-order", ",".join(TOWERS)],
            cap_hours=_remaining_hours(encode_deadline, "stage-C pre-encode"))
    record["stage_c_preencode_hours"] = encode_hours
    record["stage_c_score_hours"] = run_stage(
        "stageC-score", [PY, "-u", "m20src/beir15.py", "--device", "cuda"],
        cap_hours=_remaining_hours(deadline, "stage-C scoring"))
    return record


def _remaining_hours(deadline, what):
    """Time left before an absolute deadline, or a refusal -- never a fresh floor of minutes.

    `max(0.05, remaining)` handed every phase three more minutes after the deadline had passed, so
    a sequence of phases could carry execution beyond the registered cap (Sol re-review,
    2026-09-19, P1). Past the deadline the answer is to stop and let a human decide, not to grant
    another slice.
    """
    remaining = (deadline - time.time()) / 3600
    if remaining <= 0:
        raise RuntimeError(f"{what}: the registered stage-C deadline has passed; stopping rather "
                           f"than granting more time. Everything encoded so far is hash-recorded "
                           f"and resumable.")
    return remaining


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true",
                        help="run stages A and B. Without it this is preflight only and changes "
                             "nothing.")
    parser.add_argument("--stage-c", action="store_true",
                        help="run stage C, BEIR-15: pre-encode its own corpora on all three "
                             "towers, then score, under one retained deadline. Requires stage B "
                             "complete. Without --execute this is preflight only.")
    args = parser.parse_args(argv)

    if args.stage_c:
        cap_c = stage_c_cap()
        # The clock starts BEFORE the preflight, not after it. The registered stage explicitly
        # includes download, and the stage-C preflight loads every BEIR-15 corpus -- up to hours of
        # it on a cold cache. Starting the deadline afterwards would leave that work outside the
        # cap it belongs to (Sol re-review, 2026-09-19, P1).
        started = time.time()
        problems, facts = preflight(stage="C")
        record = {"status": "REFUSED" if problems else "PASSED", "mode": "local", "stage": "C",
                  "authority": "m13/RULINGS.md R24; m20/REGISTRATION.md stage C",
                  "stage_caps_hours": {"C": cap_c},
                  "started_utc": datetime.now(timezone.utc).isoformat(),
                  "problems": problems, "facts": facts}
        if problems or not args.execute:
            print(json.dumps(record, indent=2))
            return 2 if problems else 0
        if STAGE_C_RECEIPT.exists():
            raise RuntimeError(f"preserve the existing stage-C receipt {STAGE_C_RECEIPT}")
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        # Fix the deadline and make it DURABLE before five days of work begins. Held only in
        # memory it would survive an exception but not a kill, a host failure or a power cut, and a
        # relaunch would then mint a fresh 120 h instead of inheriting this one (Sol re-review,
        # 2026-09-19, P1). The earliest preserved attempt receipt binds, exactly as stage A's does.
        deadline = started + cap_c * 3600
        for prior in sorted(RECEIPT.parent.glob("m20_stage_c_run_attempt*.json")):
            earlier = json.loads(prior.read_text()).get("stage_c_deadline_epoch")
            if earlier:
                deadline = min(deadline, float(earlier))
        record["stage_c_deadline_epoch"] = deadline
        record["stage_c_deadline_inherited"] = deadline < started + cap_c * 3600 - 1
        record["status"] = "RUNNING"
        STAGE_C_RECEIPT.write_text(json.dumps(record, indent=2) + "\n")
        try:
            record = run_stage_c(cap_c, record)
            record["status"] = "COMPLETE"
        except BaseException as error:
            record["status"] = "FAILED"
            record["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            record["finished_utc"] = datetime.now(timezone.utc).isoformat()
            STAGE_C_RECEIPT.write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps({"status": record["status"],
                          "hours": {k: v for k, v in record.items() if k.endswith("_hours")}},
                         indent=2))
        return 0

    cap_a, cap_b = stage_caps()
    problems, facts = preflight()
    record = {"status": "REFUSED" if problems else "PASSED", "mode": "local",
              "authority": "m13/RULINGS.md R24", "stage_caps_hours": {"A": cap_a, "B": cap_b},
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "problems": problems, "facts": facts}
    if problems or not args.execute:
        print(json.dumps(record, indent=2))
        RECEIPT.write_text(json.dumps(record, indent=2) + "\n") if args.execute else None
        return 2 if problems else 0

    if RECEIPT.exists():
        raise RuntimeError(f"preserve the existing execution receipt {RECEIPT}")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    # The registration defines ONE wall clock for stage A, not one per attempt. Minting a fresh
    # `now + cap` on every execution would let repeated relaunches accumulate past the registered
    # 52 h while each process stayed inside its own newly issued deadline -- silently assuming an
    # owner decision the plan reserves (Sol review, 2026-09-17, P1). Every preserved attempt
    # receipt carries the deadline it ran under; the earliest one binds.
    deadline = time.time() + cap_a * 3600
    for prior in sorted(RECEIPT.parent.glob("m20_local_run_attempt*.json")):
        earlier = json.loads(prior.read_text()).get("stage_a_deadline_epoch")
        if earlier:
            deadline = min(deadline, float(earlier))
    record["stage_a_deadline_epoch"] = deadline
    record["stage_a_deadline_inherited"] = deadline < time.time() + cap_a * 3600 - 1
    try:
        elapsed_a = 0.0
        for tower in TOWERS:
            elapsed_a += run_stage(
                f"stageA-{tower}",
                [PY, "-u", "m8src/pre_encode.py", "--system", tower, "--device", "cuda",
                 "--stage-deadline", f"{deadline:.0f}", "--tower-order", ",".join(TOWERS)],
                cap_hours=max(0.05, (deadline - time.time()) / 3600))
        record["stage_a_hours"] = elapsed_a

        # Completed shards do not prove BM25 can load its corpus offline, and BM25 runs INSIDE the
        # transaction. Prove the offline route while the access is still unspent.
        run_stage("stageA-offline-corpus-preflight",
                  [PY, "m8src/pre_encode.py", "--preflight-only"], cap_hours=1.0,
                  env={"HF_DATASETS_OFFLINE": "1"})
        run_stage("stageB-query-tower-preflight",
                  [PY, "-m", "reserved_support", "--preflight-models", "--device", "cuda"],
                  cap_hours=1.0, env={"PYTHONPATH": "m13src:m20src"})

        record["stage_b_hours"] = run_stage(
            "stageB-transaction", [PY, "-u", "m13src/score13.py", "--reserved-only"],
            cap_hours=cap_b,
            env={"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"})
        record["status"] = "COMPLETE"
    except BaseException as error:
        record.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        record["finished_utc"] = datetime.now(timezone.utc).isoformat()
        RECEIPT.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"status": record["status"], "stage_a_hours": record.get("stage_a_hours"),
                      "stage_b_hours": record.get("stage_b_hours")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
