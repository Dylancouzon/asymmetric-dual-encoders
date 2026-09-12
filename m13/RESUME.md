# Active build on original E host — 2026-09-12

Build supervisor PID43900 successfully adopted k3aee2m68765em (one A100-SXM4-80GB).
Execution commit8ed37ed. Sourcewnzk8eeqrrkw4m STOP confirmed. Migration and recovery
handoff PASSED. Runtime bootstrap passed and final394-file rehash passed in57seconds.
Current stage: full-build-preflight, then automatic fixed200M training. No training
progress observed at this update. Check results/m13_cloud_build_migrated.json and
logs/m13-build-migrated-controller.log plus logs/m13-build-migrated.log. Monitor user
service is active. Hourly rolling backups and final backup/STOP belong to PID43900.
Do not launch another supervisor, alter bound sources, or move the running remote HEAD.

Published cumulative allocation:135.29694h/$225.0815 remaining at admission, before
runtime/preflight deductions. Original144h/$239.56 stage allowance and$1000 ceiling
remain unchanged. Fresh pretraining reconciliation is mandatory and automatic.

Historical state follows.

# Verified migration — fresh build handoff recovery

All 394 destination files passed hashing; source STOP confirmed. Original E target
k3aee2m68765em retains its A100 under migration PID38227's bounded handoff lease.
The first coordinator PID38228 failed before dispatch because temporary extension
preparation changed its bound source. Those edits were restored; no extension or
migration takeover ran. The original migration completed within its reviewed cap.

Fresh scripts/m13_after_migration_recovery.py changes only the coordinator's receipt
and self-binding paths. It uses results/m13_after_migration_recovery.json, PID file
work/m13cloud-launchers/after_migration_recovery.pid and its matching log. Read actual
receipts before action. The allocator and build supervisor remain the reviewed versions.

Historical state follows.

# Active migration — source zero GPU, original E host A100 reserved

MigrationPID38227 and automatic build handoffPID38228 are active from pushed8b67154.
Both Pods resumed successfully; exactA100-SXM4-80GB/81920MiB confirmed on target.
Direct transfer is running around19MB/s, reusing nearly all shared root caches.
Inspect results/m13_migration.json and logs/m13-migrate-inputs.log before action.

User explicitly authorized moving to the available retained original E host and starting
training. Source `wnzk8eeqrrkw4m` has the verified394 inputs; target `k3aee2m68765em` is the
same A100-SXM4-80GB model at quoted$1.59/h, persistent500GB, container30GB. No new Pod.
Old capacity coordinatorPID28201 was terminated while waiting_gpu, before any build dispatch;
its final superseded receipt is preserved at results/m13_after_verification.json.

New direct cloud migration uses an ephemeral read key, exact manifest paths, destination
hashes, sourceSTOP, and a bounded target-GPU lease through adoption by the build supervisor.
No inputs pass back through the home upload connection. Prior E outputs and runtime are
preserved before deployment. The unchanged200M recipe, preflight, finalization and backups
remain mandatory. New allocation must include both Pods' concurrent charges and all prior
elapsed intervals; original144h/$239.56 stage and$1000 ceiling remain unchanged.

Operational paths: scripts/m13_migrate_inputs.py, scripts/m13_migration_allocation.py,
scripts/m13_after_migration.py, scripts/m13_build_migrated.py. Corresponding receipts:
results/m13_migration.json (mutable), results/m13_migration_ready.json (immutable),
results/m13_after_migration.json, results/m13_cloud_build_migrated.json. Check them and
PID files before action; never race a live migration or build. Independent reviews and
checks must clear the concrete code before paid resume. Historical state follows.

# Current state: upload verified, waiting for GPU — 2026-09-12

Verification PASSED: all394 files,72,623,148,280bytes,259.6seconds remote hashing.
`results/m13_storage_verification.json` is published; retained Pod STOP confirmed.
The active coordinatorPID28201 (`results/m13_after_verification.json`) restored the
stopped container to30GB, applied/published the reviewed supervisor in commit a15cf1e,
and is polling GPU availability read-only. Last observed count0. No training has started.
Do not race or rerun the coordinator. It has a12-hour capacity wait and a fresh cumulative
budget gate before its single GPU dispatch. Independent monitor service remains active.

The recovery history below preserves prior failures and their evidence.

# Verification timeout recovery — 2026-09-12

The full rsync transfer completed, but silent destination hashing exceeded1800s. Original
storage receipt FAILED with verified STOP; preserved at `results/m13_storage_upload.json`
(SHA256413c7856c1d6c53473c425fdf51b4f373bbc346e4c70d09c7a312310ce4bfcdc).
The old handoff also failed safely; `results/m13_after_upload.json` is preserved. No training.

Verification-only recovery: `scripts/m13_verify_uploaded.py`, new receipt
`results/m13_storage_verification.json`, log `logs/m13-storage-verification-retry.log`, PID file
`work/m13cloud-launchers/storage_verification_retry.pid`. It reuses the394 uploaded files, streams
hashes with progress, and caps paid zero-GPU work at3h while preserving training+4h and the
original cumulative allowance. Initial prestart refused provider-reported5GB container;
refusal preserved. Verification-only permits5or30GB; retryPID27527 launched fromfd84d45.
The coordinator restores only containerDiskInGb30 while all Pods are stopped, verifies
persistent volume identity, and STOPs on an uncertain update before any GPU handoff.
Monitor job covers the durable progress log. STOP only.
The downstream allocation now includes both failed storage and successful verification.
The updated coordinator is active asPID28201 from pushed01ba410 and uses separate receipt `results/m13_after_verification.json`, log
`logs/m13-after-verification.log`, PID `work/m13cloud-launchers/after_verification.pid`.
It pins the previous failed coordinator and waits for verification PASS/STOP before applying
the reviewed supervisor patch (SHA256846e93b71fca78f6b4174b9dfc0e667175db11cb0bc453046e50b8d5d49c67d7).
GPU-side verification also streams progress with a90-minute command limit, still inside the
original stage alarm and fresh pretraining budget gate. No duplicate retry or new allowance.

Earlier overnight state below is historical and superseded by this timeout recovery.

# Current overnight continuation — 2026-09-12

The storage upload is active on the SAME retained Pod `wnzk8eeqrrkw4m`, through the
provider's dedicated `podResumeZeroGpu` operation. Supervisor PID11028:
`scripts/m13_upload_only.py --storage-retry`. Captured/pushed code: `e1833a4`.
Receipt `results/m13_storage_upload.json`; log `logs/m13-storage-upload.log`; PID file
`work/m13cloud-launchers/storage_upload.pid`. The monitor service is active and includes
this exact job. Observe its state before any further action; never rerun a live controller.

Both attempts preceding it failed before any transfer or training and are retained:
`results/m13_cloud_build_resume.json` (no GPU capacity, STOP confirmed) and
`results/m13_cpu_upload.json` (normal podResume ignores gpuCount0, STOP confirmed).
The public Runpod console uses a SEPARATE `podResumeZeroGpu(input:{podId})` operation;
the corrected launcher verified returned count0, SSH and exact code, and resumed rsync.
Measured compute quote$0.795/h plus conservative running storage$0.073611/h; the upload
still charges planning time against the original$1.663611/h ceiling. No new Pods created.

The exact394 source files (72.62GB,254 root/140 worktree mappings) verified locally.
Completed remote files are reused and interrupted transfers use persistent partial dirs.
The supervisor verifies ALL destination hashes, then STOPs. Its cap is at most10h,
with planned training plus4h preserved within the original144h/$239.56 stage allowance.
All failed intervals and paid storage enter that allowance. Do not treat transfer bytes
as a passed receipt. No GPU bootstrap, preflight, training or quality read in this job.

**Automatic handoff is active:** PID14853, launched from pushed commit `8c11ce6`.
`scripts/m13_after_upload.py` waits for PASSED,
transfer_verified,394 files and EXITED, publishes the receipt, applies the reviewed
`m13/after_cpu_supervisor.patch` and verifies its exact SHA. It then waits read-only for
retained-host GPU capacity (up to12h), reconciles all elapsed intervals and current paid
charges, publishes the new allowance and starts the build once. Receipt
`results/m13_after_upload.json`; log `logs/m13-after-upload.log`; PID file
`work/m13cloud-launchers/after_upload.pid`. Check these before any manual action; do not
race a live coordinator. Bound source/configuration files must remain unchanged while it waits.

The next build uses explicit --after-cpu, separate receipt
`results/m13_cloud_build_after_cpu.json`, GPU1 resume, current capacity and price checks,
all394 destination hashes, pinned bootstrap and full preflight before a fresh200M build.
There is NO training checkpoint yet, so no scientific --resume. Its supervisor owns backups
and unconditional STOP. An unconfirmed child startup is terminated before the coordinator's
STOP-only fallback, preventing a late paid start. The monitor service covers upload, handoff
and build independently. Failed receipts are retained; no automatic repeated launch.

Two inexpensive independent reviewers cleared the concrete storage retry and the next-step
code conditionally; details `research/m13-reboot-continuation-review-2026-09-12.md`.
Current user instruction: continue overnight, no further questions. All disks are retained,
STOP only, $1,000 project ceiling. Keep the local host awake; monitor cannot wake the assistant.

The original reboot handoff below is historical context; preserve its interrupted evidence.

# M13 reboot handoff — 2026-09-12

The user requested a safe session clear and Windows reboot. Resume **M13**, in
`/home/dylan/asymetric-dual-encoders/work/m13cloud`, branch
`m13-stage1-execution-prep`. M17 owns the original checkout; do not change its files or
use its GPU without checking availability. Read this worktree's `CLAUDE.md`,
`m13/STATUS.md`, and this file before continuing. Historical stage-table text in
STATUS may lag the completed stages described below.

## Reboot safety receipt

At the pause request, local build supervisor PID **160213** was uploading inputs
(approximately **12.4 GB** transferred). **No build preflight or training had started.**
Verified safe pause at **2026-09-12T04:40:32.688444+00:00**:
all three retained Pods report `EXITED`; local supervisor160213 and transfer children
have exited. The supervisor receipt is `FAILED` solely because of the intentional
user-requested SIGTERM; its backup completed and STOP was confirmed. No training
or build preflight ran. No local GPU compute process was reported at the check.

- Verification: `results/m13_reboot_pause.json` (`PASSED`, `safely_paused_for_user_reboot`).
- Preserved interrupted receipt SHA256: `47715c8852ed27ed1975ba032a255abe5620a16ca61118d3ba6baeebdffefc23`.
- The handoff and receipts are committed together; use `git log -1 -- m13/RESUME.md`
  and compare the branch HEAD with origin to verify publication.
- The monitor service was stopped for the intentional pause; it remains enabled.
  It may restart with WSL, but no cloud workload restarts automatically. Its pause
  receipt distinguishes intentional interruption from completion of training.

## Completed work — do not repeat

- Both registered E arms completed with verified backups; descriptive DEV-6 and E1
  completed. E1 selects **bs32**: COV 0.503626 versus 0.493800, delta 0.009826,
  lower bound 0.006236. Records and selection are pushed.
- All seven registered LoTTE gate slices completed, with veto **skipped** under the
  bs32 rule. Gate outputs and full backups passed verification; publication commit
  **9d97291**. `m13/LOTTE_GATE.json` is the completed gate record.
- The automatic handoff's allocation step failed because system Python lacked
  NumPy. Running the unchanged helper with the repository virtual environment fixed
  the environment issue; failed receipts and recovery provenance remain preserved.
- Build allocation and recovery were pushed in **d43f32fd5d02ec424700dcfe9297577a2c4be3c3**.
  Allocation: maximum **144 hours / $239.56** for setup, training, finalization and
  backup; project projection **$706.20**, headroom **$293.80** under the user's
  **$1,000** ceiling. Balance at that measurement was approximately **$489.58**.
  These are historical measurements; reconcile new paid charges before resuming.

The scientific build remains the fixed **200,000,000 examples, three cycles, bs32**,
with no extensions or recipe changes. Training alone extrapolates **62.86 hours** at
the measured 883.8 examples/s; conservative planning uses 441.9 examples/s
(125.72 hours). A DEV-6 finalization exception now preserves the completed checkpoint
as resumable `frozen_unverified`, consistent with export/parity failures. Tests and
independent review passed; this does not relax the finalization bar.

## Resume procedure

1. Confirm the prior supervisor has exited and inspect its final receipt. Preserve
   the interrupted receipt, logs, backup directory and all earlier failure records.
   `work/m13cloud-launchers/build.py` intentionally refuses existing receipt/backup
   evidence; **do not blindly rerun it or delete its guards/evidence**. Prepare a
   narrowly scoped continuation with a new receipt/log and a hash link to the
   preserved attempt. No scientific build `--resume` is appropriate unless a real
   training receipt/checkpoint exists; none existed when this pause was requested.
2. **Before restarting or renting any Pod**, reconcile paid charges and the unused
   portion of the144-hour/$239.56 stage allowance; preserve the original allocation.
   The previous attempt already consumed setup/upload time. Do not reset the full
   allowance or ignore storage charges during this user-requested pause.
   Prefer the same Pod **wnzk8eeqrrkw4m**. Its persistent mount is `/home/dylan`, with
   cloud checkout `/home/dylan/asymetric-dual-encoders`. STOP preserves completed
   uploads. `/opt/m13-runtime` is **container storage** and disappears on STOP;
   rebuild that runtime with the reviewed pinned bootstrap before using the
   checkout's `.venv` symlink. Refresh SSH endpoint details from the authenticated
   provider response. Keep exact pushed code and all registered input bindings.
3. Continue the **394-file** transfer specified by committed
   `m13/build_transfer_manifest.json`; it contains exact source/destination paths,
   sizes and SHA256 values. Reuse completed files. The reviewed staging needs
   `rsync -a --no-owner --no-group --info=progress2`; use a partial directory for
   interrupted-file continuation, and preserve the manifest's root/worktree mapping
   and its required symlink handling. The provider volume rejects ownership changes.
   Verify **all 394 destination hashes** after transfer; progress bytes alone do not
   establish completion. Do not broaden the file list.
4. Run `scripts/m13_build_preflight.py` with the cloud `.venv/bin/python` after
   staging. This checks full uncut registered corpus assembly, exact admitted COV
   cache integrity and all descriptive DEV-6 cache reuse with encoders disabled.
   It may populate CPU token caches and refresh existing cache metadata. Require its
   PASSED receipt and source/input bindings before training. Preserve legacy DEV-6
   provenance; do not re-encode or silently upgrade its labels.
5. Reconcile the remaining stage allocation, actual paid spend and account balance.
   Run `work/m13cloud-launchers/build_allocation.py` using the **local M13
   `.venv/bin/python`**, with `CUDA_VISIBLE_DEVICES=''` and small CPU thread limits;
   system `/usr/bin/python3` cannot import its budget dependencies. Existing
   allocation output is protected against overwrite, so preserve it and use an
   explicitly reviewed continuation allocation. Do not silently reset spent runtime
   or budget. The cloud build supervisor itself uses only the standard library and
   may run under system Python.
6. Resume the supervised build only after the gate's committed/pushed identities,
   refreshed allocation and full preflight pass. Retain hourly verified rolling
   backups, final full checksum verification, bounded runtime and unconditional STOP
   handling. Add the continuation to the monitor before launch. Readiness and review
   apply to the concrete continuation, not an automatic retry of a failed attempt.

Runpod has twice lacked free GPU capacity after a stopped Pod was restarted. If this
recurs, do not create duplicate Pods or retry creation blindly. Reconcile any
capacity fallback against existing paid spend, retained storage and the $1,000
ceiling, then review the minimal continuation. All existing persistent disks must
be retained: **STOP, never TERMINATE**. Prior retained Pods are
`k3aee2m68765em` and `exulxoxelug5um`.

## Evidence, credentials and boundaries

- Gate: `results/m13_cloud_gate_chain_upload.json`,
  `results/m13_encode_benchmark_fp16.json`, `m13/LOTTE_GATE.json`;
  gate backup manifest: `work/m13cloud-launchers/gate-chain-upload-outputs.json`.
- Handoff/allocation: `results/m13_after_gate.json`,
  `results/m13_after_gate_recovery.json`, `results/m13_build_allocation.json`.
- Interrupted build: `results/m13_cloud_build.json`,
  `logs/m13-build-controller.log`, and any `work/m13cloud-build-backup` contents.
  An empty build-output backup is expected: no training output existed.
  Operational source: `work/m13cloud-launchers/build.py` and
  `work/m13cloud-launchers/build_allocation.py`.
- Exact operational source is also pushed on branch `m13-execution-audit`, commit
  `e37f837`, under `audit/m13/20260912T010808Z/work/m13cloud-launchers/`.
  The manifest there verifies all archived source bytes.
- Local credential locations only: `~/.config/runpod/api_key`,
  `~/.config/runpod/m13_ssh`, `~/.config/runpod/m13_ssh.pub`,
  `~/.config/runpod/m13_gate_chain_ssh_config`; earlier aliases use
  `m13_ssh_config` and `m13_replacement_ssh_config` in the same directory.
  Never print key values or upload the Runpod API key to a Pod.
- After Windows/WSL restart, verify `m13-monitor.service` is active and update its
  exact job/PID/receipt paths. Monitoring requires the host to remain awake; it does
  not wake an assistant session. See `m13/MONITORING.md`.

**No repeat of E, E1 or the LoTTE gate. No final six-set or reserved access during
recovery.** Do not read or overwrite `results/perquery.json`,
`results/frozen_eval/untouched-*`, reserved qrels or `work/m9reserve`; do not inspect
LoTTE payloads to diagnose infrastructure. Final six/reserved evaluation remains a
separate registered gate after a verified build freeze.
