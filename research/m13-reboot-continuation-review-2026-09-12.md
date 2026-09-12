# M13 upload continuation — 2026-09-12

The reboot interrupted input staging before preflight or training. The original attempt,
allocation, pause receipt, logs and backup remain intact. The continuation uses tracked
`scripts/m13_resume_build.py` and `scripts/m13_resume_allocation.py`, separate receipts/logs/
backups, and the same retained Pod. Scientific configuration and all 394 transfer entries
are unchanged. Completed files are reused; partial transfers use persistent writable paths.

The continuation subtracts the original attempt through verified STOP from 144 hours, and
subtracts the greater of elapsed-time cost and all new paid charges from $239.56. Balance
charges include storage during the pause. Whole-second rounding cannot increase either cap.
Live reconciliation runs before restart and before training, including continuation wall time.
The original project projection is retained conservatively plus consumed cost; no original
allocation or failed evidence is overwritten. No initial scientific `--resume` is issued.

Two independent gpt-5.6-luna reviewers checked the named launcher, allocation and handoff files.
Launcher reviewer GO after writable absolute partial directories and unique source/destination
checks. Budget review also requested archive reuse verification and conservative elapsed spend;
these were implemented. The inherited STOP block remains independent of SSH readiness.
Reviewers accessed only named operational code, configuration, and receipt metadata; no cloud,
credentials, protected six/reserved data, frozen comparator payloads or LoTTE payloads.

Validation: 28 synthetic allocation, preflight and monitor checks passed, including delayed
billing, interrupted runtime, insufficient funds and changed balances. Existing monitor is
active after reboot and now includes the unique continuation job, logs and PID file.

The 56 existing build-controller tests also passed. Refreshed live allocation:142.2828h,
$236.7032 stage cap, $709.06 conservative project projection, $486.7241 balance.

Final budget reviewer GO: archive reuse, consumed-cost projection, continuation elapsed
accounting and original bindings verified. The tentative STOP finding was retracted after
AST/indentation inspection: STOP is in the inner finally outside `if ready`. Both independent
reviews are GO for the concrete continuation, conditional on fresh live checks (allocation
PASSED and pushed in c8dc2bb). No P1 remains.

## Same-Pod capacity failure and zero-GPU staging

The reviewed GPU restart failed with HTTP500 before SSH/staging. The supervisor confirmed
EXITED; `results/m13_cloud_build_resume.json` is preserved. A read-only provider GraphQL
query reports `machine.gpuAvailable: 0` on all three retained hosts. No new Pod was created.
Runpod documents zero-GPU restart for storage access:
https://docs.runpod.io/pods/troubleshooting/zero-gpus . The GraphQL `podResume` API accepts
`gpuCount: 0`; the upload helper requires the returned count to be exactly zero.

`scripts/m13_upload_only.py` resumes only the same persistent Pod for exact input staging.
It bounds upload to at most10 hours and also reserves the conservative planned training
plus4 hours inside the remaining original144h/$239.56 cap. Failed restart wall time and
paid storage are deducted by the same reviewed allocation helper. It verifies the current
quote against the prior GPU+storage ceiling, all394 local source hashes and destination
hashes, and STOPs unconditionally. No runtime bootstrap, preflight, training or scoring.
Separate receipt/log/PID and monitor job preserve the failed GPU attempt unchanged.

CPU-only budget reviewer final GO after exact zero-GPU response validation; all original
runtime/paid-charge bounds remain. Synthetic upload inventory (including duplicate and
protected-path refusals), allocation and monitoring:25 tests passed.

The complete local manifest verification passed:394 pinned hashes (254 root-mapped and140
worktree-mapped),72.62GB,178.6 seconds. No input bytes or registration changed.

CPU-only launcher reviewer final GO as well. Before a later GPU launch, its allocation must
bind the CPU upload receipt, deduct that interval/charges, and rehash all394 destination
files before preflight. Both reviews used named code and receipt metadata only.

## Correct storage-access API

The standard `podResume(gpuCount:0)` rejected the request as unavailable GPU capacity;
`results/m13_cpu_upload.json` preserves the failed1.743-second attempt and confirmed STOP.
This matches the provider SDK's open issue https://github.com/runpod/runpod-python/issues/371 .
The current public console JavaScript instead uses `podResumeZeroGpu(input:{podId})` for
zero-GPU storage access (console bundle026mioaja-76o.js defines it;0g1voboygx6xl.js calls it
with only podId). The CLI now exposes a narrowly guarded `--storage-retry` for exactly this
failure, writing `results/m13_storage_upload.json` and adding the failed interval and hash.
Same inputs, cap, zero-GPU response check and STOP. Budget reviewer final GO;25 synthetic
checks rerun and passed. No alternative Pod was created.

Launcher recheck also GO with the failed CPU receipt SHA256 pinned explicitly; that guard
is now present (`bfffcac4…462002b`). This prevents altered timestamps from entering the
retry allowance. The public console operation, response validation and all prior limits
were reviewed; neither reviewer accessed credentials or protected payloads.

The dedicated storage operation succeeded; supervisor11028 verified zero GPU response,
SSH, exact e1833a4 checkout, restored rsync and reached uploading-build-inputs. Observed
compute$0.795/h plus conservative storage$0.073611/h, within the prior ceiling. Local source
hashes passed again; no new Pod, preflight, training or protected quality read.

After-upload allocation helper and supervisor draft received both code-readiness GOs, strictly
conditional on a passed394-file storage receipt with STOP, exact prior receipt bindings,
current free GPU capacity and fresh balance. Helper scratch tests cover all elapsed intervals
and incomplete/changed receipts. Apply the supervisor draft only after upload completion.


## Independent overnight transition

The after-upload coordinator publishes only the final verified storage receipt, applies the
reviewed supervisor patch with exact SHA verification, polls retained-host capacity without
resuming compute, then calculates/publishes a fresh cumulative allowance and dispatches once.
Source and original input bindings remain checked while waiting. Monitor-active verification
is required before dispatch. It never allocates a new Pod or opens quality payloads.

Budget review caught a startup-timeout race leaving an unconfirmed child alive. The fix
terminates the owned process group, waits, escalates if needed, and only after quiescence
issues a bounded STOP fallback to the exact retained Pod. Budget recheck GO. Meaningful tests
exercise actual process-group cancellation before mocked provider STOP and forced-kill order,
as well as isolated Git publication and refusals. Combined handoff, allocation, upload,
recovery and monitor suite:37 passed. Launcher final review recorded before launch.

Launcher review additionally required STOP in finally and guarded receipt I/O so cleanup
errors cannot bypass the provider fallback. Implemented, with injected termination-error
coverage;38 tests pass. Both reviewers issued final GO on the concrete final coordinator.

Published8c11ce6 and launched coordinatorPID14853. Confirmed WAITING receipt and independent
monitor waiting state with live process identity. Storage supervisorPID11028 remains RUNNING
with advancing transfer. Windows AC automatic sleep is already disabled (read-only query).
No GPU training has started. Automatic handoff requires no further user interaction.


## Completed transfer, verification timeout

Both rsync groups finished; destination sha256 --status exceeded its1800s command timeout.
No checksum mismatch was observed, but verification did not pass. Storage receipt413c7856
and coordinator receipt2fcad24c are preserved; Pod EXITED, no training. Transfer averaged
roughly2–3MB/s, explaining the hours to move tens of GB. The silent checker provided no
read-rate diagnosis; do not claim a specific network or disk bottleneck without measurements.

Verification-only continuation streams the exact394 manifest files with15-second byte progress
and per-file completion, using a transferred hash-bound stdlib helper and manifest. No large
retransfer, runtime setup or evaluation. Caps3h and totalstorage10h, fresh balance and original
training+4h preserved. Live read-only reconciliation:136.1397h remaining, conservative stage
consumption$13.0760, project projection$719.28. Budget reviewer GO; launcher review addresses
durable log capture. Synthetic394-file pass, same-size corruption and size mismatch tests plus
prior allocation/upload tests:18 passed. Downstream chain update required before GPU launch.

Launcher final GO with exclusive stdout/stderr log and PID reservation before dispatch;
monitor tracks that log. Both reviews cleared verification-only paid restart.

Verification-only controllerPID25534 launched from6c75510, exclusive log/PID reserved first.
Downstream allocation now includes failed storage and successful verification intervals; exact
failed receipt pin and combined10h storage ceiling remain enforced. The new handoff receipt is
m13_after_verification, with the previous failure pinned and preserved. GPU supervisor draft
24cbc20e uses the same progress verifier with90-minute command cap inside its existing stage
alarm, then fresh training admission. Budget reviewer GO. Combined synthetic suite:41 passed.

Launcher required remote helper/manifest SHA assertions before GPU rehash; implemented in
final draft846e93b7. Both reviewers GO. The first verification launcher refused before any
paid resume: provider reports container5GB (persistent500GB mount unchanged). Preserved
prestart refusal receipt/log/PID. Narrow verification-only guard permits5or30GB, still priced
at the original530GB bound; reviewers recheck. GPU still requires explicit30GB restoration.

Both reviewers GO on the narrow5or30GB verification-only guard; new controllerPID27527
launched fromfd84d45 with exclusive retry log/PID. No prior verification receipt overwritten.
For GPU readiness, the coordinator now restores only containerDiskInGb30 via PATCH while
all retained Pods are EXITED, verifies persistent500GB mount unchanged and stopped status,
and STOPs on any uncertain update. Official API documents this single-field update:
https://docs.runpod.io/api-reference/pods/POST/pods/podId/update . Three synthetic tests cover
exact field, active-Pod refusal and STOP on uncertain update;29 affected tests pass.

Both reviewers issued final GO for concrete stopped-container restoration and coordinator.

Published01ba410 and launched replacement coordinatorPID28201; confirmedWAITING receipt.
VerifierPID27527 completed local checks and reachedRUNNING starting-zero-gpu.

Remote verification completed394files/72,623,148,280bytes in259.6s, no mismatch. Verified
receiptPASSED andSTOP confirmed. Coordinator published it (0e51d83), restored container30
whileSTOPPED, applied/published exactreviewed supervisor (a15cf1e), then reachedwaiting_gpu
with0available. The previous silent timeout has no proven specific bottleneck; the new
measurement disproves any claim that hashing this payload necessarily needs30minutes.


## User-authorized return to the original E host

New provider checks found originalEhost k3aee2m68765em with6 then5 free GPUs while the
uploaded source stayed0. Exact model matchesNVIDIA A100-SXM4-80GB and quotedGPUprice1.59.
The earlier Ehost switch was justified by two no-capacity restarts, documented in
research/m13-e-preparation-review-2026-09-11.md. Today's capacity wait was too narrow;
all retained hosts should have been rechecked sooner. User authorized migration and training.
Old waiting coordinator stopped before dispatch and receipt preserved. Concrete migration,
cumulative allowance and direct build-adoption supervisors prepared with two inexpensive
reviewers. Runtime-preservation and isolated handoff tests:9passed, no real payload access.

Migration integration review corrected startup timestamps, all-in source/target pricing,
preallocation ancestor binding after publication, exact GraphQL model/count checks, rsync
restoration after container reset, matching verifier manifest paths, exclusive atomic READY,
and cleanup before sourceSTOP. Transfer deadline reserves a bounded build-claim lease; target
stays running only after the separately reviewed build takes responsibility for STOP. Budget
reviewer GO, final integration recheck underway.23 focused synthetic checks passed.

Both reviewers issued final GO after source pre-resume all-in quote and committed migration
reserve checks. Real read-only preallocation passed:134.53694h/$223.817 remaining after the
full worst-case1h concurrent migration reserve, project$721.95, balance$483.9475. Original
allowance is not reset.23 focused checks rerun after final edits; reviewer broader38pass.

Published8b67154, launched migrationPID38227 and waiting handoffPID38228. Exact source0GPU
and target1GPU resumes succeeded. Remote nvidia-smi confirmsA100-SXM4-80GB/81920MiB. Shared
root cache group reused all but6.7KB; tree group direct transfer averages roughly19MB/s.
No training yet; target remains reserved for automatic adoption.

## Completed migration and fresh handoff

The direct transfer completed with about24.45GB actually transferred, reusing remaining
files. All394 destination hashes passed and source STOP was confirmed. A proposed longer
migration allowance/takeover was abandoned without execution; temporary source edits were
restored. The first waiting coordinator correctly failed on source binding change before
any child dispatch. Its immutable failure is preserved. A fresh coordinator differs only
in receipt/self-binding paths and keeps the original budget and build admission checks.
Focused handoff/allocation checks:15 passed. Independent budget review GO (40 checks).

Both independent reviewers cleared the recovery coordinator. Published39ab12d and launched
PID43732; it published READY7b563ed and actual cumulative allocation8ed37ed, then handed
off to buildPID43900. Migration supervisor confirmed the claim and exited PASSED. Runtime
bootstrap passed on the original E A100; build input rehash394/72.62GB passed in57seconds.
Full-build-preflight active. No training steps observed at this update.

## DEV-6 imported-reader hook correction and preflight retry

Full6,070,049-document tokenization completed. All COV cache checks passed and all six
DEV readers returned, but the preflight refused its final four-text-cache inventory:
dev_eval imports encode_cached by value, bypassing the wrapper installed on teacher.
Corrected checker wraps/restores both bindings and refuses memoized evaluator reads.
Six synthetic checks cover imported alias, corruption refusal, restoration, and memoized
bypass. No scoring, training or protected-final data reads. The supervisor backed up an
empty build manifest and confirmed targetSTOP. Completed token cache remains reusable.
New separate retry supervisor uses same recipe/runtime/backup/STOP path and resumes only
the exact stopped A100 target. New allocation charges this attempt through verifiedSTOP.

Retry arithmetic replays immutable migration at the failed build start, then adds its
interval through confirmedSTOP. A narrow historical replay correction passes the supplied
time through validate_ready; the one-hour migration cap is unchanged. Actual prior chain
replays to7.747302977812561h before the original pause interval. Failed receipt exact SHA
2089ceb3769a9779541d14c6c3b0c4820426eb4a379d795ed5d73c93c28e750a is pinned.
Focused22 tests passed; reviewer broader42passed. Monitor has unique retry job/receipt/PID.

Both reviewers gave finalGO for the corrected DEV6 hook, historical replay, retry supervisor
and unique monitoring. Published54eb078 then live read-only allocationPASSED134.8888889h/
$224.4026543, balance482.0799693/project721.36. Allocationpushedee1bd66. LaunchedPID50427;
exactoneA100resume succeeded and deploymentactive. No trainingsteps observed yet.

## Training confirmed

Correctedpreflight PASSED all394inputhashes, fulluncutassembly, COVintegrity and allDEV6
cachechecks. Completed6.07Mdocument cache reused. Registeredwarmstart60000examples
finished in74seconds. PID50427 executionee1bd66 reachedoptimizerstep4142, 132544examples,
reported1010ex/s. Observation recorded separately from mutablelive receipt. Monitoractive/
enabled; rollingbackup/finalbackup/STOP ownedby supervisor. Remainingtrainingprojection
about55.0h excludingevaluations/finalization. No final/reserved access.
