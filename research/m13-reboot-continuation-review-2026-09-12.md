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
