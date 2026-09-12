# M13 status — E complete; batch 32 selected (2026-09-11)

**Now:** Both registered cloud E arms completed, their full backups verified, and Runpod
STOP confirmed at 21:45 UTC. Both deferred DEV-6 evaluations completed locally and their
records were pushed. E1 resolves in favor of batch32 under the unchanged registered rule:
COV 0.503626 versus 0.493800, delta 0.009826, lower bound 0.006236. E-bs32 took 97 minutes;
E-bs128 took 35 minutes. Cloud execution through backup/STOP took 2.25 hours (about $3.74
at the quoted running rate, excluding earlier preparation and subsequent retained storage).

The local follow-on completed evaluation and selection but stopped before publication because
its final check expected an integer batch rather than the actual `bs32` string. The check is
fixed; the original failed receipt is preserved, and the existing outputs were validated without
rerunning observations. Receipts: `results/m13_cloud_e.json`, `results/m13_after_e.json`,
`results/m13_after_e_recovery.json`; decision: `results/m10_contrast_E1.json`.

**Next (active launch):** the checkpoint manifest and seven-slice pin are committed and
pushed; metadata-only gate preflight passed for bs32. Prior Pod restarts failed on host GPU
capacity. The combined launcher retains one replacement A100 through
setup, fp16 timing and the registered observational LoTTE read, then verifies backups and
stops. Persistent-volume package installation failed with a stale file handle before encoding.
A bounded takeover repairs the runtime on container-local disk with identical package pins;
the original receipt remains preserved and its 17-hour deadline is not reset.
Runtime repair passed all pinned package/GPU checks. Transfer then encountered MFS
file-ownership restrictions; the continuation disables ownership preservation, with file
content still verified by SHA. Live receipt `results/m13_cloud_gate_chain_upload.json`,
log `logs/m13-gate-chain-upload-continuation.log` and
`logs/m13-lotte-gate.log`. It owns no full build or final six/reserved evaluation.
Stage ceiling$28.29; revised conservative project allocation$669.52 under$1,000, including
three retained disks. `results/m13_gate_chain_allocation.json` binds the prerequisites.
M17 keeps the original checkout and local GPU; M13 uses `work/m13cloud`.

Independent monitoring is active as `m13-monitor.service`: five-minute checks, local Windows
alerts, and durable health/transition records. Add each next job before launch; configuration,
operating limits and verified failure tests are documented in `m13/MONITORING.md`.

| Stage | State / exit |
|---|---|
| Execution preparation | Done. Fixed 200M `build13` (53 tests, GPU smoke) and `score13`/`access13` (49 tests, rehearsal with crash and recover). Two Codex reviews plus a P1 re-check; its three residual P1s (gate/E1 consistency and registered checkpoints, fastembed in the freeze bar with a checkpoint-bound build record, teacher pin before spend) closed with tests. Reviews and triage: `research/m13-codex-*-2026-09-10.md`, `m13/REVIEW_TRIAGE.md` |
| Cloud E comparison | Complete. Both arms and deferred local DEV-6 passed; E1 resolves for bs32. Full local backups verified and Runpod STOP confirmed |
| Pre-build gate | Registered (`m13/LOTTE_GATE_REGISTRATION.json`, amended 2026-09-10 under R17/R18); executor `m13src/lotte_gate13.py` (R16). Reviewed to **GO** (Astra nine, Sol ten, Astra three, closing re-check GO; `m13/REVIEW_TRIAGE.md` §Stage 2). **Before the read:** both E records pushed, the committed and pushed manifest and pin (R18); the gate record is still owed |
| Build | Actual rate/price and complete allocation under $1,000; then fixed 200M examples in three cycles (no extensions, R13), freeze/provenance |
| Evaluation | Locked/reviewed six-set executor, M9 close-out, nano decisions and conditional reserved access |
| Cost frontier | Comparable zero/bge-small/nano serving and index costs on the reference hardware |

Detailed execution defects and the day-one runbook have one home: `m13/EXECUTION.md`. Recipe:
`m10/M102_LOCK.md`. Lessons: `m13/FINDINGS.md`. M13 worktree: `work/m13cloud`; M17 owns the original checkout.
A recipe push alone triggers neither M9 close-out nor final access. M9's rows must not inform
an open recipe decision. LoTTE handling belongs before the expensive build when its veto applies.

Done means a frozen candidate or documented stop, durable measurements/decisions and an evidence
handoff. A miss is publishable. Release is M14; the paper is M15. `ROADMAP.md` maps the old numbers.
