# M13 status — storage upload resumed (2026-09-12)

**Live:** same-Pod zero-GPU upload supervisor11028; `results/m13_storage_upload.json`,
`logs/m13-storage-upload.log`. Exact code deployed and rsync active. No training yet.
Both GPU restart and normal zero-count API attempts failed/STOPped; dedicated
`podResumeZeroGpu` succeeded. Current details and next-step gates: `m13/RESUME.md`.

The reboot pause remains preserved in `m13/RESUME.md` and its original receipts. The
reviewed continuation is `scripts/m13_resume_build.py`, with separate receipt
`results/m13_cloud_build_resume.json`, controller log `logs/m13-build-resume-controller.log`,
training log `logs/m13-build-resume.log`, and backups `work/m13cloud-build-resume-backup`.
It restarts only the same retained Pod, resumes the exact 394-file upload, verifies every
hash, runs full preflight, and checks live remaining funds before the first training step.

Refreshed allocation: 142.2828 hours / $236.7032 remaining, project projection $709.06,
account balance $486.7241. Source: `results/m13_build_resume_allocation.json`; later live
checks may only reduce the allowance. Both original elapsed setup and pause storage are
charged, with no reset of the original 144-hour/$239.56 ceiling. No training existed at
pause, so this is upload recovery followed by a fresh registered build.

Independent monitoring is active and includes the continuation. The supervisor enforces
stage deadlines, hourly verified rolling backups, final checksums and unconditional STOP.
Reviews: `research/m13-reboot-continuation-review-2026-09-12.md`. Validation: 28 recovery/
preflight/monitor and 56 existing build-controller tests passed. No final evaluation access.

Both registered cloud E arms completed, their full backups verified, and Runpod
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

**Now (2026-09-12):** All seven LoTTE gate slices completed. Veto skipped under the
registered bs32 branch; descriptive macro nDCG@10 0.4610 and Success@5 0.7528. Full
backups verified, Pod STOP confirmed, and gate results pushed in9d97291. The automatic
handoff failed after publication because system Python lacked NumPy for budget imports.
The unchanged allocation helper passed under the repository virtual environment; the
original failure and hash-bound recovery are preserved. No observations repeated.

Build allocation: maximum144h/$239.56 including setup, training, finalization and backup;
project projection$706.20, remaining headroom$293.80, account balance$489.58. These are
conservative caps; training alone extrapolates62.86h at the measured E-bs32 rate.
Receipts: `results/m13_build_allocation.json`, `results/m13_after_gate_recovery.json`,
then `results/m13_cloud_build.json`. Build logs: `logs/m13-build-controller.log` and
`logs/m13-build.log`. M17 retains the original checkout/GPU; M13 uses this isolated worktree.

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
