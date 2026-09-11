# M13 findings — cloud build and evaluation

Read with `m13/STATUS.md` and `m13/EXECUTION.md`; result JSONs are authoritative. As of 2026-09-10
no M13 observation exists: both E arms are unrun, LoTTE is unread, no build has started and no
protected surface has been opened. This file holds the harness lessons from preparing the
executors; the observations table starts with the first cloud record.

| Observation | Supported interpretation / limit | Evidence |
|---|---|---|
| — | none yet | — |

## Harness lessons worth carrying forward (2026-09-10)

- **Scope cut before spend, not after.** Asked plainly, the lead called the extension cycles and the
  post-tag crash-recovery machinery over-engineered; rulings R13 to R16 removed them and replaced
  reliability-by-recovery with rehearsal on open data. The build is a fixed 200M with two stop
  rules; the LoTTE gate is a script, not an executor family.
- **A gate record is written by the code that performed the read, and the consumer enforces the
  contract.** `build13.check_gate` once accepted `{}`; it now refuses anything without
  `executed: true`, a decision, a branch, both checkpoint shas and the live E1 verdict sha, and a
  record for the branch E1 did not select.
- **Shared id spaces break silent assumptions.** LoTTE's qids and pids are both small integers, so
  the repo's BEIR self-hit rule would have dropped every query's same-numbered positive without any
  error. `m13/CODEMAP.md` pitfall 16.
- **One protected-access claim per process; claim at run time.** `m13/CODEMAP.md` pitfall 17.
- **A code identity hashes the tree that runs.** `m13/CODEMAP.md` pitfall 18.
- **Deferring a descriptive read is not skipping it.** DEV-6 for the cloud E arms moves to the box,
  from the identical checkpoint bytes, with the deferral and the fill both recorded.
  `m13/CODEMAP.md` pitfall 19.
- **Reviews are counted.** Two Codex passes plus one P1 re-check closed stage 1 (R15). The LoTTE
  gate script is newer than those reviews and performs a one-shot protected read, so it still owes
  the two independent reviews `CLAUDE.md` requires before irreversible execution — before stage 3,
  not before renting.

## Cloud restart smoke — 2026-09-11

The initial 512-token E-bs32 smoke was interrupted at step 100, then correctly refused a
mismatched fingerprint. Root cause: `run_arm.fingerprint` hashed query `generated_at` and
per-source `seconds` from the loader. These operational fields change despite identical data.
The fix excludes only these exact nested fields, without mutating the evidence manifest;
source hashes, row counts, recipe fields and code identity remain checked. `build13` uses the
same fingerprint and inherits the repair. Old checkpoints are not migrated or bypassed.
Both registered E arms remain unrun. Failure receipt: `results/m13_cloud_stage0_attempt1.json`;
checkpoint and logs: `work/m13cloud-attempt1/m13cloud-evidence/`.

Fresh retry on `ca9ccdc` passed both 512-token, 600-step real E smokes: each resumed from
step 100, preserved prior loss history and finished all remaining steps with matching checkpoint
identity/counters/hashes. Cloud CPU checks also passed. Local backups were verified and STOP
confirmed. `results/m13_cloud_stage0.json` and `results/m13_cloud_resume_smoke.json` contain
the evidence. Stub COV values are synthetic smoke checks, not retrieval-quality observations.
Short smoke rates vary across attempts and include instrumentation/checkpoint effects; they
are not yet the measured allocation needed to authorize the full build.
