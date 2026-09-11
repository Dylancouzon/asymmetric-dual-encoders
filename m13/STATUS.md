# M13 status — Checkpoint restart repair (2026-09-11)

**Now:** Initial cloud validation uploaded and verified all 266 artifacts; CPU checks and both
512-token E shape smokes passed. The checkpoint restart smoke failed because recipe identity
included loader timestamps/timings. Evidence was backed up and the Pod stopped automatically
(about $3.94 through shutdown). The narrow fingerprint fix passes 151 runner/build tests;
independent reviews and a fresh cloud restart check are next. Initial receipt:
`results/m13_cloud_stage0_attempt1.json`; full local evidence: `work/m13cloud-attempt1/`.
No registered E training or protected evaluation has run. M17 keeps the original checkout.

**Next:** inspect the stage-0 receipt/evidence, measure the remaining encode allowance, put
measured rate and billed price into the allocation table (`build13.py --plan --rate --price`),
then both E arms with
`run_arm.py <arm> --dev6 defer`; DEV-6 on the box from the returned `cycle3.pt` via
`m13src/dev6_from_checkpoint.py`. Stage 3: `m13src/lotte_gate13.py --preflight-only`, then the read,
once. Ship list: `m13/SHIP_LIST.md`. Branch `m13-stage1-execution-prep`.
Rulings R1–R16 are recorded and applied (`m13/RULINGS.md`); R6 flips at the pinning commit.
M10's prepared data, completed screen and selected components are the input, not work to repeat.

| Stage | State / exit |
|---|---|
| Execution preparation | Done. Fixed 200M `build13` (53 tests, GPU smoke) and `score13`/`access13` (49 tests, rehearsal with crash and recover). Two Codex reviews plus a P1 re-check; its three residual P1s (gate/E1 consistency and registered checkpoints, fastembed in the freeze bar with a checkpoint-bound build record, teacher pin before spend) closed with tests. Reviews and triage: `research/m13-codex-*-2026-09-10.md`, `m13/REVIEW_TRIAGE.md` |
| Cloud E comparison | `pending` cleared under R9, F verdict and twelve contrasts re-bound byte-identically; both arms unrun; E1 applies after both finish. DEV-6 deferral (`--dev6 defer` + `dev6_from_checkpoint.py`, 16 tests) lets the arms run without the 35 GB DEV-6 caches |
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
