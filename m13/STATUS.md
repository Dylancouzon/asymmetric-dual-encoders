# M13 status — Registered E running (2026-09-11)

**Now:** Registered E execution is RUNNING on Runpod, pinned to `2ec2f0a`.
E-bs32 started 2026-09-11T19:30:14Z, with its registered 5M examples, batch32, bf16, 60k
warmstart; E-bs128 follows unchanged. DEV-6 is deferred to the box. Reviewed local controller
has a 24h work cap (11.5h per arm), full checkpoint/COV backups with hashes, and STOP cleanup.
First scheduled COV evaluation completed at step26,041 without errors; training rate before
that evaluation was 827 ex/s (early cumulative rate, not a final cost).
Live receipt `results/m13_cloud_e.json`; logs `logs/m13-cloud-e-*.log`, wrapper
`logs/m13-e-launch.log`. Keep WSL awake. M17 keeps the original checkout.

Both full-shape restart smokes passed. Dependency preflight verified all ten COV units and
five teacher document caches. The measured allocation is $740.53 including contingencies,
not expected spend; about $4.74 was paid before E launch, with $500.26 credit. E's full24h
reserve is $39.93. `results/m13_cloud_allocation.json` binds all passing receipts. Stella's
101–111 passage/s benchmark was fp32 with TF32 off; actual LoTTE uses fp16, so the 15h LoTTE
allowance is a conservative surrogate and must be replaced with matched-path timing before
expensive encoding. No protected evaluation or full build has run.

**Next:** finish both E arms, verify the final local backups and STOP receipt, commit their
records, then fill deferred DEV-6 locally and compute E1. Local DEV-6 dependencies are ready:
`work/m13cloud-dev6-links.json` lists exact binary/data links with independent metadata copies.
All six dependency checks passed (`results/m13_dev6_preflight.json`), with historical cache
provenance preserved. The twice-reviewed `scripts/m13_after_e.py` automates the local handoff
after verified cloud completion and STOP, ending at E1 selection. Its live receipt is
`results/m13_after_e.json`; a busy local GPU stops the handoff for manual continuation.
An extra rolling E-bs32 checkpoint at step15,624 is verified in `work/m13cloud-e-live/`.
The existing encode benchmark now accepts `--dtype fp16` with a separate receipt, ready for
matched-path timing after E; it is not deployed to the active Pod. Then follow the registered
LoTTE manifest/pin/preflight/gate sequence. Ship list: `m13/SHIP_LIST.md`.
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
