# M13 execution backlog — 2026-09-10

**Next (2026-09-10, stage 1 closed):** stage 2 — LoTTE gate script (R16), DEV-6-from-checkpoint or
ship the DEV-6 caches, allocation table on day one, provider (Dylan). The nano scorer is
`m13src/score13.py` (rehearsed on the synthetic fixture); M9's path in `final9.py` is still the stub
and is wired last, behind the recipe-decision gate.

| Work package | Completion condition |
|---|---|
| Pre-build LoTTE | Metric and seven slices fixed before access; compare correct cloud E checkpoints; enforce pre-build skip/veto and execution record |
| Executor identity | Verify comparator hash; bind rows to checkpoint/system, code/registry and exact frozen qids; iterate all six, not bench's five-set default |
| Durable access/recovery | Authenticated empty manifest before spend; handle crash before first score; allow only declared output drift; pin code/config on continuation/recovery |
| Preflight / M9 | No protected query/qrel reads in preflight; restore inherited 120 GB space requirement; resolve M9's legacy bridge and ratification before close-out |
| Review and budgets | Rehearse the real caller; clean independent reviews; include conditional reserved cost in M13 allocation; register paired M9/nano row before either final run |
| Cost frontier | Same-machine zero/bge-small/nano serving costs and complete system/index costs; Edge fusion still unmeasured |
| Build controller (found 2026-09-10) | 200M pin, extension rule and cap, ONNX/parity at freeze, spend accounting have no code; `arm_doc_count` demands 50M unique documents at 200M against a ~6.15M pool; wall-clock checkpoint cadence; full uncut A4 for the build. Design and owner questions: `m13/STAGE1_DESIGN.md` |
| Cloud E (found 2026-09-10) | `E-bs32` missing from runner shape tables; `pending` blocks E1 and clearing it re-binds the F verdict and ten contrasts (ruling R9); DEV-6 caches or box-side DEV-6 for the E arms; reserved-batch encodes unpriced (R10). Details: `m13/STAGE1_DESIGN.md` §4 |
| Scoring executor (found 2026-09-10) | No access machinery under M10 paths yet; registry cites M7's payload-reading preflight while this file requires manifests only (ruling R1); `min_free_gb` 120 absent from M10's registry and code; M9 still carries the withdrawn 0.0003 bridge (R3) |

Original findings: `research/archive/m10-cleanup-2026-09-10/m10/M10_4_DECISION_LOCK.md`.
The missing ±1.5 decision-range test is now covered. Scope/freeze prose and budget-range errors
were removed from the active handoff; remaining executor findings are **not claimed fixed**.

**Gates:** no six/reserved/LoTTE access, no spent tag, and no M9 close-out triggered merely by a
recipe-lock push. LoTTE handling may block M13's full build; final six-set scoring waits for its
frozen candidate. M9's rows must not inform any still-open recipe decision.

The bridge's 0.0003 per-query tolerance was never validated for re-encoding. Its alleged minimum
nDCG quantum is false: multiple rank changes can cancel. See
`research/m10-cleanup-review-2026-09-10.md`. Resolve the intended parity contract before execution;
do not copy a changed tolerance into M9 without its pre-observation amendment.
