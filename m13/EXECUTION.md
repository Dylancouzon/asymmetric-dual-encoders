# M13 execution backlog — 2026-09-10

**Next:** build and rehearse the nano scoring transaction. Decision helpers already have tests;
there is no production nano scorer. `m9src/final9.py` stops at `SCORING PATH NOT IMPLEMENTED`.

| Work package | Completion condition |
|---|---|
| Pre-build LoTTE | Metric and seven slices fixed before access; compare correct cloud E checkpoints; enforce pre-build skip/veto and execution record |
| Executor identity | Verify comparator hash; bind rows to checkpoint/system, code/registry and exact frozen qids; iterate all six, not bench's five-set default |
| Durable access/recovery | Authenticated empty manifest before spend; handle crash before first score; allow only declared output drift; pin code/config on continuation/recovery |
| Preflight / M9 | No protected query/qrel reads in preflight; restore inherited 120 GB space requirement; resolve M9's legacy bridge and ratification before close-out |
| Review and budgets | Rehearse the real caller; clean independent reviews; include conditional reserved cost in M13 allocation; register paired M9/nano row before either final run |
| Cost frontier | Same-machine zero/bge-small/nano serving costs and complete system/index costs; Edge fusion still unmeasured |

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
