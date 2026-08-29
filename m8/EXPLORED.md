# M8 closed avenues

One row per closed avenue: what was tried, why it is closed, and what would reopen it. A closed
avenue with no reopening condition is a guess, not a finding.

| avenue | why closed | what would reopen it |
|---|---|---|
| **B11 / fusion complementarity** | NOT closed by E11, as the plan's v1 wrongly said — C1 is fused-vs-fused, so fusion is live for the PRIMARY leg. Closed instead by the frozen fusion operator's own no-routing rule, plus the optics of a query-dependent weight. | Registering a length-conditioned fusion family as THE fusion operator, before Stage R. After Stage R it is out for the milestone. |
| **Direct dense-Gram ridge at V ≥ 50K** | 20.3 GB fp64 at 50,368 rows against an 18 GB budget — the arithmetic that closed granite-r2 and gte-modernbert in M7. | **REOPENED 2026-08-29 and confirmed**: `m8src/blockcg.py` never forms the Gram; B7 solves 64K in 10 s / 4.4 GB and 128K in 17 s / 5.7 GB. The closure was an artifact of the solver, not of the problem. |
| **Unpreconditioned CG on a Zipfian Gram** | Does not converge in 1,500 iterations (5.9e-4). Real token frequencies span orders of magnitude, so the Gram's spectrum is extremely skewed. | Nothing — Jacobi preconditioning (61 iterations) is the fix and is in the solver. Recorded because a UNIFORM synthetic converged in 131 and would have produced a feasibility PASS that said nothing about the real problem. |
| **T1 in a single shared student frame** | M7's shared bag matrix works only because all ten registered encoders ship a byte-identical `bert-wordpiece-30522` vocabulary. None of T1's four challengers does. | Nothing: it is a fact about the candidates. The frame is now fixed *within* a tokenizer family and cross-family screens are labelled teacher-plus-tokenizer. |
| **LoTTE `search` queries** | Non-commercial-research-only, inherited from the GooAQ licence (paper Appendix D, quoted in LEDGER §2.3). | A licence change, or a ruling from Dylan that a non-commercial research measurement is acceptable for an internal shadow gate — the same class as M7's clean-stack-tax arm. |
| **PMC-OA** | E8: its unique value is duplicated by cleaner sources. | Only one condition: the genre probe showing a biomedical-specific gap. |
