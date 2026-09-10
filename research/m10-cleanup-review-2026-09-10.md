# M10 cleanup review — 2026-09-10

Read-only agent audit of the screen/final registries, recipe/LoTTE locks, runner, trainer,
`m9src/final9.py` and archived half-B findings. Access report: only those eight named files;
no protected queries/qrels or new quality outputs. Findings independently checked by the lead.

| Finding | Disposition |
|---|---|
| Arm CLI requires a nonterminal record for resume but never writes one | Fixed durable running receipt; caught failures remain terminal; SIGKILL/restart through the real caller matches uninterrupted training |
| Alleged minimum nDCG change is false | Correct rationale only; leave decision constants unchanged; preserve prior reasoning in git/archive |
| LoTTE chooses the build recipe | Keep its applicable gate before the full build; metric/slice manifest still pending |
| Build controller/final scorer absent | Explicit M13 prerequisites, not implied by unit-test counts |
| Freeze wording overclaims unchanged registration | Preserve dated unrun-family amendments; do not change computed contrast inputs |

## nDCG counterexample (synthetic arithmetic, no evaluation data)

With binary relevance and at least ten relevant documents, IDCG@10 is
`sum(1/log2(r+1) for r in range(1,11))`.
Relevant ranks `(2,4,6,8,10)` yield **0.44510108735750964**;
`(2,5,6,7,8)` yield **0.44519897452998813**.
Their difference is **0.00009788717247849466**, below 0.0003. Multiple swaps can cancel;
equal-relevance permutations can also change rankings without changing nDCG at all.
Thus both proposed “minimum quantum” claims are withdrawn. A stringent re-encoding bridge may
still be operationally fragile, but its feasibility must be measured before protected execution.

Final checks: 469 M10 tests, 16 M9 statistics tests and 31 fusion checks passed. No screen
constant or result changed; the final-registry edit corrects bridge rationale only.
