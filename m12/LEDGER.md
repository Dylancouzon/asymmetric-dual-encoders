# M12 — registration

Written **before any M12 number exists**. Mandate `instructions-m12.md`; cut avenues `m12/EXPLORED.md`.
Pushed before scoring; the remote timestamp is the external witness (M7 convention).

## Scope

Dev only, four text-backed components (`nq-250k`, `hotpotqa`, `cqadup-programmers`,
`cqadup-physics`), depth 1000, int8 release table `work/runs/p35w-2m-s2500.release.npz`
(sha must match `m7/FREEZE.json`). Descriptive: does **not** replace the published 0.4911, and no
M12 result may claim to reproduce a six-set number. The six, the reserved four and LoTTE are not
touched. A **cloud** finding — edge fusion is unbuilt and uncosted, reported as a limitation note.

## Reproduction gate (runs first, gates everything)

Recompute all **21** M7 grid points on M12's own runs. Gate: `max |Δ| ≤ 1e-4` against
`m7/FREEZE.json`'s `fusion.grid`. Fail ⇒ stop and report a reproducibility finding, score no new
operator. The **recomputed** convex0 `w=0.8` macro is the comparator `C`; the frozen literal
0.5726634997854769 is reported beside it.

## Bar

`B = C − 0.004`. The 0.004 is `max(0.0040, 2×floor)` from `results/m8_noise_floor_fused.json`, a
**training-seed** floor measured with the operator frozen. Borrowed here as "would this survive a
retrained table", not as measurement precision. It carries no fitting allowance.

## Operators, grids pinned

| tier | operator | candidates |
|---|---|---|
| 1 | DBSF (parameter-free) | 0 |
| 1 | RRF over `k_q ∈ {1,2,3,4,6,11,21,31,61,101}` | 10 |
| 2 | weighted RRF, `k_q ∈ {2,6,11,61}` × `(w_d,w_b) ∈ {(1,1),(2,1),(3,1),(4,1),(1,2),(1,3)}` | 24 |

Qdrant parity: RRF `= Σ 1/((posᵢ+1)/wᵢ + k_q − 1)`, pos 0-based, `w ≤ 0 → 0.0`; `k_q = 0` excluded
(divides by zero, unvalidated server-side). DBSF `= Σ (s − (μ−3σ))/(6σ)` per prefetch, **sample** SD
(ddof=1), no clamp, **0.5** for singleton or constant lists, statistics over the returned list at
depth 1000. **A document absent from a prefetch contributes 0**, which is not the bottom of the
normalised range — registered choice, matching Qdrant's sum-merge.

## Decision rule

| tier | operator | passes if |
|---|---|---|
| 1 | DBSF | dev macro ≥ B |
| 1 | RRF over `k` | dev macro ≥ B |
| 2 | weighted RRF | macro of the two held-out halves ≥ B, split-half on `int(sha256(qid)) % 2` (fit on even, score on odd; then the reverse) |

**A shipping operator matches** iff any row passes; recommend the highest-strength passing row
(DBSF > RRF-`k` > weighted RRF — fewest fitted parameters wins). **Tier 2 is attempted iff neither
Tier 1 row passes**; the skip is recorded here when it happens.

## Statistics

Paired bootstrap on each operator-vs-convex0 difference: resample **within component**, then
macro-of-means. `B = 10000`, seed `12`, percentile 95% CI. A CI on a fitted winner is
post-selection and optimistic — reported as such.

## Also reported, not gated

Depth curve `d ∈ {10,50,100,1000}` for convex0 and each tier's winner (truncation on the same
stable sort `fusion.rrf` uses); per-component qid∈doc_ids collision counts (both runs drop
self-hits after retrieval, Qdrant does not).

## Log

- 2026-09-04 — registered, before any M12 number. Nothing scored yet.
- 2026-09-04 — Tier 1 executed. Gate passed (max |delta| 1.11e-16 over 21 points; comparator
  identical to the frozen literal). **NO MATCH at depth 1000**: DBSF 0.5580 (-0.0146), RRF best
  k_q=3 0.5535 (-0.0192), both CIs excluding 0 and the bar. Tier 2 therefore REQUIRED and is being
  run. `m12/tier1.json`.
- 2026-09-04 — Tier 2 executed: weighted RRF held-out 0.5601 vs bar 0.5689, FAIL. Both halves
  independently selected `k_q=2 w=(2,1)`; held-out (0.5601) ~ dev-fitted (0.5598), so no material
  overfit. **M12 RESULT: NO MATCH.** `m12/tier2.json`, findings `m12/FINDINGS.md`.
- 2026-09-04 — verification of the two results the review was to attack, run directly after the
  external review stalled and was killed: (a) `bootstrap()` confirmed genuinely paired — one index
  vector per component applied to both systems; (b) the depth-10 inversion is **not** a degenerate
  artifact — DBSF's singleton/zero-variance branch fires **0.00%** at every depth (10/50/100/1000)
  with median list length exactly the depth, and all macros reproduce. Mechanism is real: DBSF
  standardises per list, convex0's `s/max(s)` flattens a bunched head. `logs/m12_depth_check.log`.
