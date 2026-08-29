# M8 runs

One row per run. Detail belongs in the run JSON; never restate a number a `results/m8_*.json`
already holds. Written by hand until a sweep driver exists.

| run | what | outcome | artifact |
|---|---|---|---|
| `m8_power` | joint power simulation of the full ship rule | macro SE 0.00209, MDE 0.0068, P(ship) 0.67/0.57/0.15/0.002/0.46 across the five registered scenarios | `results/m8_power.json` |
| `m8_retention_decomposition` | descriptive re-read of M7's final run: what is the query-side loss made of? | the short-query premise fails within datasets; **fragmentation** is the consistent channel (+0.050 gap per +1.0 subwords/word, t=4.6) | `results/m8_retention_decomposition.json` |
| `m8_schedule` | ridge control timing + serial GPU/RAM/disk plan | reserved-4 pre-encode 20.7 GB fp16 per system | `results/m8_schedule.json` |
| `S0` (smoke, 2K docs/slice) | LoTTE overlap screen | every slice DROPs — reproduced at full scale | `results/m8_lotte_overlap.SMOKE.json` |
| `m8nf-seed0` | noise-floor seed-0 arm: the M7 candidate's exact config, retrained | **proxy macro 0.5105689103506673 — byte-identical to `p35w-2m-s2500`'s `final_macro`.** The floor's frame IS the shipped artifact's frame, and the harness reproduced a 2,500-step run exactly (M7 measured replay noise at ~4.5e-6 and expected non-bit-identical GPU reductions; this run was exact) | `work/runs/m8nf-seed0.json` |
| `blockcg` (smoke) | block-CG vs the direct ridge solve | agreement 2.4e-8 relative Frobenius | — |
| `B7` (full, registered) | block-CG vocabulary curve, Zipf + Jacobi | **PASS.** 30,522: 26 its / 5.2 s / 3.77 GB RSS · **65,536: 51 its / 10.4 s / 4.42 GB** (dense fp64 Gram would be 34.4 GB) · 131,072: 68 its / 16.6 s / 5.72 GB (137.4 GB). Agrees with the direct solve to 4.6e-7. Rows reached at 128K: 84.4% | `results/m8_b7_solver.json` |
| `blockcg` (conditioning) | Zipf vs uniform token draw, with and without Jacobi | **unpreconditioned CG does not converge in 1,500 iterations on Zipfian data (5.9e-4); Jacobi converges in 61 (7.0e-7).** Uniform draws converge in 131 unpreconditioned — an easy problem that would have produced a wrong feasibility PASS | `m8/CODEMAP.md` pitfall 8 |
| `S0` (full) | LoTTE overlap screen, 5.25M docs, 19 min | **all ten slices DROP; E10 reopens with Dylan.** 3 on community intersection with the protected sets, 7 on query leakage (0.1–0.75%). Exact matches concentrate in the community-overlapping slices | `results/m8_lotte_overlap.json` |
| `filter` | protected-query fingerprint inventory | 80,954 queries over four partitions (six 3,727 · dev 12,772 · reserved-4 9,335 · M9-reserve 55,120); 4.22M gram keys | `results/m8_protected_filter.json` |
| `shadow_alternatives` | counts for the 8 unused CQADupStack subforums | 323,488 docs / 8,961 queries, licence already cleared — a third option for E10 | `results/m8_shadow_alternatives.json` |
| `m8_fragmentation_attribution` | which words carry the fragmentation cost | hyphenated compounds, date strings, post-2018 named entities; 6/6 sign-consistent | `results/m8_fragmentation_attribution.json` |
