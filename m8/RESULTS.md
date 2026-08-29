# M8 runs

One row per run. Detail belongs in the run JSON; never restate a number a `results/m8_*.json`
already holds. Written by hand until a sweep driver exists.

| run | what | outcome | artifact |
|---|---|---|---|
| `m8_power` | joint power simulation of the full ship rule | macro SE 0.00209, MDE 0.0068, **P(ship) 0.84/0.80/0.21/0.002/0.57** (an earlier run read 0.67/0.57/0.15/0.002/0.46 from stale guard constants and is superseded) | `results/m8_power.json` |
| `m8_retention_decomposition` | descriptive re-read of M7's final run: what is the query-side loss made of? | the short-query premise fails within datasets; **fragmentation** is the consistent channel (+0.050 gap per +1.0 subwords/word, t=4.6) | `results/m8_retention_decomposition.json` |
| `m8_schedule` | ridge control timing + serial GPU/RAM/disk plan | reserved-4 pre-encode 20.7 GB fp16 per system | `results/m8_schedule.json` |
| `S0` (smoke, 2K docs/slice) | LoTTE overlap screen | every slice DROPs — reproduced at full scale | `results/m8_lotte_overlap.SMOKE.json` |
| **`B6-pre`** | doc-side ONNX fuse gate (E3's hard condition) | **PASS.** One file, 3,415 nodes, **zero custom-domain ops**, parity min-cosine 0.99999994 / max-abs 2.05e-07. D1 survives | `results/m8_b6_pre.json` |
| **`T1`** | teacher screen, 3 candidates, clean fit list, CG frame | **NO SWAP.** stella 0.3438 · granite-r2 0.2915 (−0.052 [−0.066, −0.039]) · gte-modernbert 0.2349 (−0.109 [−0.123, −0.094]). All optima interior. Condition 1 fails for both, so 2–4 never arise | `results/m8_t1_decision.json` |
| **fit list** | regenerate TRAIN query texts through the CURRENT 80,954-query protected index | 338,076 derived (M7's list was 349,934, dumped from an older kept set) → **337,981 kept, 95 removed (0.028%)**, 64 exact / 31 near. Attribution to M9-reserve is **by difference across two runs** (0 removals against six+dev+reserved alone, 95 against the extended index), not a per-partition field in the artifact | `results/m8_trainq_manifest.json` |
| **`B17`** | in-domain oracle generalization, 957 fit queries, oracle λ | held-out **0.1999** (41.6% of the 0.4806 teacher), init-only floor 0.0174. **Registered ≤0.40 branch fired — but DISOWNED**: the same class on 350K general queries scores 0.3439, so this measured the fit-set size, not a class ceiling | `results/m8_b17_oracle.json` |
| **`B7` real-data precondition** | block CG vs direct on the REAL system, all four λ | **identical dev macro at every λ (|Δ|=0.0)**; argmax λ=1e-2 at 0.343924, reproducing M7's 0.3439 for stella. At 30,522 rows the DIRECT solve is faster at small λ | `results/m8_b7_realdata.json` |
| **`B2`** | KL-term degeneracy, both sides, 4,000 TRAIN queries, the recipe's own (seeded random) bank | **H2 CONFIRMED.** Teacher target median entropy **4.73e-07 nats**, p_max median 1.000000, 84.3% below 1e-4. Student (shipped M7 table) ranks the positive first in **99.75%** of queries, so **the KL term's own median value is 1.08e-07 nats** | `results/m8_b2_entropy.json` |
| **`NF` (fused floor)** | fused macro, frozen convex0 w=0.8, 3 seed arms | floor **0.00059–0.00066** (~3x tighter than dense); bars **0.0040**. Seed-0 at sqrt reproduces M7's fusion dev_macro 0.57266 | `results/m8_noise_floor_fused.json` |
| **`NF` (noise floor, dense)** | 3 seed arms + 2 step arms, full pinned dev suite, both precisions, both pooling rules | **floor 0.00095–0.00227; bars 0.0040 everywhere except fp16·mean worst-group/OOD (0.00454).** Step sensitivity −0.0009/+0.0015 | `results/m8_noise_floor.json` |
| `m8nf-seed0` | noise-floor seed-0 arm: the M7 candidate's exact config, retrained | proxy macro matches `p35w-2m-s2500` to 16 digits — but the **weights differ** (0.066% of `rows_fp16`, max 1 fp16 ULP): nDCG is rank-based and quantized the noise away. ONE replay seen through three rank-invariant lenses, not three validations; the frame is the shipped artifact's frame for every purpose a bar reads | `work/runs/m8nf-seed0.json` |
| `blockcg` (smoke) | block-CG vs the direct ridge solve | agreement 2.4e-8 relative Frobenius | — |
| `B7` (full, registered) | block-CG vocabulary curve, Zipf + Jacobi | **PASS.** 30,522: 26 its / 5.2 s / 3.77 GB RSS · **65,536: 51 its / 10.4 s / 4.42 GB** (dense fp64 Gram would be 34.4 GB) · 131,072: 68 its / 16.6 s / 5.72 GB (137.4 GB). Agrees with the direct solve to 4.6e-7. Rows reached at 128K: 84.4% | `results/m8_b7_solver.json` |
| `blockcg` (conditioning) | Zipf vs uniform token draw, with and without Jacobi | **unpreconditioned CG does not converge in 1,500 iterations on Zipfian data (5.9e-4); Jacobi converges in 61 (7.0e-7).** Uniform draws converge in 131 unpreconditioned — an easy problem that would have produced a wrong feasibility PASS | `m8/CODEMAP.md` pitfall 8 |
| `S0` (full) | LoTTE overlap screen, 5.25M docs, 19 min | **all ten slices DROP; E10 reopens with Dylan.** 3 on community intersection with the protected sets, 7 on query leakage (0.1–0.75%). Exact matches concentrate in the community-overlapping slices | `results/m8_lotte_overlap.json` |
| `filter` | protected-query fingerprint inventory | 80,954 queries over four partitions (six 3,727 · dev 12,772 · reserved-4 9,335 · M9-reserve 55,120); 4.22M gram keys | `results/m8_protected_filter.json` |
| `shadow_alternatives` | counts for the 8 unused CQADupStack subforums | 323,488 docs / 8,961 queries, licence already cleared — a third option for E10 | `results/m8_shadow_alternatives.json` |
| `m8_fragmentation_attribution` | which words carry the fragmentation cost | binary contrast **4/5 informative datasets positive, one-sided p=0.19 — NOT resolved**; the continuous slope is the instrument to quote. Words are a MIX (compounds, domain terms, ordinary English), not dominated by drifted vocabulary | `results/m8_fragmentation_attribution.json` |

## Training arms (the rows `sweep.py` appends, kept HERE)

`m7src/sweep.one` appends every run it makes to **`m7/RESULTS.md`** — M7's experiment
ledger. G3 forbids M8 altering M7's record, so those rows were reverted there and are
preserved here instead. The five FAILED rows are the teacher-mismatch the first smoke
caught (`M7_ENCODER` defaults to M7's pre-swap bge-base); the `-smoke` rows are 90-step
arms. Per-run detail is in `results/m7_run_m8nf-*.json`.

| run | artifact | dev proxy | verdict |
|---|---|---|---|
| m8nf-seed0 | `work/runs/m8nf-seed0.json` | — | FAILED — AssertionError: init 'run:p35b-2m' was trained against NovaSearch/stella_en_400M_v5 but the active encoder is BAAI/bge-base-en-v1.5 |
| m8nf-seed1 | `work/runs/m8nf-seed1.json` | — | FAILED — AssertionError: init 'run:p35b-2m' was trained against NovaSearch/stella_en_400M_v5 but the active encoder is BAAI/bge-base-en-v1.5 |
| m8nf-seed2 | `work/runs/m8nf-seed2.json` | — | FAILED — AssertionError: init 'run:p35b-2m' was trained against NovaSearch/stella_en_400M_v5 but the active encoder is BAAI/bge-base-en-v1.5 |
| m8nf-steps2250 | `work/runs/m8nf-steps2250.json` | — | FAILED — AssertionError: init 'run:p35b-2m' was trained against NovaSearch/stella_en_400M_v5 but the active encoder is BAAI/bge-base-en-v1.5 |
| m8nf-steps2750 | `work/runs/m8nf-steps2750.json` | — | FAILED — AssertionError: init 'run:p35b-2m' was trained against NovaSearch/stella_en_400M_v5 but the active encoder is BAAI/bge-base-en-v1.5 |
| m8nf-seed0-smoke | `work/runs/m8nf-seed0-smoke.json` | 0.5012 | ok |
| m8nf-seed1-smoke | `work/runs/m8nf-seed1-smoke.json` | 0.5013 | ok |
| m8nf-seed2-smoke | `work/runs/m8nf-seed2-smoke.json` | 0.5014 | ok |
| m8nf-steps2250-smoke | `work/runs/m8nf-steps2250-smoke.json` | 0.5012 | ok |
| m8nf-steps2750-smoke | `work/runs/m8nf-steps2750-smoke.json` | 0.5012 | ok |
| m8nf-seed0 | `work/runs/m8nf-seed0.json` | 0.5106 | ok |
| m8nf-seed1 | `work/runs/m8nf-seed1.json` | 0.5123 | ok |
| m8nf-seed2 | `work/runs/m8nf-seed2.json` | 0.5123 | ok |
| m8nf-steps2250 | `work/runs/m8nf-steps2250.json` | 0.5108 | ok |
| m8nf-steps2750 | `work/runs/m8nf-steps2750.json` | 0.5107 | ok |
