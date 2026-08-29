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

### B-leg noise floor chains (2026-08-29) — full B→A chains varying only the seed

| m8nfb-seed1-b | `work/runs/m8nfb-seed1-b.json` | 0.4944 | ok |
| m8nfb-seed1-a | `work/runs/m8nfb-seed1-a.json` | 0.5103 | ok |
| m8nfb-seed2-b | `work/runs/m8nfb-seed2-b.json` | 0.4949 | ok |
| m8nfb-seed2-a | `work/runs/m8nfb-seed2-a.json` | 0.5113 | ok |

### B3 pool-scaling arms (2026-08-29) — nested real-pair fractions at fixed compute

| m8b3-p025-s0-smoke | `work/runs/m8b3-p025-s0-smoke.json` | 0.5011 | ok |
| m8b3-p025-s0 | `work/runs/m8b3-p025-s0.json` | 0.5098 | ok |
| m8b3-p025-s1 | `work/runs/m8b3-p025-s1.json` | 0.5087 | ok |
| m8b3-p025-s2 | `work/runs/m8b3-p025-s2.json` | 0.5102 | ok |
| m8b3-p050-s0 | `work/runs/m8b3-p050-s0.json` | 0.5104 | ok |
| m8b3-p050-s1 | `work/runs/m8b3-p050-s1.json` | 0.5095 | ok |
| m8b3-p050-s2 | `work/runs/m8b3-p050-s2.json` | 0.5110 | ok |
| m8b3-p075-s0 | `work/runs/m8b3-p075-s0.json` | 0.5121 | ok |
| m8b3-p075-s1 | `work/runs/m8b3-p075-s1.json` | 0.5119 | ok |
| m8b3-p075-s2 | `work/runs/m8b3-p075-s2.json` | 0.5140 | ok |

### M8 crossed BxA seed grid (NF), 2026-08-29

| m8nfx-b1a0 | `work/runs/m8nfx-b1a0.json` | — | FAILED — BrokenPipeError: [Errno 32] Broken pipe |
| m8nfx-b1a0-smoke | `work/runs/m8nfx-b1a0-smoke.json` | 0.4970 | ok |
| m8nfx-b1a2-smoke | `work/runs/m8nfx-b1a2-smoke.json` | 0.4974 | ok |
| m8nfx-b2a0-smoke | `work/runs/m8nfx-b2a0-smoke.json` | 0.4996 | ok |
| m8nfx-b2a1-smoke | `work/runs/m8nfx-b2a1-smoke.json` | 0.4986 | ok |
| m8nfx-b1a0 | `work/runs/m8nfx-b1a0.json` | 0.5088 | ok |
| m8nfx-b1a2 | `work/runs/m8nfx-b1a2.json` | 0.5103 | ok |
| m8nfx-b2a0 | `work/runs/m8nfx-b2a0.json` | 0.5094 | ok |
| m8nfx-b2a1 | `work/runs/m8nfx-b2a1.json` | 0.5107 | ok |

### E14-HEAD smoke (2026-08-29)

| m8e14-lad-lin-lr1e3-smoke | `work/runs/m8e14-lad-lin-lr1e3-smoke.json` | -0.3275 | ok |
| m8e14-r0n-s0-smoke | `work/runs/m8e14-r0n-s0-smoke.json` | 0.5012 | ok |
| m8e14-lin-s0-smoke | `work/runs/m8e14-lin-s0-smoke.json` | 0.4974 | ok |
| m8e14-r0n-s0-smoke | `work/runs/m8e14-r0n-s0-smoke.json` | 0.5012 | ok |

### E14-HEAD lr ladder, VOID (see LEDGER 15, 2026-08-29): mlp seeding defect, lin statistic replaced

| m8e14-lad-lin-lr3e4 | `work/runs/m8e14-lad-lin-lr3e4.json` | -0.2434 | ok |
| m8e14-lad-lin-lr1e3 | `work/runs/m8e14-lad-lin-lr1e3.json` | -0.2404 | ok |
| m8e14-lad-lin-lr3e3 | `work/runs/m8e14-lad-lin-lr3e3.json` | -0.2430 | ok |
| m8e14-lad-mlp-lr3e4 | `work/runs/m8e14-lad-mlp-lr3e4.json` | -0.2427 | ok |
| m8e14-lad-mlp-lr1e3 | `work/runs/m8e14-lad-mlp-lr1e3.json` | -0.2386 | ok |
| m8e14-lad-mlp-lr3e3 | `work/runs/m8e14-lad-mlp-lr3e3.json` | -0.2498 | ok |

## E14-HEAD step-adequacy (2026-08-29) — the gate fired on the PRIMARY

Two budgets per head at the pre-registered lr 1e-3, tuning seed 3, each fully annealed under its
own schedule; read on the repaired training-holdout statistic only (LEDGER §15).

| head | holdout @1250 | @2500 | @5000 | within-2500 | from doubling | ratio | verdict |
|---|---|---|---|---|---|---|---|
| **lin** (primary) | −0.14024 | −0.12607 | −0.12112 | 0.01417 | 0.00495 | **0.350** | **OPTIMIZATION-INADEQUATE** |
| mlp (control) | −0.14307 | −0.12328 | −0.11892 | 0.01979 | 0.00436 | 0.220 | ADEQUATE |

**Consequence, as pre-registered:** a null on LIN's endpoints reports UNINFORMATIVE, not a method
null. A positive is unaffected — the gate can never overturn an arm that reached the bar.

**Two honest qualifications, neither of which reinterprets the rule after the fact.** (1) Both
heads gain a SIMILAR ABSOLUTE amount from doubling (0.00495 vs 0.00436); LIN fails on the
denominator — it improved less over 1250→2500 — not because more steps help it more. (2) The gate
is ONE arm per budget with no replication and the holdout statistic has no measured floor, so
0.350 against 0.220 across a 0.25 line is not a resolved difference. The rule was applied exactly
as written; these are disclosures, not grounds for re-reading it.

**What would change it:** a reported arm set at 5,000 steps, with R0N re-run at 5,000 as its paired
comparator — the noise floor and the frozen recipe are both at 2,500, so the budget cannot be
changed for one arm alone. That is ~9 further training arms and ~9 further scoring passes, and it
is a spend decision, not a free follow-up.

## E14-HEAD — COMPLETE, 2026-08-29. NO SURVIVOR: the cheap re-shaping HARMS.

Bar +0.0040 on BOTH scalars vs `R0N`, 3 paired seeds, int8/sqrt. `results/m8_e14_head.json`.

| treatment | DENSE (out-of-domain macro) | FUSED (4-comp, frozen operator) | registered verdict |
|---|---|---|---|
| **LIN** (primary, 1.05M) | **−0.02441** | −0.00236 | OPTIMIZATION-INADEQUATE |
| **MLP** (control, 4.2M) | **−0.02932** | −0.00417 | NULL |

Per-seed dense gains: LIN [−0.02447, −0.02478, −0.02397]; MLP [−0.02798, −0.02865, −0.03132].
Not "fails to clear" — **~6× the bar in the wrong direction**, all six arms agreeing in sign.

**THE PATCH STACK IS A MEASURED NULL, which is what licenses reading the rest as a result.**
`R0N` vs the existing `R0` arms: dense **−0.00001** (σ_A 0.00106), fused **−0.000015** (σ 0.00039).
Four rebindings, a lazy proxy over 1.92M document rows per arm, and renormalized cached vectors
introduce no endpoint artifact whatsoever. The review's BLOCKER 2 was right to demand `R0N` as the
comparator on principle; empirically the precaution cost nothing and measured zero.

**THE `lin` LABEL, STATED CORRECTLY (revised after review; the first version overreached).**
**LIN is a strong negative result for the registered 2,500-step configuration, but remains
OPTIMIZATION-INADEQUATE for the method-level question. MLP met the registered adequacy heuristic
and was also harmful. Together these observations make a generic undertraining explanation less
plausible, but they do not resolve LIN at an adequate budget.** The earlier claim that the evidence
"disqualifies" the gate is **withdrawn**: MLP is a different architecture, its adequacy came from
separate tuning-seed holdout-reduced arms rather than from the reported arms, and MLP harming more
does not establish that a longer-trained LIN could not recover. Reporting the registered label and
setting contrary evidence beside it is honest; declaring the label void is not — especially since
this same section already discloses that the adequacy comparison is one unreplicated arm per budget
and that 0.350 vs 0.220 is unresolved.

**A factual correction to the earlier characterisation:** the endpoint did not "move steadily away".
The in-training out-of-domain contrasts vs R0N were LIN −0.0210 / −0.0268 / −0.0264 and MLP
−0.0309 / −0.0359 / −0.0309 at 500 / 1,500 / 2,500 steps — **early persistent harm, not monotone
divergence**. The holdout and the endpoint also differ in seed, pool, schedule, precision and
purpose, so the "training objective improved while the endpoint worsened" framing compares two
things that were never the same measurement.

**The disciplined resolution, if it is ever wanted:** a 5,000-step reported LIN set with a paired
5,000-step R0N.

**THE MECHANISM CONTROL IS THE MOST USEFUL NUMBER HERE, and it is not a flat null.** Per treatment,
mean over 3 seeds, on the two dense components:

| treatment | vs BAG queries | vs TEACHER queries | bag-specific (bag − teacher) |
|---|---|---|---|
| LIN | −0.02183 | −0.03095 | **+0.00912** |
| MLP | −0.02744 | −0.03493 | **+0.00749** |

**CORRECTED after adversarial review (2026-08-29) — the first write-up over-read this.** What the
data establish: across all seeds and both components, applying the trained head reduced bag-query
nDCG LESS than frozen-teacher-query nDCG (LIN +0.0091, MLP +0.0075 difference of differences; all
twelve treatment × seed × component values positive). That is **descriptive evidence consistent
with relative alignment toward the co-trained bag representation.** It does **not** show an
absolute bag benefit — both absolute bag gains are NEGATIVE — it does not identify bag
reachability, and it does not demonstrate information destruction. The earlier claim that the head
"buys bag-reachability only by destroying information" is **withdrawn**: a poorer match to the
teacher's query geometry is not evidence of information loss, and the linear residual map may well
remain full-rank.

**What would make it credible, and why it is not claimed today:** at n=3 the smallest exact
one-sided sign-test p per treatment is 0.125, and the four cells are NOT independent — their
covariance should be exploited by a paired PER-QUERY difference of differences, which
`e14_score.py` currently discards by reducing each cell to a component mean before saving. The
existing heads and tables can be re-scored for that without retraining; more seeds would need
retraining.

**Bearing on M9 and the paired release** (§15, Dylan's 2026-08-29 rulings): teacher-style queries
lose MORE than bag queries (−0.031 vs −0.022). A document transform co-trained with a bag taxes a
transformer query path HARDER than it taxes the bag. Moot for this head, which does not ship;
**direct evidence for `E14-LORA`'s still-unwritten bar**, and the first measurement on whether a
shared document side is free for the pair. It is not.

**Fusion absorbs ~10× of the dense degradation** (−0.0244 dense → −0.0024 fused under the frozen
convex operator, param 0.8). An independent read on how much of the fused system's quality is
carried by BM25 rather than by the table, corroborating M7's fusion finding from the other
direction. Carried forward: dense-side regressions are cheap in the fused product, and dense-side
GAINS are presumably discounted just as heavily.

**Scope, unchanged from the registration and binding on how this is written up:** a null here is
WEAK evidence about `E14-LORA` and may NEVER be written as closing E14. What it removes is the
strongest argument FOR buying the LoRA. Both dense components are CQADupStack forums, so this is a
CQADupStack-family result.

Cost: 13 training arms (~2.7 h), 9 dense passes (1,450 s), 9 fused passes, 9 mechanism passes.

## E10-REMEDY — ran 2026-08-29. PROCEED, pending review.

`results/m8_lotte_remedy.json`. All seven surviving slices pass remedy + zero-tolerance re-screen.

| slice | docs → after | removed | queries → after | removed | re-screen |
|---|---|---|---|---|---|
| writing/dev | 277,072 → 277,049 | 23 | 2,003 → 1,988 | 15 | 0 / 0 |
| recreation/dev | 263,025 → 263,000 | 25 | 2,002 → 1,994 | 8 | 0 / 0 |
| recreation/test | 166,975 → 166,975 | **0** | 2,002 → 1,990 | 12 | 0 / 0 |
| science/dev | 343,642 → 343,634 | 8 | 2,013 → 2,002 | 11 | 0 / 0 |
| technology/dev | 1,276,222 → 1,276,195 | 27 | 2,003 → 1,993 | 10 | 0 / 0 |
| lifestyle/dev | 268,893 → 268,890 | 3 | 2,076 → 2,074 | 2 | 0 / 0 |
| lifestyle/test | 119,461 → 119,458 | 3 | 2,002 → 1,993 | 9 | 0 / 0 |

**14,034 surviving queries — the pre-registered total, and every per-slice count matches §15's
figure exactly.** That is evidence the remedy is deterministic and did what was specified.

**AN ASYMMETRY THAT IS NOT YET EXPLAINED, and the shadow is NOT pinned until it is.** The document
screen removed **89 of 2,715,290** documents (0.003%) with one slice removing ZERO, while the query
screen removed **67 of 14,101** (0.5%) — over a hundred times the rate, against the same protected
content. Either that has a mechanism worth stating, or the document screen is materially weaker
than the query screen and 0.003% measures the screen rather than the corpus. Related and worse:
every re-screen returned exactly 0/0, and a re-screen that removes the items it would flag may not
be capable of returning anything else — CODEMAP pitfalls 17 and 19's family.

**Therefore NOT DONE:** `freeze_lotte.py pin`, the `paths_guard` partition entry, feeding the
surviving queries into the filter's index, and the fit-list regeneration are all deliberately NOT
run. Pinning the shadow IS trusting it, and a shadow that is quietly still contaminated is worse
than no shadow — it gives false reassurance immediately before a one-shot access. Adversarial
review briefed on exactly these two questions; disposition to follow.
