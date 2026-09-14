# M13 first scheduled quality evaluation

Evaluation: `mid1041666`, reported step **1,041,666**, approximately **33.3M examples**.

Metric: nDCG@10 on the admitted COV surface; units averaged within families, then four families weighted equally.

| Family | M13 | Teacher | Teacher retention |
|---|---:|---:|---:|
| BRIGHT | 0.184328 | 0.219125 | 84.12% |
| consumer-health | 0.713016 | 0.750712 | 94.98% |
| finance | 0.325429 | 0.372625 | 87.33% |
| legal | 0.856656 | 0.884456 | 96.86% |
| **Overall** | **0.519857** | **0.556730** | **93.38%** |

This is the first quality point; there is no earlier M13 quality evaluation for a trend comparison. Training resumed afterward. COV was used in recipe selection, so this is development evidence, not fresh final validation. It does not establish superiority over M9 on a different suite.

Artifact: `results/m13_cov_mid1041666.json`, copied byte-for-byte from the training host and verified against its remote SHA256:
`1f8627196f45bd9cd9feadaa559c93dd1743223327c55d80288445366be9f689`.

Teacher baseline: `results/m10_cov_teacher_ceiling.json`. Execution commit: `ee1bd6617c4fec7f4c97c73e7c6eed52f66ff50a`. No DEV-6, final six-set, reserved or LoTTE evaluation was performed for this checkpoint.

## Frozen M9 comparison on the same COV suite

Post-hoc CPU-only diagnostic completed in130.1seconds. Current M13 uses the published
33.3M-example midpoint evaluation; M9 uses its hash-verified final checkpoint at
step457265. Same admitted COV queries, teacher document embeddings, scorer and equal-family
macro. No extra Runpod charge; active training continued throughout.

| Family | Frozen M9 | Current M13 | M13 minus M9 |
|---|---:|---:|---:|
| BRIGHT | 0.160192 | 0.184328 | +0.024136 |
| consumer-health | 0.606373 | 0.713016 | +0.106643 |
| finance | 0.246139 | 0.325429 | +0.079290 |
| legal | 0.791242 | 0.856656 | +0.065414 |
| **Overall** | **0.450986** | **0.519857** | **+0.068871** |

Teacher-score retention: M9 **81.01%**, current M13 **93.38%**. All four family
point estimates improved. This supports continued training; no stopping rule changed.

Caveats: COV informed M13 selection, so this is not unrelated or fresh validation.
M9 predates the COV training-data rescreen; overlap protection differs. M9 is a final
checkpoint and M13 is intermediate with different training dose/recipe. M9 was evaluated
on CPU fp32 versus the current CUDA evaluation; this is not a precision-parity test.
No confidence interval is claimed, and this is not a causal recipe-effect estimate.

Full diagnostic: `results/m13_m9_cov_diagnostic.json`. Plan: `m13/COV_DIAGNOSTIC.json`.
Code: `scripts/m13_m9_cov_diagnostic.py`. The initial attempt failed before scoring
because it looked for caches in the worktree; its log/failure receipt are preserved.
The successful attempt used the existing shared cache directory after integrity checks.

## Cycle 1 complete — approximately 66.7M examples

| Family | First midpoint | Cycle 1 end | Change |
|---|---:|---:|---:|
| BRIGHT | 0.184328 | 0.187885 | +0.003558 |
| consumer-health | 0.713016 | 0.717428 | +0.004412 |
| finance | 0.325429 | 0.335703 | +0.010274 |
| legal | 0.856656 | 0.861085 | +0.004429 |
| **Overall** | **0.519857** | **0.525525** | **+0.005668** |

Teacher retention: **94.40%**. All four family point estimates improved. Midpoint and cycle-end readings occur at different learning-rate phases; this adjacent-point change is descriptive, not a same-phase regression test. Training entered cycle2.

Artifact: `results/m13_cov_cycle1.json`, remote/local SHA256 `9658bbdcc41a86982de6239b8c04db2626ebde618c1e3037a9df7dbbed8c30e2`.

## User-requested CPU read at 79,539,107 examples

Pinned rolling checkpoint step2,485,600; CPU fp32, four threads,
full same COV query/candidate sets as the frozenM9 CPU baseline. Completed in
131.1seconds with no additional Runpod charge.

| Family | Frozen M9 CPU | Current M13 CPU | Difference |
|---|---:|---:|---:|
| BRIGHT | 0.160192 | 0.178182 | +0.017991 |
| consumer-health | 0.606373 | 0.711345 | +0.104971 |
| finance | 0.246139 | 0.327286 | +0.081148 |
| legal | 0.791242 | 0.855581 | +0.064339 |
| **Overall** | **0.450986** | **0.518099** | **+0.067112** |

Teacher retention **93.06%**. All four family point estimates exceed M9.

Current score is **-0.007427** versus the scheduled
cycle1end0.525525. This dip is disclosed, not hidden: current checkpoint is
early in cycle2 after the learning-rate reset, whereas the comparison is an annealed
cycle-end checkpoint; current evaluation uses CPU fp32, previous used CUDA. Neither a
same-phase comparison nor a calibrated precision comparison. No automatic stopping rule
changed and no inference of overfitting from this one reading. Existing COV selection
and M9/M13 training-history caveats above still apply.

Result: `results/m13_current_cov_cpu.json`. Plan: `m13/COV_CURRENT_CPU.json`.
Checkpoint SHA256: `f350a0a7babe8dab059876e03cbf1fc9c0c9bc09b8e194524ed4ea6481bcd2df`. Checkpoint copy is retained locally
under `work/m13-side-eval/current_cpu_checkpoint.pt`, separate from rolling-backup pruning.

## Second scheduled midpoint — approximately100M examples

| Family | First midpoint | Second midpoint | Change |
|---|---:|---:|---:|
| BRIGHT | 0.184328 | 0.186623 | +0.002295 |
| consumer-health | 0.713016 | 0.718283 | +0.005266 |
| finance | 0.325429 | 0.331493 | +0.006065 |
| legal | 0.856656 | 0.855619 | -0.001037 |
| **Overall** | **0.519857** | **0.523005** | **+0.003147** |

Teacher retention93.94%. Same-phase midpoint improved; no regression strike. Versus the previous annealed cycle end, change-0.002521; different learning-rate phases. Training continues.

Raw result: `results/m13_cov_mid3124999.json`. Remote/local SHA256 `549ad9206c8eb782feca82993a7692b8eb2e9bb2d1fa9fd5c15083fc7faf478d` verified.

## Cycle 2 complete — approximately133.3M examples

Evaluation finished2026-09-14 03:55:02UTC (September13,11:55p.m.Eastern).

| Family | Cycle 1 end | Cycle 2 end | Change |
|---|---:|---:|---:|
| BRIGHT | 0.187885 | 0.192275 | +0.004389 |
| consumer-health | 0.717428 | 0.720571 | +0.003143 |
| finance | 0.335703 | 0.338302 | +0.002599 |
| legal | 0.861085 | 0.860841 | -0.000244 |
| **Overall** | **0.525525** | **0.527997** | **+0.002472** |

Teacher retention94.84%. New best scheduled macro; three families improved, legal slightly declined. No regression strike. Training entered final cycle3. Raw result `results/m13_cov_cycle2.json`, remote/local SHA256 `5732eceff2ba231c528e1c4f33bd6c0f93ad1cb2ea6c5ebd3e8bdef0b385c182` verified.

## Second CPU comparison — 158.28M examples

User-requested CPU evaluation passed in 131.4 seconds using the latest verified
off-host backup, pinned separately from rolling-backup pruning. Existing M9 CPU
baseline reused; no paid GPU work or active-training changes.

| Family | M9 CPU | M13 CPU 79.54M | M13 CPU 158.28M |
|---|---:|---:|---:|
| BRIGHT | 0.160192 | 0.178182 | 0.186505 |
| consumer-health | 0.606373 | 0.711345 | 0.715619 |
| finance | 0.246139 | 0.327286 | 0.335492 |
| legal | 0.791242 | 0.855581 | 0.859829 |
| **Overall** | **0.450986** | **0.518099** | **0.524361** |

Teacher retention 94.19%; gain versus prior CPU check
+0.006263; versus M9 +0.073375. All four families improved versus both CPU baselines.

This is below the cycle-two CUDA end score 0.527997 by 0.003636. Different
learning-rate phase and CPU/CUDA execution prevent treating that as a controlled
regression comparison. CPU snapshots also differ in cycle position. No stopping
rule changed. COV is selection-informed, not fresh validation; M9/M13 recipe and
dose differences prevent causal attribution.

Result: `results/m13_cov_cpu_round2.json`. Plan: `m13/COV_CPU_ROUND2.json`.
Uses unchanged `scripts/m13_current_cov_cpu.py`, importing the module and setting
`PLAN` and `OUTPUT` to those paths before calling `main()`, under `timeout 1800`
with `.venv/bin/python`. Checkpoint: `work/m13-side-eval/cpu_round2_checkpoint.pt`.
