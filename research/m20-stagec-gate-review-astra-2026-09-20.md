# Astra review — M20 stage C projection-gate arming fix (2026-09-20)

Model `gpt-6-astra`, reasoning effort xhigh, `--sandbox read-only`. Verdict **GO WITH CONDITIONS**,
no P0/P1 blockers on the patch itself.

Brief: `work/m20/logs/astra_stagec_gate_brief.md`.
Full transcript including every command the reviewer ran: `work/m20/logs/astra_stagec_gate_review.log`.

Access audit: no recursive search over `results/` or `work/`; no read of
`results/frozen_eval/untouched-*`, reserved qrels caches or `work/m9reserve`. The `qrels` strings
it saw are `qrels_source`/`qrels_revision` pointers inside the allowlisted
`m20/beir15_registry.json`, not protected content. Two files it wanted were outside the allowlist
and it reported them rather than reading them: `results/m20_stage_c_run.json` and
`results/m20_tower_rate_benchmark.json`.

Independent check worth recording: an AST diff against `HEAD`, with docstrings stripped, reported
`Changed existing executable definitions: []` and `Added definitions: ['batch_min_projection_rows']`.

---

**No essential defect found in the patch.** The diagnosis is supported, the change is minimal, and the current resume bias is immaterial. The gate remains a useful early-stop estimate, with weaker protection before its larger arming point.

1. **Diagnosis and arithmetic**

   The log confirms refusal after 134,473 newly encoded documents, before reaching MS MARCO. Extrapolating that long-document prefix’s 59 docs/s across the entire batch is a credible explanation for the false refusal.

   The arithmetic checks:

   - `1 + 0.1349 + 0.3308 = 1.4657`.
   - `23,744,806 × 92.2 = 2.1893 billion tokens`.
   - At 13,400 tokens/s: **45.38 hours Stella**, **66.52 hours across three towers**, equivalent to **145.34 Stella documents/s**.
   - The launcher reserves **17.057 hours** for download/scoring, leaving **102.943 hours initially** for encoding.

   Your arming table also reproduces:

   | Threshold | Cumulative rate | Time to threshold | Projected remaining encoding |
   |---|---:|---:|---:|
   | 100,000 | 59.0 docs/s | 0.47 h | 163.38 h |
   | 1,000,000 | 113.70 docs/s | 2.44 h | 82.59 h |
   | 2,000,000 | 137.53 docs/s | 4.04 h | 66.25 h |

   These are modeled thresholds; actual checks occur at completed shard boundaries.

   **The denominator is correct for the registered workload.** The registry breakdown sums exactly to 23,744,806. The encoder excludes all four reserved corpora, includes ten remaining CQADupStack forums, and includes Climate-FEVER’s separate 5,416,593 documents. There is no structural double-count or omission. [Volume registration](/home/dylan/asymetric-dual-encoders/m20/beir15_registry.json:612)

   The 66.52-hour figure remains an estimate: token throughput and SQuAD-derived tower ratios need not transfer perfectly between corpora. I verified their arithmetic, not the underlying measurements. The supplied eight token means omit 657,301 documents, so they do not independently establish the exact 92.2 mean; even assigning every omitted document 512 tokens yields **72.84 hours** under the same throughput and ratio assumptions.

2. **Protection after raising the threshold**

   Once armed, the refusal calculation is unchanged and runs after every new shard. A simulation advancing wall time refused the half-speed MS MARCO case at approximately **3.68 hours**.

   **2.4 hours is an expectation, not a maximum exposure.** With every modeled rate five times slower, arming takes approximately **12.4 hours**. At a constant 10 docs/s it takes approximately **28 hours**. At roughly 0.27–2.7 docs/s, the old threshold could arm within 102.3 hours while the new threshold cannot; the launcher’s hard timeout would stop the run instead.

   That is the newly admitted class: severely slow runs can consume substantially more time before projection refusal. For this monitored local continuation, the tradeoff is acceptable. It relies on the existing first-shard monitoring instruction as well as the hard timeout.

3. **Order, partial MS MARCO resume, and loading failures**

   The current registry order places arming inside MS MARCO. The threshold does **not** enforce that property.

   - Reordering or extensively skipping completed corpora changes the sampled distribution.
   - Partially completed MS MARCO still supplies the sample if enough unencoded rows remain; otherwise sampling proceeds into later corpora.
   - A corpus-loading exception propagates and stops execution; it is not silently skipped. [Encoding path](/home/dylan/asymetric-dual-encoders/m8src/pre_encode.py:273)

   Front-loading long documents makes the initial sample conservative, but does not guarantee every later prefix is conservative. After millions of short MS MARCO documents, later Climate-FEVER and Touché documents can reverse that bias. This limits the estimator’s guarantees; it does not invalidate this particular arming fix.

4. **The coming resume**

   With 134,473 rows already complete, **137,644** of the first six corpora remain. The first eligible MS MARCO shard boundary occurs after:

   - **1,037,644 new rows**, including **900,000 MS MARCO rows**;
   - approximately **2.085 encoding hours**;
   - a cumulative **138.25 docs/s**;
   - **67.84 hours** projected remaining.

   At that rate, counting the existing 134,473 rows as outstanding adds **0.270 hours—about 16 minutes**. That is immaterial against the stated margin.

   Skipping those rows also changes the sampled rate; the resumed sample is not the original batch prefix. The accepted debt therefore stays harmless here, but this calculation should not be generalized to a much larger future resume.

5. **Scope and existing results**

   This is appropriately small. Comparing executable definitions against `HEAD` showed **no changes to any existing function or class implementation**; the addition is the threshold helper plus its call-site argument.

   Existing shards are hash-checked and skipped before encoding. The change cannot rewrite an already-written vector, change the encoding contract, or alter any Stage A/B score. The reserved threshold remains 100,000. The caution in `FINDINGS.md` does not expose a concrete evidence risk from this isolated change. [Shard reuse](/home/dylan/asymetric-dual-encoders/m8src/pre_encode.py:304)

6. **Verification and tonight’s relaunch**

   All nine supplied checks passed using isolated definitions, without importing the guarded encoder module. Separate simulations advanced wall time and checked actual arming boundaries. No encoding, evaluation, full suite, protected reads, or file writes were performed; repository reads stayed within the allowlist.

   Two receipts outside the allowlist remain unverified: `results/m20_stage_c_run.json`, needed to establish **current remaining time**, and `results/m20_tower_rate_benchmark.json`, which supplies the gate’s exact runtime ratios.

**GO WITH CONDITIONS**

- Preserve the failed receipt under the launcher’s `m20_stage_c_run_attempt*.json` convention and inherit its original deadline. Recalculate remaining time; **102.3 hours was the failure-time budget**.
- Preserve both failed-run logs; the documented `> stage_c.log` command would truncate the controller log.
- Commit/push the reviewed change and preserve the generated receipts before relaunch; retain the existing first-shard monitoring for severe slowdown.
tokens used
80,842
**No essential defect found in the patch.** The diagnosis is supported, the change is minimal, and the current resume bias is immaterial. The gate remains a useful early-stop estimate, with weaker protection before its larger arming point.

1. **Diagnosis and arithmetic**

   The log confirms refusal after 134,473 newly encoded documents, before reaching MS MARCO. Extrapolating that long-document prefix’s 59 docs/s across the entire batch is a credible explanation for the false refusal.

   The arithmetic checks:

   - `1 + 0.1349 + 0.3308 = 1.4657`.
   - `23,744,806 × 92.2 = 2.1893 billion tokens`.
   - At 13,400 tokens/s: **45.38 hours Stella**, **66.52 hours across three towers**, equivalent to **145.34 Stella documents/s**.
   - The launcher reserves **17.057 hours** for download/scoring, leaving **102.943 hours initially** for encoding.

   Your arming table also reproduces:

   | Threshold | Cumulative rate | Time to threshold | Projected remaining encoding |
   |---|---:|---:|---:|
   | 100,000 | 59.0 docs/s | 0.47 h | 163.38 h |
   | 1,000,000 | 113.70 docs/s | 2.44 h | 82.59 h |
   | 2,000,000 | 137.53 docs/s | 4.04 h | 66.25 h |

   These are modeled thresholds; actual checks occur at completed shard boundaries.

   **The denominator is correct for the registered workload.** The registry breakdown sums exactly to 23,744,806. The encoder excludes all four reserved corpora, includes ten remaining CQADupStack forums, and includes Climate-FEVER’s separate 5,416,593 documents. There is no structural double-count or omission. [Volume registration](/home/dylan/asymetric-dual-encoders/m20/beir15_registry.json:612)

   The 66.52-hour figure remains an estimate: token throughput and SQuAD-derived tower ratios need not transfer perfectly between corpora. I verified their arithmetic, not the underlying measurements. The supplied eight token means omit 657,301 documents, so they do not independently establish the exact 92.2 mean; even assigning every omitted document 512 tokens yields **72.84 hours** under the same throughput and ratio assumptions.

2. **Protection after raising the threshold**

   Once armed, the refusal calculation is unchanged and runs after every new shard. A simulation advancing wall time refused the half-speed MS MARCO case at approximately **3.68 hours**.

   **2.4 hours is an expectation, not a maximum exposure.** With every modeled rate five times slower, arming takes approximately **12.4 hours**. At a constant 10 docs/s it takes approximately **28 hours**. At roughly 0.27–2.7 docs/s, the old threshold could arm within 102.3 hours while the new threshold cannot; the launcher’s hard timeout would stop the run instead.

   That is the newly admitted class: severely slow runs can consume substantially more time before projection refusal. For this monitored local continuation, the tradeoff is acceptable. It relies on the existing first-shard monitoring instruction as well as the hard timeout.

3. **Order, partial MS MARCO resume, and loading failures**

   The current registry order places arming inside MS MARCO. The threshold does **not** enforce that property.

   - Reordering or extensively skipping completed corpora changes the sampled distribution.
   - Partially completed MS MARCO still supplies the sample if enough unencoded rows remain; otherwise sampling proceeds into later corpora.
   - A corpus-loading exception propagates and stops execution; it is not silently skipped. [Encoding path](/home/dylan/asymetric-dual-encoders/m8src/pre_encode.py:273)

   Front-loading long documents makes the initial sample conservative, but does not guarantee every later prefix is conservative. After millions of short MS MARCO documents, later Climate-FEVER and Touché documents can reverse that bias. This limits the estimator’s guarantees; it does not invalidate this particular arming fix.

4. **The coming resume**

   With 134,473 rows already complete, **137,644** of the first six corpora remain. The first eligible MS MARCO shard boundary occurs after:

   - **1,037,644 new rows**, including **900,000 MS MARCO rows**;
   - approximately **2.085 encoding hours**;
   - a cumulative **138.25 docs/s**;
   - **67.84 hours** projected remaining.

   At that rate, counting the existing 134,473 rows as outstanding adds **0.270 hours—about 16 minutes**. That is immaterial against the stated margin.

   Skipping those rows also changes the sampled rate; the resumed sample is not the original batch prefix. The accepted debt therefore stays harmless here, but this calculation should not be generalized to a much larger future resume.

5. **Scope and existing results**

   This is appropriately small. Comparing executable definitions against `HEAD` showed **no changes to any existing function or class implementation**; the addition is the threshold helper plus its call-site argument.

   Existing shards are hash-checked and skipped before encoding. The change cannot rewrite an already-written vector, change the encoding contract, or alter any Stage A/B score. The reserved threshold remains 100,000. The caution in `FINDINGS.md` does not expose a concrete evidence risk from this isolated change. [Shard reuse](/home/dylan/asymetric-dual-encoders/m8src/pre_encode.py:304)

6. **Verification and tonight’s relaunch**

   All nine supplied checks passed using isolated definitions, without importing the guarded encoder module. Separate simulations advanced wall time and checked actual arming boundaries. No encoding, evaluation, full suite, protected reads, or file writes were performed; repository reads stayed within the allowlist.

   Two receipts outside the allowlist remain unverified: `results/m20_stage_c_run.json`, needed to establish **current remaining time**, and `results/m20_tower_rate_benchmark.json`, which supplies the gate’s exact runtime ratios.

**GO WITH CONDITIONS**

- Preserve the failed receipt under the launcher’s `m20_stage_c_run_attempt*.json` convention and inherit its original deadline. Recalculate remaining time; **102.3 hours was the failure-time budget**.
- Preserve both failed-run logs; the documented `> stage_c.log` command would truncate the controller log.
- Commit/push the reviewed change and preserve the generated receipts before relaunch; retain the existing first-shard monitoring for severe slowdown.
