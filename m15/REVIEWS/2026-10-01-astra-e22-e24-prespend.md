**NO-GO for all three as written.** Findings, ordered by severity:

1. **E24 — fresh runs fail when saving the first dev grid.**  
   [e24_prospective.py:83](/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m15src/e24_prospective.py:83) creates only `e24/`, not `e24/table/` or `e24/head/`. Reused writers do not create parents ([e8_towers.py:46](/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m15src/e8_towers.py:46)). **Inference:** after paying for the first grid, saving fails on a fresh workspace.  
   **Minimal fix:** create both directories before starting teachers or students.

2. **E23 — required document-ID cache has no producer in the inspected E20 driver.**  
   [e23_fiqa25k.py:56](/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m15src/e23_fiqa25k.py:56) reads `fiqa-doc-ids.json`; `all` silently skips spaces without it at line 163. E20’s vector writer saves vectors and exact indices, but no document-ID list ([e20_ann_spaces.py:145](/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m15src/e20_ann_spaces.py:145)). **Inference:** caches produced solely by that driver cannot run E23. Existing cache contents were not inspected.  
   **Minimal fix:** reconstruct IDs from the same pinned public corpus order, verify alignment, and fail before spending if required inputs are absent.

3. **E23 — its roster includes E20-excluded spaces, breaking paired analysis.**  
   [e23_fiqa25k.py:118](/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m15src/e23_fiqa25k.py:118) uses `E20.eligible()`, which selects E8/E8x participants, not E20’s included spaces ([e20_ann_spaces.py:58](/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m15src/e20_ann_spaces.py:58)). A successful 25k sweep for an E20-excluded space receives `None` comparison gaps at lines 131–139; subtraction then fails. It also spends time on an unpairable space.  
   **Minimal fix:** sweep and assemble the committed E20 included roster; require both comparison workloads.

4. **E24 — execution order holds, but the prediction commitment is not enforced.**  
   [e24_prospective.py:135](/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m15src/e24_prospective.py:135) writes predictions and prints their hash; `_require_predictions()` only checks existence at line 140. The final result hashes the file after student scoring at line 183. This does not establish the pre-scoring commitment required by [MEASUREMENTS.md:659](/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m15/MEASUREMENTS.md:659).  
   **Minimal fix:** persist the prediction hash in the required pre-scoring commitment and verify that committed hash before every student command and assembly.

5. **E22 — feature (4) is a different measurement.**  
   [e22_gap_predictors.py:95](/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m15src/e22_gap_predictors.py:95) uses a 20k teacher fit sample; line 110 computes **table/teacher workload-query rank**, omitting table fit-cloud rank. The declared method requires both fit-cloud ranks and their ratio ([MEASUREMENTS.md:620](/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m15/MEASUREMENTS.md:620)).  
   **Minimal fix:** implement the declared feature, or amend the method **before computing features** to specify the sample, workload-cloud ratio, and omitted quantity. Disclosure after observations is insufficient.

6. **E22 — hubness drops zero-occurrence documents.**  
   [e22_gap_predictors.py:70](/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m15src/e22_gap_predictors.py:70) calculates Gini only among retrieved documents. This changes the declared corpus-wide occurrence-count statistic: concentrating every query on the same ten equally frequent documents yields zero.  
   **Minimal fix:** use `gini(counts)` with the full document support.

7. **E22 — bootstrap uses rows instead of spaces and omits within-workload intervals.**  
   [e22_gap_predictors.py:157](/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m15src/e22_gap_predictors.py:157) independently resamples the 75 observations, separating each space’s three workloads. Lines 192–194 emit within-workload correlations without intervals. Both depart from [MEASUREMENTS.md:624](/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m15/MEASUREMENTS.md:624).  
   **Minimal fix:** resample space IDs, retaining their workload rows together; also calculate the declared within-workload intervals.

8. **E22 — constant baseline uses held-out outcomes.**  
   [e22_gap_predictors.py:195](/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m15src/e22_gap_predictors.py:195) evaluates the full-data mean against held-out ridge predictions. That is not an equivalent out-of-sample comparison.  
   **Minimal fix:** predict each held-out family/workload using its training-fold mean and report separate baseline metrics. Ridge standardization itself correctly uses training folds.

9. **E23 — concurrent shards can attach to the wrong Qdrant process.**  
   [e23_fiqa25k.py:74](/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m15src/e23_fiqa25k.py:74) inherits E20’s ports. Startup accepts any successful `/readyz` response without checking the launched process ([e20_ann_spaces.py:99](/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m15src/e20_ann_spaces.py:99)). **Inference under the requested concurrent-shard scenario:** an existing server can satisfy readiness while the new process fails to bind.  
   **Minimal fix:** allocate distinct HTTP/gRPC port pairs per shard and reject occupied ports before launch.

The remaining requested checks:

- **E24 routing:** dev/freeze/six result paths follow the rerouted globals. `E19.FEATS` remains `work/m15/e19/feats`; existing features are reused, while missing features are computed and written there ([e19_head_screen.py:102](/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m15src/e19_head_screen.py:102)). Thus it does **not** write exclusively under E24. For strict isolation, reuse existing E19 features read-only and put cache misses under E24.
- **E22 teacher selection:** `all` correctly sets `M7_ENCODER=name` before each fresh subprocess imports `teacher` ([e22_gap_predictors.py:217](/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m15src/e22_gap_predictors.py:217)).
- **E23 sampling/exactness:** the seeded uniform draw conditional on retaining positive-qrel documents is correct; exact rankings and tie thresholds are recomputed over the subsample ([e23_fiqa25k.py:35](/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m15src/e23_fiqa25k.py:35), [line 64](/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m15src/e23_fiqa25k.py:64)).
- No explicit reserved-set read or old E8/E8x/E19 result overwrite appeared in the inspected call paths. Imported cache/access helpers and `write_result` were outside the allowed inspection scope, so their protections are not certified.

**Verdicts:** E22 **NO-GO**; E23 **NO-GO**; E24 **NO-GO**. No experiment code was executed and no files were changed.

Codex session ID: 01a0f848-8525-78e3-bd20-34e432bcf971
Resume in Codex: codex resume 01a0f848-8525-78e3-bd20-34e432bcf971
