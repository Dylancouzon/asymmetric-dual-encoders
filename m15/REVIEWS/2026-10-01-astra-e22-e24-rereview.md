1. **Resolved** — `_dirs()` creates table/head directories and runs before command dispatch. [e24:61](/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m15src/e24_prospective.py:61), [e24:240](/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m15src/e24_prospective.py:240).

2. **Resolved** — missing document IDs and ID/vector length mismatches stop E23 before exact-search computation. [e23:56](/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m15src/e23_fiqa25k.py:56). Correction to the earlier review: E20’s `sweep()` already writes `<ds>-doc-ids.json`. [e20:276](/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m15src/e20_ann_spaces.py:276).

3. **Resolved** — sweep orchestration and assembly use `included()`, sourced from E20’s result spaces. [e23:121](/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m15src/e23_fiqa25k.py:121), [e23:174](/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m15src/e23_fiqa25k.py:174).

4. **Resolved** — prediction hash/timestamp are persisted; fitting/scoring commands and assembly verify the hash. [e24:152](/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m15src/e24_prospective.py:152), [e24:156](/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m15src/e24_prospective.py:156).

5. **Resolved** — amendment explicitly declares the 20k teacher sample, workload-cloud ratio, and omitted table fit-cloud rank, stating it precedes feature computation. Historical timing is not independently verified. [MEASUREMENTS:668](/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m15/MEASUREMENTS.md:668).

6. **Resolved** — hubness includes zero-occurrence documents. [e22:68](/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m15src/e22_gap_predictors.py:68).

7. **Not resolved operationally** — space-cluster resampling and within-workload intervals are implemented, but introduce an essential crash: `log10_ndocs` is constant within each workload; every bootstrap correlation is nonfinite, leaving `vals=[]`, so `np.quantile` fails before results are written. Handle undefined intervals explicitly. [e22:170](/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m15src/e22_gap_predictors.py:170), [e22:203](/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m15src/e22_gap_predictors.py:203).

8. **Resolved** — each held-out fold uses its training mean and reports baseline MAE. [e22:150](/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m15src/e22_gap_predictors.py:150).

9. **Not resolved** — the check rejects only an already-ready HTTP endpoint. It ignores gRPC occupancy and permits simultaneous launches to pass the check; readiness still never verifies the launched process. [e20:99](/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m15src/e20_ann_spaces.py:99), [e20:113](/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m15src/e20_ann_spaces.py:113).

**E19 isolation note: not fully resolved.** Preflight checks all required features, but direct head commands bypass that assertion; missing features can still trigger E19 writes. [e24:74](/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m15src/e24_prospective.py:74), [e24:246](/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m15src/e24_prospective.py:246).

Codex session ID: 01a0f84c-481a-7060-9194-5cd192911c04
Resume in Codex: codex resume 01a0f84c-481a-7060-9194-5cd192911c04
