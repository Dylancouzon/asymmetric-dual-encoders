# Opus sub-agent, independent read-only review of the E8 driver at 878d475

Date 2026-09-30. Brief: `briefs-2026-09-30/e8_driver_brief.md` (same brief as Astra, run independently). Access: the agent's own file list names only allowlisted files; it opened nothing under `work/` or `results/frozen_eval/`. Returned text below, verbatim.

---

**E8 driver review (`m15src/e8_towers.py` at 878d475), second reviewer**

**Findings**

1. **P1: A failed convergence gate stops the whole run, and the failed configuration never gets listed.** Location: `e8_towers.py:95`, with effects at `:316-323`, `:215` and `:269-270`.
   - The registration says a failed solve "stops that configuration, and the result lists it with its diagnostics." In the driver, `_gate` raises `SystemExit` inside that tower's subprocess, and `all_steps` (`check=True`) then aborts every tower that follows. No `dev-<name>.json` is written, so `freeze` and `assemble` crash on the missing file. The result cannot list the configuration without a code change in the middle of the run, while the pod keeps billing.
   - The trigger is likely. `arctic-embed-l-mean` is third in `CONFIGS`. Its M7 lambda curve was 0.221 at 1e-4, 0.221 at 1e-3 and 0.2138 at 1e-2 (`results/m7_learnability_report.json`), so a best value at the 1e-4 edge, and then a 1e-5 solve, is close to a coin flip. The 1e-5 solve is the worst-conditioned one in the plan. The B7 real-data check (`m8/RESULTS.md:20`) covered only the four grid values, on Stella.
   - Minimal fix: in `dev()` and `six()`, catch the gate failure and write the file with `"status": "STOPPED"` plus that solve's diagnostics. Then make `freeze` and `six` skip STOPPED configurations, and make `assemble` list them under `stopped` and compute rho over the rest, with n recorded.

2. **P2: The tokenizer check reports but does not exclude.** Location: `e8_towers.py:80-86`.
   - The registration says an incompatible configuration "is reported and excluded". The driver computes `compatible` and never acts on it.
   - No number changes as long as all 11 configurations are 30,522 WordPiece with CLS id 101, which the registry notes state.
   - Fix: in `preflight`, drop every configuration with `compatible` false from the roster, or stop.

3. **P2: Nothing binds the frozen file to the dev files after the freeze.** Location: `e8_towers.py:168-173` and `:269-273`.
   - `dev()` does not refuse to run once `FROZEN` exists. `assemble` takes `dev_table` from `dev-<name>.json`, not from `frozen_lambdas.json`. If someone deletes and reruns a dev file after the freeze, the result mixes a new dev score with the frozen lambda, and no check notices.
   - Fix: make `dev()` refuse when `FROZEN` exists. In `assemble`, assert that the frozen `lambda`/`dev_macro_2`, the dev file's `best_lambda`/`best_dev_macro_2`, and the six file's `lambda` all agree.

4. **P2: A registered output can drop out silently.** Location: `e8_towers.py:307`.
   - `exposure` becomes `None` when `m15/e8_exposure.json` is missing, and that file first appears in bcc93a0. A pod that runs 878d475 would ship without the registered exposure matrix.
   - Fix: make the file required, and run the pod at bcc93a0 or later.

5. **P2: Two things can crash at the last step on a Linux pod.** Location: `common.py:67` and `e8_towers.py:259`.
   - `receipt()` runs `sysctl` without error handling. A container without procps raises `FileNotFoundError`.
   - `.statistic` needs scipy 1.10 or later.
   - No spend is lost, because `assemble` reruns from the six files. But the fix would be an uncommitted patch after the run.
   - Fix: wrap the `sysctl` call in `try/except OSError`, and use `spearmanr(a, b)[0]`. I checked that `[0]` works on scipy 1.18.1.

6. **P2: `six()` resumes per tower only.** Location: `e8_towers.py:236-252`.
   - If a tower crashes on, say, trec-covid, a rerun loads and scores the sets it already finished again, and writes more entries to `SIX_ACCESS.log`.
   - The numbers stay the same, but the "run once" record gets harder to read.
   - Fix: write each dataset's result to a partial file and skip datasets that are done.

7. **P2: `dev <name>` fails when run by hand.** Location: `e8_towers.py:169`.
   - `import devsuite` runs before `Tower()` imports `m8base`, so `m7src` is not on `sys.path` unless `PYTHONPATH` is set. `all` sets it; a manual resume of one tower fails at once, with no spend.
   - Fix: move `import devsuite` after `t = Tower(name)`.

**Checks that pass**
- The lambdas are frozen before any BEIR load. `six` refuses to run without `FROZEN`, and no BEIR data loads before step 3.
- The gate runs before every score, including the final re-solve. It checks `converged` and `worst <= tol`, and a NaN residual stops the solve.
- The fit list is checked by sha256 and by count.
- The e5 document path uses `passage: ` everywhere, and the ceilings use each tower's `query: ` prefix.
- `arctic-embed-l` and `arctic-embed-l-mean` do not share cache keys: `pooling_key` separates the encode caches and `spec_tag` uses the spec name for the init cache.
- The control is excluded from `checkpoints_10`.
- The M20 pins match all six committed zero-dense rows.
- Retrieval takes the top 100 with self-hits dropped.
- Memory is small: 171k trec-covid documents take about 700 MB on the GPU, and Y takes 1.4 GB of host memory.
- No `common.py` in `bench/`, `m7src/` or `m8src/` shadows the M15 one.

**What I could not verify**
- `bench/core.py` (`_m7_access_trail`, `doc_text`) is outside the allowlist, so I only confirmed that the file exists.
- I did not recompute the Stella `trainq-337981-fp16-7423fa42cd3e` cache key, because runs were not allowed. The driver uses the same name, prefix and dtype as `teacher_screen.screen_one`.
- Convergence of a 1e-5 solve is untested.

**Files opened**
- `m15/REVIEWS/briefs-2026-09-30/e8_driver_brief.md`
- `m15src/e8_towers.py`, `m15src/common.py`, `m15src/test_m15.py`, `m15src/roster_ids.py`
- `m15/MEASUREMENTS.md`
- `m8src/teacher_screen.py`, `m8src/blockcg.py`, `m8src/init_m8.py`, `m8src/m8base.py`, `m8src/paths_guard.py`
- `m7src/encoders.py`, `m7src/teacher.py`, `m7src/table.py`, `m7src/stage0_ridge.py`, `m7src/evalkit.py`, `m7src/devsuite.py`, `m7src/_paths.py`, `m7src/dev_eval.py`, `m7src/init_table.py`
- `m7src/teacher_probe.py` (lines 1-140 only)
- `m8/RESULTS.md` (grep only)
- `results/m8_trainq_manifest.json`, `results/m7_learnability_report.json`
- `results/m20_beir15_scores/{scifact,nfcorpus,fiqa,arguana,scidocs,trec-covid}/zero-dense.json` (metadata fields and query count only)

**Verdict:** GO after one fix. Fix finding 1 (a failed solve must stop only its own configuration and be listed) before the pod starts. The rest is P2 debt.
