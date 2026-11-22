# Codex gpt-6-astra, read-only review of the E8 driver at 878d475

Date 2026-09-30. Brief: `briefs-2026-09-30/e8_driver_brief.md`. Access audited: committed source via `git show` with reserved names filtered; one out-of-allowlist read of a Codex plugin skill file outside the repository; no protected path. Returned text below, verbatim.

---

Reviewed **`878d475`** directly from Git; checkout HEAD is `bcc93a0`. Static review only—no edits or experiment execution.

1. **P1 — Failed convergence prevents the registered result.** `m15src/e8_towers.py:95`  
   `_gate()` exits without persisting diagnostics. `all_steps()` then aborts because subprocesses use `check=True`; freeze and assembly require successful files for every configuration. Registration requires stopping **that configuration** and listing its failure diagnostics in the result.  
   **Minimal fix:** atomically persist failed status and solve diagnostics, preserve preceding lambda diagnostics, and make orchestration/assembly explicitly handle failed configurations without scoring them.

2. **P1 — Resume repeats already completed six-set scoring.** `m15src/e8_towers.py:236`, `:248`  
   All six datasets are scored before any scoring results are saved. An interruption during a later dataset loses earlier completed scores; restarting loads and scores those datasets again, violating the registered one-shot step and wasting compute. A truncated final JSON is also treated as completed merely because it exists.  
   **Minimal fix:** persist atomic per-dataset results bound to the frozen-lambda hash; validate completion before skipping. Record an in-progress state and refuse automatic replay of an interrupted scoring transaction.

3. **P1 — The documented standalone `dev` command lacks import setup.** `m15src/e8_towers.py:21`, `:169`  
   The script adds only `m15src` and `m8src`, then imports `devsuite` before `Tower()` imports `m8base`, which supplies `m7src`. With `M7_ENCODER` correctly set but no externally supplied `PYTHONPATH`, the documented standalone command fails with `ModuleNotFoundError`. `all` masks this by supplying its own `PYTHONPATH`.  
   **Minimal fix:** initialize `m8base` before importing `devsuite`, after setting/validating the requested encoder, or explicitly bootstrap `m7src`.

4. **P1 — Tokenizer compatibility is reported but never enforced.** `m15src/e8_towers.py:80`, `:315`  
   Compatibility checks only vocabulary size and CLS ID, not equality with the shared ordered vocabulary. More importantly, no later stage consumes `compatible`: every configuration still fits and scores. This fails the registered exclusion rule.  
   **Minimal fix:** compare the ordered-vocabulary hash against the shared reference, persist eligibility, and enforce the same frozen eligible roster in dev, freeze, six-set scoring, and assembly. Preserve excluded configurations and reasons in the output.

5. **P1 — Assembly drops required outputs and provenance.** `m15src/e8_towers.py:272`, `:307`, `m15src/common.py:63`  
   The final artifact discards per-forum dev ceiling scores and model repository/revision metadata already present in dev files. Model file hashes and dev-data provenance are never collected. Its receipt starts at assembly, so it does not record the run’s actual start. The required exposure matrix can silently become `null` while status remains `COMPLETE`. These omissions matter because intermediate files stay on the pod.  
   **Minimal fix:** carry required dev fields and model identities into the final JSON, collect file hashes and dev-data provenance during execution, persist the actual run start, and require a complete exposure matrix before starting expensive work.

Files opened, wholly or partially:

- `m15/MEASUREMENTS.md`
- `m15src/e8_towers.py`
- `m15src/common.py`
- `m7src/encoders.py`
- `m7src/teacher.py`
- `m7src/table.py`
- `m7src/stage0_ridge.py`
- `m7src/evalkit.py`
- `m7src/devsuite.py`
- `m7src/init_table.py`
- `m7src/_paths.py`
- `m7src/dev_eval.py`
- `m8src/teacher_screen.py`
- `m8src/blockcg.py`
- `m8src/init_m8.py`
- `m8src/m8base.py`
- `m8src/paths_guard.py`
- `results/m20_beir15_scores/trec-covid/zero-dense.json`

Also opened, outside your allowlist in error: `/Users/dylanc/.codex/plugins/cache/sendbird/cc/1.5.0/skills/adversarial-review/SKILL.md`. No protected payload paths were opened or listed.

**Verdict: Fix the P1 findings before launching the paid one-shot run.**
