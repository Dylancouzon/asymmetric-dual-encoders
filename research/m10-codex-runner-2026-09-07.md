# Codex adversarial review — M10 screen-arm runner (`run_arm.py`, ff15eeb), 2026-09-07 (verbatim; read-exclusion audit clean)

NO-GO for family F.

Ranked findings:

1. **P0 — Backbone trains in evaluation mode.** [run_arm.py:477](/home/dylan/asymetric-dual-encoders/m10src/run_arm.py:477), [nano10.py:144](/home/dylan/asymetric-dual-encoders/m10src/nano10.py:144), [trainer10.py:114](/home/dylan/asymetric-dual-encoders/m10src/trainer10.py:114) — Every F warm start calls `pooled_features()`, which sets `model.eval()`; the trainer never calls `train()`. Input: either F arm → dropout remains disabled for all 20M examples. Fix: call `model.train()` immediately before training and after restore.

2. **P0 — The registered training recipe is not implemented.** [trainer10.py:89](/home/dylan/asymetric-dual-encoders/m10src/trainer10.py:89), [trainer10.py:120](/home/dylan/asymetric-dual-encoders/m10src/trainer10.py:120), [instructions-m10.md:531](/home/dylan/asymetric-dual-encoders/instructions-m10.md:531), [instructions-m10.md:578](/home/dylan/asymetric-dual-encoders/instructions-m10.md:578) — Training is full fp32, not bf16-autocast, and AdamW applies weight decay to biases/LayerNorm although the recipe says `wd=0.01` only for `dim>1`. Fix: add CUDA bf16 autocast and registered decay/no-decay parameter groups.

3. **P0 — The 2,000-step cycle-1 warmup is absent.** [instructions-m10.md:580](/home/dylan/asymetric-dual-encoders/instructions-m10.md:580), [nano10.py:305](/home/dylan/asymetric-dual-encoders/m10src/nano10.py:305) — Step 0 runs at `1e-4`; Recipe specifies a 2,000-step warmup. Fix: encode warmup in `lr_at()` and test steps 0, 1,999, 2,000 and every cycle end.

4. **P0 — A kill at the final cycle end becomes success.** [trainer10.py:148](/home/dylan/asymetric-dual-encoders/m10src/trainer10.py:148), [run_arm.py:511](/home/dylan/asymetric-dual-encoders/m10src/run_arm.py:511) — Example scheduled macros `mid/end = .50/.50, .51/.51, .49/.49`: kill fires on final `end`, but three end values make `finished=True`; runner records `complete`, exposes `cycle3.pt` as final, and runs DEV-6. Fix: success requires `stopped is None` or exactly `plateau at cycle 3`; every kill/non-finite reason must fail.

5. **P1 — Real-arm knobs are caller-overridable.** [run_arm.py:401](/home/dylan/asymetric-dual-encoders/m10src/run_arm.py:401), [run_arm.py:457](/home/dylan/asymetric-dual-encoders/m10src/run_arm.py:457), [run_arm.py:598](/home/dylan/asymetric-dual-encoders/m10src/run_arm.py:598) — `F-bge-small --n-fit 1 --max-len 8 --compile` writes a real record for a materially different arm. Fix: permit these overrides only under smoke/test mode; reject them for registered runs.

6. **P1 — The registry/`SHAPES` consistency check is ineffective.** [run_arm.py:194](/home/dylan/asymetric-dual-encoders/m10src/run_arm.py:194), [run_arm.py:197](/home/dylan/asymetric-dual-encoders/m10src/run_arm.py:197) — `spec["batch"]` is overwritten from the registry before comparison; changing `SHAPES["E-bs128"]["batch"]` to 32 still passes. Pattern, default objective, head and warm-start fields are also incompletely compared. Fix: compare the untouched shape against a fully resolved registry recipe before copying anything.

7. **P1 — Resume loses the evidence later contrasts need.** [trainer10.py:97](/home/dylan/asymetric-dual-encoders/m10src/trainer10.py:97), [run_arm.py:486](/home/dylan/asymetric-dual-encoders/m10src/run_arm.py:486), [run_arm.py:549](/home/dylan/asymetric-dual-encoders/m10src/run_arm.py:549) — Resume restores scalar macro histories, but creates a fresh `CovEval.records`. Resume after cycle 2 → final record omits cycle-1/2 family macros and per-query paths, so last-two-cycle contrasts cannot be computed from the record. Fix: checkpoint and restore complete evaluator records, including file hashes/paths.

8. **P1 — Resume is not bound to its original recipe.** [trainer10.py:47](/home/dylan/asymetric-dual-encoders/m10src/trainer10.py:47), [run_arm.py:507](/home/dylan/asymetric-dual-encoders/m10src/run_arm.py:507) — Resume with changed registry, `--max-len`, compilation mode, or corpus manifest continues a hybrid run and stamps only the new registry hash. Fix: store and verify an arm/recipe/registry/manifest fingerprint in every checkpoint.

9. **P1 — Crashes and OOMs are not recorded as failures.** [run_arm.py:504](/home/dylan/asymetric-dual-encoders/m10src/run_arm.py:504), [run_arm.py:578](/home/dylan/asymetric-dual-encoders/m10src/run_arm.py:578), [screen_registry.json:367](/home/dylan/asymetric-dual-encoders/m10/screen_registry.json:367) — A CUDA OOM propagates before any record is written, contrary to `rules.arm_failure`. Fix: wrap the run with a terminal failure-record writer that never labels a partial checkpoint final.

10. **P1 — Real evaluation is reachable at unregistered smoke points.** [run_arm.py:486](/home/dylan/asymetric-dual-encoders/m10src/run_arm.py:486), [run_arm.py:527](/home/dylan/asymetric-dual-encoders/m10src/run_arm.py:527), [run_arm.py:602](/home/dylan/asymetric-dual-encoders/m10src/run_arm.py:602) — `--smoke-steps 6 --real-eval` reads real COV at 64/128/192 examples and DEV-6 at 192 examples. Fix: prohibit real evaluators whenever `smoke_steps` is set.

11. **P1 — Post-F arms do not use F’s winner.** [screen_registry.json:19](/home/dylan/asymetric-dual-encoders/m10/screen_registry.json:19), [instructions-m10.md:619](/home/dylan/asymetric-dual-encoders/instructions-m10.md:619), [arm_smoke.py:56](/home/dylan/asymetric-dual-encoders/m10src/arm_smoke.py:56) — If L6 wins F, launching `ANCHOR` or later arms still constructs hard-coded bge-small. Fix: resolve and cryptographically bind the registered F verdict before permitting any post-F arm.

12. **P2 — COV validates its surface only after scoring it.** [cov_eval10.py:61](/home/dylan/asymetric-dual-encoders/m10src/cov_eval10.py:61), [cov_eval10.py:77](/home/dylan/asymetric-dual-encoders/m10src/cov_eval10.py:77) — An extra unit returned by `cov_probe.units()` is encoded/scored before `cov_macro` rejects it. Fix: call `assert_surface()` before the scoring loop.

13. **P2 — Re-run protection checks only one non-atomic copy.** [run_arm.py:436](/home/dylan/asymetric-dual-encoders/m10src/run_arm.py:436), [run_arm.py:578](/home/dylan/asymetric-dual-encoders/m10src/run_arm.py:578) — A complete `results/m10_arm_F-bge-small.json` with a missing/malformed work record is overwritten. Fix: atomically write one canonical record and refuse if either published path already exists.

Operationally, the supplied smoke artifact is not a valid gate for this commit: it is only 60 steps for A1 and records git `adc9f1…`, not current `ff15eeb` ([record.json:8](/home/dylan/asymetric-dual-encoders/work/m10arms/smoke/A1/record.json:8), [record.json:451](/home/dylan/asymetric-dual-encoders/work/m10arms/smoke/A1/record.json:451)).

Verified correct: F read arithmetic is exactly 5M/10M/20M; the extra 5M read stays outside kill/plateau; final LR is `1e-5`; dose flooring is correct; smoke and real output trees differ; `streams_of()` is fragile but fails closed. Four checkpoints per arm consume about **21.1 GiB for the 14-arm band**, plus roughly **0.4 GiB transient space** during atomic saves.

Tests could not execute because the read-only environment provides no writable temporary directory. No files were changed.
