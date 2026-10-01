1. **E20 reports the wrong multiplier distribution.** [e20_ann_spaces.py:337](/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m15src/e20_ann_spaces.py:337) flattens both builds and computes quantiles over uncensored build rows. The method requires a distribution across spaces **with builds averaged**. This changes the reported quantiles and lets a partially censored space contribute only its successful build.  
   **Minimal fix:** average the two multipliers per space before computing distribution statistics; retain a censored space-level result when either build is censored. The individual multiplier calculation and `None` censoring are correct.

2. **E19 omits pre-specified clean-four leave-one-family-out analyses.** [e19_head_screen.py:336](/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m15src/e19_head_screen.py:336) computes family exclusions only for the two all-six correlations. E19 specifies clean-four variants of those analyses. The existing pooled Spearman calculations are not invalidated, but the required family sensitivity results are missing.  
   **Minimal fix:** add clean-four recipe-2-versus-recipe-1 and recipe-2-versus-teacher correlations to the same family-exclusion loop for every roster.

3. **E20 never performs the promised E2 consistency comparison.** [e20_ann_spaces.py:357](/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m15src/e20_ann_spaces.py:357) writes the result without comparing Stella’s FiQA rows against E2 or reporting agreement within the two-build spread.  
   **Minimal fix:** emit the non-gating comparison for corresponding unquantized rows. Identify comparable paths explicitly: E20 re-solves an E8 table, whereas E2 uses its `zero` encoder; those student paths should not silently be treated as identical.

E19’s selection order, ridge equation, feature application, and normalization match the stated recipe. Cache calls reuse E8’s keys; missing shards can trigger encoding. Final results are protected against overwrite, and I found no reserved-set access in the inspected paths. E20’s self-hit handling and parity gate are consistent with E2’s scoring intent. It creates fresh collections with different insertion orders; distinct graph contents cannot be established by static inspection.

No experiments or tests were executed, files changed, or excluded data read. **No other essential issues found.**

Codex session ID: 01a0f5c9-14af-7941-8111-03147b8eb337
Resume in Codex: codex resume 01a0f5c9-14af-7941-8111-03147b8eb337
