# Codex gpt-6-astra, focused check of PAPER.md draft v6 (38b5ffa)

Date 2026-09-30. Access: named files only. All six findings applied. Verbatim below.

---

V5 findings **1, 5, 6, 7, 9, 10, 11 are fixed; 2, 3, 4, 8, 12 are only partly fixed.** The new ranges, FiQA ranks, 28%/24% shares, 0.4834 score, and 2.7/3.8 ms estimates match their sources.

1. **Screen timing:** “The screen took 4 to 7 minutes per tower on one A100, including encoding the 337,981 training queries and both forums with the tower.”  
   **Source:** E13 records integer durations of 4–7 minutes without timing boundaries; `MEASUREMENTS.md` explicitly says Stella’s fit vectors were cached. **Minimal fix:** retain the recorded durations but remove “including encoding…” unless separately substantiated; disclose cached inputs.

2. **V5 #2:** “For the Stella path the forums chose a = 0 (`results/m15_e9_blend.json`, registered; a code check printed SciFact's curve before the weights were frozen, which changed nothing in the procedure and is recorded in `m15/MEASUREMENTS.md`).”  
   **Source:** E9’s scope is “descriptive”; the pre-freeze SciFact peek is now disclosed correctly. **Minimal fix:** label E9 “registered descriptive.”

3. **V5 #3:** “That move costs 0.009 nDCG@10 on average against 0.047 for shuffling, and its damage correlates with Zero's gap at only 0.07.”  
   **Source:** E12b reports 0.008802, 0.047312 and 0.066465 on **3,069 Stella-positive queries**, rather than all 3,727. The table identifies the restriction only for its partial-correlation row. **Minimal fix:** begin this sentence “Among the 3,069 queries Stella scores above zero…”

4. **V5 #4; new conclusion error:** “Expect the table to fail on queries whose meaning lives in their word order, and route on its own retrieval margin to catch a quarter of them.”  
   **Source:** E12/E12b establish shuffle-loss associations (ρ = 0.456; partial ρ = 0.417), not semantic causation. E10’s **0.24737** is the fraction of oracle quality gain over matched random routing, not the fraction of affected queries caught. **Minimal fix:** “Shuffle-sensitive queries show larger table losses; margin routing recovers about a quarter of the oracle’s gain over matched random routing.” Apply the association qualification to the introduction and FiQA interpretation too.

5. **V5 #8:** “And measure the saving inside the search engine: a table's query costs more graph search, so a 60-fold encode saving becomes 2- to 3-fold end to end.”  
   **Source:** E2 gives Nano/Zero **2.23–3.12× on MS MARCO-1M**, versus **3.76–4.26× on FiQA**. **Minimal fix:** specify Nano, the MS MARCO-1M diagnostic, and 1–5% relative quality-loss targets.

6. **V5 #12:** “M15 result files (`results/m15_*.json`) carry the script hash, git commit, seed and machine; model and dataset revisions are recorded in the result, in the scripts it names, or in the result it was derived from (E8's frozen lambdas are bound by hash into E8's result; E12 and E12b take their pins from the committed M20 rows).”  
   **Source:** `m15_e8_frozen_lambdas.json` still contains none of those receipt fields. **Minimal fix:** explicitly exempt frozen configuration files from the first clause.

Files opened at `38b5ffa`: the named v5 review; `m15/{PAPER,EVIDENCE,MEASUREMENTS,PROOF_ABSORB}.md`; `m8/FINDINGS.md`; `results/m20_beir15_run.json`; `results/m15_{e1_latency,e2_ann_msmarco1m,e2_ann_fiqa,e4_prefix,e8_towers,e8_frozen_lambdas,e9_blend,e10_router,e12_failure_modes,e12b_fragility,e13_robustness}.json`.

**Verdict:** Numerical checks pass; the six qualifications above remain necessary.
