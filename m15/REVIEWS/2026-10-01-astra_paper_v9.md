# Codex review (astra_paper_v9) of PAPER.md draft v9 (664811b)

Date 2026-10-01. Model: gpt-6-astra for correctness, gpt-6-sol for the reader review. Access: named files only. Verbatim below; all Astra findings and Sol's scope points applied in v10 (E14 added; served-recipe test on a second tower not run, stated as a limitation).

---

1. **§7:** “On this subset it scores 0.6183 for Zero, against 0.6169 for Zero's exact dense search and 0.037 below Nano's.”  
   **Source:** `results/m15_e2_ann_msmarco1m.json`: Nano exact = 0.696254; Nano fused = 0.659764. Zero fused is **0.0779 below Nano exact**, or **0.0414 below Nano fused**.  
   **Minimal fix:** replace the ending with “0.0779 below Nano’s exact dense score.”

2. **Abstract:** “Over one frozen `stella_en_400M_v5` index, the table keeps 81.4% of the tower's nDCG@10 on 15 BEIR datasets at 0.044 ms per query on a laptop CPU.”  
   **Source:** `results/m20_beir15_run.json` measures trained Zero; E8 measures a different, closed-form table. The abstract currently implies continuity between them.  
   **Minimal fix:** replace “the table” with “the separately trained Zero table.”

3. **Introduction:** “Under one closed-form lookup-table recipe fitted to 26 towers, tower retrieval quality and three scalar proxies for ease of distillation did not rank the resulting tables.”  
   **Source:** `results/m15_e11_mechanism.json` and `results/m15_e11b_margin.json` explicitly cover **10 checkpoints**, not 26.  
   **Minimal fix:** separate the tower-quality result from “three exploratory proxies tested on the registered ten.”

4. **§4:** “No checkpoint's authors disclose training on these six test sets; most trained on their source families (`m15/e8_exposure.json`).”  
   **Source:** that file covers only the original **10 checkpoints**. Some source-family entries are community guesses or inherited exposure; Stella’s FiQA/ArguAna flags are community metadata.  
   **Minimal fix:** restrict the statement to the audited ten, distinguish evidence tiers, and state that the additional 16 lack this exposure audit.

5. **§8:** “Nano against LEAF on the four exposure-free sets was unresolved (-0.0011).”  
   **Source:** `m21/BENCHMARKS.md` defines clean-4 as **no disclosed Stella contact**, expressly not proven absence of exposure; `m15/e8_exposure.json` records source-family contact.  
   **Minimal fix:** replace “exposure-free” throughout with “clean-4,” retaining Appendix B’s operational definition.

6. **§5.2:** “Thus word order accounts directly for only part of the loss, while the remaining loss falls on those same queries.”  
   **Source:** `results/m15_e12_failure_modes.json` labels this an **association, not a cause**. The verified 0.034957 shuffle drop / 0.093883 gap = 37.2% does not identify a causal word-order contribution.  
   **Minimal fix:** “The shuffle-induced drop equals 37% of the gap; the gap also concentrates on shuffle-sensitive queries.”

7. **§5.1:** “Every such table is blind to word order by construction.”  
   **Source:** `results/m7_absorb_check.json` explicitly says an **n-gram row can distinguish different orders**; the preceding sentence includes word-pair rows.  
   **Minimal fix:** “The unigram tables used here are blind to token order.”

8. **§7:** “Nano shares the Stella index with Zero and Stella, enabling Section 6's routing and blending; its distance to a 400M tower is larger than the teacher-student gaps in that prior work.”  
   **Source:** `m15/RELATED_WORK.md` lists NanoVDR’s **2B→69–70M**, approximately 29×, versus Nano’s approximately 11.6×.  
   **Minimal fix:** restrict the compression comparison to named supported examples, such as LEAF.

**Files opened:** `m15/{PAPER,EVIDENCE,MEASUREMENTS,PROOF_ABSORB,RELATED_WORK}.md`, `m15/e8_exposure.json`, `m8/FINDINGS.md`, `m21/BENCHMARKS.md`, `m12/FINDINGS.md`; all 21 `results/m15_*.json`; `results/{m20_beir15_run,m13_reserved_run,m13_build_record,m7_absorb_check}.json`.

The 17%/26% tie split and several Appendix C historical claims lack verification in the permitted sources; references outside the allowlist remain unchecked.

**Verdict:** Not ready for correctness sign-off; the new headline statistics match, but the numerical and qualification fixes above remain.
