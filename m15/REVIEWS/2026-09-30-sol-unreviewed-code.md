# Codex gpt-5.6-sol, adversarial review of unreviewed M15 scripts

Date 2026-09-30. Brief: `briefs-2026-09-30/sol_unreviewed_code.md` (rerun with a do-it-directly preface after a first attempt asked a question). Access: named files only. All three findings applied before E12 ran. Verbatim below.

---

1. **Major — [m15src/e12_failure_modes.py:37](/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m15src/e12_failure_modes.py:37)**  
   E12 computes order dependence by subtracting a locally encoded shuffled-query score from the committed M20 full-query score. The reproduction gate permits up to 2% per-query disagreement, so this difference can include encoder/index reproduction error rather than word-order sensitivity.  
   **Numbers affected:** every `*_vs_order` correlation, `mean_order_dependence`, `share_order_dependent`, and the order-dependent/free gap means.  
   **Minimal fix:** score full and shuffled queries locally with the same Stella encoder and document vectors; use the paired local scores for `order`, retaining M20 rows only for Stella/Nano-minus-Zero gaps.

2. **Moderate — [m15src/e11b_margin.py:59](/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m15src/e11b_margin.py:59), [m15/make_evidence.py:97](/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m15/make_evidence.py:97), [m15/PAPER.md:229](/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m15/PAPER.md:229)**  
   E11b does not calculate the single “vector error divided by median gap” described by the paper. It calculates an RMS cosine-derived error divided by each dataset’s median gap, then averages the six ratios. Likewise, `median_gap_six` is the mean of six dataset medians, not one median. Consequently the displayed inputs do not produce the displayed ratio: Stella’s 0.783 additivity and 0.0839 gap imply approximately 7.86, not 9.35.  
   **Numbers affected:** all E11b error/margin values and potentially the reported −0.42/−0.13 Spearman values; the 0.026, 0.084, and 0.109 “median” claims are mislabeled.  
   **Minimal fix:** either compute the literal pooled statistic and rerun correlations, or describe the existing estimator as the “macro-average of per-dataset RMS-error/median-gap ratios” and label the gap as “mean of dataset medians.”

3. **Minor — [m15src/e12_failure_modes.py:61](/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m15src/e12_failure_modes.py:61)**  
   E12 promises mean gaps by tercile of each factor, but emits only fertility-tercile values; order-dependence terciles are absent.  
   **Numbers affected:** missing Stella and Nano gap-by-order-tercile results.  
   **Minimal fix:** add both order-tercile arrays or narrow the stated analysis.

**Files opened:** all eight requested scripts; all eleven named context files; `m15src/encode_more.py`; `m15/PAPER.md`, `m15/EVIDENCE.md`, `m15/MEASUREMENTS.md`; the four permitted result JSONs; the two relevant document-vector metadata files.

**Verdict:** Three material number/claim issues; no query-ID alignment, cache-key, self-hit, Stella document-path, figure-field, or evidence-field defect otherwise found.
