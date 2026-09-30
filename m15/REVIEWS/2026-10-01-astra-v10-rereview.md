# Codex gpt-6-astra, focused re-review of the v9 fixes

Date 2026-10-01. All eight fixed; one minor bootstrap-description fix applied. Verbatim below.

---

1. **Fixed** — §7: Zero fusion trails Nano exact by 0.0779244, correctly rounded to 0.0779.
2. **Fixed** — Abstract explicitly identifies the “separately trained Zero table.”
3. **Fixed** — Introduction restricts the three proxies to the registered ten; §4 labels their analysis exploratory.
4. **Fixed** — Exposure statement covers only the registered ten, acknowledges community metadata, and identifies the additional 16 as unaudited.
5. **Fixed** — “Exposure-free” is replaced throughout by “clean-4”; Appendix B retains its operational definition.
6. **Fixed** — §5.2 describes the 37% ratio and loss concentration as associations, removing causal attribution.
7. **Fixed** — Order blindness is correctly restricted to unigram tables.
8. **Fixed** — Size-gap comparison is restricted to LEAF; 400M/34.5M versus 109M/23M supports it.

**E14 check:** Single-forum correlations round correctly to 0.87 and 0.92 for the registered ten. Family-bootstrap intervals correctly round to [0.74, 0.95] and [−0.36, 0.53]; the file groups all 26 checkpoints into ten families.

**New minor error:** §4’s Scope still says “Intervals are bootstraps over checkpoints,” contradicting the added family-bootstrap intervals. Qualify this with “unless otherwise stated.” Also explicitly identify the family-bootstrap result as exploratory and covering all 26, since its sentence begins with the registered-ten analysis.

**Verdict:** All eight findings are fixed; E14 numbers match, with a minor bootstrap-description correction remaining.
