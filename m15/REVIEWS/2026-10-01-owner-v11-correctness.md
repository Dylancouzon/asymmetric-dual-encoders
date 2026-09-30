# Focused scientific re-review of draft v11, 2026-10-01

**Outcome: no essential scientific finding in this focused re-review.** The four issues in `2026-10-01-owner-correctness.md` are addressed sufficiently for the revised scope. No further experiment is required to support this version's central argument.

The central claim is now appropriately specific: one closed-form table recipe shows a large teacher/student ranking reversal, a development screen helps for that recipe and target, and the separately trained Stella query paths have measured build and serving costs. The text does not claim that ridge screening validates teacher selection for trained Zero or Nano.

## Disposition of the four findings

1. **Teacher-choice scope — addressed.** Sections 3.1, 3.3, 4.2, and 4.3 distinguish the closed-form representation from trained students. The 0.1519 selection difference is explicitly exploratory, its teacher-score comparator is identified as hindsight, and clean-4/per-domain regret is visible. Section 4.2 states that the correlation interval allows moderate associations and does not establish independence.
2. **Shuffle interpretation — addressed.** Appendix D calls this a diagnostic, names the shared-score coupling, labels the 90% as a net gap, and explains why broad score bins and isotropic noise do not establish a word-order mechanism or exclude general fragility. It no longer carries the paper's main argument.
3. **Dimension and retention — addressed.** Section 4.3 gives the absolute-score correlation before the ratio and states the denominator issue and confounds. The size claim is removed from the abstract and conclusion.
4. **Build accounting — addressed.** Table 1 distinguishes measured Nano optimization time, its historical rental-price calculation, estimated Zero preparation/retraining, and coarse screen timing. The omitted upstream work is explicit. Neither the abstract nor the conclusion presents $95 as complete reproduction cost or evidence of cost optimality.

## Focused checks of revised material

- Recomputed E15's hours from seconds and priced rental cost from hours times rate; checked both roster selection differences. All agree with v11's rounding. E15's construction and source-field logic were checked in the first review; this re-review did not repeat the historical source audit.
- The main deployment comparison now explicitly preserves each encoder's own exact quality, rather than implying an equal-quality comparison. The positive-preserving MS MARCO subset and hardware limits remain visible.
- Appendix B retains the Nano contrasts, including unresolved and negative held-out results, adds the published four-dataset aggregate table, and states Zero's failed/unresolved historical outcomes without turning them into equivalence claims.
- Appendix C retains label-aware oracle wording, distinguishes pre-search from post-search signals, and charges escalated margin-router queries for two searches. It no longer attaches synthetic latency accounting to an apparently measured joint quality/latency point.
- The regenerated evidence file reflects the narrowed shuffle and teacher claims and adds E15's selection and accounting limits.

The stronger two-teacher trained-Zero experiment remains a possible expansion of the scientific scope, not a correctness gate for v11. The current paper can make its narrower practical argument honestly without it.

## Access

Opened `m15/PAPER.md`, selected sections of regenerated `m15/EVIDENCE.md`, and `results/m15_e15_decision_audit.json`. Ran a bounded read-only Python arithmetic check against E15. Relied on the earlier review's inspected method, scripts, and receipts rather than repeating that audit. No raw queries/qrels, protected paths, `work/` content, model runs, or writes to result files. This review file is the only mutation.
