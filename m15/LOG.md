# M15 working log

Branch `m15-whitepaper`, opened 2026-09-17 on Dylan's instruction to start the whitepaper draft
while M20 stage A runs on the local box. Every session appends here: what was done, what was
verified, what was decided and by whom.

Access discipline for all M15 work, including sub-agents: no reads of `results/frozen_eval/untouched-*`,
reserved qrels caches or `work/m9reserve`; no repo-wide content searches across `results/` or `work/`;
named-file allowlists only.

## 2026-09-17

- Created branch `m15-whitepaper` and `m15/`.
- Owner instruction (Dylan, 2026-09-17): draft the paper now; sub-agents allowed for research and
  review; Codex Sol for adversarial review; focus on net-new knowledge; high bar for inclusion;
  the hot-swappable query encoder over one frozen document index is the lead differentiator;
  report failures only where they add value.
- Flagged to Dylan: that last point narrows the CLAUDE.md rule "Keep negative results, failed
  approaches, provenance and limitations alongside successful runs" and the M15 mandate line
  "negative results, limitations". Awaiting his ruling; drafting proceeds with limitations kept
  and failed avenues admitted only where they carry a lesson.
- Launched three research sub-agents: evidence inventory, prior-art positioning, hot-swap evidence.
- Owner decisions (Dylan, 2026-09-17, asked and answered in-session):
  - **Negatives.** Limitations, caveats and contamination disclosure stay in full. A failed approach
    enters the paper only where it changes how a reader must read a headline number. M8, M17, M18
    and M19 drop to one appendix line each. This narrows, and does not delete, the CLAUDE.md rule.
  - **Form.** arXiv-style preprint. A Qdrant whitepaper or blog cut can be derived later.
  - **Git.** Commit and push the `m15-whitepaper` branch as work proceeds.
  - Sub-agents: Dylan authorized "plenty" of them for research and review, which overrides the
    global one-to-two limit for this milestone. Codex Sol is the adversarial reviewer.
- Research sub-agents returned. `EVIDENCE_INDEX.md`, `NOVELTY.md` and `HOTSWAP.md` written from
  their reports, with provenance and a spot-check gate on every number.
- **Novelty finding that changes the framing.** Index-preserving query-encoder swap is prior art:
  LEAF ships it as mixed-checkpoint inference, pyNIFE published the lookup-table construction in
  November 2025, and backward/forward-compatible training name the general goal. The paper keeps
  the swap as its organizing frame and claims the measured frontier, not the mechanism. Raised with
  Dylan at the end of this session because he named hot-swap as the differentiator.
- **Sub-agent error caught and retracted.** The hot-swap audit flagged the 3.4 ms / 4.5 ms edge
  numbers as untraceable. `m9/RESULTS.md` carries them (3.387 ms and 4.469 ms at a 256 MB limit,
  202 MB resident). The audit's allowlist omitted that file. Verified directly.
- Verified against sources myself: the four registered contrasts, the serving table, the DBSF depth
  ladder, the M7 confirmatory family, the artifact sizes, and the teacher-learnability denominators.
- First full draft: `m15/PAPER.md`, sections 1 to 9 plus one appendix, section 5.4 held open for M20.
- Branch pushed to origin. Review round one launched: a number-by-number fact check, a hostile-reviewer
  pass, and a Codex `gpt-5.6-sol` adversarial correctness review. Brief: `m15/REVIEWS/BRIEF-common.md`.
- Review round one returned three passes. Records: `REVIEWS/codex-sol-2026-09-17.md`,
  `REVIEWS/factcheck-2026-09-17.md`, `REVIEWS/fable-hostile-2026-09-17.md`.
- Draft v1 did not survive it. Three problems were structural rather than editorial:
  1. **The paper contradicted itself on artifact size.** Section 6.1 had the table smaller than the
     student; section 6.3 had it 5.9 times larger. The two rows measure different packagings from
     different milestones. v2 reports both with their definitions and withdraws the one-sided claim.
  2. **The LEAF contrast is not a shared-index comparison.** LEAF-asym encodes documents with
     `snowflake-arctic-embed-m-v1.5` at 768 dimensions. Its teacher ceiling on our six is 0.5264
     against Stella's 0.5744, and the established delta is a third of that gap. v2 discloses it and
     reports retention of each system's own tower: 0.979 for LEAF-asym, 0.9256 derived for ours.
  3. **The cost numbers mixed two harnesses.** v2 reports the same-harness collapse, five times as
     encoders to 1.96 times as systems, and keeps the cross-harness figures apart.
- Also fixed: the synthetic index is now disclosed, the quantization quality claim is withdrawn
  pending recall, "tie" is gone, the unresolved count is three and enumerated, the absorbability
  result carries its pooling condition, and the clean-4 headline registration is dated per family.
- Promoted from the evidence into the paper: the fused table system reaching the inference-free
  band with no query-side network, the 19-times rescore trap, the inert objective at 99.75%, the
  subword-fragmentation correlation that moved nothing, and the index-bytes ladder.
- **A deferred deliverable surfaced.** Owner ruling, Dylan 2026-08-30 in `m9/RESULTS.md`: the
  TurboQuant comparison against binary, int8 and fp16, on latency, footprint and recall at up to one
  million documents, was deferred to the whitepaper. Section 6.4 cannot make a quality claim until
  it runs. Section 9 folds it into one experiment that also prices the Stella query tower and
  replaces the synthetic index with real vectors.
- v2 written and committed. Round two launched: a focused Codex Sol re-review and an andrey-review.
