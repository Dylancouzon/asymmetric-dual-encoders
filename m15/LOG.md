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
