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
