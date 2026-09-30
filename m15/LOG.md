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
- Round two returned. `REVIEWS/andrey-2026-09-17.md` and `REVIEWS/codex-sol-rereview-2026-09-17.md`.
  - andrey-review: **fix-then-ship, and not close.** One same-day finding, a preprint with no
    bibliography, now fixed. Its other three findings are the open measurements the paper names.
  - Codex Sol re-review: 11 of the 13 round-one findings confirmed fixed; two dispositions
    overclaimed and are now real. Five new defects found and fixed, including displayed sums in the
    latency table that do not add (the record rounds components, totals and ratios independently)
    and two sentences that were false as written, "holds no state" and "changes no stored bytes".
- Humanizer pass run last, as the gate order requires. No puffery vocabulary, no em dashes, no curly
  quotes. The real finding was density: 15 uses of "X rather than Y", the negative-parallelism
  shape. Seven rewritten as positive statements; the rest carry a genuine contrast.
- The Codex writing pass is deliberately held. It restructures a whole piece, and running it before
  M20 and the open measurements land would churn prose that has to change. It runs when the content
  is final, then the humanizer again after it.
- **State at the end of this session.** Draft v2 plus references, ten sections and one appendix,
  reviewed by four independent passes and pushed on `m15-whitepaper`. Three numbers block a
  shippable paper: the Stella query tower's latency under the common serving protocol, TurboQuant
  recall against binary, int8 and fp16, and a swap against a real persistent collection. The Section
  9 experiment produces all three in one run on the M5 Pro, with committed vectors, no training and
  no protected access. It awaits Dylan's go.
- Owner decisions at session close (Dylan, 2026-09-17):
  - **Hold the Section 9 experiment until M20 finishes.** One machine at a time. The paper stays at
    v2 and M15 continues on sections that need no new numbers. The experiment keeps its design in
    Section 9 and in `HOTSWAP.md`, and it runs after M20 closes.
  - **Hot-swap is the frame, not the claim.** The current draft stands: LEAF and pyNIFE cited as
    prior art in the introduction, the contribution stated as the measured frontier.
  - **One paper, both tiers.** The registered default holds, and it is the only framing that carries
    the knee result.

## 2026-09-30

Session on `main`, then on this branch. M20's measurements had landed (`results/m20_beir15_run.json`,
`results/m13_reserved_run.json`).

- **Merged `main` into this branch** (`75916db`, no rewrite). ROADMAP.md and the plain-English page
  conflicted; `main`'s newer M20 text was taken for both. The branch now differs from `main` in `m15/`
  and in this session's rule edits only.
- Consultants, all within the access rules (Astra logs audited: named files only, no `work/`, no
  protected paths, no reserved-dataset web searches). Verbatim returns in `REVIEWS/2026-09-30-*.md`,
  briefs in `REVIEWS/briefs-2026-09-30/`:
  - Sonnet: literature and published reference numbers; repository survey (build recipe, harness
    reuse, local artifacts).
  - Fable: shape of the v3 plan; research-paper shape; qualitative novelty review.
  - Codex `gpt-6-astra`: adversarial review of v3; evidence rating of the seven findings; qualitative
    novelty review.
- **v3 plan (comparator framing) rejected by the owner:** "You seem to write this paper like a
  comparison, which is NOT what I want... Our goals in this project are not the goals of the
  whitepaper."
- **Owner decisions (Dylan, 2026-09-30, asked and answered in session):**
  - The paper is a research paper. Changed the CLAUDE.md rule "The paper's headline is the registered
    clean-4" (owner chose "change the rule"): the paper leads with findings; the registered test stays
    in full in the evaluation section. `instructions-m15.md` carries a dated amendment. The frozen
    `m20/beir15_registry.json` `reporting.headline` line is left unchanged because its hash is pinned
    in result files; the ruling overrides it for presentation only.
  - Lead: retention as query compute drops. The ANN question ("does a cheaper query encoder make ANN
    search harder?") was briefly proposed as the lead after the novelty reviews; the owner asked
    whether it deviated from the research goal, and it moved to one section of the system-cost
    chapter.
  - Retraining without re-ingestion and use-case-specific encoders: in the discussion section at the
    size the evidence allows.
  - Machines: all measurements on the Mac; no second benchmark machine (gcloud VM considered and not
    needed once the load test was cut). E8 runs on Runpod.
  - Approved: E8 (tower generality on public BEIR, closed-form tables, no training) and figure F1.
    Deferred: the closed-form table tool and the browser search-as-you-type demo.
  - Logged for M22: the Zero card's "L2 regression" training description is wrong
    (`instructions-m22.md` item 3).
- **Review outcomes and dispositions.** Both novelty reviews: "partly". Applied: ANN question as a
  system-cost section; routing as a result (E5, E6); cosine-is-not-the-metric and fusion depth in the
  body; load test, container demo and "no ML runtime" cut; coverage compressed; absorbability limited
  to affine transforms and fixed token weights; precedent (Cho and Hariharan 2019) cited for finding 2.
  Astra's number corrections are in `EVIDENCE_INDEX.md`, "Corrections 2026-09-30". Not adopted:
  Fable's suggestion to drop +0.017648 from the abstract (the owner's rule change places the registered
  test in the evaluation section instead); Fable's statement that Zero is a closed-form ridge solve
  (Zero is trained; the ridge fit is the tower screen).
- **Errors found in the v2 draft**, now in `EVIDENCE_INDEX.md`: the M9 system-latency "nano" was not
  Nano and its index and table were synthetic; Nano inherits bge-small's MS MARCO exposure; Zero trained
  on FEVER-train; KL/entropy mix-up; the 0.009 prediction belongs to Zero.
- **State.** Plan v5.2 agreed (`PLAN.md`); owner artifact at
  https://claude.ai/artifact/U2Ph1TXSUmLDXG7bHXRKkN. Next: `HANDOFF.md`.
- **Adversarial plan review before handoff** (owner request): Codex `gpt-6-astra`,
  `REVIEWS/2026-09-30-astra-plan-review.md`. Verdict "not ready": one P0, five P1, two P2, all applied
  in PLAN.md v5.3. P0: E8's fit list must be M8's cleaned `work/m8_trainq_texts.json` (on the WSL box),
  not M7's 349,934-query superset with 4,582 protected-query hits; E8 is blocked until it is recovered.
  P1s: E8 narrowed to "does a dev-selected table ranking predict the six-set ranking under one
  recipe"; E2's subset labelled a positive-preserving diagnostic; F1 keeps one workload per axis; E2
  gets loss targets fixed in advance; E6 gets a frozen Nano budget and routing baselines. The paper's
  new-knowledge risk Astra names: a retention table without a transferable decision rule.
