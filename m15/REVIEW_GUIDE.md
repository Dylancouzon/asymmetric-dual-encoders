# Coworker review guide

## What is ready to review

The current manuscript is **v20, Your Index Is Fine, Your Query Encoder Is Not: Query Distillation over Frozen Indexes** on `m15-whitepaper` (v15 and v16 are in git history; `REVISION_PLAN_V17.md` records the
frame, the voice contract, and the agreed claim wording). It is an empirical whitepaper for experienced search engineers. The goal is useful,
credible research that improves Qdrant's reputation in that community. The thesis and content remain
open to challenge; the owner has not accepted this draft.

The owner is reading and revising v20 on 2026-10-02. The abstract, production motivation, and
narration have changed since the recorded reviewer sign-offs. Use the latest commit for the text
under discussion; [LOG.md](LOG.md) records these revisions and their checks.

The branch contains the current paper, its numerical result summaries, methods, experiment code,
figures, original registered outcomes, negative findings, full-history synthesis, author preferences,
and previous reviews with dispositions. All 55 local targets linked by the current manuscript's
evidence map are tracked. That establishes source availability, not a new independent verification
of every scientific conclusion.

**A fresh clone is sufficient for substantive review. It is not a complete, offline rerun bundle.**
Large vectors, model weights, fit matrices, environments, and some raw logs are outside Git. The
cleaned teacher-screen fit list is not in the branch; its recorded hash does not replace the list.
Some experiments would need artifact recovery or reconstruction before an exact rerun. See
[PROVENANCE_AUDIT.md](PROVENANCE_AUDIT.md) for the checked scope and remaining limits.

## Start with the question you want to answer

| Review question | Read first | Go deeper |
|---|---|---|
| Is this clear, interesting, and useful for a production search engineer? | [PAPER.md](PAPER.md) | [Owner expectations](OWNER_PREFERENCES.md), [all potential learnings](LEARNINGS.md) |
| Does a particular claim follow from the experiment? | [Current evidence map](PAPER_EVIDENCE_MAP.md) | [Numerical cards](EVIDENCE.md), the source result named in that row, [methods](MEASUREMENTS.md) |
| Did we choose the right direction and findings from the whole project? | [Editorial rationale and alternatives](EDITORIAL_RATIONALE.md) | [LEARNINGS.md](LEARNINGS.md), its milestone coverage table and named historical FINDINGS/STATUS files |
| Are confirmatory outcomes and unfavorable comparisons visible? | Paper Appendix A | [Nano final run](../results/m10_final_run.json), [Zero final run](../results/m7_final_run.json), [published reserved aggregate](../results/m13_reserved_run.json), [canonical benchmarks](../m21/BENCHMARKS.md) |
| How strong are the ANN and query-precision conclusions? | Paper §3 and Appendix B | [C10/C14/C15 cards](EVIDENCE.md), E2/E16–E18 in [methods](MEASUREMENTS.md), [follow-up options](FOLLOWUP.md) |
| What did earlier reviewers find, and what was fixed? | [V15 dispositions](REVIEWS/2026-10-01-v15-synthesis.md) | [Reader review](REVIEWS/2026-10-01-v15-reader.md), [correctness review](REVIEWS/2026-10-01-v15-correctness.md), [LOG.md](LOG.md) |
| Is a claim already established in prior work? | [Paper references](PAPER.md) | [Primary-source literature notes](RELATED_WORK.md) and the linked papers |
| What exactly is in the branch? | [BRANCH_INVENTORY.md](BRANCH_INVENTORY.md) | [All baseline tracked paths and blob identities](BRANCH_FILES.tsv) |

The old [EVIDENCE_INDEX.md](EVIDENCE_INDEX.md) is a historical candidate list. It contains superseded
claims and status statements; it is not a current reviewer entry point. Older drafts and plans also
remain in history. The current map and synthesis govern how those historical results are interpreted.

## Recommended collaboration workflow

**Creation is deferred.** Dylan will review with Astra in a separate session before creating the
Google Doc. [GOOGLE_DOC_HANDOFF.md](GOOGLE_DOC_HANDOFF.md) records the future creation instructions;
[REVIEW_APPENDIX.md](REVIEW_APPENDIX.md) is its current reviewer-material source. No Doc exists yet.

Use **one Google Doc as the discussion copy**, containing the paper with its existing scientific
appendices, followed by a clearly marked reviewer appendix with evidence links and a repository
index. Google Doc comments and suggestions keep the discussion attached to the text. Git keeps
the experimental record and accepted manuscript revisions auditable.

1. Identify the source commit and draft at the top of the Doc. Evidence links should use that commit,
   so a claim does not silently change underneath an active review. Link the branch separately for
   readers who want subsequent work.
2. Comment on the relevant sentence, table, or figure. For a scientific objection, include the claim,
   the named evidence card or result, and what would change the assessment. C10 has separate FiQA
   and million-passage cards: name the workload as well as the card number.
3. Use suggestions for proposed prose changes. Start broad discussions about thesis, missing
   findings, or additional experiments in an anchored comment near the introduction or conclusion.
   Rewrites and removals are welcome; keep the whitepaper genre and owner goals in view.
4. After agreement, implement targeted changes in `PAPER.md`, preserve the underlying receipts,
   rebuild the PDF, and commit/push a coherent revision. Summarize consequential decisions and their
   rationale in `LOG.md`; owner direction also belongs in `OWNER_PREFERENCES.md`.
5. Update the same Google Doc with targeted edits, then record its new repository source commit.
   Avoid replacing the whole Doc after comments begin: that can detach or lose the useful context.
   Resolve a thread only when its disposition is recorded or the participants agree it is closed.

The Google Doc and repository are **not automatically synchronized**. Its link and source snapshot
will be recorded in the handoff. Before making an edit, check which version you are discussing.
Coworkers who want to change the repo should use individual branches and PRs based on
`m15-whitepaper`, rather than simultaneous direct edits to the shared manuscript branch.

## Using Codex for a review

The following prompt is a starting point; replace the focus with the question under discussion.

> Review the Constella v20 whitepaper on m15-whitepaper for [specific focus]. Read CLAUDE.md,
> instructions-m15.md, m15/REVIEW_GUIDE.md, m15/OWNER_PREFERENCES.md, and the relevant section of
> m15/PAPER.md. Use m15/PAPER_EVIDENCE_MAP.md to select exact result/method files. Use
> m15/EDITORIAL_RATIONALE.md and m15/LEARNINGS.md if challenging the direction or selection across
> the whole history. Distinguish
> scientific evidence from prior agent opinions. Report consequential findings with the claim,
> evidence, effect on the conclusion, and a concrete proposed fix. This is a whitepaper, not a
> tutorial or Qdrant sales piece. Do not run experiments or alter existing result files for this review.
> Never read results/frozen_eval/untouched-*, reserved qrels caches, work/m9reserve, raw closed M18
> confirmation, or sealed M19 confirmation. Never overwrite results/perquery.json. No repo-wide
> content search across results/ or work/: select a named-file allowlist first. Use published aggregate
> receipts for protected evaluations. A filename in the inventory is not authorization to read it.

Ordinary literature rechecking can use the public primary-source links. The ignored `.firecrawl/`
scratch directory is not part of the branch; the committed notes and references are the review trail.

## Boundaries that affect interpretation

- Exact-search relevance, ANN losses, candidate coverage, and latency are different quantities.
  Encoder savings are not equal-quality, end-to-end production speedups.
- Three aligned query paths share Stella's index. The teacher screen compares different teacher
  spaces with their own indexes; it does not authorize swapping unrelated teachers into this index.
- Registered tests and exploratory analyses remain distinct. Negative/unresolved outcomes and
  training/exposure differences are part of the record, including Zero's failed Holm BM25 test.
- Query-bootstrap intervals do not include retraining variance. The repeated precision control
  does not establish native engine latency or a general result across embedding spaces.
- The approximately $95 Nano figure prices the final optimization loop, not the whole build.
- Protected raw evaluations remain excluded from ordinary review. The published aggregates are
  available; this collaboration pass does not reopen those evaluations.

The collaboration package adds navigation and provenance. It does not change experiment results,
upgrade exploratory evidence to confirmation, or certify publication readiness.

## Added 2026-10-01: v17 reviews and pending results

Owner-directed reviews and their dispositions, newest first: `REVIEWS/2026-10-01-fable-voice-v17.md`
and `-astra-voice-v17.md` (voice contract), `-fable-register-v16.md` and `-astra-register-v16.md`
(why v16 read as an article), `-astra-correctness-v16.md`, `-sol-reader-v16.md`,
`-andrey-review-v16.md`, `-astra-e19-e20-reading.md`, `-astra-fable-plan-rounds.md`, and the
pre-spend reviews `-astra-e19-e20-prespend.md`, `-astra-e22-e24-prespend.md`,
`-astra-e22-e24-rereview.md`, `-astra-e25-prespend.md`; complete-draft reviews
`2026-10-02-astra-complete-v17.md` and `-fable-complete-v17.md`. New results: E19 to E25
(`results/m15_e19_*` to `m15_e25_*`; E24 reported from `m15_e24_prospective_amend1.json`), cards C16 to C22.

## Added 2026-10-02: v18, cuts, and final gates

v18 applies the cuts both complete-draft reviewers agreed on (`REVISION_PLAN_V17.md`, "Cut
decisions") and E24 amendment 1 (`MEASUREMENTS.md`; head row n=8). Gates run on v18, in order:
Andrey review (links, figures, claims; nothing essential), Astra narration rounds 1 to 3
(`REVIEWS/2026-10-02-astra-v18-rounds.md`; signed at round 3), Sol reader pass
(`REVIEWS/2026-10-02-sol-reader-v18.md`; four rewrites, number set unchanged), humanizer, and the
voice-contract check (4,195 main-text prose words; both reviewers judged no further cut essential).
A coworker review starts at the PDF and `LOG.md` "2026-10-02: v18, cuts and final gates".

## Added 2026-10-02: v19 full-paper independent reviews

Current v19 includes owner-directed architecture/training exposition, introduction framing and clarity/table revisions. Parallel GPT-6-Astra and GPT-6.1-Sol reports and synthesis are in [v19 synthesis](REVIEWS/2026-10-02-v19-full-synthesis.md). Both distinguish a coherent experiment from an under-explained narrative; recommendations remain pending. The manuscript snapshot is 68509e7, not the subsequently committed review-document snapshot. Prior v18 sign-offs do not certify v19.

## Added 2026-10-02: v20 implementation

Owner keeps the title and authorizes the v19 review changes, with explicit emphasis on interest and value density. [v20 dispositions](REVIEWS/2026-10-02-v20-dispositions.md) records changes, relocations and cuts. The paper now connects the paired studies and trained illustration, aligns outcomes/claims and explains the timing diagnostic. v19 reports remain opinions about their named snapshot; no new independent sign-off is claimed.
