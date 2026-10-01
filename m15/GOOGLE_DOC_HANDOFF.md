# Instructions for the future collaboration Doc

## Current instruction

**Do not create the Google Doc yet.** Dylan will read the paper and review it with Astra in a
separate session first. These are creation/handoff instructions for afterward, when the owner
asks to proceed. No Google Doc has been created or uploaded by this collaboration pass.

Start the separate review at [REVIEW_GUIDE.md](REVIEW_GUIDE.md). Challenge the framing using
[EDITORIAL_RATIONALE.md](EDITORIAL_RATIONALE.md) and the broader [LEARNINGS.md](LEARNINGS.md).
The current draft and rationale are recommendations under review, not coauthor consensus.

## Recommended arrangement

Use one Google Doc for anchored comments, replies, and suggestions. Keep accepted revisions,
scientific receipts, and consequential decisions in Git. There is no automatic synchronization.

The future Doc should contain:

1. The paper title, author roster from [AUTHORS.json](AUTHORS.json), draft/version label, and an
   exact repository source commit with a link. Use the post-review version, not a cached v15 export.
2. The full [PAPER.md](PAPER.md), including both scientific appendices, references, all tables,
   and the figures actually referenced in that version. Use editable text and native tables,
   rather than screenshots of the PDF. Preserve measured values, signs, units, and qualifications.
3. A separate **Reviewer appendix**, clearly excluded from the publication manuscript. The
   current [REVIEW_APPENDIX.md](REVIEW_APPENDIX.md) is a starting source, not a final frozen Doc.
   It includes the direction/alternatives summary, current claim-to-source map, every numerical
   evidence card, evidence-availability limits, broader-learning links, and branch contents.
4. Links to the full rationale, learning catalog, reviewer guide, provenance audit, branch
   inventory, and owner preferences. Evidence remains available in the repository; do not paste
   1,841 file paths or raw logs into the Doc.

Use normal paper citations in the manuscript. Repository paths and review-process instructions
belong in the reviewer appendix. Do not turn the whitepaper into a tutorial.

## Preparation after the Astra review

- Apply agreed changes to the manuscript and relevant support files. Record significant decisions
  and reasons in `LOG.md`; update owner preferences only for actual owner direction. Rebuild and
  check the paper PDF if the manuscript changes, then commit/push a coherent snapshot.
- Choose that committed snapshot as the Doc's source. Refresh the reviewer appendix's map,
  evidence-card list, rationale, and active snapshot URLs against it. Current appendix links are
  pinned to `5ce1ced`; they must not silently accompany a different draft.
- Preserve deliberate historical source-version links in `PROVENANCE_AUDIT.md`. Those point to
  executed script/method bytes and must not be blanket-replaced with the newest commit.
- Confirm the provisional authors from `AUTHORS.json`; use the supplied names/emails and current
  order. Do not infer authorship or sharing roles from a review comment.
- Use the connected Google Drive/Docs tools when available. They were available in this session;
  no plugin installation was needed. Follow the current Docs/Documents skills for creation,
  import, native formatting, and verification. Codex desktop is the owner's offered fallback.

## Creation and verification when authorized

1. Create a new private working Doc or use a destination explicitly supplied by the owner. Preserve
   editable headings, paragraphs, tables, hyperlinks, and the paper's actual figure images.
   If using a DOCX import, keep staging temporary and follow the skills' title sanitization,
   rendering, import/readback, and native normalization steps.
2. Read back the created Doc. Verify the complete manuscript, references, all active figures and
   tables, author information, version/source commit, and reviewer appendix. Check numerical values
   against the source. Inspect rendered pages for clipping, table/figure breaks, missing glyphs,
   and unreadable links; do not claim visual verification from text readback alone.
3. Check the actual sharing state. Do not make the draft public or send invitations without the
   owner's instruction. Recommended starting role is **Commenter** for review; use **Editor** for
   coauthors expected to make suggestions/edits if the owner wants that workflow. Repository access
   does not automatically grant Doc access.
4. Record the observed Doc URL/id, source commit, source-file hashes, verification result, and sharing
   state in a small committed collaboration receipt. Add the link to `HANDOFF.md`/`README.md`.
   State any unavailable verification plainly. Remove temporary staging after successful creation.

## Iteration once comments begin

Use one discussion copy. Apply targeted updates to preserve comment anchors; do not recreate the
whole Doc on every revision. Read the live document/comments before writing and respect collaborator
changes. Implement accepted manuscript changes in Git, record their rationale, rebuild as needed,
then update the same Doc and its source-commit label. Resolve threads when their disposition is
recorded or participants agree, not merely because an agent produced a new draft.

For substantive evidence objections, identify the claim, exact result/method, and what would change
the interpretation. For a new experiment, select the claim first; more compute is optional, full-model
retraining remains outside the owner's instruction. Protected raw evaluations remain excluded under
CLAUDE.md even when a coworker has repository access.
