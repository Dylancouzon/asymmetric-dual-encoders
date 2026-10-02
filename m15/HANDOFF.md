# M15 handoff (2026-10-01)

Branch `m15-whitepaper`. Read **OWNER_PREFERENCES.md**, **LEARNINGS.md**, the last LOG.md entry,
and **PAPER.md** before revising. The owner keeps challenging purpose, readability, and value;
previous agent passes are not owner acceptance.

For coworker review, start at **REVIEW_GUIDE.md**. **BRANCH_INVENTORY.md / BRANCH_FILES.tsv** list
the baseline tracked content; **PROVENANCE_AUDIT.md** resolves exact source versions and states
which inputs are external/unavailable. A fresh clone supports substantive review, not every exact
rerun. EVIDENCE_INDEX.md is explicitly superseded historical discovery material.
**EDITORIAL_RATIONALE.md** explains the current choice, its strongest objection, the revision
history, and eight alternative directions with evidence, limitations, and what could change the
choice. It is an agent recommendation for coauthor discussion, not owner acceptance.
**Google Doc creation is deferred by the owner** until after a separate Astra review.
**GOOGLE_DOC_HANDOFF.md** has future creation/verification/collaboration instructions;
**REVIEW_APPENDIX.md** has the current discussion appendix source, pinned to `5ce1ced`.
No native Doc was created/uploaded. Refresh its content/links after the separate review before
using it for a new collaboration document.

## Current deliverables

- **PAPER.md v16:** “Constella: Teacher Selection and Search Cost in Query-Side Distillation.”
  Two findings (teacher selection across 26 teachers under two student recipes, E8/E8x/E19; search
  effort for cheap queries across 25 teacher spaces, E2/E20), about 3,500 main words, five main
  figures. Plan and agreed claim wording: `REVISION_PLAN_V16.md`. Gates run on 2026-10-01: Andrey,
  Astra correctness, Sol reader, humanizer; dispositions in `LOG.md` and `REVIEWS/2026-10-01-*-v16.md`.
  v15 (“Lower-Cost Queries over a Fixed Document Index”) is in git history.
- **LEARNINGS.md:** 33 candidate lessons from the whole history (M0–M23), with milestone coverage,
  evidence, production decision, strength, limits, editorial selection, and justified research gaps.
  Unscheduled M16 and incomplete release/archive work are identified as lacking new retrieval
  evidence. Keep this catalog broader than the paper.
- **PAPER_EVIDENCE_MAP.md:** file-level provenance for the current manuscript. Internal repository
  paths are removed from the manuscript; normal literature references and artifact availability
  remain. EVIDENCE.md retains all numerical cards, including material omitted editorially.
- **AUTHORS.json:** provisional owner-supplied author order, Qdrant affiliation, and all four emails.
  The PDF build reads this roster. Query/encoding/loading latency uses ms consistently.
- **latex/paper.tex / paper.pdf:** build with `sh m15/latex/build.sh` (pandoc/tectonic installed).
  Figures use scientific Matplotlib, not brand diagrams. Regenerate only changed figure functions
  with `.venv-mac/bin/python`; the main figures are f4_system and f3_towers, printed as Figures 1/2.

## Direction and owner corrections

The paper's central evidence is the distinction between encoding savings and retrieval savings
with the same document vectors. The teacher/table reversal is a separate construction finding
before choosing a teacher/index. The fixed-index search and fusion results precede that section.
Fusion-depth reversal and actual runtime failures bring useful findings from older milestones into
view. Router/blend failures remain concise. Query precision is a bounded control, not an already
measured native remedy.

Owner rejects “search service,” repository filenames used as paper citations, a verbose inventory,
and a goal unclear to a search engineer. The ultimate audience outcome is interesting, applicable,
worth sharing, and production value. Owner explicitly reiterates **whitepaper, not tutorial**.
The final draft presents findings and interpretation, removing repeated instructional prompts.
A worked absolute-quality/time example is explicitly hypothetical and limited to the measured
rows; it is not a prospectively selected routing policy or recommended SLO.

The full-history audits and Astra synthesis review are in REVIEWS/2026-10-01-learning-*.md.
V15 reader and correctness reviews plus their focused closures are in REVIEWS/2026-10-01-v15-*.md.
Root dispositions are in REVIEWS/2026-10-01-v15-synthesis.md. They resolve identified issues;
none certifies owner acceptance, publication readiness, or a breakthrough.

## Scientific boundaries

- Nano/Zero quality budgets are different, with mismatched training data. BEIR-15 aggregates are
  descriptive. Original registered clean-four/all-six tests and reserved aggregate results remain
  visible, including Zero's failed Holm BM25 test and Nano's unfavorable broader references.
- E2 measures sequential warmed queries on FiQA and a positive-preserving 1M validation diagnostic.
  Encode-plus-search is a sum of separately measured per-query phases. It is not a request stopwatch,
  throughput, concurrent failover, or equal-absolute-quality speedup.
- E8/E8x screens one ridge-table recipe over 26 checkpoints (10 registered, 16 exploratory), each
  with its own teacher index. It does not select trained Zero/Nano or permit unrelated teacher swaps.
- E16's strong native FiQA interaction has a weaker primary E17 replica. E18 fixes document sign
  codes and tests sign versus full-query candidate scoring, without a graph or native latency.
  Larger absolute Zero gains have more miss headroom; mechanism/proportional sensitivity unproven.
- $95 is final Nano optimization at a historical rental rate, not complete reproduction cost.
- M12 fusion-depth curve is descriptive development evidence, bm25s with Lucene settings and
  fixed convex weight .8. The DBSF recommendation changed on implementability grounds after prior
  evaluation. Native BM25 substitution and equivalence are not established.
- Runtime failures are historical candidates/defects, not claims that current released fp32 paths
  fail. Small internal specialization/patch studies do not establish an improved encoder.

## Next work

1. Continue editorial selection against LEARNINGS.md and the owner's latest feedback. Do not fill
   the paper with every candidate or default back to a methods inventory/checklist.
2. For stronger research, choose a specific claim first. Native magnitude-preserving query scoring
   at matched recovery/relevance and measured cost has the clearest deployment value. Account for
   collection rebuild/graph variation. Another teacher space would test reach beyond Stella.
   FOLLOWUP.md preserves these options. No full retraining is authorized or needed by this plan.
3. Before actual publication, reconfirm the provisional roster only if it changes, and verify release/licence state. No public paper
   publication happened during this rewrite. Twenty literature references retain their existing
   primary-source audits in RELATED_WORK.md.

## Execution and access

All existing result payloads remain immutable. No new experiment, teacher/document encoding,
model training, rental, or protected-data access occurred in the full-history rewrite. M15 historical
cloud spend remains $22.79; Runpod pod `k3aee2m68765em` is stopped. Container `/opt/m15-venv` is lost
on restart; rebuild from `m13/cloud_requirements.txt` with uv at
`/home/dylan/.local/share/m13-uv-0.12.5/bin/uv`. Remote worktree `/home/dylan/m15run`.

Never read `results/frozen_eval/untouched-*`, reserved qrels caches, `work/m9reserve`, raw M18
confirmation, or sealed M19 confirmation. Never overwrite `results/perquery.json`. No repo-wide
content searches over results/work. Published aggregate receipts are the allowed reserved evidence.
Commit and push coherent batches frequently; no force-push. Models: Sol writer/reader, Astra
adversarial correctness reviewer, per mandate and owner permission.

## Mac pitfalls learned this milestone

- One MPS job at a time; an MPS encode hung for six hours in a Metal wait (process state `UN`). Watch
  shard timestamps, not only the log. Identify Mac jobs by pid and start time: `ps` truncates the
  command line, which once made a live job look dead and caused a duplicate run.
- Latency runs need an idle machine; background load moved FiQA encode latencies by up to 1.7x.
- `pgrep -f` over SSH matches its own command line; use a bracket pattern (`[e]8x_towers`).

## Update 2026-10-01 (owner rounds 4 to 6): v17 and pending experiments

- **PAPER.md is v17**, "Your Index Is Fine, Your Query Encoder Is Not: Query Distillation over Frozen Indexes":
  research questions and hypotheses first, RQ1 (width-conditioned retention, E21) and RQ2
  (cross-space search effort, E20), Constella as the worked instance with build cost and CPU hours
  per million queries, four appendices. Complete: E22 to E25 are in; prose about 5,100 words against
  the contract's 4,000, with two reviewers' cut lists held for the owner. Plan, voice contract, and
  agreed claims: `REVISION_PLAN_V17.md`.
  Reviews run on v16 and v17 are listed in `REVIEW_GUIDE.md`; dispositions in `LOG.md`.
- **New results:** E19 to E25 (`results/m15_e19_*` to `m15_e25_*`). E24 is reported from
  `m15_e24_prospective_amend1.json` (head n=8, table n=7; the first assembly with n=7 in both rows is
  kept, see E24 amendment 1 in `MEASUREMENTS.md`). Cards C16 to C22 in EVIDENCE.md. Figures f7, f8, f9.
- **Open:** the owner's decision on the merged cut list (`LOG.md`, 2026-10-02); then humanizer,
  PDF rebuild, voice-contract recheck.
- **Pod:** `k3aee2m68765em` holds the fit list and all caches; its host had no free GPU from 16:27
  local on 2026-10-01; `work/m15/podwait/k3aee2.log` records the retry loop. The venv on container
  disk must be rebuilt after each restart (`m13/cloud_requirements.txt`, uv with
  `--index-strategy unsafe-best-match`); use the musl Qdrant build and `E20_STORAGE=/opt/...`.
- **Owner decisions this day:** standing commit/push approval; $150 budget; science-first title;
  frozen-index frame; hot-swap goes to the M22 model card; include everything valuable, reviewers
  cut together. Session pod spend so far about $6.

## Update 2026-10-02 (owner's final round): v18 signed

- **PAPER.md is v18.** Both reviewers signed: Fable's sign-off blocker (E24 head row) is closed by
  E24 amendment 1; Astra signed at round 3 after the comparative-scope corrections. Cuts and
  gates are recorded in `REVISION_PLAN_V17.md` and `REVIEW_GUIDE.md`; dispositions in `LOG.md`.
- **Open before circulation:** archive the cleaned fit list and the one-million-passage identifier
  list (identified by hash in the artifact statement); the owner reads the PDF; author roster and
  contact details in `latex/build.sh`.
- **Pods:** both stopped; session spend about $11; nothing running.

## Update 2026-10-02: owner reading and narration pass

- The owner requests a critical IR-research collaborator, a standalone problem-first abstract,
  operational migration motivation, and narration that states findings directly. All direction
  is in OWNER_PREFERENCES.md; subsequent iterations should read its owner-reading section.
- Current v18 includes revisions after the reviewer-signed snapshot: an abstract opening on
  document reuse and recurring uncached-query cost, a production-migration introduction, and
  a narration pass. Main prose is about 19% shorter by the same before/after counting method;
  scientific tables and the manuscript's numeric-token set are unchanged. Secondary detail
  is retained in appendices. The log records dispositions; prior sign-offs are historical.

- **Further owner feedback:** the abstract remained hard to parse; its rhetorical question was
  removed and its argument rewritten in four paragraphs. Manuscript terminology is now embedding
  dimensionality, defined in the introduction. This revision is logged; the owner has not accepted
  the abstract merely because it was rewritten.

## Update 2026-10-02: architecture and training exposition

- Main methods distinguish both closed-form screen students from released Zero and Nano; Appendix B supplies their full recipes before the long encoder roster. Abstract remains the three-paragraph continuous-argument revision from commit 16de036; the older four-paragraph note above is historical.
- Read OWNER_PREFERENCES.md before further writing. Fresh-context GPT-6-Astra, explicitly selected after Fable proved unavailable, reviewed the construction prose; bounded scope and dispositions are in REVIEWS/2026-10-02-astra-construction-writing.md. This does not renew a whole-paper sign-off.
- Corrected fitted-table pooling and initialization against code. Zero's real two-phase objective supersedes the historical L2 description for manuscript purposes. A separate future release-card correction is needed in m11/release/MODEL_CARD.md; no model card was changed or published in this batch.
- Prepared training texts are described by source/file manifests, not bundled. Fit-list and timing-passage identifier archival work remains open before circulation. No new measurements or result payload changes.

## Update 2026-10-02: introduction premise

- Owner-approved introduction now starts with query/document computational roles and independent query-encoder choice. Migration is concise supporting context after the scientific motivation. Preserve this order in future iterations; OWNER_PREFERENCES.md records the direction.

## Update 2026-10-02: v19 clarity and table pass

- PAPER.md is now draft v19 after owner-directed clarity work. §4.4/Table C separates candidate pools, explains the score gap and three selection rules, and distinguishes ranking from top-choice success. Chosen names remain in Appendix B.
- Tables use shorter labels; Table D separates relevance and neighbor recovery. Table F's within-workload correlations remain in Appendix C. Preserve the data and readable column widths rather than reverting to text-heavy combined cells. Search measures and validation splits are explained before results.
- This is a prose/layout revision with no new experiments or whole-paper sign-off. The older signed v18 snapshot is distinct; current preferences and detailed dispositions are in OWNER_PREFERENCES.md and LOG.md.

## Update 2026-10-02: independent v19 full-paper reviews

- Parallel fresh-context GPT-6-Astra and GPT-6.1-Sol read the entire v19 at 68509e7. Reports and synthesis: REVIEWS/2026-10-02-{astra,sol61}-v19-full.md and REVIEWS/2026-10-02-v19-full-synthesis.md.
- Both find a coherent research object but want clearer synthesis and Constella's illustrative role. Outcome/scope mismatches and omitted 1M diagnostic explanation need repair. Optional cuts and a title disagreement remain proposals, not owner decisions.
- Manuscript/PDF unchanged by review. Await reading discussion before implementing the recommended revision; no new experiment requirement or publication sign-off.
