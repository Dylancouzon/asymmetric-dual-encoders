# Owner decisions and preferences for the paper

Updated 2026-10-02. This is the durable record of Dylan's instructions for iterations on the M15 paper. Read it with `instructions-m15.md` and `HANDOFF.md` before revising. Entries below paraphrase the owner's messages; tentative ideas remain tentative. Agent recommendations and scientific conclusions belong in reviews and evidence, not in this record.

## Purpose and audience

- **Community credibility is the goal.** Improve Qdrant's image within the search engineering community through useful research. The goal is not to sell Qdrant.
- **Aim for a paper people find worth reading.** The research preview has attracted some search-engineering attention. The paper should have a clear purpose, an interesting finding, and a reason for that audience to care.
- **Write for a seasoned search engineer, not a project insider.** Do not assume familiarity with every embedding model, technique, or prior paper. Define models such as Stella at first use, including in the standalone abstract, and explain their roles before comparing numbers.
- **Readability matters even in a whitepaper.** The owner finds the current narration hard to read and jargon heavy. A narration pass must improve the flow and explanations, not merely expand acronyms or add a glossary.
- **Use the field's language.** The owner specifically rejects “search service” as unfamiliar terminology. Prefer concrete descriptions of query encoding, retrieval, indexes, and production systems.
- **The manuscript must stand alone.** Internal repository filenames are not suitable citations or explanations in the paper. Keep detailed file-level provenance in the accompanying evidence package; use normal references and an artifact-availability statement in the manuscript.
- **Make the reader's benefit explicit.** The ultimate goal is for a search engineer to find the paper interesting, applicable, worth sharing, and useful for improving a production system. A verbose inventory without a clear lesson does not meet that goal.
- **Keep the genre a whitepaper.** The owner explicitly reiterates that this is a whitepaper, not a tutorial. Lead with research questions, methods, findings, and interpretation. Production relevance does not require repeated instructions, deployment checklists, or a how-to structure.
- **Keep units consistent.** The owner flags milliseconds mixed with microseconds. Use milliseconds consistently for query timing throughout the manuscript and figures.

## Settled answers: do not reopen

The owner has given these answers several times. Agents and reviewers must not raise them again as open questions. To disagree, state the disagreement once to the owner with new evidence; do not re-propose it in a review or plan.

- **Why keep the index instead of re-embedding with bge-small or LEAF (2026-10-02, owner, repeated).** One shared index lets each request choose its query compute. In e-commerce search, for example, Zero encodes while the user types, Nano when typing stops, and Stella when the user submits. bge-small and LEAF each need their own document index, so the same choice would need several indexes and ingestion pipelines. The paper gives this answer wherever the re-embedding comparison comes up, and the registered bge-small/LEAF contrasts stay reported in full in the appendix. Evidence boundary: per-path quality and encoding cost are measured (Table 1), and all three paths were verified against the same populated collections; no typeahead or e-commerce workload was evaluated, so the example describes an operating policy, not a measured result. This supersedes, for this purpose, round 6 item (4) and the v18 cut of the typing/pause/submit sentence: timed loading and failover stay model-card material, but per-request choice of query path is the paper's reason to keep the index.

## Authors

- The provisional author list, in the order supplied by the owner, is Dylan Couzon, Evgeniya Sukhodolskaya, Kumar Shivendu, and Andrey Vasnetsov, with Qdrant affiliation. Names and the supplied contact emails are maintained in `AUTHORS.json` and used by the PDF build.

## Editorial freedom and research scope

- **Existing content is expendable.** The owner is not attached to the current thesis, sections, or findings. Remove material with little relevance or importance; challenge assumptions and ideas.
- **Review the whole project before settling on an angle.** The owner challenges an emphasis on one aspect that hides the rest of Constella. Its interchangeable query paths and minimal-compute option should be considered for their actual relevance to the paper.
- **Maintain a synthesis of all milestone learnings.** This branch must contain a file covering potential lessons from the entire research history, including evidence and limits, so the owner and agents can choose paper content during subsequent reviews. This is broader than the findings added on the paper branch.
- **Independent research is authorized.** The owner gives broad discretion to improve the paper and permits cheaper subagents for research.
- **Additional work is welcome when justified.** There is no major publication rush. Some compute is acceptable if it materially strengthens a finding or makes the paper more interesting.
- **Do not retrain the whole models.** Additional analysis, inference, or bounded experiments should respect this constraint.

## Scientific expectations

- **Challenge rather than flatter.** The owner explicitly welcomes disagreement about angles, assumptions, importance, and the evidence.
- **Explain what the paper contributes.** The owner was unsure of the earlier paper's goal and was unimpressed by its state. A careful inventory of measurements is insufficient without a clear question and useful conclusions.
- **Findings must withstand a reasonable skeptic.** Distinguish controlled and repeated findings from anecdotal examples or narrow diagnostics. State how far a result can generalize and which units were actually replicated. The owner does not require convincing every possible skeptic.
- **Keep claims proportional to evidence.** Do not present an illustrative workload, recipe, or model family as a universal law. Broadening a claim may require an additional independent model space or workload, rather than more queries from the same setup.

## Tentative ideas, not mandatory theses

- **Training cost and approachability.** The owner suggests that the cost of training small models, and whether others can do it, might be interesting. This is a possible angle, not an instruction to make the entire paper about cost or promise turnkey reproduction.
- **Teacher differences.** The owner points to evidence that teachers are not interchangeable in their suitability for these small encoders. Its significance and scope should be evaluated, rather than assumed.
- **Constella capabilities.** Hot swapping and what the owner calls “zero latency options” might belong in the paper. The intent is to assess their fit and value, not to claim literal zero retrieval time or instantaneous model loading.
- **External comparators.** The owner questions whether bge-small and LEAF, selected as project targets, give a balanced paper comparison. A full model resweep may cost too much; whether and how to include these references remains a question. This is not an instruction to remove unfavorable results or to call these two models universally best in class.
- **Quantization.** The owner proposes it tentatively and worries about fit and fair comparisons. This authorizes investigating a useful controlled question, not adding a broad quantization sweep or treating the suggestion as a required headline.

## Auditability and iteration

- **Commit and push often.** Coherent changes, methods, results, and dispositions should remain auditable in the repository.
- **Preserve the owner's preferences across sessions.** The owner explicitly requests a persistent decision record so repeated iterations do not repeat earlier mistakes. Append new direction or mark superseded direction here; keep the detailed work history in `LOG.md`.
- **Do not turn agent judgments into owner decisions.** For example, calling v13 a good applied study or recommending native precision validation is an agent assessment. It is not an owner endorsement of the draft or approval of a final thesis.
- **Make coworker review easy.** Coworkers have repository access and Codex. The owner wants them to have the paper, all relevant evidence, and a clear inventory of the branch, with a place for comments and discussion. A Google Doc with a reviewer evidence appendix is the owner's suggested collaboration surface; accepted work must remain auditable in Git.
- **Future Doc capability.** The owner permits installing a plugin or MCP if necessary and offers Codex desktop as a fallback. The connected Google Drive plugin is available; no installation is needed. This is not authorization to email invitations or publish the draft publicly.
- **Defer Google Doc creation.** Latest direction supersedes immediate creation: write the instructions, finish logging, and leave the branch clean. The owner has read the paper and will do a separate Astra review before creating the Doc. Wait for a later instruction to create it.
- **Explain the editorial choice and expose alternatives.** Reviewers need the reasoning behind the current direction and other approaches considered, so they can question the selection, emphasize a different finding, or recommend another thesis. The rationale is supporting review material, not a requirement for coauthors to endorse the current direction.

## Checks before the next draft

1. Can an engineer explain what Stella, Zero, Nano, and Constella are after the opening, without consulting another file?
2. Does each experiment explain what stays fixed, what changes, what is measured, and what decision the result supports?
3. Are changes in workload, quality target, or timing protocol explicit before comparing numbers?
4. Are exact retrieval quality, candidate coverage, encoding time, and full search cost kept distinct?
5. Does each main conclusion state its reach: capability, repeated observation within this family, recipe-specific result, or exploratory diagnostic?
6. Does the main narrative lead with a useful question and findings, with implementation inventories and project identifiers subordinate to the explanation?
7. What can a search engineer decide or do differently after reading it, and which evidence supports that advice?
8. Was the selection checked against the full milestone synthesis, and can the manuscript be read without opening repository files?

These checks operationalize the owner's readability and evidence concerns; they do not add an approval gate or authorize new experiments by themselves.

## Dated changes

### 2026-10-02, owner reading feedback

- Owner authorizes adding a coherent architecture and training account for the fitted students and released Zero/Nano models, with concise main-text explanations and reproducible appendix recipes. Original encoders need identities, relevant properties, and references. Owner requests a fresh-context Fable collaborator with all preferences; Fable is unavailable in the sub-agent tool, and owner explicitly chooses GPT-6-Astra as the replacement.

- Owner says the rewritten abstract feels like unrelated sentences. The abstract needs a continuous argument connecting motivation, evidence, and implication; grouping result summaries into paragraphs is insufficient. This is continued criticism of the draft, not acceptance of earlier rewrites.

- Owner finds the abstract still awkward and hard to parse after the narration pass, flags its rhetorical question about encoding savings and approximate search as a poor fit, and questions “vector width” as unfamiliar terminology. Revise the abstract as a standalone argument and prefer established embedding terminology; do not treat sentence-level shortening as sufficient evidence of readability.

- Owner finds v18's abstract impossible to understand without first reading the paper. It must answer why the reader should care and stand alone. Owner requests the rewrite after discussion of a problem-first abstract that explains the question, study, findings, and practical implication with fewer numerical details.
- Owner explicitly asks for a critical IR-research collaborator in this session, not agreement or flattery. Challenge proposed changes when the evidence or scientific argument does not support them.
- Owner rejects the introduction's BEIR document-encoding duration as representative of a production workload. Prefers motivation grounded in migration time and cost, availability, risks, infrastructure challenges, and added load; estimates are a possible way to explain those costs, not measured production results.
- Owner suggests motivating the abstract through reuse: an expensive document representation can serve many searches, while a query may be used once. Authorizes inclusion if it fits. Implemented as amortized document-encoding cost versus recurring uncached-query encoding, without assuming a reuse count or calling large query encoders wasteful.
- Owner requests a narration pass questioning whether each statement is needed, and flags a tendency to state what things are not or do not do. Prefer direct findings and their interpretation; remove redundant negative framing and repeated defensive qualifications. Essential distinctions and observed negative results still need accurate reporting. Owner explicitly requests that these preferences and revision decisions be logged so later iterations do not revert them.

### 2026-10-01

- Owner-directed independent review and rewrite, permission to remove low-value content, community credibility over selling Qdrant, frequent commits/pushes.
- Cost/approachability and teacher differences proposed as possible angles; owner questions whether the paper has a clear goal.
- Owner asks whether Constella's switching and minimal-compute capabilities fit, and challenges overfocus on one part of the project.
- Justified additional research/compute welcomed; no full-model retraining and no major deadline pressure.
- Quantization proposed with uncertainty about relevance and comparability.
- Owner identifies undefined Stella, asks for a narration pass, and clarifies the audience's prior knowledge.
- Owner says the paper is generally hard to read and jargon heavy, then challenges anecdotal evidence and asks for some generalizability under skeptical reading.
- Owner questions the relevance and balance of the selected bge-small/LEAF comparators, and the feasibility of rerunning a much broader roster.
- Owner requests this persistent record to prevent repeated mistakes.
- Owner rejects “search service,” internal file references, verbosity, and an unclear goal in v14; challenges whether the draft draws on the entire repository rather than this branch alone.
- Owner requests a persistent synthesis of all potential milestone learnings for editorial selection and explicitly authorizes subagents and an Astra reviewer.
- Owner defines the ultimate reader outcome: interesting, applicable, worth sharing, and value for a production system. No current draft is endorsed by this instruction.
- Owner reiterates that the deliverable is a whitepaper, not a tutorial; practical value must stay within an empirical research narrative.
- Owner provides Dylan's email and three additional authors with their emails, for now; recorded in AUTHORS.json. Owner flags inconsistent ms/microsecond units.
- Owner asks whether coworkers have all evidence, requests easy collaborative review with comments/back-and-forth, and suggests a Google Doc containing the paper and evidence/branch inventory appendix. Coworkers have Codex and repository access.
- Owner permits plugin/MCP installation to create the Doc, with Codex desktop as a fallback if needed.
- Owner asks for the direction's rationale and alternative approaches to be available to reviewers, enabling them to make their own informed editorial decisions.
- Owner explicitly defers Google Doc creation, requests instructions and a clean/logged repository, and will review with Astra in a separate session first. No Google Doc is created or uploaded.

### 2026-10-01, owner review with Astra and Fable

- Owner states the reader outcome: the paper should make people want to use the Constella models, or apply the findings to their own indexes and replicate what we did to build their own family. Coherent and applicable; not a benchmark reading or a sea of numbers.
- Owner rejects Nano-versus-bge-small as a main-text framing: the selling point is the family and the shared index, and Nano alone is not the proposition. Registered reference contrasts still appear in full in the appendix (instructions-m15).
- Owner questions the value of section 4, wants section 5.2's use-case specialization idea explored, and asks why section 6 lists repaired bugs. Both reviewers answered in `REVIEWS/2026-10-01-*-owner-*.md`; the agreed proposal is `REVISION_PLAN_V16.md`, not yet owner-approved.
- Owner is willing to run further experiments on the Mac and cloud GPUs to strengthen the paper. Astra is given more authority on narration, Fable on technical correctness; both weigh in on every decision.

### 2026-10-01, owner round 3 on experiments and framing

- Owner judges the paper thin on whitepaper-grade findings. Goal restated: people should talk about the techniques; credibility in IR research; model usage numbers do not matter. Findings that disagree with existing literature are wanted.
- Owner rejects the native scalar8 precision experiment (a knob, not knowledge) and defers the domain-fit table to future work. Owner approves experiment A (cross-recipe teacher screen) and accepts B (cross-space ANN effort) on Astra's agreement, which was given. Plenty of A100 budget remains within the recorded ceiling.
- Owner asks that cost efficiency and applicability to an existing index without re-embedding be more central. Recorded technical boundary: QED, EmbedDistill, LEAF, and pyNIFE also keep the document tower frozen; the paper attributes that and centers closed-form construction cost as the enabling method.
- Owner clarifies that the repository's training-data rules protect shipped weights; evaluation and validation use of datasets is not restricted by them. The reserved four remain known-test under CLAUDE.md regardless.
- Owner asks that every decision be logged per repository convention so ideas are not re-litigated or reverted.
- Astra has more authority on narration, Fable on technical correctness; both weigh in on every decision.
- **2026-10-01, approval.** Owner approves `REVISION_PLAN_V16.md` including experiment B, with a $150 compute budget (slightly over is acceptable). Multiple pods exist; all prior experiments on them are finished, so concurrent work and renting more GPUs to go faster are permitted.
- **2026-10-01, standing approval.** Owner grants everlasting commit and push approval for this project on the working branch. Coherent batches, no force-push, research history preserved.

### 2026-10-01, owner round 4 decisions

- Owner: v16 reads like an article and is not scientifically exciting; "Stella retains 100%"
  against itself is absurd. Two independent reviews (Fable, Astra) and the E21 analysis followed.
- Decisions: title is science-first ("Teacher Selection and Search Cost for Closed-Form Query
  Students over a Frozen Document Index"); E22, E23, E24 approved (about $25 to $35) with the
  pre-spend review rule; register rules in `REVISION_PLAN_V17.md` agreed.
- Owner's standing warning: revisions have oscillated between a boring benchmark reading and a
  blog post. Neither is acceptable. The target is a research paper that is rigorous and interesting
  at once: research questions and hypotheses up front, a surprising structural result carried by
  tables with intervals, one narrative thread, no inventories, no advice voice.
- **2026-10-01, round 5.** Owner proposed that the insight is reducing query cost over a frozen,
  possibly immutable index, and that teacher selection is incidental. Fable argued the frozen
  index is the question and the width result is its answer (teacher = the index you are stuck
  with; selection is the corollary). Owner: "let's test out your way." Title becomes "Reducing
  Query Cost over a Frozen Document Index: What the Index Decides"; RQ1 what a closed-form query
  student recovers from a frozen index and which index properties predict it; RQ2 how much more
  search effort it needs over the index's graph and whether that is predictable. Constella is the
  worked instance.
- **2026-10-01, round 6.** (3) Owner wants the low construction cost and the recurring query-side
  compute saving made explicit, without re-ingestion or downtime; implemented as a measured-cost
  paragraph and a CPU-hours-per-million-queries table in Section 6, no dollar-per-month claim
  because no price or QPS was measured. (4) Hot-swappability is a product capability to showcase in
  the official model card (M22), not a paper finding. (Superseded in part 2026-10-02: per-request choice of query
  path is the paper's answer to re-embedding; see "Settled answers".) (5) Philosophy: include everything that may be
  valuable and meets the criteria; reviewers decide cuts together, unless inclusion harms coherence.
  Material cut from the body stays in appendices rather than being dropped.
- **2026-10-01, round 8, title.** Owner wanted a hook that still reads as a paper and chose
  "The Index Is Fine; the Query Encoder Is the Cost: Query-Side Distillation over Frozen Document
  Vectors". Supersedes "Reducing Query Cost over a Frozen Document Index: What the Index Decides".
- **2026-10-01, round 9, title.** Owner chose "Your Index Is Fine; the Query Encoder Is the Cost:
  Query Distillation over Frozen Indexes" (supersedes the round-8 title). Second person in the title
  is deliberate; the body keeps the third-person contract.
- **2026-10-01, round 10, title.** Owner chose "Your Index Is Fine, Your Query Encoder Is Not: Query
  Distillation over Frozen Indexes" (no semicolon; supersedes round 9).
- **2026-10-02, final round, authority.** Owner ends his part of the session: "Consult with Astra,
  work together to get the best paper based on the goals ... Do the best version possible and do all
  of the gates ... The rule of thumb is that I trust you and Astra to make the best decision." The
  balance he named: interesting, useful, applicable, something people want to share, still a serious
  paper. Astra holds narration authority, Fable technical authority, as equals; the two settle the
  cut list and the final text without a further owner round. Everything is logged, committed, and
  pushed before the session closes; open forks are reported to him at the end.

- **2026-10-02, introduction framing.** Owner approves opening with the different computational roles of query and document encoders and the question of choosing a query encoder independently of the document representation. Do not assume the reader already plans to reduce query cost or migrate an index. Keep migration as a concise practical consequence after the research motivation, rather than the opening premise. Query efficiency remains central; no claim of universal applicability.

- **2026-10-02, clarity and tables.** Owner likes the paper overall so far, finds §4.4/Table C hard to read, and authorizes a paper-wide clarity pass. Tables overall need less text: shorten headers and labels, separate distinct comparisons, and explain definitions in surrounding prose. Preserve results and uncertainty. This is feedback on the reading so far, not whole-paper acceptance.

- **2026-10-02, independent full-paper review.** Owner requests parallel GPT-6-Astra and GPT-6.1-Sol reviews with shared context, each reading the entire paper against intended audience and goals. Explicitly assess whether the paper tells a cohesive story versus unrelated findings joined together, whether each part earns space, omissions and wording. Synthesize findings before changing the manuscript. Reviewer recommendations are not owner approvals.

- **2026-10-02, v20 direction.** Owner explicitly keeps the current title after the full-paper reviews and authorizes implementing their changes. Preserve an interesting, value-dense research paper: foreground surprising and useful findings, strengthen their connections, and trim secondary detail. Do not make accuracy improvements into a bland inventory or repeated defensive qualifications. Title reconsideration is closed unless the owner reopens it.

- **2026-10-02, training-cost visibility.** Owner explicitly requests that both released models' training/construction costs remain visible in the main paper, linked to the possibility of amortizing one-time construction over recurring query-compute savings. Keep this practical motivation in subsequent revisions. Distinguish measured Nano optimization from historical Zero/target-preparation estimates and screening fits; preserve full-build exclusions and retained-relevance requirements. Owner authorizes the balanced framing discussed in this session, not an automatic-payback claim.

### 2026-10-02, review proposals read as benchmark expansion

- Owner judges the v20 review proposals (full-request concurrency benchmark, a three-space trained-table bridge with seeds and matched baselines, Discussion additions on quantization, hybrid, memory, and serving regime) over-engineered: the paper already reads too close to a benchmark reading, and these would make it worse. Answer a reader's question with the paper's argument before adding a measurement. A proposal that adds tables, rows, or caveats must show it strengthens the story; repeats the round-4 oscillation warning.

### 2026-10-02, narration and terminology

Owner approves the following recommendations for subsequent revisions:

- **Preserve the surprising result, simplify the machinery around it.** Let concrete, consequential findings carry the argument; explain their surprise and scope without burying them in procedural detail. Maintain scientific substance and value density.
- Lead findings passages with the finding and why it matters. Introduce methods and settings where they help the reader assess the evidence; retain the detail needed for reproducibility in the appropriate methods or appendix location.
- Use consistent names for measured quantities. Avoid alternating among “space quality,” “index quality,” and “encoder quality” when the intended quantity is the original query path's measured retrieval score.
- Put concise qualifications beside the claims they constrain, and develop their implications in the discussion. Avoid repeatedly restating defensive caveats; retain adverse results and material uncertainty.
- Keep established IR terms such as nDCG, distillation, ridge regression, and HNSW. Define compressed shorthand such as recovery gap, held-out prediction, and loss multiplier at first use, then use it consistently.
- Scrutinize internal experiment labels such as E24, H2d, and NDO-3 in reader-facing prose. Prefer meaningful descriptions where labels provide only project navigation; preserve identifiers in evidence records and where they materially help traceability.
