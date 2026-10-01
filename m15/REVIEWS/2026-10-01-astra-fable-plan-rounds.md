# Astra and Fable plan rounds (2026-10-01)

Three rounds between gpt-6-astra (narration lead) and Claude Fable (technical lead), run through codex-companion after the two owner-directed reviews. Fable's prompts are summarized; Astra's replies are verbatim. Outcome: `REVISION_PLAN_V16.md`.

## Round 1

Fable asked for an outline under the owner's new direction ("interesting, coherent, applicable; make people want to use the models or replicate the family"), proposed a two-part shape (using a family over an existing index; building a family for your own index), and listed eight positions: keep the teacher screen; replace the 62x-to-2.2x ratio with the effort multiplier plus receipt geometry; add a per-dataset retention figure; drop Nano-versus-bge-small from the main text but keep one re-indexing cost sentence; cut section 6 and shrink section 4; move the specialization vision to the introduction; consolidate hedging; accept the reproducibility finding as a pre-circulation task.

Astra, verbatim:

> Fable, I accept the two-part structure under the owner's new direction. I withdraw the companion-paper recommendation, provided construction remains subordinate to the shared-index proposition.
>
> **Working title:** *Constella: One Document Index, Several Query Budgets*
>
> **Thesis:** Constella demonstrates interchangeable query budgets over one document index; realizing their value requires understanding where cheap queries preserve relevance, how they change search effort, and which construction measurements predict useful retrieval.
>
> Proposed outline, approximately **2,950 words** excluding appendices: Abstract 150; 1. The index stays; query computation changes 400; 2. A family of query budgets (Part A) 500; 3. Cheap encoding changes the search problem 650; 4. What sharing the index makes possible 250; 5. Building a family (Part B) 650; 6. Limits and implications 250; Artifact statement 100.
>
> The evaluation appendix retains every registered clean-four/all-six contrast, reserved-four table, and bge-small/LEAF/BM25 reference. Add complete dataset scores and CQADupStack aggregation details. Separate protocol, registered outcomes, construction details, and search controls.
>
> **Cuts:** ratio headline, hypothetical threshold example, standalone serving section, main-text fusion-depth table, repeated caveats, bge-small advocacy, and unsupported economical-specialization framing.
>
> **Adds:** retention figure, explicit ANN effort comparison, exploratory geometry paragraph, concise index-cost motivation, and a clearer bridge from using this family to constructing another.
>
> My positions on your eight points: 1. Agree, with a narrative boundary: distinguish an existing index, which fixes the teacher, from choosing a teacher before creating an index; the screen predicts its ridge recipe, not Zero/Nano success. 2. Agree on replacement; disagree with "mechanism" as established: lower top-neighbor cosines and flatter margins are consistent with harder graph search, not causal identification; label exploratory; differing oversampling prevents interpreting `ef` alone as total work; "encoder-independent" should become "similar at fixed settings in these measurements." 3. Agree on the figure; disagree on the proposed domain rule: Quora, ArguAna, and climate-fever do not establish an entity/duplicate-question category; show named datasets and observed variation with exposure visible. 4. Agree: one hardware-qualified corpus-encoding sentence; identify whether 40 hours is measured or extrapolated. 5. Agree. 6. Agree. 7. Agree in principle: one caveat sentence is a target, not a quota. 8. Agree conditionally: if preservation is unresolved, the artifact statement must disclose missing inputs.

## Round 2

Fable accepted the outline and refined: effort multiplier presented on the uncompressed collection where ef is the only knob (512/256/128; 3.58/2.18/1.53 ms); no category label on retention, one exploratory sentence on the word-order correlate; 39.9 h is measured; section 2 defines only interpretation essentials, construction told once in section 5.

Astra, verbatim:

> We have an agreed plan. Point 2, confirm: lead with the uncompressed comparison, then the shared-binary case with both knobs explicit; geometry remains exploratory evidence consistent with the search penalty. Point 3, confirm with wording amendment: say the Stella-minus-Zero gap "correlates with measured word-order sensitivity," label Spearman 0.52 exploratory and identify the unit of analysis, "the only correlate measured in this analysis." Point 4, confirm: a measured 39.9 hours on one RTX 3080; not a universal re-indexing estimate. Section 2 addition, confirm; exposure annotations remain beside the retention figure. Remaining narration request: make the transition into Part B explicit: the released family demonstrates the capability; constructing another family raises a separate question about which measurements predict useful students.

## Round 3

Owner feedback relayed: thin on whitepaper-grade findings; wants findings that disagree with existing literature; native-precision experiment rejected, domain-fit deferred, experiment A approved, experiment B conditional on Astra's agreement; cost efficiency and no re-embedding to be more central; log every decision. Fable's caution: QED, EmbedDistill, LEAF, and pyNIFE also freeze the document tower; what is ours is closed-form fitting in minutes, which made a 26-teacher screen affordable.

Astra, verbatim:

> **1. B, agree, with a tighter analysis contract.** It addresses a research question: is additional ANN effort a recurring consequence of table-based query substitution, or a property of Stella? Two workloads and repeated graph builds materially strengthen the present evidence. Its value is the distribution of outcomes, including absent or reversed penalties, not merely finding another positive example.
>
> Explanation for the owner: "Keeping document vectors unchanged does not guarantee that a cheaper query encoder can search them equally efficiently. This experiment tests across 26 embedding spaces whether the encoding savings systematically require more index exploration, and whether measurable query geometry helps explain when that happens."
>
> Before observing results, fix the teacher reference `ef`, define loss relative to each encoder's own exact baseline, and distinguish neighbor recovery from relevance loss. Report targets unreachable within the sweep as censored, rather than assigning 512. Treat graph builds as repetitions, not independent teacher samples; related checkpoints also limit independence. Geometry correlations remain exploratory, not causal explanations.
>
> **2. Cost framing, agree, amended.** Use: "Inexpensive student probes make teacher selection empirically testable; the selected query representation must then be evaluated against both exact retrieval and approximate-search cost." This makes cost the enabling method, not a novelty claim. Closed-form fitting without gradients and frozen document towers need explicit attribution; the contribution is what the affordable comparison reveals. Keep fitting time separate from teacher-target generation, document encoding, and index construction. Nano's 57-hour optimization component should not establish that the entire investigation was cheap.
>
> I disagree with deciding in advance that A contradicts LEAF, QED, or EmbedDistill. Their teacher choices are not necessarily a claim that teacher quality predicts student quality. A can challenge that selection heuristic; a contradiction requires an explicit prior claim and comparable conditions. Two frozen-backbone/head recipes also do not justify a universal statement about distilled encoders.
>
> **3. Outline and title.** The owner's changed objective warrants moving construction forward: opening on teacher selection and the cost of testing it; compact family/setup section; A: existing table screen plus cross-recipe ranking transfer, surrogate failures, and complete cost boundaries; B: consequences for ANN effort across teacher spaces; brief family illustration, fusion/routing limits, consolidated limitations, artifacts. Precision becomes supporting evidence, without a promised remedy. Domain fitting stays future work. Optional MiniLM belongs only if it adds a meaningfully different test.
>
> Working title: **"Constella: Teacher Selection and Search Cost in Query-Side Distillation."**
>
> The narrative should remain outcome-contingent: A may support transferable rankings or recipe-specific screening; B may establish recurring penalties or explain their limits.

Fable accepted all three answers. Disagreement recorded and resolved: Fable had proposed "contradicts LEAF/QED/EmbedDistill"; Astra's wording "challenges the common selection heuristic" is adopted.
