1. **Register diagnosis.**

The owner is right about the register. The opening introduces a product family before establishing an unresolved research problem. “Two findings follow from the cheapness of the students” supplies narrative organization, not hypotheses. “Finding A: teacher quality does not rank student quality” announces a conclusion before defining its scope.

“Screen candidates with the student recipe you intend to use” and “What a search engineer can take from this” make the ending instructional. “The savings also move” is a magazine transition. Plain language is not the problem; advice substitutes for inference.

Methods are distributed across setup, results, and appendices. Section 4 packs endpoints, censoring, effects, and explanations into one paragraph. However, the brief overstates the absence of effect sizes: the draft already reports nDCG differences and recovery deficits. It needs an organized statistical argument, not merely more tables.

The paper mostly answers “What happened in these screens?” It should ask which properties predict student quality and search difficulty, then distinguish answered questions from unresolved ones.

2. **Scientific weight.**

**Teacher selection:** This is stronger than an anecdote: a consistent protocol, consequential reversals, and two representations. But 26 related checkpoints are not 26 independent replications. Both recipes use ridge fitting; the head has backbone affinity. The +0.09 correlation interval permits a substantial positive association.

The exciting result could be **conditional predictability concealed by pooled rankings**. Within-width head correlations are positive; width, family, and teacher quality covary. Model absolute student nDCG using teacher quality, width, recipe, and family, with dataset effects and grouped validation. Retention alone introduces a denominator confound. Report selection regret and predictive improvement, not just correlations. Existing evidence cannot establish a width mechanism or transfer to trained Nano.

**Search cost:** The recurring recovery deficit is credible measurement. Its scientific opportunity is predicting difficulty before constructing the full graph, using sampled document/query geometry, then testing across families and workloads. The current top-1 cosine and margin fail that test.

The disagreement between neighbor recovery and relevance is itself important. TREC-COVID’s primary relevance multiplier is mixed despite universal recovery deficits. A useful paper would explain when missed neighbors matter to relevance. It cannot currently claim a general fourfold relevance-preserving search requirement or latency penalty.

3. **Decisions to reopen.**

Keep construction first provisionally, but promote width from a qualifier to a central competing explanation. Expand the geometry null into an explicit failed predictive hypothesis. Keep the family as setup; move its per-dataset figure to the appendix.

Keep fusion and routing demoted. LEARNINGS contains interesting negatives—fragmentation interventions, failed blends, fusion-depth reversals—but adding them would broaden the report without resolving either central question.

Reopen native query precision only as a cost-qualified intervention with a falsifiable interaction hypothesis. The owner is right that another setting sweep adds little; wrong if “a knob” is taken to exclude a controlled remedy experiment. It would address binary scoring, however, not explain the uncompressed HNSW penalty.

A narrower teacher-selection paper, with conditional analysis and prospective validation, would be stronger than the present two-finding paper. Keep two findings only if search develops comparable explanatory or predictive depth.

4. **Proposed shape.**

For a two-question version, target **4,000 main-text words**, plus a 180-word abstract:

| Section | Words | Argument-bearing material |
|---|---:|---|
| Problem and research questions | 400 | Teacher selection before indexing; query substitution afterward |
| Related work | 350 | Distillation limits, frozen-index alignment, query-distribution effects |
| Method and experimental design | 850 | Two recipes, sampling units, endpoints, exposure, censoring |
| Teacher-selection results | 750 | Hypotheses first; width-stratified plot, adjusted estimates, selection-regret table |
| Search results | 750 | Hypotheses first; recovery and relevance curves, multipliers with censoring |
| Analysis | 550 | Conditional associations, failed geometry predictors, prospective predictions |
| Limitations and conclusion | 350 | Population scope, dependence, unresolved mechanisms |

One compact setup table suffices. Put uncertainty beside effects. Preserve registered versus exploratory status; newly written hypotheses must not retroactively become preregistration.

5. **Three experiments or analyses, ranked.**

**First: conditional teacher-selection analysis.**  
**Claim:** recipe-specific development scores add predictive value beyond teacher quality, width, and family. Fit a small regularized model to absolute student scores; compare nested predictors using leave-family-out validation and dataset sensitivity.  
**Required outcome:** lower prediction error and selection regret, with uncertainty; stable conditional estimates.  
**Plausible negative:** width/backbone affinity explains most apparent reversal, or estimates remain unidentified. That changes the headline.  
**Cost:** existing data; roughly 1–2 analyst days, negligible rental.

**Second: prospective teacher-selection test.**  
**Claim:** the frozen selection rule generalizes to previously unscored checkpoints. Pre-specify a modest roster spanning widths and less-represented families, both recipes, baseline rules, and regret endpoints before scoring the six public sets.  
**Required outcome:** development screening consistently reduces regret against teacher-only and width-aware rules.  
**Plausible negative:** no improvement outside the existing roster.  
**Cost:** approximately $40–70 with bounded corpus encoding; no full-model training.

**Third: pre-index ANN difficulty prediction.**  
**Claim:** sampled query/document properties predict recovery effort beyond width and workload. Test a small declared feature set—local density, neighborhood concentration, query/document distribution mismatch—across the 25 spaces, holding out families and workloads. Treat unreachable thresholds as censored.  
**Required outcome:** calibrated predictions beat a constant multiplier; relevance effects remain separately reported.  
**Plausible negative:** prediction fails across workloads, leaving a bounded empirical regularity.  
**Cost:** approximately $0–60 using caches; several analysis days. Existing-data validation remains exploratory.

6. **Sentences I would not sign.**

“Over 15 BEIR datasets they retain 100%, 90.5%, and 81.4%...” is mathematically valid but scientifically empty for Stella. Replace it with: **“Nano and Zero retain 90.5% and 81.4% of Stella’s exact-search macro nDCG@10.”**

Also revise:

- “A frozen head is a floor for a trained student”: no guaranteed retrieval ordering exists.
- “Matching the teacher takes a median four times…”: specify recovery, workloads, and reached spaces.
- “Expect a cheap student to need a larger search budget…”: exceeds the demonstrated relevance scope.
- “Teacher quality does not rank student quality”: qualify the pooled association and tested recipes.

No files changed.

Codex session ID: 01a0f7ab-b46a-78c0-88a9-2c549ae010d3
Resume in Codex: codex resume 01a0f7ab-b46a-78c0-88a9-2c549ae010d3
