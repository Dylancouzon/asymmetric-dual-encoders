I accept the two-finding spine, but Fable’s wording overstates the width explanation and E20’s relevance-quality generality.

1. **E19: accept the selection lesson; narrow the claim.**

   > “Across 26 checkpoints, teacher retrieval quality was a poor standalone guide to student quality under either tested recipe. Teacher rankings transferred only moderately between recipes, while each recipe’s own dev screen tracked its evaluation ranking. Screen candidates using the intended student recipe.”

   “Screen the student you will ship” exceeds this experiment: neither probe validates selection for trained Zero or Nano. “Does not predict” also sounds stronger than +0.09 with an interval extending to +0.51. Lead with the actionable comparison: the registered head screen chose a teacher yielding **0.325 versus 0.223** from the strongest registered teacher. That makes the consequence concrete.

2. **E20: accept recurring recovery difficulty, not a universal relevance-loss penalty.**

   > “Under the tested uncompressed HNSW configuration, table queries recovered fewer of their own exact neighbors at ef=64 in all 25 included spaces on FiQA and TREC-COVID. Matching teacher recovery required a median 4× ef. The recovery gap was smaller on SCIDOCS. Matching relative nDCG loss gave a consistent penalty on FiQA but mixed results elsewhere.”

   Specify each path’s **own** exact baseline: these are different neighbor targets and different relevance ceilings. The head’s 24/25 and 23/25 recovery deficits provide secondary cross-recipe support. Corpus size is a possible explanation, not an identified cause; three datasets confound size with workload. Nor is 4× ef a 4× latency claim.

3. **Reporting both metrics is honest—and required by the protocol.** Recovery was pre-specified as secondary; the 50-query concern preceded scoring. Exact framing:

   > “We report the pre-specified relative-nDCG-loss multiplier first, alongside the pre-specified secondary exact-neighbor-recovery multiplier. On TREC-COVID these measures diverge: recovery is consistently worse, whereas relevance-loss multipliers are mixed. Its 50 queries and near-zero teacher loss make the latter threshold sensitive; this is a plausible explanation, not evidence that the relevance result should be discarded.”

   Keep censoring visible: FiQA’s loss median of 4× describes **21 reached spaces**, with four space-level censored results. A censored space means at least one build failed by ef=512, not necessarily that its build-averaged multiplier exceeds eight. Document all amendments, the tie tolerance, and TAS-B’s actual exclusion trigger: **SCIDOCS head parity 0.994**, not simply FiQA ties.

4. **Geometry belongs briefly in the body**, with details in limits/appendix:

   > “The two measured geometry summaries showed no consistent association with the recovery penalty across workloads.”

   Avoid “do not predict”: these exploratory correlations neither establish a null nor test predictive performance. TREC-COVID’s −0.34/−0.37 estimates particularly resist a categorical dismissal.

   **Width is a qualifier on Finding A**, not a third finding. Report the negative retention associations and positive within-width head correlations together. Reject “the student pays for the wider space” as causal wording: width covaries with family and teacher quality, and retention itself divides by teacher quality. The small bands support a hypothesis, not a mechanism.

5. **Keep the title and outline.** Revise the abstract to emphasize **recipe-specific teacher screening** and **recurring ANN recovery penalties with workload-dependent relevance effects**. Put width beside Finding A and metric divergence inside Finding B; neither belongs hidden in limits. These qualifications sharpen the paper’s contribution rather than weaken it.

Codex session ID: 01a0f651-b50c-72f0-9249-b1de47148e7d
Resume in Codex: codex resume 01a0f651-b50c-72f0-9249-b1de47148e7d
