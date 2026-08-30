codex
# Verdict

Do not launch the five-chain D2 program under the current registration.

D2 is a defensible next capacity hypothesis, but the exclusive re-route is not. The registration contains two decision-breaking loopholes, the §17b relationship is not a causal upper bound, D2 does not subsume additive overlapping n-grams, and the automatic exit would bypass at least two live probes that the repository itself says remain unresolved.

I would keep D2 first in line, but only after a sub-hour closed-form preflight and registration repair. I would not spend the reserved sets yet.

## Findings by severity

- **BLOCKER — verified in repo, algebraically confirmed:** The “mean constituent row” initialization is not a performance floor. The registration calls it one at [m8/registry.json:501](/home/dylan/asymetric-dual-encoders/m8/registry.json:501), but it does not preserve the incumbent query function.

- **BLOCKER — verified in repo:** The coverage-gate escape clause permits “expand the pool and re-measure” at [m8/registry.json:500](/home/dylan/asymetric-dual-encoders/m8/registry.json:500), contradicting the fixed-pool declaration at [m8/registry.json:499](/home/dylan/asymetric-dual-encoders/m8/registry.json:499). If exercised, the old R0 chains cease to be valid controls and 0.00519 is not calibrated for the resulting experiment.

- **BLOCKER — verified in repo:** The precommitted exit is premature. B2 explicitly leaves hard-candidate listwise distillation live, and E14 explicitly says its head result must not close LoRA. The exit nevertheless fires after only D2+B10 at [m8/registry.json:510](/home/dylan/asymetric-dual-encoders/m8/registry.json:510).

- **MAJOR — verified method, causal conclusion inferred:** The §17b slope is neither an expected gain nor an upper bound. It is an uncontrolled between-query association using ordinary OLS standard errors. It identifies correlated headroom, not recoverable headroom.

- **MAJOR — verified in repo, algebraically confirmed:** Non-overlapping D2 tokenization does not supersede additive overlapping n-gram rows. The ledger’s “superseded by D2” statement at [m8/LEDGER.md:1145](/home/dylan/asymetric-dual-encoders/m8/LEDGER.md:1145) collapses two different hypothesis classes.

- **MAJOR — verified in repo, statistical interpretation inferred:** The 0.00519 threshold is not anti-conservative merely because fixed pool text is retokenized. Retokenization is part of the B intervention, not a new pool draw. But the threshold is mismatched to the registered three-seed mean and does not cover tokenizer-selection variance.

- **MAJOR — verified in repo:** Nested selection protects the untouched reserved sets, but it does not make Wikipedia/heldout selection versus CQA evaluation independent. The M8 dev-reuse counter promised by G8 at [m8/LEDGER.md:1173](/home/dylan/asymetric-dual-encoders/m8/LEDGER.md:1173) is also absent from HEAD.

- **MAJOR — verified design, type-I consequence inferred:** Dropping fused is defensible as a power decision, but only if the claim becomes explicitly “dense capacity improved.” It removes evidence about the actual fused system and makes the D2 development criterion less aligned with release.

- **MAJOR — verified in repo:** The M7 `0.000 ± 0.005` conclusion supports retiring the four selected post-gate tweaks, not eleven materially different objectives and data interventions.

- **MAJOR — verified in repo:** NF-CROSSED-FUSED is optional and its consequence depends on whether D2 can “plausibly clear” the resulting bar, an undefined judgment at [m8/registry.json:531](/home/dylan/asymetric-dual-encoders/m8/registry.json:531). That is exploitable.

- **MINOR — verified against the paper:** VDR is a weak negative prior. It is not a clean vocabulary-size ablation of this artifact class, and the authors themselves emphasize the simultaneous BERT→mBERT swap.

## 1. What §17b does—and does not—say

The recorded relationship is real in the narrow descriptive sense:

- Gap slope: +0.04998 per extra subword/word at [results/m8_retention_decomposition.json:178](/home/dylan/asymetric-dual-encoders/results/m8_retention_decomposition.json:178).
- Table slope: −0.01213 at [results/m8_retention_decomposition.json:185](/home/dylan/asymetric-dual-encoders/results/m8_retention_decomposition.json:185).
- Teacher slope: +0.03784 at [results/m8_retention_decomposition.json:192](/home/dylan/asymetric-dual-encoders/results/m8_retention_decomposition.json:192).
- The analysis itself warns that different queries are being compared and rare technical terms can confound the result at [m8src/frag_attrib.py:158](/home/dylan/asymetric-dual-encoders/m8src/frag_attrib.py:158).
- The pooled regression uses ordinary query-level OLS rather than cluster-robust or joint controlled inference at [m8src/retention_decomp.py:127](/home/dylan/asymetric-dual-encoders/m8src/retention_decomp.py:127).

Calling it an “upper bound” is too generous. It is not a bound in either direction:

- The observed slope could be entirely due to query specificity, rarity, domain, length, entity content, or teacher advantage.
- D2 could recover none of it.
- Conversely, a successful phrase feature could improve more than the slope predicts through denoising or conjunction effects.

The strongest negative case is:

1. High fertility tags rare and technical terms.
2. Those terms are particularly informative for retrieval.
3. The contextual teacher benefits from that specificity; the static table does not.
4. A frequency-trained multi-word tokenizer spends most added vocabulary on common words and phrases, not necessarily the rare within-word fragments that generated the association.
5. Non-overlapping segmentation then replaces useful constituent evidence with one polysemous phrase row.
6. Coverage falls as the vocabulary grows.

Under that story, fragmentation is a marker of where contextualization matters, not the cause of the table’s failure.

The strongest positive case is narrower: stable entities, compounds, and conjunctions can have a residual meaning that no additive constituent representation can express. Rows for “New York”, “hot dog”, “machine learning”, or technical compounds can encode that residual directly. This is especially plausible for short entity queries where one phrase dominates the query vector.

### Required fertility reduction

Using the observed gap slope purely as a heuristic:

\[
\Delta f_{+0.005}=\frac{0.005}{0.049976}=0.1001
\]

For the registered 0.00519 bar:

\[
\Delta f_{bar}=\frac{0.00519}{0.049976}=0.1039
\]

The current pooled fertility is about 1.375, so this is roughly a 7.5% reduction. Mechanically, that is reachable at 65K–128K vocabulary size.

But the causal conversion requirement is the harder number:

- If D2 reduces fertility by 0.15, it must convert about 69% of the associated headroom into dense quality.
- At 0.25 reduction, it still needs about 42%.
- If the table’s own observed slope, −0.01213, is the relevant response rather than the gap slope, +0.005 requires a 0.412 fertility reduction.

My magnitude prior is therefore approximately **+0.002 to +0.008**, with +0.005 plausible but nowhere near assured. The raw slope supports running a diagnostic, not five chains.

## 2. The registration problems

### The initialization floor is false

For an incumbent unnormalized query vector

\[
q_{\text{old}} = r + w_a + w_b,
\]

replacing `a b` with a phrase row initialized to the mean gives

\[
q_{\text{new}} = r + \frac{w_a+w_b}{2}.
\]

Those vectors are not generally collinear, so final L2 normalization does not restore equivalence. The phrase has been downweighted relative to all other tokens.

For ordinary sum/mean pooling followed by L2 normalization, initializing the phrase row to the **sum** of its constituent contributions preserves the incumbent representation exactly. For multiplicity-dependent pooling, use the coefficient implied by that pooling rule.

For idioms, random initialization can accidentally be better than the mean, but neither is a floor. The safe design is:

- Baseline-preserving constituent sum.
- Learn a residual on top.
- For rows with insufficient updates, keep the residual at zero.
- Optionally compare teacher-centroid or document-centroid residual initialization for adequately covered rows.

“Hot dog” being non-compositional is a reason to learn a residual, not a reason to destroy the incumbent function before learning begins.

### The coverage gate is gameable

The registered threshold is based on the fraction of unique dev-reachable rows receiving zero updates. That can be manipulated by:

- Changing the vocabulary denominator.
- Creating many one-occurrence rows.
- Expanding the pseudo-query pool after inspecting dev reachability.
- Choosing a tokenizer that avoids generating hard-to-cover rows without actually covering the queries where capacity matters.

The M7 5.71% figure is also overall cold rows, not directly the percentage of unique dev-reachable D2 rows. It does not calibrate a 20% threshold.

A better gate would report:

1. Dev token mass assigned to rows with zero, fewer than 5, and fewer than 20 effective updates.
2. Fraction of dev queries containing at least one such row.
3. Fraction of the measured fragmentation opportunity covered by adequately trained rows.
4. Baseline-preservation loss under sum initialization.
5. A corruption curve showing how retrieval changes as increasingly frequent phrase rows are suppressed.

I would replace the arbitrary 20% threshold with a performance-based one: the baseline-preserving compiled tokenizer must be within roughly 0.001 dense nDCG of R0 before any phrase residual is trained.

Pool expansion must either be forbidden or treated as a separate factorial intervention with freshly paired R0 controls and a measured pool-draw floor.

### Nested selection is not independence

Selecting vocabulary/initialization on Wikipedia+heldout and evaluating on CQA is better than selecting directly on CQA. It does protect the reserved four as long as they remain untouched.

It does not protect the CQA development bar from:

- Shared training data and pseudo-query pool.
- Correlated task response.
- Repeated human reading of the same dev suite.
- Winner’s curse from choosing vocabulary size and initialization on one seed.
- Using dev query text in a coverage gate.

M7 records 322 in-training dev evaluations at [m7/LEDGER.md:641](/home/dylan/asymetric-dual-encoders/m7/LEDGER.md:641). M8 promises a new counter, but the promised `results/m8_dev_reuse_count.json` is absent. That needs fixing before interpreting a 0.005-scale development difference as clean evidence.

## 3. Is 0.00519 the right bar?

Not for the reason proposed.

A fixed collection of pseudo-query texts remains one fixed pool even if the intervention tokenizes it differently. Otherwise every tokenizer experiment would definitionally be pool-varying. The tokenizer belongs to B.

The pool becomes variable only if the “expand the pool” clause is exercised.

The larger statistical problem is that 0.00519 comes from a K=3 range heuristic applied to single-chain variability. The crossed result reports \(\sigma_{\text{chain}}=0.00153\) at [m8/LEDGER.md:2475](/home/dylan/asymetric-dual-encoders/m8/LEDGER.md:2475), while D2 is judged on a three-seed mean at [m8/registry.json:505](/home/dylan/asymetric-dual-encoders/m8/registry.json:505).

Under an idealized independent Gaussian null:

- One fresh arm-control difference has SD  
  \(\sqrt{2}\times0.00153=0.00217\).
- A 0.00519 threshold is about 2.39 SD, one-sided \(p\approx0.0083\).
- A mean of three independent differences has SD about 0.00125.
- The same threshold is about 4.15 SD, one-sided \(p\approx1.7\times10^{-5}\).

So 0.00519 is probably conservative for the registered mean statistic. But this calculation excludes:

- Tokenizer-training randomness.
- Vocabulary winner selection.
- Heteroskedasticity from radically different row coverage.
- Reuse of historical controls.
- Any pool expansion.

The correct remedy is not automatically raising the bar. It is to measure D2-specific null variability or use paired fresh controls for the finalist.

## 4. Dense-only and the lost fused guard

Dropping the “all three fused signs positive” condition is statistically defensible. If the fused effect is truly zero and signs are independent, all three positive occurs with probability \(1/8\); it would veto 87.5% of genuine dense wins whose fused effect is merely too small to resolve.

But this does change the claim.

For a **dense-only** claim, dropping fused causes no type-I inflation: you are still testing the same dense hypothesis.

For the former joint claim—dense improves and the fused system does not regress—the type-I saving from intersection-union cannot be calculated without dense/fused correlation. Illustratively:

- Independent fused signs could multiply the dense false-positive rate by about \(1/8\).
- With nearly perfect dense/fused correlation, they may add almost no protection.
- Intersection-union formally controls the joint claim through its weakest false component; it does not generally produce a simple product of marginal \(p\)-values.

The bigger objection is decision alignment. If fusion damps dense changes by about 10×, a +0.005 dense improvement predicts only about +0.0005 fused. That may prove the tokenizer mechanism without making M8 shippable.

My replacement would be:

- Dense remains the powered D2 gate.
- NF-CROSSED-FUSED becomes mandatory before declaring success.
- Fused gets a predeclared non-inferiority guard or posterior/headroom analysis—not “positive in all three seeds.”
- A dense-only win must be labeled a mechanism success, not a release success.

The optional wording and undefined “plausibly clear” escape in NF-CROSSED-FUSED should be removed.

## 5. The exit and the deferral

The exit is disciplined in form but violates the standing directive in substance.

The directive requires best-component arithmetic, diagnosis of every failure, a literature sweep, algebraic checking, and an explicit reopening condition before declaring a bar unreachable at [CLAUDE.md:76](/home/dylan/asymetric-dual-encoders/CLAUDE.md:76). It also says structural premises remain revisitable at [CLAUDE.md:138](/home/dylan/asymetric-dual-encoders/CLAUDE.md:138).

Two live contradictions matter:

- B2’s uniform KL is degenerate, but teacher-top-200 entropy is 0.777 nats at [results/m8_b2_entropy.json:33](/home/dylan/asymetric-dual-encoders/results/m8_b2_entropy.json:33). The result explicitly names R-LIST as the consequence at [results/m8_b2_entropy.json:3](/home/dylan/asymetric-dual-encoders/results/m8_b2_entropy.json:3). Hard-candidate listwise distillation therefore remains mechanically live.
- The ledger says the E14 MLP null is “WEAK evidence” about LoRA and must never close E14 at [m8/LEDGER.md:1337](/home/dylan/asymetric-dual-encoders/m8/LEDGER.md:1337).

A precommit cannot perform a future milestone audit in advance. After D2+B10 miss, the directive still has to be rerun.

The `0.000 ± 0.005` transfer is also overextended. Its source is four selected M7 post-gate tweaks and their matched-control behavior at [m7/LEDGER.md:626](/home/dylan/asymetric-dual-encoders/m7/LEDGER.md:626). It is not evidence that listwise hard-candidate KL, document-centroid targets, fused-aware training, LoRA, and D-FINEWEB all share the same null distribution.

The cheapest false-economy deferral is **B8**: it is a closed-form document-centroid target at [m8/registry.json:214](/home/dylan/asymetric-dual-encoders/m8/registry.json:214), with the archived review estimating roughly 15 minutes plus scoring. The most important diagnostic follow-up is **R-LIST**, because B2 directly triggered it.

I would not run all eleven before D2. I would require B8 and R-LIST before the automatic exit can fire.

Not spending the reserved access is ultimately the correct default. A “publishable miss” does not by itself justify consuming a clean panel; those sets retain option value for a structurally revised M8/M9. But refusing access after only D2+B10 would be premature abandonment.

## 6. Is the algebra complete?

Only under a narrower class than the prose suggests.

For

\[
q(x)=\operatorname{normalize}\left(\sum_j \phi_j(x)w_j\right)
\]

with fixed single-vector documents and dot-product scoring, fixed type-level linear transforms and scalar weights are absorbable. That part is sound.

But capacity can also enter through:

- A richer feature map \(\phi(x)\): overlapping phrases, character n-grams, boundaries, positions, ordered pairs, conditional conjunctions.
- Query-dependent nonlinear aggregation rather than fixed additive pooling.
- Multiple document vectors or a different scoring operator.
- Changes to the document encoder.
- A contextual online query encoder.
- Query-adaptive dense/sparse fusion.

Thus the statement at [CLAUDE.md:90](/home/dylan/asymetric-dual-encoders/CLAUDE.md:90) is complete only after freezing feature activation, aggregation, document representation, and scoring. Multiplicity-dependent pooling is one nonlinear direction, not the only one.

Most importantly, D2 non-overlap is not a superset of additive n-grams:

- D2 chooses one segmentation and removes constituent activations.
- Additive n-grams retain the incumbent unigrams and can activate several overlapping phrases.
- An additive phrase row with zero residual exactly recovers R0; D2 does not automatically do so.
- Character n-grams can cover rare and unseen technical strings that a frequency tokenizer may never allocate a full token to.

That is why “no auto-revival” for n-grams is unjustified.

## Three capacity directions worth taking seriously

### 1. Overlapping residual word/character n-gram rows

Mechanism: retain the complete R0 query and add learned residual rows for overlapping phrases, affixes, and character n-grams. This directly tests conjunction and rare-term capacity without forcing a lossy segmentation.

This is the same useful principle behind [fastText’s character n-gram representations](https://aclanthology.org/Q17-1010/) and [Charagram](https://aclanthology.org/D16-1157/): rare or unseen strings inherit subword evidence while retaining compositional structure.

- Rough magnitude: +0.003 to +0.015; the lower half is more credible.
- Cost: 16K–64K extra int8 rows, roughly 16–64 MB at 1024 dimensions, plus extra lookups but still one ANN query.
- Kill fast: frozen-R0 residual ridge on existing teacher targets, cross-fitted, comparing D2 non-overlap versus overlapping phrase and character features at identical row budgets.

This is the direction I would test before declaring D2 the only remaining tokenizer lever.

### 2. Co-adapt the document tower with LoRA/last-block tuning

Mechanism: reshape document vectors toward the additive query manifold. A post-hoc linear head failed, but an input-dependent change inside the document encoder is not algebraically equivalent to that head. The repo itself says E14-HEAD does not close LoRA.

Jointly trained dual encoders such as [DPR](https://aclanthology.org/2020.emnlp-main.550/) demonstrate the basic value of adapting both sides, though your query side would remain constrained.

- Rough magnitude: +0.005 to +0.025, bounded roughly by the existing table-to-dense gap.
- Cost: adapter training plus full document re-encoding and index replacement—the expensive part.
- Kill fast: train only the last block/LoRA on a 50K–100K-document subset and test exact retrieval on a self-contained corpus. Require both improved alignment loss and heldout retrieval before authorizing the full corpus encode.

This is the highest-upside structural reopening, not a current M8 convenience probe.

### 3. Small-k document facets

Mechanism: store 2–4 vectors per document and score with max-over-facets. One document vector cannot simultaneously represent all subjects, entities, and answer-bearing passages in a long document; a lookup query may be adequate while the document bottleneck remains.

This is supported by [Multi-View Document Representation](https://aclanthology.org/2022.acl-long.414/) and, at the larger extreme, [ColBERTv2](https://aclanthology.org/2022.naacl-main.272/). ColBERTv2 is outside your current scope, but it demonstrates what multiple document-side vectors and late interaction can recover.

- Rough magnitude: +0.005 to +0.030 at k=2–4; highly corpus-dependent.
- Cost: 2–4× dense index entries, deduplication, and more ANN work. Query computation stays at lookup-table level.
- Kill fast: build exact k-facet representations on one 100K-document dev corpus from cached document token embeddings. If exact scoring cannot gain about 0.01 before ANN approximation, stop.

The owner-level fourth option is to reopen “no transformer at query time.” Existing compact-query work reports that a two-layer query encoder can retain much of a full model’s retrieval quality: see [Query Encoder Distillation](https://aclanthology.org/2023.sustainlp-1.23/) and [Efficient Online Query Encoding](https://aclanthology.org/2024.findings-naacl.4/). This sacrifices the defining zero-compute artifact, so it is better treated as the M9 alternative than smuggled into M8.

## The cheap D2 preflight I would run

One deterministic 64K tokenizer is enough initially.

1. **Opportunity check:** Retokenize the existing train/dev queries and measure actual reduction in the original §17b fragmentation opportunity, especially among the high-gap queries. If the OOD reduction is below 0.104, even the optimistic association cannot clear 0.00519.

2. **Compile check:** Initialize new phrase rows to the constituent **sum**, freeze everything, and score the existing R0 table. This should reproduce R0 closely. Score mean initialization as a negative control.

3. **Closed-form residual check:** Keep all incumbent rows fixed and solve only new phrase/character residual rows against the existing teacher-query targets—or B8 document-centroid targets—using block-CG.

4. **Cross-fit:** Learn residuals without the scored dataset’s queries, then evaluate on the existing development groups.

5. **Directly compare hypothesis classes:** At equal row budget, score:
   - D2 non-overlapping segmentation.
   - Additive overlapping word n-grams.
   - Additive character n-grams.
   - D2 plus zero-residual cold-row fallback.

6. **Proceed criterion:** Authorize full chains only if the closed-form result has clear positive sign, plausible headroom above 0.00519, acceptable coverage by occurrence mass, and no material fused degradation.

Given B7’s 17-second solve and the archived evaluation timings, this should cost well under an hour. It is more predictive of D2’s sign than fertility reduction alone because it directly asks whether new rows contain useful residual capacity.

## What I would do

1. Repair D2: sum/residual initialization; fixed pool; deterministic selection; coverage by occurrence mass; mandatory fused-floor consequence.
2. Run the closed-form D2-versus-overlapping-n-gram preflight.
3. Run full three-seed D2 only if that preflight clears.
4. If D2 misses, run B10, B8, and the already-triggered hard-candidate R-LIST.
5. Conduct an explicit owner review of LoRA and small-k documents.
6. Preserve the reserved sets unless one of those routes produces a credible release candidate.

So: the intuition that tokenizer capacity is now the best cheap lever is reasonable. The claim that this particular D2 registration is the sole remaining route—and that D2+B10 are enough to terminate M8—is not.

No files were modified.
tokens used
285,204
# Verdict

Do not launch the five-chain D2 program under the current registration.

D2 is a defensible next capacity hypothesis, but the exclusive re-route is not. The registration contains two decision-breaking loopholes, the §17b relationship is not a causal upper bound, D2 does not subsume additive overlapping n-grams, and the automatic exit would bypass at least two live probes that the repository itself says remain unresolved.

I would keep D2 first in line, but only after a sub-hour closed-form preflight and registration repair. I would not spend the reserved sets yet.

## Findings by severity

- **BLOCKER — verified in repo, algebraically confirmed:** The “mean constituent row” initialization is not a performance floor. The registration calls it one at [m8/registry.json:501](/home/dylan/asymetric-dual-encoders/m8/registry.json:501), but it does not preserve the incumbent query function.

- **BLOCKER — verified in repo:** The coverage-gate escape clause permits “expand the pool and re-measure” at [m8/registry.json:500](/home/dylan/asymetric-dual-encoders/m8/registry.json:500), contradicting the fixed-pool declaration at [m8/registry.json:499](/home/dylan/asymetric-dual-encoders/m8/registry.json:499). If exercised, the old R0 chains cease to be valid controls and 0.00519 is not calibrated for the resulting experiment.

- **BLOCKER — verified in repo:** The precommitted exit is premature. B2 explicitly leaves hard-candidate listwise distillation live, and E14 explicitly says its head result must not close LoRA. The exit nevertheless fires after only D2+B10 at [m8/registry.json:510](/home/dylan/asymetric-dual-encoders/m8/registry.json:510).

- **MAJOR — verified method, causal conclusion inferred:** The §17b slope is neither an expected gain nor an upper bound. It is an uncontrolled between-query association using ordinary OLS standard errors. It identifies correlated headroom, not recoverable headroom.

- **MAJOR — verified in repo, algebraically confirmed:** Non-overlapping D2 tokenization does not supersede additive overlapping n-gram rows. The ledger’s “superseded by D2” statement at [m8/LEDGER.md:1145](/home/dylan/asymetric-dual-encoders/m8/LEDGER.md:1145) collapses two different hypothesis classes.

- **MAJOR — verified in repo, statistical interpretation inferred:** The 0.00519 threshold is not anti-conservative merely because fixed pool text is retokenized. Retokenization is part of the B intervention, not a new pool draw. But the threshold is mismatched to the registered three-seed mean and does not cover tokenizer-selection variance.

- **MAJOR — verified in repo:** Nested selection protects the untouched reserved sets, but it does not make Wikipedia/heldout selection versus CQA evaluation independent. The M8 dev-reuse counter promised by G8 at [m8/LEDGER.md:1173](/home/dylan/asymetric-dual-encoders/m8/LEDGER.md:1173) is also absent from HEAD.

- **MAJOR — verified design, type-I consequence inferred:** Dropping fused is defensible as a power decision, but only if the claim becomes explicitly “dense capacity improved.” It removes evidence about the actual fused system and makes the D2 development criterion less aligned with release.

- **MAJOR — verified in repo:** The M7 `0.000 ± 0.005` conclusion supports retiring the four selected post-gate tweaks, not eleven materially different objectives and data interventions.

- **MAJOR — verified in repo:** NF-CROSSED-FUSED is optional and its consequence depends on whether D2 can “plausibly clear” the resulting bar, an undefined judgment at [m8/registry.json:531](/home/dylan/asymetric-dual-encoders/m8/registry.json:531). That is exploitable.

- **MINOR — verified against the paper:** VDR is a weak negative prior. It is not a clean vocabulary-size ablation of this artifact class, and the authors themselves emphasize the simultaneous BERT→mBERT swap.

## 1. What §17b does—and does not—say

The recorded relationship is real in the narrow descriptive sense:

- Gap slope: +0.04998 per extra subword/word at [results/m8_retention_decomposition.json:178](/home/dylan/asymetric-dual-encoders/results/m8_retention_decomposition.json:178).
- Table slope: −0.01213 at [results/m8_retention_decomposition.json:185](/home/dylan/asymetric-dual-encoders/results/m8_retention_decomposition.json:185).
- Teacher slope: +0.03784 at [results/m8_retention_decomposition.json:192](/home/dylan/asymetric-dual-encoders/results/m8_retention_decomposition.json:192).
- The analysis itself warns that different queries are being compared and rare technical terms can confound the result at [m8src/frag_attrib.py:158](/home/dylan/asymetric-dual-encoders/m8src/frag_attrib.py:158).
- The pooled regression uses ordinary query-level OLS rather than cluster-robust or joint controlled inference at [m8src/retention_decomp.py:127](/home/dylan/asymetric-dual-encoders/m8src/retention_decomp.py:127).

Calling it an “upper bound” is too generous. It is not a bound in either direction:

- The observed slope could be entirely due to query specificity, rarity, domain, length, entity content, or teacher advantage.
- D2 could recover none of it.
- Conversely, a successful phrase feature could improve more than the slope predicts through denoising or conjunction effects.

The strongest negative case is:

1. High fertility tags rare and technical terms.
2. Those terms are particularly informative for retrieval.
3. The contextual teacher benefits from that specificity; the static table does not.
4. A frequency-trained multi-word tokenizer spends most added vocabulary on common words and phrases, not necessarily the rare within-word fragments that generated the association.
5. Non-overlapping segmentation then replaces useful constituent evidence with one polysemous phrase row.
6. Coverage falls as the vocabulary grows.

Under that story, fragmentation is a marker of where contextualization matters, not the cause of the table’s failure.

The strongest positive case is narrower: stable entities, compounds, and conjunctions can have a residual meaning that no additive constituent representation can express. Rows for “New York”, “hot dog”, “machine learning”, or technical compounds can encode that residual directly. This is especially plausible for short entity queries where one phrase dominates the query vector.

### Required fertility reduction

Using the observed gap slope purely as a heuristic:

\[
\Delta f_{+0.005}=\frac{0.005}{0.049976}=0.1001
\]

For the registered 0.00519 bar:

\[
\Delta f_{bar}=\frac{0.00519}{0.049976}=0.1039
\]

The current pooled fertility is about 1.375, so this is roughly a 7.5% reduction. Mechanically, that is reachable at 65K–128K vocabulary size.

But the causal conversion requirement is the harder number:

- If D2 reduces fertility by 0.15, it must convert about 69% of the associated headroom into dense quality.
- At 0.25 reduction, it still needs about 42%.
- If the table’s own observed slope, −0.01213, is the relevant response rather than the gap slope, +0.005 requires a 0.412 fertility reduction.

My magnitude prior is therefore approximately **+0.002 to +0.008**, with +0.005 plausible but nowhere near assured. The raw slope supports running a diagnostic, not five chains.

