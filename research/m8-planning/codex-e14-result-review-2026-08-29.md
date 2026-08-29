# Codex adversarial review — E14-HEAD RESULT and interpretation, 2026-08-29

gpt-5.6-sol, read-only, effort high. Two BLOCKERs, both over-claims in a write-up already
committed as permanent; both withdrawn with the corrected wording adopted verbatim.
Disposition in `m8/RESULTS.md` and `m8/STATUS.md`.

## BLOCKER

1. Claim 4 over-identifies the difference of differences.

The saved result establishes:

> Applying the trained head hurts bag-query nDCG less than it hurts frozen-teacher-query nDCG on the two CQADupStack components.

It does not establish that documents became more bag-reachable. Both absolute bag gains are negative. Calling `+0.009` a “re-shaping benefit” converts relative preservation into an absolute benefit.

The statistic is descriptively consistent:

- LIN: +0.0069, +0.0130, +0.0074
- MLP: +0.0085, +0.0088, +0.0051
- All twelve treatment × seed × component values are positive.

But with three seeds, the smallest exact one-sided sign-test p-value per treatment is 0.125. A normal-theory seed t-interval happens to exclude zero—LIN `[+0.0007,+0.0176]`, MLP `[+0.0024,+0.0126]`—but n=3 makes that assumption-driven, and the registered control was explicitly descriptive ([registry.json](/home/dylan/asymetric-dual-encoders/m8/registry.json:426)).

More importantly, the four cells are not independent. Their covariance should be exploited by a paired per-query difference of differences. The scorer currently discards those data by immediately reducing each cell to a component mean ([e14_score.py](/home/dylan/asymetric-dual-encoders/m8src/e14_score.py:155)).

Replace [RESULTS.md lines 163–168](/home/dylan/asymetric-dual-encoders/m8/RESULTS.md:163):

> Across all seeds and both CQADupStack components, applying the trained head reduced bag-query nDCG less than frozen-teacher-query nDCG (LIN +0.0091; MLP +0.0075 difference of differences). This is descriptive evidence consistent with relative alignment toward the co-trained bag representation. It does not show an absolute bag benefit, identify bag reachability, or demonstrate information destruction.

“Destroying information” should be deleted. A poorer match to the original teacher query geometry is not evidence of information loss; the linear residual map may remain full-rank.

What would make it credible: persist all four per-query score vectors, prespecify the component-equal paired DiD, use a query-within-component and seed-aware bootstrap/randomization analysis, and add more training seeds. Existing heads and tables can be rescored; no retraining is needed for the per-query part.

2. Claim 3 improperly overrides LIN’s registered gate.

Reporting the registered label and then presenting contrary post-hoc evidence is honest. Saying the evidence “disqualifies” the label is not.

MLP is a different architecture, and its adequacy classification came from separate seed-3, holdout-reduced arms—not the reported MLP arms. MLP harming more does not establish that a longer-trained LIN could not recover. It only weakens a generic story that every head is failing solely because 2,500 steps are insufficient.

The record itself correctly admits that the adequacy comparison is one unreplicated arm per budget and that 0.350 versus 0.220 is unresolved ([RESULTS.md](/home/dylan/asymetric-dual-encoders/m8/RESULTS.md:116)); the subsequent declaration that it disqualifies the gate contradicts that disclosure ([RESULTS.md](/home/dylan/asymetric-dual-encoders/m8/RESULTS.md:146)).

Nor did the endpoint simply “move steadily away.” The in-training mean-pool OOD contrasts versus R0N were:

- LIN: −0.0210 at 500, −0.0268 at 1,500, −0.0264 at 2,500.
- MLP: −0.0309, −0.0359, −0.0309.

That is early persistent harm, not a monotone divergence. The holdout and final endpoint also differ in seed, pool, schedule, precision and purpose.

Correct statement:

> LIN is a strong negative result for the registered 2,500-step configuration, but remains OPTIMIZATION-INADEQUATE for the method-level question. MLP met the registered adequacy heuristic and was also harmful. Together these observations make a generic undertraining explanation less plausible, but they do not resolve LIN at an adequate budget.

A 5,000-step reported LIN set with a paired 5,000-step R0N is the disciplined resolution.

## MAJOR

3. The main contrast identifies the jointly optimized system, not an isolated head effect.

This is not technically a confound: table co-adaptation is a mediator caused by enabling the head, so LIN/MLP versus R0N validly estimates the policy effect of “allow joint head–table training.”

It does not estimate a standalone head effect.

The existing mechanism artifacts provide a useful conditional decomposition:

- LIN total −0.02441 = head applied to its co-trained table −0.02183, plus co-trained-table/raw-doc difference −0.00258.
- MLP total −0.02932 = −0.02744 plus −0.00188.

Thus most observed OOD harm occurs when the learned head is applied, even on its own co-adapted table. That strongly reduces the “the table alone did it” concern, but remains path-dependent.

To separate the pieces, complete the evaluation cross:

- R0N table × identity head
- R0N table × learned head
- treatment table × identity head — already measured
- treatment table × learned head — already measured

For training-level isolation, use a factorial/staged design with table frozen/trainable × head frozen/trainable.

4. Claim 5 is only an M9 warning, not an M9 result.

The teacher leg directly measures stock teacher query vectors against these particular headed documents. M9’s student is not the teacher: it has approximation error, its own ranking-preservation objective, possible contrastive adaptation, and its own document-tower selection. Those errors may amplify or reduce sensitivity to the head.

The M9 instructions additionally require a CI-resolved direct comparison before breaking the shared pair; E14’s teacher leg has no such CI.

Replace [RESULTS.md lines 170–174](/home/dylan/asymetric-dual-encoders/m8/RESULTS.md:170):

> Frozen teacher queries were more sensitive than the co-trained bag table to these heads on two CQADupStack components. This is an early warning that sharing this particular headed document side with an exact teacher-like query path may carry a cost. It does not estimate the effect on M9’s eventual distilled student or on an E14-LORA document tower; those require direct paired evaluation.

5. Claim 6 is numerically invalid because the endpoints differ.

The −0.0244 number is an OOD macro over two CQADupStack components. The −0.0024 fused number is a four-component macro. Their ratio is not fusion attenuation.

Using the same four components, the dense changes are:

- LIN: −0.01283 versus fused −0.00236: about 5.4× attenuation.
- MLP: −0.01484 versus fused −0.00417: about 3.6×.

Even those ratios describe only these arms at this frozen operating point. `convex0` performs per-query min-max normalization followed by ranking; `w=0.8` is not a quality-share coefficient. Attenuated nDCG does not tell how much system quality is “carried by BM25,” and gains need not be attenuated symmetrically.

Replace [RESULTS.md lines 176–180](/home/dylan/asymetric-dual-encoders/m8/RESULTS.md:176):

> On the four components shared by the dense and fused evaluations, the frozen hybrid attenuated these arms’ dense regressions by about 5.4× for LIN and 3.6× for MLP. This demonstrates robustness of this hybrid at this operating point; it does not identify BM25’s share of quality or imply symmetric attenuation of future gains.

6. “Damaging the space for every query type” hides measured heterogeneity.

The saved dense dump shows:

| Component | LIN | MLP |
|---|---:|---:|
| CQADup programmers | −0.03207 | −0.03910 |
| CQADup physics | −0.01675 | −0.01954 |
| NQ | −0.00661 | −0.00574 |
| HotpotQA | +0.00412 | +0.00502 |
| heldout-train | +0.02132 | +0.02108 |
| heldout-longq | −0.02656 | −0.01524 |

This looks like redistribution/domain-objective mismatch, not universal destruction. The endpoint conclusion remains CQADup-specific, but the broader explanatory story needs these numbers.

7. “Removes the strongest argument for LoRA” is too strong.

The result supplies no positive performance evidence for buying LoRA and raises a concrete OOD-risk signal. But a final-vector head cannot recover discarded information, while LoRA can change the representation upstream. Moreover, your claimed positive relative-alignment pattern could itself be read as motivation for trying a better instrument.

Correct statement:

> E14-HEAD provides no positive performance evidence for escalating to LoRA and raises the expected risk of OOD trade-offs. It is weak negative evidence about E14-LORA and does not decide that probe’s value.

## MINOR

8. The patch null is good but overstated.

R0N versus R0 is persuasive evidence that normalization plus the patched identity path is negligible at the registered resolution. “Introduce no endpoint artifact whatsoever” is absolute and tests only the identity-head path, not trained-head gradients or semantics.

Use:

> R0N versus R0 found no detectable endpoint effect from the identity-head patch path at the measured resolution.

The dense harm also reproduces at fp16—LIN −0.02436, MLP −0.02931—so final int8 folding is not the explanation.

9. Provenance enforcement is incomplete, although current files match.

The loader verifies the table hash but does not verify the recorded head-file or head-state hash before scoring ([e14_score.py](/home/dylan/asymetric-dual-encoders/m8src/e14_score.py:75)). I independently checked the current nine head files, tables, Phase-B checkpoint and E14 source hashes; all presently match their sidecars. This does not suggest the numbers are wrong, but the implementation does less than the registry claims.

Also, `results/m8_e14_head.json` and the mechanism/fused artifacts are presently untracked even though the committed permanent record cites them. For a permanent record, that evidence chain should be versioned.

## Bottom line by claim

- Claim 1: substantially right; say “no detectable identity-path artifact,” not “no artifact whatsoever.”
- Claim 2: defensible for the tested systems, especially dense. Limit LIN’s method-level conclusion.
- Claim 3: not defensible as written. The generic defense is weakened; LIN’s registered inadequacy remains.
- Claim 4: descriptive pattern is real; claimed mechanism is not identified.
- Claim 5: early warning for this head and an exact teacher-like path, not evidence about M9’s actual student or LoRA.
- Claim 6: wrong comparison and wrong attribution.
- Claim 7: too strong; this lowers the case for escalation but does not remove its strongest argument.

So: yes, you have a result. The defensible headline is:

> At 2,500 steps, jointly enabling either renormalized final-vector head caused large, seed-consistent CQADupStack dense regressions; MLP met the registered adequacy heuristic, while LIN remains formally optimization-inadequate. Applying the learned heads hurt frozen-teacher queries more than their co-trained bag tables, a descriptive relative-alignment pattern requiring uncertainty analysis and direct mechanistic follow-up.
