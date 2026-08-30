Verdict: **STOP.** The draft is not ready to become binding `m8/LEDGER.md` registrations. It contains three direct protocol conflicts, an incomplete confirmatory rule, and an execution plan that does not mechanically produce one candidate.

1. **BLOCKER — The proposed OpenSearch “comparator freeze” burns the reserved panel.**

   - **Attacks:** “[Comparator freeze on the reserved four, NOW](</home/dylan/asymetric-dual-encoders/m8/PLAN-DRAFT.md:152>)”.
   - **Failure:** Scoring OpenSearch requires reserved queries and qrels and creates per-query effectiveness vectors. The mandate says these sets become development-visible “the moment they are scored” and restricts comparator bars to frozen M7, the frozen BM25 builder, and published numbers as context; it explicitly says the absence of comparator vectors must be stated, not papered over ([instructions-m8.md](</home/dylan/asymetric-dual-encoders/instructions-m8.md:21>), [line 36](</home/dylan/asymetric-dual-encoders/instructions-m8.md:36>)). “Computed and sealed” does not reverse access. The reserved payloads are ordinary readable files in the workspace.
   - **Fix:** Delete the pre-training OpenSearch/LR scoring action and the OpenSearch confirmatory leg. Use published six-set numbers as labelled context only. `m7_bars_clean4.json` is not precedent: it performed arithmetic over already-frozen six-set vectors and scored no reserved data.

2. **BLOCKER — The draft falsely says nothing is frozen.**

   - **Attacks:** “Nothing here is frozen” and “all previous settled decisions can be revisited” ([PLAN-DRAFT](</home/dylan/asymetric-dual-encoders/m8/PLAN-DRAFT.md:3>)).
   - **Failure:** Already frozen are the four datasets, paired frozen-M7/frozen-M8 comparison in one M8 access, the three-part statistics family, six-set labelling, comparator sources, and the minimum M7 replacement bar ([instructions-m8.md](</home/dylan/asymetric-dual-encoders/instructions-m8.md:19>)). Dylan’s reopen directive expressly excludes silent or post-number protocol changes ([CLAUDE.md](</home/dylan/asymetric-dual-encoders/CLAUDE.md:138>)).
   - **Fix:** Add an explicit inheritance table: `FROZEN / AMENDABLE BEFORE FIRST M8 NUMBER / OWNER-RULING REQUIRED`. Do this before converting any section into ledger prose.

3. **BLOCKER — The confirmatory statistics omit a frozen mandatory leg and leave the family undefined.**

   - **Attacks:** “Holm … raw CI leg, point gain ≥ +0.005” ([PLAN-DRAFT](</home/dylan/asymetric-dual-encoders/m8/PLAN-DRAFT.md:157>)).
   - **Failure:** The mandate explicitly requires Holm, raw CI, **and simultaneous bound**. M7’s rule requires the raw one-sided lower bound at the Bonferroni family level from the same draws ([m7/LEDGER.md](</home/dylan/asymetric-dual-encoders/m7/LEDGER.md:147>)). The draft does not fix α, hypotheses, family membership, directions, or whether dense/BM25/OpenSearch affect shipping. It is therefore neither executable nor the inherited statistics family.
   - **Fix:** Register exact hypotheses, family α, Holm membership, raw two-sided CI rule, and simultaneous one-sided level `α/m`. Remove OpenSearch. State the inherited weak-null caveat; M7’s procedure was mildly anti-conservative in one calibration and never established uniform weak-null FWER.

4. **MAJOR — Grouping CQA is legally possible now, but the current change is neither explicit nor principled.**

   - **Attacks:** `(FEVER + DBpedia + (android+english)/2)/3`.
   - **Failure:** The paired registration fixes sets and pairing, not unmistakably their weights; levels and bars were deferred. Therefore changing weights is still legal **before any M8/reserved score exists**, but it is an amendment to M7’s inherited equal-component macro and must be named as such. The proposed grouping collapses CQA while leaving the Wikipedia pair separate, giving FEVER+DBpedia two-thirds of the estimand even though both are train-adjacent. It also increases the approximate 95% half-width from 0.0096 to 0.0104 by upweighting 400-query DBpedia. The cited 0.0094 power arithmetic was for equal-four weighting ([premises report](</home/dylan/asymetric-dual-encoders/research/m8-planning/opus-premises-2026-08-28.md:225>)).
   - **Fix:** Prefer inherited equal-four primary plus grouped sensitivity. Alternatively group both known families—Wikipedia pair and CQA pair—and recompute power/calibration before freezing it.

5. **BLOCKER — The primary bar can ship no “v2 table” at all.**

   - **Attacks:** fused-M8 versus fused-M7 as the only primary, with D4 lexical upgrade as a possible winning direction.
   - **Failure:** D4 can leave the query table unchanged or worse and win through BM25F/doc expansion. That does not satisfy the mandate to build a stronger v2 lookup-table encoder ([instructions-m8.md](</home/dylan/asymetric-dual-encoders/instructions-m8.md:7>)). Likewise, fused-M8 versus BM25 does not prevent a weak table from being rescued by BM25; M7 explicitly required dense-alone versus BM25 for that claim ([instructions-m7.md](</home/dylan/asymetric-dual-encoders/instructions-m7.md:28>)).
   - **Fix:** Require a materially changed table and register dense-M8 versus dense-M7 as a release co-condition or non-inferiority guard. D4 may be an auxiliary system upgrade, not the sole M8 direction.

6. **BLOCKER — D1/D3/D5 require scope amendments before registration, not after their probes win.**

   - **Attacks:** two-artifact doc head, per-corpus fitted table, and nonlinear query MLP.
   - **Failure:** D5 is no longer a lookup-only query encoder; D1/D3 change the release from one fixed table against an off-the-shelf document tower. These may be good ideas, but the standing directive requires Dylan’s sign-off to reopen architecture and milestone scope. E1/E3/E5 remain unanswered.
   - **Fix:** Resolve those owner rulings first and amend the mandate explicitly. Until then, exclude those families from the confirmatory candidate menu.

7. **BLOCKER — D3’s “unreadable in code” protection is not enforceable as described.**

   - **Attacks:** documents-only fitting with query/qrel paths “provably unreadable”.
   - **Failure:** A Python convention cannot make plaintext repo files unreadable. The M7 ledger already admits that access control was convention-based because any script can open committed qrels ([m7/LEDGER.md](</home/dylan/asymetric-dual-encoders/m7/LEDGER.md:47>)). B5 on a DEV corpus is fine. Confirmatory corpus adaptation is also defensible in principle as a frozen transductive index-build function, but only if hyperparameters cannot be selected using reserved labels.
   - **Fix:** Run adaptation under OS-level isolation: fixed image/source digest, network disabled, only hashed corpus/doc-vector inputs mounted, no `results/frozen_eval` mount, syscall/open audit, and output schema restricted to table plus provenance. Freeze the λ-by-corpus-size rule and sampling seed before adaptation. If that cannot be implemented, keep D3 research-only.

8. **MAJOR — B1 is not a “bag ceiling,” and the compressed-sensing claim is not proved.**

   - **Attacks:** shuffled-teacher performance as “the ceiling of every order-free query encoder” and the “provably information-preserving up to ~150 tokens” claim.
   - **Failure:** A transformer on unnatural shuffled text measures that transformer’s sensitivity to shuffling, not the best permutation-invariant function. Collapse does not cap DeepSets/MLP-style bag encoders; stability does not show they can attain the teacher. The sparse-recovery arithmetic assumes incoherence/RIP that was never measured for the learned table, and normalization discards scale.
   - **Fix:** Relabel B1 as a teacher order-sensitivity diagnostic. Estimate a bag ceiling with a sufficiently expressive permutation-invariant model trained on an exploratory split and evaluated on a grouped holdout; separately measure token-bag recoverability on the actual table. Do not gate D5 solely on B1.

9. **MAJOR — Three hypotheses are written as diagnoses before their probes.**

   - **Attacks:** “pair-starved, not capacity-starved,” “one-hot to ~1e-4 nats,” and “no teacher ranking information reaches the student.”
   - **Failure:** G2 is train=eval memorization and establishes expressibility only. A 3.7-epoch plateau does not distinguish pair supply from objective, sampling, or optimization. The KL claim substitutes marginal mean/p99 scores for the actual 32-way entropy distribution; B2 exists because that quantity has not been measured.
   - **Fix:** Change these to hypotheses. B2 must report entropy/teacher mass quantiles over the actual sampled candidate sets. B3 must compare equal optimizer updates and sampling exposure and report both dense and fused OOD outcomes.

10. **MAJOR — The LightRetriever evidence is materially misstated.**

   - **Attacks:** LR-pertask as a “corpus adaptation” analogue and M7’s “like-for-like WIN”.
   - **Failure:** LR-pertask uses a different instruction-specific table per dataset, not documents-only corpus fitting ([verification](</home/dylan/asymetric-dual-encoders/research/verification-m2.md:152>)). Its +0.0263 therefore does not anchor D3. Independently recomputing M7 versus LR-websearch from the frozen vectors gives +0.00194, CI [−0.01523, +0.01923], sign-flip p=0.417: unresolved.
   - **Fix:** Call LR-pertask an instruction-oracle comparator. Say M7 has a slightly higher point estimate than LR-websearch, not a win. Treat D3 as an unanchored hypothesis whose evidence begins with B5.

11. **BLOCKER — The “recipe floor” is itself the milestone’s structural direction.**

   - **Attacks:** ICT fraction selection, listwise mining, continuous objective, matched negatives, hyperparameter region, pool rebuild, Wikipedia, and optional synthetic queries all riding “under whichever direction wins”.
   - **Failure:** If the final system wins, the result cannot be attributed to D1–D5. More importantly, B3/B13 select settings on the old B checkpoint and old objective, then transplant them into a different objective, pool, and architecture where their ordering may reverse. “Free riders” recreate M7’s adaptive lever ladder.
   - **Fix:** Either make the recipe rebuild the one primary direction, or freeze one recipe before comparing structural families. Anything selected adaptively becomes part of that direction and consumes the one-shot attribution. No optional riders after shadow-dev results.

12. **BLOCKER — The advertised pre-registered probe gates mostly do not exist.**

   - **Attacks:** “Each has a pre-registered read-out” and “kill/keep each direction”.
   - **Failure:** B1 has no equivalence margin; B4 no ceiling threshold; B5/B6 no registered adoption statistic in the draft; B7 no slope threshold; B8–B10 no selection rules; B12 no quantization margin; B13 selects an undefined “region”. D1–D4 then say merely “gate: Bx”. Multiple outcomes can support multiple surviving directions with discretionary interpretation.
   - **Fix:** For every probe register input hashes, split, endpoint, comparator, exact threshold/CI, multiplicity treatment, tie rule, no-survivor result, and unique mapping to one direction. If more than one survives, a predeclared rule—not judgment—must choose one before shadow access.

13. **MAJOR — The shadow-dev and seed policies still permit table soup.**

   - **Attacks:** plural “family winners” see shadow; “table soup or mechanical median”.
   - **Failure:** Letting several family winners onto shadow and selecting among them makes shadow another tuning set. Existing M7 dev data cannot become genuinely shadow merely by splitting it now; its per-query outcomes already exist. “Soup or median” is not predeclared. Parameter averaging is safe only for identically parameterized, aligned tables—not independently trained MLP heads or tokenizer families.
   - **Fix:** Freeze genuinely unscored M8 data before Phase 0. Select exactly one candidate on exploratory dev; it crosses shadow once as a go/no-go with no fallback. Predeclare one aggregation rule per eligible architecture before seeds run.

14. **MAJOR — The perturbation band is misused, and the power target is not tied to the actual test.**

   - **Attacks:** +0.005 because it is “outside the perturbation band” and “≥+0.02” as a statistical conclusion.
   - **Failure:** The M7 ledger explicitly says the band concerns DEV selection and does not weaken a frozen confirmatory comparison ([m7/LEDGER.md](</home/dylan/asymetric-dual-encoders/m7/LEDGER.md:323>)). A +0.005 point guard is a defensible product margin, but it does not prove a true gain of 0.005; that would require the lower confidence bound to exceed 0.005. The +0.02 sizing is a safety target derived from equal-four arithmetic, not the grouped metric or defined Holm family.
   - **Fix:** Label +0.02 as a planning target only. Power the exact frozen estimand/family. Decide whether +0.005 is merely a point guard or a minimum-effect hypothesis. Add a registered worst-group non-inferiority guard if the system must not buy a win by damaging CQA or DBpedia.

15. **BLOCKER — Full-dose D4 is computationally impossible on this box.**

   - **Attacks:** 30 generated questions per document over the confirmatory corpora.
   - **Failure:** The four corpora contain 10.1M documents, requiring about 303M generations. At the source report’s 300–800 queries/min estimate, generation alone is approximately **263–702 days**. HotpotQA alone would take 136–363 days. Embedding and building the second index come afterward.
   - **Fix:** Remove full-dose dual-index expansion from M8’s confirmatory menu. BM25F can remain. Any question-expansion work must have a bounded document/sample rule and be labelled a research probe, not extrapolated to the reserved system.

16. **MAJOR — “Phase 0 ≈ one week” is not a costed schedule.**

   - **Attacks:** fourteen probes, B14 at 2–4h, B10 under 1h, B7 half-day, plus the unresolved 0.5–4h chain estimate.
   - **Failure:** B14 entails at least 5.55M text-backed DEV documents—and potentially the 6.17M shared pool—so the document-encode estimate is not reconciled with the same report’s ~100 docs/s reserved estimate. B10 multiplies searches by query token count unless ANN is introduced as a confound. B7’s all-1024-RHS block-CG “one second/iteration” is unbenchmarked. Most probes also require new implementation, conformance, provenance, and smoke tests.
   - **Fix:** Benchmark each new path on 10K documents/queries, resolve the chain discrepancy, and publish a serial GPU/RAM/disk schedule before promising a week. Start with a minimal bounding subset rather than all fourteen.

17. **MAJOR — Binding inherited work and source-report items were silently dropped.**

   - **Attacks:** the plan’s claim to synthesize the mandate and reports.
   - **Failure:** There is no disposition for the carried full-chain `sqrt`-training lever, although it requires its own M8 preregistration; the teacher revisit lacks a complete M8 rule; inherited mandatory ablations, ANN sweep, cost reporting, and hardened one-shot mechanics are absent. Touché is proposed despite being explicitly banned in inherited M7 dev protocol; changing that is legal only as an explicit pre-result amendment. The literature’s context-averaged row initialization and MEV approximability probe also disappeared without disposition.
   - **Fix:** Add a requirement matrix covering every inherited mandate item and every source proposal: adopted, rejected with reason, or deferred. Copy the M7 spent-receipt, exclusive lock, strict hashes, atomic write, snapshot, and retry semantics into the M8 one-shot specification.

## Verdict

**Not ready for `m8/LEDGER.md`.** Mandatory before preregistration:

1. Remove all pre-final OpenSearch/LR reserved scoring.
2. State the frozen/amendable boundary explicitly.
3. Freeze an exact executable statistics family including the simultaneous bound.
4. Resolve the macro weighting and recompute power.
5. Resolve E1/E3/E5 and require an actual v2 table.
6. Implement protected-asset isolation, especially for D3.
7. Replace the recipe-floor/free-rider ladder with one mechanically selected direction.
8. Add falsifiable gates that produce exactly one candidate.
9. Repair shadow/seed policy.
10. Drop infeasible full-dose doc2query and cost Phase 0 from measurements.
11. Restore or explicitly dispose of all inherited obligations.

After those fixes, the draft could become a defensible preregistration. As written, converting it would freeze contradictions rather than a protocol.
