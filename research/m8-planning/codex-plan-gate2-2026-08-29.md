## Disposition audit of the prior 17 findings

| Prior finding | Status | Audit |
|---|---|---|
| 1 OpenSearch burns reserve | Resolved | Reserved scoring and the OpenSearch leg are deleted. |
| 2 False “nothing frozen” claim | Resolved | §0 now separates frozen, amendable, and owner-ruled items. |
| 3 Undefined statistics | Partial | α, `m=3`, hypotheses, Holm, and the simultaneous leg exist, but the raw-CI rule and executable resampling specification remain incomplete; C2 is ambiguous. |
| 4 Grouped primary macro | Resolved | Equal-four is primary; grouped weighting is sensitivity-only. |
| 5 Can ship without a v2 table | Partial | C2 was added, but “materially changed table” is undefined and conflicts with R1-only/D4 eligibility. |
| 6 D1/D3/D5 scope changes | Resolved | They remain research-only pending E1/E3/E5. |
| 7 D3 isolation unenforceable | Resolved at plan level | The proposed OS-isolated harness is adequate if implemented; otherwise D3 remains research-only. |
| 8 B1 mislabeled as ceiling | Partial | B1 was correctly relabeled, but B4 is again called a ceiling and its negative outcome is overinterpreted. |
| 9 Diagnoses stated as facts | Resolved | H1–H3 are now explicitly hypotheses. |
| 10 LightRetriever misstatement | Resolved | The websearch result is described as a tie and per-task as an instruction oracle. |
| 11 Recipe free-rider ladder | Not resolved | Stage R is still assembled by accepting many individually selected levers and transplanting them into one combined recipe. |
| 12 Missing probe gates | Not resolved | Several “bars” only authorize more experimentation, transfer conclusions across frames, or leave a discretionary menu. |
| 13 Shadow/seed policy | Not resolved | Only one candidate sees shadow, but its identity subsequently changes through seeds, aggregation, fusion, and potentially teacher selection; the shadow bar is absent. |
| 14 Band/power misuse | Partial | The point guard is now labeled correctly and +0.02 is planning-only, but exact joint power has not been computed and the worst-group endpoint is undefined. |
| 15 Full-dose doc2query infeasible | Resolved | Removed from the confirmatory menu. |
| 16 Uncosted “one week” | Resolved at plan level | Benchmark-first scheduling replaces the unsupported duration claim. |
| 17 Dropped inherited work | Partial | A matrix exists, but explicit n-gram disposition, the carried sqrt arm, and the exact mandatory ablations are still missing or weakened. |

## Findings

1. **BLOCKER — Stage R remains an adaptive lever ladder.**

   - **Attacked text:** “each entering only if its probe clears its registered bar” and the multi-leg R1 assembly rule in [PLAN-DRAFT §2a](/home/dylan/asymetric-dual-encoders/m8/PLAN-DRAFT.md:90).
   - **Failure scenario:** ICT fraction, listwise loss, replay, negatives, hyperparameters, target design, low-rank restriction, pooling, initialization, document instruction, and genre data are each selected on repeatedly reused exploratory dev data and then combined without testing their interactions. A set of individually positive changes can produce a negative assembled recipe. Conversely, selecting among enough noisy +0.005 effects manufactures an optimistic bundle. Several purported R1 legs are not even enumerated in §2a, while the pool rebuild and Wikipedia ICT appear unconditional.
   - **Concrete fix:** Enumerate every R1 degree of freedom and its M7 fallback. Either pre-register one fixed R1 bundle as the recipe direction, or use a fully specified design that produces one bundle. After assembly, require one fresh, common-frame R1-vs-R0 validation gate under matched updates, data volume, seed policy, dense and fused endpoints. No component may be added or removed afterward.

2. **BLOCKER — Teacher selection occurs too late to preserve the meaning of R1 and Stage S.**

   - **Attacked text:** “screen now, re-probe under the final M8 frame” in [§2f](/home/dylan/asymetric-dual-encoders/m8/PLAN-DRAFT.md:189), combined with “runs parallel to Stage R, never blocks it” in [§5](/home/dylan/asymetric-dual-encoders/m8/PLAN-DRAFT.md:352).
   - **Failure scenario:** The teacher determines document vectors, query targets, negative rankings, initialization, output dimension, and possibly tokenizer/byte choices. R1 and the structural winner are selected under Stella; a later teacher swap changes the frame that selected them. Re-probing only the already-selected architecture under the challenger does not repair this—the architecture and recipe might have been different had the challenger been present earlier. The inherited swap bar explicitly required re-adjudicating affected levers and fusion ([M7 ledger](/home/dylan/asymetric-dual-encoders/m7/LEDGER.md:220)).
   - **Concrete fix:** Finish and freeze teacher selection before Stage R. If a final-frame check reverses the teacher decision, restart R1 and Stage S under the new teacher. The only alternative is a fully nested comparison that independently runs recipe and architecture selection for each teacher finalist. All of this must remain dev-only and precede shadow and reserved access.

3. **BLOCKER — The teacher screen can prune on a confounded frame and may reuse contaminated fit queries.**

   - **Attacked text:** the “NEW CG-frame sweep” and closed-form pruning design in [§2f-T](/home/dylan/asymetric-dual-encoders/m8/PLAN-DRAFT.md:196).
   - **Failure scenario:** Re-running Stella controls the solver change, but not student-frame changes. A 30.5K/1024 Stella table, a 50K/768 ModernBERT table, and a 151K/256 Qwen table differ in tokenizer capacity, dimension, regularization, and byte allocation. The screen therefore ranks teacher-plus-student designs, not teachers. Because the screen “prunes,” it can discard a teacher that would win under the final self-trained D2 tokenizer. Separately, the existing M7 closed-form fit list is a stale superset containing 4,582 protected-query hits; its use was disclosed but not repaired ([M7 ledger](/home/dylan/asymetric-dual-encoders/m7/LEDGER.md:274)). The plan does not require a fresh clean fit list.
   - **Concrete fix:** Hold student tokenizer, dimension/byte budget, fit queries, λ search, solver tolerance, and dtype constant within a teacher screen. Treat alternative tokenizer/dimension combinations as architecture candidates, not teacher effects. Regenerate the fit list through the current protected-query filter, log removals by six/reserved/shadow/M9 partition, and forbid stale cache reuse. Add teacher-training provenance against all protected sets. Dev-only teacher probing does not itself spend the reserved access; contaminated fitting data is the actual leak risk.

4. **BLOCKER — The candidate that crosses shadow is not the candidate that gets frozen.**

   - **Attacked text:** the sequence in [§2a](/home/dylan/asymetric-dual-encoders/m8/PLAN-DRAFT.md:105): shadow crossing, then seeds/aggregation, then fusion re-selection; teacher and ONNX work are scheduled elsewhere.
   - **Failure scenario:** A single-seed dense candidate can pass shadow, after which three-seed aggregation creates different table bytes and fusion selection creates a different released system. A teacher swap or final quantization/ONNX export can change it again. The released candidate has therefore never passed the shadow gate. The shadow gate also has no actual statistic, threshold, tie rule, or explicit STOP outcome.
   - **Concrete fix:** Order the pipeline as: teacher freeze → R1 freeze → Stage-S family and variant selection → seed aggregation → final quantization → ONNX parity → fusion selection → immutable candidate manifest and hashes → one shadow go/no-go → freeze → reserved access. Shadow NO-GO means stop with no fallback. Any post-shadow candidate mutation invalidates the crossing.

5. **BLOCKER — Stage-S mechanical selection is not executable and is readily gameable.**

   - **Attacked text:** “highest exploratory-dev worst-group gain among survivors, ties broken by smaller artifact bytes” in [§2a](/home/dylan/asymetric-dual-encoders/m8/PLAN-DRAFT.md:100).
   - **Failure scenario:** The baseline for “gain,” exact group definitions, seed used, precision, training budget, and family-internal winner are unspecified. §2d says “median/worst-group,” while §2a says worst-group. D2 does not specify how 64K versus 128K is selected; D4 uses a different fused gate. A family with more attempted variants receives more chances to win. Exact floating-point ties are effectively nonexistent, so the byte tie-break does no real complexity control. “Artifact bytes” can also omit tokenizer assets, second heads, and document-index costs.
   - **Concrete fix:** Produce one mechanically chosen finalist per family using a fixed within-family rule or nested dev split. Compare all family finalists against R1 on the same named group vector, precision, aggregation, and budget. Define worst-group as an explicit formula. Use a practical equivalence band, then compare complete downloadable bytes and document-index delta. Specify no-survivor, D4-only-survivor, and D4-best outcomes.

6. **BLOCKER — C2 does not define either “materially changed” or a table-only comparison.**

   - **Attacked text:** “dense-M8-table > dense-M7-table” in [§2e](/home/dylan/asymetric-dual-encoders/m8/PLAN-DRAFT.md:263), “R1-only … legitimate v2” in [§2c](/home/dylan/asymetric-dual-encoders/m8/PLAN-DRAFT.md:169), and the exclusion of another unigram table with only better hyperparameters in [§2c](/home/dylan/asymetric-dual-encoders/m8/PLAN-DRAFT.md:177).
   - **Failure scenario:** Different vocabulary does not prevent paired evaluation, but a changed teacher, MRL dimension, document head, or corpus-adapted table means C2 compares complete dense systems, not tables in a common document space. Calling the result a stronger table overattributes the effect. Separately, “materially changed” could mean anything from a different random seed/hash to a new tokenizer. It is unclear when R1-only qualifies or when D4 may be selected.
   - **Concrete fix:** Define C2 as `dense released M8 system > frozen dense M7 system`, with each endpoint’s exact tokenizer, table, document encoder/head, dimension, normalization, precision, and adaptation policy frozen. Do not present C2 as isolating table causality. Separately define a qualifying v2 table ex ante—e.g. a registered change to its generating recipe/features/tokenizer plus a distinct int8 payload; seed-only changes do not count. State exactly when R1-only and R1+D4 satisfy that definition.

7. **BLOCKER — The DATA workstream maps contamination but does not specify its consequences.**

   - **Attacked text:** “licence verdict + contamination map against ALL protected sets” and the automatic Wikipedia/pool additions in [§2f-DATA](/home/dylan/asymetric-dual-encoders/m8/PLAN-DRAFT.md:220) and [§2a](/home/dylan/asymetric-dual-encoders/m8/PLAN-DRAFT.md:96).
   - **Failure scenario:** ICT turns corpus documents into positive training examples. Full Wikipedia and technical corpora can therefore contain exact or near-duplicate FEVER, DBpedia, shadow, or M9-reserve documents. Merely reporting a map does not prevent training on them. The hard-coded FEVER 11.3% and DBpedia 9.32% disclosures are M7 rates; they cease to describe M8 after Wikipedia and new corpora are added. E8’s claim that the reserved four are unaffected is not established until the sweep runs.
   - **Concrete fix:** Before any data probe, register and run query-overlap removal, positive-document/span removal, and source-family disclosure against every protected partition. Freeze post-filter source hashes and counts. Recompute M8-specific overlap rates after the final data mix is known. Distinguish legal reuse, exact benchmark-content contamination, and broader training adjacency. E8 must be decided from those results, not from the current assumption.

8. **MAJOR — The genre-diversity bar is not falsifiable as written.**

   - **Attacked text:** “add the cleared technical corpora … matched arm vs the Wikipedia-only rebuild; … ≥ +0.005 OOD” in [§2f-DATA-2](/home/dylan/asymetric-dual-encoders/m8/PLAN-DRAFT.md:225).
   - **Failure scenario:** The corpus list, total pair count, per-source dose, sampling unit, optimizer exposure, comparator recipe, exact OOD groups, CI rule, and group-sign guard are unspecified. Multiple technical sources are bundled, so a positive result cannot identify what enters R1. The existing OOD dev pair is CQA, not scientific/legal/technical, so the endpoint may be insensitive to the proposed mechanism. A ≤25% per-source cap still permits four technical sources to occupy 100% of updates.
   - **Concrete fix:** Freeze the source set and dose before scoring; hold total examples and optimizer updates constant; define a total technical-share cap; name the exact endpoint and comparator; require a raw CI rule plus group signs; and include a genuinely technical, nonprotected exploratory evaluation group. Decide whether the whole fixed bundle enters or use a predeclared source-selection rule.

9. **MAJOR — Finding 12’s “no-bar-no-run” repair is mostly declarative.**

   - **Attacked text:** the general promise in [§2b](/home/dylan/asymetric-dual-encoders/m8/PLAN-DRAFT.md:122) versus the individual probes.
   - **Failure scenario:** B2 confirms a defective sampler but does not establish that listwise training improves retrieval. B9 merely makes a rider “allowed into the menu.” B13 selects an undefined region and allows multiple later confirmations. B7 promotes a closed-form tokenizer result into a trained R1 architecture. B12 measures quantization on an existing table, although the actual D2 table can have different quantization sensitivity. B15 similarly transfers a closed-form initialization result into a trained system.
   - **Concrete fix:** Every probe needs one output state: adopt a named setting, retain a named fallback, or stop a named direction. Diagnostics such as B2 may trigger a separately registered performance arm but cannot themselves admit a leg. Any conclusion transferred across frames must be reconfirmed on the assembled candidate; B12 must run on the actual D2 finalist. Remove “allowed menu” outcomes.

10. **MAJOR — The confirmatory family is still not fully executable, and power is unevaluated for the conjunction.**

   - **Attacked text:** [§2e](/home/dylan/asymetric-dual-encoders/m8/PLAN-DRAFT.md:253).
   - **Failure scenario:** “Raw CI” does not state whether the inherited requirement is the two-sided 95% `ci95_raw` lower endpoint greater than zero. Bootstrap replicate count, seed, strict qid alignment, resampling algorithm, and shared-draw implementation are deferred. The worst-group guard does not identify fused versus dense endpoints or the exact group formula. Most importantly, shipping requires all three tests, but no joint power calculation exists. The estimated ±0.0096 dissimilar-system interval already makes the +0.005 point guard much smaller than the detectable confirmatory effect; C3’s margin on the unseen reserve is unknown.
   - **Concrete fix:** Before the first M8 probe number, commit executable decision code specifying B, seed, stratified paired resampling, strict alignment, exact raw-CI lower-bound rule, Holm ordering/ties, and α/3 bound. Define the worst-group guard against the fused M7 endpoint. Simulate power for the all-C1/C2/C3 shipping rule across plausible effect vectors and dependence, and publish minimum detectable effects. Grouped sensitivity must remain outside all shipping logic.

11. **MAJOR — ONNX both contradicts the reopened teacher decision and quietly acts as an unregistered performance filter.**

   - **Attacked text:** “stella stays the teacher (… every probed alternative is CI-resolved below)” and “any future teacher candidate must be ONNX-exportable” in [§3](/home/dylan/asymetric-dual-encoders/m8/PLAN-DRAFT.md:304).
   - **Failure scenario:** The first statement is stale once §2f reopens unprobed teachers. The second can exclude a stronger model merely because nobody has attempted its export—even though Stella itself lacks a validated official artifact. That conflicts with “benchmark numbers dominate” unless ONNX eligibility is explicitly elevated to a hard product constraint. If exportability is tested only after final model selection and fails, teacher and all downstream selections change.
   - **Concrete fix:** Parameterize §3 by “the selected teacher.” State explicitly whether ONNX feasibility is a hard eligibility condition or a post-performance engineering obligation. Define acceptable feasibility evidence; absence of an existing ONNX file is not failure. Resolve feasibility before Stage R freezes, or restart the entire downstream pipeline after a teacher change.

12. **MAJOR — ONNX parity is placed and specified too weakly for the artifact that ships.**

   - **Attacked text:** “bit-identical rankings … on one dev component” in [§3.1](/home/dylan/asymetric-dual-encoders/m8/PLAN-DRAFT.md:299).
   - **Failure scenario:** One retrieval component does not cover tokenizer configuration, special tokens, repeated counts, sqrt pooling, truncation, empty queries, dynamic batch/sequence axes, dequantization, or near-zero norms. “Bit-identical rankings” also does not define score/vector tolerances and may fail spuriously on ties. If this is run after shadow or freeze, the production artifact differs from the validated candidate.
   - **Concrete fix:** Run parity on the final aggregated and quantized table before shadow. Require the complete query conformance fixture suite plus vector/cosine tolerances, top-k agreement with a declared tie policy, and an nDCG delta bound on pinned dev. Pin graph, tokenizer, opset, ORT version, precision, and preprocessing hashes in the candidate manifest.

13. **MAJOR — B4 is still not a “ceiling,” and its negative branch can kill the wrong architecture.**

   - **Attacked text:** “Bag-generalization ceiling” and “below ⇒ doc side / lexical arm is where M8 spends” in [B4](/home/dylan/asymetric-dual-encoders/m8/PLAN-DRAFT.md:142).
   - **Failure scenario:** A trained DeepSets model provides an empirical lower bound on what one implementation learned, not an upper bound on all permutation-invariant functions. Failure can result from optimization, capacity, regularization, data volume, or its own selection. Routing the whole milestone away from the query side on that result repeats B1’s original logical error.
   - **Concrete fix:** Rename it an empirical bag-capability probe. Pre-register architecture sizes, optimization checks, seeds, train/holdout split, and saturation evidence. A positive result can establish headroom; a negative result is descriptive unless multiple sufficiently expressive variants converge to the same bound.

14. **MAJOR — The inherited-obligation matrix is partly cosmetic.**

   - **Attacked text:** [§5](/home/dylan/asymetric-dual-encoders/m8/PLAN-DRAFT.md:345).
   - **Failure scenario:** The binding explicit n-gram/phrase-row lever is not separately adopted, rejected, or identified as replaced by D2. The carried full-chain sqrt experiment is made conditional on B10/B13, neither of which falsifies training through sqrt from initialization. “Ablation table for R1 legs” does not preserve M7’s exact mandatory flat-vs-learned weights, prefix variants, initialization controls, dense/BM25/fusion decomposition, and int8 ablation from [instructions-m7](/home/dylan/asymetric-dual-encoders/instructions-m7.md:70).
   - **Concrete fix:** Add one row per binding lever and mandatory ablation. State whether D2 supersedes explicit overlapping n-gram rows and why. Give the redesigned sqrt full-chain arm its own registration or explicitly defer it with owner-approved reasoning. Map every legacy ablation to each eligible architecture, including “not applicable” with a reason.

## Verdict

**STOP — structural work remains.** The draft must not yet be transcribed into binding ledger registrations.

Mandatory fixes:

1. Freeze teacher selection before R1, or require a complete downstream restart after any swap.
2. Replace the per-leg R1 ladder with one fully enumerated assembly rule and a common-frame assembled-recipe validation.
3. Define an executable Stage-S rule and one immutable candidate identity.
4. Move seed aggregation, quantization, ONNX parity, and fusion before the single shadow crossing; register the shadow bar and STOP outcome.
5. Define C2’s endpoints and “materially changed table,” including R1-only and D4 eligibility.
6. Turn teacher/data contamination maps into enforced, hash-pinned filtering and updated disclosures.
7. Finish every probe gate and the exact confirmatory code/power analysis.
8. Reconcile ONNX with the reopened teacher workstream and restore all inherited obligations.
