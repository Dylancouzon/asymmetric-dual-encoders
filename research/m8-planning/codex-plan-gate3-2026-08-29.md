# Verdict: STOP

v3 contains substantial real fixes, but it still does not mechanically produce one eligible, shadow-validated artifact. Four structural contradictions remain.

## G2 disposition audit

| G2 | Disposition in v3 |
|---|---|
| G2-1 Stage-R ladder | **Real fix.** All levers now feed one assembled R1 followed by a common-frame R1-vs-R0 gate. R0’s exact instantiation still needs clarification below. |
| G2-2 teacher too late | **Newly contradictory.** The corrected teacher-first rule appears at [PLAN-DRAFT.md:224](/home/dylan/asymetric-dual-encoders/m8/PLAN-DRAFT.md:224), but the preceding paragraph still requires final-frame re-probing, and §5 still says teacher work runs parallel to R and never blocks it ([line 219](/home/dylan/asymetric-dual-encoders/m8/PLAN-DRAFT.md:219), [line 465](/home/dylan/asymetric-dual-encoders/m8/PLAN-DRAFT.md:465)). |
| G2-3 confounded/contaminated teacher screen | **Partial.** The common student frame and fresh protected-query filtering are real fixes. B16 nevertheless uses cached candidates, may act as a pruning rule, and is scheduled in Wave 2 even though teacher selection must open Phase 0. |
| G2-4 post-shadow mutation | **Partial.** The headline sequence and manifest rule are fixed. Quantization and fusion still have unresolved post-selection effects, and §2d permits deleting the shadow gate entirely. |
| G2-5 Stage-S rule | **Real plan-level fix.** Per-family finalists, common comparison frame, equivalence band, complete-cost tie-break, and outcome cases now exist. Freezing exact formulas at LEDGER time is sound. |
| G2-6 C2/table definition | **Newly contradictory.** C2 endpoints are repaired, but “hyperparameter-only changes do not qualify” conflicts with “R1 qualifies iff it adopted any recipe change” ([lines 334–338](/home/dylan/asymetric-dual-encoders/m8/PLAN-DRAFT.md:334)). |
| G2-7 contamination consequences | **Real fix.** Filtering, post-filter hashes/counts, all protected partitions, and recomputed M8 rates are specified. |
| G2-8 genre probe | **Real fix.** Fixed bundle, matched exposure, total-share cap, technical exploratory group, CI and sign guard. |
| G2-9 probe outputs | **Partial/cosmetic.** The global tri-state rule is correct, but B9 still says “allowed into … menu,” B13 permits plural adopted settings, and B16 can become a proxy selection rule ([lines 163–180](/home/dylan/asymetric-dual-encoders/m8/PLAN-DRAFT.md:163)). |
| G2-10 statistics/power | **Real plan-level fix.** The exact raw-CI rule, shared draws, C1/C2/C3 conjunction, worst-group endpoint, and joint-power deliverable are now specified. Deferring executable code and simulation to LEDGER transcription is sound only if completed before every M8 effectiveness number, including teacher probes. |
| G2-11 ONNX vs teacher | **Real fix.** It is parameterized by the selected teacher, feasibility is assessed before teacher freeze, and absence of a ready export is not treated as failure. |
| G2-12 parity | **Real fix.** Final quantized artifact, full fixtures, tolerances, tie policy, nDCG bound, and manifest hashes are present. The stale “one component/bit-identical” sentence is explicitly superseded. |
| G2-13 B4 ceiling | **Real fix.** It is an empirical lower bound and its negative branch is appropriately constrained. |
| G2-14 inherited obligations | **Real for the attacked obligations.** Sqrt has its own slot, D2 explicitly supersedes additive n-grams, and exact ablation mapping is promised. The teacher row in the same matrix separately regresses G2-2. |

## End-to-end pipeline walk

- **Teacher probe:** Cannot yet run coherently. B16 is scheduled after the teacher should already be frozen, and its cached inputs conflict with the clean-screen rule. The fresh filter also requires the shadow and M9-reserve partitions, whose identities are unresolved.
- **Teacher freeze:** Coherent only after deleting the two surviving “re-probe later/parallel R” rules.
- **R1:** Produces an assembled bundle without mutating the teacher. However, R0 is not defined under a changed teacher and current clean data.
- **Stage S:** Produces one family finalist, structurally sound.
- **Seeds:** Occur before shadow and do not mutate a frozen artifact.
- **Quantization:** Not coherent. B12 gates D2 before the finalist exists, then reruns after family selection; a failure can alter eligibility and byte tie-breaking after Stage S.
- **ONNX parity:** Coherent once the quantized artifact is unambiguous.
- **Fusion:** Inputs exist, but R and S already used fused endpoints before the final fusion function is selected. The final system can therefore differ from the system that won those gates.
- **Manifest:** Coherent if quantization and fusion are resolved first.
- **Shadow:** Direct contradiction: §2a makes it mandatory with STOP-on-failure, while §2d permits dropping it.
- **Freeze/reserved access:** Coherent if all preceding contradictions are removed; no post-shadow candidate mutation is otherwise authorized.

## Findings requiring fixes

1. **BLOCKER — Teacher-first has three incompatible execution paths.**  
   Remove the final-frame re-probe language at lines 219–222 and the parallel/non-blocking matrix row. Move B16 into the teacher workstream or make it descriptive-only. If it selects or prunes, recompute it on fresh clean-screen artifacts before teacher freeze. Separate “build/benchmark block-CG” from the later B7 experiment.

2. **BLOCKER — Quantization is circular and the release format is contradictory.**  
   B12 first gates D2 on an old artifact, then reruns after Stage S; meanwhile D2 says “int8 if 4-bit clears,” C2 requires an int8 payload, and ONNX embeds int8 ([lines 166, 189–191, 334–338, 379–381](/home/dylan/asymetric-dual-encoders/m8/PLAN-DRAFT.md:166)). Define one exact release-format state machine—q4/PQ versus int8, bytes, ONNX representation, fallback or STOP—and ensure family eligibility and cost are known before Stage-S selection. The final aggregated-artifact rerun must have a predeclared failure outcome.

3. **BLOCKER — Shadow is both mandatory and optional.**  
   The plan cannot claim byte-identical shadow validation while allowing E10 to delete the gate ([lines 304–315](/home/dylan/asymetric-dual-encoders/m8/PLAN-DRAFT.md:304)). Freeze an acceptable shadow before Phase 0, construct one under a registered procedure, or explicitly redesign the pipeline without claiming G2-4 resolved. Its hash must also exist before teacher/data contamination filtering begins.

4. **BLOCKER — “Qualifying v2 table” can still admit a forbidden hyperparameter-only R1.**  
   Enumerate which R1 changes qualify—e.g. objective, data construction, features, tokenizer—and which do not—seed, steps, temperature, negative count, ordinary tuning. R1-only and R1+D4 must be report-only if no qualifying change survives.

5. **MAJOR — Fusion is selected after gates that already depend on fused performance.**  
   Freeze a deterministic candidate-specific fusion-selection operator before R. Apply the same operator during R, Stage S, and the final aggregated/quantized run; the last invocation may instantiate parameters but may not introduce a new family or tuning rule.

6. **MAJOR — R0 is not defined in the teacher/data frame.**  
   Define `R0(T*)`: selected frozen teacher, current protected-data filters, same data volume, precision and seed policy, with only the registered M7 recipe settings differing. Otherwise “common-frame” R1 validation is not reproducible after a teacher swap.

7. **MAJOR — Operative probe rows still permit discretion.**  
   Replace B9’s “allowed into menu,” make B13 produce one named confirmed setting or fallback/stop, and prohibit B16 from pruning unless separately validated. The general tri-state promise is insufficient while tables say otherwise.

8. **MAJOR — FEVER contamination statements assume the teacher cannot change.**  
   C1/C2 do not share a teacher if M8 swaps away from Stella, so contamination does not necessarily cancel ([lines 445–451](/home/dylan/asymetric-dual-encoders/m8/PLAN-DRAFT.md:445)). Make that statement conditional on retaining Stella. Also replace the stale M7 11.3%/9.32% confirmatory disclosures with placeholders populated from the final M8 mix, as §2f-DATA already requires.

Once these fixes and the named LEDGER deliverables are completed, the plan should be gateable again. As written, transcription would freeze contradictory execution paths.
