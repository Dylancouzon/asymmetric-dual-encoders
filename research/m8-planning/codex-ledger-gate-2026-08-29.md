codex
# Verdict: BLOCK — do not activate this ledger

The core estimand is sound, but the protocol is not executable or guard-safe. I found nine blockers. Teacher screens, noise-floor runs, probes, and shadow construction should not run under this text.

## BLOCKER

1. **The ship predicate contains unset variables.**

   Location: [m8/LEDGER.md:284](/home/dylan/asymetric-dual-encoders/m8/LEDGER.md:284).

   Exact defects:

   - “registered margin” for the six-set guard has no value.
   - “reserved group” is undefined: four datasets, three grouped-sensitivity groups, or two domain halves?
   - The `+0.005` guard does not explicitly say `delta_raw(C1) > 0.005`.
   - “qualifying” categories are labels, not a mechanical manifest comparison. “Structural rider” and “data-construction change” can be assigned after seeing dev results.
   - M8’s six-set vectors do not yet exist, so “frozen per-query vectors, zero new access” is false or incomplete.

   Exploit: choose the most favorable grouping/margin, classify ordinary tuning as “data construction,” or score the six during selection and call the later check frozen.

   Fix: register one literal Boolean expression in decision code now, including exact unrounded inequalities, exact group membership and weighting, a numeric six-set margin, a whitelist of qualifying config keys, and the timing of the six-set read. The six must be scored only after the immutable manifest, with no fallback or model change; log it as a separate known-test access or compute it atomically inside the final decision.

2. **C2 does not enforce Dylan’s strict-table ruling.**

   Locations: [m8/LEDGER.md:220](/home/dylan/asymetric-dual-encoders/m8/LEDGER.md:220), [m8/LEDGER.md:424](/home/dylan/asymetric-dual-encoders/m8/LEDGER.md:424), [m8/LEDGER.md:589](/home/dylan/asymetric-dual-encoders/m8/LEDGER.md:589).

   E11 says “the dense table must beat M7’s dense table,” but C2 compares the “dense released-M8 system,” which may include D1’s learned document-side head. The qualifying-table condition merely requires some table change to survive; it does not require that table-alone endpoint to win.

   Exploit: ship a table that loses to M7, rescued by D1, while claiming strict C2 passed.

   Fix: define C2 as M8 query table versus M7 query table against the same frozen incumbent document vectors, with D1 disabled. If D1’s full-system benefit needs confirmation, add a separately registered C4 and redo the family/multiplicity before access.

3. **Stage R, Stage S, and most probes are prose placeholders.**

   Locations: [m8/LEDGER.md:355](/home/dylan/asymetric-dual-encoders/m8/LEDGER.md:355), [m8/LEDGER.md:397](/home/dylan/asymetric-dual-encoders/m8/LEDGER.md:397), [m8/LEDGER.md:467](/home/dylan/asymetric-dual-encoders/m8/LEDGER.md:467).

   Examples:

   - Stage S still says “registered groups, precision, aggregation and budget” but supplies none, and has no equivalence-band value.
   - B2 can trigger a “separately registered” arm that does not exist.
   - B3 has four arms but no exact comparisons, multiplicity procedure, survivor selection, or tie rule.
   - B17 uses `~0.45`, `~0.40`, and “budget split as registered”; no budget is registered.
   - B9/B10 remain `TBD-noise-floor`.
   - Wave 2 lacks endpoint, comparator, bar, multiplicity, and no-survivor fields entirely.
   - The phase-structure test, genre probe, synthetic-query arm, R1 assembly gate, D1/D2 variant selection, and FineWeb data probe lack complete registrations.

   Exploit: select favorable arms or thresholds after observing their values.

   Fix: no affected evaluation may run until every decision is a machine-readable registry row with exact inputs, endpoint formula, comparator, threshold, family, survivor/tie rule, fallback, and immutable registry digest.

4. **The overnight teacher screen has no executable swap rule.**

   Location: [m8/LEDGER.md:519](/home/dylan/asymetric-dual-encoders/m8/LEDGER.md:519).

   Undefined degrees of freedom:

   - “probe components” are not named.
   - “CI-resolved” has no statistic or level.
   - `near-sibling ≈0.005` versus `dissimilar ≈0.0096` leaves the model classification to the session.
   - “does not reverse the sign” could mean each dataset, their macro, point estimates, or CIs.
   - “within noise” in the tie-break has no boundary.

   Exploit: use favorable components, call a challenger near-sibling, and read a positive pooled off-family delta despite one reversing dataset.

   Fix: name the components and weights, exact paired procedure, exact penalty per challenger fixed before its result, exact off-family condition, and exact tie interval. Give the screen a probe ID governed by G1.

5. **The binding pipeline has no legal ordering.**

   Locations: [m8/LEDGER.md:320](/home/dylan/asymetric-dual-encoders/m8/LEDGER.md:320), [m8/LEDGER.md:272](/home/dylan/asymetric-dual-encoders/m8/LEDGER.md:272), [m8/LEDGER.md:461](/home/dylan/asymetric-dual-encoders/m8/LEDGER.md:461), [m8/LEDGER.md:503](/home/dylan/asymetric-dual-encoders/m8/LEDGER.md:503), [m8/NEXT-SESSION.md:55](/home/dylan/asymetric-dual-encoders/m8/NEXT-SESSION.md:55).

   The ledger orders `teacher freeze → noise floor`, while §4.7/§9 require noise before any probe/bar, and teacher freezing requires teacher probes. `NEXT-SESSION` instead puts noise before teacher screens. A later teacher swap invalidates a floor measured in the incumbent frame.

   Fix: register one order, for example: protected inventories/filter → code benchmark → incumbent-frame floor → exact teacher-screen registration → screens/teacher freeze → if swapped, repeat all affected floors → freeze Stage-R bars → Stage R. Explicitly identify which diagnostic thresholds are exempt from noise calibration.

6. **G2 demonstrably fails open.**

   Locations: [m8/LEDGER.md:619](/home/dylan/asymetric-dual-encoders/m8/LEDGER.md:619), current untracked [paths_guard.py:105](/home/dylan/asymetric-dual-encoders/m8src/paths_guard.py:105).

   Concrete bypasses already exist:

   - Reserved Android/English qrels are available outside protected paths in `work/dev/cqadup-{android,english}.json`.
   - The same test payloads and FEVER/DBpedia qrels exist in the Hugging Face cache.
   - A loader can redownload the public datasets.
   - `_HINTS` is checked before `resolve()`, so a symlink alias without `frozen_eval`, `lotte`, or `m9reserve` bypasses classification.
   - Any code can call `claim("m8src.final_run")`; caller identity is not verified.
   - Any code can call `uninstall()`.
   - Pre-opened file descriptors are explicitly unclassified.
   - A claim grants repeated access; it does not verify freeze state, manifest identity, or a spent receipt.
   - The filter claim receives the combined query+qrel files although it needs query text, creating an unrestricted label-bearing process.

   Fix: run all pre-access work in an offline filesystem namespace where protected labels, aliases, and relevant HF caches are absent. Give filtering a separately generated query-only hash inventory. Enforce final/shadow capabilities in minimal processes with freeze/manifest checks, exclusive locks and durable spent receipts. At minimum, inventory duplicate content hashes, guard canonical inodes and dataset loader IDs, resolve before any string hint, remove public `claim`/`uninstall`, and test imports/subprocesses/file descriptors.

7. **G1 is neither implemented nor sufficient by design.**

   Locations: [m8/LEDGER.md:456](/home/dylan/asymetric-dual-encoders/m8/LEDGER.md:456), [m8/LEDGER.md:618](/home/dylan/asymetric-dual-encoders/m8/LEDGER.md:618).

   `m8src/probe_guard.py` does not exist in the current workspace. The promised design checks only whether prose fields exist; it would accept `TBD-noise-floor`, approximate thresholds, or non-executable text. It also does not verify that HEAD is on the pushed remote, and an entry-point-only call is bypassed by directly invoking a helper or evaluator.

   Fix: gate the shared evaluation primitive, not entry points. Require a complete machine registry with no placeholders, verify remote ancestry and exact ledger/registry blob hash, and write that hash into every result before metrics are emitted.

8. **The mandated dependence path cannot produce the third confirmatory leg.**

   Locations: [m8/LEDGER.md:212](/home/dylan/asymetric-dual-encoders/m8/LEDGER.md:212), [m8/LEDGER.md:229](/home/dylan/asymetric-dual-encoders/m8/LEDGER.md:229), [m7src/boot.py:275](/home/dylan/asymetric-dual-encoders/m7src/boot.py:275).

   The ledger says M7’s machinery is unchanged and both dev and confirmatory code take the dependence-preserving route. But `paired_dep()` returns neither `one_sided_lower_raw` nor the α/3 bound; only ordinary `paired()` does. Thus the prescribed route cannot evaluate leg 3.

   Fix: implement an M8 dependence-preserving function returning raw quantiles at exactly `100×0.025/3`, with tests showing it reduces to ordinary stratified resampling for the disjoint reserved four. Do not use the rounded `0.8333` percentile as the authority.

9. **The frozen ONNX product contradicts the pooling probe.**

   Locations: [m8/LEDGER.md:473](/home/dylan/asymetric-dual-encoders/m8/LEDGER.md:473), [m8/LEDGER.md:562](/home/dylan/asymetric-dual-encoders/m8/LEDGER.md:562), [m8/LEDGER.md:599](/home/dylan/asymetric-dual-encoders/m8/LEDGER.md:599).

   B10 may adopt sum/max/top-k/LSE, while §11 fixes the shipped graph as Gather → sqrt-count pool → normalize, and §13 separately keeps a sqrt full-chain arm open.

   Exploit: evaluate one function and silently export another, or discard a B10 win after observing it.

   Fix: either freeze sqrt and remove B10’s alternatives, or define the graph as using the selected frozen pooling operator and require export feasibility/parity as an adoption precondition.

## MAJOR

1. **Post-number amendment is expressly permitted.** [m8/LEDGER.md:22](/home/dylan/asymetric-dual-encoders/m8/LEDGER.md:22) newly says a bar may move harder after its numbers exist, contradicting lines 25–28, §15, and `CLAUDE.md`. Delete it: no affected rule changes after any dependent raw number exists, harder or easier.

2. **Noise-floor mapping is underdefined and one “null” is not null.** [m8/LEDGER.md:272](/home/dylan/asymetric-dual-encoders/m8/LEDGER.md:272). A ±10% step change is a treatment change that can have real effect. “Measured floor” does not say max/mean/CI or how multiple endpoints combine. Register a formula, preferably using true replay and independent seeds, with `bar = max(planning minimum, 2×predefined floor statistic)`.

3. **The LoTTE gate is judgment-driven and incompletely decontaminated.** [m8/LEDGER.md:104](/home/dylan/asymetric-dual-encoders/m8/LEDGER.md:104). “Material overlap” is undefined; exact document hashes miss near duplicates and query leakage; the GO endpoint/comparator/statistic is absent; “the MDE” is not a unique scalar for a multi-leg ship rule. Define near-duplicate query/document checks, a numeric materiality threshold, slice-drop mechanics, and the exact shadow comparison and GO value.

4. **“One-shot mechanics copied verbatim” is not true as an executable port checklist.** Compare [m8/LEDGER.md:337](/home/dylan/asymetric-dual-encoders/m8/LEDGER.md:337) with [m7/LEDGER.md:845](/home/dylan/asymetric-dual-encoders/m7/LEDGER.md:845). Missing explicit obligations include BM25 package/config verification, corpus-only loading, raw-float recovery, re-running releasability/gate checks rather than trusting the freeze, evaluator-source identity, and the ban on post-hoc subgroups. Copy these individually with M8 acceptance tests.

5. **The hash-pin provenance disclosure was dropped.** M7 records that `qtexts_sha256` was computed from already committed payloads, not a fresh download ([m7/LEDGER.md:82](/home/dylan/asymetric-dual-encoders/m7/LEDGER.md:82)); M8 presents the pin without that limitation at [m8/LEDGER.md:55](/home/dylan/asymetric-dual-encoders/m8/LEDGER.md:55). Restore the disclosure: it proves forward immutability, not independent provenance.

6. **M7’s anti-overclaim dev disclosures were dropped.** M7 requires every adoption to report the full dev macro and OOD subset and label concentrated gains “in-distribution only” ([m7/LEDGER.md:592](/home/dylan/asymetric-dual-encoders/m7/LEDGER.md:592)). M8 retains only a generic perturbation-band warning. Reinstate the exact disclosure and the ≈0.005 OOD resolution limitation.

7. **Decontamination has no required count artifact.** [m8/LEDGER.md:142](/home/dylan/asymetric-dual-encoders/m8/LEDGER.md:142) requires filtering and hashes but not M7’s per-rule/per-source removal counts and denominators. Require R1/R2/R3 and near-duplicate counts by source and protected partition before training.

8. **FEVER-excluded results can become an unbudgeted alternate claim.** [m8/LEDGER.md:307](/home/dylan/asymetric-dual-encoders/m8/LEDGER.md:307) says all legs are reported FEVER-excluded but does not explicitly deny ship or claim consequence. State that grouped and FEVER-excluded results are descriptive sensitivity only and cannot rescue or replace any C1–C3 verdict.

9. **The inherited git safety contract has no ledger home.** `instructions-m8.md` imports it; M7 forbids main and force-push ([instructions-m7.md:74](/home/dylan/asymetric-dual-encoders/instructions-m7.md:74)). G3 does not. Add the M8 branch, no-main/no-force-push rule, commit/push cadence, and artifact exclusions.

## MINOR

- The locked release names from [m8/PLAN-DRAFT.md:10](/home/dylan/asymetric-dual-encoders/m8/PLAN-DRAFT.md:10) are absent from the ledger.
- The 233 MB cap does not say whether it applies to the int8 initializer, query ONNX container, or total downloadable system; D1 and tokenizer bytes make this consequential.
- E1–E13 are substantively present, but not literally verbatim. The twelve-plus-one rulings themselves are otherwise intact.

## What is genuinely fine

The equal-weight reserved-four estimand, `m=3`, family α=0.025, Holm handling, strict qid alignment, weak-null caveat, and E12’s exclusion from the family are conceptually correct. The extra ship guards are conjunctive, so they reduce rather than inflate false-positive shipping—provided they are frozen, exact, and applied only after candidate selection is irrevocably over.

Audit contact note: I did not score any protected set or inspect query/qrel values. I performed a schema-only read to establish that label-bearing duplicates and caches exist; it produced no model or metric number.
tokens used
175,312
# Verdict: BLOCK — do not activate this ledger

The core estimand is sound, but the protocol is not executable or guard-safe. I found nine blockers. Teacher screens, noise-floor runs, probes, and shadow construction should not run under this text.

## BLOCKER

1. **The ship predicate contains unset variables.**

   Location: [m8/LEDGER.md:284](/home/dylan/asymetric-dual-encoders/m8/LEDGER.md:284).

   Exact defects:

   - “registered margin” for the six-set guard has no value.
   - “reserved group” is undefined: four datasets, three grouped-sensitivity groups, or two domain halves?
   - The `+0.005` guard does not explicitly say `delta_raw(C1) > 0.005`.
   - “qualifying” categories are labels, not a mechanical manifest comparison. “Structural rider” and “data-construction change” can be assigned after seeing dev results.
   - M8’s six-set vectors do not yet exist, so “frozen per-query vectors, zero new access” is false or incomplete.

   Exploit: choose the most favorable grouping/margin, classify ordinary tuning as “data construction,” or score the six during selection and call the later check frozen.

   Fix: register one literal Boolean expression in decision code now, including exact unrounded inequalities, exact group membership and weighting, a numeric six-set margin, a whitelist of qualifying config keys, and the timing of the six-set read. The six must be scored only after the immutable manifest, with no fallback or model change; log it as a separate known-test access or compute it atomically inside the final decision.

2. **C2 does not enforce Dylan’s strict-table ruling.**

   Locations: [m8/LEDGER.md:220](/home/dylan/asymetric-dual-encoders/m8/LEDGER.md:220), [m8/LEDGER.md:424](/home/dylan/asymetric-dual-encoders/m8/LEDGER.md:424), [m8/LEDGER.md:589](/home/dylan/asymetric-dual-encoders/m8/LEDGER.md:589).

   E11 says “the dense table must beat M7’s dense table,” but C2 compares the “dense released-M8 system,” which may include D1’s learned document-side head. The qualifying-table condition merely requires some table change to survive; it does not require that table-alone endpoint to win.

   Exploit: ship a table that loses to M7, rescued by D1, while claiming strict C2 passed.

   Fix: define C2 as M8 query table versus M7 query table against the same frozen incumbent document vectors, with D1 disabled. If D1’s full-system benefit needs confirmation, add a separately registered C4 and redo the family/multiplicity before access.

3. **Stage R, Stage S, and most probes are prose placeholders.**

   Locations: [m8/LEDGER.md:355](/home/dylan/asymetric-dual-encoders/m8/LEDGER.md:355), [m8/LEDGER.md:397](/home/dylan/asymetric-dual-encoders/m8/LEDGER.md:397), [m8/LEDGER.md:467](/home/dylan/asymetric-dual-encoders/m8/LEDGER.md:467).

   Examples:

   - Stage S still says “registered groups, precision, aggregation and budget” but supplies none, and has no equivalence-band value.
   - B2 can trigger a “separately registered” arm that does not exist.
   - B3 has four arms but no exact comparisons, multiplicity procedure, survivor selection, or tie rule.
   - B17 uses `~0.45`, `~0.40`, and “budget split as registered”; no budget is registered.
   - B9/B10 remain `TBD-noise-floor`.
   - Wave 2 lacks endpoint, comparator, bar, multiplicity, and no-survivor fields entirely.
   - The phase-structure test, genre probe, synthetic-query arm, R1 assembly gate, D1/D2 variant selection, and FineWeb data probe lack complete registrations.

   Exploit: select favorable arms or thresholds after observing their values.

   Fix: no affected evaluation may run until every decision is a machine-readable registry row with exact inputs, endpoint formula, comparator, threshold, family, survivor/tie rule, fallback, and immutable registry digest.

4. **The overnight teacher screen has no executable swap rule.**

   Location: [m8/LEDGER.md:519](/home/dylan/asymetric-dual-encoders/m8/LEDGER.md:519).

   Undefined degrees of freedom:

   - “probe components” are not named.
   - “CI-resolved” has no statistic or level.
   - `near-sibling ≈0.005` versus `dissimilar ≈0.0096` leaves the model classification to the session.
   - “does not reverse the sign” could mean each dataset, their macro, point estimates, or CIs.
   - “within noise” in the tie-break has no boundary.

   Exploit: use favorable components, call a challenger near-sibling, and read a positive pooled off-family delta despite one reversing dataset.

   Fix: name the components and weights, exact paired procedure, exact penalty per challenger fixed before its result, exact off-family condition, and exact tie interval. Give the screen a probe ID governed by G1.

5. **The binding pipeline has no legal ordering.**

   Locations: [m8/LEDGER.md:320](/home/dylan/asymetric-dual-encoders/m8/LEDGER.md:320), [m8/LEDGER.md:272](/home/dylan/asymetric-dual-encoders/m8/LEDGER.md:272), [m8/LEDGER.md:461](/home/dylan/asymetric-dual-encoders/m8/LEDGER.md:461), [m8/LEDGER.md:503](/home/dylan/asymetric-dual-encoders/m8/LEDGER.md:503), [m8/NEXT-SESSION.md:55](/home/dylan/asymetric-dual-encoders/m8/NEXT-SESSION.md:55).

   The ledger orders `teacher freeze → noise floor`, while §4.7/§9 require noise before any probe/bar, and teacher freezing requires teacher probes. `NEXT-SESSION` instead puts noise before teacher screens. A later teacher swap invalidates a floor measured in the incumbent frame.

   Fix: register one order, for example: protected inventories/filter → code benchmark → incumbent-frame floor → exact teacher-screen registration → screens/teacher freeze → if swapped, repeat all affected floors → freeze Stage-R bars → Stage R. Explicitly identify which diagnostic thresholds are exempt from noise calibration.

6. **G2 demonstrably fails open.**

   Locations: [m8/LEDGER.md:619](/home/dylan/asymetric-dual-encoders/m8/LEDGER.md:619), current untracked [paths_guard.py:105](/home/dylan/asymetric-dual-encoders/m8src/paths_guard.py:105).

   Concrete bypasses already exist:

   - Reserved Android/English qrels are available outside protected paths in `work/dev/cqadup-{android,english}.json`.
   - The same test payloads and FEVER/DBpedia qrels exist in the Hugging Face cache.
   - A loader can redownload the public datasets.
   - `_HINTS` is checked before `resolve()`, so a symlink alias without `frozen_eval`, `lotte`, or `m9reserve` bypasses classification.
   - Any code can call `claim("m8src.final_run")`; caller identity is not verified.
   - Any code can call `uninstall()`.
   - Pre-opened file descriptors are explicitly unclassified.
   - A claim grants repeated access; it does not verify freeze state, manifest identity, or a spent receipt.
   - The filter claim receives the combined query+qrel files although it needs query text, creating an unrestricted label-bearing process.

   Fix: run all pre-access work in an offline filesystem namespace where protected labels, aliases, and relevant HF caches are absent. Give filtering a separately generated query-only hash inventory. Enforce final/shadow capabilities in minimal processes with freeze/manifest checks, exclusive locks and durable spent receipts. At minimum, inventory duplicate content hashes, guard canonical inodes and dataset loader IDs, resolve before any string hint, remove public `claim`/`uninstall`, and test imports/subprocesses/file descriptors.

7. **G1 is neither implemented nor sufficient by design.**

   Locations: [m8/LEDGER.md:456](/home/dylan/asymetric-dual-encoders/m8/LEDGER.md:456), [m8/LEDGER.md:618](/home/dylan/asymetric-dual-encoders/m8/LEDGER.md:618).

   `m8src/probe_guard.py` does not exist in the current workspace. The promised design checks only whether prose fields exist; it would accept `TBD-noise-floor`, approximate thresholds, or non-executable text. It also does not verify that HEAD is on the pushed remote, and an entry-point-only call is bypassed by directly invoking a helper or evaluator.

   Fix: gate the shared evaluation primitive, not entry points. Require a complete machine registry with no placeholders, verify remote ancestry and exact ledger/registry blob hash, and write that hash into every result before metrics are emitted.

8. **The mandated dependence path cannot produce the third confirmatory leg.**

   Locations: [m8/LEDGER.md:212](/home/dylan/asymetric-dual-encoders/m8/LEDGER.md:212), [m8/LEDGER.md:229](/home/dylan/asymetric-dual-encoders/m8/LEDGER.md:229), [m7src/boot.py:275](/home/dylan/asymetric-dual-encoders/m7src/boot.py:275).

   The ledger says M7’s machinery is unchanged and both dev and confirmatory code take the dependence-preserving route. But `paired_dep()` returns neither `one_sided_lower_raw` nor the α/3 bound; only ordinary `paired()` does. Thus the prescribed route cannot evaluate leg 3.

   Fix: implement an M8 dependence-preserving function returning raw quantiles at exactly `100×0.025/3`, with tests showing it reduces to ordinary stratified resampling for the disjoint reserved four. Do not use the rounded `0.8333` percentile as the authority.

9. **The frozen ONNX product contradicts the pooling probe.**

   Locations: [m8/LEDGER.md:473](/home/dylan/asymetric-dual-encoders/m8/LEDGER.md:473), [m8/LEDGER.md:562](/home/dylan/asymetric-dual-encoders/m8/LEDGER.md:562), [m8/LEDGER.md:599](/home/dylan/asymetric-dual-encoders/m8/LEDGER.md:599).

   B10 may adopt sum/max/top-k/LSE, while §11 fixes the shipped graph as Gather → sqrt-count pool → normalize, and §13 separately keeps a sqrt full-chain arm open.

   Exploit: evaluate one function and silently export another, or discard a B10 win after observing it.

   Fix: either freeze sqrt and remove B10’s alternatives, or define the graph as using the selected frozen pooling operator and require export feasibility/parity as an adoption precondition.

## MAJOR

1. **Post-number amendment is expressly permitted.** [m8/LEDGER.md:22](/home/dylan/asymetric-dual-encoders/m8/LEDGER.md:22) newly says a bar may move harder after its numbers exist, contradicting lines 25–28, §15, and `CLAUDE.md`. Delete it: no affected rule changes after any dependent raw number exists, harder or easier.

2. **Noise-floor mapping is underdefined and one “null” is not null.** [m8/LEDGER.md:272](/home/dylan/asymetric-dual-encoders/m8/LEDGER.md:272). A ±10% step change is a treatment change that can have real effect. “Measured floor” does not say max/mean/CI or how multiple endpoints combine. Register a formula, preferably using true replay and independent seeds, with `bar = max(planning minimum, 2×predefined floor statistic)`.

3. **The LoTTE gate is judgment-driven and incompletely decontaminated.** [m8/LEDGER.md:104](/home/dylan/asymetric-dual-encoders/m8/LEDGER.md:104). “Material overlap” is undefined; exact document hashes miss near duplicates and query leakage; the GO endpoint/comparator/statistic is absent; “the MDE” is not a unique scalar for a multi-leg ship rule. Define near-duplicate query/document checks, a numeric materiality threshold, slice-drop mechanics, and the exact shadow comparison and GO value.

4. **“One-shot mechanics copied verbatim” is not true as an executable port checklist.** Compare [m8/LEDGER.md:337](/home/dylan/asymetric-dual-encoders/m8/LEDGER.md:337) with [m7/LEDGER.md:845](/home/dylan/asymetric-dual-encoders/m7/LEDGER.md:845). Missing explicit obligations include BM25 package/config verification, corpus-only loading, raw-float recovery, re-running releasability/gate checks rather than trusting the freeze, evaluator-source identity, and the ban on post-hoc subgroups. Copy these individually with M8 acceptance tests.

5. **The hash-pin provenance disclosure was dropped.** M7 records that `qtexts_sha256` was computed from already committed payloads, not a fresh download ([m7/LEDGER.md:82](/home/dylan/asymetric-dual-encoders/m7/LEDGER.md:82)); M8 presents the pin without that limitation at [m8/LEDGER.md:55](/home/dylan/asymetric-dual-encoders/m8/LEDGER.md:55). Restore the disclosure: it proves forward immutability, not independent provenance.

6. **M7’s anti-overclaim dev disclosures were dropped.** M7 requires every adoption to report the full dev macro and OOD subset and label concentrated gains “in-distribution only” ([m7/LEDGER.md:592](/home/dylan/asymetric-dual-encoders/m7/LEDGER.md:592)). M8 retains only a generic perturbation-band warning. Reinstate the exact disclosure and the ≈0.005 OOD resolution limitation.

7. **Decontamination has no required count artifact.** [m8/LEDGER.md:142](/home/dylan/asymetric-dual-encoders/m8/LEDGER.md:142) requires filtering and hashes but not M7’s per-rule/per-source removal counts and denominators. Require R1/R2/R3 and near-duplicate counts by source and protected partition before training.

8. **FEVER-excluded results can become an unbudgeted alternate claim.** [m8/LEDGER.md:307](/home/dylan/asymetric-dual-encoders/m8/LEDGER.md:307) says all legs are reported FEVER-excluded but does not explicitly deny ship or claim consequence. State that grouped and FEVER-excluded results are descriptive sensitivity only and cannot rescue or replace any C1–C3 verdict.

9. **The inherited git safety contract has no ledger home.** `instructions-m8.md` imports it; M7 forbids main and force-push ([instructions-m7.md:74](/home/dylan/asymetric-dual-encoders/instructions-m7.md:74)). G3 does not. Add the M8 branch, no-main/no-force-push rule, commit/push cadence, and artifact exclusions.

## MINOR

- The locked release names from [m8/PLAN-DRAFT.md:10](/home/dylan/asymetric-dual-encoders/m8/PLAN-DRAFT.md:10) are absent from the ledger.
- The 233 MB cap does not say whether it applies to the int8 initializer, query ONNX container, or total downloadable system; D1 and tokenizer bytes make this consequential.
- E1–E13 are substantively present, but not literally verbatim. The twelve-plus-one rulings themselves are otherwise intact.

## What is genuinely fine

The equal-weight reserved-four estimand, `m=3`, family α=0.025, Holm handling, strict qid alignment, weak-null caveat, and E12’s exclusion from the family are conceptually correct. The extra ship guards are conjunctive, so they reduce rather than inflate false-positive shipping—provided they are frozen, exact, and applied only after candidate selection is irrevocably over.

Audit contact note: I did not score any protected set or inspect query/qrel values. I performed a schema-only read to establish that label-bearing duplicates and caches exist; it produced no model or metric number.
