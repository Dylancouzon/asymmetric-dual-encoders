# Codex pre-freeze review, 2026-08-28 (gpt-5.6-sol, high effort, read-only)

Fifth adversarial pass: post-gate, pre-freeze, briefed on the freeze path, the gate GO,
the fusion selection, the FWER memo, and 'what must be locked before the final run'.
Findings reported verbatim below; dispositions in m7/LEDGER.md § Reviews and audits.

# Verdict: STOP

Do not write the freeze or run `final_run.py` yet. The reported GO numbers are present and their candidate artifact hashes match, but I found eight BLOCKER paths. Several directly permit a wrong fused system, repeated protected access, or a second scored run.

## BLOCKER

1. **The final run does not recheck the frozen BM25 implementation.**

   Freeze-time selection checks package versions in [freeze.py:173](/home/dylan/asymetric-dual-encoders/m7src/freeze.py:173), but final verification only checks fusion family/depth/grid in [freeze.py:450](/home/dylan/asymetric-dual-encoders/m7src/freeze.py:450). The final scorer calls uncached BM25 in [final_run.py:313](/home/dylan/asymetric-dual-encoders/m7src/final_run.py:313); with `cache_path=None`, [fusion.py:162](/home/dylan/asymetric-dual-encoders/m7src/fusion.py:162) does not even call `_pkg_versions()`.

   Failure path: freeze, upgrade `bm25s` or PyStemmer, then run final. Every guard passes, but fusion—and therefore C3—is a different function from the one selected on dev.

2. **Deleting the untracked result resurrects the one shot.**

   `results/m7_final_run.json` is not tracked. `six_already_scored()` treats a missing file as unspent at [final_run.py:99](/home/dylan/asymetric-dual-encoders/m7src/final_run.py:99), while ledger and access-log edits are explicitly allowed during retry at [final_run.py:125](/home/dylan/asymetric-dual-encoders/m7src/final_run.py:125).

   Failure path after a completed run:

   - Delete `results/m7_final_run.json`.
   - Remove the `complete` and digest lines from the ledger, retaining one `FINAL-RUN-BEGIN`.
   - Invoke `--infra-retry`.

   There is no result to trigger `spent`, `n_begin == 1 < 2`, and the same-commit checks pass at [final_run.py:226](/home/dylan/asymetric-dual-encoders/m7src/final_run.py:226). This is the exact editable-ledger threat model the previous fix claimed to close.

3. **Two concurrent launches can both score the six.**

   The guard is read-only and returns at [final_run.py:260](/home/dylan/asymetric-dual-encoders/m7src/final_run.py:260). The first durable `BEGIN` is not written until [final_run.py:478](/home/dylan/asymetric-dual-encoders/m7src/final_run.py:478). Two processes can pass together before either writes the ledger or result. There is no exclusive lock or atomic access claim; both then score, and they also race on the same snapshot and `.tmp` result path.

4. **`--untouched-only` rereads all six protected payloads.**

   `main()` calls `preflight()` unconditionally at [final_run.py:445](/home/dylan/asymetric-dual-encoders/m7src/final_run.py:445). Preflight opens and hashes every six-set query/qrels payload at [final_run.py:335](/home/dylan/asymetric-dual-encoders/m7src/final_run.py:335) and [final_run.py:354](/home/dylan/asymetric-dual-encoders/m7src/final_run.py:354). Only afterward does execution branch into `untouched_only` at [final_run.py:472](/home/dylan/asymetric-dual-encoders/m7src/final_run.py:472).

   Thus the advertised “never goes anywhere near the six” resume performs a second protected read.

5. **Preflight spends access before failure-prone checks and before any durable marker.**

   Preflight reads qrels at [final_run.py:354](/home/dylan/asymetric-dual-encoders/m7src/final_run.py:354). Teacher-code verification occurs afterward at [final_run.py:450](/home/dylan/asymetric-dual-encoders/m7src/final_run.py:450), snapshot copying/hashing afterward at [final_run.py:462](/home/dylan/asymetric-dual-encoders/m7src/final_run.py:462), and `FINAL-RUN-BEGIN` later still.

   A teacher-pin mismatch, disk error, or snapshot race therefore reads the protected labels but leaves neither a result nor a BEGIN. The next invocation is treated as a fresh first run. The message “the six were NOT touched” at [final_run.py:413](/home/dylan/asymetric-dual-encoders/m7src/final_run.py:413) is false under the ledger’s own definition of access.

6. **Freeze accepts a diagnostic subset or edited gate as official GO.**

   `gate.run(components=...)` explicitly permits a subset at [gate.py:75](/home/dylan/asymetric-dual-encoders/m7src/gate.py:75), merely prints a warning, then overwrites the official `m7_gate_<run>.json` at [gate.py:207](/home/dylan/asymetric-dual-encoders/m7src/gate.py:207).

   `assert_gate_passed()` requires only a nonempty condition mapping whose included rows say true at [freeze.py:242](/home/dylan/asymetric-dual-encoders/m7src/freeze.py:242). It does not require:

   - The exact G1–G4 condition set.
   - The exact six dev components.
   - G1’s Stage-0 checkpoint/hash/teacher.
   - The G2 probe hash or dev provenance.
   - Pin evidence, code identity, or the per-query dump hash.
   - Recalculation of decisions from unrounded evidence.

   A one-condition edited gate or subset diagnostic with the same candidate table hashes is accepted.

7. **`final_run` can bypass `freeze.write`’s gate and licence protections entirely.**

   `freeze.write()` calls `assert_releasable()` and `assert_gate_passed()` at [freeze.py:282](/home/dylan/asymetric-dual-encoders/m7src/freeze.py:282) and [freeze.py:302](/home/dylan/asymetric-dual-encoders/m7src/freeze.py:302). But `load_and_verify()` never revalidates either predicate and does not require a gate or lineage block anywhere in [freeze.py:389](/home/dylan/asymetric-dual-encoders/m7src/freeze.py:389).

   Failure path: hand-author or edit `FREEZE.json` before committing/tagging it, retaining valid table, fusion, and manifest hashes but omitting gate/licence evidence. A clean tagged final run accepts it. There is no proof that `freeze.write()` produced the manifest.

8. **The FWER calibration does not simulate the actual family or the full weak null.**

   The final family uses dense for C1/C2 and the released fusion for C3 at [final_run.py:47](/home/dylan/asymetric-dual-encoders/m7src/final_run.py:47). Calibration instead uses one identical candidate vector against all three comparators at [tier_rule_calibration.py:133](/home/dylan/asymetric-dual-encoders/m7src/tier_rule_calibration.py:133). That overstates shared-candidate dependence, the very dependence invoked to explain why the union rate is low.

   It also centers every comparison within every dataset separately at [tier_rule_calibration.py:62](/home/dylan/asymetric-dual-encoders/m7src/tier_rule_calibration.py:62). That samples one narrow sub-null—six dataset means all exactly zero—not the composite macro weak null, which permits positive and negative dataset means to cancel. It is neither a least-favourable construction nor evidence of uniform FWER control.

   Therefore 0.0198/0.0283 are sensitivity results for two constructed null distributions, not a calibration of the actual released rule.

## MAJOR

1. **The present GO is not provenance-clean.** Its own record says `m7src_dirty: true` and names pre-fix HEAD `7f0467e…` at [m7_gate…json:1183](/home/dylan/asymetric-dual-encoders/results/m7_gate_p35w-2m-s2500.json:1183). `gate.py` itself is absent from the source hashes. G1’s Stage-0 bytes and G2’s probe bytes are also unbound. Re-run after fixing the gate schema, from a clean commit.

2. **The fusion report evaluated the wrong artifact.** Selection correctly names `p35w-2m-s2500.release.npz` at [m7_fusion…json:121](/home/dylan/asymetric-dual-encoders/results/m7_fusion_p35w-2m-s2500.json:121), but `fusion_report.py` loads the raw training NPZ at [fusion_report.py:34](/home/dylan/asymetric-dual-encoders/m7src/fusion_report.py:34). Its +0.0356 decomposition and CQADupStack numbers are not proven to describe the release artifact.

3. **Exact `w=0.8` stability is unmeasured.** It leads the runner-up by only 0.00358: 0.572663 versus convex `w=0.7` at 0.569088, from [m7_fusion…json:52](/home/dylan/asymetric-dual-encoders/results/m7_fusion_p35w-2m-s2500.json:52) and [m7_fusion…json:97](/home/dylan/asymetric-dual-encoders/results/m7_fusion_p35w-2m-s2500.json:97). There is no holdout or selection-frequency analysis over the 21-point grid. “Observed dev argmax” is sound; “stable optimum” is not.

4. **The +0.024…+0.031 six-set expectation is not defensible numerically.** Those are two post-selection CQADupStack component effects, not an estimate over new datasets. If transfer occurred only on FiQA, the six-macro contribution would be roughly 0.004–0.005, fully compatible with the evidence. Use the pair only as qualitative evidence that fusion can help on two non-Wikipedia dev domains.

5. **R=2,000 is too coarse for option (d).** The calibration explicitly differs from the final run’s R=100,000 at [m7_tier_rule_calibration.json:2](/home/dylan/asymetric-dual-encoders/results/m7_tier_rule_calibration.json:2). Near 0.00833, an R=2,000 Monte Carlo p-value has sampling noise around 0.002—comparable to the entire claimed 0.0033 FWER excess.

6. **Training-lineage checking has its own TOCTOU gap.** `assert_releasable()` reads policy fields at [freeze.py:67](/home/dylan/asymetric-dual-encoders/m7src/freeze.py:67), then later reopens mutable run records to hash them at [freeze.py:82](/home/dylan/asymetric-dual-encoders/m7src/freeze.py:82). A concurrent replacement can make it validate clean content but record the hash of different content. Read bytes once, parse and hash those same bytes, then recheck before writing.

7. **A crash can permanently omit registered clean-4 robustness.** The confirmatory file is written at [final_run.py:577](/home/dylan/asymetric-dual-encoders/m7src/final_run.py:577), clean-4 is added at [final_run.py:589](/home/dylan/asymetric-dual-encoders/m7src/final_run.py:589), but `--untouched-only` jumps directly to the tail. There is no registered resume path for `clean4_robustness: null`.

8. **The audit artifact rounds away the decisive inputs.** Candidate six-set per-query scores are rounded to six decimals at [final_run.py:562](/home/dylan/asymetric-dual-encoders/m7src/final_run.py:562). The in-memory decision uses unrounded values, but a close result cannot be reproduced exactly from the saved artifact. Persist raw floats or a raw payload plus digest.

9. **Test coverage is not end-to-end.** `test_final_guard.py` tests `guard()` with shell stubs at [test_final_guard.py:60](/home/dylan/asymetric-dual-encoders/m7src/test_final_guard.py:60); it never exercises `main`, ordering, preflight, concurrency, BM25 drift, or resume access. The freeze suite explicitly admits it does not exercise `freeze.write()` end-to-end at [test_freeze_binding.py:13](/home/dylan/asymetric-dual-encoders/m7src/test_freeze_binding.py:13). All BLOCKER paths above are therefore outside current coverage.

## MINOR

- Option (a) still says the weak-null rate is 0.019–0.025 at [m7-fwer-decision…md:103](/home/dylan/asymetric-dual-encoders/research/m7-fwer-decision-2026-08-28.md:103); the superseding result is 0.0198–0.0283.
- Despite claiming qrels are read only from `frozen_eval`, `verify_and_load()` calls `load_beir()` at [final_run.py:268](/home/dylan/asymetric-dual-encoders/m7src/final_run.py:268), which downloads and parses fresh test qrels at [bench/core.py:45](/home/dylan/asymetric-dual-encoders/bench/core.py:45). They are discarded, but the provenance/access statement is false.

## FWER choice

I would take **(c)**: keep the registered rule and report the calibration strictly as sensitivity evidence, not control:

> Sharp-null Holm validity is exact under exchangeability. Under two empirical weak-null constructions, the complete rule rejected at 0.0198 and 0.0283; these simulations do not establish uniform weak-null FWER control.

I would not take (d) as currently phrased. “Tighten until both observed rates are ≤0.025” is tuning on the null simulations and is especially unsafe given the wrong candidate dependence, narrow null construction, and R mismatch.

If (d) is retained, first preregister:

- A finite level grid, seeds, S/R/B, and null-scenario family.
- Dense/dense/fusion candidate dependence matching C1/C2/C3.
- Macro-null scenarios with heterogeneous dataset means.
- Separate calibration and validation simulations.
- A selection rule such as: choose on calibration, then accept only if the one-sided upper confidence bound is ≤0.025 on every validation scenario.

Even then it is empirical control over those scenarios, not a general weak-null guarantee.

The shared sign/bootstrap draws themselves do match the present `SEED=0` implementation for comparisons with identical qid shapes. That part holds; the candidate-vector dependence does not.

## Must be locked before the final run

1. **One-shot semantics and recovery** — write the exact definition of “access spent,” retry eligibility, concurrency ownership, result deletion, clean-4 recovery, and untouched-only behaviour into `m7/LEDGER.md`; implement it in `final_run.py` with an exclusive lock and durable external or append-only spent receipt. It cannot be invented after a crash.

2. **The statistical rule** — close the FWER option, exact lower-bound level, R/B/seeds, null language, and Holm family in the decision memo, ledger, `boot.py`, and tests. Any change after seeing the six is outcome-dependent threshold movement.

3. **Gate schema and provenance** — require exact components/G1–G4; bind Stage-0, capacity probe, references, per-query dump, source hashes, and code commit. Re-run the gate from a clean commit and record that commit. The current gate cannot retroactively acquire provenance.

4. **Fusion status** — rerun the decomposition on the release artifact; decide whether fusion-vs-dense on the six is:

   - A fourth confirmatory hypothesis with multiplicity adjustment, or
   - Pre-registered descriptive evidence only.

   Also lock whether any neighbouring-weight sensitivity will be reported. Alternative weights must never be selected after the six.

5. **Robustness register** — keep clean-4 descriptive, exact dataset rows mandatory, and decide now whether there will be any other fixed subgroup such as exposed-2, query-length, or fusion-vs-dense. Write it in `m7/LEDGER.md` and the final output schema. Do not create new “robust” subsets after seeing results.

6. **Outcome-contingent report/model-card skeleton** — create `m7/REPORT_SKELETON.md`, hash it into `FREEZE.json`, and prewrite wording for every pass/miss combination. It should lock:

   - No headline switching to clean-4 or untouched-final.
   - ArguAna/FiQA exposure labels.
   - All prior accesses and any retry.
   - Dev reuse and fusion selection caveats.
   - Dense artifact versus fused-system distinction.
   - No numerical +0.024…+0.031 transfer forecast.
   - Release-bar miss language.

7. **Runtime/release environment** — freeze and verify installed package versions, not merely `requirements.lock.txt`; include BM25, PyStemmer, torch, transformers, datasets, pytrec-eval and tokenizer/model snapshot hashes. Also lock exact HF files and make clear that the Tier-1 system requires BM25 in addition to the table.

8. **M8/v2 confirmatory partition** — `instructions-m8.md` currently says to reuse M7’s final-run protocol and data at [instructions-m8.md:3](/home/dylan/asymetric-dual-encoders/instructions-m8.md:3). That will not be fresh confirmatory evidence for a learnings-driven follow-up. Before M7 results exist, create `m8/LEDGER.md` and preregister/pin new hidden evaluation sets, splits, metric, primary comparator, family, release rule, and access mechanism.

9. **M7-versus-v2 comparison** — decide now whether it is confirmatory or descriptive. For a valid paired comparison, score both frozen M7 and frozen v2 on the new M8 sets in the same one-shot access. Do not treat M7’s six-set number as a newly confirmatory v2 baseline.

10. **Reserve or burn untouched-final** — decide before running the tail whether FEVER, DBpedia, android and English are M7 descriptive sets or reserved for v2. Once scored, they are development-visible to v2. Record the choice in both ledgers. The same applies to the planned clean-stack-tax six-set read.

## What did hold

- The current candidate release NPZ/meta hashes match both the gate and fusion selection.
- The gate file contains all four expected conditions and they are true; its per-query gzip hashes verify.
- `s1-objB` currently has the correct stella teacher metadata.
- Normal retraining that changes release bytes is caught by table SHA binding.
- Fusion run-id/table/meta/dev-manifest binding and mechanical `released_system=fusion` derivation hold.
- Raw CI endpoints are used for the actual tier decisions.
- Annotated-tag peeling and atomic confirmatory writes are correctly implemented.

No files were changed. I attempted the committed Python guard suite, but this read-only audit environment provides no writable temporary directory; the coverage findings above come from tracing the test and production code directly.
