# M13 stage 1 design — execution readiness (2026-09-10)

Two executors are owed before any GPU is rented: the **200M build controller** and the **guarded
scoring transaction**. Both reuse `m10src`/`m9src`/`m7src` components by import and live in a new
`m13src/`. Registered constants, contrasts and `m10/screen_registry.json` bytes do not change:
the screen verdicts and the F verdict are bound to that registry's hash and `run_arm.py` refuses
a stale one. Surveys behind this design: two read-only Opus passes (build surface, scoring
surface), claims re-derived in-session where they are load-bearing.

## 1. Build controller — `m13src/build13.py`, `m13/build_config.json`, `m13src/test_build13.py`

**Already implemented in `m10src` (reuse, do not rewrite):** three cycles (`run_arm.py:102`,
`nano10.cycle_ends`), linear 1e-4 → 1e-5 (`nano10.lr_at`), warmup in EXAMPLES
(`nano10.WARMUP_EXAMPLES = 64_000`, `warmup_steps_for(batch)`, wired at `trainer10.py:166`; so
`E_warmup_parity` has code behind it), AdamW β/eps (`trainer10.py:119`), wd on dim>1
(`trainer10.param_groups`), clip 1.0, bf16 autocast, `model.train()` after warm start, length
buckets (`data10.length_buckets`), kill/plateau (`nano10.kill_fires`, `plateau_fires`, constants
0.0056 / 0.003 / cycle 3), per-cycle retained checkpoints, durable run receipt + resume, COV at
cycle ends, DEV-6 at the final checkpoint, compile smoke-only (`run_arm.SMOKE_ONLY_KNOBS`).

**Missing, and what the controller adds:**

| Gap | Design |
|---|---|
| 200M pin | `dose = 200_000_000`; `total_steps = dose // batch` must divide exactly (6,250,000 at bs32; 1,562,500 at bs128); cycle boundaries from `nano10.cycle_ends(total_steps, 3)`; the test asserts the three cycle example counts sum to exactly 200,000,000 (bs32: 66,666,656 / 66,666,656 / 66,666,688) and refuses any config totalling 200,100,000 |
| Batch | a config field filled from the E1 verdict record; the controller refuses to start a non-smoke build without that record |
| Extension cycles | after every annealed cycle end k ≥ 3: `gain = m_k − max(m_1..m_{k−1})`; extend iff `gain ≥ 0.003` AND `extensions_run < max_extension_cycles` AND `spent + projected_cycle_cost ≤ ceiling`; else freeze. One extension = 66,700,000 registered examples (bs128 floors to 521,093 steps = 66,699,904 examples; the drop is recorded). LR 1e-4 → 1e-5 over the cycle, `warmup=0` passed explicitly (`lr_at`'s default is 2,000 steps), warm-loaded from `cycle{k}.pt`, data position continued from the GLOBAL example count (CODEMAP pitfall 11) |
| `max_extension_cycles` | a runtime input computed at the day-one benchmark from billed $/h and measured examples/s (`archive .../M102_LOCK.md:126`), written to `state.json` before cycle 1; refused if absent |
| Kill | a registered kill at any annealed end → terminal FAILED record, no `final_checkpoint` |
| Documents | **blocker:** `corpus_loader.arm_doc_count = ceil(dose × 0.25)` demands 50,000,000 unique documents at 200M; the screened M9 pool has ~6.15M and the lock registers "document epochs ≈ 8 over the 6.15M pool". The build arm therefore carries an explicit document policy: all eligible re-screened documents, repeated, reshuffled per epoch with a seed derived from (seed, epoch). Implemented as an additive, registry-driven branch in the loader (default path unchanged; all M10 tests still pass) |
| Corpus | the lock's "query epochs ≈ 37 over 4.0M texts" is the FULL A4, not the screen's cut A4 (2,651,572 unique texts, `data_cut.applies_to = A2, A3, A4/ANCHOR`). The build arm is a new arm entry passed to `assemble_arm(..., registry=)` as the screen registry merged with `m13/build_config.json`; the lock validator checks every data knob equals ANCHOR's except dose, batch, documents and cut |
| Checkpoints | rolling checkpoint on a wall-clock cadence (target ≈ 30 min; `total // 20` would leave ~10 GPU-hours at risk); `extra()`'s full loss history moves to a sidecar JSONL so a checkpoint stays small |
| Provenance | `results/m13_build_record.json`: config/registry/data-manifest/corpus-manifest hashes, git HEAD, environment (GPU, driver, torch, CUDA, instance), billed rate, cumulative spend, dose actually run, cycle macros + evidence hashes, extension decisions, checkpoint hashes, ONNX export hash and fastembed parity result |
| Refusals | compile or any recipe knob outside `--smoke-steps`; non-CUDA outside smoke; invalid/stale screen lock; missing E1 verdict; missing LoTTE gate record (`m13/LOTTE_GATE.json`, written by stage 3, may say `skipped`); missing cap; dose ≠ 200M; existing terminal record |

**State machine.** `preflight → benchmark (writes cap) → cycle 1..3 as ONE train_arm schedule →
at each annealed end: retained ckpt, COV read, append m_k → k ≥ 3: kill? FAILED; gain < 0.003?
PLATEAU-freeze; gain ≥ 0.003 & cap & budget? extension cycle (fresh 1-cycle train_arm, extension-
aware fingerprint); else CAPPED-freeze → freeze: DEV-6, nano10.export_onnx, fastembed parity,
record.` On-disk: `work/m13build/<run>/{state.json, receipt.json, ckpt.pt, cycle{k}.pt, ext{j}.pt,
cov_*.json, losses.jsonl}`. CLI: `build13.py --config m13/build_config.json [--plan] [--benchmark]
[--resume] [--smoke-steps N]`.

**Tests (CPU, synthetic, `Toy` model as in `test_run_arm.py:311-352`):** cycle arithmetic at
bs32/bs128; 200.1M refused; bs128 extension floor recorded; SIGKILL-and-resume equivalence across
a cycle boundary and mid-extension (weights, macros, example count); extension accept (gain
0.004), reject (0.002), cap reached, budget reached before cap; kill at an extension end →
FAILED; each refusal above; the document policy yields every eligible document once per epoch
with a different order per epoch and a position that is a pure function of the global step.
**Box smoke:** the whole controller with stub evaluators, plus a short real bs32 run at max_len
512. **Cloud-only:** bs128 at real lengths (pitfall 13), the 200M dose, the day-one benchmark.

## 2. Scoring transaction — `m13src/score13.py`, `m13src/access13.py`, `m13src/rehearse13.py`

`m10src/final10.py` stays a pure decision module (conjuncts, sequence, pass rule, bootstrap,
sign-flip, registry guard, headline; ~60 tests). The executor is a sibling that imports it.

**Outside the transaction (no protected reads):** flock; spent-tag state (fails closed, pins
`origin_url`); clean tree and HEAD pushed; freeze + checkpoint sha; `ratified_by_owner`;
`results/perquery.json` sha == `comparator_source.sha256`; ≥ 120 GB free (inherited
`min_free_gb`, currently in no M10 registry field and no code); serving-parity artifacts present;
the six document caches present (`work/enc/final-six-<ds>-docs-fp32-*`, 17M–670M each) and
shard-clean; **manifest-only qid check**: `sha(sorted(perquery[ds].qids))` against
`results/eval_manifest.json:datasets[ds].qids_sha256`, lengths and duplicates.

**Inside:** ledger BEGIN commit + push → annotated tag + push (delete local tag if the push
fails) → per dataset in `partitions.all6`, never `bench/core.py:DATASETS` (five by default unless
`m7src/_paths` is imported first): `final_run.verify_and_load(ds, "six")` → document vectors via
`teacher.encode_cached(..., verify=True)` (never re-encode; a foreign cache would re-encode, not
silently pass) → student `encode_queries` (torch path; ONNX emits token embeddings and is the
parity surface, not the quality surface) → anchor bge-small encode → `evalkit.topk_ids_scores`
→ `per_query_ndcg` (pytrec_eval silently omits a qid without qrels, so the scored qid set is
asserted equal to the frozen list and to the comparator's, as strings) → atomic per-dataset
write of string-keyed scores + run-time registry sha → bridge: hard-fail on qid-set inequality
or dataset |mean Δ| > 0.003, report per-query movement, discard the anchor row → `evidence_for`
×4 → `decide` → `headline` → reserved batch iff `reserved_batch_runs`, per-system atomic resume
→ END + digest push. Crash before the first score with the tag on origin = access lost; say so
and exit nonzero. Continuation pins HEAD == BEGIN commit and live registry sha == persisted;
`--recover` compares outside `decide()`.

**One executor for M9 and nano:** the student is an injected adapter
(`load_student(freeze) → encode(texts) → ndarray`; M9's `m9src/nano.py` differs from `Nano10`);
the decision layer is selected by registry (nano → `final10`; M9 → `final_stats` +
`final9.decide`, untouched); separate tags (`m9-six-spent`, `m10-six-spent`). M9 rows are not
produced until every recipe decision is fixed.

**Synthetic rehearsal (`work/m13rehearse/`):** six fake datasets named exactly as `all6` (~200
docs / 30 queries each, generated qrels), a fake frozen-eval payload and a fake comparator in
`perquery.json`'s schema with its sha injected into a COPY of the registry, a
`hf-internal-testing/tiny-random-BertModel` student, and a bare `git init` origin so BEGIN, tag
and push are real. All path constants come from one injected config so production code runs
unmodified. **Tests:** comparator sha mismatch refused pre-tag; five-set env refused; qid
misalignment (missing, reordered, int-vs-str) refused; SIGKILL before the first write → tag
present, no scores, `--recover` reports loss and exits nonzero; SIGKILL after dataset 3 →
continuation opens only 4–6 and refuses when HEAD or registry sha moved; undeclared drift
refused; reserved batch runs iff a conjunct rejected.

## 3. Owner rulings needed before protected access (none block stage 1 code)

| # | Question | Recommendation |
|---|---|---|
| R1 | Preflight standard: registry `implementation` cites M7's payload-reading preflight; `m13/EXECUTION.md` says manifests only. Which? | Manifest-only outside; the payload-level length/duplicate/qid checks run INSIDE, first after the tag. Amend the registry, dated |
| R2 | Re-encoding bridge contract: dataset-mean 0.003 alone, or add a per-query bound? | Keep 0.003 as the sole hard gate; rehearse re-encoding on open data first and register the measured per-query envelope as report-only |
| R3 | M9 close-out still carries the withdrawn 0.0003 per-query bridge | Dated M9-specific amendment adopting the dataset-mean bridge, written before any M9 score exists |
| R4 | Paired M9/nano recipe-delta row | Register a one-page delta in `m13/` referenced by both executors, descriptive only |
| R5 | Build document policy (all eligible documents, ≈8 epochs, reshuffled per epoch) and full uncut A4 | Confirm this reading of the archived lock (:77-78); the controller implements it as registered, disclosed |
| R6 | `ratified_by_owner` flip | In the same commit that pins the reviewed executor, not before |
| R7 | Reserved-batch allowance and `min_free_gb` 120 in M13's allocation and registry | Add both to the M13 allocation and an M10 registry field, dated |
| R8 | LoTTE metric, seven slices and identities | Stage 3 registration; blocks access, not code |
