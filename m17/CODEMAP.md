# M17 code and artifact map

| Path | Purpose |
|---|---|
| `m17/PLANNING.md`, `registry.json` | Prospective design and draft constants; no runnable training protocol |
| `m17/LEDGER.md` | Owner scope, pre-observation diagnostics, A1 adoption of the three follow-up comparisons, A2 plan-review revision |
| `m17/REVIEW.md` | Dispositions of the two Luna planning reviews |
| `m17src/planning_probe.py` | P0 resource/tokenization check; `--followup` runs P0b only. Hand-authored text, disposable random GPU weights, offline Nano tokenizer comparison. |
| `results/m17_planning_probe.json` | Original P0 observation; original source preserved at `2bca40d` |
| `results/m17_tokenizer_followup.json` | P0b: shared-piece counterexamples, broader illustrative terms, Nano vocabulary equality |
| `research/m17-vocabulary-*.md`, `m17-static-interaction-*.md`, `m17-data-training-*.md` | Three primary-source Luna research notes with exact access logs |
| `research/m17-plan-review-loss-*.md`, `m17-plan-review-data-*.md` | The two Luna planning reviews behind `REVIEW.md` |
| `research/m17-additional-avenues-*.md` | Follow-up idea shortlist that led to A1 |
| `research/m17-astra-plan-review-*.md`, `m17-astra-plan-brief-*.md` | Codex Astra whole-plan review after A2, and its brief |
| `research/m7-data-licensing.md` (Kubernetes row) | The completed step-2a licensing row: pinned SHA, clone route, quoted CC BY 4.0 text and carve-outs, acquisition-terms finding, decontamination coverage. The source is admitted for download only because this row is complete |
| `research/m17-k8s-attribution.md` | The CC BY 4.0 attribution notice, source, pinned revision and modification statement. **Ships with any derived weights**; update the modification statement if the processing changes |
| `results/m17_k8s_source_manifest.json` | Step-2a acquisition record: counts, bytes, JSONL sha256, dev-suite screen result and the deferred protected-screen blocker. No document text |
| `work/m17/sources/kubernetes-website`, `work/m17/sources/k8s_docs_en.jsonl` | Gitignored clone at the pinned SHA and the extracted English docs (path, title, stripped text, sha256). Not admitted training data until the executor's protected screen passes |
| `m17src/common.py` | Paths, registry/freeze loading, hashes and `require_executable` — the status gate every entry point calls. Sets `M7_ENCODER=stella-400M-v5` before legacy imports |
| `m17src/cache.py` | Candidate-cache schema, deterministic construction (quota walk, backfill, tie-break, per-query RNG), provenance counts, identity hash and the B2-format entropy block |
| `m17src/vocab.py` | Term discovery, ranking, selection (minima, abbreviation policy, caps, owner pins), count-weighted row init, `single_word` tokenizer extension, old-vocabulary parity and the P0b drift report |
| `m17src/train.py` | **The one M17 driver.** Arms C/V/L/VL/VL-A, warm-start lineage checks, per-bucket without-replacement streams, the four loss terms, snapshots/recovery/resume, run record. Refuses a real run while the registry is a draft |
| `m17src/export.py` | Fold-once effective rows, registered snapshot averaging, bundle build with the copied frozen `encoder_spec`, and the five M17 gates (separate from M11's, which stay bound to `m7/FREEZE.json`) |
| `m17src/loader_np.py` | Standalone numpy loader: eager fp32 vs resident int8 codes/scales with per-query dequantization, parity check, weight bytes reported separately from process RSS |
| `m17src/evaluate.py` | Exact dense retrieval, per-domain nDCG@10/Recall@10, paired family bootstrap, DBSF@100 through `m12src/qfusion.py`, alias overlap/rank correlation. Dev-suite and panel surfaces are flag-gated and unwired pre-clock |
| `m17src/panel_build.py` | **Step 2c.** Held-out-only draws (`trainmix.heldout`'s mod-50 rule) from SQuAD/HotpotQA/Mr.TyDi plus Kubernetes candidate queries; the heuristic domain classifier; families by shared gold document group and identical normalized text; the 50/50 family split; the declared corpus; the ancestor-manifest exposure screen; the expected-SE block; the seal. `--reuse-cache` re-labels from `work/m17/panel/*_cache.jsonl` without re-reading 2.9 GB of stores |
| `m17src/alias_test_build.py` | **Step 2c.** The 200-pair held-out alias test: source-stated acronym/expansion and alias/canonical equivalences with quoted citations, ambiguous short forms flagged and left `PENDING_HUMAN`, families, and the exclusion file the training-pair builder reads |
| `results/m17_panel_manifest.json` | The sealed panel: per-domain query/family counts, partitions, corpus size, relevant counts, exposure labels, pending counts, expected SE and every file hash. **PROVISIONAL** until the pending judgments are resolved |
| `results/m17_panel_selection.jsonl` | The selection partition mirrored out of `work/` because it is under 2 MB. The audit partition's text stays in `work/m17/panel/audit.jsonl`; only its hash is published |
| `results/m17_panel_pending_judgments.jsonl` | The one human review sheet: 360 Kubernetes (query, candidate) rows and the ambiguous alias pairs. `relevant_yes_no` and `judge` arrive empty and must stay empty until a person fills them |
| `results/m17_alias_test_manifest.json` | Alias-test counts by source/kind/domain/judgment status, the extraction rules and the file hashes. No pair text |
| `work/m17/panel/`, `work/m17/alias/` | Gitignored panel and alias text: `panel.jsonl`, `selection.jsonl`, `audit.jsonl`, `corpus.jsonl`, the two classifier caches, `ancestry_screen.json`, `alias_test.jsonl` |
| `m17src/rehearse17.py`, `m17src/conftest.py`, `m17src/test_*.py` | The tiny synthetic end-to-end rehearsal and the 66 pytest checks that run on it and on fixture-scale inputs |
| `results/m17_rehearsal.json` | The pre-clock rehearsal record: stages, scaled fixture constants, gate results, loader parity, synthetic-only evaluation. Not a quality observation or a rate forecast |

Reproduce diagnostics from the repository root using `.venv/bin/python
m17src/planning_probe.py --gpu` or `--followup`. Both refuse to overwrite their existing result.
For a new measurement, register its reason and use a separately named result in an isolated
checkout; do not remove the recorded observation to make a command pass. The script's source,
bundle and ledger hashes are embedded in each result. Ledger changes after a measurement are
expected; git retains the version bound to that observation.

Future M17 implementation belongs in `m17src/`, small mutable artifacts in `results/m17_*`,
and heavy caches/checkpoints/bundles in gitignored `work/m17/`. Reuse existing modules where they
fit; no generic experiment framework or legacy-path rename.

The step-3 implementation now exists and rehearses end to end, but it is **not** an executable
experiment: the registry is still `DRAFT_NOT_EXECUTABLE`, `train.py` refuses a real arm, and the
real data path (teacher encoding, the admitted-source query pool, bank mining, the panel and
dev-suite readers) is deliberately unwritten — `train.py --data` expects a prepared directory
that step 5 produces. Run the checks with `.venv/bin/python -m pytest -q -ra m17src` and the
rehearsal with `.venv/bin/python m17src/rehearse17.py` (see `HARNESS.md`).

## Reuse hazards to resolve before training

- `m7src/train.py` hardcodes tokenizer/init/data assumptions and invokes development scoring;
  `sweep.one` also appends to M7 reporting files. Do not call old drivers as an M17 launcher.
- `m7src/table.py::QueryTable`, ragged bags, sqrt occurrence weights, folding and normalization
  are reusable primitives, but bind the new tokenizer/table identity and M17 output paths.
- Set `M7_ENCODER=stella-400M-v5` before importing legacy teacher modules; their default may be
  bge-base. Cache dtype, prompt, Dense projection/bias and teacher revision matter.
- The unfolded checkpoint includes learned weights. The release already folds those weights
  into the rows. Verify old-vocabulary parity before constructing new rows; never fold twice.
- New-token IDs belong only to the student. Feed the unchanged raw text through the teacher's
  own tokenizer for targets. Reuse cached targets for unchanged text/teacher preprocessing.
- Alias views share a same-intent query-family split and carry verified pair IDs in the same
  cache schema. Both views count as examples in every arm; only VL-A adds their consistency
  loss. Preserve individual teacher targets and nullable positive labels for each view.
- Keep the registry's late step-bound snapshots as well as time-bound recovery checkpoints.
  Average effective float32 folded rows only; no separate averaging of rows/scalars or mixing
  tokenizers, runs or row scales. Each full run is read as endpoint and fixed average only.
- New whole-word rows change sqrt-count sharing with constituent subwords elsewhere in a query.
  Sum initialization is not universally output-preserving; P0b demonstrates the mechanism.
- `single_word=True` uses word boundaries; underscores and punctuation deserve fixtures.
  Padding stays disabled. Preserve CLS/SEP, empty fallback and truncation parity through numpy,
  the serialized tokenizer, ONNX and the intended FastEmbed loader.
- The numpy int8 path materializes the whole table as fp32. Report process RSS separately from
  compressed weight bytes. P0 GPU rates used synthetic resident tensors and are not a forecast.
- M17 plans resident int8 codes/scales with per-query dequantization. Compare isolated fresh
  processes on identical model bytes; preserve float32 math, unique-ID/count order, fallback and
  truncation. Bench the registry's length/batch strata and retain eager loading if adoption
  checks fail. This does not redesign ONNX/FastEmbed or change the public query API.
- M11's v1 gates are deliberately tied to `m7/FREEZE.json`. Reuse their parity approach for a
  separate M17 bundle; never weaken the old gates so new bytes can masquerade as the v1 freeze.
- Copy the frozen `encoder_spec` into the candidate bundle's document-encoder metadata and
  assert its revision, dimensions, pooling, projection and prefixes match during the build.
  Record training/evaluation bank hashes in provenance. Do not require a production corpus hash
  or a new caller argument: the same encoder space supports different users' document corpora.
- Reuse `m12src/qfusion.py` for fixed DBSF@100. Quality comes from exact runs; ANN measurements
  establish serving costs only.
- No reads of `results/frozen_eval/untouched-*`, reserved qrels caches, `work/m9reserve`, or
  six-set/LoTTE payloads. Existing fingerprint products are not authority to open their inputs.

Verification for this planning batch is JSON/arithmetic consistency, Python compilation,
completed diagnostic output, links and scope diff. Legacy training/evaluation suites are not
needed for documentation and isolated feasibility measurements.
