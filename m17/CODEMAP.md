# M17 code and artifact map

| Path | Purpose |
|---|---|
| `m17/PLANNING.md`, `registry.json` | Prospective design and draft constants; no runnable training protocol |
| `m17/LEDGER.md` | Owner scope, pre-observation diagnostics and A1 adoption of the three follow-up comparisons |
| `m17src/planning_probe.py` | P0 resource/tokenization check; `--followup` runs P0b only. Hand-authored text, disposable random GPU weights, offline Nano tokenizer comparison. |
| `results/m17_planning_probe.json` | Original P0 observation; original source preserved at `2bca40d` |
| `results/m17_tokenizer_followup.json` | P0b: shared-piece counterexamples, broader illustrative terms, Nano vocabulary equality |
| `research/m17-*-2026-09-11.md` | Three primary-source Luna research notes and exact access logs |

Reproduce diagnostics from the repository root using `.venv/bin/python
m17src/planning_probe.py --gpu` or `--followup`. Both refuse to overwrite their existing result.
For a new measurement, register its reason and use a separately named result in an isolated
checkout; do not remove the recorded observation to make a command pass. The script's source,
bundle and ledger hashes are embedded in each result. Ledger changes after a measurement are
expected; git retains the version bound to that observation.

Future M17 implementation belongs in `m17src/`, small mutable artifacts in `results/m17_*`,
and heavy caches/checkpoints/bundles in gitignored `work/m17/`. Reuse existing modules where they
fit; no generic experiment framework or legacy-path rename. No M17 training driver exists yet.

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
