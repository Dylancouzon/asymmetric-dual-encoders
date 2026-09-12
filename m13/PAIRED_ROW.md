# Paired M9 / nano row — registration (ruling R4, Dylan 2026-09-10)

Registered before either final run, as `instructions-m13.md` requires. **Descriptive only.** The row
is the per-query paired difference `nano − M9` in nDCG@10 on the identical frozen qid lists of the
six registered datasets, with a paired bootstrap interval (the registered resample plan,
`m9src.final_stats.draw_plan`). It selects nothing, gates nothing and feeds no conjunct. It is not a
causal coverage experiment: the two systems differ in every axis below at once, so the delta cannot
be attributed to any one of them.

| Axis | M9 frozen candidate (`m9/FREEZE.json`, `m9/registry.json`) | Nano build (`m10/M102_LOCK.md`, `m13/build_config.json`) |
|---|---|---|
| Student | bge-small-en-v1.5, revision `5c38ec7c…` | bge-small-en-v1.5 (same backbone) |
| Head | post-pooling `Linear(hidden, 1024)`, mean pooling over the mask | per-token 1152-wide linear head over layers 12/8/4, pooled after |
| Objective | squared L2 to the stella target on the normalized output | squared L2 (same family) |
| Query corpus | M9 screen pool (esci, hotpotqa-train, squad-train, mrtydi-en, nqopen, triviaqa; fever-train excluded) | A4: M9 pool re-screened and deduplicated, PAQ-build, harvest, 834,463 generated queries (3,486,034 unique texts) |
| Documents | eligible rows of the stella document pool, fever-pos store excluded | the same pool after the M10 re-screen, every eligible document once per epoch, reshuffled per epoch (R5) |
| Mix | per `m9/registry.json:dose.mix_arm` | 75/25 query/document over a 4-step window |
| Dose | 51,670,945 examples, 3.743B tokens, stopped by the plateau rule | 200,000,000 examples, three cycles, no extensions (R13) |
| Batch, schedule | 128; linear warmup then cosine | bs32 selected by completed E1; LoTTE veto skipped under that branch; three linear cycles 1e-4 → 1e-5, 64,000 warmup examples |
| Warm start | per `m9/registry.json:warm_start` | ridge head, 60,000 fit rows, seed 21 |
| Hardware | RTX 3080 box | cloud A100 (or H100 per the day-one measurement) |
| Provenance caveat | two-build-lock disclosure stays beside M9's scores (`m9/STATUS.md`) | build record `results/m13_build_record.json` |

## Row contract

- Both executors run the same `m13src/score13.py` scoring path with an injected student adapter and
  persist string-keyed per-query nDCG@10 on `sorted(frozen qids)` per dataset. The paired row is
  computed only from those two persisted score files, never from re-derived numbers.
- Reported: per-dataset mean delta, avg-6 and clean-4 mean delta, the paired bootstrap 95% interval
  (B and seed from `m10/final_run_registry.json:bootstrap`), and the count of queries where the sign
  differs. No p-value is attached and no threshold exists.
- M9's rows are produced only after every nano recipe decision is fixed (`m13/STATUS.md`); the
  paired row is therefore computed after the nano final run, from files that already exist.
- Forbidden readings: "coverage effect", "data effect", "capacity effect" or any single-cause
  attribution. Permitted reading: "the second build, under a different recipe, scored X relative to
  the first on the same queries."

**Ratification (Dylan, 2026-09-10):** accepted under `m13/RULINGS.md` R4.

Documentation reconciliation2026-09-12: the dose reflects the already-ratified R13 scope cut
and the batch reflects the completed E1/LoTTE branch. The paired estimand, bootstrap, output
contract and descriptive-only interpretation are unchanged; neither final row has been read.
