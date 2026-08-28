# M7 status

**Stage: all ablations and levers closed (2026-08-28 08:07). Candidate `p4n-teacher16-a` served at
`pool_mode=sqrt`, full-suite dev macro 0.6258.** Nothing is running.

Progression: gate#2 0.5987 -> lever#2 0.6113 -> +sqrt pooling 0.6153 -> +teacher-mined negatives
0.6258. Retention 0.931 all-six but **0.857 text-backed**, and CQADupStack retention is still
0.72/0.81 — see the biased-estimator warning in `LEDGER.md` before believing any of it transfers.

## Settled

- **The three lever-#2 decisions STAND** under dependence-preserving statistics, against a newly
  standardized survival bar (signflip p<0.05 AND raw paired CI>0, fp16 **and** int8):
  +0.0065 / +0.0038 / +0.0023, all resolved. `results/m7_dev_audit_full.json`.
- **The matrix shortcut and the released `QueryTable` path are equivalent**: per-query nDCG
  deviation exactly 0, 2 of 161,216 queries with a changed ordered top-10.
- **Capacity lever #4 ADOPTED: `sqrt` count saturation** — Holm rank 1 in both precisions,
  +0.0040 fp16 / +0.0039 int8, positive on all six components, for **zero extra bytes and zero
  query-time cost**. `Preproc.pool_mode` is now part of the frozen rule; conformance is 42/42.
- **All six dev components are hash-pinned**, pool bytes included; the audit and gate refuse to run
  unpinned. Late pin for the two held-out ones, disclosed in `LEDGER.md`.

## Run next, in this order

1. Finish `./run_ablations.sh`: 2 attribution controls → 7 mandatory chains → the **negatives
   ablation the mandate ordered and that never ran** → 1 exploratory chain. Every arm lands in
   `RESULTS.md` whatever it says.
2. **Lever #6 arm (a)**: A-phase only from the candidate's B checkpoint at `pool_mode=sqrt`
   (~5 min). Smoke `sweep.smoke_chain` first — the training forward now takes pooling weights and
   that path has never executed. Arm (b) only if (a) wins.
3. `longspan_probe.py` — does teacher agreement fall with query length? Decides whether a
   long-span distillation chain is worth buying. No qrels, ~10 min.
4. `lever5_shrinkage.py` — pre-registered, eval-only, ~45 min.
5. Teacher learnability probe on `arctic-embed-m-v1.5` and `gte-base-en-v1.5` (registered Specs,
   `validate_encoder.py` first). Both 768-d with no disclosed six-set overlap, so a win there would
   shrink the artifact AND remove stella's ArguAna/FiQA2018 disclosure problem. Measurement only —
   a swap is Dylan's call and costs a full re-encode.
6. Then: fusion re-selection on the candidate → ANN sweep + costs → gate as a mechanical
   eligibility audit → freeze.

7. **TEACHER RE-EXAMINATION, BEFORE THE FREEZE** (pre-registered in `LEDGER.md`). The probes in
   step 5 may justify swapping the teacher and retraining from scratch. If so it happens **before**
   the freeze and the final run — a teacher is a dev-stage decision, and choosing one after seeing
   six-set results, or running the final run twice and reporting the better, is selection on test
   data. Swap bar, tie-break and the full cost (~8-12 h re-encode; levers #4/#5/#6 re-adjudicated)
   are all fixed in the ledger. After the final run, a new teacher is a NEW milestone, not an edit
   to M7's claim.
8. **THE VERY LAST M7 TASK — the clean-stack tax** (Dylan approved 2026-08-28, pre-registered in
   `LEDGER.md`). One arm: the frozen final recipe plus decontaminated MS MARCO, everything else
   identical, to measure what excluding non-commercial training data costs. Runs ONLY after the
   final run and the freeze, so it cannot inform any decision already made. The artifact is
   research-only and never released — `freeze.assert_releasable` refuses any lineage containing an
   msmarco source, and it resolves `sources: []` against the live mix so an empty list cannot slip
   through.

**The final run is NOT scheduled.** It is the one-shot confirmatory access to the six and the
recipe is still moving; it waits for Dylan.

## Open for Dylan

1. Nothing blocking.
2. doc2query revival (licensing ruling on a clean generator) remains yours. The audit notes it was
   closed at 1/8 the published dose with a positive-leaning p=0.085; an N=20 rerun is still
   diagnostic and needs no ruling.
3. Windows Update reboots remain the top operational risk.
