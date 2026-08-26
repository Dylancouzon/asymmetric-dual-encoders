# M7 status

**Stage:** stella swap encode RUNNING (`logs/stella_swap.log`, hotpotqa ~1/3 done ~17:30; realistic
finish ~20:00–20:30, then pool → teacher refs → closed-form ridge, all in the same idempotent
driver). Meanwhile this session closed **every open Codex-gate finding** and tightened
decontamination — all logged in `LEDGER.md` before any number they affect. Detail:
`m7/RESULTS.md` (runs), `m7/LEDGER.md` (protocol), `m7/EXPLORED.md`, `results/m7_*.json`.

## Closed today (dispositions + evidence in LEDGER)

- **B2** pool-negatives decontamination: 4,413 of 6.17M rows banned; `train.py` refuses to run
  unmasked; miners' cache sigs carry the mask digest. `results/m7_decontam_pool.json`.
- **B3** sign-flip randomization p-values (`boot.signflip`), Holm consumes only these; type-I
  verified on real vectors, FWER 0.013 at α=0.025. `results/m7_signflip_calibration.json`.
- **B5** one shared BM25 builder (`fusion.bm25_run`) + `test_fusion_paths.py`; found and fixed a
  real crash (convex on empty runs). p1-objB fusion selection superseded — re-select on stella.
- **B6** `load_beir` writes `m7/SIX_ACCESS.log` (audit, not a lock; concession still goes in report).
- **M-perquery** per-qid hashes frozen + independent BM25 recompute matched 3,727/3,727.
- **M-decontam-short** word-4-grams for 4–7-word queries; R1 family re-run: TRAIN now
  **345,372 pairs** + 220,679 querytext rows. **MINOR-int8-weights** `table.save_release` folds
  weights into rows (exact); G4 gates that shape from now on.
- **Untouched-final repair:** cqadup-android/english added pre-freeze by a deterministic rule;
  measured TRAIN overlap ~0 (vs FEVER 11.3%, DBpedia 9.32%). Wired into decontam/freeze/final_run.
- Ridge cache paths now encoder-tagged (a stale 768-d bge table would have crashed tonight's step 7).

## Run next, in order

1. **Wait for `run_stella_swap.sh` to finish** (idempotent; re-run to resume). The ridge number at
   the end is the first real read on stella retention. Then commit `results/m7_stage0_*` under the
   new tagged name.
2. **Stella objective-B checkpoint** (`s1-objB`: p1-objB's cfg under `M7_ENCODER=stella-400M-v5`;
   trainq re-encode happens automatically — kept.json changed). Then `gate.py s1-objB`.
3. **Phase-2 confirm on stella:** A-only arms from s1-objB, lr {5e-5, 1e-4, 3e-4}, `hard_neg_k=0`,
   `eval_every=500`; **best-eval selection, re-run winner at its best step** (pre-registered in
   LEDGER). One confirmation, not a re-sweep.
4. **Capacity levers, BEFORE freeze (review #2 MAJOR 24 — the confirmed contrastive gain is
   ~0.011 against a ~0.047 bar gap; only capacity can bridge it):** n-gram rows (top-K frequent
   bigrams as extra table rows, closed-form-distilled first), the pseudo-query coverage phase
   (`program.py`, unused), and the reopened cheap doc2query test (EXPLORED.md). Run in that order,
   kill on evidence, log every arm.
5. **Mandatory ablations before freeze** (`instructions-m7.md`; review #2 BLOCKER 7): three inits,
   two prefix variants, flat vs learned weights, int8 (release shape) — then **capacity probe
   re-run under stella** (G2 refuses the bge probe now), fusion re-selection (grid now includes
   convex0), released-table ANN sweep + costs, freeze, single final run.
6. Codex review #2 landed (24 findings, `research/m7-codex-review-2026-08-26b.md`): every
   BLOCKER + the mechanical MAJORs fixed same-day (bank OOB crash, release path wired into
   gate/freeze, probe-teacher guard, containment decontam, tier rule = Holm(sign-flip) AND CI,
   clean-4 in the final scorer, retry guard). Evidence added: significance cross-check
   (10/10 M4-era CIs reproduced from perquery.json — pairing verified for ALL systems), weak-null
   sign-flip simulation. Judgement items (teacher family-exposure read, capacity levers) are in
   the run order above.

## Standing constraints

Six-set claim primary (Dylan 2026-08-26); clean-4 robustness bars precomputed
(`results/m7_bars_clean4.json`); stella's ArguAna/FiQA2018 exposure labelled at the dataset row.
Sequential GPU jobs, 18 GB RAM ceiling, smoke before long runs, commit+push after every experiment.

## Open for Dylan

1. Nothing blocking. Encode + fixes running/landed; confirmation chain queues on the GPU tonight.
2. Host: Windows Update reboots remain the biggest risk to long encodes (one already this morning).
3. Later: HF release go; the stella-exposure presentation question is answered by the pre-registered
   clean-4 robustness bars.
