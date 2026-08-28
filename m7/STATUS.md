# M7 status

**Stage: pre-freeze cleanup, and the candidate moved BACKWARDS today — twice, both times because a
pre-registered rule was honoured against our own interest.** Candidate is **`p35w-2m-s2500` served
at `pool_mode=sqrt`, full-suite dev macro 0.6153.** Not the 0.6258 that has been quoted anywhere.

What changed 2026-08-28, in order:

1. **Lever #4 (count-saturation pooling) failed re-adjudication** on the negatives candidate:
   +0.0033, CI [−0.0010, +0.0073], p=0.063/0.067, nothing clears Holm. Re-running it was
   pre-registered as a consequence of the negatives adoption.
2. **The negatives avenue then closed entirely.** The step-selection rule had never been applied to
   those arms; the corrected arms score +0.0023 (p=0.107), −0.0007 and −0.0056 (a resolved loss),
   so **zero survivors**. Candidate reverts to the pre-negatives artifact — for which lever #4's
   `sqrt` adoption *does* stand, since it was adjudicated there and passed.

**The revert costs almost nothing where it matters.** Macro 0.6225 → 0.6153; **out-of-domain subset
0.3674 → 0.3673.** The whole negatives effect was `heldout-train` +0.0297 and `hotpotqa` +0.0187 —
a seen-document slice and a component whose train split is a training source — while the two
out-of-domain components moved −0.0009 and +0.0013 and `heldout-longq` got worse for every arm.

## The finding that outlives the arms

**The step-selection rule failed, and it prices the noise floor for everything else.** The proxy
peak did not reproduce on re-run (0.5130 → 0.5126); the proxy ranked the three arms *exactly
backwards* from the full suite; and a step count — a nuisance parameter — moved the dev macro by
**0.0049, more than lever #4's adopted effect of 0.0040**.

So: **every CI in this project is a query-sampling interval with no recipe-replication term.**
Training is deterministic, so there is nothing to resample. The bars answer "would another sample
of queries agree", not "would another equally-defensible recipe agree". `run_stepspread.sh` is
measuring the spread across three arms; it is diagnostic and cannot change any adoption.

Read `LEDGER.md` § "THE DEV MACRO IS A BIASED ESTIMATOR" before believing any dev gain transfers.
`m7_dev_reuse_count.json`: **53 trained arms, 299 in-training dev evaluations, 74 eval-only
variants**, Holm applied inside named families only. `retention.py`: retention is **0.926 on the
dev macro but 0.764 on the two out-of-domain components**, where BM25 scores 0.3223 — all six
confirmatory datasets are out-of-domain, so the second figure is the honest one.

## Settled

- Levers **#4 (on the current candidate is moot — it passed there), #5, #6, and negatives**: all
  closed under their own pre-registered bars. `#7 long-span` is pre-registered and unrun.
- **Novelty re-swept 2026-08-28**: claim stands, phrasing weakened to "we found none". New nearest
  miss **KAHM** (arXiv 2605.02950) clears the frozen-doc axis, misses the dense-rows one.
- The ledger's "any doc-side linear map is absorbable" was **half wrong** and is corrected: true
  without renormalization, false with it, and this system renormalizes. Closes the last review's
  open MINOR item.
- `apply_unseen_policy` is a checked non-choice: only 1,743 rows (5.71%) are never trained, 994 of
  them `[unused]` placeholders, and the reachable ones contribute at 0.143x a trained row.

## Running / queued, in order

1. **Simplification arm** (`p5s-simple-nohn-a`) — the baseline changed with the negatives closure,
   so the arm that faces the bar trains with `hard_neg_k=0`. Non-inferiority at δ=0.0040.
2. **Step-count spread** — diagnostic, queued behind it.
3. **Teacher probes** on `arctic-embed-m-v1.5` and `gte-base-en-v1.5` (`run_learnability_base.sh`).
   Both 768-d, no disclosed six-set overlap. A swap is Dylan's call and **must happen before the
   freeze or become a new milestone**.
4. **Lever #7, long-span distillation** (`run_lever7.sh <candidate>`) — the only lever aimed at a
   named weakness of a confirmatory dataset (ArguAna). Primary bar is the length probe; the dev
   suite is a veto, not the instrument.
5. **Adversarial review gate** — Codex plus a reviewer briefed on over-fitting and over-engineering.
6. `./run_freeze_prep.sh <run_id>` → then stop.
7. Post-freeze, pre-registered: teacher re-examination consequences, then the clean-stack tax.

**The final run is NOT scheduled.** It is the one-shot confirmatory access to the six.

## Open for Dylan

1. Nothing blocking.
2. doc2query revival (licensing ruling on a clean generator) remains yours.
3. Windows Update reboots remain the top operational risk.
