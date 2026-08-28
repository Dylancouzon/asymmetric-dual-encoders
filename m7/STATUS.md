# M7 status

**Stage: pre-freeze cleanup. Levers closed; the candidate's honest full-suite dev macro is
0.6225, not the 0.6258 that has been quoted** — count-saturation pooling failed its own bar when
re-adjudicated on the negatives candidate, so `pool_mode` is back to `mean` and the sqrt number is
an unadopted arm. Two pre-registered corrections are running (2026-08-28).

Read `LEDGER.md` § "THE DEV MACRO IS A BIASED ESTIMATOR" before believing any dev gain transfers:
four of six dev components are Wikipedia/train-adjacent, and 90% of the negatives gain landed on
the three in-distribution ones. `results/m7_dev_reuse_count.json` now counts the reuse: **53
trained arms, 299 in-training dev evaluations, 74 eval-only variants**, with Holm applied inside
named families only.

## Settled

- **Lever #4 (pooling) does NOT survive on the new candidate**: sqrt +0.0033, CI [−0.0010,
  +0.0073], p=0.063/0.067, nothing clears Holm. Pre-registered as a consequence of the negatives
  adoption, and the outcome is worse than the first adjudication, so it is not a second bite.
  `m7_lever4_pooling_full.json`; the earlier adjudication is archived under its own name.
- **Lever #5 (row shrinkage) — no arm adopted.** **Lever #6 (train-through pooling) arm (a) —
  fails**: +0.0011, p=0.051 fp16 / 0.073 int8. Both closed per their pre-registered bars.
- **Long-span probe**: a gap at the endpoints (overlap@10 0.3443 at 8 words → 0.2997 at 256), but
  not a clean trend — the 64-word bucket sits above the 8-word one. Enough to license lever #7.
- **Novelty re-swept 2026-08-28.** Claim stands; phrasing weakened to "we found none". New nearest
  miss **KAHM** (arXiv 2605.02950) clears the frozen-doc axis and misses the dense-rows one.
  LightRetriever is v5/ICLR+KDD 2026; LEAF's paper (arXiv 2509.12539) now cited — matters for M8.

## In flight

1. **Step-selection rule was never applied to the negatives arms** — self-reported. teacher16 and
   bm2516 peak on the proxy at 1500, mixed32 at 1000, and `warmup_linear` decays over `steps_a`,
   so those are real re-runs. All three re-run; the negatives comparison and its tie-break are
   re-decided on the corrected artifacts. **The proxy picks the step; the full-suite number gets
   no vote, including when the corrected arm scores lower.**
2. **Recipe simplification**, pre-registered as an equivalence test, not a quality lever: drop the
   teacher-context init (30,522 forward passes), IDF weight seeding, `reg_init`, and 2M→500k
   pseudo-queries — all four ablation-inert — and accept only on non-inferiority at δ=0.0040,
   fp16 and int8. One arm, no fallback ladder. If it fails, the measured recipe ships.

## Queued, in order

3. **Teacher probes** on `arctic-embed-m-v1.5` and `gte-base-en-v1.5` (`run_learnability_base.sh`,
   `validate_encoder.py` first). Both 768-d with no disclosed six-set overlap, so a win shrinks the
   artifact AND removes stella's ArguAna/FiQA2018 exposure. A swap is Dylan's call, costs an
   ~8–12 h re-encode, and **must happen before the freeze or become a new milestone**.
4. **Lever #7, long-span distillation** — pre-registered, the only lever aimed at a named weakness
   of a confirmatory dataset (ArguAna) rather than at the dev macro. Primary bar is the length
   probe; the dev suite is a veto, not the instrument.
5. **Adversarial review gate** — Codex plus a reviewer briefed on over-fitting and over-engineering.
   The last moment a finding can change the system.
6. `./run_freeze_prep.sh <run_id>` → then stop. Freeze and the final run are Dylan's calls.
7. Post-freeze, pre-registered: teacher re-examination consequences, then the clean-stack tax
   (MS MARCO research-only variant, refused by `freeze.assert_releasable`).

**The final run is NOT scheduled.** It is the one-shot confirmatory access to the six.

## Open for Dylan

1. Nothing blocking.
2. doc2query revival (licensing ruling on a clean generator) remains yours.
3. Windows Update reboots remain the top operational risk.
