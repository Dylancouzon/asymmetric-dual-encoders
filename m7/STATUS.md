# M7 status

**Stage: pre-freeze. Candidate `p35w-2m-s2500` served at `pool_mode=sqrt`, dev macro 0.6153.**
Not the 0.6258 quoted before 2026-08-28. The candidate moved *backwards* twice that day, both
times because a pre-registered rule was honoured against our own interest.

1. **Lever #4 (count-saturation pooling) failed re-adjudication** on the negatives candidate
   (+0.0033, CI [−0.0010,+0.0073], p=0.063/0.067, nothing clears Holm) — a consequence the
   negatives pre-registration had named in advance.
2. **The negatives avenue then closed entirely.** The step-selection rule had never been applied
   to those arms; the corrected arms score +0.0023 (p=0.107), −0.0007, and −0.0056 (a resolved
   loss) — **zero survivors**. The candidate reverted to the pre-negatives artifact, for which
   lever #4's `sqrt` *does* stand.
3. **The recipe simplification failed** non-inferiority (−0.0048, raw CI lower −0.0102 against a
   −0.0040 margin), so the recipe ships unchanged and no back-off ladder was run.

**The revert cost almost nothing where it matters:** macro 0.6225 → 0.6153, out-of-domain subset
0.3674 → 0.3672.

## The finding that outlives the arms

**A nuisance A-phase step count moves the dev macro by 0.0027–0.0078** across three arms at
matched pooling. **Every effect this project has adopted or adjudicated is inside that band** —
lever #4's +0.0040, lever #2's +0.0065/+0.0038/+0.0023, the simplification's −0.0048. Replay noise
is only ~5e-6, so the band is a property of the *recipe choice*, not of re-running. Every interval
in the repo is a **query-sampling** interval with no recipe-replication term: the bars answer
"would another sample of queries agree", not "would another equally-defensible recipe agree".

This bears on **dev selection claims only**. The final run fixes the recipe first and scores frozen
comparator vectors on datasets never used for selection, so the tier comparisons are unaffected.

Read `LEDGER.md` § "THE DEV MACRO IS A BIASED ESTIMATOR" before believing any dev gain transfers.
`m7_dev_reuse_count.json`: **58 trained arms, 322 in-training dev evaluations, 90 eval-only
variants**, Holm inside named families only. `m7_retention_p35w-2m-s2500.json`: retention **0.915**
all-six, **0.846** text-backed, **0.764** out-of-domain, where BM25 scores 0.3223. All six
confirmatory datasets are out-of-domain, so the last figure is the honest one.

## Settled

- Levers **#4** (passed on this artifact, failed on the next — kept because the bar was
  pre-registered and the rule is free), **#5**, **#6**, and **mined negatives**: all closed under
  their own bars. The mandated ablations are finally scored on the full suite and **no arm clears
  Holm over the family of eight**; learned token weights are the one component that clearly earns
  its place (−0.0062 to remove).
- **Novelty re-swept**: claim stands, phrasing weakened to "we found none". Nearest miss **KAHM**
  (arXiv 2605.02950) clears the frozen-doc axis, misses the dense-rows one.
- **Corrections made rather than papered over**: the doc-side-linear-map absorbability claim was
  half wrong; the pseudo-query doses are 924,704 / 324,704, not 2m / 500k (2.85x, not 4x); a
  lever-4 evidence pointer resolved to a different artifact's failure; "deterministic to the last
  digit" read a *rounded* CI, which this ledger's own statistics rule forbids — the real figure is
  4.47e-06; and the headline retention 0.926 was the reverted candidate's, not the shipping one's.

## Running

- **Lever #7, long-span distillation** — the only lever aimed at a named weakness of a
  confirmatory dataset (ArguAna). Realised dose recorded before it trains: pool +0.14% vs the
  short pool (so the one-knob design holds) but only **31.9% long**, and **67.8% of the long spans
  are Amazon product text**. A null therefore closes *this dose and composition*, not the idea.
  ~2 h of teacher encode at 55 texts/s through the esci block.
- **Adversarial review gate** — Codex freeze-gate pass, plus an over-fitting/over-engineering pass
  that has already returned (2 BLOCKER / 6 MAJOR; the two Q6 catches above are its).
- **Teacher probes** on `arctic-embed-m-v1.5` and `gte-base-en-v1.5`, moved to Dylan's M5 Mac on
  branch `m7-teacher-probe-mac`. `gte-modernbert-base` and `granite-r2` are closed on arithmetic:
  the fp64 Gram at V=50,368 is 20.3 GB.

## Queued, in order

1. Action every review finding.
2. Lever #7 verdict → if adopted, lever #4 re-adjudicates on the new artifact.
3. `./run_freeze_prep.sh <run_id>` (fusion re-selection → ANN → costs → gate), then stop.
4. Post-freeze, pre-registered: teacher re-examination consequences, then the clean-stack tax.

**The final run is NOT scheduled.** It is the one-shot confirmatory access to the six, and the
freeze and that run are Dylan's calls.

## Open for Dylan

1. **Nothing blocking *you*** — but the recipe is NOT frozen yet and `p35w-2m-s2500` is still a
   provisional candidate. The queue above has to close first; the freeze and the final run are
   yours to call after it does.
2. doc2query revival (licensing ruling on a clean generator) remains yours.
3. Windows Update reboots remain the top operational risk.
