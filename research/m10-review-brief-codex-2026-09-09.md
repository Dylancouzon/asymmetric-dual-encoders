# Adversarial review: eleven FIXES applied today. Do they actually do what they claim?

You are the second independent reviewer on an ML research protocol. A Fable review a few minutes
ago raised eleven findings and I applied all of them. **Your job is to review the FIXES, not the
findings.** This project's single most valuable review to date returned NO-GO on a batch of eleven
fixes because they were registry prose that no code ever read — *writing something down is not
making it work*. Assume I have made that mistake again and find where.

## READ EXCLUSION — MANDATORY

Do **NOT** read, grep, glob or open: `results/frozen_eval/untouched-*`, `work/m9reserve/**`, any
reserved qrels cache, `results/perquery.json`. These are RESERVED confirmatory data with one
unspent access; a repo-wide grep once dumped two such sets into a reviewer's context. **No
repo-wide or recursive grep/find.** Read only what is listed below. Name any other file you want
rather than opening it.

## Read exactly these

- `git show 365d7ef` (the fixes) and `git show 0e08a7f --stat`
- `m10/screen_registry.json` — keys `rules`, `confirmation`, `_interpretation`, `arms.E-bs128`
- `m10src/contrasts.py`, `m10src/test_contrasts.py`
- `m10src/nano10.py` (the schedule section only, around `WARMUP_STEPS`), `m10src/trainer10.py`
  (the training loop and its recorded settings), `m10src/run_arm.py` (`fingerprint`, `f_verdict`)
- `m10src/decision_rules.py`, `m10src/test_decision_rules.py`
- `results/m10_screen_verdicts.json`, `results/m10_contrast_A4-A3.json`,
  `results/m10_exposure_table.json`, `results/m10_dcov_gradient_audit.json`

## The eleven fixes, each with the claim I am making about it

| # | fix | my claim |
|---|---|---|
| 1 | `rules.E_warmup_parity` + `nano10.WARMUP_EXAMPLES`/`warmup_steps_for(batch)` | warmup is now equalised in EXAMPLES (64,000) and **the code reads it** — trainer, and the run fingerprint. At bs32 it returns 2,000, identical to the old constant, so **no already-run arm changes its fingerprint**. |
| 2 | `arms.E-bs128.pending: CLOUD_ONLY` + a check in `contrasts.compute` | the disposition is REGISTERED, not inferred from a missing file |
| 3 | sign stability `is not False` → `is True` | an arm with one cycle end can no longer satisfy a clause that requires the last two. No live effect on the ten computed contrasts. |
| 4 | `check_against_arm_record` | the COV file read must be the exact file the committed arm record hashes (`per_query_scores_sha256`) |
| 5 | `best()` in `selection()` | `multi_arm_winner`'s "ties go to the default" is implemented; `max()` on tuples was breaking ties on the label string |
| 6 | `serve_cost_order` branch | implemented for the case where F1 does not resolve |
| 7 | family-A exemption from one source of truth (the registered rule name) | no hardcoded contrast ids |
| 8 | `_interpretation.D1_D2_precondition` rewritten | the clipping claim was asserted, not measured, and is retracted; two candidate mechanisms are named, neither established |
| 9 | `_interpretation._exposure_timing` rewritten | the registry no longer carries two contradictory timing statements |
| 10 | `confirmation.budget_released` recomputed | the freed dose is a range (−5M to +61.7M), not the worst-case 91.7M |
| 11 | `confirmation.synth_20M._VACUOUS_AS_REGISTERED` + `seed_variance_gap` | SYNTH-20M compares the anchor to itself; the seed-variance gap is registered as a reporting obligation with the measured 0.0013869 bound |

## Attack these specifically

1. **Fix 1 is the only one that can change a model.** Verify it end-to-end: does `trainer10`
   actually pass a batch-dependent warmup into `nano10.lr_at`? Does `run_arm.fingerprint` read the
   same function? Is my "bs32 fingerprints unchanged" claim true — check the argument it is called
   with, not just the function. Is there any other place `WARMUP_STEPS` is still read directly and
   would now disagree? And: **is equalising warmup in examples the right fix at all**, or does it
   trade one confound (warmup exposure) for another (bs128 now gets 500 warmup steps, which may be
   too few for a 4× larger batch)? Say which you would rather defend in a paper.
2. **Fix 4 could be a false sense of safety.** It checks the file it is about to read. Does
   anything check the *previous* cycle end used for sign stability, or the arm record itself?
3. **Fixes 5, 6, 7 have no live effect today** — every one is dead code on the current numbers. Is
   any of them wrong in a way a test would not catch? `test_multi_arm_ties_go_to_the_default...`
   in particular tests a **local copy** of the helper, not the helper — I know, and I want you to
   tell me whether that makes it worthless.
4. **Fixes 8–11 are prose.** They change no code and no threshold. Is each one *sufficient*, or
   does one of them still leave a number that some later step reads and acts on? Specifically:
   does anything read `budget_released` or `synth_20M` programmatically?
5. **Is the E decision now sound?** `E-bs128` has not run. The selection carries `batch: PENDING`.
   `rules.E_cost` says "bs32 iff E1 resolves, else bs128". Is PENDING defensible, or am I
   inventing a disposition to avoid an outcome I dislike? Note the build's GPU-hour budget assumes
   bs32 and bs128 is 2.2× cheaper.
6. **The A4−A3 form-match finding.** I claim MedicalQA contributes +0.009885 of the +0.012080
   (81.8%), BRIGHT is net negative, and the mechanism is that A4 adds `health` and `finance` forms
   matching exactly the two families that gained. **Check my arithmetic against the artifact**, and
   tell me whether the conclusion "this is a form-match effect, not a coverage effect" is
   over-claimed in the other direction now.

## How to answer

Per fix: **WORKS / PARTIAL / DOES NOT WORK**, with the file:line that decides it. Then a ranked
list of anything still broken. If a fix is fine, one line. Spend your effort on 1, 4 and 5.
