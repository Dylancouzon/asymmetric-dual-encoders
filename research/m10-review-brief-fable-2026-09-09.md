# Adversarial review: the M10 screen amendment and its ten contrast verdicts

You are reviewing an ML research protocol, adversarially. I state below what I believe. **Your job
is to break it.** A review that returns "looks good" is a badly written brief, not a clean bill of
health — so find the strongest objection you can, and say plainly which of my claims is wrong.

## READ EXCLUSION — MANDATORY, THIS IS NOT TIDINESS

Do **NOT** read, grep, glob, list or otherwise open any of:

- `results/frozen_eval/untouched-*`
- `work/m9reserve/**`
- any reserved qrels cache
- `results/perquery.json`

These hold RESERVED confirmatory evaluation data (queries AND qrels) with one unspent access. A
repo-wide grep once dumped two of these sets into a reviewer's context and potentially contaminated
its recommendations. **Do not run a repo-wide or recursive grep/find.** Read only the files named
below. If you believe you need another file, name it in your output and say why; do not open it.

## Read exactly these

1. `m10/screen_registry.json` — the authority. Rules, contrasts, arms, statistics.
2. `m10/AMENDMENT_STAGED_2026-09-08.md` — the amendment I applied today, verbatim.
3. `m10src/contrasts.py` — the contrast step I wrote today (new file).
4. `m10src/test_contrasts.py` — its tests.
5. `m10src/cov_macro.py` — the estimator and the bootstrap.
6. `m10src/decision_rules.py` — the three amended rules, executable.
7. `results/m10_screen_verdicts.json` — the selection.
8. `results/m10_contrast_A3-A2.json`, `results/m10_contrast_A4-A3.json`,
   `results/m10_contrast_D2.json`, `results/m10_contrast_G1.json` — four representative records.
9. `results/m10_dcov_gradient_audit.json` — the D-COV optimizer-scale audit.
10. `git show --stat HEAD` and `git log --oneline -5` for what changed.

## What happened today, in order

The W8 band-1 screen chain finished (`work/m10arms/rest_chain.log`: `rest chain COMPLETE
Wed Sep 9 14:13:48`), 13 arms, all `exit 0`. Then, in this order:

1. Applied `m10/AMENDMENT_STAGED_2026-09-08.md` to the registry and the mandate.
2. Re-issued `results/m10_F_verdict.json` under the new registry hash (`run_arm.f_verdict` pins it).
3. Wrote `m10src/contrasts.py` and computed the contrasts. **Family A first**, then the rest.

## The verdicts (MDE 0.0056; one-sided lower bound at alpha/12; F1 two-sided at alpha/24 per tail)

| id | a − b | point | lower | label |
|---|---|---:|---:|---|
| F1 | F-bge-small − F-MiniLM-L6 @20M | +0.011595 | +0.007221 | RESOLVED |
| A3-A2 | A3 − A2 | +0.010068 | +0.005363 | POSITIVE, NOT RESOLVED |
| A4-A3 | ANCHOR(A4) − A3 | +0.012080 | +0.006909 | **RESOLVED** |
| G1 | ANCHOR(1152) − G-384 | +0.021651 | +0.015273 | RESOLVED |
| G2 | G-1536 − ANCHOR | +0.000335 | −0.002787 | NOT RESOLVED |
| G3 | G-MLP − ANCHOR | +0.002415 | −0.000307 | NOT RESOLVED |
| B1 | B-100/0 − ANCHOR | −0.001057 | −0.004704 | NOT RESOLVED |
| B2 | B-50/50 − ANCHOR | +0.002690 | −0.000537 | NOT RESOLVED |
| D1 | D-NORM − ANCHOR | −0.000202 | −0.001859 | NOT RESOLVED |
| D2 | D-COV − ANCHOR | −0.018391 | −0.023786 | NOT RESOLVED |
| E1 | E-bs32 − E-bs128 | — | — | NOT COMPUTED (E-bs128 is CLOUD_ONLY, never run) |
| C1 | C-M9init − C-fresh | — | — | NOT COMPUTED (arm CUT pre-observation) |

Family A's two contrasts use the corrected bar `lower > MDE` and are exempt from sign stability.
The other resolve on `point >= MDE AND lower > 0 AND sign stable across the last two cycle ends`.

Selected recipe: student **bge-small**, corpus **A4 (harvest + ~1.0M generated queries)**, head
**1152-wide linear**, mix **75/25**, objective **squared L2**, batch **PENDING**.

## What I believe, and want attacked

1. **The amendment was legitimately pre-observation.** No contrast had been computed and no
   bootstrap run when it was applied. But the arms' per-cycle COV *macros* were already on disk and
   visible, and I say so in the registry rather than claiming otherwise. I claim none of the changes
   is keyed to an arm outcome. **Attack this**: find a change in the amendment whose value to the
   project depends on a number that was already visible.
2. **A4−A3 RESOLVED means the ~1.0M generated queries enter the build.** That is the registered
   rule (`rules.generated_half_in_build`). But the registry itself now says every family-A contrast
   is an **exposure re-allocation** — the sampler is form-balanced, so A4's 12 forms get 312,500
   presentations each against A3's 5 forms at 750,000, and 58.3% of A4's exposure is generated text.
   **Attack this**: is "generated queries help" a defensible reading of A4−A3 at all, or has the
   screen measured *form breadth* and mislabelled it? Does the build follow from it either way?
3. **A3−A2 at lower = 0.005363 against MDE 0.0056 is POSITIVE, NOT RESOLVED.** It missed by
   0.00024. I have not moved the bar and will not. **Attack this**: is there any place in today's
   work where a threshold or a rule was chosen after a number was visible?
4. **D2 (−0.018391) is NOT evidence against document-aware regression.** D-COV's loss is ~612×
   smaller and its gradients ~500× smaller on identical batches
   (`results/m10_dcov_gradient_audit.json`: grad norms 0.0017 vs 0.868), at the same peak LR. The
   registry says epsilon is not the confound (step fractions 0.99999 both) but gradient **clipping**
   is. **Attack this**: is my confound story right? Is there a reading where D-COV's arm *does*
   test its own hypothesis, or where the confound cuts the other way?
5. **E stays PENDING and does not fall through `rules.E_cost`.** That rule reads "select bs32 iff
   E1 RESOLVES; in every other case select bs128", and an unread contrast trivially does not
   resolve — which would hand E's decision to a measurement nobody made. I refuse that reading.
   **Attack this**: am I inventing a disposition the registry does not have? Or is reading
   E_cost literally the actual protocol violation?
6. **`m10src/contrasts.py` reads the right thing.** It reproduces the pre-existing F1 record
   bit-for-bit including the draws digest `cb392db4…`. **Attack the code, not the claim**: read
   points, alias resolution, the sign-stability implementation, the per-contrast quantile check,
   the paired alignment, the family-rule selection in `selection()`. Find a case where it silently
   reads the wrong checkpoint, the wrong quantile, or the wrong arm.
7. **Nothing here touched a bar, partition, comparator, frozen artifact or statistic.**
   **Verify it** against `git show HEAD`.

## How to answer

For each objection: the exact file and line, what is wrong, what it would change about the build or
the report, and what would settle it. Rank by whether it can change the selected recipe. If you
think one of my seven claims is simply correct, say so in one line and move on — spend your effort
on the ones that are not.
