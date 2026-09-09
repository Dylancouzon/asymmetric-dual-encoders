# You are deciding, not advising. Two rulings, plus whatever we are not seeing.

The project owner (Dylan) has delegated **two live decisions to you** and asked for a review
alongside them. **Your ruling on the two decisions will be executed as written.** So do not hedge
into "it depends" — pick, and say what you are trading away. Where you genuinely need a fact we
have not given you, name the file and state your ruling conditional on it.

Two other reviewers have already been over this today (Fable on the decisions, Codex on the fixes).
**Do not re-audit their findings** — everything they raised is applied and archived. Your value is
the thing nobody on this project has considered at all: a wrong premise, a missing option, a
cheaper path to the same goal, or a better goal.

## READ-EXCLUSION (mandatory)

Never read: `results/frozen_eval/untouched-*`, any reserved qrels cache, `work/m9reserve`,
`results/perquery.json`. **No repo-wide or recursive grep/find.** These hold RESERVED confirmatory
data with one unspent access. Read at most: `m10/STATUS.md`, `m10/screen_registry.json`
(keys `confirmation`, `_interpretation`, `rules`, `statistics`, `arms`), `m10/RESULTS.md`
§M10.2, `results/m10_screen_verdicts.json`, `results/m10_calib_report.json`,
`results/m10_contrast_A4-A3.json`, `results/m10_exposure_table.json`. Name any other file you want
rather than opening it.

## Where the project is

Asymmetric dual encoders: a big frozen document encoder (stella_en_400M_v5, 1024d) in the cloud,
a near-zero-compute query path on the edge. Two query paths share ONE index — `zero` (a
token→vector lookup table, shipped, public) and `nano` (a ≤35M distilled transformer, this
milestone, M10). M9's nano attempt missed: 82.2% teacher retention overall but 93.8% on NQ against
50–71% on two CQADupStack components — a **coverage** failure, not a parameter-count one. M10 is
the retry: wider head from three pooled layers, warm start from M9, and query-form breadth from
~1.25M harvested real titles/headings/claims plus 834,463 generated queries for seven forms no
corpus contains.

M10's screen just finished: 13 arms on a local RTX 3080, 12 registered contrasts, MDE 0.0056,
one-sided lower bounds at α/12. The build itself runs on a rented A100 under a **$1,000 ceiling**
(~$90–245 committed so far). Release bars for nano, pre-registered: avg-6 **0.5155**
(leaf-ir-asym), clean-4 **0.5233**; the release floor is bge-small at 0.5042 / 0.5046.

## The screen's results

| id | contrast | point | lower | label |
|---|---|---:|---:|---|
| F1 | bge-small − MiniLM-L6 @20M | +0.011595 | +0.007221 | RESOLVED → student = bge-small |
| A4-A3 | full corpus − harvest-only | +0.012080 | +0.006909 | **RESOLVED** → generated queries enter the build |
| A3-A2 | harvest − PAQ | +0.010068 | +0.005363 | POSITIVE, NOT RESOLVED (bar is lower > 0.0056) |
| G1 | 1152-wide − 384-wide | +0.021651 | +0.015273 | RESOLVED (G-384 is a diagnostic; selects nothing) |
| G2 | 1536-wide − 1152 | +0.000335 | −0.002787 | NOT RESOLVED |
| G3 | MLP head − 1152 linear | +0.002415 | −0.000307 | NOT RESOLVED |
| B1 | 100/0 query/doc mix − 75/25 | −0.001057 | −0.004704 | NOT RESOLVED |
| B2 | 50/50 − 75/25 | +0.002690 | −0.000537 | NOT RESOLVED |
| D1 | LEAF ‖e‖₂ loss − squared L2 | −0.000202 | −0.001859 | NOT RESOLVED |
| D2 | document-covariance loss − squared L2 | −0.018391 | −0.023786 | NOT RESOLVED |
| E1 | batch 32 − batch 128 | — | — | **NOT COMPUTED** — the bs128 arm is cloud-only and has not run |
| C1 | warm-start variant | — | — | not computed, arm cut pre-observation |

**So the screen selected NO non-default component.** Selected recipe = bge-small · A4's corpus ·
1152-wide linear head · 75/25 mix · squared L2 · **batch PENDING**. That is the anchor recipe in
every field except batch.

Three facts you will need:

- **A4−A3's +0.012080 is 81.8% one evaluation unit** (MedicalQA +0.009885 of it). BRIGHT is net
  negative. Drop consumer-health and the three-family point is +0.0029, half the MDE. The
  intervention is bundled: A4 adds seven forms to A3's five (two of them `health` and `finance`,
  matching the two families that gained) *while diluting every existing form's exposure* — the
  sampler is form-balanced, so A4 gives each form 312,500 presentations against A3's 750,000.
- **No interval in this screen contains training-seed variance.** All bootstraps resample queries.
  A confirmation block (re-training winners at two fresh seeds) was CUT earlier today — correctly,
  since zero decisions turned out to be confirmable — but it was the only thing that would have
  measured seed noise. The anchor is the comparator in ten of the twelve contrasts. The only seed
  figure we have is **0.0013869** (one observation, `results/m10_calib_report.json`).
- **The box (RTX 3080) is idle and free.** ~624 examples/s; a 5M-example arm is ~2.2 h. Cloud time
  is the constrained resource, not box time.

## DECISION 1 — SYNTH-20M: re-point it, cut it, or defer it

Registered: train the **assembled selected recipe** at 20M examples on its own 3-cycle schedule
against the anchor at the same dose, read on development surfaces, selecting nothing. It replaced
the confirmation block on the rationale that *"the screen's isolated 5M effects are assumed to
transfer to a 200M combined build, and that transfer is assumed, not measured."*

The problem: **the screen selected no non-default component**, so the "assembled recipe" IS the
anchor, except for `batch`, which is PENDING. If E later selects bs32, SYNTH-20M compares the
anchor to itself. If E selects bs128, it is a batch-size read at 20M.

Candidates we have thought of: (a) cut it and bank the 20M; (b) re-point at **A4 vs A3 at 20M** —
the one contrast that resolved, and the one the build's corpus rests on, testing whether a 4×-dose
read agrees with the 5M one; (c) re-point at **anchor seeds 1/2** (see decision 2); (d) defer until
E resolves, which is free because E must run before the build regardless.

**Rule on it.** If you think all four options are wrong, say what to do instead.

## DECISION 2 — buy a training-seed number, or not

Proposal on the table: re-train **ANCHOR at seed 1**, 5M examples, ~2.2 h on the idle box, zero
cloud cost. Descriptive, selects nothing, re-decides nothing (COV is already observed and the
verdicts stand on their registered rules). It would give one paired seed-to-seed delta on the exact
arm that is the comparator in ten of twelve contrasts.

For: six of the ten computed contrasts sit within ±0.004 of their bar; a shift the size of the one
seed figure we have (0.0014) moves A4−A3's lower bound to almost exactly the MDE. Every interval in
the paper currently excludes seed variance and must say so.

Against: n=2 is still not a distribution; it cannot re-decide anything; and it may simply invite a
reader to ask for n=5.

**Rule on it.** If you rule yes, say what the number would license us to write, and what we must
NOT write with it. If you rule no, say what we write in the paper instead about seed variance.

## AND THE PART ONLY YOU DO

What is nobody here seeing? Some candidates for your attention, but do not feel bound by them:

- The screen cost ~52 box-hours and moved **nothing**: every alternative lost or failed to resolve.
  Is that a well-designed screen reporting an honest null, or a screen whose 5M dose and
  query-resampling intervals cannot see the effects it was built to find? What would you have run
  instead for the same hours?
- The decision surface (COV: BRIGHT, consumer-health, finance, legal) shares no dataset with the
  release bar (clean-4: scientific/biomedical). The build can be optimised away from the bar and
  nothing would notice until the final run.
- M9 failed on **coverage**, and the one thing that resolved in family A is concentrated in a
  single family that happens to match an added query form. Does the M10 recipe actually address
  M9's failure, or has it been measured on a surface that cannot tell?
- The cost axis is barely measured. `nano` has ~34.5M params against bge-small's ~33.4M and the
  same backbone forward pass — it buys quality at equal edge cost sharing stella's index, NOT
  near-zero query compute (that is `zero`'s claim alone).

## Output

1. **DECISION 1: <ruling>** — one paragraph of reasoning, and what it costs us.
2. **DECISION 2: <ruling>** — same, plus the sentence we may write.
3. **What we are not seeing** — ranked, most consequential first, with what would settle each.
