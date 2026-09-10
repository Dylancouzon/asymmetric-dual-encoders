# M10.2 — THE RECIPE LOCK (half A). Ready to push.

**THE LOCK IS SPLIT IN TWO (Dylan, 2026-09-10).** The mandate (`instructions-m10.md`:346) asks for
ONE pushed commit carrying the recipe, the final-run registry and the LoTTE manifest. That bundling
made the cloud spend wait on decision code for a step three weeks away, so it is split — **an
explicit, dated deviation from the mandate, on Dylan's ruling, not a quiet one.**

| half | what it decides | status |
|---|---|---|
| **A — THIS FILE** | which model to build, on what data, for how long, at what cost | **READY** — stable for two days, and the only half the cloud spend needs |
| **B — `m10/M10_4_DECISION_LOCK.md`** | how the finished model is judged on the six datasets | **DEFERRED to M10.4**, before the final run |

**What the split does NOT change.** The protocol requires a decision to be fixed BEFORE the numbers
it affects exist. Half B is still locked and reviewed clean before a single six-dataset number is
computed — the split changes *when it is written*, not *whether it precedes the result*.

**The hard gate that makes that true:** no six-dataset evaluation, no `m10-six-spent` tag and no
LoTTE read may happen until half B is locked and a review returns clean. Half B is NOT nearly done;
its open findings are listed in its own file.

**Why half B was deferred rather than finished.** The scoring path that will CALL its decision code
does not exist (`m9src/final9.py`:348 raises `SCORING PATH NOT IMPLEMENTED`). Nine adversarial
rounds kept finding gaps in code whose only caller is unwritten — reviewing it now is premature in
a way care does not fix. It gets written during the build, beside the executor that uses it.

**Nothing in half A may be edited after its push** except by a dated amendment governing a decision
not yet made.

Every field below is read from `m10/screen_registry.json` and the committed artifacts; where this
file states a number, the artifact is authoritative.

## What the screen selected

| axis | selected | on what |
|---|---|---|
| student | **bge-small-en-v1.5** (34,540,672 params with the head) | F1 RESOLVED, +0.011595 [+0.007221] |
| corpus | **A4** — m9-pool + PAQ-build 1.0M + harvest 1.25M + **834,463 generated** | A4−A3 RESOLVED, +0.012080 [+0.006909] |
| head | **1152-wide linear**, 3 pooled layers {12, 8, 4} | G2/G3 NOT RESOLVED → the registered default; G1 (+0.021651) is evidence on M9's 384-wide diagnosis and selects nothing |
| mix | **75/25** query/document over a 4-step window | B1/B2 NOT RESOLVED → default |
| objective | **squared L2** | D1/D2 NOT RESOLVED → default. **D2's mechanism is unresolved, not a clean loss** — `_interpretation.D1_D2_precondition` |
| batch | **PENDING — `rules.E_cost`, resolved on the A100 before the build starts** | E1 NOT COMPUTED: `E-bs128` is CLOUD_ONLY |
| warm start | closed-form ridge head, n_fit 60,000, fit seed 21, λ from the locked grid | `warm_start.all_arms` |
| init | bge-small (family C cut under W8 band 1) | — |
| seed | 0 | `seed_rule` |

**The screen selected no non-default component.** That is the honest headline: the anchor recipe
survived every alternative offered to it, and the two contrasts that resolved (F1, A4−A3) confirmed
choices the anchor already embodied. `_interpretation.what_the_screen_could_not_tell_us` records
what that does and does not license.

## The one field still open, and why it is not a hole

`batch` is the only unfilled axis. `E-bs128` reproducibly faults on the box at realistic sequence
lengths, so **both** E arms run on the rented A100 together — both, not just bs128, because a
box-trained bs32 against an A100-trained bs128 would put a hardware difference inside the one
contrast that decides a build parameter.

> **CORRECTED 2026-09-10**, in both `rules.E_warmup_parity` and `_interpretation.E1` — the first
> was fixed once `A3-20M` finished (its resume path pinned the registry hash) and the second was
> missed, leaving the correction half-applied for a day until round 7 caught it. Both now say the
> same thing: **both E arms run on the A100 together**, which is what removes the hardware
> difference E1 would otherwise carry.

- **The rule is fixed and pre-registered:** `rules.E_cost` — select bs32 iff E1 (`bs32 − bs128`)
  RESOLVES; in every other case bs128. E is exempt from quality confirmation (amended 2026-09-09).
- **The parity fix is in the code, not the prose:** `rules.E_warmup_parity` — warmup is registered
  in EXAMPLES (64,000) and `nano10.warmup_steps_for(batch)` is what the trainer and the run
  fingerprint read, so bs128 gets 500 warmup steps and bs32 gets 2,000. Without it bs128 would have
  had 4× the warmup exposure and ¼ the updates, handicapping the arm whose selection saves 2.2× of
  the build's cost.
- **Disclosed, not removed:** bs128 still takes ~¼ the AdamW updates at the same peak LR. Read E1
  narrowly — it compares an optimizer policy, not batch size alone.
- **The build does not start until E1 is read.** Its two arms are a mandatory budget line below.

## Dose, schedule, optimizer

- **Dose: 200,000,000 examples**, three cycles of 66,700,000. At 75/25 that is ≈ **16.8B tokens**
  (150M × ~35 + 50M × ~230); query epochs ≈ 37 over 4.0M texts, document epochs ≈ 8 over the 6.15M
  pool.
- **Schedule:** 3 cycles of equal example count, each linear 1e-4 → 1e-5; warmup **64,000 examples**
  in cycle 1. Evaluation at every cycle end (annealed) and at cycle midpoints (curve watch only).
- **Optimizer:** AdamW β=(0.9, 0.999), eps 1e-8, weight decay 0.01 on dim > 1, gradient clip 1.0.
- **Batching:** length-bucketed single-chunk. This is **part of the build, not an optimisation** —
  M9's two-chunk collate costs ≈139 h against ≈81 h for the same work (amendment A7), which is why
  the mix is a share over a 4-step window rather than per batch.
- **`torch.compile` is SMOKE-ONLY** for registered arms unless Dylan rules otherwise (~1.7×).

## Plateau, extension and kill — read on annealed checkpoints only

Let `m_k` be the COV macro at the end of cycle `k`, full precision, by
`m10src/cov_macro.macro` on the locked surface (`cov_macro.SURFACE`, four families at equal weight,
units equal within family).

- **Extension:** after every cycle `k ≥ 3`, one further cycle of 66,700,000 examples
  (linear 1e-4 → 1e-5, as cycle 3) starts **iff** `m_k − max(m₁ … m_{k−1}) ≥ 0.003` **and** the
  extension cycles already run are fewer than `max_extension_cycles`. Whole cycles only.
- **Plateau:** fires when that same improvement test **fails** at a cycle end `k ≥ 3`, independent
  of the cycle cap.
- **Kill:** non-finite loss or gradient; or two consecutive scheduled evaluations more than 0.0056
  below the best evaluation **of their own kind** — midpoint against midpoints, cycle end against
  cycle ends — so the rule can fire inside the build and not only at its end.
- **The FORMS-12 plateau top-up is REMOVED** (2026-09-09). It was adaptive training driven by a
  surface the registry calls descriptive only, gated on a forecast the mandate calls "forecasting
  only, never gating".

## GPU-hour allocation under the $1,000 ceiling

Mandatory lines first at the measured rates and the **billed** price, then whole extension cycles
from the remainder. A cycle whose projected cost plus billed spend to date would exceed the ceiling
does not start.

| line | GPU-hours | $ at 1.5–2.5/h |
|---|---:|---:|
| day-one rate benchmark | 1 | 2–3 |
| **family E, both arms, 2 × 5M examples** — new mandatory line; the mandate's table predates the ruling that both E arms run on the A100 | 3–6 | 5–15 |
| build, 200M examples (bs128 → bs32) | 37–81 | 55–205 |
| cloud-side encodes + export, parity, final run | 6 | 9–15 |
| **LoTTE-clean encode for read #1** — ~2.8M passages with stella, mandate :748; was missing from this table | 1.3 | 2–4 |
| persistent disk, egress | — | ≈ 25 |
| **mandatory total** | **48–95** | **≈ $98–267** |
| each extension cycle, 66.7M examples | 13–25 | 20–63 |

**`max_extension_cycles` is fixed by this formula at the day-one benchmark, not now**, because the
billed price is not known until the instance is rented:

    max_extension_cycles = floor( (1000 − sum(mandatory lines at the billed price)) / (cost of one 66.7M cycle at the measured rate and billed price) )

Planning value at the assumed $1.5–2.5/h: **11–36**. The money is not the binding constraint; the
plateau rule is, and that is the intended design.

**The freed budget from the 2026-09-09 amendment is a RANGE, not 91.7M examples**
(`confirmation.budget_released`): between **−5M and +61.7M**. Zero confirmable decisions existed, so
cutting the confirmation block freed no quality-confirmation examples; the FORMS-12 top-up's 66.7M
was conditional on a branch that may never have fired. Any dose above 200M is an extension cycle
under the rule above, never a silent increase.

## Provenance

- **Data:** `results/m10_data_manifest.json` — 4.57 GB hashed on disk including the raw harvest
  pools, 12 cited measurements, and it refuses to emit with a hole. Cite this file's sha256; it
  inherits the rest.
- **Registry:** `m10/screen_registry.json`. Every contrast record and the F verdict pin its sha256.

  **The freeze rule, restated honestly 2026-09-10 (Codex round 8).** I wrote "the registry is
  frozen the moment a contrast is computed against it" and then edited it repeatedly — prose
  corrections, interpretation entries, the `E_warmup_parity` fix — recomputing all twelve contrasts
  and re-stamping their `registry_sha256` each time. The statistics never moved; the provenance
  string did. Written as an absolute, the rule was one I was breaking, and a rule nobody keeps is
  worse than a narrower one that holds. So:

  - **DECISION-BEARING fields are frozen** once any contrast is computed: `contrasts`, `arms`,
    `statistics`, `anchor`, `data_cut`, `order`, `rules`, `outcome_to_action`, `anchor_aliases`.
    A change to any of these invalidates the screen and is not a re-stamp, it is a re-run.
  - **PROSE fields may be corrected** — `_interpretation`, `_what`, `_amended*`, and the narrative
    halves of `rules.*` — because a stale or contradictory explanation beside a correct number is
    itself a defect, and three of this session's reviews found exactly that. Every such correction
    is dated in the field it touches, the contrasts are recomputed (they are deterministic, so the
    numbers reproduce bit-for-bit), and the re-stamp is recorded below.
  - **Re-stamps so far:** six, all prose-only, all with every `delta_raw`, `lower_bound_raw` and
    `draws_sha256` unchanged. That reproduction IS the evidence the edits were non-decisional; the
    audit is `git log -p m10/screen_registry.json`.
- **Verdicts:** `results/m10_screen_verdicts.json`, `results/m10_contrast_*.json`.
- **Teacher:** `NovaSearch/stella_en_400M_v5`, frozen, 1024d. Discloses ArguAna and FiQA (2 of the
  six) and FEVER (1 of the reserved four) in its training data; every stella-based claim carries
  that qualification.

## What the report must carry beside the build

Registered before the numbers exist, so they cannot be dropped after them:

1. **Every interval is a query-resampling interval and excludes training-seed variability.**
   `ANCHOR-seed1` gives one anchor sensitivity observation — n=1, not a bound, and it licenses
   `arms.ANCHOR-seed1._what_it_licenses` and nothing beyond it.
2. **A4−A3's gain is concentrated in one family** — MedicalQA carries 81.8% of it, BRIGHT is net
   negative, and the three-family point without consumer-health is +0.0029, half the MDE. The
   defensible sentence is registered in `_interpretation.A4_A3_is_a_FORM_MATCH_effect`. Every
   resolved contrast is reported with a leave-one-family-out row.
3. **Repetition at build dose:** each of the 12 forms takes 12.5M presentations, which is 201.4 per
   available `health` text against 9.4 per `factoid` text — a 21× spread, landing on the family that
   carries the one resolved corpus contrast. "40× dose" is not "40× coverage".
4. **Recovery is not release.** Passing clean-4 and *repairing* M9's two weak CQADupStack
   components are different claims; the report states what counts as recovery per component, using
   teacher-normalized retention and absolute quality together. Not a retroactive veto.
5. **D2 is not evidence against document-aware regression** — its mechanism is unresolved.
6. **nano buys quality at equal edge cost sharing stella's index, NOT near-zero query compute.**
   That is `zero`'s claim alone; nano is 34.5M params against bge-small's ~33.4M with the same
   backbone forward pass.

## The final run and LoTTE — MOVED TO HALF B

`m10/final_run_registry.json`, `m10/LOTTE_LOCK.md` and `m10src/final10.py` are half B's artifacts.
They are written and heavily reviewed but **NOT locked**, and their open findings are recorded in
`m10/M10_4_DECISION_LOCK.md`. **Half A makes no claim about them.**

## Still to do before THIS half is pushed

- [x] every recipe field filled from the registry and the committed artifacts
- [x] both descriptive reads folded in (`ANCHOR-seed1` +0.000712; `A3-20M` +0.016453 at 20M)
- [x] `E-bs32` registered as a real A100 arm, so E1 is not a cross-hardware contrast
- [x] the GPU-hour table, including family E and the LoTTE-clean encode
- [ ] **Dylan's approval of the split, recorded here** — he ruled for it 2026-09-10; this line is
      the placeholder for the push commit that records it
