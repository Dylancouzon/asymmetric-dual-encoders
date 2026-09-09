# STAGED AMENDMENT — apply when the W8 band-1 chain finishes, before any confirmation arm

**Authority.** Dylan, 2026-09-08: *"My knowledge of the small details is pretty limited... Take
whatever decision is best and will guarantee us the best model."* Explicit delegation of the calls
below, recorded here because several are Tier-3-shaped. Everything here changes only decisions
**not yet made**, which is the only window the protocol allows. **No bar, partition, comparator or
statistic is touched.** Evidence: `research/m10-astra-v2-dispositions-2026-09-08.md`, every claim
verified against the registry.

**Why it is STAGED and not applied.** `results/m10_F_verdict.json` pins `registry_sha256` and
`run_arm.f_verdict` refuses every post-F arm if the registry hash changes. Applying this mid-chain
halts the screen. **Apply after `rest_chain.log` reads COMPLETE, then re-issue the F verdict.**

---

## 1. Fix the three decision rules (`m10src/decision_rules.py` holds both versions, 8 tests)

| rule | from | to |
|---|---|---|
| `confirmation.stands_iff` | seed-0 margin > range of **absolute** scores | **every paired difference positive AND the fresh seeds retain ≥50% of the original effect** (`confirmation_stands_corrected`) |
| `rules.E_cost` + `confirmation_eligibility` | E is quality-confirmable, which reverts its cost pick every time | **E is EXEMPT from quality confirmation**; the cost rule stands alone (`e_after_confirmation_corrected`) |
| `rules.D_tie` | both resolve within 1e-4 → revert to squared L2 | **keep a winner**, deterministic tie-break to `leaf_norm_e2`, report the tie (`d_selection_corrected`) |

Rationale, all three: the current rules can select a *worse* recipe than doing nothing —
confirmation can accept an effect that reversed on both fresh seeds. That is a direct model-quality
risk, and it costs nothing to fix.

## 2. Cut the confirmation block; redirect its budget to validating the ASSEMBLED recipe

**Cut:** the two-decision confirmation block (≤45,000,000 examples, ~20 h).
**Add:** `SYNTH-20M` — the assembled selected recipe, trained at **20,000,000 examples on its own
full 3-cycle schedule** (never a continuation of an annealed 5M candidate), against the anchor
recipe at the same dose, read on existing development surfaces. ~8.9 h.

Rationale: confirmation re-tests *isolated* effects at the *screen's* dose. The build combines
several components at 200M. astra's central finding is that this transfer is assumed, not
measured — *"the screen can execute perfectly and answer a question nobody needs answered."*
SYNTH-20M measures the thing the build actually is. **Descriptive and development-only**: it
selects nothing by itself and enters no confirmatory accounting, because COV is already observed.

## 3. Remove the FORMS-12 plateau top-up (`instructions-m10.md:552`)

**Cut** the rule that doubles the presentation weights of the bottom two FORMS-12 forms
(66,700,000 examples, ~29.7 h).

Rationale, two independent reasons. (i) It is **adaptive training driven by a validation surface
the registry calls "descriptive only"** (`registry:318`), gated on a dev→six forecast the mandate
calls *"forecasting only, never gating"* (`instructions-m10.md:750`). A prewritten branch does not
remove the influence. (ii) Even setting that aside, *"worst current score"* and *"best next use of
compute"* are different quantities — the bottom two forms by teacher agreement are not necessarily
where more training buys the most C1b.

## 4. Where the freed budget goes

| | examples | at 624 ex/s |
|---|---:|---:|
| freed: confirmation block | 45,000,000 | 20.0 h |
| freed: FORMS-12 top-up | 66,700,000 | 29.7 h |
| **freed total** | **111,700,000** | **49.7 h** |
| spent: SYNTH-20M validation | −20,000,000 | −8.9 h |
| **to DOSE on the build** | **91,700,000** | **40.8 h** |

Dose is the lever M9's own failure points at — it stopped at 3.74B tokens on a plateau — and it is
the one knob with no selection risk attached. Subject to the $1,000 cloud ceiling
(~$90–245 committed); price the increment before committing it.

## 5. Interpretation fixes that carry no compute (apply with the above)

**Timing, on the record, because §5 relabels a family-A outcome:** this amendment was committed
**20:13:07** and the exposure table **20:14:53**; **A3's record — the last family-A arm — was
written 20:35:40**, 21 minutes later. So no A contrast could be computed, and no A number existed,
when the relabel was written. The relabel changes no threshold and no action either way; the
timing is recorded so it cannot be read as motivated by a result.

- **MEASURED, `results/m10_exposure_table.json`** (built 2026-09-08, `m10src/exposure_table.py`):
  per-form query exposure swings **6×** across family A —

  | arm | forms | presentations PER FORM | generated share of exposure |
  |---|---:|---:|---:|
  | A1 / A2 | 2 | **1,875,000** | 0% |
  | A3 | 5 | 750,000 | 0% |
  | A4 / ANCHOR | 12 | **312,500** | **58.3%** |
  | F-bge-small / F-MiniLM-L6 | 12 | 1,250,000 | 58.3% |

  The sampler is form-balanced, so every form present takes an equal share of presentations no
  matter how much text backs it; `by_form` in the manifest is AVAILABLE TEXTS and is not exposure.
  **So EVERY family-A contrast is an exposure re-allocation, not only A4−A3** — adding forms is
  inseparable from cutting per-form training across the whole family. (F's contrast is clean by
  comparison: both arms had identical per-form exposure.)
- **`registry:377` "measured null" → "did not establish improvement under the registered test."**
  Family A's arms are all form-balanced, so an existing form gets ~750,000 presentations in A3
  against 312,500 in A4: **A4−A3 tests re-allocating 58% of query presentations to seven generated
  forms**, not "does synthetic data help". A3 also already contains PAQ, which the mandate itself
  calls machine-generated.
- **Family B** is an augmentation experiment at fixed query exposure. At a fixed 200M build,
  adopting 50/50 *removes 50M query presentations* — a trade its contrast never tested. Limit what
  B's result decides.
- **E1** compares a whole optimizer policy: `WARMUP_STEPS` is fixed in **steps**, so bs128 gets
  256,000 warmup examples (5.12%) against bs32's 64,000 (1.28%), and ~¼ the AdamW updates.
- **G-1536** cannot raise output rank (teacher is 1024-d; the 1152 anchor already permits full
  rank). It tests extra-layer features and parameterization.
- **G-384** may not select the recipe (`registry:99`, `:193`), so stop saying the experiment
  "decides" width; it is evidence for or against a prescribed width.
- **Bonferroni denominator is 12** (amendment C2; `rules.F_orientation` closes at α exactly).
  Stale `/13` mentions annotated 2026-09-08. F1 was computed at α/24, so no number changes.

## 6. Deliberately NOT done

- **No new COV admission, no MDE change, no re-weighting.** COV is observed; those are forbidden.
- **No change to C1/C1b, the bars, clean-4, the six, the reserved four, or any frozen comparator.**
- **Completed arms keep their records and their registered interpretation.** Their cost is sunk;
  nothing here reinterprets a measurement already taken.
- **A4-G20 and the arm-level cuts (B-50/50, G-1536, D-NORM) are NOT taken.** Those arms are cheap
  (~2.2 h each), already smoked, and their contrasts are registered; cutting them mid-chain buys
  ~7.8 h and loses registered evidence. Not worth it.
