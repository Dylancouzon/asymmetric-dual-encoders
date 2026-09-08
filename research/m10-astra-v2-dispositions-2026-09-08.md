# gpt-6-astra pass 2 (grounded) — verified findings and dispositions, 2026-09-08

Brief `research/m10-astra-brief-v2-2026-09-08.md`; verbatim `research/m10-astra-v2-whole-plan-2026-09-08.md`.
**147,746 tokens** (pass 1 was 6,314 and read NO files). It read all six named files with `nl`/`sed`
only — read-exclusion audited clean, no reserved path touched, no repo-wide grep. It also read
`instructions-m14.md`, the calibration report and some `m10src` code, and said so.

**THE ENABLING FACT: every decision-logic defect below governs a decision NOT YET MADE.** No
confirmation arm has run, D and E have not run, and the top-up governs the build. So all of them are
still fixable **pre-observation**, which is the only window the protocol allows.

**THE BLOCKING FACT: `results/m10_F_verdict.json` pins `registry_sha256`, and `run_arm.f_verdict`
refuses any post-F arm if the registry hash changes.** Editing `m10/screen_registry.json` right now
HALTS the running chain (8 arms left). Sequence: let the chain finish → amend → re-issue the verdict.

---

## P0 — three decision rules that do not implement their stated intent. All VERIFIED verbatim.

### 1. The confirmation rule can reject perfect replication and accept a reversal
`confirmation.stands_iff` = *"the winner's margin exceeds the largest seed range observed in either
arm"*, where `margin` is the seed-0 difference in the ORIGINAL screen and `seed_range` is
*"max minus min of an arm's COV macro over its three seeds"* — i.e. variation in **absolute
scores**, when the decision depends on variation in **paired treatment effects**.

| case | winner seeds 0/1/2 | default seeds 0/1/2 | paired diffs | rule says |
|---|---|---|---|---|
| perfect replication + a shared seed shift | .520 / .480 / .480 | .500 / .460 / .460 | +.020, +.020, +.020 | **REJECT** (.020 ≯ .040) |
| reversal on both fresh seeds | .520 / .506 / .506 | .500 / .514 / .514 | +.020, −.008, −.008 | **ACCEPT** (.020 > .014) |

It also keeps the *selected* seed-0 margin as the numerator instead of asking whether the fresh
seeds reproduce it. **Fix:** test the paired difference per seed — e.g. all three paired diffs same
sign and the mean of seeds 1–2 retaining the effect — not absolute-score ranges.

### 2. Family E's cost decision cannot survive its own confirmation
`E_cost` selects **bs128** whenever E1 is unresolved (an explicit, deliberate "accept a small
unresolved quality disadvantage to save cost"). But E is in `confirmation_eligibility`, and
confirmation computes margin = winner − default. With bs32 ahead by .003 unresolved, that margin is
**−.003**, which cannot exceed a non-negative seed range, so confirmation fails and
`unconfirmed_non_defaults` **reverts to default**. The cost branch is unreachable through the full
procedure. **A quality-superiority confirmation rule cannot implement a cost-based choice.**
**Fix:** exempt E from quality confirmation, or give it a cost-aware confirmation of its own.

### 3. `D_tie` discards a demonstrated improvement
*"if D-NORM and D-COV both resolve with margins within 1e-4, the default (squared L2) stands."* If
both alternatives resolve at, say, +.015, their agreement is a reason to pick one by a deterministic
preference — not to return to the objective both beat. `multi_arm_winner` (highest resolving point
estimate) is the coherent rule; `D_tie` overrides it in exactly the case where the evidence is
strongest. Same shape in the generic default-on-tie path.

---

## P0 — documentation defect: the Bonferroni denominator is 12, and our files still say 13
`instructions-m10.md:160` (amendment C2, 2026-09-05) is operative: **denominator 12** — 11
one-sided at α/12 plus F1 two-sided at α/24 per tail = α **exactly**; `rules.F_orientation` states
the same identity. But `instructions-m10.md:77`, `:337`, `:641` and ≥4 places in `m10/LEDGER.md`
still say **0.025/13**, which does not close.

**No number is wrong:** F1 was computed at quantile 0.025/24, the /12-consistent value, so the
verdict stands. The stale /13 is a trap for the next session, and astra explicitly does NOT
recommend moving the denominator after observation. **Fix by pointer, not by rewriting history.**

---

## P1 — the biggest structural error: screen effects are assumed to transfer to the build
The screen asks *"which isolated change improves a 5M three-cycle run on COV around the
A4/bs32/1152/squared-L2 anchor?"*. The release asks *"does a 200M combined recipe beat bge-small on
clean-4?"*. Between them the plan changes dose **5M → 200M**, data **A4 → A3**, batch **32 → 128**,
several components at once, and the evaluation distribution. **This is beyond W14's surface
mismatch:** even a perfect COV↔clean-4 proxy would leave effects measured on a recipe the build
never uses. The synthesized 5M arm and the LoTTE veto guard combination and gross harm, not the
long-dose extrapolation.

## P1 — contrasts that need narrower interpretations (all verified against the registry)
- **Family A measures training-DISTRIBUTION changes, not "forms" or "generation".** Every arm is
  form-balanced, so at 3.75M query presentations an existing form gets **~750,000 in A3 vs 312,500
  in A4**; generated forms take **58.3%** of A4's presentations. So A4−A3 tests *replacing* existing-
  form exposure with seven generated forms. **Drop "measured null"** (`registry:377`) — the valid
  reading is "this allocation did not establish improvement under the registered test".
- **A3 already contains PAQ, which the mandate itself calls machine-generated.** So A4−A3 is
  *incremental Qwen generation vs a corpus that already contains synthetic questions*, not
  synthetic-vs-real.
- **Family B's screen and build are different interventions.** The screen holds query presentations
  at 3.75M and adds documents; at a fixed 200M build, adopting 50/50 *removes 50M query
  presentations*. The screen never tested that trade.
- **E1 measures a whole optimizer policy, not batch size.** `WARMUP_STEPS = 2000` is fixed in
  **steps**: 64,000 examples (1.28% of the screen) for bs32 vs **256,000 (5.12%)** for bs128, and
  bs128 takes ~¼ the AdamW updates. Verified in `m10src/nano10.py:320`.
- **G-384 may challenge the diagnosis but may not correct the recipe** (`registry:99`, `:193`): even
  a large reproducible win cannot select it. Either let it win on quality+cost, or stop saying the
  experiment "decides" width.
- **G-1536 cannot raise output rank.** The teacher is 1024-d and the 1152 anchor already permits
  full output rank, so G-1536 tests extra-layer features and parameterization — not the rank
  bottleneck the diagnosis names.
- **The head diagnosis' PCA argument does not hold for the implemented loss.** The student
  normalizes and the loss compares unit vectors, so the relevant quantity is E‖P_S t‖, while
  ordinary reconstruction maximizes E‖P_S t‖². `PLANNING.md:314` conflates them. The PCA result
  still shows compressed teacher representations lose quality; it does not show normalized-L2
  training is driven toward them.
- **"The surface cannot resolve an MDE-sized contrast" is too broad.** It rests on the
  **unrelated-model** distance 0.008619; same-init calibration measured 0.002877, and F1 measured
  **0.004374** today. Width depends on the actual paired differences.

## P1 — `FORMS-12` is labelled "descriptive only" but drives training
`registry:318` says descriptive; `instructions-m10.md:552` has the plateau top-up **doubling the
presentation weights of the bottom two FORMS-12 forms**, gated on a dev→six forecast that
`instructions-m10.md:750` calls *"forecasting only, never gating"*. A prewritten branch does not
make the influence disappear. Astra would **remove the top-up**: "worst current score" and "best
next use of compute" are different quantities. Cost if removed: **66.7M examples, ~29.7 h**.

## P2 — cuts it would make (~7.8 h of arms, plus ~20 h confirmations, plus the 29.7 h top-up)
Retire A1/A2−A1 repetitions · cut B-50/50 from prospective selection · cut G-1536 before G-384/G-MLP
· defer D-NORM · **replace the confirmation block's rule and redirect its ≤45M examples (~20 h) to
build-regime validation**. Keep G-384, G-MLP, D-COV, B-100/0, the A3/A4 comparison, the batch
comparison. Completed arms are sunk cost and their records must not be reinterpreted retroactively.

## P2 — additions, ranked by information per cost
1. **Executable decision examples** for the three counterexamples incl. E's full confirmation path —
   1–2 engineering hours, no GPU. *This is the one that stops a broken rule shipping.*
2. **Realized-exposure table per arm** (source × form presentations, distinct texts visited,
   repeats, token lengths) — CPU only. Corrects family A's interpretation.
3. **Deployment-quality bridge:** exact fp search vs the intended compressed representation vs the
   deployed ANN config, for zero/nano/bge-small. Guards against plotting full-precision quality
   against compressed-index latency — M9 found the index viable only under binary quantization.
4. **Two mechanism audits:** fit a rank-384 head on the *same 1152-d features*; measure D-COV vs L2
   gradient/update scale on identical batches (unit-trace normalization makes L_COV = ‖s−t‖²/1024 in
   the isotropic case, so a negative D-COV could be an optimizer-scale artifact). ≤1 GPU-hour.
5. **SYNTH-20M vs the anchor at 20M**, same full schedule — tests whether combined gains survive a
   longer dose. ~8.9 h if F-bge-small is a reusable comparator. Fund from the confirmation budget.
6. *conditional* **A4-G20**: A4's corpus with generated forms at 20% total exposure — separates
   "generated text does not help" from "58% exposure was the wrong allocation". ~2.2 h.
7. **Oracle escalation curve** for the PAIR: using existing per-query zero/nano scores, the best
   achievable quality if only a fraction of queries may use nano, vs random and vs either alone. An
   optimistic bound, not a router. **If even the oracle is weak, never build a router.**

## P2 — what M14 should stop claiming
- *"M9 was a coverage failure, not a capacity failure"* — 94% on one distribution does not establish
  joint capacity across many.
- *"Normalized L2 cannot push a 384-wide head past 90–93%"* — the PCA argument does not establish it.
- *"A3−A2 establishes query-form coverage" / "A4−A3 measures whether synthetic data helps"* — both
  are bundled distribution interventions.
- *"Unresolved means a measured null / a tie"* — needs effect estimates and intervals.
- *"Clean-4 is contamination-free"* — its property is no *disclosed* teacher overlap.
- **"Fused systems are nearly contamination-immune"** (`instructions-m14.md:38`) — **unsupported by
  its own arithmetic.** S̄₆ − S̄₄ = ⅓(S̄_excluded2 − S̄₄) is a difference between two dataset groups,
  not an identification of a contamination effect. Report **partition sensitivity**, not immunity.
- Credit the frozen-teacher lookup-table construction as **prior art** (its own pyNIFE instruction
  requires it); the contribution is the evidence, analysis and deployable implementation.
- The Spearman result's strongest form is **an empirical counterexample to selecting by teacher
  quality**, not a universal law — and M14 quotes Spearman over **eight** and **seven** comparable
  rows while saying eleven teachers were measured. Keep the denominators distinct.

## Where pass 1's five findings stand
All five survive; astra adds that #3 (missing cost measurements) is **necessary but insufficient** —
"even perfectly measured costs are insufficient if the quality coordinate belongs to another
retrieval configuration". That is addition 3 above.

## Its own closing recommendation
*"Repair the decision rules, preserve the running screen's record, and spend the next substantial
compute allocation on validating the assembled recipe at a more relevant duration."*
