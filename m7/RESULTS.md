# M7 experiment results (append-only)

Every run lands here, including stopped, failed and OOM ones — that is the experiment ledger the
mandate requires. Written by `m7src/sweep.py`.

**Read the dev metric carefully.** The column is the *fast proxy macro* over three components
(nq-250k, cqadup-programmers, cqadup-physics) that in-training evaluation uses, so a sweep of ~25
configs stays affordable. Every **selection and gate** decision instead uses the full pinned dev
suite of six components (adding hotpotqa, heldout-train, heldout-longq) via `gate.py`. The two
are not comparable numbers: the teacher scores 0.5722 on the proxy and 0.6672 on the full suite.

Reference rows, full pinned suite (`work/devres/refs.json`): teacher (bge-base symmetric,
prefixed) **0.6672** · BM25 **0.4525** and potion-retrieval-32M **0.3801** on the four
text-backed components, which is where the G1 and G3 comparisons run — BM25 and potion have no
row on the held-out slices, whose corpora are pool row indices carrying no document text.

| run id | config | dev metric (proxy macro-3) | verdict |
|---|---|---|---|
| p1-objB | `work/runs/p1-objB.json` | 0.4548 | ok |
| p1-objB | `work/runs/p1-objB.json` | 0.4548 | ok — objective B (distillation, 8k steps). Matches the closed-form flat bound (0.4542) to +0.0006, so learned per-token weights + the KL ranking term buy nothing over flat MSE distillation. Confirms no optimisation pathology. Coverage 27,314/30,522 rows (89.5%), median 262 updates/row. |
| p1-objA | `work/runs/p1-objA.json` | **0.3248** (monotone decline) | ok but the objective is broken, not weak. Declines at every eval: 0.3532 / 0.3528 / 0.3437 / 0.3366 / 0.3292 / 0.3248. Cause: with `hard_neg_k=0` all 32,768 negatives per step are random draws from a 6.17M pool, and against a frozen document space those are trivially separable — so InfoNCE drives its loss down while carrying almost no fine-grained ranking signal, and the table drifts. (`reg_init` was the first suspect but does not fit: a pull toward the init weakens as update counts grow, which would flatten the curve, not bend it down.) Consequence: **the negatives ablation is load-bearing, not tuning.** Frozen doc vectors make huge negative pools nearly free, but cheap-and-plentiful is worth less than few-and-hard here. Coverage 26,798/30,522 (87.8%), median 380 updates/row. |
| p1-objA | `work/runs/p1-objA.json` | 0.3248 | ok |
| p1-objC | `work/runs/p1-objC.json` | 0.3721 | ok |
| p1-objC | `work/runs/p1-objC.json` | **0.3721** | ok — B(4k) reached 0.4449, reproducing p1-objB step-for-step, then the 8k contrastive phase degraded it monotonically to 0.3721 (−7.3 points): 0.4105 / 0.4011 / 0.3838 / 0.3721. Confirms the contrastive objective is broken independent of initialisation, and exonerates `reg_init` (weakest at the high update counts where this run degraded fastest). Coverage 27,312/30,522 (89.5%). |

### Corrections (append-only; the rows above are left as written)

- **p1-objA's "Cause:" above is REFUTED.** "Random negatives trivially separable" predicts stasis
  (near-zero loss, near-zero gradient), not monotone decline — and the log shows loss falling
  0.51 -> 0.13 over 12k steps *while* dev fell at every eval, i.e. the objective was being
  successfully optimised and what it optimised is anti-correlated with retrieval. Commit 6e01793
  called the mechanism "confirmed"; it was asserted, never tested. Leading suspects now, in order:
  (i) `fn_margin=0.02` deleting the hardest negatives in a compressed score space, leaving a
  gradient that is mostly "pull every query toward its positive" = a pull toward the corpus mean
  direction; (ii) tau=0.02 concentrating softmax mass on the anisotropy tail; (iii) Adam at
  lr 3e-3 taking full-size steps on that weak signal. `m7src/diag_scores.py` measures all three.
  Note the post-mask negative count was never logged, so suspect (i) ran unobserved all grid.
- **p1-objB's "+0.0006 over the closed form" has no CI** and the comparison is confounded (flat+MSE+
  closed-form vs learned-weights+IDF+KL+SGD changes three things at once). The controlled ablation
  is `program.phase4_mandatory` p4-weights and has not run. Do not treat "ship the flat table" as
  decided.
| p2s-sane-5e5 | `work/runs/p2s-sane-5e5.json` | — | FAILED — RuntimeError: shape '[256, 31, 768]' is invalid for input of size 3145728 |
| p2s-sane-1e5 | `work/runs/p2s-sane-1e5.json` | — | FAILED — RuntimeError: shape '[256, 31, 768]' is invalid for input of size 3145728 |
| p2s-sane-1e4 | `work/runs/p2s-sane-1e4.json` | — | FAILED — RuntimeError: shape '[256, 31, 768]' is invalid for input of size 3145728 |
| p2s-warmup-only | `work/runs/p2s-warmup-only.json` | — | FAILED — RuntimeError: shape '[256, 31, 768]' is invalid for input of size 3145728 |
| p2s-start | `work/runs/p2s-start.json` | 0.4548 | ok |
| p2s-start | `work/runs/p2s-start.json` | 0.4548 | ok |
| p2s-sane-1e5 | `work/runs/p2s-sane-1e5.json` | 0.4584 | ok |
| p2s-sane-5e5 | `work/runs/p2s-sane-5e5.json` | 0.4626 | ok |
| p2s-sane-1e4 | `work/runs/p2s-sane-1e4.json` | 0.4653 | ok |
| p2s-old-lr-3e3 | `work/runs/p2s-old-lr-3e3.json` | 0.4546 | ok |
| p2s-sane-randneg | `work/runs/p2s-sane-randneg.json` | 0.4659 | ok |
| p2x-rn-3e4 | `work/runs/p2x-rn-3e4.json` | 0.4649 | ok |

## Phase-2 screen, 2026-08-26 — the contrastive objective is NOT broken

All arms objective A only, 2,000 steps, from the SAME `p1-objB` checkpoint (see the redesign note in
`LEDGER.md`). CIs and Holm: `results/m7_phase2_screen_cis.json`. Verdict file:
`results/m7_contrastive_verdict.json` — **kill criterion may not fire**, four arms beat the 0.4548
bar CI-resolved.

| arm | proxy macro-3 | vs start | verdict |
|---|---|---|---|
| `p2s-start` (0 steps) | 0.4548 | — | reproduces the checkpoint exactly; pins the baseline in-harness |
| `p2s-sane-1e5` | 0.4584 | +0.0036 [0.0022, 0.0050] | resolved |
| `p2s-sane-5e5` | 0.4626 | +0.0077 [0.0050, 0.0105] | resolved |
| `p2s-sane-1e4` | 0.4653 | +0.0104 [0.0069, 0.0139] | resolved |
| `p2s-old-lr-3e3` | 0.4546 | -0.0002 [-0.0062, 0.0059] | **flat, unresolved — with warmup AND mined negatives** |
| `p2s-sane-randneg` | 0.4659 | +0.0111 [0.0084, 0.0139] | resolved; best arm |

Three findings, and two of them overturn entries above:

1. **Phase 1's "the contrastive objective is broken" was a learning-rate artifact.** At published
   rates the same objective *improves* a good table, monotonically in lr from 1e-5 to 1e-4. At phase
   1's 3e-3 it is flat even with warmup and mined hard negatives — so the lr, not the objective, is
   what phase 1 measured. The diagnosis that named lr the "leading untested hypothesis" is now
   tested and confirmed.
2. **Mined hard negatives HURT**, at matched lr with one variable changed: random-only beats
   teacher-mined by +0.0034 [0.0019, 0.0049], resolved. This contradicts the ledger's "scale without
   hardness wasted the objective" and restores the mandate's premise that large cheap negative pools
   are the thing to exploit. `fn_masked_frac` is 0.0051, so the false-negative filter is not what
   removes the benefit — STATUS predicted it would "bite far harder" with mined negatives; it does not.
3. **The magnitudes are small**: the best arm is +0.0111 on the proxy over a checkpoint whose own
   projection to the six is ~0.41. A real, resolved, cheap gain, and not a tier-changer on its own.

Caveat on selection: `p2x-rn-3e4` peaks at step 500 (0.4661) and declines by step 1000, so an arm's
final-step macro is not its best-step macro. Any config taken forward must fix the step budget as
part of the config, or select on best-eval consistently across arms and say so.

| p2x-rn-1e3 | `work/runs/p2x-rn-1e3.json` | 0.4629 | ok |
| p2x-rn-3e3 | `work/runs/p2x-rn-3e3.json` | 0.4521 | ok |
