# M12 — `constella-zero-hybrid`: is there anything for a fusion-aware objective to move?

Created 2026-09-04 (Dylan); **rewritten the same day after Fable's review**, which showed the first
draft was not executable — see `m12/EXPLORED.md` §1 for what was cut and why. Binds from
`instructions-m7.md` unchanged. Working files `m12/`, branch `m12-work`.

**Local 3080 only.** No cloud budget. M10 stays paused; M12 does not touch it.

## The question, stated correctly

`zero`'s final phase is **objective A, InfoNCE against frozen document vectors** (32,768 negatives,
temp 0.02, 2,500 steps — `work/runs/p35w-2m-s2500.meta.json`), NOT regression onto stella's query
vector. So "train against the fused ranking" means: add BM25 to the score the InfoNCE loss ranks on.

Under convex0, `F = w·s/s_max + (1−w)·b/b_max` with `b` frozen — BM25 enters as a **fixed additive
logit bias per candidate**, and the gradient reaches `W` only through `s`. That is not a no-op in
general; it moves the optimum wherever training candidates exist for which the dense and fused
margins disagree.

**But M8 measured the shipped objective inert on the current candidate set**: the table already
ranks the positive first for **99.75%** of training queries, uniform-bank KL median **4.73e-07
nats** (`m8/FINDINGS.md` §3.1). A logit bias on a saturated softmax changes nothing. The same
source shows where signal survives: the `teacher_top200` variant measures **0.777 nats**.

**So the milestone's real question is whether a fusion-aware objective has any gradient at all, and
that is decided by the CANDIDATE SET, not by the loss.** Candidates must be mined from the
**union of dense top-k and BM25 top-k**. A hybrid arm trained on uniform banks is a measured no-op
and must not be run.

Miss-is-publishable, and here a miss is informative: it says the table has no non-lexical capacity
to redistribute — the sharpest available statement about this architecture's ceiling.

## Gate A — DBSF on existing runs. Do this first; it may be the whole milestone.

**RRF cannot be a training target**: it is a function of ranks, piecewise-constant, zero gradient
almost everywhere. Surrogates need a temperature this mandate will not sweep. **DBSF is
score-based** (per-query mean±3σ, then sum), differentiable with the statistics detached, and Qdrant
ships it beside RRF.

Add ~30 lines of `dbsf` to `m7src/fusion.py` and score the **existing, unchanged** `zero` on the
existing `work/fusionruns` BM25 runs and dense runs. CPU, ~1 hour, no training, dev only.

- **If DBSF ≈ convex0 (~0.57)** the 0.022 RRF gap (dev RRF k=10 **0.5504** vs convex0 **0.5727**,
  `m7/LEDGER.md:922`) closes with an *operator recommendation and no retraining* — worth more than
  anything the training gates can deliver, and it fixes the caveat now standing in `m11/STATUS.md`.
- **If DBSF ≈ RRF**, the operator is the bottleneck and no table fixes it. Say so and stop.

Register the winning operator here; every later number uses it. Disclose that FastEmbed's
`Qdrant/bm25` (fixed `avg_len`, own tokenizer) is not `bm25s`-lucene, so the fusion **weight** is
tied to the lexical function even where the complement is only weakly tied.

## Gate B — two probes that CAN fail. Under an hour, no training.

The first draft's "headroom = fused(teacher) − fused(zero), kill below 0.005" was **not a gate**:
teacher dense alone (0.6350) already beats fused zero (0.5727) by 0.062, so it could not fail.

1. **Tension count.** Over union(dense top-200, BM25 top-200) on the training pool, the fraction of
   training queries whose fused ranking is not already optimal. **≲1% → no gradient exists → STOP.**
2. **Weighted ridge.** Refit the closed-form table (`blockcg.py`) with per-query weights ∝ BM25
   weakness; score fused on dev against the unweighted ridge. **< 0.004** (the fused noise floor,
   `results/m8_noise_floor_fused.json`) **→ nothing to redistribute → STOP.**

Both are descriptive, dev/train only, no confirmatory access. Per-query bucketing of the
teacher−zero loss by BM25 nDCG is kept as description, not as a gate.

## Gate C — if both survive: 2 recipes × 3 seeds, and a control

**A hybrid arm alone proves nothing.** Warm-starting and running more A-phase steps moves the dev
macro **0.0027–0.0078 on step count alone** (`m7/LEDGER.md:314-324`). So:

| arm | loss | candidates | seeds |
|---|---|---|---|
| **control** | unchanged objective A | union-mined (same as hybrid) | 3 |
| **hybrid** | objective A on the fused score | union-mined | 3 |

Same warm start, same steps, same candidates. This is also **the first recipe-replication term this
repo has ever had** (`m7/FINDINGS.md` 9 records that every CI to date is query-sampling only).

**Registered bar: hybrid − control ≥ 0.008 on the fused dev macro** — the recipe-perturbation band,
not the 0.0006 seed floor. Below that, M12 reports a null and stops.

**Fusion parameter, registered now to settle a live contradiction.** The plan may not say
"co-registered, not re-fitted": `m7/LEDGER.md:655-656` requires re-selection whenever the checkpoint
changes, and a hybrid table changes the `s/s_max` distribution by construction. **Both arms
re-select their fusion parameter on dev, by the identical procedure**, and the fitting-procedure
floor from `m8_noise_floor_fused.json` is reported alongside. Fixed-`w` would structurally handicap
the hybrid arm and bias the experiment toward the null.

**Precompute, to be budgeted and smoked before launch** (CLAUDE.md's long-run rule; the last BM25
mining pass cost 3.6 hours against an estimated 3 minutes): BM25 over the 6.17M-doc training pool
for union mining. `train.mine_bm25_negatives` exists (`m7src/train.py:217`). Read the first progress
line and check the rate.

## What M12 does NOT spend

- **The reserved four: not touched.** No stella encode of them exists — **10.1M docs, tens of hours
  and ~21 GB** on this box (`m7/LEDGER.md:100-101`), no comparator vectors (`m8/LEDGER.md:19`), and
  M8's registered design for exactly this comparison has **MDE 0.0068 with P(ship | δ=0.005) = 0.21**.
  Spending the project's last access at one-in-five odds, on an effect class where every lever to
  date is ≤0.005, is the worst available trade. M16's co-adaptation is its natural owner.
- **LoTTE:** only if Gate C's controlled dev effect clears 0.008. Corpora are at `work/lotte`.
- **The six:** descriptive only, labelled as such. `instructions-m8.md` item 2 already makes every
  post-M7 six-set claim development-informed for any successor — this is not fusion-specific.
- **Hybrid comparators:** at most **one** (`bge-small` + BM25, the release bar's natural hybrid),
  and only under the Gate-A operator. Each additional comparator needs its own dev encodes
  (hotpotqa alone is 5.23M docs) and a registered six-set access class. Prefer deferring all of them
  to M14's paper.
- **nano, the document tower, a better teacher, cloud compute, fusion sweeps.** Out.
- **The offline BM25 index cost row** — a real gap in the M7 headline, but it does not bear on this
  hypothesis. Separate ticket, not a gate here.

## Ship shape — decided at the end, not now

Do **not** pre-commit to "ships alongside, never replacing". Two artifacts means two cards, two
FastEmbed entries (M13's "one clean PR, all three" becomes four), two ports and two freezes. Gate C's
**dense-only regression decides**, and every result row carries fused AND dense-only for the same
table. If the regression is small, a single-artifact route exists: mixed batches with the BM25 bias
present/absent at one fixed registered ratio, shipped as `constella-zero` v2 under M8's registered
replacement rule (`m8/LEDGER.md:19`).

## Deliverables

1. Gate A's operator result — possibly the whole milestone.
2. Gate B's two probe numbers, and the STOP if either fires.
3. If reached: control-vs-hybrid × 3 seeds against the registered 0.008 bar, fused and dense-only.
4. `m12/FINDINGS.md` (transferable), `m12/EXPLORED.md` (dead ends, starting with the cut first
   draft), `m12/LEDGER.md` (registrations), decisions in `CLAUDE.md`.
