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
