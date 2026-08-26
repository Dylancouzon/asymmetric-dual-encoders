# M7 status

**Stage:** teacher swapped to **stella_en_400M_v5** (Dylan, 2026-08-26, on the learnability probe);
its 6.17M-doc encode is running (~3.5 h, `logs/stella_swap.log`). Phase 2 answered. Next session:
finish the swap, then the four Codex blockers. Detail in `m7/RESULTS.md` (runs), `m7/LEDGER.md`
(protocol + the Codex gate's open list), `m7/EXPLORED.md` (closed avenues), `results/m7_*.json`.

## Run these next, in this order

1. **Check `logs/stella_swap.log` finished** (`run_stella_swap.sh` is idempotent; re-run it to
   resume — every step is cached). It does: cache-key gate → init-row gate → dev encodes
   (nq-250k, hotpotqa) → 6.17M-doc pool → teacher reference rows → closed-form ridge on the full
   dev suite. That last number is the first real read on stella's retention.
2. **`gate.py` under `M7_ENCODER=stella-400M-v5`** — its refs file is separate
   (`work/devres/refs-stella-400M-v5.json`, BM25/potion rows copied since they are
   teacher-independent), so no comparison can mix teachers.
3. **Re-run the phase-2 winner under stella**: objective A from a stella-distilled B checkpoint, lr
   in the 5e-5…3e-4 band, `hard_neg_k=0`. Everything about that band was established on bge-base and
   needs one confirmation, not a re-sweep.
4. **Then the four Codex blockers** (below). They block claims, not compute, and nothing confirmatory
   can be reported until they are fixed.

## Why stella, and what it cost

`results/m7_learnability_report.json`: eight candidates ranked by the **closed-form table distilled
from each**, fitted on 349,934 TRAIN query vectors, scored against each teacher's own documents.
Every row CI-resolved against the incumbent; only stella beats it (**+0.0365 [0.0249, 0.0481]**).

**Spearman(teacher ceiling, distilled table) = 0.000.** arctic-embed-l has the best ceiling of the
eight and a table **0.0480 below the incumbent** — it was approved that morning on a symmetric probe,
and swapping to it would have shipped a worse system after a 3-hour encode. Two mechanisms were
tested and refuted: pooling (`arctic-embed-l-mean`, same weights and dim, ratio 0.526 → 0.472) and
cosine agreement as a proxy (rises with lambda while nDCG falls; mis-ranks candidates). **Stella's
advantage is unexplained**, so there is no attribute on which to search further candidates.

Cost: stella discloses **ArguAna and FiQA2018** — 2 of the 6 confirmatory datasets. Dylan's call is
the **six-set claim as primary**, with the four clean datasets (SciFact, NFCorpus, SCIDOCS,
TREC-COVID) as a robustness number. Both bars are precomputed in `results/m7_bars_clean4.json`, and
the restriction is not a soft option: Tier 2 gets easier (0.4583 → 0.4541), Tier 1 harder
(0.4868 → 0.4974). Promoting the four-set to headline later is legitimate **but must be labelled
post-hoc** — see `LEDGER.md`.

## What phase 2 settled

`results/m7_phase2_screen_cis.json`, all arms from one fixed p1-objB checkpoint:

- **The contrastive objective was never broken — phase 1 measured its learning rate.** 5e-5 to 3e-4
  all improve the table, CI-resolved and mutually unresolved, so the optimum is a **band**. 1e-3
  still helps; 3e-3 (phase 1's lr) is negative. `contrastive_verdict()` records that the kill
  criterion may not fire.
- **Mined hard negatives HURT** at matched lr: random-only +0.0034 [0.0019, 0.0049]. `phase2_negatives`
  is demoted. `fn_masked_frac` is 0.0051, so the false-negative filter is not the mechanism.
- **The gain is small**: +0.0111 proxy macro, against the retention points the bars need. An arm's
  final-step macro is not its best-step macro (3e-4 peaks at step 500) — fix the step budget in the
  config or select on best-eval consistently, and say which.

## Still open from the Codex gate (full list + dispositions in `LEDGER.md`)

Four blockers, none stopping compute, all stopping a *claim*: decontamination indexed only the 855K
positives while training touches all 6.17M pool docs; the bootstrap p-values are percentile tails, so
Holm controls nothing over them; the frozen fusion function differs between dev selection and final
scoring; and the two-access rule is already breached (`bench_throughput` read FiQA qrels), now logged.

## Open for Dylan

1. **Nothing is blocking.** The teacher ruling is made and logged; the encode is running.
2. **Host:** Windows Update rebooted the box mid-morning once already — a 3.5 h encode is exposed.
3. Later: HF release go, and a view on shipping a teacher whose training data covers 2 of the 6 eval
   sets (the four-set robustness number is the technical answer; the presentation is a judgement).
