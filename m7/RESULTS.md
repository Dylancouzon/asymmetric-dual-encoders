# M7 experiment results

Every run lands here, including stopped, failed and OOM ones — the experiment ledger the mandate
requires. Written by `m7src/sweep.py`.

> **Compacted 2026-08-28** from 3.6K tokens to budget: one row per run, duplicate rows merged,
> per-run narrative moved to the run JSONs and the surviving findings to `m7/LEDGER.md` /
> `m7/EXPLORED.md`. Nothing is lost — `git log -p m7/RESULTS.md` has every original row.

**Read the dev metric carefully.** The table column is the *fast proxy macro* over three
components (nq-250k, cqadup-programmers, cqadup-physics) used by in-training evaluation, so a
sweep of ~25 configs stays affordable. Every **selection and gate** decision instead uses the full
pinned six-component dev suite. The two are not comparable: under stella the teacher scores 0.6151
on the proxy and 0.6724 on the full suite.

Reference rows, full pinned suite (`work/devres/refs-stella-400M-v5.json`): teacher (symmetric,
prefixed) **0.6724** · BM25 **0.4525** and potion-retrieval-32M **0.3801** on the four text-backed
components, which is where G1/G3 run — BM25 and potion have no row on the held-out slices.

## Runs

| run id | proxy macro-3 | verdict |
|---|---|---|
| p1-objB | 0.4548 | ok — objective B, 8k steps. Matches the closed-form flat bound (0.4542) to +0.0006 (no CI, and confounded: three things differ at once). Coverage 27,314/30,522 rows. |
| p1-objA | 0.3248 | ok, objective declines monotonically at every eval. See the correction below. |
| p1-objC | 0.3721 | ok — B(4k) reached 0.4449, then the contrastive phase degraded it to 0.3721. |
| p2s-sane-{1e5,5e5,1e4}, p2s-warmup-only | — | FAILED — `RuntimeError: shape '[256, 31, 768]' invalid`, one shape bug hit four times because no arm with mined hard negatives had ever run. Fixed; `sweep.smoke` exists because of this. |
| p2s-start (0 steps) | 0.4548 | reproduces the checkpoint exactly; pins the baseline in-harness |
| p2s-sane-1e5 | 0.4584 | +0.0036 [0.0022, 0.0050] vs start, resolved |
| p2s-sane-5e5 | 0.4626 | +0.0077 [0.0050, 0.0105], resolved |
| p2s-sane-1e4 | 0.4653 | +0.0104 [0.0069, 0.0139], resolved |
| p2s-old-lr-3e3 | 0.4546 | −0.0002 [−0.0062, 0.0059] — flat, WITH warmup and mined negatives |
| p2s-sane-randneg | 0.4659 | +0.0111 [0.0084, 0.0139], best bge arm |
| p2x-rn-3e4 / p2x-rn-1e3 / p2x-rn-3e3 | 0.4649 / 0.4629 / 0.4521 | lr edge extension; the curve turns over |
| s1-objB | 0.4903 | first stella B checkpoint (8k), 0.007 UNDER its closed form |
| s2-start / s2-rn-5e5 / s2-rn-1e4 / s2-rn-3e4 | 0.4903 / 0.4993 / 0.5035 / 0.5049 | band confirmed under stella, monotone in lr |
| s2x-rn-1e3 | 0.5051 | labelled edge extension; best 0.5059@1000, peak-and-turn |
| s2w-3e4-s1500 | 0.5051 | full-suite 0.5907 |
| **s2w-1e3-s1000** | 0.5052 | **cross-arm winner on the FULL suite, 0.5987** — the proxy tie hid a real gap. Retention 0.890. Gate #2: GO. |
| p35b-500k / p35a-500k-1e3 / p35w-500k-s1500 | 0.4934 / 0.5077 / 0.5075 | lever #2, 500k pseudo mix; full-suite 0.6052 |
| p35b-2m / p35a-2m-1e3 | 0.4981 / 0.5109 | lever #2, 2m mix; full-suite 0.6090 |
| p35a-2m-1e3-x4000 | 0.5115 | labelled steps extension: peak 0.5119@2500 then plateau |
| **p35w-2m-s2500** | 0.5106 | **the candidate**, full-suite 0.6113 (0.6153 with the adopted `sqrt` pooling rule) |

## Findings that overturned an earlier entry

1. **"The contrastive objective is broken" was a learning-rate artifact.** At published rates
   (1e-5..1e-4) the same objective *improves* a good table, monotonically in lr; at phase 1's 3e-3
   it is flat even with warmup and mined negatives. `m7_contrastive_verdict.json`: the kill
   criterion may not fire — four arms beat the bar CI-resolved.
2. **p1-objA's recorded cause was refuted.** "Random negatives trivially separable" predicts
   stasis, not monotone decline, and the loss fell 0.51 → 0.13 *while* dev fell at every eval:
   the objective was being optimised successfully and what it optimised was anti-correlated with
   retrieval. The mechanism was asserted, never tested.
3. **Mined hard negatives HURT** at matched lr with one variable changed: random-only beat
   teacher-mined by +0.0034 [0.0019, 0.0049]. `fn_masked_frac` 0.0051, so the false-negative
   filter is not the cause. **Caveat, 2026-08-28**: this is ONE bge-era pair at lr 5e-5, and
   `hard_neg_k=0` has been assumed ever since without ever being written down as a closed avenue.
   `phase4_negatives` tests it properly at the shipping lr under stella.
4. **An arm's final-step macro is not its best-step macro** (`p2x-rn-3e4` peaks at step 500).
   Hence the pre-registered every-500 step rule.

## Teacher learnability, 2026-08-26 — the approved teacher was worse than the incumbent

`results/m7_learnability_report.json` holds all eight candidates' rows. A closed-form flat table
per teacher, scored against **that teacher's own documents** on the two CQADupStack dev
components. Only **stella-400M-v5** beats the incumbent (+0.0365 [0.0249, 0.0481]);
**arctic-embed-l, approved that morning on its symmetric ceiling, is −0.0480 BELOW it**.

1. **Ceiling does not predict what ships**: Spearman(ceiling, table) = 0.000 over the eight; the
   best ceiling ranks fifth on the metric that matters.
2. **Pooling is not the mechanism**: `arctic-embed-l-mean` is the same weights read out as a mean
   — ratio falls 0.526 → 0.472. Stella's advantage is unexplained, so there is no rule to search
   other candidates on.
3. **Cosine agreement is not the metric**: it rises with lambda while nDCG falls, and mis-ranks.
4. Within a family, lower dim is more approximable (bge-base > bge-large, e5-base > e5-large);
   stella breaks the pattern.

Caveats: closed-form and flat, so it ranks candidates rather than predicting scores; two
components of one dataset family; dev-only.

## Cloud-Inference teacher probe, 2026-09-04 — mxbai is 8th of 11

Dylan: prefer a teacher hosted by **Qdrant Cloud Inference** (stella is not hosted); vendor rule
bent for Mixedbread **because Qdrant Cloud hosts it**. Same criterion, same fit set (gram nnz
12,936,008 identical to the 2026-08-26 sweep), same two dev components.

**`mxbai-embed-large-v1` table 0.2509, d = −0.0930 [−0.1054, −0.0805] vs stella, RESOLVED** —
below bge-base by 0.0565, between arctic-embed-l and e5-large-v2. Loader exact (pairwise
max|Δ| 0.00e+00 vs sentence-transformers); λ-curve flat 0.247–0.251 then falls at 1e-1.

- Its one advantage: **no disclosed MTEB training exposure** (card/blog: none except MS MARCO), so
  no ArguAna/FiQA/FEVER caveat, unlike stella.
- Best hosted candidate remains **bge-base-en-v1.5**, −0.0365. Six-set cost of that swap is
  **unmeasured**; two unvalidated transfers put it at 0.045–0.07. The 0.406→0.489 MTEB→six tower
  projection may NOT be used for it (refuted, `EXPLORED.md`).
- λ grid completed 2026-09-04 (Codex: bge's optimum sat at the grid EDGE, biasing the
  comparison): **bge-base λ=0.1 → 0.2848**, down from 0.3074. Optimum interior; objection closed.
- A swap requires the regenerated clean fit list (`m7_trainq_manifest.json`); this probe ranks only.

## Off-family teacher check, 2026-09-04 — the confound is refuted

Every component of the pinned dev suite carries stella exposure (NQ, HotpotQA disclosed;
CQADupStack is StackExchange, of which stella discloses four datasets); bge-base discloses none.
The 2026-08-26 family-exposure read used nq-250k as the control — itself on stella's list. Codex
finding 14 (MAJOR) named the fix; run now. `scripts/offfamily_check.py`,
`results/m7_offfamily_report.json`.

Held-out SQuAD + ESCI, on **no** candidate's disclosed list, full 6.17M pool, λ*=0.01 each:

| stratum | n | stella | bge-base | d [CI95] |
|---|---|---|---|---|
| squad-train | 1,790 | 0.5284 | 0.3632 | **+0.1652** [0.1503, 0.1801] |
| esci-us | 1,598 | 0.3495 | 0.3111 | **+0.0384** [0.0293, 0.048] |
| macro | | | | **+0.1018** [0.093, 0.1107] |

Stella first on both, by ~3× the CQADupStack delta (+0.0365) that selected it — the selection set
**understated** its advantage. **ESCI carries the argument**: Amazon product text is neither
Wikipedia nor StackExchange nor a QA benchmark, so it is independent of stella's disclosed
training in domain as well as by name, and stella still wins CI-resolved. SQuAD is Wikipedia, so
it is domain-adjacent to stella's NQ/HotpotQA/FEVER/MIRACL training even though the dataset is not
disclosed — do not lean on that row alone. Neither stratum resembles the six: this is a
contamination control, not a prediction of the six-set gap.

## Teacher swap, 2026-08-26 — stella's closed-form table beats bge's best TRAINED arm

`m7_stage0_ridge_stella.json`: proxy-3 **0.4973** at lam=0.01 (bge closed form 0.4542, bge best
trained arm 0.4659); teacher proxy ceiling 0.6151 → retention 0.808. The pre-registered
family-exposure read PASSES — the advantage on Wikipedia nq-250k (+0.063) exceeds that on the two
StackExchange components (+0.047/+0.020), so the ranking was not StackExchange-specific.
| p4x-nopseudo-b | `work/runs/p4x-nopseudo-b.json` | 0.4938 | ok |
| p4x-nopseudo-a | `work/runs/p4x-nopseudo-a.json` | 0.5072 | ok |
| p4x-pseudo500k-b | `work/runs/p4x-pseudo500k-b.json` | 0.4970 | ok |
| p4x-pseudo500k-a | `work/runs/p4x-pseudo500k-a.json` | 0.5098 | ok |
| p4-base-b | `work/runs/p4-base-b.json` | 0.4981 | ok |
| p4-base-a | `work/runs/p4-base-a.json` | 0.5106 | ok |
| p4-input-emb-b | `work/runs/p4-input-emb-b.json` | 0.4977 | ok |
| p4-input-emb-a | `work/runs/p4-input-emb-a.json` | 0.5113 | ok |
| p4-random-b | `work/runs/p4-random-b.json` | 0.4986 | ok |
| p4-random-a | `work/runs/p4-random-a.json` | 0.5112 | ok |
| p4-prefix-b | `work/runs/p4-prefix-b.json` | 0.4976 | ok |
| p4-prefix-a | `work/runs/p4-prefix-a.json` | 0.5104 | ok |
| p4-flat-b | `work/runs/p4-flat-b.json` | 0.5000 | ok |
| p4-flat-a | `work/runs/p4-flat-a.json` | 0.5091 | ok |
| p4-uniform-w-b | `work/runs/p4-uniform-w-b.json` | 0.4982 | ok |
| p4-uniform-w-a | `work/runs/p4-uniform-w-a.json` | 0.5115 | ok |
| p4-reg0-b | `work/runs/p4-reg0-b.json` | 0.4981 | ok |
| p4-reg0-a | `work/runs/p4-reg0-a.json` | 0.5106 | ok |
| p4n-bank-a | `work/runs/p4n-bank-a.json` | 0.5106 | ok |
| p4n-teacher16-a | `work/runs/p4n-teacher16-a.json` | 0.5125 | ok |
| p4n-bm2516-a | `work/runs/p4n-bm2516-a.json` | 0.5131 | ok |
| p4n-mixed32-a | `work/runs/p4n-mixed32-a.json` | 0.5131 | ok |
| p4e-prefix-init-b | `work/runs/p4e-prefix-init-b.json` | 0.4975 | ok |
| p4e-prefix-init-a | `work/runs/p4e-prefix-init-a.json` | 0.5105 | ok |
| p4p-sqrt-a | `work/runs/p4p-sqrt-a.json` | 0.5111 | ok |
| p4n-teacher16-s1500-a | `work/runs/p4n-teacher16-s1500-a.json` | 0.5126 | ok |
| p4n-bm2516-s1500-a | `work/runs/p4n-bm2516-s1500-a.json` | 0.5138 | ok |
| p4n-mixed32-s1000-a | `work/runs/p4n-mixed32-s1000-a.json` | 0.5149 | ok |
| p5s-simple-b | `work/runs/p5s-simple-b.json` | 0.4970 | ok |
| p5s-simple-a | `work/runs/p5s-simple-a.json` | 0.5139 | ok |
| p5s-simple-nohn-a | `work/runs/p5s-simple-nohn-a.json` | 0.5106 | ok |
| tax-msmarco-b | `work/runs/tax-msmarco-b.json` | 0.5050 | ok |
| tax-msmarco-a | `work/runs/tax-msmarco-a.json` | 0.5116 | ok |
