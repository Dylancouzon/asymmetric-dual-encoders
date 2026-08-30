# M9.2 — the recipe lock for the seven-day build (DRAFT, blanks filled when the screen lands)

Status: **DRAFT.** Every `‹…›` is filled from `results/m9_screen_decisions.json` before this file is
committed and before `longrun` opens a session. Reviewed adversarially, then locked, then launched.
`m9/LEDGER.md` gets a short §14 pointing here; the bulk lives in this file so the guarded protocol
file stays small.

## 1. What the screen decided

| field | value | from |
|---|---|---|
| teacher | ‹teacher› | `m9s2`/`m9s3` vs `m9s1`, SCREEN-3 family weights, margin 0.010 |
| student | ‹student› | `m9s4` vs anchor, DEV-6, margin 0.0056 |
| prompt policy | ‹prompt› | `m9s5` |
| mix | ‹mix› | `m9s6` |
| head warm start | **on**, ridge λ = 1e-4 from a training-only holdout | `results/m9_warmfit.json` |
| capacity probe | **withdrawn before running** (Dylan, on Codex #5) — M9 cannot act on either answer under the 35M cap; carries to M10 | `m9/RESULTS.md` |

## 2. Why this run exists, in one paragraph

The anchor reached **73.3%** of the teacher's SCREEN-3 ceiling on 59.5 M non-pad tokens, with
quarter-on-quarter gains of **+0.0330, +0.0132, +0.0054** — halving each time. More epochs on
242,786 queries asymptote near 74%, so the remaining ~27% is not behind more SGD on that data. It
is behind **unique text**: the document pool is 6,149,679 pre-screened rows whose stella vectors
already exist, and the extended screen added 220,528 real questions. Seven days at the measured
26,854 tokens/s is **16.24 B tokens, 273× the anchor dose**, over ~6.6 M unique texts against
LEAF's 6.7 M. That is the first M9 run at a dose where the aim is reachable rather than arithmetic.

## 3. Corpora (every byte hashed in `work/m9long/manifest.json`, recomputed by `longrun verify`)

| corpus | texts | tokens/epoch | role |
|---|---|---|---|
| `queries_pair` | 242,786 | 3,719,242 | query |
| `nqopen` | 85,863 | 1,008,102 | query |
| `triviaqa` | 134,665 | 2,685,360 | query |
| **real queries total** | **463,314** | **7,412,704** | |
| `documents` | 6,149,679 | ~583 M | document |
| `pseudoq` | 923,408 | 38,220,549 | short document spans (M7 vocabulary mitigation, **not** questions) |

Every one is decontaminated against the six, dev, untouched-final, the LoTTE shadow and the
M9-reserve — 95,055 protected queries. MS MARCO stays out. FEVER stays out.

## 4. The mix — the one open decision, and the arithmetic behind it

At 16.24 B tokens, a token share buys these epoch counts:

| query share | real-query epochs | document epochs | span epochs |
|---|---|---|---|
| 20% | 438 | 19.5 | 42 |
| 15% | 328 | 20.9 | 42 |
| **10%** | **219** | **22.3** | **42** |
| 5% | 110 | 23.7 | 42 |

The query side is thin however it is sliced: 463,314 texts against 16 B tokens. Two facts bound the
choice. LEAF's own ablation found documents did the heavy lifting even for a query-serving student
(queries-only 46.7 vs queries+docs 60.7 on NanoMSMARCO), and its own corpus was overwhelmingly
document text. Against that, nano serves **only** queries, and the query manifold is what it is
scored on.

**REGISTERED (before `m9s6` ran): the m9s6 verdict maps to the build as** — `70/30` (documents
help at matched dose) → the 5/5/90 build below; `query-only` (documents failed their only direct
test) → **`make_config` refuses and the build STOPS for Dylan's ruling on shares**, because a
seven-day 90%-document bet would then be unsupported by its own screen.

**REGISTERED: 5% real queries / 5% spans / 90% documents by non-pad token**, on Codex review #5's
recommendation and against my own 20/10/70 draft. The reason is repetition, not weight: with a true
combined-example mean, queries are short enough that a 5% *token* share is still ~23% of the
*objective*, while real-query presentations fall from ~438 to **109.6**. Realized epochs at the
seven-day horizon: queries 109.6, spans 21.2, documents 25.1; **113 examples a step** (25 queries,
10 spans, 78 documents), which is the same order as LEAF's batch 32 once documents dominate. The in-run instrument is a **SCREEN-3 macro on held-out dev queries every
`eval_every` steps** — if that curve turns down while training loss keeps falling, the query side
is being fitted and the run stops or the share drops. Both outcomes are actions.

## 5. Schedule — warmup, stable, decay on demand

Not cosine-to-a-horizon: a cosine commits to a length, so stopping early leaves the LR high and
extending is impossible. Instead `warmup → stable → decay-on-demand`, with the cooldown run against
whatever stable checkpoint we choose to stop at. **How long to run becomes an observation.**

| field | value |
|---|---|
| warmup | **2,000** steps, linear to `lr_peak` |
| stable | `lr_peak` = 1e-4, indefinite |
| cooldown | **7,265 steps = 59,507,872 tokens**, cosine to `lr_final` = 1e-5, resumable, triggered by `longrun.py decay`. That is the anchor arm's entire dose — the only annealing scale this project has measured. The 4,000-step default it replaces was five minutes and supported by nothing |
| optimizer | AdamW, β (0.9, 0.999), eps 1e-8, wd 0.01 on dim>1, grad-clip 1.0 |
| precision | bf16 autocast, loss in fp32 |
| tokens/step | **8,192** |
| checkpoint | every **5,000** steps, atomic (tmp + fsync + `os.replace` + directory fsync) |
| eval | every **20,000** steps (~164 M tokens, ~1.7 h; 99 over the horizon), SCREEN-3, appended to `history.jsonl` after the checkpoint so replay can dedupe |

## 6. Stopping rules, registered before the run

All four are implemented in `longrun.check_kill` and the training loop, and each names itself in
the stop message.

- **Non-finite loss or gradient norm** → stop at that step. One NaN can poison Adam state and every
  checkpoint after it, silently.
- **Regression:** two consecutive evaluations more than **0.0056** (the MDE) below the best
  checkpoint → stop.
- **Plateau:** less than **+0.001** SCREEN-3 over **1 B tokens** → the trainer enters the cooldown
  itself, then stops. Judged in tokens, never in steps, because that is the unit the dose is
  registered in. The **stable-phase token cap** ends the same way: cooldown, then stop. Neither
  leaves an unannealed checkpoint waiting for a human.
- **Throughput collapse:** below **50%** of the session's early median → stop. Page-cache thrashing
  against a 12.6 GB target map would otherwise silently reduce delivered dose while the wall clock
  ran out.
- **Stop on wall clock:** seven days, or whenever Dylan needs the box; the `STOP` file halts
  cleanly at the next step boundary and `decay` still produces a servable artifact. Under the
  watchdog the trainer runs `--anneal-before-deadline`: when the remaining session time no longer
  fits the cooldown at the measured rate (×1.25 margin), it enters decay by itself, so the horizon
  cannot truncate the anneal even at degraded throughput. Every decay entry checkpoints
  immediately, so a restart resumes into the cooldown, never into stable LR.
- **Any stop yields a model.** There is no state in which the run has to be thrown away.

## 7. What this run may and may not do

It produces **one candidate**. It may not touch the six, the reserved four, or LoTTE. Its SCREEN-3
evaluations are DEV reads and continue the reuse counter. The confirmatory six-set transaction and
the reserved batch are M9.4 and are governed by `m9/LEDGER.md` §Final run, unchanged.

## 8. Before launch — the checklist

- [ ] `results/m9_screen_decisions.json` complete, every mandatory arm present
- [ ] teacher targets encoded for `nqopen`, `triviaqa`, `pseudoq` (~1.14 M texts, one stella pass)
- [ ] `longrun prepare` finished, `corpora.json` hashes written
- [ ] `work/m9long/config.json` generated from this file
- [ ] **resume tested for real**: train N steps, kill, resume, verify the stream continues and the
      loss curve is continuous — not assumed
- [ ] adversarial review of this lock actioned
- [ ] committed and pushed before the run opens its session
