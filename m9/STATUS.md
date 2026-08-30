# M9 status — M9.1 in flight

**Stage: nothing is running. M9.1's three recipe arms are outstanding; the M9.3 build is written and reviewed but NOT launchable — see the blocker list below.** Branch `m9-work`. Nothing has touched
the six or the reserved four; LoTTE unread. `results/perquery.json` untouched.

M9.1 is **staged**, on Codex's recommendation after it reviewed the lock *and the code*: stage A is
the four gates plus the **anchor arm** and its warm-start contrast; stage B — the seed replica and
the five contrast arms — runs only if the anchor clears a pre-registered **adequacy gate**
(retention ≥ 0.60 of the **SCREEN-3** ceiling and a late-checkpoint slope ≤ 0.02). Its words were: *"the
only defensible next GPU action is a corrected, fully guarded anchor curve — not all nine arms."*
Spending six GPU-hours on contrasts measured at a dose where the student sits far from the teacher
surface would have ranked early imitability, not the factors under test.

## Needs Dylan (two items, both logged, neither blocking tonight)

1. **Mandate amendment — ratify or reject.** `instructions-m9.md` fixes the screen's tuning-dev
   macro at all six pinned components. That is impossible for a *challenger teacher*: re-encoding
   full dev in one costs 11.72M document encodes — 46.5 GPU-hours for stella-1.5B, 22.3 for
   Qwen3-0.6B on this box, ~69 for the pair, against a whole-screen budget of ~6. So the teacher
   contrast alone runs on a 3-component family-weighted surface; **student, prompt and mix keep all
   six**, exactly as the mandate says. Made before any number was observed and written down with
   the arithmetic (`m9/LEDGER.md` §0). If a challenger were to win, the run **stops** and comes back
   to you rather than proxying.
2. **FineWeb: approved by you, not exercised.** The mandate allows it only against *pre-existing*
   reserved-set **document** fingerprints. There are none — M7 persisted query-side fingerprints
   and R3 *counts* only, and the document index it streamed was discarded. Building one now would
   open reserved corpora, which the mandate forbids. FineWeb is out for M9; doc-side text comes
   from the 6.15M pre-screened M7 pool rows instead. Reopens in M10. (`m9/LEDGER.md` §1.3)

## What M9.0 locked

242,786 non-FEVER query texts · 6,149,679 eligible document rows · LEAF plain-L2 regression, mean
pooling, `Linear(hidden, 1024)` · 16 epochs = 3,884,576 examples = 30,349 steps = 59,507,872 non-pad
tokens · two surfaces (DEV-6 equal weight decides student/prompt/mix; family-weighted SCREEN-3
decides the teacher) · **MDE = 0.0056**, one number derived from 2,031 historical dev contrasts ·
the head **warm-started in closed form** for every arm, with λ chosen on a training-only holdout. `nqopen`/`triviaqa` were excluded at M9.0 and later **admitted** for the build by
`m9src/extended_screen.py` (220,528 real questions); the screen itself still runs on the
242,786-text pool it locked. Full text: `m9/LEDGER.md`; machine copy: `m9/registry.json`.

## Reviews — three adversarial passes, all before any arm ran

| pass | target | verdict | disposition |
|---|---|---|---|
| 1 | the first draft | **DO NOT COMMIT** — 7 BLOCKER / 8 MAJOR / 4 MINOR + a post-number-freedom table | `m9/LEDGER.md` §10 |
| 2 | the amended lock **and the code** | **DO NOT COMMIT. DO NOT SPEND THE 6 GPU-HOURS** — the v1 fixes had moved failures out of the prose and into the implementation | `m9/LEDGER.md` §11 |
| 3 | v3, the warm start and the adequacy gate | **v3 is broken; do not let `m9s1` open stage B** — it caught a **false statement in the lock**: the warm-start ridge λ was described as chosen on the training residual and had in fact been chosen on SCREEN-3, a dev surface | `m9/LEDGER.md` §12 |

The third pass is why the first anchor run was **killed at 11,000 of 30,349 steps and thrown
away**: it had been trained with a dev-selected λ, so nothing it produced could be preregistered
evidence. λ selection moved to a training-only holdout under the real normalized objective and the
anchor was re-run from scratch. That is the cost of the standing directive working as intended —
one wasted GPU-hour instead of a milestone built on a number chosen after seeing dev.

What the reviews actually changed: the mandate-surface conflict; two arms that confounded their own
factor with token dose; a guard that let a session amend the lock after seeing a number; a decision
threshold whose effective value was unstated; a "noise floor" built from two seeds; a batch pilot
that would have measured update count; a sorted document sample that was store-biased; and a
deterministic crash in the mandatory port pilot.

## Measured so far (no decisions read these yet)

| quantity | value |
|---|---|
| stella-400M symmetric ceiling, DEV-6 | **0.6724** |
| stella-400M symmetric ceiling, SCREEN-3 (family weights) | **0.6822** |
| student throughput, bge-small @ bs128 | ~1,990 ex/s (real texts) |
| teacher encode, stella-400M / Qwen3-0.6B | 2,076 / 1,152 q/s · 210 / 146 doc/s |
| closed-form head on a **frozen** backbone (diagnostic) | **0.3463** SCREEN-3 = **50.8%** of the ceiling |
| the same backbone, random head, 2,000 trained steps | 12.4% |
| fp16 target gate | PASS — min-cos 0.999959, max-abs 2.0e-4 |
| bridge-tolerance dry run | PASS — zero qid drift, max \|Δ nDCG@10\| **0.0** across processes |
| ONNX export, both students | PASS — min-cos 0.9999993, opset 17, **zero custom-domain ops** |
| **anchor `m9s1`, final (16 epochs)** | **SCREEN-3 0.4998 = 73.3% · DEV-6 0.4806 = 71.5%** |
| anchor curve, SCREEN-3 at 4/8/12/16 epochs | 0.4481 → 0.4812 → 0.4944 → 0.4998 |
| adequacy gate | **PASS** (0.7326 ≥ 0.60; late slope 0.0054 ≤ 0.02) → stage B authorised |
| shipped fp16 artifact, bge-small nano / MiniLM nano | **68.5 MB / 47.0 MB** — inside the 70 MB target |
| fastembed `add_custom_model` | accepted and listed |
| fp16 ONNX parity | **0.99953 vs a locked 0.9999 — MISS**, recorded rather than re-thresholded |

The head-probe rows are the session's most consequential finding so far, and they changed the
recipe: at ~1% of LEAF's dose a randomly-initialized projection head spends a large share of the
whole budget re-deriving a linear map that has a closed form. Every arm now warm-starts it, and arm
`m9s1c` repeats the anchor without it to price exactly what that is worth.

**What the anchor curve says, and it is the point of stage A.** Quarter-on-quarter gains are
**+0.0330, +0.0132, +0.0054** — each roughly half the last. Sixteen epochs is close to what 242,786
unique queries yield, so the remaining ~27% of the ceiling is not behind more SGD on this data. The
levers that could reach it are more unique text (the document pool has 6.15M pre-screened rows, and
the mix arm prices exactly that), a better student, or a co-adapted document side — which is what
`m8/FINDINGS.md` already named as the one untested high-capacity lever.

The honest headline risk is unchanged: **LEAF's published 97.9% retention came from ~100 A100-hours
and 6.7M unique texts; M9's affordable dose is ~1% of that on 243K unique queries.** The
four-checkpoint curve is the registered instrument for saying what retention this budget buys, and
it is the main thing stage A exists to produce.

## Re-runs, and why they were cheap

Four anchor attempts, and every one of them was killed for a reason worth the GPU-hour:

| # | killed because | fix |
|---|---|---|
| 1 | Codex pass 3 found the warm-start λ had been chosen on SCREEN-3, a dev surface | λ moved to a training-only holdout under the real objective |
| 2 | training collapsed 1,990 → 786 ex/s: allocator fragmentation from interleaved 23 GB evaluations | `expandable_segments`, and DEV-6 read once at the end |
| 3 | the session manifest keyed on HEAD, so committing an arm's own result would have voided it | keyed on the fingerprint instead |
| 4 | the challenger-teacher path had three defects the incumbent arms could never reach | all three fixed, path smoked before spending |

The runs are deterministic — checkpoint 1 reproduced to six decimals across attempts — so each
re-run cost time and nothing else. That is the point of the guard: it made every one of these
*visible* rather than letting a milestone rest on a number chosen after seeing dev, a truncated
arm, or arms from three different code states.

## Deferred to M10 / the whitepaper

- **TurboQuant (int4) benchmarks** — Dylan's ruling, 2026-08-30: it is Qdrant's preferred method
  and belongs in the whitepaper's all-in comparison against binary, int8 and fp16, on latency,
  footprint *and* recall. M9 measured enough to establish the shape (quantization is a
  precondition, not an optimisation). 1M documents is the confirmed upper bound.
- **fp16 ONNX parity** missed its locked 0.9999 threshold at 0.99953 — recorded as a fail rather
  than re-thresholded; the right follow-up is a preregistered retrieval-impact tolerance.

## HANDOFF — read this first if you are a fresh session

**All nine Codex #7 launch blockers are fixed** (commits `198e9c8` +
the follow-up actioning Codex #8, `research/` has both reviews). Codex #8 re-reviewed the first
five fixes and found five more blockers — mix verdict ignored, first-eval gate impossible, stale
corpora blessable, decisions outliving their arms, cooldown not durable — all actioned the same
day. Key behaviors a fresh session must know:

- `make_config` **fails closed**: no decision file, missing fields, a challenger teacher, or a
  `query-only` mix verdict each refuse to build (the last one goes back to Dylan — M92_LOCK §4).
- `prepare` takes `--student` and `--prompt-policy` and skips corpora already tokenized under the
  same recipe (identity-bound: student, prefix, source hash). Meta was verified by sample
  re-tokenization and backfilled 2026-08-30.
- Plateau and the stable token cap **enter the cooldown automatically** and checkpoint at once;
  under the watchdog the trainer also anneals before the wall-clock horizon
  (`--anneal-before-deadline`).
- The build has its own guard session (`SESSION-build.json`); session mismatch is scoped to the
  run's dependency scopes.
- The watchdog restarts on checkpoint-stale/eval-overdue (judged over observed time), survives
  its own exceptions, and gives evals a 1 h heartbeat grace.

### Launch sequence
```bash
cd /home/dylan/asymetric-dual-encoders            # branch m9-work
./m9status.sh && .venv/bin/python m9src/guard9.py   # expect problems: []

# 1. the screen (~2.5 h). WAIT for it -- do not background and continue.
rm -f work/m9tokens/*.json results/m9_screen_m9s*.json results/m9_adequacy.json \
      results/m9_screen_state.json results/m9_screen_decisions.json
.venv/bin/python -c "import sys;sys.path[:0]=['m9src'];import guard9;guard9.open_session(force=True)"
./run_m9_stage.sh gate:warmfit gate:fp16_gate gate:bridge_dryrun:verify \
    m9s1 adequacy m9s1c m9s1b m9s4 decide m9s5 decide m9s6 decide
test -s results/m9_screen_decisions.json || { echo "no FINAL decision -- stop"; exit 1; }

# 2. build prerequisites (~40 min; longer if the screen picked MiniLM or policy (a) --
#    prepare then re-tokenizes what changed). Each must succeed before the next.
#    STUDENT and POLICY come from results/m9_screen_decisions.json "selected".
.venv/bin/python m9src/longrun.py prepare --student <SELECTED> --prompt-policy <SELECTED> && \
.venv/bin/python m9src/longrun.py targets && \
.venv/bin/python m9src/longrun.py manifest && \
.venv/bin/python m9src/longrun.py verify && \
.venv/bin/python m9src/make_config.py && \
.venv/bin/python m9src/test_resume.py            # MUST pass; never skip
rm -f work/m9long/terminal.json work/m9long/ckpt/STOP work/m9long/trainer.lock \
      work/m9tokens/SESSION-build.json work/m9tokens/m9-build.json

# 3. fill m9/M92_LOCK.md from the decision (it is still DRAFT with blank fields), commit, push,
#    THEN launch (the watchdog starts the trainer; the trainer opens the build session)
setsid nohup .venv/bin/python m9src/watchdog.py --hours 168 > logs/m9_watchdog.log 2>&1 &
```

> **WARNING, learned the expensive way.** `m9src/guard9.py` sits inside the `protocol` scope it
> enforces, and `warmfit.py`/`nano.py`/`screen.py` sit inside `train`. **Editing anything under
> `m9src/` while an arm runs voids that arm at write time** — it cost five completed arms today.
> `m9/LEDGER.md`, `m9/registry.json` and `results/m9_lock_constants.json` are guarded too. Batch
> every guarded-file edit into a window when nothing is running.

## Where the seven-day build stands

Dylan is available until ~00:30 tonight and away for three days after, so the build launches
tonight and runs unattended. Everything for it is written, reviewed and committed:

| piece | state |
|---|---|
| `m9src/longrun.py` | resumable, stoppable, guarded trainer. Rewritten after Codex #5 returned DO NOT LAUNCH on seven blockers |
| `m9src/watchdog.py` | out-of-process timer. Hardened after Codex #6 returned DO NOT LAUNCH UNATTENDED on six more |
| corpora | tokenized and hashed: **7,536,401 texts, 627 M tokens/epoch** (documents 581 M) |
| dose | **5% queries / 5% spans / 90% documents** by token — 109.6 query epochs, not 438 |
| schedule | warmup → stable → **decay on demand**, cooldown 59.5 M tokens (the anchor's whole dose) |
| kill envelope | non-finite, regression, plateau, throughput collapse, first-eval vs step-0 baseline |
| teacher | **stella-400M, not screened** — `m9s2` (stella-1.5B) was behind the anchor at every checkpoint (final −0.00229 against a +0.010 bar), so the teacher arms were withdrawn on measurement and on the owner's product preference: one document model, one collection, shared by `zero` and `nano` |
| remaining | the three recipe arms · teacher targets for 1.14 M new texts · manifest · **resume-equivalence test on the real path** · fill the M9.2 lock from `decide` · launch |

`m9/M92_LOCK.md` is the recipe lock; `m9/RUN_STATUS.md` will be published on branch `m9-status`
every 30 minutes so the run is legible from anywhere while nobody is at the machine.

## Files

| file | contract |
|---|---|
| `LEDGER.md` | the M9.0 lock: protocol, rulings, and every number a rule reads |
| `registry.json` | the machine copy of those constants |
| `RESULTS.md` | runs, in order |
| `EXPLORED.md` | closed avenues, each with what would reopen it |
| `M92_LOCK.md` | the recipe lock for the seven-day build |
| `RUN_STATUS.md` | live build status, republished on branch `m9-status` |
| `PLANNING.md` · `BRIEF.md` | the pre-M9.0 evidence and context |
| `EDGE_COST_MAC.md` · `EDGE_PROTOTYPE_MAC.md` | task cards for the second machine |
| `CODEMAP.md` | modules and the pitfalls this milestone earned |
