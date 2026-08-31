# M9.2 — seven-day build recipe (LOCKED 2026-08-30)

Status: **LOCKED.** The complete eligible screen selects stella-400M-v5 × bge-small-en-v1.5 ×
bare-query prompt (b). `m9s6` selected query-only (−0.0060 DEV-6); Dylan's artifact-bound ruling
authorizes the registered 5/5/90 build because documents were indistinguishable from neutral at
that dose and query-only had already flattened. This build's SCREEN-3 curve is the test.

## Recipe and data

Token shares: **5% real queries / 5% short document spans / 90% documents**. The loss is the plain
mean over examples; token shares only set sampling. The realized step is 113 examples: 13
`queries_pair`, 5 `nqopen`, 7 `triviaqa`, 10 `pseudoq`, 78 `documents`.

| corpus | texts | tokens/epoch | role |
|---|---:|---:|---|
| `queries_pair` | 242,786 | 3,719,242 | query |
| `nqopen` | 85,863 | 1,008,102 | query |
| `triviaqa` | 134,665 | 2,685,360 | query |
| `pseudoq` | 923,408 | 38,220,549 | short document spans, not questions |
| `documents` | 6,149,679 | **581,469,041** | document |

All bytes are bound by `work/m9long/manifest.json` and recomputed by `longrun.py verify`. All
sources are decontaminated against the six, dev, untouched-final, LoTTE shadow and M9 reserve. MS
MARCO and FEVER stay out.

## Dose and schedule

The representative measurement is mixed arm `m9s6`: **59,507,877 / 3,134.7 = 18,983.6 tok/s**,
rounded to **18,984 tok/s**. The old 26,854 tok/s query-only rate is not used.

- Seven-day budget: `18,984 × 168 × 3,600 = 11,481,523,200` tokens; 1,401,553 whole steps.
- Tokens/step: **8,192**. At the horizon the source epochs are 77.4 real queries, 15.0 spans,
  17.8 documents.
- Warmup: 2,000 steps linear to 1e-4; stable at 1e-4; AdamW β=(0.9, 0.999), eps 1e-8,
  weight decay 0.01 on dim>1, grad clip 1.0, bf16 autocast with fp32 loss.
- Cooldown target: 59,507,877 tokens. Whole-step schedule:
  `ceil(59,507,877 / 8,192) = 7,265 steps`; `7,265 × 8,192 = 59,514,880` scheduled tokens
  (+7,003). Stable cap is therefore `11,481,523,200 − 59,514,880 = 11,422,008,320` tokens.
  Cooldown is cosine to 1e-5, resumable, and based on the anchor arm's measured full dose.
- Eval: every **15,000 steps = 122,880,000 tokens**: 1.798 h at 18,984 tok/s and 3.596 h at
  the permitted 50% floor. There are 93 evals at the nominal horizon.
- Checkpoint: every **3,000 steps = 24,576,000 tokens**: 0.360 h nominal and 0.719 h at the floor.
  Writes are atomic. Eval checkpoints persist the best including that same evaluation.
- Watchdog: `--eval-stale 18,000` (5 h), `--ckpt-stale 7,200` (2 h). Both exceed the healthy
  floor intervals, including practical I/O margin, so floor-rate training cannot self-restart.

## Stops and supervision

- Non-finite loss/gradient stops. Two consecutive evals >0.0056 below best stop.
- Plateau <+0.001 over 1B tokens enters cooldown. With 122.88M-token eval spacing, the comparison
  spans nine intervals = **1.10592B tokens**, so the rule remains token-denominated and actionable.
- Throughput baseline is persisted once as
  `max(median(rolling rates from minutes 15–30), 18,984 measured mixed tok/s)`; the floor is 50%.
  A cold-start-slow process cannot bless its own degradation.
- Plateau and stable-cap stops enter cooldown automatically. Regression, non-finite, operator STOP,
  and watchdog give-up write terminal state and never restart silently.
- Watchdog and trainer share one persisted absolute `deadline.json`. The watchdog remains alive
  until the trainer writes terminal state, including after the deadline. Every give-up requests
  STOP, escalates if needed, records exactly which signals it sent, and writes a terminal marker.

## Operator procedure (identical in `STATUS.md` and generated `RUN_STATUS.md`)

1. Stop safely: `touch work/m9long/ckpt/STOP`. Keep the watchdog running;
   it supervises until `terminal.json` confirms the trainer exited.
2. Cool down: after that terminal marker appears, run
   `setsid nohup .venv/bin/python m9src/watchdog.py --cooldown --hours 4 >> logs/m9_watchdog.log 2>&1 &`.
   The cooldown command safely consumes the acknowledged STOP and terminal markers, resumes
   `last.pt` in decay, and supervises it through `cooldown complete`.
3. Restart after a crash: if the watchdog is alive, do nothing; it restarts the trainer exactly.
   If the watchdog died, rerun the original watchdog launch command. It reuses `deadline.json`,
   attaches to a live trainer or resumes `last.pt`, and never resets the seven-day horizon.

## Boundaries and launch checklist

One candidate only. No six, reserved four, LoTTE, or confirmatory access during the build.
`m9s1b`/`m9s1c` are withdrawn and must never run. Never force/re-open `SESSION.json`; never delete
screen artifacts or their tokens. M9.4 owns confirmatory evaluation.

- [x] Complete eligible decision and all four required arm artifacts verified
- [x] Corpora and teacher targets prepared; manifest recomputes cleanly
- [x] Config generated from the assembled decision and this arithmetic
- [x] Resume equivalence tested
- [x] Pre-launch adversarial review actioned
- [x] Historical traceback log rotated out of the active failure-signature log
- [ ] Owner final action: commit and push these requested repairs, then require `guard9.py` to print
      `problems: []` before launch. This agent was explicitly forbidden to commit or push.
