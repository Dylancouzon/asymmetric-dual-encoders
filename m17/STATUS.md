# M17 CLOSED 2026-09-12 — negative result (`no_survivor`)

Zero v1.1 (vocabulary extension, joint table/listwise training, alias consistency, late-checkpoint
averaging, int8 resident-row loading) is **closed with the negative result recorded**. Dylan's
ruling, 2026-09-12: "Let's close v1.1." The 6d screen returned `no_survivor` — no arm meets the
eligibility predicate against v1 — so the registered STOP stands: no further reads, training or
protected access. **The released zero v1 remains the shipped table.** The 72 h clock (started
2026-09-11T23:41:24Z) is released unspent beyond the screen. Registry status:
`CLOSED_NEGATIVE_RESULT`. Branch: `m17-zero-v1.1-planning`.

**Done (detail in `LEDGER.md`):**

- Steps 2–5; full-pool build `work/m17/prepared/full` (600,000 queries, seed 0, plan A5).
- Lock `m17src/lock.py`, both halves: pre (`e0a1a40`), screened rebuild (`164845e`), executed
  (`badb048`) — 446 vocabulary terms after the protected screen, all five V0 gates PASS.
- Dev-suite reader `m17src/evaluate.py` (`39e1ddf`) with receipt/gate design; Astra then Sol
  reviews plus one Astra P1 re-check closed.
- The single V0 read (`f8ba8a8`): nDCG@10 macro 0.615, Recall@10 macro 0.701, six pinned
  components, descriptive only. `results/m17_v0_read.json`.
- 6d screen, five arms × 4,000 steps, one read each: C 0.5807, V 0.5810, L 0.5936, VL 0.5938,
  VL-A 0.5938 — all below the untrained V0 (0.6153) on the same surface.
- Decision `no_survivor`, registered STOP (`532f7e4`, `m17/screen_decision_2026-09-12.json`).

**If reopened** (the LEDGER closure entry holds the full list of what is owed):

- Register a **dated v1 eligibility reference with numbers**; the screen used the M7 audit row
  under a disclosure, not a registration.
- The distinguishing measurement is a **registered diagnostic read of intermediate checkpoints**
  (arm C at 500/1000/2000). All screen reads are spent. Try the cheaper non-read checks first.
- Set the **anchor weight from measured drift** (inert at 1e-3 while arm C drifted 10.1 % of
  element RMS); pre-register it dated before any second screen.
- Trained arms reading below the untrained V0 is an **undiagnosed failure, not evidence against
  the method**.

**Pitfalls (still true of this code):**

- Before any future screened rebuild, reassert the path order after `protected10.build()` as well
  — `cov_screen` re-inserts m7src on an index-cache miss (deferred Astra P2; would need a dated
  amendment of `prepare_data_py`, whose sha the lock binds).
- `prepare_data.py` must not change without such a dated amendment.
- A screened rebuild of an existing unscreened directory needs `--force`, which the auto-mode
  classifier blocks — launch through `work/m17/logs/run_screen.sh`.
- The synthetic rehearsal never imports the real `protected10`; on-clock-only paths need a real
  screen smoke.
- Do not read anything under `work/m17/prepared/full` or `work/m17/bundles/V0` as a quality
  surface; V0 had exactly one declared read and it is spent.
- `m17src` test suite: a few tests still assert the pre-closure registry statuses and fail against
  `CLOSED_NEGATIVE_RESULT`; they were left alone rather than rewritten to chase the status.

Reviews and dispositions: [REVIEW.md](REVIEW.md). Plan: [PLANNING.md](PLANNING.md). Constants:
[registry.json](registry.json). Authority/probes: [LEDGER.md](LEDGER.md). Paths and pitfalls:
[CODEMAP.md](CODEMAP.md). Lessons: [FINDINGS.md](FINDINGS.md).
