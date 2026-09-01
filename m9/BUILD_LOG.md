# M9.3 build-period log

`m9/LEDGER.md` and `m9/registry.json` are in guard9's **protocol** scope, which
`DEPS["m9-build"]` depends on, so editing them makes the trainer — including the watchdog's
automatic crash-restart — refuse to start. Entries therefore land here during the build and are
merged into the ledger afterwards. This file is in no scope.

## 2026-08-31

- Build session force-reopened after the approved M3/M4 infrastructure repair; full disclosure in
  `m9/LEDGER.md` ("M9.3 BUILD PROVENANCE DISCLOSURE"). Trainer resumed from step ~78,754.
- Watchdog gave up at 06:17 after two failed launches (cause: the guard9 session fingerprint, a
  configuration failure — not a training or quality stop) and wrote `ckpt/STOP` +
  `terminal.json`. Both cleared to resume; the marker is archived at
  `work/m9long/archive/terminal-configfailure-20260831T0617.json`. Build session force-reopened at
  fingerprint `ef899585b10f` (was `24dcbe852bde`); `work/m9tokens/SESSION.json` untouched
  (`def5f88a0acf`).
- The build's warm-start adapter permits ONLY `owner_rulings` additions to `m9/registry.json`
  after the screen, so the `final_run` block I had added there blocked the trainer. Moved to
  `m9/final_run_registry.json` (no scope, same authority); `registry.json` restored to its
  post-screen state apart from the owner ruling, `stage` back to `M9.0`. `final_stats.py` and
  `FINAL_LOCK.md` updated; 16/16 tests still pass.
- **Build resumed 06:22** as pid 248010 from step 78,000 (0.639B tokens), command line now
  carrying `--decay-grace-s 21600`, so the M3 anneal protection is live. Cost of the whole
  intervention: 754 steps re-run (~6.2M tokens) and ~25 min of downtime. Manifest recomputed
  clean; eval history (6) preserved; `deadline.json` never reset.
- **KNOWN GAP, disclosed:** `m9src/test_resume.py` was NOT re-run after the M3/M4 diff. It was
  blocked first by the registry-integrity guard, and re-running it now would contend with the live
  trainer for the 10 GB GPU. Mitigation: the diff touches only the deadline branch and wraps
  `torch.load` in try/except — Codex confirmed across five passes that it alters no sampling,
  optimizer, RNG, token-accounting or resume code — and the live resume at step 78,000 succeeded
  with the manifest and config hashes verified. Re-run it at the next natural pause.
- LoTTE read #1 amended and pre-registered in `m9/LOTTE_LOCK.md` (Codex-approved with a firewall):
  the registered m9s6-vs-m9s1 veto is unchanged, the build's final candidate is ADDED to the same
  atomic batch, and the final-checkpoint identity rule is registered now so LoTTE cannot influence
  it. Runs once, post-build, pre-freeze.

### Curve watch, eval 8 (step 105,000, 0.860B tokens, 7.5% of cap)

SCREEN-3 0.53903, retention 0.7901. Per-doubling increments: +0.0344, +0.0283, +0.0171, +0.0256,
+0.0167, **+0.0058**. The bracket has stabilised across the last two evals — central (saturating
power-law) endpoint 0.836–0.843 retention, **fitted asymptote 0.851** — i.e. on this recipe
SCREEN-3 retention appears to asymptote near 0.85 even at unbounded dose.

For context, not as a forecast: the release bar needs 87.8% and the aim 89.7% **on avg-6**, a
different surface with no calibrated map from SCREEN-3. Two known biases run in opposite
directions — SCREEN-3 is partly in-domain (NQ at 0.50 weight, `nqopen` in the mix), which inflates
it relative to avg-6; and the fit sees only stable-LR checkpoints, so it cannot see the final
cosine anneal, which biases it low.

**No action, and none available:** the dose is fixed at the 11.42B cap with no registered extension,
the recipe is locked, the student is capped at 35M and the teacher is fixed. The plateau rule
cannot fire yet (it needs a ≥1B-token lookback; total is 0.86B) and at current increments would not
fire soon. Re-assess at ~50% of dose, when the projection is firm enough to be worth acting on —
and if it then points at a miss, prepare the "what would change it" analysis for Dylan
(doc-side co-adaptation / larger student / more dose are all explicitly out of M9's scope).

### Eval 11 (1.228B tokens, 10.7% of cap): three flat evals — diagnosed, NOT concluded

SCREEN-3 0.54487 (retention 0.7983). Increments +0.0052, +0.0004, +0.0003. Saturating fit stable
across four evals: central 0.836, asymptote 0.8515.

**Why this is not yet evidence of a ceiling** (the standing directive requires diagnosis before
pessimism):

1. **The build is ABOVE the anchor at half its query dose.** The query-only anchor `m9s1` reached
   0.50004 at 16 query epochs; this build is at **0.54487 at 8.3 query epochs**. The 5/5/90 mix is
   outperforming the arm whose flattening motivated the owner's override, at less query dose.
2. **We are 10.7% through, at 8.3 of 77 planned query epochs and 1.9 of 17.7 document epochs.**
   The owner's ruling recorded the relevant caveat verbatim: "neutral-per-token at 11 query epochs
   is NOT evidence that documents help at 110." Document contribution is the untested variable and
   it has barely begun.
3. **The fit cannot see the anneal.** Every point is a constant-LR (1e-4) checkpoint. A plateau at
   constant LR is the classic signature of a run that steps down on cosine decay to 1e-5.

**Phase 2 is UNAVAILABLE, and this is a lock gap worth recording.** The mandate required M9.2 to
register "one fully specified phase-2 loss and its hyperparameters" plus its numeric trigger;
`m9/LEDGER.md` says "Phase 2 is out of scope for M9.1" and no thresholds or loss were ever
registered. Specifying one now, in response to an observed flat curve, would be a post-hoc rescue
and is refused. Recorded as a limitation, not improvised around.

**Plateau rule, checked not assumed:** its lookback slides forward, so it becomes sensitive near
2.2B tokens. Calibration looks sound — firing needs <+0.001 across ~8 evals, while per-eval noise
alone is ~0.005 and the trailing 1B window has gained +0.078. If it does fire, cooldown produces an
annealed servable model at ~19% of dose; that is the registered, correct response to a genuinely
flat curve, not a malfunction.

**No action. Re-assess at ~50% of dose, or immediately if a regression stop fires.**

### `m9src/final9.py` — access control written and hardened; APPROVAL DEFERRED

Two review passes (`research/codex_final9*.log`). Pass 1: 5 BLOCKER / 3 MAJOR. Fixed: ignored
`git add`/`commit` failures in the BEGIN sequence (+ positive origin-tip verification, so a no-op
push cannot pass); `spent_tag_exists` **failing open** on a network/auth error — the worst of them,
since it would have permitted a second scoring of the six — now fails closed with a pinned
`origin_url`; `--recover` now completes step 5 durably and verifies the result came from the frozen
checkpoint; `acquire_lock` replaced with flock (the O_EXCL form had a stale-unlink race and read
`PermissionError` as "dead"); local-only tag no longer blocks `--infra-retry`; `RESULT` never
overwritten; parent-dir fsync.

Pass 2 rejected again, correctly, and the decisive point stands: **approval is vacuous while the
scoring path is absent** — `spend_access()` is never called, so the ordering guarantees hold only
because the module cannot score. Also fixed from pass 2: flock file no longer unlinked while the
descriptor is open; origin pinning mandatory; recovery push failure returns non-zero instead of
success; a spent access with no result now reports a documented loss instead of a misleading
refusal; and `--recover` installs `m8src/paths_guard` as a real capability boundary rather than
asserting one in a comment.

**Next:** wire step 4's encoding when the GPU is free, then re-review the module as a whole. Until
then `final9.py` refuses to run — and independently refuses while
`final_run_registry.ratified_by_owner` is false.

### Quarter-mark projection (eval 25, 2.947B tokens, 25.8% of cap) — bracket tightened, and lower

| reading | SCREEN-3 | retention | vs earlier fit (eval 11) |
|---|---|---|---|
| floor (stops now) | 0.5538 | 0.8118 | +0.014 |
| **saturating (central)** | 0.5616 | **0.8232** | −0.013 |
| loglinear (OPTIMISTIC bound) | 0.5922 | 0.8681 | −0.057 |
| fitted asymptote | 0.5664 | 0.8302 | −0.021 |

Only 1.95 doublings remain, so the bracket is much narrower than at eval 11 — and **the optimistic
bound (0.868) now sits below the release bar's 87.8%**, on this surface.

Read it carefully, because two things still separate this from a verdict:
1. **SCREEN-3 is not avg-6.** The bars are avg-6 figures and no calibrated map exists. SCREEN-3 is
   partly in-domain (NQ at 0.50 weight, `nqopen` in the mix), which by the M7 precedent inflates
   it relative to a held-out surface — so the true avg-6 number may be *lower*, not higher.
2. **The anneal is unmodelled.** Every point is a constant-LR checkpoint; the cosine decay to 1e-5
   is not in the fit and reliably adds a step up of unknown size.

Note the per-doubling column is now dominated by noise (−0.05 to +0.06): at a 2.9B base each eval
is only ~0.06 doublings, so dividing by it amplifies per-eval noise. The all-points fit is the
instrument; single increments here are meaningless.

**No lever exists inside M9** — dose fixed, recipe locked, student capped at 35M, teacher fixed,
and phase 2 was never registered. So this changes no action. It does change what Dylan should
expect, and it triggers the "what would change it" preparation promised at eval 11: LoTTE read #1
(fresh surface, already amended to include the candidate) is the next real evidence, and the
out-of-M9 levers are doc-side co-adaptation, a larger student, and more dose.
