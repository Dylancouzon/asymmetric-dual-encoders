# M8 code map

Read `STATUS.md` first, then `LEDGER.md`. This file exists so a future session can resume without
reading every module. Nothing here restates `LEDGER.md` (protocol) or `registry.json` (bars).

## Layout

- `m8src/` — all M8 code. **`m7src/` is frozen**: M8 imports from it and never edits it (G3).
  `bench/` and `scripts/` are the shared M1–M6 harness; do not fork them.
- `work/` — gitignored heavy data. New tonight: `work/lotte/` (shadow, protected),
  `work/m9reserve/` (M9 inventories, protected), `work/decontam/` (the filter's output).
- `results/m8_*.json` — every committed M8 artifact. Each carries a `_registration` block naming
  the registry sha it ran under.

## Modules

**Foundation** — `m8base.py` paths, the reserved four, the six, the registered dev groups; importing
it installs the protected-path guard, so a module cannot opt out by forgetting ·
`paths_guard.py` **G2**: the allowlist over every route to a protected partition, with the
justification for each entry written next to it · `probe_guard.py` **G1**: reads
`m8/registry.json` at the current commit, refuses incomplete or `TBD` rows, and `write_result()`
stamps the registry sha into the artifact so gating does not depend on a caller remembering to ask.

**Decision machinery** — `decide.py` the confirmatory rule and the complete ship predicate, every
threshold a literal; `self_test()` runs it end to end on synthetic data · `power.py` the joint
power simulation → `results/m8_power.json` (macro SE, MDE, P(ship) by scenario) ·
`noise_floor.py` the true-seed-null floor every bar is set against.

**Instruments** — `blockcg.py` the Gram-free ridge solve (B7): the thing that decides whether a
64–128K vocabulary and any non-WordPiece teacher screen are computable here at all ·
`protected_filter.py` S0's overlap screen and the query-only fingerprint inventory (the ONE module
that may read protected query text) · `retention_decomp.py` the descriptive re-read of M7's final
run that reframed H3 · `bench.py` throughput, the ridge control timing, and the serial schedule.

**Tests** — `test_guards.py`, 26 checks, both halves of G2 (runtime refusals and a static scan) plus
G1. Run it after touching either guard.

## Things that must move with the encoder

`m7/CODEMAP.md` names two: anything assuming **dim 768**, and `table.py`'s **`CLS_ID`**. Screening
a non-BERT teacher found two more, and both are silent:

3. **`table.py`'s `CLS_ID = 101`** is a module default but a constructor PARAMETER — pass
   `spec.cls_id`. ModernBERT's is **50281**. The default would put a wrong vector behind every
   degenerate empty query, and nothing would raise.
4. **Anything assuming `tok.vocab_size == len(tok)`.** `m7src/init_table` sizes the teacher init
   by `tok.vocab_size`. For all ten registry encoders that equals `len(tok)` equals 30,522. For
   ModernBERT it is **50,280 against 50,368**, and **`[CLS]` is 50,281 — inside the gap**, so the
   init has no row for the token `add_special_tokens=True` puts at the front of every query.
   `m8src/init_m8.py` builds it at `max(len(tok), tok.vocab_size)` and falls through to m7src
   whenever the two agree. Note the two published figures disagree too: `tok.vocab_size` says
   50,280 and `len(tok)`/`config.vocab_size` say 50,368 — **measure it, do not read it off a card**.

## Pitfalls this milestone earned

1. **An `m8src` module name that collides with an `m7src` one shadows it.** `m8src/_paths.py` sat
   ahead of `m7src/_paths.py` on `sys.path` and broke every m7src import. Renamed `m8base.py`.
   Any new shared-sounding name (`table.py`, `boot.py`, `train.py`) has the same hazard.
2. **A protected partition is defined by its CONTENT, not by where one copy lives.** The reserved
   android/english qrels existed at `work/dev/`, in the HF cache, and one `load_dataset` call away.
   Guarding `results/frozen_eval/untouched-*` alone was guarding one door of four.
3. **A progress line's rate must be that slice's rate.** The first S0 progress print divided a
   per-slice counter by a global elapsed time and reported 407/s for a 4,100/s job. The whole
   long-run discipline rests on that number being readable.
4. **A CG tolerance must follow its dtype.** Asking fp32 for 1e-8 burns every iteration against
   machine epsilon and never converges. fp32 + 1e-6 is already four orders tighter than the int8
   quantization it feeds.
5. **Never build a Gram in a Python loop.** `g[np.ix_(r, r)] += 1` over 200K rows took longer than
   the solve it was timing. One sparse matmul.
6. **An adversarial-review brief written in security vocabulary gets content-filtered.** A Codex
   pass phrased around "exploit / bypass / attack the guard" was cut off mid-read by the provider.
   The same questions in protocol language ("does this leave a degree of freedom open?") ran fine.
   Cost one full review cycle.
7. **A `pgrep`/`pkill` pattern matches the waiting shell's OWN command line.** `pkill` from such a
   shell kills itself (exit 144); an `until ! pgrep -f "x"` loop never exits because it always
   finds itself. Hit three times in one session. Two fixes, both cheap: kill by pid
   (`kill $(pgrep -f ...)` in a *separate* command from any relaunch), and write the wait pattern
   so it cannot match itself — `pgrep -f "noise_floor[.]py train"` matches the real process while
   the waiting shell's literal `noise_floor[.]py train` does not match that regex.
8. **Running a frozen `m7src` script can OVERWRITE an M7 artifact.** `validate_encoder.py` writes
   `results/m7_encoder_validation.json` unconditionally, so validating an M8 challenger Spec
   silently replaced M7's own validation record. Caught by `git status`, restored with
   `git checkout --`, and the M8 result kept as `results/m8_encoder_validation.json`.
   **`sweep.one` is worse**: it APPENDS every run to `m7/RESULTS.md`, M7's experiment ledger, so
   fifteen M8 arm rows landed in it. Reverted, and the rows preserved in `m8/RESULTS.md` instead.
   (It also writes `results/m7_run_<id>.json`, which is harmless — new files — and explains the
   `m7_` prefix on M8 run records.)
   **Check `git status m7/ results/` after running ANYTHING out of `m7src/`** — G3 protects M8
   from editing m7src's CODE, not from its scripts' side effects, and three of them write into
   M7's record without being asked.
9. **`M7_ENCODER` defaults to bge-base — M7's PRE-SWAP teacher — and nothing warns you.**
   Every noise-floor arm died with "init 'run:p35b-2m' was trained against stella but the active
   encoder is bge-base". `m8base.py` now sets it, so every M8 process inherits the incumbent.
   The refusal itself is good design in `m7src` (it is what stops a mixed-teacher comparison); the
   defect was relying on an operator to export a variable.
10. **`sweep.one` catches its exception, records a FAILED row, and returns None.** A driver that
   only checks the subprocess return code sees exit 0 and marches on — mine ran all five arms
   after the first one failed. Any wrapper must turn `None` into a nonzero exit itself.
11. **A smoke must not occupy the real run id.** `noise_floor.train(smoke=True)` wrote 90-step
   artifacts to `m8nf-seed0` etc, so the next `plan()` would have seen `_exists=True` and the
   floor would have been measured on 90-step tables. Smokes get a `-smoke` suffix.
12. **Never restate a constant that lives in the registry.** `power.py` carried its own copies of
   the six-set margin and the worst-group shape; §5 was then given its measured values and the
   copies went stale within hours, so the P(ship) table handed to the owner was wrong by a factor
   that mattered. Read the registry; do not mirror it.
13. **A pooled within-dataset OLS is not automatically a within-dataset result.** It weights each
   dataset by that dataset's variance in x, and ArguAna carries **99.7%** of the six's
   within-dataset query-LENGTH variance (174-word queries against 2-12). The "pooled" length slope
   was ArguAna's slope — the exact one-dataset dependence the diagnostic existed to escape, on the
   one dataset that is also a disclosed teacher-training set. Always report the variance share per
   dataset and a leave-one-out slope before believing a pooled within-dataset number. (The
   fragmentation slope passed the same test: ArguAna holds only 2.2% of that variance.)
14. **Strip punctuation before tokenizing, not after.** Counting subwords on the raw whitespace
   token put ordinary words over a fragmentation threshold; fixing it flipped one dataset's
   contrast from +0.062 to −0.007 and turned a "6/6 sign-consistent" claim into 4/5, p=0.19.
15. **Synthetic data for a FEASIBILITY measurement must match the real distribution's hard
   property, not its easy one.** B7's first bag matrix drew token ids UNIFORMLY. Real text is
   Zipfian, and CG's cost is set by the condition number, so uniform draws are an unrealistically
   easy problem: unpreconditioned CG converged in 131 iterations on uniform data and **failed to
   converge in 1,500** on Zipfian data with the same shape and sparsity. A feasibility PASS from
   the uniform version would have been a wrong number, not a crash — the class that has cost this
   project the most. Jacobi preconditioning (the Gram's diagonal is just the column sum of X², one
   pass, no Gram) brings it to 61 iterations. Measure the preconditioner's effect; do not assume it.
16. **A run's `meta.json` records no code vintage, so nothing tells you when two runs were trained
   under different code.** The B-leg floor reuses M7's `p35b-2m` as its seed-0 arm; that checkpoint
   was written 2026-08-27 21:44 and **nine commits touched `m7src/` afterwards**, three of them on
   the training path. Only a hand diff of every hunk established that the arm is still a pure seed
   variant — the pseudoq change was docstring-only, `train.py`'s `side_pos_sources` defaults to
   `()` and takes the identical `index.get` branch, and the `teacher.py`/`table.py` additions are
   refusals on `encode_cached` layout and on `ensure_release`, neither of which a B leg reaches nor
   which alter a returned vector. **The direction of the risk is not symmetric**: an undetected code
   difference *inflates* a measured noise floor, which raises bars and is conservative — so this
   class of error hides behind a result that looks merely disappointing. Do the diff. Better,
   stamp the commit into `meta.json` on the next run that writes one, and never compare two
   checkpoints from different days without checking `git log --since` on the training path.
17. **A test that iterates a collection can assert nothing and still print PASS.**
   `test_guards.py`'s `probe_guard_refuses_bar_pending` looped over registry rows carrying
   `bar_pending` and asserted each one refuses. For most of the milestone **no row had that field**,
   so the loop body never executed and the check passed by testing nothing — while sitting in a
   suite whose whole job is to prove the guards refuse. It only surfaced when a row with the field
   was finally added and the test failed for an unrelated reason. The fix is not to add a row: it is
   to **synthesize the case** so the code path runs whatever the data happens to contain. Before
   trusting any `for x in <discovered set>: assert ...` test, ask what it does when the set is
   empty; if the answer is "passes", it is not a test yet. The same question applies to a monitor
   grep, and to the `grep -cE "^(FAIL"` I used to check this very suite — it returned 0 because the
   real FAIL lines are indented.
