# M17 status — on the clock, LOCKED_EXECUTABLE (executed half `badb048`), checkpoint before the V0 read

**Planned:** a supported vocabulary extension, joint table/listwise training, and the owner-approved
alias consistency, late-checkpoint averaging and int8 resident-row loading comparisons. Same query
API and frozen document index; Nano/M13 stays unchanged. Branch: `m17-zero-v1.1-planning`.

**Done (see `LEDGER.md` for detail):**

- Steps 2–5.
- Step 6 preparation: full-pool build `work/m17/prepared/full` (600,000 queries, seed 0, plan A5),
  1.496 h across two invocations; no quality surface read.
- Lock code `m17src/lock.py` with the two dated halves; Codex Astra then Sol reviews closed.
- 6a (2026-09-11): pre half committed as `e0a1a40`; registry `EXECUTABLE`.
- 6b (2026-09-12): screened rebuild complete (screen `complete` with receipt, 1,512 rows dropped
  and refilled to 600,000). An import-order crash at `parity` was fixed in `d8f2b43`/`bfa7644`,
  and `prepare_data.py`'s bound sha was amended by hand with a dated `lock.amendments` entry.
  Astra re-check verdict: proceed (`REVIEW.md`).
- 6c (2026-09-12): executed half committed.
  - `lock.py --phase executed` ran clean; all five V0 gates PASS (files, artifact, encoder_spec,
    tokenizer, conformance). Log: `work/m17/logs/lock_executed.log`.
  - 446 vocabulary terms after the protected screen (445 unscreened); V0 exported with
    `read: false`; registry status `LOCKED_EXECUTABLE`.
  - Committed and pushed as `badb048` ("M17 lock: executed half"). **V0 has not been read.**

**Dylan owes:** optional — the human double-check slice
`results/m17_human_doublecheck_slice.jsonl`. (Clock-start ruling and the go for the pre half are
both given; see `LEDGER.md`.)

**Next (in order):**

1. **Dev-suite reader** — wired in `2c5321c` (gated by `--allow-dev-suite` plus the executable
   registry; load-only real-path smoke matched all six pinned components; V0 unread). Astra review
   (2026-09-12, `work/m17/logs/astra_devreader_review.log`) returned six P1 / three P2 that block
   the read: unbound document vectors and query texts, V0 bundle digests not compared to the lock,
   gate accepting the pre half only, validation after scoring starts, no receipt / no-overwrite
   boundary, surface redefinable by arguments, Recall@10 missing, no success-path test.
   Fixed in `4824f84`. Sol review of the fix (`work/m17/logs/sol_devreader_fix_review.log`):
   seven P1 / three P2, read still blocked. Dropped one (HEAD moved because a brief was
   committed). **In progress:** Opus fix (production entry point without fixture overrides,
   canonical out path with an atomic receipt claim, per-component atomic persistence with an
   identity-checked resume, dirty-tree refusal, one pytrec_eval run for both metrics, pool
   identity parity, real cache-verification tests). Two items may need Dylan: whether the M7
   teacher cache's shard digests are trust-on-first-use (read would be refused under
   `verify=True`), and pinning ordered (qid, text) digests for the four text-backed components
   (a manifest amendment). Then at most one Astra P1 re-check, then the read.
   Astra P1 re-check done; two real items fixed, three dropped with dispositions (`REVIEW.md`).
   Dylan's rulings (2026-09-12, `LEDGER.md`): TOFU teacher caches accepted through the dated
   `m17/tofu_disclosure.json` gate (spot-check `results/m17_teacher_spotcheck.json`); text-component
   query texts recorded-only. Reader final at `39e1ddf`, 34 tests passing. **Ready to read.**
2. **The V0 read — DONE 2026-09-12** (`f8ba8a8`, registry read_record `c687268`). Descriptive
   only; numbers in `results/m17_v0_read.json` (nDCG@10 macro 0.615, Recall@10 macro 0.701 over
   the six pinned components). Not a bar, not a comparison. `reads: 1`; no further V0 read.
3. **6d screen arms — RUNNING since 2026-09-11T23:17 local** (`bash work/m17/logs/run_6d.sh`,
   code at `7292882`; screen-read entry point `screen_read` reviewed by Sol, dispositions in
   `REVIEW.md`). Sequential C, V, L, VL, VL-A at 4,000 steps, seed 0; per-arm logs
   `work/m17/logs/6d_<arm>.log`, launcher log `6d_launcher.log`; each arm exports its endpoint
   and takes its one registered read into `work/m17/runs/m17-<arm>-screen-s0/screen.json`.
   Smoke estimate ≈ 5–8 min per arm. Re-running the launcher resumes (flock-guarded; skips
   only arms with a complete receipt). Dylan added the launcher permission rule
   (`Bash(bash work/m17/logs/*.sh:*)`).
4. **After the screen:** apply `training.decision_protocol` (eligibility vs v1, `screen_ordering`
   walk VL-A→VL→{L,V}→C, `simple_arm_tie_band`); record the verdict in `LEDGER.md`. `no_survivor`
   is a registered STOP. Otherwise the registered full runs follow under the same launcher rule
   unless Dylan says "stop after the screen".

**Working model for every M17 session (Dylan, 2026-09-11):** Fable orchestrates; Opus subagents
do execution; at most two subagents run concurrently; subagents never spawn subagents. Codex
Astra and Sol are the reviewers and run **alternately** (review, fix, next reviewer), never in
parallel on one brief; Codex Astra is also the registered judge for descriptive sheets (A6).
Commit and push after every coherent batch. Do not over-engineer.

**Next session is OVERNIGHT and unsupervised (Dylan, 2026-09-12). Fable usage is very high:**

- Fable is the **orchestrator only**: it reads this file, writes briefs, dispatches, reads
  reports, decides, commits. It does not read source, write code, run tests or read logs itself;
  every one of those is an Opus subagent (execution) or a Codex run (review). One exception: a
  single-line shell check whose answer is needed to pick the next dispatch.
- Briefs name files, forbid recursive searches, carry the reserved read-exclusion and the
  "no quality surface under `work/m17/prepared/full` or `work/m17/bundles/V0`" line, and ask for
  a report under 20 lines. Reports, not transcripts, come back to Fable.
- Reviews: Astra on the dev-suite reader before the V0 read (irreversible); Sol on the fix; one
  Astra P1 re-check at most. Codex must not run while a memory-heavy build stage runs (a Codex
  run was OOM-killed beside the cache stage on 2026-09-11).
- **No context clear until Dylan is back**, whatever the checkpoint. Keep this file current at
  each step instead, and push, so a crash loses nothing.
- Decisions: registered branches proceed. Anything unregistered follows the unsupervised-window
  rule (brief a Fable subagent adversarially, decide, record in `LEDGER.md`, push, one-line iOS
  ping under 200 characters leading with the action). Never ping for progress. Dylan's hard lines
  (licences, teacher, bars, cap, protocol, release policy) wait for him.
- Long runs: `setsid nohup` through a launcher under `work/m17/logs/`, a Monitor on
  `Traceback|Error|FAILED|OOM|Killed|REFUSED` plus stage lines, first progress line and rate
  checked by a subagent, resume with the same command on a crash.
- Order of work is the **Next** list above. Stop the night at the first registered STOP or the
  first unresolved P1, with this file stating exactly where.

**Pitfalls for the next session:**

- **Deferred Astra P2:** before any future screened rebuild, reassert the path order after
  `protected10.build()` as well — `cov_screen` re-inserts m7src on an index-cache miss. It is a
  dated amendment of `prepare_data_py` (the lock binds that sha).
- `prepare_data.py` must not change without such a dated amendment; its source hash is a bound
  invariant input.
- A screened rebuild of an existing unscreened directory needs `--force`, and the auto-mode
  classifier blocks commands containing `--force` — launch through `work/m17/logs/run_screen.sh`.
- The synthetic rehearsal never imports the real `protected10`, so on-clock-only code paths need
  a real-screen smoke before they are trusted.
- Do not read anything under `work/m17/prepared/full` or `work/m17/bundles/V0` as a quality
  surface; V0 has exactly one declared read.

Reviews and dispositions: [REVIEW.md](REVIEW.md). Plan: [PLANNING.md](PLANNING.md). Constants:
[registry.json](registry.json). Authority/probes: [LEDGER.md](LEDGER.md). Paths and pitfalls:
[CODEMAP.md](CODEMAP.md). Lessons: [FINDINGS.md](FINDINGS.md).
