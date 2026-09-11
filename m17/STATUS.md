# M17 status — pre-clock execution, checkpoint 4 (before the lock and clock start), 2026-09-11

**Planned:** a supported vocabulary extension, joint table/listwise training, and the owner-approved
alias consistency, late-checkpoint averaging and int8 resident-row loading comparisons. Same query
API and frozen document index; Nano/M13 stays unchanged. Branch: `m17-zero-v1.1-planning`.

**Done (2026-09-11):** steps 2–5 (see `LEDGER.md`), and step 6's preparation:

- **Full-pool build complete:** `work/m17/prepared/full`, 600,000 queries, seed 0, plan exactly A5.
  Measured **1.496 h** across two invocations (attempt 1 died at 120k teacher texts on a transient
  CUDA fault; the teacher cache now flushes in durable 25k chunks; attempt 2 resumed and finished
  without a fault). RSS high-water 12.1 GiB, anonymous peak 9.5 GiB, GPU 3.0 GiB. Vocabulary:
  445 terms selected of 210,447 candidates (support minima, not caps, bound it; `FINDINGS.md`).
  Protected screen deferred to the clock (629 pool rows, 1,648 bank documents unscreened).
  No quality surface read.
- **Lock code:** `m17src/lock.py`, two dated halves. `--phase pre` (pre-clock) binds
  protocol/recipe/seeds/fixed manifests/invariant inputs and the measured allocation, sets
  `EXECUTABLE` (preparation only). `--phase executed` (on the clock, after the screened rebuild,
  BEFORE the V0 read) binds the executed identities for both forms, exports V0 through the five
  gates, sets `LOCKED_EXECUTABLE`; real training refuses anything less and checks the loaded
  directory against it. Why two halves: the protected flag is part of every stage identity, so the
  screened rebuild re-derives vocabulary, tokenizer and cache — the unscreened build only prices them.
- **Reviews closed:** Codex Astra (11 findings → fixes) then Codex Sol (all confirmed, nothing
  remaining); dispositions in `REVIEW.md`. 306 `m17src` tests pass.
- **Pre half dry-run against the full build passes:** measured 1.496 h → preparation phase
  ceiling 3 h (placeholder was 16), allocation total 59 h, 13 h unallocated and recorded; panel
  `d1b017a8…`; registry untouched (`DRAFT_NOT_EXECUTABLE`).

**Dylan owes:**

1. **Clock-start ruling (Astra P1-8, blocks the clock, not the pre half):** which invocation
   starts the 72 h and what pre-clock work is exempt. Record it in `LEDGER.md`; the code stores
   `clock_started` as given.
2. Go/no-go on committing the pre half and starting the clock (this checkpoint).
3. Optional: the human double-check slice `results/m17_human_doublecheck_slice.jsonl`.

**Next (in this order), after Dylan's go:**

6a. Commit the pre half:
    `.venv/bin/python m17src/lock.py --phase pre --build work/m17/prepared/full --prior-seconds 146.211 --prior-stages pool,domain,protected --prior-source "work/m17/logs/prepare_full_crash1.log stage lines (attempt 1, 2026-09-11: pool 24.3 s, domain 121.9 s, protected deferred 0 s)"`
    (drop `--dry-run`), commit "M17 lock: pre half", push. Registry → `EXECUTABLE`.
6b. Start the clock under the ruling. Detached, monitored:
    `.venv/bin/python m17src/prepare_data.py --out work/m17/prepared/full --protected-screen`
    (rebuilds teacher→manifests after the screen; caches warm; budget ≤ 3 h for the phase).
6c. Commit the executed half:
    `.venv/bin/python m17src/lock.py --phase executed --build work/m17/prepared/full --v0-out work/m17/bundles/V0 --clock-started <ISO>`
    → `LOCKED_EXECUTABLE`; commit, push. Only then read V0 (one declared read).
6d. Screen arms (C, V, L, VL, VL-A at 4,000 steps, seed 0) per `training.decision_protocol`.

**Working model for every M17 session (Dylan, 2026-09-11):** Fable orchestrates; Opus subagents
do execution; at most two subagents run concurrently; subagents never spawn subagents. Codex
Astra and Sol are the reviewers and run **alternately** (review, fix, next reviewer), never in
parallel on one brief; Codex Astra is also the registered judge for descriptive sheets (A6).
Commit and push after every coherent batch. Do not over-engineer. Plan a context clear at each
checkpoint: update this file first, push, then Dylan clears. Next checkpoint: after the executed
half is committed and before the screen arms start.

**Pitfalls for the next session:** `prepare_data.py` must not change between the pre and
executed halves (its source hash is an invariant input; a change is a dated amendment). Watch the
screened rebuild's log for `Traceback|Error|FAILED|OOM|Killed`; if it dies, resume with the same
command. Do not read anything under `work/m17/prepared/full` as a quality surface.

Reviews and dispositions: [REVIEW.md](REVIEW.md). Plan: [PLANNING.md](PLANNING.md). Constants:
[registry.json](registry.json). Authority/probes: [LEDGER.md](LEDGER.md). Paths and pitfalls:
[CODEMAP.md](CODEMAP.md). Lessons: [FINDINGS.md](FINDINGS.md).
