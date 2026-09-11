# M17 status — pre-clock execution, checkpoint 3 (after step 5), 2026-09-11

**Planned:** a supported vocabulary extension, joint table/listwise training, and the owner-approved
alias consistency, late-checkpoint averaging and int8 resident-row loading comparisons. Same query
API and frozen document index; Nano/M13 stays unchanged. Branch: `m17-zero-v1.1-planning`.

**Done (2026-09-11):** steps 2–5. Step 5: `m17src/prepare_data.py` builds the directories
`train.py --data` consumes (admitted pool under the A5 cap-fill rule, document→domain join,
deferred protected screen, cached teacher targets, 262,144-document bank, int8 v1 vectors,
old-vocabulary parity 9.76e-4 under the real FREEZE hash, vocabulary extension into paired
`base/`+`ext/` directories, candidate cache with the entropy, alias-cosine and warm-start-KL
diagnostics). Measured at 2,000 / 10,000 / 50,000 real queries: full 600,000-row pool ≈ 1.5 h,
anonymous memory ≈ 8 GiB projected (`results/m17_prepare_timing.json`). The candidate cache was
made ninefold faster with the registered ordering provably unchanged. Resume smoke: VL-A interrupted
and resumed under a changed checkpoint interval; `grad_shares` show listwise ≈ 0.88, cosine ≈ 0.25,
alias ≈ 0.003, anchor 0 of the row-gradient norm (anchor inert, `FINDINGS.md`). Reviews: Codex Astra
(21 findings) → fixes → Codex Sol (11) → fixes → one Sol P1-only re-check (12 of 14 P1s
confirmed, the two remaining fixed directly: after-screen refill to the planned total, full
recipe binding); dispositions in `REVIEW.md`; 298 `m17src` tests pass. Review loop closed. Owner rulings A4–A6 (`LEDGER.md`, registry
`accepted_plan_revision_a4/5/6`). The three descriptive sheets were **model-judged** under A6 and
ingested: panel FINAL, sha `d1b017a8a7a770ac…`, cloud-software disclosed as model-judged; alias test
57 verified / 23 rejected senses; spot check 2/308 wrong, pool stands. No candidate trained, no
development-suite or panel read, nothing published.

**Dylan owes (optional, not a clock dependency):** the seeded double-check slice
`results/m17_human_doublecheck_slice.jsonl` — 20 Kubernetes rows, 10 alias senses, 20 spot-check
pairs, model answers hidden; fill `human_relevant_yes_no` / `human_wrong` and `human_judge`.
Agreement is reported beside the model judgments whenever it arrives.

**Running at the clear (2026-09-11 ~18:10):** the full-pool build, detached:
`work/m17/prepared/full`, log `work/m17/logs/prepare_full.log`, expected ~1.5 h. Its pool plan
matched A5 exactly (496,229 general, 72,985 coverage, 15,393 alias pairs = 600,000). First thing
next session: `tail -n 20` the log and `grep -iE 'Traceback|Error|FAILED|OOM|Killed'`; if it
died, resume with the same command (stages are cached and identity-bound). Do not read any
quality surface from it.

**Next (in this order):** the registry stays `DRAFT_NOT_EXECUTABLE` until step 6.

6. Full-pool build (`prepare_data.py` without `--size`, ~1.5 h, detached, monitored), then the
   lock commit: protocol block, vocabulary list hash, tokenizer hash, seeds, cache recipe and
   artifact identities, final panel hash `d1b017a8…`. Write the measured allocation into
   `allocation_hours` from the timing result. Flip the registry status. Only then start the clock,
   run the protected screen inside the executor, and read V0.

**Working model for every M17 session (Dylan, 2026-09-11):** Fable orchestrates; Opus subagents
do execution; at most two subagents run concurrently; subagents never spawn subagents. Codex
Astra and Sol are the reviewers and run **alternately** (review, fix, next reviewer), never in
parallel on one brief; Codex Astra is also the registered judge for descriptive sheets (A6).
Commit and push after every coherent batch. Do not over-engineer. Plan a context clear at each
checkpoint: update this file first, push, then Dylan clears. Remaining checkpoint: before the
lock and clock start.

**Open dependencies:** the full-pool build and the lock (step 6). The Kubernetes slice is a
nice-to-have (Dylan, A4/A6): its documents stay in the pool inside the 10 % new-source cap and
its protected screen runs on the clock; nothing else is built around it.

Reviews and dispositions: [REVIEW.md](REVIEW.md). Plan: [PLANNING.md](PLANNING.md). Constants:
[registry.json](registry.json). Authority/probes: [LEDGER.md](LEDGER.md). Paths and pitfalls:
[CODEMAP.md](CODEMAP.md). Lessons: [FINDINGS.md](FINDINGS.md).
