# M20 status — stage A all but done; stage B HELD on an Astra NO-GO; access still UNSPENT

**2026-09-18: the run is stopped before stage B and awaiting owner direction.** Astra reviewed
stage-B readiness at `7d72f56` and returned **NO-GO** with two P1 defects and one P2; all three
were reproduced against the code. The launcher process was killed between stages so it could not
enter stage B automatically; the `leaf-ir-asym` pre-encode was left running because stage A is
unprotected and its output is durable. No tag exists on origin and no protected payload has been
opened. Findings and the decision awaiting you: `research/m20-stageb-review-astra-2026-09-18.md`.

Astra's three findings were fixed in `cca23bb` with seven synthetic tests (286 m13src tests pass).
Sol's focused P1 re-review of those fixes returned **NO-GO again**, with three P1s: a crash in the
narrow window between writing a dataset's archive files and persisting its record is unrecoverable
without another protected read; archive reuse is verified only against the mutable record it wrote
itself, not against the registered payload hashes; and `publish` authenticates two pointer fields
rather than the result's content, so the new completion path would accept an altered result. One of
the added tests demonstrates the third defect rather than guarding against it.
Record: `research/m20-stageb-rereview-sol-2026-09-18.md`.

**This is the second NO-GO, and the governance budget — one implementation review plus one focused
P1 re-review — is spent.** `CLAUDE.md` directs pausing to simplify or take owner direction rather
than adding further locks, schemas or review cycles, so no third remediation round has been
started. Note that all three remaining findings live in recovery paths that exist only to make an
unattended stage B survive a crash; every one of them is new, untested code that would run *after*
the access is spent.

**Opened 2026-09-16 under R22. Host changed to the local box 2026-09-17 under R24.**

Reserved access is **UNSPENT**: no `m8-reserved-spent` tag locally or on origin, and
`results/m10_final_run.json` still ends `INCOMPLETE_RESERVED`. No pod was ever resumed and no
protected payload has been opened. All three retained pods are `EXITED` and stay STOP-only.

Everything is merged to `main` and pushed.

## Start here to execute

```bash
cd /home/dylan/asymetric-dual-encoders
.venv/bin/python -u scripts/m20_local_run.py               # preflight only, changes nothing
setsid nohup .venv/bin/python -u scripts/m20_local_run.py --execute \
  > work/m20/logs/local_run.log 2>&1 < /dev/null &         # stage A then stage B, ~2 days
```

The launcher keeps every readiness gate of the cloud controller that is not cloud plumbing, and its
own docstring lists what it keeps and what it drops. Run the preflight first and read it. Stage B
spends the one-shot reserved access; `reserved.crash` (R23) governs a crash after the tag.

Stage C, BEIR-15, is `m20src/beir15.py` and comes after stage B completes. It is launched
separately and sets no environment of its own, so its invocation must carry
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — see the stage-A stop below. Stage D, the
archive, is `m20src/archive.py` and is blocked on object storage; see the open items below.

## Stage A stopped once on 2026-09-17, and why it now fits

The projection gate stopped the first launch at a shard boundary after 100,000 documents:
37 docs/s measured, 58.3 h past the stage-A deadline. It worked exactly as registered — nothing
protected was touched, access stayed unspent, both shards are hash-recorded and resumable.

The cause was not the data (FEVER token length is flat across the two shards, 85.2 vs 86.3 mean),
not thermal (40k documents in 2,000-document calls hold 141 docs/s at 68–74 °C), and not the
launcher's thread pinning (the rates reproduce in a clean 8-thread process). A length-sorted
50,000-document call fragments the caching allocator — 23.06 GiB reserved on a 10.24 GiB card for
1.76 GiB of live tensors — and WSL's driver then falls back to host memory over PCIe instead of
raising OOM, so `num_alloc_retries` stayed 0 and `results/m20_vram_probe.json` passed honestly
without seeing it.

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` bounds the reservation to 4.40 GiB and gives
179 docs/s, and re-encoding FEVER 0:50000 under it reproduces the already-written shard **byte for
byte**. No registered parameter, estimand or encoding contract changed. Stage A now projects 23 h
(29 h at the conservative 141 docs/s) against the 52 h cap, and the gate stays armed.
Receipt: `results/m20_allocator_probe.json`.

Sol reviewed the stop, the fix and the relaunch: **GO WITH CONDITIONS**, one P1 — the launcher
minted a fresh `now + 52 h` on every execution, so repeated relaunches could have accumulated past
a cap the registration defines as **one** wall clock. Fixed: the earliest preserved attempt
receipt's deadline now binds. Effective slip to date is 1.83 h. The binding stage-A deadline is
**epoch 1789812044.886**; stage A must finish, or the owner must rule, before it.
Record: `research/m20-allocator-review-sol-2026-09-17.md`.

## Done

- **Registration** (`m20/REGISTRATION.md`, `m20/beir15_registry.json`, the dated
  `_amended_2026_09_16` block in `m10/final_run_registry.json`). Reversing the registry amendment
  reproduces the pre-amendment bytes exactly, which is the hash the frozen six-set result pinned.
- **Executor** extended to the eight-system roster across `m20src/roster.py`,
  `m13src/reserved_support.py`, `m13src/score13.py`, `m13src/reserved_transaction.py` and
  `m8src/pre_encode.py`. 279 `m13src` tests pass.
- **BEIR-15 runner and archive builder**, `m20src/beir15.py` and `m20src/archive.py`.
- **Two review rounds.** Astra NO-GO with seven P1 blockers, all fixed; Fable GO with no blockers.
  `m20/REVIEW_TRIAGE.md`.
- **Host validation.** Astra ENDORSE WITH CONDITIONS, all conditions discharged. `m13/RULINGS.md`
  R24, `m20/REGISTRATION.md` §8.
- **Evidence the numbers are right.** `m20/SMOKE.md`: the real BEIR-15 path on SciFact reproduces
  every independently published value. `results/m20_dbsf_reproduction.json`: three zero deltas.
- **Feasibility measured, not assumed.** `results/m20_vram_probe.json` and
  `results/m20_bm25_memory_probe_*.json`. No registered parameter needed changing.

## Open items for the owner

1. **Object storage.** No account, bucket, credentials or `rclone` on the box. The `D:` half of the
   archive can be built and verified without it; the object-storage half cannot. This blocks the
   exit criterion "archive verified at both targets", nothing earlier. `m20src/archive.py
   --verify-remote` is implemented and waiting.
2. **Retained storage bills about $0.41/hour** across the three stopped pods, roughly double the
   $0.2174/hour the M13 allocation assumed, whether or not anything runs. Measured over a 7-hour
   window with everything stopped. Retirement is M22's to request; they stay STOP-only until then.
3. **Stage C is the long pole:** about 130 hours of local GPU for BEIR-15 against about 52 for the
   reserved four. Nothing blocks it, but it is five days of the machine.

## The tightest point in the plan

BM25's peak host memory. FEVER needs 20.1 GB and MS MARCO 20.9 GB on a 26.7 GB box, and two large
corpora in one process reached 23.7 GB. Document count does not predict it: FEVER needs 60% more
than HotpotQA at almost the same count. The datasets run largest-first and each corpus's text is
freed and collected before the next loads. If a future change touches that ordering or that
freeing, re-measure before running.

## Noted, not fixed

- Running the `m10src` suite rewrites `results/m10_contrast_E1.json`, a real registered result
  file, which `CLAUDE.md` forbids. Pre-existing on `main`. Check `git status` after `run_checks.sh`.
- Three review debt items are recorded in `m20/REVIEW_TRIAGE.md` and deliberately not fixed.

## Pointers

- Mandate `instructions-m20.md`. Rulings `m13/RULINGS.md` R19, R20, R22, R23, R24.
- Registration `m20/REGISTRATION.md`, `m20/beir15_registry.json`. Smoke `m20/SMOKE.md`.
- Reviews `m20/REVIEW_TRIAGE.md`, `research/m20-codex-impl-review-2026-09-16.md`,
  `research/m20-host-decision-brief-astra-2026-09-17.md`.
- Failed cloud attempts `results/m13_reserved_cloud_attempt{1,2}_nocapacity_*.json`.
- Artifact paths, hashes, restore commands `m14/HANDOFF.md`. Checks `HARNESS.md`.
- Follow-on `instructions-m22.md` (release), `instructions-m23.md` (upstream PRs).
