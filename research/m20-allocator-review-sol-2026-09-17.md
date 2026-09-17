# M20 stage-A allocator fix — Sol review, 2026-09-17

Reviewer: Codex `gpt-5.6-sol`, read-only sandbox. Brief: `work/m20/logs/sol_allocator_brief.md`.
Full transcript: `work/m20/logs/sol_allocator_review.log`.

Reviewed: the stage-A projection stop of 2026-09-17, its diagnosis
(`results/m20_allocator_probe.json`), the `PYTORCH_CUDA_ALLOC_CONF` fix and the relaunch.
Commits `9a26727`, `fcd1566`.

## Verdict: GO WITH CONDITIONS

Let the current stage-A run continue, but enforce the original stage-A deadline (`1789812044.8856924`) or obtain an owner ruling before exceeding it or entering stage B. At the observed rate, completion projects to roughly 32 hours including attempt 1, comfortably within 52 hours.

## Essential finding

**P1 — Relaunch resets the registered stage-A wall clock.**

The launcher creates a fresh `now + 52 h` deadline on every execution ([scripts/m20_local_run.py:206](/home/dylan/asymetric-dual-encoders/scripts/m20_local_run.py:206)), although the registration defines one wall clock and absolute deadline for the whole stage ([m20/REGISTRATION.md:257](/home/dylan/asymetric-dual-encoders/m20/REGISTRATION.md:257)). Attempt 1 already consumed 0.8 hours ([results/m20_local_run_attempt1_projection_stop.json:25](/home/dylan/asymetric-dual-encoders/results/m20_local_run_attempt1_projection_stop.json:25)).

Consequently, repeated relaunches could accumulate more than 52 hours while each process remains inside its own newly issued deadline. That breaks the registered cap and could silently assume an owner decision reserved by the plan.

The skipped-shard accounting is not the permissive part: `Projection` starts at zero ([m8src/pre_encode.py:373](/home/dylan/asymetric-dual-encoders/m8src/pre_encode.py:373)) and only observes newly written shards, so it treats the existing 100,000 rows as unfinished. This makes its estimate pessimistic by about 0.21 hours at 130 docs/s and could stop a near-cap healthy run; it cannot itself let one overrun. The fresh deadline can.

## Other determinations

- The allocator change is an operational fix allowed by [CLAUDE.md:74](/home/dylan/asymetric-dual-encoders/CLAUDE.md:74), not a change to fp32 computation, TF32 policy, batch construction, or storage dtype. R23 is inapplicable before the tag. No separate owner ruling was required for the allocator setting itself.
- The one-shard comparison supports contract equivalence and the mixed-shard resume. Existing shards are hash-verified before being skipped ([m8src/pre_encode.py:309](/home/dylan/asymetric-dual-encoders/m8src/pre_encode.py:309)), and no reviewed code path makes allocator policy an encoding operator.
- It does not prove the receipt’s literal “changes no vector” claim for every tower and shape. If literal byte-universality must be established, compare the second old FEVER shard—the one produced in the fragmented state—against a fixed-allocator re-encode, then compare representative BGE and Arctic batches under both settings. This is additional assurance, not a blocker to the registered numerical contract.
- I found no other essential error in the diagnosis. The exact WSL host-memory mechanism is asserted rather than directly measured, but the allocator intervention and throughput response adequately support the operational diagnosis and do not affect result validity.

## Access log

Read only the 13 allowlisted files:

- `results/m20_allocator_probe.json`
- `scripts/m20_local_run.py`
- `m8src/pre_encode.py`
- `m7src/teacher.py`
- `m20/REGISTRATION.md`
- `m20/STATUS.md`
- `m13/RULINGS.md`
- `CLAUDE.md`
- `results/m20_vram_probe.json`
- `results/m20_tower_rate_benchmark.json`
- `results/m13_encode_benchmark.json`
- `results/m20_local_run_attempt1_projection_stop.json`
- `work/m20/logs/stageA-nano-dense.log`

Also ran exactly the three authorized Git inspections. No protected paths, training, evaluation, process control, or edits were performed.
tokens used
71,996
## Verdict: GO WITH CONDITIONS

Let the current stage-A run continue, but enforce the original stage-A deadline (`1789812044.8856924`) or obtain an owner ruling before exceeding it or entering stage B. At the observed rate, completion projects to roughly 32 hours including attempt 1, comfortably within 52 hours.

## Essential finding

**P1 — Relaunch resets the registered stage-A wall clock.**

The launcher creates a fresh `now + 52 h` deadline on every execution ([scripts/m20_local_run.py:206](/home/dylan/asymetric-dual-encoders/scripts/m20_local_run.py:206)), although the registration defines one wall clock and absolute deadline for the whole stage ([m20/REGISTRATION.md:257](/home/dylan/asymetric-dual-encoders/m20/REGISTRATION.md:257)). Attempt 1 already consumed 0.8 hours ([results/m20_local_run_attempt1_projection_stop.json:25](/home/dylan/asymetric-dual-encoders/results/m20_local_run_attempt1_projection_stop.json:25)).

Consequently, repeated relaunches could accumulate more than 52 hours while each process remains inside its own newly issued deadline. That breaks the registered cap and could silently assume an owner decision reserved by the plan.

The skipped-shard accounting is not the permissive part: `Projection` starts at zero ([m8src/pre_encode.py:373](/home/dylan/asymetric-dual-encoders/m8src/pre_encode.py:373)) and only observes newly written shards, so it treats the existing 100,000 rows as unfinished. This makes its estimate pessimistic by about 0.21 hours at 130 docs/s and could stop a near-cap healthy run; it cannot itself let one overrun. The fresh deadline can.

## Other determinations

- The allocator change is an operational fix allowed by [CLAUDE.md:74](/home/dylan/asymetric-dual-encoders/CLAUDE.md:74), not a change to fp32 computation, TF32 policy, batch construction, or storage dtype. R23 is inapplicable before the tag. No separate owner ruling was required for the allocator setting itself.
- The one-shard comparison supports contract equivalence and the mixed-shard resume. Existing shards are hash-verified before being skipped ([m8src/pre_encode.py:309](/home/dylan/asymetric-dual-encoders/m8src/pre_encode.py:309)), and no reviewed code path makes allocator policy an encoding operator.

## Disposition

- **P1 fixed** in `scripts/m20_local_run.py`: every preserved attempt receipt carries the
  deadline it ran under, and the earliest one binds, so relaunches can no longer accumulate
  past the registered 52 h. The running process is unaffected; the change binds future
  attempts. Effective slip to date is 1.83 h (attempt 2 started that long after attempt 1),
  against a stage projecting ~32 h.
- **Receipt wording corrected** in `results/m20_allocator_probe.json`: the byte-identity claim
  now states its scope (one FEVER shard, nano-dense) rather than implying universality, and
  the WSL host-memory mechanism is marked inferred rather than measured.
- **No other finding.** Sol confirmed the allocator change is an operational fix under
  `CLAUDE.md`, that R23 is inapplicable pre-tag, that no owner ruling was required for it, and
  that the mixed-shard resume is sound because existing shards are hash-verified before being
  skipped.
