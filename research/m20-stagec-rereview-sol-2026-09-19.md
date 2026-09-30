# M20 stage-C invocation re-review — Sol, 2026-09-19

Reviewer: Codex `gpt-5.6-sol`, read-only. Brief: `work/m20/logs/sol_stagec_brief.md`.
Transcript: `work/m20/logs/sol_stagec_review.log`. Reviewed at `eb31964`.

# Verdict: NO-GO

Do not launch stage C at `eb31964`.

1. **P1 — The registered 120-hour clock starts after stage work has already begun and can overrun at expiry.**  
   [scripts/m20_local_run.py:203](/home/dylan/asymetric-dual-encoders/scripts/m20_local_run.py:203), [scripts/m20_local_run.py:240](/home/dylan/asymetric-dual-encoders/scripts/m20_local_run.py:240), [scripts/m20_local_run.py:255](/home/dylan/asymetric-dual-encoders/scripts/m20_local_run.py:255), [scripts/m20_local_run.py:274](/home/dylan/asymetric-dual-encoders/scripts/m20_local_run.py:274)

   Stage-C preflight runs before the deadline is created. That preflight loads every BEIR-15 corpus and may spend up to two hours downloading them, although the registered stage explicitly includes download. After the deadline, `max(0.05, remaining)` still grants each subsequently launched phase three minutes; phases that complete inside their floors can cumulatively carry execution beyond the deadline.

   **What breaks:** the single registered 120-hour download/encode/score cap is not a hard bound.

2. **P1 — The deadline is not durable before the five-day job starts.**  
   [scripts/m20_local_run.py:240](/home/dylan/asymetric-dual-encoders/scripts/m20_local_run.py:240), [scripts/m20_local_run.py:295](/home/dylan/asymetric-dual-encoders/scripts/m20_local_run.py:295)

   The deadline exists only in memory until the `finally` block writes the receipt. An ordinary Python exception preserves it, but process termination, host failure, or power loss can leave no attempt receipt. A relaunch then creates a fresh 120-hour deadline rather than inheriting the original one.

   **What breaks:** repeated attempts can cumulatively exceed the registered cap.

3. **P1 — The projection gate does not project completion of the whole stage.**  
   [m8src/pre_encode.py:384](/home/dylan/asymetric-dual-encoders/m8src/pre_encode.py:384), [scripts/m20_local_run.py:257](/home/dylan/asymetric-dual-encoders/scripts/m20_local_run.py:257)

   Its estimate contains only the current tower’s remaining document encodes and later towers. It reserves no time for the scorer, which runs afterward under the same deadline. Encoding can therefore be allowed to consume essentially the entire cap before scoring starts.

   **What breaks:** the projection gate can approve a run that cannot produce the BEIR-15 result within the cap, discovering that only after the expensive encoding work.

4. **P1 — Projection accounting is not resume-aware.**  
   [m8src/pre_encode.py:309](/home/dylan/asymetric-dual-encoders/m8src/pre_encode.py:309), [m8src/pre_encode.py:332](/home/dylan/asymetric-dual-encoders/m8src/pre_encode.py:332), [m8src/pre_encode.py:373](/home/dylan/asymetric-dual-encoders/m8src/pre_encode.py:373)

   Each relaunched tower initializes projected rows to zero. Existing hash-verified shards are skipped without being counted, while remaining work is calculated from the full `beir15_new_docs_expected`. A partially completed tower therefore treats already encoded documents as still outstanding and can falsely trip the gate even when the actual remaining work fits the inherited deadline.

   **What breaks:** the registered resumability path can wrongly refuse a viable continuation.

5. **P1 — The active status document still directs the operator to the rejected scorer-only entry point.**  
   [m20/STATUS.md:47](/home/dylan/asymetric-dual-encoders/m20/STATUS.md:47)

   It says stage C is launched separately through `m20src/beir15.py` and that the launcher does not configure it. Following that instruction recreates Astra’s original finding: no document encoding, projection gate, or stage deadline.

   **What breaks:** reproducible execution at the reviewed commit; the repository still gives two contradictory stage-C invocations, one of which cannot complete the stage.

The remaining scoped checks passed: the fresh-run projected volume equals the registered 23,744,806 documents; the projection object spans all non-reserved corpus configurations in each tower; `--batch beir15` excludes the reserved four; the scorer refuses to rescore them and copies their atomic outputs into separate M20 artifacts; the SPENT/COMPLETE gates are correctly inverted; and CQADupStack uses the unweighted mean of its twelve forum means. The disk threshold is exactly 225,337,395,712 bytes, leaving about 303.6 GB beyond it at the recorded free-space reading.

Access log:

- `CLAUDE.md`
- `scripts/m20_local_run.py`
- `m20src/beir15.py`
- `m8src/pre_encode.py`
- `m20/beir15_registry.json`
- `m20/REGISTRATION.md`

## Disposition

Four fixed, one accepted. The standard applied is the owner's: M20 exists to produce
unimpeachable benchmark numbers, and findings are weighed by whether they threaten that.

- **P1 1 (clock starts after work begins; soft floor) — FIXED.** The stage-C deadline now
  starts before the preflight, so the corpus download the registered stage explicitly includes
  falls inside the cap. `_remaining_hours` refuses past the deadline instead of granting every
  phase another three minutes.
- **P1 2 (deadline not durable) — FIXED.** The receipt is written with the deadline in it,
  status `RUNNING`, before any stage work starts, so a kill or power cut still leaves the
  original deadline for a relaunch to inherit.
- **P1 3 (no scorer reserve) — FIXED.** The towers get a deadline short of the real one by
  17.06 h. That figure is derived from the registration's own numbers — its 113 h expectation
  for stage C minus the 95.94 h encode share implied by the registered volume, combined tower
  ratio and A100 rate — not a new constant. This was the finding with real teeth: encoding
  could have consumed the whole cap and left no time to score, spending five days for no
  BEIR-15 numbers.
- **P1 5 (STATUS.md points at the rejected entry point) — FIXED.** It now gives the
  `--stage-c` command and says plainly not to invoke the scorer directly.
- **P1 4 (projection not resume-aware) — ACCEPTED, not fixed.** Real, and fail-safe: a
  relaunched tower counts already-encoded documents as outstanding and can refuse a viable
  continuation. It wastes time; it cannot corrupt a number. The fix would mean changing
  `Projection` in `m8src/pre_encode.py`, the module that produced the reserved vectors and
  numbers now that the access is spent and unrepeatable. Not worth that risk for a
  false-refusal. If it trips, the answer is to inspect and relaunch.
