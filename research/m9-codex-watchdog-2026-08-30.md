# Codex adversarial review #6 — the watchdog and the heartbeat

Reviewer: `gpt-5.6-sol`, high effort, read-only. Brief: `work/briefs/m9_review6.md`.
Date: 2026-08-30. Read-exclusion honoured.

Verdict: **DO NOT LAUNCH UNATTENDED** — six blockers. The first defeated every stopping rule the
trainer had just gained; the second is that the throughput guard written for the measured `m9s2`
slowdown would not have caught `m9s2`. Disposition: `m9/LEDGER.md` §16.

---

## Verdict: DO NOT LAUNCH unattended

Read-only static review; exclusions honored. The motivating 5× GPU slowdown still gets through, and the watchdog currently defeats the trainer’s stopping rules.

### Launch blockers

1. **The watchdog resurrects every intentional stop.**

The trainer exits normally after first-eval failure, regression, plateau, throughput collapse, non-finite values, cooldown completion, or the token cap. It writes no terminal marker. The watchdog sees only “no matching PID” and restarts it ([watchdog.py:172](/home/dylan/asymetric-dual-encoders/m9src/watchdog.py:172)).

Concrete failure: first eval scores 0.40, trainer stops and checkpoints; watchdog restarts at the following step, where the first-eval gate can never run again. Training then continues for days.

Write an atomic terminal state containing the stop reason and checkpoint step. The watchdog must restart only after an unexpected disappearance, never after a registered trainer stop.

2. **The exact motivating slowdown can still look healthy.**

Throughput is cumulative session throughput:

```python
rate = sess_tok / (time.time() - t0)
```

([longrun.py:515](/home/dylan/asymetric-dual-encoders/m9src/longrun.py:515)). It is not throughput since the previous heartbeat.

Consequences:

- If the teacher is resident when training starts, the early median is already ~5,400 tok/s and the floor becomes ~2,700. The run remains “healthy” forever.
- If slowdown begins after three good days, a 5× slowdown needs roughly five additional days to pull the cumulative average below 50%—outside the unattended window.
- After any restart, `tput` is empty again, so the degraded rate becomes the new healthy baseline.
- `gpu_ok()` only establishes that `nvidia-smi` returns something; a teacher occupying 9.6 GB passes ([watchdog.py:72](/home/dylan/asymetric-dual-encoders/m9src/watchdog.py:72)).

Use `Δtokens / Δwall` over a rolling 5–10 minute window. Freeze and persist a known-good baseline across restarts. Based on the registered 26,854 tok/s, the current 50% policy means an absolute initial floor of about **13,427 tok/s**, pending a same-mixture pilot.

3. **Restart can create two checkpoint writers.**

The lock is a check-then-write race, not an atomic lock ([longrun.py:375](/home/dylan/asymetric-dual-encoders/m9src/longrun.py:375)). Meanwhile the watchdog:

- Finds trainers with a broad repository-independent `pgrep -f` pattern.
- Sends SIGTERM.
- Waits exactly 15 seconds.
- Deletes the lock unconditionally.
- Launches a replacement without proving the old PID exited.

([watchdog.py:59](/home/dylan/asymetric-dual-encoders/m9src/watchdog.py:59), [watchdog.py:178](/home/dylan/asymetric-dual-encoders/m9src/watchdog.py:178)).

If the old process is stuck in uninterruptible I/O or slow shutdown, both trainers can write the same `last.tmp`. Atomic `os.replace` is safe against one writer crashing, but not two writers sharing the same temporary filename ([longrun.py:356](/home/dylan/asymetric-dual-encoders/m9src/longrun.py:356)).

Use `flock` or `O_CREAT|O_EXCL`, retain the lock descriptor, bind it to PID plus `/proc` start time, and refuse to unlink/start until the exact old process is confirmed gone.

4. **A fresh-start wedge is invisible forever.**

With no heartbeat yet:

```python
stale = hb and ...
no_progress = hb and ...
```

Both are false ([watchdog.py:165](/home/dylan/asymetric-dual-encoders/m9src/watchdog.py:165)). Therefore a trainer hanging during verification, target mapping, model construction, or warm start remains “alive” indefinitely.

Add a startup deadline and emit heartbeat states during `verify`, `load_targets`, `warm_start`, `train`, `eval`, and `checkpoint`.

Also, despite the module documentation, there is no implemented checkpoint-freshness or eval-cadence check. A cleanup process can delete `last.pt` or truncate history while steps and heartbeat continue advancing; the watchdog says nothing.

5. **The plateau rule cannot fire.**

It takes only the final two evaluations, then requires them to span 1B tokens ([longrun.py:590](/home/dylan/asymetric-dual-encoders/m9src/longrun.py:590)). Adjacent evals are approximately 164M tokens apart, so the condition is unreachable.

Compare the current evaluation with the latest record at or before `current_tokens - 1B`—about seven evals back—not `rows[-2]`.

6. **History and checkpoint are not recoverably transactional.**

The eval checkpoints are saved before history is appended ([longrun.py:540](/home/dylan/asymetric-dual-encoders/m9src/longrun.py:540)). A crash after `last.pt` is replaced but before the append resumes after that eval, so:

- The evaluation is permanently absent from history.
- The first-eval gate can be skipped.
- Regression/plateau windows lose evidence.

A torn final JSONL line also glues to the next appended record, losing both records to the tolerant parser. Reconcile history from the `eval` embedded in `step*.pt`, or use atomic per-eval record files.

`test_resume.py` does not test this: it performs a clean `max_steps` return with a final checkpoint, disables evaluation/history, and never sends SIGTERM or interrupts a save ([test_resume.py:35](/home/dylan/asymetric-dual-encoders/m9src/test_resume.py:35), [test_resume.py:112](/home/dylan/asymetric-dual-encoders/m9src/test_resume.py:112)). It also does not exercise interrupted decay or corrupt-checkpoint fallback.

### Threshold disposition

- `--period 60`: keep.
- `--stale 900`: replace the heartbeat design first. Emit every 60 seconds independently of logging, then use **300 seconds** plus a separately measured eval timeout. Currently a 5× slowdown produces a heartbeat about every 763 seconds, deliberately staying below 900.
- `--max-restarts 8`: acceptable only with intentional-stop recognition, exponential backoff, and a rolling cap such as **3 restarts per 6 hours / 8 total**.
- `--min-disk-gb 25`: adequate tonight; the target filesystem currently has roughly 726 GB free. Check the filesystem containing `CKPT`, not `REPO`, and define the floor as at least two full checkpoint sizes plus reserve.
- Throughput `0.5×`: fraction is reasonable; cumulative/session-local measurement is not. Freeze it across restarts.
- `first_eval_floor 0.45`: unsupported as a hard absolute gate. The anchor used a different training mixture; this build has seen only about 8.2M query tokens by its first eval. Make 0.45 log-only until a same-recipe pilot exists. For a catastrophe gate tonight, evaluate step zero and use something broad such as `first_eval >= step0 - 0.02`.
- Plateau `+0.001 / 1B`: policy is defensible; implementation is dead.
- Regression `0.0056` for two evals: keep, once history integrity is fixed.

### Git status push

`RUN_STATUS.md` is outside the guard fingerprint, so writing it does not directly dirty the build scope. HEAD movement is also tolerated because the session binds fingerprints rather than HEAD.

The implementation is still unsafe:

- `git commit` commits **all already-staged human changes**, not just the status file.
- A human commit between the watchdog’s commit and push can be pushed automatically.
- Index-lock, authentication, non-fast-forward, and network failures are completely suppressed ([watchdog.py:116](/home/dylan/asymetric-dual-encoders/m9src/watchdog.py:116)).
- A failed status push leaves HEAD unpushed, so the next restart fails the guard and gives up after two launches.

Use a dedicated status branch/worktree or separate repository/index, push an exact commit SHA, and log failures prominently. Do not mutate or push `m9-work` from the watchdog.

Finally, the build scope omits both `m9src/watchdog.py` and `m9/M92_LOCK.md` ([guard9.py:56](/home/dylan/asymetric-dual-encoders/m9src/guard9.py:56)). Both should be pinned; `RUN_STATUS.md` should remain excluded.
