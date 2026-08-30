# Codex adversarial review #7 — final pre-launch audit

Reviewer: `gpt-5.6-sol`, high effort, read-only. Brief: `work/briefs/m9_review7.md`.
Date: 2026-08-30. Read-exclusion honoured.

Verdict: **M9 is not closed and the build is not safe to launch** — nine launch blockers, the
first a `NameError` that would crash the trainer at step 500. Disposition: `m9/LEDGER.md` §17.

---

No. M9 is not closed, and the seven-day build is not safe to launch. There are multiple deterministic launch blockers; the trainer would currently crash around step 500.

## Launch blockers

1. **Trainer crashes at its first periodic log.**  
   [`longrun.py:625`](/home/dylan/asymetric-dual-encoders/m9src/longrun.py:625) references undefined `tput`. With `log_every=500`, this raises `NameError` at step 500. The resume test hides it by setting `log_every=10**9` at [`test_resume.py:39`](/home/dylan/asymetric-dual-encoders/m9src/test_resume.py:39).

2. **The handoff backgrounds the screen and immediately continues.**  
   [`STATUS.md:135`](/home/dylan/asymetric-dual-encoders/m9/STATUS.md:135) launches the screen with `&`; there is no wait or assertion before prerequisites begin at [`STATUS.md:138`](/home/dylan/asymetric-dual-encoders/m9/STATUS.md:138). That can:

   - run target encoding and the resume test concurrently with the screen;
   - run `make_config.py` before a final decision exists;
   - commit while arms are still running.

   Worse, `make_config.py` silently supplies defaults when the decision file is absent at [`make_config.py:41`](/home/dylan/asymetric-dual-encoders/m9src/make_config.py:41) and [`make_config.py:50`](/home/dylan/asymetric-dual-encoders/m9src/make_config.py:50). It must refuse absence, not generate a plausible-looking default config.

3. **The handoff invalidates its own guard session.**  
   The session is opened before the build manifest/config are generated. Those files are in the guarded build scope at [`guard9.py:58`](/home/dylan/asymetric-dual-encoders/m9src/guard9.py:58), while `open_session()` compares the entire fingerprint at [`guard9.py:185`](/home/dylan/asymetric-dual-encoders/m9src/guard9.py:185). Creating manifest/config therefore changes the fingerprint. When the watchdog launches the trainer, `begin_run("m9-build")` can reject the session.

   This also means the advertised dependency-scoped guard is not actually scoped at session level.

4. **`test_resume.py` contaminates the real build and can prevent launch.**  
   It claims it cannot touch the real build at [`test_resume.py:15`](/home/dylan/asymetric-dual-encoders/m9src/test_resume.py:15), but redirects only `RUN`, `CKPT`, `HISTORY`, and `LOCKFILE` at [`test_resume.py:103`](/home/dylan/asymetric-dual-encoders/m9src/test_resume.py:103). It does not redirect `HEARTBEAT`, `TERMINAL`, or `MANIFEST`, defined at [`longrun.py:57`](/home/dylan/asymetric-dual-encoders/m9src/longrun.py:57).

   Each test leg ends through the normal stop path, which writes the real `work/m9long/terminal.json` at [`longrun.py:679`](/home/dylan/asymetric-dual-encoders/m9src/longrun.py:679). The handoff runs this immediately before the watchdog, and the watchdog refuses to start after any terminal marker at [`watchdog.py:225`](/home/dylan/asymetric-dual-encoders/m9src/watchdog.py:225).

5. **The selected mix is ignored.**  
   `make_config.py` hard-codes 5/5/90 at [`make_config.py:27`](/home/dylan/asymetric-dual-encoders/m9src/make_config.py:27) and never reads `selected["mix"]`. If `m9s6` selects `query-only`, the build still trains on 90% document tokens. That directly contradicts the unfilled screen decision at [`M92_LOCK.md:15`](/home/dylan/asymetric-dual-encoders/m9/M92_LOCK.md:15).

6. **Prompt policy (a) is not implemented in the training corpus.**  
   `make_config` records the selected prefix, but `longrun.prepare()` always tokenizes queries with policy (b) at [`longrun.py:177`](/home/dylan/asymetric-dual-encoders/m9src/longrun.py:177). The selected prefix is used only for the warm-start fit. If `m9s5` selects (a), warm start and SGD train different recipes.

7. **The first-evaluation gate can never fire as intended.**  
   Step zero is first appended to history at [`longrun.py:530`](/home/dylan/asymetric-dual-encoders/m9src/longrun.py:530). The first trained evaluation then checks `len(read_history()) == 1` at [`longrun.py:659`](/home/dylan/asymetric-dual-encoders/m9src/longrun.py:659); the length is already two.

8. **Stops do not perform the registered cooldown.**  
   The lock says plateau triggers cooldown at [`M92_LOCK.md:95`](/home/dylan/asymetric-dual-encoders/m9/M92_LOCK.md:95). In code, plateau merely returns a stop reason at [`longrun.py:728`](/home/dylan/asymetric-dual-encoders/m9src/longrun.py:728), after which a terminal marker is written. The stable token cap likewise stops at [`longrun.py:560`](/home/dylan/asymetric-dual-encoders/m9src/longrun.py:560). The watchdog treats both as final and exits. Therefore the normal unattended horizon ends with an unannealed stable-phase checkpoint unless someone manually starts `decay`.

9. **The watchdog itself is not robustly supervised.**  
   `write_status()` assumes every heartbeat contains `step`, `stable_token_cap`, `tok_per_s`, and `phase` at [`watchdog.py:120`](/home/dylan/asymetric-dual-encoders/m9src/watchdog.py:120), but verify/model/eval/stopped heartbeats do not necessarily contain them. A status refresh can crash the watchdog. There is no process supervising the watchdog itself. Checkpoint-stale and eval-overdue conditions also only log at [`watchdog.py:245`](/home/dylan/asymetric-dual-encoders/m9src/watchdog.py:245); they do not stop or restart the trainer.

## What `test_resume.py` really proves

It proves only clean, in-process, stable-phase split equivalence for a short run: parameters, Adam moments, stream positions, cumulative ledger, phase, and next LR.

It does not test:

- an actual process kill;
- watchdog restart;
- decay interruption/resume;
- periodic logging—the current step-500 crash is bypassed;
- normal checkpoints or evaluations;
- checkpoint/history reconciliation;
- first-evaluation gating;
- terminal/STOP cleanup.

The handoff’s “kill, resume” and CODEMAP’s broad “split-run equivalence” framing are overstated.

## Milestone-file inconsistencies

The most important ones are:

- [`M92_LOCK.md:1`](/home/dylan/asymetric-dual-encoders/m9/M92_LOCK.md:1) remains DRAFT; all four decision fields are blank at lines 12–15. No handoff command fills them.
- [`STATUS.md:3`](/home/dylan/asymetric-dual-encoders/m9/STATUS.md:3) says stage B is running, while [`STATUS.md:124`](/home/dylan/asymetric-dual-encoders/m9/STATUS.md:124) says nothing is running.
- [`STATUS.md:9`](/home/dylan/asymetric-dual-encoders/m9/STATUS.md:9) says adequacy reads the DEV-6 ceiling; the registry correctly says SCREEN-3 at [`registry.json:524`](/home/dylan/asymetric-dual-encoders/m9/registry.json:524).
- [`STATUS.md:36`](/home/dylan/asymetric-dual-encoders/m9/STATUS.md:36) says `nqopen`/`triviaqa` are excluded from all M9, while [`EXPLORED.md:12`](/home/dylan/asymetric-dual-encoders/m9/EXPLORED.md:12) and the M9.2 corpus table say they were admitted for the build.
- [`RESULTS.md:10`](/home/dylan/asymetric-dual-encoders/m9/RESULTS.md:10) still says λ was selected on the training residual. That claim is explicitly withdrawn in [`LEDGER.md:506`](/home/dylan/asymetric-dual-encoders/m9/LEDGER.md:506).
- `RESULTS.md` contains incompatible “final” anchor values: 0.50004/0.48071 at [`RESULTS.md:28`](/home/dylan/asymetric-dual-encoders/m9/RESULTS.md:28), versus 0.4998/0.4806 at [`RESULTS.md:68`](/home/dylan/asymetric-dual-encoders/m9/RESULTS.md:68). Seed sensitivity is likewise 0.00078/0.00200 at line 34 versus 0.00123/0.00230 at lines 106–108.
- [`RESULTS.md:276`](/home/dylan/asymetric-dual-encoders/m9/RESULTS.md:276) still describes all stage B as deliberately not run.
- [`registry.json:506`](/home/dylan/asymetric-dual-encoders/m9/registry.json:506) says stage B’s purpose is “the seed-sensitivity row and the five contrast arms,” although its run list contains only `m9s4`–`m9s6`.
- [`M92_LOCK.md:37`](/home/dylan/asymetric-dual-encoders/m9/M92_LOCK.md:37) says documents are ~583M tokens; the materialized corpus is 581,469,041 tokens, matching STATUS’s 581M.
- [`LEDGER.md:573`](/home/dylan/asymetric-dual-encoders/m9/LEDGER.md:573) says `make_config.py` generates constants “from this file.” It never reads `M92_LOCK.md`.
- `screen.decide()` still emits a withdrawn-reorder warning saying teacher arms may later fire at [`screen.py:399`](/home/dylan/asymetric-dual-encoders/m9src/screen.py:399).

I found no owner decision existing only in the recent commit messages: teacher withdrawal, capacity-probe withdrawal, 5/5/90, cooldown scale, and TurboQuant deferral all exist somewhere in milestone files. They are not consistently propagated.

## Teacher-screen framing

The core framing is honest: [`LEDGER.md:550`](/home/dylan/asymetric-dual-encoders/m9/LEDGER.md:550), [`registry.json:145`](/home/dylan/asymetric-dual-encoders/m9/registry.json:145), and [`STATUS.md:172`](/home/dylan/asymetric-dual-encoders/m9/STATUS.md:172) clearly label the observed curve diagnostic, the artifact void, and stella-400M as standing by default/product preference—not as a registered win.

Closure is incomplete:

- [`M92_LOCK.md:12`](/home/dylan/asymetric-dual-encoders/m9/M92_LOCK.md:12) still describes a live two-challenger decision.
- `RESULTS.md` has no diagnostic `m9s2` row or teacher-withdrawal disposition.
- [`EXPLORED.md:34`](/home/dylan/asymetric-dual-encoders/m9/EXPLORED.md:34) still uses “the teacher screen fires” as a reopening condition.
- The active screen table in [`LEDGER.md:193`](/home/dylan/asymetric-dual-encoders/m9/LEDGER.md:193) still presents both teacher arms as live.

## Guard scoping

Do not remove `guard9.py` from the protocol scope. Changing the code that decides eligibility during an arm must invalidate that arm.

The needed change is to make sessions genuinely dependency-scoped—or give M9.3 a separate build session—rather than comparing the full fingerprint in `open_session()`. The “batch edits between arms” warning is useful operational discipline, but it does not repair the current global-session contradiction.

## Seven-day resource outlook

- Disk is presently safe: 726 GB free. A bge longrun checkpoint is likely about 405 MB including model plus Adam state; approximately 99 retained eval checkpoints would consume roughly 40 GB. There is no pruning, but this box has headroom.
- Logs and history growth are small.
- Memory is tight but plausible: 25 GiB RAM against the 12.6 GB document target map, 2.33 GB packed document tokens, extra target maps, and page cache. The rolling throughput guard is the right response once the step-500 crash is fixed.
- The unresolved danger is watchdog loss: if it dies, no external component restarts it.

## Before `M92_LOCK.md` can be committed

At minimum:

1. Fix the trainer, guard-session, resume-test, prompt/mix, first-eval, cooldown, and watchdog issues above.
2. Produce an eligible, complete `results/m9_screen_decisions.json`; currently it is absent.
3. Fill teacher as stella-400M by withdrawal/default and owner ruling—not as an `m9s2/m9s3` registered screen result—and fill student, prompt, and mix.
4. Make the build behavior conditional on those selected fields, or explicitly amend the lock explaining why the seven-day mix overrides `m9s6`.
5. Generate and verify targets, manifest, and config; record their hashes in the lock.
6. Run an isolated real-process kill/resume test, including decay, without touching real terminal/heartbeat/history.
7. Ensure no stale `STOP`, terminal, lock, heartbeat, or checkpoint state remains.
8. Open the build guard session only after all guarded build artifacts are final.
9. Action the adversarial review, remove DRAFT/blanks, commit, and push.
10. Add an explicit wait-and-assert-success boundary between handoff stages.

No files were changed, and all read exclusions were honored.
