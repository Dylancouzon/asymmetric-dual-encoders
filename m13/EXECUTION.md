# M13 execution backlog — 2026-09-10

**Next (2026-09-10, stage 2 code ready):** provider and account (Dylan); day one on the instance:
measured rate and billed price into the allocation table, then both E arms with `--dev6 defer`.
Stage 3 runs `m13src/lotte_gate13.py --preflight-only` then the read, once, after both E records
are pushed. The nano scorer is `m13src/score13.py` (rehearsed on the synthetic fixture); M9's path
in `final9.py` is still the stub and is wired last, behind the recipe-decision gate.

| Work package | Completion condition |
|---|---|
| Pre-build LoTTE | Metric and seven slices fixed before access; compare correct cloud E checkpoints; enforce pre-build skip/veto and execution record. **Code complete 2026-09-10:** `m13src/lotte_gate13.py` (R16) with 28 synthetic tests and its LEDGER 15 allowlist entry; the read itself waits for both E records |
| Executor identity | Verify comparator hash; bind rows to checkpoint/system, code/registry and exact frozen qids; iterate all six, not bench's five-set default |
| Durable access/recovery | Authenticated empty manifest before spend; handle crash before first score; allow only declared output drift; pin code/config on continuation/recovery |
| Preflight / M9 | No protected query/qrel reads in preflight; restore inherited 120 GB space requirement; resolve M9's legacy bridge and ratification before close-out |
| Review and budgets | Rehearse the real caller; clean independent reviews; include conditional reserved cost in M13 allocation; register paired M9/nano row before either final run |
| Cost frontier | Same-machine zero/bge-small/nano serving costs and complete system/index costs; Edge fusion still unmeasured |
| Build controller (found 2026-09-10) | 200M pin, extension rule and cap, ONNX/parity at freeze, spend accounting have no code; `arm_doc_count` demands 50M unique documents at 200M against a ~6.15M pool; wall-clock checkpoint cadence; full uncut A4 for the build. Design and owner questions: `m13/STAGE1_DESIGN.md` |
| Cloud E (found 2026-09-10) | `E-bs32` missing from runner shape tables (fixed); `pending` blocks E1 and clearing it re-binds the F verdict and ten contrasts (ruling R9, applied); DEV-6 for the E arms is box-side: `run_arm.py --dev6 defer` on the instance, `m13src/dev6_from_checkpoint.py <arm>` on the box from the returned `cycle3.pt` (2026-09-10); reserved-batch encodes priced as an estimate (R10), re-measured on day one. Details: `m13/STAGE1_DESIGN.md` §4 |
| Scoring executor (found 2026-09-10) | No access machinery under M10 paths yet; registry cites M7's payload-reading preflight while this file requires manifests only (ruling R1); `min_free_gb` 120 absent from M10's registry and code; M9 still carries the withdrawn 0.0003 bridge (R3) |

Original findings: `research/archive/m10-cleanup-2026-09-10/m10/M10_4_DECISION_LOCK.md`.
The missing ±1.5 decision-range test is now covered. Scope/freeze prose and budget-range errors
were removed from the active handoff; remaining executor findings are **not claimed fixed**.

**Gates:** no six/reserved/LoTTE access, no spent tag, and no M9 close-out triggered merely by a
recipe-lock push. LoTTE handling may block M13's full build; final six-set scoring waits for its
frozen candidate. M9's rows must not inform any still-open recipe decision.

The bridge's 0.0003 per-query tolerance was never validated for re-encoding. Its alleged minimum
nDCG quantum is false: multiple rank changes can cancel. See
`research/m10-cleanup-review-2026-09-10.md`. Resolve the intended parity contract before execution;
do not copy a changed tolerance into M9 without its pre-observation amendment.

## Readiness and day-one runbook (recorded 2026-09-10, session closed)

**Ready now, verified on the box with `./run_checks.sh` (M10 474, M9 16, M12, M13 144 tests):**
both E arms in the runner (`E-bs32` shape, `pending` cleared under R9, `--dev6 defer`); the ship
list with the recommended split; `build13.py --plan --rate --price`, which prints both batch tables
and the allocation under a PENDING E1 batch; the fixed 200M controller; the six-set executor with
its synthetic rehearsal; the LoTTE gate script with `--preflight-only` (today it refuses at the
PENDING verdict, as it must).

**Owed before each spend:**

| Spend | Owed | Owner |
|---|---|---|
| Renting | provider, account, ≥ 500 GB persistent disk, SSH, GitHub deploy key (`m13/RULINGS.md` Provider) | Dylan |
| LoTTE read #1 | two independent reviews of `m13src/lotte_gate13.py` (Astra 2026-09-10 done, nine findings fixed; the re-review must return GO); both E records pushed; the committed manifest and the pin (R18); `--preflight-only` clean | lead |
| 200M build | `m13/LOTTE_GATE.json`; allocation at the measured rate and billed price under $1,000; reserved-batch allowance re-measured (R10) | lead |
| Final six-set access | frozen candidate; R6 flip in the commit that pins the reviewed executor; M9's dated R3 amendment before any M9 score | Dylan, lead |

**Day one, in order** (stop the instance between stages; detach anything long with `setsid nohup`
and monitor `Traceback|Error|FAILED|OOM|Killed|assert`; read the first progress line's rate):

1. Clone at `/home/dylan/asymetric-dual-encoders`, build `.venv` per `HARNESS.md`, rsync per
   `m13/SHIP_LIST.md` without the DEV-6 group, `sha256sum results/perquery.json` must match.
2. `./run_checks.sh`; `.venv/bin/python m10src/arm_smoke.py --device cuda`;
   `.venv/bin/python m10src/run_arm.py --plan`.
3. Benchmark hour: examples/s at bs32 and bs128 from the smoke, billed $/h from the provider, then
   `.venv/bin/python m13src/build13.py --config m13/build_config.json --plan --rate EX_S --price USD_H`.
   Record rate, price and the printed allocation in this file, dated; the build record captures
   them again.
4. `.venv/bin/python m10src/run_arm.py E-bs32 --device cuda --dev6 defer`, then the same for
   `E-bs128`; commit and push both records from the instance; copy `work/m10arms/E-bs32/` and
   `E-bs128/` (record and `cycle3.pt`) back to the same paths on the box.
5. On the box, after `git pull`: `.venv/bin/python m13src/dev6_from_checkpoint.py E-bs32` and
   `E-bs128`; compute E1 and re-issue `results/m10_screen_verdicts.json` with
   `m10src/contrasts.py`; commit and push.
6. Stage 3, on whichever machine holds both `cycle3.pt` files and the stella weights, after
   `git pull`: `.venv/bin/python m13src/lotte_gate13.py --write-manifest`, commit and push
   `m13/LOTTE_GATE_MANIFEST.json` (the lock's second manifest commit); `.venv/bin/python
   m8src/freeze_lotte.py pin`, commit and push `results/m8_lotte_pin.json` (R18, same day);
   `.venv/bin/python m13src/lotte_gate13.py --preflight-only`; then the read, once. If it crashes,
   `--recover` completes it after the crashed process has exited; nothing else re-opens the
   surface. Commit and push `m13/LOTTE_GATE.json`: the build refuses a gate record or manifest
   that is not committed and pushed, and recomputes the decision from the recorded bootstrap.
7. Build, on the instance after `git pull`: `.venv/bin/python m13src/build13.py --config
   m13/build_config.json --device cuda --rate EX_S --price USD_H`.
