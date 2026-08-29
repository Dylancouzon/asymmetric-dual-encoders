# M8 status

**Stage: re-routed by a milestone audit, 2026-08-29. `D2` is registered and is the next thing to
run.** Nine probes have run; all returned nulls, negatives or instrument readings. **No M8
training candidate exists. No protected set has been scored. The reserved four are untouched.**

## For Dylan — one decision, and it can wait for the number

The audit registered a **pre-committed exit**: if `D2` and its one alternate (`B10`/`pool_mode`)
both miss their bars, **M8 does not spend its confirmatory access** — it closes as a measurement
and the reserved four stay clean for M9. Reasoning: an access is one-shot, and with every measured
lever dead the chance of a resolved win is ≈0.002 (`m7_repeat`, `results/m8_power.json`), so
spending it re-publishes M7's miss while permanently costing M9 its clean sets. **You can override
this either way, but only before D2's number exists** (LEDGER §15, 2026-08-29). Nothing else is
blocked on you.

## What the audit found

Every cheap lever is measured dead, and the one lever with a mechanism pointing up was unregistered
and off the worklist.

| probe | result | closes |
|---|---|---|
| `B3` | +0.00135 dense at 4× dose; bar needs ~17.6× the pool | data volume |
| `E14-HEAD` | −0.0244 / −0.0293 dense, ~6× the bar the wrong way | the cheap doc-side head (= menu item D1) |
| `T1` | −0.052 / −0.109, CI-resolved | teacher swap |
| `B2` | target median entropy 4.73e-07 nats | the KL term |
| M7 clean-stack tax | +0.0058, miss survives it | licensing as the explanation |
| `B7` · `B6-pre` · `NF` ×4 | PASS / floors | instrument only — and they gate `D2` |

**`D2` is now registered** (`m8/registry.json`): a self-trained multi-word tokenizer (64–128K),
trained through the forward. Its mechanism is the strongest positive evidence in the milestone —
§17b, the table falls 0.050 nDCG behind the teacher per +1.0 subwords/word, t=4.61, surviving every
single-dataset exclusion at t ≥ 3.28. Its prior is **not** clean: the only published vocabulary-size
ablation on a static retriever (VDR, 30K→110K) moved BEIR 44.5 → 42.6, and §17b's slope sizes an
upper bound rather than a forecast because it is the *teacher* pulling away.

**Bar 0.00519, not 0.0040.** `D2` retrains the B leg, and the audit adopted §23's measured chain
floor for B-leg-varying arms — registered before the arm exists rather than after.

## Rulings in force (§15)

| # | question | ruling |
|---|---|---|
| 1 | `E14` doc-side co-adaptation | Measured small: `E14-HEAD` NO SURVIVOR. `E14-LORA` refused, TBD bar. C2 and E11 stand. |
| 2 | `E10` the shadow | Seven clean-community LoTTE slices, per-question remedy — then **REOPENED on review**; artifact unpinned, now time-boxed. |
| 3 | `harrier` | Closed on undisclosed training data. **stella stands; T1's NO SWAP is final for M8.** |
| 4 | HUPD / patents | Deferred to M9. Trigger: settle before any web-crawl data enters training. |
| 5 | `E12` LR-dense | Published numbers only, labelled. No head-to-head may be stated or implied. |

## Results

Detail lives in `m8/RESULTS.md` and the `results/m8_*.json` artifacts; no number is restated here.
The three that shape the plan: **`B3`** — reaching the bar on data volume needs ~5.9M pairs against
M7's 490K addition, which is the argument for spending M8 on capacity. **`E14-HEAD`** — fusion
absorbs ~10× of a dense change, so dense-side gains are discounted just as heavily as regressions.
**`T1`** — the tower again fails to order the table.

## Infrastructure

| item | artifact |
|---|---|
| LEDGER v2 + machine-readable registry (29 probes) | `m8/LEDGER.md`, `m8/registry.json` |
| Executable ship rule, B3's and E14's verdicts as code | `m8src/decide.py`, `b3_decide.py`, `e14_decide.py` (+ suites) |
| Guards G1/G2 | `m8src/probe_guard.py`, `paths_guard.py`, `test_guards.py` |
| Rule audit — diffs each result against the registry **at that result's commit** | `m8src/rule_audit.py` |
| E10 remediation + protected-query filter | `m8src/protected_filter.py`, `freeze_lotte.py` |
| Solver, teacher screens, entropy probe, floors, dose-curve runner | `m8src/blockcg.py`, `teacher_screen.py`, `b2_entropy.py`, `noise_floor.py`, `fused_floor.py`, `b3_pool.py` |

`./run_m8_tests.sh` runs everything. `m8/CODEMAP.md` carries the pitfalls.

**Scope discipline, in force:** `freeze.py`/`final_run.py` wait for a candidate; no new floors,
guards or registry machinery until `D2` has a number.

## File contract

| file | contract | read when |
|---|---|---|
| `m8/STATUS.md` | This. Stage, the open decision, rulings. | always, first |
| `m8/LEDGER.md` | Binding protocol: rules, bars, verdicts, amendments (§15). | before any decision |
| `m8/registry.json` | The executable half of §9. `probe_guard` reads this, not the prose. | before any run |
| `m8/NEXT-SESSION.md` | Remaining worklist. | at session start |
| `m8/RESULTS.md` / `EXPLORED.md` / `CODEMAP.md` | runs / closed avenues / modules and pitfalls. | as needed |
| `research/m8-planning/*` | Archival: reviews, literature sweep, challenger specs. | on demand |

Every number carries an artifact pointer; no file restates another; a future session cold-starts
from STATUS + LEDGER + registry alone.
