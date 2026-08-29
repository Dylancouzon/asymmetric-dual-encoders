# M8 status

**Stage: re-routed by a milestone audit, 2026-08-29. `D2` is registered and is the next thing to
run.** Nine probes have run; all returned nulls, negatives or instrument readings. **No M8
training candidate exists. No protected set has been scored, and the confirmatory access is
unspent** — though the reserved four are no longer *unread*, see the incident below.

## Incident, RULED and closed

**2026-08-29: an adversarial reviewer read two RESERVED sets in full** (queries *and* qrels:
`untouched-cqadup-english`, `untouched-dbpedia-entity`) via a repo-wide grep. Nothing was scored;
no model or decision read them. `paths_guard` cannot cover an external process — a structural hole
the routine-review grant reopens each time. **Dylan ruled: "the sets are fine, continue with the
codex review."** The reserved four stand, the access is intact, no quarantine. Process fix kept
anyway: briefs carry a read-exclusion, logs are audited before findings are read (LEDGER §15;
CLAUDE.md).

## `E14-LORA` is REOPENED — and one question is now the milestone's real fork

Dylan, 2026-08-29: *"We wouldn't say keep your normal document encoder. Since most people are not
currently using stella. I'm not against LoRA on the document tower"* + *"stella is good for derived
weights, no license blocker."* Both premises the refusal rested on are gone: the user re-indexes
with stella either way, so co-adapting the tower has **zero marginal product cost**, and MIT covers
derived weights.

**The reframe behind it:** LightRetriever **trains its document encoder**
(`research/lightretriever.md:19,23,382`). `LR-dense-pertask 0.4583` — M7's missed release bar — was
set by a system with a co-adapted document side, while M7 and M8 fit a table to a tower never
trained to be fit. **M8 has been solving a harder problem than the system it is benchmarked
against**, which is the best available explanation for why every cheap table-side lever comes back
flat. Unmeasured — that is the point of the probe. `E14-HEAD` does not settle it: a head on a
*finished* document vector cannot recover what the tower discarded, and §15 registered that scope
limit before any arm ran.

**Needed from Dylan before it can SHIP — not before it can be measured: does M8 ship a better
SYSTEM, or must it ship a better TABLE?** `E11`/§5.4 say a document-side win is not a qualifying v2
table, and `C2` falls to its registered `teacher_swapped` branch. It cannot be decided after a
number. Staging is binding either way: dev-scale first, on the two out-of-domain components against
their own re-encoded corpora; only a clearing dev result buys the 10.12M pre-encode.

## The other decision, and it can wait for the number

A **pre-committed exit** is registered: M8 does not spend its confirmatory access without a credible
release candidate — it would close as a measurement, leaving the reserved four clean for M9. An
access is one-shot, and with the cheap levers dead the chance of a resolved win is ≈0.002
(`m7_repeat`, `results/m8_power.json`), so spending it re-publishes M7's miss while permanently
costing M9 its clean sets.

**The exit cannot fire early.** It was registered firing after `D2` + `B10`; a review showed that
was premature on our own evidence, and it is now gated on `D2`, `B10`, `B8` and `R-LIST` all having
run and missed, plus a re-run of CLAUDE.md's standing directive at that point. **You can override
the default either way, but only before the numbers exist.** Nothing else is blocked on you.

## What the audit found

Every cheap lever is measured dead, and the one lever with a mechanism pointing up was unregistered
and off the worklist.

| probe | result | closes |
|---|---|---|
| `B3` | +0.00135 dense at 4× dose; bar needs ~17.6× the pool | data volume |
| `E14-HEAD` | −0.0244 / −0.0293 dense, ~6× the bar the wrong way | the cheap doc-side head (= menu item D1) |
| `T1` | −0.052 / −0.109, CI-resolved | teacher swap |
| `B2` | uniform-bank target median entropy 4.73e-07 nats | **the recipe's uniform-bank KL — NOT the KL class.** Its `teacher_top200` arm is 0.777 nats mean and B2's own artifact names `R-LIST` as the consequence, so hard-candidate listwise distillation stays live (corrected 2026-08-29 after review) |
| M7 clean-stack tax | +0.0058, miss survives it | licensing as the explanation |
| `B7` · `B6-pre` · `NF` ×4 | PASS / floors | instrument only — and they gate `D2` |

**`D2` is registered, then amended after an adversarial review** (LEDGER §15) that returned three
BLOCKERs, two against the audit's own decisions: the "compositional init floor" was the **mean** of
constituent rows and is not a floor at all — the **sum** is, exactly — the coverage gate contained a
pool-expansion clause that would have voided the bar, and the exit fired too early. **`D2-PRE` is
now registered ahead of it**: a sub-hour closed-form preflight on the solver `B7` already proved,
comparing D2's segmentation against additive overlapping word and character n-gram rows at equal
row budget. It can reverse the plan — if an additive arm wins, D2 stands down, and that reversal is
registered before the number.

D2 is a self-trained multi-word tokenizer (64–128K), trained through the forward. Its mechanism is
the strongest positive evidence in the milestone — §17b, the table falls 0.050 nDCG behind the
teacher per +1.0 subwords/word, t=4.61, surviving every single-dataset exclusion at t ≥ 3.28 — but
that is **correlated headroom, not a bound in either direction** (downgraded after review; it is
uncontrolled between-query OLS). Its prior is **not** clean: the only published vocabulary-size
ablation on a static retriever (VDR, 30K→110K) moved BEIR 44.5 → 42.6, and §17b's slope sizes an
upper bound rather than a forecast because it is the *teacher* pulling away.

**Bar 0.00519, not 0.0040.** `D2` retrains the B leg, and the audit adopted §23's measured chain
floor for B-leg-varying arms — registered before the arm exists rather than after.

## Rulings in force (§15)

| # | question | ruling |
|---|---|---|
| 1 | `E14` doc-side co-adaptation | `E14-HEAD` NO SURVIVOR — but it tested a head on *finished* vectors. **`E14-LORA` REOPENED 2026-08-29**; licence closed (stella MIT). Ship-vs-measure fork open, above. |
| 2 | `E10` the shadow | Seven clean-community LoTTE slices, per-question remedy — then **REOPENED on review**; artifact unpinned, now time-boxed. |
| 3 | `harrier` | Closed on undisclosed training data. **stella stands; T1's NO SWAP is final for M8.** |
| 4 | HUPD / patents | Deferred to M9. Trigger (web-crawl data entering training) most likely never fires — see ruling 6. |
| 6 | Qdrant/FineWeb (`D-FINEWEB`) | **Default is EXCLUSION** (Dylan): it "should really prove its value"; web-crawl enters only on a clearly-resolved gain, never a marginal one. |
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
