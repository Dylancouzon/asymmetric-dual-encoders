# M8 status

**`D2-PRE` ran 2026-08-29: DO NOT AUTHORISE. `D2` is CLOSED and so is the additive n-gram class**
(§24). Ten probes have run; all returned nulls, negatives or instrument reads. **No M8 candidate
exists; no protected set has been scored; the confirmatory access is unspent.**

**Three more probes ran 2026-08-29, all NO SURVIVOR:** `D2-PRE` (§24), `B8` (§25, doc-centroid
target −0.167), `VECTOR-PRF` (§26, −0.051 dense / −0.021 fused, negative on all six components).
**Next: `R-LIST`, then `B10`** — the last two exit preconditions. Both — all three were REFUSED by `probe_guard` until
2026-08-29 because their bars still read `TBD-noise-floor` long after the floor was measured, which
made the fallback path unreachable and the pre-committed exit unable to fire. Bars now frozen (§15).

## Open for Dylan — one question, and it must be answered before any number

**Does M8 ship a better SYSTEM, or must it ship a better TABLE?** `E14-LORA` is reopened on your
ruling (below), but `E11`/§5.4 say a document-side win is not a qualifying v2 *table*, and `C2`
falls to its registered `teacher_swapped` branch. Not decidable after a result.

Also standing, overridable before the numbers exist: the **pre-committed exit** — M8 does not spend
its access without a credible release candidate; it closes as a measurement and the reserved four
stay clean for M9 (P(ship) under `m7_repeat` ≈ 0.002). It cannot fire until `D2`, `B10`, `B8` and
`R-LIST` have all missed, plus a re-run of CLAUDE.md's standing directive.

## The plan

**`D2` is closed.** `D2-PRE` scored all four new-row classes NEGATIVE out-of-fold against +0.00519 —
best `add_word` −0.0028, D2's own segmentation −0.0052 (§24, `results/m8_d2_pre.json`). Not a
coverage, compile, leakage or λ artifact; all four were checked and excluded. Zeroing 23,601 of
35,014 rows moved the result 0.0002, so the added rows are inert rather than under-trained. The
registered reversal fired (additive > segmentation by 0.0024), so **both** classes close and a D2
miss cannot be re-read as the wrong parameterisation.

Remaining order: ~~`B8`~~ and ~~`VECTOR-PRF`~~ **both done, NO SURVIVOR (§25, §26)** → **`R-LIST`** (`B2` triggered it directly: `teacher_top200` is
0.777 nats, so the KL class is NOT closed) → **`B10`** (`pool_mode`, weak prior +0.0011). Then
re-run CLAUDE.md's standing directive, and only then is the exit eligible.

**What the three negatives say jointly.** `D2-PRE` made the query's vocabulary finer; `VECTOR-PRF`
moved the query vector onto the document manifold; `B8` re-aimed the training target at that
manifold. All three failed. **The query-side gap is not a vocabulary-resolution problem, not a
query-placement problem, and not a target-design problem.** What remains untested at capacity is
document-side co-adaptation (`E14-LORA`) — which is also the one thing LightRetriever does and M7/M8
do not (§15).

Worklist: `m8/NEXT-SESSION.md`. Protocol: `m8/LEDGER.md`. Bars: `m8/registry.json`.

## `E14-LORA` reopened — and it may be the milestone's real fork

Dylan, 2026-08-29: *"most people are not currently using stella. I'm not against LoRA on the
document tower"* + *"stella is good for derived weights, no license blocker."* Both premises of the
refusal are gone — the user re-indexes with stella either way, so co-adaptation has **zero marginal
product cost**, and MIT covers derived weights.

**Why it matters: LightRetriever trains its document encoder** (`research/lightretriever.md:19,23,382`).
`LR-dense-pertask 0.4583` — M7's missed bar — was set with a co-adapted document side while M7/M8
fit a table to a frozen tower. **M8 has been solving a harder problem than the system it is
benchmarked against.** Best available explanation for the flat table-side levers; unmeasured.
`E14-HEAD` does not settle it — a head on a *finished* vector cannot recover what the tower
discarded, and that scope limit was registered before any arm ran. Staging binding: dev-scale on the
two OOD components against their own re-encoded corpora before any 10.12M pre-encode.

## What the audit found

| probe | result | closes |
|---|---|---|
| `B3` | +0.00135 dense at 4× dose; bar needs ~17.6× the pool | data volume |
| `E14-HEAD` | −0.0244 / −0.0293 dense, ~6× the bar the wrong way | the doc-side **head** — not the LoRA |
| `T1` | −0.052 / −0.109, CI-resolved | teacher swap |
| `B2` | uniform-bank median entropy 4.73e-07 nats | the recipe's uniform-bank KL — **not the KL class** (`teacher_top200` is 0.777 nats; `R-LIST` stays live) |
| M7 clean-stack tax | +0.0058, miss survives it | licensing as the explanation |
| `B7` · `B6-pre` · `NF` ×4 | PASS / floors | instrument only — and they gate `D2` |

`D2` was registered by the audit, then **amended after review**: the "compositional init floor" was
the *mean* of constituent rows and is not a floor — the **sum** is, exactly; the coverage gate had a
pool-expansion clause that would have voided the bar; and the exit fired too early. Details in §15.

## Rulings in force (§15)

| question | ruling |
|---|---|
| `E14` doc-side | `E14-HEAD` NO SURVIVOR (head on finished vectors). **`E14-LORA` REOPENED**; licence closed. |
| `E10` shadow | Seven LoTTE slices, per-question remedy — then **REOPENED on review**; artifact unpinned, time-boxed. |
| `harrier` | Closed on undisclosed training data. stella stands; T1's NO SWAP final for M8. |
| `D-FINEWEB` | **Default EXCLUSION** — must "really prove its value"; web-crawl only on a clearly-resolved gain. |
| HUPD / patents | Deferred to M9; its web-crawl trigger now most likely never fires. |
| `E12` LR-dense | Published numbers only, labelled. No head-to-head may be stated or implied. |
| Reserved sets | Read by an external reviewer, **ruled fine**; access intact. Briefs now carry a read-exclusion. |

## Infrastructure

| item | artifact |
|---|---|
| Protocol + machine-readable registry (30 probes) | `m8/LEDGER.md`, `m8/registry.json` |
| Ship rule and probe verdicts as code | `m8src/decide.py`, `b3_decide.py`, `e14_decide.py` (+ suites) |
| Guards G1/G2 · rule audit (diffs each result against the registry at its own commit) | `m8src/probe_guard.py`, `paths_guard.py`, `rule_audit.py` |
| Solver, teacher screens, entropy probe, floors, dose curve | `m8src/blockcg.py`, `teacher_screen.py`, `b2_entropy.py`, `noise_floor.py`, `fused_floor.py`, `b3_pool.py` |
| E10 remediation + protected-query filter | `m8src/protected_filter.py`, `freeze_lotte.py` |

`./run_m8_tests.sh` runs everything. Pitfalls: `m8/CODEMAP.md`. **Scope discipline in force:**
`freeze.py`/`final_run.py` wait for a candidate; no new floors, guards or machinery until a capacity
lever has a number. **Missing:** G8's dev-reuse counter.

## File contract

| file | contract |
|---|---|
| `m8/STATUS.md` | This. Stage, open questions, rulings. Read first. |
| `m8/LEDGER.md` | Binding protocol; amendments in §15. Read before any decision. |
| `m8/registry.json` | The executable half of §9 — what `probe_guard` reads. Read before any run. |
| `m8/NEXT-SESSION.md` | Remaining worklist. |
| `RESULTS.md` · `EXPLORED.md` · `CODEMAP.md` | runs · closed avenues · modules and pitfalls. |
| `research/m8-planning/*` | Archival: reviews, literature, specs. |

**G10: keep these tight.** One fact, one home; no file restates another. A cold session starts from
STATUS + LEDGER + registry alone.
