# M8 status

**Stage: Phase 0 complete, 2026-08-29.** LEDGER **v2** is live (`m8/LEDGER.md` +
`m8/registry.json`), gated by seven adversarial reviews. All three noise floors are measured, six
probes have run, the teacher question is closed, and all five owner decisions are ruled. Every
result carries a registration stamp. **No M8 training candidate exists yet. No protected set has
been scored. The reserved four are untouched.**

---

## Owner rulings — all five decided 2026-08-29 (reasoning in LEDGER §15)

| # | question | ruling |
|---|---|---|
| 1 | **E14** doc-side co-adaptation | **Measure it small first.** Specified as two staged probes: `E14-HEAD` (runnable) and `E14-LORA` (refused, TBD bar). The doubled 10.12M pre-encode, the stella licence check and any C2 redefinition are **NOT** authorised — **C2 and E11 stand unchanged**. |
| 2 | **E10** the shadow | **Seven clean-community LoTTE slices, per-question remedy** (~14,034 queries). Subforums **rejected — not on contamination, which they passed, but on correlation with the exam**: two of the reserved four *are* CQADupStack. Re-screen after remedy; the shadow is a check, **never** a selection surface. |
| 3 | **harrier** | **Closed, on undisclosed training data** — no comparison design repairs not knowing what a teacher has read. **stella stands; T1's NO SWAP is final for M8.** stella-1.5B remains unscreened, no new ruling needed. |
| 4 | **HUPD** / patents | **Deferred to M9** — safe because no patent text exists anywhere in M8's training mix. **Trigger:** settle before any web-crawl data (`D-FINEWEB`) enters training; that row carries the note. |
| 5 | **E12** LR-dense | **Published numbers only, labelled.** The report may never state or imply a head-to-head on our data. |

**Nothing is blocked on Dylan.**

---

## Results

**B3 — Phase A is not meaningfully pair-starved** (`results/m8_b3_decision.json`). Twelve arms:
nested real-pair fractions {0.25, 0.50, 0.75, 1.00} × three seeds, at fixed updates, batch,
negatives and Phase-B checkpoint — 1,280,000 draws in every arm, so only the count of *distinct*
pairs moves.

| contrast | dense | fused | meets 0.0040? |
|---|---|---|---|
| **4× dose**, 1.00 vs 0.25 | +0.00135 | +0.00369 | **no, neither** |
| primary, 1.00 vs 0.50 | +0.00112 | +0.00201 | no |
| 1.00 vs 0.75 | **−0.00107** | +0.00076 | no |

Verdict **UNINFORMATIVE**, which the registration defined in advance as the strongest
no-starvation evidence this probe can produce. **The actionable number**: the slope is +0.00097
dense per doubling, so reaching the bar needs **~17.6× the pool (~5.9M pairs)**. M7's entire
MS MARCO addition was 490K, under 1.5×. **This is the argument for spending M8 on capacity.**
*Scope: `p35b-2m` had already distilled on every training query, so this measures pairs in Phase A
given B absorbed them, and says nothing about B-side levers such as `D-FINEWEB`.*

**T1 — NO SWAP** (`results/m8_t1_decision.json`, §21). First measurements of two teachers M7 closed
on arithmetic: granite-r2 −0.052 [−0.066, −0.039], gte-modernbert −0.109 [−0.123, −0.094], both
CI-resolved at 5–11× the swap penalty. The frame reproduces M7's own number (0.3438 vs 0.3439)
through a new solver, init builder and fit list. The tower again fails to order the table.

**B2 — the KL term is dead, measured on both sides** (§19). Teacher target median entropy
**4.73e-07 nats** against a ln(32) ceiling; the **shipped table ranks the positive first 99.75%** of
the time, so the term's own median value is **1.08e-07 nats**. Does *not* say a listwise objective
wins — that is `R-LIST`, whose bar is unfrozen.

**B7 — PASSED** (§18). Gram-free preconditioned solver: 65,536 rows in 51 iterations / 10 s /
4.4 GB where the dense fp64 Gram would be 34 GB. Reopens the vocabulary door.

**B6-pre — PASSED** (`results/m8_b6_pre.json`). Teacher + doc-side head exports to **one** ONNX
file, 3,415 nodes, zero custom-domain ops, parity 0.99999994. D1 survives E3. **Caveat: run with
`--head linear` only — a nonlinear head is unproven and `E14-HEAD` now depends on it.**

**Noise floors — all three measured.** Dense 0.00095–0.00227, fused 0.00059–0.00066, B-leg
0.00070–0.00218. **The B-leg comparison I first drew is withdrawn**: at K = 3 the statistic is the
sample **range** (CV 0.525) — two identical-noise experiments differ by ≥2× **40%** of the time and
P(R_B ≤ R_A) is exactly **0.500**. Two design faults recorded: the chain seed **aliases both legs**
(making the range an *under*-estimate, the anti-conservative direction), and the pool is held fixed
so it does not bound pool-varying levers. **Bars stand as a pre-registered convention, not a
bound**: 0.0040 everywhere except `int8/mean` worst-group and OOD at 0.004369.

**B17 — registered branch fired and was DISOWNED** (§20); it had measured its own 957-query fit
set. **§17 — fragmentation survives every single-dataset exclusion** (worst case t = 3.28); my own
query-length claim was ArguAna-only and is withdrawn. **Fit list**: 337,981 kept of 338,076, all 95
removals from M9-reserve — regenerate once the LoTTE survivors land.

**A near-miss, found by review.** `work/dev/cqadup-{android,english}.json` held the complete corpora
**and qrels** of two reserved confirmatory sets; any `devsuite.load(...)` would have scored one
silently. **Nothing scored them.** Now a protected kind (§15).

---

## Infrastructure

| item | artifact |
|---|---|
| LEDGER v2 + machine-readable registry (26 probes, 10 runnable) | `m8/LEDGER.md`, `m8/registry.json` |
| Executable ship rule, and B3's verdict as code with a test per branch | `m8src/decide.py`, `b3_decide.py`, `test_decide.py`, `test_b3_decide.py` |
| Guards G1/G2, hardened against four concrete routes | `m8src/paths_guard.py`, `probe_guard.py`, `test_guards.py` |
| Rule audit — diffs each result's registry **from git at that result's commit** | `m8src/rule_audit.py` |
| Gram-free solver, teacher screens, entropy probe, all three floors, dose-curve runner | `m8src/blockcg.py`, `teacher_screen.py`, `b2_entropy.py`, `noise_floor.py`, `fused_floor.py`, `b3_pool.py` |

`./run_m8_tests.sh` runs everything. `m8/CODEMAP.md` carries 18 pitfalls, several earned this week.

---

## Next

1. **`E14-HEAD`** — runnable, bar frozen at 0.0040. The milestone's main bet, and the review point
   that matters most before it runs.
2. **`E10-REMEDY`** — drop leaked queries *and* near-duplicate documents from the seven slices,
   re-screen requiring **zero** hits, hash-pin, then regenerate the fit list.
3. **B6-pre with `--head mlp`** — cheap, and it gates whether `E14-HEAD`'s output is shippable.
4. **Crossed B×A seed design** — six A legs (~30 min); the cheapest way to turn the floor
   convention into a bound.
5. Still open: `m8src/freeze.py` and `final_run.py` and their suites (§4.4 gap list, weeks out);
   `D-FINEWEB` prep (bar unfrozen; carries the patent trigger).

## File contract

| file | contract | read when |
|---|---|---|
| `m8/STATUS.md` | This. Stage, rulings, results. | always, first |
| `m8/LEDGER.md` | Binding protocol: rules, bars, verdicts, amendments. | before any decision |
| `m8/registry.json` | The executable half of §9. `probe_guard` reads this, not the prose. | before any run |
| `m8/NEXT-SESSION.md` | Remaining worklist. | at session start |
| `m8/RESULTS.md` / `EXPLORED.md` / `CODEMAP.md` | runs / closed avenues / modules and pitfalls. | as needed |
| `research/m8-planning/*` | Archival record: reviews, literature sweep, challenger Specs. | on demand |

`m8/PLAN-DRAFT.md` was **deleted** 2026-08-29 — after 17 amendments it was a second source of truth
that disagreed with the binding one. Git history has it.

Every number carries an artifact pointer; no file restates another; a future session cold-starts
from STATUS + LEDGER + registry alone.
