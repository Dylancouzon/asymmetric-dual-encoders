# M8 status

**Stage: Phase 0 complete; the milestone's main bet has REPORTED, 2026-08-29.** LEDGER **v2** is
live (`m8/LEDGER.md` + `m8/registry.json`). All four noise floors are measured, nine probes have
run, the teacher question is closed, and all owner decisions are ruled. Every result carries a
registration stamp. **No M8 training candidate exists yet. No protected set has been scored. The
reserved four are untouched.**

## `E14-HEAD` — NO SURVIVOR. Both heads HARM; the mechanism was real and the instrument wrong.

| treatment | DENSE | FUSED | bar | registered verdict |
|---|---|---|---|---|
| **LIN** (primary) | **−0.0244** | −0.0024 | +0.0040 | OPTIMIZATION-INADEQUATE |
| **MLP** (control) | **−0.0293** | −0.0042 | +0.0040 | NULL |

~6× the bar in the WRONG direction on dense, all six arms agreeing in sign. **The patch stack is a
measured null** (`R0N` vs `R0`: −0.00001 dense, −0.000015 fused), which is what licenses reading
the rest as a result rather than a defect.

**`lin`'s label, stated correctly** (revised after review — the first write-up overreached):
LIN is a strong negative for the registered 2,500-step configuration but remains
OPTIMIZATION-INADEQUATE for the METHOD-level question. MLP met the adequacy heuristic and was also
harmful, which makes a generic undertraining story less plausible without resolving LIN at an
adequate budget. A 5,000-step reported set with a paired 5,000-step R0N is the disciplined
resolution if it is ever wanted.

**The mechanism control, stated correctly.** Across all seeds and both components the head reduced
bag-query nDCG LESS than teacher-query nDCG (LIN +0.0091, MLP +0.0075; all twelve values positive).
That is descriptive evidence of **relative** alignment toward the co-trained bag representation. It
does NOT show an absolute bag benefit — both absolute bag gains are negative — and the earlier
"buys bag-reachability by destroying information" claim is **withdrawn**. Teacher-style queries do
lose more than bag queries (−0.031 vs −0.022), which is a first, descriptive read on whether the M9
pair can share a document side, and an input to `E14-LORA`'s still-unwritten bar.

**Scope, binding:** a null here is WEAK evidence about `E14-LORA` and may NEVER be written as
closing E14. It removes the strongest argument FOR buying it. CQADupStack-family only.

**Also banked:** fusion absorbs ~10× of the dense degradation (−0.0244 → −0.0024), an independent
read on how much of the fused system's quality is BM25 rather than the table.

---

## Owner rulings — all five decided 2026-08-29 (reasoning in LEDGER §15)

| # | question | ruling |
|---|---|---|
| 1 | **E14** doc-side co-adaptation | **Measure it small first.** Two staged probes: `E14-HEAD` (registered, **amended after review**, not yet run) and `E14-LORA` (refused, TBD bar). The doubled 10.12M pre-encode, the stella licence check and any C2 redefinition are **NOT** authorised — **C2 and E11 stand unchanged**. |
| 2 | **E10** the shadow | **Seven clean-community LoTTE slices, per-question remedy** (~14,034 queries). Subforums **rejected — not on contamination, which they passed, but on correlation with the exam**: two of the reserved four *are* CQADupStack. The shadow is a check, **never** a selection surface. |
| 3 | **harrier** | **Closed, on undisclosed training data.** **stella stands; T1's NO SWAP is final for M8.** |
| 4 | **HUPD** / patents | **Deferred to M9** — safe because no patent text exists anywhere in M8's training mix. **Trigger:** settle before any web-crawl data (`D-FINEWEB`) enters training. |
| 5 | **E12** LR-dense | **Published numbers only, labelled.** The report may never state or imply a head-to-head on our data. |

**Nothing is blocked on Dylan.**

---

## The main bet: `E14-HEAD` is registered but its premise had to be repaired first

An adversarial review of the design (Codex, **before any arm ran**) returned three BLOCKERs, all
reproduced independently and all adopted. LEDGER §15 has the full disposition; the three that
matter:

1. **The registration required a nonlinear head on a premise this ledger already refuted.** A
   *renormalized* linear doc-side map is **not** absorbable into the table — the score carries a
   per-document `1/|Md|` that cannot move into a shared row, and §6's D1 entry recorded rank
   agreement 1.000 without renormalization and **0.000 with it**. So **`LIN` is now the primary**
   (1.05M params, better conditioned) and **`MLP` is its nonlinearity control**.
2. **Zero-init gives `normalize(d)`, not `d`.** The cached vectors are only approximately
   unit-norm — 0.36% exactly 1, max `|norm−1|` 4.8e-05 over 100,000 pool rows. New comparator
   **`R0N`** (same patched path, head frozen at identity), which doubles as an end-to-end null on
   the patch stack.
3. **The lr ladder would have observed the endpoint before selecting.** `train.run()` evaluates
   `eval_components` — which include *both* DENSE endpoint components — every 500 steps *and* once
   more unconditionally. The ladder is now **dev-blind by construction**, on a disjoint tuning seed.

Also registered: a **mechanism control** (headed documents against frozen *teacher* queries, so a
win can be attributed to bag-reachability rather than generic supervised doc adaptation), a
**step-adequacy** continuation whose plateau rule makes an under-trained null report
UNINFORMATIVE, streamed head application (a materializing patch needs ~21.4 GB on HotpotQA), and
sha256 provenance binding. **Bar unchanged at 0.0040**; multiplicity is now Holm across the two
treatments.

`m8src/e14_head.py` holds both heads and the verbatim `infonce` copy — bit-identical to `m7src`'s
on all six branch cases, with the false-negative mask deliberately kept in **raw teacher space**
(the head would otherwise control which negatives are masked out of its own loss, and masking more
makes InfoNCE trivially smaller). **The training driver is not written.**

---

## Results

**The crossed B × A floor — the B leg costs about what the A leg costs** (§23,
`results/m8_noise_floor_crossed.json`). Nine cells, four newly trained. On the out-of-domain
macro **σ_B 0.00103 against σ_A 0.00106**, σ_chain 0.00153, and the standing 0.0040 carries
**~6.5%** type-I against a fresh null difference between two independent chains.
**Bars in force are unaffected** — B3 and E14-HEAD read *A-leg-only* arms, whose null is σ_A alone
(`2 × 1.693 × 0.00106 = 0.0036`, under the planning minimum). For a **B-leg-varying** probe the
formula would give **0.00519**: reported, **not adopted**. The §4.4 gap narrows to a
pool-varying null — all nine cells share one pseudo-query pool.
*The claim that motivated this design was withdrawn before it ran*: the aliased diagonal is
`B_s + A_s + e`, which already has the chain variance, so it is unbiased-but-noisy, not
anti-conservative. Confirmed on this data — the diagonal's range sits at 0.43× its own
expectation, inside the [0.25, 1.96] interval a K=3 range spans.

**B6-pre — PASSED for a NONLINEAR head** (§22, `results/m8_b6_pre_mlp.json`). 3,426 nodes, GELU
exports as a plain `Erf`, zero custom-domain ops, parity 0.99999988. E3's condition is met, so
`E14-HEAD`'s output is shippable in principle. **Both passes used near-identity weights**, so the
registry requires the *actual trained head* to be re-exported before anything is called shippable.

**B3 — Phase A is not meaningfully pair-starved** (`results/m8_b3_decision.json`). A 4× dose moves
dense +0.00135 and fused +0.00369 against a 0.0040 bar; verdict **UNINFORMATIVE**, which the
registration defined in advance as the strongest no-starvation evidence this probe can produce.
The actionable number: reaching the bar would need **~17.6× the pool (~5.9M pairs)**. M7's entire
MS MARCO addition was 490K. **This is the argument for spending M8 on capacity.**

**T1 — NO SWAP** (§21). granite-r2 −0.052 [−0.066, −0.039], gte-modernbert −0.109 [−0.123,
−0.094], both CI-resolved. The tower again fails to order the table.

**B2 — the KL term is dead** (§19): teacher target median entropy 4.73e-07 nats; the shipped table
ranks the positive first 99.75% of the time. **B7 — PASSED** (§18): 65,536 rows in 51 iterations /
10 s / 4.4 GB. **B17 — DISOWNED** (§20). **§17 — fragmentation survives every single-dataset
exclusion**; the query-length claim was ArguAna-only and is withdrawn.

**A near-miss, found by review.** `work/dev/cqadup-{android,english}.json` held the complete
corpora **and qrels** of two reserved confirmatory sets. **Nothing scored them.** Now a protected
kind (§15).

---

## Infrastructure

| item | artifact |
|---|---|
| LEDGER v2 + machine-readable registry (27 probes) | `m8/LEDGER.md`, `m8/registry.json` |
| Executable ship rule, B3's verdict as code, a test per branch | `m8src/decide.py`, `b3_decide.py`, `test_decide.py`, `test_b3_decide.py` |
| Guards G1/G2, hardened against four concrete routes | `m8src/paths_guard.py`, `probe_guard.py`, `test_guards.py` |
| Rule audit — diffs each result's registry **from git at that result's commit** | `m8src/rule_audit.py` |
| E10 remediation + the protected-query filter (one module, because one process gets one `claim()`) | `m8src/protected_filter.py`, `freeze_lotte.py` |
| E14 heads, the verbatim loss copy and its equivalence proof | `m8src/e14_head.py` |
| Solver, teacher screens, entropy probe, all four floors, dose-curve runner | `m8src/blockcg.py`, `teacher_screen.py`, `b2_entropy.py`, `noise_floor.py`, `fused_floor.py`, `b3_pool.py` |

`./run_m8_tests.sh` runs everything. `m8/CODEMAP.md` carries 21 pitfalls, several earned this week.

## Next

`m8/NEXT-SESSION.md` has the ordered worklist. In short: `E14-HEAD`'s driver (implement from the
**amended registry row**, not the superseded brief), then run `E10-REMEDY` and regenerate the fit
list, then `D-FINEWEB`.

## File contract

| file | contract | read when |
|---|---|---|
| `m8/STATUS.md` | This. Stage, rulings, results. | always, first |
| `m8/LEDGER.md` | Binding protocol: rules, bars, verdicts, amendments. | before any decision |
| `m8/registry.json` | The executable half of §9. `probe_guard` reads this, not the prose. | before any run |
| `m8/NEXT-SESSION.md` | Remaining worklist. | at session start |
| `m8/RESULTS.md` / `EXPLORED.md` / `CODEMAP.md` | runs / closed avenues / modules and pitfalls. | as needed |
| `research/m8-planning/*` | Archival record: reviews, literature sweep, challenger Specs. | on demand |

Every number carries an artifact pointer; no file restates another; a future session cold-starts
from STATUS + LEDGER + registry alone.
