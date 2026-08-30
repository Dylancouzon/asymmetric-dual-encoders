# M8 findings — the durable record

**M8 closed 2026-08-30 as a MEASUREMENT. No candidate, no release, no confirmatory access spent;
the reserved four are clean for M9.** Twelve probes ran. This file is what M8 is *for*: the map of
where the quality is not, written for the M10 whitepaper and for anyone training the next model in
this harness. Numbers live in `results/m8_*.json`; this is the reading of them.

---

## 1. The question M8 asked, and the answer

**Can a zero-compute query encoder — a lookup table of token vectors, no transformer at query
time — be improved enough to beat a co-adapted system, with the document tower held frozen?**

**No, on this evidence.** M7 shipped at avg-6 **0.4339** against LightRetriever's
`LR-dense-pertask` **0.4583** — a CI-resolved miss of **−0.0243**. Every lever M8 measured moved the
dev endpoint by **0.000–0.005**, and M7's whole post-gate lever programme transferred to the six at
**0.000 ± 0.005**. The gap is roughly five times the size of the largest effect anything in this
class produces.

**The single most useful sentence for the next model:** the deficit is not in the table's
*resolution*, its *placement*, or its *training target*. Those were each tested and each failed.
What was never tested at capacity is **co-adaptation of the document side** — and that is exactly
what the system we lost to does.

---

## 2. What is closed, and what closed it

*Headline reading only. **`m8/EXPLORED.md` is the canonical register** — one row per closed avenue
with the condition that would REOPEN it, which is what makes these findings rather than guesses.*

| hypothesis | probe | result | reading |
|---|---|---|---|
| More data | `B3` | +0.00135 at 4× dose; bar needs ~17.6× the pool | direction is POSITIVE but sub-linear; closed against a 0.024 gap, not closed as a direction |
| A better teacher | `T1` | granite-r2 −0.052, gte-modernbert −0.109 | and the tower does **not** order the table: gte has the higher published score and the lower distilled table |
| The distillation objective | `B2` | uniform-bank KL median **4.73e-07 nats** | the shipped objective is **inert** — the table already ranks the positive first in **99.75%** of training queries |
| Finer query vocabulary | `D2-PRE` | all four new-row classes negative; best −0.0028, D2's own −0.0052 | closes **both** segmentation and additive n-gram rows at equal budget |
| Query placement | `VECTOR-PRF` | −0.051 dense, −0.021 fused, negative on **all six** components | train-free post-hoc refinement does not recover the gap |
| The training target | `B8` | doc-centroid −0.167, 50/50 blend −0.003 | aiming at the document manifold is **worse**; the teacher's query encoder already does that mapping |
| A doc-side head | `E14-HEAD` | −0.0244 / −0.0293 | a head on a **finished** vector cannot recover what pooling discarded |
| Licensing | clean-stack tax | +0.0058 | MS MARCO exclusion costs 0.0058 — real, measured, and declined on commercial grounds, not technical ones |

### Never run, and why it matters that they weren't

- **`R-LIST`** — hard-candidate listwise distillation. `B2` triggered it directly: the shipped
  objective is inert, but the `teacher_top200` variant measures **0.777 nats**. So the KL *class* is
  open; only the recipe's degenerate instance is closed. Its class prior is poor (0.000 ± 0.005
  transfer) but it has never been measured.
- **`B10`** (`pool_mode`) — untested, weak prior (+0.0011, CI straddling zero).
- **`E14-LORA`** — authorised by Dylan and **deliberately not run**. What was affordable here was a
  dev-scale LoRA against a shrunken stale negative bank on two CQADupStack components. That is a
  *proxy* for co-adaptation, not a test of it — the same weakness that made `E14-HEAD` uninformative
  about the tower. **M9/M10 should test it with a real training budget or not at all.**
- **`E14-PRE`** — a screen for the above, registered and then cancelled before spending: on
  inspection its probes could not have settled the question. It varies how the tower's output is
  *read* and *mapped*; a LoRA changes what the tower *computes*. Registered, unrun, and the row
  records why (`registry.json`).

---

## 3. Mechanisms worth carrying into the next model

1. **The objective is exhausted, not the architecture.** `B2`'s 99.75% is the number to remember:
   under the shipped recipe the table already ranks the positive first for nearly every training
   query, so the gradient carries almost no information. Any future table training should start from
   **hard candidates**, not a uniform bank. This is the strongest untested lead M8 leaves behind.
2. **A teacher's retrieval quality does not predict its distilled table** (Spearman 0.000 over eight
   candidates, M7; corroborated by `T1`). **Select a teacher on the artifact you will ship**, never
   on the tower's leaderboard row. M7 lost time to exactly this error.
3. **Fragmentation correlates with the gap but is not recoverable through it.** §17b measured the
   table falling 0.050 nDCG behind the teacher per +1.0 subwords/word (t = 4.61). `D2-PRE` then
   *moved* fertility by 0.164–0.176 and the metric did not follow. **A correlated channel is not a
   lever** — this is the cleanest example the project produced, and it belongs in the whitepaper.
4. **Fusion is the real bright spot.** M7 measured fusion at **+0.057** over dense alone, and the
   fused system (0.4911) ties OpenSearch (0.4868). Every table-side lever is worth <0.005; the
   lexical channel is worth ten times that. A zero-compute product should ship fused.
5. **Pooling multiplicity is genuine capacity; almost nothing else on the query side is.**
   `m7_absorb_check` proved per-token scalar weights, centering, whitening and top-PC removal are
   all *absorbable* into the rows — they cannot add capacity. Only multiplicity-dependent pooling
   and new rows can, and `D2-PRE` closed new rows.
6. **The compositional floor is exact, and it is the SUM.** Replacing constituents `a,b` by a phrase
   row `w_a + w_b` leaves the served vector **exactly** unchanged under sqrt pooling; the *mean*
   downweights the phrase ~2× against every other token. Verified algebraically on real rows
   (sum 1.5e-08, mean 0.040) and at full scale (compile reproduces R0 to **−2.8e-06**). Reusable
   whenever anyone adds rows to a table.

---

## 4. Method learnings — the part that generalises past this model

1. **Screen in closed form before spending training chains.** `D2-PRE` cost 96 minutes and saved
   five full chains; `B8` cost ~40 minutes and closed a hypothesis outright. The pattern —
   ridge-solve the table, score the real endpoint, route on a pre-registered numeric rule — is the
   highest-leverage thing in this harness.
2. **A screen must be able to change the decision.** `E14-PRE` was designed, registered, launched
   and then cancelled because its own registered scope limit meant a negative could not close the
   question it gated. **Before building a screen, write down what each outcome would cause you to
   do; if both outcomes lead to the same action, do not build it.**
3. **Register the alternative parameterisation alongside the hypothesis.** `D2-PRE` ran additive
   n-gram arms beside D2's segmentation at equal row budget specifically so a miss could not be
   re-read as "wrong parameterisation." Both lost, which is a far stronger claim than D2 alone.
4. **Exclude the artifact explanations before believing a negative.** For `D2-PRE`: coverage
   (zero-update mass 0.0001–0.001 against a 20% gate), compile fidelity (−2.8e-06), leakage (0/0
   overlap) and λ (interior for 3 of 4 arms) were each measured. A negative with the alternatives
   still open is not a finding.
5. **A measured floor does not freeze a bar — an amendment does.** `B8`, `R-LIST` and `B10` sat
   unrunnable for the entire milestone because their bars still read `TBD-noise-floor` long after
   the floor was measured. The fallback path was unreachable and the pre-committed exit could never
   have fired, which made one lever look like the only route when it was merely the only *permitted*
   one. **When a floor lands, sweep every row whose bar reads TBD in the same commit.**
6. **Adversarial review earns its cost, including against your own design.** Reviews in this
   milestone caught: a five-fold cross-fit whose folds all scored the same queries (making a
   sign-agreement condition nearly automatic), an arm algebraically identical to another arm, an
   arm-dependent ridge denominator that would have picked winners, and a registry row contradicting
   its own ledger's physics.
7. **Fixture data must come from the real distribution.** A pooling canary built from two-token
   queries scored 0.0 for both the right and the wrong init — a phrase that is a query's whole
   content normalizes them to the same vector. It could never have failed. Context tokens are the
   mechanism; without them the test asserts nothing.
8. **Match the dtype of the artifact you are comparing against.** Every cached vector here was
   encoded with fp16 weights. Loading the tower in fp32 for a comparison would have made the
   "incumbent" arm not the incumbent — a wrong number, not a crash.
9. **Dev reuse is now counted, not recalled:** **494** cumulative in-training dev evaluations across
   M7+M8 (`results/m8_dev_reuse_count.json`). Nested selection protects the *reserved* sets; it does
   not make the out-of-domain macro a fresh surface. Quote this beside any dev-read verdict.

---

## 5. What M8 leaves the next milestone

- **Reserved four unspent and unscored** (FEVER, DBpedia-entity, cqadup-android, cqadup-english).
  One confirmatory access remains.
- **A calibrated noise model**: A-leg σ ≈ 0.00106, B-leg σ ≈ 0.00103, chain σ = 0.00153 → the
  0.0040 planning minimum binds on A-leg arms, 0.00519 on chain-varying arms (§4.7, §23). Still
  unbounded: a **pool-varying** lever.
- **Reusable instruments**: `blockcg.py` (Gram-free ridge — 65,536 rows in 10 s where a dense Gram
  would need 34 GB), `probe_guard.py` (no bar, no run), `decide.py`, `noise_floor.py`,
  `d2_pre.py`'s serving-exact encoder and bag machinery, `dev_reuse_m8.py`.
- **`m8/CODEMAP.md`** — the pitfalls, which are the harness's real inheritance.
- **A recommendation for M9/M10:** ship **fused**; start table training from **hard candidates**
  (`R-LIST`); and treat document-side co-adaptation as a **budgeted experiment**, not a proxy.
