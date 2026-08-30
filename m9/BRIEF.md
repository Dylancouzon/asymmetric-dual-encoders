# M9 planning brief — read this before planning

**`instructions-m9.md` is the mandate** (teacher rule, student shortlist, bars, pair economics,
costs). It is not restated here. **This file is the context around it**: what M7 and M8 established,
what M10 will demand of M9's artifact, and the questions the planning session has to answer.

---

## 1. Where M9 sits

| milestone | artifact | state |
|---|---|---|
| M7 | **zeo** — zero-compute lookup table, stella-derived, int8, no transformer at query time | **frozen, verified releasable** (`m7/FREEZE.json`; artifact sha `a7007b1a…` re-verified 2026-08-30) |
| M8 | v2 of the table | **CLOSED as a measurement.** No candidate. See `m8/FINDINGS.md` |
| **M9** | **nano** — LEAF-style distilled small query tower against a frozen document index | **next** |
| M10 | **zeo + nano release, whitepaper, ONNX port (incl. the document model), fastembed integration** | after M9 |

**The product is the PAIR, not a winner.** Dylan, 2026-08-30: M7 is releasable *paired with a good
low-compute model*. So M9 is not trying to beat zeo — it is building the other point on a
quality-vs-query-cost frontier. That reframes what "success" means and it should shape the plan:
the deliverable is a defensible frontier with honest costs, not a leaderboard row.

---

## 2. What M10 will demand of M9's artifact — design constraints, not later problems

1. **ONNX-exportable, and parity-verified.** M10 ports both models *and the document model*.
   `B6-pre` (§22) exported the document graph at opset 17 with zero custom-domain ops and parity
   min-cosine 0.99999994 — **but on near-identity weights**, so the real artifact has never been
   exported. Whatever backbone M9 picks must export cleanly; check this **before** committing to it,
   not after training. A backbone needing custom ops is disqualified on M10 grounds alone.
2. **fastembed-integrable.** Its conventions (tokenizer packaging, pooling, output naming) should be
   checked at student-selection time.
3. **Size budget.** zeo's query-side system is capped at 233 MB (E7). nano needs its own stated
   budget; `instructions-m9.md` expects ~3–5 ms/query plus model load. Report the same three cost
   rows M7 used so the frontier is comparable: **query asset ≠ document index ≠ hydration.**

---

## 3. What M8 established that bears on M9

Full record in `m8/FINDINGS.md`; the four that change M9's plan:

1. **Ship fused.** Fusion is worth **+0.057** over dense alone — ten times any table-side lever, and
   M7's fused system (0.4911) ties OpenSearch (0.4868). The frozen operator is `convex0` w=0.8.
   M9 should evaluate fused from the start rather than bolting it on.
2. **Select the teacher on the artifact you will ship.** A teacher's own retrieval quality does
   **not** predict its distilled table (Spearman 0.000 over eight candidates). `T1`'s NO SWAP does
   **not** transfer to M9 — it is a fact about distilled *tables*, and nano is a *tower*, where LEAF
   reports 97.1–98.6% retention. **M9 must run its own teacher screen on its own artifact.**
   Inheriting T1's answer would repeat M7's original error in reverse.
3. **The distillation objective may be exhausted before the architecture is.** `B2` measured the
   shipped objective at 4.73e-07 nats — the table already ranks the positive first for **99.75%** of
   training queries, so the gradient carries almost nothing. The `teacher_top200` variant measures
   **0.777 nats**. A tower has far more capacity than a table, so this may bite less — but **start
   from hard candidates rather than a uniform bank**, and measure the objective's entropy early.
   This is the strongest untested lead M8 leaves behind (`R-LIST`, never run).
4. **Document-side co-adaptation is the one untested high-capacity lever.** LightRetriever *trains*
   its document encoder; M7/M8 fit to a frozen tower, and `LR-dense-pertask 0.4583` was set with a
   co-adapted document side. `E14-LORA` is authorised by Dylan (stella is MIT, zero marginal product
   cost since the user re-indexes anyway) and was **deliberately not run** — what M8 could afford was
   a proxy, not a test. If M9 or M10 tests it, budget it properly. `E14-HEAD`'s −0.0244 does **not**
   close it: a head on a finished vector cannot recover what pooling discarded.

---

## 4. Protocol state M9 inherits

- **Reserved four unspent and unscored** — FEVER, DBpedia-entity, cqadup-android, cqadup-english.
  **One confirmatory access.** M8 spent none. Guarded by `m8src/paths_guard.py`; note the guard is
  in-process and **cannot** constrain an external reviewer — every review brief needs a written
  read-exclusion, and the review log must be grepped for reserved-set reads afterwards (2026-08-29
  incident, `m8/LEDGER.md` §15).
- **Frozen comparator vectors are IRREPLACEABLE.** `results/perquery.json` holds leaf-ir-asym 0.5155,
  mdbr-leaf-ir 0.5123, arctic-m 0.5264, frozen while the Mac caches still existed. They cannot be
  regenerated on this box. Do not overwrite that file.
- **Noise model, measured:** A-leg σ ≈ 0.00106, B-leg σ ≈ 0.00103, chain σ = 0.00153 → bars **0.0040**
  (A-leg-only) and **0.00519** (chain-varying), by `bar = max(0.0040, 2 × floor)`. **A pool-varying
  lever is still unbounded** — measure its floor before any bar reads it.
- **Dev reuse is counted: 494** cumulative in-training dev evaluations across M7+M8
  (`results/m8_dev_reuse_count.json`). The out-of-domain macro is *not* a fresh surface. M9 should
  decide deliberately whether to inherit this dev suite or build headroom into its selection.
- **Licensing:** clean stack; MS MARCO permanently excluded from the release stack (priced at +0.0058 **[−0.0015, +0.0131]** on the six and
  **−0.0030** [−0.0066, +0.0003] on dev — **unresolved on both and opposite in sign**, so it is not a
  clean available gain; declined commercially). CC BY-SA sources approved with model-card attribution. Vendor
  rule as amended in `CLAUDE.md`. `freeze.assert_releasable` enforces it.

---

## 5. Harness inventory — what exists, so M9 builds none of it again

| need | module |
|---|---|
| Gram-free ridge solve (65,536 rows in 10 s; a dense Gram would need 34 GB) | `m8src/blockcg.py` |
| No bar, no run — registration gate + result stamping | `m8src/probe_guard.py` |
| Ship predicate / confirmatory rule, with self-test | `m8src/decide.py` |
| True-seed-null floors, crossed B×A design | `m8src/noise_floor.py`, `fused_floor.py` |
| Serving-exact query encoder + bag machinery (reproduces the released `QueryTable` path to **3.7e-08**, and R0's crossed-floor cell to all 16 digits — `results/m8_d2_pre.json` `conformance`) | `m8src/d2_pre.py` |
| Dev-reuse counter | `m8src/dev_reuse_m8.py` |
| Protected-path guard | `m8src/paths_guard.py` |
| Frozen fusion operator, BM25 caches | `m7src/fusion.py`, `select_fusion.py` |
| Freeze + release-licence enforcement | `m7src/freeze.py` |

**`m8/CODEMAP.md` is the highest-value file in the repo for anyone writing new code here** — 24
pitfalls, each one a real incident. Read it before writing a probe.

---

## 6. Method learnings worth applying (from `m8/FINDINGS.md` §4)

- **Screen in closed form before spending training chains.** `D2-PRE` cost 96 min and saved five
  full chains.
- **Before building a screen, write down what each outcome makes you do.** If both outcomes lead to
  the same action, do not build it — `E14-PRE` was cancelled for exactly this.
- **Register the alternative parameterisation alongside the hypothesis**, so a miss cannot be
  re-read as "wrong parameterisation."
- **Exclude the artifact explanations before believing a negative** (coverage, fidelity, leakage,
  regularisation) — each measured, not assumed.
- **When a floor lands, sweep every row whose bar reads TBD in the same commit.** `B8`, `R-LIST` and
  `B10` remained unreachable until a late amendment (2026-08-29) froze their bars from a floor
  measured that same day — which had made one lever look like the only route when it was merely the
  only *permitted* one. `B8` then ran and missed; the other two never ran.
- **Adversarial review (Codex/Fable) is a routine instrument, not a ceremony** — standing grant in
  `CLAUDE.md`. Brief it to break your design, with a read-exclusion.

---

## 7. Questions the planning session must answer

1. **Teacher.** Re-probe stella in M9's own frame against the shortlist — on the *tower*, since that
   is the artifact. If the pick differs from the table line's, that is **two document indexes**, not
   one index with two query paths. `instructions-m9.md` sets the tie-break: **prefer the pair**,
   broken only on a CI-resolved loss.
2. **Student backbone** — ≤35M, permissive clean vendor, **and ONNX/fastembed-exportable** (§2).
   Re-derive the shortlist; the original draft's rationale went stale at the 2026-08-26 teacher swap.
3. **Objective** — uniform bank or hard candidates? See §3.3. Measure the entropy early.
4. **Dev suite** — inherit M7's (494 reads deep) or build fresh headroom?
5. **Does M9 test `E14-LORA`**, or does that go to M10 with the release work?
6. **What does the frontier look like** — which points does the whitepaper need, and does M9 owe any
   measurement that only makes sense alongside zeo?
7. **Confirmatory design** — one access, four reserved sets. What is the candidate, what is the
   comparator, and what is the pre-registered rule? Nothing is scored until that is written down.
