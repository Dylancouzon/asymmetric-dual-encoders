# M10 — nano v2: the retry, built around coverage (mandate)

*Written 2026-09-01 by the planning session on Dylan's direction ("M9 failed to achieve our goals
… make M10 a retry … build something unimpeachable by competitors"). Evidence `m10/PLANNING.md`;
M9's record `m9/FINDINGS.md`. Adversarial review: gpt-5.6-terra, read-only, six passes
(`research/m10-codex-plan-2026-09-01.md`, `-plan2-` … `-plan6-`; full logs are gitignored `.log`
files beside them); every finding and its disposition is in PLANNING §8, including two reviewer
dissents the owner can overrule. **Amended 2026-09-01 after pass 6 on Dylan's compute ruling:** M10
runs on a rented GPU budget or not at all (§Owner rulings, §Compute); the review amendments taken
with it are decision 9 and PLANNING §8's amendment block. **Amended 2026-09-04 after a full plan
review on Dylan's direction ("is this the best way to go about it? Is it the most efficient?
… I'm not sure why we need to generate synthetic data?"): see §Amendment 2026-09-04, which is
authoritative wherever an older sentence in this file disagrees.** The M9 model is **nano**; M7's
table is **zero**; the product is still the pair on one stella index.*

## What binds from M7 and M9 — exhaustive

From `instructions-m7.md`: decision authority; vendor and source-licence rules (as relaxed in
CLAUDE.md); frozen-comparator validation and pairing; pushed-freeze, ledger, crash/abort
disclosure; Sonnet-only research subagents; the headless reporting files under `m10/`
(STATUS/RESULTS/EXPLORED/LEDGER); the eval-use licence standard (source-level evidence before a
selection set is used); CLAUDE.md's tightness rule. From `instructions-m9.md`: the bars, the
permitted claim and its qualification, the six-set transaction design (`m9/FINAL_LOCK.md`,
re-registered for M10 with new hashes and the one change in §Final run), the LoTTE atomic-read
protocol, R1/R2/R3, the stage discipline (no evaluation before its lock; "diagnosed defect" =
implementation divergence only), the 70 MB fp16 query-asset target, screen-in-closed-form first,
the watch-long-runs checklist. **M10 supersedes** M9's data pool, mix, optimizer, schedule, plateau
rule, phase-2 handling, screen arm list, and selection surfaces. Where M10 changes a carried
mechanism, M10 controls. Read `m8/CODEMAP.md` and `m9/CODEMAP.md` before writing code.

## Why M9 missed, and what M10 changes

M9 retained **93.8%** of the teacher on NQ and **50–71%** on the two CQADupStack dev components
(`m9/FINDINGS.md` §1): its 463K Wikipedia-QA and product queries did not cover the forms the model
is tested on, and its 384-wide linear head is the subspace L2 regression cannot push past once
queries are diverse (PLANNING §9). Parameter count is not the constraint. Both causes are tested
before anything is built (§Screen, families A and G). M10 changes, in order of evidence (PLANNING §3):

1. **Coverage** — ~4.0M query texts across 12 forms, most of them REAL text (§Data).
2. **Optimizer regime** — LEAF's batch 32 and three linear 1e-4→1e-5 cycles; plateau read on
   annealed checkpoints, best-to-best.
3. **A wider linear output** — three mean-pooled layers concatenated (1152-d) before the head.
4. **Warm start from the M9 candidate** as a screen arm.
5. **LEAF's own loss** — the L2 norm ‖e‖₂, not M9's squared L2 — as the one objective arm.

**The student cap is 35M and it is hard** (Dylan, 2026-09-01: "109M is not an option. This isn't
low compute anymore. 33M was already in the upper bound of what I think is acceptable"). bge-small
with the three-layer head is 34.5M; MiniLM-L6 with it is 23.9M. The cheaper student gets the
benefit of a tie (§Screen, family F).

## Owner rulings already made (Dylan, 2026-09-01)

- **Git:** M9 was merged to `main` on 2026-09-01 after the repo cleanup (its six-set close-out is
  still pending and runs from `m9-work`); M10 execution work happens on branch **`m10-work`**
  under the headless commit-and-push contract; merges to main at stage boundaries
  need Dylan's go. M9's registered six-set close-out still runs from `m9-work`, because `guard9`
  pins that branch (`m9src/guard9.py:35`); the branch is kept until then.
- **Compute (Dylan, 2026-09-01): "M10 won't be done on a 3080. M10 will be done on a GPU budget, if
  allowed, or not at all."** Every M10 stage from M10.0-c on runs on a rented GPU under the budget in
  §Compute, or M10 does not run. The RTX 3080 is not an execution target for any M10 step; it only
  supplies M9's frozen checkpoint once (§Compute). Every dose and duration here is set from the
  budget, not from the box.

## Amendment 2026-09-04 — plan review (authoritative over any older sentence here)

Dylan asked whether the plan was the best and most efficient route, questioned the synthetic data,
and said *"don't over-engineer"* and *"keeping the same teacher (Stella) is the goal"*. Nothing
below touches stella, the 35M cap, the frozen tower, the pair, or any observed number — no M10
number exists. Evidence: PLANNING §11 (measured rates), §12 (the synthetic-data question),
`m10/LEDGER.md` §3. Full disposition of what was cut: `m10/EXPLORED.md`.

| # | change | why |
|---|---|---|
| A1 | **Family D cut from three ranking-aware arms to one arm: LEAF's L2-norm loss `L=‖e‖₂`.** Deletes the 1M-document candidate bank, the exact-mining pass and its HNSW fallback, the τ entropy rule, the 129-way D-NCE spec and the seed-rank provenance field. **A registered plateau response replaces it** (§Recipe) | LEAF (97.7% asym), EmbedDistill, arXiv 2306.11550, mxbai-edge-colbert and DistilVDR all reach 92–98% retention by pure embedding regression, so the class is *unnecessary* at our target; and it was the most machinery-heavy block in the plan. LEAF's loss is the **norm**, not the square — a one-line arm we had silently diverged from. **Corrected after the Fable review: LEAF's Appendix B is about intermediate-layer KD (MiniLM / TinyBERT / DistilBERT losses, 54.9 / 53.7 / 55.3 vs 60.7), NOT about a ranking term on top of regression — no published null exists for the class we cut, so this is a cut on cost and sufficiency, not on evidence of inertness** |
| A2 | **Generation cut from 3.0M to ≈1.0M, and only for the forms no corpus contains.** The harvestable forms come from **real** text mined out of the licensed pool (titles, headings, lead claim sentences, extracted interrogatives) — new arm A3 | Three independent saturation curves put diminishing returns under ~1–1.5M (DistilVDR saturates above 75% of a 1.49M pool; SPEED log-linear to ~920K; doc2query 50–75% coverage ≈ 90–95% of max gain). And **three of the four clean-4 headline datasets have a real-text counterpart** (scidocs↔titles, scifact↔claim sentences, trec-covid/nfcorpus↔headings), so the headline can rest on real text plus the teacher instead of on the generator's prior |
| A3 | **C1/C2 registered on clean-4 as well as avg-6** (§Goal, §Final run). clean-4 bars: bge-small **0.5046**, leaf-ir-asym **0.5233** | M14 registered clean-4 as the headline partition for both zero and nano on 2026-09-04 (`instructions-m14.md`), while M10 had C1/C2 on avg-6 with NDO-4 *descriptive*. Left alone, the paper's headline would carry no pre-registered pass/fail for nano. Fixed now, before any M10 number exists |
| A4 | ~~**The COV resolution number is measured first and SIZES the screen**~~ **— STRUCK by the Codex pass the same day** (an α that adapts to an observed width is not pre-registration): the number is measured and *reported* as the power disclosure; MDE 0.0056 and α 0.025/14 are fixed (§Screen) | The registered MDE 0.0056 sits **below the surface's own resolution**: on a family-weighted macro whose paired SE is 0.0033–0.0048 (BRIGHT ~100 queries/slice, CorporateLobbying 340), a contrast needs ≈0.009–0.0135 to resolve at the Bonferroni bound — 0.025/13 is z = 2.89, and it was z = 2.96 at the /16 count this finding was first written against — so a contrast landing at the MDE can never resolve. Codex pass 7's objection was to comparators drawn from family F; a direction-free power quantity on non-candidates (e5-small-v2, gte-small) is a power calculation, not a selection |
| A5 | **Decision 8 (second build seed, ≈100 GPU-hours) withdrawn; confirmations capped at two decisions** | Seed 1 is descriptive by construction and can trigger no action; the same hours buy ~3 extension cycles that can move the number. A screen-dose seed pair gives a replication band for ~5% of the cost |
| A6 | **Family F runs SECOND (right after A), and the remaining families run on its winner** | It was seventh, which made every other verdict transfer to the build student by assumption. Reordering costs nothing and removes the assumption. Order was A → F → G → B → E → C → D; **F → A → … after the Opus pass**, so family A's stop rule is also decided on the build student |
| A7 | **⚠ This is the one amendment that reinterprets a verbatim owner ruling — Dylan should strike it explicitly if he does not want it.** **The box is an execution target again for everything that is not generation.** The 2026-09-01 ruling stands where it bites — the *dose* is not set by the box — but the box is where the screens and (optionally) the build run, and it holds ~200 GB of M9 caches the plan had budgeted 12 GPU-hours and a day of network to re-derive | Measured on the box 2026-09-04 (PLANNING §11): the M10 recipe shape runs at **400 examples/s** in M9's two-chunk collate and **683** blended at 75/25 in length-bucketed single chunks (718 query-bucket / 596 document-bucket; `results/m10_rate_bench_box.json` — the Opus pass caught a stale 745 here), against the plan's imported LEAF planning rate of 560. The 3080 meets or beats the assumed A100 rate because at batch 32 with 35-token queries the job is launch-bound, not FLOP-bound. Generation still needs the cloud: Qwen3-8B bf16 is 16 GB on a 10 GB card |
| A8 | **Two pre-training data quality gates added** (§Data): near-duplicate and dispersion metrics per form, and a **distribution-overlap check against real MS MARCO dev queries in stella's own space** | The synthetic risk here is not wrong labels — the teacher's embedding of any text is a correct target by construction — it is distribution shift and diversity collapse, both measurable before a training step. MS MARCO is permitted for validation by Dylan's 2026-09-04 rule. FORMS-12 cannot serve this purpose: it scores student-teacher agreement on the same synthetic queries and is circular for it |

### Fable adversarial review of A1–A8, 2026-09-04 — all findings actioned

Verbatim findings, dispositions and the reviewer's own file list:
`research/m10-fable-plan-2026-09-04.md`. Read-exclusion audit **clean**. 3 BLOCKER / 8 MAJOR /
7 MINOR; verdict "not lock-ready"; **every one actioned**, and two of its numeric claims were
reproduced here before acting (B2's bootstrap widths; M1's pool composition).

| # | actioned as |
|---|---|
| B1 | A1's justification corrected — LEAF's Appendix B is intermediate-layer KD, **no published null exists for the class we cut** — and a **registered plateau response** added (§Recipe) so a flat curve is not left without an answer, which is what M9 finding #4 demanded |
| B2 | **fixed-sequence gatekeeping** replaces the contradictory "Holm + fixed sequence"; the four planning proxies and the `trec-covid` n=50 disclosure are in §Goal |
| B3 | the two-step remedy is registered (admit LEDGER, then α = 0.05 uncorrected with seed confirmation as the guard); z corrected; MDE-redundancy and comparator bias stated (§Surfaces) |
| M1 | **the review's most valuable finding** — the pool is Wikipedia + ESCI with no scientific text, so arXiv is added as a licence-gated source, yields are measured before quotas lock, a form under 100K reverts to generation, and the clean-4 mapping is reworded as the hypothesis A3−A2 tests (§Data) |
| M2 | family F runs at 20M read as a curve, gains MiniLM-L12-v2, and its cheap-student tie-break is labelled a product preference (§Screen) |
| M3 | benchmark re-run as pass 2 (300 steps, seven single-chunk shapes, memory and allocator retries); the build is priced as a **range from hardware bound to M9 pipeline efficiency** (PLANNING §11) |
| M4 | reopening the ranking class is explicitly a next-milestone condition on the post-final six-set number (§Recipe) |
| M5 | `m10/LEDGER.md`, `m10/EXPLORED.md`, `m10/COV_CANDIDATES.md` corrected; FORMS-12 names an evaluation sample, not the deleted bank |
| M6 | A7 flagged ⚠ for Dylan to strike; `m10/STATUS.md`'s box window ordered with a drop rule |
| M7 | folded into B3's remedy — screen alpha is an internal selection, not a published claim |
| M8 | **form-balanced query sampling** is the anchor default (§Data), zero cost |
| MINORs | all adopted; hosted generation becomes the **default** (≈$20–60 against 10–20 GPU-hours plus setup), and **D-COV** is added as a second family-D arm (§Recipe) — the one cheap lever aimed at the in-distribution ceiling |

**The one thing the review found that the plan still does not answer** (PLANNING §13): M9 retained
**93.8% on NQ while training on NQ-like data**, and C2b's proxy sits near **95%** of the ceiling. Coverage
explains the 50–71%, not the 93.8%. Only families G and E and the new D-COV touch the covered half.
Recorded as an open weakness, not closed.

**Withdrawn from the review's own recommendations, and why** (kept so it is not re-proposed):
dropping family F to anchor on MiniLM-L6 was recommended and then **withdrawn** — it would have
killed family C too (M9's candidate is bge-small, so there is no MiniLM warm start), and arXiv
2306.11550's depth curve (1/2/4 layers → 86.1/92.5/96.2% retention) disagrees with LEAF's success
on 6 layers. One 5M arm ≈ 2 GPU-hours is not the place to economise when it picks the build student.
F stays, C stays, bge-small stays the anchor; A6 fixes the ordering problem instead.

## Amendment 2026-09-04b — feasibility review (second review of the day; authoritative over older text)

Dylan: *"is our goal feasible? … gaps, weaknesses, or avenues to improve? … No over-engineering."*
Evidence and dispositions: `research/m10-feasibility-review-2026-09-04.md`. Nothing here touches
stella, the cap, the pair, the frozen tower or any observed number.

| # | change | why |
|---|---|---|
| B1 | **Feasibility statement in §Goal** with per-dataset arithmetic (`results/m10_conjunct_arithmetic.json`) | C1b is harder than C2a and no ≥10×-gap regression distillation exceeds ~95%; registering C2b without saying so invites "you always knew" |
| B2 | **Four-conjunct claim decision table** (§Goal); release rule = **decision 11** | "C1 passes → release" was ambiguous on the likely outcome (C1a pass, C1b fail) |
| B3 | **G-768 → G-MLP**, a per-token *residual* nonlinear head `W_lin·x + W₂·GELU(W₁·x+b₁)`, W₁: 1152→192 (§Screen, §Recipe); `m10/EXPLORED.md` row corrected | the "no serving path" closure held only for post-pooling heads (`results/m10_head_mlp_parity_box.json`, 34.96M, min-cos 0.99999989); the Codex pass replaced the first bottleneck form, whose output rank was ≤512 |
| B4 | **Constructed scientific surface** `arxiv-title` (§Surfaces) — a COV family as proposed; a one-action secondary after the Codex pass; a *descriptive diagnostic with no action* after the Opus pass | every clean-4 set is scientific/biomedical and the screen had no surface that could see those forms; as an equal-weight family it would have rewarded A3 by construction and dominated the macro's power |
| B5 | DEV-6 recipe pre-screen (STATUS item 10) **dropped** | read DEV-6 twice for defaults the screen re-decides |
| B6 | **Query-asset size line** (§Recipe): ≈70.8 MB fp16 vs M9's 70 MB target | nobody had done the arithmetic for the wider head |

**Considered, not changed:** a three-seed anchor noise floor in place of the two-decision
confirmations (saves ≈10 GPU-h, weakens a reviewed rule); the dose, cap, teacher, family D,
gatekeeping. **Open to Dylan:** decisions 11 and 12, and A7 (⚠).

**Codex pass on B1–B6** (`research/m10-codex-feasibility-2026-09-04.md`; read-exclusion audit
clean): 4 BLOCKER / 8 MAJOR / 4 MINOR, all actioned there — the 0.025 quantile in the proxies and
the final bound; G-MLP's residual form and its training/export wrappers; family F's L12 rule and the
fourteenth contrast; decision 11's default; `arxiv-title` demoted to a secondary surface (then to a diagnostic by the Opus pass); **A4's
MDE sizing and its two-step remedy struck** (MDE 0.0056 and α fixed; the resolution number is a
power disclosure); MedlinePlus / CDC / ClinicalTrials.gov out of M10; superseded prose deleted.

**Opus pass after Codex** (`research/m10-opus-review-2026-09-04.md`; audit clean): 3 BLOCKER / 9
MAJOR / 7 MINOR, all actioned there — the mandate's 745 / 1,331 examples/s were not in the artifact
(683 / 1,517 blended; 81 h; family E reads 2.2×); LEDGER §0 split into §0a (design, M10.0-e) and §0b
(data-dependent constants, close of M10.1); gatekeeping order **C1b → C1a → C2a → C2b**; the mix as a
4-step window; a kill rule that can fire mid-build; the plateau rule defined; C1 worded as a system
comparison; family order **F → A → …**; `arxiv-title` descriptive only; B = 200,000 with
`inverted_cdf`; the licensing row that would have excluded BRIGHT corrected; PLANNING §5–§6 marked
superseded; LEDGER §5's duplicate tables deleted.

## Owner decisions (defaults apply until Dylan rules; each is recorded in `m10/LEDGER.md`)

**States:** 1, 4, 7 default-active pending ratification · 2, 3, 5, 6, 8, 9 closed · 10, 13 adopted
(strike an item to revert it) · 11, 12 proposed, default-active. This is the only pending-decision
list; `m10/STATUS.md` points here.

| # | decision | default while open |
|---|---|---|
| 1 | Ratify M9's final-lock amendment **together with the close-out amendment that strikes M9's reserved conditional** (§Stage plan, M10.2): M9's close-out is six-only and cannot spend the reserved access | blocks the close-out only |
| 2 | **GPU budget — VALIDATED by Dylan 2026-09-04.** Re-priced at the measured rates after the plan review: hybrid (screens on the box, cloud for generation and the build) **≈ 56–101 cloud GPU-hours ≈ $85–250**; all-cloud ≈ 130–190 GPU-hours ≈ $200–475. Ceiling **$1,000** unchanged and unspent. PLANNING §6 (re-derived) and §11 | approved; the day-one benchmark still re-derives §6 before anything scales |
| 3 | FineWeb as a seed — **ruled out 2026-09-01** (delegated): Wikipedia and the approved corpora carry the topics; FineWeb adds a rights review and a blocklist for no measured gain. Reopening condition in `m10/EXPLORED.md` | closed |
| 4 | PAQ (machine-generated questions over Wikipedia; data CC BY-SA, generation code CC BY-NC) as query text | include, from Facebook's official release (never the unofficial HF mirror); 1.0M uniform sample in the build (seed 0, file hashes pinned), 4.037M in the volume-control screen arm A2 only; attribution recorded |
| 5 | FineWeb documents — **excluded 2026-09-01**: no reserved-set document fingerprints exist and creating them would open reserved corpora (`m9/LEDGER.md` §1.3) | closed |
| 6 | A >35M tier — **ruled out by Dylan 2026-09-01**; 35M is a hard cap | closed |
| 7 | Confirm: LoTTE read #1 withdrawn unexecuted in M9; renumbering M10/M11/M12 | as recorded in `m9/STATUS.md` and CLAUDE.md |
| 8 | Second build seed at full dose — **WITHDRAWN 2026-09-04** (amendment A5): descriptive by construction, could trigger no action, and cost ≈100 GPU-hours. Replaced by a screen-dose seed pair on the selected recipe as the replication band; the freed hours go to extension cycles | closed |
| 9 | The 2026-09-01 review amendments taken with the compute ruling: dose 200M examples, screen dose 5M, G-1536, the COV resolution number, extension capped by budget (PLANNING §8, amendment block and pass 7). **D-KL1, D-NCE and the seed-rank field are struck by amendment A1; the resolution number is repurposed by A4** | adopted as amended |
| 10 | The **2026-09-04 plan-review amendments A1–A8** (§Amendment 2026-09-04) | adopted on Dylan's "make your changes to the plan"; any item he strikes reverts to the 2026-09-01 text |
| 11 | **Release rule under four conjuncts** (§Goal claim table): does a C1a pass with a C1b fail ship as "nano"? | **no** — "release" needs C1b, the headline partition (Codex 2026-09-04: releasing on the partition whose margin is 73% fiqa would be motivated). A C1a-only pass is still *published* as a frontier measurement labelled not recommended, as `zero` was. Dylan may loosen to "C1a suffices" |
| 12 | **CUREv1 as a validation-only biomedical read** — 2,000 real clinician queries, CC BY-NC 4.0, PMC-OA full-text passages (PubMed-family), pools annotated by Qwen 2.5 72B; fingerprint-screened vs the six; never training data. Reopens the validation clause of M7's source-family rule (`research/m10-feasibility-review-2026-09-04.md` §4c) | **not adopted** until Dylan rules. Recommendation after the Codex pass: **as a reported diagnostic beside every arm — yes; as a selection-bearing COV family — no** (LLM-judged pools; provenance overlap fingerprints cannot see; Codex M6). **Withdrawn by the review:** PubMed titles / PubMedQA as training text — no affirmative grant on PubMed abstracts (NLM disclaims copyright; publishers may hold it); reopens only with per-record licence provenance (PMC-OA CC BY records) *and* the contamination rule resolved (§4a) |
| 13 | The **2026-09-04b feasibility-review amendments B1–B6** (§Amendment 2026-09-04b) | adopted; strike any item and it reverts |

## Goal, bars, and the permitted claim — unchanged from M9

nano = a ≤35M transformer query encoder serving stella-400M's frozen 1024d index (the SAME index
as zero). Bars, paired on `results/perquery.json` (sha `6b18e3dd…`, irreplaceable):

| | avg-6 | NDO-4 |
|---|---|---|
| teacher ceiling (stella symmetric) | 0.5744 | 0.5640 |
| **AIM: leaf-ir-asym** | **0.5155** (89.7% retention) | 0.5233 |
| **RELEASE BAR: bge-small** | **0.5042** (87.8%) | 0.5046 |

**NDO-4 is clean-4** (`nfcorpus`, `scidocs`, `scifact`, `trec-covid`), the partition M14 registered
on 2026-09-04 as the headline for both halves of the pair.

C1 nano-dense > bge-small (release); C2 nano-dense > leaf-ir-asym (aim). Both dense-vs-dense;
fused rows descriptive only. **Amendment A3, 2026-09-04: C1 and C2 are each registered on BOTH
partitions** — `C1a`/`C2a` on avg-6 (bars 0.5042 / 0.5155) and `C1b`/`C2b` on clean-4 (bars
**0.5046 / 0.5233**), all four reproducible from the frozen comparator rows by
`scripts/clean4_bars.py` → `results/m10_bars.json` (a comparator-only read taken 2026-09-04 before
any nano number existed).

**Multiplicity: fixed-sequence gatekeeping, NOT Holm** (corrected after the Fable review, which
found the two named together and incompatible — Holm orders by p-value, a fixed sequence does not,
and a text permitting either lets a future session choose after seeing the numbers). Registered
order **C1b → C1a → C2a → C2b** — the release gate of decision 11 first (the stress table has a case,
`arguana` at 65%, where clean-4 passes and avg-6 fails, which the old C1a-first order could never
test; Opus 2026-09-04), then the M9 headline sentence's conjunct C2a before the stretch aim C2b; **if
Dylan makes C1a the release gate, C1a goes first**. Each conjunct is tested at the **full one-sided
0.025** and testing stops at the first non-rejection. This controls the family error rate and, unlike Holm-4, **costs
the avg-6 release claim nothing** — under Holm-4 the smallest-p conjunct would have needed 0.00625
and C1a's bound would have widened ~11% against the two-conjunct design it replaces.

**Planning proxies, not pass points** (`results/m10_bars.json`, `planning_proxies`): the bar plus the
comparator-pair (leaf vs bge) bootstrap width at the **registered 0.025 quantile** — the Codex pass of
2026-09-04 caught the script still on M9's Holm-2 0.0125. Nano's own interval depends on its per-query
differences, so no retention figure "passes" by arithmetic; the rule is the final run's own bound.

| conjunct | bar | proxy width | proxy point | = retention of ceiling |
|---|---|---|---|---|
| C1a release, avg-6 | 0.5042 | 0.0089 | 0.5131 | 89.3% of 0.5744 |
| C1b release, clean-4 | 0.5046 | 0.0122 | 0.5168 | 91.6% of 0.5640 |
| C2a aim, avg-6 | 0.5155 | 0.0089 | 0.5244 | 91.3% |
| C2b aim, clean-4 | 0.5233 | 0.0122 | 0.5355 | **94.9%** |

**Disclosed with it (B1, `results/m10_conjunct_arithmetic.json`):** under a *uniform-retention
proxy* the four sit at 89.3 / 91.3 / 91.6 / 94.9% (C1a / C2a / C1b / C2b) — a lens, not an ordering:
retention is distribution-specific (M9: 93.8% covered, 50–71% uncovered), so heterogeneous
retention can reorder them. To equal bge-small per dataset: scifact 91.4 · nfcorpus 83.0 · fiqa 72.9
· arguana 94.7 · scidocs 85.7 · trec-covid 92.0%; LEAF beats the ceiling on trec-covid. **Stress:**
one dataset at 65% with the rest at 94% clears no conjunct if it is `trec-covid` or `scifact`, only
C1a if `nfcorpus`. At uniform 92%, fiqa (disclosed stella training data) supplies 73% of nano's avg-6
margin — why clean-4 is the headline. **Prior, registered before any nano number:** M9 held 93.8% on
its covered form; a bounded literature search (pure embedding regression, ≥10× teacher/student
parameter gap; `research/m10-feasibility-review-2026-09-04.md` §2) found no result above ~95%, and
EmbedDistill's 95–97% at ~10× used labels and score distillation. So C1a is expected if coverage
works, **C1b and C2a are the contest, and C2b is a low-prior stretch aim** — the screen results that
would raise the prior are G-MLP or D-COV moving `nq-250k` retention from ~94% toward 96–97%.
clean-4's interval is ~36% wider than avg-6's because `trec-covid` carries a
quarter of the clean-4 macro on **50 queries**, and C2b therefore demands LEAF-level retention on
the headline partition. Registering clean-4 is still right — M14 headlines it — but the plan is
registered knowing C2b is the hardest of the four by a wide margin, not knowing only that its bar
is 0.0078 higher. **clean-4 is also the most data-designed partition** (Opus): §Data names
scidocs↔titles, scifact↔claims, nfcorpus/trec-covid↔headings, so the headline controls *teacher*
contamination while maximising *training-design* targeting — two different exposures, both stated. **C2 is a whole-system comparison** (stella documents + nano queries
vs arctic documents + LEAF queries): different document towers, index sizes, encode costs and
disclosed teacher overlap. It supports exactly M9's verbatim headline sentence and **no statement
about nano versus LEAF's query tower**; the report carries both systems' retention against their
own teachers, index bytes, document-encode cost and query latency beside the number. NDO-4 and
reserved NDO-3 stay descriptive.

**Claim decision table under four conjuncts (B2).** Testing stops at the first non-rejection:

| outcome (order C1b → C1a → C2a → C2b) | release? | permitted claim |
|---|---|---|
| C1b fails | **not a release** (decision 11 default): the checkpoint is published for reproducibility and as a frontier point, labelled *not recommended*; the card states it did not resolve above the bge-small system on the contamination-controlled partition. **M13 consequence (Opus):** a not-recommended model does not enter the upstream FastEmbed PR, so the single-PR ruling then covers zero and the tower only — Dylan decides this with decision 11 | none |
| C1b passes, C1a fails | release | "the stella-document + nano-query system outperformed the bge-small system on the four contamination-controlled datasets"; avg-6 reported unresolved |
| C1b, C1a pass, C2a fails | release | the same sentence on both partitions |
| C1b, C1a, C2a pass, C2b fails | release | M9's verbatim headline sentence (avg-6); the paper states the aim was missed on the headline partition |
| all four pass | release | the headline sentence on both partitions |

**C1 is a system comparison too** (Opus M7): bge-small is a symmetric 384-d system with its own
66.7 MB query asset and its own index, so every C1 sentence carries index bytes and document-encode
cost beside it, exactly as C2 does. M9's "C1 fail, C2 pass → aim claim permitted" row does not
survive gatekeeping and is deleted.

Additional mandatory disclosures: per-dataset retention, the
synthetic-data provenance table (generator revision and terms, prompts, seed sources and ids,
counts per form, removal counts per screen), the selection-surface table (licence evidence,
revisions, sizes, removals from training data), dose in examples / tokens / GPU-hours beside
LEAF's ~100 A100-hours, and the dev-reuse count.

## Stage plan

- **M10.0 DIAGNOSIS + SCREEN LOCK** (no six-set, reserved or LoTTE output exists during this stage).
  (a) Mac diagnostics — **done 2026-09-01**, `m10/RESULTS.md`, PLANNING §9–9c: the rank probe
  (a 384-d subspace keeps 99.5% of one query distribution and 90–93% of three; 98–100% at 640 —
  evidence about the class under L2 regression, not a bound), the head-width probe (a frozen
  bge-small retains more with each pooled layer), and the serving-parity check (fastembed 0.8.0
  reproduces the three-layer per-token head to 2e-7; 34.5M parameters). **Action taken:** the head
  is applied per token — linear by default, G-MLP's residual form as an arm (§Screen); the pooled
  feature widens to three layers; family G decides. Repeat the
  parity check on MiniLM's head before family F may select it, on the four-layer head before G's
  1536 arm runs, and on the trained artifact with M9's locked parity sample before the freeze.
  (b) Capacity probe (`m9src/capacity_probe.py`, unchanged) — **optional, report-only**: no outcome
  changes an M10 action under the hard cap; runs only on an idle GPU; its 109M student is
  768-hidden, so any gain is partly width.
  (c) Per-component DEV-6 read of the M9 candidate incl. `heldout-longq` (the baseline row).
  **Runs on the box** (amendment A7): it holds the checkpoint and the encode caches.
  (d) **COV admission** (§Surfaces): for every candidate component, record in `m10/LEDGER.md` §2
  its primary-source licence URL and terms, HF repo and revision, corpus size, query count, qrels
  format and metric, its corpus-level contamination check and its fingerprint screen against the
  six and the reserved four. A component is named COV only after that record is pushed. **COV
  admits only surfaces no M10 decision has read**: the two CQADupStack dev components were scored
  by the Mac diagnostics (PLANNING §9, 86 raw reads) and stay in DEV-6. Then every admitted COV
  corpus, query set and document set joins the protected index (`m8src/protected_filter`) before
  any seed is drawn or any harvested, PAQ or synthetic text is constructed. **Re-run admission under
  Dylan's 2026-09-04 licence rule before judging the family floor: ConsumerContractsQA (CC BY-NC) is
  re-admissible, giving four families without LEDGER.** The **COV resolution number** (§Surfaces) is
  measured on the admitted surface and pushed before (e) as the screen's **power disclosure** (A4's
  sizing struck by the Codex pass). **Also here (B4):** draw the `arxiv-title` held-out papers by
  id-without-version with seed 0, protect them, encode them.
  (e) **Screen lock**: `m10/LEDGER.md` §0 (skeleton committed 2026-09-01) fixes every arm of
  §Screen (fifteen arms), order, doses, seeds, surfaces, the fourteen contrasts, MDE 0.0056 and the
  fixed 0.025/14 bound, confirmation design and outcome→action maps.
- **M10.1 DATA.** Harvesting of the real query-like text (§Data), generation for the non-harvestable
  forms under the §Data contract (200-query smoke per form first), PAQ samples, decontamination
  against the protected index (now including COV) and the six's documents, the FORMS-12 hold-out,
  teacher targets, `results/m10_data_manifest.json` with hashes and the provenance table.
  **Before any arm and recorded in the manifest**, the two A8 quality gates: per-form
  near-duplicate and dispersion metrics, and the stella-space distribution-overlap check against
  real MS MARCO dev queries (validation only).
- **M10.2 SCREEN + RECIPE LOCK.** The arms of §Screen, the confirmation runs, then one pushed lock
  commit with every field of M9's M9.2 list filled — including the objective, the best-to-best
  plateau/extension rule on annealed checkpoints, the GPU-hour allocation under the ceiling (build,
  extensions), the final-run registry with the four C-conjuncts and their fixed sequence, and
  LoTTE read #1's manifest. Codex and Fable review the pushed lock. **Then, and only then,** M9's close-out runs:
  its registered six-set transaction **amended before execution to six-only** (the `if C1 then
  execute` reserved conditional is struck; disclosed and ratified under decision 1), so M9 cannot
  spend the reserved access. Its rows are a forecasting calibration and a whitepaper frontier
  point; nothing in M10 may change on them. **Wherever those rows are published — frontier table,
  whitepaper, model card — they carry M9's build-provenance disclosure with them:** the M9
  candidate's provenance spans TWO build lock states (mid-build infrastructure repair, session
  force-reopened; `m9/LEDGER.md` "M9.3 BUILD PROVENANCE DISCLOSURE" and `m9/BUILD_LOG.md`), and it
  may never be described as an uninterrupted single-lock run. LoTTE read #1 then runs as the
  registered veto on the selected recipe.
- **M10.3 BUILD.** One candidate under the kill and extension rules; export + parity; freeze;
  pre-freeze review; LoTTE read #2 (audit only). The screen-dose seed pair (amendment A5) is the
  replication band; no full-dose replica runs.
- **M10.4 FINAL.** The six-set transaction → decision → the registered reserved conditional.

## Data

**Query corpus (~4.0M unique texts, query role) — amendment A2: most of the form breadth is REAL text.**

| source | build count | licence | role |
|---|---|---|---|
| M9 real queries (hotpotqa, squad, esci, mrtydi, nqopen, triviaqa; fever out) | 463K | CC BY-SA / Apache | real forms |
| PAQ sample (decision 4) | 1.0M | CC BY-SA (data) | factoid volume — capped so it cannot dominate |
| **Harvested real query-like text** from the licensed document pool (§Harvest) | ≈1.5M | inherits the pool's licences; no new rights surface | the harvestable forms, with **no generator prior** |
| **Synthetic, Qwen/Qwen3-8B (Apache-2.0; revision `b968826d…`). DEFAULT: hosted open-weights inference of that pinned revision (provider and served revision recorded in the manifest) — ≈100M output plus ≈300M prompt tokens is ≈$20–60 at Sept-2026 list prices, against 10–20 GPU-hours plus instance setup for self-hosting. Registered fallback: bf16 via vLLM on the rented GPU** | ≈1.0M | generated under the generator's terms; provenance pinned; **not redistributed** without review | ONLY the non-harvestable forms |

**Form taxonomy — 12 forms**, quotas locked at M10.1 (±10% realized), and each form is assigned to
exactly one of the two sources. **Harvested (real text, ~250K each):** paper-title query ·
scientific claim (a statement) · 2–4-word keyword query · factoid question · consumer-health
question · product-search query. **Generated (~165K each):** how-to / troubleshooting question with
title and body · long counter-argument paragraph (120–220 words) · finance / personal-economics
question · comparison question · yes/no verification question · conversational multi-sentence
request. **The assignment's motivation, stated as the hypothesis it is:** three of the four clean-4 headline
datasets have a real-text counterpart (scidocs↔titles, scifact↔claim sentences, trec-covid and
nfcorpus↔headings and consumer-health), so *if* harvesting covers those forms the headline partition
rests on real text plus the teacher rather than on the generator's prior. **That is what contrast
A3−A2 tests; it is not established.** The generated half covers the interactive forms no corpus
contains, which map to the non-headline sets and to COV's forum-style families.

**Harvest sources — the Fable review found the existing pool cannot carry the scientific forms.**
The 6.15M pool is `esci-prod` (808K product listings), `hotpotqa-corpus` (5.23M Wikipedia),
`mrtydi-docs`, `squad-ctx` and `fever-pos` (excluded) — Wikipedia and products, **no scientific
text**. Harvested from it, "paper title" would be Wikipedia article titles and product names and
"scientific claim" would be Wikipedia lead sentences, neither of which is the form scidocs and
scifact test. Two consequences, both binding:
- **A scientific source is added for those two forms: arXiv metadata (titles and abstracts).** It
  must clear the same licence gate as any training source before use — primary-source evidence of a
  commercial-use grant recorded in `m10/LEDGER.md` §2 (the Kaggle release is CC0 1.0; the OAI feed
  is not, so the record must name which artifact and revision is used). It is a different source
  family from S2ORC, PubMed, CORD-19 and NutritionFacts, all of which stay excluded for
  contamination, and every document passes the existing screens against the six's documents. **If
  the licence evidence does not clear, both forms revert to generation and the report says so.**
- **Yields are measured before quotas are locked.** For every extraction rule: post-dedup,
  post-screen unique count, pushed to `m10/LEDGER.md` §1 at M10.1 before the per-form quotas are
  fixed. **A harvested form that yields under 100K post-dedup reverts to generation**, and the
  realized harvested/generated split is reported rather than assumed. `consumer-health`
  is **generated** from Wikipedia-medical seeds. Sources verified 2026-09-04b (clauses in
  `research/m10-feasibility-review-2026-09-04.md` §4b) but **kept out of M10**: MedlinePlus
  government-authored topics and CDC pages are US public domain yet are MedQuAD's sources, so
  harvesting them would let the MedicalQA COV read reward A3 by construction (Codex M7);
  ClinicalTrials.gov's terms clause could not be read. Out on licence: DailyMed, OpenAlex,
  bioRxiv/medRxiv, WHO, Cochrane, everything PubMed / PMC / Europe PMC (no affirmative grant on
  abstracts). In: **arXiv metadata** (CC0).

**§Harvest — the real-text pipeline (M10.1, deterministic, no model in the loop).** From the 6.15M
pool documents and the Wikipedia seed corpus, all under licences already approved for training:
titles and headings as-is (title and keyword forms); declarative lead sentences of abstract-like
passages, filtered to 8–40 words with a finite verb and no first person (claim form); sentences
ending in `?` extracted with their preceding sentence as optional body (factoid, consumer-health,
product forms, routed by the source corpus). Every harvested string goes through the same screens,
quotas and hold-out as a generated one, and the manifest records the extraction rule, source
document id and per-rule yield. **Seeds for the generated half:** Wikipedia stratified by
top-level category (CC BY-SA) and the approved pool corpora. FineWeb is out of M10 (decisions 3
and 5). MS MARCO may never seed either half. **Contaminating source families — never seeds, never regression text, and
excluded from COV** (`research/m7-data-licensing.md` map): S2ORC / Semantic Scholar; PubMed;
NutritionFacts.org and its mirrors; CORD-19; StackExchange personal finance (money.SE) and Reddit
finance; args.me / idebate; every six-set and reserved corpus.

**Generation contract (M10.1):** generator pinned by HF repo + revision in the manifest (bf16
weights; the hosted provider and its served revision recorded if the fallback fires); vLLM sampling temperature
0.8, top-p 0.95, `max_new_tokens` per form (60, or 400 for the argument and conversational
forms), `seed = blake2b-64(seed_passage_id)`; the reply must parse as one JSON list of exactly n
strings (`m10src/forms.parse`, strict — no preamble); one retry on a contract failure, then the
seed is dropped; exact-duplicate queries removed. **Smoke:** 200 queries per form, read by Dylan,
who is the approver; a form passes when ≥ 90% of replies meet the contract and ≥ 80% of a
50-query sample are judged on-form; a failing form's prompt may be revised at most twice, each
revision recorded in `m10/LEDGER.md` §1 before the next smoke. **Seeds are pre-filtered:** a seed
passage that exact- or near-matches the protected index is never used.

**Screens on every generated or PAQ query, thresholds fixed here (M7's fingerprints,
`m7src/decontam.py`):** exact `blake2b-64` match or word-8-gram bottom-32 sketch ≥ 8/32 against
(i) the protected index (six + dev + reserved + LoTTE **+ admitted COV queries and documents**),
(ii) the six's documents; (iii) any word-5-gram shared with the query's own seed passage (a
copied span is not a query) — **for harvested text this screen is against the source document
minus the harvested span itself, since a harvested title IS a span of its document**;
word-4-gram containment for 4–7-word queries. The M9 real-query pool
and the document pool are re-screened against the COV additions (R1 removes matching queries;
matching pool documents are
removed too). Removal counts per screen, per form and per COV component are recorded **before any
COV component is scored**. **FORMS-12 hold-out:** 500 seed documents per form are set aside first;
queries generated or harvested from them are never trained on.

**Quality gates (amendment A8), executed on the immutable manifest before any arm, recorded in
`m10/LEDGER.md` §1; each is a REPORTED number with one registered action.** The risk this addresses
is not label noise — there are no labels, and the teacher's embedding of any text is a correct
target by construction — but distribution shift and diversity collapse.
1. **Diversity, per form:** near-duplicate rate under the existing word-8-gram bottom-32 sketch at
   ≥ 16/32 within the form, and mean pairwise stella-space cosine. **Action:** a form whose
   near-duplicate rate exceeds 25% has its quota cut to its post-dedup unique count rather than
   being topped up by more generation.
2. **Distribution overlap against real queries:** encode a 50,000-query sample of **MS MARCO dev**
   (validation only — never a seed, target, negative or gradient; no cache under
   `work/train/sources/`) and each form's sample with stella in the query role, and report each
   form's mean cosine to its nearest real-query neighbours plus the two-sample energy distance.
   **Action: none — this is a disclosed diagnostic**, because MS MARCO is web-search-shaped and a
   form legitimately unlike it (paper titles, arguments) must not be penalised for that. It exists
   so the report can state how far the training distribution sits from a real one, which is the
   one thing M9's coverage failure had no outside measurement of. FORMS-12 cannot serve this
   purpose: it scores student-teacher agreement on the same synthetic queries and is circular for it.

**Document corpus (document role):** the M9 pool, 6.15M documents, re-screened as above. FineWeb
documents are excluded (decision 5).

**Form-balanced query sampling (Fable M8, zero cost).** 463K real + 1.0M PAQ makes ~37% of query
texts factoid — the one form M9 already retains at 93.8% — so uniform-by-example sampling would
spend 37% of the query gradient where there is nothing to gain. The anchor therefore samples
query-role examples **balanced across the 12 forms** (each form's presentation share equal, texts
drawn with replacement within a form), and the lock records the realized shares. Family A2 still
measures raw PAQ volume, and the unbalanced variant is available as a reported diagnostic, not an arm.

**Mix:** by *example*, decided by screen family B; default **75% query-role / 25% document-role**
examples as a **presentation share over a window of 4 steps** — 3 query-bucket steps, 1
document-bucket step — because single-chunk length bucketing makes every step pure-query or
pure-document (Opus M4; the two-chunk per-step form costs the whole speedup A7 rests on). Family B:
100/0 = query steps only; 50/50 = alternating. Query-role examples get raw bytes (prompt policy (b)); document-role examples
carry M9's fixed document-role marker; teacher targets use the s2p template for queries and raw
bytes for documents. The same student input never maps to two teacher targets.

**Hard candidates, the mining pass, the 1M bank, the HNSW fallback and the seed-rank field are all
struck by amendment A1** — nothing in M10 needs a candidate list once family D is a single
loss-form arm. Reopening condition in `m10/EXPLORED.md`.

## Recipe (defaults; screen families decide the marked items)

- **Student:** bge-small-en-v1.5 is the **screen anchor**; the **build student is decided by family
  F**, MiniLM-L6-v2 by default. **Amendment A6: F runs SECOND, right after A, and every later family
  is screened on F's winner** — it used to run seventh, which made every other verdict transfer to
  the build student by assumption. Before F can select MiniLM, its three-layer head passes the same
  export and fastembed parity check as bge-small's (M10.0-a; the Mac or the box can run it). **Feature [family G]:** masked mean-pooled hidden states of
  layers 12, 8 and 4 concatenated (1152-d; MiniLM-L6: layers 6, 4, 2) → Linear(1152→1024) → L2
  normalize; head 1.18M parameters, 34.5M total for bge-small. G's fourth arm adds layer 2
  (MiniLM-L6: layer 1) → 1536-d, head 1.57M, 34.9M total, under the cap; its head passes the same
  parity check first. Warm-started in closed form (ridge)
  for the bge-small init; the M9-candidate init keeps its 384-d head and zero-initializes the two
  extra layers' columns. Exported per token so fastembed's mean pooling reproduces it exactly.
  Any per-token head, linear or not, ships this way (B3; `m10/EXPLORED.md`): the **trainer pools
  after the per-token head** (masked mean, then normalize — the parity script's reference form) and
  the **export wrapper emits the per-token output** for fastembed to pool; a wrapper-parity test
  precedes any G-MLP arm (M9's trainer pooled before the head and must be ported). **Query-asset
  size (B6):** a *projection* — bge-small or MiniLM-L12 with the three-layer head ≈70.8 MB fp16
  (10⁶ bytes) against M9's 70 MB *target*, whose rule is a logged, measured quality justification
  (`instructions-m9.md` §3); MiniLM-L6 ≈49 MB. Measured at export with tokenizer bytes and CPU
  latency (`m9src/edge_cost.py`), including G-MLP's per-token cost if it wins.
- **Phase 1:** squared L2 on unit-norm teacher vectors, fp32 loss, bf16 autocast (M9 form). A
  cosine-space variant is closed by algebra, not measurement (`m10/EXPLORED.md`): the head ends in
  L2 normalize and the targets are unit-norm, so ‖a−b‖² = 2−2cos and the gradients agree up to a
  factor of 2.
- **Objective [family D], amendment A1 — one arm, not three.** The single arm is **D-NORM**:
  LEAF's own loss, the L2 **norm** `L = ‖t − s‖₂` on unit-norm vectors, in place of the squared L2,
  from the first step. This is what the 97.7%-retention system minimised and M9 had silently
  diverged from it; it is a one-line change and adds no data structure. Default is the squared L2.
  **The ranking-aware class (KL over mined candidates, InfoNCE on the seed passage) is CUT** — with
  it the τ entropy rule, the 1M-document bank, the mining pass and the 129-way spec. Evidence:
  LEAF's Appendix B added distillation terms to plain regression and found no improvement, and
  every comparable asymmetric system (LEAF, EmbedDistill, arXiv 2306.11550, mxbai-edge-colbert,
  DistilVDR) reaches 92–98% retention without one, so the class is **unnecessary at our target**;
  cutting it is a cost-and-sufficiency judgement. **Corrected after the Fable review:** LEAF's
  Appendix B is about *intermediate-layer* KD (MiniLM / TinyBERT / DistilBERT losses at
  54.9 / 53.7 / 55.3 against 60.7), not about a ranking term on top of regression, and EmbedDistill's
  own ablation composes the two additively — **no published null exists for the class we cut.** The
  earlier wording claimed one and was wrong.
- **The registered plateau response [replaces the cut class] — this is what M9 finding #4 asked
  for.** M9's lesson was that a phase 2 must be specified at lock or a flat curve has no registered
  answer; cutting family D without one would recreate exactly that. So: at every cycle end the run
  records per-form retention (FORMS-12 by form) and per-family COV. **If the plateau rule fires (the extension condition fails at a cycle end k ≥ 3, §Kill) while
  the dev→six forecast is below the release bar, exactly one registered top-up cycle runs, pre-empting
  the stop once** — one
  further cycle of 66.7M examples, linear 1e-4→1e-5 as cycle 3, with the **bottom two forms by
  FORMS-12 retention at 2× presentation weight** and everything else unchanged. It needs no new data,
  no candidate list and no new loss; it is the plan's own coverage thesis applied to its own curve.
  It fires at most once, costs one extension cycle, is counted against `max_extension_cycles`, and
  the lock records the forecast formula and the tie-break for "bottom two".
- **Reopening the ranking-aware class:** only as its own milestone, never mid-M10. The condition is
  read on the **post-final six-set number** (the release bar is not observable during the build —
  Fable M4), so it is a next-milestone decision by construction: it reopens if M10's final avg-6 or
  clean-4 dense row misses its release conjunct by less than 0.01. Recorded so nobody re-derives it.
- **D-COV [family D], added 2026-09-04 from the Fable review.** `L = (s − t)ᵀ Σ (s − t)` with `Σ`
  the covariance of the pool's frozen stella **document** vectors, normalized to unit trace and
  shrunk as `(1−α)Σ̂ + αI/1024` with **α = 0.1** fixed here (never tuned on a selection surface).
  One 1024×1024 matrix computed once from vectors that already exist; no bank, no mining, no new
  data, no serving change — the head stays linear and fastembed-exact. **Why it is not a
  reparametrisation:** plain L2 is reconstruction, which spends a rank-limited output equally on
  every direction of stella's query space; weighting by document covariance spends it on the
  directions in which documents actually differ, which is what nDCG rewards. Under the rank limit
  PLANNING §9 measured, that is a different subspace, so it is genuine new behaviour rather than an
  absorbable transform. It is the only arm in the plan aimed at the 94% → 98% in-distribution gap.
- **One mechanism note on D-NORM, so its result is not over-read:** on unit vectors the gradient of
  ‖t−s‖ has unit magnitude per example, so a poorly-fit (under-covered) form receives *less* relative
  gradient than under squared L2. Under a coverage thesis the sign of its effect is not obvious in
  advance, which is why it is screened rather than adopted.
- **Optimizer:** AdamW β=(0.9, 0.999), eps 1e-8, wd 0.01 on dim>1, clip 1.0. **Batch 32 examples**
  [family E].
- **Schedule:** 3 cycles of equal example count, each linear 1e-4→1e-5; 2,000 warmup steps in
  cycle 1. Evaluation at every cycle end (annealed) and at cycle midpoints (curve watch only).
- **Init [family C]:** bge-small (default) or the M9 candidate.
- **Dose:** **200M examples** registered — LEAF's dose (6.7M texts × 30 epochs ≈ 201M; PLANNING §5),
  three cycles of 66.7M. Tokens follow the mix: at 75/25, 150M × ~35 + 50M × ~230 ≈ **16.8B**; at
  50/50 ≈ 26.5B. Query epochs ≈ 37 over 4.0M texts, document epochs ≈ 8 over the 6.15M pool. At the
  **measured** bucket rates (718 query / 596 document examples/s, blended **683** at 75/25; PLANNING
  §11, on the box) that is ≈ **81 GPU-hours**, and ≈ 139 h in M9's two-chunk collate — so
  **length-bucketed single-chunk batching is part of the build, not an optimisation** (amendment A7),
  which is why the mix is a share over a 4-step window (§Data). The day-one benchmark's rate replaces
  both, and family E's batch-32 penalty is recorded in the lock as the build's GPU-hour line. **Extension:** let m_k be the COV macro (full precision, the locked formula and
  evaluation hashes) at the end of cycle k. After every cycle k ≥ 3, one more cycle of 66.7M examples
  (linear 1e-4→1e-5, as cycle 3) starts iff m_k − max(m₁, …, m_{k−1}) ≥ 0.003 **and** the extension
  cycles already run are fewer than `max_extension_cycles`, an integer the lock fixes from the
  approved dollars minus every mandatory line of **this file's §Compute table** at the measured rates
  and the billed price (PLANNING §5–§6 are superseded evidence, Opus M12). Whole cycles only; a cycle whose projected cost plus billed spend to date would
  exceed the ceiling does not start. The lock records m_k's formula, the evaluation hashes,
  `max_extension_cycles`, and the spend source.
- **Kill:** non-finite loss/grad; two consecutive scheduled evaluations more than 0.0056 below the
  best evaluation *of their own kind* (midpoint against midpoints, cycle end against cycle ends), so
  the rule can fire inside the build and not only at its end (Opus M5). **Plateau** is read
  best-to-best on annealed checkpoints only; **the plateau rule fires when the extension condition
  below fails at a cycle end k ≥ 3** (Opus M6).
- **Seeds:** one shipping seed for the build (seed 0); confirmation seeds at screen dose per §Screen.
  **No full-dose replica runs** (decision 8 withdrawn, amendment A5); the replication band is the
  selected recipe re-trained at screen dose under two further seeds, reported descriptively.

## Screen — seven families, fifteen arms, fourteen contrasts, locked at M10.0-e

**Screen dose = 5M examples** (2.5% of the build; ≈ 420M tokens at 75/25; ≈ 2.0 GPU-hours per arm at
the measured blended 683 examples/s), full 3-cycle schedule compressed to that dose, one seed, identical
evaluation: **COV at every cycle end; DEV-6 once, at the final checkpoint** (its 5.2M-document
hotpotqa and 6.17M-row heldout components cost ~13 GB of reads per pass — M9's practice).
Throughput is recorded for every arm and decides nothing except family E (below).
**Order (A6, amended after the Opus pass): F → A → G → B → E → C → D**, and family F carries a higher
dose than the rest (§F row) because every later verdict — family A's stop rule included — is taken on
its winner (A4 at 5M is the winner's 5M checkpoint from F's curve; F's ≈17 GPU-h are spent before A
can stop the milestone, accepted). Student first, then the data thesis, then the two architecture/regime questions, then init and objective; every family after F
runs on F's winner, so no verdict transfers to the build student by assumption.
**Anchor** = the full M10 corpus (A4's data), mix 75/25, bge-small init, squared L2, bs 32, 1152-d
feature. Screens run **on the box** (amendment A7).

| family | arms | contrasts | rule and default |
|---|---|---|---|
| **A — data (the thesis)** | A1: M9 pool (463,314 queries) · A2: M9 pool + PAQ (factoid forms only — the volume control) · A3: A2 + the **harvested real** query-like text · A4: A3 + the **generated** forms (the full M10 corpus, = anchor). **A2, A3 and A4 are cut to the identical post-screen unique-text count** (the smallest of the three after decontamination, the larger two downsampled with seed 0) and all hashes are locked before any arm | **A3−A2** (forms from real text, at equal volume) · **A4−A3** (what generation adds over harvesting) · A4−A2 and A2−A1 descriptive | **Three registered outcomes on A3−A2** (the forms contrast, now carried by the real-text arm): corrected lower bound > MDE → coverage **resolved on the COV families**, build proceeds; point ≥ MDE and lower bound > 0 but ≤ MDE → **positive, not resolved**, build proceeds and the report says so; otherwise → **M10 stops before any build and returns to Dylan with all four rows**. **A4−A3 decides whether the generated half is in the build at all**: if it does not resolve, the build uses A3's corpus and the ≈1.0M generated queries are dropped from the build (they stay in the report as a measured null). A2−A1 is the volume effect; if it resolves, the build keeps volume as well as forms |
| **F — student** | bge-small (34.5M with the head) · MiniLM-L6-v2 (23.9M) · **MiniLM-L12-v2 (33.4M with the head)** — the first two at **20M examples each, read as a curve at 5M / 10M / 20M**; L12 at 5M | 2: L6−bge-small at 20M · L12−(winner) at 20M **if L12 is extended**; the count is fourteen whether or not it runs | **Amended after the Fable review, made executable after the Codex pass.** The old rule ("bge-small only if it wins resolved") pre-decided MiniLM-L6 by construction, because M9's same contrast was −0.0026 *unresolved* and a 5M screen cannot resolve 0.003; that also silently skipped family C, since only bge-small has an M9 warm start. So F gets the dose its consequence deserves — it picks the build student and every later verdict — and a third arm, because arXiv 2306.11550's depth curve (1/2/4 layers → 86.1/92.5/96.2% retention) says depth buys retention while LEAF reached 97.7% on 6 layers, and L12 is 12 layers inside the cap. **Rule:** L12 is a 5M *elimination probe* — extended to 20M only if its 5M COV macro is within the MDE of the better of the other two arms at 5M, else eliminated and reported; among the arms with a 20M reading the best COV macro wins; among arms whose 20M macro is statistically indistinguishable, the cheapest to serve wins, and **that tie-break is labelled in the report as a product preference, not evidence** (Dylan 2026-09-01 called 33M "the upper bound of what I think is acceptable", so 33M is admissible). L12's three- and four-layer heads pass the parity check first. Cost ≈ 17 GPU-hours at the measured rate, on the box |
| **G — output width and head form** | feature = last layer only (384, M9's head) · three layers (1152, = anchor) · four layers (1536, §Recipe) · **G-MLP**: the anchor's linear head plus a per-token rank-192 GELU correction, `W_lin·x + W₂·GELU(W₁·x+b₁)`, W₁: 1152→192 (B3, residual form after the Codex pass; replaces the 768 arm, whose step the head-width probe already shows — the paper reports 384 vs 1152 vs 1536, not a trained curve) | 1152−384 · 1536−1152 · **MLP−1152** | resolved winner; **default 1152** (PLANNING §9–9b; the screen, not the probe, decides). The 384 arm is the paper's evidence for the M9 diagnosis. **G-MLP** keeps the linear path's full rank and adds the one nonlinearity the cap allows (34.96M for bge-small); fastembed serves it exactly because it precedes the mean pool (`results/m10_head_mlp_parity_box.json`, min-cos 0.99999989, zero custom ops). **Warm start, exact and deterministic:** `W_lin` = the anchor's ridge head; `W₁,b₁` = the top-192 principal directions of the frozen backbone's *per-token* 1152-d states on the fit set, centred (`b₁ = −W₁μ`; sign of each direction fixed so its largest-magnitude component is positive); `W₂,b₂` = ridge from the pooled `mean_t GELU(W₁x_t+b₁)` to the residual `t − W_lin·x̄` — exact for the training form because pooling commutes with the linear maps, so G-MLP starts *at* the anchor's fitted head plus a fitted correction; both ridges use the anchor's `warm_start` λ rule and `n_fit` (`m9/registry.json`), and the per-token PCA is a streamed 1152×1152 Gram matrix |
| **B — mix** | 100/0 · 50/50 query/document (75/25 = anchor), **matched query presentations** (3.75M query examples in every arm; document examples 0 / 1.25M / 3.75M on top; totals 3.75M / 5M / 7.5M; the document cost in tokens and GPU-hours is reported) | 100/0−75/25 · 50/50−75/25 | resolved winner; default 75/25 |
| **E — batch** | 32 · 128 at equal examples and identical schedule | 1 | resolved winner; default 32 (LEAF). **Amendment A7: E is the one family whose throughput is read** — bs128 measured 1,517 examples/s blended against bs32's 683 (`results/m10_rate_bench_box.json`), so a bs32 win must also be worth its 2.2× build cost; the lock records both the quality contrast and the GPU-hour delta, and a bs32 win that does not resolve reverts to bs128 |
| **C — init** | bge-small (or MiniLM, per F) · the M9 candidate | 1 | resolved winner; default the off-the-shelf backbone. Available only if F selects bge-small: M9's candidate is a bge-small student, so a MiniLM build has no warm start and C is skipped and reported as skipped. The closed-form ridge head warm start (M9's `m9s1c`, +0.0272) is retained in every arm regardless |
| **D — objective** | anchor (squared L2) · **D-NORM** (LEAF's ‖e‖₂) · **D-COV** (document-covariance-weighted regression, §Recipe) | D-NORM−anchor · D-COV−anchor | resolved winner with the larger margin; default squared L2. The ranking-aware class is cut (amendment A1, §Recipe); **D-COV is added from the Fable review as the one cheap lever aimed at the in-distribution ceiling** (PLANNING §13 idea 8) |

A2 exists only as a control; the build never uses more than 1.0M PAQ. **Equal examples** holds for
every family except B, which is matched on query presentations by design. **Definitions:** a
decision's *margin* is the COV macro difference between the winner's and the default's final
checkpoints in the original screen; an arm's *seed range* is max minus min of its COV macro over
its three seeds.

**Rule, per contrast (families B–G):** the difference in COV macro (family-weighted, §Surfaces)
between the two arms' final checkpoints; paired stratified bootstrap over queries within component,
**B = 200,000**, seed 0, empirical quantile (`inverted_cdf`; at 0.025/14 that is the 357th order
statistic — at B = 20,000 it would have been the 36th, Opus M9); a contrast **resolves** when the point estimate ≥ the **MDE 0.0056** **and** the
one-sided lower bound at the **0.025/14 quantile** (Bonferroni over the fourteen contrasts, fixed
whether or not F's second comparison runs) is > 0, and the sign is stable across the last two
cycle-end checkpoints. **Both constants are fixed here** — the Codex pass of 2026-09-04 struck
amendment A4's `MDE = max(0.0056, distance)` and its two-step remedy (admit LEDGER, then α = 0.05)
as an α that adapts to an observed width. The resolution number (§Surfaces) is *reported* as the
power disclosure; an underpowered contrast is reported as unresolved and reverts to default, and
the report names the contrasts the surface could not have resolved. **Family A's contrasts A3−A2 and A4−A3 are exempt from the
generic rule** and use the three-outcome rule in the table (resolved requires the corrected lower
bound > MDE); A4−A2 and A2−A1 are descriptive. **Confirmation:** for every decision whose
non-default option won, both the winner and the default are re-trained with two more seeds at
screen dose — **at most two such decisions, largest margins first** (amendment A5; it was four);
the rest revert to default. The decision stands only if the winner's margin exceeds the largest
seed range observed in either arm. Worst-case confirmation cost is now 45M examples ≈ 17
GPU-hours, plus the synthesized selected-recipe arm (5M) and the replication seed pair (10M);
PLANNING §5 has the arithmetic. Every screen verdict is artifact-specific at screen dose; never
"resolved" in the report's sense.

## Surfaces

- **COV** — the primary selection surface: **qrel-bearing** retrieval components with an
  affirmative licence at the dataset's primary source (M7's eval-use standard), decontaminated,
  and **read by no M10 decision**, admitted at M10.0-d, **weighted equally per family**
  (slices within a family averaged first). Candidates, draft records in `m10/COV_CANDIDATES.md`
  (primary-source licences checked 2026-09-01): **consumer-health** MTEB MedicalQARetrieval (CC BY
  4.0 at MedQuAD; NIH sources); **BRIGHT**, one family, six slices (biology, earth-science,
  economics, psychology, robotics, sustainable-living; CC BY 4.0 on the benchmark at its primary
  source — the same dataset-level standard that admitted CQADupStack and the six, whose documents
  are also third-party text; the document-rights caveat is disclosed and the data is evaluation-only,
  never redistributed); **legal** MTEB LegalBenchCorporateLobbying (CC BY 4.0; ConsumerContractsQA
  refused, CC BY-NC); **finance** LEDGER (CC BY 4.0 annual-report QA) once its structure and a
  100K-chunk cap are verified. Climate-FEVER refused (no licence at its primary source, as in M7).

  **Admission reopened 2026-09-04 (Dylan: non-commercial licences are admissible for validation, not
  training — `research/m7-data-licensing.md` §Rule change 2026-09-04).** A set refused *for its licence
  alone* is re-admissible, and the resuming session must re-run admission before the family floor is
  judged: **MTEB ConsumerContractsQA (CC BY-NC) is admissible**, restoring a second legal set and
  taking the count to four **without** LEDGER, whose structure was never verified. Any other CC BY-NC
  or research-only eval set is now in scope. **Still refused:** Climate-FEVER (no affirmative grant —
  a different class), and anything excluded for contamination. MS MARCO itself is admissible but is a
  **poor COV member** — every comparator trains on it and neither of ours does, so it is biased
  against us; prefer it for the within-system read under FORMS-12.
  **At least three families must survive admission** or M10 returns to Dylan; the report names the
  family count (Codex pass 6 preferred four — with the CQADupStack pair demoted, four needs LEDGER;
  Dylan may raise the floor). The two CQADupStack components are **DEV**, reported beside every COV
  read, never in the macro.

  **Scientific diagnostic surface (B4; a COV family until the Codex pass, an actioned secondary until the Opus pass).** Every clean-4
  set is scientific or biomedical and no licensed published set covers those forms outside S2ORC /
  PubMed / CORD-19, so as registered the screen could not see the headline forms. **`arxiv-title`**:
  2,000 held-out arXiv titles → their own abstract among 100K arXiv abstracts (CC0 metadata; ≈8 min
  of encode), one relevant per query, nDCG@10, teacher score reported as the denominator. **Not in
  the COV macro** — it shares form and source with arm A3's harvest, so as a family it would reward A3
  by construction and its 2,000 low-variance queries would dominate the macro's power. It is a
  **descriptive diagnostic with no action** (Opus M11: title→own abstract is near-lexical known-item
  retrieval, so a null there could be a floor effect, and it is not scidocs' title→cited-abstract
  relation); A3−A2 on the COV macro carries every data decision. Reported per arm beside FORMS-12 as
  the one real-text scientific read. Leakage rules: the
  held-out set is drawn by arXiv id *without version* at M10.0-d (seed 0), every version of a
  held-out paper is excluded from every training role, the harvest is fingerprint-screened against
  the held-out titles and abstracts, and the set joins the protected index before any harvest.
  ClinicalTrials.gov is not used (terms clause unread). COV still has no argument retrieval
  (six-set only; FORMS-12 descriptive). **Decision 12 (open)**: CUREv1 as a validation-only read.
  **Resolution number (M10.0-d, before the lock) — a power disclosure, nothing more** (A4's sizing
  struck by the Codex pass). With the contrast rule's own bootstrap, measure the distance between the
  point estimate and the one-sided 0.025/14 lower bound for the COV-macro difference between two
  models that are candidates in no M10 family — **e5-small-v2 and gte-small** (`results/FINAL_MATRIX.md`
  rows, fresh COV encodes) — scored symmetrically on the admitted surface. Only the distance is
  recorded, never which led. It is the first disclosed COV read (`m10/LEDGER.md` §4) and is
  published beside every contrast, so a reader can tell an unresolved verdict from an invisible one.
  With MedicalQA 2,048, BRIGHT 6 × ~100, CorporateLobbying 340, ConsumerContractsQA 396 and paired
  per-query SDs of 0.07–0.27, the expected distance is 0.009–0.0135 — above the MDE, so **most B–G
  contrasts are expected to be unresolved at screen dose and to revert to default; that is disclosed
  in advance, not repaired by loosening α.** LEDGER (118,048 questions) is admitted if its structure
  verifies at M10.0-d, an ordinary admission decided before any read, because it is the one
  candidate large enough to move the surface's power. The distance is measured between *unrelated*
  models, so it over-estimates the width of a same-init contrast; recorded, not corrected.
- **DEV-6** (incl. the two CQADupStack components) secondary, reported beside every COV read.
  SCREEN-3 is retired.
- **FORMS-12**: 12 × 500 held-out generated or harvested queries, overlap@10 between student and
  teacher rankings over a **1M-document evaluation sample of the already-encoded pool** (seed 0,
  fixed at M10.1) — amendment A1 deleted the *mined candidate bank*, not the evaluation sample, and
  this needs no mining. Per form. **Descriptive only** — teacher agreement on generated
  queries is a coverage diagnostic, not retrieval quality.
  **Now also permitted (2026-09-04, optional, descriptive):** the same overlap@10 diagnostic on a
  sample of **real MS MARCO dev queries**, as an external check that the 12 forms cover the natural
  query distribution. Validation only: MS MARCO text may **never** seed generation or enter the pool,
  and no MS MARCO-derived cache may be written under `work/train/sources/` (`m7src/mix.py:22-25`).
  **The stella-space distribution-overlap gate of amendment A8 (§Data) is the primary use of this
  permission** and runs on the manifest before any arm; this overlap@10 row is the post-hoc version
  of the same idea and stays optional. Neither is a substitute for COV: both are teacher-agreement
  or distributional diagnostics, not retrieval quality.
- **LoTTE-clean** (7 slices, macro over slices; its corpora — ~2.8M passages — are encoded with
  stella once, ≈ 1.3 GPU-hours at the assumed 600 docs/s, budgeted in PLANNING §6): **read #1**
  after the recipe lock.
  Before LoTTE opens, the **selected recipe is trained once as a single synthesized arm at screen
  dose** (its checkpoint hash committed); read #1 scores that checkpoint and the anchor's in one atomic batch.
  Veto rule (M9 §7, unchanged): the selection is vetoed if the selected recipe's 7-slice macro is
  worse than the anchor's by more than 0.004 AND the one-sided 97.5% paired-bootstrap upper bound
  (B = 10,000, seed 903, paired within slice) on (selected − anchor) is below −0.004; a veto means
  **the anchor recipe builds**. No other action may follow from read #1. **Read #2** pre-freeze,
  audit only. No third read.
- **Dev→six calibration**: M9's close-out rows (available only after M10.2) and M7's; forecasting
  only, never gating.
- Dev reuse is counted (`m8src/dev_reuse_m8.py`) and published.

## Final run

`m9/FINAL_LOCK.md` re-registered verbatim for M10 in the M10.2 lock with new BOUND-AT-FREEZE hashes
and an `m10-six-spent` tag — bridge as phase 1, the C-conjuncts with the empirical bootstrap bound
and the sign-flip conjunct, FEVER labelled — **with four changes (Opus: "two" understated it). (1) The
reserved conditional is `if any C-conjunct passes then execute`**, so an aim claim never stands
without its descriptive reserved rows. **(2) Amendment A3: four conjuncts, not two** — `C1a`/`C2a`
on avg-6 and `C1b`/`C2b` on clean-4 (bars 0.5042 / 0.5155 / 0.5046 / 0.5233), under
**fixed-sequence gatekeeping in the order C1b → C1a → C2a → C2b at the full one-sided 0.025 each,
stopping at the first non-rejection** (§Goal has the procedure, the planning proxies and the
`trec-covid` n=50 disclosure). **(3) The decision field is the empirical one-sided 0.025 quantile
(`lower_q025_raw`, the 250th order statistic at B = 10,000), not M9's Holm-2 0.0125, and (4) the
sign-flip conjunct is tested at 0.025 inside the same sequence** — the Codex pass of 2026-09-04
found the two quantiles named in different files. M9's claim-table row "C1 fail, C2 pass → aim
permitted" is deleted by (2). The sequence is fixed here, before any six-set
output exists.
clean-4 is M14's registered headline partition for both halves of the pair, so it carries a
pre-registered pass/fail rather than arriving as a descriptive row after the fact. Zero alpha on the reserved batch is unchanged. `m9src/final9.py`'s
scoring path is written and reviewed before M9's close-out and reused.

## Compute and costs — re-priced 2026-09-04 on measured rates (amendment A7)

**Split execution.** The 2026-09-01 ruling stands where it bites — no dose or screen size is set by
the box's wall-clock — but the box is an execution target again for everything that is not
generation, because the rate it was withdrawn on was wrong:

| configuration, M10 recipe shape, **measured on the RTX 3080 2026-09-04** (PLANNING §11) | examples/s | 200M examples |
|---|---|---|
| batch 32, 75/25, M9's two-chunk collate | 400 | 139 h |
| batch 32, single-chunk buckets (718 query / 596 document), blended 75/25 | **683** | **81 h** |
| batch 128, single-chunk buckets (2,311 / 748), blended 75/25 | 1,517 | 37 h |

The plan's imported LEAF planning rate was 560 examples/s on a rented A100; the box meets or beats
it, because at batch 32 with ~35-token queries the step is launch-bound, not FLOP-bound, so a
bigger card buys ~1.5–2× rather than 10×. Caveat, disclosed: these are fixed-shape random-token
microbenchmarks with no data loading and no evaluation, so the real trainer is slower; they bound
the *hardware*, not the pipeline. **Consequences, all registered here:** length-bucketed
single-chunk batching is part of the build (§Recipe); the screens, confirmations, COV admission and
encodes, and M10.0-c run **on the box**, which already holds ~200 GB of M9 caches the plan had
budgeted 12 GPU-hours and a day of network to re-derive; the day-one benchmark still re-derives
PLANNING §6 before anything scales.

**What the money buys.** Generation needs the cloud — Qwen3-8B bf16 is ~16.4 GB on a 10 GB card —
and the build is rented for wall-clock, not because the box cannot do it (the box would take
≈ 3–6 days at the measured rates). One rented **A100 80 GB** (H100 if its cost per example measures
lower on the smoke), ≥ 500 GB persistent disk that survives stopping the instance, SSH, and a GitHub
deploy key so the headless commit-and-push contract holds. Provider is Dylan's choice; $1.5–2.5/h
assumed, unverified Sept 2026. The instance is stopped between stages.

| line | GPU-hours | $ at 1.5–2.5/h |
|---|---|---|
| generation, ≈1.0M queries (was 3.0M) | 10–20 | 15–50 |
| build, 200M examples at the measured rate | 40–75 | 60–190 |
| cloud-side encodes the build needs + export, parity, final run | 6 | 9–15 |
| day-one rate benchmark | 1 | 2–3 |
| persistent disk, egress | — | ≈ 25 |
| **cloud total, hybrid (screens on the box)** | **57–102** | **≈ $110–280** |
| all-cloud variant (screens and confirmations rented too) | 130–190 | ≈ $220–500 |
| optional: extension cycle, 66.7M examples, each | 13–25 | 20–63 |

Hard ceiling **$1,000**, unchanged and now far from binding. **Allocation order at the lock:** every
mandatory line first at the measured rates and the billed price, then whole extension cycles from the
remainder (the second build seed is withdrawn, amendment A5). A cycle whose projected cost plus
billed spend to date would exceed the ceiling does not start.

**Day-one rate benchmark (≈ 1 GPU-hour, before any scale-up):** stella docs/s on 20K pool documents;
training examples/s at batch 32 for 2,000 steps on the 75/25 anchor mix, the 50/50 mix and MiniLM-L6;
generation requests/s per form on the 200-query smoke, end to end (prefill, decode, retries, JSON
failures); the provider's billed price. Wall-clock ≈ 1.5–2 weeks; screen arms are independent and the
box runs them while the cloud generates.

On the instance, `results/perquery.json` (tracked) is sha-verified against `6b18e3dd…` before any
scoring, and M9's parity sample is regenerated from `m9/registry.json` (tracked; seed and sha256
pinned) — `m9src/port.py` refuses a sample that does not hash to the lock, in which case the 512
texts are transferred from the box. Only what training needs travels to the instance (tokenized
corpora and teacher targets, or re-derived there from the repo and HF); nothing that only the
evaluation stages need. Reserved-set encodes run only if the reserved conditional fires.

M9's cost protocol (`instructions-m9.md` §Costs) unchanged; the frontier is reported per index
configuration, naming the one measured (`m9/RESULTS.md` rounds 1–4). TurboQuant 4-bit (Qdrant 1.19)
joins the M13 all-in quantization comparison.

## Unimpeachable by competitors — what the report must carry

1. Comparators byte-verified against the official artifacts, scored in the same exact-search
   harness, frozen as per-query vectors before any M10 number existed; the bridge check proves
   the harness unchanged.
2. Pre-registration with pushed commits as the external witness; one six-set transaction per
   milestone, each disclosed (M7, M9 close-out, M10); the reserved four's single access; dev-reuse
   count and the selection-surface table published.
3. Contamination handled three ways: stella's disclosed overlap (ArguAna, FiQA) at every headline;
   NDO-4 rows; reserved NDO-3 and LoTTE as surfaces no decision touched.
4. Training data affirmatively licensed, no MS MARCO, attribution recorded; synthetic queries from
   an open-weights generator with pinned revision, per-query provenance and removal counts.
5. Compute disclosed in examples, tokens and GPU-hours beside LEAF's ~100 A100-hours.
6. System-level framing only (§Goal): no claim isolates nano from its document tower.
7. Reproducibility: code, corpus manifests with hashes, model revisions, seeds, statistics code
   with tests, and the full screen table including losing arms and confirmation seeds.
8. **No general retrieval-quality language.** The six are development-informed by construction (the
   data taxonomy targets their forms; COV selects; LoTTE vetoes); the reserved four are the only
   untouched surface and are descriptive. Every conclusion is stated on the named datasets (Codex
   2026-09-04, "what the review missed").

## Deliverables

Frozen candidate + `m10/FREEZE.json` (`assert_releasable` with a proper run record — it walks
`work/runs/<id>.json` for `cfg.sources` and `cfg.init` across the whole lineage and fails CLOSED on
a missing ancestor, which is why M9's freeze was refused; the record must be **derived from the
run's own hash-bound `manifest.json`, never asserted by hand**, because a post-hoc record that
happens to satisfy a licence guard is precisely the artifact that must not be fabricated; see
`m9/BUILD_LOG.md`), the frontier
update, the M10 section of the report artifact, decisions logged in CLAUDE.md, handoff to M13.

## Out of scope (reopening conditions in PLANNING §7)

Document-side co-adaptation (inside M10 it breaks the pair; a tower co-trained against both query paths at once keeps it and is recommended 2026-09-01 as the next-milestone candidate (that slot was M12 then; **M16** after the 2026-09-04 renumbering), Dylan's call, PLANNING §7) · any student above 35M
(hard cap, Dylan 2026-09-01) · teacher change (stella-1.5B measured worse; Qwen3-0.6B never
screened and not the pair) · a **post-pooling** nonlinear head (no fastembed path; per-token heads
ship, §Recipe) · a cosine-space phase-1 loss (closed by algebra, `m10/EXPLORED.md`) · a
ranking-aware phase-2 loss (cut by amendment A1; reopening condition in §Recipe) · MS MARCO **in any training role** (unchanged; **validation/diagnostic use was
permitted 2026-09-04**, see the admission note in §COV) · FineWeb in any role (decisions 3 and 5) · any
change to zero.
