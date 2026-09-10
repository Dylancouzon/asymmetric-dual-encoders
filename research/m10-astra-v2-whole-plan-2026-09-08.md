OpenAI Codex v0.153.4
--------
workdir: /home/dylan/asymetric-dual-encoders
model: gpt-6-astra
provider: openai
approval: never
sandbox: read-only
reasoning effort: high
reasoning summaries: none
session id: 01a08367-e914-77e2-9ba7-797081a4537c
--------
user
# Improve this plan. Read it first — do not answer from this brief.

Your previous pass reviewed a one-page summary because the brief said everything needed was
inline. That was a mistake on our side: you never saw the plan. **This time, read the files.** Do
not answer from this page alone; it only tells you where to look and what we already know.

**No terseness limit.** Depth is what we are paying for. Long is fine if it is load-bearing.

## READ-EXCLUSION (mandatory, audited afterwards)
Never read, at any depth: `results/frozen_eval/untouched-*`, any reserved qrels cache,
`work/m9reserve`, `results/perquery.json`. No repo-wide or recursive grep.

## READ THESE
- `instructions-m10.md` — the mandate, plus the 2026-09-04 and 2026-09-05 amendment blocks
- `m10/screen_registry.json` — the ~35 arms and contrasts, MDE, quantiles, family rules,
  `outcome_to_action`, `serve_cost_order`
- `m10/PLANNING.md` — §9 (why we diagnosed the head), §11 (rates), §12 (the synthetic-data question)
- `m9/FINDINGS.md` — why attempt 1 failed
- `CLAUDE.md` — north star, hard constraints, standing directives, decision log
- `m10/STATUS.md` — current state
Name any other file you want rather than grepping for it.

## What we already decided — do NOT re-propose these
Re-deriving a settled item wastes the call. All of these are ruled and on the record:
- Teacher = `stella_en_400M_v5`, chosen on the distilled TABLE not the tower. Eleven teachers swept.
- Doc-side linear maps, document centering, whitening, top-PC removal, per-token scalar weights:
  **provably absorbable** into the table, verified to machine precision. Cosine-space distillation
  on a normalized output is likewise a no-op. Only n-gram rows and multiplicity-dependent pooling
  add capacity.
- MS MARCO excluded from training (priced +0.0058). Non-commercial licences are validation-only.
- Student ≤35M hard. Family F selected bge-small over MiniLM-L6, resolved.
- C-M9init cut; L12 cut; family A's STOP removed; MDE fixed at 0.0056 and the COV surface is now
  OBSERVED, so it cannot take a new admission.
- CUREv1 is a reported diagnostic, never selection-bearing.

## What your last pass already gave us (verified, being actioned)
1. No small-model+BM25 comparator exists — our fused headline dodges the release bar's own hybrid.
2. nano (34.5M) is LARGER than bge-small (~33.4M) and runs the same forward pass, so it buys no
   compute saving; only the lookup table does.
3. The edge cost axis (cold start, peak RSS, p95) is unmeasured for the table and for bge-small.
4. Lexical query aliases (doc2query over our generated queries) as a no-student alternative.
5. Spearman≈0 over teachers may mean compressibility is trainable, not that teachers are alike.
Do not repeat these. Build past them or contradict them.

## What we want

**A. Attack the screen you have now read.** ~35 arms and contrasts, Bonferroni denominator 13,
MDE 0.0056, a 13,416-query family-weighted selection surface, an `outcome_to_action` map, and
~52 box hours. Most contrasts were expected unresolved before we started. Name specifically:
which arms or contrasts would you **cut**, which are **mis-specified** (measuring something other
than what their rule claims), and which **decision the map gets wrong**. Quote the registry.

**B. Name the plan's biggest structural error.** One thing. The one that, if we are wrong about it,
makes the rest of the effort answer a question nobody asked.

**C. Concrete additions.** For each: the exact arm, contrast, measurement or decision to add, what
it would cost, and what it would change. Rank by (information gained) / (cost).

**D. Where the pre-registration machinery is doing harm.** We have a lot of it. Say where it is
buying rigour and where it is buying paperwork — and what to drop.

**E. The M14 paper.** Given everything you have now read, what is the strongest defensible claim
this project can make, and what should it stop claiming?

**F. Anything above these in value.**

If a number or an argument in the files is wrong, lead with that. Quote file and line.

codex
I’ll read the six named files, including the amendments and registry decision rules, then audit whether the screen’s contrasts support the decisions it claims to make. I’ll keep all reads within those paths and name any additional file before opening it.
exec
/bin/bash -lc 'wc -l instructions-m10.md m10/screen_registry.json m10/PLANNING.md m9/FINDINGS.md CLAUDE.md m10/STATUS.md' in /home/dylan/asymetric-dual-encoders
 succeeded in 0ms:
   879 instructions-m10.md
   415 m10/screen_registry.json
   522 m10/PLANNING.md
    97 m9/FINDINGS.md
   295 CLAUDE.md
   166 m10/STATUS.md
  2374 total

exec
/bin/bash -lc 'nl -ba instructions-m10.md' in /home/dylan/asymetric-dual-encoders
 succeeded in 0ms:
     1	# M10 — nano v2: the retry, built around coverage (mandate)
     2	
     3	*Written 2026-09-01 by the planning session on Dylan's direction ("M9 failed to achieve our goals
     4	… make M10 a retry … build something unimpeachable by competitors"). Evidence `m10/PLANNING.md`;
     5	M9's record `m9/FINDINGS.md`. Adversarial review: gpt-5.6-terra, read-only, six passes
     6	(`research/m10-codex-plan-2026-09-01.md`, `-plan2-` … `-plan6-`; full logs are gitignored `.log`
     7	files beside them); every finding and its disposition is in PLANNING §8, including two reviewer
     8	dissents the owner can overrule. **Amended 2026-09-01 after pass 6 on Dylan's compute ruling:** M10
     9	runs on a rented GPU budget or not at all (§Owner rulings, §Compute); the review amendments taken
    10	with it are decision 9 and PLANNING §8's amendment block. **Amended 2026-09-04 after a full plan
    11	review on Dylan's direction ("is this the best way to go about it? Is it the most efficient?
    12	… I'm not sure why we need to generate synthetic data?"): see §Amendment 2026-09-04, which is
    13	authoritative wherever an older sentence in this file disagrees.** The M9 model is **nano**; M7's
    14	table is **zero**; the product is still the pair on one stella index.*
    15	
    16	## What binds from M7 and M9 — exhaustive
    17	
    18	From `instructions-m7.md`: decision authority; vendor and source-licence rules (as relaxed in
    19	CLAUDE.md); frozen-comparator validation and pairing; pushed-freeze, ledger, crash/abort
    20	disclosure; Sonnet-only research subagents; the headless reporting files under `m10/`
    21	(STATUS/RESULTS/EXPLORED/LEDGER); the eval-use licence standard (source-level evidence before a
    22	selection set is used); CLAUDE.md's tightness rule. From `instructions-m9.md`: the bars, the
    23	permitted claim and its qualification, the six-set transaction design (`m9/FINAL_LOCK.md`,
    24	re-registered for M10 with new hashes and the one change in §Final run), the LoTTE atomic-read
    25	protocol, R1/R2/R3, the stage discipline (no evaluation before its lock; "diagnosed defect" =
    26	implementation divergence only), the 70 MB fp16 query-asset target, screen-in-closed-form first,
    27	the watch-long-runs checklist. **M10 supersedes** M9's data pool, mix, optimizer, schedule, plateau
    28	rule, phase-2 handling, screen arm list, and selection surfaces. Where M10 changes a carried
    29	mechanism, M10 controls. Read `m8/CODEMAP.md` and `m9/CODEMAP.md` before writing code.
    30	
    31	## Why M9 missed, and what M10 changes
    32	
    33	M9 retained **93.8%** of the teacher on NQ and **50–71%** on the two CQADupStack dev components
    34	(`m9/FINDINGS.md` §1): its 463K Wikipedia-QA and product queries did not cover the forms the model
    35	is tested on, and its 384-wide linear head is the subspace L2 regression cannot push past once
    36	queries are diverse (PLANNING §9). Parameter count is not the constraint. Both causes are tested
    37	before anything is built (§Screen, families A and G). M10 changes, in order of evidence (PLANNING §3):
    38	
    39	1. **Coverage** — ~4.0M query texts across 12 forms, most of them REAL text (§Data).
    40	2. **Optimizer regime** — LEAF's batch 32 and three linear 1e-4→1e-5 cycles; plateau read on
    41	   annealed checkpoints, best-to-best.
    42	3. **A wider linear output** — three mean-pooled layers concatenated (1152-d) before the head.
    43	4. **Warm start from the M9 candidate** as a screen arm.
    44	5. **LEAF's own loss** — the L2 norm ‖e‖₂, not M9's squared L2 — as the one objective arm.
    45	
    46	**The student cap is 35M and it is hard** (Dylan, 2026-09-01: "109M is not an option. This isn't
    47	low compute anymore. 33M was already in the upper bound of what I think is acceptable"). bge-small
    48	with the three-layer head is 34.5M; MiniLM-L6 with it is 23.9M. The cheaper student gets the
    49	benefit of a tie (§Screen, family F).
    50	
    51	## Owner rulings already made (Dylan, 2026-09-01)
    52	
    53	- **Git:** M9 was merged to `main` on 2026-09-01 after the repo cleanup (its six-set close-out is
    54	  still pending and runs from `m9-work`); M10 execution work happens on branch **`m10-work`**
    55	  under the headless commit-and-push contract; merges to main at stage boundaries
    56	  need Dylan's go. M9's registered six-set close-out still runs from `m9-work`, because `guard9`
    57	  pins that branch (`m9src/guard9.py:35`); the branch is kept until then.
    58	- **Compute (Dylan, 2026-09-01): "M10 won't be done on a 3080. M10 will be done on a GPU budget, if
    59	  allowed, or not at all."** Every M10 stage from M10.0-c on runs on a rented GPU under the budget in
    60	  §Compute, or M10 does not run. The RTX 3080 is not an execution target for any M10 step; it only
    61	  supplies M9's frozen checkpoint once (§Compute). Every dose and duration here is set from the
    62	  budget, not from the box.
    63	
    64	## Amendment 2026-09-04 — plan review (authoritative over any older sentence here)
    65	
    66	Dylan asked whether the plan was the best and most efficient route, questioned the synthetic data,
    67	and said *"don't over-engineer"* and *"keeping the same teacher (Stella) is the goal"*. Nothing
    68	below touches stella, the 35M cap, the frozen tower, the pair, or any observed number — no M10
    69	number exists. Evidence: PLANNING §11 (measured rates), §12 (the synthetic-data question),
    70	`m10/LEDGER.md` §3. Full disposition of what was cut: `m10/EXPLORED.md`.
    71	
    72	| # | change | why |
    73	|---|---|---|
    74	| A1 | **Family D cut from three ranking-aware arms to one arm: LEAF's L2-norm loss `L=‖e‖₂`.** Deletes the 1M-document candidate bank, the exact-mining pass and its HNSW fallback, the τ entropy rule, the 129-way D-NCE spec and the seed-rank provenance field. **A registered plateau response replaces it** (§Recipe) | LEAF (97.7% asym), EmbedDistill, arXiv 2306.11550, mxbai-edge-colbert and DistilVDR all reach 92–98% retention by pure embedding regression, so the class is *unnecessary* at our target; and it was the most machinery-heavy block in the plan. LEAF's loss is the **norm**, not the square — a one-line arm we had silently diverged from. **Corrected after the Fable review: LEAF's Appendix B is about intermediate-layer KD (MiniLM / TinyBERT / DistilBERT losses, 54.9 / 53.7 / 55.3 vs 60.7), NOT about a ranking term on top of regression — no published null exists for the class we cut, so this is a cut on cost and sufficiency, not on evidence of inertness** |
    75	| A2 | **Generation cut from 3.0M to ≈1.0M, and only for the forms no corpus contains.** The harvestable forms come from **real** text mined out of the licensed pool (titles, headings, lead claim sentences, extracted interrogatives) — new arm A3 | Three independent saturation curves put diminishing returns under ~1–1.5M (DistilVDR saturates above 75% of a 1.49M pool; SPEED log-linear to ~920K; doc2query 50–75% coverage ≈ 90–95% of max gain). And **three of the four clean-4 headline datasets have a real-text counterpart** (scidocs↔titles, scifact↔claim sentences, trec-covid/nfcorpus↔headings), so the headline can rest on real text plus the teacher instead of on the generator's prior |
    76	| A3 | **C1/C2 registered on clean-4 as well as avg-6** (§Goal, §Final run). clean-4 bars: bge-small **0.5046**, leaf-ir-asym **0.5233** | M14 registered clean-4 as the headline partition for both zero and nano on 2026-09-04 (`instructions-m14.md`), while M10 had C1/C2 on avg-6 with NDO-4 *descriptive*. Left alone, the paper's headline would carry no pre-registered pass/fail for nano. Fixed now, before any M10 number exists |
    77	| A4 | ~~**The COV resolution number is measured first and SIZES the screen**~~ **— STRUCK by the Codex pass the same day** (an α that adapts to an observed width is not pre-registration): the number is measured and *reported* as the power disclosure; MDE 0.0056 and α 0.025/13 are fixed (§Screen) | The registered MDE 0.0056 sits **below the surface's own resolution**: on a family-weighted macro whose paired SE is 0.0033–0.0048 (BRIGHT ~100 queries/slice, CorporateLobbying 340), a contrast needs ≈0.009–0.0135 to resolve at the Bonferroni bound — 0.025/13 is z = 2.89, and it was z = 2.96 at the /16 count this finding was first written against — so a contrast landing at the MDE can never resolve. Codex pass 7's objection was to comparators drawn from family F; a direction-free power quantity on non-candidates (e5-small-v2, gte-small) is a power calculation, not a selection |
    78	| A5 | **Decision 8 (second build seed, ≈100 GPU-hours) withdrawn; confirmations capped at two decisions** | Seed 1 is descriptive by construction and can trigger no action; the same hours buy ~3 extension cycles that can move the number. A screen-dose seed pair gives a replication band for ~5% of the cost |
    79	| A6 | **Family F runs SECOND (right after A), and the remaining families run on its winner** | It was seventh, which made every other verdict transfer to the build student by assumption. Reordering costs nothing and removes the assumption. Order was A → F → G → B → E → C → D; **F → A → … after the Opus pass**, so family A's stop rule is also decided on the build student |
    80	| A7 | **CONFIRMED by Dylan 2026-09-04** ("Yes, you will have 3 days of uninterrupted access to the 3080 during the weekend. The more we can prove before spending the better"). **The box is an execution target again for everything that is not generation** (and generation, decision 14). The 2026-09-01 ruling stands where it bites — the *dose* is not set by the box — but the box is where the screens and (optionally) the build run, and it holds ~200 GB of M9 caches the plan had budgeted 12 GPU-hours and a day of network to re-derive | Measured on the box 2026-09-04 (PLANNING §11): the M10 recipe shape runs at **400 examples/s** in M9's two-chunk collate and **683** blended at 75/25 in length-bucketed single chunks (718 query-bucket / 596 document-bucket; `results/m10_rate_bench_box.json` — the Opus pass caught a stale 745 here), against the plan's imported LEAF planning rate of 560. The 3080 meets or beats the assumed A100 rate because at batch 32 with 35-token queries the job is launch-bound, not FLOP-bound. Generation still needs the cloud: Qwen3-8B bf16 is 16 GB on a 10 GB card |
    81	| A8 | **Two pre-training data quality gates added** (§Data): near-duplicate and dispersion metrics per form, and a **distribution-overlap check against real MS MARCO dev queries in stella's own space** | The synthetic risk here is not wrong labels — the teacher's embedding of any text is a correct target by construction — it is distribution shift and diversity collapse, both measurable before a training step. MS MARCO is permitted for validation by Dylan's 2026-09-04 rule. FORMS-12 cannot serve this purpose: it scores student-teacher agreement on the same synthetic queries and is circular for it |
    82	
    83	### Fable adversarial review of A1–A8, 2026-09-04 — all findings actioned
    84	
    85	Verbatim findings, dispositions and the reviewer's own file list:
    86	`research/m10-fable-plan-2026-09-04.md`. Read-exclusion audit **clean**. 3 BLOCKER / 8 MAJOR /
    87	7 MINOR; verdict "not lock-ready"; **every one actioned**, and two of its numeric claims were
    88	reproduced here before acting (B2's bootstrap widths; M1's pool composition).
    89	
    90	| # | actioned as |
    91	|---|---|
    92	| B1 | A1's justification corrected — LEAF's Appendix B is intermediate-layer KD, **no published null exists for the class we cut** — and a **registered plateau response** added (§Recipe) so a flat curve is not left without an answer, which is what M9 finding #4 demanded |
    93	| B2 | **fixed-sequence gatekeeping** replaces the contradictory "Holm + fixed sequence"; the four planning proxies and the `trec-covid` n=50 disclosure are in §Goal |
    94	| B3 | the two-step remedy is registered (admit LEDGER, then α = 0.05 uncorrected with seed confirmation as the guard); z corrected; MDE-redundancy and comparator bias stated (§Surfaces) |
    95	| M1 | **the review's most valuable finding** — the pool is Wikipedia + ESCI with no scientific text, so arXiv is added as a licence-gated source, yields are measured before quotas lock, a form under 100K reverts to generation, and the clean-4 mapping is reworded as the hypothesis A3−A2 tests (§Data) |
    96	| M2 | family F runs at 20M read as a curve, gains MiniLM-L12-v2, and its cheap-student tie-break is labelled a product preference (§Screen) |
    97	| M3 | benchmark re-run as pass 2 (300 steps, seven single-chunk shapes, memory and allocator retries); the build is priced as a **range from hardware bound to M9 pipeline efficiency** (PLANNING §11) |
    98	| M4 | reopening the ranking class is explicitly a next-milestone condition on the post-final six-set number (§Recipe) |
    99	| M5 | `m10/LEDGER.md`, `m10/EXPLORED.md`, `m10/COV_CANDIDATES.md` corrected; FORMS-12 names an evaluation sample, not the deleted bank |
   100	| M6 | A7 flagged ⚠ for Dylan to strike; `m10/STATUS.md`'s box window ordered with a drop rule |
   101	| M7 | folded into B3's remedy — screen alpha is an internal selection, not a published claim |
   102	| M8 | **form-balanced query sampling** is the anchor default (§Data), zero cost |
   103	| MINORs | all adopted; hosted generation becomes the **default** (≈$20–60 against 10–20 GPU-hours plus setup), and **D-COV** is added as a second family-D arm (§Recipe) — the one cheap lever aimed at the in-distribution ceiling |
   104	
   105	**The one thing the review found that the plan still does not answer** (PLANNING §13): M9 retained
   106	**93.8% on NQ while training on NQ-like data**, and C2b's proxy sits near **95%** of the ceiling. Coverage
   107	explains the 50–71%, not the 93.8%. Only families G and E and the new D-COV touch the covered half.
   108	Recorded as an open weakness, not closed.
   109	
   110	**Withdrawn from the review's own recommendations, and why** (kept so it is not re-proposed):
   111	dropping family F to anchor on MiniLM-L6 was recommended and then **withdrawn** — it would have
   112	killed family C too (M9's candidate is bge-small, so there is no MiniLM warm start), and arXiv
   113	2306.11550's depth curve (1/2/4 layers → 86.1/92.5/96.2% retention) disagrees with LEAF's success
   114	on 6 layers. One 5M arm ≈ 2 GPU-hours is not the place to economise when it picks the build student.
   115	F stays, C stays, bge-small stays the anchor; A6 fixes the ordering problem instead.
   116	
   117	## Amendment 2026-09-04b — feasibility review (second review of the day; authoritative over older text)
   118	
   119	Dylan: *"is our goal feasible? … gaps, weaknesses, or avenues to improve? … No over-engineering."*
   120	Evidence and dispositions: `research/m10-feasibility-review-2026-09-04.md`. Nothing here touches
   121	stella, the cap, the pair, the frozen tower or any observed number.
   122	
   123	| # | change | why |
   124	|---|---|---|
   125	| B1 | **Feasibility statement in §Goal** with per-dataset arithmetic (`results/m10_conjunct_arithmetic.json`) | C1b is harder than C2a and no ≥10×-gap regression distillation exceeds ~95%; registering C2b without saying so invites "you always knew" |
   126	| B2 | **Four-conjunct claim decision table** (§Goal); release rule = **decision 11** | "C1 passes → release" was ambiguous on the likely outcome (C1a pass, C1b fail) |
   127	| B3 | **G-768 → G-MLP**, a per-token *residual* nonlinear head `W_lin·x + W₂·GELU(W₁·x+b₁)`, W₁: 1152→192 (§Screen, §Recipe); `m10/EXPLORED.md` row corrected | the "no serving path" closure held only for post-pooling heads (`results/m10_head_mlp_parity_box.json`, 34.96M, min-cos 0.99999989); the Codex pass replaced the first bottleneck form, whose output rank was ≤512 |
   128	| B4 | **Constructed scientific surface** `arxiv-title` (§Surfaces) — a COV family as proposed; a one-action secondary after the Codex pass; a *descriptive diagnostic with no action* after the Opus pass | every clean-4 set is scientific/biomedical and the screen had no surface that could see those forms; as an equal-weight family it would have rewarded A3 by construction and dominated the macro's power |
   129	| B5 | DEV-6 recipe pre-screen (STATUS item 10) **dropped** | read DEV-6 twice for defaults the screen re-decides |
   130	| B6 | **Query-asset size line** (§Recipe): ≈70.8 MB fp16 vs M9's 70 MB target | nobody had done the arithmetic for the wider head |
   131	
   132	**Considered, not changed:** a three-seed anchor noise floor in place of the two-decision
   133	confirmations (saves ≈10 GPU-h, weakens a reviewed rule); the dose, cap, teacher, family D,
   134	gatekeeping. **Open to Dylan:** decisions 11 and 12, and A7 (⚠).
   135	
   136	**Codex pass on B1–B6** (`research/m10-codex-feasibility-2026-09-04.md`; read-exclusion audit
   137	clean): 4 BLOCKER / 8 MAJOR / 4 MINOR, all actioned there — the 0.025 quantile in the proxies and
   138	the final bound; G-MLP's residual form and its training/export wrappers; family F's L12 rule and the
   139	F's second contrast; decision 11's default; `arxiv-title` demoted to a secondary surface (then to a diagnostic by the Opus pass); **A4's
   140	MDE sizing and its two-step remedy struck** (MDE 0.0056 and α fixed; the resolution number is a
   141	power disclosure); MedlinePlus / CDC / ClinicalTrials.gov out of M10; superseded prose deleted.
   142	
   143	**Opus pass after Codex** (`research/m10-opus-review-2026-09-04.md`; audit clean): 3 BLOCKER / 9
   144	MAJOR / 7 MINOR, all actioned there — the mandate's 745 / 1,331 examples/s were not in the artifact
   145	(683 / 1,517 blended; 81 h; family E reads 2.2×); LEDGER §0 split into §0a (design, M10.0-e) and §0b
   146	(data-dependent constants, close of M10.1); gatekeeping order **C1b → C1a → C2a → C2b**; the mix as a
   147	4-step window; a kill rule that can fire mid-build; the plateau rule defined; C1 worded as a system
   148	comparison; family order **F → A → …**; `arxiv-title` descriptive only; B = 200,000 with
   149	`inverted_cdf`; the licensing row that would have excluded BRIGHT corrected; PLANNING §5–§6 marked
   150	superseded; LEDGER §5's duplicate tables deleted.
   151	
   152	## Amendment 2026-09-05 — owner rulings after the verification pass (authoritative over older text)
   153	
   154	Ruled by Dylan 2026-09-05 with nothing registered trained; `m10/screen_registry.json` carries the
   155	executable form and `m10/LEDGER.md` §3 the record.
   156	
   157	| # | change | why |
   158	|---|---|---|
   159	| C1 | **Family A's STOP is removed.** A3−A2 keeps its three labels (resolved / positive not resolved / not positive) as REPORTED outcomes; the build proceeds in every case. Family A is also reported on DEV-6's CQADupStack components and FORMS-12, descriptive. M10's failure condition is the C1b release gate | COV holds none of A3's forms, and the contrast needed a point above ~0.0142 to resolve, so the STOP would most likely have fired under a true coverage effect (W12; both reviewers) |
   160	| C2 | **MiniLM-L12 and contrast F2 are cut.** F = bge-small vs MiniLM-L6 at 20M, F1 only; the Bonferroni denominator is **12** (11 one-sided at α/12, F1 two-sided at α/24 per tail = 0.025 exactly). Trained arms 14 | L12 is 12 layers / 384 hidden / 34.54M, identical to bge-small, so it tested the pretraining checkpoint, not depth; its conditional extension was an adaptive selection step (W13) |
   161	| C3 | **The weekend timeline is not binding.** The box runs as much of the screen as it can first; whatever remains moves to the rented GPU under the same registry, no re-shaping (E's two arms run together on the A100 so the contrast is not confounded by hardware) | Dylan: *"I don't care if we go over the weekend timeline … if things need to be delegated to cloud gpu it's fine. Though I prefer to do as much as we can here first"* |
   162	
   163	## Delegated authority during unsupervised windows (Dylan, 2026-09-04)
   164	
   165	Dylan: *"the agent should be able to make decisions that will help us achieve our goals if something
   166	happens during those 3 days"*, with *"good guardrails"*, and *"the model can ask a Fable sub-agent on
   167	decisions it needs to make"*. Three tiers, and the tier is decided by what the decision touches, not
   168	by how urgent it feels.
   169	
   170	- **Tier 1 — decide alone:** operational choices that change no registered constant: retries and
   171	  restarts, bug fixes, the order of runbook steps, extending the window's timeline, choosing among
   172	  alternatives the runbook already names, when to ping.
   173	- **Tier 2 — decide after a Fable consultation:** anything not covered by a registered branch that
   174	  touches the recipe, the data or the screen's evidence — an arm that cannot run as specified, a rate
   175	  that makes the plan infeasible in the window, a yield or gate outcome the text did not anticipate,
   176	  an ambiguous rule. **Mechanism:** brief a Fable subagent (`model: fable`) adversarially — the
   177	  decision, the options, the goal (win C1b), the guardrails below, the standing read-exclusion, no
   178	  sub-subagents; it returns a recommendation, its risks and what would make it wrong; the session
   179	  decides (it may disagree, saying why), records the decision **and the subagent's verbatim
   180	  recommendation** in `m10/LEDGER.md` §3 "Delegated decisions" with timestamp and commit, pushes,
   181	  pings Dylan. Prefer the reversible option; when two admissible options tie, prefer the one that
   182	  preserves the window's proof value (data, F, A). Every delegated decision is disclosed in the report.
   183	- **Tier 3 — never; Dylan only:** changing a protocol after a number it affects is observed; opening
   184	  any six, reserved or LoTTE surface outside its registered transaction; spending cloud money;
   185	  changing the teacher, the 35M cap, the pair, the bars, the conjuncts or their order, the datasets;
   186	  editing a lock after an arm has started; **overriding a registered STOP** (the family floor; A3−A2's STOP was removed by Dylan 2026-09-05);
   187	  relaxing a licence or contamination rule; releasing or publishing anything; changing a registered
   188	  constant (MDE, α, doses, seeds, quotas). A Tier-3 question stops **that branch only**: record it,
   189	  ping, continue every other branch.
   190	
   191	## Owner decisions (defaults apply until Dylan rules; each is recorded in `m10/LEDGER.md`)
   192	
   193	**States:** 1, 4, 7 default-active pending ratification · 2, 3, 5, 6, 8, 9 closed · 10, 13 adopted
   194	(strike an item to revert it) · 11 default confirmed · 12, 14, 15 adopted. This is
   195	the only pending-decision list; `m10/STATUS.md` points here.
   196	
   197	| # | decision | default while open |
   198	|---|---|---|
   199	| 1 | Ratify M9's final-lock amendment **together with the close-out amendment that strikes M9's reserved conditional** (§Stage plan, M10.2): M9's close-out is six-only and cannot spend the reserved access | blocks the close-out only |
   200	| 2 | **GPU budget — VALIDATED by Dylan 2026-09-04.** Re-priced at the measured rates after the plan review: hybrid (screens on the box, cloud for generation and the build) **≈ 44–88 cloud GPU-hours ≈ $90–245** with screens and generation on the box (A7, decision 14); all-cloud ≈ 130–190 GPU-hours ≈ $220–500. Ceiling **$1,000** unchanged and unspent. PLANNING §6 (re-derived) and §11 | approved; the day-one benchmark still re-derives §6 before anything scales |
   201	| 3 | FineWeb as a seed — **ruled out 2026-09-01** (delegated): Wikipedia and the approved corpora carry the topics; FineWeb adds a rights review and a blocklist for no measured gain. Reopening condition in `m10/EXPLORED.md` | closed |
   202	| 4 | PAQ (machine-generated questions over Wikipedia; data CC BY-SA, generation code CC BY-NC) as query text | include, from Facebook's official release (never the unofficial HF mirror); 1.0M uniform sample in the build (seed 0, file hashes pinned), 4.037M in the volume-control screen arm A2 only; attribution recorded |
   203	| 5 | FineWeb documents — **excluded 2026-09-01**: no reserved-set document fingerprints exist and creating them would open reserved corpora (`m9/LEDGER.md` §1.3) | closed |
   204	| 6 | A >35M tier — **ruled out by Dylan 2026-09-01**; 35M is a hard cap | closed |
   205	| 7 | Confirm: LoTTE read #1 withdrawn unexecuted in M9; renumbering M10/M11/M12 | as recorded in `m9/STATUS.md` and CLAUDE.md |
   206	| 8 | Second build seed at full dose — **WITHDRAWN 2026-09-04** (amendment A5): descriptive by construction, could trigger no action, and cost ≈100 GPU-hours. Replaced by a screen-dose seed pair on the selected recipe as the replication band; the freed hours go to extension cycles | closed |
   207	| 9 | The 2026-09-01 review amendments taken with the compute ruling: dose 200M examples, screen dose 5M, G-1536, the COV resolution number, extension capped by budget (PLANNING §8, amendment block and pass 7). **D-KL1, D-NCE and the seed-rank field are struck by amendment A1; the resolution number is repurposed by A4** | adopted as amended |
   208	| 10 | The **2026-09-04 plan-review amendments A1–A8** (§Amendment 2026-09-04) | adopted on Dylan's "make your changes to the plan"; any item he strikes reverts to the 2026-09-01 text |
   209	| 11 | **Release rule under four conjuncts** (§Goal claim table): does a C1a pass with a C1b fail ship as "nano"? | **no** — "release" needs C1b, the headline partition (Codex 2026-09-04: releasing on the partition whose margin is 73% fiqa would be motivated). A C1a-only pass is still *published* as a frontier measurement labelled not recommended, as `zero` was. **Dylan 2026-09-04: "Let's make sure we win enough so this isn't a question"** — the default stands and the instruction is to build for C1b |
   210	| 12 | **CUREv1 as a validation-only biomedical read** — 2,000 real clinician queries, CC BY-NC 4.0, PMC-OA full-text passages (PubMed-family), pools annotated by Qwen 2.5 72B; fingerprint-screened vs the six; never training data. Reopens the validation clause of M7's source-family rule (`research/m10-feasibility-review-2026-09-04.md` §4c) | **ADOPTED by Dylan 2026-09-04 as a reported diagnostic beside every arm** — never selection-bearing (LLM-judged pools; provenance overlap fingerprints cannot see; Codex M6). Admitted at M10.0-d with revision, licence clause and corpus provenance recorded; fingerprint-screened against the six; joins the protected index. **Withdrawn by the review:** PubMed titles / PubMedQA as training text — no affirmative grant on PubMed abstracts (NLM disclaims copyright; publishers may hold it); reopens only with per-record licence provenance (PMC-OA CC BY records) *and* the contamination rule resolved (§4a) |
   211	| 13 | The **2026-09-04b feasibility-review amendments B1–B6** (§Amendment 2026-09-04b) | adopted; strike any item and it reverts |
   212	| 14 | **Generation on the box with Qwen's official 4-bit release** — `Qwen/Qwen3-8B-AWQ`, revision `4da05a8e…`, Apache-2.0 — served by vLLM (to be installed; smoked first), instead of hosted bf16. The 12-form smoke (200 per form) is pushed as `m10/SMOKE.md` so Dylan can approve remotely; ≈1.0M queries then cost ≈1 box-day, and family F can start inside the weekend window before any cloud dollar. Hosted bf16 of the pinned base stays the fallback if vLLM will not run on the card or the smoke / A8 gates fail | **ADOPTED by Dylan 2026-09-04 ("Go on 14")**. The §Data contract now reads "an official Qwen release, pinned by revision"; the A8 diversity gate guards against 4-bit repetition; the weekend runbook in `m10/STATUS.md` carries the fallback branch |
   213	| 15 | **Conditional pre-approval of the generation smoke** (§Data contract): the contract rate is computed by the session, the on-form rate is judged by an independent Fable subagent against each form's registered description, the sample and verdicts are pushed as `m10/SMOKE.md` with a GitHub issue and a phone ping, and generation starts after a **six-hour veto window** unless Dylan objects; a late veto drops the form from the build. The report discloses that the on-form read was model-judged with owner review after the fact | **ADOPTED by Dylan 2026-09-04 ("Yes to decision 15")** — the risk of waiting (a lost weekend) outweighs the bounded risk of one off-form form, which quotas per form make cheap to drop or regenerate |
   214	
   215	## Goal, bars, and the permitted claim — unchanged from M9
   216	
   217	nano = a ≤35M transformer query encoder serving stella-400M's frozen 1024d index (the SAME index
   218	as zero). Bars, paired on `results/perquery.json` (sha `6b18e3dd…`, irreplaceable):
   219	
   220	| | avg-6 | NDO-4 |
   221	|---|---|---|
   222	| teacher ceiling (stella symmetric) | 0.5744 | 0.5640 |
   223	| **AIM: leaf-ir-asym** | **0.5155** (89.7% retention) | 0.5233 |
   224	| **RELEASE BAR: bge-small** | **0.5042** (87.8%) | 0.5046 |
   225	
   226	**NDO-4 is clean-4** (`nfcorpus`, `scidocs`, `scifact`, `trec-covid`), the partition M14 registered
   227	on 2026-09-04 as the headline for both halves of the pair.
   228	
   229	C1 nano-dense > bge-small (release); C2 nano-dense > leaf-ir-asym (aim). Both dense-vs-dense;
   230	fused rows descriptive only. **Amendment A3, 2026-09-04: C1 and C2 are each registered on BOTH
   231	partitions** — `C1a`/`C2a` on avg-6 (bars 0.5042 / 0.5155) and `C1b`/`C2b` on clean-4 (bars
   232	**0.5046 / 0.5233**), all four reproducible from the frozen comparator rows by
   233	`scripts/clean4_bars.py` → `results/m10_bars.json` (a comparator-only read taken 2026-09-04 before
   234	any nano number existed).
   235	
   236	**Multiplicity: fixed-sequence gatekeeping, NOT Holm** (corrected after the Fable review, which
   237	found the two named together and incompatible — Holm orders by p-value, a fixed sequence does not,
   238	and a text permitting either lets a future session choose after seeing the numbers). Registered
   239	order **C1b → C1a → C2a → C2b** — the release gate of decision 11 first (the stress table has a case,
   240	`arguana` at 65%, where clean-4 passes and avg-6 fails, which the old C1a-first order could never
   241	test; Opus 2026-09-04), then the M9 headline sentence's conjunct C2a before the stretch aim C2b; **if
   242	Dylan makes C1a the release gate, C1a goes first**. Each conjunct is tested at the **full one-sided
   243	0.025** and testing stops at the first non-rejection. This controls the family error rate and, unlike Holm-4, **costs
   244	the avg-6 release claim nothing** — under Holm-4 the smallest-p conjunct would have needed 0.00625
   245	and C1a's bound would have widened ~11% against the two-conjunct design it replaces.
   246	
   247	**Planning proxies, not pass points** (`results/m10_bars.json`, `planning_proxies`): the bar plus the
   248	comparator-pair (leaf vs bge) bootstrap width at the **registered 0.025 quantile** — the Codex pass of
   249	2026-09-04 caught the script still on M9's Holm-2 0.0125. Nano's own interval depends on its per-query
   250	differences, so no retention figure "passes" by arithmetic; the rule is the final run's own bound.
   251	
   252	| conjunct | bar | proxy width | proxy point | = retention of ceiling |
   253	|---|---|---|---|---|
   254	| C1a release, avg-6 | 0.5042 | 0.0089 | 0.5131 | 89.3% of 0.5744 |
   255	| C1b release, clean-4 | 0.5046 | 0.0122 | 0.5168 | 91.6% of 0.5640 |
   256	| C2a aim, avg-6 | 0.5155 | 0.0089 | 0.5244 | 91.3% |
   257	| C2b aim, clean-4 | 0.5233 | 0.0122 | 0.5355 | **94.9%** |
   258	
   259	**Disclosed with it (B1, `results/m10_conjunct_arithmetic.json`):** under a *uniform-retention
   260	proxy* the four sit at 89.3 / 91.3 / 91.6 / 94.9% (C1a / C2a / C1b / C2b) — a lens, not an ordering:
   261	retention is distribution-specific (M9: 93.8% covered, 50–71% uncovered), so heterogeneous
   262	retention can reorder them. To equal bge-small per dataset: scifact 91.4 · nfcorpus 83.0 · fiqa 72.9
   263	· arguana 94.7 · scidocs 85.7 · trec-covid 92.0%; LEAF beats the ceiling on trec-covid. **Stress:**
   264	one dataset at 65% with the rest at 94% clears no conjunct if it is `trec-covid` or `scifact`, only
   265	C1a if `nfcorpus`. At uniform 92%, fiqa (disclosed stella training data) supplies 73% of nano's avg-6
   266	margin — why clean-4 is the headline. **Prior, registered before any nano number:** M9 held 93.8% on
   267	its covered form; a bounded literature search (pure embedding regression, ≥10× teacher/student
   268	parameter gap; `research/m10-feasibility-review-2026-09-04.md` §2) found no result above ~95%, and
   269	EmbedDistill's 95–97% at ~10× used labels and score distillation. So C1a is expected if coverage
   270	works, **C1b and C2a are the contest, and C2b is a low-prior stretch aim** — the screen results that
   271	would raise the prior are G-MLP or D-COV moving `nq-250k` retention from ~94% toward 96–97%.
   272	clean-4's interval is ~36% wider than avg-6's because `trec-covid` carries a
   273	quarter of the clean-4 macro on **50 queries**, and C2b therefore demands LEAF-level retention on
   274	the headline partition. Registering clean-4 is still right — M14 headlines it — but the plan is
   275	registered knowing C2b is the hardest of the four by a wide margin, not knowing only that its bar
   276	is 0.0078 higher. **clean-4 is also the most data-designed partition** (Opus): §Data names
   277	scidocs↔titles, scifact↔claims, nfcorpus/trec-covid↔headings, so the headline controls *teacher*
   278	contamination while maximising *training-design* targeting — two different exposures, both stated. **C2 is a whole-system comparison** (stella documents + nano queries
   279	vs arctic documents + LEAF queries): different document towers, index sizes, encode costs and
   280	disclosed teacher overlap. It supports exactly M9's verbatim headline sentence and **no statement
   281	about nano versus LEAF's query tower**; the report carries both systems' retention against their
   282	own teachers, index bytes, document-encode cost and query latency beside the number. NDO-4 and
   283	reserved NDO-3 stay descriptive.
   284	
   285	**Claim decision table under four conjuncts (B2).** Testing stops at the first non-rejection:
   286	
   287	| outcome (order C1b → C1a → C2a → C2b) | release? | permitted claim |
   288	|---|---|---|
   289	| C1b fails | **not a release** (decision 11 default): the checkpoint is published for reproducibility and as a frontier point, labelled *not recommended*; the card states it did not resolve above the bge-small system on the contamination-controlled partition. **M13 consequence (Opus):** a not-recommended model does not enter the upstream FastEmbed PR, so the single-PR ruling then covers zero and the tower only — Dylan decides this with decision 11 | none |
   290	| C1b passes, C1a fails | release | "the stella-document + nano-query system outperformed the bge-small system on the four contamination-controlled datasets"; avg-6 reported unresolved |
   291	| C1b, C1a pass, C2a fails | release | the same sentence on both partitions |
   292	| C1b, C1a, C2a pass, C2b fails | release | M9's verbatim headline sentence (avg-6); the paper states the aim was missed on the headline partition |
   293	| all four pass | release | the headline sentence on both partitions |
   294	
   295	**C1 is a system comparison too** (Opus M7): bge-small is a symmetric 384-d system with its own
   296	66.7 MB query asset and its own index, so every C1 sentence carries index bytes and document-encode
   297	cost beside it, exactly as C2 does. M9's "C1 fail, C2 pass → aim claim permitted" row does not
   298	survive gatekeeping and is deleted.
   299	
   300	Additional mandatory disclosures: per-dataset retention, the
   301	synthetic-data provenance table (generator revision and terms, prompts, seed sources and ids,
   302	counts per form, removal counts per screen), the selection-surface table (licence evidence,
   303	revisions, sizes, removals from training data), dose in examples / tokens / GPU-hours beside
   304	LEAF's ~100 A100-hours, and the dev-reuse count.
   305	
   306	## Stage plan
   307	
   308	- **M10.0 DIAGNOSIS + SCREEN LOCK** (no six-set, reserved or LoTTE output exists during this stage).
   309	  (a) Mac diagnostics — **done 2026-09-01**, `m10/RESULTS.md`, PLANNING §9–9c: the rank probe
   310	  (a 384-d subspace keeps 99.5% of one query distribution and 90–93% of three; 98–100% at 640 —
   311	  evidence about the class under L2 regression, not a bound), the head-width probe (a frozen
   312	  bge-small retains more with each pooled layer), and the serving-parity check (fastembed 0.8.0
   313	  reproduces the three-layer per-token head to 2e-7; 34.5M parameters). **Action taken:** the head
   314	  is applied per token — linear by default, G-MLP's residual form as an arm (§Screen); the pooled
   315	  feature widens to three layers; family G decides. Repeat the
   316	  parity check on MiniLM's head before family F may select it, on the four-layer head before G's
   317	  1536 arm runs, and on the trained artifact with M9's locked parity sample before the freeze.
   318	  (b) Capacity probe (`m9src/capacity_probe.py`, unchanged) — **optional, report-only**: no outcome
   319	  changes an M10 action under the hard cap; runs only on an idle GPU; its 109M student is
   320	  768-hidden, so any gain is partly width.
   321	  (c) Per-component DEV-6 read of the M9 candidate incl. `heldout-longq` (the baseline row).
   322	  **Runs on the box** (amendment A7): it holds the checkpoint and the encode caches.
   323	  (d) **COV admission** (§Surfaces): for every candidate component, record in `m10/LEDGER.md` §2
   324	  its primary-source licence URL and terms, HF repo and revision, corpus size, query count, qrels
   325	  format and metric, its corpus-level contamination check and its fingerprint screen against the
   326	  six and the reserved four. A component is named COV only after that record is pushed. **COV
   327	  admits only surfaces no M10 decision has read**: the two CQADupStack dev components were scored
   328	  by the Mac diagnostics (PLANNING §9, 86 raw reads) and stay in DEV-6. Then every admitted COV
   329	  corpus, query set and document set joins the protected index (`m8src/protected_filter`) before
   330	  any seed is drawn or any harvested, PAQ or synthetic text is constructed. **Re-run admission under
   331	  Dylan's 2026-09-04 licence rule before judging the family floor: ConsumerContractsQA (CC BY-NC) is
   332	  re-admissible, restoring the legal family's second component (three family IDs without LEDGER, §Surfaces).** The **COV resolution number** (§Surfaces) is
   333	  measured on the admitted surface and pushed before (e) as the screen's **power disclosure** (A4's
   334	  sizing struck by the Codex pass). The `arxiv-title` draw happens at the start of M10.1's harvest, once the artifact exists (§Surfaces).
   335	  (e) **Screen lock**: `m10/LEDGER.md` §0 (skeleton committed 2026-09-01) fixes every arm of
   336	  §Screen (fifteen arms), order, doses, seeds, surfaces, the thirteen contrasts, MDE 0.0056 and the
   337	  fixed 0.025/13 bound, confirmation design and outcome→action maps. **Pushed right after M10.0-d,
   338	  before any harvest** (Codex B4); the data-dependent constants are §0b, pushed before the first arm.
   339	- **M10.1 DATA.** Harvesting of the real query-like text (§Data), generation for the non-harvestable
   340	  forms under the §Data contract (200-query smoke per form first), PAQ samples, decontamination
   341	  against the protected index (now including COV) and the six's documents, the FORMS-12 hold-out,
   342	  teacher targets, `results/m10_data_manifest.json` with hashes and the provenance table.
   343	  **Before any arm and recorded in the manifest**, the two A8 quality gates: per-form
   344	  near-duplicate and dispersion metrics, and the stella-space distribution-overlap check against
   345	  real MS MARCO dev queries (validation only).
   346	- **M10.2 SCREEN + RECIPE LOCK.** The arms of §Screen, the confirmation runs, then one pushed lock
   347	  commit with every field of M9's M9.2 list filled — including the objective, the best-to-best
   348	  plateau/extension rule on annealed checkpoints, the GPU-hour allocation under the ceiling (build,
   349	  extensions), the final-run registry with the four C-conjuncts and their fixed sequence, and
   350	  LoTTE read #1's manifest. Codex and Fable review the pushed lock. **Then, and only then,** M9's close-out runs:
   351	  its registered six-set transaction **amended before execution to six-only** (the `if C1 then
   352	  execute` reserved conditional is struck; disclosed and ratified under decision 1), so M9 cannot
   353	  spend the reserved access. Its rows are a forecasting calibration and a whitepaper frontier
   354	  point; nothing in M10 may change on them. **Wherever those rows are published — frontier table,
   355	  whitepaper, model card — they carry M9's build-provenance disclosure with them:** the M9
   356	  candidate's provenance spans TWO build lock states (mid-build infrastructure repair, session
   357	  force-reopened; `m9/LEDGER.md` "M9.3 BUILD PROVENANCE DISCLOSURE" and `m9/BUILD_LOG.md`), and it
   358	  may never be described as an uninterrupted single-lock run. LoTTE read #1 then runs as the
   359	  registered veto on the selected recipe.
   360	- **M10.3 BUILD.** One candidate under the kill and extension rules; export + parity; freeze;
   361	  pre-freeze review; LoTTE read #2 (audit only). The screen-dose seed pair (amendment A5) is the
   362	  replication band; no full-dose replica runs.
   363	- **M10.4 FINAL.** The six-set transaction → decision → the registered reserved conditional.
   364	
   365	## Data
   366	
   367	**Query corpus (~3.7M unique texts, query role) — amendment A2: most of the form breadth is REAL text.**
   368	
   369	| source | build count | licence | role |
   370	|---|---|---|---|
   371	| M9 real queries (hotpotqa, squad, esci, mrtydi, nqopen, triviaqa; fever out) | 463K | CC BY-SA / Apache | real forms |
   372	| PAQ sample (decision 4) | 1.0M | CC BY-SA (data) | factoid volume — capped so it cannot dominate |
   373	| **Harvested real query-like text** from the licensed document pool (§Harvest) | ≈1.25M | inherits the pool's licences; no new rights surface | the harvestable forms, with **no generator prior** |
   374	| **Synthetic, Qwen3-8B (Apache-2.0). DEFAULT (decision 14): Qwen's official 4-bit release `Qwen/Qwen3-8B-AWQ`, revision `4da05a8e…`, served by vLLM on the box. Fallback, the only one: the bf16 base (revision `b968826d…`) under the same vLLM contract on a rented GPU after Dylan returns, ≈10–20 GPU-hours; never a third-party API** | ≈1.0M | generated under the generator's terms; provenance pinned; **not redistributed** without review | ONLY the non-harvestable forms |
   375	
   376	**Form taxonomy — 12 forms**, quotas locked at M10.1 (±10% realized), and each form is assigned to
   377	exactly one of the two sources. **Harvested (real text, ~250K each, ≈1.25M):** paper-title query ·
   378	scientific claim (a statement) · 2–4-word keyword query · factoid question · product-search query.
   379	**Generated (~143K each, ≈1.0M):** how-to / troubleshooting question with title and body · long
   380	counter-argument paragraph (120–220 words) · finance / personal-economics question · comparison
   381	question · yes/no verification question · conversational multi-sentence request · **consumer-health
   382	question** (generated, not harvested, since MedlinePlus and CDC were kept out of M10 — Codex B6
   383	found it listed under both). A harvested form that falls under 100K reverts to generation at ≈143K
   384	and the split is reported as realized. **The assignment's motivation, stated as the hypothesis it is:** three of the four clean-4 headline
   385	datasets have a real-text counterpart (scidocs↔titles, scifact↔claim sentences, trec-covid and
   386	nfcorpus↔headings and consumer-health), so *if* harvesting covers those forms the headline partition
   387	rests on real text plus the teacher rather than on the generator's prior. **That is what contrast
   388	A3−A2 tests; it is not established.** The generated half covers the interactive forms no corpus
   389	contains, which map to the non-headline sets and to COV's forum-style families.
   390	
   391	**Harvest sources — the Fable review found the existing pool cannot carry the scientific forms.**
   392	The 6.15M pool is `esci-prod` (808K product listings), `hotpotqa-corpus` (5.23M Wikipedia),
   393	`mrtydi-docs`, `squad-ctx` and `fever-pos` (excluded) — Wikipedia and products, **no scientific
   394	text**. Harvested from it, "paper title" would be Wikipedia article titles and product names and
   395	"scientific claim" would be Wikipedia lead sentences, neither of which is the form scidocs and
   396	scifact test. Two consequences, both binding:
   397	- **A scientific source is added for those two forms: arXiv metadata (titles and abstracts).** It
   398	  must clear the same licence gate as any training source before use — primary-source evidence of a
   399	  commercial-use grant recorded in `m10/LEDGER.md` §2 (the Kaggle release is CC0 1.0; the OAI feed
   400	  is not, so the record must name which artifact and revision is used). It is a different source
   401	  family from S2ORC, PubMed, CORD-19 and NutritionFacts, all of which stay excluded for
   402	  contamination, and every document passes the existing screens against the six's documents. **If
   403	  the licence evidence does not clear, both forms revert to generation and the report says so.**
   404	- **Yields are measured before quotas are locked.** For every extraction rule: post-dedup,
   405	  post-screen unique count, pushed to `m10/LEDGER.md` §1 at M10.1 before the per-form quotas are
   406	  fixed. **A harvested form that yields under 100K post-dedup reverts to generation**, and the
   407	  realized harvested/generated split is reported rather than assumed. `consumer-health`
   408	  is **generated** from Wikipedia-medical seeds. Sources verified 2026-09-04b (clauses in
   409	  `research/m10-feasibility-review-2026-09-04.md` §4b) but **kept out of M10**: MedlinePlus
   410	  government-authored topics and CDC pages are US public domain yet are MedQuAD's sources, so
   411	  harvesting them would let the MedicalQA COV read reward A3 by construction (Codex M7);
   412	  ClinicalTrials.gov's terms clause could not be read. Out on licence: DailyMed, OpenAlex,
   413	  bioRxiv/medRxiv, WHO, Cochrane, everything PubMed / PMC / Europe PMC (no affirmative grant on
   414	  abstracts). In: **arXiv metadata** (CC0).
   415	
   416	**§Harvest — the real-text pipeline (M10.1, deterministic, no model in the loop).** From the 6.15M
   417	pool documents and the Wikipedia seed corpus, all under licences already approved for training:
   418	titles and headings as-is (title and keyword forms); declarative lead sentences of abstract-like
   419	passages, filtered to 8–40 words with a finite verb and no first person (claim form); sentences
   420	ending in `?` extracted with their preceding sentence as optional body (factoid, consumer-health,
   421	product forms, routed by the source corpus). Every harvested string goes through the same screens,
   422	quotas and hold-out as a generated one, and the manifest records the extraction rule, source
   423	document id and per-rule yield. **Seeds for the generated half:** Wikipedia stratified by
   424	top-level category (CC BY-SA) and the approved pool corpora. FineWeb is out of M10 (decisions 3
   425	and 5). MS MARCO may never seed either half. **Contaminating source families — never seeds, never regression text, and
   426	excluded from COV** (`research/m7-data-licensing.md` map): S2ORC / Semantic Scholar; PubMed;
   427	NutritionFacts.org and its mirrors; CORD-19; StackExchange personal finance (money.SE) and Reddit
   428	finance; args.me / idebate; every six-set and reserved corpus.
   429	
   430	**Generation contract (M10.1, decisions 14 and 15):** generator `Qwen/Qwen3-8B-AWQ`, revision
   431	`4da05a8edb55c6046cce958586c33b61da07bb79`, served by vLLM on the box (version, launch flags and health assertion pinned in
   432	`m10/STATUS.md` step 0a; **thinking disabled**, `chat_template_kwargs={"enable_thinking": false}`);
   433	sampling temperature 0.8, top-p 0.95, `max_new_tokens` per form (60, or 400 for the argument and
   434	conversational forms), `seed = blake2b-64(seed_passage_id)`; the reply must parse as one JSON list of
   435	exactly n strings (`m10src/forms.parse`, strict — no preamble); one retry on a contract failure, then
   436	the seed is dropped; exact-duplicate queries removed. **Fallback, and the only one:** the bf16 base
   437	`Qwen/Qwen3-8B`, revision `b968826d9c46dd6066d109eabc6255188de91218`, served by the **same vLLM contract on a rented GPU after
   438	Dylan returns** — never a third-party API, which could not honour the pinned revision and per-request
   439	seed. It fires if vLLM will not run on the card (step 0a's bounded attempts), if the throughput floor
   440	is not met, or if a form fails its gates twice under AWQ (that form alone is re-smoked under bf16).
   441	**Smoke (decision 15):** the **seven generated forms**, 200 queries each. A form passes when
   442	≥ 90% of replies meet the contract (computed by the session) **and** ≥ 80% of a 50-query sample is
   443	judged on-form by an **independent Fable subagent** against the form's registered description in
   444	`m10src/forms.FORMS` (verdicts pushed with the sample). The sample is pushed as `m10/SMOKE.md`, a
   445	GitHub issue "M10 smoke approval" is opened and Dylan is pinged; **generation of a passing form
   446	starts after a six-hour veto window** unless Dylan objects there (`redraft: <form>: <note>`) or via
   447	Remote Control. A failing or vetoed form's prompt is revised at most twice, each revision recorded in
   448	`m10/LEDGER.md` §1 with its hash before the next smoke; a form that fails twice is **dropped from the
   449	build**, its quota not redistributed, and reported. A veto arriving after generation started drops
   450	that form's queries from the build and, time permitting, regenerates under a redraft. The report
   451	discloses that the on-form read was model-judged with owner review after the fact. The full state
   452	machine is `m10/STATUS.md` step 1. **Seeds are pre-filtered:** a seed
   453	passage that exact- or near-matches the protected index is never used.
   454	
   455	**Screens on every generated or PAQ query, thresholds fixed here (M7's fingerprints,
   456	`m7src/decontam.py`):** exact `blake2b-64` match or word-8-gram bottom-32 sketch ≥ 8/32 against
   457	(i) the protected index (six + dev + reserved + LoTTE **+ admitted COV queries and documents**),
   458	(ii) the six's documents; (iii) any word-5-gram shared with the query's own seed passage (a
   459	copied span is not a query) — **for harvested text this screen is against the source document
   460	minus the harvested span itself, since a harvested title IS a span of its document**;
   461	word-4-gram containment for 4–7-word queries. The M9 real-query pool
   462	and the document pool are re-screened against the COV additions (R1 removes matching queries;
   463	matching pool documents are
   464	removed too). Removal counts per screen, per form and per COV component are recorded **before any
   465	COV component is scored**. **FORMS-12 hold-out:** 500 seed documents per form are set aside first;
   466	queries generated or harvested from them are never trained on.
   467	
   468	**Quality gates (amendment A8), executed on the immutable manifest before any arm, recorded in
   469	`m10/LEDGER.md` §1; each is a REPORTED number with one registered action.** The risk this addresses
   470	is not label noise — there are no labels, and the teacher's embedding of any text is a correct
   471	target by construction — but distribution shift and diversity collapse.
   472	1. **Diversity, per form:** after exact dedup, a query is a *near-duplicate* if its word-8-gram
   473	   bottom-32 sketch matches ≥ 16/32 with an earlier query of the same form (the earlier one is the
   474	   representative); the rate is near-duplicates ÷ the post-exact-dedup count. **Action:** above 25%
   475	   the form keeps only its representatives (a real cut, never topped up); if fewer than **50,000**
   476	   remain the form is dropped from the build and reported. Mean pairwise stella-space cosine is a
   477	   diagnostic with no threshold.
   478	2. **Distribution overlap against real queries:** encode a 50,000-query sample of **MS MARCO dev**
   479	   (validation only — never a seed, target, negative or gradient; no cache under
   480	   `work/train/sources/`) and each form's sample with stella in the query role, and report each
   481	   form's mean cosine to its nearest real-query neighbours plus the two-sample energy distance.
   482	   **Action: none — this is a disclosed diagnostic**, because MS MARCO is web-search-shaped and a
   483	   form legitimately unlike it (paper titles, arguments) must not be penalised for that. It exists
   484	   so the report can state how far the training distribution sits from a real one, which is the
   485	   one thing M9's coverage failure had no outside measurement of. FORMS-12 cannot serve this
   486	   purpose: it scores student-teacher agreement on the same synthetic queries and is circular for it.
   487	
   488	**Document corpus (document role):** the M9 pool, 6.15M documents, re-screened as above. FineWeb
   489	documents are excluded (decision 5).
   490	
   491	**Form-balanced query sampling (Fable M8, zero cost).** 463K real + 1.0M PAQ makes ~37% of query
   492	texts factoid — the one form M9 already retains at 93.8% — so uniform-by-example sampling would
   493	spend 37% of the query gradient where there is nothing to gain. The anchor therefore samples
   494	query-role examples **balanced across the 12 forms** (each form's presentation share equal, texts
   495	drawn with replacement within a form), and the lock records the realized shares. Family A2 still
   496	measures raw PAQ volume, and the unbalanced variant is available as a reported diagnostic, not an arm.
   497	
   498	**Mix:** by *example*, decided by screen family B; default **75% query-role / 25% document-role**
   499	examples as a **presentation share over a window of 4 steps** — 3 query-bucket steps, 1
   500	document-bucket step — because single-chunk length bucketing makes every step pure-query or
   501	pure-document (Opus M4; the two-chunk per-step form costs the whole speedup A7 rests on). Family B:
   502	100/0 = query steps only; 50/50 = alternating. Query-role examples get raw bytes (prompt policy (b)); document-role examples
   503	carry M9's fixed document-role marker; teacher targets use the s2p template for queries and raw
   504	bytes for documents. The same student input never maps to two teacher targets.
   505	
   506	**Hard candidates, the mining pass, the 1M bank, the HNSW fallback and the seed-rank field are all
   507	struck by amendment A1** — nothing in M10 needs a candidate list once family D is a single
   508	loss-form arm. Reopening condition in `m10/EXPLORED.md`.
   509	
   510	## Recipe (defaults; screen families decide the marked items)
   511	
   512	- **Student:** bge-small-en-v1.5 is the **screen anchor**; the **build student is decided by family
   513	  F**, MiniLM-L6-v2 by default. **Amendment A6, as amended after the Opus pass: F runs FIRST, and every later family including
   514	  A is screened on F's winner** — it used to run seventh, which made every other verdict transfer to
   515	  the build student by assumption. Before F can select MiniLM, its three-layer head passes the same
   516	  export and fastembed parity check as bge-small's (M10.0-a; the Mac or the box can run it). **Feature [family G]:** masked mean-pooled hidden states of
   517	  layers 12, 8 and 4 concatenated (1152-d; MiniLM-L6: layers 6, 4, 2) → Linear(1152→1024) → L2
   518	  normalize; head 1.18M parameters, 34.5M total for bge-small. G's fourth arm adds layer 2
   519	  (MiniLM-L6: layer 1) → 1536-d, head 1.57M, 34.9M total, under the cap; its head passes the same
   520	  parity check first. Warm-started in closed form (ridge)
   521	  for the bge-small init; the M9-candidate init keeps its 384-d head and zero-initializes the two
   522	  extra layers' columns. Exported per token so fastembed's mean pooling reproduces it exactly.
   523	  Any per-token head, linear or not, ships this way (B3; `m10/EXPLORED.md`): the **trainer pools
   524	  after the per-token head** (masked mean, then normalize — the parity script's reference form) and
   525	  the **export wrapper emits the per-token output** for fastembed to pool; a wrapper-parity test
   526	  precedes any G-MLP arm (M9's trainer pooled before the head and must be ported). **Query-asset
   527	  size (B6):** a *projection* — bge-small or MiniLM-L12 with the three-layer head ≈70.8 MB fp16
   528	  (10⁶ bytes) against M9's 70 MB *target*, whose rule is a logged, measured quality justification
   529	  (`instructions-m9.md` §3); MiniLM-L6 ≈49 MB. Measured at export with tokenizer bytes and CPU
   530	  latency (`m9src/edge_cost.py`), including G-MLP's per-token cost if it wins.
   531	- **Phase 1:** squared L2 on unit-norm teacher vectors, fp32 loss, bf16 autocast (M9 form). A
   532	  cosine-space variant is closed by algebra, not measurement (`m10/EXPLORED.md`): the head ends in
   533	  L2 normalize and the targets are unit-norm, so ‖a−b‖² = 2−2cos and the gradients agree up to a
   534	  factor of 2.
   535	- **Objective [family D], amendment A1 — one arm, not three.** The single arm is **D-NORM**:
   536	  LEAF's own loss, the L2 **norm** `L = ‖t − s‖₂` on unit-norm vectors, in place of the squared L2,
   537	  from the first step. This is what the 97.7%-retention system minimised and M9 had silently
   538	  diverged from it; it is a one-line change and adds no data structure. Default is the squared L2.
   539	  **The ranking-aware class (KL over mined candidates, InfoNCE on the seed passage) is CUT** — with
   540	  it the τ entropy rule, the 1M-document bank, the mining pass and the 129-way spec. Evidence:
   541	  LEAF's Appendix B added distillation terms to plain regression and found no improvement, and
   542	  every comparable asymmetric system (LEAF, EmbedDistill, arXiv 2306.11550, mxbai-edge-colbert,
   543	  DistilVDR) reaches 92–98% retention without one, so the class is **unnecessary at our target**;
   544	  cutting it is a cost-and-sufficiency judgement. **Corrected after the Fable review:** LEAF's
   545	  Appendix B is about *intermediate-layer* KD (MiniLM / TinyBERT / DistilBERT losses at
   546	  54.9 / 53.7 / 55.3 against 60.7), not about a ranking term on top of regression, and EmbedDistill's
   547	  own ablation composes the two additively — **no published null exists for the class we cut.** The
   548	  earlier wording claimed one and was wrong.
   549	- **The registered plateau response [replaces the cut class] — this is what M9 finding #4 asked
   550	  for.** M9's lesson was that a phase 2 must be specified at lock or a flat curve has no registered
   551	  answer; cutting family D without one would recreate exactly that. So: at every cycle end the run
   552	  records per-form retention (FORMS-12 by form) and per-family COV. **If the plateau rule fires (§Kill) while the dev→six forecast is below the release bar and
   553	  `max_extension_cycles` is not exhausted, exactly one registered top-up cycle runs, pre-empting the
   554	  stop once; afterwards the form weights revert to balanced and the extension rule resumes** — one
   555	  further cycle of 66.7M examples, linear 1e-4→1e-5 as cycle 3, with the **bottom two forms by
   556	  FORMS-12 retention at 2× presentation weight** and everything else unchanged. It needs no new data,
   557	  no candidate list and no new loss; it is the plan's own coverage thesis applied to its own curve.
   558	  It fires at most once, costs one extension cycle, is counted against `max_extension_cycles`, and
   559	  the lock records the forecast formula and the tie-break for "bottom two".
   560	- **Reopening the ranking-aware class:** only as its own milestone, never mid-M10. The condition is
   561	  read on the **post-final six-set number** (the release bar is not observable during the build —
   562	  Fable M4), so it is a next-milestone decision by construction: it reopens if M10's final avg-6 or
   563	  clean-4 dense row misses its release conjunct by less than 0.01. Recorded so nobody re-derives it.
   564	- **D-COV [family D], added 2026-09-04 from the Fable review.** `L = (s − t)ᵀ Σ (s − t)` with `Σ`
   565	  the covariance of the pool's frozen stella **document** vectors, normalized to unit trace and
   566	  shrunk as `(1−α)Σ̂ + αI/1024` with **α = 0.1** fixed here (never tuned on a selection surface).
   567	  One 1024×1024 matrix computed once from vectors that already exist; no bank, no mining, no new
   568	  data, no serving change — the head stays linear and fastembed-exact. **Why it is not a
   569	  reparametrisation:** plain L2 is reconstruction, which spends a rank-limited output equally on
   570	  every direction of stella's query space; weighting by document covariance spends it on the
   571	  directions in which documents actually differ, which is what nDCG rewards. Under the rank limit
   572	  PLANNING §9 measured, that is a different subspace, so it is genuine new behaviour rather than an
   573	  absorbable transform. It is the only arm in the plan aimed at the 94% → 98% in-distribution gap.
   574	- **One mechanism note on D-NORM, so its result is not over-read:** on unit vectors the gradient of
   575	  ‖t−s‖ has unit magnitude per example, so a poorly-fit (under-covered) form receives *less* relative
   576	  gradient than under squared L2. Under a coverage thesis the sign of its effect is not obvious in
   577	  advance, which is why it is screened rather than adopted.
   578	- **Optimizer:** AdamW β=(0.9, 0.999), eps 1e-8, wd 0.01 on dim>1, clip 1.0. **Batch 32 examples**
   579	  [family E].
   580	- **Schedule:** 3 cycles of equal example count, each linear 1e-4→1e-5; 2,000 warmup steps in
   581	  cycle 1. Evaluation at every cycle end (annealed) and at cycle midpoints (curve watch only).
   582	- **Init [family C]:** bge-small (default) or the M9 candidate.
   583	- **Dose:** **200M examples** registered — LEAF's dose (6.7M texts × 30 epochs ≈ 201M; PLANNING §5),
   584	  three cycles of 66.7M. Tokens follow the mix: at 75/25, 150M × ~35 + 50M × ~230 ≈ **16.8B**; at
   585	  50/50 ≈ 26.5B. Query epochs ≈ 37 over 4.0M texts, document epochs ≈ 8 over the 6.15M pool. At the
   586	  **measured** bucket rates (718 query / 596 document examples/s, blended **683** at 75/25; PLANNING
   587	  §11, on the box) that is ≈ **81 GPU-hours**, and ≈ 139 h in M9's two-chunk collate — so
   588	  **length-bucketed single-chunk batching is part of the build, not an optimisation** (amendment A7),
   589	  which is why the mix is a share over a 4-step window (§Data). The day-one benchmark's rate replaces
   590	  both, and family E's batch-32 penalty is recorded in the lock as the build's GPU-hour line. **Extension:** let m_k be the COV macro (full precision, the locked formula and
   591	  evaluation hashes) at the end of cycle k. After every cycle k ≥ 3, one more cycle of 66.7M examples
   592	  (linear 1e-4→1e-5, as cycle 3) starts iff m_k − max(m₁, …, m_{k−1}) ≥ 0.003 **and** the extension
   593	  cycles already run are fewer than `max_extension_cycles`, an integer the lock fixes from the
   594	  approved dollars minus every mandatory line of **this file's §Compute table** at the measured rates
   595	  and the billed price (PLANNING §5–§6 are superseded evidence, Opus M12). Whole cycles only; a cycle whose projected cost plus billed spend to date would
   596	  exceed the ceiling does not start. The lock records m_k's formula, the evaluation hashes,
   597	  `max_extension_cycles`, and the spend source.
   598	- **Kill:** non-finite loss/grad; two consecutive scheduled evaluations more than 0.0056 below the
   599	  best evaluation *of their own kind* (midpoint against midpoints, cycle end against cycle ends), so
   600	  the rule can fire inside the build and not only at its end (Opus M5). **Plateau** is read
   601	  best-to-best on annealed checkpoints only; **the plateau rule fires when the improvement test
   602	  `m_k − max(m₁…m_{k−1}) ≥ 0.003` fails at a cycle end k ≥ 3**, independent of the cycle cap (Opus M6;
   603	  Codex 2026-09-04).
   604	- **Seeds:** one shipping seed for the build (seed 0); confirmation seeds at screen dose per §Screen.
   605	  **No full-dose replica runs** (decision 8 withdrawn, amendment A5); the replication band is the
   606	  selected recipe re-trained at screen dose under two further seeds, reported descriptively.
   607	
   608	## Screen — seven families, fifteen arms, thirteen contrasts, locked at M10.0-e (the registry is authoritative: C and L12 cut, A's STOP removed, 14 trained arms, 12 contrasts — §Amendment 2026-09-05)
   609	
   610	**Screen dose = 5M examples** (2.5% of the build; ≈ 420M tokens at 75/25; ≈ 2.0 GPU-hours per arm at
   611	the measured blended 683 examples/s), full 3-cycle schedule compressed to that dose, one seed, identical
   612	evaluation: **COV at every cycle end; DEV-6 once, at the final checkpoint** (its 5.2M-document
   613	hotpotqa and 6.17M-row heldout components cost ~13 GB of reads per pass — M9's practice).
   614	Throughput is recorded for every arm and decides nothing except family E (below).
   615	**Order (A6, amended after the Opus pass): F → A → G → B → E → C → D**, and family F carries a higher
   616	dose than the rest (§F row) because every later verdict — family A's stop rule included — is taken on
   617	its winner (A4 at 5M is the winner's 5M checkpoint from F's curve; F's ≈17 GPU-h are spent before A
   618	can stop the milestone, accepted). Student first, then the data thesis, then the two architecture/regime questions, then init and objective; every family after F
   619	runs on F's winner, so no verdict transfers to the build student by assumption.
   620	**Anchor** = the full M10 corpus (A4's data), mix 75/25, bge-small init, squared L2, bs 32, 1152-d
   621	feature. Screens run **on the box** (amendment A7).
   622	
   623	| family | arms | contrasts | rule and default |
   624	|---|---|---|---|
   625	| **A — data (the thesis)** | A1: M9 pool (463,314 queries) · A2: M9 pool + PAQ (factoid forms only — the volume control) · A3: A2 + the **harvested real** query-like text · A4: A3 + the **generated** forms (the full M10 corpus, = anchor). **A2, A3 and A4 are cut to the identical post-screen unique-text count** (the smallest of the three after decontamination, the larger two downsampled with seed 0) and all hashes are locked before any arm | **A3−A2** (forms from real text, at equal volume) · **A4−A3** (what generation adds over harvesting) · A4−A2 and A2−A1 descriptive | **Three registered outcomes on A3−A2** (the forms contrast, now carried by the real-text arm): corrected lower bound > MDE → coverage **resolved on the COV families**, build proceeds; point ≥ MDE and lower bound > 0 but ≤ MDE → **positive, not resolved**, build proceeds and the report says so; otherwise → **M10 stops before any build and returns to Dylan with all four rows**. **A4−A3 decides whether the generated half is in the build at all**: if it does not resolve, the build uses A3's corpus and the ≈1.0M generated queries are dropped from the build (they stay in the report as a measured null). A2−A1 is the volume effect; if it resolves, the build keeps volume as well as forms |
   626	| **F — student** | bge-small (34.5M with the head) · MiniLM-L6-v2 (23.9M) · **MiniLM-L12-v2 (33.4M with the head)** — the first two at **20M examples each, read as a curve at 5M / 10M / 20M**; L12 at 5M | 2: L6−bge-small at 20M · L12−(winner) at 20M **if L12 is extended**; the count is thirteen whether or not it runs | **Amended after the Fable review, made executable after the Codex pass.** The old rule ("bge-small only if it wins resolved") pre-decided MiniLM-L6 by construction, because M9's same contrast was −0.0026 *unresolved* and a 5M screen cannot resolve 0.003; that also silently skipped family C, since only bge-small has an M9 warm start. So F gets the dose its consequence deserves — it picks the build student and every later verdict — and a third arm, because arXiv 2306.11550's depth curve (1/2/4 layers → 86.1/92.5/96.2% retention) says depth buys retention while LEAF reached 97.7% on 6 layers, and L12 is 12 layers inside the cap. **Rule:** L12 is a 5M *elimination probe* — extended to 20M iff `m_L12(5M) ≥ max(m_L6(5M), m_bge(5M)) − 0.0056`, else eliminated and reported; among the arms with a 20M reading the winner is the highest COV macro **if its margin over the runner-up resolves under the contrast rule**; otherwise the cheapest to serve wins, and **that tie-break is labelled in the report as a product preference, not evidence** (Dylan 2026-09-01 called 33M "the upper bound of what I think is acceptable", so 33M is admissible). L12's three- and four-layer heads pass the parity check first. Cost ≈ 17 GPU-hours at the measured rate, on the box |
   627	| **G — output width and head form** | feature = last layer only (384, M9's head) · three layers (1152, = anchor) · four layers (1536, §Recipe) · **G-MLP**: the anchor's linear head plus a per-token rank-192 GELU correction, `W_lin·x + W₂·GELU(W₁·x+b₁)`, W₁: 1152→192 (B3, residual form after the Codex pass; replaces the 768 arm, whose step the head-width probe already shows — the paper reports 384 vs 1152 vs 1536, not a trained curve) | 1152−384 · 1536−1152 · **MLP−1152** | resolved winner; **default 1152** (PLANNING §9–9b; the screen, not the probe, decides). The 384 arm is the paper's evidence for the M9 diagnosis. **G-MLP** keeps the linear path's full rank and adds the one nonlinearity the cap allows (34.96M for bge-small); fastembed serves it exactly because it precedes the mean pool (`results/m10_head_mlp_parity_box.json`, min-cos 0.99999989, zero custom ops). **Warm start, exact and deterministic:** `W_lin` = the anchor's ridge head; `W₁,b₁` = the top-192 principal directions of the frozen backbone's *per-token* 1152-d states on the fit set, centred (`b₁ = −W₁μ`; sign of each direction fixed so its largest-magnitude component is positive); `W₂,b₂` = ridge from the pooled `mean_t GELU(W₁x_t+b₁)` to the residual `t − W_lin·x̄` — exact for the training form because pooling commutes with the linear maps, so G-MLP starts *at* the anchor's fitted head plus a fitted correction; all three solves share one fit sample — `m9/registry.json` `warm_start`: n_fit 60,000, seed 21, λ reselected on the M10 sample by the registry's locked grid on a training-only holdout (ties to the larger λ) — and the per-token PCA is a streamed 1152×1152 Gram matrix |
   628	| **B — mix** | 100/0 · 50/50 query/document (75/25 = anchor), **matched query presentations** (3.75M query examples in every arm; document examples 0 / 1.25M / 3.75M on top; totals 3.75M / 5M / 7.5M, 4-step window patterns 4Q · 3Q+1D · 2Q+2D; the document cost in tokens and GPU-hours is reported) | 100/0−75/25 · 50/50−75/25 | resolved winner; default 75/25 |
   629	| **E — batch** | 32 · 128 at equal examples and identical schedule | 1 | resolved winner; default 32 (LEAF). **Amendment A7: E is the one family whose throughput is read** — bs128 measured 1,517 examples/s blended against bs32's 683 (`results/m10_rate_bench_box.json`), so a bs32 win must also be worth its 2.2× build cost; the lock records both the quality contrast and the GPU-hour delta, and a bs32 win that does not resolve reverts to bs128 |
   630	| **C — init** | bge-small (or MiniLM, per F) · the M9 candidate | 1 | resolved winner; default the off-the-shelf backbone. Available only if F selects bge-small: M9's candidate is a bge-small student, so a MiniLM build has no warm start and C is skipped and reported as skipped. The closed-form ridge head warm start (M9's `m9s1c`, +0.0272) is retained in every arm regardless |
   631	| **D — objective** | anchor (squared L2) · **D-NORM** (LEAF's ‖e‖₂) · **D-COV** (document-covariance-weighted regression, §Recipe) | D-NORM−anchor · D-COV−anchor | resolved winner with the larger margin; default squared L2. The ranking-aware class is cut (amendment A1, §Recipe); **D-COV is added from the Fable review as the one cheap lever aimed at the in-distribution ceiling** (PLANNING §13 idea 8) |
   632	
   633	A2 exists only as a control; the build never uses more than 1.0M PAQ. **Equal examples** holds for
   634	every family except B, which is matched on query presentations by design. **Definitions:** a
   635	decision's *margin* is the COV macro difference between the winner's and the default's final
   636	checkpoints in the original screen; an arm's *seed range* is max minus min of its COV macro over
   637	its three seeds.
   638	
   639	**Rule, per contrast (families B–G):** the difference in COV macro (family-weighted, §Surfaces)
   640	between the two arms' final checkpoints; paired stratified bootstrap over queries within component,
   641	**B = 200,000**, seed 0, empirical quantile (`inverted_cdf`; at 0.025/13 that is the 384th order
   642	statistic — at B = 20,000 it would have been the 38th, Opus M9); a contrast **resolves** when the point estimate ≥ the **MDE 0.0056** **and** the
   643	one-sided lower bound at the **0.025/13 quantile** (Bonferroni over the thirteen contrasts — F 2 ·
   644	A 2 · G 3 · B 2 · E 1 · C 1 · D 2 — fixed whether or not F's second comparison runs; Codex
   645	2026-09-04 corrected a count of fourteen) is > 0, and the sign is stable across the last two
   646	cycle-end checkpoints. **Both constants are fixed here** — the Codex pass of 2026-09-04 struck
   647	amendment A4's `MDE = max(0.0056, distance)` and its two-step remedy (admit LEDGER, then α = 0.05)
   648	as an α that adapts to an observed width. The resolution number (§Surfaces) is *reported* as the
   649	power disclosure; an underpowered contrast is reported as unresolved and reverts to default, and
   650	the report names the contrasts the surface could not have resolved. **Family A's contrasts A3−A2 and A4−A3 are exempt from the
   651	generic rule** and use the three-outcome rule in the table (resolved requires the corrected lower
   652	bound > MDE); A4−A2 and A2−A1 are descriptive. **Confirmation:** for every decision whose
   653	non-default option won, both the winner and the default are re-trained with two more seeds at
   654	screen dose — **at most two such decisions, largest margins first** (amendment A5; it was four);
   655	the rest revert to default. The decision stands only if the winner's margin exceeds the largest
   656	seed range observed in either arm. Worst-case confirmation cost is now 45M examples ≈ 17
   657	GPU-hours, plus the synthesized selected-recipe arm (5M) and the replication seed pair (10M);
   658	PLANNING §5 has the arithmetic. Every screen verdict is artifact-specific at screen dose; never
   659	"resolved" in the report's sense.
   660	
   661	## Surfaces
   662	
   663	- **COV** — the primary selection surface: **qrel-bearing** retrieval components with an
   664	  affirmative licence at the dataset's primary source (M7's eval-use standard), decontaminated,
   665	  and **read by no M10 decision**, admitted at M10.0-d, **weighted equally per family**
   666	  (slices within a family averaged first). Candidates, draft records in `m10/COV_CANDIDATES.md`
   667	  (primary-source licences checked 2026-09-01): **consumer-health** MTEB MedicalQARetrieval (CC BY
   668	  4.0 at MedQuAD; NIH sources); **BRIGHT**, one family, six slices (biology, earth-science,
   669	  economics, psychology, robotics, sustainable-living; CC BY 4.0 on the benchmark at its primary
   670	  source — the same dataset-level standard that admitted CQADupStack and the six, whose documents
   671	  are also third-party text; the document-rights caveat is disclosed and the data is evaluation-only,
   672	  never redistributed); **legal** MTEB LegalBenchCorporateLobbying (CC BY 4.0; ConsumerContractsQA
   673	  refused, CC BY-NC); **finance** LEDGER (CC BY 4.0 annual-report QA) once its structure and a
   674	  100K-chunk cap are verified. Climate-FEVER refused (no licence at its primary source, as in M7).
   675	
   676	  **Admission reopened 2026-09-04 (Dylan: non-commercial licences are admissible for validation, not
   677	  training — `research/m7-data-licensing.md` §Rule change 2026-09-04).** A set refused *for its licence
   678	  alone* is re-admissible, and the resuming session must re-run admission before the family floor is
   679	  judged: **MTEB ConsumerContractsQA (CC BY-NC) is admissible**, restoring a second legal component;
   680	  the family count stays three without LEDGER (one legal family). Any other CC BY-NC
   681	  or research-only eval set is now in scope. **Still refused:** Climate-FEVER (no affirmative grant —
   682	  a different class), and anything excluded for contamination. MS MARCO itself is admissible but is a
   683	  **poor COV member** — every comparator trains on it and neither of ours does, so it is biased
   684	  against us; prefer it for the within-system read under FORMS-12.
   685	  **Family IDs: `consumer-health` (MedicalQA), `BRIGHT`, `legal` (CorporateLobbying and
   686	  ConsumerContractsQA, one family) and `finance` iff LEDGER admits; CUREv1 and `arxiv-title` are
   687	  diagnostics and never count. At least three IDs must admit** or M10 returns to Dylan — without LEDGER
   688	  the count is exactly three (Codex 2026-09-04 corrected an earlier "four"); the report names it. The two CQADupStack components are **DEV**, reported beside every COV
   689	  read, never in the macro.
   690	
   691	  **Scientific diagnostic surface (B4; a COV family until the Codex pass, an actioned secondary until the Opus pass).** Every clean-4
   692	  set is scientific or biomedical and no licensed published set covers those forms outside S2ORC /
   693	  PubMed / CORD-19, so as registered the screen could not see the headline forms. **`arxiv-title`**:
   694	  2,000 held-out arXiv titles → their own abstract among 100K arXiv abstracts (CC0 metadata; ≈8 min
   695	  of encode), one relevant per query, nDCG@10, teacher score reported as the denominator. **Not in
   696	  the COV macro** — it shares form and source with arm A3's harvest, so as a family it would reward A3
   697	  by construction and its 2,000 low-variance queries would dominate the macro's power. It is a
   698	  **descriptive diagnostic with no action** (Opus M11: title→own abstract is near-lexical known-item
   699	  retrieval, so a null there could be a floor effect, and it is not scidocs' title→cited-abstract
   700	  relation); A3−A2 on the COV macro carries every data decision. Reported per arm beside FORMS-12 as
   701	  the one real-text scientific read. **Draw, registered (Codex B8):** the Kaggle arXiv metadata artifact (`Cornell-University/arxiv`,
   702	  the version downloaded on day 1; its file sha256 recorded in `m10/LEDGER.md` §0b before the draw);
   703	  ID universe = every record's `id` with the version suffix stripped, deduplicated, sorted
   704	  lexicographically; `numpy.random.default_rng(0).choice(N, 100_000, replace=False)` in that order;
   705	  the first 2,000 drawn are the queries (title → own abstract), the other 98,000 distractors; every
   706	  version of a drawn paper is excluded from every training role; the harvest is fingerprint-screened
   707	  against the drawn titles and abstracts; the set joins the protected index before any extraction.
   708	  It is drawn at the start of the harvest step, once the artifact exists.
   709	  ClinicalTrials.gov is not used (terms clause unread). COV still has no argument retrieval
   710	  (six-set only; FORMS-12 descriptive). **Decision 12 (adopted)**: CUREv1 as a validation-only diagnostic.
   711	  **Resolution number (M10.0-d, before the lock) — a power disclosure, nothing more** (A4's sizing
   712	  struck by the Codex pass). With the contrast rule's own bootstrap, measure the distance between the
   713	  point estimate and the one-sided 0.025/13 lower bound for the COV-macro difference between two
   714	  models that are candidates in no M10 family — **e5-small-v2 and gte-small** (`results/FINAL_MATRIX.md`
   715	  rows, fresh COV encodes) — scored symmetrically on the admitted surface. Only the distance is
   716	  recorded, never which led. It is the first disclosed COV read (`m10/LEDGER.md` §4) and is
   717	  published beside every contrast, so a reader can tell an unresolved verdict from an invisible one.
   718	  With MedicalQA 2,048, BRIGHT 6 × ~100, CorporateLobbying 340, ConsumerContractsQA 396 and paired
   719	  per-query SDs of 0.07–0.27, the expected distance is 0.009–0.0135 — above the MDE, so **most B–G
   720	  contrasts are expected to be unresolved at screen dose and to revert to default; that is disclosed
   721	  in advance, not repaired by loosening α.** LEDGER (118,048 questions) is admitted if its structure
   722	  verifies at M10.0-d, an ordinary admission decided before any read, because it is the one
   723	  candidate large enough to move the surface's power. The distance is measured between *unrelated*
   724	  models, so it over-estimates the width of a same-init contrast; recorded, not corrected.
   725	- **DEV-6** (incl. the two CQADupStack components) secondary, reported beside every COV read.
   726	  SCREEN-3 is retired.
   727	- **FORMS-12**: 12 × 500 held-out generated or harvested queries, overlap@10 between student and
   728	  teacher rankings over a **1M-document evaluation sample of the already-encoded pool** (seed 0,
   729	  fixed at M10.1) — amendment A1 deleted the *mined candidate bank*, not the evaluation sample, and
   730	  this needs no mining. Per form. **Descriptive only** — teacher agreement on generated
   731	  queries is a coverage diagnostic, not retrieval quality.
   732	  **Now also permitted (2026-09-04, optional, descriptive):** the same overlap@10 diagnostic on a
   733	  sample of **real MS MARCO dev queries**, as an external check that the 12 forms cover the natural
   734	  query distribution. Validation only: MS MARCO text may **never** seed generation or enter the pool,
   735	  and no MS MARCO-derived cache may be written under `work/train/sources/` (`m7src/mix.py:22-25`).
   736	  **The stella-space distribution-overlap gate of amendment A8 (§Data) is the primary use of this
   737	  permission** and runs on the manifest before any arm; this overlap@10 row is the post-hoc version
   738	  of the same idea and stays optional. Neither is a substitute for COV: both are teacher-agreement
   739	  or distributional diagnostics, not retrieval quality.
   740	- **LoTTE-clean** (7 slices, macro over slices; its corpora — ~2.8M passages — are encoded with
   741	  stella once, ≈ 1.3 GPU-hours at the assumed 600 docs/s, budgeted in PLANNING §6): **read #1**
   742	  after the recipe lock.
   743	  Before LoTTE opens, the **selected recipe is trained once as a single synthesized arm at screen
   744	  dose** (its checkpoint hash committed); read #1 scores that checkpoint and the anchor's in one atomic batch.
   745	  Veto rule (M9 §7, unchanged): the selection is vetoed if the selected recipe's 7-slice macro is
   746	  worse than the anchor's by more than 0.004 AND the one-sided 97.5% paired-bootstrap upper bound
   747	  (B = 10,000, seed 903, paired within slice) on (selected − anchor) is below −0.004; a veto means
   748	  **the anchor recipe builds**. No other action may follow from read #1. **Read #2** pre-freeze,
   749	  audit only. No third read.
   750	- **Dev→six calibration**: M9's close-out rows (available only after M10.2) and M7's; forecasting
   751	  only, never gating.
   752	- Dev reuse is counted (`m8src/dev_reuse_m8.py`) and published.
   753	
   754	## Final run
   755	
   756	`m9/FINAL_LOCK.md` re-registered verbatim for M10 in the M10.2 lock with new BOUND-AT-FREEZE hashes
   757	and an `m10-six-spent` tag — bridge as phase 1, the C-conjuncts with the empirical bootstrap bound
   758	and the sign-flip conjunct, FEVER labelled — **with four changes (Opus: "two" understated it). (1) The
   759	reserved conditional is `if any C-conjunct passes then execute`**, so an aim claim never stands
   760	without its descriptive reserved rows. **(2) Amendment A3: four conjuncts, not two** — `C1a`/`C2a`
   761	on avg-6 and `C1b`/`C2b` on clean-4 (bars 0.5042 / 0.5155 / 0.5046 / 0.5233), under
   762	**fixed-sequence gatekeeping in the order C1b → C1a → C2a → C2b at the full one-sided 0.025 each,
   763	stopping at the first non-rejection** (§Goal has the procedure, the planning proxies and the
   764	`trec-covid` n=50 disclosure). **(3) The decision field is the empirical one-sided 0.025 quantile
   765	(`lower_q025_raw`, the 250th order statistic at B = 10,000), not M9's Holm-2 0.0125, and (4) the
   766	sign-flip conjunct is tested at 0.025 inside the same sequence** — the Codex pass of 2026-09-04
   767	found the two quantiles named in different files. M9's claim-table row "C1 fail, C2 pass → aim
   768	permitted" is deleted by (2). The sequence is fixed here, before any six-set
   769	output exists.
   770	clean-4 is M14's registered headline partition for both halves of the pair, so it carries a
   771	pre-registered pass/fail rather than arriving as a descriptive row after the fact. Zero alpha on the reserved batch is unchanged. `m9src/final9.py`'s
   772	scoring path is written and reviewed before M9's close-out and reused for scoring; **its `decide()` is
   773	hard-coded to M9's two conjuncts and is NOT reused — an M10 `decide()` for four conjuncts under
   774	gatekeeping is written, unit-tested and reviewed before the recipe lock** (Codex 2026-09-04).
   775	
   776	## Compute and costs — re-priced 2026-09-04 on measured rates (amendment A7)
   777	
   778	**Split execution.** The 2026-09-01 ruling stands where it bites — no dose or screen size is set by
   779	the box's wall-clock — but the box is an execution target again for everything that is not
   780	generation, because the rate it was withdrawn on was wrong:
   781	
   782	| configuration, M10 recipe shape, **measured on the RTX 3080 2026-09-04** (PLANNING §11) | examples/s | 200M examples |
   783	|---|---|---|
   784	| batch 32, 75/25, M9's two-chunk collate | 400 | 139 h |
   785	| batch 32, single-chunk buckets (718 query / 596 document), blended 75/25 | **683** | **81 h** |
   786	| batch 128, single-chunk buckets (2,311 / 748), blended 75/25 | 1,517 | 37 h |
   787	
   788	The plan's imported LEAF planning rate was 560 examples/s on a rented A100; the box meets or beats
   789	it, because at batch 32 with ~35-token queries the step is launch-bound, not FLOP-bound, so a
   790	bigger card buys ~1.5–2× rather than 10×. Caveat, disclosed: these are fixed-shape random-token
   791	microbenchmarks with no data loading and no evaluation, so the real trainer is slower; they bound
   792	the *hardware*, not the pipeline. **Consequences, all registered here:** length-bucketed
   793	single-chunk batching is part of the build (§Recipe); the screens, confirmations, COV admission and
   794	encodes, and M10.0-c run **on the box**, which already holds ~200 GB of M9 caches the plan had
   795	budgeted 12 GPU-hours and a day of network to re-derive; the day-one benchmark still re-derives
   796	PLANNING §6 before anything scales.
   797	
   798	**What the money buys.** Generation runs on the box (decision 14, Qwen's official AWQ release), so
   799	the build is the cloud's job — rented for wall-clock, not because the box cannot do it (the box would take
   800	≈ 3–6 days at the measured rates). One rented **A100 80 GB** (H100 if its cost per example measures
   801	lower on the smoke), ≥ 500 GB persistent disk that survives stopping the instance, SSH, and a GitHub
   802	deploy key so the headless commit-and-push contract holds. Provider is Dylan's choice; $1.5–2.5/h
   803	assumed, unverified Sept 2026. The instance is stopped between stages.
   804	
   805	| line | GPU-hours | $ at 1.5–2.5/h |
   806	|---|---|---|
   807	| generation, ≈1.0M queries — **on the box under decision 14** | 0 | 0 |
   808	| build, 200M examples at the measured rates (bs128 → bs32) | 37–81 | 55–205 |
   809	| cloud-side encodes the build needs + export, parity, final run | 6 | 9–15 |
   810	| day-one rate benchmark | 1 | 2–3 |
   811	| persistent disk, egress | — | ≈ 25 |
   812	| **cloud total (screens and generation on the box)** | **44–88** | **≈ $90–245** |
   813	| all-cloud variant (screens and confirmations rented too) | 130–190 | ≈ $220–500 |
   814	| optional: extension cycle, 66.7M examples, each | 13–25 | 20–63 |
   815	
   816	Hard ceiling **$1,000**, unchanged and now far from binding. **Allocation order at the lock:** every
   817	mandatory line first at the measured rates and the billed price, then whole extension cycles from the
   818	remainder (the second build seed is withdrawn, amendment A5). A cycle whose projected cost plus
   819	billed spend to date would exceed the ceiling does not start.
   820	
   821	**Day-one rate benchmark (≈ 1 GPU-hour, before any scale-up):** stella docs/s on 20K pool documents;
   822	training examples/s at batch 32 for 2,000 steps on the 75/25 anchor mix, the 50/50 mix and MiniLM-L6;
   823	generation requests/s per form on the 200-query smoke, end to end (prefill, decode, retries, JSON
   824	failures); the provider's billed price. Wall-clock ≈ 1.5–2 weeks; screen arms are independent and the
   825	box runs them while the cloud generates.
   826	
   827	On the instance, `results/perquery.json` (tracked) is sha-verified against `6b18e3dd…` before any
   828	scoring, and M9's parity sample is regenerated from `m9/registry.json` (tracked; seed and sha256
   829	pinned) — `m9src/port.py` refuses a sample that does not hash to the lock, in which case the 512
   830	texts are transferred from the box. Only what training needs travels to the instance (tokenized
   831	corpora and teacher targets, or re-derived there from the repo and HF); nothing that only the
   832	evaluation stages need. Reserved-set encodes run only if the reserved conditional fires.
   833	
   834	M9's cost protocol (`instructions-m9.md` §Costs) unchanged; the frontier is reported per index
   835	configuration, naming the one measured (`m9/RESULTS.md` rounds 1–4). TurboQuant 4-bit (Qdrant 1.19)
   836	joins the M13 all-in quantization comparison.
   837	
   838	## Unimpeachable by competitors — what the report must carry
   839	
   840	1. Comparators byte-verified against the official artifacts, scored in the same exact-search
   841	   harness, frozen as per-query vectors before any M10 number existed; the bridge check proves
   842	   the harness unchanged.
   843	2. Pre-registration with pushed commits as the external witness; one six-set transaction per
   844	   milestone, each disclosed (M7, M9 close-out, M10); the reserved four's single access; dev-reuse
   845	   count and the selection-surface table published.
   846	3. Contamination handled three ways: stella's disclosed overlap (ArguAna, FiQA) at every headline;
   847	   NDO-4 rows; reserved NDO-3 as the surface no decision touched. **LoTTE read #1 is decision-bearing
   848	   (the veto), so LoTTE is not untouched, and read #2 is an audit, not independent confirmation.**
   849	4. Training data affirmatively licensed, no MS MARCO, attribution recorded; synthetic queries from
   850	   an open-weights generator with pinned revision, per-query provenance and removal counts.
   851	5. Compute disclosed in examples, tokens and GPU-hours beside LEAF's ~100 A100-hours.
   852	6. System-level framing only (§Goal): no claim isolates nano from its document tower.
   853	7. Reproducibility: code, corpus manifests with hashes, model revisions, seeds, statistics code
   854	   with tests, and the full screen table including losing arms and confirmation seeds.
   855	8. **No general retrieval-quality language.** The six are development-informed by construction (the
   856	   data taxonomy targets their forms; COV selects; LoTTE vetoes); the reserved four are the only
   857	   untouched surface and are descriptive. Every conclusion is stated on the named datasets, and the
   858	   released artifact is labelled **"selected on development-informed datasets; the reserved four are
   859	   descriptive"** (Codex 2026-09-04, twice).
   860	
   861	## Deliverables
   862	
   863	Frozen candidate + `m10/FREEZE.json` (`assert_releasable` with a proper run record — it walks
   864	`work/runs/<id>.json` for `cfg.sources` and `cfg.init` across the whole lineage and fails CLOSED on
   865	a missing ancestor, which is why M9's freeze was refused; the record must be **derived from the
   866	run's own hash-bound `manifest.json`, never asserted by hand**, because a post-hoc record that
   867	happens to satisfy a licence guard is precisely the artifact that must not be fabricated; see
   868	`m9/BUILD_LOG.md`), the frontier
   869	update, the M10 section of the report artifact, decisions logged in CLAUDE.md, handoff to M13.
   870	
   871	## Out of scope (reopening conditions in PLANNING §7)
   872	
   873	Document-side co-adaptation (inside M10 it breaks the pair; a tower co-trained against both query paths at once keeps it and is recommended 2026-09-01 as the next-milestone candidate (that slot was M12 then; **M16** after the 2026-09-04 renumbering), Dylan's call, PLANNING §7) · any student above 35M
   874	(hard cap, Dylan 2026-09-01) · teacher change (stella-1.5B measured worse; Qwen3-0.6B never
   875	screened and not the pair) · a **post-pooling** nonlinear head (no fastembed path; per-token heads
   876	ship, §Recipe) · a cosine-space phase-1 loss (closed by algebra, `m10/EXPLORED.md`) · a
   877	ranking-aware phase-2 loss (cut by amendment A1; reopening condition in §Recipe) · MS MARCO **in any training role** (unchanged; **validation/diagnostic use was
   878	permitted 2026-09-04**, see the admission note in §COV) · FineWeb in any role (decisions 3 and 5) · any
   879	change to zero.

exec
/bin/bash -lc 'nl -ba CLAUDE.md' in /home/dylan/asymetric-dual-encoders
 succeeded in 0ms:
     1	# Asymmetric Dual Encoders — Viability Scoping
     2	
     3	## North star
     4	
     5	Decide, with defensible numbers, whether asymmetric dual encoders (big document encoder in the cloud, near-zero-compute query encoder on the edge) beat the obvious alternative: running a small 20–100M conventional embedding model on the edge. The research question: **how much retrieval quality do you retain as query-side computation approaches zero?**
     6	
     7	Architecture: the server indexes documents once with a large frozen encoder (**stella_en_400M_v5**, 1024d, since M7); the edge client holds the document index plus a query path. Two query paths exist: **zero** (M7, a token→vector lookup table, no transformer) and **nano** (M9/M10, a ≤35M distilled transformer). Both serve the SAME index — the product is the pair, two points on a quality-vs-query-cost frontier.
     8	
     9	## Deliverable
    10	
    11	A decision report (Artifact) with a quality-vs-query-side-cost frontier: nDCG@10 on the six named datasets for each point, plus edge costs (query asset, document index, hydration, CPU latency), a Qdrant Edge prototype of the pair, and the released models. Every headline claim is paired on frozen comparator vectors with pre-registered statistics; the miss-is-publishable framing applies to every milestone.
    12	
    13	## Constraints
    14	
    15	- Mac: Apple M5 Pro, 24 GB RAM, MPS — probes and CPU cost rows only. The Windows/RTX 3080 box ran M7–M9 and holds their gitignored `work/` artifacts; **M10's build runs on a rented cloud GPU under an approved budget (Dylan, 2026-09-01; amended by A7 2026-09-04 — the box runs the screens and the generation). Amended again 2026-09-05: the box is PREPARATION for the cloud run, not a measurement target — an arm that does not fit here moves to the cloud rather than being re-shaped (`E-bs128` is the first)**. Headless either way: Dylan follows on GitHub.
    16	- Quality numbers come from exact search so ANN recall is not a confound; Qdrant Edge is for architecture and latency.
    17	- Dylan is hands-off; decisions get made here and logged. The evaluation protocol (partitions, decontamination, frozen comparators, single final run, pre-registered statistics) may change only BEFORE the numbers it affects are observed, never retroactively.
    18	
    19	## Stage plan
    20	
    21	- [x] M0–M6 (2026-08-24/25): environment, research (`research/landscape.md`, `literature.md`, `methodology.md`, `sparse-inference-free.md`, `lightretriever.md`), the 21-system × 6-dataset matrix (`results/FINAL_MATRIX.md`, costs, significance), the two-collection Edge prototype, and the M6 report artifact (https://claude.ai/code/artifact/db771dd1-2d59-4c34-9d6e-70dd4d337d16). Headline numbers, rerun outcomes, the M2 Codex gate and the M1/M2 findings logs are archived verbatim in `research/m1-m6-findings.md`.
    22	- [x] M7 (closed 2026-08-29): **zero**, the int8 lookup table distilled from stella. Final run avg-6 **0.4339** vs `LR-dense-pertask` 0.4583 (CI-resolved miss); fused with BM25 0.4911 (convex0, a statistical tie with OpenSearch 0.4868 — the registered C3 comparison, unchanged). **M12 re-measured the fused row under a Qdrant-runnable operator: `Fusion.DBSF` at prefetch 100 gives 0.4887 on the six and 0.4912 on clean-4 vs convex0's 0.4911/0.4866 — a TIE on both (inside the ~0.005 band, no CI), so the case is deployability at no measured cost. Recorded as a post-M7 PRODUCT-POLICY OVERRIDE of the `m7/LEDGER.md:691` release freeze, on Dylan's implementability ruling — convex0 stays the M7 operator of record and C1/C2/C3 are untouched.** Frozen and verified releasable (`m7/FREEZE.json`); **published 2026-09-03, PUBLIC, as `constella-zero`** — https://huggingface.co/DylanCouzon/constella-zero (renamed from `zero-query-encoder-v1`; old URL redirects), see `m11/STATUS.md`. Mandate `instructions-m7.md`, detail `m7/STATUS.md`, transferable lessons `m7/FINDINGS.md`. Teacher decision: stella_en_400M_v5, chosen on the distilled TABLE, not the tower (see Key decisions).
    23	- [x] M8 (closed 2026-08-30 as a measurement): twelve probes, no lever moved the table more than 0.005; no confirmatory access spent. `m8/FINDINGS.md`, `m8/EXPLORED.md`.
    24	- [x] M9 (closed 2026-09-01 as a measurement, six-set close-out pending): **nano**, bge-small distilled into stella's query space by L2 regression. Build stopped by the plateau rule at 3.74B tokens: SCREEN-3 0.5606 = **82.2% retention**, but **93.8% on NQ against 50–71% on the two CQADupStack components** — a coverage failure, compounded by a 384-wide linear head that L2 regression cannot push past ~90–93% once queries are diverse (`m10/PLANNING.md` §9); not a parameter-count failure. Aim (leaf-ir-asym 0.5155) and release bar (bge-small 0.5042) both need ≥88% on avg-6; projected miss. **Read `m9/FINDINGS.md`.** Mandate `instructions-m9.md`; recipe `m9/M92_LOCK.md`; runs `m9/RESULTS.md`; closed avenues `m9/EXPLORED.md`. The registered six-set transaction (`m9/FINAL_LOCK.md`) is executed on the frozen candidate as M9's close-out measurement only after M10's recipe lock is pushed (so it cannot inform an M10 decision), six-only (its reserved conditional struck by a ratified amendment), and once Dylan ratifies it; LoTTE stays unread for M10.
    25	- [ ] M10 (**budget VALIDATED and plan re-cut 2026-09-04**; resume from `m10/STATUS.md`. Retry of nano, planned 2026-09-01): same pair, same bars, same teacher, new recipe built around the coverage finding — a wider linear output from pooling three layers, LEAF's small-batch cyclic schedule, warm start from the M9 candidate, and query-form breadth from **≈1.25M harvested real titles/headings/claim sentences plus ≈1.0M generated queries** for the seven forms no corpus contains (FineWeb ruled out of M10). **The 2026-09-04 efficiency review (amendments A1–A8, `instructions-m10.md` §Amendment 2026-09-04) cut the ranking-aware phase-2 class to one arm — LEAF's own ‖e‖₂ loss — deleting the candidate bank, the mining pass and the τ rule; cut generation 3.0M → 1.0M on three published saturation curves; withdrew the full-dose second seed; capped confirmations at two decisions; and re-registered C1/C2 on clean-4 as well as avg-6 (bars 0.5046 / 0.5233) so M14's headline partition carries a pre-registered pass/fail.** Compute re-priced on measured rates: the M10 recipe runs at **910 examples/s blended on the RTX 3080** with `torch.compile` on fixed length buckets, measured on REAL tokenized corpora with zero alloc retries (`results/m10_rate_bench_real_box.json`; the earlier 683 was the uncompiled path, 560 an imported A100 planning figure) — confirmed in flight by the M10.0-e calibration's 962 ex/s, falling to ~850 under CPU contention, so **the box runs the screens and the cloud buys generation and the build — ≈$90–245 against a $1,000 ceiling**, and the 2026-09-01 "not on a 3080" ruling now binds only the dose, not the hardware. **Reviewed four ways on 2026-09-04 (Fable ×2, Codex, Opus): amendments A1–A8 and B1–B6, every finding actioned; decisions 11 and 12 both RULED 2026-09-04 (12 adopted but never EXECUTED — see the whole-plan review); `research/m10-feasibility-review-2026-09-04.md` has the feasibility verdict.** Mandate `instructions-m10.md`; evidence `m10/PLANNING.md` (§11 rates, §12 the synthetic-data question); status `m10/STATUS.md`.
    26	- [x] M11 (**closed 2026-09-03**): the zero half of the pair, shipped. **`constella-zero`** — https://huggingface.co/DylanCouzon/constella-zero — and the frozen document tower **`stella-en-400M-v5-doc-onnx`** — https://huggingface.co/DylanCouzon/stella-en-400M-v5-doc-onnx — both public, ONNX, byte-verified, served by FastEmbed as built-in models on branch `Dylancouzon/fastembed@add-constella-models`. Serving parity vs the numpy reference **4.470e-08** on 1,024 dev queries. Mandate `instructions-m11.md` (Amendments A and B), detail `m11/STATUS.md`, reusable ONNX-port checklist `m11/CODEMAP.md`. Everything that depended on M10 moved to M13 (was M12 until 2026-09-04).
    27	- [x] M12 (**closed 2026-09-04**; local box, no training): **fusion operator audit.** M7's published fused row used `convex0 w=0.8` at prefetch 1000 — an operator **Qdrant does not implement**. Audited every shipping alternative to Qdrant parity (`m12src/qfusion.py`, vendored reference, parity + degenerate tests). **Dev, depth 1000: NO MATCH** — DBSF −0.0146, RRF over `k` −0.0192, weighted RRF held-out −0.0129 [−0.0160, −0.0098]; correcting M7's unfair comparison (unfitted RRF, `k` grid monotone to its own boundary, never reaching Qdrant's default `k=2`) recovered only +0.0097 of 0.0223. The gap is real: convex0 beats **both** endpoints, so it exploits score magnitudes that rank-based operators discard. **But the ranking inverts at realistic depth** — DBSF beats convex0 at prefetch 10 and ties at 50 (degenerate branch fires 0.00%, not an artifact). On Dylan's ruling (*"convex needs to go if it can't be done in Qdrant"*) the six were re-scored: **`Fusion.DBSF` @ prefetch 100 → 0.4887 all-6 / 0.4912 clean-4** vs convex0's 0.4911 / 0.4866 — **a TIE both ways** (inside the ~0.005 band, no CI), so the case is deployability at no measured cost, not superiority. **Recorded as a post-M7 PRODUCT-POLICY OVERRIDE of the `m7/LEDGER.md:691` release freeze** — convex0 stays the M7 operator of record, C1/C2/C3 untouched. Card/README/`m11/STATUS.md` now lead with the DBSF recipe; **the live HuggingFace card push is Dylan's, not done here.** A CLOUD finding — edge fusion is unbuilt and uncosted, a note not a metric. `instructions-m12.md`, `m12/FINDINGS.md`, `m12/LEDGER.md`, cut avenues `m12/EXPLORED.md`.
    28	- [ ] M13 (**blocked on M10**, was M12 until 2026-09-04): the nano half of the pair — release `constella-nano`, its ONNX port and its FastEmbed registration. `instructions-m13.md`. The whitepaper left for **M14** on 2026-09-04. **The upstream FastEmbed PR is an M13 deliverable and waits for nano** (Dylan, 2026-09-04): one clean PR adding all three model entries, branched fresh from upstream `main`, not the throwaway `add-constella-models` branch.
    29	- [ ] M14 (**created 2026-09-04**, split out of M13): **the paper.** Runs after nano; runs anyway on the zero-only frontier if M10/M13 never do. **One paper is the default**, decided at M14's start. The deliverable is the frontier; the FINDING is that teacher quality does not predict distilled-table quality — Spearman ≈ 0 over eleven teachers, mxbai matching bge-base's tower (0.4433 vs 0.4484) with a table 0.057 worse — measured today, needing no nano. Carries the comparator table kept off the model cards. **Benchmark composition is decided HERE, once, for both zero and nano (Dylan, 2026-09-04) — but the RULE was registered 2026-09-04 before nano's numbers exist: headline = the pre-registered clean-4, all six reported beside it, contamination robustness as its own row, no re-picking the six, MS MARCO validation-only if built.** M12 measured why that is cheap: clean-4 costs the FUSED system 0.0045 and the dense table 0.0241. `instructions-m14.md`.
    30	- [ ] M15 (noted, not scoped — Dylan, 2026-08-30): an IMAGE model. Edge workloads are largely vision, so the asymmetric premise has a bigger market there. Open: whether the query side is a lookup table at all, image→image vs text→image, which frozen document tower. Do not inherit the text architecture's assumptions. `instructions-m15.md`.
    31	- [ ] M16 (**parked, long-term, may never run** — Dylan, 2026-09-03): a better `zero` BEYOND the hybrid objective (that is M12). Idea register only, no compute committed: `instructions-m16.md`. The ranked lead is document-side co-adaptation (M8's `E14-LORA`, authorised and unrun), which reopens the frozen-tower premise and costs the drop-in-to-stella's-index property. The cheap lead is cosine-space distillation, priced at ~+0.035 by a pyNIFE retention comparison on fiqa (they retain 73.4% of a weaker teacher against our 67.1%).
    32	
    33	## Standing directive: push for the best model, not a model (Dylan, 2026-08-26)
    34	
    35	Added after a session declared a target "unreachable" and was wrong within the hour. **We do not
    36	want A model. We want the best one obtainable under the constraints.** Being first to say "this
    37	projects to a negative result" is not rigour — rigour is exhausting the angles and *then* reporting
    38	honestly. Honest pessimism arrived at lazily is still lazy.
    39	
    40	**Before writing that any bar is unreachable, all of these must be true and shown:**
    41	
    42	1. **The arithmetic has been redone with the best available component**, not the current one. The
    43	   2026-08-26 case: "Tier 1 unreachable" assumed the current teacher. Swapping in a stronger
    44	   permissive teacher (stella_en_400M_v5, MTEB-Ret 58.97 vs bge-base 53.25) moved the projection
    45	   from 0.406 to 0.489 and cleared the bar outright. Quality is usually a *product* of factors —
    46	   check every factor before declaring the product capped.
    47	2. **Every failing component has been diagnosed, not just observed.** Same day: contrastive
    48	   training was called "broken" when the learning rate was 30-300x above every published recipe
    49	   and the literature has an analytic result (arXiv 2110.09348) for precisely that symptom. A
    50	   failure you cannot explain mechanistically is not evidence about the method — it is evidence
    51	   about your configuration.
    52	3. **The literature has been swept for the specific failure**, in parallel subagents, with the
    53	   real numbers extracted. Cheap, fast, and it has twice overturned a conclusion here.
    54	4. **A capability claim has been checked algebraically before being believed or dismissed.** Some
    55	   "obvious" fixes are provably no-ops (a doc-side linear map reparametrises the table;
    56	   document centering cannot change ranking at all). Do the algebra — and note the example
    57	   originally given here was itself wrong: query-side centering *was* claimed to be genuine new
    58	   capacity, and `results/m7_absorb_check.json` later disproved it to machine precision
    59	   (`mean(W-mu) = mean(W)-mu`, so it is absorbable, as are whitening, top-PC removal and any
    60	   per-token scalar weight). Only n-gram rows and multiplicity-dependent pooling add capacity.
    61	   The lesson stands; this file had the sign of it backwards for a day.
    62	5. **The negative result is reported with what would change it** — the specific measurement,
    63	   component, or bound that would flip the verdict.
    64	
    65	Corollary: a pre-registered kill criterion (e.g. the phase-2 contrastive bar) exists to stop
    66	*grinding a diagnosed dead end*, not to license abandoning an undiagnosed one. Kill the avenue
    67	after you understand why it failed, never before.
    68	
    69	### Vendor rule, relaxed (Dylan, 2026-08-26)
    70	
    71	The original rule — no component from any vendor shipping a competing vector-search product —
    72	disqualified almost every strong encoder and was costing us real quality. **Relaxed: a vendor whose
    73	vector-search offering is far from their main business is acceptable if the choice is heavily
    74	justified in the report.** Direct competitors stay out.
    75	
    76	Operationalised so a future session does not have to guess:
    77	
    78	- **OUT — vector search *is* the business:** Pinecone, Weaviate, Zilliz/Milvus, Chroma, Vespa,
    79	  Nomic (Atlas), Mixedbread, Jina (and Elastic), MongoDB (Atlas Vector Search is a flagship push;
    80	  also Voyage AI, which MongoDB acquired), Cohere (Embed/Rerank is core product). And obviously
    81	  nothing where we would be shipping a competitor's model as a *component* of a Qdrant release.
    82	- **OK WITH JUSTIFICATION — vector search is one service among hundreds:** Alibaba (gte, Qwen3-Embedding
    83	  — OpenSearch Vector Search Edition / AnalyticDB), Microsoft (e5 — Azure AI Search), IBM (granite —
    84	  watsonx managed Milvus), Google (Vertex AI Vector Search; note Gemma terms still fail the
    85	  *licence* rule independently). **Snowflake (arctic-embed) needs the strongest justification of
    86	  this group** — Cortex Search is built directly on Arctic Embed, so it is the closest of these to
    87	  core business.
    88	- **CLEAN — no vector product at all:** BAAI, NovaSearch, NVIDIA, Salesforce, academic labs,
    89	  individual/community releases (e.g. intfloat).
    90	
    91	Unchanged and non-negotiable: the **licence** must permit commercial release of derived weights
    92	(no Gemma terms, no CC-BY-NC), and **table size still binds** — the released artifact is
    93	vocab x dim, so a 250K-vocab model is disqualified on arithmetic regardless of vendor
    94	(250,002 x 1024 fp16 = 512 MB, larger than the 466 MB LightRetriever table we exist to beat).
    95	Practical filter: **vocab <= ~50K and dim <= 1024, or MRL-truncatable**.
    96	
    97	The pre-relaxation shortlist (`research/m7-teacher-shortlist.md`) was written under the strict
    98	rule, superseded by `research/m7-teacher-shortlist-2026-08-26.md`, and deleted in the 2026-08-28
    99	cleanup — git history has it. The re-run sweep happened and the teacher question is closed
   100	(stella, see the decision log).
   101	
   102	### Past decisions are revisitable (Dylan, 2026-08-26)
   103	
   104	**"If we need to revisit any past decisions to make this better, I'm open for it. Achieving our
   105	goal supersedes anything else."** So do not treat an earlier choice as settled just because it is
   106	written down — including choices in `instructions-m7.md` and in the decision log below. Anything
   107	in this list is fair game to reopen with evidence and Dylan's sign-off: the teacher, the
   108	architecture, the objective, the data mix, the tier definitions and their comparators, the
   109	milestone scope, the release target, and even structural premises like "frozen off-the-shelf
   110	document tower" or "no transformer at query time" if a better system lies the other side of them.
   111	Bring the arithmetic and the trade-off, not just the idea.
   112	
   113	**The one class of exception, and why it is not an exception to the goal.** The evaluation
   114	protocol — partitions, decontamination, the frozen comparator vectors, the single final run,
   115	pre-registered statistics — exists to make a good number *believable*. Relaxing it after seeing
   116	results does not achieve the goal, it destroys the thing the goal is for: an unbelievable 0.50 is
   117	worth less than a defensible 0.46. So protocol changes are allowed, but only **before** the
   118	numbers they would affect are observed, and only written down with the reasoning. Never
   119	retroactively, never silently. Same rule for the licensing and vendor constraints: they are
   120	commercial reality, so reopening them is Dylan's call and needs an explicit answer, not an
   121	inference.
   122	
   123	## Long runs must be watched, not launched and hoped for (Dylan, 2026-08-26)
   124	
   125	Added after a session lost most of a day to jobs that were running but wrong. Four failures in one
   126	morning: a probe that thrashed the allocator for 50 minutes per component, a mining pass costing
   127	3.6 hours instead of 3 minutes, four screen arms crashing on one shape error, and a pool whose
   128	width check would have silently overwritten 9.5 GB. Every one was visible within two minutes of
   129	starting, and none announced itself.
   130	
   131	**Before launching anything that runs longer than ~10 minutes:**
   132	
   133	1. **Smoke the code path first.** `sweep.smoke(base, over)` runs an arm at 90 steps. Prefer a path
   134	   with no execution history — that is where the bug is. A grid's arms share code, so
   135	   `sweep.grid(..., fail_fast=True)` (the default) stops at the first arm that *raises*; an OOM
   136	   does not trip it, being a real per-arm resource result.
   137	2. **Arm a monitor with the failure signatures, not just the success line.** Silence is not
   138	   success: grep an alternation covering `Traceback|Error|FAILED|OOM|Killed|assert` alongside the
   139	   progress marker. A monitor that only matches the happy path is indistinguishable from a
   140	   crashloop.
   141	3. **Read the first progress line and sanity-check the RATE against an estimate** before walking
   142	   away. "mine 2048/349934 (76s)" is a three-hour job stated in the units of a fast one. Two
   143	   minutes of arithmetic at launch is worth hours later.
   144	4. **Watch the machine, not only the log.** 100% GPU utilisation with ~1% memory-bandwidth
   145	   utilisation and low power draw means allocator thrash or launch-bound tiny kernels, not work.
   146	   `/proc/<pid>/stat` utime against elapsed says whether a "GPU job" is actually burning one CPU
   147	   core; per-thread CPU time says whether it is Python or a kernel. `py-spy` needs
   148	   `kernel.yama.ptrace_scope=0`, which needs root on this box — so these cheaper signals matter.
   149	5. **`tail -F`, never `tail -f`, and match the log's own casing.** A monitor naming a chained
   150	   job's log before that file exists follows nothing when it appears — plain `-f` does not retry —
   151	   and if the liveness check still matches the process, **silence looks exactly like progress**.
   152	   That cost 3.5 h of unwatched training on 2026-09-08, noticed only because Dylan asked. The same
   153	   monitor also missed the completion line by grepping `COMPLETE` for a log that writes `complete`.
   154	6. **Never trust a docstring's cost estimate.** Both "a few minutes for the whole set" claims in
   155	   `train.py` were wrong by two orders of magnitude, in the same file, on the same day.
   156	
   157	## A number from one measurement is an assumption (2026-09-07)
   158	
   159	Written after a memory bug took four wrong diagnoses and most of a session. Each was stated as
   160	settled and each died to the next measurement: per-row growth, page cache, the chunk size, the
   161	tokenizer. The bug was a fixed 2.2 GB load in `row_texts`, found only by asking it for **ten rows**.
   162	
   163	- **A per-unit rate needs two points.** "+3,453 MB at 1M → 4.3 KB/row → 20.4 GiB at 5M" was a
   164	  fixed cost divided by n. Two sizes showed 5× the work costs **+29 MB**. One point cannot tell a
   165	  constant from a slope, so it cannot support an extrapolation — and 6× wrong here nearly cancelled
   166	  a family.
   167	- **Vary the knob you are blaming.** Chunk-invariance was read as "the tokenizer's transient". A
   168	  10× smaller *tokenizer batch* moved the peak 4%. If you have not moved the thing you are
   169	  accusing, you have not tested it.
   170	- **Measure at the registered dose, on the metric that failed.** The 300-step smoke passes because
   171	  it shrinks the very count that breaks. The crashes presented as **host free 239 MB**, not as
   172	  process RSS — so that is the number a pre-flight has to report.
   173	- **Know the instrument.** `VmHWM` is monotonic; a 5-second `VmRSS` sample is a lower bound and
   174	  hides any transient shorter than the interval. `RssAnon` is unreclaimable, `RssFile` is page
   175	  cache the kernel drops — opposite consequences. A first probe wrote to `/tmp`, which is tmpfs,
   176	  and mis-categorized the pages entirely. And a `pgrep -f` pattern matches the *launching shell*.
   177	- **Reconcile a new number against the ones already written down.** Three figures got labelled
   178	  "three runs of one path"; two were this repo's own per-implementation measurements. That
   179	  fabricated an error bar where none existed.
   180	
   181	## Durable knowledge goes in the repo, not in the assistant's memory (Dylan, 2026-08-26)
   182	
   183	The point of this repo is a reusable harness, so anything a future session would need belongs in a
   184	file it will read: protocol in `m7/LEDGER.md`, dead ends in `m7/EXPLORED.md`, module facts and
   185	pitfalls in `m7/CODEMAP.md`, standing directives here. Assistant memory is for *Dylan-and-workflow*
   186	facts only (how he wants sessions run, what tooling exists on the box), and anything in it that
   187	would help a future session must be mirrored into one of those files. A lesson that lives only in a
   188	memory file is lost to every session that does not happen to recall it.
   189	
   190	## Markdown files must be TIGHT (Dylan, 2026-08-29)
   191	
   192	*"Information is wayyyy too verbose in there. We're diluting context for next session."* Said of an
   193	M8 ledger that had reached 2,694 lines, ~40% of it one session's amendment prose.
   194	
   195	**Every line in a `.md` file costs a future session context it could have spent on the work.** A
   196	file a session is *told* to read before deciding — `CLAUDE.md`, `STATUS.md`, `LEDGER.md` — is loaded
   197	whether or not the reader needs the paragraph you enjoyed writing. Verbosity there is not
   198	thoroughness; it is a tax levied on every session that follows.
   199	
   200	**The rule:** write the decision, the number a rule reads, and the pointer. Nothing else.
   201	
   202	- **One fact, one home.** Numbers live in the result JSON, bars in `registry.json`, runs in
   203	  `RESULTS.md`, closed avenues in `EXPLORED.md`, long-form reviews in `research/*`. A `.md` that
   204	  restates any of them is duplication that will go stale in exactly one direction — the wrong one.
   205	- **An amendment is: what changed, why, and the pointer.** Not the reasoning that got you there.
   206	  If the reasoning matters, it belongs in the archived review, cited by path.
   207	- **Prefer a table to prose, a clause to a sentence, a pointer to a summary.** Cut every sentence
   208	  that only restates the previous one with more emphasis.
   209	- **Withdrawn claims and owner rulings are the exception — always keep them**, because a future
   210	  session that re-derives a withdrawn claim wastes far more than the lines cost. Keep them *short*.
   211	- **When you add to a long file, budget for it**: if an entry runs past ~10 lines, compress an old
   212	  one or move detail out. Files grow by default; only deliberate effort shrinks them.
   213	- **Check the size when you touch it.** `wc -l` on the file you just edited. If a protocol file has
   214	  grown past ~1,500 lines, compressing it is part of the task, not a separate one.
   215	
   216	## Verification gates (Dylan, 2026-08-24)
   217	
   218	Results dictate Qdrant engineering decisions: correct, not decimal-precise; blind spots stated openly.
   219	- M2 gate: adversarial Opus review of methodology + code + M2 numbers → `research/verification-m2.md` (done).
   220	- Pre-report gate: Codex second opinion on the full result set + report draft, briefed for pushback; findings reported verbatim.
   221	- Significance: paired bootstrap on key system pairs before the report; deltas within noise get labeled as such.
   222	- **Codex CLI is installed on the box** (`/usr/local/bin/codex`, 0.149.1). Invoke it read-only and
   223	  with high reasoning effort, since the default profile here is effort `none`:
   224	  `codex exec -s read-only -m gpt-5.6-sol -c model_reasoning_effort="high" - < brief.md`. Read-only
   225	  matters: a previous review committed to the files the session was editing.
   226	- **STANDING GRANT, Dylan 2026-08-28 — supersedes the earlier "sparingly, at milestones only".**
   227	  *"Feel free to do adversarial reviews with Fable and Codex as much as needed. Use them to push
   228	  you to get the best result possible. They've proven valuable every time."* So reviews are a
   229	  **routine instrument, not a milestone ceremony**: brief one whenever a decision is about to
   230	  become expensive or irreversible, whenever a document is about to become permanent, and whenever
   231	  a result is surprising enough that you want it attacked before you believe it. The 2026-08-28
   232	  record is why — between them the two reviews caught a rounded CI endpoint in `final_run.py` (the
   233	  one irreversible decision in the project), a gate that returned 0 on NO-GO, a release guard that
   234	  failed open, a confounded diagnostic that had licensed a four-hour lever, and a retention figure
   235	  belonging to a reverted candidate. **Brief them adversarially**: give them the numbers, name what
   236	  you believe, and ask them to break it. A review told "confirm this" returns nothing.
   237	- **EVERY brief must carry a READ-EXCLUSION, and the log must be audited afterwards (2026-08-29
   238	  incident).** An external reviewer is a separate process with ordinary read access to the whole
   239	  repo, so `paths_guard`/G2 — an in-process Python bulkhead — is structurally incapable of
   240	  constraining it. A repo-wide grep dumped two RESERVED confirmatory sets, queries **and qrels**,
   241	  complete, into a reviewer's context. Nothing was scored and no decision read them, but the
   242	  reviewer's recommendations were thereby potentially informed by held-out data. So: state in the
   243	  brief that `results/frozen_eval/untouched-*`, the reserved qrels caches and `work/m9reserve` must
   244	  not be read; prefer naming the files to read over inviting a repo-wide search; and **grep the
   245	  review log for reserved-set reads before reading its findings.** Quarantine anything that draws
   246	  on them.
   247	
   248	## How to run an adversarial review, and why it is not optional (Dylan, 2026-09-05)
   249	
   250	*"Note the adversarial reviews process somewhere so next sessions keep doing it."* The standing
   251	grant above says reviews are routine; this is **how**. On 2026-09-05 six reviews caught, among
   252	other things: two arms that **could not win their own contrast**, a familywise α that was 0.0289
   253	not 0.025, a diversity gate **structurally blind** to 5 of 12 forms, the fact that **the M10 corpus
   254	had no path into the trainer at all**, and — on a re-review — that my own "fixes" for the first
   255	batch were **prose the validator never evaluated**.
   256	
   257	**The five rules that made those catches happen.**
   258	
   259	1. **Scope the read, always.** Every brief names the exact files to read and forbids a repo-wide or
   260	   recursive grep. This is not tidiness: a repo-wide grep once dumped two RESERVED confirmatory
   261	   sets into a reviewer's context. **Then grep the review log for reserved reads BEFORE reading its
   262	   findings**, and quarantine anything that drew on them.
   263	2. **State what you believe and ask them to break it.** Give the numbers, name your conclusion,
   264	   invite the attack. A review told "confirm this" returns nothing — and a review that returns
   265	   nothing is a badly written brief, not a clean bill of health.
   266	3. **Review the FIX, not just the decision.** The single most valuable catch of the day was a
   267	   re-review of eleven fixes that returned **NO-GO** because they were registry prose no code read.
   268	   *Writing something down is not making it work.* Same failure shape as a test that asserts a
   269	   sentence exists, or one that compares 0.0 ≤ 0.0.
   270	4. **Verify every checkable claim yourself before acting.** Reviewers are wrong sometimes, and a
   271	   claim you reproduce is one you can defend. Everything acted on today was re-derived first.
   272	5. **Two independent reviewers on anything expensive or irreversible.** They disagree usefully:
   273	   one said documenting a contradiction was enough, the other showed the registry then carried
   274	   *both* readings at once. Use different models, not the same one twice.
   275	
   276	**And record what the review overturned, including your own withdrawn claims** — four of mine were
   277	withdrawn on 2026-09-05 and each is in `m10/LEDGER.md` with the reasoning. A future session that
   278	re-derives a withdrawn claim wastes far more than the lines cost.
   279	
   280	## Key decisions (log)
   281	
   282	- **Teacher = `NovaSearch/stella_en_400M_v5` (2026-08-26), chosen on the distilled table, not the tower.** `arctic-embed-l` had been picked the same morning on its own dev retrieval quality (+0.0447 over bge-base) and was **WITHDRAWN the same day**: ranked by the table distilled from it, arctic sits −0.0480 [−0.0608, −0.0349] below bge-base, and a teacher's own quality does not predict its table (Spearman 0.000 over eight candidates). Only stella beat the incumbent (+0.0365 [0.0249, 0.0481]). Stella discloses ArguAna and FiQA (2 of the six) and FEVER (1 of the reserved four) in its training data; every stella-based claim carries that qualification and the NDO-4 rows. `results/m7_learnability_report.json`, `results/m7_teacher_contamination.json`. Kept because the failure mode — selecting a teacher on the tower — is the lesson. M9 re-confirmed it for towers: stella-1.5B distils WORSE than stella-400M (−0.0023 at equal dose).
   283	- **Clean data stack (2026-08-25, amended 2026-09-04):** MS MARCO permanently excluded from anything released (non-commercial terms; priced at +0.0058 [−0.0015, +0.0131] on the six — not what the miss is made of). CC BY-SA sources (NQ, SQuAD, HotpotQA, FEVER, MIRACL, Mr.TyDi) approved with model-card attribution. FineWeb was approved 2026-08-30 as document-side regression text but is **out of M10 in every role** (ruled 2026-09-01 under delegated authority: no reserved-set document fingerprints exist, and seeding from it would add a rights review for no measured gain). `research/m7-data-licensing.md`.
   284	- **Non-commercial licences are admissible for VALIDATION, not training (Dylan, 2026-09-04):** *"MS-MARCO is allowed for validation. Not training"*, generalised by him to any similarly-licensed dataset. Reaches only sets with an **affirmative but non-commercial grant** (MS MARCO, CC BY-NC eval sets, PAQ's NC generator, GooAQ). **Does NOT reach** sets with no grant at all (Quora QQP, WikiAnswers, ELI5, StackExchange downloads — QQP's eval exclusion stands) or sets excluded for contamination (S2ORC). Validation = measuring, screening, diagnosing, reporting; **never** gradients, distillation targets, mined negatives or synthetic-generation seeds. The training exclusion and its +0.0058 price are unchanged, as are `m7src/freeze.py`'s `NON_COMMERCIAL_SOURCES` guard and the `sources-research/` quarantine. Live consequences: M10's COV admission reopens (**ConsumerContractsQA back in, three families without LEDGER**) and gains a real-query coverage diagnostic; M12 may add a held-out fusion-weight transfer check; M14 may report an MS MARCO row. **Not reopened:** M7 (frozen), M9's `FINAL_LOCK.md` (frozen, six named datasets), the reserved four, any released artifact. Rule and the two confounds: `research/m7-data-licensing.md` §Rule change 2026-09-04.
   285	- **Reserved four** (FEVER, DBpedia-entity, cqadup-android, cqadup-english): one confirmatory access, unspent through M9. **`results/perquery.json` is irreplaceable** — frozen comparator vectors regenerated from caches that no longer exist. Never overwrite it.
   286	- **`zero` published 2026-09-03 (Dylan's ask); public from the first push, renamed `constella-zero` the same day.** The M7 artifact unchanged, on Dylan's own HF account, so the architecture can be tested as a PoC. Card rewritten FastEmbed-first 2026-09-03: competitive comparison and the missed bar REMOVED (`instructions-m11.md` Amendment B), measured numbers and the stella-contamination caveat kept. **Licence: MIT, ruled by Dylan 2026-09-03 and valid for a public release** (matching stella), with CC BY-SA attribution for NQ/SQuAD/HotpotQA/FEVER/Mr.TyDi. `m11/STATUS.md`.
   287	- **Renumbering 2026-09-01 (Dylan):** M10 becomes the nano retry; release/port/whitepaper moves to M11; the image model to M12.
   288	- **The upstream FastEmbed PR is M13, not now (Dylan, 2026-09-04):** *"once we have both models we add all of them in one clean PR."* `constella-zero` and the document tower are already published and registered, but the PR waits for nano so a maintainer gets one PR with all three entries. It branches fresh from upstream `main`; `Dylancouzon/fastembed@add-constella-models` is a throwaway (it also carries the #703 padding fix) and is never merged.
   289	- **Doc-tower card left unchanged, and the prompt-pair stays out of the CTO ask (Dylan, 2026-09-04).** The published doc graph IS a complete query encoder (prompt prepended → min-cos 1.00000000 vs torch, `m11/STATUS.md` addendum), but the card's narrower wording stands because FastEmbed's `query_embed` returns a wrong-protocol vector silently. The Cloud Inference ask covers document hosting only.
   290	- **Renumbering 2026-09-04 (Dylan), and the hybrid milestone:** `constella-zero-hybrid` is inserted as **M12**; nano's release M12→**M13**, the image model M13→**M14**, the better-zero idea register M14→**M15**. **Amended the same day**: the paper was split out of M13 into its own **M14** (Dylan: *"probably only 1"* paper, decided at M14's start), so the image model is **M15** and the better-zero register **M16**. Historical renumbering entries above are left as written. Review-finding IDs named `M12`–`M15` in `m9/`, `m10/` and `research/` were NOT touched — they are findings, not milestones; the two E14-LORA pointers that did mean a milestone were rewritten to name M15 explicitly.
   291	- **Renumbering 2026-09-03 (Dylan), and M11 CLOSED:** M11 shipped everything not depending on M10 (the zero half, both ONNX ports, the FastEmbed integration, both cards). The blocked remainder — nano's release and port, its FastEmbed entry, the whitepaper — becomes **M12**; the image model moves **M12 → M13**. Note for a cold session: most `M12` strings in `m9/`, `m10/` and `research/` are review-finding IDs, NOT milestones — they were deliberately left alone.
   292	- **Student cap 35M, hard (Dylan, 2026-09-01):** "109M is not an option. This isn't low compute anymore. 33M was already in the upper bound of what I think is acceptable." No larger student in any role; the cheaper student gets the benefit of a tie.
   293	- **M10 compute (Dylan, 2026-09-01):** "M10 won't be done on a 3080. M10 will be done on a GPU budget, if allowed, or not at all." The box is not an execution target for M10; doses and screen sizes are set from the budget, not the box (`m10/PLANNING.md` §5–6 and the §8 amendment block). Budget approval is decision 2; refusal closes M10 unstarted. **Amended 2026-09-04 (A7 confirmed; decisions 14, 15): the box runs the screens and the generation (Qwen's official AWQ release), the cloud buys the build; the ruling still binds the dose.**
   294	- **M9 close-out (2026-09-01):** LoTTE read #1 withdrawn unexecuted so M10 inherits a fresh surface; the six-set transaction runs six-only (reserved conditional struck, ratification pending) after M10's recipe lock; the `m9-status` branch deleted. M10 execution work happens on branch `m10-work` under the headless commit-and-push contract.
   295	- Harness, BEIR subset, LightRetriever reproduction details and the M1/M2 findings: `research/m1-m6-findings.md`.

exec
/bin/bash -lc 'nl -ba m10/screen_registry.json' in /home/dylan/asymetric-dual-encoders
 succeeded in 0ms:
     1	{
     2	 "_what": "M10 screen lock (LEDGER §0a), data-independent constants. Prose is not authoritative; this file is. Nothing here is chosen after seeing an arm. `m10src/screen_lock.py` validates it and is the only reader a rule may use.",
     3	 "_registered": "2026-09-05, before any arm and before any build seed draw",
     4	 "order": [
     5	  "F",
     6	  "A",
     7	  "G",
     8	  "B",
     9	  "E",
    10	  "C",
    11	  "D"
    12	 ],
    13	 "anchor": {
    14	  "_what": "the arm every non-F contrast compares against, TRAINED IN ITS OWN RIGHT at the screen recipe. It was registered as 'F's winner at its 5M checkpoint'; a Fable pass showed that is ~75% through cycle 1 of a 20M schedule at LR ~3e-5, un-annealed, against genuine 5M arms that run three cycles annealed to 1e-5 -- a handicap on the `b` side of ten contrasts, in the direction that adopts a non-default. It also left the anchor with no second cycle end (breaking the sign-stability clause) and no DEV-6 read beside its COV reads.",
    15	  "arm": "ANCHOR",
    16	  "data": "A4, the CUT corpus (see data_cut)",
    17	  "mix": "75/25",
    18	  "window_pattern": "3Q+1D",
    19	  "init": "F's winner backbone",
    20	  "objective": "squared_l2",
    21	  "batch": 32,
    22	  "feature_layers": 3,
    23	  "feature_dim": 1152,
    24	  "dose_examples": 5000000,
    25	  "schedule": "the screen recipe: 3 cycles of equal example count, each linear 1e-4 -> 1e-5",
    26	  "seed": 0
    27	 },
    28	 "screen_dose_examples": 5000000,
    29	 "seed_rule": "one seed, seed 0, for every screen arm; confirmation arms use seeds 1 and 2",
    30	 "arms": {
    31	  "F-bge-small": {
    32	   "family": "F",
    33	   "dose_examples": 20000000,
    34	   "read_at": [
    35	    5000000,
    36	    10000000,
    37	    20000000
    38	   ],
    39	   "student": "bge-small",
    40	   "note": "20M curve; its 5M reading is NOT the anchor (S1)",
    41	   "trained": true
    42	  },
    43	  "F-MiniLM-L6": {
    44	   "family": "F",
    45	   "dose_examples": 20000000,
    46	   "read_at": [
    47	    5000000,
    48	    10000000,
    49	    20000000
    50	   ],
    51	   "student": "MiniLM-L6-v2",
    52	   "trained": true
    53	  },
    54	  "F-MiniLM-L12": {
    55	   "family": "F",
    56	   "dose_examples": 5000000,
    57	   "student": "MiniLM-L12-v2",
    58	   "trained": false,
    59	   "schedule": "the 20M three-cycle schedule TRUNCATED at 5M, not a 5M schedule. W9, Codex finding 5: F-bge-small and F-MiniLM-L6 are READ at 5M inside a 20M schedule, so at that point they are ~75% through cycle 1 and un-annealed, while a genuine 5M arm would have completed three compressed annealed cycles -- the exact handicap S1 created the standalone anchor to remove, reintroduced one family over. Truncating L12's 20M schedule at 5M costs nothing extra and reads all three arms at the SAME schedule position.",
    60	   "selectable": false,
    61	   "cut": "CUT, Dylan 2026-09-05 (verification-pass rulings, pre-observation: nothing registered has trained). MiniLM-L12-v2 is 12 layers / 384 hidden / 34,540,672 params with the head -- identical in size and depth to bge-small -- so F2 tested the pretraining checkpoint, not depth (the hypothesis it was added for). Cutting it removes 5-20M screen examples and closes W13 (the conditional extension was an adaptive selection step)."
    62	  },
    63	  "A1": {
    64	   "family": "A",
    65	   "dose_examples": 5000000,
    66	   "data": "M9 pool (463,314 queries)",
    67	   "trained": true,
    68	   "selectable": true
    69	  },
    70	  "A2": {
    71	   "family": "A",
    72	   "dose_examples": 5000000,
    73	   "data": "M9 pool + PAQ (volume control)",
    74	   "trained": true,
    75	   "selectable": true
    76	  },
    77	  "A3": {
    78	   "family": "A",
    79	   "dose_examples": 5000000,
    80	   "data": "A2 + harvested real query-like text",
    81	   "trained": true,
    82	   "selectable": true,
    83	   "_note_selectable": "A3's corpus IS selectable as the build corpus: `rules.generated_half_in_build` says an unresolved A4-A3 makes the build use A3's corpus and drops the ~1.0M generated queries."
    84	  },
    85	  "A4": {
    86	   "family": "A",
    87	   "dose_examples": 5000000,
    88	   "trained": false,
    89	   "data": "A3 + the generated forms (the full M10 corpus, cut)",
    90	   "note": "not trained separately: A4 IS the ANCHOR arm"
    91	  },
    92	  "G-384": {
    93	   "family": "G",
    94	   "dose_examples": 5000000,
    95	   "feature_layers": 1,
    96	   "feature_dim": 384,
    97	   "trained": true,
    98	   "selectable": false,
    99	   "_w9r": "**DIAGNOSTIC-ONLY, never selectable.** The mandate registers the contrast as `1152 - 384` (:616) and the arm's stated purpose as 'the paper's evidence for the M9 diagnosis', so 384 is not a candidate recipe -- it is the thing the M9 diagnosis says is too narrow. Keeping the orientation AND letting 384 win were two incompatible readings the registry briefly carried at once (W9 review #3). G1 therefore tests whether 1152 beats 384 and can select only 1152; G-384 is excluded from winner selection and from confirmation."
   100	  },
   101	  "G-1536": {
   102	   "family": "G",
   103	   "dose_examples": 5000000,
   104	   "feature_layers": 4,
   105	   "feature_dim": 1536,
   106	   "trained": true,
   107	   "selectable": true
   108	  },
   109	  "G-MLP": {
   110	   "family": "G",
   111	   "dose_examples": 5000000,
   112	   "head": "W_lin*x + W2*GELU(W1*x+b1), W1 1152->192, per-token, before the pool",
   113	   "trained": true,
   114	   "selectable": true
   115	  },
   116	  "B-100/0": {
   117	   "family": "B",
   118	   "dose_examples": 3750000,
   119	   "pattern": "4Q",
   120	   "trained": true,
   121	   "selectable": true
   122	  },
   123	  "B-50/50": {
   124	   "family": "B",
   125	   "dose_examples": 7500000,
   126	   "pattern": "2Q+2D",
   127	   "trained": true,
   128	   "selectable": true
   129	  },
   130	  "E-bs128": {
   131	   "family": "E",
   132	   "dose_examples": 5000000,
   133	   "batch": 128,
   134	   "trained": true,
   135	   "selectable": true
   136	  },
   137	  "C-M9init": {
   138	   "family": "C",
   139	   "dose_examples": 5000000,
   140	   "init": "the M9 candidate",
   141	   "trained": false,
   142	   "head_init": "the M9 candidate keeps its 384-d head and the two extra layers' columns are ZERO-initialised (§Recipe). C1 therefore confounds backbone init with head init; that is registered and disclosed, not resolved.",
   143	   "selectable": true,
   144	   "cut": "**CUT under W8 band 1, 2026-09-05.** `C-M9init` at 5M warm-starts from a model already trained 3.69B tokens, so it wins its own contrast BY CONSTRUCTION and says nothing about 200M; it is separately disclosed as confounding backbone init with head init; and it would consume one of only two confirmation slots. The build init is the registered default, the off-the-shelf backbone. C1 does not run and is reported as cut. **The Bonferroni denominator STAYS 13** (`rules.C_skipped`, which already registers that for the skip path)."
   145	  },
   146	  "D-NORM": {
   147	   "family": "D",
   148	   "dose_examples": 5000000,
   149	   "objective": "leaf_norm_e2",
   150	   "trained": true,
   151	   "selectable": true
   152	  },
   153	  "D-COV": {
   154	   "family": "D",
   155	   "dose_examples": 5000000,
   156	   "objective": "document_covariance_weighted",
   157	   "trained": true,
   158	   "selectable": true
   159	  },
   160	  "ANCHOR": {
   161	   "family": "anchor",
   162	   "dose_examples": 5000000,
   163	   "trained": true,
   164	   "note": "the comparator for G, B, E, C, D and A4-A3; trained once"
   165	  }
   166	 },
   167	 "contrasts": {
   168	  "F1": {
   169	   "a": "F-MiniLM-L6",
   170	   "b": "F-bge-small",
   171	   "at_examples": 20000000,
   172	   "decisive": true,
   173	   "tails": 2,
   174	   "quantile": 0.0010416666666666667,
   175	   "family_rule": "F_tournament"
   176	  },
   177	  "A3-A2": {
   178	   "a": "A3",
   179	   "b": "A2",
   180	   "decisive": true,
   181	   "rule": "three_outcome"
   182	  },
   183	  "A4-A3": {
   184	   "a": "A4",
   185	   "b": "A3",
   186	   "decisive": true,
   187	   "rule": "generated_half_in_build"
   188	  },
   189	  "G1": {
   190	   "a": "G-1152",
   191	   "b": "G-384",
   192	   "decisive": true,
   193	   "_w9": "Orientation `1152 - 384` is the MANDATE's (:616) and is kept. `G-384` is registered `selectable: false` -- a diagnostic arm, not a candidate recipe -- so G1 tests whether the wider head beats M9's 384 and can select only 1152. A negative point estimate is REPORTED as evidence bearing on the M9 diagnosis; it is not an action, and an unresolved negative is noise, not a contradiction (W9 review #3).",
   194	   "family_rule": "multi_arm_winner"
   195	  },
   196	  "G2": {
   197	   "a": "G-1536",
   198	   "b": "G-1152",
   199	   "decisive": true,
   200	   "family_rule": "multi_arm_winner"
   201	  },
   202	  "G3": {
   203	   "a": "G-MLP",
   204	   "b": "G-1152",
   205	   "decisive": true,
   206	   "family_rule": "multi_arm_winner"
   207	  },
   208	  "B1": {
   209	   "a": "B-100/0",
   210	   "b": "B-75/25",
   211	   "decisive": true,
   212	   "family_rule": "multi_arm_winner"
   213	  },
   214	  "B2": {
   215	   "a": "B-50/50",
   216	   "b": "B-75/25",
   217	   "decisive": true,
   218	   "family_rule": "multi_arm_winner"
   219	  },
   220	  "E1": {
   221	   "a": "E-bs32",
   222	   "b": "E-bs128",
   223	   "decisive": true,
   224	   "rule": "E_cost",
   225	   "_w9": "REORIENTED 2026-09-05 (W9, Codex finding 1). It read `E-bs128 - E-bs32`, so a bs32 quality win was a NEGATIVE point estimate and `point >= threshold` refused it -- making the mandate's own action, 'a RESOLVED bs32 win is adopted', unreachable. Read `bs32 - bs128` the action is reachable with ONE tail and the cost rule is unchanged: resolve -> bs32 earns its 2.2x build cost; do not resolve -> bs128."
   226	  },
   227	  "C1": {
   228	   "a": "C-M9init",
   229	   "b": "C-fresh",
   230	   "decisive": true
   231	  },
   232	  "D1": {
   233	   "a": "D-NORM",
   234	   "b": "anchor",
   235	   "decisive": true,
   236	   "family_rule": "multi_arm_winner"
   237	  },
   238	  "D2": {
   239	   "a": "D-COV",
   240	   "b": "anchor",
   241	   "decisive": true,
   242	   "family_rule": "multi_arm_winner"
   243	  }
   244	 },
   245	 "descriptive_contrasts": {
   246	  "A4-A2": {
   247	   "a": "A4",
   248	   "b": "A2",
   249	   "decisive": false
   250	  },
   251	  "A2-A1": {
   252	   "a": "A2",
   253	   "b": "A1",
   254	   "decisive": false,
   255	   "note": "the volume effect; DESCRIPTIVE ONLY, no build action (rules.A2-A1)"
   256	  },
   257	  "arxiv-title": {
   258	   "a": "arxiv-title diagnostic",
   259	   "b": null,
   260	   "decisive": false,
   261	   "note": "not a contrast; a descriptive surface read, no action"
   262	  }
   263	 },
   264	 "statistics": {
   265	  "surface": "COV macro, family-weighted, four family IDs at equal weight, units averaged within family",
   266	  "families": [
   267	   "BRIGHT",
   268	   "consumer-health",
   269	   "finance",
   270	   "legal"
   271	  ],
   272	  "bootstrap": {
   273	   "kind": "paired stratified over queries within unit",
   274	   "B": 200000,
   275	   "seed": 0,
   276	   "quantile": 0.0020833333333333333,
   277	   "quantile_method": "inverted_cdf",
   278	   "alpha_family": 0.025,
   279	   "n_contrasts": 12,
   280	   "chunk": 5000,
   281	   "alpha_accounting": "11 one-sided contrasts at alpha/12 plus F1, which is TWO-SIDED because F is oriented highest-minus-runner-up after the reading. F1's tails take alpha/24 each, so the union bound is 11*(0.025/12) + 2*(0.0125/12) = 0.025 EXACTLY. History: 13 contrasts until F2 was cut with L12 (Dylan 2026-09-05); before W9 every tail took alpha/13 and the familywise rate was 0.02885.",
   282	   "_runner_contract": "`cov_macro.contrast` takes `quantile` as a parameter (default alpha/13). **The runner MUST pass each contrast's own `quantile` field**; the global default is correct only for the eleven one-sided contrasts. `screen_lock` now validates that the per-contrast quantiles sum to the registered alpha, but it cannot see the call site -- the arm runner is responsible and must be reviewed when it is written (W9 review #5)."
   283	  },
   284	  "MDE": 0.0056,
   285	  "resolve_rule": "point >= MDE AND one-sided lower bound at the quantile > 0 AND the sign is stable across the last two cycle-end checkpoints",
   286	  "measured_resolution_distance": 0.008619,
   287	  "measured_resolution_note": "results/m10_cov_resolution.json. The MDE is BELOW the distance, so a contrast landing at the MDE cannot resolve. Power disclosure only: it sizes nothing (amendment A4's sizing struck by the Codex pass) and alpha does not move.",
   288	  "implementation": "m10src/cov_macro.py",
   289	  "_MDE_naming": "`MDE` is a DECISION THRESHOLD, not a minimum detectable effect. `resolve_rule` requires the POINT estimate to reach it, which caps power at 50% at a true effect of exactly 0.0056 however narrow the interval; family A's rule is `lower > MDE`, stronger still. The name is kept because it is referenced throughout, but it does not mean what it says (W9, Codex finding 11)."
   290	 },
   291	 "outcome_to_action": {
   292	  "F": "F1 = MiniLM-L6 vs bge-small at 20M. Winner = the higher 20M COV macro if its margin over the other resolves under the two-sided F rule at alpha/24 per tail; else the cheapest to serve (`rules.serve_cost_order`), labelled in the report as a product preference, not evidence. L12 is cut (Dylan 2026-09-05). Every later family runs on F's winner and every later verdict is labelled conditional on it.",
   293	  "A": "A3-A2: three reported labels (RESOLVED / POSITIVE NOT RESOLVED / NOT POSITIVE), NO STOP, build proceeds; all four A rows and their DEV-6 CQADupStack + FORMS-12 reads are reported. A4-A3: if it does not resolve, the build uses A3's corpus and the generated queries are dropped from the build and reported as a measured null. The milestone's failure condition is C1b at M10.4.",
   294	  "G": "resolved winner; default 1152.",
   295	  "B": "resolved winner; default 75/25.",
   296	  "E": "E's decision table is `rules.E_cost`: bs32 iff E1 resolves, else bs128. E is the one family whose throughput is read (amendment A7). **`E-bs128` does not run on the box at realistic sequence lengths and runs on the rented A100** -- Dylan 2026-09-05, the box is preparation for the cloud run, not a measurement target.",
   297	  "C": "resolved winner; default the off-the-shelf backbone. Skipped and reported skipped if F does not select bge-small.",
   298	  "D": "resolved winner with the larger margin; default squared L2.",
   299	  "_default": "an unresolved contrast reverts to its default and is REPORTED as unresolved, beside the measured resolution distance, so a reader can tell an unresolved verdict from an invisible one. **EXCEPTION: family E** -- see `rules.E_cost`; an unresolved E selects bs128, not the default.",
   300	  "anchor": "not a decision family: ANCHOR is the comparator every non-F contrast is read against, and it has no contrast of its own."
   301	 },
   302	 "confirmation": {
   303	  "max_decisions": 2,
   304	  "selection": "decisions whose non-default option won, largest margin first",
   305	  "seeds": [
   306	   1,
   307	   2
   308	  ],
   309	  "margin": "the COV macro difference between the winner's and the default's final checkpoints in the ORIGINAL screen",
   310	  "seed_range": "max minus min of an arm's COV macro over its three seeds",
   311	  "stands_iff": "the winner's margin exceeds the largest seed range observed in either arm",
   312	  "plus": "the synthesized selected-recipe arm (5M) and the replication seed pair (10M)",
   313	  "unconfirmed_non_defaults": "non-default winners beyond the two confirmed REVERT TO DEFAULT and are reported as unconfirmed. This existed only in prose, in a file that declares prose non-authoritative (W9, Codex finding 7)."
   314	 },
   315	 "evaluation": {
   316	  "COV": "at every cycle end",
   317	  "DEV-6": "ONCE, at the final checkpoint, reported beside every COV read; never selection-bearing",
   318	  "FORMS-12": "descriptive only",
   319	  "arxiv-title": "descriptive, no action",
   320	  "CUREv1": "validation-only diagnostic (decision 12); never selection-bearing",
   321	  "LoTTE": "read #1 is decision-bearing (the veto) and happens after the recipe lock; read #2 is the audit after the freeze",
   322	  "family_A_diagnostics": "DEV-6 CQADupStack components and FORMS-12, reported beside every A arm; descriptive, no action (Dylan 2026-09-05, W12)"
   323	 },
   324	 "warm_start": {
   325	  "G-MLP": {
   326	   "n_fit": 60000,
   327	   "seed": 21,
   328	   "lambda": "reselected on the M10 sample by m9/registry.json's locked grid on a training-only holdout, ties to the larger lambda",
   329	   "pca": "per-token, streamed 1152x1152 Gram matrix, each direction's sign fixed so its largest-magnitude component is positive"
   330	  },
   331	  "all_arms": "the closed-form ridge head warm start (M9's m9s1c) is retained in every arm EXCEPT C-M9init, which uses its own head (see arms.C-M9init.head_init)"
   332	 },
   333	 "anchor_aliases": {
   334	  "_what": "the anchor appears in a contrast under the name of the axis being varied; each of these resolves to the single trained ANCHOR arm. Sharing one comparator across ten contrasts does not break Bonferroni (a union bound needs no independence) nor the paired bootstrap (per-contrast); the hazard it does carry -- one unlucky anchor seed lifting every non-default at once -- is what `confirmation` exists to contain.",
   335	  "G-1152": "ANCHOR",
   336	  "B-75/25": "ANCHOR",
   337	  "E-bs32": "ANCHOR",
   338	  "C-fresh": "ANCHOR",
   339	  "anchor": "ANCHOR",
   340	  "A4": "ANCHOR",
   341	  "F-winner": "the winner of contrast F1 at its 20M checkpoint (NOT of F itself: F2 is L12 - F1's winner, so the reference cannot be circular)",
   342	  "_resolver": "every value here is an arm id EXCEPT `F-winner`, which is resolved at read time by `rules.F_selection_aware` and is prose by necessity, not oversight. `runs_iff`/`skipped_iff`/`conditional_extension.iff` are likewise human-readable conditions evaluated by the session, not by a generic engine (W9, Codex finding 9)."
   343	 },
   344	 "trained_arms_expected": 14,
   345	 "_arms_note": "17 entries, 14 TRAINED: A4 is not trained separately -- it IS the ANCHOR arm. C-M9init is CUT under W8 band 1. F-MiniLM-L12 is CUT (Dylan 2026-09-05), taking contrast F2 with it: the Bonferroni denominator is 12.",
   346	 "_amended": "2026-09-05, on a Fable adversarial pass (S1-S10), before any arm ran and with nothing trained. S1 was a structural confound: the anchor was a mid-cycle snapshot of a 20M arm and is now its own 5M arm.",
   347	 "data_cut": {
   348	  "_mandate": "A2, A3 and A4 are cut to the identical post-screen unique-text count",
   349	  "rule": "min of the three post-decontamination unique-text counts; the larger two are downsampled with seed 0; the count itself and the hashes are §0b",
   350	  "applies_to": [
   351	   "A2",
   352	   "A3",
   353	   "A4/ANCHOR"
   354	  ],
   355	  "why": "without it the forms contrast is confounded with volume, which is the one thing family A exists to separate",
   356	  "anchor_corpus": "the CUT A4, not the full corpus -- so the anchor and A3 differ only in which text they carry, never in how much",
   357	  "A1": "not cut: A1 is the M9 pool and A2-A1 is the volume effect by construction",
   358	  "unique_text_count": 2651572,
   359	  "registered": {
   360	   "when": "2026-09-07T14:45-04:00",
   361	   "before_any_arm": true,
   362	   "counts": {
   363	    "A2": 4439407,
   364	    "A3": 2651572,
   365	    "A4": 3486034
   366	   },
   367	   "min": "A3",
   368	   "report": "results/m10_data_cut.json",
   369	   "report_sha256": "d41920dc5e7cd7654681d029bd43adaf986c2c870a08d4f52aceeb6c9ac2a326",
   370	   "generated_corpus": "work/m10gen/generated_queries.jsonl (834,463 rows; results/m10_assemble10.json)",
   371	   "note": "post-screen, post-dedup unique texts through corpus_loader.load_segments on the real sources; A3 is the min because A3 ⊂ A4 and PAQ-A2 (4.037M) > harvest (1.25M)"
   372	  }
   373	 },
   374	 "rules": {
   375	  "generic": "resolves iff point >= MDE AND lower bound at the quantile > 0 AND the sign is stable across the last two cycle-end checkpoints",
   376	  "three_outcome": "A3-A2 only. **NO STOP** (Dylan 2026-09-05, W12: both reviewers -- COV holds none of A3's forms, so a small A3-A2 is expected under a true coverage effect, and the registered STOP would most likely have fired spuriously; the contrast must read above ~0.0142 to resolve at all). Three REPORTED labels, none of which stops M10: lower > MDE -> RESOLVED, coverage established on the COV families; point >= MDE and 0 < lower <= MDE -> POSITIVE, NOT RESOLVED; otherwise -> NOT POSITIVE. The build proceeds in every case and all four A rows are reported with intervals. Family A is additionally reported on DEV-6's two CQADupStack components (M9's actual failure surface) and FORMS-12 (mechanism), both descriptive. The failure condition for M10 is the registered C1b release gate (decision 11), not a screen contrast.",
   377	  "generated_half_in_build": "A4-A3 only. RESOLVED iff lower > MDE (the same corrected bar as three_outcome; the mandate exempts BOTH A contrasts from the generic rule). Not resolved -> the build uses A3's corpus and the ~1.0M generated queries are dropped from the build and reported as a measured null.",
   378	  "sign_stability_exemption": "family A's two contrasts are exempt from the sign-stability clause (mandate); every other contrast is subject to it.",
   379	  "F_orientation": "F1 is read highest-minus-runner-up, i.e. oriented AFTER the reading, so it is TWO-SIDED and takes alpha/24 per tail (`contrasts.F1` carries `tails: 2` and its own `quantile`), which makes the union bound 11(alpha/12) + 2(alpha/24) = alpha EXACTLY. History: at alpha/13 per F tail the familywise rate was 0.02885 (W9 #3).",
   380	  "serve_cost_order": "when F's margin does not resolve, the cheapest to serve wins, ordered by (1) parameter count including the head, (2) layer count, (3) **bge-small, as the retrieval-tuned backbone and the one M9 already distilled into, so the M9-vs-M10 comparison stays within one backbone family**. Measured counts (`results/m10_student_parity_box.json`): MiniLM-L6 23,893,888 < bge-small 34,540,672 = MiniLM-L12 34,540,672 (the mandate's 33.4M for L12 was an estimate; the tie is broken by rule 3, to bge-small). This tie-break is labelled in the report as a product preference, not evidence. **Rule 3's ORIGINAL rationale — 'bge-small alone has an M9 warm start and so alone keeps family C' — went STALE the moment W8 band 1 cut C, and was caught by the W12/W13 validation. The re-stated rationale is not a new preference: it is why the tie was already broken this way, minus the family-C clause.**",
   381	  "E_cost": "**E's decision table, and it is an explicit EXCEPTION to `_default`.** Select **bs32 iff E1 (`bs32 - bs128`) RESOLVES**; in every other case -- unresolved, or bs128 ahead -- select **bs128**. The build's GPU-hour budget assumes bs32, so a resolved bs32 win needs no extra threshold and none is invented after the number; the GPU-hour delta is DISCLOSED beside the contrast. Three places previously disagreed (E's default said bs32, `_default` said unresolved reverts to default, this rule said an unresolved bs32 win reverts to bs128), and under the generic rule a bs128 advantage read NEGATIVE and could not resolve, so the slower arm would have been selected (W9 review #2). **Consequence for confirmation:** selecting bs128 IS a non-default outcome and is therefore eligible for a confirmation slot.",
   382	  "arm_failure": "an arm that crashes, OOMs or trips the §Recipe kill rule has NO final checkpoint: its contrasts are reported UNRESOLVED and revert to default, the failure is reported, and the arm is not silently re-run at different settings.",
   383	  "confirmation_failure": "a confirmation arm that fails leaves its seed range undefined; the decision then reverts to default and is reported as unconfirmed.",
   384	  "confirmation_eligibility": "families G, B, E, C and D only. F and A are EXCLUDED: F confirmation would cost 80M examples against the registered 45M worst case, and A's rules are already three-outcome.",
   385	  "C_skipped": "C-M9init is CUT under W8 band 1; this rule is moot. Had C been merely skippable, the denominator would have stayed fixed -- a contrast's inclusion in the count is decided before any reading, never by whether it happens to run.",
   386	  "D_tie": "if D-NORM and D-COV both resolve with margins within 1e-4, the default (squared L2) stands and the tie is reported.",
   387	  "A2-A1": "DESCRIPTIVE, and it triggers no build action: the build's PAQ volume is fixed unconditionally at 1.0M by §Recipe, whatever A2-A1 reads. (The mandate's 'if it resolves the build keeps volume' would have made a descriptive contrast decisive and outside the 13-way correction; it is resolved here in favour of the fixed cap.)",
   388	  "F_selection_aware": "RETIRED with F2 (Dylan 2026-09-05): it governed F2's adaptively-selected comparator `F-winner`. Kept as a record: a bootstrap that holds an observed winner fixed ignores the selection step (W9, Codex finding 4). No live contrast uses it.",
   389	  "multi_arm_winner": "G, B and D each offer more than one SELECTABLE alternative against the default. **Among the alternatives whose own registered default comparison RESOLVES, the highest point estimate is selected; NO inferential claim is made that it beats the other alternatives, and no alternative-vs-alternative contrast is run.** The earlier form required the winner to 'also resolve against the other alternative', which would have created five post-hoc-oriented comparisons absent from the thirteen and broken the Bonferroni accounting (W9 review #4). Ties on the point estimate go to the default. `D_tie` remains the special case.",
   390	  "F_tournament": "**How F picks the student.** Read at 20M, F1 only (L12 and F2 cut, Dylan 2026-09-05). The winner is the arm with the higher COV macro; it is adopted as RESOLVED only if its margin over the other resolves under the two-sided F rule at alpha/24 per tail. If not, the cheapest to serve wins by `rules.serve_cost_order` and the report labels that a PRODUCT PREFERENCE, not evidence."
   391	 },
   392	 "parity_preconditions": {
   393	  "_rule": "a head that fails serving parity disqualifies its arm (§Screen)",
   394	  "artifact": "results/m10_student_parity_box.json",
   395	  "status": "all six heads PASS: bge-small, MiniLM-L6, MiniLM-L12 at 3 and 4 layers, min-cos >= 0.99999988 through fastembed, zero custom ONNX ops, all under the 35M cap",
   396	  "serving_note": "MiniLM's own `tokenizer_config.json` carries max_length 128 beside model_max_length 512, and fastembed serves min(...) = 128. The exported tokenizer sets both to 512. A first run read 0.93-0.95 min-cos and would have disqualified both MiniLM students on a mis-specified check."
   397	 },
   398	 "_mandate_arm_count": "this file registers 16 trained arms; `instructions-m10.md`:597 still says fifteen. THIS FILE IS AUTHORITATIVE (see `_what`). The mandate predates the S1 amendment that made the anchor its own trained arm (W9, Codex finding 10).",
   399	 "_w8_band": "W8 band 1 (M10.0-e `lr_pair.distance_raw` = 0.00288 <= 0.0056): run F, ANCHOR, A, G, B, E, D; CUT C. `E-bs128` runs on the rented A100, not the box (Dylan 2026-09-05). | Dylan 2026-09-05: L12 cut (F1 only); family A's STOP removed; the weekend timeline is not binding, the box runs as much as it can first and the remainder goes to the cloud under the same registry.",
   400	 "_cut_contrasts": {
   401	  "F2": {
   402	   "a": "F-MiniLM-L12",
   403	   "b": "F-winner",
   404	   "at_examples": 20000000,
   405	   "decisive": true,
   406	   "runs_iff": "L12 extended",
   407	   "note": "counted in the thirteen whether or not it runs",
   408	   "tails": 2,
   409	   "quantile": 0.0009615384615384616,
   410	   "rule": "F_selection_aware",
   411	   "family_rule": "F_tournament",
   412	   "cut": "CUT with its arm, Dylan 2026-09-05 (verification-pass rulings, pre-observation: nothing registered has trained). Was counted in the thirteen whether or not it ran; the denominator is now 12."
   413	  }
   414	 }
   415	}

exec
/bin/bash -lc 'nl -ba m9/FINDINGS.md
nl -ba m10/STATUS.md' in /home/dylan/asymetric-dual-encoders
 succeeded in 0ms:
     1	# M9 findings — the durable record
     2	
     3	**M9 closed 2026-09-01 as a MEASUREMENT (Dylan).** One candidate was built and frozen
     4	(`m9/FREEZE.json`, sha `9d631b2c…`); it misses both bars on every projection, so it is not released.
     5	No six-set, reserved, or LoTTE access was spent. The registered six-set transaction
     6	(`m9/FINAL_LOCK.md`) remains to be executed on the frozen candidate as the close-out measurement,
     7	which gives M10 the dev→six calibration M9 never had. Numbers live in `results/m9_*.json` and
     8	`m9/RESULTS.md`; this file is the reading of them.
     9	
    10	## 1. The question M9 asked, and the answer
    11	
    12	**Can a ≤35M transformer, distilled by plain L2 regression into stella-400M's query space, retain
    13	enough of the teacher (≥87.8% avg-6 for the release bar, ≥89.7% for the aim) to beat bge-small
    14	and the LEAF asymmetric system while serving zero's frozen index?**
    15	
    16	**Not with M9's data.** The build reached SCREEN-3 **0.5606 = 82.2%** of the 0.6822 ceiling at
    17	3.74B tokens and had been flat for 1.1B tokens at constant LR; the anneal added +0.004. The macro
    18	hides the finding:
    19	
    20	| SCREEN-3 component | ceiling | candidate | retention |
    21	|---|---|---|---|
    22	| nq-250k (Wikipedia QA; NQ-adjacent data is in the training mix) | 0.8839 | 0.8289 | **93.8%** |
    23	| cqadup-physics (forum questions) | 0.4931 | 0.3501 | **71.0%** |
    24	| cqadup-programmers (forum questions) | 0.4681 | 0.2345 | **50.1%** |
    25	
    26	Where the training queries look like the test queries, nano is already inside LEAF's band (LEAF
    27	retains 97.9% of its teacher on our six; 97.7% on BEIR-14 in its paper). Where they do not, it retains half. **This is a coverage failure, not a
    28	capacity failure**: a 33M student that reaches 94% on one distribution has the capacity to reach
    29	94%. The six are scientific claims, biomedical questions, finance questions, 170-word
    30	counter-arguments, paper titles and COVID queries — none of which the pool contains — so the
    31	avg-6 retention is expected below the SCREEN-3 macro, not above it.
    32	
    33	## 2. What is closed, and what closed it
    34	
    35	`m9/EXPLORED.md` is the register with reopening conditions. Headline reading:
    36	
    37	| hypothesis | probe | result | reading |
    38	|---|---|---|---|
    39	| More dose on the same pool | screen anchor + build curve | anchor 73% at 60M tokens (separate 16-epoch run); build 68.5% at 0.12B → 79% at 0.74–0.86B → 82% at 3.7B, then flat for 1.1B | 30× the build's first-eval dose bought +14 points, then nothing; the pool is exhausted, not the student |
    40	| Student choice | `m9s4` | MiniLM-L6 −0.0026 DEV-6 vs bge-small, unresolved | a tie at screen dose; MiniLM is 2× cheaper to serve and to train |
    41	| Instruction template on the student | `m9s5` | −0.0204, resolved | the student gets raw query bytes; the teacher keeps its s2p template |
    42	| Documents as regression text | `m9s6` + build | neutral at equal budget (−0.0060 DEV-6); as *extra* text they beat 16 epochs of query repetition (0.545 at 8 query-epochs vs the anchor's 0.500 at 16) | text volume and breadth help; documents are the cheap form of breadth, not a substitute for query forms |
    43	| A bigger teacher | `m9s2` (voided artifact, log read) | stella-1.5B −0.0023 at every checkpoint | M7's "select on the artifact" lesson holds for towers too |
    44	| Head initialization | `m9s1c` | random head −0.0272 at equal SGD dose | warm-start the head in closed form; 8 s of ridge is 5× the decision threshold |
    45	| Seed | `m9s1b` | 0.0008–0.0023 | below threshold; with a closed-form head there is no random init left |
    46	| fp16 target cache | gate | min-cos 0.99996 | fp16 targets are safe |
    47	| ONNX export, document model included | pilots | zero custom ops, parity 0.9999994 (student) / 0.9999994 (stella) | the M11 port risk is retired |
    48	| fastembed serving | pilot | exact (6e-08) once the linear head is exported per token and fastembed pools | fastembed has no slot for a post-pooling head; linearity of mean pooling is what makes the trick exact — a **nonlinear** head cannot ship this way |
    49	| Edge deployability | Mac rounds 1–4 | 1M×1024 index serves in a 256 MB container only binary-quantized (3.4 ms zero / 4.5 ms nano); fp16 is 100× slower at every limit | quantization is a precondition, not an optimisation |
    50	
    51	## 3. Mechanisms worth carrying into M10
    52	
    53	1. **Retention is a per-distribution quantity.** Report it per component, never only as a macro,
    54	   and put the out-of-distribution components in the headline. M9 would have diagnosed itself at
    55	   eval 8 had the curve watch been per component.
    56	2. **The query pool is the lever the recipe never touched.** 463K real queries, all Wikipedia QA
    57	   or product search, longest 108 words. LEAF's ~1.8M query-like texts spanned web search, product
    58	   QA, biomedical QA and forums. The teacher supplies a target for any text, so query-form breadth
    59	   costs generation, not labelling.
    60	3. **A plateau rule that compares two single evaluations fires on noise.** Per-eval noise ~0.005,
    61	   threshold +0.001 over 1.1B tokens: it fired while the best-so-far was still rising. Use
    62	   best-to-best or a smoothed curve, and read annealed checkpoints — the constant-LR curve cannot
    63	   see the anneal (+0.004 here).
    64	4. **Phase 2 must be specified at lock or it does not exist.** The mandate required one fully
    65	   specified ranking-aware loss and its trigger; the lock recorded "out of scope" and the flat
    66	   curve then had no registered response. Specifying it after seeing the curve was correctly
    67	   refused.
    68	5. **Closed-form probes before chains** (M8 §4.1) paid again: the head probe (12 s) changed every
    69	   arm's initialization; the mix decomposition read the m9s6 verdict without another run.
    70	6. **Protocol-scope guards and protocol logging collide during a build.** Editing the ledger
    71	   silently disabled unattended crash-restart. The build-period log (`m9/BUILD_LOG.md`) is the
    72	   pattern: notes in an unscoped file, merged after the run.
    73	7. **A frontier ratio has no meaning without its index configuration** (1.11×–3.28× across mmap /
    74	   RAM / int8 / binary / memory-limited), and **the zero-compute model is the larger artifact**
    75	   (270 MB table vs 46 MB tower). The whitepaper's frontier must name the index it was measured on.
    76	
    77	## 4. What M9 leaves M10
    78	
    79	- **Frozen candidate** `work/m9long/ckpt/last.pt` (sha `9d631b2c…`, 33.4M, bge-small × stella
    80	  space, 3.74B tokens) — a warm-start init already at 94% on NQ, and the subject of the pending
    81	  six-set close-out.
    82	- **Infrastructure that works**: `m9src/longrun.py` (resumable trainer, kill envelope, cooldown),
    83	  `watchdog.py` + `guardian.sh` + `sentinel.sh` (four-layer supervision), `final9.py` (access state
    84	  machine; scoring path still unwritten), `final_stats.py` (16/16 tests), the fp16 target cache,
    85	  the ONNX/fastembed export path including the document model.
    86	- **Protocol state**: reserved four unspent; LoTTE-clean unread (M9's registered read #1 is
    87	  withdrawn unexecuted so M10 inherits a fresh surface — decision 2026-09-01, see `m9/STATUS.md`);
    88	  `results/perquery.json` intact (sha `6b18e3dd…`); dev reuse 494 at M8 close plus M9's screen
    89	  checkpoints and 32 build evaluations (`m8src/dev_reuse_m8.py` gives the exact count).
    90	- **Diagnostics for M10's design**: the rank-bottleneck probe ran on the Mac 2026-09-01
    91	  (`m10/PLANNING.md` §9): the reconstruction-optimal 384-d output subspace serves one query
    92	  distribution at 99.5% and three at 90–93%, strong evidence that M9's 384-wide linear head bound
    93	  it under L2 regression — the second cause of the miss beside coverage. Still to run on the box: the per-component DEV-6 read incl. `heldout-longq`; the capacity
    94	  probe is optional and report-only now that the 35M cap is hard (Dylan, 2026-09-01).
    95	- **A recommendation**: build M10's recipe around coverage first (synthetic queries in every form
    96	  the six use and beyond, FineWeb breadth), then LEAF's optimizer regime (small batch, cyclic
    97	  anneals), with a ranking-aware phase-2 loss registered at lock rather than symptom-gated.
     1	# M10 status — 2026-09-08. **Family F DONE and F1 RESOLVED: the student is bge-small.** **W8 band-1 remainder RUNNING** (11 arms, ~25 h, `work/m10arms_run_rest.sh`). §0b registered; the screen is live.
     2	
     3	**Read this, then `m10/LEDGER.md`.** The box is **preparation for the cloud GPU run, not a
     4	measurement target** (Dylan, 2026-09-05): run as much as it can here first, no re-shaping; the
     5	remainder moves to the A100 under the same registry. **The weekend timeline is not binding.**
     6	Working model (Dylan, 2026-09-05): the session is the ML lead, Opus/Sonnet subagents do the build
     7	work, Codex is the adversarial reviewer.
     8	
     9	## RUNNING: `work/m10arms_run_F.sh` → F-bge-small then F-MiniLM-L6, under `m10src/memtrace.sh`
    10	
    11	**TRAINING CONFIRMED 19:53** — `step 12500/625000 loss 0.3898 lr 9.54e-05 628 ex/s`, the first
    12	arm to get past data prep after three crashes. **628 ex/s measured in the real arm**, 1.23× the
    13	registered `PLAN_RATES[32] = 512`, so **~8.8 h each**, not 10.9 (and not the old 6.2, which read
    14	the 890 ex/s `fixed_bucket_compile` row — `--compile` is SMOKE-ONLY for registered arms).
    15	`PLAN_RATES` is deliberately NOT changed mid-run: the W8 band arithmetic reads it, and 512 being
    16	conservative costs nothing. Re-derive it at the next planning pass. Kill by PID only.
    17	`--resume` needs a parseable non-terminal record AND the rolling checkpoint — it never falls back
    18	to a fresh run, and the arm dir was empty after the crashes, so this is a clean fresh start.
    19	
    20	**Memory: RESOLVED and VALIDATED (2026-09-07).** Two causes: document targets materialized
    21	(19.1 GiB at 5M → `DocTargetView`) and ~3 GB of glibc arena residue in the doc-id stream
    22	(→ `release_arena()`). **Full-dose pre-flight at the registered `n_docs=5,000,000` PASSES**:
    23	peak arm RSS **8,228 MB** (was 24,383), peak guest used **9,799** (cap 26,624), and the metric that
    24	actually failed — **min host free 3,297 MB, against 239 MB pre-fix**. LEDGER has the numbers, four
    25	withdrawn claims of mine, and four open non-blocking items.
    26	
    27	**If it crashes again, read this first:** the peak is dominated by `m9src/data.row_texts`, not the
    28	tokenizer — ten rows cost 2,200 MB `RssAnon` because it materializes a whole store via
    29	`mix.load_store`, once per chunk. Left unfixed on purpose (guard9 "train"/"eval" scope, M9's
    30	close-out pending); the M10-side fix is to sort the draw by store and load each once.
    31	Trust `arm_hwm_mb` in the trace, not `arm_rss_mb` — a 5 s sample of an instantaneous gauge is a
    32	lower bound. `n_match=0` prints **-1**, so a bad pattern cannot look like an idle arm.
    33	
    34	## RUNNING: the W8 band-1 remainder, 11 arms in registered order (~25 h)
    35	
    36	`work/m10arms_run_rest.sh` (tracked copy `m10src/run_rest.sh`), launched 10:54 after **12/12 on the
    37	90-step CUDA shape smoke with the current code** (`results/m10_arm_smoke.json`, `all_shapes_pass`).
    38	Order is the registry's: ANCHOR · A1 · A2 · A3 · G-384 · G-1536 · G-MLP · B-100/0 · B-50/50 ·
    39	D-NORM · D-COV. **C is CUT; E-bs128 is CLOUD_ONLY** (the A100) and `run_arm` refuses it on the box.
    40	
    41	Doses are NOT uniform — read them from `run_arm.py <arm> --plan`, not from the screen dose:
    42	5,000,000 for ANCHOR/A/G/D (156,250 steps), **3,750,000 for B-100/0**, **7,500,000 for B-50/50**.
    43	Projected 52.2 box hours for the whole band at the conservative 512 ex/s, of which F's 21.7 h are
    44	already spent; the remainder is ~30 h projected, ~25 h at the measured 624.
    45	
    46	**`results/m10_F_verdict.json` is what unblocks these arms.** `run_arm.f_verdict` refuses every
    47	post-F arm without it — *"this runner never guesses the student"* — and validates it hard: the
    48	schema (`winner`, `contrast{rule,point,lower,resolved}`, `registry_sha256`,
    49	`sha256_of_F_records`), that it was decided under the CURRENT registry hash, that each F record
    50	hashes exactly, that each has `status: "complete"` and a `final_checkpoint_sha256`, and that the
    51	winner is the student of a `trained`, uncut family-F arm. Writing the contrast to
    52	`m10_contrast_F1.json` alone left all 11 arms refused in 14 s.
    53	
    54	**The chain distinguishes a refusal from a failure:** non-zero within 120 s = a shared
    55	precondition, so it STOPS; a later non-zero is that arm's own outcome and the chain continues. The
    56	first version lacked this and burned all 11 arms on one missing file.
    57	
    58	## F chain — arm 1 done, arm 2 in flight
    59	
    60	| | F-bge-small | F-MiniLM-L6 |
    61	|---|---|---|
    62	| status | **complete, `exit 0`** 04:41:25 | running, ~60% at 08:12 |
    63	| rate | 624 ex/s, 33,298 s (9.25 h) | **1,066 ex/s** (6 layers vs 12), ETA ~09:55 |
    64	| `stopped` | **`plateau at cycle 3`** = registered SUCCESS (`_stopped_note`: PLATEAU_FROM_CYCLE is 3 and a screen arm runs 3 cycles, so it finished its dose and stopped one step short). `complete: true` | — |
    65	| COV cycle ends | 0.5080 → 0.5137 → **0.5160** (+0.0057, +0.0023) | 0.4923 (cycle 1) |
    66	| artifacts | `results/m10_arm_F-bge-small.json`, `work/m10arms/F-bge-small/{record.json,ckpt.pt,cov_*.json}` | — |
    67	
    68	**No verdict exists yet and none may be inferred here.** `f_verdict` is `null` and `_contrasts`
    69	says *"NOT computed here — the contrast step reads the per-query COV files and passes each
    70	contrast's own registered quantile"*. The raw macro gap at the four matching read points is
    71	**0.013–0.016 in bge-small's favour** (0.4871/0.4889/0.4923/0.4942 vs 0.4996/0.5048/0.5080/0.5075),
    72	wider than the 0.0086 resolution distance — but that is a **raw difference, not the registered
    73	statistic**, and the contrast is a separate step over the per-query files.
    74	
    75	**DEV-6 ran once at the final checkpoint** as registered (`nq-250k`, `hotpotqa`,
    76	`cqadup-programmers`, `cqadup-physics`, `heldout-train`, `heldout-longq`) and is **never
    77	selection-bearing** — it informs nothing here, and its CQADupStack components are exactly where
    78	M9's coverage failure showed, which makes it tempting to read as a verdict. It is not one.
    79	
    80	## What is DONE and verified
    81	
    82	| step | outcome |
    83	|---|---|
    84	| **0a/0b/1/2/3** | vLLM + generator smoke; all seven generated forms approved; four COV families admitted (13,416 queries); §0a design lock |
    85	| **4 harvest** | **A3 corpus = 1,250,000 rows** from 21,087,043 harvested (wiki 16.06M · arXiv 4.98M · pool 46.6K). title/keyword/claim at 417K/417K/416K, no form short. FORMS-12 hold-out applied by DOCUMENT across all forms: **1,500 docs held → 1,614 eval rows, 1,248,386 train rows** (`work/m10harvest/harvest_{train,forms12}.jsonl`) |
    86	| **5 PAQ** | **A2 4,037,000** (`8f32bcdf…`) + **build 1,000,000** nested inside it, from `dl.fbaipublicfiles.com`, CC BY-SA 3.0 shipped in the tarball. Protected screen removed only 0.35% |
    87	| **6 trainer** | `nano10` · `data10` · `trainer10` · `qfilter` · `corpus10` · `arm_smoke` · `screen_lock` · **`corpus_loader` · `targets10` (built 2026-09-05 evening, Codex review pending)**. **190 tests green.** `assemble_arm` is the only launcher path. M9 pools re-screened against the M10 protected index (709 queries / 79,630 docs removed). Arm-shape smoke **12/12 on CUDA at max_len 512** |
    88	| **M10.0-e** | **COMPLETE.** P0 0.477528 · P1 0.476141 · P2 0.473892. **Same-init distance 0.00288**, seed effect 0.00139 (n=1). `results/m10_calib_report.json` |
    89	| **seeds** | resolved for **all seven** generated forms; health/finance 33,000 each from `wikipedia-body`, howto 37,927, the other four route `"general"` |
    90	
    91	## OPEN — needs Dylan (none blocks the next step; W12/W13 ruled 2026-09-05, kept one line each)
    92	
    93	| # | what |
    94	|---|---|
    95	| ~~W12~~ | **RULED: STOP removed.** Family A reports three labels and its DEV-6 CQADupStack + FORMS-12 reads; C1b is the failure condition (`instructions-m10.md` §Amendment 2026-09-05 C1) |
    96	| ~~W13~~ | **RULED: L12 CUT.** F = bge-small vs MiniLM-L6 at 20M, 12 contrasts, 14 trained arms (§Amendment 2026-09-05 C2) |
    97	| **paired row** | **DRAFT REGISTRATION READY for ratification: `research/m10-paired-row-registration-draft.md`** (8 numbered items, drafted from both reviewer passes, nothing new added). **Item 7 expires first** — both six-set transactions must emit aligned PER-QUERY scores and qids, and that must be in force BEFORE M9's close-out runs or the row cannot be built at all, ever. Original: **Register it — as a whole-protocol RECIPE delta, not a coverage test.** M9→M10 changes dose 4.5× (3.69B → ≈16.8B), head width 384→1152, schedule, objective, mix, batch, and possibly the student. **Normalise on the TEACHER**, not M9 (`Δr_d = (S10−S9)/T_d`); never call `S10/S9` "retention". Register now: the confound list, a fixed non-causal claim sentence, the exact datasets/statistic/B/seed, a conditional "same backbone family" label (F can pick a MiniLM), and **that both six-set transactions emit aligned PER-QUERY scores and qids** — otherwise the paired row cannot be built at all |
    98	| **W14** | **No decision-bearing surface sees the headline forms.** COV selects on forum/medical/legal/finance; C1b is clean-4 (scientific, biomedical). The screen can optimise the build away from the release bar and nothing would notice until M10.4. Codex frames the root question: **is family A a causal experiment, a catastrophe veto, or diagnostics? It cannot be all three** |
    99	
   100	## gpt-6-astra whole-plan review, 2026-09-08 — 3 verified gaps, all on the COST axis
   101	
   102	Dispositions and verification: `research/m10-astra-review-dispositions-2026-09-08.md`; verbatim
   103	`research/m10-astra-whole-plan-2026-09-08.log`. It did NOT rediscover W14 or the frozen-tower
   104	debate — the brief withheld them, and it went elsewhere.
   105	
   106	| # | finding | status | owner |
   107	|---|---|---|---|
   108	| 1 | **No small-model + BM25 row exists.** `zero`+BM25 = 0.4911 is compared against OpenSearch, LR-hybrid and dense-only baselines, never against **bge-small + BM25** — the obvious hybrid of the release bar. Not recoverable from `perquery.json` (it holds per-query nDCG *values*, not score lists) | **VERIFIED** in `FINAL_MATRIX.md` | **Dylan** — adding a comparator post-hoc, but it can only WEAKEN our claim, so conservative |
   109	| 2 | **nano buys no compute saving.** 34,540,672 params vs bge-small's ~33.4M, same backbone forward pass. nano's claim is *quality at equal edge cost sharing stella's index*, NOT "near-zero query compute" — that is `zero`'s claim alone | **VERIFIED** from our own counts | session — a CLAIM fix owed to `CLAUDE.md`'s north star, M14 and nano's card |
   110	| 3 | **The cost axis is unmeasured for the two points that matter.** `m9_edge_cost_Apple_M5_Pro.json` profiles cold start, peak RSS and p50/p95 for nano variants + mdbr-leaf-ir only; `zero` itself and conventional bge-small have no such profile under one harness | **VERIFIED** | session — **do it**, Mac-only, no labels, no GPU, no protocol impact |
   111	| 4 | **Compile queries into the INDEX, not a student** — our 0.83M generated queries carry source doc IDs; index them as lexical aliases (doc2query-shaped). No student at all. Trades index expansion for query-model complexity | new avenue | **Dylan** to scope; needs the contamination rule re-applied to index content |
   112	| 5 | **Spearman ≈ 0 may mean compressibility is TRAINABLE**, not that teachers are interchangeable — no independently-trained tower was trained to be compressible. Reframes M14's headline; connects to M8's unrun `E14-LORA` | framing | **Dylan** / M14 |
   113	
   114	**Its #1 claim is wrong** — it assumed the index needs a query-time server. It does not: the edge
   115	client holds the index. The salvageable half is to state whether the requirement is near-zero
   116	**edge** or **total** compute, which finding 2 makes load-bearing.
   117	
   118	**The through-line:** *"gate further training spend on finding an operational region where
   119	asymmetry wins."* We have measured quality carefully and cost loosely.
   120	
   121	## NEXT, in order — rewritten 2026-09-08; three items on the old list were already DONE
   122	
   123	**DONE, removed from this list** (they were still listed as pending and cost a re-read):
   124	corpus→trainer path · M10.0-c baseline · generation (finished 09-05/06, `work/m10gen/*.jsonl`;
   125	ANCHOR consumed 834,463 generated rows) · `data_cut.unique_text_count` = **2,651,572**, registered ·
   126	**A8 gate 2 re-run 09-07 across all 12 forms** (`results/m10_a8_gate2.json`) · family F, and F1
   127	resolved to **bge-small**.
   128	
   129	| # | open item | blocks | who |
   130	|---|---|---|---|
   131	| 1 | **The W8 band-1 remainder is RUNNING** — 3 of 11 done (ANCHOR · A1 · A2, all `exit 0`, avg 2.39 h vs 2.71 h projected). Then the registered contrasts per family | the M10.2 recipe lock | in flight |
   132	| 2 | **CUREv1 admission** (decision 12, adopted 2026-09-04, **never executed**). Its own precondition: the harvest, PAQ and seed draws were screened against an index lacking it — **re-screen or disclose before reading it** | nothing on the critical path; it is a reported diagnostic, never selection-bearing | session |
   133	| 3 | **`results/m10_data_manifest.json`** — still MISSING (§1 owes it) | the recipe lock's provenance | session |
   134	| 4 | **Own-source word-5-gram screen** on the harvest corpus (W11's second half): run it if a pass over the sources is cheap, else **disclose it as not run** in §1 (near-vacuous by construction) | a disclosure in the report | session |
   135	| 5 | **Register the paired M9-vs-M10 row** — **draft ready for ratification**, `research/m10-paired-row-registration-draft.md`. Its only conditional item (§5, backbone family) self-resolved to *within one family* when F picked bge-small. **§7 expires first:** both six-set transactions must emit aligned per-query scores AND qids before M9's close-out runs, or the row is unbuildable forever | M9's close-out | **Dylan** |
   136	| 6 | **W14** — is family A a causal experiment, a catastrophe veto, or diagnostics? *"It cannot be all three."* Note its first option NARROWED: a clean-4 surface can no longer be added to COV, which is now observed | the M10.2 recipe lock. Tier 3 | **Dylan** |
   137	| 7 | **`torch.compile` for registered arms** (~1.7×). Currently SMOKE-ONLY | screen wall-clock only | **Dylan** |
   138	
   139	## Screen design, settled
   140	
   141	**W8 band 1** (P2 = 0.00288 ≤ 0.0056): run F · ANCHOR · A · G · B · E · D, **cut C** — `C-M9init`
   142	at 5M starts from 3.69B tokens and wins its own contrast by construction. **L12 cut, A's STOP
   143	removed** (2026-09-05). Both E arms run on the **A100**, together.
   144	
   145	## Hazards a cold session will hit
   146	
   147	- **`E-bs128` fails on this box** at max_len ≥ 256 (`CUDA driver error: device not ready`)
   148	  on three morning runs, then PASSED at 512 on the evening re-run — intermittent, not deterministic. Passes at 128 (2,188 ex/s). **Accepted, not worked around** — do
   149	  NOT add gradient accumulation. The 1,517 ex/s in the rate table is a random-token microbenchmark.
   150	- **`arm_smoke` reaches ~8 GB RSS**; torch's CPU allocator does not release across 12 model loads.
   151	  Run it alone. **Kill by PID** — `pkill -f arm_smoke` matches your own waiting shell.
   152	- **Smokes overwrite real artifacts.** `calib.run_arm` writes the real `P0.json`; `harvest.draw`
   153	  wrote the real `harvest_draw.json` until `out_dir` was added. Check `total_steps`/`_partial`.
   154	- **The system python has no numpy.** Use `.venv/bin/python`.
   155	- **`m10src/forms.RUBRIC` is the frozen gate standard; `forms.FORMS` is the revisable prompt.**
   156	- Never overwrite `results/perquery.json`. No six/reserved/LoTTE read outside a registered
   157	  transaction. Every review brief carries the reserved read-exclusion; **audit the log afterwards**.
   158	
   159	## Do NOT redo
   160	
   161	- Do not move the seed-precision bar (W6: there is no bar). Do not re-weight or drop BRIGHT
   162	  (LEDGER §5 — **legal** is the uninformative family, 32.5% of variance for 12% of signal).
   163	  Do not run T2-8's rungs 2/3 (demoted to diagnostics). Do not treat the on-form diagnostic as an
   164	  admission instrument. There is **no post-generation admission test** — only A8's manifest gates
   165	  and the FORMS-12 hold-out.
   166	

exec
/bin/bash -lc "nl -ba m10/screen_registry.json | sed -n '330,415p'
nl -ba m9/FINDINGS.md
nl -ba CLAUDE.md | sed -n '1,130p'" in /home/dylan/asymetric-dual-encoders
 succeeded in 0ms:
   330	  },
   331	  "all_arms": "the closed-form ridge head warm start (M9's m9s1c) is retained in every arm EXCEPT C-M9init, which uses its own head (see arms.C-M9init.head_init)"
   332	 },
   333	 "anchor_aliases": {
   334	  "_what": "the anchor appears in a contrast under the name of the axis being varied; each of these resolves to the single trained ANCHOR arm. Sharing one comparator across ten contrasts does not break Bonferroni (a union bound needs no independence) nor the paired bootstrap (per-contrast); the hazard it does carry -- one unlucky anchor seed lifting every non-default at once -- is what `confirmation` exists to contain.",
   335	  "G-1152": "ANCHOR",
   336	  "B-75/25": "ANCHOR",
   337	  "E-bs32": "ANCHOR",
   338	  "C-fresh": "ANCHOR",
   339	  "anchor": "ANCHOR",
   340	  "A4": "ANCHOR",
   341	  "F-winner": "the winner of contrast F1 at its 20M checkpoint (NOT of F itself: F2 is L12 - F1's winner, so the reference cannot be circular)",
   342	  "_resolver": "every value here is an arm id EXCEPT `F-winner`, which is resolved at read time by `rules.F_selection_aware` and is prose by necessity, not oversight. `runs_iff`/`skipped_iff`/`conditional_extension.iff` are likewise human-readable conditions evaluated by the session, not by a generic engine (W9, Codex finding 9)."
   343	 },
   344	 "trained_arms_expected": 14,
   345	 "_arms_note": "17 entries, 14 TRAINED: A4 is not trained separately -- it IS the ANCHOR arm. C-M9init is CUT under W8 band 1. F-MiniLM-L12 is CUT (Dylan 2026-09-05), taking contrast F2 with it: the Bonferroni denominator is 12.",
   346	 "_amended": "2026-09-05, on a Fable adversarial pass (S1-S10), before any arm ran and with nothing trained. S1 was a structural confound: the anchor was a mid-cycle snapshot of a 20M arm and is now its own 5M arm.",
   347	 "data_cut": {
   348	  "_mandate": "A2, A3 and A4 are cut to the identical post-screen unique-text count",
   349	  "rule": "min of the three post-decontamination unique-text counts; the larger two are downsampled with seed 0; the count itself and the hashes are §0b",
   350	  "applies_to": [
   351	   "A2",
   352	   "A3",
   353	   "A4/ANCHOR"
   354	  ],
   355	  "why": "without it the forms contrast is confounded with volume, which is the one thing family A exists to separate",
   356	  "anchor_corpus": "the CUT A4, not the full corpus -- so the anchor and A3 differ only in which text they carry, never in how much",
   357	  "A1": "not cut: A1 is the M9 pool and A2-A1 is the volume effect by construction",
   358	  "unique_text_count": 2651572,
   359	  "registered": {
   360	   "when": "2026-09-07T14:45-04:00",
   361	   "before_any_arm": true,
   362	   "counts": {
   363	    "A2": 4439407,
   364	    "A3": 2651572,
   365	    "A4": 3486034
   366	   },
   367	   "min": "A3",
   368	   "report": "results/m10_data_cut.json",
   369	   "report_sha256": "d41920dc5e7cd7654681d029bd43adaf986c2c870a08d4f52aceeb6c9ac2a326",
   370	   "generated_corpus": "work/m10gen/generated_queries.jsonl (834,463 rows; results/m10_assemble10.json)",
   371	   "note": "post-screen, post-dedup unique texts through corpus_loader.load_segments on the real sources; A3 is the min because A3 ⊂ A4 and PAQ-A2 (4.037M) > harvest (1.25M)"
   372	  }
   373	 },
   374	 "rules": {
   375	  "generic": "resolves iff point >= MDE AND lower bound at the quantile > 0 AND the sign is stable across the last two cycle-end checkpoints",
   376	  "three_outcome": "A3-A2 only. **NO STOP** (Dylan 2026-09-05, W12: both reviewers -- COV holds none of A3's forms, so a small A3-A2 is expected under a true coverage effect, and the registered STOP would most likely have fired spuriously; the contrast must read above ~0.0142 to resolve at all). Three REPORTED labels, none of which stops M10: lower > MDE -> RESOLVED, coverage established on the COV families; point >= MDE and 0 < lower <= MDE -> POSITIVE, NOT RESOLVED; otherwise -> NOT POSITIVE. The build proceeds in every case and all four A rows are reported with intervals. Family A is additionally reported on DEV-6's two CQADupStack components (M9's actual failure surface) and FORMS-12 (mechanism), both descriptive. The failure condition for M10 is the registered C1b release gate (decision 11), not a screen contrast.",
   377	  "generated_half_in_build": "A4-A3 only. RESOLVED iff lower > MDE (the same corrected bar as three_outcome; the mandate exempts BOTH A contrasts from the generic rule). Not resolved -> the build uses A3's corpus and the ~1.0M generated queries are dropped from the build and reported as a measured null.",
   378	  "sign_stability_exemption": "family A's two contrasts are exempt from the sign-stability clause (mandate); every other contrast is subject to it.",
   379	  "F_orientation": "F1 is read highest-minus-runner-up, i.e. oriented AFTER the reading, so it is TWO-SIDED and takes alpha/24 per tail (`contrasts.F1` carries `tails: 2` and its own `quantile`), which makes the union bound 11(alpha/12) + 2(alpha/24) = alpha EXACTLY. History: at alpha/13 per F tail the familywise rate was 0.02885 (W9 #3).",
   380	  "serve_cost_order": "when F's margin does not resolve, the cheapest to serve wins, ordered by (1) parameter count including the head, (2) layer count, (3) **bge-small, as the retrieval-tuned backbone and the one M9 already distilled into, so the M9-vs-M10 comparison stays within one backbone family**. Measured counts (`results/m10_student_parity_box.json`): MiniLM-L6 23,893,888 < bge-small 34,540,672 = MiniLM-L12 34,540,672 (the mandate's 33.4M for L12 was an estimate; the tie is broken by rule 3, to bge-small). This tie-break is labelled in the report as a product preference, not evidence. **Rule 3's ORIGINAL rationale — 'bge-small alone has an M9 warm start and so alone keeps family C' — went STALE the moment W8 band 1 cut C, and was caught by the W12/W13 validation. The re-stated rationale is not a new preference: it is why the tie was already broken this way, minus the family-C clause.**",
   381	  "E_cost": "**E's decision table, and it is an explicit EXCEPTION to `_default`.** Select **bs32 iff E1 (`bs32 - bs128`) RESOLVES**; in every other case -- unresolved, or bs128 ahead -- select **bs128**. The build's GPU-hour budget assumes bs32, so a resolved bs32 win needs no extra threshold and none is invented after the number; the GPU-hour delta is DISCLOSED beside the contrast. Three places previously disagreed (E's default said bs32, `_default` said unresolved reverts to default, this rule said an unresolved bs32 win reverts to bs128), and under the generic rule a bs128 advantage read NEGATIVE and could not resolve, so the slower arm would have been selected (W9 review #2). **Consequence for confirmation:** selecting bs128 IS a non-default outcome and is therefore eligible for a confirmation slot.",
   382	  "arm_failure": "an arm that crashes, OOMs or trips the §Recipe kill rule has NO final checkpoint: its contrasts are reported UNRESOLVED and revert to default, the failure is reported, and the arm is not silently re-run at different settings.",
   383	  "confirmation_failure": "a confirmation arm that fails leaves its seed range undefined; the decision then reverts to default and is reported as unconfirmed.",
   384	  "confirmation_eligibility": "families G, B, E, C and D only. F and A are EXCLUDED: F confirmation would cost 80M examples against the registered 45M worst case, and A's rules are already three-outcome.",
   385	  "C_skipped": "C-M9init is CUT under W8 band 1; this rule is moot. Had C been merely skippable, the denominator would have stayed fixed -- a contrast's inclusion in the count is decided before any reading, never by whether it happens to run.",
   386	  "D_tie": "if D-NORM and D-COV both resolve with margins within 1e-4, the default (squared L2) stands and the tie is reported.",
   387	  "A2-A1": "DESCRIPTIVE, and it triggers no build action: the build's PAQ volume is fixed unconditionally at 1.0M by §Recipe, whatever A2-A1 reads. (The mandate's 'if it resolves the build keeps volume' would have made a descriptive contrast decisive and outside the 13-way correction; it is resolved here in favour of the fixed cap.)",
   388	  "F_selection_aware": "RETIRED with F2 (Dylan 2026-09-05): it governed F2's adaptively-selected comparator `F-winner`. Kept as a record: a bootstrap that holds an observed winner fixed ignores the selection step (W9, Codex finding 4). No live contrast uses it.",
   389	  "multi_arm_winner": "G, B and D each offer more than one SELECTABLE alternative against the default. **Among the alternatives whose own registered default comparison RESOLVES, the highest point estimate is selected; NO inferential claim is made that it beats the other alternatives, and no alternative-vs-alternative contrast is run.** The earlier form required the winner to 'also resolve against the other alternative', which would have created five post-hoc-oriented comparisons absent from the thirteen and broken the Bonferroni accounting (W9 review #4). Ties on the point estimate go to the default. `D_tie` remains the special case.",
   390	  "F_tournament": "**How F picks the student.** Read at 20M, F1 only (L12 and F2 cut, Dylan 2026-09-05). The winner is the arm with the higher COV macro; it is adopted as RESOLVED only if its margin over the other resolves under the two-sided F rule at alpha/24 per tail. If not, the cheapest to serve wins by `rules.serve_cost_order` and the report labels that a PRODUCT PREFERENCE, not evidence."
   391	 },
   392	 "parity_preconditions": {
   393	  "_rule": "a head that fails serving parity disqualifies its arm (§Screen)",
   394	  "artifact": "results/m10_student_parity_box.json",
   395	  "status": "all six heads PASS: bge-small, MiniLM-L6, MiniLM-L12 at 3 and 4 layers, min-cos >= 0.99999988 through fastembed, zero custom ONNX ops, all under the 35M cap",
   396	  "serving_note": "MiniLM's own `tokenizer_config.json` carries max_length 128 beside model_max_length 512, and fastembed serves min(...) = 128. The exported tokenizer sets both to 512. A first run read 0.93-0.95 min-cos and would have disqualified both MiniLM students on a mis-specified check."
   397	 },
   398	 "_mandate_arm_count": "this file registers 16 trained arms; `instructions-m10.md`:597 still says fifteen. THIS FILE IS AUTHORITATIVE (see `_what`). The mandate predates the S1 amendment that made the anchor its own trained arm (W9, Codex finding 10).",
   399	 "_w8_band": "W8 band 1 (M10.0-e `lr_pair.distance_raw` = 0.00288 <= 0.0056): run F, ANCHOR, A, G, B, E, D; CUT C. `E-bs128` runs on the rented A100, not the box (Dylan 2026-09-05). | Dylan 2026-09-05: L12 cut (F1 only); family A's STOP removed; the weekend timeline is not binding, the box runs as much as it can first and the remainder goes to the cloud under the same registry.",
   400	 "_cut_contrasts": {
   401	  "F2": {
   402	   "a": "F-MiniLM-L12",
   403	   "b": "F-winner",
   404	   "at_examples": 20000000,
   405	   "decisive": true,
   406	   "runs_iff": "L12 extended",
   407	   "note": "counted in the thirteen whether or not it runs",
   408	   "tails": 2,
   409	   "quantile": 0.0009615384615384616,
   410	   "rule": "F_selection_aware",
   411	   "family_rule": "F_tournament",
   412	   "cut": "CUT with its arm, Dylan 2026-09-05 (verification-pass rulings, pre-observation: nothing registered has trained). Was counted in the thirteen whether or not it ran; the denominator is now 12."
   413	  }
   414	 }
   415	}
     1	# M9 findings — the durable record
     2	
     3	**M9 closed 2026-09-01 as a MEASUREMENT (Dylan).** One candidate was built and frozen
     4	(`m9/FREEZE.json`, sha `9d631b2c…`); it misses both bars on every projection, so it is not released.
     5	No six-set, reserved, or LoTTE access was spent. The registered six-set transaction
     6	(`m9/FINAL_LOCK.md`) remains to be executed on the frozen candidate as the close-out measurement,
     7	which gives M10 the dev→six calibration M9 never had. Numbers live in `results/m9_*.json` and
     8	`m9/RESULTS.md`; this file is the reading of them.
     9	
    10	## 1. The question M9 asked, and the answer
    11	
    12	**Can a ≤35M transformer, distilled by plain L2 regression into stella-400M's query space, retain
    13	enough of the teacher (≥87.8% avg-6 for the release bar, ≥89.7% for the aim) to beat bge-small
    14	and the LEAF asymmetric system while serving zero's frozen index?**
    15	
    16	**Not with M9's data.** The build reached SCREEN-3 **0.5606 = 82.2%** of the 0.6822 ceiling at
    17	3.74B tokens and had been flat for 1.1B tokens at constant LR; the anneal added +0.004. The macro
    18	hides the finding:
    19	
    20	| SCREEN-3 component | ceiling | candidate | retention |
    21	|---|---|---|---|
    22	| nq-250k (Wikipedia QA; NQ-adjacent data is in the training mix) | 0.8839 | 0.8289 | **93.8%** |
    23	| cqadup-physics (forum questions) | 0.4931 | 0.3501 | **71.0%** |
    24	| cqadup-programmers (forum questions) | 0.4681 | 0.2345 | **50.1%** |
    25	
    26	Where the training queries look like the test queries, nano is already inside LEAF's band (LEAF
    27	retains 97.9% of its teacher on our six; 97.7% on BEIR-14 in its paper). Where they do not, it retains half. **This is a coverage failure, not a
    28	capacity failure**: a 33M student that reaches 94% on one distribution has the capacity to reach
    29	94%. The six are scientific claims, biomedical questions, finance questions, 170-word
    30	counter-arguments, paper titles and COVID queries — none of which the pool contains — so the
    31	avg-6 retention is expected below the SCREEN-3 macro, not above it.
    32	
    33	## 2. What is closed, and what closed it
    34	
    35	`m9/EXPLORED.md` is the register with reopening conditions. Headline reading:
    36	
    37	| hypothesis | probe | result | reading |
    38	|---|---|---|---|
    39	| More dose on the same pool | screen anchor + build curve | anchor 73% at 60M tokens (separate 16-epoch run); build 68.5% at 0.12B → 79% at 0.74–0.86B → 82% at 3.7B, then flat for 1.1B | 30× the build's first-eval dose bought +14 points, then nothing; the pool is exhausted, not the student |
    40	| Student choice | `m9s4` | MiniLM-L6 −0.0026 DEV-6 vs bge-small, unresolved | a tie at screen dose; MiniLM is 2× cheaper to serve and to train |
    41	| Instruction template on the student | `m9s5` | −0.0204, resolved | the student gets raw query bytes; the teacher keeps its s2p template |
    42	| Documents as regression text | `m9s6` + build | neutral at equal budget (−0.0060 DEV-6); as *extra* text they beat 16 epochs of query repetition (0.545 at 8 query-epochs vs the anchor's 0.500 at 16) | text volume and breadth help; documents are the cheap form of breadth, not a substitute for query forms |
    43	| A bigger teacher | `m9s2` (voided artifact, log read) | stella-1.5B −0.0023 at every checkpoint | M7's "select on the artifact" lesson holds for towers too |
    44	| Head initialization | `m9s1c` | random head −0.0272 at equal SGD dose | warm-start the head in closed form; 8 s of ridge is 5× the decision threshold |
    45	| Seed | `m9s1b` | 0.0008–0.0023 | below threshold; with a closed-form head there is no random init left |
    46	| fp16 target cache | gate | min-cos 0.99996 | fp16 targets are safe |
    47	| ONNX export, document model included | pilots | zero custom ops, parity 0.9999994 (student) / 0.9999994 (stella) | the M11 port risk is retired |
    48	| fastembed serving | pilot | exact (6e-08) once the linear head is exported per token and fastembed pools | fastembed has no slot for a post-pooling head; linearity of mean pooling is what makes the trick exact — a **nonlinear** head cannot ship this way |
    49	| Edge deployability | Mac rounds 1–4 | 1M×1024 index serves in a 256 MB container only binary-quantized (3.4 ms zero / 4.5 ms nano); fp16 is 100× slower at every limit | quantization is a precondition, not an optimisation |
    50	
    51	## 3. Mechanisms worth carrying into M10
    52	
    53	1. **Retention is a per-distribution quantity.** Report it per component, never only as a macro,
    54	   and put the out-of-distribution components in the headline. M9 would have diagnosed itself at
    55	   eval 8 had the curve watch been per component.
    56	2. **The query pool is the lever the recipe never touched.** 463K real queries, all Wikipedia QA
    57	   or product search, longest 108 words. LEAF's ~1.8M query-like texts spanned web search, product
    58	   QA, biomedical QA and forums. The teacher supplies a target for any text, so query-form breadth
    59	   costs generation, not labelling.
    60	3. **A plateau rule that compares two single evaluations fires on noise.** Per-eval noise ~0.005,
    61	   threshold +0.001 over 1.1B tokens: it fired while the best-so-far was still rising. Use
    62	   best-to-best or a smoothed curve, and read annealed checkpoints — the constant-LR curve cannot
    63	   see the anneal (+0.004 here).
    64	4. **Phase 2 must be specified at lock or it does not exist.** The mandate required one fully
    65	   specified ranking-aware loss and its trigger; the lock recorded "out of scope" and the flat
    66	   curve then had no registered response. Specifying it after seeing the curve was correctly
    67	   refused.
    68	5. **Closed-form probes before chains** (M8 §4.1) paid again: the head probe (12 s) changed every
    69	   arm's initialization; the mix decomposition read the m9s6 verdict without another run.
    70	6. **Protocol-scope guards and protocol logging collide during a build.** Editing the ledger
    71	   silently disabled unattended crash-restart. The build-period log (`m9/BUILD_LOG.md`) is the
    72	   pattern: notes in an unscoped file, merged after the run.
    73	7. **A frontier ratio has no meaning without its index configuration** (1.11×–3.28× across mmap /
    74	   RAM / int8 / binary / memory-limited), and **the zero-compute model is the larger artifact**
    75	   (270 MB table vs 46 MB tower). The whitepaper's frontier must name the index it was measured on.
    76	
    77	## 4. What M9 leaves M10
    78	
    79	- **Frozen candidate** `work/m9long/ckpt/last.pt` (sha `9d631b2c…`, 33.4M, bge-small × stella
    80	  space, 3.74B tokens) — a warm-start init already at 94% on NQ, and the subject of the pending
    81	  six-set close-out.
    82	- **Infrastructure that works**: `m9src/longrun.py` (resumable trainer, kill envelope, cooldown),
    83	  `watchdog.py` + `guardian.sh` + `sentinel.sh` (four-layer supervision), `final9.py` (access state
    84	  machine; scoring path still unwritten), `final_stats.py` (16/16 tests), the fp16 target cache,
    85	  the ONNX/fastembed export path including the document model.
    86	- **Protocol state**: reserved four unspent; LoTTE-clean unread (M9's registered read #1 is
    87	  withdrawn unexecuted so M10 inherits a fresh surface — decision 2026-09-01, see `m9/STATUS.md`);
    88	  `results/perquery.json` intact (sha `6b18e3dd…`); dev reuse 494 at M8 close plus M9's screen
    89	  checkpoints and 32 build evaluations (`m8src/dev_reuse_m8.py` gives the exact count).
    90	- **Diagnostics for M10's design**: the rank-bottleneck probe ran on the Mac 2026-09-01
    91	  (`m10/PLANNING.md` §9): the reconstruction-optimal 384-d output subspace serves one query
    92	  distribution at 99.5% and three at 90–93%, strong evidence that M9's 384-wide linear head bound
    93	  it under L2 regression — the second cause of the miss beside coverage. Still to run on the box: the per-component DEV-6 read incl. `heldout-longq`; the capacity
    94	  probe is optional and report-only now that the 35M cap is hard (Dylan, 2026-09-01).
    95	- **A recommendation**: build M10's recipe around coverage first (synthetic queries in every form
    96	  the six use and beyond, FineWeb breadth), then LEAF's optimizer regime (small batch, cyclic
    97	  anneals), with a ranking-aware phase-2 loss registered at lock rather than symptom-gated.
     1	# Asymmetric Dual Encoders — Viability Scoping
     2	
     3	## North star
     4	
     5	Decide, with defensible numbers, whether asymmetric dual encoders (big document encoder in the cloud, near-zero-compute query encoder on the edge) beat the obvious alternative: running a small 20–100M conventional embedding model on the edge. The research question: **how much retrieval quality do you retain as query-side computation approaches zero?**
     6	
     7	Architecture: the server indexes documents once with a large frozen encoder (**stella_en_400M_v5**, 1024d, since M7); the edge client holds the document index plus a query path. Two query paths exist: **zero** (M7, a token→vector lookup table, no transformer) and **nano** (M9/M10, a ≤35M distilled transformer). Both serve the SAME index — the product is the pair, two points on a quality-vs-query-cost frontier.
     8	
     9	## Deliverable
    10	
    11	A decision report (Artifact) with a quality-vs-query-side-cost frontier: nDCG@10 on the six named datasets for each point, plus edge costs (query asset, document index, hydration, CPU latency), a Qdrant Edge prototype of the pair, and the released models. Every headline claim is paired on frozen comparator vectors with pre-registered statistics; the miss-is-publishable framing applies to every milestone.
    12	
    13	## Constraints
    14	
    15	- Mac: Apple M5 Pro, 24 GB RAM, MPS — probes and CPU cost rows only. The Windows/RTX 3080 box ran M7–M9 and holds their gitignored `work/` artifacts; **M10's build runs on a rented cloud GPU under an approved budget (Dylan, 2026-09-01; amended by A7 2026-09-04 — the box runs the screens and the generation). Amended again 2026-09-05: the box is PREPARATION for the cloud run, not a measurement target — an arm that does not fit here moves to the cloud rather than being re-shaped (`E-bs128` is the first)**. Headless either way: Dylan follows on GitHub.
    16	- Quality numbers come from exact search so ANN recall is not a confound; Qdrant Edge is for architecture and latency.
    17	- Dylan is hands-off; decisions get made here and logged. The evaluation protocol (partitions, decontamination, frozen comparators, single final run, pre-registered statistics) may change only BEFORE the numbers it affects are observed, never retroactively.
    18	
    19	## Stage plan
    20	
    21	- [x] M0–M6 (2026-08-24/25): environment, research (`research/landscape.md`, `literature.md`, `methodology.md`, `sparse-inference-free.md`, `lightretriever.md`), the 21-system × 6-dataset matrix (`results/FINAL_MATRIX.md`, costs, significance), the two-collection Edge prototype, and the M6 report artifact (https://claude.ai/code/artifact/db771dd1-2d59-4c34-9d6e-70dd4d337d16). Headline numbers, rerun outcomes, the M2 Codex gate and the M1/M2 findings logs are archived verbatim in `research/m1-m6-findings.md`.
    22	- [x] M7 (closed 2026-08-29): **zero**, the int8 lookup table distilled from stella. Final run avg-6 **0.4339** vs `LR-dense-pertask` 0.4583 (CI-resolved miss); fused with BM25 0.4911 (convex0, a statistical tie with OpenSearch 0.4868 — the registered C3 comparison, unchanged). **M12 re-measured the fused row under a Qdrant-runnable operator: `Fusion.DBSF` at prefetch 100 gives 0.4887 on the six and 0.4912 on clean-4 vs convex0's 0.4911/0.4866 — a TIE on both (inside the ~0.005 band, no CI), so the case is deployability at no measured cost. Recorded as a post-M7 PRODUCT-POLICY OVERRIDE of the `m7/LEDGER.md:691` release freeze, on Dylan's implementability ruling — convex0 stays the M7 operator of record and C1/C2/C3 are untouched.** Frozen and verified releasable (`m7/FREEZE.json`); **published 2026-09-03, PUBLIC, as `constella-zero`** — https://huggingface.co/DylanCouzon/constella-zero (renamed from `zero-query-encoder-v1`; old URL redirects), see `m11/STATUS.md`. Mandate `instructions-m7.md`, detail `m7/STATUS.md`, transferable lessons `m7/FINDINGS.md`. Teacher decision: stella_en_400M_v5, chosen on the distilled TABLE, not the tower (see Key decisions).
    23	- [x] M8 (closed 2026-08-30 as a measurement): twelve probes, no lever moved the table more than 0.005; no confirmatory access spent. `m8/FINDINGS.md`, `m8/EXPLORED.md`.
    24	- [x] M9 (closed 2026-09-01 as a measurement, six-set close-out pending): **nano**, bge-small distilled into stella's query space by L2 regression. Build stopped by the plateau rule at 3.74B tokens: SCREEN-3 0.5606 = **82.2% retention**, but **93.8% on NQ against 50–71% on the two CQADupStack components** — a coverage failure, compounded by a 384-wide linear head that L2 regression cannot push past ~90–93% once queries are diverse (`m10/PLANNING.md` §9); not a parameter-count failure. Aim (leaf-ir-asym 0.5155) and release bar (bge-small 0.5042) both need ≥88% on avg-6; projected miss. **Read `m9/FINDINGS.md`.** Mandate `instructions-m9.md`; recipe `m9/M92_LOCK.md`; runs `m9/RESULTS.md`; closed avenues `m9/EXPLORED.md`. The registered six-set transaction (`m9/FINAL_LOCK.md`) is executed on the frozen candidate as M9's close-out measurement only after M10's recipe lock is pushed (so it cannot inform an M10 decision), six-only (its reserved conditional struck by a ratified amendment), and once Dylan ratifies it; LoTTE stays unread for M10.
    25	- [ ] M10 (**budget VALIDATED and plan re-cut 2026-09-04**; resume from `m10/STATUS.md`. Retry of nano, planned 2026-09-01): same pair, same bars, same teacher, new recipe built around the coverage finding — a wider linear output from pooling three layers, LEAF's small-batch cyclic schedule, warm start from the M9 candidate, and query-form breadth from **≈1.25M harvested real titles/headings/claim sentences plus ≈1.0M generated queries** for the seven forms no corpus contains (FineWeb ruled out of M10). **The 2026-09-04 efficiency review (amendments A1–A8, `instructions-m10.md` §Amendment 2026-09-04) cut the ranking-aware phase-2 class to one arm — LEAF's own ‖e‖₂ loss — deleting the candidate bank, the mining pass and the τ rule; cut generation 3.0M → 1.0M on three published saturation curves; withdrew the full-dose second seed; capped confirmations at two decisions; and re-registered C1/C2 on clean-4 as well as avg-6 (bars 0.5046 / 0.5233) so M14's headline partition carries a pre-registered pass/fail.** Compute re-priced on measured rates: the M10 recipe runs at **910 examples/s blended on the RTX 3080** with `torch.compile` on fixed length buckets, measured on REAL tokenized corpora with zero alloc retries (`results/m10_rate_bench_real_box.json`; the earlier 683 was the uncompiled path, 560 an imported A100 planning figure) — confirmed in flight by the M10.0-e calibration's 962 ex/s, falling to ~850 under CPU contention, so **the box runs the screens and the cloud buys generation and the build — ≈$90–245 against a $1,000 ceiling**, and the 2026-09-01 "not on a 3080" ruling now binds only the dose, not the hardware. **Reviewed four ways on 2026-09-04 (Fable ×2, Codex, Opus): amendments A1–A8 and B1–B6, every finding actioned; decisions 11 and 12 both RULED 2026-09-04 (12 adopted but never EXECUTED — see the whole-plan review); `research/m10-feasibility-review-2026-09-04.md` has the feasibility verdict.** Mandate `instructions-m10.md`; evidence `m10/PLANNING.md` (§11 rates, §12 the synthetic-data question); status `m10/STATUS.md`.
    26	- [x] M11 (**closed 2026-09-03**): the zero half of the pair, shipped. **`constella-zero`** — https://huggingface.co/DylanCouzon/constella-zero — and the frozen document tower **`stella-en-400M-v5-doc-onnx`** — https://huggingface.co/DylanCouzon/stella-en-400M-v5-doc-onnx — both public, ONNX, byte-verified, served by FastEmbed as built-in models on branch `Dylancouzon/fastembed@add-constella-models`. Serving parity vs the numpy reference **4.470e-08** on 1,024 dev queries. Mandate `instructions-m11.md` (Amendments A and B), detail `m11/STATUS.md`, reusable ONNX-port checklist `m11/CODEMAP.md`. Everything that depended on M10 moved to M13 (was M12 until 2026-09-04).
    27	- [x] M12 (**closed 2026-09-04**; local box, no training): **fusion operator audit.** M7's published fused row used `convex0 w=0.8` at prefetch 1000 — an operator **Qdrant does not implement**. Audited every shipping alternative to Qdrant parity (`m12src/qfusion.py`, vendored reference, parity + degenerate tests). **Dev, depth 1000: NO MATCH** — DBSF −0.0146, RRF over `k` −0.0192, weighted RRF held-out −0.0129 [−0.0160, −0.0098]; correcting M7's unfair comparison (unfitted RRF, `k` grid monotone to its own boundary, never reaching Qdrant's default `k=2`) recovered only +0.0097 of 0.0223. The gap is real: convex0 beats **both** endpoints, so it exploits score magnitudes that rank-based operators discard. **But the ranking inverts at realistic depth** — DBSF beats convex0 at prefetch 10 and ties at 50 (degenerate branch fires 0.00%, not an artifact). On Dylan's ruling (*"convex needs to go if it can't be done in Qdrant"*) the six were re-scored: **`Fusion.DBSF` @ prefetch 100 → 0.4887 all-6 / 0.4912 clean-4** vs convex0's 0.4911 / 0.4866 — **a TIE both ways** (inside the ~0.005 band, no CI), so the case is deployability at no measured cost, not superiority. **Recorded as a post-M7 PRODUCT-POLICY OVERRIDE of the `m7/LEDGER.md:691` release freeze** — convex0 stays the M7 operator of record, C1/C2/C3 untouched. Card/README/`m11/STATUS.md` now lead with the DBSF recipe; **the live HuggingFace card push is Dylan's, not done here.** A CLOUD finding — edge fusion is unbuilt and uncosted, a note not a metric. `instructions-m12.md`, `m12/FINDINGS.md`, `m12/LEDGER.md`, cut avenues `m12/EXPLORED.md`.
    28	- [ ] M13 (**blocked on M10**, was M12 until 2026-09-04): the nano half of the pair — release `constella-nano`, its ONNX port and its FastEmbed registration. `instructions-m13.md`. The whitepaper left for **M14** on 2026-09-04. **The upstream FastEmbed PR is an M13 deliverable and waits for nano** (Dylan, 2026-09-04): one clean PR adding all three model entries, branched fresh from upstream `main`, not the throwaway `add-constella-models` branch.
    29	- [ ] M14 (**created 2026-09-04**, split out of M13): **the paper.** Runs after nano; runs anyway on the zero-only frontier if M10/M13 never do. **One paper is the default**, decided at M14's start. The deliverable is the frontier; the FINDING is that teacher quality does not predict distilled-table quality — Spearman ≈ 0 over eleven teachers, mxbai matching bge-base's tower (0.4433 vs 0.4484) with a table 0.057 worse — measured today, needing no nano. Carries the comparator table kept off the model cards. **Benchmark composition is decided HERE, once, for both zero and nano (Dylan, 2026-09-04) — but the RULE was registered 2026-09-04 before nano's numbers exist: headline = the pre-registered clean-4, all six reported beside it, contamination robustness as its own row, no re-picking the six, MS MARCO validation-only if built.** M12 measured why that is cheap: clean-4 costs the FUSED system 0.0045 and the dense table 0.0241. `instructions-m14.md`.
    30	- [ ] M15 (noted, not scoped — Dylan, 2026-08-30): an IMAGE model. Edge workloads are largely vision, so the asymmetric premise has a bigger market there. Open: whether the query side is a lookup table at all, image→image vs text→image, which frozen document tower. Do not inherit the text architecture's assumptions. `instructions-m15.md`.
    31	- [ ] M16 (**parked, long-term, may never run** — Dylan, 2026-09-03): a better `zero` BEYOND the hybrid objective (that is M12). Idea register only, no compute committed: `instructions-m16.md`. The ranked lead is document-side co-adaptation (M8's `E14-LORA`, authorised and unrun), which reopens the frozen-tower premise and costs the drop-in-to-stella's-index property. The cheap lead is cosine-space distillation, priced at ~+0.035 by a pyNIFE retention comparison on fiqa (they retain 73.4% of a weaker teacher against our 67.1%).
    32	
    33	## Standing directive: push for the best model, not a model (Dylan, 2026-08-26)
    34	
    35	Added after a session declared a target "unreachable" and was wrong within the hour. **We do not
    36	want A model. We want the best one obtainable under the constraints.** Being first to say "this
    37	projects to a negative result" is not rigour — rigour is exhausting the angles and *then* reporting
    38	honestly. Honest pessimism arrived at lazily is still lazy.
    39	
    40	**Before writing that any bar is unreachable, all of these must be true and shown:**
    41	
    42	1. **The arithmetic has been redone with the best available component**, not the current one. The
    43	   2026-08-26 case: "Tier 1 unreachable" assumed the current teacher. Swapping in a stronger
    44	   permissive teacher (stella_en_400M_v5, MTEB-Ret 58.97 vs bge-base 53.25) moved the projection
    45	   from 0.406 to 0.489 and cleared the bar outright. Quality is usually a *product* of factors —
    46	   check every factor before declaring the product capped.
    47	2. **Every failing component has been diagnosed, not just observed.** Same day: contrastive
    48	   training was called "broken" when the learning rate was 30-300x above every published recipe
    49	   and the literature has an analytic result (arXiv 2110.09348) for precisely that symptom. A
    50	   failure you cannot explain mechanistically is not evidence about the method — it is evidence
    51	   about your configuration.
    52	3. **The literature has been swept for the specific failure**, in parallel subagents, with the
    53	   real numbers extracted. Cheap, fast, and it has twice overturned a conclusion here.
    54	4. **A capability claim has been checked algebraically before being believed or dismissed.** Some
    55	   "obvious" fixes are provably no-ops (a doc-side linear map reparametrises the table;
    56	   document centering cannot change ranking at all). Do the algebra — and note the example
    57	   originally given here was itself wrong: query-side centering *was* claimed to be genuine new
    58	   capacity, and `results/m7_absorb_check.json` later disproved it to machine precision
    59	   (`mean(W-mu) = mean(W)-mu`, so it is absorbable, as are whitening, top-PC removal and any
    60	   per-token scalar weight). Only n-gram rows and multiplicity-dependent pooling add capacity.
    61	   The lesson stands; this file had the sign of it backwards for a day.
    62	5. **The negative result is reported with what would change it** — the specific measurement,
    63	   component, or bound that would flip the verdict.
    64	
    65	Corollary: a pre-registered kill criterion (e.g. the phase-2 contrastive bar) exists to stop
    66	*grinding a diagnosed dead end*, not to license abandoning an undiagnosed one. Kill the avenue
    67	after you understand why it failed, never before.
    68	
    69	### Vendor rule, relaxed (Dylan, 2026-08-26)
    70	
    71	The original rule — no component from any vendor shipping a competing vector-search product —
    72	disqualified almost every strong encoder and was costing us real quality. **Relaxed: a vendor whose
    73	vector-search offering is far from their main business is acceptable if the choice is heavily
    74	justified in the report.** Direct competitors stay out.
    75	
    76	Operationalised so a future session does not have to guess:
    77	
    78	- **OUT — vector search *is* the business:** Pinecone, Weaviate, Zilliz/Milvus, Chroma, Vespa,
    79	  Nomic (Atlas), Mixedbread, Jina (and Elastic), MongoDB (Atlas Vector Search is a flagship push;
    80	  also Voyage AI, which MongoDB acquired), Cohere (Embed/Rerank is core product). And obviously
    81	  nothing where we would be shipping a competitor's model as a *component* of a Qdrant release.
    82	- **OK WITH JUSTIFICATION — vector search is one service among hundreds:** Alibaba (gte, Qwen3-Embedding
    83	  — OpenSearch Vector Search Edition / AnalyticDB), Microsoft (e5 — Azure AI Search), IBM (granite —
    84	  watsonx managed Milvus), Google (Vertex AI Vector Search; note Gemma terms still fail the
    85	  *licence* rule independently). **Snowflake (arctic-embed) needs the strongest justification of
    86	  this group** — Cortex Search is built directly on Arctic Embed, so it is the closest of these to
    87	  core business.
    88	- **CLEAN — no vector product at all:** BAAI, NovaSearch, NVIDIA, Salesforce, academic labs,
    89	  individual/community releases (e.g. intfloat).
    90	
    91	Unchanged and non-negotiable: the **licence** must permit commercial release of derived weights
    92	(no Gemma terms, no CC-BY-NC), and **table size still binds** — the released artifact is
    93	vocab x dim, so a 250K-vocab model is disqualified on arithmetic regardless of vendor
    94	(250,002 x 1024 fp16 = 512 MB, larger than the 466 MB LightRetriever table we exist to beat).
    95	Practical filter: **vocab <= ~50K and dim <= 1024, or MRL-truncatable**.
    96	
    97	The pre-relaxation shortlist (`research/m7-teacher-shortlist.md`) was written under the strict
    98	rule, superseded by `research/m7-teacher-shortlist-2026-08-26.md`, and deleted in the 2026-08-28
    99	cleanup — git history has it. The re-run sweep happened and the teacher question is closed
   100	(stella, see the decision log).
   101	
   102	### Past decisions are revisitable (Dylan, 2026-08-26)
   103	
   104	**"If we need to revisit any past decisions to make this better, I'm open for it. Achieving our
   105	goal supersedes anything else."** So do not treat an earlier choice as settled just because it is
   106	written down — including choices in `instructions-m7.md` and in the decision log below. Anything
   107	in this list is fair game to reopen with evidence and Dylan's sign-off: the teacher, the
   108	architecture, the objective, the data mix, the tier definitions and their comparators, the
   109	milestone scope, the release target, and even structural premises like "frozen off-the-shelf
   110	document tower" or "no transformer at query time" if a better system lies the other side of them.
   111	Bring the arithmetic and the trade-off, not just the idea.
   112	
   113	**The one class of exception, and why it is not an exception to the goal.** The evaluation
   114	protocol — partitions, decontamination, the frozen comparator vectors, the single final run,
   115	pre-registered statistics — exists to make a good number *believable*. Relaxing it after seeing
   116	results does not achieve the goal, it destroys the thing the goal is for: an unbelievable 0.50 is
   117	worth less than a defensible 0.46. So protocol changes are allowed, but only **before** the
   118	numbers they would affect are observed, and only written down with the reasoning. Never
   119	retroactively, never silently. Same rule for the licensing and vendor constraints: they are
   120	commercial reality, so reopening them is Dylan's call and needs an explicit answer, not an
   121	inference.
   122	
   123	## Long runs must be watched, not launched and hoped for (Dylan, 2026-08-26)
   124	
   125	Added after a session lost most of a day to jobs that were running but wrong. Four failures in one
   126	morning: a probe that thrashed the allocator for 50 minutes per component, a mining pass costing
   127	3.6 hours instead of 3 minutes, four screen arms crashing on one shape error, and a pool whose
   128	width check would have silently overwritten 9.5 GB. Every one was visible within two minutes of
   129	starting, and none announced itself.
   130	

codex
The executable registry already differs from the brief: it specifies 14 trained arms and a denominator of 12, while several historical notes still say 13 or 16. More consequentially, its synthetic-data rule calls an unresolved contrast a “measured null,” and its confirmation rule appears incompatible with choosing batch 128 for cost. I’m checking those against the mandate before drawing conclusions.
exec
/bin/bash -lc "nl -ba instructions-m10.md | sed -n '152,330p'" in /home/dylan/asymetric-dual-encoders
 succeeded in 0ms:
   152	## Amendment 2026-09-05 — owner rulings after the verification pass (authoritative over older text)
   153	
   154	Ruled by Dylan 2026-09-05 with nothing registered trained; `m10/screen_registry.json` carries the
   155	executable form and `m10/LEDGER.md` §3 the record.
   156	
   157	| # | change | why |
   158	|---|---|---|
   159	| C1 | **Family A's STOP is removed.** A3−A2 keeps its three labels (resolved / positive not resolved / not positive) as REPORTED outcomes; the build proceeds in every case. Family A is also reported on DEV-6's CQADupStack components and FORMS-12, descriptive. M10's failure condition is the C1b release gate | COV holds none of A3's forms, and the contrast needed a point above ~0.0142 to resolve, so the STOP would most likely have fired under a true coverage effect (W12; both reviewers) |
   160	| C2 | **MiniLM-L12 and contrast F2 are cut.** F = bge-small vs MiniLM-L6 at 20M, F1 only; the Bonferroni denominator is **12** (11 one-sided at α/12, F1 two-sided at α/24 per tail = 0.025 exactly). Trained arms 14 | L12 is 12 layers / 384 hidden / 34.54M, identical to bge-small, so it tested the pretraining checkpoint, not depth; its conditional extension was an adaptive selection step (W13) |
   161	| C3 | **The weekend timeline is not binding.** The box runs as much of the screen as it can first; whatever remains moves to the rented GPU under the same registry, no re-shaping (E's two arms run together on the A100 so the contrast is not confounded by hardware) | Dylan: *"I don't care if we go over the weekend timeline … if things need to be delegated to cloud gpu it's fine. Though I prefer to do as much as we can here first"* |
   162	
   163	## Delegated authority during unsupervised windows (Dylan, 2026-09-04)
   164	
   165	Dylan: *"the agent should be able to make decisions that will help us achieve our goals if something
   166	happens during those 3 days"*, with *"good guardrails"*, and *"the model can ask a Fable sub-agent on
   167	decisions it needs to make"*. Three tiers, and the tier is decided by what the decision touches, not
   168	by how urgent it feels.
   169	
   170	- **Tier 1 — decide alone:** operational choices that change no registered constant: retries and
   171	  restarts, bug fixes, the order of runbook steps, extending the window's timeline, choosing among
   172	  alternatives the runbook already names, when to ping.
   173	- **Tier 2 — decide after a Fable consultation:** anything not covered by a registered branch that
   174	  touches the recipe, the data or the screen's evidence — an arm that cannot run as specified, a rate
   175	  that makes the plan infeasible in the window, a yield or gate outcome the text did not anticipate,
   176	  an ambiguous rule. **Mechanism:** brief a Fable subagent (`model: fable`) adversarially — the
   177	  decision, the options, the goal (win C1b), the guardrails below, the standing read-exclusion, no
   178	  sub-subagents; it returns a recommendation, its risks and what would make it wrong; the session
   179	  decides (it may disagree, saying why), records the decision **and the subagent's verbatim
   180	  recommendation** in `m10/LEDGER.md` §3 "Delegated decisions" with timestamp and commit, pushes,
   181	  pings Dylan. Prefer the reversible option; when two admissible options tie, prefer the one that
   182	  preserves the window's proof value (data, F, A). Every delegated decision is disclosed in the report.
   183	- **Tier 3 — never; Dylan only:** changing a protocol after a number it affects is observed; opening
   184	  any six, reserved or LoTTE surface outside its registered transaction; spending cloud money;
   185	  changing the teacher, the 35M cap, the pair, the bars, the conjuncts or their order, the datasets;
   186	  editing a lock after an arm has started; **overriding a registered STOP** (the family floor; A3−A2's STOP was removed by Dylan 2026-09-05);
   187	  relaxing a licence or contamination rule; releasing or publishing anything; changing a registered
   188	  constant (MDE, α, doses, seeds, quotas). A Tier-3 question stops **that branch only**: record it,
   189	  ping, continue every other branch.
   190	
   191	## Owner decisions (defaults apply until Dylan rules; each is recorded in `m10/LEDGER.md`)
   192	
   193	**States:** 1, 4, 7 default-active pending ratification · 2, 3, 5, 6, 8, 9 closed · 10, 13 adopted
   194	(strike an item to revert it) · 11 default confirmed · 12, 14, 15 adopted. This is
   195	the only pending-decision list; `m10/STATUS.md` points here.
   196	
   197	| # | decision | default while open |
   198	|---|---|---|
   199	| 1 | Ratify M9's final-lock amendment **together with the close-out amendment that strikes M9's reserved conditional** (§Stage plan, M10.2): M9's close-out is six-only and cannot spend the reserved access | blocks the close-out only |
   200	| 2 | **GPU budget — VALIDATED by Dylan 2026-09-04.** Re-priced at the measured rates after the plan review: hybrid (screens on the box, cloud for generation and the build) **≈ 44–88 cloud GPU-hours ≈ $90–245** with screens and generation on the box (A7, decision 14); all-cloud ≈ 130–190 GPU-hours ≈ $220–500. Ceiling **$1,000** unchanged and unspent. PLANNING §6 (re-derived) and §11 | approved; the day-one benchmark still re-derives §6 before anything scales |
   201	| 3 | FineWeb as a seed — **ruled out 2026-09-01** (delegated): Wikipedia and the approved corpora carry the topics; FineWeb adds a rights review and a blocklist for no measured gain. Reopening condition in `m10/EXPLORED.md` | closed |
   202	| 4 | PAQ (machine-generated questions over Wikipedia; data CC BY-SA, generation code CC BY-NC) as query text | include, from Facebook's official release (never the unofficial HF mirror); 1.0M uniform sample in the build (seed 0, file hashes pinned), 4.037M in the volume-control screen arm A2 only; attribution recorded |
   203	| 5 | FineWeb documents — **excluded 2026-09-01**: no reserved-set document fingerprints exist and creating them would open reserved corpora (`m9/LEDGER.md` §1.3) | closed |
   204	| 6 | A >35M tier — **ruled out by Dylan 2026-09-01**; 35M is a hard cap | closed |
   205	| 7 | Confirm: LoTTE read #1 withdrawn unexecuted in M9; renumbering M10/M11/M12 | as recorded in `m9/STATUS.md` and CLAUDE.md |
   206	| 8 | Second build seed at full dose — **WITHDRAWN 2026-09-04** (amendment A5): descriptive by construction, could trigger no action, and cost ≈100 GPU-hours. Replaced by a screen-dose seed pair on the selected recipe as the replication band; the freed hours go to extension cycles | closed |
   207	| 9 | The 2026-09-01 review amendments taken with the compute ruling: dose 200M examples, screen dose 5M, G-1536, the COV resolution number, extension capped by budget (PLANNING §8, amendment block and pass 7). **D-KL1, D-NCE and the seed-rank field are struck by amendment A1; the resolution number is repurposed by A4** | adopted as amended |
   208	| 10 | The **2026-09-04 plan-review amendments A1–A8** (§Amendment 2026-09-04) | adopted on Dylan's "make your changes to the plan"; any item he strikes reverts to the 2026-09-01 text |
   209	| 11 | **Release rule under four conjuncts** (§Goal claim table): does a C1a pass with a C1b fail ship as "nano"? | **no** — "release" needs C1b, the headline partition (Codex 2026-09-04: releasing on the partition whose margin is 73% fiqa would be motivated). A C1a-only pass is still *published* as a frontier measurement labelled not recommended, as `zero` was. **Dylan 2026-09-04: "Let's make sure we win enough so this isn't a question"** — the default stands and the instruction is to build for C1b |
   210	| 12 | **CUREv1 as a validation-only biomedical read** — 2,000 real clinician queries, CC BY-NC 4.0, PMC-OA full-text passages (PubMed-family), pools annotated by Qwen 2.5 72B; fingerprint-screened vs the six; never training data. Reopens the validation clause of M7's source-family rule (`research/m10-feasibility-review-2026-09-04.md` §4c) | **ADOPTED by Dylan 2026-09-04 as a reported diagnostic beside every arm** — never selection-bearing (LLM-judged pools; provenance overlap fingerprints cannot see; Codex M6). Admitted at M10.0-d with revision, licence clause and corpus provenance recorded; fingerprint-screened against the six; joins the protected index. **Withdrawn by the review:** PubMed titles / PubMedQA as training text — no affirmative grant on PubMed abstracts (NLM disclaims copyright; publishers may hold it); reopens only with per-record licence provenance (PMC-OA CC BY records) *and* the contamination rule resolved (§4a) |
   211	| 13 | The **2026-09-04b feasibility-review amendments B1–B6** (§Amendment 2026-09-04b) | adopted; strike any item and it reverts |
   212	| 14 | **Generation on the box with Qwen's official 4-bit release** — `Qwen/Qwen3-8B-AWQ`, revision `4da05a8e…`, Apache-2.0 — served by vLLM (to be installed; smoked first), instead of hosted bf16. The 12-form smoke (200 per form) is pushed as `m10/SMOKE.md` so Dylan can approve remotely; ≈1.0M queries then cost ≈1 box-day, and family F can start inside the weekend window before any cloud dollar. Hosted bf16 of the pinned base stays the fallback if vLLM will not run on the card or the smoke / A8 gates fail | **ADOPTED by Dylan 2026-09-04 ("Go on 14")**. The §Data contract now reads "an official Qwen release, pinned by revision"; the A8 diversity gate guards against 4-bit repetition; the weekend runbook in `m10/STATUS.md` carries the fallback branch |
   213	| 15 | **Conditional pre-approval of the generation smoke** (§Data contract): the contract rate is computed by the session, the on-form rate is judged by an independent Fable subagent against each form's registered description, the sample and verdicts are pushed as `m10/SMOKE.md` with a GitHub issue and a phone ping, and generation starts after a **six-hour veto window** unless Dylan objects; a late veto drops the form from the build. The report discloses that the on-form read was model-judged with owner review after the fact | **ADOPTED by Dylan 2026-09-04 ("Yes to decision 15")** — the risk of waiting (a lost weekend) outweighs the bounded risk of one off-form form, which quotas per form make cheap to drop or regenerate |
   214	
   215	## Goal, bars, and the permitted claim — unchanged from M9
   216	
   217	nano = a ≤35M transformer query encoder serving stella-400M's frozen 1024d index (the SAME index
   218	as zero). Bars, paired on `results/perquery.json` (sha `6b18e3dd…`, irreplaceable):
   219	
   220	| | avg-6 | NDO-4 |
   221	|---|---|---|
   222	| teacher ceiling (stella symmetric) | 0.5744 | 0.5640 |
   223	| **AIM: leaf-ir-asym** | **0.5155** (89.7% retention) | 0.5233 |
   224	| **RELEASE BAR: bge-small** | **0.5042** (87.8%) | 0.5046 |
   225	
   226	**NDO-4 is clean-4** (`nfcorpus`, `scidocs`, `scifact`, `trec-covid`), the partition M14 registered
   227	on 2026-09-04 as the headline for both halves of the pair.
   228	
   229	C1 nano-dense > bge-small (release); C2 nano-dense > leaf-ir-asym (aim). Both dense-vs-dense;
   230	fused rows descriptive only. **Amendment A3, 2026-09-04: C1 and C2 are each registered on BOTH
   231	partitions** — `C1a`/`C2a` on avg-6 (bars 0.5042 / 0.5155) and `C1b`/`C2b` on clean-4 (bars
   232	**0.5046 / 0.5233**), all four reproducible from the frozen comparator rows by
   233	`scripts/clean4_bars.py` → `results/m10_bars.json` (a comparator-only read taken 2026-09-04 before
   234	any nano number existed).
   235	
   236	**Multiplicity: fixed-sequence gatekeeping, NOT Holm** (corrected after the Fable review, which
   237	found the two named together and incompatible — Holm orders by p-value, a fixed sequence does not,
   238	and a text permitting either lets a future session choose after seeing the numbers). Registered
   239	order **C1b → C1a → C2a → C2b** — the release gate of decision 11 first (the stress table has a case,
   240	`arguana` at 65%, where clean-4 passes and avg-6 fails, which the old C1a-first order could never
   241	test; Opus 2026-09-04), then the M9 headline sentence's conjunct C2a before the stretch aim C2b; **if
   242	Dylan makes C1a the release gate, C1a goes first**. Each conjunct is tested at the **full one-sided
   243	0.025** and testing stops at the first non-rejection. This controls the family error rate and, unlike Holm-4, **costs
   244	the avg-6 release claim nothing** — under Holm-4 the smallest-p conjunct would have needed 0.00625
   245	and C1a's bound would have widened ~11% against the two-conjunct design it replaces.
   246	
   247	**Planning proxies, not pass points** (`results/m10_bars.json`, `planning_proxies`): the bar plus the
   248	comparator-pair (leaf vs bge) bootstrap width at the **registered 0.025 quantile** — the Codex pass of
   249	2026-09-04 caught the script still on M9's Holm-2 0.0125. Nano's own interval depends on its per-query
   250	differences, so no retention figure "passes" by arithmetic; the rule is the final run's own bound.
   251	
   252	| conjunct | bar | proxy width | proxy point | = retention of ceiling |
   253	|---|---|---|---|---|
   254	| C1a release, avg-6 | 0.5042 | 0.0089 | 0.5131 | 89.3% of 0.5744 |
   255	| C1b release, clean-4 | 0.5046 | 0.0122 | 0.5168 | 91.6% of 0.5640 |
   256	| C2a aim, avg-6 | 0.5155 | 0.0089 | 0.5244 | 91.3% |
   257	| C2b aim, clean-4 | 0.5233 | 0.0122 | 0.5355 | **94.9%** |
   258	
   259	**Disclosed with it (B1, `results/m10_conjunct_arithmetic.json`):** under a *uniform-retention
   260	proxy* the four sit at 89.3 / 91.3 / 91.6 / 94.9% (C1a / C2a / C1b / C2b) — a lens, not an ordering:
   261	retention is distribution-specific (M9: 93.8% covered, 50–71% uncovered), so heterogeneous
   262	retention can reorder them. To equal bge-small per dataset: scifact 91.4 · nfcorpus 83.0 · fiqa 72.9
   263	· arguana 94.7 · scidocs 85.7 · trec-covid 92.0%; LEAF beats the ceiling on trec-covid. **Stress:**
   264	one dataset at 65% with the rest at 94% clears no conjunct if it is `trec-covid` or `scifact`, only
   265	C1a if `nfcorpus`. At uniform 92%, fiqa (disclosed stella training data) supplies 73% of nano's avg-6
   266	margin — why clean-4 is the headline. **Prior, registered before any nano number:** M9 held 93.8% on
   267	its covered form; a bounded literature search (pure embedding regression, ≥10× teacher/student
   268	parameter gap; `research/m10-feasibility-review-2026-09-04.md` §2) found no result above ~95%, and
   269	EmbedDistill's 95–97% at ~10× used labels and score distillation. So C1a is expected if coverage
   270	works, **C1b and C2a are the contest, and C2b is a low-prior stretch aim** — the screen results that
   271	would raise the prior are G-MLP or D-COV moving `nq-250k` retention from ~94% toward 96–97%.
   272	clean-4's interval is ~36% wider than avg-6's because `trec-covid` carries a
   273	quarter of the clean-4 macro on **50 queries**, and C2b therefore demands LEAF-level retention on
   274	the headline partition. Registering clean-4 is still right — M14 headlines it — but the plan is
   275	registered knowing C2b is the hardest of the four by a wide margin, not knowing only that its bar
   276	is 0.0078 higher. **clean-4 is also the most data-designed partition** (Opus): §Data names
   277	scidocs↔titles, scifact↔claims, nfcorpus/trec-covid↔headings, so the headline controls *teacher*
   278	contamination while maximising *training-design* targeting — two different exposures, both stated. **C2 is a whole-system comparison** (stella documents + nano queries
   279	vs arctic documents + LEAF queries): different document towers, index sizes, encode costs and
   280	disclosed teacher overlap. It supports exactly M9's verbatim headline sentence and **no statement
   281	about nano versus LEAF's query tower**; the report carries both systems' retention against their
   282	own teachers, index bytes, document-encode cost and query latency beside the number. NDO-4 and
   283	reserved NDO-3 stay descriptive.
   284	
   285	**Claim decision table under four conjuncts (B2).** Testing stops at the first non-rejection:
   286	
   287	| outcome (order C1b → C1a → C2a → C2b) | release? | permitted claim |
   288	|---|---|---|
   289	| C1b fails | **not a release** (decision 11 default): the checkpoint is published for reproducibility and as a frontier point, labelled *not recommended*; the card states it did not resolve above the bge-small system on the contamination-controlled partition. **M13 consequence (Opus):** a not-recommended model does not enter the upstream FastEmbed PR, so the single-PR ruling then covers zero and the tower only — Dylan decides this with decision 11 | none |
   290	| C1b passes, C1a fails | release | "the stella-document + nano-query system outperformed the bge-small system on the four contamination-controlled datasets"; avg-6 reported unresolved |
   291	| C1b, C1a pass, C2a fails | release | the same sentence on both partitions |
   292	| C1b, C1a, C2a pass, C2b fails | release | M9's verbatim headline sentence (avg-6); the paper states the aim was missed on the headline partition |
   293	| all four pass | release | the headline sentence on both partitions |
   294	
   295	**C1 is a system comparison too** (Opus M7): bge-small is a symmetric 384-d system with its own
   296	66.7 MB query asset and its own index, so every C1 sentence carries index bytes and document-encode
   297	cost beside it, exactly as C2 does. M9's "C1 fail, C2 pass → aim claim permitted" row does not
   298	survive gatekeeping and is deleted.
   299	
   300	Additional mandatory disclosures: per-dataset retention, the
   301	synthetic-data provenance table (generator revision and terms, prompts, seed sources and ids,
   302	counts per form, removal counts per screen), the selection-surface table (licence evidence,
   303	revisions, sizes, removals from training data), dose in examples / tokens / GPU-hours beside
   304	LEAF's ~100 A100-hours, and the dev-reuse count.
   305	
   306	## Stage plan
   307	
   308	- **M10.0 DIAGNOSIS + SCREEN LOCK** (no six-set, reserved or LoTTE output exists during this stage).
   309	  (a) Mac diagnostics — **done 2026-09-01**, `m10/RESULTS.md`, PLANNING §9–9c: the rank probe
   310	  (a 384-d subspace keeps 99.5% of one query distribution and 90–93% of three; 98–100% at 640 —
   311	  evidence about the class under L2 regression, not a bound), the head-width probe (a frozen
   312	  bge-small retains more with each pooled layer), and the serving-parity check (fastembed 0.8.0
   313	  reproduces the three-layer per-token head to 2e-7; 34.5M parameters). **Action taken:** the head
   314	  is applied per token — linear by default, G-MLP's residual form as an arm (§Screen); the pooled
   315	  feature widens to three layers; family G decides. Repeat the
   316	  parity check on MiniLM's head before family F may select it, on the four-layer head before G's
   317	  1536 arm runs, and on the trained artifact with M9's locked parity sample before the freeze.
   318	  (b) Capacity probe (`m9src/capacity_probe.py`, unchanged) — **optional, report-only**: no outcome
   319	  changes an M10 action under the hard cap; runs only on an idle GPU; its 109M student is
   320	  768-hidden, so any gain is partly width.
   321	  (c) Per-component DEV-6 read of the M9 candidate incl. `heldout-longq` (the baseline row).
   322	  **Runs on the box** (amendment A7): it holds the checkpoint and the encode caches.
   323	  (d) **COV admission** (§Surfaces): for every candidate component, record in `m10/LEDGER.md` §2
   324	  its primary-source licence URL and terms, HF repo and revision, corpus size, query count, qrels
   325	  format and metric, its corpus-level contamination check and its fingerprint screen against the
   326	  six and the reserved four. A component is named COV only after that record is pushed. **COV
   327	  admits only surfaces no M10 decision has read**: the two CQADupStack dev components were scored
   328	  by the Mac diagnostics (PLANNING §9, 86 raw reads) and stay in DEV-6. Then every admitted COV
   329	  corpus, query set and document set joins the protected index (`m8src/protected_filter`) before
   330	  any seed is drawn or any harvested, PAQ or synthetic text is constructed. **Re-run admission under

exec
/bin/bash -lc "nl -ba instructions-m10.md | sed -n '330,507p'" in /home/dylan/asymetric-dual-encoders
 succeeded in 0ms:
   330	  any seed is drawn or any harvested, PAQ or synthetic text is constructed. **Re-run admission under
   331	  Dylan's 2026-09-04 licence rule before judging the family floor: ConsumerContractsQA (CC BY-NC) is
   332	  re-admissible, restoring the legal family's second component (three family IDs without LEDGER, §Surfaces).** The **COV resolution number** (§Surfaces) is
   333	  measured on the admitted surface and pushed before (e) as the screen's **power disclosure** (A4's
   334	  sizing struck by the Codex pass). The `arxiv-title` draw happens at the start of M10.1's harvest, once the artifact exists (§Surfaces).
   335	  (e) **Screen lock**: `m10/LEDGER.md` §0 (skeleton committed 2026-09-01) fixes every arm of
   336	  §Screen (fifteen arms), order, doses, seeds, surfaces, the thirteen contrasts, MDE 0.0056 and the
   337	  fixed 0.025/13 bound, confirmation design and outcome→action maps. **Pushed right after M10.0-d,
   338	  before any harvest** (Codex B4); the data-dependent constants are §0b, pushed before the first arm.
   339	- **M10.1 DATA.** Harvesting of the real query-like text (§Data), generation for the non-harvestable
   340	  forms under the §Data contract (200-query smoke per form first), PAQ samples, decontamination
   341	  against the protected index (now including COV) and the six's documents, the FORMS-12 hold-out,
   342	  teacher targets, `results/m10_data_manifest.json` with hashes and the provenance table.
   343	  **Before any arm and recorded in the manifest**, the two A8 quality gates: per-form
   344	  near-duplicate and dispersion metrics, and the stella-space distribution-overlap check against
   345	  real MS MARCO dev queries (validation only).
   346	- **M10.2 SCREEN + RECIPE LOCK.** The arms of §Screen, the confirmation runs, then one pushed lock
   347	  commit with every field of M9's M9.2 list filled — including the objective, the best-to-best
   348	  plateau/extension rule on annealed checkpoints, the GPU-hour allocation under the ceiling (build,
   349	  extensions), the final-run registry with the four C-conjuncts and their fixed sequence, and
   350	  LoTTE read #1's manifest. Codex and Fable review the pushed lock. **Then, and only then,** M9's close-out runs:
   351	  its registered six-set transaction **amended before execution to six-only** (the `if C1 then
   352	  execute` reserved conditional is struck; disclosed and ratified under decision 1), so M9 cannot
   353	  spend the reserved access. Its rows are a forecasting calibration and a whitepaper frontier
   354	  point; nothing in M10 may change on them. **Wherever those rows are published — frontier table,
   355	  whitepaper, model card — they carry M9's build-provenance disclosure with them:** the M9
   356	  candidate's provenance spans TWO build lock states (mid-build infrastructure repair, session
   357	  force-reopened; `m9/LEDGER.md` "M9.3 BUILD PROVENANCE DISCLOSURE" and `m9/BUILD_LOG.md`), and it
   358	  may never be described as an uninterrupted single-lock run. LoTTE read #1 then runs as the
   359	  registered veto on the selected recipe.
   360	- **M10.3 BUILD.** One candidate under the kill and extension rules; export + parity; freeze;
   361	  pre-freeze review; LoTTE read #2 (audit only). The screen-dose seed pair (amendment A5) is the
   362	  replication band; no full-dose replica runs.
   363	- **M10.4 FINAL.** The six-set transaction → decision → the registered reserved conditional.
   364	
   365	## Data
   366	
   367	**Query corpus (~3.7M unique texts, query role) — amendment A2: most of the form breadth is REAL text.**
   368	
   369	| source | build count | licence | role |
   370	|---|---|---|---|
   371	| M9 real queries (hotpotqa, squad, esci, mrtydi, nqopen, triviaqa; fever out) | 463K | CC BY-SA / Apache | real forms |
   372	| PAQ sample (decision 4) | 1.0M | CC BY-SA (data) | factoid volume — capped so it cannot dominate |
   373	| **Harvested real query-like text** from the licensed document pool (§Harvest) | ≈1.25M | inherits the pool's licences; no new rights surface | the harvestable forms, with **no generator prior** |
   374	| **Synthetic, Qwen3-8B (Apache-2.0). DEFAULT (decision 14): Qwen's official 4-bit release `Qwen/Qwen3-8B-AWQ`, revision `4da05a8e…`, served by vLLM on the box. Fallback, the only one: the bf16 base (revision `b968826d…`) under the same vLLM contract on a rented GPU after Dylan returns, ≈10–20 GPU-hours; never a third-party API** | ≈1.0M | generated under the generator's terms; provenance pinned; **not redistributed** without review | ONLY the non-harvestable forms |
   375	
   376	**Form taxonomy — 12 forms**, quotas locked at M10.1 (±10% realized), and each form is assigned to
   377	exactly one of the two sources. **Harvested (real text, ~250K each, ≈1.25M):** paper-title query ·
   378	scientific claim (a statement) · 2–4-word keyword query · factoid question · product-search query.
   379	**Generated (~143K each, ≈1.0M):** how-to / troubleshooting question with title and body · long
   380	counter-argument paragraph (120–220 words) · finance / personal-economics question · comparison
   381	question · yes/no verification question · conversational multi-sentence request · **consumer-health
   382	question** (generated, not harvested, since MedlinePlus and CDC were kept out of M10 — Codex B6
   383	found it listed under both). A harvested form that falls under 100K reverts to generation at ≈143K
   384	and the split is reported as realized. **The assignment's motivation, stated as the hypothesis it is:** three of the four clean-4 headline
   385	datasets have a real-text counterpart (scidocs↔titles, scifact↔claim sentences, trec-covid and
   386	nfcorpus↔headings and consumer-health), so *if* harvesting covers those forms the headline partition
   387	rests on real text plus the teacher rather than on the generator's prior. **That is what contrast
   388	A3−A2 tests; it is not established.** The generated half covers the interactive forms no corpus
   389	contains, which map to the non-headline sets and to COV's forum-style families.
   390	
   391	**Harvest sources — the Fable review found the existing pool cannot carry the scientific forms.**
   392	The 6.15M pool is `esci-prod` (808K product listings), `hotpotqa-corpus` (5.23M Wikipedia),
   393	`mrtydi-docs`, `squad-ctx` and `fever-pos` (excluded) — Wikipedia and products, **no scientific
   394	text**. Harvested from it, "paper title" would be Wikipedia article titles and product names and
   395	"scientific claim" would be Wikipedia lead sentences, neither of which is the form scidocs and
   396	scifact test. Two consequences, both binding:
   397	- **A scientific source is added for those two forms: arXiv metadata (titles and abstracts).** It
   398	  must clear the same licence gate as any training source before use — primary-source evidence of a
   399	  commercial-use grant recorded in `m10/LEDGER.md` §2 (the Kaggle release is CC0 1.0; the OAI feed
   400	  is not, so the record must name which artifact and revision is used). It is a different source
   401	  family from S2ORC, PubMed, CORD-19 and NutritionFacts, all of which stay excluded for
   402	  contamination, and every document passes the existing screens against the six's documents. **If
   403	  the licence evidence does not clear, both forms revert to generation and the report says so.**
   404	- **Yields are measured before quotas are locked.** For every extraction rule: post-dedup,
   405	  post-screen unique count, pushed to `m10/LEDGER.md` §1 at M10.1 before the per-form quotas are
   406	  fixed. **A harvested form that yields under 100K post-dedup reverts to generation**, and the
   407	  realized harvested/generated split is reported rather than assumed. `consumer-health`
   408	  is **generated** from Wikipedia-medical seeds. Sources verified 2026-09-04b (clauses in
   409	  `research/m10-feasibility-review-2026-09-04.md` §4b) but **kept out of M10**: MedlinePlus
   410	  government-authored topics and CDC pages are US public domain yet are MedQuAD's sources, so
   411	  harvesting them would let the MedicalQA COV read reward A3 by construction (Codex M7);
   412	  ClinicalTrials.gov's terms clause could not be read. Out on licence: DailyMed, OpenAlex,
   413	  bioRxiv/medRxiv, WHO, Cochrane, everything PubMed / PMC / Europe PMC (no affirmative grant on
   414	  abstracts). In: **arXiv metadata** (CC0).
   415	
   416	**§Harvest — the real-text pipeline (M10.1, deterministic, no model in the loop).** From the 6.15M
   417	pool documents and the Wikipedia seed corpus, all under licences already approved for training:
   418	titles and headings as-is (title and keyword forms); declarative lead sentences of abstract-like
   419	passages, filtered to 8–40 words with a finite verb and no first person (claim form); sentences
   420	ending in `?` extracted with their preceding sentence as optional body (factoid, consumer-health,
   421	product forms, routed by the source corpus). Every harvested string goes through the same screens,
   422	quotas and hold-out as a generated one, and the manifest records the extraction rule, source
   423	document id and per-rule yield. **Seeds for the generated half:** Wikipedia stratified by
   424	top-level category (CC BY-SA) and the approved pool corpora. FineWeb is out of M10 (decisions 3
   425	and 5). MS MARCO may never seed either half. **Contaminating source families — never seeds, never regression text, and
   426	excluded from COV** (`research/m7-data-licensing.md` map): S2ORC / Semantic Scholar; PubMed;
   427	NutritionFacts.org and its mirrors; CORD-19; StackExchange personal finance (money.SE) and Reddit
   428	finance; args.me / idebate; every six-set and reserved corpus.
   429	
   430	**Generation contract (M10.1, decisions 14 and 15):** generator `Qwen/Qwen3-8B-AWQ`, revision
   431	`4da05a8edb55c6046cce958586c33b61da07bb79`, served by vLLM on the box (version, launch flags and health assertion pinned in
   432	`m10/STATUS.md` step 0a; **thinking disabled**, `chat_template_kwargs={"enable_thinking": false}`);
   433	sampling temperature 0.8, top-p 0.95, `max_new_tokens` per form (60, or 400 for the argument and
   434	conversational forms), `seed = blake2b-64(seed_passage_id)`; the reply must parse as one JSON list of
   435	exactly n strings (`m10src/forms.parse`, strict — no preamble); one retry on a contract failure, then
   436	the seed is dropped; exact-duplicate queries removed. **Fallback, and the only one:** the bf16 base
   437	`Qwen/Qwen3-8B`, revision `b968826d9c46dd6066d109eabc6255188de91218`, served by the **same vLLM contract on a rented GPU after
   438	Dylan returns** — never a third-party API, which could not honour the pinned revision and per-request
   439	seed. It fires if vLLM will not run on the card (step 0a's bounded attempts), if the throughput floor
   440	is not met, or if a form fails its gates twice under AWQ (that form alone is re-smoked under bf16).
   441	**Smoke (decision 15):** the **seven generated forms**, 200 queries each. A form passes when
   442	≥ 90% of replies meet the contract (computed by the session) **and** ≥ 80% of a 50-query sample is
   443	judged on-form by an **independent Fable subagent** against the form's registered description in
   444	`m10src/forms.FORMS` (verdicts pushed with the sample). The sample is pushed as `m10/SMOKE.md`, a
   445	GitHub issue "M10 smoke approval" is opened and Dylan is pinged; **generation of a passing form
   446	starts after a six-hour veto window** unless Dylan objects there (`redraft: <form>: <note>`) or via
   447	Remote Control. A failing or vetoed form's prompt is revised at most twice, each revision recorded in
   448	`m10/LEDGER.md` §1 with its hash before the next smoke; a form that fails twice is **dropped from the
   449	build**, its quota not redistributed, and reported. A veto arriving after generation started drops
   450	that form's queries from the build and, time permitting, regenerates under a redraft. The report
   451	discloses that the on-form read was model-judged with owner review after the fact. The full state
   452	machine is `m10/STATUS.md` step 1. **Seeds are pre-filtered:** a seed
   453	passage that exact- or near-matches the protected index is never used.
   454	
   455	**Screens on every generated or PAQ query, thresholds fixed here (M7's fingerprints,
   456	`m7src/decontam.py`):** exact `blake2b-64` match or word-8-gram bottom-32 sketch ≥ 8/32 against
   457	(i) the protected index (six + dev + reserved + LoTTE **+ admitted COV queries and documents**),
   458	(ii) the six's documents; (iii) any word-5-gram shared with the query's own seed passage (a
   459	copied span is not a query) — **for harvested text this screen is against the source document
   460	minus the harvested span itself, since a harvested title IS a span of its document**;
   461	word-4-gram containment for 4–7-word queries. The M9 real-query pool
   462	and the document pool are re-screened against the COV additions (R1 removes matching queries;
   463	matching pool documents are
   464	removed too). Removal counts per screen, per form and per COV component are recorded **before any
   465	COV component is scored**. **FORMS-12 hold-out:** 500 seed documents per form are set aside first;
   466	queries generated or harvested from them are never trained on.
   467	
   468	**Quality gates (amendment A8), executed on the immutable manifest before any arm, recorded in
   469	`m10/LEDGER.md` §1; each is a REPORTED number with one registered action.** The risk this addresses
   470	is not label noise — there are no labels, and the teacher's embedding of any text is a correct
   471	target by construction — but distribution shift and diversity collapse.
   472	1. **Diversity, per form:** after exact dedup, a query is a *near-duplicate* if its word-8-gram
   473	   bottom-32 sketch matches ≥ 16/32 with an earlier query of the same form (the earlier one is the
   474	   representative); the rate is near-duplicates ÷ the post-exact-dedup count. **Action:** above 25%
   475	   the form keeps only its representatives (a real cut, never topped up); if fewer than **50,000**
   476	   remain the form is dropped from the build and reported. Mean pairwise stella-space cosine is a
   477	   diagnostic with no threshold.
   478	2. **Distribution overlap against real queries:** encode a 50,000-query sample of **MS MARCO dev**
   479	   (validation only — never a seed, target, negative or gradient; no cache under
   480	   `work/train/sources/`) and each form's sample with stella in the query role, and report each
   481	   form's mean cosine to its nearest real-query neighbours plus the two-sample energy distance.
   482	   **Action: none — this is a disclosed diagnostic**, because MS MARCO is web-search-shaped and a
   483	   form legitimately unlike it (paper titles, arguments) must not be penalised for that. It exists
   484	   so the report can state how far the training distribution sits from a real one, which is the
   485	   one thing M9's coverage failure had no outside measurement of. FORMS-12 cannot serve this
   486	   purpose: it scores student-teacher agreement on the same synthetic queries and is circular for it.
   487	
   488	**Document corpus (document role):** the M9 pool, 6.15M documents, re-screened as above. FineWeb
   489	documents are excluded (decision 5).
   490	
   491	**Form-balanced query sampling (Fable M8, zero cost).** 463K real + 1.0M PAQ makes ~37% of query
   492	texts factoid — the one form M9 already retains at 93.8% — so uniform-by-example sampling would
   493	spend 37% of the query gradient where there is nothing to gain. The anchor therefore samples
   494	query-role examples **balanced across the 12 forms** (each form's presentation share equal, texts
   495	drawn with replacement within a form), and the lock records the realized shares. Family A2 still
   496	measures raw PAQ volume, and the unbalanced variant is available as a reported diagnostic, not an arm.
   497	
   498	**Mix:** by *example*, decided by screen family B; default **75% query-role / 25% document-role**
   499	examples as a **presentation share over a window of 4 steps** — 3 query-bucket steps, 1
   500	document-bucket step — because single-chunk length bucketing makes every step pure-query or
   501	pure-document (Opus M4; the two-chunk per-step form costs the whole speedup A7 rests on). Family B:
   502	100/0 = query steps only; 50/50 = alternating. Query-role examples get raw bytes (prompt policy (b)); document-role examples
   503	carry M9's fixed document-role marker; teacher targets use the s2p template for queries and raw
   504	bytes for documents. The same student input never maps to two teacher targets.
   505	
   506	**Hard candidates, the mining pass, the 1M bank, the HNSW fallback and the seed-rank field are all
   507	struck by amendment A1** — nothing in M10 needs a candidate list once family D is a single

exec
/bin/bash -lc "nl -ba instructions-m10.md | sed -n '508,690p'" in /home/dylan/asymetric-dual-encoders
 succeeded in 0ms:
   508	loss-form arm. Reopening condition in `m10/EXPLORED.md`.
   509	
   510	## Recipe (defaults; screen families decide the marked items)
   511	
   512	- **Student:** bge-small-en-v1.5 is the **screen anchor**; the **build student is decided by family
   513	  F**, MiniLM-L6-v2 by default. **Amendment A6, as amended after the Opus pass: F runs FIRST, and every later family including
   514	  A is screened on F's winner** — it used to run seventh, which made every other verdict transfer to
   515	  the build student by assumption. Before F can select MiniLM, its three-layer head passes the same
   516	  export and fastembed parity check as bge-small's (M10.0-a; the Mac or the box can run it). **Feature [family G]:** masked mean-pooled hidden states of
   517	  layers 12, 8 and 4 concatenated (1152-d; MiniLM-L6: layers 6, 4, 2) → Linear(1152→1024) → L2
   518	  normalize; head 1.18M parameters, 34.5M total for bge-small. G's fourth arm adds layer 2
   519	  (MiniLM-L6: layer 1) → 1536-d, head 1.57M, 34.9M total, under the cap; its head passes the same
   520	  parity check first. Warm-started in closed form (ridge)
   521	  for the bge-small init; the M9-candidate init keeps its 384-d head and zero-initializes the two
   522	  extra layers' columns. Exported per token so fastembed's mean pooling reproduces it exactly.
   523	  Any per-token head, linear or not, ships this way (B3; `m10/EXPLORED.md`): the **trainer pools
   524	  after the per-token head** (masked mean, then normalize — the parity script's reference form) and
   525	  the **export wrapper emits the per-token output** for fastembed to pool; a wrapper-parity test
   526	  precedes any G-MLP arm (M9's trainer pooled before the head and must be ported). **Query-asset
   527	  size (B6):** a *projection* — bge-small or MiniLM-L12 with the three-layer head ≈70.8 MB fp16
   528	  (10⁶ bytes) against M9's 70 MB *target*, whose rule is a logged, measured quality justification
   529	  (`instructions-m9.md` §3); MiniLM-L6 ≈49 MB. Measured at export with tokenizer bytes and CPU
   530	  latency (`m9src/edge_cost.py`), including G-MLP's per-token cost if it wins.
   531	- **Phase 1:** squared L2 on unit-norm teacher vectors, fp32 loss, bf16 autocast (M9 form). A
   532	  cosine-space variant is closed by algebra, not measurement (`m10/EXPLORED.md`): the head ends in
   533	  L2 normalize and the targets are unit-norm, so ‖a−b‖² = 2−2cos and the gradients agree up to a
   534	  factor of 2.
   535	- **Objective [family D], amendment A1 — one arm, not three.** The single arm is **D-NORM**:
   536	  LEAF's own loss, the L2 **norm** `L = ‖t − s‖₂` on unit-norm vectors, in place of the squared L2,
   537	  from the first step. This is what the 97.7%-retention system minimised and M9 had silently
   538	  diverged from it; it is a one-line change and adds no data structure. Default is the squared L2.
   539	  **The ranking-aware class (KL over mined candidates, InfoNCE on the seed passage) is CUT** — with
   540	  it the τ entropy rule, the 1M-document bank, the mining pass and the 129-way spec. Evidence:
   541	  LEAF's Appendix B added distillation terms to plain regression and found no improvement, and
   542	  every comparable asymmetric system (LEAF, EmbedDistill, arXiv 2306.11550, mxbai-edge-colbert,
   543	  DistilVDR) reaches 92–98% retention without one, so the class is **unnecessary at our target**;
   544	  cutting it is a cost-and-sufficiency judgement. **Corrected after the Fable review:** LEAF's
   545	  Appendix B is about *intermediate-layer* KD (MiniLM / TinyBERT / DistilBERT losses at
   546	  54.9 / 53.7 / 55.3 against 60.7), not about a ranking term on top of regression, and EmbedDistill's
   547	  own ablation composes the two additively — **no published null exists for the class we cut.** The
   548	  earlier wording claimed one and was wrong.
   549	- **The registered plateau response [replaces the cut class] — this is what M9 finding #4 asked
   550	  for.** M9's lesson was that a phase 2 must be specified at lock or a flat curve has no registered
   551	  answer; cutting family D without one would recreate exactly that. So: at every cycle end the run
   552	  records per-form retention (FORMS-12 by form) and per-family COV. **If the plateau rule fires (§Kill) while the dev→six forecast is below the release bar and
   553	  `max_extension_cycles` is not exhausted, exactly one registered top-up cycle runs, pre-empting the
   554	  stop once; afterwards the form weights revert to balanced and the extension rule resumes** — one
   555	  further cycle of 66.7M examples, linear 1e-4→1e-5 as cycle 3, with the **bottom two forms by
   556	  FORMS-12 retention at 2× presentation weight** and everything else unchanged. It needs no new data,
   557	  no candidate list and no new loss; it is the plan's own coverage thesis applied to its own curve.
   558	  It fires at most once, costs one extension cycle, is counted against `max_extension_cycles`, and
   559	  the lock records the forecast formula and the tie-break for "bottom two".
   560	- **Reopening the ranking-aware class:** only as its own milestone, never mid-M10. The condition is
   561	  read on the **post-final six-set number** (the release bar is not observable during the build —
   562	  Fable M4), so it is a next-milestone decision by construction: it reopens if M10's final avg-6 or
   563	  clean-4 dense row misses its release conjunct by less than 0.01. Recorded so nobody re-derives it.
   564	- **D-COV [family D], added 2026-09-04 from the Fable review.** `L = (s − t)ᵀ Σ (s − t)` with `Σ`
   565	  the covariance of the pool's frozen stella **document** vectors, normalized to unit trace and
   566	  shrunk as `(1−α)Σ̂ + αI/1024` with **α = 0.1** fixed here (never tuned on a selection surface).
   567	  One 1024×1024 matrix computed once from vectors that already exist; no bank, no mining, no new
   568	  data, no serving change — the head stays linear and fastembed-exact. **Why it is not a
   569	  reparametrisation:** plain L2 is reconstruction, which spends a rank-limited output equally on
   570	  every direction of stella's query space; weighting by document covariance spends it on the
   571	  directions in which documents actually differ, which is what nDCG rewards. Under the rank limit
   572	  PLANNING §9 measured, that is a different subspace, so it is genuine new behaviour rather than an
   573	  absorbable transform. It is the only arm in the plan aimed at the 94% → 98% in-distribution gap.
   574	- **One mechanism note on D-NORM, so its result is not over-read:** on unit vectors the gradient of
   575	  ‖t−s‖ has unit magnitude per example, so a poorly-fit (under-covered) form receives *less* relative
   576	  gradient than under squared L2. Under a coverage thesis the sign of its effect is not obvious in
   577	  advance, which is why it is screened rather than adopted.
   578	- **Optimizer:** AdamW β=(0.9, 0.999), eps 1e-8, wd 0.01 on dim>1, clip 1.0. **Batch 32 examples**
   579	  [family E].
   580	- **Schedule:** 3 cycles of equal example count, each linear 1e-4→1e-5; 2,000 warmup steps in
   581	  cycle 1. Evaluation at every cycle end (annealed) and at cycle midpoints (curve watch only).
   582	- **Init [family C]:** bge-small (default) or the M9 candidate.
   583	- **Dose:** **200M examples** registered — LEAF's dose (6.7M texts × 30 epochs ≈ 201M; PLANNING §5),
   584	  three cycles of 66.7M. Tokens follow the mix: at 75/25, 150M × ~35 + 50M × ~230 ≈ **16.8B**; at
   585	  50/50 ≈ 26.5B. Query epochs ≈ 37 over 4.0M texts, document epochs ≈ 8 over the 6.15M pool. At the
   586	  **measured** bucket rates (718 query / 596 document examples/s, blended **683** at 75/25; PLANNING
   587	  §11, on the box) that is ≈ **81 GPU-hours**, and ≈ 139 h in M9's two-chunk collate — so
   588	  **length-bucketed single-chunk batching is part of the build, not an optimisation** (amendment A7),
   589	  which is why the mix is a share over a 4-step window (§Data). The day-one benchmark's rate replaces
   590	  both, and family E's batch-32 penalty is recorded in the lock as the build's GPU-hour line. **Extension:** let m_k be the COV macro (full precision, the locked formula and
   591	  evaluation hashes) at the end of cycle k. After every cycle k ≥ 3, one more cycle of 66.7M examples
   592	  (linear 1e-4→1e-5, as cycle 3) starts iff m_k − max(m₁, …, m_{k−1}) ≥ 0.003 **and** the extension
   593	  cycles already run are fewer than `max_extension_cycles`, an integer the lock fixes from the
   594	  approved dollars minus every mandatory line of **this file's §Compute table** at the measured rates
   595	  and the billed price (PLANNING §5–§6 are superseded evidence, Opus M12). Whole cycles only; a cycle whose projected cost plus billed spend to date would
   596	  exceed the ceiling does not start. The lock records m_k's formula, the evaluation hashes,
   597	  `max_extension_cycles`, and the spend source.
   598	- **Kill:** non-finite loss/grad; two consecutive scheduled evaluations more than 0.0056 below the
   599	  best evaluation *of their own kind* (midpoint against midpoints, cycle end against cycle ends), so
   600	  the rule can fire inside the build and not only at its end (Opus M5). **Plateau** is read
   601	  best-to-best on annealed checkpoints only; **the plateau rule fires when the improvement test
   602	  `m_k − max(m₁…m_{k−1}) ≥ 0.003` fails at a cycle end k ≥ 3**, independent of the cycle cap (Opus M6;
   603	  Codex 2026-09-04).
   604	- **Seeds:** one shipping seed for the build (seed 0); confirmation seeds at screen dose per §Screen.
   605	  **No full-dose replica runs** (decision 8 withdrawn, amendment A5); the replication band is the
   606	  selected recipe re-trained at screen dose under two further seeds, reported descriptively.
   607	
   608	## Screen — seven families, fifteen arms, thirteen contrasts, locked at M10.0-e (the registry is authoritative: C and L12 cut, A's STOP removed, 14 trained arms, 12 contrasts — §Amendment 2026-09-05)
   609	
   610	**Screen dose = 5M examples** (2.5% of the build; ≈ 420M tokens at 75/25; ≈ 2.0 GPU-hours per arm at
   611	the measured blended 683 examples/s), full 3-cycle schedule compressed to that dose, one seed, identical
   612	evaluation: **COV at every cycle end; DEV-6 once, at the final checkpoint** (its 5.2M-document
   613	hotpotqa and 6.17M-row heldout components cost ~13 GB of reads per pass — M9's practice).
   614	Throughput is recorded for every arm and decides nothing except family E (below).
   615	**Order (A6, amended after the Opus pass): F → A → G → B → E → C → D**, and family F carries a higher
   616	dose than the rest (§F row) because every later verdict — family A's stop rule included — is taken on
   617	its winner (A4 at 5M is the winner's 5M checkpoint from F's curve; F's ≈17 GPU-h are spent before A
   618	can stop the milestone, accepted). Student first, then the data thesis, then the two architecture/regime questions, then init and objective; every family after F
   619	runs on F's winner, so no verdict transfers to the build student by assumption.
   620	**Anchor** = the full M10 corpus (A4's data), mix 75/25, bge-small init, squared L2, bs 32, 1152-d
   621	feature. Screens run **on the box** (amendment A7).
   622	
   623	| family | arms | contrasts | rule and default |
   624	|---|---|---|---|
   625	| **A — data (the thesis)** | A1: M9 pool (463,314 queries) · A2: M9 pool + PAQ (factoid forms only — the volume control) · A3: A2 + the **harvested real** query-like text · A4: A3 + the **generated** forms (the full M10 corpus, = anchor). **A2, A3 and A4 are cut to the identical post-screen unique-text count** (the smallest of the three after decontamination, the larger two downsampled with seed 0) and all hashes are locked before any arm | **A3−A2** (forms from real text, at equal volume) · **A4−A3** (what generation adds over harvesting) · A4−A2 and A2−A1 descriptive | **Three registered outcomes on A3−A2** (the forms contrast, now carried by the real-text arm): corrected lower bound > MDE → coverage **resolved on the COV families**, build proceeds; point ≥ MDE and lower bound > 0 but ≤ MDE → **positive, not resolved**, build proceeds and the report says so; otherwise → **M10 stops before any build and returns to Dylan with all four rows**. **A4−A3 decides whether the generated half is in the build at all**: if it does not resolve, the build uses A3's corpus and the ≈1.0M generated queries are dropped from the build (they stay in the report as a measured null). A2−A1 is the volume effect; if it resolves, the build keeps volume as well as forms |
   626	| **F — student** | bge-small (34.5M with the head) · MiniLM-L6-v2 (23.9M) · **MiniLM-L12-v2 (33.4M with the head)** — the first two at **20M examples each, read as a curve at 5M / 10M / 20M**; L12 at 5M | 2: L6−bge-small at 20M · L12−(winner) at 20M **if L12 is extended**; the count is thirteen whether or not it runs | **Amended after the Fable review, made executable after the Codex pass.** The old rule ("bge-small only if it wins resolved") pre-decided MiniLM-L6 by construction, because M9's same contrast was −0.0026 *unresolved* and a 5M screen cannot resolve 0.003; that also silently skipped family C, since only bge-small has an M9 warm start. So F gets the dose its consequence deserves — it picks the build student and every later verdict — and a third arm, because arXiv 2306.11550's depth curve (1/2/4 layers → 86.1/92.5/96.2% retention) says depth buys retention while LEAF reached 97.7% on 6 layers, and L12 is 12 layers inside the cap. **Rule:** L12 is a 5M *elimination probe* — extended to 20M iff `m_L12(5M) ≥ max(m_L6(5M), m_bge(5M)) − 0.0056`, else eliminated and reported; among the arms with a 20M reading the winner is the highest COV macro **if its margin over the runner-up resolves under the contrast rule**; otherwise the cheapest to serve wins, and **that tie-break is labelled in the report as a product preference, not evidence** (Dylan 2026-09-01 called 33M "the upper bound of what I think is acceptable", so 33M is admissible). L12's three- and four-layer heads pass the parity check first. Cost ≈ 17 GPU-hours at the measured rate, on the box |
   627	| **G — output width and head form** | feature = last layer only (384, M9's head) · three layers (1152, = anchor) · four layers (1536, §Recipe) · **G-MLP**: the anchor's linear head plus a per-token rank-192 GELU correction, `W_lin·x + W₂·GELU(W₁·x+b₁)`, W₁: 1152→192 (B3, residual form after the Codex pass; replaces the 768 arm, whose step the head-width probe already shows — the paper reports 384 vs 1152 vs 1536, not a trained curve) | 1152−384 · 1536−1152 · **MLP−1152** | resolved winner; **default 1152** (PLANNING §9–9b; the screen, not the probe, decides). The 384 arm is the paper's evidence for the M9 diagnosis. **G-MLP** keeps the linear path's full rank and adds the one nonlinearity the cap allows (34.96M for bge-small); fastembed serves it exactly because it precedes the mean pool (`results/m10_head_mlp_parity_box.json`, min-cos 0.99999989, zero custom ops). **Warm start, exact and deterministic:** `W_lin` = the anchor's ridge head; `W₁,b₁` = the top-192 principal directions of the frozen backbone's *per-token* 1152-d states on the fit set, centred (`b₁ = −W₁μ`; sign of each direction fixed so its largest-magnitude component is positive); `W₂,b₂` = ridge from the pooled `mean_t GELU(W₁x_t+b₁)` to the residual `t − W_lin·x̄` — exact for the training form because pooling commutes with the linear maps, so G-MLP starts *at* the anchor's fitted head plus a fitted correction; all three solves share one fit sample — `m9/registry.json` `warm_start`: n_fit 60,000, seed 21, λ reselected on the M10 sample by the registry's locked grid on a training-only holdout (ties to the larger λ) — and the per-token PCA is a streamed 1152×1152 Gram matrix |
   628	| **B — mix** | 100/0 · 50/50 query/document (75/25 = anchor), **matched query presentations** (3.75M query examples in every arm; document examples 0 / 1.25M / 3.75M on top; totals 3.75M / 5M / 7.5M, 4-step window patterns 4Q · 3Q+1D · 2Q+2D; the document cost in tokens and GPU-hours is reported) | 100/0−75/25 · 50/50−75/25 | resolved winner; default 75/25 |
   629	| **E — batch** | 32 · 128 at equal examples and identical schedule | 1 | resolved winner; default 32 (LEAF). **Amendment A7: E is the one family whose throughput is read** — bs128 measured 1,517 examples/s blended against bs32's 683 (`results/m10_rate_bench_box.json`), so a bs32 win must also be worth its 2.2× build cost; the lock records both the quality contrast and the GPU-hour delta, and a bs32 win that does not resolve reverts to bs128 |
   630	| **C — init** | bge-small (or MiniLM, per F) · the M9 candidate | 1 | resolved winner; default the off-the-shelf backbone. Available only if F selects bge-small: M9's candidate is a bge-small student, so a MiniLM build has no warm start and C is skipped and reported as skipped. The closed-form ridge head warm start (M9's `m9s1c`, +0.0272) is retained in every arm regardless |
   631	| **D — objective** | anchor (squared L2) · **D-NORM** (LEAF's ‖e‖₂) · **D-COV** (document-covariance-weighted regression, §Recipe) | D-NORM−anchor · D-COV−anchor | resolved winner with the larger margin; default squared L2. The ranking-aware class is cut (amendment A1, §Recipe); **D-COV is added from the Fable review as the one cheap lever aimed at the in-distribution ceiling** (PLANNING §13 idea 8) |
   632	
   633	A2 exists only as a control; the build never uses more than 1.0M PAQ. **Equal examples** holds for
   634	every family except B, which is matched on query presentations by design. **Definitions:** a
   635	decision's *margin* is the COV macro difference between the winner's and the default's final
   636	checkpoints in the original screen; an arm's *seed range* is max minus min of its COV macro over
   637	its three seeds.
   638	
   639	**Rule, per contrast (families B–G):** the difference in COV macro (family-weighted, §Surfaces)
   640	between the two arms' final checkpoints; paired stratified bootstrap over queries within component,
   641	**B = 200,000**, seed 0, empirical quantile (`inverted_cdf`; at 0.025/13 that is the 384th order
   642	statistic — at B = 20,000 it would have been the 38th, Opus M9); a contrast **resolves** when the point estimate ≥ the **MDE 0.0056** **and** the
   643	one-sided lower bound at the **0.025/13 quantile** (Bonferroni over the thirteen contrasts — F 2 ·
   644	A 2 · G 3 · B 2 · E 1 · C 1 · D 2 — fixed whether or not F's second comparison runs; Codex
   645	2026-09-04 corrected a count of fourteen) is > 0, and the sign is stable across the last two
   646	cycle-end checkpoints. **Both constants are fixed here** — the Codex pass of 2026-09-04 struck
   647	amendment A4's `MDE = max(0.0056, distance)` and its two-step remedy (admit LEDGER, then α = 0.05)
   648	as an α that adapts to an observed width. The resolution number (§Surfaces) is *reported* as the
   649	power disclosure; an underpowered contrast is reported as unresolved and reverts to default, and
   650	the report names the contrasts the surface could not have resolved. **Family A's contrasts A3−A2 and A4−A3 are exempt from the
   651	generic rule** and use the three-outcome rule in the table (resolved requires the corrected lower
   652	bound > MDE); A4−A2 and A2−A1 are descriptive. **Confirmation:** for every decision whose
   653	non-default option won, both the winner and the default are re-trained with two more seeds at
   654	screen dose — **at most two such decisions, largest margins first** (amendment A5; it was four);
   655	the rest revert to default. The decision stands only if the winner's margin exceeds the largest
   656	seed range observed in either arm. Worst-case confirmation cost is now 45M examples ≈ 17
   657	GPU-hours, plus the synthesized selected-recipe arm (5M) and the replication seed pair (10M);
   658	PLANNING §5 has the arithmetic. Every screen verdict is artifact-specific at screen dose; never
   659	"resolved" in the report's sense.
   660	
   661	## Surfaces
   662	
   663	- **COV** — the primary selection surface: **qrel-bearing** retrieval components with an
   664	  affirmative licence at the dataset's primary source (M7's eval-use standard), decontaminated,
   665	  and **read by no M10 decision**, admitted at M10.0-d, **weighted equally per family**
   666	  (slices within a family averaged first). Candidates, draft records in `m10/COV_CANDIDATES.md`
   667	  (primary-source licences checked 2026-09-01): **consumer-health** MTEB MedicalQARetrieval (CC BY
   668	  4.0 at MedQuAD; NIH sources); **BRIGHT**, one family, six slices (biology, earth-science,
   669	  economics, psychology, robotics, sustainable-living; CC BY 4.0 on the benchmark at its primary
   670	  source — the same dataset-level standard that admitted CQADupStack and the six, whose documents
   671	  are also third-party text; the document-rights caveat is disclosed and the data is evaluation-only,
   672	  never redistributed); **legal** MTEB LegalBenchCorporateLobbying (CC BY 4.0; ConsumerContractsQA
   673	  refused, CC BY-NC); **finance** LEDGER (CC BY 4.0 annual-report QA) once its structure and a
   674	  100K-chunk cap are verified. Climate-FEVER refused (no licence at its primary source, as in M7).
   675	
   676	  **Admission reopened 2026-09-04 (Dylan: non-commercial licences are admissible for validation, not
   677	  training — `research/m7-data-licensing.md` §Rule change 2026-09-04).** A set refused *for its licence
   678	  alone* is re-admissible, and the resuming session must re-run admission before the family floor is
   679	  judged: **MTEB ConsumerContractsQA (CC BY-NC) is admissible**, restoring a second legal component;
   680	  the family count stays three without LEDGER (one legal family). Any other CC BY-NC
   681	  or research-only eval set is now in scope. **Still refused:** Climate-FEVER (no affirmative grant —
   682	  a different class), and anything excluded for contamination. MS MARCO itself is admissible but is a
   683	  **poor COV member** — every comparator trains on it and neither of ours does, so it is biased
   684	  against us; prefer it for the within-system read under FORMS-12.
   685	  **Family IDs: `consumer-health` (MedicalQA), `BRIGHT`, `legal` (CorporateLobbying and
   686	  ConsumerContractsQA, one family) and `finance` iff LEDGER admits; CUREv1 and `arxiv-title` are
   687	  diagnostics and never count. At least three IDs must admit** or M10 returns to Dylan — without LEDGER
   688	  the count is exactly three (Codex 2026-09-04 corrected an earlier "four"); the report names it. The two CQADupStack components are **DEV**, reported beside every COV
   689	  read, never in the macro.
   690	

exec
/bin/bash -lc "nl -ba instructions-m10.md | sed -n '690,879p'
nl -ba m10/PLANNING.md | sed -n '260,355p'" in /home/dylan/asymetric-dual-encoders
 succeeded in 0ms:
   690	
   691	  **Scientific diagnostic surface (B4; a COV family until the Codex pass, an actioned secondary until the Opus pass).** Every clean-4
   692	  set is scientific or biomedical and no licensed published set covers those forms outside S2ORC /
   693	  PubMed / CORD-19, so as registered the screen could not see the headline forms. **`arxiv-title`**:
   694	  2,000 held-out arXiv titles → their own abstract among 100K arXiv abstracts (CC0 metadata; ≈8 min
   695	  of encode), one relevant per query, nDCG@10, teacher score reported as the denominator. **Not in
   696	  the COV macro** — it shares form and source with arm A3's harvest, so as a family it would reward A3
   697	  by construction and its 2,000 low-variance queries would dominate the macro's power. It is a
   698	  **descriptive diagnostic with no action** (Opus M11: title→own abstract is near-lexical known-item
   699	  retrieval, so a null there could be a floor effect, and it is not scidocs' title→cited-abstract
   700	  relation); A3−A2 on the COV macro carries every data decision. Reported per arm beside FORMS-12 as
   701	  the one real-text scientific read. **Draw, registered (Codex B8):** the Kaggle arXiv metadata artifact (`Cornell-University/arxiv`,
   702	  the version downloaded on day 1; its file sha256 recorded in `m10/LEDGER.md` §0b before the draw);
   703	  ID universe = every record's `id` with the version suffix stripped, deduplicated, sorted
   704	  lexicographically; `numpy.random.default_rng(0).choice(N, 100_000, replace=False)` in that order;
   705	  the first 2,000 drawn are the queries (title → own abstract), the other 98,000 distractors; every
   706	  version of a drawn paper is excluded from every training role; the harvest is fingerprint-screened
   707	  against the drawn titles and abstracts; the set joins the protected index before any extraction.
   708	  It is drawn at the start of the harvest step, once the artifact exists.
   709	  ClinicalTrials.gov is not used (terms clause unread). COV still has no argument retrieval
   710	  (six-set only; FORMS-12 descriptive). **Decision 12 (adopted)**: CUREv1 as a validation-only diagnostic.
   711	  **Resolution number (M10.0-d, before the lock) — a power disclosure, nothing more** (A4's sizing
   712	  struck by the Codex pass). With the contrast rule's own bootstrap, measure the distance between the
   713	  point estimate and the one-sided 0.025/13 lower bound for the COV-macro difference between two
   714	  models that are candidates in no M10 family — **e5-small-v2 and gte-small** (`results/FINAL_MATRIX.md`
   715	  rows, fresh COV encodes) — scored symmetrically on the admitted surface. Only the distance is
   716	  recorded, never which led. It is the first disclosed COV read (`m10/LEDGER.md` §4) and is
   717	  published beside every contrast, so a reader can tell an unresolved verdict from an invisible one.
   718	  With MedicalQA 2,048, BRIGHT 6 × ~100, CorporateLobbying 340, ConsumerContractsQA 396 and paired
   719	  per-query SDs of 0.07–0.27, the expected distance is 0.009–0.0135 — above the MDE, so **most B–G
   720	  contrasts are expected to be unresolved at screen dose and to revert to default; that is disclosed
   721	  in advance, not repaired by loosening α.** LEDGER (118,048 questions) is admitted if its structure
   722	  verifies at M10.0-d, an ordinary admission decided before any read, because it is the one
   723	  candidate large enough to move the surface's power. The distance is measured between *unrelated*
   724	  models, so it over-estimates the width of a same-init contrast; recorded, not corrected.
   725	- **DEV-6** (incl. the two CQADupStack components) secondary, reported beside every COV read.
   726	  SCREEN-3 is retired.
   727	- **FORMS-12**: 12 × 500 held-out generated or harvested queries, overlap@10 between student and
   728	  teacher rankings over a **1M-document evaluation sample of the already-encoded pool** (seed 0,
   729	  fixed at M10.1) — amendment A1 deleted the *mined candidate bank*, not the evaluation sample, and
   730	  this needs no mining. Per form. **Descriptive only** — teacher agreement on generated
   731	  queries is a coverage diagnostic, not retrieval quality.
   732	  **Now also permitted (2026-09-04, optional, descriptive):** the same overlap@10 diagnostic on a
   733	  sample of **real MS MARCO dev queries**, as an external check that the 12 forms cover the natural
   734	  query distribution. Validation only: MS MARCO text may **never** seed generation or enter the pool,
   735	  and no MS MARCO-derived cache may be written under `work/train/sources/` (`m7src/mix.py:22-25`).
   736	  **The stella-space distribution-overlap gate of amendment A8 (§Data) is the primary use of this
   737	  permission** and runs on the manifest before any arm; this overlap@10 row is the post-hoc version
   738	  of the same idea and stays optional. Neither is a substitute for COV: both are teacher-agreement
   739	  or distributional diagnostics, not retrieval quality.
   740	- **LoTTE-clean** (7 slices, macro over slices; its corpora — ~2.8M passages — are encoded with
   741	  stella once, ≈ 1.3 GPU-hours at the assumed 600 docs/s, budgeted in PLANNING §6): **read #1**
   742	  after the recipe lock.
   743	  Before LoTTE opens, the **selected recipe is trained once as a single synthesized arm at screen
   744	  dose** (its checkpoint hash committed); read #1 scores that checkpoint and the anchor's in one atomic batch.
   745	  Veto rule (M9 §7, unchanged): the selection is vetoed if the selected recipe's 7-slice macro is
   746	  worse than the anchor's by more than 0.004 AND the one-sided 97.5% paired-bootstrap upper bound
   747	  (B = 10,000, seed 903, paired within slice) on (selected − anchor) is below −0.004; a veto means
   748	  **the anchor recipe builds**. No other action may follow from read #1. **Read #2** pre-freeze,
   749	  audit only. No third read.
   750	- **Dev→six calibration**: M9's close-out rows (available only after M10.2) and M7's; forecasting
   751	  only, never gating.
   752	- Dev reuse is counted (`m8src/dev_reuse_m8.py`) and published.
   753	
   754	## Final run
   755	
   756	`m9/FINAL_LOCK.md` re-registered verbatim for M10 in the M10.2 lock with new BOUND-AT-FREEZE hashes
   757	and an `m10-six-spent` tag — bridge as phase 1, the C-conjuncts with the empirical bootstrap bound
   758	and the sign-flip conjunct, FEVER labelled — **with four changes (Opus: "two" understated it). (1) The
   759	reserved conditional is `if any C-conjunct passes then execute`**, so an aim claim never stands
   760	without its descriptive reserved rows. **(2) Amendment A3: four conjuncts, not two** — `C1a`/`C2a`
   761	on avg-6 and `C1b`/`C2b` on clean-4 (bars 0.5042 / 0.5155 / 0.5046 / 0.5233), under
   762	**fixed-sequence gatekeeping in the order C1b → C1a → C2a → C2b at the full one-sided 0.025 each,
   763	stopping at the first non-rejection** (§Goal has the procedure, the planning proxies and the
   764	`trec-covid` n=50 disclosure). **(3) The decision field is the empirical one-sided 0.025 quantile
   765	(`lower_q025_raw`, the 250th order statistic at B = 10,000), not M9's Holm-2 0.0125, and (4) the
   766	sign-flip conjunct is tested at 0.025 inside the same sequence** — the Codex pass of 2026-09-04
   767	found the two quantiles named in different files. M9's claim-table row "C1 fail, C2 pass → aim
   768	permitted" is deleted by (2). The sequence is fixed here, before any six-set
   769	output exists.
   770	clean-4 is M14's registered headline partition for both halves of the pair, so it carries a
   771	pre-registered pass/fail rather than arriving as a descriptive row after the fact. Zero alpha on the reserved batch is unchanged. `m9src/final9.py`'s
   772	scoring path is written and reviewed before M9's close-out and reused for scoring; **its `decide()` is
   773	hard-coded to M9's two conjuncts and is NOT reused — an M10 `decide()` for four conjuncts under
   774	gatekeeping is written, unit-tested and reviewed before the recipe lock** (Codex 2026-09-04).
   775	
   776	## Compute and costs — re-priced 2026-09-04 on measured rates (amendment A7)
   777	
   778	**Split execution.** The 2026-09-01 ruling stands where it bites — no dose or screen size is set by
   779	the box's wall-clock — but the box is an execution target again for everything that is not
   780	generation, because the rate it was withdrawn on was wrong:
   781	
   782	| configuration, M10 recipe shape, **measured on the RTX 3080 2026-09-04** (PLANNING §11) | examples/s | 200M examples |
   783	|---|---|---|
   784	| batch 32, 75/25, M9's two-chunk collate | 400 | 139 h |
   785	| batch 32, single-chunk buckets (718 query / 596 document), blended 75/25 | **683** | **81 h** |
   786	| batch 128, single-chunk buckets (2,311 / 748), blended 75/25 | 1,517 | 37 h |
   787	
   788	The plan's imported LEAF planning rate was 560 examples/s on a rented A100; the box meets or beats
   789	it, because at batch 32 with ~35-token queries the step is launch-bound, not FLOP-bound, so a
   790	bigger card buys ~1.5–2× rather than 10×. Caveat, disclosed: these are fixed-shape random-token
   791	microbenchmarks with no data loading and no evaluation, so the real trainer is slower; they bound
   792	the *hardware*, not the pipeline. **Consequences, all registered here:** length-bucketed
   793	single-chunk batching is part of the build (§Recipe); the screens, confirmations, COV admission and
   794	encodes, and M10.0-c run **on the box**, which already holds ~200 GB of M9 caches the plan had
   795	budgeted 12 GPU-hours and a day of network to re-derive; the day-one benchmark still re-derives
   796	PLANNING §6 before anything scales.
   797	
   798	**What the money buys.** Generation runs on the box (decision 14, Qwen's official AWQ release), so
   799	the build is the cloud's job — rented for wall-clock, not because the box cannot do it (the box would take
   800	≈ 3–6 days at the measured rates). One rented **A100 80 GB** (H100 if its cost per example measures
   801	lower on the smoke), ≥ 500 GB persistent disk that survives stopping the instance, SSH, and a GitHub
   802	deploy key so the headless commit-and-push contract holds. Provider is Dylan's choice; $1.5–2.5/h
   803	assumed, unverified Sept 2026. The instance is stopped between stages.
   804	
   805	| line | GPU-hours | $ at 1.5–2.5/h |
   806	|---|---|---|
   807	| generation, ≈1.0M queries — **on the box under decision 14** | 0 | 0 |
   808	| build, 200M examples at the measured rates (bs128 → bs32) | 37–81 | 55–205 |
   809	| cloud-side encodes the build needs + export, parity, final run | 6 | 9–15 |
   810	| day-one rate benchmark | 1 | 2–3 |
   811	| persistent disk, egress | — | ≈ 25 |
   812	| **cloud total (screens and generation on the box)** | **44–88** | **≈ $90–245** |
   813	| all-cloud variant (screens and confirmations rented too) | 130–190 | ≈ $220–500 |
   814	| optional: extension cycle, 66.7M examples, each | 13–25 | 20–63 |
   815	
   816	Hard ceiling **$1,000**, unchanged and now far from binding. **Allocation order at the lock:** every
   817	mandatory line first at the measured rates and the billed price, then whole extension cycles from the
   818	remainder (the second build seed is withdrawn, amendment A5). A cycle whose projected cost plus
   819	billed spend to date would exceed the ceiling does not start.
   820	
   821	**Day-one rate benchmark (≈ 1 GPU-hour, before any scale-up):** stella docs/s on 20K pool documents;
   822	training examples/s at batch 32 for 2,000 steps on the 75/25 anchor mix, the 50/50 mix and MiniLM-L6;
   823	generation requests/s per form on the 200-query smoke, end to end (prefill, decode, retries, JSON
   824	failures); the provider's billed price. Wall-clock ≈ 1.5–2 weeks; screen arms are independent and the
   825	box runs them while the cloud generates.
   826	
   827	On the instance, `results/perquery.json` (tracked) is sha-verified against `6b18e3dd…` before any
   828	scoring, and M9's parity sample is regenerated from `m9/registry.json` (tracked; seed and sha256
   829	pinned) — `m9src/port.py` refuses a sample that does not hash to the lock, in which case the 512
   830	texts are transferred from the box. Only what training needs travels to the instance (tokenized
   831	corpora and teacher targets, or re-derived there from the repo and HF); nothing that only the
   832	evaluation stages need. Reserved-set encodes run only if the reserved conditional fires.
   833	
   834	M9's cost protocol (`instructions-m9.md` §Costs) unchanged; the frontier is reported per index
   835	configuration, naming the one measured (`m9/RESULTS.md` rounds 1–4). TurboQuant 4-bit (Qdrant 1.19)
   836	joins the M13 all-in quantization comparison.
   837	
   838	## Unimpeachable by competitors — what the report must carry
   839	
   840	1. Comparators byte-verified against the official artifacts, scored in the same exact-search
   841	   harness, frozen as per-query vectors before any M10 number existed; the bridge check proves
   842	   the harness unchanged.
   843	2. Pre-registration with pushed commits as the external witness; one six-set transaction per
   844	   milestone, each disclosed (M7, M9 close-out, M10); the reserved four's single access; dev-reuse
   845	   count and the selection-surface table published.
   846	3. Contamination handled three ways: stella's disclosed overlap (ArguAna, FiQA) at every headline;
   847	   NDO-4 rows; reserved NDO-3 as the surface no decision touched. **LoTTE read #1 is decision-bearing
   848	   (the veto), so LoTTE is not untouched, and read #2 is an audit, not independent confirmation.**
   849	4. Training data affirmatively licensed, no MS MARCO, attribution recorded; synthetic queries from
   850	   an open-weights generator with pinned revision, per-query provenance and removal counts.
   851	5. Compute disclosed in examples, tokens and GPU-hours beside LEAF's ~100 A100-hours.
   852	6. System-level framing only (§Goal): no claim isolates nano from its document tower.
   853	7. Reproducibility: code, corpus manifests with hashes, model revisions, seeds, statistics code
   854	   with tests, and the full screen table including losing arms and confirmation seeds.
   855	8. **No general retrieval-quality language.** The six are development-informed by construction (the
   856	   data taxonomy targets their forms; COV selects; LoTTE vetoes); the reserved four are the only
   857	   untouched surface and are descriptive. Every conclusion is stated on the named datasets, and the
   858	   released artifact is labelled **"selected on development-informed datasets; the reserved four are
   859	   descriptive"** (Codex 2026-09-04, twice).
   860	
   861	## Deliverables
   862	
   863	Frozen candidate + `m10/FREEZE.json` (`assert_releasable` with a proper run record — it walks
   864	`work/runs/<id>.json` for `cfg.sources` and `cfg.init` across the whole lineage and fails CLOSED on
   865	a missing ancestor, which is why M9's freeze was refused; the record must be **derived from the
   866	run's own hash-bound `manifest.json`, never asserted by hand**, because a post-hoc record that
   867	happens to satisfy a licence guard is precisely the artifact that must not be fabricated; see
   868	`m9/BUILD_LOG.md`), the frontier
   869	update, the M10 section of the report artifact, decisions logged in CLAUDE.md, handoff to M13.
   870	
   871	## Out of scope (reopening conditions in PLANNING §7)
   872	
   873	Document-side co-adaptation (inside M10 it breaks the pair; a tower co-trained against both query paths at once keeps it and is recommended 2026-09-01 as the next-milestone candidate (that slot was M12 then; **M16** after the 2026-09-04 renumbering), Dylan's call, PLANNING §7) · any student above 35M
   874	(hard cap, Dylan 2026-09-01) · teacher change (stella-1.5B measured worse; Qwen3-0.6B never
   875	screened and not the pair) · a **post-pooling** nonlinear head (no fastembed path; per-token heads
   876	ship, §Recipe) · a cosine-space phase-1 loss (closed by algebra, `m10/EXPLORED.md`) · a
   877	ranking-aware phase-2 loss (cut by amendment A1; reopening condition in §Recipe) · MS MARCO **in any training role** (unchanged; **validation/diagnostic use was
   878	permitted 2026-09-04**, see the admission note in §COV) · FineWeb in any role (decisions 3 and 5) · any
   879	change to zero.
   260	
   261	**Pass 6 — verification of the pass-5 fixes and the trim** (`research/m10-codex-plan6-2026-09-01.md`;
   262	read-exclusion audited clean). 8 of 10 land; the trim removed nothing a box session needs. Two
   263	items remain as **recorded dissents**, plus one contradiction fixed:
   264	
   265	| # | finding | disposition |
   266	|---|---|---|
   267	| B | BRIGHT admitted although its third-party documents' rights are not conveyed | **not adopted, dissent recorded** — the reviewer's per-document standard would also disqualify the six and CQADupStack; the plan applies M7's dataset-level eval-use standard consistently, discloses the caveat, and uses COV for selection only, never for a claim. Owner may overrule (`m10/COV_CANDIDATES.md`) |
   268	| B | COV floor three, reviewer wants four | **not adopted, dissent recorded** — with the CQA pair demoted, four untouched families exist only if LEDGER admits; the floor stays three, the report names the family count, and Dylan may raise it |
   269	| M | mandate said M9 "is merged after the close-out cleanup" while STATUS says merged with close-out pending | **adopted** — one wording: merged 2026-09-01 after the repo cleanup; close-out pending from `m9-work` |
   270	
   271	**Owner amendment 2026-09-01, after pass 6 — the compute ruling and the review taken with it.**
   272	Dylan: "M10 won't be done on a 3080. M10 will be done on a GPU budget, if allowed, or not at all."
   273	The same-day review of the plan against the goal (a best-in-class asymmetric pair) found four
   274	places where the box, not the evidence, had set a number; they changed together:
   275	
   276	| # | change | why | where |
   277	|---|---|---|---|
   278	| 1 | box path withdrawn; one rented A100; itemized budget, ceiling $1,000 | the ruling | §6, mandate §Compute, decision 2 |
   279	| 2 | build dose 50M → **200M** examples; extension while a cycle end gains ≥ 0.003, capped by budget | LEAF's dose is 201M on an easier target (768-d, 109M teacher); the old cap was the box's days | §5, mandate §Recipe |
   280	| 3 | screen dose 2.5M → **5M** | 2.5M was 5% of the old build; a 5% screen of 200M would cost ≈ 70 GPU-hours | §5, mandate §Screen |
   281	| 4 | family D gains **D-KL1** and **D-NCE**; family G gains **1536**; 16 contrasts, 0.025/16 | a ranking signal only in cycle 3 leaves §9's own caveat untested; the seed passage is a free positive (CARE stage 1, EmbedDistill); §9b is still rising at three layers and a fourth costs 0.39M parameters | mandate §Recipe, §Screen |
   282	| 5 | **COV resolution number** before the lock (descriptive since pass 7) | MedicalQA 2,048 documents, CorporateLobbying 340 queries, BRIGHT ~100 per slice: an MDE-sized contrast may be unresolvable, and the report must say whether an unresolved verdict was invisible | mandate §Surfaces, M10.0-d |
   283	| 6 | seed-rank provenance field | a round-trip filter without a second generation pass | mandate §Data |
   284	| 7 | generator in bf16; 4-bit and Qwen3-4B fallback withdrawn; hosted fallback if the end-to-end projection exceeds 60 GPU-hours | an 80 GB card | mandate §Data, `m10/COV_CANDIDATES.md` |
   285	| 8 | decision 8: second build seed inside the ceiling | the headline's CI is a query-sampling interval only (`m7/FINDINGS.md` 9) | mandate decisions |
   286	| 9 | co-training the tower recorded as the next-milestone candidate (now M16) | every ≥ 96% near-zero-query system co-trained the document side | §7 |
   287	
   288	**Pass 7 — gpt-5.6-terra, high effort, read-only, on the amendment**
   289	(`research/m10-codex-plan7-2026-09-01.md`; read-exclusion audited clean: the reviewer opened the
   290	named plan files, `m9src/guard9.py`, `m9/PLANNING.md` §3, `m7/FINDINGS.md`, and ran one web search
   291	on A100 vLLM throughput). Verdict "not decision-grade"; 2 BLOCKER / 6 MAJOR / 1 MINOR / 5
   292	contradictions. **All actioned:**
   293	
   294	| # | finding | disposition |
   295	|---|---|---|
   296	| B | the COV resolution check scored bge-small and MiniLM — family F's own backbones — and made a COV-read decision (screen scope) before selection | **adopted** — demoted to a descriptive **resolution number**: e5-small-v2 and gte-small (candidates in no family), distance only, no direction, first disclosed COV read, cuts nothing; the screen always runs fourteen arms |
   297	| B | the hosted-generation branch was outside the budget table; at the high end seed 1 and extensions could not fit under $1,000 | **adopted** — §6 carries scenario A (GPU generation, $400–715) and scenario B (hosted, $465–895), an explicit allocation order, and the statement that at the high end nothing optional fits |
   298	| M | D-NCE not executable: logits, denominator, positive's template, reduction, τ reuse, missing seeds | **adopted** — the literal 129-way softmax cross-entropy is in §Recipe, τ reuse disclosed, removed-seed fallback to L2 counted |
   299	| M | extension rule not mechanical: baseline undefined, partial cycles, dollar→hour conversion, no ledger fields | **adopted** — m_k − max(m₁…m_{k−1}) ≥ 0.003, whole cycles, `max_extension_cycles` fixed at the lock from dollars minus mandatory lines at measured rates, hard stop on projected spend, ledger fields |
   300	| M | decision 8 contradicted "full-dose replicas stay waived"; "if the lock says so" a loophole; STATUS weaker | **adopted** — waiver sentence replaced; seed-1 boolean, seeds and row labels locked before any seed-0 six-set output; seed 0 alone controls every action; STATUS repeats the ≥ 100 GPU-hour criterion |
   301	| M | generation budget assumed 60 output tokens for every form; two forms allow 400 | **adopted** — ≈ 250M generated tokens (350M at caps), 30–60 GPU-hours, fallback gate on the end-to-end projection at 60 h |
   302	| M | parity sample and `results/perquery.json` had no cloud source or hash check | **adopted** — perquery.json is tracked and sha-verified on the instance; the parity sample regenerates from `m9/registry.json` (tracked, sha pinned) and `port.py` refuses a mismatch, in which case the 512 texts transfer from the box |
   303	| M | rates provisional; measuring only the first arm is too late; 46 h understated | **adopted** — a day-one rate benchmark (encode, three training mixes, generation smoke, billed price) re-derives §6 before generation and again before the remaining arms; 47 h |
   304	| m | STATUS still scheduled a Mac 4-bit generator pass | **adopted** — renamed prompt prototyping that produces nothing entering the smoke record, the data, or the manifest |
   305	
   306	## 9. Rank-bottleneck probe (`m10src/rank_probe.py` + `rank_probe_mix.py`, Mac, 2026-09-01)
   307	
   308	`results/m10_rank_probe_mac.json`. Stella-400M query vectors (s2p prompt) projected onto their
   309	top-k principal components, renormalized, retrieved by exact search against the unmodified stella
   310	document vectors of the two CQADupStack dev components (manifest hashes verified). **The Mac
   311	reproduces the box ceiling**: programmers 0.4681 vs 0.46807, physics 0.4932 vs 0.49314.
   312	A student with hidden width h and a linear head emits pre-normalization vectors in ONE
   313	h-dimensional affine subspace (L2 normalization does not change a ranking, so the subspace is what
   314	matters for retrieval). PCA gives the **reconstruction-optimal** k-d subspace of the fit set — the
   315	subspace an L2-regression objective pushes such a student toward — so the k = h row is the
   316	retention of that subspace, **not an upper bound over every k-d subspace** (a ranking-optimal one
   317	may do better). Retention of the component's own full-rank score:
   318	
   319	| basis fit on | k | programmers | physics |
   320	|---|---|---|---|
   321	| NQ-open questions (20,000) | 384 | 79.8% | 89.3% |
   322	| NQ + the other forum component | 384 | 82.7% | 92.0% |
   323	| NQ + other + **the target's own queries** (oracle mixture) | 384 | 90.4% | 93.1% |
   324	| the target's own queries only (single-distribution oracle) | 384 | 99.5% | 99.6% |
   325	| NQ + the other forum component | 512 | 92.5% | 97.4% |
   326	| NQ + the other forum component | 640 | 97.9% | 100.2% |
   327	| NQ-open questions | 768 | 99.4% | 100.6% |
   328	
   329	Explained variance of NQ queries at 384 components: 84.3% — stella's query space is not
   330	low-rank. Stella's 768-d and 256-d MRL heads do not help (768-space k=384: 82.1% / 89.2%).
   331	
   332	**Reading.** The reconstruction-optimal 384-d subspace serves ONE query distribution almost
   333	perfectly (99.5% / 99.6%) and several distributions poorly: fit on NQ it retains 80–89% on forum
   334	queries; fit on a mixture that includes the target's own queries, 90–93%. The aim needs 89.7% across
   335	six distributions. Under L2 distillation — whose objective is reconstruction — this is **strong
   336	evidence that a 384-wide linear head binds before training starts**, and consistent with M9's
   337	50–71% on the same components; it is not a theorem about every 384-d subspace, and a ranking-aware
   338	loss (phase 2) could pull the student elsewhere. At width 640 the same subspace retains 98–100%.
   339	Within 35M parameters the width comes from the *feature*, not the backbone: mean-pool three layers
   340	(bge-small layers 12, 8, 4) and concatenate → 1152-d feature → Linear(1152→1024), +786,432
   341	parameters over the 384-d head (head 1.18M; ≈ 34.5M total), and still exportable per token so fastembed's own mean pooling
   342	reproduces it exactly (mean pooling is linear; M9's trick; identical masking required). This is
   343	screen family G in the mandate, default 1152; **the screen decides, not the probe.** Caveats: two
   344	forum components stand in for the six; the capacity probe (now optional, report-only) has a
   345	768-hidden student, so any gain it showed would be partly width. Diagnostic; read by no rule; dev reads
   346	counted (2 components, 3 scoring passes each per basis).
   347	
   348	## 9b. Head-width probe in closed form (`m10src/head_width_probe.py`, Mac, 2026-09-01)
   349	
   350	`results/m10_head_width_probe_mac.json`. M9's head-probe design (frozen bge-small backbone, ridge
   351	head to stella targets fit on 20,000 NQ-open questions, λ on a training-only holdout) with three
   352	pooled features, scored on the two CQA components against the cached stella documents:
   353	
   354	| feature | dim | programmers (retention of stella) | physics |
   355	|---|---|---|---|

codex
The main issue is becoming clearer: the screen measures changes around one 5M-example anchor, then applies them to a 200M-example build that may use different data and batch size. It therefore needs evidence that the selected changes survive both combination and longer training. I also found a direct contradiction: FORMS-12 is labelled “descriptive only,” but the plateau rule uses its results to change training weights.
exec
/bin/bash -lc "nl -ba m10/PLANNING.md | sed -n '348,522p'" in /home/dylan/asymetric-dual-encoders
 succeeded in 0ms:
   348	## 9b. Head-width probe in closed form (`m10src/head_width_probe.py`, Mac, 2026-09-01)
   349	
   350	`results/m10_head_width_probe_mac.json`. M9's head-probe design (frozen bge-small backbone, ridge
   351	head to stella targets fit on 20,000 NQ-open questions, λ on a training-only holdout) with three
   352	pooled features, scored on the two CQA components against the cached stella documents:
   353	
   354	| feature | dim | programmers (retention of stella) | physics |
   355	|---|---|---|---|
   356	| mean of layer 12 (M9's head) | 384 | 0.1271 (27.1%) | 0.1750 (35.5%) |
   357	| layers 12 + 8 | 768 | 0.1545 (33.0%) | 0.2022 (41.0%) |
   358	| layers 12 + 8 + 4 (M10 default) | 1152 | 0.1722 (36.8%) | 0.2167 (43.9%) |
   359	
   360	Each added layer helps a frozen backbone, monotonically, by 6 and 4 points on programmers. This is
   361	a floor (no training, NQ-only fit set), not a forecast — M9's trained 384-d student reached 50% /
   362	71% on the same components — and it is consistent with §9: with more output directions the
   363	frozen features already reach further into stella's space. Family G's default is 1152; the screen
   364	decides. Diagnostic, read by no rule; dev reads counted (2 components × 3 features).
   365	
   366	## 9c. Serving parity of the three-layer per-token head (`m10src/head_width_parity.py`, Mac CPU)
   367	
   368	`results/m10_head_width_parity_mac.json`. bge-small with the Linear(1152→1024) head applied per
   369	token over the concatenated states of layers 12, 8, 4, exported at opset 17: **1073 nodes, zero
   370	custom-domain ops, 34.54M parameters** (head 1.18M). fastembed 0.8.0 serves it as a custom
   371	MEAN-pooled normalized model and reproduces the pool-then-head reference to **min-cos
   372	0.99999984, max-abs 2.0e-07** on 64 texts of 1–300 words. **M10.0-a2 passes** (weights are
   373	random; parity does not depend on them). Two serving pitfalls (both hit and resolved in M11, see `m11/CODEMAP.md`): fastembed needs
   374	`config.json` and `special_tokens_map.json` beside the graph, and transformers 5.x fast tokenizers
   375	no longer write the latter.
   376	
   377	## 10. Reuse, do not rebuild
   378	
   379	`m9src/longrun.py`, `watchdog.py`, `guardian.sh`, `sentinel.sh`, `sacrificial.sh` (build and
   380	supervision) · `m9src/final9.py` + `final_stats.py` (access machine, statistics, 16 tests) ·
   381	`m9src/nano.py`, `warmfit.py`, `data.py` (student, closed-form head, pools) · `m9src/eval9.py`,
   382	`m7src/evalkit.py`, `devsuite.py` (scoring) · `m9src/port.py`, `export_doc_model.py`,
   383	`edge_cost.py` (ONNX, fastembed, costs) · `m9src/capacity_probe.py` (unchanged) ·
   384	`m8src/blockcg.py`, `decide.py`, `noise_floor.py`, `paths_guard.py`, `protected_filter`,
   385	`dev_reuse_m8.py` · `m7src/decontam.py` (R1/R2/R3 fingerprints), `fusion.py`, `freeze.py`,
   386	`boot.py`. `m8/CODEMAP.md` and `m9/CODEMAP.md` carry the pitfalls.
   387	
   388	## 11. Measured trainer rates (box, 2026-09-04) — the plan review's central number
   389	
   390	`results/m10_rate_bench_box.json`, script `m10src/rate_bench.py`. Random token ids only; no corpus,
   391	no protected path, no teacher, no data loading, no evaluation — it bounds the **hardware**, not the
   392	pipeline. M10 recipe shape: bge-small + Linear(1152→1024) per token over pooled layers 12/8/4,
   393	34.54M parameters, bf16 autocast, fp32 loss. Pass 2 (cite this one): single homogeneous chunk per
   394	step — what length bucketing produces — 300 timed steps, fused AdamW, peak memory and allocator
   395	retries logged.
   396	
   397	| shape | steps/s | examples/s | padded tok/s | peak GB | retries |
   398	|---|---|---|---|---|---|
   399	| bs32 × 64 (query bucket) | 22.4 | 718 | 45.9K | 0.88 | 0 |
   400	| bs32 × 128 (mixed padding) | 22.0 | 702 | 89.9K | 1.29 | 0 |
   401	| bs32 × 256 (document bucket) | 18.6 | 596 | 152K | 2.11 | 0 |
   402	| bs64 × 64 | 21.1 | 1,350 | 86.4K | 1.29 | 0 |
   403	| bs128 × 64 | 18.1 | 2,311 | 148K | 2.11 | 0 |
   404	| bs128 × 256 | 5.8 | 748 | **191K** | 7.03 | 0 |
   405	| bs512 × 64 | 5.7 | 2,893 | 185K | 7.04 | 0 |
   406	
   407	Pass 1 (60 steps, kept for two rows only): M9's **two-chunk** collate at bs32 gives **400 ex/s**
   408	against pass 2's 702 for the same examples in one padded chunk — the 1.9× that makes length-bucketed
   409	single-chunk batching part of the build. Plain vs fused AdamW: 400 vs 397, so the cost is not the
   410	optimizer.
   411	
   412	**Reading.** Launch-bound at batch 32 is now *established*, not asserted: 4× the tokens per step
   413	(32×64 → 32×256) costs 20% more wall-clock, and 4× the batch at fixed length (32×64 → 128×64) costs
   414	24%. The card's roof is 185–191K padded tok/s, reached only at 128×256 and 512×64; batch 32 runs at
   415	24–80% of it. §5's planning rate — 560 ex/s, LEAF's realized A100 rate — is therefore *met or
   416	beaten by the 3080*, and M9's realized 226 ex/s was a pipeline artifact rather than a hardware
   417	limit. Build cost under length bucketing, 200M examples at the 75/25 example mix (150M query-bucket
   418	+ 50M document-bucket, each at its own rate): **81.3 GPU-hours at bs32, 36.6 at bs128** — bs128 is
   419	2.2× cheaper, which family E must weigh against any bs32 quality win.
   420	
   421	**Corrected after the Fable review (2026-09-04).** Pass 1 read its slow bs512 two-chunk row as
   422	"past the launch-bound regime". Wrong: padded tok/s *fell* to 66K there, and a compute-bound card
   423	plateaus rather than dropping 2.9× below its own roof. Pass 2 shows 512×64 and 128×256 both reach
   424	185–191K at 7.0 GB with zero allocator retries, so pass 1's row was memory pressure on a 10 GB card.
   425	Two consequences: an 80 GB A100 is **not** launch-bound at bs128+, so family E's cost ratio and the
   426	cloud build line are device-dependent and must be re-measured there; and the document bucket — the
   427	shape 25% of build steps will run — was untimed in pass 1 and is the expensive one.
   428	
   429	**The caveat that binds, and it is the reason nothing is committed on these numbers.** These are
   430	hardware bounds. M9's realized pipeline ran at **18,984 tok/s** (`m9/M92_LOCK.md`) against a
   431	comparable-shape roof of ~191K here — roughly **10% efficiency** — and the M10 trainer is being
   432	ported from that pipeline (`m9src/longrun.py`: per-step numpy collate, `length_chunks`, target
   433	fetch). So the build is priced as a **range from hardware bound to M9 efficiency**, and box-window
   434	item 1 (real-data re-measure with `torch.cuda.memory_stats()['num_alloc_retries']` logged) is the
   435	gate before any dollar or box-versus-cloud decision. At M9's efficiency the box build is 300+ hours,
   436	not 81, and a screen arm is 6–9 hours, not 2 — which is why closing that gap is worth more than the
   437	GPU rental (§13, idea 1).
   438	
   439	## 12. Why generate at all, and why ≈1.0M rather than 3.0M (the 2026-09-04 question)
   440	
   441	Dylan: *"I'm not sure why we need to generate synthetic data?"* and *"isn't synthetic data lower
   442	quality?"* Research sweep: one Sonnet subagent, web only, 2026-09-04.
   443	
   444	**The mechanism.** Phase 1 is `‖student(x) − stella(x)‖` — there is no label, and the teacher's
   445	embedding of any text is a correct target by construction. So the classic synthetic-data failure
   446	(a wrong relevance judgment) cannot occur here; the exposure is only (i) distribution shift, and
   447	(ii) diversity collapse. Both are measurable before a training step, which is what amendment A8's
   448	two gates do. Every comparable system is *more* exposed than we are — LEAF, Qwen3-Embedding and the
   449	InPars/Promptagator lineage all pair synthetic queries with implied positives — and they worked.
   450	
   451	**Why real queries cannot substitute.** LEAF's breadth came from MS MARCO 502K (non-commercial),
   452	Amazon QA 979K and CC-News 900K (no commercial grant) and PubMedQA 272K (contaminates NFCorpus and
   453	TREC-COVID). Our permissively-licensed real-query stock is factoid Wikipedia QA plus ESCI product
   454	search — exactly M9's two forms. Generation is a licence workaround, not a scientific preference,
   455	and the report says so.
   456	
   457	**But most of the breadth is harvestable as real text**, which is amendment A2: titles, headings and
   458	declarative lead sentences already sit in the licensed pool, and **three of the four clean-4
   459	headline datasets fall in that half** (scidocs↔titles, scifact↔claim sentences,
   460	trec-covid/nfcorpus↔headings and consumer-health). Generation is then confined to the interactive
   461	forms no corpus contains. Indirect support: arXiv 2502.19712's query-type ablation on TREC DL19/20
   462	has titles-only at 71.6–72.9 against real-MS-MARCO-style-only 72.8 and the six-type mix at
   463	72.4–73.2, while claims-only (a single generated form) sits at 65.3–68.0 — form diversity, not
   464	LLM origin, is what pays.
   465	
   466	**Why 1.0M and not 3.0M.** Every scale curve found saturates below it: DistilVDR (arXiv 2608.10636,
   467	Aug 2026 — asymmetric, pure cosine-regression, 70M query tower from an 8B teacher) saturates above
   468	75% of a 1.49M pool, with ~5 points from quarter-scale to full; SPEED (2410.18634) is log-linear
   469	only to ~920K and its sub-generators plateau at 10K–50K; doc2query-style scaling (2509.16442) gets
   470	90–95% of maximum gain at 50–75% corpus coverage; mxbai-edge-colbert's distillation stage used
   471	1.45M queries against a stella-1.5B teacher. **Nobody has published 100K vs 1M vs 3M**, so this is
   472	a prior, not a measurement — which is why A4−A3 is a registered screen contrast that can drop the
   473	generated half entirely.
   474	
   475	**Corrections to §3 the sweep forced.**
   476	- LEAF's loss is the L2 **norm** `‖e‖₂`, not squared L2, and its **Appendix B added distillation
   477	  terms to plain regression and found no improvement**. §3's "Ranking-aware KD" row is evidence
   478	  about *contrastive* systems, not about adding a term to embedding regression — the basis for
   479	  cutting family D to one arm (amendment A1).
   480	- LEAF's student is **MiniLM-L6-v2 plus a linear projection**, initialized off-the-shelf, document
   481	  side frozen — the same architecture as ours, which is why its 97.7% is the right aim.
   482	- LEAF pools the **last layer only**. Multi-layer pooling (family G) is ours, evidenced only by
   483	  §9–9b's two Mac probes, so G's losing arms are load-bearing for the paper.
   484	- LEAF's 800K "vocabulary" texts are Claude-generated definitions, on the *document* side — so even
   485	  LEAF's corpus is partly synthetic.
   486	- Nothing on-topic published 2026-08-01 to 2026-09-04 beyond DistilVDR; no new ≤35M permissively
   487	  licensed embedding backbone released in that window.
   488	
   489	## 13. Carried from the Fable review (2026-09-04) — adopted, and the one gap still open
   490	
   491	Full findings and dispositions: `research/m10-fable-plan-2026-09-04.md`, tabulated in
   492	`instructions-m10.md` §Amendment 2026-09-04. Read-exclusion audit clean.
   493	
   494	**Adopted into the plan:** the arXiv scientific harvest source (§Data), form-balanced query sampling
   495	(§Data), the registered plateau response and D-COV (§Recipe), family F at 20M with a third arm
   496	(§Screen), fixed-sequence gatekeeping and the four planning proxies (§Goal), the resolution-number remedy
   497	(§Surfaces — struck the same day by the Codex pass), hosted generation as the default, and the benchmark re-run (§11).
   498	
   499	**Not yet done, ranked, with cost — these are the next actions, not ideas:**
   500	
   501	| # | action | cost | why it is worth it |
   502	|---|---|---|---|
   503	| 1 | `torch.compile(mode="reduce-overhead")` / CUDA graphs on the fixed length buckets, then a real-data re-measure | 0 GPU-h, ~2–4 h engineering | §11's caveat is the whole cost story: M9's pipeline ran at ~10% of the hardware roof. Closing that gap is worth more than the GPU rental, and 1.5–3× at bs32 also dissolves most of family E's cost argument. **Box-window item 1** |
   504	| 2 | arXiv licence evidence and harvest yields | CPU + ~500K stella encodes ≈ minutes | amendment A2's scientific forms do not exist without it, and the fallback (revert to generation) must be chosen on a measured yield, not assumed. **Box-window item 5** |
   505	| 3 | `nq-250k` retention as a named per-arm diagnostic | ≈0 | see the open gap below |
   506	
   507	### The gap the plan still does not answer: the in-distribution ceiling
   508	
   509	M9 retained **93.8% on NQ while training on NQ-like data**. Coverage explains the 50–71% on forum
   510	questions; it does **not** explain the 93.8%. LEAF reaches 97.7% overall on an easier target
   511	(109M/768-d teacher). The four planning proxies sit at 89.3% / 91.6% / 91.3% / **94.9%** of the
   512	ceiling (§Goal, 0.025 quantile), so **C2b demands better in-distribution retention than M9 achieved even where it was
   513	fully covered.** Everything M10 adds — coverage, width, the optimizer regime — attacks the
   514	out-of-distribution half. Only family G (width) and family E (regime) touch the covered half, and
   515	D-COV is now the one arm aimed squarely at it.
   516	
   517	This is recorded as an open weakness rather than closed. What would move it, in rough order of
   518	promise: D-COV or another loss that weights document-discriminative directions; more output width
   519	than 1536 (the parameter budget allows it only by shrinking the backbone); a longer dose on covered
   520	forms; and — outside M10's premise — document-side co-adaptation, which is M16 and which every
   521	≥96% near-zero-query system used. If M10 lands the release conjuncts but misses C2b, this section is
   522	the reason, and it was known in advance.

exec
/bin/bash -lc "nl -ba m10/PLANNING.md | sed -n '1,145p'" in /home/dylan/asymetric-dual-encoders
 succeeded in 0ms:
     1	# M10 planning evidence (2026-09-01 planning session, Mac, no box access)
     2	
     3	Facts behind `instructions-m10.md`. One section per decision. Research sweeps ran in two Sonnet
     4	subagents (web only); every number below that a rule reads was re-derived from the repo's own JSON
     5	or checked against the cited primary source.
     6	
     7	## 1. The M9 diagnosis in numbers
     8	
     9	| SCREEN-3 component | ceiling (`m9_dev_symmetric_stella-400M-v5.json`) | M9 candidate (`m9_final_checkpoint_dev.json`) | retention |
    10	|---|---|---|---|
    11	| nq-250k | 0.8839 | 0.8289 | 93.8% |
    12	| cqadup-physics | 0.4931 | 0.3501 | 71.0% |
    13	| cqadup-programmers | 0.4681 | 0.2345 | 50.1% |
    14	| macro | 0.6822 | 0.5606 | 82.2% |
    15	
    16	Build curve (`m9/RUN_STATUS.md`, `m9/BUILD_LOG.md`): 0.346 (closed-form head) → 0.500 at 0.25B
    17	tokens → 0.538 at 0.74B → 0.545 at 1.23B → flat 0.552–0.558 from 2.6B to 3.7B → 0.5606 annealed.
    18	Query pool (`m9/M92_LOCK.md`): 242,786 `queries_pair` (hotpotqa/squad/esci/mrtydi) + 85,863 nqopen
    19	+ 134,665 triviaqa = 463K real queries, all Wikipedia QA or product search; longest 108 words.
    20	Documents 6.15M (581M tokens/epoch). Token mix 5/5/90; realized step 113 examples.
    21	
    22	Reading: the student matched the teacher on the distribution it saw and not elsewhere. The six's
    23	query forms (claims, biomedical questions, finance questions, 170-word arguments, paper titles,
    24	COVID queries) are all "elsewhere". Capacity is not excluded by this table alone; §9 and the
    25	M10.0 capacity probe test it directly.
    26	
    27	## 2. Bars and headroom — unchanged
    28	
    29	Ceiling 0.5744 avg-6; aim 0.5155 needs 89.7%, release 0.5042 needs 87.8% (`m9/PLANNING.md` §1).
    30	The six are all out of M9's training distribution, so M9's avg-6 is expected in the 50–75%
    31	retention band (0.29–0.43) — below BM25 on some datasets. M9's close-out six-set run replaces this
    32	guess with a number and gives the first dev→six calibration point for a tower (M7's is for a
    33	table: dev out-of-domain 0.764 vs six 0.755).
    34	
    35	## 3. Levers with published numbers
    36	
    37	| lever | evidence | source |
    38	|---|---|---|
    39	| Small batch | LEAF 1-epoch val loss: bs16 0.4194, bs32 0.4214, bs256 0.4593 | arXiv 2509.12539 |
    40	| Cyclic anneals | LEAF: 3 × 10 epochs, 1e-4→1e-5 linear per cycle; M9's single anneal was worth +0.004 | 2509.12539; `m9/BUILD_LOG.md` |
    41	| Query+document text | LEAF 1-epoch NanoMSMARCO: queries-only 46.7 vs both 60.7 | 2509.12539 |
    42	| Query-form breadth | LEAF texts: 502K MS MARCO queries, 979K Amazon QA, 273K PubMedQA/TriviaQA, 27K LoTTE, 3M FineWeb, 900K CC-News — ~1.8M query-like across five styles; ours 463K across two | 2509.12539; `m9/M92_LOCK.md` |
    43	| Ranking-aware KD | mxbai-edge-colbert: listwise KL over teacher scores, 17M/32M students at ~90%/~95% of a 130M+ teacher; EmbedDistill ablation NQ R@5: score distillation 44.3 → +doc embeddings 56.3 → +L2 query matching 61.2 → +synthetic queries 64.3 | arXiv 2510.14880; arXiv 2301.12005 |
    44	| Init | 2306.11550: init strategy moves retention up to 6 points; 1/2/4-layer students retain 86.1/92.5/96.2% of BERT-base | arXiv 2306.11550 |
    45	| Synthetic queries for distillation | EmbedDistill +3.1 R@5 from generated queries; Qwen3 generator precedent (Qwen3-Embedding, arXiv 2506.05176) | 2301.12005; 2506.05176 |
    46	| Capacity gap | TAKD: a too-large teacher hurts a fixed student; a 2026 study finds student capacity, not the gap, gates KD gains. No published asymmetric-retrieval distillation uses a teacher above ~220M/768d — our 400M/1024d pair is outside the literature | arXiv 1902.03393; arXiv 2605.31191 |
    47	| Alternative ≤35M students | ettin-32m (MIT, ModernBERT; ONNX friction, excluded in M9), granite-embedding-30m (Apache, BEIR 50.4), bge-small (BEIR ~51.7) | arXiv 2507.11412; arXiv 2502.20204 |
    48	| Smaller teacher target | stella card: 1024d is 0.001 below 8192d; heads at 256/768/1024/2048+ exist (`2_Dense_*`); 512 does not | HF model card, revision `ffeb2b7e` |
    49	
    50	## 4. Qdrant releases and the FineWeb-10B drop — what they change
    51	
    52	- **Qdrant/FineWeb-10B** (ODC-By; Common Crawl terms): 10.07B FineWeb chunks embedded with
    53	  `Alibaba-NLP/gte-multilingual-base` (768d) plus sparse vectors, with brute-force ground truth for
    54	  100K queries. **The vectors are in the wrong space for nano** (gte-ml-base, not stella) and the
    55	  bundled ground-truth queries are **MS MARCO-derived** — never used here. What transfers: the
    56	  FineWeb text (ODC-By, already approved as document text), and Supernova's `nova-embed` as a
    57	  GPU-batch encoder pattern for large teacher encodes. Companion drops (PubMed-MV on BGE-M3;
    58	  Coyo-VE on Qwen3-VL-Embedding-2B) are whitepaper context; PubMed is a contaminating corpus here.
    59	- **Qdrant 1.18 (TurboQuant) and 1.19 (TurboQuant 4-bit as primary storage, cold/cached/pinned
    60	  memory tiers, per-tenant IDF)**: index-side levers for the M13 all-in quantization comparison
    61	  and the edge footprint story; they do not change nano's training. **Cloud Inference does not host
    62	  stella**; the document tower stays self-hosted. **Qdrant Edge** versions separately (private
    63	  beta). **fastembed 0.8.0**: `add_custom_model()` has no post-pooling Dense slot — M9's per-token
    64	  linear-head export remains the serving path, and a nonlinear head would need its own graph.
    65	- **No Qdrant-published static or distilled-tiny query encoder exists** (HF org: `bm25`,
    66	  `minicoil-v1`, `Qwen3-Embedding-0.6B-onnx`, CLIP pairs for Edge). zero and nano would be first.
    67	
    68	## 5. Data arithmetic — SUPERSEDED (2026-09-04 evidence only; the mandate's §Data and §Compute are authoritative)
    69	
    70	- Query corpus ≈ 463K real + 1.0M PAQ + 3.0M synthetic = **4.5M** texts; ≈ 35 tokens mean (long
    71	  forms 120–220 words at ~10% share) → ≈ 160M tokens per query epoch.
    72	- Generation: 3.0M queries × ~300 prompt tokens ≈ 900M prefill; output ≈ 60 tokens for ten forms
    73	  and ≈ 120–270 for the conversational and argument forms (`max_new_tokens` 400) ≈ **250M generated
    74	  tokens** (350M at the caps), plus retries. Qwen3-8B bf16 via vLLM on the A100 — **unmeasured**; at
    75	  1,500–3,000 decode tok/s with prefill overlapped ≈ **30–60 GPU-hours** ≈ $45–150. Fallback if the
    76	  smoke's end-to-end projection of the full job (all forms, prefill, retries, JSON failures) exceeds
    77	  60 GPU-hours: hosted open-weights inference at $0.1–0.3 per M tokens ≈ $110–330, the second §6
    78	  scenario. The per-form 200-query smoke measures requests/s end to end (watch-long-runs rule).
    79	- Teacher targets: stella query encode was 2,076/s on the box (`m9_throughput_probe.json`); at an
    80	  assumed 3× on the A100, 4.5M queries + A2's 4.037M PAQ control texts + ≤ 3M seed passages (D-NCE
    81	  positives) ≈ **2 GPU-hours** with mining. Hard-candidate mining: 4.5M × 1M × 1024 ≈ 9.2e15 FLOP,
    82	  minutes of matmul plus top-k over 1M columns per 1,024-query chunk — **unmeasured**; the mandate
    83	  gates it on a 10K-query smoke and registers Qdrant HNSW with an audited recall@64 ≥ 0.98 as the
    84	  fallback. Bank: 1M pool documents, seed 0.
    85	- **Planning rate.** LEAF trained 6.7M texts × 30 epochs ≈ 201M examples at batch 32 in ~100
    86	  A100-hours (`m9/PLANNING.md` §3) → **≈ 560 examples/s**. Our examples are shorter on average (≈ 84
    87	  tokens at 75/25 against LEAF's 256-token FineWeb documents), so the rate is conservative in
    88	  direction; it is unmeasured on our stack. M9's box rate (18,984 tok/s ≈ 226 examples/s,
    89	  `m9/M92_LOCK.md`) is retired with the box. The first screen arm measures examples/s and §6 is
    90	  re-derived from it.
    91	- **Build dose 200M examples** (LEAF's; the 2026-09-01 amendment — the old 50M and its 83.4M cap
    92	  were set by the box's days, not by evidence; M9's plateau was on a narrow pool). At 75/25: 150M
    93	  query examples × ~35 tokens + 50M document examples × ~230 ≈ **16.8B tokens**; at 50/50 ≈ 26.5B.
    94	  ≈ **100 GPU-hours** at 560 examples/s. Query epochs ≈ 33 over 4.5M texts; document epochs ≈ 8
    95	  over the 6.15M pool (LEAF: 30 over everything). Each extension cycle is 66.7M ≈ 33 GPU-hours.
    96	- Screens: 5M examples per arm (2.5% of the build; 2.5M was 5% of the old 50M build) ≈ 2.5
    97	  GPU-hours at 560 examples/s, plus ≈ 0.3 h of DEV-6 and COV evaluation. Fourteen arms: A1–A3 (15M),
    98	  B 100/0 (3.75M) and B 50/50 (7.5M; the anchor doubles as B 75/25), C, D-KL3, D-KL1, D-NCE, E, F,
    99	  G-384, G-768, G-1536 (9 × 5M = 45M) = **71.25M examples ≈ 40 GPU-hours**. Confirmations, worst
   100	  case: B's 50/50 wins (2 seeds × 7.5M + default 2 × 5M = 25M) plus three ordinary decisions (each
   101	  winner + default × 2 seeds = 20M) = **85M ≈ 47 GPU-hours** with its 16 evaluations, plus the
   102	  synthesized selected-recipe arm for LoTTE read #1 (5M ≈ 2.5 h). Capping confirmations at two
   103	  decisions saves ≈ 20 GPU-hours.
   104	- Encodes with stella, all re-derived on the instance (the box's `work/` caches do not travel): the
   105	  6.15M pool (targets and the heldout dev components) ≈ 2.8 h, hotpotqa 5.2M + nq-250k ≈ 2.5 h, the
   106	  six ≈ 0.1 h, COV (MedicalQA 2K, BRIGHT a few K per slice, CorporateLobbying 319, LEDGER ≤ 100K
   107	  chunks) ≈ 0.3 h, LoTTE-clean 2.8M ≈ 1.3 h, the reserved four (FEVER 5.4M, DBpedia 4.6M, two CQA)
   108	  ≈ 4.6 h **only if the reserved conditional fires** — ≈ **12 GPU-hours** at an assumed 600 docs/s
   109	  (3× the box's 210, unmeasured), plus about a day of CPU and network to pull the pool, dev suite
   110	  and fingerprints from HF.
   111	
   112	## 6. Compute plan — SUPERSEDED (the all-cloud plan; the mandate's §Compute table is authoritative and `max_extension_cycles` reads it, not this)
   113	
   114	"M10 won't be done on a 3080. M10 will be done on a GPU budget, if allowed, or not at all." The box
   115	path is withdrawn (§7). One A100 80 GB (H100 if cheaper per example on the smoke), ≥ 500 GB
   116	persistent disk, stopped between stages; $1.5–2.5/h assumed, unverified Sept 2026. Every line is at
   117	the §5 planning rates and is re-derived from the day-one rate benchmark (mandate §Compute) before
   118	generation starts, and again before the remaining screen arms.
   119	
   120	| mandatory line | GPU-hours | $ at 1.5–2.5/h |
   121	|---|---|---|
   122	| re-derive encodes (pool, dev suite, six, COV, LoTTE; reserved four only if the conditional fires) | 12 | 18–30 |
   123	| day-one rate benchmark (encode, three training mixes, generation smoke) | 2 | 3–5 |
   124	| query and seed teacher targets, mining | 2 | 3–5 |
   125	| screens, 14 arms at 5M with evaluations | 40 | 60–100 |
   126	| confirmations, worst case 85M with 16 evaluations | 47 | 71–118 |
   127	| synthesized selected-recipe arm, LoTTE reads, M9 close-out scoring, M10.0-c | 4 | 6–10 |
   128	| build, 200M examples | 100 | 150–250 |
   129	| export, parity, final run | 2 | 3–5 |
   130	| persistent disk ≈ 3 weeks, egress | — | ≈ 40 |
   131	| **subtotal without generation** | **209** | **$354–563** |
   132	| **scenario A — generation on the GPU** (Qwen3-8B bf16, vLLM) | 30–60 | 45–150 |
   133	| **scenario A total** | **239–269** | **≈ $400–715** |
   134	| **scenario B — hosted generation** (if the smoke projects > 60 GPU-hours on the GPU) | 0 | 110–330 |
   135	| **scenario B total** | **209** | **≈ $465–895** |
   136	| optional: second build seed (decision 8) | 100 | 150–250 |
   137	| optional: extension cycle, 66.7M examples, each | 33 | 50–83 |
   138	| **ceiling requested** | | **$1,000, hard** |
   139	
   140	**Allocation order at the lock:** every mandatory line first at the measured rates and the billed
   141	price; then decision 8 if ≥ 100 GPU-hours remain; then `max_extension_cycles` whole cycles from the
   142	remainder (mandate §Recipe). At $2.5/h nothing optional fits in either scenario and the plan still
   143	completes; at $1.5/h with scenario A, seed 1 and three extensions fit.
   144	
   145	Wall-clock ≈ 2.5–3 weeks sequential on one GPU (≈ 240 GPU-hours, a day of CPU and network, the

exec
/bin/bash -lc "nl -ba m10/PLANNING.md | sed -n '145,259p'
nl -ba CLAUDE.md | sed -n '131,295p'" in /home/dylan/asymetric-dual-encoders
 succeeded in 0ms:
   145	Wall-clock ≈ 2.5–3 weeks sequential on one GPU (≈ 240 GPU-hours, a day of CPU and network, the
   146	smoke approval and the review gates between stages); screen arms are independent, so 2–4 GPUs
   147	compress the screen and confirmation stages at the same total cost. The same dose on the RTX 3080
   148	would be ≈ 10 days for the build and ≈ 4 for the screens, before any confirmation seed. **From the
   149	box, once, sha-verified:** `work/m9long/ckpt/last.pt` (M9's frozen candidate, `9d631b2c…`) for
   150	family C, the M10.0-c baseline read and M9's close-out, plus any `work/` file `m9src/guard9.py`
   151	hashes; nothing else transfers. The Mac stays what it was: probes, code, the generation smoke's
   152	prompt development (stella only in `.venv-mac`).
   153	
   154	## 7. Considered and rejected (reopening condition per row)
   155	
   156	| avenue | why rejected | reopens if |
   157	|---|---|---|
   158	| Teacher change (Qwen3-Embedding-0.6B, gte-large-v1.5, arctic-embed-l-v2) | breaks the one-index pair; stella-1.5B measured −0.0023; gte-large-v1.5 is stella's own backbone; Qwen3-0.6B +0.004 nominal, never screened | the pair story is dropped by Dylan |
   159	| >35M student in any role | **hard cap, Dylan 2026-09-01**: "109M is not an option. This isn't low compute anymore. 33M was already in the upper bound" | never |
   160	| Regress to stella's 768d or 256d head | a smaller index is a separate system and a full re-encode of every reserved corpus; §9 says whether the 384-rank bottleneck even binds | §9 shows <95% at k=384 AND the MLP-head arm fails |
   161	| Document-side co-adaptation (E14-LORA) | inside M10 it breaks the pair. Co-training the tower against **both** query paths at once keeps it, and it is the lever every ≥ 96% near-zero-query system used (LightRetriever's lookup 96% vs zero's 75.5%; ScalingNote 99%; CARE stage 2); costs a new index and a rebuild of zero | never inside M10; **recommended 2026-09-01 as the next-milestone candidate** (that slot was M12 then; it is **M16** after the 2026-09-04 renumbering), Dylan's call |
   162	| The RTX 3080 as M10's execution target | Dylan 2026-09-01: cloud GPU budget or not at all. A LEAF-scale build is ≈ 10 days on it and the screens 4 more; the 50M dose and 83.4M cap it forced were box artifacts | never for M10 |
   163	| FineWeb-10B vectors as targets | wrong embedding space (gte-multilingual-base 768d) | never |
   164	| FineWeb text in any role (Dylan delegated 2026-09-01; ruled out) | documents: no reserved-set fingerprints exist (Codex pass 1 B2); seeds: a rights review and a URL blocklist for topics Wikipedia and the pool already seed | family A wins on forms yet COV shows a topic gap Wikipedia cannot seed |
   165	| PubMedQA, Amazon QA, CC-News as query text (LEAF used them) | PubMed is a contaminating corpus for NFCorpus/TREC-COVID; Amazon QA and CC-News have no affirmative commercial grant | a licence review clears Amazon QA / CC-News |
   166	| Symptom-gated phase 2 (M9 design) | the gate was never specified; a flat curve then had no registered response | never in that form — phase 2 is a screen arm |
   167	| SCREEN-3 as the selection surface | NQ at weight 0.50 with NQ-adjacent training data inflated M9's read by ~11 points against its out-of-domain components | never |
   168	| Spending LoTTE read #1 on the M9 candidate | would burn the only fresh surface on a candidate that misses | never |
   169	| Hosted proprietary API as the query generator | output-use terms would be the weakest link in the licence story | never |
   170	
   171	## 8. Adversarial review disposition
   172	
   173	*Each table records the state at that pass; later passes superseded some numbers (multiplicity
   174	/6 → /10 → /13, confirmations 3 → 4, the >35M conditional → hard cap, FineWeb → out, 3–4 weeks →
   175	4–5). The mandate is authoritative where a row here disagrees.*
   176	
   177	**Pass 1 — gpt-5.6-terra, high effort, read-only, 2026-09-01** (`research/m10-codex-plan-2026-09-01.md`;
   178	read-exclusion audited: the reviewer opened only the twelve named files). Verdict "not
   179	decision-grade"; 3 BLOCKER / 10 MAJOR / 2 MINOR. **All 15 actioned in the mandate rewrite:**
   180	
   181	| # | finding | disposition |
   182	|---|---|---|
   183	| B1 | M9's six-set close-out before M10's lock is development on the six | **adopted** — close-out moves to after the M10.2 lock push; no M9 six-set output exists while any M10 decision is open |
   184	| B2 | reserved-set document fingerprints are an unspent held-out access; "hash-only" is not blindness | **adopted** — decision 5 inverted: FineWeb documents excluded; no reserved bytes opened |
   185	| B3 | CQA-2 cannot validate coverage of the six's forms; FORMS-12 is teacher agreement, not retrieval | **adopted** — COV (licensed, decontaminated, qrel-bearing, multi-domain) is the selector, admitted at M10.0-d with a four-component floor; FORMS-12 descriptive; the remaining form gap (claims, titles, arguments) is stated |
   186	| M4 | arm 5 confounds data with recipe; a null stops M10 on weak evidence | **adopted** — family A: A1/A2/A3 differ only in data; A3 vs A2 isolates forms from volume; stop rule and its consequence written |
   187	| M5 | 50/50 mix repeats M9's unresolved mix result | **adopted** — family B: 100/0, 75/25, 50/50 at matched query presentations; default 75/25 |
   188	| M6 | batch pilot confounded (M9's own objection) | **adopted** — family E at equal examples and identical schedule; throughput recorded, decides nothing; default 32 |
   189	| M7 | phase-2 KL not decision-ready | **adopted** — τ by a registered entropy rule before any arm, seed-document exclusion, fixed negative policy, phase 2 as a cycle-3 continuation compared at matched examples |
   190	| M8 | rank probe cannot govern a head decision; MLP has no serving path | **adopted** — head fixed linear; probe descriptive. (The probe measures retrieval retention of projected queries against the frozen index, not reconstruction — the reviewer had not seen the script; the action is unchanged) |
   191	| M9 | FineWeb seeding contradicts the licensing record; outputs called "ours" | **adopted** — seeds default to Wikipedia + approved corpora; FineWeb seeding is decision 3 with a rights review; outputs described under the generator's terms, not redistributed |
   192	| M10 | contamination controls unspecified | **adopted** — thresholds (exact / 8-gram ≥ 8/32 / 4-gram containment), three screen targets incl. the seed passage, per-form quotas, PAQ revision + seed, FORMS-12 hold-out — all in §Data |
   193	| M11 | screens multiplicity- and seed-blind | **adopted** — Bonferroni 0.025/6 on the lower bound; two confirmation seeds at screen dose; revert-to-default rule |
   194	| M12 | cost arithmetic inconsistent; 25,970 tok/s is not the measured rate | **adopted** — one budget from 18,984 tok/s; generation rate marked unmeasured and gated on smoke; §5–6 rewritten |
   195	| M13 | "beat LEAF" is not isolated | **adopted as already present, made explicit** — C2 is a whole-system comparison; the mandate says so and forbids any tower-isolating statement |
   196	| m14 | 35M cap is a product choice presented as science | **adopted** — decision 6: a conditional >35M tier if the capacity probe clears 85% |
   197	| m15 | reserved batch conditional on C1 only | **adopted** — `if C1 or C2` in the M10 re-registration |
   198	
   199	**Pass 2 — same reviewer, on the rewritten text** (`research/m10-codex-plan2-2026-09-01.md`;
   200	read-exclusion audited clean). Disposition audit: 10 of 15 pass-1 fixes landed, 5 did not
   201	(M4, M5, M11, M12 arithmetic, M7 timing). New: 3 BLOCKER / 8 MAJOR / 1 MINOR. **All actioned in
   202	the second rewrite:**
   203	
   204	| # | finding | disposition |
   205	|---|---|---|
   206	| B1 | COV not protected from training data | **adopted** — admitted COV queries and documents join the protected index before any PAQ/synthetic construction; the M9 pools are re-screened; removals published before COV is scored |
   207	| B2 | M9's close-out could spend the reserved access if its C1 passed | **adopted** — M9's close-out is amended to six-only (reserved conditional struck), disclosed and ratified with decision 1 |
   208	| B3 | family B's "equal examples" contradicts "matched query presentations" | **adopted** — B is matched on 1.875M query presentations with documents added on top; totals and document cost stated; equal examples holds for the other families |
   209	| M | family A conflates volume with forms; inequality undefined | **adopted** — A2 is a factoid-only volume control at 4.5M unique texts; A3−A2 isolates forms; the rule is point ≥ MDE and corrected lower bound > 0; the stop rule reads A3−A2 and its conclusion is scoped to COV families |
   210	| M | COV can collapse to a narrow surface | **adopted** — equal weight per family, slices averaged first; four-family floor |
   211	| M | COV eligibility unestablished | **adopted** — admission record (licence URL and terms, repo + revision, sizes, qrels format, metric) committed before a component is named COV |
   212	| M | COV cannot carry the full coverage conclusion | **adopted** — the conclusion and stop rule are scoped to COV families; claims / titles / arguments are tested only by the six-set transaction, stated in §Surfaces |
   213	| M | contamination exclusions omit NutritionFacts and Reddit finance | **adopted** — the full source-family list is in §Data and applies to seeds, regression text and COV |
   214	| M | τ cannot be selected at M10.0-e | **adopted** — rule locked at M10.0-e; executed after the immutable manifest, before any arm; sample spec, tie-break and fallback fixed |
   215	| M | multiplicity and confirmation ill-defined | **adopted** — ten contrasts enumerated; Bonferroni 0.025/10; confirmation re-trains winner and default with two seeds each, at most three decisions |
   216	| M | dose and cost arithmetic (3.8B vs 4.2B; mining; screen days) | **adopted** — 4.2B in both files; mining FLOP count stated, measured on a smoke, HNSW fallback registered; screens re-budgeted at 2.5M examples: 1.3 days + ≤ 1.5 days confirmations; box path 3–4 weeks |
   217	| m | rank-probe docstring says 512d; STATUS asks for a nonlinear-head dry run | **adopted** — docstring corrected (1024/768/256); the dry-run line removed |
   218	
   219	**Pass 3 — same reviewer, verification of the second rewrite** (`research/m10-codex-plan3-2026-09-01.md`;
   220	read-exclusion audited clean). 12 of 15 pass-2 dispositions land; two "not fully"; 4 new MAJOR.
   221	**All actioned:**
   222	
   223	| # | finding | disposition |
   224	|---|---|---|
   225	| M | family A's stop rule is weaker than "beat by the MDE" | **adopted** — three registered outcomes on A3−A2: corrected lower bound > MDE = resolved; point ≥ MDE with bound in (0, MDE] = positive-not-resolved, build proceeds labelled; else stop |
   226	| M | A2 cannot reach 4.5M with a 4.0M PAQ cap | **adopted** — A2 = 463K + 4.037M PAQ = 4.5M; decision 4 says so |
   227	| M | LoTTE read #1 has no candidate and no veto statistic | **adopted** — one synthesized selected-recipe arm at screen dose, hash committed before LoTTE opens; M9 §7's 0.004 non-inferiority rule verbatim; veto → the anchor recipe builds |
   228	| M | STATUS says four surviving components, mandate says four families | **adopted** — STATUS corrected to four families |
   229	| M | confirmation budget understated after the family-B redesign | **adopted** — worst case recomputed at 3.1B tokens ≈ 1.9 days plus the synthesized arm; box path ≈ 4 weeks; cloud GPU-hours 70–100 |
   230	
   231	**Pass 4 — same reviewer, on the third rewrite, family G and the report page**
   232	(`research/m10-codex-plan4-2026-09-01.md`; read-exclusion audited clean). Pass-3 items all land;
   233	family G's export algebra and parameter count confirmed. 2 BLOCKER / 2 MAJOR / 2 MINOR on the plan,
   234	5 number disagreements on the page. **All actioned:**
   235	
   236	| # | finding | disposition |
   237	|---|---|---|
   238	| B | §9 promoted the PCA row to an upper bound on every 384-d subspace | **adopted** — reworded everywhere: the row is the reconstruction-optimal subspace's retention, the target L2 regression pushes toward; evidence, not a bound; the screen decides G |
   239	| B | family A's rule contradicted the generic rule | **adopted** — A3−A2 exempted explicitly; resolved requires the corrected lower bound > MDE; A3−A1 and A2−A1 descriptive |
   240	| M | stale counts (ten contrasts, nine arms) | **adopted** — eleven arms, thirteen contrasts everywhere |
   241	| m | probe JSON provenance text said 512-d head | **adopted** — text field corrected to 768/256 (numbers untouched) |
   242	| M/m | report page: LEAF 97.9 vs 97.7; build-curve first point; Mac docs/s; 33M; premature "four passes" | **adopted** — LEAF labelled 97.9% on our six (97.7% is BEIR-14) in `m9/FINDINGS.md` and the page; the curve's first point labelled as the build's 0.12B eval with the screen anchor drawn separately; 20–100 docs/s; 33.4M; the review paragraph rewritten after this pass |
   243	
   244	**Pass 5 — same reviewer, cross-file consistency and omissions on the frozen set**
   245	(`research/m10-codex-plan5-2026-09-01.md`; read-exclusion audited clean). 3 BLOCKER / 5 MAJOR /
   246	2 MINOR. **All actioned:**
   247	
   248	| # | finding | disposition |
   249	|---|---|---|
   250	| B | the CQADupStack components were scored by the Mac diagnostics yet proposed as COV | **adopted** — COV admits only surfaces no M10 decision has read; the CQA pair is DEV-6, reported beside every COV read; 86 raw reads logged in `m10/RESULTS.md`; floor three untouched families |
   251	| B | A2 and A3 were not volume-matched (4,500,314 vs 4,463,314) | **adopted** — identical post-screen unique counts, the larger downsampled with seed 0, both hashes locked before any arm |
   252	| B | BRIGHT's third-party documents fail an "every document's rights" standard | **adopted modified** — the standard applied is the dataset-level licence at the primary source, the same one that admitted CQADupStack and the six (also third-party text); BRIGHT is one family, its caveat disclosed, evaluation-only. Demanding per-document rights would disqualify the six themselves |
   253	| M | student contradiction (bge-small architecture vs MiniLM default) | **adopted** — bge-small is the screen anchor, family F decides the build student, MiniLM default; MiniLM's head passes the same parity check first |
   254	| M | generation gate not executable (20 vs 200, unpinned fallback, no decoding/seed/retry/rubric/approver) | **adopted** — §Data generation contract: 200 per form everywhere, pinned artifacts, sampling parameters, deterministic seeds, one retry, dedup, the 90%/80% rubric, Dylan as approver, two prompt revisions max |
   255	| M | the ledger does not exist | **adopted** — `m10/LEDGER.md` skeleton committed with the sections the lock must fill |
   256	| M | extension, margin, seed range, negatives RNG, missing-seed behaviour undefined | **adopted** — all defined in §Recipe and §Screen |
   257	| M | seed-level leakage; 5-word rule vs 8-gram screen; lenient parser | **adopted** — seeds pre-filtered against the protected index; word-5-gram containment against the seed; `forms.parse` strict |
   258	| m | stale numbers in §8 read as policy; "five passes" premature | **adopted** — this note at the top of §8; pass 5 recorded |
   259	| m | dev-reuse accounting not fillable | **adopted** — exact counts (86) in `m10/RESULTS.md` |
   131	**Before launching anything that runs longer than ~10 minutes:**
   132	
   133	1. **Smoke the code path first.** `sweep.smoke(base, over)` runs an arm at 90 steps. Prefer a path
   134	   with no execution history — that is where the bug is. A grid's arms share code, so
   135	   `sweep.grid(..., fail_fast=True)` (the default) stops at the first arm that *raises*; an OOM
   136	   does not trip it, being a real per-arm resource result.
   137	2. **Arm a monitor with the failure signatures, not just the success line.** Silence is not
   138	   success: grep an alternation covering `Traceback|Error|FAILED|OOM|Killed|assert` alongside the
   139	   progress marker. A monitor that only matches the happy path is indistinguishable from a
   140	   crashloop.
   141	3. **Read the first progress line and sanity-check the RATE against an estimate** before walking
   142	   away. "mine 2048/349934 (76s)" is a three-hour job stated in the units of a fast one. Two
   143	   minutes of arithmetic at launch is worth hours later.
   144	4. **Watch the machine, not only the log.** 100% GPU utilisation with ~1% memory-bandwidth
   145	   utilisation and low power draw means allocator thrash or launch-bound tiny kernels, not work.
   146	   `/proc/<pid>/stat` utime against elapsed says whether a "GPU job" is actually burning one CPU
   147	   core; per-thread CPU time says whether it is Python or a kernel. `py-spy` needs
   148	   `kernel.yama.ptrace_scope=0`, which needs root on this box — so these cheaper signals matter.
   149	5. **`tail -F`, never `tail -f`, and match the log's own casing.** A monitor naming a chained
   150	   job's log before that file exists follows nothing when it appears — plain `-f` does not retry —
   151	   and if the liveness check still matches the process, **silence looks exactly like progress**.
   152	   That cost 3.5 h of unwatched training on 2026-09-08, noticed only because Dylan asked. The same
   153	   monitor also missed the completion line by grepping `COMPLETE` for a log that writes `complete`.
   154	6. **Never trust a docstring's cost estimate.** Both "a few minutes for the whole set" claims in
   155	   `train.py` were wrong by two orders of magnitude, in the same file, on the same day.
   156	
   157	## A number from one measurement is an assumption (2026-09-07)
   158	
   159	Written after a memory bug took four wrong diagnoses and most of a session. Each was stated as
   160	settled and each died to the next measurement: per-row growth, page cache, the chunk size, the
   161	tokenizer. The bug was a fixed 2.2 GB load in `row_texts`, found only by asking it for **ten rows**.
   162	
   163	- **A per-unit rate needs two points.** "+3,453 MB at 1M → 4.3 KB/row → 20.4 GiB at 5M" was a
   164	  fixed cost divided by n. Two sizes showed 5× the work costs **+29 MB**. One point cannot tell a
   165	  constant from a slope, so it cannot support an extrapolation — and 6× wrong here nearly cancelled
   166	  a family.
   167	- **Vary the knob you are blaming.** Chunk-invariance was read as "the tokenizer's transient". A
   168	  10× smaller *tokenizer batch* moved the peak 4%. If you have not moved the thing you are
   169	  accusing, you have not tested it.
   170	- **Measure at the registered dose, on the metric that failed.** The 300-step smoke passes because
   171	  it shrinks the very count that breaks. The crashes presented as **host free 239 MB**, not as
   172	  process RSS — so that is the number a pre-flight has to report.
   173	- **Know the instrument.** `VmHWM` is monotonic; a 5-second `VmRSS` sample is a lower bound and
   174	  hides any transient shorter than the interval. `RssAnon` is unreclaimable, `RssFile` is page
   175	  cache the kernel drops — opposite consequences. A first probe wrote to `/tmp`, which is tmpfs,
   176	  and mis-categorized the pages entirely. And a `pgrep -f` pattern matches the *launching shell*.
   177	- **Reconcile a new number against the ones already written down.** Three figures got labelled
   178	  "three runs of one path"; two were this repo's own per-implementation measurements. That
   179	  fabricated an error bar where none existed.
   180	
   181	## Durable knowledge goes in the repo, not in the assistant's memory (Dylan, 2026-08-26)
   182	
   183	The point of this repo is a reusable harness, so anything a future session would need belongs in a
   184	file it will read: protocol in `m7/LEDGER.md`, dead ends in `m7/EXPLORED.md`, module facts and
   185	pitfalls in `m7/CODEMAP.md`, standing directives here. Assistant memory is for *Dylan-and-workflow*
   186	facts only (how he wants sessions run, what tooling exists on the box), and anything in it that
   187	would help a future session must be mirrored into one of those files. A lesson that lives only in a
   188	memory file is lost to every session that does not happen to recall it.
   189	
   190	## Markdown files must be TIGHT (Dylan, 2026-08-29)
   191	
   192	*"Information is wayyyy too verbose in there. We're diluting context for next session."* Said of an
   193	M8 ledger that had reached 2,694 lines, ~40% of it one session's amendment prose.
   194	
   195	**Every line in a `.md` file costs a future session context it could have spent on the work.** A
   196	file a session is *told* to read before deciding — `CLAUDE.md`, `STATUS.md`, `LEDGER.md` — is loaded
   197	whether or not the reader needs the paragraph you enjoyed writing. Verbosity there is not
   198	thoroughness; it is a tax levied on every session that follows.
   199	
   200	**The rule:** write the decision, the number a rule reads, and the pointer. Nothing else.
   201	
   202	- **One fact, one home.** Numbers live in the result JSON, bars in `registry.json`, runs in
   203	  `RESULTS.md`, closed avenues in `EXPLORED.md`, long-form reviews in `research/*`. A `.md` that
   204	  restates any of them is duplication that will go stale in exactly one direction — the wrong one.
   205	- **An amendment is: what changed, why, and the pointer.** Not the reasoning that got you there.
   206	  If the reasoning matters, it belongs in the archived review, cited by path.
   207	- **Prefer a table to prose, a clause to a sentence, a pointer to a summary.** Cut every sentence
   208	  that only restates the previous one with more emphasis.
   209	- **Withdrawn claims and owner rulings are the exception — always keep them**, because a future
   210	  session that re-derives a withdrawn claim wastes far more than the lines cost. Keep them *short*.
   211	- **When you add to a long file, budget for it**: if an entry runs past ~10 lines, compress an old
   212	  one or move detail out. Files grow by default; only deliberate effort shrinks them.
   213	- **Check the size when you touch it.** `wc -l` on the file you just edited. If a protocol file has
   214	  grown past ~1,500 lines, compressing it is part of the task, not a separate one.
   215	
   216	## Verification gates (Dylan, 2026-08-24)
   217	
   218	Results dictate Qdrant engineering decisions: correct, not decimal-precise; blind spots stated openly.
   219	- M2 gate: adversarial Opus review of methodology + code + M2 numbers → `research/verification-m2.md` (done).
   220	- Pre-report gate: Codex second opinion on the full result set + report draft, briefed for pushback; findings reported verbatim.
   221	- Significance: paired bootstrap on key system pairs before the report; deltas within noise get labeled as such.
   222	- **Codex CLI is installed on the box** (`/usr/local/bin/codex`, 0.149.1). Invoke it read-only and
   223	  with high reasoning effort, since the default profile here is effort `none`:
   224	  `codex exec -s read-only -m gpt-5.6-sol -c model_reasoning_effort="high" - < brief.md`. Read-only
   225	  matters: a previous review committed to the files the session was editing.
   226	- **STANDING GRANT, Dylan 2026-08-28 — supersedes the earlier "sparingly, at milestones only".**
   227	  *"Feel free to do adversarial reviews with Fable and Codex as much as needed. Use them to push
   228	  you to get the best result possible. They've proven valuable every time."* So reviews are a
   229	  **routine instrument, not a milestone ceremony**: brief one whenever a decision is about to
   230	  become expensive or irreversible, whenever a document is about to become permanent, and whenever
   231	  a result is surprising enough that you want it attacked before you believe it. The 2026-08-28
   232	  record is why — between them the two reviews caught a rounded CI endpoint in `final_run.py` (the
   233	  one irreversible decision in the project), a gate that returned 0 on NO-GO, a release guard that
   234	  failed open, a confounded diagnostic that had licensed a four-hour lever, and a retention figure
   235	  belonging to a reverted candidate. **Brief them adversarially**: give them the numbers, name what
   236	  you believe, and ask them to break it. A review told "confirm this" returns nothing.
   237	- **EVERY brief must carry a READ-EXCLUSION, and the log must be audited afterwards (2026-08-29
   238	  incident).** An external reviewer is a separate process with ordinary read access to the whole
   239	  repo, so `paths_guard`/G2 — an in-process Python bulkhead — is structurally incapable of
   240	  constraining it. A repo-wide grep dumped two RESERVED confirmatory sets, queries **and qrels**,
   241	  complete, into a reviewer's context. Nothing was scored and no decision read them, but the
   242	  reviewer's recommendations were thereby potentially informed by held-out data. So: state in the
   243	  brief that `results/frozen_eval/untouched-*`, the reserved qrels caches and `work/m9reserve` must
   244	  not be read; prefer naming the files to read over inviting a repo-wide search; and **grep the
   245	  review log for reserved-set reads before reading its findings.** Quarantine anything that draws
   246	  on them.
   247	
   248	## How to run an adversarial review, and why it is not optional (Dylan, 2026-09-05)
   249	
   250	*"Note the adversarial reviews process somewhere so next sessions keep doing it."* The standing
   251	grant above says reviews are routine; this is **how**. On 2026-09-05 six reviews caught, among
   252	other things: two arms that **could not win their own contrast**, a familywise α that was 0.0289
   253	not 0.025, a diversity gate **structurally blind** to 5 of 12 forms, the fact that **the M10 corpus
   254	had no path into the trainer at all**, and — on a re-review — that my own "fixes" for the first
   255	batch were **prose the validator never evaluated**.
   256	
   257	**The five rules that made those catches happen.**
   258	
   259	1. **Scope the read, always.** Every brief names the exact files to read and forbids a repo-wide or
   260	   recursive grep. This is not tidiness: a repo-wide grep once dumped two RESERVED confirmatory
   261	   sets into a reviewer's context. **Then grep the review log for reserved reads BEFORE reading its
   262	   findings**, and quarantine anything that drew on them.
   263	2. **State what you believe and ask them to break it.** Give the numbers, name your conclusion,
   264	   invite the attack. A review told "confirm this" returns nothing — and a review that returns
   265	   nothing is a badly written brief, not a clean bill of health.
   266	3. **Review the FIX, not just the decision.** The single most valuable catch of the day was a
   267	   re-review of eleven fixes that returned **NO-GO** because they were registry prose no code read.
   268	   *Writing something down is not making it work.* Same failure shape as a test that asserts a
   269	   sentence exists, or one that compares 0.0 ≤ 0.0.
   270	4. **Verify every checkable claim yourself before acting.** Reviewers are wrong sometimes, and a
   271	   claim you reproduce is one you can defend. Everything acted on today was re-derived first.
   272	5. **Two independent reviewers on anything expensive or irreversible.** They disagree usefully:
   273	   one said documenting a contradiction was enough, the other showed the registry then carried
   274	   *both* readings at once. Use different models, not the same one twice.
   275	
   276	**And record what the review overturned, including your own withdrawn claims** — four of mine were
   277	withdrawn on 2026-09-05 and each is in `m10/LEDGER.md` with the reasoning. A future session that
   278	re-derives a withdrawn claim wastes far more than the lines cost.
   279	
   280	## Key decisions (log)
   281	
   282	- **Teacher = `NovaSearch/stella_en_400M_v5` (2026-08-26), chosen on the distilled table, not the tower.** `arctic-embed-l` had been picked the same morning on its own dev retrieval quality (+0.0447 over bge-base) and was **WITHDRAWN the same day**: ranked by the table distilled from it, arctic sits −0.0480 [−0.0608, −0.0349] below bge-base, and a teacher's own quality does not predict its table (Spearman 0.000 over eight candidates). Only stella beat the incumbent (+0.0365 [0.0249, 0.0481]). Stella discloses ArguAna and FiQA (2 of the six) and FEVER (1 of the reserved four) in its training data; every stella-based claim carries that qualification and the NDO-4 rows. `results/m7_learnability_report.json`, `results/m7_teacher_contamination.json`. Kept because the failure mode — selecting a teacher on the tower — is the lesson. M9 re-confirmed it for towers: stella-1.5B distils WORSE than stella-400M (−0.0023 at equal dose).
   283	- **Clean data stack (2026-08-25, amended 2026-09-04):** MS MARCO permanently excluded from anything released (non-commercial terms; priced at +0.0058 [−0.0015, +0.0131] on the six — not what the miss is made of). CC BY-SA sources (NQ, SQuAD, HotpotQA, FEVER, MIRACL, Mr.TyDi) approved with model-card attribution. FineWeb was approved 2026-08-30 as document-side regression text but is **out of M10 in every role** (ruled 2026-09-01 under delegated authority: no reserved-set document fingerprints exist, and seeding from it would add a rights review for no measured gain). `research/m7-data-licensing.md`.
   284	- **Non-commercial licences are admissible for VALIDATION, not training (Dylan, 2026-09-04):** *"MS-MARCO is allowed for validation. Not training"*, generalised by him to any similarly-licensed dataset. Reaches only sets with an **affirmative but non-commercial grant** (MS MARCO, CC BY-NC eval sets, PAQ's NC generator, GooAQ). **Does NOT reach** sets with no grant at all (Quora QQP, WikiAnswers, ELI5, StackExchange downloads — QQP's eval exclusion stands) or sets excluded for contamination (S2ORC). Validation = measuring, screening, diagnosing, reporting; **never** gradients, distillation targets, mined negatives or synthetic-generation seeds. The training exclusion and its +0.0058 price are unchanged, as are `m7src/freeze.py`'s `NON_COMMERCIAL_SOURCES` guard and the `sources-research/` quarantine. Live consequences: M10's COV admission reopens (**ConsumerContractsQA back in, three families without LEDGER**) and gains a real-query coverage diagnostic; M12 may add a held-out fusion-weight transfer check; M14 may report an MS MARCO row. **Not reopened:** M7 (frozen), M9's `FINAL_LOCK.md` (frozen, six named datasets), the reserved four, any released artifact. Rule and the two confounds: `research/m7-data-licensing.md` §Rule change 2026-09-04.
   285	- **Reserved four** (FEVER, DBpedia-entity, cqadup-android, cqadup-english): one confirmatory access, unspent through M9. **`results/perquery.json` is irreplaceable** — frozen comparator vectors regenerated from caches that no longer exist. Never overwrite it.
   286	- **`zero` published 2026-09-03 (Dylan's ask); public from the first push, renamed `constella-zero` the same day.** The M7 artifact unchanged, on Dylan's own HF account, so the architecture can be tested as a PoC. Card rewritten FastEmbed-first 2026-09-03: competitive comparison and the missed bar REMOVED (`instructions-m11.md` Amendment B), measured numbers and the stella-contamination caveat kept. **Licence: MIT, ruled by Dylan 2026-09-03 and valid for a public release** (matching stella), with CC BY-SA attribution for NQ/SQuAD/HotpotQA/FEVER/Mr.TyDi. `m11/STATUS.md`.
   287	- **Renumbering 2026-09-01 (Dylan):** M10 becomes the nano retry; release/port/whitepaper moves to M11; the image model to M12.
   288	- **The upstream FastEmbed PR is M13, not now (Dylan, 2026-09-04):** *"once we have both models we add all of them in one clean PR."* `constella-zero` and the document tower are already published and registered, but the PR waits for nano so a maintainer gets one PR with all three entries. It branches fresh from upstream `main`; `Dylancouzon/fastembed@add-constella-models` is a throwaway (it also carries the #703 padding fix) and is never merged.
   289	- **Doc-tower card left unchanged, and the prompt-pair stays out of the CTO ask (Dylan, 2026-09-04).** The published doc graph IS a complete query encoder (prompt prepended → min-cos 1.00000000 vs torch, `m11/STATUS.md` addendum), but the card's narrower wording stands because FastEmbed's `query_embed` returns a wrong-protocol vector silently. The Cloud Inference ask covers document hosting only.
   290	- **Renumbering 2026-09-04 (Dylan), and the hybrid milestone:** `constella-zero-hybrid` is inserted as **M12**; nano's release M12→**M13**, the image model M13→**M14**, the better-zero idea register M14→**M15**. **Amended the same day**: the paper was split out of M13 into its own **M14** (Dylan: *"probably only 1"* paper, decided at M14's start), so the image model is **M15** and the better-zero register **M16**. Historical renumbering entries above are left as written. Review-finding IDs named `M12`–`M15` in `m9/`, `m10/` and `research/` were NOT touched — they are findings, not milestones; the two E14-LORA pointers that did mean a milestone were rewritten to name M15 explicitly.
   291	- **Renumbering 2026-09-03 (Dylan), and M11 CLOSED:** M11 shipped everything not depending on M10 (the zero half, both ONNX ports, the FastEmbed integration, both cards). The blocked remainder — nano's release and port, its FastEmbed entry, the whitepaper — becomes **M12**; the image model moves **M12 → M13**. Note for a cold session: most `M12` strings in `m9/`, `m10/` and `research/` are review-finding IDs, NOT milestones — they were deliberately left alone.
   292	- **Student cap 35M, hard (Dylan, 2026-09-01):** "109M is not an option. This isn't low compute anymore. 33M was already in the upper bound of what I think is acceptable." No larger student in any role; the cheaper student gets the benefit of a tie.
   293	- **M10 compute (Dylan, 2026-09-01):** "M10 won't be done on a 3080. M10 will be done on a GPU budget, if allowed, or not at all." The box is not an execution target for M10; doses and screen sizes are set from the budget, not the box (`m10/PLANNING.md` §5–6 and the §8 amendment block). Budget approval is decision 2; refusal closes M10 unstarted. **Amended 2026-09-04 (A7 confirmed; decisions 14, 15): the box runs the screens and the generation (Qwen's official AWQ release), the cloud buys the build; the ruling still binds the dose.**
   294	- **M9 close-out (2026-09-01):** LoTTE read #1 withdrawn unexecuted so M10 inherits a fresh surface; the six-set transaction runs six-only (reserved conditional struck, ratification pending) after M10's recipe lock; the `m9-status` branch deleted. M10 execution work happens on branch `m10-work` under the headless commit-and-push contract.
   295	- Harness, BEIR subset, LightRetriever reproduction details and the M1/M2 findings: `research/m1-m6-findings.md`.

codex
I’m adding a narrow code check of `m10src/screen_lock.py`, `m10src/corpus_loader.py`, `m10src/trainer10.py`, `m10src/run_arm.py`, and `m10src/cov_macro.py`, plus `instructions-m14.md` for the paper’s actual claim contract. This will distinguish errors in the operative rules from stale explanatory prose; I won’t execute training or evaluation code.
exec
/bin/bash -lc "wc -l m10src/screen_lock.py m10src/corpus_loader.py m10src/trainer10.py m10src/run_arm.py m10src/cov_macro.py instructions-m14.md
rg -n 'confirm|stands|seed_range|E_cost|D_tie|multi_arm|def |covariance|trace|objective|loss|warmup|dose|sample|balanced|plateau|forecast|forms' m10src/screen_lock.py m10src/corpus_loader.py m10src/trainer10.py m10src/run_arm.py m10src/cov_macro.py" in /home/dylan/asymetric-dual-encoders
exec
/bin/bash -lc 'nl -ba instructions-m14.md' in /home/dylan/asymetric-dual-encoders
 succeeded in 0ms:
   177 m10src/screen_lock.py
  1508 m10src/corpus_loader.py
   237 m10src/trainer10.py
  1063 m10src/run_arm.py
   142 m10src/cov_macro.py
    90 instructions-m14.md
  3217 total
m10src/cov_macro.py:38:def assert_surface(unit_family):
m10src/cov_macro.py:49:def weights(unit_family):
m10src/cov_macro.py:62:def macro(per_unit, unit_family):
m10src/cov_macro.py:73:def align(a, b, unit_family):
m10src/cov_macro.py:90:def contrast(aligned, unit_family, B=200_000, seed=0, quantile=0.025 / 13,
m10src/cov_macro.py:134:        "plan_sample_sha256": h.hexdigest(),
m10src/corpus_loader.py:1:"""The M10 query corpora -> the trainer. Sources, the manifest, and the form-balanced sampler.
m10src/corpus_loader.py:4:the corpus and the sampler the anchor is registered to use:
m10src/corpus_loader.py:7:  rows map onto the 12 forms. The generated half does not exist yet; its entry is here and reads a
m10src/corpus_loader.py:13:- **Form-balanced sampling is the anchor default** (`instructions-m10.md`:478-485): equal
m10src/corpus_loader.py:14:  presentation share per form, with replacement within a form. `balanced=False` is the unbalanced
m10src/corpus_loader.py:52:# The 12 registered forms, in `m10src/forms.RUBRIC` order. Imported rather than retyped so a form
m10src/corpus_loader.py:54:def form_names():
m10src/corpus_loader.py:55:    import forms
m10src/corpus_loader.py:56:    return tuple(forms.RUBRIC)
m10src/corpus_loader.py:64:# arrived in (Codex 2026-09-05 finding 3 -- `cp harvest_forms12.jsonl generated_queries.jsonl`).
m10src/corpus_loader.py:65:HOLDOUT_FILES = {str((WORK / "m10harvest" / "harvest_forms12.jsonl").resolve())}
m10src/corpus_loader.py:69:def text_hash(t):
m10src/corpus_loader.py:73:def holdout_hashes(path=None):
m10src/corpus_loader.py:81:    p = Path(path) if path else (WORK / "m10harvest" / "harvest_forms12.jsonl")
m10src/corpus_loader.py:96:def refuse_holdout_texts(segs, path=None):
m10src/corpus_loader.py:106:            f"REFUSED: the FORMS-12 hold-out ({path or WORK / 'm10harvest' / 'harvest_forms12.jsonl'}) "
m10src/corpus_loader.py:120:# PAQ row or to an M9-pool row, and the balanced sampler needs one per row. The reading here
m10src/corpus_loader.py:122:# and the M9 pool's own source labels map onto two of the twelve forms -- `esci-us` is a shopper's
m10src/corpus_loader.py:135:                   "what": "the build's PAQ sample, nested inside the A2 sample"},
m10src/corpus_loader.py:138:                   "what": "the A2 volume-control PAQ sample"},
m10src/corpus_loader.py:144:                   "what": "the seven generated forms; does not exist yet, same row shape"},
m10src/corpus_loader.py:147:# Which sources an arm's corpus is. A2 takes the 4.037M volume-control sample and A3/A4 the 1.0M
m10src/corpus_loader.py:148:# build sample nested inside it -- `m10/LEDGER.md` §1: "A2 is the volume control, so 'the build
m10src/corpus_loader.py:159:def sha_file(p, chunk=1 << 22):
m10src/corpus_loader.py:174:    def __init__(self, name, texts, forms, array, rowmap):
m10src/corpus_loader.py:176:        self.forms = np.asarray(forms, dtype=np.int16)
m10src/corpus_loader.py:178:        assert len(self.texts) == len(self.forms) == len(self.rowmap), name
m10src/corpus_loader.py:180:    def __len__(self):
m10src/corpus_loader.py:186:def _rows_from_jsonl(path, default_form=None, limit=None, with_ids=False, require_id=None):
m10src/corpus_loader.py:204:    texts, forms, ids = [], [], []
m10src/corpus_loader.py:214:                raise SystemExit(f"{path}: form {f!r} is not one of the 12 registered forms")
m10src/corpus_loader.py:223:            forms.append(f)
m10src/corpus_loader.py:235:        return texts, forms, ids
m10src/corpus_loader.py:236:    return texts, forms
m10src/corpus_loader.py:239:def held_out_doc_ids(path=None):
m10src/corpus_loader.py:249:    p = Path(path) if path else (WORK / "m10harvest" / "harvest_forms12.jsonl")
m10src/corpus_loader.py:263:def source_texts(name, limit=None):
m10src/corpus_loader.py:270:        texts, forms, man = _m9_texts()
m10src/corpus_loader.py:277:        texts, forms, ids = _rows_from_jsonl(p, default_form=spec.get("form"), with_ids=True,
m10src/corpus_loader.py:286:                "by_form": {f: forms.count(f) for f in sorted(set(forms))}})
m10src/corpus_loader.py:288:        texts, forms = texts[:limit], forms[:limit]
m10src/corpus_loader.py:292:    return texts, forms, ids, man
m10src/corpus_loader.py:295:def _m9_texts(screen=True):
m10src/corpus_loader.py:326:    forms = [M9_SOURCE_FORM[s] for s in labels]
m10src/corpus_loader.py:329:    return out, forms, man
m10src/corpus_loader.py:335:def _m9_labelled():
m10src/corpus_loader.py:346:def _m9_extra(name):
m10src/corpus_loader.py:355:def _m9_segments(screen=True, consumed=None):
m10src/corpus_loader.py:366:    qp_forms = [FORM_ID[M9_SOURCE_FORM[srcs[int(i)]]] for i in keep]
m10src/corpus_loader.py:367:    segs = [Segment("m9-queries_pair", [texts[int(i)] for i in keep], qp_forms,
m10src/corpus_loader.py:383:            out.append(Segment(sg.name, [sg.texts[int(i)] for i in sel], sg.forms[sel], sg.array,
m10src/corpus_loader.py:389:def dedup_segments(segs):
m10src/corpus_loader.py:407:        out.append(Segment(s.name, [s.texts[int(i)] for i in sel], s.forms[sel], s.array,
m10src/corpus_loader.py:412:def load_segments(names, head_per_source=None, verbose=True, consumed=None):
m10src/corpus_loader.py:416:    sample (the harvest file is grouped by form), and it is applied BEFORE the target lookup so a
m10src/corpus_loader.py:428:                new = [Segment(sg.name, sg.texts[:head_per_source], sg.forms[:head_per_source],
m10src/corpus_loader.py:430:            _texts, _forms, _ids, sman = source_texts(name)
m10src/corpus_loader.py:435:            texts, forms, ids, sman = source_texts(name, limit=head_per_source)
m10src/corpus_loader.py:455:            new = [Segment(name, texts, [FORM_ID[f] for f in forms], cache.vecs(), rowmap)]
m10src/corpus_loader.py:481:        for fid, c in zip(*np.unique(s.forms, return_counts=True)):
m10src/corpus_loader.py:504:    def __init__(self, flat, offs):
m10src/corpus_loader.py:508:    def __len__(self):
m10src/corpus_loader.py:511:    def __getitem__(self, i):
m10src/corpus_loader.py:515:def _publish_dir(tmp, d):
m10src/corpus_loader.py:528:def pack_tokenize(tok, texts, max_len=512, prefix="", batch=20_000, label="", verbose=True):
m10src/corpus_loader.py:559:def tokenizer_ident(tok):
m10src/corpus_loader.py:606:def tokenize_corpus(tok, segs, man, student, max_len=512, prefix="", cache=True, verbose=True,
m10src/corpus_loader.py:667:    def __init__(self, segs):
m10src/corpus_loader.py:671:    def __len__(self):
m10src/corpus_loader.py:674:    def __getitem__(self, idx):
m10src/corpus_loader.py:702:    Windows to **239 MB** and took the VM down twice on 2026-09-07 (`work/memtrace.csv`).
m10src/corpus_loader.py:718:    def __init__(self, store, rows):
m10src/corpus_loader.py:723:    def __len__(self):
m10src/corpus_loader.py:726:    def __getitem__(self, idx):
m10src/corpus_loader.py:743:def corpus_forms(segs):
m10src/corpus_loader.py:744:    return np.concatenate([s.forms for s in segs]) if segs else np.zeros(0, dtype=np.int16)
m10src/corpus_loader.py:747:# ------------------------------------------------------------------------------- the sampler ----
m10src/corpus_loader.py:749:def _form_draw(rows, lengths, batch_size, seed, form, occurrence):
m10src/corpus_loader.py:767:    forms from `default_rng([seed, c])`, and within a cycle each form contributes exactly one
m10src/corpus_loader.py:771:    `balanced=False` samples by example over the whole corpus (`data10.length_buckets`): the A2
m10src/corpus_loader.py:775:    def __init__(self, ids, targets, forms, pad_id, batch_size=32, seed=0, balanced=True,
m10src/corpus_loader.py:776:                require_forms=None):
m10src/corpus_loader.py:778:        self.forms = np.asarray(forms, dtype=np.int16)
m10src/corpus_loader.py:779:        self.balanced = balanced
m10src/corpus_loader.py:784:        if balanced:
m10src/corpus_loader.py:785:            self.present = [int(f) for f in np.unique(self.forms)]
m10src/corpus_loader.py:786:            self.rows = {f: np.flatnonzero(self.forms == f) for f in self.present}
m10src/corpus_loader.py:787:            # `require_forms` is the anchor/A4 contract: EVERY registered form must have rows, not
m10src/corpus_loader.py:789:            # fire -- `self.present` is built FROM the forms that survived `np.unique`, so a form
m10src/corpus_loader.py:790:            # missing entirely was silently absent from both the check and the sampler (Codex
m10src/corpus_loader.py:792:            if require_forms:
m10src/corpus_loader.py:794:                missing = [f for f in require_forms if f not in have]
m10src/corpus_loader.py:796:                    raise ValueError(f"forms with no rows: {missing}")
m10src/corpus_loader.py:805:    def __len__(self):
m10src/corpus_loader.py:810:    def _pick(self, k):
m10src/corpus_loader.py:813:        if not self.balanced:
m10src/corpus_loader.py:820:    def batch(self, k):
m10src/corpus_loader.py:825:    def realized_shares(self, n_batches):
m10src/corpus_loader.py:827:        §0b records. Counted from the rows actually drawn, so it is a measurement of the sampler
m10src/corpus_loader.py:828:        and not a restatement of its design -- and so the unbalanced variant reports the per-form
m10src/corpus_loader.py:832:        cnt = np.bincount(self.forms[idx], minlength=len(FORMS))
m10src/corpus_loader.py:839:def _registry_data_cut(registry):
m10src/corpus_loader.py:847:def data_cut_count(registry=None):
m10src/corpus_loader.py:854:def cut_arms(registry=None):
m10src/corpus_loader.py:864:def is_cut_corpus(name, registry=None):
m10src/corpus_loader.py:878:def apply_data_cut(segs, count, seed=0):
m10src/corpus_loader.py:879:    """-> (segments, report). Uniform seed-0 downsample of the WHOLE corpus to `count` rows.
m10src/corpus_loader.py:896:            out.append(Segment(s.name, [s.texts[int(i)] for i in sel], s.forms[sel], s.array,
m10src/corpus_loader.py:905:def _id_hashes(ids):
m10src/corpus_loader.py:914:def cross_role_collisions(q_ids, d_ids):
m10src/corpus_loader.py:926:def guard_cross_role(q_ids, d_ids, skip=False):
m10src/corpus_loader.py:940:def arm_sources(name, registry=None):
m10src/corpus_loader.py:944:    F/G/B/E/D/C) varies student, head, objective, dose or batch -- never data -- and trains on
m10src/corpus_loader.py:959:def build_query_stream(arm_or_sources, tok, student, *, batch_size=32, seed=0, balanced=True,
m10src/corpus_loader.py:961:                       allow_uncut=False, require_forms=None, verbose=True,
m10src/corpus_loader.py:967:    volumes and family A's forms contrast is confounded with volume, which is the one thing it
m10src/corpus_loader.py:1004:    stream = FormBalancedStream(ids, TargetView(segs), corpus_forms(segs),
m10src/corpus_loader.py:1006:                                balanced=balanced, require_forms=require_forms)
m10src/corpus_loader.py:1010:                "balanced": balanced, "n_batches": len(stream),
m10src/corpus_loader.py:1013:                "require_forms": list(require_forms) if require_forms else None})
m10src/corpus_loader.py:1017:def doc_marker():
m10src/corpus_loader.py:1023:def _screened_doc_pool(n, seed, banned, margin=1.05, floor=2_000):
m10src/corpus_loader.py:1063:def build_doc_stream(n, tok, *, batch_size=32, seed=0, max_len=512, allow_unscreened=False,
m10src/corpus_loader.py:1075:    uniform sample -- `doc_pool_rows` returns them in draw order, so a prefix of the survivors is
m10src/corpus_loader.py:1076:    a uniform sample of the survivors.
m10src/corpus_loader.py:1113:def release_arena():
m10src/corpus_loader.py:1132:    def _anon():
m10src/corpus_loader.py:1173:def _stream_doc_ids(rows, tok, prefix, max_len, chunk=DOC_TEXT_CHUNK, verbose=True,
m10src/corpus_loader.py:1286:# arms that must see all 12 registered forms: A4 IS the anchor (`ARM_SOURCES`), so both names land
m10src/corpus_loader.py:1291:def _registry_arms(reg):
m10src/corpus_loader.py:1297:def _registry_anchor(reg):
m10src/corpus_loader.py:1302:def resolve_arm_name(name, registry=None):
m10src/corpus_loader.py:1345:def arm_doc_count(arm, pattern, batch_size=32):
m10src/corpus_loader.py:1346:    """Documents an arm draws from the screened M9 pool: its document-EXAMPLE count (dose x the
m10src/corpus_loader.py:1347:    pattern's document share), so no document is presented twice at screen dose. M9 drew from
m10src/corpus_loader.py:1350:    dose = int(arm["dose_examples"])
m10src/corpus_loader.py:1351:    return max(int(math.ceil(dose * DOC_SHARE[pattern])), batch_size)
m10src/corpus_loader.py:1354:def resolve_arm_pattern(name, entry, reg):
m10src/corpus_loader.py:1387:def arm_batch(entry, reg):
m10src/corpus_loader.py:1394:def assemble_arm(arm_name, tok, student, *, batch_size=None, seed=0, max_len=512, verbose=True,
m10src/corpus_loader.py:1403:    own `pattern` or `anchor.mix`) and the document count (`arm_doc_count`, via `dose_examples`).
m10src/corpus_loader.py:1404:    `balanced` is always `True` -- the mandate's unbalanced variant is `build_query_stream`'s own
m10src/corpus_loader.py:1413:    see all 12 forms, raising and naming any missing -- unconditionally, since `balanced` can no
m10src/corpus_loader.py:1420:    require_forms = FORMS if name in REQUIRE_ALL_FORMS else None
m10src/corpus_loader.py:1434:                                        balanced=True, max_len=max_len, prefix="",
m10src/corpus_loader.py:1435:                                        allow_uncut=False, require_forms=require_forms,
m10src/corpus_loader.py:1458:          "n_docs": n_docs, "require_forms": list(require_forms) if require_forms else None,
m10src/corpus_loader.py:1468:    def __init__(self, lengths):
m10src/corpus_loader.py:1471:    def __len__(self):
m10src/corpus_loader.py:1475:def main():
m10src/corpus_loader.py:1482:    ap.add_argument("--unbalanced", action="store_true")
m10src/corpus_loader.py:1485:    man, forms, words = {"sources": [], "sources_used": a.sources}, [], []
m10src/corpus_loader.py:1489:        forms += [FORM_ID[x] for x in f]
m10src/corpus_loader.py:1492:    forms = np.asarray(forms, dtype=np.int16)
m10src/corpus_loader.py:1494:    man["n_rows"] = int(len(forms))
m10src/corpus_loader.py:1495:    man["by_form"] = {FORMS[int(f)]: int(c) for f, c in zip(*np.unique(forms, return_counts=True))}
m10src/corpus_loader.py:1496:    st = FormBalancedStream(LengthsOnly(words), None, forms, pad_id=0,
m10src/corpus_loader.py:1497:                            batch_size=a.batch_size, seed=0, balanced=not a.unbalanced)
m10src/corpus_loader.py:1498:    man["balanced"] = not a.unbalanced
m10src/run_arm.py:8:(`assemble_arm`, the ONLY corpus path), `nano10` (student, head, losses, cycle ends, LR),
m10src/run_arm.py:9:`trainer10` (loop, checkpoint, resume, kill/plateau), `cov_eval10`/`cov_macro` (the COV macro),
m10src/run_arm.py:31:    the kill/plateau state: §Kill compares "midpoint against midpoints, cycle end against cycle
m10src/run_arm.py:37:**Success is narrow** (`classify`): `stopped is None`, or exactly `plateau at cycle 3` —
m10src/run_arm.py:38:`PLATEAU_FROM_CYCLE` is 3 and a screen arm runs 3 cycles, so that plateau means the arm finished
m10src/run_arm.py:39:its dose and stopped one step short. Every KILL and every non-finite stop is FAILED, the final
m10src/run_arm.py:58:import traceback
m10src/run_arm.py:108:# ridge warm start without a sample size. Read from the registry's G-MLP block, defaulting to
m10src/run_arm.py:109:# `m9/registry.json` `warm_start` (n_fit 60,000, seed 21) — the sample `calib.py` also used.
m10src/run_arm.py:121:def refuse(msg):
m10src/run_arm.py:125:def slug(name):
m10src/run_arm.py:130:def rel(p):
m10src/run_arm.py:139:def sha256_file(p, chunk=1 << 22):
m10src/run_arm.py:147:def git_head():
m10src/run_arm.py:155:def cfg():
m10src/run_arm.py:159:def record_paths(arm, smoke):
m10src/run_arm.py:169:def write_record(rec, rec_path, results_path):
m10src/run_arm.py:186:def _record_status_at(p):
m10src/run_arm.py:206:def f_arms(reg):
m10src/run_arm.py:212:def f_verdict(reg, path=None):
m10src/run_arm.py:294:def shape_for(arm):
m10src/run_arm.py:297:    cover). `COVERS` maps the arms that differ only in data or dose onto the shape they share."""
m10src/run_arm.py:306:def resolved_recipe(name, entry, reg, pattern, batch):
m10src/run_arm.py:317:        "objective": entry.get("objective") or a.get("objective") or "squared_l2",
m10src/run_arm.py:332:def check_shape(arm, recipe, spec):
m10src/run_arm.py:348:    for field, in_spec in (("objective", "loss"), ("feature_layers", "n_layers"),
m10src/run_arm.py:359:def schedule(dose, batch, entry):
m10src/run_arm.py:362:    The schedule is 3 cycles over THE ARM'S OWN dose (F's 20M arms therefore anneal over 20M and
m10src/run_arm.py:363:    are merely READ at 5M/10M/20M), so `total_steps = dose // batch` and the cycle ends are
m10src/run_arm.py:366:    **Registry ambiguity, floored.** `E-bs128`'s registered dose 5,000,000 is not a whole number
m10src/run_arm.py:368:    exact at both batch sizes. The dose is treated as a CAP: 39,062 steps = 4,999,936 examples, 64
m10src/run_arm.py:372:    total = dose // batch
m10src/run_arm.py:373:    rounding = {"dose_registered": int(dose), "examples_run": int(total * batch),
m10src/run_arm.py:374:                "examples_dropped": int(dose - total * batch), "rule": "floor (the dose is a cap)"}
m10src/run_arm.py:379:        if r > dose:
m10src/run_arm.py:380:            refuse(f"read_at {r:,} exceeds the arm's dose {dose:,}")
m10src/run_arm.py:388:def arm_plan(name, reg, verdict=None):
m10src/run_arm.py:408:    dose = int(entry["dose_examples"])
m10src/run_arm.py:409:    total, ends, reads, rounding = schedule(dose, batch, entry)
m10src/run_arm.py:412:            "dose_examples": dose, "batch": batch, "pattern": pattern,
m10src/run_arm.py:418:            "objective": recipe["objective"], "warm_start": recipe["warm_start"],
m10src/run_arm.py:422:            "total_steps": total, "cycle_end_steps": ends, "dose_rounding": rounding,
m10src/run_arm.py:430:def plan(reg, order=None):
m10src/run_arm.py:441:    hdr = (f"{'arm':13s} {'dose':>10s} {'bs':>4s} {'pattern':>8s} {'student':11s} "
m10src/run_arm.py:442:           f"{'objective':28s} {'warm':6s} {'cut':4s} {'steps':>8s} {'h':>6s}  where")
m10src/run_arm.py:451:        print(f"{r['arm']:13s} {r['dose_examples']:>10,} {r['batch']:>4d} {r['pattern']:>8s} "
m10src/run_arm.py:452:              f"{st:11s} {r['objective']:28s} {r['warm_start']:6s} "
m10src/run_arm.py:474:    def __init__(self, model, out_dir, batch_size=256, verbose=True):
m10src/run_arm.py:479:    def units(self):
m10src/run_arm.py:485:    def __call__(self, step, label):
m10src/run_arm.py:516:    def state(self):
m10src/run_arm.py:519:    def restore(self, d):
m10src/run_arm.py:544:    Returns a deterministic increasing series so the plateau rule cannot fire spuriously."""
m10src/run_arm.py:546:    def __init__(self):
m10src/run_arm.py:549:    def __call__(self, step, label):
m10src/run_arm.py:555:    def state(self):
m10src/run_arm.py:558:    def restore(self, d):
m10src/run_arm.py:562:def dev6(model, verbose=True):
m10src/run_arm.py:586:def streams_of(batch_fn):
m10src/run_arm.py:591:    arm's own fit texts and teacher targets, and D-COV's Sigma is the covariance of the pool's
m10src/run_arm.py:610:def fit_sample(q_stream, n_fit, seed):
m10src/run_arm.py:615:    the registered sample (`warm_start`, `m9/registry.json` n_fit 60,000 / seed 21).
m10src/run_arm.py:629:def warm_start(model, spec, q_stream, ws_cfg, verbose=True):
m10src/run_arm.py:635:    texts, Y = fit_sample(q_stream, ws_cfg["n_fit"], ws_cfg["seed"])
m10src/run_arm.py:647:        import traceback
m10src/run_arm.py:648:        print(traceback.format_exc()[-1500:], flush=True)
m10src/run_arm.py:655:              f"train objective {rec.get('train_objective')} ({rec['seconds']:.0f}s)", flush=True)
m10src/run_arm.py:659:def classify(stopped, n_cycle_ends, cycles=CYCLES):
m10src/run_arm.py:660:    """-> (status, ok, plateau_at_last). SUCCESS is `stopped is None` or exactly "plateau at
m10src/run_arm.py:667:    plateau_at_last = stopped == f"plateau at cycle {cycles}"
m10src/run_arm.py:668:    ok = bool(n_cycle_ends >= cycles and (stopped is None or plateau_at_last))
m10src/run_arm.py:669:    return ("complete" if ok else "failed"), ok, plateau_at_last
m10src/run_arm.py:677:def code_identity():
m10src/run_arm.py:688:def fingerprint(arm, p, man, seed, smoke_steps, device):
m10src/run_arm.py:695:            "recipe": {k: p[k] for k in ("dose_examples", "batch", "pattern", "student",
m10src/run_arm.py:696:                                         "n_layers", "head", "objective", "warm_start",
m10src/run_arm.py:698:            "warmup_steps": int(N.WARMUP_STEPS),
m10src/run_arm.py:714:def run(arm, *, device="cpu", resume=False, smoke_steps=None, max_len=None, ckpt_every=None,
m10src/run_arm.py:741:def _record_failure(ctx, arm, exc, verbose=True):
m10src/run_arm.py:759:                       "traceback_tail": traceback.format_exc()[-4000:]},
m10src/run_arm.py:777:def _run(ctx, arm, *, device, resume, smoke_steps, max_len, ckpt_every, n_fit, real_eval,
m10src/run_arm.py:811:    # tiny dose rather than 1.25M documents, and its OWN output tree (§Hazards: smokes have
m10src/run_arm.py:815:        reg["arms"][arm]["dose_examples"] = int(smoke_steps) * base["batch"]
m10src/run_arm.py:847:    p = arm_plan(arm, reg)                      # re-read: a smoke changed the dose
m10src/run_arm.py:883:    print(f"=== arm {arm} ({p['family']}) on {device}: {p['dose_examples']:,} examples, "
m10src/run_arm.py:885:          f"student {p['student']}, objective {p['objective']}, "
m10src/run_arm.py:904:    if p["objective"] == "document_covariance_weighted":
m10src/run_arm.py:913:    def eval_fn(_m, step, kind):
m10src/run_arm.py:931:                     cycles=CYCLES, peak=PEAK, final=FINAL, loss_name=p["objective"],
m10src/run_arm.py:938:    status, ok, plateau_at_last = classify(r["stopped"], len(r["cycle_end_evals"]))
m10src/run_arm.py:965:        "recipe": {k: p[k] for k in ("dose_examples", "batch", "pattern", "pattern_source",
m10src/run_arm.py:966:                                     "student", "n_layers", "head", "objective", "warm_start",
m10src/run_arm.py:968:                                     "cycle_end_steps", "read_points", "dose_rounding")},
m10src/run_arm.py:970:                     "_what": "3 cycles of equal example count over the ARM'S OWN dose, each "
m10src/run_arm.py:987:        "loss_tail": r["losses"][-200:],
m10src/run_arm.py:989:        "stopped_is_completion": plateau_at_last,
m10src/run_arm.py:990:        "_stopped_note": ("SUCCESS is `stopped is None` or exactly `plateau at cycle 3`: "
m10src/run_arm.py:992:                          "plateau means the arm finished its dose and stopped one step short "
m10src/run_arm.py:993:                          "(LEDGER §M10.0-e trace). EVERY kill and every non-finite stop is "
m10src/run_arm.py:1006:        "registry_dose_overridden_for_smoke": (reg["arms"][arm]["dose_examples"] if smoke else None),
m10src/run_arm.py:1019:def build_argparser():
m10src/run_arm.py:1031:                    help="a SMOKE: N steps, its own dose, stub evals, and its own output tree "
m10src/run_arm.py:1042:                    help="SMOKE ONLY warm-start fit sample; a registered arm uses 60,000")
m10src/run_arm.py:1049:def main(argv=None):
m10src/trainer10.py:9:  plateau rules read off those evaluations;
m10src/trainer10.py:31:def eager(model):
m10src/trainer10.py:36:def param_groups(model, wd=0.01):
m10src/trainer10.py:45:def autocast_for(device, dtype=torch.bfloat16):
m10src/trainer10.py:46:    """§Recipe: "fp32 loss, bf16 autocast" — the form `rate_bench_real` measured the 910 ex/s on.
m10src/trainer10.py:49:    and the screens' CPU smokes are a path check, not a rate. The loss is computed OUTSIDE this
m10src/trainer10.py:56:def save(path, model, opt, step, extra=None):
m10src/trainer10.py:78:def load(path, model, opt):
m10src/trainer10.py:88:def train_arm(model, batch_fn, total_steps, *, pattern="75/25", cycles=3, peak=1e-4, final=1e-5,
m10src/trainer10.py:89:              loss_name="squared_l2", sigma=None, eval_fn=None, evals_per_cycle=2,
m10src/trainer10.py:93:    """Run one arm. -> a record: losses, evaluations, rates, and why it stopped.
m10src/trainer10.py:100:    `read_evals`, **outside** the kill/plateau state: §Kill compares "midpoint against midpoints,
m10src/trainer10.py:107:    before the kill/plateau test, so an arm that stops AT a cycle end still has that checkpoint.
m10src/trainer10.py:118:    loss_fn = N.LOSSES[loss_name]
m10src/trainer10.py:122:    losses, evals, kinds, cycle_end_evals, read_evals = [], [], [], [], []
m10src/trainer10.py:126:        # `cycle_end_evals` empty, so the plateau rule reads one cycle where it needs three and
m10src/trainer10.py:127:        # the registered kill/plateau decision cannot fire at all (Codex 2026-09-05 finding 4).
m10src/trainer10.py:136:        losses = list(ex.get("losses", []))
m10src/trainer10.py:148:    def extra():
m10src/trainer10.py:149:        e = {"losses": losses, "evals": evals, "eval_kinds": kinds,
m10src/trainer10.py:171:        loss = loss_fn(pred, tgt.to(device)) if sigma is None \
m10src/trainer10.py:172:            else loss_fn(pred, tgt.to(device), sigma)
m10src/trainer10.py:173:        if not torch.isfinite(loss):
m10src/trainer10.py:174:            stopped = f"non-finite loss at step {step}"
m10src/trainer10.py:175:            losses.append(float("nan"))
m10src/trainer10.py:178:        loss.backward()
m10src/trainer10.py:184:        losses.append(float(loss.detach()))
m10src/trainer10.py:195:                # part of the kill/plateau state (see the docstring).
m10src/trainer10.py:208:                pf, at = N.plateau_fires(cycle_end_evals)
m10src/trainer10.py:210:                    stopped = f"plateau at cycle {at}"
m10src/trainer10.py:216:            print(f"  step {step + 1}/{total_steps} loss {np.mean(losses[-log_every:]):.4f} "
m10src/trainer10.py:222:        # the plateau that ended the run (Codex re-review 2026-09-05). This must fire even when
m10src/trainer10.py:227:    # `losses`/`evals` are the WHOLE arm's history (restored on resume); `steps_run` and the rate
m10src/trainer10.py:231:            "loss": loss_name, "losses": losses, "evals": evals, "eval_kinds": kinds,
m10src/trainer10.py:236:            "warmup_steps": int(N.WARMUP_STEPS),
m10src/trainer10.py:237:            "mix": N.window_shares(pattern, max(len(losses), 1))}
m10src/screen_lock.py:25:def cfg():
m10src/screen_lock.py:29:def validate(r=None):
m10src/screen_lock.py:60:    def resolve(name):
m10src/screen_lock.py:73:                if arm and at > max([arm.get("dose_examples", 0)]
m10src/screen_lock.py:76:                    bad.append(f"contrast {cid} reads {c[side]} at {at:,} examples, beyond its dose")
m10src/screen_lock.py:124:    # -- the anchor's own settings must agree with every axis it stands in for

 succeeded in 0ms:
     1	# M14 — the paper
     2	
     3	Created 2026-09-04 (Dylan): *"paper writing can be its own milestone after building nano. We'll
     4	decide then if we want to do 2 papers or 1 but probably only 1."* Split out of M13, which held it
     5	as deliverable 4 and would have kept it hostage to a milestone that may never run.
     6	
     7	Binds from `instructions-m7.md` unchanged. Working files under `m14/`.
     8	
     9	## Sequencing
    10	
    11	**Runs after M13 delivers nano.** If M10/M13 never run, M14 still runs — on the zero-only frontier
    12	plus whatever M12 produced. **One paper is the default**; the one-vs-two decision is Dylan's, taken
    13	at the start of M14 when the evidence is known, not now.
    14	
    15	## Benchmark composition — deferred here, but the RULE is registered now
    16	
    17	**Deferred to M14 (Dylan, 2026-09-04):** *"Benchmarks can be differed to M14, when we do the paper
    18	so that way we'll cross that bridge for both Zero and nano."* Right call — the product is the pair,
    19	and settling zero's partition now while nano's waits risks reporting the two halves of one frontier
    20	on different benchmarks.
    21	
    22	**But the choice is registered BEFORE nano's numbers exist, not made at M14 having seen them.**
    23	Deferring the decision is legitimate; deferring the *rule* would be post-hoc selection among
    24	pre-registered partitions, which is the one thing that would discredit the frontier. So:
    25	
    26	1. **Headline = clean-4** (`nfcorpus`, `scidocs`, `scifact`, `trec-covid`) — the pre-registered
    27	   no-disclosed-teacher-overlap partition (`results/m7_final_run.json` `clean4_robustness`).
    28	   Registered now, on the contamination argument alone, for **both** zero and nano.
    29	2. **All six always reported beside it**, for comparability with published BEIR numbers.
    30	3. **Contamination robustness (all-6 minus clean-4) is its own reported row** for dense and fused.
    31	4. **No re-picking the six.** Datasets may not be added to or removed from the headline after any
    32	   number is seen. New sets may only enter as labelled diagnostics.
    33	5. **MS MARCO, if built, is a validation-only diagnostic** — never a headline. Permitted by the
    34	   2026-09-04 licence rule (validation, not training). It is the MIRROR confound: every comparator
    35	   trains on it and neither of ours does, so we expect to lose and losing is still informative.
    36	   nq-250k-style build (~6,980 queries, 250K distractors, a 250K-doc stella encode, ~1h).
    37	
    38	**The number that makes this cheap** (M12, `m12/FINDINGS.md`): FUSED systems are nearly
    39	contamination-immune — clean-4 costs convex0 **0.0045** (0.4911 → 0.4866) and *gains* DBSF@100
    40	**0.0025** (0.4887 → 0.4912) — while the dense table loses **0.0241** (0.4339 → 0.4098). And on clean-4 the fused story is *stronger*: 0.4866 vs BM25's 0.4409,
    41	where dense-only (0.4098) sits below BM25. Honesty costs 0.0045 here and buys the whole objection.
    42	
    43	## What the paper argues
    44	
    45	The frontier is the *deliverable*, but it is not the *finding*. The finding is:
    46	
    47	**Teacher quality does not predict distilled-table quality.** Spearman ≈ 0 (ceiling→table 0.000 over
    48	eight; MTEB→table −0.11 over seven verified rows) across **eleven** measured teachers. The
    49	best-ceiling candidate ranked 7th. `mxbai-embed-large-v1` has the same measured tower quality as
    50	`bge-base-en-v1.5` (0.4433 vs 0.4484) while its table is **0.057 worse** — three times the
    51	parameters, higher MTEB, worse product. Selection must run on the distilled artifact, never the
    52	tower. Survives a contamination control on data the winner never trained on (SQuAD +0.1652, ESCI
    53	+0.0384, `results/m7_offfamily_report.json`).
    54	
    55	This is measured today and needs no nano. Everything else is evidence for it or around it.
    56	
    57	## Contents
    58	
    59	1. The finding above, with the eleven-candidate table and the exposure-free control.
    60	2. The quality-vs-query-cost frontier: `zero` and `nano` (M13, if it lands), against the
    61	   21-system × 6-dataset matrix, plus M12's fusion-operator result. (`zero-hybrid` was **cut** from
    62	   M12 on 2026-09-04 and moved to M16; it is not an M14 deliverable.)
    63	3. Edge cost rows — query asset, document index, hydration, CPU latency — plus the offline BM25
    64	   index cost M12 measures.
    65	4. The Qdrant Edge prototype: architecture and latency, with the exact-search caveat.
    66	5. **The comparator table deliberately kept OFF the model cards** (`instructions-m11.md`
    67	   Amendment B): `LR-dense-pertask 0.4583`, the OpenSearch tie, the missed bar. This is its home.
    68	6. Negative results, stated as results: M8's twelve levers, M9's coverage failure, and every kill.
    69	
    70	## Rules that bind the writing
    71	
    72	- **Every headline number is paired on frozen comparator vectors with pre-registered statistics.**
    73	  No number enters the paper that is not already in a committed result JSON.
    74	- **stella's ArguAna/FiQA/FEVER exposure is disclosed wherever a stella-derived number appears.**
    75	- **Non-commercially-licensed sets may be reported as VALIDATION rows** (Dylan, 2026-09-04;
    76	  `research/m7-data-licensing.md` §Rule change 2026-09-04) — an MS MARCO row is now reportable, which
    77	  reviewers will look for. Two things must appear beside it: that it informed no training, and that
    78	  every comparator trains on MS MARCO while `zero` and `nano` do not, so the row is biased **against
    79	  us**. The training exclusion and its +0.0058 price are reported unchanged.
    80	- **Deltas inside the ~0.005 lever band are labelled as ties**, not wins.
    81	- Do not quote the off-family macro (+0.1018) as a product delta — it is a contamination control.
    82	- Do not claim `zero` confirmatorily beat BM25: C2 was +0.0165 [0.0017, 0.0311], Holm p=0.0149 >
    83	  0.0083, **not rejected**.
    84	- **Cite pyNIFE** (Stephan Tulkens, MIT, 2025-11-03) as prior art for the construction — a per-token
    85	  dense lookup table distilled from a frozen teacher, reusing that teacher's index unchanged. It was
    86	  dropped from the model card 2026-09-04 as card clutter; a paper without it would be a different
    87	  matter. Its retention comparison on fiqa (73.4% of a weaker teacher vs our 67.1%) is also the
    88	  cheap lead in M16.
    89	- No new measurement. If the paper wants a number that does not exist, that is a milestone, not a
    90	  paragraph.

exec
/bin/bash -lc "nl -ba m10src/screen_lock.py
nl -ba m10src/cov_macro.py
nl -ba m10src/corpus_loader.py | sed -n '1280,1385p'" in /home/dylan/asymetric-dual-encoders
 succeeded in 0ms:
     1	"""The M10 screen lock: `m10/screen_registry.json` is the authority, and this validates it.
     2	
     3	M9's lesson, copied deliberately (`m9src/final_stats.py`): *"Constants come from the registry.
     4	Prose is not authoritative; the registry is."* A rule that reads a number reads it from here, so
     5	a mandate sentence and a running arm cannot disagree silently.
     6	
     7	What this refuses, and why each one has bitten something before:
     8	- a contrast naming an arm that does not exist, or an arm no contrast reads;
     9	- a contrast count that is not thirteen (the Bonferroni denominator is 13; a fourteenth contrast
    10	  silently changes every bound -- a Codex pass corrected exactly that count on 2026-09-04);
    11	- statistics that disagree with `m10src/cov_macro`'s implementation or with the measured
    12	  resolution number's artifact;
    13	- a family with no outcome->action entry (an unresolved contrast with no registered default is a
    14	  decision taken after seeing the number).
    15	"""
    16	import json, sys
    17	from pathlib import Path
    18	
    19	REPO = Path(__file__).resolve().parents[1]
    20	sys.path.insert(0, str(REPO / "m10src"))
    21	REGISTRY = REPO / "m10" / "screen_registry.json"
    22	RESOLUTION = REPO / "results" / "m10_cov_resolution.json"
    23	
    24	
    25	def cfg():
    26	    return json.loads(REGISTRY.read_text())
    27	
    28	
    29	def validate(r=None):
    30	    """-> list of problems; empty means the lock is coherent.
    31	
    32	    Every check below exists because a Fable or Codex pass showed the registry passing without
    33	    it (S10, 2026-09-05): a dozen mutations a reviewer would reject validated clean.
    34	    """
    35	    import cov_macro
    36	    r = r or cfg()
    37	    bad = []
    38	    arms, contrasts, alias = r["arms"], r["contrasts"], r["anchor_aliases"]
    39	    NON_FAMILY = {"anchor"}
    40	
    41	    # -- arms and the trained count. A boolean, not a note-prefix heuristic: an arm could be
    42	    # moved into the "not trained" bucket by editing prose.
    43	    for a, v in arms.items():
    44	        if not isinstance(v.get("trained"), bool):
    45	            bad.append(f"arm {a} has no boolean `trained` field")
    46	    trained = [a for a, v in arms.items() if v.get("trained") is True]
    47	    if len(trained) != r["trained_arms_expected"]:
    48	        bad.append(f"{len(trained)} trained arms, registry expects {r['trained_arms_expected']}")
    49	    if len(contrasts) != r["statistics"]["bootstrap"]["n_contrasts"]:
    50	        bad.append(f"{len(contrasts)} contrasts vs a Bonferroni denominator of "
    51	                   f"{r['statistics']['bootstrap']['n_contrasts']}")
    52	
    53	    # -- aliases must land on a real arm, or they are a name for nothing
    54	    for k, v in alias.items():
    55	        if k.startswith("_") or not isinstance(v, str):
    56	            continue
    57	        if v not in arms and not v.startswith("the winner"):
    58	            bad.append(f"anchor alias {k} -> '{v}' is not an arm")
    59	
    60	    def resolve(name):
    61	        return alias.get(name, name)
    62	    for cid, c in contrasts.items():
    63	        for side in ("a", "b"):
    64	            if c[side] not in arms and c[side] not in alias:
    65	                bad.append(f"contrast {cid}: '{c[side]}' is neither an arm nor an anchor alias")
    66	        rule = c.get("rule")
    67	        if rule and rule not in r.get("rules", {}):
    68	            bad.append(f"contrast {cid} names rule '{rule}', which is not in `rules`")
    69	        at = c.get("at_examples")
    70	        if at:
    71	            for side in ("a", "b"):
    72	                arm = arms.get(resolve(c[side]))
    73	                if arm and at > max([arm.get("dose_examples", 0)]
    74	                                    + list(arm.get("read_at", []))
    75	                                    + [arm.get("conditional_extension", {}).get("to_examples", 0)]):
    76	                    bad.append(f"contrast {cid} reads {c[side]} at {at:,} examples, beyond its dose")
    77	    # -- W9 review: the W9 fixes were PROSE and the validator never evaluated them, so a
    78	    # misspelled `family_rule` or an ignored per-contrast quantile passed the lock. These check the
    79	    # fields a decision actually reads.
    80	    st = r["statistics"]["bootstrap"]
    81	    q, alpha = st["quantile"], st["alpha_family"]
    82	    total = 0.0
    83	    for cid, c in contrasts.items():
    84	        tails = c.get("tails", 1)
    85	        if tails not in (1, 2):
    86	            bad.append(f"contrast {cid}: tails={tails!r}, expected 1 or 2")
    87	        cq = c.get("quantile", q)
    88	        if tails == 2 and abs(cq - q / 2) > 1e-15:
    89	            bad.append(f"contrast {cid} is two-sided but its quantile {cq} != alpha/(2*n)")
    90	        if tails == 1 and "quantile" in c and abs(cq - q) > 1e-15:
    91	            bad.append(f"contrast {cid} overrides the quantile without being two-sided")
    92	        total += tails * cq
    93	        fr = c.get("family_rule")
    94	        if fr and fr not in r.get("rules", {}):
    95	            bad.append(f"contrast {cid} names family_rule '{fr}', which is not in `rules`")
    96	    if abs(total - alpha) > 1e-12:
    97	        bad.append(f"familywise alpha is {total}, not the registered {alpha}")
    98	
    99	    # -- an arm that cannot be selected must not be reachable as a winner
   100	    for a, v in arms.items():
   101	        if v.get("trained") and "selectable" not in v and v.get("family") not in NON_FAMILY:
   102	            if a not in {alias.get(k, k) for k in alias} and v.get("family") != "F":
   103	                bad.append(f"arm {a} has no boolean `selectable` field")
   104	    if arms.get("G-384", {}).get("selectable") is not False:
   105	        bad.append("G-384 must be registered selectable:false -- it is the M9 diagnostic arm")
   106	
   107	    # -- a conditionally-skippable arm needs a conditional count; a CUT arm must not be trained
   108	    if arms.get("C-M9init", {}).get("skipped_iff"):
   109	        if r.get("trained_arms_if_C_skipped") != r["trained_arms_expected"] - 1:
   110	            bad.append("C is skippable but trained_arms_if_C_skipped != expected - 1")
   111	    for a, v in arms.items():
   112	        if v.get("cut") and v.get("trained"):
   113	            bad.append(f"arm {a} is marked cut but still trained: true")
   114	        if v.get("cut") and r.get("trained_arms_if_C_skipped") is not None and a == "C-M9init":
   115	            bad.append("C is CUT, not skippable: trained_arms_if_C_skipped is meaningless")
   116	
   117	    everything = list(contrasts.values()) + list(r["descriptive_contrasts"].values())
   118	    read = {resolve(c[s]) for c in everything for s in ("a", "b") if c.get(s)}
   119	    for a, v in arms.items():
   120	        if a in read or a in alias.values() or v.get("family") == "F" or not v.get("trained"):
   121	            continue
   122	        bad.append(f"arm {a} is trained but no contrast reads it")
   123	
   124	    # -- the anchor's own settings must agree with every axis it stands in for
   125	    anc = r["anchor"]
   126	    for k, field, want in (("E-bs32", "batch", 32), ("G-1152", "feature_dim", 1152),
   127	                           ("B-75/25", "mix", "75/25")):
   128	        if k in alias and anc.get(field) != want:
   129	            bad.append(f"alias {k} claims the anchor is {want}, but anchor.{field} = "
   130	                       f"{anc.get(field)!r}")
   131	
   132	    fams = {v["family"] for v in arms.values() if "family" in v}
   133	    for f in fams:
   134	        v = r["outcome_to_action"].get(f)
   135	        if not v or not str(v).strip():
   136	            bad.append(f"family {f} has no outcome->action entry")
   137	    if set(r["order"]) != fams - NON_FAMILY:
   138	        bad.append(f"order {r['order']} does not cover the decision families "
   139	                   f"{sorted(fams - NON_FAMILY)}")
   140	
   141	    st = r["statistics"]
   142	    if tuple(sorted(st["families"])) != tuple(sorted(cov_macro.FAMILIES)):
   143	        bad.append(f"registry families {st['families']} != cov_macro.FAMILIES "
   144	                   f"{list(cov_macro.FAMILIES)}")
   145	    b = st["bootstrap"]
   146	    if b.get("alpha_family") != 0.025:
   147	        bad.append(f"alpha_family is {b.get('alpha_family')}, not the registered 0.025")
   148	    if abs(b["quantile"] - b["alpha_family"] / b["n_contrasts"]) > 1e-15:
   149	        bad.append("quantile is not alpha_family / n_contrasts")
   150	    if b["quantile_method"] != "inverted_cdf":
   151	        bad.append("quantile_method is not the registered inverted_cdf")
   152	
   153	    # -- the resolution artifact is REQUIRED: absent, three checks would silently not run
   154	    if not RESOLUTION.exists():
   155	        bad.append(f"{RESOLUTION.name} is missing; the power disclosure cannot be checked")
   156	    else:
   157	        got = json.loads(RESOLUTION.read_text())
   158	        if abs(got["resolution_distance"] - st["measured_resolution_distance"]) > 5e-7:
   159	            bad.append(f"registry resolution {st['measured_resolution_distance']} != artifact "
   160	                       f"{got['resolution_distance']}")
   161	        if got["MDE_registered"] != st["MDE"]:
   162	            bad.append("MDE disagrees with the resolution artifact")
   163	        if not got["mde_below_resolution"]:
   164	            bad.append("artifact says the MDE is at or above the resolution distance")
   165	        for k in ("B", "seed", "chunk", "quantile_method"):
   166	            if got["bootstrap"][k] != b.get(k):
   167	                bad.append(f"bootstrap.{k}: registry {b.get(k)!r} != artifact "
   168	                           f"{got['bootstrap'][k]!r}")
   169	    return bad
   170	
   171	
   172	if __name__ == "__main__":
   173	    problems = validate()
   174	    for p in problems:
   175	        print("PROBLEM:", p)
   176	    print("LOCK OK" if not problems else f"{len(problems)} PROBLEMS")
   177	    sys.exit(1 if problems else 0)
     1	"""The COV macro and its paired stratified bootstrap — the contrast rule of §Screen.
     2	
     3	`m9src/final_stats.bootstrap` is the registered M9 implementation and is hardcoded to six equally
     4	weighted datasets (`k != 6` raises). The COV surface is **family-weighted**: four family IDs at
     5	equal weight, units averaged within family, so BRIGHT's six slices carry 1/24 each and MedicalQA
     6	carries 1/4 on its own. That is a different estimator, so it is a separate module rather than a
     7	loosened one — M9's gate keeps its refusal.
     8	
     9	Registered constants (§Screen): B = 200,000, seed 0, `inverted_cdf`, one-sided lower bound at
    10	0.025/13. Nothing here reads them from prose; the caller passes them and `record()` stamps them.
    11	
    12	The draw plan is materialised per unit from `SeedSequence(seed).spawn()` in **sorted unit order**,
    13	consumed in fixed `chunk`-sized blocks, so it is reproducible from (seed, B, chunk, unit list) and
    14	its digest is recorded. B = 200,000 x 13,416 queries is 21 GB as one array; chunking is what makes
    15	it runnable, and it is therefore part of the plan definition, not an implementation detail.
    16	"""
    17	from __future__ import annotations
    18	
    19	import hashlib
    20	import numpy as np
    21	
    22	FAMILIES = ("BRIGHT", "consumer-health", "finance", "legal")
    23	
    24	# The admitted COV surface, locked (`m10/LEDGER.md` §2). This is the M9 `_assert_six` lesson
    25	# transplanted: checking only that observed labels are *known* lets a dropped unit or a dropped
    26	# family renormalise silently -- a three-family macro, or a five-slice BRIGHT, would still report
    27	# as a clean number. A Codex pass caught exactly that hole here (2026-09-05).
    28	SURFACE = {
    29	    "MedicalQARetrieval": "consumer-health",
    30	    "BRIGHT/biology": "BRIGHT", "BRIGHT/earth_science": "BRIGHT",
    31	    "BRIGHT/economics": "BRIGHT", "BRIGHT/psychology": "BRIGHT",
    32	    "BRIGHT/robotics": "BRIGHT", "BRIGHT/sustainable_living": "BRIGHT",
    33	    "LegalBenchCorporateLobbying": "legal", "LegalBenchConsumerContractsQA": "legal",
    34	    "LEDGER": "finance",
    35	}
    36	
    37	
    38	def assert_surface(unit_family):
    39	    """The macro is defined on exactly the admitted surface, or not at all."""
    40	    if dict(unit_family) != SURFACE:
    41	        missing = sorted(set(SURFACE) - set(unit_family))
    42	        extra = sorted(set(unit_family) - set(SURFACE))
    43	        relabelled = sorted(u for u in set(SURFACE) & set(unit_family)
    44	                            if unit_family[u] != SURFACE[u])
    45	        raise ValueError(f"COV surface does not match the admitted lock: missing={missing} "
    46	                         f"extra={extra} relabelled={relabelled}")
    47	
    48	
    49	def weights(unit_family):
    50	    """{unit: family} -> {unit: weight}. Families equal, units equal within family.
    51	
    52	    No bypass flag: an escape hatch on a surface lock is the lock's own defect.
    53	    """
    54	    assert_surface(unit_family)
    55	    fams = sorted(set(unit_family.values()))
    56	    if not set(fams) <= set(FAMILIES):
    57	        raise ValueError(f"unknown family in {fams}; registered are {FAMILIES}")
    58	    n_in = {f: sum(1 for v in unit_family.values() if v == f) for f in fams}
    59	    return {u: 1.0 / (len(fams) * n_in[f]) for u, f in unit_family.items()}
    60	
    61	
    62	def macro(per_unit, unit_family):
    63	    """{unit: {qid: score}} -> (macro, {family: mean}, {unit: mean})."""
    64	    w = weights(unit_family)
    65	    um = {u: float(np.mean(list(v.values()))) for u, v in per_unit.items()}
    66	    if set(um) != set(w):
    67	        raise ValueError(f"unit mismatch: scored {sorted(um)} vs weighted {sorted(w)}")
    68	    fm = {f: float(np.mean([um[u] for u, g in unit_family.items() if g == f]))
    69	          for f in sorted(set(unit_family.values()))}
    70	    return float(sum(w[u] * um[u] for u in um)), fm, um
    71	
    72	
    73	def align(a, b, unit_family):
    74	    """Two {unit: {qid: score}} -> {unit: (qids, x, y)}, strict: same units, same qids."""
    75	    if set(a) != set(b) or set(a) != set(unit_family):
    76	        raise ValueError(f"unit sets differ: {sorted(a)} / {sorted(b)} / {sorted(unit_family)}")
    77	    out = {}
    78	    for u in sorted(a):
    79	        qa, qb = set(a[u]), set(b[u])
    80	        if qa != qb:
    81	            raise ValueError(f"{u}: {len(qa)} vs {len(qb)} qids, {len(qa ^ qb)} unshared")
    82	        if not qa:
    83	            raise ValueError(f"{u}: empty")
    84	        qids = sorted(qa)
    85	        out[u] = (qids, np.array([a[u][q] for q in qids], dtype=np.float64),
    86	                  np.array([b[u][q] for q in qids], dtype=np.float64))
    87	    return out
    88	
    89	
    90	def contrast(aligned, unit_family, B=200_000, seed=0, quantile=0.025 / 13,
    91	             method="inverted_cdf", chunk=5_000):
    92	    """Paired stratified bootstrap of the family-weighted macro difference (a - b).
    93	
    94	    Returns the point estimate, the one-sided lower bound at `quantile`, and the DISTANCE
    95	    between them, which is the quantity §Surfaces registers as the resolution number.
    96	    """
    97	    w = weights(unit_family)
    98	    units = sorted(aligned)
    99	    diffs = {u: aligned[u][1] - aligned[u][2] for u in units}
   100	    point = float(sum(w[u] * d.mean() for u, d in diffs.items()))
   101	    seeds = np.random.SeedSequence(seed).spawn(len(units))
   102	    rngs = {u: np.random.default_rng(s) for u, s in zip(units, seeds)}
   103	    draws = np.empty(B, dtype=np.float64)
   104	    h = hashlib.sha256(f"B={B};seed={seed};chunk={chunk};units={units}".encode())
   105	    done = 0
   106	    while done < B:
   107	        m = min(chunk, B - done)
   108	        acc = np.zeros(m, dtype=np.float64)
   109	        for u in units:
   110	            n = diffs[u].size
   111	            idx = rngs[u].integers(0, n, size=(m, n), dtype=np.int64)
   112	            # First row of every block only: hashing all B x n indices is 21 GB of sha256 for
   113	            # a plan whose reproducibility the draws digest already establishes. Named for what
   114	            # it is -- a SAMPLE of the plan, not the plan (Codex 2026-09-05).
   115	            h.update(idx[:1].tobytes())
   116	            acc += w[u] * diffs[u][idx].mean(axis=1)
   117	        draws[done:done + m] = acc
   118	        done += m
   119	    lower = float(np.quantile(draws, quantile, method=method))
   120	    # §Screen names BOTH `inverted_cdf` and "the 384th order statistic", and they differ by one
   121	    # observation: `inverted_cdf` at B = 200,000 takes 0-based index 384, i.e. the 385th. The
   122	    # method name is the operative constant and is what decides; the neighbouring order statistic
   123	    # is returned so the size of the discrepancy is on the record (Codex 2026-09-05).
   124	    i0 = int(np.ceil(quantile * B)) - 1
   125	    prev = float(np.partition(draws, i0 - 1)[i0 - 1]) if i0 >= 1 else lower
   126	    return {
   127	        "delta_raw": point,
   128	        "lower_bound_raw": lower,
   129	        "distance_raw": point - lower,
   130	        "lower_bound_prev_order_statistic": prev,
   131	        "order_statistic_index0": i0,
   132	        "quantile": quantile, "quantile_method": method, "B": int(B), "seed": int(seed),
   133	        "chunk": int(chunk),
   134	        "plan_sample_sha256": h.hexdigest(),
   135	        "draws_sha256": hashlib.sha256(np.ascontiguousarray(draws).tobytes()).hexdigest(),
   136	        "draws_sd": float(draws.std(ddof=1)),
   137	        "per_unit_delta_raw": {u: float(d.mean()) for u, d in diffs.items()},
   138	        "n_by_unit": {u: int(d.size) for u, d in diffs.items()},
   139	        "weights": w,
   140	        "_note": "distance_raw = delta_raw - lower_bound_raw; a contrast resolves when "
   141	                 "delta_raw >= MDE and lower_bound_raw > 0 (sign stability checked elsewhere)",
   142	    }
  1280	    freed = release_arena()
  1281	    if verbose and freed:
  1282	        print(f"  released {freed:,} MB of allocator residue back to the OS", flush=True)
  1283	    return PackedIds(np.load(d / "flat.npy", mmap_mode="r"), offs), len(rows)
  1284	
  1285	
  1286	# arms that must see all 12 registered forms: A4 IS the anchor (`ARM_SOURCES`), so both names land
  1287	# on the same requirement.
  1288	REQUIRE_ALL_FORMS = {"A4", "ANCHOR"}
  1289	
  1290	
  1291	def _registry_arms(reg):
  1292	    """The registry's `arms` table, falling back to the real file's when `reg` carries none (the
  1293	    escape a stub test registry already relied on for `n_docs` -- kept, not invented)."""
  1294	    return reg.get("arms") or json.loads((REPO / "m10" / "screen_registry.json").read_text())["arms"]
  1295	
  1296	
  1297	def _registry_anchor(reg):
  1298	    return reg.get("anchor") or json.loads(
  1299	        (REPO / "m10" / "screen_registry.json").read_text()).get("anchor", {})
  1300	
  1301	
  1302	def resolve_arm_name(name, registry=None):
  1303	    """-> the registered, TRAINABLE arm name `name` names, resolving one `anchor_aliases` hop
  1304	    FIRST if needed (so `A4` -> `ANCHOR` before anything checks `A4` itself, which is never
  1305	    trained separately).
  1306	
  1307	    NEVER a source list: a list or tuple is refused here, before anything downstream can treat it
  1308	    as if it carried the guards a real arm name does. `F-winner` (the one alias that is prose, not
  1309	    an arm id -- resolved by `rules.F_selection_aware`, per `screen_registry.json`) is refused by
  1310	    name rather than silently resolving to nonsense. Once resolved, the name must be a registry
  1311	    `arms` entry with `trained: true` and no `cut` field -- `C-M9init` and `F-MiniLM-L12` are
  1312	    registered arms that never ran and are refused here rather than at a training launch.
  1313	    """
  1314	    if not isinstance(name, str):
  1315	        raise SystemExit(f"assemble_arm takes a registered arm name or an anchor_aliases key, "
  1316	                         f"never a source list: got {name!r}")
  1317	    reg = registry or json.loads((REPO / "m10" / "screen_registry.json").read_text())
  1318	    aliases = reg.get("anchor_aliases", {})
  1319	    arms = _registry_arms(reg)
  1320	    if name in aliases and not name.startswith("_"):
  1321	        val = aliases[name]
  1322	        if not isinstance(val, str) or (val not in arms and val not in ARM_SOURCES):
  1323	            raise SystemExit(f"anchor_aliases[{name!r}] = {val!r} does not resolve to a "
  1324	                             f"registered arm id (e.g. 'F-winner' is resolved by "
  1325	                             f"rules.F_selection_aware, not this function) -- pass the resolved "
  1326	                             f"arm name instead")
  1327	        name = val
  1328	    elif name not in arms and name not in ARM_SOURCES:
  1329	        raise SystemExit(f"{name!r} is not a registered arm name (m10/screen_registry.json "
  1330	                         f"`arms`) or an `anchor_aliases` key")
  1331	    entry = arms.get(name)
  1332	    if entry is not None:
  1333	        if entry.get("cut"):
  1334	            raise SystemExit(f"arm {name!r} is CUT (registry `cut`) and cannot be assembled")
  1335	        if not entry.get("trained"):
  1336	            raise SystemExit(f"arm {name!r} is not a trained arm (registry `trained` is not "
  1337	                             f"true) and has no corpus to assemble")
  1338	    return name
  1339	
  1340	
  1341	DOC_SHARE = {"100/0": 0.0, "75/25": 0.25, "50/50": 0.5}
  1342	DEFAULT_MIX = "75/25"          # the anchor's own mix, when a registry carries no `anchor.mix`
  1343	
  1344	
  1345	def arm_doc_count(arm, pattern, batch_size=32):
  1346	    """Documents an arm draws from the screened M9 pool: its document-EXAMPLE count (dose x the
  1347	    pattern's document share), so no document is presented twice at screen dose. M9 drew from
  1348	    every eligible pool row (`m9src/longrun.py`: `n_eligible_doc_rows`); a fixed small draw would
  1349	    repeat each document many times and is not what "the M9 pool" means. At least one batch."""
  1350	    dose = int(arm["dose_examples"])
  1351	    return max(int(math.ceil(dose * DOC_SHARE[pattern])), batch_size)
  1352	
  1353	
  1354	def resolve_arm_pattern(name, entry, reg):
  1355	    """-> (pattern, report). The document/query mix window an arm uses, read from the registry
  1356	    rather than accepted from a caller (item A).
  1357	
  1358	    Family B carries its OWN mix, but encoded in its own arm entry's `pattern` field as a
  1359	    'kQ[+mD]' shorthand (e.g. `B-50/50`'s `"pattern": "2Q+2D"`), not the `DOC_SHARE`/`WINDOWS` key
  1360	    string ('50/50') a stream actually wants -- translated here by COUNTING the Q/D steps rather
  1361	    than hand-copied, so a typo in either registry field cannot silently disagree. Every other
  1362	    trained arm carries no `pattern` of its own and uses the anchor's (`anchor.mix`,
  1363	    cross-checked against `anchor.window_pattern` for the record).
  1364	    """
  1365	    raw = (entry or {}).get("pattern")
  1366	    if raw is not None:
  1367	        m = re.match(r"^(\d+)Q(?:\+(\d+)D)?$", str(raw))
  1368	        if not m:
  1369	            raise SystemExit(f"arm {name!r}: pattern {raw!r} is not the registered 'kQ[+mD]' "
  1370	                             f"shorthand")
  1371	        q, d = int(m.group(1)), int(m.group(2) or 0)
  1372	        pct_q = round(100 * q / max(q + d, 1))
  1373	        resolved = f"{pct_q}/{100 - pct_q}"
  1374	        if resolved not in DOC_SHARE:
  1375	            raise SystemExit(f"arm {name!r}: pattern {raw!r} resolves to {resolved!r}, not one "
  1376	                             f"of {sorted(DOC_SHARE)}")
  1377	        return resolved, {"raw": raw, "resolved": resolved, "source": "arm.pattern"}
  1378	    anchor = _registry_anchor(reg)
  1379	    mix = anchor.get("mix", DEFAULT_MIX)
  1380	    if mix not in DOC_SHARE:
  1381	        raise SystemExit(f"registry anchor.mix {mix!r} is not one of {sorted(DOC_SHARE)}")
  1382	    return mix, {"raw": None, "resolved": mix,
  1383	                "source": "anchor.mix" if "mix" in anchor else "default (no anchor.mix)",
  1384	                "anchor_window_pattern": anchor.get("window_pattern")}
  1385	

exec
/bin/bash -lc "nl -ba m10src/corpus_loader.py | sed -n '118,158p'
nl -ba m10src/corpus_loader.py | sed -n '749,837p'
nl -ba m10src/corpus_loader.py | sed -n '1386,1508p'
nl -ba m10src/trainer10.py | sed -n '88,212p'
nl -ba m10src/run_arm.py | sed -n '899,935p'" in /home/dylan/asymetric-dual-encoders
 succeeded in 0ms:
   118	
   119	# --- AMBIGUITY (reported, not decided): neither the mandate nor the registry assigns a form to a
   120	# PAQ row or to an M9-pool row, and the balanced sampler needs one per row. The reading here
   121	# changes nothing registered: PAQ is registered as "factoid volume" (`instructions-m10.md`:360),
   122	# and the M9 pool's own source labels map onto two of the twelve forms -- `esci-us` is a shopper's
   123	# product search, the other five are questions. Both tables are constants so a ruling is a
   124	# one-line change.
   125	PAQ_FORM = "factoid"
   126	M9_SOURCE_FORM = {"esci-us": "product", "hotpotqa-train": "factoid", "squad-train": "factoid",
   127	                  "mrtydi-en": "factoid", "nqopen": "factoid", "triviaqa": "factoid"}
   128	
   129	SOURCES = {
   130	    "m9-pool":    {"kind": "m9", "n_expected": 463_314,
   131	                   "what": "M9 real queries: queries_pair (esci/hotpotqa/squad/mrtydi, fever out) "
   132	                           "+ nqopen + triviaqa. Targets already exist."},
   133	    "paq-build":  {"kind": "jsonl", "path": WORK / "m10paq" / "paq_build.jsonl",
   134	                   "form": PAQ_FORM, "n_expected": 1_000_000,
   135	                   "what": "the build's PAQ sample, nested inside the A2 sample"},
   136	    "paq-a2":     {"kind": "jsonl", "path": WORK / "m10paq" / "paq_a2.jsonl",
   137	                   "form": PAQ_FORM, "n_expected": 4_037_000,
   138	                   "what": "the A2 volume-control PAQ sample"},
   139	    "harvest":    {"kind": "jsonl", "path": WORK / "m10harvest" / "harvest_train.jsonl",
   140	                   "n_expected": 1_248_386, "require_id": "doc",
   141	                   "what": "A3's harvested real text, FORMS-12 hold-out already removed"},
   142	    "generated":  {"kind": "jsonl", "path": WORK / "m10gen" / "generated_queries.jsonl",
   143	                   "optional": True, "require_id": "seed_id",
   144	                   "what": "the seven generated forms; does not exist yet, same row shape"},
   145	}
   146	
   147	# Which sources an arm's corpus is. A2 takes the 4.037M volume-control sample and A3/A4 the 1.0M
   148	# build sample nested inside it -- `m10/LEDGER.md` §1: "A2 is the volume control, so 'the build
   149	# with less PAQ volume' is the coherent nesting".
   150	ARM_SOURCES = {
   151	    "A1":     ("m9-pool",),
   152	    "A2":     ("m9-pool", "paq-a2"),
   153	    "A3":     ("m9-pool", "paq-build", "harvest"),
   154	    "A4":     ("m9-pool", "paq-build", "harvest", "generated"),
   155	    "ANCHOR": ("m9-pool", "paq-build", "harvest", "generated"),      # A4 IS the anchor arm
   156	}
   157	
   158	
   749	def _form_draw(rows, lengths, batch_size, seed, form, occurrence):
   750	    """One batch of ONE form, drawn WITH REPLACEMENT, then sorted by length for the padded chunk.
   751	
   752	    §Data (`instructions-m10.md`:478-485): "each form's presentation share equal, texts drawn with
   753	    replacement within a form". Cycling fixed length-bucketed batches is not that -- a form of
   754	    three rows at batch 8 yielded the identical `0,1,2,0,1,2,0,1` every time it came up, and a
   755	    large form cycled a fixed partition (Codex 2026-09-05 finding 7). The RNG is a pure function of
   756	    (seed, form, occurrence), so the draw is still addressable by step and resume is exact.
   757	    """
   758	    rng = np.random.default_rng([int(seed), int(form) + 1, int(occurrence)])
   759	    drawn = rng.choice(rows, size=int(batch_size), replace=True)
   760	    return drawn[np.argsort(lengths[drawn], kind="stable")]
   761	
   762	
   763	class FormBalancedStream:
   764	    """Query batches with an equal presentation share per form, addressable by step.
   765	
   766	    `batch(k)` is a pure function of `k`: cycle `c = k // F` draws a fresh permutation of the F
   767	    forms from `default_rng([seed, c])`, and within a cycle each form contributes exactly one
   768	    batch -- so shares are exactly equal over every full cycle and the resume property holds.
   769	    Each batch is one form, which is also what keeps length bucketing worth having.
   770	
   771	    `balanced=False` samples by example over the whole corpus (`data10.length_buckets`): the A2
   772	    volume-control arm's variant and the reported diagnostic.
   773	    """
   774	
   775	    def __init__(self, ids, targets, forms, pad_id, batch_size=32, seed=0, balanced=True,
   776	                require_forms=None):
   777	        self.ids, self.T, self.pad, self.bs, self.seed = ids, targets, pad_id, batch_size, seed
   778	        self.forms = np.asarray(forms, dtype=np.int16)
   779	        self.balanced = balanced
   780	        lengths = getattr(ids, "lengths", None)
   781	        if lengths is None:
   782	            lengths = np.array([len(ids[i]) for i in range(len(ids))], dtype=np.int64)
   783	        self.lengths = np.asarray(lengths, dtype=np.int64)
   784	        if balanced:
   785	            self.present = [int(f) for f in np.unique(self.forms)]
   786	            self.rows = {f: np.flatnonzero(self.forms == f) for f in self.present}
   787	            # `require_forms` is the anchor/A4 contract: EVERY registered form must have rows, not
   788	            # just every form already present. The old check (`for f in self.present`) could never
   789	            # fire -- `self.present` is built FROM the forms that survived `np.unique`, so a form
   790	            # missing entirely was silently absent from both the check and the sampler (Codex
   791	            # 2026-09-05 whole-plan review).
   792	            if require_forms:
   793	                have = {FORMS[f] for f in self.present}
   794	                missing = [f for f in require_forms if f not in have]
   795	                if missing:
   796	                    raise ValueError(f"forms with no rows: {missing}")
   797	            self.n_batches = sum(int(np.ceil(len(v) / batch_size)) for v in self.rows.values())
   798	        else:
   799	            self.present = [-1]
   800	            self.batches = {-1: D.length_buckets(ids, batch_size, seed=seed)}
   801	            if not self.batches[-1]:
   802	                raise ValueError("no full batches: corpus smaller than one batch")
   803	            self.n_batches = len(self.batches[-1])
   804	
   805	    def __len__(self):
   806	        """The batches one full presentation of the corpus costs. With replacement the stream is
   807	        endless, so this is a scale, not a bound -- `batch(k)` accepts any k."""
   808	        return self.n_batches
   809	
   810	    def _pick(self, k):
   811	        F = len(self.present)
   812	        c, j = divmod(int(k), F)
   813	        if not self.balanced:
   814	            bl = self.batches[-1]
   815	            return -1, bl[c % len(bl)]
   816	        f = self.present[int(np.random.default_rng([self.seed, c]).permutation(F)[j])] \
   817	            if F > 1 else self.present[0]
   818	        return f, _form_draw(self.rows[f], self.lengths, self.bs, self.seed, f, c)
   819	
   820	    def batch(self, k):
   821	        _f, idx = self._pick(k)
   822	        x, m = D.collate(self.ids, idx, self.pad)
   823	        return x, m, torch.from_numpy(np.ascontiguousarray(self.T[idx]))
   824	
   825	    def realized_shares(self, n_batches):
   826	        """-> {form: share of presented EXAMPLES} over the first `n_batches` steps, which is what
   827	        §0b records. Counted from the rows actually drawn, so it is a measurement of the sampler
   828	        and not a restatement of its design -- and so the unbalanced variant reports the per-form
   829	        shares it produces rather than a single bucket.
   830	        """
   831	        idx = np.concatenate([self._pick(k)[1] for k in range(int(n_batches))])
   832	        cnt = np.bincount(self.forms[idx], minlength=len(FORMS))
   833	        tot = max(int(cnt.sum()), 1)
   834	        return {FORMS[i]: round(float(c) / tot, 6) for i, c in enumerate(cnt) if c}
   835	
   836	
   837	# --------------------------------------------------------------------------------- data cut ----
  1386	
  1387	def arm_batch(entry, reg):
  1388	    """The arm's registered batch (`arms.<name>.batch`, else `anchor.batch`). E-bs128 is the one
  1389	    arm that differs (Codex pass 6, finding 2)."""
  1390	    b = entry.get("batch")
  1391	    return int(b if b is not None else _registry_anchor(reg).get("batch", 32))
  1392	
  1393	
  1394	def assemble_arm(arm_name, tok, student, *, batch_size=None, seed=0, max_len=512, verbose=True,
  1395	                 registry=None):
  1396	    """-> (batch_fn, manifest). The ONLY function a training launcher may use to build an arm's
  1397	    corpus (Codex 2026-09-05 whole-plan review: the cut, the masks, the 12-form requirement and
  1398	    the cross-role guard were each correct on their own but each OPTIONAL -- a source list or a
  1399	    flag bypassed every one of them; the fix is one mandatory path).
  1400	
  1401	    Every data-affecting knob comes from the REGISTRY, never the caller (item A): the corpus
  1402	    (`arm_sources`, via the arm's `data`), the mix pattern (`resolve_arm_pattern`, via the arm's
  1403	    own `pattern` or `anchor.mix`) and the document count (`arm_doc_count`, via `dose_examples`).
  1404	    `balanced` is always `True` -- the mandate's unbalanced variant is `build_query_stream`'s own
  1405	    diagnostic, never an arm.
  1406	
  1407	    Takes a registered arm name or an `anchor_aliases` key, never a source list
  1408	    (`resolve_arm_name`, which also refuses an arm that is not `trained` or is `cut`). Applies the
  1409	    registered cut for a cut arm, raising if `data_cut.unique_text_count` is not registered yet
  1410	    (no `allow_uncut` here -- that escape belongs to `build_query_stream` and a smoke alone).
  1411	    Requires the M10 re-screen masks for BOTH pools (no `allow_unscreened`) and cross-checks them
  1412	    against `results/m10_rescreen10.json` via `rescreen10.validate`. Requires the anchor/A4 arm to
  1413	    see all 12 forms, raising and naming any missing -- unconditionally, since `balanced` can no
  1414	    longer be turned off here. Runs `guard_cross_role` on the two streams before returning.
  1415	    Everything applied is recorded in the manifest.
  1416	    """
  1417	    reg = registry or json.loads((REPO / "m10" / "screen_registry.json").read_text())
  1418	    name = resolve_arm_name(arm_name, reg)
  1419	    entry = _registry_arms(reg).get(name) or {}
  1420	    require_forms = FORMS if name in REQUIRE_ALL_FORMS else None
  1421	    reg_batch = arm_batch(entry, reg)
  1422	    if batch_size is not None and int(batch_size) != reg_batch:
  1423	        raise SystemExit(f"{name}: caller batch_size {batch_size} != the registered batch "
  1424	                         f"{reg_batch}; the registry owns it")
  1425	    batch_size = reg_batch
  1426	    pattern, pattern_rep = resolve_arm_pattern(name, entry, reg)
  1427	    n_docs = arm_doc_count(entry, pattern, batch_size)
  1428	    import rescreen10
  1429	    # `consumed` collects the EXACT mask arrays the two streams are built from; validation runs
  1430	    # on those, never on a second load (Codex pass 5: a cache swap between the load that built
  1431	    # the stream and a reload for validation could admit evaluation text unnoticed).
  1432	    consumed = {"protected10": rescreen10.protected_ident()}
  1433	    q_stream, q_man = build_query_stream(name, tok, student, batch_size=batch_size, seed=seed,
  1434	                                        balanced=True, max_len=max_len, prefix="",
  1435	                                        allow_uncut=False, require_forms=require_forms,
  1436	                                        verbose=verbose, registry=reg, consumed=consumed)
  1437	    # The doc build peaks at ~7.4 GB (measured) and runs SECOND. Returning the query build's
  1438	    # allocator residue first is cheap insurance, NOT a proven peak reduction: if the doc build's
  1439	    # allocations can reuse those freed chunks, the pages go live again without raising RSS and the
  1440	    # trim only forces them to be re-faulted (Codex 2026-09-07). What IS measured is that the trim
  1441	    # lowers the steady-state baseline handed to the model and optimizer. The A/B that would settle
  1442	    # the ordering -- identical assemble_arm, same cache state, this call enabled vs stubbed,
  1443	    # comparing document VmHWM and the first optimizer-step peak -- has NOT been run.
  1444	    release_arena()
  1445	    doc_stream, doc_man = build_doc_stream(n_docs, tok, batch_size=batch_size, seed=seed,
  1446	                                          max_len=max_len, allow_unscreened=False, verbose=verbose,
  1447	                                          consumed=consumed)
  1448	    cross = guard_cross_role(q_stream.ids, doc_stream.ids)
  1449	    segs = tuple(getattr(rescreen10, "QUERY_SEGMENTS", ()))
  1450	    missing = [k for k in (*segs, "documents") if k not in consumed]
  1451	    if missing:
  1452	        raise SystemExit(f"assemble_arm: the streams did not record the masks they consumed for "
  1453	                         f"{missing} -- refusing to validate a reload instead")
  1454	    rescreen10.validate(rescreen10.load_report(), consumed)
  1455	
  1456	    man = {"arm": name, "requested_as": arm_name, "student": student, "batch_size": batch_size,
  1457	          "seed": seed, "max_len": max_len, "pattern": pattern, "pattern_source": pattern_rep,
  1458	          "n_docs": n_docs, "require_forms": list(require_forms) if require_forms else None,
  1459	          "query": q_man, "document": doc_man, "cross_role": cross,
  1460	          "rescreen10_report_validated": True}
  1461	    return D.batch_fn(q_stream, doc_stream, pattern=pattern), man
  1462	
  1463	
  1464	class LengthsOnly:
  1465	    """Just enough of a pretokenized corpus to bucket and count with: the lengths. Lets the
  1466	    manifest report realized shares before a single teacher vector exists."""
  1467	
  1468	    def __init__(self, lengths):
  1469	        self.lengths = np.asarray(lengths, dtype=np.int64)
  1470	
  1471	    def __len__(self):
  1472	        return len(self.lengths)
  1473	
  1474	
  1475	def main():
  1476	    """Report the manifest and the realized form shares. Needs no teacher targets and no GPU."""
  1477	    import argparse
  1478	    ap = argparse.ArgumentParser()
  1479	    ap.add_argument("--sources", nargs="+", default=["harvest"])
  1480	    ap.add_argument("--batch-size", type=int, default=32)
  1481	    ap.add_argument("--shares-over", type=int, default=100_000, help="batches to realize shares on")
  1482	    ap.add_argument("--unbalanced", action="store_true")
  1483	    ap.add_argument("--out", default=str(REPO / "results" / "m10_corpus_manifest.json"))
  1484	    a = ap.parse_args()
  1485	    man, forms, words = {"sources": [], "sources_used": a.sources}, [], []
  1486	    for name in a.sources:
  1487	        texts, f, _ids, sman = source_texts(name)
  1488	        man["sources"].append(sman)
  1489	        forms += [FORM_ID[x] for x in f]
  1490	        words.append(np.fromiter((len(t.split()) for t in texts), dtype=np.int64, count=len(texts)))
  1491	        print(f"  {name}: {len(texts):,} rows {sman['by_form']}", flush=True)
  1492	    forms = np.asarray(forms, dtype=np.int16)
  1493	    words = np.concatenate(words)
  1494	    man["n_rows"] = int(len(forms))
  1495	    man["by_form"] = {FORMS[int(f)]: int(c) for f, c in zip(*np.unique(forms, return_counts=True))}
  1496	    st = FormBalancedStream(LengthsOnly(words), None, forms, pad_id=0,
  1497	                            batch_size=a.batch_size, seed=0, balanced=not a.unbalanced)
  1498	    man["balanced"] = not a.unbalanced
  1499	    man["realized_shares"] = st.realized_shares(a.shares_over)
  1500	    man["shares_over_batches"] = a.shares_over
  1501	    man["word_len"] = {"mean": round(float(words.mean()), 2), "p50": int(np.percentile(words, 50)),
  1502	                       "p95": int(np.percentile(words, 95))}
  1503	    Path(a.out).write_text(json.dumps(man, indent=1, default=str))
  1504	    print(json.dumps({k: v for k, v in man.items() if k != "sources"}, indent=1, default=str))
  1505	
  1506	
  1507	if __name__ == "__main__":
  1508	    main()
    88	def train_arm(model, batch_fn, total_steps, *, pattern="75/25", cycles=3, peak=1e-4, final=1e-5,
    89	              loss_name="squared_l2", sigma=None, eval_fn=None, evals_per_cycle=2,
    90	              ckpt_path=None, ckpt_every=0, resume_from=None, seed=0, log_every=0,
    91	              device="cpu", batch_size=32, read_steps=(), cycle_ckpt_fmt=None,
    92	              eval_state=None, fingerprint=None, wd=0.01):
    93	    """Run one arm. -> a record: losses, evaluations, rates, and why it stopped.
    94	
    95	    `eval_fn(model, step, kind)` returns the arm's COV macro at a scheduled evaluation; `kind` is
    96	    'end' at a cycle end and 'mid' otherwise, which is the distinction §Kill compares within.
    97	
    98	    `read_steps` are EXTRA reading points (family F is read at 5M/10M/20M inside a 20M schedule,
    99	    and 5M is neither a cycle end nor a midpoint). They get `kind='read'` and are recorded in
   100	    `read_evals`, **outside** the kill/plateau state: §Kill compares "midpoint against midpoints,
   101	    cycle end against cycle ends", so a third kind entering that sequence would change a
   102	    registered rule rather than implement it.
   103	
   104	    `cycle_ckpt_fmt` is a path template (`.../cycle{cycle}.pt`) saved at every cycle end and
   105	    RETAINED — the rolling `ckpt_path` is a resume point that gets overwritten, while the final
   106	    cycle end is the arm's final checkpoint and its sha256 goes in the arm's record. It is written
   107	    before the kill/plateau test, so an arm that stops AT a cycle end still has that checkpoint.
   108	
   109	    `eval_state` is the CALLER'S evaluator (`run_arm.CovEval`), whose `.state()`/`.restore()` go
   110	    into the checkpoint beside the loop's own state: its per-family macros and per-query score
   111	    paths are part of the run and a resume that loses them loses the contrast step's inputs
   112	    (Codex runner review 2026-09-07, finding 7).
   113	
   114	    `fingerprint` binds a checkpoint to the recipe that wrote it. A resume whose fingerprint
   115	    differs is REFUSED rather than continuing one arm's schedule into another arm's weights
   116	    (finding 8). Checkpoints written without one never satisfy a fingerprinted resume.
   117	    """
   118	    loss_fn = N.LOSSES[loss_name]
   119	    opt = torch.optim.AdamW(param_groups(model, wd), lr=peak, betas=(0.9, 0.999), eps=1e-8)
   120	    amp, amp_on = autocast_for(device)
   121	    start = 0
   122	    losses, evals, kinds, cycle_end_evals, read_evals = [], [], [], [], []
   123	    n_examples, stopped = 0, None
   124	    if resume_from:
   125	        # The EVALUATION history is part of the run state. Without it a resumed arm restarts
   126	        # `cycle_end_evals` empty, so the plateau rule reads one cycle where it needs three and
   127	        # the registered kill/plateau decision cannot fire at all (Codex 2026-09-05 finding 4).
   128	        start, ex = load(resume_from, model, opt)
   129	        if fingerprint is not None and ex.get("fingerprint") != fingerprint:
   130	            raise SystemExit(
   131	                f"REFUSED: {resume_from} was written by a different recipe "
   132	                f"(fingerprint {ex.get('fingerprint')!r} vs {fingerprint!r}). A resume continues "
   133	                f"ONE arm; move the checkpoint aside deliberately.")
   134	        if eval_state is not None and ex.get("eval_state") is not None:
   135	            eval_state.restore(ex["eval_state"])
   136	        losses = list(ex.get("losses", []))
   137	        evals = list(ex.get("evals", []))
   138	        kinds = list(ex.get("eval_kinds", []))
   139	        cycle_end_evals = list(ex.get("cycle_end_evals", []))
   140	        read_evals = list(ex.get("read_evals", []))
   141	        n_examples = int(ex.get("examples", 0))
   142	        stopped = ex.get("stopped")
   143	    else:
   144	        torch.manual_seed(seed)
   145	
   146	    model.train()               # the warm start's `pooled_features` left it in eval() (finding 1)
   147	
   148	    def extra():
   149	        e = {"losses": losses, "evals": evals, "eval_kinds": kinds,
   150	             "cycle_end_evals": cycle_end_evals, "read_evals": read_evals,
   151	             "examples": n_examples, "stopped": stopped, "fingerprint": fingerprint}
   152	        if eval_state is not None:
   153	            e["eval_state"] = eval_state.state()
   154	        return e
   155	
   156	    ends = set(N.cycle_ends(total_steps, cycles))
   157	    reads = {int(s) for s in read_steps} - ends
   158	    per = max(total_steps // cycles, 1)
   159	    mids = {c * per + per // 2 for c in range(cycles)}
   160	    t0, run_examples, run_steps = time.time(), 0, 0
   161	
   162	    for step in range(start, total_steps if stopped is None else start):
   163	        kind = N.mix_window(pattern, step)
   164	        ids, mask, tgt = batch_fn(step, kind)
   165	        lr = N.lr_at(step, total_steps, cycles, peak, final)
   166	        for g in opt.param_groups:
   167	            g["lr"] = lr
   168	        with amp:                                       # bf16 forward on CUDA, no-op on CPU
   169	            pred = model(ids.to(device), mask.to(device))
   170	        pred = pred.float()                             # the LOSS is fp32 (§Recipe)
   171	        loss = loss_fn(pred, tgt.to(device)) if sigma is None \
   172	            else loss_fn(pred, tgt.to(device), sigma)
   173	        if not torch.isfinite(loss):
   174	            stopped = f"non-finite loss at step {step}"
   175	            losses.append(float("nan"))
   176	            break
   177	        opt.zero_grad(set_to_none=True)
   178	        loss.backward()
   179	        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
   180	        if not torch.isfinite(gn):
   181	            stopped = f"non-finite grad norm at step {step}"
   182	            break
   183	        opt.step()
   184	        losses.append(float(loss.detach()))
   185	        n_examples += len(ids)
   186	        run_examples += len(ids)
   187	        run_steps += 1
   188	
   189	        if eval_fn is not None and (step in ends or step in mids or step in reads):
   190	            k = "end" if step in ends else ("mid" if step in mids else "read")
   191	            m = eval_fn(model, step, k)
   192	            model.train()               # every evaluator encodes in eval(); restore train mode
   193	            if k == "read":
   194	                # a READING point, not a scheduled evaluation: recorded, and deliberately not
   195	                # part of the kill/plateau state (see the docstring).
   196	                read_evals.append({"step": int(step), "macro": m})
   197	            else:
   198	                evals.append(m); kinds.append(k)
   199	                if k == "end":
   200	                    cycle_end_evals.append(m)
   201	                    if cycle_ckpt_fmt:
   202	                        save(str(cycle_ckpt_fmt).format(cycle=len(cycle_end_evals)), model, opt,
   203	                             step + 1, extra=extra())
   204	                fired, why = N.kill_fires(evals, lambda i: kinds[i])
   205	                if fired:
   206	                    stopped = f"kill: {why}"
   207	                    break
   208	                pf, at = N.plateau_fires(cycle_end_evals)
   209	                if pf:
   210	                    stopped = f"plateau at cycle {at}"
   211	                    break
   212	        if ckpt_path and ckpt_every and (step + 1) % ckpt_every == 0:
   899	    # FAILURE to be recorded, not a pre-flight refusal.
   900	    ctx["started"] = True
   901	    ws = warm_start(model, {"warm_start": p["warm_start"]}, q_stream, ws_cfg, verbose=verbose)
   902	
   903	    sigma = None
   904	    if p["objective"] == "document_covariance_weighted":
   905	        sigma = torch.from_numpy(np.asarray(N.cov_matrix(d_stream.T), dtype=np.float32)).to(device)
   906	
   907	    total, ends, reads = p["total_steps"], p["cycle_end_steps"], p["read_points"]
   908	    read_steps = {int(r["step"]): int(r["examples"]) for r in reads}
   909	    extra_reads = sorted(s for s in read_steps if s not in set(ends))
   910	    ev = StubEval() if smoke else CovEval(model, out_dir, verbose=verbose)
   911	    ctx["cov_records"] = ev.records
   912	
   913	    def eval_fn(_m, step, kind):
   914	        if kind == "end":
   915	            label = f"cycle{ends.index(step) + 1}"
   916	        elif kind == "mid":
   917	            label = f"mid{step}"
   918	        else:
   919	            label = f"read{read_steps[step]}"
   920	        if kind != "read" and step in read_steps:
   921	            # a read point that coincides with a cycle end (20M) or a midpoint (10M): one eval,
   922	            # labelled as both, so the record's `reads` list finds it.
   923	            label = f"{label}_read{read_steps[step]}"
   924	        return ev(step, label)
   925	
   926	    ck = out_dir / "ckpt.pt"
   927	    every = int(ckpt_every or max(total // 20, 1))
   928	    train_model = torch.compile(model) if compile_step else model
   929	    fp = fingerprint(arm, p, man, seed, smoke_steps, device)
   930	    r = Tr.train_arm(train_model, batch_fn, total_steps=total, pattern=p["pattern"],
   931	                     cycles=CYCLES, peak=PEAK, final=FINAL, loss_name=p["objective"],
   932	                     sigma=sigma, eval_fn=eval_fn, ckpt_path=ck, ckpt_every=every,
   933	                     resume_from=(str(ck) if resume else None), seed=seed,
   934	                     log_every=max(total // 50, 1), device=device, batch_size=p["batch"],
   935	                     read_steps=extra_reads, cycle_ckpt_fmt=str(out_dir / "cycle{cycle}.pt"),

codex
The data sampler makes the synthetic-data question materially different from its wording. A3 has five forms; A4 balances twelve, seven generated. Thus A4 assigns about 58% of query presentations to generated text and reduces each existing form’s share from 20% to 8.3%. “What generation adds” is actually a large reallocation of training exposure, even though the arms have equal unique-text counts.
codex
Two further checks matter before I price changes. The head argument treats normalized embedding regression as though it had PCA’s unnormalized reconstruction optimum; that equivalence does not hold. Also, the plan’s own calibration reports a same-initialization interval distance of 0.00288, far below the unrelated-model distance of 0.008619. I’m checking that calibration artifact so the review doesn’t repeat the claim that every MDE-sized contrast is inherently unresolvable.
exec
/bin/bash -lc "nl -ba m10src/nano10.py | sed -n '1,250p'
nl -ba results/m10_calib_report.json" in /home/dylan/asymetric-dual-encoders
 succeeded in 0ms:
     1	"""The M10 student: a PER-TOKEN head over concatenated layers, pooled AFTER the head.
     2	
     3	M9's student was `mean_t(h_t) -> Linear`, one layer wide. M10's diagnosis (`m9/FINDINGS.md`,
     4	`m10/PLANNING.md` §9) is that a 384-wide linear head cannot be pushed past ~90-93% retention once
     5	queries are diverse, so the head widens to three concatenated layers (1152-d; four = 1536-d,
     6	family G) and moves BEFORE the pool:
     7	
     8	    training      normalize(mean_t(head(x_t)))
     9	    serving       the ONNX graph emits head(x_t) per token; fastembed does the masked mean and
    10	                  the normalize itself
    11	
    12	Those are the same function, and for a LINEAR head that is not an approximation -- pooling and the
    13	head commute, so `head(mean_t x_t) == mean_t(head(x_t))` exactly. Two consequences the code below
    14	depends on, both asserted by `m10src/test_nano10.py` rather than believed:
    15	
    16	1. **The ridge warm start can be solved on POOLED features.** Fitting per-token would be 1152-d
    17	   features x every token; fitting on the pooled 1152-d vector gives the identical head. That is
    18	   why the M9 warm-start machinery ports unchanged to a per-token architecture.
    19	2. **The serving parity is exact.** `m10src/student_parity.py` measures it end to end through
    20	   fastembed (min-cos >= 0.99999988 on all six family-F heads) -- this module only has to not
    21	   break the algebra.
    22	
    23	G-MLP (family G) adds a per-token rank-192 GELU correction to the same linear path. It stays
    24	exportable for the same reason -- it is applied per token, before the pool -- but it does NOT
    25	commute with pooling, so its warm start is the three-solve recipe registered in
    26	`m10/screen_registry.json` `warm_start.G-MLP`, not a pooled ridge.
    27	
    28	`m9src/nano.py` is a `guard9` "train"-scope file and is not touched.
    29	"""
    30	import json
    31	import sys
    32	from pathlib import Path
    33	
    34	REPO = Path(__file__).resolve().parents[1]
    35	for p in ("m7src", "m9src", "m10src"):
    36	    sys.path.insert(0, str(REPO / p))
    37	
    38	import numpy as np
    39	import torch
    40	import torch.nn as nn
    41	import torch.nn.functional as F
    42	
    43	# Which hidden states the head reads, top-down (§Recipe): the last layer, then two-thirds, then
    44	# one-third of depth; family G's fourth arm adds one more. Hard-coded rather than derived, because
    45	# these exact indices are what the parity artifacts were measured on.
    46	#
    47	# `1` is family G's **G-384** arm -- "feature = last layer only (384, M9's head)"
    48	# (`instructions-m10.md`:616) -- and the key was MISSING, so `Nano10(..., n_layers=1)` raised
    49	# `KeyError: 1` and a registered arm of the locked design could not be built at all. Found by the
    50	# registered 90-step smoke of every arm shape, which is exactly what that requirement is for.
    51	# Family G runs on family F's winner, so every student needs the key.
    52	LAYERS = {
    53	    "bge-small":  {1: (12,), 3: (12, 8, 4), 4: (12, 8, 4, 2)},
    54	    "MiniLM-L6":  {1: (6,),  3: (6, 4, 2),  4: (6, 4, 2, 1)},
    55	    "MiniLM-L12": {1: (12,), 3: (12, 8, 4), 4: (12, 8, 4, 2)},
    56	}
    57	REPOS = {"bge-small": "BAAI/bge-small-en-v1.5",
    58	         "MiniLM-L6": "sentence-transformers/all-MiniLM-L6-v2",
    59	         "MiniLM-L12": "sentence-transformers/all-MiniLM-L12-v2"}
    60	OUT_DIM = 1024
    61	CAP = 35_000_000
    62	
    63	
    64	class MLPHead(nn.Module):
    65	    """G-MLP: the anchor's full-rank linear path plus a rank-k GELU correction, per token."""
    66	
    67	    def __init__(self, d_in, k, d_out):
    68	        super().__init__()
    69	        self.lin = nn.Linear(d_in, d_out)
    70	        self.down = nn.Linear(d_in, k)
    71	        self.up = nn.Linear(k, d_out)
    72	
    73	    def forward(self, x):
    74	        return self.lin(x) + self.up(F.gelu(self.down(x)))
    75	
    76	
    77	class Nano10(nn.Module):
    78	    """Backbone -> concat chosen hidden states -> head PER TOKEN -> masked mean -> (normalize)."""
    79	
    80	    def __init__(self, student_key, n_layers=3, head="linear", mlp_k=192, out_dim=OUT_DIM,
    81	                 max_seq=512):
    82	        super().__init__()
    83	        from transformers import AutoModel, AutoTokenizer
    84	        repo = REPOS[student_key]
    85	        self.key, self.n_layers, self.head_kind = student_key, n_layers, head
    86	        self.layers = LAYERS[student_key][n_layers]
    87	        self.tok = AutoTokenizer.from_pretrained(repo)
    88	        self.backbone = AutoModel.from_pretrained(repo)
    89	        d_in = self.backbone.config.hidden_size * n_layers
    90	        self.d_in, self.out_dim, self.max_seq = d_in, out_dim, max_seq
    91	        self.head = nn.Linear(d_in, out_dim) if head == "linear" \
    92	            else MLPHead(d_in, mlp_k, out_dim)
    93	        if max(self.layers) > self.backbone.config.num_hidden_layers:
    94	            raise ValueError(f"{student_key}: layer {max(self.layers)} of "
    95	                             f"{self.backbone.config.num_hidden_layers}")
    96	
    97	    def features(self, input_ids, attention_mask):
    98	        """-> (b, s, d_in): the concatenated hidden states, before the head."""
    99	        hs = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
   100	                           output_hidden_states=True).hidden_states
   101	        return torch.cat([hs[l] for l in self.layers], dim=-1)
   102	
   103	    def token_out(self, input_ids, attention_mask):
   104	        """-> (b, s, out_dim): what the exported ONNX graph emits. Serving starts here."""
   105	        return self.head(self.features(input_ids, attention_mask))
   106	
   107	    def forward(self, input_ids, attention_mask, normalize=True):
   108	        t = self.token_out(input_ids, attention_mask)
   109	        m = attention_mask.unsqueeze(-1).to(t.dtype)
   110	        v = (t * m).sum(1) / m.sum(1).clamp(min=1e-9)
   111	        return F.normalize(v, dim=-1, eps=1e-12) if normalize else v
   112	
   113	    def n_params(self):
   114	        return sum(p.numel() for p in self.parameters())
   115	
   116	    def under_cap(self):
   117	        return self.n_params() <= CAP
   118	
   119	    @torch.inference_mode()
   120	    def encode_queries(self, texts, batch_size=256, prefix=""):
   121	        """-> (n, out_dim) fp32 unit-norm, in input order. The serving path."""
   122	        self.eval()
   123	        dev = next(self.parameters()).device
   124	        out = np.empty((len(texts), self.out_dim), dtype=np.float32)
   125	        order = np.argsort([len(t) for t in texts], kind="stable")
   126	        for i in range(0, len(order), batch_size):
   127	            sel = order[i:i + batch_size]
   128	            b = self.tok([prefix + texts[j] for j in sel], padding=True, truncation=True,
   129	                         max_length=self.max_seq, return_tensors="pt").to(dev)
   130	            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
   131	                v = self(b["input_ids"], b["attention_mask"])
   132	            out[sel] = v.float().cpu().numpy()
   133	        return out
   134	
   135	
   136	@torch.inference_mode()
   137	def pooled_features(model, texts, prefix="", batch_size=128):
   138	    """-> (n, d_in) the MEAN-POOLED concatenated features, the warm start's design matrix.
   139	
   140	    Pooling before a LINEAR head is the same map as pooling after it, so the ridge solve on these
   141	    is exactly the per-token solve — at a fraction of the memory. `test_nano10` proves the
   142	    equality rather than asserting it.
   143	    """
   144	    model.eval()
   145	    dev = next(model.parameters()).device
   146	    out = np.empty((len(texts), model.d_in), dtype=np.float32)
   147	    order = np.argsort([len(t) for t in texts], kind="stable")
   148	    for i in range(0, len(order), batch_size):
   149	        sel = order[i:i + batch_size]
   150	        b = model.tok([prefix + texts[j] for j in sel], padding=True, truncation=True,
   151	                      max_length=model.max_seq, return_tensors="pt").to(dev)
   152	        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
   153	            f = model.features(b["input_ids"], b["attention_mask"])
   154	        m = b["attention_mask"].unsqueeze(-1).to(f.dtype)
   155	        out[sel] = ((f * m).sum(1) / m.sum(1).clamp(min=1e-9)).float().cpu().numpy()
   156	    return out
   157	
   158	
   159	def ridge_head(X, Y, lam):
   160	    """The M9 closed form, unchanged: solve on [X | 1], then row-normalize the prediction."""
   161	    Xc = np.hstack([X, np.ones((X.shape[0], 1), dtype=np.float32)])
   162	    G = Xc.T @ Xc
   163	    scale = float(np.trace(G) / G.shape[0])
   164	    A = np.linalg.solve(G + lam * scale * np.eye(G.shape[0], dtype=np.float32), Xc.T @ Y)
   165	    P = Xc @ A
   166	    P = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-12)
   167	    return A, float(np.mean(np.sum((P - Y) ** 2, axis=1)))
   168	
   169	
   170	def warm_start_linear(model, X, Y, lam):
   171	    """Install the ridge solution into a LINEAR per-token head. Identical treatment every arm."""
   172	    if model.head_kind != "linear":
   173	        raise ValueError("G-MLP has its own three-solve warm start (screen_registry warm_start)")
   174	    A, resid = ridge_head(X, np.asarray(Y, dtype=np.float32), lam)
   175	    dev = next(model.parameters()).device
   176	    with torch.no_grad():
   177	        model.head.weight.copy_(torch.from_numpy(np.ascontiguousarray(A[:-1].T)).to(dev))
   178	        model.head.bias.copy_(torch.from_numpy(np.ascontiguousarray(A[-1])).to(dev))
   179	    return {"n_fit": int(X.shape[0]), "lambda": lam, "train_objective": round(resid, 5),
   180	            "d_in": int(X.shape[1])}
   181	
   182	
   183	# ---- objectives (family D) ------------------------------------------------------------------
   184	
   185	def loss_sq_l2(pred, target):
   186	    """The anchor: squared L2 between unit-norm student and unit-norm teacher."""
   187	    return ((pred - target) ** 2).sum(-1).mean()
   188	
   189	
   190	def loss_norm_e2(pred, target):
   191	    """D-NORM — LEAF's own loss is the NORM of the error, not its square (amendment A1).
   192	
   193	    A one-line arm we had silently diverged from; the square is what M9 trained.
   194	    """
   195	    return (pred - target).norm(dim=-1).mean()
   196	
   197	
   198	def loss_cov_weighted(pred, target, sigma):
   199	    """D-COV — `L = (s - t)^T Σ (s - t)`, the registered form (§Recipe).
   200	
   201	    Σ is the FULL covariance of the pool's frozen stella DOCUMENT vectors, unit-trace normalised
   202	    and shrunk as `(1-α)Σ̂ + αI/1024` with α = 0.1 fixed. Not a per-dimension weight: a diagonal
   203	    would keep the coordinate basis, and the whole claim is that error should be charged in the
   204	    directions documents actually differ along, which are not axis-aligned. Plain L2 is
   205	    reconstruction and spends a rank-limited output equally on every direction of the teacher's
   206	    query space; this spends it where nDCG is decided.
   207	
   208	    Frozen before the arm runs — it is data about the corpus, never a learned parameter.
   209	    """
   210	    d = pred - target
   211	    return ((d @ sigma) * d).sum(-1).mean()
   212	
   213	
   214	def cov_matrix(doc_vecs, alpha=0.10, chunk=100_000):
   215	    """Σ for D-COV: unit-trace normalised, shrunk toward the identity. Computed ONCE from
   216	    document vectors that already exist. α is fixed at 0.10 and is never tuned on a surface.
   217	
   218	    **Accumulated in chunks, never materialized.** `np.asarray(doc_vecs, dtype=np.float64)` on the
   219	    registered 5,000,000-document stream is 1024 x 5e6 x 8 = **38 GiB** — worse than the 19.1 GiB
   220	    fp32 materialisation that took WSL down twice on 2026-09-07, and it would have hit D-COV the
   221	    same way. One pass accumulates `sum(x)` and `x^T x`; the centred Gram is
   222	    `x^T x − n·mean·mean^T` exactly, so nothing is approximated and the peak is one chunk
   223	    (100,000 x 1024 float64 = 0.8 GiB). Accepts an ndarray or any indexable view
   224	    (`corpus_loader.DocTargetView`).
   225	    """
   226	    n = len(doc_vecs)
   227	    d = int(np.asarray(doc_vecs[0:1], dtype=np.float64).shape[1])
   228	    s1 = np.zeros(d, dtype=np.float64)
   229	    S2 = np.zeros((d, d), dtype=np.float64)
   230	    for i in range(0, n, chunk):
   231	        blk = np.asarray(doc_vecs[np.arange(i, min(i + chunk, n))], dtype=np.float64)
   232	        s1 += blk.sum(0)
   233	        S2 += blk.T @ blk
   234	    mean = s1 / max(n, 1)
   235	    S = (S2 - n * np.outer(mean, mean)) / max(n - 1, 1)
   236	    S = S / np.trace(S)                              # unit trace, as registered
   237	    # `(1-α)Σ̂ + αI/d`: the shrunk matrix also has unit trace, so α is a pure mixing weight and
   238	    # cannot rescale the loss. That invariance is what makes D-COV comparable to the anchor at
   239	    # the same learning rate, and it is asserted in `test_nano10`.
   240	    return (1 - alpha) * S + alpha * np.eye(d) / d
   241	
   242	
   243	LOSSES = {"squared_l2": loss_sq_l2, "leaf_norm_e2": loss_norm_e2,
   244	          "document_covariance_weighted": loss_cov_weighted}
   245	
   246	
   247	# ---- the kill and plateau rules (§Kill) -------------------------------------------------------
   248	
   249	KILL_DROP, PLATEAU_MIN_GAIN, PLATEAU_FROM_CYCLE = 0.0056, 0.003, 3
   250	
     1	{
     2	 "_what": "M10.0-e same-init calibration. Brackets B, D, G and C ONLY.",
     3	 "_f_and_e": "read against the unrelated-models distance 0.008619, not this",
     4	 "_changes": "no constant; MDE 0.0056 and alpha 0.025/13 untouched",
     5	 "macro": {
     6	  "P0": 0.47752795346253507,
     7	  "P1": 0.47614103191078433,
     8	  "P2": 0.4738919632001084
     9	 },
    10	 "seed_effect_point_estimate": 0.0013869215517507416,
    11	 "_seed_note": "the SEED effect is a point estimate; the bootstrap resamples QUERIES and cannot see it. A screen that resolves less than this has resolved noise.",
    12	 "seed_pair": {
    13	  "delta_raw": 0.0013869215517507644,
    14	  "lower_bound_raw": -0.00276096580013143,
    15	  "distance_raw": 0.004147887351882195,
    16	  "draws_sd": 0.001419113480192664,
    17	  "B": 200000,
    18	  "seed": 0,
    19	  "chunk": 5000,
    20	  "quantile_method": "inverted_cdf",
    21	  "plan_sample_sha256": "8091783fa66a21d8fcdd3a1c6458920c3c52442fc85cb5c5d9990f37a90e3d68",
    22	  "draws_sha256": "5a2a4b687f6045f60cba941bdc2d91b7c93fcced9d962121ad7a4af50b26a1e9",
    23	  "per_family_delta": {
    24	   "BRIGHT": 0.00260343697525285,
    25	   "consumer-health": 0.0041850955601164955,
    26	   "finance": -0.001665033318789806,
    27	   "legal": 0.00042418699042351816
    28	  }
    29	 },
    30	 "lr_pair": {
    31	  "delta_raw": 0.0036359902624266867,
    32	  "lower_bound_raw": 0.0007586538468388103,
    33	  "distance_raw": 0.0028773364155878765,
    34	  "draws_sd": 0.0010107934977571485,
    35	  "B": 200000,
    36	  "seed": 0,
    37	  "chunk": 5000,
    38	  "quantile_method": "inverted_cdf",
    39	  "plan_sample_sha256": "8091783fa66a21d8fcdd3a1c6458920c3c52442fc85cb5c5d9990f37a90e3d68",
    40	  "draws_sha256": "7e6efb05a500cba8029c1d0dbef85483bf4d1247ac0a2a9bb084f04167230dd4",
    41	  "per_family_delta": {
    42	   "BRIGHT": 0.0027951801074408184,
    43	   "consumer-health": 0.002777557500422136,
    44	   "finance": 0.00483775866386807,
    45	   "legal": 0.004133464777975722
    46	  }
    47	 },
    48	 "unrelated_models_distance": 0.008618964156860275,
    49	 "ratio_lr_pair_to_unrelated": 0.333837844458102,
    50	 "MDE": 0.0056,
    51	 "reading": "A same-init contrast's distance is 0.00288 against 0.00862 between unrelated models (0.33x). The seed effect is 0.00139. Both are DISCLOSURES: B/D/G/C are read against the first, F and E against the second, and the MDE decision is Dylan's before family F."
    52	}

exec
/bin/bash -lc "nl -ba m10src/nano10.py | sed -n '247,335p'
nl -ba m10src/run_arm.py | sed -n '306,385p'
python3 - <<'PY'
import json, math
from statistics import NormalDist
with open('m10/screen_registry.json') as f:
    r=json.load(f)
a=[(k,v['dose_examples']) for k,v in r['arms'].items() if v['trained']]
print('trained:', len(a), 'contrasts:', len(r['contrasts']), 'total_examples:',sum(n for _,n in a))
print('box_examples:',sum(n for k,n in a if k"'!='"'E-bs128'))
print('box_hours_at512:', sum(n for k,n in a if k"'!='"'E-bs128')/512/3600)
print('training_hours_at624:', {str(n):n/624/3600 for n in (3750000,5000000,7500000,20000000,45000000,66700000)})
print('normal_z:', NormalDist().inv_cdf(1-0.025/12))
print('generated_query_share:',7/12, 'all_example_share:',.75*7/12, 'A3_old_form_dose:',3750000/5,'A4_old_form_dose:',3750000/12)
print('isotropic_D_COV_scale:',1/1024)
for label,w,d in [('replicated reversal',[.52,.48,.48],[.50,.46,.46]),('unresolved bs128',[.497,.498,.498],[.50,.501,.501])]:
    margin=w[0]-d[0]
    spread=max(max(w)-min(w),max(d)-min(d))
    print(label, {'margin':margin,'seed_range':spread,'stands':margin>spread,'paired_seed_deltas':[x-y for x,y in zip(w,d)]})
print('sign reversal accepted:', .02 > max(.014,.014), 'seed0_delta', .02,'replicate_deltas',[-.008,-.008])
PY" in /home/dylan/asymetric-dual-encoders
 succeeded in 0ms:
   247	# ---- the kill and plateau rules (§Kill) -------------------------------------------------------
   248	
   249	KILL_DROP, PLATEAU_MIN_GAIN, PLATEAU_FROM_CYCLE = 0.0056, 0.003, 3
   250	
   251	
   252	def kill_fires(evals, kind_of, drop=KILL_DROP):
   253	    """§Kill: two CONSECUTIVE scheduled evaluations more than `drop` below the best evaluation OF
   254	    THEIR OWN KIND — midpoints against midpoints, cycle ends against cycle ends — so the rule can
   255	    fire INSIDE a build and not only at its end (Opus M5).
   256	
   257	    **A registered rule with two readings, and the plain one is implemented (T2-9).**
   258	    "Two consecutive scheduled evaluations" can mean consecutive in the SCHEDULE (a midpoint and
   259	    the cycle end after it, each below its own kind's best) or consecutive WITHIN a kind (two
   260	    successive midpoints). The schedule reading is the plain sense of the words and is the safer
   261	    of the two: it demands both kinds be failing at once, where the per-kind reading kills an arm
   262	    on two bad midpoints alone. A false kill costs a whole arm, and the rule's stated purpose —
   263	    "so the rule can fire inside the build and not only at its end" — is served either way,
   264	    because it is satisfied by midpoints entering the comparison at all.
   265	
   266	    `evals` is the ordered list of scheduled evaluation values; `kind_of(i)` returns that
   267	    evaluation's kind. A non-finite value fires immediately: that is the other half of the rule.
   268	    -> (fired, reason)
   269	    """
   270	    best = {}
   271	    bad = 0
   272	    for i, m in enumerate(evals):
   273	        if m is None or not np.isfinite(m):
   274	            return True, f"non-finite evaluation at index {i}"
   275	        k = kind_of(i)
   276	        b = best.get(k)
   277	        if b is not None and m < b - drop:
   278	            bad += 1
   279	            if bad >= 2:
   280	                return True, (f"two consecutive evaluations more than {drop} below the best of "
   281	                              f"their own kind (index {i}, kind {k!r}, {m:.4f} vs best {b:.4f})")
   282	        else:
   283	            bad = 0
   284	        best[k] = m if b is None else max(b, m)
   285	    return False, None
   286	
   287	
   288	def plateau_fires(cycle_end_evals, min_gain=PLATEAU_MIN_GAIN, from_cycle=PLATEAU_FROM_CYCLE):
   289	    """§Kill: read BEST-TO-BEST on annealed checkpoints only. Fires at the first cycle end
   290	    k >= `from_cycle` (1-based) where `m_k - max(m_1..m_{k-1}) >= min_gain` FAILS. Independent of
   291	    the cycle cap (Opus M6, Codex 2026-09-04). -> (fired, cycle_index_1based or None)
   292	    """
   293	    for k in range(from_cycle, len(cycle_end_evals) + 1):
   294	        gain = cycle_end_evals[k - 1] - max(cycle_end_evals[:k - 1])
   295	        if gain < min_gain:
   296	            return True, k
   297	    return False, None
   298	
   299	
   300	# ---- the 4-step mix window (family B) --------------------------------------------------------
   301	
   302	WINDOWS = {"100/0": ("Q", "Q", "Q", "Q"), "75/25": ("Q", "Q", "Q", "D"),
   303	           "50/50": ("Q", "Q", "D", "D")}
   304	
   305	
   306	def mix_window(pattern, step):
   307	    """-> 'Q' or 'D' for a given step. A fixed 4-step window, not a per-step coin flip, so an
   308	    arm's query/document ratio is exact at every checkpoint rather than only in expectation."""
   309	    w = WINDOWS[pattern]
   310	    return w[step % len(w)]
   311	
   312	
   313	def window_shares(pattern, steps):
   314	    q = sum(1 for s in range(steps) if mix_window(pattern, s) == "Q")
   315	    return {"Q": q, "D": steps - q, "q_share": q / max(steps, 1)}
   316	
   317	
   318	# ---- the cyclic schedule (§Recipe) -----------------------------------------------------------
   319	
   320	WARMUP_STEPS = 2000        # §Recipe (instructions-m10.md:580): "2,000 warmup steps in cycle 1"
   321	
   322	
   323	def lr_at(step, total_steps, cycles=3, peak=1e-4, final=1e-5, warmup=WARMUP_STEPS):
   324	    """LEAF's small-batch cyclic schedule: `cycles` cycles, each a linear decay from `peak` to
   325	    `final`, restarting at `peak`. The last step of every cycle is a cycle END, which is where COV
   326	    is read and where the sign-stability clause looks.
   327	
   328	    **Cycle 1 opens with `warmup` linear warmup steps** — §Recipe: "3 cycles of equal example
   329	    count, each linear 1e-4 -> 1e-5; 2,000 warmup steps in cycle 1". So cycle 1 ramps
   330	    `peak * (i+1) / warmup` for its first `warmup` steps and then anneals peak -> final over the
   331	    REST of the cycle; cycles 2 and 3 restart at `peak` and are unchanged. Registered but
   332	    unimplemented until 2026-09-07 (Codex runner review, finding 3) — the M10.0-e calibration
   333	    therefore ran without it, which is disclosed rather than back-fitted.
   334	
   335	    A schedule shorter than the warmup still has to anneal (a 60-step smoke), so the warmup is
   306	def resolved_recipe(name, entry, reg, pattern, batch):
   307	    """The arm's recipe as the REGISTRY resolves it — every field defaulted from `anchor` where
   308	    the arm's own entry is silent, which is what `arm_smoke.SHAPES` is a hand copy OF.
   309	
   310	    `student` is None for every arm after family F: the registry says `anchor.init: "F's winner
   311	    backbone"` and names no student, so there is nothing to cross-check and the student comes
   312	    from F's verdict instead (`f_verdict`).
   313	    """
   314	    a = reg.get("anchor") or {}
   315	    return {
   316	        "student": entry.get("student"),
   317	        "objective": entry.get("objective") or a.get("objective") or "squared_l2",
   318	        "feature_layers": int(entry.get("feature_layers")
   319	                              if entry.get("feature_layers") is not None
   320	                              else a.get("feature_layers", 3)),
   321	        # G-MLP is the one arm carrying its own `head` (a formula string); every other arm is the
   322	        # anchor's linear head. C-M9init is the one arm carrying `head_init`, so its warm start is
   323	        # M9's checkpoint and every other arm's is the ridge (`warm_start.all_arms`).
   324	        "head": "mlp" if entry.get("head") else "linear",
   325	        "warm_start": ("m9" if entry.get("head_init") else
   326	                       ("mlp" if entry.get("head") else "linear")),
   327	        "pattern": pattern,
   328	        "batch": int(batch),
   329	    }
   330	
   331	
   332	def check_shape(arm, recipe, spec):
   333	    """Cross-check the UNTOUCHED `arm_smoke.SHAPES` entry against the fully resolved registry
   334	    recipe, BEFORE anything is copied into it.
   335	
   336	    Finding 6 (Codex runner review 2026-09-07): `arm_plan` used to write the registry's batch INTO
   337	    `spec` and then compare the two, so the batch check could not fail; head, mix pattern and warm
   338	    start were never compared at all. Every field the two representations share is compared here.
   339	    """
   340	    bad = []
   341	    if recipe["student"] is not None:
   342	        want = STUDENT_ALIAS.get(recipe["student"])
   343	        if want is None:
   344	            bad.append(f"registry student {recipe['student']!r} is not a known nano10 student")
   345	        elif want != spec["student"]:
   346	            bad.append(f"student: registry {recipe['student']!r} -> {want!r} vs shape "
   347	                       f"{spec['student']!r}")
   348	    for field, in_spec in (("objective", "loss"), ("feature_layers", "n_layers"),
   349	                           ("head", "head"), ("pattern", "pattern"), ("batch", "batch")):
   350	        if recipe[field] != spec[in_spec]:
   351	            bad.append(f"{field}: registry {recipe[field]!r} vs shape {spec[in_spec]!r}")
   352	    if recipe["warm_start"] != spec.get("warm_start", "linear"):
   353	        bad.append(f"warm_start: registry {recipe['warm_start']!r} vs shape "
   354	                   f"{spec.get('warm_start', 'linear')!r}")
   355	    if bad:
   356	        refuse(f"arm {arm!r}: registry and arm_smoke.SHAPES disagree — " + "; ".join(bad))
   357	
   358	
   359	def schedule(dose, batch, entry):
   360	    """-> (total_steps, cycle_end_steps, [(examples, step)] read points, rounding report).
   361	
   362	    The schedule is 3 cycles over THE ARM'S OWN dose (F's 20M arms therefore anneal over 20M and
   363	    are merely READ at 5M/10M/20M), so `total_steps = dose // batch` and the cycle ends are
   364	    `nano10.cycle_ends`, which is what `lr_at` anneals to.
   365	
   366	    **Registry ambiguity, floored.** `E-bs128`'s registered dose 5,000,000 is not a whole number
   367	    of batches of 128 (39,062.5), so family E's "equal examples and identical schedule" cannot be
   368	    exact at both batch sizes. The dose is treated as a CAP: 39,062 steps = 4,999,936 examples, 64
   369	    fewer than bs32's 5,000,000 (0.0013%), because the other rounding spends unregistered compute.
   370	    Recorded in the arm's record, never silently.
   371	    """
   372	    total = dose // batch
   373	    rounding = {"dose_registered": int(dose), "examples_run": int(total * batch),
   374	                "examples_dropped": int(dose - total * batch), "rule": "floor (the dose is a cap)"}
   375	    ends = N.cycle_ends(total, CYCLES)
   376	    reads = []
   377	    for r in entry.get("read_at") or []:
   378	        r = int(r)
   379	        if r > dose:
   380	            refuse(f"read_at {r:,} exceeds the arm's dose {dose:,}")
   381	        step = r // batch - 1
   382	        reads.append((r, step))
   383	        if r % batch:
   384	            rounding.setdefault("read_at_floored", {})[str(r)] = int((step + 1) * batch)
   385	    return total, ends, reads, rounding
trained: 14 contrasts: 12 total_examples: 101250000
box_examples: 96250000
box_hours_at512: 52.218967013888886
training_hours_at624: {'3750000': 1.669337606837607, '5000000': 2.2257834757834756, '7500000': 3.338675213675214, '20000000': 8.903133903133902, '45000000': 20.03205128205128, '66700000': 29.691951566951566}
normal_z: 2.8652602385321333
generated_query_share: 0.5833333333333334 all_example_share: 0.4375 A3_old_form_dose: 750000.0 A4_old_form_dose: 312500.0
isotropic_D_COV_scale: 0.0009765625
replicated reversal {'margin': 0.020000000000000018, 'seed_range': 0.040000000000000036, 'stands': False, 'paired_seed_deltas': [0.020000000000000018, 0.019999999999999962, 0.019999999999999962]}
unresolved bs128 {'margin': -0.0030000000000000027, 'seed_range': 0.0010000000000000009, 'stands': False, 'paired_seed_deltas': [-0.0030000000000000027, -0.0030000000000000027, -0.0030000000000000027]}
sign reversal accepted: True seed0_delta 0.02 replicate_deltas [-0.008, -0.008]

codex
The arithmetic checks are complete. The base screen is 101.25M training examples, with 96.25M assigned to the box; 52.2 hours is consistent with the conservative 512 examples/s assumption. The bigger problems are decision logic: the confirmation rule can accept a winner that loses on both new seeds, batch 128 can be selected and then automatically rejected, and two successful objective alternatives can tie and send the build back to the inferior default. I’m organizing the recommendations around those fixes and a smaller number of experiments that test transfer to the build.
codex
**The confirmation rule is wrong in a way that can change the build. The head diagnosis also rests on an optimization argument that does not hold for the implemented loss.** I would fix those before treating this screen as a reliable recipe selector.

I read all six requested files, including the amendment blocks, and additionally read `instructions-m14.md`, the calibration report, and targeted implementation code. No excluded files were read, no evaluations were run, and no files were changed.

**First, the errors that should lead the review.**

1. **The confirmation rule can reject perfect replication and accept a reversal.**

   The registry defines the original-screen margin, then says:

   > “the winner’s margin exceeds the largest seed range observed in either arm”

   [screen_registry.json:309](/home/dylan/asymetric-dual-encoders/m10/screen_registry.json:309)

   These hypothetical scores demonstrate the problem:

   | Case | Original winner, seeds 0/1/2 | Default, seeds 0/1/2 | Paired differences | Registered outcome |
   |---|---|---|---|---|
   | Perfect replication with a shared seed shift | .520 / .480 / .480 | .500 / .460 / .460 | +.020 / +.020 / +.020 | **Reject**: original .020 margin does not exceed .040 range |
   | Reversal on both confirmation seeds | .520 / .506 / .506 | .500 / .514 / .514 | +.020 / −.008 / −.008 | **Accept**: original .020 margin exceeds .014 range |

   The rule measures variation in absolute scores when the relevant quantity is variation in **paired treatment effects**. It also retains the selected seed-0 advantage as the numerator instead of asking whether the fresh seeds reproduce it.

   This is not a philosophical objection to three seeds. It is a counterexample to the actual decision rule.

2. **Family E’s cost decision is undone by confirmation.**

   The registry says:

   > “Select **bs32 iff E1 (`bs32 - bs128`) RESOLVES**; in every other case … select **bs128**.”

   It then explicitly makes bs128 a non-default outcome eligible for confirmation. [screen_registry.json:381](/home/dylan/asymetric-dual-encoders/m10/screen_registry.json:381)

   Suppose bs32 leads by .003, unresolved. E selects bs128 to save training cost. But the confirmation margin is now:

   \[
   \text{bs128}-\text{bs32}=-0.003.
   \]

   That cannot exceed a nonnegative seed range. Confirmation necessarily returns to bs32. If E fails to receive one of the two confirmation slots, it also returns to the default.

   Thus the advertised “accept a small unresolved quality disadvantage to save cost” branch cannot survive the full decision procedure. **A quality-superiority confirmation rule cannot implement a cost-based choice.**

3. **The objective-family tie rule discards successful improvements.**

   > “if D-NORM and D-COV both resolve with margins within 1e-4, the default (squared L2) stands”

   [screen_registry.json:386](/home/dylan/asymetric-dual-encoders/m10/screen_registry.json:386)

   If both alternatives reliably improve on squared L2 by about .015, their agreement is a reason to choose one using a deterministic preference. It is not evidence for returning to squared L2. The generic multi-arm rule similarly returns to the default on a point-estimate tie. [screen_registry.json:389](/home/dylan/asymetric-dual-encoders/m10/screen_registry.json:389)

4. **PCA does not identify the optimum of your normalized regression objective.**

   The head diagnosis says:

   > “PCA gives the reconstruction-optimal k-d subspace … the subspace an L2-regression objective pushes such a student toward”

   [PLANNING.md:314](/home/dylan/asymetric-dual-encoders/m10/PLANNING.md:314)

   But the implemented student normalizes its output, and the loss compares unit vectors. [nano10.py:107](/home/dylan/asymetric-dual-encoders/m10src/nano10.py:107), [nano10.py:185](/home/dylan/asymetric-dual-encoders/m10src/nano10.py:185)

   Even in the simpler bias-free case, let \(S\) be an output subspace and \(t\) a unit teacher target. The best unit student vector in \(S\) minimizes

   \[
   \min_{\substack{s\in S\\\|s\|=1}}\|s-t\|^2
   =2-2\|P_St\|.
   \]

   Selecting the subspace therefore maximizes \(\mathbb E\|P_St\|\). Ordinary unnormalized reconstruction maximizes \(\mathbb E\|P_St\|^2\). These are different objectives. Your affine head and restricted backbone add further differences.

   Saying normalization leaves retrieval rankings unchanged does not make normalization irrelevant to training.

   **The PCA experiment is useful evidence that particular compressed teacher representations lose retrieval quality. It does not establish that normalized L2 training is driven toward those representations.** The existing caveat—“not an upper bound”—does not repair that separate inference.

   There is another important distinction: the 1152→1024 anchor already permits full output rank. G-1536 cannot increase the maximum output rank beyond 1024. It tests access to another layer’s features and a different parameterization, not further removal of a 1024-dimensional output-rank bottleneck.

5. **The current count and power story differ from the brief.**

   I recomputed these directly from the registry:

   | Quantity | Current registry |
   |---|---:|
   | Trained base-arm entries | **14** |
   | Contrast records | **12**, including the cut C1 |
   | Base-screen training examples | **101.25M** |
   | Box-assigned examples, excluding E-bs128 | **96.25M** |
   | Box estimate at 512 examples/s | **52.22 hours** |
   | Ordinary one-sided quantile | **0.025/12** |
   | F1 quantile per tail | **0.025/24** |

   So the 52-hour arithmetic is sound. The denominator of 13 is stale; I am not recommending changing the current denominator after observation. The authoritative amendment and numeric registry fields agree on 12. [instructions-m10.md:159](/home/dylan/asymetric-dual-encoders/instructions-m10.md:159), [screen_registry.json:274](/home/dylan/asymetric-dual-encoders/m10/screen_registry.json:274)

   More importantly, “the surface cannot resolve an MDE-sized contrast” is too broad. The registry uses the unrelated-model distance **.008619** to make that statement, but your same-init calibration measured a distance of **.002877** for its LR contrast. [screen_registry.json:286](/home/dylan/asymetric-dual-encoders/m10/screen_registry.json:286), [m10_calib_report.json:30](/home/dylan/asymetric-dual-encoders/results/m10_calib_report.json:30)

   Neither number is a universal resolution limit. Width depends on the actual paired differences. The calibration is also imperfect—the code records that it ran before warmup was implemented—but it is sufficient to invalidate the universal impossibility argument. [nano10.py:328](/home/dylan/asymetric-dual-encoders/m10src/nano10.py:328)

   The distinction matters: **some experiments are badly targeted; that does not mean all small effects are statistically invisible.**

**The biggest structural error is assuming that the screen’s treatment effects transfer to the build.**

The screen principally asks:

> Which isolated changes improve a short, three-cycle training run on COV, around the A4/bs32/1152/squared-L2 anchor?

The release decision asks:

> Does a much longer, combined recipe beat the bge-small system on clean-4?

Between those questions, the plan changes:

- Training duration: **5M → 200M examples**.
- Potentially the data: **A4 → A3**.
- Potentially the batch: **32 → 128**.
- Several recipe components simultaneously.
- The evaluation distribution and relevance tasks.

The reordered F family fixed one transfer assumption: backbone identity. The remaining assumptions are still substantial.

This goes beyond W14’s already-recorded surface mismatch. Even if COV were a perfect proxy for clean-4, **the individual effects were measured on a recipe that the build may never use**. For example, a head or objective can help under A4’s heavy generated-form exposure and fail under A3. A small-batch advantage after 5M examples can disappear after 200M. Independent improvements can interfere when combined.

The synthesized 5M arm and LoTTE veto provide some protection against combination failure. They do not validate the long-dose extrapolation, and the LoTTE procedure detects sufficiently clear harm rather than establishing benefit. [instructions-m10.md:743](/home/dylan/asymetric-dual-encoders/instructions-m10.md:743)

**If this transfer assumption is wrong, the screen can execute perfectly and answer a question nobody needs answered: which recipe wins a short COV tournament.**

I would leave the fixed teacher and shared-index requirement alone. They define a coherent engineering problem. The unsupported step is promoting these development measurements into confidence about the build.

**Here is what I would cut from a redesigned screen.**

Because the screen is live, these recommendations must not erase completed rows or retroactively change their registered interpretation. Where an arm is already complete, its training cost is sunk. The useful savings then come from avoiding confirmations, extensions, or repetitions.

| Arm or contrast | Recommendation | Reason | Avoidable training at 624 examples/s |
|---|---|---|---:|
| **A1 / A2−A1** | Retire after the existing measurement; no repetitions | It cannot change the build’s PAQ allocation, and it is not a clean pure-volume experiment | About 2.2 h per repetition |
| **B-50/50 / B2** | Cut from the prospective recipe-selection budget | It measures adding document training at fixed query exposure, but the build fixes total examples and would remove query exposure | About 3.3 h |
| **G-1536 / G2** | Cut before G-384 or G-MLP | The anchor already allows full output rank; this is an additional-feature experiment with lower diagnostic value | About 2.2 h |
| **D-NORM / D1** | Defer in a smaller design | It is an inexpensive implementation change, but still a full training arm. Its strongest rationale is recipe resemblance, not a diagnosis specific to this miss | About 2.2 h |
| **Generic two-decision confirmation block** | Replace its decision rule and redirect substantial compute toward build-regime validation | Its current statistic is invalid for its purpose; its priority order rewards large original wins rather than resolving consequential uncertainty | Up to 45M examples, about 20 h |
| **Automatic bottom-two-form top-up** | Remove from the proposed build policy | It spends a full extension cycle on an unvalidated proxy for which training allocation will improve retrieval | 66.7M examples, about 29.7 h at this reference rate |

These are reference training times, excluding evaluation, and are not claims about A100 throughput. Different heads and mixes need their own measured rates.

I would retain **G-384**, **G-MLP**, **D-COV**, **B-100/0**, the A3/A4 comparison, and the batch comparison, with the corrections below. They test more consequential possibilities: whether the wider representation matters, whether nonlinear features help, whether corpus-sensitive regression helps, whether document-role training is needed, and whether the proposed training distribution and batch policy are worth using.

The discretionary base-arm cuts above save roughly **7.8 hours** if those three arms remain unspent. That is useful but not transformative. The larger opportunity is to stop spending replication and extension compute on rules that do not establish what the build needs.

**Several retained contrasts need narrower interpretations or different actions.**

**Family A measures changes in the training distribution, not isolated “forms” or “generation.”** The registry says the equal-count cut exists because:

> “without it the forms contrast is confounded with volume, which is the one thing family A exists to separate”

[screen_registry.json:348](/home/dylan/asymetric-dual-encoders/m10/screen_registry.json:348)

Equal available-text counts remove one difference. They do not isolate form from source, topic, teacher difficulty, or presentation allocation.

The actual implementation makes every arm form-balanced. The M9 and PAQ texts occupy factoid/product forms; the realized harvest contributes title/keyword/claim; generation contributes seven more. [corpus_loader.py:125](/home/dylan/asymetric-dual-encoders/m10src/corpus_loader.py:125), [corpus_loader.py:1401](/home/dylan/asymetric-dual-encoders/m10src/corpus_loader.py:1401), [STATUS.md:85](/home/dylan/asymetric-dual-encoders/m10/STATUS.md:85)

Consequently:

| Arm | Present forms | Query-presentation share per form | Generated-form share of query presentations |
|---|---:|---:|---:|
| A2 | 2 | 50% | 0% new Qwen forms |
| A3 | 5 | 20% | 0% new Qwen forms |
| A4 | 12 | 8.33% | **58.33%** |

At the registered 3.75M query presentations, an existing form receives about **750,000 presentations in A3 versus 312,500 in A4**.

Thus A4−A3 tests replacing a large fraction of existing-form training with seven generated forms. A null could mean:

- Those generated forms do not help COV.
- They help, but removing existing-form exposure offsets the gain.
- They need a different exposure allocation.
- They need more training.
- They help tasks that COV does not represent.

The current rule collapses these possibilities into:

> “dropped from the build and reported as a measured null”

[screen_registry.json:377](/home/dylan/asymetric-dual-encoders/m10/screen_registry.json:377)

**Drop “measured null.”** The valid conclusion is “this allocation did not establish improvement under the registered test.” A positive effect with an uncertain interval is not a measured absence of benefit.

Also, A3 contains PAQ, which the mandate itself identifies as machine-generated questions. This is **incremental Qwen generation versus an existing corpus containing PAQ**, not synthetic versus real data. [instructions-m10.md:202](/home/dylan/asymetric-dual-encoders/instructions-m10.md:202)

A3−A2 likewise estimates a particular source/form-distribution replacement. It cannot establish that *query form itself* caused a gain.

**Family B’s screen and build budgets define different interventions.** The screen holds query presentations at 3.75M while document presentations increase from 0 to 1.25M to 3.75M. That is a reasonable experiment about adding document training. [instructions-m10.md:628](/home/dylan/asymetric-dual-encoders/instructions-m10.md:628)

But at a fixed 200M-example build:

| Mix | Query presentations | Document presentations |
|---|---:|---:|
| 100/0 | 200M | 0 |
| 75/25 | 150M | 50M |
| 50/50 | 100M | 100M |

Adopting 50/50 therefore removes 50M query presentations relative to the default. Its screen contrast did not test that tradeoff.

For a future design, either compare mixes at the actual build resource constraint or keep query exposure fixed in the build and explicitly purchase extra document exposure. Until then, describe B as an augmentation experiment and limit what its result decides.

**G-384 is allowed to challenge the diagnosis but forbidden from correcting the recipe.**

> “384 is not a candidate recipe — it is the thing the M9 diagnosis says is too narrow”

[screen_registry.json:99](/home/dylan/asymetric-dual-encoders/m10/screen_registry.json:99)

And:

> “A negative point estimate is REPORTED … it is not an action”

[screen_registry.json:193](/home/dylan/asymetric-dual-encoders/m10/screen_registry.json:193)

This is a poor design if the scientific premise is uncertain. Even a very large, reproducible advantage for G-384 cannot select it. Its exclusion follows from the diagnosis that the experiment supposedly tests.

I would retain the measurement and, in a revised selection policy, permit G-384 to win on quality and resource cost. If its diagnostic-only status stays, remove claims that this experiment “decides” the appropriate width. It provides evidence for or against a prescribed width.

**E1 measures a whole optimizer policy, not batch size under an identical exposure schedule.** Warmup is fixed at 2,000 **steps**. That is 64,000 examples for bs32 and 256,000 for bs128—1.28% versus 5.12% of the screen. [nano10.py:320](/home/dylan/asymetric-dual-encoders/m10src/nano10.py:320)

Batch 128 also gets approximately one-quarter as many AdamW updates, including weight-decay applications. These are legitimate consequences of a batch-policy comparison, but they weaken a mechanistic “small batches generalize better” interpretation and its extrapolation to a 40× longer run.

The material discrepancy is the warmup exposure and update regime, not the 64-example rounding difference the runner carefully documents. [run_arm.py:366](/home/dylan/asymetric-dual-encoders/m10src/run_arm.py:366)

**D-COV is a useful score-error surrogate, but its stated mechanism is too strong.** For independent documents drawn from the covariance population,

\[
\mathbb E\left[((s-t)^\top(d_i-d_j))^2\right]
=2(s-t)^\top\Sigma(s-t).
\]

That gives D-COV a clean interpretation: it penalizes errors in teacher/student score differences between random corpus documents. It does not specifically weight relevant-versus-near-top-irrelevant comparisons, which determine nDCG.

Also, the code says unit-trace normalization makes it comparable to the anchor at the same learning rate. [nano10.py:237](/home/dylan/asymetric-dual-encoders/m10src/nano10.py:237) In the isotropic case, however,

\[
L_{\mathrm{COV}}=\frac{1}{1024}\|s-t\|^2.
\]

AdamW may largely cancel constant gradient scaling in some conditions; clipping and epsilon mean this is not an exact optimizer equivalence. Audit the gradients and updates before interpreting a negative D-COV result as evidence against document-aware regression.

**The concrete additions I would prioritize are below.** The ranking is qualitative information gained per cost. It deliberately starts with analyses that require no new training.

| Rank | Exact addition | Cost | What it changes |
|---|---|---|---|
| **1** | **Executable decision examples** covering the three confirmation/tie counterexamples above, including E’s complete path through confirmation | No GPU; roughly 1–2 engineering hours | Establishes that the final recipe selector implements its intended decisions. Fresh-seed confirmation should read paired differences; E needs its own cost/quality rule |
| **2** | **One realized-exposure table per arm**: source × form presentations, distinct texts actually visited, repeated presentations, token-length distribution, and query/document tokens | CPU analysis of manifests/sampler draws; no teacher encoding | Corrects the interpretation of A and reprices the build using the distribution actually presented |
| **3** | **Deployment-quality bridge**: on a fixed development sample, compare exact floating-point search, exact search using the intended compressed representation, and the deployed ANN configuration for zero/nano/bge-small | No training or new teacher encodes if vectors exist; budget a few CPU hours for a bounded sample | Determines whether the quality and latency coordinates describe the same deployed systems |
| **4** | **Two bounded mechanism audits**: normalized-loss fitting of a rank-384 head using the *same 1152-d features*; D-COV versus L2 gradient/update-scale measurements on identical training batches | Small matrix work and short runs; cap at 1 GPU-hour | Separates the rank explanation from extra-layer information, and detects an optimizer-scale explanation for D-COV behavior |
| **5** | **`SYNTH-20M` versus the exact anchor recipe at 20M**, both with the same complete 20M schedule, evaluating the assembled recipe on existing development surfaces | 20M new examples if F-bge-small is an exact reusable comparator; otherwise 40M. Approximately 8.9 or 17.8 training hours at the reference rate | Tests whether combined improvements survive beyond the compressed 5M experiment. Fund this by replacing much of the existing confirmation allocation |
| **6, conditional** | **`A4-G20`**: same A4 text corpus and 5M recipe, but generated forms receive **20% total query exposure**, split equally across seven forms; the five existing forms receive 16% each. Compare descriptively with A3 and current A4 | One 5M arm, about 2.2 training hours plus evaluation; no regeneration | Distinguishes failure of a 58% generated-exposure policy from failure of generated text to help at a moderate allocation |

`A4-G20` is an illustrative fixed policy, not an estimated optimum. I would run it only if the generation decision remains consequential enough to justify another arm. I would not start a mixture-weight sweep.

The rank audit is also conditional on wanting to claim a mechanism. If the paper only reports that the tested multilayer recipe performed better, you can save that work by dropping the rank-causation claim.

The 20M comparison must start under a genuine matching schedule. Simply continuing the already-annealed 5M candidate would reproduce the schedule mismatch the project previously caught.

Because COV is observed, these additions cannot be smuggled into the original confirmatory accounting. Preserve the original registry and verdicts; any action-bearing redesign needs a disclosed amendment. The analyses can still provide useful development evidence without changing the headline partition, admitting another COV family, or moving the registered thresholds.

**The pre-registration machinery is buying real rigor at the boundaries, and considerable paperwork inside the search.**

The parts I would preserve are concrete:

- Frozen comparator identities, query-ID alignment, and paired scoring.
- Protected-data boundaries and training-data provenance.
- A fixed final candidate and final claim sequence.
- Disclosure of every trained arm, failure, and development exposure.
- Serving parity and resource-cap checks.
- Atomic evaluation transactions and preservation of their outputs.

The fixed-sequence final test is a defensible way to align the release decision with the headline partition. The conservative family-weighted surface also prevents a large dataset from silently determining the entire development objective. Neither guarantees deployment generalization, but both have a clear job.

The harmful parts are these.

**First, the registry confuses conservative selection with confirmation of scientific explanations.** Requiring a substantial lower bound to select an expensive intervention can be sensible. Reporting a failed superiority test as a null is not. Likewise, choosing a default under uncertainty is a product policy, not evidence that the default is equivalent or best.

For a future screen, separate:

- **Development decisions:** expected quality, resource cost, downside risk, and replication.
- **Paper claims:** explicitly scoped effects with appropriate uncertainty and multiplicity control.

You do not need to spend the same inferential machinery on a diagnostic arm that cannot affect the recipe and on a change that materially affects build cost.

**Second, “descriptive only” is being used where the measurement actually controls training.**

The registry says:

> “FORMS-12”: “descriptive only”

[screen_registry.json:318](/home/dylan/asymetric-dual-encoders/m10/screen_registry.json:318)

But the plateau response doubles the presentation weights of the bottom two FORMS-12 forms. [instructions-m10.md:552](/home/dylan/asymetric-dual-encoders/instructions-m10.md:552)

That is adaptive training driven by a validation set. It can be legitimate if disclosed, but it is not descriptive.

The same response depends on the dev→six forecast, while the surfaces section says:

> “forecasting only, never gating”

[instructions-m10.md:750](/home/dylan/asymetric-dual-encoders/instructions-m10.md:750)

If the forecast includes M9’s close-out results and controls whether an extra training cycle runs, those six-set results affect M10. A deterministic, prewritten branch does not make that influence disappear.

I would remove this top-up rule. The bottom two forms by teacher agreement are not necessarily the forms whose additional training has the greatest marginal value for C1b. “Worst current score” and “best next use of compute” are different quantities.

**Third, much of the validator checks that a decision is named, not that its behavior is coherent.** It verifies family-rule names and nonempty outcome entries. [screen_lock.py:83](/home/dylan/asymetric-dual-encoders/m10src/screen_lock.py:83), [screen_lock.py:132](/home/dylan/asymetric-dual-encoders/m10src/screen_lock.py:132)

That is why a document can accumulate multiple successful reviews while still carrying E’s impossible confirmation path. A handful of concrete decision examples would buy more than another layer of explanatory prose.

**Fourth, historical amendments remain mixed with operative instructions.** The mandate still contains the struck STOP, the retired anchor definition, old arm counts, and old quantiles. The JSON also contains incompatible historical count notes. The latest values can be recovered, but every reader has to reconstruct them.

The existing standing rule already supplies the right remedy:

> “One fact, one home.”

[CLAUDE.md:202](/home/dylan/asymetric-dual-encoders/CLAUDE.md:202)

Use a small current specification generated from executable fields. Keep the historical decisions and withdrawn claims in an archive with short pointers. Preserve history without making history the interface to the current experiment.

**Finally, drop the rule that every missing paper measurement requires a separate milestone.** M14 correctly forbids inventing numbers in prose, but “no new measurement” is stronger than needed. [instructions-m14.md:89](/home/dylan/asymetric-dual-encoders/instructions-m14.md:89) A labelled supplemental diagnostic with its own scope and provenance need not become another milestone bureaucracy.

**For M14, the strongest defensible paper is an empirical study of replacing the query encoder while preserving a pretrained document index.**

A defensible central claim would be:

> “We measure the retrieval quality and deployment costs of lookup-table and compact-transformer query encoders that reuse one frozen document index. Under the tested data and training policies, performance depends strongly on task distribution and the deployed retrieval configuration; the resulting gains and failures are reported on named benchmarks.”

The final numbers can strengthen that statement. A C1b pass would support the registered whole-system superiority claim on clean-4. A miss still supports a useful, reproducible account of the feasible tradeoffs under the constraints.

The teacher-selection evidence belongs in the paper, but its strongest formulation is an empirical counterexample to selecting by teacher quality—not a universal law of zero predictiveness. The M14 mandate itself gives Spearman values over **eight** and **seven** comparable rows while saying eleven teachers were measured. Those denominators must remain distinct. [instructions-m14.md:47](/home/dylan/asymetric-dual-encoders/instructions-m14.md:47)

There are several claims it should stop making:

- **“M9 was a coverage failure, not a capacity failure.”** Reaching 94% on one distribution does not establish capacity to reach that level jointly across many distributions. The M9 table supports distribution-dependent failure and a coverage hypothesis. It does not isolate the cause. [FINDINGS.md:26](/home/dylan/asymetric-dual-encoders/m9/FINDINGS.md:26)
- **“Normalized L2 cannot push a 384-wide head past 90–93%.”** The PCA argument does not establish that ceiling.
- **“A3−A2 establishes query-form coverage” or “A4−A3 measures whether synthetic data helps.”** Both are bundled training-distribution interventions.
- **“Unresolved means a measured null,” “not positive,” or a statistical tie.** These need effect estimates and uncertainty, with practical-policy labels kept separate.
- **“Clean-4 is contamination-free.”** Its useful property is no *disclosed* teacher overlap; the mandate correctly acknowledges development-informed training design.
- **“Fused systems are nearly contamination-immune.”** This claim appears explicitly in M14 and is unsupported by the cited arithmetic. [instructions-m14.md:38](/home/dylan/asymetric-dual-encoders/instructions-m14.md:38)

That last claim deserves a direct correction. The difference between all-6 and clean-4 is

\[
\overline S_6-\overline S_4
=\frac{1}{3}\left(\overline S_{\text{excluded two}}-\overline S_4\right).
\]

It measures a difference between two groups of datasets. It does not identify the effect of teacher contamination. A small difference could reflect fusion’s relative strengths across those tasks, BM25 contributions, or cancellation. Report **partition sensitivity**, not immunity.

Finally, the paper should credit the frozen-teacher lookup-table construction as prior art, as its own pyNIFE instruction requires. Its contribution must be the empirical evidence, analysis, and deployable implementation—not an architectural first claim. [instructions-m14.md:84](/home/dylan/asymetric-dual-encoders/instructions-m14.md:84)

**Two items are worth putting above another broad arm sweep.**

The first is the **deployment-quality bridge** in the additions table. M9 says the memory-constrained index was viable only under binary quantization, while the standing quality protocol uses exact search and treats Qdrant Edge as a latency exercise. [FINDINGS.md:49](/home/dylan/asymetric-dual-encoders/m9/FINDINGS.md:49), [CLAUDE.md:16](/home/dylan/asymetric-dual-encoders/CLAUDE.md:16)

That creates a risk of plotting full-precision quality against compressed-index latency. This is separate from the already-actioned missing cost measurements: **even perfectly measured costs are insufficient if the quality coordinate belongs to another retrieval configuration.** Query encoders can have different sensitivity to the same document compression. Measure that interaction.

The second is a cheap test of whether the **pair** provides useful complementarity: an **oracle escalation curve on existing development data**. Using per-query zero/nano scores, calculate the best achievable quality if only a fraction of queries can use nano. Compare it with random allocation and always using either encoder. This is explicitly an optimistic bound using relevance judgments, not a deployable router.

If even the oracle curve is weak, do not spend effort learning a router. If it is strong, you have identified a concrete potential benefit of maintaining both query paths on the same index. The incremental cost is score generation where missing and a small analysis, not another student build.

There is also one urgent preservation task already correctly identified in STATUS: ensure both pending six-set transactions emit aligned query IDs and per-query scores before M9’s close-out. That is more time-sensitive than any new hypothesis because the paired M9→M10 analysis may otherwise become irrecoverable. [STATUS.md:135](/home/dylan/asymetric-dual-encoders/m10/STATUS.md:135)

My recommendation is to **repair the decision rules, preserve the running screen’s record, and spend the next substantial compute allocation on validating the assembled recipe at a more relevant duration**. That would give the build a stronger evidential basis than confirming isolated short-run wins under the current machinery.
tokens used
147,746
**The confirmation rule is wrong in a way that can change the build. The head diagnosis also rests on an optimization argument that does not hold for the implemented loss.** I would fix those before treating this screen as a reliable recipe selector.

I read all six requested files, including the amendment blocks, and additionally read `instructions-m14.md`, the calibration report, and targeted implementation code. No excluded files were read, no evaluations were run, and no files were changed.

**First, the errors that should lead the review.**

1. **The confirmation rule can reject perfect replication and accept a reversal.**

   The registry defines the original-screen margin, then says:

   > “the winner’s margin exceeds the largest seed range observed in either arm”

   [screen_registry.json:309](/home/dylan/asymetric-dual-encoders/m10/screen_registry.json:309)

   These hypothetical scores demonstrate the problem:

   | Case | Original winner, seeds 0/1/2 | Default, seeds 0/1/2 | Paired differences | Registered outcome |
   |---|---|---|---|---|
   | Perfect replication with a shared seed shift | .520 / .480 / .480 | .500 / .460 / .460 | +.020 / +.020 / +.020 | **Reject**: original .020 margin does not exceed .040 range |
   | Reversal on both confirmation seeds | .520 / .506 / .506 | .500 / .514 / .514 | +.020 / −.008 / −.008 | **Accept**: original .020 margin exceeds .014 range |

   The rule measures variation in absolute scores when the relevant quantity is variation in **paired treatment effects**. It also retains the selected seed-0 advantage as the numerator instead of asking whether the fresh seeds reproduce it.

   This is not a philosophical objection to three seeds. It is a counterexample to the actual decision rule.

2. **Family E’s cost decision is undone by confirmation.**

   The registry says:

   > “Select **bs32 iff E1 (`bs32 - bs128`) RESOLVES**; in every other case … select **bs128**.”

   It then explicitly makes bs128 a non-default outcome eligible for confirmation. [screen_registry.json:381](/home/dylan/asymetric-dual-encoders/m10/screen_registry.json:381)

   Suppose bs32 leads by .003, unresolved. E selects bs128 to save training cost. But the confirmation margin is now:

   \[
   \text{bs128}-\text{bs32}=-0.003.
   \]

   That cannot exceed a nonnegative seed range. Confirmation necessarily returns to bs32. If E fails to receive one of the two confirmation slots, it also returns to the default.

   Thus the advertised “accept a small unresolved quality disadvantage to save cost” branch cannot survive the full decision procedure. **A quality-superiority confirmation rule cannot implement a cost-based choice.**

3. **The objective-family tie rule discards successful improvements.**

   > “if D-NORM and D-COV both resolve with margins within 1e-4, the default (squared L2) stands”

   [screen_registry.json:386](/home/dylan/asymetric-dual-encoders/m10/screen_registry.json:386)

   If both alternatives reliably improve on squared L2 by about .015, their agreement is a reason to choose one using a deterministic preference. It is not evidence for returning to squared L2. The generic multi-arm rule similarly returns to the default on a point-estimate tie. [screen_registry.json:389](/home/dylan/asymetric-dual-encoders/m10/screen_registry.json:389)

4. **PCA does not identify the optimum of your normalized regression objective.**

   The head diagnosis says:

   > “PCA gives the reconstruction-optimal k-d subspace … the subspace an L2-regression objective pushes such a student toward”

   [PLANNING.md:314](/home/dylan/asymetric-dual-encoders/m10/PLANNING.md:314)

   But the implemented student normalizes its output, and the loss compares unit vectors. [nano10.py:107](/home/dylan/asymetric-dual-encoders/m10src/nano10.py:107), [nano10.py:185](/home/dylan/asymetric-dual-encoders/m10src/nano10.py:185)

   Even in the simpler bias-free case, let \(S\) be an output subspace and \(t\) a unit teacher target. The best unit student vector in \(S\) minimizes

   \[
   \min_{\substack{s\in S\\\|s\|=1}}\|s-t\|^2
   =2-2\|P_St\|.
   \]

   Selecting the subspace therefore maximizes \(\mathbb E\|P_St\|\). Ordinary unnormalized reconstruction maximizes \(\mathbb E\|P_St\|^2\). These are different objectives. Your affine head and restricted backbone add further differences.

   Saying normalization leaves retrieval rankings unchanged does not make normalization irrelevant to training.

   **The PCA experiment is useful evidence that particular compressed teacher representations lose retrieval quality. It does not establish that normalized L2 training is driven toward those representations.** The existing caveat—“not an upper bound”—does not repair that separate inference.

   There is another important distinction: the 1152→1024 anchor already permits full output rank. G-1536 cannot increase the maximum output rank beyond 1024. It tests access to another layer’s features and a different parameterization, not further removal of a 1024-dimensional output-rank bottleneck.

5. **The current count and power story differ from the brief.**

   I recomputed these directly from the registry:

   | Quantity | Current registry |
   |---|---:|
   | Trained base-arm entries | **14** |
   | Contrast records | **12**, including the cut C1 |
   | Base-screen training examples | **101.25M** |
   | Box-assigned examples, excluding E-bs128 | **96.25M** |
   | Box estimate at 512 examples/s | **52.22 hours** |
   | Ordinary one-sided quantile | **0.025/12** |
   | F1 quantile per tail | **0.025/24** |

   So the 52-hour arithmetic is sound. The denominator of 13 is stale; I am not recommending changing the current denominator after observation. The authoritative amendment and numeric registry fields agree on 12. [instructions-m10.md:159](/home/dylan/asymetric-dual-encoders/instructions-m10.md:159), [screen_registry.json:274](/home/dylan/asymetric-dual-encoders/m10/screen_registry.json:274)

   More importantly, “the surface cannot resolve an MDE-sized contrast” is too broad. The registry uses the unrelated-model distance **.008619** to make that statement, but your same-init calibration measured a distance of **.002877** for its LR contrast. [screen_registry.json:286](/home/dylan/asymetric-dual-encoders/m10/screen_registry.json:286), [m10_calib_report.json:30](/home/dylan/asymetric-dual-encoders/results/m10_calib_report.json:30)

   Neither number is a universal resolution limit. Width depends on the actual paired differences. The calibration is also imperfect—the code records that it ran before warmup was implemented—but it is sufficient to invalidate the universal impossibility argument. [nano10.py:328](/home/dylan/asymetric-dual-encoders/m10src/nano10.py:328)

   The distinction matters: **some experiments are badly targeted; that does not mean all small effects are statistically invisible.**

**The biggest structural error is assuming that the screen’s treatment effects transfer to the build.**

The screen principally asks:

> Which isolated changes improve a short, three-cycle training run on COV, around the A4/bs32/1152/squared-L2 anchor?

The release decision asks:

> Does a much longer, combined recipe beat the bge-small system on clean-4?

Between those questions, the plan changes:

- Training duration: **5M → 200M examples**.
- Potentially the data: **A4 → A3**.
- Potentially the batch: **32 → 128**.
- Several recipe components simultaneously.
- The evaluation distribution and relevance tasks.

The reordered F family fixed one transfer assumption: backbone identity. The remaining assumptions are still substantial.

This goes beyond W14’s already-recorded surface mismatch. Even if COV were a perfect proxy for clean-4, **the individual effects were measured on a recipe that the build may never use**. For example, a head or objective can help under A4’s heavy generated-form exposure and fail under A3. A small-batch advantage after 5M examples can disappear after 200M. Independent improvements can interfere when combined.

The synthesized 5M arm and LoTTE veto provide some protection against combination failure. They do not validate the long-dose extrapolation, and the LoTTE procedure detects sufficiently clear harm rather than establishing benefit. [instructions-m10.md:743](/home/dylan/asymetric-dual-encoders/instructions-m10.md:743)

**If this transfer assumption is wrong, the screen can execute perfectly and answer a question nobody needs answered: which recipe wins a short COV tournament.**

I would leave the fixed teacher and shared-index requirement alone. They define a coherent engineering problem. The unsupported step is promoting these development measurements into confidence about the build.

**Here is what I would cut from a redesigned screen.**

Because the screen is live, these recommendations must not erase completed rows or retroactively change their registered interpretation. Where an arm is already complete, its training cost is sunk. The useful savings then come from avoiding confirmations, extensions, or repetitions.

| Arm or contrast | Recommendation | Reason | Avoidable training at 624 examples/s |
|---|---|---|---:|
| **A1 / A2−A1** | Retire after the existing measurement; no repetitions | It cannot change the build’s PAQ allocation, and it is not a clean pure-volume experiment | About 2.2 h per repetition |
| **B-50/50 / B2** | Cut from the prospective recipe-selection budget | It measures adding document training at fixed query exposure, but the build fixes total examples and would remove query exposure | About 3.3 h |
| **G-1536 / G2** | Cut before G-384 or G-MLP | The anchor already allows full output rank; this is an additional-feature experiment with lower diagnostic value | About 2.2 h |
| **D-NORM / D1** | Defer in a smaller design | It is an inexpensive implementation change, but still a full training arm. Its strongest rationale is recipe resemblance, not a diagnosis specific to this miss | About 2.2 h |
| **Generic two-decision confirmation block** | Replace its decision rule and redirect substantial compute toward build-regime validation | Its current statistic is invalid for its purpose; its priority order rewards large original wins rather than resolving consequential uncertainty | Up to 45M examples, about 20 h |
| **Automatic bottom-two-form top-up** | Remove from the proposed build policy | It spends a full extension cycle on an unvalidated proxy for which training allocation will improve retrieval | 66.7M examples, about 29.7 h at this reference rate |

These are reference training times, excluding evaluation, and are not claims about A100 throughput. Different heads and mixes need their own measured rates.

I would retain **G-384**, **G-MLP**, **D-COV**, **B-100/0**, the A3/A4 comparison, and the batch comparison, with the corrections below. They test more consequential possibilities: whether the wider representation matters, whether nonlinear features help, whether corpus-sensitive regression helps, whether document-role training is needed, and whether the proposed training distribution and batch policy are worth using.

The discretionary base-arm cuts above save roughly **7.8 hours** if those three arms remain unspent. That is useful but not transformative. The larger opportunity is to stop spending replication and extension compute on rules that do not establish what the build needs.

**Several retained contrasts need narrower interpretations or different actions.**

**Family A measures changes in the training distribution, not isolated “forms” or “generation.”** The registry says the equal-count cut exists because:

> “without it the forms contrast is confounded with volume, which is the one thing family A exists to separate”

[screen_registry.json:348](/home/dylan/asymetric-dual-encoders/m10/screen_registry.json:348)

Equal available-text counts remove one difference. They do not isolate form from source, topic, teacher difficulty, or presentation allocation.

The actual implementation makes every arm form-balanced. The M9 and PAQ texts occupy factoid/product forms; the realized harvest contributes title/keyword/claim; generation contributes seven more. [corpus_loader.py:125](/home/dylan/asymetric-dual-encoders/m10src/corpus_loader.py:125), [corpus_loader.py:1401](/home/dylan/asymetric-dual-encoders/m10src/corpus_loader.py:1401), [STATUS.md:85](/home/dylan/asymetric-dual-encoders/m10/STATUS.md:85)

Consequently:

| Arm | Present forms | Query-presentation share per form | Generated-form share of query presentations |
|---|---:|---:|---:|
| A2 | 2 | 50% | 0% new Qwen forms |
| A3 | 5 | 20% | 0% new Qwen forms |
| A4 | 12 | 8.33% | **58.33%** |

At the registered 3.75M query presentations, an existing form receives about **750,000 presentations in A3 versus 312,500 in A4**.

Thus A4−A3 tests replacing a large fraction of existing-form training with seven generated forms. A null could mean:

- Those generated forms do not help COV.
- They help, but removing existing-form exposure offsets the gain.
- They need a different exposure allocation.
- They need more training.
- They help tasks that COV does not represent.

The current rule collapses these possibilities into:

> “dropped from the build and reported as a measured null”

[screen_registry.json:377](/home/dylan/asymetric-dual-encoders/m10/screen_registry.json:377)

**Drop “measured null.”** The valid conclusion is “this allocation did not establish improvement under the registered test.” A positive effect with an uncertain interval is not a measured absence of benefit.

Also, A3 contains PAQ, which the mandate itself identifies as machine-generated questions. This is **incremental Qwen generation versus an existing corpus containing PAQ**, not synthetic versus real data. [instructions-m10.md:202](/home/dylan/asymetric-dual-encoders/instructions-m10.md:202)

A3−A2 likewise estimates a particular source/form-distribution replacement. It cannot establish that *query form itself* caused a gain.

**Family B’s screen and build budgets define different interventions.** The screen holds query presentations at 3.75M while document presentations increase from 0 to 1.25M to 3.75M. That is a reasonable experiment about adding document training. [instructions-m10.md:628](/home/dylan/asymetric-dual-encoders/instructions-m10.md:628)

But at a fixed 200M-example build:

| Mix | Query presentations | Document presentations |
|---|---:|---:|
| 100/0 | 200M | 0 |
| 75/25 | 150M | 50M |
| 50/50 | 100M | 100M |

Adopting 50/50 therefore removes 50M query presentations relative to the default. Its screen contrast did not test that tradeoff.

For a future design, either compare mixes at the actual build resource constraint or keep query exposure fixed in the build and explicitly purchase extra document exposure. Until then, describe B as an augmentation experiment and limit what its result decides.

**G-384 is allowed to challenge the diagnosis but forbidden from correcting the recipe.**

> “384 is not a candidate recipe — it is the thing the M9 diagnosis says is too narrow”

[screen_registry.json:99](/home/dylan/asymetric-dual-encoders/m10/screen_registry.json:99)

And:

> “A negative point estimate is REPORTED … it is not an action”

[screen_registry.json:193](/home/dylan/asymetric-dual-encoders/m10/screen_registry.json:193)

This is a poor design if the scientific premise is uncertain. Even a very large, reproducible advantage for G-384 cannot select it. Its exclusion follows from the diagnosis that the experiment supposedly tests.

I would retain the measurement and, in a revised selection policy, permit G-384 to win on quality and resource cost. If its diagnostic-only status stays, remove claims that this experiment “decides” the appropriate width. It provides evidence for or against a prescribed width.

**E1 measures a whole optimizer policy, not batch size under an identical exposure schedule.** Warmup is fixed at 2,000 **steps**. That is 64,000 examples for bs32 and 256,000 for bs128—1.28% versus 5.12% of the screen. [nano10.py:320](/home/dylan/asymetric-dual-encoders/m10src/nano10.py:320)

Batch 128 also gets approximately one-quarter as many AdamW updates, including weight-decay applications. These are legitimate consequences of a batch-policy comparison, but they weaken a mechanistic “small batches generalize better” interpretation and its extrapolation to a 40× longer run.

The material discrepancy is the warmup exposure and update regime, not the 64-example rounding difference the runner carefully documents. [run_arm.py:366](/home/dylan/asymetric-dual-encoders/m10src/run_arm.py:366)

**D-COV is a useful score-error surrogate, but its stated mechanism is too strong.** For independent documents drawn from the covariance population,

\[
\mathbb E\left[((s-t)^\top(d_i-d_j))^2\right]
=2(s-t)^\top\Sigma(s-t).
\]

That gives D-COV a clean interpretation: it penalizes errors in teacher/student score differences between random corpus documents. It does not specifically weight relevant-versus-near-top-irrelevant comparisons, which determine nDCG.

Also, the code says unit-trace normalization makes it comparable to the anchor at the same learning rate. [nano10.py:237](/home/dylan/asymetric-dual-encoders/m10src/nano10.py:237) In the isotropic case, however,

\[
L_{\mathrm{COV}}=\frac{1}{1024}\|s-t\|^2.
\]

AdamW may largely cancel constant gradient scaling in some conditions; clipping and epsilon mean this is not an exact optimizer equivalence. Audit the gradients and updates before interpreting a negative D-COV result as evidence against document-aware regression.

**The concrete additions I would prioritize are below.** The ranking is qualitative information gained per cost. It deliberately starts with analyses that require no new training.

| Rank | Exact addition | Cost | What it changes |
|---|---|---|---|
| **1** | **Executable decision examples** covering the three confirmation/tie counterexamples above, including E’s complete path through confirmation | No GPU; roughly 1–2 engineering hours | Establishes that the final recipe selector implements its intended decisions. Fresh-seed confirmation should read paired differences; E needs its own cost/quality rule |
| **2** | **One realized-exposure table per arm**: source × form presentations, distinct texts actually visited, repeated presentations, token-length distribution, and query/document tokens | CPU analysis of manifests/sampler draws; no teacher encoding | Corrects the interpretation of A and reprices the build using the distribution actually presented |
| **3** | **Deployment-quality bridge**: on a fixed development sample, compare exact floating-point search, exact search using the intended compressed representation, and the deployed ANN configuration for zero/nano/bge-small | No training or new teacher encodes if vectors exist; budget a few CPU hours for a bounded sample | Determines whether the quality and latency coordinates describe the same deployed systems |
| **4** | **Two bounded mechanism audits**: normalized-loss fitting of a rank-384 head using the *same 1152-d features*; D-COV versus L2 gradient/update-scale measurements on identical training batches | Small matrix work and short runs; cap at 1 GPU-hour | Separates the rank explanation from extra-layer information, and detects an optimizer-scale explanation for D-COV behavior |
| **5** | **`SYNTH-20M` versus the exact anchor recipe at 20M**, both with the same complete 20M schedule, evaluating the assembled recipe on existing development surfaces | 20M new examples if F-bge-small is an exact reusable comparator; otherwise 40M. Approximately 8.9 or 17.8 training hours at the reference rate | Tests whether combined improvements survive beyond the compressed 5M experiment. Fund this by replacing much of the existing confirmation allocation |
| **6, conditional** | **`A4-G20`**: same A4 text corpus and 5M recipe, but generated forms receive **20% total query exposure**, split equally across seven forms; the five existing forms receive 16% each. Compare descriptively with A3 and current A4 | One 5M arm, about 2.2 training hours plus evaluation; no regeneration | Distinguishes failure of a 58% generated-exposure policy from failure of generated text to help at a moderate allocation |

`A4-G20` is an illustrative fixed policy, not an estimated optimum. I would run it only if the generation decision remains consequential enough to justify another arm. I would not start a mixture-weight sweep.

The rank audit is also conditional on wanting to claim a mechanism. If the paper only reports that the tested multilayer recipe performed better, you can save that work by dropping the rank-causation claim.

The 20M comparison must start under a genuine matching schedule. Simply continuing the already-annealed 5M candidate would reproduce the schedule mismatch the project previously caught.

Because COV is observed, these additions cannot be smuggled into the original confirmatory accounting. Preserve the original registry and verdicts; any action-bearing redesign needs a disclosed amendment. The analyses can still provide useful development evidence without changing the headline partition, admitting another COV family, or moving the registered thresholds.

**The pre-registration machinery is buying real rigor at the boundaries, and considerable paperwork inside the search.**

The parts I would preserve are concrete:

- Frozen comparator identities, query-ID alignment, and paired scoring.
- Protected-data boundaries and training-data provenance.
- A fixed final candidate and final claim sequence.
- Disclosure of every trained arm, failure, and development exposure.
- Serving parity and resource-cap checks.
- Atomic evaluation transactions and preservation of their outputs.

The fixed-sequence final test is a defensible way to align the release decision with the headline partition. The conservative family-weighted surface also prevents a large dataset from silently determining the entire development objective. Neither guarantees deployment generalization, but both have a clear job.

The harmful parts are these.

**First, the registry confuses conservative selection with confirmation of scientific explanations.** Requiring a substantial lower bound to select an expensive intervention can be sensible. Reporting a failed superiority test as a null is not. Likewise, choosing a default under uncertainty is a product policy, not evidence that the default is equivalent or best.

For a future screen, separate:

- **Development decisions:** expected quality, resource cost, downside risk, and replication.
- **Paper claims:** explicitly scoped effects with appropriate uncertainty and multiplicity control.

You do not need to spend the same inferential machinery on a diagnostic arm that cannot affect the recipe and on a change that materially affects build cost.

**Second, “descriptive only” is being used where the measurement actually controls training.**

The registry says:

> “FORMS-12”: “descriptive only”

[screen_registry.json:318](/home/dylan/asymetric-dual-encoders/m10/screen_registry.json:318)

But the plateau response doubles the presentation weights of the bottom two FORMS-12 forms. [instructions-m10.md:552](/home/dylan/asymetric-dual-encoders/instructions-m10.md:552)

That is adaptive training driven by a validation set. It can be legitimate if disclosed, but it is not descriptive.

The same response depends on the dev→six forecast, while the surfaces section says:

> “forecasting only, never gating”

[instructions-m10.md:750](/home/dylan/asymetric-dual-encoders/instructions-m10.md:750)

If the forecast includes M9’s close-out results and controls whether an extra training cycle runs, those six-set results affect M10. A deterministic, prewritten branch does not make that influence disappear.

I would remove this top-up rule. The bottom two forms by teacher agreement are not necessarily the forms whose additional training has the greatest marginal value for C1b. “Worst current score” and “best next use of compute” are different quantities.

**Third, much of the validator checks that a decision is named, not that its behavior is coherent.** It verifies family-rule names and nonempty outcome entries. [screen_lock.py:83](/home/dylan/asymetric-dual-encoders/m10src/screen_lock.py:83), [screen_lock.py:132](/home/dylan/asymetric-dual-encoders/m10src/screen_lock.py:132)

That is why a document can accumulate multiple successful reviews while still carrying E’s impossible confirmation path. A handful of concrete decision examples would buy more than another layer of explanatory prose.

**Fourth, historical amendments remain mixed with operative instructions.** The mandate still contains the struck STOP, the retired anchor definition, old arm counts, and old quantiles. The JSON also contains incompatible historical count notes. The latest values can be recovered, but every reader has to reconstruct them.

The existing standing rule already supplies the right remedy:

> “One fact, one home.”

[CLAUDE.md:202](/home/dylan/asymetric-dual-encoders/CLAUDE.md:202)

Use a small current specification generated from executable fields. Keep the historical decisions and withdrawn claims in an archive with short pointers. Preserve history without making history the interface to the current experiment.

**Finally, drop the rule that every missing paper measurement requires a separate milestone.** M14 correctly forbids inventing numbers in prose, but “no new measurement” is stronger than needed. [instructions-m14.md:89](/home/dylan/asymetric-dual-encoders/instructions-m14.md:89) A labelled supplemental diagnostic with its own scope and provenance need not become another milestone bureaucracy.

**For M14, the strongest defensible paper is an empirical study of replacing the query encoder while preserving a pretrained document index.**

A defensible central claim would be:

> “We measure the retrieval quality and deployment costs of lookup-table and compact-transformer query encoders that reuse one frozen document index. Under the tested data and training policies, performance depends strongly on task distribution and the deployed retrieval configuration; the resulting gains and failures are reported on named benchmarks.”

The final numbers can strengthen that statement. A C1b pass would support the registered whole-system superiority claim on clean-4. A miss still supports a useful, reproducible account of the feasible tradeoffs under the constraints.

The teacher-selection evidence belongs in the paper, but its strongest formulation is an empirical counterexample to selecting by teacher quality—not a universal law of zero predictiveness. The M14 mandate itself gives Spearman values over **eight** and **seven** comparable rows while saying eleven teachers were measured. Those denominators must remain distinct. [instructions-m14.md:47](/home/dylan/asymetric-dual-encoders/instructions-m14.md:47)

There are several claims it should stop making:

- **“M9 was a coverage failure, not a capacity failure.”** Reaching 94% on one distribution does not establish capacity to reach that level jointly across many distributions. The M9 table supports distribution-dependent failure and a coverage hypothesis. It does not isolate the cause. [FINDINGS.md:26](/home/dylan/asymetric-dual-encoders/m9/FINDINGS.md:26)
- **“Normalized L2 cannot push a 384-wide head past 90–93%.”** The PCA argument does not establish that ceiling.
- **“A3−A2 establishes query-form coverage” or “A4−A3 measures whether synthetic data helps.”** Both are bundled training-distribution interventions.
- **“Unresolved means a measured null,” “not positive,” or a statistical tie.** These need effect estimates and uncertainty, with practical-policy labels kept separate.
- **“Clean-4 is contamination-free.”** Its useful property is no *disclosed* teacher overlap; the mandate correctly acknowledges development-informed training design.
- **“Fused systems are nearly contamination-immune.”** This claim appears explicitly in M14 and is unsupported by the cited arithmetic. [instructions-m14.md:38](/home/dylan/asymetric-dual-encoders/instructions-m14.md:38)

That last claim deserves a direct correction. The difference between all-6 and clean-4 is

\[
\overline S_6-\overline S_4
=\frac{1}{3}\left(\overline S_{\text{excluded two}}-\overline S_4\right).
\]

It measures a difference between two groups of datasets. It does not identify the effect of teacher contamination. A small difference could reflect fusion’s relative strengths across those tasks, BM25 contributions, or cancellation. Report **partition sensitivity**, not immunity.

Finally, the paper should credit the frozen-teacher lookup-table construction as prior art, as its own pyNIFE instruction requires. Its contribution must be the empirical evidence, analysis, and deployable implementation—not an architectural first claim. [instructions-m14.md:84](/home/dylan/asymetric-dual-encoders/instructions-m14.md:84)

**Two items are worth putting above another broad arm sweep.**

The first is the **deployment-quality bridge** in the additions table. M9 says the memory-constrained index was viable only under binary quantization, while the standing quality protocol uses exact search and treats Qdrant Edge as a latency exercise. [FINDINGS.md:49](/home/dylan/asymetric-dual-encoders/m9/FINDINGS.md:49), [CLAUDE.md:16](/home/dylan/asymetric-dual-encoders/CLAUDE.md:16)

That creates a risk of plotting full-precision quality against compressed-index latency. This is separate from the already-actioned missing cost measurements: **even perfectly measured costs are insufficient if the quality coordinate belongs to another retrieval configuration.** Query encoders can have different sensitivity to the same document compression. Measure that interaction.

The second is a cheap test of whether the **pair** provides useful complementarity: an **oracle escalation curve on existing development data**. Using per-query zero/nano scores, calculate the best achievable quality if only a fraction of queries can use nano. Compare it with random allocation and always using either encoder. This is explicitly an optimistic bound using relevance judgments, not a deployable router.

If even the oracle curve is weak, do not spend effort learning a router. If it is strong, you have identified a concrete potential benefit of maintaining both query paths on the same index. The incremental cost is score generation where missing and a small analysis, not another student build.

There is also one urgent preservation task already correctly identified in STATUS: ensure both pending six-set transactions emit aligned query IDs and per-query scores before M9’s close-out. That is more time-sensitive than any new hypothesis because the paired M9→M10 analysis may otherwise become irrecoverable. [STATUS.md:135](/home/dylan/asymetric-dual-encoders/m10/STATUS.md:135)

My recommendation is to **repair the decision rules, preserve the running screen’s record, and spend the next substantial compute allocation on validating the assembled recipe at a more relevant duration**. That would give the build a stronger evidential basis than confirming isolated short-run wins under the current machinery.
