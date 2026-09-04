# M10 ledger — protocol, rulings, and the numbers a rule reads

Skeleton committed 2026-09-01 (Codex pass 5). Every section is filled by the GPU session, pushed before the
step it governs, and never edited after that step's output exists. Numbers live in the JSON the
row points at; this file records the decision, the number a rule reads, and the pointer.

## §0 Screen lock — fill at M10.0-e, before any arm

- COV resolution number (mandate §Surfaces; power disclosure only — A4's sizing struck by the Codex pass): the measured distance for e5-small-v2 vs gte-small at 0.025/14, beside the fixed MDE 0.0056; no direction, no verdict, no effect on α.
- Arms (fifteen) **in the amendment-A6 order A → F → G → B → E → C → D**: A1 A2 A3-harvested A4-full (= anchor) · F-MiniLM-L6 (20M, read at 5/10/20M) F-MiniLM-L12 (5M elimination probe; extended to 20M iff within the MDE of the better 5M reading) with the anchor extended to 20M · G-384 G-1536 G-MLP (per-token residual `W_lin·x + W₂·GELU(W₁·x+b₁)`, W₁ 1152→192, B3) · B 100/0 B 50/50 (B 75/25 = anchor) · E-bs128 · C-M9init (skipped and reported skipped if F does not select bge-small) · D-NORM D-COV: per arm the data manifest hash, mix, init, objective, batch, student, feature layers, dose in examples and tokens (5M screen dose), seed.
- Day-one rate benchmark: stella docs/s; examples/s at batch 32 on the 75/25 mix, the 50/50 mix and MiniLM-L6; generation requests/s per form; the billed $/h; the re-derived PLANNING §6.
- Allocation under the $1,000 ceiling: every mandatory line at measured rates; `max_extension_cycles`; m_k's formula and evaluation hashes; the billed-spend source; which stages run on the box and which are rented (amendment A7). Decision 8 is withdrawn (A5), so no seed-1 boolean is fixed.
- A2, A3 and A4 post-screen unique-text counts (identical) and corpus hashes.
- ~~τ entropy table~~ — struck with the ranking-aware class (amendment A1).
- Fourteen contrasts (F carries two; the count is fixed whether or not L12 is extended), the 0.025/14 bound, **MDE 0.0056 fixed** — A4's `max(0.0056, distance)` and its two-step remedy were struck by the Codex pass of 2026-09-04. Rank-stability rule; family A's three-outcome rule verbatim on A3−A2 and the A4−A3 drop rule; the `arxiv-title` secondary rule (harvested scientific forms in or out of the build).
- Confirmation design: which decisions (**at most two**, amendment A5), seeds, the margin and seed-range definitions; the replication seed pair on the selected recipe.
- COV macro formula (families, slice averaging, weights); the `arxiv-title` **secondary** surface (B4, not in the macro: 100K held-out papers drawn by id-without-version with seed 0 at M10.0-d, every version excluded from training, in the protected index; teacher denominator reported; its one action); DEV-6-once evaluation rule.
- Release rule under four conjuncts (decision 11) and decision 12's ruling on PubMed-family text, copied verbatim.
- Outcome → action map for every family; the synthesized selected-recipe arm; LoTTE read #1 manifest and veto rule.

## §1 Data manifest — fill at M10.1

- Generator (repo, revision, bf16; hosted provider and served revision if the fallback fired), sampling parameters, seed rule, retry/dedup policy; per-form smoke results (contract %, on-form %), approver, prompt revisions (≤2 per form, each recorded here with the diff).
- Seed sources and revisions; seed pre-filter removals; per-form quotas realized, per source (harvested vs generated, amendment A2).
- **§Harvest:** per extraction rule, the rule text, source corpus, yield, and the form it feeds; the span-exclusion form of the seed-passage screen.
- **A8 quality gates:** per-form near-duplicate rate and mean pairwise cosine with any quota cut taken; the stella-space distribution-overlap table against the MS MARCO dev sample (disclosed diagnostic, no action).
- PAQ release files and hashes; build sample (1.0M) and A2 sample; attribution.
- Decontamination removals per screen, per form, per COV component; FORMS-12 hold-out seed ids.
- Teacher-target cache keys. ~~bank, mining method, recall@64 audit~~ — struck with the ranking-aware class (amendment A1).
- `results/m10_data_manifest.json` sha256.

## §2 COV admission records — fill at M10.0-d, one row per component

| component | family | repo · revision | licence at primary source (URL) | corpus / queries / qrels / metric | corpus-level contamination check | fingerprint screen vs six + reserved | verdict |
|---|---|---|---|---|---|---|---|

## §3 Owner rulings and decisions

Copied from `instructions-m10.md` §Owner decisions as each is taken, with date and wording.

- 2026-09-01 — Student cap: "109M is not an option. This isn't low compute anymore. 33M was already in the upper bound of what I think is acceptable." 35M hard.
- 2026-09-01 — FineWeb: out of M10 in every role (delegated ruling; `m10/EXPLORED.md`).
- 2026-09-01 — Compute (decision 2 reframed): "M10 won't be done on a 3080. M10 will be done on a GPU budget, if allowed, or not at all." Box withdrawn as an execution target; budget request expected $400–715 with generation on the GPU or $465–895 hosted, ceiling $1,000 (PLANNING §6).
- 2026-09-04 — **Budget VALIDATED** by Dylan, together with "do a full review yourself … is this the most efficient?", "I'm not sure why we need to generate synthetic data?", "isn't synthetic data lower quality? I wanna make sure we have the best chances on our side", "don't over-engineer", "keeping the same teacher (Stella) is the goal", and "make your changes to the plan, then have Fable do an adversarial review". Re-priced on measured rates to ≈$110–280 hybrid; ceiling $1,000 unchanged. Amendments **A1–A8** adopted (`instructions-m10.md` §Amendment 2026-09-04, decision 10); cuts and their reopening conditions in `m10/EXPLORED.md`; evidence PLANNING §11–12.
- 2026-09-04 — Three days of box compute offered before the cloud instance ("I will be leaving for 3 days tonight"). Used for the no-approval-needed stages only; generation cannot start in that window (Dylan is the smoke approver and Qwen3-8B bf16 does not fit 10 GB).

## §4 Dev-reuse log

| date | surface | raw score reads | artifact |
|---|---|---|---|
| 2026-09-01 | cqadup-programmers, cqadup-physics (Mac diagnostics) | 43 + 43 | `results/m10_rank_probe_mac.json`, `results/m10_head_width_probe_mac.json` |
| 2026-09-04 | frozen comparator rows of `results/perquery.json` (bge-small, leaf-ir-asym, lr-dense-pertask, opensearch, bm25) on all-6 and clean-4 | comparator-only, no nano existed | amendment A3's clean-4 bars 0.5046 / 0.5233; not a dev-surface read |

## §5 Amendments and withdrawn claims (never compressed away)

**2026-09-04 plan review, A1–A8** — what changed, in one line each; reasoning in
`instructions-m10.md` §Amendment 2026-09-04, evidence in PLANNING §11–12.

| id | change |
|---|---|
| A1 | family D cut to one arm, LEAF's ‖e‖₂; candidate bank, mining, HNSW fallback, τ rule, D-NCE spec and seed-rank field deleted |
| A2 | generation 3.0M → ≈1.0M and confined to the six non-harvestable forms; ≈1.5M harvested real query-like text added as arm A3 |
| A3 | C1/C2 registered on clean-4 as well as avg-6 — four conjuncts under Holm inside the unchanged 0.025 family alpha; clean-4 bars 0.5046 / 0.5233 |
| A4 | ~~the COV resolution number sizes the screen: MDE = max(0.0056, distance), two-step remedy~~ — **struck the same day by the Codex pass**; the number is a power disclosure, MDE 0.0056 and 0.025/14 fixed |
| A5 | decision 8 (full-dose seed 1) withdrawn; confirmations capped at two decisions |
| A6 | family F runs second; every later family screens on its winner. Order A → F → G → B → E → C → D |
| A7 | the box is an execution target again for everything but generation, on measured rates |
| A8 | two pre-training data quality gates: per-form diversity (with an action) and stella-space distribution overlap vs MS MARCO dev (disclosed diagnostic, no action) |

**2026-09-04b feasibility review, B1–B6** (`instructions-m10.md` §Amendment 2026-09-04b;
`research/m10-feasibility-review-2026-09-04.md`): B1 feasibility statement and per-dataset arithmetic
registered (C1b harder than C2a; C2b out of reach on the literature) · B2 four-conjunct claim table,
release rule = decision 11 · B3 G-768 → G-MLP per-token nonlinear head (parity proven, 34.48M) ·
B4 constructed scientific surface · B5 DEV-6 pre-screen dropped · B6 asset-size line. **Codex pass
on B1–B6** (`research/m10-codex-feasibility-2026-09-04.md`, 4/8/4, all actioned): 0.025 quantile in
the proxies and the final bound; G-MLP residual form + training/export wrappers; F's L12 probe rule
and the fourteenth contrast; decision 11 default = release needs C1b; `arxiv-title` secondary, not a
family; A4 struck; MedlinePlus / CDC / ClinicalTrials.gov out of M10. **Decision
12 (CUREv1 as a validation-only diagnostic) is open** — it reopens M7's "excluded from COV" clause
for selection surfaces and waits for Dylan. **Withdrawn by the same review:** PubMed titles /
PubMedQA as training text — no affirmative grant on PubMed abstracts (NLM disclaims copyright,
publishers may hold it), so the licence rule fails first. Harvest sources verified clean: arXiv
metadata (CC0), MedlinePlus government-authored topics and CDC (US public domain), ClinicalTrials.gov
pending its terms clause; DailyMed, OpenAlex, bioRxiv/medRxiv, WHO, Cochrane out.

**Withdrawn in the same review, kept so it is not re-proposed:** dropping family F to anchor on
MiniLM-L6. It would have killed family C as well (M9's candidate is a bge-small student) and arXiv
2306.11550's depth curve disagrees with LEAF's 6-layer success. F and C stay; A6 fixes the ordering
problem instead.

**Corrected in the same review:** `instructions-m16.md`'s "if only one thing here ever runs, run
B3" — B3 (cosine-space distillation) is a no-op on a normalized output, closed by algebra. The
pyNIFE retention gap must be attributed to B4/B5.

