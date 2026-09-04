# M10 ledger — protocol, rulings, and the numbers a rule reads

Skeleton committed 2026-09-01 (Codex pass 5). Every section is filled by the GPU session, pushed before the
step it governs, and never edited after that step's output exists. Numbers live in the JSON the
row points at; this file records the decision, the number a rule reads, and the pointer.

## §0 Screen lock — fill at M10.0-e, before any arm

- COV resolution number (mandate §Surfaces; descriptive): the measured distance for e5-small-v2 vs gte-small, beside the MDE 0.0056; no direction, no verdict.
- Arms (fourteen) in order A1 A2 A3 · B 100/0 B 50/50 (B 75/25 = anchor) · C-M9init · D-KL3 D-KL1 D-NCE · E-bs128 · F-MiniLM · G-384 G-768 G-1536: per arm the data manifest hash, mix, init, objective, batch, student, feature layers, dose in examples and tokens (5M screen dose), seed.
- Day-one rate benchmark: stella docs/s; examples/s at batch 32 on the 75/25 mix, the 50/50 mix and MiniLM-L6; generation requests/s per form; the billed $/h; the re-derived PLANNING §6.
- Allocation under the $1,000 ceiling: every mandatory line at measured rates; decision 8 boolean (≥ 100 GPU-hours remaining), seed-1 data and model seeds and six-set row labels; `max_extension_cycles`; m_k's formula and evaluation hashes; the billed-spend source.
- A2 and A3 post-screen unique-text counts (identical) and corpus hashes.
- τ: the entropy table over 10,000 queries (seed 0, equal thirds) and the chosen value.
- Sixteen contrasts, the 0.025/16 bound, MDE 0.0056, rank-stability rule; family A's three-outcome rule verbatim; the literal D-NCE loss and the τ reused in it.
- Confirmation design: which decisions, seeds, the margin and seed-range definitions.
- COV macro formula (families, slice averaging, weights); DEV-6-once evaluation rule.
- Outcome → action map for every family; the synthesized selected-recipe arm; LoTTE read #1 manifest and veto rule.

## §1 Data manifest — fill at M10.1

- Generator (repo, revision, bf16; hosted provider and served revision if the fallback fired), sampling parameters, seed rule, retry/dedup policy; per-form smoke results (contract %, on-form %), approver, prompt revisions (≤2 per form, each recorded here with the diff); the seed-rank field present for every synthetic query.
- Seed sources and revisions; seed pre-filter removals; per-form quotas realized.
- PAQ release files and hashes; build sample (1.0M) and A2 sample; attribution.
- Decontamination removals per screen, per form, per COV component; FORMS-12 hold-out seed ids.
- Teacher-target cache keys; bank (1M, seed 0); mining method and, if HNSW, the recall@64 audit.
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
| A4 | the COV resolution number is measured first and sizes the screen: MDE = max(0.0056, measured distance at 0.025/11) |
| A5 | decision 8 (full-dose seed 1) withdrawn; confirmations capped at two decisions |
| A6 | family F runs second; every later family screens on its winner. Order A → F → G → B → E → C → D |
| A7 | the box is an execution target again for everything but generation, on measured rates |
| A8 | two pre-training data quality gates: per-form diversity (with an action) and stella-space distribution overlap vs MS MARCO dev (disclosed diagnostic, no action) |

**Withdrawn in the same review, kept so it is not re-proposed:** dropping family F to anchor on
MiniLM-L6. It would have killed family C as well (M9's candidate is a bge-small student) and arXiv
2306.11550's depth curve disagrees with LEAF's 6-layer success. F and C stay; A6 fixes the ordering
problem instead.

**Corrected in the same review:** `instructions-m16.md`'s "if only one thing here ever runs, run
B3" — B3 (cosine-space distillation) is a no-op on a normalized output, closed by algebra. The
pyNIFE retention gap must be attributed to B4/B5.

