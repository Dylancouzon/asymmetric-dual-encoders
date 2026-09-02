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
- 2026-09-01 — Compute (decision 2 reframed): "M10 won't be done on a 3080. M10 will be done on a GPU budget, if allowed, or not at all." Box withdrawn as an execution target; budget request expected $400–715 with generation on the GPU or $465–895 hosted, ceiling $1,000 (PLANNING §6). **Approval pending; no GPU stage runs before it.**

## §4 Dev-reuse log

| date | surface | raw score reads | artifact |
|---|---|---|---|
| 2026-09-01 | cqadup-programmers, cqadup-physics (Mac diagnostics) | 43 + 43 | `results/m10_rank_probe_mac.json`, `results/m10_head_width_probe_mac.json` |

## §5 Amendments and withdrawn claims (never compressed away)

