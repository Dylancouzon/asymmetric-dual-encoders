# M10 ledger — protocol, rulings, and the numbers a rule reads

Skeleton committed 2026-09-01 (Codex pass 5). Every section is filled on the box, pushed before the
step it governs, and never edited after that step's output exists. Numbers live in the JSON the
row points at; this file records the decision, the number a rule reads, and the pointer.

## §0 Screen lock — fill at M10.0-e, before any arm

- Arms (eleven) in order A1 A2 A3 · B 100/0 B 50/50 (B 75/25 = anchor) · C-M9init · D-KL · E-bs128 · F-MiniLM · G-384 G-768: per arm the data manifest hash, mix, init, objective, batch, student, feature layers, dose in examples and tokens, seed.
- A2 and A3 post-screen unique-text counts (identical) and corpus hashes.
- τ: the entropy table over 10,000 queries (seed 0, equal thirds) and the chosen value.
- Thirteen contrasts, the 0.025/13 bound, MDE 0.0056, rank-stability rule; family A's three-outcome rule verbatim.
- Confirmation design: which decisions, seeds, the margin and seed-range definitions.
- COV macro formula (families, slice averaging, weights); DEV-6-once evaluation rule.
- Outcome → action map for every family; the synthesized selected-recipe arm; LoTTE read #1 manifest and veto rule.

## §1 Data manifest — fill at M10.1

- Generator and quantized artifact (repo, revision), sampling parameters, seed rule, retry/dedup policy; per-form smoke results (contract %, on-form %), approver, prompt revisions (≤2 per form, each recorded here with the diff).
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

## §4 Dev-reuse log

| date | surface | raw score reads | artifact |
|---|---|---|---|
| 2026-09-01 | cqadup-programmers, cqadup-physics (Mac diagnostics) | 43 + 43 | `results/m10_rank_probe_mac.json`, `results/m10_head_width_probe_mac.json` |

## §5 Amendments and withdrawn claims (never compressed away)

