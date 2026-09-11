# M17 planning review dispositions — 2026-09-11

Two independent Luna planning reviews, with named-file briefs and protected-read exclusions:
[loss/design](../research/m17-plan-review-loss-2026-09-11.md) and
[data/budget](../research/m17-plan-review-data-2026-09-11.md). Their access logs were inspected:
only the named design, historical guidance and permitted diagnostic artifacts are listed; no
reserved or six-set/LoTTE payload reads. These reviews do not certify an unwritten executor.

| Finding | Parent disposition and exit |
|---|---|
| Loss P1 / data F4: query-only candidates | Clarified in registry: separate labeled/query-only mixtures, `has_label`, nullable positive ID and deterministic dedup/fill; teacher hits never become labels. One shared cache schema with complete identities. |
| Loss P2: exact objective | Adopted: normalized-query definitions, shared document cache/temperature, query-teacher cosine target, batch-mean KL direction/reduction and effective-row anchor now explicit in registry. |
| Loss P3 / data F1: unpriced real workload | Already an explicit execution dependency; strengthened with phase stop/recovery rules. Reject selecting from an accidentally incomplete matrix. No synthetic throughput extrapolation or unrecorded post-lock dose/seed cuts. Real timing remains owed. |
| Loss P4: index identity | Adopted build-time equality of bundle document metadata with M7's encoder spec. Declined binding a general encoder to one production corpus hash or adding loader arguments: that would change the promised API and confuse document-space identity with corpus identity. Bank hashes remain training/evaluation provenance. |
| Data F2: broad support is conditional | Support manifest is the first data exit, with per-domain deduplicated counts and fewer rows when unsupported. No claim that current fixtures establish broad coverage. Actual manifest remains owed. |
| Data F3: frozen v1 / DBSF | Existing plan already requires pre-update folded-int8 v1 parity and fixed DBSF comparisons. Local source table, training checkpoint and lineage hashes were verified against M7's freeze. No v1/M13 assets changed. |
| Data F5: implementation breadth | Bounded to one driver/cache schema and existing primitives; optional polish can be deferred, mandatory missing evaluation/parity yields an incomplete candidate. No framework or M13 change. |
| Parent: positive-bank memory | Clarified: the selected labeled queries' known positives must fit inside the bank cap, rather than adding an unbounded extra positive bank. |

**Owner:** M17 implementing session. **Execution exit:** real inputs/panel, measured allocation,
ratified prospective protocol, working driver and two independent implementation reviews.
Planning fixes do not remove those dependencies.

## A1 scope update

Dylan's subsequent adoption of alias consistency, checkpoint averaging and int8 resident
loading is recorded in `LEDGER.md` A1. The earlier Luna reviews covered the preceding draft;
they are not represented as reviews of this amendment or its future implementation. A1's
planning check covers the matched alias views, fixed forms/read counts, budget arithmetic,
audit isolation, links and preservation of the original results. Implementation reviews must
cover the pair sampler/loss, snapshot/resume/export path and selected-row dequantization too.

## Codex Astra whole-plan review after A2 — 2026-09-11

Brief: [m17-astra-plan-brief](../research/m17-astra-plan-brief-2026-09-11.md). Log with the full
report: `research/m17-astra-plan-review-2026-09-11.log`. Read-only, xhigh effort. Its access
log lists only the eighteen named files; no reserved, six-set or LoTTE payload. Astra confirmed
the arithmetic and found no violation of the cap, frozen index or M13 isolation, and would keep
A2 non-executable. Dispositions:

| Finding | Disposition and exit |
|---|---|
| P1 decision rules incomplete | Adopted: `decision_protocol` block pins the six dev components, the technical metric (`cqadup-programmers` dense nDCG@10), eligibility, control-map walk, tie, seed-disagreement, no-survivor and averaging-eligibility rules, and int8 artifact binding. |
| P1 panel could still decide via vetoes | Adopted: ambiguous-sense veto and coverage-slice alternative removed from every predicate; panel and alias test are report-only. |
| P1 sealed panel not necessarily unseen by warm start | Adopted: ancestor decontamination against all M7 training manifests through approved fingerprint interfaces; exposure-unknown labeling where ancestry is unavailable. |
| P1 rehearsal/V0 observation boundary | Adopted: rehearsal scores synthetic fixtures only; protocol, vocabulary hash, tokenizer hash, seeds and cache identity committed before V0 or any real read. |
| P1 Kubernetes evidence absent | Adopted: proposal row added to `research/m7-data-licensing.md` with the required checklist; still not admitted for download. |
| P2 bucket repetition hidden by "2.6 passes" | Adopted: per-bucket populations, four-pass ceiling, share-then-steps shrink rule, processed-view basis for the 10% cap, structural bulk alias pairs with spot checks. |
| P2 teacher entropy wrong stop criterion | Adopted: replaced with warm-start student-teacher KL rule; surviving matrix defined (C, V). Entropy kept descriptive. |
| P2 candidate mixture underdefined | Adopted: `candidate_construction` block: positive choice, unique-hit quotas, backfill, tie order, RNG derivation, bank sampling, provenance counts. |
| P2 alias weight 1.6x per view | Adopted: alias term batch-normalized (2/B times pair sum); pre-lock teacher-pair cosine and gradient-share diagnostic. |
| P2 count-weighted initialization | Adopted: new row = sum sqrt(c_j) R_j over the isolated term's pieces, special tokens excluded. |
| P2 discretionary vocabulary rules | Adopted: `vocabulary_ranking` block: residual, support weight, ordering, domain assignment, abbreviation threshold, breadth condition. `k8s` exception kept. |
| P2 anchor inertness not established | Adopted: described as coefficient continuity; optimizer pinned to M7's Adam defaults; anchor update share measured in smoke. |
| P2 preparation underpriced, pair verification cost | Adopted in part: bulk structural pairs remove the per-pair judging cost; measured preparation costs at two sizes added to execution-entry requirements. Real timing still owed. |
| P3 loader decided by latency noise | Adopted: 200 batches x 5 processes, nearest-rank quantile, natural-reuse fixtures, Linux labeling; Mac gate separate. |

**Owner:** M17 implementing session. **Exit:** unchanged; the executor and its two independent
implementation reviews remain owed. This planning review does not certify an unwritten executor.
