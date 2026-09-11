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
