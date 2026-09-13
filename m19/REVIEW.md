# M19 review dispositions

Astra's own scope, implementation, query, serving and reconciliation records live in
[`reviews/`](reviews/). This file carries owner-facing review of the milestone as a whole.

## Judgment-stop assessment — 2026-09-13 (Claude, session-external)

Scope: was `ENCODER_INCONCLUSIVE` the right stop, and is a bounded continuation warranted?
Read-only. Files inspected: `m19/STATUS.md`, `m19/FINDINGS.md`,
`m19/reviews/final-reconciliation-astra-2026-09-13.md`,
`results/m19_development_judgment_incomplete.json`, `results/m19_final_system_manifest.json`,
`results/m19_algebra_gates.json`, `results/m19_serving_benchmark_10000.json`, and
`instructions-m19.md`. No judgment content, no query text, no confirmation path, no protected or
spent surface. This review reads no quality number because none exists.

Continues the M18 post-run diagnosis in [`../m18/REVIEW.md`](../m18/REVIEW.md).

### The stop was procedurally correct

A pre-registered gate failed and the run stopped rather than resolving 60 items while looking at
the blocker. That is what pre-registration exists to prevent, and it is the first milestone in this
sequence to refuse a measurement instead of shipping a weak one. M17 proceeded past an undiagnosed
inversion; M18 scored a candidate against a task its own ceiling showed to be misposed. M19 did
neither. Nothing below should be read as criticism of that instinct.

One sharpening that strengthens the case for the stop: the 60 unresolved items are *concealed
repeats* drawn from the audit set, whose purpose is to establish whether the primary judge can be
trusted at all. An auditor abstaining on 4.4% of blind re-checks is evidence about the reliability
of the 1,316 labels underneath, not only about those 60 rows.

### J1 — the binary-only requirement was a design defect, not a discovery

The protocol admitted no abstention outcome. A judgment set that cannot represent "undecidable"
must halt the first time a judge is honest, and a 4.4% abstention rate is unremarkable in relevance
judging; independent human assessors disagree by considerably more. The gate therefore did not
detect a problem with the candidate, the corpus or the pool. It detected a missing category in its
own schema.

This matters because the registered decision is a *comparison* of released v1 against T0 over one
shared pool. Ambiguous items are served to both systems. Unless the unresolved set concentrates in
one system's results, a comparison tolerates label ambiguity far better than any absolute score
does.

**Proposed exit.** Any future judgment protocol in this repository registers an explicit `unsure`
label, a rule for scoring it, and an abstention rate above which the pool rather than the run is
considered defective. Abstention becomes recorded evidence, not a halt condition.

### J2 — a bounded continuation is available without post-hoc discretion

Discarding 1,376 completed judgments over 60 undecidable ones is the expensive choice. The standard
remedy requires no guessing, because it computes both extremes rather than choosing between them:

1. **Cluster check first.** Determine whether the 60 unresolved items concentrate by route or by
   candidate. Only `unresolved_repeat_ids_sha256` reached the repository; the identifiers are on the
   execution host. If they concentrate in one system's results, the comparison is genuinely
   compromised and the stop stands as final.
2. **Amend before computing.** If they are dispersed, register a dated amendment fixing unjudged
   items as non-relevant for the primary metric and additionally reporting the optimistic bound in
   which all 60 are relevant. Both rules are fixed before any metric is produced.
3. **Read the development numbers once, under both bounds.** Agreement in direction across the two
   bounds supports a development conclusion. Bounds that straddle mean the ambiguity decides the
   answer, no conclusion is available, and `ENCODER_INCONCLUSIVE` is then correct on its merits
   rather than by schema accident.

This is a sensitivity analysis, not a relaxation: it substitutes a pre-registered pair of extremes
for a discretionary judgment. It needs an owner ruling, changes no locked margin, creates no new
qrels by inference, and leaves the confirmation split sealed and unclaimed in every branch.

**Recommended framing.** Scope this as a bounded continuation of M19, not a new milestone and not a
re-run. The pool, packets and primary labels already exist and are hash-bound.

### J3 — the passages shown to the judge reproduce a defect M19 had already fixed

`FINDINGS.md` records the reason the 60 items could not be decided: the displayed passages lacked
material context. M19's own retrieval contribution is artifact collapse, which exists because a
passage is the wrong unit to show a person. The judging surface kept the passage unit after the
retrieval surface abandoned it.

**Proposed exit.** Judgment packets present the artifact with the matched passage marked inside it,
matching the unit the metric scores and the unit a reviewer can actually decide. Expect this alone
to resolve a large share of the abstentions.

### J4 — governance cost remains the dominant execution risk

53 commits over 4h21m reached the judgment boundary, with roughly four fifths spent locking,
freezing, authenticating and recording review outcomes. Three review cycles each found new issues in
code written to close the previous cycle's issues, and one loop was escaped only by the agent
declaring it non-binding. None of that produced or prevented a quality result.

**Proposed exit.** Future instruction files carry an explicit governance budget: one implementation
review plus one re-review, P2 findings recorded rather than remediated, and a fixed cap on lock and
freeze artifacts. Protected-data rules in `CLAUDE.md` remain absolute and are not what this exit
bounds.

### Free check that precedes all of the above

Both candidate bundles are built and gated. A twelve-term side-by-side of released v1 against
T0-teacher needs no qrels, no judgments and no amendment, and its evidentiary status is the same
report-only impression as M18's spent 21-term probe. Run it before deciding how much the metric
needs to settle. M18's `k8s` lists are the precedent: released v1 returned `release v0.8.0` and
`v0.8.2`, the trained table returned the Kubernetes persistence issue at rank two.

Note for interpretation, established and not in dispute: the exact-token change alone does nothing.
M18's untrained `V0` arm added the same tokens under count-weighted composition and was equal to v1
on observed rankings. The row's direction carries the entire effect. Additive construction over
frozen inherited rows also bounds the blast radius exactly — `results/m19_algebra_gates.json`
records `no_match_encoder_max_abs` of `0.0` and the 10,000-query benchmark records
`no_match_ranking_parity` true, so a query containing no roster term is bit-identical to released
v1.

### Not in scope

Term selection, the row algebra, quantization, serving parity, the inheritance guard and the
reconciliation record. Those are covered in `reviews/` and this review found no reason to revisit
them. Nothing here reopens the M19 decision, relaxes a registered gate, creates a qrel, or touches
the sealed confirmation split; J2 requires an owner ruling before any of it is acted on.

## Owner disposition — bounded sensitivity completed 2026-09-13

The owner authorized a development-only continuation. The executed method strengthened J2: because
unknown artifacts occur asymmetrically across routes, assigning all unknowns zero and then all one
does not bound a system contrast. The committed method instead enumerated every assignment per query
and aggregated exact extrema under the frozen term macro.

The result is `LABEL_SENSITIVE`, not a recovered encoder decision. Short-context Precision@10 spans
`+0.016667` to `+0.075000` around the `+0.03` gate; term wins span 6 to 10 around the required strict
majority. The original `ENCODER_INCONCLUSIVE` therefore remains substantively correct. No qrels,
enriched rejudgment or confirmation read followed. J1 and J4 remain recommendations for future
protocols; J3 is not worth pursuing for this closed internal proof of concept.
