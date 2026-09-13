# M18 review dispositions

## Use-case alignment review — 2026-09-13 (Claude, session-external)

Scope: does the M18 design, as committed at `e60deab`, measure the failure Andrey actually
reported? Read-only. Files inspected: `instructions-m18.md`, `m18/STATUS.md`, `m18/LEDGER.md`,
`m18/FINDINGS.md`, `m18src/protocol.py`, `m18src/vocab.py`,
`results/m18_protocol_manifest.json`, plus the Slack source record summarized in
[`research/andrey-use-case-profile-2026-09-12.md`](../research/andrey-use-case-profile-2026-09-12.md).
No development or confirmation query text was read: the M18 work directory is not present on this
host. No protected surface, no M17 record, no result JSON beyond the protocol manifest.

This is a design review against the stated use case. It does not certify the corpus, the qrel
predicates, the driver or the training path, all of which are covered by the Astra and Sol
reviews in `LEDGER.md`.

### Confirmed sound, do not revisit on this review's account

| Design choice | Assessment |
|---|---|
| Qdrant repo + issues + PRs as the whole first corpus | Correct. This is literally the corpus Andrey indexed for his lgtm@10 experiment. Dropping Stack Exchange removes a licence dependency that only ever existed to proxy this corpus. |
| Freeze inherited v1 rows, train only new exact-token rows | The strongest choice in the plan. It makes M17's "loss falls, retrieval degrades" outcome structurally unavailable, and it is one gradient mask rather than a new recipe. |
| `V0-Q` as a first-class baseline that may be selected | Right. The untrained count-weighted extension is a credible winner and the plan does not assume otherwise. |
| T2 defined as masking `^##[0-9]+$` only, as a falsifier | An improvement on the external suggestion that prompted it. The observation that suppressing `##3` does not synthesize an `s3` representation, and would collapse `S3`/`S4`, is correct; exact tokens before suppression is the right order. |
| Fragmentation gate in `m18src/vocab.py:76` | Selects on "carries a digit or is vowel-free", so `k8s`, `s3`, `tls` are caught generatively rather than from a hand list. This closes the main vocabulary-coverage risk. |
| Hybrid DBSF primary with dense reported separately | Correct, and consistent with BM25 properly owning exact literals. |
| Step-500 diagnostic with a hard halt and a parity report | Proportionate response to the M17 failure. |

### P1 — the evaluation set does not contain the query shape that motivated M18

Every development query originates as authored prose. `development_source_kinds` in
`results/m18_protocol_manifest.json` resolves to issue openings, pull-request openings and review
comments across all five strata, with `concept_howto` now entirely issue openings after the v4
rebuild.

Strata are assigned by `_stratum()` matching cues against that prose, so a query is
`exact_version_numeric` because its issue body contains a version number or a backticked token,
not because the query *is* an identifier. `EXACT_CUES` matches nearly any issue containing a
backtick or an ALLCAPS run, which is the likely reason that stratum reports 312 available families
against 56 to 73 elsewhere. The stratum is over-populated by classification breadth, not by being
identifier-query shaped.

The reported defect is short-query specific. Zero pools one fixed row per piece with no
interaction, so `s3` is the mean of a generic `s` row and a numeric-context `##3` row with nothing
else in the query to dilute them. A multi-sentence issue body contains many ordinary word rows
that dominate the pooled vector, which is precisely where the defect is weakest.

Consequence: a candidate can satisfy every eligibility gate in `instructions-m18.md` while leaving
Andrey's queries unchanged, and the run can report `ENCODER_NO_IMPROVEMENT` for a reason that is
an artifact of the query distribution rather than a property of the added rows.

**Unverified component.** Query length was inferred from source kinds, not measured. Before acting
on this finding, confirm on the execution host with a median word count per stratum over
`development_queries.jsonl`. If the realized median is already short, this finding is void.

**Proposed exit.** While full Stella encoding is paused for the Astra re-audit, and before the
protocol is sealed:

1. Promote the owner-supplied question slice from optional to required-if-offered. The current
   instruction not to wait is right about velocity and wrong about leverage: Andrey is the judge,
   is engaged, and 30 to 50 real short queries is a single Slack request. Admit them as a sixth
   stratum carrying its own eligibility gate, not as a descriptive slice.
2. Add a short-form identifier stratum mined from the corpus: take terms that already pass the
   `vocab.py` fragmentation and support gates, use the bare term as the query, and record what
   structural evidence can and cannot supply a qrel for it. Where deterministic qrels are not
   defensible for a bare term, keep the stratum report-only rather than manufacturing labels.

### P2 — no artifact reproduces the acceptance test that will actually be applied

The required-artifact list produces machine-readable baselines, stratum results, audit results and
an int8 bundle. None of these is the comparison Andrey will make. His stated metric is LGTM@10:
index the corpus, type a query, look at the top ten.

**Proposed exit.** Add one artifact: a side-by-side top-10 for released Zero v1 against the
selected candidate over a fixed list of bare identifiers (`s3`, `k8s`, `hnsw`, `wal`, `mmap`,
`grpc`, and the terms the vocabulary step actually admitted). It needs no qrels, costs one
retrieval pass per model, and is the evidence that decides acceptance. Report it alongside the
metric result, never in place of it.

### P3 — alias coverage for the target class may be thin

`ALIAS_RE` matches an `Expansion (ABBR)` construction in running text. The manifest reports 466
alias views of 901 training views, which is good coverage of short forms in general. The specific
terms at issue are unlikely to appear in that construction inside this repository: `Simple Storage
Service (S3)` is not a sentence the Qdrant repo writes.

**Proposed exit.** Before the registry locks, report alias-pool support restricted to the
digit-bearing and vowel-free terms admitted by `vocab.py`. If support is near zero for that slice,
record it as a known limitation of the alias-consistency term rather than discovering it after the
screen.

### Observation, no action requested

The qrel progression v1 through v4 in a single day, with training views moving 1,594 to 1,043 to
901 and structural candidates 921 to 733, reads as appropriate tightening rather than instability;
each step is justified in `LEDGER.md` and the leakage intersections stayed at zero throughout. The
residual risk is only that successive rules optimize for what a deterministic predicate can
defend, which is not identical to what a maintainer would call a correct top-10. P1's proposed
exit is also the cheapest mitigation for that risk, since owner-supplied queries are judged by the
owner rather than by a predicate.

### Not in scope of this review

Corpus provenance, pagination receipts, deduplication, split leakage, the v4 predicates, driver
correctness, resume/export parity and the training recipe. Those remain with the existing reviews
and their recorded exits. Nothing in this review relaxes an eligibility rule, and none of its
proposals may be adopted after observing endpoint quality.
