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

## Post-run diagnosis — 2026-09-13 (Claude, session-external)

Scope: why the run ended `ENCODER_NO_IMPROVEMENT` while the Stella ceiling was also very low.
Read-only. Files inspected: `m18/FINDINGS.md`, `results/m18_development_baselines.json`,
`results/m18_confirmation.json`, `results/m18_bare_term_probe.json`,
`results/m18_encoder_decision.json`. No protected surface, no sealed content beyond the already
published confirmation aggregates, no source mutation.

The prospective P1 in the section above was not avoided. It is now the primary finding, with
direct evidence rather than an inference from source kinds.

### D1 — the development metric does not measure the deployed task

`stratum_macro` nDCG@10 on the 100-query development set: BM25 `0.1092`, released Zero v1 dense
`0.0873`, v1 + DBSF `0.1034`, Stella dense `0.1366`, Stella + DBSF `0.1410`. Stella Recall@10 is
`0.2320`. For roughly three quarters of development queries no route places the labeled answer in
the top ten of 79,269 units.

A ceiling that low on a frozen 400M teacher is a property of the labels and the task, not of the
tower. The task as posed is: given a multi-sentence issue body, locate the single maintainer
comment that resolved it, with thread siblings defined as non-relevant. That is asymmetric answer
linking between two different genres. It is not the lookup the CLI performs.

Two corroborations that this is labeling rather than retrieval:

- Confirmation error/troubleshooting nDCG@10 is exactly `0.000` for BM25, for dense and for
  fusion across all ten of its queries. Confirmation concept/how-to is `0.000` for BM25 across its
  five. When a lexical and a semantic route both return a clean zero on the same stratum, the
  labeled target is unreachable by any means, which is a label property.
- Realized strata are far below the registered 35/15 target. Development is n = 22, 12, 25, 28, 13
  and confirmation contains buckets of five. `stratum_macro` therefore averages five small, noisy
  estimates; development concept/how-to Stella Recall@10 of `0.0833` is one query in twelve.

**Required check before any further work.** Report Stella Recall@100 and Recall@1000 against the
current qrels. If the labeled target is absent from the top 1,000 for most queries, the qrels are
wrong and every downstream comparison in this run is uninterpretable. This is one evaluation pass
and it gates everything else.

### D2 — development and confirmation invert, so the gates sat below the noise

Development places v1 dense (`0.0873`) below BM25 (`0.1092`). Confirmation places v1 dense
(`0.1532`) above BM25 (`0.1092`) and above the *development* Stella ceiling. The same system on
two samples of the same protocol produces opposite orderings between the lexical and dense routes.

The registered margins are +0.010 fused and +0.005 dense absolute. T0 returned +0.007591 dense
with a 95% paired interval of [-0.000844, 0.018995], and +0.002953 fused with [0, 0.008858]. On a
scale whose entire dense-to-lexical spread is roughly 0.02 and whose intervals straddle zero,
those margins are neither reachable in practice nor informative when reached.

The decision was procedurally correct against the locked registry. The inference drawn from it is
not supported: this run cannot distinguish "no improvement" from "improvement smaller than the
measurement error". `ENCODER_NO_IMPROVEMENT` should be restated as not measurable under this
protocol, in `FINDINGS.md` and in anything shown to the owner.

### D3 — the report-only probe contains the result the protocol was built to find

`results/m18_bare_term_probe.json`, dense route, term `k8s`:

| Rank | released Zero v1 | T0 step 250 (ineligible) |
|---:|---|---|
| 1 | (untitled) | (untitled) |
| 2 | `release v0.8.0` | `Data persistence for k8s deployment` |
| 3 | `v0.8.2` | (untitled) |
| 4 | `Data persistence for k8s deployment` | (untitled) |

The v1 column is the reported defect reproduced inside this repository: the bare token `k8s`
retrieving release-version artifacts. The trained column removes both version releases from the
top five and lifts the substantive k8s issue from rank four to rank two. The `s3` dense list
improves less and still shows `build(deps): bump syn from 3.0.2 to 3.0.3` at rank four.

Per the registry this probe is report-only and correctly could not affect selection. The
governance was right and the experimental design was wrong: the only instrument pointed at the
motivating defect was, by construction, unable to influence the outcome. P2 above proposed exactly
this artifact as a decision input and it was built as an impression instead.

### D4 — the trainable signal was too small to conclude anything about capacity

Training carried 785 views, of which 298 are non-held-out query-derived views. Sixteen new rows
were trained. No complete alias pair survived the trainable-query filter, so the registered
alias-consistency term was inactive. Gradients reach only batches containing a trainable row.
A few hundred distinct query contexts over 4,000 steps at batch 256 is a repetition regime, not a
capacity test.

The admitted vocabulary is itself diagnostic: `qdrant`, `shard`, `grpc`, `snapshot`, `deps`,
`json`, `flaky`, `docker`, `kubernetes`, `hnsw`, `config`, `s3`, `turboquant`, `arm64`, `k8s`,
`gridstore`. Several of these are ordinary words the base tokenizer already handles. On a single
project's 79,269 units the support threshold admits frequent common words rather than the
digit-bearing and vowel-free class the gate in `m18src/vocab.py:76` was written to select. One
repository does not supply enough distinct contexts per identifier to fit a 1,024-dimensional row.

### D5 — no per-artifact collapsing, in scoring or in display

BM25 for `arm64` returns four chunks of issue `5952` in the top four ranks; the `s3` and `k8s`
lists repeat artifacts `4705` and `6515` similarly. Nothing reduces a result list to one entry per
artifact. This depresses nDCG whenever sibling chunks outrank the labeled chunk, and it degrades
the owner's own top-ten judgement, which would show ten rows covering three threads.

Related: confirmation exact/version/numeric is BM25 `0.372`, v1 dense `0.300`, fused `0.186`.
Fusion is below both inputs on the stratum where one route is clearly strong, which deserves a
look independent of the encoder question.

### Recommended order of work

1. Run the D1 recall check. Treat its outcome as a gate on everything below.
2. Rebuild the evaluation on pooled judgments: take the top ten from BM25, v1 dense, Stella and
   any candidate, deduplicate by artifact, judge the pool, and admit multi-relevance. Forty to
   sixty pooled queries is worth more than 140 structurally labeled singles. Include bare and
   short queries as a first-class stratum.
3. Collapse results to one entry per artifact before scoring and before display.
4. Do not re-ask the owner for a verdict he has already given. His 2026-09-11 lgtm@10 experiment
   indexed the same corpus and rejected Zero dense; repeating it changes nothing. The one thing
   his own message records as untried is hybrid retrieval ("haven't tried in hybrid setup"), and
   the probe shows fusion already repairs both terms he named, with no retraining: `zero_v1/dbsf`
   returns `Implement S3 snapshot manager` at rank one for `s3`, and `Data persistence for k8s
   deployment` at rank one for `k8s`, against the version-release lists his dense-only run
   produced. Send that side-by-side, not a request to re-judge. Fix D5 first: a top-ten eyeball
   judgement is precisely what per-artifact flooding ruins. Collect his 30 to 50 queries as
   protocol input, which is still outstanding, and keep it separate from asking for a verdict.

   Two honesty constraints on that message. Fusion is not uniformly better: confirmation
   exact/version/numeric is BM25 `0.372`, dense `0.300`, fused `0.186`. And his second complaint,
   that a full-model top document falls to rank twelve under Zero, is an ordering problem that
   neither fusion nor vocabulary addresses; it belongs with the static late-interaction lever and
   should not be presented as solved.
5. Only then revisit row training, and supply it from text beyond one repository. Broad software
   corpora return here as training support, with Qdrant as the evaluation corpus, which is the
   reverse of the arrangement this run was given.

Nothing in this section reopens the M18 decision, relaxes a locked margin, or revises a recorded
result. It argues that the recorded result answers a different question than the one asked, and
names the checks that would settle it.
