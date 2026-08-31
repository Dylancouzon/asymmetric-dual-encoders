# M9.4 final-run lock — the single six-set transaction

**Classification: DISCLOSED AMENDMENT to the M9.2 lock, awaiting Dylan's ratification.**
Not the original M9.2 preregistration and must never be described as one. The mandate required
these fields in the M9.2 commit, before the main run; they were absent (`registry.json` had no
final-run key). Written 2026-08-31 during M9.3, revised after adversarial review
(`research/m9-codex-finallock-2026-08-31.log`, 7 BLOCKER / 6 MAJOR / 1 MINOR, all actioned).

**Full disclosure of what its author had observed when writing it:** SCREEN-3 dev evaluations of
the running build at steps 0 / 15,000 / 30,000 (0.34619 / 0.46710 / 0.50148) and throughput
telemetry. **No six-set, reserved, LoTTE, or confirmatory output existed or was read** — none has
been produced in M9 at all. A reviewer may judge influence against exactly that set.

Frozen on push except for the freeze-commit bindings marked BOUND-AT-FREEZE. Nothing may be
edited once any six-set output is revealed.

## Access state machine — durable, and ordered so ambiguity is impossible

The prior design let a crash land in a state that was simultaneously retry-eligible and spent.
The receipt is therefore durable **before** the first protected byte is opened.

| # | transition | durable artifact |
|---|---|---|
| 1 | acquire process lock; verify guard, freeze hashes, clean tree, HEAD pushed | none |
| 2 | append `FINAL-RUN-BEGIN` to the ledger, commit, **push** | ledger on origin |
| 3 | **create and PUSH the annotated `m9-six-spent` tag.** Push failure aborts here, before any protected read | tag on origin |
| 4 | only now open six-set queries/qrels; bridge; score; write `results/m9_final_run.json` | result file |
| 5 | compute decisions; append `FINAL-RUN-END` + result digest; commit; push | ledger digest |

- **Any failure at or after step 3 consumes the access.** No rerun, under any flag.
- `--infra-retry` is admissible **only** for failures durably proven to precede step 3 — i.e. the
  tag does not exist on origin. It is not a judgement call.
- A crash between 4 and 5 is **recovery-only**: a separate `--recover` path recomputes decisions
  from the already-written `results/m9_final_run.json` and **never re-reads the six**. This closes
  the wedge where a spent access could not establish C1 for the reserved conditional.
- Preflight must not open six-set queries or qrels. It may check only paths, hashes and sizes.

## Phase 1 — bridge (validation only)

Anchor `bge-small-en-v1.5`, freshly scored, compared to the frozen row. Proceed only if ALL hold:

| check | tolerance |
|---|---|
| qids per dataset | zero missing, zero extra, zero reordered |
| per-query nDCG@10 vs frozen | `max abs delta <= 3e-4` |
| hashes verified | qrels, preprocessing, model revisions, dtype, exact-search code, tie-breaking |

**The freshly-scored anchor row is validation-only and is discarded.** It never enters C1, C2 or
any reported number. Bridge failure consumes the access (step 3 already passed).

## Comparator source — one immutable snapshot

C1 and C2 use **only** the named rows loaded from `results/perquery.json`, pinned at
`sha256 6b18e3dd74fd308b087d4e652c0999c9d6a2e729c78edb73f000f9b7a25fda7e`. The scorer verifies
that digest before use and aborts on mismatch; the file is opened read-only and never rewritten.
This matters most for C2's `leaf-ir-asym` row, which the bridge does not validate.

## Phase 2 — manifest (revision-complete)

Datasets, equal weight 1/6: scifact, nfcorpus, fiqa, arguana, scidocs, trec-covid.

| system | role | status | pinned by |
|---|---|---|---|
| `nano-dense` | C1 + C2 subject | INCLUDED | BOUND-AT-FREEZE: checkpoint sha256 in `m9/FREEZE.json`; scorer aborts unless it matches the loaded checkpoint |
| `bge-small-en-v1.5` | release bar; bridge anchor | INCLUDED, frozen row | perquery digest above; `registry.models.students` revision `5c38ec7c…` |
| `leaf-ir-asym` | aim bar | INCLUDED, frozen row | perquery digest above |
| `arctic-embed-m-v1.5`, `mdbr-leaf-ir`, `opensearch-doc-v3-gte`, `lr-dense-pertask`, `bm25` | context | INCLUDED, frozen rows, descriptive only | perquery digest above |
| stella-400M symmetric ceiling | retention denominator | INCLUDED, not rescored | `results/m7_final_run.json`, digest BOUND-AT-FREEZE |
| `nano-symmetric` | — | **OMITTED** — document path never trained, exported or frozen | — |
| every other system | — | **OMITTED** | — |

No system may be added after any output is revealed.

## Statistics

Both contrasts assert, before computing anything: exactly the six named datasets present in both
inputs, frozen ordered qids identical, then `strict=True`. A five-dataset macro must be
impossible, not merely unlikely.

- **C1** = nano − bge-small (RELEASE). **C2** = nano − leaf-ir-asym (AIM).
- Statistic: per-query nDCG@10 differences on aligned qids; dataset means at weight exactly 1/6.
- **Bootstrap**: B = 10,000; resample `n_d` queries with replacement within each dataset;
  seed `900`; one **frozen draw plan** generated once and reused byte-identically by C1 and C2,
  its digest serialized into the result. Decision field is
  **`lower_q0125_raw` = `np.quantile(draws, 0.0125, method="linear")`, serialized at full
  precision. ONLY this field decides.** Rounded fields and `ci95_raw[0]` (a 2.5% endpoint) are
  reporting-only and are explicitly NOT the gate.
- **Sign-flip**: B = 100,000 one-sided dependent replicates on the same equal-weight statistic;
  independent Rademacher signs per paired query, one frozen sign plan shared by C1 and C2;
  seed `901`; `p = (1 + #(T* >= T_obs)) / (B + 1)`. Holm step-down over the two p-values at
  family alpha = 0.025.
- A contrast passes **only if both** the bootstrap bound and Holm reject. The sign-flip is a
  required sensitivity conjunct, not evidence its weak-null assumptions hold.
- Implementation: `m9src/final_stats.py`, a thin wrapper over `m7src/boot.py`
  (`signflip_dep`, `holm`, `strata`, `unit_key`) that additionally returns the full-precision
  draw vector and `lower_q0125_raw`. **This is new code and must be reviewed and unit-tested
  before the run** — the earlier claim that M9 adds no statistics code was wrong: `boot.py`
  exposes no 1.25% quantile and rounds `one_sided_lower_2.5` to 4 dp.

## Claim decision table

| C1 | C2 | outcome |
|---|---|---|
| pass | pass | release **and** aim; the verbatim headline below is permitted |
| pass | fail | **release.** Ships as the pair's low-compute point, reported as "did not resolve above the LEAF system" |
| fail | **pass** | **no release**, but the aim claim IS permitted under the exact qualification. C2 does not gate the ship, and C1 does not gate the claim |
| fail | fail | no release; reported as a measurement under the pre-registered miss-is-publishable framing |

**The only permitted performance headline, verbatim and unparaphrased:**

> "the selected-teacher-document + nano-query asymmetric system outperformed the arctic-document +
> leaf-query system on the six development-informed datasets in our exact-search harness"

always carrying the disclosed-overlap qualification. Never "nano beats LEAF". No paraphrase and no
stronger construction is permitted. The words **"resolved", "confirmed" and "unrestricted"** may
not be applied to either contrast.

**Mandatory disclosures beside the headline** (each binds a value, not a topic): teacher sizes and
dims 400M/1024d vs 109M/768d · index bytes · doc-encode cost · retention vs each system's own
teacher on this same six-set harness, LEAF = 97.9%, with its 97.7% BEIR-14 figure used only as a
differently-surfaced literature number · per-dataset rows including the expected TREC-COVID loss
(ceiling 0.8234 vs arctic 0.8461) · the disclosed-overlap table · seed-replica variability
UNMEASURED (owner waiver 2026-08-30).

NDO-4 and reserved NDO-3 are descriptive robustness checks only; they cannot grant, remove or
weaken the qualification. Each preregistered nano−leaf estimate is reported with its CI, and a
non-positive point estimate or a CI including zero is stated prominently.

## Reserved batch — complete, and unmodifiable after any six-set output

Registered conditional: **`if C1 passes then execute`**, exactly as written, else never.

- **Systems**: `nano-dense` INCLUDED; `bge-small-en-v1.5` INCLUDED; `leaf-ir-asym` INCLUDED;
  every other system OMITTED. All three encoded fresh (no frozen reserved comparator rows exist).
  Model revisions: nano = the freeze checkpoint sha256; bge-small `5c38ec7c…`; leaf-ir-asym as
  pinned in `results/FINAL_MATRIX.md`. Executor asserts the six-set freeze commit sha before start.
- **Estimands and directions**: R1 = nano − bge-small, R2 = nano − leaf-ir-asym, both on the
  family-weighted NDO-3 macro = 0.50·DBpedia + 0.25·cqadup-android + 0.25·cqadup-english.
  Positive favours nano. **Descriptive only; zero alpha; no gate, no claim.**
- **Confidence procedure**: identical machinery to the six — B = 10,000, seed `902`, frozen draw
  plan, strata = one per component, weights as above (NOT renormalized), reported as a two-sided
  95% interval with `ci95_raw` at full precision. No pass/fail threshold exists.
- **Additional cuts**, all descriptive: equal-weight dataset macro (1/3 each), query-pooled,
  per-dataset, leave-one-out **with the remaining weights renormalized to sum to 1** (stated
  because unrenormalized leave-one-out is a different estimand).
- **FEVER**: labelled double-contaminated sensitivity row, zero alpha, never gate-relevant.
- **Resources**: ~30M doc encodes across three towers, ~44 GB fp16. Acceptance gate: **>= 120 GB
  free before start**, asserted by the executor, which aborts otherwise.
- **Crash semantics**: per-system atomic outputs; a crash resumes at the first system lacking a
  complete output and never re-encodes a complete one; all intermediate scores suppressed until
  every system completes; no partial reserved result is ever reported or read.
- Rehearsal on open sets first, with pinned hashes, before any reserved byte is opened.

## Registry binding

Every constant above is mirrored into `registry.json -> final_run` and the executor reads it from
there, never from this prose and never from the M9.0 screen defaults (B = 20,000, seed 0 — which
must never be used for the final run). Prose and registry disagreeing is a hard abort.
