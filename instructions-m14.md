# M14 — the paper

Created 2026-09-04 (Dylan): *"paper writing can be its own milestone after building nano. We'll
decide then if we want to do 2 papers or 1 but probably only 1."* Split out of M13, which held it
as deliverable 4 and would have kept it hostage to a milestone that may never run.

Binds from `instructions-m7.md` unchanged. Working files under `m14/`.

## Sequencing

**Runs after M13 delivers nano.** If M10/M13 never run, M14 still runs — on the zero-only frontier
plus whatever M12 produced. **One paper is the default**; the one-vs-two decision is Dylan's, taken
at the start of M14 when the evidence is known, not now.

## Benchmark composition — deferred here, but the RULE is registered now

**Deferred to M14 (Dylan, 2026-09-04):** *"Benchmarks can be differed to M14, when we do the paper
so that way we'll cross that bridge for both Zero and nano."* Right call — the product is the pair,
and settling zero's partition now while nano's waits risks reporting the two halves of one frontier
on different benchmarks.

**But the choice is registered BEFORE nano's numbers exist, not made at M14 having seen them.**
Deferring the decision is legitimate; deferring the *rule* would be post-hoc selection among
pre-registered partitions, which is the one thing that would discredit the frontier. So:

1. **Headline = clean-4** (`nfcorpus`, `scidocs`, `scifact`, `trec-covid`) — the pre-registered
   no-disclosed-teacher-overlap partition (`results/m7_final_run.json` `clean4_robustness`).
   Registered now, on the contamination argument alone, for **both** zero and nano.
2. **All six always reported beside it**, for comparability with published BEIR numbers.
3. **Contamination robustness (all-6 minus clean-4) is its own reported row** for dense and fused.
4. **No re-picking the six.** Datasets may not be added to or removed from the headline after any
   number is seen. New sets may only enter as labelled diagnostics.
5. **MS MARCO, if built, is a validation-only diagnostic** — never a headline. Permitted by the
   2026-09-04 licence rule (validation, not training). It is the MIRROR confound: every comparator
   trains on it and neither of ours does, so we expect to lose and losing is still informative.
   nq-250k-style build (~6,980 queries, 250K distractors, a 250K-doc stella encode, ~1h).

**The number that makes this cheap** (M12, `m12/FINDINGS.md`): fused systems show far less
**PARTITION SENSITIVITY** than the dense table — clean-4 costs convex0 **0.0045**
(0.4911 → 0.4866) and *gains* DBSF@100 **0.0025** (0.4887 → 0.4912) — while the dense table loses
**0.0241** (0.4339 → 0.4098).

> **CORRECTED 2026-09-08 (astra): this is partition sensitivity, NOT contamination immunity, and
> the earlier "nearly contamination-immune" wording is withdrawn.** The quantity is
> `S̄₆ − S̄₄ = ⅓(S̄_excluded-two − S̄₄)` — a difference between two *groups of datasets*, which does
> not identify a contamination effect. A small difference is equally consistent with fusion's
> relative strengths across those particular tasks, with BM25's contribution on them, or with
> cancellation. **Report the partition delta and say what it is.** The claim that survives is
> narrow and still useful: the fused system's score is far less sensitive to WHICH partition is
> reported than the dense table's is. And on clean-4 the fused story is *stronger*: 0.4866 vs BM25's 0.4409,
where dense-only (0.4098) sits below BM25. Honesty costs 0.0045 here and buys the whole objection.

## What the paper argues

The frontier is the *deliverable*, but it is not the *finding*. The finding is:

**Teacher quality does not predict distilled-table quality.** Spearman ≈ 0 (ceiling→table 0.000 over
eight; MTEB→table −0.11 over seven verified rows) across **eleven** measured teachers. The
best-ceiling candidate ranked 7th. `mxbai-embed-large-v1` has the same measured tower quality as
`bge-base-en-v1.5` (0.4433 vs 0.4484) while its table is **0.057 worse** — three times the
parameters, higher MTEB, worse product. Selection must run on the distilled artifact, never the
tower. Survives a contamination control on data the winner never trained on (SQuAD +0.1652, ESCI
+0.0384, `results/m7_offfamily_report.json`).

This is measured today and needs no nano. Everything else is evidence for it or around it.

## Contents

1. The finding above, with the eleven-candidate table and the exposure-free control.
2. The quality-vs-query-cost frontier: `zero` and `nano` (M13, if it lands), against the
   21-system × 6-dataset matrix, plus M12's fusion-operator result. (`zero-hybrid` was **cut** from
   M12 on 2026-09-04 and moved to M16; it is not an M14 deliverable.)
3. Edge cost rows — query asset, document index, hydration, CPU latency — plus the offline BM25
   index cost M12 measures.
4. The Qdrant Edge prototype: architecture and latency, with the exact-search caveat.
5. **The comparator table deliberately kept OFF the model cards** (`instructions-m11.md`
   Amendment B): `LR-dense-pertask 0.4583`, the OpenSearch tie, the missed bar. This is its home.
6. Negative results, stated as results: M8's twelve levers, M9's coverage failure, and every kill.

## Rules that bind the writing

- **Every headline number is paired on frozen comparator vectors with pre-registered statistics.**
  No number enters the paper that is not already in a committed result JSON.
- **stella's ArguAna/FiQA/FEVER exposure is disclosed wherever a stella-derived number appears.**
- **Non-commercially-licensed sets may be reported as VALIDATION rows** (Dylan, 2026-09-04;
  `research/m7-data-licensing.md` §Rule change 2026-09-04) — an MS MARCO row is now reportable, which
  reviewers will look for. Two things must appear beside it: that it informed no training, and that
  every comparator trains on MS MARCO while `zero` and `nano` do not, so the row is biased **against
  us**. The training exclusion and its +0.0058 price are reported unchanged.
- **Deltas inside the ~0.005 lever band are labelled as ties**, not wins.
- Do not quote the off-family macro (+0.1018) as a product delta — it is a contamination control.
- Do not claim `zero` confirmatorily beat BM25: C2 was +0.0165 [0.0017, 0.0311], Holm p=0.0149 >
  0.0083, **not rejected**.
- **Cite pyNIFE** (Stephan Tulkens, MIT, 2025-11-03) as prior art for the construction — a per-token
  dense lookup table distilled from a frozen teacher, reusing that teacher's index unchanged. It was
  dropped from the model card 2026-09-04 as card clutter; a paper without it would be a different
  matter. Its retention comparison on fiqa (73.4% of a weaker teacher vs our 67.1%) is also the
  cheap lead in M16.
- No new measurement. If the paper wants a number that does not exist, that is a milestone, not a
  paragraph.

## Claims this paper must NOT make (astra, 2026-09-08 — verified; `research/m10-astra-v2-dispositions-2026-09-08.md`)

Each of these is currently asserted somewhere in the project's own files and is **not supported by
the evidence cited for it.** The corrected form follows each.

| do not claim | why not | claim instead |
|---|---|---|
| *"M9 was a coverage failure, not a capacity failure"* | reaching 93.8% on ONE distribution does not establish the capacity to reach it JOINTLY across many. The M9 table supports distribution-dependent failure and a coverage *hypothesis*; it does not isolate the cause | "M9's retention was strongly distribution-dependent (93.8% NQ vs 50–71% CQADupStack); we did not isolate the cause" |
| *"normalized L2 cannot push a 384-wide head past 90–93%"* | the PCA argument does not establish it. The student normalizes and the loss compares unit vectors, so the relevant quantity is `E‖P_S t‖`, while ordinary reconstruction maximizes `E‖P_S t‖²` — different objectives (`m10/PLANNING.md` §9) | "compressing the teacher's query representation to 384-d costs retrieval quality" — a fact about representations, not about what training converges to |
| *"A3−A2 establishes query-form coverage"* / *"A4−A3 measures whether synthetic data helps"* | both are **bundled training-distribution interventions**. Every arm is form-balanced, so an existing form gets ~750,000 presentations in A3 vs 312,500 in A4; and A3 already contains PAQ, which the mandate itself calls machine-generated | "A4−A3 estimates the effect of re-allocating 58% of query presentations to seven generated forms, against a corpus that already contains synthetic questions" |
| *"unresolved means a measured null / not positive / a statistical tie"* | a failed superiority test is not evidence of absence. Choosing a default under uncertainty is a **product policy**, not a finding | report the effect estimate and its interval, and label the policy choice separately |
| *"clean-4 is contamination-free"* | its actual property is **no *disclosed* teacher overlap**, and the mandate already concedes development-informed training design | "no disclosed overlap with the teacher's stated training data" |
| *"fused systems are nearly contamination-immune"* | **withdrawn above** — `S̄₆ − S̄₄ = ⅓(S̄_excluded − S̄₄)` is a difference between dataset groups, not a contamination effect | "the fused system's score is far less sensitive to which partition is reported" |
| the architecture as a **first** | the frozen-teacher lookup-table construction is prior art, as this mandate's own pyNIFE instruction requires | the contribution is the empirical evidence, the analysis, and a deployable implementation |
| the Spearman result as a **law** | its strongest defensible form is an empirical **counterexample** to selecting a teacher by tower quality. Note also that this file quotes Spearman over **eight** and **seven** comparable rows while stating eleven teachers were measured — **keep those denominators distinct** | "over the teachers we measured, tower quality did not predict distilled-table quality (Spearman ≈ 0 over N=8 / N=7 comparable rows)" |

**And the framing the review recommends for the central claim**, which survives either outcome:
an empirical study of *replacing the query encoder while preserving a pretrained document index* —
reporting retrieval quality and deployment cost for lookup-table and compact-transformer query
paths over one frozen index, with the finding that performance depends strongly on task
distribution and on the deployed retrieval configuration. A C1b pass then *strengthens* it into the
registered whole-system superiority claim; a miss still leaves a reproducible account of the
feasible trade-offs under the constraints.

