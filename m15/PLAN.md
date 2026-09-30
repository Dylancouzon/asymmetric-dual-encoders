# M15 paper plan, v5.3 (2026-09-30)

Supersedes the 2026-09-17 plan, which is in git history. v5.3 applies Astra's adversarial plan review
(`REVIEWS/2026-09-30-astra-plan-review.md`): one P0 (E8's fit list), five P1, two P2, all applied. Owner-facing version with charts:
https://claude.ai/artifact/U2Ph1TXSUmLDXG7bHXRKkN (private to Dylan). Reviews behind it:
`REVIEWS/2026-09-30-*.md`.

## Owner direction (Dylan, 2026-09-30)

- A research paper, not a benchmark report or competitor overview: findings, implications, what the
  design makes possible. The project's comparator bars were release gates, not the paper's goals.
- The registered clean-4 contrast and the one-shot reserved-four result are reported in full as the
  pre-registered test in the evaluation section. bge-small, LEAF and BM25 are reference points only.
  CLAUDE.md and `instructions-m15.md` carry this ruling. The frozen `m20/beir15_registry.json`
  `reporting.headline` line is unchanged (hash-pinned); this ruling overrides it for presentation only.
- A failed approach enters only where it carries significant value.
- Goal: new knowledge for the IR industry, and a paper that raises Qdrant's reputation in IR.
- All measurements on the Apple M5 Pro except E8, which runs on Runpod. No second benchmark machine:
  the load test that needed one is cut.
- Approved additions: E8 (tower generality on public BEIR) and figure F1 (the frontier).

## Question and title

How much retrieval quality does one frozen document index keep as the query encoder shrinks from the
tower's own 400M path, to a 34.5M distilled transformer, to an int8 token lookup table with no neural
network at query time, and what decides where the quality is lost?

Working title: *How Much Query Encoder Does a Frozen Document Index Need?* Alternative (Fable):
*Retention as Query Compute Drops: Cheap Query Encoders on a Frozen Document Index.*

## Findings, in paper order

Novelty from the 2026-09-30 qualitative reviews (both verdicts "partly" on v4, fixes applied here);
evidence strength from Astra's rating pass. Numbers are verified against the named sources.

1. **Retention as query compute drops (the lead).** BEIR-15 macro nDCG@10, exact search, single-prompt
   Stella query path 0.5614 = 1.000; Nano 0.5081 (0.905); Nano + BM25 DBSF@100 0.5110; Zero + BM25
   0.4933 (0.879); Zero 0.4572 (0.814); BM25 0.4006 (`results/m20_beir15_run.json`). Retention is a
   ratio of macros. Zero beats BM25 on 11 of 15. Per-dataset map: Zero 0.667 (trec-covid) to 0.958
   (climate-fever); Zero scores above Nano on fever, hotpotqa, climate-fever, with confounds stated
   (Zero trained on FEVER-train, Nano's pool excluded FEVER, Nano starts from bge-small). Novelty:
   first numbers. Evidence: adequate, descriptive. Needs E1 and E2 for a real cost axis.
   Disclose: Stella's card reports 58.97 with per-task instructions, max_len 400, bf16; ours is the
   registered single-prompt path. Nano's 90.5% is a ratio of macros; arXiv 2306.11550 (92.5%) averages
   relative performance across datasets and LEAF (97.7%) uses its own tower and benchmark, so the paper
   presents these as differently defined reported results, not a common ranking.
2. **Tower quality did not predict how well a tower distills into a table (one fixed recipe).** The
   paper claims this for the closed-form table recipe, not that distillability is intrinsic to the
   tower: E8 varies checkpoint, exposure, dimension and readout while holding the recipe. Nine complete tower pairs on two CQADupStack dev forums
   (`results/m7_learnability_report.json`): the highest-ceiling tower (arctic-embed-l, 0.4931) gave the
   fifth-best closed-form table; Stella kept 71.56%, gte-large 43.15%. Base beat large in bge, e5, gte
   and arctic (two base towers lack a ceiling; the arctic pair mixes m-v1.5 with l). Spearman 0.000 over
   the original eight configurations, 0.0833 over the current nine configurations, 0.2619 over the
   eight distinct complete checkpoints (0.1429 was the original seven). Label each roster. The
   historical screen used a fit list with 4,582 protected-query hits (1.31%, `m7/LEDGER.md:274`);
   disclose it, and let E8's clean refit carry the claim. Also: cosine to the teacher rose while nDCG fell; highest-cosine candidate sixth of ten
   (`m7/FINDINGS.md:26`). Precedent: Cho and Hariharan 2019; verify NanoVDR arXiv 2603.12824, which
   may report the opposite. Novelty: new for embedding towers. Evidence: weak n, strong counterexample,
   until E8 runs.
3. **The lookup table's limit.** Exact lemma: affine post-pooling transforms and fixed per-token weights
   are absorbable into a freely parameterized mean-pooled table (max abs difference 9.31e-14,
   `results/m7_absorb_check.json`); nonlinear transforms are not covered. Recipe evidence kept separate:
   KL term median 1.08e-07 nats (teacher-target entropy 4.73e-07), positive first for 99.75% of 4,000
   TRAIN queries against the seeded random bank (`m8/RESULTS.md:21`); no table-side lever improved dev
   by more than about 0.005; the hard-candidate objective (0.777 nats) was never trained; document-side
   co-adaptation was never tested. Fusion is the lever that worked. Half a page.
4. **Routing each query to a tier.** E5 oracle headroom plus E6 fertility router. Fertility predicted
   the Zero-to-Stella gap (-0.050 nDCG@10 per +1 subword per word, t = 4.61, `m8/FINDINGS.md:78`), not
   the Zero-to-Nano gap, so E6 tests a hypothesis, and fertility failed as a lever (D2-PRE).
5. **Short queries and prefixes.** M7 finding 7 (table agreement flat from 16 to 256 words) plus E4.
6. **Does the saving survive the search?** One section of the system-cost chapter, not the lead. M1-M6
   hint: a lookup-query system lost 2.1 points (0.021 nDCG@10) on FiQA at default `ef` against 0.7
   (0.007) for bge-small, on different indexes (`research/m1-m6-findings.md:27`). E2 tests it on one
   index, against a loss target fixed in advance.
7. **Fusion depends on candidate depth** (`m12/FINDINGS.md`), in the body.
8. **The pre-registered test.** Clean-4 Nano - bge-small +0.017648 (lower bound +0.003674),
   established; all six +0.027449; Nano - LEAF +0.016181 all six, -0.001063 clean-4 unresolved.
   Reserved NDO-3 (DBpedia 0.5, cqad-android 0.25, cqad-english 0.25): +0.0032 [-0.0069, +0.0134]
   unresolved, query-pooled -0.0121 [-0.0219, -0.0024]; Nano - LEAF -0.0389 [-0.0488, -0.0290].
   FEVER separate. The registry forbids superiority, equivalence or significance claims from BEIR-15
   or reserved rows.

Compressed to one paragraph: coverage (M9 93.8% vs 50.1%) with the M10 width result (+0.021651 for a
1152-wide head), which shows capacity matters too. Appendix: swap hazards in two lines (prompt cosine
0.80 `m11/STATUS.md:150`, padding cosine 0.35 `m11/STATUS.md:98`, float64 pooling), absorbability
proof, full BEIR-15 table with contact labels, statistics, hashes, closed avenues (six rows at most).
Cut: load test, 256 MB container demo, "no ML runtime", LightRetriever/OpenSearch bar contrasts, the
M9 synthetic prototype as cost evidence (its "nano" was a pretrained MiniLM-L6 with a random head, and
its index and table were synthetic: `bench/edge_prototype_pair.py:6-11`, `m9src/edge_cost.py:88-90`).

## Use cases (discussion section)

| Use case | Size | Evidence |
|---|---|---|
| Per-request tiering | a result | E5 + E6 |
| Search as you type | a result if E4 is informative | E4, a synthetic incomplete-query test |
| New or retrained query encoders without re-ingestion | a paragraph + cost table | Zero, Nano and M17-M19 variants all built on one frozen tower; Zero ~20 min training after 8-12 h re-encoding; Nano 57.3 h on one A100 (~$95); component costs only |
| Use-case-specific encoders, one collection for many use cases | two sentences | M18 +0.007591 dense, missed fused bar (+0.002953 vs +0.010); M19 inconclusive because its relevance labels could not separate candidates (the label bottleneck is M19's finding only; M18 was a bounded failure under an evaluation task that did not test its motivating defect, cause unassigned); cite arXiv 2601.04646 |
| Edge | part of finding 6 | per-encoder quantization loss, E2 |
| High QPS | one sentence | encoder share of latency, E2 |

## Outline

1 Introduction (1 p) · 2 Setup, build costs (1 p) · 3 Retention as query compute drops, F1 and F2 (2 p)
· 4 The tower decides what you can distill, F3, E8 (1.5 p) · 5 What a table cannot do and what routing
can (2 p) · 6 Short queries and prefixes (1 p) · 7 System cost: encoder share, does the saving survive
the search, fusion depth, F4 (2 p) · 8 The pre-registered test (1 p) · 9 Limitations (0.5 p) · Appendix.

## Figures

- **F1, the frontier (approved).** One qualified figure, one workload per axis: y is BEIR-15 exact
  retention of the Stella query path; x is E1's common-protocol encode p50 on the M5 Pro, log scale.
  Points: Stella query, Nano, Zero, and bge-small and LEAF as hollow reference markers on their own
  index (both are timed in E1). BM25 and the fused rows appear on F1 only if their query-side cost is
  measured under the same protocol; otherwise they stay in the table. End-to-end ANN latency never
  shares an axis with exact quality: it goes in F4, with E2's own nDCG on E2's own data.
- F2 per-dataset retention, Zero and Nano. F3 tower score against table score (E8 adds the BEIR
  panel). F4 end-to-end latency against `ef` and quantization per encoder (E2).

## Measurements

Method file first (`m15/MEASUREMENTS.md`, dated), then one Astra review, then runs. No training, no
protected data, no reserved dataset in any role.

| ID | What | Machine | Time |
|---|---|---|---|
| E1 | Encoder latency: Zero, Nano, Stella query path, bge-small, LEAF query encoder (`mdbr-leaf-ir`); three query-length buckets; common protocol | Mac | ~2 h |
| E2 | A *positive-preserving 1M MS MARCO subset diagnostic*: all dev-judged positives plus random fill, IDs and seed pinned, Stella vectors encoded on the Mac; exact and ANN both on that subset; its nDCG and ANN penalty are never presented as full-corpus numbers; MS MARCO carries its validation-only licence role. Plus FiQA 57k (vectors local, check they match the registered encoding). Per query encoder: HNSW `ef` sweep; original vs int8, binary, TurboQuant 4-bit with oversampling; neighbour recovery@10 vs exact; nDCG@10 under ANN vs exact; end-to-end latency; query-to-nearest-document geometry. Qdrant native macOS binary for latency, Docker only for memory limits. BM25 + DBSF@100 in Qdrant for fused costs. Decision rule fixed in the method file: loss targets relative to each encoder's own exact nDCG (for example 1%, 2%, 5%), and the latency each encoder needs to meet them; failing settings stay visible | Mac | ~1 day + overnight encode |
| E4 | Prefix robustness on SciFact, NFCorpus, FiQA; controls: length-matched random token subsets, BM25, retention against each encoder's own full-query score | Mac | ~3 h |
| E5 | Per-query oracle headroom Zero vs Nano on the 12 non-reserved BEIR-15 datasets (BEIR-15 minus FEVER, DBpedia-entity and CQADupStack, whose android and english forums are reserved), committed per-query rows | Mac | ~15 min |
| E6 | Fertility router: a Nano-use budget (or quality-cost objective) and its threshold frozen on the two M7 dev CQADupStack forums (programmers, physics), outside the 12 evaluation datasets; evaluated on those 12 against always-Zero, always-Nano and random routing at the same Nano fraction, with feature cost included. Neither E5 nor the 12 datasets selects anything | Mac | ~1 h |
| E7 | Optional: word-order sensitivity of the towers against table retention | Mac | ~2 h |
| E8 | Tower generality (approved). Question: under this fixed closed-form recipe, does a dev-selected table ranking predict the six-set table ranking, and does tower quality? Fit list: M8's cleaned, hash-pinned successor (337,981 kept, `m8/RESULTS.md:18`, `results/m8_trainq_manifest.json`), never the stale 349,934 superset and never a reconstruction that reopens protected data; **blocked until that file is recovered**. With it, refit every configuration in `results/m7_learnability_report.json` and rescore the two dev forums, freeze each lambda on the dev forums, then score the six public BEIR sets of M7 (scifact, nfcorpus, fiqa, arguana, scidocs, trec-covid; ~272k documents) once. Report absolute quality and retention, checkpoint-level results with arctic-l's mean readout as a separate control, a per-tower and per-dataset disclosed/unknown exposure matrix, and a tokenizer-compatibility check (the recipe assumes a shared BERT WordPiece vocabulary, `m7src/encoders.py:58`) | Runpod GPU | ~3-5 h GPU |

## Release

Paper source with one command that regenerates every table and figure from committed JSON. Stella
BEIR-15 document vectors on Hugging Face with a scoring script, after a licence decision for Quora,
Climate-FEVER and MS MARCO. Deferred and not in scope: the bring-your-own-tower kit, the
closed-form table tool and the browser search-as-you-type demo (companion items the owner may revisit).
