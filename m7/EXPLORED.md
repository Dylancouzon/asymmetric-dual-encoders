# M7 avenues explored and closed

Check this before starting anything new. One row per avenue; detail lives in the cited artifact.

| avenue | why killed | evidence |
|---|---|---|
| MIRACL (English) as a training source | Its 2,863 English train queries need the 32.9M-passage corpus (no parquet mirror; loader is a removed script). Wikipedia coverage already comes from HotpotQA/FEVER/SQuAD/NQ-open. Cost/benefit fails by 2 orders. | `trainmix.py` docstring |
| Climate-FEVER in UNTOUCHED-FINAL | No affirmative licence at any primary source; only HF mirrors claim CC-BY-SA, and a wrapper tag is not evidence. | `LEDGER.md` partition ledger |
| fp32 teacher encodes on dev/train corpora | fp16 is 2.4x faster and indistinguishable (cos 1.000000; \|Δ nDCG\| ≤ 3e-4). fp32 kept for the six + untouched-final only. | `results/m7_throughput.json` |
| bare (unprefixed) teacher vectors as the distillation target | The bge query prefix is worth +1.85 dev macro to the teacher itself, so the prefixed vector is the better target and the higher ceiling. | `work/devres/refs.json` |
| a dev component validating long-query behaviour | Training mix has no long queries (held-out p50=13 WordPiece, p90=24; 55 of 7,325 reach 64). ArguAna's are ~250. The only long-argumentative sources are ArguAna's own family, excluded by the contamination map. The ArguAna row is an extrapolation and is labelled one. | `LEDGER.md`; `work/dev/heldout-longq.json` |
| **fusing with `opensearch-neural-sparse-doc-v3-gte`** | Disqualified twice: vendor rule (OpenSearch Service is a competing product) **and** circularity — Tier 1 is *defined* as beating it. Recorded because a sweep proposed it as the top route. | `instructions-m7.md`; shortlist |
| BM42 as a sparse arm | Query-side attention, plus independent reproductions (Reimers, Bergum) failed to beat BM25 and Qdrant downgraded it to experimental. | Qdrant BM42 article + reproductions |
| **query-side centering / whitening / top-PC removal / SIF or IDF weighting as NEW CAPACITY** | **Refuted algebraically, to machine precision.** All are absorbable into the table: `mean(W-mu)=mean(W)-mu`; `normalize(A·mean)=normalize(mean(A·W))`; a per-token scalar is absorbed by scaling that row. So they cannot raise the ceiling — only act as a prior/init. Empirically confirmed: p1-objB's learned weights already *are* IDF-like (spearman −0.44 vs row update count; [CLS]/[SEP] downweighted to 0.61x median). | `results/m7_absorb_check.json` |
| length scaling (1/sqrt\|T\| instead of 1/\|T\|) | A no-op: any positive scalar function of \|T\| is removed by the final L2 normalize. | `results/m7_absorb_check.json` |
| `fn_margin` as the contrastive-collapse cause | **Refuted by measurement.** At 0.02 it removes 0.18% of negatives overall and 4.3% of the top-100 hardest. The leading suspect is exonerated. | `results/m7_diag_scores.json` |
| "random negatives are trivially separable" as the collapse cause | **Refuted by measurement.** 32.7 random negatives per query outscore the positive on average (15.9% of queries ≥1). Already flagged as asserted-never-tested; now measured. | `results/m7_diag_scores.json` |
| ~~Qwen3-Embedding-0.6B as teacher~~ | **REOPENED (review #2 MAJOR 15): closed on symmetric MTEB, the criterion this project refuted** (Spearman 0.000 vs the table). Untested on the adopted criterion. Practical blocker to testing it: a 151,669-vocab ridge solve needs an iterative solver (V x V normal equations do not fit in 25 GB). Reopen only with that solver and idle GPU. | shortlist + learnability report |
| ~~the Snowflake "justify-max" vendor tier~~ | **FALSIFIED 2026-08-26 — do not treat as closed.** The row claimed the tier was moot because arctic-embed-l is "~2 points below two admissible alternatives" on MTEB. Measured, it is the BEST of five candidates on the two CQADupStack dev components (+0.0447 over bge-base, CI-resolved) and the only one disclosing zero overlap with our six. Dylan ruled on the tier and approved it. Kept as a warning: this row closed an avenue on projected numbers that the measurement inverted. | `results/m7_teacher_probe.json`, `results/m7_teacher_contamination.json` |
| ~~granite-embedding-english-r2 as teacher~~ | **REOPENED (review #2 MAJOR 15): same refuted criterion.** Vocab 50,368 is within the relaxed line; a 50K ridge solve is borderline on RAM (chunked/CG). Untested on the table criterion; cheap-ish once the solver generalizes. | shortlist |
| a 2025–2026 teacher we had not seen | Swept; nothing survives. NVIDIA/Tencent/KaLM/mmBERT die on 128K–256K vocab; Salesforce/Linq/LGAI/NVIDIA-8b on non-commercial weights; ByteDance never released weights. Near-misses `ettin-encoder-400m` and `NeoBERT` clear every structural filter but their strong numbers belong to unreleased fine-tunes. | shortlist 2026-08-26 |
| MTEB v1 Retrieval as a *ranking* signal between teacher candidates | Not merely imprecise — **wrong on this evidence**. arctic-embed-l has the lowest MTEB v1 of the three 1024-d candidates and the highest measured macro; bge-large ties bge-base (p=0.48) across a ~1-point MTEB gap. Use it as a shortlist filter, never as an ordering. | `results/m7_teacher_probe.json` vs `results/m7_calibration.json` |
| a fixed-step objective-C sweep as the contrastive learning-rate test | Cannot isolate the lr: objective C runs the B phase at the same lr, and B is far from converged at the low end (0.2731 at 4k steps at 5e-5 vs 0.4449 at 3e-3), so every arm enters the contrastive phase from a different table. Replaced by A-only arms from one fixed checkpoint. | `m7/LEDGER.md` screen-redesign note |
| **the symmetric teacher probe as the teacher-selection criterion** | **Refuted by measurement.** Spearman(ceiling, distilled-table) = 0.000 over eight candidates. arctic-embed-l has the best ceiling and a table 0.0480 BELOW the incumbent's, CI-resolved. Select teachers on the table, never on the tower. | `results/m7_learnability_report.json` |
| arctic-embed-l as teacher | Chosen on the symmetric probe and **withdrawn on the table**: −0.0480 [−0.0608, −0.0349] vs the incumbent. Its ceiling is the best of eight and its approximability the second-worst. | same |
| gte-large-en-v1.5, e5-large-v2, e5-base-v2, bge-large-en-v1.5 as teacher | All CI-resolved BELOW the incumbent's table (−0.104 to −0.032). e5 was added specifically to test mean pooling; both e5 models sit below CLS bge-base. | same |
| **mean pooling as the mechanism behind stella's approximability** | **Refuted by a controlled test.** `arctic-embed-l-mean` — identical weights and dim, mean read-out instead of CLS — falls from ratio 0.526 to 0.472. Mean pooling made it worse. Stella's advantage remains unexplained, so there is no attribute to search new candidates on. | same |
| cosine agreement to the teacher's query vector as a selection metric | Rises with lambda while nDCG falls, and mis-ranks candidates (e5-large-v2: highest cosine 0.90, sixth of eight on retrieval). Diagnostic only. | same |

## Reopened, do not treat as closed

- **doc2query-style document expansion — DEMOTED, NOT CLOSED (review 2026-08-26).** Weller et al.
  (arXiv 2309.08541) find expansion gain anti-correlates with **retriever strength**, and it is
  cited here to close the avenue. But our *doc encoder* is strong (harmed regime) while our
  *end-to-end system* scores in BM25's class (dev 0.4795 vs 0.4525 — the helped regime), and no
  cited source covers a frozen-strong-doc-tower + bag-of-tokens-query architecture. Which "strength"
  governs is exactly the untested question. Close it on cost if at all, not on the literature. The
  cheapest test is one re-encode of the two CQADupStack dev components (~70K docs), the same price
  as the teacher probe.

- **"no clean untouched-final member is available."** The 10 unused CQADupStack subforums carry the
  same ADCS-2015 CC BY-SA evidence as the two in dev, are non-Wikipedia, and R3 measured their
  TRAIN-positive overlap at ~0 (vs 9.32% DBpedia, 11.3% FEVER). Adding one or two before the freeze
  is protocol-compatible ("default:") and would materially rescue the partition. A family-level
  "development-informed" caveat applies and must be labelled. (`results/m7_decontam.json` R3)
- **"dev cannot test long-query behaviour at all."** False for teacher-agreement metrics:
  overlap@10 and cosine need no qrels and are implemented. The mandate pre-authorises synthetic
  queries in the six's forms "including long counter-arguments". Note the 55-query slice is 54/55
  HotpotQA, so it tests long *multi-hop Wikipedia*, not long *argumentative* — n is not its only
  problem. (`LEDGER.md`)
