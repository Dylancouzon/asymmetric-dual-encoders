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
| Qwen3-Embedding-0.6B as teacher | **Dominated, not disqualified on vocab.** Verified vocab 151,669 (safetensors header); at MRL-256 the table is an acceptable 38.8 MB int8, so bytes were never the objection. But 55.52 v1 is below both front-runners at *full* dim and no MRL curve is published at any dim, so truncation can only lower it. | shortlist 2026-08-26 |
| the Snowflake "justify-max" vendor tier | **Moot.** Snowflake's best filter-passing model (arctic-embed-l, 55.98) is ~2 points below two admissible alternatives, so the tier is never needed and Dylan never has to rule on it. | shortlist 2026-08-26 |
| granite-embedding-english-r2 as teacher | Ties the current teacher on our anchor scale (53.10 vs 53.25); its lead is only on the non-comparable MTEB v2. Vocab is also 368 over the line. | shortlist 2026-08-26 |
| a 2025–2026 teacher we had not seen | Swept; nothing survives. NVIDIA/Tencent/KaLM/mmBERT die on 128K–256K vocab; Salesforce/Linq/LGAI/NVIDIA-8b on non-commercial weights; ByteDance never released weights. Near-misses `ettin-encoder-400m` and `NeoBERT` clear every structural filter but their strong numbers belong to unreleased fine-tunes. | shortlist 2026-08-26 |

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
