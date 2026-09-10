# COV admission records — drafts (Mac, 2026-09-01; web research, Sonnet subagent, primary sources checked)

The coverage selection surface (`instructions-m10.md` §Surfaces) admits a component only after a
pushed record of its licence at the primary source, repo and revision, sizes, qrels format, metric,
corpus-level contamination check and fingerprint screen. This file drafts the first five columns;
the fingerprint screen and the ledger entry happen on the GPU instance at M10.0-d. A Hugging Face tag is
not licence evidence; the primary-source column is what counts.

| family | component | HF repo · revision | licence at the primary source | corpus / queries / qrels | contamination vs the six and reserved | status |
|---|---|---|---|---|---|---|
| (DEV, not COV) | cqadup-programmers, cqadup-physics | `mteb/cqadupstack-*` (M7 dev) | CC BY-SA 3.0 (2014 dump) | 32,176 / 876 and 38,316 / 1,039 | dev since M7; **scored by the M10 Mac diagnostics (86 raw reads)**, so not an untouched selection surface | **stay in DEV-6**, reported beside every COV read |
| consumer-health | MTEB MedicalQARetrieval | `mteb/medical_qa` · `a77efe81ec0c03aff7fecde742a5c9c4c46f6005` | **CC BY 4.0** at MedQuAD's repo (github.com/abachaa/MedQuAD); the HF card tags CC0, a mismatch to disclose; attribute NIH sources | 2,048 / 2,048; binary, one relevant document per query | built from 12 NIH consumer-health sites (cancer.gov, MedlinePlus, GARD…): not PubMed, not NutritionFacts, not CORD-19 | **admit**, disclose the tag mismatch |
| BRIGHT (one family, six slices) | biology, earth-science, economics, psychology, robotics, sustainable-living | `xlangai/BRIGHT` · `3066d29c9651a576c8aba4832d249807b181ecae` | CC BY 4.0 on the benchmark at its primary source (GitHub + paper). The documents are third-party web pages cited in accepted answers; BRIGHT does not convey their rights. **Standard applied:** the dataset-level licence at the primary source, the same standard that admitted CQADupStack and the six (whose documents are also third-party text); the caveat is disclosed; evaluation-only, never redistributed | per slice ~100 queries, corpora of hundreds to thousands of pages; graded qrels; slices averaged into one family macro | queries are StackExchange posts from six sites; none is money.SE (FiQA's source) or a reserved/dev site; documents are external pages; fingerprint screen against the six still runs | **admit** as one family, caveat disclosed. **Reviewer dissent (Codex passes 5 and 6):** exclude unless per-document processing rights are evidenced. **Planning decision:** the per-document standard would also disqualify the six and CQADupStack; the dataset-level standard is applied consistently and disclosed. Dylan may overrule |
| legal | MTEB LegalBenchCorporateLobbying | `mteb/legalbench_corporate_lobbying` · `f43436957b41692dd3e1b06a6d7116cd09f6a1db` | CC BY 4.0 (LegalBench README, John Nay) | 319 bill summaries / 340 queries; binary | congressional bills and SEC self-descriptions | **admit**; tiny corpus, read as a weak component |
| legal | MTEB LegalBenchConsumerContractsQA | `mteb/legalbench_consumer_contracts_qa` · `f9eafd458f9c61e531d4a2510d8a11dfd2282b21` | **CC BY-NC 4.0** at the LegalBench README (Kolt 2022) | 154 / 396 | — | **ADMITTED 2026-09-04**: Dylan's rule of that date makes a non-commercial licence admissible for validation, and COV is a selection surface that never enters training. It restores a second legal set and takes the family count to four without LEDGER. The verdict was **refused** until then, on licence alone |
| finance | LEDGER (artefactory) | github.com/artefactory/LEDGER; HF collection `artefactory/ledger` (revision to pin) | data CC BY 4.0, code MIT (repo LICENSE) | 4,999 annual reports / 118,048 questions; graded 0/1/2; the paper's metric is MRR, we score nDCG@10 | SEC/annual-report text, not StackExchange or Reddit; documents are long OCR'd filings, so a chunking rule must be fixed at admission and the total capped at 100K chunks (≈ 8 min of stella encode) or the component is dropped | **candidate**, verify structure and pin a revision at admission |
| scientific claims | Climate-FEVER | `mteb/climate-fever` | **none at the primary source** (github.com/tdiggelm/climate-fever-dataset has no licence statement; the HF `cc-by-sa-4.0` is a wrapper tag) — the same finding that dropped it from M7's untouched set | 5.4M Wikipedia docs / 1,535 claims | — | **refused**, as in M7 |

**Families available (untouched by any M10 decision):** consumer-health, BRIGHT, legal (now two
components, one family) and finance if LEDGER verifies — **three family IDs without LEDGER, exactly the
floor; four iff LEDGER admits** (Codex 2026-09-04 corrected an earlier "four"). The CQADupStack pair stays in DEV-6.
**LEDGER (2026-09-04, after the Codex pass):** an ordinary admission — admitted if its structure
verifies at M10.0-d (chunk rule, 100K cap), decided before any read; at 118,048 questions it is the
one candidate large enough to move the surface's power. Amendment A4's conditional-remedy role is struck.
**Forms without a published qrel-bearing surface:** scientific claims, paper titles, arguments.
Paper titles get a constructed *diagnostic* surface, `arxiv-title` (mandate §Surfaces B4: 2,000
held-out titles → own abstract among 100K; descriptive, no action — the Opus pass showed a null there
could be a floor effect); claims and arguments are tested only by the six-set transaction. **CUREv1**
(biomedical, PMC-OA full text, CC BY-NC, 2,000 real clinician queries, Qwen-annotated pools) is
decision 12 — **adopted 2026-09-04 as a reported diagnostic**, never selection-bearing.

## Generator and PAQ facts for M10.1

- **Qwen3-8B** (Apache-2.0). **Decision 14 (2026-09-04): the generator is Qwen's official 4-bit release
  `Qwen/Qwen3-8B-AWQ`, revision `4da05a8edb55…`, served by vLLM on the box** (≈6 GB on the 10 GB card);
  the bf16 base (`Qwen/Qwen3-8B`, revision `b968826d9c46…`) served by the same vLLM contract on a rented
  GPU after Dylan returns is the only fallback (never a third-party API — it could not honour the pinned
  revision and per-request seed), firing if vLLM will not run on the card, the throughput floor is
  unmet, or a form fails its gates twice. The earlier bf16-only rule
  was written for an 80 GB rented card, not for quality; the smoke rubric and the A8 diversity gate
  are the quality guards either way. Provider and served revision go in the manifest if the fallback fires.
- **PAQ**: data CC BY-SA, code CC BY-NC (github.com/facebookresearch/PAQ README). Download the
  official release from Facebook's file server (PAQ full: 64.9M pairs, 5.8 GB tar.gz of JSONL);
  the HF mirror `embedding-data/PAQ_pairs` is unofficial and carries no licence chain — do not use it.
  Generated by a BART-base question generator over Wikipedia with DPR/FiD consistency filtering.
