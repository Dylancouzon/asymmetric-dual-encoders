# COV admission records — drafts (Mac, 2026-09-01; web research, Sonnet subagent, primary sources checked)

The coverage selection surface (`instructions-m10.md` §Surfaces) admits a component only after a
pushed record of its licence at the primary source, repo and revision, sizes, qrels format, metric,
corpus-level contamination check and fingerprint screen. This file drafts the first five columns;
the fingerprint screen and the ledger entry happen on the box at M10.0-d. A Hugging Face tag is
not licence evidence; the primary-source column is what counts.

| family | component | HF repo · revision | licence at the primary source | corpus / queries / qrels | contamination vs the six and reserved | status |
|---|---|---|---|---|---|---|
| forum-technical | cqadup-programmers, cqadup-physics | `mteb/cqadupstack-*` (M7 dev, hashes in `results/m7_dev_manifest.json`) | CC BY-SA 3.0 (2014 StackExchange dump, predates the 2024 clickwrap; eval-only) | 32,176 / 876 and 38,316 / 1,039; binary | dev components since M7; different sites from reserved android/english | **admit** |
| consumer-health | MTEB MedicalQARetrieval | `mteb/medical_qa` · `a77efe81ec0c03aff7fecde742a5c9c4c46f6005` | **CC BY 4.0** at MedQuAD's repo (github.com/abachaa/MedQuAD); the HF card tags CC0, a mismatch to disclose; attribute NIH sources | 2,048 / 2,048; binary, one relevant document per query | built from 12 NIH consumer-health sites (cancer.gov, MedlinePlus, GARD…): not PubMed, not NutritionFacts, not CORD-19 | **admit**, disclose the tag mismatch |
| long technical questions | BRIGHT biology, earth-science, psychology, robotics, sustainable-living | `xlangai/BRIGHT` · `3066d29c9651a576c8aba4832d249807b181ecae` | CC BY 4.0 on the benchmark (GitHub + HF); the documents are third-party web pages cited in accepted answers, whose own rights BRIGHT does not convey — eval-only here, never redistributed | per slice: ~100 queries, corpora of a few hundred to a few thousand pages; graded qrels | queries are posts from biology/earthscience/psychology/robotics/sustainability.stackexchange — none is money.SE, none is a reserved or dev site; documents are external pages | **admit** with the rights caveat; small per-slice n, so the family is read as one macro |
| economics | BRIGHT economics | as above | as above | as above | economics.stackexchange ≠ money.stackexchange (FiQA's source); fingerprint screen against FiQA documents still runs | **admit**, same caveat |
| legal | MTEB LegalBenchCorporateLobbying | `mteb/legalbench_corporate_lobbying` · `f43436957b41692dd3e1b06a6d7116cd09f6a1db` | CC BY 4.0 (LegalBench README, John Nay) | 319 bill summaries / 340 queries; binary | congressional bills and SEC self-descriptions | **admit**; tiny corpus, read as a weak component |
| legal | MTEB LegalBenchConsumerContractsQA | `mteb/legalbench_consumer_contracts_qa` · `f9eafd458f9c61e531d4a2510d8a11dfd2282b21` | **CC BY-NC 4.0** at the LegalBench README (Kolt 2022) | 154 / 396 | — | **refused**: non-commercial |
| finance | LEDGER (artefactory) | github.com/artefactory/LEDGER; HF collection `artefactory/ledger` (revision to pin) | data CC BY 4.0, code MIT (repo LICENSE) | 4,999 annual reports / 118,048 questions; graded 0/1/2; the paper's metric is MRR, we score nDCG@10 | SEC/annual-report text, not StackExchange or Reddit; documents are long OCR'd filings, so a chunking rule must be fixed at admission | **candidate**, verify structure and pin a revision on the box |
| scientific claims | Climate-FEVER | `mteb/climate-fever` | **none at the primary source** (github.com/tdiggelm/climate-fever-dataset has no licence statement; the HF `cc-by-sa-4.0` is a wrapper tag) — the same finding that dropped it from M7's untouched set | 5.4M Wikipedia docs / 1,535 claims | — | **refused**, as in M7 |

**Families available:** forum-technical, consumer-health, long technical, economics, legal, and
finance if LEDGER verifies — five or six against the four-family floor. **Forms still without a
qrel-bearing surface:** scientific claims, paper titles, arguments (no licensed, non-contaminating
set exists; searched 2026-09-01). They are tested only by the six-set transaction.

## Generator and PAQ facts for M10.1

- **Qwen/Qwen3-8B**: Apache-2.0 (LICENSE file in the repo); main revision `b968826d9c46dd6066d109eabc6255188de91218`
  on 2026-09-01. 4-bit candidates: `pytorch/Qwen3-8B-AWQ-INT4` (7.82 GB weights — leaves ~2 GB for
  KV cache on the 10 GB card, likely too tight at batch ≥ 16), `kaitchup/Qwen3-8B-autoround-4bit-gptq`,
  `RedHatAI/Qwen3-8B-quantized.w4a16`. No report of Qwen3-8B 4-bit under vLLM on an RTX 3080 exists;
  **registered fallback: Qwen/Qwen3-4B (Apache-2.0), same prompts**, if the 8B smoke cannot hold
  batch 16 at 1,024 context. Hosted open-weights inference sidesteps this if decision 2 funds it.
- **PAQ**: data CC BY-SA, code CC BY-NC (github.com/facebookresearch/PAQ README). Download the
  official release from Facebook's file server (PAQ full: 64.9M pairs, 5.8 GB tar.gz of JSONL);
  the HF mirror `embedding-data/PAQ_pairs` is unofficial and carries no licence chain — do not use it.
  Generated by a BART-base question generator over Wikipedia with DPR/FiD consistency filtering.
