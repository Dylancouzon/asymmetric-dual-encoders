# M7 teacher shortlist, re-run under the relaxed vendor rule (2026-08-26)

Supersedes `m7-teacher-shortlist.md` (built under the strict rule, marked STALE). Four parallel
Sonnet sweeps. **Every size number was read from the model's own `config.json`, never from prose.**

> **\* Correction (Fable review, 2026-08-26): `config.json`'s `vocab_size` is not the tokenizer's.**
> All five candidates ship a **byte-identical `vocab.txt`** (sha256 `07eced375cec144d`, **30,522**
> lines, stock bert-base-uncased WordPiece). gte-large, gte-base and stella report 30,528 only
> because the embedding matrix is padded to a multiple of 64; this table originally quoted the
> padded figure. **The released table is 30,522 × dim for every candidate**, and allocating 30,528
> would ship 6 dead rows. The consequence that matters for the harness: **a teacher swap does not
> change tokenization**, so row indexing, the decontamination fingerprints and the frozen
> preprocessing rule all survive it — only the vectors change.

Filters applied: licence permits commercial release of derived weights · vendor tier per CLAUDE.md
· vocab ≤ ~50K and dim ≤ 1024 (the artifact is vocab × dim) · runs fp16 on a 10 GB RTX 3080.

> **CORRECTION, 2026-08-26 (post-review).** The vocab column below is `config.json`'s
> `vocab_size`, which is the **padded embedding-matrix width, not the tokenizer size**. All five
> BERT-WordPiece candidates (bge-base, bge-large, gte-large, stella, arctic-embed-l) ship a
> **byte-identical `vocab.txt`** — sha256 `07eced375cec144d`, 30,522 lines — and gte-large and
> stella merely pad to 30,528 (a multiple of 64). **The released table is 30,522 × dim for every
> one of them**, so the int8 column is right but the vocab column overstates two rows by 6.
> Pinned in `m7src/test_encoders.py`, which now checks each Spec against the real tokenizer.
> A useful consequence: **a teacher swap among these does not change tokenization**, so row
> indexing, the decontamination fingerprints and the frozen preprocessing rule all survive it —
> only the vectors change.

Scores are **MTEB v1 English Retrieval (15-task BEIR average)** — the only scale
`results/m7_calibration.json` is fitted on. MTEB v2 / MTEB(eng,v2) numbers are several points
higher and are *not* interchangeable; where only v2 exists it is labelled.

## Survivors, best first

| model | MTEB v1 Ret | vocab | dim | params | licence | vendor tier | int8 table |
|---|---|---|---|---|---|---|---|
| NovaSearch/stella_en_400M_v5 | **58.97** | 30,522\* | 1024 | 435M | MIT | **clean** | 31.3 MB |
| Alibaba-NLP/gte-large-en-v1.5 | **57.91** | 30,522\* | 1024 | 434M | Apache-2.0 | justify (Alibaba) | 31.3 MB |
| Snowflake/snowflake-arctic-embed-l | 55.98 | 30,522 | 1024 | 335M | Apache-2.0 | justify-max (Snowflake) | 31.3 MB |
| Alibaba-NLP/gte-modernbert-base | 55.33 | 50,368 | 768 | 149M | Apache-2.0 | justify (Alibaba) | 38.7 MB |
| Snowflake/snowflake-arctic-embed-m-v1.5 | 55.14 | 30,522 | 768 | 109M | Apache-2.0 | justify-max | 23.4 MB |
| BAAI/bge-large-en-v1.5 | 54.29 | 30,522 | 1024 | 335M | MIT | clean | 31.3 MB |
| Alibaba-NLP/gte-base-en-v1.5 | 54.09 | 30,522* | 768 | 137M | Apache-2.0 | justify (Alibaba) | 23.4 MB |
| BAAI/bge-base-en-v1.5 *(current)* | 53.25 | 30,522 | 768 | 109M | MIT | clean | 23.4 MB |
| ibm-granite/granite-embedding-english-r2 | 53.10 | 50,368 | 768 | 149M | Apache-2.0 | justify (IBM) | 38.7 MB |

**The Snowflake question is moot.** Snowflake's best filter-passing model (arctic-embed-l, 55.98)
is ~2 points below two admissible alternatives, so we never need the "justify-max" tier and never
need to ask Dylan for it. Recorded because the sweep was commissioned partly to answer it.
For the record if it ever matters: Cortex Search's *documented default* embedding model is
`snowflake-arctic-embed-m-v1.5`, which is as close to core business as this tier gets.

**granite-embedding-english-r2 ties the current teacher** (53.10 vs 53.25) on our anchor scale and
so is not an upgrade, despite leading on MTEB v2 Retrieval (56.4). **Dead on quality alone** — an
earlier version of this file also dinged it for a 50,368 vocab while listing `gte-modernbert-base`
with the *identical* 50,368 ModernBERT vocab as a survivor. That was inconsistent; 50,368 is a real
tokenizer size for both, it is 0.7% over a "~50K" guideline, and it disqualifies neither. Neither is
a quality upgrade, which is the reason both are out.

## Disqualified, with the reason (so nobody re-derives these)

| model | killed by |
|---|---|
| snowflake-arctic-embed-l-v2.0 / -m-v2.0 | vocab 250,002 / 250,048 (XLM-R, GTE-multilingual). MRL truncates *dim*, never vocab, so its near-free 256d curve (0.556→0.547) cannot rescue it. m-v2.0 also needs `trust_remote_code` + xformers. |
| intfloat/multilingual-e5-large-instruct, BAAI/bge-m3 | vocab 250,002 |
| Qwen/Qwen3-Embedding-0.6B | vocab 151,669 (verified against the safetensors header, *not* the 151,936 that circulates). At MRL-256 the table would be an acceptable 38.8 MB int8, so bytes are not the real objection — it is **dominated**: 55.52 v1 (reconstructed from the official results repo) is below gte-large and stella at full dim, and truncation can only lower it. **No published MRL curve exists at any dim** — the paper's "MRL Support: Yes" is a capability flag with no ablation behind it. |
| Qwen3-Embedding-4B | vocab 151,665, dim 2560 |
| intfloat/e5-large-v2 (50.56), e5-base-v2 (~50.3) | below the current teacher despite 3x the params; also need `query:`/`passage:` prefixes on **docs** as well as queries |
| nvidia/llama-nemotron-embed-1b-v2, Nemotron-3-Embed-1B | vocab 128,256 / 131,072 and dim 2048 |
| nvidia/llama-embed-nemotron-8b, Salesforce SFR-Embedding-*, Linq-Embed-Mistral, LGAI-Embedding-Preview | licence: research/non-commercial only (LGAI's Apache tag covers code; the *weights* are CC-BY-NC) |
| TencentBAC/Conan-embedding-v1 | CC-BY-NC, and Chinese-only |
| TencentBAC/Conan-embedding-v2 | vocab 150,000, dim 3584 |
| HIT-TMG/KaLM-embedding-* | vocab 151,936 (Qwen2 backbone) |
| ByteDance Seed1.5-Embedding | weights never released (API only) |
| jhu-clsp/mmBERT-base, lightonai/mDenseOn | vocab 256,000 |
| WhereIsAI/UAE-Large-V1 (54.66), GIST-large (52.31) | bge-large fine-tunes that do not clearly beat their own base |
| google/embeddinggemma-300m | Gemma terms |

**BAAI English BGE v2/v3 dense successor: searched again, none exists.** BGE-M3 is multilingual on
a 250K vocab. Treat as verified within the limits of the search, not certain.

## Two near-misses worth knowing about

Both clear licence + vendor + size and fail only on *released* quality — the strong number belongs
to weights that were never published. Neither is a teacher swap; both would be training projects.

- **jhu-clsp/ettin-encoder-400m** (MIT, 50,368 × 1024). Released checkpoint is base pretraining,
  not a contrastive retriever: 45.7–48.4 MTEB **v2** Retrieval, ~10 points under bar.
- **chandar-lab/NeoBERT** (MIT, 30,522 × 768, 250M). Base MLM only. The paper's CDE fine-tune
  reaches **56.37 MTEB v1 Retrieval** — a documented existence proof on a clean small backbone —
  but no fine-tuned checkpoint is on the Hub.

\* `config.json` says 30,528; the tokenizer is 30,522. See the correction above.

## Live caveats on the two front-runners

- **stella: ArguAna and FiQA2018 are recorded as in-domain training data** (MTEB registry
  `training_datasets`, community-maintained, *not* an author disclosure — the model card says the
  training data "will be released in the future" and never did). Those are **2 of our 6 final eval
  datasets**, and NQ/HotpotQA/FEVER/MSMARCO are on the same list, which covers **2 of our 4
  text-backed dev components** too. Our comparators do not carry the same flag on ArguAna/FiQA.
  This is a defensibility risk on the headline number, not a quality risk.
- **stella reproduction:** discussion #21 (an MTEB maintainer) found large gaps on
  classification/clustering (AmazonCounterfactual 92.36 → 72.59) but **small ones on retrieval**
  (SCIDOCS −1.08, SciFact −0.27 — both are our datasets). Author acknowledged, promised eval
  scripts, never delivered.
- **stella provenance confirmed**: the card states outright it is trained from
  `Alibaba-NLP/gte-large-en-v1.5`. NovaSearch is a 3-person community org with no product, so the
  *releasing* party is clean — but the weights are a derivative of the #2 candidate. Under the
  relaxed rule Alibaba is admissible anyway, so this is no longer a blocker either way.
- **stella needs its post-pooling Dense head.** Its `modules.json` is
  Transformer → Pooling(**mean**) → `2_Dense_1024`, and that Dense is 1024→1024 **with bias**
  (Identity activation). Encoding with plain mean pooling omits a trained affine map and is
  therefore a *different model* from the one that scored 58.97 — the same loader-mismatch class as
  the M2 potion blocker. Handled by `Spec.post_dense`; verified at the pinned revision.
- **gte-large-en-v1.5** needs `trust_remote_code=True`; `unpad_inputs` and
  `use_memory_efficient_attention` both default to **false**, so it runs on plain sdpa and
  **xformers is an optional accelerator, not a dependency**. Needs `transformers>=4.41`. Ships fp32.
- **bge-family reproduction flag** (mteb issue #1912, unresolved): large self-reported-vs-reproduced
  gaps on MSMARCO and NQ for bge-base/large/small. Does not touch the BEIR-15 aggregates we anchor
  on (independently reproduced by Snowflake's own comparison table to the same 53.25), and it
  applies to our *current* teacher, so it is not a risk introduced by a swap.

## Why this table does not by itself pick the teacher

`results/m7_calibration.json`: the MTEB → six-set map has ratio spread 0.926–1.001 across the nine
models we measured, affine residual sd 0.0102. The stella–gte-large gap (1.06 MTEB ≈ 0.009 on the
six) is **inside that noise**, so this ranking cannot separate the top two. `m7src/teacher_probe.py`
measures them directly on the two CQADupStack dev components — the only dev components on no
candidate's disclosed training list — which is the number that should decide it.
