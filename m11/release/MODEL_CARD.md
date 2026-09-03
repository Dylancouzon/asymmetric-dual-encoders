---
license: mit
language: en
library_name: numpy
tags:
  - retrieval
  - sentence-similarity
  - asymmetric-dual-encoder
  - edge
  - quantized
base_model: NovaSearch/stella_en_400M_v5
pipeline_tag: feature-extraction
---

# zero — a query encoder with no transformer in it

`zero` is the query side of an **asymmetric dual encoder**. Documents are indexed once, in the
cloud, by a large frozen encoder ([`NovaSearch/stella_en_400M_v5`](https://huggingface.co/NovaSearch/stella_en_400M_v5),
1024-d). Queries are encoded on the edge by **this table**: 30,522 × 1024 int8 rows and one
pooling rule. Encoding a query is a gather and a weighted sum — **no transformer, no matmul,
no GPU**. Sub-millisecond per query on one CPU core; the whole query asset is **31.8 MB**.

It was distilled from stella so that its output lands in stella's document space. Cosine
similarity against stella document vectors is the score.

> **Research preview.** This model **missed its own release bar** — see [Results](#results).
> It is published so the architecture can be tested, not as a recommended drop-in retriever.
> Read the results section before using it for anything.

## Usage

The query side needs `numpy` and `tokenizers`. That is the entire runtime.

```python
# pip install numpy tokenizers huggingface_hub
from huggingface_hub import snapshot_download
import sys

d = snapshot_download("REPO_ID")          # ~94 MB
sys.path.insert(0, d)
from zero_encoder import ZeroQueryEncoder

enc = ZeroQueryEncoder(d, variant="int8")           # or "fp16"
q = enc.encode(["how do mrna vaccines work?"])      # (1, 1024), L2-normalized
```

Documents are encoded by the frozen teacher — **pin the revision**, the table is only valid
against this exact document space:

```python
# pip install sentence-transformers
from sentence_transformers import SentenceTransformer
import numpy as np

doc_model = SentenceTransformer(
    "NovaSearch/stella_en_400M_v5",
    revision="ffeb2b7ee715c226d4ffe5e4619f7dbb48624c20",
    trust_remote_code=True,
    # required unless xformers is installed; also the pinned setting the table was distilled under
    config_kwargs={"use_memory_efficient_attention": False, "unpad_inputs": False},
)
docs = [
    "mRNA vaccines deliver a strand of messenger RNA encoding a viral antigen.",
    "The Treaty of Westphalia ended the Thirty Years' War in 1648.",
]
D = doc_model.encode(docs, normalize_embeddings=True)   # no prefix on the document side

scores = q @ D.T
print(docs[int(np.argmax(scores))])
```

That asymmetry is the point: `doc_model` runs once per document, in the cloud. `enc` runs on
every query, on the device, and costs almost nothing.

## Using it with Qdrant

The output is an ordinary 1024-d dense vector, so no special handling is needed. Both sides are
L2-normalized, which means `DOT` ranks identically to `COSINE` and is cheaper.

```python
from qdrant_client import QdrantClient, models

client = QdrantClient(":memory:")           # or your cluster
client.create_collection("docs", vectors_config=models.VectorParams(
    size=1024, distance=models.Distance.DOT))
client.upsert("docs", points=[
    models.PointStruct(id=i, vector=D[i].tolist(), payload={"text": t})
    for i, t in enumerate(docs)])           # D from the document encoder above

hits = client.query_points("docs", query=q[0].tolist(), limit=5).points
print(hits[0].payload["text"])          # q is the (1, 1024) array from Usage above
```

**On the edge, put the table in the store too.** A second collection holds one point per vocab
row, created with `hnsw_config=models.HnswConfigDiff(m=0)` — it is retrieve-by-id only, and
indexing it inflated the shard from 466 MB to 1.82 GB for no benefit. The query path becomes
tokenize → fetch rows by id → pool → search, with no model weights in your process at all.

**If you fuse with BM25, the fusion rule matters.** The system that ties OpenSearch (0.4911)
uses **convex score fusion at w=0.8**, not RRF: on development sets RRF scored 0.5504 against
convex's 0.5727. Qdrant's native `Fusion.RRF` is therefore a *different, weaker* operating point,
not the published one — combine the scores yourself to reproduce it. Dense-only in Qdrant
reproduces 0.4339 exactly **with exact search** — the HNSW advice above is about cost, and ANN
recall is a separate confound the published numbers deliberately avoid.

Scalar-quantizing the document index to int8 halves it to ~1.02 GB per 1M vectors, but **that
was never measured for quality here** — treat it as untested. (The int8 quality-free result
above is about the *query table*, not the document index.)

### The rule, if you reimplement it

Tokenize with the bundled WordPiece tokenizer (`add_special_tokens=True`, truncate at 512,
**no padding**, no prefix). A token appearing `c` times in the query carries **total weight
`sqrt(c)`**, not `c`. Sum the rows, divide by the weight sum, L2-normalize. An empty or
near-zero-norm bag falls back to the normalized `[CLS]` row (id **101**; row 0 is `[PAD]`).
`zero_encoder.py` is 89 lines and is the reference. `config.json` carries the rule and its
fingerprint (`adb24fb2e8cad66f`).

## Files

| file | what |
|---|---|
| `model.npz` | `rows_int8` (30522×1024) + `int8_scale`, and `rows_fp16` for reference |
| `config.json` | the frozen preprocessing rule, teacher pin, document-encoder spec, shas |
| `zero_encoder.py` | the whole query path — numpy + tokenizers, no torch |
| `tokenizer.json`, `vocab.txt`, … | stella's WordPiece tokenizer, copied at the pinned revision — with the two edits below |

**Two deliberate edits to the copied tokenizer files.** stella's own files declare
`model_max_length: 32768`, `max_length: 8000` and fixed-512 padding; the frozen rule here — and
the document index — are **512 with no padding**. This bundle ships `model_max_length: 512`,
`max_length: 512` and `padding: null`. `config.json` records the original values under
`tokenizer_deviation_from_teacher`.

Precisely what that changes, by caller:

| caller | before | after |
|---|---|---|
| `zero_encoder.py` (the reference path) | 512, no padding | unchanged — **byte-identical output**; it calls `no_padding()` and reads truncation from `config.json` |
| `tokenizers` directly | every `encode` padded to 512 | ragged unless you enable padding yourself |
| `transformers` | `model_max_length` 32768, so `truncation=True` with no explicit length truncated at 32768 | 512 — but truncation and padding still happen only when the *call* asks for them |
| Sentence Transformers | requests truncation itself and may impose its own `max_seq_length` | unchanged in that respect; only the underlying default moves to 512 |
| fastembed's tokenizer loader | truncation 8000, fixed-512 padding kept | truncation 512, dynamic padding |

The reference path is unaffected either way, so no published number changes; only what a
third-party loader does with the same files. The fastembed row is about
`fastembed.common.preprocessor_utils.load_tokenizer` only — **this repo ships no ONNX graph**, so
there is no fastembed inference path to describe yet.

Per-token learned weights are **folded into the rows**, so the int8 artifact is self-contained.
`int8` is the variant every published number below was measured on; it was measured
quality-free against `fp16` (upper bound **0.00013** nDCG@10).

## Results

nDCG@10, exact search (no ANN, so recall is not a confound), on six BEIR datasets. These are
**confirmatory** numbers from a single pre-registered run against frozen comparator vectors —
the system was frozen (sha `a7007b1a…`) before any of them were observed.

| system | arguana | fiqa | nfcorpus | scidocs | scifact | trec-covid | **avg-6** |
|---|---|---|---|---|---|---|---|
| **`zero` (int8)** | 0.5916 | 0.3728 | 0.3124 | 0.1677 | 0.6101 | 0.5490 | **0.4339** |
| `zero` + BM25, fused | 0.5975 | 0.4026 | 0.3497 | 0.1881 | 0.7068 | 0.7018 | **0.4911** |
| BM25 alone | 0.4878 | 0.2532 | 0.3180 | 0.1565 | 0.6791 | 0.6099 | 0.4174 |
| stella, symmetric (the teacher ceiling) | 0.6369 | 0.5536 | 0.4134 | 0.2395 | 0.7796 | 0.8234 | 0.5744 |

The three registered comparisons, paired bootstrap + sign-flip, Holm-corrected across the family:

| | Δ | 95% CI | verdict |
|---|---|---|---|
| `zero` > LightRetriever dense (0.4583) | **−0.0243** | [−0.0405, −0.0086] | **miss, resolved below the bar** |
| `zero` > BM25 (0.4174) | +0.0165 | [+0.0017, +0.0311] | CI passes, multiplicity does not — unresolved |
| fused > OpenSearch sparse (0.4868) | +0.0043 | [−0.0063, +0.0151] | **statistical tie** |

**What that means.** A pure lookup table retains **75.5%** of its teacher's quality on these six
sets at ~1/1000 of the query-side cost, and beats BM25 on average — but it does **not** beat a
comparable small dense query encoder, and the honest headline is the miss. The one bright spot
is fusion: `zero` + BM25 is a **statistical tie with OpenSearch's learned sparse retriever**
while its query side remains a table lookup plus token counts.

### Caveats you should read before trusting a number

- **Teacher contamination.** stella discloses **ArguAna** and **FiQA** in its training data —
  exactly this system's two strongest datasets. On the four sets with no disclosed overlap
  (nfcorpus, scidocs, scifact, trec-covid), `zero` is **below BM25** (−0.0311 [−0.0517, −0.0109]).
- **Distribution.** Training was Wikipedia- and e-commerce-shaped. Retention is 0.915 on
  in-distribution development sets and 0.755 on these six. Expect the low number out of domain.
- **English only**, 512 wordpieces, WordPiece-30522 vocab. Out-of-vocabulary terms degrade to
  their subword rows; heavily fragmented queries are where the gap with a real encoder is widest.
- The table is a **bag of tokens**. Word order, negation and syntax are not represented at all.

## Costs

| | |
|---|---|
| query asset (int8 rows + scales + tokenizer) | **31.8 MB** |
| query encode, batch 1, one CPU core | **0.38 ms** (`zero_encoder.py` measures ~0.07 ms) |
| hydration (cold load to first query) | **0.22 s** |
| document index, 1024-d fp16 | 2.05 GB per 1M documents |
| document index, 1024-d int8 | 1.02 GB per 1M documents |

For reference at the document side: LightRetriever 3.07, OpenSearch sparse 1.40,
bge-small 0.77 GB/1M. `zero`'s document index is not cheap — the trade is all on the query side.

## Training

L2 regression of the table's pooled output onto stella's query embeddings, over 340,850
approved pairs plus 220,632 query-text-only rows. Sources: **Amazon ESCI** (Apache-2.0),
**FEVER**, **HotpotQA**, **SQuAD**, **NQ-open**, **TriviaQA**, **Mr. TyDi (en)**.

**MS MARCO is permanently excluded** from this lineage — its terms forbid commercial use.
(Measured cost of that exclusion: +0.0058 [−0.0015, +0.0131] avg-6 had it been included, which
still misses the bar. The gap is architectural, not licensing.)

### Attribution

NQ, SQuAD, HotpotQA, FEVER and Mr. TyDi derive from Wikipedia and are **CC BY-SA** (3.0/4.0).
Amazon ESCI is Apache-2.0. The teacher, `NovaSearch/stella_en_400M_v5`, is MIT.

## Revisions

| | |
|---|---|
| first published | 2026-09-03 — the frozen M7 bundle, with stella's tokenizer files copied verbatim |
| this revision | 2026-09-03, commit `1aa60418` — `tokenizer_config.json` `model_max_length`/`max_length` 32768/8000 → 512, `tokenizer.json` `padding` `Fixed(512)` → `null`, and one broken snippet in this card fixed |

`model.npz` is byte-identical across both (sha `a7007b1a…`) and the reference encoder's output is
unchanged, so **no published number differs between revisions**. Pass `revision=` to
`snapshot_download` if you need to pin one.

## Citation / provenance

```
run_id             p35w-2m-s2500
table sha256       a7007b1a6af120b976f093fd69ddcb5001996ec0b84b5864b4fd25d7af878abf
teacher            NovaSearch/stella_en_400M_v5 @ ffeb2b7ee715c226d4ffe5e4619f7dbb48624c20
preproc            prefix="" · add_special_tokens · max_length=512 · pool_mode=sqrt
preproc fingerprint adb24fb2e8cad66f
```
