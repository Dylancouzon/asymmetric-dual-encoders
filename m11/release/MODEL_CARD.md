---
license: mit
language: en
tags:
  - fastembed
  - qdrant
  - onnx
  - retrieval
  - sentence-similarity
  - asymmetric-dual-encoder
  - edge
base_model: NovaSearch/stella_en_400M_v5
pipeline_tag: feature-extraction
---

# constella-zero — a query encoder with no transformer in it

`constella-zero` is the query side of an **asymmetric dual encoder**, built to be served with
[**FastEmbed**](https://github.com/qdrant/fastembed) and [**Qdrant**](https://qdrant.tech).

Documents are indexed once, in the cloud, by a large frozen encoder
([`stella_en_400M_v5`](https://huggingface.co/NovaSearch/stella_en_400M_v5), 1024-d — published
here as [`DylanCouzon/stella-en-400M-v5-doc-onnx`](https://huggingface.co/DylanCouzon/stella-en-400M-v5-doc-onnx)).
Queries are encoded on the device by **this table**: 30,522 × 1024 int8 rows and one pooling rule.

Encoding a query is a gather and a weighted sum — **no transformer, no matmul, no GPU**. The
whole query asset is **31.8 MB** and a query costs about **0.05 ms** on one CPU core.

> **Research preview.** Read [Results](#results) and
> [What it cannot do](#what-it-cannot-do) before using this. It is published so the architecture
> can be tested; it is a bag-of-tokens model and it behaves like one.

## Quickstart — FastEmbed

```python
from fastembed import TextEmbedding

NAME = "REPO_ID"
query_model = TextEmbedding(NAME)
q = next(iter(query_model.embed(["how do mrna vaccines work?"])))   # (1024,), L2-normalized
```

Support for this model is in review upstream. Until it is released, install FastEmbed from the
branch that carries it:

    pip install "fastembed @ git+https://github.com/Dylancouzon/fastembed@add-constella-models"

FastEmbed downloads only `model.onnx` and the tokenizer — about 31 MB, not the whole repo — and
serves the graph untouched: pooling and L2 normalization happen **inside** the graph, under the
rule described below, which is not a masked mean.

## The pair — encoding documents

The table only means anything against document vectors from the encoder it was distilled from, so
that encoder is published in the same ONNX form and served the same way:

```python
DOC_NAME = "DylanCouzon/stella-en-400M-v5-doc-onnx"

doc_model = TextEmbedding(DOC_NAME)          # 1.75 GB, runs in the cloud, once per document
docs = [
    "mRNA vaccines deliver a strand of messenger RNA encoding a viral antigen.",
    "The Treaty of Westphalia ended the Thirty Years' War in 1648.",
]
D = list(doc_model.embed(docs))
```

That asymmetry is the whole point: `doc_model` is a 400M-parameter transformer that runs once per
document, in the cloud. `query_model` runs on every query, on the device, and costs almost nothing.

## Search with Qdrant

Use `COSINE`. Qdrant implements cosine *as* a dot product — it normalizes vectors once, on upsert,
and compares with dot at query time — so `COSINE` costs the same as `DOT` here and does not depend
on the caller having preserved unit norm.

```python
# pip install qdrant-client
from qdrant_client import QdrantClient, models

client = QdrantClient(":memory:")           # or your cluster
client.create_collection("docs", vectors_config=models.VectorParams(
    size=1024, distance=models.Distance.COSINE))
client.upsert("docs", points=[
    models.PointStruct(id=i, vector=D[i].tolist(), payload={"text": t})
    for i, t in enumerate(docs)])

hits = client.query_points("docs", query=q.tolist(), limit=5).points
print(hits[0].payload["text"])
```

**On the edge, put the table in the store too.** A second collection holds one point per vocab
row, created with `hnsw_config=models.HnswConfigDiff(m=0)` — it is retrieve-by-id only, and
indexing it inflated the shard from 466 MB to 1.82 GB for no benefit. The query path becomes
tokenize → fetch rows by id → pool → search, with no model weights in your process at all.

**If you fuse with BM25, the fusion rule matters.** The fused number below uses **convex score
fusion at w=0.8**, not RRF: on development sets RRF scored 0.5504 against convex's 0.5727.
Qdrant's native `Fusion.RRF` is a *different, weaker* operating point — combine the scores
yourself to reproduce the number quoted here.

Scalar-quantizing the document index to int8 halves it to ~1.02 GB per 1M vectors, but **that was
never measured for quality here** — treat it as untested. (The int8 result quoted below is about
the *query table*, not the document index.)

## Other runtimes

The repo ships three interchangeable forms of the same table. **You need exactly one of them.**

| file | for | size |
|---|---|---|
| `model.onnx` | FastEmbed, or any ONNX runtime — pooled + normalized, `(b, 1024)` | 31 MB |
| `model_tokens.onnx` | pipelines that insist on doing their own masked-mean pooling, `(b, s, 1024)` | 31 MB |
| `model.npz` | the numpy reference path, `zero_encoder.py` | 94 MB |

```python
# pip install numpy tokenizers huggingface_hub
from huggingface_hub import snapshot_download
import sys, numpy as np

d = snapshot_download("REPO_ID")
sys.path.insert(0, d)
from zero_encoder import ZeroQueryEncoder

enc = ZeroQueryEncoder(d, variant="int8")            # or "fp16"
q_np = enc.encode(["how do mrna vaccines work?"])    # (1, 1024), L2-normalized
assert np.abs(q_np[0] - q).max() < 1e-5              # same vector FastEmbed just produced
```

`zero_encoder.py` is 89 lines, needs only `numpy` and `tokenizers`, and is the reference
implementation — every number below was measured through it.

Both ONNX graphs are opset 17, standard operators only, and carry the table as an **int8
initializer with a per-row fp32 scale** dequantized in-graph, so each is ~31 MB rather than the
125 MB fp32 rows would cost.

## How it works

Tokenize with the bundled WordPiece tokenizer (`add_special_tokens=True`, truncate at 512, **no
padding**, no prefix). A token appearing `c` times in the query carries **total weight `sqrt(c)`**,
not `c` — so repetition saturates. Sum the rows, divide by the weight sum, L2-normalize. An empty
or near-zero-norm bag falls back to the normalized `[CLS]` row (id **101**; row 0 is `[PAD]`).

That count saturation is why the graph does not pool with a masked mean, and why FastEmbed passes
its output through untouched. `config.json` carries the rule and its fingerprint
(`adb24fb2e8cad66f`).

Per-token learned weights are **folded into the rows**, so the int8 artifact is self-contained.
`int8` is the variant every number below was measured on; it was measured quality-free against
`fp16` (upper bound **0.00013** nDCG@10).

### Tokenizer files

stella's own files declare `model_max_length: 32768`, `max_length: 8000` and fixed-512 padding.
The rule here — and the document index — are **512 with no padding**, so this bundle ships
`model_max_length: 512`, `max_length: 512` and `padding: null`. `config.json` records the
originals under `tokenizer_deviation_from_teacher`.

This matters for any loader that honours those fields, FastEmbed's included: shipped unedited,
a 513–8000 token input would be tokenized under a rule the document index was never built with.
The numpy reference path is unaffected either way — it calls `no_padding()` and reads truncation
from `config.json` — so **no published number depends on this edit**.

## Results

nDCG@10 on six BEIR datasets, with exact search so that ANN recall is not a confound. Measured
once, on the frozen table shipped here (sha `a7007b1a…`).

| system | arguana | fiqa | nfcorpus | scidocs | scifact | trec-covid | **average** |
|---|---|---|---|---|---|---|---|
| **`constella-zero` (int8)** | 0.5916 | 0.3728 | 0.3124 | 0.1677 | 0.6101 | 0.5490 | **0.4339** |
| `constella-zero` + BM25, convex fusion | 0.5975 | 0.4026 | 0.3497 | 0.1881 | 0.7068 | 0.7018 | **0.4911** |
| BM25 alone | 0.4878 | 0.2532 | 0.3180 | 0.1565 | 0.6791 | 0.6099 | 0.4174 |
| the teacher, used symmetrically (ceiling) | 0.6369 | 0.5536 | 0.4134 | 0.2395 | 0.7796 | 0.8234 | 0.5744 |

A pure lookup table retains **75.5%** of the teacher's quality on these six sets at roughly 1/1000
of the query-side cost. Fusing it with BM25 recovers a good deal of the rest, and the query side is
still a table lookup plus token counts.

### What it cannot do

- **Teacher contamination.** stella discloses **ArguAna** and **FiQA** in its training data —
  exactly this model's two strongest datasets above. On the four sets with no disclosed overlap
  (nfcorpus, scidocs, scifact, trec-covid) it averages **0.4098 against BM25's 0.4409** — i.e.
  below BM25. Weight the headline average accordingly.
- **It is a bag of tokens.** Word order, negation and syntax are not represented at all.
  "dog bites man" and "man bites dog" produce the same vector.
- **Distribution.** Training was Wikipedia- and e-commerce-shaped. Retention is 0.915 on
  in-distribution development sets and 0.755 on these six. Expect the lower number out of domain.
- **English only**, 512 wordpieces, WordPiece-30522 vocab. Out-of-vocabulary terms degrade to
  their subword rows; heavily fragmented queries are the weakest case.
- **The document side is not cheap.** The trade is entirely on the query side: 2.05 GB per 1M
  documents at 1024-d fp16.

## Costs

| | |
|---|---|
| query asset (int8 rows + scales + tokenizer) | **31.8 MB** |
| `model.onnx`, batch 1, one thread, 8-token query | **0.047 ms** |
| `model.onnx`, batch 1, one thread, 512-token query | **1.22 ms** |
| `zero_encoder.py`, batch 1, one CPU core | **0.38 ms** |
| hydration (cold load to first query) | **0.22 s** |
| document index, 1024-d fp16 | 2.05 GB per 1M documents |
| document index, 1024-d int8 | 1.02 GB per 1M documents |

The ONNX graph derives token counts from an all-pairs comparison, so its cost grows with the
**square** of the sequence length — 26x from an 8-token query to a 512-token one. Real queries sit
at the short end (the development set's median is 13 wordpieces), but a long one is not free.

## Training

L2 regression of the table's pooled output onto the teacher's query embeddings, over 340,850
approved pairs plus 220,632 query-text-only rows. Sources: **Amazon ESCI** (Apache-2.0),
**FEVER**, **HotpotQA**, **SQuAD**, **NQ-open**, **TriviaQA**, **Mr. TyDi (en)**.

**MS MARCO is permanently excluded** from this lineage — its terms forbid commercial use.

### Attribution

NQ, SQuAD, HotpotQA, FEVER and Mr. TyDi derive from Wikipedia and are **CC BY-SA** (3.0/4.0).
Amazon ESCI is Apache-2.0. The teacher, `NovaSearch/stella_en_400M_v5`, is MIT.

## Provenance

```
run_id             p35w-2m-s2500
table sha256       a7007b1a6af120b976f093fd69ddcb5001996ec0b84b5864b4fd25d7af878abf
teacher            NovaSearch/stella_en_400M_v5 @ ffeb2b7ee715c226d4ffe5e4619f7dbb48624c20
preproc            prefix="" · add_special_tokens · max_length=512 · pool_mode=sqrt
preproc fingerprint adb24fb2e8cad66f
```

`model.npz` is byte-identical across every revision of this repo (sha `a7007b1a…`) and the
reference encoder's output is unchanged, so **no published number differs between revisions**.
This repo was published as `zero-query-encoder-v1` on 2026-09-03 and renamed to `constella-zero`
the same day; the old URL redirects.
