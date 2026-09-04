---
license: mit
language: en
library_name: fastembed
tags:
  - fastembed
  - qdrant
  - onnx
  - retrieval
  - asymmetric-dual-encoder
  - edge
base_model: NovaSearch/stella_en_400M_v5
pipeline_tag: feature-extraction
---

# constella-zero

The query side of an **asymmetric dual encoder**: documents are indexed once, in the cloud, by a
large frozen encoder; queries are encoded on the device by **a lookup table**.

There is no transformer here. The model is 30,522 × 1024 int8 rows and one pooling rule —
encoding a query is a gather and a weighted sum. The query asset is **31.8 MB**, and the reference
implementation encodes a query end to end, tokenization included, in **0.38 ms** on one CPU core
(the ONNX graph alone runs an 8-token query in 0.047 ms — see [Costs](#costs)).

It was distilled from [`stella_en_400M_v5`](https://huggingface.co/NovaSearch/stella_en_400M_v5)
so that its output lands in that model's document space. The matching document encoder is
published as [`stella-en-400M-v5-doc-onnx`](https://huggingface.co/DylanCouzon/stella-en-400M-v5-doc-onnx);
the two are only meaningful together.

*constella = constellation + stella: navigate by fixed stars, no engine.*

**Prior art.** This construction — a per-token dense lookup table distilled from a frozen teacher,
reusing that teacher's index unchanged — is not new here. [pyNIFE](https://github.com/stephantul/pynife)
(Stephan Tulkens, MIT, 2025-11-03) published it first, and its models are worth comparing against.
What is specific to this release is the measurement (six BEIR sets, exact search, frozen
comparators, pre-registered statistics), the artifact constraints (30,522 int8 rows, no MS MARCO
anywhere in the lineage), and the fact that a second, larger query encoder is coming for the same
index.

> **Research preview.** It is a bag of tokens and behaves like one. Read
> [Results](#results) and [Limits](#limits) first.

## Usage

The snippets in this section run in order, sharing state.

```python
from fastembed import TextEmbedding

NAME = "REPO_ID"
query_model = TextEmbedding(NAME)
q = next(iter(query_model.embed(["how do mrna vaccines work?"])))   # (1024,), L2-normalized
```

Not in a FastEmbed release yet. Until it is:

    pip install "fastembed @ git+https://github.com/Dylancouzon/fastembed@add-constella-models"

FastEmbed fetches only `model.onnx` and the tokenizer — about 31 MB, not the whole repo. Pooling
and L2 normalization happen inside the graph.

### The document side

```python
DOC_NAME = "DylanCouzon/stella-en-400M-v5-doc-onnx"

doc_model = TextEmbedding(DOC_NAME)          # 1.75 GB, runs in the cloud, once per document
docs = [
    "mRNA vaccines deliver a strand of messenger RNA encoding a viral antigen.",
    "The Treaty of Westphalia ended the Thirty Years' War in 1648.",
]
D = list(doc_model.embed(docs))
```

That asymmetry is the point: `doc_model` is a 400M-parameter transformer that runs once per
document. `query_model` runs on every query, on the device, and costs almost nothing.

### With Qdrant

```python
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

Qdrant implements cosine as a dot product — it normalizes on upsert and compares with dot — so
`COSINE` costs the same as `DOT` here without assuming the caller preserved unit norm.

The table itself can also live in Qdrant, as a retrieve-by-id collection of one point per vocab
row (`hnsw_config=models.HnswConfigDiff(m=0)` — indexing it is pure waste), so the query path holds
no model weights at all.

### Without FastEmbed

`zero_encoder.py` is the reference implementation — 93 lines, `numpy` and `tokenizers`, no torch.
This downloads the whole repo, not just the 31 MB graph.

```python
from huggingface_hub import snapshot_download
import sys, numpy as np

d = snapshot_download("REPO_ID")
sys.path.insert(0, d)
from zero_encoder import ZeroQueryEncoder

enc = ZeroQueryEncoder(d, variant="int8")            # or "fp16"
q_np = enc.encode(["how do mrna vaccines work?"])    # (1, 1024), L2-normalized
assert np.abs(q_np[0] - q).max() < 1e-5              # the vector FastEmbed just produced
```

## How it works

Tokenize (WordPiece, special tokens on, truncate at 512, no padding, no prefix). A token appearing
`c` times carries **total weight `sqrt(c)`** — repetition saturates. Sum the rows, divide by the
weight sum, L2-normalize. An empty or near-zero-norm bag falls back to the normalized `[CLS]` row
(id 101). Per-token learned weights are folded into the rows, so the artifact is self-contained.

Because pooling is not a masked mean, it is done inside the ONNX graph rather than by the caller.
`config.json` carries the rule and its fingerprint (`adb24fb2e8cad66f`).

`int8` is the variant every number below was measured on; it is loss-free against `fp16` to within
0.00013 nDCG@10.

## Files

You need exactly one of these three.

| file | for | size |
|---|---|---|
| `model.onnx` | FastEmbed, or any ONNX runtime — pooled and normalized, `(b, 1024)` | 31 MB |
| `model_tokens.onnx` | pipelines that insist on pooling themselves, `(b, s, 1024)` | 31 MB |
| `model.npz` | the numpy reference path | 94 MB |

Both graphs are opset 17, standard operators only, carrying the table as an int8 initializer with a
per-row fp32 scale dequantized in-graph.

The bundled tokenizer files are stella's, with `model_max_length`/`max_length` set to **512** and
`padding` to **null** — the rule the document index was built with. stella ships 32768/8000 and
fixed-512 padding, which any loader honouring those fields would otherwise apply.
`config.json` records the originals under `tokenizer_deviation_from_teacher`.

## Results

nDCG@10 on six BEIR datasets, exact search so ANN recall is not a confound. Measured once, on the
table shipped here (sha `a7007b1a…`).

| system | arguana | fiqa | nfcorpus | scidocs | scifact | trec-covid | **average** |
|---|---|---|---|---|---|---|---|
| **constella-zero (int8)** | 0.5916 | 0.3728 | 0.3124 | 0.1677 | 0.6101 | 0.5490 | **0.4339** |
| + BM25, convex fusion | 0.5975 | 0.4026 | 0.3497 | 0.1881 | 0.7068 | 0.7018 | **0.4911** |
| BM25 alone | 0.4878 | 0.2532 | 0.3180 | 0.1565 | 0.6791 | 0.6099 | 0.4174 |
| the teacher, used on both sides | 0.6369 | 0.5536 | 0.4134 | 0.2395 | 0.7796 | 0.8234 | 0.5744 |

A lookup table retains **75.5%** of the teacher's quality (0.4339 / 0.5744), with a query side
that does no matrix multiplication at all.

The fused row is **weighted score fusion**, `0.8 × dense + 0.2 × BM25` on min-max normalized
scores — not reciprocal rank fusion, so Qdrant's `Fusion.RRF` will not reproduce it.

## Limits

- **Teacher contamination.** stella discloses **ArguAna** and **FiQA** in its training data —
  two of the six above, and ArguAna is its second-highest score. On the four sets with no
  disclosed overlap it averages **0.4098 against BM25's 0.4409** — below BM25. Weight the
  average accordingly.
- **It is a bag of tokens.** Word order, negation and syntax are not represented: "dog bites man"
  and "man bites dog" give the same vector.
- **Out of domain it drops.** Training was Wikipedia- and e-commerce-shaped, and the six sets
  above are further from that than the data it was fitted on.
- **English only**, 512 wordpieces, 30,522-token WordPiece vocab. Out-of-vocabulary terms degrade
  to subword rows.
- **The document side is not cheap** — 2.05 GB per 1M documents at 1024-d fp16. The whole trade is
  on the query side.

## Costs

| | |
|---|---|
| query asset (int8 rows + scales + tokenizer) | 31.8 MB |
| `model.onnx` graph execution, batch 1, one thread, 8-token query | 0.047 ms |
| `model.onnx` graph execution, batch 1, one thread, 512-token query | 1.22 ms |
| `zero_encoder.py` end to end, batch 1, one CPU core, incl. tokenization | 0.38 ms |
| hydration (cold load to first query) | 0.22 s |
| document vectors, 1024-d fp16 / int8 | 2.05 / 1.02 GB per 1M — raw payload, before index overhead |

The graph rows exclude tokenization; `zero_encoder.py`'s 0.38 ms is the end-to-end figure and the
honest one to compare against another encoder. No end-to-end FastEmbed timing is published here.

The graph derives token counts from an all-pairs comparison, so cost grows with the **square** of
sequence length — 26x from an 8-token query to a 512-token one. Real queries sit at the short end
(median 13 wordpieces).

## Training

L2 regression of the table's pooled output onto the teacher's query embeddings, over 340,850
pairs plus 220,632 query-text-only rows, from **Amazon ESCI**, **FEVER**, **HotpotQA**, **SQuAD**,
**NQ-open**, **TriviaQA** and **Mr. TyDi (en)**. No MS MARCO.

Attribution: NQ, SQuAD, HotpotQA, FEVER and Mr. TyDi are Wikipedia-derived and **CC BY-SA**
(3.0/4.0); Amazon ESCI and TriviaQA are Apache-2.0; the teacher is MIT.

## Provenance

```
run_id             p35w-2m-s2500
table sha256       a7007b1a6af120b976f093fd69ddcb5001996ec0b84b5864b4fd25d7af878abf
teacher            NovaSearch/stella_en_400M_v5 @ ffeb2b7ee715c226d4ffe5e4619f7dbb48624c20
preproc            prefix="" · add_special_tokens · max_length=512 · pool_mode=sqrt
preproc fingerprint adb24fb2e8cad66f
```

Published as `zero-query-encoder-v1` and renamed on 2026-09-03; the old URL redirects.
