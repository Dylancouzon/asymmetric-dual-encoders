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
  - research-preview
base_model: NovaSearch/stella_en_400M_v5
pipeline_tag: feature-extraction
---

# constella-zero

constella-zero is a small English query encoder for semantic search. It uses an int8 token lookup
table instead of a transformer, making it useful when query latency matters more than maximum
retrieval quality.

It produces normalized 1024-dimensional vectors that search documents encoded by
[`stella-en-400M-v5-doc-onnx`](https://huggingface.co/DylanCouzon/stella-en-400M-v5-doc-onnx).
The same document index also works with the stronger
[`constella-nano`](https://huggingface.co/DylanCouzon/constella-nano) query encoder.

> Research preview: Native FastEmbed support currently requires the Constella preview branch
> shown below. The published evaluation is limited to the results described in this card.

| Property | Value |
|---|---|
| Role | Query encoder |
| Output | 1024-dimensional normalized fp32 vector |
| Architecture | 30,522 x 1,024 int8 token lookup table |
| Languages | English |
| Maximum input length | 512 tokens |
| Query prefix | None |
| Document encoder | `DylanCouzon/stella-en-400M-v5-doc-onnx` |
| Recommended retrieval | Hybrid with BM25 and DBSF at prefetch 100 |

## The Constella family

The name Constella combines "constellation" and "Stella." The document embeddings are the fixed
stars, and the query encoder navigates their shared vector space.

Zero and Nano are swappable at query time. Both can search the same document index, so you can
choose between them without re-encoding documents or rebuilding the collection. They do not
produce identical rankings: Zero is the faster option, while Nano has higher retrieval scores on
the six reported datasets. The "zero" name refers to its transformer-free query path.

## Installation

Native FastEmbed support is currently available from the Constella preview branch:

```console
pip install "fastembed @ git+https://github.com/Dylancouzon/fastembed.git@constella-research-preview"
pip install qdrant-client
```

## Usage

Encode documents once with the document model, then encode queries with constella-zero. The
example below creates an in-memory Qdrant collection, but the vectors can be used with any vector
database that supports cosine similarity.

```python
from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models

NAME = "REPO_ID"
DOC_NAME = "DylanCouzon/stella-en-400M-v5-doc-onnx"

documents = [
    "mRNA vaccines deliver messenger RNA encoding a viral antigen.",
    "The Treaty of Westphalia ended the Thirty Years' War in 1648.",
]

document_model = TextEmbedding(DOC_NAME)

client = QdrantClient(":memory:")
client.create_collection(
    "documents",
    vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE),
)
client.upsert(
    "documents",
    points=[
        models.PointStruct(id=i, vector=embedding.tolist(), payload={"text": text})
        for i, (text, embedding) in enumerate(
            zip(documents, document_model.embed(documents))
        )
    ],
)

query_model = TextEmbedding(NAME)
query_embedding = next(iter(query_model.embed(["how do mRNA vaccines work?"])))
results = client.query_points(
    "documents", query=query_embedding.tolist(), limit=2
).points

for result in results:
    print(result.score, result.payload["text"])
```

FastEmbed handles pooling and L2 normalization. Do not use the document model as an unprompted
query encoder. Use constella-zero, constella-nano, or Stella's prompted query path instead.

### NumPy reference implementation

The repository also includes `zero_encoder.py`, a reference implementation that does not require
FastEmbed or ONNX Runtime:

```python
from huggingface_hub import snapshot_download
import sys

model_directory = snapshot_download("REPO_ID")
sys.path.insert(0, model_directory)

from zero_encoder import ZeroQueryEncoder

model = ZeroQueryEncoder(model_directory, variant="int8")
query_embeddings = model.encode(["how do mRNA vaccines work?"])
```

## How it works

The encoder tokenizes each query with WordPiece, looks up a learned vector for every token, and
combines those vectors into one query embedding. Repeated tokens receive diminishing weight: a
token that occurs `c` times contributes a total weight of `sqrt(c)`. The result is L2-normalized.

This is a bag-of-tokens model. It does not represent word order, syntax, or negation directly.
Learned token weights are already included in the table.

## Retrieval results

The table reports exact-search nDCG@10. ArguAna and FiQA are marked because the Stella teacher
discloses training or evaluation contact with those datasets. Results on those two datasets should
therefore be interpreted separately from the other four.

| System | NFCorpus | SCIDOCS | SciFact | TREC-COVID | ArguAna* | FiQA* |
|---|---:|---:|---:|---:|---:|---:|
| constella-zero | 0.3124 | 0.1677 | 0.6101 | 0.5490 | 0.5916 | 0.3728 |
| constella-nano | 0.363080 | 0.217710 | 0.721097 | 0.787116 | 0.623296 | 0.477765 |
| BM25 | 0.3180 | 0.1565 | 0.6791 | 0.6099 | 0.4878 | 0.2532 |
| Stella query encoder | 0.4134 | 0.2395 | 0.7796 | 0.8234 | 0.6369 | 0.5536 |

Note: Stella discloses training or evaluation contact with ArguAna and FiQA.

The recommended deployment setup for Zero is hybrid retrieval. Retrieve with both Zero and BM25,
then combine their results with Qdrant's distribution-based score fusion (DBSF), prefetching 100
candidates from each side. This setup scored 0.4887 mean nDCG@10 across all six datasets and
0.4912 across the four datasets without disclosed Stella contact. The evaluated lexical side used
`bm25s` with Lucene defaults, so results may differ with another BM25 implementation. Dense-only
retrieval remains supported when a lexical index is unavailable or unnecessary.

Zero did not establish an improvement over BM25 in the six-dataset statistical test. Its measured
difference was +0.0165 nDCG@10, but the adjusted test threshold was not met. This is not an
equivalence claim.

## Query encoding cost

These measurements cover the query encoder only. They use batch size 1, four CPU threads, five
warmups, and twenty synthetic 20-word queries in each of three fresh processes. They do not
include vector search or end-to-end application latency.

| Model | Load time | First query | Warm query p50 | Peak RSS | Measured assets |
|---|---:|---:|---:|---:|---:|
| constella-zero | 0.2618 s | 0.3529 ms | 0.1119 ms | 275.4 MiB | 90.1 MiB |
| bge-small | 0.6726 s | 8.2401 ms | 6.8400 ms | 291.0 MiB | 127.6 MiB |
| constella-nano | 0.6907 s | 7.6685 ms | 7.2511 ms | 280.9 MiB | 132.3 MiB |

## Files

| File | Purpose | Size |
|---|---|---:|
| `model.onnx` | Pooled and normalized FastEmbed graph | 31 MB |
| `model_tokens.onnx` | Token-level output for custom pooling | 31 MB |
| `model.npz` | NumPy reference implementation | 94 MB |

Both ONNX graphs use opset 17 and standard operators. The int8 table is dequantized inside the
graph with one fp32 scale per row.

## Training

The released table was trained by L2 regression against Stella query embeddings. Training used
338,076 usable query-document pairs plus 220,632 query-only rows from Amazon ESCI, FEVER,
HotpotQA, SQuAD, NQ Open, TriviaQA, and Mr. TyDi English. MS MARCO was excluded.

Wikipedia-derived data retains CC BY-SA attribution. Amazon ESCI and TriviaQA are Apache-2.0.

## Limitations

- The model is English-only and truncates inputs after 512 tokens.
- As a bag-of-tokens model, it is weak at distinctions that depend on word order, syntax, or
  negation.
- Retrieval quality is lower than constella-nano and the full Stella query encoder on the six
  reported datasets.
- Document indexing still requires the 400M-parameter Stella document encoder.
- The reported retrieval evaluation covers six datasets and does not establish performance in
  other domains or applications.

## License and provenance

The model is MIT licensed. It was distilled from
[`NovaSearch/stella_en_400M_v5`](https://huggingface.co/NovaSearch/stella_en_400M_v5) at revision
`ffeb2b7ee715c226d4ffe5e4619f7dbb48624c20`, which is also MIT licensed.

The released int8 table has SHA-256
`a7007b1a6af120b976f093fd69ddcb5001996ec0b84b5864b4fd25d7af878abf`.
