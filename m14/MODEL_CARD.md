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
  - research-preview
base_model: BAAI/bge-small-en-v1.5
pipeline_tag: feature-extraction
---

# constella-nano

constella-nano is a 34,540,672-parameter English query encoder for semantic search. It offers
better retrieval quality than constella-zero while remaining much smaller than the Stella model
used to encode documents.

It produces normalized 1024-dimensional vectors that search documents encoded by
[`stella-en-400M-v5-doc-onnx`](https://huggingface.co/DylanCouzon/stella-en-400M-v5-doc-onnx).
The same document index also works with the faster
[`constella-zero`](https://huggingface.co/DylanCouzon/constella-zero) query encoder.

| Property | Value |
|---|---|
| Role | Query encoder |
| Output | 1024-dimensional normalized fp32 vector |
| Architecture | bge-small backbone with three tapped layers and a linear projection |
| Parameters | 34,540,672 |
| Languages | English |
| Maximum input length | 512 tokens |
| Query prefix | None |
| Document encoder | `DylanCouzon/stella-en-400M-v5-doc-onnx` |

## Installation

Native FastEmbed support is currently available from the Constella preview branch:

```console
pip install "fastembed @ git+https://github.com/Dylancouzon/fastembed.git@constella-research-preview"
pip install qdrant-client
```

## Usage

Encode documents once with the document model, then encode queries with constella-nano. The
example below creates an in-memory Qdrant collection, but the vectors can be used with any vector
database that supports cosine similarity.

<!-- card-usage-start -->

```python
from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models

NAME = "DylanCouzon/constella-nano"
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

<!-- card-usage-end -->

FastEmbed applies attention-masked mean pooling and L2 normalization. Do not use the document
model as an unprompted query encoder. Use constella-nano, constella-zero, or Stella's prompted
query path instead.

To switch from Nano to Zero without rebuilding the document index, change `NAME` to
`DylanCouzon/constella-zero`.

## How it works

Nano starts from
[`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5). Features from layers
12, 8, and 4 feed a learned linear projection into Stella's 1024-dimensional document space. The
model was trained against frozen query embeddings from
[`NovaSearch/stella_en_400M_v5`](https://huggingface.co/NovaSearch/stella_en_400M_v5).

The model does not use a query prefix. Inputs are right-truncated at 512 tokens and dynamically
padded to the longest input in each batch. The ONNX graph returns fp32 token embeddings;
FastEmbed performs pooling and normalization.

## Retrieval results

The table reports exact-search nDCG@10. ArguAna and FiQA are marked because the Stella teacher
discloses training or evaluation contact with those datasets. Results on those two datasets should
therefore be interpreted separately from the other four.

| System | NFCorpus | SCIDOCS | SciFact | TREC-COVID | ArguAna* | FiQA* |
|---|---:|---:|---:|---:|---:|---:|
| constella-nano | 0.363080 | 0.217710 | 0.721097 | 0.787116 | 0.623296 | 0.477765 |
| constella-zero | 0.3124 | 0.1677 | 0.6101 | 0.5490 | 0.5916 | 0.3728 |
| BM25 | 0.3180 | 0.1565 | 0.6791 | 0.6099 | 0.4878 | 0.2532 |
| Stella query encoder | 0.4134 | 0.2395 | 0.7796 | 0.8234 | 0.6369 | 0.5536 |

Note: Stella discloses training or evaluation contact with ArguAna and FiQA.

Nano passed the predefined superiority test against bge-small-en-v1.5 on both the four datasets
without disclosed Stella contact and all six datasets. The mean nDCG@10 differences were +0.017648
and +0.027449, respectively.

Against LEAF asym, Nano passed the test across all six datasets with a +0.016181 difference. It
did not pass on the four datasets without disclosed Stella contact, where the measured difference
was -0.001063. That result does not establish equality between the models.

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

The repository contains a 137,802,369-byte fp32 ONNX graph plus its FastEmbed configuration and
tokenizer files. The graph uses ONNX opset 17 and standard operators. It returns token embeddings;
FastEmbed supplies pooling and normalization.

## Training

The released checkpoint saw exactly 199,999,721 training examples in a 75% query and 25% document
mix. Training data included queries from Amazon ESCI, HotpotQA, Mr. TyDi, NQ Open, SQuAD,
TriviaQA, and PAQ, along with licensed documents and generated query forms. MS MARCO was used for
validation only.

CC BY-SA sources retain the required attribution. PAQ is licensed under CC BY-SA 3.0.

## Limitations

- The model is English-only and truncates inputs after 512 tokens.
- Document indexing still requires the 400M-parameter Stella document encoder.
- The reported retrieval evaluation covers six datasets and does not establish performance in
  other domains or applications.
- Stella discloses training or evaluation contact with ArguAna and FiQA, so the scores on those
  datasets are not treated as independent evidence.
- Nano is slower than constella-zero at query time.

## License and provenance

The model is MIT licensed. It uses
[`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5) at revision
`5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` as its backbone and was distilled from
[`NovaSearch/stella_en_400M_v5`](https://huggingface.co/NovaSearch/stella_en_400M_v5) at revision
`ffeb2b7ee715c226d4ffe5e4619f7dbb48624c20`. Both source models are MIT licensed.

The released ONNX graph has SHA-256
`9ba0acf57b71dc31bc5512c5445078a797fa51cf3e85587d6b8a506bfc55dbc2`.
