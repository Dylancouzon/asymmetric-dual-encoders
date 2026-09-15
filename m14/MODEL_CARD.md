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

**constella-nano** is one of two interchangeable query encoders for an asymmetric retrieval
family. Documents are indexed once, in the cloud, with the frozen
[`stella-en-400M-v5-doc-onnx`](https://huggingface.co/DylanCouzon/stella-en-400M-v5-doc-onnx)
tower; Nano or the smaller [`constella-zero`](https://huggingface.co/DylanCouzon/constella-zero)
can then query that same 1024-dimensional index without re-encoding it. Nano starts from
`bge-small-en-v1.5` and trades Zero's near-zero query compute for substantially stronger dense
retrieval.

> **Research preview.** The registered reserved-four evaluation and broad descriptive BEIR-18
> validation are pending and unspent; no result is claimed for either. The three models are
> registered on the preview branch below, not in an upstream FastEmbed release yet.

## Usage

```console
pip install "fastembed @ git+https://github.com/Dylancouzon/fastembed.git@constella-research-preview"
pip install qdrant-client
```

These snippets run in order. The optional paths support offline verification; omit them for
normal Hub-backed use. No `add_custom_model` call is needed.

<!-- m14-card-usage-start -->

```python
import os

import numpy as np
from fastembed import TextEmbedding

NANO_NAME = "DylanCouzon/constella-nano"
ZERO_NAME = "DylanCouzon/constella-zero"
DOC_NAME = "DylanCouzon/stella-en-400M-v5-doc-onnx"

nano_path = os.environ.get("CONSTELLA_NANO_PATH")
zero_path = os.environ.get("CONSTELLA_ZERO_PATH")
doc_path = os.environ.get("CONSTELLA_DOC_PATH")
nano_kwargs = {"specific_model_path": nano_path} if nano_path else {}
zero_kwargs = {"specific_model_path": zero_path} if zero_path else {}
doc_kwargs = {"specific_model_path": doc_path} if doc_path else {}
```

### The document side

Run the document tower once per document in the indexing job:

```python
doc_model = TextEmbedding(DOC_NAME, **doc_kwargs)
docs = [
    "mRNA vaccines deliver messenger RNA encoding a viral antigen.",
    "The Treaty of Westphalia ended the Thirty Years' War in 1648.",
]
D = np.stack(list(doc_model.embed(docs)))
assert D.shape == (2, 1024) and D.dtype == np.float32
```

Do not use the document model's unprompted path as a Stella query encoder.

### With Qdrant

```python
from qdrant_client import QdrantClient, models

qdrant_url = os.environ.get("QDRANT_URL")
client = QdrantClient(url=qdrant_url) if qdrant_url else QdrantClient(":memory:")
COLLECTION_NAME = "constella_nano_demo"

client.create_collection(
    COLLECTION_NAME,
    vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE),
)
client.upsert(
    COLLECTION_NAME,
    points=[
        models.PointStruct(id=i, vector=D[i].tolist(), payload={"text": text})
        for i, text in enumerate(docs)
    ],
)

query = "how do mRNA vaccines work?"
query_model = TextEmbedding(NANO_NAME, **nano_kwargs)
q = np.asarray(next(iter(query_model.embed([query]))))
assert q.shape == (1024,) and q.dtype == np.float32 and np.isfinite(q).all()

hits = client.query_points(COLLECTION_NAME, query=q.tolist(), limit=2).points
print([(hit.payload["text"], hit.score) for hit in hits])
```

Both towers emit normalized vectors. Cosine and dot-product ranking therefore agree, while
`COSINE` keeps the collection contract explicit.

### Swapping the query encoder

The same collection can be queried with Zero:

```python
zero_model = TextEmbedding(ZERO_NAME, **zero_kwargs)
q_zero = np.asarray(next(iter(zero_model.embed([query]))))
assert q_zero.shape == (1024,) and q_zero.dtype == np.float32
zero_hits = client.query_points(COLLECTION_NAME, query=q_zero.tolist(), limit=2).points
print([(hit.payload["text"], hit.score) for hit in zero_hits])
```

<!-- m14-card-usage-end -->

Zero and Nano share a document space, not a retrieval-quality guarantee: they have different
measured behavior, and interchangeability does not mean parity, equivalence, or a tie.

## How it works

Nano starts from
[`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5). A three-layer feature
tap and linear head were trained against frozen targets from
[`NovaSearch/stella_en_400M_v5`](https://huggingface.co/NovaSearch/stella_en_400M_v5). It needs no
query prefix. Inputs are right-truncated to 512 tokens; dynamic batch-longest padding is used.
The ONNX graph returns fp32 token embeddings, and FastEmbed applies attention-masked mean pooling
and L2 normalization.

## Files

The release contains the ONNX graph, FastEmbed configuration, and tokenizer files. The graph
returns token embeddings; FastEmbed supplies pooling and normalization. The preview branch
registers Nano as `PooledNormalizedEmbedding` and registers Zero and the document tower with their
native ONNX path.

## Results

The headline partition is **clean-4**: NFCorpus, SCIDOCS, SciFact, and TREC-COVID. ArguAna and
FiQA remain beside it with `†` because Stella discloses training/evaluation contact with them.
All values are exact-search nDCG@10.

| system | NFCorpus **(clean-4)** | SCIDOCS **(clean-4)** | SciFact **(clean-4)** | TREC-COVID **(clean-4)** | ArguAna† | FiQA† |
|---|---:|---:|---:|---:|---:|---:|
| constella-nano | 0.363080 | 0.217710 | 0.721097 | 0.787116 | 0.623296 | 0.477765 |
| constella-zero (int8) | 0.3124 | 0.1677 | 0.6101 | 0.5490 | 0.5916 | 0.3728 |
| BM25 | 0.3180 | 0.1565 | 0.6791 | 0.6099 | 0.4878 | 0.2532 |
| Stella teacher, symmetric | 0.4134 | 0.2395 | 0.7796 | 0.8234 | 0.6369 | 0.5536 |

† Stella-disclosed training/evaluation contact; excluded from clean-4.

The registered Nano contrasts were:

| contrast | partition | point delta | one-sided 2.5% lower bound | sign-flip p | status |
|---|---|---:|---:|---:|---|
| Nano − bge-small | **clean-4** | +0.017648 | +0.003674 | 0.006410 | **ESTABLISHED** |
| Nano − bge-small | all six | +0.027449 | +0.017271 | 0.000010 | **ESTABLISHED** |
| Nano − LEAF asym | all six | +0.016181 | +0.006504 | 0.000540 | **ESTABLISHED** |
| Nano − LEAF asym | **clean-4** | -0.001063 | -0.014456 | 0.559474 | **UNESTABLISHED** |

No absolute per-dataset score is reported for bge-small or LEAF. Clean-4 superiority over LEAF
is **UNESTABLISHED**—not parity, equivalence, or a tie—and the principal per-dataset limitation is
TREC-COVID at **-0.042982 versus LEAF**. Full intervals, per-dataset deltas, and source traces are
in the [M21 benchmark ledger](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/main/m21/BENCHMARKS.md).

## Limits

- Reserved-four and BEIR-18 evidence is pending and unspent; the six datasets do not establish
  broad domain coverage.
- Stella contact with ArguAna and FiQA makes all-six results secondary to clean-4.
- Nano is English-only and truncates beyond 512 tokens.
- The document side is not cheap: the 400M-parameter Stella tower still runs at indexing time.

## Costs

These are synthetic query latencies, not workload estimates. The common three-model protocol used
three fresh processes per model, batch one, four CPU threads; hydration includes imports,
verification, and load but excludes interpreter startup; first inference is separate; warm timing
uses five warmups and twenty 20-word samples; the OS disk cache was not flushed.

| model | hydration | first query | warm 20-word p50 | peak RSS | model assets |
|---|---:|---:|---:|---:|---:|
| constella-zero | 0.2618 s | 0.3529 ms | 0.1119 ms | 275.4 MiB | 90.1 MiB |
| bge-small | 0.6726 s | 8.2401 ms | 6.8400 ms | 291.0 MiB | 127.6 MiB |
| constella-nano | 0.6907 s | 7.6685 ms | 7.2511 ms | 280.9 MiB | 132.3 MiB |

A normalized 1024-dimensional fp32 query vector occupies 4,096 bytes.

## Training

The frozen checkpoint saw exactly **199,999,721 training examples**. Training mixed real queries
from ESCI, HotpotQA, Mr. TyDi, NQ Open, SQuAD, and TriviaQA; a PAQ sample; licensed document
sources; seven generated query forms; and the eligible document pool. MS MARCO was validation
only. CC BY-SA sources and PAQ retain their source attribution.

Every source was fingerprint-screened against the protected six-dataset/COV index. Final assembly
removed 709 queries and 79,630 documents and recorded zero cross-role collisions. This is
fingerprint-level decontamination; it does not erase the Stella-contact disclosure.

## Provenance

```text
student  BAAI/bge-small-en-v1.5 @ 5c38ec7c405ec4b44b94cc5a9bb96e735b38267a
teacher  NovaSearch/stella_en_400M_v5 @ ffeb2b7ee715c226d4ffe5e4619f7dbb48624c20
```

The preview weights are MIT licensed. The pinned bge-small backbone and Stella teacher/document
tower are also MIT and are attributed above; dataset obligations and the audit trail remain in the
project evidence.
