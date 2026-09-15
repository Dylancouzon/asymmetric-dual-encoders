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

The query side of an **asymmetric dual encoder**: documents are indexed once with a large frozen
encoder; queries are encoded at serving time with a smaller encoder.

Nano starts from `bge-small-en-v1.5` and uses a three-layer feature tap plus a linear head. It was
trained to land in the 1024-dimensional document space of `stella_en_400M_v5`. The matching
document encoder is published as
[`stella-en-400M-v5-doc-onnx`](https://huggingface.co/DylanCouzon/stella-en-400M-v5-doc-onnx);
the two towers are only meaningful together.

> **Research preview.** The registered reserved-four evaluation and broad descriptive BEIR-18
> validation are pending. Read [Results](#results) and [Limits](#limits) first. The native model
> entry currently lives only on the custom FastEmbed branch `m14-constella-preview`; it is **not
> in upstream FastEmbed**. The upstream PR belongs to later M20 work.

## Usage

Install the preview branch and Qdrant's Python client:

```console
pip install "fastembed @ git+https://github.com/Dylancouzon/fastembed@m14-constella-preview" qdrant-client
```

The snippets in this section run in order, sharing state. They use FastEmbed's native model
registrations and never call `add_custom_model`. The optional model-path variables are only for
offline verification; omit them for normal Hub-backed use.

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

Run the published Stella document tower in the indexing job, once per document:

```python
doc_model = TextEmbedding(DOC_NAME, **doc_kwargs)
docs = [
    "mRNA vaccines deliver messenger RNA encoding a viral antigen.",
    "The Treaty of Westphalia ended the Thirty Years' War in 1648.",
]
D = np.stack(list(doc_model.embed(docs))).astype(np.float32, copy=False)
assert D.shape == (2, 1024)
assert np.isfinite(D).all()
```

The document model is the large, frozen side of the system. Do not use its unprompted path as a
Stella query encoder.

### With Qdrant

This defaults to Qdrant's local, in-memory mode, so the example is runnable without a server but
the collection disappears with the process. Set `QDRANT_URL` to use an already-running Qdrant
deployment instead; choose a fresh collection name there.

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

# Serving time: Nano encodes each query; the document tower does not run here.
query_model = TextEmbedding(NANO_NAME, **nano_kwargs)
query = "how do mRNA vaccines work?"
q_native = np.asarray(next(iter(query_model.embed([query]))))
# FastEmbed's integer attention mask promotes Nano's masked mean to float64.
# Cast explicitly to keep each 1024-d query at the intended 4,096-byte fp32 size.
q = q_native.astype(np.float32, copy=False)
assert q.shape == (1024,) and np.isfinite(q).all()

hits = client.query_points(COLLECTION_NAME, query=q.tolist(), limit=2).points
print([(hit.payload["text"], hit.score) for hit in hits])
```

The collection is 1024-dimensional cosine. Both towers emit L2-normalized vectors, so cosine and
dot-product ranking agree for these vectors; cosine keeps the collection contract explicit.

### Swapping the query encoder

The same Qdrant collection can be queried with
[`constella-zero`](https://huggingface.co/DylanCouzon/constella-zero) without re-encoding its
documents:

```python
zero_model = TextEmbedding(ZERO_NAME, **zero_kwargs)
q_zero = np.asarray(next(iter(zero_model.embed([query])))).astype(np.float32, copy=False)
assert q_zero.shape == (1024,) and np.isfinite(q_zero).all()
zero_hits = client.query_points(COLLECTION_NAME, query=q_zero.tolist(), limit=2).points
print([(hit.payload["text"], hit.score) for hit in zero_hits])
```

<!-- m14-card-usage-end -->

Nano and Zero both emit normalized 1024-dimensional query vectors in the same Stella document
space. One document index therefore serves either query encoder, and the choice can be made per
deployment or per query. This compatibility is geometric, **not a retrieval-quality equivalence
claim**: the encoders have distinct measured retrieval behavior, and no parity, equivalence or
tie is implied. The query-cost side of the trade-off is measured under the same four-thread CPU
protocol: Zero's warm p50 median was **0.1119 ms** and Nano's was **7.2511 ms**. These are
**synthetic latencies, not workload estimates**. Only Zero claims near-zero query compute; Nano's
query cost is roughly bge-small's, whose warm p50 median was 6.8400 ms under that protocol.

## How it works

Nano starts from
[`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5). Its three-layer feature
tap and linear head were trained to match frozen query and document targets in the space of
[`NovaSearch/stella_en_400M_v5`](https://huggingface.co/NovaSearch/stella_en_400M_v5). Nano needs
no query prefix.

Inputs are right-truncated to 512 tokens. The tokenizer leaves padding unset so FastEmbed uses
dynamic batch-longest padding. The ONNX graph emits fp32 token embeddings; FastEmbed's native
`PooledNormalizedEmbedding` family applies attention-masked mean pooling and L2 normalization.
That family assignment is required for correct native output.

FastEmbed's integer attention mask currently promotes Nano's native masked-mean result to
float64 even though the graph output is fp32. The usage example casts the final normalized query
vector back to fp32 before sending it to Qdrant.

## Files

The release bundle contains the ONNX graph plus its FastEmbed configuration and tokenizer files.
The native FastEmbed registration downloads those assets by model name. The graph returns token
embeddings; pooling and normalization are supplied by FastEmbed, not serialized into the graph.

The custom branch registers Nano with `PooledNormalizedEmbedding`, and registers the published
Stella document tower and constella-zero with their appropriate native families. It is a preview
vehicle, not an upstream FastEmbed release.

## Results

The headline is the **clean-four** partition, which excludes ArguAna and FiQA because Stella
discloses training/evaluation contact with those two datasets. Against bge-small, Nano improved
clean-four nDCG@10 by **+0.017648**, with one-sided lower 2.5% bound **+0.003674** and sign-flip
**p=0.006410**.

Nano's exact-search nDCG@10 results are:

| dataset | Nano | Nano − bge-small | Nano − LEAF asym |
|---|---:|---:|---:|
| SciFact | 0.721097 | +0.008391 | +0.022084 |
| NFCorpus | 0.363080 | +0.020129 | +0.002301 |
| FiQA† | 0.477765 | +0.074253 | +0.061296 |
| ArguAna† | 0.623296 | +0.019848 | +0.040043 |
| SCIDOCS | 0.217710 | +0.012498 | +0.014345 |
| TREC-COVID | 0.787116 | +0.029573 | -0.042982 |

† Stella-disclosed training/evaluation contact; excluded from clean-four.

Against LEAF asym, Nano established an all-six improvement of **+0.016181** (one-sided lower 2.5%
bound **+0.006504**, sign-flip **p=0.000540**). The clean-four result against LEAF is
**UNESTABLISHED**: the contrast was **-0.001063**, lower bound **-0.014456**, **p=0.559474**. This
is not parity, equivalence or a tie. The **-0.042982** TREC-COVID difference versus LEAF is the
principal per-dataset limitation.

## Limits

- **Pending registered evidence.** The reserved four remain unestablished and unspent at preview
  time; no result is claimed for them. Broad BEIR-18 validation is also pending, so the six-set
  table must not be read as broad domain coverage.
- **Teacher contact.** Stella discloses training/evaluation contact with ArguAna and FiQA. They
  are excluded from the headline clean-four partition; all-six results are secondary.
- **LEAF comparison.** Clean-four performance against LEAF is UNESTABLISHED. No parity,
  equivalence or tie follows, and TREC-COVID is the main observed per-dataset weakness.
- **Length and language.** Nano is English-only and truncates beyond 512 tokens.
- **The document side is not cheap.** The Stella document tower still runs once per document;
  Nano changes query-side cost, not indexing cost.

## Costs

The measurements below are **synthetic query latencies, not workload estimates**. They used the
same four-thread, batch-one CPU protocol, with medians across three fresh processes.

| model | hydration | first query | warm 20-word p50 | peak RSS | model assets |
|---|---:|---:|---:|---:|---:|
| constella-zero | 0.2618 s | 0.3529 ms | 0.1119 ms | 275.4 MiB | 90.1 MiB |
| bge-small | 0.6726 s | 8.2401 ms | 6.8400 ms | 291.0 MiB | 127.6 MiB |
| constella-nano | 0.6907 s | 7.6685 ms | 7.2511 ms | 280.9 MiB | 132.3 MiB |

A native 1024-dimensional Nano query occupies 8,192 bytes as float64; the explicit fp32 cast in
the example reduces it to 4,096 bytes. If query vectors are stored, the corresponding raw-vector
sizes are 8.192 GB per million for native float64 and 4.096 GB per million for cast fp32, before
index overhead. Any 4.096 GB figure assumes the fp32 cast.

## Training

The frozen checkpoint saw exactly **199,999,721 examples**, a reconciled shortfall of **279**
from the nominal 200,000,000-example plan. Nine one-row document tail batches each lost 31 rows;
the original failed supervisor receipt remains part of the audit trail.

Training mixed real queries from ESCI, HotpotQA, Mr. TyDi, NQ Open, SQuAD and TriviaQA; a PAQ
sample; harvested titles, headings and claim sentences from Wikipedia, arXiv and the licensed
pool; seven generated query forms; and the eligible document pool. MS MARCO was excluded from
training and used only as a validation surface. The CC BY-SA sources and PAQ retain their source
attribution.

Every training source was fingerprint-screened against the protected index containing the six
evaluation datasets and COV components. The final assembly records 709 query removals and 79,630
document removals, plus zero cross-role collisions. This is fingerprint-level decontamination;
it does not erase the Stella-contact disclosure above.

## Provenance

```text
student  BAAI/bge-small-en-v1.5 @ 5c38ec7c405ec4b44b94cc5a9bb96e735b38267a
teacher  NovaSearch/stella_en_400M_v5 @ ffeb2b7ee715c226d4ffe5e4619f7dbb48624c20
```

The preview weights are released under MIT. The pinned bge-small backbone and Stella
teacher/document tower are also MIT and are attributed above. Dataset-source obligations and the
decontamination record remain documented in the project evidence.
