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

**constella-zero** is the smallest query encoder in an asymmetric retrieval family. Documents are
indexed once, in the cloud, with the frozen
[`stella-en-400M-v5-doc-onnx`](https://huggingface.co/DylanCouzon/stella-en-400M-v5-doc-onnx)
tower; Zero or the stronger [`constella-nano`](https://huggingface.co/DylanCouzon/constella-nano)
can then query that same 1024-dimensional index without re-encoding it. Zero is a 30,522 × 1024
int8 lookup table—not a transformer—so encoding is a gather and weighted sum.

> **Research preview.** The registered reserved-four evaluation and broad descriptive BEIR-18
> validation are pending and unspent; no result is claimed for either. The three models are
> registered on the preview branch below, not in an upstream FastEmbed release yet.

## Usage

```console
pip install "fastembed @ git+https://github.com/Dylancouzon/fastembed.git@constella-research-preview"
pip install qdrant-client
```

```python
import numpy as np
from fastembed import TextEmbedding

NAME = "REPO_ID"
query_model = TextEmbedding(NAME)
q = np.asarray(next(iter(query_model.embed(["how do mrna vaccines work?"]))))
assert q.shape == (1024,) and q.dtype == np.float32
```

FastEmbed fetches `model.onnx` and the tokenizer. Pooling and L2 normalization happen inside the
graph.

### The document side

```python
DOC_NAME = "DylanCouzon/stella-en-400M-v5-doc-onnx"
doc_model = TextEmbedding(DOC_NAME)
docs = [
    "mRNA vaccines deliver messenger RNA encoding a viral antigen.",
    "The Treaty of Westphalia ended the Thirty Years' War in 1648.",
]
D = np.stack(list(doc_model.embed(docs)))
assert D.shape == (2, 1024) and D.dtype == np.float32
```

The document tower runs once per document; Zero runs on every query. Do not use the document
model's unprompted path as a Stella query encoder.

### With Qdrant

```python
from qdrant_client import QdrantClient, models

client = QdrantClient(":memory:")
client.create_collection(
    "docs", vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE)
)
client.upsert(
    "docs",
    points=[
        models.PointStruct(id=i, vector=D[i].tolist(), payload={"text": text})
        for i, text in enumerate(docs)
    ],
)
hits = client.query_points("docs", query=q.tolist(), limit=2).points
print(hits[0].payload["text"])
```

### Without FastEmbed

`zero_encoder.py` is the NumPy/tokenizers reference path:

```python
from huggingface_hub import snapshot_download
import sys

d = snapshot_download("REPO_ID")
sys.path.insert(0, d)
from zero_encoder import ZeroQueryEncoder

enc = ZeroQueryEncoder(d, variant="int8")
q_np = enc.encode(["how do mrna vaccines work?"])
assert np.abs(q_np[0] - q).max() < 1e-5
```

Zero and Nano share a document space, not a retrieval-quality guarantee: they have different
measured behavior, and interchangeability does not mean parity, equivalence, or a tie.

## How it works

Tokenize with WordPiece, special tokens on, no prefix, and truncation at 512 tokens. A token that
appears `c` times carries total weight `sqrt(c)`, so repetition saturates. The graph sums the rows,
divides by the weight sum, and L2-normalizes; an empty or near-zero bag falls back to normalized
`[CLS]`. Learned token weights are folded into the rows. This is still a bag of tokens: word order,
negation, and syntax are not represented.

The reported model is int8, which was within 0.00013 nDCG@10 of fp16.

## Files

| file | purpose | size |
|---|---|---:|
| `model.onnx` | FastEmbed/ONNX Runtime; pooled normalized `(batch, 1024)` output | 31 MB |
| `model_tokens.onnx` | token-level `(batch, sequence, 1024)` output | 31 MB |
| `model.npz` | NumPy reference path | 94 MB |

Both graphs use opset 17 and standard operators. The int8 table is dequantized in-graph with one
fp32 scale per row. Tokenizer metadata enforces the frozen 512-token rule and dynamic padding.

## Results

The headline partition is **clean-4**: NFCorpus, SCIDOCS, SciFact, and TREC-COVID. ArguAna and
FiQA remain beside it with `†` because Stella discloses training/evaluation contact with them.
All values are exact-search nDCG@10.

| system | NFCorpus **(clean-4)** | SCIDOCS **(clean-4)** | SciFact **(clean-4)** | TREC-COVID **(clean-4)** | ArguAna† | FiQA† |
|---|---:|---:|---:|---:|---:|---:|
| constella-zero (int8) | 0.3124 | 0.1677 | 0.6101 | 0.5490 | 0.5916 | 0.3728 |
| constella-nano | 0.363080 | 0.217710 | 0.721097 | 0.787116 | 0.623296 | 0.477765 |
| BM25 | 0.3180 | 0.1565 | 0.6791 | 0.6099 | 0.4878 | 0.2532 |
| Stella teacher, symmetric | 0.4134 | 0.2395 | 0.7796 | 0.8234 | 0.6369 | 0.5536 |

† Stella-disclosed training/evaluation contact; excluded from clean-4.

Zero did **not** confirmatorily beat BM25. M7 C2 was +0.0165 across all six with raw 95% interval
[+0.0017, +0.0311], but its sign-flip p=0.0149 failed the Holm threshold of 0.0083. On clean-4,
the descriptive contrast was -0.0311 [-0.0517, -0.0109], with Zero at 0.4098 versus BM25 at
0.4409. Superiority is **UNESTABLISHED**.

The deployable hybrid recommendation and registered operator of record are distinct:

| Zero + BM25 fusion | prefetch | all-six macro | clean-4 macro |
|---|---:|---:|---:|
| Qdrant DBSF | 100 | 0.4887 | 0.4912 |
| M7 convex0 (`w=0.8`) | 1000 | 0.4911 | 0.4866 |

Both rows use `bm25s` with Lucene defaults as the lexical side, not Qdrant's own BM25, which has a
fixed `avg_len` and its own tokenizer — DBSF normalises over returned scores, so a different
lexical implementation shifts its inputs. For the DBSF row, each query's own document is excluded
*before* the prefetch is truncated to 100 (`must_not` on the point id); without that filter a
plain `limit: 100` spends a slot on the self-match. That affects only ArguAna (1,298 of 1,406
queries) and FiQA (55), so the clean-4 figures are unchanged either way.

Use Qdrant DBSF at prefetch 100 in deployments. M7's convex0 is the registered operator of record,
but Qdrant does not implement it. No confidence interval compared these observations, so neither
superiority nor equivalence is established. M7 C3 likewise did not establish fusion superiority
over OpenSearch: +0.0043, raw 95% interval [-0.0063, +0.0151], p=0.219.

Full per-dataset fusion rows, registered contrasts, and source traces are in the
[M21 benchmark ledger](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/main/m21/BENCHMARKS.md).

## Limits

- Reserved-four and BEIR-18 evidence is pending and unspent; the six datasets do not establish
  broad domain coverage.
- Stella contact with ArguAna and FiQA makes all-six results secondary to clean-4.
- Zero is an English-only bag of tokens and truncates beyond 512 wordpieces.
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

This measures query encoders only, not retrieval, ANN, or end-to-end system latency. The asset
column follows the common protocol; `model.onnx` itself is the 31 MB query graph listed above.

## Training

The table was trained by L2 regression against Stella query embeddings over 340,850 pairs plus
220,632 query-only rows from Amazon ESCI, FEVER, HotpotQA, SQuAD, NQ Open, TriviaQA, and Mr. TyDi
(English); MS MARCO was excluded. Wikipedia-derived sources retain CC BY-SA attribution; ESCI and
TriviaQA are Apache-2.0.

## Provenance

```text
run_id      p35w-2m-s2500
table       a7007b1a6af120b976f093fd69ddcb5001996ec0b84b5864b4fd25d7af878abf
teacher     NovaSearch/stella_en_400M_v5 @ ffeb2b7ee715c226d4ffe5e4619f7dbb48624c20
comparator  BAAI/bge-small-en-v1.5 @ 5c38ec7c405ec4b44b94cc5a9bb96e735b38267a
preproc     prefix="" · special tokens · max_length=512 · pool_mode=sqrt
fingerprint adb24fb2e8cad66f
```

The weights are MIT licensed. The pinned Stella teacher/document tower and bge-small comparator
are also MIT and are attributed above.
