---
license: mit
language: en
base_model: NovaSearch/stella_en_400M_v5
library_name: fastembed
tags:
  - fastembed
  - qdrant
  - onnx
  - retrieval
  - asymmetric-dual-encoder
  - research-preview
pipeline_tag: feature-extraction
---

# stella-en-400M-v5-doc-onnx

This is the document encoder used by the Constella asymmetric retrieval models. It converts
English documents into normalized 1024-dimensional vectors. Index documents with this model,
then search the index with either
[`constella-zero`](https://huggingface.co/DylanCouzon/constella-zero) or
[`constella-nano`](https://huggingface.co/DylanCouzon/constella-nano).

> Research preview: Native FastEmbed support currently requires the Constella preview branch
> shown below. The published evaluation is limited to the results described in the model cards.

The model is an ONNX conversion of
[`NovaSearch/stella_en_400M_v5`](https://huggingface.co/NovaSearch/stella_en_400M_v5) at revision
`ffeb2b7ee715c226d4ffe5e4619f7dbb48624c20`. The weights were not trained or fine-tuned during
conversion. Pooling and normalization are included in the graph, and inference does not require
PyTorch or `trust_remote_code`.

| Property | Value |
|---|---|
| Role | Document encoder |
| Output | 1024-dimensional normalized fp32 vector |
| Architecture | Stella 400M transformer |
| Languages | English |
| Maximum input length | 512 tokens |
| Query prefix | Not applicable; this artifact is for documents only |
| Compatible query encoders | constella-zero and constella-nano |

## The Constella family

The name Constella combines "constellation" and "Stella." The document embeddings are the fixed
stars, and the query encoder navigates their shared vector space.

Zero and Nano are swappable at query time. Both can search documents encoded by this model, so
you can change query encoders without re-encoding documents or rebuilding the collection. This
compatibility does not mean they produce identical rankings or have equal retrieval quality.

## Installation

Native FastEmbed support is currently available from the Constella preview branch:

```console
pip install "fastembed @ git+https://github.com/Dylancouzon/fastembed.git@constella-research-preview"
```

## Usage

```python
from fastembed import TextEmbedding

NAME = "REPO_ID"

documents = [
    "Marie Curie conducted pioneering research on radioactivity.",
    "The Nile is a major north-flowing river in northeastern Africa.",
]

model = TextEmbedding(NAME)
embeddings = list(model.embed(documents))
```

Store the returned vectors in a cosine-similarity index. Use constella-zero or constella-nano to
encode queries against that index.

This graph is the document path only. If you use the original Stella model for queries, use its
`s2p_query` prompt as described on the source model card. Do not use this artifact as an
unprompted Stella query encoder.

## How it works

The graph applies Stella's document path and then computes:

```text
masked mean over last_hidden_state -> 2_Dense_1024 -> L2 normalization
```

Inputs are int64 `input_ids` and `attention_mask`. The output is a normalized fp32 array with
shape `(batch, 1024)`. Inputs are truncated at 512 tokens. Paired sequences and `token_type_ids`
are not supported.

## Conversion accuracy

The ONNX graph was compared with the original PyTorch document path on PARITY_FIXTURE_COUNT
Natural Questions passages. The fixtures cover short, medium, boundary-length, and over-length
inputs.

| Comparison | Minimum cosine similarity | Maximum absolute error |
|---|---:|---:|
| ONNX CPU vs PyTorch | PARITY_FP32_COS | PARITY_FP32_ABS |
| ONNX CUDA vs ONNX CPU | 1.000000 | 9.07e-05 |

Output norms were PARITY_FP32_NORMS. Encoding the same text alone or in a ragged batch produced
bit-identical results.

## Files

| File | Precision | Size |
|---|---|---:|
| `model.onnx` | fp32 | about 1.75 GB |

The graph uses ONNX opset 17, standard operators, and no external-data initializers. The tokenizer
uses dynamic padding and truncates at 512 tokens.

An fp16 graph is not included. The tested fp16 conversion reached a minimum cosine similarity of
0.662 against the fp32 reference on CUDA and was not accurate enough to release.

## Training

No training, fine-tuning, or distillation was performed for this conversion. The graph contains
the source Stella weights and the same document-side computation in ONNX format.

## Limitations

- This artifact encodes documents only. It does not apply Stella's query prompt.
- The model is English-only and truncates inputs after 512 tokens.
- The 400M-parameter encoder and 1.75 GB graph are intended for indexing, not lightweight query
  serving.
- Compatibility with the Constella query encoders means they share a vector space. It does not
  mean the query encoders have equal retrieval quality.

## License and provenance

NovaSearch releases the pinned Stella weights under the MIT license. This repository redistributes
the same weights in ONNX form under that license. Stella derives from
[`Alibaba-NLP/gte-large-en-v1.5`](https://huggingface.co/Alibaba-NLP/gte-large-en-v1.5), which is
Apache-2.0 licensed.

The graph was exported with `torch.onnx.export`, opset 17, and constant folding. Its SHA-256 is
`fe31555e2b40767e17487885fb67dcdf0dcee11bef31f42478e55c1ec69a4ea9`.
