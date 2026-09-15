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

**stella-en-400M-v5-doc-onnx** is the shared document tower for an asymmetric retrieval family.
It indexes documents once, in the cloud; the small
[`constella-zero`](https://huggingface.co/DylanCouzon/constella-zero) and stronger
[`constella-nano`](https://huggingface.co/DylanCouzon/constella-nano) query encoders can both search
that same 1024-dimensional index without re-encoding it. The document side is not cheap: this is a
400M-parameter transformer, deliberately moved out of the per-query path.

This is an ONNX conversion of
[`NovaSearch/stella_en_400M_v5`](https://huggingface.co/NovaSearch/stella_en_400M_v5), pinned at
`ffeb2b7ee715c226d4ffe5e4619f7dbb48624c20`. There was no training, fine-tuning, or distillation:
the format changed, not the weights. The graph includes pooling and normalization and needs
neither torch nor `trust_remote_code`.

> **Research preview.** Reserved-four evaluation and broad descriptive BEIR-18 validation for the
> family are pending and unspent; no result is claimed for either. The three models are registered
> on the preview branch below, not in an upstream FastEmbed release yet.

## Usage

```console
pip install "fastembed @ git+https://github.com/Dylancouzon/fastembed.git@constella-research-preview"
```

```python
import numpy as np
from fastembed import TextEmbedding

NAME = "REPO_ID"
doc_model = TextEmbedding(NAME)
docs = [
    "Marie Curie conducted pioneering research on radioactivity.",
    "The Nile is a major north-flowing river in northeastern Africa.",
]
D = np.stack(list(doc_model.embed(docs)))
assert D.shape == (2, 1024) and D.dtype == np.float32
```

FastEmbed fetches `model.onnx` and the tokenizer. Its native registration passes through the
graph's normalized output unchanged.

### Sentence Transformers

For the source model in torch—a different artifact—and its supported prompted query path:

```python
from sentence_transformers import SentenceTransformer

st = SentenceTransformer(
    "NovaSearch/stella_en_400M_v5",
    revision="ffeb2b7ee715c226d4ffe5e4619f7dbb48624c20",
    trust_remote_code=True,
    config_kwargs={"use_memory_efficient_attention": False, "unpad_inputs": False},
)
D_torch = st.encode(docs, normalize_embeddings=True)
Q_torch = st.encode(
    ["who discovered radium?"], prompt_name="s2p_query", normalize_embeddings=True
)
```

## How it works

The source Stella model is asymmetric. This graph is the **document path only**. Queries for the
source model require its `s2p_query` prompt; neither this graph nor FastEmbed adds it. For this
family, use Zero or Nano instead. Those query encoders share this document space, but that
geometric compatibility is not a claim of retrieval parity, equivalence, or a tie.

The graph computes:

```text
masked mean over last_hidden_state → 2_Dense_1024 → L2 normalization
```

Inputs are int64 `input_ids` and `attention_mask`; output is normalized fp32 `(batch, 1024)`.
Inputs are English-only, truncate at 512 tokens, and do not support paired sequences or
`token_type_ids`.

## Files

| file | precision | size |
|---|---|---:|
| `model.onnx` | fp32 | ~1.75 GB |

The graph uses opset 17, standard ONNX operators, and no external-data initializers. Tokenizer
metadata changes Stella's 32768/8000 length declarations to 512 and disables fixed-512 padding so
FastEmbed uses dynamic batch-longest padding. The weights and graph are unchanged.

There is deliberately no fp16 graph. A candidate reached only **0.662 minimum cosine** against the
fp32 reference on CUDA. Its CPU result was misleading because ONNX Runtime up-converted it.

## Measured parity

The fp32 graph was checked against the torch document path on 259 frozen, length-stratified Natural
Questions passages, including the 511/512/513 boundary and over-length truncation.

| comparison | minimum cosine | maximum absolute error |
|---|---:|---:|
| ONNX CPU vs torch | PARITY_FP32_COS | PARITY_FP32_ABS |
| ONNX CUDA vs CPU | 1.000000 | 9.07e-05 |

Output norms are PARITY_FP32_NORMS, and ragged-batch invariance is bit-identical.

## Licence and attribution

NovaSearch releases the pinned Stella weights under MIT; this repo redistributes them in ONNX form
under the same terms and claims no separate licence. Stella derives from
[`Alibaba-NLP/gte-large-en-v1.5`](https://huggingface.co/Alibaba-NLP/gte-large-en-v1.5)
(Apache-2.0), but this conversion redistributes no Python model implementation. Cite Stella for
the model itself.

## Provenance

Converted from `NovaSearch/stella_en_400M_v5` at
`ffeb2b7ee715c226d4ffe5e4619f7dbb48624c20` with `torch.onnx.export`, opset 17 and constant folding.
No training. Parity fixtures are 259 frozen, length-stratified Natural Questions passages.
