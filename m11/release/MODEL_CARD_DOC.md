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
  - feature-extraction
pipeline_tag: feature-extraction
---

# stella_en_400M_v5 — document path, ONNX

An **ONNX conversion of the document side** of
[`NovaSearch/stella_en_400M_v5`](https://huggingface.co/NovaSearch/stella_en_400M_v5), pinned at
revision `ffeb2b7ee715c226d4ffe5e4619f7dbb48624c20`.

**No training, no fine-tuning, no distillation** — the weights are the source weights; this repo
changes their format, not their values. It runs under ONNX Runtime and FastEmbed with the pooling
head baked into the graph, so it needs neither `trust_remote_code` nor torch.

It is published as the document half of an asymmetric pair: this model indexes documents, and
[`constella-zero`](https://huggingface.co/DylanCouzon/constella-zero) — a 31 MB lookup table with
no transformer — encodes queries into the same space on the device (*constella* = constellation +
stella). Used alone, it pairs with the prompted query path of the source stella model, below.

## Usage

```python
from fastembed import TextEmbedding

NAME = "REPO_ID"
doc_model = TextEmbedding(NAME)
docs = ["Marie Curie was a physicist and chemist who conducted pioneering research on radioactivity.",
        "The Nile is a major north-flowing river in northeastern Africa."]
D = list(doc_model.embed(docs))
print(len(D), D[0].shape)     # 2 (1024,)
```

Not in a FastEmbed release yet. Until it is:

    pip install "fastembed @ git+https://github.com/Dylancouzon/fastembed@add-constella-models"

FastEmbed fetches only `model.onnx` and the tokenizer, and does not alter the graph's output —
it matches a direct ONNX Runtime session exactly.

The blocks below continue from this one.

### Sentence Transformers

For the **source** model in torch — a different artifact, not this repo — and the only supported
way to embed *queries*, which need stella's `s2p_query` prompt:

```python
from sentence_transformers import SentenceTransformer

st = SentenceTransformer(
    "NovaSearch/stella_en_400M_v5",
    revision="ffeb2b7ee715c226d4ffe5e4619f7dbb48624c20",
    trust_remote_code=True,
    # required unless xformers is installed; also the setting this graph was exported under
    config_kwargs={"use_memory_efficient_attention": False, "unpad_inputs": False},
)
D_torch = st.encode(docs, normalize_embeddings=True)          # documents: no prompt
Q_torch = st.encode(["who discovered radium?"], prompt_name="s2p_query",
                    normalize_embeddings=True)                # queries: prompt required
```

## Document path only

stella is asymmetric. Queries need the `s2p_query` prompt
(`Instruct: Given a web search query, retrieve relevant passages that answer the query.\nQuery: `),
and nothing here adds it — neither the graph nor FastEmbed's `query_embed`. Embedding a query
through this model without the prompt produces a perfectly valid vector that is simply not
stella's query representation — cosine between the prompted and unprompted encodings of the same
text ran **0.82–0.87** across five short queries here. Nothing errors; the retrieval is just not
the protocol stella was trained for.

So: embed documents here, and embed queries either through Sentence Transformers with the prompt,
or with `constella-zero`, which was distilled to land in this document space and needs no prompt.

Paired-sequence inputs are unsupported: the graph has no `token_type_ids`.

## What the graph computes

    masked mean over last_hidden_state  →  2_Dense_1024 (Linear 1024→1024, bias, Identity)  →  L2

which is stella's `1_Pooling` + `2_Dense_1024` modules folded in. Inputs are `input_ids` and
`attention_mask` (int64); the output is `(batch, 1024)`, already L2-normalized.

`max_length` is **512** — the length the corresponding document index was built at, enforced by the
shipped tokenizer files (see below).

| file | precision | size |
|---|---|---|
| `model.onnx` | fp32 | ~1.75 GB |

opset 17, standard ONNX domain only, no external-data initializers. **There is deliberately no
fp16 graph** — see below.

## Measured parity

Against the torch module (backbone + Dense + normalize) on **259 real passages** from Natural
Questions, frozen as a length-stratified fixture set — tiny / short / mid / near-512 / the 511–513
boundary / over-512 truncated — so the truncation boundary is exercised, not assumed.

| comparison | min cosine | max abs |
|---|---|---|
| `model.onnx` vs the torch module, CPU | PARITY_FP32_COS | PARITY_FP32_ABS |
| `model.onnx` on CUDA vs on CPU | 1.000000 | 9.07e-05 |

Output norms are PARITY_FP32_NORMS. Batch invariance — the same text alone vs inside a ragged
padded batch — is bit-identical. Cosine here is a true cosine, not a bare dot product.

The CUDA row matters in practice: an index built on GPU is valid for CPU search and vice versa.

## Why there is no fp16 graph

One was built and rejected: on `CUDAExecutionProvider` it fell to **min cosine 0.662** against the
fp32 reference over the 259 fixtures. It looks fine on CPU (cos 0.99999923) only because ONNX
Runtime up-converts fp16 there — which also makes it ~10x slower than fp32 on CPU. CPU parity did
not qualify this export.

## Tokenizer deviation from the source repo

stella's own tokenizer files, from the pinned revision, with two edits:

| field | stella | here |
|---|---|---|
| `model_max_length` | 32768 | **512** |
| `max_length` | 8000 | **512** |
| `tokenizer.json: padding` | fixed 512 | **null** |

FastEmbed truncates at `min(model_max_length, max_length)` — 8000 with stella's files — so
documents of 513–8000 tokens would pass through untruncated and **would not reproduce an index
built at 512**. There is no API-level override
([qdrant/fastembed#689](https://github.com/qdrant/fastembed/issues/689)), so editing the shipped
files is the mechanism, not a preference. `padding: null` means no tokenizer-configured padding,
so FastEmbed applies its own batch-longest padding rather than padding every input to 512. The
weights and the graph are untouched.

`config.json` is stella's, with `use_memory_efficient_attention` and `unpad_inputs` set to `false`
— the values the graph was exported under.

## Licence and attribution

- The weights are stella's, and **NovaSearch releases `stella_en_400M_v5` under MIT**. This repo
  redistributes them in ONNX form under the same terms and claims no separate licence.
- stella is trained from [`Alibaba-NLP/gte-large-en-v1.5`](https://huggingface.co/Alibaba-NLP/gte-large-en-v1.5)
  (Apache-2.0), and stella's `modeling.py` carries "Copyright 2024 The GTE Team Authors and Alibaba
  Group". That code is **not** redistributed here — this repo ships no Python model implementation
  — so no Apache-2.0 licence text is included. The lineage is recorded as attribution.
- Cite stella, not this conversion, for the model itself.

## Provenance

Converted from `NovaSearch/stella_en_400M_v5` @ `ffeb2b7ee715c226d4ffe5e4619f7dbb48624c20` with
`torch.onnx.export` (opset 17, constant folding on). No training. Parity fixtures are 259 real
Natural Questions passages, length-stratified and frozen.
