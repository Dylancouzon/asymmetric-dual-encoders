---
license: mit
base_model: NovaSearch/stella_en_400M_v5
tags:
  - onnx
  - retrieval
  - feature-extraction
  - sentence-similarity
pipeline_tag: feature-extraction
---

# stella_en_400M_v5 — document path, ONNX

An **ONNX conversion of the document side** of
[`NovaSearch/stella_en_400M_v5`](https://huggingface.co/NovaSearch/stella_en_400M_v5), pinned at
revision `ffeb2b7ee715c226d4ffe5e4619f7dbb48624c20`.

**No training, no fine-tuning, no distillation.** The weights are the source weights; this repo
changes their *format*, not their values.

It exists because a research project needed the document half of an asymmetric encoder pair to run
under ONNX Runtime and fastembed, with the head baked into the graph. It may be useful to anyone
who wants stella's document embeddings without `trust_remote_code` and a torch dependency.

## What the graph computes

    masked mean over last_hidden_state  →  2_Dense_1024 (Linear 1024→1024, bias, Identity)  →  L2

which is stella's `1_Pooling` + `2_Dense_1024` modules, folded in. Input is `input_ids` and
`attention_mask` (int64); output is `embedding`, `(batch, 1024)`, already L2-normalized, so
cosine similarity is a dot product.

**`max_length` is 512.** That is the length the corresponding document index was built at, and the
shipped tokenizer files enforce it (see *Tokenizer deviation*).

**This is the DOCUMENT path only.** stella is asymmetric: queries require its `s2p_query` prompt
(`Instruct: Given a web search query, retrieve relevant passages that answer the query.\nQuery: `).
Nothing here adds that prefix, and fastembed's `query_embed` does not add one either — so using
this model for queries as well as documents is **silently wrong**, not an error you will see. Embed
documents here; embed queries with the prompt applied, through stella itself.

Paired-sequence inputs are not supported: the graph has no `token_type_ids`, which matches
single-string document encoding (where they are all zeros) and nothing else.

## Files

| file | precision | size |
|---|---|---|
| `model.onnx` | fp32 | ~1.75 GB |

opset 17, standard ONNX domain only, no external-data initializers.

**There is deliberately no fp16 graph.** One was built and rejected — see *Why no fp16* below.

## Measured parity

Against the torch module (backbone + Dense + normalize) on **259 real passages** sampled from
Natural Questions and frozen as a length-stratified fixture set — tiny / short / mid / near-512 /
**511–513 boundary** / **over-512 (truncated)** — so the truncation boundary is actually exercised
rather than asserted.

| comparison | min cosine | max abs |
|---|---|---|
| `model.onnx` vs the torch module, CPU | PARITY_FP32_COS | PARITY_FP32_ABS |
| `model.onnx` on CUDA vs on CPU | 1.000000 | 9.07e-05 |

Output norms are PARITY_FP32_NORMS. Batch invariance — the same text alone vs inside a ragged
padded batch — is **bit-identical**.

Cosine here is a true cosine, `dot / (‖a‖·‖b‖)`, not a bare dot product.

The CUDA row matters in practice: the graph returns the same embeddings on GPU as on CPU, so an
index built on one is valid for the other.

## Why there is no fp16 graph

One was built, measured and rejected. It is documented because the failure is easy to repeat:

- It **passes** CPU parity — cos 0.99999923, max-abs 1.79e-04 over all 259 fixtures — and that
  number is meaningless. ONNX Runtime has no fast CPU fp16 kernels, so it up-converts to fp32. The
  same fact makes fp16 roughly **10× slower** than fp32 on CPU.
- On `CUDAExecutionProvider`, where it actually runs in fp16, it disagrees with the fp32 reference
  on **255 of 259 passages** (min cosine 0.662, max-abs 7.4e-02), at every length from 7 tokens
  up. It is 1.78× faster and wrong.

**A CPU parity gate cannot qualify a reduced-precision ONNX graph.** Measure on the execution
provider the precision exists for, or do not ship the precision.

## Usage — ONNX Runtime

```python
import numpy as np, onnxruntime as ort
from tokenizers import Tokenizer
from huggingface_hub import snapshot_download

d = snapshot_download("REPO_ID")

tok = Tokenizer.from_file(f"{d}/tokenizer.json")          # already truncates at 512
sess = ort.InferenceSession(f"{d}/model.onnx", providers=["CPUExecutionProvider"])

docs = ["Marie Curie was a physicist and chemist who conducted pioneering research on radioactivity.",
        "The Nile is a major north-flowing river in northeastern Africa."]
enc = [tok.encode(t) for t in docs]
L = max(len(e.ids) for e in enc)
ids = np.zeros((len(enc), L), "int64")
mask = np.zeros((len(enc), L), "int64")
for i, e in enumerate(enc):
    ids[i, :len(e.ids)] = e.ids
    mask[i, :len(e.ids)] = 1

D = sess.run(None, {"input_ids": ids, "attention_mask": mask})[0]   # (2, 1024), L2-normalized
print(D.shape, np.linalg.norm(D, axis=1))
```

## Usage — fastembed

The graph pools and normalizes internally, so register it with pooling **disabled** and let
fastembed pass the output through untouched.

```python
from fastembed import TextEmbedding
from fastembed.common.model_description import ModelSource, PoolingType

TextEmbedding.add_custom_model(
    model="REPO_ID",
    pooling=PoolingType.DISABLED,      # the graph already pooled
    normalization=False,               # ... and normalized
    sources=ModelSource(hf="REPO_ID"),
    dim=1024, model_file="model.onnx",
    description="stella_en_400M_v5 document path, ONNX", license="mit", size_in_gb=1.8,
)
emb = TextEmbedding(model_name="REPO_ID", specific_model_path=d)
V = list(emb.embed(docs))
print(len(V), V[0].shape)
```

`parallel=` is **not supported** for models registered with `add_custom_model` in fastembed 0.8.0:
the worker process constructs `OnnxTextEmbedding`, which cannot resolve a name registered at
runtime, and raises `ValueError: Model ... is not supported in OnnxTextEmbedding`. Embed serially.

## Tokenizer deviation from the source repo

The tokenizer files here are stella's own, from the pinned revision, with **two edits**:

| field | stella | here | why |
|---|---|---|---|
| `tokenizer_config.json: model_max_length` | 32768 | **512** | fastembed truncates at `min(model_max_length, max_length)`, i.e. **8000** with stella's files. Documents of 513–8000 tokens would then pass through untruncated and **would not reproduce an index built at 512**. |
| `tokenizer_config.json: max_length` | 8000 | **512** | as above |
| `tokenizer.json: padding` | fixed 512 | **null** | so a reader installs dynamic batch-longest padding instead of padding every input to 512 |

There is no API-level override for the truncation limit in fastembed today (see
[qdrant/fastembed#689](https://github.com/qdrant/fastembed/issues/689)), so editing the shipped
files is the mechanism, not a preference. Nothing about the weights or the graph changes; only
readers that honour these fields are affected.

`config.json` is stella's, with `use_memory_efficient_attention` and `unpad_inputs` set to
`false` — the values the graph was exported with. If you load the *source* model in torch you must
pass them too, or it raises `assert ... 'please install xformers'`:

```
# for reference, this is the SOURCE model in torch -- this repo ships ONNX only
cfg = AutoConfig.from_pretrained("NovaSearch/stella_en_400M_v5", trust_remote_code=True)
cfg.use_memory_efficient_attention = False
cfg.unpad_inputs = False
```

## Licence and attribution

- The weights are stella's, and **NovaSearch releases `stella_en_400M_v5` under MIT**. This repo
  redistributes them in ONNX form under the same terms and **claims no separate licence of its
  own**.
- stella is trained from
  [`Alibaba-NLP/gte-large-en-v1.5`](https://huggingface.co/Alibaba-NLP/gte-large-en-v1.5), which is
  **Apache-2.0**, and stella's `modeling.py` carries "Copyright 2024 The GTE Team Authors and
  Alibaba Group". That code is **not** redistributed here — this repo contains no Python model
  implementation, only ONNX graphs — so no Apache-2.0 licence text is shipped. The lineage is
  recorded as attribution.
- Cite stella, not this conversion, for the model itself.

## Provenance

| | |
|---|---|
| source | `NovaSearch/stella_en_400M_v5` @ `ffeb2b7ee715c226d4ffe5e4619f7dbb48624c20` |
| head | `2_Dense_1024`, Linear(1024→1024) with bias, Identity activation |
| exporter | `torch.onnx.export`, opset 17, TorchScript path, constant folding on |
| training | none |
| fixtures | 259 real Natural Questions passages, length-stratified, frozen |
| precisions | fp32 only — see *Why no fp16* |
