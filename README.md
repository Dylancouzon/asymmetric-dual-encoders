# Constella asymmetric dual encoders

Constella is one frozen Stella document encoder with two interchangeable query encoders. Index
documents once; choose a lookup table (`zero`) or a 35M-parameter transformer (`nano`) per query
without rebuilding the index. On the registered clean-4 exact-search evaluation, Nano beat
bge-small by **+0.017648 nDCG@10** (one-sided 2.5% lower bound +0.003674).

| model | role | public state |
|---|---|---|
| [`stella-en-400M-v5-doc-onnx`](https://huggingface.co/DylanCouzon/stella-en-400M-v5-doc-onnx) | frozen 1024-d document tower | shipped in M11 |
| [`constella-zero`](https://huggingface.co/DylanCouzon/constella-zero) | int8 query lookup table, no transformer | shipped in M11 |
| [`constella-nano`](https://huggingface.co/DylanCouzon/constella-nano) | 34,540,672-parameter query transformer | research preview; weights frozen at [`6bb167dc`](https://huggingface.co/DylanCouzon/constella-nano/tree/6bb167dc6f60d3992602235b8e8aaa374a309168) |

For the story, read [`research/constella-in-plain-english.md`](research/constella-in-plain-english.md).
Canonical, source-traced numbers are in [`m21/BENCHMARKS.md`](m21/BENCHMARKS.md).

## Demo

Try the [`space-travel demo`](demo/README.md) to encode one playful document collection and search
it with both Nano and Zero. It doubles as a smoke test of the three published model artifacts.

## Quickstart

The three native registrations currently live on the preview FastEmbed branch:

```bash
pip install "fastembed @ git+https://github.com/Dylancouzon/fastembed.git@constella-research-preview" qdrant-client
```

This indexes with Stella, then queries the same in-memory Qdrant collection with Nano and Zero;
the optional path variables support already-downloaded model assets.

```python
import os
import numpy as np
from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models

DOC = "DylanCouzon/stella-en-400M-v5-doc-onnx"
NANO = "DylanCouzon/constella-nano"
ZERO = "DylanCouzon/constella-zero"

def load(name, path_variable):
    path = os.environ.get(path_variable)
    return TextEmbedding(name, specific_model_path=path) if path else TextEmbedding(name)

docs = [
    "mRNA vaccines deliver messenger RNA encoding a viral antigen.",
    "The Treaty of Westphalia ended the Thirty Years' War in 1648.",
]
query = "how do mRNA vaccines work?"

doc_model = load(DOC, "CONSTELLA_DOC_PATH")
D = np.stack(list(doc_model.embed(docs))).astype(np.float32, copy=False)

client = QdrantClient(":memory:")
client.create_collection(
    "docs", vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE)
)
client.upsert("docs", points=[
    models.PointStruct(id=i, vector=vector.tolist(), payload={"text": text})
    for i, (text, vector) in enumerate(zip(docs, D))
])

for label, name, path_variable in [
    ("nano", NANO, "CONSTELLA_NANO_PATH"),
    ("zero", ZERO, "CONSTELLA_ZERO_PATH"),
]:
    encoder = load(name, path_variable)
    q = np.asarray(next(iter(encoder.embed([query])))).astype(np.float32, copy=False)
    hits = client.query_points("docs", query=q.tolist(), limit=2).points
    print(label, [(hit.payload["text"], hit.score) for hit in hits])
```

The explicit cast keeps query storage at fp32 across preview-branch revisions.

## Results

Exact-search nDCG@10 for Nano is **0.363080** NFCorpus, **0.217710** SCIDOCS, **0.721097**
SciFact, **0.787116** TREC-COVID, **0.623296** ArguAna† and **0.477765** FiQA†. Against the frozen
comparators, the registered deltas are:

- Nano − bge-small: **+0.017648 clean-4** and **+0.027449 all-six**, both established.
- Nano − LEAF asym: **+0.016181 all-six**, established; **−0.001063 clean-4**, superiority
  unestablished with no equivalence claim.

† Stella discloses training/evaluation contact with ArguAna and FiQA; the other four datasets form
clean-4. Comparator absolutes are unpublished: the evidence contains per-query vectors and deltas.

Zero scores **0.4339 all-six** dense-only. For hybrid retrieval, use Qdrant **DBSF at prefetch 100**:
Zero + BM25 scores **0.4887 all-six / 0.4912 clean-4**. The registered M7 convex0 operator (`w=0.8`,
prefetch 1000) instead scores **0.4911 / 0.4866**; their observed difference establishes neither
superiority nor equivalence. See [`m12/FINDINGS.md`](m12/FINDINGS.md).

Serving cost uses one common synthetic protocol: three fresh processes per model, batch one, four
CPU threads, five warmups and twenty samples; values are medians across three trials.

| query encoder | hydration | first query | warm 20-word p50 | peak RSS | measured model assets |
|---|---:|---:|---:|---:|---:|
| constella-zero | 0.2618 s | 0.3529 ms | 0.1119 ms | 275.4 MiB | 90.1 MiB |
| bge-small | 0.6726 s | 8.2401 ms | 6.8400 ms | 291.0 MiB | 127.6 MiB |
| constella-nano | 0.6907 s | 7.6685 ms | 7.2511 ms | 280.9 MiB | 132.3 MiB |

These are query-encoder timings, not Qdrant or end-to-end workload latency.

## Two traps

- Sentence Transformers' Stella implementation asserts `please install xformers` unless passed
  `config_kwargs={"use_memory_efficient_attention": False, "unpad_inputs": False}`. Those are also
  the settings used for distillation.
- Stella's upstream `tokenizer.json` enables padding to 512. A direct `tokenizers` load therefore
  adds hundreds of `[PAD]` rows unless padding is disabled. The shipped Constella/FastEmbed bundles
  handle this; custom tokenization must call `no_padding()`.

## Repository map

| path | contents |
|---|---|
| `ROADMAP.md`, `PROJECT_STATUS.md` | current milestone and project state |
| `m7/` … `m21/` | status, findings, registrations and review records |
| `m7src/`, `m9src/`, `m10src/` | training and evaluation harnesses |
| `research/` | narrative, literature, licensing and reviews |
| `results/` | committed results and frozen evidence |

**Never overwrite `results/perquery.json`: its frozen comparator vectors cannot be rebuilt.** Start
with `m7/FINDINGS.md`, `m8/FINDINGS.md`, `m9/FINDINGS.md`, `m10/FINDINGS.md` and
`m12/FINDINGS.md`; reproducibility commands and component boundaries are in [`HARNESS.md`](HARNESS.md).
