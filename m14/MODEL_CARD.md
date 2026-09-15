---
license: mit
language: en
library_name: fastembed
tags:
  - fastembed
  - onnx
  - retrieval
  - asymmetric-dual-encoder
  - research-preview
base_model: BAAI/bge-small-en-v1.5
pipeline_tag: feature-extraction
---

# constella-nano

`constella-nano` is the query side of an asymmetric dual encoder. It maps queries into the
1024-dimensional document space of the published
[`DylanCouzon/stella-en-400M-v5-doc-onnx`](https://huggingface.co/DylanCouzon/stella-en-400M-v5-doc-onnx)
tower. Output vectors are fp32 and L2-normalized.

> **Research preview.** The registered reserved-four evaluation and broad descriptive BEIR-18
> validation are pending. This card reports only the completed six-set evidence below. At preview
> time, the native model entry is on the custom FastEmbed branch `m14-constella-preview`; it is
> **not in upstream FastEmbed**. An upstream PR belongs to later M20 work.

## Installation and native usage

Install the preview branch:

```console
pip install "fastembed @ git+https://github.com/Dylancouzon/fastembed@m14-constella-preview"
```

The example uses FastEmbed's native registrations by model name. It does not call
`add_custom_model`. The optional path variables let the identical block run against local model
directories for offline verification; omit them for normal Hub-backed use.

```python
# m14-card-usage-start
import os

import numpy as np
from fastembed import TextEmbedding

NANO_NAME = "DylanCouzon/constella-nano"
DOC_NAME = "DylanCouzon/stella-en-400M-v5-doc-onnx"

nano_path = os.environ.get("CONSTELLA_NANO_PATH")
doc_path = os.environ.get("CONSTELLA_DOC_PATH")
nano_kwargs = {"specific_model_path": nano_path} if nano_path else {}
doc_kwargs = {"specific_model_path": doc_path} if doc_path else {}

query_model = TextEmbedding(NANO_NAME, **nano_kwargs)
doc_model = TextEmbedding(DOC_NAME, **doc_kwargs)

query = "how do mRNA vaccines work?"
docs = [
    "mRNA vaccines deliver messenger RNA encoding a viral antigen.",
    "The Treaty of Westphalia ended the Thirty Years' War in 1648.",
]
q = np.asarray(next(iter(query_model.embed([query]))))
D = np.stack(list(doc_model.embed(docs)))
assert q.shape == (1024,) and D.shape == (2, 1024)
assert np.isfinite(q).all() and np.isfinite(D).all()

ranking = np.argsort(-(D @ q))
print([(docs[i], float(D[i] @ q)) for i in ranking])
# m14-card-usage-end
```

Nano needs no query prefix. The sibling Stella artifact is document-side only; do not use its
unprompted path as a Stella query encoder.

## Model and training provenance

The student starts from
[`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5) at revision
`5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` (MIT). Its three-layer feature tap and linear head
were trained to match frozen targets in the 1024-dimensional space of
[`NovaSearch/stella_en_400M_v5`](https://huggingface.co/NovaSearch/stella_en_400M_v5) at revision
`ffeb2b7ee715c226d4ffe5e4619f7dbb48624c20` (MIT). The published Stella ONNX sibling supplies the
matching document tower.

The frozen checkpoint saw exactly **199,999,721 examples**, a reconciled shortfall of **279**
from the nominal 200,000,000-example plan. Nine one-row document tail batches each lost 31 rows;
the original failed supervisor receipt remains part of the audit trail.

Training mixed real queries from ESCI, HotpotQA, Mr. TyDi, NQ Open, SQuAD and TriviaQA; a PAQ
sample; harvested titles, headings and claim sentences from Wikipedia, arXiv and the licensed
pool; seven generated query forms; and the eligible document pool. MS MARCO was excluded from
training and used only as a validation surface. The CC BY-SA sources and PAQ retain their source
attribution; the released model weights are MIT.

Every training source was fingerprint-screened against the protected index containing the six
evaluation datasets and COV components. The final assembly records 709 query removals and 79,630
document removals, plus zero cross-role collisions. This is fingerprint-level decontamination;
it does not erase the teacher-contact disclosure below.

## Evaluation

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
bound **+0.006504**, sign-flip **p=0.000540**). It **did not establish clean-four superiority**:
the contrast was **-0.001063**, lower bound **-0.014456**, **p=0.559474**. That unestablished
result is not a claim of parity or equivalence. The **-0.042982** TREC-COVID difference versus
LEAF is the principal per-dataset limitation.

## Serving behavior and costs

Inputs are right-truncated to 512 tokens. The staged tokenizer leaves padding unset so FastEmbed
uses dynamic batch-longest padding. The ONNX graph emits token embeddings; the native
`PooledNormalizedEmbedding` serving family applies attention-masked mean pooling and L2
normalization. This family assignment is required for correct native output.

The measurements below are **synthetic query latencies, not workload estimates**. They used the
same four-thread, batch-one CPU protocol, with medians across three fresh processes.

| model | hydration | first query | warm 20-word p50 | peak RSS | model assets |
|---|---:|---:|---:|---:|---:|
| constella-zero | 0.2618 s | 0.3529 ms | 0.1119 ms | 275.4 MiB | 90.1 MiB |
| bge-small | 0.6726 s | 8.2401 ms | 6.8400 ms | 291.0 MiB | 127.6 MiB |
| constella-nano | 0.6907 s | 7.6685 ms | 7.2511 ms | 280.9 MiB | 132.3 MiB |

## Limitations and pending evidence

- The reserved four remain unestablished and unspent at preview time; no result is claimed for
  them here.
- Broad BEIR-18 validation is pending. The six-set table must not be read as broad domain
  coverage.
- ArguAna and FiQA have disclosed Stella contact, so all-six results are secondary to clean-four.
- Clean-four superiority over LEAF was not established, and no parity or equivalence claim follows.
- TREC-COVID is the main observed per-dataset weakness against LEAF.
- The model is English-only and truncates beyond 512 tokens.

## License

The preview is released under MIT. The bge-small backbone and Stella teacher/document tower are
also attributed above with their pinned MIT revisions. Dataset-source obligations and the
decontamination record remain documented in the project evidence.
