---
license: cc-by-sa-4.0
language:
- en
pretty_name: Constella workload studio document vectors
size_categories:
- 1M<n<10M
tags:
- embeddings
- qdrant
- beir
---

# Constella workload studio document vectors

Stella (`NovaSearch/stella_en_400M_v5`) document vectors for two BEIR corpora. They let you build the
Qdrant Edge index behind the Constella workload studio without encoding the corpus yourself.

| dataset | passages | file size |
|---|---:|---:|
| `fiqa/` | 57,638 | 118 MB |
| `nq/` | 2,681,468 | 5.5 GB |

Each folder holds `doc_vecs.npy` (float16, shape `[n, 1024]`) and `doc_ids.json`. Row `i` is the
document `doc_ids[i]`. The files contain vectors and BEIR document IDs only, no passage text.
`manifest.json` lists the model revision, encoding settings and SHA-256 of every file.

## Use

The studio's index build downloads these files when no local vectors exist:

```bash
.venv/bin/python m15src/talk_edge.py build fiqa
.venv/bin/python m15/talk/demo/studio/app.py
```

To load them directly:

```python
import json, numpy as np
from huggingface_hub import hf_hub_download

repo = "Qdrant/constella-demo-vectors"
vecs = np.load(hf_hub_download(repo, "fiqa/doc_vecs.npy", repo_type="dataset"))
ids = json.load(open(hf_hub_download(repo, "fiqa/doc_ids.json", repo_type="dataset")))
```

Query with the matching Constella query encoders: [`Qdrant/constella-nano`](https://huggingface.co/Qdrant/constella-nano),
[`Qdrant/constella-zero`](https://huggingface.co/Qdrant/constella-zero), or Stella itself with its query prompt.

## How they were made

- Model: `NovaSearch/stella_en_400M_v5` at revision `ffeb2b7ee715c226d4ffe5e4619f7dbb48624c20`, the
  frozen document tower of Constella.
- Documents: BEIR corpus at the pinned revisions in the source repository, text as `title + " " + text`,
  no prompt, max length 512 tokens, fp32 compute, stored as float16.
- NQ was encoded on an NVIDIA RTX 3080 (CUDA). A parity check re-encoded 256 seeded passages on Apple
  MPS: minimum cosine 0.99994.
- FiQA passed the source repository's reproduction gate: exact search over these vectors reproduces the
  published per-query nDCG@10 of Stella, Zero and Nano (mean within 5e-4, at least 98% of queries within 1e-6).

Code and receipts: [github.com/Dylancouzon/asymmetric-dual-encoders](https://github.com/Dylancouzon/asymmetric-dual-encoders)
(`m15src/talk_package_vectors.py`, `m15src/talk_edge.py`, `results/m15_talk_nq_parity.json`).

## License and attribution

These vectors are derived from text under Creative Commons Attribution-ShareAlike licences, so they are
released under CC BY-SA 4.0.

- Natural Questions (Kwiatkowski et al., 2019), CC BY-SA 3.0, from Wikipedia, via BEIR (Thakur et al., 2021).
- FiQA-2018 (Maia et al., 2018), from StackExchange and Reddit financial posts, via BEIR (Thakur et al., 2021).
- The Stella model is released by NovaSearch under its own licence; see its model card.
