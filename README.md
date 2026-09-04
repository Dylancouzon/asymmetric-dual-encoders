# Asymmetric dual encoders

**How much retrieval quality survives as query-side computation approaches zero?**

A large frozen encoder indexes documents once, in the cloud. On the edge, the query side is as
close to free as we can make it. The deliverable is a frontier, not a leaderboard entry: two
query paths against the *same* document index, each with its quality and its cost measured.

| | query path | status |
|---|---|---|
| **`zero`** | a 30,522 × 1024 int8 lookup table. No transformer, no matmul. | **released** (below) |
| **`nano`** | a ≤35M distilled transformer | M9 missed its bars; M10 is the retry, paused |

Document side for both: [`NovaSearch/stella_en_400M_v5`](https://huggingface.co/NovaSearch/stella_en_400M_v5),
1024-d, frozen, revision-pinned.

---

## Running `zero` (M7)

The model is on the Hub: **https://huggingface.co/DylanCouzon/zero-query-encoder-v1** (private).
MIT, 94 MB. The query side needs `numpy` and `tokenizers` — that is the entire runtime.

```bash
pip install numpy tokenizers huggingface_hub
pip install sentence-transformers        # document side only
```

```python
from huggingface_hub import snapshot_download
import sys, numpy as np

d = snapshot_download("DylanCouzon/zero-query-encoder-v1")
sys.path.insert(0, d)
from zero_encoder import ZeroQueryEncoder

enc = ZeroQueryEncoder(d, variant="int8")           # or "fp16"
q = enc.encode(["how do mrna vaccines work?"])      # (1, 1024), L2-normalized, ~0.1 ms
```

Documents go through the frozen teacher. **Pin the revision** — the table is only valid against
this exact document space:

```python
from sentence_transformers import SentenceTransformer

doc_model = SentenceTransformer(
    "NovaSearch/stella_en_400M_v5",
    revision="ffeb2b7ee715c226d4ffe5e4619f7dbb48624c20",
    trust_remote_code=True,
    config_kwargs={"use_memory_efficient_attention": False, "unpad_inputs": False},
)
D = doc_model.encode(docs, normalize_embeddings=True)   # no prefix on the document side
scores = q @ D.T
```

### With Qdrant

An ordinary dense collection. Both sides are L2-normalized, so `DOT` ranks identically to
`COSINE` and is cheaper.

```python
from qdrant_client import QdrantClient, models

client = QdrantClient(":memory:")
client.create_collection("docs", vectors_config=models.VectorParams(
    size=1024, distance=models.Distance.DOT))
client.upsert("docs", points=[models.PointStruct(id=i, vector=D[i].tolist(),
                                                 payload={"text": t})
                              for i, t in enumerate(docs)])
hits = client.query_points("docs", query=enc.encode([query])[0].tolist(), limit=5).points
```

On the edge, put the table in the store too: a second collection, one point per vocab row,
`hnsw_config=models.HnswConfigDiff(m=0)` — retrieve-by-id only (indexing it inflated the shard
from 466 MB to 1.82 GB for no benefit). See `m7src/edge_demo.py`.

### Two traps

- **stella asserts `please install xformers`** unless you pass the `config_kwargs` above. It is
  also the pinned setting the table was distilled under, so it is correctness, not convenience.
- **stella's `tokenizer.json` ships with padding-to-512 enabled.** A naive `tokenizers` load puts
  ~500 `[PAD]` rows in every bag and cosine against the correct path drops to **0.35**.
  `zero_encoder.py` calls `no_padding()`. The `transformers` path never sees this — padding is
  off by default there.

### What the numbers are

nDCG@10 on six BEIR datasets, exact search, one pre-registered confirmatory run:
**`zero` 0.4339**, fused with BM25 **0.4911**, BM25 alone 0.4174, the teacher symmetric 0.5744.

`zero` **missed its release bar** (LightRetriever dense 0.4583) CI-resolved at −0.0243
[−0.0405, −0.0086], and on the four datasets with no disclosed teacher overlap it is *below*
BM25. The fused system ties OpenSearch's learned sparse retriever (0.4911 vs 0.4868) while its
query side stays a table lookup. Full characterisation, caveats and cost rows: the model card,
or `m7/STATUS.md`.

**If you fuse with BM25, the fusion rule matters.** The published system is convex fusion at
w=0.8, which Qdrant does not implement. **In Qdrant use `Fusion.DBSF` at a shallow prefetch**: on
dev it beats convex at `limit: 10` (0.5517 vs 0.5482) and ties it at 50, and only falls behind at
deep prefetch (0.5580 vs 0.5727 at 1000). `Fusion.RRF` is weaker at every depth, fairly swept —
best `k=3` unweighted, best `k=2 weights=[2,1]` of 24 configurations. Audit: `m12/FINDINGS.md`.

---

## Rebuilding or re-pushing the release

```bash
.venv/bin/python m11/release/verify_bundle.py            # shipped encoder vs the frozen path
.venv/bin/python m11/release/push.py --build             # rebuild work/release/zero-v1
.venv/bin/python m11/release/push.py --push              # gates, then upload (private)
```

Four gates run before any upload: the table bytes hash to `m7/FREEZE.json`'s `table_sha256`,
both training-lineage run records hash to what the freeze recorded, `freeze.assert_releasable`
(no non-commercial source anywhere in the lineage), and the conformance check. `--public` is an
explicit opt-in.

## Reproducing the evaluation

The training and evaluation harness lives in `m7src/` (M7), `m9src/`, `m10src/`. It needs the
gitignored `work/` tree — encode caches, tables, checkpoints — which is machine-local and is
re-derived, not shipped. Environment: `m7/requirements.lock.txt`, `setup-windows.md`.
Protocol and every registered bar: `m7/LEDGER.md`. What ships: `m7/RECIPE.md`.

**`results/perquery.json` must never be overwritten** — frozen comparator vectors regenerated
from caches that no longer exist.

## Repo map

| path | what |
|---|---|
| `CLAUDE.md` | standing directives, stage plan, decision log — **read this first** |
| `m7/` … `m11/` | per-milestone status, ledger, findings, closed avenues |
| `instructions-m*.md` | the mandate each milestone was run under |
| `m7src/`, `m9src/`, `m10src/` | harness, training, evaluation, probes |
| `research/` | literature, licensing, teacher shortlists, adversarial reviews |
| `results/` | every result of record, including the frozen comparators |

Findings worth reading on their own: `m7/FINDINGS.md`, `m8/FINDINGS.md` (twelve probes, no lever
moved the table more than 0.005), `m9/FINDINGS.md` (nano's coverage failure).
