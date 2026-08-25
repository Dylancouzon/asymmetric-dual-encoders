# BEIR subset + eval methodology

Research date: 2026-08-24. Goal: pick ~5 small BEIR datasets and a minimal, comparable eval stack for benchmarking custom (non-sentence-transformers) document encoders on a Mac (24 GB RAM, MPS).

## 1. BEIR dataset stats

Numbers below are from the official BEIR paper Table 1 (Thakur et al., NeurIPS 2021 Datasets & Benchmarks Track, [arxiv.org/abs/2104.08663](https://arxiv.org/abs/2104.08663)), cross-checked against the [beir-cellar/beir README](https://github.com/beir-cellar/beir) dataset table. Download sizes are live `Content-Length` reads (2026-08-24) against the official zip host, `public.ukp.informatik.tu-darmstadt.de` (still up).

| Dataset | Domain | Corpus (#docs) | Test queries | Avg doc len (words) | Avg query len (words) | Download (zip) |
|---|---|---|---|---|---|---|
| SciFact | Scientific claim verification | 5,183 | 300 | 213.6 | 12.4 | 2.69 MB |
| NFCorpus | Biomedical / nutrition | 3,633 | 323 | 232.3 | 3.3 | 2.33 MB |
| FiQA-2018 | Financial opinion QA | 57,638 | 648 | 132.3 | 10.8 | 17.1 MB |
| ArguAna | Argument / counter-argument retrieval | 8,674 | 1,406 | 166.8 | 193.0 | 3.60 MB |
| SciDocs | Citation prediction | 25,657 | 1,000 | 176.2 | 9.4 | 135.9 MB* |
| TREC-COVID | Biomedical | 171,332 | 50 | 160.8 | 10.6 | — |
| Touché-2020 | Argument retrieval | 382,545 | 49 | 292.4 | 6.6 | — |
| CQADupStack | Duplicate-question retrieval (12 subforums) | 457,199 | 13,145 | 129.1 | 8.6 | — |
| Quora | Duplicate-question retrieval | 522,931 | 10,000 | 11.4 | 9.5 | — |

*SciDocs' zip is disproportionately large because it bundles the original SciDocs evaluation assets (co-view/co-read/cite/co-cite/cluster files) beyond what BEIR's citation-retrieval subtask actually uses; the corpus itself is only 25.6K docs.

### Recommended 5-dataset subset

**SciFact, NFCorpus, FiQA-2018, ArguAna, SciDocs** — total corpus 100,785 docs (all five combined download under 165 MB).

- All five are under the 60K-doc ceiling (FiQA at 57.6K is the largest).
- TREC-COVID, Touché-2020, CQADupStack, and Quora are excluded purely on size (171K–523K docs); CQADupStack's individual subforums are much smaller, but it's a 12-way aggregate task, not a single clean dataset.
- Coverage matches the requested diversity directly: claim verification (SciFact), medical (NFCorpus), financial QA (FiQA), argument retrieval (ArguAna), citation prediction (SciDocs).
- All five appear as a named group ("TREC-COVID, NFCorpus, FiQA-2018, ArguAna, SciFact, and SCIDOCS") in the zero-shot dense-retrieval literature and are individual named tasks on the MTEB leaderboard, so published per-dataset numbers exist for virtually every embedding model, including the ones cited below.
- LightRetriever ([arxiv.org/abs/2505.12260](https://arxiv.org/abs/2505.12260)) evaluates on SciFact, NFCorpus, FiQA, ArguAna, SciDocs, TREC-COVID, Touché-2020, CQADupstack, Quora, MS MARCO, NQ — all 5 recommended sets are in its tables (dropping TREC-COVID/Touché/CQADupstack/Quora only for the size constraint here).

## 2. Eval tooling: what's current in 2026

### `beir` (PyPI)

- Latest: **2.2.0** (2025-06-04). 2.1.0 (2025-02-25) bumped the floor to Python 3.9+ and swapped the unmaintained `pytrec_eval` dependency for the maintained fork `pytrec-eval-terrier`.
- Actively maintained (single maintainer, bursty: a stale 2024, then two releases in 2025), but PyPI classifiers stop at Python 3.12 — no declared/tested 3.13 or 3.14 support, even though `requires-python` is unbounded above 3.9.
- Heavy dependency surface for what we need: pulls in `sentence-transformers`, `transformers`, `torch`, an Elasticsearch client, and optional `faiss`/`tensorflow`/`peft`/`llm2vec` extras — all in service of *beir's own* model wrappers, which a custom raw-vector encoder doesn't use.
- Its actual scoring is a thin wrapper: `EvaluateRetrieval.evaluate()` calls `pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut.1,3,5,10,100,1000", "map_cut...", "recall...", "P..."})`, with `ignore_identical_ids=True` by default (drops any result row where `query_id == doc_id`; irrelevant for our 5 datasets, matters mainly for Quora).

### `mteb` (PyPI)

- Latest: **2.20.1** (2026-08-23 — released yesterday). `requires-python: >=3.10,<3.15`, i.e. explicit Python 3.14 support.
- Underwent a major v2 rewrite recently: the old per-task `AbsTaskRetrieval.py` file layout and the old `MTEB(tasks=...).run(model)` entry point are gone, replaced by `mteb.evaluate(model, tasks=...)` and a `DataLoader`/`BatchedInput` model protocol. Any tutorial referencing `class CustomModel: def encode(self, sentences: list[str], **kwargs)` is describing the old v1 API.
- Current custom-encoder contract (`EncoderProtocol`, from `docs/get_started/usage/defining_the_model.md`): `encode(self, inputs: DataLoader[BatchedInput], task_metadata, hf_split, hf_subset, prompt_type=None, **kwargs) -> Array` — more machinery than a plain `list[str] -> vectors` function, because `inputs` is a batched dataloader, not raw strings.
- Confirmed from source (`mteb/_create_dataloaders.py`): title+text concatenation is exactly `(title + " " + text).strip()` when a title exists, else `text.strip()` — same convention BEIR itself uses.
- mteb re-hosts its own dataset mirrors (e.g. `mteb/arguana` with `default`/`corpus`/`queries` configs) rather than reading `BeIR/*` directly — one more layer of indirection and its own revision pinning to track.

### Recommendation: hand-roll it (skip both frameworks' model-wrapping layer)

Neither package's actual value-add — its own model wrappers — applies here: the encoder already produces raw vectors. What actually determines comparability is (a) identical corpus/query/qrels content and (b) identical nDCG@10 math, and both are available directly, with a much smaller footprint:

- **`datasets` (HF)** — load `BeIR/{name}` (`corpus`, `queries` configs) and `BeIR/{name}-qrels` (`train`/`test` splits) directly; this is the same source both `beir` and `mteb` repackage.
- **`pytrec-eval-terrier`** (PyPI; imports as `pytrec_eval`, drop-in for the dependency `beir` itself now uses) — prebuilt wheels through Python 3.14 on macOS (universal2, arm64+x86_64), so no C compiler needed. The original `cvangysel/pytrec_eval` was last released in 2020, ships source-only, and has open wheel-build issues — a real landmine on a fresh Python 3.14 install.
- **`numpy`** for the retrieval step: brute-force cosine/dot top-k over ≤57.6K vectors per dataset is one matmul, trivially fast on CPU; no `faiss` needed at this scale.

### Exact HF dataset ids (confirmed by loading each page)

| Dataset | Corpus + queries | Qrels | Qrels splits |
|---|---|---|---|
| SciFact | `BeIR/scifact` (configs: `corpus`, `queries`) | `BeIR/scifact-qrels` | train (919), test (339) |
| NFCorpus | `BeIR/nfcorpus` | `BeIR/nfcorpus-qrels` | train/dev/test |
| FiQA-2018 | `BeIR/fiqa` | `BeIR/fiqa-qrels` | train/dev/test |
| ArguAna | `BeIR/arguana` | `BeIR/arguana-qrels` | test only (1,406) |
| SciDocs | `BeIR/scidocs` | `BeIR/scidocs-qrels` | test only (29,928 judgments) |

Load with `datasets.load_dataset("BeIR/scifact", "corpus")` / `"queries"`, and `datasets.load_dataset("BeIR/scifact-qrels", split="test")`; corpus rows have `_id`/`title`/`text`, query rows have `_id`/`text`, qrels rows have `query-id`/`corpus-id`/`score`.

### Python 3.14 landmine check (as of 2026-08-24)

| Package | Version checked | 3.14 status |
|---|---|---|
| `torch` | 2.13.0 (Jul 2026) | Explicit `cp314` macOS arm64 wheel. Fine. |
| `datasets` | 5.0.1 (Jul 2026) | `requires-python >=3.10`; recent release, no red flags. |
| `pytrec-eval-terrier` | 0.5.10 (Oct 2025) | Explicit macOS universal2 wheels for 3.12–3.14. Fine. |
| `mteb` | 2.20.1 (Aug 2026) | Explicit `>=3.10,<3.15`. Fine (only needed if you want cross-checks against its cached leaderboard numbers). |
| `beir` | 2.2.0 (Jun 2025) | Classifiers stop at 3.12 — not declared/tested on 3.14, and its heavier transitive deps (`sentence-transformers`, elasticsearch client) are more likely to lag. This is the actual reason to skip it, not a hard blocker. |
| `pytrec_eval` (original, cvangysel) | 0.5 (2020) | Source-only, no wheels, unmaintained since 2020 — avoid; use `pytrec-eval-terrier` instead (same import name). |

Net: the hand-rolled stack (`torch`, `datasets`, `pytrec-eval-terrier`, `numpy`) is clean on Python 3.14 today. The only package with a real gap is `beir` itself, which the hand-rolled approach sidesteps entirely.

## 3. Per-dataset conventions that affect comparability

- **Title+text concatenation** (identical in official BEIR eval and current MTEB source): `doc_text = (title + " " + text).strip()` if a non-empty title exists, else just `text.strip()`. Single space, no colon, no newline. Apply this before encoding every corpus document.
- **Query/passage prefixes are model-specific, not dataset-specific** — apply the same prefix regardless of which of the 5 datasets is running:
  - **e5 family** (e5-small-v2, etc.): prepend `"query: "` to every query and `"passage: "` to every corpus document (after title+text concat). Omitting these degrades scores.
  - **bge-small-en-v1.5**: prepend `"Represent this sentence for searching relevant passages: "` to queries only; no instruction on passages. The v1.5 line made this optional (small score loss if skipped), unlike bge-v1.
  - **Snowflake arctic-embed**: same query instruction as BGE, `"Represent this sentence for searching relevant passages: "`, queries only.
  - **gte (thenlper/gte-\*)**: no prefix on either side — symmetric usage.
  - **all-MiniLM-L6-v2**: no prefix — symmetric usage, and note its 256-token max sequence length (see ArguAna note below).
- **ArguAna's long-query quirk**: unlike every other BEIR dataset (queries are short natural-language questions, 3–20 words on average), ArguAna's "queries" are full argument passages — average 193 words, almost as long as the 167-word average document, because the task is retrieving a counter-argument given a full argument. This stresses a model's max-sequence-length: `all-MiniLM-L6-v2`'s 256-token limit is not far above what a 193-word query tokenizes to, so truncation risk is real for shorter-context encoders on this dataset specifically, and it partly explains why ArguAna scores vary more between models than other datasets in this set.

## 4. Reference nDCG@10 numbers for harness validation

Target: reproduce these within ±0.5 nDCG points to validate the harness before running the custom encoders.

| Model | SciFact | NFCorpus | FiQA-2018 | ArguAna | SciDocs |
|---|---|---|---|---|---|
| bge-small-en-v1.5 | 0.713 | 0.349 | 0.403 | 0.331 | 0.198 |
| e5-small-v2 | 0.675 | 0.320 | 0.356 | 0.417 | 0.116 |
| all-MiniLM-L6-v2 | 0.645 | 0.314 | 0.369 | 0.331 | 0.217 |

**Sources**: bge-small-en-v1.5 and all-MiniLM-L6-v2 from the mxbai-edge-colbert-v0 tech report, Tables 16–17, "<35M parameters" BEIR comparison ([arxiv.org/abs/2510.14880](https://arxiv.org/abs/2510.14880)). e5-small-v2: SciFact/NFCorpus/FiQA-2018/SciDocs from the MTEB-leaderboard-derived per-task table on [Superlinked's e5-small-v2 model page](https://superlinked.com/models/intfloat-e5-small-v2); ArguAna from [aimodels.fyi's e5-small-v2 MTEB page](https://www.aimodels.fyi/models/huggingFace/e5-small-v2-intfloat). All three models are official MTEB leaderboard entries, so these numbers should be reproducible from the harness described below without re-deriving them from a paper PDF.

## 5. Recommended harness

**Packages** (all confirmed Python-3.14-clean as of 2026-08-24):

```
datasets>=5.0.1
pytrec-eval-terrier>=0.5.10   # imports as `pytrec_eval`
numpy
torch>=2.13.0                  # for the encoder + MPS backend
```

No `beir`, no `mteb` in the actual eval loop — only load `mteb` separately, optionally, if you want to cross-check the reference numbers in section 4 by pulling mteb's cached leaderboard JSON instead of trusting a paper table.

**Eval loop** (per dataset, run once per model):

```python
import pytrec_eval
from datasets import load_dataset
import numpy as np

def load_beir(name):
    corpus = load_dataset(f"BeIR/{name}", "corpus")["corpus"]
    queries = load_dataset(f"BeIR/{name}", "queries")["queries"]
    qrels = load_dataset(f"BeIR/{name}-qrels", split="test")
    return corpus, queries, qrels

def doc_text(row):
    title = row.get("title") or ""
    return f"{title} {row['text']}".strip() if title else row["text"].strip()

# 1. encode — apply the model's own query/passage prefix convention here
doc_vecs = encode_docs([doc_text(r) for r in corpus])          # (N_docs, D)
query_vecs = encode_queries([q["text"] for q in queries])       # (N_q, D)

# 2. brute-force top-k (cosine == dot product on L2-normalized vectors)
sims = query_vecs @ doc_vecs.T                                  # (N_q, N_docs)
topk_idx = np.argsort(-sims, axis=1)[:, :1000]

results = {
    q["_id"]: {corpus[i]["_id"]: float(sims[qi, i]) for i in topk_idx[qi]}
    for qi, q in enumerate(queries)
}

# 3. qrels in trec_eval shape: {query_id: {doc_id: relevance}}
qrels_dict = {}
for row in qrels:
    qrels_dict.setdefault(row["query-id"], {})[row["corpus-id"]] = row["score"]

# 4. score — identical measure strings to beir's own EvaluateRetrieval
evaluator = pytrec_eval.RelevanceEvaluator(qrels_dict, {"ndcg_cut.10"})
scores = evaluator.evaluate(results)
ndcg10 = np.mean([s["ndcg_cut_10"] for s in scores.values()])
```

Notes on the pseudocode: `ignore_identical_ids` isn't needed for these 5 datasets (no shared query/doc ID space, unlike Quora). Keep `topk_idx` at 1000 to match BEIR's own `k_values` convention even though only `ndcg_cut.10` is reported, since results are meant to hold every judged relevant doc for recall-style metrics if you add them later.
