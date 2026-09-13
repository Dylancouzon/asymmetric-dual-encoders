# M18 code and artifact map

| Path | Purpose |
|---|---|
| `m18/registry.json` | prospective source, model, protocol, recipe and decision constants |
| `m18/STATUS.md`, `PLANNING.md`, `LEDGER.md`, `FINDINGS.md` | compact state, runbook, decisions and durable result |
| `m18src/common.py` | M18 paths, hashes, atomic writes and protected-read/write guard |
| `m18src/source.py` | pinned git/GitHub acquisition and immutable pagination receipts |
| `m18src/corpus.py` | parsing, redaction, chunking, deduplication and artifact metadata |
| `m18src/protocol.py` | family split, structural qrels, strata and one-read confirmation lock |
| `m18src/adjudicate.py` | export the bounded prospective source-only qrel review pool |
| `m18src/vocab.py` | M17 vocabulary logic fork plus M18 compound extraction/T0–T2 policy |
| `m18src/cache.py` | M17 deterministic candidate cache fork |
| `m18src/train.py` | M17 listwise/cosine/alias trainer fork; only new rows train in M18 |
| `m18src/export.py`, `loader_np.py` | int8 bundle gates and standalone NumPy serving path |
| `m18src/evaluate.py` | exact dense, BM25, DBSF@100 and paired bootstrap evaluation |
| `m18src/diagnostic.py` | required step-500 training/export/V0 serving-parity receipt |
| `m18src/rehearse18.py` | one-command synthetic end-to-end rehearsal |
| `work/m18/raw/` | immutable raw git/API snapshot (gitignored; receipts/hashes published) |
| `work/m18/derived/` | parsed corpus, splits, qrels and indexes (gitignored; manifests published) |
| `work/m18/runs/` | resumable cache/training/export outputs (gitignored) |
| `results/m18_*.json` | machine-readable public-in-repo manifests and measurements, never raw audit text |

The initial fork copies `common.py`, `vocab.py`, `cache.py`, `train.py`, `export.py`,
`loader_np.py` and the synthetic rehearsal structure from `m17src`. Semantic differences are
tracked in the ledger and source headers; no M17 registry, lock or protected-screen state is used.
