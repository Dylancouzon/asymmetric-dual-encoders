# Task for the Mac: the Qdrant Edge prototype for the PAIR

~45–60 minutes, CPU + Docker. This is a stated M9/M10 deliverable — *"a Qdrant Edge two-collection
prototype to prove the operational architecture"* (`CLAUDE.md` § Deliverable) — and M5 built one
for `zero`'s lookup-table path only. The pair now has **two** query paths against **one** document
index, and nobody has run them side by side.

An M-series Mac is the right machine for it: this is an edge-latency measurement, the training box
is busy for hours, and nothing here feeds a quality decision.

## What to build

One Qdrant collection of document vectors, queried two ways, timed end to end:

| path | what happens per query |
|---|---|
| **zero** | tokenize → gather rows from the token→vector table → average → normalize → ANN search |
| **nano** | tokenize → ONNX forward (`work/m9onnx/nano-minilm-l6/model.onnx`) → ANN search |

Report, for each path and each query-length bucket (1–5, 6–10, 11–20, 21–50, 51–120 words):
**p50 / p95 end-to-end ms**, split into encode-ms and search-ms, plus collection load time, peak
RSS, and on-disk bytes of the index.

`bench/edge_prototype.py` and `bench/edge_variant.py` from M5 are the pattern to follow — read them
first, reuse what fits, and do not fork `bench/`.

## Two things to get right

1. **Latency only, not recall.** Build the index from **synthetic** 1024-d unit vectors
   (1M rows is a realistic edge shard; say so in the artifact). HNSW search time depends on
   `n`, `dim` and `ef`, not on what the vectors mean. **Recall is measured on the training box
   against the real stella index** — do not attempt it here and do not report a quality number.
2. **Sweep `ef`**, at least `{default, 128, 512}`. M5's finding was that lookup-table query
   vectors are harder for HNSW than transformer ones (−2.1 nDCG at default `ef` on FiQA, mostly
   recovered at 512), so the two paths may not sit at the same latency/quality operating point,
   and the cost table should show `ef` explicitly rather than hide it.

## Setup

```bash
cd <the repo>
git fetch origin && git checkout m9-work && git pull

uv pip install onnx onnxruntime qdrant-client      # torch/transformers already present
python m9src/edge_cost.py --threads 4              # writes the nano ONNX graphs if absent
```

For `zero`'s table: `m7/FREEZE.json` names the released artifact and `m7src/table.py` is the
serving path. If the table `.npz` is not on this machine, **generate a synthetic table of the same
shape** (30,522 × 1024 int8 plus its scale) and say so — the lookup path's latency is a function of
vocabulary size, sequence length and dtype, not of the learned values.

Write to `results/m9_edge_prototype_<cpu>.json`, then:

```bash
git add results/m9_edge_prototype_*.json m9src/<whatever you add>
git commit -m "m9: Qdrant Edge prototype for the pair, measured on the M5 Mac"
git push origin m9-work
```

## Do not do these

- **Nothing that feeds a quality decision on this machine.** Vectors and evaluations must come from
  one consistent device; MPS is not bit-identical to CUDA. This is a *cost/architecture* artifact.
- **Do not touch the held-out sets**: `results/frozen_eval/untouched-*`, `work/m9reserve/`,
  `work/dev/cqadup-android.json`, `work/dev/cqadup-english.json`, `work/lotte/`, or any HuggingFace
  cache for BeIR fever, BeIR dbpedia-entity, mteb cqadupstack-android / cqadupstack-english.
- **Do not edit** `m9/LEDGER.md`, `m9/registry.json`, or anything under `m9src/` that the training
  box is mid-run on — `nano.py`, `screen.py`, `eval9.py`, `guard9.py`, `screen_stats.py`,
  `teacher9.py`, `data.py`, `warmfit.py`, `fp16_gate.py`, `m9base.py`. A guard there refuses every
  run while those differ from what it froze. New files are fine.
- **Do not merge to `main`.**

## Report back

The per-path, per-bucket table, the `ef` sweep, and anything that surprised you — particularly if
the two query paths turn out to need different `ef` to hit the same recall, since that changes the
frontier's cost story rather than just its latency row.
