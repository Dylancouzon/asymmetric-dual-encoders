# Task for the Mac: Edge prototype round 2 — fix the cold start, then quantize the index

Round 1 is in `results/m9_edge_prototype_Apple_M5_Pro.json` and `bench/edge_prototype_pair.py`, and
it produced the most report-shaping numbers of the milestone. Two follow-ups, ~45–60 minutes total.
Same rules as before: CPU + Docker, latency and architecture only, **no quality numbers**.

## 1. Fix the cold-start artifact (small, do it first)

The 1–5-word bucket recorded a **115 ms** zero-path search where every other bucket is ~0.8 ms.
That is the index warming on the first timed query, not a property of short queries — and it is
currently the only number in the artifact that cannot be published.

- Add an explicit **warm-up pass** before any timing: at least 200 searches, discarded, per
  `(path, ef)` combination, not just once at the start.
- **Randomise bucket order** per repetition, or interleave buckets, so that if a warming effect
  survives it cannot land entirely in one bucket again.
- Report a `warmup_searches` field so the artifact says what was done.

Round 1's other rows looked stable (p95 within ~15% of p50), so this should simply make the first
bucket join them. If 1–5-word queries are *genuinely* slower after warm-up, that is a real finding
— say so rather than smoothing it.

## 2. Quantize the document index — this is the real edge story

Round 1's asset table is lopsided and nobody had noticed:

| asset | bytes |
|---|---|
| document index (1M × 1024) | **2,225.8 MB** |
| `zero` int8 token table | 270.1 MB |
| `nano` fp16 ONNX | 46.1 MB |

**The index is 8× the larger query asset and 48× the smaller one.** Every "query-side cost" number
this project has produced is rounding error next to it, and Qdrant quantization is the lever that
exists precisely for this. So sweep the document collection's storage:

- **fp16** (round 1's baseline),
- **scalar int8** quantization,
- **binary** quantization if the local Qdrant build supports it at 1024d,
- and with `on_disk` / mmap vectors on and off, since an edge device's constraint is RAM before it
  is disk.

For each configuration report: **segment bytes on disk, peak RSS, collection load time**, and the
same per-bucket p50/p95 search latency for **both** query paths. Rescoring/oversampling settings,
if you enable them, must be recorded — they trade latency for recall and the table has to show it.

**Still no recall here.** Quantization obviously costs quality; that is measured on the training
box against the real stella index. What this task establishes is the **cost side of that trade**,
so the frontier can show what a 4× or 16× smaller index actually buys in bytes, RAM and
milliseconds.

## Setup

```bash
cd <the repo>
git fetch origin && git checkout m9-work && git pull
python m9src/edge_cost.py --threads 4      # writes the nano ONNX graphs if absent
```

Extend `bench/edge_prototype_pair.py` (yours from round 1). Write
`results/m9_edge_prototype_<cpu>.json` — overwriting round 1 is fine, git history keeps it — and
add a `round: 2` field plus `warmup_searches` and the quantization configuration per row.

```bash
git add results/m9_edge_prototype_*.json bench/edge_prototype_pair.py
git commit -m "m9: edge prototype round 2 -- warmed timings and document-index quantization sweep"
git push origin m9-work
```

## Do not do these

- **Nothing that feeds a quality decision on this machine.** Vectors and evaluations must come from
  one consistent device; MPS is not bit-identical to CUDA.
- **Do not touch the held-out sets**: `results/frozen_eval/untouched-*`, `work/m9reserve/`,
  `work/dev/cqadup-android.json`, `work/dev/cqadup-english.json`, `work/lotte/`, or any HuggingFace
  cache for BeIR fever, BeIR dbpedia-entity, mteb cqadupstack-android / cqadupstack-english.
- **Do not edit** `m9/LEDGER.md`, `m9/registry.json`, or anything under `m9src/` — a guard on the
  training box refuses every run while those differ from what it froze. `bench/` and new files are
  fine.
- **Do not merge to `main`.**

## Report back

The warmed per-bucket table, and the quantization sweep as bytes / RSS / load / latency per
configuration. Most interesting outcome to look for: whether int8 or binary quantization changes
the **ratio** between the two query paths — if a smaller index makes search cheaper, nano's
transformer becomes a larger share of the query, and the frontier's shape changes again.
