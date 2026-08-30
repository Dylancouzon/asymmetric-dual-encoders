# Task for the Mac: measure nano's edge serving cost

One job, ~15 minutes, CPU only. The M9 training box is busy for the next several hours and this
measurement does not belong on it anyway — an M-series Mac is a better edge proxy than a desktop
x86, and the protocol requires naming the CPU that produced the numbers.

## Run this

```bash
cd <the repo>
git fetch origin && git checkout m9-work && git pull

uv pip install onnx onnxruntime        # or: python -m pip install onnx onnxruntime
                                       # torch + transformers are already there from M1–M6

python m9src/edge_cost.py --threads 4
python m9src/edge_cost.py --threads 1  # second operating point; overwrites the same file, so
                                       # commit after the run you want as the headline
```

Then:

```bash
git add results/m9_edge_cost_*.json
git commit -m "m9: edge cost table measured on the M5 Mac"
git push origin m9-work
```

## What it measures

Exports the serving graph — backbone → mean pool → `Linear(→1024)` → L2-normalize — at fp32 and
fp16 for **nano-bge-small**, **nano-MiniLM-L6** and the **mdbr-leaf-ir** comparator, then measures
**batch-1** latency per query-length bucket with the tokenizer inside the timed region, plus cold
load, peak RSS and shipped bytes. It writes `results/m9_edge_cost_<cpu>.json` and records the CPU
automatically.

Reference from the Linux box (i7-10700KF, 4 threads, MiniLM-nano fp16, 6–10-word queries):
p50 **18.2 ms**, p95 **27.1 ms**, cold load **82 ms**, shipped **47.0 MB**.

## Expected, not a problem

- **`mdbr-leaf-ir` may fail to export.** The script records the error per model and continues. It
  is a comparator row only — the vendor rule keeps MongoDB's tower out of any release.
- Untrained weights. Latency depends on the architecture and the tokenizer, not on the weights, so
  the numbers are valid before nano is trained.

## Do not do these

- **Nothing that feeds a quality decision on this machine.** Teacher targets and retrieval
  evaluations must come from one consistent device; MPS is not bit-identical to CUDA. This task is
  a *cost* row, which is device-specific by design — that is the whole reason it can run here.
- **Do not touch the held-out sets**: `results/frozen_eval/untouched-*`, `work/m9reserve/`,
  `work/dev/cqadup-android.json`, `work/dev/cqadup-english.json`, `work/lotte/`, or any HuggingFace
  cache for BeIR fever, BeIR dbpedia-entity, mteb cqadupstack-android / cqadupstack-english.
- **Do not merge to `main`** and do not edit `m9/LEDGER.md` or `m9/registry.json` — they are the
  M9.0 lock and a guard on the training box refuses any run while they differ from what it froze.
- Do not run any other M9 script here.

## Report back

The one-line summary the script prints per model, and confirmation that the JSON is pushed. If
`mdbr-leaf-ir` errored, paste the error — it is useful for M10's port, which has to ship a
comparator row.
