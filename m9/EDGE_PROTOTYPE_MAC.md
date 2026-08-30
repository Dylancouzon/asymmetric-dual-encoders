# Task for the Mac: Edge prototype round 4 — does it actually fit? (the last one)

Rounds 1–3 are in `results/m9_edge_prototype_Apple_M5_Pro.json` and `bench/edge_prototype_pair.py`.
~20–30 minutes. Same rules: Docker + CPU, latency and bytes only, **no quality numbers**.

## Why there is a round 4

Round 3 settled the storage side — binary quantization gives a **128 MB** hot copy against
**2,048 MB** of originals, and `binary + rescore=false` is both the smallest and the fastest
configuration measured (0.441 ms zero, 1.446 ms nano). It also found a trap worth keeping: the same
index *with* rescore costs **8.467 ms**, nineteen times slower, because rescoring against
memory-mapped originals thrashes.

What it could not settle is the question the whole edge premise rests on. Peak RSS came out at
**5.9–7.0 GB in every configuration**, including the one holding 128 MB of hot vectors — because
RSS counts resident pages of the memory-mapped originals, and those are evictable page cache.
**RSS is the wrong instrument.** It measures what the kernel happened to keep, not what the process
needs.

The right instrument is a hard memory limit: give the container less RAM than the index and see
whether it still serves.

## The measurement

Run Qdrant in a memory-capped container against the **binary + `rescore=false`** collection from
round 3 (originals `on_disk`, quantized copy `always_ram`), at these limits:

```
docker run -m 256m  ...
docker run -m 512m  ...
docker run -m 1g    ...
docker run -m 2g    ...
```

For each limit report:

- **serves or dies** — and if it dies, how (OOM-kill, refusal to load, timeouts);
- **collection load time**;
- per-bucket p50/p95 for both query paths at `ef=default`, keeping round 3's 200-search warm-up and
  randomised bucket order;
- container memory actually used (`docker stats` / cgroup `memory.current`), which unlike RSS is
  the constrained truth.

Add an fp16 row at whichever limits it survives, as the contrast — that is the configuration
someone would reach for by default.

## What the answer means

A 1M-document index is 2 GB of original vectors. If binary + `rescore=false` **serves at 512 MB or
below**, the document index stops dominating the frontier and the pair is genuinely deployable on
edge-class hardware. If it needs multiple gigabytes regardless of quantization, then the honest
report says the *document index*, not the query encoder, is what keeps this off a phone — and every
query-side millisecond this project has optimised is beside the point.

Both outcomes are publishable. Do not tune the configuration to reach the nicer one; report the
limit at which it first works.

## Setup

```bash
cd <the repo>
git fetch origin && git checkout m9-work && git pull
```

Extend `bench/edge_prototype_pair.py`. Write `results/m9_edge_prototype_<cpu>.json` with
`round: 4`; git history keeps rounds 1–3.

```bash
git add results/m9_edge_prototype_*.json bench/edge_prototype_pair.py
git commit -m "m9: edge prototype round 4 -- memory-constrained serving"
git push origin m9-work
```

## Do not do these

- **Nothing that feeds a quality decision on this machine.** MPS is not bit-identical to CUDA.
- **Do not touch the held-out sets**: `results/frozen_eval/untouched-*`, `work/m9reserve/`,
  `work/dev/cqadup-android.json`, `work/dev/cqadup-english.json`, `work/lotte/`, or any HuggingFace
  cache for BeIR fever, BeIR dbpedia-entity, mteb cqadupstack-android / cqadupstack-english.
- **Do not edit** `m9/LEDGER.md`, `m9/registry.json`, or anything under `m9src/` — a guard on the
  training box refuses every run while those differ from what it froze. `bench/` and new files are
  fine.
- **Do not merge to `main`.**

## Report back

The limit at which binary + `rescore=false` first serves, what failure looks like below it, and the
latency at each surviving limit. That single number — the smallest RAM that serves a 1M × 1024
index — is the one the edge story needs, and it is the last thing needed from this machine.
