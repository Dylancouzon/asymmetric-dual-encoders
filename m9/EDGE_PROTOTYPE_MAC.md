# Task for the Mac: Edge prototype round 3 — the storage question, properly this time

Rounds 1 and 2 are in `results/m9_edge_prototype_Apple_M5_Pro.json` and
`bench/edge_prototype_pair.py`. Round 2 fixed the cold-start artifact and produced a real finding
(cheaper search makes the encoder gap matter *more*: the nano ÷ zero ratio climbs 1.43× → 2.81×).

But it did **not** answer the question it was aimed at, and that is what round 3 is for.
~30–45 minutes. Same rules: CPU + Docker, latency and bytes only, **no quality numbers**.

## What went wrong, and it is not your fault

| config | segments |
|---|---|
| fp16 | 2,226 MB |
| **int8** | **3,254 MB** |
| binary | 2,354 MB |

Quantizing made the index **bigger**, because Qdrant keeps the original vectors alongside the
quantized ones so it can rescore. Peak RSS was ~7 GB in every configuration. So round 2 measured
quantization's *latency* benefit and never its *storage* benefit — and storage is the thing that
decides whether a 1M-document index fits on an edge device at all.

## What round 3 must measure

For each of **fp16 / int8 / binary**, configure the collection so the originals are **not resident
in RAM**, and report what actually lands on disk and in memory:

- `vectors_config.on_disk = true` (originals on disk, not RAM), **and**
- `quantization_config.*.always_ram = true` (the quantized copy is the thing kept hot).

That is the configuration the quantization feature exists for: a small hot copy in RAM, the
originals on disk only for rescoring. Report per configuration:

- **segment bytes on disk**, split into original-vector bytes vs quantized bytes if Qdrant exposes
  it (`/collections/<name>` telemetry, or just the on-disk file sizes per segment directory);
- **peak RSS** — this is the number that decides edge feasibility, and round 2's ~7 GB says the
  configuration was wrong, not that quantization does not work;
- **collection load time**;
- per-bucket p50/p95 for **both** query paths, at `ef=default` and `ef=128`, keeping round 2's
  200-search warm-up and randomised bucket order.

Also run one **`rescore=false`** row for int8 and binary. With rescoring off the originals need
never be touched at query time, which is the genuinely small-footprint mode — it costs recall, and
recall is measured on the training box, but the cost side belongs here.

## The number that matters

**Peak RSS for a 1M × 1024 index.** If binary + `always_ram` + `on_disk` originals lands near
~130 MB of hot vectors instead of ~7 GB, then the document index stops dominating the frontier and
the whole edge story changes — which would be the third time this prototype has overturned an
assumption. If it does not, that is equally worth knowing and should be said plainly.

## Setup

```bash
cd <the repo>
git fetch origin && git checkout m9-work && git pull
```

Extend `bench/edge_prototype_pair.py`. Write `results/m9_edge_prototype_<cpu>.json` with
`round: 3`, keeping rounds 1–2 recoverable from git history.

```bash
git add results/m9_edge_prototype_*.json bench/edge_prototype_pair.py
git commit -m "m9: edge prototype round 3 -- storage-configured quantization sweep"
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

Bytes and peak RSS per configuration, the latency table, and whether `rescore=false` changes the
footprint materially. This is the last thing needed from this machine.
