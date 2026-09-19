# M20 findings

Durable lessons from the reserved-four execution, 2026-09-17 to 2026-09-19. Numbers live in
`results/`; status and commands live in `m20/STATUS.md`. This file is the research trail.

## A silent throughput collapse that no OOM reported

Stage A's projection gate stopped the first launch after 100,000 documents at 37 docs/s, projecting
58.3 h past a 52 h cap. The cause was CUDA caching-allocator fragmentation, and every part of
diagnosing it was a measurement that ruled something out:

- **Not the data.** FEVER token length is flat across the two shards that ran at 73 and 25 docs/s:
  means 85.2 and 86.3, p95 252 and 265.
- **Not thermal, and not the GPU.** The same text in 2,000-document calls holds a flat 141 docs/s
  through 40,000 documents at 68–74 °C on stable clocks — above the 100.76 docs/s A100 figure the
  cap was derived from, because the real corpora average 85 tokens against the 153-token SQuAD
  surrogate the benchmark used.
- **Not thread pinning.** 50,000-document calls reproduce 74.1 and 25.4 docs/s in a clean
  8-thread process against the production run's 72.9 and 25.1.
- **The cause.** A length-sorted 50,000-document `teacher.encode` call sweeps batch shapes from
  many short documents to 64x512, so the allocator accumulates segments it cannot reuse: **23.06
  GiB reserved on a 10.24 GiB card for 1.76 GiB of live tensors.**

**The lesson worth carrying.** `num_alloc_retries` stayed **0** throughout. On WSL the driver
oversubscribes into host memory over PCIe instead of raising OOM, so the usual allocator distress
signal never fires and a memory problem presents purely as speed. `results/m20_vram_probe.json`
had measured every tower on all-512-token batches and passed honestly — it could not see this,
because the failure needs a *wide range* of shapes in one call, not a large one.
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` bounds the reservation to 4.40 GiB and restores
179 docs/s, and re-encoding a shard under it reproduces the shard written without it **byte for
byte**, which is what establishes it changes no vector.
Receipt: `results/m20_allocator_probe.json`.

**Corollary for any future host change.** R24 moved execution from an A100 to an RTX 3080 and kept
the registered hour caps, and the host-validation review covered VRAM, RAM and disk but not
throughput. The caps happened to survive because the real corpora are shorter than the surrogate
the benchmark used — luck, not a check. A host change should re-derive the rate the caps assume.

## Recovery machinery is not free, and is not part of the numbers

Two review rounds on stage B produced five P1 findings, all in code paths that existed only so an
unattended run could survive a crash: an archive exporter that reopened protected payloads, a
finalization crash window, an archive-reuse path, a parallel publish path. Each fix drew the next
finding, because each added a second, weaker-authenticated route to something the primary path
already did.

The owner's direction was to cut rather than harden, and it was right: the benchmark numbers come
from authenticated document shards, the registered roster, exact retrieval and validated per-system
atomic outputs. A crash mid-finalization leaves the scores durable and only bookkeeping unfinished;
the archive is a deliverable, not an input to any number. Deleting the reuse path and the parallel
publish path removed three P1s outright, and left a net of +143 lines against the pre-review
baseline, 92 of them tests.

**The rule.** Before adding a recovery path that will only ever run after an irreversible step,
ask what it protects. If the answer is bookkeeping rather than a number, a clear refusal and a
human decision is the cheaper and safer design. New untested code running after the access is spent
is itself a risk.

## What the reserved numbers do and do not establish

- R1 (nano vs bge-small) NDO-3 `+0.0032`, CI `[-0.0069, +0.0134]`. The interval straddles zero;
  this is **not** an equivalence result, and unresolved superiority does not establish equivalence.
  The per-dataset deltas disagree in sign and the query-pooled reweighting gives `-0.0121` with a
  CI excluding zero. The registered weighting is the headline because it was registered first, not
  because it is the only defensible one — report the others beside it.
- R2 (nano vs leaf-ir-asym) NDO-3 `-0.0389`, CI `[-0.0488, -0.0290]`. Nano is behind, consistently
  under every reweighting.
- FEVER is quarantined as the double-contamination sensitivity row. Nano sits ~0.24 below both
  comparators there and bge-small beats stella-query there while losing on all three clean
  datasets. Adversarial review found these **compatible** with the disclosed exposure but
  **declined to attribute them to it** — a pattern consistent with contamination is not a
  measurement of contamination.
- Intervals quantify query resampling only. They exclude training-seed variation.

## Accepted debt

The pre-encode projection gate is not resume-aware: a relaunched tower counts already-encoded
documents as outstanding and can refuse a viable continuation (Sol, 2026-09-19). It is fail-safe —
it wastes time, it cannot corrupt a number — and the fix would mean editing `Projection` in
`m8src/pre_encode.py`, the module that produced the reserved vectors now that the access is spent
and unrepeatable. If it trips, inspect and relaunch rather than patching under pressure.
