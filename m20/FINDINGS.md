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

## A cap gate is only as good as the sample it arms on

Stage C's first launch refused after 0.6 h: 59 docs/s measured, 59.9 h past a deadline the run
actually clears by ~31 h. The gate was not wrong about its own arithmetic. It was wrong about
what a document is.

`Projection` arms at `min_rows=100_000` and extrapolates cumulative **documents per second**. The
BEIR-15 encode order runs `scifact, nfcorpus, scidocs, trec-covid, fiqa, arguana` first --
272,117 documents, **1.1%** of the 23,744,806 volume, and the longest in the set. So the gate
measured the slowest 1.1% and projected it across the other 98.9%.

Measured mean capped-512 token lengths (stella tokenizer, 8,000 evenly spaced documents each):

| corpus | docs | mean tokens | | corpus | docs | mean tokens |
|---|---|---|---|---|---|---|
| msmarco | 8,841,823 | 77.1 | | webis-touche2020 | 382,545 | 234.9 |
| climate-fever | 5,416,593 | 116.9 | | scifact | 5,183 | 326.7 |
| hotpotqa | 5,233,329 | 67.7 | | nfcorpus | 3,633 | 348.0 |
| nq | 2,681,468 | 108.2 | | quora | 522,931 | 16.4 |

Volume-weighted mean **92.2 tokens/doc** against the ~300 the gate sampled.

**Token throughput never moved.** The completed nano-dense tower measured this across all 22
corpora, median docs/s per corpus against the sampled token length:

| corpus | shards | docs/s | tok/doc | tok/s |
|---|---|---|---|---|
| quora | 11 | 887 | 16.4 | 14,547 |
| hotpotqa | 105 | 233 | 67.7 | 15,774 |
| msmarco | 177 | 203 | 77.1 | 15,651 |
| nq | 54 | 140 | 108.2 | 15,148 |
| climate-fever | 109 | 130 | 116.9 | 15,197 |
| webis-touche2020 | 8 | 55 | 234.9 | 12,920 |
| scifact | 1 | 41 | 326.7 | 13,395 |
| nfcorpus | 1 | 41 | 348.0 | 14,268 |

**Documents per second spans 21.6x (41 to 887). Tokens per second spans 1.22x (12,920 to
15,774).** The gate refused on the first number; the machine only ever delivered the second.

The tower finished in **39.9 h**, inside the 36.9-45.4 h the token model bracketed before it
started, against the 59 docs/s extrapolation's 163 h.

**The lesson worth carrying.** Documents per second is not a stable unit across a
non-uniform-length batch, and the arming threshold of a projection gate is a *sampling* decision,
not a warm-up delay. Fixed by `batch_min_projection_rows`: 1,000,000 rows for BEIR-15, which
reaches ~728,000 documents into msmarco, and the unchanged 100,000 for the uniformly short
reserved batch. Raising it delays the gate without weakening it **only because** this order
front-loads the slowest documents, so a prefix sample keeps its pessimistic bias -- a reorder
putting msmarco first would have made the gate optimistic, which is the failure its own docstring
warns against.

Adversarial review (Astra, 2026-09-20) confirmed the diagnosis and added the limit of the claim:
front-loading makes the *initial* sample conservative, it does not make every later prefix
conservative, since climate-fever and touché documents come after millions of short msmarco ones.
Record: `research/m20-stagec-gate-review-astra-2026-09-20.md`.

## A smoke that covers one dataset certifies one dataset

Stage C scoring ran six corpora and then died entering msmarco:
`ValueError: Unknown split "dev". Should be one of ['train', 'validation', 'test']`.

`m20/beir15_registry.json` registers msmarco's split as `dev`, BEIR's standard MS MARCO evaluation
split, and the scorer passed that name straight to Hugging Face. `BeIR/msmarco-qrels` publishes it
as `validation`: the upstream repo at the pinned revision carries `dev.tsv`, and Hugging Face
normalizes `dev` filenames to the `validation` split name. Its `test` split is TREC-DL's 43
queries, which is **not** BEIR's row.

`m20/SMOKE.md` exercised "the complete stage-C path on one real BEIR-15 row, SciFact" and every
number it produced was right. It could not have caught this, because msmarco is the only corpus of
22 whose registered split name differs from its published one — verified across all 22 rather than
fixed one at a time.

**The lesson worth carrying.** A single-dataset smoke proves the path works for that dataset's
shape. Per-dataset metadata — split names, config names, label locations — is exactly the class of
thing it cannot cover, and the failure surfaces hours in, after real work. When a smoke covers one
member of a heterogeneous set, enumerate the metadata assumption across the whole set cheaply
(split names cost one API call each) rather than discovering it serially.

Fixed by reading the scored split from the registry instead of restating it in code, and remapping
only the physical name at the loader, so every score row still records the **registered** name.
Astra reviewed: **GO**, no essential defect, 48 completed rows verified unaffected (means reproduce
within `1.2e-16`, fusion input hashes match their producers).
Record: `research/m20-split-fix-review-astra-2026-09-23.md`.

**An access incident came out of that review** and is recorded there: a web search about the public
cqadupstack *gaming* forum returned a preview of the reserved cqadup-*english* forum's qrels into
the reviewer's tool output. Contained — `work/` is gitignored and the log is untracked, nothing
reached the repository, no finding depended on it, and the reserved four were already spent and
known-test. **A review brief must forbid web searches naming a reserved dataset, not only local
loads**; the brief in force forbade `datasets.load_dataset` and did not anticipate search.

## A staging root that shares inodes is not a copy

`m20src/archive.py:_place` hard-links the fp16 shards into the archive rather than copying them,
and says why: the vectors are ~147 GB, and a second copy beside them on the same volume would need
294 GB the pod does not have. Re-hashing the destination still hashes real content, and rsync and
rclone both read and transfer real bytes, so every downstream guarantee holds.

The thing to not misread is what `work/m20-archive` therefore *is*. It reported 142.6 GiB while the
filesystem's used space moved by about 2 GB — because it is the same bytes under a second name. It
is a **staging** root, and it is not independent of `work/m13-reserved-enc`: lose or corrupt the
encode tree and you lose the archive's vectors with it. The independent copies are the registered
targets, `/mnt/d/constella-archive/beir15/` and object storage.

**The lesson worth carrying.** Check whether a "copy" changed the free-space number. When it did
not, you have a second name, not a second copy — fine for staging, worthless as a backup. Until the
object-storage half lands there is exactly **one** durable copy of the 142.6 GiB, on a drive inside
the machine that already crashed once during this milestone.

## Accepted debt

The pre-encode projection gate is not resume-aware: a relaunched tower counts already-encoded
documents as outstanding and can refuse a viable continuation (Sol, 2026-09-19). It is fail-safe —
it wastes time, it cannot corrupt a number — and the fix would mean editing `Projection` in
`m8src/pre_encode.py`, the module that produced the reserved vectors now that the access is spent
and unrepeatable. If it trips, inspect and relaunch rather than patching under pressure.
