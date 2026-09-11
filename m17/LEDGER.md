# M17 ledger

## 2026-09-11 — owner scope

Dylan requested a new branch and **planning only** for Constella Zero v1.1: better vocabulary,
especially S3/k8s, modest performance improvement, a future 72-hour training allowance, the same
document tower, and simple approaches. Small checks on the local RTX 3080 are authorized.
Research uses Luna agents, at most four concurrently (this session uses three plus the parent).
Branch: `m17-zero-v1.1-planning`, based on `3006eab`.

This opens M17 planning; it does not reopen M7/M8 decisions, authorize a long training run,
change release policy or grant access to M13's evaluation surfaces. Existing rules remain.
No `CLAUDE.md` exception has been requested or adopted.

## P0 — local feasibility checks, recorded before observation

Purpose: establish tokenization and resource facts useful for planning. No model selection,
retrieval-quality measurement, corpus download, real-data training or checkpoint export.

- Read the local M11 bundle, M7 freeze and training-checkpoint metadata; preserve their bytes.
- Use only hand-authored technical terms and boundary/repetition fixtures. Inspect original
  segmentation and an in-memory `AddedToken(single_word=True, normalized=True)` extension.
- Time the released CPU query path at two batch sizes after warm-up. These fixtures are not a
  production latency distribution and this Linux box is not the Mac reference edge device.
- Check compositional sum initialization on unique and repeated/shared-piece fixtures under the
  existing sqrt-count pooling. Do not infer semantic accuracy from vector agreement.
- Measure two synthetic cached-target GPU training shapes: 128 and 256 queries, 64 candidates,
  24 tokens/query, 33,594 rows by 1,024 dimensions, fp32 table/Adam and fp16 document vectors.
  A short warm-up plus 20 measured steps per shape is allowed. Synthetic loss updates disposable
  random rows only; rates exclude data preparation, teacher encoding, mining and evaluation.
- Outputs: `results/m17_planning_probe.json`; reproducible diagnostic in
  `m17src/planning_probe.py`. Stop on error/OOM rather than alter a real recipe.

Excluded reads: `results/frozen_eval/untouched-*`, reserved qrels caches, `work/m9reserve`, and
six-set/LoTTE evaluation payloads. No evaluation or training driver is invoked. Research access
logs are recorded in each `research/m17-*-2026-09-11.md` note.

The future experiment design and decision thresholds will be explicitly marked **draft**;
planning measurements do not ratify a training protocol or establish a quality gain.

## 2026-09-11 — scope clarification and P0b

Dylan confirmed the local RTX 3080 and requested broad query-domain coverage in addition to
cloud/software terms. Nano/M13 remains unchanged. A read-only local tokenizer comparison in
response to Dylan's question found Nano's cached bge-small vocabulary map equal to Zero v1's.

P0's repetition fixtures did not actually share constituent IDs across different lexical units
(`3` and `##3` are distinct). They cannot establish general sum-initialization parity. P0b will
check actual shared-piece fixtures (`s3 s`, `k8s eks`, `kubernetes kubectl`), illustrative terms
from other domains, and persist the Nano tokenizer comparison. No quality inference, data
access or training follows. Output: `results/m17_tokenizer_followup.json`. P0's original code
and observation are preserved in commit `2bca40d`; this is a supplemental diagnostic.
