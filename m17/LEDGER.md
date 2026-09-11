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

## 2026-09-11 — A1: add the three follow-up ideas to the plan

Dylan: **“These look like great ideas, add them to the plan, commit and push.”** This accepts
alias consistency, compatible late-checkpoint averaging and retaining int8 rows in memory as
planned comparisons. Their inclusion is authorized and does not need to be asked again.

This pre-observation amendment adds one matched alias arm to the unrun screen, fixes a single
checkpoint-averaging window and selection rule, and defines an int8 loader comparison on the
same model bytes. `m17/registry.json` owns the loss/sample/window constants, comparison/read
counts and rebalanced allocation. Total local time and the recovery reserve stay unchanged.
The earlier four-arm draft and idea shortlist remain in git at `fe68910`.

All arms see the same admitted alias views; the new arm alone enables the equivalence loss.
The finalist's endpoint/average choice is fixed before the audit, with its control using the
same form. Loader choice depends on fixture parity and measured RAM/latency, not new quality
selection. No snapshot-window sweep, runtime query expansion or inference ensemble is added.

The session is still **planning only**. No model training, corpus access, quality measurement,
new data rights, teacher/index change, M13 change, release-policy change or publication follows
from this amendment. Input/panel manifests, real timing, implementation reviews and execution
authorization remain future work. Prior diagnostic artifacts retain their original provenance.

## 2026-09-11 — A2: plan review, six recommendations accepted

A Fable review of the plan verified the registry arithmetic (parameter cap, byte sizes,
candidate mixes, alias slots, allocation total, read counts) and confirmed the listwise arm is
new relative to M7's uniform-distractor KL. It raised six concerns; Dylan: **“Go with all your
recommendations.”** Dylan also clarified that the S3/k8s vocabulary is an internal request, not
a headline, and that new rows must stay broad rather than overly specific.

Pre-observation changes, all in `m17/registry.json` and `PLANNING.md`, previous values at `b9d355e`:

1. Screen routing moves to the pinned development suite; the M17 panel is descriptive, with its
   expected standard error registered.
2. The judged panel is built and sealed before the 72-hour clock; a pre-clock row is added.
3. **New-source ruling in principle:** official Kubernetes documentation (CC BY 4.0) is admitted
   as a small technical slice, capped at 10% of training queries; one further CC BY / Apache
   project-documentation source may be named at the manifest stage. Licence evidence must be
   recorded in `research/m7-data-licensing.md` before download. AWS pages remain out.
4. Final dose cut from 16,000 to 6,000 steps; query cap raised to 600k and bank to 262,144
   documents; a train/held-out divergence check is logged.
5. Anchor documented as inherited and inert; averaging snapshots moved to 4,500/5,250/6,000.
6. Candidate entropy recorded at cache build in the B2 format.

Two additions from the same review: an untrained sum-init export V0 read once, and a per-domain
cap of 1,024 new rows so no field dominates the vocabulary.

Still planning only. Nothing trained, downloaded, scored or published; no protected access.
