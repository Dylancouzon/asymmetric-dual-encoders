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

Dylan then accepted three further ideas, with one exception: a tiny pre-clock end-to-end
rehearsal, a held-out judged alias test set of about 200 pairs, and preferring whole words over
abbreviations when support is thin. **`k8s` is exempt and pinned**: it is a direct CTO request
and receives a row regardless of support, with that support recorded honestly.

Dylan then asked for a full Codex Astra review of the plan. Its five P1, eight P2 and one P3
findings and their dispositions are in `REVIEW.md`; all were adopted at the planning level and
none required an owner ruling beyond the ones already recorded. The Kubernetes source remains a
proposal until its licensing row is completed. Still planning only.

Still planning only. Nothing trained, downloaded, scored or published; no protected access.

## Step 2a — Kubernetes documentation licensing row and acquisition (2026-09-11)

Pre-clock, no GPU, no protected access. The licensing row in `research/m7-data-licensing.md` is
complete, so the source is **admitted for download**; the earlier draft's "code samples Apache
2.0" claim was wrong and is corrected in the row — a recursive listing of the pinned tree
(15,566 paths) holds exactly one licence file, the root CC BY 4.0 `LICENSE`, and `content/en/
examples` carries no separate grant. Licence text and the publisher's own footer statement are
quoted with URLs; the CC BY 4.0 trademark and no-endorsement carve-outs are quoted verbatim.
Revision pinned by `git ls-remote`: `17133089068629ec12ca15c1bdf36a60d2671a74`. Attribution
artifact to ship with derived weights: `research/m17-k8s-attribution.md`. Acquisition-terms
finding: none restricting training — an anonymous `git clone` presents no clickwrap and no
separate download agreement, unlike the 2024 StackExchange dump terms.

Acquired with `git clone --filter=blob:none` then checkout of that SHA into gitignored
`work/m17/sources/kubernetes-website` (submodules not fetched). English `content/en/docs` only
was extracted to `work/m17/sources/k8s_docs_en.jsonl`; counts, bytes and hashes are in
`results/m17_k8s_source_manifest.json`. This admits nothing to training: every document must
still pass the executor's protected screen and the registry's new-source share and per-domain
caps.

Decontamination: screened against the development suite through the approved fingerprint
interface, in `m10src/cov_screen.py::screen`'s direction and at its threshold. Over 5,553,821
dev documents and 12,772 dev queries: 0 exact and 5 near-matched k8s documents (0.302%), all on
the document side; the flagged paths are named in the manifest and must be dropped or held out
before training. One of them is the repository's own `content/en/docs/test.md`. The six, the
reserved four and LoTTE are **deferred, not waived** — every interface reaching them
materializes protected payloads in-process, which pre-clock development may not do; that screen
belongs inside the M17 executor via `protected10.build()` + `protected10.hits`. Recorded as a
blocker in the manifest.
