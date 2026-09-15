# M14 research preview — execution plan

Branch: `m14-preview-release`. Authority: owner ruling R20 in `instructions-m14.md`.
Target: research preview of `constella-nano` published 2026-09-16.

Roles for this milestone: Claude orchestrates and approves stage gates, Codex `gpt-5.6-sol`
implements, Codex `gpt-6-astra` reviews at the two gates. The reviewer follows the essential-only
rule in `CLAUDE.md`.

## Standing rules for the worker

- **Commit after every stage**, with what changed and what was verified in the message. Git
  history is the paper trail. Push the branch after each commit. Never force-push.
- **No protected access.** Do not open a reserved payload, do not create `m8-reserved-spent`, do
  not resume any pod, do not read `results/frozen_eval/untouched-*`, reserved qrels caches or
  `work/m9reserve`. No repo-wide recursive content searches across `results/` or `work/`.
- **Never overwrite `results/perquery.json`** or any existing registered result.
  `results/m10_final_run.json` keeps `end_status = INCOMPLETE_RESERVED`; do not relabel it.
- **No over-engineering.** Build the smallest artifact that satisfies the stage. Do not add
  frameworks, abstraction layers, config systems or transaction machinery. If a stage seems to
  need one, stop and report instead.
- **No scope deviation.** The deliverables are exactly stages S1–S7 below. Anything belonging to
  M20 (reserved four, BEIR-18, official release, upstream FastEmbed PR) is out of scope. If you
  believe a stage requires out-of-scope work, stop and report.
- Use `.venv/bin/python`. Scratch outputs go outside `results/`; only stage receipts land there.
- If a hash, parity check or authentication check fails, **stop and report**. Do not improvise a
  workaround, regenerate the artifact, or relax a tolerance.

## Stages

### S1 — restore and verify frozen bytes
Restore the checkpoint and ONNX bundle from the verified backup per `m14/HANDOFF.md`
("Frozen candidate and durable retrieval"), without overwriting an existing target. Re-verify all
21 files against `work/m13cloud-build-preflight-retry-backup/manifest.json` and the manifest's own
SHA-256 `cb5341e8e9bcff8f07c650345b484ea5d19e6b9d9987cc1282b5fb03337c2602`.
Receipt: `results/m14_restore_verification.json` (per-file path, bytes, expected/actual hash,
match). Zero mismatches required. Commit.

### S2 — release staging
Build a fresh staging directory from the verified bytes. Apply only the documented packaging
transforms from `m14/HANDOFF.md` ("Export and serving evidence"): `tokenizer.json` padding set to
`null`, right truncation retained at 512, `model_max_length=512` and `max_length=512` retained.
Record before/after hashes of every staged file and the exact transform applied.
Receipt: `results/m14_staging_record.json`. Commit.

### S3 — parity matrix
Create real, length-stratified fixtures including the 511/512/513 token boundary and negative
controls. The four build texts and eight serving-validation texts are evidence, not a substitute.
Run: ONNX checker; opset/domain/dtype census; Torch reference parity against
`m13src/score13.py:Nano10Student` instantiated from `m10/FREEZE.json`, tokenized with
padding/truncation at 512, compared by true cosine on normalized rows; direct ORT; stock FastEmbed
via the `add_custom_model` bridge in `m14/HANDOFF.md`; and an input longer than 512 tokens.
Report minimum cosine and maximum absolute error per comparison; state the pass thresholds you
used before running. Receipt: `results/m14_parity.json`. Commit. **Then stop for gate A.**

### S4 — custom FastEmbed branch
A branch off current upstream FastEmbed main carrying the three native model entries and
reference-derived canonical vectors. **Do not include the unrelated #703 padding fix.** Run the
upstream test suite. Verify native registration works and that the `add_custom_model` bridge is
not on the released path. This branch is the preview vehicle only; the upstream PR is M20's.
Receipt: branch name, commit sha, test output summary in `results/m14_fastembed_branch.json`.
Commit.

### S5 — preview model card
MIT card for `DylanCouzon/constella-nano`, stating clearly that it is a **research preview** and
that the reserved four and broad BEIR-18 validation are pending. Mandatory content, all from
`m14/HANDOFF.md` and `m13/STATUS.md` — do not invent or round away a number:
bge-small and Stella attribution with their pinned revisions; exact dose 199,999,721 with the
reconciled 279-example shortfall; training data and decontamination summary; ArguAna/FiQA Stella
contact with clean-four as the headline; the six-set table; the TREC-COVID loss to LEAF as the
principal limitation; the unestablished clean-four LEAF superiority stated as unestablished, not
as parity or equivalence; 512-token behavior; normalized 1024-d output; serving costs labelled
synthetic latency, not workload estimates; and a runnable asymmetric usage example against the
published Stella document tower. Then **execute the card's code offline against the staged bytes**
and refuse a wrong repo id, accidental Hub access, missing sibling model or stale import path.
Commit. **Then stop for gate B.**

### S6 — publication
Authenticate the Hub and **require the owner to be exactly `DylanCouzon`**; if it is not, stop and
report. Create `DylanCouzon/constella-nano` private with `exist_ok=False`, upload, verify large
files by LFS oid, take a post-upload hash snapshot, download the published revision into a fresh
directory and confirm it reproduces the staged parity results. Only then make it public.
Do not touch `DylanCouzon/constella-zero` or `DylanCouzon/stella-en-400M-v5-doc-onnx`.
Receipt: `results/m14_publication.json` with commit/revision URLs. Commit.

### S7 — closure
`m14/STATUS.md`: outcome, the preview revision URL, what was verified, and the explicit statement
that reserved access remains unspent and M20 inherits the reserved four, BEIR-18, the official
release and the upstream PR. Commit.

## Gates

- **Gate A** after S3 — Astra reviews restore, staging and parity for evidence integrity and
  over-engineering. Essential-only.
- **Gate B** after S5 — Astra reviews the FastEmbed branch and the card for overclaim, missing
  disclosure, scope deviation and over-engineering. Essential-only. Publication does not start
  until gate B passes.
