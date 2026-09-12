# M13 owner rulings — requested 2026-09-10

Each item names what is undecided, why it matters, the options and the lead's recommendation.
Dylan records a ruling by filling the **Ruling** line with a date; protocol changes must precede
the observations they govern and keep the original registration in git. None of these blocks
writing code; R9, R10 and the provider block renting; the rest block protected access or the
200M build.

## R1 — Preflight standard (blocks final six-set access)
`m10/final_run_registry.json` `implementation` cites M7's preflight, which opens
`results/frozen_eval/<six>.json` before the spent tag exists. `m13/EXECUTION.md` and `CLAUDE.md`
require preflight from manifests only; protected content belongs inside the executor.
Options: (a) manifest-only outside the transaction, payload-level length/duplicate/qid checks
inside, first after the tag (implemented as `access13.Config.payload_checks_inside`, default on);
(b) keep M7's payload-reading preflight. **Recommend (a)** and amend the registry line, dated. The
manifest check hashes the same sorted qid list the payload carries, produced by the same script,
so a payload failure inside is practically unreachable.
**Ruling (Dylan, 2026-09-10):** accepted as recommended: manifests only outside, payload checks inside after the tag; amend the registry, dated.

## R2 — Re-encoding bridge contract (blocks final six-set access)
The bridge re-scores bge-small on the six and compares with its frozen row to show the
re-encoding pipeline reproduces the comparator's conditions. The inherited 0.0003 per-query
tolerance was withdrawn: its "minimum nDCG quantum" rationale is false and it was never
rehearsed. The registry now hard-fails only on qid-set inequality or dataset |mean Δ| > 0.003 and
reports per-query movement. Options: (a) keep that; (b) add a rehearsed per-query bound;
(c) demand identical hardware and dtype. **Recommend (a) plus one open-data rehearsal**: encode
bge-small on open dev components on the box and on the A100, record the per-query envelope in the
build record as report-only. A gate must be measured before it is registered.
**Ruling (Dylan, 2026-09-10):** accepted as recommended: dataset-mean 0.003 is the only hard gate; open-data re-encoding rehearsal recorded report-only.

## R3 — M9 close-out bridge (blocks M9's six-set access)
`m9/final_run_registry.json` still carries `max_abs_per_query_delta: 0.0003` with "failure
consumes the access". Options: (a) dated M9-specific amendment adopting the dataset-mean bridge,
written before any M9 score exists; (b) run under 0.0003 and accept the likely loss of M9's only
access; (c) close M9 descriptively without a bridge. **Recommend (a)**, disclosed beside M9's
scores with the two-build-lock provenance note.
**Ruling (Dylan, 2026-09-10):** accepted as recommended: dated M9-specific amendment adopting the dataset-mean bridge before any M9 score.

## R4 — Paired M9/nano recipe-delta row (blocks either final run)
`instructions-m13.md` requires registering the paired recipe delta before either final run and
says it is not a causal coverage experiment. **Recommend** a one-page `m13/PAIRED_ROW.md` listing
every recipe difference (student, corpus, head, dose, mix, objective, warm start, batch), stating
the row is a paired per-query nDCG@10 delta on identical frozen qids with a bootstrap interval,
descriptive only, with no decision attached. You ratify the page.
**Ruling (Dylan, 2026-09-10):** accepted as recommended: `m13/PAIRED_ROW.md`, descriptive only, ratified by Dylan.

## R5 — Build document policy and full A4 (blocks the 200M build)
The archived lock registers "query epochs ≈ 37 over 4.0M texts, document epochs ≈ 8 over the
6.15M pool". The screen code demanded `ceil(dose × 0.25)` unique documents, 50M at 200M, so the
build could not assemble. Implemented: the build arm uses the FULL A4 (the screen's cut applies
only to `data_cut.applies_to`) and every eligible re-screened document once per epoch, reshuffled
per epoch, position a pure function of the global step. Options: confirm; or fixed-order replay
each epoch (what the screen's stream does). **Recommend confirming reshuffle per epoch**: the lock
names epochs, not order, and reshuffling is the conventional default. Disclosed in the record.
**Ruling (Dylan, 2026-09-10):** accepted as recommended: full uncut A4; every eligible document once per epoch, reshuffled per epoch.

## R6 — `ratified_by_owner` flip (blocks final six-set access)
The registry permits exactly one post-commit edit: flipping this to true. The executor refuses
to open the six without it. Options: flip now; flip in the commit that pins the reviewed executor
and freeze. **Recommend the latter**, so your ratification covers the executor and configuration
that actually run, not a draft.
**Ruling (Dylan, 2026-09-10):** accepted as recommended: flip in the commit that pins the reviewed executor and freeze.

## R7 — `min_free_gb` and reserved bookkeeping (blocks reserved access)
M9's registry required 120 GB free before reserved encodes; M10's registry has no such field and
no code had one. `access13` defaults to 120, configurable. **Recommend** adding
`reserved.min_free_gb: 120` to M10's registry as dated bookkeeping, alongside R10 and R12.
**Ruling (Dylan, 2026-09-10):** accepted as recommended: add `reserved.min_free_gb: 120` to M10's registry, dated, with R10 and R12.

## R8 — LoTTE metric, seven slices, identities (blocks the pre-build gate)
`instructions-m9.md`:171 registers LoTTE-clean as 7 slices, 20,122 queries, macro over slices,
never pooled; `m10/LOTTE_LOCK.md` registers the veto (margin 0.004, paired bootstrap B = 10,000
seed 903, one-sided 97.5% upper bound) and the identities (candidate = the selected recipe's 5M
A100 arm; comparator = `E-bs32`; veto skipped but the observational row still read in the bs32
branch). The METRIC and the slice list are not yet written down where the executor can read them.
Options: nDCG@10 (every other number in the repo, and what the 0.004 margin was calibrated
against) or Success@5 (LoTTE's native metric). **Recommend nDCG@10** as the veto metric, with
Success@5 reported beside it descriptively, and a `m13/LOTTE_GATE.json` registration listing the
seven slice identifiers and their query counts, committed before any LoTTE read. No LoTTE content
is read to write it: `work/lotte/inventory.json` and `PROVENANCE.md` name the slices.
**Ruling (Dylan, 2026-09-10):** accepted as recommended: nDCG@10 is the veto metric, Success@5 reported beside it; `m13/LOTTE_GATE.json` registers the seven slices before any read.

## R9 — Clear `pending` on both E arms (blocks E1, so blocks renting for E)
`contrasts.compute` returns `not_computed` for any arm carrying `pending`, by registered
semantics (`arms.E-bs128._pending`). Clearing the flag edits `m10/screen_registry.json`, changes
its sha, and invalidates the bindings of `results/m10_F_verdict.json` and the ten computed
contrast records. Options: (a) one dated pre-observation commit that strips `pending` from
`E-bs32` and `E-bs128` (keeping the `_pending` history notes), re-issues the F verdict with
`contrasts.f_verdict_from`, recomputes the ten contrasts and asserts every contrast value is
byte-identical (only `registry_sha256` fields move); (b) change `contrasts.py` to compute once a
record exists, leaving the registry alone. **Recommend (a)**: it is what the registry says the
field means, and the numbers do not move. (b) silently redefines a registered field.
**Ruling (Dylan, 2026-09-10):** accepted as recommended: one dated pre-observation commit strips `pending`, re-issues the F verdict and recomputes the ten contrasts byte-identically.

## R10 — Reserved-batch allowance (blocks renting: the cap formula needs it)
The conditional reserved batch (fires iff a conjunct rejects) needs stella document vectors for
FEVER and DBpedia-entity, about 10M passages that do not exist in `work/enc`, plus the two small
cqadup components. The archived budget table has no line for it. **Recommend** a named line of
8 A100-hours (estimate; measured on day one by timing 10,000 passages) plus 60 GB of disk, reserved
before `max_extension_cycles` is computed.
**Ruling (Dylan, 2026-09-10):** "fine if needed": add the reserved-batch allowance line (8 A100-hours estimate, 60 GB disk), re-measured on day one.

## R11 — Crash after the tag, before the first persisted dataset (blocks final access)
`score13` treats this window as an outright loss: with no persisted registry sha there is nothing
to authenticate a continuation against. The registry's `_six_crash` names post-tag continuation
for "incomplete scores" and `infra_retry_admissible_iff` allows a retry only pre-tag. Options:
(a) strict loss; (b) permit continuation authenticated by the BEGIN commit alone (HEAD == BEGIN,
registry at BEGIN, tag present, zero scores). **Recommend (b) as a dated amendment**: the BEGIN
commit pins code and registry as firmly as a persisted sha does, and a crash in the first
dataset's encode would otherwise forfeit the only access for a hardware hiccup. If you prefer the
registry's literal reading, (a) is already implemented.
**Ruling (Dylan, 2026-09-10):** accepted as recommended: continuation authenticated by the BEGIN commit alone, as a dated amendment.

## R12 — `paths_guard` allowlist for the reserved stage (blocks reserved access)
`paths_guard.claim()` verifies the calling module against the allowlist, so `score13`'s reserved
stage cannot borrow `m8src.final_run`'s entry. **Recommend** adding the entry in the same dated
commit as R7 and R10 (LEDGER 15 amendment). Bookkeeping, no protocol change.
**Ruling (Dylan, 2026-09-10):** accepted as recommended: allowlist entry for the reserved stage in the same dated commit as R7 and R10.

## R17 — LoTTE upper-bound quantile method (blocks the LoTTE read)
`m10/LOTTE_LOCK.md` registers a one-sided 97.5% upper bound (B = 10,000, seed 903) but not the
quantile METHOD. `inverted_cdf`, the repo's convention (`m9src/final_stats.bootstrap`,
`m10/final_run_registry.json`), takes the 9,750th order statistic; the reflected convention takes
the 9,751st. The Astra review (2026-09-10) built a boundary case where the two disagree on the veto.
**Recommend** pinning `inverted_cdf` in `m13/LOTTE_GATE_REGISTRATION.json` before any read; the
executor refuses a registration without the field.
**Ruling (Dylan, 2026-09-10):** accepted as recommended — "Fully fix and review everything until
we're at a full GO" — pinned in the registration's dated amendment.

## R18 — Slice pin and checkpoint manifest before the LoTTE read (blocks the LoTTE read)
Counts alone do not authenticate a slice, and the arm records the gate reads are mutable files. Two
committed artifacts close both: `results/m8_lotte_pin.json` from `m8src/freeze_lotte.py pin` (M8's
registered E10-REMEDY PIN, allowlisted, never executed — it hashes the remediated files and reads no
score), run immediately before the gate on the day of the read; and `m13/LOTTE_GATE_MANIFEST.json`,
the lock's second manifest commit, written by `lotte_gate13.py --write-manifest` from the published
E records and committed before the read. The executor refuses without either and compares every
field. **Recommend** both.
**Ruling (Dylan, 2026-09-10):** accepted as recommended, same words as R17. The pin runs on the day
of the read, not before; nothing under `work/lotte` is opened during development or review.

## Provider (your choice; not a registered field)

**Ruling (Dylan, 2026-09-10):** decided later; RunPod Secure Cloud is the standing recommendation.
Registered spec: one A100 80 GB (H100 only if its cost per example measures lower on the
day-one smoke), ≥ 500 GB persistent disk that survives instance stop, SSH, a GitHub deploy key
for the headless commit-and-push contract, the instance stopped between stages. Every price below
is my knowledge as of mid-2026 and UNVERIFIED; the registry requires the billed price to be
measured on day one.

| Option | Fit | Approximate on-demand price | Notes |
|---|---|---|---|
| **RunPod Secure Cloud** (recommended) | A100 80 GB SXM, network volume in the same data center, pod stop keeps the disk, SSH, no egress fee | A100 ≈ $1.2–1.7/h, H100 ≈ $2.0–2.7/h, volume ≈ $0.07/GB/month | Pick a data center that offers both the GPU and network volumes; Community Cloud is cheaper but not for an irreplaceable build |
| **Lambda Cloud** (second) | Simple, reliable A100/H100, persistent filesystems in some regions | A100 ≈ $1.3–1.8/h, H100 ≈ $2.5–3.0/h, storage ≈ $0.20/GB/month | Single-A100 availability is intermittent; check the filesystem region matches the GPU region |
| Hyperstack or Nebius | On-demand A100/H100 with persistent volumes, SSH | A100 ≈ $1.3–1.6/h, H100 ≈ $1.9–3.0/h | Good fallbacks if the first two have no capacity |
| Vast.ai / TensorDock | Cheapest marketplace GPUs | A100 ≈ $0.8–1.3/h | Host machines vary; disk persistence depends on the host staying listed. Not for a one-shot 200M build whose checkpoint is irreplaceable |
| GCP / AWS / Azure | Robust disks and SSH | A100 ≈ $3.5–4/h, quota requests | Two to three times the price; only sensible with credits |

Plan for the disk: 500 GB for three to four weeks is roughly $35–100, above the archived "$25
disk and egress" line; the allocation table carries the measured figure. Start on an A100 for
the benchmark and both E arms; choose the build's hardware from measured cost per example at the
selected batch (at bs32 the step is launch-bound and an H100 gains little; at bs128 it may win).

## Scope cut — Dylan, 2026-09-10 (after the implementation review)

Asked plainly whether stage 1 was over-engineered, the lead answered yes in two places: the
extension cycles and the crash-recovery machinery around the one-shot access. Dylan agreed with all
four recommendations below.

| # | Ruling (Dylan, 2026-09-10) |
|---|---|
| R13 | **No extension cycles.** The build is a fixed 200,000,000 examples, three cycles, kill and plateau the only stop rules. The lock's "permitted extensions" are not exercised; the budget becomes a fixed allocation table with no cap formula. |
| R14 | **No post-tag continuation.** The registry's original wording stands: a crash after the tag consumes the access and the executor reports what was persisted. The R11 amendment is withdrawn unexecuted. `--recover` recomputes decisions from persisted scores only, with zero protected reads. Reliability comes from rehearsing the exact scoring code on open data until it is boring. |
| R15 | **One more review, P1s only.** Finish the two fix passes, keep the fixes that survive R13/R14, one short Codex re-check of the P1 findings, then stop reviewing and rent the GPU. |
| R16 | **LoTTE gate as a small script**, since it is registered and costs about an hour of GPU; no general executor. |
