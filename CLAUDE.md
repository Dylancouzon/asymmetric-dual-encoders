# Asymmetric dual encoders — working guidance

## Purpose

Measure retrieval quality and deployment cost when replacing the query encoder while preserving
one pretrained document index. The document tower is frozen `stella_en_400M_v5`, 1024d.
`zero` is a token-vector lookup table; `nano` is a ≤35M distilled transformer using the same index.
Only zero claims near-zero query compute. Nano must justify quality at roughly bge-small's query
cost, plus the value of sharing stella's index.

This repository is a reusable experimental harness and the evidence base for a whitepaper.
Keep negative results, failed approaches, provenance and limitations alongside successful runs.

## Start here

1. `ROADMAP.md` — current milestones, dependencies and renumbering.
2. The active milestone's `STATUS.md` and `instructions-m*.md` — next work and exit criteria.
3. Its `CODEMAP.md`, then only the result or registration files needed for the task.

M10 closes preparation. M13 owns both pending cloud E arms, the build, final evaluation and costs.
M11 (zero release) and M12 (fusion audit) remain closed. Nano release is M14; the paper is M15.
Harness improvements remain ordinary maintenance.
Do not infer execution readiness from the historical phrase “half A ready to push.”

## Evidence and protocol

- **Never overwrite `results/perquery.json`.** Its frozen comparator vectors cannot be rebuilt
  from the remaining caches. Preserve registered partitions, comparators and statistics.
- No six-set, reserved or LoTTE evaluation outside its registered transaction. M13 must implement
  and rehearse the executor and finish the decision lock before spending access. A recipe-lock
  push alone does not authorize M9's close-out.
- Reserved four: FEVER, DBpedia-entity, cqadup-android, cqadup-english. Do not read
  `results/frozen_eval/untouched-*`, reserved qrels caches or `work/m9reserve` during development
  or review. Preflight uses manifests; protected content belongs inside the executor.
- A decision's protocol changes must precede the observations it governs, be dated, and preserve
  the original registration in git. Never change a computed contrast to improve a result.
  Unrun-family amendments are not permission to re-decide completed families.
- Exact search produces quality numbers. Qdrant/Edge measurements establish deployment behavior
  and latency; ANN recall must not be silently mixed into the quality comparison.
- The paper's headline is the registered clean-4, with all six beside it. “Clean” means no
  disclosed teacher overlap. Stella discloses ArguAna/FiQA and FEVER exposure. Partition
  sensitivity is not a causal contamination estimate.
- Unresolved superiority tests do not establish equivalence. Query-resampling intervals exclude
  training-seed variation. M9's dataset-dependent failure does not isolate coverage from capacity.
- M7's confirmatory fusion operator remains convex0. M12's DBSF@100 recommendation is a disclosed
  product-policy override; its small observed differences have no equivalence CI.

## Constraints and authority

- Student cap **35M**; document tower frozen; one shared index. No larger student experiment or
  change of premise inferred from a diagnostic. Prefer lower cost under a registered tie policy.
- Training sources must permit commercial derived weights. Approved CC BY-SA sources require
  attribution. MS MARCO and other affirmatively licensed non-commercial sources are validation
  only: never gradients, targets, negatives or generation seeds. No-license sources remain out.
  FineWeb is out of this nano recipe in every role. `research/m7-data-licensing.md` has the evidence.
- Direct vector-search competitors remain excluded as shipped components. Large vendors with
  incidental vector services require the recorded justification; exact rulings are archived below.
- Operational fixes, documentation cleanup and implementing a registered branch may proceed
  autonomously. Changes to licences, teacher, bars, cap, protocol or release policy require the
  applicable owner ruling. Existing authorization is not requested again.
- Cloud compute has a recorded **$1,000 ceiling**, not a requirement to spend it. M13 owns the
  measured quote, mandatory allocations and extension cap. Box limitations must not reshape an
  arm; both E arms run together on the cloud GPU. `torch.compile` remains smoke-only for registered
  arms unless separately authorized. This cleanup initiates no rental or evaluation access.
- Mac Apple M5 Pro is the reference edge-cost target; the RTX 3080 box holds historical caches.
  Check the actual runtime before scheduling GPU work; a CUDA torch installation is not a GPU.

## Working and reporting

- **Commit and push coherent batches often**; git history is a source of truth (Dylan, 2026-09-10).
  State what changed and what was verified. Rerun affected checks after changing their inputs.
  Never force-push away research history.
- Keep status short: outcome, next action, real blocker, pointer. Numbers belong in result JSONs,
  constants in registries, module pitfalls in CODEMAP, lessons in FINDINGS, closed avenues in
  EXPLORED. Archive long historical reasoning rather than requiring every session to read it.
- Do not blanket-renumber `M13`-style strings: many are review-finding IDs. Current mandates use
  new numbers; historical records retain theirs, mapped in `ROADMAP.md`.
- Tests use scratch outputs and synthetic fixtures where possible. They must not rewrite real
  results or silently depend on local checkpoints. Use `.venv/bin/python`; `run_tests.sh` covers
  legacy M7 only. `HARNESS.md` documents the current check commands.
- Preserve code/artifact paths (`m10src`, `results/m10_*`, locks, freezes and spent tags).
  Renaming a milestone does not rename a frozen experiment or its access receipt.

## Before a long run

Smoke the real path, resume and realistic shapes. Monitor failures as well as progress
(`Traceback|Error|FAILED|OOM|Killed|assert`, case-insensitive). Use `tail -F` for logs that may
appear later. Read the first progress line and check its rate; inspect host free memory, GPU
memory and allocator retries. Never launch and assume silence means success.

Measure at two sizes before extrapolating. Vary the parameter blamed for a failure. A fixed load
cost divided by row count is not a per-row cost; RSS high-water marks, anonymous pages and file
cache answer different questions. Record the measurement that disproved a diagnosis.

## Reviews and research

Review consequential implementations and surprising findings adversarially. Reproduce actionable
findings and re-review substantive fixes. Two independent reviewers are required before expensive
or irreversible execution. Every brief names files, forbids recursive searches and includes the
reserved read-exclusion above; audit its access log before accepting findings. Reviewing isolated
decision helpers cannot certify an unwritten executor. Track findings once, with an owner and exit.

Before calling a bar unreachable: redo the arithmetic with the best permitted components,
diagnose each failing component, check the literature for that failure, test capability claims
algebraically, and state the measurement that would change the verdict. A registered stop ends
an avenue; it does not turn an undiagnosed failure into evidence against the method.

## Historical authority

The pre-cleanup guidance, full M10 mandate, ledger, planning, status and old future mandates are
preserved byte-for-byte under `research/archive/m10-cleanup-2026-09-10/` with SHA-256 hashes.
Read those sections for a specific past ruling or registration, not as today's task queue.
`research/m1-m6-findings.md`, `m7/FINDINGS.md`, `m8/FINDINGS.md`, `m9/FINDINGS.md`,
`m10/FINDINGS.md` and `m12/FINDINGS.md` are the durable research trail.
