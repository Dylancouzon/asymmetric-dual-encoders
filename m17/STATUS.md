# M17 status — pre-clock execution, checkpoint 2 (after step 4), 2026-09-11

**Planned:** a supported vocabulary extension, joint table/listwise training, and the owner-approved
alias consistency, late-checkpoint averaging and int8 resident-row loading comparisons. Same query
API and frozen document index; Nano/M13 stays unchanged. Branch: `m17-zero-v1.1-planning`.

**Done (2026-09-11):** steps 2, 3 and 4. Kubernetes docs licensing row completed (CC BY 4.0,
pinned SHA, 1,648 admitted docs; attribution file `research/m17-k8s-attribution.md` ships in the
bundle). Support manifest, alias training pool (15,393 pairs after excluding 180 held-out
families), provisional judged panel (553 queries, six domains, sealed audit partition) and
200-pair alias test built and regenerated after the review fixes. Ruling A3-3 (per-document
domain classifier at the panel's pinned threshold 3) implemented; all six domains have
supporting documents. The one driver, cache, vocabulary, export, numpy loader and evaluator are
in `m17src/`; `pytest -q m17src` passes (223); the synthetic rehearsal runs clean on the RTX 3080
(`results/m17_rehearsal_step4.json`). Step 4 review loop closed: Codex Astra (29 findings), Codex
Sol (22), one Sol P1-only re-check (3 remaining, fixed). Dispositions in `REVIEW.md`; the
adversarial-only items were dropped under Dylan's ruling ("do not over-engineer, this is a fairly
easy re-training"); the prepared-data-builder items are step-5 entry conditions in the registry.
No candidate trained, no development-suite or panel read, nothing published. Result JSONs:
`results/m17_*`.

**Dylan owes (nothing else):**

- Judge the 360 Kubernetes query/candidate rows and 80 pending alias senses (59 ambiguous plus
  21 truncated extractions) in `results/m17_panel_pending_judgments.jsonl`, per the A3 plan
  (60 selection-partition cloud-software queries first; a second engineer double-judges 20).
  The builders now refuse to overwrite a sheet with any answered row.
- Spot-check the 2% alias sample `results/m17_alias_spotcheck_sample.jsonl` (308 pairs; more than
  5% wrong tightens the abbreviation filter and rebuilds the pool before lock).

**Next (in this order):** the registry stays `DRAFT_NOT_EXECUTABLE` until step 6.

5. Write the prepared-data builder (admitted-query pool, teacher encoding through the teacher's
   own tokenizer, bank mining, `(source, doc id) → domain` join, old-vocabulary parity before
   extension, protected-surface screening via `protected10.build()` before importing `m9base`),
   then time teacher encoding, mining and evaluation at two sizes and write the measured
   allocation into the registry. The smoke must interrupt an active run and resume it.
6. Commit the lock: protocol, vocabulary hash, tokenizer hash, seeds, cache identity, final panel
   hash. Flip the registry status. Only then start the clock and read V0.

**Working model for every M17 session (Dylan, 2026-09-11):** Fable orchestrates; Opus subagents
do execution; at most two subagents run concurrently; subagents never spawn subagents. Codex
Astra and Sol are the reviewers (briefs must allow read-only viewing commands; Codex reads through
the shell). Commit and push after every coherent batch. Do not over-engineer. Plan a context
clear at each checkpoint: update this file first, push, then Dylan clears. Remaining checkpoints:
after step 5 timings; before the lock and clock start.

**Open dependencies:** the pending judgments and spot check above, the step-5 data builder,
measured end-to-end allocation. No existing rule was waived; new-source or release-policy
changes need specific rulings if later pursued.

Reviews and dispositions: [REVIEW.md](REVIEW.md). Plan: [PLANNING.md](PLANNING.md). Constants:
[registry.json](registry.json). Authority/probes: [LEDGER.md](LEDGER.md). Paths and pitfalls:
[CODEMAP.md](CODEMAP.md).

**Revision A2 (2026-09-11):** plan review accepted in full; see `LEDGER.md` A2. Screen routes on
the development suite, the judged panel is pre-clock work, Kubernetes docs are admitted, the
final dose is 6,000 steps, and V0 plus a per-domain row cap were added. Vocabulary breadth across
domains is a requirement, not the S3/k8s headline. The three follow-up additions (A1) are in the
registry with the total and recovery reserve unchanged
([evidence](../research/m17-additional-avenues-2026-09-11.md)).
