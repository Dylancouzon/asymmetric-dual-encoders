# M17 status — pre-clock execution, 2026-09-11

**Planned:** a supported vocabulary extension, joint table/listwise training, and the owner-approved
alias consistency, late-checkpoint averaging and int8 resident-row loading comparisons. Same query
API and frozen document index; Nano/M13 stays unchanged. Branch: `m17-zero-v1.1-planning`.

**Done (2026-09-11, checkpoint 1):** steps 2 and 3 of the execution checklist. Kubernetes docs
licensing row completed (CC BY 4.0 whole repository, pinned SHA, 1,648 admitted docs, dev-suite
screen clean; attribution file `research/m17-k8s-attribution.md` ships with derived weights, kept
as registered by Dylan). Support manifest, alias training pool (15,573 pairs), provisional judged
panel (553 queries, six domains, sealed audit partition) and 200-pair alias test built. The one
driver, cache, vocabulary, export, numpy loader and evaluator are in `m17src/`; `pytest -q m17src`
passes (141) and the synthetic rehearsal ran clean on the RTX 3080. No candidate trained, no
development-suite or panel read, nothing published. Result JSONs: `results/m17_*`.

**Rulings A3 recorded (see `LEDGER.md`); Dylan owes the judgments and spot check below, nothing else:**

- Judge the 360 Kubernetes query/candidate rows and 59 ambiguous alias senses in
  `results/m17_panel_pending_judgments.jsonl` (options: Dylan alone; Dylan plus a second engineer
  on a double-judged subset with agreement reported; or a narrower ~60-query panel). Until then
  cloud-software has zero judgments and the panel hash is provisional.
- Spot-check the 2% alias sample `results/m17_alias_spotcheck_sample.jsonl` (311 pairs).
- Ruling: science-engineering, medicine, finance and legal are unpopulated in the source-to-domain
  map, so vocabulary breadth is predetermined "narrow". The panel already assigns those domains
  per passage with a keyword classifier. Allowing the same document-level classifier in the
  training source-to-domain map is a pre-lock protocol refinement (no new source); accepting
  "narrow" is the alternative. Also confirm the panel's exclusion of FEVER-train (reserved four,
  Stella exposure) and ESCI (no panel domain).
- Dose rule applied: alias share 16 to 10 pairs per batch, freed slots to general replay, steps
  stay 6,000. No ruling needed; recorded in the registry and `LEDGER.md`.

**Next (in this order):** the registry stays `DRAFT_NOT_EXECUTABLE` until step 6.

4. Pre-clock: two independent implementation reviews of `m17src/` (Codex Astra, then Codex Sol),
   briefs naming files, forbidding recursive searches, with the reserved read-exclusion; audit
   access logs; fix P1s with one Opus agent; one P1 re-check. Cap at two reviews plus one re-check.
5. Time teacher encoding, mining and evaluation at two sizes; write the measured allocation into
   the registry. Protected-surface screening (six, reserved four, LoTTE) belongs inside the
   executor via `protected10.build()` before importing `m9base`; see the step 2a `LEDGER.md` entry.
6. Commit the lock: protocol, vocabulary hash, tokenizer hash, seeds, cache identity, final panel
   hash. Flip the registry status. Only then start the clock and read V0.

**Working model for every M17 session (Dylan, 2026-09-11):** Fable orchestrates; Opus subagents
do execution; at most two subagents run concurrently; subagents never spawn subagents. Codex
Astra and Sol are the reviewers. Commit and push after every coherent batch. Plan a context clear
at each checkpoint: update this file first, push, then Dylan clears. Checkpoints: after step 4
reviews and fixes; after step 5 timings; before the lock and clock start.

**Open dependencies:** supported training terms, independently judged evaluation panel, measured
end-to-end allocation, implementation and two independent execution reviews. No existing rule
was waived; new-source or release-policy changes need specific rulings if later pursued.

Two independent planning reviews and dispositions: [REVIEW.md](REVIEW.md). Local diagnostics,
frozen artifact/lineage hashes, JSON/arithmetic, Python syntax and documentation links checked.

Plan: [PLANNING.md](PLANNING.md). Constants: [registry.json](registry.json).
Authority/probes: [LEDGER.md](LEDGER.md). Paths and pitfalls: [CODEMAP.md](CODEMAP.md).

The three follow-up additions are included in the draft registry and rebalanced local budget;
the total and recovery reserve are unchanged. Evidence:
[additional avenues](../research/m17-additional-avenues-2026-09-11.md).

**Revision A2 (2026-09-11):** plan review accepted in full; see `LEDGER.md` A2. Screen routes on
the development suite, the judged panel is pre-clock work, Kubernetes docs are admitted in
principle pending licence evidence, the final dose is 6,000 steps, and V0 plus a per-domain
row cap were added. Vocabulary breadth across domains is a requirement, not the S3/k8s headline.
A Codex Astra whole-plan review followed; its fourteen findings are dispositioned in `REVIEW.md`
and all planning-level fixes are in the registry. Kubernetes docs remain a proposal until the
licensing row in `research/m7-data-licensing.md` is completed.
