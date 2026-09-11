# M17 status — planning, 2026-09-11

**Planned:** a supported vocabulary extension, joint table/listwise training, and the owner-approved
alias consistency, late-checkpoint averaging and int8 resident-row loading comparisons. Same query
API and frozen document index; Nano/M13 stays unchanged. Branch: `m17-zero-v1.1-planning`.

**Done:** three Luna research notes, local tokenizer/resource diagnostics, and a bounded plan
for the owner-confirmed local RTX 3080. No candidate trained, quality benchmark scored or model
published. Research conclusions and diagnostic limitations: [FINDINGS.md](FINDINGS.md).

**Next (execution session, in this order):** the registry is `DRAFT_NOT_EXECUTABLE` until
step 6 is done; nothing below the pre-clock line touches the 72-hour budget.

1. Read `PLANNING.md`, `registry.json` (especially `decision_protocol`, `candidate_construction`,
   `bucket_populations_and_dose_rule`, `vocabulary_ranking`) and `REVIEW.md`.
2. Pre-clock, no GPU: complete the Kubernetes licensing row in `research/m7-data-licensing.md`
   (revision, clone route, licence text, attribution artifact, decontamination) before any
   download; build the admitted-source support manifest with per-domain counts; build and seal
   the judged panel and the 200-pair alias test, screened against M7 ancestry.
3. Pre-clock: write the one M17 driver, cache schema, averaging export helper and numpy loader
   in `m17src/`; run the tiny synthetic end-to-end rehearsal under `work/m17/rehearsal`.
4. Pre-clock: two independent implementation reviews (Codex Astra plus one other); fix P1s.
5. Time teacher encoding, mining and evaluation at two sizes; apply the dose rule; write the
   measured allocation into the registry.
6. Commit the lock: protocol, vocabulary hash, tokenizer hash, seeds, cache identity. Flip the
   registry status. Only then start the clock and read V0.

Owner rulings already recorded and not to be re-asked: A1 additions, A2 recommendations, `k8s`
pinned, Kubernetes docs in principle. Any new source, cap, teacher or release change still needs
Dylan.

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
