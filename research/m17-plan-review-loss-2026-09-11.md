# M17 draft planning review: loss and execution isolation

Date: 2026-09-11. Planning review only; no experiment, mutation, or execution certification.

## Findings

### P1 — “known positive” is undefined for unlabeled queries (moderate, fix before ratification)

`registry.json` says every 64-item list contains one `known_positive`, while its candidate
note says that for unlabeled queries this slot is “another teacher hit.” Those are different
semantics: an unlabeled query has no positive document ID, and relabeling a teacher hit as a
positive would silently create a supervision label. It also risks duplicate candidates and
changes the list composition by query type.

Fix: define two explicit list constructors before implementation. For labeled pairs, inject
the verified positive and count it toward `candidate_k`; for query-only examples, omit the
positive slot and allocate that slot to a deterministic teacher/v1/random candidate (or use a
separate fixed list-size rule). Record source, candidate IDs, deduplication, and the exact
teacher-score availability. Keep query-only examples out of any positive-specific loss term
unless the registration explicitly defines their target.

### P2 — The listwise loss is under-specified at the normalization boundary (moderate, fix in
registry)

The plan says “normalized-query cosine loss plus KL from the teacher distribution to the
student distribution,” but does not state whether document vectors are normalized, whether
teacher and student logits use exactly the same temperature, or whether the student KL is
computed from candidate scores before/after any query normalization. Since cosine and dot
product rankings can differ if either side is not normalized, this can create an unintended
loss change while the product contract remains normalized output.

Fix: register the exact equations: frozen document vectors normalized once; teacher logits
`t_i = q_stella · d_i`; student logits `s_i = q_zero · d_i`; `p=softmax(t/T)` and
`r=softmax(s/T)` using the single registered `T=0.05`; `KL(p||r)` with a declared reduction.
State whether the cosine term is against the frozen Stella query vector or only the known
positive document, and apply the same formula in L/VL. This also makes the distinction from
an out-of-scope token-vector MaxSim teacher auditable.

### P3 — The 72-hour schedule is a ceiling, but its decision point is not yet operationally
isolated (moderate, fix before expensive execution)

The plan allocates 10 hours to four 4,000-step screen arms and 16 hours to two selected arms
plus seed replication, while the only measured GPU rates are synthetic resident tensors that
explicitly exclude teacher encoding, mining, I/O, checkpointing, and evaluation. The plan
acknowledges this limitation, but the registry has no hard wall-clock checkpoint at which the
four-arm screen must stop if real preparation or throughput overruns. A screen that consumes
the selection window can make the later “final” comparison infeasible without changing the
registered design.

Fix: add a measured time budget and a deterministic stop rule for each phase (including
candidate-cache construction and one end-to-end training step). At the 10-hour boundary,
select only from completed arms and declare incomplete arms stopped; never silently reduce
the final step count or seed replication. If the real-path smoke cannot establish a schedule
that fits, stop as a development candidate as PLANNING.md already permits.

### P4 — Same-index compatibility is sound, but tokenizer replacement needs an explicit index
identity assertion (minor, fix in implementation checklist)

The design correctly keeps every old token ID stable and emits one normalized 1024d vector,
so a new student tokenizer does not require document re-encoding. However, a serialized
tokenizer can alter segmentation of query strings while the fixed index remains Stella's
document space; parity is a property of the output dimension, normalization, teacher revision,
and query preprocessing, not merely equal old IDs. The current plan states these constraints
in prose but does not require a machine-readable index compatibility assertion in the new
bundle.

Fix: require bundle metadata containing the frozen index encoder revision, dimension,
normalization/projection fingerprint, and document-vector artifact hash; make the loader fail
on mismatch. Keep old token IDs stable, and test the new tokenizer's special-token, truncation,
empty-input, punctuation/underscore, and int8 parity fixtures before any quality read.

## What is already appropriately scoped

The draft correctly labels P0/P0b as synthetic or hand-authored diagnostics, keeps static
late interaction out of the product contract, uses existing document vectors for candidate
lists, prohibits protected evaluation surfaces, and marks all constants as
`DRAFT_NOT_EXECUTABLE`. The 3,072-row maximum plus learned scalars remains below the 35M cap
according to the recorded arithmetic. No evidence in the reviewed artifacts supports a gain
forecast; the plan generally states that limitation correctly.

## Access log

Exact local files read: `m17/PLANNING.md`, `m17/registry.json`, `m17/LEDGER.md`,
`m17/STATUS.md`, `m17/CODEMAP.md`, `m17/FINDINGS.md`, `results/m17_planning_probe.json`,
`results/m17_tokenizer_followup.json`, `instructions-m17.md`, and `CLAUDE.md`.

Optional context files read: `research/m17-static-interaction-2026-09-11.md` and
`results/m8_b2_entropy.json`. No recursive search was used. No
`results/frozen_eval/untouched-*`, reserved qrels/cache, `work/m9reserve`, six-set, LoTTE,
or scoring payload was read. The M17 plan and all reviewed artifacts were left unchanged.

## Brief re-check disposition

The four original findings are resolved at the planning level, with execution dependencies
still open:

- P1 is resolved by explicit `candidate_mix_labeled` and `candidate_mix_query_only` entries,
  `has_label`/nullable `positive_id`, positive inclusion within the bank cap, and a teacher-hit
  no-label rule.
- P2 is resolved by the registry's exact normalized `q_t`, `q_s`, and `d` definitions, shared
  temperature/logits, `KL(p_teacher || p_student)` batch-mean reduction, cosine target, and
  effective-row anchor.
- P3 is resolved procedurally by post-lock phase ceilings, stopping dependent work on an
  incomplete phase, forbidding incomplete-matrix selection, and prohibiting silent dose/seed/
  arm cuts. Real end-to-end timing and schedule fit remain execution prerequisites.
- P4 is resolved for the intended contract by build-time equality of document-encoder metadata
  (revision, dimension, pooling, projection, prefixes) with M7. The deliberate decision not to
  bind a general loader to a production-corpus hash or add a caller API is appropriate; bank
  hashes remain provenance. A real implementation must still emit and check that metadata.

The plan remains `DRAFT_NOT_EXECUTABLE`. Support manifests, judged panel, real pipeline timing,
driver implementation, and two independent execution reviews remain explicit gates; this
re-check does not certify them.

Re-check access log: `m17/registry.json`, `m17/PLANNING.md`, `m17/CODEMAP.md`, and
`m17/REVIEW.md`. No other files were read in this re-check; protected-read exclusions and the
no-recursive-search constraint remained in force.
