# M8 status

**Stage: PLANNING, 2026-08-29. No training has run; no protected set touched; nothing frozen
beyond what `instructions-m8.md` pre-registered on 2026-08-28.**

Current object of work: `m8/PLAN-DRAFT.md` (**v4**) — three Codex gates run (STOP 17 → 14 → 8,
findings converging, all actioned) and **Dylan ruled all ten decision items on 2026-08-29** (§4
table of the draft; headline rulings: pure lookup is the product — no query-side neural head ever
in M8; doc-side head approved IF it fuses into the doc ONNX graph; byte cap 233 MB int8; reserve
BOTH new M9 sets (EUR-Lex + USPTO, we build them); LoTTE adopted as mandatory shadow gate;
index-time adaptation research-only; PMC-OA excluded; synthetic Qwen3 training queries approved;
training-only second teacher allowed if licence-clean; FEVER gets label + sensitivity read).
All three sweeps DONE (teacher: probe stella-1.5B + harrier-0.6b; data: USPTO/EUR-Lex/US-federal
cleared; details in research/m8-planning/). Pipeline: teacher freeze → Stage R (one assembly, one
validation) → Stage S (one finalist) → seeds → int8 → ONNX parity → fusion → manifest → one
mandatory LoTTE shadow crossing → freeze → the single reserved-4 access. Next: gate #4 convergence
check on v4, then write `m8/LEDGER.md` (executable pre-registrations + stats code + power
simulation + probe bars + LoTTE/reserve freezes) and start Phase 0 with the teacher workstream.

One-paragraph state of knowledge: M7 missed its release bar by −0.0243 CI-resolved; five
independent planning reviews (4 Opus + Codex, `research/m8-planning/`) converged on a corrected
diagnosis — the miss is at least as much objective/supervision (degenerate KL term, pair-starved
Phase A, discarded ICT pairs, genre-starved pool) and adaptation-asymmetry as it is "architecture";
several M7 closes were premature (negatives, bigrams-joint, doc-side map). The first plan draft was
itself gated (STOP, 17 findings — including two project-savers: the proposed comparator scoring
would have burned the reserved panel, and full-dose doc-expansion is ~300 days of compute) and
rewritten. New scope 2026-08-28/29: ONNX/fastembed serving requirement (feasibility verified:
days, not weeks); teacher and data questions reopened as first-class workstreams on Dylan's push.

## File contract (Dylan 2026-08-29: hygiene must be impeccable — full context, no pollution)

| file | contract | read when |
|---|---|---|
| `m8/STATUS.md` | ONE SCREEN. Current stage, what is running, what blocks. Updated every session before push. | always, first |
| `m8/PLAN-DRAFT.md` | The design under review. **Transitional**: superseded by LEDGER.md at registration, then archived to `research/m8-planning/` and never updated again. | while planning |
| `m8/LEDGER.md` | (not yet created) Binding pre-registrations, bars, decision rules, verdicts. Append-only in spirit; amendments dated with reasoning. The protocol authority. | before any decision |
| `m8/EXPLORED.md` | (when first avenue closes) One row per dead end + evidence pointer. | before starting anything new |
| `m8/RESULTS.md` | (when first run lands) One row per run: id, config hash, artifact, headline read. | when comparing runs |
| `research/m8-planning/*` | Archival planning record (five reviews, two gates, feasibility checks, sweeps). Long. **Point at it, never restate it; read on demand only.** | on demand |

Rules: every number in m8/ carries an artifact pointer; no file restates another's content (link
instead); STATUS never exceeds one screen — detail moves down the table; a future session should
be able to cold-start from STATUS + LEDGER alone.
