# M8 status

**Stage: PLANNING COMPLETE, 2026-08-29. Next session executes `m8/NEXT-SESSION.md` (overnight,
Dylan offline ~12h). No training has run; no protected set touched.**

The plan is `m8/PLAN-DRAFT.md` **v5** — final draft, gated by four adversarial reviews (three
Codex: 17→14→8 findings; one Opus scientific-judgment pass whose fixes are folded in), with **all
twelve owner rulings recorded** (§4 of the plan). Headlines: pure lookup is the product (no
query-side neural head — that niche belongs to constella-nano/M9); STRICT C2 (the dense table must
beat M7's dense table); byte cap 233 MB int8; LoTTE = mandatory shadow gate pending an overlap
measurement; reserve BOTH new M9 sets (EUR-Lex + USPTO); PMC-OA excluded; synthetic Qwen3 training
queries approved; training-only second teacher allowed; FEVER gets label + sensitivity;
comparators (bge-small + LR-websearch) scored descriptively INSIDE the single access. Release
names LOCKED: **qdrant/constella-zero-m8** and **qdrant/constella-nano-m9**.

Pipeline (binding order): protected freezes → filter → teacher freeze → noise floor → Stage R (one
assembly, one validation) → Stage S (one finalist) → seeds → int8 → ONNX parity → fusion →
manifest → one mandatory LoTTE shadow crossing → freeze → reserved-4 doc pre-encode → the single
access. Ship rule: C1 ∧ C2 ∧ C3 + qualifying-v2-table + point/worst-group guards + six-set
no-regression guard.

## Wake-up note for Dylan (items the overnight session may add to; nothing here yet is blocking)

- (to be filled by the overnight session: P(ship) from the power simulation; LoTTE overlap
  measurement result; teacher-screen table + any swap case needing your sign-off; harrier-0.6b's
  undisclosed training data if it matters; surprises.)

## File contract (hygiene: full context, no pollution)

| file | contract | read when |
|---|---|---|
| `m8/STATUS.md` | ONE SCREEN. Stage, running, blocked. Updated every session before push. | always, first |
| `m8/NEXT-SESSION.md` | The overnight worklist + hard guardrails. Deleted when consumed. | at session start |
| `m8/PLAN-DRAFT.md` | v5, the authoritative plan. Superseded by LEDGER.md at transcription, then archived to `research/m8-planning/`. | while it exists |
| `m8/LEDGER.md` | (next session) Binding pre-registrations, bars, verdicts. Protocol authority. | before any decision |
| `m8/EXPLORED.md` / `m8/RESULTS.md` | dead ends / runs, one row each. | as needed |
| `research/m8-planning/*` | Archival planning record (5 reviews, 4 gates, 4 sweeps/checks). Point at it, never restate; read on demand. | on demand |

Rules: every number carries an artifact pointer; no file restates another (link); STATUS stays one
screen; a future session cold-starts from STATUS + NEXT-SESSION (later: STATUS + LEDGER) alone.
