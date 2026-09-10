# Nano decision lock — owned by M13, not ready

**No final evaluation, spent tag or LoTTE read until the applicable lock and reviewed executor
are ready.** Renumbering changes no access rule. The final-run registry remains at
`m10/final_run_registry.json`; the LoTTE draft remains at `m10/LOTTE_LOCK.md`.

The active backlog is `m13/EXECUTION.md`. The original fourteen findings and review chronology are
preserved verbatim in
`research/archive/m10-cleanup-2026-09-10/m10/M10_4_DECISION_LOCK.md`.

Build the scoring executor and rehearse on open/synthetic data before certifying this lock.
`m9src/final9.py` still raises `SCORING PATH NOT IMPLEMENTED`; M7's scorer handles table models,
not nano, and cannot be reused unchanged. M9's access machinery is reusable with separate state.

The LoTTE veto belongs **before the expensive build** when applicable. Its metric/slice manifest
is unresolved. Deferring six-set judging does not defer a decision that chooses which model builds.
