# M20 status — opened, not executed

**Opened 2026-09-16 under R22 (`m13/RULINGS.md`).** Planning session only: mandates written and
reviewed, no code changed, no pod started, no protected payload opened. Reserved access verified
**UNSPENT** on 2026-09-16 (no `m8-reserved-spent` tag locally or on origin;
`results/m10_final_run.json` ends `INCOMPLETE_RESERVED`).

## Next action

Follow `instructions-m20.md` in order. First deliverable is the pre-observation registration
(`m20/REGISTRATION.md`, `m20/beir15_registry.json`, dated amendment in
`m10/final_run_registry.json`), pushed before any executor change is run against real data.
Planning inputs that save the next session a lookup are in `m20/PLAN.md`.

## Blockers

None recorded. Live pod state and wallet were not checked in the planning session; check both
before renting.

## Pointers

- Mandate: `instructions-m20.md`. Rulings: `m13/RULINGS.md` R19, R20, R22.
- Tested executor base and cost pin: `m13/RESERVED_EXECUTION.md`. Artifact paths, hashes, restore
  commands, reserved checklist steps 1–3: `m14/HANDOFF.md`.
- Follow-on milestones: `instructions-m22.md` (release), `instructions-m23.md` (upstream PRs).
