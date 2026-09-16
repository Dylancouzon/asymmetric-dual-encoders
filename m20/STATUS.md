# M20 status — registered and implemented; cloud session not started

**Opened 2026-09-16 under R22.** Reserved access is still **UNSPENT**: no `m8-reserved-spent` tag
locally or on origin, and `results/m10_final_run.json` still ends `INCOMPLETE_RESERVED`. No pod has
been resumed and no protected payload has been opened.

Work is on branch `m20-exec`.

## Done

- **Deliverable 1, registration.** `m20/REGISTRATION.md`, `m20/beir15_registry.json` and the dated
  `_amended_2026_09_16` block in `m10/final_run_registry.json`, pushed before any executor change
  ran against real data. Reversing the amendment reproduces the pre-amendment registry
  byte-for-byte, which is the hash the frozen six-set result pinned, so the amendment is provably
  confined to the roster.
- **Deliverable 2, executor extension.** `m20src/roster.py` holds the eight-system roster shared by
  the protected transaction and the BEIR-15 pass. `m13src/reserved_support.py`,
  `m13src/score13.py`, `m13src/reserved_transaction.py`, `m8src/pre_encode.py` and
  `scripts/m13_reserved_cloud.py` extended. 276 `m13src` tests pass.
- **Deliverable 4 and 5 implementations.** `m20src/beir15.py` and `m20src/archive.py`.
- **Stage C smoked end to end** on SciFact, all eight systems, reproducing every independently
  published number. See `m20/SMOKE.md`.
- **Registered DBSF prerequisite met.** `results/m20_dbsf_reproduction.json`, all three deltas
  exactly zero.

## Next action

Two reviews (essential-only, allowlisted files, reserved read-exclusion, access log audited), then
the cloud session in four stages: A pre-encode, B tagged reserved transaction, C BEIR-15,
D archive. Stages A and B are `scripts/m13_reserved_cloud.py`; C and D are
`scripts/m20_beir15_cloud.py`, with the pod stopped in between so the reserved receipt is durable
before the long repeatable work starts.

## Open items for the owner

1. **Object storage.** No account, bucket, credentials or `rclone` binary exists on the box. The
   `D:` half of the archive can be built and verified without it; the object-storage half cannot.
   This blocks the exit criterion "archive verified at both targets", nothing earlier.
2. **Stage C has about 10% cap headroom against the budget, not against the estimate.** BEIR-15
   needs roughly 23.7M new document encodes across three fp32 towers, including Climate-FEVER's
   5.42M, which `m20/PLAN.md` had omitted. The registered caps total 242.2 h, of which 187 h is new
   spend at $337.07 against $342.96 of recorded headroom. If the in-run projection gate trips, the
   run stops cleanly and the choice between raising the ceiling, narrowing BEIR-15 and accepting a
   partial descriptive table is the owner's.
3. **The inherited 55.2-hour reserved allowance priced one document tower of three.** M13's number
   is unchanged and now applies to the tagged scoring stage; the pre-encode has its own registered
   cap. Recorded in `m20/REGISTRATION.md` §5.

## Noted, not fixed

Running the `m10src` suite rewrites `results/m10_contrast_E1.json`, a real registered result file,
which `CLAUDE.md` forbids. It was restored to HEAD. This is pre-existing on `main` and unrelated to
M20, but it means `run_checks.sh` cannot be run casually on a clean tree.

## Pointers

- Registration: `m20/REGISTRATION.md`, `m20/beir15_registry.json`. Smoke: `m20/SMOKE.md`.
- Rulings: `m13/RULINGS.md` R19, R20, R22, R23. Mandate: `instructions-m20.md`.
- Artifact paths, hashes, restore commands: `m14/HANDOFF.md`.
- Follow-on milestones: `instructions-m22.md` (release), `instructions-m23.md` (upstream PRs).
