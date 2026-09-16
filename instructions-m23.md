# M23 — upstream FastEmbed pull requests

**Created 2026-09-16 under owner ruling R22 (Dylan).** Takes the upstream half of the original M20
mandate. Depends on upstream FastEmbed's state and reviewer latency, not on our compute, so it runs
on its own clock. It may start once M22 has made the official Nano revision public, because the
registrations PR points at published artifacts and canonical vectors.

## Deliverables

The three-PR plan recorded in `m21/FASTEMBED.md`, each branched from current upstream `main`:

1. **Padding fixes** — the fixed-padding-length fix and the eager `pad_token` default fix. Behaviour
   changes, reviewed on their own.
2. **Dtype fix** — `a4452ac` **plus** `47a5090`, never the former alone (the intermediate variant
   breaks float16). Includes the float16/float32/float64 and overflow tests.
3. **Model registrations** — the three native entries (`constella-nano`, `constella-zero`, the
   Stella document tower) with reference-derived canonical vectors. This PR alone satisfies the
   original "one clean upstream PR" clause. It must not include the unrelated #703 work and must
   not be the preview branch merged wholesale.

Before opening any PR: run the complete upstream test suite to green on a machine with room (M21
could not finish it on the WSL box; it downloads roughly 10 GB into tmpfs) and record the run.

## Constraints

No model bytes change. Canonical vectors are derived from the frozen references in `m14/`, not
from a FastEmbed run. Expected reviewer pushback and answers are in `m21/FASTEMBED.md`.
