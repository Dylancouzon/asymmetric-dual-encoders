# M16 — image model (noted, not scoped)

Previously M15; moved 2026-09-10. Unscheduled; text execution has priority.

See `ROADMAP.md`: an IMAGE asymmetric dual encoder, scoped once the text pair ships (M11 closed 2026-09-03; the nano half is M14). Do not inherit
the text pair's architecture assumptions.

## Use-case scoping (added 2026-09-02, Dylan)

M16's fit is fixed vocabulary, frozen document collection: query encoder and index both baked at
build time, no re-embedding path needed in the field. Candidates to scope against, ranked by fit:

- **On-device camera/sensor classification against a fixed label or rule set** (Dylan's example:
  a scooter's onboard camera checking "is this rider on a sidewalk" against a small closed set of
  scene descriptions). Vocabulary and collection are fixed by the rule at deploy time; query
  encoder never needs to know anything outside it.
- **Offline field/vehicle manuals** — technician handheld or in-cab device holds one product line's
  manual corpus, no connectivity, index frozen per firmware/hardware revision.
- **Voice assistant intent routing on a fixed skill set** — smart-speaker or appliance firmware
  matching an utterance against a bounded set of supported commands, re-flashed (not re-indexed) on
  update.
- **Regulatory/compliance lookup on embedded devices** — a fixed rule corpus (safety codes, spec
  sheets) baked into hardware with a long refresh cycle (medical devices, industrial controllers).

Each needs: the fixed vocabulary/collection size that's realistic for the use case, and why
near-zero query compute matters there (battery, silicon cost, certification cycle) rather than just
"it's on the edge." The camera/sensor case is the closest fit to M16's vision premise; the others
are text-shaped and may belong back with the text pair instead.

## Vertical-specialized zero and nano (added 2026-09-11, Dylan)

Explore versions of zero and nano trained for a single vertical or industry, for example legal,
medical, or e-commerce search, instead of the general-purpose recipe. Scope against:

- Whether a vertical vocabulary and document mix change zero's table coverage or nano's
  distillation targets enough to beat the general recipe on that vertical's own queries.
- Whether one narrow vertical still fits the 35M student cap and the frozen stella index, or
  needs its own document tower.
- Whether a customer-facing self-training tool, where a customer trains their own zero table or
  nano student on their own data, is worth building. Scope the training data licence question
  first: a customer-supplied corpus is the customer's own, so the commercial-derived-weights rule
  in `CLAUDE.md` may not apply the same way it does to our released weights, but this needs an
  explicit ruling before any tool work starts.

This is an explore item, not a scheduled build. A vertical or customer-tool experiment still needs
its own scope, budget, and pre-observation protocol, and a change to teacher, cap, or release
policy still needs the applicable owner ruling per `CLAUDE.md`.
