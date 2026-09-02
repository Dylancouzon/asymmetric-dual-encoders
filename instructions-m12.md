# M12 — image model (noted, not scoped)

See `CLAUDE.md` M12 entry: an IMAGE asymmetric dual encoder, scoped once M11 lands. Do not inherit
the text pair's architecture assumptions.

## Use-case scoping (added 2026-09-02, Dylan)

M12's fit is fixed vocabulary, frozen document collection: query encoder and index both baked at
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
"it's on the edge." The camera/sensor case is the closest fit to M12's vision premise; the others
are text-shaped and may belong back with the text pair instead.
