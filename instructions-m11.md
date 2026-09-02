# M11 — release the pair, port it, and write the whitepaper (tiny mandate)

Created 2026-08-30 as M10; renumbered M11 on 2026-09-01 when M10 became the nano retry. Runs after M10 delivers nano. Everything binds from `instructions-m7.md`
unchanged: decision authority, licensing and decontamination rules, dev-only selection, the
freeze/ledger protocol, and the headless git contract — working files under `m11/`.

M11 ships no new science. It turns two frozen artifacts into a product and a paper.

## Deliverables

1. **Release both models.**
   - **zero** — M7's zero-compute lookup table. Already frozen and verified releasable:
     `m7/FREEZE.json`, artifact sha `a7007b1a…`, licence check passed, gate PASS on G1–G4. Re-verify
     the sha before pushing; those files are gitignored and mutable.
   - **nano** — M10's distilled query tower (M9's candidate missed both bars; see `m9/FINDINGS.md`).
   - **The framing is the PAIR**: two points on a quality-vs-query-cost frontier, not a leaderboard
     claim. **Zero tier wins exist and the model cards must say so** — M7's dense table missed
     `LR-dense-pertask 0.4583` CI-resolved (0.4339, −0.0243) and its fused system ties OpenSearch
     (0.4911 vs 0.4868, CI straddling zero). Honest characterisation plus cost rows is the product.
   - Model cards carry: CC BY-SA attribution (NQ/SQuAD/HotpotQA/FEVER), teacher + revision pin,
     the claims above, and the three cost rows (query asset / document index / hydration).

2. **ONNX port, INCLUDING the document model.** `B6-pre` (§22) exported the document graph at opset
   17, zero custom-domain ops, parity min-cosine 0.99999994 — **but on near-identity weights.** The
   real trained artifacts have never been exported. Export and parity-verify all three: zero's query
   path, nano, and the document tower. §11.4 tolerances: 1e-4 min-cosine, 1e-3 max-abs.

3. **fastembed integration** for both query-side models.

4. **Whitepaper.** Primary source: `m8/FINDINGS.md` (the negative map and the method learnings) plus
   `m8/EXPLORED.md` (closed avenues, each with its reopening condition), M7's `FINAL_MATRIX.md`,
   `results/m7_costs.json`, the edge prototype and the ANN sweeps.
   **The story is the frontier and the negatives**, both of which are unusually well-measured here:
   how much retrieval quality survives as query-side computation approaches zero, and — from M8 —
   which repairs do *not* work and why. The strongest single result to carry: fragmentation
   correlates with the gap (0.050 nDCG per +1.0 subwords/word, t = 4.61) and yet moving fertility by
   0.164–0.176 moved the metric not at all. **A correlated channel is not a lever.**

## Use-case scoping (added 2026-09-02, Dylan)

The pair's edge case is fixed vocabulary, frozen document collection: query encoder and index both
baked at build time, no re-embedding path needed in the field. Whitepaper and model cards should
name concrete targets, not just "edge retrieval" in the abstract. Candidates to scope against,
ranked by fit:

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
"it's on the edge." Fold the strongest 1-2 into the whitepaper as worked examples; the rest stay
here as backlog.

## Standing constraints

- Nothing in `results/perquery.json` may be overwritten — the frozen comparator vectors cannot be
  regenerated on this box.
- The reserved four and their one confirmatory access carry forward unless M10 spends them.
- `freeze.assert_releasable` gates every upload; the MS MARCO research-only variant must never reach
  one.
- Dylan's go is required for any HF push.
