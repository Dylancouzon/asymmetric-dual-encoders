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

2. **ONNX port, INCLUDING the document model.** Export and parity-verify all three: zero's query
   path, nano, and the document tower. §11.4 tolerances: 1e-4 min-cosine, 1e-3 max-abs.
   **Status correction (2026-09-03): the document tower is already exported on the REAL weights** —
   `results/m9_doc_export.json`, opset 17, zero custom-domain ops, min-cos 0.99999940, max-abs
   3.3e-07, artifacts in `work/m9onnx/stella-400M-doc/`. What that file does NOT show, despite its
   `fastembed_local` block: that block was measured on `work/m9onnx/nano-minilm-l6`, and no
   `model_tokens.onnx` exists for stella. **The doc tower has never been served through fastembed.**

3. **fastembed integration** for both query-side models.

4. **Whitepaper.** Primary source: `m8/FINDINGS.md` (the negative map and the method learnings) plus
   `m8/EXPLORED.md` (closed avenues, each with its reopening condition), M7's `FINAL_MATRIX.md`,
   `results/m7_costs.json`, the edge prototype and the ANN sweeps.
   **The story is the frontier and the negatives**, both of which are unusually well-measured here:
   how much retrieval quality survives as query-side computation approaches zero, and — from M8 —
   which repairs do *not* work and why. The strongest single result to carry: fragmentation
   correlates with the gap (0.050 nDCG per +1.0 subwords/word, t = 4.61) and yet moving fertility by
   0.164–0.176 moved the metric not at all. **A correlated channel is not a lever.**

## Standing constraints

- Nothing in `results/perquery.json` may be overwritten — the frozen comparator vectors cannot be
  regenerated on this box.
- The reserved four and their one confirmatory access carry forward unless M10 spends them.
- `freeze.assert_releasable` gates every upload; the MS MARCO research-only variant must never reach
  one.
- Dylan's go is required for any HF push.


## Amendment A — the M11a slice (Dylan, 2026-09-03)

M10 is paused on budget. The `zero` half of M11 does not depend on it, so M11a ships zero end to end
and publishes the document tower. Nano's export, any upstream PR and the whitepaper wait for M10.

| # | ruling |
|---|---|
| 1 | `zero-query-encoder-v1` flips **PUBLIC**. Licence was ruled MIT-and-public-valid on 2026-09-03; no further sign-off needed. |
| 2 | The stella ONNX document tower is published as a **new PUBLIC HF repo** on Dylan's account. |
| 3 | fastembed work lands on a branch of the fork **`Dylancouzon/fastembed`** (created 2026-09-03). **No PR is opened this milestone.** |
| 4 | Whitepaper (deliverable 4) **deferred** until M10 resolves — the frontier has one point until nano exists. |

Dylan's go for an HF push is **granted for exactly these two repos** at these visibilities; anything
else still needs asking.

**M11a ships no new science and touches no quality number.** Nothing here reads a dev or reserved
set, so M9's registration machinery (`m9src/guard9.py`) does not apply — its results are engineering
parity artifacts, not decisions. The release gates in `m11/release/push.py` DO apply, unchanged, to
every upload. Plan and gates: `m11/PLANNING.md`.
