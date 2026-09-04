# M11 — release the pair, port it, and write the whitepaper (tiny mandate)

> **CLOSED 2026-09-03.** M11 shipped the `zero` half end to end: `constella-zero`, the ONNX
> document tower, the FastEmbed integration and both cards. Everything below that depends on
> **nano** — deliverables 1 (nano), 2 (nano), 3 (nano) and 4 (the whitepaper) — moved to
> **`instructions-m12.md`**, and the image model became `instructions-m13.md`. Read Amendments A
> and B at the bottom for what was actually ruled; the deliverable text above them is the ORIGINAL
> mandate and is superseded wherever the two disagree. Outcome: `m11/STATUS.md`.

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
   `model_tokens.onnx` exists for stella. ~~The doc tower has never been served through fastembed.~~ **Closed by T3/T4**: it is a built-in FastEmbed model and gate 4 measures it against direct ORT.

3. **fastembed integration** for both query-side models. **The bar is a stock loader**: a user
   names the model in fastembed and gets the query path, with no custom encoder file to copy and no
   `pynife`-equivalent package to install first. pyNIFE clears this bar by shipping a plain
   sentence-transformers Router (41 ms load, teacher and student swap at the call site); `zero`
   currently needs `release/zero_encoder.py`, which is 89 lines the user has to obtain and trust.
   Same test on both sides of the pair: register `zero` and `nano` so the document tower and the two
   query paths are selectable by name against one index. The two release traps in `m11/STATUS.md`
   (stella's `config_kwargs`, its padding-to-512 tokenizer) are the ones an integration will hit.

   **MET for the zero half, 2026-09-03.** `TextEmbedding("DylanCouzon/constella-zero")` and
   `TextEmbedding("DylanCouzon/stella-en-400M-v5-doc-onnx")` are stock-loader calls against one
   index: no custom encoder file, no package to install first, two entries in FastEmbed's own
   `supported_onnx_models`. `zero_encoder.py` still ships, but as the reference implementation for
   readers who want it, not as a requirement. Both traps were hit and are recorded in
   `m11/CODEMAP.md`. nano's registration is M12.


4. **Whitepaper.** Primary source: `m8/FINDINGS.md` (the negative map and the method learnings) plus
   `m8/EXPLORED.md` (closed avenues, each with its reopening condition), M7's `FINAL_MATRIX.md`,
   `results/m7_costs.json`, the edge prototype and the ANN sweeps.
   **The story is the frontier and the negatives**, both of which are unusually well-measured here:
   how much retrieval quality survives as query-side computation approaches zero, and — from M8 —
   which repairs do *not* work and why. The strongest single result to carry: fragmentation
   correlates with the gap (0.050 nDCG per +1.0 subwords/word, t = 4.61) and yet moving fertility by
   0.164–0.176 moved the metric not at all. **A correlated channel is not a lever.**

   **Novelty claims are withdrawn — `research/m7-novelty.md` §pyNIFE.** pyNIFE (MIT, PyPI
   2025-11-03) is `zero`'s construction, published before M7 started, and the 2026-08-25/28 sweep
   missed it by searching arXiv and HF rather than PyPI and GitHub. The paper claims what is still
   ours: the measurement standard, the artifact constraints, and the pair on one index. Cite pyNIFE
   as prior art and read its NanoBEIR rows as independent corroboration of M8 — both their models
   land on 59.2 regardless of teacher, which is their ceiling reading of our Spearman 0.000 between
   teacher quality and table quality. Their caveats section states the mechanism our M9 coverage
   failure is a symptom of, and states it better than we currently do: a static query path cannot
   attenuate a token by context, and cannot represent negation, because no token sees another.

## Standing constraints

- Nothing in `results/perquery.json` may be overwritten — the frozen comparator vectors cannot be
  regenerated on this box.
- The reserved four and their one confirmatory access carry forward unless M10 spends them.
- `freeze.assert_releasable` gates every upload; the MS MARCO research-only variant must never reach
  one.
- Dylan's go is required for any HF push.


## Amendment A — the M11a slice (Dylan, 2026-09-03)

M10 is paused on budget. The `zero` half of M11 does not depend on it, so M11a ships zero end to end
and publishes the document tower. Nano's export, the upstream PR and the whitepaper wait for M10 (now M12).

| # | ruling |
|---|---|
| 1 | `zero-query-encoder-v1` flips **PUBLIC**. Licence was ruled MIT-and-public-valid on 2026-09-03; no further sign-off needed. |
| 2 | The stella ONNX document tower is published as a **new PUBLIC HF repo** on Dylan's account. |
| 3 | fastembed work lands on a branch of the fork **`Dylancouzon/fastembed`** (created 2026-09-03). **No PR is opened this milestone.** |
| 4 | Whitepaper (deliverable 4) **deferred** until M10 resolves — the frontier has one point until nano exists. **Reaffirmed 2026-09-03** after both adversarial reviews proposed a cancellation contingency: deferral stands, no dated fallback. Do not re-propose it. |

Dylan's go for an HF push is **granted for exactly these two repos** at these visibilities; anything
else still needs asking.

**M11a ships no new science and touches no quality number.** It reads dev **query texts** and
nq-250k **passage texts** as parity fixtures (`export_onnx.py`, `verify_fastembed.py`,
`export_doc.py`) — no qrels, no scoring, no reserved set — so M9's registration machinery
(`m9src/guard9.py`) does not apply — its results are engineering
parity artifacts, not decisions. The release gates in `m11/release/push.py` DO apply, unchanged, to
every upload. Plan and gates: `m11/PLANNING.md`.

## Amendment B — cards are FastEmbed-first, and carry no competitive claim (Dylan, 2026-09-03)

**Supersedes deliverable 1's "Zero tier wins exist and the model cards must say so."** That rule
required the cards to carry `LR-dense-pertask 0.4583`, the OpenSearch tie and the missed bar. Ruled
out: *"remove the competitive comparison and the missed bar"* — an internal project bar means
nothing to someone downloading the model, and the comparator table belongs in the whitepaper, which
still carries all of it. Cards KEEP the measured nDCG@10 numbers and the stella contamination
disclosure; those are what a reader needs to interpret the numbers at all.

**Cards assume built-in FastEmbed support** (Dylan: *"the card should assume the model is in
Fastembed, won't be released until then. You can point the card to our branch for now"*). So both
cards use `TextEmbedding("<name>")` with the install pointed at
`Dylancouzon/fastembed@add-constella-models`, and **the release is not considered done until the
model ships in FastEmbed**. The card gate runs against that checkout (`FASTEMBED_FORK`).

**Qdrant examples use `COSINE`, not `DOT`** (Dylan, 2026-09-03): Qdrant implements cosine as a dot
product — normalize once on upsert, dot at query time — so `COSINE` is the same cost and does not
depend on the caller preserving unit norm.
