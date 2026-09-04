# M12 — fusion operator audit: does a shipping Qdrant operator match our hand-fitted convex0?

Created 2026-09-04 (Dylan). **Scoped down twice the same day** — Fable broke the first draft, Codex
broke the second and recommended killing the training half. Both reviews and what was cut:
`m12/EXPLORED.md`. Binds from `instructions-m7.md`. Working files `m12/`, branch `m12-work`.

**Local box. One to two days. No training, no new model, no released artifact changes.**

## Why this is the whole milestone

M7's headline `zero`+BM25 **0.4911** uses `convex0 w=0.8`, per-query min-max at depth 1000, over
`bm25s`-lucene — **a formula Qdrant does not implement.** On dev, unweighted RRF k=10 scores **0.5504** against
convex0's **0.5727** (`m7/LEDGER.md:922`) — a 0.022 gap, larger than every table-side lever M7 and
M8 ever measured. If our published pair is fused by a formula the product cannot run, the headline
overstates what a user gets.

**But that 0.022 is not established as an operator gap — the comparison was unfair to RRF.**
`convex0` got a **dev-fitted weight** (0.8, chosen from 8 candidates). RRF got **none**: the M7 grid
swept only `k` ∈ {10,20,30,60,100} (`m7src/fusion.py:209`) and passed no weights, while
`fusion.rrf(runs, k=60, weights=None)` (`:29`) has supported them all along and `apply()` (`:259`)
never plumbs them.

**This is already a public claim.** `constella-zero`'s card says the fused row *"is not reciprocal
rank fusion, so Qdrant's `Fusion.RRF` will not reproduce it"* (`m11/release/MODEL_CARD.md:158-159`,
live). That sentence is literally true and a reader takes it as **"you cannot get 0.4911 in
Qdrant"** — on evidence that never tried a weight. Correcting or confirming it is the point of M12.

**Qdrant ships weighted RRF** — `weights` since **v1.17.0**, `k` since v1.16.0, e.g.
`{"rrf": {"k": 60, "weights": [3.0, 1.0]}}` — plus parameter-free DBSF. So the live question is:
**does a fairly-fitted shipping operator match our hand-rolled one?** If it does, the published
number survives contact with the product and we gain a recommended configuration. If it does not,
we owe users an honest correction. Either way it costs a day and needs no training.

## The fusion-aware training question is NOT in M12

It moved to `instructions-m16.md` with the full list of what it would need. Killed here because:

- **The trainer cannot express it.** Training samples one shared bank of 32,768 negatives that every
  query scores (`m7src/train.py:679-688`); there are no per-query candidate lists. Both miners return
  **IDs only** — `mine_bm25_negatives` discards BM25 scores (`train.py:258`) and mines **within each
  query's own store, not the global pool** (`train.py:217-220`), so a union with globally-mined dense
  candidates would combine retrievers over different corpora, unlike deployed fusion. 1–3 engineering
  days before a single arm runs.
- **The bar rejects the plausible effect.** Fused seed SD is 0.000332 (`m8_noise_floor_fused.json`),
  so a 3-vs-3 mean difference has SE ≈ 0.000271. At a true effect of 0.005 — the ceiling of every
  table-side lever this project has measured — P(observed ≥ 0.008) is **effectively zero**. The
  experiment was designed to return null.
- **The null would not have meant what the draft claimed.** It would show that one miner, surrogate
  and dose missed on a heavily reused dev surface — not that the architecture lacks non-lexical
  capacity. M8 explicitly left hard-candidate listwise training open (`m8/FINDINGS.md:51`).

## The audit

**Two operators, both to Qdrant parity, not to a paraphrase.**

**(a) Weighted RRF — the fair comparison M7 never ran.** Sweep `k` × `weights` with a fitting budget
matched to convex0's (8 candidates), using the `weights` argument already in `fusion.rrf`; plumb it
through `select_on_dev` and `apply`. Qdrant's form is per-prefetch weights on the reciprocal-rank
contributions. This is the cheapest and most likely explanation of the 0.022.

**(b) DBSF — parameter-free, untested here.** Per prefetch, using the **sample** SD:
`ŝ = (s − (μ − 3σ)) / (6σ)`, **no clipping**, and **0.5 for singleton or constant lists**; statistics
are computed over the returned list at the **pinned depth 1000**; then sum the normalised channels.
`FAMILIES` in `m7src/fusion.py:19` is closed and both selection and application assume a parameterised
family (`:208`, `:252`) — DBSF is parameter-free, so plumbing plus degenerate-case tests is ~50–100
lines, not 30. Parity tests against the documented behaviour are part of the deliverable.

**Cost reality check before launching** (`work/fusionruns` holds **only** the four BM25 arrays; dense
runs are not cached): `select_fusion` re-encodes dev queries and re-runs exact top-1000 retrieval on
**CUDA** (`select_fusion.py:50,73`). Documents are not re-encoded. This is a GPU run, not a CPU one.

**Numeric decision rule, registered before scoring.** Dev macro over the four text-backed components,
against convex0 **0.5727**, with the 0.004 frozen-operator fused floor (`m8_noise_floor_fused.json`).
That floor was measured with a **fixed** operator, so it is exact for DBSF and an **under-estimate**
for weighted RRF, which fits 8 candidates as convex0 did — say so when reporting, and do not treat a
weighted-RRF margin inside 0.004 as a win:

| outcome | rule | what we do |
|---|---|---|
| **a shipping operator matches** | best of (weighted RRF, DBSF) ≥ 0.5687 (convex0 − 0.004) | recommend that exact configuration; the published number survives in the product, no retraining |
| **none matches** | best ≤ 0.5687 | state the real cost of shipping-operator fusion plainly, in M14's paper and the card caveat |

Report all three side by side. **Fitting budgets must be stated**: convex0's `w` came from 8
candidates, weighted RRF gets the same 8, DBSF fits nothing — so a DBSF tie is the strongest of the
three results and an unweighted-RRF number is not a fair comparator for anything.

## Protocol — four things that make this wrong if skipped

1. **This does not replace the published 0.4911.** M7 froze the fusion family and parameter, and no
   neighbouring choice may be preferred after the six were seen (`m7/LEDGER.md:691`). A DBSF number
   is post-M7, development-informed, **descriptive** evidence about a successor system. The table
   artifact is unchanged; the *reported system* would not be. Version the operator spec and disclose.
2. **Register the rule above before scoring.** Recording which operator won after seeing it is
   outcome logging, not pre-registration.
3. **Weights are fitted on dev and dev only** — a weighted-RRF configuration selected on the six
   would be exactly the post-hoc fusion choice `m7/LEDGER.md:691` forbids.
4. **A `bm25s`-lucene result does not license a claim about `Qdrant/bm25`** (fixed `avg_len`, own
   tokenizer). DBSF's normalisation depends on the lexical implementation. State the gap; do not
   call it weak.

**Not touched:** the six (dev only), the reserved four, LoTTE, comparators, nano, the document tower,
the teacher, cloud compute, any released artifact.

## Deliverables

1. Weighted RRF plumbed through `select_on_dev`/`apply`, and `dbsf` added, both with parity and degenerate-case tests.
2. The dev macro under convex0, unweighted RRF, **weighted RRF** and DBSF — with fitting budgets stated — against the registered rule, in `m12/FINDINGS.md`.
3. The registration itself, written before scoring, in `m12/LEDGER.md`.
4. **The card sentence, resolved either way.** If a shipping operator matches: record the exact
   configuration, correct `m11/STATUS.md:193` and rewrite `MODEL_CARD.md:158-159` to give users the
   Qdrant recipe that reproduces the number (a card edit is Dylan's call — 2026-09-04 precedent).
   If none matches: keep the sentence and add the measured cost. Either way it stops resting on an
   unweighted comparison. Carry the result into M14's paper.
