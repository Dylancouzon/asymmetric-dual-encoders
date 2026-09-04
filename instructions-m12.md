# M12 — fusion operator audit: does Qdrant's DBSF match our hand-fitted convex0?

Created 2026-09-04 (Dylan). **Scoped down twice the same day** — Fable broke the first draft, Codex
broke the second and recommended killing the training half. Both reviews and what was cut:
`m12/EXPLORED.md`. Binds from `instructions-m7.md`. Working files `m12/`, branch `m12-work`.

**Local box. One to two days. No training, no new model, no released artifact changes.**

## Why this is the whole milestone

M7's headline `zero`+BM25 **0.4911** uses `convex0 w=0.8`, per-query min-max at depth 1000, over
`bm25s`-lucene. **Qdrant ships RRF and DBSF, not that.** On dev, RRF k=10 scores **0.5504** against
convex0's **0.5727** (`m7/LEDGER.md:922`) — a 0.022 gap, larger than every table-side lever M7 and
M8 ever measured. If our published pair is fused by a formula the product cannot run, the headline
overstates what a user gets.

**DBSF is the untested third option**, is score-based, and is what Qdrant recommends beside RRF.
Measuring it costs a day and changes what we tell users. That is a better day than any table
retraining currently justifies.

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

**Implement DBSF to Qdrant parity**, not to a paraphrase. Per prefetch, using the **sample** SD:
`ŝ = (s − (μ − 3σ)) / (6σ)`, **no clipping**, and **0.5 for singleton or constant lists**; statistics
are computed over the returned list at the **pinned depth 1000**; then sum the normalised channels.
`FAMILIES` in `m7src/fusion.py:19` is closed and both selection and application assume a parameterised
family (`:208`, `:252`) — DBSF is parameter-free, so plumbing plus degenerate-case tests is ~50–100
lines, not 30. Parity tests against the documented behaviour are part of the deliverable.

**Cost reality check before launching** (`work/fusionruns` holds **only** the four BM25 arrays; dense
runs are not cached): `select_fusion` re-encodes dev queries and re-runs exact top-1000 retrieval on
**CUDA** (`select_fusion.py:50,73`). Documents are not re-encoded. This is a GPU run, not a CPU one.

**Numeric decision rule, registered before scoring.** Dev macro over the four text-backed components,
against convex0 **0.5727** and RRF **0.5504**, with the 0.004 frozen-operator fused floor
(`m8_noise_floor_fused.json` — valid here precisely because Gate A fits nothing):

| outcome | rule | what we do |
|---|---|---|
| **DBSF viable** | ≥ 0.5687 (convex0 − 0.004) | recommend DBSF; the operator gap closes with no retraining |
| **DBSF no better than RRF** | ≤ 0.5544 (RRF + 0.004) | the shipping operators cost ~0.02 and we say so plainly |
| **intermediate** | between | report the number; recommend nothing |

Note in the report that convex0's `w=0.8` was **dev-fitted** while DBSF fits nothing, so a tie
favours DBSF on honesty grounds.

## Protocol — three things that make this wrong if skipped

1. **This does not replace the published 0.4911.** M7 froze the fusion family and parameter, and no
   neighbouring choice may be preferred after the six were seen (`m7/LEDGER.md:691`). A DBSF number
   is post-M7, development-informed, **descriptive** evidence about a successor system. The table
   artifact is unchanged; the *reported system* would not be. Version the operator spec and disclose.
2. **Register the rule above before scoring.** Recording which operator won after seeing it is
   outcome logging, not pre-registration.
3. **A `bm25s`-lucene result does not license a claim about `Qdrant/bm25`** (fixed `avg_len`, own
   tokenizer). DBSF's normalisation depends on the lexical implementation. State the gap; do not
   call it weak.

**Not touched:** the six (dev only), the reserved four, LoTTE, comparators, nano, the document tower,
the teacher, cloud compute, any released artifact.

## Deliverables

1. `dbsf` in `m7src/fusion.py` with parity and degenerate-case tests.
2. The dev macro under all three operators, against the registered rule, in `m12/FINDINGS.md`.
3. The registration itself, written before scoring, in `m12/LEDGER.md`.
4. If DBSF is viable: a one-line correction to `m11/STATUS.md`'s standing operator caveat, and the
   recommendation carried into M14's paper.
