# M12 — fusion operator audit: does a shipping Qdrant operator match our hand-fitted convex0?

Created 2026-09-04 (Dylan). **Scoped down twice the same day** — Fable broke the first draft, Codex
broke the second and recommended killing the training half. Both reviews and what was cut:
`m12/EXPLORED.md`. Binds from `instructions-m7.md`. Working files `m12/`, branch `m12-work`.

**Local box. No training, no new model, no released artifact changes. Tier 1 is half a day and
answers the question; Tier 2 runs only if Tier 1 fails.**

## Where this applies: a CLOUD finding

**Qdrant Cloud — fully pertinent.** Server-side fusion (prefetch + `fusion: rrf|dbsf`) is the only
one-round-trip hybrid, and a cloud user genuinely cannot express `convex0`. This is the case the card
sentence misleads and the case the CTO ask covers.

**Edge — largely not, and say so rather than implying otherwise.** Qdrant Edge is embedded, so a user
owns the process and can fuse in their own ~20 lines with any operator. More to the point,
`bench/edge_prototype_pair.py` is **dense-only — no BM25, no fusion, ever**; fusion on edge would
mean shipping a lexical index beside the document index, which has never been built or costed.
**That gap is a NOTE, not a requirement** (Dylan, 2026-09-04): edge numbers depend heavily on hardware
and corpus size, so a complete edge-fusion cost row is not something M12 or M14 owes. State the
limitation plainly and move on.

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

**(a) Weighted RRF — the fair comparison M7 never ran.** Two things were unfitted, not one; the
missing `k` range is the likelier half.

*The operator, corrected 2026-09-04.* `fusion.rrf` computes `Σ wᵢ/(k+rankᵢ)`, rank 1-based. Qdrant
computes `Σ 1/((posᵢ+1)/wᵢ + k − 1)`, pos 0-based (`qdrant-client/qdrant_client/hybrid/fusion.py`;
docs agree). **These are different functions, so the existing `weights` argument is NOT parity.**
Server `lib/segment/src/common/reciprocal_rank_fusion.rs` and the client agree; the server is
authoritative. Algebraically it is `Σ wᵢ/(rankᵢ + wᵢ(k_q−1))` — our `rrf` family with a **per-list**
`kᵢ`, so `rrf_qdrant()` is ~5 lines. Consequences:
- **`k_qdrant = k_ours + 1`.** Report every recommended `k` in Qdrant's units.
- **Not scale-invariant** for `k_q > 1`, so `(1,2)` ≠ `(2,4)` and the grid is 3-D (`w₁, w₂, k`).
  **Except at `k_q = 1`**, where weights degenerate to pure multipliers and only the ratio matters.
- `w ≤ 0 → 0.0`; **`k_q = 0` divides by zero at pos 0 and the server does not validate it**
  (`validate.rs` returns `Ok(())`) — excluded from the grid, never recommended.

*The unbracketed `k`.* The M7 sweep is strictly monotone decreasing in `k` — 10:**0.5504**, 20:0.5453,
30:0.5405, 60:0.5352, 100:0.5324 (`m7/LEDGER.md:922`) — so its argmax sits on the **lower boundary**
and the optimum was never bracketed. **Qdrant's default `k=2` is `k_ours=1`, outside the M7 grid
entirely.**

**Grids, pinned here before the first fusion call** (an unpinned "best few k" has no statable fitting
budget):
- **TIER 1 — RRF-k, unweighted:** `k_q ∈ {1,2,3,4,6,11,21,31,61,101}` — **10 candidates**, comparable
  to convex0's 8.
- **TIER 2 — RRF weighted:** `k_q ∈ {2,6,11,61}` × `(w_dense,w_bm25) ∈ {(1,1),(2,1),(3,1),(4,1),(1,2),(1,3)}`
  — **24 candidates**, judged under split-half (see the rule).

All CPU re-fusion of runs built once — one GPU retrieval pass total, provided the dense runs are
persisted (below).

**Tier 2 runs only if Tier 1 fails, and that is a deliberate de-scope, not laziness.** Tier 1 is
`convex0` reproduced + RRF over `k` + DBSF: the two operators a cloud user actually reaches for, both
unfitted or nearly so, and therefore the two **strongest** evidence classes in the rule below.
Weighted RRF is the weakest row — 24 candidates, split-half machinery, and a weight that has to
transfer. If Tier 1 clears the bar the question is answered and fitting weights would produce a
result we would discount anyway. **Registering the skip is part of the rule: Tier 2 is attempted iff
no Tier 1 row passes.**

**(b) DBSF — parameter-free, untested here. TIER 1.** Per prefetch, using the **sample** SD:
`ŝ = (s − (μ − 3σ)) / (6σ)`, **no clipping**, and **0.5 for singleton or constant lists**; statistics
are computed over the returned list at the **pinned depth 1000**; then sum the normalised channels.
**Register the missing-document rule before scoring**: a doc in one prefetch and not the other
contributes **0**, which under `(s−(μ−3σ))/(6σ)` is *not* the bottom of the normalised range. That is
structurally M7's `floor_zero`/MAJOR 19 trap and must be a written choice, not a fall-out of the loop.
DBSF lives in `m12src`; parity and degenerate-case tests are part of the deliverable.

**(c) Depth — added 2026-09-04.** DBSF's μ and σ are computed over the returned list, so the operator
at depth 1000 is **not** the one a user runs at `prefetch limit: 20`. Re-fuse the cached runs
truncated to `d ∈ {10, 50, 100, 1000}` — for **convex0 and the registered winner of each operator
class only**, not the full weight grid. **Depth 1000 stays the registered comparison**; the rest is a
reported curve. Truncation is sound (top-d of an exact top-1000 is exact top-d; BM25 lists are already
≤1000 after `drop_zero_scores`, matching a Qdrant sparse prefetch at `limit: d`) on two conditions:
truncate on the **same stable sort `rrf()` uses** (`fusion.py:36`), and report the qid∈doc_ids
collision count per component — both runs drop self-hits *after* retrieval, which Qdrant does not, so
a non-zero count means a `d=10` list really holds 9.

**`m7src/fusion.py` is not edited.** `select_on_dev`/`apply_frozen` produced the frozen M7 selection,
and `m7src/test_fusion_paths.py:104` asserts the grid length. M12's operators and selection live in
**`m12src/`**, importing `fusion` read-only; the M7 path stays byte-stable and its tests keep passing.

**Cost reality check before launching** (`work/fusionruns` holds **only** the four BM25 arrays; dense
runs are not cached): `select_fusion` re-encodes dev queries and re-runs exact top-1000 retrieval on
**CUDA** (`select_fusion.py:50,73`). Documents are not re-encoded. This is a GPU run, not a CPU one.

**Numeric decision rule, registered before scoring.** Dev macro over the four text-backed components,
**The comparator is convex0 recomputed on M12's own runs, not the frozen literal.** M7's
0.5726634997854769 came from a different torch/CUDA build; GPU top-k tie order and fp16 accumulation
move the 4th–5th decimal, and every operator delta would then carry run-reproduction noise. So:
recompute all **21 M7 grid points** first and **gate on `max|Δ| ≤ 1e-4` against `m7/FREEZE.json`**
(M7's own reproduction tolerance, `m7/STATUS.md:92-93`) before scoring a single new operator. If the
gate fails, stop and report — that is a finding about reproducibility, not a fusion result. The
recomputed convex0 is the comparator; the frozen literal is reported beside it.

*What the 0.004 is and is not.* `m8_noise_floor_fused.json` sets it as `max(0.0040, 2×floor)` where
the floor is **training-seed** spread with the operator **frozen**. An operator comparison on one
fixed table is deterministic and has no seed noise at all, so 0.004 is borrowed here as "would this
survive a retrained table" — a defensible conservative bar, but say that; it is not the instrument's
precision. It carries no fitting allowance, so it is an **under-estimate** for any fitted operator
and adequate only for DBSF, which fits nothing. **Also report a paired bootstrap over dev queries**
on each operator-vs-convex0 difference — **resampled within component, then macro-of-means**
(pooling would weight hotpotqa's 7,405 queries 8× physics's 1,039 and is not the macro); B and seed
registered; percentile CI. A CI on a *fitted* winner is post-selection and optimistic — say so.

Bar `B = convex0_recomputed − 0.004`. **One rule per operator class**, because they do not share a
fitting budget — a single `max()` over ~34 configurations judged against a bar calibrated for one
would be the contradiction the first draft carried (it both passed a fitted 0.5690 and forbade
counting it):

| tier | operator | candidates | passes if | strength |
|---|---|---|---|---|
| 1 | **DBSF** | 0 | dev macro ≥ B | strongest — nothing fitted |
| 1 | **RRF over `k`** | 10 | dev macro ≥ B | second — one parameter, no weight to transfer |
| 2 | **RRF weighted** | 24 | **held-out half** ≥ B, under a registered split-half of dev qids (fit on `hash(qid)` even, score on odd, and the reverse; report the macro of the two held-out halves) | weakest — the only row whose margin the 0.004 bar under-estimates |

**A shipping operator matches** if any row passes; recommend the highest-strength passing row's exact
configuration **in Qdrant's units**. **Tier 2 is attempted iff neither Tier 1 row passes** — record
the skip in `m12/LEDGER.md` when it happens. **None matches** if no row does; state the real cost of
shipping-operator fusion plainly, in M14's paper and the card caveat.

Report all five rows side by side (convex0, M7's unweighted RRF, RRF-over-`k`, weighted RRF, DBSF).
**Fitting budgets must be stated**: convex0's `w` came from 8 candidates **inside a 21-point,
3-family grid** — give both numbers. The M7 unweighted-RRF row (5 badly-placed `k`) is a fair
comparator for nothing and is reported only as the thing being corrected.

## Protocol — five things that make this wrong if skipped

1. **This does not replace the published 0.4911.** M7 froze the fusion family and parameter, and no
   neighbouring choice may be preferred after the six were seen (`m7/LEDGER.md:691`). A DBSF number
   is post-M7, development-informed, **descriptive** evidence about a successor system. The table
   artifact is unchanged; the *reported system* would not be. Version the operator spec and disclose.
2. **Register the rule above before scoring.** Recording which operator won after seeing it is
   outcome logging, not pre-registration.
3. **Weights are fitted on dev and dev only** — a weighted-RRF configuration selected on the six
   would be exactly the post-hoc fusion choice `m7/LEDGER.md:691` forbids.
   Transfer is this milestone's weakest point and is handled by the **split-half in the rule above**
   — free, same runs, no new corpus. **NanoMSMARCO was considered and CUT 2026-09-04**: it has **50
   queries**, so per-query nDCG@10 SD ≈ 0.3–0.4 gives SE ≈ 0.05 — roughly 12× too coarse to read a
   0.004 effect — and it would add a loader, a stella encode and a BM25 index for a number that
   cannot answer the question. (Dylan's validation-not-training rule still permits it; the objection
   is power, not licence. A usable MS MARCO transfer set means building dev the way `nq-250k` was
   built — 6,980 queries, 250K distractors, a real stella encode — which is a separate budget line.)
4. **M12 measures DEV; 0.4911 is a SIX-set number, and M12 may not claim to reproduce it.**
   Added 2026-09-04. Re-running the six under a new operator IS the post-hoc fusion selection
   `m7/LEDGER.md:691` forbids, so no M12 result can license "this Qdrant recipe gives you 0.4911".
   The strongest sentence available is dev-scoped: *"on our development set, configuration X matches
   the convex operator the 0.4911 row was measured under."* Any card wording above that ceiling
   violates Protocol 1.
5. **A `bm25s`-lucene result does not license a claim about `Qdrant/bm25`** (fixed `avg_len`, own
   tokenizer). DBSF's normalisation depends on the lexical implementation. State the gap; do not
   call it weak.

**Not touched:** the six (dev only), the reserved four, LoTTE, comparators, nano, the document tower,
the teacher, cloud compute, any released artifact.

## Execution facts a fresh session must not re-derive

- **Artifact:** `work/runs/p35w-2m-s2500.release.npz`, **int8** variant, loaded exactly as
  `select_fusion.main:61-73` with `freeze.assert_encoder_matches_artifact`; the sha must match
  `m7/FREEZE.json`.
- **Persist the dense runs** to `work/m12/dense-<comp>-d1000.npz`, keyed on the table sha.
  `select_fusion.dense_run` returns a dict and writes nothing — without this, "one GPU pass" is false
  the moment the depth curve or a bootstrap is re-run.
- **`qdrant-client` is not installed** and the venv has no pip. **Vendor** the two reference functions
  (~40 lines) into the parity test, citing the upstream commit; pin the Rust file hash in
  `m12/LEDGER.md`. Do not add a dependency for this.
- `m12src` needs a `sys.path` entry for `m7src` to import `fusion`.

## Deliverables

1. `m12src/` with `rrf_qdrant()` and `dbsf()`, plus parity tests against the vendored reference and
   **three** degenerate-case tests that can actually fire: constant/singleton list → 0.5, a
   **missing doc** contributing 0, and a **negative normalised DBSF score** (a doc below `μ−3σ` must
   rank *below* an absent doc). `m7src/fusion.py` unchanged; `m7src/test_fusion_paths.py` still passes.
2. The 21-point M7 reproduction gate, then **Tier 1** — convex0, M7's unweighted RRF, RRF over `k`,
   DBSF — with fitting budgets stated, `k` in Qdrant's units, within-component paired-bootstrap CIs
   and the depth curve for the winners, in `m12/FINDINGS.md`. **Tier 2 (weighted RRF, split-half)
   only if Tier 1 fails.**
3. The registration itself, written **and pushed** before scoring, in `m12/LEDGER.md` — the remote
   timestamp is the external witness (M7 convention, `instructions-m7.md:67`).
4. **The claim, resolved in all three places it is live** — `m11/release/MODEL_CARD.md:158-159`,
   `m11/STATUS.md:193` **and `README.md:99`** (missed by the original draft). If a shipping operator
   matches on dev: give the exact Qdrant configuration, **dev-scoped per Protocol 4**, never as a
   route to 0.4911. If none matches: keep the sentence and add the measured cost. **Either way fix a
   second imprecision**: `MODEL_CARD.md:158` says "min-max normalized", but `convex0` is
   `floor_zero=True` → `s / max(s)` per query per channel, absent doc = 0. Describe the real operator.
   A card edit is Dylan's call (2026-09-04 precedent). **Scope every rewritten sentence to the cloud
   case** and note that edge fusion is unmeasured — as a limitation line, not a promised metric.
   Carry both into M14's paper.

---

**Amendment 2026-09-04 (pre-execution review + Fable pass).** Verified against the Qdrant **server**
Rust (`lib/segment/src/common/reciprocal_rank_fusion.rs`, authoritative), the client, the docs and the
repo. **Confirmed as written:** `k` since v1.16.0, `weights` since v1.17.0, DBSF since v1.11.0 and
parameter-free at the API, DBSF **sample** SD (ddof=1), no clamp, 0.5 for singleton/constant, every
`m7src` line reference, the 0.5504/0.022 figures, and the "this is a GPU run" cost note. Client and
server agree on everything M12 needs.

**Corrected:** the weighted-RRF operator and its algebraic identity `Σ wᵢ/(rankᵢ+wᵢ(k_q−1))`, the
`k_q=1` scale-invariance exception and the unvalidated `k_q=0` division by zero (§a); the unbracketed
`k` sweep and **pinned grids** (§a); DBSF missing-document semantics (§b); depth parity, truncation
conditions and self-hit accounting (§c); no edits to `m7src/fusion.py`; the comparator is **convex0
recomputed behind a 21-point ≤1e-4 reproduction gate**, not the frozen literal (§rule); **one rule per
operator class** with split-half for the fitted one, replacing a `max()` that contradicted its own
"no fitted margin inside 0.004" clause (§rule); within-component bootstrap (§rule); the dev-vs-six
claim ceiling (Protocol 4); `README.md:99` as a third live copy (Deliverable 4); execution facts a
fresh session would otherwise re-derive.

**Cut / de-scoped 2026-09-04 (Dylan: "don't over-engineer"):** the NanoMSMARCO transfer check (50
queries, ~12× underpowered — Protocol 3); the stale `FAMILIES`-plumbing paragraph; the depth curve
over the full weight grid; six degenerate tests down to three; and **weighted RRF made conditional on
Tier 1 failing**. Edge-fusion cost is a **note, not a requirement** — it depends too much on hardware
and corpus size to owe a number.

**Product note, out of scope:** the server's `ScoreFusion` carries `weights` and a `MinMax`
normalisation — i.e. our `convex` family exists in Qdrant, unexposed at the API. For the CTO ask, not
for M12.
