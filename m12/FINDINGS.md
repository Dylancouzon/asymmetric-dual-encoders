# M12 — fusion operator audit: findings

Registered before scoring in `m12/LEDGER.md`. Results `m12/tier1.json`, `m12/tier2.json`.
Tiers 1-2 are dev only, four text-backed components, depth 1000 unless stated. The six WERE
subsequently re-scored under DBSF (§"re-measured"), under a registered amendment and Dylan's
product ruling — see the protocol note there before citing any six-set number here.

## Result: NO MATCH. No shipping Qdrant operator reproduces `convex0 w=0.8` at depth 1000.

| operator | candidates | dev macro | Δ vs convex0 | 95% CI |
|---|---|---|---|---|
| **convex0 w=0.8** (published) | 8 of a 21-point grid | **0.5727** | — | — |
| weighted RRF `k_q=2 w=(2,1)` | 24, split-half | 0.5601 held-out | −0.0128 vs the held-out comparator 0.5729 | −0.0129 [−0.0160, −0.0098] |
| DBSF | 0 | 0.5580 | −0.0146 | [−0.0178, −0.0114] |
| RRF over `k` (`k_q=3`) | 10 | 0.5535 | −0.0192 | [−0.0224, −0.0159] |
| M7's RRF (`k_ours=10`) | 5 | 0.5504 | −0.0223 | [−0.0253, −0.0192] |

Bar `B = C − 0.004 = 0.5687` for the full-dev rows; the weighted-RRF row is judged against its own
held-out bar **0.5689** (held-out comparator 0.5729), not the full-dev one. Every row misses.
**Registration deviation, disclosed:** the LEDGER's Statistics section required a paired bootstrap on
each operator; Tier 2's first pass omitted it and the CI was added afterwards (Codex close-out
review). It excludes both 0 and the bar, and the CI is post-selection over 24 candidates and
therefore optimistic. Verdict unchanged. Reproduction gate passed at **1.11e-16** over
M7's 21 points, so the comparator is exact; `k_q = k_ours + 1` was confirmed on real data
(`k_q=11/21/31/61/101` reproduce M7's `10/20/30/60/100` to the digit).

## The card's claim was right; the evidence behind it was not

`constella-zero`'s card said `Fusion.RRF` "will not reproduce" the fused row. True — but M7 had
compared a **dev-fitted** convex weight against an **unfitted** RRF whose `k` grid was monotone to
its own lower boundary and never reached Qdrant's default. Both defects are now repaired:

- fair `k` (10 candidates, Qdrant units): **+0.0031**
- fair weights (24 candidates, split-half): **+0.0066** on top

Together +0.0097 of the original 0.0223. **The remaining ~0.0128 is a real operator gap**, ~3.2x
the 0.004 noise bar. Two pre-registered expectations were wrong in magnitude: the missing `k`
range was argued to be "the likelier half" (it is a seventh), and weights were argued to be nearly
inert for a rank-only operator (they are the single largest correction).

**Mechanism.** convex0 (0.5727) beats *both* endpoints — dense-only is 0.5370 — so it exploits
score *magnitudes*. Every rank-based operator discards exactly that; weights re-balance two rank
lists but cannot recover calibration. This also predicts the observed dilution: in Qdrant's
`1/(rank/w + k − 1)` the weight divides the rank, so weights matter at `k_q=2` (+0.0077) and are
inert by `k_q=6` (+0.0001).

## The finding that matters: the ranking INVERTS at realistic prefetch depth

| prefetch limit | convex0 | DBSF | RRF `k_q=3` | Δ (DBSF − convex0) |
|---|---|---|---|---|
| **10** | 0.5482 | **0.5517** | 0.5492 | **+0.0035** |
| **50** | **0.5578** | 0.5558 | 0.5533 | −0.0020 (inside the bar) |
| 100 | **0.5637** | 0.5574 | 0.5534 | −0.0063 |
| 1000 | **0.5727** | 0.5580 | 0.5535 | −0.0146 |

convex0's advantage is a **deep-prefetch** phenomenon. A Qdrant user at `limit: 10–50` — the
common configuration — gives up nothing by using DBSF, which fits zero parameters.

**Not an artifact.** DBSF's degenerate `→0.5` branch (singleton or zero-variance prefetch) fires
**0.00%** at every depth, median list length exactly the depth, all macros reproducing
(`logs/m12_depth_check.log`). The registered comparison is depth 1000, so this curve is
descriptive by registration and does **not** overturn the NO MATCH verdict.

## The published fused row, re-measured under a Qdrant operator (Dylan's ruling)

*"convex needs to go if it can't be done in Qdrant. 1000 prefetch to return 10 items is a lot too
and not something very realistic."* Registered before scoring (LEDGER Amendment): DBSF at prefetch
**100**, depth set by realism, not by a dev argmax. `m12/six_dbsf.json`.

> **PROTOCOL NOTE — this is a post-M7 PRODUCT-POLICY OVERRIDE by the owner, not an M7-compliant
> result.** `m7/LEDGER.md:691` froze `w=0.8` as the released parameter *"whatever the six say"*. M12
> changed the public recommendation after the six were already seen. Registering before the DBSF run
> prevents selecting **on that run's outcome**; it does **not** restore pre-six blindness, and a
> different operator family evades the "neighbouring weight" wording but not the release freeze
> itself. So this is recorded as what it is: **Dylan overriding the M7 release freeze on
> implementability grounds** (the frozen operator cannot run in the product), exercising
> `CLAUDE.md` §"Past decisions are revisitable". convex0 remains **the M7 operator of record** and
> C1/C2/C3 keep their convex0 basis. The DBSF six-set numbers are **prospective deployment
> evidence**, and may not be cited as an M7-protocol performance conclusion.
> Flagged by the Codex close-out review; the earlier framing claimed protocol compliance and was wrong.

| system | all 6 | clean 4 | contamination cost |
|---|---|---|---|
| **DBSF @ prefetch 100** (stock Qdrant, **0 fitted parameters**) | 0.4887 | **0.4912** | −0.0025 |
| convex0 @ 1000 (published, dev-fitted `w=0.8`, not in Qdrant) | 0.4911 | 0.4866 | +0.0045 |
| dense only | 0.4339 | 0.4098 | +0.0241 |
| BM25 | 0.4174 | 0.4409 | −0.0235 |

Per dataset, DBSF@100: arguana 0.5800, fiqa 0.3872, nfcorpus 0.3442, scidocs 0.1850,
scifact 0.7173, trec-covid 0.7184. **Reproduction of the published row: 0.4911 vs 0.4911**
(delta −1.1e-06), so the loading path and encode caches are intact.

**The registered depth cost us.** (Registration is established by the pushed timestamp, not by the
result being worse — but the result does defeat the cherry-picking reading.) On the six DBSF is
monotone increasing in depth — all-6 0.4660 / 0.4849 / 0.4887 / 0.4898 and clean-4 0.4625 / 0.4856 /
0.4912 / **0.4974** at depths 10 / 50 / 100 / 1000. **DBSF@1000 on clean-4 is the best number in the
table**, 0.0062 above the registered depth-100 headline. Depth 100 was fixed in advance on realism,
before any six-set number existed, and it is the *worse* of the two — so the choice cannot have been
made to flatter the comparison. We report the better number and do not claim it.

**This is a TIE, not a win.** On clean-4, DBSF@100 scored 0.4912 against convex0@1000's 0.4866.
The observed **+0.0046 is inside the project's ~0.005 lever band** (`instructions-m14.md`), and no
confidence interval was computed for a six-set operator difference — so it is **no material observed
difference**, not evidence of superiority. Likewise the −0.0024 on all six. An earlier draft of this
file said "WINS" and "strictly better"; that broke our own tie rule and is corrected here.

**The case for DBSF is therefore deployability, not quality**: it runs in Qdrant, it needs no
1000-deep prefetch, and it removes a dev-fitted weight from the released system — at no measured
cost in quality.

**A forecast that was wrong.** Before running this, the dev curve was used to predict the six-set
headline would fall to ~0.47-0.48 and that the OpenSearch tie would break. Neither happened. Dev's
four components are not representative of the six here: DBSF is much stronger on trec-covid
(0.7184 vs 0.7018) and scifact (0.7173 vs 0.7068) than dev suggested. **Dev-to-six extrapolation
of an OPERATOR difference is not reliable** — worth remembering before the next one.

**Caveats.** `bm25s`-lucene, not `Qdrant/bm25`.

**Self-hits — a measurement/product mismatch, corrected after review.** Our runs drop the query's own
document at retrieval, *then* truncate, so DBSF@100 here sees **100 non-self candidates**. A plain
Qdrant `limit: 100` returns the self-match and leaves **99** usable. The numbers therefore describe a
Qdrant configuration with a **self-ID exclusion filter in the prefetch**, which is expressible
(`must_not` on the point id) and is how the card now states it. This is NOT neutral between
operators — DBSF recomputes μ and σ over the whole list, so one extra candidate shifts every
normalised score, whereas convex0's max-normalisation usually does not move. **Affected: arguana
(1,298 self-hits / 1,406 queries) and fiqa (55) — therefore the all-6 rows. The clean-4 headline
partition has ZERO collisions on all four sets and is unaffected.**

## Consequences

- Card, `README.md` and `m11/STATUS.md` now lead with the reproducible recipe — `Fusion.DBSF` at
  prefetch 100, with the client snippet — instead of only saying RRF will not work.
- The convex0 row is **retained and labelled** "not runnable in Qdrant", never deleted: it was the
  operator of record at release.
- **C1/C2/C3 are untouched** and keep their registered convex0 basis. The OpenSearch tie claim is
  annotated with which operator it was measured under, not restated for DBSF.
- Benchmark composition is M14's call, with the rule registered in advance
  (`instructions-m14.md`): headline clean-4, all six beside it, no re-picking.
- Carry into M14: the depth-dependence is the reportable result, not the headline gap.

## Limits

- **Tiers 1-2 are dev only.** The six were touched only by the registered DBSF amendment below,
  logged in `m7/SIX_ACCESS.log`.
- **`bm25s`-lucene, not `Qdrant/bm25`** (fixed `avg_len`, own tokenizer). DBSF normalises over the
  returned scores, so a different lexical implementation shifts its inputs. Untested.
- **A CI on a fitted winner is post-selection** and optimistic. DBSF's is not (0 candidates).
- **Cloud finding.** Qdrant Edge is embedded and a user can fuse in their own code;
  `bench/edge_prototype_pair.py` is dense-only, so edge fusion is unbuilt and uncosted.
