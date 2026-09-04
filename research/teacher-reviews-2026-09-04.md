# Adversarial reviews, 2026-09-04 — Cloud-Inference teacher swap and the hosting ask

Two reviews under the CLAUDE.md standing grant, briefed to break the case, both with the
READ-EXCLUSION. Log audited for reserved-set reads before reading findings: **clean** — only
`m10/PLANNING.md`, `results/m7_calibration.json`, `work/dev/heldout-train.json` (all dev/legal)
were opened. Actions taken are marked inline.

## What these reviews CORRECTED (keep — a future session must not re-derive them)

| Claim made | Correction | Status |
|---|---|---|
| "bge-base discloses none of the dev-suite datasets" | **FALSE — bge-base discloses NQ**, same as stella. NQ is shared exposure. bge also discloses AmazonReviewsClassification, so it has Amazon-domain exposure and still loses on ESCI | corrected |
| "every component of the pinned dev suite carries stella exposure" | `devsuite.py` has four text-backed components; the binding suite has six with the held-out slices. Not established | corrected |
| bge-base and stella both had interior λ optima (LEDGER.md:240) | **FALSE at the time** — bge's curve rose to 0.01, its last point. Grid completed 2026-09-04: λ=0.1 → 0.2848, optimum interior, −0.0365 unchanged | closed, `d17c2f7` |
| a hybrid objective is "not absorbable" like centering/whitening | Category error: absorbability is a property of a query-side *transform*, not of a loss. BM25's query half (IDF, stopword zeroing) IS a per-token scalar and is already learned; its doc half is not representable in stella's 1024-d vector. Premise also false — BM25 here runs on stemmed whole words, not WordPiece | corrected |
| fusion gain "concentrates where dense and BM25 disagree" | Backwards: it concentrates where BM25 is **competitive**. nq is maximal disagreement (0.80 vs 0.58), gain +0.0009; hotpotqa near-parity (0.61 vs 0.585), gain +0.0865 | corrected |
| a bge swap costs "0.03–0.04" on the six | Used the refuted MTEB→six tower projection. Honest: dev table −0.037 CI-resolved; six-set cost **unmeasured**, plausibly 0.045–0.07 by two unvalidated transfers | corrected, in RESULTS.md |
| "decomposability" explains stella's advantage; ratio = fraction of teacher retained | Not identified. The "ceiling" is not a ceiling — the repo records tables beating their teacher on a held-out slice. Say: **stella produced the best table under this probe; the mechanism is unknown** | language fix |
| Qdrant Cloud Inference "must" host stella | **FALSE.** Bring-your-own-vectors and local inference work, and Qdrant's own docs show a stella integration via self-hosted Superlinked. Defensible form: there is no **first-party managed raw-text path** for that vector contract | ask narrowed |
| "FastEmbed-registered" | Premature if it implies upstream — the branch is deliberately unmergeable; the clean PR is M12 | corrected |
| Qwen3-0.6B table 155 MB vs "LightRetriever's 466 MB" | LR's **int8** table is 233.6 MB; 466 MB is its fp16. Qwen3 is 1.5× smaller than LR, not 3× | corrected |

## Open, NOT actioned (ranked by what they would cost to close)

1. **The teacher comparison is a screening proxy, not an equal-treatment trained comparison.**
   bge never received stella's recipe, pseudo-query chain, pooling rule, arm search or fusion
   retuning, so −0.0365 is not the demonstrated cost of the hosted alternative. Closing it
   costs clean refits, matched trained arms and downstream reselection (LEDGER: 8–12 h just to
   re-encode, plus re-adjudicating levers, fusion, gate and freeze). **This is the one hole
   that survives everything else done on 2026-09-04.**
2. **A 768-d stella table was never ranked.** Stella publishes MRL heads at 512/768/1024; the
   repo's own within-family finding is that lower dim distils better. Would also cut index and
   serving cost ~25%. Obvious missing candidate; creates a new, incompatible index contract.
3. **Serving economics of the document tower are unmeasured and look bad for a hosting ask:**
   the released graph is fp32-only, 1.75 GB, and the fp16 CUDA build failed 255/259 passages
   (`m11/STATUS.md:122,137`). No throughput, cost/token, cold-start, residency or SLA numbers exist.
4. **The stale fit list's rank-neutrality is an assertion, not a result** — identical
   contaminated rows do not imply identical effects across teachers (`m7_trainq_manifest.json`).
5. **Fusion has no shipping operator.** Released fusion is convex0 w=0.8 per-query min-max over
   bm25s-lucene; Qdrant ships RRF/DBSF and FastEmbed's BM25 uses a fixed avg_len. Best dev RRF
   0.5504 vs convex0 0.5727 — a 0.022 gap, larger than every table-side lever ever measured.
   **Any hybrid-aware training must first choose the operator the product can actually run.**
6. **Hybrid retraining of zero is a NEW MILESTONE, not an M10 amendment** — the six-set
   fused-vs-dense rows were observed 2026-08-28, so a fusion-trained table is designed with
   knowledge of them. Confirmatory surface would be the reserved four or LoTTE. **That is now M12** (`instructions-m12.md`), created 2026-09-04.
7. **No hybrid comparator rows exist** (only `lightretriever-…-hybrid` 0.4720). And do NOT say
   zero confirmatorily beat BM25: C2 was +0.0165 [0.0017, 0.0311], Holm p=0.0149 > 0.0083, **not rejected**.

*Codex's verbatim text below predates the 2026-09-04 renumbering and is left unedited; its `M12` means today's M13.*

## Codex verdict on the CTO ask

## Product split and verdict

The A/B split is directionally correct, but the ask is **not defensible as stated**.

- A is verifiable only in this narrower form: a public, MIT-compatible query artifact targets one exact pinned 1024d Stella document-vector contract, and Qdrant lacks a first-party managed raw-text ingestion path for that contract.
- A is not a necessity claim. Bring-your-own vectors, local inference, and self-hosted Superlinked already work.
- “FastEmbed-registered” is premature if it means upstream. The implementation exists only on a deliberately unmergeable personal branch; the clean upstream PR is deferred to M12 ([m11/STATUS.md:160](/home/dylan/asymetric-dual-encoders/m11/STATUS.md:160), [instructions-m12.md:31](/home/dylan/asymetric-dual-encoders/instructions-m12.md:31)).
- B supports “Stella leads our screening probe,” not “Stella is proven best” or “retraining against BGE costs 0.0365.”

The CTO’s unanswered questions will be: customer demand, why self-hosted Stella/bring-your-own vectors is insufficient, exact model revision and document preprocessing contract, fp32 serving economics, latency/throughput/SLA, 768d alternative, security review, upstream FastEmbed ownership, and the measured quality loss of the cheapest hosted counterfactual.

## Strongest defensible ask

> Approve a Cloud Inference onboarding pilot for the exact pinned 1024d Stella document encoder behind `constella-zero`. The public MIT artifact already provides a verified zero-transformer query path, but customers currently need to precompute vectors or operate a separate Stella service; first-party document inference would turn it into a native raw-text-to-search Qdrant workflow. This request does not depend on claiming Stella is universally best: Stella leads our current closed-form screen by 0.0365 over BGE-base and 0.0930 over hosted mxbai, with disclosed benchmark-exposure and matched-training caveats. Gate production support on security, throughput, cost, exact-vector parity, and the pending clean SQuAD/ESCI robustness check.

**Note added after the review: that SQuAD/ESCI check was RUN the same day and PASSED** — stella
first on both exposure-free strata, macro +0.1018 [0.0930, 0.1107] (`f515360`,
`results/m7_offfamily_report.json`). Codex's own read stands: a PASS materially weakens the
StackExchange explanation but does not make the model-choice argument bulletproof, because open
item 1 above is untouched by it.
