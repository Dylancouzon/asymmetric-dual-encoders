The current draft is not defensible as a mandate. The main failure is not the asymmetric serving idea; it is treating highly non-transferable retention numbers as a forecast while simultaneously underfunding, changing the prompt protocol, contaminating the fresh dev surface, and leaving the released artifact undefined.

1. **[BLOCKER] The draft contradicts the existing M9 mandate.**  
   The repository currently requires an artifact-level teacher screen, prohibits new data/dev/doc vectors, and specifies embedding alignment plus ranking preservation ([instructions-m9.md](/home/dylan/asymetric-dual-encoders/instructions-m9.md:18)). The draft instead fixes stella without screening, adds FineWeb and LoTTE, changes the dev suite, and makes ranking preservation contingent.  
   **Change:** Before any run, write an explicit supersession section enumerating each overridden clause and owner approval. Otherwise a successful result can be rejected as protocol-nonconforming.

2. **[BLOCKER] A — the ceiling arithmetic establishes oracle headroom, not probable retention.**  
   LEAF did use exactly the relevant serving topology—small student queries against frozen teacher documents—so there is no hidden asymmetry gap. The transfer gap is experimental: different teacher geometry, 768d versus 1024d projection, prompt handling, initialization, sequence length, data distribution, and vastly different training dose. The cited retention results are not a coherent confidence band:

   - LEAF is the closest analogue.
   - EmbedDistill used supervised retrieval structure and synthetic query generation.
   - Wang–Lyu used over eight million MS MARCO queries and teacher-related initialization; its four-layer result was 96.2%, while the 2-layer result was 92.5% ([paper](https://aclanthology.org/2023.sustainlp-1.23.pdf)).
   - ScalingNote used labeled industrial pairs and a co-trained teacher retrieval system, not generic text regression ([paper](https://arxiv.org/abs/2411.15766)).

   **Change:** Remove the “projection 0.551–0.566” from decision logic. State only: “the teacher ceiling makes success possible at ≥89.7% avg-6 / ≥92.8% no-disclosed-overlap-4 retention.” Estimate achievable retention from M9 pilot curves, not cross-paper ratios.

3. **[MAJOR] The proposed 15% kill-switch is statistically fictitious.**  
   Transformer distillation learning curves are not safely extrapolated from one early point, especially across warmup, decay, and LEAF’s repeated LR cycles. A candidate can be behind at 15% and recover; it can also fit vector loss early without recovering near-boundary rankings.  
   **Change:** Log fixed logarithmic checkpoints, but kill only when an empirically calibrated optimistic upper envelope is below the gate. Otherwise use the early read diagnostically. Do not fit a linear trajectory through one checkpoint.

4. **[BLOCKER] The “scaled LEAF recipe” is probably a severe underdose, not a scaled reproduction.**  
   LEAF used roughly 4.7M document-like texts, 1.85M queries, batch 32, 30 epochs, and 100 A100-hours; it also found that more optimizer steps at smaller batches were important ([ACL paper](https://aclanthology.org/2026.acl-long.2008.pdf)). Given the stated 2–4× hardware gap, 30–60 RTX-3080 hours is only about 7.5–30% of LEAF’s accelerator-equivalent budget. That invalidates using LEAF retention as the prior. The claimed 4–6 hours for 1–3M stella targets is also unsupported for real FineWeb length distributions.  
   **Change:** Budget and register examples, non-padding tokens, optimizer steps, sequence-length mix, and checkpoints—not wall time. Run a real 10–50K-text throughput pilot before fixing target-encoding cost.

5. **[MAJOR] The teacher decision rule is wrong even if a screen is restored.**  
   “Any CI-resolved gain breaks the pair” allows a practically meaningless +0.001 result to force a second index, despite the product premise. Conversely, ceiling alone cannot establish which teacher is most distillable.  
   **Change:** Run a cheap teacher screen with one fixed student and identical texts/steps, then screen student backbones only against the selected teacher. Switch away from stella only if the asymmetric student clears both statistical significance and a preregistered practical margin—e.g. lower CI > 0 and point gain ≥0.010—or an explicit cost-weighted utility threshold. “Pair preference” must be a margin, not merely a tie-break at zero.

6. **[BLOCKER] E — LoTTE cannot be both training data and the fresh selection surface.**  
   The draft puts the same LoTTE-clean queries in the regression pool and proposes using them for selection. That destroys freshness. The old suite has 494 accumulated reads and is explicitly not fresh ([M9 brief](/home/dylan/asymetric-dual-encoders/m9/BRIEF.md:80)).  
   **Change:** Remove all seven LoTTE slices from training and reserve them for M9 selection, with a macro over slices rather than query pooling. Use the M7 suite only as a locked descriptive diagnostic; if its result can trigger recipe changes, it remains dev, not a “held-out check.” Disclose that LoTTE is forum-heavy and therefore unusually aligned with the two reserved CQA tasks.

7. **[MAJOR] B — document text is neither obviously waste nor automatically valid query-tower data.**  
   The literature conflicts for a reason. LEAF’s one-epoch ablation reports 46.7 NanoMSMARCO for queries alone and 60.7 for queries+documents, but its student was designed to encode both roles. Wang–Lyu achieved strong query-only distillation using queries alone. LEAF itself concludes queries matter more, while both were needed in its setting ([ACL paper](https://aclanthology.org/2026.acl-long.2008.pdf)).  
   **Change:** Register an equal-step, equal-token-budget query-only versus query-heavy mixed screen—e.g. 100/0 and 70/30. Do not let a 3M-document pool swamp 361K queries implicitly. For document-like texts, explicitly decide whether targets are teacher-query embeddings or unprompted document embeddings; a query-only tower should default to query-role targets.

8. **[MAJOR] C — promptless serving is an untested architectural change, not “baking in” a constant for free.**  
   LEAF supplied instructions to both teacher and student. Stella’s official retrieval path prepends the `s2p_query` prompt, while documents are promptless ([stella model card](https://huggingface.co/NovaSearch/stella_en_400M_v5)). Training raw student text against prompted teacher targets may work, but it asks the student to internalize a fixed task transformation and departs from the closest evidence. It also creates ambiguity for document-like regression texts.  
   **Change:** Screen exactly two policies:

   1. Promptful student → prompted teacher target, matching LEAF.
   2. Raw student text → prompted teacher target, matching intended promptless service.

   Freeze the winner’s literal tokenizer input as part of the artifact. Do not mix unprompted document targets into the same promptless mapping without an explicit role marker.

9. **[BLOCKER] The contamination plan is inadequate for the reserved evaluation.**  
   Query-level decontamination does not neutralize training on FEVER pairs, FEVER task structure, or overlapping documents. FineWeb can contain Wikipedia/DBpedia/BEIR corpus duplicates. “Not disclosed” is not evidence of no overlap. CQADupStack training must also prove that Android and English slices, including near duplicates, are absent.  
   **Change:** Remove FEVER entirely from M9 training if FEVER will appear in any confirmation table. Hard-exclude the reserved CQA slices. Specify exact and near-document duplicate controls for FineWeb. If such filtering requires opening reserved corpora before training, admit that it conflicts with the one-access design and either abandon FineWeb or revise the reserve protocol before training.

10. **[BLOCKER] D — FEVER must not participate in the confirmatory gate.**  
    Stella has disclosed FEVER exposure, and the proposed M9 stack compounds it. Calling query-level decontamination sufficient will not survive review.  
    **Change:** Treat FEVER as a labeled, double-contaminated sensitivity row only. Do not allocate alpha to it and do not let it determine release.

11. **[BLOCKER] The reserved-four macro is badly structured as a primary estimand.**  
    Corpus size is not the main problem—nDCG macro need not weight by document count. The problem is that two of three nominally cleaner tasks are sibling CQA slices, so an equal dataset macro gives one task family two-thirds of the weight. It also supports inference over queries conditional on four fixed datasets, not generalization over retrieval tasks.  
    **Change:** Use a preregistered task-family macro on the three no-disclosed-overlap sets:

    \[
    0.50\,\text{DBpedia} + 0.25\,\text{CQA-Android} + 0.25\,\text{CQA-English}.
    \]

    Report the ordinary dataset macro, pooled-query result, per-dataset results, and leave-one-dataset-out values descriptively. Call this “no-disclosed-overlap-3,” not “clean-3.”

12. **[BLOCKER] The confirmatory alpha rule is underspecified and inherits a known weak-null defect.**  
    “Holm α=0.025,” “CI excluding zero,” and `signflip_dep p<0.05` do not define one coherent family rule. The repo already found that the sign-flip procedure can be mildly anti-conservative under the weak mean null.  
    **Change:** Freeze exactly two confirmatory contrasts on the grouped macro:

    1. Nano dense vs bge-small symmetric — release claim.
    2. Nano dense vs leaf-ir-asym — owner aim.

    Require, for both contrasts, a one-sided Bonferroni 98.75% stratified paired-bootstrap lower bound above zero, plus Holm-corrected dependent sign-flip rejection at family α=0.025 as a conjunct/sensitivity. Bootstrap queries within each dataset and then form the fixed weighted macro. Everything else is descriptive and receives no post hoc inferential language.

13. **[MAJOR] Scoring every frontier system is acceptable only as a descriptive batch, but the resource plan does not cover it.**  
    The run needs at least three document towers: stella, arctic, and bge. Across roughly 10.1M documents, their raw fp16 vectors alone are about 20.6GB + 15.5GB + 7.7GB ≈ 43.8GB, excluding identifiers, indexes, BM25, scratch space, and fp32 accumulation. It also entails roughly 30M document encodes, not 10M.  
    **Change:** Keep the broad one-access batch only after a full storage/throughput rehearsal. Pre-download and hash every model, rehearse the entire pipeline on open datasets, define crash/restart semantics, and suppress human-readable intermediate scores until every system completes. Broad descriptive scoring does not enlarge multiplicity; broad inferential fishing would.

14. **[BLOCKER] F — the shipping artifact is undefined at confirmation time.**  
    M10 cannot introduce int8 weights, a new ONNX graph, altered pooling, output quantization, or a changed sequence limit and still claim the M9 confirmation applies. Backbone-only ONNX feasibility before training is insufficient.  
    **Change:** M9 must freeze and confirm the exact shipping path: tokenizer, prompt policy, max length, pooling, normalization, output head, weight precision, output precision, ONNX graph, and runtime preprocessing. Export a skeleton before training, then export and parity-test the final checkpoint before the one-access run. M10 may package it, not change its numerics.

15. **[MAJOR] Fusion cannot be bolted on after dense confirmation.**  
    The repository already says changed checkpoints invalidate fusion weights, and M8 found fusion to be the dominant product gain ([M8 findings](/home/dylan/asymetric-dual-encoders/m8/FINDINGS.md:82)).  
    **Change:** Tune nano+BM25 afresh on LoTTE using a fixed grid that includes the dense endpoint. Keep “beats LEAF” strictly dense-versus-dense. Report nano+BM25 as a product point descriptively unless leaf+BM25 and bge+BM25 receive equivalently selected fusion policies and an explicitly expanded alpha family.

16. **[MAJOR] MiniLM’s long-query risk is larger than the draft admits.**  
    Its original fine-tuning used sequence length 128 and its packaged SentenceTransformer truncates beyond 256 WordPieces by default ([model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)). Stella recommends and was trained at 512. Simply setting MiniLM to 512 does not erase its short-sequence initialization.  
    **Change:** Fix serving at 512 for both finalists unless latency forces a registered alternative. Report tokenizer fertility, `[UNK]` rate, truncation rate, retained-token fraction, and retrieval loss by query-length/fragmentation bins. Include an exact head-only versus head+tail truncation probe for long queries if nonstandard token selection is allowed; otherwise state first-512 as a limitation.

17. **[MAJOR] “Pure L2” and the fp16 target path are not specified precisely enough to reproduce.**  
    LEAF uses the Euclidean norm; MSE is squared Euclidean divided by dimension. With normalized student and teacher outputs, squared L2 and cosine loss are affine equivalents, so “MSE+cosine” adds no new information—only a rescaled duplicate gradient.  
    **Change:** Specify unsquared L2 versus MSE, exactly where normalization occurs, reduction across dimensions, epsilon, and autocast behavior. Storing fp16 targets is reasonable only after measuring angular/cache error; cast minibatch targets and predictions to fp32 for normalization and loss. Verify cached fp16 targets against live fp32 teacher outputs and retrieval on a representative open subset.

18. **[MAJOR] Phase 2 is triggered by the wrong symptom.**  
    “Retention below bar” does not tell you that ranking KL is the remedy. Hard-candidate entropy from M8 belongs to a table-specific KL objective; pure embedding regression has no candidate bank and cannot be diagnosed by that 0.777-nat result.  
    **Change:** Trigger ranking preservation only if vector error is low but retrieval disagreement remains concentrated around teacher top-k margins. If vector error remains high broadly, spend on more regression steps/data coverage. Drop MSE+cosine as a nominal second objective when outputs are normalized.

19. **[MAJOR] G — two student finalists are enough; a third is wasted unless it changes a known risk.**  
    MiniLM-L6 and bge-small cover the meaningful trade: LEAF-proven compact initialization versus retrieval-tuned, 512-capable initialization.  
    **Change:** Use cheap frozen-backbone linear/Procrustes alignment to initialize or reject obviously bad heads, then run one identical short chain per backbone. Spend the marginal GPU-day on:

    - longer phase-1 training for the winner;
    - at least two additional fixed-seed robustness runs;
    - the prompt/data-mixture ablation.

    Do not spend it on a third backbone or phase-2 objective work absent the diagnostic in finding 18. Ship the preregistered seed, not the best seed; report between-seed variation because query-bootstrap CIs do not contain training uncertainty.

20. **[MAJOR] F — the 3–5 ms latency claim is currently advertising, not measurement.**  
    It lacks hardware, CPU thread count, runtime, tokenizer inclusion, input length, batch size, warm/cold status, and percentile. The 1024d stella index also costs more storage and dot-product work than LEAF’s 768d arctic index.  
    **Change:** Benchmark batch-1 ONNX Runtime on named edge CPU hardware with fixed threads, tokenizer included, length buckets, warm p50/p95, cold start, peak RSS, model bytes, and load time. Benchmark MiniLM, bge-small, and leaf under the same setup. Report document encode throughput, dimension, raw/index bytes, and exact-search latency separately from query encoding.

21. **[MINOR] MRL is not a free smaller-index option here.**  
    Stella’s alternative dimensions use separate learned dense heads; replacing the 1024d head requires re-encoding documents. Prefix-truncating the frozen 1024d vectors is not justified by the model packaging.  
    **Change:** Defer MRL/smaller-index science. If retained, define it as a separate system with a separate document index and full parity/quality evaluation, not an inherited property of nano.

22. **[BLOCKER] H — “nano beats LEAF” is misleading unless phrased as a system comparison.**  
    A hostile reviewer can correctly say the result may come from a 400M, 1024d document teacher beating a 109M, 768d teacher—not from a better student or distillation method. Stella also has disclosed overlap with two of six tasks, while arctic discloses none. TREC-COVID already shows the stronger teacher does not dominate everywhere.  
    **Change:** The report’s permitted headline should be:

    > “The stella-document + nano-query asymmetric system outperformed the arctic-document + leaf-query system in our exact-search harness.”

    It must not say “nano is better than LEAF” or imply a student-method win. Alongside the headline, report teacher-normalized retention, teacher sizes, dimensions, index bytes, document-encoding cost, query latency, per-dataset results, TREC-COVID loss, disclosed training overlap, and the no-disclosed-overlap-4 result. Rename “clean-4” accordingly. If that four-set contrast is not CI-resolved, the headline must be restricted to the contaminated six-set macro.

23. **[MAJOR] Frozen comparator provenance needs a bridge check.**  
    Nano will be produced under a later software/runtime path than the irreplaceable comparator vectors. A tokenizer, qrels alignment, missing-query, tie-breaking, normalization, or dtype drift can manufacture a paired difference.  
    **Change:** Before scoring nano, regenerate at least one reproducible anchor such as bge-small on the six and require per-query/aggregate agreement with its frozen vectors within a registered tolerance. Hash query IDs, ordering, qrels, preprocessing, revisions, dtype, and exact-search code. Never overwrite [results/perquery.json](/home/dylan/asymetric-dual-encoders/results/perquery.json).

The central correction is: make LoTTE genuinely fresh, freeze the actual shipping artifact in M9, treat the retention literature as a prior rather than a forecast, and frame the result as a comparison of asymmetric systems with unequal document towers. Without those changes, even a score above 0.5155 would be easy to dismantle.
