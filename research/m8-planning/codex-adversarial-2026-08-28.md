The main conclusion is harsher than “the clean data stack cost too much”: M7 tested an unnecessarily tiny, tokenizer-coupled additive model, trained it with an objective that changes meaning between phases, and selected it on a suite that rewarded precisely the exposures that failed to transfer. It did not establish the ceiling of zero-transformer query encoders.

## Closed avenues that are not actually closed

1. **CRITICAL — The hard-negative avenue is internally contradicted, not closed.**

   Evidence: `EXPLORED.md` says mined negatives are closed, but the ledger says the result is “NOT IDENTIFIED.” At matched 2,500 steps, teacher and mixed negatives improve the dev macro by about **+0.0112/+0.0111**; they lose only after the broken per-arm proxy selects 1,500/1,000 steps. The out-of-domain instrument could only resolve effects around 0.005, and the registered false-negative diagnostic was vacuous. See [LEDGER.md](/home/dylan/asymetric-dual-encoders/m7/LEDGER.md:307) and [EXPLORED.md](/home/dylan/asymetric-dual-encoders/m7/EXPLORED.md:60).

   What may have cost quality: reverting a +0.011 matched-step arm because a proxy known to rank arms backwards chose different training durations.

   Confirm/kill: from one fixed B checkpoint, run baseline, teacher-16, and mixed-32 at exactly the same A steps, pooling, seed family, and objective. Evaluate on a new OOD-heavy dev instrument. Replace binary treatment of negatives with teacher soft labels so unlabeled positives are not declared irrelevant. Kill only if the matched arms remain below baseline across at least three dataset families and the effect is smaller than the recipe-perturbation band.

2. **CRITICAL — N-grams were never tested in the form that could work.**

   The −0.0301 result fitted 10K bigram residuals in closed form toward the teacher query vector, onto an earlier `s2w-1e3-s1000` table—not the final M7 model. That supervision is already known to undo A-phase retrieval gains. It does not test:

   - A-only training of phrase rows while freezing the unigram table.
   - Joint B+A training through n-gram features.
   - More than 10K phrases.
   - Trigrams, cross-word units, or character/entity features.

   An earlier repo review explicitly proposed the missing A-only experiment, but it never ran. See [m7_bigram_residual_k10000.json](/home/dylan/asymetric-dual-encoders/results/m7_bigram_residual_k10000.json) and [m7-codex-review-2026-08-27.md](/home/dylan/asymetric-dual-encoders/research/m7-codex-review-2026-08-27.md:159).

   Confirm/kill: freeze the final M7 unigram rows, add 10K zero-initialized bigram rows, and train only those under the ranking objective. This is roughly **10.2 MB int8**. If that beats M7 on robust dev, escalate directly to the full phrase architecture below. If it loses broadly, it kills this initialization/training form—not jointly trained n-grams.

3. **HIGH — “Training through sqrt pooling” was killed by an invalid escalation rule.**

   Arm (a) gained +0.0011 with a CI crossing zero, so the full B16K→A chain was never run. But B is where 924K pseudo spans teach token composition; failure of an A-only adjustment cannot falsify training the representation under the served multiplicity rule from the beginning. M7 itself says the table was trained for mean pooling and that this understates the possible gain. See [LEDGER.md](/home/dylan/asymetric-dual-encoders/m7/LEDGER.md:531).

   Confirm/kill: train one full matched chain under `sqrt`, including all B pseudo spans, versus an otherwise identical mean chain. Because repetition may be rare, first report the fraction of queries with repeated emitted features and the score effect conditional on repetition. Kill if the full chain fails and fewer than roughly 10% of queries are affected materially.

4. **HIGH — Doc2query was stopped at the least informative dose.**

   The N=5 probe was positive on both tested components: **+0.0054 [−0.0007,+0.0114]**, at one-eighth of the published 40-query dose. The repo’s own audit recommended N=20 as a diagnostic; it did not run. “Protocol-closed” is not “mechanistically dead.”

   That said, this should not lead M8 because the releasable generator remains unresolved and document expansion is expensive. Run N=20 only if it determines whether to seek Dylan’s licensing ruling. A monotone or accelerating dose response justifies a clean generator; a flat N=5→20 curve kills it.

5. **HIGH — The bare-target question remains open and directly attacks the distillation target.**

   Runtime removal of the teacher’s prefix is not the experiment. The missing experiment is training or fitting the table against the teacher’s **bare query embedding**. A prompt may raise the transformer ceiling while making its representation less token-additive—the exact ceiling→table inference M7 disproved elsewhere.

   Confirm/kill: the already-designed closed-form test costs about 1–1.5 hours. Compare prefixed and bare targets on a cross-domain dev suite, including ranking quality rather than cosine alone. If bare wins, carry it into B training; also test a mixture of bare-vector and prefixed ranking targets.

6. **MEDIUM — Several algebraic closures are valid only as capacity claims.**

   Centering, whitening, token scalar weights, and top-PC removal are absorbable into a free table. That proves they add no representational capacity; it does **not** prove they are useless as optimization priors or regularizers. Learned token weights are the one M7 component with a resolved main effect. Likewise, post-hoc row shrinkage failing does not falsify hierarchical subword/phrase regularization during training.

   Do not spend a major M8 branch here, but permit these transforms as pre-registered training priors. Kill them on cross-domain validation, not algebra.

   The genuinely sound closures are fp32 training encodes, positive length scaling before final normalization, the fixed-step objective-C LR test, MTEB-as-teacher-ranking, the tested losing teachers under the M7 probe, BM42 under the zero-neural-query premise, and prohibited vendor/licence routes.

7. **HIGH — The clean-stack tax supports a narrower conclusion than M7 states.**

   The MS MARCO arm added pairs and query text but deliberately:

   - Added no MS MARCO pseudo spans.
   - Drew no MS MARCO negatives.
   - Used the frozen objective and phase schedule.
   - Used a side-bank integration unlike a normal corpus addition.

   Therefore +0.0058 establishes: “MS MARCO, inserted this way into the frozen M7 recipe, does not close the gap.” It does not establish that better clean data construction, query generation, negative mining, or source balancing cannot close it. “The remaining gap is architectural” should include objective and data-to-objective plumbing—not merely the table function. See [LEDGER.md](/home/dylan/asymetric-dual-encoders/m7/LEDGER.md:757).

## Architecture I would build

8. **CRITICAL / highest EV — Spend the conservative 233 MB budget on compositional features.**

   M7 uses:

   \[
   30{,}522\times1024\times1\text{ byte}=31.25\text{ MB}.
   \]

   Same-precision comparison with LightRetriever leaves approximately:

   \[
   233.6\text{ MB}/1024 - 30{,}522 \approx 197{,}600
   \]

   additional int8 rows. Against LightRetriever’s fp16 footprint, the absolute allowance is about 424K additional rows.

   My first M8 architecture would be:

   - 30,522 WordPiece backoff rows.
   - 96K explicit cross-token bigrams.
   - 32K explicit trigrams.
   - 64K hashed character/word-unit buckets for entities, morphology, and spelling variants.

   Total: **222,522 rows**, or **227.9 MB int8**. Per-row fp32 scales cost 0.89 MB; explicit bigram/trigram IDs roughly 1.2 MB. It fits under 233.6 MB with a few MB for metadata.

   Select explicit phrases by frequency plus association strength using release-clean training text, before dev evaluation. Train all features through the forward path, with feature dropout/backoff so a seen phrase does not replace generalizable unigrams completely.

   Expected value: **+0.01 to +0.03 nDCG**, because it attacks the only demonstrated structural deficiency and only 0.0243 was needed to reach the old dense comparator.

   Confirm/kill: require at least +0.008 on the OOD development macro, nonnegative signs on at least four heterogeneous dataset groups, and improvement concentrated on queries containing emitted phrases—not merely TRAIN-adjacent datasets.

9. **HIGH — Decouple the student tokenizer from Stella’s tokenizer.**

   There is no mathematical requirement that the table use the teacher’s 30K WordPiece vocabulary. The teacher tokenizes strings to produce targets; the student may tokenize the same strings any way it wants.

   Concrete alternatives:

   - 128K clean-trained unigram/BPE vocabulary: **131.1 MB int8**.
   - 151K vocabulary matching LightRetriever’s row count: **154.6 MB int8**.
   - 200K vocabulary: **204.8 MB int8**, still below the 233.6 MB int8 comparator.

   This exposes a stale assumption in `CLAUDE.md`: the ≤50K vocabulary filter was derived from fp16 arithmetic even though M7 established int8 as quality-free. A larger tokenizer can emit whole entities, domain terms, and longer lexical units without requiring a separate phrase matcher.

   Confirm/kill: train 64K, 128K, and 200K tokenizers once on the same approved corpus, then use a cheap ridge/B-phase probe with matched targets. Measure query length, cold-row rate, entity fragmentation, and OOD retrieval. Do not initialize 200K tokens with 200K teacher forwards; initialize each new unit from the pooled M7 WordPiece decomposition, then train.

10. **MEDIUM — Reject naive “multiple vectors per token”; test only genuinely nonlinear variants.**

   If four vectors for a token are always summed, they collapse algebraically into one row. That spends 4× the bytes for no extra function.

   Genuine options are:

   - Four independently pooled and normalized query heads, followed by four ANN searches and max/RRF fusion: **125 MB int8**, approximately 4× ANN latency.
   - Count-state rows \(v_{t,1},v_{t,2},v_{t,3},v_{t,4+}\): also **125 MB**, but useful only for repeated tokens.
   - Position-bucket/token interaction rows, which add order but risk poor transfer.

   Measure query repetition and head complementarity first. Kill count-state rows if repetition is rare; kill multihead retrieval if head top-k overlap is high or latency approaches the 20–100M transformer tier without at least +0.01 OOD gain.

11. **HIGH / Dylan scope question — A sub-0.1 ms nonlinear matmul may dominate the pure table.**

   A linear projection after pooling is absorbable and adds no capacity. A nonlinearity is different:

   \[
   h=\sum_i E[t_i],\qquad q=\operatorname{norm}(W\,\mathrm{GELU}(h)).
   \]

   With 200K rows at hidden width 768:

   - Table: \(200K\times768\) int8 = **153.6 MB**.
   - Projection: \(768\times1024\) fp16 = **1.57 MB**.
   - Compute: **0.79M MAC/query**.

   Even at only 10 GFLOP/s effective CPU throughput, this is around 0.08 ms of arithmetic—far below the repo’s approximately 4.6 ms small-transformer tier, though it must be measured end-to-end. It creates cross-token interaction through the activation and gives a 200K inventory at roughly half the conservative budget.

   This is not the exact M7 artifact class, but it remains zero-transformer and near-zero compute. Put it to Dylan explicitly. Kill if measured CPU latency exceeds 0.25 ms or robust gain is below +0.01.

## Document side and fusion

12. **CRITICAL / very high EV — Train an index-time document adapter before fine-tuning Stella.**

   The repo corrected its own algebra: a document transform followed by per-document normalization is not absorbable. Yet this obvious lever was never exploited.

   First rung:

   \[
   d'=\operatorname{norm}(Md)
   \]

   where \(M\in\mathbb{R}^{1024\times1024}\). Cost: **2.10 MB fp16**, 1.05 MB int8.

   Stronger rung:

   \[
   d'=\operatorname{norm}\bigl(d+W_2\,\mathrm{GELU}(W_1d)\bigr)
   \]

   with a 256 bottleneck. Cost: about **524K parameters**, or **1.05 MB fp16**.

   Train the adapter jointly with the table using cached Stella document vectors. That means no 400M transformer backward pass and likely hours, not days. Apply the adapter once during indexing. Regularize toward Stella’s original rankings so the system does not destroy the 0.5744 teacher geometry while making it easier for the table to address.

   Confirm/kill: compare table-only, adapter-only, and jointly trained table+adapter across three recipe seeds. Promote only if the OOD macro improves by at least +0.008 without more than a 0.01 regression in any major domain group.

13. **HIGH EV but expensive — Escalate to Stella LoRA only after the adapter proves the premise.**

   Full Adam fine-tuning of 400M parameters needs roughly:

   - 0.8 GB fp16 weights.
   - 0.8 GB gradients.
   - 1.6 GB fp32 master weights.
   - 3.2 GB fp32 Adam moments.

   That is about **6.4 GB before activations**, making full tuning marginal on 10 GB. Gradient checkpointing and tiny microbatches could fit, but LoRA is the defensible first implementation. Depending on targeted modules and rank, expect roughly 5–20M trainable parameters.

   Budget: approximately **12–48 GPU-hours** for co-training experiments, then the repo’s established **8–12 hours** to re-encode the 6.17M-document pool. The zero-query constraint is unaffected. The approved Apache-compatible Stella lineage should remain compatible, but derivative-weight and training-data provenance still needs the normal release audit.

   Objective: optimize the document tower to be table-addressable while distilling original Stella score distributions. Do not fine-tune documents against hard one-positive InfoNCE alone.

14. **HIGH EV — Replace BM25 with an in-house document-only learned sparse arm.**

   Fusion is not a side note. It added **+0.057** on the final six, including +0.153 TREC-COVID and +0.097 SciFact. That says the best system is likely hybrid, and the sparse arm deserves model capacity.

   Build a DeepCT/DeepImpact-style arm:

   - Query: binary/count-saturated lexical tokens × fixed IDF; no neural query computation.
   - Documents: Stella contextual token importance plus optional top-K vocabulary expansion, computed offline.
   - Train term impacts and expansion against clean labels plus teacher scores.
   - Release your own head/adapter, not a prohibited competitor component.

   Primary literature finds document term weighting to be the most important learned-sparse component, with query weighting comparatively small ([Unified Framework for Learned Sparse Retrieval](https://arxiv.org/abs/2303.13416)); DeepImpact demonstrates offline contextual impact weights and document expansion in a standard inverted index ([DeepImpact](https://arxiv.org/abs/2104.12016)).

   Existing cost arithmetic gives approximately:

   - 64 nonzeros/document: \(1.4\text{ GB}\times64/233\approx0.38\) GB per million documents.
   - 128 nonzeros/document: about **0.77 GB/M**.
   - 233 nonzeros/document: measured **1.4 GB/M**.

   Confirm/kill: it must beat BM25 on the robust dev macro and improve fusion on at least two domain groups. Also measure top-1000 complementarity with the dense arm; a sparse model that merely reproduces the dense ranking has no fusion value.

   Afterward, test query-adaptive fusion using only cheap features—query length, IDF statistics, dense/BM25 score gaps, and run overlap. A small linear gate is negligible compute, but promote it only under leave-one-dataset-out evaluation.

## Training recipe failures

15. **CRITICAL — The B→A phase transition discards the thing B learned.**

   In code:

   - Phase B uses cosine plus a 32-candidate KL loss.
   - Pseudo spans receive **cosine only**.
   - Phase A switches to **InfoNCE only**; distillation is gone.
   - Each A step samples one positive even when queries have many positives.

   This is a plausible forgetting mechanism. The clean-stack arm’s B-stage gain of about +0.0069 mostly disappears after A. M7 also established that cosine agreement can rise while retrieval falls, yet cosine retains unit weight in B. See [train.py](/home/dylan/asymetric-dual-encoders/m7src/train.py:419) and [train.py](/home/dylan/asymetric-dual-encoders/m7src/train.py:636).

   M8 should use one continuous mixed objective:

   \[
   L=L_{\text{multi-positive rank}}
     +\alpha L_{\text{teacher-listwise}}
     +\beta L_{\text{query-vector}}
     +\gamma L_{\text{init}}.
   \]

   Use log-sum-exp over all known positives, teacher-soft scores over mined candidates, and 20–30% B-style replay throughout relevance training. Cosine should be a small regularizer, not the main pseudo-query loss. Recent primary work reports that ordinary InfoNCE fine-tuning can degrade dense retrieval and finds listwise distillation more reliable across domains ([listwise distillation study](https://arxiv.org/abs/2502.19712)).

   Confirm/kill: matched three-arm comparison—M7 sequential, mixed-objective replay, and listwise-only—with equal optimizer updates. Track B-target retention before and after A, not just final nDCG.

16. **HIGH — The fixed temperature makes most of 32,768 negatives wasted compute.**

   At τ=0.02, the measured effective negative count is only **28.9 out of 32,768**; the top 100 hold 80% of mass. At τ=0.05, effective negatives jump to **5,305**. Thus the never-run `phase3_hparams` sweep is not housekeeping—it can change the loss qualitatively. See [m7_diag_scores.json](/home/dylan/asymetric-dual-encoders/results/m7_diag_scores.json).

   Run a matched 2×2:

   - τ ∈ {0.02, 0.05}.
   - negatives ∈ {8,192, 32,768}.

   Keep steps, pooling, and hard-negative composition fixed. Separately tune the teacher KL temperature; it need not equal the InfoNCE temperature. If 8,192 performs equivalently, A training becomes about four times cheaper at its dominant matrix multiplication, buying capacity experiments.

17. **HIGH — The pseudo-query pool is badly source- and grammar-confounded.**

   The 924,704 realized spans are:

   - ESCI: 400,001.
   - HotpotQA: 400,001.
   - FEVER: 12,484.
   - SQuAD: 18,844.
   - Mr. TyDi: 94,655.

   ESCI plus HotpotQA are therefore **86.5%** of the pool. The samples are first document sentences capped at 32 words—“query-shaped in length,” explicitly not in grammar. This almost perfectly explains why dev selection rewarded e-commerce/Wikipedia-adjacent behavior. See [pseudoq.py](/home/dylan/asymetric-dual-encoders/m7src/pseudoq.py:43).

   Rebuild the pool with:

   - Exact, logged per-source quotas.
   - Titles, headings, short keyword deletions, questions, and declarative spans as separate strata.
   - Sampling targeted at tokenizer/phrase coverage rather than document count.
   - A fixed mixture of prefixed and bare teacher targets.
   - No source exceeding 25% of pseudo updates.

   Confirm/kill with matched B16K→fixed-A runs. Report gains by query grammar, vocabulary coverage, and source family. Do not select another “2m” request whose realized composition changes invisibly.

## Evaluation and selection

18. **CRITICAL — M8’s reserved suite is a weak broad-generalization instrument.**

   The binding four contain:

   - FEVER: TRAIN-adjacent with 11.3% document overlap.
   - Android and English: the same CQADupStack family used heavily in M7 development.
   - DBpedia: only 400 queries and the sole substantially novel task family.

   Thus a successful M8 result confirms replacement quality on these four; it does not establish general OOD superiority. Two CQA components should not count as two independent domains.

   Binding-safe action: define the primary metric before training as:

   \[
   \frac{\text{FEVER}+\text{DBpedia}+
   (\text{Android}+\text{English})/2}{3}.
   \]

   Dylan-only flag: if a fifth clean, non-Wikipedia, non-CQA final set can legally be added before any M8 access, do it now. Otherwise state the limitation as part of the primary claim, not a footnote.

19. **CRITICAL — Build an OOD-first dev suite and stop selecting on the arithmetic mean.**

   Requirements:

   - At least half the selection weight comes from source/task families absent from TRAIN and absent from disclosed teacher training.
   - CQA subforums form one group.
   - Wikipedia/QA and TRAIN-adjacent components form one group rather than dominating by count.
   - Maintain an exploratory dev partition and a shadow-dev partition. Architecture families see the first; only frozen family winners see the second.
   - Treat M7’s clean four as already-burned diagnostics, not new evidence.
   - Select using median or worst-group gain, not the all-component macro.

   Promotion bar I would use: robust-dev point gain ≥0.008, raw CI lower bound >0, at least four heterogeneous domain signs nonnegative, and no domain-group regression worse than −0.01.

   Train three fixed seeds or perturbations for every finalist. Either average their tables with a pre-declared equal-weight “table soup” or select the median recipe mechanically—never the best seed. Table averaging costs zero runtime bytes.

20. **HIGH — Keep one primary replacement claim; do not squander power on another tier ladder.**

   Pre-register the required Holm family, but make the release decision depend on exactly one comparison: **M8 released system > frozen M7 released system** on the grouped reserved metric. Require:

   - Point gain of at least **+0.005**, because smaller gains live inside M7’s measured recipe-perturbation band.
   - Raw paired CI lower bound >0.
   - The registered simultaneous/Holm condition.
   - No reserved domain-group regression worse than −0.01.

   Dense-vs-dense, BM25, and learned-sparse decompositions can remain secondary registered comparisons. The full system—not merely its dense arm—must be the replacement candidate, because M7’s strongest result is its fusion.

## Recommended M8 order

21. **Highest-EV execution sequence.**

   1. Freeze the M8 grouped dev metric, shadow suite, seed policy, and final bars.
   2. Run four cheap discriminators: bare target, A-only 10K bigrams, matched hard negatives, and normalized document adapter.
   3. Run the 2×2 temperature/negative-count sweep under matched steps.
   4. Build the approximately 230 MB compositional table and train it with continuous listwise/replay supervision.
   5. Co-train it with the document adapter.
   6. Build the document-only learned sparse arm and reselect fusion.
   7. Only if the cheap document adapter shows clear headroom, spend the 20–60 hours on Stella LoRA and corpus re-encoding.
   8. Put the nonlinear 0.79M-MAC query encoder to Dylan as a scope decision before freezing the pure-table family.
   9. Freeze one full system, then spend the reserved four once.

   Do not begin with another teacher sweep. Stella was selected using a unigram closed-form probe on only two CQA components, with a stale fit set containing 1.31% protected-query hits. Once the tokenizer, phrase capacity, or document geometry changes, teacher ordering may change too—but the architecture must be fixed before that comparison is meaningful.

22. **HIGH operational severity — The claimed 31.3 MB artifact is not the frozen file in the repo.**

   `p35w-2m-s2500.release.npz` is **93,886,950 bytes**. The exporter writes both `rows_fp16` and `rows_int8` into the release container, plus scales and bookkeeping. See [table.py](/home/dylan/asymetric-dual-encoders/m7src/table.py:318) and [m7_final_run.json](/home/dylan/asymetric-dual-encoders/results/m7_final_run.json:20).

   Therefore 31.3 MB is the int8 payload, not the currently frozen release file. M8 needs an int8-only deployment format and must report both payload and actual downloadable bytes. This also matters for the proposed 230 MB model: the current exporter would silently create a roughly 690 MB dual-precision container.

   Confirm/kill: export an int8-only artifact, reload it through the production path, verify bit-identical rankings against the int8 variant, and measure actual disk size, load time, and peak resident memory.

The core bet should be compositional query capacity plus a normalized document adapter, trained under a continuous ranking/distillation objective. Hyperparameter cleanup alone is unlikely to recover 0.024; another 31 MB unigram table with a better temperature is still the wrong model class.
