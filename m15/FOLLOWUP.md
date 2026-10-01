# Research follow-up: query compute and search precision

2026-10-01. The owner welcomes justified additional research and compute, with no full-model
retraining and no publication rush. The owner also proposes quantization, provided comparisons
remain fair. This file records the scientific question, experiments, findings, and next decisions.
The v13 paper is a credible applied study; stronger discoveries require evidence, not stronger wording.

## The question worth spending on

**Does replacing only the query encoder change the search precision and candidate budget needed
by a fixed document index?** If so, can a cheap query-side intervention or request setting repair
the penalty without changing the original retrieval target or rebuilding documents?

OOD graph-search difficulty and query-aware quantized scoring have substantial prior art. Our
specific condition is a static, compact-transformer, and original-teacher query path over the same
frozen text document space. Novelty would come from a reproducible encoder/precision interaction
and an explanation or useful remedy under that fixed-index constraint, not another quantizer sweep.
Verified sources and competing explanations are in `REVIEWS/2026-10-01-followup-literature.md` and
`REVIEWS/2026-10-01-followup-design.md`.

## E16: a same-graph pilot is complete

Source: `m15src/e16_quantization_pilot.py`; immutable receipt:
`results/m15_e16_quantization_pilot.json`; method: the E16 section of `MEASUREMENTS.md`.

- 192 FiQA queries selected deterministically by query-ID hash, not by performance; all 57,638
  cached documents. No document encoding, model training, or cloud rental.
- One disk-backed binary collection and fixed graph for every query tier and request option.
  Original exact search matches NumPy top10 perfectly for all three encoders. Collection
  configuration and point/indexed counts match before and after interleaved queries.
- Query-resampling intervals condition on the one graph and fixed models. These are exploratory
  pilot results, not general architecture laws or confirmatory tests.

At the primary `ef=64`, exact-neighbor recovery@10 is:

| Query path | Original-vector scoring | Binary + rescore, oversampling 1 | Binary + rescore, oversampling 4 |
|---|---:|---:|---:|
| Zero | 0.9802 | 0.8635 | 0.9521 |
| Nano | 0.9964 | 0.9453 | 0.9927 |
| Stella | 0.9995 | 0.9766 | 0.9990 |

The paired binary-minus-original recovery penalties are 11.67 percentage points for Zero,
5.10 for Nano, and 2.29 for Stella. Zero's extra penalty is 6.56 points versus Nano
(95% query-bootstrap interval 4.69–8.44) and 9.38 versus Stella (7.40–11.41).
The interaction persists at `ef=16` and `ef=256`. Increasing `ef` alone leaves a substantial binary
penalty; widening the rescoring candidate budget helps markedly. Neighbor recovery and relevance
quality are different outcomes; the receipt records nDCG beside recovery and does not equate them.

These native API treatments change traversal/scoring behavior and potentially candidate handling.
They do not isolate exhaustive quantization error or a pure rerank over an identical candidate pool.
In particular, native rescore/no-rescore return different neighbor sets even with oversampling 1.
The original stored vectors are fp16 values decoded to fp32, not fresh full-precision encodes.
The independent pilot review finds no essential control defect and recommends a second-scale test.

## Original sequence (now completed through the precision control)

1. **E17 replication:** repeat the paired same-graph request-option test on the cached 1M MS MARCO
   diagnostic, with the same deterministic 192-query sample rule and fixed condition grid. The
   subset preserves positives and removes competitors; it remains validation-only and is not full
   MS MARCO. Local execution needs no rental. This tests domain/scale sensitivity, not universality.
2. **If the interaction repeats:** verify the native binary query precision and candidate-handling
   semantics; distinguish quantized ranking/candidate omission from graph omission before claiming
   a mechanism. Use fixed pools/original reranking or validated exhaustive compressed scoring.
3. **One bounded remedy:** only after the diagnostic, test a small navigation correction selected
   on development queries. Keep the original Zero vector as the final scorer so improvement cannot
   come merely from changing the retrieval target. Include the transform and candidate handling
   costs, and compare against retuned `ef` and oversampling. No full encoder retraining.
4. **Alternatives:** a cost-aware router is useful if it beats simple/static policies after extra
   searches; a small contrasting-teacher fit can test the probe-to-trained-recipe bridge. Both are
   less directly connected to the strongest current shared-index result than the precision study.

Quantizing Nano's inference weights is a separate deployment experiment. It could improve the
available query budget without training, but weight quantization alone is established and is less
likely to add new IR knowledge. It should preserve the frozen document space and evaluate quality
and measured encoding cost together rather than compare unmatched precision runs.

## What would justify a stronger paper

A repeated controlled interaction plus a defensible decomposition would add new information about
serving distilled queries, even without a new algorithm. A cheap fixed-index remedy that improves
search cost at matched original-target recovery and retrieval quality would be stronger still.
A null remedy or a result explained by known query hardness is useful if reported accurately;
neither should be marketed as a breakthrough. More checkpoints or an undirected quantizer search
would add bulk before adding an argument.

## E17: the larger replication attenuates the primary interaction

The same fixed protocol ran on the cached 1M MS MARCO diagnostic in about 6.7 minutes.
All three original exact parity checks are 1.0, collection configuration/counts match before/after,
and no model training or cloud rental occurred. Source/receipt hashes pass independent review.

At primary ef64, binary-rescore1 minus original-scoring recovery penalties are 3.91 points Zero,
3.07 Nano, and 2.19 Stella. The extra Zero-vs-Nano loss is 0.83 points with a query-bootstrap
interval spanning −1.15 to 2.81 points; the Zero-vs-Stella interval touches zero. This is not a
strong primary replication of FiQA's interaction. Secondary ef256 gives larger extra Zero losses,
but does not replace the primary result. The effect is workload/size sensitive on these two samples.
Oversampling4 brings recovery close to the original-scoring control for all three at ef64.

Tagged-source inspection establishes that the default one-bit setup also sign-codes queries;
it discards query magnitudes, not only document magnitudes. Native rescore effects include segment
merging. The next discriminating test holds sign-coded documents fixed and changes query precision
in exhaustive candidate scoring, with each encoder's original exact neighbor target held fixed.

## E18: query magnitudes restore candidate coverage with document codes fixed

A controlled exhaustive scoring test is complete in `results/m15_e18_query_precision.json`.
Each model keeps its own original exact-top10 target and the same E16/E17 query-vector hashes.
Both conditions use identical document sign codes; only sign(q) versus normalized full q changes.
Expected inclusion averages over uniform selection at quantized-score boundary ties. No graph,
new training, document encoding, or native scalar8/latency claim is involved.

Primary global candidate budget 40:

| Workload | Tier | Sign query coverage | Graded query coverage | Gain (percentage points) |
|---|---|---:|---:|---:|
| FiQA | Zero | 0.8217 | 0.9297 | 10.80 |
| FiQA | Nano | 0.9137 | 0.9771 | 6.33 |
| FiQA | Stella | 0.9638 | 0.9922 | 2.84 |
| 1M diagnostic | Zero | 0.9010 | 0.9677 | 6.67 |
| 1M diagnostic | Nano | 0.9484 | 0.9833 | 3.49 |
| 1M diagnostic | Stella | 0.9685 | 0.9932 | 2.47 |

The paired extra Zero gain is 4.46 points versus Nano on FiQA (95% query interval 2.53–6.43), and
3.17 on the 1M diagnostic (1.58–4.75). All budgets 10/40/100 are retained. The larger absolute gain
has more error headroom: the fraction of sign-mode misses restored at 40 is 60.6%/67.4% for Zero
on FiQA/1M, versus 73.4%/67.7% Nano and 78.4%/78.5% Stella. Therefore no claim of universally or
proportionally greater static-model sensitivity follows. Mean cosine from full query to its sign
direction is about 0.80 for every tier. Zero does not show a larger mean angular distortion;
the source of the coverage difference remains unresolved.

The useful finding is **query computation and query precision are separate budgets**, including
for a no-transformer encoder. Preserving magnitudes at search can recover substantial original
candidate information without changing document codes or the encoder. This is a specific measured
consequence of an established asymmetric scoring idea, not a new quantizer or a demonstrated
production speedup. Independent review verifies the boundary-tie mathematics and claim scopes.

### Next justified experiment

Validate native magnitude-preserving query encoding at matched original-target recovery and
retrieval quality, including its scorer and rescoring costs. Qdrant's scalar8 query setting is an
available candidate, but changing that collection setting can rebuild quantization; graph topology
and segment layout must be checked or replicated rather than assumed identical. This requires no
full model retraining. Only pursue navigation calibration if the precision control leaves a
substantial unexplained penalty. More arbitrary precision variants would add less information.

## Owner's generalization question (v14)

The current evidence repeats the scoring effect across two workloads within the same Stella-aligned family. It does not repeat across independently trained teacher spaces. Query bootstraps do not close that gap. The 26-teacher experiment is substantial evidence for its table recipe, but does not validate the probe for trained Zero/Nano.

The next experiment depends on the claim to strengthen. Native magnitude-preserving query precision at matched recovery/relevance and measured cost strengthens a deployment recommendation. A compact C40 scoring replication in a contrasting teacher space, using that teacher's query path and a fitted closed-form table, tests reach beyond Stella without full student retraining. Check artifact availability before budgeting it. One added space is a stress test, not an architecture law. Choose the scientific question before launching either; do not turn the owner's concern into an undirected model sweep.

## Owner round 3 (2026-10-01): experiments A and B replace the precision follow-up as the next work

The owner rejected the native scalar8 precision experiment as a knob rather than knowledge, deferred the domain-fit table to future work, approved A, and accepted B on Astra's agreement (`REVIEWS/2026-10-01-astra-fable-plan-rounds.md`). Both are exploratory under the paper-phase rule; A's analysis is fixed before scoring so the paper can label it pre-specified. Neither trains a full model. Reserved four stay out (known-test). Step zero: start the pod and verify the E8 caches (fit-query vectors per teacher, six-set document vectors per teacher, fitted tables); re-encode only if absent (about 8 to 10 A100 hours).

### A. Cross-recipe teacher screen (E19)

- **Question:** does the teacher ranking produced by one student recipe transfer to a second, independent student family, and does it still ignore teacher retrieval quality?
- **Fixed:** the 26 checkpoints and their own six-set indexes from E8/E8x; the cleaned 337,981 fit queries; the two dev forums for lambda selection; six-set nDCG@10 exact search.
- **Varied:** the student recipe. Recipe 2 is a frozen bge-small backbone (the Nano feature path: layers 12, 8, 4 concatenated, mean pooled, 1152-d) with a closed-form ridge head to the teacher's query vectors, output normalized. Backbone features are computed once; one head solve per teacher. Optional recipe 3, MiniLM-L6 backbone, only if it adds a different test.
- **Measured:** per teacher, recipe-2 six-set and dev-forum scores, retention versus the teacher.
- **Pre-specified analysis (write before scoring):** Spearman between recipe-1 and recipe-2 six-set rankings over the registered ten and over the 26; Spearman between recipe-2 ranking and teacher six-set score; dev-screen-to-public Spearman for recipe 2; checkpoint-resampled intervals and leave-one-family-out as in E13.
- **Reading:** rankings agree and both ignore teacher quality: the strongest-teacher heuristic is challenged for cheap students generally. Rankings disagree: screen per student recipe. Either is reportable; no outcome upgrades the screen to a selector for the trained Zero or Nano.
- **Cost:** under $20 pod time if caches exist; two to three days.

### B. Cross-space ANN effort (E20)

- **Question:** is the extra graph-search effort for table queries a recurring consequence of table-based query substitution across embedding spaces, or a property of Stella, and does query geometry predict it?
- **Fixed:** FiQA (57,638 docs) and TREC-COVID (171,332 docs) document vectors per teacher from the E8 caches; Qdrant 1.19.1, HNSW m=16, ef_construct=100, uncompressed; the ef grid 16, 32, 64, 128, 256, 512; each encoder's own exact top-10 and exact nDCG@10 as its baseline.
- **Varied:** the 26 spaces; teacher queries versus fitted-table queries; two graph builds per space and workload.
- **Measured:** per space, workload, build, encoder, ef: recall@10 against own exact, relative nDCG loss, search p50. Per space: median top-1 cosine, median top-1 minus top-10 margin, for teacher and table queries.
- **Pre-specified analysis:** teacher reference ef is 64. Effort multiplier is the smallest grid ef at which the table reaches the teacher's relative loss at ef=64; unreachable within the grid is reported as censored, not 512. Report the distribution of multipliers across spaces and the paired recovery gap at fixed ef. Geometry correlations are exploratory, not causal. Builds are repetitions; related checkpoints limit independence and are shown by family.
- **Reading:** a recurring penalty generalizes section 4 beyond Stella; absent or reversed penalties bound it. Both are reportable.
- **Cost:** hours of local Qdrant time after copying about 26 x 2 workloads of fp16 vectors from the pod; one to two days.
