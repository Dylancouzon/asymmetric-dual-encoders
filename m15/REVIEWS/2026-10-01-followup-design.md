# Follow-up research design: query substitution and graph search

2026-10-01. Design only; no benchmark, cloud rental, or model training performed. The current paper's correctness is not in question.

## Recommendation and intended new knowledge

Prioritize a small **fixed-index, fixed-scoring-target ANN intervention**. The useful question is not simply whether Zero is farther from documents. It is:

> Can a cheap change to the query used for graph navigation recover the original Zero query's exact neighbors with less search work, while the final scoring rule and document index stay fixed?

If yes, that separates query representation for navigation from query representation for final ranking, and may recover some of the deployment saving without retraining an encoder or rebuilding the index. If no, the failure can distinguish a change in which neighbors the query seeks from an easily corrected geometric mismatch. Either is more informative than another correlation between nearest-document cosine and ANN recall.

The current E2 already establishes a quantization-independent gap: at unquantized `ef=16`, recovery@10 is 0.888/0.956/0.981 for Zero/Nano/Stella on FiQA and 0.824/0.910/0.931 on the 1M diagnostic. Thus quantization cannot be the sole explanation. E2's top-1 minus top-10 gap is also not the boundary that determines top-10 membership. The 10th/11th boundary and the surrounding score plateau are more relevant diagnostics.

The literature agent identified prior work on query-dependent HNSW difficulty and adaptive `ef` (Ada-ef, arXiv:2512.06636) and on intrinsic dimensionality/insertion order changing HNSW behavior (arXiv:2405.17813). Generic “queries have different ANN difficulty” is therefore insufficient novelty. The promising contribution is an intervention when substituting a query encoder over an unchanged graph, especially a cheap improvement measured against an unchanged final scoring target. These are literature leads supplied by the parallel literature review, not an exhaustive novelty clearance.

## Pilot: one collection, two original encoders, a few controlled directions

**Data and split.** Use only cached FiQA document vectors already named by E2 (`artifacts/demo-stella/fiqa/doc_vecs.npy`, with the registered document-ID alignment) and public FiQA query text. Deterministically select 192 queries by a seed or query-ID hash before looking at their results: 64 for any calibration, 128 for assessment. This is exploratory work on known-test material; the assessment split prevents within-pilot selection leakage but is not a new pristine benchmark. Do not touch the reserved four, M18 confirmation, or other protected data.

**Compute.** Reuse cached document vectors; encode the selected queries with existing Zero and Stella paths if query vectors are not already available. E2 does not commit query-vector caches, so do not assume they exist. Build one unquantized Qdrant collection and keep it unchanged. E2 deleted collections after its sweep, so a surviving populated graph must also not be assumed. Use its pinned settings/version where available, record the graph build, and do not rebuild per arm. Start with `ef` 32 and 64, candidate limit 20, returning the final 10 after rescoring. Both candidate limit and search setting must be equal across each paired comparison.

**First gate.** Verify exact top-10 parity between normalized cached-vector dot products and Qdrant exact search on a small sample. Reproduce a positive Zero/Stella unquantized recovery gap on the selected queries at at least one setting. If not reproduced, inspect the actual graph/search path and sample uncertainty; do not proceed to tell a geometry story from a missing baseline effect.

**Natural query arms.** For each query let `z` and `t` be its normalized Zero and Stella vectors. Query the graph with `z`, `t`, and the angular midpoint from `z` toward `t` (spherical interpolation). Record each arm's own exact neighbors as well as the original Zero exact neighbors. Stella navigation is an expensive diagnostic, not a proposed efficient method.

**Direction control.** At the midpoint's angular displacement from `z`, replace the query-specific `z→t` tangent direction with a direction from another query's paired residual, projected into the tangent plane at `z` and normalized. Use three pre-fixed derangements and average them. This preserves angular displacement and samples teacher/student residual directions without keeping their query-specific alignment. Report the control's achieved angles. An isotropic tangent direction may be an additional sanity check, but must not be the only control: its high-dimensional projections need not resemble language-derived directions.

This intervention controls text identity, document vectors, graph, search settings, candidate count, and perturbation magnitude. It does not by itself isolate an abstract “manifold distance”; do not name it that way.

## Two outcomes must be kept separate

1. **Own-query ANN behavior:** recover each arm's own exact top 10, including score regret under that arm. This measures whether the changed query is easier for the graph to answer, but it can change the retrieval task.
2. **Fixed Zero target, primary intervention outcome:** use the altered vector only to obtain 20 candidates, then score those candidates with the original `z`. Compare against the exact top 10 under `z`. The target and final ranking rule remain unchanged. Measure recall@10 and exact-score regret. If the correction merely seeks a different, easier neighborhood, this outcome exposes it.

Also report original-Zero score ties/near-ties at the cutoff. Losing an almost-tied neighbor can reduce ID recall without materially changing score or relevance. For the small assessment sample, report paired mean differences and query-resampling intervals; keep all arms, including failures. Public FiQA qrels can then give secondary nDCG@10 under the original query intent, without changing the primary geometry target or tuning on assessment labels.

For each query/arm, cheaply record top-1 cosine, `s10 - s11`, `s10 - s100`, score spread on a fixed random document sample, and overlap of the Zero/Stella exact top 20. Use these to describe where the intervention works. Nearest cosine is a proximity proxy, not a measured manifold distance. Stratification or regression over these quantities remains observational; avoid claiming that it proves mediation. Common Zero/Stella exact neighbors give a useful secondary target set, but report its size because conditioning on overlap selects easier pairs.

Time warmed searches in randomized arm order, with a few repeats on an idle machine. Encoding is outside this navigation diagnostic. Fixed `ef` is not equal actual computational work; report measured search latency and returned candidate counts, and distance/visited-node counts only if already exposed. Do not build an instrumentation framework for this pilot. A latency gain is necessary before calling an arm cheaper.

## Falsifiable interpretations and stopping rules

- **Midpoint improves its own recall but fails to recover Zero's exact neighbors:** the apparent benefit may come from changing the target neighborhood. Do not advertise restored navigability for the original representation.
- **Teacher-aligned movement improves the fixed Zero target more than equal-angle shuffled residuals:** direction matters beyond perturbation magnitude. This motivates testing a cheap approximation of that direction. It does not prove that nearest-document distance alone is causal.
- **Matched directions perform similarly, conditional patterns align with narrow cutoff gaps, and score regret stays tiny:** the main issue may be indistinct nearest neighbors rather than a special teacher/student mismatch. Report that result; do not launch a correction sweep to find a favorable story.
- **Neither intervention improves the fixed Zero target:** stop the correction branch. The useful result may be the irreducible extra graph effort for the tested index and query representation, with the cause still unresolved.
- **Only quantized arms show an intervention advantage later:** limit the finding to quantized navigation/candidate recovery. It would not explain the unquantized substitution penalty.

The pilot is deliberately small. Time-box implementation and FiQA execution to roughly half a day, and the first measured run to at most two hours. These are planning limits, not machine-specific runtime promises. Do not launch a full 1M sweep if the FiQA intervention has no repeatable directional result.

## Cheap correction, only after the diagnostic succeeds

Test one minimal family before anything elaborate: a mean-shift correction `q_nav = normalize(z + a b)`, where `b` is the mean paired Stella-minus-Zero vector on the 64 calibration queries. Include `a=0` and at most two positive strengths fixed using calibration data only. It has O(d) cost. If desired, compare a document-centroid direction as a mechanism control; do not assume movement toward a centroid moves toward a data manifold.

Use the correction for navigation and retain original Zero scoring. Count the candidate rescoring and correction time. An affine correction used as the final representation could be absorbed into the table under Appendix A, but that is a different outcome: it changes exact retrieval and must pass a separate relevance-quality check. Do not mix the two uses in one headline.

A practical gain would require reduced search latency at comparable recovery of original Zero neighbors, plus no demonstrated loss in the original-quality metric beyond a preselected tolerance. A useful pilot promotion bar is a consistent positive fixed-target recall change at both search settings, with a corresponding latency/recall advantage for the cheap arm after its overhead. Do not promote on lower nearest cosine distance or a single favorable nDCG fluctuation alone. Choose the exact operational tolerance before the calibration results are examined.

If promising, freeze the correction and replay only the winning arm plus Zero and matched-direction control on the cached 1M diagnostic, using a fixed small query sample. Replicate with a second graph build only if needed to determine whether the effect survives graph-construction variability. Add binary quantization last, distinguishing candidate navigation error from rescoring error and holding candidate budget fixed. Full-precision and quantized graphs are separate conditions; comparing independently selected fastest rows cannot identify a quantization mechanism.

## Alternatives and why they are secondary

**Learned cost-aware Zero/Nano routing:** potentially practical, but the current headroom is label-aware and post-search routing pays for a first search on every escalated query. A worthwhile study must jointly count first search, second encoding/search, feature cost, and realized traffic. It should beat a simple static tier/search-setting frontier at matched quality, not merely beat random routing. Fit a small policy on the development forums or use a clearly exploratory cross-domain split; preserve source/data-role restrictions. This is a reasonable next engineering result, but generic routing and adaptive search effort already have substantial prior art, and a modest learned gain alone is less explanatory than the navigation intervention.

**Cross-teacher recipe bridge:** directly relevant if the paper again claims its ridge screen predicts trained Zero. But a short undertrained table run would test another partial recipe, not validate the released recipe's converged ranking. Do not do an inexpensive inconclusive bridge merely to tick that box. Given the owner's “no full model retraining” boundary and the paper's already explicit scope, leave this behind the ANN pilot unless the intended paper claim changes.

## Access

This design uses the previously inspected `m15/PAPER.md`, `m15/MEASUREMENTS.md`, `m15src/e2_ann.py`, and the two E2 receipts. During this design pass I additionally opened `m15src/vectors15.py` and reread only aggregate geometry/unquantized rows from `results/m15_e2_ann_fiqa.json` and `results/m15_e2_ann_msmarco1m.json`. I did not open the proposed cached vectors, raw queries/qrels, any `work/` content, or protected material. Literature leads were provided by `/root/literature_check`. This review file is the only edit.

## Added owner-directed pilot: separate quantization from graph search

**Recommended order:** run a small offline exact-quantization decomposition first, then the navigation pilot if there is still a graph-specific question. It is cheaper, directly answers the owner's comparability concern, and can prevent treating a compressed-score effect as a navigability mechanism. This is not another best-settings sweep.

### Common comparison surface

Use the same 192-query FiQA sample, the same document IDs and cached Stella vectors, and all three aligned query encoders. Normalize the reference vectors consistently. “Original” here means the E2 document values: fp32 teacher compute was cached as fp16 and promoted for serving. It does not mean newly encoded fp32 documents or a lossless representation of the teacher output.

Fit each document quantizer once using only this document collection. Every encoder searches the **same codes, scale parameters, clipping rule, and index configuration** at a given precision. Keep original query vectors in fp32 for the primary document-compression comparison. Treat any query compression as a separate crossed factor, because otherwise the encoder effect and query-code distortion are entangled. Do not tune one encoder's quantizer separately and call that a same-index comparison.

The minimal precisions are the unchanged reference, scalar int8, and binary 1-bit. TurboQuant 4-bit can follow only if its exact scoring/codes are accessible and verified; do not approximate it with a different 4-bit quantizer and report that as Qdrant TurboQuant. Start with the native engine's actual quantizer implementation or exported codes if supported. If codes/exhaustive scoring cannot be accessed cheaply, an explicitly specified standalone scalar/sign quantizer is a useful mechanism experiment, but its findings do not decompose the existing native E2 result. Document this boundary before execution.

**Critical implementation check:** `exact=true` must not be assumed to mean exhaustive *quantized* scoring. Confirm which representation the native path actually uses on a tiny synthetic fixture whose original and compressed rankings intentionally differ. If it bypasses quantization, that call is only the original-score reference. Do not spend the full pilot on a mislabeled control.

### Five paths, with fixed candidate pools

For each query encoder and precision, compute the following. Use candidate sizes 20 and 100 for every encoder; the same original query vector and original document scores define final rescoring.

| Path | Candidate discovery | Final scoring | What it isolates |
|---|---|---|---|
| A | exhaustive original vectors | original | each encoder's own exact reference |
| B | exhaustive compressed-code scores | compressed | score/rank distortion from quantization, with no graph |
| C | exhaustive compressed-code top-C | original | quantization candidate omissions that rescoring cannot repair |
| D | HNSW compressed-score top-C | original | actual quantized navigation plus original rescoring |
| E | HNSW original-score top-C | original | graph-only control, with the same candidate budget |

The offline first stage needs A, B, and C only; no new HNSW run is necessary. Evaluate D and E at two fixed search efforts only after the exact effects are understood. Separate candidate size from `ef`; ensure the chosen settings actually permit the same C for all paths. Search effort below the candidate limit may be clamped by the engine and must not be described as a distinct treatment.

A→B measures compressed ranking distortion. A→C measures the quality/neighbor loss remaining after ideal exhaustive compressed candidate generation and original rescoring. C→D measures the additional consequence of approximate navigation at that precision and candidate budget. A→E gives the original-score graph loss. Comparing the graph effects under compressed and original scoring detects an interaction; do not assume those losses add independently. Some graph outcomes can accidentally improve judged nDCG even while missing exact neighbors, so report the actual signs.

Use each encoder's own original exact top 10 for recovery; also report absolute judged nDCG@10, changes from that encoder's exact nDCG, and original-score regret. Absolute scores compare available quality points; within-encoder losses compare preservation. A 1% relative loss threshold alone should not rank quantization robustness because exact baselines differ. Do not interpret loss against one encoder's exact neighbors as loss against another encoder's ground truth.

### Discriminating outcomes

- **Zero's extra loss is already large in B and remains in C:** compressed scores/candidate selection disproportionately hurt Zero. The graph is not needed to produce that part of the penalty. Test whether top-k boundary gaps or query compression explains the difference before proposing a geometry correction.
- **B is poor but C nearly reaches A for all tiers:** original rescoring repairs most rank error when candidate generation is exhaustive. The actionable budget is candidate oversampling; this does not yet establish that HNSW can find that candidate pool cheaply.
- **C is close to A but D falls much farther for Zero:** there is a navigation-specific penalty beyond exact quantization distortion. This is the cleanest reason to promote the fixed-Zero-target navigation intervention above.
- **E already shows the same encoder ordering and D adds similar extra loss to every tier:** quantization changes the operating point without explaining the substitution penalty. Study the original graph/query mismatch.
- **Keeping query scoring fp32 removes an apparent binary disadvantage:** the mechanism concerns query-code distortion or its interaction with document codes, rather than document compression alone. If the engine has only joint quantization, report that deployment constraint rather than silently substituting an asymmetric scoring path.

The potentially new result is a precise statement about *where a fixed-index query substitution loses its search budget*: compressed score distortion, candidate omission, graph navigation, or an interaction. A demonstrated cheap remedy tied to that location would be stronger than merely observing that one quantization setting is fastest. A negative decomposition is still valuable if it prevents a false geometric explanation.

**Budget and promotion:** the exact FiQA stage should be a compact extension around one cached matrix and at most two quantizers, not a new quantization library. Give native-code access/validation a short implementation cap (about two hours). If it requires deep backend surgery, either use a clearly labelled standalone mechanism diagnostic or defer that format. Do not expand to all precisions and the 1M corpus until one contrast is clear. Validate a promising decomposition on a fixed 1M sample using the same document quantizer and all three encoders; fit no encoder-specific corrections on that assessment sample. No benchmark or additional source access was performed in drafting this addition.
