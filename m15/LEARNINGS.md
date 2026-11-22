# Potential learnings from the full Constella research history

Updated 2026-10-01. **Editorial selection catalog, not the paper and not a new evaluation.**
The owner's goal is a paper a search engineer finds interesting, applicable, worth sharing, and useful in production. This catalog keeps the broader evidence available while the manuscript remains selective. Inclusion here does not mean a result is novel, confirmed, or worth a main-text slot.

Read this with [owner preferences](OWNER_PREFERENCES.md). Detailed audits: [early history](REVIEWS/2026-10-01-learning-audit-early.md), [middle history](REVIEWS/2026-10-01-learning-audit-middle.md), [late history](REVIEWS/2026-10-01-learning-audit-late.md). Those audits read named narrative records, not raw data or protected evaluations. M15's numerical core is mapped in [EVIDENCE.md](EVIDENCE.md). Before adding a historical number to the manuscript, verify its original receipt and method; a summary is an entry point, not independent replication.

## How to choose content

For each candidate ask: **What decision changes? What evidence changes it? How far does it travel?**
An engineering counterexample can justify a check without proving the problem is common. A useful capability can be explained without claiming architectural priority. An unsuccessful intervention belongs when its diagnosis saves a reader effort; a long list of failed arms does not.

Strength labels below distinguish exact algebra, controlled local experiments, repeated observations, fixed-artifact benchmarks, operational case studies, descriptive screens, and unresolved quality claims. “Controlled” does not imply independent replication or a causal architecture comparison. Query bootstraps condition on fixed models and do not cover training variation.

## Coverage and authoritative entry points

| Milestone | What is represented here | Entry points / coverage boundary |
|---|---|---|
| M0–M6 | Baselines, index costs, table construction, alignment failures, first ANN observations | [consolidated history](../research/m1-m6-findings.md); no separate M0 result claimed |
| M7 | Frozen Zero, teacher selection, surrogate failures, recipe variation, missed confirmatory bars | [findings](../m7/FINDINGS.md), [results](../m7/RESULTS.md), [explored](../m7/EXPLORED.md) |
| M8 | Negative improvement probes, objective saturation, fragmentation intervention, algebra and solver lessons | [findings](../m8/FINDINGS.md), [results](../m8/RESULTS.md), [explored](../m8/EXPLORED.md) |
| M9 | First Nano, head initialization, distribution-dependent retention, early memory/index timing | [findings](../m9/FINDINGS.md), [results](../m9/RESULTS.md), [edge cost](../m9/EDGE_COST_MAC.md), [explored](../m9/EXPLORED.md) |
| M10 | Data/width/backbone screens, generation sampling, practical build planning | [findings](../m10/FINDINGS.md), [results](../m10/RESULTS.md), [headroom](../m10/HEADROOM.md), [explored](../m10/EXPLORED.md) |
| M11 | Serving ports, tokenizer/prompt/device parity | [status](../m11/STATUS.md) |
| M12 | Fusion operator fairness, candidate-depth reversal, deployability and policy change | [findings](../m12/FINDINGS.md), [explored](../m12/EXPLORED.md) |
| M13 | Final Nano, M9 closeout, build accounting and release constraints | [findings](../m13/FINDINGS.md), [status](../m13/STATUS.md), [ship list](../m13/SHIP_LIST.md) |
| M14 | Published-artifact and real-loader parity | [findings](../m14/FINDINGS.md), [status](../m14/STATUS.md) |
| M15 | Real-query timing, same-index ANN, teacher screen, routers, blends, query-precision controls | [evidence](EVIDENCE.md), [methods](MEASUREMENTS.md), [follow-up](FOLLOWUP.md) |
| M16 | Proposed image/vertical/customer-training scope | [mandate](../instructions-m16.md): unscheduled ideas, **no experimental learning** |
| M17 | Vocabulary support and negative joint-training screen | [findings](../m17/FINDINGS.md), [status](../m17/STATUS.md) |
| M18 | Working project-memory retrieval, no improved encoder, leakage and endpoint lessons | [findings](../m18/FINDINGS.md), [status](../m18/STATUS.md): summaries only, raw confirmation stays closed |
| M19 | Deterministic term patch, exact inheritance, inconclusive judgment surface | [findings](../m19/FINDINGS.md), [status](../m19/STATUS.md): confirmation stays sealed |
| M20 | Broader quality results, domain reversals, token-based cost, allocation failure, archive limits | [findings](../m20/FINDINGS.md), [status](../m20/STATUS.md): published aggregates only for reserved four |
| M21 | Benchmark contract, dtype/normalization repair | [benchmarks](../m21/BENCHMARKS.md), [status](../m21/STATUS.md) |
| M22–M23 | Actual upstream caller checks, three-path dtype scope, release versus documentation state | [upstream audit](../m22/UPSTREAM_FASTEMBED.md), [project status](../PROJECT_STATUS.md); official-release planning is not a new retrieval experiment |

Historical renumbering is in [ROADMAP.md](../ROADMAP.md). Earlier documents retain their original names and interpretations. The catalog resolves conflicts using later measured corrections; it does not silently rewrite the history.

## A. Choosing and building compatible query encoders

### L01. An existing document index can support several query-compute budgets

- **Observation:** Stella, Nano, and Zero query identical Stella document vectors. BEIR-15 exact macro scores are .5614/.5081/.4572; warmed medium-query CPU medians are 31.6/2.25/.044 ms. Populated Qdrant collections were not rebuilt between encoder batches.
- **Decision:** If retaining an expensive document index matters, evaluate aligned query replacements before assuming a model change requires re-encoding documents. Preserve the original query path for comparison or selection.
- **Evidence:** M11–M14 releases, M20 aggregate; M15 cards C1/C2/C10 and E1/E2 receipts.
- **Strength/reach:** Demonstrated capability and multi-dataset fixed-artifact evaluation. Prior art already establishes query distillation and static/contextual switching. Not timed concurrent failover, loading, or an equal-quality speedup. Equal dimensionality alone is insufficient.
- **Use:** Main opening; explains why the paper exists.

### L02. The strongest teacher can yield a weak cheap query encoder

- **Observation:** An early eight-teacher table screen changed the teacher decision. M15 repeats one closed-form recipe across 26 checkpoints: teacher/table rank correlation +.09, versus development-table/public-table +.88. Registered gte-large teacher .5970 produces table .2455; Stella teacher .5745 produces table .3974.
- **Decision:** Before committing to a teacher/index for cheap queries, evaluate the intended cheap representation directly. For an existing index, the teacher is already fixed; test the student in that space.
- **Evidence:** M7/M8 audits; M15 C3/C3b/C13, E8/E8x/E13–E15.
- **Strength/reach:** Repeated within a particular ridge-table recipe; ten registered plus 16 exploratory checkpoints, related families, one shared WordPiece vocabulary. Each teacher uses its own document index. Screen does **not** select trained Zero/Nano, establish independence, or identify a mechanism. Clean-four rank agreement drops to .68 and winners vary by domain.
- **Use:** Main construction finding; strongest broad model comparison.

### L03. Vector fidelity and falling loss do not certify retrieval

- **Observation:** Early ridge settings improved cosine while worsening nDCG; an objective's loss fell .51→.13 while retrieval fell. M15 e5 tables match teacher vectors at cosine .90/.89 versus Stella .78 yet retrieve worse. M17's five trained arms all trail the same-surface untrained comparator despite lower losses.
- **Decision:** Select checkpoints and configurations on ranking against intended document vectors. Use loss/cosine for debugging rather than as promotion criteria.
- **Evidence:** M7/M8, M17; M15 C4 and E11/E11b.
- **Strength/reach:** Multiple concrete failures under different local recipes. Does not imply imitation losses are useless or that a particular alternative objective will improve retrieval.
- **Use:** Main, alongside L02; connects earlier research to the larger screen.

### L04. A cheap head fit can help before long transformer training

- **Observation:** M9 ridge warm-start versus random head improves SCREEN-3 +.02717 and DEV-6 +.02451 at the same SGD dose; extra preparation takes 60,000 forwards, 918,015 tokens, about 8.4 seconds. Frozen-backbone ridge alone scores .3463 on SCREEN-3.
- **Decision:** Try a bounded head initialization before paying for long fine-tuning of a new aligned transformer.
- **Evidence:** M9 results, middle audit §5.
- **Strength/reach:** Controlled local comparison, one seed/recipe, same SGD dose but unequal total compute. Not evidence for final converged benefit or table-screen-to-transformer transfer.
- **Use:** Construction supplement; needs original receipt check before paper numerics.

### L05. Query distribution and representation width are worthwhile diagnostic axes

- **Observation:** First Nano retains 93.8% on NQ-like questions but 71.0%/50.1% on physics/programmers. Later data/exposure changes gain +.0121 at 5M and +.0165 at 20M presentations; a 1152-versus-384 representation screen gains +.0217.
- **Decision:** Diagnose per-domain/query-form weakness and test targeted data or representation changes before repeating a weak recipe for more steps.
- **Evidence:** M9 findings; M10 findings/results, middle audit §§6–7.
- **Strength/reach:** Distribution-dependent artifact evaluation plus local component screens. Data arm changes distribution **and** exposure. M9 does not prove “coverage, not capacity”; width result does not prove a universal rank ceiling.
- **Use:** Supporting construction guidance, not explanation of the entire final Nano gain.

### L06. The successful rebuild does not isolate why it succeeded

- **Observation:** Final Nano exceeds M9 on every six-set dataset, +.1540 clean-four / +.1600 all-six descriptively. Recipes, dose, representation, and data all changed.
- **Decision:** Treat final Nano as feasibility evidence; do not copy one purported cause or infer a generally optimal recipe from the before/after gain.
- **Evidence:** M13 findings/status; M9 closeout.
- **Strength/reach:** Large fixed-artifact difference, not a matched causal ablation. The larger-teacher transformer arm was voided and cannot validate the later table selector.
- **Use:** Synthesis/background; leave chronology out of main paper.

### L07. Fit cost and complete build cost are different budgets

- **Observation:** Final Nano optimization: 57.27 A100 hours, 199,999,721 example presentations, approximately $95.27 at the historical rate. Prepared teacher targets, data generation, trials, export, and engineering are excluded. Zero's 20-minute retraining and 8–12-hour alternate-teacher target encoding are historical estimates. Ridge-screen intervals are roughly 4–7 minutes, with exclusions.
- **Decision:** Budget target preparation, data, fitting, evaluation, exports, and infrastructure separately. Reuse targets when possible; do not advertise a complete $95 reproduction.
- **Evidence:** M13 build receipts and M7 ledger, audited by M15 E15/C13.
- **Strength/reach:** Measured Nano optimization component; estimates/coarse file intervals for other components. No complete project bill or minimal-dose experiment.
- **Use:** Short main construction paragraph; supports approachability honestly.

### L08. Licensed contextual support matters more than an apparent vocabulary gap

- **Observation:** M17 selected 445 candidates from 210,447 fragments/terms; 199,763 failed minimum support of 20 distinct source documents and 50 contexts. A split token is not necessarily unknown. The old 35M cap applied to Nano, not Zero.
- **Decision:** Check contextual support and permitted training use before adding rows. Budget Zero by residency and latency, not Nano's parameter cap.
- **Evidence:** M17 findings/status and later correction; data-licensing records in source milestones.
- **Strength/reach:** Concrete preparation constraint, not proof that a supported row improves relevance. Public visibility is not a training licence; MS MARCO remained validation-only for our distillation.
- **Use:** Reproduction note or future vocabulary study.

### L09. Generation smoke tests must sample the real seed distribution

- **Observation:** Widened routing admitted health/finance seeds only 28%/38% on-topic. Later generation was on-form .50–.66 for off-subject seeds versus .83–.94 for on-subject seeds; favorable early tests used top-ranked seeds.
- **Decision:** Audit uniformly sampled production seeds, useful distinct examples, and repetition per query form before scaling generation.
- **Evidence:** M10 headroom/explored; middle audit §8.
- **Strength/reach:** Small judged preparation diagnostics, not causal final-retrieval gains. Filtering quality and sourcing more seeds are separate interventions.
- **Use:** Build supplement or separate data article.

## B. Spending query and retrieval compute

### L10. ANN work can consume an encoding speedup

- **Observation:** Early LightRetriever FiQA lookup queries lose more under low-effort HNSW than bge-small. Current Constella repeats increased Zero search needs on two workloads. On one shared 1M binary collection, ~62× Zero/Nano encoding difference becomes 2.2× including search at each tier's own 1%-loss target; uncompressed times are 3.61/3.85 ms.
- **Decision:** Retune and remeasure ANN after replacing a query encoder; compare encoding plus retrieval rather than infer request speed from encoder timing.
- **Evidence:** Early consolidated history; M15 C10/E2 and E15 shared-index restriction.
- **Strength/reach:** Repeated observations, headline within Stella family/two workloads. MS MARCO subset preserves positives and removes competitors. Not equal absolute quality, production QPS, or a universal lookup penalty. Earlier and later timings are distinct protocols.
- **Use:** Main numerical systems result.

### L11. Encoder compute and query-scoring precision are separate choices

- **Observation:** On fixed document sign codes, retaining query magnitudes raises Zero's original-top-ten coverage at 40 candidates .8217→.9297 (FiQA), .9010→.9677 (1M). Nano/Stella also improve. Extra absolute Zero/Nano gains have positive query intervals on both workloads.
- **Decision:** Check query precision when evaluating document compression. A cheap encoder still produces magnitude information worth testing in candidate scoring.
- **Evidence:** M15 C15/E18; prior query-aware binary scoring credited in RELATED_WORK.
- **Strength/reach:** Controlled exhaustive scoring, fixed vectors/codes and tie treatment, two workloads in one model family. Candidate coverage is not relevance, native scalar8, HNSW, or latency. Zero has more misses to repair and no greater proportional repair rate; mechanism unresolved.
- **Use:** Short main diagnostic or appendix; a native cost-qualified remedy would strengthen deployment advice.

### L12. The native quantization interaction is workload-sensitive

- **Observation:** At ef64, extra binary Zero/Nano neighbor-recovery penalty is 6.56 points on FiQA but .83 points on 1M, where the interval spans zero. Native rescoring can change candidates through segment merging even at oversampling 1.
- **Decision:** Replicate a striking setting interaction before treating it as an architecture property. Distinguish native request treatment from a fixed-global-pool rerank.
- **Evidence:** M15 C14/E16/E17, tagged-source semantics audit.
- **Strength/reach:** Same-graph controls within each workload, one graph/sample per workload. Weaker primary replication must accompany the positive scoring control; secondary settings do not replace it.
- **Use:** Limits beside L11, not a separate sweeping headline.

### L13. Memory residency, rescoring, and disk storage can dominate cheap queries

- **Observation:** Early synthetic edge runs turn a ~50× encoding gap into ~2× with search; under a 256 MB database-container limit uncompressed requests take hundreds of ms. Original vectors remain stored beside compressed codes. Some rescoring configurations are slower than uncompressed search.
- **Decision:** Price asset files, peak process memory, database residency, original-vector reads, cold load, and disk separately under intended constraints.
- **Evidence:** M9 edge results; current M15 C2/C10 supplies quality-qualified headline numbers.
- **Strength/reach:** Operational cost case without matching recall qualification. The container limit excludes encoder processes; it does not prove a complete system fits 256 MB or that quantization is universally required. Historical asset sizes are superseded.
- **Use:** Supporting explanation; avoid old synthetic timings in main table.

### L14. No transformer is not no cost, and dense lookup is not the only cheap-query design

- **Observation:** Zero .044 ms warmed encoding still needs a tokenizer, roughly 90.1 MiB assets, loading, and retrieval. Early inference-free sparse retrieval also competes at low query compute; symmetric static models perform differently and use their own indexes.
- **Decision:** Compare lexical and document-expansion approaches as well as dense query distillation when selecting a new system. State whether retaining an existing dense index is required.
- **Evidence:** Early baseline ladder; M15 C1/C2; prior art including pyNIFE/LightRetriever.
- **Strength/reach:** Cross-system descriptive evidence under differing protocols/indexes. No unique “zero latency” capability, broad small-model frontier, or current best-in-class claim.
- **Use:** Scope/related-work sentence; broad historical roster belongs in the catalog.

## C. Fusion, routing, and query-side changes

### L15. A lexical path can recover more quality than table-side tuning did

- **Observation:** M8's executed query-table levers mostly move <~.005; historical fusion adds ~.057. BEIR-15 Zero dense .4572→.4933 with DBSF@100; Nano .5081→.5110.
- **Decision:** Test lexical fusion early as a budget alternative to a larger encoder or more table training; include its sparse index, candidate search, and fusion costs.
- **Evidence:** M8; M12; M20 aggregates/M15 C1/C10. Original M7 convex fusion and later DBSF are distinct policies.
- **Strength/reach:** Descriptive route comparisons, not equal-cost or causal component rankings. Fusion sign varies by corpus: HotpotQA Zero .6127→.7015; MS MARCO Nano .4063→.3699. No “always fuse” rule.
- **Use:** Main practical alternative; preserve cost and negative cases.

### L16. Candidate depth can reverse the observed fusion-operator ordering

- **Observation:** Development DBSF minus fitted convex score fusion: +.0035 at 10, −.0020 at 50, −.0063 at 100, −.0146 at 1000. No stock operator matches the original depth-1000 target. Fairer RRF tuning recovers part of its original deficit.
- **Decision:** Choose fusion at the candidate budget and lexical implementation you will actually deploy. Do not transplant a deep-prefetch benchmark result into a shallow request.
- **Evidence:** M12 findings/tier receipts/depth check; middle audit §2.
- **Strength/reach:** Descriptive development curve. Small differences do not establish equivalence; operator treatments do not isolate score-magnitude causality. Stock DBSF recommendation was an implementability override after prior evaluation, not untouched confirmation. Self-ID exclusion and bm25s-lucene behavior are part of the recipe.
- **Use:** Main compact example; unusually applicable finding underused in v14.

### L17. Optimize the serving route, not an intermediate dense score

- **Observation:** M18 specialization's best dense dev gain +.007591 does not clear its fused endpoint gate: best fused +.002953 versus +.010. M18 ships a usable system with released Zero, not an improved encoder.
- **Decision:** Define the production route and important slices before tuning the query model; score candidates on that route.
- **Evidence:** M18 findings/status and M17–M19 postmortem; summary only.
- **Strength/reach:** Small internal model-judged surface; dense interval crosses zero. The endpoint discrepancy is a case, not a general effect size. Confirmation did not establish a new model win.
- **Use:** Short main lesson with L15; numerical case in supplement if independently traced.

### L18. Oracle complementarity is an opportunity, not a router

- **Observation:** Label-aware Zero/Nano oracle scores .5473 versus always-Nano .5161 across 12 nonreserved datasets. Selecting 15% of each dataset's queries for Nano matches always-Nano. Cheap query-only signals recover little; retrieval-margin routing recovers about 25% of oracle gain on six sets and requires another search for escalated queries.
- **Decision:** Evaluate a real routing policy against random/static choices at the same budget, including feature and repeated-search cost. Keep oracle claims separate.
- **Evidence:** M15 C5/C7/C8, E5/E6/E10.
- **Strength/reach:** Fixed-artifact upper bound and development-frozen router tests. Different training pools confound architectural complementarity; FEVER-family sensitivity moves the matching share. No demonstrated production router or end-to-end routing speedup.
- **Use:** Main short negative; detailed frontier remains companion evidence.

### L19. Compatible query vectors need not make a useful blend

- **Observation:** Development selects 30% Zero in Nano blend and improves .4247→.4295; six-set frozen evaluation lowers Nano −.0051, paired interval [−.0092,−.0011]. Stella selects no blend.
- **Decision:** Test cheap vector combinations outside their selection queries before deployment. Compatibility permits a blend but does not justify it.
- **Evidence:** M15 C9/E9; pre-freeze SciFact curve exposure disclosed in methods.
- **Strength/reach:** One selected blend and family, registered descriptive analysis with disclosed development exposure. No rejection of all ensembling.
- **Use:** Brief negative beside routing; appendix/details optional.

### L20. Fragmentation is correlated with error, but reducing it did not repair ranking

- **Observation:** Early slope is .050 nDCG per extra subword/word. Four matched equal-budget row screens reduce fragmentation .164–.176 but all lose retrieval; zeroing most additions barely moves score. Later vocabulary/joint-training screen also produces no survivor.
- **Decision:** Use correlation to propose an intervention, then test whether the intervention moves the endpoint. Inspect contextual support rather than assume more rows fixes unfamiliar jargon.
- **Evidence:** M8 findings/results/explored; M17 summaries; early audit.
- **Strength/reach:** Controlled negative for closed-form additions and specific trained screens. Jointly trained alternatives and architecture-wide claims remain open; subword count is neither rarity nor support.
- **Use:** Supporting failure diagnosis; avoid main causal claim.

### L21. A term-triggered patch can bound changes without establishing a relevance gain

- **Observation:** M19 deterministically adds 12 rows, 12×1,028 bytes; queries without roster terms inherit v1 bit-identically. Runtime changes are small in 10,000-query measurements. Quality remains inconclusive: unknown-label assignments span +.0167 to +.075 P@10 around a +.03 gate.
- **Decision:** Separate an exact operational blast-radius claim from a relevance claim. A bounded query-side patch can be operationally inspectable before sufficient judgments exist to promote it.
- **Evidence:** M19 findings/status/postmortem; sealed confirmation not opened.
- **Strength/reach:** Exact inheritance within the trigger implementation and measured local costs, unresolved relevance. Bare-term cosine and anecdotal examples do not certify improvement. Joint training, deterministic patching, and specialization are distinct avenues.
- **Use:** Interesting appendix/future case study; do not advertise a successful query patch.

## D. Evaluation that can change a production decision

### L22. Dataset averages hide reversals and cannot identify their cause

- **Observation:** Nano's registered six-set positives coexist with unresolved clean-four LEAF contrast and lower BEIR-15 macro than both selected dense targets. Zero sometimes beats Nano (including FEVER-family rows), but their pools differ: Zero uses FEVER training, Nano excludes it.
- **Decision:** Show domain rows and the original registered partitions. Choose deployment on important query types instead of assuming macro ordering applies everywhere.
- **Evidence:** M13/M20/M21; M15 C1/C11 and Appendix B.
- **Strength/reach:** Exact fixed-model benchmarks. No causal data/architecture attribution, equivalence from unresolved superiority, or broad superiority from a selected partition. Reserved aggregate receipt can be cited; raw protected data remains unread.
- **Use:** Required evaluation context, concise in paper.

### L23. Development reuse and recipe variance matter for small improvements

- **Observation:** M7's adopted gain is ~90% training-adjacent; proxy arm ordering reverses on the fuller suite. Step variation moves dev .0027–.0078. Records count hundreds of adaptive evaluations.
- **Decision:** Keep source-adjacent and independent slices visible; fix selection rules; replicate routine recipe choices when small gains drive promotion.
- **Evidence:** M7/M8 findings/explored; early audit §5–6.
- **Strength/reach:** Controlled local variation and descriptive selection history. Query bootstrap does not cover seeds/recipes. Does not invalidate a genuinely frozen subsequent test; does not establish hard negatives always hurt.
- **Use:** Evaluation supplement and limit on historical training claims.

### L24. Label resolution can be the limiting resource

- **Observation:** M18 query-bearing source objects appear in nearly every unfiltered top ten. M19's 60-query surface contains 93 unknown labels; one flip can move a term .0333, exceeding its .03 gate. High model-audit agreement does not resolve those unknowns.
- **Decision:** Audit provenance/source leakage, provide passage context, allow uncertainty, and calculate plausible label sensitivity before buying training compute for a small claimed gain.
- **Evidence:** M18/M19 findings/postmortem, late audit §§2–3.
- **Strength/reach:** Concrete internal evaluation failure, model-judged small samples. Not new human-label evidence or a general estimator. System usability and an improved-encoder claim are separate.
- **Use:** Production evaluation paragraph; highly transferable check, local effect size only.

### L25. Shuffle and prefix probes describe failure patterns, not production policies

- **Observation:** Shuffle sensitivity correlates .52 with teacher/Zero gap; 40% of queries carry ~90% net gap. Shared teacher scores and perturbation directions complicate causality. Dense tiers have descriptively similar relative retention on cut queries, but BM25 differs between word and character cuts.
- **Decision:** Use perturbations to find examples and hypotheses; test real incomplete queries or supported interventions before claiming typing performance or a mechanism.
- **Evidence:** M15 C6/C12, E4/E12/E12b/E13.
- **Strength/reach:** Bounded diagnostic with controls, not a causal word-order explanation or equivalence test. Selected FiQA examples are not representative.
- **Use:** Companion analysis; cut from main unless a stronger tested remedy emerges.

## E. Serving and build engineering

### L26. The compatibility contract includes the actual loader and device

- **Observation:** Missing prompt changes Stella vectors; padding-to-512 corrupts a table path; Nano's custom bridge passed while native registration initially used wrong pooling. A candidate fp16 document graph passes CPU parity but fails CUDA badly (minimum cosine .662); fp32 is shipped.
- **Decision:** Qualify tokenizer, special tokens, prompt, truncation/padding, pooling, normalization, export, and provider through the real published API. Include boundary lengths and batch-size changes.
- **Evidence:** M11 status, M14 findings/status; middle audit §§3–4; M15 parity records.
- **Strength/reach:** Strong operational counterexamples, not a defect-frequency estimate or indictment of all fp16. Historical repaired bugs must not be presented as current failures. Storage precision and execution arithmetic differ.
- **Use:** Main concise production lesson; one strong example is enough.

### L27. Dtype and normalization order affect the operational contract

- **Observation:** FastEmbed int64 mask×float32 pooling promotes to float64, doubling vector bytes. Naive pre-normalization narrowing can overflow float16 norms to zero vectors. M22 audit finds a third custom caller missed by the original two-path fix; all three must be covered.
- **Decision:** Check full values **and** dtype/bytes across actual callers and precision cases. Preserve safe accumulation and cast at the appropriate output boundary.
- **Evidence:** M21 benchmarks/status; M22 upstream audit and dtype measurement receipts.
- **Strength/reach:** Concrete bug/correction on recorded versions; numerical parity alone did not satisfy dtype promise. Upstream registration merge is not dtype-fix release. Do not generalize byte equality to unchanged rankings without a ranking test.
- **Use:** Supporting paragraph with L26 or separate implementation article.

### L28. Cost document encoding in tokens and realistic batch shapes

- **Observation:** M20 throughput varies 41–887 docs/s but only 12,920–15,774 tokens/s. Token-based 36.9–45.4-hour forecast contains Nano's 39.9-hour completion; document-count extrapolation suggests ~163 hours.
- **Decision:** Estimate builds from token distributions and representative batches, especially when corpora have different passage lengths.
- **Evidence:** M20 findings/status, late audit §5.
- **Strength/reach:** Strong operational case on one setup. Not universal throughput or the cost of the Constella training loop; this is evaluation corpus encoding.
- **Use:** Build-cost sidebar or separate practical article; original receipts required for paper table.

### L29. A changing batch shape can cause allocator trouble without an OOM

- **Observation:** Large length-sorted encode call reserves 23.06 GiB for 1.76 GiB live tensors and collapses throughput on a 10.24-GiB card. A reproduced shard with expandable allocation is byte-identical and bounds reservation to 4.40 GiB at 179 docs/s.
- **Decision:** Monitor reserved/live allocation and batch-shape changes, not just exceptions; reproduce the same shard before adopting an allocator remedy.
- **Evidence:** M20 findings/status; late audit §5.
- **Strength/reach:** Particular caching allocator/stack and one reproduction. No universal environment-variable prescription or GPU comparison.
- **Use:** Separate engineering note, low main-paper relevance.

### L30. Algebra and solver feasibility can save needless experiments

- **Observation:** Affine pooled transforms and positive per-token scaling can be absorbed into freely learned rows under stated normalized pooling. Dense 65K fp64 Gram would use 34.4 GB; block CG uses 4.42 GB. Zipfian counts defeat naive convergence while Jacobi preconditioning resolves it.
- **Decision:** Check representational equivalence before adding a transform; test solver distribution and memory before rejecting a candidate fit as infeasible.
- **Evidence:** M7/M8 findings/results; [absorption proof](PROOF_ABSORB.md), early audit supporting lessons.
- **Strength/reach:** Exact algebra under assumptions; local solver observation. Not every count-dependent pool/nonlinear map is absorbable, and algebraic capacity equivalence does not imply identical optimization. A failed solver is not model-quality evidence.
- **Use:** Supplement; not the reader-facing thesis.

### L31. A failed experiment design is not a negative model result

- **Observation:** M8's co-adaptation proxy was canceled because it could not decide the intended question. M12 ranking/fusion training plans were cut before execution. M8 uniform-bank target is nearly saturated (99.75% positive-first); a harder list has informative entropy but training is unrun.
- **Decision:** Define the action for each screen outcome. Diagnose whether targets and candidates can teach the model before extending a run. Keep unrun alternatives visibly open.
- **Evidence:** M8 explored/findings; M12 explored; early/middle audits.
- **Strength/reach:** Method/design diagnosis. No universal frozen-index ceiling, closed loss class, observed hard-candidate remedy, or rejection of full co-adaptation.
- **Use:** Research backlog and limits; not a list of alleged negative algorithms.

### L32. Model and system outcomes, publication state, and archive state are separate

- **Observation:** M17 produces no survivor; M18 produces a working system but no improved encoder; M19 is inconclusive with confirmation sealed. M20 measurements finish while object-storage archive remains pending; M23 registrations merge while dtype issue remains separate.
- **Decision:** State exactly what is demonstrated and shipped. Preserve failed/unresolved results and source identities; do not let working infrastructure become evidence of a relevance gain.
- **Evidence:** Project/roadmap and late audit; M22 upstream audit.
- **Strength/reach:** State/accounting facts at recorded dates, not a retrieval contribution. Historical result payloads are immutable.
- **Use:** Audit and availability record; mostly outside manuscript.

### L33. A post-hoc linear projection did not make an existing static encoder a good aligned query path

- **Observation:** Early potion-32M→arctic projection reaches .3280, below potion's own-index .3427, even with regularization selected on test retrieval. The potion-8M projection reaches .3036. A loader mismatch was corrected before the reported rerun.
- **Decision:** Test the inexpensive alignment shortcut before budgeting bespoke distillation, but evaluate retrieval in the target space rather than assume a linear map or matching width supplies useful compatibility.
- **Evidence:** Early consolidated history and early audit's supporting lessons; original rerun receipt required before manuscript numerics.
- **Strength/reach:** Bounded local negative for these static encoders, linear maps, and target space. Test-selected regularization makes this an optimistic diagnostic, not a prospective deployment policy. It does not rule out nonlinear or jointly trained alignment.
- **Use:** Construction supplement or alternative-path paragraph; added after Astra identified the omission.

### L34. Width hides the teacher-quality signal for closed-form students

- **Observation:** over 26 spaces, student quality ~ teacher quality alone has R2 0.12 (head) / 0.14
  (table); adding log2 width gives R2 0.70 / 0.48 with standardized betas +0.69 / −0.83 and
  +0.63 / −0.64, intervals excluding zero; teacher quality and width correlate +0.61. A fit on the
  ten registered spaces ranks the 16 later ones at 0.85 / 0.70 against 0.37 / 0.16 for teacher alone.
- **Decision:** when a cheap student must be built for an existing wide index, expect the width
  penalty and judge the teacher's quality within its width; when choosing an index, do not pick by
  teacher score; screen the student.
- **Evidence:** E21 (`results/m15_e21_width_model.json`), E19, E8/E8x.
- **Strength/reach:** 26 related checkpoints, three width levels, two closed-form recipes; a
  hypothesis about what cheap students pay for width, not a mechanism. E24 (pending) tests it
  prospectively. The dev screen still makes the better pick (regret 0.035 / 0.000 versus the model's
  0.249 / 0.107).
- **Use:** v17 RQ1, the central result.

### L35. The search-effort penalty for table queries recurs across teacher spaces

- **Observation:** on FiQA and TREC-COVID the table recovers fewer exact neighbours at ef=64 than the
  teacher in 25/25 spaces (mean −2.9 and −3.9 pp); SCIDOCS 17/25; median recovery multiplier 4x ef on
  the larger corpora; the relevance-loss multiplier agrees on FiQA and is mixed elsewhere; the head
  student shows the same deficit; top-1 cosine and margin deltas do not rank the gap.
- **Decision:** after replacing the query path over an unchanged graph, re-tune ef against the
  student's own exact targets; do not assume the index's settings transfer.
- **Evidence:** E20 (`results/m15_e20_ann_spaces.json`), E2.
- **Strength/reach:** one engine, one graph configuration, closed-form tables, three workloads that
  confound size with domain; latency not claimed. E22 and E23 pending.
- **Use:** v17 RQ2.

### L36. Reversing a query's word order defeats Zero by construction, and Stella mostly too

- **Observation:** On 24 hand-written direction-sensitive pairs (two authored passages per pair), Zero produced identical vectors for all 24 reversals and ranked the intended passage first in both directions on 0 pairs. Stella changed its vector every time but succeeded on 4/24 pairs (28/48 directions, chance 24/48), all unit conversions whose passages also differ in numbers; Nano 2/24.
- **Decision:** Treat word order as a weak signal for dense retrieval generally, not only for bag-of-token encoders; test direction-sensitive query classes explicitly before choosing an encoder for them.
- **Evidence:** C23, `results/m15_talk_word_order.json`, pairs frozen in git (`ba452d9`) before any run.
- **Strength/reach:** Small exploratory probe; only two passages compete, not a corpus. Not evidence that dense models ignore order in general.
- **Use:** Talk; possible limitation paragraph next to Zero's pooling.

### L37. Cloud CPUs widen the encoding-cost gap that laptop timings show

- **Observation:** One ONNX Runtime thread on a GCP c4-standard-8 (Xeon Platinum 8581C) encoded a Stella query in about 271 ms versus about 49 ms on the M5 Pro; Nano 7.3 vs 2.3 ms; Zero 0.059 vs 0.029 ms. Saturating the 8-vCPU VM gave 18.1, 464 and 67,423 queries/s, i.e. about $6.07, $0.24 and $0.0016 per million queries of encoding at $0.395/h.
- **Decision:** Price query encoders on the hardware that will serve them; Apple-silicon latency understates the transformer cost on cloud CPUs.
- **Evidence:** C25, `results/m15_talk_throughput_nq_gcp-c4-standard-8_encode.json`.
- **Strength/reach:** One instance type, encoding only, batch 1, 30 s per point. Excludes graph search, batching and GPU serving.
- **Use:** Talk cost slide; candidate deployment-cost paragraph.

### L38. Count FLOPs from the actual block, not a generic transformer formula

- **Observation:** A first per-query estimate assumed a plain 2·d·ff feed-forward block and gave 18.8 GFLOPs for Stella; Stella's block is gated (3·d·ff weights), which gives 25.1 GFLOPs, and Nano's linear head runs per token (0.54, not 0.51 GFLOPs). An independent review caught it from the model source.
- **Decision:** Derive analytic cost from the loaded module's code and record the formula in the receipt.
- **Evidence:** C24, `results/m15_talk_flops.json`; review `m15/talk/planning/astra_deck2.md`.
- **Strength/reach:** Analytic estimate; ignores memory traffic, which dominates Zero.
- **Use:** Talk; reproducibility note.

### L39. Filling a WSL GPU spills silently into shared memory

- **Observation:** Stella document encoding on the RTX 3080 (10 GB) under WSL ran shard 0 (50k NQ passages) in 13 min at 65 passages/s, then shard 1 in 58 min with GPU memory at 9.9 GB and utilization still at 100%. No OOM was raised. Halving the length-bucketed batch from 16,384 to 8,192 tokens kept memory near 3.3 GB and cut shard time to about 6.2 min (about 135 passages/s), twice as fast as shard 0, so the larger batch was already degrading shard 0.
- **Decision:** On WSL, cap batch tokens well below the card's memory and compare per-shard time across batch sizes; a larger batch can be slower without any error, and 100% utilization does not show it.
- **Evidence:** `m15src/talk_encode_nq.py` comment and box logs `~/constella-talk/encode_nq.part1.log`; related L29.
- **Strength/reach:** One machine and driver stack; diagnosis by memory reading and timing, not a driver trace.
- **Use:** Engineering note; not paper material.

### L40. A Qdrant Edge shard is single-process, and it can hold the query encoder

- **Observation:** A second process opening the same Edge shard fails on its write-ahead-log lock (`WouldBlock`), so concurrent benchmarks share one shard across threads. Zero's 30,522 rows stored in their own Edge shard reproduce the released encoder exactly (max difference 0 on 500 queries) at about 0.25-0.28 ms more per query than the in-memory NumPy table.
- **Decision:** Benchmark Edge with in-process threads; keep the NumPy table for latency-critical serving, and use the shard form only where shipping one storage artifact matters.
- **Evidence:** C26 (`zero-edge` row), `m15src/talk_edge.py`, `m15src/talk_throughput.py`.
- **Strength/reach:** qdrant-edge-py 0.8.0 on macOS; one corpus for the latency delta.
- **Use:** Talk backup slide; Edge deployment note.

### L41. On a small corpus, binary quantization can keep Zero just short of its own exact score

- **Observation:** On FiQA (57,638 passages) in Qdrant Edge with binary quantization and fp16 rescoring, Zero's best tested setting (ef 1024, oversampling 8) reached ANN nDCG@10 0.3674 against its exact 0.3728 (98.6%), while recovering 0.907 of its own exact top 10; Stella and Nano met the 99% target at ef 128/os 2 and ef 256/os 4.
- **Decision:** Include oversampling in the search sweep for replacement encoders, and report when an encoder misses its own exact target rather than silently taking its best point.
- **Evidence:** C26, `results/m15_talk_edge_fiqa.json`; related L10 and L12.
- **Strength/reach:** One small corpus, one graph build, sequential timing on a laptop.
- **Use:** Talk; supports the graph-effort section.

### L42. A desktop GPU re-encodes a 2.7M-passage index in about five hours, if the batch fits in VRAM

- **Observation:** Stella's document path encoded BEIR NQ (2,681,468 passages) at a steady 141 passages/s on one RTX 3080 (median shard 353 s per 50,000, range 333-375), so N·c_D = 5.3 GPU-hours. At batch_tokens 16,384 the 10 GB card spilled into WSL shared memory and one shard took 58 min instead of about 6. Box (CUDA) and Mac (MPS) vectors agree to min cosine 0.99994 on 256 seeded passages. A resumed job's printed rate counted already-finished shards (it showed 316/s); shard file times gave the true rate.
- **Decision:** Quote 5.3 GPU-hours as the one-time index cost in the talk; size batches below VRAM under WSL; compute rates from shard times, and print rates over the current run only (fixed in `talk_encode_nq.py`).
- **Evidence:** `results/m15_talk_nq_parity.json`, MEASUREMENTS T6, `m15/LOG.md` 2026-11-21.
- **Strength/reach:** One GPU, one corpus, fp32 compute; fp16 or bf16 compute would be faster but changes the registered path.
- **Use:** Talk cost slide; paper deployment-cost section.

### L43. On 2.7M passages, a 51x faster encoder buys a 2.4x faster request, and the graph tax grows

- **Observation:** On BEIR NQ (2,681,468 passages, Qdrant Edge, binary quantization with fp16 rescoring, M5 Pro, sequential), median encode times are Stella 20.5 ms, Nano 1.63 ms and Zero 0.032 ms (Nano/Zero 51x). At each encoder's selected setting, median request totals are Stella 20.8 ms (ef 256, os 4, nDCG@10 0.6309 of exact 0.6371), Nano 2.04 ms (ef 512, os 4, 0.5790 of 0.5838) and Zero 0.86 ms at its best tested point (ef 1024, os 8, 0.5097 of 0.5173, 98.5%, missing its 99% target; own top-10 recovery 0.951). Search becomes 87% of Zero's request. Nano at ef 64, os 4 (1.74 ms, 0.5542) beats Zero's best quality at twice its time. BM25 in the same shard: 0.3012, 1.12 ms.
- **Decision:** Present the encoder as one variable in an (encoder, ef, oversampling) choice solved against a quality floor, not as a speedup. Report Zero's miss on both FiQA and NQ.
- **Evidence:** C26 (NQ), `results/m15_talk_edge_nq.json`; related L10, L12, L41.
- **Strength/reach:** One corpus, one graph build, sequential laptop timing; quality numbers are exact-match validated against M20.
- **Use:** Talk bet, race, split, recovery and recipe slides; paper deployment section.

### L44. Reversing a query barely moves any of the three query vectors

- **Observation:** Over 24 reversal pairs, the median cosine between a query and its reversal is 0.9964 for Stella (4.9 degrees), 0.9991 for Nano (2.5 degrees) and exactly 1 for Zero. On "cheap flights paris to rome", Zero's vector has cosine 0.878 with Stella's and Nano's has 0.963.
- **Decision:** Present Zero's word-order blindness as structural (identical vectors) and small relative to the teacher's own sensitivity, not as a large ranking loss; keep T1's 4 of 24 as context.
- **Evidence:** `results/m15_talk_slide_data.json`, MEASUREMENTS T7; related L36.
- **Strength/reach:** 24 hand-written pairs and one example query; vector geometry, not retrieval quality.
- **Use:** Talk order and lookup slides.

### L45. On a cloud VM, the whole request is 27-29x cheaper, not 3,700x, and Zero ties Nano

- **Observation:** With search included (NQ, Qdrant Edge, each encoder's selected setting, one process, best of 1-8 threads on a c4-standard-8 at $0.395/h, warm page cache), Stella served 22.4 queries/s ($4.91 per million), Nano 605 ($0.18) and Zero 653 ($0.17). Encoding alone had separated them 26x and 3,700x. Zero reached 640 queries/s on one thread and did not scale with more, and Nano topped out at the same level: the single process running the Edge search sets the ceiling. With a cold page cache, rescoring read fp16 vectors from the network disk: Zero dropped to 42 queries/s on one thread (23 ms median) and needed eight threads to reach 628.
- **Decision:** Quote end-to-end cost next to encode-only cost; warm (or RAM-resident) rescoring vectors before serving; treat the one-process ceiling as a property of this setup, and scale out with more processes or shards rather than a cheaper encoder.
- **Evidence:** `results/m15_talk_throughput_nq_gcp-c4-standard-8-warm_e2e.json`, `..._cold_e2e.json`, C29; related C25, L43.
- **Strength/reach:** One VM type, one process with threads, 30 s per point; other deployments (several processes, Qdrant server, GPU encoders) can move the ceiling.
- **Use:** Talk cost slide; paper deployment section.

### L46. A router that never drops a query needs the search width as a knob, and the process sets the ceiling

- **Observation:** In the workload studio (NQ, one Edge shard, 14 worker threads in one process, M5 Pro), capacities measured one encoder at a time overstate a Nano and Zero mix, which contends for the Python interpreter lock. An earlier version held the latency target by shedding queries past a deadline; the owner ruled that out (a service cannot discard queries). With every query served, letting Auto choose one shared search preset as well as the encoder shares gives a visible, graceful trade: 40/s Stella at ef 1024; 600/s Nano; 1,500/s Nano and Zero at ef 256 (0 of 20 seconds over a 150 ms target in four runs); 2,000/s at ef 128 (0-4 of 20). Beyond about 2,600/s the process itself is the ceiling for every mix, so lowering the floor does not help; the queue grows and drains once traffic falls. Three effects needed handling: comparing the live search time of the setting in use with offline times of the others made Auto flap between presets (fixed with one live slowdown factor for all settings and a 20% margin before widening); an encoder's time under load varies so much that one fast reading readmitted a slow one (fixed with a 5 s hold, and with probes that wait 2 s after a traffic change so the old traffic does not skew them); and a transient queue folded into the latency budget marked every encoder slow (fixed by comparing an encoder's own time with the target alone).
- **Decision:** Serve every query; plan shares and search width together; keep the floor ahead of latency; treat the process ceiling as a deployment property and scale out with processes or shards.
- **Evidence:** `m15/talk/demo/studio/`; rehearsal runs in the session scratchpad. Demo observations, not measurements.
- **Strength/reach:** One laptop, one process, simulated traffic; the mechanisms carry over, the numbers will not.
- **Use:** Live demo; talk Q&A on routing.

### L47. Among 25 open retrievers, Nano scores like a base model at small-model compute, and Zero clears MiniLM with no network

- **Observation:** On six BEIR datasets (exact search, each tower on its own index), Nano on Stella's index scores 0.532 at 0.54 GFLOPs per query; the base-size towers score 0.467-0.533 at 2.0-3.4 GFLOPs and the small ones 0.454-0.504 at 0.51-0.86. Zero scores 0.434 at 28 KFLOPs, above all four MiniLM models (0.328-0.422, 0.26-0.51 GFLOPs) and BM25 (0.417). Stella saw FiQA and ArguAna in training: on the other four datasets Zero (0.410) trails BM25 (0.441).
- **Decision:** Show the 25-model field on the talk's quality slide with the exposure caveat in the footnote and the notes; keep BEIR-15 (L-registered rows) as the paper's headline.
- **Evidence:** C30, MEASUREMENTS T8, `results/m15_talk_flops_field.json`.
- **Strength/reach:** Analytic FLOPs, not measured time; six datasets with teacher exposure on two; descriptive means without intervals.
- **Use:** Talk slide 4.

### L48. On NQ, query encoding passes the whole corpus encode after about 10 million Stella queries

- **Observation:** A Stella query (31 tokens with its prompt) costs about 2.5e10 FLOPs; the 2.68M NQ passages (mean 109 tokens) cost 2.4e17 once. Query encoding equals the corpus encode after 9.6M queries, about 27 hours at 100 queries/s. Nano needs 448M queries to do the same, and Zero about 9 trillion.
- **Decision:** Open the talk's problem slide with this crossing instead of a cost formula.
- **Evidence:** C31, MEASUREMENTS T9, `results/m15_talk_crossover.json`.
- **Strength/reach:** Analytic FLOPs, not dollars or time; batching makes passage encoding cheaper per FLOP than single queries, so the measured crossing would come sooner.
- **Use:** Talk slide 2.

## Current editorial selection (agent recommendation, not owner acceptance)

**Primary question:** When retaining a document index, how should an engineer choose and validate a cheaper query path? **Separate construction question:** before committing to a teacher and index, how should an engineer screen the cheap representation? Teacher choice changes the index; it is not a runtime option over an already fixed one.

| Reader decision | Candidates worth foregrounding | Why | What stays outside the main narrative |
|---|---|---|---|
| Keep the index and choose a query budget | L01, L07, L14 | Concrete capability, quality/cost baseline, honest build scale | Long release chronology and historical unmatched frontiers |
| Build/select on retrieval, before choosing a new teacher/index | L02, L03; supporting L33 | 26-checkpoint screen plus repeated surrogate failures | All loss arms, speculative geometric explanations |
| Retune candidate search | L10; bounded L11/L12 | Encoding saving can move into ANN work; controlled precision diagnostic | Full quantizer grid and unmeasured native remedy |
| Spend remaining budget | L15–L19 | Lexical alternative, depth reversal, weak routers and blend | Oracle as deployable policy; every threshold row |
| Approve actual deployment | L24, L26/L27, L22/L23 | Actual caller/device and judgment failures support specific checks | Generic checklists and source-access transaction machinery |

L04/L05/L08/L09 offer a construction-focused alternative paper. L21 offers a narrowly scoped patch case study, but relevance evidence is not ready. L28/L29 would make a useful engineering article rather than strengthen this paper's central research claim. Do not discard these merely because they are not in the selected manuscript.

## Work that could materially strengthen a claim

1. **Native precision remedy:** measure supported magnitude-preserving query precision, quality/recovery and total cost together, accounting for collection rebuild/graph variation. Strengthens L11's production applicability; no full retraining needed.
2. **Contrasting teacher space:** repeat a small fixed-code scoring control for another teacher plus fitted table. Tests reach beyond Stella; one added space remains a stress test, not a universal result.
3. **Representative judgments for bounded patches:** resolve M19-style unknowns with contextual judgments on a fresh surface. Strengthens a patching story more than another unjudgeable fine-tune.
4. **Cost-aware routing only with a measurable target:** beat static/random routing at the same measured request budget, including repeat searches. Oracle headroom alone does not justify a larger routing project.

Do not launch these automatically. First select the stronger claim they would support. More checkpoints, queries from the same family, or quantizer settings can add precision and bulk without closing the important generalization or deployment gap.
