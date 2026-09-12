# Constella in Plain English

*Asymmetric dual encoders, the research record from 24 August to 10 September 2026, written for a reader who knows the domain but did not watch the work. The interactive version of this page is a Claude artifact; this file is the GitHub-renderable twin. Numbers are copied from the result JSONs and the milestone findings files, which remain authoritative.*

One document index, built once with a good 400M-parameter model. Two cheap ways to ask it questions: a lookup table that costs almost nothing per query, and a 35M-parameter transformer that costs about what a small embedding model costs. This page explains what we tested, what came out, what we are building next and why the process looks the way it does.

| Where things stand | | |
|---|---:|---|
| Zero, dense only | **0.4339** | avg-6 nDCG@10, 0.024 below LightRetriever's dense table. Reported as measured; published and considered releasable. |
| Zero fused with BM25 | **0.4911** | Statistically ties the best inference-free sparse system (0.4868), at a table lookup's query cost. |
| Nano, first attempt | **82.2%** | of the teacher on the screen surface. 93.8% on Wikipedia questions, 50.1% on programming forum questions. Not released. |
| Nano, real build | **Ready** | Code complete and tested. Waiting on a cloud GPU. Budget ceiling $1,000. |

---

## Part one: the idea

Dense retrieval works by turning every document into a vector once, storing those vectors in an index, and then turning each incoming query into a vector in the same space and finding the nearest documents. Normally the same model does both jobs. That is wasteful on the query side: documents are encoded once, offline, on whatever hardware you like, but queries are encoded on every request, often on a laptop, a phone, or a small server that is also doing other things.

The bet behind this project is that the two jobs can be split. Keep a strong, expensive model for the documents, and swap in something much cheaper for the queries, trained to land in the same vector space. If it works, the expensive part is paid once and the cheap part runs everywhere. The index never has to be rebuilt when you change the query encoder, because the document side is frozen.

```mermaid
flowchart LR
    subgraph doc["Document side, frozen"]
        S["stella_en_400M_v5<br/>400M params, 1024-d<br/>run once, offline"] --> IDX["One shared index<br/>Qdrant collection, 1024-d<br/>never rebuilt"]
    end
    subgraph q["Query side, swappable"]
        Z["Zero: token-vector lookup table<br/>no transformer, ~0.02 ms/query<br/>shipped"]
        N["Nano: 35M transformer<br/>bge-small backbone, distilled<br/>being built"]
    end
    Z -- "query vectors, same space" --> IDX
    N -- "query vectors, same space" --> IDX
```

**Zero** is the extreme version. Every token in the vocabulary gets one fixed vector, computed once by pushing that token through the teacher. A query is encoded by looking up its tokens and pooling the rows. There is no neural network at query time at all, so the query cost is a few table lookups and an addition. The catch is that a bag of token vectors cannot know word order or context, so it should lose quality. The question was how much.

**Nano** is the moderate version. A small transformer, capped at 35 million parameters, reads the whole query and is trained by regression to reproduce the teacher's query vector. It costs roughly what a small symmetric embedding model like bge-small costs, so it has to earn its place on quality and on the fact that it shares stella's index instead of needing its own.

The whole thing is also a paper. The repository is deliberately kept as an evidence base: negative results, failed approaches, provenance and the limits of every claim are recorded next to the successes, because a measured miss is a publishable result and an unmeasured claim is not.

### Where the pair pays off

The pattern is worth the trouble wherever the query side runs somewhere the document side cannot, or runs so often that its cost is the bill.

- **Command-line tools and scripts.** Search a large corpus from a CLI without downloading a model or installing torch. Zero's runtime is numpy and a tokenizer, about 31 MB as an int8 graph; the first query answers in milliseconds instead of after a 1.3 second model load.
- **Edge and on-device search.** Phones, kiosks, embedded boxes, Qdrant Edge. The query side is a table lookup, so no accelerator, no warm model and no thermal budget. A one-million-document index serves inside a 256 MB container once binary-quantised, at 3.4 ms for zero and 4.5 ms for nano.
- **Frozen and long-lived collections.** Encode documents once with the strong model and never re-embed. The query side can be swapped or upgraded, from zero to nano to whatever comes next, without touching the index, because every student targets the same space.
- **Encode at ingest in the cloud, query anywhere.** Documents pass through the 400M model where GPUs live, once, at ingest. Queries are encoded on whatever the client has: a browser, a serverless function with a tight cold start, a laptop on a train.
- **High-QPS or cost-sensitive query paths.** At tens of thousands of queries per second the query encoder is the cost centre. A lookup costs about 0.02 ms and no GPU; a 33M transformer costs about 5 ms of CPU. Both are orders of magnitude cheaper than running the 400M model per query.
- **Offline and private by construction.** No model server, no network call to embed a query. Search works offline against a synced index, and the query text never has to leave the device to become a vector.
- **Hybrid out of the box.** Zero fused with BM25 through Qdrant's DBSF ties the best inference-free sparse system while the query side stays a table plus token counts. The lexical channel is part of the recipe, not an add-on.
- **One index, two speeds.** Because zero and nano share stella's index, a product can route simple queries to the table and harder ones to the small transformer, and move that boundary later without a reindex.

## Part two: how we measure, and why it is strict

Retrieval quality is scored with nDCG@10: for each query, how well the top ten returned documents match the human relevance labels, averaged over the queries of a dataset, then averaged over datasets with equal weight. A difference of 0.01 is large in this world. Our screening resolution is about 0.005.

| Surface | Role |
|---|---|
| **The six** (SciFact, NFCorpus, FiQA-2018, ArguAna, SciDocs, TREC-COVID) | Confirmatory. Standard BEIR sets with published numbers for every competitor. Each system gets exactly one scored access, spent only after its recipe is frozen. Zero spent its access in M7. Nano's is unspent. |
| **Clean-4** (NFCorpus, SciDocs, SciFact, TREC-COVID) | The headline. stella discloses ArguAna and FiQA in its own training data, so any student may inherit an advantage there. Fixed before any nano number existed; all six reported beside it. |
| **Reserved four** (FEVER, DBpedia-entity, CQADupStack android and english) | Descriptive. Never opened during development. Read only if nano clears its first release test, never to decide anything. |
| **LoTTE-clean** (seven StackExchange forum slices, 14,034 queries) | The one fresh out-of-domain surface. Unread so far. Exactly two reads: one before the expensive build, one audit before the freeze. |

Development uses other surfaces. DEV-6 is six components pinned since M7, including two CQADupStack forums and slices of Natural Questions and HotpotQA. COV, the coverage surface built in M10, is four families of consumer-health, scientific, legal and finance questions chosen to look nothing like the training data. Every development read is counted: 494 in-training evaluations by the end of M8, and hundreds more since. The count is published because the alternative, saying we were careful, is not checkable.

Two statistical habits run through everything. Every comparison gets a paired bootstrap confidence interval over queries, so a difference smaller than the noise is called a tie, not a win. And decisions are registered before their data is seen: the rule, the bar, the sequence and the constants are committed to git and dated, then the number is produced. Changing a rule after seeing the number is the one thing the process is built to prevent.

## Part three: what we did, in order

### M1 to M6, 24 to 25 August: survey and baselines

Before building anything, we measured what already exists. We reproduced LightRetriever, an academic system that also uses a lookup table for queries, and got its dense numbers to match the paper after finding that its tables must include a beginning-of-sequence token. We measured MongoDB's LEAF pair, which uses a small query model against a larger document model, and confirmed byte-for-byte that we had composed it correctly. We measured small symmetric models such as bge-small, static embedding models used symmetrically, BM25, and OpenSearch's inference-free sparse encoder.

- Small transformers on the query side scored around 0.50 to 0.53 on the six. Zero-compute systems scored 0.43 to 0.49. Symmetric static models came decisively last at 0.32 to 0.36.
- A tempting shortcut failed cleanly: fitting a linear map from a static model into a contextual document space, even with test-set-tuned regularisation, scored below the static model used on its own.
- Costs are not one number. A lookup query costs about 0.02 ms; a 33M transformer about 5 ms on CPU. But the lookup table is hundreds of megabytes while the small model is 66 MB, and a 1024-d index is four times the size of a 384-d one per document.

**What it changed.** The comparison set, the six datasets, the bootstrap habit and the cost framing were fixed here. An external review called the results not decision-grade and listed seven defects; every one was rerun or reworded before anything else started.

### M7, 25 to 28 August: zero, the lookup table

Pick a teacher, distil its query behaviour into a table of token vectors, tune the recipe on development data, freeze it, then spend the single confirmatory access to the six.

**Choosing the teacher taught the project's first big lesson.** The obvious way to pick a teacher is by how well it retrieves. We tried that, chose Snowflake's arctic-embed-l, and withdrew the choice the same day: ranked by the table distilled from it, arctic was 0.048 worse than the model we already had. Across eight candidates, a teacher's own retrieval quality had zero correlation with how good its distilled table was. Only stella_en_400M_v5 beat the incumbent, by 0.037.

**Stella is the best teacher we tested, and the reason is structural.** A lookup table can only reproduce the part of a query vector that adds up from the query's tokens. Every other candidate carries a much larger share of the meaning in context, in how the tokens modify one another inside the transformer, and a table cannot see that. Measured as the fraction of a teacher's own retrieval score that its best closed-form table recovers on the two development forums (`results/m7_learnability_report.json`):

| Teacher | Teacher's own score | Its best table | Recovered |
|---|---:|---:|---:|
| stella_en_400M_v5 | 0.481 | 0.344 | **72%** |
| bge-base-en-v1.5, the incumbent | 0.448 | 0.307 | 69% |
| e5-base-v2 | 0.393 | 0.265 | 67% |
| bge-large-en-v1.5 | 0.449 | 0.275 | 61% |
| mxbai-embed-large-v1 | 0.443 | 0.251 | 57% |
| arctic-embed-l | 0.493 | 0.259 | 53% |
| gte-large-en-v1.5 | 0.471 | 0.203 | 43% |

Two corollaries fell out of the same probe. Within every family the base model out-distils the large one, by 0.04 to 0.07, because the large models push more of the meaning into context. And cosine agreement with the teacher's query vector is the wrong metric: e5-base agrees most closely with its teacher, at 0.91, and ranks sixth on retrieval. Imitating a vector is not reproducing a ranking.

We also did the algebra before spending GPU time. Query-side centering, whitening, per-token weights and similar tricks are all absorbable into a freely trained table, so they cannot raise its ceiling. Only new rows or multiplicity-aware pooling could.

| Comparison on the six | Difference | 95% interval | Verdict |
|---|---:|---:|---|
| Zero int8 table vs LightRetriever dense table 0.4583 | −0.0243 | [−0.0405, −0.0086] | below, resolved |
| Zero vs BM25 0.4174 | +0.0165 | [+0.0017, +0.0311] | not resolved under multiplicity |
| Zero fused with BM25 vs OpenSearch 0.4868 | +0.0043 | [−0.0063, +0.0151] | statistical tie |

The macros: zero dense 0.4339, fused 0.4911, the teacher itself 0.5744. Zero retains 75.5% of its teacher on the six. On the development set it had looked like 91.5%, but the out-of-domain part of the development set had said 76.4%, and that was the honest forecast.

**What it changed.** Zero's dense-only score sits below LightRetriever's dense table on the six, and we report that as measured. Fused with BM25 it is a genuinely good system, and it is a query encoder with no neural network in it at all, which is the point: it was published in M11 and we now consider it releasable as a product component. The dev-versus-final gap became a standing rule: report an out-of-domain subset next to every macro. Excluding MS MARCO for licence reasons costs about +0.006, not resolved, so the gap is architectural, not a licensing artefact.

### M8, 28 to 30 August: twelve probes to improve the table

With the document tower frozen, what could close a 0.024 gap? Twelve pre-registered probes on development data, most in closed form, so a negative is a ceiling and not a bad run.

- More data: four times the dose bought +0.001. Closing the gap would need roughly eighteen times the pool.
- The training objective was inert: the table already ranked the correct document first for 99.75% of training queries. Future table training should start from hard candidates.
- Finer vocabulary rows, n-gram rows, pseudo-relevance feedback, aiming at the document manifold, a head on the finished document vector: all negative.
- Subword fragmentation correlates with the gap, but changing it did not move the metric. A correlated channel is not a lever.

**What it changed.** No lever improved the development endpoint by more than about 0.005, and fusion with a lexical channel was worth ten times that. Ship zero fused; treat the query side's capacity as the real limit. That made the case for nano.

### M9, 30 August to 1 September: nano, first attempt

Take bge-small's 33M-parameter backbone, put a linear head on it, and train it by squared-error regression to reproduce stella's query vectors, on about 463,000 real queries from Wikipedia question answering and product search. Release bar: retain at least 87.8% of the teacher on the six and beat bge-small and LEAF. The run consumed 3.74 billion tokens and went flat.

| Screen component | Teacher | Nano | Retention |
|---|---:|---:|---:|
| nq-250k, Wikipedia questions (similar data in the training mix) | 0.8839 | 0.8289 | 93.8% |
| cqadup-physics, forum questions | 0.4931 | 0.3501 | 71.0% |
| cqadup-programmers, forum questions | 0.4681 | 0.2345 | 50.1% |

The macro, 82.2%, hides the finding. Where training queries resemble test queries, nano is inside LEAF's band. Where they do not, it retains half. This is a coverage failure, not a capacity failure. A second cause was found later: the 384-wide linear head was a rank bottleneck under regression.

- Documents as extra regression text helped more than repeating queries.
- A closed-form warm start of the head was worth 0.027 over a random head.
- ONNX export with no custom operators; FastEmbed can serve the model exactly if the linear head is applied per token before pooling.
- Edge deployment needs binary quantisation: fp16 is a hundred times slower under a 256 MB limit.

**What it changed.** The M9 candidate was frozen and not released. Its six-set access is unspent, held back so its close-out score can calibrate development-to-six for the next model. M10 was told to build around coverage first.

### M10, 1 to 10 September: preparing nano properly

Rescoped on 10 September to preparation, with execution moved to M13. What M10 delivered is the data, the harness and the recipe.

**The data pipeline.** Coverage means query forms. Three parts, all under commercially usable licences, all screened against every evaluation set:

- **The M9 pool**, re-screened: Natural Questions, SQuAD, HotpotQA, FEVER, MIRACL and Mr.TyDi queries, plus one million PAQ questions.
- **Harvest**: real titles, headings and claim sentences mined from Wikipedia, arXiv and the licensed pool. About 1.25 million rows.
- **Generated**: 834,463 queries in seven forms no corpus provides, written by Qwen3-8B at a pinned revision, running locally. Argument, comparison, conversational, finance, health, how-to and yes-or-no. Each form passed a diversity gate and a duplicate check against MS MARCO.
- **Excluded**: MS MARCO (non-commercial: validate only, never train), FineWeb (cannot be screened against the reserved sets without opening them), Claude as a generator (terms forbid training competing models).

**The screen.** Thirteen arms at 5 million examples each, the two backbone candidates at 20 million, all evaluated on COV. Minimum detectable effect 0.0056. A contrast that does not resolve keeps the default; it does not prove the alternatives equal.

| Contrast | Question | Point | Lower bound | Result |
|---|---|---:|---:|---|
| F1 | Which backbone: bge-small or MiniLM-L6? | +0.0116 | +0.0072 | resolved: bge-small |
| A3−A2 | Does harvested real text beat the same volume of PAQ? | +0.0101 | +0.0054 | positive, not resolved |
| A4−A3 | Do the generated forms add on top of harvest? | +0.0121 | +0.0069 | resolved: use them |
| G1 | 1152-wide head vs the 384 M9 used? | +0.0217 | +0.0153 | resolved, descriptive |
| G2 | 1536 vs 1152? | +0.0003 | −0.0028 | not resolved |
| G3 | Small nonlinear head vs linear? | +0.0024 | −0.0003 | not resolved |
| B1 | All queries instead of 75/25? | −0.0011 | −0.0047 | not resolved |
| B2 | Half and half instead of 75/25? | +0.0027 | −0.0005 | not resolved |
| D1 | Normalised loss instead of squared error? | −0.0002 | −0.0019 | not resolved |
| D2 | Document-covariance-weighted loss? | −0.0184 | −0.0238 | negative, mechanism unresolved |
| E1 | Batch 32 or 128? | — | — | cloud, pending |
| C1 | Warm-start from the M9 checkpoint? | — | — | cut before computation |

Two descriptive reads sit beside the screen: a second anchor seed moved the macro by 0.0007, and the generated forms' gain held at 20 million examples (+0.0165). Eighty-two percent of the 5M gain came from the consumer-health family, which is the honest limit of the coverage story so far.

**What it changed.** The recipe is locked: bge-small backbone, the full A4 corpus, a 1152-wide linear head over three layers, 75% queries and 25% documents, squared-error loss, closed-form warm start. Only the batch size remains open, decided by two arms on the same cloud GPU.

### M11, 3 September: shipping zero

`constella-zero` and `stella-en-400M-v5-doc-onnx` went public on Hugging Face, both served by FastEmbed as built-in models. The published bytes hash to the frozen M7 artifact; serving parity against the numpy reference is 4.5e-8 over 1,024 real queries. Eight release gates run at every push. One recorded slip: the repository was public from its first push, so the plan's flip-public-last guarantee was spent early; nothing non-releasable was ever in it.

### M12, 4 to 9 September: fusion in Qdrant

Zero's fused number came from a convex combination with a development-fitted weight that Qdrant does not ship. No shipped operator reproduces it at depth 1,000: weighted RRF is 0.013 behind, DBSF 0.015. But at a realistic prefetch of 10 to 50 candidates DBSF is equal or better; the convex operator's advantage exists only at deep prefetch. On the six, DBSF at prefetch 100 scores 0.4887 on all six and 0.4912 on the clean four, against convex's 0.4911 and 0.4866: inside the noise band, a tie, not a win.

**What it changed.** The public recommendation became `Fusion.DBSF` at prefetch 100: it runs in Qdrant and fits zero parameters. Recorded as an owner's product-policy override of the M7 release freeze, on deployability grounds.

### M13, 10 September onward: the cloud run

M13 owns everything expensive or irreversible: the batch decision, the LoTTE gate, the 200M build, the final evaluation and the cost frontier.

- **The build controller** pins the dose at exactly 200,000,000 examples in three cycles, checkpoints every 30 minutes of wall-clock, and freezes with an ONNX export and a FastEmbed parity check. Kill-and-resume tested across a cycle boundary; smoked on the box GPU.
- **The scoring transaction** for nano's six-set access: authenticated manifest before the access is spent, frozen document caches only, query texts and labels hashed on the exact objects scored, a synthetic end-to-end rehearsal with no protected data.
- **The scope cut**, ruled by Dylan after review: no extension cycles, no post-tag continuation, one more review, the LoTTE gate as a small script. Reliability by rehearsal, not by recovery machinery.
- **The LoTTE gate** reads the seven cleaned slices once, before the build. If the cloud arms select batch 128 it compares that checkpoint against the batch-32 one on fresh out-of-domain data and can veto in favour of 32. It went through four review rounds in one day, two models alternating: 22 findings, none touching the arithmetic, all about identity and durability, all fixed. It gained an exclusive lock and receipt so a crashed read cannot silently repeat, a committed and pushed checkpoint manifest and per-slice hash pin so a swapped record or altered slice cannot pass, checkpoint bytes hashed and loaded from one buffer, and a build controller that recomputes the veto from the recorded numbers instead of trusting a label. The closing re-check returned GO.
- **Box-side DEV-6**: the two cloud arms skip 35 GB of development caches; their development read is filled on the box from the identical checkpoint bytes, once, with provenance.

**Where it stands.** Code complete, reviewed to GO and tested, 170 M13 tests plus the older suites all green. The provider and account are Dylan's to set up.

## Part four: what we are building now

Take bge-small, a 6-layer BERT-style encoder with 384-wide hidden states. For each query token, concatenate the hidden states from layers 12, 8 and 4 of the pinned architecture into a 1152-wide feature, apply a linear head to 1024 dimensions per token, average over the query's tokens, and normalise. Total 34,540,672 parameters, under the 35M cap. The head is applied before pooling so FastEmbed can serve the exported graph exactly.

Training is regression: 75% of examples are query texts and 25% are documents, in a repeating pattern of three query windows then one document window; the target for each is stella's vector for that text. Squared error against unit-norm targets, which equals cosine loss up to a constant. Three cycles, learning rate annealed from 1e-4 to 1e-5 in each, AdamW, mixed precision. Two stop rules: a kill if two consecutive evaluations fall more than 0.0056 below the best of their kind, and a plateau if the last cycle gains less than 0.003.

```mermaid
flowchart LR
    A["Day one<br/>measured rate, billed price, budget"] --> B["Two E arms<br/>batch 32 vs 128, 5M each, same GPU"]
    B --> C["LoTTE gate<br/>read #1, once; may veto to 32"]
    C --> D["200M build<br/>three cycles, ~37 A100 hours"]
    D --> E["Freeze<br/>ONNX, parity, LoTTE read #2"]
    E --> F["The six<br/>one access, four tests"]
```

Stops between stages: the instance is shut down, records are committed and pushed, and each stage's preconditions are checked before the next spends anything. Ceiling $1,000. Illustrative allocation at 1,500 examples/s and $1.50/h: build $56, both E arms $3, encodes and export $9, LoTTE $2, reserved batch $12, disk $25, about $108 total. Rate and price are measured on day one.

The final evaluation is four tests in a fixed order, each a one-sided bootstrap comparison at 2.5% with 10,000 resamples of queries within each dataset, stopping at the first that does not pass. Nano must beat bge-small on the clean four, then on all six (release gate); then LEAF on all six, then on the clean four (aim). A test never reached is reported as not tested. Then the cost frontier: zero, bge-small and nano on the same Apple M5 Pro under one serving protocol.

## Part five: why so many rules

- **One access per surface.** A benchmark you have already looked at cannot confirm anything. The six are opened once per system by a script that commits a manifest, pushes a tag, scores, and refuses to run again. Files under protected paths are guarded so only a named module can open them; adding a module is a dated ledger amendment.
- **Register before you look.** Partition, test sequence, bars, bootstrap seed and quantile method are fixed in the registry before a number exists. Changes are dated and the original stays in git.
- **Count the peeking.** Development reuse is tallied and published.
- **Frozen comparators.** The competitor vectors cannot be regenerated; overwriting them would destroy the ability to compare.
- **Licences are part of the recipe.** Every training source must permit commercial derived weights; non-commercial sources may validate but never train.
- **Reviews are adversarial and counted.** Reviewers are briefed to break things and forbidden from opening protected surfaces. They have caught a rounded confidence bound in the single irreversible decision path, a gate that reported failure but exited successfully, and a release guard that failed open. Two independent reviews precede anything expensive or irreversible.
- **Prefer cheaper under a tie.** Unresolved keeps the default, and the record says unresolved, not equal.

## Part six: what counts as success, and what a miss would mean

Success for nano is passing the release gate: better than bge-small on the clean four and on all six, at about bge-small's query cost, while serving stella's index unchanged. Passing the aim as well would make it the strongest small query encoder we know of against a frozen index. Success for the project is narrower and already partly in hand: a measured, reproducible answer to how much quality each cheap query side retains, at what cost, with the deployment path proven in Qdrant and FastEmbed.

A miss is publishable. The material already includes: a teacher's retrieval quality does not predict its distilled student, and the most decomposable teacher wins; a lookup-table query side retains about three quarters of its teacher and its objective saturates almost immediately; fusion with a lexical channel is worth ten times any table-side lever; a small transformer's retention is a per-distribution quantity, 94% where the training data resembles the queries and 50% where it does not; and the depth dependence of fusion operators inverts at realistic prefetch.

## Glossary

| Term | Meaning |
|---|---|
| nDCG@10 | The quality metric. Rewards relevant documents near the top of the first ten results; 1.0 is perfect. |
| Retention | A student's score divided by its teacher's on the same data. Reported per component. |
| Teacher, student | stella_en_400M_v5 is the teacher; zero and nano are students trained to reproduce its query vectors. |
| Bootstrap CI | Resample the queries thousands of times and recompute the difference; the spread is the interval. Excludes training-seed variation. |
| Resolved | A difference whose interval excludes zero at the registered level. Not resolved keeps the default and claims nothing. |
| MDE | Minimum detectable effect, 0.0056 for the M10 screen. |
| Arm, contrast | An arm is one trained model under one recipe variant; a contrast is a registered comparison between two arms. |
| COV, DEV-6 | Development surfaces: the coverage surface of four unfamiliar families, and the six components pinned since M7. |
| The six, clean-4, reserved four | The confirmatory datasets, the headline subset without the teacher's disclosed training sets, and the never-opened descriptive sets. |
| LoTTE | Stanford's long-tail StackExchange benchmark; seven cleaned forum slices are the one fresh out-of-domain surface before the build. |
| Spent tag | A git tag pushed the moment a one-shot access begins. Its existence refuses every later attempt. |
| Dose, cycle | Training examples consumed; one learning-rate ramp from 1e-4 to 1e-5. Screen arms ran 5M over three cycles; the build runs exactly 200M. |
| Fusion, DBSF | Combining dense and BM25 results. DBSF normalises each list's scores by their distribution and adds them; it is what Qdrant ships. |
| Inference-free sparse | Systems where documents are expanded by a model but a query is just token counts. Zero's closest competitor class. |

*Sources: `research/m1-m6-findings.md`, the `FINDINGS.md` files of M7 to M10 and M12, `m10/RESULTS.md`, `m10/M102_LOCK.md`, `m13/RULINGS.md`, `m13/EXECUTION.md`, `results/m7_learnability_report.json`, and the registries they cite.*
