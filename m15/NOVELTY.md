# M15 contribution and prior-art boundaries

Updated 2026-10-01 for v15 after a full-history audit and the owner's reader-value and whitepaper-genre corrections. Earlier maps remain in git history. Primary-source audits: `REVIEWS/2026-10-01-owner-literature.md`, `REVIEWS/2026-10-01-followup-literature.md`, and `REVIEWS/2026-10-01-quantization-semantics.md`.

## The research question

What does selectable query computation buy when document vectors remain fixed, and what limits the saving? Constella supplies a token table, compact transformer, and original teacher query path aligned to the same Stella document space. The paper measures quality, encoding, and graph-search effort, with a bounded precision control. Fusion depth, failed adaptive policies, and actual serving-path failures connect the wider research history to the deployment tradeoff. Teacher choice is a separate pre-index construction question. The full candidate inventory is LEARNINGS.md; manuscript provenance is PAPER_EVIDENCE_MAP.md.

## What is established prior work

Query-side distillation against fixed document vectors is established by QED, EmbedDistill, LEAF, and NanoVDR. pyNIFE aligns a static table with a frozen teacher, reuses its index, and proposes static/contextual switching. LightRetriever trains a lookup-and-average query path with a document LLM. Cho and Hariharan and PROD establish that teacher strength need not translate to a stronger student. Approachability, lookup queries, index reuse, and switching are not architectural priority claims.

OOD-DiskANN and RoarGraph establish query-distribution effects on approximate graph search. QA-Cos studies query-aware decoding of binary sketches. Qdrant already supports asymmetric query precision. The paper does not invent OOD search, a quantizer, or magnitude-aware scoring.

## What the evidence adds

- A served family of three compatible query budgets, with measured exact quality, warmed encoding, assets, and search on the same populated collections. Hot swapping means selecting an available aligned query encoder; model-loading transitions and concurrent switching are not timed.
- Encoding savings and search savings differ. The table needs more graph-search effort on the measured workloads; an approximately 62-fold Zero/Nano encoding gap becomes 2.2-fold including retrieval on one shared binary collection, at each tier's own quality target. One unquantized configuration nearly consumes the advantage.
- A fixed-document-code query-precision control (E18): preserving query magnitudes increases original-neighbor candidate coverage for all three tiers on both workloads. Zero gains 10.8 and 6.7 percentage points at 40 candidates. This is an absolute-headroom result, not greater proportional architectural sensitivity, a new scorer, or a native latency improvement. The native interaction is workload-sensitive: E17's primary replication is weaker than E16's pilot.
- One controlled closed-form table recipe over 26 checkpoints, ten registered and 16 exploratory. Teacher quality poorly ranks table quality; a development table screen ranks it much better. The gte-large/Stella reversal makes this recipe-specific decision consequential, with task sensitivity disclosed.
- Component build-cost accounting supports approachability without treating fitting cost as complete project cost. It is useful construction evidence, not a novel training method.

## Evidence gaps that stay visible

The 26-teacher screen predicts closed-form tables, not the separately trained Zero or Nano recipes. Both served recipes use Stella and differ in training data. No matched-data architectural comparison or general teacher-selection rule follows.

E18 uses exhaustive candidate scoring and an expected tie treatment, not native HNSW or scalar8 query encoding. Larger absolute gains have more miss headroom. Similar mean sign-direction cosines do not identify or exclude a geometric mechanism. Native precision/cost validation is the next bounded research opportunity.

NanoVDR's positive teacher-quality association varies datasets for one teacher, not teachers. Shuffle sensitivity is diagnostic, routing is bounded by a label-aware oracle, and blending is a measured negative result. These observations remain scoped and mostly in appendices.

## Current judgment and editorial boundary

V15 is an empirical whitepaper about the quality and systems tradeoffs of query-side distillation. Its strongest findings are the shared-index encoding/search distinction and the consequential teacher/table reversal under a controlled recipe. Fusion-depth behavior and actual-runtime failures add production relevance from the wider project. They are supporting observations, not a new architecture or universal production prescription.

The main text is about 3,300 words, compared with roughly 5,900 in v14, with two figures rather than four. Internal file citations, long diagnostic appendices, and repeated instructional prompts were removed. The owner explicitly wants a whitepaper, not a tutorial; implications remain discussion of measured findings. The hypothetical absolute-quality/time example interprets measured rows and is not an optimized or registered policy.

Independent reviews are REVIEWS/2026-10-01-learning-synthesis-review.md and the v15 reader/correctness files. Their findings were resolved; this is not owner acceptance or a publication-readiness certificate. The full-history catalog retains omitted construction, patching, evaluation, and engineering lessons for future selection. A cost-qualified native remedy would strengthen the main deployment finding; additional model spaces would strengthen its reach. Neither exists yet.

## v17 (2026-10-01): the frozen-index frame

**Question.** What does a cheap query student cost over a frozen index, and what does the index
decide about that cost? **RQ1:** the space's own quality predicts the student only once width is
held fixed; width is a strong negative effect (head +0.69 / −0.83, table +0.63 / −0.64, standardized,
intervals excluding zero); a two-variable model fitted on ten spaces ranks 16 held-out spaces at
0.85 and 0.70 (E21); the strongest-space rule loses 0.285 / 0.152 nDCG against the best student and
the student's own dev screen loses 0.035 / 0.000. **RQ2:** table queries recover fewer exact
neighbours at equal ef in 25/25 spaces on FiQA and TREC-COVID, median recovery multiplier four
(E20); the pre-specified relevance-loss multiplier agrees on FiQA and is mixed elsewhere; the two
geometry summaries do not rank the gap. **Instance:** Constella over Stella, with CPU time per
million queries 9.2 h / 1.2 h / 1.0 h (Stella / Nano / Zero).

**What stays prior art.** Query-side distillation, static tables, switching, stronger-teacher
weaker-student effects, OOD graph difficulty, query-aware binary decoding. **What is new.** The
conditional structure of teacher predictability across 26 spaces with a held-out prediction, and
the cross-space search-effort regularity. **Pending.** E24 makes RQ1 prospective; E22 and E23 decide
whether RQ2 is predictive and whether corpus size explains its attenuation.

**E25 (2026-10-01).** RQ2's penalty reproduces in a jointly trained model's own space: LightRetriever's
lookup path needs four times the ef of its full encoder to match exact-neighbour recovery over its own
FiQA and SCIDOCS indexes. This supports the paper's claim that query-encoding speedups measured alone
overstate the saving, without disputing the cited paper's retention or speedup figures.

**E22 to E24 (2026-10-01).** RQ1 is prospective (never-seen spaces, predictions hashed before
scoring: head 0.86 over eight, table 0.71 over seven, against 0.26 / 0.39 for teacher score alone;
E24 amendment 1 restored the head point the first assembly dropped). RQ2 has a predictor (the
query-to-document distance ratio ranks the penalty across 75 points and predicts held-out families)
and a size explanation for most of its attenuation. Nothing pending.
