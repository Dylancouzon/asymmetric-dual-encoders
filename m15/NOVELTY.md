# M15 contribution and prior-art boundaries

Updated 2026-10-01 for v14 after the owner challenged the earlier cost/teacher emphasis and proposed quantization. Earlier maps remain in git history. Primary-source audits: `REVIEWS/2026-10-01-owner-literature.md`, `REVIEWS/2026-10-01-followup-literature.md`, and `REVIEWS/2026-10-01-quantization-semantics.md`.

## The research question

What does selectable query computation buy when document vectors remain fixed, and what limits the saving? Constella supplies a token table, compact transformer, and original teacher query path aligned to the same Stella document space. The paper measures quality, encoding, graph-search effort, and query precision; teacher choice and build costs explain construction decisions.

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

## Judgment

V14 is a coherent applied research paper for search engineers, with a useful fixed-index systems finding and a controlled precision follow-up. It is not an architectural breakthrough. A matched-quality native remedy would strengthen it; more checkpoints or undirected precision sweeps would mostly add bulk. The focused reader and correctness reviews are `REVIEWS/2026-10-01-v14-*.md`. The narration makes the unit of replication explicit and uses external targets as selected context, not a comprehensive frontier. OWNER_PREFERENCES.md records the intended audience and scientific expectations.
