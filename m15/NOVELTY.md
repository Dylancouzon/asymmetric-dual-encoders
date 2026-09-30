# M15 contribution and prior-art boundaries

Updated 2026-10-01 after the owner-directed review. The earlier novelty map remains in git history; the primary-source audit is `REVIEWS/2026-10-01-owner-literature.md`.

## The research question

When is it worth building a cheap query encoder for a document index that will remain fixed? The paper examines teacher choice under a particular table recipe, the preparation and fitting costs of served encoders, and their quality and latency after approximate search.

## What is established prior work

Query-side distillation against teacher document vectors is established by QED, EmbedDistill, LEAF, and NanoVDR. pyNIFE aligns a static table with a frozen teacher, reuses its index, and proposes static/contextual switching. LightRetriever trains a lookup-and-average query path with a document LLM and reports full-BEIR and end-to-end measurements. Cho and Hariharan and PROD already show that teacher strength need not translate to a stronger student. Approachability, lookup queries, index reuse, switching, and the general teacher/student caution are not architectural priority claims for this paper.

## What the evidence adds

- One controlled closed-form recipe over 26 text-embedding checkpoints, ten registered and 16 exploratory. Teacher retrieval rank poorly tracks table rank; a development table screen tracks it much better. The gte-large/Stella reversal makes the choice consequential. This is limited to the measured recipe and workloads.
- Workload sensitivity made explicit: the development winner is not the best table on every dataset or clean-four objective. Absolute quality and relative retention are reported separately, avoiding a causal size interpretation.
- Query encoders measured over the same frozen index expose a larger ANN penalty for the table and a much smaller end-to-end speedup than encoding alone suggests. This is a measured interaction, not a claim that Qdrant uniquely enables the architecture.
- Component build-cost accounting makes the work easier to assess and reproduce. It is useful supporting evidence, not a novel training method or a complete cheap-build guarantee.

## Evidence gaps that stay visible

The 26-teacher screen predicts the closed-form tables, not the separately trained Zero or Nano recipes. Those served recipes were trained on Stella only. A contrasting-teacher trained-table experiment would test that bridge; until then the paper makes no general trained-student selection claim.

NanoVDR reports teacher quality predictive of student retention across datasets for one teacher. Our cross-teacher comparison addresses a different question and does not refute that result. Shuffle sensitivity remains a diagnostic association, routing is bounded by a label-aware oracle, and query-vector blending is a negative measured result without a priority claim.

## Sources

The paper bibliography and `RELATED_WORK.md` give direct primary links for QED (2306.11550), EmbedDistill (2301.12005), LEAF (2509.12539), NanoVDR (2603.12824), LightRetriever (2505.12260), pyNIFE, Cho and Hariharan (ICCV 2019), PROD (2209.13335), retrieval strategy selection (2109.10739), and query performance prediction (2302.09947). This is a bounded source check, not a systematic demonstration of absence from the literature.
