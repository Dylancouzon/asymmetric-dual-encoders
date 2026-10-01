# Paper direction and alternatives for reviewers

## Current choice and its strongest objection

V15 asks what cheaper query computation buys when the document vectors stay fixed. It leads with
Constella's three compatible query paths, then measures relevance and encoding cost, how ANN work
changes the savings, and what fusion, routing, construction, and serving failures add to that picture.
The intended contribution is an applied empirical study that helps search engineers reason about
these choices. It is not architectural priority, a model leaderboard, or a production routing method.

This is **our current editorial recommendation, not owner acceptance or coauthor consensus**.
The owner has repeatedly challenged the draft's purpose, readability, and selection. Reviewers should
reopen the direction if another question yields a more coherent or more useful paper.

The strongest objection is that v15 still contains two partly separate studies: using several query
paths over a fixed Stella index, and screening teachers before choosing a new index. The former is
the clearest production question; the latter may be the most surprising research result. Fusion and
runtime cases add practical value but can make the paper feel like a collection of project findings.
This tension is not resolved merely by shortening the draft.
The main ANN measurements also cover one teacher space, one engine, and two workloads, including
a positive-preserving million-passage subset, at unequal absolute quality. Reviewers may prefer a
narrower fixed-index operating-envelope paper if the breadth weakens its transferable conclusion.

## Why we currently lead with the fixed-index question

- **It explains the research object.** Constella supplies several aligned query budgets over one
  document representation. This is the common premise behind the shipped artifacts, measured
  quality, query timing, and ANN study.
- **It has a concrete measured consequence.** Encoding savings can shrink substantially after
  approximate retrieval. Existing document vectors do not guarantee that the old search settings
  preserve each new query path's quality. This is useful even when the architecture has prior art.
- **It accommodates unfavorable evidence.** Nano loses to the selected own-index bge-small/LEAF
  references on the broad descriptive average. Retaining an index is a conditional reason to use
  Constella, not a claim that it is always the best deployment. Migration savings were not measured.
- **It fits the owner's audience.** An experienced search engineer can recognize the cost/quality
  decision without first understanding every teacher-fitting experiment. The teacher screen then
  becomes a distinct finding about constructing a family before indexing.

The body keeps findings with distinct consequences: the ANN interaction; a bounded query-precision
control; fusion ordering changing with candidate depth; unsuccessful simple routing/blending; the
teacher/table reversal; and actual caller/device/dtype failures. The full history is preserved in
[LEARNINGS.md](LEARNINGS.md), rather than treated as mandatory manuscript content.

## How the direction changed

The early plan led with retention and contained many separate probes and use cases. A later rewrite
emphasized teacher choice and component training cost. It made the claims easier to defend but
underrepresented Constella itself. The owner challenged that narrowing; the subsequent draft put
compatible query choices and search-inclusive cost back first. Bounded precision experiments added
evidence without retraining. The owner then challenged jargon, verbosity, generalization, comparator
framing, and whether we had considered the whole repository. V15 drew on the full milestone audit,
reduced the main text, and moved detailed provenance out of the standalone paper.

The history includes reversals in agent judgment. Earlier reviews calling a draft good did not mean
the owner agreed, that the framing was final, or that the paper would be a hit. Relevant records:
[earlier plan](PLAN.md), [cost/teacher rewrite](REVIEWS/2026-10-01-owner-synthesis.md),
[broader reassessment](REVIEWS/2026-10-01-owner-broader-synthesis.md),
[precision disposition](REVIEWS/2026-10-01-v13-synthesis.md),
[narration/comparison disposition](REVIEWS/2026-10-01-v14-synthesis.md), and
[full-history v15 selection](REVIEWS/2026-10-01-v15-synthesis.md). Historical claims use their
then-current wording; the current manuscript/evidence map governs present interpretation.

## Alternative directions available to reviewers

The alternatives below were raised in owner discussion, earlier drafts/plans, or the full-history
selection. They were not all developed into complete papers. A proposed strengthening experiment
is identified separately from a result we already have.

### 1. A teacher-selection and cheap-representation paper

**Case for it:** the 26-checkpoint ridge-table screen contains a large, memorable teacher/table
reversal. Teacher retrieval scores and vector fidelity do not reliably rank the fitted tables under
this recipe. It offers a tighter research question than a broad deployment study.

**Why it is supporting in v15:** it changes teacher and document space, whereas the opening question
assumes an existing index. The screen has not validated teacher ranking for the trained Zero/Nano
recipes. A teacher-first paper must say what representation it actually tests.

**What could change the choice:** reviewers may judge this the most valuable finding already present;
it can lead a narrower paper without new compute. An added recipe/teacher control could test transfer
of the ranking, but would be new work. Evidence: L02/L03/L30 and C3/C3b/C4/C13.

### 2. Training cost and approachability

**Case for it:** small compatible encoders are potentially approachable to teams without large
training budgets. The final Nano optimization loop has measured duration and a historical price;
Zero fitting was much shorter after preparation. The owner explicitly proposed this angle.

**Why it is not the headline:** the approximately $95 figure excludes target preparation, generation,
recipe search, failed runs, evaluation, export, and engineering. The current repository is not a
turnkey cold-build kit. A partial invoice cannot establish total affordability or ease of reproduction.

**What could change the choice:** complete component accounting and recoverable build inputs would
support a stronger approachability claim. A readable accounting supplement needs no full retraining.
Evidence: L07/L28/L29 and C13; costs are supporting evidence, not an approved owner thesis.

### 3. A benchmark or model-release report

**Case for it:** the models attracted community attention, and quality/latency tables make it easy to
understand their tradeoffs. Broad exact evaluation and original registered tests are available.

**Why it does not organize v15:** the owner wants new knowledge and production value, rather than
a competitor overview. bge-small and LEAF were selected project targets, not a comprehensive frontier.
The broader comparisons include unfavorable outcomes; hiding them would damage credibility.
A broader cheap-query frontier, including sparse retrieval and own-index small models, is another
possible comparison-focused direction. Early local comparisons do not supply a matched, comprehensive
quality/cost survey; different indexes and protocols must not be merged into a universal ranking.

**What could change the choice:** reviewers could prefer an accompanying release note or clearer
selected-reference table. A larger matched model survey is new work, not something the current data
already proves. Registered outcomes remain reportable under any framing. Evidence: C1/C2/C11 and
Appendix A; L22/L23/L32.

### 4. A query-precision and ANN paper

**Case for it:** changing the query encoder changes the search work needed over the same graph.
The fixed-code control shows that preserving query magnitudes recovers candidate information while
document codes stay unchanged. This may yield a more focused systems finding with a useful remedy.

**Why it is bounded in v15:** the strongest native pilot has a weaker primary replication. The
repeated exhaustive precision control has no graph or native latency and improves all tiers.
It is not yet a new quantizer, a proved mechanism, or a measured production speedup.

**What could change the choice:** supported native precision at matched recovery/relevance and measured
cost, with graph/rebuild variation controlled, would strengthen this direction. Another teacher space
would test reach beyond Stella. Both are proposed work, not completed results. Evidence: L10–L12,
C10/C14/C15, [FOLLOWUP.md](FOLLOWUP.md).

### 5. Selectable compute and adaptive query routing

**Case for it:** multiple compatible query paths can be selected without rebuilding the document
index. Per-query quality differences show headroom for assigning compute selectively. This speaks
directly to Constella's practical appeal.

**Why it is not a successful policy paper:** sequential encoder batches establish compatibility,
not timed loading/failover. The oracle uses relevance labels; measured simple routers and blends fall
short. Post-search signals require additional encoding/search cost. No successful production router
has been established.

**What could change the choice:** a policy that beats static/random routing at a measured request
budget, including repeat work, would add a result. We can explain existing selection capability
without that claim. Evidence: L01/L18/L19 and C5/C7/C8/C9.

### 6. Construction recipes and failed shortcuts

**Case for it:** head initialization, representation width, data/query form, failed linear alignment,
loss/fidelity proxies, and vocabulary interventions can teach teams what to test before long training.
This is the strongest construction-focused alternative in the historical catalog.

**Why most details are outside v15:** many observations are local recipe screens, not a causal
decomposition of the successful rebuild. The full Nano improvement cannot be assigned to coverage
or capacity alone. Absorbability algebra applies to specific transforms; it does not prove a universal
static-model ceiling. A long inventory would dilute the serving question.

**What could change the choice:** an author may prefer a focused construction study using the existing
bounded results. Broader ablations or hard-candidate training are unrun and must not be inferred.
Evidence: L03–L09/L20/L30/L33 and historical M7–M10 findings named in the catalog.

### 7. Fusion as the main production intervention

**Case for it:** adding a lexical route recovers more Zero quality than several table-side probes;
fusion-operator ordering reverses with candidate depth. A search engineer can act on that observation
without building a new encoder. It is a useful finding from before M15.

**Why it is concise in v15:** dense and fused scores vary by task. The historical development curve
uses bm25s/Lucene settings and fixed convex weight .8; it is not a universal DBSF comparison or native
BM25 substitution test. The later DBSF policy was chosen for implementability, not a newly confirmed
win. A fusion-first paper might make Constella incidental.

**What could change the choice:** representative native fusion replication at measured quality/cost
could support a tighter paper; the existing descriptive depth reversal can also receive more emphasis
without new compute. Evidence: L15–L17, C1/C10, M12 receipts in the current evidence map.

### 8. Serving compatibility, edge deployment, and bounded adaptation

**Case for it:** real caller/device/padding/dtype failures show that aligned checkpoint vectors alone
do not certify the exported path. Bounded term-triggered patches and uncertain labels reveal other
production constraints. The full catalog also includes token-based encoding cost and allocator issues.

**Why only selected cases remain:** a failure diary is not a coherent retrieval claim. The early
256-MB prototype used synthetic data and a different Nano stand-in; it does not establish the released
family's complete edge footprint. Patch relevance is inconclusive, and internal judgments limit its
generalization. Runtime cases are supporting evidence, not current-release defect claims.

**What could change the choice:** a separate engineering case study could use these records; representative
judgments would be needed for an improved patching claim. Do not promote an operational artifact into
an improved encoder result. Evidence: L13/L21/L24/L26–L29/L31 and the M11/M14/M18–M23 records in
the learning catalog.

## What we want reviewers to decide

1. Does the fixed-index question matter enough to organize the paper, or should the teacher/table
   finding lead a narrower study?
2. Which single finding would you repeat to another search engineer? Does the title/abstract make
   that finding clear, and is its evidence strong enough?
3. Does the teacher construction section belong in this paper, a shorter supporting section, or
   a separate paper? Does the broader narrative connect its parts convincingly?
4. Which omitted catalog lesson deserves more space, and what conclusion would it change?
5. Would one bounded new experiment materially change the paper's usefulness or reach? Identify
   the claim, required outcome, and plausible negative result before choosing compute.

Reviewers may recommend another thesis, splitting the material, or removing a section. The current
direction is a reasoned selection from the evidence, not a requirement to defend prior drafts.
