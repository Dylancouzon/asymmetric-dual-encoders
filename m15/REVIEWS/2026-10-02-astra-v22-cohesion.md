# v22 cohesion and writing pass (GPT-6-Astra, fresh session)

## Brief

# Brief: full cohesion and writing-quality pass on v22 (GPT-6-Astra, fresh session)

Repository: /Users/dylanc/Documents/GitHub/asymetric-dual-encoders, branch m15-whitepaper. Read-only this round: change no files, run nothing else.

Read m15/OWNER_PREFERENCES.md in full first (the owner's goals and every writing ruling: whitepaper not article, interesting not a benchmark reading, problem-first abstract, plain words and no project shorthand, one name per thing, no compression, leaning removes repetition not grammar, no dwelling on what did not work, closed decisions). Then read m15/PAPER.md in full, as a seasoned search engineer and IR researcher reading it for the first time. Named paths only; no recursive search; never open results/frozen_eval/untouched-*, reserved qrels caches, work/m9reserve, or raw M18/sealed M19 data.

The owner's feedback on the current abstract: "Among ten registered embedding models, the best retriever produced the worst token-table student - table or head isn't defined yet. I like the general direction of the abstract but the order isn't coherent." The owner asks for a full pass on cohesion and writing quality. For this pass, wording and order are in scope (the essential-only rule is lifted for writing), but claims, numbers, and evidence labels must not change.

Report, numbered and ranked by how much each fix helps the reader:
1. The abstract: a proposed rewrite (about 200 to 230 words) that keeps its direction and its hook ("A dense index encodes each document once; its query encoder runs on every uncached search..."), defines every term before use (student, table, transformer student), and orders the argument coherently. Keep every claim and number it currently has unless one breaks the order; say what you moved and why.
2. Cohesion across the paper: places where a section or paragraph does not follow from the one before, where the argument repeats, or where the reader loses the thread. Quote the passage and give the rewrite or move.
3. Writing quality: sentences a reader must read twice, undefined or inconsistent terms, leftover shorthand, flat or defensive phrasing. Quote and rewrite. Prioritize the abstract, introduction, section openings, and Discussion.
4. Anything that reads as an article (hook-and-reveal, product voice) or as a benchmark reading (numbers without a point).

Keep the voice: third person, plain words, complete sentences, no em dashes, no new jargon. Under 1,600 words.

## Astra

The paper has a coherent research argument, but definitions and transitions sometimes obscure it. The most useful changes would clarify what is being compared before presenting the surprising results, then make each change of experiment or timing protocol explicit.

1. **Rebuild the abstract around definitions, findings, and their practical consequence.**

   Proposed rewrite, approximately 230 words:

   > A dense index encodes each document once; its query encoder runs on every uncached search. Replacing only the query encoder preserves stored vectors, graphs, and quantized codes. This study asks how much quality survives and how search cost changes.
   >
   > A student is a smaller query encoder trained to reproduce the original model’s query vectors. The study fits token tables that average learned token vectors and transformer students that project features from a frozen transformer. Among ten registered embedding models, the best retriever produces the worst table student. Across 26 models, dimensionality and measured retrieval score jointly predict student quality, with prospective support for the transformer student. On unchanged HNSW graphs, fitted tables need a median two to four times the search breadth to recover the same fraction of their own exact neighbors. This deficit recurs in a jointly trained lookup model; the index’s own queries flag where it is largest.
   >
   > Stella, a 400-million-parameter English embedding model, supports two separately trained students without document re-encoding: the 34.5-million-parameter transformer Nano and token table Zero. They retain 90.5% and 81.4% of Stella’s BEIR-15 nDCG@10, respectively. In an exploratory selection on one binary-quantized collection, encoding plus search falls from 21.3 ms to 2.48 ms and 1.11 ms, respectively, at lower relevance. Nano’s final optimization took 57.3 A100 hours. The shared index lets each request choose its query encoder.

   Student definitions now precede the reversal; prediction follows that reversal; graph cost follows representation quality; Stella supplies the concrete application. Move “31.6 ms to 0.044 ms” to Section 6 alone. That separate CPU encoding protocol interrupts the opening question and invites comparison with the later encoding-plus-search timings. All other existing abstract claims and numbers remain.

2. **Give the fitted transformer student one name, and distinguish it consistently from Nano.**

   The abstract says “transformer student,” the introduction says “a frozen transformer with a learned projection,” and Section 3 introduces “contextual head” before defining “head student.” A reader must infer that these describe the same fitted encoder.

   Use **table student** and **transformer student** throughout the narrative, figures, and tables. At first definition:

   > The transformer student uses a frozen bge-small transformer to produce contextual features. Ridge regression fits a projection from those features into the original model’s document representation. Only the projection is learned.

   Then distinguish the released models:

   > Zero and Nano use additional training beyond these fitted students. Zero learns token vectors and weights; Nano trains both the transformer and its projection.

   Move “This requirement belongs to the table comparison; the contextual head fits vectors from a separate feature encoder” until after those definitions, rewriting it as:

   > The shared-vocabulary requirement applies to the table comparison. The transformer student obtains its features from bge-small.

   This also prevents readers from treating the prospective transformer result as a prospective test of Nano.

3. **Make Section 6’s timing comparisons form an argument instead of successive benchmark reports.**

   “Search narrows the difference between the cheap encoders” sounds general. Two paragraphs later, “Zero is 2.2-fold faster than Nano including search” appears to reverse it. The explanation exists, but arrives after the reader has formed the wrong comparison.

   Replace the first opening with:

   > On the uncompressed million-passage collection, additional graph search consumes much of Zero’s encoding advantage over Nano.

   Open the binary comparison with:

   > The binary-quantized collection leaves a larger speed difference between Nano and Zero. This comparison uses encoding timings from the diagnostic’s own queries, measured separately from search.

   Keep the existing exploratory label, numbers, relevance levels, and target definition immediately afterward. The reader can then understand why the two comparisons differ before encountering another set of timings.

   Move “Construction and recurring cost” after these search comparisons and before the request-context example. The sequence becomes retained quality, recurring cost, construction cost, and the choices those costs enable. Preserve both models’ construction estimates and all exclusions.

4. **Let the introduction state the research questions before presenting Constella and the teaser.**

   “**Constella** makes that choice concrete” introduces the named family before the reader knows what the study will establish. The actual research overview comes after Figure 1.

   Move the paragraph beginning “We fit two kinds of student across 26 embedding models” ahead of Constella, opening it with:

   > The study asks which properties of an existing document representation predict student retrieval quality, and how replacing the query encoder changes the graph-search effort required.

   Follow with the existing description of the two fitted students, then introduce Constella and Figure 1. This preserves the research order and gives the figure a purpose.

   The contribution “**Cheaper queries make the unchanged graph work harder**” also overgeneralizes the cost:

   > **Replacing the query encoder increases the search breadth needed for exact-neighbor recovery.**

   Keep the effect sizes and evidence status in the accompanying text. The scientific finding supplies the interest without personifying the graph.

5. **Make Section 4’s final turn from prediction to selection explicit.**

   “Ranking prediction and choosing the best student are different tests” is important, but currently feels like a new objection after the prospective result.

   Replace that transition with:

   > The prospective result supports predicting student rankings from properties of the original model. Selecting a single student is a further test: the following comparison measures how much retrieval quality each selection rule loses relative to the best available student.

   Keep every reported loss unchanged. Then replace:

   > “Direct screening is practical because…”

   with:

   > Direct evaluation selected stronger students on the original model list, and its construction cost was modest: table fits took four to seven minutes per registered embedding model on an A100, including query encoding except for cached Stella targets.

   The cosine examples then explain why retrieval evaluation matters. They cease to look like another unrelated finding appended to the section.

6. **Use Discussion for interpretation rather than a second contributions list.**

   “The quality study identifies conditional predictors…” and “These results have three consequences…” largely repeat the introduction and results.

   Replace the opening paragraph and numbered recap with:

   > The findings separate two consequences of replacing a query encoder. Exact retrieval measures how well the replacement ranks the stored documents. Approximate search measures the additional loss introduced by searching their graph. A student can therefore reduce encoding cost while requiring greater graph-search breadth, and these effects must be considered together when interpreting the resulting savings.

   Then develop the existing conclusions about direct evaluation, the distance predictor, and per-request choice without restating their full inventories.

   Rewrite “The selected own-index references also delimit the proposition” as:

   > The comparison with bge-small and LEAF concerns the value of retaining a shared index.

   Retain their scores and the existing explanation about separate indexes and ingestion pipelines. This answers the comparison directly without defensive framing.

7. **Repair the remaining sentences that require translation.**

   - “The first four, without disclosed Stella exposure, form a sensitivity outcome.”

     > A sensitivity analysis uses the first four datasets, for which Stella discloses no training exposure.

   - “It remains 0.4 points larger than SCIDOCS’s, with interval [−0.9, −0.1] for the signed difference.”

     > Recovery remains 0.4 percentage points worse than on SCIDOCS; the interval for the signed difference is [−0.9, −0.1].

   - “The family exposes useful headroom, but these simple features do not recover it.”

     > The oracle shows the potential benefit of choosing between Zero and Nano for each query. The tested features capture little of that benefit.

   - “Descriptive BEIR-15 fusion with BM25 raises Zero to 0.4933, or 87.9% of Stella, but its sign depends on workload…”

     > In the descriptive BEIR-15 evaluation, fusion with BM25 raises Zero’s score to 0.4933, or 87.9% of Stella’s score. The effect varies by workload; fusion lowers Nano’s score on MS MARCO.

   Also replace remaining “original query path,” “spaces,” and “roster” with the relevant encoder, document representation, or model list.

8. **Remove the last inventory-like detour without suppressing its results.**

   Section 6 ends with “A lexical branch supplies another compatible route.” After the request-context policy and routing experiment, fusion introduces a different architecture and leaves the section ending on an ancillary result.

   Move this paragraph immediately after the per-dataset quality discussion. There it extends the account of Zero’s retrieval quality, with its sparse-index cost stated alongside it. End Section 6 with the distinction between request-context selection and unresolved relevance-based selection.

   Keep the chosen title, teaser, construction costs, and registered unfavorable results. The article-like tendency comes from occasional promotional transitions; the benchmark-like tendency comes from comparisons presented before their purpose is clear. Neither requires weakening the findings.
## Agreed application

Apply your pass to m15/PAPER.md, with these agreed modifications (Claude, for the owner). You now have write access; modify only m15/PAPER.md. Do not change any claim, number, or evidence label; do not compress sentences you are not rewriting.

Accepted as you proposed: items 2 (one name, "transformer student", everywhere including Table 1 row labels, figure captions and alt text, Appendix B and C headers and prose; keep Nano distinct), 3 (Section 6 timing transitions and moving the construction-cost paragraph after the search comparisons, before the request-context example), 4 (move the study paragraph ahead of Constella in the introduction, with your opening sentence; Figure 1 stays right after the Constella paragraph), 5 (Section 4 transition and the direct-evaluation sentence), 7 (all sentence repairs and remaining shorthand), 8 (fusion paragraph moved after the per-dataset quality paragraph; Section 6 ends on request-context versus relevance-based selection).

Modified:
1. Abstract: use your rewrite, but keep the hook number in the opening: "Replacing only the query encoder preserves stored vectors, graphs, and quantized codes, and can cut median CPU query-encoding time from 31.6 ms to 0.044 ms." The owner asked for a strong hook and liked this direction. If you think readers will compare it with the later encoding-plus-search timings, mark the later sentence as "encoding plus search" (already there) rather than removing the hook. Replace "This study asks how much quality survives and how search cost changes" with a sentence that keeps the question but reads with force, for example "How much quality survives, and what the replacement costs at search time, depends on the index it inherits."
4b. Contribution headline: use "Replacement queries need a wider search over the unchanged graph." instead of your flatter version; keep the effect sizes and scope in the text.
6. Discussion: replace the opening paragraph with your interpretation paragraph, and adopt your rewrite of the bge-small/LEAF sentence. Keep the "three consequences for a search system" paragraph, because the owner requires the paper to say what a search engineer can decide differently; trim only clauses that restate result inventories.

Reply under 200 words: what you changed, and the main-text prose word count.

### Astra

Applied the agreed pass to [m15/PAPER.md](/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m15/PAPER.md) only.

- Retained the abstract’s timing hook and defined students before the findings.
- Standardized “transformer student” throughout.
- Reordered the introduction and Section 6, clarified timing transitions, and repaired the specified sentences.
- Revised Discussion while retaining its three practical consequences.

Claims, numbers, and evidence labels are preserved. Numeric, terminology, paragraph-order, and whitespace checks passed.

**Main-text prose: 3,739 words** for Sections 1–8, excluding headings, captions, tables, alt text, and citations. **Including the abstract: 3,982 words.**
## Connective sentences check

1. **P2: “Savings … recur on every query” overstates both timing results.** The measurements are medians, cached queries need no encoding, and requests may select Stella. In both the abstract and construction paragraph, replace with:
   > Using a smaller encoder can reduce recurring computation on uncached queries, while construction incurs a one-time cost.

   Keep Nano’s optimization figure in the following sentence.

2. **P2: “Large differences by query type” attributes dataset-level differences to query type.** The cited results compare datasets, which also differ in documents and relevance judgments. Replace with:
   > The averages in Table 2 conceal substantial differences in retained quality across datasets.

3. **P2: “The search penalty from Section 5 reappears as time” conflates distinct experiments and sounds like an article transition.** Section 5 primarily establishes neighbor-recovery effort; this comparison uses released models, another workload, and relative nDCG targets. Replace with:
   > The Constella comparison measures how additional graph-search effort affects the encoding-time savings.

The BM25 statement is supported by the BEIR-15 macro improvement. Both coefficient intervals exclude zero for both students in Table 1. The query-placement description fairly summarizes the distance ratio, which the next sentence defines.