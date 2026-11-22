# Astra (gpt-6-astra) narration rounds on v18 (2026-10-02)

Briefs: `briefs-2026-10-02/v18-astra-round1.md`, `-round2.md`, `-round3.md`. Verbatim answers.

## Round 1

**1. Numbers in the edited passages**

The amended prospective results match C22: head n=8, +0.86 [+0.33, +1.00]; table n=7, +0.71 [−0.12, +1.00]. The coefficients, selection regrets, E23 changes, Table F predictions, and LightRetriever ranges match the supplied tables/cards. Remaining mismatches:

- Abstract: “At search time over the index's unchanged HNSW graph, the student's queries recover fewer of the encoder's exact neighbors at equal effort in all 25 measured spaces on two of three corpora, and matching the encoder's recovery takes four times the search budget on two corpora and twice on the third; the pre-specified relevance-loss measure holds on two corpora and is mixed on the third.”

  Section 3.4 measures **each path against its own exact top-10**, not student recovery of the encoder’s neighbors. Table D’s 4/2/4 multipliers are **medians**, not uniform requirements. Both distinctions must appear here. The following LightRetriever sentence also needs “of its own exact neighbors.”

- Section 1: “RQ1 is a property of the space; the corpus and the graph do not enter it.”

  Section 3.3 defines the outcome using exact retrieval on six corpora. The graph does not enter; the corpora do. Replace with: “RQ1 measures student quality under exact search, independently of approximate-search behavior.”

- Section 4.4: “The index's width sets the floor; a screen with the fitted student makes the pick.”

  C18 establishes negative conditional coefficients, −0.83 and −0.64, not a width-determined lower bound. Replace “sets the floor” with “helps predict student quality.”

- Section 6, Table 2: “CPU time per million queries on the measurement laptop, from the medians in Table 1 and Appendix C.”

  The arithmetic extrapolates elapsed query timings: 31.6 ms × one million = 8.78 hours. It does not establish CPU resource consumption, especially with four encoding threads. Call this “Extrapolated encoding-plus-search time per million queries”; change the preceding paragraph consistently.

- Section 5.2: “Corpus size accounts for most of the attenuation and not all of it; the residual is not attributed here.”

  C21 explicitly excludes a causal size conclusion from one subsample seed. The observed gap moves from −2.94 to −1.17 points, versus approximately −0.75 on SCIDOCS. Say: “Subsampling FiQA reproduces most, but not all, of SCIDOCS’s smaller gap.”

**2. Narration**

This is the strongest structure described in the brief, but not yet the strongest defensible wording. Ranked changes:

1. Current: “A query encoder distilled onto a frozen dense index inherits the index's width as a penalty.”

   Replacement: “Across 26 frozen encoder spaces, wider vectors predict lower retrieval quality for two closed-form query students after accounting for the space’s own quality.”

   This makes the opening citable without converting a predictive association into a universal causal result.

2. Current: “Methods that train both towers at once retain more (LightRetriever's lookup query side keeps about 95% of its co-trained document tower's BEIR score, against 81.4% for Zero over a frozen Stella); the difference is the price of the frozen index, since joint training re-encodes every document.”

   Replacement: “LightRetriever reports about 95% retention with a jointly trained document tower; Zero retains 81.4% over frozen Stella under this paper’s protocol. These comparisons do not isolate the quality cost of freezing the index. Query-side distillation instead preserves an existing document representation and avoids its reconstruction.”

   The present causal attribution invites an immediate, justified objection.

3. Current: “The field's working assumption, which we take as the hypothesis to test, is that a better space yields a better student.”

   Replacement: “We test whether selecting the encoder with the highest retrieval score also selects the best closed-form query student.”

   This states a consequential, falsifiable question without claiming an undocumented field consensus.

4. Current: “Query-encoding speedups measured alone therefore overstate what a cheap query path saves.”

   Replacement: “In the timed Constella instance, additional search reduces the encoding advantage; Nano and Zero nevertheless require approximately 13% and 11% of Stella’s encoding-plus-search time at thresholds defined relative to each path’s own exact quality.”

   The abstract currently ends with the efficiency qualification while omitting the measured efficiency result that motivates deployment.

**3. Coherence**

The cuts preserve the argument. Three repairs remain: numbering jumps from 5.3 to 5.5; Section 5.3 says “all nine features are in Appendix C,” although its results table contains only the seven omitted from Table F; “24 and 23 of 25 spaces” in 5.2 lacks workload labels. Name the workloads.

**4. Disagreements**

Reverse none of the accepted cuts. Keep the definitions beside Table B. I withdraw the fixed 180-word abstract target: the issue is claim selection and scope, not an arbitrary count.

**5. Sign-off**

**No.** The one blocker is evidentiary scope: the headline argument still promotes measured associations and unlike comparisons into established causal costs. Correct that boundary, including the recovery target, before coauthor sign-off.

## Round 2

**1. Numbers and claims**

No new numerical mismatch found against the tables and previously read evidence cards. The recovery target, median multipliers, prospective sample sizes, corpus-size wording, and time units are corrected.

Two substantive scope problems remain:

- The abstract pairs **13%/11% encoding-plus-search time** with **90.5%/81.4% quality** as though these were jointly measured operating points. Table 2 extrapolates timing using the million-passage collection at each path’s own within-1% threshold; Table 1 reports exact-search quality on BEIR-15. Both results are valid, but their conjunction needs that distinction. Section 6 repeats it.
- Section 1 still says co-training establishes the frozen-index method’s “ceiling.” Section 2 now correctly states that the different models and protocols **do not isolate the quality cost of freezing the index**. Those claims contradict each other.

**2. Essential changes**

1. Current: “In the timed Constella instance the extra search consumes part of the encoding saving: Nano and Zero need about 13% and 11% of Stella's extrapolated encoding-plus-search time at 90.5% and 81.4% of its quality, so query-encoding speedups measured alone overstate what a cheap query path saves.”

   Replacement: “In Constella, Nano and Zero require approximately 13% and 11% of Stella’s extrapolated encoding-plus-search time on the million-passage diagnostic, at search settings within 1% of each path’s own exact score. Separately, they retain 90.5% and 81.4% of Stella’s exact-search quality on BEIR-15.”

   Apply the same distinction to Section 6’s “at 90.5% and 81.4% of its quality” clause.

2. Current: “The capability is established; so is its ceiling, since methods that co-train the document tower retain more [LightRetriever].”

   Replacement: “The capability is established; its quality cost relative to joint training is not isolated by comparisons across different models and evaluation protocols.”

3. Current: “The answers run against the field's heuristic.”

   Replacement: “The results reject the project’s strongest-space selection heuristic.”

   The earlier paragraph now attributes the heuristic correctly; this sentence silently restores the unsupported field-wide attribution.

**3. Sign-off**

**No.** The one remaining blocker is comparative scope: the paper must distinguish results measured under different protocols rather than present them as a joint operating point or an established quality ceiling. The structure and narrative are ready; no further cuts are essential.

## Round 3

1. **Remaining mismatches:** None found in the requested passages against the previously checked tables and evidence cards. The abstract and Section 6 now distinguish extrapolated search-plus-encoding time from exact BEIR-15 quality. Section 1 no longer claims an established quality ceiling relative to joint training, and correctly attributes the selection heuristic to the project.

2. **Coauthor sign-off: Yes.** The comparative-scope blocker is resolved. The narrative supports the measured efficiency and compatibility claims without claiming quality superiority over joint training. No further essential changes.
