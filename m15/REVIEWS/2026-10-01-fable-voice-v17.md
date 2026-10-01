# Fable (fresh agent) voice-contract check of v17 (2026-10-01)

Brief: `briefs-2026-10-01/v17-voice-contract.md`. Verbatim.

**1. Contract check**

- One thread: FAIL. Section 6's third paragraph leaves the thread for a grab bag: "Fusing Zero with BM25 recovers more quality than any table-side tuning did (BEIR-15 macro 0.4572 to 0.4933) at the cost of a second index and retrieval". Fusion, routing, and the precision control answer neither RQ.
- Hypothesis first: FAIL. 4.4 opens on a table: "**Table C.** Six-set nDCG@10 lost against the best student in the roster, by selection rule." 5.3 opens on a result, with no prediction stated.
- Evidence in tables, argument in prose: FAIL. 5.2's reading paragraph restates Table D row by row, and Section 6 inventories: "to stay within 1% of their own exact score, Stella needs `ef=128`, Nano 256, and Zero 512, with search medians of 1.53, 2.18, and 3.58 ms."
- Effect sizes, not adjectives: PASS, one slip in the introduction: "on a 25,657-document corpus the gap is smaller".
- Third person, declarative: FAIL. "Width is what the index decides; the screen is how a team should decide." This is the takeaway box with the border removed.
- Surprise allowed: PASS. "The strongest index yields one of the weakest students" sits in the abstract with the effect sizes.
- Limits once: PASS. One caveat sentence closes Section 4 and one closes Section 5; 4.3 and 5.2 each carry one local caveat.
- Length: FAIL by the contract's own number, 4,199 main-text words.

**2. Register**

Mostly a paper now. Three sentences still read as blog or benchmark report: "For many teams the index is effectively immutable." (asserted, no source); "The results are these." (blog cadence); "On one binary-quantized collection the fastest setting within 1% gives Zero 0.6116 at 1.11 ms, Nano 0.6893 at 2.48 ms, and Stella 0.7226 at 21.3 ms" (benchmark table in prose).

**3. Argument**

The thread holds from the abstract through 4.3 and breaks twice. First, the frame promises "what the index decides", but RQ1's predictor is the encoder's output width, a property of the model, scored by exact search on six small sets; nothing index-specific (corpus, graph, quantization) enters RQ1. "Each model is an index in the sense of RQ1" is a renaming an IR reviewer will reject; the frame fits RQ2, where the graph is real. Second, RQ1's conclusion undercuts its own title: the two-variable model loses 0.25 and 0.11 nDCG at selection and the dev screen wins, so the index "decides" a floor, and the screen decides the pick. The paper should say that plainly as the finding instead of leaving it to the "should" sentence. Section 6 then reads as an appendix stapled into the body.

**4. Numbers**

- Section 6: "Stella is a 1024-wide index with the best own score among its width band" is false. Table A has gte-large-en-v1.5 at 0.5970 in the same band; Stella is second.
- Section 4.2: "Adding nine family indicators" is wrong. The M3 model has 10 standardized betas, so eight indicators for nine families.
- Abstract: "+0.69 for quality and −0.83 for width" and "ranks 16 held-out indexes at Spearman 0.85" are the head's numbers only; the table gives +0.63/−0.64 and 0.70, and the abstract never says which representation it quotes.
- Abstract and introduction: "a median four times the search budget" is the secondary recovery multiplier on two corpora. The pre-specified primary measure has medians 4.0, 1.25, and 0.25 across the three workloads. Quoting the secondary measure as the headline without saying so is a claim stronger than Table D.
- Section 4.4, "one of the weakest students": true for the table (rank 22 of 26), weak for the head (0.2229 is below median, with eight indexes under it).
- All other Section 4 and 5 numbers match the JSON and C16/C17, including regret, partial correlations, R², and leave-one-family-out values.

**5. Cuts, to about 3,800**

- Section 6, third paragraph: move fusion, routing, and the precision control to Appendices C and D with one pointer sentence (about 180 words).
- Section 5.2 reading paragraph: keep the sentence on TREC-COVID's noise floor and the head's agreement; drop the Table D restatement (about 90 words).
- Section 4.1: delete the numbers already in Table A, keep the correlation and the interval (about 60 words).
- Section 2, first paragraph: one sentence per prior system (about 70 words).
- Related work's second paragraph can lose its final clause; Section 3.4's tie rule can move to Appendix C (about 50 words).

**6. The one change**

Rename the object. Report RQ1 as a property of the teacher encoder (width versus own quality) and reserve "index" for RQ2, where the HNSW graph is measured. A reviewer who sees a model width called an index property in the title will stop reading at Table A; the result itself, width beating quality with a held-out 0.85, is citable as written.
