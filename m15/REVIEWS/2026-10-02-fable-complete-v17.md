# Fable (fresh agent) review of the complete v17 (2026-10-02)

Brief: `briefs-2026-10-02/v17-complete.md`. Verbatim.

Review of m15/PAPER.md v17 against m15/EVIDENCE.md C16-C22 and results/m15_e22_gap_predictors.json, m15_e23_fiqa25k.json, m15_e24_prospective.json, m15_e25_lightretriever.json (named keys only; no rows read, no protected paths touched).

## 1. Numbers

Tables B, C, D, E, F, the Appendix B roster, the prospective table, and the E23 figures match their sources. Mismatches and overclaims:

- Abstract vs 5.2 contradict each other on SCIDOCS. Abstract: "by the pre-specified relevance-loss measure the penalty is universal on one corpus and mixed on the other two." Section 5.2: "By the primary measure that holds on FiQA (25 of 25...) and on SCIDOCS (17 of 25), and it is mixed on TREC-COVID (12 of 25)." Table D: 17/8. Pick one reading; the introduction repeats the abstract's.
- Abstract and intro: "matching the encoder's recovery takes a median four times the search budget" / "with a median recovery multiplier of four". Table D: 4.0, 2.0, 4.0. Median four holds on two of three workloads.
- Abstract: "recovers 3.6 points fewer exact neighbors ... and needs four times the search budget". That is the FiQA web-search row only; Table E spans -0.7 to -3.6 and one 2x.
- 5.3: "It ranks the penalty on every workload." SCIDOCS Spearman for the encoder ratio is -0.32 with interval [-0.68, +0.13] (E22 univariate), which includes zero.
- 5.3: "The student-minus-encoder differences ... do not rank it." Hubness delta +0.40 [+0.20, +0.57], margin delta -0.28 [-0.48, -0.05], ratio delta -0.23 [-0.42, -0.03] all exclude zero over 75 points. They rank weakly and fail held-out; say that.
- 5.3: "We declared eight features". Table F has eight rows; Appendix C declares nine. The ratio of student-to-encoder effective rank (+0.18 [-0.06, +0.41], LOFO -0.30, in E22) is missing from Table F.
- 5.3: "predicts its gaps no better than the fold mean". Leave-workload-out MAE 2.55 against baseline 2.33: worse, not equal.
- 5.5: "a nine-fold gap that the extra `ef` consumes." No latency was measured; the Section 5 caveat says ef is not latency.
- 6: "Section 5.5 shows the same accounting applies to a jointly trained lookup path". 5.5 reports ef multipliers, no time accounting.
- 6: "it sees context at about a fiftieth of Stella's cost". Table 1 gives 31.6/2.25 = 14-fold; parameters give 12-fold. No source gives 50.
- 6: "both are about a ninth of the full encoder's cost" at 13% and 11%; Nano is an eighth.
- Table A: table-student range at 384 "0.291"; gte-small is 0.2903.
- 4.3: "is listed, not replaced". e5-base-unsupervised appears in no table (see sign-off).
- 5.4: "every 384-wide space shows no gap at all"; bge-small-en-v1.5 is -0.02.
- Not in the cards I was given, unverified: "+0.09 for the table (95% checkpoint interval -0.39 to +0.52)", "raises in-sample R² to 0.91 and 0.81", "24 and 23 of 25 spaces", "0.2% of nDCG at ef=64", the recovery-multiplier medians in Table D.

## 2. Contract

1. One thread: fail. "an application can match the path to the moment: a table lookup while the user is typing, the small transformer when typing pauses, and the full encoder on submit." Self-declared "not a measured result".
2. Hypothesis first: pass.
3. Evidence in tables, argument in prose: fail. 4.3 is a number inventory with no table: "ranks the 16 spaces scored later at Spearman 0.85 [0.53, 0.99] for the head and 0.70 [0.20, 0.96] for the table; the space's own score alone ranks them at 0.37 and 0.16." Same in 4.4 ("loses 0.285 and 0.152 on the roster and 0.086 and 0.107"), 5.4, and Section 6 ("Zero 0.6116 at 1.11 ms, Nano 0.6893 at 2.48 ms, and Stella 0.7226 at 21.3 ms").
4. Effect sizes, not adjectives: pass.
5. Third person, declarative: fail on the letter. "We fit query students to 26 public encoder spaces" throughout; and a sentence addressed to an objector: "Neither the model's published retention nor its encoding-speedup figure is in dispute". Owner to rule on "we".
6. Surprise: pass.
7. Limits once: fail. 5.5 "One caveat: one 1.5-billion-parameter model, our reproduction of its dense path, two corpora." is repeated in 7 and followed by a second Section 5 caveat.
8. Length: fail, 5,100.

## 3. Coherence

RQ1 reads as one argument through 4.3. 4.4 restates 4.3's regret numbers in prose and mixes roster and prospective rows in one table. 5.2 to 5.5 lose the thread: 5.3 answers "predictable?" but Table F spends eight rows on features that fail, and the prose cites "the two geometry summaries of Section 5.2", which 5.2 no longer presents. 5.4 is a robustness check for SCIDOCS, not a step in the argument. 5.5 is the strongest new result because it closes the post-hoc-fitting objection; it sits after two detours. Section 6 then re-opens the product article (cascade, binary quantization, fusion pointer). The 0.85/0.70/0.89/0.71 quartet appears three times before Section 4.

## 4. Cuts (words saved / claims lost)

1. Section 6: merge paragraphs 2, 3, and "The instance also shows RQ2's cost in time" into one paragraph keeping Table 1, Table 2, ef 128/256/512, and the 2.2-fold figure. About 330 / 0.
2. Intro paragraph "Against RQ1 ... (Section 5.5)": replace with two sentences, no numbers. About 140 / 0.
3. 5.4 to one sentence in 5.2; Appendix C already holds the method. About 150 / 0.
4. 4.4 prose: keep the dev-screen sentence; delete the restatement of Table C. About 100 / 0.
5. Table F: keep the two ratio rows, all-features, baseline; the rest to Appendix C with the missing ninth feature. About 90 / 0.
6. 5.5: fold the timing, "not in dispute", and caveat sentences into one. About 70 / 0.
7. Related work: drop the 95%-vs-81.4% sentence (Section 6 carries it). About 45 / 0.
8. Intro: "The same cheapness is what makes the substitution worth considering in production..." (Section 6 repeats it). About 40 / 0.
9. Artifact availability: fp16 sentence to Appendix A. About 40 / 0.

Total about 1,000 words, landing near 4,100.

## 5. Abstract first sentence

"A query encoder distilled onto a frozen dense index inherits the index's width as a penalty: across 26 public encoder spaces, width predicts student quality at -0.83 standard deviations against +0.69 for the index's own retrieval quality, so the strongest index yields the weakest student."

## 6. Sign-off

Not yet. One thing: the prospective test committed eight encoders by hash and reports seven in both rows. A table-fit convergence failure explains dropping e5-base-unsupervised from the table row; it does not explain dropping it from the head row, where the fit is a different solve. With n=7 and the table interval at [-0.12, +1.00], the head row is the paper's only clear prospective result. Show the pre-specified rule that excludes a table-gate failure from both students, or report the head row at n=8 with its predicted and actual values. Fix the SCIDOCS contradiction and the "fiftieth" figure in the same pass.
