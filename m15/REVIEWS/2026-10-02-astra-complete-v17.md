1. **Numbers.**

- §4.3: “Mean absolute error of the predicted six-set score is 0.032 for both.” E24 `stats`: head **0.031457 → 0.031**; table **0.032154 → 0.032**.
- §5.3: “We declared eight features.” E22 lists **nine non-covariate features plus two covariates**. Table F omits the query-effective-rank ratio and both covariates without identifying itself as selective.
- §5.3: “It ranks the penalty on every workload.” SCIDOCS encoder-ratio correlation is **−0.319, CI [−0.680, +0.126]**; student ratio **−0.448 [−0.754, +0.023]**. Neither establishes ranking there.
- §5.3: “The student-minus-encoder differences … do not rank it.” Pooled ratio-delta correlation is **−0.234 [−0.424, −0.033]**; margin-delta **−0.285 [−0.478, −0.054]**. Say they fail held-out-family ranking, not correlation.
- §5.3: “the penalty is a property … which the student inherits and worsens.” E22 establishes prediction, not inheritance or mechanism.
- §5.4: “every 384-wide space shows no gap at all.” C21 gives bge-small **−0.02 pp**. Teacher recovery at ceiling also does not imply “nothing left for the student to lose.”
- §5.4: “Corpus size therefore explains most … the remainder is domain or query form.” E23 establishes a **+1.770 pp** change and **−0.423 pp** residual in one subsample; C21 explicitly excludes causal identification.
- §5.5: “H2d holds” establishes the predicted deficit, not its proposed placement mechanism. “a nine-fold gap that the extra `ef` consumes” has no validated latency counterpart; C19 explicitly excludes that conclusion. §6’s “same accounting applies” repeats the overreach.
- §6: “Nano runs three layers … at about a fiftieth of Stella’s cost.” Appendix A specifies readouts from layers **12, 8, 4**, not execution of three layers. Table 1 gives **31.6/2.25 = 14.0-fold**, not fifty-fold.
- §6: “target encoding included” needs Appendix B’s **cached Stella exception**.

Appendix B’s roster, prospective scores, and hash match the supplied evidence. Other §6/Appendix B quantities lacking sources in the prescribed evidence remain unverified.

2. **Contract.**

- **One thread: fail.** “an application can match the path to the moment: a table lookup while the user is typing…” introduces an unmeasured product scenario.
- **Hypothesis first: fail.** §5.2 opens with “Table D. Search effort of the table student…”; its prediction follows the table.
- **Evidence in tables, argument in prose: fail.** “The strongest-space rule loses 0.285 and 0.152 on the roster and 0.086 and 0.107 on the prospective seven.”
- **Effect sizes, not adjectives: fail.** “Methods that train both towers at once reach higher retention.”
- **Third person, declarative: pass**, treating the owner-selected title as an explicit exception.
- **Surprise is allowed: pass.**
- **Limits once: fail.** “One caveat: one 1.5-billion-parameter model, our reproduction of its dense path, two corpora.” Section 7 repeats it.
- **Length: fail.** At approximately 5,100 words, it exceeds the ceiling. The inventory sentence quoted above exemplifies why.

3. **Coherence.**

Yes: frozen-index substitution → quality prediction → search penalty → deployed instance. Keep all four additions. §4.3 supplies essential prospective evidence, but mixes chronology, protocol, and results. Table F answers predictability; its geometry references to §5.2 are broken. §5.4 tests an alternative explanation. §5.5 extends scope, then repeats the speed-accounting argument in §6. Section 6 is the main digression and repetition source.

4. **Cuts.**

Ranked by approximate savings with zero substantive claims lost through retained tables or appendix relocation:

- Abstract: remove generic opening exposition and repeated model statistics; **180 words**.
- Introduction: compress the “Two properties…” and “Against RQ1…” paragraphs; **220**.
- §6: relocate per-dataset narrative, typing scenario, and construction-cost detail to existing appendices; **300**.
- §4.4: replace the table-reciting paragraph with two interpretive sentences; **100**.
- §4.2–4.3: relocate standardization/bootstrap definitions and prospective execution chronology; **130**.
- §5.5: remove table recital, speed speculation, and duplicated caveat; **100**.
- Consolidate repeated limitations; **70**.

Total: approximately **1,100 words**. Cutting table rows does not solve prose length.

5. **Abstract first sentence.**

“The strongest encoder in our registered comparison produced the weakest token-table query student, while cheap query paths required up to four times the median HNSW search budget on two corpora.”

6. **Sign-off.**

No. The blocker is evidentiary overreach: correlations become mechanisms and `ef` becomes latency.

Process disclosure: a faulty extraction also displayed E25 row arrays despite the restriction. I did not use them for these findings. No files changed.

Codex session ID: 01a0f94a-ef58-7ad2-9c24-1ac786c6560d
Resume in Codex: codex resume 01a0f94a-ef58-7ad2-9c24-1ac786c6560d
