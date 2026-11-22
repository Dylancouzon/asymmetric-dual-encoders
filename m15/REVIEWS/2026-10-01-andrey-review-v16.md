# Andrey review, v16 draft (2026-10-01)

Target: `m15/PAPER.md` v16 at `bb2287a`. Checked: reference links (20 probed, one 403), figure dimensions and sizes (five PNGs, 620 to 1240 px tall, 67 to 164 KB), in-text citation keys against the reference list, the E2 encode ratio against its receipt.

if teacher quality is useless as a guide, why does the head track teacher quality inside a width band? is finding A a width effect wearing a selection story?

- §3 width paragraph — the paper already calls this a hypothesis, which is the right call, but the abstract and the finding-A summary sentence do not mention width at all. a skeptic reads the abstract, then finds the qualifier buried. put one clause in the finding-A summary: "poor standalone guide; within one width band it does track."
- references — ten entries were never cited in the text (HNSW, OOD-DiskANN, RoarGraph, QA-Cos, DBSF, Arabzadeh, QPP, Model2Vec, StaticEmb, NanoVDR). cite or cut. fixed in the same pass: each now has one in-text citation where it does work.
- references — the DBSF Medium link returns 403. link checker fail. replaced with the Qdrant hybrid-queries documentation page, which returns 200.
- abstract — "median four times the search budget" reads as time. it is `ef`. fixed to "graph-search budget (HNSW `ef`)".
- §4 "about 60-fold on these queries" — the receipt says 62.7 (1.611 / 0.0257 ms); Table 1 implies 51 on another sample. "about 60-fold" is fine, but say in the same clause that Table 1's sample gives 51, or a reader who divides Table 1 will think the numbers disagree.
- §4 query-precision paragraph — a Qdrant paper shows the engine's default discards query magnitudes and then does not test the setting the engine has for it. owner decision (recorded), but every reader will ask. keep the paragraph to one claim and one sentence of "not tested here", or expect the question in every thread.
- Figure 1 (`f2_per_dataset.png`) — the legend labels sit on the trec-covid row's points. move the legend.
- figures — sizes are fine (largest 164 KB, 1280 x 1240). no asset problem.
- §3 "Fitting a table student takes four to seven minutes per teacher on one GPU" — correct per E8 timing, but the number includes encoding 337,981 queries only because the A100 does 800 to 10,000 texts a second. on a smaller card the encode dominates. one clause: "on an A100, including target encoding".

no metric shopping found: the loss-based multiplier is reported first because it was registered first and the recovery measure is explained, not substituted. the registered contrasts Constella failed are in Appendix A in full.

verdict: fix-then-ship — the two findings are real, measured, and non-trivial; the fixes above are citation hygiene and one framing clause, not substance.
