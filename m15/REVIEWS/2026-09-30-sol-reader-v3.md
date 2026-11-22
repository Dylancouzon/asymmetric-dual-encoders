# Codex gpt-5.6-sol, target-reader review of PAPER.md draft v3

Date 2026-09-30. Brief: `briefs-2026-09-30/sol_reader_v3.md`. Access: named files only. Verbatim below; it drove the v4 restructure.

---

## 1. Would I keep reading?

Barely. The abstract promises six papers at once, while the introduction leads with the least memorable contribution: 100/90.5/81.4% retention. That is another efficiency frontier.

The idea I would remember is: **a strong retrieval tower is not necessarily a distillable tower; directly screening the cheap student predicts success better than the teacher’s benchmark score.** That is potentially useful new knowledge, but it appears fourth in the abstract and after the benchmark table. Lead with it. E2 may validate deployment relevance; E9/E10 may strengthen the adaptive-compute story, but neither fixes the present emphasis.

## 2. Where it becomes a benchmark report

Sections 3, 4, 7.1, and 8 are mostly score inventories. The abstract and introduction repeat those inventories. Section 8 reads like an experiment ledger, not an argument.

Replace them with:

- One motivating operator question: “The index cannot move; which queries deserve contextualization?”
- A worked figure with three real queries: tiers agree; word order/context changes the answer; Zero beats Nano. Show query, judged document, and rank—not just nDCG.
- Make the E8 scatter the central figure: teacher quality versus table quality, paired by family, beside dev-screen versus public score.
- Show the oracle/router quality–cost curve, with “both fail” separated from genuine ties.
- Move full per-dataset tables, latency inventories, and preregistration bookkeeping to appendices.

## 3. Unsupported or overstated statements

- “700-fold range of query compute”: you measured latency on one CPU/runtime, not compute.
- “Every dense result searches the same document vectors”: false for E8 and the LEAF/bge reference systems, which use their own document indexes.
- “Every new measurement … fits on two … forums”: E1, E4, and E5 do not fit anything.
- “Our bge-small, LEAF and BM25 reproduce their published numbers”: the result file proves your scores, not agreement with cited published protocols.
- “Zero costs about as much as tokenizing”: no tokenizer-only measurement.
- “Short and unfinished queries … loss belongs to the query, not the encoder” and “as good a search-as-you-type encoder”: causal/general claims from artificial cuts on three datasets. `MEASUREMENTS.md` explicitly says E4 cannot show real search-as-you-type behavior.
- “Most queries do not need the transformer”: the 15% result is a qrel-aware oracle matching a macro average. It does not identify individual need; 43% ties include queries where both systems score zero.
- “Tower quality predicted nothing”: rho −0.09 with ten heterogeneous, exposure-confounded checkpoints supports “no association in this roster,” not “nothing.”
- “Smaller checkpoints made better tables”: four observational pairs confound size, training, version, dimension, and readout. The later caveat does not repair the causal heading.
- “Four exposure-free sets”: the exposure file frequently says unknown or source-family contact, not exposure-free.
- The affine-absorbability result needs the derivation and exact normalization/pooling assumptions in the paper; a numerical check is not the proof.
- “Free” fertility costs 0.018 ms—about 41% of Zero’s 0.044 ms.
- “Registered test … reported in full”: the text says one reserved component is reported separately, but supplies no result for it.

The JSON receipts are traceable but not human-readable evidence. Release generated Markdown/CSV claim cards containing estimand, inputs, result, uncertainty, and limitations.

## 4. What is new?

Potentially new:

- Comparing a transformer and static table on exactly one frozen text index, then localizing their per-query disagreement.
- Prospective evidence that direct student screening transfers while teacher quality does not, specifically for static-table distillation.
- One-search blending of aligned query encoders, if E9 succeeds.

Not new: frozen large document encoders with cheap query encoders, embedding alignment, static lookup query encoders, or per-query routing. These are already central to [QED](https://arxiv.org/abs/2306.11550), [EmbedDistill](https://arxiv.org/abs/2301.12005), [LEAF](https://arxiv.org/abs/2509.12539), [LightRetriever](https://arxiv.org/abs/2505.12260), [NanoVDR](https://arxiv.org/abs/2603.12824), and [pyNIFE](https://github.com/stephantul/pynife). Retrieval-strategy routing also predates this work ([Arabzadeh et al.](https://arxiv.org/abs/2109.10739)). “Better teachers need not distill better” is established outside IR ([Cho and Hariharan](https://arxiv.org/abs/1910.01348)).

A hostile reviewer will say: LightRetriever already offers embedding-lookup queries and reports 95% retention; your paper contributes a careful case study, not a new architecture. The answer must be the diagnostic knowledge, not another frontier.

## 5. Recommended structure

1. **Introduction** — Frozen-index constraint and the distillability thesis.
2. **Related Work** — Explicitly distinguish the paper from the works above.
3. **Experimental Contract** — Index, tiers, data roles, registration, evidence artifacts.
4. **The Cost–Quality Frontier** — One concise figure, including E2.
5. **Which Towers Admit Cheap Queries?** — E8 as the central finding plus mechanism tests.
6. **Where Context Matters** — Oracle decomposition, word order, prefixes, worked cases.
7. **Can the System Exploit It?** — E9 blending, E10 routing, realized cost.
8. **Robustness and Limitations** — Preregistered contrasts, exposure, uncertainty.
9. **Conclusion** — Practical decision rule, not benchmark recap.

## 6. Cheap overnight measurements

- For each E8 tower, measure **token additivity**: cosine/R² between the tower query vector and the best bag-of-token reconstruction on held-out queries. Test whether this explains table quality better than teacher nDCG. That could turn the correlation into a mechanism.
- Decompose the 43% ties into both-zero, equal relevant rank, and genuinely similar top-10 results; include overlap and rank correlation.
- Report leave-one-tower/family-out stability for E8 rho and paired intervals for cross-tier prefix-retention differences.
- Fit one frozen multivariate router on the development forums using E10’s cheap features. Single-feature thresholds are unlikely to establish whether routing is viable.

Files opened: `m15/PAPER.md`, `m15/PLAN.md`, `m15/MEASUREMENTS.md`, `results/m15_e1_latency.json`, `results/m15_e4_prefix.json`, `results/m15_e5_oracle.json`, `results/m15_e6_router.json`, `results/m15_e8_towers.json`, `results/m20_beir15_run.json`, and `m15/e8_exposure.json`.
