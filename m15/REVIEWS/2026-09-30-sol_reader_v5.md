# Codex review (sol_reader_v5) of PAPER.md draft v5 (be061e7)

Date 2026-09-30. Access: named files only. Verbatim below; dispositions in m15/LOG.md.

---

## 1. Would I keep reading?

Yes—unlike v3. The memorable idea is: **teacher retrieval quality is a poor guide to whether a static query encoder will work; screen the student directly.** The title and introduction now lead with it.

The abstract still dilutes that idea by cramming in the frontier, tower screening, oracle, routing, blending, prefixes, and ANN behavior. Start with the surprising result, then give one sentence each on mechanism and deployment consequence. The 100/90.5/81.4 frontier is context, not the discovery.

## 2. Where it becomes a benchmark report

- §4.1–4.2 becomes an inventory of systems, quantizers, `ef` values, and dataset results. Keep Figures 1 and 4, but reduce the prose to the operational surprise: **cheaper encoders produce harder ANN queries**.
- §8 is an experiment ledger. Move the full preregistered contrast table to an appendix; retain one paragraph explaining what survived the held-out test.
- §6.3 feels like a separate prefix-robustness note and interrupts the stronger word-order story.
- §7.1’s five-feature table is dry. Center the section on the failure of query-only signals and the partial success of post-retrieval confidence.
- Figure 5’s title, “most queries do not need Nano,” overstates a qrel-aware oracle. Rename it “Upper bound from qrel-aware routing.”

What is still missing is the worked example requested in v3: show an actual order-sensitive pair—ideally an argument reversal or negation—with Zero/Nano/Stella top documents and ranks. The current drug examples illustrate fragmentation, not the paper’s claimed dominant failure.

## 3. Claims that outrun their evidence

- “Tower quality told us nothing” / “does not predict”: this is $\rho=-0.09$ over ten heterogeneous checkpoints, without uncertainty. Say “did not rank table quality in this roster.” On clean-4 the reported association changes to +0.139.
- “The screen predicted the public ranking”: $\rho=0.90$ is encouraging, but it is one screen domain, ten towers, and falls to 0.685 on clean-4. It needs leave-one-family-out stability or softer wording.
- “A few-minute screen” is not established by the evidence card. The E8 receipt covers 2h46 for the eleven configurations; separate encoding, fitting, and scoring wall times should substantiate the deployable-screen claim.
- “Smaller checkpoints made the better table” is observational and confounded by dimension, training and, for Arctic, version. “The smaller-named checkpoint scored higher in four observed pairs” is supportable.
- “Three mechanisms do not explain it” means three scalar proxies correlated weakly in ten cases—not that the mechanisms were excluded.
- “Shuffling hurts because it changes what the query means” is causal language. Shuffling is not a semantic intervention, and an isotropic vector perturbation is not a matched control. The data show association with shuffle sensitivity.
- “Most of the loss” should include the actual loss decomposition, not only a correlation and two group means.
- “Post-processing cannot fix either” exceeds Appendix A. The proof covers affine maps and fixed token weights as representationally absorbable; it does not exclude nonlinear, count-dependent, tokenizer, n-gram, or optimization improvements.
- “Most queries do not need the transformer” confuses an oracle matching an aggregate with identifiable per-query need; 17% of queries are ties because both score zero.
- “The table does not know, from the query alone, when it will fail” was tested with only three weak query-only features.
- “The table carries no signal the transformer lacks” is not implied by one failed global convex blend.
- Similar prefix retention is not equivalence, and three artificially truncated datasets do not establish behavior on real unfinished queries.

## 4. What is new?

Frozen document indexes with aligned cheap query transformers are established by [QED](https://arxiv.org/abs/2306.11550), [EmbedDistill](https://arxiv.org/abs/2301.12005), and [LEAF](https://arxiv.org/abs/2509.12539). Embedding-lookup query encoders are established by [LightRetriever](https://arxiv.org/abs/2505.12260). More dangerously, [pyNIFE](https://github.com/stephantul/pynife) already advertises static queries over an unchanged teacher index, dynamic fast/slow paths, and the hypothesis that some queries require contextualization. Query routing predates this paper ([Arabzadeh et al.](https://arxiv.org/abs/2109.10739)); stronger teachers failing to produce better students is already explicit in dense IR ([PROD](https://arxiv.org/abs/2209.13335)).

The defensible novelty is the **controlled cross-tower comparison under one static-student recipe**, the transferred student-screen ranking, the per-query failure localization, and the observation that the static tier incurs a larger ANN penalty.

A hostile reviewer will call this a careful case study whose main recommendation—select models on development performance—is unsurprising, with an underpowered, exposure-confounded ten-tower sample and no new architecture.

## 5. Best structure

1. **Introduction** — Frozen-index decision and the distillability thesis.
2. **Prior Work and Novelty Boundary** — State plainly what QED, pyNIFE, and LightRetriever already do.
3. **Experimental Contract** — Index, tiers, data roles, registration, evidence artifacts.
4. **A Strong Tower Need Not Yield a Strong Static Student** — Make Figure 3 the central result.
5. **What the Table Cannot Represent** — Proof, controlled failure analysis, and a worked reversal example.
6. **Can We Identify Queries That Need Context?** — Oracle, routers, and honest limits.
7. **Does Cheap Encoding Survive Retrieval?** — Compact frontier plus ANN penalty.
8. **Robustness and Limitations** — Prefixes and preregistered contrasts; full tables in appendices.
9. **Conclusion** — “Screen the student,” with scope conditions.

## 6. Cheap overnight measurements

- Fit a hashed bigram or position-bucket residual table. Recovery specifically on shuffle-sensitive queries would turn the mechanism from correlation into an intervention.
- Bootstrap E8 per query and report leave-one-family-out $\rho$ distributions for screen versus tower score.
- Run multiple shuffles plus a small controlled reversal/negation suite; show retrieved documents, not only nDCG.
- Condition ANN recall on nearest-document cosine/local density to test the proposed geometry mechanism.

Files opened: `m15/PAPER.md` at `be061e7`; the v3 review; `PLAN.md`, `MEASUREMENTS.md`, `EVIDENCE.md`, `PROOF_ABSORB.md`, `RELATED_WORK.md`; E2, E4–E6, E8–E12b result JSONs listed in the prompt; and Figures 1, 3, 4, and 5. No edits made.
