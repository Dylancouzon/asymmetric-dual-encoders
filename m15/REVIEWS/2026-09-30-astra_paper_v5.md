# Codex review (astra_paper_v5) of PAPER.md draft v5 (be061e7)

Date 2026-09-30. Access: named files only. Verbatim below; dispositions in m15/LOG.md.

---

1. **“Every table and figure regenerates from committed JSON (`python m15/figures/make_figures.py`).”** `results/m15_e2_ann_msmarco1m.json` is absent from `be061e7` and currently untracked. Section 4.2’s principal evidence therefore is not committed. **Fix:** commit the cited result before making this reproducibility claim.

2. **“On the six public sets the frozen blend lowered Nano by 0.0051 (95% interval -0.0092 to -0.0011), and by 0.0070 on the four sets without disclosed exposure (`results/m15_e9_blend.json`, registered).”** The numbers match, but the registration disclosure is incomplete. `m15/MEASUREMENTS.md` records a pre-freeze SciFact peek: Nano **0.7211 → 0.7193** at weight 0.1. **Fix:** disclose the peek and unchanged dev-only selection; label the result registered descriptive.

3. **“That random move costs 0.009 nDCG@10 on average, against 0.047 for shuffling, and its damage does not predict Zero's gap (0.07).”** These values, and the table’s partial correlation **0.42**, concern **3,069 Stella-positive queries**, not the surrounding **3,727-query** population (`results/m15_e12b_fragility.json`). **Fix:** identify the restricted population and report “correlation 0.066” instead of asserting non-predictiveness.

4. **“Shuffling hurts because it changes what the query means, and those are the queries a table gets wrong.”** E12/E12b establish associations, not semantic causation. Their source values are **ρ = 0.456**, partial **ρ = 0.417**; `m15/EVIDENCE.md` explicitly says shuffling does not isolate word order from other properties. **Fix:** describe association with shuffle-induced loss; qualify the corresponding abstract and introduction claims.

5. **“So the table carries no signal the transformer lacks on these sets, and the same two development forums that predicted the tower ranking in Section 5 did not predict a blend weight.”** A failed fixed blend cannot establish absence of complementary information. `results/m15_e9_blend.json` gives **−0.005076** macro but **+0.004807** on ArguAna. **Fix:** say the dev-selected weight did not improve six-set macro quality.

6. **“Escalating routed queries to the Section 7.2 blend instead of plain Nano changes nothing (+0.0164).”** `results/m15_e10_router.json` gives router quality **0.483431 versus 0.486545**, a **−0.003115** change. The quoted **+0.016420** is against the blend’s own random baseline, not plain-Nano routing. **Fix:** report the direct difference descriptively; remove “changes nothing.”

7. **“The cost follows from Section 4: at the 25% budget the margin router pays Zero's encode and search on every query and Nano's on 28% of them, 2.7 ms per query against 3.8 ms for always-Nano with unquantized search at `ef` 128.”** These are modeled costs combining E1 medium-query medians, MS MARCO search medians and an equal-dataset routing share (`m15/MEASUREMENTS.md`; E10). **28.21%** is a macro share; pooling the six sets gives **23.77%**. **Fix:** label the **2.7116/3.7827 ms** figures cost estimates and specify the weighting.

8. **“Finally, the table's queries make approximate search work harder, so its 50- to 60-fold encode saving shrinks to a 2- to 3-fold end-to-end saving inside a vector search engine.”** The range is specific to the MS MARCO diagnostic and comparison with Nano. E2 gives **2.23–3.12×** there, versus **3.76–4.26×** on FiQA. **Fix:** name the comparator, collection and relative-loss targets; qualify the introduction and conclusion likewise.

9. **“On these three sets, a larger query encoder does not recover measurably more from a truncated query than a lookup table does.”** E4 supplies descriptive retention estimates, not an equivalence result; the 75%-word cut is **0.84644 versus 0.81248**. **Fix:** state that selected macro retention ratios were similar. Remove the abstract’s stronger “loses no more” inference and introduction’s “same share.”

10. **“Vector fidelity, the quantity a distillation loss optimizes, is not ranking fidelity, which is why only a ranking screen predicts.”** This unfixed v4 sentence remains after the qualified replacement. E8/E11/E11b report descriptive correlations at **n = 10**, including **−0.418** for error/margin and **0.903** for the screen. They establish neither exclusivity nor causation. **Fix:** delete this residual sentence and qualify equivalent abstract/introduction assertions.

11. **“Post-processing cannot fix either.”** `m15/PROOF_ABSORB.md` proves unchanged representational capacity and explicitly permits improvement through initialization or priors. **Fix:** delete this sentence; retain the narrower capacity statement.

12. **“Each M15 result file (`results/m15_*.json`) carries a receipt: script hash, git commit, model and dataset revisions, seed and machine; earlier results (BEIR-15, the held-out test, the tier builds) carry the provenance records of the milestone that produced them.”** The narrowed claim remains false: E8 frozen lambdas have no receipt; E12/E12b receipts have empty `inputs` and no model/dataset revisions. **Fix:** identify actual provenance locations and exceptions.

Files opened:

- `m15/`: `PAPER.md`, `EVIDENCE.md`, `PROOF_ABSORB.md`, `RELATED_WORK.md`, `e8_exposure.json`, `MEASUREMENTS.md`, `PLAN.md`, `EVIDENCE_INDEX.md`; both named review files.
- `results/`: all 22 JSON files explicitly permitted in the request.
- `m21/BENCHMARKS.md`, `m12/FINDINGS.md`, `m7/RECIPE.md`, `m11/STATUS.md`, `m11/release/zero_encoder.py`.
