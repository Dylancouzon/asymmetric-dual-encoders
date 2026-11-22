# Fable qualitative review: does v4 create new knowledge for the IR industry

Date 2026-09-30. Access: FINDINGS files and plain-English page; web search without reserved-dataset terms. Returned text below, verbatim. Dispositions are in `m15/LOG.md` (2026-09-30) and `m15/PLAN.md`.

---

**1. Verdict**

Partly. The plan already holds two things an advanced search engineer does not have: a quantified frontier for cheap query encoders on a frozen, off-the-shelf 400M tower (LightRetriever co-trains the document side; LEAF and arXiv 2306.11550 stop at small transformers), and the counterexample that a tower's own score does not predict what distills out of it. The rest is folklore with first numbers, or known. Executed as planned, the paper is a solid measurement paper with one actionable rule. It becomes a research paper if it turns the evidence it already has into three practitioner rules (section 3 below).

**2. Ratings**

Spine 1, retention as compute drops: KNOWN-BUT-UNQUANTIFIED. The shape is expected (KALE, 2306.11550 shows a 2-layer student at 92.5%, LEAF at 97.7%). New quantity: a lookup table keeps 81% of a frozen tower, and fusion recovers most of the gap. Note that Nano's 90.5% sits below both prior small-transformer results; the paper must say why (Stella is a 400M teacher, 1024d, versus 109M for LEAF) or a reviewer will read it as a weaker recipe.

Spine 2, distillability belongs to the tower: NEW for embedding towers. Overlaps with the general KD result that a bigger teacher is not a better teacher (Cho and Hariharan 2019; Dynamic KD, arXiv 2109.11295), so cite it and state what is new: the closed-form screen that ranks towers in minutes. One 2026 distillation paper surfaced in search (NanoVDR, arXiv 2603.12824, or a neighbour) reports the opposite, that the teacher's own nDCG is the strongest predictor of student retention. Locate it and confront it; a contradiction with a citation strengthens the counterexample.

Spine 3, the lookup table's limit: the absorbability proof is KNOWN in kind (All-but-the-top, SIF, whitening are linear; any linear post-process folds into a free table), but nobody has written it down for query tables, so keep it at half a page. The recipe-specific evidence is the stronger part and is NEW: under a frozen tower, no table-side lever moved development by more than 0.005, twelve probes ran, and the KL objective was inert at 99.75%. M8's sentence is the finding: what was never tested is co-adaptation of the document side, which is exactly what the system that beat the table does. That is the claim an engineer acts on. The plan buries it.

Spine 4, coverage decides retention: KNOWN. Every distillation paper says out-of-distribution retention drops. Compress to the per-component numbers and the rule "report retention per component". One line.

Spine 5, the pre-registered test: KNOWN method, unusual in cs.IR. Keep it in full, but its value is methodological, and the result is unresolved on held-out. Do not let it carry the paper.

Spine 6, silent swap failures: KNOWN. Prompt omission and padding without a mask are folklore every embedding user has hit. The one item with substance is M12's fusion depth inversion: a score-based operator wins at prefetch 1000 and a rank-free one wins at 10 to 50. That is KNOWN-BUT-UNQUANTIFIED and belongs in the body, not the appendix.

Use cases. High QPS: KNOWN; a lookup is faster than a transformer. The encoder's share of end-to-end latency (E2) is the only number worth printing. Search as you type: NEW if E4 runs; I found no measurement of prefix robustness for lookup versus contextual query encoders, and M7's length-flat curve (finding 7) supports the hypothesis. Per-request tiering: KNOWN as an idea (cascades), NEW if a routing signal is shown to work; see below. Edge: KNOWN; quantization enables edge. The one novel item is that lookup-query vectors are harder for ANN (m1-m6: LightRetriever loses 2.1 nDCG at default ef against 0.7 for bge-small). New query encoder for an existing index: KNOWN premise (LEAF, Drift-Adapter, BCT); the cost table is useful reference detail. No ML runtime: KNOWN.

**3. Missing, from existing evidence or E1 to E5**

- A router from existing rows. The plan says a real router needs a fixed policy on held-out queries. You have one. M8 measured the table falling 0.050 nDCG per +1.0 subwords per word (t = 4.61). Subword fertility is computable in microseconds before any encoder runs. Fit one threshold on development, evaluate on the 11 non-reserved sets from the per-query rows, and report what share of the E5 oracle headroom it recovers and what share of traffic stays on the 0.11 ms path. This is the measurement that turns tiering from a paragraph into a result, and it costs about as much as E5.
- Cosine to the teacher is not the metric. M7 finding 3: the highest-cosine candidate ranks sixth of ten on retrieval, and cosine rises with ridge lambda while nDCG falls. Practitioners validate distilled encoders by cosine to the teacher. One figure, existing data.
- Query length. M7 finding 7: the bag-of-tokens encoder is flat from 16 to 256 words once measured on nested prefixes of the same documents. Against folklore, and E4 extends it to prefixes. Put both in one section.
- Fragmentation is a correlate, not a lever. D2-PRE moved fertility by 0.17 and the metric did not follow. Pair it with the router result so the reader sees the same signal used correctly (routing) and incorrectly (vocabulary extension).
- E2 should measure quantization and ef recall per encoder, not just per collection. If table-produced query vectors lose more under binary quantization and HNSW than transformer-produced ones, that is a deployment rule nobody has published: cheap query encoders need a higher oversampling budget.
- Head width. M9's rank-bottleneck probe and M10's G1 (+0.0217 for 1152 versus 384) say that a 384-wide backbone distilling into a 1024d space needs a wider head, and the layer-concatenation trick provides it. One sentence in section 6, existing data.
- A mechanism for spine 2, optional and small: encode a few thousand development queries with each of the nine towers, shuffled and unshuffled, and correlate order sensitivity with table retention. If large models are more contextual and distill worse, spine 2 gets a cause. Minutes per tower, no training.

**4. Cut**

E3, the load test; 1/p50 caveats do not need a sweep to state, and E2 gives the encoder share. The swap box, down to two lines in the appendix. Spine 4 to one paragraph. The bge-small and LEAF reference points to a single row. The edge paragraph to the per-encoder quantization result; the 256 MB container is a demo, not knowledge. The "no ML runtime" sentence.

**5. The sentence**

Pick the tower for what distills out of it, not for its leaderboard score, because under a frozen document index the query side reaches its ceiling within 0.005 nDCG of a closed-form fit, and everything after that comes from the lexical channel or from routing.

The plan supports the first two clauses today. The third needs the router result from section 3.

Sources: [2306.11550](https://arxiv.org/abs/2306.11550), [KALE 2304.01016](https://arxiv.org/abs/2304.01016), [pyNIFE](https://github.com/stephantul/pynife), [Drift-Adapter 2509.23471](https://arxiv.org/abs/2509.23471), [LEAF 2509.12539](https://arxiv.org/abs/2509.12539), [LightRetriever 2505.12260](https://arxiv.org/abs/2505.12260), [Dynamic KD 2109.11295](https://arxiv.org/pdf/2109.11295), [NanoVDR 2603.12824](https://arxiv.org/pdf/2603.12824).
