

M15 · research paper plan v4 · 2026-09-30 · arXiv preprint (cs.IR)

## How much query encoder does a frozen document index need?

Dense retrieval pays for its document encoder once and for its query encoder on every request. The paper holds one document index fixed and shrinks only the query side, from the tower's own 400M-parameter path to a 34.5M student to a lookup table with no neural network. It then asks two questions: what survives at each step, and what decides where the quality goes.

Alternative title (Fable): "Retention as Query Compute Drops: Cheap Query Encoders on a Frozen Document Index." This version replaces v3, which read as a comparator report. Fable proposed the shape. Codex Astra checked every finding against the committed evidence and rated it. The ratings appear beside each finding.

## Draft abstract

Dense retrieval encodes a corpus once with an expensive document encoder, but pays for its query encoder on every request. We fix one document index built with stella_en_400M_v5. We then measure how much retrieval quality survives as the query encoder shrinks from the tower's own 400M-parameter path, to a 34.5M distilled transformer, to an int8 token lookup table with no neural network at query time. We also ask what decides where the quality is lost. On the 15 BEIR datasets of MTEB English retrieval, under exact search, the transformer keeps 90.5% of the tower's nDCG@10. The table keeps 81.4% at 0.11 ms per query on a CPU, and BM25 fusion lifts it to 87.9%. The losses have sources we can identify. A tower's own retrieval quality did not predict how well it distills into a table: of nine candidate towers, the one with the highest score produced the fifth-best table, and in each of four model families the base model distilled better than the large one. A mean-pooled table absorbs every post-pooling transformation exactly, so post-processing cannot raise its limit. [One sentence on system cost, after E2.] A pre-registered test placed the transformer above bge-small on the registered partition (+0.018 nDCG@10). On a one-shot held-out set, the difference was unresolved.

## Finding 1: what survives as the query side shrinks

Dense query encoderWith BM25 fusion (DBSF@100)BM25 alone
    [figure]

Retention is the BEIR-15 macro nDCG@10 of each configuration divided by the Stella query path's 0.5614. All configurations search the same frozen Stella document vectors, with exact search (results/m20_beir15_run.json). The encode p50 values come from one CPU protocol: batch one, four threads, 20-word queries. The Stella query path and the fused costs are the gaps that E1 and E2 fill.

Quality is uneven against compute. The first 0.11 ms of query compute buys 81% of the tower. The next 7 ms buys another nine points. The last 365M parameters buy the final 9.5. A lexical channel lifts the table by 6.5 points with no neural network at query time, and the table beats BM25 alone on 11 of 15 datasets. bge-small (0.5171) and LEAF (0.5402) appear in the paper once, as reference points for what a small query encoder on its own index reaches.

Retention is defined against the registered single-prompt Stella path. The Stella model card reports 58.97 with a different instruction per task, max length 400 and bf16. The paper states this, and it pins the card revision.

## Finding 2: where each tier loses

NanoZero
    [figure]

Retention of the Stella query path per dataset, sorted by Zero's retention. The table keeps 0.95 on Quora and Climate-FEVER and 0.67 on TREC-COVID. On FEVER, HotpotQA and Climate-FEVER, the table scores above the student. The x axis starts at 0.6.

The two tiers fail in different places. The table loses most on TREC-COVID, FiQA and SCIDOCS. TREC-COVID is also where BM25 alone beats it and where fusion lifts it most (+0.169), so the table's weakest set is the one where the lexical channel helps most. The student's losses concentrate on claim-style queries. The paper reports this pattern and its known confounds. Zero trained on FEVER-train and Nano's query pool excluded FEVER. Nano also starts from bge-small's weights. The paper claims no cause.

## Finding 3: the tower you index with decides what you can distill

    [figure]

Nine towers with complete rows. x is the tower's own nDCG@10 on two CQADupStack development forums. y is the score of a closed-form table fitted against that tower and its own documents (results/m7_learnability_report.json). Stella is blue. Development scale, and used to rank candidates rather than to predict final scores.

The tower with the highest score, arctic-embed-l at 0.4931, produced the fifth-best table. Stella, at 0.4806, produced the best one, keeping 71.6% of its own score against 43.2% for gte-large. In each of four families (bge, e5, gte, arctic), the base model distilled into a better table than the large one. That count includes two base towers whose own score was not measured, and the arctic pair mixes m-v1.5 with l v1. The rank correlation is between 0.00 and 0.14 depending on the candidate set. The paper states it as a counterexample to picking a tower from a leaderboard, not as a law. The practical rule for index builders: if you want cheap query tiers later, screen towers by a closed-form table fit, which takes minutes.

## The research spine

### 1 · Retention as query compute dropsadequate, descriptive
The frontier above, plus the system share of cost once E1 and E2 run. Without Stella's own latency, the paper has two measured cost points and no complete frontier.

### 2 · Distillability belongs to the towerweak n, strong counterexample
Nine pairs on two development forums. The finding is the counterexample and the base-versus-large pattern, not a zero correlation.

### 3 · The lookup table's limitstrong algebra recipe-specific evidence
Exact result: centering, whitening, top-component removal and IDF or SIF weights are absorbable into a freely parameterized mean-pooled table (max difference 9.3e-14). They cannot raise its limit. The evidence specific to our recipe stays separate: the KL term was inert (median 1.08e-07 nats, positive ranked first for 99.75% of 4,000 training queries), and no table-side lever improved development by more than about 0.005. The hard-candidate objective, still informative at 0.777 nats, was never trained. The paper says so. Lexical fusion is the lever that worked.

### 4 · A distilled encoder keeps what its queries coverdescriptive, not causal
The first student kept 93.8% near its training distribution and 50.1% on a programming forum. For the table, an out-of-domain development slice read 0.764 against a held-out 0.755, while the full development macro read 0.915. This is one retrospective agreement, and the paper reports it as that. Coverage is not separated from capacity.

### 5 · The pre-registered testregistered
This goes in the evaluation section, reported in full. Clean-4: Nano − bge-small +0.017648, established. The one-shot held-out NDO-3 (DBpedia and two CQADupStack forums): +0.0032 [−0.0069, +0.0134], unresolved. Under the query-pooled weighting the result is −0.0121. LEAF leads there by 0.0389. FEVER is reported separately.

### 6 · The swap fails silentlydocumented hazards
A half-page box. A Stella query encoded without its prompt lands at cosine 0.80 to the correct vector (m11/STATUS.md:150). Tokenizer padding to 512 drops Zero to cosine 0.35 (m11/STATUS.md:98). Float64 pooling widens the output. The fusion depth changes the ranking.

## What one index with a choice of query encoders makes possible

The discussion section. Each use case gets a paragraph only if a measurement supports it. Otherwise it gets one sentence.

Use caseIn the paper asEvidence that turns it into a resultWhat the evidence cannot show

 | High QPS | Paragraph | E3: sustained completed requests per second under an arrival-rate sweep, with p95/p99 including queueing and a fixed latency SLO. Measured for encoding alone and for full retrieval. | Service capacity of a real deployment. Note that 1/p50 is not throughput.

 | Search as you type | Paragraph, or a section if the curve is flat | E4: incomplete-query robustness. Public queries truncated to k words, scored against final-intent judgments and against the full-query ranking, per encoder. Encode cost per keystroke: 0.11 ms against 7.25 ms. | Real typing sessions. Intermediate intent, corrections and prefix relevance differ from final intent, so this is a synthetic test.

 | Per-request tiering | Paragraph | E2 interleaves the encoders on one unchanged collection, with a content hash recorded before and after. E5 bounds the headroom: the per-query oracle over existing rows. | The gain of a real router. That needs a fixed policy evaluated on held-out queries.

 | Edge and on-device | Paragraph | E2 on the M5 Pro with the released encoders and real vectors: memory, cold and warm latency, quantization recall. | Phones, battery life or thermal behaviour.

 | A new query encoder for an existing index | Paragraph with a cost table | Already measured by component: Zero retrains in about 20 min after 8 to 12 h of re-encoding, and Nano trained for 57.3 h on one A100 (about $95). | That this transfers to an arbitrary tower. One tower was tested.

 | No ML runtime at query time | One sentence | None needed: the table is a lookup and a mean. | 

## Paper outline

1Introduction. Documents are encoded once and queries forever. The question, and the three encoders presented as three cost points. Prior art in paragraph two: LEAF, pyNIFE, LightRetriever, arXiv 2306.11550, BCT/FCT, Drift-Adapter.1 p

2Setup. The frozen tower, how each tier lands in its space, exact search on BEIR-15, and the contact labels. Reference systems named once.1 p

3Retention as query compute drops. Finding 1 and its system cost (F1, F4).2 p

4Distillability belongs to the tower. Finding 3 above, with the closed-form screen as a practical tool (F3).1.5 p

5The lookup table's limit. The absorbability proof, the recipe evidence, and fusion as the lever that works.1.5 p

6Where each tier loses, and what the query pool decides. The per-dataset map (F2), coverage evidence, and the development-slice result.1.5 p

7The pre-registered test. Registration timeline, clean-4, the one-shot held-out result, weighting sensitivity.1 p

8What this makes possible. The use cases in the preceding table, each at the size its evidence allows.1.5 p

9Limitations. One tower, one seed, query-only intervals, English only, development-scale tower evidence.0.5 p

AAppendix. Full BEIR-15 table with contact labels, fusion dataset-dependence, the swap-contract box in full, statistics, registration and receipt hashes, closed avenues (six rows at most).~4 p

Out of the paper: M17, M18 and M19 (no result), the LightRetriever and OpenSearch bar contrasts, and the synthetic-index prototype as evidence of system cost.

## Measurements

None of these needs training or protected data. Each gets a dated method file and one Astra review before it runs, as the M15 mandate requires for new measurements.

IDWhatForStatusTime

 | E1 | Encoder latency and per-core throughput for Zero, Nano and the Stella query path in one harness on the M5 Pro, in three query-length buckets. | Finding 1's missing cost point | necessary | ~2 h

 | E2 | A real 1M-document Stella collection from the M20 archive in Qdrant 1.18. The encoder's share of end-to-end latency. Original vectors against int8, binary and TurboQuant 4-bit, with recall@10 against exact search reported as neighbour recovery. Encoders interleaved on the unchanged collection. BM25 and DBSF run in Qdrant for the fused costs. | Finding 1, tiering, edge | necessary | ~1 day

 | E3 | Load test: an arrival-rate sweep with a p99 SLO, for encoding alone and for full retrieval. | High QPS | if we keep the paragraph | ~3 h

 | E4 | Incomplete-query robustness on public, non-reserved sets (SciFact, NFCorpus, FiQA, Quora, NQ): k-word prefixes, per encoder. | Search as you type | if we keep the paragraph | ~3 h

 | E5 | Per-query oracle headroom between Zero and Nano, from existing per-query rows on the 11 non-reserved datasets. | Tiering | cheap | ~15 min

## Release

The models, the registration trail and the public repository already exist. Two items would make the paper something others build on. The first is the paper source, with one command that regenerates every table and figure from committed JSON. The second is the Stella BEIR-15 document vectors on Hugging Face with a scoring script, so anyone can test a new query encoder against the same frozen index without the 40-hour encode. Quora, Climate-FEVER and MS MARCO need a licence decision first. A full "bring your own tower" kit is deferred: it means about 4,500 lines to extract and 600 to 1,000 new lines.

## Decisions for you

### Run the measurements?
Recommend: E1, E2 and E5 now. E3 and E4 only if you want the high-QPS and search-as-you-type paragraphs. E4 is the one that could produce a surprising result, because Zero has no notion of word order and may degrade gracefully on prefixes. Blocker: E2 needs about 2 GB of Stella vectors copied from the RTX box's D: drive to the Mac. Can you do the transfer, or give me SSH to the box?

### Publish the Stella vectors as a benchmark?
Recommend: yes, on the Qdrant Hugging Face org, with the three licence-unclear corpora left out or gated. The licence question may need Qdrant legal.

### Title, authors, affiliation, endorsement
Recommend: the question title, and Qdrant as the affiliation. arXiv asks first-time cs.IR submitters for an endorsement. Do you or a coauthor already have one?

Done today, not committed: CLAUDE.md now says the paper leads with research findings, and the registered test stays in the evaluation section in full. instructions-m15.md carries a dated amendment. The Zero card's "L2 regression" error is logged as M22 item 3 in instructions-m22.md. The frozen M20 registry's headline line is left untouched, because its hash is pinned in the result files. Today's ruling overrides it for presentation only.

