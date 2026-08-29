# M8 literature sweep — three targeted questions (2026-08-29)

Read first (per brief): `research/lightretriever.md`, `research/landscape.md`, `research/literature.md`,
`research/m7-novelty.md`. Also found and read `research/m8-planning/opus-literature-2026-08-28.md`
(an existing M8 Opus sweep) — its §6 (co-training) and §3 (n-gram/vocab) cover ground that overlaps
Q1 and Q3 here; this file cross-checks that sweep's numbers (they match, see below) and adds what it
didn't cover: the VDR vocab-size ablation, the multi-word-token retrieval paper, the fertility→MRR
literature for Q2, and the representation-degeneration mechanism for Q3's negative-result case.

Caveat carried over from every research file in this repo: several numbers below came from
LLM-summarized fetches of PDFs that partially failed to decode (ChEmbed, "On the Effect of
Low-Frequency Terms", the raw VDR PDF). Flagged inline wherever that happened; treat those as
"reported, not independently re-verified" until someone re-pulls the table.

---

## Q1 — Co-adapting the document tower to a bag-of-embeddings query side

### Short answer

**No paper does both (a) and (b).** LightRetriever, the one system that jointly trains a document
tower against a bag-of-embeddings query side, never runs the frozen-document ablation — every row
in its own ablation table (Table 5) trains the document transformer fully; the manipulation is
always on the *query* side (full transformer vs. embedding-bag-of-the-raw-embedding-layer vs. the
trained lookup table). So LightRetriever cannot tell us what co-adaptation of the *document* tower
is worth, because it never holds the document tower fixed. The nearest published isolation of
"frozen vs. co-adapted document tower" (EmbedDistill, Google) runs the opposite experiment — freeze
the document tower and adapt only the query side — and its query side is always a small
*transformer*, never non-contextual. The Opus 2026-08-28 sweep in this same folder reached the
identical conclusion independently ("nothing found trains the doc tower to compensate for a fixed
weak query encoder — that structural inversion is unpublished"), which is corroborating, not new,
evidence. **Conclusion for the plan: this is a genuinely open experiment, not a re-run of
something published.** Whether it's worth opening depends on how much you trust the one indirect
signal available (below), not on a literature verdict either way.

### Evidence

**LightRetriever's own ablations never freeze the document tower** (arXiv 2505.12260, Table 5,
Llama-3.2-1B, confirmed by direct re-fetch of the v5 HTML and cross-checked against
`research/lightretriever.md`'s existing transcription — numbers match exactly):

| Row | Query (train) | Query (infer) | Doc (train) | BEIR | Δ |
|---|---|---|---|---|---|
| Ours (main) | Full transformer | Lookup table | Full transformer | 48.7 | — |
| A1: both sides light | Full transformer | Lookup table | **also reduced to lookup at inference** | 34.9 | −13.8 |
| A2: embedding-bag-of-raw-embedding-layer | Embedding bag over the *untrained* input embedding matrix | Lookup table | Full transformer | 37.5 | −11.2 |

Note what these actually vary: A1 changes what happens to the *document* side at **inference**
(full→light), not whether the document *tower's weights* were ever specifically adapted toward
reachability by a non-contextual query. In every row, training uses a full document transformer
jointly optimized with the query-side objective — so even the "Ours" baseline already reflects
*some* document-side co-adaptation, it's just never isolated against a frozen alternative. Re-fetch
confirmed: no sentence in the paper (main text, appendix, or limitations) describes a frozen/
off-the-shelf document-tower condition. This is the direct answer to "check LightRetriever's own
ablations" — checked, and the specific comparison the brief asks for does not exist there.

**EmbedDistill — A Geometric Knowledge Distillation for Information Retrieval** (Google, arXiv
2301.12005, 2023). Runs the frozen-document experiment, but for a small **transformer** query
student (DistilBERT/BERT-mini), not a bag-of-embeddings. Two independently-fetched tables, both
confirmed and mutually consistent with each other and with the Opus sweep's numbers:

- MSMARCO dev MRR@10, 11.3M-param student (teacher 37.2): symmetric direct training 18.6 → +
  score distillation 28.6 → **+ inherit frozen teacher doc embeddings 30.3** (+1.7) → + query
  embedding matching 35.4 (+5.1) → + query generation 34.8.
- NQ dev, Recall@5, 11.3M-param student (teacher 72.3): direct training 24.8 → + score
  distillation 44.3 → **+ inherit frozen teacher doc embeddings 56.1** (+11.8) → + query embedding
  matching 61.1 (+5.0) → + query generation 64.3.

Two things worth stating plainly: (1) the frozen-document step alone is a real, positive lever in
both cases, but it is not the largest one — query-side alignment training (embedding matching) adds
as much or more on top of it; (2) the paper's own claimed retention ("95-97% of teacher
performance") is for a query encoder that is still a multi-layer transformer, so it says nothing
about whether the same frozen-document logic holds up when the query side is reduced all the way to
lookup+average. Whether the value of "inherit the frozen document tower" survives at the
bag-of-embeddings extreme, or whether the document tower needs active co-adaptation to be reachable
by something that weak, is exactly the question neither paper answers.

**CARE — Benchmarking and Enabling Efficient Chinese Medical Retrieval via Asymmetric Encoders**
(arXiv 2604.10937, 2026) is the closest published *design* to the ablation this question wants —
it explicitly runs a two-stage recipe: Stage 1 aligns a query tower against a **frozen** document
encoder (contrastive + MSE on unlabeled data), Stage 2 **unfreezes both towers** and fine-tunes
jointly with asymmetric InfoNCE on labeled data. That is a controlled frozen-then-joint design, but
(i) it's domain-specific (Chinese medical, CMedTEB, not BEIR), (ii) the query tower is a 305M
BERT-style transformer, not a lookup table, and (iii) what we could extract does not include a
clean Stage-1-only-vs-Stage-1+2 ablation number — only the final CARE-0.3B-4B (55.91) and
CARE-0.3B-8B (56.75) numbers. Their own design choice to add Stage 2 implies they found the frozen
tower insufficient on their own benchmark, but the isolated size of that gain is not something we
could confirm from a primary re-read in this pass — flag before citing a number.

**ScalingNote** (arXiv 2411.15766, RedNote/Xiaohongshu, production system) is the strongest
"frozen document tower is fine" data point in the whole sweep, but again for a transformer query
student (4-layer BERT, 29M params) against a document tower that is explicitly **decoupled and
kept frozen** after the joint-training phase: 99% of R@50 retained (54.57 vs 55.15) at 83x QPS.
This reinforces that frozen document towers pair well with distilled transformer query students; it
is silent on bag-of-embeddings queries specifically.

**Google's ADE-SPL result** (Dong et al., EMNLP 2022, arXiv 2204.07120) is adjacent but orthogonal:
naive fully-asymmetric dual encoders (separate weights, no sharing at all) underperform symmetric
ones, and the fix that closes the gap is **sharing the final projection layer**, not adapting the
document tower's body. This is a different lever (projection-head sharing) from document-tower
co-adaptation, but it's a reminder that "asymmetric underperforms until *something* is shared or
aligned" is an established, general pattern — just never demonstrated with a bag-of-embeddings
query specifically. (Full-text PDF fetch of this paper failed to decode in this pass; the finding
is carried over from `research/landscape.md`'s prior citation, not independently re-verified here.)

### What we did NOT find

- No paper LoRA-adapts or fine-tunes a document encoder *specifically so a non-contextual /
  bag-of-embeddings query representation can reach it*, with a reported before/after number against
  a frozen baseline. This is the precise Q1 ask, and it is not in the literature as far as this
  sweep and the independent Opus sweep both found.
- No paper runs LightRetriever's exact recipe with the document tower frozen (i.e., "what if only
  the query-side embedding bag trains, and the document side is a genuinely off-the-shelf, never
  touched checkpoint") to give a same-architecture apples-to-apples number.
- No citation graph successor to LightRetriever exists that adds this ablation (confirmed again in
  this pass via Semantic Scholar; consistent with `m7-novelty.md`'s re-sweep).

---

## Q2 — Subword fragmentation as a retrieval failure mode

### Short answer

Tokenizer fertility (subwords per word) is an established, named metric correlated with downstream
quality (Rust et al. 2021), and there is at least one paper (an Amharic passage-retrieval study)
that puts concrete numbers on fertility vs. retrieval quality. But that evidence is about a
**cross-lingual tokenizer mismatch** (an English/multilingual tokenizer badly over-segmenting a
low-resource language), it degrades a **contextual** transformer too, and it never contrasts a
static/non-contextual model against a contextual one on the same fragmentation axis. **The specific
divergent-direction shape you measured — the teacher's quality going up with fragmentation while the
table's stays flat — has no match in what this sweep found.** It looks like a genuine, reportable
finding, not a known result, and it should be flagged as mildly surprising against the naive
"fragmentation is universally bad" framing the fertility literature otherwise supports.

### Evidence

**Fertility is a real, named, measured construct.** Rust, Pfeiffer, Vulić, Ruder, Gurevych, "How
Good is Your Tokenizer? On the Monolingual Performance of Multilingual Language Models," ACL 2021
(also on arXiv). Defines fertility = average subwords per word and correlates it with downstream
monolingual task performance across languages. A 2026 NeurIPS-workshop follow-up, "Beyond Fertility:
Analyzing STRR as a Metric for Multilingual Tokenization Evaluation" (arXiv 2510.09947, Nayeem et
al.), argues fertility *collapses* over-fragmentation behavior into a single average and proposes
Single-Token Retention Rate (STRR, a type-level "does this word get one token" diagnostic) as a
sharper replacement — worth knowing about if you want a better metric than raw fertility for a
tokenizer-redesign writeup, but it is a fairness/allocation diagnostic across languages, not a
retrieval-quality mechanism study.

**Concrete fertility→retrieval-quality numbers exist, but from a cross-lingual mismatch, not a
same-language rare-term effect.** "Optimized Text Embedding Models and Benchmarks for Amharic
Passage Retrieval" (arXiv 2505.19356, 2025) reports (their Figure 1, re-fetched and quoted exactly):
gte-modernbert-base fertility 13.80 → MRR@10 0.019 (worst); snowflake-arctic-embed-l-v2.0 fertility
2.35 → MRR@10 0.659; RoBERTa-Base-Amharic-Embed (Amharic-native tokenizer) fertility 1.46 → MRR@10
0.775 (best). Their stated mechanism: "excessive subword segmentation... fragments semantic
representations, which degrades retrieval accuracy," and this is presented as a **contextualized
transformer model** problem (all three rows are transformer-based rerankers/retrievers) — i.e. this
paper's own framing is that fragmentation hurts a *contextual* model, which is the reverse of what
we measured (our teacher got *better*, not worse, as fragmentation increased). Two confounds to flag
before treating this as a match: (1) 13.80 fertility is an extreme, out-of-distribution tokenizer
mismatch (an English-trained tokenizer applied to Amharic's templatic morphology), not the modest
within-English fragmentation increase (e.g. 1.0→2.0 subwords/word for domain terms) implied by your
"+1.0 subwords-per-word" framing; (2) no static/non-contextual baseline is in their comparison at
all, so the paper cannot speak to the contextual-vs-non-contextual divergence you found.

**Rare words are a known problem for contextual models too — this cuts against assuming contextual
composition is a free win at high fragmentation.** Schick & Schütze, "Rare Words: A Major Problem
for Contextualized Embeddings And How to Fix it by Attentive Mimicking" (arXiv 1904.06707) and its
follow-up BERTRAM (arXiv 1910.07181) establish that BERT-style contextual models do not automatically
compose fragmented rare words into good representations — they needed a dedicated fix (surface-form
+ context mimicking) to close the gap. This is evidence that "the teacher gets better on fragmented
queries" is not a trivial consequence of having attention; if anything the closest prior work expected
the opposite default. Worth citing as the reason your finding is worth stating as a finding, not
assumed.

**A generic mechanism for why a static table's rare/fragmented-term rows would be weak**:
representation degeneration / anisotropy in trained token embeddings. Gao et al.'s "representation
degeneration problem" and Chen, Gan, et al., "Rare Tokens Degenerate All Tokens: Improving Neural
Text Generation via Adaptive Gradient Gating for Rare Token Embeddings" (ACL 2022, arXiv 2109.03127)
show rare-token embeddings in trained models cluster into a narrow, low-quality cone because they
receive few, noisy gradient updates. This is a plausible *mechanism* for "the table stays flat" (its
rare-subword rows are undertrained and geometrically clustered so averaging them adds little
signal), but it's a generic LM-embedding-training finding, not specific to retrieval tables or to the
contrast against a contextual encoder's behavior. Cite as mechanism, not as a direct replication.

**No paper studies temporal vocabulary drift (a 2018/2019-era fixed WordPiece vocabulary meeting
2020+ terms like "covid") as a retrieval-quality question specifically.** The "COVID-19 → co / vid /
- / 19" example is widely repeated as an anecdote (found in multiple secondary sources, e.g.
biomedical-BERT tokenizer-adaptation papers) but every paper we found that studies vocabulary
adaptation frames it as cross-lingual or cross-domain adaptation (see Q3's "Teaching Old Tokenizers
New Words," arXiv 2512.03989, which explicitly is about language/domain extension and contains no
temporal-drift framing or COVID example at all, confirmed by direct re-fetch), never as "the same
English tokenizer, six years later, meeting new vocabulary" with a measured retrieval cost. This
looks like a genuine, well-established absence: the anecdote is common, the measurement is not.

### What we did NOT find

- No paper quantifies "contextual model quality rises with query fragmentation while a
  non-contextual/static model's stays flat" as a joint, controlled comparison — this specific shape
  was not found anywhere in this sweep.
- No paper measures the retrieval-quality cost of temporal (as opposed to cross-lingual or
  cross-domain) vocabulary drift on a fixed tokenizer.
- No paper isolates "contextual composition of subwords" as a mechanism using a controlled
  same-language, varying-fragmentation dataset (as opposed to cross-lingual fertility, which
  confounds fragmentation with language identity).

---

## Q3 — Multi-word / larger vocabularies for lookup-table or static retrieval

### Short answer

The literature is thin and mixed, and there is no clean, isolated "vocabulary size vs. bag-of-
embeddings retrieval quality" scaling curve anywhere — this specific experiment (a self-trained
64K–128K vocab with multi-word merges, measured against a fixed teacher, on retrieval) looks like
open ground, consistent with the rest of this project's novelty findings. What exists: (1) one
paper (multi-word product-retrieval) shows adding multi-word terms as single tokens helps, but on a
contextual ColBERT-style model, not a static one, and without a clean before/after number; (2) one
static-lookup-table retrieval paper (VDR) ran a vocabulary-size ablation and found a **small
regression** going from a 30K to a 110K vocabulary, but the two vocabularies also swapped languages
(English BERT vs. multilingual BERT), so the authors themselves attribute the drop to the language
mismatch, not vocabulary size — leaving the vocab-size question itself unresolved even there; (3)
general-purpose (non-retrieval) vocabulary-scaling work in language modeling reports no saturation
up to ~130K, which is at best weak transferable evidence; (4) a real mechanistic reason (rare-token
representation degeneration, same citation as Q2) to expect diminishing or even negative returns from
naively adding many new low-frequency multi-word rows unless the training procedure gives them
enough signal. Net: nothing says "don't bother," but nothing says "this reliably works" either — the
week of work is not pre-empted by a known negative result, but it also is not de-risked by a known
positive one.

### Evidence

**Positive precedent, contextual model, e-commerce domain.** Krasnov & Shcherbakov, "Multi-word Term
Embeddings Improve Lexical Product Retrieval" (arXiv 2406.01233, 2024). Method: SentencePiece
tokenization configured to keep multi-word brand names as single tokens (both BPE and unigram
variants tested), fed into a BERT-based dual encoder that scores token-wise (ColBERT-like), on the
WANDS e-commerce retrieval dataset. Best configuration (BPE + brand-name multi-word tokens): mAP@12
= 56.1%, R@1k = 86.6%. The paper's own claim is that "both BPE and unigram tokenizations with
multi-word variations produce consistently better results compared to standard tokenization," but a
precise isolated before/after delta and the exact vocabulary size before/after could not be extracted
from what we fetched — re-read the primary PDF/HTML directly before quoting a number in the report.
Caveat that matters for us: this is a full contextual transformer, not a static/bag-of-embeddings
model, so it demonstrates the *tokenization* idea (multi-word terms as atomic tokens help retrieval)
but not that it survives when the representation is a plain average of context-free rows.

**The one static/non-contextual retrieval paper we found with a direct vocabulary-size ablation.**
VDR — "Retrieval-based Disentangled Representation Learning with Natural Language Supervision" (Zhou
et al., arXiv 2212.07699, ICLR 2024). Its nonparametric variant (VDRα) represents queries as a plain
normalized bag-of-words vector over the vocabulary — the closest published architecture to our exact
query-side representation, though the *document* side there is still a trained sparse projection,
and the domain is image-text/cross-modal, not text-text BEIR retrieval throughout (their BEIR number,
44.5 nDCG@10 for the parametric VDR-t2t variant vs. DPR's 35.8, is for the text-to-text setting).
Their appendix vocabulary-size ablation: BERT vocabulary (~30K, English) → 44.5 nDCG@10 on BEIR;
multilingual BERT vocabulary (~110K) → 42.6 nDCG@10 — a real, if modest, regression when moving to
the bigger vocabulary. Critically, **the authors attribute the drop to the multilingual encoder being
mismatched to the (English) downstream tasks, not to vocabulary size itself**, and state that
increased sparsity from the larger vocabulary "doesn't significantly hinder learning." So this is the
closest thing to a directly relevant negative data point, and even its own authors decline to call it
a vocabulary-size effect — it's confounded exactly the same way the Q2 Amharic evidence is confounded
(size and language change together). No clean isolated vocab-size-only ablation (same language,
varying V) was found anywhere in this sweep, for any static or bag-of-embeddings retrieval model.

**Domain-adapted vocabulary, chemistry — reported but unverifiable in this pass.** ChEmbed
(arXiv 2508.01643, 2024/2025) reportedly uses "progressive tokenizer adaptation" for chemical-domain
retrieval and a headline figure of a 9.0% nDCG@10 improvement circulated in secondary summaries
found during search. Direct PDF fetch failed to decode into readable text twice; the exact baseline,
final vocabulary size, and whether the model is contextual or static could not be confirmed from a
primary source in this pass. **Do not cite the 9.0% number without a direct re-read** — this is
exactly the kind of secondary-source number this project has been burned on before.

**Tokenizer-adaptation methodology exists, but is not about vocabulary *size* returns and is not
retrieval-evaluated.** "Teaching Old Tokenizers New Words: Efficient Tokenizer Adaptation for
Pre-trained Models" (Purason et al., arXiv 2512.03989, EACL 2026 Findings) proposes continued-BPE
training to extend a tokenizer's vocabulary for new languages/domains without breaking the existing
merge structure, evaluated on machine translation (FLORES-200/COMET) and classification benchmarks —
no retrieval evaluation, no discussion of vocabulary-size returns, and (confirmed on direct re-fetch)
no mention of temporal drift or "covid"-style examples at all. Useful as a *method* reference if you
build a custom 64K–128K tokenizer (continued-BPE avoids unreachable tokens, a documented failure mode
of the naive "train a new tokenizer, append the non-overlapping tokens" approach), not as evidence
about whether the bigger vocabulary itself pays off.

**General (non-retrieval) vocabulary-scaling literature leans positive but doesn't transfer
cleanly.** "Scaling Embedding Layers in Language Models" (Yu, Cohen, Ghazi et al., Google, arXiv
2502.01637, 2025) and "Over-Tokenized Transformer: Vocabulary is Generally Worth Scaling" (arXiv
2501.16975) both argue larger vocabularies/embedding layers keep helping language-modeling perplexity
and downstream LM benchmarks (MMLU-Var, Hellaswag, etc.) up to the scales tested, with no reported
saturation. Full numeric tables did not extract cleanly from either PDF in this pass (structurally
present, not decodable as text) — the qualitative direction ("bigger vocab helps, no saturation
found yet") is as far as this sweep can responsibly go, and it's a language-modeling result, not a
retrieval one, and not about non-contextual bag-of-embeddings representations at all. Treat as weak
background, not evidence for our specific case.

**A mechanistic reason to expect the returns are not free.** Same citation as Q2: rare/low-frequency
token embeddings in trained models are known to under-update and cluster (representation
degeneration; Chen et al., ACL 2022, arXiv 2109.03127). If a self-trained 64K-128K vocabulary adds
many new multi-word rows for rare compounds (e.g. "covid-19"), each such row gets exactly as much
training signal as our table-construction procedure gives it — for LightRetriever's own construction
(and presumably ours), that's a *single* forward pass with an instruction prompt, not gradient descent
against many contrastive examples, so the "insufficient updates" failure mode this literature warns
about may or may not even apply the same way. This is a reason to check the mechanism (per the
project's standing directive) before either committing to or ruling out the lever — not a reason to
skip it.

### What we did NOT find

- No paper reports a controlled vocabulary-size scaling curve (holding language, domain, and
  architecture fixed, varying only V) for any bag-of-embeddings or static retrieval model. This is
  the literal Q3(d) ask and it is absent.
- No paper builds a custom multi-word tokenizer specifically for a lookup-table/static dense
  retriever (as opposed to a contextual model, or a sparse/lexical model) and reports BEIR-style
  nDCG@10 before/after.
- No negative result stating "bigger vocabularies are known not to help a bag-of-embeddings
  retriever" was found either — the honest state is absence of evidence in both directions, not a
  documented failure.
- Model2Vec's own documentation mentions optional vocabulary expansion for domain adaptation as a
  supported feature, but no quantified before/after retrieval numbers for that specific option were
  found in the model cards or results tables checked in `research/landscape.md` or this pass.

---

## Sources consulted in this pass (in addition to those already cited in the four "read first" files)

LightRetriever v5 full HTML re-fetch (arxiv.org/html/2505.12260v5); EmbedDistill full HTML
(arxiv.org/html/2301.12005, arxiv.org/pdf/2301.12005); Dong et al. ADE-SPL PDF (fetch failed to
decode, citation carried from `landscape.md`); CARE (arxiv.org/pdf/2604.10937); ScalingNote
(previously read, re-confirmed via `literature.md`); Rust et al. 2021 (via secondary search, not
directly refetched — already well-established); STRR (arxiv.org/pdf/2510.09947,
arxiv.org/html/2510.09947); Amharic retrieval paper (arxiv.org/html/2505.19356, direct fetch);
Schick & Schütze rare-words (arXiv 1904.06707, BERTRAM 1910.07181, via search only); representation
degeneration / rare-token clustering (arXiv 2109.03127, via search only); Multi-word Term Embeddings
(arxiv.org/html/2406.01233v1, direct fetch); VDR (arxiv.org/pdf/2212.07699 fetch failed to decode;
ar5iv.labs.arxiv.org/html/2212.07699 succeeded); ChEmbed (arxiv.org/pdf/2508.01643, fetch failed to
decode); Teaching Old Tokenizers New Words (arxiv.org/html/2512.03989v1, direct fetch); Scaling
Embedding Layers in Language Models (arxiv.org/pdf/2502.01637, structural fetch only, tables not
decoded); Over-Tokenized Transformer (arXiv 2501.16975, via search only); Dense Retrievers Can Fail
on Simple Queries / granularity dilemma (arxiv.org/html/2506.08592v1, direct fetch — read and found
not on-target for Q2, see note below); Distill-VQ (arXiv 2204.00185, via search, not on-target);
Semi-Parametric Retrieval via Binary Bag-of-Tokens Index / SiDR (arXiv 2405.01924, via search, not
on-target — sparse doc-side token index, not a dense query-side lookup table).

**Checked and ruled not relevant, for the record:** "Dense Retrievers Can Fail on Simple Queries"
(arXiv 2506.08592) uses "granularity" to mean semantic-importance hierarchy (keyword salience vs.
overall topic), not tokenizer/subword fragmentation — despite surfacing on fragmentation-adjacent
search terms, it does not bear on Q2. Distill-VQ and SiDR are about vector-quantization / sparse
document-side indexing efficiency, not query-side representation, so not on-target for Q1 or Q3
despite "distill" and "bag-of-tokens" name matches.
