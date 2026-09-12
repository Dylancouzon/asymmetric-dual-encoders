# M17 vocabulary research — 2026-09-11

## Scope and recommendation

Constella Zero remains a lookup table over the frozen Stella 1024-dimensional document index: query
encoding is tokenization, row gathers, pooling and normalization, with no transformer at runtime
([m11/release/zero_encoder.py](../m11/release/zero_encoder.py)). The practical v1.1 proposal is a
small, domain-aware tokenizer extension for S3/Kubernetes terms, trained jointly with the table
under hard-candidate/listwise distillation. It should be treated as a bounded quality probe, with no
performance promise.

Under a 35M raw-parameter cap, the existing 30,522 x 1,024 table has 31,254,528 entries, leaving
3,745,472 entries, or **3,657 additional 1,024d rows**. A 3,072-row extension uses 33,594 x 1,024
= 34,400,256 table parameters. The extra 3,072 rows are about 6.0 MiB as fp16; the resulting full
table is about 65.6 MiB as fp16 before tokenizer metadata, leaving roughly
0.6M parameters of headroom. The 65,536-row D2 shape is therefore not compatible with the cap,
although the block-CG implementation shows that 64K/128K closed-form solves are computationally
possible; its quality half still requires a real tokenizer and is descriptive until registered
([m8src/blockcg.py](../m8src/blockcg.py)).

Recommendation: reserve at most 3,072 rows, with **up to 1,024 provisional must-cover technical
rows** (S3, Kubernetes and adjacent DevOps terms) and allocate the remainder from an evidence-driven
broad audit. Select rows from high-support lexical units; initialize from constituent rows only as a
starting point; then update incumbent and added rows together with a pre-registered hard-candidate/
listwise loss. Preserve the original 30,522 token IDs, pin and hash the *new* serialized tokenizer,
and use the matching tokenizer consistently for each arm's training and evaluation. Learned scalar
weights also count against the 35M parameter cap; tokenizer metadata counts toward bytes, not trainable
parameters. The student and teacher already share the same WordPiece-30,522 vocabulary; there is no
teacher/student vocabulary-count mismatch to repair. This is distinct from M8's additive-row
screen, whose sum initialization exactly reproduced the incumbent served vector and whose four
segmentation/additive arms all lost (best −0.0028; D2 −0.0052 against +0.00519).

## What the literature supports

* Vocabulary expansion work treats new-token initialization as a real optimization problem. Mundra
  et al. establish a convex-hull framing and report constrained Word2Vec and simple multivariate
  initialization as competitive baselines across expanded RoBERTa/LLaMA vocabularies. That supports
  constituent or weighted-subword initialization as a cheap warm start, not as evidence that frozen
  additive rows improve retrieval ([CoNLL 2024 paper](https://aclanthology.org/2024.conll-1.8/)).
* FOCUS (EMNLP 2023) is another primary reference for initializing new tokenizer embeddings from
  overlapping source-token combinations ([paper](https://aclanthology.org/2023.emnlp-main.829/)). Its
  setting is language/model vocabulary adaptation and does not establish compatibility with a frozen
  document index or Zero's sqrt pooling.
* Model2Vec's official repository describes a static embedding recipe: forward a vocabulary through a
  sentence transformer, then pool/post-process, with dataset-free distillation. It is useful
  literature for the *tokenizer-first/static-table* idea and custom vocabulary screening, but is a
  separate software/vendor claim and must not be assumed approved or transferable to this artifact
  ([official repository](https://github.com/MinishLab/model2vec), [official documentation](https://minish.ai/packages/model2vec/introduction)).
* The distillation objective should expose meaningful relative scores. Tamber et al. report that
  ordinary InfoNCE/hard-negative fine-tuning can degrade retrieval, while listwise distillation can
  improve several datasets; they also report dataset-dependent decreases and teacher bottlenecks
  ([paper](https://arxiv.org/abs/2502.19712)). ADAM similarly argues that very hard negatives can be
  too trivial for a teacher and constructs moderate-relevance “dark” examples to preserve soft-score
  information ([paper](https://arxiv.org/abs/2212.10192)). This is consistent with M8 B2's nearly
  one-hot **teacher candidate entropy** under the uniform bank (median 4.732e-7 nats; mean 0.0182)
  versus mean entropy 0.7769 nats for teacher-top-200 candidates. The student positive was already
  top-ranked on 99.75% of diagnostic queries. These are diagnostic entropy/KL readings, not a forecast
  of a gain.

## Coverage audit and candidate discovery

The vocabulary budget must cover the teacher's query distribution, not just DevOps terminology.
Before mining candidates, report per-domain query counts, token fertility, OOV/rare-piece rates,
added-token activation/support, and teacher residual after the incumbent table prediction. At
minimum, stratify general web/factual, science, medicine, finance, legal, software/DevOps, and
multilingual or non-English queries when they are present in the approved training pool. If the
teacher is English-only, multilingual rows need a measured teacher signal and an explicit policy;
they should not consume budget merely because another tokenizer would fragment them.

Candidate discovery is training-only and must be fit-honest: mine whole-word/compound units from
the training partition, then rank candidates using a joint score based on activation/support,
fertility reduction, and teacher residual explained by queries containing the candidate. Residual
explained is essential: M8 showed that fertility moved while quality did not, so fragmentation alone
cannot prioritize rows. Require provisional support minima (for example, a nonzero activation in
multiple query groups and a documented minimum count) and publish the sensitivity to those minima;
the exact thresholds belong in the protocol amendment. Allocate up to 1,024 rows to must-cover
technical concepts only when they pass these support and residual checks; allocate the remaining
2,048 rows by the broad audit's marginal residual/support score, with per-domain floors or caps fixed
before seeing endpoint quality. A candidate list that is all technical terms is not a coverage audit.

This audit can discover candidates from training text and teacher outputs only. It must not inspect
reserved evaluation content, and it must obey the repository's licensing rules for text, negatives,
targets and generation seeds.

## Tokenizer and collision risks

The BERT reference pipeline lowercases (for uncased models), splits punctuation, then applies
WordPiece independently per whitespace token. Thus `S3/k8s` naturally crosses punctuation and can
become several pieces; an added `s3` or `k8s` token must be tested against the exact normalization and
pre-tokenization path ([Google BERT reference](https://github.com/google-research/bert#tokenization)).
Hugging Face's tokenizer API makes boundary behavior explicit: `AddedToken(single_word=True)` avoids
matching inside a word, while `normalized`, `lstrip`, and `rstrip` change matching and whitespace
consumption ([AddedToken API](https://huggingface.co/docs/tokenizers/main/api/added-tokens)).

For this reason, candidates should be lowercase canonical forms (`s3`, `k8s`, `kubernetes`, selected
compound forms) with `single_word=True` unless a deliberate hyphen/slash form is registered. Reject
tokens whose match changes punctuation or whitespace semantics, special-token behavior, or the
tokenization of a control query. Validate round-trip tokenizer IDs and offsets on case variants,
`S3`, `s3://`, `k8s`, `k8s/ingress`, `Kubernetes`, hyphenation, possessives and adjacent punctuation.
Do not add short substrings such as `k`, `s`, or `8`: they create boundary collisions and can steal
matches from existing WordPiece paths. Preserve a deterministic tie/order rule and hash the tokenizer
artifact.

## Why joint training is the only credible reopening

M8 explicitly says D2-PRE closed both segmentation and additive overlapping word/character n-gram
rows at equal budget. Its diagnostic found zero-update mass only 0.0001–0.001, so this was not a
coverage failure; many added rows being inert is a selection/usefulness finding. Sum-initialized new
rows reproduce R0 to about 2.8e-6; changing segmentation/fertility alone did not move quality
([m8/FINDINGS.md](../m8/FINDINGS.md), [m8/EXPLORED.md](../m8/EXPLORED.md),
[m8src/d2_pre.py](../m8src/d2_pre.py)). `m7/RECIPE.md` records that a joint n-gram retrain remained
open, while M8's closed-form additive integration did not test it
([m7/RECIPE.md](../m7/RECIPE.md)).

Jointly updating the incumbent rows, added rows and (if retained) pooling weights changes the
representation itself, so the exact sum-initialization null no longer applies. The loss should use
teacher scores over a fixed, sufficiently informative candidate set (positive plus hard and
moderate-score negatives), with a small contrastive anchor only if registered. Hold out query groups,
measure exact retrieval on the declared endpoint, and compare against the unchanged v1 table. A
single dev improvement is insufficient: require the pre-registered bar, component sign agreement,
and tokenizer-fidelity/coverage checks before considering a six-set follow-up.

## Support data and alternatives

Support text should be limited to already approved, commercially usable training sources in this
recipe; do not download datasets or models for this planning note. Use the project's licensed query
text and an explicitly registered broad support audit, with S3/Kubernetes as a must-cover slice.
Candidate mining may use text only;
teacher targets, negatives and generation seeds must obey the repository's licensing rules. If domain
support is too sparse, the safer alternatives are (1) keep v1 and improve fusion/lexical handling,
(2) run `R-LIST` with the incumbent 30,522 rows first. A wholesale tokenizer replacement preserves
the document vector space mathematically, but changes serialized query behavior and adds compatibility
and testing work; it should not be bundled into v1.1.

## Conditions that reopen M8's closed D2

Reopen only with a dated protocol amendment that changes the question from frozen closed-form row
integration to **trained joint optimization**, supplies a meaningful candidate-score distribution,
and stays within the full parameter cap including scalars (3,657 is the rows-only arithmetic,
not the scalar-inclusive maximum). The amendment must predefine support-data provenance, tokenizer
collision tests, group-honest validation, a quality bar/noise floor, and an exit rule for a negative.
It must show enough activation coverage to make the added rows relevant. A larger block-CG solve,
better fertility, or a new additive initialization by itself does not reopen D2; the M8 evidence
already separates those effects from capacity.

## Exact access log

Repository files read (the only repository files accessed):

* `CLAUDE.md`
* `instructions-m17.md`
* `m8/FINDINGS.md`
* `m8/EXPLORED.md`
* `m7/RECIPE.md`
* `m11/release/zero_encoder.py`
* `m8src/d2_pre.py`
* `m8src/blockcg.py`
* `results/m8_b2_entropy.json`

The optional `results/m17_planning_probe.json` was checked and was absent at the time of this update.

Web URLs accessed (primary papers or official project documentation):

* https://aclanthology.org/2024.conll-1.8/
* https://aclanthology.org/2023.emnlp-main.829/
* https://github.com/MinishLab/model2vec
* https://minish.ai/packages/model2vec/introduction
* https://arxiv.org/abs/2502.19712
* https://arxiv.org/abs/2212.10192
* https://github.com/google-research/bert#tokenization
* https://huggingface.co/docs/tokenizers/main/api/added-tokens

No datasets or models were downloaded. No experiments, evaluation access, or source mutations were
performed; this note is planning research only.
