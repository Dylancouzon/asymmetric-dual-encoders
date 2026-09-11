# M17 cheap Luna research: static late interaction and hard-candidate distillation

Date: 2026-09-11. Planning note only; no experiment, download, evaluation access, or
index mutation was performed.

## Scope and local evidence

M17 keeps the document tower frozen (`NovaSearch/stella_en_400M_v5`, 1024 dimensions) and
the existing single-vector document index. The released zero query path is a 30,522 x 1024
token table: tokenize, gather rows, count-saturated pool, then L2-normalize. It has no
transformer or query-time matrix multiply (`m11/release/zero_encoder.py`, `m7/RECIPE.md`).

M8's entropy diagnostic reports a nearly one-hot teacher distribution under the shipped
uniform bank (median teacher entropy 4.732e-7 nats); the shipped table's student KL median
was 1.078e-7 nats, and the positive ranked first for 99.75% of queries. The `teacher_top200`
condition had mean teacher entropy 0.7769 nats, which shows a richer candidate distribution;
it is not a measured training improvement. The unrun `R-LIST` hard-candidate listwise
variant remains open. This is a reason to test candidate-conditioned supervision, not
evidence that it will clear the quality gap. Mined-hard-negative gains in the old recipe were seen-document
memorisation and are closed as a recipe avenue; `R-LIST` is a distinct, still-unrun class.

## What “static late interaction” can mean

Model2Vec is the closest genuine static analogue: it forwards vocabulary items through a
sentence transformer once, stores one vector per token, and serves text by token lookup and
pooling. Its primary repository documents sequence-token output and dataset-free
distillation, while also stating that token vectors are static (context-independent).
[Model2Vec primary repository](https://github.com/MinishLab/model2vec)

ColBERT’s late interaction is a different architecture. It encodes each document as a
matrix of contextualized token vectors and scores a query/document pair with a sum of
per-query-token maxima (`MaxSim`). [ColBERT primary repository](https://github.com/stanford-futuredata/ColBERT)
The ColBERT paper likewise describes contextualized late interaction and its document
token representation. [ColBERT paper](https://people.eecs.berkeley.edu/~matei/papers/2020/sigir_colbert.pdf)

SLIM preserves this multi-vector premise, then sparsifies token vectors and uses an inverted
index plus refinement; it does not turn one frozen dense document vector into MaxSim.
[SLIM paper](https://arxiv.org/abs/2302.06587)
ColBERTv2’s compression reduces the footprint of late interaction, but still compresses
multi-vector document representations rather than reusing a single-vector index.
[ColBERTv2 paper](https://arxiv.org/abs/2112.01488)

## Algebra and fit to the current index

Let static query token rows be (u_i), document index vector be (d), and pooled query be

\[
q = \operatorname{norm}(\sum_i a_i u_i).
\]

The existing score is (q\cdot d). Any proposed “static interaction” that computes a
weighted sum of token/document-vector dots has

\[
\sum_i a_i(u_i\cdot d) = (\sum_i a_i u_i)\cdot d,
\]

up to the same final query normalization. It is therefore just pooled-vector ranking in
another order of operations; there is no MaxSim gain or new query capacity. A genuine
ColBERT score requires document token vectors (v_j):

\[
\operatorname{MaxSim}(q,d)=\sum_i \max_j(q_i\cdot v_j).
\]

Storing those vectors requires a new document representation, storage, and reindexing.
Alternatively, one could use a max-over-query score over the *existing* candidate document
vectors, but that is an inference-time candidate reranker with a serving change, not
ColBERT late interaction; it cannot recover documents absent from the initial candidate
set. Teacher-only MaxSim scores used as training targets would be a separate training change
and do not alter the shipped scorer or index. The compatible teacher signal here is the
frozen Stella cosine score between each candidate's existing document vector and the
teacher's query vector; introducing a new MaxSim teacher would require document token
vectors and is outside this scope.

## Hard-candidate listwise distillation

Primary retrieval work supports scoring a candidate list and distilling a teacher’s relative
distribution: RocketQAv2 explicitly constructs candidate passage lists, stresses the need
for hard negatives, and uses dynamic listwise distillation. [RocketQAv2 paper](https://aclanthology.org/2021.emnlp-main.224.pdf)
ColBERTv2 reports denoised supervision with mined negatives as part of its quality recipe,
but its student remains a multi-vector retriever. [ColBERTv2 paper](https://aclanthology.org/2021.naacl-main.272/)

For this project, a compatible prospective R-LIST plan would use the frozen Stella document vectors to
form a fixed candidate set per training query (for example, top-k from the current zero
table plus the known positive and a small BM25/declared-negative union), compute teacher
scores only for that candidate list, temperature-softmax the teacher scores, and minimize
listwise KL against scores from the pooled zero vector. False-negative filtering and strict
decontamination must remain as registered. This keeps the same document index and changes
only training targets/candidates. It would be a new, bounded hard-candidate listwise
experiment, not a claim of static MaxSim.

## Recommendation for the 72-hour budget

Do not implement “static late interaction” as a product architecture: under the frozen
single-vector index it collapses algebraically to the current pooled score, or else requires
reindexing/serving changes outside M17. Do not spend the budget on a token max-sum with the
same document vector; it has no magic gain.

Recommendation: prioritize `R-LIST` candidate-list distillation, with a small candidate list
and one predeclared temperature/configuration. It is the only listed lead directly motivated
by the measured inert uniform-bank objective. Treat it as a single cheap probe against the
closed hard-negative contrastive baseline, with an explicit stop rule and no protected
evaluation access. Keep B10 or other ideas out of this 72-hour scope unless R-LIST is
measured and earns a reopening decision.

## Access log

Local files read (only the requested set): `CLAUDE.md`, `instructions-m17.md`,
`m8/FINDINGS.md`, `m8/EXPLORED.md`, `m7/RECIPE.md`, `m11/release/zero_encoder.py`, and
`results/m8_b2_entropy.json`. `results/m8_b2_kl_audit.json` was checked and is absent. No recursive repository search was
used; no `results/frozen_eval/untouched-*`, reserved qrels/cache, `work/m9reserve`, six-set,
LoTTE, or scoring data was accessed.

Primary URLs consulted: Model2Vec repository; ColBERT repository; ColBERT paper; SLIM paper;
ColBERTv2 paper; RocketQAv2 paper (links above). Claims beyond these sources or the local
files are marked as planning inference.
