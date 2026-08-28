# The released recipe, end to end

What a third party needs to reproduce the artifact, in the order they would do it. Written
2026-08-28, after the simplification test failed and fixed the recipe at what was measured.
Everything here is derivable from the repo; this file exists so it does not have to be
reconstructed from `program.py` plus four drivers plus a ledger.

**Candidate: `p35w-2m-s2500`, served at `pool_mode=sqrt`.** Full pinned dev macro **0.6153**;
out-of-domain subset (the two CQADupStack components) **0.3672**. Both are dev SELECTION numbers,
not evidence about the six.

## What the artifact is

A **30,522 x 1024** fp16 table, one row per WordPiece token of the BERT vocabulary, plus a
per-token scalar weight folded into the rows at export. A query is tokenized, its rows are gathered
and pooled, and the result is L2-normalized. **No transformer runs at query time.** Documents are
encoded by a frozen off-the-shelf teacher, which is what the rows are trained to agree with.

| | |
|---|---|
| teacher (frozen, doc side) | `NovaSearch/stella_en_400M_v5` @ `ffeb2b7ee715c226d4ffe5e4619f7dbb48624c20` |
| table | 30,522 x 1024 · 62.5 MB fp16 · **31.3 MB int8** (per-row absmax, quality-free) |
| query rule | `Preproc(prefix="", add_special_tokens=True, max_length=512, pool_mode="sqrt")`, fingerprint `adb24fb2e8cad66f` |
| pooling | `sqrt` count saturation: a token occurring *c* times contributes weight sqrt(*c*), not *c*. Counts post-truncation WordPiece occurrences **including specials**. |
| unseen rows | left at initialization. 1,743 rows (5.71%) are never trained; 994 are `[unusedN]` placeholders and the reachable 749 contribute at 0.143x a trained row. A checked non-choice, not an oversight. |

`pool_mode=sqrt` is a **query-rule** change with no bytes and no query-time cost attached: the rows
and the int8 codes are byte-identical either way. It may only be set through `adopt_pool_mode.py`,
which refuses unless `results/m7_lever4_pooling_<run_id>.json` records that mode adopted for that
run id. It was adopted on this artifact and **failed** on the next candidate tried, so it is
artifact-specific and not a general property of the method.

## Data

TRAIN is **340,850 pairs** plus 220,632 query-text-only rows, from approved sources only
(`research/m7-data-licensing.md`): ESCI, FEVER-train, HotpotQA-train, Mr. TyDi (en), SQuAD-train,
plus NQ-open and TriviaQA query text. **MS MARCO is permanently excluded** — its terms are
non-commercial-research-only and the deliverable is Apache-2.0. `freeze.assert_releasable` walks
the whole init chain and refuses any artifact that inherits it.

Decontamination before anything trains: **R1** removes query overlap against every protected set,
**R2** removes positive-document overlap with the six, **R3** measures and discloses document
overlap without removing it. Counts in `m7/LEDGER.md`. `train.py` refuses to run without the pool
ban mask, and `pseudoq.build_decontaminated` raises rather than fall back to unfiltered text.

## Phase B — teach the token->direction map (16,000 steps)

Objective B needs no labels, only query text and the teacher's embedding of it, so it can be
taught over far more vocabulary than the real queries cover.

```
init                 teacher-context rows: each vocab token forwarded through the teacher
                     inside a query context (30,522 forward passes). NOT the input embedding
                     matrix, and NOT random -- see below.
b_pseudo_queries     2000000     the REQUEST. The pool saturates at 924,704 spans: build()
                                 draws n//5+1 per doc store and three of five exhaust. Set the
                                 request, not the realised count -- it reproduces exactly.
b_pseudo_frac        0.5         share of each batch drawn from them
learned_weights      True        per-token scalar, seeded from IDF (idf_init_weights=True)
reg_init             1e-3        pull toward init, scaled by 1/(1 + row update count)
lr 3e-3 (rows) / 1e-2 (weights), constant · batch 512 · seed 0
kl_weight 1.0, kl_k 32 · cos_weight 1.0 · temp 0.02 · n_neg 32,768 · bank_size 2,000,000
hard_neg_k           0           mined negatives are a CLOSED avenue, see below
fn_margin            0.02        false-negative filter: an in-batch/bank negative whose TEACHER
                                 score exceeds (positive's teacher score - 0.02) is masked out
                                 of the InfoNCE denominator. Non-zero in the shipping loss, and
                                 it is NOT what the mined-negatives arms varied. Measured cost:
                                 it removes 4.3% of the top-100 hardest negatives
                                 (`results/m7_diag_scores.json`).
use_provided_hardneg True        the DATASET-supplied negatives -- ESCI "Irrelevant" rows and
                                 Mr. TyDi's own negatives -- are used. Distinct from `hard_neg_k`
                                 (which mines them with the teacher) and unaffected by that
                                 avenue's closure.
```

Both of the last two are `train.Cfg` defaults (`m7src/train.py`), which is exactly why they were
missing from the first version of this file: a default is easy to reproduce and easy to forget to
write down. Anything else not named here is a `Cfg` default too.

## Phase A — 2,500 steps from that checkpoint

```
init                 run:<the B checkpoint>   (restores rows AND the trained token weights)
lr 1e-3 (rows) / 1e-2 (weights), warmup_linear, warmup 200 · batch 512 · seed 0
steps_a              2500
everything else      unchanged from B
```

Export with `table.save_release`, which folds the learned weights into the rows so the shipped
int8 artifact is self-contained. Training checkpoints keep the unfolded shape; a folded table
cannot resume training.

## Four things a joint test could not remove

Careful with the claim here. The simplification test (`m7/LEDGER.md`, "Recipe simplification")
removed all four at once and measured **−0.0048** on the full dev suite, raw CI [−0.0102, +0.0007]
— non-inferiority at a −0.0040 margin **not demonstrated**, in both precisions, with the
out-of-domain subset falling 0.3672 → 0.3627. So they stay.

But **why** they stay is not established. Single-knob full-suite ablations put the main effects at
about +0.0015 in total, against a joint −0.0048 — a gap the same size as the recipe-perturbation
band, so interaction and one unlucky draw are not separable. And **no single-knob arm clears Holm
over the family of eight**, so the ablations license no change either. The honest statement is
that a pre-registered joint test failed and nothing licenses removing any component individually
— not that each component is earning its keep. Backing off one at a time until something passed
would have been adaptive dev search.

| component | the cheaper alternative that fails | cost of the four together |
|---|---|---|
| teacher-context init | `input_emb` (no forward passes) | |
| 924,704-span pseudo pool (request 2m) | 324,704 (request 500k) | −0.0048 dev macro |
| IDF-seeded token weights | uniform | −0.0045 out-of-domain |
| `reg_init` 1e-3 | 0.0 | |

## Fusion with BM25 — one family, one parameter, selected on dev

The released system may be the dense table alone or the table fused with BM25; **which one is not a
judgement call**. `select_fusion.py` grid-searches both families on the four text-backed dev
components at `fusion.DEPTH = 1000` against the **int8 release** artifact, and the convex grid
contains **w = 1.0, the dense-only endpoint**, so "do not fuse" competes in the same mechanical
selection as the parameter. `freeze.write` derives `released_system` from which point wins.

| | |
|---|---|
| families searched | `rrf` (k ∈ 10, 20, 30, 60, 100) · `convex` and `convex0` (w ∈ 0.3 … 1.0) |
| depth | 1000, for selection **and** application — both families are depth-sensitive |
| BM25 | `bm25s` lucene, k1 = 1.2, b = 0.75, English stopwords, Snowball stemmer |
| run construction | zero-score padding dropped, self-hits (`doc_id == qid`) dropped — part of the frozen function, not a detail of the harness |
| fitted against | the int8 table, because that is what ships |
| selected on | the four text-backed dev components; BM25 has no run on the held-out slices, whose corpora are pool row indices with no document text |

The spec is written to `results/m7_fusion_<run_id>.json` with the artifact hashes, preprocessing
fingerprint, encoder identity and the BM25 run keys it was fitted against; the freeze re-derives all
of them and refuses a spec that was fitted on a different artifact. **A changed checkpoint
invalidates the fusion parameter** — re-select, never carry it over.

## Environment and data pins

| | |
|---|---|
| Python / torch / transformers | 3.12.14 · 2.8.0+cu126 · 4.57.6 (full lock: `m7/requirements.lock.txt`) |
| datasets · pytrec-eval-terrier · qdrant-edge-py | 5.0.1 · 0.5.10 · 0.8.0 |
| GPU | RTX 3080, 10 GB VRAM; 25 GB host RAM, 18 GB peak budget |
| teacher revision | `ffeb2b7ee715c226d4ffe5e4619f7dbb48624c20`, loaded with `trust_remote_code` |
| training dataset revisions | every HF repo pinned in `results/m7_trainmix_revisions.json`; `trainmix.load` raises rather than fall back to `main` |
| dev suite | hash-pinned in `results/m7_dev_manifest.json`; `dev_eval.dev_components()` aborts on any drift |
| eval assets | `results/eval_manifest.json` + `results/frozen_eval/` (vendored queries and qrels) |

Determinism: training is reproducible to **~5e-6** on the dev macro, not bit-identical — same
config, re-run, gives an almost-identical table (small GPU-reduction nondeterminism).

## Closed avenues, so they are not re-attempted

`m7/EXPLORED.md` is the full list. The ones a reimplementer would most likely try:

- **Mined hard negatives** (teacher / BM25 / mixed) — closed. The apparent gain was
  `heldout-train` +0.0297 and `hotpotqa` +0.0187, i.e. seen-document memorisation; the
  out-of-domain subset spans 0.3658–0.3688 across every arm including the baseline.
- **n-gram / bigram rows** — the only structurally new lever besides pooling, and closed-form
  integration onto the trained winner is −0.0301. A joint retrain stays open.
- **Query-side centering, whitening, top-PC removal, IDF/SIF weighting** — all absorbable into the
  table, therefore not capacity. Proved to machine precision.
- **Training through the pooling rule** (lever #6) and **update-count row shrinkage** (lever #5) —
  both fail their pre-registered bars.

## Reading any dev number in this file

Four of the six dev components are Wikipedia or train-adjacent, and every confirmatory dataset is
out-of-domain, so the macro is the least predictive figure available. Retention against the teacher
is **0.915 on the dev macro, 0.846 text-backed and 0.764 on the out-of-domain pair**, where BM25
scores 0.3223 (`m7_retention_p35w-2m-s2500.json`). The suite has absorbed **58 trained arms, 322
in-training dev evaluations and 90 eval-only variants** (`m7_dev_reuse_count.json` — quote the
artifact, not this line; the earlier "53+ / ~300" here was a recollection and was already stale)
with Holm applied inside named families only. And every interval here is a **query-sampling**
interval: training is
deterministic, so no CI in this repo contains a recipe-replication term, while a nuisance step
count was measured moving the dev macro by 0.0049 — more than the largest adopted lever effect.
