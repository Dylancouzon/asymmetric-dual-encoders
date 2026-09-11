# M17 — additional avenues after the first plan

2026-09-11. Dylan asked whether any other ideas are worth exploring. Two Luna research passes
and a parent primary-source check produced the shortlist below. These are **proposals**, not
additional registered arms or measured improvements. M17 remains planning only; its current
matrix, budget, protected-access rules and M13 comparator remain unchanged.

## Recommended additions to consider before recipe lock

### 1. Teach equivalent query forms explicitly

Examples: `k8s ingress` / `Kubernetes ingress`, `S3 bucket policy` / `Amazon S3 bucket policy`,
and context-valid spelling or punctuation variants. Standard targeted replay does not explicitly
require such views to agree. This proposal adds a small equivalence loss between their Zero
query vectors alongside ordinary frozen-Stella distillation and general replay. It can help
existing subword rows as well as new whole-term rows; it is not dependent on expanding the vocab.

CAPOT provides a particularly relevant precedent: it aligns noisy queries to their original
forms while freezing the document encoder, avoiding index regeneration. Extending that idea
to verified technical aliases in a static table is **our inference**, not a result from that
paper. Its transformer results or training datasets do not transfer automatically to M17.
[Campos et al., CAPOT](https://arxiv.org/abs/2304.03401).

Use only high-confidence equivalences derived from admitted source material. Check the context
and intended sense independently; teacher agreement alone does not certify a synonym. `S3` has
other meanings, and “EC2” is not a synonym for “cloud.” Keep ordinary meaning-bearing context;
do not globally expand acronyms at inference. Split complete equivalence families together,
including spelling variants, before training or judging; filter both generated views through
the applicable provenance/decontamination path.

**Decisive comparison:** the same vocabulary, raw base data, total exposure, optimizer and
candidate lists, with and without the added equivalence supervision. Supply both views to both
arms so extra examples do not masquerade as a loss improvement. Predeclare the equivalence
weight and sample share before quality reads. Retain it only for better held-out variant
retrieval with acceptable ordinary-query and ambiguous-sense behavior. It needs paired-query
data and some extra teacher encodes; price those from the real pipeline before allocating it.

This is the strongest additional **training** idea for the stated complaint. It adds no query
runtime operation, document change or new teacher. No experiment has run.

### 2. Average compatible late checkpoints into one table

Checkpoint/weight averaging is a cheap way to test whether the last optimizer iterate is
unnecessarily noisy. SWA provides a generalization precedent under constant/cyclical learning
rates; model soups provide a separate precedent for averaging compatible fine-tuned weights
without an inference ensemble. Neither establishes a gain for Zero's table and linear-decay
Adam schedule. Call our proposal **late-checkpoint averaging**, not a reproduction of SWA.
[Izmailov et al.](https://arxiv.org/abs/1803.05407),
[Wortsman et al.](https://proceedings.mlr.press/v162/wortsman22a.html).

Use checkpoints from the same run, with identical tokenizer IDs, preprocessing and document
space. Average effective float32 rows after folding each checkpoint's learned scalar into its
row, then quantize once. Averaging rows and scalars separately is wrong because
`mean(weight * row)` generally differs from `mean(weight) * mean(row)`. Inspect effective-table
scale drift: output normalization makes a global table scale unidentifiable, so arbitrary
rescaling of one table would change its contribution to an average. Avoid cross-recipe soups.

**Decisive comparison:** one predeclared checkpoint window and one selection read against the
ordinary endpoint, followed by the unchanged locked audit if selected. No window sweep and no
choosing after audit results. It costs saved snapshots, an offline table pass and evaluation;
it does not require a new training chain, a second runtime table or a second query encoding.

The result is not exactly an ensemble of separately normalized query embeddings. Pooling is
linear in effective rows for a fixed bag, but final L2 normalization is nonlinear. Export and
evaluate the actual averaged int8 table. A null result is plausible; this buys stability rather
than more representational capacity.

## Conditional regression-repair option

If a candidate improves technical queries but harms general ones, a single predeclared blend
with v1 is a cheap alternative to another retraining cycle. BCWI found fewer negative prediction
flips in classification after interpolating old/new weights, without extra inference cost.
Applying this to retrieval is an untested transfer.
[Schumann et al.](https://aclanthology.org/2024.eacl-long.174/).

This is cleanest for original-vocabulary candidates, whose rows align exactly. Expanded
vocabularies require an explicit expanded baseline and still do not inherit v1 behavior at
the interpolation endpoint: M17 P0b already demonstrates sqrt-count sharing drift after
retokenization. Do not claim safe rollback merely because old rows were blended back. Keep
v1 itself available, and prefer same-run averaging as the first cheap finishing experiment.

## Separate engineering avenue: retain int8 rows in memory

The current standalone numpy loader eagerly dequantizes every row to float32
(`m11/release/zero_encoder.py`). It could retain the int8 codes and scales, gather the unique
IDs used by a query, and reconstruct only those rows before the existing sqrt pooling.

The arithmetic in `results/m17_planning_probe.json` implies about **125.0 MB → 31.4 MB** for
the current resident row arrays, or **137.6 MB → 34.5 MB** at the proposed maximum vocab. This
is an array-storage estimate, not a measured total-RSS saving or latency result. The bundle
and index need not change. Dequantization still uses the existing per-row scales and float32
arithmetic; this is not a lower-precision model or new quantization scheme.

**Decisive comparison:** isolated loaders, actual process RSS and cold start, repeated queries
at multiple lengths/batch sizes, full preprocessing/fallback conformance, and p50/p95 latency.
Decode-only-used-rows may lose latency to extra per-query work. Do not assume a speedup or
attribute it to the ONNX/FastEmbed paths without separately testing those implementations.
This is a modest optional serving change; no implementation or new measurement ran this turn.

## Lower-priority ideas and why

- **Multiple natural contexts for row initialization:** static-embedding work supports context
  selection as useful, but M7 already used teacher-context initialization and M17 already selects
  training contexts by coverage/residual. A new token-level loss needs an explicit bridge from
  teacher hidden states into Stella's final document space. Raw same-width hidden states are
  not automatically compatible rows. Keep this behind the cleaner alias comparison.
  [Gupta and Jaggi](https://aclanthology.org/2021.acl-long.408/),
  [Wang et al.](https://aclanthology.org/2022.lrec-1.277/).
- **Unprefixed teacher query targets (B4):** a legitimate old unrun option, not a new discovery.
  It changes supervision while keeping the teacher weights and document space fixed, but needs
  fresh target encodes and could weaken retrieval alignment. Historical claims that a prompt
  is just an absorbable constant are not valid for a contextual teacher. Keep this as a reserve
  hypothesis, not a default recipe change; current `instructions-m17.md` already corrects it.
- **More aggressive quantization/QAT:** M7 found very little int8 quality loss. There is no
  measured quantization deficit to prioritize over semantic coverage. Memory-preserving int8
  loading above addresses a different, concrete implementation cost.
- **Fusion-aware training:** still open but its same-corpus candidate scores, operator-specific
  loss and calibration are substantial extra work. M12's reopening conditions remain; nothing
  in this note fulfills them.
- **A separate hardest-query curriculum:** partly duplicates M17's teacher-residual selection.
  An additive bag can map two order-reversed queries to the same representation; repeated
  emphasis on such contradictions cannot recover missing order capacity. Avoid equating a
  large teacher residual with a learnable example.

## Budget recommendation and scope

Prioritize alias consistency as the next training comparison and a single predeclared averaging
check as inexpensive finishing work. Retain int8 loading as a serving task if RAM is important.
Before adding either quality comparison, amend the draft registry's selection count, window/
loss definitions and measured allocation; keep the recovery reserve and hard total intact.
Do not automatically append these ideas to the four-arm matrix or call them free because their
arithmetic is cheap: data preparation and exact evaluation still cost time.

No `CLAUDE.md` exception is required to research these ideas. This note changes no cap, licence,
teacher, release bar, protected-access protocol or runtime implementation. No model was trained.

## Research access record

Both Luna briefs named only `CLAUDE.md`, `m17/PLANNING.md`, `m17/registry.json`,
`m7/EXPLORED.md`, `m8/EXPLORED.md`, `m8/FINDINGS.md`, and their respective earlier
`research/m17-vocabulary-2026-09-11.md` / `research/m17-static-interaction-2026-09-11.md` notes.
Both returned exact matching local access lists, no recursive searches, no writes or experiments,
and no protected payload reads. Parent additionally read the M17 status, instructions, historical
`research/archive/m10-cleanup-2026-09-10/instructions-m16.md`, `m8/registry.json` (protocol only),
and `m11/release/zero_encoder.py`. Repository-wide content search was not used.

Primary sources linked above were checked by the parent except the lower-priority historical
ideas, which are grounded in repo records. Additional agent-only background URLs:
https://aclanthology.org/W19-5010/ (acronym disambiguation),
https://aclanthology.org/2020.acl-main.431/ (contextual-to-static embeddings),
https://openaccess.thecvf.com/content_cvpr_2018/html/Jacob_Quantization_and_Training_CVPR_2018_paper.html
(quantization background, not evidence of a Zero quality deficit).
No datasets, model weights, reserved qrels, `results/frozen_eval/untouched-*`, `work/m9reserve`,
or six-set/LoTTE payloads were accessed.
