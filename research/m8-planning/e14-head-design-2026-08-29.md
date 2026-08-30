> **SUPERSEDED, and kept only as the brief that was reviewed.** Codex returned three BLOCKERs
> against this design (`codex-e14-head-review-2026-08-29.md`), all reproduced independently and
> all adopted. The binding design is now the amended `E14-HEAD` row in `m8/registry.json` plus the
> `m8/LEDGER.md` §15 entry of 2026-08-29. **Do not implement from this file.** Three things below
> are simply wrong: the claim that a linear doc-side head is a no-op, the claim that zero-init
> makes the head exactly the identity, and the learning-rate ladder, which would have observed the
> endpoint before selecting on it.

# E14-HEAD: implementation design, for adversarial review before anything runs

You are reviewing a **design**, not code. Nothing here has been run. The point of the review is to
find the flaws now, because the probe costs a training campaign and its result feeds a much more
expensive decision (`E14-LORA`). Two previous reviews of a sibling probe (`B3`) each found a fatal
flaw pre-run, and the fix to B3's primary contrast is visibly what saved that result — the contrast
we retired came out NEGATIVE on the primary endpoint.

**Brief yourself adversarially.** I state below what I believe. Try to break it. A review that
confirms me is worth nothing.

---

## 1. What the probe is, as registered

The project ships a **zero-query-compute retriever**: a lookup table maps each vocabulary token to
a vector, a query is `normalize(sum_t w_t * r_t)` over its token ids, and documents are encoded by
a frozen transformer teacher (`NovaSearch/stella_en_400M_v5`, dim 1024). No transformer runs at
query time. The table is trained in two phases against the frozen teacher:

- **Phase B** (distillation): match the teacher's query vector (cosine + KL over a candidate set).
- **Phase A** (contrastive): InfoNCE against the teacher's *document* vectors, with a large
  in-VRAM negative bank.

`E14` asks: **the document tower was never trained to be reachable by a bag of token vectors — how
much is that costing us?** The owner ruled "measure it small first". `E14-HEAD` is the small
measurement, registered in `m8/registry.json` as:

- **lever**: an MLP head (`Linear -> GELU -> Linear`) applied to the **frozen teacher's cached
  document vectors**, trained jointly with the table. It **must be nonlinear**: a linear doc-side
  map is provably absorbable into the table, so it would measure nothing while producing a number.
- **held fixed**: query side stays a pure lookup table; teacher weights untouched; Phase A/B
  structure, updates, batch, negatives, temperature, learning rate and the pair pool all as R0.
- **comparator**: `R0` = the frozen recipe with no head, same seeds. On disk as
  `m8nf-seed0/1/2` (2500 Phase-A steps from the shared Phase-B checkpoint `p35b-2m`).
- **endpoints** (the same two scalars B3 used, both at **int8 / sqrt**):
  - `DENSE` = out-of-domain macro over `cqadup-programmers` + `cqadup-physics`
  - `FUSED` = the 4-component fused macro under the frozen fusion operator
- **bar**: mean-over-seeds gain **>= 0.0040 on BOTH** scalars (intersection-union). Frozen.
- **seeds**: 3, paired — same Phase-B checkpoint, differing only in the Phase-A seed.
- **scope limit, registered**: an MLP on the final document vector cannot recover information the
  tower discarded. So a **null is WEAK evidence** about the LoRA and may never be written as
  closing E14; a **positive is STRONG evidence** for buying it. That asymmetry is the whole reason
  the cheap stage runs first.
- **shippability gate**: `B6-pre` PASSED with `--head linear` only. Before any head-bearing
  candidate is called shippable, B6-pre must re-pass with `--head mlp` (E3 requires the head to
  fuse into ONE document ONNX file as plain nodes).

R0's config, verbatim from its artifact: `objective=A`, `steps_a=2500`, `steps_b=0`,
`batch=512`, `n_neg=32768`, `bank_size=2_000_000`, `temp=0.02`, `fn_margin=0.02`,
`lr=1e-3`, `lr_weights=1e-2`, `lr_schedule=warmup_linear`, `warmup_steps=200`, `reg_init=1e-3`,
`kl_weight=1.0` (inert: `steps_b=0`), `pool_mode=mean` at train time, served under `sqrt`,
338,076 training pairs over esci-us / fever-train / hotpotqa-train / mrtydi-en / squad-train.

The measured noise floors this bar sits on: dense 0.00095–0.00227, fused 0.00059–0.00066, both
from K=3 true-seed nulls. Bar = `max(0.0040, 2 x floor)` = 0.0040 on both.

---

## 2. Engineering constraint you need to know

`m7src/` is **frozen** (rule G3): M8 imports from it and may never edit it. The training loop
lives in `m7src/train.py::run()`, and the pieces that matter are:

- `infonce(qv, pos_v, neg_v, temp, teacher_q, teacher_pos, fn_margin, neg_pool_idx, pos_pool_idx,
  all_pos_idx, stats)` — **module-level**, so patchable from outside.
- `step_a(idx)` — a **closure inside `run()`**, so NOT patchable. It gathers `pos_v`, `neg`
  (bank sample + hard negatives) and `tp` from the fp16 document pool memmap, casts to float32,
  and calls `infonce`.
- `opt = torch.optim.Adam([...])` — created **inside `run()`**, so the head's parameters can only
  join it by patching `torch.optim.Adam` in the subprocess before `run()` is called.
- `opt.zero_grad(); loss.backward(); opt.step()` in `train_phase`, plus `set_lr` which rescales
  **every** param group by the same warmup/decay factor from `base_lrs`.
- `dev_eval.doc_vecs(comp)` — the single choke point through which **every** evaluation path
  (in-training dev, and `multieval.eval_makers`, which `compare_full.py` drives) obtains document
  vectors for a dev component.

So the implementation is a set of monkeypatches installed in a per-arm subprocess, plus M8-side
persistence and scoring. There is no way to do this without patching, short of vendoring an
800-line copy of `train.py`, which I judge worse (it would rot silently against the original).

---

## 3. The proposed design

### 3.1 The head

```
head(d) = L2normalize( d + W2 @ GELU(W1 @ d + b1) )        # bias-free second layer
W1: (2048, 1024)   Kaiming-uniform init
W2: (1024, 2048)   ZERO init
```

Rationale, and what I believe:

- **Residual with a zero-init output projection makes the head EXACTLY the identity at step 0.**
  So the arm is literally "R0 plus capacity": at step 0 the system is bit-identical to R0, and any
  divergence is the head learning something. Without this, 2500 steps would be partly spent
  recovering from a randomly scrambled document space, and a null would be confounded with
  "not enough steps to recover".
- Zero-init on `W2` gives zero gradient to `W1` at step 0, but non-zero gradient to `W2`, so `W2`
  leaves zero at step 1 and `W1` starts moving at step 2. This is LoRA's initialization, and it is
  standard.
- Hidden width `2 * dim = 2048`, ~4.2M parameters. All doc-side, so zero query-time cost; it ships
  inside the document ONNX graph.
- The renormalize is required: `infonce` takes plain dot products and R0's document vectors are
  unit-norm, so an un-normalized head would silently change the effective temperature.

### 3.2 Where the head is applied during training

Only Phase A runs (`steps_b=0`), so only `infonce` matters. **I will not patch `infonce` by
wrapping it**, because of a specific hazard:

`infonce` computes the **false-negative mask** internally from the same `neg_v` tensor it scores
with:

```python
t_neg = teacher_q @ neg_v.T
t_pos = (teacher_q * teacher_pos).sum(1, keepdim=True)
mask  = t_neg > (t_pos - fn_margin)
s_neg = s_neg.masked_fill(mask, -inf)
```

If the head is applied to `neg_v` before this, then **the trainable head controls which negatives
are masked out of its own loss** — and masking more negatives makes InfoNCE trivially smaller.
That is a direct reward-hacking channel: the head could reduce training loss by inflating the
mask rather than by improving the geometry, and `fn_margin` is 0.02 and active in R0. The
false-negative mask is a statement about what *the teacher* considers relevant — a property of the
data, not of the head — so it must be computed on **raw** vectors.

Therefore: an M8 function `infonce_head(...)`, a **verbatim copy** of `m7src`'s `infonce` with one
change — the scored `pos_v`/`neg_v` are head-transformed while `teacher_pos` and the `neg_v` used
inside the `fn_margin` block stay raw. `train.infonce` is rebound to it in the subprocess.

**Equivalence test, run before any arm**: with `head = identity`, `infonce_head` must return
bit-identical loss to `m7src.train.infonce` on random inputs across a grid of `fn_margin`,
`all_pos_idx` and `stats` settings. If it does not, the copy has drifted and nothing runs.

### 3.3 Optimizer

```python
_Adam = torch.optim.Adam
def _AdamWithHead(params, **kw):
    return _Adam(list(params) + [{"params": list(head.parameters()), "lr": HEAD_LR}], **kw)
torch.optim.Adam = _AdamWithHead
```

The head therefore rides R0's `warmup_linear` schedule (`set_lr` scales every group by the same
factor from its recorded `base_lrs`), and `reg_init`'s pull-toward-init penalty touches only the
table rows, as in R0.

### 3.4 The head's learning rate — the one genuinely new hyperparameter

`held_fixed` says "learning rate as R0", but R0 has no head, so the head's lr is not covered. This
matters more than it looks: **a null at an untuned lr is evidence about my configuration, not
about the method** (this project has a standing directive about exactly that failure, written
after a session called contrastive training "broken" at an lr 30–300x above every published
recipe).

Proposed, and to be registered as an amendment **before** it runs:

1. **Selection stage**: lr in `{3e-4, 1e-3, 3e-3}`, one seed (0), 2500 steps.
2. **Selection statistic**: InfoNCE loss on a **held-out 2% slice of the TRAINING pairs**, drawn
   once with a fixed seed and excluded from training in every ladder arm.
   - Not dev, because dev *is* the endpoint and selecting on it would leak.
   - Not training loss, because that would reward an lr that merely memorizes.
3. **Then** 3 paired seeds at the winning lr; those three are the arm the bar reads.
4. The ladder arms are reported beside the result, so the reader can see whether the endpoint was
   flat in lr or whether one setting was doing the work.

### 3.5 Evaluation — the part I think is most likely to be wrong

The head changes the **document** vectors, so every evaluation of a head-bearing arm must apply
that arm's head to the dev corpora. The choke point is `dev_eval.doc_vecs(comp)`.

But `multieval.eval_makers` reads each corpus **once and shares it across all variants** in the
process, and a `maker` only produces *query* vectors. So a single `compare_full` run containing
both R0 arms and head arms cannot give them different document vectors. Consequences:

- **One scoring process per head-bearing arm**, with `dev_eval.doc_vecs` patched to apply that
  arm's head and renormalize. Three E14 arms -> three corpus passes.
- **R0's rows are reused** from the existing dump `results/m7_devperquery_m8b3.json.gz`, which
  already holds `m8nf-seed0/1/2` at int8/sqrt. Justification for reusing rather than re-scoring:
  scoring is deterministic, the dev suite is pinned by hash, and **no commit has touched `m7src/`
  or `bench/` since those rows were produced** (verified with `git log --since` on both paths).
- **Determinism check, not assumed**: one R0 arm is nevertheless re-scored unpatched in a fourth
  pass, and its per-query values must match the b3 dump exactly. If they do not, the reuse is
  abandoned and everything is re-scored in one campaign.
- The four dumps are merged (run-id keys are disjoint) and handed to an `e14_decide.py` built on
  `b3_decide.py`'s reader, so the two scalars are extracted by the same code that produced B3's.
- The in-training dev logs also get the patch, so the training curve in the artifact is honest
  rather than scoring a head-trained table against un-headed documents.
- FUSED: `m8src/fused_floor.py` is M8 code and gets the same per-arm head parameter.

### 3.6 Persistence and the shippability gate

`m7src`'s `save_table` knows only about the `QueryTable`, so the head's weights are saved
separately by the M8 driver as `work/runs/<run_id>.dochead.pt`, with its architecture and the
`run_id` it belongs to recorded inside. An arm whose head file is missing may not be scored.

`B6-pre --head mlp` must export the **same architecture** — residual, hidden 2048, bias-free second
layer — or its PASS does not cover what E14 trains. `b6_pre.py`'s current `mlp` branch is a plain
square `Sequential(Linear, GELU, Linear)` with both layers identity-initialized (note: that is
*not* the identity function, since GELU sits between them — harmless for an export-parity test,
misleading as a comment). I intend to align it to the trained architecture before running it,
so that one artifact answers the question that is actually asked.

---

## 4. What I believe, stated so you can attack it

1. The residual zero-init head makes this a clean "R0 + capacity" contrast, and without it a null
   would be uninterpretable.
2. Keeping the false-negative mask in raw teacher space closes the only reward-hacking channel I
   can find into the training objective.
3. Reusing R0's rows from the b3 dump is safe because scoring is deterministic and the training
   path has not changed, and the determinism check makes that falsifiable rather than assumed.
4. Selecting the head lr on held-out training pairs is the cheapest defence against a null that is
   really a configuration artifact, and it does not touch the endpoint.
5. 2500 Phase-A steps is enough to train a 4.2M-parameter shallow head on 1024-dim inputs at batch
   512 — so a null would be about the method, not about optimization budget.
6. The measurement is worth running even though its scope limit means a null cannot close E14.

## 5. Questions I want attacked specifically

- **Is there a degenerate solution I have not closed?** The head is shared across all documents and
  sits inside a softmax over a 32,768-negative bank. Can it reduce InfoNCE in a way that does not
  correspond to "the document space became more bag-reachable"? Rank-collapsing directions the
  table cannot express? Exploiting the `neg_pool_idx` positive-masking? Interacting with
  `reg_init`'s pull of the rows toward `W0`?
- **Is the endpoint still measuring what it claims** once documents move? The out-of-domain macro
  is two CQADupStack components; the head is trained on esci/fever/hotpotqa/mrtydi/squad. Is a
  gain there attributable to re-shaping, or could it be distribution shift in disguise?
- **Does reusing R0's rows across processes break the paired structure** the bar assumes? The bar
  is a mean-over-seeds difference on macro scalars, but per-query values are dumped and could
  support a paired bootstrap later.
- **Is belief 5 actually true?** If 2500 steps is not enough, what is the cheapest evidence that
  would settle it without burning the endpoint?
- **Is the head lr ladder a protocol hole?** It is a selection stage in front of a probe whose bar
  is frozen. I claim held-out-training-pair loss is insulated from the endpoint. Is it?
- **Does anything here quietly redefine C2 or the released artifact's shape** in a way the owner's
  ruling did not authorise? The ruling explicitly did NOT authorise the doubled 10.12M pre-encode,
  the stella derived-weights licence check, or any C2 redefinition.
- **Is there a cheaper design that answers the same question better?** I would rather hear that now.

Report findings as BLOCKER / MAJOR / MINOR with the reasoning, and say plainly if you think the
probe as designed should not run.
