# M11a plan — ship `zero` end to end

Mandate + the four rulings: `instructions-m11.md` Amendment A. Branch `m11-work`. Status:
`m11/STATUS.md`. No quality number is read or written here, so `m9src/guard9.py` registration does
not apply.

**Reviewed adversarially before execution, 2026-09-03** — Codex (`work/briefs/m11a-codex.log`) and
Fable, against `work/briefs/m11a-review.md`. Both logs audited for reserved-set reads: clean, the
only matches are the exclusion text itself. The first draft of this plan asserted four things that
are false; they are recorded in §Corrections so no one re-derives them. Fable's numbers below are
measured, not argued.

## Blocking findings — nothing is published until T0 and T1 are done

**The release gates do not bind the bytes that get uploaded.** `push.py` gate 1 hashes the *source*
table at `FREEZE.json:table_relpath`; `upload_folder` ships `work/release/zero-v1/`. Gate 4
(`verify_bundle.py:41-46`) loads `BUNDLE/model.npz` on *both* sides of its comparison, so it is
self-consistent for any bundle. A wrong-but-coherent bundle passes all four gates. `--push` is
supported without `--build`, `README.md` is written *after* the gates, and no gate covers an ONNX
file at all. Today the bytes happen to match (`a7007b1a…` both places); the gate does not make that
true.

**The shipped tokenizer makes the fastembed route wrong on real inputs.** stella's
`tokenizer_config.json` carries `model_max_length: 32768`, `max_length: 8000`, and `tokenizer.json`
carries padding `{"Fixed": 512}` — verified in the released bundle AND in the doc-tower directory.
fastembed's `load_tokenizer` (`common/preprocessor_utils.py:120-125`) enables truncation at
`min(model_max_length, max_length)` = **8000**, and calls `enable_padding` only `if not
tokenizer.padding` — so stella's fixed 512 survives. Measured consequences:

| symptom | measured |
|---|---|
| a 1202-token query is **not** truncated to 512 | max-abs 4.5e-4 vs the numpy path — 45x the gate |
| `embed([long, short])` | **raises** `ValueError: inhomogeneous shape` (`onnx_text_model.py:82`) |
| every query padded to 512 | 2.53 ms/query end-to-end vs the card's published **0.38 ms** |
| after sanitising the two files | 0.211 ms/query, mixed batches work, cos vs numpy 0.999999989 |

Queries under 512 tokens pass perfectly (214 of them, max-abs 2.7e-8) — precisely the "passes on
synthetic, fails on real" class. **Worse on the doc side**: the index was built at `max_length 512`
(`FREEZE.json:encoder_spec`), so documents of 513–8000 tokens served through fastembed would not
reproduce the index, and long documents are common where long queries are not.

## T0 — bind the release path — **DONE 2026-09-03**

1. `build()` rebuilds the staging dir from scratch against an explicit **manifest**; gate 5 refuses
   any extra or missing file; `--push` requires `--build` in the same invocation. ✔
2. Gate 1 hashes **`OUT/model.npz`** against `FREEZE.json`, not only the source. ✔
3. Gate 4 loads the reference from the **frozen source path** *and* imports the **staged**
   `zero_encoder.py`, so neither side can be the substituted file. The staged-import half was the
   bigger hole: both gates read `m11/release/zero_encoder.py` and never executed what shipped. ✔
4. ~~ONNX sha recorded in a passing parity JSON~~ — **deferred to T2 with a different design.** A
   recorded verdict binds the file to a claim, not to the arithmetic; T2's gate re-runs parity
   in-gate. No ONNX artifact may be staged until then.
5. `README.md` is generated **before** the gates and gate 6 executes its python against the staging
   dir, offline, with the substitution count asserted so a card edit cannot redirect the gate at
   the published bundle. ✔
6. ~~Upload private → verify → flip visibility~~ — **unavailable: the repo was never private.**
   See `m11/STATUS.md`. Upload → re-download at the returned commit → compare file by file, which
   is what remains meaningful. ✔

Also fixed here: `config.json`'s frozen fields now come from `FREEZE.json` with the mutable
`.meta.json` sidecar cross-checked against it (it previously *set* the shipped preproc rule), and
`fallback_token_id` is read from `encoder_spec.cls_id` instead of being hardcoded.

`test_gates.py`: 1 positive control + 11 breakages, each asserting the refusal message matches the
reason. Reviewed by Codex and Fable; both broke the first implementation. See `m11/STATUS.md`
§Scope note for what was deliberately NOT built.

## T1 — sanitise the tokenizer — **DONE for `zero` 2026-09-03; T3 must repeat it for the doc tower**

Ship `model_max_length: 512`, `max_length: 512`, and `tokenizer.json` `padding: null`. Done in
`push.sanitise_tokenizer`, so it is part of every build rather than a one-off edit. Gate 7
(`verify_tokenizer.py`) measures what `fastembed.load_tokenizer` actually returns: truncation 512,
padding `length: None`, a mixed `[long, short]` batch rectangular at `[512, 512]`, and masked ids
identical to the numpy encoder's. Conformance after the edit is unchanged at 5.513e-07 max-abs,
confirming the numpy path is byte-identical. This does not
touch the frozen rule: `zero_encoder.py` calls `no_padding()` and takes truncation from
`config.json:preproc.max_length`, so the numpy path is byte-identical before and after; only
fastembed reads the changed fields. Re-run `verify_bundle.py` on the edited bundle, state the edit
in both cards, and gate a fastembed batch containing a >512-token text beside a short one.

**Editing the file is the only available mechanism, not a preference.** Two halves, different owners:

- **Truncation 8000 is ours and is not a fastembed bug.** fastembed truncates at
  `min(model_max_length, max_length)` as documented; our repo declares 8000 because it copies
  stella's file, while the frozen rule and the index are 512. Dylan's own **qdrant/fastembed #689**
  (2026-08-24) records that there is no API override — "the only override today is
  `specific_model_path` with an edited `tokenizer_config.json`". #689 is the mirror case (a config
  capping models *below* upstream); ours sits *above* our real limit. Related: #685/#687
  (`max_length: 0`), #500 (`sys.maxsize` sentinel), #531 (docstring drift).
- **Fixed-512 padding surviving is an upstream defect** — see §T4.

Ruled 2026-09-03 (Dylan): edit, and state the deviation in both cards.

## T2 — zero's query path → ONNX — **DONE 2026-09-03**

`m11/release/export_onnx.py`, results in `results/m11_zero_export.json`, live at commit
`fb8e5c5b`. The design below was followed as written; ten checks, all passing:

| check | measured |
|---|---|
| standard domain only, opset 17, checker | 13 op types, domain `''` |
| 1,024 **real dev queries** (`heldout-train`) vs the numpy encoder | max-abs **4.47e-08**, min-cos **0.99999976** |
| batch invariance, padded beside a 600-token query | 1.86e-08 |
| `model_tokens` + masked mean + L2 == `model.onnx` | 2.77e-08 |
| literal `[PAD]` / `[CLS]` / `[SEP]` / `[UNK]` / `[MASK]` | 1.49e-08 |
| 13 edge cases at b=1 (lengths 511/512/513, unmappable scripts, 120-char word, …) | 5.96e-07 |
| permutations of one bag agree | 1.49e-08 |
| all-masked row returns the `[CLS]` fallback | **0.0** |
| cost, 1 thread | s=8 **0.047 ms**, s=512 **1.22 ms** |
| **negative control**: the count-mask IS load-bearing | defective graph off by **1.43e-02** on a padded batch of literal-`[PAD]` fixtures, **3.73e-08** on the 1,024 dev queries |

**Reachability of that defect, measured 2026-09-03 — it is NIL, and the first write-up of this row
overstated it.** Two conditions must BOTH hold: the text must tokenize to id 0, and that row must
sit in a padded batch (so its id-0 position counts the batch's padding zeros). Consequences:

- run individually at b=1, each literal-`[PAD]` fixture is off by only ~1e-8 — no padding, nothing
  to miscount. The 1.43e-02 comes from running the three as a BATCH;
- **0 of 7,325 real dev queries contain token id 0**, which is the only route in. `[UNK]` is 100,
  truncation cannot introduce 0, and a WordPiece miss goes to `[UNK]`. So real text never gets
  there;
- 2,048 real dev queries in padded batches: **4.47e-08**. A maximally ragged batch (8-token query
  beside a 512-token one): **8.94e-08**.

So the check guards a graph-construction error with no user-facing consequence. Keep it — it is
two lines and it pins the count semantics — but do not cite it as a caught production risk.

Two deviations from the plan, both deliberate:
- **The ONNX graphs are not optional.** `--with-onnx` was removed: the card documents the files, so
  a build that omits them ships an incoherent card (gate 6 caught exactly that).
- **Gate 8 re-runs `export_onnx.py --check` against the STAGED graphs** rather than checking a sha
  against a recorded `"pass": true`, which was the plan's T0 item 4. A recorded verdict binds the
  file to a claim, not to the arithmetic; the negative-control test corrupts the staged table and
  requires gate 8 to notice.

### Design as executed

Rule (`m11/release/zero_encoder.py:79-88`): unique ids with counts `c_u`, weight each unique row by
`sqrt(c_u)`, divide by `sum_u sqrt(c_u)`, L2 normalize.

**Do not use ONNX `Unique`** — without an axis it flattens the batch, with one it uniques whole
slices, so neither is per-row. Use the per-occurrence identity (verified exact, both sides 4.449490
on `"the the the the the the"`):

    sum_u sqrt(c_u)·row_u == sum_i row_{t_i}/sqrt(c_{t_i});   sum_u sqrt(c_u) == sum_i 1/sqrt(c_{t_i})

Counts without `Unique`: `Equal(ids[:,:,None], ids[:,None,:])` → `(b,s,s)`, mask the **key** axis,
`ReduceSum` → `(b,s)`. Graph, opset 17, standard domain only:

    w    = mask / sqrt(max(c, 1))                       # mask on BOTH count-key axis and w
    rows = Gather(TABLE_INT8, ids) → Cast(f32) * scale[ids][:,:,None]
    num  = ReduceSum(rows * w[:,:,None], axis=1)
    vec  = num / max(ReduceSum(w, axis=1), EPS)
    out  = Where(‖vec‖ ≤ EPS, FALLBACK, vec/‖vec‖)      # EPS 1e-6, FALLBACK = normalized row 101

Table stays an **int8 initializer + fp32 per-row scale**, dequantized in-graph: ~31 MB, against
125 MB for fp32 rows. Gather→Cast→Mul is **bit-identical** to `rows_int8.astype(f32) *
int8_scale[:,None]` (measured). The pooled *output* is not — `ReduceSum` order differs from numpy's
`.sum(0)` — max-abs 8.9e-8 padded, 5.4e-7 at b=1, comfortably inside gate 2.

Two graphs from one table: `model.onnx` `(b,1024)` pooled+normalized for direct ORT callers, and
`model_tokens.onnx` `(b,s,1024)` emitting `row_{t_i}/sqrt(c_{t_i})` for fastembed.

Gates → `results/m11_zero_export.json`, each recording the achieved number and the file sha256:
1. zero custom-domain ops, opset 17, `onnx.checker` passes.
2. vs `ZeroQueryEncoder(variant="int8")` on ≥512 **real dev queries**: min-cos ≥ 1−1e-6, max-abs
   ≤ 1e-5. (§11.4's 1e-4/1e-3 is the floor for a *learned* port; this is the same arithmetic twice.)
3. **Batch invariance** — alone vs inside a padded batch beside a 500-token query, to 1e-6.
4. `model_tokens.onnx` + masked mean + L2 == `model.onnx`, to 1e-6.
5. Fixtures must include **`"[PAD]"` and `"[PAD] [PAD] hello"`**. See §Corrections: a graph masking
   the weight but not the count passes every ordinary query and fails only on literal `[PAD]`.
6. Edge cases: single-token, all-tokens-identical, mixed repeat+unique at 512, permutations of one
   bag, lengths 510–513, `[CLS]`/`[SEP]`/`[UNK]`/unused ids, empty and all-masked.
7. **Cost row**: measure s=8 vs s=512 latency (0.032 ms vs 1.29 ms single-thread — the S×S term is
   not free) and publish the post-T1 fastembed figure, not the 0.38 ms table-only number.

## T3 — document tower

Export exists (`work/m9onnx/stella-400M-doc/`, `results/m9_doc_export.json`: opset 17, no custom
ops, fp32 min-cos 0.99999940). Four gaps, all blocking publication:

- **`model_fp16.onnx` FAILS the mandate tolerance**: measured max-abs **1.37e-3**, min-cos
  **0.99970**, output norms 0.9997–1.0004 — the final normalize ran in fp16. The recorded `pass:
  true` covers only fp32 (`export_doc_model.py:150` builds its session on `p32`). Re-export with the
  normalize and Dense in fp32 (`op_block_list`); **if it still misses 1e-4/1e-3, publish fp32 only.**
- **No `config.json`** in the directory — `fastembed.load_tokenizer` raises on it as-is, so the repo
  as planned would be unusable through fastembed.
- **No `model_tokens.onnx`.** Masked mean is linear, so `mean(W·h_i+b) == W·mean(h_i)+b`; measure it
  here rather than inheriting the claim (see §Corrections on where that 6.3e-08 actually came from).
- **Recorded fp32 parity used word salad** from a 20-word vocabulary (`export_doc_model.py:30,34`),
  n=40, never real text. Re-run on real passages. Good news, measured: fp32 batch invariance is
  exact, and fixed-512 padding gives bit-identical output, so the attention mask survived export.

Card must carry: `base_model: NovaSearch/stella_en_400M_v5` and the pinned revision `ffeb2b7e…`;
**the Apache-2.0 lineage** — stella is trained from `Alibaba-NLP/gte-large-en-v1.5` and its
`modeling.py` is "Copyright 2024 The GTE Team Authors and Alibaba Group", so ship the Apache text
and notices alongside MIT (the pinned stella snapshot has no LICENSE file, which is a gap to fill,
not permission to omit); a statement that this is a **format conversion with unchanged weights**;
the two mandatory `config_kwargs`; the head definition (masked mean → `2_Dense_1024` → L2); measured
parity per graph; and that it is **document-only** — stella's query side needs the `s2p_query`
prompt and fastembed's `query_embed` adds no prefix, so symmetric use through fastembed is silently
wrong. Note for the log: Alibaba is "OK WITH JUSTIFICATION" under the vendor rule and the decision
log records NovaSearch as CLEAN without the lineage; the justification is that this is a conversion
of an already-chosen teacher, not a new component choice.

## T4 — fastembed fork branch

Fork `Dylancouzon/fastembed` created 2026-09-03. Clone to `/home/dylan/fastembed` (sibling of this
repo, **not** inside it), branch `zero-query-encoder`.

Two integration routes; test both and pick on evidence:
- **MEAN on `model_tokens.onnx`** (`normalization=True`). `fastembed.mean_pooling`
  (`common/utils.py:26-32`) is the masked mean, divisor = real token count, a positive scalar the
  normalize annihilates. Measured vs numpy: **3.3e-8**. Loses the frozen fallbacks, which are
  unreachable in practice (no table row has norm ≤ EPS; min 0.196, row 101 = 2.15).
- **`PoolingType.DISABLED` on the pooled `model.onnx`**, which preserves the fallbacks exactly.
  DISABLED exists in 0.8.0; whether the pipeline accepts a `(b,1024)` graph is **untested** — the
  first draft asserted it does not, which was unfounded.

Gate serving parity end-to-end against the numpy encoder (the M9 pilot only ever got a *description*
accepted for anything but nano), **including `parallel>1`**: `CustomTextEmbedding` does not override
`_get_worker_class()`, so the inherited worker constructs `OnnxTextEmbedding`, which cannot resolve a
runtime-registered name. Unverified but cheap to check, and a serial-only smoke would miss it.

**Upstream defect in `load_tokenizer` (under independent verification, 2026-09-03).**
`common/preprocessor_utils.py` overrides truncation **unconditionally** but sets padding only
`if not tokenizer.padding`. Nothing checks the two are consistent, so a repo shipping
`padding = Fixed(512)` with truncation 8000 leaves every 513–8000-token input neither padded nor
truncated → guaranteed ragged batch → opaque `ValueError: inhomogeneous shape` at
`onnx_text_model.py:82`. Dynamic padding is the intended default (fastembed's own
`enable_padding()` call yields `length: None`, i.e. batch-longest). Present on current `main`
(HEAD `a34e7bc`), not only 0.8.0. No issue or PR covers the padding path — #689, #685/#687, #500 and
#531 are all truncation precedence.

**CONFIRMED and FIXED on the fork, 2026-09-03.** Independently verified, then reproduced directly:
it is a **0.8.0 regression that breaks `thenlper/gte-base`, a model on fastembed's own supported
list** (ships `Fixed: 128` against truncation 512).

| | padding | lengths for `["hello world", 200x"retrieval "]` | result |
|---|---|---|---|
| `main` a34e7bc | fixed 128 | `[128, 202]` | `ValueError: inhomogeneous shape` |
| patched | dynamic | `[202, 202]` | works; batched vs single **max abs delta 0.0** |

Cause: `800f388` (PR #588, colmodernvbert) added the `if not tokenizer.padding` guard so a
left-padding model could keep `direction: Left`; first tagged in `v0.8.0`. The call was
unconditional in 0.7.4, where gte-base worked. CI misses it because every model test uses strings
shorter than 128 tokens.

Fix (`Dylancouzon/fastembed` branch `fix-fixed-padding-ragged-batch`, commit `a8390e3`): always
`enable_padding`, never passing `length`, carrying `direction`/`pad_id`/`pad_type_id`/`pad_token`/
`pad_to_multiple_of` over from the tokenizer's own config so #588's left-padding case still works.
Three regression tests in `tests/test_preprocessor_utils.py`; the two that matter fail on unpatched
`main` and pass patched. Rejected alternatives: capping `length` to the truncation limit does not
fix the reported case (128 < 512 is unchanged); forcing `length = truncation` pads stella to 8000.

**T4 must run against this branch, not released 0.8.0** — that is the point of fixing it first.
**Amended 2026-09-03, measured after T1:** the reason above is now wrong for *our* repo. With
`padding: null` shipped, `not tokenizer.padding` is true, so stock 0.8.0 installs its own dynamic
padding and the defect is not triggered — gate 8 passes on unpatched **0.8.0** (`verify_tokenizer.py`:
truncation 512, padding `length: None`, mixed `[long, short]` batch `[512, 512]`). Confirmed by
Codex. So released 0.8.0 is the correct regression target for the sanitised bundle, and the fork
branch remains necessary only to validate the fix against *unsanitised* tokenizer files — which is
still worth doing, for a different reason than the plan gave.

**Blast radius, audited 2026-09-03:** of the 34 `TextEmbedding` models with an HF source, four ship
fixed padding and **only `thenlper/gte-base` has it below its truncation limit** (128 vs 512);
`gte-large`, `all-MiniLM-L6-v2` and `siglip2-base-patch16-224` are padding == truncation and fine.
The sparse/late-interaction/cross-encoder/multimodal registries were not audited. **The exposed
population is `add_custom_model` repos** — which is how we hit it, and why this matters to us more
than the supported-model count suggests.

**Reported upstream as qdrant/fastembed#703** (2026-09-03, Dylan's go), Codex-reviewed before
filing. The issue offers a PR but does not open one; **PR scope remains undecided**. Codex's review
also caught a bug in the first patch — `padding.get("pad_token", tokenizer_config["pad_token"])`
evaluates its default eagerly, so a tokenizer serializing its own pad token with no
`tokenizer_config.json` entry would newly raise `KeyError`; fixed in `c16cce6` with a test that
fails on the eager form. Claims cut from the draft as unverifiable or wrong: an exhaustive
blast-radius audit (siglip2 is not in v0.8.0), the reverse-shape "does not crash" claim, and an
inference about #588's author intent.

No model-integration PR this milestone. Leave the branch pushed and PR-ready; a PR would still need
canonical reference vectors per `CONTRIBUTING.md` and an honest description — zero **missed**
`LR-dense-pertask 0.4583` at 0.4339 (CI-resolved), its fused variant ties OpenSearch.

## T5 — card fixes before anything goes public

`MODEL_CARD.md:44` sets `q = enc.encode([...])`, a `(1,1024)` array; `:90` then calls
`enc.encode([q])[0]`, which raises. Fix, then execute every block in the generated README.

## Corrections — claims the first draft of this plan got wrong

| claimed | actual |
|---|---|
| "bit-identical to what the numpy encoder does" | dequantization only; the pooled output differs by ≤5.4e-7 (ReduceSum order) |
| "the ONNX path is masked so it is immune" to the padding trap | immune in correctness, not cost; and fastembed truncates at 8000, not 512 |
| "at s ≤ 512 this is a trivial cost against a gather" | 40x: 1.29 ms at s=512 vs 0.032 ms at s=8 |
| "gate 3 is what catches a masking bug" | catches count-only masking; **weight-only masking fails only on literal `[PAD]`** |
| "`DISABLED` … a shape its pipeline does not expect" | unfounded; a `(b,1024)` graph may be fine, untested either way |
| "artifacts exist and pass" (doc tower) | fp32 pooled only; fp16 fails §11.4, and n=40 on synthetic word salad |

**Provenance defect in inherited evidence.** `results/m9_doc_export.json`'s `fastembed_local` block
cannot have been written by `export_doc_model.py:try_fastembed` — that function emits
`min_cos_vs_onnxruntime`/`shape`/`pass` and passes `model_file="model_fp16.onnx"`; the JSON has
`min_cos_vs_self_contained_graph`/`max_abs`/`finding` and names `model_tokens.onnx`. Whatever
produced it is not the named script. Treat that block as unattributed and re-measure; the 6.3e-08
figure quoted for masked-mean linearity inherits the same doubt.

## Open for Dylan

- ~~Whitepaper contingency~~ — **CLOSED 2026-09-03 (Dylan): deferral stands, no contingency.** Both
  reviews proposed a dated fallback if the M10 budget is refused; ruled against. Not to be
  re-proposed. Nothing in M11a needs redoing either way, provided the doc-tower repo name and both
  cards avoid presupposing `nano`.
- **The tokenizer edit changes published bytes** in an already-released repo. It does not touch the
  frozen table or the numpy path, but it is a change to a shipped artifact and is recorded here
  rather than made silently.

## Traps carried forward

- Padded positions must contribute zero to BOTH the count and the weight (§T2 gate 5).
- `results/perquery.json` is irreplaceable and is not touched here.
- The reserved four and their single confirmatory access stay unspent; M11a reads no eval set beyond
  dev queries used as parity fixtures.
