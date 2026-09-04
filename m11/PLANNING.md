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
| 8 **raw id vectors** — `[unused*]` ids, lengths 510/512, repeat+unique mix, last table row, `[UNK]` alone | 1.19e-07 |
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

Gate 6's list named three cases text cannot reach — `[unused*]` ids (1–99, unreachable by any
tokenization), length 510, and a 512-length repeat+unique mix — so those go through raw id vectors
compared against `_encode_ids`, which is the frozen rule itself.

**Gate 7's cost row is only half done, and the other half is blocked on T4.** The ONNX graph
latency is measured and published (0.047 ms at s=8, 1.22 ms at s=512, one thread). The
*fastembed end-to-end* figure the plan asks for cannot be measured until fastembed actually serves
the graph, which is T4. Until then the card's 0.38 ms row remains the numpy path, labelled as such.

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

## T3 — document tower — **IN PROGRESS 2026-09-03**

Repo (new, PUBLIC, Amendment A ruling 2): **`DylanCouzon/stella-en-400M-v5-doc-onnx`**.
Re-exported from the pinned revision by `m11/release/export_doc.py`; pushed by
`m11/release/push_doc.py`; card `m11/release/MODEL_CARD_DOC.md`; numbers
`results/m11_doc_export.json`. The M9 artifacts and `m9src/export_doc_model.py` are left as the
historical record and are NOT the shipped path — see §T3 corrections.

**Ships ONE graph.** `model.onnx`, fp32, pool→Dense→L2.
- No `model_tokens.onnx`: `PoolingType.DISABLED` + `normalization=False` serves an already-pooled
  graph unchanged (max-abs 2.46e-08). The M9 note claiming fastembed "has no slot for a dense layer
  after pooling" is **wrong** and would have cost a third 1.75 GB file.
- No `model_fp16.onnx`: **rejected on measurement, see below.** `export_doc.py --fp16` still builds
  it; `push_doc.build()` refuses to run if one is sitting in the export dir.

**Fixtures are frozen**: `m11/release/doc_fixtures.json`, 259 real nq-250k passages in six strata
by pinned-tokenizer token count — tiny/short/mid/near 48 each, **boundary(511–513) 19**,
**over(514–1131) 48**. Strata are re-asserted on every load. hotpotqa is not used (1.6 GB / 5.2M
strings, nothing over 445 tokens). Replaces the inherited n=40 of 20-word word salad.

### Measured 2026-09-03 (24–259 real passages; final numbers in the result JSON)

| | measured |
|---|---|
| fp32 graph vs torch | cos 0.99999988, max-abs 2.9e-07, **batch invariance exactly 0.0** |
| fp32 graph on CUDA vs on CPU | min-cos 1.000000, max-abs 9.07e-05, 0/259 bad — **no accelerator caveat needed** |
| the OLD M9 fp16 on real text | max-abs 1.355e-03 — **fails**, as the plan said |
| fp16 (backbone fp16, head fp32) vs torch, **CPU** | cos 0.99999923, max-abs 1.79e-04 — passes §11.4 **and means nothing** |
| fp16 vs fp32 reference, **CUDA** | **min-cos 0.662, mean 0.989, 255/259 below 0.999** — unusable |
| fp16 latency | CPU **9.9x SLOWER** (2048 s vs 207 s); CUDA 1.78x faster |
| fp16, hand-built all-zero attention mask | **NaN**; fp32 returns a unit vector. Unreachable by any tokenizer (`encode("")` → `[CLS] [SEP]`) |

### The T3 finding worth carrying forward

**A CPU parity gate cannot qualify a reduced-precision ONNX graph.** ONNX Runtime has no fast CPU
fp16 kernels, so it up-converts to fp32 — which is simultaneously why the fp16 graph *passes* CPU
parity at cos 0.99999923 and why it runs 9.9x slower there. The passing number certifies fp32
arithmetic on fp16 weights, i.e. a code path nobody would choose fp16 for. Run it on the provider
the precision exists for and 255 of 259 passages are wrong, across every stratum from 7 tokens up.
Had this shipped on the CPU number with a "use it on GPU" note, the card would have carried
**exactly inverted** advice. `results/m11_doc_fp16_gpu.json`, incl. what would change the verdict.

Two review rounds both reasoned about fp16 from CPU evidence alone — one arguing to ship it, one
arguing to drop it for want of GPU evidence. Neither would have caught the inversion; only the
measurement did. The reviewer who said "no supported execution path has demonstrated a benefit"
was right for a reason weaker than the true one.

### Traps this task found — all of them cost nothing only because they were caught

- **`2_Dense_1024` HAS A BIAS** (`linear.weight` + `linear.bias`). A rewrite selecting the weight
  by rank dropped it; the exact-key assertion is what caught it. The matrix is square, so a wrong
  or transposed key stays shape-valid and a torch reference built the same way would certify it.
- **`convert_float_to_float16(node_block_list=…)` emits an UNLOADABLE graph.** It inserts one
  cast-to-fp32 node per CONSUMER, all sharing a name and an output tensor (ORT: `two nodes with
  same node name`), and APPENDS them, breaking topological order (`onnx.checker`). `repair()`
  drops the byte-identical clones and stable-Kahn re-sorts: 2 clones, 41 positions. A
  level-by-level sort "works" but churns 3440 positions — use the stable form.
- **The block list must be DERIVED**, never hard-coded: nodes after the last `LayerNormalization`
  (17 of them). Hand-listing 14 omitted `/Constant_1` and `/Constant_2`, and
  `convert_float_to_float16` matches names EXACTLY — a rename silently blocks nothing. With the
  derived list the eps constants stay fp32 (`1e-09`, `1e-12`, verified in the converted graph).
- **A dot product is not a cosine, and here the difference decided a verdict.** `m9src`'s
  `(ref*got).sum(1)` is only a cosine when both sides are unit-norm; the fp16 graph's outputs are
  0.9995–1.0004. Same graph: **dot 0.99954 (fails 1e-4), true cosine 0.99999940 (passes)**.
  `export_onnx.py` had the same bare-dot form and now computes a real cosine and records norms.
- **`export_onnx.py` check 1 claimed "checker passes" and never ran the checker** (predicate was
  `not custom` alone), so `zero`'s live gate 8 asserted something it did not test. Fixed: the
  checker runs with `full_check=True` and the opset is asserted.
- **`parallel>1` cannot work** for `add_custom_model` repos in fastembed 0.8.0 — the worker builds
  `OnnxTextEmbedding`, which cannot resolve a runtime-registered name. A card sentence, not a gate.
- **A dtype census is what proves a filename.** A copied fp32 graph renamed `model_fp16.onnx`
  passes opset/domain/IO checks perfectly; the initializer census (fp16 everywhere except
  `dense.weight`/`dense.bias`) is what catches it.

### Tokenizer and config

`push.sanitise_tokenizer`'s edit applies here unchanged and matters more: `model_max_length` 32768
and `max_length` 8000 against an index built at 512, so fastembed would send documents of 513–8000
tokens through untruncated. `config.json` is stella's own with the two xformers flags set to what
the graph was exported with. **The plan's claim that the M9 directory ships `padding: Fixed(512)`
is wrong** — it holds `BatchLongest`, because `save_pretrained` recorded the transformers state at
export time. The pinned snapshot is the source of truth and is what the build copies.

### Licence

stella's own card frontmatter declares **`license: mit`**, so the card declares MIT because
upstream does and claims no separate licence. The `Alibaba-NLP/gte-large-en-v1.5` Apache-2.0
lineage is recorded as attribution only; `modeling.py` is not redistributed here, so **no Apache
licence text is shipped**. This supersedes the earlier instruction to ship Apache text alongside
MIT, which would have implied undocumented dual licensing.

### Gates (`push_doc.py`)

(1) `export_doc.py --check` re-derived against the staged files on the **full** fixture set — no
`--n`, since a shortened run still exits zero; (2) manifest exactness; (3) the tokenizer as
`fastembed.load_tokenizer` reads it; (4) fastembed serves the staged graph via `DISABLED` and
agrees with ORT; (5) the card's python blocks execute offline against the staged bytes. Then:
create PRIVATE with `exist_ok=False` (a pre-existing destination is refused, the repo id is a
constant), upload, verify the returned commit against a **hash snapshot captured after the last
gate**, flip PUBLIC. `push.py` compares the remote against the staging dir as it stands at
verification time, which would compare changed bytes against themselves.

## T4 — fastembed fork branch

Fork `Dylancouzon/fastembed` created 2026-09-03. Clone to `/home/dylan/fastembed` (sibling of this
repo, **not** inside it), branch `zero-query-encoder`.

Two integration routes. **Both questions below were open in the plan and are now CLOSED by T3
measurements — do not re-test them:**
- **MEAN on `model_tokens.onnx`** (`normalization=True`). `fastembed.mean_pooling`
  (`common/utils.py:26-32`) is the masked mean, divisor = real token count, a positive scalar the
  normalize annihilates. Measured vs numpy: **3.3e-8**. Loses the frozen fallbacks, which are
  unreachable in practice (no table row has norm ≤ EPS; min 0.196, row 101 = 2.15).
- **`PoolingType.DISABLED` on the pooled `model.onnx`** preserves the fallbacks exactly. **Settled:
  the pipeline DOES accept a `(b,1024)` graph** — `DISABLED` + `normalization=False` returns the
  model output untouched, measured against direct ORT at **max-abs 0.00e+00** (T3 gate 4). This is
  the preferred route: one graph, no per-token duplicate.

Still to do: gate serving parity end-to-end against the numpy encoder (the M9 pilot only ever got a
*description* accepted for anything but nano). **`parallel>1` is NOT a gate — it cannot pass:**
`CustomTextEmbedding` does not override `_get_worker_class()`, so the worker constructs
`OnnxTextEmbedding`, which cannot resolve a runtime-registered name and raises
`ValueError: Model ... is not supported in OnnxTextEmbedding`. Measured, not inferred. It is a card
sentence and, if anything, a candidate upstream fix — not a release gate.

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

**DONE 2026-09-03 — `m11/release/verify_fastembed.py`, 6 checks, green on BOTH stock 0.8.0 and
the fork branch** (`results/m11_fastembed_serving_{pypi,fork}.json`; the fork carries the same
`__version__`, so the result file is keyed on the import path, not the version).

| check | number |
|---|---|
| 2 parity vs the staged numpy encoder, 1,024 real dev queries | max-abs **4.470e-08**, min-cos 0.999999881 |
| 2b parity on 4 inputs past the 512-token rule | max-abs 5.513e-07 |
| 3 batch invariance, bs 1 vs 64 and beside a 1,200-token query | 4.470e-08 |
| 4 served vectors unit-norm with `normalization=False` | [0.9999999, 1.0000001] |
| 5 MEAN route on `model_tokens.onnx`, direction | min-cos 0.999999881 |

Serving adds nothing to the graph: check 2 equals T2's direct-ORT number exactly. Route as
planned — `PoolingType.DISABLED` + `normalization=False` on the pooled `model.onnx`.

**Negative control (`--negative-control`): truncation 8000 put back is caught at max-abs 4.475e-04**,
44x the threshold. Two facts it settles, both contrary to the plan's reasoning above:
- with `padding: null` shipped there is **no ragged-batch crash** — stock 0.8.0 pads dynamically
  and serves a **silently wrong vector**. The crash is not our exposure; a wrong vector is.
- check 2 alone would not have caught it: dev queries are short, and the tokenizer rule only bites
  past 512 tokens. That is why 2b exists.

`parallel=2` fails as T3 measured (`RuntimeError: Thread unexpectedly terminated`, each worker
raising `Model ... is not supported in OnnxTextEmbedding`). Recorded, not gated. It also floods
stdout from every worker process, so the probe silences fds 1/2 at the OS level — a Python-level
redirect does not reach a child, and the flood deadlocked the first run against a full pipe.

### T4 REOPENED and widened — full FastEmbed integration (Dylan, 2026-09-03)

*"I want a full fastembed integration proof of concept ... at a level that is mergeable (so no
crazy custom implementation) ... the model cards should reflect as if those were Fastembed models
first so we promote our library. Use the same branch (maybe rename it)."* Then: *"the card should
assume the model is in Fastembed, won't be released until then. You can point the card to our
branch for now."*

Branch `fix-fixed-padding-ragged-batch` → **`add-constella-models`**, carrying the #703 padding fix
(2 commits) plus the integration. **Ruled fine by Dylan**: this branch will not be merged, a clean
single-concern PR follows when we are ready. Same ruling covers the `DylanCouzon/...` registry
names, which upstream would want under `Qdrant/`.

**The integration**, 3 files, no new machinery:
- `fastembed/text/pre_pooled_embedding.py` — `PrePooledEmbedding`, mirroring `pooled_embedding.py`:
  `_post_process_onnx_output` returns the model output untouched. Needed because neither existing
  class fits — `OnnxTextEmbedding` expects per-token output, and the pooled classes would apply a
  masked mean on top of an already-pooled vector. `constella-zero` pools with count-saturated sqrt
  weights; the stella tower applies a dense head AFTER pooling, so pooling cannot be a
  post-processing step at all.
- `text_embedding.py` — one import, one registry entry.
- `tests/test_text_onnx_embeddings.py` — 2 canonical vectors from the reference implementations
  (the numpy encoder; the torch module for the tower), per `CONTRIBUTING.md`.

**Measured on a clean HF download** (`scratchpad/check_builtin.py`): both models listed; zero vs the
numpy reference **max-abs 4.470e-08** over 1,024 dev queries; both canonical vectors pass upstream's
own `atol=1e-3` assertion; doc output unit-norm.

**`parallel>1` works now, and could not before.** `add_custom_model` cannot support it —
`CustomTextEmbedding` has no `_get_worker_class`, so the worker builds `OnnxTextEmbedding` and
cannot resolve a runtime-registered name. Natively: **3.725e-08** (query, vs numpy) and
**0.00e+00** (doc, vs serial). This is the concrete argument for the PR. *Measuring it needs a
`__main__` guard — the workers spawn and re-import the script; without one the first attempt
re-ran the whole file per worker and looked like a fastembed failure.*

**Card gates now run against the fork** (`FASTEMBED_FORK` in `push.py`/`push_doc.py`), because the
cards use built-in model names that only resolve there. `TextEmbedding(NAME)` is rewritten to add
`specific_model_path=BUNDLE_DIR` so the gate still certifies the STAGED bytes, and a literal repo id
inside `TextEmbedding(...)` is refused. That rewrite is applied **uncounted** — every negative
fixture in `test_gates.py` is a card with no FastEmbed block, so a fixed substitution count of 3
fails them for the wrong reason.

**A card teaching `add_custom_model` breaks when the model ships natively** — measured: the old doc
card raised `ValueError: already registered` the moment the fork registered the name. That is why
both cards use the built-in name only.

No model-integration PR this milestone. Leave the branch pushed and PR-ready; a PR would still need
canonical reference vectors per `CONTRIBUTING.md` and an honest description — zero **missed**
`LR-dense-pertask 0.4583` at 0.4339 (CI-resolved), its fused variant ties OpenSearch.

## T6 — rename `constella-zero` and rewrite both cards (after T4; Dylan, 2026-09-03)

**Naming ruling.** The locked family name is `constella` (`m8/LEDGER.md` §6.1: constellation +
stella, navigate by fixed stars, no engine). The milestone suffix is **dropped**, so the query
model is **`constella-zero`** and the tower will be `constella-nano`. `zero` shipped as
`zero-query-encoder-v1` because the open ruling at `m8/LEDGER.md:716` was never sought before the
push; HF `move_repo` leaves a redirect, so the old URL keeps working.

The stella ONNX doc tower keeps `stella-en-400M-v5-doc-onnx`: it is a **format conversion of a
third-party model**, not a constella artifact, and a `constella-*` name would misattribute
someone else's weights. Confirm with Dylan if he wants otherwise.

**Card rewrite (Dylan's ask).** Both cards, `constella-zero` especially:
- **use the fastembed examples** — T3 settled the route, `PoolingType.DISABLED` +
  `normalization=False` on the pooled graph, bit-identical to direct ORT. Both cards should show
  the same registration pattern so the pair reads as one product.
- **remove the competitive comparison and the missed bar** — LightRetriever `0.4583`, the
  OpenSearch tie, "missed its own release bar". An internal project bar means nothing to someone
  downloading the model, and the comparator table belongs in the whitepaper.
- **KEEP** the measured nDCG@10 numbers and the **stella contamination disclosure** (ArguAna,
  FiQA and FEVER are in stella's training data). That is not a competitive claim — it is what a
  reader needs to interpret the numbers at all.
- **be more informative about the model itself**: what it is, the pooling rule, the cost rows
  (query asset / document index / hydration / CPU latency), the shipped file choice, limits.

**`instructions-m11.md` deliverable 1 must be amended in the same change** — it currently *requires*
the cards to carry the competitive claims ("Zero tier wins exist and the model cards must say so"),
so removing them without amending it leaves a rule the repo knowingly breaks.

Touch points for the rename: `push.py:363` default repo id, `MODEL_CARD.md` `REPO_ID`
substitution, `m11/STATUS.md`, `instructions-m11.md` Amendment A ruling 1, `CLAUDE.md` M7 entry.

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
