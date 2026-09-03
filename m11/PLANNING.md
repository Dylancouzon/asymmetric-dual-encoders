# M11a plan — ship `zero` end to end

Mandate + the four rulings: `instructions-m11.md` Amendment A. Branch `m11-work`, headless
commit-and-push contract. Status/pointers: `m11/STATUS.md`. No quality number is read or written
here, so `m9src/guard9.py` registration does not apply; `push.py`'s four release gates do.

## T1 — flip `zero` public

**`push.py --public` does not flip an EXISTING repo.** `create_repo(exist_ok=True)` is a no-op on
visibility, so the current call would report `PUBLIC →` while the repo stayed private — a gate that
fails open, the same class of bug the 2026-08-28 reviews caught twice. Fix `push()` to call
`HfApi().update_repo_settings(repo_id, private=private)` after `create_repo` (hub 0.36.2;
`update_repo_visibility` is the deprecated spelling), then re-read `api.repo_info(...).private` and
refuse if it disagrees with the flag. Re-run all four gates before the flip.

## T2 — zero's query path → ONNX

The released rule (`m11/release/zero_encoder.py:79-88`): tokenize, take **unique** ids with counts
`c_u`, weight each unique row by `sqrt(c_u)`, divide by `sum sqrt(c_u)`, L2 normalize.

**Do not use the ONNX `Unique` op.** It is not per-row, so it breaks the moment fastembed hands the
graph a padded batch. Use the per-occurrence identity instead — exact, not an approximation:

    sum_u sqrt(c_u)·row_u  ==  sum_i row_{t_i} / sqrt(c_{t_i})       (each u contributes c_u terms)
    sum_u sqrt(c_u)        ==  sum_i 1 / sqrt(c_{t_i})               (same weights, so the
                                                                      denominator is free)

Per-occurrence counts without `Unique`, batch-safe and standard-op only: `Equal` on
`ids[:,:,None]` vs `ids[:,None,:]` → `(b,s,s)`, mask the key axis with `attention_mask`, `ReduceSum`
over it → `c` of shape `(b,s)`. At s ≤ 512 this is a trivial cost against a gather.

Graph, opset 17, no custom domains:

    w    = mask / sqrt(max(c, 1))                       # padded positions → 0
    rows = Gather(TABLE_INT8, ids) → Cast(f32) * scale[ids][:,:,None]
    num  = ReduceSum(rows * w[:,:,None], axis=1)        # (b, 1024)
    vec  = num / max(ReduceSum(w, axis=1), EPS)
    out  = Where(‖vec‖ ≤ EPS, FALLBACK, vec/‖vec‖)      # EPS 1e-6, FALLBACK = normalized row 101

**Keep the table as an int8 initializer plus the fp32 per-row scale, and dequantize inside the
graph.** That is bit-identical to what the numpy encoder does and keeps the graph at ~31 MB;
materialising fp32 rows would cost 125 MB (30522 × 1024 × 4) and an fp16 initializer would lose
precision the released artifact does not lose.

**Ship two graphs from one table**, the pattern M9 established for nano:

| file | shape | for |
|---|---|---|
| `model.onnx` | `(b, 1024)` pooled + normalized | direct ONNX Runtime callers |
| `model_tokens.onnx` | `(b, s, 1024)`, per token `row_{t_i}/sqrt(c_{t_i})` | fastembed, which pools itself |

fastembed's `MEAN` pooling then computes `sum_i y_i / S`, and `S` is a positive scalar that the
subsequent L2 normalize annihilates — so **`PoolingType.MEAN` + `normalization=True` reproduces the
frozen rule exactly.** `PoolingType.DISABLED` exists in 0.8.0 but is not needed and would ask
fastembed to post-process a shape its pipeline does not expect.

Gates (all must pass, `results/m11_zero_export.json`):
1. zero custom-domain ops, opset 17.
2. `model.onnx` vs `ZeroQueryEncoder(variant="int8")` on ≥512 real dev queries: **min-cos ≥ 1−1e-6,
   max-abs ≤ 1e-5.** The mandate's §11.4 tolerances (1e-4 / 1e-3) are the floor for a *learned*
   port; this is the same arithmetic twice and must be far tighter. State the achieved number.
3. **Batch invariance**: a query encoded alone equals the same query inside a padded batch beside a
   500-token one, to 1e-6. This is what would catch a `Unique`-shaped or masking bug.
4. `model_tokens.onnx` + MEAN + L2 equals `model.onnx` to 1e-6.
5. Edge cases: single-token query, all-tokens-identical query, max_length truncation at 512.

## T3 — document tower

Artifacts exist and pass (`work/m9onnx/stella-400M-doc/`, `results/m9_doc_export.json`). Two gaps:

- **No `model_tokens.onnx`** — the fastembed route for the doc side is unverified. Export the
  per-token variant (backbone → per-token `Dense(1024)`, no pooling, no normalize). Masked mean is
  linear, so `mean(W·h_i + b) == W·mean(h_i) + b`; M9 measured that equality at 6.3e-08 for nano.
  Verify it here rather than inheriting it.
- **Re-verify before publishing.** Re-run parity against the torch path on the artifacts as they sit
  on disk; they are gitignored and mutable, and were written 2026-08-30.

Publish fp32 + fp16 + per-token + tokenizer to a new **public** repo. Card must carry: stella
attribution and MIT, the pinned revision `ffeb2b7e…`, the two mandatory
`config_kwargs` (`use_memory_efficient_attention=False`, `unpad_inputs=False` — stella asserts on
xformers without them), the exact head definition (masked mean → `2_Dense_1024` → L2), measured
parity, and that this is the document half of an asymmetric pair whose query half is `zero`.

## T4 — fastembed fork branch

Fork `Dylancouzon/fastembed` created 2026-09-03. Clone to `/home/dylan/fastembed` (a sibling of this
repo, NOT inside it), branch `zero-query-encoder`. Register both models via `add_custom_model`
(`ModelSource(hf=…)`, `model_file="model_tokens.onnx"`, `pooling=MEAN`, `normalization=True`,
`dim=1024`) and prove end-to-end serving parity against the numpy encoder — the M9 pilot only ever
got a *description* accepted, never a served vector, for anything but nano.

No PR this milestone (ruling 3). Leave the branch pushed and PR-ready, and write down what a PR
would still need: canonical reference vectors per `CONTRIBUTING.md`, and an honest description —
zero **missed** `LR-dense-pertask 0.4583` at 0.4339 (CI-resolved), and its fused variant ties
OpenSearch. The card's framing carries into any PR text verbatim.

## Traps carried forward

- **stella's `tokenizer.json` ships with padding-to-512 enabled.** A naive `tokenizers` load puts
  ~500 `[PAD]` rows in every bag and cosine drops to 0.35 (`m11/STATUS.md`). The ONNX path is
  masked so it is immune, but any *tokenizer* comparison harness must call `no_padding()`.
- Padded positions must contribute zero to BOTH the count `c` and the weight `w`. A mask applied to
  only one of them is the likeliest silent wrong answer in T2, and gate 3 is what catches it.
- `results/perquery.json` is irreplaceable and is not touched here.
- The reserved four and their single confirmatory access stay unspent; M11a reads no eval set
  beyond dev queries used as parity fixtures.
