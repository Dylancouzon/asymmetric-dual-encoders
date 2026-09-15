# M21 FastEmbed handoff

## Root cause

`fastembed/common/utils.py:mean_pooling` expands the already-`int64` attention mask and explicitly casts it to `int64`. NumPy promotes `float32 * int64` to `float64`; pooling and normalization therefore stay `float64`, and FastEmbed exposes a `float64` vector. This is not Nano-specific. On upstream main it affects all 16 models routed through `mean_pooling`: 10 `PooledNormalizedEmbedding` models and six `PooledEmbedding` models. Zero and the Stella document tower pool inside their ONNX graphs and already return `float32`.

## Dtype variants and parity

Both candidates were run with `m14/parity.py`, the frozen checkpoint, and all 10 existing length-stratified fixtures, including raw token lengths 511/512/513. Environment: CPU, four threads, Python 3.12.14, NumPy 2.3.5, Torch 2.8.0+cu126, ONNX Runtime 1.29.0. Both returned `float32` from FastEmbed and passed every registered threshold.

| Variant / comparison | Minimum cosine | Maximum absolute error |
|---|---:|---:|
| A: Torch vs direct ORT | 1.0 | 1.1548399925231934e-07 |
| A: Torch vs FastEmbed | 1.0 | 1.1548399925231934e-07 |
| A: direct ORT vs FastEmbed | 1.0 | 0.0 |
| B: Torch vs direct ORT | 1.0 | 1.1548399925231934e-07 |
| B: Torch vs FastEmbed | 0.9999999403953552 | 1.0058283805847168e-07 |
| B: direct ORT vs FastEmbed | 1.0 | 1.043081283569336e-07 |

For both variants, maximum norm deviation across the three paths was `1.1920928955078125e-07`; the negative-control cosine was `0.1960805058479309` (maximum absolute difference `0.15240468084812164`). Variant A exactly matches the direct-ORT harness because both accumulate in `float32`. Variant B is marginally closer to Torch by maximum absolute error but one float32 ULP lower by minimum cosine. Neither distinction is material against the registered `1e-5` error and `0.9999` cosine limits.

**Decision: variant B.** Keep the existing `float64` accumulation and cast only the returned pooled array to `input_array.dtype`. Both variants have equivalent parity at the registered precision; B is the smaller compatibility change for 16 existing models and preserves their current summation precision. A would also change the intermediate arithmetic, including for non-`float32` input, without a measured parity benefit that warrants it.

The regression test asserts that `mean_pooling` output dtype equals its `float32` embedding input dtype. It failed against the unfixed implementation (`float64 != float32`) and passed after B.

## Demo branch

Local branch: `constella-research-preview`, based directly on `0dab99c` and clean at `eea7705`.

It contains four commits:

1. `83dbc5f` — remove a serialized fixed padding length while retaining padding direction and token settings, with regression tests.
2. `d04ccba` — avoid eager evaluation of a missing tokenizer-config `pad_token`, with a regression test.
3. `a4452ac` — preserve the embedding dtype at the `mean_pooling` return boundary, with the red/green regression test.
4. `eea7705` — native registrations and reference-derived canonical-vector prefixes for `DylanCouzon/constella-nano`, `DylanCouzon/constella-zero`, and `DylanCouzon/stella-en-400M-v5-doc-onnx`.

The authorized push was attempted with `git push -u origin constella-research-preview` and failed before authentication because the sandbox could not resolve `github.com` (exit 128). Nothing was pushed anywhere; no PR was opened. When DNS is available, that exact command is the remaining publication step.

## Verification

- Final M14 parity run: passed all comparisons, row norms, boundary cases, and the negative control with the variant-B numbers above.
- Changed-code unit tests: `tests/test_common.py tests/test_preprocessor_utils.py` — **8 passed**.
- Native smoke tests: all three names appear in `TextEmbedding.list_supported_models()` with dimension 1024. Nano loaded natively as `PooledNormalizedEmbedding`; Zero and Stella loaded as `OnnxTextEmbedding`. Each produced a normalized `(1024,)` `float32` vector. Nano used the frozen local staging directory; the other two used their existing local FastEmbed cache. No `add_custom_model` or user cast was used.
- Complete upstream suite, run with the required interpreter in explicit offline mode: **18 passed, 13 skipped, 66 failed in 4.78 s**. The failures are model-backed tests whose upstream artifacts are absent from `/tmp/fastembed_cache`; the representative and repeated cause is `ValueError: Could not load model ... from any source` with `local_files_only=True`. An initial online run was stopped after repeated DNS timeouts. This environment therefore cannot supply a green whole-suite result; there was no assertion failure in the focused changed-code tests.

## M20 upstream PR plan

Use three PRs. This agrees with the proposed split and keeps model review independent from two behavior fixes.

1. **Padding fixes.** Include `83dbc5f` and `d04ccba`, with their tokenizer-unit regressions and the `thenlper/gte-base` mixed-length reproduction. Evidence: failure on current main, success after the fix, bit-identical mixed-batch versus single-item embeddings, preservation of left-padding metadata, and green CI. Likely pushback: whether always re-enabling padding regresses unusual tokenizer configurations, whether these two fixes belong together, and whether the eager-default case is reachable in supported artifacts.
2. **Dtype fix.** Include `a4452ac` only. Evidence: red/green dtype regression, an explicit list of the 16 affected current models, both A/B measurements above, representative existing-model checks, Nano frozen-reference parity, and green CI. Likely pushback: whether `float64` accumulation was intentional, whether returning the input dtype is the API contract for `float16`/`float64` inputs, and whether changing the exposed dtype could affect callers. Variant B directly addresses the first concern by retaining accumulation precision.
3. **Model registrations.** Include `eea7705` rebased after accepted prerequisites. Evidence: accessible immutable Hub artifacts, licenses/sizes/source paths, canonical vectors for all three, Nano's correct masked-mean-plus-normalization routing, Zero/Stella internal pooling, native serial/parallel smoke tests, Torch/direct-ORT/FastEmbed parity including 511/512/513, and green CI. Likely pushback: asymmetric query/document usage clarity, Stella prompt requirements, research-preview naming/maintenance, namespace capitalization, and why Nano cannot use generic `OnnxTextEmbedding`.

## Card recommendation

After the branch is pushed, all three cards should use exactly:

```bash
pip install "fastembed @ git+https://github.com/Dylancouzon/fastembed.git@constella-research-preview"
```

With variant B landed, the Nano card needs no `float64` warning and no manual-cast instruction. `TextEmbedding("DylanCouzon/constella-nano").embed()` natively yields normalized 1024-dimensional `float32` vectors, matching ordinary FastEmbed use. The only current blocker to the install line working from a fresh machine is the failed DNS-blocked push; the local implementation and behavior are complete.
