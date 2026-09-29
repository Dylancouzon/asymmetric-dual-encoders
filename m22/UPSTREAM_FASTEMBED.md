# Upstream FastEmbed #751: Constella registrations merged (2026-09-29)

George Panchuk (FastEmbed maintainer) opened and merged
[qdrant/fastembed#751](https://github.com/qdrant/fastembed/pull/751) on 2026-09-29 as upstream
commit `a56e1e13508980441cf4f2072951c92c224e7fbe`, co-authored with Dylan. CI passed 17 of 17
checks. The PR reports NFCorpus nDCG@10 reruns: Nano 0.363080 (exact), Zero 0.312416 and Stella
with `s2p_query` 0.413415 (both match the cards at their precision). No PyPI release contains it
yet: the latest release is 0.8.1 (2026-09-22), which precedes the merge.

This is M23's third deliverable (registrations), done upstream before M22. This record checks it
on 2026-09-30. No model bytes, Hub repos or registered numbers changed.

## What #751 contains

The three registrations and canonical-vector prefixes from our preview branch (`eea7705` renamed
by `2704c34`), byte for byte: same classes, descriptions, licences, sizes, sources and five-value
vectors. Nano is a `PooledNormalizedEmbedding`; Zero and the Stella document tower are
`OnnxTextEmbedding` because they pool inside their graphs.

It carries no bug fix. The preview branch's two padding fixes are covered by upstream's own
#716 (padding always normalized to batch-longest, `length=None`) and #717 (keeps
`pad_to_multiple_of`); upstream now reads `pad_token` lazily, which covers the eager-default fix.
The dtype fix is **not** upstream.

## Verification

`m22/verify_upstream_fastembed.py`: the published Hub bytes, the ten M14 length fixtures (served
lengths 31 to 512, including the 511/512/513 boundary), batch size 2 against batch size 1, a direct
ONNX Runtime reference and the M14 thresholds (cosine ≥ 0.9999, absolute error ≤ 1e-5). The Torch
leg was not rerun: its checkpoint is not on this Mac, and M14 measured direct ORT within 1.2e-07
of Torch. Environment: macOS, Python 3.12, NumPy 2.5.3, ONNX Runtime 1.30.0, four threads.

| install | Nano dtype | Zero / Stella dtype | worst max abs error vs ORT | parity |
|---|---|---|---:|---|
| upstream `a56e1e1` | **float64** | float32 | 1.62e-07 | PASSED |
| upstream `a56e1e1` + dtype patch | float32 | float32 | 1.64e-07 | PASSED |
| preview branch `2704c34` | float32 | float32 | 1.64e-07 | PASSED |

- Padding is dynamic (`length: None`, right, `[PAD]`) and truncation is 512 for all three.
  Batch-size-2 against batch-size-1 rows differ by at most 1.3e-07.
- The canonical-vector prefix for each model is within 1.6e-07 of its current output (test
  tolerance 1e-3).
- Upstream `a56e1e1` and the preview branch agree to 5.2e-09 on Nano and exactly on Zero and
  Stella. Patched upstream matches the preview branch exactly on all three.
- The live Nano and Zero card examples run on upstream `a56e1e1` and rank the mRNA passage first
  (Nano 0.742 against 0.133, Zero 0.642 against 0.047). The document-tower example returns two
  normalized `(1024,)` float32 vectors.

Receipts: `results/m22_upstream_fastembed_a56e1e1.json`,
`results/m22_upstream_fastembed_a56e1e1_dtype_patched.json`,
`results/m22_upstream_fastembed_preview_2704c34.json`.

## The remaining defect: Nano returns float64 upstream

Upstream `mean_pooling` still promotes `float32 * int64` to float64 (the `m21/FASTEMBED.md` root
cause). Rankings and values are correct; the vector is 8,192 bytes instead of 4,096, and the Nano
card's "normalized fp32 vector" is false for upstream users. It affects every model routed through
`mean_pooling`, not only Nano.

`m22/fastembed-dtype-fix-on-a56e1e1.patch` is the preview branch's `a4452ac` + `47a5090` net
change rebased onto `a56e1e1`: `mean_pooling` stays byte-identical to upstream, narrowing happens
after `normalize()`. The only conflict was one import line in `tests/test_common.py`. Red/green on
upstream: the new tests fail three of nine without the patch and pass nine of nine with it.
`test_common.py` and `test_preprocessor_utils.py` pass 40 of 40 with it.
