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

## README fix found during this check

Since `2704c34` (2026-09-25) the preview branch registers only the `Qdrant/` names, and it
rejects `TextEmbedding("DylanCouzon/constella-nano")` with `ValueError`. The README quickstart
still used the `DylanCouzon/` names, so it failed on a fresh install. Commit `c907c22` renames
the model IDs and links in `README.md` and `research/constella-in-plain-english.md`. The
quickstart then ran end to end on both the preview branch and upstream `a56e1e1`: Nano 0.7418
against 0.1331 and Zero 0.6423 against 0.0470, with the mRNA passage first. The install line is
unchanged.

## Owner decisions (Dylan, 2026-09-30)

- **Install lines wait for PyPI.** The Hub cards, the README and the plain-English page keep the
  preview-branch install line until a PyPI release contains #751. They then switch once to plain
  `pip install fastembed`.
- **Hub cards wait for PyPI.** No card commit now. The next card revision carries the install
  line and removes the "requires the Constella preview branch" banner sentence.
- **Dtype PR: go.** After the explanation, Dylan asked to check for an existing ticket, prepare
  the GitHub issue with Codex Astra as adversarial reviewer and co-writer, and open the PR once
  fully sure. This moves M23's dtype deliverable ahead of M22.

## Dtype issue preparation (2026-09-30)

- **No existing ticket.** Searches of qdrant/fastembed issues and PRs for float64, dtype,
  float32, mean pooling, astype, precision and pooled found none on this bug. #308 is a Faiss
  shape error.
- **It is a regression from upstream #492** ("preserve embeddings in a type set by their model",
  `1729aab`, first released in v0.6.1). #492 removed a final `.astype(np.float32)` so outputs keep
  the model's dtype; the float64 from `mean_pooling` then reached users. Measured on PyPI builds:
  `all-MiniLM-L6-v2` returns float32 on 0.6.0 and float64 on 0.6.1 and 0.8.1; `bge-small-en-v1.5`
  (CLS pooling) returns float32 on all three.
- **The M21 fix missed a third caller.** `CustomTextEmbedding` with `PoolingType.MEAN` also calls
  `mean_pooling` and still returned float64 with the preview-branch fix. The PR patch now casts
  there too, with a test that fails on float32 and float16 without it.
- **Values change by float32 rounding only.** PyPI 0.8.1 against the fix, four inputs each:
  `all-MiniLM-L6-v2` 6.8e-09, `paraphrase-multilingual-MiniLM-L12-v2` 9.5e-08 (unnormalized,
  norms about 5), `nomic-embed-text-v1` 1.4e-07 (unnormalized, norms about 22), `gte-base`
  5.9e-09. Norms unchanged.
- ruff and ruff-format pass; mypy with the CI flags reports no issues in 64 files.

## Astra review of the issue and PR (2026-09-30)

Codex `gpt-6-astra` (xhigh, read-only) reviewed the draft issue, draft PR, diff and measured facts
and co-wrote the text. Record: `research/m22-dtype-issue-review-astra-2026-09-30.md`. It found
all three `mean_pooling` callers covered, parallel workers and query/passage paths safe, and the
tests meaningful.

| finding | disposition |
|---|---|
| P1: the custom-path cast truncates integer graph outputs (`int8` `[3, 4]` normalized became `[0, 0]`) | **Fixed.** The cast applies only to float outputs; new test `test_custom_integer_output_is_not_truncated` |
| P2: custom MEAN models returned float64 before v0.6.1 | **Confirmed and fixed in text.** In `v0.6.0` the custom path already returned the float64 mean without a cast; it dates from #479 |
| P2: "search results do not change" was not measured | **Fixed.** The text now reports the measured differences (at most 1.4e-07) and says values match float32 up to rounding |
| P2: checklist boxes overstated | **Fixed.** Pre-commit hooks installed and run on the commit; the existing-tests box follows the full-suite result |

Astra's rewritten text was more formal than the brief asked for; the final issue keeps our plainer
structure with its corrections. The patch is `m22/fastembed-dtype-fix-on-a56e1e1.patch`
(scratch commit `bf96abc` on `a56e1e1`); with it `tests/test_common.py` passes 13 of 13, and five
of the new tests fail on `main` without it.

## Issue opened (2026-09-30)

Dylan narrowed the deliverable to one GitHub issue with a proposed fix, and no PR for now. The
full-suite rerun was stopped: it only gates a PR. It had not finished, because of a stalled
Hugging Face download and then a short local HTTPS outage. Its ten failures (ColPali multimodal,
cross-encoder rerank, jina-v3 multitask) are in paths that do not call `mean_pooling`. They were
not compared against clean `main`.

Opened [qdrant/fastembed#752](https://github.com/qdrant/fastembed/issues/752), "[Bug]: Mean-pooling
text models return float64 instead of float32", on the bug-report template. It carries the
consequences, the 16 affected built-in models plus custom `PoolingType.MEAN` models, the #492 and
#479 cause, the v0.6.0/0.6.1/0.8.1 repro and the proposed three-path diff. The tested patch with
tests remains `m22/fastembed-dtype-fix-on-a56e1e1.patch` for a PR if the maintainers want one.
