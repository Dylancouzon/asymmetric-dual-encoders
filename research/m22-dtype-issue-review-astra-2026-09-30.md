# Astra review of the FastEmbed dtype issue and PR (2026-09-30)

Reviewer: Codex `gpt-6-astra`, reasoning `xhigh`, read-only sandbox. Brief: `m22/UPSTREAM_FASTEMBED.md` records its scope (the draft issue, draft PR, diff, measured facts and the upstream checkout with the fix). The output follows verbatim; dispositions are in `m22/UPSTREAM_FASTEMBED.md`.

---

## Findings

1. **P1: The unconditional cast can corrupt integer-output custom models.** In [custom_text_embedding.py:77](/private/tmp/claude-501/-Users-dylanc-Documents-GitHub-asymetric-dual-encoders/8f262e96-dcae-4b81-92c3-b2a1c3a5fc65/scratchpad/fe-fix/fastembed/text/custom_text_embedding.py:77), the cast applies to every pooling mode. Using the actual source methods with synthetic `int8` output:
   - `DISABLED`, normalization enabled: `[3, 4]` previously produced `[0.6, 0.8]`; the patch produces `[0, 0]`.
   - `MEAN`, normalization disabled: a mean of `[1.5, 2.5]` becomes `[1, 2]`.

   **Fix:** Restrict conversion to floating output dtypes. In the custom hook, scope it to `PoolingType.MEAN`. Add regression assertions for these integer cases. This concerns integer graph outputs, rather than quantized weights with floating outputs.

2. **P2: The version claim incorrectly includes custom models.** Commit `1729aab` (#492), first released in v0.6.1, removed the built-in pooled casts. But `git show v0.6.0:fastembed/text/custom_text_embedding.py` already returns the widened mean without a cast. That implementation arrived in #479. Also, the 16-model list describes current main; Constella was added at `a56e1e1`.

   **Fix:** Attribute the v0.6.1 regression to built-in pooled paths. State that custom MEAN models were already affected in v0.6.0, and label the model list with its revision. Custom models whose graph already outputs `float64` do not have this dtype mismatch.

3. **P2: The numerical guarantees exceed the evidence.** “Search results do not change” was not measured. “Only the dtype changes” overlooks rounding. The measurements cover four documents per model for four models, not retrieval rankings or every affected graph. The source registry establishes pooling paths, but does not independently establish every graph’s output dtype.

   **Fix:** Report the measured differences. Remove guarantees about rankings and exact numerical equality. Describe the bug as widening floating model outputs.

4. **P2: The checklist overstates readiness.** `facts.md` explicitly records untriaged full-suite failures. Passing lint through pre-commit does not establish that hooks were installed. The allowed material also does not verify the contributing-guidelines or duplicate-PR checks.

   **Fix:** Leave those boxes unchecked. Keep only the added-tests box checked, and disclose the outstanding suite failures.

**Coverage: No essential findings.** All three callers of `mean_pooling` are covered. Parallel workers return raw output; the parent invokes the same post-processing at `onnx_text_model.py:168`. Query and passage methods delegate to embedding. Token counting only uses tokenization. No other embedding family calls this helper. BM42 has separate float64-producing processing, outside this change.

**Floating-point behavior: No essential findings.** Casting after normalization avoids the demonstrated float16 overflow. Unnormalized paths remain unnormalized. Converting float64 to float16/float32 allocates even with `copy=False`; float64 results avoid that copy. The memory benefit concerns returned arrays, not all temporary allocations or worker traffic. Runtime performance and bfloat16 compatibility remain unverified.

**Existing floating-point tests: No essential findings.** They test observable dtype, masked means, and normalization behavior without deriving expectations from the new cast. An in-memory execution of the source assertions reproduced five failures without the patch and seven passing new cases with it. This was not a full pytest run; the reported `12/12` result comes from `facts.md`.

## Revised issue

TITLE: [Bug]: Mean pooling widens text embeddings to float64

### What happened?

Mean pooling promotes `float32` and `float16` model outputs to `float64`, then returns that dtype to callers.

This affects every model using these paths. Returned arrays use more memory, mixed arrays can be promoted to `float64`, and expected-dtype checks can fail. Raw bytes from `embedding.tobytes()` cannot be read correctly as the model’s original dtype.

For `float32` outputs, storage doubles: 1 million 384-dimensional embeddings occupy about **3.1 GB instead of 1.5 GB**, excluding container overhead. Raw values occupy 8 bytes instead of 4.

The cause is multiplication by the `int64` attention mask in `mean_pooling`. #492 removed the final casts from the built-in pooled paths, exposing this in v0.6.1. Custom `PoolingType.MEAN` models already had the problem in v0.6.0.

At `main` (`a56e1e1`), 16 built-in models use these paths. Custom MEAN models with `float32` or `float16` outputs are also affected.

<details>
<summary>Affected built-in models</summary>

- `sentence-transformers/all-MiniLM-L6-v2`
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
- `intfloat/multilingual-e5-large`
- `nomic-ai/nomic-embed-text-v1`
- `nomic-ai/nomic-embed-text-v1.5`
- `nomic-ai/nomic-embed-text-v1.5-Q`
- `jinaai/jina-embeddings-v2-base-en`
- `jinaai/jina-embeddings-v2-small-en`
- `jinaai/jina-embeddings-v2-base-de`
- `jinaai/jina-embeddings-v2-base-code`
- `jinaai/jina-embeddings-v2-base-zh`
- `jinaai/jina-embeddings-v2-base-es`
- `thenlper/gte-base`
- `thenlper/gte-large`
- `Qdrant/constella-nano`

</details>

### What is the expected behaviour?

Preserve the model’s floating output dtype after pooling and any requested normalization. For the example below, both models should return `float32`.

### A minimal reproducible example

```python
from fastembed import TextEmbedding

for name in [
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-small-en-v1.5",
]:
    embedding = next(iter(TextEmbedding(name).embed(["hello world"])))
    print(name, embedding.dtype)
```

### What Python version are you on? e.g. python --version

Python 3.12, installed with uv (pip). NumPy 2.5.3, onnxruntime 1.30.0.

### FastEmbed version

v0.8.1 and `main` at `a56e1e1`. Compared with PyPI v0.6.0 and v0.6.1.

### What os are you seeing the problem on?

MacOS

### Relevant stack traces and/or logs

```text
Version   all-MiniLM-L6-v2   bge-small-en-v1.5
0.6.0     float32           float32
0.6.1     float64           float32
0.8.1     float64           float32
```

## Revised PR

TITLE: fix: preserve floating output dtype after mean pooling

Fixes #ISSUE.

Mean pooling widens floating model outputs to `float64`. This affects 16 built-in models and custom MEAN models, increasing output storage and breaking consumers that expect the model’s dtype.

This PR casts results back to the ONNX output dtype in `PooledEmbedding`, `PooledNormalizedEmbedding`, and `CustomTextEmbedding`. It casts after normalization where enabled, avoiding float16 norm overflow. Unnormalized models remain unnormalized.

#492 introduced the built-in regression in v0.6.1. Custom MEAN models were already affected in v0.6.0.

**Before merge:** guard the casts against integer model outputs, which the current patch can truncate, and triage the full-suite failures.

<details>
<summary>Validation and measured differences</summary>

With the patch, `all-MiniLM-L6-v2`, `Qdrant/constella-nano`, `nomic-embed-text-v1`, and a custom MEAN model return `float32`.

Maximum absolute differences against v0.8.1, using four documents per model:

| Model | Maximum difference |
|---|---:|
| `all-MiniLM-L6-v2` | 6.8e-09 |
| `paraphrase-multilingual-MiniLM-L12-v2` | 9.5e-08 |
| `nomic-embed-text-v1` | 1.4e-07 |
| `gte-base` | 5.9e-09 |

Documents: `"hello world"`, `"flag embedding"`, `"retrieval " * 300`, and `"a"`. The two unnormalized models retained norms of approximately 5 and 22, respectively. Retrieval rankings were not tested.

Tests cover `float32`, `float16`, and `float64` in all three paths. Five new cases fail on unfixed main; `tests/test_common.py` passes `12/12` with the patch.

The float16 case uses 1,024 values of 10.0. Their squared sum is 102,400, exceeding float16’s maximum of 65,504. Casting before normalization produces an infinite norm and zero embeddings.

Ruff and ruff-format pass through pre-commit. Mypy reports no issues in 64 files. The full suite has untriaged failures and is not confirmed green.

</details>

### All Submissions:

* [ ] Have you followed the guidelines in our Contributing document?
* [ ] Have you checked to ensure there aren't other open [Pull Requests](../../../pulls) for the same update/change?

### New Feature Submissions:

* [ ] Does your submission pass the existing tests?
* [x] Have you added tests for your feature?
* [ ] Have you installed `pre-commit` with `pip3 install pre-commit` and set up hooks with `pre-commit install`?
