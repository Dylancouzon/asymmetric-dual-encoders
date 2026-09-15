# M14 findings

## Native FastEmbed pooling defect

`constella-nano` was initially placed in FastEmbed's generic `OnnxTextEmbedding` registry. That
family treats a three-dimensional ONNX output as token embeddings and selects the CLS row before
normalization. Nano's frozen graph intentionally emits `token_embeddings`; its contract requires
the serving layer to apply attention-masked mean pooling and then L2 normalization.

The wrong generic path produced a maximum error of `0.017370834554605485` over the five-value
canonical prefix. Registering nano in `supported_pooled_normalized_models` instead resolved it to
`PooledNormalizedEmbedding` and reduced the maximum prefix error to
`2.2383970527811714e-08`. The comparison and full vectors are recorded in
`results/m14_fastembed_branch.json`.

The failure pattern isolated the defect: the two already-published entries passed their canonical
checks while nano failed. `constella-zero` and `stella-en-400M-v5-doc-onnx` emit already-pooled,
normalized two-dimensional vectors from their graphs, so the generic family is correct for them.
Nano alone emits per-token vectors and therefore needs FastEmbed's masked-mean-plus-normalization
family.

The accepted fix changes only native registration and its canonical test vector. It preserves the
staged ONNX bytes and the S2/S3 provenance. Rewriting the frozen graph was rejected because doing
so would invalidate those passed gates. The unrelated FastEmbed #703 padding fix is excluded.

## Lesson for release verification

Parity through `TextEmbedding.add_custom_model` does not certify the native registration path.
That bridge explicitly supplies pooling and normalization, so it can produce correct stock-
FastEmbed output even when the eventual built-in model entry is assigned to the wrong family.
Native release verification must exercise the model by its published name, confirm the resolved
FastEmbed family, and compare that output with a reference-derived canonical vector.
The S3 parity cast also hid the native integer-mask pooling's promotion from fp32 to float64, so
native verification must assert returned dtype and per-vector bytes as well as vector values.

The model-card verifier's stale-import guard correctly refused site-packages FastEmbed. The
verified invocation must prepend `work/m14-preview/fastembed` to `PYTHONPATH`; the S5 receipt
records that checkout's absolute path, branch and commit, plus the resolved `fastembed` module
path. The guard remains load-bearing and must not be weakened.
