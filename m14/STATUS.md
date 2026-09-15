# M14 status — public research preview

M14 is complete. [`DylanCouzon/constella-nano`](https://huggingface.co/DylanCouzon/constella-nano)
is public as a research preview at immutable revision
[`6bb167dc6f60d3992602235b8e8aaa374a309168`](https://huggingface.co/DylanCouzon/constella-nano/tree/6bb167dc6f60d3992602235b8e8aaa374a309168).

The release gates verified the frozen checkpoint and 21-file backup, the six-file staging
transform, ONNX structure, 10-fixture Torch/direct-ORT/FastEmbed parity including the 511/512/513
boundary, native FastEmbed registration and its upstream test suite, the model card offline, the
private-first upload and LFS metadata, all seven files in a fresh revision download, parity from
that downloaded copy, and anonymous access after the public transition. Downloaded parity
reproduced minimum cosine `1.0`, maximum comparison error `1.1548399925231934e-7`, maximum norm
deviation `1.1920928955078125e-7`, and negative-control cosine `0.1960805058479309`.

Two release defects were found and fixed before publication: Nano's native FastEmbed entry
initially used CLS pooling instead of the required attention-masked mean pooling, and the card
initially claimed fp32 output even though native FastEmbed's integer-mask pooling promotes the
returned vector to float64. The corrected registration uses `PooledNormalizedEmbedding`; the card
now states the native dtype and casts explicitly to fp32 for the intended 4,096-byte query vector.

Reserved access remains **UNSPENT**. M20 inherits the reserved four, BEIR-18, the official release,
and the upstream FastEmbed PR.
