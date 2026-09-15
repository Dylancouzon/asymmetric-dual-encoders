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

## Review gates

Two adversarial gates ran, both Codex `gpt-6-astra`, read-only, under briefs carrying the reserved
read-exclusion; each access log was audited and stayed inside its allowlist.

| gate | reviewed | verdict |
|---|---|---|
| A | S1-S3 (`ab8d30d`, `e0fa8dc`, `0e88bae`) | GO, no essential findings; hashes, tokenizer transform and parity independently reproduced |
| B | S4-S5 (`fe06dcf`, `b7d8798`) | **NO-GO**, one P2: the card claimed fp32 while the native path returns float64 |
| B re-review | `d1affb3` (dtype fix plus the owner-requested card restyle) | GO for public publication |

Gate B's finding was fixed in `d1affb3` before any upload occurred; that commit also carried the
owner's card-restyle requirements. A closure audit of the whole branch (`608f488`) raised four P2
record gaps — unlogged publication retries, missing gate verdicts, an unrecorded Git-metadata
recovery, and an overstated evidence description in `m14/FINDINGS.md` — all fixed in this commit.

## Execution provenance

Some stage runs met a read-only `.git` mount in the worker sandbox. Those commits were created
through temporary Git metadata pointed at the same worktree and pushed; the primary checkout's
local metadata was left stale and re-synchronized to the remote afterwards, most recently to
`608f488`. The tracked checkout is clean and the remote branch is authoritative throughout.
Publication attempt-and-recovery history is recorded in `results/m14_publication.json`.

Public state independently re-verified after closure: `private=false`, head revision
`6bb167dc6f60d3992602235b8e8aaa374a309168`, eight files (the seven staged files plus the
Hub-managed `.gitattributes`), card retrievable anonymously.

Reserved access remains **UNSPENT**. M20 inherits the reserved four, BEIR-18, the official release,
and the upstream FastEmbed PR.
