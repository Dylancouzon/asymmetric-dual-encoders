# M19 ledger

## 2026-09-13 — execution boundary opened

- Owner instruction: execute `instructions-m19.md` on the existing M18 branch and keep a frequent
  commit paper trail.
- Worktree correction: the supplied shell directory resolved to `main`; execution moved to the
  already registered `/home/dylan/asymetric-dual-encoders-m18` worktree on
  `m18-qdrant-project-memory`. No file on `main` was changed.
- Fresh-context condition satisfied. The session read `CLAUDE.md`, `instructions-m19.md`,
  `ROADMAP.md`, `m18/STATUS.md`, `m18/CODEMAP.md` and `m18/FINDINGS.md` before execution.
- Admitted inheritance reads are limited to the explicit M18 source/corpus/index/system manifests,
  the redacted corpus/index bytes and released Zero-v1 bundle. No M18 confirmation query or qrel
  payload was opened. No M7–M13 spent/reserved surface was opened.
- Stage: bootstrap only. No query authoring, retrieval-quality read, candidate construction or
  judgment has begun.
- Generated inheritance lock identity
  `bd0c5dc8e6904cf123cd7a78713a688f5c57d988b5545c07f302ad55daa93052`. It binds each BM25
  component, ordered document IDs, the document vector file, both corpus copies, six tracked M18
  control manifests and three released-v1 bundle files. Verification: `8 passed` for
  `m19src/test_common.py m19src/test_inherit.py`.

## 2026-09-13 — bootstrap review correction

- Astra returned `NO-GO`: two P1 immutable-lock defects and four P2 guard/registry/test gaps.
  Full bounded brief, access log and dispositions are in `m19/reviews/bootstrap-astra-2026-09-13.md`.
- Replaced the v1 lock with v2 while retaining v1 in commit history. V2 binds the prospective
  registry, governing instruction, global guidance, guard/lock implementations, expected hashes
  for every inherited data file, canonical ordered document IDs, indexed-corpus identity and the
  released effective int8 rows/pooling parameters.
- Lock publication is now exclusive: a repeated identical invocation is idempotent; a differing
  existing lock is refused. Fresh M19 confirmation reads require a claimed-state exact-path/hash
  boundary.
- Corrected lock identity:
  `6eff27e371dda116f6d0613d9e269aaf8afb809d2edc8842299fc2bd3fb34612`.
  Verification: `23 passed`; real `--verify` passed; a second default invocation left the lock
  byte-identical.

## 2026-09-13 — bootstrap re-review GO and residual closure

- Astra re-review of `655ee6c` returned `GO` for bounded term inventory; no P0/P1 remained.
- Closed its three residual P2s before advancing: atomic no-clobber initial publication, permanent
  exclusions/root validation in the claimed confirmation reader, and production-dependency plus
  race/semantic regression tests.
- Final bootstrap lock identity:
  `de6d89676edd6b33e97aad08c1b28dce638c6c59c00f2e39a16f827ca84791bb`.
  Verification: `20 passed`; real `--verify` passed. No quality surface was accessed.

## 2026-09-13 — prospective term roster frozen

- Scanned only the 79,269 indexable records in the pinned redacted corpus and the released-v1
  tokenizer. The fixed catalog/order preceded the support scan; no retrieval or qrel data entered
  selection.
- Selected the first 12 qualifying stable-meaning terms: `k8s`, `s3`, `hnsw`, `grpc`, `rocksdb`,
  `mmap`, `arm64`, `tls`, `cuda`, `simd`, `turboquant`, `gridstore`.
- The minimum support is 13 artifacts (`cuda`), above the registered five; original tokenization is
  2–3 pieces for every term. The joint `single_word=True, normalized=True` AddedToken audit assigns
  stable IDs 30522–30533 without changing inherited IDs.
- `qdrant`, `quantization` and `vectorstore` also passed the mechanical gates but were excluded by
  the predeclared 12-term cap, not a quality result.
- Lock: `m19/term-roster-lock.json`, identity
  `3ac4b8190cfd3827857346ca675dc578139fc518b0b4ccd2cea7cbf3fdc482f9`.
  Evidence: `results/m19_term_inventory.json`. Verification: three inventory tests plus immutable
  output recomputation passed.

## 2026-09-13 — roster review NO-GO and v2 correction

- Fresh Astra review found one P1 scan-time binding race, four P2 boundary/qualification/resume/
  provenance issues and one P3 status contradiction. See
  `m19/reviews/roster-astra-2026-09-13.md`.
- Preserved the v1 roster/result and superseded them with versioned v2 artifacts. The v1 claim that
  the catalog was prospectively git-frozen is narrowed: catalog and first result were committed
  together, runtime order was fixed, and no retrieval quality informed it, but no independent
  pre-scan receipt exists.
- V2 matches terms with the actual joint AddedToken tokenizer, excludes explicit non-indexable rows
  and text-equivalent cross-artifact support, and hashes the exact tokenizer/corpus bytes it
  consumes. Corrected distinct-artifact support is 13–1,002; all terms still clear the gate.
- Source-only qualification inspected an ignored bounded packet, not rankings or evaluation
  queries. Each term now binds five non-equivalent intended-sense witnesses and evidence for all
  three intent axes. `s3` excludes identifier uses, `tls` excludes thread-local storage, and CUDA's
  distinction from Qdrant's Vulkan path is explicit.
- V2 roster lock identity:
  `864c4073674a5c207d680092a84e713178ad98b69cc86262409d7bfee48b5c08`.
  Seven inventory tests pass. Re-review is required before stage 3.

## 2026-09-13 — roster re-review GO and v3 selected-token audit

- Astra re-review returned `GO` for stage-3 implementation and verified all prior dispositions.
- One residual P2 showed v2's final roster copied the 15-term catalog matcher audit. The selected
  terms' IDs were unchanged, but the declared added/final vocabulary counts were wrong for bundle
  construction.
- Preserved v2 and issued v3. The inventory retains its 15-term diagnostic matcher; the roster has
  a separate selected audit with exactly terms `k8s` through `gridstore`, IDs 30522–30533,
  `added=12` and `final_vocab=30534`.
- V3 lock identity:
  `8170f9b0432d4b7460601e1dbf39536d99b1e4af026d9a3afef649a9c89ef892`.
  Eight inventory tests pass, including driver-level consumed-input drift. Stage 3 may begin; real
  judging remains unauthorized until the required end-to-end implementation reviews.

## 2026-09-13 — deterministic row/compact-loader implementation

- Implemented the registered formula directly over released-v1 effective int8 rows. V0 uses the
  original fragment contribution; T0 sets `r_t = alpha*u_t - a_t` with alpha equal to the old bare
  sum norm. No optimizer/checkpoint/sweep path exists.
- The compact table appends one int8 code row plus one float32 scale while copying every inherited
  code and scale exactly. The loader dequantizes only query-referenced rows and has no full-table
  float32 member.
- Bundle bytes are deterministic and publish resumably with `complete.json` last; incomplete
  bundles are unreadable and differing existing bytes are refused.
- Six synthetic row/bundle tests pass. Real teacher rows and bundles are stage 5 work, after the
  complete stage-3 implementation review.

## 2026-09-13 — artifact retrieval and metric primitives

- Implemented source-artifact/text-copy exclusion before the fixed 500-passage truncation,
  deterministic best-passage collapse to 100 artifacts and unchanged sample-standard-deviation
  DBSF on artifact scores. Fused rows retain each dense/lexical supporting passage and contribution.
- Implemented binary P@10 with denominator ten, pooled-pool IDCG/Recall, MRR, success, within-term
  then equal-term aggregation, complete paired query/term differences, headroom and development/
  confirmation eligibility gates. Empty registered safety slices are refused.
- Nine synthetic tests pass. No inherited index, query, qrel or retrieval-quality output was read.

## 2026-09-13 — prospective query split sealed

- Made split validation part of the sealing operation, added deterministic lexical near-duplicate
  detection and source-exclusion binding, and made interrupted no-clobber publication resumable by
  exact digest. The manifest is the completion marker and publishes last.
- Authored 60 development queries (12 bare, 36 short, 12 longer) and 30 confirmation queries
  (24 short, 6 longer) solely from the frozen roster and pinned source evidence, before retrieval.
- Fresh Astra pre-seal review rejected the first two drafts for semantic family leakage and
  unsupported numeric/version details. The final exact draft
  `288d2089505aef08ba556b19ef162415c06bb417461077456a12a2a228198b66` received `GO` with no P0–P2.
- Development and confirmation JSONL hashes are respectively
  `8317a85b9c5969be44330a5bd181e9df357d3657c9b3904ba8462e6b7d05f72f` and
  `bc3d57abe0d3c77ec2c2993153feaa1d3fe7fada71b2f68010e6a6b40d6ac3a8`. Seal-manifest hash:
  `09f4f6439d0519eeb9037af472ab8151c7e58fd70ebc93f415df30e3222d736a`.
- Removed the temporary readable draft after sealing. Confirmation content is now behind the claim
  guard. No retrieval, quality metric or relevance judgment was run or read.

## 2026-09-13 — actual V0/T0 bundles built

- Encoded the 12 raw roster terms once with the pinned local Stella-400M revision, registered query
  prefix and released no-xformers configuration. The teacher array is normalized float32 with
  identity `7336d3097663bae26f025cd56624f3de5e0a33209b8b6a6acbe648a0c7d8d2b2`.
- Published deterministic compact V0-compose and T0-teacher bundles with completion markers. Their
  identities are `056b78d1184b0be8032a5493c75d033d109787e979730a5c6637e25c3785f968` and
  `b25a5fcb019f19c3bd48ba82f6fa05b9f7a64b74ff19a8cc1d83884fe8f6ae0d`.
- T0 minimum int8 cosine is `0.9999541` and maximum coordinate error is `0.0005364`; corresponding
  gates are `0.999` and `0.02`. V0 also passes. Both report exactly 31,388,952 resident table bytes,
  no eager float32 table, unchanged inherited rows and exact no-match encoder parity.
- A second builder execution reproduced and verified every immutable output byte. Latency gates
  remain open; no retrieval quality or confirmation content was accessed.

## 2026-09-13 — official serving gates pass

- A 100-query development-only pilot exercised the inherited BM25 index, exact GPU dense scoring,
  artifact collapse and unchanged DBSF path without storing rankings or quality output. It exposed
  one benign read-only memmap warning; the loader now uses copy-on-write mapping so the source bytes
  remain immutable while PyTorch receives a writable view.
- The fixed alternating V1/T0 sequence then completed 10,000 measured development queries after 20
  warmups on the RTX 3080 in 96.78 seconds. T0/V1 encoder median and p95 ratios are `0.9590` and
  `0.9763`; end-to-end ratios are `0.9944` and `0.9969`.
- Loader parity is exact, no-match ranking parity passes, all algebra/tokenizer/pooling/memory and
  latency checks are true, and the T0 gate receipt binds bundle
  `b25a5fcb019f19c3bd48ba82f6fa05b9f7a64b74ff19a8cc1d83884fe8f6ae0d`.
- No confirmation query, relevance judgment or retrieval-quality metric was accessed or produced.

## 2026-09-13 — serving v1 superseded after Astra review

- Fresh Astra independently reconstructed both bundles and regenerated all 12 Stella teacher
  vectors byte-for-byte, but returned `NO-GO` for stage-5 sign-off. The actual candidate was sound;
  the blockers were fail-open added-row provenance, incompletely bound benchmark inputs and a timed
  top-k shortcut that did not implement the registered float32/exclusion/tie semantics.
- `zero.verify_bundle` now checks the exact file set, expected build identity, variant/config links
  and current model/tokenizer digests. A negative test rehashes an altered added row and is refused.
- Serving v2 verifies inheritance, the query seal, candidate build and actual bundle before timing;
  binds every registry/roster/query/build/bundle/index input; uses the released V1 loader; and selects
  exact float32 passage top-500 results with exclusion before truncation and passage-ID tie breaks.
  Confirmation recomputes that fixed workload and rejects stale sequence/host/warmup/input claims.
- The superseding 10,000-query run completed in 274.14 seconds. Released-loader parity is
  `2.24e-8`; T0/V1 encoder median/p95 ratios are `0.9580`/`1.0370` and end-to-end ratios are
  `0.9956`/`0.9696`. All seven checks pass. The original v1 receipt remains preserved as superseded.
- The full M19 suite has 71 passing tests. No rankings, judgments, quality metrics or confirmation
  content were opened or stored.

## 2026-09-13 — stage-5 Astra closure GO

- Astra accepted all three P1 dispositions at exact commit `9dcddb8` after 17 targeted tests, 70
  independent full-sort tie/exclusion comparisons and exact receipt/bundle reconciliation.
- No P0/P1 remains in the closure scope. The three documented P2 qualifications do not block the
  fixed ten-query pool-cost pilot. This GO does not authorize real judging or confirmation.
