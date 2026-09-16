# M20 implementation review — Astra (gpt-6-astra, xhigh), 2026-09-16

Read-only review of branch `m20-exec` against `research/m20-review-brief-astra-2026-09-16.md`.
Access log audited: every file it opened is on the brief's list, no protected payload, no
recursive search, nothing written. Its report follows verbatim.

---
**NO-GO — M20 implementation review, 2026-09-16.**

The implementation does not yet satisfy the complete reserved-four → BEIR-15 → verified-archive goal. The P1 findings below are blockers. No P0 findings.

This was a static, read-only review. No tests or repository scripts were executed, no protected payloads or caches were opened, and no files were written. HEAD advanced from `c829eba` to `86d9881` during the review; I reread the updated qrels loader and registry. The corrected qrels pinning is not reported as an outstanding defect.

1. **P1 — Blocker: the registered cap exceeds the $1,000 ceiling.**

   The original committed allocation includes 55.2 hours at the target-only rate, $1.6636111111/hour. M20 subtracts those hours from its new allocation but prices the remaining 187 hours at the all-retained rate. That omits sibling-volume storage during the inherited 55.2 hours.

   Using the registration’s figures:

   - Registered additional cost: `187 × $1.8025 = $337.0675`.
   - Omitted storage: `55.2 × $0.1388888889 = $7.6667`.
   - Total commitment: `$657.04 + $337.0675 + $7.6667 = $1,001.7742`.

   This already exceeds the ceiling before stopped time between controllers or any transfer overrun. The launch check trusts `new_spend_at_cap_usd` instead of deriving this total from the live quote and all stage hours. See [budget registration](/home/dylan/asymetric-dual-encoders/m20/beir15_registry.json:577) and [controller ceiling check](/home/dylan/asymetric-dual-encoders/scripts/m13_reserved_cloud.py:298).

   **Required:** correct the allocation and recompute the ceiling check from all retained storage, stage durations, and already committed costs.

2. **P1 — Blocker: individual stage caps and the promised projection safeguard are not enforced.**

   The reserved controller enforces one A+B deadline; the BEIR controller enforces one C+D deadline. Each document-tower process separately receives the entire stage’s cap. Neither controller enforces the registered boundary between its two stages.

   The projection also omits not-yet-pinned public corpora by assigning them zero documents. Loading those corpora never updates the remaining-document total. It resets for every tower and counts encoding seconds rather than total stage elapsed time. Thus the first large BEIR pass cannot provide the registered projection of remaining work. See [projection initialization](/home/dylan/asymetric-dual-encoders/m8src/pre_encode.py:418) and [projection calculation](/home/dylan/asymetric-dual-encoders/m8src/pre_encode.py:357).

   Finally, archive pulling happens after the deadline-controlled polling loop, with `rsync timeout=None`; local verification also has no timeout. The paid pod remains running until those operations return. See [archive transfer](/home/dylan/asymetric-dual-encoders/scripts/m20_beir15_cloud.py:101) and [completion sequence](/home/dylan/asymetric-dual-encoders/scripts/m20_beir15_cloud.py:250).

   **Required:** enforce each stage’s elapsed-time cap through completion, including transfers, and project all remaining towers and corpora using registered volumes until actual counts are available.

3. **P1 — Blocker before protected access: the archive intentionally omits a required R22 deliverable.**

   R22 and `instructions-m20.md` require raw corpora, queries, and qrels for every evaluated dataset at both archive targets. The new registration instead excludes all four reserved query/qrel payloads, and [archive construction](/home/dylan/asymetric-dual-encoders/m20src/archive.py:82) substitutes a repository pointer. It does not even embed the referenced payload hashes.

   This is a substantive reduction of the required artifact, not an implementation detail. The concern about reopening protected data is valid, but does not authorize dropping the deliverable. See [the exclusion](/home/dylan/asymetric-dual-encoders/m20/REGISTRATION.md:166) and [R22](/home/dylan/asymetric-dual-encoders/m13/RULINGS.md:199).

   **Required:** resolve the archive handoff within the authorized transaction before spending access, or obtain an explicit owner amendment narrowing R22. Do not repair this omission afterward by reopening completed reserved payloads.

4. **P1 — Blocker: the C/D controller can report success without delivering the BEIR-15 results.**

   The remote command writes the BEIR table, per-query scores, corpus pins, and pre-encode receipt on the pod. It never commits or pushes them. The local controller copies the archive and its manifest, then fetches Git—where those new results have never been published.

   It nevertheless sets `status="PASSED"`. The BEIR result hash is merely conditional on a local file existing; its absence is accepted. See [remote command](/home/dylan/asymetric-dual-encoders/scripts/m20_beir15_cloud.py:73) and [success handling](/home/dylan/asymetric-dual-encoders/scripts/m20_beir15_cloud.py:250).

   **Required:** transfer and authenticate the complete BEIR result set and reproducibility receipts, verify `COMPLETE`, and make them durable before reporting success.

5. **P1 — Blocker: archive verification can certify an incomplete or changed vector archive.**

   [Vector staging](/home/dylan/asymetric-dual-encoders/m20src/archive.py:106) silently skips a tower/corpus when its manifest is absent. It never requires a complete manifest or enumerates the required shards from it. Existing destination files are accepted solely by equal size. The builder then hashes whatever files happen to be staged into a fresh archive manifest.

   Consequently, a missing vector set is omitted from verification, and a changed same-size staged shard receives a new accepted hash. [Verification](/home/dylan/asymetric-dual-encoders/m20src/archive.py:158) only checks this newly generated file list.

   This is reachable through the supported `--archive-only` path: the controller requires reserved completion but does not require BEIR completion before building and verifying an archive.

   **Required:** require every mandatory Stella corpus/vector set, validate source manifests and every expected shard against their original hashes, and authenticate existing staged files before accepting them. Archive completion must require the complete evaluation inventory.

6. **P1 — Blocker for archive acceptance: object-storage re-hash verification is missing.**

   [Upload](/home/dylan/asymetric-dual-encoders/m20src/archive.py:178) runs `rclone copy --checksum` and reports `UPLOADED`. No subsequent operation reads and hashes destination objects against the archive manifest. Supplying `--upload --verify` still verifies the local `--root`, not the remote destination.

   This does not implement the explicitly registered re-hash at both targets. The acknowledged missing account and credentials are a separate operational dependency; providing them does not close this implementation gap.

   **Required:** implement destination verification against the manifest and persist verification receipts identifying both exact targets.

7. **P1 — Blocker for BEIR reproducibility: resume replaces completed fusion inputs without validating their recorded identities.**

   When a fusion row is unfinished, [the BEIR runner](/home/dylan/asymetric-dual-encoders/m20src/beir15.py:160) re-runs BM25 and the required dense producer even when their completed score files and persisted runs exist. It overwrites their run files but leaves their completed score records unchanged. Fusion then hashes the newly written files instead of checking the hashes recorded by the completed producers.

   Existing public score files are accepted by presence alone, including during assembly. Their registry, query, encoder, and cache identities are not validated. Dense scoring additionally uses `verify=False` for document-shard hashes. See [fusion input loading](/home/dylan/asymetric-dual-encoders/m20src/beir15.py:199), [assembly](/home/dylan/asymetric-dual-encoders/m20src/beir15.py:245), and [shard loading](/home/dylan/asymetric-dual-encoders/m20src/beir15.py:111).

   **Required:** authenticate completed rows, reuse their recorded top-100 runs without re-retrieval, check query-set agreement, and bind scoring to verified document-shard identities.

8. **P2 — Nonblocking recovery debt: a crash during reserved-result publication has no supported completion path.**

   The transaction writes `m13_reserved_run.json`, updates the six-set result, and only afterward commits and pushes. A crash between those steps—or a failed push—leaves persisted results, but the next invocation immediately refuses because `RESULT.exists()`. See [entry refusal](/home/dylan/asymetric-dual-encoders/m13src/reserved_transaction.py:266) and [publication sequence](/home/dylan/asymetric-dual-encoders/m13src/reserved_transaction.py:295).

   Completed system outputs survive, so this does not require another protected read. It does require manual publication recovery.

   **Required:** document or implement recovery that authenticates and publishes only existing completed outputs, without rescoring.

The following parts passed static inspection:

- The reserved system loop validates completed outputs and skips their scoring functions. The aggregate requires all eight systems and matching query sets; no incomplete reserved aggregate path was found.
- Reserved DBSF derivation authenticates producer run hashes and checks payload identities. Both inputs are truncated before DBSF.
- The amendment reversal retains operational fields outside the three permitted edits. No semantic bypass allowing a changed weight, seed, or estimand was found. I did not independently execute the byte-reconstruction test.
- Nano, Zero, and Stella-query share the same document-cache directory.
- The supplied reproduction receipt records zero deltas on SciFact and NFCorpus. This is recorded evidence, not a reproduction performed during this review.
- The default controllers order reserved work before BEIR and BEIR before archive. The `--archive-only` exception lacks the completeness prerequisite described above.

The access-boundary question remains **unverified**. The brief explicitly asks about `m8src/paths_guard.py` but excludes it from the files permitted to open. I therefore did not inspect it. The visible BEIR loader rejects the four reserved corpus names, but the sufficiency of the corpus-only capability and the breadth of its reserved-corpus exemption cannot be certified from callers or comments. **Additional file needed: `m8src/paths_guard.py`.**

The exact file-access log follows. For the two scoped source documents, a heading/definition-name index was also displayed; other function bodies and ruling bodies were not displayed.

```text
research/m20-review-brief-astra-2026-09-16.md
CLAUDE.md
instructions-m20.md
m13/RULINGS.md [R19, R20, R22, R23; heading index]
m13/RESERVED_EXECUTION.md
m20/PLAN.md
m20/STATUS.md
m20/SMOKE.md
m20/REGISTRATION.md
m20/beir15_registry.json
m10/final_run_registry.json [reserved key only]
m20src/roster.py
m20src/beir15.py
m20src/archive.py
m20src/dbsf_reproduction.py
m20src/tower_rate_benchmark.py
m13src/reserved_support.py
m13src/reserved_transaction.py
m13src/test_reserved_support.py
m8src/pre_encode.py
scripts/m13_reserved_cloud.py
scripts/m20_beir15_cloud.py
m13src/score13.py [reserved_batch only; definition-name index]
m12src/qfusion.py
m7src/fusion.py
m11/release/zero_encoder.py
results/m20_dbsf_reproduction.json
results/m20_tower_rate_benchmark.json
```
