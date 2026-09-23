# Sol review — M20 stage C crash recovery (2026-09-22)

Model `gpt-5.6-sol`, reasoning effort xhigh, `--sandbox read-only`. Verdict **GO WITH CONDITIONS**,
**no P0/P1**, run continues.

Brief: `work/m20/logs/sol_crash_recovery_brief.md`.
Full transcript including every command run: `work/m20/logs/sol_crash_recovery_review.log`.

Access audit: 17 bounded exec commands. Every `rg` was scoped to explicitly allowlisted files
(`m20/*.md`, `m20/beir15_registry.json`, `CLAUDE.md`, one named manifest); no recursive search over
`results/` or `work/`, no read of `results/frozen_eval/untouched-*`, reserved qrels caches or
`work/m9reserve`.

Both findings were accepted and fixed: P2 in `m20/STATUS.md` and `CLAUDE.md`, P3 in
`results/m20_crash_shard_verification.json`.

Independent corroborations worth keeping: Sol re-derived the repaired shard's sha256 and got
`9c46302e…`, matching the pre-crash record; confirmed all three stage-C receipts carry deadline
`1790291204.4492238`; recomputed the scoring reserve from the registry as
`113 − 95.942760 = 17.057240 h`; and ran full strict git object verification, which passed.

---

The recovery is technically sound, and the running job should continue. I found no P0/P1 issue.

### Findings

- **P2 — operational documentation is stale and contradictory.** [m20/STATUS.md](/home/dylan/asymetric-dual-encoders/m20/STATUS.md:1) still calls this attempt 2, says BGE is encoding, and instructs launching stage C even though the live run is now attempt 3 on Leaf. More importantly, the later command uses truncating `>` at [line 113](/home/dylan/asymetric-dual-encoders/m20/STATUS.md:113), contradicting the explicit `>>` safety rule at line 17. The present launch used `>>`, so current evidence is intact, but this must be corrected before another handoff or relaunch. [CLAUDE.md](/home/dylan/asymetric-dual-encoders/CLAUDE.md:26) is stale too.

- **P3 — the forensic diagnosis overstates the evidence.** [m20_crash_shard_verification.json](/home/dylan/asymetric-dual-encoders/results/m20_crash_shard_verification.json) proves crash-associated loss of the shard’s persisted bytes. The zero-length file plus general ext4 orphan recovery does not specifically prove that “ext4 ordered data mode” caused this pathname’s damage. Describe the mechanism as undetermined/likely crash-consistency loss. Also, spent evaluation access makes the queries/qrels unavailable, not the public document corpora technically “unrepeatable”; “closed and should be preserved” is more accurate.

### Answers

1. **Deleting the shard was correct.** In [encode_dataset](/home/dylan/asymetric-dual-encoders/m8src/pre_encode.py:305), skipping depends on `path.exists()`, not whether `manifest["shards"][sid]` exists. A deleted-but-recorded shard cannot be silently skipped. It is regenerated and the old entry replaced at lines 315–329.

   Leaving the stale entry caused no ordering, `n_shards`, status, resume, or summary problem. Removing it was unnecessary and would have discarded useful evidence. While running, `n_shards` properly remains absent and the summary shows `null`; completion recalculates it as 109.

   The code does not enforce equality with the old hash when regenerating. I therefore checked the completed repair directly: the new 76,800,128-byte file hashes to exactly:

   `9c46302ee7f5c3fe655c1a039e468ac4fb0dd32c8b48e421e477dc7f01519f0a`

   That is the original recorded hash. The repair is byte-identical and the potential mixed-index concern is discharged.

2. **No consequential crash damage class appears missed.**

   - Partial manifests fail JSON parsing; manifest writes are atomic and fsynced. Missing manifests plus existing shards fail closed as “unrecorded shard.”
   - Identity fields are checked against the current registered corpus/model contract.
   - `doc_ids.json` is rehashed against freshly derived pinned corpus IDs before any shard is skipped.
   - Corpus IDs/text are recomputed against `m20_corpus_pins.json` on every dataset visit.
   - `m20_beir15_preencode.json` is derived progress state and is regenerated atomically.
   - The preflight established a clean pushed HEAD. I additionally ran full strict Git object verification; it passed.

   A more forensic receipt could also have recorded every manifest/doc-ID hash and validated each `.npy` header, dtype, shape, and expected byte count. Those would improve audit completeness, but their absence does not invalidate this recovery. Your stated limitation is correct: matching the recorded hash cannot independently detect a semantic error already present when the original file was written and hashed. Full semantic proof would require independent re-encoding; the repaired shard’s exact reproduction now provides a strong Leaf pipeline check.

3. **The original deadline is retained.** Attempt 1, preserved attempt 2, and the live receipt all contain `1790291204.4492238`. The launcher takes the minimum deadline from `m20_stage_c_run_attempt*.json`; the live receipt says `stage_c_deadline_inherited: true`. `_remaining_hours` refuses once it reaches zero. This matches the earlier stage-A receipt design and Sol’s P1 fix.

   This relies on preserving receipts under the matching filename pattern, but the present files satisfy that requirement.

4. **The budget arithmetic is right.** The registry recomputes:

   - Registered encode expectation: `95.942760 h`
   - Scoring reserve: `113 − 95.942760 = 17.057240 h`

   The 31.99 h/49.05 h figures describe invocation before the roughly 25-minute preflight; the first child received 31.586 h and Leaf later received 31.089 h, which is consistent.

   Climate’s recorded rate is 410.7 docs/s. Repair-inclusive remaining climate work projects to about 2.65 h; the ten remaining CQADup forums add roughly 0.54 h using the 0.3308 cost ratio. Approximately 3.2 h plus overhead makes 3.3 h credible. The outage consumed encode headroom, not the scoring reserve: encoding is still bounded by `deadline − 17.05724 h`.

5. **The verification receipt’s core claims are supported.** Counts, damaged path, old/new sizes and hashes, no-number impact, and first-write limitation are consistent. Only the filesystem-mechanism attribution and “unrepeatable documents” wording should be qualified. The now-reproduced hash should also be added to durable recovery evidence at the next safe documentation commit.

6. **Relaunching with the reserved shard directories present was safe.** `beir15_encode_datasets()` explicitly excludes the four reserved names at [pre_encode.py:471](/home/dylan/asymetric-dual-encoders/m8src/pre_encode.py:471). The current `--batch beir15 --system leaf-ir-asym` process reads public corpora and document artifacts only; directory presence does not trigger protected payload access. The registration requires the later scorer to copy reserved results rather than reopen or rescore them.

7. **No other result-bearing or scope deviation was found.** The run remains descriptive, exact-retrieval, eight-system BEIR-15 with `alpha = 0`; nothing in this recovery creates a gate, threshold, superiority, equivalence, or release claim.

Conditions: let the current run continue; before any future handoff/relaunch, correct the stale runbook and truncating command, and durably record the reproduced shard hash plus the qualified crash diagnosis.

**GO WITH CONDITIONS**
