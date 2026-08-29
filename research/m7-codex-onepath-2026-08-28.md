# Codex targeted review: the one-shot result path (2026-08-28, gpt-5.6-sol, read-only)

Narrow deep audit of `run_freeze_prep.sh` -> `freeze.write` -> `final_run.py` and the statistics
they call, briefed adversarially: I stated what I believed and asked it to break it. Findings
verbatim below. Dispositions are in `m7/LEDGER.md`; the three BLOCKERs and both MINORs are fixed
(commits 4413763, 6eee401, ed2f5b8, 749a8d9).

## BLOCKER

1. The hand-written freeze can bind the wrong encoder while remaining internally “verified.”

[freeze.py:88](/home/dylan/asymetric-dual-encoders/m7src/freeze.py:88) reads the table metadata, but [freeze.py:113](/home/dylan/asymetric-dual-encoders/m7src/freeze.py:113) obtains `encoder_spec` independently from `M7_ENCODER`. Neither `write` nor [load_and_verify:168](/home/dylan/asymetric-dual-encoders/m7src/freeze.py:168) checks that `meta["teacher"]`/`teacher_revision` agree with that spec.

This is especially dangerous here: `run_freeze_prep.sh` exports Stella only inside its child shell. The current parent environment has no `M7_ENCODER`, so a subsequent naïve `freeze.write(...)` records default BGE while freezing the Stella table. A final run under the same default BGE then passes `encoder_drift` and scores Stella rows against BGE documents—a plausible but wrong number.

Concrete fix: in `freeze.write`, resolve and compare the artifact’s `teacher`, revision, dimension, vocabulary and tokenizer identity against the active full encoder spec, refusing any mismatch. Repeat that cross-check in `load_and_verify`. Prefer deriving the frozen spec from the artifact identity rather than the ambient environment.

2. The current final run is guaranteed to fail after consuming the six.

[final_run.py:43](/home/dylan/asymetric-dual-encoders/m7src/final_run.py:43) includes `cqadup-android` and `cqadup-english`. [final_run.py:113](/home/dylan/asymetric-dual-encoders/m7src/final_run.py:113) requires corresponding manifest entries, but [eval_manifest.json:161](/home/dylan/asymetric-dual-encoders/results/eval_manifest.json:161) contains only FEVER and DBpedia under `m7_untouched_final`; both CQADup frozen files are also absent.

Because untouched evaluation starts only at [final_run.py:262](/home/dylan/asymetric-dual-encoders/m7src/final_run.py:262), it will score all six, compute and print the confirmatory decisions, score FEVER and DBpedia, then raise `KeyError` at Android. No complete result is written.

Concrete fix: generate and commit both entries/files before freezing. Add a complete static preflight—required manifest keys, frozen files, hashes and payload structure—for all ten datasets before `FINAL-RUN-BEGIN` or any six access.

3. The “freeze commit” and retry controls do not enforce one shot.

[final_run.py:78](/home/dylan/asymetric-dual-encoders/m7src/final_run.py:78) merely checks HEAD against the caller-provided `--freeze-hash`; [final_run.py:81](/home/dylan/asymetric-dual-encoders/m7src/final_run.py:81) checks that the same caller-selected hash is at a caller-selected branch. Nothing independently identifies it as the reviewed freeze commit. Any clean pushed HEAD can be declared the freeze commit.

Additionally, [final_run.py:89](/home/dylan/asymetric-dual-encoders/m7src/final_run.py:89) allows `--infra-retry` after any prior BEGIN, including a successfully completed run, does not cap retries, and waives the local HEAD equality check. Thus completed results can be rerun indefinitely.

Concrete fix: anchor the freeze commit with a dedicated pushed immutable tag/ref and resolve it internally; do not accept its identity solely from argv. On retry require `HEAD == remote tag == prior BEGIN hash`, the last attempt has no COMPLETE, and no retry has already been consumed. Refuse retry after COMPLETE.

## MAJOR

1. Fusion is not bound to the artifact or gate it was selected on.

[select_fusion.py:69](/home/dylan/asymetric-dual-encoders/m7src/select_fusion.py:69) writes no run ID, table hash, metadata hash or preprocessing fingerprint into the fusion spec. [freeze.py:81](/home/dylan/asymetric-dual-encoders/m7src/freeze.py:81) accepts an arbitrary caller-supplied spec and released-system choice, without consulting the selected file or gate result. Therefore a spec selected on artifact A can be frozen with artifact B, or the release can change between gate and freeze, without detection.

`released_system` is also not validated as an enum: [final_run.py:198](/home/dylan/asymetric-dual-encoders/m7src/final_run.py:198) silently treats every value other than exactly `"fusion"` as dense. Likewise, [fusion.py:128](/home/dylan/asymetric-dual-encoders/m7src/fusion.py:128) silently treats every unknown family as ordinary convex fusion.

Concrete fix: have `select_fusion` record artifact/meta hashes, run ID, preproc fingerprint, depth and dev-manifest hash. Have `freeze.write(run_id)` load that exact file itself, require the gate JSON to be `PASS` for the same artifact hashes, validate the full schema, and derive `released_system` mechanically. Make unknown family/system values fatal.

The `w=1.0` endpoint itself is reachable and applies the same dense ranking. It is recorded as `{"family":"convex","param":1.0}`, however, rather than automatically becoming the dense released-system choice.

2. The authoritative fusion selection can consume an unrelated BM25 cache.

[fusion.py:65](/home/dylan/asymetric-dual-encoders/m7src/fusion.py:65) trusts an existing cache solely by pathname. The cached files contain only integer positions and scores—no corpus/query hashes, BM25 configuration or version. `select_fusion` maps those positions onto the current corpus, while [final_run.py:154](/home/dylan/asymetric-dual-encoders/m7src/final_run.py:154) rebuilds BM25 fresh.

A reordered or changed corpus/query set with compatible shapes therefore selects a parameter on one lexical run and applies it to another, plausibly.

Concrete fix: key and validate caches using ordered document IDs/text, query IDs/text, depth, BM25 configuration and implementation version, or rebuild them for the authoritative selection.

3. Frozen query/qrels bytes are not verified by the final scorer.

[final_run.py:125](/home/dylan/asymetric-dual-encoders/m7src/final_run.py:125) verifies only corpus fields. It then reads query texts and qrels at [final_run.py:132](/home/dylan/asymetric-dual-encoders/m7src/final_run.py:132), despite the manifest already carrying qid/qrels hashes. Query-text hashes are not recorded at all. `freeze.load_and_verify` hashes only the manifest, not the payload files.

The six and two existing untouched payloads currently match their qid/qrels manifest entries, but enforcement is absent. Changed query text with unchanged qids, or changed qrels, can produce a plausible number and pass strict comparator alignment.

Concrete fix: record a SHA-256 for every complete frozen payload—or ordered qids, texts and qrels separately—and verify it before scoring.

4. Final document vectors come from mutable, unverified gitignored cache bytes.

[final_run.py:142](/home/dylan/asymetric-dual-encoders/m7src/final_run.py:142) uses `encode_cached`. [teacher.py:214](/home/dylan/asymetric-dual-encoders/m7src/teacher.py:214) trusts any existing shard, while [teacher.py:238](/home/dylan/asymetric-dual-encoders/m7src/teacher.py:238) trusts `combined.f16` based only on byte size. These files can change after the freeze without detection.

The cache key is well constructed and temporary writes are atomic, so ordinary cross-encoder staleness is handled; corruption or replacement of same-shaped bytes is not.

Concrete fix: hash and validate each shard/combined file, with the hashes anchored in the freeze, or force a verified rebuild for the final run.

5. The three-leg rule does not establish weak-null familywise validity.

The code implements the registered conjunction as written: raw `ci95_raw`, the `0.8333` percentile from the same bootstrap draws, and Holm over sign-flip p-values are all wired correctly at [final_run.py:208](/home/dylan/asymetric-dual-encoders/m7src/final_run.py:208)–[239](/home/dylan/asymetric-dual-encoders/m7src/final_run.py:239).

But the ledger itself reports a marginal false-positive rate of 0.013 at nominal 0.008333. Bonferroni only guarantees familywise 0.025 when each marginal procedure is valid at 0.008333; three such 0.013 procedures can bound only to 0.039. Conjoining with a sharp-null sign-flip test does not supply a weak-null guarantee.

Concrete fix: either define the inferential claim as the exchangeability/sharp null actually tested, or replace the decisive leg with a weak-null-valid studentized/simultaneous procedure and validate its familywise calibration before access.

## MINOR

1. A third rounded decision read remains in the gate path.

[capacity_probe.py:104](/home/dylan/asymetric-dual-encoders/m7src/capacity_probe.py:104) decides `passed` from rounded `ci95[0]`; [gate.py:125](/home/dylan/asymetric-dual-encoders/m7src/gate.py:125) trusts that stored boolean. The committed probe predates raw fields and has lower bound `0.5793`, so this cannot flip the current outcome, but it is exactly the forbidden failure class.

Concrete fix: use `ci95_raw[0]`, strict alignment, regenerate the pinned probe evidence, and have the gate reject legacy evidence lacking raw fields.

2. Confirmatory bootstrap alignment is only indirectly strict.

[final_run.py:208](/home/dylan/asymetric-dual-encoders/m7src/final_run.py:208) calls `boot.paired` without `strict=True`; the subsequent sign-flip call is strict and would abort before a decision is emitted, so current delivery cannot silently shrink. Still, it violates the registered “strict on every confirmatory path” rule. The clean-4 bootstrap has the same pattern at [final_run.py:256](/home/dylan/asymetric-dual-encoders/m7src/final_run.py:256).

Concrete fix: pass `strict=True` to both `paired` calls and validate uniqueness/vector lengths when loading `perquery.json`.

The gate’s nonzero exit is now correctly propagated by `set -e`. The release/int8 table path and preprocessing—including `pool_mode`—are otherwise consistent. The dev-fp16/final-fp32 document-encode difference is intentional and explicitly registered in the ledger, not an accidental variant load.
