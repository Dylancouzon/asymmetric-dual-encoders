# M20 review triage

## Round 1 — Astra (gpt-6-astra, xhigh), 2026-09-16, **NO-GO**

Report: `research/m20-codex-impl-review-2026-09-16.md`. Brief:
`research/m20-review-brief-astra-2026-09-16.md`. Access log audited clean: every file it opened is
on the brief's list, no protected payload, no recursive search, nothing written. Seven P1 blockers,
one P2, no P0. All seven accepted and fixed; the P2 was also fixed because the fix was smaller than
the documentation it would otherwise have needed.

| # | finding | verdict | fix |
|---|---|---|---|
| 1 | Registered cap exceeds the $1,000 ceiling by $1.77: the 187 new hours were priced at the all-retained rate, which omits sibling-volume storage during the inherited 55.2 h | accepted | Stage C cap 125 → 120 h, total 237.2 h, new spend $335.72, committed plus new $992.76. The cost formula is registered explicitly, and the controller now recomputes it from the **live** quote and refuses if either the live price or the ceiling is breached |
| 2 | Stage caps not enforced per stage; each tower process got the whole cap; the projection counted encode seconds, ignored towers not yet run, and treated unpinned corpora as zero documents | accepted | The controller hands all three tower processes **one absolute stage deadline** and wraps each stage in `timeout`. The projection now converts its measured rate through the registered tower cost ratios to cover towers still to run, and projects against the **registered** document volume the cap was built from. Archive transfers carry explicit timeouts drawn from the remaining stage budget |
| 3 | The archive dropped the reserved queries and qrels, a registered R22 deliverable | accepted | The tagged transaction now exports them into the archive layout itself, from payloads it has already opened for scoring. No additional protected read, deliverable intact. The archive builder **requires** those files for a reserved corpus and fails without them |
| 4 | The C/D controller could report PASSED while the BEIR-15 table existed only on a pod about to be stopped | accepted | It now pulls the table, per-query rows, corpus pins and pre-encode receipt, checks the table is `COMPLETE` and was produced against this registration, then commits and pushes before reporting success |
| 5 | Archive verification could certify an incomplete or changed vector archive: a missing tower was silently skipped, staged files were accepted on size alone, and the builder hashed whatever it found into a fresh manifest | accepted | Shards are enumerated **from the source manifest**, which must be `COMPLETE`; every shard is hashed at source and at destination against its recorded hash; a build over less than the full registered inventory is `PARTIAL` and raises; `verify` refuses a `PARTIAL` manifest |
| 6 | No re-hash of the object-storage copy, which the registration promises | accepted | `--verify-remote` asks the destination for its own SHA-256 of every object and compares against the manifest. Still blocked on the owner providing a bucket and credentials |
| 7 | BEIR-15 resume re-ran completed producers, overwrote their run files, and hashed the new files instead of the recorded ones; scored rows were accepted by presence; shards loaded with `verify=False` | accepted | A completed producer's run is **reused** and checked against the hash its own scored row recorded; a rebuilt run must reproduce that hash; scored rows are checked for status, identity and registration; every system on a corpus must have scored the same query set; shards load with `verify=True` |
| 8 (P2) | A crash between writing the reserved result and pushing it left no supported completion path | fixed anyway | `score13.py --reserved-publish-only` re-commits and pushes an already-computed result. It opens no payload, scores nothing, and refuses unless the result is present and already `COMPLETE` |

Astra could not certify the access boundary because `m8src/paths_guard.py` was not on its file
list. That is the brief's error, not a finding; the file is on round 2's list.

### Found in self-review during the same round, not by Astra

- **P0.** `reserved_support.corpus_for` imported `m8src/pre_encode.py`, which claims the
  corpus-only allowlist entry at import time. `paths_guard.claim` refuses a second, different claim
  in one process, so BM25 would have raised **after** the one-shot access was spent. Corpus loading
  is now inlined under the capability the transaction already holds. Two tests cover it.
- **P1.** BEIR publishes labels in a separate repository with its own revision, and the loader
  passed `revision=None`. All twelve public qrels revisions are now pinned.
- **P1.** A derived DBSF row hashed whatever run file was on disk rather than the hash its input's
  scored row recorded.

## Round 2 — Fable

Brief: `research/m20-review-brief-fable-2026-09-16.md`. Scope: the P1 fixes above, plus the access
boundary Astra could not certify.
