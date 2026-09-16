# M20 review triage

## Round 1 — Astra (gpt-6-astra, xhigh), 2026-09-16, **NO-GO**

Report: `research/m20-codex-impl-review-2026-09-16.md`. Brief:
`research/m20-review-brief-astra-2026-09-16.md`. Access log audited clean: every file it opened is
on the brief's list, no protected payload, no recursive search, nothing written. Seven P1 blockers,
one P2, no P0. All seven accepted and fixed; the P2 was also fixed because the fix was smaller than
the documentation it would otherwise have needed.

| # | finding | verdict | fix |
|---|---|---|---|
| 1 | Registered cap exceeds the $1,000 ceiling by $1.77: the 187 new hours were priced at the all-retained rate, which omits sibling-volume storage during the inherited 55.2 h | accepted | Stage C cap 125 → 120 h, total 237.2 h. At the live quote read the same day (sibling pods hold 530 GB and 505 GB, not two bare 500 GB volumes) that is $336.87 of new spend against a registered $337.50 allowance, so committed plus allowance is $994.54. The cost formula is registered explicitly, and the controller now recomputes it from the **live** quote and refuses if either the live price or the ceiling is breached |
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

## Round 2 — Fable, 2026-09-16, **GO**

Brief: `research/m20-review-brief-fable-2026-09-16.md`. Reviewed at `0e9aefa` with a clean tree.
Access log audited: everything it opened is on the brief's list except `.gitignore`,
`m13src/access13.py` and `m10src/m9base.py` (grepped for `paths_guard` only), a non-recursive
`import pre_encode` grep over source directories, and `ls ~/.cache/huggingface` — all named in its
report, none a protected payload, nothing written.

**No blockers.** It certified the access boundary that round 1 could not, and confirmed all seven
P1 fixes in the code rather than in the triage table:

- A process holding the corpus-only claim is refused every frozen payload, `work/dev` alias,
  `work/m9reserve`, `work/lotte` and every reserved cache path that is not a corpus config. The
  exemption requires an exact reserved corpus repository followed by a literal `corpus` segment, so
  a `-qrels` repository and a `queries` or `default` config cannot match, and the loader guard
  closes the network route.
- `reserved_support.corpus_for` is bounded to the corpus config, re-verifies the frozen manifest
  hashes and the pre-encode's document order, and runs under a claim strictly broader than
  corpus-only, so the P0 the team found itself is genuinely closed.
- `export_reserved_payload_archive` introduces no read that would not have happened anyway.

### Debt recorded, not fixed

1. **The guard classifies the `datasets`-style cache but not the hub-style blob cache.** No code
   path on this branch opens raw cache files and the loader-identity guard blocks the API route, so
   this is a blind spot in the bulkhead rather than a reachable hole. Pre-existing, not introduced
   by M20. Owner: whoever next touches `m8src/paths_guard.py`.
2. **The projection's rate is documents per encode second.** Corpus download, per-tower re-hashing
   and model load sit outside it, so it under-projects non-encode overhead. The direction of harm
   is a hard `timeout` kill instead of a clean shard-boundary stop, in stage C only, which is
   unprotected and resumable.
3. **The reserved archive payloads live only on the pod volume between stages B and D.** If that
   volume is lost, `archive --build` hard-fails and cannot be repaired without a second protected
   access. The substance survives in git, so this is a deliverable-completion risk, not evidence
   loss. Mitigation available if the owner wants it: run stage D's build immediately after stage B.

### Debt fixed anyway, because each was a line or two

- Two stale docstrings that contradicted the code beside them: `m20src/archive.py`'s header still
  said the reserved labels were excluded, and the registration's layout block still annotated
  queries and qrels as "public corpora only".
- `archive.build_corpus` accepted the reserved payload files on presence. They are the only archive
  payloads the builder cannot regenerate, so they are now checked against the hashes the
  transaction recorded in `results/m13_reserved_run.json`.
- `_score_bm25` held two corpora's text at once across the FEVER-to-DBpedia boundary, about 10M
  passages. It now frees each dataset's text before loading the next.
