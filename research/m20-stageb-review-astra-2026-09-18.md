# M20 stage-B readiness — Astra review, 2026-09-18

Reviewer: Codex `gpt-6-astra`, read-only sandbox. Brief: `work/m20/logs/astra_stageb_brief.md`.
Full transcript: `work/m20/logs/astra_stageb_review.log`. Reviewed at `7d72f56`.

**NO-GO on entering stage B with the reviewed code (`7d72f56`).** Two crash/access defects need correction before tagging. A separate handoff defect should currently cause a safe pre-tag refusal.

1. **P1 — Archive export reopens protected payloads after every system has completed.**  
   [reserved_support.py:490](/home/dylan/asymetric-dual-encoders/m13src/reserved_support.py:490) calls `load_payload` for all four datasets; that function performs a fresh file read at line 214. The exporter runs **after** all eight atomic system outputs exist, through [reserved_transaction.py:317](/home/dylan/asymetric-dual-encoders/m13src/reserved_transaction.py:317).

   Consequently, continuation with all systems persisted still reopens all four payloads. A crash during export repeats these reads again. This contradicts the explicit “no additional protected read” archive contract in [REGISTRATION.md:166](/home/dylan/asymetric-dual-encoders/m20/REGISTRATION.md:166), including the earlier triage’s claim that this was satisfied.

   **Before the tag:** export from authenticated payload objects already loaded for scoring, persist the required archive evidence before completion makes those reads unavailable, and ensure continuation after all system outputs exist performs zero protected reads.

2. **P1 — A finalization crash leaves neither supported completion path usable.**  
   [reserved_transaction.py:329](/home/dylan/asymetric-dual-encoders/m13src/reserved_transaction.py:329) writes the reserved result before updating the six-set result at line 336. A crash between those writes leaves:
   - Normal continuation refusing because `RESULT.exists()` — line 294.
   - `--reserved-publish-only` refusing because the six-set result remains `INCOMPLETE_RESERVED` — lines 266–268.

   The scores survive, but the registered completion artifacts cannot be finished through either provided path. The publish-only fix recorded in triage does not cover this window.

   **Before the tag:** provide an authenticated, payload-free completion path from the persisted reserved result that can finish the remaining result/ledger writes without rescoring or changing the tagged identity. Verify this specific crash window with synthetic evidence.

3. **P2 — The current automatic handoff rejects its own tracked pre-encode receipt.**  
   [reserved_transaction.py:80](/home/dylan/asymetric-dual-encoders/m13src/reserved_transaction.py:80) permits only `?? results/m13_reserved_preencode.json`. The authorized `git show 7d72f56 --stat` shows that commit added this receipt to Git. Stage A subsequently rewrites it, including its timestamp, through [pre_encode.py:435](/home/dylan/asymetric-dual-encoders/m8src/pre_encode.py:435).

   At handoff it is therefore a **modified tracked file**, which `_clean_pushed` rejects before `_begin`. The launcher provides no intervening commit step. This preserves access but prevents unattended completion.

   **Before entering B:** the final `COMPLETE` receipt must be committed and pushed, or the preflight must narrowly accept this generated modification after validation and include it in the BEGIN commit.

No additional essential defect was established in the reviewed roster, per-query persistence, allocator handoff, or descriptive calculations. The mixed allocator settings do not themselves invalidate the registered encoding identity; existing shards are hash-checked. Stage B reports the two CQADupStack forums separately; the registered twelve-forum mean requires stage C’s other ten forums. Reserved reporting preserves R1/R2, zero alpha, FEVER’s caveat, and the existing six-set decision.

The visible transaction pushes the tag before calling the protected scorer. **Certification limits:** `access13.py`, `paths_guard.py`, and `.gitignore` were requested but not read; their underlying guarantees remain unverified here. `git ls-remote --tags origin` failed because GitHub DNS resolution failed, so I could not independently verify current remote tag absence.

**Access log — files read, wholly or partially:**

```text
CLAUDE.md
m13src/score13.py
m13src/reserved_transaction.py
m13src/reserved_support.py
m20src/roster.py
m8src/pre_encode.py
scripts/m20_local_run.py
m20/REGISTRATION.md
m10/final_run_registry.json
m13/RULINGS.md
m13/RESERVED_EXECUTION.md
m20/REVIEW_TRIAGE.md
m20/STATUS.md
m20/beir15_registry.json
m14/HANDOFF.md
results/m20_allocator_probe.json
research/m20-allocator-review-sol-2026-09-17.md
work/m20/logs/local_run.log
```

Git inspections: exactly the four authorized commands. No protected payloads, evaluations, tests, edits, or process-control actions were performed.
tokens used
122,448
**NO-GO on entering stage B with the reviewed code (`7d72f56`).** Two crash/access defects need correction before tagging. A separate handoff defect should currently cause a safe pre-tag refusal.

1. **P1 — Archive export reopens protected payloads after every system has completed.**  
   [reserved_support.py:490](/home/dylan/asymetric-dual-encoders/m13src/reserved_support.py:490) calls `load_payload` for all four datasets; that function performs a fresh file read at line 214. The exporter runs **after** all eight atomic system outputs exist, through [reserved_transaction.py:317](/home/dylan/asymetric-dual-encoders/m13src/reserved_transaction.py:317).

