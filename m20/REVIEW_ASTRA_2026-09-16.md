- `instructions-m20.md`, `m13/RULINGS.md`, `m13/RESERVED_EXECUTION.md`, `m10/final_run_registry.json` — M20 and the implementation pin allow resuming protected reads after the spent tag, while R14 and the registry’s withdrawal note explicitly prohibit post-tag continuation. This could cause a second protected access.  
  Recommendation: reconcile every document and registry field to R14—after the tag, report persisted output and permit recovery only from persisted scores with zero protected reads—or obtain a new explicit pre-observation owner ruling.

- `instructions-m20.md`, `m20/PLAN.md`, `m13/RULINGS.md` — “Ten of thirteen are already scored” is incorrect when CQADupStack counts as one dataset: only two of its twelve forums are reserved, and no forum-to-dataset aggregation is specified. Touché is conditional, so failed licence verification would also leave twelve datasets while the documents still require “BEIR-13.”  
  Recommendation: define all twelve forum rows and their aggregation, and specify whether failed Touché verification blocks M20 or requires truthful BEIR-12 naming.

- `m20/PLAN.md`, `instructions-m20.md` — The roster does not pin an immutable Constella Zero revision/file identity or the literal Stella query prompt and preprocessing contract. Either omission can change reported scores.  
  Recommendation: place these exact identities in the pre-observation registration before implementation and review.

- `m20/PLAN.md`, `m7/LEDGER.md` lines 110–120, `CLAUDE.md` — The plan rejects wrapper licence tags and says the M7 primary-source evidence governs, but that allowed evidence section contains no primary-source MS MARCO licence record.  
  Recommendation: record the primary-source MS MARCO terms and validation-only permission before pinning it, or exclude it.

- `instructions-m20.md`, `m20/PLAN.md`, `m13/RULINGS.md` — The archive destinations are only “object storage” and `D:`; no bucket/prefix or mounted local path is identified. The archive exit gate and M22 storage-retirement handoff therefore cannot be reproduced from these documents.  
  Recommendation: register the exact two destinations and manifest locations before the run.

**NO-GO** — resolve these before M20 execution or any protected access.

No edits, tests, or side-effecting commands were run. I also ran the permitted `git log --oneline -5` and `git tag`. During a final non-recursive keyword check, isolated `m7/LEDGER.md` matches outside the permitted line slice were inadvertently displayed; they were not used in the findings.

Files opened: `ROADMAP.md`; `PROJECT_STATUS.md`; `CLAUDE.md`; `instructions-m20.md`; `instructions-m22.md`; `instructions-m23.md`; `instructions-m14.md` (R20 section); `instructions-m15.md`; `m13/RULINGS.md`; `m13/RESERVED_EXECUTION.md`; `m13/STATUS.md`; `m14/HANDOFF.md`; `m14/STATUS.md`; `m20/STATUS.md`; `m20/PLAN.md`; `m21/STATUS.md`; `m21/FASTEMBED.md`; `m21/BENCHMARKS.md`; `m13src/reserved_support.py` (header, `SYSTEMS`, report derivation); `m7/LEDGER.md` (requested lines 56–125, plus the accidental isolated matches noted above); `m10/final_run_registry.json` (`reserved` key); `results/eval_manifest.json`.