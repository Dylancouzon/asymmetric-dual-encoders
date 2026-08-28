# transfer/ — side-branch only, do NOT merge into `m7-query-encoder`

One artifact, here so a second machine can run the teacher-learnability probe:

- `trainq_texts.json.gz` — the 349,934 TRAIN query texts, gzip of `work/trainq_texts.json`.

**Why it has to travel rather than be re-derived.** `scripts/encode_trainq.py dump` resolves the
list through the 6.17M-row pool index and the decontamination outputs (`kept_pairs`, the B2
banned-row mask). Both are gitignored and cost most of `run_stage0.sh` to rebuild, so a machine
without them has no path to the *identical* list — and a probe fitted on a different query set is
not comparable to the committed incumbent row.

**It is verified, not trusted.** `encode_trainq.load_texts()` restores from this file when
`work/trainq_texts.json` is absent and then checks sha256 against
`results/m7_trainq_manifest.json`, refusing on mismatch. Both machines run the same check.

**Licensing.** Query text only, no documents and no qrels, from ESCI (Apache 2.0), Mr. TyDi
(Apache 2.0), and FEVER / HotpotQA / SQuAD (CC BY-SA, approved by Dylan 2026-08-25 for training
with model-card attribution). The repo is private and already vendors 11 MB of dataset queries and
qrels under `results/frozen_eval/`, so this is the same class of act with a precedent. It stays on
this branch so the release lineage's history does not carry it; merge back only the
`results/m7_learnability_*.json` outputs, then this branch can be deleted.
