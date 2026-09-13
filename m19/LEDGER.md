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
