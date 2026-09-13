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
