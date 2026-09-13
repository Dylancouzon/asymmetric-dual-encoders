# M19 bounded implementation review request — 2026-09-13

Review type: independent, adversarial, read-only implementation/protocol review.

Commit under review: `8a5e637a` (resolve the full hash with `git rev-parse HEAD`; refuse if the
worktree has changes in the allowed files).

Decision requested: `GO` or `NO-GO` for proceeding from implementation/rehearsal to prospective
query construction. This is not authorization for real relevance judgments or confirmation. Grade
findings P0–P3 and give precise file/line evidence and the smallest safe remediation.

Inspect only these exact repository files:

- `CLAUDE.md`
- `instructions-m19.md`
- `m19/registry.json`
- `m19/inheritance-lock.json`
- `m19/term-roster-lock-v3.json`
- `m19/CODEMAP.md`
- `m19/STATUS.md`
- `m19/LEDGER.md`
- `m19/FINDINGS.md`
- `m19src/common.py`
- `m19src/inherit.py`
- `m19src/zero.py`
- `m19src/retrieval.py`
- `m19src/metrics.py`
- `m19src/judgments.py`
- `m19src/queries.py`
- `m19src/confirmation.py`
- `m19src/rehearse.py`
- `m19src/test_common.py`
- `m19src/test_inherit.py`
- `m19src/test_zero.py`
- `m19src/test_retrieval.py`
- `m19src/test_metrics.py`
- `m19src/test_judgments.py`
- `m19src/test_queries.py`
- `m19src/test_confirmation.py`
- `m19src/test_rehearse.py`
- `results/m19_rehearsal.json`
- `m19/reviews/implementation-review-request-2026-09-13.md`

You may run the prescribed interpreter on only the listed test modules, with bytecode disabled.
Synthetic temporary files must be outside the repository. Do not import a module if doing so opens
an unlisted repository file. Git commands must name only the paths above. Do not edit, create,
delete, commit, push, browse the network, or run real retrieval.

Adversarially verify:

1. Exact protected-read and write boundaries, decision-lock authentication, pre-claim query
   inaccessibility, no metric visibility before qrels freeze, immutable/no-clobber publication and
   interruption resume.
2. V0/T0 formula and scaling, int8 quantization and inherited-row preservation, deterministic NPZ,
   tokenizer behavior and absence of eager full-table float expansion. Note that no real M19 bundle
   exists yet; distinguish source/rehearsal readiness from later actual-bundle gates.
3. Source/copy exclusion before passage depth 500, artifact collapse to 100, deterministic ties,
   unchanged artifact-level DBSF@100, and supporting-passage traceability.
4. Seven-route completeness, artifact-union semantics, evidence passage selection/blinding/caps,
   seeded audit coverage/floor, exact agreement/disagreement handling, rubric clarification and
   independent reviewer/query-author policies.
5. Query cardinality/shape/AddedToken checks, split leakage and safety controls, and whether sealing
   can expose confirmation content prematurely.
6. Binary metric definitions, within-term/equal-term aggregation, headroom and eligibility gates,
   complete safety slices, and absence of inferential overclaim.
7. Whether `python -m m19src.rehearse` genuinely joins pool, blind packet, judgments, audit, qrels,
   metrics, interruption and final reconciliation rather than merely asserting disconnected units.

Absolute exclusions (do not open, hash, stat through recursive traversal, search, list, or infer
content from):

- `/home/dylan/asymetric-dual-encoders-m18/results/perquery.json`
- `/home/dylan/asymetric-dual-encoders-m18/results/frozen_eval/untouched-*`
- `/home/dylan/asymetric-dual-encoders-m18/work/m9reserve/**`
- any reserved qrels or LoTTE data
- `/home/dylan/asymetric-dual-encoders-m18/work/m18/**/confirmation_queries*`
- `/home/dylan/asymetric-dual-encoders-m18/work/m18/**/confirmation_qrels*`
- `/home/dylan/asymetric-dual-encoders-m18/results/m18_confirmation*`
- all other files under `work/` and `results/`, including the real corpus, index and released bundle

Do not recursively search `work/` or `results/`. Return a complete access/command log and explicitly
state whether any protected or unlisted content was accessed.
