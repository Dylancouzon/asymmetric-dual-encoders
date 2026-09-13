# M19 bootstrap Astra review — 2026-09-13

## Identity and brief

- Reviewer: `gpt-6-astra`, reasoning `xhigh`, task `/root/m19_bootstrap_review`.
- Mode: read-only adversarial implementation/protocol review of commit `1a705b2`.
- Result: `NO-GO` before term inventory; no P0, two P1 and four P2 findings.

The prompt admitted only `CLAUDE.md`, `instructions-m19.md`, the M19 bootstrap documentation,
registry, lock, common/inheritance implementation and tests, and six named M18 manifests. It
forbade edits, recursive searches and every protected surface: `results/perquery.json`, untouched
frozen evaluation, reserved qrels, `work/m9reserve`, LoTTE, spent M7–M13 evaluation and all raw or
tracked M18 confirmation payloads.

## Findings and dispositions

| Severity | Finding | Disposition |
|---|---|---|
| P1 | Inheritance verification did not bind the prospective M19 registry. | Fixed: v2 binds exact registry, instruction and global-guidance bytes; mutation tests cover every consequential recipe section. |
| P1 | Default generation could replace an existing lock and bless changed index/BM25 bytes. | Fixed: every inherited component has a reviewed expected file hash; ordered IDs and indexed corpus receive semantic checks; publication is exclusive/idempotent and refuses replacement. |
| P2 | Read admission allowed arbitrary inherited-index children and did not seal fresh confirmation. | Fixed: inherited data is an exact-file set; generic confirmation reads are refused; the dedicated reader requires claimed state, exact claimed path and matching bytes. |
| P2 | Registry omitted mandatory safety, judgment, packet and source-exclusion rules. | Fixed: numeric/version slice minima, audit coverage, clarification policy, packet construction, split and metric rules are explicit. |
| P2 | Released effective-row identity was absent. | Fixed: registry and lock bind chunk-dequantized effective int8 rows plus code/scale and pooling identities without a full-table float32 allocation. |
| P2 | Tests read real inheritance and used a fixed mutable symlink path. | Fixed: unit tests now use `tmp_path`, mocks and small synthetic arrays. Real inheritance is verified separately by `python -m m19src.inherit --verify`. |

The pre-review v1 lock remains preserved in commit `1a705b2`; the corrected v2 lock supersedes it
without rewriting that history. A fresh Astra re-review is required before inventory starts.

## Reviewer access log

The reviewer reported opening exactly: `CLAUDE.md`, `instructions-m19.md`, the six bootstrap M19
documents/artifacts, four M19 source/test files, `m18/registry.json`, `m18/execution-lock.json`, and
the named M18 source/corpus/index/system manifests. Commands were bounded `git rev-parse`, explicit
`git status`, `wc`, `nl`, `sed`, `sha256sum`, and two in-memory Python reproductions with inherited
data access and writes mocked. It reported no recursive search, work-payload read, prohibited read,
write or pytest execution.

## Re-review of `655ee6c`

The same Astra reviewer returned `GO` for bounded term inventory with no P0/P1. Its independent
in-memory checks rejected registry/instruction mutations, changes to all thirteen inherited files,
document/BM25 order and indexed-corpus inconsistencies, pooling inconsistencies, and invalid
confirmation state/path/hash. It confirmed the effective-row hash was computed in chunks.

Three residual P2s were reported and fixed before inventory: first publication now uses an atomic
hard-link no-clobber operation; permanent historical exclusions also apply inside the claimed
confirmation reader and its root cannot be redirected; and tests exercise the production registry
and instruction dependency keys, a publication race, exact claim membership, permanent exclusion,
document order and indexed-corpus identity.

The re-review opened the original bounded set plus this review record, used only explicit status,
diff, line-number, hash and one in-memory Python command, and reported no recursive search,
work-payload/protected read, write or pytest execution.
