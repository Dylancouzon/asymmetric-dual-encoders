# M23 — upstream FastEmbed pull requests

**Created 2026-09-16 under owner ruling R22 (Dylan).** Takes the upstream half of the original M20
mandate. Depends on upstream FastEmbed's state and reviewer latency, not on our compute, so it runs
on its own clock. It may start once M22 has made the official Nano revision public, because the
registrations PR points at published artifacts and canonical vectors.

## Owner note (2026-10-07)

Dylan reports that the models are integrated into FastEmbed. M22 release work remains deferred at his request.

## Closure (verified 2026-10-07)

**M23 is complete.** Upstream fixed the padding defects in #716 and #717, merged the three model registrations in #751, and merged the dtype fix in [#754](https://github.com/qdrant/fastembed/pull/754) on 2026-10-02 (`2f2a8bfadb6d643c8774ae0234197f28b7e2b46c`). Issue [#752](https://github.com/qdrant/fastembed/issues/752) closed as completed that day. [FastEmbed v0.9.0](https://github.com/qdrant/fastembed/releases/tag/v0.9.0), published on 2026-10-07, includes both the registrations and dtype fix; the wheel and source archive are available on [PyPI](https://pypi.org/project/fastembed/0.9.0/).

Verification used the GitHub issue, merge record, changed source files, release notes, and PyPI metadata. Model inference was not rerun. Install-line and card updates belong to the deferred M22 release. The earlier parity receipts remain historical; `m22/UPSTREAM_FASTEMBED.md` records both states.

## Historical state on 2026-09-30

Upstream moved first; see `m22/UPSTREAM_FASTEMBED.md`. Deliverable 3 is **done**: the FastEmbed
maintainer merged the three registrations as qdrant/fastembed#751 (`a56e1e1`), byte-identical to
our preview-branch entries. Deliverable 1 is **superseded**: upstream #716 and #717 fixed both
padding defects on their own. Deliverable 2, the dtype fix, is the only open item: upstream Nano
still returns float64. The rebased patch is `m22/fastembed-dtype-fix-on-a56e1e1.patch`; opening
no PR is planned: the owner narrowed the deliverable to issue qdrant/fastembed#752 with the
proposed fix (2026-09-30).

## Deliverables (as registered 2026-09-16)

The three-PR plan recorded in `m21/FASTEMBED.md`, each branched from current upstream `main`:

1. **Padding fixes** — the fixed-padding-length fix and the eager `pad_token` default fix. Behaviour
   changes, reviewed on their own.
2. **Dtype fix** — `a4452ac` **plus** `47a5090`, never the former alone (the intermediate variant
   breaks float16). Includes the float16/float32/float64 and overflow tests.
3. **Model registrations** — the three native entries (`constella-nano`, `constella-zero`, the
   Stella document tower) with reference-derived canonical vectors. This PR alone satisfies the
   original "one clean upstream PR" clause. It must not include the unrelated #703 work and must
   not be the preview branch merged wholesale.

Before opening any PR: run the complete upstream test suite to green on a machine with room (M21
could not finish it on the WSL box; it downloads roughly 10 GB into tmpfs) and record the run.

## Constraints

No model bytes change. Canonical vectors are derived from the frozen references in `m14/`, not
from a FastEmbed run. Expected reviewer pushback and answers are in `m21/FASTEMBED.md`.
