# R25 — Constella models move to the `Qdrant` Hub org (Dylan, 2026-09-25)

Requested by Neil on qdrant/landing_page#2787 ("can we not release on the Qdrant HF with a
'preview' tag?"). The owner approved on 2026-09-25 and confirmed Qdrant's endorsement, the licence
and his org write access. Recorded here because M13 and M14 are closed; M22 owns the release.

**Ruling.** The three Constella repos move from `DylanCouzon/` to `Qdrant/` under the same names,
by Hub repository transfer. A transfer keeps commit history, so every pinned revision (Nano weights
`6bb167dc`) stays valid and the old paths redirect. Names carry no `-preview` suffix: the cards'
existing `research-preview` tag and banner carry the status, and M22 removes them without another
rename. No weight, tokenizer, config or benchmark number changes.

**Consequences for later work.** M22 and M23 target `Qdrant/`. `m14/verify_card.py` and
`m14/MODEL_CARD.md` are closed M14 records and keep `DylanCouzon/`; M22 copies the verifier into
`m22/` with the new names instead of editing M14. Other historical records keep the names they
were written with.

## Receipt (2026-09-25)

| Repo | Head before = head after transfer | Card commit (IDs renamed) |
|---|---|---|
| `Qdrant/constella-nano` | `e1a51bff5b15210c93e903406fb655a40ff4e61a` | `b037b20d1c220bf33dce254a7f386e401ac0703a` |
| `Qdrant/constella-zero` | `ebee6ea94999f26182505548ffc5df035e727351` | `b3cc655bab21cb17fe0cc13d92bb131ae8d555ee` |
| `Qdrant/stella-en-400M-v5-doc-onnx` | `9ec7ac44fc2a6e1e768407d1fba9e81c4d6d9317` | `293ec490f8e8c56ca468f84c84b8095bac954254` |

Verified anonymously after the transfer: the head SHA and every file's LFS oid or blob id match the
pre-transfer snapshot; all three repos are public and keep the `research-preview` tag; Nano
revision `6bb167dc6f60d3992602235b8e8aaa374a309168` resolves; each `DylanCouzon/` path resolves to
its `Qdrant/` repo, and a download through an old ID succeeds. Each card commit changes only
`README.md` (`DylanCouzon/` → `Qdrant/` in Hub IDs and links), and the downloaded bytes at that
commit match the uploaded card.

**Pushed 2026-09-25.** FastEmbed `constella-research-preview` at `2704c34`: registrations and
canonical-vector test keys renamed to `Qdrant/` (three files, nine lines; tests not run in this
session). Blog on landing_page#2787 at `3f5fc9b6b`: seven Hub references renamed; the "family of models" link points at the Nano
card.
