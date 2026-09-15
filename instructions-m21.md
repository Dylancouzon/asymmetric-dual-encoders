# M21 — research-preview polish

Opened 2026-09-15 by owner request. A short, bounded milestone: make the already-published preview
pair (`constella-zero`, `constella-nano`) and this repository presentable for a public research
preview introduction on 2026-09-16. No new measurement, no new training, no reserved access.

M15 keeps the whitepaper mandate; this is a separate number, not a renumbering.

## Scope

Seven deliverables, all documentation, packaging or verification:

1. **Nano float64 diagnosis.** M14 shipped with the card stating that FastEmbed's integer
   attention mask promotes Nano's masked-mean output to float64 and instructing the user to cast.
   Find the exact promotion site, state whether it is Nano-specific or affects every
   `PooledNormalizedEmbedding` model, and land the smallest correct fix in the fork if the fix is
   upstream-shaped. Parity against the frozen Torch reference must not move.
2. **FastEmbed integration verified to PR standard.** Treat `m14-constella-preview` as if a
   Qdrant maintainer were about to review it: based on current upstream main, three native entries
   (nano, zero, document tower), reference-derived canonical vectors, upstream test suite green,
   no `add_custom_model` in the shipped path, and none of the unrelated #703 padding work. Do not
   open the PR — that is M20's.
3. **One benchmark table.** Consolidate the scattered numbers into a single canonical source that
   README, both cards and the plain-English page all cite, traced to committed result JSONs.
4. **Model cards cleaned.** Nano (`m14/MODEL_CARD.md`), Zero and the document tower
   (`m11/release/MODEL_CARD.md`, `m11/release/MODEL_CARD_DOC.md`): one voice, current status,
   the consolidated table, honest preview framing.
5. **README descriptive and lean.** It currently says Nano is unbuilt and M17 is planning.
6. **`research/constella-in-plain-english.md` brought current** in its existing narration style.
7. **Repository tidy.** Root clutter and stale pointers only.

## Constraints

Standing `CLAUDE.md` rules apply unchanged. The ones this milestone will actually meet:

- **No reserved or six-set access.** No reading `results/frozen_eval/untouched-*`, reserved qrels
  caches or `work/m9reserve`. Reserved access stays UNSPENT for M20.
- **No repo-wide content searches across `results/` or `work/`.** Every worker gets an explicit
  file allowlist.
- Never overwrite `results/perquery.json` or any frozen artifact, lock, freeze or spent tag.
  Preserve `m10src`, `results/m10_*` and every historical path name.
- Numbers may be **copied and re-presented, never recomputed or re-partitioned**. The headline
  stays the pre-registered clean-4 with all six beside it; ArguAna and FiQA keep their disclosed
  Stella-contact marker; unresolved contrasts stay UNESTABLISHED and are never called parity,
  equivalence or a tie. Exact-search quality numbers are never mixed with ANN or fusion numbers.
- No Hub upload, no new public revision and no PR without a fresh owner go-ahead. The published
  revision `6bb167dc6f60d3992602235b8e8aaa374a309168` stands until then.
- `.venv/bin/python`. Tests use scratch outputs; they must not rewrite real results.

## Exit criteria

- The float64 behaviour is explained in one paragraph an outsider can follow, and either fixed
  with parity evidence or recorded as won't-fix with the reason.
- The preview branch's verification result is stated as a maintainer would read it: green, or a
  named list of what a PR would still need.
- README, both cards and the plain-English page agree with each other and with the result JSONs,
  with no stale milestone status left in any of them.
- One adversarial review (Codex `gpt-6-astra`, read-only, essential-only feedback against the
  goal above) with findings and dispositions recorded in `m21/STATUS.md`.
- Any card change that would alter the published Hub revision is staged and flagged, not pushed.
