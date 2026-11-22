# Codex `gpt-5.6-sol` focused re-review of `m15/PAPER.md` v2, 2026-09-17

Read-only through the Codex companion runtime, named files and targeted string searches only.

## Round-one findings: 11 of 13 fixed

Findings 1 to 9, 11 and 12 confirmed fixed in v2. Two were still live:

- **10, BM25.** Section 2 fixed the classification, but Contribution 1 still said "a lexical channel
  on the same document vectors". BM25 uses its own inverted index. **Fixed.**
- **13, the evidence assurance.** The header still claimed categorically that numbers trace through
  the evidence index, although Section 3's parity figures have no index entry. **Fixed:** the header
  now scopes the claim to result numbers and names Section 3's figures as not yet indexed.

## New defects found in v2, all fixed

| Defect | Fix |
|---|---|
| The Section 6.2 table displayed sums that do not add: 0.234 + 0.845 is 1.079, not 1.09, and the same for two other rows. The ratios 1.96x and 5.10x likewise differ from the displayed components | The committed record rounds components, totals, and ratios independently. The table now lists the three figures without an equals sign, and a line states the rounding |
| The abstract and Section 1 said the protocol was "registered before the numbers existed" while Section 4 admits the table family's headline designation came afterwards | Both now point at Section 4, which dates registration per family |
| Section 4's "no dataset entered or left the headline after a number was seen" read as contradicting the same paragraph | Reworded to "either partition", which is the claim the record supports |
| "The query encoder holds no state" and "replacing it changes no stored bytes" are false: Section 6.3 prices those artifacts at tens to hundreds of megabytes | Now: it stores nothing per document, and replacing it rewrites no document vector, with a pointer to 6.3 |
| The abstract's "below the transformer tier, query-encoder compute stops deciding system cost" is contradicted by 6.2, where long queries push the ratio above five | Qualified by query length |
| Section 5.4's "recovers most of the drop at no query-side compute" is false literally, since tokenizing and lexical scoring are work | Now: with no query-side neural network, and the distinction stated |

## Confirmed clean

- Section 5.4 retention arithmetic: 0.9256, 0.979, 0.755, and the 0.074 and 0.171 drops are
  consistent.
- Section 6.3's packaging table: 270.1/46.1 is about 5.9, and the text distinguishes the two
  packaging definitions.
