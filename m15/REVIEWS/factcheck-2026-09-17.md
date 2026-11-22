# Number-by-number fact check of `m15/PAPER.md`, 2026-09-17

Sub-agent pass over the brief's source allowlist. Every registered contrast, the M7 confirmatory
family, the DBSF depth ladder, the fusion table, the serving-cost table, the edge and container
tables, the two no-rounding rules, and the coverage-against-capacity numbers matched their sources
digit for digit.

## Essential findings and disposition

| # | Finding | Disposition |
|---|---|---|
| 1 | "65 times faster than either transformer tier" is 64.8x for the student and 61.1x for bge-small | Fixed: "61 to 65 times" |
| 2 | The draft called `m14/STATUS.md`'s "maximum comparison error" a "cosine error". The source keeps the two distinct | Fixed in `PAPER.md` and `HOTSWAP.md` |
| 3 | "Every comparator in Section 5 trains on MS MARCO" is false for BM25, which trains on nothing | Fixed: the claim now names the neural comparators and excludes BM25 |
| 4 | Unsourced within the allowlist: "11 teachers", "ranked fifth", and the absorbability "rank agreement 1.000 / 0.000" | "11 teachers" and the fifth-place rank are sourced in `m7/RESULTS.md`, which the allowlist omitted; both verified and kept, with the rank reassigned to the distilled table. The rank-agreement figures appear in no source and are withdrawn |
| 5 | Three Appendix A lines trace to `m17/FINDINGS.md`, `m18/FINDINGS.md` and `m19/FINDINGS.md`, outside the allowlist | Allowlist gap, not a drafting error. Those files stay on the spot-check list |
| 6 | The evidence index listed every headline batch as an unverified candidate while the draft already used the numbers | Fixed: the spot-check table now records who checked what and when |

## Noted, no change

- The draft says "Resolved below the bar" where `m7/STATUS.md` says "ESTABLISHED BELOW THE BAR".
  Same direction, and the draft's wording avoids reading "established" as a win.
- The Appendix says eight levers; `m8/FINDINGS.md` reports 12 probes and a closed-hypothesis table
  of eight rows. Both numbers are real and describe different things.
