# /andrey-review of PAPER.md draft v7 (session, 2026-10-01)

Legwork: every cited file exists; figure PNGs at most 2040 px and 125 KB (PDFs for LaTeX); Qdrant facts checked against the live docs (TurboQuant v1.18+, server-side `qdrant/bm25` with IDF, DBSF). Verdict fix-then-ship; all five fixes applied.

- Scope the screen to the closed-form table in the abstract (served Zero is trained; one transfer point).
- Replace the family-by-family narration in Section 4 with a small table.
- State the `indexing_threshold` benchmark setting in Section 7 (not a recommendation).
- Cut the Section 6.3 blend null result to one paragraph.
- Length: keep either the routing table or Figure 2's right panel (kept both; the table carries the numbers the figure shows as shares).
