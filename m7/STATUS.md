# M7 status

**Stage: Codex review #3 findings being actioned BEFORE the ablation spend (2026-08-27 late).**
Candidate: **`p35w-2m-s2500`** (B 16k + 923,590-span pseudo mix → A @ 1e-3 s2500), full-suite
dev macro 0.6113 — lever #2 (pseudo-query coverage) adopted through the pre-registered chain,
+0.0126 over `s2w-1e3-s1000`. Levers #1 (bigram closed-form: −0.0301, structural) and #3
(doc2query: +0.0054 unresolved at cheap-test price) closed on evidence. Review #3 (gpt-5.6-sol,
read-only): **4 BLOCKER / 8 MAJOR / 4 MINOR + 4 ideas** — verbatim in
`research/m7-codex-review-2026-08-27.md`; it stopped the ablation launch. Key reframe accepted:
dev p-values are SELECTION evidence, only the three frozen-test comparisons are confirmatory.

## Run next, in this order (from the review; nothing GPU-heavy until 1–4 are done)

1. **B1 dependence recompute**: `heldout-longq` ⊂ `heldout-train` (55 shared qids) — recompute
   tonight's three lever comparisons with one shared sign per underlying qid + stratified
   bootstrap, ordinary and dependence-preserving side by side. Could reverse the marginal s2500
   pick (+0.0023); the candidate id may revert to `p35a-2m-1e3`.
2. **B2 pin the held-out components** (hashes of both JSONs, ordered qids, six-name abort-on-
   missing list) into the manifest; disclose the late pinning in LEDGER.
3. **B3/B4 ablation harness fixes**: `init_preproc` field so the mandatory prefix arm toggles
   runtime prefix ONLY; every arm = two runs (B then fresh-optimizer A), fixed B16k→A2500, no
   per-arm step selection. Also: slice `input_emb_rows` to vocab_size (MAJOR 4), reg control is
   1e-3-vs-0 (MINOR), dedupe baseline arms → 7 chains + the **matched no-pseudo B16k→A2500
   control** (MAJOR 2 — needed to attribute lever #2's gain honestly).
4. **MAJOR 3 equivalence check**: matrix eval vs `QueryTable` path per query for every decision
   artifact; the gate/final path uses QueryTable, never the matrix shortcut.
5. **Count-saturation probe** (review idea #1): binary/cap-2/sqrt pooling on the existing
   candidate — eval-only, pre-register first. Cheapest untested genuine capacity.
6. **Corrected mandatory ablations** overnight (~7 chains + control).
7. Then: fusion re-selection (int8, DEPTH=1000, incl. convex0) → ANN sweep + costs → gate as
   mechanical eligibility audit (spec in the review: QueryTable path, hash verification, abort
   on missing, unrounded per-query dumps, no recipe change after) → freeze → single final run.
8. Exploratory, pre-register before running, compute permitting: B-phase extension (B curve
   still rising at 16k — review idea #4), A-only bigram rows (idea #2).

## Provenance debts from the review (do with 3/4, cheap)

Compare/adoption artifacts must store unrounded macros, per-component CIs, per-query values (or
hashed blobs), encoder fingerprint, table hashes (MAJOR 5/6); doc2query JSONL sha recorded
(MAJOR 7); `compare_release.py` records the active encoder fingerprint (MINOR).

## Open for Dylan

1. Nothing blocking. Review #3 findings all actionable locally.
2. doc2query revival (licensing ruling on a clean generator) remains yours if ever wanted.
3. Windows Update reboots remain the top operational risk.
