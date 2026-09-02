# M1–M6 findings and decisions (archived from CLAUDE.md, 2026-09-01)

Verbatim sections moved out of CLAUDE.md during the M9 close-out so the file every session loads carries only the standing rules and the stage plan. Nothing here was edited; numbers are as recorded at the time. Later milestones (M7–M9) supersede where they conflict.

## Candidate routes (to be confirmed in M1)

- LightRetriever (arXiv 2505.12260) with released pretrained artifacts — the zero-compute-query anchor point.
- Small symmetric baselines (20–100M): bge-small-en-v1.5, e5-small-v2, all-MiniLM-L6-v2, gte-small, arctic-embed-xs — the "just run a small model" alternative.
- Static embedding models used symmetrically (Model2Vec/potion, sentence-transformers static-retrieval) — the zero-compute symmetric strawman.
- Aligned asymmetric pairs that need no training by us (e.g., MongoDB LEAF: docs with teacher, queries with distilled small model) — the middle of the spectrum.
- Inference-free sparse retrieval (SPLADE-doc, uniCOIL/TILDE doc expansion, OpenSearch inference-free doc encoders, doc2query+BM25, BM25 itself) — same zero-query-compute property, sparse instead of dense, Qdrant-native. Added 2026-08-24 after Dylan flagged the instructions as a surface-level draft; this family is a direct competitor the draft missed.
- Broader literature sweep for LightRetriever successors, industry asymmetric towers, and tiny-adapter alignment (static model + learned projection into a frozen big-model doc space).

## Headline results (6 named datasets, exact search; claims stated only where bootstrap-resolved)

Quality groups (avg-6 nDCG@10; strict within-group ordering NOT established — Codex blocker 2):
- Group A, small/mid transformers on the query side: arctic-m-v1.5 0.5264, leaf-ir-asym 0.5155, mdbr-leaf-ir 0.5123, bge-small 0.5042, arctic-s 0.4993, granite-r2 0.4947, gte-small 0.4837.
- Group B, zero-neural-query-compute: opensearch-doc-v3-gte 0.4868 (overlaps group A's tail), LR-hybrid-pertask 0.4720, LR-hybrid-websearch 0.4594, LR-dense-int8/pertask 0.4586/0.4583, LR-dense-websearch 0.4320, BM25 0.4174. (LR sparse/hybrid carry the unresolved −3..−5 reproduction gap → "conservative local reproduction", not a finished competitor result.)
- Group C, symmetric statics: 0.3193–0.3601 — decisively last.
- Resolved pairwise: LR-dense-websearch vs bge-small −7.2 [−8.8,−5.7]; vs best static +7.2 [+5.2,+9.2]; per-task tables +2.6 [+2.0,+3.3]; leaf-asym vs teacher −1.1 [−1.5,−0.7]; leaf-asym vs leaf-sym unresolved (+0.3 [−0.2,+0.8]) → say "retains comparable quality", not "beats".
- Costs: LR lookup 0.023 ms / 466 MB fp16 raw table (int8 233 MB, quality-free); OpenSearch query side 0.018 ms / 0.9 MB idf (doc-side postings cost measured separately, see reruns); 33M transformer ~5 ms / 66 MB fp16 / 1.3 s load; statics ~0.2 ms / 15–65 MB. Cost rows are not one number: query asset ≠ doc index ≠ hydration.

## Rerun outcomes (2026-08-25, post-Codex)

- Projection (fixed loaders, oracle λ): potion-8M→arctic-m best 0.3036; potion-32M→arctic-m best 0.3280, below its own symmetric 0.3427. Negative result is airtight: linear post-hoc alignment into a contextual doc space fails even with test-set-tuned regularization.
- New resolved pairs (6-ds bootstrap): leaf-asym > bge-small +1.1 [+0.2,+2.0] p=0.016; opensearch > lr-hybrid-pertask +1.5 [+0.2,+2.8] p=0.022; opensearch ties gte-small (p=0.66); opensearch < bge-small −1.7 [−2.8,−0.7]; **lr-dense-websearch ties BM25** +1.5 [−0.1,+3.0] p=0.07; lr-dense < e5-small −2.2 p=0.003; hybrid adds +2.8 over dense (websearch).
- ANN sweep: lookup-query vectors are harder for HNSW — LR default ef loses 2.1 nDCG on FiQA vs bge-small's 0.7; at ef=512 both mostly recover (−0.5 vs −0.2). TREC-COVID shows no visible ANN penalty (n=50).
- OpenSearch doc-side postings (full vocab, 5K-doc sample): 233 nnz/doc mean → ~1.4 GB per 1M docs. Doc-index ladder per 1M docs: bge-small 0.77 GB (384d fp16) < opensearch 1.4 GB < leaf/arctic-m 1.54 GB (768d) < LR 3.07 GB (1536d).

## Codex gate (2026-08-25) — verdict "not decision-grade yet"; all reruns executed

Findings and dispositions:
- BLOCKER 1 projection loader mismatch (trained on ST-wrapper potion, evaluated on model2vec-native vectors): FIXED — re-encoded with StaticModel end-to-end, refit with oracle-λ selected directly on test retrieval (strongest possible shot for the method). Rerun in `results/codex_reruns.log`.
- BLOCKER 2 ±0.007 was 5-ds-derived, ladder overclaims: FIXED — headline reworded as groups; significance rerun on 6-ds with 21 pairs incl. all near-neighbors; p reported as bounds (p<2e-4).
- MAJOR 3 LR sparse/hybrid unresolved reproduction: quarantined with explicit label.
- MAJOR 4 Edge prototype proves query-path composition, not production storage/cold start: reworded; token shard as a default-indexed collection is the wrong storage shape (1.82 GB vs 466 MB raw fp16); load times labeled warm-cache.
- MAJOR 5 ANN behavior measured for one system/corpus only: FIXED — ef sweeps rerun for bge-small (fiqa + trec-covid) and LR (trec-covid) in `results/ann_sweep.json`.
- MAJOR 7 OpenSearch doc-side postings cost unmeasured: FIXED — full-vocab nnz/doc on 5K-doc sample → `results/opensearch_index_cost.json`.
- MINOR 8/9/10 (p=0 as bound, cost definitions, LEAF wording): adopted.
- Ship-scope per Codex: all comparative claims scoped to "the six named datasets"; no generalization to production workloads, million-scale, filtered search, or non-English.


## Key decisions (log)

- **WITHDRAWN THE SAME DAY, on evidence: `arctic-embed-l` is worse than the teacher we already
  have.** Ranked by the closed-form table distilled from it — the artifact that ships — arctic is
  −0.0480 [−0.0608, −0.0349] below bge-base, and a teacher's own retrieval quality turns out not to
  predict its distilled table at all (Spearman 0.000 over eight candidates). Only
  **stella_en_400M_v5** beats the incumbent (+0.0365 [0.0249, 0.0481]). The teacher question is back
  with Dylan because stella's disclosed training data covers 2 of our 6 eval datasets;
  `m7/LEDGER.md` pre-registers a four-dataset primary comparison as the answer to that.
  `results/m7_learnability_report.json`. The entry below records what was decided and why, and is
  kept because the failure mode — selecting a teacher on the tower instead of on the table — is the
  lesson.
- **[SUPERSEDED THE SAME DAY — see the entry above; the teacher is stella_en_400M_v5.]**
  **Teacher for M7 was `Snowflake/snowflake-arctic-embed-l` (Dylan, 2026-08-26).** Chosen on
  measurement, not projection: best of five candidates on the two CQADupStack dev components
  (+0.0447 [0.0339, 0.0557] over bge-base; arctic > stella +0.0125 [0.0008, 0.0241] raw, which
  would NOT survive multiplicity over the ten pairs, so the top is arctic ~= stella), Apache-2.0,
  and the only candidate whose MTEB registry entry discloses **zero overlap with our six** — stella
  lists ArguAna and FiQA2018, 2 of the 6. Dylan ruled on the vendor question explicitly, because the
  released table only works against its teacher's document vectors, so the doc side of a Qdrant
  release would be Snowflake's model. `results/m7_teacher_probe.json`,
  `results/m7_teacher_contamination.json`. The projection that had ranked stella first is not merely
  imprecise on this evidence, it is wrongly ordered.

- BEIR subset: SciFact, NFCorpus, FiQA-2018, ArguAna, SciDocs (100,785 docs total; all appear in LightRetriever's tables and on MTEB → like-for-like comparison possible).
- Harness: hand-rolled — HF `datasets` (BeIR/* repos) + `pytrec-eval-terrier` + numpy brute force. `beir`/`mteb` packages skipped (their value is model wrappers we don't use). Title+text join: `(title + " " + text).strip()`. Python 3.12 venv, torch 2.13 MPS.
- LightRetriever config: `lightretriever/lightretriever-qwen2.5-1.5b` (ungated, 3.1 GB bf16, fits 24 GB RAM). Paper BEIR-15: dense 48.9 / sparse 47.3 / hybrid 52.1. MPS gotchas: use `sdpa` not flash-attn, explicit bf16, no autocast.
- Llama-based MRL adapter (dimension truncation) is gated on HF — skipped unless Dylan requests access. Mistral-7B variant ships a reference lookup table to validate our table construction against.

## Findings (log, M2)

- Harness validation (corrected per verification B2): 50 of 58 overlapping cells within 0.001 of official MTEB; 10 of 12 configs within 0.0005 on the 5-ds average. The ArguAna gap was self-hits: BEIR drops doc_id==query_id (`ignore_identical_ids`); harness now does too. Query prefixes stay on everywhere (empirically validated; an earlier prefix diagnosis was wrong, reverted).
- Verification B1 (fixed): transformers 5.x loads checkpoints in their config dtype — granite ran bf16, gte fp16, non-comparable. dtype now pinned fp32 everywhere; both models' artifacts deleted for re-encode; caches carry meta.json (dtype/prefix/max_seq) and refuse stale reuse.
- Verification B2 (fixed): potion models must be encoded with the native model2vec loader — it reproduces official FiQA to 5 decimals (0.187609 vs 0.18761); the sentence-transformers wrapper deviates +0.0027. Both potions re-encoded.
- Verification B3 (adopted): every reported delta gets a bootstrap CI; ±0.007 resolution stated once in the report; sub-resolution orderings reported as ties. LEAF asym-vs-sym per-dataset signs disagree with the LEAF paper on 3/5 (ArguAna reversal significant, −1.07 [−1.91,−0.22]) → resolved: our composition is byte-identical to MongoDB/mdbr-leaf-ir-asym (sha256-verified: query tower, Dense 384→768, teacher doc tower, prompts). The ArguAna reversal is a genuine finding on this subset, reported as such.
- Verification M1/M2 (adopted): 5-ds subset correlates with BEIR-15 at Spearman only 0.55 → adding TREC-COVID (171K docs; balances the subset: LightRetriever's best category) + BM25 baseline (done, 0.379). All "X beats Y" claims scoped to the named datasets in the report.
- Verification N7 (adopted): report doc-side costs too — LightRetriever's 1536-dim index is 4x bge-small's 384-dim per doc; index bytes/doc + encoding throughput go in the frontier table.
- Remaining from verification: M9 sparse-mask ablation on SciFact (paper's reference doesn't mask padding; we do), M5 headline = single websearch table (per-task tables reported as oracle upper bound), MI8 ArguAna truncation caveat.
- OpenSearch doc-v3-gte (2026-08-25): validated against its card ≤0.004 on 4/5 datasets; the card's SCIDOCS 0.455 is wrong (measured 0.1686, consistent with sibling models — use ours). 5-ds avg 0.4375: **best zero-query-compute system on this subset**, above LR hybrid per-task (0.4225) and LR hybrid websearch (0.4144), with a 133M doc encoder and Qdrant-native sparse output.
- LR sparse reproduction (2026-08-25): our sparse runs −3 to −5 nDCG under the paper (scifact 0.631 vs 0.664). Tested and excluded: padding mask before amax (masked 0.6306 vs unmasked 0.6272 — not the cause), doc-token-restricted pooling flags (all default False in their args). Cause not isolated; capped per "correct, not decimal" — dense reproduces faithfully, sparse/hybrid reported as "our reproduction, conservative under-estimate" with this caveat. Hybrid per-task 0.4225 (paper 0.4374); hybrid-websearch 0.4144. int8 table is quality-free (0.4117 vs 0.4114) and halves the fp16 table.
- M3 root-cause fix (2026-08-25): first LR dense run was −4pt avg vs paper (ArguAna −11.8). Cause: lookup tables built without `<|bos|>`. The shipped adapter tokenizer bakes `<|bos|>…<|endoftext|>` into add_special_tokens=True (base Qwen adds nothing), so the reference's bos check fires — table rows must be `[bos]+prompt+[tok]+[eos]`. Partial-table A/B (diag_bos.py): scifact 0.642→0.663 (paper 0.665), arguana 0.394→0.518 (0.512), scidocs 0.147→0.176 (0.181). Reproduction faithful after fix. No-bos tables archived in `artifacts/…/tables-nobos/`. Doc encoding was always correct (post-processor adds bos/eos there).
- 2026-08-24 machine OOM incident: full-vocab fp32 logits in the sparse doc encoder + two concurrent MPS jobs exhausted 24 GB RAM (macOS killed jobs, Dylan saw the out-of-memory dialog). Fixed: sparse projection restricted to per-dataset query-token columns, batch cap 32, MPS watermarks (HIGH 0.7 / LOW 0.5 — both must be set, high-only crashes), strictly sequential jobs. Docker Desktop holds ~17 GB for Dylan's own containers; not touched.

## Findings (log, M1)

- LightRetriever lookup table ≠ input embedding matrix: each vocab token is forwarded through the full trained model ([bos]+instruction+token+[eos], take EOS hidden state). Using the raw embedding matrix instead costs −11.2 BEIR (paper ablation A2). Tables are per-instruction.
- Paper headline numbers are hybrid (dense+sparse); dense-only (what the pure two-collection edge architecture gives) is 2–3 pts lower. But LightRetriever's sparse query side is also zero-compute (token counts) → hybrid on Qdrant Edge is plausible.
- Inference-free sparse SOTA: `opensearch-neural-sparse-encoding-doc-v3-gte` (133M doc encoder) BEIR-13 avg 0.546; SPLADE-v3-doc 0.517. BM42 disqualified (query-side attention). miniCOIL lacks comparable BEIR numbers and needs per-token multi-vectors.
- LEAF asymmetric verified: docs via teacher `Snowflake/snowflake-arctic-embed-m-v1.5` (109M), queries via `MongoDB/mdbr-leaf-ir` (23M) = 54.03 BEIR-en (beats symmetric-either). Packaged as `MongoDB/mdbr-leaf-ir-asym`.
- Best small symmetric (MTEB v1 BEIR-en): arctic-embed-s 51.98, bge-small-en-v1.5 51.68, granite-small-english-r2 50.9. Best static: potion-retrieval-32M 35.06 MTEB-Ret.
- Literature: no LightRetriever successor; ScalingNote (industry) 29M query tower keeps 99% R@50 vs 7B teacher; DeepMind LIMIT gives formal single-vector ceiling; "static model + linear map into frozen big-doc space" appears unpublished → candidate original experiment.
- ArguAna queries avg 193 words (counter-argument retrieval) — stress case for bag-of-tokens query encoders; expect LightRetriever to lose there.
