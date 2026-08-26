# M7 status

**Stage:** data pipeline complete and reviewed → Stage 0 (representation compatibility) next
**Updated:** 2026-08-26

## Machine

RTX 3080, **10 GB VRAM** (not 12) · 25 GB RAM (peak budget 18 GB) · 16 cores · ext4 · nvcc 12.6
→ torch 2.8.0+cu126. Teacher **BAAI/bge-base-en-v1.5 @ a5beb1e3e68b9ab74eb54cfd186867f64f240e1a**.
Encode 891 texts/s fp16 (2.4× fp32, cosine-identical). Qdrant v1.19 server binary + `qdrant-edge-py`
0.8.0 both available, so the ANN sweep and the real Edge shard demo can both run.

## Where things stand

**Green.** Harness re-validated on this box (bge-small ArguAna 0.6038, SciFact 0.7127, bm25 FiQA
0.2532 — all inside 0.003). All six datasets hash-match the frozen manifest. Conformance suite
30/30 including the real save→load→encode path. Dev + untouched-final assets pinned into the
manifest. Frozen doc-vector pool: 6,169,142 × 768 fp16 = 9.48 GB.

**Training mix, after all decontamination passes: 349,934 pairs** (hotpotqa · fever · squad ·
esci · mrtydi) plus 221,395 query-text-only rows for objective B. Full provenance, rights and
counts in `results/m7_field_table.md`.

## Dev reference rows (`work/devres/refs.json`)

| system | nq-250k | hotpotqa | cqa-prog | cqa-phys | macro-4 |
|---|---|---|---|---|---|
| bge-base symmetric, prefixed — **teacher ceiling** | 0.8198 | 0.7258 | 0.4240 | 0.4727 | **0.6106** |
| bge-base symmetric, bare | 0.7982 | 0.7182 | 0.4020 | 0.4608 | 0.5948 |
| **bm25 — the gate bar (G3)** | 0.5804 | 0.5851 | 0.2975 | 0.3471 | **0.4525** |
| **potion-retrieval-32M — the Stage-0 bar (G1)** | 0.5479 | 0.4630 | 0.2261 | 0.2835 | **0.3801** |

The table needs ~74% teacher retention to clear BM25 on dev.

## Three findings that shape the report

1. **The untouched-final partition has no clean member.** Climate-FEVER was dropped (no
   affirmative licence at any primary source). BEIR FEVER shares its corpus with fever-train by
   construction. And DBpedia-entity — the intended clean probe — turns out to have **9.32%**
   document overlap with our training positives, because DBpedia abstracts are Wikipedia lead
   paragraphs and so are HotpotQA's documents. Both rows get their overlap rate attached; neither
   is presented as an uncontaminated generalisation number.
2. **Dev cannot validate long queries.** Held-out query length is p50 = 13 WordPiece tokens,
   p90 = 24; only 55 of 7,325 reach ≥64. ArguAna's are ~250. So the ArguAna row is an
   extrapolation, the learned-weight "long-query hypothesis" is untestable here, and no approved
   source fixes it (args.me is ArguAna's own source family).
3. **Document-side domain transfer is unmitigable, as pre-registered.** Every in-domain candidate
   for the six is on the contamination map. The one available lever is vocabulary-coverage
   distillation on pseudo-queries (objective B needs no labels), built and labelled as a
   vocabulary mitigation only.

## Running

`run_stage0c.sh` step 4/5 — the two R3 sweeps the first decontamination run missed (nq-250k dev,
FEVER untouched). Then dev reference rows for the held-out components.

## Next

`run_stage0b.sh`: ridge probe → capacity probe → objective grid A/B/C → go/no-go gate.

## Open items for Dylan

Nothing blocking. The HF release go remains yours. WSL holds ~25 GB of the host's 32 GB, which is
why one concurrent-job OOM took the whole distro down; jobs are strictly sequential now.

## Reviews run

Fable adversarial review of the protocol code, **before any candidate number existed** —
3 BLOCKER / 6 MAJOR / 10 MINOR, all blockers and majors actioned. Detail in `LEDGER.md`. The
blockers were: tier decisions pairing against a re-run BM25 instead of the frozen vectors; the
freeze pinning code but not the table bytes, preprocessing or fusion; and a silent
undecontaminated path in objective B (fixing it removed 368 overlapping queries).
