# M10 status — PLANNED, budget VALIDATED 2026-09-04, plan re-cut the same day; nothing has trained

Mandate `instructions-m10.md` (read **§Amendment 2026-09-04** first — it is authoritative over any
older sentence there) · evidence `m10/PLANNING.md` (§11 measured rates, §12 the synthetic-data
question) · runs `m10/RESULTS.md` · closed avenues `m10/EXPLORED.md` · selection-surface drafts
`m10/COV_CANDIDATES.md` · code `m10/CODEMAP.md` · M9's record `m9/FINDINGS.md`.
Owner report: https://claude.ai/code/artifact/fce61c94-5444-4c78-bb2e-46112cb7547a

**Where things stand.** The 2026-09-01 plan went through seven Codex passes and Dylan's compute
ruling. On 2026-09-04 he validated the budget and asked for a full efficiency review before any
money was spent ("is this the best way … is it the most efficient?", "I'm not sure why we need to
generate synthetic data?", "don't over-engineer", "keeping the same teacher (Stella) is the goal").
The review produced amendments **A1–A8**. What changed, shortest form:

- **Measured, not assumed:** the M10 recipe runs at **745 examples/s on the box** against the
  plan's imported 560 on a rented A100 — the step is launch-bound at batch 32
  (`results/m10_rate_bench_box.json`). The box is an execution target again for everything except
  generation; the build is ≈75 GPU-hours, not 100; cloud spend re-prices to **≈$110–280** hybrid.
- **Less machinery:** family D cut from three ranking-aware arms to one (LEAF's ‖e‖₂ loss), which
  deletes the candidate bank, the mining pass, the HNSW fallback, the τ rule and the seed-rank
  field. Thirteen arms, eleven contrasts. Confirmations capped at two decisions. The full-dose
  seed-1 replica is withdrawn.
- **Mostly real text, not generated:** ≈1.5M harvested titles / headings / claim sentences from the
  licensed pool as new arm A3, generation cut to ≈1.0M for the six forms no corpus contains. Three
  of the four clean-4 headline datasets fall in the harvested half.
- **Two protocol fixes:** C1/C2 are registered on **clean-4 as well as avg-6** (four Holm conjuncts;
  clean-4 bars 0.5046 / 0.5233), because M14 made clean-4 the headline for both halves of the pair;
  and the COV resolution number now **sizes** the screen instead of being measured after the lock,
  because the registered MDE 0.0056 sat below the surface's own resolution.

## Dylan — open decisions (defaults apply meanwhile)

| # | decision | default |
|---|---|---|
| 1 | Ratify M9's final lock plus the six-only amendment (one sentence: "run M9's six-set scoring as registered, six only, no reserved batch") | blocks only the M9 close-out |
| 4 | PAQ as query text (CC BY-SA data, official release) | include |
| 7 | Confirm LoTTE read #1 withdrawal and the renumbering | as recorded |
| 10 | The 2026-09-04 amendments A1–A8 | adopted; strike any item and it reverts to the 2026-09-01 text |
| — | **Generation smoke approval** — you are the approver (200 queries × 12 forms, 90% contract / 80% on-form). It cannot run until a cloud instance exists (Qwen3-8B bf16 is 16.4 GB on a 10 GB card), so it will be waiting when you are back | blocks generation only |

Decisions 2 (budget), 3, 5, 6, 8 and 9 are closed — see `instructions-m10.md` §Owner decisions.

## The three-day box window (2026-09-04 → 09-07, Dylan away)

Everything here needs no approval, spends no cloud dollars, and touches no protected surface.
Generation is deliberately absent: it needs both Dylan and a bigger card.

1. **Rate work.** Length-bucketed single-chunk batching in the trainer, then re-measure on real
   tokenized corpora (§11's numbers are random-token, no data loading). This is the number the
   build's cost line reads.
2. **COV admission (M10.0-d)**, re-run under the 2026-09-04 licence rule: ConsumerContractsQA
   (CC BY-NC) is re-admissible, giving four families without LEDGER; verify LEDGER's structure and
   chunk cap; per-component licence, revision, size, qrels and metric records into `m10/LEDGER.md`
   §2; corpus-level and fingerprint contamination screens; add every admitted corpus, query set and
   document set to the protected index; encode with stella.
3. **The COV resolution number** (e5-small-v2 vs gte-small, distance only, no direction) — under
   amendment A4 this now sizes the screen, so it must be pushed before the lock.
4. **M10.0-c**: per-component DEV-6 read of the M9 candidate incl. `heldout-longq` (the baseline
   row). The checkpoint and caches are on the box.
5. **The §Harvest pipeline** — titles, headings, declarative lead sentences, extracted
   interrogatives out of the licensed pool, with per-rule yields. No model in the loop.
6. **PAQ** download from Facebook's official release and the samplers (1.0M build, A2 control).
7. **Trainer port** to the M10 recipe: cyclic schedule, example-mix batcher, three- and four-layer
   pooled heads, the ‖e‖₂ loss arm, `test_resume.py` equivalence, an examples/s counter.
8. **Parity checks** (CPU, minutes) for MiniLM-L6's three-layer head and both students' four-layer
   heads, so families F and G may run those arms.
9. **Prompt prototyping** for the six generated forms (4-bit is fine — prototyping enters no
   record, produces no smoke result and no manifest row).
10. If time remains: the **recipe pre-screen** (B, C, E, G on the M9 pool, **DEV-6 only — COV stays
    unread**), labelled a pre-screen that sets defaults, never the screen.

## Then, in order

M10.0-e screen lock (thirteen arms, eleven contrasts, MDE from the resolution number) → cloud
instance for the generation smoke (Dylan reads it) → generation at ≈1.0M → M10.1 manifest with the
A8 quality gates → M10.2 arms on the box, confirmations, the synthesized selected-recipe arm, the
lock, Codex and Fable review, M9's six-only close-out from `m9-work`, LoTTE read #1 → M10.3 build
(200M examples, ≈75 GPU-hours, whole extension cycles) → export, parity, freeze, LoTTE read #2 →
M10.4 final: the six-set transaction with four C-conjuncts, then the reserved conditional.

## Guardrails that bite here

No six/reserved/LoTTE access outside the registered transactions. `results/perquery.json` is never
rewritten. Never edit a `guard9` protocol-scope file before M9's close-out runs. Every review brief
carries the reserved read-exclusion; audit the log after. Long runs: smoke, arm the
failure-signature monitor, check the rate, watch the machine. Stella on the Mac runs only in
`.venv-mac`. A stopped cloud instance costs disk only; an idle running one costs the budget.
