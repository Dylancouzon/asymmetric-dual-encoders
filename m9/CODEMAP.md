# M9 code map

Read `STATUS.md`, then `LEDGER.md`. This file exists so a future session can resume without
reading every module. Nothing here restates `LEDGER.md` (protocol) or `registry.json` (constants).
`m8/CODEMAP.md`'s 24 pitfalls still apply and are not repeated.

## Layout

- `m9src/` — all M9 code. It **imports** `m7src` and `m8src` and edits neither.
- `work/m9runs/` — checkpoints. `work/m9tokens/` — the session manifest and per-run tokens.
  `work/m9smoke/` — diagnostic artifacts, a namespace the decision loader cannot address.
- `results/m9_*.json` — every committed M9 artifact, each carrying a `_registration` block.

## Modules

**Foundation** — `m9base.py` paths, the two surfaces, and `paths_guard.install()` at import;
`M7_ENCODER` is **assigned and a conflicting value refused**, never `setdefault` ·
`guard9.py` the session manifest + one-use run tokens + `eligible()` that recomputes rather than
trusting a boolean · `data.py` the query pool (M8's extended-filter list, re-labelled by source,
minus fever-train) and the document pool (draw order, never sorted).

**Recipe** — `nano.py` the student (backbone → mean pool → `Linear(h,1024)` → L2), the closed-form
head warm start, the epoch order, the fixed-size and token-budgeted batchers, and the trainer ·
`teacher9.py` challenger teachers through `SentenceTransformer`, with `parity_vs_frozen()` proving
that path reproduces M7's frozen one · `screen.py` assembles an arm from the registry, enforces
stage order and the adequacy gate, and holds the single decision function.

**Measurement** — `eval9.py` DEV-6 / SCREEN-3 scoring against a teacher's document space ·
`screen_stats.py` the one statistic and the one decision rule · `lock_constants.py` materializes
every number the lock states · `fp16_gate.py`, `port.py`, `bridge_dryrun.py`, `head_probe.py` the
M9.1 gates and diagnostics.

## Pitfalls this milestone earned

1. **A random `Linear(384,1024)` head is not a neutral starting point at a small dose.** A frozen
   bge-small backbone with a closed-form ridge head scores **0.3463** on SCREEN-3 — 50.8% of the
   teacher ceiling — while the same backbone with a *random* head reaches 12.4% after 2,000 trained
   steps. At ~1% of LEAF's dose the initialization is a large fraction of the entire budget, and a
   screen run from random init would partly rank arms by how fast each recovers from its own head.
   Twelve seconds of `np.linalg.solve` answered a question that would otherwise have cost a day.
   **Screen in closed form first** (`m8/FINDINGS.md` §4) applies to initializations too.
2. **A `--strict=False` escape hatch outlives the guard it belonged to.** `port.py` kept calling
   `write_result(..., strict=...)` after `guard9` had dropped the parameter, so the mandatory port
   pilot crashed deterministically at its final line — *after* doing all the work. A guard's API
   change must be grepped across every caller in the same commit.
3. **Sorting a sampled index list destroys the sample.** `doc_pool_rows` drew 400,000 rows
   uniformly and then sorted them, so any prefix was a low-global-row prefix — `esci-prod` first —
   and the mix arm's documents would have been store-biased rather than uniform. The sort looked
   like tidiness. If a downstream consumer takes a **prefix**, the order *is* the sampling design.
4. **A per-step budget accumulates its rounding.** The first token-budgeted batcher filled
   `budget × share` each step and overshot by up to one example every time, which over 30,349 steps
   drifts far enough to exhaust a stream before the final checkpoint — and an exhausted stream
   produced a *shorter* arm whose `history[-1]` then silently became checkpoint 3. Track the budget
   **cumulatively** (fill until the running total reaches what is due by the end of this step) and
   assert the step count, the non-emptiness of every batch and the absence of exhaustion.
5. **`hist[-2:]` is not "the last two checkpoints".** It is "the last two rows that exist", which
   on a truncated arm is a different pair. Read checkpoints by their registered **step id** and
   refuse when one is missing.
6. **A cosine schedule with denominator `steps - warmup` never reaches `lr_final`.** The last
   executed zero-based step has `t < 1`. Use `steps - warmup - 1`.
7. **Two dev components can share a corpus, and that is worth exploiting — with assertions.**
   `heldout-longq`'s 55 queries are a subset of `heldout-train`'s 7,325 over the identical 6.17M-row
   pool, so scoring it separately reads 12.6 GB twice per checkpoint for nothing. Subsetting the
   already-computed result is *exactly* equal only if corpus, vectors, qrels and self-hit policy
   match — so assert all four rather than assuming them.
8. **A DEV-6 bootstrap draw is 3.2 GB if you materialize it in one block.** 20,000 replicates ×
   20,152 queries × int64. Draw in blocks of 1,000 and accumulate; use int32.
9. **An external reviewer will read your code against your document and find that they disagree.**
   Three passes here: the first attacked the design, the second found that the first round of fixes
   had moved failures *out of the prose and into the implementation*, and the third attacked the
   two things the second had never seen. Reviewing the document alone would have shipped a lock
   whose statistic, batcher and guard did something else.
10. **A guard keyed on HEAD makes "commit frequently" and "run a multi-arm screen" mutually
    exclusive.** The session manifest originally bound to the lock commit *and* the fingerprint, so
    committing an arm's own RESULT moved HEAD and would have voided every arm already run. It only
    showed up in use — the self-test passed, and so did the first arm. What must not move mid-screen
    is the lock, the code and the input data; that is the fingerprint. HEAD's only job is "is the
    lock pushed", which `check_state()` still enforces. Ask of any freshness check: *what is the
    smallest thing whose change should invalidate this?* — and key on exactly that.
11. **Never rebind a function's parameter name to a loop-local.** A cache-hashing patch set
    `key = p.name` inside `encode_cached(key, ...)`, where `key` was the *teacher id*. The next call
    asked `TEACHERS['chunk_00000.npy']`. It cost a two-hour arm's launch — and nothing but running
    the path would have found it, because every unit-level check passed.
12. **Smoke a path the arm has never executed, even when the arm around it is already proven.**
    Three defects sat in the challenger-teacher path that the incumbent arms could never reach: the
    shadowing above, `stella-1.5B`'s vendored `modeling_qwen.py` calling
    `DynamicCache.get_usable_length` (removed in transformers 4.57 — an embedding model needs no KV
    cache, so `use_cache=False` avoids the legacy path without patching a third-party file), and a
    measured 8.53 GB peak at document batch 32 on a 10 GB card with the student resident. Four
    minutes of smoking found all three; the alternative was finding them 91 minutes into an encode.
13. **The most valuable thing a review produced was a smaller experiment.** Pass 2's verdict was
    "DO NOT SPEND THE 6 GPU-HOURS — run one corrected, fully guarded anchor curve instead". That is
    now stage A, and the adequacy gate decides whether stage B is worth running at all.
