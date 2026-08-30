# M9 runs, in order

Every row points at a committed artifact carrying its own `_registration` block. Numbers live in
the JSON; this file is the index and the one-line reading.

## M9.1 stage A — gates

| run | artifact | outcome |
|---|---|---|
| head probe (diagnostic, `-diag`) | `results/m9_head_probe.json` | a **frozen** bge-small backbone with a closed-form ridge head scores **0.3463** on SCREEN-3 = **50.8%** of the 0.6822 ceiling; λ selected on the training residual. Changed the recipe: every arm now warm-starts the head (`m9/LEDGER.md` §3.2a) |
| ST-vs-frozen teacher parity | in `m9src/teacher9.parity_vs_frozen` | min-cos **0.99959**, max-abs 1.45e-4 on 64 real texts — the `SentenceTransformer` path used for challenger teachers reproduces M7's frozen path, so challenger numbers are admissible |
| fp16 target gate | `results/m9_fp16_gate.json` | **PASS** — min-cos 0.999959 (≥0.9999), max-abs 2.02e-4 (≤1e-3) on the locked 10,000-text decile-stratified sample. Arms train on the fp16 target cache |
| bridge-tolerance dry run | `results/m9_bridge_dryrun.json` | **PASS, exactly** — 1,915 queries, zero missing/extra/reordered qids, max per-query \|Δ nDCG@10\| **0.0** across a fresh process. The scorer is bit-reproducible, which is the drift class the real six-set bridge exists to catch |
| ONNX / fastembed port pilot | `results/m9_port_pilot.json` | parity **PASS** for both students — min-cos 0.9999993 / 0.9999992, max-abs 1.6e-7, opset 17, **zero custom-domain ops**. Size and fastembed registration re-run after the anchor (see below) |

## M9.1 stage A — the anchor curve

| run | what it is |
|---|---|
| `m9s1` | the anchor: stella-400M teacher, bge-small student, prompt (b), query-only, seed 0, warm-started head, 16 epochs / 30,349 steps / 59,507,872 non-pad tokens |
| `m9s1c` | the same arm **without** the warm start — a registered diagnostic that prices it |

**Aborted attempt 1 (quarantined, `logs/m9_arm_m9s1_aborted.log`).** Reached checkpoint 1
(7,588 steps, 4 epochs, 14,878,695 non-pad tokens): SCREEN-3 **0.448139** = **65.7%** of the 0.6822
ceiling, up from 50.8% at the warm-started start. Killed at ~11,000 steps for two independent
reasons: Codex pass 3 found its warm-start λ had been selected on SCREEN-3 rather than on training
data, which makes anything it produced diagnostic; and it had degraded from 1,990 to 786 ex/s
because each checkpoint evaluation left ~10 GB in the PyTorch allocator. Both fixed; re-run.

*(rows filled in as they land; the adequacy gate reads `m9s1`'s DEV-6 curve)*

## Reference rows measured this milestone

| row | DEV-6 | SCREEN-3 (family weights) |
|---|---|---|
| stella-400M symmetric (the ceiling, and the retention denominator) | **0.67238** | **0.68223** |

## Deliberately not run

- **The batch-32-versus-128 pilot.** Registered, then removed before any arm ran: two matched
  epochs give batch 32 four times the optimizer updates and compress a separate warmup+cosine
  schedule into a miniature, so it would have measured early optimization speed rather than the
  batch size that wins at final dose.
- **The head+tail long-query probe.** Will not run in M9; first-512 truncation is stated as a
  limitation and `heldout-longq` may not change any decision.
- **Stage B** (`m9s1b`, `m9s2`, `m9s3`, `m9s4`, `m9s5`, `m9s6`) — gated on the adequacy gate.
