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
| ONNX / fastembed port pilot | `results/m9_port_pilot.json` | **`pass_all` FALSE — on one conjunct, and it is an informative miss.** fp32 export parity **PASS** (min-cos 0.9999994, max-abs 1.6e-7, opset 17, **zero custom-domain ops**). Shipped **fp16** artifact **68.501 MB** (bge-small) and **47.048 MB** (MiniLM) — both **inside the 70 MB target**. **fastembed registration PASS** — `add_custom_model` accepts the description and the model is listed. **fp16 parity FAILS**: min-cos 0.99953 against the locked 0.9999 |

## M9.1 stage A — the anchor curve

| run | what it is |
|---|---|
| `m9s1` | the anchor: stella-400M teacher, bge-small student, prompt (b), query-only, seed 0, warm-started head, 16 epochs / 30,349 steps / 59,507,872 non-pad tokens |
| `m9s1c` | the same arm **without** the warm start — a registered diagnostic that prices it |

### `m9s1c` — what the closed-form head is worth

| | SCREEN-3 | DEV-6 |
|---|---|---|
| `m9s1` warm-started | 0.49958 | 0.48041 |
| `m9s1c` random head | 0.47304 | 0.45632 |
| **delta** | **+0.0265** | **+0.0241** |

The estimand is the **fixed-SGD-dose warm-start delta**: same seed, same 30,349 steps, same
59.5M non-pad tokens, differing only in whether the head started from a ridge solution or from
PyTorch's default init. It is not compute-matched — the warm start adds a Stage-0 phase of 60,000
forwards and 918,015 tokens for 8.4 s — and one seed cannot separate initialization value from its
interaction with data order. At **+0.0265** it is nonetheless about 4.7x the 0.0056 decision
threshold, and it costs 0.5% of the arm's wall-clock. Twelve seconds of `np.linalg.solve`, found by
a closed-form probe before any chain was spent.

**Aborted attempt 1 (quarantined, `logs/m9_arm_m9s1_aborted.log`).** Reached checkpoint 1
(7,588 steps, 4 epochs, 14,878,695 non-pad tokens): SCREEN-3 **0.448139** = **65.7%** of the 0.6822
ceiling, up from 50.8% at the warm-started start. Killed at ~11,000 steps for two independent
reasons: Codex pass 3 found its warm-start λ had been selected on SCREEN-3 rather than on training
data, which makes anything it produced diagnostic; and it had degraded from 1,990 to 786 ex/s
because each checkpoint evaluation left ~10 GB in the PyTorch allocator. Both fixed; re-run.

### `m9s1` — the anchor curve (the headline of M9.1)

| step | examples | non-pad tokens | SCREEN-3 | retention | DEV-6 | retention |
|---|---|---|---|---|---|---|
| 0 (warm-started head, frozen backbone) | 0 | 0 | 0.3463 | 0.508 | — | — |
| 7,588 | 971,264 | 14,878,695 | 0.4481 | 0.657 | — | — |
| 15,175 | 1,942,400 | 29,755,611 | 0.4812 | 0.705 | — | — |
| 22,762 | 2,913,536 | 44,632,656 | 0.4944 | 0.725 | — | — |
| 30,349 | 3,884,576 | 59,507,872 | 0.4998 | 0.733 | 0.4806 | 0.715 |

**Final: SCREEN-3 0.4998 (73.3% of the 0.68223 ceiling), DEV-6 0.4806 (71.5% of 0.67238).**
Stage-0 dose (the warm start, reported apart from the SGD dose): 60,000 examples, 918,015 non-pad
tokens, 8.5 s. Training: 2015 s at ~2,010 ex/s.

The curve decelerates hard — quarter-on-quarter gains of **+0.0330, +0.0132, +0.0054** — so the
16-epoch dose is close to what this data volume yields, and more SGD on the same 242,786 queries is
not where the remaining 27% of the ceiling lives.

Adequacy gate: **PASS** (retention 0.7326 ≥ 0.60; late slope 0.0054 ≤ 0.02) → stage B authorised.
It is a budget trigger, not a certification that a stage-B contrast would rank the same way at
final dose.

### The fp16 parity miss — read it correctly

The 1 − 1e-4 threshold was registered for **torch-versus-ONNX export fidelity at the same
precision**, and it passes there by four orders of magnitude. It is then being applied to a
**precision change**, where 0.99953 is ordinary fp16 rounding on a 33M model, not an export defect.
The threshold is not moved after seeing the number — that is the one thing the protocol forbids —
so the pilot stands as a fail and the decision goes to M10 with three facts:

1. fp32 passes every fidelity conjunct and **misses the size target** (135.6 MB vs 70 MB).
2. fp16 **meets** the size target and fastembed registration, and misses a cosine threshold written
   for a different comparison.
3. The measurement that should decide this is not a cosine at all: it is **whether the fp16 graph
   changes retrieval**. Register a macro-shift threshold on SCREEN-3 *before* measuring it, the way
   M7 priced its int8 table as quality-free (0.4117 vs 0.4114). That is an M10 task; M9 does not
   get to invent a threshold tonight for a number it has already seen.

### `m9s1b` — seed sensitivity (reported, read by no rule)

Identical recipe and dose at seed 1 instead of seed 0. With the head warm-started in closed form
the model has **no random initialization at all** — a pretrained backbone and a deterministic head
— so the seed moves only data order and dropout.

| | SCREEN-3 | DEV-6 |
|---|---|---|
| seed 0 (`m9s1`) | 0.49958 | 0.48041 |
| seed 1 (`m9s1b`) | 0.50081 | 0.48271 |
| \|Δ\| | **0.00123** | **0.00230** |

Both are well under the 0.0056 decision threshold, which is reassuring but proves little: a range
over K = 2 is one half-normal draw, not an estimate of σ (`m8/CODEMAP.md` pitfall 18). It is why
this number is **reported beside** every contrast and never **read by** one.

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
