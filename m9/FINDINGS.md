# M9 findings — the durable record

**M9 closed 2026-09-01 as a MEASUREMENT (Dylan).** One candidate was built and frozen
(`m9/FREEZE.json`, sha `9d631b2c…`); it misses both bars on every projection, so it is not released.
No six-set, reserved, or LoTTE access was spent. The registered six-set transaction
(`m9/FINAL_LOCK.md`) remains to be executed on the frozen candidate as the close-out measurement,
which gives M10 the dev→six calibration M9 never had. Numbers live in `results/m9_*.json` and
`m9/RESULTS.md`; this file is the reading of them.

## 1. The question M9 asked, and the answer

**Can a ≤35M transformer, distilled by plain L2 regression into stella-400M's query space, retain
enough of the teacher (≥87.8% avg-6 for the release bar, ≥89.7% for the aim) to beat bge-small
and the LEAF asymmetric system while serving zero's frozen index?**

**Not with M9's data.** The build reached SCREEN-3 **0.5606 = 82.2%** of the 0.6822 ceiling at
3.74B tokens and had been flat for 1.1B tokens at constant LR; the anneal added +0.004. The macro
hides the finding:

| SCREEN-3 component | ceiling | candidate | retention |
|---|---|---|---|
| nq-250k (Wikipedia QA; NQ-adjacent data is in the training mix) | 0.8839 | 0.8289 | **93.8%** |
| cqadup-physics (forum questions) | 0.4931 | 0.3501 | **71.0%** |
| cqadup-programmers (forum questions) | 0.4681 | 0.2345 | **50.1%** |

Where the training queries look like the test queries, nano is already inside LEAF's band (LEAF
retains 97.9% of its teacher on our six; 97.7% on BEIR-14 in its paper). Where they do not, it retains half. **This is a coverage failure, not a
capacity failure**: a 33M student that reaches 94% on one distribution has the capacity to reach
94%. The six are scientific claims, biomedical questions, finance questions, 170-word
counter-arguments, paper titles and COVID queries — none of which the pool contains — so the
avg-6 retention is expected below the SCREEN-3 macro, not above it.

## 2. What is closed, and what closed it

`m9/EXPLORED.md` is the register with reopening conditions. Headline reading:

| hypothesis | probe | result | reading |
|---|---|---|---|
| More dose on the same pool | screen anchor + build curve | anchor 73% at 60M tokens (separate 16-epoch run); build 68.5% at 0.12B → 79% at 0.74–0.86B → 82% at 3.7B, then flat for 1.1B | 30× the build's first-eval dose bought +14 points, then nothing; the pool is exhausted, not the student |
| Student choice | `m9s4` | MiniLM-L6 −0.0026 DEV-6 vs bge-small, unresolved | a tie at screen dose; MiniLM is 2× cheaper to serve and to train |
| Instruction template on the student | `m9s5` | −0.0204, resolved | the student gets raw query bytes; the teacher keeps its s2p template |
| Documents as regression text | `m9s6` + build | neutral at equal budget (−0.0060 DEV-6); as *extra* text they beat 16 epochs of query repetition (0.545 at 8 query-epochs vs the anchor's 0.500 at 16) | text volume and breadth help; documents are the cheap form of breadth, not a substitute for query forms |
| A bigger teacher | `m9s2` (voided artifact, log read) | stella-1.5B −0.0023 at every checkpoint | M7's "select on the artifact" lesson holds for towers too |
| Head initialization | `m9s1c` | random head −0.0272 at equal SGD dose | warm-start the head in closed form; 8 s of ridge is 5× the decision threshold |
| Seed | `m9s1b` | 0.0008–0.0023 | below threshold; with a closed-form head there is no random init left |
| fp16 target cache | gate | min-cos 0.99996 | fp16 targets are safe |
| ONNX export, document model included | pilots | zero custom ops, parity 0.9999994 (student) / 0.9999994 (stella) | the M11 port risk is retired |
| fastembed serving | pilot | exact (6e-08) once the linear head is exported per token and fastembed pools | fastembed has no slot for a post-pooling head; linearity of mean pooling is what makes the trick exact — a **nonlinear** head cannot ship this way |
| Edge deployability | Mac rounds 1–4 | 1M×1024 index serves in a 256 MB container only binary-quantized (3.4 ms zero / 4.5 ms nano); fp16 is 100× slower at every limit | quantization is a precondition, not an optimisation |

## 3. Mechanisms worth carrying into M10

1. **Retention is a per-distribution quantity.** Report it per component, never only as a macro,
   and put the out-of-distribution components in the headline. M9 would have diagnosed itself at
   eval 8 had the curve watch been per component.
2. **The query pool is the lever the recipe never touched.** 463K real queries, all Wikipedia QA
   or product search, longest 108 words. LEAF's ~1.8M query-like texts spanned web search, product
   QA, biomedical QA and forums. The teacher supplies a target for any text, so query-form breadth
   costs generation, not labelling.
3. **A plateau rule that compares two single evaluations fires on noise.** Per-eval noise ~0.005,
   threshold +0.001 over 1.1B tokens: it fired while the best-so-far was still rising. Use
   best-to-best or a smoothed curve, and read annealed checkpoints — the constant-LR curve cannot
   see the anneal (+0.004 here).
4. **Phase 2 must be specified at lock or it does not exist.** The mandate required one fully
   specified ranking-aware loss and its trigger; the lock recorded "out of scope" and the flat
   curve then had no registered response. Specifying it after seeing the curve was correctly
   refused.
5. **Closed-form probes before chains** (M8 §4.1) paid again: the head probe (12 s) changed every
   arm's initialization; the mix decomposition read the m9s6 verdict without another run.
6. **Protocol-scope guards and protocol logging collide during a build.** Editing the ledger
   silently disabled unattended crash-restart. The build-period log (`m9/BUILD_LOG.md`) is the
   pattern: notes in an unscoped file, merged after the run.
7. **A frontier ratio has no meaning without its index configuration** (1.11×–3.28× across mmap /
   RAM / int8 / binary / memory-limited), and **the zero-compute model is the larger artifact**
   (270 MB table vs 46 MB tower). The whitepaper's frontier must name the index it was measured on.

## 4. What M9 leaves M10

- **Frozen candidate** `work/m9long/ckpt/last.pt` (sha `9d631b2c…`, 33.4M, bge-small × stella
  space, 3.74B tokens) — a warm-start init already at 94% on NQ, and the subject of the pending
  six-set close-out.
- **Infrastructure that works**: `m9src/longrun.py` (resumable trainer, kill envelope, cooldown),
  `watchdog.py` + `guardian.sh` + `sentinel.sh` (four-layer supervision), `final9.py` (access state
  machine; scoring path still unwritten), `final_stats.py` (16/16 tests), the fp16 target cache,
  the ONNX/fastembed export path including the document model.
- **Protocol state**: reserved four unspent; LoTTE-clean unread (M9's registered read #1 is
  withdrawn unexecuted so M10 inherits a fresh surface — decision 2026-09-01, see `m9/STATUS.md`);
  `results/perquery.json` intact (sha `6b18e3dd…`); dev reuse 494 at M8 close plus M9's screen
  checkpoints and 32 build evaluations (`m8src/dev_reuse_m8.py` gives the exact count).
- **Diagnostics for M10's design**: the rank-bottleneck probe ran on the Mac 2026-09-01
  (`m10/PLANNING.md` §9): the reconstruction-optimal 384-d output subspace serves one query
  distribution at 99.5% and three at 90–93%, strong evidence that M9's 384-wide linear head bound
  it under L2 regression — the second cause of the miss beside coverage. Still to run on the box: the per-component DEV-6 read incl. `heldout-longq`; the capacity
  probe is optional and report-only now that the 35M cap is hard (Dylan, 2026-09-01).
- **A recommendation**: build M10's recipe around coverage first (synthetic queries in every form
  the six use and beyond, FineWeb breadth), then LEAF's optimizer regime (small batch, cyclic
  anneals), with a ranking-aware phase-2 loss registered at lock rather than symptom-gated.
